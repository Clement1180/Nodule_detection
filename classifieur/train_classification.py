"""
Pipeline de classification binaire de nodules pulmonaires sur radiographies thoraciques.

Ce module implémente une approche en trois volets :
  - Prétraitement SEPE (Sparse Edge-Preserving Enhancement) pour rehausser
    les contours et textures subtiles dans les images CT,
  - Mécanisme d'attention multi-échelle (MEAM) pour fusionner les
    représentations extraites à différentes résolutions,
  - Classification binaire (Nodule / No Finding) à l'aide d'un backbone
    EfficientNetV2 augmenté par le module MEAM.

Le jeu de données est construit en 3 canaux complémentaires :
  canal 1  densité brute normalisée,
  canal 2  SEPE (calculé sur GPU pendant l'entraînement),
  canal 3  CLAHE pour le contraste local.

Auteur  : Clément B.
Date    : 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, List
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.ndimage import gaussian_filter
import torchvision.models as models
import albumentations as A
from torchmetrics.classification import BinaryFBetaScore
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryAveragePrecision

torch.manual_seed(42)


class SEPEPreprocessor:
    """
    Rehaussement adaptatif des contours par filtrage laplacien régularisé.

    L'idée est de combiner un masque de poids basé sur le gradient local
    (qui préserve les bords) avec un terme de régularisation qui empêche
    l'amplification du bruit haute fréquence.

    Paramètres
    ----------
    sigma      : écart-type du noyau gaussien de lissage.
    alpha      : facteur d'amplification des détails.
    sigma_e    : sensibilité du masque de poids aux gradients.
    lambda_reg : coefficient de régularisation laplacienne.
    """

    def __init__(self, sigma: float = 1.0, alpha: float = 1.5,
                 sigma_e: float = 0.1, lambda_reg: float = 0.1):
        self.sigma = sigma
        self.alpha = alpha
        self.sigma_e = sigma_e
        self.lambda_reg = lambda_reg

    def __call__(self, image: np.ndarray) -> np.ndarray:
        I = image.astype(np.float32)

        # Lissage gaussien
        G_sigma = gaussian_filter(I, sigma=self.sigma)

        # Magnitude du gradient
        grad_x = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        W = np.exp(-(grad_magnitude ** 2) / (self.sigma_e ** 2))

        # Rehaussement : ajout des hautes fréquences pondérées
        I_enh = I + self.alpha * W * (I - G_sigma)

        # Régularisation par le laplacien pour limiter le bruit
        laplacian = cv2.Laplacian(I, cv2.CV_32F)
        I_SEPE = I_enh - self.lambda_reg * np.abs(laplacian)

        return np.clip(I_SEPE, I.min(), I.max())

    def batch_process(self, images: torch.Tensor) -> torch.Tensor:
        """
        Version batch GPU du rehaussement SEPE.

        Reproduit les mêmes opérations que __call__ mais en exploitant
        les convolutions PyTorch pour traiter un batch complet en parallèle.
        """
        batch_size, c, h, w = images.shape

        kernel_size = int(6 * self.sigma) | 1
        x = torch.arange(kernel_size, dtype=torch.float32, device=images.device) - kernel_size // 2
        gauss = torch.exp(-x**2 / (2 * self.sigma**2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel_2d = kernel_2d.expand(c, 1, -1, -1)

        padding = kernel_size // 2
        G_sigma = F.conv2d(images, kernel_2d, padding=padding, groups=c)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        sobel_x = sobel_x.expand(c, 1, -1, -1)
        sobel_y = sobel_y.expand(c, 1, -1, -1)

        grad_x = F.conv2d(images, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(images, sobel_y, padding=1, groups=c)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        W = torch.exp(-(grad_mag ** 2) / (self.sigma_e ** 2))
        I_enh = images + self.alpha * W * (images - G_sigma)

        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                        dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.expand(c, 1, -1, -1)
        laplacian = F.conv2d(images, laplacian_kernel, padding=1, groups=c)

        I_SEPE = I_enh - self.lambda_reg * torch.abs(laplacian)
        return I_SEPE


class MEAM(nn.Module):
    """
    Fusionne les cartes de caractéristiques extraites à différentes résolutions
    via un mécanisme d'attention croisée inter-échelles.

    Chaque échelle produit Q, K, V par projection 1x1.  L'attention est
    calculée entre toutes les paires d'échelles, puis les résultats sont
    concaténés et projetés vers la dimension de sortie.

    Paramètres
    ----------
    channels : nombre de canaux à chaque niveau d'extraction.
    d_k, d_v : dimensions des projections query/key et value.
    """

    def __init__(self, channels: List[int], d_k: int = 64, d_v: int = 64):
        super().__init__()
        self.scales = len(channels)
        self.d_k = d_k

        self.q_proj = nn.ModuleList([nn.Conv2d(c, d_k, 1) for c in channels])
        self.k_proj = nn.ModuleList([nn.Conv2d(c, d_k, 1) for c in channels])
        self.v_proj = nn.ModuleList([nn.Conv2d(c, d_v, 1) for c in channels])

        self.output_proj = nn.Conv2d(d_v * self.scales, channels[-1], 1)
        self.norm = nn.BatchNorm2d(channels[-1])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[-1].shape[2:]
        resized_features = []
        for f in features:
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(f)

        Q = [proj(f) for proj, f in zip(self.q_proj, resized_features)]
        K = [proj(f) for proj, f in zip(self.k_proj, resized_features)]
        V = [proj(f) for proj, f in zip(self.v_proj, resized_features)]

        # attention croisée
        attended_features = []
        for i in range(self.scales):
            attn_scores = []
            for j in range(self.scales):
                qi = Q[i].flatten(2)              # (B, d_k, HW)
                kj = K[j].flatten(2)
                score = torch.bmm(qi.transpose(1, 2), kj) / np.sqrt(self.d_k)
                attn_scores.append(score)

            attn_weights = [torch.softmax(s, dim=-1) for s in attn_scores]

            attended = torch.zeros_like(V[i].flatten(2))
            for j, w in enumerate(attn_weights):
                vj = V[j].flatten(2)
                attended += torch.bmm(vj, w.transpose(1, 2))

            attended = attended.view_as(V[i])
            attended_features.append(attended)

        # Concaténation et projection finale
        fused = torch.cat(attended_features, dim=1)
        output = self.output_proj(fused)
        output = self.norm(output)
        return F.relu(output)

class EfficientBackbone(nn.Module):
    """
    Encapsule un EfficientNetV2 pré-entraîné et expose les features
    intermédiaires nécessaires au MEAM.

    Les indices d'extraction et les dimensions de canaux dépendent du
    variant choisi (s / m / l).
    """

    EXTRACT_LAYERS = [2, 3, 4, 6]

    _CHANNEL_MAP = {
        's': [48, 64, 128, 256],
        'm': [48, 80, 160, 304],
        'l': [64, 96, 192, 384],
    }

    def __init__(self, variant: str = 's', pretrained: bool = True):
        super().__init__()
        self.extract_layers = self.EXTRACT_LAYERS
        self.channels = self._CHANNEL_MAP[variant]
        # Le modèle fournis utilise la version 's' pour des soucis de Vram
        if variant == 's':
            from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_v2_s(weights=weights).features
        elif variant == 'm':
            from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_v2_m(weights=weights).features
        else:
            from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
            weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_v2_l(weights=weights).features

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features = []
        for i, block in enumerate(self.backbone):
            x = block(x)
            if i in self.extract_layers:
                features.append(x)
        return features, features[-1]



class NoduleClassifier(nn.Module):
    """
    Backbone EfficientNetV2 + MEAM + tête de classification linéaire.

    Le backbone produit 4 cartes multi-échelles, le MEAM les fusionne
    en une seule représentation, et la tête linéaire prédit la classe.
    """

    def __init__(self, backbone_variant='s'):
        super().__init__()
        self.backbone = EfficientBackbone(variant=backbone_variant)
        self.meam = MEAM(channels=self.backbone.channels)

        final_channels = self.backbone.channels[-1]
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        features_list, _ = self.backbone(x)
        fused_features = self.meam(features_list)
        return self.classifier_head(fused_features)



class NoduleClassificationDataset(Dataset):
    """
    Charge les radiographies thoraciques et construit 2 canaux sur CPU
    (brut + CLAHE).  Le 3e canal (SEPE) est ajouté sur GPU pendant
    l'entraînement pour éviter un goulot d'étranglement CPU.

    Paramètres
    ----------
    image_dir  : répertoire contenant les images.
    csv_file   : fichier CSV avec colonnes ``file_name`` et ``label``.
    image_size : dimensions de sortie (H, W).
    augment    : active les augmentations géométriques et photométriques.
    """

    def __init__(self, image_dir: str, csv_file: str,
                 image_size: Tuple[int, int] = (512, 512),
                 augment: bool = False):
        self.image_size = image_size
        self.augment = augment
        self.samples: List[Tuple[str, int]] = []
        self.labels: List[int] = []

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        print("--- Chargement Dataset Classification (3 Canaux) ---")

        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()

        col_filename = 'file_name' if 'file_name' in df.columns else 'filename'
        if col_filename not in df.columns:
            raise KeyError(f"Colonne filename introuvable. Colonnes : {list(df.columns)}")

        img_path = Path(image_dir)
        nodule_count, nofinding_count = 0, 0

        for _, row in df.iterrows():
            fname = row[col_filename]
            label_str = row['label'].strip()
            full_path = img_path / fname

            if not full_path.exists():
                continue

            if label_str == 'Nodule':
                self.samples.append((str(full_path), 1))
                self.labels.append(1)
                nodule_count += 1
            elif label_str == 'No Finding':
                self.samples.append((str(full_path), 0))
                self.labels.append(0)
                nofinding_count += 1

        print(f"  {nodule_count} images 'Nodule'")
        print(f"  {nofinding_count} images 'No Finding'")
        print(f"  Total: {len(self.samples)} images")

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                                   rotate_limit=10, p=0.55),
                A.RandomBrightnessContrast(brightness_limit=0.1,
                                           contrast_limit=0.1, p=0.45),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05,
                                   alpha_affine=120 * 0.03, p=0.4),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            image = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        if self.augment:
            augmented = self.transform(image=image)
            image = augmented['image']

        img_float = image.astype(np.float32)
        img_min, img_max = img_float.min(), img_float.max()
        if img_max > img_min:
            img_float = (img_float - img_min) / (img_max - img_min)
        else:
            img_float = np.zeros_like(img_float)

        # Canal 1 : densité brute normalisée
        c1 = img_float

        # Canal 2 : CLAHE
        image_uint8 = (img_float * 255.0).astype(np.uint8)
        c3 = self.clahe.apply(image_uint8).astype(np.float32) / 255.0

        # On empile uniquement 2 canaux ici ; le SEPE sera calculé sur GPU
        img_stacked = np.stack([c1, c3], axis=0)
        return torch.from_numpy(img_stacked).float(), torch.tensor(label, dtype=torch.long)


def make_weighted_sampler(dataset: NoduleClassificationDataset) -> WeightedRandomSampler:
    """Retourne un sampler qui sur-échantillonne la classe minoritaire."""
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).

    Pondère la cross-entropie par (1 - p_t)^γ pour réduire la contribution
    des exemples faciles et focaliser le gradient sur les cas difficiles.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()



def _build_3ch_batch(images: torch.Tensor, sepe: SEPEPreprocessor) -> torch.Tensor:
    """
    Reconstruit le tenseur 3 canaux (brut, SEPE, CLAHE) sur GPU à partir
    du tenseur 2 canaux chargé par le DataLoader.
    """
    c1 = images[:, 0:1, :, :]          # brut
    c3 = images[:, 1:2, :, :]          # CLAHE
    c2 = sepe.batch_process(c1)         # SEPE

    # Normalisation du canal SEPE (division par le max du batch)
    c2_max = c2.view(c2.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    c2 = c2 / (c2_max + 1e-8)

    return torch.cat([c1, c2, c3], dim=1)


def train_classifier(model, train_loader, val_loader,
                     num_epochs=20, device='cuda'):
    """
    Entraîne le classificateur avec Focal Loss, AdamW et cosine annealing.

    Le meilleur modèle (selon l'Average Precision sur la validation) est
    sauvegardé automatiquement.
    """
    model = model.to(device)

    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    try:
        test_imgs, test_labels = next(iter(train_loader))
        print(f"  Batch test OK — shape: {test_imgs.shape}, labels: {test_labels}")
    except Exception as e:
        print(f"  ERREUR DataLoader : {e}")
        raise

    best_val_ap = 0.0
    train_metric = BinaryAveragePrecision().to(device)
    val_metric = BinaryAveragePrecision().to(device)
    sepe_gpu = SEPEPreprocessor()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images_3c = _build_3ch_batch(images, sepe_gpu)

            optimizer.zero_grad()
            outputs = model(images_3c)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)[:, 1]
            train_metric.update(probs, labels)

        train_ap = train_metric.compute()
        train_metric.reset()
        avg_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                images_3c = _build_3ch_batch(images, sepe_gpu)

                outputs = model(images_3c)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = F.softmax(outputs, dim=1)[:, 1]
                val_metric.update(probs, labels)

        val_ap = val_metric.compute()
        val_metric.reset()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | Train AP: {train_ap:.4f} | "
              f"Val AP: {val_ap:.4f}")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save(model.backbone.state_dict(), "dernier_train.pth")
            torch.save(model.state_dict(), "best_dernier_train.pth")
            print(f"  Sauvegarde (meilleur Val AP : {val_ap:.4f})")

    print(f"\nMeilleur Average Precision (validation) : {best_val_ap:.4f}")
    return model

def main():
    config = {
        'image_dir': 'nih_filtered_images',
        'csv': 'classification_labels.csv',
        'batch_size': 12,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print("Préparation des données")

    df = pd.read_csv(config['csv'])
    df.columns = df.columns.str.strip()

    # Split stratifié 80/20 pour conserver la distribution des classes
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )

    train_csv_path = 'train_split_temp.csv'
    val_csv_path = 'val_split_temp.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    train_ds = NoduleClassificationDataset(
        image_dir=config['image_dir'],
        csv_file=train_csv_path,
        augment=True,
    )
    val_ds = NoduleClassificationDataset(
        image_dir=config['image_dir'],
        csv_file=val_csv_path,
        augment=False,
    )

    # Sampler pondéré : on tire autant de nodules que de sains par époque
    train_labels = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels)
    sample_weights = (1.0 / class_counts)[train_labels]
    num_nodules_train = int(class_counts[1])

    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=num_nodules_train * 2,
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'],
        sampler=train_sampler, num_workers=8,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'],
        shuffle=False, num_workers=8,
    )

    model = NoduleClassifier(backbone_variant='s')

    print(f"\nDevice : {config['device']}")
    print(f"Batches par époque : {len(train_loader)}")

    try:
        train_classifier(model, train_loader, val_loader,
                         config['num_epochs'], config['device'])
    except Exception as e:
        print(f"\nERREUR : {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        Path(train_csv_path).unlink(missing_ok=True)
        Path(val_csv_path).unlink(missing_ok=True)

    print("\nClassification terminée.")
    print("  dernier_train.pth       backbone pré-entraîné (pour localisation)")
    print("  best_dernier_train.pth  meilleur modèle complet")


if __name__ == "__main__":
    main()
