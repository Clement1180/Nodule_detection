import os
import argparse
import math
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
from typing import List, Tuple
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def thorax_mask(shape, margin=0.97, blur=51):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    ry, rx = (h / 2) * margin, (w / 2) * margin
    mask = ((y - cy) ** 2) / (ry ** 2) + ((x - cx) ** 2) / (rx ** 2)
    mask = (mask <= 1).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask

def preprocess_image(img: np.ndarray, clip_percentiles=(1, 99)) -> np.ndarray:
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, clip_percentiles)
    img = np.clip(img, p_low, p_high)
    denom = p_high - p_low if (p_high - p_low) != 0 else 1e-6
    img = (img - p_low) / denom
    mask = thorax_mask(img.shape)
    background = np.percentile(img, 5)
    img = img * mask + background * (1 - mask)
    img = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

class SEPEPreprocessor:
    def __init__(self, sigma=1.0, alpha=1.5, sigma_e=0.1, lambda_reg=0.1):
        self.sigma = sigma
        self.alpha = alpha
        self.sigma_e = sigma_e
        self.lambda_reg = lambda_reg

    def __call__(self, image: np.ndarray) -> np.ndarray:
        I = image.astype(np.float32)
        G_sigma = gaussian_filter(I, sigma=self.sigma)
        grad_x = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        W = np.exp(-(grad_magnitude ** 2) / (self.sigma_e ** 2))
        I_enh = I + self.alpha * W * (I - G_sigma)
        laplacian = cv2.Laplacian(I, cv2.CV_32F)
        I_SEPE = I_enh - self.lambda_reg * np.abs(laplacian)
        return np.clip(I_SEPE, I.min(), I.max())

    def batch_process(self, images: torch.Tensor) -> torch.Tensor:
        c = images.shape[1]
        kernel_size = int(6 * self.sigma) | 1
        x = torch.arange(kernel_size, dtype=torch.float32, device=images.device) - kernel_size // 2
        gauss = torch.exp(-x ** 2 / (2 * self.sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = (gauss.unsqueeze(0) * gauss.unsqueeze(1)).expand(c, 1, -1, -1)
        padding = kernel_size // 2
        G_sigma = F.conv2d(images, kernel_2d, padding=padding, groups=c)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3).expand(c, 1, -1, -1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3).expand(c, 1, -1, -1)
        grad_x = F.conv2d(images, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(images, sobel_y, padding=1, groups=c)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        W = torch.exp(-(grad_mag ** 2) / (self.sigma_e ** 2))
        I_enh = images + self.alpha * W * (images - G_sigma)
        lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3).expand(c, 1, -1, -1)
        laplacian = F.conv2d(images, lap_k, padding=1, groups=c)
        return I_enh - self.lambda_reg * torch.abs(laplacian)

def _build_3ch_batch(images: torch.Tensor, sepe: SEPEPreprocessor) -> torch.Tensor:
    c1 = images[:, 0:1]
    c3 = images[:, 1:2]
    c2 = sepe.batch_process(c1)
    c2_max = c2.view(c2.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    c2 = c2 / (c2_max + 1e-8)
    return torch.cat([c1, c2, c3], dim=1)

class MEAM(nn.Module):
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
        resized = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) if f.shape[2:] != target_size else f for f in features]
        Q = [p(f) for p, f in zip(self.q_proj, resized)]
        K = [p(f) for p, f in zip(self.k_proj, resized)]
        V = [p(f) for p, f in zip(self.v_proj, resized)]
        attended = []
        for i in range(self.scales):
            scores = [torch.bmm(Q[i].flatten(2).transpose(1, 2), K[j].flatten(2)) / math.sqrt(self.d_k) for j in range(self.scales)]
            weights = [torch.softmax(s, dim=-1) for s in scores]
            agg = sum(torch.bmm(V[j].flatten(2), weights[j].transpose(1, 2)) for j in range(self.scales))
            attended.append(agg.view_as(V[i]))
        fused = torch.cat(attended, dim=1)
        return F.relu(self.norm(self.output_proj(fused)))

class EfficientBackbone(nn.Module):
    EXTRACT_LAYERS = [2, 3, 4, 6]
    _CHANNEL_MAP = {'s': [48, 64, 128, 256], 'm': [48, 80, 160, 304], 'l': [64, 96, 192, 384]}

    def __init__(self, variant='s', pretrained=True):
        super().__init__()
        self.extract_layers = self.EXTRACT_LAYERS
        self.channels = self._CHANNEL_MAP[variant]
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

    def forward(self, x):
        features = []
        for i, block in enumerate(self.backbone):
            x = block(x)
            if i in self.extract_layers:
                features.append(x)
        return features, features[-1]

def _letterbox(img_rgb: np.ndarray, target: int) -> Tuple[np.ndarray, float, int, int]:
    h, w = img_rgb.shape[:2]
    scale = min(target / w, target / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y

class NoduleClassifier(nn.Module):
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
        fused = self.meam(features_list)
        return self.classifier_head(fused)

class Classifieur:
    def __init__(self, model_path, device=device):
        self.device = device
        self.model = NoduleClassifier()
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

    def preprocess(self, img_np: np.ndarray) -> torch.Tensor:
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif len(img_np.shape) == 3:
            img_gray = img_np[:, :, 0]
        else:
            img_gray = img_np
        if img_gray.dtype != np.uint8:
            mn, mx = img_gray.min(), img_gray.max()
            img_gray = ((img_gray - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        t_raw   = torch.from_numpy(img_gray).float() / 255.0
        t_clahe = torch.from_numpy(img_clahe).float() / 255.0
        t_input = torch.stack([t_raw, t_clahe], dim=0).unsqueeze(0).to(self.device)
        return _build_3ch_batch(t_input, SEPEPreprocessor())

    def pred(self, img_np: np.ndarray, threshold: float) -> torch.Tensor:
        img_tensor = self.preprocess(img_np)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            return (probs[:, 1] >= threshold).long()

class Localisation:
    def __init__(self, model_path: str, device=device):
        self.model = YOLO(model_path).to(device)

    def pred(self, img_np: np.ndarray, conf_threshold: float, img_size: int) -> List[dict]:
        img_rgb = preprocess_image(img_np)
        img_lb, scale, pad_x, pad_y = _letterbox(img_rgb, img_size)
        h_orig, w_orig = img_rgb.shape[:2]
        x_min_v = pad_x
        y_min_v = pad_y
        x_max_v = pad_x + int(round(w_orig * scale))
        y_max_v = pad_y + int(round(h_orig * scale))
        img_gray_lb = cv2.cvtColor(img_lb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        results = self.model(img_lb, conf=conf_threshold, verbose=False)
        nodules = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if not (x_min_v <= cx <= x_max_v and y_min_v <= cy <= y_max_v):
                continue
            rx1, ry1 = int(max(0, x1)), int(max(0, y1))
            rx2, ry2 = int(min(img_size, x2)), int(min(img_size, y2))
            if rx2 > rx1 and ry2 > ry1:
                patch = img_gray_lb[ry1:ry2, rx1:rx2]
                mean_intensity = float(patch.mean())
                if mean_intensity > 200:
                    continue
            cx_orig = (cx - pad_x) / scale
            cy_orig = (cy - pad_y) / scale
            nodules.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': cx_orig,
                'cy': cy_orig,
                'cx_lb': cx,
                'cy_lb': cy,
                'conf': conf
            })
        return nodules

class Model:
    def __init__(
        self,
        classif_path: str,
        weight: str,
        seuil_conf_class: float,
        seuil_rattrapage: float,
        seuil_conf_local_lo: float,
        yolo_img_size: int,
        device=device,
    ):
        self.classifieur   = Classifieur(model_path=classif_path, device=device)
        self.localisateur  = Localisation(model_path=weight, device=device)
        self.seuil_conf_class = seuil_conf_class
        self.seuil_rattrapage = seuil_rattrapage
        self.seuil_conf_local_lo = seuil_conf_local_lo
        self.yolo_img_size = yolo_img_size

    def predict(self, img_np: np.ndarray) -> dict:
        pred_class = self.classifieur.pred(img_np, threshold=self.seuil_conf_class).item()
        positive = bool(pred_class)
        source   = "classifier" if positive else None
        if not positive:
            fallback = self.localisateur.pred(img_np, conf_threshold=self.seuil_rattrapage, img_size=self.yolo_img_size)
            if fallback:
                positive = True
                source   = "fallback_localizer"
        if positive:
            nodules = self.localisateur.pred(img_np, conf_threshold=self.seuil_conf_local_lo, img_size=self.yolo_img_size)
            return {"positive": True, "source": source, "nodules": nodules}
        return {"positive": False, "source": "negative", "nodules": []}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--classif_weights", type=str, default="weights/classifieur.pth")
    parser.add_argument("--local_weights", type=str, default="weights/localisateur.pt")
    parser.add_argument("--seuil_conf_class", type=float, default=0.38)
    parser.add_argument("--seuil_conf_local_lo", type=float, default=0.2)
    parser.add_argument("--seuil_rattrapage", type=float, default=0.86)
    parser.add_argument("--yolo_img_size", type=int, default=640)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    model = Model(
        classif_path=args.classif_weights,
        weight=args.local_weights,
        seuil_conf_class=args.seuil_conf_class,
        seuil_rattrapage=args.seuil_rattrapage,
        seuil_conf_local_lo=args.seuil_conf_local_lo,
        yolo_img_size=args.yolo_img_size
    )

    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".bmp")
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)]

    classif_data = []
    localiz_data = []

    for img_name in tqdm(image_files, desc="Inference"):
        img_path = os.path.join(args.input_dir, img_name)
        
        try:
            img_pil = Image.open(img_path)
            img_np = np.array(img_pil)
        except Exception:
            continue
        
        result = model.predict(img_np)
        
        label = "Nodule" if result["positive"] else "No Finding"
        
        conf_class = 1.0
        if result["nodules"]:
            conf_class = max([n['conf'] for n in result["nodules"]])
        elif not result["positive"]:
            conf_class = 1.0

        classif_data.append({
            "file_name": img_name,
            "label": label,
            "confidence": conf_class
        })

        if result["positive"] and result["nodules"]:
            for nodule in result["nodules"]:
                localiz_data.append({
                    "file_name": img_name,
                    "x": nodule["cx"],
                    "y": nodule["cy"],
                    "confidence": nodule["conf"]
                })

    df_class = pd.DataFrame(classif_data, columns=["file_name", "label", "confidence"])
    df_class.to_csv(os.path.join(args.output_dir, "classification_test_results.csv"), index=False)

    df_loc = pd.DataFrame(localiz_data, columns=["file_name", "x", "y", "confidence"])
    df_loc.to_csv(os.path.join(args.output_dir, "localization_test_results.csv"), index=False)