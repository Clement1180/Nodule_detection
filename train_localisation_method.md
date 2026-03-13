# Methodology for Nodule Localizer Training

**Team:** UQAC

## 1. Overview
This document details the training and optimization pipeline for our pulmonary nodule localization model. Due to the limited number of fully annotated bounding boxes in the provided dataset, our methodology relies on a semi-supervised learning approach. We utilize a highly capable pre-trained teacher model, an iterative pseudo-labeling loop, and a rigorous grid-search optimization to maximize the final F1-Score.

## 2. Base Model and Pre-training (Teacher Model)
To overcome the "cold start" problem associated with small medical datasets, we initialized our localizer using a **YOLOv8** architecture that was already heavily fine-tuned for thoracic anomalies. 

Specifically, the model was pre-trained on the **VinBigData Chest X-ray Abnormalities Detection** dataset. This large-scale dataset contains thousands of chest X-rays annotated by multiple radiologists for various pulmonary lesions, including nodules.
* **Dataset Source:** [Kaggle - VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)
* **Model Checkpoint Reference:** [Kaggle Notebook - Final VinBigData CXR AD YOLOv8](https://www.kaggle.com/code/istiyaque6ty3/final-vinbigdata-cxr-ad-yolov8)

Using this externally pre-trained model provides our pipeline with strong initial feature extractors (edges, textures, and anatomical structures specific to X-rays), allowing the model to converge quickly and accurately on our specific target domain.

## 3. Iterative Semi-Supervised Training Loop
Instead of training the model solely on the small subset of strictly annotated images, we implemented an iterative pseudo-labeling loop. This allows the model to continuously expand its own training dataset by generating high-confidence annotations on unannotated, but globally positive, images (images known to contain nodules but lacking spatial bounding boxes).

The loop operates through the following steps:
1.  **Dataset Generation:** The initial dataset is built using the provided spatial ground truth. Images are letterboxed to a resolution of 640x640 to maintain aspect ratios without distortion. Ground truth (x, y) coordinates are converted into fixed-size bounding boxes biologically representative of standard nodules.
2.  **Training Phase:** The YOLOv8 model is trained on the available dataset for a fixed number of epochs. Hyperparameters are strictly controlled to prevent overfitting (e.g., low learning rate, freezing early backbone layers during the first iteration, and applying targeted spatial augmentations like horizontal flipping).
3.  **Evaluation Gate:** Before generating pseudo-labels, the model's Mean Average Precision (mAP50) is evaluated. If the model does not reach a minimum reliability threshold (e.g., mAP50 > 0.35), the pseudo-labeling phase is bypassed to prevent poisoning the dataset with noisy predictions.
4.  **Pseudo-Labeling (Data Augmentation):** The trained model performs inference on unannotated positive images. Detections that exceed a strict confidence threshold are extracted.
5.  **Deduplication:** A spatial Non-Maximum Suppression (NMS) algorithm filters out overlapping pseudo-labels to retain only the single most confident bounding box per localized cluster.
6.  **Iteration:** These new high-confidence pseudo-labels are added to the ground truth dataframe, and the loop restarts. With each iteration, the model learns from a richer, more diverse dataset, steadily increasing its recall and robustness.

## 4. Threshold Optimization via Grid Search
The final performance of the pipeline depends heavily on the decision thresholds used during inference. Our pipeline employs a dual-check system combining a binary image classifier and the YOLOv8 localizer. 

To find the perfect mathematical balance between Precision (minimizing false alarms) and Recall (minimizing missed tumors), we conduct a vector-based **Grid Search**.

**Grid Search Process:**
1.  **Pre-computation:** To ensure the search is completed in seconds rather than hours, the raw probability outputs are pre-calculated for a balanced sample of 1,000 images (500 positive, 500 negative). We store the raw continuous probability from the classifier and the maximum confidence score outputted by YOLO for each image.
2.  **Combinatorial Testing:** The algorithm tests dozens of combinations across two main axes:
    * **Classifier Threshold:** The minimum probability required for the classifier to flag an image as positive.
    * **Catch-up Threshold (Seuil de Rattrapage):** A secondary threshold. If the classifier considers the image healthy, but YOLO detects a nodule with a confidence strictly higher than this catch-up threshold, the image is forcefully re-classified as positive (acting as a "second-reader" safety net).
3.  **F1-Score Maximization:** For every combination, the True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN) are calculated. The pipeline automatically selects the pair of thresholds that yields the highest absolute **F1-Score**.
4.  **Heatmap Generation:** A 2D heatmap is generated to visually validate the stability of the chosen thresholds, ensuring the selected hyperparameters lie in a robust optimal zone rather than an isolated peak.

## 5. Conclusion
By integrating a robust externally pre-trained model with an iterative semi-supervised loop, we effectively solved the challenge of limited spatial annotations. Furthermore, the automated grid-search ensures that the final model's decision boundaries are clinically optimized, successfully maximizing both spatial accuracy and overall predictive reliability.

written bt Romain AMIGON