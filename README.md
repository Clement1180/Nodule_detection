# NoduLoCC2026 - Setup and Inference Guide

**TEAM UQAC**



This guide provides instructions to set up the environment and run the inference script for the NoduLoCC2026 challenge.

## 1. Installation

1. Ensure you have Python 3.9+ installed.
2. Open your terminal in the root directory of the project.
3. Install the required dependencies by running:

pip install -r requirements.txt

## 2. Project Structure

Make sure your trained weights are correctly placed in the `weights` directory before running the script:
- weights/classifieur.pth
- weights/localisateur.pt

## 3. Running the Prediction

Use the following command to run the inference script. Replace `<INPUT_DIR>` with the path to your image folder and `<OUTPUT_DIR>` with the folder where you want the standardized CSV files to be saved.

python pred.py --input <INPUT_DIR> --output <OUTPUT_DIR>

### Example:
python pred.py --input dataset/test_images --output results

## 4. Expected Output

After execution, the script will generate two files in your output directory:
- classification_test_results.csv
- localization_test_results.csv

## 5. Pipeline explaination

The prediction pipeline employs a hybrid, two-stage approach combining a custom classifier and a YOLO-based localizer. This architecture is designed to maximize sensitivity while ensuring precise nodule detection.

The inference process follows four main steps:

- Initial Classification: The input image is first processed by the Classifieur (using SEPE preprocessing and CLAHE). It evaluates the image and returns a POSITIVE or NEGATIVE status based on a predefined confidence threshold (SEUIL_CONF_CLASS = 0.38).

- Fallback Mechanism (Double-Check):
If the classifier predicts NEGATIVE, the pipeline triggers a safety check using the YOLO Localisation model with a very strict, high confidence threshold (SEUIL_RATTRAPAGE = 0.86). If a nodule is detected at this stage, the overall status is overridden and switched to POSITIVE (Source: fallback_localizer).

- Fine Localization:
If the image is flagged as POSITIVE (either by the initial classification or the fallback mechanism), the YOLO model performs a fine-grained localization. It uses a lower, more permissive threshold (SEUIL_CONF_LOCAL_LO = 0.2) to detect all potential nodules. The model also applies an anatomical filter to reject false positives located in highly intense regions (like bones or the spine).

- Final Output:
If no nodules are found and the classifier is negative, the image is confirmed as NEGATIVE. The pipeline ultimately returns a dictionary containing the final status (positive: boolean), the source of the positive trigger, and a list of detected nodules with their coordinates and confidence scores.

## 6. Hardware used for the training
- For the classification task, we used an NVIDIA RTX 4070 Super; the training took one hour and a half.

- For the localization task, we used an NVIDIA RTX 4070 Super; the training took forty minutes.

## 7. Inference Time

- On an NVIDIA RTX 3060 laptop, the inference time was 1.60 seconds.


written by Romain AMIGON and Clément BARDIN
