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
## 6. Hardware used for the training
- For the classification task, we used an NVIDIA RTX 4070 Super; the training took one hour and a half.

- For the localization task, we used an NVIDIA RTX 4070 Super; the training took forty minutes.

## 7. Inference Time

- On an NVIDIA RTX 3060 laptop, the inference time was 1.60 seconds.


written by Romain AMIGON and Clément BARDIN
