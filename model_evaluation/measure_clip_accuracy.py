import json
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Evaluate model accurracy")
parser.add_argument("ground_truth", type=str, help="Path to the ground truth JSON file ")
parser.add_argument("predictions", type=str, help="Path to the predictions JSON file")

# Parse arguments
args = parser.parse_args()

# Load the original JSON file
# Load the ground truth and predictions JSON files
with open(args.ground_truth, "r") as f:
    ground_truth = json.load(f)

with open(args.predictions, "r") as f:
    predictions = json.load(f)

# Convert ground truth and predictions to dictionaries for easier comparison
ground_truth_dict = {entry["img"]: entry["label"] for entry in ground_truth}
predictions_dict = {entry["img"]: entry["label"] for entry in predictions}

# Variables to keep track of correct predictions and total comparisons
correct_predictions = 0
total_images = 0

# Initialize counts for True Positives, False Positives, False Negatives
TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives
TN = 0  # True Negatives


# Compare predictions to ground truth and calculate TP, FP, FN, TN
for img, true_label in ground_truth_dict.items():
    if img in predictions_dict:
        predicted_label = predictions_dict[img]
        if true_label == 1 and predicted_label == 1:
            TP += 1  # True positive: correctly predicted offensive
        elif true_label == 0 and predicted_label == 1:
            FP += 1  # False positive: predicted offensive but not offensive
        elif true_label == 1 and predicted_label == 0:
            FN += 1  # False negative: predicted not offensive but it was offensive
        elif true_label == 0 and predicted_label == 0:
            TN += 1  # True negative: correctly predicted not offensive

# Calculate precision, recall, and F1-score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print the results
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Negatives (TN): {TN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
