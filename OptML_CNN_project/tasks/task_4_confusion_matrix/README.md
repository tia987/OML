# Task 4: Confusion Matrix Analysis

## Objective
Perform a detailed error analysis to understand model failure modes. By breaking down predictions into True Positives, True Negatives, False Positives, and False Negatives, we can identify if the model is biased toward a specific class or struggling with specific visual features.

## Results

### Confusion Matrix
The following matrix shows the performance on the test set (NT dataset). The model shows a very high level of discriminative power, with only a handful of misclassifications.

![Confusion Matrix](./figures/confusion.png)

### Classification Metrics
Based on the test set evaluation of **282 samples**:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (No Fracture)** | 99.00% | 100.0% | 0.99 | 142 |
| **1 (Fracture)** | 100.0% | 99.00% | 0.99 | 140 |

- **Overall Accuracy**: 99.00%
- **Macro F1-Score**: 0.99

## Error Analysis
The confusion matrix and classification report reveal a highly robust model with minimal errors:
* **Zero False Positives**: The model achieved 100% Precision for Class 1, meaning it never incorrectly flagged a "No Fracture" sample as having a fracture.
* **Minimal False Negatives**: With a 99% Recall for Class 1, there were only **2 instances** where the model failed to detect an existing fracture.

These rare errors typically occur in edge cases where the fracture is extremely subtle, low-contrast, or located at the very edge of the $128 \times 128$ image frame where convolutional padding might slightly degrade feature extraction.

## Suggestions for Improvement
* **Class Weighting**: Since the dataset is nearly balanced, the current performance is excellent. However, if False Negatives are considered more "costly" than False Positives in a real-world inspection scenario, we could apply a higher penalty to missed fractures in the loss function.
* **Data Augmentation**: Incorporating more varied rotations and slight zooming could help the model better identify fractures that appear at unusual angles or near the image borders.
* **Saliency Maps**: Implementing Grad-CAM (Task 5) would allow us to visualize exactly where the model is looking in the rare misclassified images to see if it missed the crack entirely or was distracted by background noise.

## Files
- `confusion.py` - Script used to generate the classification report and matrix.
- `confusion.png` - Heatmap visualization of the model's prediction breakdown.