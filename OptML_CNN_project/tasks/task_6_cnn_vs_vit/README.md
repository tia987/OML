# Task 6: Architecture Comparison (CNN vs ViT)

## Objective
Compare the performance, efficiency, and training dynamics of a custom-built Convolutional Neural Network (CNN) against a state-of-the-art Vision Transformer (ViT) on a relatively small dataset of mechanical fractures.

## Models Compared
- **CNN**: A custom, lightweight sequential model trained entirely from scratch (Task 0). It consists of 4 convolutional blocks (with Batch Normalization and LeakyReLU) and processes native $128 \times 128$ grayscale images.
- **ViT**: A pre-trained `ViT-B/16` architecture loaded from `torchvision.models`. To adapt it to our specific task:
  - The base transformer weights were **frozen** (Transfer Learning) to prevent overfitting on our small dataset.
  - The final classification head was replaced with a new linear layer mapping to our 2 classes.
  - Grayscale inputs were dynamically upsampled to $224 \times 224$ and duplicated across 3 channels to match the ViT's expected input dimensions.

## Comparison Results

| Metric | CNN (Baseline) | ViT (Pre-trained) |
| :--- | :--- | :--- |
| **Test Accuracy** | 96.81% | 97.87% |
| **Total Parameters** | `[Insert CNN Params]` | ~86.5 M |
| **Trainable Parameters** | `[Insert CNN Params]` | 1,538 (Linear Head) |
| **Training Time** | 404.68 min | 4731.95 sec (~78.87 min) |

*Refer to the training curves below for loss and validation trends:*
* ![CNN Training Progress](../task_0_baseline_cnn/results/CNN.png) *(CNN convergence from scratch)*
* ![ViT Training Progress](./figures/ViT.png) *(ViT convergence using transfer learning)*

## Analysis
The architectural comparison reveals distinct trade-offs between custom convolutional networks and large-scale attention-based models:

* **Data Efficiency & Transfer Learning**: Training a massive architecture like ViT from scratch on roughly 6,000 images would almost certainly lead to severe overfitting. By leveraging transfer learning, the ViT can utilize robust, generalized feature extractors learned from millions of images (ImageNet), requiring only the final head (1,538 parameters) to be fine-tuned. 
* **Computational Load**: The ViT trained significantly faster than the baseline CNN (approx. 79 minutes vs. 404 minutes) because the vast majority of its weights were frozen. However, during inference, the ViT is heavier due to its 86+ million total parameters and the need to upsample inputs to $224 \times 224 \times 3$.
* **Performance**: The ViT achieved an impressive **97.87% test accuracy**, outperforming the baseline CNN (96.81%). This demonstrates that pre-trained attention mechanisms are highly capable of detecting subtle, microscopic fractures, even when adapted from general-purpose image datasets.

## Files
- `vit.py` - Script containing the `PretrainedViTClassifier` implementation and training loop.
- `CNN.png` - Visual representation of the CNN's training and validation metrics.
- `ViT.png` - Visual representation of the Vision Transformer's training and validation metrics.