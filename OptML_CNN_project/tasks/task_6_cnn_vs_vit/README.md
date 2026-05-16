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
| **Test Accuracy** | 96.81% | `[Insert %]` |
| **Trainable Parameters** | `[Insert CNN Param Count]` | `[Insert ViT Head Params]` |
| **Total Parameters** | `[Insert CNN Param Count]` | ~86.5 M |
| **Training Time** | 404.68 min | `[Insert min]` |

*Refer to the training curves below for loss and validation trends:*
* ![CNN Training Progress](CNN.png) *(CNN convergence from scratch)*
* ![ViT Training Progress](ViT.png) *(ViT convergence using transfer learning)*

## Analysis
The architectural comparison reveals distinct trade-offs between custom convolutional networks and large-scale attention-based models:

* **Data Efficiency & Transfer Learning**: Training a massive architecture like ViT from scratch on roughly 6,000 images would almost certainly lead to severe overfitting. By leveraging transfer learning, the ViT can utilize robust, generalized feature extractors learned from millions of images (ImageNet), requiring only the final head to be fine-tuned. 
* **Computational Load**: While the CNN takes a significant amount of time to learn edge/texture features from scratch (404+ minutes), it is much lighter during inference. The ViT, conversely, has over 86 million parameters and requires upsampling the images, making it substantially heavier to run, even if fine-tuning just the head is relatively quick.
* **Performance**: *(Add a brief sentence here once you insert your ViT accuracy—e.g., "The ViT achieved X% accuracy, showing that pre-trained attention mechanisms are highly capable of detecting microscopic fractures, though the custom CNN still remains highly competitive for this specific domain.")*

## Files
- `vit.py` - Script containing the `PretrainedViTClassifier` implementation and training loop.
- `CNN.png` - Visual representation of the CNN's training and validation metrics.
- `ViT.png` - Visual representation of the Vision Transformer's training and validation metrics.