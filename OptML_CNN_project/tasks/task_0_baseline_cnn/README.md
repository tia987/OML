# Task 0: Baseline CNN Classifier

## Objective
Build and train a foundational Convolutional Neural Network (CNN) to perform binary classification (fracture vs. no fracture) on the provided dataset. This baseline serves as the performance benchmark for all subsequent optimizations and robustness tests.

## Your Implementation
The baseline model is a custom sequential CNN designed to extract spatial features through four consecutive convolutional blocks followed by a dense classifier.

### Architecture Details:
- **Feature Extractor**: 
    - **4 Convolutional Blocks**: Each block consists of a `Conv2d` layer (increasing filters from 16 to 128), followed by **Batch Normalization** to stabilize training, a **LeakyReLU** activation function to prevent "dying ReLU" issues, and **MaxPool2d** for spatial downsampling.
- **Classifier**:
    - The output of the final convolutional block is flattened and passed through three **Fully Connected (Linear) layers**.
    - **Dropout** (p=0.5) is applied between the linear layers to mitigate overfitting and improve generalization.
- **Training Setup**:
    - **Loss Function**: Cross-Entropy Loss.
    - **Optimizer**: Adam (selected as the default robust optimizer).
    - **Input**: Grayscale images resized/normalized to $128 \times 128$.

## Results
The baseline model achieved high accuracy relatively quickly, though it required a significant amount of wall-clock time depending on the hardware environment.

- **Final Test Accuracy**: `96.81%`
- **Total Training Time**: `404.68 minutes`

*Refer to the training curves below for loss and validation trends:*
* ![Training Progress](./results/CNN.png)

*(The plot illustrates the convergence of training and validation loss, ensuring the model isn't significantly over-fitting before reaching the final epochs.)*

## Files
- `model.py` - Contains the `CNN` class definition and the `block` architecture.
- `train.py` - The main execution script that handles the training loop, validation, and performance logging.
- `Figure_1.png` - Visual representation of the training and validation metrics over time.
- `model.pt` - The saved weights of the trained baseline model.