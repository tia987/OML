# Task 2: Robustness Analysis (Gaussian Noise)

## Objective
Analyze how model accuracy degrades with increasing Gaussian noise to evaluate the robustness of our binary fracture classification CNN.

## Approach
To simulate noisy imaging conditions, we systematically injected Gaussian noise into the normalized test images. For each defined noise level ($\sigma$), noise was generated using `torch.randn_like(images) * sigma` and added to the original image tensors. The resulting pixel values were then clamped to the valid range of `[0, 1]`. The model (trained with the optimal parameters from Task 1) was evaluated on these corrupted datasets to measure the resulting drop in test accuracy.

## Noise Levels Tested
$\sigma \in \{0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50\}$

## Results

| $\sigma$ (Noise Level) | Accuracy (%) |
| :--- | :--- |
| **0.00** (Baseline) | `[Insert %]` |
| **0.05** | `[Insert %]` |
| **0.10** | `[Insert %]` |
| **0.15** | `[Insert %]` |
| **0.20** | `[Insert %]` |
| **0.30** | `[Insert %]` |
| **0.50** | `[Insert %]` |

*Please refer to the visual report in this directory:*
* ![Noise Robustness Plot](./figures/noise_robustness_plot.png) *(Displays the model's test accuracy as a function of the Gaussian noise level)*

## Analysis
As visualized in the accuracy vs. noise plot, the model's performance steadily declines as $\sigma$ increases. 
* **Low Noise ($\sigma \le 0.10$):** The model generally maintains a reasonable level of robustness, likely because the core structural features (edges, macroscopic cracks) remain distinguishable.
* **High Noise ($\sigma \ge 0.20$):** The model experiences a significant degradation in accuracy. At these noise levels, the Gaussian artifacts begin to mask the high-frequency features (such as micro-fractures) that the CNN relies on for classification. The added noise effectively shifts the test data distribution away from the clean training distribution, causing the model to struggle to extract meaningful feature maps.

## Files
- `noise.py` - The main Python script used to apply Gaussian noise, evaluate the model, and generate the plots.
- `noise_robustness_plot.png` - The line plot illustrating the relationship between test accuracy and noise $\sigma$.