# Task 1: Hyperparameter Studies & Fine-tuning

## Objective
Systematically explore how hyperparameters affect model performance when training a Convolutional Neural Network (CNN) for binary fracture classification on the **NT** dataset. 

## Approach
This study utilized **Optuna** to perform an automated hyperparameter search. Optuna systematically sampled the search space over multiple trials to maximize the model's evaluation metrics. The search space included both categorical choices (like batch size and optimizer types) and floating-point distributions (like learning rate). 

## Hyperparameters Explored
- **Learning rate (`lr`)**: Float distribution (Log scale), range `[1e-06, 0.01]`
- **Batch size**: Categorical, choices: `[16, 32, 64, 128]`
- **Optimizer**: Categorical, choices: `["Adam", "SGD"]`
- **Number of Epochs (`num_epochs`)**: Categorical, choices: `[1, 4, 16, 64]`
- **Momentum**: Categorical, choices: `[0.1, 0.2, 0.4, 0.7, 0.9]`

## Results
The hyperparameter search yielded a comprehensive understanding of which parameters most significantly impacted the model's learning trajectory. 

*Please refer to the generated visual reports in this directory:*
* ![Optimization History](./results/optimization_history.png) *(Shows the objective value progression over all Optuna trials)*
* ![Parameter Importances](./results/param_importances.png) *(Displays the relative importance of each hyperparameter on the final objective)*

## Best Configuration
After completing the trials, Optuna identified the following combination as the optimal setup:

| Hyperparameter | Optimal Value |
| :--- | :--- |
| **Learning Rate** | `0.0010666286480155401` |
| **Batch Size** | `16` |
| **Optimizer** | `Adam` |
| **Number of Epochs** | `64` |
| **Momentum** | `0.9` |

## Files
- `objective.py` - The Optuna objective script defining the model, data loaders, and search space.
- `best_search_optuna.db` - The SQLite database storing the full trial history and intermediate values.
- `best_params.json` - JSON export of the optimal hyperparameter configuration.
- `optimization_history.png` - Scatter plot visualization of the objective values across all trials.
- `param_importances.png` - Bar chart illustrating the relative importance of each searched parameter.