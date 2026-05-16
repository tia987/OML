import argparse
import optuna
import json
import os

import matplotlib.pyplot as plt

from tasks.task_0_baseline_cnn import *
from tasks.task_1_hyperparameter import *
from tasks.task_2_robustness import *
from tasks.task_4_confusion_matrix import *
from tasks.task_6_cnn_vs_vit import *

from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)

if __name__ == "__main__":
    # Get parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--mode", choices=["task0", "task1", "task2", "task4", "task6"], default="search", help="Select task (taskX)")
    parser.add_argument("-n_trials", "--n_trials", type=int, default=10, help="Number of trials to add.")
    args = parser.parse_args()

    study_name = "fracture_classification"
    best_search_optuna = f"sqlite:///tasks/task_1_hyperparameter/results/best_search_optuna.db"

    if args.mode == "task1":
        study = optuna.create_study(study_name=study_name, storage=best_search_optuna, load_if_exists=True, direction="maximize")
    
        study.optimize(objective, n_trials=args.n_trials)

        with open("tasks/task_1_hyperparameter/results/best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=4)
        
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (Accuracy): {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

         # Save Optimization History
        fig1 = plot_optimization_history(study)
        fig1.figure.savefig(
            os.path.join(".", "tasks/task_1_hyperparameter/optimization_history.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig1.figure)

        # Save Parameter Importances
        fig2 = plot_param_importances(study)
        fig2.figure.savefig(
            os.path.join(".", "tasks/task_1_hyperparameter/param_importances.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig2.figure)

    elif args.mode == "task0":
        try:
            study = optuna.load_study(study_name=study_name, storage=best_search_optuna)
            main(study.best_params)
        except KeyError:
            print("No existing study found. Please run with --mode search first.")

    elif args.mode == "task2":
        try:
            study = optuna.load_study(study_name=study_name, storage=best_search_optuna)
            noise(study.best_params)
        except KeyError:
            print("No existing study found. Please run with --mode search first.")

    elif args.mode == "task4":
        try:
            study = optuna.load_study(study_name=study_name, storage=best_search_optuna)
            confusion(study.best_params)
        except KeyError:
            print("No existing study found. Please run with --mode search first.")

    elif args.mode == "task6":
        try:
            study = optuna.load_study(study_name=study_name, storage=best_search_optuna)
            vit(study.best_params)
        except KeyError:
            print("No existing study found. Please run with --mode search first.")