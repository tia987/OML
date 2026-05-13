import argparse
import optuna
import json

from tasks.task_0_baseline_cnn import *
from tasks.task_1_hyperparameter import *

if __name__ == "__main__":
    # Get parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--mode", choices=["search", "train"], default="search", help="Search best parameters or just train (sarch/train)")
    parser.add_argument("-n_trials", "--n_trials", type=int, default=10, help="Number of trials to add.")
    args = parser.parse_args()

    study_name = "fracture_classification"
    best_search_optuna = f"sqlite:///tasks/task_1_hyperparameter/results/best_search_optuna.db"

    if args.mode == "search":
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

        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()

    elif args.mode == "train":
        try:
            study = optuna.load_study(study_name=study_name, storage=best_search_optuna)
            main(study.best_params)
        except KeyError:
            print("No existing study found. Please run with --mode search first.")