import torch
import optuna

import torchvision.transforms as T

from src.loader import *
from torch.utils.data import DataLoader
from tasks.task_0_baseline_cnn import *

def main(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    mode = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # Create DataLoaders
    parsed = parser()
    BATCH_SIZE = batch_size
    NUM_EPOCHS = parsed["NUM_EPOCHS"]
    verbose = parsed["verbose"]

    # Example: Load ASB dataset
    train_dataset = load_dataset("ASB", "train")
    val_dataset = load_dataset("ASB", "val")
    test_dataset = load_dataset("ASB", "test")

    if verbose:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Test iteration
    if verbose:
        for batch_images, batch_labels in train_loader:
            print(f"Batch images shape: {batch_images.shape}")  # (32, 1, 128, 128)
            print(f"Batch labels shape: {batch_labels.shape}")  # (32,)
            print(f"Image dtype: {batch_images.dtype}")
            print(f"Label dtype: {batch_labels.dtype}")
            break

    # Example usage with custom Dataset
    train_dataset = ImageClassificationDataset("ASB", "train")

    # Get a single sample
    image, label = train_dataset[0]
    if verbose:
        print(f"\nSingle sample:")
        print(f"  Image shape: {image.shape}")
        print(f"  Label: {label}")

    # Using torchvision transforms with the custom Dataset.
    # Define augmentation transforms
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    # Create dataset with transforms
    train_dataset_aug = ImageClassificationDataset(
        category="ASB", 
        split="train", 
        transform=train_transform
    )

    # Test augmentation
    image_aug, label = train_dataset_aug[0]
    if verbose:
        print(f"Augmented image shape: {image_aug.shape}")

    # Create model
    model = CNN(block=block, layers=[3, 4, 6, 3], image_channels=1, num_classes=2)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    model = model.to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=lr, momentum=0.9, mode=mode)

    # Load data
    train_dataset = load_dataset("ASB", "train")
    val_dataset = load_dataset("ASB", "val")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_model(model, train_loader, val_loader, NUM_EPOCHS, optimizer, criterion, device, plot=parsed["plot"])
    accuracy = test_model(model, test_loader, device)
    
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()