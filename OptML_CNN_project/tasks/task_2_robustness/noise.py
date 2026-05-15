import torch

from src.loader import *
from torch.utils.data import DataLoader
from tasks.task_0_baseline_cnn import *


def add_gaussian_noise(images, sigma):
    noise = torch.randn_like(images)*sigma
    return torch.clamp(images+noise, 0, 1)

def noise(best_params):
    """Function to train the final model using the best found parameters."""
    print(f"--- Training final model with best params: {best_params} ---")


    # Create DataLoaders
    parsed = parser() 
    BATCH_SIZE = best_params["batch_size"]
    NUM_EPOCHS = best_params["num_epochs"]
    noise_sigma = parsed["noise_sigma"]
    verbose = parsed["verbose"]

    # Load ASB dataset
    # train_dataset = load_dataset("ASB", "train")
    train_dataset = load_dataset("NT", "train")
    val_dataset = load_dataset("NT", "val")
    test_dataset = load_dataset("NT", "test")

    # Create model
    model = CNN()

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=best_params["lr"], momentum=best_params["momentum"], mode=best_params["optimizer"])

    # Load data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    accuracy = []

    for sigma in noise_sigma:
        test_loader = add_gaussian_noise(test_loader, sigma)

        train_model(model, train_loader, val_loader, NUM_EPOCHS, optimizer, criterion, device, plot=parsed["plot"], save_model=True)
        accuracy.append(test_model(model, test_loader, device))

    if verbose:
        image_aug, label = train_dataset[0]
        print(f"Augmented image shape: {image_aug.shape}")
        print(f"Using device: {device}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        # Test iteration
        for batch_images, batch_labels in train_loader:
            print(f"Batch images shape: {batch_images.shape}")  # (32, 1, 128, 128)
            print(f"Batch labels shape: {batch_labels.shape}")  # (32,)
            print(f"Image dtype: {batch_images.dtype}")
            print(f"Label dtype: {batch_labels.dtype}")
            break
    
    return accuracy