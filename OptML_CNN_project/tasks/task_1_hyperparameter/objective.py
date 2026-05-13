import torch

import torchvision.transforms as T

from src.loader import *
from torch.utils.data import DataLoader
from tasks.task_0_baseline_cnn import *

def objective(trial):
    # Load the config
    parsed = parser() 

    search_space = parsed["search_space"]
    lr = trial.suggest_float("lr", search_space["lr"]["low"], search_space["lr"]["high"], log=search_space["lr"]["log"])
    batch_size = trial.suggest_categorical("batch_size", search_space["batch_size_options"])
    mode = trial.suggest_categorical("optimizer", search_space["optimizer_options"])
    num_epochs = trial.suggest_categorical("num_epochs", search_space["epoch_options"])
    momentum = trial.suggest_categorical("momentum", search_space["momentum_options"])

    # Create DataLoaders
    BATCH_SIZE = batch_size
    NUM_EPOCHS = num_epochs
    verbose = parsed["verbose"]

    # Load ASB dataset
    # train_dataset = load_dataset("ASB", "train")
    train_dataset = load_dataset("NT", "train")
    val_dataset = load_dataset("NT", "val")
    test_dataset = load_dataset("NT", "test")
    # Using torchvision transforms with the custom Dataset.
    # Define augmentation transforms
    # base_transform = T.Compose([
    #     T.Normalize(mean=[0.5], std=[0.5])
    # ])
    # train_transform = T.Compose([
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.RandomRotation(degrees=10),
    #     T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    #     T.Normalize(mean=[0.5], std=[0.5]),
    # ])
    # # Create dataset with transforms
    # # train_dataset_aug = ImageClassificationDataset(
    # train_dataset = ImageClassificationDataset(category="ASB", split="train", transform=train_transform)
    # val_dataset = ImageClassificationDataset(category="ASB", split="val", transform=base_transform)
    # test_dataset = ImageClassificationDataset(category="ASB", split="test", transform=base_transform)

    # Create model
    model = CNN()

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    model = model.to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=lr, momentum=momentum, mode=mode)

    # Load data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_model(model, train_loader, val_loader, NUM_EPOCHS, optimizer, criterion, device, plot=parsed["plot"])
    accuracy = test_model(model, test_loader, device)
    
    return accuracy