import torch

import matplotlib.pyplot as plt
import numpy as np

from src.loader import *
from torch.utils.data import DataLoader

def train_model(
    model,
    train_loader,
    val_loader, 
    num_epochs,
    optimizer,
    criterion,
    device, 
    save_path=f"./model.pt",
    plot=False,
    save_model=False
):
    training_loss_per_epoch = []
    val_loss_per_epoch = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()

        run_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f'[Epoch: {epoch + 1} / {num_epochs},'
                    f' Iter: {i + 1:5d} / {len(train_loader)}]'
                    f' Training loss: {run_loss / (i + 1):.3f}'
                )
            
        mean_loss = run_loss / len(train_loader)
        training_loss_per_epoch.append(mean_loss)

        model.eval()

        run_loss = 0.0
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            run_loss += loss.item()

        mean_loss = run_loss / len(val_loader)
        val_loss_per_epoch.append(mean_loss)

        print(
            f'[Epoch: {epoch + 1} / {num_epochs}]'
            f' Validation loss: {mean_loss:.3f}'
        )

        if mean_loss < best_val_loss and save_model:
            best_val_loss = mean_loss
            torch.save(model.state_dict(), save_path)
            print("Model saved!")

    if plot:
        plt.figure()
        plt.plot(np.array(training_loss_per_epoch))
        plt.plot(np.array(val_loss_per_epoch))
        plt.legend(['Training loss', 'Val loss'])
        plt.xlabel('Epoch')
        plt.show()
        plt.close()

def test_model(model, test_loader, device):
    """
    Evaluates the model on the held-out test set.
    """
    # Switch to evaluation mode
    model.eval()
    
    correct = 0
    total = 0
    
    # Disable gradient calculation for memory efficiency and speed
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions (the index of the max log-probability)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the held-out test set: {accuracy:.2f}%')
    return accuracy

def main(best_params):
    """Function to train the final model using the best found parameters."""
    print(f"--- Training final model with best params: {best_params} ---")

    # Create DataLoaders
    parsed = parser() 
    BATCH_SIZE = best_params["batch_size"]
    NUM_EPOCHS = best_params["num_epochs"]
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

    train_model(model, train_loader, val_loader, NUM_EPOCHS, optimizer, criterion, device, plot=parsed["plot"], save_model=True)
    accuracy = test_model(model, test_loader, device)


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