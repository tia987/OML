from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

import numpy as np

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
):
    training_loss_per_epoch = []
    val_loss_per_epoch = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()

        run_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

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
            # inputs, labels = data
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

        if mean_loss < best_val_loss:
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