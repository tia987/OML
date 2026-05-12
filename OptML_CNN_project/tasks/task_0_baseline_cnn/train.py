from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from evaluation import evaluate_model
import numpy as np


def train_model(
    model,
    train_loader,
    val_loader, 
    num_epochs,
    optimizer,
    criterion,
    device, 
    save_path=f"./ckpt/model.pt"
):
    """
    Feel free to change the arguments of this function - if necessary.

    Trains the model on the given dataset. Selects the best model based on the
    validation set and saves it to the given path. 
    Inputs: 
        model: The model to train [nn.Module]
        train_loader: The training data loader [DataLoader]
        val_loader: The validation data loader [DataLoader]
        num_epochs: The number of epochs to train for [int]
        optimizer: The optimizer [Any]
        best_of: The metric to use for validation [str: "loss" or "accuracy"]
        device: The device to train on [str: cpu, cuda, or mps]
        save_path: The path to save the model to [str]
    Output:
        Dictionary containing the training and validation losses and accuracies
        at each epoch. Also contains the epoch number of the best model.
    """

    #
    # You can put your training loop here
    #

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

    plt.figure()
    plt.plot(np.array(training_loss_per_epoch))
    plt.plot(np.array(val_loss_per_epoch))
    plt.legend(['Training loss', 'Val loss'])
    plt.xlabel('Epoch')
    plt.show()
    plt.close()