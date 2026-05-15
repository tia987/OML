import torch

import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.loader import *
from tasks.task_0_baseline_cnn import *

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fracture', 'Fracture'], 
                yticklabels=['No Fracture', 'Fracture'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.savefig("./tasks/task_4_confusion_matrix/figures/confusion.png")
    plt.show()

def confusion(best_params):
    """
    Trains a model using best_params and evaluates it with a confusion matrix.
    Mimics the training flow of Task 0[cite: 115, 116].
    """
    print(f"--- Running Task 4 with best params: {best_params} ---")

    BATCH_SIZE = best_params["batch_size"]
    NUM_EPOCHS = best_params["num_epochs"]
    
    train_dataset = load_dataset("NT", "train")
    val_dataset = load_dataset("NT", "val")
    test_dataset = load_dataset("NT", "test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=best_params["lr"], 
                              momentum=best_params["momentum"], 
                              mode=best_params["optimizer"])

    from tasks.task_0_baseline_cnn.train import train_model
    train_model(model, train_loader, val_loader, NUM_EPOCHS, optimizer, criterion, device)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Task 4: Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
    
    plot_confusion_matrix(all_labels, all_preds)

    print(f"F1 = {f1_score(all_labels, all_preds):.2f}")