import time
import torch

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from src.loader import *
from torch.utils.data import DataLoader
from tasks.task_0_baseline_cnn import *

class PretrainedViTClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PretrainedViTClassifier, self).__init__()
        # Load the standard pre-trained ViT
        weights = models.ViT_B_16_Weights.DEFAULT1
        self.vit = models.vit_b_16(weights=weights)

        for param in self.vit.parameters():
            param.requires_grad = False
        
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Resize from 128x128 to 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Duplicate grayscale channel to 3 channels (B, 3, 224, 224)
        x = x.repeat(1, 3, 1, 1)
        # Pass through the ViT
        return self.vit(x)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def vit(best_params):
    """Function to train the ViT."""
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
    model = PretrainedViTClassifier()

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=best_params["lr"], momentum=best_params["momentum"], mode=best_params["optimizer"])

    # Load data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    start_time = time.time()
    
    train_model(model, train_loader, val_loader, NUM_EPOCHS, optimizer, criterion, device, plot=True, save_model=False)
    
    end_time = time.time()
    training_duration = end_time - start_time

    accuracy = test_model(model, test_loader, device)
    
    print(f"\n--- Training Statistics ---")
    print(f"Total Training Time: {training_duration:.2f} seconds")
    print(f"Final Test Accuracy: {accuracy:.2f}%")

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