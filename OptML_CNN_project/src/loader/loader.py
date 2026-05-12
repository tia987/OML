import torch
import sys

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import TensorDataset, Dataset
from typing import Optional, Callable
from tasks.task_0_baseline_cnn import *

current_file_path = Path(__file__).resolve()
ROOT_DIR = current_file_path.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

def load_data():
    data_path = ROOT_DIR/"data"/"processed"/"ASB"/"train.npz"
    data = np.load(data_path)

    # Access images and labels
    images = data["images"]
    labels = data["labels"]

    print(f"Images shape: {images.shape}")  # (N, 1, 128, 128)
    print(f"Labels shape: {labels.shape}")  # (N,)
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Pixel value range: [{images.min():.2f}, {images.max():.2f}]")

    # ## 2. Data Inspection
    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({100*c/len(labels):.1f}%)")

    # Load all splits and summarize
    categories = ["ASB", "NT", "UT"]
    splits = ["train", "val", "test"]

    print("Dataset Summary:")
    print("=" * 60)
    for cat in categories:
        print(f"\n{cat}:")
        for split in splits:
            data = np.load(f"./data/processed/{cat}/{split}.npz")
            imgs, lbls = data["images"], data["labels"]
            class_dist = dict(zip(*np.unique(lbls, return_counts=True)))
            print(f"  {split:5s}: {len(lbls):4d} samples | Class 0: {class_dist.get(0, 0):4d}, Class 1: {class_dist.get(1, 0):4d}")

def show_samples(images, labels, n_samples=8, title="Sample Images"):
    """Display a grid of sample images."""
    fig, axes = plt.subplots(2, n_samples//2, figsize=(12, 5))
    axes = axes.flatten()
    
    # Get random indices
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    for ax, idx in zip(axes, indices):
        # Remove channel dimension for display: (1, 128, 128) -> (128, 128)
        img = images[idx, 0]
        label = labels[idx]
        
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    # Load and visualize samples from each category
    for cat in ["ASB", "NT", "UT"]:
        data = np.load(f"./data/processed/{cat}/train.npz")
        show_samples(data["images"], data["labels"], title=f"{cat} Training Samples")

def load_dataset(category: str, split: str) -> TensorDataset:
    """Load a dataset split as a PyTorch TensorDataset."""
    data = np.load(f"./data/processed/{category}/{split}.npz")
    images = torch.from_numpy(data["images"])
    labels = torch.from_numpy(data["labels"])
    return TensorDataset(images, labels)

class ImageClassificationDataset(Dataset):
    """Custom PyTorch Dataset for image classification."""
    
    def __init__(
        self, 
        category: str, 
        split: str, 
        data_dir: str = ROOT_DIR/"data"/"processed",
        transform: Optional[Callable] = None
    ):
        """
        Args:
            category: One of 'ASB', 'NT', 'UT'
            split: One of 'train', 'val', 'test'
            data_dir: Path to the processed data directory
            transform: Optional transform to apply to images
        """
        data_path = Path(data_dir) / category / f"{split}.npz"
        data = np.load(data_path)
        
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transform
        self.category = category
        self.split = split
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[idx])
        label = torch.tensor(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __repr__(self) -> str:
        return f"ImageClassificationDataset(category='{self.category}', split='{self.split}', size={len(self)})"