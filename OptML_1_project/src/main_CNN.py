import torch

import numpy as np

from torch.utils.data import TensorDataset, DataLoader

# Load data
data = np.load("data/processed/ASB/train.npz")
images = torch.from_numpy (data["images"]) # (N , 1 , 128 , 128)
labels = torch.from_numpy (data["labels"]) # (N ,)

 # Create DataLoader
dataset = TensorDataset(images, labels )
loader = DataLoader(dataset, batch_size=32, shuffle=True)