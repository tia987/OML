import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    """
    Convolutional Neural Network.
    """

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(1 , 16, 5, padding=1) # https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)

        self.pool = nn.MaxPool2d(4, 4) # https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

        self.fc1 = nn.Linear(32 * 7 * 7, 256) # https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        self.fc2 = nn.Linear(256, 6 * 5)
        self.fc3 = nn.Linear(6 * 5, 2)

        self.leakyrelu = nn.LeakyReLU() # https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
        
    def forward(self, x):
        """Forward pass of network."""
        # Conv block 1
        x = self.leakyrelu(self.conv1(x))
        x = self.pool(x)

        # Conv block 2
        x = self.leakyrelu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        # Classifier
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.fc3(x)

        return x

    def write_weights(self, fname):
        """ Store learned weights of CNN """
        torch.save(self.state_dict(), fname)

    def load_weights(self, fname):
        """
        Load weights from file in fname.
        The evaluation server will look for a file called checkpoint.pt
        """
        ckpt = torch.load(fname, weights_only=True)
        self.load_state_dict(ckpt)

def get_criterion():
    """
    Return the loss function to use during training. We use
        the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()

def get_optimizer(network, lr=0.001, momentum=0.9, mode="Adam"):
    """
    Return the optimizer to use during training.
    Network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """
    if mode == "Adam":
        return optim.AdamW(network.parameters(), lr=lr) # https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
    else:
        return optim.SGD(network.parameters(), lr=lr, momentum=momentum)
