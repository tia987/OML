import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        return self.relu(x)

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels * 4)
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers) # Important: Must return the sequence

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

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

class CNN(nn.Module):
    """
    Convolutional Neural Network.
    """

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(1 , 16, 3, padding=1) # https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2) # https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

        self.fc1 = nn.Linear(128 * 8 * 8, 256) # https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        self.fc2 = nn.Linear(256, 30)
        self.fc3 = nn.Linear(30, 2)

        self.dropout = nn.Dropout(0.1)

        self.leakyrelu = nn.LeakyReLU() # https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
        
    def forward(self, x):
        """Forward pass of network."""
        # Conv block 1
        x = self.pool(self.leakyrelu(self.conv1(x)))
        # Conv block 2
        x = self.pool(self.leakyrelu(self.conv2(x)))
        # Conv block 3
        x = self.pool(self.leakyrelu(self.conv3(x)))
        # Conv block 4
        x = self.pool(self.leakyrelu(self.conv4(x)))

        # Flatten
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        # Classifier
        x = self.leakyrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.leakyrelu(self.fc2(x))
        x = self.dropout(x)
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
