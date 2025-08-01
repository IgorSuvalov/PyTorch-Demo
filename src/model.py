import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SimpleCNN(nn.Module):
    """
    A simple 3-layer CNN
    """
    def __init__(self, in_channels=1, img_size=28):
        super(SimpleCNN, self).__init__()
        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels=16, kernel_size=3, padding=1)
        # Second conv layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Third conv layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Feature‑map
        fm_size = img_size // 8
        self.fc1 = nn.Linear(64 * fm_size * fm_size, 128) # 64*3*3 to 128
        self.fc2 = nn.Linear(128, 10) # 128 to 10

    def forward(self, x):
        """
        Forward pass
        """
        # x shape: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))       # (16, 28, 28)
        x = F.max_pool2d(x, 2)      # (16, 14, 14)
        x = F.relu(self.conv2(x))       # (32, 14, 14)
        x = F.max_pool2d(x, 2)      # (32, 7, 7)
        x = F.relu(self.conv3(x))        # (64,7,7)
        x = F.max_pool2d(x,2)       # (64,3,3)
        x = x.view(x.size(0), -1)        # (batch, 64*3*3)
        x = F.relu(self.fc1(x))          # (batch, 128)
        x = self.fc2(x)                  # (batch, 10)
        return x

