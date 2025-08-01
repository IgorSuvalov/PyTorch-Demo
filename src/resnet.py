import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetClassifier(nn.Module):
    """
    A classifier that adapts torch's resnet model for our purposes
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        # Define the backbone
        self.backbone = resnet18(weights=None)
        # Alter the resnet
        self.backbone.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                                        kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass
        """
        return self.backbone(x)