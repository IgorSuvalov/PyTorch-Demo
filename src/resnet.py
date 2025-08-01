import torch.nn as nn
from torchvision.models import resnet18

class ResNetClassifier(nn.Module):

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()

        self.backbone = resnet18(pretrained=False)

        self.backbone.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                                        kernel_size=7, stride=2, padding=3, bias=False)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)