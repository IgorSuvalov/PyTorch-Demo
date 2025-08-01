import torch
from src.resnet import ResNetClassifier

def test_resnet_cifdar10_shape():
    # Simple test to check the output shape
    model = ResNetClassifier(num_classes=10, in_channels=3)
    test = torch.randn(4, 3, 32, 32)
    out = model(test)
    assert out.shape == (4, 10)

def test_resnet_mnist_shape():
    # Simple test to check the output shape
    model = ResNetClassifier(num_classes=10, in_channels=1)
    test = torch.randn(3, 1, 28, 28)
    out = model(test)
    assert out.shape == (3, 10)