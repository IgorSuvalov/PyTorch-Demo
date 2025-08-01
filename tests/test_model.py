import torch
from src.model import SimpleCNN

def test_simple_cnn_forward():
    """
    Unit test to check the output shape
    """
    model = SimpleCNN()
    test = torch.randn(2,1,28,28)
    out = model(test)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2,10)
