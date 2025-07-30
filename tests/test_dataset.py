import shutil
import torch
from src.dataset import get_mnist_loaders

def test_get_mnist_loaders_shapes(tmp_path):

    train_loader, test_loader = get_mnist_loaders(batch_size=4, data_dir=str(tmp_path/"data"))

    images, labels = next(iter(train_loader))
    assert isinstance(images, torch.Tensor)
    assert images.shape == (4,1,28,28)
    assert labels.shape == (4,)

    shutil.rmtree(tmp_path/"data")

