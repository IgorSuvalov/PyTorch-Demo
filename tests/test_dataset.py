import shutil
import torch
from src.dataset import get_mnist_loaders
from src.dataset import get_cifar10_loaders

def test_get_mnist_loaders_shapes(tmp_path):

    train_loader, test_loader = get_mnist_loaders(batch_size=4, data_dir=str(tmp_path/"data"))

    images, labels = next(iter(train_loader))
    assert isinstance(images, torch.Tensor)
    assert images.shape == (4,1,28,28)
    assert labels.shape == (4,)

    shutil.rmtree(tmp_path/"data")



def test_get_cifar10_loaders_shapes(tmp_path):

    train_loader, test_loader = get_cifar10_loaders(batch_size=4, data_dir=str(tmp_path / "data"))

    images, labels = next(iter(train_loader))
    assert isinstance(images, torch.Tensor)
    assert images.shape == (4,3,32,32)
    assert labels.shape == (4,)

    assert int(labels.min()) >= 0
    assert int(labels.max()) < 10

    shutil.rmtree(tmp_path/"data")

