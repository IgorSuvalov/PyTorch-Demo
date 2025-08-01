import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64, data_dir="../data"):
    """
    Returns Dataloaders for MNIST
    """
    # ensure data/ exists
    os.makedirs(data_dir, exist_ok=True)

    # Create a temporary dataset to calculate the mean and std
    temp_ds = datasets.MNIST(root=data_dir, train=True, download=True,transform=transforms.ToTensor())
    temp_loader = DataLoader(temp_ds, batch_size=len(temp_ds), shuffle=False)
    all_images = next(iter(temp_loader))[0]
    mean, std = all_images.mean().item(), all_images.std().item()

    # Normalize
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])

    # Create the test and train datasets
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Wrap into DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar10_loaders(batch_size=64, data_dir="../data", augment=True):
    """
    Returns Dataloaders for CIFAR-10
    """
    cifar_dir = os.path.join(data_dir, "cifar10")
    os.makedirs(cifar_dir, exist_ok=True)

    # Create a temporary dataset to calculate the mean and std
    temp_ds = datasets.CIFAR10(root=cifar_dir, train=True, download=True,transform=transforms.ToTensor())
    temp_loader = DataLoader(temp_ds, batch_size=len(temp_ds), shuffle=False)
    all_images, _ = next(iter(temp_loader))
    mean, std = all_images.mean(dim=(0, 2, 3)), all_images.std(dim=(0, 2, 3))

    # Use random crop and horizontal flip for better training and normalize, or just normalize
    if augment:
        train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), transforms.Normalize(mean.tolist(), std.tolist())])

    else:
        train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean.tolist(), std.tolist())])

    # Create the test and train datasets
    train_ds = datasets.CIFAR10(root=cifar_dir, train=True, download=False, transform=train_tf)
    test_ds = datasets.CIFAR10(root=cifar_dir, train=False, download=False, transform=train_tf)

    # Wrap into Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Check
    print(f"Testing MNIST loader...")
    mnist_train, mnist_test = get_mnist_loaders(batch_size=1, data_dir="../data")
    print(f"Downloaded MNIST: {len(mnist_train)} train batches, {len(mnist_test)} test batches")

    print(f"Testing CIFAR-10 loader...")
    cifar_train, cifar_test = get_cifar10_loaders(batch_size=1, data_dir="../data", augment=False)
    print(f"Downloaded CIFAR-10: {len(cifar_train)} train batches, {len(cifar_test)} test batches")
