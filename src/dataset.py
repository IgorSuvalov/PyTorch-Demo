import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64, data_dir="../data"):
    # ensure data/ exists
    os.makedirs(data_dir, exist_ok=True)

    # Load the data and prepare it for normalization
    temp_ds = datasets.MNIST(root=data_dir, train=True, download=True,transform=transforms.ToTensor())
    temp_loader = DataLoader(temp_ds, batch_size=len(temp_ds), shuffle=False)
    all_images = next(iter(temp_loader))[0]
    mean, std = all_images.mean().item(), all_images.std().item()

    # Transform
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))])

    train_ds = datasets.MNIST(root=data_dir,train=True,download=True,transform=transform)
    test_ds = datasets.MNIST(root=data_dir,train=False,download=True,transform=transform)

    # Wrap into DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Check
    train_loader, test_loader = get_mnist_loaders(batch_size=1, data_dir="../data")
    print(f"Downloaded MNIST: {len(train_loader)} train batches, {len(test_loader)} test batches")

