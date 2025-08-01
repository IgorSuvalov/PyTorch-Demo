import os
import argparse

import torch
import torch.optim as optim
import torch.nn as nn

from src.dataset import get_mnist_loaders
from src.dataset import get_cifar10_loaders
from src.model import SimpleCNN
from src.resnet import ResNetClassifier

def train(batch_size=64, epochs=5, lr=1e-3, data_dir="../data", model_dir="../models", loader_fn=None,
          model_cls=None, model_kwargs=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = loader_fn(batch_size, data_dir)
    model = model_cls(**(model_kwargs or {})).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    os.makedirs(model_dir, exist_ok=True)

    best_acc = 0.0
    best_path = None

    for epoch in range(1,epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_ind, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_ind % 100 == 0 or batch_ind == len(train_loader):
                avg = running_loss / batch_ind
                print(f"Epoch {epoch} [{batch_ind}/{len(train_loader)}] Loss: {avg:.4f}")

        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch} complete. Test accuracy: {test_acc:.2f}%\n")

        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f" New best model saved to {best_path}\n")
        else:
            print()


    print(f"Training complete. Best test accuracy: {best_acc:.2f}%")
    return best_path, best_acc


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predictions = torch.max(output, 1)
            correct += (labels == predictions).sum().item()
            total += labels.size(0)
    return 100 * correct/total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"],
                        default="mnist", help="Which dataset to train the model on")

    parser.add_argument("--model", choices=["simple", "resnet"],
                        default="simple", help="Which model to use")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")

    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    args = parser.parse_args()

    if args.dataset == "mnist":
        loader_fn = get_mnist_loaders
        in_ch = 1
    else:
        loader_fn = get_cifar10_loaders
        in_ch = 3

    if args.model == "simple":
        model_cls = SimpleCNN
        model_kwargs = {}
    else:
        model_cls = ResNetClassifier
        model_kwargs = {"in_channels": in_ch, "num_classes": 10}

    train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, data_dir="../data", model_dir="../models",
          loader_fn=loader_fn, model_cls=model_cls, model_kwargs=model_kwargs)