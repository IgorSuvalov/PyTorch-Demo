import os
import torch
import torch.optim as optim
import torch.nn as nn
from src.dataset import get_mnist_loaders
from src.model import SimpleCNN

def train(batch_size=64, epochs=5, lr=1e-3, data_dir="../data", model_dir="../models"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_loaders(batch_size, data_dir)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(model_dir, exist_ok=True)


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
    train()


