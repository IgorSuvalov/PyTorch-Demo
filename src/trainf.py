import os
import torch
import torch.nn as nn

def train(batch_size=64, epochs=5, data_dir="../data", model_dir="../models",
          loader_fn=None, model_cls=None, model_kwargs=None, optimizer_fn=torch.optim.Adam,
          optim_kwargs=None, scheduler_fn=None, scheduler_kwargs=None,
          criterion_fn=nn.CrossEntropyLoss, criterion_kwargs=None):
    """
    Training loop
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = loader_fn(batch_size, data_dir)
    model = model_cls(**(model_kwargs or {})).to(device)

    criterion = criterion_fn(**(criterion_kwargs or {}))
    optimizer = optimizer_fn(model.parameters(), **(optim_kwargs or {}))
    scheduler = (scheduler_fn(optimizer, **(scheduler_kwargs or {}))
                 if scheduler_fn
                 else None)

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

        if scheduler:
            scheduler.step()

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
    """
    Evaluates the model
    """
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
