import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.dataset import get_mnist_loaders, get_cifar10_loaders
from src.model import SimpleCNN
from src.resnet import ResNetClassifier
from src.trainf import train

def main():
    """
    Lets you choose between models and datasets
    """
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
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum ")
    args = parser.parse_args()

    if args.dataset == "mnist":
        loader_fn = get_mnist_loaders
        in_ch = 1
    else:
        loader_fn = get_cifar10_loaders
        in_ch = 3

    if args.model == "simple":
        model_cls = SimpleCNN
        model_kwargs = {"in_channels": in_ch, "img_size": (28 if args.dataset == "mnist" else 32)}
        optimizer_fn = optim.Adam
        optim_kwargs = {"lr": args.lr}
        scheduler_fn = None
        scheduler_kwargs = None
    else:
        model_cls = ResNetClassifier
        model_kwargs = {"in_channels": in_ch, "num_classes": 10}
        optimizer_fn = optim.SGD
        optim_kwargs = {"lr": args.lr, "momentum": 0.9, "weight_decay": 2e-4}
        scheduler_fn = StepLR
        scheduler_kwargs = {"step_size": 30, "gamma": 0.1}


    train(batch_size=args.batch_size, epochs=args.epochs, data_dir="../data", model_dir="../models",
          loader_fn=loader_fn, model_cls=model_cls, model_kwargs=model_kwargs, optimizer_fn=optimizer_fn,
          optim_kwargs=optim_kwargs, scheduler_fn=scheduler_fn, scheduler_kwargs=scheduler_kwargs,
          criterion_fn=nn.CrossEntropyLoss, criterion_kwargs=None)


if __name__ == "__main__":
    main()