import argparse

import torch
from torch import nn

from architectures import models
from datasets import get_dataloaders
from utils import test_loop, train_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", choices=list(models.keys()))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    # Hyperparameters
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prerequisite Code
    train_dataloader, test_dataloader = get_dataloaders(batch_size)

    model = models[args.model](num_classes=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device, batch_size)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")


if __name__ == "__main__":
    main()
