# conda create --name pytorch_env --file requirements.txt
# conda activate pytorch_env
# conda install pytorch torchvision torchaudio -c pytorch
# python main.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import tensorboardX


def train(model, train_dataloader, criterion, optimizer):
    running_loss = 0.0
    running_corrects = 0

    model.train()

    for data in tqdm(train_dataloader):
        # get the inputs
        inputs, labels = data

        # clear out gradients of the parameters
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()  # reevaluates the model and returns the loss

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / (len(train_dataloader) * batch_size)
    epoch_acc = running_corrects / (len(train_dataloader) * batch_size)

    return epoch_loss, epoch_acc


def validate(model, test_dataloader, criterion):
    running_loss = 0.0
    running_corrects = 0

    model.eval()  # switches layer to evaluation . eg. nn.Dropout() disabled
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            # get the inputs
            inputs, labels = data

            # forward pass
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / (len(test_dataloader) * batch_size)
        epoch_acc = running_corrects / (len(test_dataloader) * batch_size)

    return epoch_loss, epoch_acc


if __name__ == "__main__":
    torch.manual_seed(321)
    working_dir = "./data"

    pets_path_train = os.path.join(working_dir, "OxfordPets", "train")
    pets_path_test = os.path.join(working_dir, "OxfordPets", "test")

    data_transforms = {
        "Training": T.Compose(
            [
                T.Resize(256),
                T.RandomRotation(45),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "Testing": T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    pets_train = torchvision.datasets.OxfordIIITPet(
        root=pets_path_train,
        split="trainval",
        target_types="category",
        download=True,
        transform=data_transforms["Training"],
    )
    pets_test = torchvision.datasets.OxfordIIITPet(
        root=pets_path_train,
        split="test",
        target_types="category",
        download=True,
        transform=data_transforms["Testing"],
    )

    batch_size = 8

    train_dataloader = DataLoader(pets_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(pets_test, batch_size=batch_size, shuffle=True)

    train_writer = tensorboardX.SummaryWriter()
    val_writer = tensorboardX.SummaryWriter()

    model_ft = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )  # loading a pre-trained(trained on image net) resnet18 model
    num_ftrs = model_ft.fc.in_features  # number of features
    model_ft.fc = nn.Linear(num_ftrs, 120)

    # checkpoint = torch.load('drive/point_resnet_best_dogs.pth')
    # model_ft.load_state_dict(checkpoint['model'])
    # optimizer_ft.load_state_dict(checkpoint['optim'])

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

    train_err = []
    train_acc = []
    val_err = []
    val_acc = []
    best_acc = 0
    num_epochs = 10

    for epoch in range(num_epochs):
        phase = "Training"

        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        epoch_train_err, epoch_train_acc = train(
            model_ft, train_dataloader, criterion, optimizer_ft
        )
        print(
            "{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_train_err, epoch_train_acc
            )
        )
        train_writer.add_scalar("Train/Err", epoch_train_err, epoch)
        train_writer.add_scalar("Train/Acc", epoch_train_acc, epoch)

        train_err.append(epoch_train_err)
        train_acc.append(train_acc.append)

        phase = "Validation"
        epoch_val_err, epoch_val_acc = validate(model_ft, test_dataloader, criterion)
        print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_val_err, epoch_val_acc))
        val_err.append(epoch_val_err)
        val_acc.append(epoch_val_acc)

        val_writer.add_scalar("Val/Err", epoch_val_err, epoch)
        val_writer.add_scalar("Val/Acc", epoch_val_acc, epoch)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model_ft.state_dict()
            state = {"model": model_ft.state_dict(), "optim": optimizer_ft.state_dict()}
            torch.save(state, "resnet_best_model.pth")
