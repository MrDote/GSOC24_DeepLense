from models.resnet import ResNet
from train_loops import train_loop

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms


if __name__ == '__main__':
    model = ResNet()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale()
    ])

    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    desired_classes = [0, 1, 2]


    # import numpy as np

    # indices = np.arange(len(trainset))
    # np.random.shuffle(indices)

    # targets_array = np.array(trainset.targets)
    # shuffled_targets_array = targets_array[indices]
    # shuffled_targets_list = shuffled_targets_array.tolist()

    # trainset.targets = shuffled_targets_list



    trainset = torch.utils.data.Subset(trainset, [i for i, label in enumerate(trainset.targets) if (label in desired_classes) and (i<15000)])
    testset = torch.utils.data.Subset(testset, [i for i, label in enumerate(testset.targets) if (label in desired_classes) and (i<1500)])


    print(len(trainset))
    print(len(testset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=2, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=2, drop_last=True)


    print("Datasets loaded")
    train_loop(model, trainloader, testloader, criterion, optimizer, epochs=10)