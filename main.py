from models.resnet import ResNet

# from scratch_resnet import ResNet, ResidualBlock

from train_loops import train_loop
from loaders.load_data_lens import Lens

import torch.optim as optim
import torch.nn as nn


if __name__ == '__main__':
    #* resnet50
    model = ResNet()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss()


    train_loader, test_loader, img_size = Lens(batch_size=1, crop_size=100, img_size=80, rotation_degrees=90, class_samples=2)()

    print("Datasets loaded")
    train_loop(model, train_loader, test_loader, criterion, optimizer, epochs=20)