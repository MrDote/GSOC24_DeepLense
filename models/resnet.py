import torch
import torch.nn as nn
import torchvision
from torchsummary import summary




class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18()

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model.relu = nn.ReLU(inplace=False)

        # self.model.maxpool = nn.Identity()
        # self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #* resnet34
        self.model.fc = nn.Linear(512, 3)
        # self.model.fc = nn.Linear(2048, 3)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x
    

if __name__ == '__main__':
    model = ResNet()
    summary(model, input_size=(1, 80, 80), batch_size=2)
    # summary(model, input_size=(1, 224, 224), batch_size=2)