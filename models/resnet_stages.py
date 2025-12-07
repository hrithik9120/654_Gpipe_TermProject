# models/resnet_stages.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_cifar import ResNet20

class Stage0(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.conv1 = full.conv1
        self.bn1 = full.bn1
        self.layer1 = full.layer1

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.layer1(x)


class Stage1(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.layer2 = full.layer2

    def forward(self, x):
        return self.layer2(x)


class Stage2(nn.Module):
    """ layer3 block 0 """
    def __init__(self, full):
        super().__init__()
        self.block = full.layer3[0]

    def forward(self, x):
        return self.block(x)


class Stage3(nn.Module):
    """ layer3 block 1 """
    def __init__(self, full):
        super().__init__()
        self.block = full.layer3[1]

    def forward(self, x):
        return self.block(x)


class Stage4(nn.Module):
    """ layer3 block 2 """
    def __init__(self, full):
        super().__init__()
        self.block = full.layer3[2]

    def forward(self, x):
        return self.block(x)


class Stage5(nn.Module):
    """ final pooling + fc """
    def __init__(self, full):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = full.linear

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
