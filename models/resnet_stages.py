# models/resnet_stages.py

import torch.nn as nn
import torch.nn.functional as F
from .resnet_cifar import ResNet20

class Stage0(nn.Module):
    """
    Stage 0:
        conv1 -> bn1 -> relu -> layer1
    Output: (B, 16, 32, 32)
    """
    def __init__(self, full: ResNet20):
        super().__init__()
        self.conv1 = full.conv1
        self.bn1 = full.bn1
        self.layer1 = full.layer1

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.layer1(x)

class Stage1(nn.Module):
    """
    Stage 1:
        layer2 (contains stride=2 transition)
    Output: (B, 32, 16, 16)
    """
    def __init__(self, full: ResNet20):
        super().__init__()
        self.layer2 = full.layer2

    def forward(self, x):
        return self.layer2(x)

class Stage2(nn.Module):
    """
    Stage 2:
        layer3 -> adaptive avg pool -> fc
    Output: (B, 10)
    """
    def __init__(self, full: ResNet20):
        super().__init__()
        self.layer3 = full.layer3
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # ALWAYS safe
        self.fc = full.linear

    def forward(self, x):
        print("[Stage2 forward] input:", x.shape)
        x = self.layer3(x)
        # print("[Stage2 forward] after layer3:", x.shape)
        x = self.pool(x)
        # print("[Stage2 forward] after pool:", x.shape)
        x = x.view(x.size(0), -1)
        # print("[Stage2 forward] after view:", x.shape)
        out = self.fc(x)
        # print("[Stage2 forward] after fc:", out.shape)
        return out
