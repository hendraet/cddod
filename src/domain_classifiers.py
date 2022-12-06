import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from default_config import config as cfg


class FPADomainClassifier(nn.Module):
    """Feature Pyramid Classifier"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        out = torch.sigmoid(x)
        return out


class RADomainClassifier(nn.Module):
    """Region Alignment domain classifier to classify
       region of proposal"""
    def __init__(self, in_channels: int) -> None:
        super(RADomainClassifier, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = F.dropout(F.relu(self.bn1(self.fc1(x))))
        x = F.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.dropout(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer"""
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output * -cfg.LAMBDA


def grad_reverse(x):
    gr = GradReverse.apply
    return gr(x)
