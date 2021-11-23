import torch
import torch.nn as nn
import torch.nn.functional as F

class ReweightingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            ConvBlock(32, 64, 3, 1, 1),
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 256, 3, 1, 1, pool=False)
        )

    def forward(self, x):
        features = self.net(x)
        return [features] * 5


class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel, stride, pad, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_feat, out_feat, kernel, stride, pad),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU()
        ]

        if pool:
            layers.append(nn.MaxPool2d(2, 2))

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.net(x)
