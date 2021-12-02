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


class MSReweightingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(ConvBlock(3, 32, 3, 1, 1),
                                 ConvBlock(32, 64, 3, 1, 1),
                                 ConvBlock(64, 128, 3, 1, 1),
                                 ConvBlock(128, 256, 3, 1, 1))

        self.out_blocks = nn.ModuleList([
            ConvBlock(256, 256, 3, 1, 1, pool=False),
            ConvBlock(256, 256, 3, 1, 1, pool=False),
            ConvBlock(256, 256, 3, 1, 1, pool=False)
        ])
        self.head1 = ConvBlock(256, 256, 3, 1, 1, pool=False)
        self.head2 = ConvBlock(256, 256, 3, 1, 1, pool=False)

    def forward(self, x):
        features = self.trunk(x)

        y1 = self.out_blocks[0](features)
        features = self.head1(features)
        y2 = self.out_blocks[1](features)
        features = self.head2(features)
        y3 = self.out_blocks[2](features)

        return [y1, y2, y3]



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
