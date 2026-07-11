import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class FractalBlock(nn.Module):
    def __init__(self, columns, in_channels, out_channels, dropout=0.0, drop_path=0.15):
        super(FractalBlock, self).__init__()
        self.columns = columns
        self.drop_path = drop_path

        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

        if columns > 1:
            self.fractal_1 = FractalBlock(columns - 1, in_channels, out_channels, dropout, drop_path)
            self.fractal_2 = FractalBlock(columns - 1, out_channels, out_channels, dropout, drop_path)

    def join(self, path1, path2):
        if self.training:
            keep1 = random.random() > self.drop_path
            keep2 = random.random() > self.drop_path
            if not keep1 and not keep2:
                if random.random() < 0.5:
                    keep1 = True
                else:
                    keep2 = True

            paths = []
            if keep1:
                paths.append(path1)
            if keep2:
                paths.append(path2)
            return torch.mean(torch.stack(paths, dim=0), dim=0)
        else:
            return torch.mean(torch.stack([path1, path2], dim=0), dim=0)

    def forward(self, x):
        if self.columns == 1:
            return self.conv(x)

        deep = self.fractal_2(self.fractal_1(x))
        shallow = self.conv(x)
        return self.join(deep, shallow)


class FractalNet(nn.Module):
    def __init__(self, columns=3, num_classes=10):
        super(FractalNet, self).__init__()

        channels = [64, 128, 256, 512, 512]
        dropouts = [0.0, 0.1, 0.2, 0.3, 0.4]

        self.layers = nn.ModuleList()

        in_channels = 3
        for i in range(len(channels)):
            self.layers.append(
                FractalBlock(
                    columns=columns,
                    in_channels=in_channels,
                    out_channels=channels[i],
                    dropout=dropouts[i],
                    drop_path=0.15,
                )
            )
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = channels[i]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def FractalNet20(num_classes=10):
    return FractalNet(columns=3, num_classes=num_classes)


def FractalNet40(num_classes=10):
    return FractalNet(columns=4, num_classes=num_classes)
