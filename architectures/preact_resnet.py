import torch.nn as nn


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class PreActResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_x = self._make_layer(block, 64, layers[0])
        self.conv3_x = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, layers[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def PreActResNet34(num_classes=10):
    return PreActResNet(PreActBasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def PreActResNet50(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes)
