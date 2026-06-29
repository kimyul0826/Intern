import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlockB(nn.Module):
    def __init__(self, num_layer, in_channels, growth_rate=12):
        super(DenseBlockB, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            ith_in_channels = in_channels + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(ith_in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(ith_in_channels, growth_rate * 4, kernel_size=1, stride=1, bias=False),

                nn.BatchNorm2d(growth_rate * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            ))

    def forward(self, x):
        feature = [x]
        for layer in self.layers:
            feature.append(layer(torch.cat(feature, dim=1)))

        return torch.cat(feature, dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layer, in_channels, growth_rate=12):
        super(DenseBlock,self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            ith_in_channels = in_channels + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(ith_in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(ith_in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            ))
    
    def forward(self, x):
        feature = [x]

        for layer in self.layers:
            feature.append(layer(torch.cat(feature, dim=1)))
        
        return torch.cat(feature, dim=1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding=0, bias=False
            ),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)




class DenseNet(nn.Module):
    def __init__(self, depth=40, growth_rate=12, num_classes=10, bottleneck=False, compression=1.0):
        super(DenseNet, self).__init__()

        if bottleneck:
            num_layer = (depth - 4) // 6
            block = DenseBlockB
            init_channels = 2 * growth_rate
        else:
            num_layer = (depth - 4) // 3
            block = DenseBlock
            init_channels = 16

        self.default = nn.Conv2d(3, init_channels, kernel_size=3, stride=1, padding=1, bias=False)

        channels = init_channels

        self.block1 = block(num_layer, channels, growth_rate)
        channels = channels + num_layer * growth_rate

        out_channels = int(channels * compression)
        self.trans1 = TransitionLayer(channels, out_channels)
        channels = out_channels

        self.block2 = block(num_layer, channels, growth_rate)
        channels = channels + num_layer * growth_rate

        out_channels = int(channels * compression)
        self.trans2 = TransitionLayer(channels, out_channels)
        channels = out_channels

        self.block3 = block(num_layer, channels, growth_rate)
        channels = channels + num_layer * growth_rate

        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.default(x)

        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)

        x = self.block3(x)

        x = F.avg_pool2d(x, kernel_size=8)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

def DenseNet40(num_classes=10):
    return DenseNet(depth=40, growth_rate=12, num_classes=num_classes)

def DenseNet100(num_classes=10):
    return DenseNet(depth=100, growth_rate=12, num_classes=num_classes)

def DenseNet100_k24(num_classes=10):
    return DenseNet(depth=100, growth_rate=24, num_classes=num_classes)

def DenseNetBC100(num_classes=10):
    return DenseNet(depth=100, growth_rate=12, num_classes=num_classes, bottleneck=True, compression=0.5)

def DenseNetBC250(num_classes=10):
    return DenseNet(depth=250, growth_rate=24, num_classes=num_classes, bottleneck=True, compression=0.5)

def DenseNetBC190(num_classes=10):
    return DenseNet(depth=190, growth_rate=40, num_classes=num_classes, bottleneck=True, compression=0.5)