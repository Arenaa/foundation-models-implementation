import torch
import torch.nn as nn
from collections import OrderedDict

"""This is a simple implementation of MobileNet(V1)
Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017).
Mobilenets: Efficient convolutional neural networks for mobile vision applications.
arXiv preprint arXiv:1704.04861.
"""
class Depthwise_conv(nn.Module):
    def __init__(self, in_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class pointwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Depthwise_seperable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = Depthwise_conv(in_channels, stride=stride)
        self.pw = pointwise_conv(in_channels, out_channels)

    def forward(self, x):
        x = self.pw(self.dw(x))
        return x

class MobileNet(nn.Module):
    def __init__(self, in_channels=3, num_filter=32, num_classes=1000):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filter, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )
        self.in_channels = num_filter

        self.nlayer_filter = [
            num_filter * 2,
            [num_filter * pow(2,2)],
            num_filter * pow(2,2),
            [num_filter * pow(2, 3)],
            num_filter * pow(2, 3),
            [num_filter * pow(2, 4)],
            [5, num_filter * pow(2, 4)],
            [num_filter * pow(2, 5)],
            num_filter * pow(2, 5)
        ]

        self.layers = self.create_layer()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def create_layer(self):
        block = OrderedDict()
        index = 1
        for l in self.nlayer_filter:
            if type(l) == list:
                if len(l) == 2:
                    for _ in range(l[0]):
                        block[str(index)] = Depthwise_seperable_conv(self.in_channels, l[1])
                        index +=1
                else:
                    block[str(index)] = Depthwise_seperable_conv(self.in_channels, l[0], stride=2)
                    self.in_channels = l[0]
                    index += 1
            else:
                block[str(index)] = Depthwise_seperable_conv(self.in_channels, l)
                self.in_channels = l
                index += 1
        return nn.Sequential(block)

if __name__ == "__main__":
    x = torch.randn((2, 3, 224, 224))
    model = MobileNet()
    print(model(x).shape)
