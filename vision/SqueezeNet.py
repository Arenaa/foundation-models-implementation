import torch
import torch.nn as nn

""" This is a simple implementation of SqueezeNet.
Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016).
SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size.
arXiv preprint arXiv:1602.07360.
"""
class Fire(nn.Module):
    def __init__(self, num_filters, squeeze_filters, expand_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, squeeze_filters, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_filters, expand_filters, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_filters)
        self.conv3 = nn.Conv2d(squeeze_filters, expand_filters, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu2(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out

class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire2 = Fire(96, 16, 64)
        self.fire3 = Fire(128, 16, 64)
        self.fire4 = Fire(128, 32, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire5 = Fire(256, 32, 128)
        self.fire6 = Fire(256, 48, 192)
        self.fire7 = Fire(384, 48, 192)
        self.fire8 = Fire(384, 64, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire9 = Fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, 10000, kernel_size=1, stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=13, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print(x.shape)

        x = self.conv1(x)
        print(x.shape)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        print(x.shape)

        x = self.fire2(x)
        print(x.shape)

        x = self.fire3(x)
        x = self.fire4(x)
        x = self.pool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.pool3(x)
        print(x.shape)

        x = self.fire9(x)
        x = self.conv2(x)
        print(x.shape)
        x = self.avgpool(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = SqueezeNet()
    print(model(x).shape)