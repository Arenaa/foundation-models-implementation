import torch
import torch.nn as nn


"""This is a simple implementation of VGG models.
    Simonyan, K., & Zisserman, A. (2014).
    Very deep convolutional networks for large-scale image recognition.
    arXiv preprint arXiv:1409.1556.
"""
class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                      512, 512, 512, "M", 512, 512, 512, "M"]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layer(self.VGG16)
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1),
                                     nn.ReLU()]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    # Test model with random data
    model = VGG_net(in_channels=3, num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)
