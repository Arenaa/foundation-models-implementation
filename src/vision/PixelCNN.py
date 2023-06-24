import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_channels(mask_type, in_channels, out_channels, data_channels=3):

    mask_type = mask_type.upper()
    in_factor = in_channels // data_channels + 1
    out_factor = out_channels // data_channels + 1

    mask = torch.ones([data_channels, data_channels])

    if mask_type == 'A':
        mask = mask.tril(-1)
    else:
        mask = mask.tril(0)

    mask = torch.cat([mask] * in_factor, dim=1)
    mask = torch.cat([mask] * out_factor, dim=0)

    mask = mask[0:out_channels, 0:in_channels]

    return mask

class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type='A', data_channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mask_type = mask_type.upper()

        mask = torch.ones_like(self.weight)
        out_channels, in_channels, height, width = self.weight.shape

        mask[:, :, height // 2, width // 2 + (mask_type == 'B'):] = 0
        mask[:, :, height // 2+1:] = 0

        mask[:, :, height // 2, width // 2] = mask_channels(mask_type, in_channels, out_channels, data_channels)
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data = self.weight.data.mul(self.mask)
        return super().forward(x)


class ResNetBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.h = in_channels // 2
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, self.h, kernel_size=1),
            nn.BatchNorm2d(self.h),
            nn.ReLU(),
            MaskConv2d(in_channels=self.h, out_channels=self.h, kernel_size=3, padding=1, mask_type='B'),
            nn.BatchNorm2d(self.h),
            nn.ReLU(),
            nn.Conv2d(self.h, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.out(x) + x

class PixelCNN(nn.Module):
    def __init__(self, image_shape, n_layers=12, n_filters=128, n_classes=4, final_Channels=3):
        super().__init__()

        self.n_classes = n_classes
        _, self.final_channels, self.height, self.width = image_shape[0], image_shape[1], image_shape[2], image_shape[3]

        layers = [MaskConv2d(in_channels=3, out_channels=n_filters, kernel_size=7, padding=3, mask_type='A')]

        for i in range(n_layers):
            layers.append(ResNetBlock(n_filters))

        layers += [nn.ReLU()] + \
                [MaskConv2d(in_channels=n_filters, out_channels=1024, kernel_size=1, mask_type='B')] + \
                [nn.ReLU()] + \
                [MaskConv2d(in_channels=1024, out_channels=final_Channels * n_classes, kernel_size=1, mask_type='B')]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        logits = x.view([x.shape[0], self.n_classes, self.final_channels, self.height, self.width])
        out = F.softmax(logits, dim=1)
        return logits, out

    def sample_ones(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.zeros([1, self.final_channels, self.height, self.width]).float().to(device)

        for i in range(28):
            for j in range(28):
                for k in range(3):
                    _, dist = self(x)
                    x[0, k, i, j] = np.random.choice(4, p=dist[0, :, k, i , j].cpu().data.numpy())

        return x