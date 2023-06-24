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