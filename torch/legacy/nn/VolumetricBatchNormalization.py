import torch
from torch.legacy import nn

class VolumetricBatchNormalization(nn.BatchNormalization):
    nDim = 5
