import torch
from .Module import Module
from .BatchNormalization import BatchNormalization


class VolumetricBatchNormalization(BatchNormalization):
    nDim = 5
