import torch
from torch.legacy import nn

class ReLU(nn.Threshold):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)

