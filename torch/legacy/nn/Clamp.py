import torch
from torch.legacy import nn

class Clamp(nn.HardTanh):
    def __init__(self, min_value, max_value):
        super(Clamp, self,).__init__(min_value, max_value)
