import torch
from .HardTanh import HardTanh


class Clamp(HardTanh):

    def __init__(self, min_value, max_value):
        super(Clamp, self,).__init__(min_value, max_value)
