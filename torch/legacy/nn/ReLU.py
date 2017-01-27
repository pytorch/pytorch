import torch
from .Threshold import Threshold


class ReLU(Threshold):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)
