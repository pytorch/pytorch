import torch

from .module import Module
from .. import functional as F


class CenterCrop(Module):

    def __init__(self, size):
        super(CenterCrop, self).__init__()
        self.size = size

    def __repr__(self):
        return self.__class__.__name__ + '(size=' + str(self.size) + ')'

    def forward(self, input):
        return F.center_crop(input, *self.size)
