from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from .conv_relu import ConvReLU2d
from .linear_relu import LinearReLU

class ConvBn2d(torch.nn.Sequential):
    def __init__(self, conv, bn):
        super(ConvBn2d, self).__init__(conv, bn)

class ConvBnReLU2d(torch.nn.Sequential):
    def __init__(self, conv, bn, relu):
        super(ConvBnReLU2d, self).__init__(linear, bn, relu)


__all__ = [
    'ConvReLU2d', 'LinearReLU', 'ConvBn2d', 'ConvBnReLU2d'
]
