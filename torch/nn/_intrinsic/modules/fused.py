from __future__ import absolute_import, division, print_function, unicode_literals
import torch

class ConvReLU2d(torch.nn.Sequential):
    def __init__(self, conv, relu):
        super(ConvReLU2d, self).__init__(conv, relu)

class LinearReLU(torch.nn.Sequential):
    def __init__(self, linear, relu):
        super(LinearReLU, self).__init__(linear, relu)

class ConvBn2d(torch.nn.Sequential):
    def __init__(self, conv, bn):
        super(ConvBn2d, self).__init__(conv, bn)

class ConvBnReLU2d(torch.nn.Sequential):
    def __init__(self, conv, bn, relu):
        super(ConvBnReLU2d, self).__init__(linear, bn, relu)
