from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Conv2d, Conv3d, ReLU, Linear, BatchNorm2d

class ConvReLU2d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        assert type(conv) == Conv2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super(ConvReLU2d, self).__init__(conv, relu)

class ConvReLU3d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        assert type(conv) == Conv3d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super(ConvReLU3d, self).__init__(conv, relu)

class LinearReLU(torch.nn.Sequential):
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, relu):
        assert type(linear) == Linear and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(linear), type(relu))
        super(LinearReLU, self).__init__(linear, relu)

class ConvBn2d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn):
        assert type(conv) == Conv2d and type(bn) == BatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        super(ConvBn2d, self).__init__(conv, bn)

class ConvBnReLU2d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert type(conv) == Conv2d and type(bn) == BatchNorm2d and \
            type(relu) == ReLU, 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(relu))
        super(ConvBnReLU2d, self).__init__(conv, bn, relu)
