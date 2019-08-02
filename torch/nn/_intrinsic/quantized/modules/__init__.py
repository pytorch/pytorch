from __future__ import absolute_import, division, print_function, unicode_literals

from torch.nn._intrinsic.quantized.modules.linear_relu import LinearReLU
from torch.nn._intrinsic.quantized.modules.conv_relu import ConvReLU2d

__all__ = [
    'LinearReLU',
    'ConvReLU2d',
]
