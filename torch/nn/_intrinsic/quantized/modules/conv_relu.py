from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn.quantized import Conv2d
from torch.nn._intrinsic import ConvReLU2d as NNConvReLU2d
import torch

class ConvReLU2d(Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We adopt the same interface as :class:`torch.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.nn.quantized.Conv2d

    """
    __FLOAT_MODULE = NNConvReLU2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(ConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        if input.ndim != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        output = torch.ops.quantized.fbgemm_conv2d_relu(input.permute([0, 2, 3, 1]),
                                                        self._packed_weight, self.bias,
                                                        self.stride, self.padding,
                                                        self.dilation, self.groups,
                                                        self.scale, self.zero_point)
        return output.permute([0, 3, 1, 2])
