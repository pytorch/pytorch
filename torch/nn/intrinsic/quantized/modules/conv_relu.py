from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.intrinsic
import torch.nn.intrinsic.qat
import torch.nn.quantized as nnq

from torch.nn.utils import fuse_conv_bn_weights


class ConvReLU2d(nnq.Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We adopt the same interface as :class:`torch.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.nn.quantized.Conv2d

    """
    _FLOAT_MODULE = torch.nn.intrinsic.ConvReLU2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(ConvReLU2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode)

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return torch.ops.quantized.conv2d_relu(
            input, self._packed_params, self.stride, self.padding,
            self.dilation, self.groups, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedConvReLU2d'

    @classmethod
    def from_float(cls, mod):
        if type(mod) == torch.nn.intrinsic.qat.ConvBnReLU2d:
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight, mod.bias, mod.running_mean, mod.running_var,
                mod.eps, mod.gamma, mod.beta)
        return super(ConvReLU2d, cls).from_float(mod)


class ConvReLU3d(nnq.Conv3d):
    r"""
    A ConvReLU3d module is a fused module of Conv3d and ReLU

    We adopt the same interface as :class:`torch.nn.quantized.Conv3d`.

    .. note::
    Attributes: Same as torch.nn.quantized.Conv3d

    """
    _FLOAT_MODULE = torch.nn.intrinsic.ConvReLU3d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(ConvReLU3d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode)

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        return torch.ops.quantized.conv3d_relu(
            input, self._packed_params, self.stride, self.padding,
            self.dilation, self.groups, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedConvReLU3d'

    @classmethod
    def from_float(cls, mod):
        # TODO: Add qat support for ConvReLU3d and ConvBnReLU3d
        return super(ConvReLU3d, cls).from_float(mod)
