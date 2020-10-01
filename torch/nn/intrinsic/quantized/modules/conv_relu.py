
import torch
import torch.nn.intrinsic
import torch.nn.intrinsic.qat
import torch.nn.quantized as nnq

from torch.nn.utils import fuse_conv_bn_weights

class ConvReLU1d(nnq.Conv1d):
    r"""
    A ConvReLU1d module is a fused module of Conv1d and ReLU

    We adopt the same interface as :class:`torch.nn.quantized.Conv1d`.

    Attributes:
        Same as torch.nn.quantized.Conv1d

    """
    _FLOAT_MODULE = torch.nn.intrinsic.ConvReLU1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(ConvReLU1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode)

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        return torch.ops.quantized.conv1d_relu(
            input, self._packed_params, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedConvReLU1d'

    @classmethod
    def from_float(cls, mod):
        return super(ConvReLU1d, cls).from_float(mod)

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
            input, self._packed_params, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedConvReLU2d'

    @classmethod
    def from_float(cls, mod):
        if type(mod) == torch.nn.intrinsic.qat.ConvBnReLU2d:
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                mod.bn.eps, mod.bn.weight, mod.bn.bias)
        return super(ConvReLU2d, cls).from_float(mod)


class ConvReLU3d(nnq.Conv3d):
    r"""
    A ConvReLU3d module is a fused module of Conv3d and ReLU

    We adopt the same interface as :class:`torch.nn.quantized.Conv3d`.

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
            input, self._packed_params, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedConvReLU3d'

    @classmethod
    def from_float(cls, mod):
        # TODO: Add qat support for ConvReLU3d and ConvBnReLU3d
        return super(ConvReLU3d, cls).from_float(mod)
