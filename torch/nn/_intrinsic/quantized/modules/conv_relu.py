from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn.quantized as nnq
import torch.nn._intrinsic as nni
import torch.nn._intrinsic.qat as nniqat
from torch.nn.utils import fuse_conv_bn_weights
import torch

class ConvReLU2d(nnq.Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We adopt the same interface as :class:`torch.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.nn.quantized.Conv2d

    """
    _FLOAT_MODULE = nni.ConvReLU2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(ConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode)

    @property
    def weight(self):
        return torch.ops.quantized.fbgemm_conv_unpack(self._packed_weight).permute([0, 3, 1, 2])

    @weight.setter
    def weight(self, w):
        self._packed_weight = torch.ops.quantized.fbgemm_conv_prepack(w.permute([0, 2, 3, 1]),
                                                                      self.stride,
                                                                      self.padding,
                                                                      self.dilation,
                                                                      self.groups)

    def forward(self, input):
        if input.ndim != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        output = torch.ops.quantized.fbgemm_conv2d_relu(input.permute([0, 2, 3, 1]),
                                                        self._packed_weight, self.bias,
                                                        self.stride, self.padding,
                                                        self.dilation, self.groups,
                                                        self.scale, self.zero_point)
        return output.permute([0, 3, 1, 2])

    @classmethod
    def from_float(cls, mod):
        if type(mod) == nniqat.ConvBnReLU2d:
            conv_w, conv_b = fuse_conv_bn_weights(mod.weight, mod.bias, mod.running_mean, mod.running_var, mod.eps, mod.gamma, mod.beta)
            mod.weight = torch.nn.Parameter(conv_w)
            mod.bias = torch.nn.Parameter(conv_b)
        return super(ConvReLU2d, cls).from_float(mod)
