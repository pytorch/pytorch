# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import torch
from torch.nn.modules.utils import _pair
from torch.nn.quantized import functional as qF

from torch.nn.modules.conv import _ConvNd

"""Computes the output shape given convolution parameters."""
def _conv_output_shape(input_size, kernel_size, padding, stride, dilation,
                       output_padding=0):
    return np.floor((input_size + 2 * padding - kernel_size - (kernel_size - 1)
                     * (dilation - 1)) / stride) + 2 * output_padding + 1


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        if padding_mode == 'circular':
            raise NotImplementedError("Circular padding is not implemented!")
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        kernel_size = _pair(kernel_size)
        transposed = False
        output_padding = _pair(0)
        super(Conv2d, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     transposed=transposed,
                                     output_padding=output_padding,
                                     groups=groups,
                                     bias=True,
                                     padding_mode=padding_mode)
        del self.weight
        del self.bias

        qweight = torch._empty_affine_quantized(
            [out_channels, kernel_size[0], kernel_size[1],
             in_channels // self.groups],
            scale=1, zero_point=0, dtype=torch.qint8)
        qbias = torch._empty_affine_quantized([out_channels],
                                              scale=1, zero_point=0,
                                              dtype=torch.qint32)
        self.register_buffer('_packed_weight',
                             torch.ops.quantized.fbgemm_conv_prepack(qweight, self.groups))
        self.register_buffer('bias', qbias)
        self.register_buffer('_scale', torch.tensor([1.0], dtype=torch.double))
        self.register_buffer('_zero_point', torch.tensor([0], dtype=torch.long))

    @property
    def weight(self):
        return self._packed_weight

    @weight.setter
    def weight(self, w):
        self._packed_weight = torch.ops.quantized.fbgemm_conv_prepack(w, self.groups)

    @property
    def scale(self):
        return self._scale.item()

    @scale.setter
    def scale(self, s):
        if isinstance(s, torch.Tensor):
            self._scale = s
        else:
            self._scale = torch.Tensor([s])

    @property
    def zero_point(self):
        return self._zero_point.item()

    @zero_point.setter
    def zero_point(self, zp):
        if isinstance(zp, torch.Tensor):
            self._zero_point = zp
        else:
            self._zero_point = torch.Tensor([zp]).to(torch.int)

    @staticmethod
    def from_float(mod):
        assert hasattr(mod, 'observer'), "No observer in module."
        qparams = mod.observer.calculate_qparams()
        return Quantize(qparams[0].item(), qparams[1].item(),
                        mod.observer.dtype)

    def forward(self, input):
        return qF.conv2d(input=input,
                         weight=self._packed_weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=self.padding,
                         dilation=self.dilation,
                         groups=self.groups,
                         padding_mode=self.padding_mode,
                         scale=self.scale,
                         zero_point=self.zero_point,
                         dtype=torch.quint8,
                         prepacked=True)
