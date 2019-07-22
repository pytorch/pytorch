# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.nn.modules.utils import _pair
from torch.nn.quantized import functional as qF

from torch.nn.modules.conv import _ConvNd
from torch.nn import Conv2d as NNConv2d
# from torch.nn.qat import Conv2d as QATConv2d


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
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.double))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.long))

    @property
    def weight(self):
        return self._packed_weight

    @weight.setter
    def weight(self, w):
        self._packed_weight = torch.ops.quantized.fbgemm_conv_prepack(w, self.groups)

    def forward(self, input):
        return qF.conv2d(input=input.permute([0, 2, 3, 1]).contiguous(),
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
                         prepacked=True).permute([0, 3, 1, 2]).contiguous()


    @staticmethod
    def from_float(mod):
        r"""Create a quantized module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        if hasattr(mod, 'weight_fake_quant'):
            # assert type(mod) == QATConv2d, 'nnq.Conv2d.from_float only works for nn.Conv2d or nn.qat.Conv2d'
            assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
            weight_observer = mod.weight_fake_quant
        else:
            assert type(mod) == NNConv2d, 'nnq.Conv2d.from_float only works for nn.Conv2d or nn.qat.Conv2d'
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
            weight_observer = mod.qconfig.weight()
            weight_observer(mod.weight)
        activation_observer = mod.observer
        act_scale, act_zp = activation_observer.calculate_qparams()
        wt_scale, wt_zp = weight_observer.calculate_qparams()
        bias_scale = (wt_scale * act_scale).float()
        qweight = torch.quantize_linear(
            mod.weight.float().permute([0, 2, 3, 1]).contiguous(),
            wt_scale, wt_zp.long().item(), torch.qint8)
        qbias = torch.quantize_linear(mod.bias.float(), bias_scale, 0, torch.qint32)
        qconv = Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size,
                       mod.stride, mod.padding, mod.dilation, mod.groups,
                       mod.bias is not None, mod.padding_mode)
        qconv._packed_weight = torch.ops.quantized.fbgemm_conv_prepack(qweight, qconv.groups)
        qconv.bias = qbias
        qconv.scale = torch.tensor([act_scale], dtype=torch.double)
        qconv.zero_point = torch.tensor([act_zp], dtype=torch.long)
        return qconv
