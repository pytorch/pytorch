# coding=utf-8
r"""Quantized convolution modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn._intrinsic as nni
import torch.nn._intrinsic.qat as nniqat
from torch.nn.utils import fuse_conv_bn_weights
from torch._ops import ops
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> # quantize input to qint8
        >>> q_input = torch.quantize_linear(input, scale=1.0, zero_point=0, dtype=torch.qint32)
        >>> output = m(input)

    """
    _FLOAT_MODULE = nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        if padding_mode != 'zeros':
            raise NotImplementedError(
                "Currently only zero-padding is supported!")
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
        self.register_buffer('_packed_weight', torch.ops.quantized.fbgemm_conv_prepack(qweight.permute([0, 2, 3, 1]),
                             self.stride, self.padding, self.dilation, self.groups))
        self.register_buffer('bias', qbias)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.double))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.long))

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
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        output = ops.quantized.fbgemm_conv2d(input.permute([0, 2, 3, 1]),
                                             self._packed_weight, self.bias,
                                             self.stride, self.padding,
                                             self.dilation, self.groups,
                                             float(self.scale), int(self.zero_point))
        return output.permute([0, 3, 1, 2])


    @classmethod
    def from_float(cls, mod):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by the user
        """
        if hasattr(mod, 'weight_fake_quant'):
            # assert type(mod) == cls.__QAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
            #     cls.__QAT_MODULE.__name__
            if type(mod) == nniqat.ConvBn2d:
                mod.weight, mod.bias = \
                    fuse_conv_bn_weights(mod.weight, mod.bias, mod.running_mean,
                                         mod.running_var, mod.eps, mod.gamma, mod.beta)
            assert hasattr(mod, 'observer'), 'Input QAT module must have observer attached'
            weight_observer = mod.weight_fake_quant
        else:
            assert type(mod) == cls._FLOAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
                cls._FLOAT_MODULE.__name__
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            weight_observer = mod.qconfig.weight()
            # workaround for sequential, ConvReLU2d should probably
            # inherit from Conv2d instead
            if type(mod) == nni.ConvReLU2d:
                mod = mod[0]
            weight_observer(mod.weight)
        activation_observer = mod.observer
        act_scale, act_zp = activation_observer.calculate_qparams()
        wt_scale, wt_zp = weight_observer.calculate_qparams()
        bias_scale = (wt_scale * act_scale).float()
        qweight = torch.quantize_linear(
            mod.weight.float().permute([0, 2, 3, 1]).contiguous(),
            wt_scale, wt_zp.long().item(), torch.qint8)
        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    mod.stride, mod.padding, mod.dilation, mod.groups,
                    mod.bias is not None, mod.padding_mode)
        qconv._packed_weight = torch.ops.quantized.fbgemm_conv_prepack(qweight,
                                                                       qconv.stride,
                                                                       qconv.padding,
                                                                       qconv.dilation,
                                                                       qconv.groups)
        if mod.bias is not None:
            qconv.bias = torch.quantize_linear(mod.bias, bias_scale, 0, torch.qint32)
        else:
            qconv.bias = None
        qconv.scale = torch.tensor([act_scale], dtype=torch.double)
        qconv.zero_point = torch.tensor([act_zp], dtype=torch.long)
        return qconv
