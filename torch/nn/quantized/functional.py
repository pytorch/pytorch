r""" Functional interface (quantized)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch._ops import ops
from torch._jit_internal import List
from torch.nn.modules.utils import _pair

relu = ops.quantized.relu
add_relu = ops.quantized.add_relu


def conv2d(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8):
    r"""
    conv2d(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8) -> Tensor

    Applies a 2D convolution over a quantized 2D input composed of several input
    planes.

    See :class:`~torch.nn.quantized.Conv2d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        weight: quantized filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out\_channels})`. The tensor type must be `torch.int32`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
          number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``

    Examples::

        >>> from torch.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(4, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype = torch.quint8
        >>>
        >>> q_filters = torch.quantize_linear(filters, scale, zero_point, dtype)
        >>> q_inputs = torch.quantize_linear(inputs, scale, zero_point, dtype)
        >>> q_bias = torch.quantize_linear(bias, scale, zero_point, torch.quint8)
        >>> qF.conv2d(q_inputs, q_filters, q_bias, scale, zero_point, padding=1)
    """  # noqa: E501
    if padding_mode != 'zeros':
        raise NotImplementedError("Only zero-padding is supported!")
    if input.ndim != 4:
        raise ValueError("Input shape must be `(N, C, H, W)`!")
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    prepacked_weight = ops.quantized.fbgemm_conv_prepack(weight.permute([0, 2, 3, 1]), stride, padding, dilation, groups)

    return ops.quantized.fbgemm_conv2d(input.permute([0, 2, 3, 1]),
                                       prepacked_weight, bias,
                                       stride, padding, dilation,
                                       groups, scale, zero_point).permute([0, 3, 1, 2])

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    r"""Applies a 2D max pooling over an input signal composed of several
    quantized input planes.

    See :class:`~torch.nn.quantized.MaxPool2d` for details.
    """
    if return_indices:
        raise NotImplementedError("return_indices is not yet implemented!")
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.nn.functional.max_pool2d(input, kernel_size, stride, padding,
                                          dilation, ceil_mode, return_indices)
