r""" Functional interface (quantized)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch._ops import ops

def _extend_to_list(val, length=2):
    if not isinstance(val, (tuple, list)):
        val = [val] * length
    return val

relu = ops.quantized.relu
add_relu = ops.quantized.add_relu


def conv2d(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8,
           prepacked=True):
    r"""
    conv2d(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8,
           prepacked=True) -> Tensor

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
        prepacked: assume that the weights are prepacked. Default: True

    Examples::

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
    spatial_dim_len = len(input.shape) - 2  # no batches and channels
    stride = _extend_to_list(stride, spatial_dim_len)
    padding = _extend_to_list(padding, spatial_dim_len)
    dilation = _extend_to_list(dilation, spatial_dim_len)

    if not prepacked:
        weight = ops.quantized.fbgemm_conv_prepack(weight, groups)
    return ops.quantized.fbgemm_conv2d(input, weight, bias,
                                       stride, padding, dilation,
                                       groups, scale, zero_point)
