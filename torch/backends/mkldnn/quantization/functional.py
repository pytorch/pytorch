r""" Functional interface for MKLDNN backend (quantized)."""
r""" Aligned with torch/nn/quantized/functional.py """
from typing import List, Optional
import warnings

import torch
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple
from torch.nn.quantized.modules.utils import _pair_from_first
from torch.jit.annotations import BroadcastingList2

# Conv
def conv1d_mkldnn(input, weight, bias,
                  stride=1, padding=0, dilation=1, groups=1,
                  padding_mode='zeros',
                  scale=1.0, zero_point=0,
                  dtype=torch.quint8):
    r"""
    Applies a 1D convolution over a quantized 1D input composed of several input
    planes.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
        weight: quantized filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , iW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sW,)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padW,)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dW,)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
          number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``
    """
    stride = _pair_from_first(stride)
    padding = _pair_from_first(padding)
    dilation = _pair_from_first(dilation)

    packed_params = torch.ops.quantized.conv1d_prepack_mkldnn(
        weight, bias, stride, padding, dilation, groups)
    return torch.ops.quantized.conv1d_mkldnn(input, packed_params, scale, zero_point)

def conv1d_relu_mkldnn(input, weight, bias,
                       stride=1, padding=0, dilation=1, groups=1,
                       padding_mode='zeros',
                       scale=1.0, zero_point=0,
                       dtype=torch.quint8):
    stride = _pair_from_first(stride)
    padding = _pair_from_first(padding)
    dilation = _pair_from_first(dilation)

    packed_params = torch.ops.quantized.conv1d_prepack_mkldnn(
        weight, bias, stride, padding, dilation, groups)
    return torch.ops.quantized.conv1d_relu_mkldnn(input, packed_params, scale, zero_point)

def conv2d_mkldnn(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8):
    r"""
    Applies a 2D convolution over a quantized 2D input composed of several input
    planes.

    See :class:`~torch.nn.quantized.Conv2d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        weight: quantized filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out\_channels})`. The tensor type must be `torch.float`.
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
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    packed_params = torch.ops.quantized.conv2d_prepack_mkldnn(
        weight, bias, stride, padding, dilation, groups)
    return torch.ops.quantized.conv2d_mkldnn(input, packed_params, scale, zero_point)

def conv2d_relu_mkldnn(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8):
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    packed_params = torch.ops.quantized.conv2d_prepack_mkldnn(
        weight, bias, stride, padding, dilation, groups)
    return torch.ops.quantized.conv2d_relu_mkldnn(input, packed_params, scale, zero_point)

def conv3d_mkldnn(input, weight, bias, stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros', scale=1.0, zero_point=0, dtype=torch.quint8):
    r"""
    Applies a 3D convolution over a quantized 3D input composed of several input
    planes.

    See :class:`~torch.nn.quantized.Conv3d` for details and output shape.

    Args:
        input: quantized input tensor of shape
          :math:`(\text{minibatch} , \text{in\_channels} , iD , iH , iW)`
        weight: quantized filters of shape
          :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kD , kH , kW)`
        bias: **non-quantized** bias tensor of shape
          :math:`(\text{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padD, padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dD, dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be
          divisible by the number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for
          quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``
    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    packed_params = torch.ops.quantized.conv3d_prepack_mkldnn(
        weight, bias, stride, padding, dilation, groups)
    return torch.ops.quantized.conv3d_mkldnn(input, packed_params, scale, zero_point)

def conv3d_relu_mkldnn(input, weight, bias, stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros', scale=1.0, zero_point=0, dtype=torch.quint8):
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    packed_params = torch.ops.quantized.conv3d_prepack_mkldnn(
        weight, bias, stride, padding, dilation, groups)
    return torch.ops.quantized.conv3d_relu_mkldnn(input, packed_params, scale, zero_point)

# Linear
def linear_mkldnn(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    scale: Optional[float] = None, zero_point: Optional[int] = None
) -> Tensor:
    r"""
    Applies a linear transformation to the incoming quantized data:
    :math:`y = xA^T + b`.

    .. note::

      Current implementation packs weights on every call, which has penalty on performance.
      If you want to avoid the overhead, use :class:`~torch.nn.quantized.Linear`.

    Args:
      input (Tensor): Quantized input of type `torch.quint8`
      weight (Tensor): Quantized weight of type `torch.qint8`
      bias (Tensor): None or fp32 bias of type `torch.float`
      scale (double): output scale. If None, derived from the input scale
      zero_point (long): output zero point. If None, derived from the input zero_point

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if scale is None:
        scale = input.q_scale()
    if zero_point is None:
        zero_point = input.q_zero_point()
    _packed_params = torch.ops.quantized.linear_prepack_mkldnn(weight, bias)
    return torch.ops.quantized.linear_mkldnn(input, _packed_params, scale, zero_point)

def linear_relu_mkldnn(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    scale: Optional[float] = None, zero_point: Optional[int] = None
) -> Tensor:
    if scale is None:
        scale = input.q_scale()
    if zero_point is None:
        zero_point = input.q_zero_point()
    _packed_params = torch.ops.quantized.linear_prepack_mkldnn(weight, bias)
    return torch.ops.quantized.linear_relu_mkldnn(input, _packed_params, scale, zero_point)
