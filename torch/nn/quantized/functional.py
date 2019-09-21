r""" Functional interface (quantized)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch._jit_internal import List as _List
from torch.nn.modules.utils import _pair


def relu(input, inplace=False):
    # type: (Tensor, bool) -> Tensor
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.relu' must be quantized!")
    if inplace:
        return torch.relu_(input)
    else:
        return torch.relu(input)

def linear(input, weight, bias=None, scale=None, zero_point=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    r"""
    Applies a linear transformation to the incoming quantized data:
    :math:`y = xA^T + b`.
    See :class:`~torch.nn.Linear`

    .. note::

      Current implementation uses packed weights. This has penalty on performance.
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
    _packed_params = torch.ops.quantized.linear_prepack(weight, bias)
    return torch.ops.quantized.linear(input, _packed_params, scale,
                                      zero_point)

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

    See :class:`~torch.nn.Conv2d` for details and output shape.

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

    Examples::

        >>> from torch.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(4, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype = torch.quint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype)
        >>> qF.conv2d(q_inputs, q_filters, bias, scale, zero_point, padding=1)
    """  # noqa: E501
    if padding_mode != 'zeros':
        raise NotImplementedError("Only zero-padding is supported!")
    if input.ndim != 4:
        raise ValueError("Input shape must be `(N, C, H, W)`!")
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    prepacked_weight = torch.ops.quantized.conv_prepack(
        weight, bias, stride, padding, dilation, groups)
    return torch.ops.quantized.conv2d(input,
                                      prepacked_weight,
                                      stride, padding, dilation,
                                      groups, scale, zero_point)

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    r"""Applies a 2D max pooling over a quantized input signal composed of
    several quantized input planes.

    See :class:`~torch.nn.quantized.MaxPool2d` for details.
    """
    if return_indices:
        raise NotImplementedError("return_indices is not yet implemented!")
    if stride is None:
        stride = torch.jit.annotate(_List[int], [])
    return torch.nn.functional.max_pool2d(input, kernel_size, stride, padding,
                                          dilation, ceil_mode, return_indices)

# TODO(zaf): Add documentation
adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d
