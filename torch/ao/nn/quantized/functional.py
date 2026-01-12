# mypy: allow-untyped-defs
r"""Functional interface (quantized)."""

import warnings

import torch
from torch import Tensor
from torch.jit.annotations import BroadcastingList2
from torch.nn.modules.utils import _pair, _triple
from .modules.utils import _pair_from_first


# Although some of the functions and docstrings are mirrored from the torch.nn,
# we want to have them here for future changes.

__all__ = [
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "conv1d",
    "conv2d",
    "conv3d",
    "interpolate",
    "linear",
    "max_pool1d",
    "max_pool2d",
    "celu",
    "leaky_relu",
    "hardtanh",
    "hardswish",
    "threshold",
    "elu",
    "hardsigmoid",
    "clamp",
    "upsample",
    "upsample_bilinear",
    "upsample_nearest",
]


def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    r"""
    Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size
    :math:`sH \times sW` steps. The number of output features is equal to the number of
    input planes.

    .. note:: The input quantization parameters propagate to the output.

    See :class:`~torch.ao.nn.quantized.AvgPool2d` for details and output shape.

    Args:
        input: quantized input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
          tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
          tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise
             size of the pooling region will be used. Default: None
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.avg_pool2d' must be quantized!")
    return torch.nn.functional.avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def avg_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    r"""
    Applies 3D average-pooling operation in :math:`kD \ times kH \times kW` regions by step size
    :math:`sD \times sH \times sW` steps. The number of output features is equal to the number of
    input planes.

    .. note:: The input quantization parameters propagate to the output.

    Args:
        input: quantized input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
          tuple `(kD, kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple `(padD, padH, padW)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise
             size of the pooling region will be used. Default: None
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.avg_pool3d' must be quantized!")
    return torch.nn.functional.avg_pool3d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    r"""
    Applies a 2D adaptive average pooling over a quantized input signal composed
    of several quantized input planes.

    .. note:: The input quantization parameters propagate to the output.

    See :class:`~torch.ao.nn.quantized.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
                     double-integer tuple)
    """
    if not input.is_quantized:
        raise ValueError(
            "Input to 'quantized.functional.adaptive_avg_pool2d' must be quantized!"
        )
    return torch.nn.functional.adaptive_avg_pool2d(input, output_size)


def adaptive_avg_pool3d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    r"""
    Applies a 3D adaptive average pooling over a quantized input signal composed
    of several quantized input planes.

    .. note:: The input quantization parameters propagate to the output.

    See :class:`~torch.ao.nn.quantized.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
                     double-integer tuple)
    """
    if not input.is_quantized:
        raise ValueError(
            "Input to 'quantized.functional.adaptive_avg_pool3d' must be quantized!"
        )
    return torch.nn.functional.adaptive_avg_pool3d(input, output_size)


def conv1d(
    input,
    weight,
    bias,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    scale=1.0,
    zero_point=0,
    dtype=torch.quint8,
):
    r"""
    Applies a 1D convolution over a quantized 1D input composed of several input
    planes.

    See :class:`~torch.ao.nn.quantized.Conv1d` for details and output shape.

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

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> from torch.ao.nn.quantized import functional as qF
        >>> filters = torch.randn(33, 16, 3, dtype=torch.float)
        >>> inputs = torch.randn(20, 16, 50, dtype=torch.float)
        >>> bias = torch.randn(33, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv1d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """  # noqa: E501
    if padding_mode != "zeros":
        raise NotImplementedError("Only zero-padding is supported!")
    if input.dtype != torch.quint8:
        raise NotImplementedError(
            "Only torch.quint8 is supported for activation tensor!"
        )
    if weight.dtype != torch.qint8:
        raise NotImplementedError("Only torch.qint8 is supported for weight tensor!")
    if input.ndim != 3:
        raise ValueError("Input shape must be `(N, C, L)`!")
    stride = _pair_from_first(stride)
    padding = _pair_from_first(padding)
    dilation = _pair_from_first(dilation)

    packed_params = torch.ops.quantized.conv1d_prepack(
        weight, bias, stride, padding, dilation, groups
    )
    return torch.ops.quantized.conv1d(input, packed_params, scale, zero_point)


def conv2d(
    input,
    weight,
    bias,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    scale=1.0,
    zero_point=0,
    dtype=torch.quint8,
):
    r"""
    Applies a 2D convolution over a quantized 2D input composed of several input
    planes.

    See :class:`~torch.ao.nn.quantized.Conv2d` for details and output shape.

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

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> from torch.ao.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(8, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv2d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """  # noqa: E501
    if padding_mode != "zeros":
        raise NotImplementedError("Only zero-padding is supported!")
    if input.dtype != torch.quint8:
        raise NotImplementedError(
            "Only torch.quint8 is supported for activation tensor!"
        )
    if weight.dtype != torch.qint8:
        raise NotImplementedError("Only torch.qint8 is supported for weight tensor!")
    if input.ndim != 4:
        raise ValueError("Input shape must be `(N, C, H, W)`!")
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    packed_params = torch.ops.quantized.conv2d_prepack(
        weight, bias, stride, padding, dilation, groups
    )
    return torch.ops.quantized.conv2d(input, packed_params, scale, zero_point)


def conv3d(
    input,
    weight,
    bias,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    scale=1.0,
    zero_point=0,
    dtype=torch.quint8,
):
    r"""
    Applies a 3D convolution over a quantized 3D input composed of several input
    planes.

    See :class:`~torch.ao.nn.quantized.Conv3d` for details and output shape.

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

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> from torch.ao.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(8, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv3d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """  # noqa: E501
    if padding_mode != "zeros":
        raise NotImplementedError("Only zero-padding is supported!")
    if input.dtype != torch.quint8:
        raise NotImplementedError(
            "Only torch.quint8 is supported for activation tensor!"
        )
    if weight.dtype != torch.qint8:
        raise NotImplementedError("Only torch.qint8 is supported for weight tensor!")
    if input.ndim != 5:
        raise ValueError("Input shape must be `(N, C, D, H, W)`!")
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    packed_params = torch.ops.quantized.conv3d_prepack(
        weight, bias, stride, padding, dilation, groups
    )
    return torch.ops.quantized.conv3d(input, packed_params, scale, zero_point)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    r"""Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    See :func:`torch.nn.functional.interpolate` for implementation details.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D/3D input is supported for quantized inputs

    .. note:: Only the following modes are supported for the quantized inputs:

        - `bilinear`
        - `nearest`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'bilinear'``.
            Default: ``False``
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.interpolate' must be quantized!")
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    scale: float | None = None,
    zero_point: int | None = None,
) -> Tensor:
    r"""
    Applies a linear transformation to the incoming quantized data:
    :math:`y = xA^T + b`.
    See :class:`~torch.ao.nn.quantized.Linear`

    .. note::

      Current implementation packs weights on every call, which has penalty on performance.
      If you want to avoid the overhead, use :class:`~torch.ao.nn.quantized.Linear`.

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
    return torch.ops.quantized.linear(input, _packed_params, scale, zero_point)


def max_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    r"""Applies a 1D max pooling over a quantized input signal composed of
    several quantized input planes.

    .. note:: The input quantization parameters are propagated to the output.

    See :class:`~torch.ao.nn.quantized.MaxPool1d` for details.
    """
    if return_indices:
        raise NotImplementedError("return_indices is not yet implemented!")
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch.nn.functional.max_pool1d(
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


def max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    r"""Applies a 2D max pooling over a quantized input signal composed of
    several quantized input planes.

    .. note:: The input quantization parameters are propagated to the output.

    See :class:`~torch.ao.nn.quantized.MaxPool2d` for details.
    """
    if return_indices:
        raise NotImplementedError("return_indices is not yet implemented!")
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch.nn.functional.max_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


def celu(input: Tensor, scale: float, zero_point: int, alpha: float = 1.0) -> Tensor:
    r"""celu(input, scale, zero_point, alpha=1.) -> Tensor

    Applies the quantized CELU function element-wise.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x / \alpha) - 1))

    Args:
        input: quantized input
        alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.celu' must be quantized!")
    return torch.ops.quantized.celu(input, scale, zero_point, alpha)


def leaky_relu(
    input: Tensor,
    negative_slope: float = 0.01,
    inplace: bool = False,
    scale: float | None = None,
    zero_point: int | None = None,
):
    r"""
    Quantized version of the.
    leaky_relu(input, negative_slope=0.01, inplace=False, scale, zero_point) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`

    Args:
        input: Quantized input
        negative_slope: The slope of the negative input
        inplace: Inplace modification of the input tensor
        scale, zero_point: Scale and zero point of the output tensor.

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    if scale is not None and zero_point is not None:
        if inplace:
            raise AssertionError("Cannot rescale with `inplace`")
        output = torch._empty_affine_quantized(
            input.shape, scale=scale, zero_point=int(zero_point), dtype=input.dtype
        )
        torch._C._nn.leaky_relu(input, negative_slope, out=output)
        return output
    if inplace:
        result = torch._C._nn.leaky_relu_(input, negative_slope)
    else:
        result = torch._C._nn.leaky_relu(input, negative_slope)
    return result


def hardtanh(
    input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False
) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.hardtanh`."""
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardtanh' must be quantized!")
    if inplace:
        return torch._C._nn.hardtanh_(input, min_val, max_val)
    return torch._C._nn.hardtanh(input, min_val, max_val)


def hardswish(input: Tensor, scale: float, zero_point: int) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.hardswish`.

    Args:
        input: quantized input
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardswish' must be quantized!")
    return torch._ops.ops.quantized.hardswish(input, scale, zero_point)


def threshold(input: Tensor, threshold: float, value: float) -> Tensor:
    r"""Applies the quantized version of the threshold function element-wise:

    .. math::
        x = \begin{cases}
                x & \text{if~} x > \text{threshold} \\
                \text{value} & \text{otherwise}
            \end{cases}

    See :class:`~torch.nn.Threshold` for more details.
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.threshold' must be quantized!")
    if threshold is None:
        raise ValueError("Input to 'threshold' must be specified!")
    if value is None:
        raise ValueError("Input to 'value' must be specified!")
    return torch._ops.ops.quantized.threshold(input, threshold, value)


def elu(input: Tensor, scale: float, zero_point: int, alpha: float = 1.0) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.elu`.

    Args:
        input: quantized input
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.elu' must be quantized!")
    return torch.ops.quantized.elu(input, scale, zero_point, alpha)


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.hardsigmoid`."""
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardsigmoid' must be quantized!")
    if inplace:
        return torch._C._nn.hardsigmoid_(input)  # type: ignore[attr-defined]
    return torch._C._nn.hardsigmoid(input)


def clamp(input: Tensor, min_: float, max_: float) -> Tensor:
    r"""float(input, min\_, max\_) -> Tensor

    Applies the clamp function element-wise.
    See :class:`~torch.ao.nn.quantized.clamp` for more details.

    Args:
        input: quantized input
        min_: minimum value for clamping
        max_: maximum value for clamping
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.clamp' must be quantized!")
    return torch.clamp(input, min_, max_)


def upsample(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    r"""Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with ``nn.quantized.functional.interpolate(...)``.

    See :func:`torch.nn.functional.interpolate` for implementation details.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D input is supported for quantized inputs

    .. note:: Only the following modes are supported for the quantized inputs:

        - `bilinear`
        - `nearest`

    Args:
        input (Tensor): quantized input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to be an integer.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'bilinear'``.
            Default: ``False``

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`bilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.
    """
    warnings.warn(
        "nn.quantized.functional.upsample is deprecated. Use nn.quantized.functional.interpolate instead.",
        stacklevel=2,
    )
    return interpolate(input, size, scale_factor, mode, align_corners)


def upsample_bilinear(input, size=None, scale_factor=None):
    r"""Upsamples the input, using bilinear upsampling.

    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with
        ``nn.quantized.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D inputs are supported

    Args:
        input (Tensor): quantized input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int or Tuple[int, int]): multiplier for spatial size
    """
    # DeprecationWarning is ignored by default
    warnings.warn(
        "nn.quantized.functional.upsample_bilinear is deprecated. Use nn.quantized.functional.interpolate instead.",
        stacklevel=2,
    )
    return interpolate(input, size, scale_factor, mode="bilinear", align_corners=True)


def upsample_nearest(input, size=None, scale_factor=None):
    r"""Upsamples the input, using nearest neighbours' pixel values.

    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with ``nn.quantized.functional.interpolate(..., mode='nearest')``.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D inputs are supported

    Args:
        input (Tensor): quantized input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatial
            size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
    """
    # DeprecationWarning is ignored by default
    warnings.warn(
        "nn.quantized.functional.upsample_nearest is deprecated. Use nn.quantized.functional.interpolate instead.",
        stacklevel=2,
    )
    return interpolate(input, size, scale_factor, mode="nearest")
