r"""Functional interface"""
from __future__ import division

import warnings
import math
import types
from operator import mul
from functools import reduce

import torch
from torch._C import _infer_size, _add_docstr
from . import _reduction as _Reduction
from . import _functions
from .modules import utils
from ._functions import vision
from ._functions.thnn.fold import Col2Im, Im2Col
from .modules.utils import _single, _pair, _triple, _list_with_default
from . import grad
from . import _VF
from .._jit_internal import weak_script


conv1d = _add_docstr(torch.conv1d, r"""
conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See :class:`~torch.nn.Conv1d` for details and output shape.

.. include:: cudnn_deterministic.rst

Args:
    input: input tensor of shape :math:`(\text{minibatch} \times \text{in\_channels} \times iW)`
    weight: filters of shape :math:`(\text{out\_channels} \times \frac{\text{in\_channels}}{\text{groups}} \times kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or
      a one-element tuple `(sW,)`. Default: 1
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a one-element tuple `(padW,)`. Default: 0
    dilation: the spacing between kernel elements. Can be a single number or
      a one-element tuple `(dW,)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
      the number of groups. Default: 1

Examples::

    >>> filters = torch.randn(33, 16, 3)
    >>> inputs = torch.randn(20, 16, 50)
    >>> F.conv1d(inputs, filters)
""")

conv2d = _add_docstr(torch.conv2d, r"""
conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 2D convolution over an input image composed of several input
planes.

See :class:`~torch.nn.Conv2d` for details and output shape.

.. include:: cudnn_deterministic.rst

Args:
    input: input tensor of shape :math:`(\text{minibatch} \times \text{in\_channels} \times iH \times iW)`
    weight: filters of shape :math:`(\text{out\_channels} \times \frac{\text{in\_channels}}{\text{groups}} \times kH \times kW)`
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sH, sW)`. Default: 1
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padH, padW)`. Default: 0
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dH, dW)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1

Examples::

    >>> # With square kernels and equal stride
    >>> filters = torch.randn(8,4,3,3)
    >>> inputs = torch.randn(1,4,5,5)
    >>> F.conv2d(inputs, filters, padding=1)
""")  # noqa: E501

conv3d = _add_docstr(torch.conv3d, r"""
conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 3D convolution over an input image composed of several input
planes.

See :class:`~torch.nn.Conv3d` for details and output shape.

.. include:: cudnn_deterministic.rst

Args:
    input: input tensor of shape :math:`(\text{minibatch} \times \text{in\_channels} \times iT \times iH \times iW)`
    weight: filters of shape :math:`(\text{out\_channels} \times \frac{\text{in\_channels}}{\text{groups}} \times kT \times kH \times kW)`
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sT, sH, sW)`. Default: 1
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padT, padH, padW)`. Default: 0
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dT, dH, dW)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
      the number of groups. Default: 1

Examples::

    >>> filters = torch.randn(33, 16, 3, 3, 3)
    >>> inputs = torch.randn(20, 16, 50, 10, 20)
    >>> F.conv3d(inputs, filters)
""")  # noqa: E501

conv_transpose1d = _add_docstr(torch.conv_transpose1d, r"""
conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

See :class:`~torch.nn.ConvTranspose1d` for details and output shape.

.. include:: cudnn_deterministic.rst

Args:
    input: input tensor of shape :math:`(\text{minibatch} \times \text{in\_channels} \times iW)`
    weight: filters of shape :math:`(\text{in\_channels} \times \frac{\text{out\_channels}}{\text{groups}} \times kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sW,)``. Default: 1
    padding: ``kernel_size - 1 - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padW,)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple ``(dW,)``. Default: 1

Examples::

    >>> inputs = torch.randn(20, 16, 50)
    >>> weights = torch.randn(16, 33, 5)
    >>> F.conv_transpose1d(inputs, weights)
""")

conv_transpose2d = _add_docstr(torch.conv_transpose2d, r"""
conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution".

See :class:`~torch.nn.ConvTranspose2d` for details and output shape.

.. include:: cudnn_deterministic.rst

Args:
    input: input tensor of shape :math:`(\text{minibatch} \times \text{in\_channels} \times iH \times iW)`
    weight: filters of shape :math:`(\text{in\_channels} \times \frac{\text{out\_channels}}{\text{groups}} \times kH \times kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sH, sW)``. Default: 1
    padding: ``kernel_size - 1 - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padH, padW)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.
      Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple ``(dH, dW)``. Default: 1

Examples::

    >>> # With square kernels and equal stride
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> weights = torch.randn(4, 8, 3, 3)
    >>> F.conv_transpose2d(inputs, weights, padding=1)
""")  # noqa: E501

conv_transpose3d = _add_docstr(torch.conv_transpose3d, r"""
conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 3D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution"

See :class:`~torch.nn.ConvTranspose3d` for details and output shape.

.. include:: cudnn_deterministic.rst

Args:
    input: input tensor of shape :math:`(\text{minibatch} \times \text{in\_channels} \times iT \times iH \times iW)`
    weight: filters of shape :math:`(\text{in\_channels} \times \frac{\text{out\_channels}}{\text{groups}} \times kT \times kH \times kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sT, sH, sW)``. Default: 1
    padding: ``kernel_size - 1 - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padT, padH, padW)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple
      ``(out_padT, out_padH, out_padW)``. Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dT, dH, dW)`. Default: 1

Examples::

    >>> inputs = torch.randn(20, 16, 50, 10, 20)
    >>> weights = torch.randn(16, 33, 3, 3, 3)
    >>> F.conv_transpose3d(inputs, weights)
""")  # noqa: E501

conv_tbc = _add_docstr(torch.conv_tbc, r"""
Applies a 1-dimensional sequence convolution over an input sequence.
Input and output dimensions are (Time, Batch, Channels) - hence TBC.

Args:
    input: input tensor of shape :math:`(\text{sequence length} \times batch \times \text{in\_channels})`
    weight: filter of shape (:math:`\text{kernel width} \times \text{in\_channels} \times \text{out\_channels}`)
    bias: bias of shape (:math:`\text{out\_channels}`)
    pad: number of timesteps to pad. Default: 0
""")


# Pooling
avg_pool1d = _add_docstr(torch.avg_pool1d, r"""
avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor

Applies a 1D average pooling over an input signal composed of several
input planes.

See :class:`~torch.nn.AvgPool1d` for details and output shape.

Args:
    input: input tensor of shape :math:`(\text{minibatch} \times \text{in\_channels} \times iW)`
    kernel_size: the size of the window. Can be a single number or a
      tuple :math:`(kW,)`
    stride: the stride of the window. Can be a single number or a tuple
      `(sW,)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padW,)`. Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` to compute the
        output shape. Default: ``False``
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation. Default: ``True``

Examples::
    >>> # pool of square window of size=3, stride=2
    >>> input = torch.tensor([[[1,2,3,4,5,6,7]]])
    >>> F.avg_pool1d(input, kernel_size=3, stride=2)
    tensor([[[ 2.,  4.,  6.]]])

""")


avg_pool2d = _add_docstr(torch._C._nn.avg_pool2d, r"""
avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor

Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size
:math:`sH \times sW` steps. The number of output features is equal to the number of
input planes.

See :class:`~torch.nn.AvgPool2d` for details and output shape.

Args:
    input: input tensor :math:`(\text{minibatch} \times \text{in\_channels} \times iH \times iW)`
    kernel_size: size of the pooling region. Can be a single number or a
      tuple :math:`(kH \times kW)`
    stride: stride of the pooling operation. Can be a single number or a
      tuple `(sH, sW)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padH, padW)`. Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` in the formula
        to compute the output shape. Default: ``False``
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation. Default: ``True``
""")

avg_pool3d = _add_docstr(torch._C._nn.avg_pool3d, r"""
avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor

Applies 3D average-pooling operation in :math:`kT \times kH \times kW` regions by step
size :math:`sT \times sH \times sW` steps. The number of output features is equal to
:math:`\lfloor\frac{\text{input planes}}{sT}\rfloor`.

See :class:`~torch.nn.AvgPool3d` for details and output shape.

Args:
    input: input tensor :math:`(\text{minibatch} \times \text{in\_channels} \times iT \times iH \times iW)`
    kernel_size: size of the pooling region. Can be a single number or a
      tuple :math:`(kT \times kH \times kW)`
    stride: stride of the pooling operation. Can be a single number or a
      tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padT, padH, padW)`, Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` in the formula
        to compute the output shape
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation
""")


def fractional_max_pool2d(input, kernel_size, output_size=None,
                          output_ratio=None, return_indices=False,
                          _random_samples=None):
    r"""Applies 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \times k`)
                     or a tuple :math:`(kH \times kW)`
        output_size: the target output size of the image of the form :math:`oH \times oW`.
                     Can be a tuple `(oH, oW)` or a single number :math:`oH` for a square image :math:`oH \times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool2d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32)
        >>> # pool of square window of size=3, and target output size 13x12
        >>> F.fractional_max_pool2d(input, 3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> F.fractional_max_pool2d(input, 3, output_ratio=(0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
    if output_size is None and output_ratio is None:
        raise ValueError("fractional_max_pool2d requires specifying either "
                         "an output_size, or a output_ratio")
    if output_size is None:
        output_ratio = _pair(output_ratio)
        output_size = (int(input.size(2) * output_ratio[0]),
                       int(input.size(3) * output_ratio[1]))

    if _random_samples is None:
        _random_samples = input.new(input.size(0), input.size(1), 2).uniform_()
    ret = torch._C._nn.fractional_max_pool2d(input, kernel_size, output_size, _random_samples)
    return ret if return_indices else ret[0]


def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    r"""Applies a 1D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool1d` for details.
    """
    ret = torch.max_pool1d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)
    return ret if return_indices else ret[0]


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool2d` for details.
    """
    ret = torch._C._nn.max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)
    return ret if return_indices else ret[0]


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    r"""Applies a 3D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool3d` for details.
    """
    ret = torch._C._nn.max_pool3d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)
    return ret if return_indices else ret[0]


def _unpool_output_size(input, kernel_size, stride, padding, output_size):
    input_size = input.size()
    default_size = []
    for d in range(len(kernel_size)):
        default_size.append((input_size[d + 2] - 1) * stride[d] +
                            kernel_size[d] - 2 * padding[d])
    if output_size is None:
        return default_size

    output_size = list(output_size)
    if len(output_size) == len(kernel_size) + 2:
        output_size = output_size[2:]
    if len(output_size) != len(kernel_size):
        raise ValueError("output_size should be a sequence containing "
                         "{} or {} elements, but it has a length of '{}'"
                         .format(len(kernel_size), len(kernel_size) + 2,
                                 len(output_size)))
    for d in range(len(kernel_size)):
        min_size = default_size[d] - stride[d]
        max_size = default_size[d] + stride[d]
        if not (min_size < output_size[d] < max_size):
            raise ValueError(
                'invalid output_size "{}" (dim {} must be between {} and {})'
                .format(output_size, d, min_size, max_size))

    return output_size


@torch._jit_internal.weak_script
def max_unpool1d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    r"""Computes a partial inverse of :class:`MaxPool1d`.

    See :class:`~torch.nn.MaxUnpool1d` for details.
    """
    kernel_size = _single(kernel_size)
    stride = _single(stride or kernel_size)
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool2d(input.unsqueeze(3), indices.unsqueeze(3), output_size + [1]).squeeze(3)


def max_unpool2d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    r"""Computes a partial inverse of :class:`MaxPool2d`.

    See :class:`~torch.nn.MaxUnpool2d` for details.
    """
    kernel_size = _pair(kernel_size)
    stride = _pair(stride or kernel_size)
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool2d(input, indices, output_size)


def max_unpool3d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    r"""Computes a partial inverse of :class:`MaxPool3d`.

    See :class:`~torch.nn.MaxUnpool3d` for details.
    """
    kernel_size = _triple(kernel_size)
    stride = _triple(stride or kernel_size)
    padding = _triple(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool3d(input, indices, output_size, stride, padding)


def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    r"""Applies a 2D power-average pooling over an input signal composed of
    several input planes. If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool2d` for details.
    """
    kw, kh = utils._pair(kernel_size)
    out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    return (torch.sign(out) * relu(torch.abs(out))).mul(kw * kh).pow(1. / norm_type)


def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    # type: (Tensor, int, int, Optional[List[int]], bool) -> Tensor
    r"""Applies a 1D power-average pooling over an input signal composed of
    several input planes. If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool1d` for details.
    """
    out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    return (torch.sign(out) * relu(torch.abs(out))).mul(kernel_size).pow(1. / norm_type)


def adaptive_max_pool1d(input, output_size, return_indices=False):
    r"""Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    ret = torch.adaptive_max_pool1d(input, output_size)
    return ret if return_indices else ret[0]


def adaptive_max_pool2d(input, output_size, return_indices=False):
    r"""Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    output_size = _list_with_default(output_size, input.size())
    ret = torch._C._nn.adaptive_max_pool2d(input, output_size)
    return ret if return_indices else ret[0]


def adaptive_max_pool3d(input, output_size, return_indices=False):
    r"""Applies a 3D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    output_size = _list_with_default(output_size, input.size())
    ret = torch._C._nn.adaptive_max_pool3d(input, output_size)
    return ret if return_indices else ret[0]


adaptive_avg_pool1d = _add_docstr(torch.adaptive_avg_pool1d, r"""
adaptive_avg_pool1d(input, output_size) -> Tensor

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.

Args:
    output_size: the target output size (single integer)
""")


@torch._jit_internal.weak_script
def adaptive_avg_pool2d(input, output_size):
    # type: (Tensor, BroadcastingList2[int]) -> Tensor
    r"""
    Applies a 2D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
    """
    _output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_avg_pool2d(input, _output_size)


@torch._jit_internal.weak_script
def adaptive_avg_pool3d(input, output_size):
    # type: (Tensor, BroadcastingList3[int]) -> Tensor
    r"""
    Applies a 3D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
    """
    _output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_avg_pool3d(input, _output_size)


# Activation functions
@torch._jit_internal.weak_script
def dropout(input, p=0.5, training=True, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Defualt: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    return (_VF.dropout_(input, p, training)
            if inplace
            else _VF.dropout(input, p, training))


@torch._jit_internal.weak_script
def alpha_dropout(input, p=0.5, training=False, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    r"""Applies alpha dropout to the input.

    See :class:`~torch.nn.AlphaDropout` for details.
    """
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    return (_VF.alpha_dropout_(input, p, training)
            if inplace
            else _VF.alpha_dropout(input, p, training))


@torch._jit_internal.weak_script
def dropout2d(input, p=0.5, training=True, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    r"""
    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call.
    with probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout2d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Defualt: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    return (_VF.feature_dropout_(input, p, training)
            if inplace
            else _VF.feature_dropout(input, p, training))


@torch._jit_internal.weak_script
def dropout3d(input, p=0.5, training=True, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    r"""
    Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call.
    with probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Defualt: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    # This is 100% the same code as dropout2d. We duplicate this code so that
    # stack traces are not confusing.
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    return (_VF.feature_dropout_(input, p, training)
            if inplace
            else _VF.feature_dropout(input, p, training))


@torch._jit_internal.weak_script
def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    return (_VF.feature_alpha_dropout_(input, p, training)
            if inplace
            else _VF.feature_alpha_dropout(input, p, training))


@torch._jit_internal.weak_script
def threshold(input, threshold, value, inplace=False):
    # type: (Tensor, float, float, bool) -> Tensor
    r"""Thresholds each element of the input Tensor.

    See :class:`~torch.nn.Threshold` for more details.
    """
    if inplace:
        result = _VF.threshold_(input, threshold, value)
    else:
        result = _VF.threshold(input, threshold, value)
    return result


threshold_ = _add_docstr(_VF.threshold_, r"""
threshold_(input, threshold, value) -> Tensor

In-place version of :func:`~threshold`.
""")


@torch._jit_internal.weak_script
def relu(input, inplace=False):
    # type: (Tensor, bool) -> Tensor
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    if inplace:
        result = torch.relu_(input)
    else:
        result = torch.relu(input)
    return result


relu_ = _add_docstr(torch.relu_, r"""
relu_(input) -> Tensor

In-place version of :func:`~relu`.
""")


@torch._jit_internal.weak_script
def glu(input, dim=-1):
    # type: (Tensor, int) -> Tensor
    r"""
    glu(input, dim=-1) -> Tensor

    The gated linear unit. Computes:

    .. math ::

        H = A \times \sigma(B)

    where `input` is split in half along `dim` to form `A` and `B`.

    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.

    Args:
        input (Tensor): input tensor
        dim (int): dimension on which to split the input
    """
    if input.dim() == 0:
        raise RuntimeError("glu does not suppport scalars because halving size must be even")
    return torch._C._nn.glu(input, dim)


@torch._jit_internal.weak_script
def hardtanh(input, min_val=-1., max_val=1., inplace=False):
    # type: (Tensor, float, float, bool) -> Tensor
    r"""
    hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor

    Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
    details.
    """
    if inplace:
        result = torch._C._nn.hardtanh_(input, min_val, max_val)
    else:
        result = torch._C._nn.hardtanh(input, min_val, max_val)
    return result


hardtanh_ = _add_docstr(torch._C._nn.hardtanh_, r"""
hardtanh_(input, min_val=-1., max_val=1.) -> Tensor

In-place version of :func:`~hardtanh`.
""")


@torch._jit_internal.weak_script
def relu6(input, inplace=False):
    # type: (Tensor, bool) -> Tensor
    r"""relu6(input, inplace=False) -> Tensor

    Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.

    See :class:`~torch.nn.ReLU6` for more details.
    """
    return hardtanh(input, 0., 6., inplace)


@torch._jit_internal.weak_script
def elu(input, alpha=1., inplace=False):
    # type: (Tensor, float, bool) -> Tensor
    r"""Applies element-wise,
    :math:`\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))`.

    See :class:`~torch.nn.ELU` for more details.
    """
    if inplace:
        result = torch._C._nn.elu_(input, alpha)
    else:
        result = torch._C._nn.elu(input, alpha)
    return result


elu_ = _add_docstr(torch._C._nn.elu_, r"""
elu_(input, alpha=1.) -> Tensor

In-place version of :func:`~elu`.
""")


@torch._jit_internal.weak_script
def selu(input, inplace=False):
    # type: (Tensor, bool) -> Tensor
    r"""selu(input, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`,
    with :math:`\alpha=1.6732632423543772848170429916717` and
    :math:`scale=1.0507009873554804934193349852946`.

    See :class:`~torch.nn.SELU` for more details.
    """
    if inplace:
        result = torch.selu_(input)
    else:
        result = torch.selu(input)
    return result


selu_ = _add_docstr(torch.selu_, r"""
selu_(input) -> Tensor

In-place version of :func:`~selu`.
""")


@torch._jit_internal.weak_script
def celu(input, alpha=1., inplace=False):
    # type: (Tensor, float, bool) -> Tensor
    r"""celu(input, alpha=1., inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))`.

    See :class:`~torch.nn.CELU` for more details.
    """
    if inplace:
        result = torch.celu_(input, alpha)
    else:
        result = torch.celu(input, alpha)
    return result

celu_ = _add_docstr(torch.celu_, r"""
celu_(input, alpha=1.) -> Tensor

In-place version of :func:`~celu`.
""")


@torch._jit_internal.weak_script
def leaky_relu(input, negative_slope=0.01, inplace=False):
    # type: (Tensor, float, bool) -> Tensor
    r"""
    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    if inplace:
        result = torch._C._nn.leaky_relu_(input, negative_slope)
    else:
        result = torch._C._nn.leaky_relu(input, negative_slope)
    return result


leaky_relu_ = _add_docstr(torch._C._nn.leaky_relu_, r"""
leaky_relu_(input, negative_slope=0.01) -> Tensor

In-place version of :func:`~leaky_relu`.
""")


@torch._jit_internal.weak_script
def prelu(input, weight):
    # type: (Tensor, Tensor) -> Tensor
    r"""prelu(input, weight) -> Tensor

    Applies element-wise the function
    :math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a
    learnable parameter.

    See :class:`~torch.nn.PReLU` for more details.
    """
    return torch.prelu(input, weight)


@torch._jit_internal.weak_script
def rrelu(input, lower=1. / 8, upper=1. / 3, training=False, inplace=False):
    # type: (Tensor, float, float, bool, bool) -> Tensor
    r"""rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor

    Randomized leaky ReLU.

    See :class:`~torch.nn.RReLU` for more details.
    """
    if inplace:
        result = torch.rrelu_(input, lower, upper, training)
    else:
        result = torch.rrelu(input, lower, upper, training)
    return result


rrelu_ = _add_docstr(torch.rrelu_, r"""
rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor

In-place version of :func:`~rrelu`.
""")

logsigmoid = _add_docstr(torch._C._nn.log_sigmoid, r"""
logsigmoid(input) -> Tensor

Applies element-wise :math:`\text{LogSigmoid}(x) = \log \left(\frac{1}{1 + \exp(-x_i)}\right)`

See :class:`~torch.nn.LogSigmoid` for more details.
""")


@torch._jit_internal.weak_script
def hardshrink(input, lambd=0.5):
    # type: (Tensor, float) -> Tensor
    r"""
    hardshrink(input, lambd=0.5) -> Tensor

    Applies the hard shrinkage function element-wise

    See :class:`~torch.nn.Hardshrink` for more details.
    """
    return torch.hardshrink(input, lambd)


@torch._jit_internal.weak_script
def tanhshrink(input):
    r"""tanhshrink(input) -> Tensor

    Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)`

    See :class:`~torch.nn.Tanhshrink` for more details.
    """
    return input - input.tanh()


@torch._jit_internal.weak_script
def softsign(input):
    r"""softsign(input) -> Tensor

    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

    See :class:`~torch.nn.Softsign` for more details.
    """
    return input / (input.abs() + 1)


softplus = _add_docstr(torch._C._nn.softplus, r"""
softplus(input, beta=1, threshold=20) -> Tensor
""")


@torch._jit_internal.weak_script
def _get_softmax_dim(name, ndim, stacklevel):
    # type: (str, int, int) -> int
    warnings.warn("Implicit dimension choice for {} has been deprecated. "
                  "Change the call to include dim=X as an argument.".format(name), stacklevel=stacklevel)
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def softmin(input, dim=None, _stacklevel=3, dtype=None):
    r"""Applies a softmin function.

    Note that :math:`\text{Softmin}(x) = \text{Softmax}(-x)`. See softmax definition for mathematical formula.

    See :class:`~torch.nn.Softmin` for more details.

    Arguments:
        input (Tensor): input
        dim (int): A dimension along which softmin will be computed (so every slice
            along dim will sum to 1).
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    """
    if dim is None:
        dim = _get_softmax_dim('softmin', input.dim(), _stacklevel)
    if dtype is None:
        return (-input).softmax(dim)
    else:
        return (-input).softmax(dim, dtype=dtype)


@torch._jit_internal.weak_script
def softmax(input, dim=None, _stacklevel=3, dtype=None):
    # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
    r"""Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Arguments:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.


    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).

    """
    if dim is None:
        dim = _get_softmax_dim('softmax', input.dim(), _stacklevel)
    else:
        dim = torch.jit._unwrap_optional(dim)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        dtype = torch.jit._unwrap_optional(dtype)
        ret = input.softmax(dim, dtype=dtype)
    return ret


def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + gumbel_noise
    return softmax(y / tau, dims - 1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    r"""
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: `[batch_size, num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd

    Returns:
      Sampled tensor of shape ``batch_size x num_features`` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across features

    Constraints:

    - Currently only work on 2D input :attr:`logits` tensor of shape ``batch_size x num_features``

    Based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y


@torch._jit_internal.weak_script
def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
    r"""Applies a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    See :class:`~torch.nn.LogSoftmax` for more details.

    Arguments:
        input (Tensor): input
        dim (int): A dimension along which log_softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    """
    if dim is None:
        dim = _get_softmax_dim('log_softmax', input.dim(), _stacklevel)
    else:
        dim = torch.jit._unwrap_optional(dim)
    if dtype is None:
        ret = input.log_softmax(dim)
    else:
        _dtype = torch.jit._unwrap_optional(dtype)
        ret = input.log_softmax(dim, dtype=_dtype)
    return ret


softshrink = _add_docstr(torch._C._nn.softshrink, r"""
softshrink(input, lambd=0.5) -> Tensor

Applies the soft shrinkage function elementwise

See :class:`~torch.nn.Softshrink` for more details.
""")


@torch._jit_internal.weak_script
def tanh(input):
    r"""tanh(input) -> Tensor

    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
    warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
    return input.tanh()


@torch._jit_internal.weak_script
def sigmoid(input):
    r"""sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    return input.sigmoid()


@torch._jit_internal.weak_script
def linear(input, weight, bias=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:

        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(torch.jit._unwrap_optional(bias), input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += torch.jit._unwrap_optional(bias)
        ret = output
    return ret


@torch._jit_internal.weak_script
def bilinear(input1, input2, weight, bias=None):
    # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tensor
    return torch.bilinear(input1, input2, weight, bias)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2,
              scale_grad_by_freq=False, sparse=False):
    r"""A simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`torch.nn.Embedding` for more details.

    Args:
        input (LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
                            where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape

    Examples::

        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([[1,2,4,5],[4,3,2,9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> F.embedding(input, embedding_matrix)
        tensor([[[ 0.8490,  0.9625,  0.6753],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.6246,  0.9751,  0.3618],
                 [ 0.4161,  0.2419,  0.7383]],

                [[ 0.6246,  0.9751,  0.3618],
                 [ 0.0237,  0.7794,  0.0528],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.3385,  0.8612,  0.1867]]])

        >>> # example with padding_idx
        >>> weights = torch.rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = weights
        >>> input = torch.tensor([[0,2,0,5]])
        >>> F.embedding(input, embedding_matrix, padding_idx=0)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.5609,  0.5384,  0.8720],
                 [ 0.0000,  0.0000,  0.0000],
                 [ 0.6262,  0.2438,  0.7471]]])
    """
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(0), 'Padding_idx must be within num_embeddings'
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(0), 'Padding_idx must be within num_embeddings'
            padding_idx = weight.size(0) + padding_idx
    elif padding_idx is None:
        padding_idx = -1
    if max_norm is not None:
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        input = input.contiguous()
        with torch.no_grad():
            torch.embedding_renorm_(weight, input, max_norm, norm_type)
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2,
                  scale_grad_by_freq=False, mode='mean', sparse=False):
    r"""Computes sums, means or maxes of 'bags' of embeddings, without instantiating the
    intermediate embeddings.

    See :class:`torch.nn.EmbeddingBag` for more details.
    .. include:: cuda_deterministic_backward.rst

    Args:
        input (LongTensor): Tensor containing bags of indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        offsets (LongTensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                             the starting index position of each bag (sequence) in :attr:`input`.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The ``p`` in the ``p``-norm to compute for the :attr:`max_norm` option.
                                     Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.
                                 Note: this option is not supported when ``mode="max"``.

    Shape:

        - :attr:`input` (LongTensor) and :attr:`offsets` (LongTensor, optional)

          - If :attr:`input` is 2D of shape ``B x N``,

            it will be treated as ``B`` bags (sequences) each of fixed length ``N``, and
            this will return ``B`` values aggregated in a way depending on the :attr:`mode`.
            :attr:`offsets` is ignored and required to be ``None`` in this case.

          - If :attr:`input` is 1D of shape ``N``,

            it will be treated as a concatenation of multiple bags (sequences).
            :attr:`offsets` is required to be a 1D tensor containing the
            starting index positions of each bag in :attr:`input`. Therefore,
            for :attr:`offsets` of shape ``B``, :attr:`input` will be viewed as
            having ``B`` bags. Empty bags (i.e., having 0-length) will have
            returned vectors filled by zeros.

        - :attr:`weight` (Tensor): the learnable weights of the module of
          shape ``(num_embeddings x embedding_dim)``

        - :attr:`output`: aggregated embedding values of shape ``B x embedding_dim``

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([1,2,4,5,4,3,2,9])
        >>> offsets = torch.tensor([0,4])
        >>> F.embedding_bag(embedding_matrix, input, offsets)
        tensor([[ 0.3397,  0.3552,  0.5545],
                [ 0.5893,  0.4386,  0.5882]])
    """
    # Check for backward compatibility.
    # Used to be embedding_bag(weight, input, ...)
    # Now is     embedding_bag(input, weight, ...)
    if weight.dtype == torch.long and input.is_floating_point():
        warnings.warn("Argument order of nn.functional.embedding_bag was changed. "
                      "Usage `embedding_bag(weight, input, ...)` is deprecated, "
                      "and should now be `embedding_bag(input, weight, ...)`.")
        weight, input = input, weight

    if input.dim() == 2:
        if offsets is not None:
            raise ValueError("if input is 2D, then offsets has to be None"
                             ", as input is treated is a mini-batch of"
                             " fixed length sequences. However, found "
                             "offsets of type {}".format(type(offsets)))
        else:
            offsets = torch.arange(0, input.numel(), input.size(1),
                                   dtype=torch.long, device=input.device)

            input = input.reshape(-1)
    elif input.dim() == 1:
        if offsets is None:
            raise ValueError("offsets has to be a 1D Tensor but got None")
        if offsets.dim() != 1:
            raise ValueError("offsets has to be a 1D Tensor")
        if offsets[0].item() != 0:
            raise ValueError("offsets[0] has to be 0, i.e., the first sequence "
                             "in the mini-batch has to start from position 0. "
                             "However, got {}".format(offsets[0].item()))
        if offsets[-1].item() > input.size(0):
            raise ValueError("offsets[-1] can not be greater than input's length"
                             " ({}), but got offsets[-1] of {}"
                             .format(input.size(0), offsets[-1].item()))
    else:
        raise ValueError("input has to be 1D or 2D Tensor,"
                         " but got Tensor of dimension {}".format(input.dim()))

    if mode == 'sum':
        mode = 0
    elif mode == 'mean':
        mode = 1
    elif mode == 'max':
        mode = 2

        if scale_grad_by_freq:
            raise ValueError("max mode does not support scaling the gradient by the frequency")

        if sparse:
            raise ValueError("max mode does not support sparse weights")

    else:
        raise ValueError("mode has to be one of sum, mean or max")

    if max_norm is not None:
        with torch.no_grad():
            torch.embedding_renorm_(weight, input, max_norm, norm_type)

    ret, _, _, _ = torch.embedding_bag(
        weight,
        input,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse)
    return ret


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    r"""Applies Batch Normalization for each channel across a batch of data.

    See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
    :class:`~torch.nn.BatchNorm3d` for details.
    """
    if training:
        size = list(input.size())
        if reduce(mul, size[2:], size[0]) == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
    return torch.batch_norm(
        input, weight, bias, running_mean, running_var,
        training, momentum, eps, torch.backends.cudnn.enabled
    )


def instance_norm(input, running_mean=None, running_var=None, weight=None,
                  bias=None, use_input_stats=True, momentum=0.1, eps=1e-5):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], bool, float, float) -> Tensor  # noqa
    r"""Applies Instance Normalization for each channel in each data sample in a
    batch.

    See :class:`~torch.nn.InstanceNorm1d`, :class:`~torch.nn.InstanceNorm2d`,
    :class:`~torch.nn.InstanceNorm3d` for details.
    """
    return torch.instance_norm(
        input, weight, bias, running_mean, running_var,
        use_input_stats, momentum, eps, torch.backends.cudnn.enabled
    )


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    r"""Applies Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
    return torch.layer_norm(input, normalized_shape, weight, bias, eps,
                            torch.backends.cudnn.enabled)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    r"""Applies Group Normalization for last certain number of dimensions.

    See :class:`~torch.nn.GroupNorm` for details.
    """
    return torch.group_norm(input, num_groups, weight, bias, eps,
                            torch.backends.cudnn.enabled)


def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.):
    # type: (Tensor, int, float, float, float) -> Tensor
    r"""Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)
    if dim == 3:
        div = pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        div = div.view(sizes)
    div = div.mul(alpha).add(k).pow(beta)
    return input / div


# loss

@torch._jit_internal.weak_script
def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
             reduction='mean'):
    # type: (Tensor, Tensor, Tensor, Tensor, int, str) -> Tensor
    r"""The Connectionist Temporal Classification loss.

    See :class:`~torch.nn.CTCLoss` for details.

    .. include:: cudnn_deterministic.rst
    .. include:: cuda_deterministic_backward.rst

    Args:
        log_probs: :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
            `T = input length`, and `N = batch size`.
            The logarithmized probabilities of the outputs
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: :math:`(N, S)` or `(sum(target_lengths))`.
            Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)`.
            Lengths of the inputs (must each be :math:`\leq T`)
        target_lengths: :math:`(N)`.
            Lengths of the targets
        blank (int, optional):
            Blank label. Default :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: 'mean'

    Example::

        >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
        >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
        >>> loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        >>> loss.backward()
    """
    return torch.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, _Reduction.get_enum(reduction))


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
    r"""The negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K > 1`
            in the case of K-dimensional loss.
        target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> output = F.nll_loss(F.log_softmax(input), target)
        >>> output.backward()
    """
    if size_average is not None or reduce is not None:
        _size_average = torch.jit._unwrap_optional(size_average)
        _reduce = torch.jit._unwrap_optional(reduce)
        reduction = _Reduction.legacy_get_string(_size_average, _reduce)
    dim = input.dim()
    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))
    if dim == 2:
        ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
    elif dim == 4:
        ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
    elif dim == 3 or dim > 4:
        n = input.size(0)
        c = input.size(1)
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(
                out_size, target.size()))
        input = input.contiguous().view(n, c, 1, -1)
        target = target.contiguous().view(n, 1, -1)
        reduction_enum = _Reduction.get_enum(reduction)
        if reduction is not 'none':
            ret = torch._C._nn.nll_loss2d(
                input, target, weight, reduction_enum, ignore_index)
        else:
            out = torch._C._nn.nll_loss2d(
                input, target, weight, reduction_enum, ignore_index)
            ret = out.view(out_size)
    return ret


def poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-8,
                     reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, bool, bool, Optional[bool], float, Optional[bool], str) -> Tensor
    r"""Poisson negative log likelihood loss.

    See :class:`~torch.nn.PoissonNLLLoss` for details.

    Args:
        input: expectation of underlying Poisson distribution.
        target: random sample :math:`target \sim \text{Poisson}(input)`.
        log_input: if ``True`` the loss is computed as
            :math:`\exp(\text{input}) - \text{target} * \text{input}`, if ``False`` then loss is
            :math:`\text{input} - \text{target} * \log(\text{input}+\text{eps})`. Default: ``True``
        full: whether to compute full loss, i. e. to add the Stirling
            approximation term. Default: ``False``
            :math:`\text{target} * \log(\text{target}) - \text{target} + 0.5 * \log(2 * \pi * \text{target})`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input`=``False``. Default: 1e-8
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if log_input:
        loss = torch.exp(input) - target * input
    else:
        loss = input - target * torch.log(input + eps)
    if full:
        mask = target > 1
        loss[mask] += (target * torch.log(target) - target + 0.5 * torch.log(2 * math.pi * target))[mask]
    if reduction is 'none':
        return loss
    if reduction is 'mean':
        return torch.mean(loss)
    return torch.sum(loss)


@torch._jit_internal.weak_script
def kl_div(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""The `Kullback-Leibler divergence`_ Loss.

    See :class:`~torch.nn.KLDivLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'
    """
    if size_average is not None or reduce is not None:
        _size_average = torch.jit._unwrap_optional(size_average)
        _reduce = torch.jit._unwrap_optional(reduce)
        _reduction_enum = _Reduction.legacy_get_enum(_size_average, _reduce)
    else:
        _reduction_enum = _Reduction.get_enum(reduction)
    return torch.kl_div(input, target, _reduction_enum)


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    r"""This criterion combines `log_softmax` and `nll_loss` in a single
    function.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        input (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K > 1`
            in the case of K-dimensional loss.
        target (Tensor) : :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Examples::

        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randint(5, (3,), dtype=torch.int64)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)


def binary_cross_entropy(input, target, weight=None, size_average=None,
                         reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
    r"""Function that measures the Binary Cross Entropy
    between the target and the output.

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Examples::

        >>> input = torch.randn((3, 2), requires_grad=True)
        >>> target = torch.rand((3, 2), requires_grad=False)
        >>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
        >>> loss.backward()
    """
    if size_average is not None or reduce is not None:
        _size_average = torch.jit._unwrap_optional(size_average)
        _reduce = torch.jit._unwrap_optional(reduce)
        _reduction_enum = _Reduction.legacy_get_enum(_size_average, _reduce)
    else:
        _reduction_enum = _Reduction.get_enum(reduction)
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    if input.nelement() != target.nelement():
        raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                         "!= input nelement ({})".format(target.nelement(), input.nelement()))

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)

    return torch._C._nn.binary_cross_entropy(
        input, target, weight, _reduction_enum)


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    r"""Function that measures Binary Cross Entropy between target and output
    logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Examples::

         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    return torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction)


def _pointwise_loss(lambd, lambd_optimized, input, target, reduction='mean'):
    if target.requires_grad:
        d = lambd(input, target)
        if reduction == 'none':
            return d
        return torch.mean(d) if reduction == 'mean' else torch.sum(d)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        return lambd_optimized(expanded_input, expanded_target, _Reduction.get_enum(reduction))


def _smooth_l1_loss(input, target):
    t = torch.abs(input - target)
    return torch.where(t < 1, 0.5 * t ** 2, t - 0.5)


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    r"""Function that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return _pointwise_loss(
        _smooth_l1_loss,
        torch._C._nn.smooth_l1_loss, input, target, reduction)


def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Function that takes the mean element-wise absolute value difference.

    See :class:`~torch.nn.L1Loss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return _pointwise_loss(lambda a, b: torch.abs(a - b), torch._C._nn.l1_loss,
                           input, target, reduction)


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :class:`~torch.nn.MSELoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)


def margin_ranking_loss(input1, input2, target, margin=0, size_average=None,
                        reduce=None, reduction='mean'):
    r"""margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MarginRankingLoss` for details.
    """  # noqa
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    if input1.dim() == 0 or input2.dim() == 0 or target.dim() == 0:
        raise RuntimeError(("margin_ranking_loss does not support scalars, got sizes: "
                            "input1: {}, input2: {}, target: {} ".format(input1.size(), input2.size(), target.size())))
    return torch.margin_ranking_loss(input1, input2, target, margin, reduction)


def hinge_embedding_loss(input, target, margin=1.0, size_average=None,
                         reduce=None, reduction='mean'):
    r"""hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.HingeEmbeddingLoss` for details.
    """  # noqa
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    return torch.hinge_embedding_loss(input, target, margin, reduction)


def multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    r"""multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiLabelMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    return torch._C._nn.multilabel_margin_loss(input, target, reduction)


def soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    r"""soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.SoftMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    return torch._C._nn.soft_margin_loss(input, target, reduction)


def multilabel_soft_margin_loss(input, target, weight=None, size_average=None,
                                reduce=None, reduction='mean'):
    r"""multilabel_soft_margin_loss(input, target, weight=None, size_average=None) -> Tensor

    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))

    if weight is not None:
        loss = loss * weight

    loss = loss.sum(dim=1) / input.size(1)  # only return N loss values

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(reduction + " is not valid")


def cosine_embedding_loss(input1, input2, target, margin=0, size_average=None,
                          reduce=None, reduction='mean'):
    r"""cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.CosineEmbeddingLoss` for details.
    """  # noqa
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    return torch.cosine_embedding_loss(input1, input2, target, margin, reduction)


def multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
                      reduce=None, reduction='mean'):
    r"""multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
                          reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    if p != 1 and p != 2:
        raise ValueError('only p == 1 and p == 2 supported')
    if weight is not None and weight.dim() != 1:
        raise ValueError('weight must be one-dimensional')

    return torch._C._nn.multi_margin_loss(input, target, p, margin, weight, reduction)


pixel_shuffle = _add_docstr(torch.pixel_shuffle, r"""
Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a
tensor of shape :math:`(C, H \times r, W \times r)`.

See :class:`~torch.nn.PixelShuffle` for details.

Args:
    input (Tensor): the input tensor
    upscale_factor (int): factor to increase spatial resolution by

Examples::

    >>> input = torch.randn(1, 9, 4, 4)
    >>> output = torch.nn.functional.pixel_shuffle(input, 3)
    >>> print(output.size())
    torch.Size([1, 1, 12, 12])
""")


def upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    r"""Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(...)``.

    .. include:: cuda_deterministic_backward.rst

    The algorithm used for upsampling is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric upsampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for upsampling are: `nearest`, `linear` (3D-only),
    `bilinear` (4D-only), `trilinear` (5D-only)

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
        mode (string): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear'. Default: 'nearest'
        align_corners (bool, optional): if True, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is `linear`,
            `bilinear`, or `trilinear`. Default: False

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.

    """
    warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
    return interpolate(input, size, scale_factor, mode, align_corners)


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    r"""Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for interpolation is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric sampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear` (4D-only), `trilinear` (5D-only), `area`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (string): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
        align_corners (bool, optional): if True, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is `linear`,
            `bilinear`, or `trilinear`. Default: False

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.

    .. include:: cuda_deterministic_backward.rst
    """
    from numbers import Integral
    from .modules.utils import _ntuple

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple)\
                and len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. '
                             'Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]

    if mode in ('nearest', 'area'):
        if align_corners is not None:
            raise ValueError("align_corners option can only be set with the "
                             "interpolating modes: linear | bilinear | trilinear")
    else:
        if align_corners is None:
            warnings.warn("Default upsampling behavior when mode={} is changed "
                          "to align_corners=False since 0.4.0. Please specify "
                          "align_corners=True if the old behavior is desired. "
                          "See the documentation of nn.Upsample for details.".format(mode))
            align_corners = False

    if input.dim() == 3 and mode == 'nearest':
        return torch._C._nn.upsample_nearest1d(input, _output_size(1))
    elif input.dim() == 4 and mode == 'nearest':
        return torch._C._nn.upsample_nearest2d(input, _output_size(2))
    elif input.dim() == 5 and mode == 'nearest':
        return torch._C._nn.upsample_nearest3d(input, _output_size(3))
    elif input.dim() == 3 and mode == 'area':
        return adaptive_avg_pool1d(input, _output_size(1))
    elif input.dim() == 4 and mode == 'area':
        return adaptive_avg_pool2d(input, _output_size(2))
    elif input.dim() == 5 and mode == 'area':
        return adaptive_avg_pool3d(input, _output_size(3))
    elif input.dim() == 3 and mode == 'linear':
        return torch._C._nn.upsample_linear1d(input, _output_size(1), align_corners)
    elif input.dim() == 3 and mode == 'bilinear':
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    elif input.dim() == 3 and mode == 'trilinear':
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    elif input.dim() == 4 and mode == 'linear':
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    elif input.dim() == 4 and mode == 'bilinear':
        return torch._C._nn.upsample_bilinear2d(input, _output_size(2), align_corners)
    elif input.dim() == 4 and mode == 'trilinear':
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    elif input.dim() == 5 and mode == 'linear':
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    elif input.dim() == 5 and mode == 'bilinear':
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")
    elif input.dim() == 5 and mode == 'trilinear':
        return torch._C._nn.upsample_trilinear3d(input, _output_size(3), align_corners)
    else:
        raise NotImplementedError("Input Error: Only 3D, 4D and 5D input Tensors supported"
                                  " (got {}D) for the modes: nearest | linear | bilinear | trilinear"
                                  " (got {})".format(input.dim(), mode))


def upsample_nearest(input, size=None, scale_factor=None):
    r"""Upsamples the input, using nearest neighbours' pixel values.

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(..., mode='nearest')``.

    Currently spatial and volumetric upsampling are supported (i.e. expected
    inputs are 4 or 5 dimensional).

    Args:
        input (Tensor): input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatia
            size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.

    .. include:: cuda_deterministic_backward.rst
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.upsample_nearest is deprecated. Use nn.functional.interpolate instead.")
    return interpolate(input, size, scale_factor, mode='nearest')


def upsample_bilinear(input, size=None, scale_factor=None):
    r"""Upsamples the input, using bilinear upsampling.

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with
        ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    Expected inputs are spatial (4 dimensional). Use `upsample_trilinear` fo
    volumetric (5 dimensional) inputs.

    Args:
        input (Tensor): input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int or Tuple[int, int]): multiplier for spatial size

    .. include:: cuda_deterministic_backward.rst
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead.")
    return interpolate(input, size, scale_factor, mode='bilinear', align_corners=True)


GRID_SAMPLE_INTERPOLATION_MODES = {
    'bilinear': 0,
    'nearest': 1,
}

GRID_SAMPLE_PADDING_MODES = {
    'zeros': 0,
    'border': 1,
    'reflection': 2,
}


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros'):
    r"""Given an :attr:`input` and a flow-field :attr:`grid`, computes the
    ``output`` using :attr:`input` values and pixel locations from :attr:`grid`.

    Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are
    supported.

    In the spatial (4-D) case, for :attr:`input` with shape
    :math:`(N, C, H_\text{in}, W_\text{in})` and :attr:`grid` with shape
    :math:`(N, H_\text{out}, W_\text{out}, 2)`, the output will have shape
    :math:`(N, C, H_\text{out}, W_\text{out})`.

    For each output location ``output[n, :, h, w]``, the size-2 vector
    ``grid[n, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,
    which are used to interpolate the output value ``output[n, :, h, w]``.
    In the case of 5D inputs, ``grid[n, d, h, w]`` specifies the
    ``x``, ``y``, ``z`` pixel locations for interpolating
    ``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or
    ``bilinear`` interpolation method to sample the input pixels.

    :attr:`grid` should have most values in the range of ``[-1, 1]``. This is
    because the pixel locations are normalized by the :attr:`input` spatial
    dimensions. For example, values ``x = -1, y = -1`` is the left-top pixel of
    :attr:`input`, and values  ``x = 1, y = 1`` is the right-bottom pixel of
    :attr:`input`.

    If :attr:`grid` has values outside the range of ``[-1, 1]``, those locations
    are handled as defined by :attr:`padding_mode`. Options are

        * ``padding_mode="zeros"``: use ``0`` for out-of-bound values,
        * ``padding_mode="border"``: use border values for out-of-bound values,
        * ``padding_mode="reflection"``: use values at locations reflected by
          the border for out-of-bound values. For location far away from the
          border, it will keep being reflected until becoming in bound, e.g.,
          (normalized) pixel location ``x = -3.5`` reflects by ``-1`` and
          becomes ``x' = 2.5``, then reflects by border ``1`` and becomes
          ``x'' = -0.5``.

    .. Note:: This function is often used in building Spatial Transformer Networks.
    .. include:: cuda_deterministic_backward.rst

    Args:
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_\text{out}, W_\text{out}, 2)` (4-D case)
                       or :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)` (5-D case)
        mode (str): interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'

    Returns:
        output (Tensor): output Tensor

    """
    if mode not in GRID_SAMPLE_INTERPOLATION_MODES:
        raise ValueError("nn.functional.grid_sample(): expected mode to be "
                         "'bilinear' or 'nearest', but got: '{}'".format(mode))
    if padding_mode not in GRID_SAMPLE_PADDING_MODES:
        raise ValueError("nn.functional.grid_sample(): expected padding_mode "
                         "to be 'zeros', 'border', or 'reflection', "
                         "but got: '{}'".format(padding_mode))
    return torch.grid_sampler(input, grid, GRID_SAMPLE_INTERPOLATION_MODES[mode],
                              GRID_SAMPLE_PADDING_MODES[padding_mode])


def affine_grid(theta, size):
    r"""Generates a 2d flow field, given a batch of affine matrices :attr:`theta`
    Generally used in conjunction with :func:`grid_sample` to
    implement Spatial Transformer Networks.

    Args:
        theta (Tensor): input batch of affine matrices (:math:`N \times 2 \times 3`)
        size (torch.Size): the target output image size (:math:`N \times C \times H \times W`)
                           Example: torch.Size((32, 3, 24, 24))

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)
    """
    return vision.affine_grid_generator(theta, size)


def pad(input, pad, mode='constant', value=0):
    r"""Pads tensor.

    Pading size:
        The number of dimensions to pad is :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor`
        and the dimensions that get padded begins with the last dimension and moves forward.
        For example, to pad the last dimension of the input tensor, then `pad` has form
        `(padLeft, padRight)`; to pad the last 2 dimensions of the input tensor, then use
        `(padLeft, padRight, padTop, padBottom)`; to pad the last 3 dimensions, use
        `(padLeft, padRight, padTop, padBottom, padFront, padBack)`.

    Padding mode:
        See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and
        :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the
        padding modes works. Constant padding is implemented for arbitrary dimensions.
        Replicate padding is implemented for padding the last 3 dimensions of 5D input
        tensor, or the last 2 dimensions of 4D input tensor, or the last dimension of
        3D input tensor. Reflect padding is only implemented for padding the last 2
        dimensions of 4D input tensor, or the last dimension of 3D input tensor.

    .. include:: cuda_deterministic_backward.rst

    Args:
        input (Tensor): `Nd` tensor
        pad (tuple): m-elem tuple, where :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: 'constant', 'reflect' or 'replicate'. Default: 'constant'
        value: fill value for 'constant' padding. Default: 0

    Examples::

        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.data.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.data.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.data.size())
        torch.Size([3, 9, 7, 3])

    """
    assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(pad) // 2 <= input.dim(), 'Padding length too large'
    if mode == 'constant':
        return _VF.constant_pad_nd(input, pad, value)
    else:
        assert value == 0, 'Padding mode "{}"" doesn\'t take in value argument'.format(mode)
        if input.dim() == 3:
            assert len(pad) == 2, '3D tensors expect 2 values for padding'
            if mode == 'reflect':
                return torch._C._nn.reflection_pad1d(input, pad)
            elif mode == 'replicate':
                return torch._C._nn.replication_pad1d(input, pad)
        elif input.dim() == 4:
            assert len(pad) == 4, '4D tensors expect 4 values for padding'
            if mode == 'reflect':
                return torch._C._nn.reflection_pad2d(input, pad)
            elif mode == 'replicate':
                return torch._C._nn.replication_pad2d(input, pad)
        elif input.dim() == 5:
            assert len(pad) == 6, '5D tensors expect 6 values for padding'
            if mode == 'reflect':
                raise NotImplementedError
            elif mode == 'replicate':
                return torch._C._nn.replication_pad3d(input, pad)
        else:
            raise NotImplementedError("Only 3D, 4D, 5D padding with non-constant padding are supported for now")


# distance

@torch._jit_internal.weak_script
def pairwise_distance(x1, x2, p=2., eps=1e-6, keepdim=False):
    # type: (Tensor, Tensor, float, float, bool) -> Tensor
    r"""
    See :class:`torch.nn.PairwiseDistance` for details
    """
    return torch.pairwise_distance(x1, x2, p, eps, keepdim)


pdist = _add_docstr(torch.pdist, r"""
pdist(input, p=2) -> Tensor

Computes the p-norm distance between every pair of row vectors in the input.
This is identical to the upper triangular portion, excluding the diagonal, of
`torch.norm(input[:, None] - input, dim=2, p=p)`. This function will be faster
if the rows are contiguous.

If input has shape :math:`N \times M` then the output will have shape
:math:`\frac{1}{2} N (N - 1)`.

This function is equivalent to `scipy.spatial.distance.pdist(input,
'minkowski', p=p)` if :math:`p \in (0, \infty)`. When :math:`p = 0` it is
equivalent to `scipy.spatial.distance.pdist(input, 'hamming') * M`.
When :math:`p = \infty`, the closest scipy function is
`scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())`.

Args:
    input: input tensor of shape :math:`N \times M`.
    p: p value for the p-norm distance to calculate between each vector pair
        :math:`\in [0, \infty]`.
""")


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    r"""Returns cosine similarity between x1 and x2, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        x1 (Tensor): First input.
        x2 (Tensor): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.

    Example::

        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> output = F.cosine_similarity(input1, input2)
        >>> print(output)
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False, size_average=None,
                        reduce=None, reduction="mean"):
    r"""
    See :class:`~torch.nn.TripletMarginLoss` for details
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    return torch.triplet_margin_loss(anchor, positive, negative, margin, p, eps,
                                     swap, reduction)


def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
        out (Tensor, optional): the output tensor. If :attr:`out` is used, this
                                operation won't be differentiable.
    """
    if out is None:
        denom = input.norm(p, dim, True).clamp(min=eps).expand_as(input)
        return input / denom
    else:
        denom = input.norm(p, dim, True).clamp_(min=eps).expand_as(input)
        return torch.div(input, denom, out=out)


def assert_int_or_pair(arg, arg_name, message):
    assert isinstance(arg, int) or len(arg) == 2, message.format(arg_name)


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    r"""Extracts sliding local blocks from an batched input tensor.

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    See :class:`torch.nn.Unfold` for details
    """

    if input.dim() == 4:
        msg = '{} must be int or 2-tuple for 4D input'
        assert_int_or_pair(kernel_size, 'kernel_size', msg)
        assert_int_or_pair(dilation, 'dilation', msg)
        assert_int_or_pair(padding, 'padding', msg)
        assert_int_or_pair(stride, 'stride', msg)

        return Im2Col.apply(input, _pair(kernel_size),
                            _pair(dilation), _pair(padding), _pair(stride))
    else:
        raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(input.dim()))


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    r"""Combines an array of sliding local blocks into a large containing
    tensor.

    .. warning::
        Currently, only 4-D output tensors (batched image-like tensors) are
        supported.

    See :class:`torch.nn.Fold` for details
    """
    if input.dim() == 3:
        msg = '{} must be int or 2-tuple for 3D input'
        assert_int_or_pair(output_size, 'output_size', msg)
        assert_int_or_pair(kernel_size, 'kernel_size', msg)
        assert_int_or_pair(dilation, 'dilation', msg)
        assert_int_or_pair(padding, 'padding', msg)
        assert_int_or_pair(stride, 'stride', msg)

        return Col2Im.apply(input, _pair(output_size), _pair(kernel_size),
                            _pair(dilation), _pair(padding), _pair(stride))
    else:
        raise NotImplementedError("Input Error: Only 3D input Tensors are supported (got {}D)".format(input.dim()))
