r"""Functional interface"""
from __future__ import division

import warnings
import math

import torch
from torch._C import _infer_size, _add_docstr
from . import _reduction as _Reduction
from .modules import utils
from .modules.utils import _single, _pair, _triple, _list_with_default
from . import grad  # noqa: F401
from . import _VF
from .._jit_internal import boolean_dispatch, List


conv1d = _add_docstr(torch.conv1d, r"""
conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See :class:`~torch.nn.Conv1d` for details and output shape.

.. include:: cudnn_deterministic.rst

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or
      a one-element tuple `(sW,)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a
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
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sH, sW)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a
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
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)`
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sT, sH, sW)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a
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
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sW,)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
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
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sH, sW)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
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
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kT , kH , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sT, sH, sW)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
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
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    kernel_size: the size of the window. Can be a single number or a
      tuple `(kW,)`
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
    >>> input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)
    >>> F.avg_pool1d(input, kernel_size=3, stride=2)
    tensor([[[ 2.,  4.,  6.]]])

""")


avg_pool2d = _add_docstr(torch._C._nn.avg_pool2d, r"""
avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size
:math:`sH \times sW` steps. The number of output features is equal to the number of
input planes.

See :class:`~torch.nn.AvgPool2d` for details and output shape.

Args:
    input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
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
""")

avg_pool3d = _add_docstr(torch._C._nn.avg_pool3d, r"""
avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

Applies 3D average-pooling operation in :math:`kT \times kH \times kW` regions by step
size :math:`sT \times sH \times sW` steps. The number of output features is equal to
:math:`\lfloor\frac{\text{input planes}}{sT}\rfloor`.

See :class:`~torch.nn.AvgPool3d` for details and output shape.

Args:
    input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iT \times iH , iW)`
    kernel_size: size of the pooling region. Can be a single number or a
      tuple `(kT, kH, kW)`
    stride: stride of the pooling operation. Can be a single number or a
      tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padT, padH, padW)`, Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` in the formula
        to compute the output shape
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation
    divisor_override: if specified, it will be used as divisor, otherwise
        size of the pooling region will be used. Default: None
""")


def fractional_max_pool2d_with_indices(input, kernel_size, output_size=None,
                                       output_ratio=None, return_indices=False,
                                       _random_samples=None):
    # type: (Tensor, BroadcastingList2[int], Optional[BroadcastingList2[int]], Optional[BroadcastingList2[float]], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor]  # noqa
    r"""Applies 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \times k`)
                     or a tuple `(kH, kW)`
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
                         "an output_size or an output_ratio")
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _pair(output_ratio)
        output_size = [int(input.size(2) * _output_ratio[0]),
                       int(input.size(3) * _output_ratio[1])]

    if _random_samples is None:
        _random_samples = torch.rand(input.size(0), input.size(1), 2, dtype=input.dtype, device=input.device)
    return torch._C._nn.fractional_max_pool2d(input, kernel_size, output_size, _random_samples)


def _fractional_max_pool2d(input, kernel_size, output_size=None,
                           output_ratio=None, return_indices=False,
                           _random_samples=None):
    # type: (Tensor, BroadcastingList2[int], Optional[BroadcastingList2[int]], Optional[BroadcastingList2[float]], bool, Optional[Tensor]) -> Tensor  # noqa
    return fractional_max_pool2d_with_indices(input, kernel_size, output_size,
                                              output_ratio, return_indices,
                                              _random_samples)[0]

fractional_max_pool2d = boolean_dispatch(
    arg_name='return_indices',
    arg_index=4,
    default=False,
    if_true=fractional_max_pool2d_with_indices,
    if_false=_fractional_max_pool2d,
    module_name=__name__,
    func_name='fractional_max_pool2d')


def fractional_max_pool3d_with_indices(input, kernel_size, output_size=None,
                                       output_ratio=None, return_indices=False,
                                       _random_samples=None):
    # type: (Tensor, BroadcastingList3[int], Optional[BroadcastingList3[int]], Optional[BroadcastingList3[float]], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor]  # noqa
    r"""Applies 3D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kT \times kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \times k \times k`)
                     or a tuple `(kT, kH, kW)`
        output_size: the target output size of the form :math:`oT \times oH \times oW`.
                     Can be a tuple `(oT, oH, oW)` or a single number :math:`oH` for a cubic output
                      :math:`oH \times oH \times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool3d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32, 16)
        >>> # pool of cubic window of size=3, and target output size 13x12x11
        >>> F.fractional_max_pool3d(input, 3, output_size=(13, 12, 11))
        >>> # pool of cubic window and target output size being half of input size
        >>> F.fractional_max_pool3d(input, 3, output_ratio=(0.5, 0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
    if output_size is None and output_ratio is None:
        raise ValueError("fractional_max_pool3d requires specifying either "
                         "an output_size or an output_ratio")
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _triple(output_ratio)
        output_size = [int(input.size(2) * _output_ratio[0]),
                       int(input.size(3) * _output_ratio[1]),
                       int(input.size(4) * _output_ratio[2])]

    if _random_samples is None:
        _random_samples = torch.rand(input.size(0), input.size(1), 3, dtype=input.dtype, device=input.device)
    return torch._C._nn.fractional_max_pool3d(input, kernel_size, output_size, _random_samples)


def _fractional_max_pool3d(input, kernel_size, output_size=None,
                           output_ratio=None, return_indices=False,
                           _random_samples=None):
    # type: (Tensor, BroadcastingList3[int], Optional[BroadcastingList3[int]], Optional[BroadcastingList3[float]], bool, Optional[Tensor]) -> Tensor  # noqa
    return fractional_max_pool3d_with_indices(input, kernel_size, output_size,
                                              output_ratio, return_indices,
                                              _random_samples)[0]

fractional_max_pool3d = boolean_dispatch(
    arg_name='return_indices',
    arg_index=4,
    default=False,
    if_true=fractional_max_pool3d_with_indices,
    if_false=_fractional_max_pool3d,
    module_name=__name__,
    func_name='fractional_max_pool3d')


def max_pool1d_with_indices(input, kernel_size, stride=None, padding=0,
                            dilation=1, ceil_mode=False, return_indices=False):
    # type: (Tensor, BroadcastingList1[int], Optional[BroadcastingList1[int]], BroadcastingList1[int], BroadcastingList1[int], bool, bool) -> Tuple[Tensor, Tensor]  # noqa
    r"""Applies a 1D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool1d` for details.
    """
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool1d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode)


def _max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False, return_indices=False):
    # type: (Tensor, BroadcastingList1[int], Optional[BroadcastingList1[int]], BroadcastingList1[int], BroadcastingList1[int], bool, bool) -> Tensor  # noqa
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool1d(
        input, kernel_size, stride, padding, dilation, ceil_mode)

max_pool1d = boolean_dispatch(
    arg_name='return_indices',
    arg_index=6,
    default=False,
    if_true=max_pool1d_with_indices,
    if_false=_max_pool1d,
    module_name=__name__,
    func_name='max_pool1d')


def max_pool2d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1,
                            ceil_mode=False, return_indices=False):
    # type: (Tensor, BroadcastingList2[int], Optional[BroadcastingList2[int]], BroadcastingList2[int], BroadcastingList2[int], bool, bool) -> Tuple[Tensor, Tensor]  # noqa
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool2d` for details.
    """
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch._C._nn.max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)


def _max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False, return_indices=False):
    # type: (Tensor, BroadcastingList2[int], Optional[BroadcastingList2[int]], BroadcastingList2[int], BroadcastingList2[int], bool, bool) -> Tensor  # noqa
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool2d(
        input, kernel_size, stride, padding, dilation, ceil_mode)

max_pool2d = boolean_dispatch(
    arg_name='return_indices',
    arg_index=6,
    default=False,
    if_true=max_pool2d_with_indices,
    if_false=_max_pool2d,
    module_name=__name__,
    func_name='max_pool2d')


def max_pool3d_with_indices(input, kernel_size, stride=None, padding=0,
                            dilation=1, ceil_mode=False, return_indices=False):
    # type: (Tensor, BroadcastingList3[int], Optional[BroadcastingList3[int]], BroadcastingList3[int], BroadcastingList3[int], bool, bool) -> Tuple[Tensor, Tensor]  # noqa
    r"""Applies a 3D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool3d` for details.
    """
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch._C._nn.max_pool3d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode)


def _max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False, return_indices=False):
    # type: (Tensor, BroadcastingList3[int], Optional[BroadcastingList3[int]], BroadcastingList3[int], BroadcastingList3[int], bool, bool) -> Tensor  # noqa
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool3d(
        input, kernel_size, stride, padding, dilation, ceil_mode)

max_pool3d = boolean_dispatch(
    arg_name='return_indices',
    arg_index=6,
    default=False,
    if_true=max_pool3d_with_indices,
    if_false=_max_pool3d,
    module_name=__name__,
    func_name='max_pool3d')


def _unpool_output_size(input, kernel_size, stride, padding, output_size):
    # type: (Tensor, List[int], List[int], List[int], Optional[List[int]]) -> List[int]
    input_size = input.size()
    default_size = torch.jit.annotate(List[int], [])
    for d in range(len(kernel_size)):
        default_size.append((input_size[d + 2] - 1) * stride[d] +
                            kernel_size[d] - 2 * padding[d])
    if output_size is None:
        ret = default_size
    else:
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

        ret = output_size
    return ret


def max_unpool1d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    # type: (Tensor, Tensor, BroadcastingList1[int], Optional[BroadcastingList1[int]], BroadcastingList1[int], Optional[BroadcastingList1[int]]) -> Tensor  # noqa
    r"""Computes a partial inverse of :class:`MaxPool1d`.

    See :class:`~torch.nn.MaxUnpool1d` for details.
    """
    kernel_size = _single(kernel_size)
    if stride is not None:
        _stride = _single(stride)
    else:
        _stride = kernel_size
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding,
                                      output_size)
    if isinstance(output_size, list):
        output_size = output_size + [1]
    else:
        output_size = output_size + (1,)
    return torch._C._nn.max_unpool2d(input.unsqueeze(3), indices.unsqueeze(3),
                                     output_size).squeeze(3)


def max_unpool2d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    # type: (Tensor, Tensor, BroadcastingList2[int], Optional[BroadcastingList2[int]], BroadcastingList2[int], Optional[BroadcastingList2[int]]) -> Tensor  # noqa
    r"""Computes a partial inverse of :class:`MaxPool2d`.

    See :class:`~torch.nn.MaxUnpool2d` for details.
    """
    kernel_size = _pair(kernel_size)
    if stride is not None:
        _stride = _pair(stride)
    else:
        _stride = kernel_size
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool2d(input, indices, output_size)


def max_unpool3d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    # type: (Tensor, Tensor, BroadcastingList3[int], Optional[BroadcastingList3[int]], BroadcastingList3[int], Optional[BroadcastingList3[int]]) -> Tensor  # noqa
    r"""Computes a partial inverse of :class:`MaxPool3d`.

    See :class:`~torch.nn.MaxUnpool3d` for details.
    """
    kernel_size = _triple(kernel_size)
    if stride is not None:
        _stride = _triple(stride)
    else:
        _stride = kernel_size
    padding = _triple(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool3d(
        input, indices, output_size, _stride, padding)


def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    # type: (Tensor, float, int, Optional[BroadcastingList2[int]], bool) -> Tensor
    r"""Applies a 2D power-average pooling over an input signal composed of
    several input planes. If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool2d` for details.
    """
    kw, kh = utils._pair(kernel_size)
    if stride is not None:
        out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool2d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (torch.sign(out) * relu(torch.abs(out))).mul(kw * kh).pow(1. / norm_type)


def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    # type: (Tensor, float, int, Optional[BroadcastingList1[int]], bool) -> Tensor
    r"""Applies a 1D power-average pooling over an input signal composed of
    several input planes. If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool1d` for details.
    """
    if stride is not None:
        out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool1d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (torch.sign(out) * relu(torch.abs(out))).mul(kernel_size).pow(1. / norm_type)


def adaptive_max_pool1d_with_indices(input, output_size, return_indices=False):
    # type: (Tensor, BroadcastingList1[int], bool) -> Tuple[Tensor, Tensor]
    r"""Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    return torch.adaptive_max_pool1d(input, output_size)


def _adaptive_max_pool1d(input, output_size, return_indices=False):
    # type: (Tensor, BroadcastingList1[int], bool) -> Tensor
    return adaptive_max_pool1d_with_indices(input, output_size)[0]

adaptive_max_pool1d = boolean_dispatch(
    arg_name='return_indices',
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool1d_with_indices,
    if_false=_adaptive_max_pool1d,
    module_name=__name__,
    func_name='adaptive_max_pool1d')


def adaptive_max_pool2d_with_indices(input, output_size, return_indices=False):
    # type: (Tensor, BroadcastingList2[int], bool) -> Tuple[Tensor, Tensor]
    r"""Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_max_pool2d(input, output_size)


def _adaptive_max_pool2d(input, output_size, return_indices=False):
    # type: (Tensor, BroadcastingList2[int], bool) -> Tensor
    return adaptive_max_pool2d_with_indices(input, output_size)[0]

adaptive_max_pool2d = boolean_dispatch(
    arg_name='return_indices',
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool2d_with_indices,
    if_false=_adaptive_max_pool2d,
    module_name=__name__,
    func_name='adaptive_max_pool2d')


def adaptive_max_pool3d_with_indices(input, output_size, return_indices=False):
    # type: (Tensor, BroadcastingList3[int], bool) -> Tuple[Tensor, Tensor]
    r"""Applies a 3D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_max_pool3d(input, output_size)


def _adaptive_max_pool3d(input, output_size, return_indices=False):
    # type: (Tensor, BroadcastingList3[int], bool) -> Tensor
    return adaptive_max_pool3d_with_indices(input, output_size)[0]

adaptive_max_pool3d = boolean_dispatch(
    arg_name='return_indices',
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool3d_with_indices,
    if_false=_adaptive_max_pool3d,
    module_name=__name__,
    func_name='adaptive_max_pool3d')


adaptive_avg_pool1d = _add_docstr(torch.adaptive_avg_pool1d, r"""
adaptive_avg_pool1d(input, output_size) -> Tensor

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.

Args:
    output_size: the target output size (single integer)
""")


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
def dropout(input, p=0.5, training=True, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    return (_VF.dropout_(input, p, training)
            if inplace
            else _VF.dropout(input, p, training))


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


def dropout2d(input, p=0.5, training=True, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    r"""
    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout2d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    return (_VF.feature_dropout_(input, p, training)
            if inplace
            else _VF.feature_dropout(input, p, training))


def dropout3d(input, p=0.5, training=True, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    r"""
    Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
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


def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    return (_VF.feature_alpha_dropout_(input, p, training)
            if inplace
            else _VF.feature_alpha_dropout(input, p, training))


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


def glu(input, dim=-1):
    # type: (Tensor, int) -> Tensor
    r"""
    glu(input, dim=-1) -> Tensor

    The gated linear unit. Computes:

    .. math ::
        \text{GLU}(a, b) = a \otimes \sigma(b)

    where `input` is split in half along `dim` to form `a` and `b`, :math:`\sigma`
    is the sigmoid function and :math:`\otimes` is the element-wise product between matrices.

    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.

    Args:
        input (Tensor): input tensor
        dim (int): dimension on which to split the input. Default: -1
    """
    if input.dim() == 0:
        raise RuntimeError("glu does not suppport scalars because halving size must be even")
    return torch._C._nn.glu(input, dim)


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


def relu6(input, inplace=False):
    # type: (Tensor, bool) -> Tensor
    r"""relu6(input, inplace=False) -> Tensor

    Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.

    See :class:`~torch.nn.ReLU6` for more details.
    """
    return hardtanh(input, 0., 6., inplace)


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


def prelu(input, weight):
    # type: (Tensor, Tensor) -> Tensor
    r"""prelu(input, weight) -> Tensor

    Applies element-wise the function
    :math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a
    learnable parameter.

    See :class:`~torch.nn.PReLU` for more details.
    """
    return torch.prelu(input, weight)


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

Applies element-wise :math:`\text{LogSigmoid}(x_i) = \log \left(\frac{1}{1 + \exp(-x_i)}\right)`

See :class:`~torch.nn.LogSigmoid` for more details.
""")

def gelu(input):
    r"""gelu(input) -> Tensor

    Applies element-wise the function
    :math:`\text{GeLU}(x) = x * \Phi(x)`

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    """
    return torch._C._nn.gelu(input)


def hardshrink(input, lambd=0.5):
    # type: (Tensor, float) -> Tensor
    r"""
    hardshrink(input, lambd=0.5) -> Tensor

    Applies the hard shrinkage function element-wise

    See :class:`~torch.nn.Hardshrink` for more details.
    """
    return torch.hardshrink(input, lambd)


def tanhshrink(input):
    r"""tanhshrink(input) -> Tensor

    Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)`

    See :class:`~torch.nn.Tanhshrink` for more details.
    """
    return input - input.tanh()


def softsign(input):
    r"""softsign(input) -> Tensor

    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

    See :class:`~torch.nn.Softsign` for more details.
    """
    return input / (input.abs() + 1)


softplus = _add_docstr(torch._C._nn.softplus, r"""
softplus(input, beta=1, threshold=20) -> Tensor
""")


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
    # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
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
        ret = (-input).softmax(dim)
    else:
        ret = (-input).softmax(dim, dtype=dtype)
    return ret


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
    r"""Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

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
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


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
    if dtype is None:
        ret = input.log_softmax(dim)
    else:
        ret = input.log_softmax(dim, dtype=dtype)
    return ret


softshrink = _add_docstr(torch._C._nn.softshrink, r"""
softshrink(input, lambd=0.5) -> Tensor

Applies the soft shrinkage function elementwise

See :class:`~torch.nn.Softshrink` for more details.
""")


def tanh(input):
    r"""tanh(input) -> Tensor

    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
    warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
    return input.tanh()


def sigmoid(input):
    r"""sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    return input.sigmoid()


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
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


def bilinear(input1, input2, weight, bias=None):
    # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tensor
    r"""
    Applies a bilinear transformation to the incoming data:
    :math:`y = x_1 A x_2 + b`

    Shape:

        - input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}`
          and :math:`*` means any number of additional dimensions.
          All but the last dimension of the inputs should be the same.
        - input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`
        - weight: :math:`(\text{out\_features}, \text{in1\_features},
          \text{in2\_features})`
        - bias: :math:`(\text{out\_features})`
        - output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.
    """
    return torch.bilinear(input1, input2, weight, bias)


def _no_grad_embedding_renorm_(weight, input, max_norm, norm_type):
    # type: (Tensor, Tensor, float, float) -> Tensor
    with torch.no_grad():
        torch.embedding_renorm_(weight, input, max_norm, norm_type)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.,
              scale_grad_by_freq=False, sparse=False):
    # type: (Tensor, Tensor, Optional[int], Optional[float], float, bool, bool) -> Tensor
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
    else:
        padding_idx = -1
    if max_norm is not None:
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        input = input.contiguous()
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.nembedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2,
                  scale_grad_by_freq=False, mode='mean', sparse=False,
                  per_sample_weights=None):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[float], float, bool, str, bool, Optional[Tensor]) -> Tensor
    r"""Computes sums, means or maxes of `bags` of embeddings, without instantiating the
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
        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be 1. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not None.


    Shape:

        - :attr:`input` (LongTensor) and :attr:`offsets` (LongTensor, optional)

          - If :attr:`input` is 2D of shape `(B, N)`,

            it will be treated as ``B`` bags (sequences) each of fixed length ``N``, and
            this will return ``B`` values aggregated in a way depending on the :attr:`mode`.
            :attr:`offsets` is ignored and required to be ``None`` in this case.

          - If :attr:`input` is 1D of shape `(N)`,

            it will be treated as a concatenation of multiple bags (sequences).
            :attr:`offsets` is required to be a 1D tensor containing the
            starting index positions of each bag in :attr:`input`. Therefore,
            for :attr:`offsets` of shape `(B)`, :attr:`input` will be viewed as
            having ``B`` bags. Empty bags (i.e., having 0-length) will have
            returned vectors filled by zeros.

        - :attr:`weight` (Tensor): the learnable weights of the module of
          shape `(num_embeddings, embedding_dim)`

        - :attr:`per_sample_weights` (Tensor, optional). Has the same shape as
          :attr:`input`.

        - :attr:`output`: aggregated embedding values of shape `(B, embedding_dim)`

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

    if per_sample_weights is not None and input.size() != per_sample_weights.size():
        raise ValueError("embedding_bag: If per_sample_weights ({}) is not None, "
                         "then it must have the same shape as the input ({})"
                         .format(per_sample_weights.shape, input.shape))

    if input.dim() == 2:
        if offsets is not None:
            raise ValueError("if input is 2D, then offsets has to be None"
                             ", as input is treated is a mini-batch of"
                             " fixed length sequences. However, found "
                             "offsets of type {}".format(type(offsets)))
        offsets = torch.arange(0, input.numel(), input.size(1),
                               dtype=torch.long, device=input.device)

        input = input.reshape(-1)
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.reshape(-1)
    elif input.dim() == 1:
        if offsets is None:
            raise ValueError("offsets has to be a 1D Tensor but got None")
        if offsets.dim() != 1:
            raise ValueError("offsets has to be a 1D Tensor")
        if int(offsets[0]) != 0:
            raise ValueError("offsets[0] has to be 0, i.e., the first sequence "
                             "in the mini-batch has to start from position 0. "
                             "However, got {}".format(offsets[0].item()))
        if int(offsets[-1]) > input.size(0):
            raise ValueError("offsets[-1] can not be greater than input's length"
                             " ({}), but got offsets[-1] of {}"
                             .format(input.size(0), offsets[-1].item()))
    else:
        raise ValueError("input has to be 1D or 2D Tensor,"
                         " but got Tensor of dimension {}".format(input.dim()))
    if mode == 'sum':
        mode_enum = 0
    elif mode == 'mean':
        mode_enum = 1
    elif mode == 'max':
        mode_enum = 2

        if scale_grad_by_freq:
            raise ValueError("max mode does not support scaling the gradient by the frequency")

        if sparse:
            raise ValueError("max mode does not support sparse weights")

    else:
        raise ValueError("mode has to be one of sum, mean or max")

    if max_norm is not None:
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.nembedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)

    if per_sample_weights is not None and mode != 'sum':
        raise NotImplementedError("embedding_bag: per_sample_weights was not None. "
                                  "per_sample_weights is only supported for mode='sum' "
                                  "(got mode='{}'). Please open a feature request on GitHub."
                                  .format(mode))

    ret, _, _, _ = torch.embedding_bag(
        weight,
        input,
        offsets,
        scale_grad_by_freq,
        mode_enum,
        sparse,
        per_sample_weights)
    return ret


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], bool, float, float) -> Tensor  # noqa
    r"""Applies Batch Normalization for each channel across a batch of data.

    See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
    :class:`~torch.nn.BatchNorm3d` for details.
    """
    if training:
        size = input.size()
        # XXX: JIT script does not support the reduce from functools, and mul op is a
        # builtin, which cannot be used as a value to a func yet, so rewrite this size
        # check to a simple equivalent for loop
        #
        # TODO: make use of reduce like below when JIT is ready with the missing features:
        # from operator import mul
        # from functools import reduce
        #
        #   if reduce(mul, size[2:], size[0]) == 1
        size_prods = size[0]
        for i in range(len(size) - 2):
            size_prods *= size[i + 2]
        if size_prods == 1:
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
    # type: (Tensor, List[int], Optional[Tensor], Optional[Tensor], float) -> Tensor
    r"""Applies Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
    return torch.layer_norm(input, normalized_shape, weight, bias, eps,
                            torch.backends.cudnn.enabled)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    # type: (Tensor, int, Optional[Tensor], Optional[Tensor], float) -> Tensor
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

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
             reduction='mean', zero_infinity=False):
    # type: (Tensor, Tensor, Tensor, Tensor, int, str, bool) -> Tensor
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
            Targets cannot be blank. In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)`.
            Lengths of the inputs (must each be :math:`\leq T`)
        target_lengths: :math:`(N)`.
            Lengths of the targets
        blank (int, optional):
            Blank label. Default :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Example::

        >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
        >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
        >>> loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        >>> loss.backward()
    """
    return torch.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, _Reduction.get_enum(reduction),
                          zero_infinity)


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
    r"""The negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
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
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> output = F.nll_loss(F.log_softmax(input), target)
        >>> output.backward()
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
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
    else:
        # dim == 3 or dim > 4
        n = input.size(0)
        c = input.size(1)
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(
                out_size, target.size()))
        input = input.contiguous().view(n, c, 1, -1)
        target = target.contiguous().view(n, 1, -1)
        reduction_enum = _Reduction.get_enum(reduction)
        if reduction != 'none':
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
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        ret = input
        raise ValueError(reduction + " is not valid")

    ret = torch.poisson_nll_loss(input, target, log_input, full, eps, _Reduction.get_enum(reduction))
    return ret


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
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean'``: the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'mean'``

    .. note::
        :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,
        and in the meantime, specifying either of those two args will override :attr:`reduction`.

    .. note::
        :attr:``reduction`` = ``'mean'`` doesn't return the true kl divergence value, please use
        :attr:``reduction`` = ``'batchmean'`` which aligns with KL math definition.
        In the next major release, ``'mean'`` will be changed to be the same as 'batchmean'.
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        if reduction == 'mean':
            warnings.warn("reduction: 'mean' divides the total loss by both the batch size and the support size."
                          "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
                          "'mean' will be changed to behave the same as 'batchmean' in the next major release.")

        # special case for batchmean
        if reduction == 'batchmean':
            reduction_enum = _Reduction.get_enum('sum')
        else:
            reduction_enum = _Reduction.get_enum(reduction)

    reduced = torch.kl_div(input, target, reduction_enum)

    if reduction == 'batchmean' and input.dim() != 0:
        reduced = reduced / input.size()[0]

    return reduced


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
    r"""This criterion combines `log_softmax` and `nll_loss` in a single
    function.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        input (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
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
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

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
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> input = torch.randn((3, 2), requires_grad=True)
        >>> target = torch.rand((3, 2), requires_grad=False)
        >>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
        >>> loss.backward()
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if target.size() != input.size():
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if input.numel() != target.numel():
        raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                         "!= input nelement ({})".format(target.numel(), input.numel()))

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)

    return torch._C._nn.binary_cross_entropy(
        input, target, weight, reduction_enum)


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str, Optional[Tensor]) -> Tensor
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
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Examples::

         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    return torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction_enum)


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
    # type: (Tensor, Tensor) -> Tensor
    t = torch.abs(input - target)
    return torch.where(t < 1, 0.5 * t ** 2, t - 0.5)


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""Function that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = _smooth_l1_loss(input, target)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.smooth_l1_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret


def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Function that takes the mean element-wise absolute value difference.

    See :class:`~torch.nn.L1Loss` for details.
    """
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = torch.abs(input - target)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.l1_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = (input - target) ** 2
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret


def margin_ranking_loss(input1, input2, target, margin=0, size_average=None,
                        reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Tensor, float, Optional[bool], Optional[bool], str) -> Tensor
    r"""margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MarginRankingLoss` for details.
    """  # noqa
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if input1.dim() == 0 or input2.dim() == 0 or target.dim() == 0:
        raise RuntimeError(("margin_ranking_loss does not support scalars, got sizes: "
                            "input1: {}, input2: {}, target: {} ".format(input1.size(), input2.size(), target.size())))
    return torch.margin_ranking_loss(input1, input2, target, margin, reduction_enum)


def hinge_embedding_loss(input, target, margin=1.0, size_average=None,
                         reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, float, Optional[bool], Optional[bool], str) -> Tensor
    r"""hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.HingeEmbeddingLoss` for details.
    """  # noqa
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch.hinge_embedding_loss(input, target, margin, reduction_enum)


def multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiLabelMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch._C._nn.multilabel_margin_loss(input, target, reduction_enum)


def soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.SoftMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch._C._nn.soft_margin_loss(input, target, reduction_enum)


def multilabel_soft_margin_loss(input, target, weight=None, size_average=None,
                                reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
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
        ret = loss
    elif reduction == 'mean':
        ret = loss.mean()
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret


def cosine_embedding_loss(input1, input2, target, margin=0, size_average=None,
                          reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Tensor, float, Optional[bool], Optional[bool], str) -> Tensor
    r"""cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.CosineEmbeddingLoss` for details.
    """  # noqa
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch.cosine_embedding_loss(input1, input2, target, margin, reduction_enum)


def multi_margin_loss(input, target, p=1, margin=1., weight=None, size_average=None,
                      reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, int, float, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
    r"""multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
                          reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if p != 1 and p != 2:
        raise ValueError('only p == 1 and p == 2 supported')
    if weight is not None:
        if weight.dim() != 1:
            raise ValueError('weight must be one-dimensional')

    return torch._C._nn.multi_margin_loss(input, target, p, margin, weight, reduction_enum)


pixel_shuffle = _add_docstr(torch.pixel_shuffle, r"""
Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a
tensor of shape :math:`(*, C, H \times r, W \times r)`.

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
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only)

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to be an integer.
        mode (string): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

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
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

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

        # make scale_factor a tensor in tracing so constant doesn't get baked in
        if torch._C._get_tracing_state():
            return [(torch.floor((input.size(i + 2) * torch.tensor(float(scale_factors[i]))).float())) for i in range(dim)]
        else:
            return [int(math.floor(float(input.size(i + 2)) * scale_factors[i])) for i in range(dim)]

    if mode in ('nearest', 'area'):
        if align_corners is not None:
            raise ValueError("align_corners option can only be set with the "
                             "interpolating modes: linear | bilinear | bicubic | trilinear")
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
    elif input.dim() == 4 and mode == 'bicubic':
        return torch._C._nn.upsample_bicubic2d(input, _output_size(2), align_corners)
    else:
        raise NotImplementedError("Input Error: Only 3D, 4D and 5D input Tensors supported"
                                  " (got {}D) for the modes: nearest | linear | bilinear | bicubic | trilinear"
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


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    # type: (Tensor, Tensor, str, str, bool) -> Tensor
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

    :attr:`grid` specifies the sampling pixel locations normalized by the
    :attr:`input` spatial dimensions. Therefore, it should have most values in
    the range of ``[-1, 1]``. For example, values ``x = -1, y = -1`` is the
    left-top pixel of :attr:`input`, and values  ``x = 1, y = 1`` is the
    right-bottom pixel of :attr:`input`.

    If :attr:`grid` has values outside the range of ``[-1, 1]``, the corresponding
    outputs are handled as defined by :attr:`padding_mode`. Options are

        * ``padding_mode="zeros"``: use ``0`` for out-of-bound grid locations,
        * ``padding_mode="border"``: use border values for out-of-bound grid locations,
        * ``padding_mode="reflection"``: use values at locations reflected by
          the border for out-of-bound grid locations. For location far away
          from the border, it will keep being reflected until becoming in bound,
          e.g., (normalized) pixel location ``x = -3.5`` reflects by border ``-1``
          and becomes ``x' = 1.5``, then reflects by border ``1`` and becomes
          ``x'' = -0.5``.

    .. note::
        This function is often used in conjunction with :func:`affine_grid`
        to build `Spatial Transformer Networks`_ .
    .. include:: cuda_deterministic_backward.rst

    Args:
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_\text{out}, W_\text{out}, 2)` (4-D case)
                       or :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)` (5-D case)
        mode (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
    """
    if mode != 'bilinear' and mode != 'nearest':
        raise ValueError("nn.functional.grid_sample(): expected mode to be "
                         "'bilinear' or 'nearest', but got: '{}'".format(mode))
    if padding_mode != 'zeros' and padding_mode != 'border' and padding_mode != 'reflection':
        raise ValueError("nn.functional.grid_sample(): expected padding_mode "
                         "to be 'zeros', 'border', or 'reflection', "
                         "but got: '{}'".format(padding_mode))

    if mode == 'bilinear':
        mode_enum = 0
    else:  # mode == 'nearest'
        mode_enum = 1

    if padding_mode == 'zeros':
        padding_mode_enum = 0
    elif padding_mode == 'border':
        padding_mode_enum = 1
    else:  # padding_mode == 'reflection'
        padding_mode_enum = 2

    if align_corners is None:
        warnings.warn("Default grid_sample and affine_grid behavior has changed "
                      "to align_corners=False since 1.3.0. Please specify "
                      "align_corners=True if the old behavior is desired. "
                      "See the documentation of grid_sample for details.")
        align_corners = False

    return torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)


def affine_grid(theta, size, align_corners=None):
    # type: (Tensor, List[int], bool) -> Tensor
    r"""Generates a 2D or 3D flow field (sampling grid), given a batch of
    affine matrices :attr:`theta`.

    .. note::
        This function is often used in conjunction with :func:`grid_sample`
        to build `Spatial Transformer Networks`_ .

    Args:
        theta (Tensor): input batch of affine matrices with shape
            (:math:`N \times 2 \times 3`) for 2D or
            (:math:`N \times 3 \times 4`) for 3D
        size (torch.Size): the target output image size.
            (:math:`N \times C \times H \times W` for 2D or
            :math:`N \times C \times D \times H \times W` for 3D)
            Example: torch.Size((32, 3, 24, 24))
        align_corners (bool, optional): if ``True``, consider ``-1`` and ``1``
            to refer to the centers of the corner pixels rather than the image corners.
            Refer to :func:`grid_sample` for a more complete description.
            A grid generated by :func:`affine_grid` should be passed to :func:`grid_sample`
            with the same setting for this option.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
    .. warning::
        When ``align_corners = True``, 2D affine transforms on 1D data and
        3D affine transforms on 2D data (that is, when one of the spatial
        dimensions has unit size) are ill-defined, and not an intended use case.
        This is not a problem when ``align_corners = False``.
        Up to version 1.2.0, all grid points along a unit dimension were
        considered arbitrarily to be at ``-1``.
        From version 1.3.0, under ``align_corners = True`` all grid points
        along a unit dimension are condsidered to be at ```0``
        (the center of the input image).
    """
    if align_corners is None:
        warnings.warn("Default grid_sample and affine_grid behavior has changed "
                      "to align_corners=False since 1.3.0. Please specify "
                      "align_corners=True if the old behavior is desired. "
                      "See the documentation of grid_sample for details.")
        align_corners = False

    # enforce floating point dtype on theta
    if not theta.is_floating_point():
        raise ValueError("Expected theta to have floating point type, but got {}"
                         .format(theta.dtype))
    # check that shapes and sizes match
    if len(size) == 4:
        if theta.dim() != 3 or theta.shape[-2] != 2 or theta.shape[-1] != 3:
            raise ValueError("Expected a batch of 2D affine matrices of shape Nx2x3 "
                             "for size {}. Got {}.".format(size, theta.shape))
        spatial_size = size[-2:]  # spatial dimension sizes
    elif len(size) == 5:
        if theta.dim() != 3 or theta.shape[-2] != 3 or theta.shape[-1] != 4:
            raise ValueError("Expected a batch of 3D affine matrices of shape Nx3x4 "
                             "for size {}. Got {}.".format(size, theta.shape))
        spatial_size = size[-3:]  # spatial dimension sizes
    else:
        raise NotImplementedError("affine_grid only supports 4D and 5D sizes, "
                                  "for 2D and 3D affine transforms, respectively. "
                                  "Got size {}.".format(size))
    # check for empty span
    if align_corners and min(spatial_size) == 1:
        warnings.warn("Since version 1.3.0, affine_grid behavior has changed "
                      "for unit-size grids when align_corners=True. "
                      "This is not an intended use case of affine_grid. "
                      "See the documentation of affine_grid for details.")
    elif min(size) <= 0:
        raise ValueError("Expected non-zero, positive output size. Got {}"
                         .format(size))

    return torch.affine_grid_generator(theta, size, align_corners)


def pad(input, pad, mode='constant', value=0):
    # type: (Tensor, List[int], str, float) -> Tensor
    r"""Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding\_left}, \text{padding\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom}`
        :math:`\text{padding\_front}, \text{padding\_back})`.

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
        input (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

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
            elif mode == 'circular':
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError

        elif input.dim() == 4:
            assert len(pad) == 4, '4D tensors expect 4 values for padding'
            if mode == 'reflect':
                return torch._C._nn.reflection_pad2d(input, pad)
            elif mode == 'replicate':
                return torch._C._nn.replication_pad2d(input, pad)
            elif mode == 'circular':
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError

        elif input.dim() == 5:
            assert len(pad) == 6, '5D tensors expect 6 values for padding'
            if mode == 'reflect':
                raise NotImplementedError
            elif mode == 'replicate':
                return torch._C._nn.replication_pad3d(input, pad)
            elif mode == 'circular':
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("Only 3D, 4D, 5D padding with non-constant padding are supported for now")

# distance


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


cosine_similarity = _add_docstr(torch.cosine_similarity, r"""
cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor

Returns cosine similarity between x1 and x2, computed along dim.

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
""")


one_hot = _add_docstr(torch._C._nn.one_hot, r"""
one_hot(tensor, num_classes=-1) -> LongTensor

Takes LongTensor with index values of shape ``(*)`` and returns a tensor
of shape ``(*, num_classes)`` that have zeros everywhere except where the
index of last dimension matches the corresponding value of the input tensor,
in which case it will be 1.

See also `One-hot on Wikipedia`_ .

.. _One-hot on Wikipedia:
    https://en.wikipedia.org/wiki/One-hot

Arguments:
    tensor (LongTensor): class values of any shape.
    num_classes (int):  Total number of classes. If set to -1, the number
        of classes will be inferred as one greater than the largest class
        value in the input tensor.

Returns:
    LongTensor that has one more dimension with 1 values at the
    index of last dimension indicated by the input, and 0 everywhere
    else.

Examples:
    >>> F.one_hot(torch.arange(0, 5) % 3)
    tensor([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]])
    >>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
    tensor([[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]])
    >>> F.one_hot(torch.arange(0, 6).view(3,2) % 3)
    tensor([[[1, 0, 0],
             [0, 1, 0]],
            [[0, 0, 1],
             [1, 0, 0]],
            [[0, 1, 0],
             [0, 0, 1]]])
""")


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False, size_average=None,
                        reduce=None, reduction="mean"):
    # type: (Tensor, Tensor, Tensor, float, float, float, bool, Optional[bool], Optional[bool], str) -> Tensor
    r"""
    See :class:`~torch.nn.TripletMarginLoss` for details
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch.triplet_margin_loss(anchor, positive, negative, margin, p, eps,
                                     swap, reduction_enum)


def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    # type: (Tensor, float, int, float, Optional[Tensor]) -> Tensor
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
        denom = input.norm(p, dim, True).clamp_min(eps).expand_as(input)
        return input / denom
    else:
        denom = input.norm(p, dim, True).clamp_min(eps).expand_as(input)
        return torch.div(input, denom, out=out)


def assert_int_or_pair(arg, arg_name, message):
    assert isinstance(arg, int) or len(arg) == 2, message.format(arg_name)


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    # type: (Tensor, BroadcastingList2[int], BroadcastingList2[int], BroadcastingList2[int], BroadcastingList2[int]) -> Tensor  # noqa
    r"""Extracts sliding local blocks from an batched input tensor.

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    .. warning::

        More than one element of the unfolded tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensor, please clone it first.


    See :class:`torch.nn.Unfold` for details
    """

    if input.dim() == 4:
        msg = '{} must be int or 2-tuple for 4D input'
        assert_int_or_pair(kernel_size, 'kernel_size', msg)
        assert_int_or_pair(dilation, 'dilation', msg)
        assert_int_or_pair(padding, 'padding', msg)
        assert_int_or_pair(stride, 'stride', msg)

        return torch._C._nn.im2col(input, _pair(kernel_size),
                                   _pair(dilation), _pair(padding), _pair(stride))
    else:
        raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(input.dim()))


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    # type: (Tensor, BroadcastingList2[int], BroadcastingList2[int], BroadcastingList2[int], BroadcastingList2[int], BroadcastingList2[int]) -> Tensor  # noqa
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

        return torch._C._nn.col2im(input, _pair(output_size), _pair(kernel_size),
                                   _pair(dilation), _pair(padding), _pair(stride))
    else:
        raise NotImplementedError("Input Error: Only 3D input Tensors are supported (got {}D)".format(input.dim()))


def _pad_circular(input, padding):
    # type: (Tensor, List[int]) -> Tensor
    """
    Arguments
        :param input: tensor of shape :math:`(N, C_{\text{in}}, H, [W, D]))`
        :param padding: (tuple): m-elem tuple where m is the degree of convolution
    Returns
        :return: tensor of shape :math:`(N, C_{\text{in}}, [D + 2 * padding[0],
                 H + 2 * padding[1]], W + 2 * padding[2]))`
    """

    input = torch.cat([input, input[:, :, 0:padding[-1]]], dim=2)
    input = torch.cat([input[:, :, -(padding[-1] + padding[-2]):-padding[-1]], input], dim=2)

    if len(padding) > 2:
        input = torch.cat([input, input[:, :, :, 0:padding[-3]]], dim=3)
        input = torch.cat([input[:, :, :, -(padding[-3] + padding[-4]):-padding[-3]], input], dim=3)

    if len(padding) > 4:
        input = torch.cat([input, input[:, :, :, :, 0:padding[-5]]], dim=4)
        input = torch.cat([input[:, :, :, :, -(padding[-5] + padding[-6]):-padding[-5]], input], dim=4)

    return input


def multi_head_attention_forward(query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None                    # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                      torch.zeros((attn_mask.size(0), 1),
                                                  dtype=attn_mask.dtype,
                                                  device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
