
from torch import nn
from .utils import __alias

_module_name = __name__

AdaptiveAvgPool2d = __alias(nn.AdaptiveAvgPool2d, module=_module_name, docstring=r"""
Applies a 2D adaptive average pooling over a quantized input signal composed of several input planes.

The output is of size H x W, for any input size.
The number of output features is equal to the number of input planes.

.. note:: The quantization parameters of the output are propagated from the input.

Args:
    output_size: the target output size of the image of the form H x W.
                 Can be a tuple (H, W) or a single H for a square image H x H.
                 H and W can be either a ``int``, or ``None`` which means the size will
                 be the same as that of the input.

Examples::

    >>> # target output size of 5x7
    >>> m = nnq.AdaptiveAvgPool2d((5,7))
    >>> input = torch.randn(1, 64, 8, 9)
    >>> input = torch.quantize_per_tensor(input, 1e-6, 0, torch.qint32)
    >>> output = m(input)
    >>> # target output size of 7x7 (square)
    >>> m = nnq.AdaptiveAvgPool2d(7)
    >>> input = torch.randn(1, 64, 10, 9)
    >>> input = torch.quantize_per_tensor(input, 1e-6, 0, torch.qint32)
    >>> output = m(input)
    >>> # target output size of 10x7
    >>> m = nnq.AdaptiveMaxPool2d((None, 7))
    >>> input = torch.randn(1, 64, 10, 9)
    >>> input = torch.quantize_per_tensor(input, 1e-6, 0, torch.qint32)
    >>> output = m(input)

See :class:`~torch.nn.AdaptiveAvgPool2d` for non-quantized version.
""")

AvgPool2d = __alias(nn.AvgPool2d, module=_module_name, docstring=r"""
Applies a 2D average pooling over a quantized input signal composed of several input planes.

In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
can be precisely described as:

.. math::

    out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                           input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
for :attr:`padding` number of points.

The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

    - a single ``int`` -- in which case the same value is used for the height and width dimension
    - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
      and the second `int` for the width dimension

.. note:: The quantization parameters of the output are propagated from the input.

Args:
    kernel_size: the size of the window
    stride: the stride of the window. Default value is :attr:`kernel_size`
    padding: implicit zero padding to be added on both sides
    ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
    count_include_pad: when True, will include the zero-padding in the averaging calculation
    divisor_override: if specified, it will be used as divisor, otherwise attr:`kernel_size` will be used

Shape:
    - Input: :math:`(N, C, H_{in}, W_{in})`
    - Output: :math:`(N, C, H_{out}, W_{out})`, where

      .. math::
          H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
            \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

      .. math::
          W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
            \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

Examples::

    >>> # pool of square window of size=3, stride=2
    >>> m = nnq.AvgPool2d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nnq.AvgPool2d((3, 2), stride=(2, 1))
    >>> input = torch.randn(20, 16, 50, 32)
    >>> input = torch.quantize_per_tensor(input, 1e-6, 0, torch.qint32)
    >>> output = m(input)

See :class:`~torch.nn.AvgPool2d` for non-quantized version.
""")

MaxPool2d = __alias(nn.MaxPool2d, module=_module_name, docstring=r"""
Applies a 2D max pooling over a quantized input signal composed of several input planes.

In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
can be precisely described as:

.. note:: The quantization parameters of the output are propagated from the input.

.. math::
    \begin{aligned}
        out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                               \text{stride[1]} \times w + n)
    \end{aligned}

If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

    - a single ``int`` -- in which case the same value is used for the height and width dimension
    - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
      and the second `int` for the width dimension

Args:
    kernel_size: the size of the window to take a max over
    stride: the stride of the window. Default value is :attr:`kernel_size`
    padding: implicit zero padding to be added on both sides
    dilation: a parameter that controls the stride of elements in the window
    return_indices: if ``True``, will return the max indices along with the outputs.
                    Useful for :class:`torch.nn.MaxUnpool2d` later
    ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

Shape:
    - Input: :math:`(N, C, H_{in}, W_{in})`
    - Output: :math:`(N, C, H_{out}, W_{out})`, where

      .. math::
          H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

      .. math::
          W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

Examples::

    >>> # pool of square window of size=3, stride=2
    >>> m = nnq.MaxPool2d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nnq.MaxPool2d((3, 2), stride=(2, 1))
    >>> input = torch.randn(20, 16, 50, 32)
    >>> input = torch.quantize_per_tensor(input, 1e-6, 0, torch.qint32)
    >>> output = m(input)

See :class:`~torch.nn.MaxPool2d` for non-quantized version.

.. _link:
    https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
""")
