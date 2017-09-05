import torch
from torch.autograd import Variable

from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F


class MaxPool1d(Module):
    r"""Applies a 1D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, k)  = \max_{{m}=0}^{{kernel\_size}-1} input(N_i, C_j, stride * k + m)
        \end{array}

    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
      for :attr:`padding` number of points
    | :attr:`dilation` controls the spacing between the kernel points. It is harder to describe,
      but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if True, will return the max indices along with the outputs.
                        Useful when Unpooling later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})` where
          :math:`L_{out} = floor((L_{in}  + 2 * padding - dilation * (kernel\_size - 1) - 1) / stride + 1)`

    Examples::

        >>> # pool of size=3, stride=2
        >>> m = nn.MaxPool1d(3, stride=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50))
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool1d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', dilation=' + str(self.dilation) \
            + ', ceil_mode=' + str(self.ceil_mode) + ')'


class MaxPool2d(Module):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, h, w)  = \max_{{m}=0}^{kH-1} \max_{{n}=0}^{kW-1}
                               input(N_i, C_j, stride[0] * h + m, stride[1] * w + n)
        \end{array}

    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
      for :attr:`padding` number of points
    | :attr:`dilation` controls the spacing between the kernel points. It is harder to describe,
      but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if True, will return the max indices along with the outputs.
                        Useful when Unpooling later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 and padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        return self.__class__.__name__ + ' (' \
            + 'size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ')'


class MaxUnpool1d(Module):
    r"""Computes a partial inverse of :class:`MaxPool1d`.

    :class:`MaxPool1d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool1d` takes in as input the output of :class:`MaxPool1d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    .. note:: `MaxPool1d` can map several input sizes to the same output sizes.
              Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument `output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to ``kernel_size`` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by `MaxPool1d`
        - `output_size` (optional) : a `torch.Size` that specifies the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in})`
        - Output: :math:`(N, C, H_{out})` where
          :math:`H_{out} = (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0]`
          or as given by :attr:`output_size` in the call operator

    Example::

        >>> pool = nn.MaxPool1d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool1d(2, stride=2)
        >>> input = Variable(torch.Tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]]))
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        Variable containing:
        (0 ,.,.) =
           0   2   0   4   0   6   0   8
        [torch.FloatTensor of size 1x1x8]

        >>> # Example showcasing the use of output_size
        >>> input = Variable(torch.Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]))
        >>> output, indices = pool(input)
        >>> unpool(output, indices, output_size=input.size())
        Variable containing:
        (0 ,.,.) =
           0   2   0   4   0   6   0   8   0
        [torch.FloatTensor of size 1x1x9]

        >>> unpool(output, indices)
        Variable containing:
        (0 ,.,.) =
           0   2   0   4   0   6   0   8
        [torch.FloatTensor of size 1x1x8]

    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool1d, self).__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)

    def forward(self, input, indices, output_size=None):
        return F.max_unpool1d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) + ')'


class MaxUnpool2d(Module):
    r"""Computes a partial inverse of :class:`MaxPool2d`.

    :class:`MaxPool2d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool2d` takes in as input the output of :class:`MaxPool2d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    .. note:: `MaxPool2d` can map several input sizes to the same output sizes.
              Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument `output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to ``kernel_size`` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by `MaxPool2d`
        - `output_size` (optional) : a `torch.Size` that specifies the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = (H_{in} - 1) * stride[0] -2 * padding[0] + kernel\_size[0]`
          :math:`W_{out} = (W_{in} - 1) * stride[1] -2 * padding[1] + kernel\_size[1]`
          or as given by :attr:`output_size` in the call operator

    Example::

        >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool2d(2, stride=2)
        >>> input = Variable(torch.Tensor([[[[ 1,  2,  3,  4],
        ...                                  [ 5,  6,  7,  8],
        ...                                  [ 9, 10, 11, 12],
        ...                                  [13, 14, 15, 16]]]]))
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        Variable containing:
        (0 ,0 ,.,.) =
           0   0   0   0
           0   6   0   8
           0   0   0   0
           0  14   0  16
        [torch.FloatTensor of size 1x1x4x4]

        >>> # specify a different output size than input size
        >>> unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))
        Variable containing:
        (0 ,0 ,.,.) =
           0   0   0   0   0
           6   0   8   0   0
           0   0   0  14   0
          16   0   0   0   0
           0   0   0   0   0
        [torch.FloatTensor of size 1x1x5x5]

    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)

    def forward(self, input, indices, output_size=None):
        return F.max_unpool2d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) + ')'


class MaxUnpool3d(Module):
    r"""Computes a partial inverse of :class:`MaxPool3d`.

    :class:`MaxPool3d` is not fully invertible, since the non-maximal values are lost.
    :class:`MaxUnpool3d` takes in as input the output of :class:`MaxPool3d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    .. note:: `MaxPool3d` can map several input sizes to the same output sizes.
              Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument `output_size` in the forward call.
              See the Inputs section below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to ``kernel_size`` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by `MaxPool3d`
        - `output_size` (optional) : a `torch.Size` that specifies the targeted output size

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = (D_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0]`
          :math:`H_{out} = (H_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1]`
          :math:`W_{out} = (W_{in} - 1) * stride[2] - 2 * padding[2] + kernel\_size[2]`
          or as given by :attr:`output_size` in the call operator

    Example::

        >>> # pool of square window of size=3, stride=2
        >>> pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool3d(3, stride=2)
        >>> output, indices = pool(Variable(torch.randn(20, 16, 51, 33, 15)))
        >>> unpooled_output = unpool(output, indices)
        >>> unpooled_output.size()
        torch.Size([20, 16, 51, 33, 15])
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool3d, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if stride is not None else kernel_size)
        self.padding = _triple(padding)

    def forward(self, input, indices, output_size=None):
        return F.max_unpool3d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) + ')'


class AvgPool1d(Module):
    r"""Applies a 1D average pooling over an input signal composed of several
    input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,
    output :math:`(N, C, L_{out})` and :attr:`kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, l)  = 1 / k * \sum_{{m}=0}^{k}
                               input(N_i, C_j, stride * l + m)
        \end{array}

    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
      for :attr:`padding` number of points

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can each be
    an ``int`` or a one-element tuple.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})` where
          :math:`L_{out} = floor((L_{in}  + 2 * padding - kernel\_size) / stride + 1)`

    Examples::

        >>> # pool with window of size=3, stride=2
        >>> m = nn.AvgPool1d(3, stride=2)
        >>> m(Variable(torch.Tensor([[[1,2,3,4,5,6,7]]])))
        Variable containing:
        (0 ,.,.) =
          2  4  6
        [torch.FloatTensor of size 1x1x3]
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool1d, self).__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return F.avg_pool1d(
            input, self.kernel_size, self.stride, self.padding, self.ceil_mode,
            self.count_include_pad)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'


class AvgPool2d(Module):
    r"""Applies a 2D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, h, w)  = 1 / (kH * kW) * \sum_{{m}=0}^{kH-1} \sum_{{n}=0}^{kW-1}
                               input(N_i, C_j, stride[0] * h + m, stride[1] * w + n)
        \end{array}

    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
      for :attr:`padding` number of points

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - kernel\_size[0]) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - kernel\_size[1]) / stride[1] + 1)`

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m(input)
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'


class MaxPool3d(Module):
    r"""Applies a 3D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, d, h, w)  = \max_{{k}=0}^{kD-1} \max_{{m}=0}^{kH-1} \max_{{n}=0}^{kW-1}
                         input(N_i, C_j, stride[0] * k + d, stride[1] * h + m, stride[2] * w + n)
        \end{array}

    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
      for :attr:`padding` number of points
    | :attr:`dilation` controls the spacing between the kernel points. It is harder to describe,
      but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if True, will return the max indices along with the outputs.
                        Useful when Unpooling later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = floor((D_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`H_{out} = floor((H_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[2] - dilation[2] * (kernel\_size[2] - 1) - 1) / stride[2] + 1)`

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool3d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', dilation=' + str(self.dilation) \
            + ', ceil_mode=' + str(self.ceil_mode) + ')'


class AvgPool3d(Module):
    r"""Applies a 3D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, d, h, w)  = 1 / (kD * kH * kW) * \sum_{{k}=0}^{kD-1} \sum_{{m}=0}^{kH-1} \sum_{{n}=0}^{kW-1}
                               input(N_i, C_j, stride[0] * d + k, stride[1] * h + m, stride[2] * w + n)
        \end{array}

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = floor((D_{in}  - kernel\_size[0]) / stride[0] + 1)`
          :math:`H_{out} = floor((H_{in}  - kernel\_size[1]) / stride[1] + 1)`
          :math:`W_{out} = floor((W_{in}  - kernel\_size[2]) / stride[2] + 1)`

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
        >>> output = m(input)
    """

    def __init__(self, kernel_size, stride=None):
        super(AvgPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        return F.avg_pool3d(input, self.kernel_size, self.stride)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) + ')'


class FractionalMaxPool2d(Module):
    """Applies a 2D fractional max pooling over an input signal composed of several input planes.

    Fractiona MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in kHxkW regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        output_size: the target output size of the image of the form oH x oW.
                     Can be a tuple (oH, oW) or a single number oH for a square image oH x oH
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if True, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool2d . Default: False

    Examples:
        >>> # pool of square window of size=3, and target output size 13x12
        >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m(input)

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """

    def __init__(self, kernel_size, output_size=None, output_ratio=None,
                 return_indices=False, _random_samples=None):
        super(FractionalMaxPool2d, self).__init__()
        self.kh, self.kw = _pair(kernel_size)
        self.return_indices = return_indices
        self.register_buffer('_random_samples', _random_samples)
        if output_size is not None:
            self.outh, self.outw = _pair(output_size)
            self.rh, self.rw = None, None
            assert output_ratio is None
        elif output_ratio is not None:
            self.outh, self.outw = None, None
            self.rh, self.rw = _pair(output_ratio)
            assert output_size is None
            assert 0 < self.rh < 1
            assert 0 < self.rw < 1
        else:
            raise ValueError("FractionalMaxPool2d requires specifying either "
                             "an output size, or a pooling ratio")

    def forward(self, input):
        output_size, output_ratio = None, None
        if self.outh is not None:
            output_size = self.outh, self.outw
        else:
            output_ratio = self.rh, self.rw
        return self._backend.FractionalMaxPool2d.apply(input, self.kw, self.kh, output_size, output_ratio,
                                                       self.return_indices, self._random_samples)


class LPPool2d(Module):
    r"""Applies a 2D power-average pooling over an input signal composed of several input
    planes.

    On each window, the function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`

        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`

    Examples::

        >>> # power-2 pool of square window of size=3, stride=2
        >>> m = nn.LPPool2d(2, 3, stride=2)
        >>> # pool of non-square window of power 1.2
        >>> m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m(input)

    """

    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False):
        super(LPPool2d, self).__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.lp_pool2d(input, self.norm_type, self.kernel_size,
                           self.stride, self.ceil_mode)


class AdaptiveMaxPool1d(Module):
    """Applies a 1D adaptive max pooling over an input signal composed of several input planes.

    The output size is H, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size H
        return_indices: if True, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool2d . Default: False

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveMaxPool1d(5)
        >>> input = autograd.Variable(torch.randn(1, 64, 8))
        >>> output = m(input)

    """

    def __init__(self, output_size, return_indices=False):
        super(AdaptiveMaxPool1d, self).__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input):
        return F.adaptive_max_pool1d(input, self.output_size, self.return_indices)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'output_size=' + str(self.output_size) + ')'


class AdaptiveMaxPool2d(Module):
    """Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single number H for a square image H x H
        return_indices: if True, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool2d . Default: False

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveMaxPool2d((5,7))
        >>> input = autograd.Variable(torch.randn(1, 64, 8, 9))
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveMaxPool2d(7)
        >>> input = autograd.Variable(torch.randn(1, 64, 10, 9))
        >>> output = m(input)

    """

    def __init__(self, output_size, return_indices=False):
        super(AdaptiveMaxPool2d, self).__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input):
        return F.adaptive_max_pool2d(input, self.output_size, self.return_indices)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'output_size=' + str(self.output_size) + ')'


class AdaptiveAvgPool1d(Module):
    """Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    The output size is H, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size H

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveAvgPool1d(5)
        >>> input = autograd.Variable(torch.randn(1, 64, 8))
        >>> output = m(input)

    """

    def __init__(self, output_size):
        super(AdaptiveAvgPool1d, self).__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool1d(input, self.output_size)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'output_size=' + str(self.output_size) + ')'


class AdaptiveAvgPool2d(Module):
    """Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single number H for a square image H x H

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = autograd.Variable(torch.randn(1, 64, 8, 9))
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = autograd.Variable(torch.randn(1, 64, 10, 9))
        >>> output = m(input)

    """

    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool2d(input, self.output_size)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'output_size=' + str(self.output_size) + ')'
