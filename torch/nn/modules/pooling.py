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
        padding_str=', padding=(' + str(padh) + ', ' + str(padw) + ')' \
                      if padh != 0 and padw !=0 else ''
        dilation_str=(', dilation=(' + str(dilh) + ', ' + str(dilw) + ')' \
                        if dilh != 0 and dilw != 0 else '')
        return  self.__class__.__name__ + ' (' \
            + 'size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ')'



class MaxUnpool2d(Module):
    """Computes the inverse operation of `MaxPool2d`
    `MaxPool2d` is not invertible, as the locations of the max locations are lost.
    :func:`MaxUnpool2d` takes in as input the output of `MaxPool2d` and the indices of the max locations
    and computes the inverse.

    Args:
        kernel_size: the size of the max window.
                     Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        stride: the stride of the window. Can be a single number s or a tuple (sh x sw). Default: kernel_size
        padding: implicit padding that was added to the input. Can be a single number or a tuple. Default: 0

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = padding[0] * (H_{in} - 1) * stride[0] + kernel_size[0]`
          :math:`W_{out} = padding[1] * (W_{in} - 1) * stride[1] + kernel_size[1]`
          or as given by :attr:`output_size` in the call operator

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(2, stride=2, return_indices = True)
        >>> mu = nn.MaxUnpool2d(2, stride=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output, indices = m(input)
        >>> unpooled_output = mu.forward(output, indices)
        >>> # exact output size can be also specified as an argument
        >>> input = autograd.Variable(torch.randn(1, 16, 11, 11))
        >>> downsample = nn.MaxPool2d(3, 3, return_indices=True)
        >>> upsample = nn.MaxUnpool2d(3, 3)
        >>> h, indices = downsample(input)
        >>> output = upsample(h, indices, output_size=input.size())

    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool2d, self).__init__()
        self.kh, self.kw = _pair(kernel_size)
        self.dh, self.dw = _pair(stride or kernel_size)
        self.padh, self.padw = _pair(padding)

    def forward(self, input, indices, output_size=None):
        out_height = (input.size(2) - 1) * self.dh + self.kh - 2*self.padh
        out_width = (input.size(3) - 1) * self.dw + self.kw - 2*self.padw
        if output_size:
            output_size = list(output_size)
            if len(output_size) == 4:
                output_size = output_size[-2:]
            if len(output_size) != 2:
                raise ValueError("output_size should be a sequence containing "
                        "2 or 4 elements, but it has a length of {}".format(
                            len(output_size)))
            h, w = output_size
            h_ok = out_height - self.dh < h < out_height + self.dh
            w_ok = out_width - self.dw < w < out_width + self.dw
            if not h_ok or not w_ok:
                raise ValueError(("specified incorrect output size. Got {}x{}, "
                        "but valid sizes range from {}x{} to {}x{}").format(
                            h, w,
                            out_height - self.dh + 1, out_width - self.dw + 1,
                            out_height + self.dh - 1, out_width + self.dw - 1))
            out_height, out_width = h, w
        return self._backend.MaxUnpool2d(out_width,
                out_height)(input, indices)


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

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the width dimension and the third `int` for the width dimension

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

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the width dimension and the third `int` for the width dimension

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
        kwargs = {}
        if self.outh is not None:
            kwargs['output_size'] = self.outh, self.outw
        else:
            kwargs['output_ratio'] = self.rh, self.rw
        func = self._backend.FractionalMaxPool2d(self.kw, self.kh,
                return_indices=self.return_indices,
                _random_samples=self._random_samples, **kwargs)
        return func(input)


class MaxUnpool3d(Module):
    """Computes the inverse operation of `MaxPool3d`
    `MaxPool3d` is not invertible, as the locations of the max locations are lost.
    :func:`MaxUnpool3d` takes in as input the output of `MaxPool3d` and the indices of the max locations
    and computes the inverse.

    Args:
        kernel_size: the size of the max window.
                     Can be a single number k (for a square kernel of k x k x k) or a tuple (kd x kh x kw)
        stride: the stride of the window. Can be a single number s or a tuple (sd x sh x sw). Default: kernel_size
        padding: implicit padding that was added to the input. Can be a single number or a tuple. Default: 0

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = padding[0] * (D_{in} - 1) * stride[0] + kernel_size[0]`
          :math:`H_{out} = padding[1] * (H_{in} - 1) * stride[1] + kernel_size[1]`
          :math:`W_{out} = padding[2] * (W_{in} - 1) * stride[2] + kernel_size[2]`
          or as given by :attr:`output_size` in the call operator

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2, return_indices = True)
        >>> mu = nn.MaxUnpool3d(3, stride=2)
        >>> input, indices = autograd.Variable(torch.randn(20, 16, 50, 32, 15))
        >>> output = m(input)
        >>> unpooled_output = m2.forward(output, indices)

    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool3d, self).__init__()
        self.kt, self.kh, self.kw = _triple(kernel_size)
        self.dt, self.dh, self.dw = _triple(stride or kernel_size)
        self.padt, self.padh, self.padw = _triple(padding)

    def forward(self, input, indices):
        out_depth = (input.size(2) - 1) * self.dt + self.kt - 2*self.padt
        out_height = (input.size(3) - 1) * self.dh + self.kh - 2*self.padh
        out_width = (input.size(4) - 1) * self.dw + self.kw - 2*self.padw
        return self._backend.MaxUnpool3d(out_depth, out_width, out_height,
                self.dt, self.dw, self.dh,
                self.padt, self.padw, self.padh)(input, indices)


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


# TODO: AdaptiveMaxPool2d
