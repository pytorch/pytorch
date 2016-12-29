import math
import torch
from torch.nn.parameter import Parameter
from .. import functional as F
from .module import Module
from .utils import _single, _pair, _triple


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv1d(_ConvNd):
    """Applies a 1D convolution over an input signal composed of several input
    planes.

    ```
    The output value of the layer with input (b x iC x W) and output (b x oC x oW)
    can be precisely described as:
    output[b_i][oc_i][w_i] = bias[oc_i]
                + sum_iC sum_{ow = 0, oW-1} sum_{kw = 0 to kW-1}
                    weight[oc_i][ic_i][kw] * input[b_i][ic_i][stride_w * ow + kw)]
    ```

    Note that depending of the size of your kernel, several (of the last)
    columns of the input might be lost. It is up to the user
    to add proper padding.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel.
        stride: the stride of the convolving kernel.
    Input Shape: [ * , in_channels  , * ] : Input is minibatch x in_channels x iW
    Output Shape:[ * , out_channels , * ]  : Output shape is precisely minibatch x out_channels x floor((iW  + 2*padW - kW) / dW + 1)
    Members:
        weight: the learnable weights of the module of shape (out_channels x in_channels x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50))
        >>> output = m(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv2d(_ConvNd):
    """Applies a 2D convolution over an input image composed of several input
    planes.

    ```
    The output value of the layer with input (b x iC x H x W) and output (b x oC x oH x oW)
    can be precisely described as:
    output[b_i][oc_i][h_i][w_i] = bias[oc_i]
                + sum_iC sum_{oh = 0, oH-1} sum_{ow = 0, oW-1} sum_{kh = 0 to kH-1} sum_{kw = 0 to kW-1}
                    weight[oc_i][ic_i][kh][kw] * input[b_i][ic_i][stride_h * oh + kh)][stride_w * ow + kw)]
    ```

    Note that depending of the size of your kernel, several (of the last)
    columns or rows of the input image might be lost. It is up to the user
    to add proper padding in images.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        stride: the stride of the convolving kernel. Can be a single number s or a tuple (sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number s or a tuple. Default: 0
        dilation: If given, will do dilated (or atrous) convolutions. Can be a single number s or a tuple. Default: None
        bias: If set to False, the layer will not learn an additive bias. Default: True
    Input Shape: [ * , in_channels  , * , * ] : Input is minibatch x in_channels x iH x iW
    Output Shape:[ * , out_channels , * , * ]  : Output shape is precisely minibatch x out_channels x floor((iH  + 2*padH - kH) / dH + 1) x floor((iW  + 2*padW - kW) / dW + 1)
    Members:
        weight: the learnable weights of the module of shape (out_channels x in_channels x kH x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv3d(_ConvNd):
    """Applies a 3D convolution over an input image composed of several input
    planes.

    Note that depending of the size of your kernel, several (of the last)
    columns or rows of the input image might be lost. It is up to the user
    to add proper padding in images.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
        stride: the stride of the convolving kernel. Can be a single number s or a tuple (kt x sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number s or a tuple. Default: 0
    Input Shape: [ * , in_channels  , * , * , * ] : Input is minibatch x in_channels x iT x iH x iW
    Output Shape:[ * , out_channels , * , * , * ]  : Output shape is precisely minibatch x out_channels x floor((iT  + 2*padT - kT) / dT + 1) x floor((iH  + 2*padH - kH) / dH + 1) x floor((iW  + 2*padW - kW) / dW + 1)
    Members:
        weight: the learnable weights of the module of shape (out_channels x in_channels x kT x kH x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias)

    def forward(self, input):
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class _ConvTransposeMixin(object):
    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        func = self._backend.ConvNd(
            self.stride, self.padding, self.dilation, self.transposed,
            output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])


class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    """Applies a 2D deconvolution operator over an input image composed of several input
    planes.
    The deconvolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.
    This module can be seen as the exact reverse of the Conv2d module.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        stride: the stride of the convolving kernel. Can be a single number or a tuple (sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number or a tuple. Default: 0
        output_padding: A zero-padding of 0 <= padding < stride that should be added to the output. Can be a single number or a tuple. Default: 0
        bias: If set to False, the layer will not learn an additive bias. Default: True
    Input Shape: [ * , in_channels  , * , * ] : Input is minibatch x in_channels x iH x iW
    Output Shape:[ * , out_channels , * , * ]  : Output shape is minibatch x out_channels x (iH - 1) * sH - 2*padH + kH + output_paddingH x (iW - 1) * sW - 2*padW + kW, or as specified in a second argument to the call.
    Members:
        weight: the learnable weights of the module of shape (in_channels x out_channels x kH x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> output = upsample(h, output_size=input.size())
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(1)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups)


class ConvTranspose3d(_ConvTransposeMixin, _ConvNd):
    """Applies a 3D deconvolution operator over an input image composed of several input
    planes.
    The deconvolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.
    This module can be seen as the exact reverse of the Conv3d module.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
        stride: the stride of the convolving kernel. Can be a single number or a tuple (st x sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number or a tuple. Default: 0
        output_padding: A zero-padding of 0 <= padding < stride that should be added to the output. Can be a single number or a tuple. Default: 0
    Input Shape: [ * , in_channels  , * , * , * ] : Input is minibatch x in_channels x iH x iW
    Output Shape:[ * , out_channels , * , * , * ]  : Output shape is precisely minibatch x out_channels x (iT - 1) * sT - 2*padT + kT + output_paddingT x (iH - 1) * sH - 2*padH + kH + output_paddingH x (iW - 1) * sW - 2*padW + kW
    Members:
        weight: the learnable weights of the module of shape (in_channels x out_channels x kT x kH x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(1)
        output_padding = _triple(output_padding)
        super(ConvTranspose3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose3d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups)


# TODO: Conv2dLocal
# TODO: Conv2dMap
# TODO: ConvTranspose2dMap
