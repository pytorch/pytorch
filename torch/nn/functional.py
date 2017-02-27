"""Functional interface"""

import torch
from . import _functions
from .modules import utils
from torch.nn._functions.conv import ConvNd
from .modules.utils import _single, _pair, _triple
# Convolutions


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    """Applies a 2D convolution over an input image composed of several input
    planes.

    See :class:`~torch.nn.Conv2d` for details and output shape.

    Args:
        input: input tensor (minibatch x in_channels x iH x iW)
        weight: filters tensor (out_channels, in_channels/groups, kH, kW)
        bias: optional bias tensor (out_channels)
        stride: the stride of the convolving kernel. Can be a single number or
          a tuple (sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number or
          a tuple. Default: 0
        groups: split input into groups, in_channels should be divisible by
          the number of groups

    Examples:
        >>> # With square kernels and equal stride
        >>> filters = autograd.Variable(torch.randn(8,4,3,3))
        >>> inputs = autograd.Variable(torch.randn(1,4,5,5))
        >>> F.conv2d(inputs, filters, padding=1)
    """
    f = ConvNd(_pair(stride), _pair(padding), _pair(dilation), False,
               _pair(0), groups)
    return f(input, weight, bias) if bias is not None else f(input, weight)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    """Applies a 1D convolution over an input signal composed of several input
    planes.

    See :class:`~torch.nn.Conv1d` for details and output shape.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight: filters of shape (out_channels, in_channels, kW)
        bias: optional bias of shape (out_channels)
        stride: the stride of the convolving kernel, default 1

    Examples:
        >>> filters = autograd.Variable(torch.randn(33, 16, 3))
        >>> inputs = autograd.Variable(torch.randn(20, 16, 50))
        >>> F.conv1d(inputs, filters)
    """
    f = ConvNd(_single(stride), _single(padding), _single(dilation), False,
               _single(0), groups)
    return f(input, weight, bias) if bias is not None else f(input, weight)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    """Applies a 3D convolution over an input image composed of several input
        planes.

    See :class:`~torch.nn.Conv3d` for details and output shape.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight: filters tensor of shape (out_channels, in_channels, kT, kH, kW)
        bias: optional bias tensor of shape (out_channels)
        stride: the stride of the convolving kernel. Can be a single number or
          a tuple (st x sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number or
          a tuple. Default: 0

    Examples:
        >>> filters = autograd.Variable(torch.randn(33, 16, 3, 3, 3))
        >>> inputs = autograd.Variable(torch.randn(20, 16, 50, 10, 20))
        >>> F.conv3d(inputs, filters)
    """
    f = ConvNd(_triple(stride), _triple(padding), _triple(dilation), False,
               _triple(0), groups)
    return f(input, weight, bias) if bias is not None else f(input, weight)


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1):
    f = ConvNd(_single(stride), _single(padding), _single(1), True,
               _single(output_padding), groups)
    return f(input, weight, bias) if bias is not None else f(input, weight)


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1):
    """Applies a 2D transposed convolution operator over an input image
    composed of several input planes, sometimes also called "deconvolution".

    See :class:`~torch.nn.ConvTranspose2d` for details and output shape.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight: filters of shape (in_channels x out_channels x kH x kW)
        bias: optional bias of shape (out_channels)
        stride: the stride of the convolving kernel, a single number or a
          tuple (sh x sw). Default: 1
        padding: implicit zero padding on the input, a single number or a
          tuple (padh x padw). Default: 0
        groups: split input into groups, in_channels should be divisible by
          the number of groups
        output_padding: A zero-padding of 0 <= padding < stride that should be
          added to the output. Can be a single number or a tuple. Default: 0
    """
    f = ConvNd(_pair(stride), _pair(padding), _pair(1), True,
               _pair(output_padding), groups)
    return f(input, weight, bias) if bias is not None else f(input, weight)


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1):
    """Applies a 3D transposed convolution operator over an input image
    composed of several input planes, sometimes also called "deconvolution"

    See :class:`~torch.nn.ConvTranspose3d` for details and output shape.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight: filters of shape (in_channels x out_channels x kH x kW)
        bias: optional bias of shape (out_channels)
        stride: the stride of the convolving kernel, a single number or a
          tuple (sh x sw). Default: 1
        padding: implicit zero padding on the input, a single number or a
          tuple (padh x padw). Default: 0
    """
    f = ConvNd(_triple(stride), _triple(padding), _triple(1), True,
               _triple(output_padding), groups)
    return f(input, weight, bias) if bias is not None else f(input, weight)


# Pooling
def avg_pool1d(input, kernel_size, stride=None, padding=0,
               ceil_mode=False, count_include_pad=True):
    r"""Applies a 1D average pooling over an input signal composed of several
    input planes.

    See :class:`~torch.nn.AvgPool1d` for details and output shape.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation

    Example:
        >>> # pool of square window of size=3, stride=2
        >>> input = Variable(torch.Tensor([[[1,2,3,4,5,6,7]]]))
        >>> F.avg_pool1d(input, kernel_size=3, stride=2)
        Variable containing:
        (0 ,.,.) =
          2  4  6
        [torch.FloatTensor of size 1x1x3]
    """
    if input.dim() != 3:
        raise ValueError('expected 3D input (got {} dimensions)'
                         .format(input.dim()))
    kernel_size = _single(kernel_size) + (1,)
    stride = _single(stride) + (1,) if stride is not None else kernel_size
    padding = _single(padding) + (0,)
    f = _functions.thnn.AvgPool2d(kernel_size, stride, padding,
                                  ceil_mode, count_include_pad)
    return f(input.unsqueeze(3)).squeeze(3)


def avg_pool2d(input, kernel_size, stride=None, padding=0,
               ceil_mode=False, count_include_pad=True):
    """Applies 2D average-pooling operation in kh x kw regions by step size
    dh x dw steps. The number of output features is equal to the number of
    input planes.

    See :class:`~torch.nn.AvgPool2d` for details and output shape.

    Args:
        input: input tensor (minibatch x in_channels x iH x iW)
        kernel_size: size of the pooling region, a single number or a
          tuple (kh x kw)
        stride: stride of the pooling operation, a single number or a
          tuple (sh x sw). Default is equal to kernel size
        padding: implicit zero padding on the input, a single number or
          a tuple (padh x padw), Default: 0
        ceil_mode: operation that defines spatial output shape
        count_include_pad: divide by the number of elements inside the
          original non-padded image or kh * kw
    """
    return _functions.thnn.AvgPool2d(kernel_size, stride, padding,
                                     ceil_mode, count_include_pad)(input)


def avg_pool3d(input, kernel_size, stride=None):
    """Applies 3D average-pooling operation in kt x kh x kw regions by step
    size kt x dh x dw steps. The number of output features is equal to the
    number of input planes / dt.
    """
    return _functions.thnn.AvgPool3d(kernel_size, stride)(input)


# share the same interface
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    return _functions.thnn.MaxPool1d(kernel_size, stride, padding, dilation,
                                     return_indices, ceil_mode)(input)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    return _functions.thnn.MaxPool2d(kernel_size, stride, padding, dilation,
                                     return_indices, ceil_mode)(input)


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    return _functions.thnn.MaxPool3d(kernel_size, stride, padding, dilation,
                                     return_indices, ceil_mode)(input)


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


def max_unpool1d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    kernel_size = _single(kernel_size)
    stride = _single(stride)
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    f = _functions.thnn.MaxUnpool2d(output_size + [1])
    return f(input.unsqueeze(3), indices.unsqueeze(3)).squeeze(3)


def max_unpool2d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    f = _functions.thnn.MaxUnpool2d(output_size)
    return f(input, indices)


def max_unpool3d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    f = _functions.thnn.MaxUnpool3d(output_size, stride, padding)
    return f(input, indices)


def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    kw, kh = utils._pair(kernel_size)
    out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    return out.mul(kw * kh).pow(1. / norm_type)


# Activation functions

def dropout(input, p=0.5, training=False, inplace=False):
    return _functions.dropout.Dropout(p, training, inplace)(input)


def threshold(input, threshold, value, inplace=False):
    return _functions.thnn.auto.Threshold(threshold, value, inplace)(input)


def relu(input, inplace=False):
    return _functions.thnn.auto.Threshold(0, 0, inplace)(input)


def hardtanh(input, min_val=-1., max_val=1., inplace=False):
    return _functions.thnn.auto.Hardtanh(min_val, max_val, inplace)(input)


def relu6(input, inplace=False):
    return _functions.thnn.auto.Hardtanh(0, 6, inplace)(input)


def elu(input, alpha=1., inplace=False):
    return _functions.thnn.auto.ELU(alpha, inplace)(input)


def leaky_relu(input, negative_slope=1e-2, inplace=False):
    return _functions.thnn.auto.LeakyReLU(negative_slope, inplace)(input)


def prelu(input, weight):
    return _functions.thnn.PReLU()(input, weight)


def rrelu(input, lower=1. / 8, upper=1. / 3, training=False, inplace=False):
    return _functions.thnn.RReLU(lower, upper, training, inplace)(input)


def logsigmoid(input):
    return _functions.thnn.LogSigmoid()(input)


def hardshrink(input, lambd=0.5):
    return _functions.thnn.auto.Hardshrink(lambd)(input)


def tanhshrink(input):
    return input - torch.tanh(input)


def softsign(input):
    return _functions.activation.Softsign()(input)


def softplus(input, beta=1, threshold=20):
    return _functions.thnn.auto.Softplus(beta, threshold)(input)


def softmin(input):
    return _functions.thnn.Softmin()(input)


def softmax(input):
    return _functions.thnn.auto.Softmax()(input)


def softshrink(input, lambd=0.5):
    return _functions.thnn.auto.Softshrink(lambd)(input)


def log_softmax(input):
    return _functions.thnn.LogSoftmax()(input)


def tanh(input):
    return torch.tanh(input)


def sigmoid(input):
    return torch.sigmoid(input)


# etc.

def linear(input, weight, bias=None):
    state = _functions.linear.Linear()
    return bias and state(input, weight, bias) or state(input, weight)


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    state = _functions.batchnorm.BatchNorm(
        running_mean, running_var, training, momentum, eps)
    return weight and state(input, weight, bias) or state(input)


# loss

def nll_loss(input, target, weight=None, size_average=True):
    r"""The negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes`
        target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`
        weight (Variable, optional): a manual rescaling weight given to each
                class. If given, has to be a Variable of size "nclasses"
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch.

    Attributes:
        weight: the class-weights given as input to the constructor

    Example:
        >>> # input is of size nBatch x nClasses = 3 x 5
        >>> input = autograd.Variable(torch.randn(3, 5))
        >>> # each element in target has to have 0 <= value < nclasses
        >>> target = autograd.Variable(torch.LongTensor([1, 0, 4]))
        >>> output = F.nll_loss(F.log_softmax(input), target)
        >>> output.backward()
    """
    return _functions.thnn.NLLLoss(size_average, weight=weight)(input, target)


def kl_div(input, target, size_average=True):
    r"""The `Kullback-Leibler divergence`_ Loss.

    See :class:`~torch.nn.KLDivLoss` for details.

    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        size_average: if True the output is divided by the number of elements
          in input tensor
    """
    return _functions.thnn.KLDivLoss(size_average)(input, target)


def cross_entropy(input, target, weight=None, size_average=True):
    r"""This criterion combines `log_softmax` and `nll_loss` in one single class.

    See :class:`torch.nn.CrossEntropyLoss` for details.

    Args:
        input: Variable :math:`(N, C)` where `C = number of classes`
        target: Variable :math:`(N)` where each value is `0 <= targets[i] <= C-1`
        weight (Tensor, optional): a manual rescaling weight given to each
                class. If given, has to be a Tensor of size "nclasses"
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch.
    """
    return nll_loss(log_softmax(input), target, weight, size_average)


def binary_cross_entropy(input, target, weight=None, size_average=True):
    r"""Function that measures the Binary Cross Entropy
    between the target and the output:

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch.
    """
    return _functions.thnn.BCELoss(size_average, weight=weight)(input, target)


def smooth_l1_loss(input, target, size_average=True):
    return _functions.thnn.SmoothL1Loss(size_average)(input, target)


def pixel_shuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C*r^2, H, W]`` to a
    tensor of shape ``[C, H*r, W*r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> ps = nn.PixelShuffle(3)
        >>> input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])
    """
    batch_size, channels, in_height, in_width = input.size()
    channels //= upscale_factor ** 2

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, upscale_factor, upscale_factor,
        in_height, in_width)

    shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


def upsample_nearest(input, size=None, scale_factor=None):
    """Upsamples the input, using nearest neighbours' pixel values.

    Currently only spatial upsampling is supported (i.e. expected inputs
    are 4 dimensional).

    Args:
        input (Variable): input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
    """
    return _functions.thnn.UpsamplingNearest2d(size, scale_factor)(input)


def upsample_bilinear(input, size=None, scale_factor=None):
    """Upscales the input, using the bilinear upsampling.

    Currently only spatial upsampling is supported (i.e. expected inputs
    are 4 dimensional).

    Args:
        input (Variable): input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
    """
    return _functions.thnn.UpsamplingBilinear2d(size, scale_factor)(input)
