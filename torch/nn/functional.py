"""Functional interface"""

import torch
from . import functions
from .modules import utils
from torch.nn.functions.conv import Conv, ConvTranspose
# Convolutions


def conv2d(input, weight, bias=None, stride=1, padding=0, groups=1):
    """Applies a 2D convolution over an input image composed of several input
    planes.

    ```
    The output value of the layer with input (b x iC x H x W) and filters
    (oC x iC x kH x kW) can be precisely described as:
    output[b_i][oc_i][h_i][w_i] = bias[oc_i]
                + sum_iC sum_{oh = 0, oH-1} sum_{ow = 0, oW-1} \
                                    sum_{kh = 0 to kH-1} sum_{kw = 0 to kW-1}
                   weight[oc_i][ic_i][kh][kw]
                   * input[b_i][ic_i][stride_h * oh + kh)][stride_w * ow + kw)]
    ```

    Note that depending of the size of your kernel, several (of the last)
    columns or rows of the input image might be lost. It is up to the user
    to add proper padding in images.

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

    Output Shape: [ * , out_channels , * , * ]  : Output shape is precisely
                        minibatch
                        x out_channels
                        x floor((iH  + 2*padH - kH) / dH + 1)
                        x floor((iW  + 2*padW - kW) / dW + 1)
    Examples:
        >>> # With square kernels and equal stride
        >>> filters = autograd.Variable(torch.randn(8,4,3,3))
        >>> inputs = autograd.Variable(torch.randn(1,4,5,5))
        >>> F.conv2d(inputs, filters, padding=1)
    """
    f = Conv(2, stride, padding, groups)
    return f(input, weight, bias) if bias is not None else f(input, weight)


def conv1d(input, weight, bias=None, stride=1):
    """Applies a 1D convolution over an input signal composed of several input
    planes.

    ```
    The output value of the layer with input (b x iC x W) and filters
    (oC x oC x kw) can be precisely described as:
    output[b_i][oc_i][w_i] = bias[oc_i]
               + sum_iC sum_{ow = 0, oW-1} sum_{kw = 0 to kW-1}
                 weight[oc_i][ic_i][kw] * input[b_i][ic_i][stride_w * ow + kw)]
    ```

    Note that depending on the size of your kernel, several (of the last)
    columns of the input might be lost. It is up to the user
    to add proper padding.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight: filters of shape (out_channels, in_channels, kW)
        bias: optional bias of shape (out_channels)
        stride: the stride of the convolving kernel, default 1
    Output Shape:[ * , out_channels , * ]  : Output shape is precisely
                 minibatch
                 x out_channels
                 x floor((iW  + 2*padW - kW) / dW + 1)
    Examples:
        >>> filters = autograd.Variable(torch.randn(33, 16, 3))
        >>> inputs = autograd.Variable(torch.randn(20, 16, 50))
        >>> F.conv1d(inputs)
    """
    return conv2d(input.unsqueeze(2), weight.unsqueeze(2), bias,
                  (1, stride)).squeeze(2)


def conv3d(input, weight, bias=None, stride=1, padding=0):
    """Applies a 3D convolution over an input image composed of several input
    planes.


    Note that depending of the size of your kernel, several (of the last)
    columns or rows of the input image might be lost. It is up to the user
    to add proper padding in images.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight: filters tensor of shape (out_channels, in_channels, kT, kH, kW)
        bias: optional bias tensor of shape (out_channels)
        stride: the stride of the convolving kernel. Can be a single number or
          a tuple (st x sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number or
          a tuple. Default: 0
    Output Shape:[ * , out_channels , * , * , * ]  : Output shape is precisely
                minibatch
                x out_channels x floor((iT  + 2*padT - kT) / dT + 1)
                x floor((iH  + 2*padH - kH) / dH + 1)
                x floor((iW  + 2*padW - kW) / dW + 1)

    Examples:
        >>> filters = autograd.Variable(torch.randn(33, 16, 3, 3, 3))
        >>> inputs = autograd.Variable(torch.randn(20, 16, 50, 10, 20))
        >>> F.conv3d(inputs)
    """
    f = Conv(3, stride, padding)
    return f(input, weight, bias) if bias is not None else f(input, weight)


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, groups=1,
                     output_padding=0):
    """Applies a 2D transposed convolution operator over an input image
    composed of several input planes, sometimes also called "deconvolution"
    The operator multiplies each input value element-wise by a
    kernel, and sums over the outputs from all input feature planes.
    This module can be seen as the exact reverse of the conv2d function

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
    Output Shape:[ * , out_channels , * , * ]  : Output shape is
                        minibatch x
                        out_channels x
                        (iH - 1) * sH - 2*padH + kH + output_paddingH x
                        (iW - 1) * sW - 2*padW + kW + output_paddingW
    Examples:
        >>> #TODO
    """
    f = ConvTranspose(2, stride, padding, groups, output_padding)
    return f(input, weight, bias) if bias is not None else f(input, weight)


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0):
    """Applies a 3D transposed convolution operator over an input image
    composed of several input planes, sometimes also called "deconvolution"
    The operator multiplies each input value element-wise by a
    kernel, and sums over the outputs from all input feature planes.
    This module can be seen as the exact reverse of the conv3d function

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight: filters of shape (in_channels x out_channels x kH x kW)
        bias: optional bias of shape (out_channels)
        stride: the stride of the convolving kernel, a single number or a
          tuple (sh x sw). Default: 1
        padding: implicit zero padding on the input, a single number or a
          tuple (padh x padw). Default: 0
    Output Shape:[ * , out_channels , * , * , * ]  : Output shape is precisely
                        minibatch
                        x out_channels
                        x (iT - 1) * sT - 2*padT + kT
                        x (iH - 1) * sH - 2*padH + kH
                        x (iW - 1) * sW - 2*padW + kW
    Examples:
        >>> #TODO
    """
    f = ConvTranspose(3, stride, padding)
    return f(input, weight, bias) if bias is not None else f(input, weight)


# Pooling

def avg_pool2d(input, kernel_size, stride=None, padding=0,
               ceil_mode=False, count_include_pad=True):
    """Applies 2D average-pooling operation in kh x kw regions by step size
    dh x dw steps. The number of output features is equal to the number of
    input planes.

    By default, the output of each pooling region is divided by the number of
    elements inside the padded image (which is usually kh x kw, except in some
    corner cases in which it can be smaller). You can also divide by the number
    of elements inside the original non-padded image.
    To switch between different division factors, set count_include_pad to
    True or False. If padW=padH=0, both options give the same results.

    Args:
        :param input: input tensor (minibatch x in_channels x iH x iW)
        :param kernel_size: size of the pooling region, a single number or a
          tuple (kh x kw)
        :param stride: stride of the pooling operation, a single number or a
          tuple (sh x sw). Default is equal to kernel size
        :param padding: implicit zero padding on the input, a single number or
          a tuple (padh x padw), Default: 0
        :param ceil_mode: operation that defines spatial output shape
        :param count_include_pad: divide by the number of elements inside the
          original non-padded image or kh * kw
        :return: output tensor of shape
    Output Shape: [ * , in_channels, * , * ] : Output shape is precisely
                        minibatch
                        x in_channels
                        x op((iH + 2*padh - kh) / dh + 1)
                        x op((iW + 2*padw - kw) / dw + 1)
    Examples:
        >>> #TODO
    """
    return functions.thnn.AvgPool2d(kernel_size, stride, padding,
                                    ceil_mode, count_include_pad)(input)


def avg_pool3d(input, kernel_size, stride=None):
    """Applies 3D average-pooling operation in kt x kh x kw regions by step
    size kt x dh x dw steps. The number of output features is equal to the
    number of input planes / dt.

    Args:
        :param input:
        :param kernel_size:
        :param stride:
        :return:
    Examples:
        >>> #TODO
    """
    return functions.thnn.AvgPool3d(kernel_size, stride)(input)


# share the same interface
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    return functions.thnn.MaxPool1d(kernel_size, stride, padding, dilation,
                                    return_indices, ceil_mode)(input)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    return functions.thnn.MaxPool2d(kernel_size, stride, padding, dilation,
                                    return_indices, ceil_mode)(input)


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    return functions.thnn.MaxPool3d(kernel_size, stride, padding, dilation,
                                    return_indices, ceil_mode)(input)


def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    kw, kh = utils._pair(kernel_size)
    out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    return out.mul(kw * kh).pow(1./norm_type)


# Activation functions

def dropout(input, p=0.5, training=False, inplace=False):
    return functions.dropout.Dropout(p, training, inplace)(input)


def threshold(input, threshold, value, inplace=False):
    return functions.thnn.auto.Threshold(threshold, value, inplace)(input)


def relu(input, inplace=False):
    return functions.thnn.auto.Threshold(0, 0, inplace)(input)


def hardtanh(input, min_val=-1., max_val=1., inplace=False):
    return functions.thnn.auto.Hardtanh(min_val, max_val, inplace)(input)


def relu6(input, inplace=False):
    return functions.thnn.auto.Hardtanh(0, 6, inplace)(input)


def elu(input, alpha=1., inplace=False):
    return functions.thnn.auto.ELU(alpha, inplace)(input)


def leaky_relu(input, negative_slope=1e-2, inplace=False):
    return functions.thnn.auto.LeakyReLU(negative_slope, inplace)(input)


def prelu(input, weight):
    return functions.thnn.PReLU()(input, weight)


def rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False):
    return functions.thnn.RReLU(lower, upper, training, inplace)(input)


def logsigmoid(input):
    return functions.thnn.LogSigmoid()(input)


def hardshrink(input, lambd=0.5):
    return functions.thnn.auto.Hardshrink(lambd)(input)


def tanhshrink(input):
    return input - torch.tanh(input)


def softsign(input):
    return functions.activation.Softsign()(input)


def softplus(input, beta=1, threshold=20):
    return functions.thnn.auto.Softplus(beta, threshold)(input)


def softmin(input):
    return functions.thnn.Softmin()(input)


def softmax(input):
    return functions.thnn.auto.Softmax()(input)


def softshrink(input, lambd=0.5):
    return functions.thnn.auto.Softshrink(lambd)(input)


def log_softmax(input):
    return functions.thnn.LogSoftmax()(input)


def tanh(input):
    return torch.tanh(input)


def sigmoid(input):
    return torch.sigmoid(input)


# etc.

def linear(input, weight, bias=None):
    state = functions.linear.Linear()
    return bias and state(input, weight, bias) or state(input, weight)


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    state = functions.thnn.BatchNorm(running_mean, running_var,
                                     training, momentum, eps)
    return weight and state(input, weight, bias) or state(input)


def nll_loss(input, target, weight=None, size_average=True):
    return functions.thnn.NLLLoss(size_average, weight=weight)(input, target)


def cross_entropy(input, target, weight=None, size_average=True):
    return nll_loss(log_softmax(input), target, weight, size_average)


def binary_cross_entropy(input, target, weight=None, size_average=True):
    return functions.thnn.BCELoss(size_average, weight=weight)(input, target)
