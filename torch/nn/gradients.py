"""Gradient interface"""

import torch
from .modules.utils import _pair, _triple


def _output_padding(input, output_size, stride, padding, kernel_size):
    output_size = list(output_size)
    k = input.dim() - 2
    print(k)
    print(output_size)
    if len(output_size) == k + 2:
        output_size = output_size[-k:]
    if len(output_size) != k:
        raise ValueError("output_size must have {} or {} elements (got {})"
                         .format(k, k + 2, len(output_size)))

    def dim_size(d):
        return ((input.size(d + 2) - 1) * stride[d] - 2 * padding[d] +
                kernel_size[d])

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                ("requested an output size of {}, but valid sizes range "
                 "from {} to {} (for an input of {})").format(
                     output_size, min_sizes, max_sizes,
                     input.size()[2:]))

    return tuple([output_size[d] - min_sizes[d] for d in range(k)])


def conv2d_input(in_channels, out_channels, out_backprop, weight, input_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=None):
    r"""
    Computes the gradient of conv2d with respect to the input of the convolution.
    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels in the output gradient
        out_backprop : output gradient tensor (minibatch x out_channels x oH x oW)
        weight: filters tensor (out_channels x in_channels/groups x kH x kW)
        input_size : Shape of the input gradient tensor
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias: optional bias tensor (out_channels). Default: None

    Examples::

        >>> out_backprop = autograd.Variable(torch.randn(1,1,3,2))
        >>> weight = autograd.Variable(torch.randn(1,1,1,2))
        >>> F.grad.conv2d_input(out_backprop, weight, stride=1, input_size=(1,1,3,3))
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel_size = (weight.shape[2], weight.shape[3])

    if input_size is None:
        raise ValueError("grad.conv2d_input requires specifying an input_size")

    input_gradient_padding = _output_padding(out_backprop, input_size, stride,
                                             padding, kernel_size)

    return torch._C._VariableFunctions.conv_transpose2d(
        out_backprop, weight, bias, stride, padding, input_gradient_padding,
        groups, dilation)


def conv2d_weight(in_channels, out_channels, input, out_backprop, stride=1,
                  padding=0, dilation=1, groups=1, bias=None):
    r"""
    Computes the gradient of conv2d with respect to the weight of the convolution.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels in the output gradient
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        out_backprop : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias: optional bias tensor (out_channels). Default: None

    Examples::

        >>> input = torch.autograd.Variable(torch.randn(1,1,3,3))
        >>> out_backprop = torch.autograd.Variable(torch.randn(1,1,3,2))
        >>> F.grad.conv2d_weight(1,1, input, out_backprop)
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    dims = out_backprop.shape
    min_batch = input.shape[0]

    out_backprop_ = out_backprop.repeat(1, in_channels // groups, 1, 1)
    dims = out_backprop_.shape
    out_backprop_ = out_backprop_.contiguous().view(dims[0] * dims[1], 1,
                                                    dims[2], dims[3])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2], input.shape[3])

    grad_w = torch._C._VariableFunctions.conv2d(input, out_backprop_, bias,
                                                dilation, padding, stride,
                                                in_channels * min_batch)

    grad_w = grad_w.view(min_batch, grad_w.shape[1] // min_batch,
                         grad_w.shape[2], grad_w.shape[3])

    return grad_w.sum(dim=0).view(out_channels, in_channels // groups,
                                  grad_w.shape[2], grad_w.shape[3])


def conv3d_input(in_channels, out_channels, out_backprop, weight, input_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=None):
    r"""
    Computes the gradient of conv3d with respect to the input of the convolution.
    This is same as the 3D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels in the output gradient
        out_backprop : output gradient tensor (minibatch x out_channels x oH x oW)
        weight: filters tensor (out_channels x in_channels/groups x kT x kH x kW)
        input_size : Shape of the input gradient tensor
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias: optional bias tensor (out_channels). Default: None

    Examples::

        >>> weight = torch.autograd.Variable(torch.randn(33, 16, 3, 3, 3))
        >>> out_backprop = torch.autograd.Variable(torch.randn(20, 33, 48, 8, 18))
        >>> F.grad.conv3d_input(16, 33, out_backprop, weight, input_size=(20, 16, 50, 10, 20))
    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    kernel_size = (weight.shape[2], weight.shape[3], weight.shape[4])

    if input_size is None:
        raise ValueError("grad.conv3d_input requires specifying an input_size")

    input_gradient_padding = _output_padding(out_backprop, input_size, stride,
                                             padding, kernel_size)

    return torch._C._VariableFunctions.conv_transpose3d(
        out_backprop, weight, bias, stride, padding, input_gradient_padding,
        groups, dilation)


def conv3d_weight(in_channels, out_channels, input, out_backprop, stride=1,
                  padding=0, dilation=1, groups=1, bias=None):
    r"""
    Computes the gradient of conv3d with respect to the weight of the convolution.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels in the output gradient
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        out_backprop : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias: optional bias tensor (out_channels). Default: None

    Examples::

        >>> input = torch.autograd.Variable(torch.randn(20, 16, 50, 10, 20))
        >>> out_backprop = torch.autograd.Variable(torch.randn(20, 33, 48, 8, 18))
        >>> F.grad.conv3d_weight(16, 33, input, out_backprop)
    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    dims = out_backprop.shape
    min_batch = input.shape[0]

    out_backprop_ = out_backprop.repeat(1, in_channels // groups, 1, 1, 1)
    dims = out_backprop_.shape
    out_backprop_ = out_backprop_.contiguous().view(dims[0] * dims[1], 1,
                                                    dims[2], dims[3], dims[4])

    input = input.view(1, input.shape[0] * input.shape[1], input.shape[2],
                       input.shape[3], input.shape[4])

    grad_w = torch._C._VariableFunctions.conv3d(input, out_backprop_, bias,
                                                dilation, padding, stride,
                                                in_channels * min_batch)

    grad_w = grad_w.view(min_batch, grad_w.shape[1] // min_batch,
                         grad_w.shape[2], grad_w.shape[3], grad_w.shape[4])

    return grad_w.sum(dim=0).view(out_channels, in_channels // groups,
                                  grad_w.shape[2], grad_w.shape[3],
                                  grad_w.shape[4])
