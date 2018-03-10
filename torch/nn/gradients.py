"""Gradient interface"""

import torch
from .modules.utils import _pair, _triple


def _output_padding(input, output_size, stride, padding, kernel_size):
    output_size = list(output_size)
    k = input.dim() - 2

    if len(output_size) == k + 2:
        output_size = output_size[-k:]
    if len(output_size) != k:
        raise ValueError("input_size must have {} elements (got {})"
                         .format(k, len(output_size)))

    def dim_size(d):
        return ((input.size(d + 2) - 1) * stride[d] - 2 * padding[d] +
                kernel_size[d])

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                ("requested an output size of {}, but valid sizes range "
                 "from {} to {} (for an out_backprop of {})").format(
                     output_size, min_sizes, max_sizes,
                     input.size()[2:]))

    return tuple([output_size[d] - min_sizes[d] for d in range(k)])


def conv2d_input(input_size, weight, out_backprop, stride=1, padding=0, dilation=1, groups=1, bias=None):
    r"""
    Computes the gradient of conv2d with respect to the input of the convolution.
    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        out_backprop : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias: optional bias tensor (out_channels). Default: None

    Examples::

        >>> input = torch.autograd.Variable(torch.randn(1,1,3,3), requires_grad=True)
        >>> weight = torch.autograd.Variable(torch.randn(1,1,1,2), requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> gradient_o = torch.autograd.Variable(torch.randn(output.shape))
        >>> gradient_i = torch.autograd.grad(output, input, gradient_o)
        >>> F.grad.conv2d_input(input.shape, weight, gradient_o)

    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel_size = (weight.shape[2], weight.shape[3])

    if input_size is None:
        raise ValueError("grad.conv2d_input requires specifying an input_size")

    grad_i_padding = _output_padding(out_backprop, input_size, stride, padding, kernel_size)

    return torch._C._VariableFunctions.conv_transpose2d(
        out_backprop, weight, bias, stride, padding, grad_i_padding,
        groups, dilation)


def conv2d_weight(input, weight_size, out_backprop, stride=1, padding=0, dilation=1, groups=1, bias=None):
    r"""
    Computes the gradient of conv2d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
        out_backprop : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias: optional bias tensor (out_channels). Default: None

    Examples::

        >>> input = torch.autograd.Variable(torch.randn(1,1,3,3), requires_grad=True)
        >>> weight = torch.autograd.Variable(torch.randn(1,1,1,2), requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> gradient_o = torch.autograd.Variable(torch.randn(output.shape))
        >>> gradient_w = torch.autograd.grad(output, filter, gradient_o)
        >>> F.grad.conv2d_weight(input, weight.shape, gradient_o)

    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    in_channels = input.shape[1]
    out_channels = out_backprop.shape[1]
    min_batch = input.shape[0]

    out_backprop_ = out_backprop.contiguous().repeat(1, in_channels // groups,
                                                     1, 1)
    out_backprop_ = out_backprop_.contiguous().view(
        out_backprop_.shape[0] * out_backprop_.shape[1], 1,
        out_backprop_.shape[2], out_backprop_.shape[3])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2], input.shape[3])

    grad_w = torch._C._VariableFunctions.conv2d(input, out_backprop_, bias,
                                                dilation, padding, stride,
                                                in_channels * min_batch)

    grad_w = grad_w.contiguous().view(min_batch, grad_w.shape[1] // min_batch,
                                      grad_w.shape[2], grad_w.shape[3])

    return grad_w.sum(dim=0).view(in_channels // groups, out_channels,
                                  grad_w.shape[2], grad_w.shape[3]).transpose(
                                      0, 1).narrow(2, 0,
                                                   weight_size[2]).narrow(
                                                       3, 0, weight_size[3])


def conv3d_input(input_size, weight, out_backprop, stride=1, padding=0, dilation=1, groups=1, bias=None):
    r"""
    Computes the gradient of conv3d with respect to the input of the convolution.
    This is same as the 3D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weights tensor (out_channels x in_channels/groups x kT x kH x kW)
        out_backprop : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias: optional bias tensor (out_channels). Default: None

    Examples::

        >>> input = torch.autograd.Variable(torch.randn(2, 8, 10, 10, 20), requires_grad=True)
        >>> weight = torch.autograd.Variable(torch.randn(4, 8, 2, 3, 3), requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> gradient_o = torch.autograd.Variable(torch.randn(output.shape))
        >>> gradient_i = torch.autograd.grad(output, input, gradient_o)
        >>> F.grad.conv3d_input(input.shape, weight, gradient_o)

    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    kernel_size = (weight.shape[2], weight.shape[3], weight.shape[4])

    if input_size is None:
        raise ValueError("grad.conv3d_input requires specifying an input_size")

    grad_i_padding = _output_padding(out_backprop, input_size, stride, padding, kernel_size)

    return torch._C._VariableFunctions.conv_transpose3d(
        out_backprop, weight, bias, stride, padding, grad_i_padding,
        groups, dilation)


def conv3d_weight(input, weight_size, out_backprop, stride=1, padding=0, dilation=1, groups=1, bias=None):
    r"""
    Computes the gradient of conv3d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight_size : Shape of the weight gradient tensor
        out_backprop : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias: optional bias tensor (out_channels). Default: None

    Examples::

        >>> input = torch.autograd.Variable(torch.randn(2, 8, 10, 10, 20), requires_grad=True)
        >>> weight = torch.autograd.Variable(torch.randn(4, 8, 2, 3, 3), requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> gradient_o = torch.autograd.Variable(torch.randn(output.shape))
        >>> gradient_w = torch.autograd.grad(output, weight, gradient_o)
        >>> F.grad.conv3d_weight(input, weight.shape, gradient_o)

    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    in_channels = input.shape[1]
    out_channels = out_backprop.shape[1]
    min_batch = input.shape[0]

    out_backprop_ = out_backprop.repeat(1, in_channels // groups, 1, 1, 1)
    out_backprop_ = out_backprop_.contiguous().view(
        out_backprop_.shape[0] * out_backprop_.shape[1], 1,
        out_backprop_.shape[2], out_backprop_.shape[3], out_backprop_.shape[4])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2], input.shape[3],
                                    input.shape[4])

    grad_w = torch._C._VariableFunctions.conv3d(input, out_backprop_, bias,
                                                dilation, padding, stride,
                                                in_channels * min_batch)

    grad_w = grad_w.contiguous().view(min_batch, grad_w.shape[1] // min_batch,
                                      grad_w.shape[2], grad_w.shape[3],
                                      grad_w.shape[4])

    return grad_w.sum(dim=0).view(
        in_channels // groups, out_channels, grad_w.shape[2], grad_w.shape[3],
        grad_w.shape[4]).transpose(0, 1).narrow(2, 0, weight_size[2]).narrow(
            3, 0, weight_size[3]).narrow(4, 0, weight_size[4])
