# mypy: allow-untyped-defs
"""Gradient interface."""

import torch
from torch.nn.modules.utils import _pair, _single, _triple


def _conv_input(
    input_size,
    weight,
    grad_output,
    stride,
    padding,
    dilation,
    groups,
    output_padding,
    ntuple,
):
    """Shared body of :func:`conv1d_input`, :func:`conv2d_input` and :func:`conv3d_input`."""
    stride, padding = ntuple(stride), ntuple(padding)
    dilation, output_padding = ntuple(dilation), ntuple(output_padding)

    if not any(output_padding):
        input = grad_output.new_empty(1).expand(input_size)
        return torch.ops.aten.convolution_backward(
            grad_output,
            input,
            weight,
            None,
            stride,
            padding,
            dilation,
            False,
            [0],
            groups,
            (True, False, False),
        )[0]

    # A transposed convolution with a non-zero output_padding produces a shape no
    # forward convolution produces, so convolution_backward has nothing to check
    # input_size against and rejects it. Ask it instead for the gradient a
    # zero-padded convolution would give, which is the uncropped one, and take the
    # window the transposed convolution takes out of it: padding off the front, and
    # zeros past the end for whatever output_padding reaches beyond it.
    kernel_size = weight.shape[2:]
    uncropped = [
        (out - 1) * s + d * (k - 1) + 1
        for out, s, d, k in zip(grad_output.shape[2:], stride, dilation, kernel_size)
    ]
    input = grad_output.new_empty(1).expand(list(input_size[:2]) + uncropped)
    grad_input = torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        stride,
        (0,) * len(uncropped),
        dilation,
        False,
        [0],
        groups,
        (True, False, False),
    )[0]

    tail_pad = []
    for dim in reversed(range(len(uncropped))):
        available = uncropped[dim] - padding[dim]
        tail_pad += [0, max(0, input_size[2 + dim] - available)]
    grad_input = torch.constant_pad_nd(grad_input, tail_pad, 0)
    for dim in range(len(uncropped)):
        grad_input = grad_input.narrow(2 + dim, padding[dim], input_size[2 + dim])
    return grad_input



def conv1d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    output_padding=0,
):
    r"""Compute the gradient of conv1d with respect to the input of the convolution.

    This is same as the 1D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        output_padding (int or tuple, optional): Additional size added to one side of each
            dimension of the output shape, matching the transposed convolution argument of
            the same name. Default: 0

    Examples::

        >>> input = torch.randn(1, 1, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv1d_input(input.shape, weight, grad_output)

    """
    return _conv_input(
        input_size,
        weight,
        grad_output,
        stride,
        padding,
        dilation,
        groups,
        output_padding,
        _single,
    )


def conv1d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv1d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> # xdoctest: +SKIP
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv1d_weight(input, weight.shape, grad_output)

    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _single(stride),
        _single(padding),
        _single(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]


def conv2d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    output_padding=0,
):
    r"""Compute the gradient of conv2d with respect to the input of the convolution.

    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        output_padding (int or tuple, optional): Additional size added to one side of each
            dimension of the output shape, matching the transposed convolution argument of
            the same name. Default: 0

    Examples::

        >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv2d_input(input.shape, weight, grad_output)

    """
    return _conv_input(
        input_size,
        weight,
        grad_output,
        stride,
        padding,
        dilation,
        groups,
        output_padding,
        _pair,
    )


def conv2d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv2d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> # xdoctest: +SKIP
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv2d_weight(input, weight.shape, grad_output)

    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]


def conv3d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    output_padding=0,
):
    r"""Compute the gradient of conv3d with respect to the input of the convolution.

    This is same as the 3D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weights tensor (out_channels x in_channels/groups x kT x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        output_padding (int or tuple, optional): Additional size added to one side of each
            dimension of the output shape, matching the transposed convolution argument of
            the same name. Default: 0

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv3d_input(input.shape, weight, grad_output)

    """
    return _conv_input(
        input_size,
        weight,
        grad_output,
        stride,
        padding,
        dilation,
        groups,
        output_padding,
        _triple,
    )


def conv3d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv3d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, weight, grad_output)
        >>> F.grad.conv3d_weight(input, weight.shape, grad_output)

    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _triple(stride),
        _triple(padding),
        _triple(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]
