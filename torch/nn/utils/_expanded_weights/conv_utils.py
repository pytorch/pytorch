import torch
import torch.nn.functional as F

import numpy as np
from typing import List, Optional

from .expanded_weights_utils import \
    set_grad_sample_if_exists, unpack_expanded_weight_or_tensor

THRESHOLD = 32

def conv_picker(func, conv1dOpt, conv2dOpt, conv3dOpt):
    if func == F.conv1d:
        return conv1dOpt
    if func == F.conv2d:
        return conv2dOpt
    else:
        assert func == F.conv3d
        return conv3dOpt

def conv_args_and_kwargs(kwarg_names, expanded_args_and_kwargs):
    args = expanded_args_and_kwargs[:len(expanded_args_and_kwargs) - len(kwarg_names)]
    kwargs = expanded_args_and_kwargs[len(expanded_args_and_kwargs) - len(kwarg_names):]
    kwargs = {name: arg for (name, arg) in zip(kwarg_names, kwargs)}

    return conv_normalizer(*args, **kwargs)

def conv_normalizer(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return (input, weight), {'bias': bias, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}

def conv_backward(func, ctx, grad_output):

    def weight_grad_sample(weight):
        if (batch_size < THRESHOLD and groups == 1):
            return conv_group_weight_grad_sample(ctx.input, grad_output, weight_shape, stride, padding, dilation, batch_size, func)
        else:
            return conv_unfold_weight_grad_sample(ctx.input, grad_output, weight_shape, kernel_size,
                                                  stride, padding, dilation, groups, func)

    def expand(param):
        if isinstance(param, int):
            return conv_picker(func, (param,), (param, param), (param, param, param))
        else:
            return param

    weight_shape = ctx.weight.shape
    stride, padding, dilation, groups = expand(ctx.stride), expand(ctx.padding), expand(ctx.dilation), ctx.groups

    kernel_size = []
    for i in range(2, conv_picker(func, 3, 4, 5)):
        kernel_size.append(weight_shape[i])

    batch_size = ctx.batch_size
    results: List[Optional[torch.Tensor]] = []
    results.append(None)  # for kwarg names
    results.append(None)  # for op reference

    if ctx.input_required_grad:
        output_padding = []
        input_dims = conv_picker(func, 1, 2, 3)
        for i in range(input_dims):
            input_dim = ctx.input.shape[2 + i]
            output_padding.append((2 * padding[i] + input_dim - (kernel_size[i] * dilation[i] - dilation[i] + 1)) % stride[i])
        weight_ = unpack_expanded_weight_or_tensor(ctx.weight)
        transpose_func = conv_picker(func, F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)
        results.append(transpose_func(grad_output, weight_, None, stride, padding, tuple(output_padding), groups, dilation))
    else:
        results.append(None)
    # weight and bias don't compute batched gradients; no other arguments are differentiable
    results = results + [None] * 6

    # set grad_sample field for weight and bias with per sample gradients
    set_grad_sample_if_exists(ctx.weight, weight_grad_sample)
    set_grad_sample_if_exists(ctx.bias, lambda _: grad_output.reshape(*grad_output.shape[:2], -1).sum(dim=2))
    return tuple(results)

def conv_unfold_weight_grad_sample(input, grad_output, weight_shape, kernel_size, stride, padding, dilation, groups, func):
    n = input.shape[0]
    in_channels = input.shape[1]

    unfold_func = conv_picker(
        func,
        lambda: F.unfold(input.unsqueeze(-2),
                         kernel_size=(1, kernel_size[0]),
                         dilation=(1, dilation[0]),
                         padding=(0, padding[0]),
                         stride=(1, stride[0])),
        lambda: F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride),
        lambda: unfold3d(input, kernel_size, dilation, padding, stride)
    )

    input = unfold_func()
    grad_output = grad_output.reshape(n, -1, input.shape[-1])

    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    weight_grad_sample = torch.einsum("noq,npq->nop", grad_output, input)
    # rearrange the above tensor and extract diagonals.
    weight_grad_sample = weight_grad_sample.view(
        n,
        groups,
        -1,
        groups,
        int(in_channels / groups),
        np.prod(kernel_size),
    )
    weight_grad_sample = torch.einsum("ngrg...->ngr...", weight_grad_sample).contiguous()
    shape = [n] + list(weight_shape)
    weight_grad_sample = weight_grad_sample.view(shape)
    return weight_grad_sample

def conv_group_weight_grad_sample(input, grad_output, weight_shape, stride, padding, dilation, batch_size, func):
    I = input.shape[1]
    O = grad_output.shape[1]

    input_ = input.transpose(0, 1)
    grad_output_ = grad_output.view(grad_output.shape[0] * grad_output.shape[1], 1, *grad_output.shape[2:])

    weight_grad_sample = func(input_, grad_output_, None, stride=dilation, padding=padding, dilation=stride, groups=batch_size)
    input_dims = conv_picker(func, 3, 4, 5)
    for i in range(2, input_dims):
        weight_grad_sample = weight_grad_sample.narrow(i, 0, weight_shape[i])
    weight_grad_sample = weight_grad_sample.view(I, batch_size, O, *weight_grad_sample.shape[2:])
    weight_grad_sample = weight_grad_sample.movedim(0, 2)
    return weight_grad_sample


def unfold3d(
    tensor,
    kernel_size,
    padding,
    stride,
    dilation,
):
    r"""
    Extracts sliding local blocks from an batched input tensor.
    :class:`torch.nn.Unfold` only supports 4D inputs (batched image-like tensors).
    This method implements the same action for 5D inputs
    Args:
        tensor: An input tensor of shape ``(B, C, D, H, W)``.
        kernel_size: the size of the sliding blocks
        padding: implicit zero padding to be added on both sides of input
        stride: the stride of the sliding blocks in the input spatial dimensions
        dilation: the spacing between the kernel points.
    Returns:
        A tensor of shape ``(B, C * np.product(kernel_size), L)``, where L - output spatial dimensions.
        See :class:`torch.nn.Unfold` for more details
    Example:
        >>> B, C, D, H, W = 3, 4, 5, 6, 7
        >>> tensor = torch.arange(1, B*C*D*H*W + 1.).view(B, C, D, H, W)
        >>> unfold3d(tensor, kernel_size=2, padding=0, stride=1).shape
        torch.Size([3, 32, 120])
    """

    if len(tensor.shape) != 5:
        raise ValueError(
            f"Input tensor must be of the shape [B, C, D, H, W]. Got{tensor.shape}"
        )

    if dilation != (1, 1, 1):
        raise NotImplementedError(f"dilation={dilation} not supported.")

    batch_size, channels, _, _, _ = tensor.shape

    # Input shape: (B, C, D, H, W)
    tensor = F.pad(
        tensor, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
    )
    # Output shape: (B, C, D+2*padding[2], H+2*padding[1], W+2*padding[0])

    tensor = tensor.unfold(dimension=2, size=kernel_size[0], step=stride[0])
    tensor = tensor.unfold(dimension=3, size=kernel_size[1], step=stride[1])
    tensor = tensor.unfold(dimension=4, size=kernel_size[2], step=stride[2])
    # Output shape: (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
    # For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`

    tensor = tensor.permute(0, 2, 3, 4, 1, 5, 6, 7)
    # Output shape: (B, D_out, H_out, W_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    tensor = tensor.reshape(batch_size, -1, channels * np.prod(kernel_size)).transpose(
        1, 2
    )
    # Output shape: (B, D_out * H_out * W_out, C * kernel_size[0] * kernel_size[1] * kernel_size[2]

    return tensor
