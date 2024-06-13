from __future__ import annotations

import copy
from typing import Optional, Tuple, TypeVar

import torch


__all__ = [
    "fuse_conv_bn_eval",
    "fuse_conv_bn_weights",
    "fuse_linear_bn_eval",
    "fuse_linear_bn_weights",
]

ConvT = TypeVar("ConvT", bound="torch.nn.modules.conv._ConvNd")
LinearT = TypeVar("LinearT", bound="torch.nn.Linear")


def fuse_conv_bn_eval(
    conv: ConvT,
    bn: torch.nn.modules.batchnorm._BatchNorm,
    transpose: bool = False,
) -> ConvT:
    r"""Fuse a convolutional module and a BatchNorm module into a single, new convolutional module.

    Args:
        conv (torch.nn.modules.conv._ConvNd): A convolutional module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.
        transpose (bool, optional): If True, transpose the convolutional weight. Defaults to False.

    Returns:
        torch.nn.modules.conv._ConvNd: The fused convolutional module.

    .. note::
        Both ``conv`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    assert not (conv.training or bn.training), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    assert bn.running_mean is not None and bn.running_var is not None
    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        transpose,
    )

    return fused_conv


def fuse_conv_bn_weights(
    conv_w: torch.Tensor,
    conv_b: Optional[torch.Tensor],
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: Optional[torch.Tensor],
    bn_b: Optional[torch.Tensor],
    transpose: bool = False,
) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    r"""Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.

    Args:
        conv_w (torch.Tensor): Convolutional weight.
        conv_b (Optional[torch.Tensor]): Convolutional bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (Optional[torch.Tensor]): BatchNorm weight.
        bn_b (Optional[torch.Tensor]): BatchNorm bias.
        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused convolutional weight and bias.
    """
    conv_weight_dtype = conv_w.dtype
    conv_bias_dtype = conv_b.dtype if conv_b is not None else conv_weight_dtype
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape)).to(
        dtype=conv_weight_dtype
    )
    fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(
        dtype=conv_bias_dtype
    )

    return (
        torch.nn.Parameter(fused_conv_w, conv_w.requires_grad),
        torch.nn.Parameter(fused_conv_b, conv_b.requires_grad),
    )


def fuse_linear_bn_eval(
    linear: LinearT,
    bn: torch.nn.modules.batchnorm._BatchNorm,
) -> LinearT:
    r"""Fuse a linear module and a BatchNorm module into a single, new linear module.

    Args:
        linear (torch.nn.Linear): A Linear module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.

    Returns:
        torch.nn.Linear: The fused linear module.

    .. note::
        Both ``linear`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    assert not (linear.training or bn.training), "Fusion only for eval!"
    fused_linear = copy.deepcopy(linear)

    """
    Linear-BN needs to be fused while preserving the shapes of linear weight/bias.
    To preserve the shapes of linear weight/bias, the channel dim of bn needs to be broadcastable with the last dim of linear,
    because bn operates over the channel dim, (N, C_in, H, W) while linear operates over the last dim, (*, H_in).
    To be broadcastable, the number of features in bn and
    the number of output features from linear must satisfy the following condition:
    1. they are equal, or
    2. the number of features in bn is 1
    Otherwise, skip the folding path
    """
    assert (
        linear.out_features == bn.num_features or bn.num_features == 1
    ), "To fuse, linear.out_features == bn.num_features or bn.num_features == 1"

    assert bn.running_mean is not None and bn.running_var is not None
    fused_linear.weight, fused_linear.bias = fuse_linear_bn_weights(
        fused_linear.weight,
        fused_linear.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    return fused_linear


def fuse_linear_bn_weights(
    linear_w: torch.Tensor,
    linear_b: Optional[torch.Tensor],
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    r"""Fuse linear module parameters and BatchNorm module parameters into new linear module parameters.

    Args:
        linear_w (torch.Tensor): Linear weight.
        linear_b (Optional[torch.Tensor]): Linear bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (torch.Tensor): BatchNorm weight.
        bn_b (torch.Tensor): BatchNorm bias.
        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused linear weight and bias.
    """
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)

    fused_w = linear_w * bn_scale.unsqueeze(-1)
    fused_b = (linear_b - bn_rm) * bn_scale + bn_b

    return torch.nn.Parameter(fused_w, linear_w.requires_grad), torch.nn.Parameter(
        fused_b, linear_b.requires_grad
    )
