import torch
from torch import Tensor


def sgd_out_of_place(param, grad, weight_decay, lr, maximize) -> Tensor:
    """
    Computes a single step of SGD on a single parameter Tensor with grad.

    Assumes:
    - param and grad are the same shape and are 1D.
    - param and grad are float and on CPU

    Args:
        param: a 1D tensor of floats
        grad: a 1D tensor of floats
        weight_decay: a python double between 0 and 1
        lr: a python double

    Returns:
        a 1D float Tensor the same shape as param

    """
    return torch.ops.libtorch_agnostic.sgd_out_of_place.default(
        param, grad, weight_decay, lr, maximize
    )


def identity(t) -> Tensor:
    """
    Returns the input tensor

    Args:
        t: any Tensor

    Returns:
        a Tensor, the same as input.
    """
    return torch.ops.libtorch_agnostic.identity.default(t)


def my_abs(t) -> Tensor:
    """
    Returns abs on the input tensor, outputs a new Tensor

    Args:
        t: any Tensor

    Returns:
        a Tensor
    """
    return torch.ops.libtorch_agnostic.my_abs.default(t)
