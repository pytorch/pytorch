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


def my_ones_like(tensor, device) -> Tensor:
    """
    Returns a new Tensor like the input tensor, but with all ones

    Args:
        tensor: any Tensor
        device: a device string

    Returns:
        a ones Tensor with the same dtype and shape and other attributes
        like the input tensor
    """
    return torch.ops.libtorch_agnostic.my_ones_like.default(tensor, device)


def exp_neg_is_leaf(t1, t2, t3) -> tuple[Tensor, Tensor, bool]:
    """
    Returns a Tensor, Tensor, bool tuple corresponding to the respective inputs
    t1, t2, and t3.

    Args:
        t1: Tensor
        t2: Tensor
        t3: Tensor

    Returns:
        (exp(t1), neg(t2), is_leaf(t3))
    """
    return torch.ops.libtorch_agnostic.exp_neg_is_leaf.default(t1, t2, t3)


def neg_exp(t) -> Tensor:
    """
    Returns a Tensor composing neg of exp

    Args:
        t: Tensor

    Returns: neg(exp(t))
    """
    return torch.ops.libtorch_agnostic.neg_exp.default(t)


def divide_neg_exp(t) -> Tensor:
    """
    Returns a Tensor division of neg and exp

    Args:
        t: Tensor

    Returns: divide(neg(t), exp(t))
    """
    return torch.ops.libtorch_agnostic.divide_neg_exp.default(t)


def is_contiguous(t) -> bool:
    """
    Returns a bool indicating if the input tensor is contiguous

    Args:
        t: Tensor

    Returns: is_contiguous(t)
    """
    return torch.ops.libtorch_agnostic.is_contiguous.default(t)


def my_transpose(t, dim0, dim1) -> Tensor:
    """
    Returns t.transpose(dim0, dim1)

    Args:
        t: Tensor

    Returns: my_transpose(t, dim0, dim1)
    """
    return torch.ops.libtorch_agnostic.my_transpose.default(t, dim0, dim1)


def my_empty_like(t) -> Tensor:
    """
    Returns t.empty_like()

    Args:
        t: Tensor

    Returns: my_empty_like(t)
    """
    return torch.ops.libtorch_agnostic.my_empty_like.default(t)


def my_zero_(t) -> Tensor:
    """
    Returns t.zero_()

    Args:
        t: Tensor

    Returns: my_zero_(t)
    """
    return torch.ops.libtorch_agnostic.my_zero_.default(t)
