import torch

import torch._prims.utils as utils
from torch._prims.utils import (
    TensorLike,
    TensorLikeType,
    NumberType,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
)
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims.wrappers import (
    elementwise_type_promotion_wrapper,
    elementwise_unary_scalar_wrapper,
    out_wrapper,
)
from torch._refs import (
    _make_elementwise_unary_reference,
    _make_elementwise_binary_reference,
)

from typing import Optional

__all__ = [
    "celu",
    "dropout",
    "elu",
    "relu",
    "hinge_embedding_loss",
    "margin_ranking_loss",
    "mish",
    "selu",
    "softplus",
    "tanhshrink",
]

# celu is implemented specially because it has an alpha argument
# celu is very similar to elu
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def celu(
    a: TensorLikeType, alpha: Optional[NumberType] = None, inplace: bool = False
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.celu
    """

    if inplace:
        raise NotImplementedError

    rhs: TensorLikeType
    if alpha is not None:
        python_type = utils.dtype_to_type(a.dtype)
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = (
                "alpha argument of type {0} cannot be safely cast to type {1}!".format(
                    type(alpha), python_type
                )
            )
            raise ValueError(msg)
        rhs = refs.mul(alpha, refs.expm1(refs.true_divide(a, alpha)))
    else:
        rhs = refs.expm1(a)

    return refs.where(refs.gt(a, 0), a, rhs)


# TODO: should we allow the user to set a different dtype for the mask generation?
def dropout(
    a: TensorLikeType, p: float = 0.5, training: bool = True, inplace: bool = False
) -> TensorLikeType:

    if inplace:
        raise NotImplementedError

    if not training:
        return a

    assert p <= 1
    assert p >= 0

    if p == 1:
        return refs.zeros_like(a)

    if p == 0:
        return a

    p1m = 1 - p
    scale = 1 / p1m
    mask = refs.lt(
        refs.uniform(a.shape, low=0.0, high=1.0, dtype=torch.float32, device=a.device),
        p1m,
    )
    return refs.mul(refs.mul(a, mask), scale)


# elu is implemented specially because it has an alpha argument
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def elu(
    a: TensorLikeType, alpha: Optional[NumberType] = None, inplace: bool = False
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.elu
    """

    if inplace:
        raise NotImplementedError

    rhs: TensorLikeType
    if alpha is not None:
        python_type = utils.dtype_to_type(a.dtype)
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = (
                "alpha argument of type {0} cannot be safely cast to type {1}!".format(
                    type(alpha), python_type
                )
            )
            raise ValueError(msg)
        rhs = refs.mul(alpha, refs.expm1(a))
    else:
        rhs = refs.expm1(a)

    return refs.where(refs.gt(a, 0), a, rhs)


@register_decomposition(torch.ops.aten.relu)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def relu(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.relu
    """

    if inplace:
        raise NotImplementedError

    return torch.where(torch.gt(a, 0), a, 0)


@register_decomposition(torch.ops.aten.leaky_relu)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def leaky_relu(
    a: TensorLikeType, negative_slope: float = 0.01, inplace: bool = False
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.leaky_relu
    """

    if inplace:
        raise NotImplementedError

    python_type = utils.dtype_to_type(a.dtype)
    if not utils.is_weakly_lesser_type(type(negative_slope), python_type):
        msg = f"negative_slope argument of type {type(negative_slope)} cannot be safely cast to type {python_type}!"
        raise ValueError(msg)
    return torch.where(torch.gt(a, 0), a, torch.mul(a, negative_slope))


@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def mish(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.mish
    """

    if inplace:
        raise NotImplementedError

    return refs.mul(a, refs.tanh(refs.nn.functional.softplus(a)))


@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def selu(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.selu
    """

    if inplace:
        raise NotImplementedError

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    rhs = refs.mul(alpha, refs.expm1(a))

    return refs.mul(scale, refs.where(refs.gt(a, 0), a, rhs))


# softplus is implemented specially because it has beta and threshold arguments
@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def softplus(
    a: TensorLikeType,
    beta: Optional[NumberType] = None,
    threshold: NumberType = 20,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.softplus
    """

    if inplace:
        raise NotImplementedError

    rhs: TensorLikeType
    if beta is not None:
        python_type = utils.dtype_to_type(a.dtype)
        if not utils.is_weakly_lesser_type(type(beta), python_type):
            msg = "beta argument of type {0} cannot be safely cast to type {1}!".format(
                type(beta), python_type
            )
            raise ValueError(msg)
        scaled_input = refs.mul(a, beta)
        rhs = refs.true_divide(refs.log1p(refs.exp(scaled_input)), beta)
    else:
        scaled_input = a
        rhs = refs.log1p(refs.exp(scaled_input))

    return refs.where(refs.gt(scaled_input, threshold), a, rhs)


# Losses
def _apply_loss_reduction(loss: TensorLikeType, reduction: str) -> TensorLikeType:
    if reduction == "sum":
        return refs.sum(loss)
    elif reduction == "mean":
        return refs.mean(loss)
    else:  # reduction == "none"
        return loss


def _check_reduction_value(reduction: str):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("{} is not a valid value for reduction".format(reduction))


def margin_ranking_loss(
    input1: TensorLikeType,
    input2: TensorLikeType,
    target: TensorLikeType,
    margin: float = 0.0,
    reduction: str = "mean",
) -> TensorLikeType:
    # Formula of loss (implementation gets confusing with all the refs.foo)
    # loss_without_reduction = max(0, −target * (input1 − input2) + margin)
    if input1.ndim != input2.ndim or input1.ndim != target.ndim:
        raise RuntimeError(
            (
                "margin_ranking_loss : All input tensors should have same dimension but got sizes: "
                "input1: {}, input2: {}, target: {} ".format(
                    input1.shape, input2.shape, target.shape
                )
            )
        )
    _check_reduction_value(reduction)
    neg_target = refs.neg(target)
    input_diff = refs.sub(input1, input2)
    mul_target_input = refs.mul(neg_target, input_diff)
    add_margin = refs.add(mul_target_input, margin)
    loss = refs.maximum(add_margin, 0)
    return _apply_loss_reduction(loss, reduction)


def hinge_embedding_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    margin: float = 1.0,
    reduction: str = "mean",
) -> TensorLikeType:
    # Formula of loss (implementation gets confusing with all the refs.foo)
    # loss_without_reduction = input if y == 1
    #                        = max(0, margin - input) if y == -1
    _check_reduction_value(reduction)
    margin_clamp = refs.maximum(refs.sub(margin, input), 0)
    output_margin = refs.where(refs.ne(target, 1), margin_clamp, 0)
    output_self = refs.where(refs.ne(target, -1), input, 0)
    loss = refs.add(output_margin, output_self)
    return _apply_loss_reduction(loss, reduction)


# tanhshrink does not use _make_elementwise_unary_reference because it does not support out
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def tanhshrink(a: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.tanhshrink
    """
    if not isinstance(a, TensorLike):
        raise RuntimeError(
            "Expected a tensor input for an elementwise unary operation!"
        )
    return refs.sub(a, refs.tanh(a))
