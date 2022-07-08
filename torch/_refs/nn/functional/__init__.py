import torch

import torch._prims as prims
import torch._prims.utils as utils
from torch._prims.utils import (
    check,
    ShapeType,
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

from typing import Optional, Union

__all__ = [
    "celu",
    "dropout",
    "elu",
    "hardshrink",
    "hardtanh",
    "hinge_embedding_loss",
    "margin_ranking_loss",
    "mish",
    "relu",
    "selu",
    "softplus",
    "softshrink",
    "tanhshrink",
    "threshold",
]

Tensor = torch.Tensor

# celu is implemented specially because it has an alpha argument
# celu is very similar to elu
@register_decomposition(torch.ops.aten.celu)
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
        rhs = alpha * torch.expm1(torch.true_divide(a, alpha))  # type: ignore[arg-type]
    else:
        rhs = torch.expm1(a)

    return torch.where(a > 0, a, rhs)


# TODO: should we allow the user to set a different dtype for the mask generation?
@register_decomposition(torch.ops.aten.dropout)
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
# This cannot be used as a decomposition because the aten op takes in 2 extra kwargs
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
        rhs = alpha * torch.expm1(a)
    else:
        rhs = torch.expm1(a)

    return torch.where(a > 0, a, rhs)


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

    return torch.where(torch.le(a, 0), 0, a)


def layer_norm(
    input: Tensor,
    normalized_shape: ShapeType,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Reference implementation of :func:`torch.nn.functional.layer_norm`.
    """
    return torch.native_layer_norm(input, normalized_shape, weight, bias, eps)[0]


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


@register_decomposition(torch.ops.aten.mish)
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
    return a * torch.tanh(torch.nn.functional.softplus(a))


@register_decomposition(torch.ops.aten.selu)
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

    rhs = alpha * torch.expm1(a)

    return scale * torch.where(a > 0, a, rhs)


# softplus is implemented specially because it has beta and threshold arguments
@register_decomposition(torch.ops.aten.softplus)
@out_wrapper()
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
        scaled_input = a * beta
        rhs = torch.true_divide(torch.log1p(torch.exp(scaled_input)), beta)  # type: ignore[arg-type]

    else:
        scaled_input = a
        rhs = torch.log1p(torch.exp(scaled_input))

    return torch.where(scaled_input > threshold, a, rhs)


@register_decomposition(torch.ops.aten.hardshrink)
@out_wrapper()
def hardshrink(a: TensorLikeType, lambd: float = 0.5):
    # Formula for reference,
    # hardshrink(x) = x if x > lambd
    #               = x if x < -lambd
    #               = 0 otherwise
    mask = refs.logical_or(refs.logical_or(a < -lambd, a > lambd), refs.isnan(a))
    return refs.where(mask, a, 0)


@register_decomposition(torch.ops.aten.softshrink)
@out_wrapper()
def softshrink(a: TensorLikeType, lambd: float = 0.5):
    # Formula for reference,
    # softshrink(x) = x - lambd if x > lambd
    #               = x + lambd if x < -lambd
    #               = 0 otherwise
    check(
        lambd >= 0,
        lambda: f"lambda must be greater or equal to 0, but found to be {lambd}",
    )
    ge_mask = a > lambd
    le_mask = a < -lambd
    zero_mask = torch.logical_not(refs.logical_or(ge_mask, le_mask))
    result = refs.where(ge_mask, a - lambd, a)
    result = refs.where(le_mask, a + lambd, result)
    return refs.where(zero_mask, 0, result)


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


@register_decomposition(torch.ops.aten.margin_ranking_loss)
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


@register_decomposition(torch.ops.aten.hinge_embedding_loss)
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


@register_decomposition(torch.ops.aten.threshold)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def threshold(
    a: TensorLikeType,
    threshold: NumberType,
    value: Union[bool, int, float],
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.threshold
    """

    if inplace:
        raise NotImplementedError

    return torch.where(a <= threshold, value, a)


@register_decomposition(torch.ops.aten.hardtanh)
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def hardtanh(
    a: TensorLikeType,
    min_val: NumberType = -1,
    max_val: NumberType = 1,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.hardtanh
    """
    if inplace:
        raise NotImplementedError
    if utils.is_boolean_dtype(a.dtype):
        raise RuntimeError("Bool inputs not supported for hardtanh")

    # preserve legacy behavior of boundaries not causing type promotion
    if utils.is_integer_dtype(a.dtype):
        min_val = int(min_val)  # type: ignore[arg-type]
        max_val = int(max_val)  # type: ignore[arg-type]
        if not (a.dtype != torch.uint8 or (min_val >= 0 and max_val >= 0)):
            raise RuntimeError(
                "Cannot do hardtanh on an unsigned type with negative limits"
            )
    return torch.clamp(a, min_val, max_val)  # type: ignore[arg-type]


@register_decomposition(torch.ops.aten.gelu)
@out_wrapper()
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def gelu(a: TensorLikeType, approximate: str = "none") -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.gelu
    """
    if not isinstance(a, TensorLike):
        raise RuntimeError(
            "Expected a tensor input for an elementwise unary operation!"
        )
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    if approximate == "tanh":
        kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
        kKappa = 0.044715
        a_cube = a * a * a
        inner = kBeta * (a + kKappa * a_cube)
        return 0.5 * a * (1 + torch.tanh(inner))
    elif approximate == "none":
        kAlpha = M_SQRT1_2
        return a * 0.5 * (1 + torch.erf(a * kAlpha))
    else:
        raise RuntimeError("approximate argument must be either none or tanh.")


@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "weight"),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def prelu(a: TensorLikeType, weight: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.prelu
    """
    check(
        isinstance(a, TensorLike),
        lambda: f"prelu: Expected `a` to be tensor, but got: {type(a)}",
    )
    check(
        isinstance(weight, TensorLike),
        lambda: f"prelu: Expected `weight` to be tensor, but got: {type(weight)}",
    )

    if weight.numel() != 1:
        check(a.ndim > 0, lambda: "Not allow zero-dim input tensor.")
        channel_size = a.shape[1] if a.ndim >= 2 else 1
        check(
            weight.numel() == channel_size,
            lambda: f"Mismatch of parameter numbers and input channel size. Found parameter numbers ="
            f" {weight.numel()} and channel size = {channel_size}.",
        )

    check(
        weight.ndim == 0 or weight.ndim == 1,
        lambda: f"prelu: Expected `weight` to be a scalar or 1D tensor, but got: "
        f"ndim = {weight.ndim}",
    )
    weight = prims.broadcast_in_dim(
        weight, a.shape, tuple() if weight.ndim == 0 else (1,)
    )

    return refs.where(a > 0, a, a * weight)

def relu6(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.prelu
    """
    if inplace:
        raise NotImplementedError

    return refs.nn.functional.hardtanh(a, 0, 6)
