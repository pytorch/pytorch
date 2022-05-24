import torch

import torch._prims.utils as utils
from torch._prims.utils import (
    TensorLikeType,
    NumberType,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
)
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims.wrappers import (
    elementwise_type_promotion_wrapper,
    out_wrapper,
)

from typing import Optional

__all__ = [
    "celu",
    "dropout",
    "elu",
    "mish",
    "selu",
    "softplus",
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
