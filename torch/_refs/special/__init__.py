import math
from typing import Optional

import torch
import torch._prims as prims
import torch._prims_common as utils

from torch import Tensor
from torch._decomp import register_decomposition
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, TensorLikeType
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper
from torch._refs import (
    _make_elementwise_binary_reference,
    _make_elementwise_unary_reference,
)


__all__ = [
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "exp2",
    "expit",
    "expm1",
    "gammaln",
    "gammainc",
    "gammaincc",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "log1p",
    "logit",
    "log_softmax",
    "logsumexp",
    "multigammaln",
    "psi",
    "round",
    "sinc",
    "softmax",
    "zeta",
]


digamma = torch.digamma  # alias


erf = torch.erf  # alias


erfc = torch.erfc  # alias


erfinv = torch.erfinv  # alias


exp2 = torch.exp2  # alias


expit = torch.sigmoid  # alias


expm1 = torch.expm1  # alias


gammaln = torch.lgamma  # alias


gammainc = torch.igamma  # alias


gammaincc = torch.igammac  # alias


i0 = torch.i0  # alias


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=torch.ops.aten.special_i0e
)
def i0e(a):
    return prims.bessel_i0e(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=torch.ops.aten.special_i1
)
def i1(a):
    return prims.bessel_i1(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=torch.ops.aten.special_i1e
)
def i1e(a):
    return prims.bessel_i1e(a)


@register_decomposition(torch.ops.aten.logit)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logit(input: TensorLikeType, eps: Optional[float] = None) -> TensorLikeType:
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps
    input = torch.clamp(input, lo, hi)
    return torch.log(torch.true_divide(input, torch.sub(1, input)))


log1p = torch.log1p  # alias


multigammaln = torch.mvlgamma  # alias


# Forwarding alias: the special variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def log_softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    return torch.log_softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


logsumexp = torch.logsumexp  # alias


# Autograd note: This will give the right first derivative at zero (by chance),
# but not the right second derivative
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sinc(a):
    a = math.pi * a
    return torch.where(a == 0, 1, torch.sin(a) / a)


# Forwarding alias: the special variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    return torch.softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


psi = torch.digamma  # alias


round = torch.round  # alias


zeta = _make_elementwise_binary_reference(
    prims.zeta,  # type: ignore[has-type]
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_zeta,
)
