import math
from typing import Optional, Union

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs

from torch import Tensor
from torch._decomp import register_decomposition
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number,
    NumberType,
    TensorLike,
    TensorLikeType,
)
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper
from torch._refs import (
    _make_elementwise_binary_reference,
    _make_elementwise_unary_reference,
)


__all__ = [
    "bessel_j0",
    "bessel_j1",
    "digamma",
    "entr",
    "erf",
    "erfc",
    "erfcx",
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
    "log_ndtr",
    "logit",
    "log_softmax",
    "logsumexp",
    "multigammaln",
    "ndtr",
    "ndtri",
    "psi",
    "round",
    "sinc",
    "softmax",
    "spherical_bessel_j0",
    "xlog1py",
    "xlogy",
    "zeta",
]


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_bessel_j0,
)
def bessel_j0(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_j0(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_bessel_j1,
)
def bessel_j1(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_j1(a)


digamma = torch.digamma  # alias


@register_decomposition(torch.ops.aten.special_entr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def entr(a: TensorLikeType) -> TensorLikeType:
    return torch.where(
        torch.isnan(a),
        a,
        torch.where(a > 0, -a * torch.log(a), torch.where(a == 0, 0, -torch.inf)),
    )


erf = torch.erf  # alias


erfc = torch.erfc  # alias


@register_decomposition(torch.ops.aten.special_erfcx)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def erfcx(a: TensorLikeType) -> TensorLikeType:
    return prims.erfcx(a)


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
def i0e(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i0e(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=torch.ops.aten.special_i1
)
def i1(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i1(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=torch.ops.aten.special_i1e
)
def i1e(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i1e(a)


@register_decomposition(torch.ops.aten.special_log_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def log_ndtr(a: TensorLikeType) -> TensorLikeType:
    # Note: M_SQRT1_2 is the value of 1 / √2
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    t = a * M_SQRT1_2
    return torch.where(
        a < 1.0,
        torch.log(torch.special.erfcx(-t) / 2) - t * t,
        torch.log1p(-refs.erfc(t) / 2),
    )


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


@register_decomposition(torch.ops.aten.special_xlog1py)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlog1py(a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]):
    utils.check(
        isinstance(a, TensorLike) or isinstance(b, TensorLike),
        lambda: 'Expected either argument a or b to be a Tensor"',
    )

    # Operations like eq and log do not handle scalar values, so we convert them to scalar_tensors.
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = refs.scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = refs.scalar_tensor(a, dtype=b.dtype, device=b.device)

    # mypy: expected "Tensor"
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)
    rhs = torch.where(torch.eq(a, 0), 0, torch.mul(a, refs.log1p(b)))
    return torch.where(torch.isnan(b), float("nan"), rhs)


xlogy = torch.xlogy  # alias


multigammaln = torch.mvlgamma  # alias


@register_decomposition(torch.ops.aten.special_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtr(a: TensorLikeType) -> TensorLikeType:
    # Note: M_SQRT1_2 is the value of 1 / √2
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    a_sqrt_2 = a * M_SQRT1_2
    return (1 + torch.erf(a_sqrt_2)) * 0.5


@register_decomposition(torch.ops.aten.special_ndtri)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtri(a: TensorLikeType) -> TensorLikeType:
    return prims.ndtri(a)


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


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_spherical_bessel_j0,
)
def spherical_bessel_j0(a: TensorLikeType) -> TensorLikeType:
    return prims.spherical_bessel_j0(a)


zeta = _make_elementwise_binary_reference(
    prims.zeta,  # type: ignore[has-type]
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_zeta,
)
