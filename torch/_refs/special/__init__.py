import torch

import torch._prims as prims
import torch._prims.utils as utils
from typing import Sequence, Optional, Union, Callable, List, Tuple
from torch._prims.utils import TensorLikeType, NumberType, Number
from torch._prims.wrappers import out_wrapper, elementwise_type_promotion_wrapper
from torch._refs import (
    _make_elementwise_unary_reference,
    _make_elementwise_binary_reference,
)

import torch._refs as refs

Tensor = torch.Tensor

__all__ = [
    "i1",
    "i0e",
    "i1e",
    "zeta",
    "xlog1py",
]

i1 = _make_elementwise_unary_reference(
    prims.bessel_i1,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_i1,
)

i0e = _make_elementwise_unary_reference(
    prims.bessel_i0e,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_i0e,
)

i1e = _make_elementwise_unary_reference(
    prims.bessel_i1e,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_i1e,
)

def _xlog1py(a: Union[Tensor, NumberType], b: Union[Tensor, NumberType]):
    if isinstance(a, Tensor) and isinstance(b, Number):
        b = prims._wrap_scalar(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, Tensor) and isinstance(a, Number):
        a = prims._wrap_scalar(a, dtype=b.dtype, device=b.device)

    cond = refs.bitwise_and(refs.eq(a, 0), refs.bitwise_not(refs.isnan(b)))
    rhs = refs.where(cond, a, refs.mul(a, refs.log1p(b)))
    return refs.where(refs.isnan(b), b, rhs)

# TODO add docstring
xlog1py = _make_elementwise_binary_reference(
    _xlog1py,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_xlog1py,
)

zeta = _make_elementwise_binary_reference(
    prims.zeta,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_zeta,
)
