import torch

from torch import Tensor
from typing import Optional
import torch._prims as prims
import torch._prims.utils as utils
import torch._refs as refs
from torch._prims.utils import TensorLikeType
from torch._prims.wrappers import out_wrapper, elementwise_type_promotion_wrapper
from torch._refs import (
    _make_elementwise_unary_reference,
    _make_elementwise_binary_reference,
)
from torch._decomp import register_decomposition


__all__ = [
    "i0e",
    "i1",
    "i1e",
    "logit",
    "zeta",
]

i0e = _make_elementwise_unary_reference(
    prims.bessel_i0e,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_i0e,
)
i1 = _make_elementwise_unary_reference(
    prims.bessel_i1,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_i1,
)
i1e = _make_elementwise_unary_reference(
    prims.bessel_i1e,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_i1e,
)


@register_decomposition(torch.ops.aten.logit)
@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logit(self: TensorLikeType, eps: Optional[float] = None) -> TensorLikeType:
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps
    self = refs.clamp(self, lo, hi)
    return refs.log(refs.true_divide(self, refs.sub(1, self)))


zeta = _make_elementwise_binary_reference(
    prims.zeta,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_zeta,
)
