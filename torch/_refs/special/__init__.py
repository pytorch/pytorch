import torch

from torch import Tensor
from typing import Optional
import torch._prims as prims
import torch._prims.utils as utils
from torch._prims.utils import TensorLikeType
from torch._prims.wrappers import out_wrapper, elementwise_type_promotion_wrapper
from torch._refs import _make_elementwise_unary_reference
from torch._decomp import register_decomposition

__all__ = [
    "i0e",
    "i1e",
    "logit",
]

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

@register_decomposition(torch.ops.aten.logit)
@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logit(self: Tensor, eps: Optional[float] = None) -> Tensor:
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps
    self = torch.clamp(self, lo, hi)
    return (self / (1 - self)).log()
