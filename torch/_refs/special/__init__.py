import torch

import torch._prims as prims
import torch._prims.utils as utils
from torch._prims.utils import TensorLikeType
from torch._prims.wrappers import out_wrapper, elementwise_type_promotion_wrapper
from torch._refs import _make_elementwise_unary_reference

__all__ = [
    "i0e",
    "i1e",
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
