import torch

import torch._prims as prims
import torch._prims.utils as utils
from typing import Sequence, Optional, Union, Callable, List, Tuple
from torch._prims.utils import (
    TensorLike,
    TensorLikeType,
    Number,
    NumberType,
)
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
    "xlog1py",
    "zeta",
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


def _xlog1py(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    assert isinstance(a, TensorLike) or isinstance(b, TensorLike)

    # torch.xlog1py supports scalar inputs but torch.log does not.
    # TODO Add support for scalar inputs to refs.log (and other elementwise unary ops)
    if isinstance(a, TensorLike):
        if isinstance(b, Number):
            b = prims._wrap_scalar(b, dtype=a.dtype, device=a.device)
        elif utils.is_cpu_scalar_tensor(b):
            b = prims.device_put(b, device=a.device)
    elif isinstance(b, TensorLike):
        if isinstance(a, Number):
            a = prims._wrap_scalar(a, dtype=b.dtype, device=b.device)
        elif utils.is_cpu_scalar_tensor(a):
            a = prims.device_put(a, device=b.device)

    rhs = refs.where(refs.eq(a, 0), 0, refs.mul(a, refs.log1p(b)))
    return refs.where(refs.isnan(b), float("nan"), rhs)


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
