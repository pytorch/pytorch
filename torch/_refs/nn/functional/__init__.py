import torch

import torch._prims.utils as utils
from torch._prims.utils import (
    TensorLikeType,
    NumberType,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
)
import torch._refs as refs
from torch._prims.wrappers import elementwise_type_promotion_wrapper

from typing import Optional

__all__ = [
    "elu",
]

# elu is implemented specially because it has an alpha argument
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.OP_MATH,
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
