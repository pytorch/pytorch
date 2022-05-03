from torch._prims.utils import TensorLikeType, NumberType
import torch._refs as refs

from numbers import Number


all = [
    "elu",
]


def elu(
    a: TensorLikeType, alpha: NumberType = 1.0, inplace: bool = False
) -> TensorLikeType:
    if inplace:
        raise NotImplementedError

    lhs = refs.maximum(0, a)
    rhs = refs.minimum(0, refs.mul(alpha, refs.sub(refs.exp(a), 1)))
    return refs.add(lhs, rhs)
