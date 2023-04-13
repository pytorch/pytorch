from typing import Callable, Dict, Sequence

import torch
import torch._prims_common as utils
from torch._decomp.utils import _add_op_to_registry
from torch._prims_common import Number, TensorLikeType, TensorOrNumberLikeType


aten = torch._ops.ops.aten
canonicalizer_registry: Dict["torch._ops.OpOverload", Callable] = {}

# A canonicalizer rewrites high level operations in a more "canonical" way. e.g.
#   torch.mul(x, 1) -> x.clone()
#   torch.pow(2, x) -> torch.exp2(x)
#
# Unlike a decomposition, canonicalizers are additionally required to:
#   1. use similar high-level operations, not many lower level operations
#   2. only call torch and aten functions, never prims
#   3. produce valid gradients when run before AOTAutograd


def register_canonicalizer(aten_ops):
    if not isinstance(aten_ops, Sequence):
        aten_ops = (aten_ops,)

    def inner(fn):
        global canonicalizer_registry

        for op in aten_ops:
            _add_op_to_registry(canonicalizer_registry, op, fn)

        return fn

    return inner


@register_canonicalizer(aten.pow)
def pow(
    a: TensorOrNumberLikeType,
    b: TensorOrNumberLikeType,
) -> TensorLikeType:
    if isinstance(b, Number):
        assert isinstance(a, TensorLikeType)
        if b == 1.0:
            return a.clone()
        elif b == 2.0:
            return a * a
        elif b == 0.5:
            return torch.sqrt(a)

    if isinstance(a, Number):
        assert isinstance(b, TensorLikeType)
        if a == 1.0:
            return torch.fill(b, True)
        if a == 2.0 and (
            utils.is_float_dtype(b.dtype) or utils.is_complex_dtype(b.dtype)
        ):
            return torch.exp2(b)

    return NotImplemented


@register_canonicalizer(aten.cat.default)
def cat(tensors, dim=0):
    if len(tensors) == 1:
        t = tensors[0]
        memory_format = utils.suggest_memory_format(t)
        return t.clone(memory_format=memory_format)
    return NotImplemented


@register_canonicalizer(aten.ceil.default)
def ceil(x):
    if utils.is_integer_dtype(x.dtype):
        return x.clone()
    return NotImplemented


@register_canonicalizer(aten.isnan)
def isnan(x):
    if utils.is_integer_dtype(x.dtype):
        return torch.full_like(x, False, dtype=torch.bool)
    return NotImplemented
