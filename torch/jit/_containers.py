import torch
from typing import Dict, Optional, TypeVar

from .annotations import ann_to_type

from torch._C import ScriptDict, Type

K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')


def dict(data: Dict[K, V], type_hint: Optional[T] = None) -> ScriptDict:
    # TODO: Does it make sense to pass in loc=None here?
    if type_hint:
        ty = ann_to_type(type_hint, None)
    else:
        inferred_ty = torch._C._jit_try_infer_type(data)
        ty = inferred_ty.type()

    return ScriptDict(data, ty)


def empty_dict(type_hint: Optional[T]) -> ScriptDict:
    ty = ann_to_type(type_hint, None)

    return ScriptDict(ty)
