import torch
from typing import Dict, Optional, TypeVar

from .annotations import ann_to_type

from torch._C import ScriptDict  # type: ignore

K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')


def dict(data: Dict[K, V], type_hint: Optional[T] = None) -> ScriptDict:
    """
    Create a torch._C.ScriptDict that has reference semantics across the
    Python/TorchScript boundary and can be used as if it were a dictionary
    in Python.

    Arguments:
        data: A dictionary whose data should be used to initialize the ScriptDict.
        type_hint: The type of the ScriptDict. If not provided, it is inferred
                    based on `data`.
    """
    # TODO: Does it make sense to pass in loc=None here?
    if type_hint:
        ty = ann_to_type(type_hint, None)
    else:
        inferred_ty = torch._C._jit_try_infer_type(data)
        ty = inferred_ty.type()

    return ScriptDict(data, ty)


def empty_dict(type_hint: Optional[T]) -> ScriptDict:
    """
    Create an empty torch._C.ScriptDict that has reference semantics across the
    Python/TorchScript boundary and can be used as if it were a dictionary
    in Python.

    Arguments:
        type_hint: The type of the ScriptDict. If not provided, it is inferred
                    based on `data`.
    """
    ty = ann_to_type(type_hint, None)

    return ScriptDict(ty)
