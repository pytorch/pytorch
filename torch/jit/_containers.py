from typing import List, Optional, TypeVar

import torch
from torch._C import ScriptList  # type: ignore

from .annotations import ann_to_type

U = TypeVar("U")
T = TypeVar("T")


def list(data: List[U], type_hint: Optional[T] = None) -> ScriptList:
    """
    Create a torch._C.ScriptList that has reference semantics across the
    Python/TorchScript boundary and can be used as if it were a list
    in Python.

    Arguments:
        data: A list whose data should be used to initialize the ScriptList.
        type_hint: The type of the ScriptList. If not provided, it is inferred
                    based on `data`.
    """
    # TODO: Does it make sense to pass in loc=None here?
    if type_hint:
        ty = ann_to_type(type_hint, None)
    else:
        inferred_ty = torch._C._jit_try_infer_type(data)
        ty = inferred_ty.type()

    return ScriptList(data, ty)


def empty_list(type_hint: Optional[T]) -> ScriptList:
    """
    Create an empty torch._C.ScriptList that has reference semantics across the
    Python/TorchScript boundary and can be used as if it were a list
    in Python.

    Arguments:
        type_hint: The type of the ScriptList. If not provided, it is inferred
                    based on `data`.
    """
    ty = ann_to_type(type_hint, None)

    return ScriptList(ty)
