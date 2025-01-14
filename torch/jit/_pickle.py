# These functions are referenced from the pickle archives produced by
# ScriptModule.save()


# These (`build_*`) functions used to be used by `pickler.cpp` to specify
# the type of the list for certain special types, but now all lists get
# a type attached and restored via `restore_type_tag` below. The legacy
# functions should stick around for backwards-compatibility.

from typing import List, Union


def build_intlist(data: List[int]) -> List[int]:
    return data


def build_tensorlist(data: List[object]) -> List[object]:
    return data


def build_doublelist(data: List[float]) -> List[float]:
    return data


def build_boollist(data: List[bool]) -> List[bool]:
    return data


def build_tensor_from_id(data: Union[int, object]) -> Union[int, None]:
    if isinstance(data, int):
        # just the id, can't really do anything
        return data
    return None


def restore_type_tag(value: object, type_str: str) -> object:
    # The type_ptr is used by the jit unpickler to restore the full static type
    # to container types like list when they are re-loaded, but this doesn't
    # matter for Python, so just return the plain value
    return value
