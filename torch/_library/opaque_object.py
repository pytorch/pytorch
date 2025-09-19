from typing import Any

import torch


def make_opaque(payload: Any) -> torch._C.ScriptObject:
    """
    Creates an opaque object which stores the given Python object.
    This opaque object can be passed to any custom operator as an argument.
    The Python object can then be accessed from the opaque object using the `get_payload()` API.
    """
    return torch._C._make_opaque_object(payload)


def get_payload(opaque_object: torch._C.ScriptObject) -> Any:
    return torch._C._get_opaque_object_payload(opaque_object)
