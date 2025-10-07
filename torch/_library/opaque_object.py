from typing import Any, NewType

import torch


OpaqueTypeStr = "__torch__.torch.classes.aten.OpaqueObject"

OpaqueType = NewType("OpaqueType", torch._C.ScriptObject)


def make_opaque(payload: Any = None) -> torch._C.ScriptObject:
    """
    Creates an opaque object which stores the given Python object.
    This opaque object can be passed to any custom operator as an argument.
    The Python object can then be accessed from the opaque object using the `get_payload()` API.
    The opaque object has `._type()`
    "__torch__.torch.classes.aten.OpaqueObject", which should be the type used
    when creating custom operator schemas.

    Args:
        payload (Any): The Python object to store in the opaque object. This can
        be empty, and can be set with `set_payload()` later.

    Returns:
        torch._C.ScriptObject: The opaque object that stores the given Python object.

    Example:

        >>> import random
        >>> import torch
        >>> from torch._library.opaque_object import (
        ...     make_opaque,
        ...     get_payload,
        ...     set_payload,
        ... )
        >>>
        >>> class RNGState:
        >>>     def __init__(self, seed):
        >>>         self.rng = random.Random(seed)
        >>>
        >>> rng = RNGState(0)
        >>> obj = make_opaque()
        >>> set_payload(obj, rng)
        >>>
        >>> assert get_payload(obj) == rng
        >>>
        >>> lib = torch.library.Library("mylib", "FRAGMENT")
        >>>
        >>> torch.library.define(
        >>>     "mylib::noisy_inject",
        >>>     "(Tensor x, __torch__.torch.classes.aten.OpaqueObject obj) -> Tensor",
        >>>     tags=torch.Tag.pt2_compliant_tag,
        >>>     lib=lib,
        >>> )
        >>>
        >>> @torch.library.impl(
        >>>     "mylib::noisy_inject", "CompositeExplicitAutograd", lib=lib
        >>> )
        >>> def noisy_inject(x: torch.Tensor, obj: torch._C.ScriptObject) -> torch.Tensor:
        >>>     rng_state = get_payload(obj)
        >>>     assert isinstance(rng_state, RNGState)
        >>>     out = x.clone()
        >>>     for i in range(out.numel()):
        >>>         out.view(-1)[i] += rng_state.rng.random()
        >>>     return out
        >>>
        >>> print(torch.ops.mylib.noisy_inject(torch.ones(3), obj))
    """
    return torch._C._make_opaque_object(payload)


def get_payload(opaque_object: torch._C.ScriptObject) -> Any:
    """
    Retrieves the Python object stored in the given opaque object.

    Args:
        torch._C.ScriptObject: The opaque object that stores the given Python object.

    Returns:
        payload (Any): The Python object stored in the opaque object. This can
        be set with `set_payload()`.
    """
    if not (
        isinstance(opaque_object, torch._C.ScriptObject)
        and opaque_object._type().qualified_name() == OpaqueTypeStr
    ):
        type_ = (
            opaque_object._type().qualified_name()
            if isinstance(opaque_object, torch._C.ScriptObject)
            else type(opaque_object)
        )
        raise ValueError(
            f"Tried to get the payload from a non-OpaqueObject of type `{type_}`"
        )
    return torch._C._get_opaque_object_payload(opaque_object)


def set_payload(opaque_object: torch._C.ScriptObject, payload: Any) -> None:
    """
    Sets the Python object stored in the given opaque object.

    Args:
        torch._C.ScriptObject: The opaque object that stores the given Python object.
        payload (Any): The Python object to store in the opaque object.
    """
    if not (
        isinstance(opaque_object, torch._C.ScriptObject)
        and opaque_object._type().qualified_name() == OpaqueTypeStr
    ):
        type_ = (
            opaque_object._type().qualified_name()
            if isinstance(opaque_object, torch._C.ScriptObject)
            else type(opaque_object)
        )
        raise ValueError(
            f"Tried to get the payload from a non-OpaqueObject of type `{type_}`"
        )
    torch._C._set_opaque_object_payload(opaque_object, payload)
