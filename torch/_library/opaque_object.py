from typing import Any

import torch


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

        >>> import torch
        >>> from torch._library.opaque_object import (
        ...     make_opaque,
        ...     get_payload,
        ...     set_payload,
        ... )
        >>>
        >>> class OpaqueQueue:
        >>>     def __init__(self, queue: list[torch.Tensor], init_tensor_: torch.Tensor) -> None:
        >>>         super().__init__()
        >>>         self.queue = queue
        >>>         self.init_tensor_ = init_tensor_
        >>>
        >>>     def push(self, tensor: torch.Tensor) -> None:
        >>>         self.queue.append(tensor)
        >>>
        >>>     def pop(self) -> torch.Tensor:
        >>>         if len(self.queue) > 0:
        >>>             return self.queue.pop(0)
        >>>         return self.init_tensor_
        >>>
        >>>     def size(self) -> int:
        >>>         return len(self.queue)
        >>>
        >>> queue = OpaqueQueue([], torch.zeros(2, 2))
        >>> obj = make_opaque()
        >>> set_payload(obj, queue)
        >>>
        >>> assert get_payload(obj) == queue
        >>>
        >>> lib = torch.library.Library("mylib", "FRAGMENT")
        >>>
        >>> torch.library.define(
        >>>     "mylib::queue_push",
        >>>     "(__torch__.torch.classes.aten.OpaqueObject a, Tensor b) -> ()",
        >>>     tags=torch.Tag.pt2_compliant_tag,
        >>>     lib=lib,
        >>> )
        >>>
        >>> @torch.library.impl(
        >>>     "mylib::queue_push", "CompositeExplicitAutograd", lib=lib
        >>> )
        >>> def push_impl(q: torch._C.ScriptObject, b: torch.Tensor) -> None:
        >>>     queue = get_payload(q)
        >>>     assert isinstance(queue, OpaqueQueue)
        >>>     queue.push(b)
        >>>
        >>> torch.ops.mylib.queue_push(obj, torch.ones(3) + 1)
        >>> assert get_payload(obj).size() == 1
    """
    return torch._C.make_opaque_object(payload)


def get_payload(opaque_object: torch._C.ScriptObject) -> Any:
    """
    Retrieves the Python object stored in the given opaque object.

    Args:
        torch._C.ScriptObject: The opaque object that stores the given Python object.

    Returns:
        payload (Any): The Python object stored in the opaque object. This can
        be set with `set_payload()`.
    """
    return torch._C.get_opaque_object_payload(opaque_object)


def set_payload(opaque_object: torch._C.ScriptObject, payload: Any) -> None:
    """
    Sets the Python object stored in the given opaque object.

    Args:
        torch._C.ScriptObject: The opaque object that stores the given Python object.
        payload (Any): The Python object to store in the opaque object.
    """
    torch._C.set_opaque_object_payload(opaque_object, payload)
