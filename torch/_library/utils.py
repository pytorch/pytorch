import dataclasses
import inspect
import sys
from typing import Any, Callable, Tuple

import torch


@dataclasses.dataclass
class Kernel:
    """Models a (function, source location)"""

    func: Callable
    source: str

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class RegistrationHandle:
    """Does something when someone calls .destroy() on it"""

    def __init__(self, on_destroy: Callable):
        self._on_destroy = on_destroy

    def destroy(self) -> None:
        self._on_destroy()


def get_source(stacklevel: int) -> str:
    """Get a string that represents the caller.

    Example: "/path/to/foo.py:42"

    Use stacklevel=1 to get the caller's source
    Use stacklevel=2 to get the caller's caller's source
    etc.
    """
    frame = inspect.getframeinfo(sys._getframe(stacklevel))
    source = f"{frame.filename}:{frame.lineno}"
    return source


def parse_namespace(qualname: str) -> Tuple[str, str]:
    splits = qualname.split("::")
    if len(splits) != 2:
        raise ValueError(
            f"Expected `qualname` to be of the form "
            f'"namespace::name", but got {qualname}. '
            f"The qualname passed to the torch.library APIs must consist "
            f"of a namespace and a name, e.g. aten::sin"
        )
    return splits[0], splits[1]


def lookup_op(qualname: str) -> torch._ops.OpOverloadPacket:
    namespace, name = parse_namespace(qualname)
    if "." in name:
        name, overload = name.split(".")
    else:
        overload = "default"
    ns = getattr(torch.ops, namespace)
    packet = getattr(ns, name)
    return getattr(packet, overload)


def is_functional_schema(schema: Any) -> bool:
    """Check if the schema is functional.

    An operator is functional if:
    - it does not mutate any of its inputs
    - it does not return a view on any of its inputs
    - it has at least one return
    """

    # Lazy import because not all PyTorch builds have torchgen
    from torchgen.model import FunctionSchema, SchemaKind

    assert isinstance(schema, (str, FunctionSchema))
    if isinstance(schema, str):
        schema = FunctionSchema.parse(schema)

    if schema.kind() != SchemaKind.functional:
        return False
    rets = schema.returns
    is_non_mutating_view = len(rets) > 0 and any(
        r.annotation is not None and not r.annotation.is_write for r in rets
    )
    if is_non_mutating_view:
        return False
    if not schema.returns:
        return False
    return True
