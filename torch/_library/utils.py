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


def is_builtin(op: torch._ops.OpOverload) -> bool:
    assert isinstance(op, torch._ops.OpOverload)
    return op.namespace in {"aten", "prim", "prims"}


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


def mutates_and_returns_first_arg(op: torch._ops.OpOverload):
    """Check if an op is an inplace aten op, i.e. it mutates and returns the first arg.

    TODO: torchgen/model.py's FunctionSchema.parse is the source of truth for this,
    but not all PyTorch builds have torchgen (due to the yaml dependency being weird).
    Figure this out.

    Example: add_(Tensor(a!) x, Tensor y) -> Tensor(a)
    """
    if op.namespace != "aten":
        return False
    schema = op._schema
    if not len(schema.returns) == 1:
        return False
    if schema.returns[0].alias_info is None:
        return False
    alias_set = schema.returns[0].alias_info.after_set
    if len(alias_set) != 1:
        return False
    loc = next(iter(alias_set))
    if len(schema.arguments) < 1:
        return False
    first_arg = schema.arguments[0]
    if first_arg.alias_info is None:
        return False
    if not first_arg.alias_info.is_write:
        return False
    alias_set = first_arg.alias_info.after_set
    if len(alias_set) != 1:
        return False
    if loc != next(iter(alias_set)):
        return False
    for arg in schema.arguments[1:]:
        if arg.alias_info is not None:
            return False
    return True


def zip_schema(schema, args, kwargs):
    """zips schema.arguments and (args, kwargs) together.

    Assumes that (args, kwargs) were the inputs to some torch._ops.OpOverload:
    that is, kwargs must be keyword-only arguments and default values may be omitted.
    """
    assert len(schema.arguments) >= len(args) + len(kwargs)
    for i in range(len(schema.arguments)):
        info = schema.arguments[i]
        if info.kwarg_only:
            if info.name in kwargs:
                yield info, kwargs[info.name]
            continue
        if i >= len(args):
            # args that are equal to their default values are not populated
            # if they are followed by args that are equal to their defaults.
            # Skip these.
            continue
        yield info, args[i]
    return
