import weakref
from collections.abc import Callable
from typing import Any

from torch._dynamo.source import Source


PyCodegen = Any

# This file is to handle types that we don't want to support
# as explicit FX graph inputs. This uses a sidetable which
# we populate in bytecode and is loaded during graph execution

# We use a dynamo-generated index as a level of indirection
# this allows us to register objects externally in pre-graph bytecode that we want
# to pass to the graph, but not support their types as graph inputs.
# Pre-graph bytecode fills this table on every call; while compiling, it holds
# the trace-time objects of the trace being compiled (UserObjectRegistry.install).
index_to_external_object_weakref: list[weakref.ReferenceType[object]] = []

keep_alive: list[object] = []


def stash_graph_created_object(obj: object) -> object:
    keep_alive.append(obj)
    return obj


CURRENT_STREAM_INDEX = 0


class _UnusedCurrentStream:
    pass


unused_current_stream = _UnusedCurrentStream()


class UserObjectRegistry:
    """Objects one trace passes to its graph by index, owned by its OutputGraph.

    Each entry has a bytecode constructor, emitted in the pre-graph bytecode to
    fetch or build the object on every call, and a weakref to its trace-time
    value for lookups during compilation.
    """

    def __init__(self) -> None:
        self.bytecode_constructors: list[Callable[[PyCodegen], None]] = []
        self.weakrefs: list[weakref.ReferenceType[object]] = []
        # Graph-created trace-time objects; user objects are only weakly held.
        self.keep_alive: list[object] = []
        # True while the graph hasn't referenced the current stream registered
        # at CURRENT_STREAM_INDEX, so pre-graph bytecode can skip looking it up.
        self.current_stream_unused = False

    def needs_pre_graph_store(self) -> bool:
        n = len(self.bytecode_constructors)
        return n > 1 or (n == 1 and not self.current_stream_unused)

    def install(self) -> None:
        index_to_external_object_weakref[:] = self.weakrefs

    def register(
        self,
        ref: weakref.ReferenceType[object],
        construct: Callable[[PyCodegen], None],
    ) -> int:
        self.bytecode_constructors.append(construct)
        self.weakrefs.append(ref)
        self.install()
        return len(self.weakrefs) - 1


def _active_registry() -> UserObjectRegistry | None:
    from .symbolic_convert import tls

    tx = getattr(tls, "current_tx", None)
    return None if tx is None else tx.output.user_objects


def _require_active_registry() -> UserObjectRegistry:
    registry = _active_registry()
    if registry is None:
        raise AssertionError("User objects can only be registered while tracing")
    return registry


def install_active_registry() -> None:
    """Make the runtime table hold the objects of the trace being compiled.

    Compiled functions run during the compile (e.g. by the backend) replace it
    with their own objects.
    """
    if (registry := _active_registry()) is not None:
        registry.install()


def register_current_stream(stream: object, source: Source) -> None:
    registry = _require_active_registry()
    if registry.bytecode_constructors:
        raise AssertionError(
            f"Current stream must be registered at index {CURRENT_STREAM_INDEX}"
        )

    def construct(cg: PyCodegen) -> None:
        if registry.current_stream_unused:
            cg.load_import_from(__name__, "unused_current_stream")
        else:
            cg(source)

    registry.register(weakref.ref(stream), construct)
    registry.current_stream_unused = True


def mark_current_stream_used() -> None:
    _require_active_registry().current_stream_unused = False


def set_external_object_by_index(index: int, value: object) -> None:
    """Add or update an entry in the external object registry at runtime."""
    keep_alive.append(value)
    if index == len(index_to_external_object_weakref):
        index_to_external_object_weakref.append(weakref.ref(value))
    elif index < len(index_to_external_object_weakref):
        index_to_external_object_weakref[index] = weakref.ref(value)
    else:
        raise AssertionError("Index past the end of index_to_user_object_weakref")


def get_external_object_by_index(index: int) -> object:
    if index >= len(index_to_external_object_weakref):
        raise AssertionError("Index not registered in index_to_user_object_weakref")
    obj = index_to_external_object_weakref[index]()
    if obj is None:
        raise AssertionError("User object is no longer alive")
    return obj


def store_user_object_weakrefs(*args: object) -> None:
    index_to_external_object_weakref[:] = map(weakref.ref, args)


def reset_user_object_tracking() -> None:
    index_to_external_object_weakref.clear()
    keep_alive.clear()


def register_graph_created_object(
    example_value: object, construct_fn: Callable[[int, PyCodegen], None]
) -> int:
    try:
        ref = weakref.ref(example_value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to graph-created external object",
            context=f"user_object: {example_value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    registry = _require_active_registry()
    registry.keep_alive.append(example_value)
    index = registry.register(ref, lambda cg: construct_fn(index, cg))
    return index


# Register a user object to be used in the graph
def register_user_object(value: object, source: Source) -> int:
    try:
        ref = weakref.ref(value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to User Object",
            context=f"user_object: {value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return _require_active_registry().register(ref, lambda cg: cg(source))


# Register a callback so invoke_leaf_function can retrieve nn.Module instances at runtime.
# We use a callback pattern instead of having invoke_leaf_function import get_external_object_by_index
# directly, because higher-order ops should not depend on dynamo (dynamo depends on them, not vice versa).
from torch._higher_order_ops.invoke_leaf_function import (
    set_leaf_function_module_retriever,
)


set_leaf_function_module_retriever(get_external_object_by_index)
