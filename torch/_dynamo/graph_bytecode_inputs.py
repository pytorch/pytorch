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
# to pass to the graph, but not support their types as graph inputs
index_to_bytecode_constructor: dict[int, Callable[[PyCodegen], None]] = {}

index_to_external_object_weakref: dict[int, weakref.ReferenceType[Any]] = {}

keep_alive: list[Any] = []

# Keep index 0 available for the ambient current stream so cudagraph wrappers
# can replace it with the capture stream at runtime.
CURRENT_STREAM_INDEX = 0
FIRST_USER_OBJECT_INDEX = CURRENT_STREAM_INDEX + 1
next_user_object_index = FIRST_USER_OBJECT_INDEX


def has_user_objects() -> bool:
    return bool(index_to_bytecode_constructor)


def stash_graph_created_object(obj: Any) -> Any:
    keep_alive.append(obj)
    return obj


def set_external_object_by_index(index: int, value: Any) -> None:
    """Update an entry in the external object registry at runtime."""
    keep_alive.append(value)
    index_to_external_object_weakref[index] = weakref.ref(value)


def get_external_object_by_index(index: int) -> Any:
    assert index in index_to_external_object_weakref, (
        "Index not registered in index_to_user_object_weakref"
    )
    obj = index_to_external_object_weakref[index]()
    assert obj is not None, "User object is no longer alive"
    return index_to_external_object_weakref[index]()


def store_user_object_weakrefs_by_index(indices: tuple[int, ...], *args: Any) -> None:
    global index_to_external_object_weakref
    assert len(indices) == len(args)
    index_to_external_object_weakref.clear()
    index_to_external_object_weakref.update(
        {i: weakref.ref(arg) for i, arg in zip(indices, args)}
    )


def reset_user_object_tracking() -> None:
    global next_user_object_index
    index_to_bytecode_constructor.clear()
    index_to_external_object_weakref.clear()
    keep_alive.clear()
    next_user_object_index = FIRST_USER_OBJECT_INDEX


def _try_store_external_object_weakref(index: int, value: Any) -> TypeError | None:
    try:
        index_to_external_object_weakref[index] = weakref.ref(value)
        return None
    except TypeError as e:
        return e


def _store_user_object_weakref(index: int, value: Any) -> None:
    if e := _try_store_external_object_weakref(index, value):
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to User Object",
            context=f"user_object: {value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )


def _store_graph_created_object_weakref(index: int, example_value: Any) -> None:
    if e := _try_store_external_object_weakref(index, example_value):
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to graph-created external object",
            context=f"user_object: {example_value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )


def _next_user_object_index() -> int:
    global next_user_object_index
    index = next_user_object_index
    next_user_object_index += 1
    return index


def register_graph_created_object(
    example_value: Any, construct_fn: Callable[[int, PyCodegen], None]
) -> int:
    global index_to_bytecode_constructor
    global keep_alive
    keep_alive.append(example_value)
    index = _next_user_object_index()
    index_to_bytecode_constructor[index] = lambda cg: construct_fn(index, cg)
    _store_graph_created_object_weakref(index, example_value)
    return index


# Register a user object to be used in the graph
def register_user_object(value: Any, source: Source) -> int:
    global index_to_bytecode_constructor
    index = _next_user_object_index()
    index_to_bytecode_constructor[index] = lambda cg: cg(source)
    _store_user_object_weakref(index, value)
    return index


def register_current_stream(value: Any, source: Source) -> int:
    global index_to_bytecode_constructor
    assert CURRENT_STREAM_INDEX not in index_to_bytecode_constructor, (
        f"Current stream index {CURRENT_STREAM_INDEX} is already registered"
    )
    index_to_bytecode_constructor[CURRENT_STREAM_INDEX] = lambda cg: cg(source)
    _store_user_object_weakref(CURRENT_STREAM_INDEX, value)
    return CURRENT_STREAM_INDEX


# Register a callback so invoke_leaf_function can retrieve nn.Module instances at runtime.
# We use a callback pattern instead of having invoke_leaf_function import get_external_object_by_index
# directly, because higher-order ops should not depend on dynamo (dynamo depends on them, not vice versa).
from torch._higher_order_ops.invoke_leaf_function import (
    set_leaf_function_module_retriever,
)


set_leaf_function_module_retriever(get_external_object_by_index)
