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


def has_user_objects() -> bool:
    return bool(index_to_bytecode_constructor)


def stash_graph_created_object(obj: Any) -> Any:
    keep_alive.append(obj)
    return obj


def get_external_object_by_index(index: int) -> Any:
    assert index in index_to_external_object_weakref, (
        "Index not registered in index_to_user_object_weakref"
    )
    obj = index_to_external_object_weakref[index]()
    assert obj is not None, "User object is no longer alive"
    return index_to_external_object_weakref[index]()


def store_user_object_weakrefs(*args: Any) -> None:
    global index_to_external_object_weakref
    index_to_external_object_weakref.clear()
    index_to_external_object_weakref.update(
        {i: weakref.ref(arg) for i, arg in enumerate(args)}
    )


def reset_user_object_tracking() -> None:
    index_to_bytecode_constructor.clear()
    index_to_external_object_weakref.clear()
    keep_alive.clear()


def register_graph_created_object(
    example_value: Any, construct_fn: Callable[[int, PyCodegen], None]
) -> int:
    global index_to_bytecode_constructor
    global keep_alive
    keep_alive.append(example_value)
    index = len(index_to_bytecode_constructor)
    index_to_bytecode_constructor[index] = lambda cg: construct_fn(index, cg)
    try:
        index_to_external_object_weakref[index] = weakref.ref(example_value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to graph-created external object",
            context=f"user_object: {example_value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index


# Register a user object to be used in the graph
def register_user_object(value: Any, source: Source) -> int:
    global index_to_bytecode_constructor
    index = len(index_to_bytecode_constructor)
    index_to_bytecode_constructor[index] = lambda cg: cg(source)
    try:
        index_to_external_object_weakref[index] = weakref.ref(value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to User Object",
            context=f"user_object: {value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index
