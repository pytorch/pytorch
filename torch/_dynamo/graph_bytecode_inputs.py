import weakref
from typing import Any

from torch._dynamo.source import Source


# This file is to handle types that we don't want to support
# as explicit FX graph inputs. This uses a sidetable which
# we populate in bytecode and is loaded during graph execution

# We use a dynamo-generated index as a level of indirection
# this allows us to register objects externally in pre-graph bytecode that we want
# to pass to the graph, but not support their types as graph inputs
index_to_source: dict[int, Source] = {}

index_to_user_object_weakref: dict[int, weakref.ReferenceType[Any]] = {}


def has_user_objects() -> bool:
    return bool(index_to_source)


def get_user_object_by_index(index: int) -> Any:
    assert index in index_to_user_object_weakref, (
        "Index not registered in index_to_user_object_weakref"
    )
    obj = index_to_user_object_weakref[index]()
    assert obj is not None, "User object is no longer alive"
    return index_to_user_object_weakref[index]()


def store_user_object_weakrefs(*args: Any) -> None:
    global index_to_user_object_weakref
    index_to_user_object_weakref.clear()
    index_to_user_object_weakref.update(
        {i: weakref.ref(arg) for i, arg in enumerate(args)}
    )


def reset_user_object_tracking() -> None:
    index_to_source.clear()
    index_to_user_object_weakref.clear()


# Register a user object to be used in the graph
def register_user_object(value: Any, source: Source) -> int:
    global index_to_source
    index = len(index_to_source)
    index_to_source[index] = source
    try:
        index_to_user_object_weakref[index] = weakref.ref(value)
    except TypeError as e:
        from .exc import unimplemented_v2

        unimplemented_v2(
            gb_type="Failed to make weakref to User Object",
            context=f"user_object: {value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index
