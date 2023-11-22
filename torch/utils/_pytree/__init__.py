"""
Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_leaves` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.
"""

import warnings
from typing import Any

from .api import (
    Context,
    DumpableContext,
    FlattenFunc,
    FromDumpableContextFn,
    LeafSpec,
    PyTree,
    register_pytree_node,
    ToDumpableContextFn,
    tree_all,
    tree_all_only,
    tree_any,
    tree_any_only,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_map_,
    tree_map_only,
    tree_map_only_,
    tree_structure,
    tree_unflatten,
    TreeSpec,
    treespec_dumps,
    treespec_loads,
    treespec_pprint,
    UnflattenFunc,
)
from .api.python import (  # used by internals and/or third-party packages
    _broadcast_to_and_flatten,
    _dict_flatten,
    _dict_unflatten,
    _get_node_type,
    _is_leaf,
    _list_flatten,
    _list_unflatten,
    _namedtuple_flatten,
    _namedtuple_unflatten,
    _odict_flatten,
    _odict_unflatten,
    _ordereddict_flatten,
    _ordereddict_unflatten,
    _register_pytree_node,
    _tuple_flatten,
    _tuple_unflatten,
    arg_tree_leaves,
    SUPPORTED_NODES,
)


__all__ = [
    "PyTree",
    "Context",
    "FlattenFunc",
    "UnflattenFunc",
    "DumpableContext",
    "ToDumpableContextFn",
    "FromDumpableContextFn",
    "TreeSpec",
    "LeafSpec",
    "register_pytree_node",
    "tree_flatten",
    "tree_unflatten",
    "tree_leaves",
    "tree_structure",
    "tree_map",
    "tree_map_",
    "tree_map_only",
    "tree_map_only_",
    "tree_all",
    "tree_any",
    "tree_all_only",
    "tree_any_only",
    "treespec_dumps",
    "treespec_loads",
    "treespec_pprint",
]


def __getattr__(name: str) -> Any:
    """Fallback path to forward private members in Python pytree to the top level module."""
    from .api import python

    try:
        member = getattr(python, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    warnings.warn(
        f"{__name__}.{name} is a private member "
        "which might be changed or removed in future releases.",
        stacklevel=2,
    )
    return member
