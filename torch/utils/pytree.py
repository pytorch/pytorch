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

import os as _os
from dataclasses import dataclass as _dataclass
from typing import (
    Any as _Any,
    Callable as _Callable,
    Literal as _Literal,
    TYPE_CHECKING as _TYPE_CHECKING,
    TypeVar as _TypeVar,
)
from typing_extensions import ParamSpec as _ParamSpec

import torch.utils._pytree as python


if _TYPE_CHECKING:
    from types import ModuleType

    import torch.utils._cxx_pytree as cxx
    from torch.utils._cxx_pytree import (  # noqa: TCH004; noqa: F401
        _broadcast_to_and_flatten as _broadcast_to_and_flatten,
        PyTreeSpec as PyTreeSpec,
        register_pytree_node as register_pytree_node,
        tree_all as tree_all,
        tree_all_only as tree_all_only,
        tree_any as tree_any,
        tree_any_only as tree_any_only,
        tree_flatten as tree_flatten,
        tree_iter as tree_iter,
        tree_leaves as tree_leaves,
        tree_map as tree_map,
        tree_map_ as tree_map_,
        tree_map_only as tree_map_only,
        tree_map_only_ as tree_map_only_,
        tree_structure as tree_structure,
        tree_unflatten as tree_unflatten,
        treespec_pprint as treespec_pprint,
    )


__all__ = [
    "PyTreeSpec",
    "register_pytree_node",
    "tree_flatten",
    "tree_unflatten",
    "tree_iter",
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
    "treespec_pprint",
]


PYTORCH_USE_CXX_PYTREE: bool = _os.getenv("PYTORCH_USE_CXX_PYTREE", "0") not in {
    "0",
    "",
}


@_dataclass(frozen=True)
class PyTreeImplementation:
    """The underlying implementation for PyTree utilities."""

    module: "ModuleType" = python
    name: _Literal["python", "cxx"] = "python"


implementation = PyTreeImplementation(module=python, name="python")
if PYTORCH_USE_CXX_PYTREE:
    if not python._cxx_pytree_exists:
        raise ImportError(
            "Cannot import package `optree`. "
            "Please install `optree` via `python -m pip install --upgrade optree`."
        )

    import torch.utils._cxx_pytree as cxx

    implementation = PyTreeImplementation(module=cxx, name="cxx")


_P = _ParamSpec("_P")
_T = _TypeVar("_T")


def _reexport(func: _Callable[_P, _T]) -> _Callable[_P, _T]:
    import functools

    name = func.__name__

    @functools.wraps(func)
    def exported(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        impl: _Callable[_P, _T] = getattr(implementation.module, name)
        return impl(*args, **kwargs)

    exported.__module__ = __name__
    return exported


# flake8: noqa: F811
register_pytree_node = _reexport(implementation.module.register_pytree_node)
tree_flatten = _reexport(implementation.module.tree_flatten)
tree_unflatten = _reexport(implementation.module.tree_unflatten)
tree_iter = _reexport(implementation.module.tree_iter)
tree_leaves = _reexport(implementation.module.tree_leaves)
tree_structure = _reexport(implementation.module.tree_structure)
tree_map = _reexport(implementation.module.tree_map)
tree_map_ = _reexport(implementation.module.tree_map_)
tree_map_only = _reexport(implementation.module.tree_map_only)
tree_map_only_ = _reexport(implementation.module.tree_map_only_)
tree_all = _reexport(implementation.module.tree_all)
tree_any = _reexport(implementation.module.tree_any)
tree_all_only = _reexport(implementation.module.tree_all_only)
tree_any_only = _reexport(implementation.module.tree_any_only)
treespec_pprint = _reexport(implementation.module.treespec_pprint)


# Used in vmap
_broadcast_to_and_flatten = _reexport(implementation.module._broadcast_to_and_flatten)


del _reexport
del PyTreeImplementation


# Use the __getattr__ function allowing us to change the underlying `implementation` at runtime.
def __getattr__(name: str) -> _Any:
    name = {"PyTreeSpec": "TreeSpec"}.get(name, name)
    try:
        return getattr(implementation.module, name)
    except AttributeError as ex:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}: "
            f"no attribute {name!r} in "
            f"{implementation.name} implementation: {implementation.module.__name__!r}"
        ) from ex
