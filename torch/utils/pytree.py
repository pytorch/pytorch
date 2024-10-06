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

import os
from typing import Callable, NamedTuple, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch.utils._pytree as python


if TYPE_CHECKING:
    from types import ModuleType

    import torch.utils._cxx_pytree as cxx
    from torch.utils._cxx_pytree import (
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


PYTORCH_USE_CXX_PYTREE: bool = os.getenv("PYTORCH_USE_CXX_PYTREE", "0") not in {"0", ""}


class PyTreeImplementation(NamedTuple):
    """The underlying implementation for PyTree utilities."""

    module: "ModuleType"
    name: str


implementation = PyTreeImplementation(module=python, name="python")
if PYTORCH_USE_CXX_PYTREE:
    if not python._cxx_pytree_exists:
        raise ImportError(
            "Cannot import package `optree`. "
            "Please install `optree` via `python -m pip install --upgrade optree`."
        )

    import torch.utils._cxx_pytree as cxx  # noqa: F811

    implementation = PyTreeImplementation(module=cxx, name="cxx")


_P = ParamSpec("_P")
_T = TypeVar("_T")


def _reexport(func: Callable[_P, _T]) -> Callable[_P, _T]:
    import functools

    name = func.__name__

    @functools.wraps(func)
    def exported(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        impl: Callable[_P, _T] = getattr(implementation.module, name)
        return impl(*args, **kwargs)

    exported.__module__ = __name__
    return exported


tree_flatten = _reexport(implementation.module.tree_flatten)  # noqa: F811
tree_unflatten = _reexport(implementation.module.tree_unflatten)  # noqa: F811
tree_iter = _reexport(implementation.module.tree_iter)  # noqa: F811
tree_leaves = _reexport(implementation.module.tree_leaves)  # noqa: F811
tree_structure = _reexport(implementation.module.tree_structure)  # noqa: F811
tree_map = _reexport(implementation.module.tree_map)  # noqa: F811
tree_map_ = _reexport(implementation.module.tree_map_)  # noqa: F811
tree_map_only = _reexport(implementation.module.tree_map_only)  # noqa: F811
tree_map_only_ = _reexport(implementation.module.tree_map_only_)  # noqa: F811
tree_all = _reexport(implementation.module.tree_all)  # noqa: F811
tree_any = _reexport(implementation.module.tree_any)  # noqa: F811
tree_all_only = _reexport(implementation.module.tree_all_only)  # noqa: F811
tree_any_only = _reexport(implementation.module.tree_any_only)  # noqa: F811
treespec_pprint = _reexport(implementation.module.treespec_pprint)  # noqa: F811


del _reexport
del PyTreeImplementation
