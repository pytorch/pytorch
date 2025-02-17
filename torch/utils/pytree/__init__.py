# Owner(s): ["module: pytree"]

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

from __future__ import annotations

import os as _os
import sys as _sys
from dataclasses import dataclass as _dataclass
from typing import (
    Any as _Any,
    Callable as _Callable,
    Literal as _Literal,
    TYPE_CHECKING as _TYPE_CHECKING,
    TypeVar as _TypeVar,
)
from typing_extensions import ParamSpec as _ParamSpec, Self as _Self

import torch.utils._pytree as python
from torch._utils import classproperty as _classproperty
from torch.utils._pytree import (  # these type aliases are identical in both implementations
    FlattenFunc as FlattenFunc,
    FlattenWithKeysFunc as FlattenWithKeysFunc,
    FromDumpableContextFunc as FromDumpableContextFunc,
    ToDumpableContextFunc as ToDumpableContextFunc,
    UnflattenFunc as UnflattenFunc,
)


if _TYPE_CHECKING:
    from types import ModuleType

    from torch.utils._cxx_pytree import (  # noqa: TC004
        PyTreeSpec as PyTreeSpec,
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


if _TYPE_CHECKING:
    __all__ += [
        "FlattenFunc",
        "UnflattenFunc",
        "FlattenWithKeysFunc",
        "ToDumpableContextFunc",
        "FromDumpableContextFunc",
    ]


PYTORCH_USE_CXX_PYTREE: bool = _os.getenv("PYTORCH_USE_CXX_PYTREE", "0") not in {
    "0",
    "",
}


@_dataclass(frozen=True)
class PyTreeImplementation:
    """The underlying implementation for PyTree utilities."""

    module: ModuleType = python
    name: _Literal["python", "cxx"] = "python"

    @_classproperty  # type: ignore[misc]
    @classmethod
    def python(cls) -> _Self:
        """The Python implementation."""
        return cls(module=python, name="python")

    @_classproperty  # type: ignore[misc]
    @classmethod
    def cxx(cls) -> _Self:
        """The C++ implementation."""
        import torch.utils._cxx_pytree as cxx

        return cls(module=cxx, name="cxx")


implementation = PyTreeImplementation.python
if PYTORCH_USE_CXX_PYTREE:
    import torch.utils._cxx_pytree as cxx  # noqa: F401

    if not python._cxx_pytree_dynamo_traceable:
        raise ImportError(
            "Cannot import package `optree`. "
            "Please install `optree` via `python -m pip install --upgrade optree`."
        )
    implementation = PyTreeImplementation.cxx


_sys.modules[f"{__name__}.python"] = python
_sys.modules[f"{__name__}.cxx"] = _sys.modules.get("torch.utils._cxx_pytree")  # type: ignore[assignment]


def register_pytree_node(
    cls: type[_Any],
    /,
    # intentionally use `*_func` over `*_fn` to match annotations
    flatten_func: FlattenFunc,
    unflatten_func: UnflattenFunc,
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFunc | None = None,
    from_dumpable_context: FromDumpableContextFunc | None = None,
    # intentionally use `*_func` over `*_fn` to match annotations
    flatten_with_keys_func: FlattenWithKeysFunc | None = None,
) -> None:
    """Register a container-like type as pytree node.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_func (callable): A function to be used during flattening, taking an instance of
            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened
            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be
            passed to the ``unflatten_func``.
        unflatten_func (callable): A function taking two arguments: the unflattened children, and
            the auxiliary data that was returned by ``flatten_func`` and stored in the treespec.
            The function should return an instance of ``cls``.
        serialized_type_name (str, optional): A keyword argument used to specify the fully
            qualified name used when serializing the tree spec.
        to_dumpable_context (callable, optional): An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable representation. This is
            used for json serialization, which is being used in :mod:`torch.export` right now.
        from_dumpable_context (callable, optional): An optional keyword argument to custom specify
            how to convert the custom json dumpable representation of the context back to the
            original context. This is used for json deserialization, which is being used in
            :mod:`torch.export` right now.

    Example::

        >>> # xdoctest: +SKIP
        >>> # Registry a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None),
        ...     lambda children, _: set(children),
        ... )
    """
    implementation.module.register_pytree_node(
        cls,
        # intentionally use `*_func` over `*_fn` to match annotations
        flatten_fn=flatten_func,
        unflatten_fn=unflatten_func,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
        # intentionally use `*_func` over `*_fn` to match annotations
        flatten_with_keys_fn=flatten_with_keys_func,
    )


_P = _ParamSpec("_P")
_R = _TypeVar("_R")


def _reexport(func: _Callable[_P, _R]) -> _Callable[_P, _R]:
    import functools

    name = func.__name__

    @functools.wraps(func)
    def exported(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # Dynamically get the implementation function from the module to allow changing the
        # implementation at runtime.
        impl: _Callable[_P, _R] = getattr(implementation.module, name)
        return impl(*args, **kwargs)

    exported.__module__ = __name__
    return exported


# flake8: noqa: F811
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


del _reexport
del PyTreeImplementation


# Use the __getattr__ function allowing us to change the underlying `implementation` at runtime.
def __getattr__(name: str) -> _Any:
    if name == "cxx":
        import torch.utils._cxx_pytree as cxx

        globals()["cxx"] = cxx
        _sys.modules[f"{__name__}.cxx"] = cxx
        return cxx

    try:
        return getattr(implementation.module, name)
    except AttributeError as ex:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}: "
            f"no attribute {name!r} in "
            f"{implementation.name} implementation: {implementation.module.__name__!r}"
        ) from ex
