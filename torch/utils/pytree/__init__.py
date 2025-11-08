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

import os as _os
import sys as _sys
from types import ModuleType as _ModuleType
from typing import Any as _Any, Optional as _Optional, TYPE_CHECKING as _TYPE_CHECKING

import torch.utils._pytree as python
from torch.utils._pytree import (  # these type aliases are identical in both implementations
    FlattenFunc,
    FlattenWithKeysFunc,
    FromDumpableContextFunc,
    PyTree,
    ToDumpableContextFunc,
    UnflattenFunc,
)


if _TYPE_CHECKING:
    import torch.utils._cxx_pytree as cxx


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
    "is_namedtuple",
    "is_namedtuple_class",
    "is_namedtuple_instance",
    "is_structseq",
    "is_structseq_class",
    "is_structseq_instance",
]


# NB: Once this variable is read from the environment, the underlying pytree
#     implementation is frozen. It cannot be swapped to another at runtime.
PYTORCH_USE_CXX_PYTREE: bool = _os.getenv("PYTORCH_USE_CXX_PYTREE", "0") not in {
    "0",
    "",
}


def _import_cxx_pytree_and_store() -> _ModuleType:
    if not python._cxx_pytree_dynamo_traceable:
        raise ImportError(
            "Cannot import package `optree`. "
            "Please install `optree` via `python -m pip install --upgrade optree`. "
            "Or set the environment variable `PYTORCH_USE_CXX_PYTREE=0`."
        )

    import torch.utils._cxx_pytree as cxx

    # This allows the following statements to work properly:
    #
    #     import torch.utils.pytree
    #
    #     torch.utils.pytree.cxx
    #     torch.utils.pytree.cxx.tree_map
    #
    _sys.modules[f"{__name__}.cxx"] = globals()["cxx"] = cxx
    return cxx


if PYTORCH_USE_CXX_PYTREE:
    cxx = _import_cxx_pytree_and_store()  # noqa: F811
else:
    cxx = _sys.modules.get("torch.utils._cxx_pytree")  # type: ignore[assignment]


_sys.modules[f"{__name__}.python"] = python
if cxx is not None:
    _sys.modules[f"{__name__}.cxx"] = cxx
else:
    del cxx

    class LazyCxxModule(_ModuleType):
        def __getattr__(self, name: str) -> _Any:
            if name == "__name__":
                return f"{__name__}.cxx"
            if name == "__file__":
                return python.__file__.removesuffix("_python.py") + "_cxx_pytree.py"

            cxx = globals().get("cxx")
            if cxx is None:
                if name.startswith("_"):
                    raise AttributeError(
                        f"module {self.__name__!r} has not been imported yet: "
                        f"accessing attribute {name!r}. "
                        f"Please import {self.__name__!r} explicitly first."
                    )

                # Lazy import on first member access
                cxx = _import_cxx_pytree_and_store()

            return getattr(cxx, name)

        def __setattr__(self, name: str, value: _Any) -> None:
            # Lazy import
            cxx = _import_cxx_pytree_and_store()
            return setattr(cxx, name, value)

    # This allows the following statements to work properly:
    #
    #     import torch.utils.pytree.cxx
    #     from torch.utils.pytree.cxx import tree_map
    #
    _sys.modules[f"{__name__}.cxx"] = LazyCxxModule(f"{__name__}.cxx")

    del LazyCxxModule


if not PYTORCH_USE_CXX_PYTREE:
    from torch.utils._pytree import (
        is_namedtuple,
        is_namedtuple_class,
        is_namedtuple_instance,
        is_structseq,
        is_structseq_class,
        is_structseq_instance,
        PyTreeSpec,
        register_pytree_node as _register_pytree_node,
        tree_all,
        tree_all_only,
        tree_any,
        tree_any_only,
        tree_flatten,
        tree_iter,
        tree_leaves,
        tree_map,
        tree_map_,
        tree_map_only,
        tree_map_only_,
        tree_structure,
        tree_unflatten,
        treespec_pprint,
    )
else:
    from torch.utils._cxx_pytree import (  # type: ignore[assignment,no-redef]
        is_namedtuple,
        is_namedtuple_class,
        is_namedtuple_instance,
        is_structseq,
        is_structseq_class,
        is_structseq_instance,
        PyTreeSpec,
        register_pytree_node as _register_pytree_node,
        tree_all,
        tree_all_only,
        tree_any,
        tree_any_only,
        tree_flatten,
        tree_iter,
        tree_leaves,
        tree_map,
        tree_map_,
        tree_map_only,
        tree_map_only_,
        tree_structure,
        tree_unflatten,
        treespec_pprint,
    )


def register_pytree_node(
    cls: type[_Any],
    /,
    # intentionally use `*_func` over `*_fn` to match annotations
    flatten_func: FlattenFunc,
    unflatten_func: UnflattenFunc,
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

    Example::

        >>> # xdoctest: +SKIP
        >>> from collections import UserList
        ... class MyList(UserList): pass
        >>> # Registry a Python type with lambda functions
        ... register_pytree_node(
        ...     MyList,
        ...     lambda lst: (list(lst), None),
        ...     lambda children, _: MyList(children),
        ... )
    """
    _register_pytree_node(
        cls,
        flatten_func,
        unflatten_func,
    )


def __getattr__(name: str) -> _Any:
    if name == "cxx":
        # Lazy import
        return _import_cxx_pytree_and_store()

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
