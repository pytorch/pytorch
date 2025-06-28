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
from typing import Any as _Any, Optional as _Optional

import torch.utils._pytree as python
from torch.utils._exposed_in import exposed_in as _exposed_in
from torch.utils._pytree import (  # these type aliases are identical in both implementations
    FlattenFunc,
    FlattenWithKeysFunc,
    FromDumpableContextFunc,
    PyTree,
    ToDumpableContextFunc,
    UnflattenFunc,
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


if PYTORCH_USE_CXX_PYTREE:
    import torch.utils._cxx_pytree as cxx  # noqa: F401

    if not python._cxx_pytree_dynamo_traceable:
        raise ImportError(
            "Cannot import package `optree`. "
            "Please install `optree` via `python -m pip install --upgrade optree`. "
            "Or set the environment variable `PYTORCH_USE_CXX_PYTREE=0`."
        )


_sys.modules[f"{__name__}.cxx"] = _sys.modules.get("torch.utils._cxx_pytree")  # type: ignore[assignment]


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

    PyTreeSpec = _exposed_in(__name__)(PyTreeSpec)  # type: ignore[misc]
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


# Change `__module__` of reexported public APIs to 'torch.utils.pytree'
__func_names = frozenset(
    {
        "tree_all",
        "tree_all_only",
        "tree_any",
        "tree_any_only",
        "tree_flatten",
        "tree_iter",
        "tree_leaves",
        "tree_map",
        "tree_map_",
        "tree_map_only",
        "tree_map_only_",
        "tree_structure",
        "tree_unflatten",
        "treespec_pprint",
        "is_namedtuple",
        "is_namedtuple_class",
        "is_namedtuple_instance",
        "is_structseq",
        "is_structseq_class",
        "is_structseq_instance",
    }
)
globals().update(
    {
        name: _exposed_in(__name__)(member)
        for name, member in globals().items()
        if name in __func_names
    }
)
del __func_names, _exposed_in


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
        import torch.utils._cxx_pytree as cxx  # noqa: F811

        _sys.modules[f"{__name__}.cxx"] = globals()["cxx"] = cxx
        return cxx

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
