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
    PyTree as PyTree,
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
]


# NB: Once this variable is read from the environment, the underlying pytree
#     implementation is frozen. It cannot be swapped to another at runtime.
PYTORCH_USE_CXX_PYTREE: bool = _os.getenv("PYTORCH_USE_CXX_PYTREE", "0") not in {
    "0",
    "",
}


def _import_cxx_pytree() -> _ModuleType:
    if not python._cxx_pytree_dynamo_traceable:
        raise ImportError(
            "Cannot import package `optree`. "
            "Please install `optree` via `python -m pip install --upgrade optree`. "
            "Or set the environment variable `PYTORCH_USE_CXX_PYTREE=0`."
        )

    import torch.utils._cxx_pytree as cxx

    return cxx


if PYTORCH_USE_CXX_PYTREE:
    cxx = _import_cxx_pytree()  # noqa: F811
else:
    cxx = _sys.modules.get("torch.utils._cxx_pytree")  # type: ignore[assignment]


_sys.modules[f"{__name__}.python"] = python
if cxx is not None:
    _sys.modules[f"{__name__}.cxx"] = cxx
else:
    del cxx

    class CxxModule(_ModuleType):
        def __getattr__(self, name: str) -> _Any:
            if name == "__name__":
                return f"{__name__}.cxx"
            if name == "__file__":
                return python.__file__.removesuffix("_python.py") + "_cxx_pytree.py"
            if name.startswith("_"):
                raise AttributeError(
                    f"module {self.__name__!r} has not been imported yet: "
                    f"accessing attribute {name!r}"
                )

            # Lazy import
            cxx = _import_cxx_pytree()

            # Replace the temporary module object (`self`) in sys.modules
            _sys.modules[f"{__name__}.cxx"] = globals()["cxx"] = cxx
            return getattr(cxx, name)

    # This allows the following statements to work properly:
    #
    #     from torch.utils.pytree.cxx import tree_map
    #
    _sys.modules[f"{__name__}.cxx"] = CxxModule(f"{__name__}.cxx")

    del CxxModule


if not PYTORCH_USE_CXX_PYTREE:
    from torch.utils._pytree import (
        PyTreeSpec as PyTreeSpec,
        register_pytree_node as _register_pytree_node,
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
else:
    from torch.utils._cxx_pytree import (  # type: ignore[assignment,no-redef]
        PyTreeSpec as PyTreeSpec,
        register_pytree_node as _register_pytree_node,
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


def register_pytree_node(  # type: ignore[no-any-unimported]
    cls: type[_Any],
    /,
    # intentionally use `*_func` over `*_fn` to match annotations
    flatten_func: FlattenFunc,
    unflatten_func: UnflattenFunc,
    *,
    serialized_type_name: _Optional[str] = None,
    to_dumpable_context: _Optional[ToDumpableContextFunc] = None,
    from_dumpable_context: _Optional[FromDumpableContextFunc] = None,
    # intentionally use `*_func` over `*_fn` to match annotations
    flatten_with_keys_func: _Optional[FlattenWithKeysFunc] = None,
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
    _register_pytree_node(
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


def __getattr__(name: str) -> _Any:
    if name == "cxx":
        # Lazy import
        cxx = _import_cxx_pytree()

        # This allows the following statements to work properly:
        #
        #     import torch.utils.pytree
        #
        #     torch.utils.pytree.cxx
        #
        _sys.modules[f"{__name__}.cxx"] = globals()["cxx"] = cxx
        return cxx

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
