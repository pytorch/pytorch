# Owner(s): ["module: pytree"]

"""
Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `map` to map a function over all Tensors inside some nested
collection of Tensors and `leaves` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.
"""

from __future__ import annotations

from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

from torch.utils.pytree import (
    is_namedtuple,
    is_namedtuple_class,
    is_namedtuple_instance,
    is_structseq,
    is_structseq_class,
    is_structseq_instance,
    PyTree,
    PyTreeSpec,
    register_pytree_node as register_node,
    tree_all as all,
    tree_all_only as all_only,
    tree_any as any,
    tree_any_only as any_only,
    tree_flatten as flatten,
    tree_iter as iter,
    tree_leaves as leaves,
    tree_map as map,
    tree_map_ as map_,
    tree_map_only as map_only,
    tree_map_only_ as map_only_,
    tree_structure as structure,
    tree_unflatten as _tree_unflatten,
)


if _TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "PyTreeSpec",
    "register_node",
    "flatten",
    "unflatten",
    "iter",
    "leaves",
    "structure",
    "map",
    "map_",
    "map_only",
    "map_only_",
    "all",
    "any",
    "all_only",
    "any_only",
    "is_namedtuple",
    "is_namedtuple_class",
    "is_namedtuple_instance",
    "is_structseq",
    "is_structseq_class",
    "is_structseq_instance",
]


def unflatten(treespec: PyTreeSpec, leaves: Iterable[_Any]) -> PyTree:
    """Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`flatten`.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> leaves, treespec = torch.pytree.flatten(tree)
    >>> tree == torch.pytree.unflatten(treespec, leaves)
    True

    .. warning::

        This function has a different signature than :func:`torch.utils.pytree.tree_unflatten`.
        The ``treespec`` argument comes first to have a better :class:`functools.partial` support:

        .. code-block:: python

            import functools

            unflatten_fn = functools.partial(torch.pytree.unflatten, treespec)
            tree1 = unflatten_fn(leaves1)
            tree2 = unflatten_fn(leaves2)

    Args:
        treespec (PyTreeSpec): The treespec to reconstruct.
        leaves (iterable): The list of leaves to use for reconstruction. The list must match the
            number of leaves of the treespec.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by
        ``treespec``.
    """
    # pyrefly: ignore [bad-argument-type]
    return _tree_unflatten(leaves, treespec)
