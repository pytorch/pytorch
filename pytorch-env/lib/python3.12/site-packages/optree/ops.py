# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""OpTree: Optimized PyTree Utilities."""

# pylint: disable=too-many-lines

from __future__ import annotations

import difflib
import functools
import itertools
import textwrap
from collections import OrderedDict, defaultdict, deque
from typing import Any, Callable, ClassVar, Iterable, Mapping, overload

from optree import _C
from optree.accessor import PyTreeAccessor, PyTreeEntry
from optree.registry import PyTreeNodeRegistryEntry, register_pytree_node
from optree.typing import (
    CustomTreeNode,
    MetaData,
    NamedTuple,
    PyTree,
    PyTreeSpec,
    S,
    T,
    U,
    is_namedtuple_instance,
    is_structseq_instance,
)
from optree.typing import structseq as PyStructSequence  # noqa: N812


__all__ = [
    'MAX_RECURSION_DEPTH',
    'NONE_IS_NODE',
    'NONE_IS_LEAF',
    'tree_flatten',
    'tree_flatten_with_path',
    'tree_flatten_with_accessor',
    'tree_unflatten',
    'tree_iter',
    'tree_leaves',
    'tree_structure',
    'tree_paths',
    'tree_accessors',
    'tree_is_leaf',
    'all_leaves',
    'tree_map',
    'tree_map_',
    'tree_map_with_path',
    'tree_map_with_path_',
    'tree_map_with_accessor',
    'tree_map_with_accessor_',
    'tree_replace_nones',
    'tree_transpose',
    'tree_transpose_map',
    'tree_transpose_map_with_path',
    'tree_transpose_map_with_accessor',
    'tree_broadcast_prefix',
    'broadcast_prefix',
    'tree_broadcast_common',
    'broadcast_common',
    'tree_broadcast_map',
    'tree_broadcast_map_with_path',
    'tree_broadcast_map_with_accessor',
    'tree_reduce',
    'tree_sum',
    'tree_max',
    'tree_min',
    'tree_all',
    'tree_any',
    'tree_flatten_one_level',
    'treespec_paths',
    'treespec_accessors',
    'treespec_entries',
    'treespec_entry',
    'treespec_children',
    'treespec_child',
    'treespec_is_leaf',
    'treespec_is_strict_leaf',
    'treespec_is_prefix',
    'treespec_is_suffix',
    'treespec_leaf',
    'treespec_none',
    'treespec_tuple',
    'treespec_list',
    'treespec_dict',
    'treespec_namedtuple',
    'treespec_ordereddict',
    'treespec_defaultdict',
    'treespec_deque',
    'treespec_structseq',
    'treespec_from_collection',
    'prefix_errors',
]

MAX_RECURSION_DEPTH: int = _C.MAX_RECURSION_DEPTH  # 1000
"""Maximum recursion depth for pytree traversal. It is 1000.

This limit prevents infinite recursion from causing an overflow of the C stack
and crashing Python.
"""
NONE_IS_NODE: bool = False  # literal constant
"""Literal constant that treats :data:`None` as a pytree non-leaf node."""
NONE_IS_LEAF: bool = True  # literal constant
"""Literal constant that treats :data:`None` as a pytree leaf node."""


def tree_flatten(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[T], PyTreeSpec]:
    """Flatten a pytree.

    See also :func:`tree_flatten_with_path` and :func:`tree_unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> tree_flatten(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [1, 2, 3, 4, None, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    )
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*))
    >>> tree_flatten(None)
    ([], PyTreeSpec(None))
    >>> tree_flatten(None, none_is_leaf=True)
    ([None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [2, 3, 4, 1, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}))
    )
    >>> tree_flatten(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [2, 3, 4, 1, None, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}), NoneIsLeaf)
    )

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the
        second element is a treespec representing the structure of the pytree.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf, namespace)


def tree_flatten_with_path(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[tuple[Any, ...]], list[T], PyTreeSpec]:
    """Flatten a pytree and additionally record the paths.

    See also :func:`tree_flatten`, :func:`tree_paths`, and :func:`treespec_paths`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten_with_path(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('d',)],
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> tree_flatten_with_path(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('c',), ('d',)],
        [1, 2, 3, 4, None, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    )
    >>> tree_flatten_with_path(1)
    ([()], [1], PyTreeSpec(*))
    >>> tree_flatten_with_path(None)
    ([], [], PyTreeSpec(None))
    >>> tree_flatten_with_path(None, none_is_leaf=True)
    ([()], [None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten_with_path(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [('b', 0), ('b', 1, 0), ('b', 1, 1), ('a',), ('d',)],
        [2, 3, 4, 1, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}))
    )
    >>> tree_flatten_with_path(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [('b', 0), ('b', 1, 0), ('b', 1, 1), ('a',), ('c',), ('d',)],
        [2, 3, 4, 1, None, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}), NoneIsLeaf)
    )

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A triple ``(paths, leaves, treespec)``. The first element is a list of the paths to the leaf
        values, while each path is a tuple of the index or keys. The second element is a list of
        leaf values and the last element is a treespec representing the structure of the pytree.
    """
    return _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)


def tree_flatten_with_accessor(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[PyTreeAccessor], list[T], PyTreeSpec]:
    """Flatten a pytree and additionally record the accessors.

    See also :func:`tree_flatten`, :func:`tree_accessors`, and :func:`treespec_accessors`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten_with_accessor(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [
            PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
            PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
            PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
            PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
        ],
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> tree_flatten_with_accessor(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [
            PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
            PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
            PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
            PyTreeAccessor(*['c'], (MappingEntry(key='c', type=<class 'dict'>),)),
            PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
        ],
        [1, 2, 3, 4, None, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    )
    >>> tree_flatten_with_accessor(1)
    ([PyTreeAccessor(*, ())], [1], PyTreeSpec(*))
    >>> tree_flatten_with_accessor(None)
    ([], [], PyTreeSpec(None))
    >>> tree_flatten_with_accessor(None, none_is_leaf=True)
    ([PyTreeAccessor(*, ())], [None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten_with_accessor(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [
            PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
            PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
            PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'collections.OrderedDict'>),)),
            PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'collections.OrderedDict'>),))
        ],
        [2, 3, 4, 1, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}))
    )
    >>> tree_flatten_with_accessor(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [
            PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
            PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'collections.OrderedDict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
            PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'collections.OrderedDict'>),)),
            PyTreeAccessor(*['c'], (MappingEntry(key='c', type=<class 'collections.OrderedDict'>),)),
            PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'collections.OrderedDict'>),))
        ],
        [2, 3, 4, 1, None, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}), NoneIsLeaf)
    )

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A triple ``(accessors, leaves, treespec)``. The first element is a list of accessors to the
        leaf values. The second element is a list of leaf values and the last element is a treespec
        representing the structure of the pytree.
    """  # pylint: disable=line-too-long
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    return treespec.accessors(), leaves, treespec


def tree_unflatten(treespec: PyTreeSpec, leaves: Iterable[T]) -> PyTree[T]:
    """Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> leaves, treespec = tree_flatten(tree)
    >>> tree == tree_unflatten(treespec, leaves)
    True

    Args:
        treespec (PyTreeSpec): The treespec to reconstruct.
        leaves (iterable): The list of leaves to use for reconstruction. The list must match the
            number of leaves of the treespec.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by
        ``treespec``.
    """
    return treespec.unflatten(leaves)


def tree_iter(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> Iterable[T]:
    """Get an iterator over the leaves of a pytree.

    See also :func:`tree_flatten` and :func:`tree_leaves`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> list(tree_iter(tree))
    [1, 2, 3, 4, 5]
    >>> list(tree_iter(tree, none_is_leaf=True))
    [1, 2, 3, 4, None, 5]
    >>> list(tree_iter(1))
    [1]
    >>> list(tree_iter(None))
    []
    >>> list(tree_iter(None, none_is_leaf=True))
    [None]

    Args:
        tree (pytree): A pytree to iterate over.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        An iterator over the leaf values.
    """
    return _C.PyTreeIter(tree, is_leaf, none_is_leaf, namespace)


def tree_leaves(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[T]:
    """Get the leaves of a pytree.

    See also :func:`tree_flatten` and :func:`tree_iter`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_leaves(tree)
    [1, 2, 3, 4, 5]
    >>> tree_leaves(tree, none_is_leaf=True)
    [1, 2, 3, 4, None, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    []
    >>> tree_leaves(None, none_is_leaf=True)
    [None]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A list of leaf values.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf, namespace)[0]


def tree_structure(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Get the treespec for a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_structure(tree)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    >>> tree_structure(tree, none_is_leaf=True)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    >>> tree_structure(1)
    PyTreeSpec(*)
    >>> tree_structure(None)
    PyTreeSpec(None)
    >>> tree_structure(None, none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec object representing the structure of the pytree.
    """
    return _C.flatten(tree, is_leaf, none_is_leaf, namespace)[1]


def tree_paths(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[tuple[Any, ...]]:
    """Get the path entries to the leaves of a pytree.

    See also :func:`tree_flatten`, :func:`tree_flatten_with_path`, and :func:`treespec_paths`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_paths(tree)
    [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('d',)]
    >>> tree_paths(tree, none_is_leaf=True)
    [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('c',), ('d',)]
    >>> tree_paths(1)
    [()]
    >>> tree_paths(None)
    []
    >>> tree_paths(None, none_is_leaf=True)
    [()]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A list of the paths to the leaf values, while each path is a tuple of the index or keys.
    """
    return _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)[0]


def tree_accessors(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[PyTreeAccessor]:
    """Get the accessors to the leaves of a pytree.

    See also :func:`tree_flatten`, :func:`tree_flatten_with_accessor`, and
    :func:`treespec_accessors`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_accessors(tree)  # doctest: +IGNORE_WHITESPACE
    [
        PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
        PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
        PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
        PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
        PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
    ]
    >>> tree_accessors(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    [
        PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
        PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
        PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
        PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>))),
        PyTreeAccessor(*['c'], (MappingEntry(key='c', type=<class 'dict'>),)),
        PyTreeAccessor(*['d'], (MappingEntry(key='d', type=<class 'dict'>),))
    ]
    >>> tree_accessors(1)
    [PyTreeAccessor(*, ())]
    >>> tree_accessors(None)
    []
    >>> tree_accessors(None, none_is_leaf=True)
    [PyTreeAccessor(*, ())]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A list of accessors to the leaf values.
    """  # pylint: disable=line-too-long
    return _C.flatten(tree, is_leaf, none_is_leaf, namespace)[1].accessors()


def tree_is_leaf(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether the given object is a leaf node.

    See also :func:`tree_flatten`, :func:`tree_leaves`, and :func:`all_leaves`.

    >>> tree_is_leaf(1)
    True
    >>> tree_is_leaf(None)
    False
    >>> tree_is_leaf(None, none_is_leaf=True)
    True
    >>> tree_is_leaf({'a': 1, 'b': (2, 3)})
    False

    Args:
        tree (pytree): A pytree to check if it is a leaf node.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than a leaf. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A boolean indicating if all elements in the input iterable are leaves.
    """
    return _C.is_leaf(tree, is_leaf, none_is_leaf, namespace)  # type: ignore[arg-type]


def all_leaves(
    iterable: Iterable[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether all elements in the given iterable are all leaves.

    See also :func:`tree_flatten`, :func:`tree_leaves`, and :func:`tree_is_leaf`.

    >>> tree = {'a': [1, 2, 3]}
    >>> all_leaves(tree_leaves(tree))
    True
    >>> all_leaves([tree])
    False
    >>> all_leaves([1, 2, None, 3])
    False
    >>> all_leaves([1, 2, None, 3], none_is_leaf=True)
    True

    Note that this function iterates and checks the elements in the input iterable object, which
    uses the :func:`iter` function. For dictionaries, ``iter(d)`` for a dictionary ``d`` iterates
    the keys of the dictionary, not the values.

    >>> list({'a': 1, 'b': (2, 3)})
    ['a', 'b']
    >>> all_leaves({'a': 1, 'b': (2, 3)})
    True

    This function is useful in advanced cases. For example, if a library allows arbitrary map
    operations on a flat list of leaves it may want to check if the result is still a flat list
    of leaves.

    Args:
        iterable (iterable): A iterable of leaves.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than a leaf. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A boolean indicating if all elements in the input iterable are leaves.
    """
    return _C.all_leaves(iterable, is_leaf, none_is_leaf, namespace)


def tree_map(
    func: Callable[..., U],
    tree: PyTree[T],
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`, :func:`tree_map_with_path`, :func:`tree_map_with_path_`,
    and :func:`tree_broadcast_map`.

    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (43, 65), 'z': None}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': None}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None}, none_is_leaf=True)
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:

    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, *flat_args))


def tree_map_(
    func: Callable[..., Any],
    tree: PyTree[T],
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`, :func:`tree_map_with_path`, and :func:`tree_map_with_path_`.

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.
    """
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, *flat_args), maxlen=0)  # consume and exhaust the iterable
    return tree


def tree_map_with_path(
    func: Callable[..., U],
    tree: PyTree[T],
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree.

    See also :func:`tree_map`, :func:`tree_map_`, and :func:`tree_map_with_path_`.

    >>> tree_map_with_path(lambda p, x: (len(p), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> tree_map_with_path(lambda p, x: x + len(p), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> tree_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}})
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: None}}
    >>> tree_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}}, none_is_leaf=True)
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: ('z', 1.5)}}

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(p, x, *xs)`` where ``(p, x)`` are the path and value at the corresponding leaf in
        ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    paths, leaves, treespec = _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, paths, *flat_args))


def tree_map_with_path_(
    func: Callable[..., Any],
    tree: PyTree[T],
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:
    """Like :func:`tree_map_with_path`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`, :func:`tree_map_`, and :func:`tree_map_with_path`.

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(p, x, *xs)`` (not the return value) where ``(p, x)`` are the path and value at the
        corresponding leaf in ``tree`` and ``xs`` is the tuple of values at values at corresponding
        nodes in ``rests``.
    """
    paths, leaves, treespec = _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, paths, *flat_args), maxlen=0)  # consume and exhaust the iterable
    return tree


def tree_map_with_accessor(
    func: Callable[..., U],
    tree: PyTree[T],
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree accessors to produce a new pytree.

    See also :func:`tree_map`, :func:`tree_map_`, and :func:`tree_map_with_accessor_`.

    >>> tree_map_with_accessor(lambda a, x: f'{a.codify("tree")} = {x!r}', {'x': 7, 'y': (42, 64)})
    {'x': "tree['x'] = 7", 'y': ("tree['y'][0] = 42", "tree['y'][1] = 64")}
    >>> tree_map_with_accessor(lambda a, x: x + len(a), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> tree_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: a,
    ...     {'x': 7, 'y': (42, 64), 'z': {1.5: None}},
    ... )
    {
        'x': PyTreeAccessor(*['x'], (MappingEntry(key='x', type=<class 'dict'>),)),
        'y': (
            PyTreeAccessor(*['y'][0], (MappingEntry(key='y', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['y'][1], (MappingEntry(key='y', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>)))
        ),
        'z': {1.5: None}
    }
    >>> tree_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: a,
    ...     {'x': 7, 'y': (42, 64), 'z': {1.5: None}},
    ...     none_is_leaf=True,
    ... )
    {
        'x': PyTreeAccessor(*['x'], (MappingEntry(key='x', type=<class 'dict'>),)),
        'y': (
            PyTreeAccessor(*['y'][0], (MappingEntry(key='y', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
            PyTreeAccessor(*['y'][1], (MappingEntry(key='y', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>)))
        ),
        'z': {
            1.5: PyTreeAccessor(*['z'][1.5], (MappingEntry(key='z', type=<class 'dict'>), MappingEntry(key=1.5, type=<class 'dict'>)))
        }
    }

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra accessors.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(a, x, *xs)`` where ``(a, x)`` are the accessor and value at the corresponding leaf in
        ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """  # pylint: disable=line-too-long
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, treespec.accessors(), *flat_args))


def tree_map_with_accessor_(
    func: Callable[..., Any],
    tree: PyTree[T],
    *rests: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:
    """Like :func:`tree_map_with_accessor`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`, :func:`tree_map_`, and :func:`tree_map_with_accessor`.

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra accessors.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(a, x, *xs)`` (not the return value) where ``(a, x)`` are the accessor and value at
        the corresponding leaf in ``tree`` and ``xs`` is the tuple of values at values at
        corresponding nodes in ``rests``.
    """
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, treespec.accessors(), *flat_args), maxlen=0)  # consume and exhaust the iterable
    return tree


def tree_replace_nones(sentinel: Any, tree: PyTree[T] | None, namespace: str = '') -> PyTree[T]:
    """Replace :data:`None` in ``tree`` with ``sentinel``.

    See also :func:`tree_flatten` and :func:`tree_map`.

    >>> tree_replace_nones(0, {'a': 1, 'b': None, 'c': (2, None)})
    {'a': 1, 'b': 0, 'c': (2, 0)}
    >>> tree_replace_nones(0, None)
    0

    Args:
        sentinel (object): The value to replace :data:`None` with.
        tree (pytree): A pytree to be transformed.
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with :data:`None` replaced.
    """
    if tree is None:
        return sentinel
    return tree_map(
        lambda x: x if x is not None else sentinel,
        tree,
        none_is_leaf=True,
        namespace=namespace,
    )


def tree_transpose(
    outer_treespec: PyTreeSpec,
    inner_treespec: PyTreeSpec,
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
) -> PyTree[T]:  # PyTree[PyTree[T]]
    """Transform a tree having tree structure (outer, inner) into one having structure (inner, outer).

    See also :func:`tree_flatten`, :func:`tree_structure`, and :func:`tree_transpose_map`.

    >>> outer_treespec = tree_structure({'a': 1, 'b': 2, 'c': (3, 4)})
    >>> outer_treespec
    PyTreeSpec({'a': *, 'b': *, 'c': (*, *)})
    >>> inner_treespec = tree_structure((1, 2))
    >>> inner_treespec
    PyTreeSpec((*, *))
    >>> tree = {'a': (1, 2), 'b': (3, 4), 'c': ((5, 6), (7, 8))}
    >>> tree_transpose(outer_treespec, inner_treespec, tree)
    ({'a': 1, 'b': 3, 'c': (5, 7)}, {'a': 2, 'b': 4, 'c': (6, 8)})

    For performance reasons, this function is only checks for the number of leaves in the input
    pytree, not the structure. The result is only enumerated up to the original order of leaves in
    ``tree``, then transpose depends on the number of leaves in structure (inner, outer). The caller
    is responsible for ensuring that the input pytree has a prefix structure of ``outer_treespec``
    followed by a prefix structure of ``inner_treespec``. Otherwise, the result may be incorrect.

    >>> tree_transpose(outer_treespec, inner_treespec, list(range(1, 9)))
    ({'a': 1, 'b': 3, 'c': (5, 7)}, {'a': 2, 'b': 4, 'c': (6, 8)})

    Args:
        outer_treespec (PyTreeSpec): A treespec object representing the outer structure of the pytree.
        inner_treespec (PyTreeSpec): A treespec object representing the inner structure of the pytree.
        tree (pytree): A pytree to be transposed.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.

    Returns:
        A new pytree with the same structure as ``inner_treespec`` but with the value at each leaf
        has the same structure as ``outer_treespec``.
    """
    if outer_treespec.none_is_leaf != inner_treespec.none_is_leaf:
        raise ValueError('Tree structures must have the same none_is_leaf value.')
    outer_size = outer_treespec.num_leaves
    inner_size = inner_treespec.num_leaves
    if outer_size == 0 or inner_size == 0:
        raise ValueError('Tree structures must have at least one leaf.')
    if (
        outer_treespec.namespace
        and inner_treespec.namespace
        and outer_treespec.namespace != inner_treespec.namespace
    ):
        raise ValueError(
            f'Tree structures must have the same namespace, '
            f'got {outer_treespec.namespace!r} vs. {inner_treespec.namespace!r}.',
        )

    leaves, treespec = tree_flatten(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=outer_treespec.none_is_leaf,
        namespace=outer_treespec.namespace or inner_treespec.namespace,
    )
    if treespec.num_leaves != outer_size * inner_size:
        expected_treespec = outer_treespec.compose(inner_treespec)
        raise TypeError(f'Tree structure mismatch; expected: {expected_treespec}, got: {treespec}.')

    grouped = [
        leaves[offset : offset + inner_size]
        for offset in range(0, outer_size * inner_size, inner_size)
    ]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)  # type: ignore[arg-type]


def tree_transpose_map(
    func: Callable[..., PyTree[U]],
    tree: PyTree[T],
    *rests: PyTree[S],
    inner_treespec: PyTreeSpec | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:  # PyTree[PyTree[U]]
    """Map a multi-input function over pytree args to produce a new pytree with transposed structure.

    See also :func:`tree_map`, :func:`tree_map_with_path`, and :func:`tree_transpose`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> tree_transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': 2 * x},
    ...     tree,
    ... )
    {
        'identity': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
        'double': {'b': (4, [6, 8]), 'a': 2, 'c': (10, 12)}
    }
    >>> tree_transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': (x, x)},
    ...     tree,
    ... )
    {
        'identity': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
        'double': (
            {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
            {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
        )
    }
    >>> tree_transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': (x, x)},
    ...     tree,
    ...     inner_treespec=tree_structure({'identity': 0, 'double': 0}),
    ... )
    {
        'identity': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
        'double': {'b': ((2, 2), [(3, 3), (4, 4)]), 'a': (1, 1), 'c': ((5, 5), (6, 6))}
    }

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        inner_treespec (PyTreeSpec, optional): The treespec object representing the inner structure
            of the result pytree. If not specified, the inner structure is inferred from the result
            of the function ``func`` on the first leaf. (default: :data:`None`)
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new nested pytree with the same structure as ``inner_treespec`` but with the value at each
        leaf has the same structure as ``tree``. The subtree at each leaf is given by the result of
        function ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and
        ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, outer_treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    if outer_treespec.num_leaves == 0:
        raise ValueError(f'The outer structure must have at least one leaf. Got: {outer_treespec}.')
    flat_args = [leaves] + [outer_treespec.flatten_up_to(r) for r in rests]
    outputs = list(map(func, *flat_args))

    if inner_treespec is None:
        inner_treespec = tree_structure(
            outputs[0],
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    if inner_treespec.num_leaves == 0:
        raise ValueError(f'The inner structure must have at least one leaf. Got: {inner_treespec}.')

    grouped = [inner_treespec.flatten_up_to(o) for o in outputs]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)  # type: ignore[arg-type]


def tree_transpose_map_with_path(
    func: Callable[..., PyTree[U]],
    tree: PyTree[T],
    *rests: PyTree[S],
    inner_treespec: PyTreeSpec | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:  # PyTree[PyTree[U]]
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree with transposed structure.

    See also :func:`tree_map_with_path`, :func:`tree_transpose_map`, and :func:`tree_transpose`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> tree_transpose_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x: {'depth': len(p), 'value': x},
    ...     tree,
    ... )
    {
        'depth': {'b': (2, [3, 3]), 'a': 1, 'c': (2, 2)},
        'value': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    }
    >>> tree_transpose_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x: {'path': p, 'value': x},
    ...     tree,
    ...     inner_treespec=tree_structure({'path': 0, 'value': 0}),
    ... )
    {
        'path': {
            'b': (('b', 0), [('b', 1, 0), ('b', 1, 1)]),
            'a': ('a',),
            'c': (('c', 0), ('c', 1))
        },
        'value': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    }

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        inner_treespec (PyTreeSpec, optional): The treespec object representing the inner structure
            of the result pytree. If not specified, the inner structure is inferred from the result
            of the function ``func`` on the first leaf. (default: :data:`None`)
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new nested pytree with the same structure as ``inner_treespec`` but with the value at each
        leaf has the same structure as ``tree``. The subtree at each leaf is given by the result of
        function ``func(p, x, *xs)`` where ``(p, x)`` are the path and value at the corresponding
        leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """  # pylint: disable=line-too-long
    paths, leaves, outer_treespec = _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)
    if outer_treespec.num_leaves == 0:
        raise ValueError(f'The outer structure must have at least one leaf. Got: {outer_treespec}.')
    flat_args = [leaves] + [outer_treespec.flatten_up_to(r) for r in rests]
    outputs = list(map(func, paths, *flat_args))

    if inner_treespec is None:
        inner_treespec = tree_structure(
            outputs[0],
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    if inner_treespec.num_leaves == 0:
        raise ValueError(f'The inner structure must have at least one leaf. Got: {inner_treespec}.')

    grouped = [inner_treespec.flatten_up_to(o) for o in outputs]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)  # type: ignore[arg-type]


def tree_transpose_map_with_accessor(
    func: Callable[..., PyTree[U]],
    tree: PyTree[T],
    *rests: PyTree[S],
    inner_treespec: PyTreeSpec | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:  # PyTree[PyTree[U]]
    """Map a multi-input function over pytree args as well as the tree accessors to produce a new pytree with transposed structure.

    See also :func:`tree_map_with_accessor`, :func:`tree_transpose_map`, and :func:`tree_transpose`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> tree_transpose_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: {'depth': len(a), 'code': a.codify('tree'), 'value': x},
    ...     tree,
    ... )
    {
        'depth': {
            'b': (2, [3, 3]),
            'a': 1,
            'c': (2, 2)
        },
        'code': {
            'b': ("tree['b'][0]", ["tree['b'][1][0]", "tree['b'][1][1]"]),
            'a': "tree['a']",
            'c': ("tree['c'][0]", "tree['c'][1]")
        },
        'value': {
            'b': (2, [3, 4]),
            'a': 1,
            'c': (5, 6)
        }
    }
    >>> tree_transpose_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: {'path': a.path, 'accessor': a, 'value': x},
    ...     tree,
    ...     inner_treespec=tree_structure({'path': 0, 'accessor': 0, 'value': 0}),
    ... )
    {
        'path': {
            'b': (('b', 0), [('b', 1, 0), ('b', 1, 1)]),
            'a': ('a',),
            'c': (('c', 0), ('c', 1))
        },
        'accessor': {
            'b': (
                PyTreeAccessor(*['b'][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
                [
                    PyTreeAccessor(*['b'][1][0], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=0, type=<class 'list'>))),
                    PyTreeAccessor(*['b'][1][1], (MappingEntry(key='b', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>), SequenceEntry(index=1, type=<class 'list'>)))
                ]
            ),
            'a': PyTreeAccessor(*['a'], (MappingEntry(key='a', type=<class 'dict'>),)),
            'c': (
                PyTreeAccessor(*['c'][0], (MappingEntry(key='c', type=<class 'dict'>), SequenceEntry(index=0, type=<class 'tuple'>))),
                PyTreeAccessor(*['c'][1], (MappingEntry(key='c', type=<class 'dict'>), SequenceEntry(index=1, type=<class 'tuple'>)))
            )
        },
        'value': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    }

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra accessors.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        inner_treespec (PyTreeSpec, optional): The treespec object representing the inner structure
            of the result pytree. If not specified, the inner structure is inferred from the result
            of the function ``func`` on the first leaf. (default: :data:`None`)
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new nested pytree with the same structure as ``inner_treespec`` but with the value at each
        leaf has the same structure as ``tree``. The subtree at each leaf is given by the result of
        function ``func(a, x, *xs)`` where ``(a, x)`` are the accessor and value at the corresponding
        leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """  # pylint: disable=line-too-long
    leaves, outer_treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    if outer_treespec.num_leaves == 0:
        raise ValueError(f'The outer structure must have at least one leaf. Got: {outer_treespec}.')
    flat_args = [leaves] + [outer_treespec.flatten_up_to(r) for r in rests]
    outputs = list(map(func, outer_treespec.accessors(), *flat_args))

    if inner_treespec is None:
        inner_treespec = tree_structure(
            outputs[0],
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    if inner_treespec.num_leaves == 0:
        raise ValueError(f'The inner structure must have at least one leaf. Got: {inner_treespec}.')

    grouped = [inner_treespec.flatten_up_to(o) for o in outputs]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)  # type: ignore[arg-type]


def tree_broadcast_prefix(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[T]:  # PyTree[PyTree[T]]
    """Return a pytree of same structure of ``full_tree`` with broadcasted subtrees in ``prefix_tree``.

    See also :func:`broadcast_prefix`, :func:`tree_broadcast_common`, and :func:`treespec_is_prefix`.

    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be
    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.

    This function returns a pytree with the same size as ``full_tree``. The leaves are replicated
    from ``prefix_tree``. The number of replicas is determined by the corresponding subtree in
    ``full_tree``.

    >>> tree_broadcast_prefix(1, [2, 3, 4])
    [1, 1, 1]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, 6])
    [1, 2, 3]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4; list: [4, 5, 6, 7].
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, (6, 7)])
    [1, 2, (3, 3)]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}])
    [1, 2, {'a': 3, 'b': 3, 'c': (None, 3)}]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}], none_is_leaf=True)
    [1, 2, {'a': 3, 'b': 3, 'c': (3, 3)}]

    Args:
        prefix_tree (pytree): A pytree with the prefix structure of ``full_tree``.
        full_tree (pytree): A pytree with the suffix structure of ``prefix_tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A pytree of same structure of ``full_tree`` with broadcasted subtrees in ``prefix_tree``.
    """

    def broadcast_leaves(x: T, subtree: PyTree[S]) -> PyTree[T]:
        subtreespec = tree_structure(
            subtree,
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        return subtreespec.unflatten(itertools.repeat(x, subtreespec.num_leaves))

    # If prefix_tree is not a tree prefix of full_tree, this code can raise a ValueError;
    # use prefix_errors to find disagreements and raise more precise error messages.
    # errors = prefix_errors(
    #     prefix_tree,
    #     full_tree,
    #     is_leaf=is_leaf,
    #     none_is_leaf=none_is_leaf,
    #     namespace=namespace,
    # )
    return tree_map(
        broadcast_leaves,  # type: ignore[arg-type]
        prefix_tree,
        full_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def broadcast_prefix(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[T]:
    """Return a list of broadcasted leaves in ``prefix_tree`` to match the number of leaves in ``full_tree``.

    See also :func:`tree_broadcast_prefix`, :func:`broadcast_common`, and :func:`treespec_is_prefix`.

    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be
    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.

    This function returns a list of leaves with the same size as ``full_tree``. The leaves are
    replicated from ``prefix_tree``. The number of replicas is determined by the corresponding
    subtree in ``full_tree``.

    >>> broadcast_prefix(1, [2, 3, 4])
    [1, 1, 1]
    >>> broadcast_prefix([1, 2, 3], [4, 5, 6])
    [1, 2, 3]
    >>> broadcast_prefix([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4; list: [4, 5, 6, 7].
    >>> broadcast_prefix([1, 2, 3], [4, 5, (6, 7)])
    [1, 2, 3, 3]
    >>> broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}])
    [1, 2, 3, 3, 3]
    >>> broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}], none_is_leaf=True)
    [1, 2, 3, 3, 3, 3]

    Args:
        prefix_tree (pytree): A pytree with the prefix structure of ``full_tree``.
        full_tree (pytree): A pytree with the suffix structure of ``prefix_tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A list of leaves in ``prefix_tree`` broadcasted to match the number of leaves in ``full_tree``.
    """
    result: list[T] = []

    def add_leaves(x: T, subtree: PyTree[S]) -> None:
        subtreespec = tree_structure(
            subtree,
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        result.extend(itertools.repeat(x, subtreespec.num_leaves))

    # If prefix_tree is not a tree prefix of full_tree, this code can raise a ValueError;
    # use prefix_errors to find disagreements and raise more precise error messages.
    # errors = prefix_errors(
    #     prefix_tree,
    #     full_tree,
    #     is_leaf=is_leaf,
    #     none_is_leaf=none_is_leaf,
    #     namespace=namespace,
    # )
    tree_map_(
        add_leaves,
        prefix_tree,
        full_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    return result


def tree_broadcast_common(
    tree: PyTree[T],
    other_tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[PyTree[T], PyTree[T]]:
    """Return two pytrees of common suffix structure of ``tree`` and ``other_tree`` with broadcasted subtrees.

    See also :func:`broadcast_common`, :func:`tree_broadcast_prefix`, and :func:`treespec_is_prefix`.

    If a ``suffix_tree`` is a suffix of a ``tree``, this means the ``suffix_tree`` can be
    constructed by replacing the leaves of ``tree`` with appropriate **subtrees**.

    This function returns two pytrees with the same structure. The tree structure is the common
    suffix structure of ``tree`` and ``other_tree``. The leaves are replicated from ``tree`` and
    ``other_tree``. The number of replicas is determined by the corresponding subtree in the suffix
    structure.

    >>> tree_broadcast_common(1, [2, 3, 4])
    ([1, 1, 1], [2, 3, 4])
    >>> tree_broadcast_common([1, 2, 3], [4, 5, 6])
    ([1, 2, 3], [4, 5, 6])
    >>> tree_broadcast_common([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4.
    >>> tree_broadcast_common([1, (2, 3), 4], [5, 6, (7, 8)])
    ([1, (2, 3), (4, 4)], [5, (6, 6), (7, 8)])
    >>> tree_broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}])
    ([1, {'a': (2, 3)}, {'a': 4, 'b': 4, 'c': (None, 4)}],
     [5, {'a': (6, 6)}, {'a': 7, 'b': 8, 'c': (None, 9)}])
    >>> tree_broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}], none_is_leaf=True)
    ([1, {'a': (2, 3)}, {'a': 4, 'b': 4, 'c': (4, 4)}],
     [5, {'a': (6, 6)}, {'a': 7, 'b': 8, 'c': (None, 9)}])
    >>> tree_broadcast_common([1, None], [None, 2])
    ([None, None], [None, None])
    >>> tree_broadcast_common([1, None], [None, 2], none_is_leaf=True)
    ([1, None], [None, 2])

    Args:
        tree (pytree): A pytree has a common suffix structure of ``other_tree``.
        other_tree (pytree): A pytree has a common suffix structure of ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        Two pytrees of common suffix structure of ``tree`` and ``other_tree`` with broadcasted subtrees.
    """
    leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    other_leaves, other_treespec = _C.flatten(other_tree, is_leaf, none_is_leaf, namespace)
    common_suffix_treespec = treespec.broadcast_to_common_suffix(other_treespec)

    sentinel: T = object()  # type: ignore[assignment]
    common_suffix_tree: PyTree[T] = common_suffix_treespec.unflatten(
        itertools.repeat(sentinel, common_suffix_treespec.num_leaves),
    )

    def broadcast_leaves(x: T, subtree: PyTree[T]) -> PyTree[T]:
        subtreespec = tree_structure(
            subtree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        return subtreespec.unflatten(itertools.repeat(x, subtreespec.num_leaves))

    broadcasted_tree: PyTree[T] = treespec.unflatten(
        map(
            broadcast_leaves,  # type: ignore[arg-type]
            leaves,
            treespec.flatten_up_to(common_suffix_tree),
        ),
    )
    other_broadcasted_tree: PyTree[T] = other_treespec.unflatten(
        map(
            broadcast_leaves,  # type: ignore[arg-type]
            other_leaves,
            other_treespec.flatten_up_to(common_suffix_tree),
        ),
    )
    return broadcasted_tree, other_broadcasted_tree


def broadcast_common(
    tree: PyTree[T],
    other_tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[T], list[T]]:
    """Return two lists of leaves in ``tree`` and ``other_tree`` broadcasted to match the number of leaves in the common suffix structure.

    See also :func:`tree_broadcast_common`, :func:`broadcast_prefix`, and :func:`treespec_is_prefix`.

    If a ``suffix_tree`` is a suffix of a ``tree``, this means the ``suffix_tree`` can be
    constructed by replacing the leaves of ``tree`` with appropriate **subtrees**.

    This function returns two pytrees with the same structure. The tree structure is the common
    suffix structure of ``tree`` and ``other_tree``. The leaves are replicated from ``tree`` and
    ``other_tree``. The number of replicas is determined by the corresponding subtree in the suffix
    structure.

    >>> broadcast_common(1, [2, 3, 4])
    ([1, 1, 1], [2, 3, 4])
    >>> broadcast_common([1, 2, 3], [4, 5, 6])
    ([1, 2, 3], [4, 5, 6])
    >>> broadcast_common([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4.
    >>> broadcast_common([1, (2, 3), 4], [5, 6, (7, 8)])
    ([1, 2, 3, 4, 4], [5, 6, 6, 7, 8])
    >>> broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}])
    ([1, 2, 3, 4, 4, 4], [5, 6, 6, 7, 8, 9])
    >>> broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}], none_is_leaf=True)
    ([1, 2, 3, 4, 4, 4, 4], [5, 6, 6, 7, 8, None, 9])
    >>> broadcast_common([1, None], [None, 2])
    ([], [])
    >>> broadcast_common([1, None], [None, 2], none_is_leaf=True)
    ([1, None], [None, 2])

    Args:
        tree (pytree): A pytree has a common suffix structure of ``other_tree``.
        other_tree (pytree): A pytree has a common suffix structure of ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        Two lists of leaves in ``tree`` and ``other_tree`` broadcasted to match the number of leaves
        in the common suffix structure.
    """  # pylint: disable=line-too-long
    broadcasted_tree, other_broadcasted_tree = tree_broadcast_common(
        tree,
        other_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )

    broadcasted_leaves: list[T] = []
    other_broadcasted_leaves: list[T] = []

    def add_leaves(x: T, y: T) -> None:
        broadcasted_leaves.append(x)
        other_broadcasted_leaves.append(y)

    tree_map_(
        add_leaves,
        broadcasted_tree,
        other_broadcasted_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    return broadcasted_leaves, other_broadcasted_leaves


def _tree_broadcast_common(
    tree: PyTree[T],
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[PyTree[T], ...]:
    if not rests:
        return (tree,)
    if len(rests) == 1:
        return tree_broadcast_common(
            tree,
            rests[0],
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )

    broadcasted_tree = tree
    broadcasted_rests = list(rests)
    for _ in range(2):
        for i, rest in enumerate(rests):
            broadcasted_tree, broadcasted_rests[i] = tree_broadcast_common(
                broadcasted_tree,
                rest,
                is_leaf=is_leaf,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )

    return (broadcasted_tree, *broadcasted_rests)


# pylint: disable-next=too-many-locals
def tree_broadcast_map(
    func: Callable[..., U],
    tree: PyTree[T],
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_broadcast_map_with_path`, :func:`tree_map`, :func:`tree_map_`,
    and :func:`tree_map_with_path`.

    If only one input is provided, this function is the same as :func:`tree_map`:

    >>> tree_broadcast_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_broadcast_map(lambda x: x + 1, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (43, 65), 'z': None}
    >>> tree_broadcast_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': None}
    >>> tree_broadcast_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None}, none_is_leaf=True)
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, all input trees will be broadcasted to the common suffix structure
    of all inputs:

    >>> tree_broadcast_map(lambda x, y: x * y, [5, 6, (3, 4)], [{'a': 7, 'b': 9}, [1, 2], 8])
    [{'a': 35, 'b': 45}, [6, 12], (24, 32)]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, they should have a common suffix structure with
            each other and with ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the structure as the common suffix structure of ``tree`` and ``rests`` but
        with the value at each leaf given by ``func(x, *xs)`` where ``x`` is the value at the
        corresponding leaf (may be broadcasted) in ``tree`` and ``xs`` is the tuple of values at
        corresponding leaves (may be broadcasted) in ``rests``.
    """
    return tree_map(
        func,
        *_tree_broadcast_common(
            tree,
            *rests,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


# pylint: disable-next=too-many-locals
def tree_broadcast_map_with_path(
    func: Callable[..., U],
    tree: PyTree[T],
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree.

    See also :func:`tree_broadcast_map`, :func:`tree_map`, :func:`tree_map_`,
    and :func:`tree_map_with_path`.

    If only one input is provided, this function is the same as :func:`tree_map`:

    >>> tree_broadcast_map_with_path(lambda p, x: (len(p), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> tree_broadcast_map_with_path(lambda p, x: x + len(p), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> tree_broadcast_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}})
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: None}}
    >>> tree_broadcast_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}}, none_is_leaf=True)
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: ('z', 1.5)}}

    If multiple inputs are given, all input trees will be broadcasted to the common suffix structure
    of all inputs:

    >>> tree_broadcast_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x, y: (p, x * y),
    ...     [5, 6, (3, 4)],
    ...     [{'a': 7, 'b': 9}, [1, 2], 8],
    ... )
    [
        {'a': ((0, 'a'), 35), 'b': ((0, 'b'), 45)},
        [((1, 0), 6), ((1, 1), 12)],
        (((2, 0), 24), ((2, 1), 32))
    ]

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, they should have a common suffix structure with
            each other and with ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the structure as the common suffix structure of ``tree`` and ``rests`` but
        with the value at each leaf given by ``func(p, x, *xs)`` where ``(p, x)`` are the path and
        value at the corresponding leaf (may be broadcasted) in and ``xs`` is the tuple of values at
        corresponding leaves (may be broadcasted) in ``rests``.
    """
    return tree_map_with_path(
        func,
        *_tree_broadcast_common(
            tree,
            *rests,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def tree_broadcast_map_with_accessor(
    func: Callable[..., U],
    tree: PyTree[T],
    *rests: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree accessors to produce a new pytree.

    See also :func:`tree_broadcast_map`, :func:`tree_map`, :func:`tree_map_`,
    and :func:`tree_map_with_accessor`.

    If only one input is provided, this function is the same as :func:`tree_map`:

    >>> tree_broadcast_map_with_accessor(lambda a, x: (len(a), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> tree_broadcast_map_with_accessor(lambda a, x: x + len(a), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> tree_broadcast_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: a.codify('tree'),
    ...     {'x': 7, 'y': (42, 64), 'z': {1.5: None}},
    ... )
    {
        'x': "tree['x']",
        'y': ("tree['y'][0]", "tree['y'][1]"),
        'z': {1.5: None}
    }
    >>> tree_broadcast_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x: a.codify('tree'),
    ...     {'x': 7, 'y': (42, 64), 'z': {1.5: None}},
    ...     none_is_leaf=True,
    ... )
    {
        'x': "tree['x']",
        'y': ("tree['y'][0]", "tree['y'][1]"),
        'z': {1.5: "tree['z'][1.5]"}
    }

    If multiple inputs are given, all input trees will be broadcasted to the common suffix structure
    of all inputs:

    >>> tree_broadcast_map_with_accessor(  # doctest: +IGNORE_WHITESPACE
    ...     lambda a, x, y: f'{a.codify("tree")} = {x * y}',
    ...     [5, 6, (3, 4)],
    ...     [{'a': 7, 'b': 9}, [1, 2], 8],
    ... )
    [
        {'a': "tree[0]['a'] = 35", 'b': "tree[0]['b'] = 45"},
        ['tree[1][0] = 6', 'tree[1][1] = 12'],
        ('tree[2][0] = 24', 'tree[2][1] = 32')
    ]

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra accessors.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, they should have a common suffix structure with
            each other and with ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the structure as the common suffix structure of ``tree`` and ``rests`` but
        with the value at each leaf given by ``func(a, x, *xs)`` where ``(a, x)`` are the accessor
        and value at the corresponding leaf (may be broadcasted) in and ``xs`` is the tuple of
        values at corresponding leaves (may be broadcasted) in ``rests``.
    """
    return tree_map_with_accessor(
        func,
        *_tree_broadcast_common(
            tree,
            *rests,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


# pylint: disable-next=missing-class-docstring,too-few-public-methods
class MissingSentinel:  # pragma: no cover
    __slots__: ClassVar[tuple[()]] = ()

    def __repr__(self) -> str:
        return '<MISSING>'


__MISSING: T = MissingSentinel()  # type: ignore[valid-type]
del MissingSentinel


@overload
def tree_reduce(
    func: Callable[[T, T], T],
    tree: PyTree[T],
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


@overload
def tree_reduce(
    func: Callable[[T, S], T],
    tree: PyTree[S],
    initial: T = __MISSING,
    *,
    is_leaf: Callable[[S], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


def tree_reduce(
    func: Callable[[T, S], T],
    tree: PyTree[S],
    initial: T = __MISSING,
    *,
    is_leaf: Callable[[S], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T:
    """Traversal through a pytree and reduce the leaves in left-to-right depth-first order.

    See also :func:`tree_leaves` and :func:`tree_sum`.

    >>> tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, 3)})
    6
    >>> tree_reduce(lambda x, y: x + y, {'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node with arity 0 by default
    6
    >>> tree_reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3})
    3
    >>> tree_reduce(lambda x, y: x and y, {'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    None

    Args:
        func (callable): A function that takes two arguments and returns a value of the same type.
        tree (pytree): A pytree to be traversed.
        initial (object, optional): An initial value to be used for the reduction. If not provided,
            the first leaf value is used as the initial value.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The result of reducing the leaves of the pytree using ``func``.
    """  # pylint: disable=line-too-long
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if initial is __MISSING:
        return functools.reduce(func, leaves)  # type: ignore[arg-type,return-value]
    return functools.reduce(func, leaves, initial)


def tree_sum(
    tree: PyTree[T],
    start: T = 0,  # type: ignore[assignment]
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T:
    """Sum ``start`` and leaf values in ``tree`` in left-to-right depth-first order and return the total.

    See also :func:`tree_leaves` and :func:`tree_reduce`.

    >>> tree_sum({'x': 1, 'y': (2, 3)})
    6
    >>> tree_sum({'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node with arity 0 by default
    6
    >>> tree_sum({'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
    >>> tree_sum({'x': 'a', 'y': ('b', None), 'z': 'c'}, start='')
    'abc'
    >>> tree_sum({'x': [1], 'y': ([2], [None]), 'z': [3]}, start=[], is_leaf=lambda x: isinstance(x, list))
    [1, 2, None, 3]

    Args:
        tree (pytree): A pytree to be traversed.
        start (object, optional): An initial value to be used for the sum. (default: :data:`0`)
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The total sum of ``start`` and leaf values in ``tree``.
    """
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    # sum() rejects string values for `start` parameter
    if isinstance(start, str):
        return ''.join([start, *leaves])  # type: ignore[list-item,return-value]
    if isinstance(start, (bytes, bytearray)):
        return b''.join([start, *leaves])  # type: ignore[list-item,return-value]
    return sum(leaves, start)  # type: ignore[call-overload]


@overload
def tree_max(
    tree: PyTree[T],
    *,
    is_leaf: Callable[[T], bool] | None = None,
    key: Callable[[T], Any] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


@overload
def tree_max(
    tree: PyTree[T],
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


def tree_max(
    tree: PyTree[T],
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T:
    """Return the maximum leaf value in ``tree``.

    See also :func:`tree_leaves` and :func:`tree_min`.

    >>> tree_max({})
    Traceback (most recent call last):
        ...
    ValueError: max() iterable argument is empty
    >>> tree_max({}, default=0)
    0
    >>> tree_max({'x': 0, 'y': (2, 1)})
    2
    >>> tree_max({'x': 0, 'y': (2, 1)}, key=lambda x: -x)
    0
    >>> tree_max({'a': None})  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: max() iterable argument is empty
    >>> tree_max({'a': None}, default=0)  # `None` is a non-leaf node with arity 0 by default
    0
    >>> tree_max({'a': None}, none_is_leaf=True)
    None
    >>> tree_max(None)  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: max() iterable argument is empty
    >>> tree_max(None, default=0)
    0
    >>> tree_max(None, none_is_leaf=True)
    None

    Args:
        tree (pytree): A pytree to be traversed.
        default (object, optional): The default value to return if ``tree`` is empty. If the ``tree``
            is empty and ``default`` is not specified, raise a :exc:`ValueError`.
        key (callable or None, optional): An one argument ordering function like that used for
            :meth:`list.sort`.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The maximum leaf value in ``tree``.
    """
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if default is __MISSING:
        if key is None:  # special handling for Python 3.7
            return max(leaves)  # type: ignore[type-var]
        return max(leaves, key=key)
    if key is None:  # special handling for Python 3.7
        return max(leaves, default=default)  # type: ignore[type-var]
    return max(leaves, default=default, key=key)


@overload
def tree_min(
    tree: PyTree[T],
    *,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


@overload
def tree_min(
    tree: PyTree[T],
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T: ...


def tree_min(
    tree: PyTree[T],
    *,
    default: T = __MISSING,
    key: Callable[[T], Any] | None = None,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> T:
    """Return the minimum leaf value in ``tree``.

    See also :func:`tree_leaves` and :func:`tree_max`.

    >>> tree_min({})
    Traceback (most recent call last):
        ...
    ValueError: min() iterable argument is empty
    >>> tree_min({}, default=0)
    0
    >>> tree_min({'x': 0, 'y': (2, 1)})
    0
    >>> tree_min({'x': 0, 'y': (2, 1)}, key=lambda x: -x)
    2
    >>> tree_min({'a': None})  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: min() iterable argument is empty
    >>> tree_min({'a': None}, default=0)  # `None` is a non-leaf node with arity 0 by default
    0
    >>> tree_min({'a': None}, none_is_leaf=True)
    None
    >>> tree_min(None)  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: min() iterable argument is empty
    >>> tree_min(None, default=0)
    0
    >>> tree_min(None, none_is_leaf=True)
    None

    Args:
        tree (pytree): A pytree to be traversed.
        default (object, optional): The default value to return if ``tree`` is empty. If the ``tree``
            is empty and ``default`` is not specified, raise a :exc:`ValueError`.
        key (callable or None, optional): An one argument ordering function like that used for
            :meth:`list.sort`.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The minimum leaf value in ``tree``.
    """
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if default is __MISSING:
        if key is None:  # special handling for Python 3.7
            return min(leaves)  # type: ignore[type-var]
        return min(leaves, key=key)
    if key is None:  # special handling for Python 3.7
        return min(leaves, default=default)  # type: ignore[type-var]
    return min(leaves, default=default, key=key)


def tree_all(
    tree: PyTree[T],
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether all leaves in ``tree`` are true (or if ``tree`` is empty).

    See also :func:`tree_leaves` and :func:`tree_any`.

    >>> tree_all({})
    True
    >>> tree_all({'x': 1, 'y': (2, 3)})
    True
    >>> tree_all({'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node by default
    True
    >>> tree_all({'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    False
    >>> tree_all(None)  # `None` is a non-leaf node by default
    True
    >>> tree_all(None, none_is_leaf=True)
    False

    Args:
        tree (pytree): A pytree to be traversed.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        :data:`True` if all leaves in ``tree`` are true, or if ``tree`` is empty.
        Otherwise, :data:`False`.
    """
    return all(
        tree_iter(
            tree,  # type: ignore[arg-type]
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
    )


def tree_any(
    tree: PyTree[T],
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether all leaves in ``tree`` are true (or :data:`False` if ``tree`` is empty).

    See also :func:`tree_leaves` and :func:`tree_all`.

    >>> tree_any({})
    False
    >>> tree_any({'x': 0, 'y': (2, 0)})
    True
    >>> tree_any({'a': None})  # `None` is a non-leaf node with arity 0 by default
    False
    >>> tree_any({'a': None}, none_is_leaf=True)  # `None` is evaluated as false
    False
    >>> tree_any(None)  # `None` is a non-leaf node with arity 0 by default
    False
    >>> tree_any(None, none_is_leaf=True)  # `None` is evaluated as false
    False

    Args:
        tree (pytree): A pytree to be traversed.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        :data:`True` if any leaves in ``tree`` are true, otherwise, :data:`False`. If ``tree`` is
        empty, return :data:`False`.
    """
    return any(
        tree_iter(
            tree,  # type: ignore[arg-type]
            is_leaf=is_leaf,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
    )


def tree_flatten_one_level(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[
    list[PyTree[T]],
    MetaData,
    tuple[Any, ...],
    Callable[[MetaData, list[PyTree[T]]], PyTree[T]],
]:
    """Flatten the pytree one level, returning a 4-tuple of children, metadata, path entries, and an unflatten function.

    See also :func:`tree_flatten`, :func:`tree_flatten_with_path`.

    >>> children, metadata, entries, unflatten_func = tree_flatten_one_level({'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5})
    >>> children, metadata, entries
    ([1, (2, [3, 4]), None, 5], ['a', 'b', 'c', 'd'], ('a', 'b', 'c', 'd'))
    >>> unflatten_func(metadata, children)
    {'a': 1, 'b': (2, [3, 4]), 'c': None, 'd': 5}
    >>> children, metadata, entries, unflatten_func = tree_flatten_one_level([{'a': 1, 'b': (2, 3)}, (4, 5)])
    >>> children, metadata, entries
    ([{'a': 1, 'b': (2, 3)}, (4, 5)], None, (0, 1))
    >>> unflatten_func(metadata, children)
    [{'a': 1, 'b': (2, 3)}, (4, 5)]

    Args:
        tree (pytree): A pytree to be traversed.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A 4-tuple ``(children, metadata, entries, unflatten_func)``. The first element is a list of
        one-level children of the pytree node. The second element is the metadata used to
        reconstruct the pytree node. The third element is a tuple of path entries to the children.
        The fourth element is a function that can be used to unflatten the metadata and
        children back to the pytree node.
    """  # pylint: disable=line-too-long
    node_type = type(tree)
    if (tree is None and none_is_leaf) or (is_leaf is not None and is_leaf(tree)):  # type: ignore[unreachable,arg-type]
        raise ValueError(f'Cannot flatten leaf-type: {node_type} (node: {tree!r}).')

    handler: PyTreeNodeRegistryEntry | None = register_pytree_node.get(node_type, namespace=namespace)  # type: ignore[attr-defined]
    if handler:
        flattened = tuple(handler.flatten_func(tree))  # type: ignore[arg-type]
        if len(flattened) == 2:
            flattened = (*flattened, None)
        elif len(flattened) != 3:
            raise RuntimeError(
                f'PyTree custom flatten function for type {node_type} should return a 2- or 3-tuple, '
                f'got {len(flattened)}.',
            )
        children, metadata, entries = flattened
        children = list(children)  # type: ignore[arg-type]
        entries = tuple(range(len(children)) if entries is None else entries)
        if len(children) != len(entries):
            raise RuntimeError(
                f'PyTree custom flatten function for type {node_type} returned inconsistent '
                f'number of children ({len(children)}) and number of entries ({len(entries)}).',
            )
        return children, metadata, entries, handler.unflatten_func  # type: ignore[return-value]

    raise ValueError(f'Cannot flatten leaf-type: {node_type} (node: {tree!r}).')


def treespec_paths(treespec: PyTreeSpec) -> list[tuple[Any, ...]]:
    """Return a list of paths to the leaves of a treespec.

    See also :func:`tree_flatten_with_path`, :func:`tree_paths`, and :meth:`PyTreeSpec.paths`.
    """
    return treespec.paths()


def treespec_accessors(treespec: PyTreeSpec) -> list[PyTreeAccessor]:
    """Return a list of accessors to the leaves of a treespec.

    See also :func:`tree_flatten_with_accessor`, :func:`tree_accessors` and
    :meth:`PyTreeSpec.accessors`.
    """
    return treespec.accessors()


def treespec_entries(treespec: PyTreeSpec) -> list[Any]:
    """Return a list of one-level entries of a treespec to its children.

    See also :func:`treespec_entry`, :func:`treespec_paths`, :func:`treespec_children`,
    and :meth:`PyTreeSpec.entries`.
    """
    return treespec.entries()


def treespec_entry(treespec: PyTreeSpec, index: int) -> Any:
    """Return the entry of a treespec at the given index.

    See also :func:`treespec_entries`, :func:`treespec_children`, and :meth:`PyTreeSpec.entry`.
    """
    return treespec.entry(index)


def treespec_children(treespec: PyTreeSpec) -> list[PyTreeSpec]:
    """Return a list of treespecs for the children of a treespec.

    See also :func:`treespec_child`, :func:`treespec_paths`, :func:`treespec_entries`,
    and :meth:`PyTreeSpec.children`.
    """
    return treespec.children()


def treespec_child(treespec: PyTreeSpec, index: int) -> PyTreeSpec:
    """Return the treespec of the child of a treespec at the given index.

    See also :func:`treespec_children`, :func:`treespec_entries`, and :meth:`PyTreeSpec.child`.
    """
    return treespec.child(index)


def treespec_is_leaf(treespec: PyTreeSpec, strict: bool = True) -> bool:
    """Return whether the treespec is a leaf that has no children.

    See also :func:`treespec_is_strict_leaf` and :meth:`PyTreeSpec.is_leaf`.

    This function is equivalent to ``treespec.is_leaf(strict=strict)``. If ``strict=False``, it will
    return :data:`True` if and only if the treespec represents a strict leaf. If ``strict=False``,
    it will return :data:`True` if the treespec represents a strict leaf or :data:`None` or an empty
    container (e.g., an empty tuple).

    >>> treespec_is_leaf(tree_structure(1))
    True
    >>> treespec_is_leaf(tree_structure((1, 2)))
    False
    >>> treespec_is_leaf(tree_structure(None))
    False
    >>> treespec_is_leaf(tree_structure(None), strict=False)
    True
    >>> treespec_is_leaf(tree_structure(None, none_is_leaf=False))
    False
    >>> treespec_is_leaf(tree_structure(None, none_is_leaf=True))
    True
    >>> treespec_is_leaf(tree_structure(()))
    False
    >>> treespec_is_leaf(tree_structure(()), strict=False)
    True
    >>> treespec_is_leaf(tree_structure([]))
    False
    >>> treespec_is_leaf(tree_structure([]), strict=False)
    True

    Args:
        treespec (PyTreeSpec): A treespec.
        strict (bool, optional): Whether not to treat :data:`None` or an empty
            container (e.g., an empty tuple) as a leaf. (default: :data:`True`)

    Returns:
        :data:`True` if the treespec represents a leaf that has no children, otherwise, :data:`False`.
    """
    if strict:
        return treespec.num_nodes == 1 and treespec.num_leaves == 1
    return treespec.num_nodes == 1


def treespec_is_strict_leaf(treespec: PyTreeSpec) -> bool:
    """Return whether the treespec is a strict leaf.

    See also :func:`treespec_is_leaf` and :meth:`PyTreeSpec.is_leaf`.

    This function respects the ``none_is_leaf`` setting in the treespec. It is equivalent to
    ``treespec.is_leaf(strict=True)``. It will return :data:`True` if and only if the treespec
    represents a strict leaf.

    >>> treespec_is_strict_leaf(tree_structure(1))
    True
    >>> treespec_is_strict_leaf(tree_structure((1, 2)))
    False
    >>> treespec_is_strict_leaf(tree_structure(None))
    False
    >>> treespec_is_strict_leaf(tree_structure(None, none_is_leaf=False))
    False
    >>> treespec_is_strict_leaf(tree_structure(None, none_is_leaf=True))
    True
    >>> treespec_is_strict_leaf(tree_structure(()))
    False
    >>> treespec_is_strict_leaf(tree_structure([]))
    False

    Args:
        treespec (PyTreeSpec): A treespec.

    Returns:
        :data:`True` if the treespec represents a strict leaf, otherwise, :data:`False`.
    """
    return treespec.num_nodes == 1 and treespec.num_leaves == 1


def treespec_is_prefix(
    treespec: PyTreeSpec,
    other_treespec: PyTreeSpec,
    strict: bool = False,
) -> bool:
    """Return whether ``treespec`` is a prefix of ``other_treespec``.

    See also :func:`treespec_is_prefix` and :meth:`PyTreeSpec.is_prefix`.
    """
    return treespec.is_prefix(other_treespec, strict=strict)


def treespec_is_suffix(
    treespec: PyTreeSpec,
    other_treespec: PyTreeSpec,
    strict: bool = False,
) -> bool:
    """Return whether ``treespec`` is a suffix of ``other_treespec``.

    See also :func:`treespec_is_suffix` :meth:`PyTreeSpec.is_suffix`.
    """
    return treespec.is_suffix(other_treespec, strict=strict)


def treespec_leaf(
    *,
    none_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec:
    """Make a treespec representing a leaf node.

    See also :func:`tree_structure`, :func:`treespec_none`, and :func:`treespec_tuple`.

    >>> treespec_leaf()
    PyTreeSpec(*)
    >>> treespec_leaf(none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec_leaf(none_is_leaf=False) == treespec_leaf(none_is_leaf=True)
    False
    >>> treespec_leaf() == tree_structure(1)
    True
    >>> treespec_leaf(none_is_leaf=True) == tree_structure(1, none_is_leaf=True)
    True
    >>> treespec_leaf(none_is_leaf=True) == tree_structure(None, none_is_leaf=True)
    True
    >>> treespec_leaf(none_is_leaf=True) == tree_structure(None, none_is_leaf=False)
    False
    >>> treespec_leaf(none_is_leaf=True) == treespec_none(none_is_leaf=True)
    True
    >>> treespec_leaf(none_is_leaf=True) == treespec_none(none_is_leaf=False)
    False
    >>> treespec_leaf(none_is_leaf=False) == treespec_none(none_is_leaf=True)
    False
    >>> treespec_leaf(none_is_leaf=False) == treespec_none(none_is_leaf=False)
    False

    Args:
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a leaf node.
    """
    return _C.make_leaf(
        none_is_leaf,
        namespace,  # unused
    )


def treespec_none(
    *,
    none_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec:
    """Make a treespec representing a :data:`None` node.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_tuple`.

    >>> treespec_none()
    PyTreeSpec(None)
    >>> treespec_none(none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec_none(none_is_leaf=False) == treespec_none(none_is_leaf=True)
    False
    >>> treespec_none() == tree_structure(None)
    True
    >>> treespec_none() == tree_structure(1)
    False
    >>> treespec_none(none_is_leaf=True) == tree_structure(1, none_is_leaf=True)
    True
    >>> treespec_none(none_is_leaf=True) == tree_structure(None, none_is_leaf=True)
    True
    >>> treespec_none(none_is_leaf=True) == tree_structure(None, none_is_leaf=False)
    False
    >>> treespec_none(none_is_leaf=True) == treespec_leaf(none_is_leaf=True)
    True
    >>> treespec_none(none_is_leaf=False) == treespec_leaf(none_is_leaf=True)
    False
    >>> treespec_none(none_is_leaf=True) == treespec_leaf(none_is_leaf=False)
    False
    >>> treespec_none(none_is_leaf=False) == treespec_leaf(none_is_leaf=False)
    False

    Args:
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a :data:`None` node.
    """
    return _C.make_none(
        none_is_leaf,
        namespace,  # unused
    )


def treespec_tuple(
    iterable: Iterable[PyTreeSpec] = (),
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a tuple treespec from a list of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_tuple([treespec_leaf(), treespec_leaf()])
    PyTreeSpec((*, *))
    >>> treespec_tuple([treespec_leaf(), treespec_leaf(), treespec_none()])
    PyTreeSpec((*, *, None))
    >>> treespec_tuple()
    PyTreeSpec(())
    >>> treespec_tuple([treespec_leaf(), treespec_tuple([treespec_leaf(), treespec_leaf()])])
    PyTreeSpec((*, (*, *)))
    >>> treespec_tuple([treespec_leaf(), tree_structure({'a': 1, 'b': 2})])
    PyTreeSpec((*, {'a': *, 'b': *}))
    >>> treespec_tuple([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a tuple node with the given children.
    """
    return _C.make_from_collection(
        tuple(iterable),  # type: ignore[arg-type]
        none_is_leaf,
        namespace,
    )


def treespec_list(
    iterable: Iterable[PyTreeSpec] = (),
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a list treespec from a list of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_list([treespec_leaf(), treespec_leaf()])
    PyTreeSpec([*, *])
    >>> treespec_list([treespec_leaf(), treespec_leaf(), treespec_none()])
    PyTreeSpec([*, *, None])
    >>> treespec_list()
    PyTreeSpec([])
    >>> treespec_list([treespec_leaf(), treespec_tuple([treespec_leaf(), treespec_leaf()])])
    PyTreeSpec([*, (*, *)])
    >>> treespec_list([treespec_leaf(), tree_structure({'a': 1, 'b': 2})])
    PyTreeSpec([*, {'a': *, 'b': *}])
    >>> treespec_list([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a list node with the given children.
    """
    return _C.make_from_collection(
        list(iterable),  # type: ignore[arg-type]
        none_is_leaf,
        namespace,
    )


def treespec_dict(
    mapping: Mapping[Any, PyTreeSpec] | Iterable[tuple[Any, PyTreeSpec]] = (),
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    """Make a dict treespec from a dict of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_dict({'a': treespec_leaf(), 'b': treespec_leaf()})
    PyTreeSpec({'a': *, 'b': *})
    >>> treespec_dict([('b', treespec_leaf()), ('c', treespec_leaf()), ('a', treespec_none())])
    PyTreeSpec({'a': None, 'b': *, 'c': *})
    >>> treespec_dict()
    PyTreeSpec({})
    >>> treespec_dict(a=treespec_leaf(), b=treespec_tuple([treespec_leaf(), treespec_leaf()]))
    PyTreeSpec({'a': *, 'b': (*, *)})
    >>> treespec_dict({'a': treespec_leaf(), 'b': tree_structure([1, 2])})
    PyTreeSpec({'a': *, 'b': [*, *]})
    >>> treespec_dict({'a': treespec_leaf(), 'b': tree_structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a dict node with the given children.
    """
    return _C.make_from_collection(
        dict(mapping, **kwargs),  # type: ignore[arg-type]
        none_is_leaf,
        namespace,
    )


def treespec_namedtuple(
    namedtuple: NamedTuple[PyTreeSpec],  # type: ignore[type-arg]
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a namedtuple treespec from a namedtuple of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> from collections import namedtuple
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=treespec_leaf()))
    PyTreeSpec(Point(x=*, y=*))
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=treespec_tuple([treespec_leaf(), treespec_leaf()])))
    PyTreeSpec(Point(x=*, y=(*, *)))
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=tree_structure([1, 2])))
    PyTreeSpec(Point(x=*, y=[*, *]))
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=tree_structure([1, 2], none_is_leaf=True)))
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        namedtuple (namedtuple of PyTreeSpec): A namedtuple of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a dict node with the given children.
    """
    if not is_namedtuple_instance(namedtuple):
        raise ValueError(f'Expected a namedtuple of PyTreeSpec(s), got {namedtuple!r}.')
    return _C.make_from_collection(
        namedtuple,  # type: ignore[arg-type]
        none_is_leaf,
        namespace,
    )


def treespec_ordereddict(
    mapping: Mapping[Any, PyTreeSpec] | Iterable[tuple[Any, PyTreeSpec]] = (),
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    """Make an OrderedDict treespec from an OrderedDict of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_ordereddict({'a': treespec_leaf(), 'b': treespec_leaf()})
    PyTreeSpec(OrderedDict({'a': *, 'b': *}))
    >>> treespec_ordereddict([('b', treespec_leaf()), ('c', treespec_leaf()), ('a', treespec_none())])
    PyTreeSpec(OrderedDict({'b': *, 'c': *, 'a': None}))
    >>> treespec_ordereddict()
    PyTreeSpec(OrderedDict())
    >>> treespec_ordereddict(a=treespec_leaf(), b=treespec_tuple([treespec_leaf(), treespec_leaf()]))
    PyTreeSpec(OrderedDict({'a': *, 'b': (*, *)}))
    >>> treespec_ordereddict({'a': treespec_leaf(), 'b': tree_structure([1, 2])})
    PyTreeSpec(OrderedDict({'a': *, 'b': [*, *]}))
    >>> treespec_ordereddict({'a': treespec_leaf(), 'b': tree_structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing an OrderedDict node with the given children.
    """
    return _C.make_from_collection(
        OrderedDict(mapping, **kwargs),  # type: ignore[arg-type]
        none_is_leaf,
        namespace,
    )


def treespec_defaultdict(
    default_factory: Callable[[], Any] | None = None,
    mapping: Mapping[Any, PyTreeSpec] | Iterable[tuple[Any, PyTreeSpec]] = (),
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    """Make a defaultdict treespec from a defaultdict of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_defaultdict(int, {'a': treespec_leaf(), 'b': treespec_leaf()})
    PyTreeSpec(defaultdict(<class 'int'>, {'a': *, 'b': *}))
    >>> treespec_defaultdict(int, [('b', treespec_leaf()), ('c', treespec_leaf()), ('a', treespec_none())])
    PyTreeSpec(defaultdict(<class 'int'>, {'a': None, 'b': *, 'c': *}))
    >>> treespec_defaultdict()
    PyTreeSpec(defaultdict(None, {}))
    >>> treespec_defaultdict(int)
    PyTreeSpec(defaultdict(<class 'int'>, {}))
    >>> treespec_defaultdict(int, a=treespec_leaf(), b=treespec_tuple([treespec_leaf(), treespec_leaf()]))
    PyTreeSpec(defaultdict(<class 'int'>, {'a': *, 'b': (*, *)}))
    >>> treespec_defaultdict(int, {'a': treespec_leaf(), 'b': tree_structure([1, 2])})
    PyTreeSpec(defaultdict(<class 'int'>, {'a': *, 'b': [*, *]}))
    >>> treespec_defaultdict(int, {'a': treespec_leaf(), 'b': tree_structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        default_factory (callable or None, optional): A factory function that will be used to create
            a missing value. (default: :data:`None`)
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a defaultdict node with the given children.
    """
    return _C.make_from_collection(
        defaultdict(default_factory, mapping, **kwargs),  # type: ignore[arg-type]
        none_is_leaf,
        namespace,
    )


def treespec_deque(
    iterable: Iterable[PyTreeSpec] = (),
    maxlen: int | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a deque treespec from a deque of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_deque([treespec_leaf(), treespec_leaf()])
    PyTreeSpec(deque([*, *]))
    >>> treespec_deque([treespec_leaf(), treespec_leaf(), treespec_none()], maxlen=5)
    PyTreeSpec(deque([*, *, None], maxlen=5))
    >>> treespec_deque()
    PyTreeSpec(deque([]))
    >>> treespec_deque([treespec_leaf(), treespec_tuple([treespec_leaf(), treespec_leaf()])])
    PyTreeSpec(deque([*, (*, *)]))
    >>> treespec_deque([treespec_leaf(), tree_structure({'a': 1, 'b': 2})], maxlen=5)
    PyTreeSpec(deque([*, {'a': *, 'b': *}], maxlen=5))
    >>> treespec_deque([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)], maxlen=5)
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        maxlen (int or None, optional): The maximum size of a deque or :data:`None` if unbounded.
            (default: :data:`None`)
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a deque node with the given children.
    """
    return _C.make_from_collection(
        deque(iterable, maxlen=maxlen),  # type: ignore[arg-type]
        none_is_leaf,
        namespace,
    )


def treespec_structseq(
    structseq: PyStructSequence[PyTreeSpec],
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a PyStructSequence treespec from a PyStructSequence of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    Args:
        structseq (PyStructSequence of PyTreeSpec): A PyStructSequence of child treespecs. They must
            have the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a PyStructSequence node with the given children.
    """
    if not is_structseq_instance(structseq):
        raise ValueError(f'Expected a PyStructSequence of PyTreeSpec(s), got {structseq!r}.')
    return _C.make_from_collection(
        structseq,  # type: ignore[arg-type]
        none_is_leaf,
        namespace,
    )


def treespec_from_collection(
    collection: CustomTreeNode[PyTreeSpec],
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Make a treespec from a collection of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_from_collection(None)
    PyTreeSpec(None)
    >>> treespec_from_collection(None, none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec_from_collection(object())
    PyTreeSpec(*)
    >>> treespec_from_collection([treespec_leaf(), treespec_none()])
    PyTreeSpec([*, None])
    >>> treespec_from_collection({'a': treespec_leaf(), 'b': treespec_none()})
    PyTreeSpec({'a': *, 'b': None})
    >>> treespec_from_collection(deque([treespec_leaf(), tree_structure({'a': 1, 'b': 2})], maxlen=5))
    PyTreeSpec(deque([*, {'a': *, 'b': *}], maxlen=5))
    >>> treespec_from_collection({'a': treespec_leaf(), 'b': (treespec_leaf(), treespec_none())})
    Traceback (most recent call last):
        ...
    ValueError: Expected a(n) dict of PyTreeSpec(s), got {'a': PyTreeSpec(*), 'b': (PyTreeSpec(*), PyTreeSpec(None))}.
    >>> treespec_from_collection([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.


    Args:
        collection (collection of PyTreeSpec): A collection of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing the same structure of the collection with the given children.
    """
    return _C.make_from_collection(collection, none_is_leaf, namespace)


def prefix_errors(
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[Callable[[str], ValueError]]:
    """Return a list of errors that would be raised by :func:`broadcast_prefix`."""
    return list(
        _prefix_error(
            PyTreeAccessor(),
            prefix_tree,
            full_tree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ),
    )


STANDARD_DICT_TYPES: frozenset[type] = frozenset({dict, OrderedDict, defaultdict})


# pylint: disable-next=too-many-locals
def _prefix_error(
    accessor: PyTreeAccessor,
    prefix_tree: PyTree[T],
    full_tree: PyTree[S],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> Iterable[Callable[[str], ValueError]]:
    # A leaf is a valid prefix of any tree
    if tree_is_leaf(prefix_tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace):
        return

    # The subtrees may disagree because their roots are of different types:
    prefix_tree_type = type(prefix_tree)
    full_tree_type = type(full_tree)
    both_standard_dict = (
        prefix_tree_type in STANDARD_DICT_TYPES and full_tree_type in STANDARD_DICT_TYPES
    )
    both_deque = prefix_tree_type is deque and full_tree_type is deque  # type: ignore[comparison-overlap]
    if prefix_tree_type is not full_tree_type and (
        # Special handling for dictionary types
        not both_standard_dict
    ):
        yield lambda name: ValueError(
            f'pytree structure error: different types at key path\n'
            f'    {accessor.codify(name) if accessor else name + " tree root"}\n'
            f'At that key path, the prefix pytree {name} has a subtree of type\n'
            f'    {type(prefix_tree)}\n'
            f'but at the same key path the full pytree has a subtree of different type\n'
            f'    {type(full_tree)}.',
        )
        return  # don't look for more errors in this subtree

    # Or they may disagree if their roots have different numbers of children (note that because both
    # prefix_tree and full_tree have the same type at this point, and because prefix_tree is not a
    # leaf, each can be flattened once):
    prefix_tree_children, prefix_tree_metadata, _, __ = tree_flatten_one_level(
        prefix_tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    full_tree_children, full_tree_metadata, _, __ = tree_flatten_one_level(
        full_tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    # Special handling for dictionary types
    if both_standard_dict:
        prefix_tree_keys = (
            prefix_tree_metadata
            if prefix_tree_type is not defaultdict  # type: ignore[comparison-overlap]
            else prefix_tree_metadata[1]  # type: ignore[index]
        )
        full_tree_keys = (
            full_tree_metadata
            if full_tree_type is not defaultdict  # type: ignore[comparison-overlap]
            else full_tree_metadata[1]  # type: ignore[index]
        )
        prefix_tree_keys_set = set(prefix_tree_keys)
        full_tree_keys_set = set(full_tree_keys)
        if prefix_tree_keys_set != full_tree_keys_set:
            missing_keys = sorted(prefix_tree_keys_set.difference(full_tree_keys_set))
            extra_keys = sorted(full_tree_keys_set.difference(prefix_tree_keys_set))
            key_difference = ''
            if missing_keys:
                key_difference += f'\nmissing key(s):\n    {missing_keys}'
            if extra_keys:
                key_difference += f'\nextra key(s):\n    {extra_keys}'
            yield lambda name: ValueError(
                f'pytree structure error: different pytree keys at key path\n'
                f'    {accessor.codify(name) if accessor else name + " tree root"}\n'
                f'At that key path, the prefix pytree {name} has a subtree of type\n'
                f'    {prefix_tree_type}\n'
                f'with {len(prefix_tree_keys)} key(s)\n'
                f'    {prefix_tree_keys}\n'
                f'but at the same key path the full pytree has a subtree of type\n'
                f'    {full_tree_type}\n'
                f'but with {len(full_tree_keys)} key(s)\n'
                f'    {full_tree_keys}{key_difference}',
            )
            return  # don't look for more errors in this subtree

        # If the keys agree, we should ensure that the children are in the same order:
        full_tree_children = [full_tree[k] for k in prefix_tree_keys]  # type: ignore[index]

    if len(prefix_tree_children) != len(full_tree_children):
        yield lambda name: ValueError(
            f'pytree structure error: different numbers of pytree children at key path\n'
            f'    {accessor.codify(name) if accessor else name + " tree root"}\n'
            f'At that key path, the prefix pytree {name} has a subtree of type\n'
            f'    {prefix_tree_type}\n'
            f'with {len(prefix_tree_children)} children, '
            f'but at the same key path the full pytree has a subtree of the same '
            f'type but with {len(full_tree_children)} children.',
        )
        return  # don't look for more errors in this subtree

    # Or they may disagree if their roots have different pytree metadata:
    if (
        prefix_tree_metadata != full_tree_metadata
        and (not both_deque)  # ignore maxlen mismatch for deque
        and (
            # Special handling for dictionary types already done in the keys check above
            not both_standard_dict
        )
    ):
        prefix_tree_metadata_repr = repr(prefix_tree_metadata)
        full_tree_metadata_repr = repr(full_tree_metadata)
        metadata_diff = textwrap.indent(
            '\n'.join(
                difflib.ndiff(
                    prefix_tree_metadata_repr.splitlines(),
                    full_tree_metadata_repr.splitlines(),
                ),
            ),
            prefix='    ',
        )
        yield lambda name: ValueError(
            f'pytree structure error: different pytree metadata at key path\n'
            f'    {accessor.codify(name) if accessor else name + " tree root"}\n'
            f'At that key path, the prefix pytree {name} has a subtree of type\n'
            f'    {prefix_tree_type}\n'
            f'with metadata\n'
            f'    {prefix_tree_metadata_repr}\n'
            f'but at the same key path the full pytree has a subtree of the same '
            f'type but with metadata\n'
            f'    {full_tree_metadata_repr}\n'
            f'so the diff in the metadata at these pytree nodes is\n'
            f'{metadata_diff}',
        )
        return  # don't look for more errors in this subtree

    # If the root types and numbers of children agree, there must be an error in a subtree,
    # so recurse:
    entries = _child_entries(
        prefix_tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    entries_ = _child_entries(
        full_tree,
        is_leaf=is_leaf,  # type: ignore[arg-type]
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    assert entries == entries_ or (
        # Special handling for dictionary types already done in the keys check above
        both_standard_dict
    ), f'equal pytree nodes gave different keys: {entries} and {entries_}'
    # pylint: disable-next=invalid-name
    for e, t1, t2 in zip(entries, prefix_tree_children, full_tree_children):
        yield from _prefix_error(
            accessor + e,
            t1,
            t2,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )


def _child_entries(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[PyTreeEntry]:
    assert not tree_is_leaf(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    ), 'tree must be a non-leaf node'

    counter = itertools.count()
    one_level_accessors = tree_accessors(
        tree,
        is_leaf=lambda x: next(counter) > 0 or (is_leaf is not None and is_leaf(x)),
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )

    assert all(len(a) == 1 for a in one_level_accessors)
    return [a[0] for a in one_level_accessors]
