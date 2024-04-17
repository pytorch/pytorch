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

import functools
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    overload,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch

if torch._running_with_deploy():
    raise ImportError("C++ pytree utilities do not work with torch::deploy.")

import optree
from optree import PyTreeSpec  # direct import for type annotations


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


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")


Context = Any
PyTree = Any
TreeSpec = PyTreeSpec
FlattenFunc = Callable[[PyTree], Tuple[List[Any], Context]]
UnflattenFunc = Callable[[Iterable[Any], Context], PyTree]
OpTreeUnflattenFunc = Callable[[Context, Iterable[Any]], PyTree]
DumpableContext = Any  # Any json dumpable text
ToDumpableContextFn = Callable[[Context], DumpableContext]
FromDumpableContextFn = Callable[[DumpableContext], Context]


def _reverse_args(func: UnflattenFunc) -> OpTreeUnflattenFunc:
    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return func(*reversed(args), **kwargs)

    return wrapped


def register_pytree_node(
    cls: Type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
) -> None:
    """Register a container-like type as pytree node.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_fn (callable): A function to be used during flattening, taking an instance of
            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened
            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be
            passed to the ``unflatten_fn``.
        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was
            returned by ``flatten_fn`` and stored in the treespec, and the unflattened children.
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
        ...     lambda s: (sorted(s), None, None),
        ...     lambda children, _: set(children),
        ... )
    """
    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )

    from . import _pytree as python

    python._private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )


def _register_pytree_node(
    cls: Type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
) -> None:
    """Register a container-like type as pytree node for the C++ pytree only.

    The ``namespace`` argument is used to avoid collisions that occur when different libraries
    register the same Python type with different behaviors. It is recommended to add a unique prefix
    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify
    the same class in different namespaces for different use cases.

    .. warning::
        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is
        used to isolate the behavior of flattening and unflattening a pytree node type. This is to
        prevent accidental collisions between different libraries that may register the same type.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_fn (callable): A function to be used during flattening, taking an instance of
            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened
            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be
            passed to the ``unflatten_fn``.
        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was
            returned by ``flatten_fn`` and stored in the treespec, and the unflattened children.
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
    """
    warnings.warn(
        "torch.utils._cxx_pytree._register_pytree_node is deprecated. "
        "Please use torch.utils._cxx_pytree.register_pytree_node instead.",
        stacklevel=2,
    )

    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )


def _private_register_pytree_node(
    cls: Type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
) -> None:
    """This is an internal function that is used to register a pytree node type
    for the C++ pytree only. End-users should use :func:`register_pytree_node`
    instead.
    """
    # TODO(XuehaiPan): remove this condition when we make Python pytree out-of-box support
    # PyStructSequence types
    if not optree.is_structseq_class(cls):
        optree.register_pytree_node(
            cls,
            flatten_fn,
            _reverse_args(unflatten_fn),
            namespace="torch",
        )


def tree_flatten(tree: PyTree) -> Tuple[List[Any], TreeSpec]:
    """Flatten a pytree.

    See also :func:`tree_unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten(tree)
    ([1, 2, 3, 4, None, 5], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf))
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*, NoneIsLeaf))
    >>> tree_flatten(None)
    ([None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten(tree)
    ([2, 3, 4, 1, None, 5], PyTreeSpec(OrderedDict([('b', (*, [*, *])), ('a', *), ('c', *), ('d', *)]), NoneIsLeaf))

    Args:
        tree (pytree): A pytree to flatten.

    Returns:
        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the
        second element is a treespec representing the structure of the pytree.
    """
    return optree.tree_flatten(  # type: ignore[return-value]
        tree,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    """Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> leaves, treespec = tree_flatten(tree)
    >>> tree == tree_unflatten(leaves, treespec)
    True

    Args:
        leaves (iterable): The list of leaves to use for reconstruction. The list must match the
            number of leaves of the treespec.
        treespec (TreeSpec): The treespec to reconstruct.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by
        ``treespec``.
    """
    if not isinstance(treespec, TreeSpec):
        raise TypeError(
            f"tree_unflatten(values, spec): Expected `spec` to be instance of "
            f"TreeSpec but got item of type {type(treespec)}."
        )
    return optree.tree_unflatten(treespec, leaves)  # type: ignore[arg-type]


def tree_leaves(tree: PyTree) -> List[Any]:
    """Get the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_leaves(tree)
    [1, 2, 3, 4, None, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    [None]

    Args:
        tree (pytree): A pytree to flatten.

    Returns:
        A list of leaf values.
    """
    return optree.tree_leaves(tree, none_is_leaf=True, namespace="torch")


def tree_structure(tree: PyTree) -> TreeSpec:
    """Get the treespec for a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_structure(tree)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    >>> tree_structure(1)
    PyTreeSpec(*, NoneIsLeaf)
    >>> tree_structure(None)
    PyTreeSpec(*, NoneIsLeaf)

    Args:
        tree (pytree): A pytree to flatten.

    Returns:
        A treespec object representing the structure of the pytree.
    """
    return optree.tree_structure(  # type: ignore[return-value]
        tree,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_map(func: Callable[..., Any], tree: PyTree) -> PyTree:
    """Map a function over leaves in a pytree to produce a new pytree.

    See also :func:`tree_map_`.

    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': True}

    Args:
        func (callable): A function that takes a single argument, to be applied at the corresponding
            leaves of the pytree.
        tree (pytree): A pytree to be mapped over, with each leaf providing the argument to function
            ``func``.

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x)`` where ``x`` is the value at the corresponding leaf in ``tree``.
    """
    return optree.tree_map(
        func,
        tree,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_map_(func: Callable[..., Any], tree: PyTree) -> PyTree:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`.

    Args:
        func (callable): A function that takes a single argument, to be applied at the corresponding
            leaves of the pytree.
        tree (pytree): A pytree to be mapped over, with each leaf providing the argument to function
            ``func``.

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x)`` (not the return value) where ``x`` is the value at the corresponding leaf in
        ``tree``.
    """
    return optree.tree_map_(
        func,
        tree,
        none_is_leaf=True,
        namespace="torch",
    )


Type2 = Tuple[Type[T], Type[S]]
Type3 = Tuple[Type[T], Type[S], Type[U]]
TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]

Fn2 = Callable[[Union[T, S]], R]
Fn3 = Callable[[Union[T, S, U]], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]


# These specializations help with type inference on the lambda passed to this
# function
@overload
def map_only(__type_or_types: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
    ...


@overload
def map_only(__type_or_types: Type3[T, S, U]) -> MapOnlyFn[Fn3[T, S, U, Any]]:
    ...


@overload
def map_only(__type_or_types: Type[T]) -> MapOnlyFn[Fn[T, Any]]:
    ...


# This specialization is needed for the implementations below that call
@overload
def map_only(__type_or_types: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    ...


def map_only(__type_or_types: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    """
    Suppose you are writing a tree_map over tensors, leaving everything
    else unchanged.  Ordinarily you would have to write:

        def go(t):
            if isinstance(t, Tensor):
                return ...
            else:
                return t

    With this function, you only need to write:

        @map_only(Tensor)
        def go(t):
            return ...

    You can also directly use 'tree_map_only'
    """

    def wrapper(func: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(func)
        def wrapped(x: T) -> Any:
            if isinstance(x, __type_or_types):
                return func(x)
            return x

        return wrapped

    return wrapper


@overload
def tree_map_only(
    __type_or_types: Type[T],
    func: Fn[T, Any],
    tree: PyTree,
) -> PyTree:
    ...


@overload
def tree_map_only(
    __type_or_types: Type2[T, S],
    func: Fn2[T, S, Any],
    tree: PyTree,
) -> PyTree:
    ...


@overload
def tree_map_only(
    __type_or_types: Type3[T, S, U],
    func: Fn3[T, S, U, Any],
    tree: PyTree,
) -> PyTree:
    ...


def tree_map_only(
    __type_or_types: TypeAny,
    func: FnAny[Any],
    tree: PyTree,
) -> PyTree:
    return tree_map(map_only(__type_or_types)(func), tree)


@overload
def tree_map_only_(
    __type_or_types: Type[T],
    func: Fn[T, Any],
    tree: PyTree,
) -> PyTree:
    ...


@overload
def tree_map_only_(
    __type_or_types: Type2[T, S],
    func: Fn2[T, S, Any],
    tree: PyTree,
) -> PyTree:
    ...


@overload
def tree_map_only_(
    __type_or_types: Type3[T, S, U],
    func: Fn3[T, S, U, Any],
    tree: PyTree,
) -> PyTree:
    ...


def tree_map_only_(
    __type_or_types: TypeAny,
    func: FnAny[Any],
    tree: PyTree,
) -> PyTree:
    return tree_map_(map_only(__type_or_types)(func), tree)


def tree_all(pred: Callable[[Any], bool], tree: PyTree) -> bool:
    flat_args = tree_leaves(tree)
    return all(map(pred, flat_args))


def tree_any(pred: Callable[[Any], bool], tree: PyTree) -> bool:
    flat_args = tree_leaves(tree)
    return any(map(pred, flat_args))


@overload
def tree_all_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
) -> bool:
    ...


@overload
def tree_all_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
) -> bool:
    ...


@overload
def tree_all_only(
    __type_or_types: Type3[T, S, U],
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
) -> bool:
    ...


def tree_all_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
) -> bool:
    flat_args = tree_leaves(tree)
    return all(pred(x) for x in flat_args if isinstance(x, __type_or_types))


@overload
def tree_any_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
) -> bool:
    ...


@overload
def tree_any_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
) -> bool:
    ...


@overload
def tree_any_only(
    __type_or_types: Type3[T, S, U],
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
) -> bool:
    ...


def tree_any_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
) -> bool:
    flat_args = tree_leaves(tree)
    return any(pred(x) for x in flat_args if isinstance(x, __type_or_types))


def broadcast_prefix(prefix_tree: PyTree, full_tree: PyTree) -> List[Any]:
    """Return a list of broadcasted leaves in ``prefix_tree`` to match the number of leaves in ``full_tree``.

    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be
    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.

    This function returns a list of leaves with the same size as ``full_tree``. The leaves are
    replicated from ``prefix_tree``. The number of replicas is determined by the corresponding
    subtree in ``full_tree``.

    >>> broadcast_prefix(1, [1, 2, 3])
    [1, 1, 1]
    >>> broadcast_prefix([1, 2, 3], [1, 2, 3])
    [1, 2, 3]
    >>> broadcast_prefix([1, 2, 3], [1, 2, 3, 4])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4; list: [1, 2, 3, 4].
    >>> broadcast_prefix([1, 2, 3], [1, 2, (3, 4)])
    [1, 2, 3, 3]
    >>> broadcast_prefix([1, 2, 3], [1, 2, {'a': 3, 'b': 4, 'c': (None, 5)}])
    [1, 2, 3, 3, 3, 3]

    Args:
        prefix_tree (pytree): A pytree with the same structure as a prefix of ``full_tree``.
        full_tree (pytree): A pytree with the same structure as a suffix of ``prefix_tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.

    Returns:
        A list of leaves in ``prefix_tree`` broadcasted to match the number of leaves in ``full_tree``.
    """
    return optree.broadcast_prefix(
        prefix_tree,
        full_tree,
        none_is_leaf=True,
        namespace="torch",
    )


# Broadcasts a pytree to the provided TreeSpec and returns the flattened
# values. If this is not possible, then this function returns None.
#
# For example, given pytree=0 and spec=TreeSpec(list, None, [LeafSpec(), LeafSpec()]),
# would return [0, 0]. This is useful for part of the vmap implementation:
# a user can pass in vmap(fn, in_dims)(*inputs). `in_dims` should be
# broadcastable to the tree structure of `inputs` and we use
# _broadcast_to_and_flatten to check this.
def _broadcast_to_and_flatten(tree: PyTree, treespec: TreeSpec) -> Optional[List[Any]]:
    assert isinstance(treespec, TreeSpec)
    full_tree = tree_unflatten([0] * treespec.num_leaves, treespec)
    try:
        return broadcast_prefix(tree, full_tree)
    except ValueError:
        return None


def treespec_dumps(treespec: TreeSpec, protocol: Optional[int] = None) -> str:
    """Serialize a treespec to a JSON string."""
    if not isinstance(treespec, TreeSpec):
        raise TypeError(
            f"treespec_dumps(spec): Expected `spec` to be instance of "
            f"TreeSpec but got item of type {type(treespec)}."
        )
    from ._pytree import (
        tree_structure as _tree_structure,
        treespec_dumps as _treespec_dumps,
    )

    orig_treespec = _tree_structure(tree_unflatten([0] * treespec.num_leaves, treespec))
    return _treespec_dumps(orig_treespec, protocol=protocol)


def treespec_loads(serialized: str) -> TreeSpec:
    """Deserialize a treespec from a JSON string."""
    from ._pytree import (
        tree_unflatten as _tree_unflatten,
        treespec_loads as _treespec_loads,
    )

    orig_treespec = _treespec_loads(serialized)
    dummy_tree = _tree_unflatten([0] * orig_treespec.num_leaves, orig_treespec)
    treespec = tree_structure(dummy_tree)
    return treespec


class _DummyLeaf:
    def __repr__(self) -> str:
        return "*"


def treespec_pprint(treespec: TreeSpec) -> str:
    dummy_tree = tree_unflatten(
        [_DummyLeaf() for _ in range(treespec.num_leaves)],
        treespec,
    )
    return repr(dummy_tree)


class LeafSpecMeta(type(TreeSpec)):  # type: ignore[misc]
    def __instancecheck__(self, instance: object) -> bool:
        return isinstance(instance, TreeSpec) and instance.is_leaf()


class LeafSpec(TreeSpec, metaclass=LeafSpecMeta):
    def __new__(cls) -> "LeafSpec":
        return optree.treespec_leaf(none_is_leaf=True)  # type: ignore[return-value]
