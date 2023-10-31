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

import optree
from optree import PyTreeSpec  # direct import for type annotations


__all__ = [
    "PyTree",
    "PyTreeSpec",
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
    "broadcast_prefix",
    "_broadcast_to_and_flatten",
    "treespec_dumps",
    "treespec_loads",
]


T = TypeVar("T")
S = TypeVar("S")
R = TypeVar("R")


Context = Optional[Any]
PyTree = Any
TreeSpec = PyTreeSpec
FlattenFunc = Callable[[PyTree], Tuple[List, Context]]
UnflattenFunc = Callable[[Iterable, Context], PyTree]
OpTreeUnflattenFunc = Callable[[Context, Iterable], PyTree]


def _reverse_args(func: UnflattenFunc) -> OpTreeUnflattenFunc:
    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return func(*reversed(args), **kwargs)

    return wrapped


def register_pytree_node(
    cls: Type[Any],
    flatten_func: FlattenFunc,
    unflatten_func: UnflattenFunc,
    namespace: str = "torch",
) -> None:
    """Extend the set of types that are considered internal nodes in pytrees.

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
        flatten_fn (callable): A function to be used during flattening, taking an instance of ``cls``
            and returning a triple or optionally a pair, with (1) an iterable for the children to be
            flattened recursively, and (2) some hashable auxiliary data to be stored in the treespec
            and to be passed to the ``unflatten_func``, and (3) (optional) an iterable for the tree
            path entries to the corresponding children. If the entries are not provided or given by
            :data:`None`, then `range(len(children))` will be used.
        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was returned
            by ``flatten_func`` and stored in the treespec, and the unflattened children. The function
            should return an instance of ``cls``.
        namespace (str, optional): A non-empty string that uniquely identifies the namespace of the
            type registry. This is used to isolate the registry from other modules that might register
            a different custom behavior for the same type. (default: :const:`"torch"`)

    Example::

        >>> # xdoctest: +SKIP
        >>> # Registry a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda children, _: set(children),
        ...     namespace='set',
        ... )

        >>> # xdoctest: +SKIP
        >>> # Register a Python type into a namespace
        >>> import torch
        >>> register_pytree_node(
        ...     torch.Tensor,
        ...     flatten_func=lambda tensor: (
        ...         (tensor.cpu().detach().numpy(),),
        ...         {'dtype': tensor.dtype, 'device': tensor.device, 'requires_grad': tensor.requires_grad},
        ...     ),
        ...     unflatten_func=lambda children, metadata: torch.tensor(children[0], **metadata),
        ...     namespace='torch2numpy',
        ... )

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> tree = {'weight': torch.ones(size=(1, 2)).cuda(), 'bias': torch.zeros(size=(2,))}
        >>> tree
        {'weight': tensor([[1., 1.]], device='cuda:0'), 'bias': tensor([0., 0.])}

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # Flatten without specifying the namespace
        >>> tree_flatten(tree)  # `torch.Tensor`s are leaf nodes  # xdoctest: +SKIP
        ([tensor([0., 0.]), tensor([[1., 1.]], device='cuda:0')], PyTreeSpec({'bias': *, 'weight': *}))

        >>> # xdoctest: +SKIP
        >>> # Flatten with the namespace
        >>> tree_flatten(tree, namespace='torch2numpy')  # xdoctest: +SKIP
        (
            [array([0., 0.], dtype=float32), array([[1., 1.]], dtype=float32)],
            PyTreeSpec(
                {
                    'bias': CustomTreeNode(Tensor[{'dtype': torch.float32, ...}], [*]),
                    'weight': CustomTreeNode(Tensor[{'dtype': torch.float32, ...}], [*])
                },
                namespace='torch2numpy'
            )
        )

        >>> # xdoctest: +SKIP
        >>> # Register the same type with a different namespace for different behaviors
        >>> def tensor2flatparam(tensor):
        ...     return [torch.nn.Parameter(tensor.reshape(-1))], tensor.shape, None
        ...
        >>> def flatparam2tensor(children, metadata):
        ...     return children[0].reshape(metadata)
        ...
        >>> register_pytree_node(
        ...     torch.Tensor,
        ...     flatten_func=tensor2flatparam,
        ...     unflatten_func=flatparam2tensor,
        ...     namespace='tensor2flatparam',
        ... )

        >>> # xdoctest: +SKIP
        >>> # Flatten with the new namespace
        >>> tree_flatten(tree, namespace='tensor2flatparam')  # xdoctest: +SKIP
        (
            [
                Parameter containing: tensor([0., 0.], requires_grad=True),
                Parameter containing: tensor([1., 1.], device='cuda:0', requires_grad=True)
            ],
            PyTreeSpec(
                {
                    'bias': CustomTreeNode(Tensor[torch.Size([2])], [*]),
                    'weight': CustomTreeNode(Tensor[torch.Size([1, 2])], [*])
                },
                namespace='tensor2flatparam'
            )
        )
    """
    from ._pytree import _register_pytree_node

    _register_pytree_node(
        cls,
        flatten_func,
        unflatten_func,
    )

    optree.register_pytree_node(
        cls,
        flatten_func,
        _reverse_args(unflatten_func),
        namespace=namespace,
    )


def tree_flatten(
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> Tuple[List[Any], PyTreeSpec]:
    """Flatten a pytree.

    See also :func:`tree_unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten(tree)
    ([1, 2, 3, 4, None, 5], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf))
    >>> tree_flatten(tree, none_is_leaf=False)
    ([1, 2, 3, 4, 5], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *}))
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*, NoneIsLeaf))
    >>> tree_flatten(None)
    ([None], PyTreeSpec(*, NoneIsLeaf))
    >>> tree_flatten(None, none_is_leaf=False)
    ([], PyTreeSpec(None))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten(tree)
    ([2, 3, 4, 1, None, 5], PyTreeSpec(OrderedDict([('b', (*, [*, *])), ('a', *), ('c', *), ('d', *)]), NoneIsLeaf))
    >>> tree_flatten(tree, none_is_leaf=False)
    ([2, 3, 4, 1, 5], PyTreeSpec(OrderedDict([('b', (*, [*, *])), ('a', *), ('c', None), ('d', *)])))

    Args:
        tree (pytree): A pytree to flatten.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`True`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`"torch"`)

    Returns:
        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the
        second element is a treespec representing the structure of the pytree.
    """
    return optree.tree_flatten(  # type: ignore[return-value]
        tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def tree_unflatten(leaves: Iterable[Any], treespec: PyTreeSpec) -> PyTree:
    """Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> leaves, treespec = tree_flatten(tree)
    >>> tree == tree_unflatten(leaves, treespec)
    True

    Args:
        leaves (iterable): The list of leaves to use for reconstruction. The list must match the
            number of leaves of the treespec.
        treespec (PyTreeSpec): The treespec to reconstruct.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by
        ``treespec``.
    """
    if not isinstance(treespec, PyTreeSpec):
        raise TypeError(
            f"tree_unflatten(values, spec): Expected `spec` to be instance of "
            f"PyTreeSpec but got item of type {type(treespec)}."
        )
    return optree.tree_unflatten(treespec, leaves)  # type: ignore[arg-type]


def tree_leaves(
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> List[Any]:
    """Get the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_leaves(tree)
    [1, 2, 3, 4, None, 5]
    >>> tree_leaves(tree, none_is_leaf=False)
    [1, 2, 3, 4, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    [None]
    >>> tree_leaves(None, none_is_leaf=False)
    []

    Args:
        tree (pytree): A pytree to flatten.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`True`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`"torch"`)

    Returns:
        A list of leaf values.
    """
    return optree.tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)


def tree_structure(
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTreeSpec:
    """Get the treespec for a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_structure(tree)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    >>> tree_structure(tree, none_is_leaf=False)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    >>> tree_structure(1)
    PyTreeSpec(*, NoneIsLeaf)
    >>> tree_structure(None)
    PyTreeSpec(*, NoneIsLeaf)
    >>> tree_structure(None, none_is_leaf=False)
    PyTreeSpec(None)

    Args:
        tree (pytree): A pytree to flatten.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`True`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`"torch"`)

    Returns:
        A treespec object representing the structure of the pytree.
    """
    return optree.tree_structure(  # type: ignore[return-value]
        tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`.

    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': True}
    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64), 'z': None}, none_is_leaf=False)
    {'x': 8, 'y': (43, 65), 'z': None}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None}, none_is_leaf=False)
    {'x': False, 'y': (False, False), 'z': None}

    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:

    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytrees): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`True`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`"torch"`)

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    """
    return optree.tree_map(
        func,
        tree,
        *rests,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def tree_map_(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`.

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytrees): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`True`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`"torch"`)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.
    """
    return optree.tree_map_(
        func,
        tree,
        *rests,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


Type2 = Tuple[Type[T], Type[S]]
TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]

Fn2 = Callable[[Union[T, S]], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]


# These specializations help with type inference on the lambda passed to this
# function
@overload
def map_only(__type_or_types: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
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
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


@overload
def tree_map_only(
    __type_or_types: Type2[T, S],
    func: Fn2[T, S, Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


def tree_map_only(
    __type_or_types: TypeAny,
    func: FnAny[Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return tree_map(
        map_only(__type_or_types)(func),
        tree,
        *rests,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@overload
def tree_map_only_(
    __type_or_types: Type[T],
    func: Fn[T, Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


@overload
def tree_map_only_(
    __type_or_types: Type2[T, S],
    func: Fn2[T, S, Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


def tree_map_only_(
    __type_or_types: TypeAny,
    func: FnAny[Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return tree_map_(
        map_only(__type_or_types)(func),
        tree,
        *rests,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def tree_all(
    pred: Callable[[Any], bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return all(map(pred, flat_args))


def tree_any(
    pred: Callable[[Any], bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return any(map(pred, flat_args))


@overload
def tree_all_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


@overload
def tree_all_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


def tree_all_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return all(pred(x) for x in flat_args if isinstance(x, __type_or_types))


@overload
def tree_any_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


@overload
def tree_any_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


def tree_any_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return any(pred(x) for x in flat_args if isinstance(x, __type_or_types))


def broadcast_prefix(
    prefix_tree: PyTree,
    full_tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> List[Any]:
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
    >>> broadcast_prefix([1, 2, 3], [1, 2, {'a': 3, 'b': 4, 'c': (None, 5)}], none_is_leaf=False)
    [1, 2, 3, 3, 3]

    Args:
        prefix_tree (pytree): A pytree with the same structure as a prefix of ``full_tree``.
        full_tree (pytree): A pytree with the same structure as a suffix of ``prefix_tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`True`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`"torch"`)

    Returns:
        A list of leaves in ``prefix_tree`` broadcasted to match the number of leaves in ``full_tree``.
    """
    return optree.broadcast_prefix(
        prefix_tree,
        full_tree,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


# Broadcasts a pytree to the provided TreeSpec and returns the flattened
# values. If this is not possible, then this function returns None.
#
# For example, given pytree=0 and spec=TreeSpec(list, None, [LeafSpec(), LeafSpec()]),
# would return [0, 0]. This is useful for part of the vmap implementation:
# a user can pass in vmap(fn, in_dims)(*inputs). `in_dims` should be
# broadcastable to the tree structure of `inputs` and we use
# _broadcast_to_and_flatten to check this.
def _broadcast_to_and_flatten(
    tree: PyTree,
    treespec: PyTreeSpec,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> Optional[List[Any]]:
    assert isinstance(treespec, PyTreeSpec)
    full_tree = tree_unflatten([0] * treespec.num_leaves, treespec)
    try:
        return broadcast_prefix(
            tree,
            full_tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    except ValueError:
        return None


def treespec_dumps(treespec: PyTreeSpec) -> str:
    """Serialize a treespec to a JSON string."""
    if not isinstance(treespec, PyTreeSpec):
        raise TypeError(
            f"treespec_dumps(spec): Expected `spec` to be instance of "
            f"PyTreeSpec but got item of type {type(treespec)}."
        )
    from ._pytree import (
        tree_structure as _tree_structure,
        treespec_dumps as _treespec_dumps,
    )

    orig_treespec = _tree_structure(tree_unflatten([0] * treespec.num_leaves, treespec))
    return _treespec_dumps(orig_treespec)


def treespec_loads(serialized: str) -> PyTreeSpec:
    """Deserialize a treespec from a JSON string."""
    from ._pytree import (
        tree_unflatten as _tree_unflatten,
        treespec_loads as _treespec_loads,
    )

    orig_treespec = _treespec_loads(serialized)
    dummy_tree = _tree_unflatten([0] * orig_treespec.num_leaves, orig_treespec)
    treespec = tree_structure(dummy_tree)
    return treespec


class PyTreeLeafSpecMeta(type(PyTreeSpec)):  # type: ignore[misc]
    def __instancecheck__(self, instance: object) -> bool:
        return isinstance(instance, PyTreeSpec) and instance.is_leaf()


class PyTreeLeafSpec(PyTreeSpec, metaclass=PyTreeLeafSpecMeta):
    def __new__(cls, none_is_leaf: bool = True) -> "PyTreeLeafSpec":
        return optree.treespec_leaf(none_is_leaf=none_is_leaf)  # type: ignore[return-value]


LeafSpec = PyTreeLeafSpec
