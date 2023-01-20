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
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    namespace: str = "torch",
) -> None:
    optree.register_pytree_node(
        cls, flatten_fn, _reverse_args(unflatten_fn), namespace=namespace
    )


def tree_flatten(
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> Tuple[List[Any], PyTreeSpec]:
    return optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)


def tree_unflatten(leaves: Iterable[Any], treespec: PyTreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
    if not isinstance(treespec, PyTreeSpec):
        raise ValueError(
            f"tree_unflatten(values, spec): Expected `spec` to be instance of "
            f"PyTreeSpec but got item of type {type(treespec)}."
        )
    return optree.tree_unflatten(treespec, leaves)


def tree_leaves(
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> List[Any]:
    return optree.tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)


def tree_structure(
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTreeSpec:
    return optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)


def tree_map(
    func: Any,
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return optree.tree_map(
        func, tree, *rests, none_is_leaf=none_is_leaf, namespace=namespace
    )


def tree_map_(
    func: Any,
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return optree.tree_map_(
        func, tree, *rests, none_is_leaf=none_is_leaf, namespace=namespace
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
    return optree.broadcast_prefix(
        prefix_tree, full_tree, none_is_leaf=none_is_leaf, namespace=namespace
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
            tree, full_tree, none_is_leaf=none_is_leaf, namespace=namespace
        )
    except ValueError:
        return None
