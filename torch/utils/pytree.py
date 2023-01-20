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
    "_broadcast_to_and_flatten",
]


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")


Context = Any
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


def tree_unflatten(values: Iterable[Any], spec: PyTreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
    if not isinstance(spec, PyTreeSpec):
        raise ValueError(
            f"tree_unflatten(values, spec): Expected `spec` to be instance of "
            f"PyTreeSpec but got item of type {type(spec)}."
        )
    return optree.tree_unflatten(spec, values)


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
    fn: Any,
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return optree.tree_map(
        fn, tree, *rests, none_is_leaf=none_is_leaf, namespace=namespace
    )


def tree_map_(
    fn: Any,
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return optree.tree_map_(
        fn, tree, *rests, none_is_leaf=none_is_leaf, namespace=namespace
    )


Type2 = Tuple[Type[T], Type[S]]
Type3 = Tuple[Type[T], Type[S], Type[U]]
TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]

Fn = Callable[[T], R]
Fn2 = Callable[[Union[T, S]], R]
Fn3 = Callable[[Union[T, S, U]], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]

# These specializations help with type inference on the lambda passed to this
# function
@overload
def map_only(ty: Type[T]) -> MapOnlyFn[Fn[T, Any]]:
    ...


@overload
def map_only(ty: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
    ...


@overload
def map_only(ty: Type3[T, S, U]) -> MapOnlyFn[Fn3[T, S, U, Any]]:
    ...


# This specialization is needed for the implementations below that call
@overload
def map_only(ty: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    ...


def map_only(ty: TypeAny) -> MapOnlyFn[FnAny[Any]]:
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

    def deco(f: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(f)
        def inner(x: T) -> Any:
            if isinstance(x, ty):
                return f(x)
            else:
                return x

        return inner

    return deco


@overload
def tree_map_only(
    ty: Type[T],
    fn: Fn[T, Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


@overload
def tree_map_only(
    ty: Type2[T, S],
    fn: Fn2[T, S, Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


@overload
def tree_map_only(
    ty: Type3[T, S, U],
    fn: Fn3[T, S, U, Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


def tree_map_only(
    ty: TypeAny,
    fn: FnAny[Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return tree_map(
        map_only(ty)(fn),
        tree,
        *rests,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@overload
def tree_map_only_(
    ty: Type[T],
    fn: Fn[T, Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


@overload
def tree_map_only_(
    ty: Type2[T, S],
    fn: Fn2[T, S, Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


@overload
def tree_map_only_(
    ty: Type3[T, S, U],
    fn: Fn3[T, S, U, Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


def tree_map_only_(
    ty: TypeAny,
    fn: FnAny[Any],
    tree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return tree_map_(
        map_only(ty)(fn),
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
    ty: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


@overload
def tree_all_only(
    ty: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


@overload
def tree_all_only(
    ty: Type3[T, S, U],
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


def tree_all_only(
    ty: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return all(pred(x) for x in flat_args if isinstance(x, ty))


@overload
def tree_any_only(
    ty: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


@overload
def tree_any_only(
    ty: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


@overload
def tree_any_only(
    ty: Type3[T, S, U],
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


def tree_any_only(
    ty: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return any(pred(x) for x in flat_args if isinstance(x, ty))


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
    spec: PyTreeSpec,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> Optional[List[Any]]:
    assert isinstance(spec, PyTreeSpec)
    full_tree = tree_unflatten([0] * spec.num_leaves, spec)
    try:
        return optree.broadcast_prefix(
            tree, full_tree, none_is_leaf=none_is_leaf, namespace=namespace
        )
    except ValueError:
        return None
