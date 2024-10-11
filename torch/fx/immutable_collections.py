# mypy: allow-untyped-defs
from typing import Any, Dict, Iterable, List, Tuple, Dict, TypeVar, NoReturn, Type
from typing_extensions import Self

from torch.utils._pytree import (
    _dict_flatten,
    _dict_flatten_with_keys,
    _dict_unflatten,
    _list_flatten,
    _list_flatten_with_keys,
    _list_unflatten,
    Context,
    register_pytree_node,
)

from ._compatibility import compatibility


__all__ = ["immutable_list", "immutable_dict"]


_help_mutation = """\
If you are attempting to modify the kwargs or args of a torch.fx.Node object,
instead create a new copy of it and assign the copy to the node:
    new_args = ... # copy and mutate args
    node.args = new_args
"""


_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _no_mutation(self, *args: Any, **kwargs: Any) -> NoReturn:
    raise NotImplementedError(
        f"'{type(self).__name__}' object does not support mutation. {_help_mutation}",
    )


class _ImmutableMixin:
    def __init_subclass__(cls, mutable_methods_to_disable: Iterable[str]) -> None:
        super().__init_subclass__()
        for method in mutable_methods_to_disable:
            setattr(cls, method, _no_mutation)


@compatibility(is_backward_compatible=True)
class immutable_list(
    _ImmutableMixin,
    List[_T],
    mutable_methods_to_disable=(
        "__delitem__",
        "__iadd__",
        "__imul__",
        "__setitem__",
        "append",
        "clear",
        "extend",
        "insert",
        "pop",
        "remove",
        "reverse",
        "sort",
    ),
):
    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self))

    def __reduce__(self)-> Tuple[Type[Self], Tuple[Tuple[_T, ...]]]:
        return (type(self), (tuple(self),))


@compatibility(is_backward_compatible=True)
class immutable_dict(
    _ImmutableMixin,
    Dict[_KT, _VT],
    mutable_methods_to_disable=(
        "__delitem__",
        "__ior__",
        "__setitem__",
        "clear",
        "pop",
        "popitem",
        "setdefault",
        "update",
    ),
):
    def __hash__(self) -> int: # type: ignore[override]
        return hash(tuple(self.items()))

    def __reduce__(self) -> Tuple[Type[Self], Tuple[Tuple[Tuple[_KT, _VT], ...]]]:
        return (type(self), (tuple(self.items()),))


# Register immutable collections for PyTree operations
def _immutable_dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return _dict_flatten(d)


def _immutable_dict_unflatten(
    values: Iterable[Any],
    context: Context,
) -> Dict[Any, Any]:
    return immutable_dict(_dict_unflatten(values, context))


def _immutable_list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return _list_flatten(d)


def _immutable_list_unflatten(
    values: Iterable[Any],
    context: Context,
) -> List[Any]:
    return immutable_list(_list_unflatten(values, context))


register_pytree_node(
    immutable_dict,
    _immutable_dict_flatten,
    _immutable_dict_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_dict",
    flatten_with_keys_fn=_dict_flatten_with_keys,
)
register_pytree_node(
    immutable_list,
    _immutable_list_flatten,
    _immutable_list_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_list",
    flatten_with_keys_fn=_list_flatten_with_keys,
)
