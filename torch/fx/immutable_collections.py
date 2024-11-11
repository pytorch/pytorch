from typing import Any, Dict, Iterable, List, NoReturn, Tuple, Type, TypeVar
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


_help_mutation = """
If you are attempting to modify the kwargs or args of a torch.fx.Node object,
instead create a new copy of it and assign the copy to the node:

    new_args = ...  # copy and mutate args
    node.args = new_args
""".strip()


_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _no_mutation(self, *args: Any, **kwargs: Any) -> NoReturn:  # type: ignore[no-untyped-def]
    raise TypeError(
        f"{type(self).__name__!r} object does not support mutation. {_help_mutation}",
    )


@compatibility(is_backward_compatible=True)
class immutable_list(List[_T]):
    """An immutable version of :class:`list`."""

    __delitem__ = _no_mutation
    __iadd__ = _no_mutation
    __imul__ = _no_mutation
    __setitem__ = _no_mutation
    append = _no_mutation
    clear = _no_mutation
    extend = _no_mutation
    insert = _no_mutation
    pop = _no_mutation
    remove = _no_mutation
    reverse = _no_mutation
    sort = _no_mutation

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self))

    def __reduce__(self) -> Tuple[Type[Self], Tuple[Tuple[_T, ...]]]:
        return (type(self), (tuple(self),))


@compatibility(is_backward_compatible=True)
class immutable_dict(Dict[_KT, _VT]):
    """An immutable version of :class:`dict`."""

    __delitem__ = _no_mutation
    __ior__ = _no_mutation
    __setitem__ = _no_mutation
    clear = _no_mutation
    pop = _no_mutation
    popitem = _no_mutation
    setdefault = _no_mutation
    update = _no_mutation  # type: ignore[assignment]

    def __hash__(self) -> int:  # type: ignore[override]
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
