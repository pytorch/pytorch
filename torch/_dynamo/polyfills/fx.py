from typing import Any, Callable

from torch._C import _fx_map_aggregate, _fx_map_arg
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.node import Node

from ..decorators import substitute_in_graph


@substitute_in_graph(_fx_map_arg, can_constant_fold_through=True)
def map_arg(a: Any, fn: Callable[[Node], Any]) -> Any:
    return map_aggregate(a, lambda x: fn(x) if isinstance(x, Node) else x)


@substitute_in_graph(_fx_map_aggregate, can_constant_fold_through=True)
def map_aggregate(a: Any, fn: Callable[[Any], Any]) -> Any:
    result: Any
    if isinstance(a, tuple):
        it = (map_aggregate(elem, fn) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        result = type(a)(*it) if hasattr(a, "_fields") else tuple(it)
    elif isinstance(a, list):
        result = immutable_list([map_aggregate(elem, fn) for elem in a])
    elif isinstance(a, dict):
        result = immutable_dict([(k, map_aggregate(v, fn)) for k, v in a.items()])
    elif isinstance(a, slice):
        result = slice(
            map_aggregate(a.start, fn),
            map_aggregate(a.stop, fn),
            map_aggregate(a.step, fn),
        )
    else:
        result = fn(a)
    return result


__all__ = [
    "map_arg",
    "map_aggregate",
]
