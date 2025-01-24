from collections import namedtuple
from typing import Any, Callable, Optional, TypeVar
from typing_extensions import NamedTuple

import torch.return_types
from torch.utils._pytree import PyTree, TreeSpec


FlattenFuncSpec = Callable[[PyTree, TreeSpec], list]
FlattenFuncExactMatchSpec = Callable[[PyTree, TreeSpec], bool]

SUPPORTED_NODES: dict[type[Any], FlattenFuncSpec] = {}
SUPPORTED_NODES_EXACT_MATCH: dict[type[Any], Optional[FlattenFuncExactMatchSpec]] = {}

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")


def register_pytree_flatten_spec(
    cls: type[Any],
    flatten_fn_spec: FlattenFuncSpec,
    flatten_fn_exact_match_spec: Optional[FlattenFuncExactMatchSpec] = None,
) -> None:
    SUPPORTED_NODES[cls] = flatten_fn_spec
    SUPPORTED_NODES_EXACT_MATCH[cls] = flatten_fn_exact_match_spec


def _deregister_pytree_flatten_spec(
    cls: type[Any],
) -> None:
    del SUPPORTED_NODES[cls]
    del SUPPORTED_NODES_EXACT_MATCH[cls]


def tree_flatten_spec(
    pytree: PyTree,
    spec: TreeSpec,
    exact_structural_match: bool = False,
) -> list[Any]:
    if spec.is_leaf():
        return [pytree]
    if spec.type not in SUPPORTED_NODES:
        raise RuntimeError(
            f"{type(pytree)} does not have a flatten_fn_spec associated with it. Please register one with "
            "torch.fx._pytree.register_pytree_flatten_spec.  If you have serialized your model, make "
            "sure that any custom pytrees have been registered before loading it.",
        )
    flatten_fn_spec = SUPPORTED_NODES[spec.type]
    child_pytrees = flatten_fn_spec(pytree, spec)
    if exact_structural_match:
        flatten_fn_exact_match_spec = SUPPORTED_NODES_EXACT_MATCH[spec.type]
        if flatten_fn_exact_match_spec and not flatten_fn_exact_match_spec(
            pytree,
            spec,
        ):
            raise RuntimeError(f"Cannot flatten pytree {pytree}, given spec: {spec}")
    result = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = tree_flatten_spec(child, child_spec, exact_structural_match)
        result += flat
    return result


def _dict_flatten_spec(d: dict[_K, _V], spec: TreeSpec) -> list[_V]:
    return [d[k] for k in spec.context]


def _list_flatten_spec(d: list[_T], spec: TreeSpec) -> list[_T]:
    return [d[i] for i in range(spec.num_children)]


def _tuple_flatten_spec(d: tuple[_T, ...], spec: TreeSpec) -> list[_T]:
    return [d[i] for i in range(spec.num_children)]


def _namedtuple_flatten_spec(d: NamedTuple, spec: TreeSpec) -> list[Any]:
    return [d[i] for i in range(spec.num_children)]


def _dict_flatten_spec_exact_match(d: dict[_K, _V], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _list_flatten_spec_exact_match(d: list[_T], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _tuple_flatten_spec_exact_match(d: tuple[_T, ...], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _namedtuple_flatten_spec_exact_match(d: NamedTuple, spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


register_pytree_flatten_spec(dict, _dict_flatten_spec, _dict_flatten_spec_exact_match)
register_pytree_flatten_spec(list, _list_flatten_spec, _list_flatten_spec_exact_match)
register_pytree_flatten_spec(
    tuple,
    _tuple_flatten_spec,
    _tuple_flatten_spec_exact_match,
)
for return_type in torch.return_types.all_return_types:
    register_pytree_flatten_spec(
        return_type,
        _tuple_flatten_spec,
        _tuple_flatten_spec_exact_match,
    )
register_pytree_flatten_spec(
    namedtuple,  # type: ignore[arg-type]
    _namedtuple_flatten_spec,
    _namedtuple_flatten_spec_exact_match,
)
