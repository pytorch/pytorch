from collections import namedtuple
from collections.abc import Callable
from typing import Any, Optional, TypeVar
from typing_extensions import NamedTuple

import torch.return_types
from torch.utils._pytree import PyTree, tree_flatten, TreeSpec


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
) -> list[Any]:
    if spec.is_leaf():
        return [pytree]
    # I guess these exist for BC, FC reasons.
    # In general, we should be able to directly
    # use pytree tree flattener to flatten them,
    # as export serializes the pytree separately.
    # Will remove it in follow up PR.
    if spec.type in SUPPORTED_NODES:
        flatten_fn_spec = SUPPORTED_NODES[spec.type]
        child_pytrees = flatten_fn_spec(pytree, spec)
        result = []
        for child, child_spec in zip(child_pytrees, spec.children()):
            flat = tree_flatten_spec(child, child_spec)
            result += flat
        return result
    flat_result, real_spec = tree_flatten(pytree)
    if spec != real_spec:
        raise RuntimeError(
            f"Real spec {real_spec} of object {pytree} is different from expected spec {spec}. "
            f"Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml"
        )
    return flat_result


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
