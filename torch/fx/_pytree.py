from collections import namedtuple
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type, Optional

from torch.utils._pytree import LeafSpec, PyTree, TreeSpec

FlattenFuncSpec = Callable[[PyTree, TreeSpec], List]

FlattenFuncExactMatchSpec = Callable[[PyTree, TreeSpec], bool]

SUPPORTED_NODES: Dict[Type[Any], FlattenFuncSpec] = {}

SUPPORTED_NODES_EXACT_MATCH: Dict[Type[Any], Optional[FlattenFuncExactMatchSpec]] = {}

def register_pytree_flatten_spec(
    cls: Type[Any],
    flatten_fn_spec: FlattenFuncSpec,
    flatten_fn_exact_match_spec: Optional[FlattenFuncExactMatchSpec] = None
) -> None:
    SUPPORTED_NODES[cls] = flatten_fn_spec
    SUPPORTED_NODES_EXACT_MATCH[cls] = flatten_fn_exact_match_spec

def tree_flatten_spec(pytree: PyTree, spec: TreeSpec, exact_structural_match=False) -> List[Any]:
    if isinstance(spec, LeafSpec):
        return [pytree]
    if spec.type not in SUPPORTED_NODES:
        raise RuntimeError(
            f"{type(pytree)} does not have a flatten_fn_spec associated with it. Please register one with "
            "torch.fx._pytree.register_pytree_flatten_spec.  If you have serialized your model, make "
            "sure that any custom pytrees have been registered before loading it.")
    flatten_fn_spec = SUPPORTED_NODES[spec.type]
    child_pytrees = flatten_fn_spec(pytree, spec)
    if exact_structural_match:
        flatten_fn_exact_match_spec = SUPPORTED_NODES_EXACT_MATCH[spec.type]
        if flatten_fn_exact_match_spec and not flatten_fn_exact_match_spec(pytree, spec):
            raise RuntimeError(f"Cannot flatten pytree {pytree}, given spec: {spec}")
    result = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = tree_flatten_spec(child, child_spec, exact_structural_match)
        result += flat
    return result

def _dict_flatten_spec(d: Dict[Any, Any], spec: TreeSpec) -> List[Any]:
    return [d[k] for k in spec.context]

def _list_flatten_spec(d: List[Any], spec: TreeSpec) -> List[Any]:
    return [d[i] for i in range(len(spec.children_specs))]

def _tuple_flatten_spec(d: Tuple[Any], spec: TreeSpec) -> List[Any]:
    return [d[i] for i in range(len(spec.children_specs))]

def _namedtuple_flatten_spec(d: NamedTuple, spec: TreeSpec) -> List[Any]:
    return [d[i] for i in range(len(spec.children_specs))]

def _dict_flatten_spec_exact_match(d: Dict[Any, Any], spec: TreeSpec) -> bool:
    return len(d) == len(spec.context)

def _list_flatten_spec_exact_match(d: List[Any], spec: TreeSpec) -> bool:
    return len(d) == len(spec.children_specs)

def _tuple_flatten_spec_exact_match(d: Tuple[Any], spec: TreeSpec) -> bool:
    return len(d) == len(spec.children_specs)

def _namedtuple_flatten_spec_exact_match(d: NamedTuple, spec: TreeSpec) -> bool:
    return len(d) == len(spec.children_specs)

register_pytree_flatten_spec(dict, _dict_flatten_spec, _dict_flatten_spec_exact_match)
register_pytree_flatten_spec(list, _list_flatten_spec, _list_flatten_spec_exact_match)
register_pytree_flatten_spec(tuple, _tuple_flatten_spec, _tuple_flatten_spec_exact_match)
register_pytree_flatten_spec(namedtuple, _namedtuple_flatten_spec, _tuple_flatten_spec_exact_match)  # type: ignore[arg-type]
