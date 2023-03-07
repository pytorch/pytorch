from typing import Callable, Any, Tuple, List, Dict, Type, NamedTuple
from torch.utils._pytree import PyTree, TreeSpec, LeafSpec
from collections import namedtuple

FlattenFuncSpec = Callable[[PyTree, TreeSpec], List]

SUPPORTED_NODES: Dict[Type[Any], Any] = {}
def register_pytree_flatten_spec(typ: Any, flatten_fn_spec: FlattenFuncSpec) -> None:
    SUPPORTED_NODES[typ] = flatten_fn_spec

def tree_flatten_spec(pytree: PyTree, spec: TreeSpec) -> List[Any]:
    if isinstance(spec, LeafSpec):
        return [pytree]
    if spec.type not in SUPPORTED_NODES:
        raise RuntimeError(
            f"{type(pytree)} does not have a flatten_fn_spec associated with it. Please register one with"
            "torch.fx._pytree.register_pytree_flatten_spec.  If you have serialized your model, make"
            "sure that any custom pytrees have been registered before loading it.")
    flatten_fn_spec = SUPPORTED_NODES[spec.type]
    child_pytrees = flatten_fn_spec(pytree, spec)
    result = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = tree_flatten_spec(child, child_spec)
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

register_pytree_flatten_spec(dict, _dict_flatten_spec)
register_pytree_flatten_spec(list, _list_flatten_spec)
register_pytree_flatten_spec(tuple, _tuple_flatten_spec)
register_pytree_flatten_spec(namedtuple, _tuple_flatten_spec)
