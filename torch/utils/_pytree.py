from typing import NamedTuple, Callable, Any, Tuple, List, Dict, Type, cast, Optional, TypeVar, overload, Union
from collections import namedtuple, OrderedDict
import dataclasses
import json
import warnings


T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
R = TypeVar('R')

"""
Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_unflatten` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.

This pytree implementation is not very performant due to Python overhead
To improve the performance we can move parts of the implementation to C++.
"""

DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL = 1

Context = Any
PyTree = Any
FlattenFunc = Callable[[PyTree], Tuple[List, Context]]
UnflattenFunc = Callable[[List, Context], PyTree]
DumpableContext = Any  # Any json dumpable text
ToDumpableContextFn = Callable[[Context], DumpableContext]
FromDumpableContextFn = Callable[[DumpableContext], Context]
ToStrFunc = Callable[["TreeSpec", List[str]], str]
MaybeFromStrFunc = Callable[[str], Optional[Tuple[Any, Context, str]]]

# A NodeDef holds two callables:
# - flatten_fn should take the collection and return a flat list of values.
#   It can also return some context that is used in reconstructing the
#   collection.
# - unflatten_fn should take a flat list of values and some context
#   (returned by flatten_fn). It returns the collection by reconstructing
#   it from the list and the context.
class NodeDef(NamedTuple):
    type: Type[Any]
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc

SUPPORTED_NODES: Dict[Type[Any], NodeDef] = {}

# _SerializeNodeDef holds the following:
# - typ: the type of the node (e.g., "Dict", "List", etc)
# - type_fqn: the fully qualified name of the type, e.g. "collections.OrderedDict"
# - to_dumpable_context takes a TreeSpec, and returns a serialized string format of the
#   context, and the version number
# - from_dumpable_context takes in a string representation of the context, and the
#   version, and returns the deserialized context
class _SerializeNodeDef(NamedTuple):
    typ: Type[Any]
    type_fqn: str
    to_dumpable_context: Optional[ToDumpableContextFn]
    from_dumpable_context: Optional[FromDumpableContextFn]

SUPPORTED_SERIALIZED_TYPES: Dict[Type[Any], _SerializeNodeDef] = {}
SERIALIZED_TYPE_TO_PYTHON_TYPE: Dict[str, Type[Any]] = {}

def _register_pytree_node(
    typ: Any,
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    to_str_fn: Optional[ToStrFunc] = None,
    maybe_from_str_fn: Optional[MaybeFromStrFunc] = None,
    *,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
) -> None:
    """
    Args:
        typ: the type to register
        flatten_fn: A callable that takes a pytree and returns a flattened
            representation of the pytree and additional context to represent the
            flattened pytree.
        unflatten_fn: A callable that takes a flattened version of the pytree,
            additional context, and returns an unflattedn pytree.
        to_dumpable_context: An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable
            representation. This is used for json serialization, which is being
            used in torch.export right now.
        from_dumpable_context: An optional keyword argument to custom specify how
            to convert the custom json dumpable representation of the context
            back to the original context. This is used for json deserialization,
            which is being used in torch.export right now.
    """
    if to_str_fn is not None or maybe_from_str_fn is not None:
        warnings.warn(
            "to_str_fn and maybe_from_str_fn is deprecated. "
            "Please use to_dumpable_context and from_dumpable_context instead."
        )

    node_def = NodeDef(
        typ,
        flatten_fn,
        unflatten_fn,
    )
    SUPPORTED_NODES[typ] = node_def

    if (to_dumpable_context is None) ^ (from_dumpable_context is None):
        raise ValueError(
            f"Both to_dumpable_context and from_dumpable_context for {typ} must "
            "be None or registered."
        )

    type_fqn = f"{typ.__module__}.{typ.__name__}"
    serialize_node_def = _SerializeNodeDef(
        typ, type_fqn, to_dumpable_context, from_dumpable_context
    )
    SUPPORTED_SERIALIZED_TYPES[typ] = serialize_node_def
    SERIALIZED_TYPE_TO_PYTHON_TYPE[type_fqn] = typ


def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())

def _dict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return dict(zip(context, values))

def _list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return d, None

def _list_unflatten(values: List[Any], context: Context) -> List[Any]:
    return list(values)

def _tuple_flatten(d: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    return list(d), None

def _tuple_unflatten(values: List[Any], context: Context) -> Tuple[Any, ...]:
    return tuple(values)

def _namedtuple_flatten(d: NamedTuple) -> Tuple[List[Any], Context]:
    return list(d), type(d)

def _namedtuple_unflatten(values: List[Any], context: Context) -> NamedTuple:
    return cast(NamedTuple, context(*values))

def _namedtuple_serialize(context: Context) -> DumpableContext:
    json_namedtuple = {
        "class_name": context.__name__,
        "fields": context._fields,
    }
    return json_namedtuple

def _namedtuple_deserialize(dumpable_context: DumpableContext) -> Context:
    class_name = dumpable_context["class_name"]
    assert isinstance(class_name, str)
    context = namedtuple(class_name, dumpable_context["fields"])  # type: ignore[misc]
    return context

def _odict_flatten(d: 'OrderedDict[Any, Any]') -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())

def _odict_unflatten(values: List[Any], context: Context) -> 'OrderedDict[Any, Any]':
    return OrderedDict((key, value) for key, value in zip(context, values))


_register_pytree_node(dict, _dict_flatten, _dict_unflatten)
_register_pytree_node(list, _list_flatten, _list_unflatten)
_register_pytree_node(tuple, _tuple_flatten, _tuple_unflatten)
_register_pytree_node(
    namedtuple,
    _namedtuple_flatten,
    _namedtuple_unflatten,
    to_dumpable_context=_namedtuple_serialize,
    from_dumpable_context=_namedtuple_deserialize,
)
_register_pytree_node(OrderedDict, _odict_flatten, _odict_unflatten)


# h/t https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
def _is_namedtuple_instance(pytree: Any) -> bool:
    typ = type(pytree)
    bases = typ.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(typ, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(entry) == str for entry in fields)

def _get_node_type(pytree: Any) -> Any:
    if _is_namedtuple_instance(pytree):
        return namedtuple
    return type(pytree)

# A leaf is defined as anything that is not a Node.
def _is_leaf(pytree: PyTree) -> bool:
    return _get_node_type(pytree) not in SUPPORTED_NODES


# A TreeSpec represents the structure of a pytree. It holds:
# "type": the type of root Node of the pytree
# context: some context that is useful in unflattening the pytree
# children_specs: specs for each child of the root Node
# num_leaves: the number of leaves
@dataclasses.dataclass
class TreeSpec:
    type: Any
    context: Context
    children_specs: List['TreeSpec']

    def __post_init__(self) -> None:
        self.num_leaves: int = sum([spec.num_leaves for spec in self.children_specs])

    def __repr__(self, indent: int = 0) -> str:
        repr_prefix: str = f'TreeSpec({self.type.__name__}, {self.context}, ['
        children_specs_str: str = ''
        if len(self.children_specs):
            indent += 2
            children_specs_str += self.children_specs[0].__repr__(indent)
            children_specs_str += ',' if len(self.children_specs) > 1 else ''
            children_specs_str += ','.join(['\n' + ' ' * indent + child.__repr__(indent) for child in self.children_specs[1:]])
        repr_suffix: str = f'{children_specs_str}])'
        return repr_prefix + repr_suffix


class LeafSpec(TreeSpec):
    def __init__(self) -> None:
        super().__init__(None, None, [])
        self.num_leaves = 1

    def __repr__(self, indent: int = 0) -> str:
        return '*'

def tree_flatten(pytree: PyTree) -> Tuple[List[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree.
    """
    if _is_leaf(pytree):
        return [pytree], LeafSpec()

    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)

    # Recursively flatten the children
    result : List[Any] = []
    children_specs : List[TreeSpec] = []
    for child in child_pytrees:
        flat, child_spec = tree_flatten(child)
        result += flat
        children_specs.append(child_spec)

    return result, TreeSpec(node_type, context, children_specs)


def tree_unflatten(values: List[Any], spec: TreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
    if not isinstance(spec, TreeSpec):
        raise ValueError(
            f'tree_unflatten(values, spec): Expected `spec` to be instance of '
            f'TreeSpec but got item of type {type(spec)}.')
    if len(values) != spec.num_leaves:
        raise ValueError(
            f'tree_unflatten(values, spec): `values` has length {len(values)} '
            f'but the spec refers to a pytree that holds {spec.num_leaves} '
            f'items ({spec}).')
    if isinstance(spec, LeafSpec):
        return values[0]

    unflatten_fn = SUPPORTED_NODES[spec.type].unflatten_fn

    # Recursively unflatten the children
    start = 0
    end = 0
    child_pytrees = []
    for child_spec in spec.children_specs:
        end += child_spec.num_leaves
        child_pytrees.append(tree_unflatten(values[start:end], child_spec))
        start = end

    return unflatten_fn(child_pytrees, spec.context)

def tree_map(fn: Any, pytree: PyTree) -> PyTree:
    flat_args, spec = tree_flatten(pytree)
    return tree_unflatten([fn(i) for i in flat_args], spec)

Type2 = Tuple[Type[T], Type[S]]
Type3 = Tuple[Type[T], Type[S], Type[U]]
TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]

Fn3 = Callable[[Union[T, S, U]], R]
Fn2 = Callable[[Union[T, S]], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]

# These specializations help with type inference on the lambda passed to this
# function
@overload
def map_only(ty: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
    ...

@overload
def map_only(ty: Type[T]) -> MapOnlyFn[Fn[T, Any]]:
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
        def inner(x: T) -> Any:
            if isinstance(x, ty):
                return f(x)
            else:
                return x
        return inner
    return deco

@overload
def tree_map_only(ty: Type[T], fn: Fn[T, Any], pytree: PyTree) -> PyTree:
    ...

@overload
def tree_map_only(ty: Type2[T, S], fn: Fn2[T, S, Any], pytree: PyTree) -> PyTree:
    ...

@overload
def tree_map_only(ty: Type3[T, S, U], fn: Fn3[T, S, U, Any], pytree: PyTree) -> PyTree:
    ...

def tree_map_only(ty: TypeAny, fn: FnAny[Any], pytree: PyTree) -> PyTree:
    return tree_map(map_only(ty)(fn), pytree)

def tree_all(pred: Callable[[Any], bool], pytree: PyTree) -> bool:
    flat_args, _ = tree_flatten(pytree)
    return all(map(pred, flat_args))

def tree_any(pred: Callable[[Any], bool], pytree: PyTree) -> bool:
    flat_args, _ = tree_flatten(pytree)
    return any(map(pred, flat_args))

@overload
def tree_all_only(ty: Type[T], pred: Fn[T, bool], pytree: PyTree) -> bool:
    ...

@overload
def tree_all_only(ty: Type2[T, S], pred: Fn2[T, S, bool], pytree: PyTree) -> bool:
    ...

@overload
def tree_all_only(ty: Type3[T, S, U], pred: Fn3[T, S, U, bool], pytree: PyTree) -> bool:
    ...

def tree_all_only(ty: TypeAny, pred: FnAny[bool], pytree: PyTree) -> bool:
    flat_args, _ = tree_flatten(pytree)
    return all(pred(x) for x in flat_args if isinstance(x, ty))

@overload
def tree_any_only(ty: Type[T], pred: Fn[T, bool], pytree: PyTree) -> bool:
    ...

@overload
def tree_any_only(ty: Type2[T, S], pred: Fn2[T, S, bool], pytree: PyTree) -> bool:
    ...

def tree_any_only(ty: TypeAny, pred: FnAny[bool], pytree: PyTree) -> bool:
    flat_args, _ = tree_flatten(pytree)
    return any(pred(x) for x in flat_args if isinstance(x, ty))

# Broadcasts a pytree to the provided TreeSpec and returns the flattened
# values. If this is not possible, then this function returns None.
#
# For example, given pytree=0 and spec=TreeSpec(list, None, [LeafSpec(), LeafSpec()]),
# would return [0, 0]. This is useful for part of the vmap implementation:
# a user can pass in vmap(fn, in_dims)(*inputs). `in_dims` should be
# broadcastable to the tree structure of `inputs` and we use
# _broadcast_to_and_flatten to check this.
def _broadcast_to_and_flatten(pytree: PyTree, spec: TreeSpec) -> Optional[List[Any]]:
    assert isinstance(spec, TreeSpec)

    if _is_leaf(pytree):
        return [pytree] * spec.num_leaves
    if isinstance(spec, LeafSpec):
        return None
    node_type = _get_node_type(pytree)
    if node_type != spec.type:
        return None

    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, ctx = flatten_fn(pytree)

    # Check if the Node is different from the spec
    if len(child_pytrees) != len(spec.children_specs) or ctx != spec.context:
        return None

    # Recursively flatten the children
    result : List[Any] = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = _broadcast_to_and_flatten(child, child_spec)
        if flat is not None:
            result += flat
        else:
            return None

    return result


"""
_TreeSpecSchema is the schema used to serialize the TreeSpec
It contains the following fields:
- type: A string name of the type. null for the case of a LeafSpec.
- context: Any format which is json dumpable
- children_spec: A list of children serialized specs.
"""
@dataclasses.dataclass
class _TreeSpecSchema:
    type: Optional[str]
    context: DumpableContext
    children_spec: List['_TreeSpecSchema']

class _ProtocolFn(NamedTuple):
    treespec_to_json: Callable[[TreeSpec], DumpableContext]
    json_to_treespec: Callable[[DumpableContext], TreeSpec]

_SUPPORTED_PROTOCOLS: Dict[int, _ProtocolFn] = {}


def _treespec_to_json(spec: TreeSpec) -> _TreeSpecSchema:
    if isinstance(spec, LeafSpec):
        return _TreeSpecSchema(None, None, [])

    if spec.type not in SUPPORTED_SERIALIZED_TYPES:
        raise NotImplementedError(f"Serializing {spec.type} in pytree is not registered.")

    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[spec.type]

    type_fqn = serialize_node_def.type_fqn

    if serialize_node_def.to_dumpable_context is None:
        try:
            serialized_context = json.dumps(spec.context)
        except TypeError as e:
            raise TypeError(
                "Unable to serialize context. "
                "Please make the context json dump-able, or register a "
                "custom serializer using _register_pytree_node."
            ) from e
    else:
        serialized_context = serialize_node_def.to_dumpable_context(spec.context)

    child_schemas = [_treespec_to_json(child) for child in spec.children_specs]

    return _TreeSpecSchema(type_fqn, serialized_context, child_schemas)

def _json_to_treespec(json_schema: DumpableContext) -> TreeSpec:
    if (
        json_schema["type"] is None and
        json_schema["context"] is None and
        len(json_schema["children_spec"]) == 0
    ):
        return LeafSpec()

    if json_schema["type"] not in SERIALIZED_TYPE_TO_PYTHON_TYPE:
        raise NotImplementedError(f'Deserializing {json_schema["type"]} in pytree is not registered.')

    typ = SERIALIZED_TYPE_TO_PYTHON_TYPE[json_schema["type"]]
    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[typ]

    if serialize_node_def.from_dumpable_context is None:
        try:
            context = json.loads(json_schema["context"])
        except TypeError:
            raise TypeError(
                "Unable to deserialize context. "
                "Please make the context json load-able, or register a "
                "custom serializer using _register_pytree_node."
            )
    else:
        context = serialize_node_def.from_dumpable_context(json_schema["context"])

    children_spec = []
    for child_string in json_schema["children_spec"]:
        children_spec.append(_json_to_treespec(child_string))

    return TreeSpec(typ, context, children_spec)


_SUPPORTED_PROTOCOLS[1] = _ProtocolFn(_treespec_to_json, _json_to_treespec)


def treespec_dumps(treespec: TreeSpec, protocol: Optional[int] = None) -> str:
    if protocol is None:
        protocol = DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL

    if protocol in _SUPPORTED_PROTOCOLS:
        json_spec = _SUPPORTED_PROTOCOLS[protocol].treespec_to_json(treespec)
    else:
        raise ValueError(f"Unknown protocol {protocol}. Available protocols: {list(_SUPPORTED_PROTOCOLS.keys())}")

    str_spec = json.dumps((protocol, dataclasses.asdict(json_spec)))
    return str_spec

def treespec_loads(data: str) -> TreeSpec:
    protocol, json_schema = json.loads(data)

    if protocol in _SUPPORTED_PROTOCOLS:
        return _SUPPORTED_PROTOCOLS[protocol].json_to_treespec(json_schema)
    raise ValueError(f"Unknown protocol {protocol}. Available protocols: {list(_SUPPORTED_PROTOCOLS.keys())}")

# TODO(angelayi): remove this function after OSS/internal stabilize
def pytree_to_str(spec: TreeSpec) -> str:
    warnings.warn("pytree_to_str is deprecated. Please use treespec_dumps")
    return treespec_dumps(spec)

# TODO(angelayi): remove this function after OSS/internal stabilize
def str_to_pytree(json: str) -> TreeSpec:
    warnings.warn("str_to_pytree is deprecated. Please use treespec_loads")
    return treespec_loads(json)
