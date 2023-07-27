from typing import NamedTuple, Callable, Any, Tuple, List, Dict, Type, cast, Optional, TypeVar, overload, Union
import functools
from collections import namedtuple, OrderedDict
import dataclasses
import json


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

@dataclasses.dataclass
class TreeSpecSchema:
    type: Optional[str]
    context: Optional[str]
    children_spec: List['TreeSpecSchema']

# A NodeDef holds two callables:
# - flatten_fn should take the collection and return a flat list of values.
#   It can also return some context that is used in reconstructing the
#   collection.
# - unflatten_fn should take a flat list of values and some context
#   (returned by flatten_fn). It returns the collection by reconstructing
#   it from the list and the context.
# - to_schema_fn takes a TreeSpec with the specific type and a list of its children
#   TreeSpecs already converted to the TreeSpecSchema, and returns a
#   TreeSpecSchema representation of this TreeSpec
# - maybe_from_schema_fn takes in a string representation of a type, a string
#   representation of the context, and if the string type is a supported
#   type, then it returns the python type and deserialized context. Otherwise it
#   returns None.
Context = Any
PyTree = Any
FlattenFunc = Callable[[PyTree], Tuple[List, Context]]
UnflattenFunc = Callable[[List, Context], PyTree]
ToSchemaFunc = Callable[["TreeSpec", List[TreeSpecSchema]], TreeSpecSchema]
MaybeFromSchemaFunc = Callable[[Optional[str], Optional[str]], Optional[Tuple[Any, Context]]]

class NodeDef(NamedTuple):
    type: Type[Any]
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc
    to_schema_fn: ToSchemaFunc
    maybe_from_schema_fn: MaybeFromSchemaFunc

SUPPORTED_NODES: Dict[Type[Any], NodeDef] = {}

def _register_pytree_node(
    typ: Any,
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    to_schema_fn: Optional[ToSchemaFunc] = None,
    maybe_from_schema_fn: Optional[MaybeFromSchemaFunc] = None,
) -> None:
    if to_schema_fn is None:
        def _raise_error(spec: "TreeSpec", child_schemas: List[TreeSpecSchema]) -> TreeSpecSchema:
            raise NotImplementedError(f"Serializing {typ} not implemented")
        to_schema_fn = _raise_error

    if maybe_from_schema_fn is None:
        def dummy(type_: Optional[str], context: Optional[str]) -> Optional[Tuple[Any, Context]]:
            return None
        maybe_from_schema_fn = dummy

    assert to_schema_fn is not None
    assert maybe_from_schema_fn is not None
    node_def = NodeDef(typ, flatten_fn, unflatten_fn, to_schema_fn, maybe_from_schema_fn)
    SUPPORTED_NODES[typ] = node_def

def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())

def _dict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return dict(zip(context, values))

def _dict_to_schema(spec: "TreeSpec", child_strings: List[TreeSpecSchema]) -> TreeSpecSchema:
    assert spec.type == dict
    assert isinstance(spec.context, list)
    serialized_context = json.dumps(spec.context)
    return TreeSpecSchema("dict", serialized_context, child_strings)

def _maybe_schema_to_dict(type_: Optional[str], context_str: Optional[str]) -> Optional[Tuple[Any, Context]]:
    if not type_ == "dict":
        return None
    if context_str is None:
        return None
    context = json.loads(context_str)
    return dict, context

def _list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return d, None

def _list_unflatten(values: List[Any], context: Context) -> List[Any]:
    return list(values)

def _list_to_schema(spec: "TreeSpec", child_strings: List[TreeSpecSchema]) -> TreeSpecSchema:
    assert spec.type == list
    return TreeSpecSchema("list", None, child_strings)

def _maybe_schema_to_list(type_: Optional[str], context_str: Optional[str]) -> Optional[Tuple[Any, Context]]:
    if not type_ == "list":
        return None
    assert context_str is None
    return list, None

def _tuple_flatten(d: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    return list(d), None

def _tuple_unflatten(values: List[Any], context: Context) -> Tuple[Any, ...]:
    return tuple(values)

def _tuple_to_schema(spec: "TreeSpec", child_strings: List[TreeSpecSchema]) -> TreeSpecSchema:
    assert spec.type == tuple
    return TreeSpecSchema("tuple", None, child_strings)

def _maybe_schema_to_tuple(type_: Optional[str], context_str: Optional[str]) -> Optional[Tuple[Any, Context]]:
    if not type_ == "tuple":
        return None
    assert context_str is None
    return tuple, None

def _namedtuple_flatten(d: NamedTuple) -> Tuple[List[Any], Context]:
    return list(d), type(d)

def _namedtuple_unflatten(values: List[Any], context: Context) -> NamedTuple:
    return cast(NamedTuple, context(*values))

def _namedtuple_to_schema(spec: "TreeSpec", child_strings: List[TreeSpecSchema]) -> TreeSpecSchema:
    assert spec.type == namedtuple

    json_namedtuple = {
        "class_name": spec.context.__name__,
        "fields": spec.context._fields,
    }
    serialized_context = json.dumps(json_namedtuple)
    return TreeSpecSchema("namedtuple", serialized_context, child_strings)

def _maybe_schema_to_namedtuple(type_: Optional[str], context_str: Optional[str]) -> Optional[Tuple[Any, Context]]:
    if not type_ == "namedtuple":
        return None
    if context_str is None:
        return None
    json_context = json.loads(context_str)
    class_name = json_context["class_name"]
    assert isinstance(class_name, str)
    context = namedtuple(class_name, json_context["fields"])  # type: ignore[misc]
    return OrderedDict, context

def _odict_flatten(d: 'OrderedDict[Any, Any]') -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())

def _odict_unflatten(values: List[Any], context: Context) -> 'OrderedDict[Any, Any]':
    return OrderedDict((key, value) for key, value in zip(context, values))

def _odict_to_schema(spec: "TreeSpec", child_strings: List[TreeSpecSchema]) -> TreeSpecSchema:
    assert spec.type == OrderedDict
    assert isinstance(spec.context, list)
    serialized_context = json.dumps(spec.context)
    return TreeSpecSchema("odict", serialized_context, child_strings)

def _maybe_schema_to_odict(type_: Optional[str], context_str: Optional[str]) -> Optional[Tuple[Any, Context]]:
    if not type_ == "odict":
        return None
    if context_str is None:
        return None
    context = json.loads(context_str)
    return OrderedDict, context


_register_pytree_node(dict, _dict_flatten, _dict_unflatten, _dict_to_schema, _maybe_schema_to_dict)
_register_pytree_node(list, _list_flatten, _list_unflatten, _list_to_schema, _maybe_schema_to_list)
_register_pytree_node(tuple, _tuple_flatten, _tuple_unflatten, _tuple_to_schema, _maybe_schema_to_tuple)
_register_pytree_node(namedtuple, _namedtuple_flatten, _namedtuple_unflatten, _namedtuple_to_schema, _maybe_schema_to_namedtuple)
_register_pytree_node(OrderedDict, _odict_flatten, _odict_unflatten, _odict_to_schema, _maybe_schema_to_odict)


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
        @functools.wraps(f)
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


def pytree_to_str(spec: TreeSpec) -> str:
    def pytree_to_json(spec: TreeSpec) -> TreeSpecSchema:
        if isinstance(spec, LeafSpec):
            return TreeSpecSchema(None, None, [])
        elif spec.type in SUPPORTED_NODES:
            child_schemas = [pytree_to_json(child) for child in spec.children_specs]
            return SUPPORTED_NODES[spec.type].to_schema_fn(spec, child_schemas)
        else:
            raise NotImplementedError(f"Serializing {spec.type} in pytree not supported yet")

    json_spec = pytree_to_json(spec)
    str_spec = json.dumps(dataclasses.asdict(json_spec))
    return str_spec

def str_to_pytree(str_json_spec: str) -> TreeSpec:
    json_schema = json.loads(str_json_spec)

    def json_to_pytree(json_schema) -> TreeSpec:  # type: ignore[no-untyped-def]
        if (
            json_schema["type"] is None and
            json_schema["context"] is None and
            len(json_schema["children_spec"]) == 0
        ):
            return LeafSpec()

        for node_def in SUPPORTED_NODES.values():
            res = node_def.maybe_from_schema_fn(json_schema["type"], json_schema["context"])
            if res is not None:
                type_, context = res
                children_spec = []
                for child_string in json_schema["children_spec"]:
                    children_spec.append(json_to_pytree(child_string))
                return TreeSpec(type_, context, children_spec)
        raise NotImplementedError(f'Deserializing {json_schema["type"]} in pytree not supported yet')

    return json_to_pytree(json_schema)


def _split_nested(string: str) -> List[str]:
    nested_parentheses = 0
    splits = []
    start_index = 0

    for i, char in enumerate(string):
        if char == "(":
            nested_parentheses += 1
        elif char == ")":
            nested_parentheses -= 1

        if nested_parentheses == 0 and char == ",":
            splits.append(string[start_index:i])
            start_index = i + 1

    splits.append(string[start_index:])
    return splits


def _parse_dict_children_spec(toplevel_str: str) -> Tuple[List[str], List[TreeSpec]]:
    assert toplevel_str[1] == "("
    assert toplevel_str[-1] == ")"
    children_string = toplevel_str[2:-1]

    child_strings = []
    context_strings = []
    nested_parentheses = 0
    start_index = 0
    for i, char in enumerate(children_string):
        if char == ":":
            if nested_parentheses == 0:
                context_strings.append(children_string[start_index:i])
                start_index = i + 1
        elif char == "(":
            nested_parentheses += 1
        elif char == ")":
            nested_parentheses -= 1

        if nested_parentheses == 0 and char == ",":
            child_strings.append(children_string[start_index:i])
            start_index = i + 1

    child_strings.append(children_string[start_index:])
    children = [str_to_pytree(child_string) for child_string in child_strings]
    return context_strings, children
