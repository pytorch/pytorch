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

This pytree implementation is not very performant due to Python overhead
To improve the performance we can move parts of the implementation to C++.
"""

import dataclasses
import functools
import importlib
import json
import sys
import threading
import types
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
    Any,
    Callable,
    cast,
    DefaultDict,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Hashable,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    OrderedDict as GenericOrderedDict,
    overload,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import deprecated


__all__ = [
    "PyTree",
    "Context",
    "FlattenFunc",
    "UnflattenFunc",
    "DumpableContext",
    "ToDumpableContextFn",
    "FromDumpableContextFn",
    "TreeSpec",
    "LeafSpec",
    "keystr",
    "key_get",
    "register_pytree_node",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_unflatten",
    "tree_iter",
    "tree_leaves",
    "tree_leaves_with_path",
    "tree_structure",
    "tree_map",
    "tree_map_with_path",
    "tree_map_",
    "tree_map_only",
    "tree_map_only_",
    "tree_all",
    "tree_any",
    "tree_all_only",
    "tree_any_only",
    "treespec_dumps",
    "treespec_loads",
    "treespec_pprint",
]


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")


DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL = 1
NO_SERIALIZED_TYPE_NAME_FOUND = "NO_SERIALIZED_TYPE_NAME_FOUND"


class KeyEntry(Protocol):
    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def get(self, parent: Any) -> Any:
        ...


Context = Any
PyTree = Any
FlattenFunc = Callable[[PyTree], Tuple[List[Any], Context]]
UnflattenFunc = Callable[[Iterable[Any], Context], PyTree]
DumpableContext = Any  # Any json dumpable text
ToDumpableContextFn = Callable[[Context], DumpableContext]
FromDumpableContextFn = Callable[[DumpableContext], Context]
ToStrFunc = Callable[["TreeSpec", List[str]], str]
MaybeFromStrFunc = Callable[[str], Optional[Tuple[Any, Context, str]]]
KeyPath = Tuple[KeyEntry, ...]
FlattenWithKeysFunc = Callable[[PyTree], Tuple[List[Tuple[KeyEntry, Any]], Any]]


# A NodeDef holds two callables:
# - flatten_fn should take the collection and return a flat list of values.
#   It can also return some context that is used in reconstructing the
#   collection.
# - unflatten_fn should take a flat list of values and some context
#   (returned by flatten_fn). It returns the collection by reconstructing
#   it from the list and the context.
# - flatten_with_keys_fn, which is a callable that takes a
#   pytree and returns a list of (keypath, value) pairs and a context.
class NodeDef(NamedTuple):
    type: Type[Any]
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc
    flatten_with_keys_fn: Optional[FlattenWithKeysFunc]


_NODE_REGISTRY_LOCK = threading.Lock()
SUPPORTED_NODES: Dict[Type[Any], NodeDef] = {}


# _SerializeNodeDef holds the following:
# - typ: the type of the node (e.g., "Dict", "List", etc)
# - serialized_type_name: the fully qualified name of the type, e.g. "collections.OrderedDict"
# - to_dumpable_context takes a TreeSpec, and returns a serialized string format of the
#   context, and the version number
# - from_dumpable_context takes in a string representation of the context, and the
#   version, and returns the deserialized context
class _SerializeNodeDef(NamedTuple):
    typ: Type[Any]
    serialized_type_name: str
    to_dumpable_context: Optional[ToDumpableContextFn]
    from_dumpable_context: Optional[FromDumpableContextFn]


SUPPORTED_SERIALIZED_TYPES: Dict[Type[Any], _SerializeNodeDef] = {}
SERIALIZED_TYPE_TO_PYTHON_TYPE: Dict[str, Type[Any]] = {}

# NB: we try really hard to not import _cxx_pytree (which depends on optree)
# as much as possible. This is for isolation: a user who is not using C++ pytree
# shouldn't pay for it, and it helps makes things like cpython upgrades easier.
_cxx_pytree_exists = importlib.util.find_spec("optree")  # type: ignore[attr-defined]
_cxx_pytree_imported = False
_cxx_pytree_pending_imports: List[Any] = []


def register_pytree_node(
    cls: Type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
    flatten_with_keys_fn: Optional[FlattenWithKeysFunc] = None,
) -> None:
    """Register a container-like type as pytree node.

    Args:
        cls: the type to register
        flatten_fn: A callable that takes a pytree and returns a flattened
            representation of the pytree and additional context to represent the
            flattened pytree.
        unflatten_fn: A callable that takes a flattened version of the pytree,
            additional context, and returns an unflattened pytree.
        serialized_type_name: A keyword argument used to specify the fully qualified
            name used when serializing the tree spec.
        to_dumpable_context: An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable
            representation. This is used for json serialization, which is being
            used in torch.export right now.
        from_dumpable_context: An optional keyword argument to custom specify how
            to convert the custom json dumpable representation of the context
            back to the original context. This is used for json deserialization,
            which is being used in torch.export right now.
        flatten_with_keys_fn: An optional keyword argument to specify how to
            access each pytree leaf's keypath when flattening and tree-mapping.
            Like ``flatten_fn``, but in place of a List[leaf], it should return
            a List[(keypath, leaf)].
    """
    with _NODE_REGISTRY_LOCK:
        if cls in SUPPORTED_NODES:
            raise ValueError(f"{cls} is already registered as pytree node.")

    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
        flatten_with_keys_fn=flatten_with_keys_fn,
    )

    if not _cxx_pytree_exists:
        return

    if _cxx_pytree_imported:
        from . import _cxx_pytree as cxx

        cxx._private_register_pytree_node(
            cls,
            flatten_fn,
            unflatten_fn,
            serialized_type_name=serialized_type_name,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context,
        )
    else:
        args = (cls, flatten_fn, unflatten_fn)
        kwargs = {
            "serialized_type_name": serialized_type_name,
            "to_dumpable_context": to_dumpable_context,
            "from_dumpable_context": from_dumpable_context,
        }
        _cxx_pytree_pending_imports.append((args, kwargs))


def _register_namedtuple(
    cls: Type[Any],
    *,
    serialized_type_name: str,
) -> None:
    """
    Registers a namedtuple as a valid pytree node. By default namedtuples are
    valid pytree nodes, but they are not serializable. This API provides the
    argument `serialized_type_name` which allows these namedtuples to be
    serialized.

    Args:
        cls: the dataclass type to register
        serialized_type_name: The serialized name for the dataclass. This is
        required if you want to serialize the pytree TreeSpec containing this
        namedtuple.
    """
    _private_register_pytree_node(
        cls,
        _namedtuple_flatten,
        _namedtuple_unflatten,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=_namedtuple_serialize,
        from_dumpable_context=_namedtuple_deserialize,
        flatten_with_keys_fn=_namedtuple_flatten_with_keys,
    )


@deprecated(
    "`torch.utils._pytree._register_pytree_node` is deprecated. "
    "Please use `torch.utils._pytree.register_pytree_node` instead.",
    category=FutureWarning,
)
def _register_pytree_node(
    cls: Type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    to_str_fn: Optional[ToStrFunc] = None,  # deprecated
    maybe_from_str_fn: Optional[MaybeFromStrFunc] = None,  # deprecated
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
    flatten_with_keys_fn: Optional[FlattenWithKeysFunc] = None,
) -> None:
    """Register a container-like type as pytree node for the Python pytree only.

    Args:
        cls: the type to register
        flatten_fn: A callable that takes a pytree and returns a flattened
            representation of the pytree and additional context to represent the
            flattened pytree.
        unflatten_fn: A callable that takes a flattened version of the pytree,
            additional context, and returns an unflattened pytree.
        serialized_type_name: A keyword argument used to specify the fully qualified
            name used when serializing the tree spec.
        to_dumpable_context: An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable
            representation. This is used for json serialization, which is being
            used in torch.export right now.
        from_dumpable_context: An optional keyword argument to custom specify how
            to convert the custom json dumpable representation of the context
            back to the original context. This is used for json deserialization,
            which is being used in torch.export right now.
        flatten_with_keys_fn: An optional keyword argument to specify how to
            access each pytree leaf's keypath when flattening and tree-mapping.
            Like ``flatten_fn``, but in place of a List[leaf], it should return
            a List[(keypath, leaf)].
    """
    if to_str_fn is not None or maybe_from_str_fn is not None:
        warnings.warn(
            "`to_str_fn` and `maybe_from_str_fn` is deprecated. "
            "Please use `to_dumpable_context` and `from_dumpable_context` instead.",
            FutureWarning,
            stacklevel=2,
        )

    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
        flatten_with_keys_fn=flatten_with_keys_fn,
    )


def _private_register_pytree_node(
    cls: Type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
    flatten_with_keys_fn: Optional[FlattenWithKeysFunc] = None,
) -> None:
    """This is an internal function that is used to register a pytree node type
    for the Python pytree only. End-users should use :func:`register_pytree_node`
    instead.
    """
    with _NODE_REGISTRY_LOCK:
        if cls in SUPPORTED_NODES:
            # TODO: change this warning to an error after OSS/internal stabilize
            warnings.warn(
                f"{cls} is already registered as pytree node. "
                "Overwriting the previous registration.",
            )

        node_def = NodeDef(cls, flatten_fn, unflatten_fn, flatten_with_keys_fn)
        SUPPORTED_NODES[cls] = node_def

        if (to_dumpable_context is None) ^ (from_dumpable_context is None):
            raise ValueError(
                f"Both to_dumpable_context and from_dumpable_context for {cls} must "
                "be None or registered."
            )

        if serialized_type_name is None:
            serialized_type_name = NO_SERIALIZED_TYPE_NAME_FOUND

        serialize_node_def = _SerializeNodeDef(
            cls,
            serialized_type_name,
            to_dumpable_context,
            from_dumpable_context,
        )
        SUPPORTED_SERIALIZED_TYPES[cls] = serialize_node_def
        SERIALIZED_TYPE_TO_PYTHON_TYPE[serialized_type_name] = cls


@dataclasses.dataclass(frozen=True)
class SequenceKey(Generic[T]):
    idx: int

    def __str__(self) -> str:
        return f"[{self.idx!r}]"

    def get(self, sequence: Sequence[T]) -> T:
        return sequence[self.idx]


K = TypeVar("K", bound=Hashable)


@dataclasses.dataclass(frozen=True)
class MappingKey(Generic[K, T]):
    key: K

    def __str__(self) -> str:
        return f"[{self.key!r}]"

    def get(self, mapping: Mapping[K, T]) -> T:
        return mapping[self.key]


@dataclasses.dataclass(frozen=True)
class GetAttrKey:
    name: str

    def __str__(self) -> str:
        return f".{self.name}"

    def get(self, obj: Any) -> Any:
        return getattr(obj, self.name)


def _tuple_flatten(d: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    return list(d), None


def _tuple_flatten_with_keys(
    d: Tuple[Any, ...]
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _tuple_flatten(d)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _tuple_unflatten(values: Iterable[Any], context: Context) -> Tuple[Any, ...]:
    return tuple(values)


def _list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return d, None


def _list_flatten_with_keys(d: List[Any]) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _list_flatten(d)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _list_unflatten(values: Iterable[Any], context: Context) -> List[Any]:
    return list(values)


def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())


def _dict_flatten_with_keys(
    d: Dict[Any, Any]
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _dict_flatten(d)
    return [(MappingKey(k), v) for k, v in zip(context, values)], context


def _dict_unflatten(values: Iterable[Any], context: Context) -> Dict[Any, Any]:
    return dict(zip(context, values))


def _namedtuple_flatten(d: NamedTuple) -> Tuple[List[Any], Context]:
    return list(d), type(d)


def _namedtuple_flatten_with_keys(
    d: NamedTuple,
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _namedtuple_flatten(d)
    return (
        [(GetAttrKey(field), v) for field, v in zip(context._fields, values)],
        context,
    )


def _namedtuple_unflatten(values: Iterable[Any], context: Context) -> NamedTuple:
    return cast(NamedTuple, context(*values))


def _namedtuple_serialize(context: Context) -> DumpableContext:
    if context not in SUPPORTED_SERIALIZED_TYPES:
        raise NotImplementedError(
            f"Can't serialize TreeSpec of namedtuple class {context} because we "
            "didn't register a serializated_type_name. Please register using "
            "`_register_namedtuple`."
        )

    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[context]
    serialized_type_name = serialize_node_def.serialized_type_name

    if serialized_type_name == NO_SERIALIZED_TYPE_NAME_FOUND:
        raise NotImplementedError(
            f"Can't serialize TreeSpec of namedtuple class {context} because we "
            "couldn't find a serializated_type_name. Please register using "
            "`_register_namedtuple`."
        )
    return serialized_type_name


def _namedtuple_deserialize(dumpable_context: DumpableContext) -> Context:
    if dumpable_context not in SERIALIZED_TYPE_TO_PYTHON_TYPE:
        raise NotImplementedError(
            f"Can't deserialize TreeSpec of namedtuple class {dumpable_context} "
            "because we couldn't find a serializated name."
        )

    typ = SERIALIZED_TYPE_TO_PYTHON_TYPE[dumpable_context]
    return typ


def _ordereddict_flatten(d: GenericOrderedDict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())


def _ordereddict_flatten_with_keys(
    d: GenericOrderedDict[Any, Any]
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _ordereddict_flatten(d)
    return [(MappingKey(k), v) for k, v in zip(context, values)], context


def _ordereddict_unflatten(
    values: Iterable[Any],
    context: Context,
) -> GenericOrderedDict[Any, Any]:
    return OrderedDict((key, value) for key, value in zip(context, values))


_odict_flatten = _ordereddict_flatten
_odict_unflatten = _ordereddict_unflatten


def _defaultdict_flatten(d: DefaultDict[Any, Any]) -> Tuple[List[Any], Context]:
    values, dict_context = _dict_flatten(d)
    return values, [d.default_factory, dict_context]


def _defaultdict_flatten_with_keys(
    d: DefaultDict[Any, Any]
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _defaultdict_flatten(d)
    _, dict_context = context
    return [(MappingKey(k), v) for k, v in zip(dict_context, values)], context


def _defaultdict_unflatten(
    values: Iterable[Any],
    context: Context,
) -> DefaultDict[Any, Any]:
    default_factory, dict_context = context
    return defaultdict(default_factory, _dict_unflatten(values, dict_context))


def _defaultdict_serialize(context: Context) -> DumpableContext:
    default_factory, dict_context = context
    json_defaultdict = {
        "default_factory_module": default_factory.__module__,
        "default_factory_name": default_factory.__qualname__,
        "dict_context": dict_context,
    }
    return json_defaultdict


def _defaultdict_deserialize(dumpable_context: DumpableContext) -> Context:
    assert isinstance(dumpable_context, dict)
    assert set(dumpable_context) == {
        "default_factory_module",
        "default_factory_name",
        "dict_context",
    }

    default_factory_module = dumpable_context["default_factory_module"]
    default_factory_name = dumpable_context["default_factory_name"]
    assert isinstance(default_factory_module, str)
    assert isinstance(default_factory_name, str)
    module = importlib.import_module(default_factory_module)
    default_factory = getattr(module, default_factory_name)

    dict_context = dumpable_context["dict_context"]
    return [default_factory, dict_context]


def _deque_flatten(d: Deque[Any]) -> Tuple[List[Any], Context]:
    return list(d), d.maxlen


def _deque_flatten_with_keys(
    d: Deque[Any],
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _deque_flatten(d)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _deque_unflatten(values: Iterable[Any], context: Context) -> Deque[Any]:
    return deque(values, maxlen=context)


_private_register_pytree_node(
    tuple,
    _tuple_flatten,
    _tuple_unflatten,
    serialized_type_name="builtins.tuple",
    flatten_with_keys_fn=_tuple_flatten_with_keys,
)
_private_register_pytree_node(
    list,
    _list_flatten,
    _list_unflatten,
    serialized_type_name="builtins.list",
    flatten_with_keys_fn=_list_flatten_with_keys,
)
_private_register_pytree_node(
    dict,
    _dict_flatten,
    _dict_unflatten,
    serialized_type_name="builtins.dict",
    flatten_with_keys_fn=_dict_flatten_with_keys,
)
_private_register_pytree_node(
    namedtuple,  # type: ignore[arg-type]
    _namedtuple_flatten,
    _namedtuple_unflatten,
    serialized_type_name="collections.namedtuple",
    to_dumpable_context=_namedtuple_serialize,
    from_dumpable_context=_namedtuple_deserialize,
    flatten_with_keys_fn=_namedtuple_flatten_with_keys,
)
_private_register_pytree_node(
    OrderedDict,
    _ordereddict_flatten,
    _ordereddict_unflatten,
    serialized_type_name="collections.OrderedDict",
    flatten_with_keys_fn=_ordereddict_flatten_with_keys,
)
_private_register_pytree_node(
    defaultdict,
    _defaultdict_flatten,
    _defaultdict_unflatten,
    serialized_type_name="collections.defaultdict",
    to_dumpable_context=_defaultdict_serialize,
    from_dumpable_context=_defaultdict_deserialize,
    flatten_with_keys_fn=_defaultdict_flatten_with_keys,
)
_private_register_pytree_node(
    deque,
    _deque_flatten,
    _deque_unflatten,
    serialized_type_name="collections.deque",
    flatten_with_keys_fn=_deque_flatten_with_keys,
)


STANDARD_DICT_TYPES: FrozenSet[type] = frozenset(
    {dict, OrderedDict, defaultdict},
)
BUILTIN_TYPES: FrozenSet[type] = frozenset(
    {tuple, list, dict, namedtuple, OrderedDict, defaultdict, deque},  # type: ignore[arg-type]
)


# h/t https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
def _is_namedtuple_instance(tree: Any) -> bool:
    typ = type(tree)
    bases = typ.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(typ, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(entry) == str for entry in fields)


def _get_node_type(tree: Any) -> Any:
    if _is_namedtuple_instance(tree):
        return namedtuple
    return type(tree)


# A leaf is defined as anything that is not a Node.
def _is_leaf(tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = None) -> bool:
    return (is_leaf is not None and is_leaf(tree)) or _get_node_type(
        tree
    ) not in SUPPORTED_NODES


# A TreeSpec represents the structure of a pytree. It holds:
# "type": the type of root Node of the pytree
# context: some context that is useful in unflattening the pytree
# children_specs: specs for each child of the root Node
# num_leaves: the number of leaves
@dataclasses.dataclass(init=True, frozen=True, eq=True, repr=False)
class TreeSpec:
    type: Any
    context: Context
    children_specs: List["TreeSpec"]

    num_nodes: int = dataclasses.field(init=False)
    num_leaves: int = dataclasses.field(init=False)
    num_children: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        num_nodes = sum((spec.num_nodes for spec in self.children_specs), start=1)
        num_leaves = sum(spec.num_leaves for spec in self.children_specs)
        num_children = len(self.children_specs)
        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "num_leaves", num_leaves)
        object.__setattr__(self, "num_children", num_children)

    def __repr__(self, indent: int = 0) -> str:
        repr_prefix: str = f"TreeSpec({self.type.__name__}, {self.context}, ["
        children_specs_str: str = ""
        if self.num_children > 0:
            indent += 2
            children_specs_str += self.children_specs[0].__repr__(indent)
            children_specs_str += "," if self.num_children > 1 else ""
            children_specs_str += ",".join(
                [
                    "\n" + " " * indent + child.__repr__(indent)
                    for child in self.children_specs[1:]
                ]
            )
        repr_suffix: str = f"{children_specs_str}])"
        return repr_prefix + repr_suffix

    def is_leaf(self) -> bool:
        return self.num_nodes == 1 and self.num_leaves == 1

    def _flatten_up_to_helper(self, tree: PyTree, subtrees: List[PyTree]) -> None:
        if self.is_leaf():
            subtrees.append(tree)
            return

        node_type = _get_node_type(tree)
        if self.type not in BUILTIN_TYPES:
            # Always require custom node types to match exactly
            if node_type != self.type:
                raise ValueError(
                    f"Type mismatch; "
                    f"expected {self.type!r}, but got {node_type!r}.",
                )
            flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
            child_pytrees, context = flatten_fn(tree)
            if len(child_pytrees) != self.num_children:
                raise ValueError(
                    f"Node arity mismatch; "
                    f"expected {self.num_children}, but got {len(child_pytrees)}.",
                )
            if context != self.context:
                raise ValueError(
                    f"Node context mismatch for custom node type {self.type!r}.",
                )
        else:
            # For builtin dictionary types, we allow some flexibility
            # Otherwise, we require exact matches
            both_standard_dict = (
                self.type in STANDARD_DICT_TYPES and node_type in STANDARD_DICT_TYPES
            )
            if node_type != self.type and not both_standard_dict:
                raise ValueError(
                    f"Node type mismatch; "
                    f"expected {self.type!r}, but got {node_type!r}.",
                )
            if len(tree) != self.num_children:
                raise ValueError(
                    f"Node arity mismatch; "
                    f"expected {self.num_children}, but got {len(tree)}.",
                )

            if both_standard_dict:  # dictionary types are compatible with each other
                dict_context = (
                    self.context
                    if self.type is not defaultdict
                    # ignore mismatch of `default_factory` for defaultdict
                    else self.context[1]
                )
                expected_keys = dict_context
                got_key_set = set(tree)
                expected_key_set = set(expected_keys)
                if got_key_set != expected_key_set:
                    missing_keys = expected_key_set.difference(got_key_set)
                    extra_keys = got_key_set.difference(expected_key_set)
                    message = ""
                    if missing_keys:
                        message += f"; missing key(s): {missing_keys}"
                    if extra_keys:
                        message += f"; extra key(s): {extra_keys}"
                    raise ValueError(f"Node keys mismatch{message}.")
                child_pytrees = [tree[key] for key in expected_keys]
            else:
                flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
                child_pytrees, context = flatten_fn(tree)
                if (
                    context != self.context
                    and self.type is not deque  # ignore mismatch of `maxlen` for deque
                ):
                    raise ValueError(
                        f"Node context mismatch for node type {self.type!r}; "
                        f"expected {self.context!r}, but got {context!r}.",  # namedtuple type mismatch
                    )

        for child_pytree, child_spec in zip(child_pytrees, self.children_specs):
            child_spec._flatten_up_to_helper(child_pytree, subtrees)

    def flatten_up_to(self, tree: PyTree) -> List[PyTree]:
        subtrees: List[PyTree] = []
        self._flatten_up_to_helper(tree, subtrees)
        return subtrees

    def unflatten(self, leaves: Iterable[Any]) -> PyTree:
        if not isinstance(leaves, (list, tuple)):
            leaves = list(leaves)
        if len(leaves) != self.num_leaves:
            raise ValueError(
                f"treespec.unflatten(leaves): `leaves` has length {len(leaves)} "
                f"but the spec refers to a pytree that holds {self.num_leaves} "
                f"items ({self}).",
            )
        if self.is_leaf():
            return leaves[0]

        unflatten_fn = SUPPORTED_NODES[self.type].unflatten_fn

        # Recursively unflatten the children
        start = 0
        end = 0
        child_pytrees = []
        for child_spec in self.children_specs:
            end += child_spec.num_leaves
            child_pytrees.append(child_spec.unflatten(leaves[start:end]))
            start = end

        return unflatten_fn(child_pytrees, self.context)


class LeafSpec(TreeSpec):
    def __init__(self) -> None:
        super().__init__(None, None, [])

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_nodes", 1)
        object.__setattr__(self, "num_leaves", 1)
        object.__setattr__(self, "num_children", 0)

    def __repr__(self, indent: int = 0) -> str:
        return "*"


# All leaves are equivalent, so represent with a single object to save on
# object construction time
_LEAF_SPEC = LeafSpec()


def _tree_flatten_helper(
    tree: PyTree,
    leaves: List[Any],
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> TreeSpec:
    if _is_leaf(tree, is_leaf=is_leaf):
        leaves.append(tree)
        return _LEAF_SPEC

    node_type = _get_node_type(tree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(tree)

    # Recursively flatten the children
    children_specs = [
        _tree_flatten_helper(child, leaves, is_leaf=is_leaf) for child in child_pytrees
    ]

    return TreeSpec(node_type, context, children_specs)


def tree_flatten(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Tuple[List[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree.
    """
    leaves: List[Any] = []
    spec = _tree_flatten_helper(tree, leaves, is_leaf=is_leaf)
    return leaves, spec


def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
    if not isinstance(treespec, TreeSpec):
        raise TypeError(
            f"tree_unflatten(leaves, treespec): Expected `treespec` to be "
            f"instance of TreeSpec but got item of type {type(treespec)}.",
        )
    return treespec.unflatten(leaves)


def tree_iter(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Iterable[Any]:
    """Get an iterator over the leaves of a pytree."""
    if _is_leaf(tree, is_leaf=is_leaf):
        yield tree
    else:
        node_type = _get_node_type(tree)
        flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
        child_pytrees, _ = flatten_fn(tree)

        # Recursively flatten the children
        for child in child_pytrees:
            yield from tree_iter(child, is_leaf=is_leaf)


def tree_leaves(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> List[Any]:
    """Get a list of leaves of a pytree."""
    return list(tree_iter(tree, is_leaf=is_leaf))


def tree_structure(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> TreeSpec:
    """Get the TreeSpec for a pytree."""
    return tree_flatten(tree, is_leaf=is_leaf)[1]


def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`.

    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:

    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, *flat_args))


def tree_map_(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`.

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.
    """
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, *flat_args), maxlen=0)  # consume and exhaust the iterable
    return tree


Type2 = Tuple[Type[T], Type[S]]
Type3 = Tuple[Type[T], Type[S], Type[U]]
if sys.version_info >= (3, 10):
    TypeAny = Union[Type[Any], Tuple[Type[Any], ...], types.UnionType]
else:
    TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]

Fn2 = Callable[[Union[T, S]], R]
Fn3 = Callable[[Union[T, S, U]], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]


# These specializations help with type inference on the lambda passed to this
# function
@overload
def map_only(__type_or_types_or_pred: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: Type3[T, S, U]) -> MapOnlyFn[Fn3[T, S, U, Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: Type[T]) -> MapOnlyFn[Fn[T, Any]]:
    ...


# This specialization is needed for the implementations below that call
@overload
def map_only(__type_or_types_or_pred: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: Callable[[Any], bool]) -> MapOnlyFn[FnAny[Any]]:
    ...


def map_only(
    __type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]]
) -> MapOnlyFn[FnAny[Any]]:
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
    if isinstance(__type_or_types_or_pred, (type, tuple)) or (
        sys.version_info >= (3, 10)
        and isinstance(__type_or_types_or_pred, types.UnionType)
    ):

        def pred(x: Any) -> bool:
            return isinstance(x, __type_or_types_or_pred)  # type: ignore[arg-type]

    elif callable(__type_or_types_or_pred):
        pred = __type_or_types_or_pred  # type: ignore[assignment]
    else:
        raise TypeError("Argument must be a type, a tuple of types, or a callable.")

    def wrapper(func: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(func)
        def wrapped(x: T) -> Any:
            if pred(x):
                return func(x)
            return x

        return wrapped

    return wrapper


@overload
def tree_map_only(
    __type_or_types_or_pred: Type[T],
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only(
    __type_or_types_or_pred: Type2[T, S],
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only(
    __type_or_types_or_pred: Type3[T, S, U],
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only(
    __type_or_types_or_pred: Callable[[Any], bool],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


def tree_map_only(
    __type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    return tree_map(map_only(__type_or_types_or_pred)(func), tree, is_leaf=is_leaf)


@overload
def tree_map_only_(
    __type_or_types_or_pred: Type[T],
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only_(
    __type_or_types_or_pred: Type2[T, S],
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only_(
    __type_or_types_or_pred: Type3[T, S, U],
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only_(
    __type_or_types_or_pred: Callable[[Any], bool],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


def tree_map_only_(
    __type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    return tree_map_(map_only(__type_or_types_or_pred)(func), tree, is_leaf=is_leaf)


def tree_all(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(map(pred, flat_args))


def tree_any(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(map(pred, flat_args))


@overload
def tree_all_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


@overload
def tree_all_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


@overload
def tree_all_only(
    __type_or_types: Type3[T, S, U],
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


def tree_all_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(pred(x) for x in flat_args if isinstance(x, __type_or_types))


@overload
def tree_any_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


@overload
def tree_any_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


@overload
def tree_any_only(
    __type_or_types: Type3[T, S, U],
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


def tree_any_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(pred(x) for x in flat_args if isinstance(x, __type_or_types))


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
    treespec: TreeSpec,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Optional[List[Any]]:
    assert isinstance(treespec, TreeSpec)

    if _is_leaf(tree, is_leaf=is_leaf):
        return [tree] * treespec.num_leaves
    if treespec.is_leaf():
        return None
    node_type = _get_node_type(tree)
    if node_type != treespec.type:
        return None

    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, ctx = flatten_fn(tree)

    # Check if the Node is different from the spec
    if len(child_pytrees) != treespec.num_children or ctx != treespec.context:
        return None

    # Recursively flatten the children
    result: List[Any] = []
    for child, child_spec in zip(child_pytrees, treespec.children_specs):
        flat = _broadcast_to_and_flatten(child, child_spec, is_leaf=is_leaf)
        if flat is not None:
            result += flat
        else:
            return None

    return result


@dataclasses.dataclass
class _TreeSpecSchema:
    """
    _TreeSpecSchema is the schema used to serialize the TreeSpec
    It contains the following fields:
    - type: A string name of the type. null for the case of a LeafSpec.
    - context: Any format which is json dumpable
    - children_spec: A list of children serialized specs.
    """

    type: Optional[str]
    context: DumpableContext
    children_spec: List["_TreeSpecSchema"]


class _ProtocolFn(NamedTuple):
    treespec_to_json: Callable[[TreeSpec], DumpableContext]
    json_to_treespec: Callable[[DumpableContext], TreeSpec]


_SUPPORTED_PROTOCOLS: Dict[int, _ProtocolFn] = {}


def _treespec_to_json(treespec: TreeSpec) -> _TreeSpecSchema:
    if treespec.is_leaf():
        return _TreeSpecSchema(None, None, [])

    if treespec.type not in SUPPORTED_SERIALIZED_TYPES:
        raise NotImplementedError(
            f"Serializing {treespec.type} in pytree is not registered.",
        )

    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[treespec.type]

    serialized_type_name = serialize_node_def.serialized_type_name

    if serialized_type_name == NO_SERIALIZED_TYPE_NAME_FOUND:
        raise NotImplementedError(
            f"No registered serialization name for {treespec.type} found. "
            "Please update your _register_pytree_node call with a `serialized_type_name` kwarg."
        )

    if serialize_node_def.to_dumpable_context is None:
        try:
            serialized_context = json.dumps(treespec.context)
        except TypeError as e:
            raise TypeError(
                "Unable to serialize context. "
                "Please make the context json dump-able, or register a "
                "custom serializer using _register_pytree_node."
            ) from e
    else:
        serialized_context = serialize_node_def.to_dumpable_context(treespec.context)

    child_schemas = [_treespec_to_json(child) for child in treespec.children_specs]

    return _TreeSpecSchema(serialized_type_name, serialized_context, child_schemas)


def _json_to_treespec(json_schema: DumpableContext) -> TreeSpec:
    if (
        json_schema["type"] is None
        and json_schema["context"] is None
        and len(json_schema["children_spec"]) == 0
    ):
        return _LEAF_SPEC

    if json_schema["type"] not in SERIALIZED_TYPE_TO_PYTHON_TYPE:
        raise NotImplementedError(
            f'Deserializing {json_schema["type"]} in pytree is not registered.',
        )

    typ = SERIALIZED_TYPE_TO_PYTHON_TYPE[json_schema["type"]]
    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[typ]

    if serialize_node_def.from_dumpable_context is None:
        try:
            context = json.loads(json_schema["context"])
        except TypeError as ex:
            raise TypeError(
                "Unable to deserialize context. "
                "Please make the context json load-able, or register a "
                "custom serializer using _register_pytree_node.",
            ) from ex
    else:
        context = serialize_node_def.from_dumpable_context(json_schema["context"])

    children_specs = []
    for child_string in json_schema["children_spec"]:
        children_specs.append(_json_to_treespec(child_string))

    return TreeSpec(typ, context, children_specs)


_SUPPORTED_PROTOCOLS[1] = _ProtocolFn(_treespec_to_json, _json_to_treespec)


def treespec_dumps(treespec: TreeSpec, protocol: Optional[int] = None) -> str:
    if not isinstance(treespec, TreeSpec):
        raise TypeError(
            f"treespec_dumps(treespec, protocol): Expected `treespec` to be instance of "
            f"TreeSpec but got item of type {type(treespec)}.",
        )

    if protocol is None:
        protocol = DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL

    if protocol in _SUPPORTED_PROTOCOLS:
        json_spec = _SUPPORTED_PROTOCOLS[protocol].treespec_to_json(treespec)
    else:
        raise ValueError(
            f"Unknown protocol {protocol}. "
            f"Available protocols: {list(_SUPPORTED_PROTOCOLS.keys())}",
        )

    str_spec = json.dumps((protocol, dataclasses.asdict(json_spec)))
    return str_spec


def treespec_loads(serialized: str) -> TreeSpec:
    protocol, json_schema = json.loads(serialized)

    if protocol in _SUPPORTED_PROTOCOLS:
        return _SUPPORTED_PROTOCOLS[protocol].json_to_treespec(json_schema)
    raise ValueError(
        f"Unknown protocol {protocol}. "
        f"Available protocols: {list(_SUPPORTED_PROTOCOLS.keys())}",
    )


class _DummyLeaf:
    def __repr__(self) -> str:
        return "*"


def treespec_pprint(treespec: TreeSpec) -> str:
    dummy_tree = tree_unflatten(
        [_DummyLeaf() for _ in range(treespec.num_leaves)],
        treespec,
    )
    return repr(dummy_tree)


# TODO(angelayi): remove this function after OSS/internal stabilize
@deprecated(
    "`pytree_to_str` is deprecated. Please use `treespec_dumps` instead.",
    category=FutureWarning,
)
def pytree_to_str(treespec: TreeSpec) -> str:
    return treespec_dumps(treespec)


# TODO(angelayi): remove this function after OSS/internal stabilize
@deprecated(
    "`str_to_pytree` is deprecated. Please use `treespec_loads` instead.",
    category=FutureWarning,
)
def str_to_pytree(json: str) -> TreeSpec:
    return treespec_loads(json)


def arg_tree_leaves(*args: PyTree, **kwargs: PyTree) -> List[Any]:
    """Get a flat list of arguments to this function

    A slightly faster version of tree_leaves((args, kwargs))
    """
    leaves: List[Any] = []
    for a in args:
        leaves.extend(tree_iter(a))
    for a in kwargs.values():
        leaves.extend(tree_iter(a))
    return leaves


def tree_flatten_with_path(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Tuple[List[Tuple[KeyPath, Any]], TreeSpec]:
    """Flattens a pytree like :func:`tree_flatten`, but also returns each leaf's key path.

    Args:
        tree: a pytree to flatten. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A tuple where the first element is a list of (key path, leaf) pairs, and the
        second element is a :class:`TreeSpec` representing the structure of the flattened
        tree.
    """
    _, treespec = tree_flatten(tree, is_leaf)
    return list(_generate_key_paths((), tree, is_leaf)), treespec


def tree_leaves_with_path(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> List[Tuple[KeyPath, Any]]:
    """Gets the leaves of a pytree like ``tree_leaves`` and returns each leaf's key path.

    Args:
        tree: a pytree. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A list of (key path, leaf) pairs.
    """
    return list(_generate_key_paths((), tree, is_leaf))


def _generate_key_paths(
    key_path: KeyPath,
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Iterable[Tuple[KeyPath, Any]]:
    if is_leaf and is_leaf(tree):
        yield key_path, tree
        return

    node_type = _get_node_type(tree)
    handler = SUPPORTED_NODES.get(node_type)
    if not handler:
        # This is a leaf
        yield key_path, tree
        return

    flatten_with_keys = handler.flatten_with_keys_fn
    if flatten_with_keys:
        key_children, _ = flatten_with_keys(tree)
        for k, c in key_children:
            yield from _generate_key_paths((*key_path, k), c, is_leaf)
    else:
        # We registered this pytree but didn't add a flatten_with_keys_fn, complain.
        raise ValueError(
            f"Did not find a flatten_with_keys_fn for type: {node_type}. "
            "Please pass a flatten_with_keys_fn argument to register_pytree_node."
        )


def tree_map_with_path(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Like :func:`tree_map`, but the provided callable takes an additional key path argument.

    Args:
        func: A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees. The first positional argument
            to ``func`` is the key path of the leaf in question. The second
            positional argument is the value of the leaf.
        tree: A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests: A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(keypath, x, *xs)`` where ``keypath`` is the key path at the
        corresponding leaf in ``tree``, ``x`` is the value at that leaf, and
        ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    keypath_leaves, treespec = tree_flatten_with_path(tree, is_leaf)
    keypath_leaves = list(zip(*keypath_leaves))
    all_keypath_leaves = keypath_leaves + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(func(*xs) for xs in zip(*all_keypath_leaves))


def keystr(kp: KeyPath) -> str:
    """Given a key path, return a pretty-printed representation."""
    return "".join([str(k) for k in kp])


def key_get(obj: Any, kp: KeyPath) -> Any:
    """Given an object and a key path, return the value at the key path."""
    for k in kp:
        obj = k.get(obj)
    return obj
