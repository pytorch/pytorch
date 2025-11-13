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
import importlib.metadata
import json
import threading
import types
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from enum import Enum
from typing import (
    Any,
    cast,
    ClassVar,
    Final,
    Generic,
    NoReturn,
    overload,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
)
from typing_extensions import deprecated, NamedTuple, Self

from torch.torch_version import TorchVersion as _TorchVersion


__all__ = [
    "PyTree",
    "Context",
    "FlattenFunc",
    "UnflattenFunc",
    "DumpableContext",
    "ToDumpableContextFn",
    "FromDumpableContextFn",
    "PyTreeSpec",
    "TreeSpec",
    "LeafSpec",
    "keystr",
    "key_get",
    "register_pytree_node",
    "tree_is_leaf",
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
    "is_namedtuple",
    "is_namedtuple_class",
    "is_namedtuple_instance",
    "is_structseq",
    "is_structseq_class",
    "is_structseq_instance",
]


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")


DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL = 1
NO_SERIALIZED_TYPE_NAME_FOUND = "NO_SERIALIZED_TYPE_NAME_FOUND"


class KeyEntry(Protocol):
    def __hash__(self) -> int: ...

    def __eq__(self, other: object) -> bool: ...

    def __str__(self) -> str: ...

    def get(self, parent: Any) -> Any: ...


class EnumEncoder(json.JSONEncoder):
    def default(self, obj: object) -> str | dict[str, Any]:
        if isinstance(obj, Enum):
            return {
                "__enum__": True,
                "fqn": f"{obj.__class__.__module__}:{obj.__class__.__qualname__}",
                "name": obj.name,
            }
        return cast(str, super().default(obj))


Context = Any
PyTree = Any
FlattenFunc = Callable[[PyTree], tuple[list[Any], Context]]
UnflattenFunc = Callable[[Iterable[Any], Context], PyTree]
DumpableContext = Any  # Any json dumpable text
ToDumpableContextFn = Callable[[Context], DumpableContext]
FromDumpableContextFn = Callable[[DumpableContext], Context]
ToStrFunc = Callable[["TreeSpec", list[str]], str]
MaybeFromStrFunc = Callable[[str], tuple[Any, Context, str] | None]
KeyPath = tuple[KeyEntry, ...]
FlattenWithKeysFunc = Callable[[PyTree], tuple[list[tuple[KeyEntry, Any]], Any]]


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
    type: type[Any]
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc
    flatten_with_keys_fn: FlattenWithKeysFunc | None


_NODE_REGISTRY_LOCK = threading.RLock()
SUPPORTED_NODES: dict[type[Any], NodeDef] = {}


# _SerializeNodeDef holds the following:
# - typ: the type of the node (e.g., "Dict", "List", etc)
# - serialized_type_name: the fully qualified name of the type, e.g. "collections.OrderedDict"
# - to_dumpable_context takes a TreeSpec, and returns a serialized string format of the
#   context, and the version number
# - from_dumpable_context takes in a string representation of the context, and the
#   version, and returns the deserialized context
class _SerializeNodeDef(NamedTuple):
    typ: type[Any]
    serialized_type_name: str
    to_dumpable_context: ToDumpableContextFn | None
    from_dumpable_context: FromDumpableContextFn | None


SUPPORTED_SERIALIZED_TYPES: dict[type[Any], _SerializeNodeDef] = {}
SERIALIZED_TYPE_TO_PYTHON_TYPE: dict[str, type[Any]] = {}

# NB: we try really hard to not import _cxx_pytree (which depends on optree)
# as much as possible. This is for isolation: a user who is not using C++ pytree
# shouldn't pay for it, and it helps makes things like cpython upgrades easier.
_optree_minimum_version = _TorchVersion("0.13.0")
try:
    _optree_version = importlib.metadata.version("optree")
except importlib.metadata.PackageNotFoundError:
    # No optree package found
    _cxx_pytree_dynamo_traceable = _cxx_pytree_exists = False
    _optree_version = _TorchVersion("0.0.0a0")
else:
    _optree_version = _TorchVersion(_optree_version)
    if _optree_version < _optree_minimum_version:
        # optree package less than our required minimum version.
        # Pretend the optree package doesn't exist.
        # NB: We will raise ImportError if the user directly tries to
        # `import torch.utils._cxx_pytree` (look in that file for the check).
        _cxx_pytree_dynamo_traceable = _cxx_pytree_exists = False
    else:
        _cxx_pytree_dynamo_traceable = _cxx_pytree_exists = True

_cxx_pytree_imported = False
_cxx_pytree_pending_imports: list[Any] = []


def register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
    flatten_with_keys_fn: FlattenWithKeysFunc | None = None,
) -> None:
    """Register a container-like type as pytree node.

    Note:
        :func:`register_dataclass` is a simpler way of registering a container-like
        type as a pytree node.

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


def register_dataclass(
    cls: type[Any],
    *,
    field_names: list[str] | None = None,
    drop_field_names: list[str] | None = None,
    serialized_type_name: str | None = None,
) -> None:
    """
    Registers a type that has the semantics of a ``dataclasses.dataclass`` type
    as a pytree node.

    This is a simpler API than :func:`register_pytree_node` for registering
    a dataclass or a custom class with the semantics of a dataclass.

    Args:
        cls: The python type to register. The class must have the semantics of a
        dataclass; in particular, it must be constructed by passing the fields
        in.
        field_names (Optional[List[str]]): A list of field names that correspond
            to the **non-constant data** in this class. This list must contain
            all the fields that are used to initialize the class. This argument
            is optional if ``cls`` is a dataclass, in which case the fields will
            be taken from ``dataclasses.fields()``.
        drop_field_names (Optional[List[str]]): A list of field names that
            should not be included in the pytree.
        serialized_type_name: A keyword argument used to specify the fully
            qualified name used when serializing the tree spec. This is only
            needed for serializing the treespec in torch.export.

    Example:

        >>> from torch import Tensor
        >>> from dataclasses import dataclass
        >>> import torch.utils._pytree as pytree
        >>>
        >>> @dataclass
        >>> class Point:
        >>>     x: Tensor
        >>>     y: Tensor
        >>>
        >>> pytree.register_dataclass(Point)
        >>>
        >>> point = Point(torch.tensor(0), torch.tensor(1))
        >>> point = pytree.tree_map(lambda x: x + 1, point)
        >>> assert torch.allclose(point.x, torch.tensor(1))
        >>> assert torch.allclose(point.y, torch.tensor(2))

    """
    drop_field_names = drop_field_names or []

    if not dataclasses.is_dataclass(cls):
        if field_names is None:
            raise ValueError(
                "field_names must be specified with a list of all fields used to "
                f"initialize {cls}, as it is not a dataclass."
            )
    elif field_names is None:
        field_names = [f.name for f in dataclasses.fields(cls) if f.init]
    else:
        dataclass_init_fields = {f.name for f in dataclasses.fields(cls) if f.init}
        dataclass_init_fields.difference_update(drop_field_names)

        if dataclass_init_fields != set(field_names):
            error_msg = "field_names does not include all dataclass fields.\n"

            if missing := dataclass_init_fields - set(field_names):
                error_msg += (
                    f"Missing fields in `field_names`: {missing}. If you want "
                    "to include these fields in the pytree, please add them "
                    "to `field_names`, otherwise please add them to "
                    "`drop_field_names`.\n"
                )

            if unexpected := set(field_names) - dataclass_init_fields:
                error_msg += (
                    f"Unexpected fields in `field_names`: {unexpected}. "
                    "Please remove these fields, or add them to `drop_field_names`.\n"
                )

            raise ValueError(error_msg)

    def _flatten_fn(obj: Any) -> tuple[list[Any], Context]:
        flattened = []
        flat_names = []
        none_names = []
        for name in field_names:
            val = getattr(obj, name)
            if val is not None:
                flattened.append(val)
                flat_names.append(name)
            else:
                none_names.append(name)
        return flattened, [flat_names, none_names]

    def _unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        flat_names, none_names = context
        return cls(
            **dict(zip(flat_names, values, strict=True)), **dict.fromkeys(none_names)
        )

    def _flatten_fn_with_keys(obj: Any) -> tuple[list[Any], Context]:
        flattened, (flat_names, _none_names) = _flatten_fn(obj)  # type: ignore[misc]
        return [
            (GetAttrKey(k), v) for k, v in zip(flat_names, flattened, strict=True)
        ], flat_names

    _private_register_pytree_node(
        cls,
        _flatten_fn,
        _unflatten_fn,
        serialized_type_name=serialized_type_name,
        flatten_with_keys_fn=_flatten_fn_with_keys,
    )


CONSTANT_NODES: set[type] = set()


def register_constant(cls: type[Any]) -> None:
    """Registers a type as a pytree node with no leaves.

    In a :func:`torch.compile` region, if instances of these types get passed to
    :func:`torch._dynamo.nonstrict_trace`-ed function, they treated as a
    constant (sometimes referred to as "static"):

    1. if the instance object existed before the :func:`torch.compile` region,
    we _assume_ no mutation will happen to it inside the :func:`torch.compile`
    region, require that it has non-default `__eq__` and `__hash__` methods, and
    we guard on the instance based on its `__eq__` method, i.e., if a new
    instance fails to match any instances from the previous compilations,
    :func:`torch.compile` will recompile the function using the new instance.

    2. else if the instance object is created inside the :func:`torch.compile`
    region, we currently don't support using it in a
    :func:`torch._dynamo.nonstrict_trace`-ed function.

    In general, if your class holds Tensors or dynamic int/float/bool (values that
    may change from run-to-run of a function being compiled), then you probably
    do not want to register it as a constant.

    Otherwise if you want to pass instance of a class to a
    :func:`torch._dynamo.nonstrict_trace`-ed function, but you either can't use
    :func:`register_pytree_node` on the class, or the class is "constant" enough
    that you don't want to bother using :func:`register_pytree_node`, you should
    consider using this function.

    Args:
        cls: the type to register as a constant. This type must be hashable.

    Example:

        >>> from dataclasses import dataclass
        >>> import torch.utils._pytree as pytree
        >>>
        >>> @dataclass(frozen=True)
        >>> class Config:
        >>>     norm: str
        >>>
        >>> pytree.register_constant(Config)
        >>>
        >>> config = Config("l2")
        >>> values, spec = pytree.tree_flatten(config)
        >>> assert len(values) == 0

    """
    if cls.__eq__ is object.__eq__:  # type: ignore[comparison-overlap]
        raise TypeError(
            "register_constant(cls) expects `cls` to have a non-default `__eq__` implementation."
        )

    # Class with a custom `__eq__` without `__hash__` won't inherit the default
    # `__hash__` from object; see https://stackoverflow.com/a/1608907.
    if cls.__hash__ is None:  # type: ignore[comparison-overlap]
        raise TypeError(
            "register_constant(cls) expects `cls` to have a non-default `__hash__` implementation."
        )

    def _flatten(x):  # type: ignore[no-untyped-def]
        return [], ConstantNode(x)

    def _unflatten(_, context):  # type: ignore[no-untyped-def]
        return context.value

    def _flatten_with_keys(x):  # type: ignore[no-untyped-def]
        return [], ConstantNode(x)

    with _NODE_REGISTRY_LOCK:
        _private_register_pytree_node(
            cls,
            _flatten,
            _unflatten,
            flatten_with_keys_fn=_flatten_with_keys,
        )
        CONSTANT_NODES.add(cls)


def is_constant_class(cls: type[Any]) -> bool:
    return isinstance(cls, type) and cls in CONSTANT_NODES


@dataclasses.dataclass(frozen=True)
class ConstantNode:
    value: Any


def _is_constant_holder(spec: "TreeSpec") -> bool:
    """Checks if the spec is from a pytree registered with register_constant"""
    return isinstance(spec._context, ConstantNode)


def _retrieve_constant(spec: "TreeSpec") -> Any:
    """Given a spec from a pytree registered with register_constant, retrieves the constant"""
    if not _is_constant_holder(spec):
        raise AssertionError("spec does not correspond to a registered constant pytree")
    return tree_unflatten([], spec)


def _register_namedtuple(
    cls: type[Any],
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
    cls: type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    to_str_fn: ToStrFunc | None = None,  # deprecated
    maybe_from_str_fn: MaybeFromStrFunc | None = None,  # deprecated
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
    flatten_with_keys_fn: FlattenWithKeysFunc | None = None,
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


def _deregister_pytree_node(
    cls: type[Any],
) -> None:
    """This is an internal function that is used to deregister a pytree node type
    for the Python pytree only. This should be only used inside PyTorch.
    """
    with _NODE_REGISTRY_LOCK:
        del SUPPORTED_NODES[cls]
        node_def = SUPPORTED_SERIALIZED_TYPES[cls]
        del SERIALIZED_TYPE_TO_PYTHON_TYPE[node_def.serialized_type_name]
        del SUPPORTED_SERIALIZED_TYPES[cls]
        CONSTANT_NODES.discard(cls)


def _private_register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
    flatten_with_keys_fn: FlattenWithKeysFunc | None = None,
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
                stacklevel=2,
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


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_namedtuple(obj: object | type) -> bool:
    """Return whether the object is an instance of namedtuple or a subclass of namedtuple."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_namedtuple_class(cls)


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_namedtuple_class(cls: type) -> bool:
    """Return whether the class is a subclass of namedtuple."""
    return (
        isinstance(cls, type)
        and issubclass(cls, tuple)
        and isinstance(getattr(cls, "_fields", None), tuple)
        and all(type(field) is str for field in cls._fields)  # type: ignore[attr-defined]
        and callable(getattr(cls, "_make", None))
        and callable(getattr(cls, "_asdict", None))
    )


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_namedtuple_instance(obj: object) -> bool:
    """Return whether the object is an instance of namedtuple."""
    return is_namedtuple_class(type(obj))


_T_co = TypeVar("_T_co", covariant=True)


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
class structseq(tuple[_T_co, ...]):
    """A generic type stub for CPython's ``PyStructSequence`` type."""

    __slots__: ClassVar[tuple[()]] = ()

    n_fields: Final[int]  # type: ignore[misc]
    n_sequence_fields: Final[int]  # type: ignore[misc]
    n_unnamed_fields: Final[int]  # type: ignore[misc]

    def __init_subclass__(cls) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError("type 'structseq' is not an acceptable base type")

    def __new__(
        cls: type[Self],
        sequence: Iterable[_T_co],
        # pyrefly: ignore [bad-function-definition]
        dict: dict[str, Any] = ...,
    ) -> Self:
        raise NotImplementedError


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_structseq(obj: object | type) -> bool:
    """Return whether the object is an instance of PyStructSequence or a class of PyStructSequence."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_structseq_class(cls)


# Set if the type allows subclassing (see CPython's Include/object.h)
Py_TPFLAGS_BASETYPE: int = 1 << 10


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_structseq_class(cls: type) -> bool:
    """Return whether the class is a class of PyStructSequence."""
    return (
        isinstance(cls, type)
        # Check direct inheritance from `tuple` rather than `issubclass(cls, tuple)`
        and cls.__bases__ == (tuple,)
        # Check PyStructSequence members
        and isinstance(getattr(cls, "n_fields", None), int)
        and isinstance(getattr(cls, "n_sequence_fields", None), int)
        and isinstance(getattr(cls, "n_unnamed_fields", None), int)
        # Check the type does not allow subclassing
        and not bool(cls.__flags__ & Py_TPFLAGS_BASETYPE)  # only works for CPython
    )


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_structseq_instance(obj: object) -> bool:
    """Return whether the object is an instance of PyStructSequence."""
    return is_structseq_class(type(obj))


def _tuple_flatten(d: tuple[T, ...]) -> tuple[list[T], Context]:
    return list(d), None


def _tuple_flatten_with_keys(
    d: tuple[T, ...],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _tuple_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _tuple_unflatten(values: Iterable[T], context: Context) -> tuple[T, ...]:
    return tuple(values)


def _list_flatten(d: list[T]) -> tuple[list[T], Context]:
    return d, None


def _list_flatten_with_keys(d: list[T]) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _list_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _list_unflatten(values: Iterable[T], context: Context) -> list[T]:
    return list(values)


def _dict_flatten(d: dict[Any, T]) -> tuple[list[T], Context]:
    return list(d.values()), list(d.keys())


def _dict_flatten_with_keys(
    d: dict[Any, T],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _dict_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(MappingKey(k), v) for k, v in zip(context, values, strict=True)], context


def _dict_unflatten(values: Iterable[T], context: Context) -> dict[Any, T]:
    return dict(zip(context, values, strict=True))


def _namedtuple_flatten(d: NamedTuple) -> tuple[list[Any], Context]:
    return list(d), type(d)


def _namedtuple_flatten_with_keys(
    d: NamedTuple,
) -> tuple[list[tuple[KeyEntry, Any]], Context]:
    values, context = _namedtuple_flatten(d)
    # pyrefly: ignore [bad-return]
    return (
        [
            (GetAttrKey(field), v)
            for field, v in zip(context._fields, values, strict=True)
        ],
        context,
    )


def _namedtuple_unflatten(values: Iterable[T], context: Context) -> NamedTuple:
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


def _ordereddict_flatten(d: OrderedDict[Any, T]) -> tuple[list[T], Context]:
    return list(d.values()), list(d.keys())


def _ordereddict_flatten_with_keys(
    d: OrderedDict[Any, T],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _ordereddict_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(MappingKey(k), v) for k, v in zip(context, values, strict=True)], context


def _ordereddict_unflatten(
    values: Iterable[T],
    context: Context,
) -> OrderedDict[Any, T]:
    return OrderedDict((key, value) for key, value in zip(context, values, strict=True))


_odict_flatten = _ordereddict_flatten
_odict_unflatten = _ordereddict_unflatten


def _defaultdict_flatten(d: defaultdict[Any, T]) -> tuple[list[T], Context]:
    values, dict_context = _dict_flatten(d)
    return values, [d.default_factory, dict_context]


def _defaultdict_flatten_with_keys(
    d: defaultdict[Any, T],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _defaultdict_flatten(d)
    _, dict_context = context
    # pyrefly: ignore [bad-return]
    return [
        (MappingKey(k), v) for k, v in zip(dict_context, values, strict=True)
    ], context


def _defaultdict_unflatten(
    values: Iterable[T],
    context: Context,
) -> defaultdict[Any, T]:
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
    if not isinstance(dumpable_context, dict):
        raise AssertionError("dumpable_context must be a dict")

    expected_keys = {
        "default_factory_module",
        "default_factory_name",
        "dict_context",
    }
    if set(dumpable_context) != expected_keys:
        raise AssertionError(
            f"dumpable_context keys must be {expected_keys}, got {set(dumpable_context)}"
        )

    default_factory_module = dumpable_context["default_factory_module"]
    default_factory_name = dumpable_context["default_factory_name"]
    if not isinstance(default_factory_module, str):
        raise AssertionError("default_factory_module must be a string")
    if not isinstance(default_factory_name, str):
        raise AssertionError("default_factory_name must be a string")
    module = importlib.import_module(default_factory_module)
    default_factory = getattr(module, default_factory_name)

    dict_context = dumpable_context["dict_context"]
    return [default_factory, dict_context]


def _deque_flatten(d: deque[T]) -> tuple[list[T], Context]:
    return list(d), d.maxlen


def _deque_flatten_with_keys(
    d: deque[T],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _deque_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _deque_unflatten(values: Iterable[T], context: Context) -> deque[T]:
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


STANDARD_DICT_TYPES: frozenset[type] = frozenset({dict, OrderedDict, defaultdict})
BUILTIN_TYPES: frozenset[type] = frozenset(
    {
        tuple,
        list,
        dict,
        namedtuple,  # type: ignore[arg-type]
        OrderedDict,
        defaultdict,
        deque,
    },
)


@deprecated(
    "torch.utils._pytree._is_namedtuple_instance is private and will be removed in a future release. "
    "Please use torch.utils._pytree.is_namedtuple_instance instead.",
    category=FutureWarning,
)
def _is_namedtuple_instance(tree: Any) -> bool:
    return is_namedtuple_instance(tree)


def _get_node_type(tree: Any) -> Any:
    node_type = type(tree)
    # All namedtuple types are implicitly registered as pytree nodes.
    # XXX: Other parts of the codebase expect namedtuple types always return
    #      `namedtuple` instead of the actual namedtuple type. Even if the type
    #      is explicitly registered.
    if is_namedtuple_class(node_type):
        return namedtuple
    return node_type


# A leaf is defined as anything that is not a Node.
def tree_is_leaf(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    """Check if a pytree is a leaf.

    >>> tree_is_leaf(1)
    True
    >>> tree_is_leaf(None)
    True
    >>> tree_is_leaf([1, 2, 3])
    False
    >>> tree_is_leaf((1, 2, 3), is_leaf=lambda x: isinstance(x, tuple))
    True
    >>> tree_is_leaf({"a": 1, "b": 2, "c": 3})
    False
    >>> tree_is_leaf({"a": 1, "b": 2, "c": None})
    False
    """
    if is_leaf is not None and is_leaf(tree):
        return True
    return _get_node_type(tree) not in SUPPORTED_NODES


@deprecated(
    "torch.utils._pytree._is_leaf is private and will be removed in a future release. "
    "Please use torch.utils._pytree.tree_is_leaf instead.",
    category=FutureWarning,
)
def _is_leaf(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool:
    return tree_is_leaf(tree, is_leaf=is_leaf)


# A TreeSpec represents the structure of a pytree. It holds:
#   "type": the type of root Node of the pytree
#   context: some context that is useful in unflattening the pytree
#   children(): specs for each child of the root Node
#   num_nodes: the total number of nodes
#   num_leaves: the number of leaves
#   num_children: the number of children of the root Node (i.e., len(children()))
#   is_leaf(): whether the root Node is a leaf
@dataclasses.dataclass(init=False, frozen=True, eq=True, repr=False)
class TreeSpec:
    type: Any
    _context: Context
    _children: list[Self]

    num_nodes: int = dataclasses.field(init=False)
    num_leaves: int = dataclasses.field(init=False)
    num_children: int = dataclasses.field(init=False)

    def __init__(
        self,
        type: Any,
        context: Context,  # keep for backward compatibility
        children_specs: list[Self],  # keep for backward compatibility
    ) -> None:
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_children", children_specs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.type is None:
            assert self._context is None
            assert len(self._children) == 0
            num_nodes = 1
            num_leaves = 1
            num_children = 0
        else:
            num_nodes = sum((spec.num_nodes for spec in self._children), start=1)
            num_leaves = sum(spec.num_leaves for spec in self._children)
            num_children = len(self._children)
        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "num_leaves", num_leaves)
        object.__setattr__(self, "num_children", num_children)

    def __repr__(self, indent: int = 0) -> str:
        repr_prefix: str = f"TreeSpec({self.type.__name__}, {self._context}, ["
        children_specs_str: str = ""
        if self.num_children > 0:
            indent += 2
            children_specs_str += self._children[0].__repr__(indent)
            children_specs_str += "," if self.num_children > 1 else ""
            children_specs_str += ",".join(
                [
                    "\n" + " " * indent + child.__repr__(indent)
                    for child in self._children[1:]
                ]
            )
        repr_suffix: str = f"{children_specs_str}])"
        return repr_prefix + repr_suffix

    def __eq__(self, other: PyTree) -> bool:
        if self is other:
            return True
        elif other.__class__ is self.__class__:
            if str(self.type) != str(other.type):
                return False
            if self._context != other._context:
                return False
            elif self._children != other._children:
                return False
            return True
        return NotImplemented

    @property
    def context(self) -> Context:
        return self._context

    @property
    @deprecated(
        "`treespec.children_specs` is deprecated. "
        "Use `treespec.child(index)` to access a single child, "
        "or `treespec.children()` to get all children.",
        category=FutureWarning,
    )
    def children_specs(self) -> list[Self]:
        return self._children

    def is_leaf(self) -> bool:
        return self.num_nodes == 1 and self.num_leaves == 1

    def children(self) -> list[Self]:
        return self._children.copy()

    def child(self, index: int) -> Self:
        return self._children[index]

    def flatten_up_to(self, tree: PyTree) -> list[PyTree]:
        def helper(treespec: TreeSpec, tree: PyTree, subtrees: list[PyTree]) -> None:
            if treespec.is_leaf():
                subtrees.append(tree)
                return

            node_type = _get_node_type(tree)
            if treespec.type not in BUILTIN_TYPES:
                # Always require custom node types to match exactly
                if node_type != treespec.type:
                    raise ValueError(
                        f"Type mismatch; "
                        f"expected {treespec.type!r}, but got {node_type!r}.",
                    )
                flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
                children, context = flatten_fn(tree)
                if len(children) != treespec.num_children:
                    raise ValueError(
                        f"Node arity mismatch; "
                        f"expected {treespec.num_children}, but got {len(children)}.",
                    )
                if context != treespec._context:
                    raise ValueError(
                        f"Node context mismatch for custom node type {treespec.type!r}.",
                    )
            else:
                # For builtin dictionary types, we allow some flexibility
                # Otherwise, we require exact matches
                both_standard_dict = (
                    treespec.type in STANDARD_DICT_TYPES
                    and node_type in STANDARD_DICT_TYPES
                )
                if not both_standard_dict and node_type != treespec.type:
                    raise ValueError(
                        f"Node type mismatch; "
                        f"expected {treespec.type!r}, but got {node_type!r}.",
                    )
                if len(tree) != treespec.num_children:
                    raise ValueError(
                        f"Node arity mismatch; "
                        f"expected {treespec.num_children}, but got {len(tree)}.",
                    )

                if both_standard_dict:
                    # dictionary types are compatible with each other
                    dict_context = (
                        treespec._context
                        if treespec.type is not defaultdict
                        # ignore mismatch of `default_factory` for defaultdict
                        else treespec._context[1]
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
                    children = [tree[key] for key in expected_keys]
                else:
                    # node_type is treespec.type
                    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
                    children, context = flatten_fn(tree)
                    if (
                        node_type is not deque  # ignore mismatch of `maxlen` for deque
                    ) and context != treespec._context:
                        raise ValueError(
                            f"Node context mismatch for node type {treespec.type!r}; "
                            f"expected {treespec._context!r}, but got {context!r}.",  # namedtuple type mismatch
                        )

            for subtree, subspec in zip(children, treespec._children, strict=True):
                helper(subspec, subtree, subtrees)

        subtrees: list[PyTree] = []
        helper(self, tree, subtrees)
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
        for child_spec in self._children:
            end += child_spec.num_leaves
            child_pytrees.append(child_spec.unflatten(leaves[start:end]))
            start = end

        return unflatten_fn(child_pytrees, self._context)

    def __hash__(self) -> int:
        node_type = self.type
        if node_type is defaultdict:
            default_factory, dict_context = self._context
            hashable_context = (default_factory, tuple(dict_context))
        elif node_type in (dict, OrderedDict):
            hashable_context = tuple(self._context)
        elif node_type is None or node_type in BUILTIN_TYPES:
            hashable_context = self._context
        elif isinstance(self._context, ConstantNode):
            hashable_context = self._context.value
        else:
            # The context for user-defined node types might not be hashable.
            # Ignore it for hashing.
            # This does not break the correctness that equal objects imply the
            # same hash. This might increase the hash collision rate, but we
            # don't care about that.
            hashable_context = None
        return hash((node_type, hashable_context, tuple(self._children)))


PyTreeSpec: TypeAlias = TreeSpec


# NOTE: subclassing a dataclass is subtle. In order to enable reasoning about
# this class with `dataclasses.fields`, etc., while having a simplified
# constructor that takes no argument, we wrap with `dataclass(init=True, ...)`
# again, with fields that have `init=False`.
@deprecated(
    "`isinstance(treespec, LeafSpec)` is deprecated, "
    "use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.",
    category=FutureWarning,
)
@dataclasses.dataclass(init=True, frozen=True, eq=False, repr=False)
class LeafSpec(TreeSpec):
    type: Any = dataclasses.field(default=None, init=False)
    _context: Context = dataclasses.field(default=None, init=False)
    _children: list[Self] = dataclasses.field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # Override `__post_init__` for `num_leaves` derivation.
        object.__setattr__(self, "num_nodes", 1)
        object.__setattr__(self, "num_leaves", 1)
        object.__setattr__(self, "num_children", 0)

    def __repr__(self, indent: int = 0) -> str:
        return "*"


# All leaves are equivalent, so represent with a single object to save on
# object construction time
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=__name__, append=False
    )
    _LEAF_SPEC = LeafSpec()


def treespec_leaf() -> LeafSpec:
    """Make a treespec representing a leaf node."""
    return _LEAF_SPEC


def treespec_tuple(iterable: Iterable[TreeSpec] = (), /) -> TreeSpec:
    """Make a tuple treespec from an iterable of child treespecs."""
    children = list(iterable)
    if any(not isinstance(child, TreeSpec) for child in children):
        raise ValueError(f"Expected a tuple of TreeSpec values, got: {children!r}.")
    return TreeSpec(tuple, None, children)


def treespec_dict(
    mapping: Mapping[Any, TreeSpec] | Iterable[tuple[Any, TreeSpec]] = (),
    /,
    **kwargs: TreeSpec,
) -> TreeSpec:
    """Make a dict treespec from a dict of child treespecs."""
    dct = dict(mapping, **kwargs)
    if any(not isinstance(child, TreeSpec) for child in dct.values()):
        raise ValueError(f"Expected a dictionary of TreeSpec values, got: {dct!r}.")
    return TreeSpec(dict, list(dct.keys()), list(dct.values()))


def tree_flatten(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> tuple[list[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree.
    """

    def helper(node: PyTree, leaves: list[Any]) -> TreeSpec:
        if tree_is_leaf(node, is_leaf=is_leaf):
            leaves.append(node)
            return _LEAF_SPEC

        node_type = _get_node_type(node)
        flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
        children, context = flatten_fn(node)

        # Recursively flatten the children
        subspecs = [helper(child, leaves) for child in children]
        return TreeSpec(node_type, context, subspecs)

    leaves: list[Any] = []
    treespec = helper(tree, leaves)
    return leaves, treespec


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
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> Iterable[Any]:
    """Get an iterator over the leaves of a pytree."""
    if tree_is_leaf(tree, is_leaf=is_leaf):
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
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[Any]:
    """Get a list of leaves of a pytree."""
    return list(tree_iter(tree, is_leaf=is_leaf))


def tree_structure(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> TreeSpec:
    """Get the TreeSpec for a pytree."""
    return tree_flatten(tree, is_leaf=is_leaf)[1]


def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`.

    >>> tree_map(lambda x: x + 1, {"x": 7, "y": (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x is None, {"x": 7, "y": (42, 64), "z": None})
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
    is_leaf: Callable[[PyTree], bool] | None = None,
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


Type2 = tuple[type[T], type[S]]
Type3 = tuple[type[T], type[S], type[U]]
TypeAny = Union[type[Any], tuple[type[Any], ...], types.UnionType]

Fn2 = Callable[[T | S], R]
Fn3 = Callable[[T | S | U], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]


# These specializations help with type inference on the lambda passed to this
# function
@overload
def map_only(type_or_types_or_pred: type[T], /) -> MapOnlyFn[Fn[T, Any]]: ...


@overload
def map_only(type_or_types_or_pred: Type2[T, S], /) -> MapOnlyFn[Fn2[T, S, Any]]: ...


@overload
def map_only(
    type_or_types_or_pred: Type3[T, S, U], /
) -> MapOnlyFn[Fn3[T, S, U, Any]]: ...


# This specialization is needed for the implementations below that call
@overload
def map_only(type_or_types_or_pred: TypeAny, /) -> MapOnlyFn[FnAny[Any]]: ...


@overload
def map_only(
    type_or_types_or_pred: Callable[[Any], bool], /
) -> MapOnlyFn[FnAny[Any]]: ...


def map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool], /
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
    if isinstance(type_or_types_or_pred, (type, tuple, types.UnionType)):

        def pred(x: Any) -> bool:
            return isinstance(x, type_or_types_or_pred)  # type: ignore[arg-type]

    elif callable(type_or_types_or_pred):
        pred = type_or_types_or_pred  # type: ignore[assignment]
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
    type_or_types_or_pred: type[T],
    /,
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: TypeAny,
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


def tree_map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    return tree_map(map_only(type_or_types_or_pred)(func), tree, is_leaf=is_leaf)


@overload
def tree_map_only_(
    type_or_types_or_pred: type[T],
    /,
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: TypeAny,
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


def tree_map_only_(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    return tree_map_(map_only(type_or_types_or_pred)(func), tree, is_leaf=is_leaf)


def tree_all(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(map(pred, flat_args))


def tree_any(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(map(pred, flat_args))


@overload
def tree_all_only(
    type_or_types: type[T],
    /,
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_all_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_all_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


def tree_all_only(
    type_or_types: TypeAny,
    /,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(pred(x) for x in flat_args if isinstance(x, type_or_types))


@overload
def tree_any_only(
    type_or_types: type[T],
    /,
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_any_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_any_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


def tree_any_only(
    type_or_types: TypeAny,
    /,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(pred(x) for x in flat_args if isinstance(x, type_or_types))


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
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[Any] | None:
    if not isinstance(treespec, TreeSpec):
        raise AssertionError("treespec must be a TreeSpec")

    if tree_is_leaf(tree, is_leaf=is_leaf):
        return [tree] * treespec.num_leaves
    if treespec.is_leaf():
        return None
    node_type = _get_node_type(tree)
    if node_type != treespec.type:
        return None

    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(tree)

    # Check if the Node is different from the spec
    if len(child_pytrees) != treespec.num_children or context != treespec._context:
        return None

    # Recursively flatten the children
    result: list[Any] = []
    for child, child_spec in zip(child_pytrees, treespec._children, strict=True):
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

    type: str | None
    context: DumpableContext
    children_spec: list["_TreeSpecSchema"]


class _ProtocolFn(NamedTuple):
    treespec_to_json: Callable[[TreeSpec], DumpableContext]
    json_to_treespec: Callable[[DumpableContext], TreeSpec]


_SUPPORTED_PROTOCOLS: dict[int, _ProtocolFn] = {}


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
            serialized_context = json.dumps(treespec._context, cls=EnumEncoder)
        except TypeError as e:
            raise TypeError(
                "Unable to serialize context. "
                "Please make the context json dump-able, or register a "
                "custom serializer using _register_pytree_node."
            ) from e
    else:
        serialized_context = serialize_node_def.to_dumpable_context(treespec._context)

    child_schemas = [_treespec_to_json(child) for child in treespec._children]

    return _TreeSpecSchema(serialized_type_name, serialized_context, child_schemas)


def enum_object_hook(obj: dict[str, Any]) -> Enum | dict[str, Any]:
    if "__enum__" in obj:
        modname, _, classname = obj["fqn"].partition(":")
        mod = importlib.import_module(modname)
        enum_cls = mod
        for attr in classname.split("."):
            enum_cls = getattr(enum_cls, attr)
        enum_cls = cast(type[Enum], enum_cls)
        # pyrefly: ignore [unsupported-operation]
        return enum_cls[obj["name"]]
    return obj


def _json_to_treespec(json_schema: DumpableContext) -> TreeSpec:
    if (
        json_schema["type"] is None
        and json_schema["context"] is None
        and len(json_schema["children_spec"]) == 0
    ):
        return _LEAF_SPEC

    if json_schema["type"] not in SERIALIZED_TYPE_TO_PYTHON_TYPE:
        raise NotImplementedError(
            f"Deserializing {json_schema['type']} in pytree is not registered.",
        )

    typ = SERIALIZED_TYPE_TO_PYTHON_TYPE[json_schema["type"]]
    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[typ]

    if serialize_node_def.from_dumpable_context is None:
        try:
            context = json.loads(json_schema["context"], object_hook=enum_object_hook)
        except TypeError as ex:
            raise TypeError(
                "Unable to deserialize context. "
                "Please make the context json load-able, or register a "
                "custom serializer using _register_pytree_node.",
            ) from ex
    else:
        context = serialize_node_def.from_dumpable_context(json_schema["context"])

    children_specs = [
        _json_to_treespec(child_string) for child_string in json_schema["children_spec"]
    ]

    return TreeSpec(typ, context, children_specs)


_SUPPORTED_PROTOCOLS[1] = _ProtocolFn(_treespec_to_json, _json_to_treespec)


def treespec_dumps(treespec: TreeSpec, protocol: int | None = None) -> str:
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

    str_spec = json.dumps((protocol, dataclasses.asdict(json_spec)), cls=EnumEncoder)
    return str_spec


@functools.lru_cache
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


def arg_tree_leaves(*args: PyTree, **kwargs: PyTree) -> list[Any]:
    """Get a flat list of arguments to this function

    A slightly faster version of tree_leaves((args, kwargs))
    """
    leaves: list[Any] = []
    for a in args:
        leaves.extend(tree_iter(a))
    for a in kwargs.values():
        leaves.extend(tree_iter(a))
    return leaves


def tree_flatten_with_path(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> tuple[list[tuple[KeyPath, Any]], TreeSpec]:
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
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[tuple[KeyPath, Any]]:
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
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> Iterable[tuple[KeyPath, Any]]:
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
    is_leaf: Callable[[PyTree], bool] | None = None,
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
    keypath_leaves = list(zip(*keypath_leaves, strict=True))
    all_keypath_leaves = keypath_leaves + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(func(*xs) for xs in zip(*all_keypath_leaves, strict=True))


def keystr(kp: KeyPath) -> str:
    """Given a key path, return a pretty-printed representation."""
    return "".join([str(k) for k in kp])


def key_get(obj: Any, kp: KeyPath) -> Any:
    """Given an object and a key path, return the value at the key path."""
    for k in kp:
        obj = k.get(obj)
    return obj
