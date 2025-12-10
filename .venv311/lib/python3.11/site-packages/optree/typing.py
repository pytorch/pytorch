# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Typing utilities for OpTree."""

from __future__ import annotations

import abc
import functools
import platform
import sys
import threading
import types
from builtins import dict as Dict  # noqa: N812
from builtins import list as List  # noqa: N812
from builtins import tuple as Tuple  # noqa: N812
from collections import OrderedDict
from collections import defaultdict as DefaultDict  # noqa: N812
from collections import deque as Deque  # noqa: N812
from collections.abc import (
    Collection,
    Hashable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Sequence,
    ValuesView,
)
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    ForwardRef,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    final,
    get_origin,
    runtime_checkable,
)
from typing_extensions import (
    NamedTuple,  # Generic NamedTuple: Python 3.11+
    Never,  # Python 3.11+
    ParamSpec,  # Python 3.10+
    Self,  # Python 3.11+
    TypeAlias,  # Python 3.10+
    TypeAliasType,  # Python 3.12+
)
from weakref import WeakKeyDictionary

import optree._C as _C
from optree._C import PyTreeKind, PyTreeSpec
from optree.accessors import (
    AutoEntry,
    DataclassEntry,
    FlattenedEntry,
    GetAttrEntry,
    GetItemEntry,
    MappingEntry,
    NamedTupleEntry,
    PyTreeAccessor,
    PyTreeEntry,
    SequenceEntry,
    StructSequenceEntry,
)


__all__ = [
    'PyTreeSpec',
    'PyTreeDef',
    'PyTreeKind',
    'PyTree',
    'PyTreeTypeVar',
    'CustomTreeNode',
    'Children',
    'MetaData',
    'FlattenFunc',
    'UnflattenFunc',
    'PyTreeEntry',
    'GetItemEntry',
    'GetAttrEntry',
    'FlattenedEntry',
    'AutoEntry',
    'SequenceEntry',
    'MappingEntry',
    'NamedTupleEntry',
    'StructSequenceEntry',
    'DataclassEntry',
    'PyTreeAccessor',
    'is_namedtuple',
    'is_namedtuple_class',
    'is_namedtuple_instance',
    'namedtuple_fields',
    'is_structseq',
    'is_structseq_class',
    'is_structseq_instance',
    'structseq_fields',
    'T',
    'S',
    'U',
    'KT',
    'VT',
    'P',
    'F',
    'Iterable',
    'Sequence',
    'Tuple',
    'List',
    'Dict',
    'NamedTuple',
    'OrderedDict',
    'DefaultDict',
    'Deque',
    'StructSequence',
]


PyTreeDef: TypeAlias = PyTreeSpec  # alias

T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
KT = TypeVar('KT')
VT = TypeVar('VT')
P = ParamSpec('P')
F = TypeVar('F', bound=Callable[..., Any])


Children: TypeAlias = Iterable[T]
MetaData: TypeAlias = Optional[Hashable]


@runtime_checkable
class CustomTreeNode(Protocol[T]):  # pylint: disable=too-few-public-methods
    """The abstract base class for custom pytree nodes."""

    def __tree_flatten__(
        self,
        /,
    ) -> (
        # Use `range(num_children)` as path entries
        tuple[Children[T], MetaData]
        |
        # With optionally implemented path entries
        tuple[Children[T], MetaData, Iterable[Any] | None]
    ):
        """Flatten the custom pytree node into children and metadata."""

    @classmethod
    def __tree_unflatten__(cls, metadata: MetaData, children: Children[T], /) -> Self:
        """Unflatten the children and metadata into the custom pytree node."""


_UnionType = type(Union[int, str])


try:  # pragma: no cover
    from typing import _tp_cache  # type: ignore[attr-defined] # pylint: disable=ungrouped-imports
except ImportError:  # pragma: no cover

    def _tp_cache(func: Callable[P, T], /) -> Callable[P, T]:
        cached = functools.lru_cache(func)

        @functools.wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return cached(*args, **kwargs)  # type: ignore[arg-type]
            except TypeError:
                # All real errors (not unhashable args) are raised below.
                return func(*args, **kwargs)

        return inner


class PyTree(Generic[T]):  # pragma: no cover
    """Generic PyTree type.

    >>> import torch
    >>> TensorTree = PyTree[torch.Tensor]
    >>> TensorTree  # doctest: +IGNORE_WHITESPACE
    typing.Union[torch.Tensor,
                 tuple[ForwardRef('PyTree[torch.Tensor]'), ...],
                 list[ForwardRef('PyTree[torch.Tensor]')],
                 dict[typing.Any, ForwardRef('PyTree[torch.Tensor]')],
                 collections.deque[ForwardRef('PyTree[torch.Tensor]')],
                 optree.typing.CustomTreeNode[ForwardRef('PyTree[torch.Tensor]')]]
    """

    __slots__: ClassVar[tuple[()]] = ()
    __instances__: ClassVar[
        WeakKeyDictionary[
            TypeAliasType,
            tuple[type | TypeAliasType, str | None],
        ]
    ] = WeakKeyDictionary()
    __instance_lock__: ClassVar[threading.Lock] = threading.Lock()

    @_tp_cache
    def __class_getitem__(  # noqa: C901 # pylint: disable=too-many-branches
        cls,
        item: (
            type[T]
            | TypeAliasType
            | tuple[type[T] | TypeAliasType]
            | tuple[type[T] | TypeAliasType, str | None]
        ),
    ) -> TypeAliasType:
        """Instantiate a PyTree type with the given type."""
        if not isinstance(item, tuple):
            item = (item, None)
        if len(item) == 1:
            item = (item[0], None)
        elif len(item) != 2:
            raise TypeError(
                f'{cls.__name__}[...] only supports a tuple of 2 items, '
                f'a parameter and a string of type name, got {item!r}.',
            )
        param, name = item
        if name is not None and not isinstance(name, str):
            raise TypeError(
                f'{cls.__name__}[...] only supports a tuple of 2 items, '
                f'a parameter and a string of type name, got {item!r}.',
            )

        if isinstance(param, _UnionType) and get_origin(param) is Union:  # type: ignore[unreachable]
            with cls.__instance_lock__:  # type: ignore[unreachable]
                try:
                    if param in cls.__instances__:
                        return param  # PyTree[PyTree[T]] -> PyTree[T]
                except TypeError:
                    pass  # non-hashable type

        if name is not None:
            recurse_ref = ForwardRef(name)
        elif isinstance(param, TypeVar):
            recurse_ref = ForwardRef(f'{cls.__name__}[{param.__name__}]')  # type: ignore[unreachable]
        elif isinstance(param, type):
            if param.__module__ == 'builtins':
                typename = param.__qualname__
            else:
                try:
                    typename = f'{param.__module__}.{param.__qualname__}'
                except AttributeError:
                    typename = f'{param.__module__}.{param.__name__}'
            recurse_ref = ForwardRef(f'{cls.__name__}[{typename}]')
        else:
            recurse_ref = ForwardRef(f'{cls.__name__}[{param!r}]')

        pytree_alias = Union[
            param,  # type: ignore[valid-type]
            Tuple[recurse_ref, ...],  # type: ignore[valid-type] # Tuple, NamedTuple, PyStructSequence
            List[recurse_ref],  # type: ignore[valid-type]
            Dict[Any, recurse_ref],  # type: ignore[valid-type] # Dict, OrderedDict, DefaultDict
            Deque[recurse_ref],  # type: ignore[valid-type]
            CustomTreeNode[recurse_ref],  # type: ignore[valid-type]
        ]

        with cls.__instance_lock__:
            cls.__instances__[pytree_alias] = (param, name)  # type: ignore[index]
        return pytree_alias  # type: ignore[return-value]

    def __new__(cls, /) -> Never:  # pylint: disable=arguments-differ
        """Prohibit instantiation."""
        raise TypeError('Cannot instantiate special typing classes.')

    def __init_subclass__(cls, /, *args: Any, **kwargs: Any) -> Never:
        """Prohibit subclassing."""
        raise TypeError('Cannot subclass special typing classes.')

    def __getitem__(self, key: Any, /) -> PyTree[T] | T:
        """Emulate collection-like behavior."""
        raise NotImplementedError

    def __getattr__(self, name: str, /) -> PyTree[T] | T:
        """Emulate dataclass-like behavior."""
        raise NotImplementedError

    def __contains__(self, key: Any, /) -> bool:
        """Emulate collection-like behavior."""
        raise NotImplementedError

    def __len__(self, /) -> int:
        """Emulate collection-like behavior."""
        raise NotImplementedError

    def __iter__(self, /) -> Iterator[PyTree[T] | T | Any]:
        """Emulate collection-like behavior."""
        raise NotImplementedError

    def index(self, key: Any, /) -> int:
        """Emulate sequence-like behavior."""
        raise NotImplementedError

    def count(self, key: Any, /) -> int:
        """Emulate sequence-like behavior."""
        raise NotImplementedError

    def get(self, key: Any, /, default: S | None = None) -> PyTree[T] | T | S | None:
        """Emulate mapping-like behavior."""
        raise NotImplementedError

    def keys(self, /) -> KeysView[Any]:
        """Emulate mapping-like behavior."""
        raise NotImplementedError

    def values(self, /) -> ValuesView[PyTree[T] | T]:
        """Emulate mapping-like behavior."""
        raise NotImplementedError

    def items(self, /) -> ItemsView[Any, PyTree[T] | T]:
        """Emulate mapping-like behavior."""
        raise NotImplementedError


# pylint: disable-next=too-few-public-methods
class PyTreeTypeVar:  # pragma: no cover
    """Type variable for PyTree.

    >>> import torch
    >>> TensorTree = PyTreeTypeVar('TensorTree', torch.Tensor)
    >>> TensorTree  # doctest: +IGNORE_WHITESPACE
    typing.Union[torch.Tensor,
                 tuple[ForwardRef('TensorTree'), ...],
                 list[ForwardRef('TensorTree')],
                 dict[typing.Any, ForwardRef('TensorTree')],
                 collections.deque[ForwardRef('TensorTree')],
                 optree.typing.CustomTreeNode[ForwardRef('TensorTree')]]
    """

    @_tp_cache
    def __new__(cls, /, name: str, param: type | TypeAliasType) -> TypeAliasType:  # type: ignore[misc]
        """Instantiate a PyTree type variable with the given name and parameter."""
        if not isinstance(name, str):
            raise TypeError(f'{cls.__name__} only supports a string of type name, got {name!r}.')
        return PyTree[param, name]  # type: ignore[misc,valid-type]

    def __init_subclass__(cls, /, *args: Any, **kwargs: Any) -> Never:
        """Prohibit subclassing."""
        raise TypeError('Cannot subclass special typing classes.')


class FlattenFunc(Protocol[T]):  # pylint: disable=too-few-public-methods
    """The type stub class for flatten functions."""

    @abc.abstractmethod
    def __call__(
        self,
        container: Collection[T],
        /,
    ) -> tuple[Children[T], MetaData] | tuple[Children[T], MetaData, Iterable[Any] | None]:
        """Flatten the container into children and metadata."""


class UnflattenFunc(Protocol[T]):  # pylint: disable=too-few-public-methods
    """The type stub class for unflatten functions."""

    @abc.abstractmethod
    def __call__(self, metadata: MetaData, children: Children[T], /) -> Collection[T]:
        """Unflatten the children and metadata back into the container."""


def _override_with_(
    cxx_implementation: Callable[P, T],
    /,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to override the Python implementation with the C++ implementation.

    >>> @_override_with_(any)
    ... def my_any(iterable):
    ...     for elem in iterable:
    ...         if elem:
    ...             return True
    ...     return False
    ...
    >>> my_any([False, False, True, False, False, True])  # run at C speed
    True
    """

    def wrapper(python_implementation: Callable[P, T], /) -> Callable[P, T]:
        @functools.wraps(python_implementation)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            return cxx_implementation(*args, **kwargs)

        wrapped.__cxx_implementation__ = cxx_implementation  # type: ignore[attr-defined]
        wrapped.__python_implementation__ = python_implementation  # type: ignore[attr-defined]

        return wrapped

    return wrapper


@_override_with_(_C.is_namedtuple)
def is_namedtuple(obj: object | type, /) -> bool:
    """Return whether the object is an instance of namedtuple or a subclass of namedtuple."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_namedtuple_class(cls)


@_override_with_(_C.is_namedtuple_instance)
def is_namedtuple_instance(obj: object, /) -> bool:
    """Return whether the object is an instance of namedtuple."""
    return is_namedtuple_class(type(obj))


@_override_with_(_C.is_namedtuple_class)
def is_namedtuple_class(cls: type, /) -> bool:
    """Return whether the class is a subclass of namedtuple."""
    return (
        isinstance(cls, type)
        and issubclass(cls, tuple)
        and isinstance(getattr(cls, '_fields', None), tuple)
        # pylint: disable-next=unidiomatic-typecheck
        and all(type(field) is str for field in cls._fields)  # type: ignore[attr-defined]
        and callable(getattr(cls, '_make', None))
        and callable(getattr(cls, '_asdict', None))
    )


@_override_with_(_C.namedtuple_fields)
def namedtuple_fields(obj: tuple | type[tuple], /) -> tuple[str, ...]:
    """Return the field names of a namedtuple."""
    if isinstance(obj, type):
        cls = obj
        if not is_namedtuple_class(cls):
            raise TypeError(f'Expected a collections.namedtuple type, got {cls!r}.')
    else:
        cls = type(obj)
        if not is_namedtuple_class(cls):
            raise TypeError(f'Expected an instance of collections.namedtuple type, got {obj!r}.')
    return cls._fields  # type: ignore[attr-defined]


_T_co = TypeVar('_T_co', covariant=True)


class StructSequenceMeta(type):
    """The metaclass for PyStructSequence stub type."""

    def __subclasscheck__(cls, subclass: type, /) -> bool:
        """Return whether the class is a PyStructSequence type.

        >>> import time
        >>> issubclass(time.struct_time, StructSequence)
        True
        >>> class MyTuple(tuple):
        ...     n_fields = 2
        ...     n_sequence_fields = 2
        ...     n_unnamed_fields = 0
        >>> issubclass(MyTuple, StructSequence)
        False
        """
        return is_structseq_class(subclass)

    def __instancecheck__(cls, instance: Any, /) -> bool:
        """Return whether the object is a PyStructSequence instance.

        >>> import sys
        >>> isinstance(sys.float_info, StructSequence)
        True
        >>> isinstance((1, 2), StructSequence)
        False
        """
        return is_structseq_instance(instance)


# Reference: https://github.com/python/typeshed/blob/main/stdlib/_typeshed/__init__.pyi
# This is an internal CPython type that is like, but subtly different from a NamedTuple.
# `StructSequence` classes are unsubclassable, so are all decorated with `@final`.
# pylint: disable-next=invalid-name,missing-class-docstring
@final
class StructSequence(tuple[_T_co, ...], metaclass=StructSequenceMeta):
    """A generic type stub for CPython's ``PyStructSequence`` type."""

    __slots__: ClassVar[tuple[()]] = ()

    n_fields: Final[ClassVar[int]]  # type: ignore[misc] # pylint: disable=invalid-name
    n_sequence_fields: Final[ClassVar[int]]  # type: ignore[misc] # pylint: disable=invalid-name
    n_unnamed_fields: Final[ClassVar[int]]  # type: ignore[misc] # pylint: disable=invalid-name

    def __init_subclass__(cls, /) -> Never:
        """Prohibit subclassing."""
        raise TypeError("type 'StructSequence' is not an acceptable base type")

    # pylint: disable-next=unused-argument,redefined-builtin
    def __new__(cls, /, sequence: Iterable[_T_co], dict: dict[str, Any] = ...) -> Self:
        """Create a new :class:`StructSequence` instance."""
        raise NotImplementedError


structseq: TypeAlias = StructSequence  # noqa: PYI042 # pylint: disable=invalid-name

del StructSequenceMeta


@_override_with_(_C.is_structseq)
def is_structseq(obj: object | type, /) -> bool:
    """Return whether the object is an instance of PyStructSequence or a class of PyStructSequence."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_structseq_class(cls)


@_override_with_(_C.is_structseq_instance)
def is_structseq_instance(obj: object, /) -> bool:
    """Return whether the object is an instance of PyStructSequence."""
    return is_structseq_class(type(obj))


# Set if the type allows subclassing (see CPython's Include/object.h)
Py_TPFLAGS_BASETYPE: int = _C.Py_TPFLAGS_BASETYPE  # (1UL << 10)  # pylint: disable=invalid-name


@_override_with_(_C.is_structseq_class)
def is_structseq_class(cls: type, /) -> bool:
    """Return whether the class is a class of PyStructSequence."""
    if (
        isinstance(cls, type)
        # Check direct inheritance from `tuple` rather than `issubclass(cls, tuple)`
        and cls.__bases__ == (tuple,)
        # Check PyStructSequence members
        and isinstance(getattr(cls, 'n_fields', None), int)
        and isinstance(getattr(cls, 'n_sequence_fields', None), int)
        and isinstance(getattr(cls, 'n_unnamed_fields', None), int)
    ):
        # Check the type does not allow subclassing
        if platform.python_implementation() == 'PyPy':  # pragma: pypy cover
            try:
                types.new_class('subclass', bases=(cls,))
            except (AssertionError, TypeError):
                return True
            return False
        return not bool(cls.__flags__ & Py_TPFLAGS_BASETYPE)  # pragma: pypy no cover
    return False


# pylint: disable-next=line-too-long
StructSequenceFieldType: type[types.MemberDescriptorType] = type(type(sys.version_info).major)  # type: ignore[assignment]


@_override_with_(_C.structseq_fields)
def structseq_fields(obj: tuple | type[tuple], /) -> tuple[str, ...]:
    """Return the field names of a PyStructSequence."""
    if isinstance(obj, type):
        cls = obj
        if not is_structseq_class(cls):
            raise TypeError(f'Expected a PyStructSequence type, got {cls!r}.')
    else:
        cls = type(obj)
        if not is_structseq_class(cls):
            raise TypeError(f'Expected an instance of PyStructSequence type, got {obj!r}.')

    if platform.python_implementation() == 'PyPy':  # pragma: pypy cover
        indices_by_name = {
            name: member.index  # type: ignore[attr-defined]
            for name, member in vars(cls).items()
            if isinstance(member, StructSequenceFieldType)
        }
        fields = sorted(indices_by_name, key=indices_by_name.get)  # type: ignore[arg-type]
    else:  # pragma: pypy no cover
        fields = [
            name
            for name, member in vars(cls).items()
            if isinstance(member, StructSequenceFieldType)
        ]
    return tuple(fields[: cls.n_sequence_fields])  # type: ignore[attr-defined]


del _tp_cache
