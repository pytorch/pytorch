# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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

import functools
import platform
import types
from collections.abc import Hashable
from typing import (
    Any,
    Callable,
    DefaultDict,
    Deque,
    Dict,
    ForwardRef,
    Generic,
    Iterable,
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import Final  # Python 3.8+
from typing_extensions import NamedTuple  # Generic NamedTuple: Python 3.11+
from typing_extensions import OrderedDict  # Generic OrderedDict: Python 3.7.2+
from typing_extensions import ParamSpec  # Python 3.10+
from typing_extensions import Protocol  # Python 3.8+
from typing_extensions import Self  # Python 3.11+
from typing_extensions import TypeAlias  # Python 3.10+
from typing_extensions import runtime_checkable  # Python 3.8+

from optree import _C
from optree._C import PyTreeKind, PyTreeSpec
from optree.accessor import (
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
    'is_namedtuple_instance',
    'is_namedtuple_class',
    'namedtuple_fields',
    'is_structseq',
    'is_structseq_instance',
    'is_structseq_class',
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
    'List',
    'Tuple',
    'NamedTuple',
    'Dict',
    'OrderedDict',
    'DefaultDict',
    'Deque',
]


PyTreeDef = PyTreeSpec  # alias

T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
KT = TypeVar('KT')
VT = TypeVar('VT')
P = ParamSpec('P')
F = TypeVar('F', bound=Callable[..., Any])


Children: TypeAlias = Iterable[T]
_MetaData = TypeVar('_MetaData', bound=Hashable)
MetaData: TypeAlias = Optional[_MetaData]


@runtime_checkable
class CustomTreeNode(Protocol[T]):
    """The abstract base class for custom pytree nodes."""

    def tree_flatten(
        self,
    ) -> (
        # Use `range(num_children)` as path entries
        tuple[Children[T], MetaData]
        |
        # With optionally implemented path entries
        tuple[Children[T], MetaData, Iterable[Any] | None]
    ):
        """Flatten the custom pytree node into children and metadata."""

    @classmethod
    def tree_unflatten(cls, metadata: MetaData, children: Children[T]) -> CustomTreeNode[T]:
        """Unflatten the children and metadata into the custom pytree node."""


_UnionType = type(Union[int, str])


def _tp_cache(func: Callable[P, T]) -> Callable[P, T]:
    cached = functools.lru_cache()(func)

    @functools.wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return cached(*args, **kwargs)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover
            # All real errors (not unhashable args) are raised below.
            return func(*args, **kwargs)

    return inner


class PyTree(Generic[T]):  # pragma: no cover
    """Generic PyTree type.

    >>> import torch
    >>> TensorTree = PyTree[torch.Tensor]
    >>> TensorTree  # doctest: +IGNORE_WHITESPACE
    typing.Union[torch.Tensor,
                 typing.Tuple[ForwardRef('PyTree[torch.Tensor]'), ...],
                 typing.List[ForwardRef('PyTree[torch.Tensor]')],
                 typing.Dict[typing.Any, ForwardRef('PyTree[torch.Tensor]')],
                 typing.Deque[ForwardRef('PyTree[torch.Tensor]')],
                 optree.typing.CustomTreeNode[ForwardRef('PyTree[torch.Tensor]')]]
    """

    @_tp_cache
    def __class_getitem__(  # noqa: C901
        cls,
        item: T | tuple[T] | tuple[T, str | None],
    ) -> TypeAlias:
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

        if (
            isinstance(param, _UnionType)
            and param.__origin__ is Union  # type: ignore[attr-defined]
            and hasattr(param, '__pytree_args__')
        ):
            return param  # PyTree[PyTree[T]] -> PyTree[T]

        if name is not None:
            recurse_ref = ForwardRef(name)
        elif isinstance(param, TypeVar):
            recurse_ref = ForwardRef(f'{cls.__name__}[{param.__name__}]')
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
        pytree_alias.__pytree_args__ = item  # type: ignore[attr-defined]
        return pytree_alias

    def __new__(cls) -> NoReturn:  # pylint: disable=arguments-differ
        """Prohibit instantiation."""
        raise TypeError('Cannot instantiate special typing classes.')

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError('Cannot subclass special typing classes.')

    def __copy__(self) -> PyTree:
        """Immutable copy."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> PyTree:
        """Immutable copy."""
        return self


class PyTreeTypeVar:  # pragma: no cover
    """Type variable for PyTree.

    >>> import torch
    >>> TensorTree = PyTreeTypeVar('TensorTree', torch.Tensor)
    >>> TensorTree  # doctest: +IGNORE_WHITESPACE
    typing.Union[torch.Tensor,
                 typing.Tuple[ForwardRef('TensorTree'), ...],
                 typing.List[ForwardRef('TensorTree')],
                 typing.Dict[typing.Any, ForwardRef('TensorTree')],
                 typing.Deque[ForwardRef('TensorTree')],
                 optree.typing.CustomTreeNode[ForwardRef('TensorTree')]]
    """

    @_tp_cache
    def __new__(cls, name: str, param: type) -> TypeAlias:
        """Instantiate a PyTree type variable with the given name and parameter."""
        if not isinstance(name, str):
            raise TypeError(f'{cls.__name__} only supports a string of type name, got {name!r}.')
        return PyTree[param, name]  # type: ignore[misc,valid-type]

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError('Cannot subclass special typing classes.')

    def __copy__(self) -> TypeAlias:
        """Immutable copy."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> TypeAlias:
        """Immutable copy."""
        return self


FlattenFunc: TypeAlias = Callable[
    [CustomTreeNode[T]],
    Union[
        Tuple[Children[T], MetaData],
        Tuple[Children[T], MetaData, Optional[Iterable[Any]]],
    ],
]
UnflattenFunc: TypeAlias = Callable[[MetaData, Children[T]], CustomTreeNode[T]]


def _override_with_(cxx_implementation: F) -> Callable[[F], F]:
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

    def wrapper(python_implementation: F) -> F:
        @functools.wraps(python_implementation)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return cxx_implementation(*args, **kwargs)

        wrapped.__cxx_implementation__ = cxx_implementation  # type: ignore[attr-defined]
        wrapped.__python_implementation__ = python_implementation  # type: ignore[attr-defined]

        return wrapped  # type: ignore[return-value]

    return wrapper


@_override_with_(_C.is_namedtuple)
def is_namedtuple(obj: object | type) -> bool:
    """Return whether the object is an instance of namedtuple or a subclass of namedtuple."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_namedtuple_class(cls)


@_override_with_(_C.is_namedtuple_instance)
def is_namedtuple_instance(obj: object) -> bool:
    """Return whether the object is an instance of namedtuple."""
    return is_namedtuple_class(type(obj))


@_override_with_(_C.is_namedtuple_class)
def is_namedtuple_class(cls: type) -> bool:
    """Return whether the class is a subclass of namedtuple."""
    return (
        isinstance(cls, type)
        and issubclass(cls, tuple)
        and isinstance(getattr(cls, '_fields', None), tuple)
        and all(
            type(field) is str  # pylint: disable=unidiomatic-typecheck
            for field in cls._fields  # type: ignore[attr-defined]
        )
        and callable(getattr(cls, '_make', None))
        and callable(getattr(cls, '_asdict', None))
    )


@_override_with_(_C.namedtuple_fields)
def namedtuple_fields(obj: tuple | type[tuple]) -> tuple[str, ...]:
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


class _StructSequenceMeta(type):
    def __subclasscheck__(cls, subclass: type) -> bool:
        """Return whether the class is a PyStructSequence type.

        >>> import time
        >>> issubclass(time.struct_time, structseq)
        True
        >>> class MyTuple(tuple):
        ...     n_fields = 2
        ...     n_sequence_fields = 2
        ...     n_unnamed_fields = 0
        >>> issubclass(MyTuple, structseq)
        False
        """
        return is_structseq_class(subclass)

    def __instancecheck__(cls, instance: Any) -> bool:
        """Return whether the object is a PyStructSequence instance.

        >>> import sys
        >>> isinstance(sys.float_info, structseq)
        True
        >>> isinstance((1, 2), structseq)
        False
        """
        return is_structseq_instance(instance)


# Reference: https://github.com/python/typeshed/blob/main/stdlib/_typeshed/__init__.pyi
# This is an internal CPython type that is like, but subtly different from a NamedTuple.
# `structseq` classes are unsubclassable, so are all decorated with `@final`.
# pylint: disable-next=invalid-name,missing-class-docstring
class structseq(Tuple[_T_co], metaclass=_StructSequenceMeta):  # type: ignore[misc] # noqa: N801
    """A generic type stub for CPython's ``PyStructSequence`` type."""

    n_fields: Final[int]  # type: ignore[misc] # pylint: disable=invalid-name
    n_sequence_fields: Final[int]  # type: ignore[misc] # pylint: disable=invalid-name
    n_unnamed_fields: Final[int]  # type: ignore[misc] # pylint: disable=invalid-name

    def __init_subclass__(cls) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError("type 'structseq' is not an acceptable base type")

    # pylint: disable-next=unused-argument,redefined-builtin
    def __new__(cls, sequence: Iterable[_T_co], dict: dict[str, Any] = ...) -> Self:
        raise NotImplementedError


del _StructSequenceMeta


@_override_with_(_C.is_structseq)
def is_structseq(obj: object | type) -> bool:
    """Return whether the object is an instance of PyStructSequence or a class of PyStructSequence."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_structseq_class(cls)


@_override_with_(_C.is_structseq_instance)
def is_structseq_instance(obj: object) -> bool:
    """Return whether the object is an instance of PyStructSequence."""
    return is_structseq_class(type(obj))


# Set if the type allows subclassing (see CPython's Include/object.h)
Py_TPFLAGS_BASETYPE: int = _C.Py_TPFLAGS_BASETYPE  # (1UL << 10)


@_override_with_(_C.is_structseq_class)
def is_structseq_class(cls: type) -> bool:
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
        if platform.python_implementation() == 'PyPy':
            try:
                types.new_class('subclass', bases=(cls,))
            except (AssertionError, TypeError):
                return True
            return False
        return not bool(cls.__flags__ & Py_TPFLAGS_BASETYPE)
    return False


@_override_with_(_C.structseq_fields)
def structseq_fields(obj: tuple | type[tuple]) -> tuple[str, ...]:
    """Return the field names of a PyStructSequence."""
    if isinstance(obj, type):
        cls = obj
        if not is_structseq_class(cls):
            raise TypeError(f'Expected a PyStructSequence type, got {cls!r}.')
    else:
        cls = type(obj)
        if not is_structseq_class(cls):
            raise TypeError(f'Expected an instance of PyStructSequence type, got {obj!r}.')

    if platform.python_implementation() == 'PyPy':
        # pylint: disable-next=import-error,import-outside-toplevel
        from _structseq import structseqfield

        indices_by_name = {
            name: member.index
            for name, member in vars(cls).items()
            if isinstance(member, structseqfield)
        }
        fields = sorted(indices_by_name, key=indices_by_name.get)  # type: ignore[arg-type]
    else:
        fields = [
            name
            for name, member in vars(cls).items()
            if isinstance(member, types.MemberDescriptorType)
        ]
    return tuple(fields[: cls.n_sequence_fields])  # type: ignore[attr-defined]


del _tp_cache
