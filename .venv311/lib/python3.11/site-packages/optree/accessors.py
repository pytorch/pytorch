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
"""Access support for pytrees."""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar, overload
from typing_extensions import Self  # Python 3.11+

import optree._C as _C
from optree._C import PyTreeKind


if TYPE_CHECKING:
    import builtins

    from optree.typing import NamedTuple, StructSequence


__all__ = [
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
]


SLOTS = {'slots': True} if sys.version_info >= (3, 10) else {}  # Python 3.10+


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True, **SLOTS)
class PyTreeEntry:
    """Base class for path entries."""

    entry: Any
    type: builtins.type
    kind: PyTreeKind

    def __post_init__(self, /) -> None:
        """Post-initialize the path entry."""
        if self.kind == PyTreeKind.LEAF:
            raise ValueError('Cannot create a leaf path entry.')
        if self.kind == PyTreeKind.NONE:
            raise ValueError('Cannot create a path entry for None.')

    def __call__(self, obj: Any, /) -> Any:
        """Get the child object."""
        try:
            return obj[self.entry]  # should be overridden
        except TypeError as ex:
            raise TypeError(
                f'{self.__class__!r} cannot access through {obj!r} via entry {self.entry!r}',
            ) from ex

    def __add__(self, other: object, /) -> PyTreeAccessor:
        """Join the path entry with another path entry or accessor."""
        if isinstance(other, PyTreeEntry):
            return PyTreeAccessor((self, other))
        if isinstance(other, PyTreeAccessor):
            return PyTreeAccessor((self, *other))
        return NotImplemented

    def __eq__(self, other: object, /) -> bool:
        """Check if the path entries are equal."""
        return isinstance(other, PyTreeEntry) and (
            (
                self.entry,
                self.type,
                self.kind,
                self.__class__.__call__.__code__.co_code,
                self.__class__.codify.__code__.co_code,
            )
            == (
                other.entry,
                other.type,
                other.kind,
                other.__class__.__call__.__code__.co_code,
                other.__class__.codify.__code__.co_code,
            )
        )

    def __hash__(self, /) -> int:
        """Get the hash of the path entry."""
        return hash(
            (
                self.entry,
                self.type,
                self.kind,
                self.__class__.__call__.__code__.co_code,
                self.__class__.codify.__code__.co_code,
            ),
        )

    def __repr__(self, /) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(entry={self.entry!r}, type={self.type!r})'

    def codify(self, /, node: str = '') -> str:
        """Generate code for accessing the path entry."""
        return f'{node}[<flat index {self.entry!r}>]'  # should be overridden


del SLOTS


_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_KT_co = TypeVar('_KT_co', covariant=True)
_VT_co = TypeVar('_VT_co', covariant=True)


class AutoEntry(PyTreeEntry):
    """A generic path entry class that determines the entry type on creation automatically."""

    __slots__: ClassVar[tuple[()]] = ()

    def __new__(  # type: ignore[misc]
        cls,
        /,
        entry: Any,
        type: builtins.type,  # pylint: disable=redefined-builtin
        kind: PyTreeKind,
    ) -> PyTreeEntry:
        """Create a new path entry."""
        # pylint: disable-next=import-outside-toplevel
        from optree.typing import is_namedtuple_class, is_structseq_class

        if cls is not AutoEntry:
            # Use the subclass type if the type is explicitly specified
            return super().__new__(cls)

        if kind != PyTreeKind.CUSTOM:
            raise ValueError(f'Cannot create an automatic path entry for PyTreeKind {kind!r}.')

        # Dispatch the path entry type based on the node type
        path_entry_type: builtins.type[PyTreeEntry]
        if is_structseq_class(type):
            path_entry_type = StructSequenceEntry
        elif is_namedtuple_class(type):
            path_entry_type = NamedTupleEntry
        elif dataclasses.is_dataclass(type):
            path_entry_type = DataclassEntry
        elif issubclass(type, Mapping):
            path_entry_type = MappingEntry
        elif issubclass(type, Sequence):
            path_entry_type = SequenceEntry
        else:
            path_entry_type = FlattenedEntry

        if not issubclass(path_entry_type, AutoEntry):
            # The __init__() method will not be called if the returned instance is not a subtype of
            # AutoEntry. We should return an initialized instance. Return a fully-initialized
            # instance of the dispatched type.
            return path_entry_type(entry, type, kind)

        # The __init__() method will be called if the returned instance is a subtype of AutoEntry.
        # We should return an uninitialized instance. The __init__() method will initialize it.
        # But we will never reach here because the dispatched type is never a subtype of AutoEntry.
        raise NotImplementedError('Unreachable code.')


class GetItemEntry(PyTreeEntry):
    """A generic path entry class for nodes that access their children by :meth:`__getitem__`."""

    __slots__: ClassVar[tuple[()]] = ()

    def __call__(self, obj: Any, /) -> Any:
        """Get the child object."""
        return obj[self.entry]

    def codify(self, /, node: str = '') -> str:
        """Generate code for accessing the path entry."""
        return f'{node}[{self.entry!r}]'


class GetAttrEntry(PyTreeEntry):
    """A generic path entry class for nodes that access their children by :meth:`__getattr__`."""

    __slots__: ClassVar[tuple[()]] = ()

    entry: str

    @property
    def name(self, /) -> str:
        """Get the attribute name."""
        return self.entry

    def __call__(self, obj: Any, /) -> Any:
        """Get the child object."""
        return getattr(obj, self.name)

    def codify(self, /, node: str = '') -> str:
        """Generate code for accessing the path entry."""
        return f'{node}.{self.name}'


class FlattenedEntry(PyTreeEntry):  # pylint: disable=too-few-public-methods
    """A fallback path entry class for flattened objects."""

    __slots__: ClassVar[tuple[()]] = ()


class SequenceEntry(GetItemEntry, Generic[_T_co]):
    """A path entry class for sequences."""

    __slots__: ClassVar[tuple[()]] = ()

    entry: int
    type: builtins.type[Sequence[_T_co]]

    @property
    def index(self, /) -> int:
        """Get the index."""
        return self.entry

    def __call__(self, obj: Sequence[_T_co], /) -> _T_co:
        """Get the child object."""
        return obj[self.index]

    def __repr__(self, /) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(index={self.index!r}, type={self.type!r})'


class MappingEntry(GetItemEntry, Generic[_KT_co, _VT_co]):
    """A path entry class for mappings."""

    __slots__: ClassVar[tuple[()]] = ()

    entry: _KT_co
    type: builtins.type[Mapping[_KT_co, _VT_co]]

    @property
    def key(self, /) -> _KT_co:
        """Get the key."""
        return self.entry

    def __call__(self, obj: Mapping[_KT_co, _VT_co], /) -> _VT_co:
        """Get the child object."""
        return obj[self.key]

    def __repr__(self, /) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(key={self.key!r}, type={self.type!r})'


class NamedTupleEntry(SequenceEntry[_T]):
    """A path entry class for namedtuple objects."""

    __slots__: ClassVar[tuple[()]] = ()

    entry: int
    type: builtins.type[NamedTuple[_T]]  # type: ignore[type-arg]
    kind: Literal[PyTreeKind.NAMEDTUPLE]

    @property
    def fields(self, /) -> tuple[str, ...]:
        """Get the field names."""
        from optree.typing import namedtuple_fields  # pylint: disable=import-outside-toplevel

        return namedtuple_fields(self.type)

    @property
    def field(self, /) -> str:
        """Get the field name."""
        return self.fields[self.entry]

    def __repr__(self, /) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(field={self.field!r}, type={self.type!r})'

    def codify(self, /, node: str = '') -> str:
        """Generate code for accessing the path entry."""
        return f'{node}.{self.field}'


class StructSequenceEntry(SequenceEntry[_T]):
    """A path entry class for PyStructSequence objects."""

    __slots__: ClassVar[tuple[()]] = ()

    entry: int
    type: builtins.type[StructSequence[_T]]
    kind: Literal[PyTreeKind.STRUCTSEQUENCE]

    @property
    def fields(self, /) -> tuple[str, ...]:
        """Get the field names."""
        from optree.typing import structseq_fields  # pylint: disable=import-outside-toplevel

        return structseq_fields(self.type)

    @property
    def field(self, /) -> str:
        """Get the field name."""
        return self.fields[self.entry]

    def __repr__(self, /) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(field={self.field!r}, type={self.type!r})'

    def codify(self, /, node: str = '') -> str:
        """Generate code for accessing the path entry."""
        return f'{node}.{self.field}'


class DataclassEntry(GetAttrEntry):
    """A path entry class for dataclasses."""

    __slots__: ClassVar[tuple[()]] = ()

    entry: str | int  # type: ignore[assignment]

    @property
    def fields(self, /) -> tuple[str, ...]:  # pragma: no cover
        """Get all field names."""
        return tuple(f.name for f in dataclasses.fields(self.type))

    @property
    def init_fields(self, /) -> tuple[str, ...]:
        """Get the init field names."""
        return tuple(f.name for f in dataclasses.fields(self.type) if f.init)

    @property
    def field(self, /) -> str:
        """Get the field name."""
        if isinstance(self.entry, int):
            return self.init_fields[self.entry]
        return self.entry

    @property
    def name(self, /) -> str:
        """Get the attribute name."""
        return self.field

    def __repr__(self, /) -> str:
        """Get the representation of the path entry."""
        return f'{self.__class__.__name__}(field={self.field!r}, type={self.type!r})'


class PyTreeAccessor(tuple[PyTreeEntry, ...]):
    """A path class for PyTrees."""

    __slots__: ClassVar[tuple[()]] = ()

    @property
    def path(self, /) -> tuple[Any, ...]:
        """Get the path of the accessor."""
        return tuple(e.entry for e in self)

    def __new__(cls, /, path: Iterable[PyTreeEntry] = ()) -> Self:
        """Create a new accessor instance."""
        if not isinstance(path, (list, tuple)):
            path = tuple(path)
        if not all(isinstance(p, PyTreeEntry) for p in path):
            raise TypeError(f'Expected a path of PyTreeEntry, got {path!r}.')
        return super().__new__(cls, path)

    def __call__(self, obj: Any, /) -> Any:
        """Get the child object."""
        for entry in self:
            obj = entry(obj)
        return obj

    @overload  # type: ignore[override]
    def __getitem__(self, index: int, /) -> PyTreeEntry: ...

    @overload
    def __getitem__(self, index: slice, /) -> Self: ...

    def __getitem__(self, index: int | slice, /) -> PyTreeEntry | Self:
        """Get the child path entry or an accessor for a subpath."""
        if isinstance(index, slice):
            return self.__class__(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: object, /) -> Self:
        """Join the accessor with another path entry or accessor."""
        if isinstance(other, PyTreeEntry):
            return self.__class__((*self, other))
        if isinstance(other, PyTreeAccessor):
            return self.__class__((*self, *other))
        return NotImplemented

    def __mul__(self, value: int, /) -> Self:  # type: ignore[override]
        """Repeat the accessor."""
        return self.__class__(super().__mul__(value))

    def __rmul__(self, value: int, /) -> Self:  # type: ignore[override]
        """Repeat the accessor."""
        return self.__class__(super().__rmul__(value))

    def __eq__(self, other: object, /) -> bool:
        """Check if the accessors are equal."""
        return isinstance(other, PyTreeAccessor) and super().__eq__(other)

    def __hash__(self, /) -> int:
        """Get the hash of the accessor."""
        return super().__hash__()

    def __repr__(self, /) -> str:
        """Get the representation of the accessor."""
        return f'{self.__class__.__name__}({self.codify()}, {super().__repr__()})'

    def codify(self, /, root: str = '*') -> str:
        """Generate code for accessing the path."""
        string = root
        for entry in self:
            string = entry.codify(string)
        return string


# These classes are used internally in the C++ side for accessor APIs
_name, _cls = '', object
for _name in __all__:
    _cls = globals()[_name]
    if not isinstance(_cls, type):  # pragma: no cover
        raise TypeError(f'Expected a class, got {_cls!r}.')
    _cls.__module__ = 'optree'
    setattr(_C, _name, _cls)
del _name, _cls
