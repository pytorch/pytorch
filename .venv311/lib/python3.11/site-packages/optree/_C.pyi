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

# pylint: disable=all

import builtins
import enum
import sys
from collections.abc import Callable, Collection, Iterable, Iterator
from types import MappingProxyType
from typing import Any, ClassVar, Final, final
from typing_extensions import Self  # Python 3.11+

from optree.typing import (
    FlattenFunc,
    MetaData,
    PyTree,
    PyTreeAccessor,
    PyTreeEntry,
    T,
    U,
    UnflattenFunc,
)

# Set if the type allows subclassing (see CPython's Include/object.h)
Py_TPFLAGS_BASETYPE: Final[int]  # (1UL << 10)

# Meta-information during build-time
BUILDTIME_METADATA: Final[MappingProxyType[str, Any]]
PY_VERSION: Final[str]
PY_VERSION_HEX: Final[int]
if sys.implementation.name == 'pypy':  # noqa: PYI002
    PYPY_VERSION: Final[str]
    PYPY_VERSION_NUM: Final[int]
    PYPY_VERSION_HEX: Final[int]
Py_DEBUG: Final[bool]
Py_GIL_DISABLED: Final[bool]
PYBIND11_VERSION_HEX: Final[int]
PYBIND11_INTERNALS_VERSION: Final[int]
PYBIND11_HAS_NATIVE_ENUM: Final[bool]
PYBIND11_HAS_INTERNALS_WITH_SMART_HOLDER_SUPPORT: Final[bool]
PYBIND11_HAS_SUBINTERPRETER_SUPPORT: Final[bool]
GLIBCXX_USE_CXX11_ABI: Final[bool]

@final
class InternalError(SystemError): ...

@final
class PyTreeKind(enum.IntEnum):
    CUSTOM = 0  # a custom type
    LEAF = enum.auto()  # an opaque leaf node
    NONE = enum.auto()  # None
    TUPLE = enum.auto()  # a tuple
    LIST = enum.auto()  # a list
    DICT = enum.auto()  # a dict
    NAMEDTUPLE = enum.auto()  # a collections.namedtuple
    ORDEREDDICT = enum.auto()  # a collections.OrderedDict
    DEFAULTDICT = enum.auto()  # a collections.defaultdict
    DEQUE = enum.auto()  # a collections.deque
    STRUCTSEQUENCE = enum.auto()  # a PyStructSequence

    NUM_KINDS: ClassVar[int]

MAX_RECURSION_DEPTH: Final[int]

@final
class PyTreeSpec:
    num_nodes: int
    num_leaves: int
    num_children: int
    none_is_leaf: bool
    namespace: str
    type: builtins.type | None
    kind: PyTreeKind
    def unflatten(self, leaves: Iterable[T], /) -> PyTree[T]: ...
    def flatten_up_to(self, tree: PyTree[T], /) -> list[PyTree[T]]: ...
    def broadcast_to_common_suffix(self, other: Self, /) -> Self: ...
    def transform(
        self,
        /,
        f_node: Callable[[Self], Self] | None = None,
        f_leaf: Callable[[Self], Self] | None = None,
    ) -> Self: ...
    def compose(self, inner: Self, /) -> Self: ...
    def traverse(
        self,
        leaves: Iterable[T],
        /,
        f_node: Callable[[Collection[U]], U] | None = None,
        f_leaf: Callable[[T], U] | None = None,
    ) -> U: ...
    def walk(
        self,
        leaves: Iterable[T],
        /,
        f_node: Callable[[builtins.type, MetaData, tuple[U, ...]], U] | None = None,
        f_leaf: Callable[[T], U] | None = None,
    ) -> U: ...
    def paths(self, /) -> list[tuple[Any, ...]]: ...
    def accessors(self, /) -> list[PyTreeAccessor]: ...
    def entries(self, /) -> list[Any]: ...
    def entry(self, index: int, /) -> Any: ...
    def children(self, /) -> list[Self]: ...
    def child(self, index: int, /) -> Self: ...
    def one_level(self, /) -> Self | None: ...
    def is_leaf(self, /, *, strict: bool = True) -> bool: ...
    def is_one_level(self, /) -> bool: ...
    def is_prefix(self, other: Self, /, *, strict: bool = False) -> bool: ...
    def is_suffix(self, other: Self, /, *, strict: bool = False) -> bool: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __lt__(self, other: object, /) -> bool: ...
    def __le__(self, other: object, /) -> bool: ...
    def __gt__(self, other: object, /) -> bool: ...
    def __ge__(self, other: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __len__(self, /) -> int: ...

@final
class PyTreeIter(Iterator[T]):
    def __init__(
        self,
        tree: PyTree[T],
        /,
        leaf_predicate: Callable[[T], bool] | None = None,
        none_is_leaf: bool = False,
        namespace: str = '',
    ) -> None: ...
    def __iter__(self, /) -> Self: ...
    def __next__(self, /) -> T: ...

# Functions
def flatten(
    tree: PyTree[T],
    /,
    leaf_predicate: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[T], PyTreeSpec]: ...
def flatten_with_path(
    tree: PyTree[T],
    /,
    leaf_predicate: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[tuple[Any, ...]], list[T], PyTreeSpec]: ...

# Constructors
def make_leaf(
    none_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec: ...
def make_none(
    none_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec: ...
def make_from_collection(
    collection: Collection[PyTreeSpec],
    /,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec: ...

# Utility functions
def is_leaf(
    obj: T,
    /,
    leaf_predicate: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool: ...
def all_leaves(
    iterable: Iterable[T],
    /,
    leaf_predicate: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool: ...
def is_namedtuple(obj: object | type, /) -> bool: ...
def is_namedtuple_instance(obj: object, /) -> bool: ...
def is_namedtuple_class(cls: type, /) -> bool: ...
def namedtuple_fields(obj: tuple | type[tuple], /) -> tuple[str, ...]: ...
def is_structseq(obj: object | type, /) -> bool: ...
def is_structseq_instance(obj: object, /) -> bool: ...
def is_structseq_class(cls: type, /) -> bool: ...
def structseq_fields(obj: tuple | type[tuple], /) -> tuple[str, ...]: ...

# Registration functions
def register_node(
    cls: type[Collection[T]],
    /,
    flatten_func: FlattenFunc[T],
    unflatten_func: UnflattenFunc[T],
    path_entry_type: type[PyTreeEntry],
    namespace: str = '',
) -> None: ...
def unregister_node(
    cls: type,
    /,
    namespace: str = '',
) -> None: ...
def is_dict_insertion_ordered(
    namespace: str = '',
    inherit_global_namespace: bool = True,
) -> bool: ...
def set_dict_insertion_ordered(
    mode: bool,
    /,
    namespace: str = '',
) -> None: ...
