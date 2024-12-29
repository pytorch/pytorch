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

# pylint: disable=all

import builtins
import enum
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from typing_extensions import Self

from optree.typing import (
    CustomTreeNode,
    FlattenFunc,
    MetaData,
    PyTree,
    PyTreeAccessor,
    PyTreeEntry,
    T,
    U,
    UnflattenFunc,
)

class InternalError(RuntimeError): ...

MAX_RECURSION_DEPTH: int

# Set if the type allows subclassing (see CPython's Include/object.h)
Py_TPFLAGS_BASETYPE: int  # (1UL << 10)

GLIBCXX_USE_CXX11_ABI: bool

def flatten(
    tree: PyTree[T],
    leaf_predicate: Callable[[T], bool] | None = None,
    node_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[T], PyTreeSpec]: ...
def flatten_with_path(
    tree: PyTree[T],
    leaf_predicate: Callable[[T], bool] | None = None,
    node_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[tuple[Any, ...]], list[T], PyTreeSpec]: ...
def make_leaf(
    node_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec: ...
def make_none(
    node_is_leaf: bool = False,
    namespace: str = '',  # unused
) -> PyTreeSpec: ...
def make_from_collection(
    collection: CustomTreeNode[PyTreeSpec],
    node_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec: ...
def is_leaf(
    obj: T,
    leaf_predicate: Callable[[T], bool] | None = None,
    node_is_leaf: bool = False,
    namespace: str = '',
) -> bool: ...
def all_leaves(
    iterable: Iterable[T],
    leaf_predicate: Callable[[T], bool] | None = None,
    node_is_leaf: bool = False,
    namespace: str = '',
) -> bool: ...
def is_namedtuple(obj: object | type) -> bool: ...
def is_namedtuple_instance(obj: object) -> bool: ...
def is_namedtuple_class(cls: type) -> bool: ...
def namedtuple_fields(obj: tuple | type[tuple]) -> tuple[str, ...]: ...
def is_structseq(obj: object | type) -> bool: ...
def is_structseq_instance(obj: object) -> bool: ...
def is_structseq_class(cls: type) -> bool: ...
def structseq_fields(obj: tuple | type[tuple]) -> tuple[str, ...]: ...

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

class PyTreeSpec:
    num_nodes: int
    num_leaves: int
    num_children: int
    none_is_leaf: bool
    namespace: str
    type: builtins.type | None
    kind: PyTreeKind
    def unflatten(self, leaves: Iterable[T]) -> PyTree[T]: ...
    def flatten_up_to(self, full_tree: PyTree[T]) -> list[PyTree[T]]: ...
    def broadcast_to_common_suffix(self, other: PyTreeSpec) -> PyTreeSpec: ...
    def compose(self, inner_treespec: PyTreeSpec) -> PyTreeSpec: ...
    def walk(
        self,
        f_node: Callable[[tuple[U, ...], MetaData], U],
        f_leaf: Callable[[T], U] | None,
        leaves: Iterable[T],
    ) -> U: ...
    def paths(self) -> list[tuple[Any, ...]]: ...
    def accessors(self) -> list[PyTreeAccessor]: ...
    def entries(self) -> list[Any]: ...
    def entry(self, index: int) -> Any: ...
    def children(self) -> list[PyTreeSpec]: ...
    def child(self, index: int) -> PyTreeSpec: ...
    def is_leaf(self, strict: bool = True) -> bool: ...
    def is_prefix(self, other: PyTreeSpec, strict: bool = False) -> bool: ...
    def is_suffix(self, other: PyTreeSpec, strict: bool = False) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __len__(self) -> int: ...

class PyTreeIter(Iterator[T]):
    def __init__(
        self,
        tree: PyTree[T],
        leaf_predicate: Callable[[T], bool] | None = None,
        node_is_leaf: bool = False,
        namespace: str = '',
    ) -> None: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> T: ...

def register_node(
    cls: type[CustomTreeNode[T]],
    flatten_func: FlattenFunc,
    unflatten_func: UnflattenFunc,
    path_entry_type: type[PyTreeEntry],
    namespace: str = '',
) -> None: ...
def unregister_node(
    cls: type[CustomTreeNode[T]],
    namespace: str = '',
) -> None: ...
def is_dict_insertion_ordered(
    namespace: str = '',
    inherit_global_namespace: bool = True,
) -> bool: ...
def set_dict_insertion_ordered(
    mode: bool,
    namespace: str = '',
) -> None: ...
