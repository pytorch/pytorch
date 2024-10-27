import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Generic, TypeVar, overload

if sys.version_info >= (3, 9):
    from types import GenericAlias

_T = TypeVar("_T")

class _MISSING_TYPE: ...

MISSING: _MISSING_TYPE

@overload
def asdict(obj: Any) -> dict[str, Any]: ...
@overload
def asdict(obj: Any, *, dict_factory: Callable[[list[tuple[str, Any]]], _T]) -> _T: ...
@overload
def astuple(obj: Any) -> tuple[Any, ...]: ...
@overload
def astuple(obj: Any, *, tuple_factory: Callable[[list[Any]], _T]) -> _T: ...
@overload
def dataclass(_cls: type[_T]) -> type[_T]: ...
@overload
def dataclass(_cls: None) -> Callable[[type[_T]], type[_T]]: ...
@overload
def dataclass(
    *, init: bool = ..., repr: bool = ..., eq: bool = ..., order: bool = ..., unsafe_hash: bool = ..., frozen: bool = ...
) -> Callable[[type[_T]], type[_T]]: ...

class Field(Generic[_T]):
    name: str
    type: type[_T]
    default: _T
    default_factory: Callable[[], _T]
    repr: bool
    hash: bool | None
    init: bool
    compare: bool
    metadata: Mapping[str, Any]
    if sys.version_info >= (3, 9):
        def __class_getitem__(cls, item: Any) -> GenericAlias: ...

# NOTE: Actual return type is 'Field[_T]', but we want to help type checkers
# to understand the magic that happens at runtime.
@overload  # `default` and `default_factory` are optional and mutually exclusive.
def field(
    *,
    default: _T,
    init: bool = ...,
    repr: bool = ...,
    hash: bool | None = ...,
    compare: bool = ...,
    metadata: Mapping[str, Any] | None = ...,
) -> _T: ...
@overload
def field(
    *,
    default_factory: Callable[[], _T],
    init: bool = ...,
    repr: bool = ...,
    hash: bool | None = ...,
    compare: bool = ...,
    metadata: Mapping[str, Any] | None = ...,
) -> _T: ...
@overload
def field(
    *, init: bool = ..., repr: bool = ..., hash: bool | None = ..., compare: bool = ..., metadata: Mapping[str, Any] | None = ...
) -> Any: ...
def fields(class_or_instance: Any) -> tuple[Field[Any], ...]: ...
def is_dataclass(obj: Any) -> bool: ...

class FrozenInstanceError(AttributeError): ...

class InitVar(Generic[_T]):
    if sys.version_info >= (3, 9):
        def __class_getitem__(cls, type: Any) -> GenericAlias: ...

def make_dataclass(
    cls_name: str,
    fields: Iterable[str | tuple[str, type] | tuple[str, type, Field[Any]]],
    *,
    bases: tuple[type, ...] = ...,
    namespace: dict[str, Any] | None = ...,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
) -> type: ...
def replace(obj: _T, **changes: Any) -> _T: ...
