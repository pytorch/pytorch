# mypy: allow-untyped-defs
import functools
from collections.abc import Hashable
from dataclasses import dataclass, fields
from typing import TypeVar
from typing_extensions import dataclass_transform


T = TypeVar("T", bound="_Union")


class _UnionTag(str):
    __slots__ = ("_cls",)
    _cls: Hashable

    @staticmethod
    def create(t, cls):
        tag = _UnionTag(t)
        if hasattr(tag, "_cls"):
            raise AssertionError("tag already has _cls attribute")
        tag._cls = cls
        return tag

    def __eq__(self, cmp) -> bool:
        if not isinstance(cmp, str):
            raise AssertionError(f"expected str, got {type(cmp)}")
        other = str(cmp)
        if other not in _get_field_names(self._cls):
            raise AssertionError(
                f"{other} is not a valid tag for {self._cls}. Available tags: {_get_field_names(self._cls)}"
            )
        return str(self) == other

    def __hash__(self):
        return hash(str(self))


@functools.cache
def _get_field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


# If you turn a schema class that inherits from union into a dataclass, please use
# this decorator to configure it. It's safe, faster and allows code sharing.
#
# For example, _union_dataclass customizes the __eq__ method to only check the type
# and value property instead of default implementation of dataclass which goes
# through every field in the dataclass.
@dataclass_transform(eq_default=False)
def _union_dataclass(cls: type[T]) -> type[T]:
    if not issubclass(cls, _Union):
        raise AssertionError(f"{cls} must inherit from {_Union}.")
    return dataclass(repr=False, eq=False)(cls)


class _Union:
    _type: _UnionTag

    @classmethod
    def create(cls, **kwargs):
        if len(kwargs) != 1:
            raise AssertionError(f"expected exactly 1 kwarg, got {len(kwargs)}")
        obj = cls(**{**{f.name: None for f in fields(cls)}, **kwargs})  # type: ignore[arg-type]
        obj._type = _UnionTag.create(next(iter(kwargs.keys())), cls)
        return obj

    def __post_init__(self):
        if any(
            f.name in ("type", "_type", "create", "value")
            for f in fields(self)  # type: ignore[arg-type, misc]
        ):
            raise AssertionError(
                "field names 'type', '_type', 'create', 'value' are reserved"
            )

    @property
    def type(self) -> str:
        try:
            return self._type
        except AttributeError as e:
            raise RuntimeError(
                f"Please use {type(self).__name__}.create to instantiate the union type."
            ) from e

    @property
    def value(self):
        return getattr(self, self.type)

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if attr is None and name in _get_field_names(type(self)) and name != self.type:  # type: ignore[arg-type]
            raise AttributeError(f"Field {name} is not set.")
        return attr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Union):
            return False
        return self.type == other.type and self.value == other.value

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{type(self).__name__}({self.type}={getattr(self, self.type)})"
