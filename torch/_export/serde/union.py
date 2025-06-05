import functools
from collections.abc import Hashable
from dataclasses import fields
from typing import Any


class _UnionTag(str):
    __slots__ = ("_cls",)
    _cls: Hashable

    @staticmethod
    def create(t: str, cls: type) -> "_UnionTag":
        tag = _UnionTag(t)
        assert not hasattr(tag, "_cls")
        tag._cls = cls
        return tag

    def __eq__(self, cmp: object) -> bool:
        assert isinstance(cmp, str)
        other = str(cmp)
        assert other in _get_field_names(
            self._cls
        ), f"{other} is not a valid tag for {self._cls}. Available tags: {_get_field_names(self._cls)}"
        return str(self) == other

    def __hash__(self) -> int:
        return hash(str(self))


@functools.cache
def _get_field_names(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


class _Union:
    _type: _UnionTag

    @classmethod
    def create(cls, **kwargs: Any) -> "_Union":
        assert len(kwargs) == 1
        obj = cls(**{**{f.name: None for f in fields(cls)}, **kwargs})  # type: ignore[arg-type]
        obj._type = _UnionTag.create(next(iter(kwargs.keys())), cls)
        return obj

    def __post_init__(self) -> None:
        assert not any(f.name in ("type", "_type", "create", "value") for f in fields(self))  # type: ignore[arg-type, misc]

    @property
    def type(self) -> str:
        try:
            return self._type
        except AttributeError as e:
            raise RuntimeError(
                f"Please use {type(self).__name__}.create to instantiate the union type."
            ) from e

    @property
    def value(self) -> Any:
        return getattr(self, self.type)

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if attr is None and name in _get_field_names(type(self)) and name != self.type:  # type: ignore[arg-type]
            raise AttributeError(f"Field {name} is not set.")
        return attr

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.type}={getattr(self, self.type)})"
