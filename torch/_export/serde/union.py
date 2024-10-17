# mypy: allow-untyped-defs
import functools
from dataclasses import fields
from typing import Hashable, Set


class _UnionTag(str):
    _cls: Hashable

    @staticmethod
    def create(t, cls):
        tag = _UnionTag(t)
        assert not hasattr(tag, "_cls")
        tag._cls = cls
        return tag

    def __eq__(self, cmp) -> bool:
        assert isinstance(cmp, str)
        other = str(cmp)
        assert other in _get_field_names(
            self._cls
        ), f"{other} is not a valid tag for {self._cls}. Available tags: {_get_field_names(self._cls)}"
        return str(self) == other

    def __hash__(self):
        return hash(str(self))


@functools.lru_cache(maxsize=None)
def _get_field_names(cls) -> Set[str]:
    return {f.name for f in fields(cls)}


class _Union:
    _type: _UnionTag

    @classmethod
    def create(cls, **kwargs):
        assert len(kwargs) == 1
        obj = cls(**{**{f.name: None for f in fields(cls)}, **kwargs})  # type: ignore[arg-type]
        obj._type = _UnionTag.create(next(iter(kwargs.keys())), cls)
        return obj

    def __post_init__(self):
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
    def value(self):
        return getattr(self, self.type)

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if attr is None and name in _get_field_names(type(self)) and name != self.type:  # type: ignore[arg-type]
            raise AttributeError(f"Field {name} is not set.")
        return attr

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{type(self).__name__}({self.type}={getattr(self, self.type)})"
