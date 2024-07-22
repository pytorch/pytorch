# This code is backported from python 3.10 dataclasses. Once 3.10 becomes the
# minimum supported we should use dataclass(slots=True) instead.

from __future__ import annotations

import dataclasses
import itertools
from typing import Generator, List, Type, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = ["dataclass_slots"]

_T = TypeVar("_T", bound="DataclassInstance")


def dataclass_slots(cls: Type[_T]) -> Type[DataclassInstance]:
    assert dataclasses.is_dataclass(cls), "Can only be used on dataclasses."

    def _get_slots(cls: Type[DataclassInstance]) -> Generator[str, None, None]:
        slots = cls.__dict__.get("__slots__")
        # `__dictoffset__` and `__weakrefoffset__` can tell us whether
        # the base type has dict/weakref slots, in a way that works correctly
        # for both Python classes and C extension types. Extension types
        # don't use `__slots__` for slot creation
        if slots is None:
            slots = []
            if getattr(cls, "__weakrefoffset__", -1) != 0:
                slots.append("__weakref__")
            if getattr(cls, "__dictrefoffset__", -1) != 0:
                slots.append("__dict__")
            yield from slots
        elif isinstance(slots, str):
            yield slots
        # Slots may be any iterable, but we cannot handle an iterator
        # because it will already be (partially) consumed.
        elif not hasattr(cls, "__next__"):
            yield from slots
        else:
            raise TypeError(f"Slots of '{cls.__name__}' cannot be determined")

    def _add_slots(
        cls: Type[DataclassInstance], is_frozen: bool, weakref_slot: bool
    ) -> Type[DataclassInstance]:
        # Need to create a new class, since we can't set __slots__
        #  after a class has been created.

        # Make sure __slots__ isn't already set.
        if "__slots__" in cls.__dict__:
            raise TypeError(f"{cls.__name__} already specifies __slots__")

        # Create a new dict for our new class.
        cls_dict = dict(cls.__dict__)
        field_names = tuple(f.name for f in dataclasses.fields(cls))
        # Make sure slots don't overlap with those in base classes.
        inherited_slots = set(
            itertools.chain.from_iterable(map(_get_slots, cls.__mro__[1:-1]))
        )
        # The slots for our class.  Remove slots from our base classes.  Add
        # '__weakref__' if weakref_slot was given, unless it is already present.
        cls_dict["__slots__"] = tuple(
            itertools.filterfalse(
                inherited_slots.__contains__,
                itertools.chain(
                    # gh-93521: '__weakref__' also needs to be filtered out if
                    # already present in inherited_slots
                    field_names,
                    ("__weakref__",) if weakref_slot else (),
                ),
            ),
        )

        for field_name in field_names:
            # Remove our attributes, if present. They'll still be
            #  available in _MARKER.
            cls_dict.pop(field_name, None)

        # Remove __dict__ itself.
        cls_dict.pop("__dict__", None)

        # Clear existing `__weakref__` descriptor, it belongs to a previous type:
        cls_dict.pop("__weakref__", None)  # gh-102069

        # And finally create the class.
        qualname = getattr(cls, "__qualname__", None)
        cls = type(cls.__name__, cls.__bases__, cls_dict)
        if qualname is not None:
            cls.__qualname__ = qualname

        def _dataclass_getstate(self: _T) -> object:
            fields = dataclasses.fields(self)
            return [getattr(self, f.name) for f in fields]

        def _dataclass_setstate(self: _T, state: List[object]) -> None:
            fields = dataclasses.fields(self)
            for field, value in zip(fields, state):
                # use setattr because dataclass may be frozen
                object.__setattr__(self, field.name, value)

        if is_frozen:
            # Need this for pickling frozen classes with slots.
            if "__getstate__" not in cls_dict:
                cls.__getstate__ = _dataclass_getstate  # type: ignore[method-assign, assignment]
            if "__setstate__" not in cls_dict:
                cls.__setstate__ = _dataclass_setstate  # type: ignore[attr-defined]

        return cls

    params = getattr(cls, dataclasses._PARAMS)  # type: ignore[attr-defined]
    weakref_slot = getattr(params, "weakref_slot", False)
    return _add_slots(cls, params.frozen, weakref_slot)
