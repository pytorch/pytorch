from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    MutableSet,
    Reversible,
    Set as AbstractSet,
)
from typing import Any, cast, Optional, TypeVar


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

__all__ = ["OrderedSet", "FrozenOrderedSet"]


class CommonOrderedSet(AbstractSet[T], Reversible[T]):
    """
    A common base class for OrderedSet and FrozenOrderedSet.
    """

    __slots__ = ("_dict",)

    def __init__(self, iterable: Optional[Iterable[T]] = None):
        self._dict = dict.fromkeys(iterable, None) if iterable is not None else {}

    def __contains__(self, elem: object) -> bool:
        return elem in self._dict

    def __iter__(self) -> Iterator[T]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self)})"

    def __getstate__(self) -> list[T]:
        return list(self._dict.keys())

    def __setstate__(self, state: list[T]) -> None:
        self._dict = dict.fromkeys(state, None)

    def __reduce__(self) -> tuple[type[CommonOrderedSet[T]], tuple[list[T]]]:
        return (self.__class__, (list(self),))


class OrderedSet(CommonOrderedSet[T]):
    """
    A mutable set with insertion order.
    """

    def add(self, elem: T) -> None:
        self._dict[elem] = None

    def discard(self, elem: T) -> None:
        self._dict.pop(elem, None)

    def clear(self) -> None:
        # overridden because MutableSet impl is slow
        self._dict.clear()

    def pop(self) -> T:
        if not self:
            raise KeyError("pop from an empty set")
        return self._dict.popitem()[0]

    def update(self, *others: Iterable[T]) -> None:
        for other in others:
            self |= other

    def __ior__(self, other: Iterable[T]) -> OrderedSet[T]:
        if isinstance(other, OrderedSet):
            self._dict.update(other._dict)
            return self
        return super().__ior__(other)  # type: ignore[arg-type]


class FrozenOrderedSet(CommonOrderedSet[T]):
    """
    An immutable set with insertion order.
    """

    def __hash__(self) -> int:
        # Allow the FrozenOrderedSet to be used in hash-based collections
        return hash(tuple(self))

    # FrozenOrderedSet should not have any mutating methods.
    # If called, these will raise an exception.

    def add(self, elem: T) -> None:
        raise NotImplementedError("Cannot add to FrozenOrderedSet")

    def discard(self, elem: T) -> None:
        raise NotImplementedError("Cannot discard from FrozenOrderedSet")

    def clear(self) -> None:
        raise NotImplementedError("Cannot clear a FrozenOrderedSet")

    def pop(self) -> T:
        raise NotImplementedError("Cannot pop from a FrozenOrderedSet")

    def update(self, *others: Iterable[T]) -> None:
        raise NotImplementedError("Cannot update a FrozenOrderedSet")

    def __ior__(self, other: Iterable[T]) -> FrozenOrderedSet[T]:
        raise NotImplementedError("Cannot perform |= on a FrozenOrderedSet")
