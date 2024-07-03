from __future__ import annotations

from collections.abc import MutableSet, Set as AbstractSet
from typing import cast, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

__all__ = ["OrderedSet"]


class OrderedSet(MutableSet[T]):
    """
    Insertion ordered set, similar to OrderedDict.
    """

    def __init__(self, iterable: Optional[Iterable[T]] = None):
        self._dict: dict[T, None] = {}
        if iterable is not None:
            self.update(iterable)

    #
    # Required overriden abstract methods
    #

    def __contains__(self, elem: object) -> bool:
        return elem in self._dict

    def __iter__(self) -> Iterator[T]:
        yield from self._dict.keys()

    def __len__(self) -> int:
        return len(self._dict)

    def add(self, elem: T) -> None:
        self._dict[elem] = None

    def discard(self, elem: T) -> None:
        self._dict.pop(elem, None)

    def clear(self) -> None:
        # overridden because MutableSet impl is slow
        self._dict.clear()

    # Unimplemented set() methods in _collections_abc.MutableSet

    @classmethod
    def _wrap_in_set(cls, other: object) -> object:
        """
        Wrap non-Set Iterables in OrderedSets

        Some of the magic methods are more strict on input types than
        the public apis, so we need to wrap inputs in sets.
        """

        if not isinstance(other, AbstractSet) and isinstance(other, Iterable):
            return cls(other)
        else:
            return other

    def pop(self) -> T:
        if not self:
            raise KeyError("pop from an empty set")
        return self._dict.popitem()[0]

    def copy(self) -> OrderedSet[T]:
        return self.__class__(self)

    def difference(self, *others: Iterable[T]) -> OrderedSet[T]:
        res = self if len(others) else self.copy()
        for other in others:
            res = res - other  # type: ignore[operator, arg-type]
        return res

    def difference_update(self, *others: Iterable[T]) -> None:
        for other in others:
            self -= other  # type: ignore[operator, arg-type]

    def update(self, *others: Iterable[T]) -> None:
        for other in others:
            self |= other  # type: ignore[operator, arg-type]

    def intersection(self, *others: Iterable[T]) -> OrderedSet[T]:
        res = self if len(others) else self.copy()
        for other in others:
            res = res & other  # type: ignore[operator, arg-type]
        return res

    def intersection_update(self, *others: Iterable[T]) -> None:
        for other in others:
            self &= other  # type: ignore[operator, arg-type]

    def issubset(self, other: Iterable[T]) -> bool:
        return self <= self._wrap_in_set(other)

    def issuperset(self, other: Iterable[T]) -> bool:
        return self >= self._wrap_in_set(other)

    def symmetric_difference(self, other: Iterable[T]) -> OrderedSet[T]:
        return self ^ other  # type: ignore[operator, arg-type]

    def symmetric_difference_update(self, other: Iterable[T]) -> None:
        self ^= other  # type: ignore[operator, arg-type]

    def union(self, *others: Iterable[T]) -> OrderedSet[T]:
        res = self if len(others) else self.copy()
        for other in others:
            res = res | other  # type: ignore[operator, arg-type]
        return res

    # Specify here for correct type inference, otherwise would
    # return AbstractSet[T]
    def __sub__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        return cast(OrderedSet[T], super().__sub__(other))

    def __or__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        return cast(OrderedSet[T], super().__or__(other))

    def __and__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        return cast(OrderedSet[T], super().__and__(other))

    def __xor__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        return cast(OrderedSet[T], super().__xor__(other))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self)})"
