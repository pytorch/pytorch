"""
An OrderedSet is a custom MutableSet that remembers its order, so that every
entry has an index that can be looked up. It can also act like a Sequence.

Based on a recipe originally posted to ActiveState Recipes by Raymond Hettiger,
and released under the MIT license.
"""
import itertools as it
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableSet,
    AbstractSet,
    Sequence,
    Set,
    TypeVar,
    Union,
    overload,
)

SLICE_ALL = slice(None)
__version__ = "4.1.0"


T = TypeVar("T")

# SetLike[T] is either a set of elements of type T, or a sequence, which
# we will convert to an OrderedSet by adding its elements in order.
SetLike = Union[AbstractSet[T], Sequence[T]]
OrderedSetInitializer = Union[AbstractSet[T], Sequence[T], Iterable[T]]


def _is_atomic(obj: Any) -> bool:
    """
    Returns True for objects which are iterable but should not be iterated in
    the context of indexing an OrderedSet.

    When we index by an iterable, usually that means we're being asked to look
    up a list of things.

    However, in the case of the .index() method, we shouldn't handle strings
    and tuples like other iterables. They're not sequences of things to look
    up, they're the single, atomic thing we're trying to find.

    As an example, oset.index('hello') should give the index of 'hello' in an
    OrderedSet of strings. It shouldn't give the indexes of each individual
    character.
    """
    return isinstance(obj, str) or isinstance(obj, tuple)


class OrderedSet(MutableSet[T], Sequence[T]):
    """
    An OrderedSet is a custom MutableSet that remembers its order, so that
    every entry has an index that can be looked up.

    Example:
        >>> OrderedSet([1, 1, 2, 3, 2])
        OrderedSet([1, 2, 3])
    """

    def __init__(self, initial: OrderedSetInitializer[T] = None):
        self.items: List[T] = []
        self.map: Dict[T, int] = {}
        if initial is not None:
            # In terms of duck-typing, the default __ior__ is compatible with
            # the types we use, but it doesn't expect all the types we
            # support as values for `initial`.
            self |= initial  # type: ignore

    def __len__(self):
        """
        Returns the number of unique elements in the ordered set

        Example:
            >>> len(OrderedSet([]))
            0
            >>> len(OrderedSet([1, 2]))
            2
        """
        return len(self.items)

    @overload
    def __getitem__(self, index: slice) -> "OrderedSet[T]":
        ...

    @overload
    def __getitem__(self, index: Sequence[int]) -> List[T]:
        ...

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    # concrete implementation
    def __getitem__(self, index):
        """
        Get the item at a given index.

        If `index` is a slice, you will get back that slice of items, as a
        new OrderedSet.

        If `index` is a list or a similar iterable, you'll get a list of
        items corresponding to those indices. This is similar to NumPy's
        "fancy indexing". The result is not an OrderedSet because you may ask
        for duplicate indices, and the number of elements returned should be
        the number of elements asked for.

        Example:
            >>> oset = OrderedSet([1, 2, 3])
            >>> oset[1]
            2
        """
        if isinstance(index, slice) and index == SLICE_ALL:
            return self.copy()
        elif isinstance(index, Iterable):
            return [self.items[i] for i in index]
        elif isinstance(index, slice) or hasattr(index, "__index__"):
            result = self.items[index]
            if isinstance(result, list):
                return self.__class__(result)
            else:
                return result
        else:
            raise TypeError("Don't know how to index an OrderedSet by %r" % index)

    def copy(self) -> "OrderedSet[T]":
        """
        Return a shallow copy of this object.

        Example:
            >>> this = OrderedSet([1, 2, 3])
            >>> other = this.copy()
            >>> this == other
            True
            >>> this is other
            False
        """
        return self.__class__(self)

    # Define the gritty details of how an OrderedSet is serialized as a pickle.
    # We leave off type annotations, because the only code that should interact
    # with these is a generalized tool such as pickle.
    def __getstate__(self):
        if len(self) == 0:
            # In pickle, the state can't be an empty list.
            # We need to return a truthy value, or else __setstate__ won't be run.
            #
            # This could have been done more gracefully by always putting the state
            # in a tuple, but this way is backwards- and forwards- compatible with
            # previous versions of OrderedSet.
            return (None,)
        else:
            return list(self)

    def __setstate__(self, state):
        if state == (None,):
            self.__init__([])
        else:
            self.__init__(state)

    def __contains__(self, key: Any) -> bool:
        """
        Test if the item is in this ordered set.

        Example:
            >>> 1 in OrderedSet([1, 3, 2])
            True
            >>> 5 in OrderedSet([1, 3, 2])
            False
        """
        return key in self.map

    # Technically type-incompatible with MutableSet, because we return an
    # int instead of nothing. This is also one of the things that makes
    # OrderedSet convenient to use.
    def add(self, key: T) -> int:
        """
        Add `key` as an item to this OrderedSet, then return its index.

        If `key` is already in the OrderedSet, return the index it already
        had.

        Example:
            >>> oset = OrderedSet()
            >>> oset.append(3)
            0
            >>> print(oset)
            OrderedSet([3])
        """
        if key not in self.map:
            self.map[key] = len(self.items)
            self.items.append(key)
        return self.map[key]

    append = add

    def update(self, sequence: SetLike[T]) -> int:
        """
        Update the set with the given iterable sequence, then return the index
        of the last element inserted.

        Example:
            >>> oset = OrderedSet([1, 2, 3])
            >>> oset.update([3, 1, 5, 1, 4])
            4
            >>> print(oset)
            OrderedSet([1, 2, 3, 5, 4])
        """
        item_index = 0
        try:
            for item in sequence:
                item_index = self.add(item)
        except TypeError:
            raise ValueError(
                "Argument needs to be an iterable, got %s" % type(sequence)
            )
        return item_index

    @overload
    def index(self, key: Sequence[T]) -> List[int]:
        ...

    @overload
    def index(self, key: T) -> int:
        ...

    # concrete implementation
    def index(self, key):
        """
        Get the index of a given entry, raising an IndexError if it's not
        present.

        `key` can be an iterable of entries that is not a string, in which case
        this returns a list of indices.

        Example:
            >>> oset = OrderedSet([1, 2, 3])
            >>> oset.index(2)
            1
        """
        if isinstance(key, Iterable) and not _is_atomic(key):
            return [self.index(subkey) for subkey in key]
        return self.map[key]

    # Provide some compatibility with pd.Index
    get_loc = index
    get_indexer = index

    def pop(self, index=-1) -> T:
        """
        Remove and return item at index (default last).

        Raises KeyError if the set is empty.
        Raises IndexError if index is out of range.

        Example:
            >>> oset = OrderedSet([1, 2, 3])
            >>> oset.pop()
            3
        """
        if not self.items:
            raise KeyError("Set is empty")

        elem = self.items[index]
        del self.items[index]
        del self.map[elem]
        return elem

    def discard(self, key: T) -> None:
        """
        Remove an element.  Do not raise an exception if absent.

        The MutableSet mixin uses this to implement the .remove() method, which
        *does* raise an error when asked to remove a non-existent item.

        Example:
            >>> oset = OrderedSet([1, 2, 3])
            >>> oset.discard(2)
            >>> print(oset)
            OrderedSet([1, 3])
            >>> oset.discard(2)
            >>> print(oset)
            OrderedSet([1, 3])
        """
        if key in self:
            i = self.map[key]
            del self.items[i]
            del self.map[key]
            for k, v in self.map.items():
                if v >= i:
                    self.map[k] = v - 1

    def clear(self) -> None:
        """
        Remove all items from this OrderedSet.
        """
        del self.items[:]
        self.map.clear()

    def __iter__(self) -> Iterator[T]:
        """
        Example:
            >>> list(iter(OrderedSet([1, 2, 3])))
            [1, 2, 3]
        """
        return iter(self.items)

    def __reversed__(self) -> Iterator[T]:
        """
        Example:
            >>> list(reversed(OrderedSet([1, 2, 3])))
            [3, 2, 1]
        """
        return reversed(self.items)

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))

    def __eq__(self, other: Any) -> bool:
        """
        Returns true if the containers have the same items. If `other` is a
        Sequence, then order is checked, otherwise it is ignored.

        Example:
            >>> oset = OrderedSet([1, 3, 2])
            >>> oset == [1, 3, 2]
            True
            >>> oset == [1, 2, 3]
            False
            >>> oset == [2, 3]
            False
            >>> oset == OrderedSet([3, 2, 1])
            False
        """
        if isinstance(other, Sequence):
            # Check that this OrderedSet contains the same elements, in the
            # same order, as the other object.
            return list(self) == list(other)
        try:
            other_as_set = set(other)
        except TypeError:
            # If `other` can't be converted into a set, it's not equal.
            return False
        else:
            return set(self) == other_as_set

    def union(self, *sets: SetLike[T]) -> "OrderedSet[T]":
        """
        Combines all unique items.
        Each items order is defined by its first appearance.

        Example:
            >>> oset = OrderedSet.union(OrderedSet([3, 1, 4, 1, 5]), [1, 3], [2, 0])
            >>> print(oset)
            OrderedSet([3, 1, 4, 5, 2, 0])
            >>> oset.union([8, 9])
            OrderedSet([3, 1, 4, 5, 2, 0, 8, 9])
            >>> oset | {10}
            OrderedSet([3, 1, 4, 5, 2, 0, 10])
        """
        cls: type = OrderedSet
        if isinstance(self, OrderedSet):
            cls = self.__class__
        containers = map(list, it.chain([self], sets))
        items = it.chain.from_iterable(containers)
        return cls(items)

    def __and__(self, other: SetLike[T]) -> "OrderedSet[T]":
        # the parent implementation of this is backwards
        return self.intersection(other)

    def intersection(self, *sets: SetLike[T]) -> "OrderedSet[T]":
        """
        Returns elements in common between all sets. Order is defined only
        by the first set.

        Example:
            >>> oset = OrderedSet.intersection(OrderedSet([0, 1, 2, 3]), [1, 2, 3])
            >>> print(oset)
            OrderedSet([1, 2, 3])
            >>> oset.intersection([2, 4, 5], [1, 2, 3, 4])
            OrderedSet([2])
            >>> oset.intersection()
            OrderedSet([1, 2, 3])
        """
        cls: type = OrderedSet
        items: OrderedSetInitializer[T] = self
        if isinstance(self, OrderedSet):
            cls = self.__class__
        if sets:
            common = set.intersection(*map(set, sets))
            items = (item for item in self if item in common)
        return cls(items)

    def difference(self, *sets: SetLike[T]) -> "OrderedSet[T]":
        """
        Returns all elements that are in this set but not the others.

        Example:
            >>> OrderedSet([1, 2, 3]).difference(OrderedSet([2]))
            OrderedSet([1, 3])
            >>> OrderedSet([1, 2, 3]).difference(OrderedSet([2]), OrderedSet([3]))
            OrderedSet([1])
            >>> OrderedSet([1, 2, 3]) - OrderedSet([2])
            OrderedSet([1, 3])
            >>> OrderedSet([1, 2, 3]).difference()
            OrderedSet([1, 2, 3])
        """
        cls = self.__class__
        items: OrderedSetInitializer[T] = self
        if sets:
            other = set.union(*map(set, sets))
            items = (item for item in self if item not in other)
        return cls(items)

    def issubset(self, other: SetLike[T]) -> bool:
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

    def issuperset(self, other: SetLike[T]) -> bool:
        """
        Report whether this set contains another set.

        Example:
            >>> OrderedSet([1, 2]).issuperset([1, 2, 3])
            False
            >>> OrderedSet([1, 2, 3, 4]).issuperset({1, 2, 3})
            True
            >>> OrderedSet([1, 4, 3, 5]).issuperset({1, 2, 3})
            False
        """
        if len(self) < len(other):  # Fast check for obvious cases
            return False
        return all(item in self for item in other)

    def symmetric_difference(self, other: SetLike[T]) -> "OrderedSet[T]":
        """
        Return the symmetric difference of two OrderedSets as a new set.
        That is, the new set will contain all elements that are in exactly
        one of the sets.

        Their order will be preserved, with elements from `self` preceding
        elements from `other`.

        Example:
            >>> this = OrderedSet([1, 4, 3, 5, 7])
            >>> other = OrderedSet([9, 7, 1, 3, 2])
            >>> this.symmetric_difference(other)
            OrderedSet([4, 5, 9, 2])
        """
        cls: type = OrderedSet
        if isinstance(self, OrderedSet):
            cls = self.__class__
        diff1 = cls(self).difference(other)
        diff2 = cls(other).difference(self)
        return diff1.union(diff2)

    def _update_items(self, items: list) -> None:
        """
        Replace the 'items' list of this OrderedSet with a new one, updating
        self.map accordingly.
        """
        self.items = items
        self.map = {item: idx for (idx, item) in enumerate(items)}

    def difference_update(self, *sets: SetLike[T]) -> None:
        """
        Update this OrderedSet to remove items from one or more other sets.

        Example:
            >>> this = OrderedSet([1, 2, 3])
            >>> this.difference_update(OrderedSet([2, 4]))
            >>> print(this)
            OrderedSet([1, 3])

            >>> this = OrderedSet([1, 2, 3, 4, 5])
            >>> this.difference_update(OrderedSet([2, 4]), OrderedSet([1, 4, 6]))
            >>> print(this)
            OrderedSet([3, 5])
        """
        items_to_remove = set()  # type: Set[T]
        for other in sets:
            items_as_set = set(other)  # type: Set[T]
            items_to_remove |= items_as_set
        self._update_items([item for item in self.items if item not in items_to_remove])

    def intersection_update(self, other: SetLike[T]) -> None:
        """
        Update this OrderedSet to keep only items in another set, preserving
        their order in this set.

        Example:
            >>> this = OrderedSet([1, 4, 3, 5, 7])
            >>> other = OrderedSet([9, 7, 1, 3, 2])
            >>> this.intersection_update(other)
            >>> print(this)
            OrderedSet([1, 3, 7])
        """
        other = set(other)
        self._update_items([item for item in self.items if item in other])

    def symmetric_difference_update(self, other: SetLike[T]) -> None:
        """
        Update this OrderedSet to remove items from another set, then
        add items from the other set that were not present in this set.

        Example:
            >>> this = OrderedSet([1, 4, 3, 5, 7])
            >>> other = OrderedSet([9, 7, 1, 3, 2])
            >>> this.symmetric_difference_update(other)
            >>> print(this)
            OrderedSet([4, 5, 9, 2])
        """
        items_to_add = [item for item in other if item not in self]
        items_to_remove = set(other)
        self._update_items(
            [item for item in self.items if item not in items_to_remove] + items_to_add
        )
