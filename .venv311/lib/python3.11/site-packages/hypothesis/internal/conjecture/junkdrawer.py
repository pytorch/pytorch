# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A module for miscellaneous useful bits and bobs that don't
obviously belong anywhere else. If you spot a better home for
anything that lives here, please move it."""

import array
import gc
import itertools
import sys
import time
import warnings
from array import ArrayType
from collections.abc import Callable, Iterable, Iterator, Sequence
from threading import Lock
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    Union,
    overload,
)

from sortedcontainers import SortedList

from hypothesis.errors import HypothesisWarning

T = TypeVar("T")


def replace_all(
    ls: Sequence[T],
    replacements: Iterable[tuple[int, int, Sequence[T]]],
) -> list[T]:
    """Substitute multiple replacement values into a list.

    Replacements is a list of (start, end, value) triples.
    """

    result: list[T] = []
    prev = 0
    offset = 0
    for u, v, r in replacements:
        result.extend(ls[prev:u])
        result.extend(r)
        prev = v
        offset += len(r) - (v - u)
    result.extend(ls[prev:])
    assert len(result) == len(ls) + offset
    return result


class IntList(Sequence[int]):
    """Class for storing a list of non-negative integers compactly.

    We store them as the smallest size integer array we can get
    away with. When we try to add an integer that is too large,
    we upgrade the array to the smallest word size needed to store
    the new value."""

    ARRAY_CODES: ClassVar[list[str]] = ["B", "H", "I", "L", "Q", "O"]
    NEXT_ARRAY_CODE: ClassVar[dict[str, str]] = dict(itertools.pairwise(ARRAY_CODES))

    __slots__ = ("__underlying",)

    def __init__(self, values: Sequence[int] = ()):
        for code in self.ARRAY_CODES:
            try:
                underlying = self._array_or_list(code, values)
                break
            except OverflowError:
                pass
        else:  # pragma: no cover
            raise AssertionError(f"Could not create storage for {values!r}")
        if isinstance(underlying, list):
            for v in underlying:
                if not isinstance(v, int) or v < 0:
                    raise ValueError(f"Could not create IntList for {values!r}")
        self.__underlying: list[int] | ArrayType[int] = underlying

    @classmethod
    def of_length(cls, n: int) -> "IntList":
        return cls(array.array("B", [0]) * n)

    @staticmethod
    def _array_or_list(
        code: str, contents: Iterable[int]
    ) -> Union[list[int], "ArrayType[int]"]:
        if code == "O":
            return list(contents)
        return array.array(code, contents)

    def count(self, value: int) -> int:
        return self.__underlying.count(value)

    def __repr__(self) -> str:
        return f"IntList({list(self.__underlying)!r})"

    def __len__(self) -> int:
        return len(self.__underlying)

    @overload
    def __getitem__(self, i: int) -> int: ...  # pragma: no cover

    @overload
    def __getitem__(
        self, i: slice
    ) -> "list[int] | ArrayType[int]": ...  # pragma: no cover

    def __getitem__(self, i: int | slice) -> "int | list[int] | ArrayType[int]":
        return self.__underlying[i]

    def __delitem__(self, i: int | slice) -> None:
        del self.__underlying[i]

    def insert(self, i: int, v: int) -> None:
        self.__underlying.insert(i, v)

    def __iter__(self) -> Iterator[int]:
        return iter(self.__underlying)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, IntList):
            return NotImplemented
        return self.__underlying == other.__underlying

    def __ne__(self, other: object) -> bool:
        if self is other:
            return False
        if not isinstance(other, IntList):
            return NotImplemented
        return self.__underlying != other.__underlying

    def append(self, n: int) -> None:
        # try the fast path of appending n first. If this overflows, use the
        # __setitem__ path, which will upgrade the underlying array.
        try:
            self.__underlying.append(n)
        except OverflowError:
            i = len(self.__underlying)
            self.__underlying.append(0)
            self[i] = n

    def __setitem__(self, i: int, n: int) -> None:
        while True:
            try:
                self.__underlying[i] = n
                return
            except OverflowError:
                assert n > 0
                self.__upgrade()

    def extend(self, ls: Iterable[int]) -> None:
        for n in ls:
            self.append(n)

    def __upgrade(self) -> None:
        assert isinstance(self.__underlying, array.array)
        code = self.NEXT_ARRAY_CODE[self.__underlying.typecode]
        self.__underlying = self._array_or_list(code, self.__underlying)


def binary_search(lo: int, hi: int, f: Callable[[int], bool]) -> int:
    """Binary searches in [lo , hi) to find
    n such that f(n) == f(lo) but f(n + 1) != f(lo).
    It is implicitly assumed and will not be checked
    that f(hi) != f(lo).
    """

    reference = f(lo)

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if f(mid) == reference:
            lo = mid
        else:
            hi = mid
    return lo


class LazySequenceCopy(Generic[T]):
    """A "copy" of a sequence that works by inserting a mask in front
    of the underlying sequence, so that you can mutate it without changing
    the underlying sequence. Effectively behaves as if you could do list(x)
    in O(1) time. The full list API is not supported yet but there's no reason
    in principle it couldn't be."""

    def __init__(self, values: Sequence[T]):
        self.__values = values
        self.__len = len(values)
        self.__mask: dict[int, T] | None = None
        self.__popped_indices: SortedList[int] | None = None

    def __len__(self) -> int:
        if self.__popped_indices is None:
            return self.__len
        return self.__len - len(self.__popped_indices)

    def pop(self, i: int = -1) -> T:
        if len(self) == 0:
            raise IndexError("Cannot pop from empty list")
        i = self.__underlying_index(i)

        v = None
        if self.__mask is not None:
            v = self.__mask.pop(i, None)
        if v is None:
            v = self.__values[i]

        if self.__popped_indices is None:
            self.__popped_indices = SortedList()
        self.__popped_indices.add(i)
        return v

    def swap(self, i: int, j: int) -> None:
        """Swap the elements ls[i], ls[j]."""
        if i == j:
            return
        self[i], self[j] = self[j], self[i]

    def __getitem__(self, i: int) -> T:
        i = self.__underlying_index(i)

        default = self.__values[i]
        if self.__mask is None:
            return default
        else:
            return self.__mask.get(i, default)

    def __setitem__(self, i: int, v: T) -> None:
        i = self.__underlying_index(i)
        if self.__mask is None:
            self.__mask = {}
        self.__mask[i] = v

    def __underlying_index(self, i: int) -> int:
        n = len(self)
        if i < -n or i >= n:
            raise IndexError(f"Index {i} out of range [0, {n})")
        if i < 0:
            i += n
        assert 0 <= i < n

        if self.__popped_indices is not None:
            # given an index i in the popped representation of the list, compute
            # its corresponding index in the underlying list. given
            #   l = [1, 4, 2, 10, 188]
            #   l.pop(3)
            #   l.pop(1)
            #   assert l == [1, 2, 188]
            #
            # we want l[i] == self.__values[f(i)], where f is this function.
            assert len(self.__popped_indices) <= len(self.__values)

            for idx in self.__popped_indices:
                if idx > i:
                    break
                i += 1
        return i

    # even though we have len + getitem, mypyc requires iter.
    def __iter__(self) -> Iterable[T]:
        for i in range(len(self)):
            yield self[i]


def stack_depth_of_caller() -> int:
    """Get stack size for caller's frame.

    From https://stackoverflow.com/a/47956089/9297601 , this is a simple
    but much faster alternative to `len(inspect.stack(0))`.  We use it
    with get/set recursionlimit to make stack overflows non-flaky; see
    https://github.com/HypothesisWorks/hypothesis/issues/2494 for details.
    """
    frame = sys._getframe(2)
    size = 1
    while frame:
        frame = frame.f_back  # type: ignore[assignment]
        size += 1
    return size


class StackframeLimiter:
    # StackframeLimiter is used to make the recursion limit warning issued via
    # ensure_free_stackframes thread-safe. We track the known values we have
    # passed to sys.setrecursionlimit in _known_limits, and only issue a warning
    # if sys.getrecursionlimit is not in _known_limits.
    #
    # This will always be an under-approximation of when we would ideally issue
    # this warning, since a non-hypothesis caller could coincidentaly set the
    # recursion limit to one of our known limits. Currently, StackframeLimiter
    # resets _known_limits whenever all of the ensure_free_stackframes contexts
    # have exited. We could increase the power of the warning by tracking a
    # refcount for each limit, and removing it as soon as the refcount hits zero.
    # I didn't think this extra complexity is worth the minor power increase for
    # what is already only a "nice to have" warning.

    def __init__(self):
        self._active_contexts = 0
        self._known_limits: set[int] = set()
        self._original_limit: int | None = None

    def _setrecursionlimit(self, new_limit: int, *, check: bool = True) -> None:
        if (
            check
            and (current_limit := sys.getrecursionlimit()) not in self._known_limits
        ):
            warnings.warn(
                "The recursion limit will not be reset, since it was changed "
                f"during test execution (from {self._original_limit} to {current_limit}).",
                HypothesisWarning,
                stacklevel=4,
            )
            return

        self._known_limits.add(new_limit)
        sys.setrecursionlimit(new_limit)

    def enter_context(self, new_limit: int, *, current_limit: int) -> None:
        if self._active_contexts == 0:
            # this is the first context on the stack. Record the true original
            # limit, to restore later.
            assert self._original_limit is None
            self._original_limit = current_limit
            self._known_limits.add(self._original_limit)

        self._active_contexts += 1
        self._setrecursionlimit(new_limit)

    def exit_context(self, new_limit: int, *, check: bool = True) -> None:
        assert self._active_contexts > 0
        self._active_contexts -= 1

        if self._active_contexts == 0:
            # this is the last context to exit. Restore the true original
            # limit and clear our known limits.
            original_limit = self._original_limit
            assert original_limit is not None
            try:
                self._setrecursionlimit(original_limit, check=check)
            finally:
                self._original_limit = None
                # we want to clear the known limits, but preserve the limit
                # we just set it to as known.
                self._known_limits = {original_limit}
        else:
            self._setrecursionlimit(new_limit, check=check)


_stackframe_limiter = StackframeLimiter()
_stackframe_limiter_lock = Lock()


class ensure_free_stackframes:
    """Context manager that ensures there are at least N free stackframes (for
    a reasonable value of N).
    """

    def __enter__(self) -> None:
        cur_depth = stack_depth_of_caller()
        with _stackframe_limiter_lock:
            self.old_limit = sys.getrecursionlimit()
            # The default CPython recursionlimit is 1000, but pytest seems to bump
            # it to 3000 during test execution. Let's make it something reasonable:
            self.new_limit = cur_depth + 2000
            # Because we add to the recursion limit, to be good citizens we also
            # add a check for unbounded recursion.  The default limit is typically
            # 1000/3000, so this can only ever trigger if something really strange
            # is happening and it's hard to imagine an
            # intentionally-deeply-recursive use of this code.
            assert cur_depth <= 1000, (
                "Hypothesis would usually add %d to the stack depth of %d here, "
                "but we are already much deeper than expected.  Aborting now, to "
                "avoid extending the stack limit in an infinite loop..."
                % (self.new_limit - self.old_limit, self.old_limit)
            )
            try:
                _stackframe_limiter.enter_context(
                    self.new_limit, current_limit=self.old_limit
                )
            except Exception:
                # if the stackframe limiter raises a HypothesisWarning (under eg
                # -Werror), __exit__ is not called, since we errored in __enter__.
                # Preserve the state of the stackframe limiter by exiting, and
                # avoid showing a duplicate warning with check=False.
                _stackframe_limiter.exit_context(self.old_limit, check=False)
                raise

    def __exit__(self, *args, **kwargs):
        with _stackframe_limiter_lock:
            _stackframe_limiter.exit_context(self.old_limit)


def find_integer(f: Callable[[int], bool]) -> int:
    """Finds a (hopefully large) integer such that f(n) is True and f(n + 1) is
    False.

    f(0) is assumed to be True and will not be checked.
    """
    # We first do a linear scan over the small numbers and only start to do
    # anything intelligent if f(4) is true. This is because it's very hard to
    # win big when the result is small. If the result is 0 and we try 2 first
    # then we've done twice as much work as we needed to!
    for i in range(1, 5):
        if not f(i):
            return i - 1

    # We now know that f(4) is true. We want to find some number for which
    # f(n) is *not* true.
    # lo is the largest number for which we know that f(lo) is true.
    lo = 4

    # Exponential probe upwards until we find some value hi such that f(hi)
    # is not true. Subsequently we maintain the invariant that hi is the
    # smallest number for which we know that f(hi) is not true.
    hi = 5
    while f(hi):
        lo = hi
        hi *= 2

    # Now binary search until lo + 1 = hi. At that point we have f(lo) and not
    # f(lo + 1), as desired..
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if f(mid):
            lo = mid
        else:
            hi = mid
    return lo


class NotFound(Exception):
    pass


class SelfOrganisingList(Generic[T]):
    """A self-organising list with the move-to-front heuristic.

    A self-organising list is a collection which we want to retrieve items
    that satisfy some predicate from. There is no faster way to do this than
    a linear scan (as the predicates may be arbitrary), but the performance
    of a linear scan can vary dramatically - if we happen to find a good item
    on the first try it's O(1) after all. The idea of a self-organising list is
    to reorder the list to try to get lucky this way as often as possible.

    There are various heuristics we could use for this, and it's not clear
    which are best. We use the simplest, which is that every time we find
    an item we move it to the "front" (actually the back in our implementation
    because we iterate in reverse) of the list.

    """

    def __init__(self, values: Iterable[T] = ()) -> None:
        self.__values = list(values)

    def __repr__(self) -> str:
        return f"SelfOrganisingList({self.__values!r})"

    def add(self, value: T) -> None:
        """Add a value to this list."""
        self.__values.append(value)

    def find(self, condition: Callable[[T], bool]) -> T:
        """Returns some value in this list such that ``condition(value)`` is
        True. If no such value exists raises ``NotFound``."""
        for i in range(len(self.__values) - 1, -1, -1):
            value = self.__values[i]
            if condition(value):
                del self.__values[i]
                self.__values.append(value)
                return value
        raise NotFound("No values satisfying condition")


_gc_initialized = False
_gc_start: float = 0
_gc_cumulative_time: float = 0

# Since gc_callback potentially runs in test context, and perf_counter
# might be monkeypatched, we store a reference to the real one.
_perf_counter = time.perf_counter


def gc_cumulative_time() -> float:
    global _gc_initialized

    # I don't believe we need a lock for the _gc_cumulative_time increment here,
    # since afaik each gc callback is only executed once when the garbage collector
    # runs, by the thread which initiated the gc.

    if not _gc_initialized:
        if hasattr(gc, "callbacks"):
            # CPython
            def gc_callback(
                phase: Literal["start", "stop"], info: dict[str, int]
            ) -> None:
                global _gc_start, _gc_cumulative_time
                try:
                    now = _perf_counter()
                    if phase == "start":
                        _gc_start = now
                    elif phase == "stop" and _gc_start > 0:
                        _gc_cumulative_time += now - _gc_start  # pragma: no cover # ??
                except RecursionError:  # pragma: no cover
                    # Avoid flakiness via UnraisableException, which is caught and
                    # warned by pytest. The actual callback (this function) is
                    # validated to never trigger a RecursionError itself when
                    # when called by gc.collect.
                    # Anyway, we should hit the same error on "start"
                    # and "stop", but to ensure we don't get out of sync we just
                    # signal that there is no matching start.
                    _gc_start = 0
                    return

            gc.callbacks.insert(0, gc_callback)
        elif hasattr(gc, "hooks"):  # pragma: no cover  # pypy only
            # PyPy
            def hook(stats: Any) -> None:
                global _gc_cumulative_time
                try:
                    _gc_cumulative_time += stats.duration
                except RecursionError:
                    pass

            if gc.hooks.on_gc_minor is None:
                gc.hooks.on_gc_minor = hook
            if gc.hooks.on_gc_collect_step is None:
                gc.hooks.on_gc_collect_step = hook

        _gc_initialized = True

    return _gc_cumulative_time


def startswith(l1: Sequence[T], l2: Sequence[T]) -> bool:
    if len(l1) < len(l2):
        return False
    return all(v1 == v2 for v1, v2 in zip(l1[: len(l2)], l2, strict=False))


def endswith(l1: Sequence[T], l2: Sequence[T]) -> bool:
    if len(l1) < len(l2):
        return False
    return all(v1 == v2 for v1, v2 in zip(l1[-len(l2) :], l2, strict=False))


def bits_to_bytes(n: int) -> int:
    """The number of bytes required to represent an n-bit number.
    Equivalent to (n + 7) // 8, but slightly faster. This really is
    called enough times that that matters."""
    return (n + 7) >> 3
