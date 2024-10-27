# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from typing import Union


class IntervalSet:
    @classmethod
    def from_string(cls, s):
        """Return a tuple of intervals, covering the codepoints of characters in `s`.

        >>> IntervalSet.from_string('abcdef0123456789')
        ((48, 57), (97, 102))
        """
        x = cls((ord(c), ord(c)) for c in sorted(s))
        return x.union(x)

    def __init__(self, intervals=()):
        self.intervals = tuple(intervals)
        self.offsets = [0]
        for u, v in self.intervals:
            self.offsets.append(self.offsets[-1] + v - u + 1)
        self.size = self.offsets.pop()
        self._idx_of_zero = self.index_above(ord("0"))
        self._idx_of_Z = min(self.index_above(ord("Z")), len(self) - 1)

    def __len__(self):
        return self.size

    def __iter__(self):
        for u, v in self.intervals:
            yield from range(u, v + 1)

    def __getitem__(self, i):
        if i < 0:
            i = self.size + i
        if i < 0 or i >= self.size:
            raise IndexError(f"Invalid index {i} for [0, {self.size})")
        # Want j = maximal such that offsets[j] <= i

        j = len(self.intervals) - 1
        if self.offsets[j] > i:
            hi = j
            lo = 0
            # Invariant: offsets[lo] <= i < offsets[hi]
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if self.offsets[mid] <= i:
                    lo = mid
                else:
                    hi = mid
            j = lo
        t = i - self.offsets[j]
        u, v = self.intervals[j]
        r = u + t
        assert r <= v
        return r

    def __contains__(self, elem: Union[str, int]) -> bool:
        if isinstance(elem, str):
            elem = ord(elem)
        assert 0 <= elem <= 0x10FFFF
        return any(start <= elem <= end for start, end in self.intervals)

    def __repr__(self):
        return f"IntervalSet({self.intervals!r})"

    def index(self, value: int) -> int:
        for offset, (u, v) in zip(self.offsets, self.intervals):
            if u == value:
                return offset
            elif u > value:
                raise ValueError(f"{value} is not in list")
            if value <= v:
                return offset + (value - u)
        raise ValueError(f"{value} is not in list")

    def index_above(self, value: int) -> int:
        for offset, (u, v) in zip(self.offsets, self.intervals):
            if u >= value:
                return offset
            if value <= v:
                return offset + (value - u)
        return self.size

    def __or__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __and__(self, other):
        return self.intersection(other)

    def __eq__(self, other):
        return isinstance(other, IntervalSet) and (other.intervals == self.intervals)

    def __hash__(self):
        return hash(self.intervals)

    def union(self, other):
        """Merge two sequences of intervals into a single tuple of intervals.

        Any integer bounded by `x` or `y` is also bounded by the result.

        >>> union([(3, 10)], [(1, 2), (5, 17)])
        ((1, 17),)
        """
        assert isinstance(other, type(self))
        x = self.intervals
        y = other.intervals
        if not x:
            return IntervalSet((u, v) for u, v in y)
        if not y:
            return IntervalSet((u, v) for u, v in x)
        intervals = sorted(x + y, reverse=True)
        result = [intervals.pop()]
        while intervals:
            # 1. intervals is in descending order
            # 2. pop() takes from the RHS.
            # 3. (a, b) was popped 1st, then (u, v) was popped 2nd
            # 4. Therefore: a <= u
            # 5. We assume that u <= v and a <= b
            # 6. So we need to handle 2 cases of overlap, and one disjoint case
            #    |   u--v     |   u----v   |       u--v  |
            #    |   a----b   |   a--b     |  a--b       |
            u, v = intervals.pop()
            a, b = result[-1]
            if u <= b + 1:
                # Overlap cases
                result[-1] = (a, max(v, b))
            else:
                # Disjoint case
                result.append((u, v))
        return IntervalSet(result)

    def difference(self, other):
        """Set difference for lists of intervals. That is, returns a list of
        intervals that bounds all values bounded by x that are not also bounded by
        y. x and y are expected to be in sorted order.

        For example difference([(1, 10)], [(2, 3), (9, 15)]) would
        return [(1, 1), (4, 8)], removing the values 2, 3, 9 and 10 from the
        interval.
        """
        assert isinstance(other, type(self))
        x = self.intervals
        y = other.intervals
        if not y:
            return IntervalSet(x)
        x = list(map(list, x))
        i = 0
        j = 0
        result = []
        while i < len(x) and j < len(y):
            # Iterate in parallel over x and y. j stays pointing at the smallest
            # interval in the left hand side that could still overlap with some
            # element of x at index >= i.
            # Similarly, i is not incremented until we know that it does not
            # overlap with any element of y at index >= j.

            xl, xr = x[i]
            assert xl <= xr
            yl, yr = y[j]
            assert yl <= yr

            if yr < xl:
                # The interval at y[j] is strictly to the left of the interval at
                # x[i], so will not overlap with it or any later interval of x.
                j += 1
            elif yl > xr:
                # The interval at y[j] is strictly to the right of the interval at
                # x[i], so all of x[i] goes into the result as no further intervals
                # in y will intersect it.
                result.append(x[i])
                i += 1
            elif yl <= xl:
                if yr >= xr:
                    # x[i] is contained entirely in y[j], so we just skip over it
                    # without adding it to the result.
                    i += 1
                else:
                    # The beginning of x[i] is contained in y[j], so we update the
                    # left endpoint of x[i] to remove this, and increment j as we
                    # now have moved past it. Note that this is not added to the
                    # result as is, as more intervals from y may intersect it so it
                    # may need updating further.
                    x[i][0] = yr + 1
                    j += 1
            else:
                # yl > xl, so the left hand part of x[i] is not contained in y[j],
                # so there are some values we should add to the result.
                result.append((xl, yl - 1))

                if yr + 1 <= xr:
                    # If y[j] finishes before x[i] does, there may be some values
                    # in x[i] left that should go in the result (or they may be
                    # removed by a later interval in y), so we update x[i] to
                    # reflect that and increment j because it no longer overlaps
                    # with any remaining element of x.
                    x[i][0] = yr + 1
                    j += 1
                else:
                    # Every element of x[i] other than the initial part we have
                    # already added is contained in y[j], so we move to the next
                    # interval.
                    i += 1
        # Any remaining intervals in x do not overlap with any of y, as if they did
        # we would not have incremented j to the end, so can be added to the result
        # as they are.
        result.extend(x[i:])
        return IntervalSet(map(tuple, result))

    def intersection(self, other):
        """Set intersection for lists of intervals."""
        assert isinstance(other, type(self)), other
        intervals = []
        i = j = 0
        while i < len(self.intervals) and j < len(other.intervals):
            u, v = self.intervals[i]
            U, V = other.intervals[j]
            if u > V:
                j += 1
            elif U > v:
                i += 1
            else:
                intervals.append((max(u, U), min(v, V)))
                if v < V:
                    i += 1
                else:
                    j += 1
        return IntervalSet(intervals)

    def char_in_shrink_order(self, i: int) -> str:
        # We would like it so that, where possible, shrinking replaces
        # characters with simple ascii characters, so we rejig this
        # bit so that the smallest values are 0, 1, 2, ..., Z.
        #
        # Imagine that numbers are laid out as abc0yyyZ...
        # this rearranges them so that they are laid out as
        # 0yyyZcba..., which gives a better shrinking order.
        if i <= self._idx_of_Z:
            # We want to rewrite the integers [0, n] inclusive
            # to [zero_point, Z_point].
            n = self._idx_of_Z - self._idx_of_zero
            if i <= n:
                i += self._idx_of_zero
            else:
                # We want to rewrite the integers [n + 1, Z_point] to
                # [zero_point, 0] (reversing the order so that codepoints below
                # zero_point shrink upwards).
                i = self._idx_of_zero - (i - n)
                assert i < self._idx_of_zero
            assert 0 <= i <= self._idx_of_Z

        return chr(self[i])

    def index_from_char_in_shrink_order(self, c: str) -> int:
        """
        Inverse of char_in_shrink_order.
        """
        assert len(c) == 1
        i = self.index(ord(c))

        if i <= self._idx_of_Z:
            n = self._idx_of_Z - self._idx_of_zero
            # Rewrite [zero_point, Z_point] to [0, n].
            if self._idx_of_zero <= i <= self._idx_of_Z:
                i -= self._idx_of_zero
                assert 0 <= i <= n
            # Rewrite [zero_point, 0] to [n + 1, Z_point].
            else:
                i = self._idx_of_zero - i + n
                assert n + 1 <= i <= self._idx_of_Z
            assert 0 <= i <= self._idx_of_Z

        return i
