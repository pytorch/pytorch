# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis.internal.conjecture.junkdrawer import find_integer
from hypothesis.internal.conjecture.shrinking.common import Shrinker
from hypothesis.internal.conjecture.utils import identity


class Ordering(Shrinker):
    """A shrinker that tries to make a sequence more sorted.

    Will not change the length or the contents, only tries to reorder
    the elements of the sequence.
    """

    def setup(self, key=identity):
        self.key = key

    def make_immutable(self, value):
        return tuple(value)

    def short_circuit(self):
        # If we can flat out sort the target then there's nothing more to do.
        return self.consider(sorted(self.current, key=self.key))

    def left_is_better(self, left, right):
        return tuple(map(self.key, left)) < tuple(map(self.key, right))

    def check_invariants(self, value):
        assert len(value) == len(self.current)
        assert sorted(value) == sorted(self.current)

    def run_step(self):
        self.sort_regions()
        self.sort_regions_with_gaps()

    def sort_regions(self):
        """Guarantees that for each i we have tried to swap index i with
        index i + 1.

        This uses an adaptive algorithm that works by sorting contiguous
        regions starting from each element.
        """
        i = 0
        while i + 1 < len(self.current):
            prefix = list(self.current[:i])
            k = find_integer(
                lambda k: i + k <= len(self.current)
                and self.consider(
                    prefix
                    + sorted(self.current[i : i + k], key=self.key)
                    + list(self.current[i + k :])
                )
            )
            i += k

    def sort_regions_with_gaps(self):
        """Guarantees that for each i we have tried to swap index i with
        index i + 2.

        This uses an adaptive algorithm that works by sorting contiguous
        regions centered on each element, where that element is treated as
        fixed and the elements around it are sorted..
        """
        for i in range(1, len(self.current) - 1):
            if self.current[i - 1] <= self.current[i] <= self.current[i + 1]:
                # The `continue` line is optimised out of the bytecode on
                # CPython >= 3.7 (https://bugs.python.org/issue2506) and on
                # PyPy, and so coverage cannot tell that it has been taken.
                continue  # pragma: no cover

            def can_sort(a, b):
                if a < 0 or b > len(self.current):
                    return False
                assert a <= i < b
                split = i - a
                values = sorted(self.current[a:i] + self.current[i + 1 : b])
                return self.consider(
                    list(self.current[:a])
                    + values[:split]
                    + [self.current[i]]
                    + values[split:]
                    + list(self.current[b:])
                )

            left = i
            right = i + 1
            right += find_integer(lambda k: can_sort(left, right + k))
            find_integer(lambda k: can_sort(left - k, right))
