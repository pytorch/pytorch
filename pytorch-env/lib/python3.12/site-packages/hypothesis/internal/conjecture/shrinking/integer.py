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

"""
This module implements a shrinker for non-negative integers.
"""


class Integer(Shrinker):
    """Attempts to find a smaller integer. Guaranteed things to try ``0``,

    ``1``, ``initial - 1``, ``initial - 2``. Plenty of optimisations beyond
    that but those are the guaranteed ones.
    """

    def short_circuit(self):
        for i in range(2):
            if self.consider(i):
                return True
        self.mask_high_bits()
        if self.size > 8:
            # see if we can squeeze the integer into a single byte.
            self.consider(self.current >> (self.size - 8))
            self.consider(self.current & 0xFF)
        return self.current == 2

    def check_invariants(self, value):
        assert value >= 0

    def left_is_better(self, left, right):
        return left < right

    def run_step(self):
        self.shift_right()
        self.shrink_by_multiples(2)
        self.shrink_by_multiples(1)

    def shift_right(self):
        base = self.current
        find_integer(lambda k: k <= self.size and self.consider(base >> k))

    def mask_high_bits(self):
        base = self.current
        n = base.bit_length()

        @find_integer
        def try_mask(k):
            if k >= n:
                return False
            mask = (1 << (n - k)) - 1
            return self.consider(mask & base)

    @property
    def size(self):
        return self.current.bit_length()

    def shrink_by_multiples(self, k):
        base = self.current

        @find_integer
        def shrunk(n):
            attempt = base - n * k
            return attempt >= 0 and self.consider(attempt)

        return shrunk > 0
