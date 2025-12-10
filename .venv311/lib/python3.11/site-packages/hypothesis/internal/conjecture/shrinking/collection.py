# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections import Counter

from hypothesis.internal.conjecture.shrinking.common import Shrinker
from hypothesis.internal.conjecture.shrinking.ordering import Ordering
from hypothesis.internal.conjecture.utils import identity


class Collection(Shrinker):
    def setup(
        self, *, ElementShrinker, min_size, to_order=identity, from_order=identity
    ):
        self.ElementShrinker = ElementShrinker
        self.to_order = to_order
        self.from_order = from_order
        self.min_size = min_size

    def make_immutable(self, value):
        return tuple(value)

    def short_circuit(self):
        zero = self.from_order(0)
        return self.consider([zero] * self.min_size)

    def left_is_better(self, left, right):
        if len(left) < len(right):
            return True

        # examine elements one by one from the left until an element differs.
        for v1, v2 in zip(left, right, strict=False):
            if self.to_order(v1) == self.to_order(v2):
                continue
            return self.to_order(v1) < self.to_order(v2)

        # equal length and all values were equal by our ordering, so must be equal
        # by our ordering.
        assert list(map(self.to_order, left)) == list(map(self.to_order, right))
        return False

    def run_step(self):
        # try all-zero first; we already considered all-zero-and-smallest in
        # short_circuit.
        zero = self.from_order(0)
        self.consider([zero] * len(self.current))

        # try deleting each element in turn, starting from the back
        # TODO_BETTER_SHRINK: adaptively delete here by deleting larger chunks at once
        # if early deletes succeed. use find_integer. turns O(n) into O(log(n))
        for i in reversed(range(len(self.current))):
            self.consider(self.current[:i] + self.current[i + 1 :])

        # then try reordering
        Ordering.shrink(self.current, self.consider, key=self.to_order)

        # then try minimizing all duplicated elements together simultaneously. This
        # helps in cases like https://github.com/HypothesisWorks/hypothesis/issues/4286
        duplicated = {val for val, count in Counter(self.current).items() if count > 1}
        for val in duplicated:
            self.ElementShrinker.shrink(
                self.to_order(val),
                lambda v: self.consider(
                    tuple(self.from_order(v) if x == val else x for x in self.current)
                ),
            )

        # then try minimizing each element in turn
        for i, val in enumerate(self.current):
            self.ElementShrinker.shrink(
                self.to_order(val),
                lambda v: self.consider(
                    self.current[:i] + (self.from_order(v),) + self.current[i + 1 :]
                ),
            )
