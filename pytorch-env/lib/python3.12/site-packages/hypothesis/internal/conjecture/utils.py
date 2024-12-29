# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import enum
import hashlib
import heapq
import sys
from collections import OrderedDict, abc
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, TypeVar, Union

from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import int_from_bytes
from hypothesis.internal.floats import next_up

if TYPE_CHECKING:
    from hypothesis.internal.conjecture.data import ConjectureData


LABEL_MASK = 2**64 - 1


def calc_label_from_name(name: str) -> int:
    hashed = hashlib.sha384(name.encode()).digest()
    return int_from_bytes(hashed[:8])


def calc_label_from_cls(cls: type) -> int:
    return calc_label_from_name(cls.__qualname__)


def combine_labels(*labels: int) -> int:
    label = 0
    for l in labels:
        label = (label << 1) & LABEL_MASK
        label ^= l
    return label


SAMPLE_IN_SAMPLER_LABEL = calc_label_from_name("a sample() in Sampler")
ONE_FROM_MANY_LABEL = calc_label_from_name("one more from many()")


T = TypeVar("T")


def identity(v: T) -> T:
    return v


def check_sample(
    values: Union[type[enum.Enum], Sequence[T]], strategy_name: str
) -> Sequence[T]:
    if "numpy" in sys.modules and isinstance(values, sys.modules["numpy"].ndarray):
        if values.ndim != 1:
            raise InvalidArgument(
                "Only one-dimensional arrays are supported for sampling, "
                f"and the given value has {values.ndim} dimensions (shape "
                f"{values.shape}).  This array would give samples of array slices "
                "instead of elements!  Use np.ravel(values) to convert "
                "to a one-dimensional array, or tuple(values) if you "
                "want to sample slices."
            )
    elif not isinstance(values, (OrderedDict, abc.Sequence, enum.EnumMeta)):
        raise InvalidArgument(
            f"Cannot sample from {values!r}, not an ordered collection. "
            f"Hypothesis goes to some length to ensure that the {strategy_name} "
            "strategy has stable results between runs. To replay a saved "
            "example, the sampled values must have the same iteration order "
            "on every run - ruling out sets, dicts, etc due to hash "
            "randomization. Most cases can simply use `sorted(values)`, but "
            "mixed types or special values such as math.nan require careful "
            "handling - and note that when simplifying an example, "
            "Hypothesis treats earlier values as simpler."
        )
    if isinstance(values, range):
        return values
    return tuple(values)


class Sampler:
    """Sampler based on Vose's algorithm for the alias method. See
    http://www.keithschwarz.com/darts-dice-coins/ for a good explanation.

    The general idea is that we store a table of triples (base, alternate, p).
    base. We then pick a triple uniformly at random, and choose its alternate
    value with probability p and else choose its base value. The triples are
    chosen so that the resulting mixture has the right distribution.

    We maintain the following invariants to try to produce good shrinks:

    1. The table is in lexicographic (base, alternate) order, so that choosing
       an earlier value in the list always lowers (or at least leaves
       unchanged) the value.
    2. base[i] < alternate[i], so that shrinking the draw always results in
       shrinking the chosen element.
    """

    table: list[tuple[int, int, float]]  # (base_idx, alt_idx, alt_chance)

    def __init__(self, weights: Sequence[float], *, observe: bool = True):
        self.observe = observe

        n = len(weights)
        table: "list[list[int | float | None]]" = [[i, None, None] for i in range(n)]
        total = sum(weights)
        num_type = type(total)

        zero = num_type(0)  # type: ignore
        one = num_type(1)  # type: ignore

        small: "List[int]" = []
        large: "List[int]" = []

        probabilities = [w / total for w in weights]
        scaled_probabilities: "List[float]" = []

        for i, alternate_chance in enumerate(probabilities):
            scaled = alternate_chance * n
            scaled_probabilities.append(scaled)
            if scaled == 1:
                table[i][2] = zero
            elif scaled < 1:
                small.append(i)
            else:
                large.append(i)
        heapq.heapify(small)
        heapq.heapify(large)

        while small and large:
            lo = heapq.heappop(small)
            hi = heapq.heappop(large)

            assert lo != hi
            assert scaled_probabilities[hi] > one
            assert table[lo][1] is None
            table[lo][1] = hi
            table[lo][2] = one - scaled_probabilities[lo]
            scaled_probabilities[hi] = (
                scaled_probabilities[hi] + scaled_probabilities[lo]
            ) - one

            if scaled_probabilities[hi] < 1:
                heapq.heappush(small, hi)
            elif scaled_probabilities[hi] == 1:
                table[hi][2] = zero
            else:
                heapq.heappush(large, hi)
        while large:
            table[large.pop()][2] = zero
        while small:
            table[small.pop()][2] = zero

        self.table: "list[tuple[int, int, float]]" = []
        for base, alternate, alternate_chance in table:  # type: ignore
            assert isinstance(base, int)
            assert isinstance(alternate, int) or alternate is None
            if alternate is None:
                self.table.append((base, base, alternate_chance))
            elif alternate < base:
                self.table.append((alternate, base, one - alternate_chance))
            else:
                self.table.append((base, alternate, alternate_chance))
        self.table.sort()

    def sample(
        self,
        data: "ConjectureData",
        *,
        forced: Optional[int] = None,
        fake_forced: bool = False,
    ) -> int:
        if self.observe:
            data.start_example(SAMPLE_IN_SAMPLER_LABEL)
        forced_choice = (  # pragma: no branch # https://github.com/nedbat/coveragepy/issues/1617
            None
            if forced is None
            else next(
                (base, alternate, alternate_chance)
                for (base, alternate, alternate_chance) in self.table
                if forced == base or (forced == alternate and alternate_chance > 0)
            )
        )
        base, alternate, alternate_chance = data.choice(
            self.table,
            forced=forced_choice,
            fake_forced=fake_forced,
            observe=self.observe,
        )
        forced_use_alternate = None
        if forced is not None:
            # we maintain this invariant when picking forced_choice above.
            # This song and dance about alternate_chance > 0 is to avoid forcing
            # e.g. draw_boolean(p=0, forced=True), which is an error.
            forced_use_alternate = forced == alternate and alternate_chance > 0
            assert forced == base or forced_use_alternate

        use_alternate = data.draw_boolean(
            alternate_chance,
            forced=forced_use_alternate,
            fake_forced=fake_forced,
            observe=self.observe,
        )
        if self.observe:
            data.stop_example()
        if use_alternate:
            assert forced is None or alternate == forced, (forced, alternate)
            return alternate
        else:
            assert forced is None or base == forced, (forced, base)
            return base


INT_SIZES = (8, 16, 32, 64, 128)
INT_SIZES_SAMPLER = Sampler((4.0, 8.0, 1.0, 1.0, 0.5), observe=False)


class many:
    """Utility class for collections. Bundles up the logic we use for "should I
    keep drawing more values?" and handles starting and stopping examples in
    the right place.

    Intended usage is something like:

    elements = many(data, ...)
    while elements.more():
        add_stuff_to_result()
    """

    def __init__(
        self,
        data: "ConjectureData",
        min_size: int,
        max_size: Union[int, float],
        average_size: Union[int, float],
        *,
        forced: Optional[int] = None,
        fake_forced: bool = False,
        observe: bool = True,
    ) -> None:
        assert 0 <= min_size <= average_size <= max_size
        assert forced is None or min_size <= forced <= max_size
        self.min_size = min_size
        self.max_size = max_size
        self.data = data
        self.forced_size = forced
        self.fake_forced = fake_forced
        self.p_continue = _calc_p_continue(average_size - min_size, max_size - min_size)
        self.count = 0
        self.rejections = 0
        self.drawn = False
        self.force_stop = False
        self.rejected = False
        self.observe = observe

    def stop_example(self):
        if self.observe:
            self.data.stop_example()

    def start_example(self, label):
        if self.observe:
            self.data.start_example(label)

    def more(self) -> bool:
        """Should I draw another element to add to the collection?"""
        if self.drawn:
            self.stop_example()

        self.drawn = True
        self.rejected = False

        self.start_example(ONE_FROM_MANY_LABEL)
        if self.min_size == self.max_size:
            # if we have to hit an exact size, draw unconditionally until that
            # point, and no further.
            should_continue = self.count < self.min_size
        else:
            forced_result = None
            if self.force_stop:
                # if our size is forced, we can't reject in a way that would
                # cause us to differ from the forced size.
                assert self.forced_size is None or self.count == self.forced_size
                forced_result = False
            elif self.count < self.min_size:
                forced_result = True
            elif self.count >= self.max_size:
                forced_result = False
            elif self.forced_size is not None:
                forced_result = self.count < self.forced_size
            should_continue = self.data.draw_boolean(
                self.p_continue,
                forced=forced_result,
                fake_forced=self.fake_forced,
                observe=self.observe,
            )

        if should_continue:
            self.count += 1
            return True
        else:
            self.stop_example()
            return False

    def reject(self, why: Optional[str] = None) -> None:
        """Reject the last example (i.e. don't count it towards our budget of
        elements because it's not going to go in the final collection)."""
        assert self.count > 0
        self.count -= 1
        self.rejections += 1
        self.rejected = True
        # We set a minimum number of rejections before we give up to avoid
        # failing too fast when we reject the first draw.
        if self.rejections > max(3, 2 * self.count):
            if self.count < self.min_size:
                self.data.mark_invalid(why)
            else:
                self.force_stop = True


SMALLEST_POSITIVE_FLOAT: float = next_up(0.0) or sys.float_info.min


@lru_cache
def _calc_p_continue(desired_avg: float, max_size: int) -> float:
    """Return the p_continue which will generate the desired average size."""
    assert desired_avg <= max_size, (desired_avg, max_size)
    if desired_avg == max_size:
        return 1.0
    p_continue = 1 - 1.0 / (1 + desired_avg)
    if p_continue == 0 or max_size == float("inf"):
        assert 0 <= p_continue < 1, p_continue
        return p_continue
    assert 0 < p_continue < 1, p_continue
    # For small max_size, the infinite-series p_continue is a poor approximation,
    # and while we can't solve the polynomial a few rounds of iteration quickly
    # gets us a good approximate solution in almost all cases (sometimes exact!).
    while _p_continue_to_avg(p_continue, max_size) > desired_avg:
        # This is impossible over the reals, but *can* happen with floats.
        p_continue -= 0.0001
        # If we've reached zero or gone negative, we want to break out of this loop,
        # and do so even if we're on a system with the unsafe denormals-are-zero flag.
        # We make that an explicit error in st.floats(), but here we'd prefer to
        # just get somewhat worse precision on collection lengths.
        if p_continue < SMALLEST_POSITIVE_FLOAT:
            p_continue = SMALLEST_POSITIVE_FLOAT
            break
    # Let's binary-search our way to a better estimate!  We tried fancier options
    # like gradient descent, but this is numerically stable and works better.
    hi = 1.0
    while desired_avg - _p_continue_to_avg(p_continue, max_size) > 0.01:
        assert 0 < p_continue < hi, (p_continue, hi)
        mid = (p_continue + hi) / 2
        if _p_continue_to_avg(mid, max_size) <= desired_avg:
            p_continue = mid
        else:
            hi = mid
    assert 0 < p_continue < 1, p_continue
    assert _p_continue_to_avg(p_continue, max_size) <= desired_avg
    return p_continue


def _p_continue_to_avg(p_continue: float, max_size: int) -> float:
    """Return the average_size generated by this p_continue and max_size."""
    if p_continue >= 1:
        return max_size
    return (1.0 / (1 - p_continue) - 1) * (1 - p_continue**max_size)
