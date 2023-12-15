import math
import random
from typing import Any, List, Optional, Set, Union

from torch.testing._internal.inputgen.variable.constants import (
    BOUND_ON_INF,
    INT64_MAX,
    INT64_MIN,
)
from torch.testing._internal.inputgen.variable.space import (
    Interval,
    Intervals,
    VariableSpace,
)
from torch.testing._internal.inputgen.variable.utils import nextdown, nextup


def gen_min_float_from_interval(r: Interval) -> Optional[float]:
    if r.empty():
        return None
    if not r.lower_open:
        return float(r.lower)
    else:
        next_float = nextup(r.lower)
        if next_float < r.upper or next_float == r.upper and not r.upper_open:
            return float(next_float)
        return None


def gen_max_float_from_interval(r: Interval) -> Optional[float]:
    if r.empty():
        raise ValueError("interval must not be empty")
    if not r.upper_open:
        return float(r.upper)
    else:
        prev_float = nextdown(r.upper)
        if prev_float > r.lower or prev_float == r.lower and not r.lower_open:
            return float(prev_float)
        return None


def gen_float_from_interval(r: Interval) -> Optional[float]:
    if r.empty():
        return None
    lower = gen_min_float_from_interval(r)
    upper = gen_max_float_from_interval(r)
    if lower == -math.inf:
        lower = nextup(lower)
    if upper == math.inf:
        upper = nextdown(upper)
    if lower is None or upper is None:
        return None
    elif lower > upper:
        return None
    else:
        return random.uniform(lower, upper)


def gen_min_float_from_intervals(rs: Intervals) -> Optional[float]:
    if rs.empty():
        return None
    return gen_min_float_from_interval(rs.intervals[0])


def gen_max_float_from_intervals(rs: Intervals) -> Optional[float]:
    if rs.empty():
        return None
    return gen_max_float_from_interval(rs.intervals[-1])


def gen_float_from_intervals(rs: Intervals) -> Optional[float]:
    if rs.empty():
        return None
    r = random.choice(rs.intervals)
    return gen_float_from_interval(r)


def gen_min_int_from_interval(r: Interval) -> Optional[int]:
    if r.empty():
        return None
    if r.lower not in [-math.inf, math.inf]:
        if r.lower > INT64_MAX or (r.lower == INT64_MAX and r.lower_open):
            return None
        lower: int = math.floor(r.lower) + 1 if r.lower_open else math.ceil(r.lower)
        lower = max(INT64_MIN, lower)
        if r.contains(lower):
            return lower
    return None


def gen_max_int_from_interval(r: Interval) -> Optional[int]:
    if r.empty():
        return None
    if r.upper not in [-math.inf, math.inf]:
        if r.upper < INT64_MIN or (r.upper == INT64_MIN and r.upper_open):
            return None
        upper: int = math.ceil(r.upper) - 1 if r.upper_open else math.floor(r.upper)
        upper = min(INT64_MAX, upper)
        if r.contains(upper):
            return upper
    return None


def gen_int_from_interval(r: Interval) -> Optional[int]:
    if r.empty():
        return None
    lower = gen_min_int_from_interval(r)
    upper = gen_max_int_from_interval(r)
    if lower is None and upper is None:
        lower = -BOUND_ON_INF
        upper = BOUND_ON_INF
    elif lower is None:
        lower = upper - BOUND_ON_INF
    elif upper is None:
        upper = lower + BOUND_ON_INF
    assert lower is not None and upper is not None
    return random.randint(lower, upper)


def gen_min_int_from_intervals(rs: Intervals) -> Optional[int]:
    for r in rs.intervals:
        if r.contains_int():
            return gen_min_int_from_interval(r)
    return None


def gen_max_int_from_intervals(rs: Intervals) -> Optional[int]:
    for r in reversed(rs.intervals):
        if r.contains_int():
            return gen_max_int_from_interval(r)
    return None


def gen_int_from_intervals(rs: Intervals) -> Optional[int]:
    intervals_with_ints = [r for r in rs.intervals if r.contains_int()]
    if len(intervals_with_ints) == 0:
        return None
    r = random.choice(intervals_with_ints)
    return gen_int_from_interval(r)


class VariableGenerator:
    """
    A variable generator needs to be initialized with a variable space.
    It is equipped with methods to generate values from that variable space.
    """

    def __init__(self, space: VariableSpace):
        self.vtype = space.vtype
        self.space = space

    def gen_min(self) -> Any:
        """Returns the minimum value of the space."""
        if self.space.empty() or self.vtype not in [bool, int, float]:
            return None
        elif self.space.discrete.initialized:
            return min(self.space.discrete.values)
        elif self.vtype == int:
            return gen_min_int_from_intervals(self.space.intervals)
        elif self.vtype == float:
            return gen_min_float_from_intervals(self.space.intervals)
        else:
            raise Exception("Impossible path")

    def gen_max(self) -> Any:
        """Returns the maximum value of the space."""
        if self.space.empty() or self.vtype not in [bool, int, float]:
            return None
        elif self.space.discrete.initialized:
            return max(self.space.discrete.values)
        elif self.vtype == int:
            return gen_max_int_from_intervals(self.space.intervals)
        elif self.vtype == float:
            return gen_max_float_from_intervals(self.space.intervals)
        else:
            raise Exception("Impossible path")

    def gen_extremes(self) -> Set[Any]:
        """Returns the extreme values of the space."""
        if self.space.empty() or self.vtype not in [bool, int, float]:
            return set()
        elif self.space.discrete.initialized:
            return {min(self.space.discrete.values), max(self.space.discrete.values)}
        elif self.vtype == int:
            return {
                gen_min_int_from_intervals(self.space.intervals),
                gen_max_int_from_intervals(self.space.intervals),
            } - {None}
        elif self.vtype == float:
            return {
                gen_min_float_from_intervals(self.space.intervals),
                gen_max_float_from_intervals(self.space.intervals),
            } - {None}
        else:
            raise Exception("Impossible path")

    def gen_edges(self) -> Set[Any]:
        """Returns the edge values of the space. An edge is an interval boundary."""
        edge_vals: Set[Any] = set()
        if self.space.empty() or self.space.discrete.initialized:
            pass
        elif self.vtype == int:
            for r in self.space.intervals.intervals:
                if r.contains_int():
                    lower = gen_min_int_from_interval(r)
                    if lower is not None:
                        edge_vals.add(lower)
                    upper = gen_max_int_from_interval(r)
                    if upper is not None:
                        edge_vals.add(upper)
        elif self.vtype == float:
            for r in self.space.intervals.intervals:
                edge_vals.add(gen_min_float_from_interval(r))
                edge_vals.add(gen_max_float_from_interval(r))
        else:
            raise Exception("Impossible path")
        return edge_vals

    def gen_edges_non_extreme(self, num: int = 2) -> Set[Any]:
        """Generates edge values that are not extremal."""
        if self.space.empty() or self.space.discrete.initialized:
            return set()
        edges_not_extreme = self.gen_edges() - self.gen_extremes()
        if num >= len(edges_not_extreme):
            return edges_not_extreme
        return set(random.sample(list(edges_not_extreme), num))

    def gen_non_edges(self, num: int = 2) -> Set[Any]:
        """Generates non-edge (or interior) values of the space."""
        if self.space.empty() or self.vtype == bool:
            return set()
        edge_or_extreme_vals = self.gen_edges() | self.gen_extremes()
        vals = set()
        if self.space.discrete.initialized:
            vals = self.space.discrete.values - edge_or_extreme_vals
            if num < len(vals):
                vals = set(random.sample(list(vals), num))
        else:
            for _ in range(100):
                v: Optional[Union[int, float]] = None
                if self.vtype == int:
                    v = gen_int_from_intervals(self.space.intervals)
                else:
                    v = gen_float_from_intervals(self.space.intervals)
                if v is None:
                    continue
                v = self.vtype(v)
                if v not in edge_or_extreme_vals:
                    vals.add(v)
                if len(vals) >= num:
                    break
        return vals

    def gen_balanced(self, num: int = 6) -> Set[Any]:
        """Generates a balanced sample of the space. Balanced, in the sense that
        extremal values, edge values, and interior values are drawn as equally likely
        as possible."""
        if self.space.empty():
            return set()
        extreme_vals = self.gen_extremes()

        if self.space.discrete.initialized:
            num2 = max(2, num - len(extreme_vals))
            interior_vals = self.gen_non_edges(num2)
            balanced = extreme_vals | interior_vals
        else:
            num2 = max(2, math.ceil((num - len(extreme_vals)) / 2))
            edge_non_extreme_vals = self.gen_edges_non_extreme(num2)
            interior_vals = self.gen_non_edges(num2)
            balanced = extreme_vals | edge_non_extreme_vals | interior_vals

        if num >= len(balanced):
            return balanced
        return set(random.sample(list(balanced), num))

    def gen(self, num: int = 6) -> List[Any]:
        """Generates a sorted (if applicable), balanced sample of the space."""
        vals = list(self.gen_balanced(num))
        if self.vtype in [bool, int, float, str]:
            return sorted(vals)
        return vals
