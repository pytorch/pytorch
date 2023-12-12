import math
import random
from typing import Any, List, Set

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


def gen_min_float_from_interval(r: Interval):
    if r.empty():
        return None
    if not r.lower_open:
        return r.lower
    else:
        next_float = math.nextafter(r.lower, math.inf)
        if next_float < r.upper or next_float == r.upper and not r.upper_open:
            return next_float
        return None


def gen_max_float_from_interval(r: Interval):
    if r.empty():
        raise ValueError("interval must not be empty")
    if not r.upper_open:
        return r.upper
    else:
        prev_float = math.nextafter(r.upper, -math.inf)
        if prev_float > r.lower or prev_float == r.lower and not r.lower_open:
            return prev_float
        return None


def gen_float_from_interval(r: Interval):
    if r.empty():
        return None
    lower = gen_min_float_from_interval(r)
    upper = gen_max_float_from_interval(r)
    if lower == float("-inf"):
        lower = math.nextafter(lower, math.inf)
    if upper == float("inf"):
        upper = math.nextafter(upper, -math.inf)
    if lower > upper:
        return None
    return random.uniform(lower, upper)


def gen_min_float_from_intervals(rs: Intervals):
    if rs.empty():
        return None
    return gen_min_float_from_interval(rs.intervals[0])


def gen_max_float_from_intervals(rs: Intervals):
    if rs.empty():
        return None
    return gen_max_float_from_interval(rs.intervals[-1])


def gen_float_from_intervals(rs: Intervals):
    r = random.choice(rs.intervals)
    return gen_float_from_interval(r)


def gen_min_int_from_interval(r: Interval):
    if r.lower not in [float("-inf"), float("inf")]:
        if r.lower > INT64_MAX or (r.lower == INT64_MAX and r.lower_open):
            return None
        lower = math.floor(r.lower) + 1 if r.lower_open else math.ceil(r.lower)
        lower = max(INT64_MIN, lower)
        if r.contains(lower):
            return lower
    return None


def gen_max_int_from_interval(r: Interval):
    if r.upper not in [float("-inf"), float("inf")]:
        if r.upper < INT64_MIN or (r.upper == INT64_MIN and r.upper_open):
            return None
        upper = math.ceil(r.upper) - 1 if r.upper_open else math.floor(r.upper)
        upper = min(INT64_MAX, upper)
        if r.contains(upper):
            return upper
    return None


def gen_int_from_interval(r: Interval):
    if r.lower == float("-inf") and r.upper == float("inf"):
        lower = -BOUND_ON_INF
        upper = BOUND_ON_INF
    elif r.lower == float("-inf"):
        lower = r.upper - BOUND_ON_INF
        upper = gen_max_int_from_interval(r)
    elif r.upper == float("inf"):
        lower = gen_min_int_from_interval(r)
        upper = r.lower + BOUND_ON_INF
    else:
        lower = gen_min_int_from_interval(r)
        upper = gen_max_int_from_interval(r)
    return random.randint(lower, upper)


def gen_min_int_from_intervals(rs: Intervals):
    for r in rs.intervals:
        if r.contains_int():
            return gen_min_int_from_interval(r)
    return None


def gen_max_int_from_intervals(rs: Intervals):
    for r in reversed(rs.intervals):
        if r.contains_int():
            return gen_max_int_from_interval(r)
    return None


def gen_int_from_intervals(rs: Intervals):
    intervals_with_ints = [r for r in rs.intervals if r.contains_int()]
    if len(intervals_with_ints) == 0:
        return None
    r = random.choice(intervals_with_ints)
    return gen_int_from_interval(r)


class VariableGenerator:
    def __init__(self, space: VariableSpace):
        self.vtype = space.vtype
        self.space = space

    def gen_min(self):
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

    def gen_max(self):
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
        if self.space.empty() or self.space.discrete.initialized:
            return set()
        elif self.vtype == int:
            edge_vals = set()
            for r in self.space.intervals.intervals:
                if r.contains_int():
                    lower = gen_min_int_from_interval(r)
                    if lower is not None:
                        edge_vals.add(lower)
                    upper = gen_max_int_from_interval(r)
                    if upper is not None:
                        edge_vals.add(upper)
            return edge_vals
        elif self.vtype == float:
            edge_vals = set()
            for r in self.space.intervals.intervals:
                edge_vals.add(gen_min_float_from_interval(r))
                edge_vals.add(gen_max_float_from_interval(r))
            return edge_vals

    def gen_edges_non_extreme(self, num: int = 2):
        if self.space.empty() or self.space.discrete.initialized:
            return set()
        edges_not_extreme = self.gen_edges() - self.gen_extremes()
        if num >= len(edges_not_extreme):
            return edges_not_extreme
        return set(random.sample(list(edges_not_extreme), num))

    def gen_non_edges(self, num=2) -> List[Any]:
        if self.space.empty() or self.vtype == bool:
            return set()
        edge_or_extreme_vals = self.gen_edges() | self.gen_extremes()
        if self.space.discrete.initialized:
            vals = self.space.discrete.values - edge_or_extreme_vals
            if num >= len(vals):
                return vals
            return set(random.sample(list(vals), num))
        else:
            vals = set()
            for _ in range(100):
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
        vals = list(self.gen_balanced(num))
        if self.vtype in [bool, int, float, str]:
            return sorted(vals)
        return vals
