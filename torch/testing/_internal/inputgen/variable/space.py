import math
from typing import Any, Callable, List, Optional, Union

from torch.testing._internal.inputgen.variable.constants import INT64_MAX, INT64_MIN
from torch.testing._internal.inputgen.variable.type import (
    invalid_vtype,
    is_integer,
    ScalarDtype,
    SUPPORTED_TENSOR_DTYPES,
    VariableType,
)


class Discrete:
    """
    Representes a set of discrete values. Examples:

    >>> d = Discrete(['a','b','c'])
    >>> d.contains('a')
    True
    >>> d.contains('z')
    False
    >>> d.remove('a')
    >>> str(d)
    "{'b', 'c'}"

    >>> d = Discrete([-4,1,2,3,7,9])
    >>> d.contains(1)
    True
    >>> d.contains(-5)
    False
    >>> d.remove(1)
    >>> str(d)
    '{-4, 2, 3, 7, 9}'
    >>> d.filter(lambda x: x % 2 == 0)
    >>> str(d)
    '{-4, 2}'
    """

    def __init__(self, values: Optional[List[Any]] = None):
        if values is None:
            self.initialized = False
            self.values = set()
        else:
            if any(isinstance(v, float) and math.isnan(v) for v in values):
                raise Exception("NaN values are not supported")
            self.initialized = True
            self.values = set(values)

    def __str__(self) -> str:
        if len(self.values) == 0:
            return "{}"
        return str(self.values)

    def empty(self) -> bool:
        """Returns true iff the set is empty."""
        if not self.initialized:
            raise Exception("Discrete must be initialized before checking if empty")
        return len(self.values) == 0

    def contains(self, v: Any) -> bool:
        """Returns true iff the value is contained in the set."""
        if not self.initialized:
            raise Exception("Discrete must be initialized before checking membership")
        return v in self.values

    def remove(self, v: Any) -> None:
        """Removes the value from the set."""
        if not self.initialized:
            raise Exception("Discrete must be initialized before removing")
        self.values.difference_update({v})

    def filter(self, f: Callable[[Any], bool]) -> None:
        """Filters out all elements that do not satisfy the predicate."""
        if not self.initialized:
            raise Exception("Discrete must be initialized before filtering")
        new_values = set()
        for v in list(self.values):
            if f(v):
                new_values.add(v)
        self.values = new_values


class Interval:
    """
    Represents an interval of real numbers. By default, the interval is closed on both
    ends. Examples:

    >>> i = Interval(1, 3)
    >>> str(i)
    "[1, 3]"
    >>> i.contains(1)
    True
    >>> i.contains(2)
    True
    >>> i.contains(3)
    True
    >>> i.contains(0)
    False
    >>> i.contains(4)
    False
    >>> i.contains_int()
    True

    >>> i = Interval(1, 3, lower_open=True)
    >>> str(i)
    "(1, 3]"
    >>> i.contains(1)
    False

    >>> i = Interval(1, 2, lower_open=True, upper_open=True)
    >>> str(i)
    "(1, 2)"
    >>> i.contains(1)
    False
    >>> i.contains(2)
    False
    >>> i.contains_int()
    False
    """

    def __init__(
        self,
        lower: Union[int, float] = float("-inf"),
        upper: Union[int, float] = float("inf"),
        lower_open: bool = False,
        upper_open: bool = False,
    ):
        self.lower = lower
        self.upper = upper
        self.lower_open = lower_open
        self.upper_open = upper_open

    def __str__(self) -> str:
        lower_braket = "(" if self.lower_open else "["
        upper_braket = ")" if self.upper_open else "]"
        return f"{lower_braket}{self.lower}, {self.upper}{upper_braket}"

    def empty(self) -> bool:
        """Returns true iff the interval is empty."""
        if self.lower < self.upper:
            return False
        elif self.lower == self.upper:
            return self.lower_open or self.upper_open
        else:
            return True

    def contains(self, v: Union[int, float]) -> bool:
        """Returns true iff the value is contained in the interval."""
        if self.empty():
            return False
        if v < self.lower:
            return False
        if v == self.lower and self.lower_open:
            return False
        if v == self.upper and self.upper_open:
            return False
        if v > self.upper:
            return False
        return True

    def contains_int(self) -> bool:
        """Returns true iff the interval contains at least one integer."""
        if self.empty():
            return False
        if self.lower > INT64_MAX or (self.lower == INT64_MAX and self.lower_open):
            return False
        if self.upper < INT64_MIN or (self.upper == INT64_MIN and self.upper_open):
            return False
        if self.upper - self.lower > 1:
            return True
        return self.contains(math.ceil(self.lower)) or self.contains(
            math.floor(self.upper)
        )


class Intervals:
    """
    Represents an ordered sequence of disjoint intervals. It defaults to [-inf, inf].
    Examples:

    >>> i = Intervals()
    >>> str(i)
    "[-inf, inf]"
    >>> i.contains(float('inf'))
    True
    >>> i.remove(float('inf'))
    >>> str(i)
    "[-inf, inf)"
    >>> i.contains(float('inf'))
    False

    >>> i = Intervals([Interval(1, 3), Interval(5, 7)])
    >>> str(i)
    "[1, 3] [5, 7]"
    >>> i.set_lower(7, lower_open=True)
    >>> str(i)
    "{}"
    >>> i.empty()
    True

    >>> i = Intervals([Interval(1, 3), Interval(5, 7, lower_open=True)])
    >>> str(i)
    "[1, 3] (5, 7]"
    >>> i.contains(4)
    False
    >>> i.contains(5)
    False
    >>> i.contains_int()
    True
    >>> i.set_lower(2, lower_open=True)
    >>> str(i)
    "(2, 3] (5, 7]"
    >>> i.set_upper(6, upper_open=True)
    >>> str(i)
    "(2, 3] (5, 6)"
    >>> i.remove(3)
    >>> str(i)
    "(2, 3) (5, 6)"
    >>> i.contains_int()
    False
    """

    def __init__(self, intervals: Optional[List[Interval]] = None):
        self.intervals = [Interval()] if intervals is None else intervals

    def __str__(self) -> str:
        if len(self.intervals) == 0:
            return "{}"
        return " ".join([str(r) for r in self.intervals])

    def empty(self) -> bool:
        """Returns true iff the intervals are empty."""
        return all(r.empty() for r in self.intervals)

    def contains(self, v: Union[int, float]) -> bool:
        """Returns true iff the value is contained within one of the intervals."""
        return any(r.contains(v) for r in self.intervals)

    def contains_int(self) -> bool:
        """Returns true iff some integer is contained within one of the intervals."""
        return any(r.contains_int() for r in self.intervals)

    def remove(self, v: Union[int, float]) -> None:
        """Removes the value from the intervals."""
        for ix, r in enumerate(self.intervals):
            if r.lower <= v <= r.upper:
                if r.lower == v and r.upper == v:
                    self.intervals = self.intervals[:ix] + self.intervals[ix + 1 :]
                elif r.lower == v:
                    r.lower_open = True
                elif r.upper == v:
                    r.upper_open = True
                else:
                    new_intervals = [
                        Interval(r.lower, v, r.lower_open, True),
                        Interval(v, r.upper, True, r.upper_open),
                    ]
                    self.intervals = (
                        self.intervals[:ix] + new_intervals + self.intervals[ix + 1 :]
                    )
                return

    def set_lower(self, lower: Union[int, float], lower_open: bool = False) -> None:
        """Sets the lower bound, being open or closed depending on the flag. In other
        words, it removes all values less than the given value. It also removes the value
        itself if lower_open is True."""
        for ix, r in enumerate(self.intervals):
            if r.upper < lower or r.upper == lower and (r.upper_open or lower_open):
                continue
            if r.lower < lower:
                r.lower = lower
                r.lower_open = lower_open
            elif r.lower == lower and lower_open:
                r.lower_open = True
            self.intervals = self.intervals[ix:]
            return
        self.intervals = []

    def set_upper(self, upper: Union[int, float], upper_open: bool = False) -> None:
        """Sets the upper bound, being open or closed depending on the flag. In other
        words, it removes all values greather than the given value. It also removes the value
        itself if upper_open is True."""
        for ix, r in enumerate(self.intervals):
            if r.upper < upper:
                continue
            if r.lower > upper or (r.lower == upper and (r.lower_open or upper_open)):
                self.intervals = self.intervals[:ix]
                return
            if r.upper == upper and (r.upper_open or upper_open):
                r.upper_open = True
            else:
                r.upper = upper
                r.upper_open = upper_open
            self.intervals = self.intervals[: ix + 1]
            return


class VariableSpace:
    """
    Represents a space of values for a variable of a given type.
    The space can be discrete or continuous. Examples:

    >>> s = VariableSpace(bool)
    >>> str(s)
    '[False, True]'
    >>> s.contains(True)
    True
    >>> s.contains(1)
    True
    >>> s.remove(0)
    >>> str(s)
    '[True]'

    >>> s = VariableSpace(int)
    >>> str(s)
    '(-inf, inf)'
    >>> s.remove(1)
    >>> str(s)
    '(-inf, 1) (1, inf)'

    >>> s = VariableSpace(float)
    >>> str(s)
    '[-inf, inf]'

    >>> s = VariableSpace(ScalarDtype)
    >>> str(s)
    '{ScalarDtype.bool, ScalarDtype.int, ScalarDtype.float}'
    """

    def __init__(self, vtype: type):
        if not VariableType.contains(vtype):
            raise ValueError(f"Unsupported variable type {vtype}")
        self.vtype = vtype

        self.discrete = Discrete()
        self.intervals = Intervals()

        if vtype == VariableType.Bool.value:
            self.discrete = Discrete([False, True])
        if vtype == VariableType.Int.value:
            self.intervals = Intervals(
                [Interval(float("-inf"), float("inf"), True, True)]
            )
        if vtype == VariableType.TensorDtype.value:
            self.discrete = Discrete(SUPPORTED_TENSOR_DTYPES)
        if vtype == VariableType.ScalarDtype.value:
            self.discrete = Discrete(list(ScalarDtype))

    def __str__(self) -> str:
        if self.discrete.initialized:
            return str(self.discrete)
        else:
            return str(self.intervals)

    def empty(self) -> bool:
        """Returns true iff the space is empty."""
        if self.discrete.initialized:
            return self.discrete.empty()
        elif self.vtype == int:
            return not self.intervals.contains_int()
        elif self.vtype == float:
            return self.intervals.empty()
        return False

    def contains(self, v: Any) -> bool:
        """Returns true iff the value is contained in the space."""
        if invalid_vtype(self.vtype, v):
            raise TypeError("Variable type mismatch")
        if self.discrete.initialized:
            return self.discrete.contains(v)
        elif self.vtype == int:
            if is_integer(v):
                return self.intervals.contains(v)
            return False
        elif self.vtype == float:
            return self.intervals.contains(v)
        return True

    def remove(self, v: Any) -> None:
        """Removes the value from the space."""
        if invalid_vtype(self.vtype, v):
            raise TypeError("Variable type mismatch")
        if self.discrete.initialized:
            self.discrete.remove(v)
        elif self.vtype == int:
            if is_integer(v):
                self.intervals.remove(int(v))
        elif self.vtype == float:
            self.intervals.remove(float(v))
        else:
            self.discrete = Discrete([])
