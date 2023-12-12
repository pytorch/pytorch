import math
from typing import Any, List, Optional, Union

from torch.testing._internal.inputgen.variable.constants import INT64_MAX, INT64_MIN
from torch.testing._internal.inputgen.variable.type import (
    invalid_vtype,
    is_integer,
    ScalarDtype,
    SUPPORTED_TENSOR_DTYPES,
    VariableType,
)


class Discrete:
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
        if not self.initialized:
            raise Exception("Discrete must be initialized before checking if empty")
        return len(self.values) == 0

    def contains(self, v) -> bool:
        if not self.initialized:
            raise Exception("Discrete must be initialized before checking membership")
        return v in self.values

    def remove(self, v) -> None:
        if not self.initialized:
            raise Exception("Discrete must be initialized before removing")
        self.values.difference_update({v})

    def filter(self, f) -> None:
        if not self.initialized:
            raise Exception("Discrete must be initialized before filtering")
        new_values = set()
        for v in list(self.values):
            if f(v):
                new_values.add(v)
        self.values = new_values


class Interval:
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
        if self.lower < self.upper:
            return False
        elif self.lower == self.upper:
            return self.lower_open or self.upper_open
        else:
            return True

    def contains(self, v) -> bool:
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
    def __init__(self, intervals: Optional[List[Interval]] = None):
        self.intervals = intervals
        if self.intervals is None:
            self.intervals = [Interval()]

    def __str__(self) -> str:
        if len(self.intervals) == 0:
            return "{}"
        return " ".join([str(r) for r in self.intervals])

    def empty(self) -> bool:
        return all(r.empty() for r in self.intervals)

    def contains(self, v) -> bool:
        return any(r.contains(v) for r in self.intervals)

    def contains_int(self) -> bool:
        return any(r.contains_int() for r in self.intervals)

    def remove(self, v) -> None:
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

    def set_lower(self, lower: float, lower_open: bool = False) -> None:
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

    def set_upper(self, upper: float, upper_open: bool = False) -> None:
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
    def __init__(self, vtype):
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
        if self.discrete.initialized:
            return self.discrete.empty()
        elif self.vtype == int:
            return not self.intervals.contains_int()
        elif self.vtype == float:
            return self.intervals.empty()
        return False

    def contains(self, v) -> bool:
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

    def remove(self, v) -> None:
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
