import math
from typing import Any, List, Union

from torch.testing._internal.inputgen.variable.space import Discrete, VariableSpace
from torch.testing._internal.inputgen.variable.type import (
    convert_to_vtype,
    invalid_vtype,
    is_integer,
)


class SolvableVariable:
    def __init__(self, vtype):
        self.vtype = vtype
        self.space = VariableSpace(vtype)

    def Eq(self, v: Any) -> None:
        if invalid_vtype(self.vtype, v):
            raise TypeError("Variable type mismatch")
        if self.space.empty():
            return
        if self.space.contains(v):
            self.space.discrete = Discrete([convert_to_vtype(self.vtype, v)])
        else:
            self.space.discrete = Discrete([])

    def Ne(self, v: Any) -> None:
        if invalid_vtype(self.vtype, v):
            raise TypeError("Variable type mismatch")
        if self.space.empty():
            return
        self.space.remove(v)

    def In(self, values: List[Any]) -> None:
        for v in values:
            if invalid_vtype(self.vtype, v):
                raise TypeError("Variable type mismatch")
        if self.space.empty():
            return
        self.space.discrete = Discrete(
            [convert_to_vtype(self.vtype, v) for v in values if self.space.contains(v)]
        )

    def NotIn(self, values: List[Any]) -> None:
        for v in values:
            if invalid_vtype(self.vtype, v):
                raise TypeError("Variable type mismatch")
        if self.space.empty():
            return
        for v in values:
            self.space.remove(v)

    def Le(self, upper: Union[bool, int, float]) -> None:
        if self.vtype not in [bool, int, float]:
            raise Exception(f"Le is not valid constraint on {self.vtype}")
        if invalid_vtype(self.vtype, upper):
            raise TypeError("Variable type mismatch")
        if self.space.empty():
            return
        elif self.space.discrete.initialized:
            self.space.discrete.filter(lambda v: v <= upper)
        elif self.vtype == int:
            if math.isfinite(upper):
                self.space.intervals.set_upper(math.ceil(upper), upper_open=False)
            else:
                self.space.intervals.set_upper(upper, upper_open=False)
        else:
            self.space.intervals.set_upper(float(upper), upper_open=False)

    def Lt(self, upper: Union[bool, int, float]) -> None:
        if self.vtype not in [bool, int, float]:
            raise Exception(f"Lt is not valid constraint on {self.vtype}")
        if invalid_vtype(self.vtype, upper):
            raise TypeError("Variable type mismatch")
        if self.space.empty():
            return
        elif self.space.discrete.initialized:
            self.space.discrete.filter(lambda v: v < upper)
        elif self.vtype == int:
            if math.isfinite(upper):
                self.space.intervals.set_upper(
                    math.floor(upper), upper_open=is_integer(upper)
                )
            else:
                self.space.intervals.set_upper(upper, upper_open=True)
        else:
            self.space.intervals.set_upper(float(upper), upper_open=True)

    def Ge(self, lower: Union[bool, int, float]) -> None:
        if self.vtype not in [bool, int, float]:
            raise Exception(f"Ge is not valid constraint on {self.vtype}")
        if invalid_vtype(self.vtype, lower):
            raise TypeError("Variable type mismatch")
        if self.space.empty():
            return
        elif self.space.discrete.initialized:
            self.space.discrete.filter(lambda v: v >= lower)
        elif self.vtype == int:
            if math.isfinite(lower):
                self.space.intervals.set_lower(math.ceil(lower), lower_open=False)
            else:
                self.space.intervals.set_lower(lower, lower_open=False)
        else:
            self.space.intervals.set_lower(float(lower), lower_open=False)

    def Gt(self, lower: Union[bool, int, float]) -> None:
        if self.vtype not in [bool, int, float]:
            raise Exception(f"Gt is not valid constraint on {self.vtype}")
        if invalid_vtype(self.vtype, lower):
            raise TypeError("Variable type mismatch")
        if self.space.empty():
            return
        elif self.space.discrete.initialized:
            self.space.discrete.filter(lambda v: v > lower)
        elif self.vtype == int:
            if math.isfinite(lower):
                self.space.intervals.set_lower(
                    math.ceil(lower), lower_open=is_integer(lower)
                )
            else:
                self.space.intervals.set_lower(lower, lower_open=True)
        else:
            self.space.intervals.set_lower(float(lower), lower_open=True)
