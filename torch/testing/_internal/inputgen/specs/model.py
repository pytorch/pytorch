from dataclasses import dataclass
from enum import Enum
from typing import Callable

from torch.testing._internal.inputgen.attribute.model import Attribute


class ConstraintSuffix(str, Enum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOTIN = "notin"
    LE = "le"
    LT = "lt"
    GE = "ge"
    GT = "gt"
    ST = "st"
    GEN = "gen"
    FOCUS = "focus"


@dataclass
class Constraint:
    attribute: Attribute
    suffix: ConstraintSuffix
    fn: Callable


class ConstraintAttributeSuffixes:
    def __init__(self, attr: Attribute):
        self.Eq = lambda fn: Constraint(attr, ConstraintSuffix.EQ, fn)
        self.Ne = lambda fn: Constraint(attr, ConstraintSuffix.NE, fn)
        self.In = lambda fn: Constraint(attr, ConstraintSuffix.IN, fn)
        self.NotIn = lambda fn: Constraint(attr, ConstraintSuffix.NOTIN, fn)
        if attr in [Attribute.LENGTH, Attribute.RANK, Attribute.SIZE, Attribute.VALUE]:
            self.Le = lambda fn: Constraint(attr, ConstraintSuffix.LE, fn)
            self.Lt = lambda fn: Constraint(attr, ConstraintSuffix.LT, fn)
            self.Ge = lambda fn: Constraint(attr, ConstraintSuffix.GE, fn)
            self.Gt = lambda fn: Constraint(attr, ConstraintSuffix.GT, fn)
        self.St = lambda fn: Constraint(attr, ConstraintSuffix.ST, fn)
        self.Focus = lambda fn: Constraint(attr, ConstraintSuffix.FOCUS, fn)
        self.Gen = lambda fn: Constraint(attr, ConstraintSuffix.GEN, fn)


class ConstraintProducer:
    Optional = ConstraintAttributeSuffixes(Attribute.OPTIONAL)
    Dtype = ConstraintAttributeSuffixes(Attribute.DTYPE)
    Length = ConstraintAttributeSuffixes(Attribute.LENGTH)
    Rank = ConstraintAttributeSuffixes(Attribute.RANK)
    Size = ConstraintAttributeSuffixes(Attribute.SIZE)
    Value = ConstraintAttributeSuffixes(Attribute.VALUE)
