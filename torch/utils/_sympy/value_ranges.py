import dataclasses
import itertools
import sympy  # type: ignore[import]
import operator
import math
import logging
import torch
from typing import Union

log = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class ValueRanges:
    lower: Union[sympy.Expr, sympy.Number, int, float, bool]
    upper: Union[sympy.Expr, sympy.Number, int, float, bool]

    def __contains__(self, x):
        # TODO This needs to be generalised if lower/upper are sympy.Expr
        assert not isinstance(x, sympy.Expr)
        return self.lower <= x <= self.upper

    @classmethod
    def wrap(cls, arg):
        if isinstance(arg, ValueRanges):
            return arg
        assert isinstance(arg, (int, float, bool))
        return ValueRanges(arg, arg)

    @classmethod
    def increasing_map(cls, x, fn):
        """map lower and upper bound with fn"""
        x = cls.wrap(x)
        return ValueRanges(fn(x.lower), fn(x.upper))

    @classmethod
    def decreasing_map(cls, x, fn):
        """map lower bound to upper bound and upper bound to lower bound"""
        x = cls.wrap(x)
        return ValueRanges(fn(x.upper), fn(x.lower))

    @classmethod
    def monotone_map(cls, x, fn):
        """check the max and min of computed upper and lower bound for the output"""
        x = cls.wrap(x)
        l = fn(x.lower)
        u = fn(x.upper)
        return ValueRanges(min(l, u), max(l, u))

    @classmethod
    def convex_min_zero_map(cls, x, fn):
        """the max is at one of the ends"""
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges(0, max(fn(x.lower), fn(x.upper)))
        else:
            return cls.monotone_map(x, fn)

    @classmethod
    def coordinatewise_increasing_map(cls, x, y, fn):
        """map upper and lower bounds accessing corresponding values of inputs"""
        x, y = cls.wrap(x), cls.wrap(y)
        return ValueRanges(
            fn(x.lower, y.lower),
            fn(x.upper, y.upper),
        )

    @classmethod
    def coordinatewise_monotone_map(cls, x, y, fn):
        """compute the product of all lower and upper bounds and take min and max"""
        x, y = cls.wrap(x), cls.wrap(y)
        products = [
            fn(a, b)
            for a, b in itertools.product([x.lower, x.upper], [y.lower, y.upper])
        ]
        return ValueRanges(min(products), max(products))


class ValueRangeAnalysis:
    def __init__(self):
        self.name = "ValueRangeAnalysis"
        boolean_operators = (
            "eq",
            "ne",
            "lt",
            "gt",
            "le",
            "ge",
            "and_",
            "or_",
            "xor",
            "logical_and",
            "logical_or",
            "logical_not",
        )
        for op in boolean_operators:
            setattr(self, op, self.bool_handler)

    @staticmethod
    def bool_handler(*args, **kwargs):
        # just assuming bools can have both values
        return ValueRanges(sympy.false, sympy.true)  # type: ignore[arg-type]

    @staticmethod
    def default_handler(*args, **kwargs):
        # many ops are unlikely to show up in optimizable indexing compute,
        # so we dont have full coverage
        return ValueRanges(-math.inf, math.inf)

    def load(self, name: str, index: sympy.Expr):
        return ValueRanges(-math.inf, math.inf)

    def store(self, name, index, value, mode=None):
        return

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        return ValueRanges(-math.inf, math.inf)

    def index_expr(self, index, dtype):
        assert isinstance(index, ValueRanges)
        return index

    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        def is_bool(val):
            return isinstance(val, bool) or (
                hasattr(val, "is_Boolean") and val.is_Boolean
            )

        x = ValueRanges.wrap(x)
        low, up = x.lower, x.upper
        if is_bool(low):
            assert is_bool(up)
            if dtype.is_floating_point:
                return ValueRanges(sympy.Float(0.0), sympy.Float(1.0))
            else:
                return ValueRanges(sympy.Integer(0), sympy.Integer(1))
        return ValueRanges.wrap(x)

    @staticmethod
    def constant(value, dtype):
        # using nan makes subsequent computation throw, and for the purposes of optimization
        # returning -math.inf - math.inf is equivalent to giving up
        if math.isnan(value):
            return ValueRanges(-math.inf, math.inf)
        if isinstance(value, int):
            return ValueRanges(sympy.Integer(value), sympy.Integer(value))
        else:
            return ValueRanges(sympy.Float(value), sympy.Float(value))

    @staticmethod
    def reciprocal(x):
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges(-math.inf, math.inf)
        else:
            return ValueRanges.decreasing_map(x, lambda y: 1 / y)

    @staticmethod
    def square(x):
        return ValueRanges.convex_min_zero_map(x, lambda y: y * y)

    @staticmethod
    def abs(x):
        return ValueRanges.convex_min_zero_map(x, abs)

    @staticmethod
    def neg(x):
        return ValueRanges.decreasing_map(x, operator.neg)

    @staticmethod
    def truediv(a, b):
        b = ValueRanges.wrap(b)
        if 0 in b:
            return ValueRanges(-math.inf, math.inf)
        else:
            return ValueRangeAnalysis.mul(a, ValueRanges(1 / b.upper, 1 / b.lower))

    @staticmethod
    def div(a, b):
        # We think of this as floor(a / b)
        out = ValueRangeAnalysis.truediv(a, b)
        return ValueRangeAnalysis.floor(out)

    @staticmethod
    def add(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, operator.add)

    @staticmethod
    def mul(a, b):
        return ValueRanges.coordinatewise_monotone_map(a, b, operator.mul)

    @staticmethod
    def sub(a, b):
        b = ValueRanges.wrap(b)
        return ValueRangeAnalysis.add(a, ValueRanges(-b.upper, -b.lower))

    @staticmethod
    def exp(x):
        return ValueRanges.increasing_map(x, sympy.functions.elementary.exponential.exp)

    @staticmethod
    def log(x):
        return ValueRanges.increasing_map(
            x, lambda y: -math.inf if y <= 0 else sympy.log(y)
        )

    @staticmethod
    def sqrt(x):
        return ValueRanges.increasing_map(x, sympy.sqrt)

    @staticmethod
    def pow(a, b):
        def is_integer(val):
            return (
                isinstance(val, int)
                or (isinstance(val, float) and val == int(val))
                or (hasattr(val, "is_integer") and val.is_integer)
            )

        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.lower < 0 and not is_integer(b.lower):
            # The function is not defined
            return ValueRanges(-math.inf, math.inf)
        elif 0 in a and b.lower <= 0:
            return ValueRanges(-math.inf, math.inf)
        return ValueRanges.coordinatewise_monotone_map(a, b, operator.pow)

    @staticmethod
    def minimum(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, min)

    @staticmethod
    def maximum(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, max)

    @staticmethod
    def where(a, b, c):
        b = ValueRanges.wrap(b)
        c = ValueRanges.wrap(c)
        return ValueRanges(min(b.lower, c.lower), max(b.upper, c.upper))

    @staticmethod
    def floor(x):
        return ValueRangeAnalysis.floor_ceil(
            x, sympy.functions.elementary.integers.floor
        )

    @staticmethod
    def ceil(x):
        return ValueRangeAnalysis.floor_ceil(
            x, sympy.functions.elementary.integers.ceiling
        )

    @staticmethod
    def floor_ceil(x, fn_int):
        def is_integer(val):
            return isinstance(val, int) or (
                hasattr(val, "is_integer") and val.is_integer
            )

        if is_integer(x):
            fn = fn_int
        else:

            def fn(x):
                return sympy.Float(fn_int(x))

        return ValueRanges.increasing_map(x, fn)

    def __getattr__(self, name):
        log.warning(f"unhandled ValueRange op {name}")
        return self.default_handler
