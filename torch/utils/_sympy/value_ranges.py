import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom
import operator
import math
import logging
import torch
from typing import Union

log = logging.getLogger(__name__)

__all__ = ['ValueRanges', 'ValueRangeAnalysis']

SympyBoolean = sympy.logic.boolalg.Boolean

# Like sympify, but supports less stuff, and also ensures that direct
# sympy expressions don't have free variables
def simple_sympify(e):
    if isinstance(e, int):
        return sympy.Integer(e)
    elif isinstance(e, float):
        # infinity is special; we use it to bracket integers as well
        if math.isinf(e):
            return sympy.oo if e > 0 else -sympy.oo
        return sympy.Float(e)
    elif isinstance(e, bool):
        return sympy.true if e else sympy.false
    elif isinstance(e, sympy.Expr):
        # TODO: Eventually, we will want to do indexing calculations with
        # respect to symbols, so we can generate a dynamic kernel which will
        # use 32-bit indexing so long as the dynamic dim isn't too big.  To do
        # that, we will need to be able to do ValueRanges
        assert not e.free_symbols, f"free variables NYI: {e}"
        # NaNs can occur when doing things like 0 * sympy.oo, but it is better
        # if the operator notices this and takes care of it, because sometimes
        # the NaN is inappropriate (for example, for ints, the [-oo, oo] range
        # should go to zero when multiplied with [0, 0])
        assert e != sympy.nan
        return e
    elif isinstance(e, BooleanAtom):
        return e
    else:
        raise AssertionError(f"not simple sympy type {type(e)}")

# Sympy atomics only. Unlike <=, it also works on Sympy bools.
def sympy_generic_le(lower, upper):
    if isinstance(lower, sympy.Expr):
        assert isinstance(upper, sympy.Expr)
        return lower <= upper
    else:
        # only negative condition is True > False
        assert isinstance(lower, SympyBoolean) and isinstance(upper, SympyBoolean)
        return not (lower is sympy.true and upper is sympy.false)

@dataclasses.dataclass(frozen=True)
class ValueRanges:
    # Although the type signature here suggests you can pass any
    # sympy expression, in practice the analysis here only works
    # with sympy expressions with no free variables
    lower: Union[sympy.Expr, SympyBoolean]
    upper: Union[sympy.Expr, SympyBoolean]

    def __init__(self, lower, upper):
        lower = simple_sympify(lower)
        upper = simple_sympify(upper)
        # We don't support point-ranges on floating point inf
        assert lower != sympy.oo
        assert upper != -sympy.oo
        # TODO: when the bounds have free variables, this may be
        # nontrivial to actually verify
        assert sympy_generic_le(lower, upper)
        # Because this is a frozen class
        object.__setattr__(self, 'lower', lower)
        object.__setattr__(self, 'upper', upper)

    def __contains__(self, x):
        x = simple_sympify(x)
        return bool(self.lower <= x <= self.upper)

    # TODO: this doesn't work with bools but arguably it should
    @classmethod
    def unknown(cls):
        return cls(-sympy.oo, sympy.oo)

    @classmethod
    def wrap(cls, arg):
        if isinstance(arg, ValueRanges):
            return arg
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
        return ValueRanges.unknown()

    def load(self, name: str, index: sympy.Expr):
        return ValueRanges.unknown()

    def store(self, name, index, value, mode=None):
        return

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        return ValueRanges.unknown()

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
                return ValueRanges(0.0, 1.0)
            else:
                return ValueRanges(0, 1)
        return ValueRanges.wrap(x)

    @staticmethod
    def constant(value, dtype):
        # NB: value is NOT a sympy expression, it's a constant!
        assert isinstance(value, (int, float, bool))
        # using nan makes subsequent computation throw, and for the purposes of optimization
        # returning -math.inf - math.inf is equivalent to giving up
        if math.isnan(value):
            return ValueRanges.unknown()
        return ValueRanges.wrap(value)

    @staticmethod
    def reciprocal(x):
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges.unknown()
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
            return ValueRanges.unknown()
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
        def safe_mul(a, b):
            if a == 0:
                return 0
            elif b == 0:
                return 0
            return a * b
        return ValueRanges.coordinatewise_monotone_map(a, b, safe_mul)

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
            x, lambda y: -sympy.oo if y <= 0 else sympy.log(y)
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
            return ValueRanges.unknown()
        elif 0 in a and b.lower <= 0:
            return ValueRanges.unknown()
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
