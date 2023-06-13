import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom, Boolean as SympyBoolean
import operator
import math
import logging
import torch
from typing import Union

log = logging.getLogger(__name__)

__all__ = ["ValueRanges", "ValueRangeAnalysis"]

class ValueRangeError(RuntimeError):
    pass


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
        raise AssertionError(f"not simple sympy type {type(e)}: {e}")


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
        # TODO: when the bounds have free variables, this may be
        # nontrivial to actually verify
        if not sympy_generic_le(lower, upper):
            raise ValueRangeError(f"Invalid ranges [{lower}:{upper}]")
        # Because this is a frozen class
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "is_bool", isinstance(lower, SympyBoolean))

    def __contains__(self, x):
        x = simple_sympify(x)
        return sympy_generic_le(self.lower, x) and sympy_generic_le(x, self.upper)

    # Intersection
    def __and__(self, other):
        return ValueRanges(lower=max(self.lower, other.lower), upper=min(self.upper, other.upper))

    def is_singleton(self) -> bool:
        return self.lower == self.upper

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
    def or_(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        assert a.is_bool and b.is_bool
        if a.lower or b.lower:
            return ValueRanges.wrap(sympy.true)
        elif a.is_singleton() and b.is_singleton():
            return ValueRanges.wrap(sympy.Or(a.lower, b.lower))
        else:
            return ValueRanges(sympy.false, sympy.true)

    @staticmethod
    def and_(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        assert a.is_bool and b.is_bool
        if not a.upper or not b.upper:
            return ValueRanges.wrap(sympy.false)
        elif a.is_singleton() and b.is_singleton():
            return ValueRanges.wrap(sympy.And(a.lower, b.lower))
        else:
            return ValueRanges(sympy.false, sympy.true)

    @staticmethod
    def eq(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton() and a.lower == b.lower:
            return ValueRanges.wrap(sympy.true)
        elif a.lower > b.upper or b.lower > a.upper:  # ranges disjoint
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @staticmethod
    def lt(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.upper < b.lower:
            return ValueRanges.wrap(sympy.true)
        elif a.lower >= b.upper:
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def gt(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.lower > b.upper:
            return ValueRanges.wrap(sympy.true)
        elif a.upper <= b.lower:
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def le(cls, a, b):
        return cls.not_(cls.gt(a, b))

    @classmethod
    def ge(cls, a, b):
        return cls.not_(cls.lt(a, b))

    @staticmethod
    def not_(a):
        a = ValueRanges.wrap(a)
        assert a.is_bool
        if a.is_singleton():
            return ValueRanges.wrap(sympy.Not(a.lower))
        return ValueRanges(sympy.false, sympy.true)

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
        return x

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
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(a, b, operator.truediv)

    @staticmethod
    def floordiv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(a, b, operator.floordiv)

    @staticmethod
    def truncdiv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)):
            return ValueRanges.unknown()
        else:
            # Casting to integer does truncation
            def f(a, b):
                result = a / b
                # This won't work for sympy.Expr, so it'll need a workaround when
                # dealing with dynamic shapes
                if result.is_finite:
                    result = sympy.Integer(result)
                return result
            return ValueRanges.coordinatewise_monotone_map(a, b, f)

    @staticmethod
    def div(a, b):
        return ValueRangeAnalysis.truediv(a, b)

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
        x = ValueRanges.wrap(x)
        if x.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, sympy.log)

    @staticmethod
    def mod(x, y):
        x = ValueRanges.wrap(x)
        y = ValueRanges.wrap(y)
        if x.is_singleton() and y.is_singleton() and y.lower != 0:
            return ValueRanges.wrap(x.lower % y.lower)
        if y.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges(0, y.upper)

    @staticmethod
    def sqrt(x):
        x = ValueRanges.wrap(x)
        if x.lower < 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, sympy.sqrt)

    @classmethod
    def pow(cls, a, b):
        def is_integer(val):
            return isinstance(val, int) or (
                hasattr(val, "is_integer") and val.is_integer
            )

        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton():
            r = a.lower**b.lower
            if r == sympy.zoo:
                return ValueRanges.unknown()
            return ValueRanges.wrap(r)
        elif b.is_singleton() and is_integer(b.lower) and b.lower >= 0:
            i = ValueRanges.wrap(1)
            for _ in range(b.lower):
                i = cls.mul(i, a)
            return i
        else:
            # This is fairly difficult to analyze, so give up for anything
            # complicated
            return ValueRanges.unknown()

    @staticmethod
    def minimum(a, b):
        return ValueRangeAnalysis.min_or_max(a, b, sympy.Min)

    @staticmethod
    def maximum(a, b):
        return ValueRangeAnalysis.min_or_max(a, b, sympy.Max)

    @staticmethod
    def min_or_max(a, b, fn):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        # Performs upcasting first
        def fn_(x, y):
            # Poorman's version of upcasting in Sympy
            # This won't do for sympy.Expr as the casting does nothing for those
            # Inf is not a float...
            if x.is_Float or not x.is_finite or y.is_Float or not y.is_finite:
                result_type = sympy.Float
            else:
                assert x.is_Integer
                assert y.is_Integer
                result_type = sympy.Integer
            return fn(result_type(x), result_type(y))

        return ValueRanges.coordinatewise_increasing_map(a, b, fn_)

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
    def floor_ceil(x, fn):
        return ValueRanges.increasing_map(x, fn)

    def __getattr__(self, name):
        log.warning("unhandled ValueRange op %s", name)
        return self.default_handler
