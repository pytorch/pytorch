import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom, Boolean as SympyBoolean
import operator
import math
import logging
import torch
from typing import Union, Dict

from torch._prims_common import dtype_to_type
from .interp import sympy_interp

log = logging.getLogger(__name__)

__all__ = ["ValueRanges", "ValueRangeAnalysis", "bound_sympy"]

class ValueRangeError(RuntimeError):
    pass


# Like sympify, but supports less stuff, and also ensures that direct
# sympy expressions don't have free variables
def simple_sympify(e):
    if isinstance(e, bool):
        return sympy.true if e else sympy.false
    elif isinstance(e, int):
        return sympy.Integer(e)
    elif isinstance(e, float):
        # infinity is special; we use it to bracket integers as well
        if math.isinf(e):
            return sympy.oo if e > 0 else -sympy.oo
        return sympy.Float(e)
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
    is_bool: bool

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
        assert isinstance(upper, SympyBoolean) == self.is_bool

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

class SymPyValueRangeAnalysis:
    @staticmethod
    def constant(value, dtype):
        # using nan makes subsequent computation throw, and for the purposes of optimization
        # returning -math.inf - math.inf is equivalent to giving up
        if math.isnan(value):
            return ValueRanges.unknown()

        assert isinstance(value, (int, float, bool))
        type_ = dtype_to_type(dtype)
        value = type_(value)

        return ValueRanges.wrap(value)

    @staticmethod
    def not_(a):
        return ValueRanges.decreasing_map(a, sympy.Not)

    @staticmethod
    def or_(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.Or)

    @staticmethod
    def and_(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.And)

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

    @classmethod
    def lt(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        assert a.is_bool == b.is_bool
        if a.is_bool:
            return cls.and_(cls.not_(a), b)
        else:
            if a.upper < b.lower:
                return ValueRanges.wrap(sympy.true)
            elif a.lower >= b.upper:
                return ValueRanges.wrap(sympy.false)
            return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def gt(cls, a, b):
        return cls.lt(b, a)

    @classmethod
    def le(cls, a, b):
        return cls.not_(cls.gt(a, b))

    @classmethod
    def ge(cls, a, b):
        return cls.not_(cls.lt(a, b))

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

    @classmethod
    def div(cls, a, b):
        return cls.truediv(a, b)

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
    def mod(x, y):
        x = ValueRanges.wrap(x)
        y = ValueRanges.wrap(y)
        if x.is_singleton() and y.is_singleton() and y.lower != 0:
            return ValueRanges.wrap(x.lower % y.lower)
        if y.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges(0, y.upper)

    @classmethod
    def modular_indexing(cls, a, b, c):
        return cls.mod(cls.floordiv(a, b), c)

    @classmethod
    def pow(cls, a, b):
        def is_integer(val):
            return isinstance(val, int) or (
                hasattr(val, "is_integer") and val.is_integer
            )

        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        # Not implemented yet. It's a bit tricky
        # If you want to implement it, compute the partial derivatives of a ** b
        # and check the ranges where the function is increasing / decreasing
        if not b.is_singleton():
            return ValueRanges.unknown()

        b = b.lower
        if a.is_singleton():
            a = a.lower
            r = a ** b
            if r == sympy.zoo:
                return ValueRanges.unknown()
            return ValueRanges.wrap(r)

        if not is_integer(b):
            if b < 0 or a.lower < 0:
                return ValueRanges.unknown()
            return ValueRanges.increasing_map(a, lambda x: x ** b)

        # exponentiation by squaring
        if b < 0:
            a = cls.reciprocal(a)
            b = -b
        acc = ValueRanges.wrap(1)
        while b > 0:
            if b % 2 == 1:
                acc = cls.mul(acc, a)
            a = cls.mul(a, a)
            b = b // 2
        return acc

    @staticmethod
    def reciprocal(x):
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges.unknown()
        else:
            return ValueRanges.decreasing_map(x, lambda y: 1 / y)

    @staticmethod
    def abs(x):
        return ValueRanges.convex_min_zero_map(x, abs)

    @staticmethod
    def exp(x):
        return ValueRanges.increasing_map(x, sympy.functions.elementary.exponential.exp)

    @staticmethod
    def log(x):
        x = ValueRanges.wrap(x)
        if x.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, sympy.log)

    @classmethod
    def minimum(cls, a, b):
        return cls.min_or_max(a, b, sympy.Min)

    @classmethod
    def maximum(cls, a, b):
        return cls.min_or_max(a, b, sympy.Max)

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

class ValueRangeAnalysis(SymPyValueRangeAnalysis):
    """ Extend SymPy Ops with IR ops """
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
    def to_dtype(x, dtype: torch.dtype):
        x = ValueRanges.wrap(x)
        type_ = dtype_to_type(dtype)
        if type_ is bool:
            if x.is_singleton():
                return ValueRanges.wrap(x.lower != 0)
            elif 0 not in x:
                return ValueRanges.wrap(sympy.true)
            else:
                return ValueRanges(sympy.false, sympy.true)
        # If we want to do this properly, we'd need to track the type of the variables
        return x

    @staticmethod
    def sub(a, b):
        b = ValueRanges.wrap(b)
        return ValueRangeAnalysis.add(a, ValueRanges(-b.upper, -b.lower))

    @staticmethod
    def square(x):
        return ValueRanges.convex_min_zero_map(x, lambda y: y * y)

    @staticmethod
    def neg(x):
        return ValueRanges.decreasing_map(x, operator.neg)

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
    def sqrt(x):
        x = ValueRanges.wrap(x)
        if x.lower < 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, sympy.sqrt)

    @staticmethod
    def where(a, b, c):
        b = ValueRanges.wrap(b)
        c = ValueRanges.wrap(c)
        assert a.is_bool
        assert b.is_bool == c.is_bool
        if b.is_bool:
            return ValueRanges(sympy.And(b.lower, c.lower), sympy.Or(b.upper, c.upper))
        else:
            return ValueRanges(min(b.lower, c.lower), max(b.upper, c.upper))

    @classmethod
    def floor(cls, x):
        return cls.floor_ceil(
            x, sympy.functions.elementary.integers.floor
        )

    @classmethod
    def ceil(cls, x):
        return cls.floor_ceil(
            x, sympy.functions.elementary.integers.ceiling
        )

    @staticmethod
    def floor_ceil(x, fn):
        return ValueRanges.increasing_map(x, fn)

    def __getattr__(self, name):
        log.warning("unhandled ValueRange op %s", name)
        return self.default_handler


def bound_sympy(expr: sympy.Expr, ranges: Dict[sympy.Symbol, ValueRanges]) -> ValueRanges:
    # CeilDiv does not occur in practice
    return sympy_interp(SymPyValueRangeAnalysis(), ranges, expr)
