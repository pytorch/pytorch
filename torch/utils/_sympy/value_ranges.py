# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import operator
from typing import (
    Callable,
    Dict,
    Generic,
    Optional,
    overload,
    SupportsFloat,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import TypeGuard

import sympy
from sympy.logic.boolalg import Boolean as SympyBoolean, BooleanAtom

import torch
from torch._logging import LazyString

from torch._prims_common import dtype_to_type
from .functions import (
    _keep_float,
    FloatTrueDiv,
    FloorDiv,
    IntTrueDiv,
    OpaqueUnaryFn_exp,
    OpaqueUnaryFn_log,
    OpaqueUnaryFn_sqrt,
    PowByNatural,
    RoundDecimal,
    RoundToInt,
    safe_pow,
    ToFloat,
    TruncToFloat,
    TruncToInt,
)
from .interp import sympy_interp
from .numbers import int_oo, IntInfinity, NegativeIntInfinity

log = logging.getLogger(__name__)

__all__ = ["ValueRanges", "ValueRangeAnalysis", "bound_sympy"]

_T = TypeVar("_T", sympy.Expr, SympyBoolean)


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
        assert e.is_number, e
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
        assert isinstance(lower, SympyBoolean) and isinstance(upper, SympyBoolean), (
            lower,
            upper,
        )
        return not (lower and not upper)


def vr_is_bool(vr: ValueRanges[_T]) -> TypeGuard[ValueRanges[SympyBoolean]]:
    return vr.is_bool


def vr_is_expr(vr: ValueRanges[_T]) -> TypeGuard[ValueRanges[sympy.Expr]]:
    return not vr.is_bool


ExprIn = Union[int, float, sympy.Expr]
BoolIn = Union[bool, SympyBoolean]
AllIn = Union[ExprIn, BoolIn]
ExprFn = Callable[[sympy.Expr], sympy.Expr]
ExprFn2 = Callable[[sympy.Expr, sympy.Expr], sympy.Expr]
BoolFn = Callable[[SympyBoolean], SympyBoolean]
BoolFn2 = Callable[[SympyBoolean, SympyBoolean], SympyBoolean]
AllFn = Union[ExprFn, BoolFn]
AllFn2 = Union[ExprFn2, BoolFn2]


@dataclasses.dataclass(frozen=True)
class ValueRanges(Generic[_T]):
    if TYPE_CHECKING:
        # ruff doesn't understand circular references but mypy does
        ExprVR = ValueRanges[sympy.Expr]  # noqa: F821
        BoolVR = ValueRanges[SympyBoolean]  # noqa: F821
        AllVR = Union[ExprVR, BoolVR]

    # Although the type signature here suggests you can pass any
    # sympy expression, in practice the analysis here only works
    # with constant sympy expressions
    lower: _T
    upper: _T
    is_bool: bool
    is_int: bool
    is_float: bool

    def __repr__(self) -> str:
        return f"VR[{self.lower}, {self.upper}]"

    @overload
    def __init__(self: ValueRanges[sympy.Expr], lower: ExprIn, upper: ExprIn) -> None:
        ...

    @overload
    def __init__(self: ValueRanges[SympyBoolean], lower: BoolIn, upper: BoolIn) -> None:
        ...

    def __init__(self, lower: AllIn, upper: AllIn) -> None:
        lower = simple_sympify(lower)
        upper = simple_sympify(upper)
        # TODO: when the bounds have free variables, this may be
        # nontrivial to actually verify
        try:
            if not sympy_generic_le(lower, upper):
                raise ValueRangeError(f"Invalid ranges [{lower}:{upper}]")
        except TypeError as e:
            raise TypeError(f"Could not compare {lower} <= {upper}") from e
        # Because this is a frozen class
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        # Unlike bool/int in Python, we don't report bools are ints
        object.__setattr__(self, "is_bool", isinstance(lower, SympyBoolean))
        if self.is_bool:
            assert isinstance(upper, SympyBoolean), (lower, upper)

        # Warning: is_int/is_float is best effort.  We do pretty well in
        # Dynamo, but in Inductor these attributes are often wrong because we
        # are not very rigorous in dtype analysis.  This is also why we need
        # the flexible analysis for is_int: sometimes a sympy.oo pops in for
        # an integer bound. I would /like/ for us not to do this, but it's
        # too hard to push the invariant through right now.

        object.__setattr__(
            self,
            "is_int",
            not self.is_bool
            and (
                isinstance(lower, (sympy.Integer, NegativeIntInfinity))
                or isinstance(upper, (sympy.Integer, IntInfinity))
            ),
        )
        """
        # This assert is just impossible right now, too many sympy bugs
        if self.is_int:
            # NB: sympy will sometimes randomly lose the float-ness of zero,
            # so we also need to account for that in the assertion here.
            # See also https://github.com/sympy/sympy/issues/26620
            assert isinstance(lower, sympy.Integer) or lower in [-sympy.oo, 0], (
                lower,
                upper,
            )
            assert isinstance(upper, sympy.Integer) or upper in [sympy.oo, 0], (lower, upper)
        """
        # NB: [-oo, oo] always advertises as float!
        object.__setattr__(self, "is_float", not self.is_bool and not self.is_int)
        assert self.is_bool or self.is_int or self.is_float, (lower, upper)

    def boolify(self) -> ValueRanges[SympyBoolean]:
        if vr_is_bool(self):
            return self
        elif self == ValueRanges.unknown():
            return ValueRanges.unknown_bool()
        else:
            raise AssertionError(f"not bool like {self}")

    def __contains__(self, x: AllIn) -> bool:
        return ValueRanges.wrap(x).issubset(self)

    def issubset(self, other):
        return sympy_generic_le(other.lower, self.lower) and sympy_generic_le(
            self.upper, other.upper
        )

    def tighten(self, other) -> ValueRanges:
        """Given two ValueRanges, returns their intersection"""
        return self & other

    # Intersection
    @overload
    def __and__(
        self: ValueRanges[sympy.Expr], other: ValueRanges[sympy.Expr]
    ) -> ValueRanges[sympy.Expr]:
        ...

    @overload
    def __and__(
        self: ValueRanges[SympyBoolean], other: ValueRanges[SympyBoolean]
    ) -> ValueRanges[SympyBoolean]:
        ...

    def __and__(self: AllVR, other: AllVR) -> AllVR:
        if other == ValueRanges.unknown():
            return self
        if self == ValueRanges.unknown():
            return other
        assert self.is_bool == other.is_bool, (self, other)
        assert self.is_int == other.is_int, (self, other)
        assert self.is_float == other.is_float, (self, other)
        if self.is_bool:
            return ValueRanges(
                sympy.Or(self.lower, other.lower), sympy.And(self.upper, other.upper)
            )
        else:
            return ValueRanges(
                sympy.Max(self.lower, other.lower), sympy.Min(self.upper, other.upper)
            )

    # Union
    @overload
    def __or__(
        self: ValueRanges[sympy.Expr], other: ValueRanges[sympy.Expr]
    ) -> ValueRanges[sympy.Expr]:
        ...

    @overload
    def __or__(
        self: ValueRanges[SympyBoolean], other: ValueRanges[SympyBoolean]
    ) -> ValueRanges[SympyBoolean]:
        ...

    def __or__(self: AllVR, other: AllVR) -> AllVR:
        if ValueRanges.unknown() in (self, other):
            return ValueRanges.unknown()
        assert self.is_bool == other.is_bool, (self, other)
        if self.is_bool:
            return ValueRanges(
                sympy.And(self.lower, other.lower), sympy.Or(self.upper, other.upper)
            )
        else:
            return ValueRanges(
                sympy.Min(self.lower, other.lower), sympy.Max(self.upper, other.upper)
            )

    def is_singleton(self) -> bool:
        return self.lower == self.upper

    @staticmethod
    def unknown() -> ValueRanges[sympy.Expr]:
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def unknown_int() -> ValueRanges[sympy.Expr]:
        return ValueRanges(-int_oo, int_oo)

    @staticmethod
    def unknown_bool() -> ValueRanges[SympyBoolean]:
        return ValueRanges(sympy.false, sympy.true)

    @overload
    @staticmethod
    # work around the fact that bool and int overlap
    def wrap(arg: Union[ExprIn, ExprVR]) -> ExprVR:  # type: ignore[overload-overlap]
        ...

    @overload
    @staticmethod
    def wrap(arg: Union[BoolIn, BoolVR]) -> BoolVR:
        ...

    @staticmethod
    def wrap(arg: Union[AllIn, AllVR]) -> AllVR:
        if isinstance(arg, ValueRanges):
            return arg
        if isinstance(arg, float) and math.isnan(arg):
            return ValueRanges.unknown()
        # arg is either ExprIn or BoolIn, but we don't know it here
        return ValueRanges(arg, arg)  # type: ignore[arg-type]

    @staticmethod
    def increasing_map(x: Union[ExprIn, ExprVR], fn: ExprFn) -> ExprVR:
        """Increasing: x <= y => f(x) <= f(y)."""
        x = ValueRanges.wrap(x)
        return ValueRanges(fn(x.lower), fn(x.upper))

    @overload
    @staticmethod
    def decreasing_map(x: Union[ExprIn, ExprVR], fn: ExprFn) -> ExprVR:
        ...

    @overload
    @staticmethod
    def decreasing_map(x: Union[BoolIn, BoolVR], fn: BoolFn) -> BoolVR:
        ...

    @staticmethod
    def decreasing_map(x: Union[AllIn, AllVR], fn: AllFn) -> AllVR:
        """Decreasing: x <= y => f(x) >= f(y)."""
        x = ValueRanges.wrap(x)
        # consistently either Expr or Bool, but we don't know it here
        return ValueRanges(fn(x.upper), fn(x.lower))  # type: ignore[arg-type]

    @staticmethod
    def monotone_map(x: Union[ExprIn, ExprVR], fn: ExprFn) -> ExprVR:
        """It's increasing or decreasing."""
        x = ValueRanges.wrap(x)
        l = fn(x.lower)
        u = fn(x.upper)
        return ValueRanges(min(l, u), max(l, u))

    @staticmethod
    def convex_min_zero_map(x: Union[ExprIn, ExprVR], fn: ExprFn) -> ExprVR:
        """Fn is convex and has a minimum at 0."""
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges(0, max(fn(x.lower), fn(x.upper)))
        else:
            return ValueRanges.monotone_map(x, fn)

    @overload
    @staticmethod
    def coordinatewise_increasing_map(
        x: Union[ExprIn, ExprVR], y: Union[ExprIn, ExprVR], fn: ExprFn2
    ) -> ExprVR:
        ...

    @overload
    @staticmethod
    def coordinatewise_increasing_map(
        x: Union[BoolIn, BoolVR], y: Union[BoolIn, BoolVR], fn: BoolFn2
    ) -> BoolVR:
        ...

    @staticmethod
    def coordinatewise_increasing_map(
        x: Union[AllIn, AllVR], y: Union[AllIn, AllVR], fn: AllFn2
    ) -> AllVR:
        """
        It's increasing on each coordinate.

        Mathematically:
        For every 1 <= i <= n and x_i <= y_i we have that
        f(x1, .., xn) <= f(x1, , yi, ..., xn)
        """
        x, y = ValueRanges.wrap(x), ValueRanges.wrap(y)
        return ValueRanges(
            fn(x.lower, y.lower),  # type: ignore[arg-type]
            fn(x.upper, y.upper),  # type: ignore[arg-type]
        )

    @classmethod
    def coordinatewise_monotone_map(cls, x, y, fn):
        """It's increasing or decreasing on each coordinate."""
        x, y = cls.wrap(x), cls.wrap(y)
        products = [
            fn(a, b)
            for a, b in itertools.product([x.lower, x.upper], [y.lower, y.upper])
        ]
        return ValueRanges(min(products), max(products))


class SymPyValueRangeAnalysis:
    """
    It gives bounds on a SymPy operator given bounds on its arguments
    See the function `bound_sympy` for a function that applies this logic to a full SymPy expression
    """

    @staticmethod
    def constant(value, dtype):
        if isinstance(value, ValueRanges):
            assert value.is_singleton()
            value = value.lower
        # NB: value is NOT a sympy expression, it's a constant!
        is_python = isinstance(value, (int, float, bool))
        assert is_python or isinstance(
            value, (BooleanAtom, sympy.Integer, sympy.Number)
        )

        # using nan makes subsequent computation throw, and for the purposes of optimization
        # returning -math.inf - math.inf is equivalent to giving up
        if isinstance(value, SupportsFloat) and math.isnan(value):
            if dtype == torch.bool:
                return ValueRanges.unknown_bool()
            elif dtype.is_floating_point:
                return ValueRanges.unknown()
            else:
                return ValueRanges(-int_oo, int_oo)

        if is_python:
            type_ = dtype_to_type(dtype)
            value = type_(value)
        else:
            # We do a type check on a best-effort basis
            # We don't want to force a cast to sympy.Float if the value is Rational to avoid losing precision
            if dtype == torch.bool:
                assert isinstance(value, BooleanAtom)
            elif dtype.is_floating_point:
                assert not value.is_finite or value.is_real
            else:
                # dtype is intXX
                assert value.is_integer

        r = ValueRanges.wrap(value)
        return r

    @staticmethod
    def to_dtype(a, dtype, src_dtype=None):
        if dtype == torch.float64:
            return ValueRanges.increasing_map(a, ToFloat)
        elif dtype == torch.bool:
            return ValueRanges.unknown_bool()
        elif not dtype.is_floating_point:
            return ValueRanges.unknown_int()
        return ValueRanges.unknown()

    @staticmethod
    def trunc_to_int(a, dtype):
        return ValueRanges.increasing_map(a, TruncToInt)

    @staticmethod
    def not_(a):
        a = ValueRanges.wrap(a)
        a = a.boolify()
        assert a.is_bool
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
    def identity(cls, a):
        return ValueRanges.wrap(a)

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
        return ValueRanges.coordinatewise_increasing_map(
            a, b, _keep_float(operator.add)
        )

    @classmethod
    def mul(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        assert a.is_bool == b.is_bool
        if a.is_bool:
            return cls.and_(a, b)

        def safe_mul(a, b):
            # Make unknown() * wrap(0) == wrap(0)
            if a == 0:
                return a
            elif b == 0:
                return b
            else:
                return a * b

        return ValueRanges.coordinatewise_monotone_map(a, b, _keep_float(safe_mul))

    @staticmethod
    def int_truediv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-int_oo in a or int_oo in a) and (-int_oo in b or int_oo in b)):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(
                a, b, _keep_float(IntTrueDiv)
            )

    @staticmethod
    def truediv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or (
            (-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)
        ):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(
                a, b, _keep_float(FloatTrueDiv)
            )

    @staticmethod
    def floordiv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b:
            return ValueRanges.unknown()
        products = []
        for x, y in itertools.product([a.lower, a.upper], [b.lower, b.upper]):
            r = FloorDiv(x, y)
            if r is sympy.nan:
                products.append((sympy.sign(x) * sympy.sign(y)) * int_oo)
            else:
                products.append(r)

        return ValueRanges(min(products), max(products))

    @classmethod
    def mod(cls, x, y):
        x = ValueRanges.wrap(x)
        y = ValueRanges.wrap(y)
        # nb. We implement C semantics

        def c_mod(a, b):
            ret = abs(a) % abs(b)
            if a < 0:
                ret *= -1
            return ret

        def c_div(a, b):
            x = a / b
            return sympy.Integer(x) if x.is_finite and x not in (int_oo, -int_oo) else x

        if 0 in y:
            return ValueRanges.unknown_int()
        elif y.is_singleton():
            y_val = abs(y.lower)
            # If it wraps, we need to take the whole interval

            # The function is locally linear if they are in the same class
            if c_div(x.lower, y_val) == c_div(x.upper, y_val):
                return ValueRanges.increasing_map(x, lambda u: c_mod(u, y_val))
            if x.upper < 0:
                # Negative case
                return ValueRanges(-y_val + 1, 0)
            elif x.lower > 0:
                # Positive case
                return ValueRanges(0, y_val - 1)
            else:
                # Mixed case
                lower = max(-y_val + 1, x.lower)
                upper = min(y_val - 1, x.upper)
                return ValueRanges(lower, upper)
        else:
            # Too difficult, we bail out
            upper = cls.abs(y).upper - 1
            return ValueRanges(-upper, upper)

    @classmethod
    def modular_indexing(cls, a, b, c):
        return cls.mod(cls.floordiv(a, b), c)

    @classmethod
    def is_non_overlapping_and_dense_indicator(cls, *args):
        return ValueRanges.unknown_int()

    @classmethod
    def pow_by_natural(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton():
            return ValueRanges.wrap(safe_pow(a.lower, b.lower))
        # NB: Exclude zero, because zero is special
        elif a.lower >= 1:
            # We should know that b >= 0 but we may have forgotten this fact due
            # to replacements, so don't assert it, but DO clamp it to prevent
            # degenerate problems
            return ValueRanges.coordinatewise_increasing_map(
                a, b & ValueRanges(0, int_oo), PowByNatural
            )
        elif b.is_singleton():
            if b.lower % 2 == 0:
                # x^n where n is even
                return ValueRanges.convex_min_zero_map(
                    a, lambda x: safe_pow(x, b.lower)
                )
            else:
                # x^n where n is odd
                return ValueRanges.increasing_map(a, lambda x: safe_pow(x, b.lower))
        else:
            # a is potentially negative, and we don't know if the exponent is
            # even or odd.  So just conservatively set the upper and lower
            # bound based on what the maximum absolute value could be, in both
            # directions
            max_base = max(a.upper, -a.lower)
            return ValueRanges(
                -(safe_pow(max_base, b.upper)), safe_pow(max_base, b.upper)
            )

    @classmethod
    def pow(cls, a, b):
        return ValueRanges.unknown()

        # We could implement all this, but for floating point pow, is there
        # really a point?
        """
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        # Not implemented yet. It's a bit tricky
        # If you want to implement it, compute the partial derivatives of a ** b
        # and check the ranges where the function is increasing / decreasing
        # Another non-tight way of doing this is defaulting to doing noting that for a > 0,  a ** b == exp(b * log(a))
        # If this second option is implemented, by carefult about the types and possible infinities here and there.
        if not b.is_singleton():
            return ValueRanges.unknown()

        b = b.lower
        if a.is_singleton():
            a = a.lower
            r = a**b
            if not r.is_finite:
                return ValueRanges.unknown()
            return ValueRanges.wrap(r)

        if b == 0:
            if not a.lower.is_finite:
                return ValueRanges.unknown()
            return ValueRanges.wrap(1.0)

        if b < 0:
            a = cls.reciprocal(a)
            b = -b

        if a == ValueRanges.unknown():
            return ValueRanges.unknown()

        # If the base is positive, then we're good, otherwise nothing's defined
        if a.lower >= 0:
            return ValueRanges.increasing_map(a, lambda x: x**b)
        else:
            return ValueRanges.unknown()
        """

    @staticmethod
    def reciprocal(x):
        """Needed as it's used in pow, but it won't appear on a SymPy expression"""
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges.unknown()
        else:
            return ValueRanges.decreasing_map(x, lambda y: FloatTrueDiv(1.0, y))  # type: ignore[operator]

    @staticmethod
    def abs(x):
        return ValueRanges.convex_min_zero_map(x, abs)

    @staticmethod
    def exp(x):
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_exp)

    @staticmethod
    def log(x):
        x = ValueRanges.wrap(x)
        if x.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_log)

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
        return ValueRanges.coordinatewise_increasing_map(a, b, fn)

    @classmethod
    def floor_to_int(cls, x, dtype):
        return ValueRanges.increasing_map(x, sympy.functions.elementary.integers.floor)

    @classmethod
    def ceil_to_int(cls, x, dtype):
        return ValueRanges.increasing_map(
            x, sympy.functions.elementary.integers.ceiling
        )

    # I think these implementations are sound.  The hazard here is that sympy
    # will carry out the floor/ceil at too high precision and then something
    # bad will happen when we convert it to float.
    #
    # For truncation, the implementation is clearly sound, because the desired
    # target float is always exactly representable, since you're just chopping
    # off bits the mantissa.  But what about ceil/floor?
    #
    # The important constraint here is that we're not defining floor on
    # arbitrary real numbers, only representable float numbers.  So we can
    # take advantage of the fact that before we reach the first
    # unrepresentable integer in floating point space, we have the range of
    # numbers corresponding to exponent zero: all integers, with no fractional
    # amounts.  floor/ceil is an identity operation in this case.  In the
    # range below here, representable floating point numbers are spaced
    # exactly 1/2 apart, and notably, both the floor/ceil are defined floating
    # point numbers.  There is no "gap" as you step up to the next exponent.

    @classmethod
    def floor(cls, x):
        return ValueRanges.increasing_map(
            x, _keep_float(sympy.functions.elementary.integers.floor)
        )

    @classmethod
    def ceil(cls, x):
        return ValueRanges.increasing_map(
            x, _keep_float(sympy.functions.elementary.integers.ceiling)
        )

    @classmethod
    def round_decimal(cls, number, ndigits):
        if not ndigits.is_singleton():
            return ValueRanges.unknown()

        ndigits = ndigits.lower
        # We can't use functools.partial here since sympy doesn't support keyword arguments, but we have to bind
        # the second parameter.
        fn = lambda number: RoundDecimal(number, ndigits)  # type: ignore[misc, assignment]  # noqa: E731

        return ValueRanges.increasing_map(number, fn)

    @classmethod
    def round_to_int(cls, number, dtype):
        return ValueRanges.increasing_map(number, RoundToInt)

    # It's used in some models on symints
    @staticmethod
    def sqrt(x):
        x = ValueRanges.wrap(x)
        if x.lower < 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_sqrt)

    @staticmethod
    def where(a, b, c):
        b = ValueRanges.wrap(b)
        c = ValueRanges.wrap(c)
        a = a.boolify()
        # We sometimes write unknown without specifying the type correctly
        # In particular, we do that when initialising the bounds for loads in bounds.py
        assert b.is_bool == c.is_bool or ValueRanges.unknown() in (b, c)
        if b.is_bool:
            return ValueRanges(sympy.And(b.lower, c.lower), sympy.Or(b.upper, c.upper))
        else:
            return ValueRanges(sympy.Min(b.lower, c.lower), sympy.Max(b.upper, c.upper))

    # expr_cond_pair is used to represent a single (expr, condition) pair in piecewise.
    # We just return the value range of the expression and its corresponding condition as a tuple
    # and defer the analysis to piecewise
    @staticmethod
    def expr_cond_pair(a, b):
        b = b.boolify()
        return (a, b)

    # piecewise function can be used to convert a SymBool to SymInt:
    # int_expr = Piecewise((1, bool_expr), (0, True)), it evalutes to 1 when sym_bool is True and 0 otherwise.
    #
    # ranges is a sequence of (expr_range, condition_range) pairs. The range pair is constructed in expr_cond_pair.
    # The ValueRange of Piecewise is just the union of all expr ranges whose condition expr can be True.
    @staticmethod
    def piecewise(*ranges):
        init_range = None
        for expr_range, cond_range in ranges:
            if sympy.true in cond_range:
                if init_range is None:
                    init_range = expr_range
                else:
                    init_range = init_range | expr_range
        return init_range

    @staticmethod
    def cos(x):
        # TODO: We should tighten value ranges
        # If input range span is pi + 2*pi*k, then output range is (-1, 1)
        # otherwise the minimum of the value of the function on the extremes
        return ValueRanges(-1.0, 1.0)

    @staticmethod
    def cosh(x):
        return ValueRanges(0.0, sympy.oo)
        """
        x = ValueRanges.wrap(x)
        if x.lower > 0:
            return ValueRanges.increasing_map(x, OpaqueUnaryFn_cosh)
        elif x.upper < 0:
            return ValueRanges.decreasing_map(x, OpaqueUnaryFn_cosh)
        return ValueRanges(0.0, sympy.oo)
        """

    @staticmethod
    def sin(x):
        # TODO: We should tighten value ranges
        # See details on cos
        return ValueRanges(-1.0, 1.0)

    @staticmethod
    def sinh(x):
        # return ValueRanges.increasing_map(x, OpaqueUnaryFn_sinh)
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def tan(x):
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def tanh(x):
        # return ValueRanges.increasing_map(x, OpaqueUnaryFn_tanh)
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def asin(x):
        return ValueRanges(-sympy.oo, sympy.oo)
        """
        x = ValueRanges.wrap(x)
        if -1 <= x.lower and x.upper <= 1:
            return ValueRanges.increasing_map(x, OpaqueUnaryFn_asinh)
        return ValueRanges.unknown()
        """

    @staticmethod
    def acos(x):
        return ValueRanges(-sympy.oo, sympy.oo)
        """
        x = ValueRanges.wrap(x)
        if -1 <= x.lower and x.upper <= 1:
            return ValueRanges.decreasing_map(x, OpaqueUnaryFn_acos)
        return ValueRanges.unknown()
        """

    @staticmethod
    def atan(x):
        return ValueRanges(-sympy.oo, sympy.oo)
        # return ValueRanges.increasing_map(x, OpaqueUnaryFn_atan)

    @staticmethod
    def trunc(x):
        return ValueRanges.increasing_map(x, TruncToFloat)


class ValueRangeAnalysis(SymPyValueRangeAnalysis):
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

    @classmethod
    def index_expr(cls, index, dtype):
        assert isinstance(index, ValueRanges)
        return cls.to_dtype(index, dtype)

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None):
        x = ValueRanges.wrap(x)

        if dtype == torch.bool:
            if x.is_singleton():
                return ValueRanges.wrap(x.lower != 0)
            elif x.is_bool:
                return x
            elif 0 not in x:
                return ValueRanges.wrap(sympy.true)
            else:
                return ValueRanges(sympy.false, sympy.true)

        def cast(x, dtype):
            # dtype is int or float
            if dtype.is_floating_point:
                return sympy.Float(x)
            else:
                if x in (int_oo, -int_oo):
                    return x
                try:
                    return sympy.Integer(x)
                except TypeError:
                    # inf cannot be cast to Integer
                    return x

        if x.is_bool:
            if x.is_singleton():
                val = 1 if x.lower else 0
                return ValueRanges.wrap(cast(val, dtype))
            else:
                return ValueRanges(cast(0, dtype), cast(1, dtype))
        else:
            # int to float or float to int
            return ValueRanges(cast(x.lower, dtype), cast(x.upper, dtype))

    @staticmethod
    def square(x):
        return ValueRanges.convex_min_zero_map(x, lambda y: PowByNatural(y, 2))

    @staticmethod
    def neg(x):
        return ValueRanges.decreasing_map(x, operator.neg)

    # TODO: this is slightly inaccurate because truncdiv operates at integer
    # precision, but we're going through float truediv which means we can
    # potentially lose precision on the bounds
    @classmethod
    def truncdiv(cls, a, b):
        x = cls.truediv(a, b)
        if x == ValueRanges.unknown():
            return x

        return cls.trunc(x)

    @classmethod
    def sub(cls, a, b):
        return cls.add(a, cls.neg(b))

    def __getattr__(self, name):
        log.debug("unhandled ValueRange op %s", name)
        return self.default_handler


def bound_sympy(
    expr: sympy.Expr, ranges: Optional[Dict[sympy.Symbol, ValueRanges]] = None
) -> ValueRanges:
    log.debug(
        "bound_sympy(%s)%s",
        expr,
        LazyString(
            lambda: "\n"
            + "\n".join(
                f"  {k}: {r}" for k, r in ranges.items() if k in expr.free_symbols
            )
            if ranges
            else ""
        ),
    )
    if isinstance(expr, sympy.Number):
        return ValueRanges.wrap(expr)

    ranges = ranges or {}

    # If there's a tracing context, augment available constrained ranges.
    context = torch._guards.TracingContext.try_get()
    if context and context.fake_mode.shape_env:
        ranges = {**context.fake_mode.shape_env.var_to_range, **ranges}

    unbounded_vars = expr.free_symbols - ranges.keys()
    if unbounded_vars:
        # Give some bounds to the free variables via their SymPy assumptions
        # TODO A better way of doing this would be to assign them a range upon creation, as
        #      size variables can come with a lower bound of 2, as we specialise on 0 and 1
        unbounded_ranges: Dict[sympy.Symbol, ValueRanges] = {}
        for s in unbounded_vars:
            if s.is_integer:  # type: ignore[attr-defined]
                if s.is_positive:  # type: ignore[attr-defined]
                    lower = 1
                elif s.is_nonnegative:  # type: ignore[attr-defined]
                    lower = 0
                else:
                    lower = -math.inf  # type: ignore[assignment]
            else:
                # Don't bother trying very hard here
                lower = -math.inf  # type: ignore[assignment]
            unbounded_ranges[s] = ValueRanges(lower, math.inf)  # type: ignore[index]
        ranges = {**ranges, **unbounded_ranges}

    return sympy_interp(SymPyValueRangeAnalysis, ranges, expr)
