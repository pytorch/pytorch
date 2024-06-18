# mypy: allow-untyped-defs
import functools
import math
import operator
import sys

import sympy
from sympy import S

from .numbers import int_oo

__all__ = [
    "FloorDiv",
    "ModularIndexing",
    "CleanDiv",
    "CeilDiv",
    "IntTrueDiv",
    "FloatTrueDiv",
    "LShift",
    "RShift",
    "IsNonOverlappingAndDenseIndicator",
    "RoundToInt",
    "RoundDecimal",
    "ToFloat",
    "FloatPow",
    "PowByNatural",
    "Identity",
]


def _keep_float(f):
    @functools.wraps(f)
    def inner(*args):
        r = f(*args)
        if any(isinstance(a, sympy.Float) for a in args) and not isinstance(
            r, sympy.Float
        ):
            r = sympy.Float(float(r))
        return r

    return inner


def fuzzy_eq(x, y):
    if None in (x, y):
        return None
    return x == y


# It would be nice to have assertions on whether or not inputs is_integer
# However, with bugs like https://github.com/sympy/sympy/issues/26620 sympy
# sometimes inconsistently reports floats an integers.
#
# What we can assume from sympy is that if something is an int, it
# definitely is is_integer, but if it is a float it may or may not
# be is_integer.  So we are unable to do strong asserts that things
# are NOT integers.


# TODO: In Triton, // rounds to zero, but in Python, it is floor division.
# When we can prove both arguments are non-negative, we should just have a
# GenericFloorDiv (name pending) which can codegen efficiently in Python/C,
# and then PythonFloorDiv and CIntDiv which have the appropriate rounding
# semantics.
#
# Right now, FloorDiv de facto changes behavior if arguments are negative or
# not, this can potentially cause correctness issues.
class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)

    NB: This is Python-style floor division, round to -Inf
    """

    nargs = (2,)
    precedence = 50  # precedence of mul  # noqa: F811

    is_integer = True

    @property
    def base(self):
        return self.args[0]

    @property
    def divisor(self):
        return self.args[1]

    def _sympystr(self, printer):
        base = printer.parenthesize(self.base, self.precedence)
        divisor = printer.parenthesize(self.divisor, self.precedence)
        return f"({base}//{divisor})"

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(cls, base, divisor):
        # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
        # Assert triggered by inequality solver
        # assert base.is_integer, base
        # assert divisor.is_integer, divisor

        # We don't provide the same error message as in Python because SymPy
        # makes it difficult to check the types.
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")
        if base in (int_oo, -int_oo, sympy.oo, -sympy.oo) and divisor in (
            int_oo,
            -int_oo,
            sympy.oo,
            -sympy.oo,
        ):
            return sympy.nan
        if base is sympy.nan or divisor is sympy.nan:
            return sympy.nan

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and divisor == 1:
            return base
        if base.is_integer and divisor == -1:
            return sympy.Mul(base, -1)
        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (
                base in (int_oo, -int_oo, sympy.oo, -sympy.oo)
                or divisor in (int_oo, -int_oo, sympy.oo, -sympy.oo)
            )
        ):
            r = float(base) / float(divisor)
            if r == math.inf:
                return int_oo
            elif r == -math.inf:
                return -int_oo
            elif math.isnan(r):
                return sympy.nan
            else:
                return sympy.Integer(math.floor(r))
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return sympy.Integer(int(base) // int(divisor))
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        # Expands (x + y) // b into x // b + y // b.
        # This only works if floor is an identity, i.e. x / b is an integer.
        for term in sympy.Add.make_args(base):
            quotient = term / divisor
            if quotient.is_integer and isinstance(divisor, sympy.Integer):
                # NB: this is correct even if the divisor is not an integer, but it
                # creates rational expressions that cause problems with dynamic
                # shapes.
                return FloorDiv(base - term, divisor) + quotient

        try:
            gcd = sympy.gcd(base, divisor)
            if gcd != 1:
                return FloorDiv(
                    sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
                )
        except sympy.PolynomialError:
            pass  # https://github.com/pytorch/pytorch/issues/108276


class ModularIndexing(sympy.Function):
    """
    ModularIndexing(a, b, c) => (a // b) % c where % is the C modulus
    """

    nargs = (3,)
    is_integer = True

    @classmethod
    def eval(cls, base, divisor, modulus):
        if base == 0 or modulus == 1:
            return sympy.Integer(0)

        if (
            isinstance(base, sympy.Integer)
            and isinstance(divisor, sympy.Integer)
            and isinstance(modulus, sympy.Integer)
        ):
            return (base // divisor) % modulus

        try:
            if divisor != 1:
                gcd = sympy.gcd(base, divisor)
                if gcd != 1:
                    return ModularIndexing(
                        sympy.simplify(base / gcd),
                        sympy.simplify(divisor / gcd),
                        modulus,
                    )
        except sympy.PolynomialError:
            pass  # https://github.com/pytorch/pytorch/issues/108276

        if isinstance(base, sympy.Add):
            new_terms = []
            all_positive = True
            for term in base.args:
                if sympy.gcd(term, modulus * divisor) != modulus * divisor:
                    if (isinstance(term, sympy.Integer) and term < 0) or (
                        isinstance(term, sympy.Mul)
                        and isinstance(term.args[0], sympy.Integer)
                        and term.args[0] < 0
                    ):
                        # workaround for https://github.com/openai/triton/issues/619,
                        # if there are negative terms, // produces wrong result
                        # TODO if https://github.com/openai/triton/issues/619 is fixed
                        # this optimization would become valid
                        all_positive = False
                        break
                    else:
                        new_terms.append(term)

            if len(new_terms) != len(base.args) and all_positive:
                return ModularIndexing(sum(new_terms), divisor, modulus)

        if isinstance(base, FloorDiv):
            return ModularIndexing(base.args[0], base.args[1] * divisor, modulus)

    def _eval_is_nonnegative(self):
        p, q = self.args[:2]
        return fuzzy_eq(p.is_nonnegative, q.is_nonnegative)  # type: ignore[attr-defined]

    def _eval_is_positive(self):
        p, q = self.args[:2]
        return fuzzy_eq(p.is_positive, q.is_positive)  # type: ignore[attr-defined]


class Where(sympy.Function):
    """
    Good ol' ternary operator
    """

    nargs = (3,)

    def _eval_is_integer(self):
        return True if self.args[1].is_integer and self.args[2].is_integer else None  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self):
        return (
            True
            if self.args[1].is_nonnegative and self.args[2].is_nonnegative  # type: ignore[attr-defined]
            else None
        )

    def _eval_is_positive(self):
        return True if self.args[1].is_positive and self.args[2].is_positive else None  # type: ignore[attr-defined]

    @classmethod
    def eval(cls, c, p, q):
        if c == sympy.true:
            return p
        elif c == sympy.false:
            return q


# Python-style modulus: take sign from RHS
class PythonMod(sympy.Function):
    nargs = (2,)

    is_integer = True

    @classmethod
    def eval(cls, p, q):
        # python test/dynamo/test_export.py -k ExportTests.test_trivial_constraint
        # Triggered by sympy.solvers.inequalities.reduce_inequalities
        # assert p.is_integer, p
        # assert q.is_integer, q

        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")

        # Three cases:
        #   1. p == 0
        #   2. p is either q or -q
        #   3. p is integer and q == 1
        if p is S.Zero or p in (q, -q) or q == 1:
            return S.Zero

        # Evaluate if they are both literals.
        if q.is_Number and p.is_Number:
            return p % q

        # If q == 2, it's a matter of whether p is odd or even.
        if q.is_Number and q == 2:
            if p.is_even:
                return S.Zero
            if p.is_odd:
                return S.One

        # If p is a multiple of q.
        r = p / q
        if r.is_integer:
            return S.Zero

        # If p < q and its ratio is positive, then:
        #   - floor(p / q) = 0
        #   - p % q = p - floor(p / q) * q = p
        less = p < q
        if less.is_Boolean and bool(less) and r.is_positive:
            return p

        if sympy.Mod(p, q) == 0:
            return S.Zero

    # NB: args[1] for PythonMod
    def _eval_is_nonnegative(self):
        return True if self.args[1].is_positive else None  # type: ignore[attr-defined]

    def _eval_is_nonpositive(self):
        return True if self.args[1].is_negative else None  # type: ignore[attr-defined]


# Generic modulus: only defined on non-negative arguments
class Mod(sympy.Function):
    nargs = (2,)

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, p, q):
        # This was adapted from: sympy/core/mod.py

        # Triggered by
        # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
        # assert p.is_integer, p
        # assert q.is_integer, q

        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")

        # Three cases:
        #   1. p == 0
        #   2. p is either q or -q
        #   3. p is integer and q == 1
        if p is S.Zero or p in (q, -q) or q == 1:
            return S.Zero

        # Evaluate if they are both literals.
        if q.is_Number and p.is_Number:
            assert p >= 0, p
            assert q >= 1, q
            return p % q

        # If q == 2, it's a matter of whether p is odd or even.
        if q.is_Number and q == 2:
            if p.is_even:
                return S.Zero
            if p.is_odd:
                return S.One

        # If p is a multiple of q.
        r = p / q
        if r.is_integer:
            return S.Zero

        # If p < q and its ratio is positive, then:
        #   - floor(p / q) = 0
        #   - p % q = p - floor(p / q) * q = p
        less = p < q
        if less.is_Boolean and bool(less) and r.is_positive:
            return p


class CleanDiv(FloorDiv):
    """
    Div where we can assume no rounding.
    This is to enable future optimizations.
    """

    pass


# Don't use sympy ceiling/floor as they will attempt simplifications involving
# frac
class CeilToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number):
        # assert number.is_integer is not True, number
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.ceil(float(number)))


class FloorToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number):
        # assert number.is_integer is not True, number
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, int_oo):
            return -int_oo
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.floor(float(number)))


class CeilDiv(sympy.Function):
    """
    Div used in indexing that rounds up.
    """

    is_integer = True

    def __new__(cls, base, divisor):
        base = sympy.sympify(base)
        divisor = sympy.sympify(divisor)
        if sympy.gcd(base, divisor) == divisor:
            return CleanDiv(base, divisor)
        else:
            return FloorDiv(base + (divisor - 1), divisor)


class LShift(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, base, shift):
        if shift < 0:
            raise ValueError("negative shift count")
        return base * 2**shift


class RShift(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, base, shift):
        if shift < 0:
            raise ValueError("negative shift count")
        return base // 2**shift


def safe_pow(base, exp):
    sign = 1
    if base < 0:
        base = -base
        sign = 1 if exp % 2 == 0 else -1
    return sign * _safe_pow(base, exp)


# Prevent people from overflowing pow
def _safe_pow(base, exponent):
    if exponent < 0:
        raise ValueError("Exponent must be non-negative.")

    if exponent == 0:
        return 1

    half_exp = safe_pow(base, exponent // 2)
    if half_exp is int_oo:
        return int_oo

    # TODO: microoptimization is to avoid overflowing into arbitrary precision
    # and detect overflow prior to doing operations

    result = half_exp * half_exp
    if result > sys.maxsize:
        return int_oo

    if exponent % 2 == 1:
        result *= base
        if result > sys.maxsize:
            return int_oo

    return result


class PowByNatural(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, base, exp):
        if isinstance(base, sympy.Integer) and isinstance(exp, sympy.Integer):
            r = safe_pow(base, exp)
            if r in (-int_oo, int_oo):
                return r
            return sympy.Integer(r)
        if isinstance(exp, sympy.Integer):
            # Rely on regular sympy Pow for this (note that iterated
            # multiplication turns into a Pow anyway, you can't escape!!)
            return sympy.Pow(base, exp)
        if exp in (int_oo, sympy.oo):
            if base.is_nonnegative:
                return int_oo
            elif base.is_negative:
                return sympy.zoo  # this is apparently what (-2)**sympy.oo does
        # NB: do NOT translate into sympy.Pow, we will lose knowledge that exp
        # is a natural number if we do


# base is assumed to be nonnegative, thereby prevent complex numbers from
# occuring
class FloatPow(sympy.Function):
    is_integer = False
    is_real = True

    @classmethod
    def eval(cls, base, exp):
        # NB: These test sympy.Number, not sympy.Float, because:
        #   - Sometimes we may have sympy.oo or int_oo, and that's not a Float
        #     (but coerces to math.Inf)
        #   - Sometimes Float(0.0) will unpredictably decay to Integer(0),
        #     but we should still accept it in floatey contexts
        if isinstance(base, sympy.Number) and isinstance(exp, sympy.Number):
            return sympy.Float(float(base) ** float(exp))
        # NB: do not do any nontrivial reasoning


# Overloaded to be compatible with regular Python.
# https://github.com/pytorch/pytorch/issues/90900
#
# In particular, sympy division is willing to simplify x/x == 1
# where 1 is an integer, but this must be a float if x was float.
class FloatTrueDiv(sympy.Function):
    is_integer = False
    is_real = True

    @classmethod
    def eval(cls, base, divisor):
        # assert base.is_integer is not True, base
        # assert divisor.is_integer is not True, divisor

        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if isinstance(base, sympy.Number) and isinstance(divisor, sympy.Number):
            return sympy.Float(float(base) / float(divisor))


# Overloaded to be compatible with regular Python.  We distinguish this from
# FloatTrueDiv, because the code generation has to be different for this case:
# Python has a fancy algorithm for integer true division that isn't just
# "promote both arguments to float and use float division", so you need to
# codegen it differently.  While technically you can work it out from the
# types of the input, this is often inconvenient to do in Inductor codegen,
# so just have a different operator
# NB: Right now, Inductor codegen doesn't implement this correctly lol
class IntTrueDiv(sympy.Function):
    is_integer = False
    is_real = True

    @classmethod
    def eval(cls, base, divisor):
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (
                base in (int_oo, -int_oo, sympy.oo, -sympy.oo)
                or divisor in (int_oo, -int_oo, sympy.oo, -sympy.oo)
            )
        ):
            # Don't have to worry about precision here, you're getting zero or
            # inf from the division
            return sympy.Float(float(base) / float(divisor))
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return sympy.Float(int(base) / int(divisor))


# TODO: As an indicator, this != 0 implies == 1 (and vice versa).
# Because we do not have the ability to guard on the stride permutation
# at the moment, it is hard to make further inferences when this is true,
# as although we know the tensor is contiguous in *some* layout, we don't
# know which one (however, you could, for example, make the inference that
# reshaping this to a 1D tensor can be guard-free.)
class IsNonOverlappingAndDenseIndicator(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, *args):
        assert len(args) % 2 == 0
        dim = len(args) // 2
        sizes = args[0:dim]
        strides = args[dim:]

        # sym_node imported in torch.__init__. Local import to avoid an import cycle
        from torch.fx.experimental.symbolic_shapes import (
            eval_is_non_overlapping_and_dense,
        )

        if all(isinstance(a, sympy.Integer) for a in args):
            return eval_is_non_overlapping_and_dense(
                [int(a) for a in sizes], [int(a) for a in strides]
            )

        if dim == 1:
            # Manually implement the rank one short circuit
            if strides[0].is_Number and strides[0] == 1:
                return 1

            if sizes[0].is_Number and sizes[0] < 2:
                return 1

            # return 0 case covered by case above

            # TODO: Inability to access size-obliviousness sucks: if we have a
            # size oblivious test on a size-like unbacked SymInt, we could
            # confidently return zero when we have a size-like u0 stride
            # and a size-like u1 size.  Maybe a fancy ValueRanges analysis for
            # this function could help figure this out.

        if all(isinstance(a, sympy.Integer) for a in strides):
            assert dim != 0
            # When all strides are integral, we can sort, and the size for the
            # largest stride doesn't matter and can be arbitrarily symbolic
            s_sizes, s_strides = zip(
                *sorted(zip(sizes, strides), key=operator.itemgetter(1))
            )
            # Put something arbitrary in the max size spot, it'll be ignored
            if all(isinstance(a, sympy.Integer) for a in s_sizes[:-1]):
                s_sizes = s_sizes[:-1] + (42,)
                # We can reuse the regular eval, because it is invariant to
                # permutation of dimensions
                return eval_is_non_overlapping_and_dense(
                    [int(a) for a in s_sizes], [int(a) for a in s_strides]
                )

        return None


# NB: this is inconsistent with math.trunc in Python
class TruncToFloat(sympy.Function):
    is_integer = False
    is_real = True

    @classmethod
    def eval(cls, number):
        # assert number.is_integer is not True, number
        if isinstance(number, sympy.Number):
            # NB: It is safe to use truncation to integer, which is what
            # math.trunc does, as Python integers are arbitrary precision and
            # so we are guaranteed not to lose precision when we do this
            return sympy.Float(math.trunc(float(number)))


class TruncToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number):
        # assert number.is_integer is not True, number
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.trunc(float(number)))


# This is float -> int
class RoundToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number):
        # assert number.is_integer is not True, number

        if number is sympy.oo:
            return int_oo
        if number is -sympy.oo:
            return -int_oo
        if isinstance(number, sympy.Number):
            return sympy.Integer(round(float(number), 0))


# To get float -> int, Python style round semantics.
#
#   x = PyFloat_AsDouble(self);
#   if (o_ndigits == Py_None) {
#       /* single-argument round or with None ndigits:
#        * round to nearest integer */
#       rounded = round(x);
#       if (fabs(x-rounded) == 0.5)
#           /* halfway case: round to even */
#           rounded = 2.0*round(x/2.0);
#       return PyLong_FromDouble(rounded);
#   }


# NB: Like Round, this only ever returns floats.  ndigits cannot be None
class RoundDecimal(sympy.Function):
    is_integer = False
    is_real = True

    @classmethod
    def eval(cls, number, ndigits):
        # assert number.is_integer is not True, number

        if isinstance(number, sympy.Number) and isinstance(ndigits, sympy.Integer):
            return sympy.Float(round(float(number), int(ndigits)))


class ToFloat(sympy.Function):
    is_integer = False
    is_real = True

    @classmethod
    def eval(cls, number):
        if number in [sympy.oo, -sympy.oo]:
            return number

        if isinstance(number, sympy.Integer):
            return sympy.Float(int(number))
        if number is int_oo:
            return sympy.oo
        if number is -int_oo:
            return -sympy.oo


class Identity(sympy.Function):
    """
    Prevents expansion and other optimizations
    """

    def __repr__(self):
        return f"Identity({self.args[0]})"

    def _eval_is_real(self):
        return self.args[0].is_real

    def _eval_is_integer(self):
        return self.args[0].is_integer  # type: ignore[attr-defined]


def make_opaque_unary_fn(name):
    class OpaqueUnaryFn(sympy.Function):
        """
        Unlike the builtin sympy functions on real numbers like sympy.sqrt,
        these equivalents do not do any nontrivial reasoning besides
        constant propagation.  This helps avoid performing transformations
        that are valid for real numbers but are invalid for floating point;
        in particular, while we are willing to make optimizations that change
        numerics for Tensor compute, we are NOT willing to make optimziations
        that change numerics for size compute.
        """

        _torch_handler_name = name

        @classmethod
        def eval(cls, a):
            if isinstance(a, (sympy.Integer, sympy.Float)):
                # Python converts to float64 before computing, c.f.
                # >>> math.sin(2**53+1)
                # -0.848925964814655
                # >>> math.sin(float(2**53+1))
                # -0.848925964814655
                try:
                    return sympy.Float(getattr(math, name)(float(a)))
                # Just use sympy semantics for infinity/overflow, you might get some
                # weird objects but ask silly questions, get silly answers
                except OverflowError:
                    return getattr(sympy, name)(a)
            elif a in [sympy.oo, -sympy.oo, sympy.zoo, -sympy.zoo, int_oo, -int_oo]:
                if a is int_oo:
                    a = sympy.oo
                if a is -int_oo:
                    a = -sympy.oo
                return getattr(sympy, name)(a)
            return None

    OpaqueUnaryFn.__name__ = "OpaqueUnaryFn_" + name

    return OpaqueUnaryFn


# Keep in sync with math_op_names in torch/fx/experimental/sym_node.py
OpaqueUnaryFn_sqrt = make_opaque_unary_fn("sqrt")
OpaqueUnaryFn_cos = make_opaque_unary_fn("cos")
OpaqueUnaryFn_cosh = make_opaque_unary_fn("cosh")
OpaqueUnaryFn_sin = make_opaque_unary_fn("sin")
OpaqueUnaryFn_sinh = make_opaque_unary_fn("sinh")
OpaqueUnaryFn_tan = make_opaque_unary_fn("tan")
OpaqueUnaryFn_tanh = make_opaque_unary_fn("tanh")
OpaqueUnaryFn_asin = make_opaque_unary_fn("asin")
OpaqueUnaryFn_acos = make_opaque_unary_fn("acos")
OpaqueUnaryFn_atan = make_opaque_unary_fn("atan")
OpaqueUnaryFn_exp = make_opaque_unary_fn("exp")
OpaqueUnaryFn_log = make_opaque_unary_fn("log")
OpaqueUnaryFn_asinh = make_opaque_unary_fn("asinh")
