# mypy: allow-untyped-defs
import math

import sympy
from sympy import S
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or

__all__ = [
    "FloorDiv",
    "ModularIndexing",
    "CleanDiv",
    "CeilDiv",
    "Pow",
    "TrueDiv",
    "LShift",
    "RShift",
    "IsNonOverlappingAndDenseIndicator",
    "Round",
    "RoundDecimal",
]


def fuzzy_eq(x, y):
    if None in (x, y):
        return None
    return x == y


class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)
    """

    nargs = (2,)
    precedence = 50  # precedence of mul  # noqa: F811

    # Default return type for SymPy assumptions.
    # https://docs.sympy.org/latest/guides/assumptions.html#implementing-assumptions-handlers
    is_real = True

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

    # SymPy assumptions based on argument types.
    def _eval_is_real(self):
        return fuzzy_or([self.base.is_real, self.divisor.is_real])

    def _eval_is_integer(self):
        return fuzzy_and([self.base.is_integer, self.divisor.is_integer])

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(cls, base, divisor):
        def check_supported_type(x):
            if (
                x.is_integer is False and x.is_real is False and x.is_complex
            ) or x.is_Boolean:
                raise TypeError(
                    f"unsupported operand type(s) for //: "
                    f"'{type(base).__name__}' and '{type(divisor).__name__}'"
                    f", expected integer or real"
                )

        check_supported_type(base)
        check_supported_type(divisor)

        # We don't provide the same error message as in Python because SymPy
        # makes it difficult to check the types.
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and divisor == 1:
            return base
        if base.is_real and divisor == 1:
            return sympy.floor(base)
        if base.is_integer and divisor == -1:
            return sympy.Mul(base, -1)
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return base // divisor
        if isinstance(base, (sympy.Integer, sympy.Float)) and isinstance(
            divisor, (sympy.Integer, sympy.Float)
        ):
            return sympy.floor(base / divisor)
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)
        if isinstance(divisor, sympy.Rational) and divisor.p == 1:
            return sympy.floor(base * divisor.q)

        if isinstance(base, sympy.Add):
            for a in base.args:
                gcd = sympy.gcd(a, divisor)
                if gcd == divisor:
                    return FloorDiv(base - a, divisor) + a / gcd

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

    @classmethod
    def eval(cls, c, p, q):
        if c == sympy.true:
            return p
        elif c == sympy.false:
            return q


class Mod(sympy.Function):
    """
    We maintain this so that we avoid SymPy correctness issues, such as:
    https://github.com/sympy/sympy/issues/25146
    """

    nargs = (2,)

    @classmethod
    def eval(cls, p, q):
        # This was adapted from: sympy/core/mod.py

        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")
        # If either of them is NaN or infinite.
        if p is S.NaN or q is S.NaN or p.is_finite is False or q.is_finite is False:
            return S.NaN
        # Three cases:
        #   1. p == 0
        #   2. p is either q or -q
        #   3. p is integer and q == 1
        if p is S.Zero or p in (q, -q) or (p.is_integer and q == 1):
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

    def _eval_is_integer(self):
        p, q = self.args
        return fuzzy_and([p.is_integer, q.is_integer, fuzzy_not(q.is_zero)])  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self):
        return True if self.args[1].is_positive else None  # type: ignore[attr-defined]

    def _eval_is_nonpositive(self):
        return True if self.args[1].is_negative else None  # type: ignore[attr-defined]


class CleanDiv(FloorDiv):
    """
    Div where we can assume no rounding.
    This is to enable future optimizations.
    """

    pass


class CeilDiv(sympy.Function):
    """
    Div used in indexing that rounds up.
    """

    is_integer = True

    def __new__(cls, base, divisor):
        if sympy.gcd(base, divisor) == divisor:
            return CleanDiv(base, divisor)
        else:
            return FloorDiv(base + (divisor - 1), divisor)


class LShift(sympy.Function):
    @classmethod
    def eval(cls, base, shift):
        if shift < 0:
            raise ValueError("negative shift count")
        return base * 2**shift


class RShift(sympy.Function):
    @classmethod
    def eval(cls, base, shift):
        if shift < 0:
            raise ValueError("negative shift count")
        return base // 2**shift


# Overloaded to be compatible with regular Python.
# https://github.com/pytorch/pytorch/issues/90900
class Pow(sympy.Function):
    @classmethod
    def eval(cls, base, exp):
        if exp.is_zero:
            return sympy.Integer(1)
        elif base.is_zero and exp < 0:
            raise ZeroDivisionError(f"{base} cannot be raised to a negative power")
        else:
            return base**exp


# Overloaded to be compatible with regular Python.
# https://github.com/pytorch/pytorch/issues/90900
class TrueDiv(sympy.Function):
    @classmethod
    def eval(cls, base, divisor):
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")
        else:
            return base / divisor


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
        # TODO: it is possible to make progress evaluating this guard
        # even if not all of the inputs are known.  For example, a 2D
        # tensor with non-0/1 sizes but strides (0, 1) is definitely
        # false, because we know its numel > 1 but it's broadcasted
        # in dim 0.
        if all(isinstance(a, sympy.Integer) for a in args):
            # sym_node imported in torch.__init__. Local import to avoid an import cycle
            from torch.fx.experimental.symbolic_shapes import (
                eval_is_non_overlapping_and_dense,
            )

            size_args = args[0:dim]
            stride_args = args[dim:]
            return eval_is_non_overlapping_and_dense(
                [int(a) for a in size_args], [int(a) for a in stride_args]
            )
        return None


class Trunc(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number):
        if number.is_integer:
            return number
        elif isinstance(number, sympy.Number):
            return sympy.Integer(math.trunc(float(number)))


class Round(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number):
        if number.is_integer:
            return number
        elif isinstance(number, sympy.Number):
            return sympy.Integer(round(float(number)))

    def __int__(self):
        # This will only ever be called when computing size hints. At that point, self.args[0] should be a number and
        # no longer an expression. If it were, the float call would fail and the caller would handle this further.
        return round(float(self.args[0]))  # type: ignore[arg-type]


class RoundDecimal(sympy.Function):
    @classmethod
    def eval(cls, number, ndigits):
        if number.is_integer and ndigits >= 0:
            return number
        elif isinstance(number, sympy.Number) and isinstance(ndigits, sympy.Integer):
            value_type, output_type = (
                (int, sympy.Integer)
                if isinstance(number, sympy.Integer)
                else (float, sympy.Float)
            )
            return output_type(round(value_type(number), int(ndigits)))


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
            elif a in [sympy.oo, -sympy.oo, sympy.zoo, -sympy.zoo]:
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
