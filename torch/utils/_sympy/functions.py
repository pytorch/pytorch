import sympy
from sympy import S
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or

__all__ = ["FloorDiv", "ModularIndexing", "CleanDiv", "CeilDiv", "LShift", "RShift"]


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
            if (x.is_integer is False and x.is_real is False and x.is_complex) or x.is_Boolean:
                raise TypeError(
                    f"unsupported operand type(s) for //: "
                    f"'{type(base).__name__}' and '{type(divisor).__name__}'"
                    f", expected integer or real")

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
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return base // divisor
        if isinstance(base, (sympy.Integer, sympy.Float)) and isinstance(divisor, (sympy.Integer, sympy.Float)):
            return sympy.floor(base / divisor)
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        if isinstance(base, sympy.Add):
            for a in base.args:
                gcd = sympy.gcd(a, divisor)
                if gcd == divisor:
                    return FloorDiv(base - a, divisor) + a / gcd

        gcd = sympy.gcd(base, divisor)
        if gcd != 1:
            return FloorDiv(
                sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
            )


class ModularIndexing(sympy.Function):
    """
    ModularIndexing(a, b, c) => (a // b) % c
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

        if divisor != 1:
            gcd = sympy.gcd(base, divisor)
            if gcd != 1:
                return ModularIndexing(
                    sympy.simplify(base / gcd), sympy.simplify(divisor / gcd), modulus
                )

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
            raise ValueError('negative shift count')
        return base * 2 ** shift


class RShift(sympy.Function):
    @classmethod
    def eval(cls, base, shift):
        if shift < 0:
            raise ValueError('negative shift count')
        return base // 2 ** shift
