# mypy: allow-untyped-defs
import functools
import math
import operator
import sys
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

import sympy
from sympy import S
from sympy.core import sympify
from sympy.core.expr import Expr
from sympy.core.function import Application
from sympy.core.logic import _torf, fuzzy_and, fuzzy_or
from sympy.core.numbers import equal_valued
from sympy.core.operations import LatticeOp, ShortCircuit
from sympy.core.sorting import ordered
from sympy.core.traversal import walk
from sympy.utilities.iterables import sift

from .numbers import int_oo


_T = TypeVar("_T", bound=SupportsFloat)

# Portions of this file are adapted from the Sympy codebase, which was
# licensed as follows:
#
#   Copyright (c) 2006-2023 SymPy Development Team
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#     a. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#     b. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     c. Neither the name of SymPy nor the names of its contributors
#        may be used to endorse or promote products derived from this software
#        without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#   DAMAGE.

__all__ = [
    "FloorDiv",
    "ModularIndexing",
    "Where",
    "PythonMod",
    "Mod",
    "CleanDiv",
    "CeilToInt",
    "FloorToInt",
    "CeilDiv",
    "IntTrueDiv",
    "FloatTrueDiv",
    "LShift",
    "RShift",
    "IsNonOverlappingAndDenseIndicator",
    "TruncToFloat",
    "TruncToInt",
    "RoundToInt",
    "RoundDecimal",
    "ToFloat",
    "FloatPow",
    "PowByNatural",
    "Identity",
]


def _is_symbols_binary_summation(expr: sympy.Expr) -> bool:
    # No need to check that two args are not the same, since expr is pr-optimized but we do it anyway.
    return (
        expr.is_Add
        and len(expr._args) == 2
        and expr._args[0].is_symbol
        and expr._args[1].is_symbol
        and expr._args[0] is not expr._args[1]
    )


def _keep_float(f: Callable[..., _T]) -> Callable[..., Union[_T, sympy.Float]]:
    @functools.wraps(f)
    def inner(*args: Any) -> Union[_T, sympy.Float]:
        r: Union[_T, sympy.Float] = f(*args)
        if any(isinstance(a, sympy.Float) for a in args) and not isinstance(
            r, sympy.Float
        ):
            r = sympy.Float(float(r))
        return r

    return inner


def fuzzy_eq(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
    if None in (x, y):
        return None
    return x == y


def simple_floordiv_gcd(p: sympy.Basic, q: sympy.Basic) -> sympy.Basic:
    """
    Fast path for sympy.gcd, using a simple factoring strategy.

    We try to rewrite p and q in the form n*e*p1 + n*e*p2 and n*e*q0,
    where n is the greatest common integer factor and e is the largest
    syntactic common factor (i.e., common sub-expression) in p and q.
    Then the gcd returned is n*e, cancelling which we would be left with
    p1 + p2 and q0.

    Note that further factoring of p1 + p2 and q0 might be possible with
    sympy.factor (which uses domain-specific theories). E.g., we are unable
    to find that x*y + x + y + 1 is divisible by x + 1. More generally,
    when q is of the form q1 + q2 (instead of being already factored) it
    might be necessary to fall back on sympy.gcd.
    """

    def integer_coefficient(x: sympy.Basic) -> int:
        integer_coefficients: List[int] = [
            abs(int(arg))
            for arg in sympy.Mul.make_args(x)
            if isinstance(arg, (int, sympy.Integer))
        ]
        return math.prod(integer_coefficients)

    def integer_factor(expr: sympy.Basic) -> int:
        integer_factors: Iterable[int] = map(
            integer_coefficient, sympy.Add.make_args(expr)
        )
        return functools.reduce(math.gcd, integer_factors)

    gcd: int = math.gcd(integer_factor(p), integer_factor(q))
    p, q = p / gcd, q / gcd  # type: ignore[operator, assignment]  # remove in py3.12

    base_splits: List[Tuple[sympy.Basic, ...]] = list(
        map(sympy.Mul.make_args, sympy.Add.make_args(p))
    )
    divisor_split: Tuple[sympy.Basic, ...] = sympy.Mul.make_args(q)
    for x in divisor_split:
        if all(x in base_split for base_split in base_splits):
            gcd = gcd * x  # type: ignore[operator]  # remove in py3.12
    return gcd  # type: ignore[return-value]  # remove in py3.12


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

    nargs: Tuple[int, ...] = (2,)
    precedence: int = 35  # lower precedence than add
    is_integer: bool = True

    @property
    def base(self) -> sympy.Basic:
        return self.args[0]

    @property
    def divisor(self) -> sympy.Basic:
        return self.args[1]

    def _sympystr(self, printer: sympy.printing.StrPrinter) -> str:
        base = printer.parenthesize(self.base, self.precedence)
        divisor = printer.parenthesize(self.divisor, self.precedence)
        return f"({base}//{divisor})"

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(
        cls, base: sympy.Integer, divisor: sympy.Integer
    ) -> Union[sympy.Basic, None]:
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
        if base.is_integer and equal_valued(divisor, 1):
            return base
        if base.is_integer and equal_valued(divisor, -1):
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
        if isinstance(divisor, sympy.Integer):
            quotients = 0
            terms = []
            for term in sympy.Add.make_args(base):
                quotient = term / divisor

                if quotient.is_integer:
                    terms.append(term)
                    quotients += quotient

            if len(terms) != 0:
                # Passing evaluate = False since expression will be optimized during the subtraction post its construction.
                return (
                    FloorDiv(base - sympy.Add(*terms, evaluate=False), divisor)
                    + quotients
                )

        try:
            gcd = simple_floordiv_gcd(base, divisor)
            if equal_valued(gcd, 1) and isinstance(divisor, sympy.Add):
                gcd = sympy.gcd(base, divisor)
            if not equal_valued(gcd, 1):
                return FloorDiv(
                    sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
                )
        except sympy.PolynomialError:
            pass  # https://github.com/pytorch/pytorch/issues/108276

        return None


class ModularIndexing(sympy.Function):
    """
    ModularIndexing(a, b, c) => (a // b) % c where % is the C modulus
    """

    nargs: Tuple[int, ...] = (3,)
    is_integer: bool = True
    precedence: int = 35  # lower precedence than add

    @classmethod
    def eval(
        cls, base: sympy.Integer, divisor: sympy.Integer, modulus: sympy.Integer
    ) -> Optional[sympy.Basic]:
        if base == 0 or modulus == 1:
            return sympy.S.Zero

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
            new_terms: List[sympy.Integer] = []
            all_positive: bool = True
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

        return None

    def _eval_is_nonnegative(self) -> Optional[bool]:
        p, q = self.args[:2]
        return fuzzy_eq(p.is_nonnegative, q.is_nonnegative)  # type: ignore[attr-defined]

    def _eval_is_positive(self) -> Optional[bool]:
        p, q = self.args[:2]
        return fuzzy_eq(p.is_positive, q.is_positive)  # type: ignore[attr-defined]


class Where(sympy.Function):
    """
    Good ol' ternary operator
    """

    nargs: Tuple[int, ...] = (3,)
    precedence: int = 35  # lower precedence than add

    def _eval_is_integer(self) -> Optional[bool]:
        return True if self.args[1].is_integer and self.args[2].is_integer else None  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self) -> Optional[bool]:
        return (
            True
            if self.args[1].is_nonnegative and self.args[2].is_nonnegative  # type: ignore[attr-defined]
            else None
        )

    def _eval_is_positive(self) -> Optional[bool]:
        return True if self.args[1].is_positive and self.args[2].is_positive else None  # type: ignore[attr-defined]

    @classmethod
    def eval(
        cls, c: sympy.Basic, p: sympy.Basic, q: sympy.Basic
    ) -> Optional[sympy.Basic]:
        if c == sympy.true:
            return p
        elif c == sympy.false:
            return q
        return None


# Python-style modulus: take sign from RHS
class PythonMod(sympy.Function):
    nargs: Tuple[int, ...] = (2,)

    precedence: int = 35  # lower precedence than add
    is_integer: bool = True

    @classmethod
    def eval(cls, p: sympy.Expr, q: sympy.Expr) -> Optional[sympy.Expr]:
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

        return None

    # NB: args[1] for PythonMod
    def _eval_is_nonnegative(self) -> Optional[bool]:
        return True if self.args[1].is_positive else None  # type: ignore[attr-defined]

    def _eval_is_nonpositive(self) -> Optional[bool]:
        return True if self.args[1].is_negative else None  # type: ignore[attr-defined]


# Generic modulus: only defined on non-negative arguments
class Mod(sympy.Function):
    nargs = (2,)
    precedence: int = 35  # lower precedence than add

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


class MinMaxBase(Expr, LatticeOp):  # type: ignore[misc]
    def __new__(cls, *original_args, **assumptions):
        from sympy.core.parameters import global_parameters

        evaluate = assumptions.pop("evaluate", global_parameters.evaluate)
        args = (sympify(arg) for arg in original_args)

        # See the comment in _satisfy_unique_summations_symbols.
        unique_summations_symbols = (
            None
            if not evaluate
            else cls._satisfy_unique_summations_symbols(original_args)
        )

        if evaluate:
            try:
                # first standard filter, for cls.zero and cls.identity
                # also reshape Max(a, Max(b, c)) to Max(a, b, c)
                args = frozenset(cls._new_args_filter(args))  # type: ignore[assignment]
            except ShortCircuit:
                return cls.zero  # type: ignore[attr-defined]

            # No need to run _collapse_arguments and _find_localzeros, see the comment
            # in _satisfy_unique_summations_symbols.
            if unique_summations_symbols is None:
                # remove redundant args that are easily identified
                args = cls._collapse_arguments(args, **assumptions)

                # find local zeros
                args = cls._find_localzeros(args, **assumptions)

        args = frozenset(args)

        if not args:
            return cls.identity  # type: ignore[attr-defined]

        if len(args) == 1:
            return list(args).pop()

        # base creation
        obj = Expr.__new__(cls, *ordered(args), **assumptions)
        obj._argset = args

        obj.unique_summations_symbols = unique_summations_symbols
        return obj

    @classmethod
    def _satisfy_unique_summations_symbols(
        cls, args
    ) -> Optional[set[sympy.core.symbol.Symbol]]:
        """
        One common case in some models is building expressions of the form
        max(max(max(a+b...), c+d), e+f) which is simplified to max(a+b, c+d, e+f, ...).
        For such expressions, we call the Max constructor X times (once for each nested
        max) and the expression gets flattened.

        An expensive cost in constructing those expressions is running _collapse_arguments
        and _find_localzeros. However, those two optimizations are unnecessary when the args
        to max are all of the form a+b, c+d, ..etc where each term uses a unique set of symbols.

        This function is used to detect such properties of the expressions we are building
        and if so inform that we do not need to run those optimizations. To detect those,
        we store a property in the expression that tells that this expression is a min/max
        operation over terms that use unique symbols "unique_summations_symbols". This property
        also memoize the set of symbols used in all the terms to make it faster to detect this
        property inductively.

        When we apply max to add a new term, all we need to do is check if the new term uses
        unique symbols (with respect to existing terms and itself).
        Example:
        t = Max(a+b, c+d) ==> satisfies the property
        Max(t, h+j)       ==> h,j not in [a,b,c,d] => satisfy the property.

        The function returns None if the new expression does not satisfy the unique_summations_symbols
        property. Otherwise, it returns a new set of unique symbols.
        """
        if len(args) != 2:
            return None

        (lhs, rhs) = (
            (args[1], args[0])
            if isinstance(args[1], MinMaxBase)
            else (args[0], args[1])
        )

        if not _is_symbols_binary_summation(rhs):
            return None

        # base case max(a+b, c+d) ==> satisfies the property if a+b and c+d use unique symbols.
        if _is_symbols_binary_summation(lhs):
            return cls._unique_symbols(args)

        # inductive case max(t, h+j) ==> satisfies the property if h, j not in t.unique_summations_symbols
        if isinstance(lhs, MinMaxBase):
            lhs_unique_summations_symbols = getattr(
                lhs, "unique_summations_symbols", None
            )
            if lhs_unique_summations_symbols is not None:
                return cls._unique_symbols([rhs], lhs_unique_summations_symbols)

        return None

    @classmethod
    def _unique_symbols(
        cls, args, initial_set: Optional[set[sympy.core.symbol.Symbol]] = None
    ) -> Optional[set[sympy.core.symbol.Symbol]]:
        """
        Return seen_symbols if all atoms in all args are all unique symbols,
        else returns None. initial_set can be used to represent initial value for seen_symbols
        """
        seen_symbols = set() if initial_set is None else initial_set
        for arg in args:
            for element in arg.atoms():
                if not isinstance(element, sympy.core.symbol.Symbol):
                    return None
                elif element in seen_symbols:
                    return None
                else:
                    seen_symbols.add(element)
        return seen_symbols

    @classmethod
    def _collapse_arguments(cls, args, **assumptions):
        """Remove redundant args.

        Examples
        ========

        >>> from sympy import Min, Max
        >>> from sympy.abc import a, b, c, d, e

        Any arg in parent that appears in any
        parent-like function in any of the flat args
        of parent can be removed from that sub-arg:

        >>> Min(a, Max(b, Min(a, c, d)))
        Min(a, Max(b, Min(c, d)))

        If the arg of parent appears in an opposite-than parent
        function in any of the flat args of parent that function
        can be replaced with the arg:

        >>> Min(a, Max(b, Min(c, d, Max(a, e))))
        Min(a, Max(b, Min(a, c, d)))
        """
        if not args:
            return args
        args = list(ordered(args))
        if cls is Min:
            other = Max
        else:
            other = Min  # type: ignore[assignment]

        # find global comparable max of Max and min of Min if a new
        # value is being introduced in these args at position 0 of
        # the ordered args
        if args[0].is_number:
            sifted = mins, maxs = [], []  # type: ignore[var-annotated]
            for i in args:
                for v in walk(i, Min, Max):
                    if v.args[0].is_comparable:
                        sifted[isinstance(v, Max)].append(v)
            small = Min.identity
            for i in mins:
                v = i.args[0]
                if v.is_number and (v < small) == True:  # noqa: E712
                    small = v
            big = Max.identity
            for i in maxs:
                v = i.args[0]
                if v.is_number and (v > big) == True:  # noqa: E712
                    big = v
            # at the point when this function is called from __new__,
            # there may be more than one numeric arg present since
            # local zeros have not been handled yet, so look through
            # more than the first arg
            if cls is Min:
                for arg in args:
                    if not arg.is_number:
                        break
                    if (arg < small) == True:  # noqa: E712
                        small = arg
            elif cls == Max:
                for arg in args:
                    if not arg.is_number:
                        break
                    if (arg > big) == True:  # noqa: E712
                        big = arg
            T = None
            if cls is Min:
                if small != Min.identity:
                    other = Max
                    T = small
            elif big != Max.identity:
                other = Min  # type: ignore[assignment]
                T = big
            if T is not None:
                # remove numerical redundancy
                for i in range(len(args)):
                    a = args[i]
                    if isinstance(a, other):
                        a0 = a.args[0]
                        if (  # noqa: E712
                            (a0 > T) if other == Max else (a0 < T)  # noqa: E712
                        ) == True:  # noqa: E712
                            args[i] = cls.identity  # type: ignore[attr-defined]

        # remove redundant symbolic args
        def do(ai, a):
            if not isinstance(ai, (Min, Max)):
                return ai
            cond = a in ai.args
            if not cond:
                return ai.func(*[do(i, a) for i in ai.args], evaluate=False)
            if isinstance(ai, cls):
                return ai.func(*[do(i, a) for i in ai.args if i != a], evaluate=False)
            return a

        for i, a in enumerate(args):
            args[i + 1 :] = [do(ai, a) for ai in args[i + 1 :]]

        # factor out common elements as for
        # Min(Max(x, y), Max(x, z)) -> Max(x, Min(y, z))
        # and vice versa when swapping Min/Max -- do this only for the
        # easy case where all functions contain something in common;
        # trying to find some optimal subset of args to modify takes
        # too long

        def factor_minmax(args):
            is_other = lambda arg: isinstance(arg, other)  # noqa: E731
            other_args, remaining_args = sift(args, is_other, binary=True)
            if not other_args:
                return args

            # Min(Max(x, y, z), Max(x, y, u, v)) -> {x,y}, ({z}, {u,v})
            arg_sets = [set(arg.args) for arg in other_args]
            common = set.intersection(*arg_sets)
            if not common:
                return args

            new_other_args = list(common)
            arg_sets_diff = [arg_set - common for arg_set in arg_sets]

            # If any set is empty after removing common then all can be
            # discarded e.g. Min(Max(a, b, c), Max(a, b)) -> Max(a, b)
            if all(arg_sets_diff):
                other_args_diff = [other(*s, evaluate=False) for s in arg_sets_diff]
                new_other_args.append(cls(*other_args_diff, evaluate=False))

            other_args_factored = other(*new_other_args, evaluate=False)
            return remaining_args + [other_args_factored]

        if len(args) > 1:
            args = factor_minmax(args)

        return args

    @classmethod
    def _new_args_filter(cls, arg_sequence):
        """
        Generator filtering args.

        first standard filter, for cls.zero and cls.identity.
        Also reshape ``Max(a, Max(b, c))`` to ``Max(a, b, c)``,
        and check arguments for comparability
        """
        for arg in arg_sequence:
            # pre-filter, checking comparability of arguments
            if (
                not isinstance(arg, Expr)
                or arg.is_extended_real is False
                or (arg.is_number and not arg.is_comparable)
            ):
                raise ValueError(f"The argument '{arg}' is not comparable.")

            if arg == cls.zero:  # type: ignore[attr-defined]
                raise ShortCircuit(arg)
            elif arg == cls.identity:  # type: ignore[attr-defined]
                continue
            elif arg.func == cls:
                yield from arg.args
            else:
                yield arg

    @classmethod
    def _find_localzeros(cls, values, **options):
        """
        Sequentially allocate values to localzeros.

        When a value is identified as being more extreme than another member it
        replaces that member; if this is never true, then the value is simply
        appended to the localzeros.

        Unlike the sympy implementation, we only look for zero and one, we don't
        do generic is connected test pairwise which is slow
        """

        # First, collapse all numeric arguments
        other_values = set()
        num_value = None
        for arg in values:
            if arg.is_Number:
                if num_value is None:
                    num_value = arg
                else:
                    if cls is Max:
                        num_value = max(num_value, arg)
                    elif cls is Min:
                        num_value = min(num_value, arg)
                    else:
                        raise AssertionError(f"impossible {cls}")
            else:
                other_values.add(arg)

        # Special cases when there is only one symbolic value
        if num_value is None:
            return other_values

        if len(other_values) == 0:
            return {num_value}

        if len(other_values) == 1:
            other_value = next(iter(other_values))
            if num_value in (0.0, 0) and other_value.is_nonnegative:
                return other_values if cls is Max else {num_value}
            if num_value == 1 and other_value.is_positive:
                return other_values if cls is Max else {num_value}

        other_values.add(num_value)
        return other_values

    _eval_is_algebraic = lambda s: _torf(i.is_algebraic for i in s.args)  # noqa: E731
    _eval_is_antihermitian = lambda s: _torf(  # noqa: E731
        i.is_antihermitian for i in s.args  # noqa: E731
    )  # noqa: E731
    _eval_is_commutative = lambda s: _torf(  # noqa: E731
        i.is_commutative for i in s.args  # noqa: E731
    )  # noqa: E731
    _eval_is_complex = lambda s: _torf(i.is_complex for i in s.args)  # noqa: E731
    _eval_is_composite = lambda s: _torf(i.is_composite for i in s.args)  # noqa: E731
    _eval_is_even = lambda s: _torf(i.is_even for i in s.args)  # noqa: E731
    _eval_is_finite = lambda s: _torf(i.is_finite for i in s.args)  # noqa: E731
    _eval_is_hermitian = lambda s: _torf(i.is_hermitian for i in s.args)  # noqa: E731
    _eval_is_imaginary = lambda s: _torf(i.is_imaginary for i in s.args)  # noqa: E731
    _eval_is_infinite = lambda s: _torf(i.is_infinite for i in s.args)  # noqa: E731
    _eval_is_integer = lambda s: _torf(i.is_integer for i in s.args)  # noqa: E731
    _eval_is_irrational = lambda s: _torf(i.is_irrational for i in s.args)  # noqa: E731
    _eval_is_negative = lambda s: _torf(i.is_negative for i in s.args)  # noqa: E731
    _eval_is_noninteger = lambda s: _torf(i.is_noninteger for i in s.args)  # noqa: E731
    _eval_is_nonnegative = lambda s: _torf(  # noqa: E731
        i.is_nonnegative for i in s.args  # noqa: E731
    )  # noqa: E731
    _eval_is_nonpositive = lambda s: _torf(  # noqa: E731
        i.is_nonpositive for i in s.args  # noqa: E731
    )  # noqa: E731
    _eval_is_nonzero = lambda s: _torf(i.is_nonzero for i in s.args)  # noqa: E731
    _eval_is_odd = lambda s: _torf(i.is_odd for i in s.args)  # noqa: E731
    _eval_is_polar = lambda s: _torf(i.is_polar for i in s.args)  # noqa: E731
    _eval_is_positive = lambda s: _torf(i.is_positive for i in s.args)  # noqa: E731
    _eval_is_prime = lambda s: _torf(i.is_prime for i in s.args)  # noqa: E731
    _eval_is_rational = lambda s: _torf(i.is_rational for i in s.args)  # noqa: E731
    _eval_is_real = lambda s: _torf(i.is_real for i in s.args)  # noqa: E731
    _eval_is_extended_real = lambda s: _torf(  # noqa: E731
        i.is_extended_real for i in s.args  # noqa: E731
    )  # noqa: E731
    _eval_is_transcendental = lambda s: _torf(  # noqa: E731
        i.is_transcendental for i in s.args  # noqa: E731
    )  # noqa: E731
    _eval_is_zero = lambda s: _torf(i.is_zero for i in s.args)  # noqa: E731


class Max(MinMaxBase, Application):  # type: ignore[misc]
    r"""
    Return, if possible, the maximum value of the list.
    """

    zero = S.Infinity
    identity = S.NegativeInfinity

    def _eval_is_positive(self):  # type:ignore[override]
        return fuzzy_or(a.is_positive for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self):  # type:ignore[override]
        return fuzzy_or(a.is_nonnegative for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_negative(self):  # type:ignore[override]
        return fuzzy_and(a.is_negative for a in self.args)


class Min(MinMaxBase, Application):  # type: ignore[misc]
    """
    Return, if possible, the minimum value of the list.
    """

    zero = S.NegativeInfinity
    identity = S.Infinity

    def _eval_is_positive(self):  # type:ignore[override]
        return fuzzy_and(a.is_positive for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self):  # type:ignore[override]
        return fuzzy_and(a.is_nonnegative for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_negative(self):  # type:ignore[override]
        return fuzzy_or(a.is_negative for a in self.args)


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

    precedence: int = 50  # precedence of mul

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
    is_real = True

    precedence: int = 60  # precedence of pow

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
    is_real = True

    precedence: int = 35  # lower precedence than add

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
    is_real = True

    precedence: int = 35  # lower precedence than add

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
    is_real = True

    @classmethod
    def eval(cls, number, ndigits):
        # assert number.is_integer is not True, number

        if isinstance(number, sympy.Number) and isinstance(ndigits, sympy.Integer):
            return sympy.Float(round(float(number), int(ndigits)))


class ToFloat(sympy.Function):
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

    precedence = 10

    def __repr__(self):  # type: ignore[override]
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
                if name == "log2":
                    return sympy.log(a, 2)
                return getattr(sympy, name)(a)
            return None

    nm = "OpaqueUnaryFn_" + name
    OpaqueUnaryFn.__name__ = nm
    OpaqueUnaryFn.__qualname__ = nm

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
OpaqueUnaryFn_log2 = make_opaque_unary_fn("log2")


def make_opaque_bitwise_fn(name, real_op_name):
    class BitwiseFn(sympy.Function):
        _torch_handler_name = name

        @classmethod
        def eval(cls, a, b):
            if a.is_Boolean and b.is_Boolean:
                return getattr(operator, real_op_name)(a, b)
            if a.is_Boolean:
                a = sympy.Integer(1 if a else 0)
            if b.is_Boolean:
                b = sympy.Integer(1 if b else 0)
            if isinstance(a, (sympy.Integer, int)) and isinstance(
                b, (sympy.Integer, int)
            ):
                return sympy.Integer(getattr(operator, real_op_name)(int(a), int(b)))
            return None

    BitwiseFn.__name__ = "BitwiseFn_" + name
    return BitwiseFn


BitwiseFn_bitwise_and = make_opaque_bitwise_fn("bitwise_and", "and_")
BitwiseFn_bitwise_or = make_opaque_bitwise_fn("bitwise_or", "or_")
