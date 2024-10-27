"""
Handlers for predicates related to set membership: integer, rational, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Mul, Pow, S
from sympy.core.numbers import (AlgebraicNumber, ComplexInfinity, Exp1, Float,
    GoldenRatio, ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity,
    Number, NumberSymbol, Pi, pi, Rational, TribonacciConstant, E)
from sympy.core.logic import fuzzy_bool
from sympy.functions import (Abs, acos, acot, asin, atan, cos, cot, exp, im,
    log, re, sin, tan)
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.functions.elementary.complexes import conjugate
from sympy.matrices import Determinant, MatrixBase, Trace
from sympy.matrices.expressions.matexpr import MatrixElement

from sympy.multipledispatch import MDNotImplementedError

from .common import test_closed_group
from ..predicates.sets import (IntegerPredicate, RationalPredicate,
    IrrationalPredicate, RealPredicate, ExtendedRealPredicate,
    HermitianPredicate, ComplexPredicate, ImaginaryPredicate,
    AntihermitianPredicate, AlgebraicPredicate)


# IntegerPredicate

def _IntegerPredicate_number(expr, assumptions):
    # helper function
        try:
            i = int(expr.round())
            if not (expr - i).equals(0):
                raise TypeError
            return True
        except TypeError:
            return False

@IntegerPredicate.register_many(int, Integer) # type:ignore
def _(expr, assumptions):
    return True

@IntegerPredicate.register_many(Exp1, GoldenRatio, ImaginaryUnit, Infinity,
        NegativeInfinity, Pi, Rational, TribonacciConstant)
def _(expr, assumptions):
    return False

@IntegerPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_integer
    if ret is None:
        raise MDNotImplementedError
    return ret

@IntegerPredicate.register_many(Add, Pow)
def _(expr, assumptions):
    """
    * Integer + Integer       -> Integer
    * Integer + !Integer      -> !Integer
    * !Integer + !Integer -> ?
    """
    if expr.is_number:
        return _IntegerPredicate_number(expr, assumptions)
    return test_closed_group(expr, assumptions, Q.integer)

@IntegerPredicate.register(Mul)
def _(expr, assumptions):
    """
    * Integer*Integer      -> Integer
    * Integer*Irrational   -> !Integer
    * Odd/Even             -> !Integer
    * Integer*Rational     -> ?
    """
    if expr.is_number:
        return _IntegerPredicate_number(expr, assumptions)
    _output = True
    for arg in expr.args:
        if not ask(Q.integer(arg), assumptions):
            if arg.is_Rational:
                if arg.q == 2:
                    return ask(Q.even(2*expr), assumptions)
                if ~(arg.q & 1):
                    return None
            elif ask(Q.irrational(arg), assumptions):
                if _output:
                    _output = False
                else:
                    return
            else:
                return

    return _output

@IntegerPredicate.register(Abs)
def _(expr, assumptions):
    return ask(Q.integer(expr.args[0]), assumptions)

@IntegerPredicate.register_many(Determinant, MatrixElement, Trace)
def _(expr, assumptions):
    return ask(Q.integer_elements(expr.args[0]), assumptions)


# RationalPredicate

@RationalPredicate.register(Rational)
def _(expr, assumptions):
    return True

@RationalPredicate.register(Float)
def _(expr, assumptions):
    return None

@RationalPredicate.register_many(Exp1, GoldenRatio, ImaginaryUnit, Infinity,
    NegativeInfinity, Pi, TribonacciConstant)
def _(expr, assumptions):
    return False

@RationalPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_rational
    if ret is None:
        raise MDNotImplementedError
    return ret

@RationalPredicate.register_many(Add, Mul)
def _(expr, assumptions):
    """
    * Rational + Rational     -> Rational
    * Rational + !Rational    -> !Rational
    * !Rational + !Rational   -> ?
    """
    if expr.is_number:
        if expr.as_real_imag()[1]:
            return False
    return test_closed_group(expr, assumptions, Q.rational)

@RationalPredicate.register(Pow)
def _(expr, assumptions):
    """
    * Rational ** Integer      -> Rational
    * Irrational ** Rational   -> Irrational
    * Rational ** Irrational   -> ?
    """
    if expr.base == E:
        x = expr.exp
        if ask(Q.rational(x), assumptions):
            return ask(~Q.nonzero(x), assumptions)
        return

    if ask(Q.integer(expr.exp), assumptions):
        return ask(Q.rational(expr.base), assumptions)
    elif ask(Q.rational(expr.exp), assumptions):
        if ask(Q.prime(expr.base), assumptions):
            return False

@RationalPredicate.register_many(asin, atan, cos, sin, tan)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@RationalPredicate.register(exp)
def _(expr, assumptions):
    x = expr.exp
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@RationalPredicate.register_many(acot, cot)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return False

@RationalPredicate.register_many(acos, log)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x - 1), assumptions)


# IrrationalPredicate

@IrrationalPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_irrational
    if ret is None:
        raise MDNotImplementedError
    return ret

@IrrationalPredicate.register(Basic)
def _(expr, assumptions):
    _real = ask(Q.real(expr), assumptions)
    if _real:
        _rational = ask(Q.rational(expr), assumptions)
        if _rational is None:
            return None
        return not _rational
    else:
        return _real


# RealPredicate

def _RealPredicate_number(expr, assumptions):
    # let as_real_imag() work first since the expression may
    # be simpler to evaluate
    i = expr.as_real_imag()[1].evalf(2)
    if i._prec != 1:
        return not i
    # allow None to be returned if we couldn't show for sure
    # that i was 0

@RealPredicate.register_many(Abs, Exp1, Float, GoldenRatio, im, Pi, Rational,
    re, TribonacciConstant)
def _(expr, assumptions):
    return True

@RealPredicate.register_many(ImaginaryUnit, Infinity, NegativeInfinity)
def _(expr, assumptions):
    return False

@RealPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_real
    if ret is None:
        raise MDNotImplementedError
    return ret

@RealPredicate.register(Add)
def _(expr, assumptions):
    """
    * Real + Real              -> Real
    * Real + (Complex & !Real) -> !Real
    """
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)
    return test_closed_group(expr, assumptions, Q.real)

@RealPredicate.register(Mul)
def _(expr, assumptions):
    """
    * Real*Real               -> Real
    * Real*Imaginary          -> !Real
    * Imaginary*Imaginary     -> Real
    """
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)
    result = True
    for arg in expr.args:
        if ask(Q.real(arg), assumptions):
            pass
        elif ask(Q.imaginary(arg), assumptions):
            result = result ^ True
        else:
            break
    else:
        return result

@RealPredicate.register(Pow)
def _(expr, assumptions):
    """
    * Real**Integer              -> Real
    * Positive**Real             -> Real
    * Real**(Integer/Even)       -> Real if base is nonnegative
    * Real**(Integer/Odd)        -> Real
    * Imaginary**(Integer/Even)  -> Real
    * Imaginary**(Integer/Odd)   -> not Real
    * Imaginary**Real            -> ? since Real could be 0 (giving real)
                                    or 1 (giving imaginary)
    * b**Imaginary               -> Real if log(b) is imaginary and b != 0
                                    and exponent != integer multiple of
                                    I*pi/log(b)
    * Real**Real                 -> ? e.g. sqrt(-1) is imaginary and
                                    sqrt(2) is not
    """
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)

    if expr.base == E:
        return ask(
            Q.integer(expr.exp/I/pi) | Q.real(expr.exp), assumptions
        )

    if expr.base.func == exp or (expr.base.is_Pow and expr.base.base == E):
        if ask(Q.imaginary(expr.base.exp), assumptions):
            if ask(Q.imaginary(expr.exp), assumptions):
                return True
        # If the i = (exp's arg)/(I*pi) is an integer or half-integer
        # multiple of I*pi then 2*i will be an integer. In addition,
        # exp(i*I*pi) = (-1)**i so the overall realness of the expr
        # can be determined by replacing exp(i*I*pi) with (-1)**i.
        i = expr.base.exp/I/pi
        if ask(Q.integer(2*i), assumptions):
            return ask(Q.real((S.NegativeOne**i)**expr.exp), assumptions)
        return

    if ask(Q.imaginary(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            odd = ask(Q.odd(expr.exp), assumptions)
            if odd is not None:
                return not odd
            return

    if ask(Q.imaginary(expr.exp), assumptions):
        imlog = ask(Q.imaginary(log(expr.base)), assumptions)
        if imlog is not None:
            # I**i -> real, log(I) is imag;
            # (2*I)**i -> complex, log(2*I) is not imag
            return imlog

    if ask(Q.real(expr.base), assumptions):
        if ask(Q.real(expr.exp), assumptions):
            if expr.exp.is_Rational and \
                    ask(Q.even(expr.exp.q), assumptions):
                return ask(Q.positive(expr.base), assumptions)
            elif ask(Q.integer(expr.exp), assumptions):
                return True
            elif ask(Q.positive(expr.base), assumptions):
                return True
            elif ask(Q.negative(expr.base), assumptions):
                return False

@RealPredicate.register_many(cos, sin)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
            return True

@RealPredicate.register(exp)
def _(expr, assumptions):
    return ask(
        Q.integer(expr.exp/I/pi) | Q.real(expr.exp), assumptions
    )

@RealPredicate.register(log)
def _(expr, assumptions):
    return ask(Q.positive(expr.args[0]), assumptions)

@RealPredicate.register_many(Determinant, MatrixElement, Trace)
def _(expr, assumptions):
    return ask(Q.real_elements(expr.args[0]), assumptions)


# ExtendedRealPredicate

@ExtendedRealPredicate.register(object)
def _(expr, assumptions):
    return ask(Q.negative_infinite(expr)
               | Q.negative(expr)
               | Q.zero(expr)
               | Q.positive(expr)
               | Q.positive_infinite(expr),
            assumptions)

@ExtendedRealPredicate.register_many(Infinity, NegativeInfinity)
def _(expr, assumptions):
    return True

@ExtendedRealPredicate.register_many(Add, Mul, Pow) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.extended_real)


# HermitianPredicate

@HermitianPredicate.register(object) # type:ignore
def _(expr, assumptions):
    if isinstance(expr, MatrixBase):
        return None
    return ask(Q.real(expr), assumptions)

@HermitianPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian + Hermitian  -> Hermitian
    * Hermitian + !Hermitian -> !Hermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    return test_closed_group(expr, assumptions, Q.hermitian)

@HermitianPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    As long as there is at most only one noncommutative term:

    * Hermitian*Hermitian         -> Hermitian
    * Hermitian*Antihermitian     -> !Hermitian
    * Antihermitian*Antihermitian -> Hermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    nccount = 0
    result = True
    for arg in expr.args:
        if ask(Q.antihermitian(arg), assumptions):
            result = result ^ True
        elif not ask(Q.hermitian(arg), assumptions):
            break
        if ask(~Q.commutative(arg), assumptions):
            nccount += 1
            if nccount > 1:
                break
    else:
        return result

@HermitianPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian**Integer -> Hermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    if expr.base == E:
        if ask(Q.hermitian(expr.exp), assumptions):
            return True
        raise MDNotImplementedError
    if ask(Q.hermitian(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            return True
    raise MDNotImplementedError

@HermitianPredicate.register_many(cos, sin) # type:ignore
def _(expr, assumptions):
    if ask(Q.hermitian(expr.args[0]), assumptions):
        return True
    raise MDNotImplementedError

@HermitianPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    if ask(Q.hermitian(expr.exp), assumptions):
        return True
    raise MDNotImplementedError

@HermitianPredicate.register(MatrixBase) # type:ignore
def _(mat, assumptions):
    rows, cols = mat.shape
    ret_val = True
    for i in range(rows):
        for j in range(i, cols):
            cond = fuzzy_bool(Eq(mat[i, j], conjugate(mat[j, i])))
            if cond is None:
                ret_val = None
            if cond == False:
                return False
    if ret_val is None:
        raise MDNotImplementedError
    return ret_val


# ComplexPredicate

@ComplexPredicate.register_many(Abs, cos, exp, im, ImaginaryUnit, log, Number, # type:ignore
    NumberSymbol, re, sin)
def _(expr, assumptions):
    return True

@ComplexPredicate.register_many(Infinity, NegativeInfinity) # type:ignore
def _(expr, assumptions):
    return False

@ComplexPredicate.register(Expr) # type:ignore
def _(expr, assumptions):
    ret = expr.is_complex
    if ret is None:
        raise MDNotImplementedError
    return ret

@ComplexPredicate.register_many(Add, Mul) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.complex)

@ComplexPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    if expr.base == E:
        return True
    return test_closed_group(expr, assumptions, Q.complex)

@ComplexPredicate.register_many(Determinant, MatrixElement, Trace) # type:ignore
def _(expr, assumptions):
    return ask(Q.complex_elements(expr.args[0]), assumptions)

@ComplexPredicate.register(NaN) # type:ignore
def _(expr, assumptions):
    return None


# ImaginaryPredicate

def _Imaginary_number(expr, assumptions):
    # let as_real_imag() work first since the expression may
    # be simpler to evaluate
    r = expr.as_real_imag()[0].evalf(2)
    if r._prec != 1:
        return not r
    # allow None to be returned if we couldn't show for sure
    # that r was 0

@ImaginaryPredicate.register(ImaginaryUnit) # type:ignore
def _(expr, assumptions):
    return True

@ImaginaryPredicate.register(Expr) # type:ignore
def _(expr, assumptions):
    ret = expr.is_imaginary
    if ret is None:
        raise MDNotImplementedError
    return ret

@ImaginaryPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Imaginary + Imaginary -> Imaginary
    * Imaginary + Complex   -> ?
    * Imaginary + Real      -> !Imaginary
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)

    reals = 0
    for arg in expr.args:
        if ask(Q.imaginary(arg), assumptions):
            pass
        elif ask(Q.real(arg), assumptions):
            reals += 1
        else:
            break
    else:
        if reals == 0:
            return True
        if reals in (1, len(expr.args)):
            # two reals could sum 0 thus giving an imaginary
            return False

@ImaginaryPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    * Real*Imaginary      -> Imaginary
    * Imaginary*Imaginary -> Real
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)
    result = False
    reals = 0
    for arg in expr.args:
        if ask(Q.imaginary(arg), assumptions):
            result = result ^ True
        elif not ask(Q.real(arg), assumptions):
            break
    else:
        if reals == len(expr.args):
            return False
        return result

@ImaginaryPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Imaginary**Odd        -> Imaginary
    * Imaginary**Even       -> Real
    * b**Imaginary          -> !Imaginary if exponent is an integer
                               multiple of I*pi/log(b)
    * Imaginary**Real       -> ?
    * Positive**Real        -> Real
    * Negative**Integer     -> Real
    * Negative**(Integer/2) -> Imaginary
    * Negative**Real        -> not Imaginary if exponent is not Rational
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)

    if expr.base == E:
        a = expr.exp/I/pi
        return ask(Q.integer(2*a) & ~Q.integer(a), assumptions)

    if expr.base.func == exp or (expr.base.is_Pow and expr.base.base == E):
        if ask(Q.imaginary(expr.base.exp), assumptions):
            if ask(Q.imaginary(expr.exp), assumptions):
                return False
            i = expr.base.exp/I/pi
            if ask(Q.integer(2*i), assumptions):
                return ask(Q.imaginary((S.NegativeOne**i)**expr.exp), assumptions)

    if ask(Q.imaginary(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            odd = ask(Q.odd(expr.exp), assumptions)
            if odd is not None:
                return odd
            return

    if ask(Q.imaginary(expr.exp), assumptions):
        imlog = ask(Q.imaginary(log(expr.base)), assumptions)
        if imlog is not None:
            # I**i -> real; (2*I)**i -> complex ==> not imaginary
            return False

    if ask(Q.real(expr.base) & Q.real(expr.exp), assumptions):
        if ask(Q.positive(expr.base), assumptions):
            return False
        else:
            rat = ask(Q.rational(expr.exp), assumptions)
            if not rat:
                return rat
            if ask(Q.integer(expr.exp), assumptions):
                return False
            else:
                half = ask(Q.integer(2*expr.exp), assumptions)
                if half:
                    return ask(Q.negative(expr.base), assumptions)
                return half

@ImaginaryPredicate.register(log) # type:ignore
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        if ask(Q.positive(expr.args[0]), assumptions):
            return False
        return
    # XXX it should be enough to do
    # return ask(Q.nonpositive(expr.args[0]), assumptions)
    # but ask(Q.nonpositive(exp(x)), Q.imaginary(x)) -> None;
    # it should return True since exp(x) will be either 0 or complex
    if expr.args[0].func == exp or (expr.args[0].is_Pow and expr.args[0].base == E):
        if expr.args[0].exp in [I, -I]:
            return True
    im = ask(Q.imaginary(expr.args[0]), assumptions)
    if im is False:
        return False

@ImaginaryPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    a = expr.exp/I/pi
    return ask(Q.integer(2*a) & ~Q.integer(a), assumptions)

@ImaginaryPredicate.register_many(Number, NumberSymbol) # type:ignore
def _(expr, assumptions):
    return not (expr.as_real_imag()[1] == 0)

@ImaginaryPredicate.register(NaN) # type:ignore
def _(expr, assumptions):
    return None


# AntihermitianPredicate

@AntihermitianPredicate.register(object) # type:ignore
def _(expr, assumptions):
    if isinstance(expr, MatrixBase):
        return None
    if ask(Q.zero(expr), assumptions):
        return True
    return ask(Q.imaginary(expr), assumptions)

@AntihermitianPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Antihermitian + Antihermitian  -> Antihermitian
    * Antihermitian + !Antihermitian -> !Antihermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    return test_closed_group(expr, assumptions, Q.antihermitian)

@AntihermitianPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    As long as there is at most only one noncommutative term:

    * Hermitian*Hermitian         -> !Antihermitian
    * Hermitian*Antihermitian     -> Antihermitian
    * Antihermitian*Antihermitian -> !Antihermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    nccount = 0
    result = False
    for arg in expr.args:
        if ask(Q.antihermitian(arg), assumptions):
            result = result ^ True
        elif not ask(Q.hermitian(arg), assumptions):
            break
        if ask(~Q.commutative(arg), assumptions):
            nccount += 1
            if nccount > 1:
                break
    else:
        return result

@AntihermitianPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian**Integer  -> !Antihermitian
    * Antihermitian**Even -> !Antihermitian
    * Antihermitian**Odd  -> Antihermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    if ask(Q.hermitian(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            return False
    elif ask(Q.antihermitian(expr.base), assumptions):
        if ask(Q.even(expr.exp), assumptions):
            return False
        elif ask(Q.odd(expr.exp), assumptions):
            return True
    raise MDNotImplementedError

@AntihermitianPredicate.register(MatrixBase) # type:ignore
def _(mat, assumptions):
    rows, cols = mat.shape
    ret_val = True
    for i in range(rows):
        for j in range(i, cols):
            cond = fuzzy_bool(Eq(mat[i, j], -conjugate(mat[j, i])))
            if cond is None:
                ret_val = None
            if cond == False:
                return False
    if ret_val is None:
        raise MDNotImplementedError
    return ret_val


# AlgebraicPredicate

@AlgebraicPredicate.register_many(AlgebraicNumber, Float, GoldenRatio, # type:ignore
    ImaginaryUnit, TribonacciConstant)
def _(expr, assumptions):
    return True

@AlgebraicPredicate.register_many(ComplexInfinity, Exp1, Infinity, # type:ignore
    NegativeInfinity, Pi)
def _(expr, assumptions):
    return False

@AlgebraicPredicate.register_many(Add, Mul) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.algebraic)

@AlgebraicPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    if expr.base == E:
        if ask(Q.algebraic(expr.exp), assumptions):
            return ask(~Q.nonzero(expr.exp), assumptions)
        return
    return expr.exp.is_Rational and ask(Q.algebraic(expr.base), assumptions)

@AlgebraicPredicate.register(Rational) # type:ignore
def _(expr, assumptions):
    return expr.q != 0

@AlgebraicPredicate.register_many(asin, atan, cos, sin, tan) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@AlgebraicPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    x = expr.exp
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@AlgebraicPredicate.register_many(acot, cot) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return False

@AlgebraicPredicate.register_many(acos, log) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x - 1), assumptions)
