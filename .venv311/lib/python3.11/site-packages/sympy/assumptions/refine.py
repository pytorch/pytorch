from __future__ import annotations
from typing import Callable

from sympy.core import S, Add, Expr, Basic, Mul, Pow, Rational
from sympy.core.logic import fuzzy_not
from sympy.logic.boolalg import Boolean

from sympy.assumptions import ask, Q  # type: ignore


def refine(expr, assumptions=True):
    """
    Simplify an expression using assumptions.

    Explanation
    ===========

    Unlike :func:`~.simplify` which performs structural simplification
    without any assumption, this function transforms the expression into
    the form which is only valid under certain assumptions. Note that
    ``simplify()`` is generally not done in refining process.

    Refining boolean expression involves reducing it to ``S.true`` or
    ``S.false``. Unlike :func:`~.ask`, the expression will not be reduced
    if the truth value cannot be determined.

    Examples
    ========

    >>> from sympy import refine, sqrt, Q
    >>> from sympy.abc import x
    >>> refine(sqrt(x**2), Q.real(x))
    Abs(x)
    >>> refine(sqrt(x**2), Q.positive(x))
    x

    >>> refine(Q.real(x), Q.positive(x))
    True
    >>> refine(Q.positive(x), Q.real(x))
    Q.positive(x)

    See Also
    ========

    sympy.simplify.simplify.simplify : Structural simplification without assumptions.
    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.
    """
    if not isinstance(expr, Basic):
        return expr

    if not expr.is_Atom:
        args = [refine(arg, assumptions) for arg in expr.args]
        # TODO: this will probably not work with Integral or Polynomial
        expr = expr.func(*args)
    if hasattr(expr, '_eval_refine'):
        ref_expr = expr._eval_refine(assumptions)
        if ref_expr is not None:
            return ref_expr
    name = expr.__class__.__name__
    handler = handlers_dict.get(name, None)
    if handler is None:
        return expr
    new_expr = handler(expr, assumptions)
    if (new_expr is None) or (expr == new_expr):
        return expr
    if not isinstance(new_expr, Expr):
        return new_expr
    return refine(new_expr, assumptions)


def refine_abs(expr, assumptions):
    """
    Handler for the absolute value.

    Examples
    ========

    >>> from sympy import Q, Abs
    >>> from sympy.assumptions.refine import refine_abs
    >>> from sympy.abc import x
    >>> refine_abs(Abs(x), Q.real(x))
    >>> refine_abs(Abs(x), Q.positive(x))
    x
    >>> refine_abs(Abs(x), Q.negative(x))
    -x

    """
    from sympy.functions.elementary.complexes import Abs
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions) and \
            fuzzy_not(ask(Q.negative(arg), assumptions)):
        # if it's nonnegative
        return arg
    if ask(Q.negative(arg), assumptions):
        return -arg
    # arg is Mul
    if isinstance(arg, Mul):
        r = [refine(abs(a), assumptions) for a in arg.args]
        non_abs = []
        in_abs = []
        for i in r:
            if isinstance(i, Abs):
                in_abs.append(i.args[0])
            else:
                non_abs.append(i)
        return Mul(*non_abs) * Abs(Mul(*in_abs))


def refine_Pow(expr, assumptions):
    """
    Handler for instances of Pow.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.refine import refine_Pow
    >>> from sympy.abc import x,y,z
    >>> refine_Pow((-1)**x, Q.real(x))
    >>> refine_Pow((-1)**x, Q.even(x))
    1
    >>> refine_Pow((-1)**x, Q.odd(x))
    -1

    For powers of -1, even parts of the exponent can be simplified:

    >>> refine_Pow((-1)**(x+y), Q.even(x))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+z), Q.odd(x) & Q.odd(z))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+2), Q.odd(x))
    (-1)**(y + 1)
    >>> refine_Pow((-1)**(x+3), True)
    (-1)**(x + 1)

    """
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions import sign
    if isinstance(expr.base, Abs):
        if ask(Q.real(expr.base.args[0]), assumptions) and \
                ask(Q.even(expr.exp), assumptions):
            return expr.base.args[0] ** expr.exp
    if ask(Q.real(expr.base), assumptions):
        if expr.base.is_number:
            if ask(Q.even(expr.exp), assumptions):
                return abs(expr.base) ** expr.exp
            if ask(Q.odd(expr.exp), assumptions):
                return sign(expr.base) * abs(expr.base) ** expr.exp
        if isinstance(expr.exp, Rational):
            if isinstance(expr.base, Pow):
                return abs(expr.base.base) ** (expr.base.exp * expr.exp)

        if expr.base is S.NegativeOne:
            if expr.exp.is_Add:

                old = expr

                # For powers of (-1) we can remove
                #  - even terms
                #  - pairs of odd terms
                #  - a single odd term + 1
                #  - A numerical constant N can be replaced with mod(N,2)

                coeff, terms = expr.exp.as_coeff_add()
                terms = set(terms)
                even_terms = set()
                odd_terms = set()
                initial_number_of_terms = len(terms)

                for t in terms:
                    if ask(Q.even(t), assumptions):
                        even_terms.add(t)
                    elif ask(Q.odd(t), assumptions):
                        odd_terms.add(t)

                terms -= even_terms
                if len(odd_terms) % 2:
                    terms -= odd_terms
                    new_coeff = (coeff + S.One) % 2
                else:
                    terms -= odd_terms
                    new_coeff = coeff % 2

                if new_coeff != coeff or len(terms) < initial_number_of_terms:
                    terms.add(new_coeff)
                    expr = expr.base**(Add(*terms))

                # Handle (-1)**((-1)**n/2 + m/2)
                e2 = 2*expr.exp
                if ask(Q.even(e2), assumptions):
                    if e2.could_extract_minus_sign():
                        e2 *= expr.base
                if e2.is_Add:
                    i, p = e2.as_two_terms()
                    if p.is_Pow and p.base is S.NegativeOne:
                        if ask(Q.integer(p.exp), assumptions):
                            i = (i + 1)/2
                            if ask(Q.even(i), assumptions):
                                return expr.base**p.exp
                            elif ask(Q.odd(i), assumptions):
                                return expr.base**(p.exp + 1)
                            else:
                                return expr.base**(p.exp + i)

                if old != expr:
                    return expr


def refine_atan2(expr, assumptions):
    """
    Handler for the atan2 function.

    Examples
    ========

    >>> from sympy import Q, atan2
    >>> from sympy.assumptions.refine import refine_atan2
    >>> from sympy.abc import x, y
    >>> refine_atan2(atan2(y,x), Q.real(y) & Q.positive(x))
    atan(y/x)
    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.negative(x))
    atan(y/x) - pi
    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.negative(x))
    atan(y/x) + pi
    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.negative(x))
    pi
    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.zero(x))
    pi/2
    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.zero(x))
    -pi/2
    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.zero(x))
    nan
    """
    from sympy.functions.elementary.trigonometric import atan
    y, x = expr.args
    if ask(Q.real(y) & Q.positive(x), assumptions):
        return atan(y / x)
    elif ask(Q.negative(y) & Q.negative(x), assumptions):
        return atan(y / x) - S.Pi
    elif ask(Q.positive(y) & Q.negative(x), assumptions):
        return atan(y / x) + S.Pi
    elif ask(Q.zero(y) & Q.negative(x), assumptions):
        return S.Pi
    elif ask(Q.positive(y) & Q.zero(x), assumptions):
        return S.Pi/2
    elif ask(Q.negative(y) & Q.zero(x), assumptions):
        return -S.Pi/2
    elif ask(Q.zero(y) & Q.zero(x), assumptions):
        return S.NaN
    else:
        return expr


def refine_re(expr, assumptions):
    """
    Handler for real part.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_re
    >>> from sympy import Q, re
    >>> from sympy.abc import x
    >>> refine_re(re(x), Q.real(x))
    x
    >>> refine_re(re(x), Q.imaginary(x))
    0
    """
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return arg
    if ask(Q.imaginary(arg), assumptions):
        return S.Zero
    return _refine_reim(expr, assumptions)


def refine_im(expr, assumptions):
    """
    Handler for imaginary part.

    Explanation
    ===========

    >>> from sympy.assumptions.refine import refine_im
    >>> from sympy import Q, im
    >>> from sympy.abc import x
    >>> refine_im(im(x), Q.real(x))
    0
    >>> refine_im(im(x), Q.imaginary(x))
    -I*x
    """
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return S.Zero
    if ask(Q.imaginary(arg), assumptions):
        return - S.ImaginaryUnit * arg
    return _refine_reim(expr, assumptions)

def refine_arg(expr, assumptions):
    """
    Handler for complex argument

    Explanation
    ===========

    >>> from sympy.assumptions.refine import refine_arg
    >>> from sympy import Q, arg
    >>> from sympy.abc import x
    >>> refine_arg(arg(x), Q.positive(x))
    0
    >>> refine_arg(arg(x), Q.negative(x))
    pi
    """
    rg = expr.args[0]
    if ask(Q.positive(rg), assumptions):
        return S.Zero
    if ask(Q.negative(rg), assumptions):
        return S.Pi
    return None


def _refine_reim(expr, assumptions):
    # Helper function for refine_re & refine_im
    expanded = expr.expand(complex = True)
    if expanded != expr:
        refined = refine(expanded, assumptions)
        if refined != expanded:
            return refined
    # Best to leave the expression as is
    return None


def refine_sign(expr, assumptions):
    """
    Handler for sign.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_sign
    >>> from sympy import Symbol, Q, sign, im
    >>> x = Symbol('x', real = True)
    >>> expr = sign(x)
    >>> refine_sign(expr, Q.positive(x) & Q.nonzero(x))
    1
    >>> refine_sign(expr, Q.negative(x) & Q.nonzero(x))
    -1
    >>> refine_sign(expr, Q.zero(x))
    0
    >>> y = Symbol('y', imaginary = True)
    >>> expr = sign(y)
    >>> refine_sign(expr, Q.positive(im(y)))
    I
    >>> refine_sign(expr, Q.negative(im(y)))
    -I
    """
    arg = expr.args[0]
    if ask(Q.zero(arg), assumptions):
        return S.Zero
    if ask(Q.real(arg)):
        if ask(Q.positive(arg), assumptions):
            return S.One
        if ask(Q.negative(arg), assumptions):
            return S.NegativeOne
    if ask(Q.imaginary(arg)):
        arg_re, arg_im = arg.as_real_imag()
        if ask(Q.positive(arg_im), assumptions):
            return S.ImaginaryUnit
        if ask(Q.negative(arg_im), assumptions):
            return -S.ImaginaryUnit
    return expr


def refine_matrixelement(expr, assumptions):
    """
    Handler for symmetric part.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_matrixelement
    >>> from sympy import MatrixSymbol, Q
    >>> X = MatrixSymbol('X', 3, 3)
    >>> refine_matrixelement(X[0, 1], Q.symmetric(X))
    X[0, 1]
    >>> refine_matrixelement(X[1, 0], Q.symmetric(X))
    X[0, 1]
    """
    from sympy.matrices.expressions.matexpr import MatrixElement
    matrix, i, j = expr.args
    if ask(Q.symmetric(matrix), assumptions):
        if (i - j).could_extract_minus_sign():
            return expr
        return MatrixElement(matrix, j, i)

handlers_dict: dict[str, Callable[[Expr, Boolean], Expr]] = {
    'Abs': refine_abs,
    'Pow': refine_Pow,
    'atan2': refine_atan2,
    're': refine_re,
    'im': refine_im,
    'arg': refine_arg,
    'sign': refine_sign,
    'MatrixElement': refine_matrixelement
}
