from typing import Tuple as tTuple

from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
    AppliedUndef, expand_mul)
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise

###############################################################################
######################### REAL and IMAGINARY PARTS ############################
###############################################################################


class re(Function):
    """
    Returns real part of expression. This function performs only
    elementary analysis and so it will fail to decompose properly
    more complicated expressions. If completely simplified result
    is needed then use ``Basic.as_real_imag()`` or perform complex
    expansion on instance of this function.

    Examples
    ========

    >>> from sympy import re, im, I, E, symbols
    >>> x, y = symbols('x y', real=True)
    >>> re(2*E)
    2*E
    >>> re(2*I + 17)
    17
    >>> re(2*I)
    0
    >>> re(im(x) + x*I + 2)
    2
    >>> re(5 + I + 2)
    7

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Real part of expression.

    See Also
    ========

    im
    """

    args: tTuple[Expr]

    is_extended_real = True
    unbranched = True  # implicitly works on the projection to C
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        if arg is S.NaN:
            return S.NaN
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif arg.is_extended_real:
            return arg
        elif arg.is_imaginary or (I*arg).is_extended_real:
            return S.Zero
        elif arg.is_Matrix:
            return arg.as_real_imag()[0]
        elif arg.is_Function and isinstance(arg, conjugate):
            return re(arg.args[0])
        else:

            included, reverted, excluded = [], [], []
            args = Add.make_args(arg)
            for term in args:
                coeff = term.as_coefficient(I)

                if coeff is not None:
                    if not coeff.is_extended_real:
                        reverted.append(coeff)
                elif not term.has(I) and term.is_extended_real:
                    excluded.append(term)
                else:
                    # Try to do some advanced expansion.  If
                    # impossible, don't try to do re(arg) again
                    # (because this is what we are trying to do now).
                    real_imag = term.as_real_imag(ignore=arg)
                    if real_imag:
                        excluded.append(real_imag[0])
                    else:
                        included.append(term)

            if len(args) != len(included):
                a, b, c = (Add(*xs) for xs in [included, reverted, excluded])

                return cls(a) - im(b) + c

    def as_real_imag(self, deep=True, **hints):
        """
        Returns the real number with a zero imaginary part.

        """
        return (self, S.Zero)

    def _eval_derivative(self, x):
        if x.is_extended_real or self.args[0].is_extended_real:
            return re(Derivative(self.args[0], x, evaluate=True))
        if x.is_imaginary or self.args[0].is_imaginary:
            return -I \
                * im(Derivative(self.args[0], x, evaluate=True))

    def _eval_rewrite_as_im(self, arg, **kwargs):
        return self.args[0] - I*im(self.args[0])

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_is_zero(self):
        # is_imaginary implies nonzero
        return fuzzy_or([self.args[0].is_imaginary, self.args[0].is_zero])

    def _eval_is_finite(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_finite:
            return True


class im(Function):
    """
    Returns imaginary part of expression. This function performs only
    elementary analysis and so it will fail to decompose properly more
    complicated expressions. If completely simplified result is needed then
    use ``Basic.as_real_imag()`` or perform complex expansion on instance of
    this function.

    Examples
    ========

    >>> from sympy import re, im, E, I
    >>> from sympy.abc import x, y
    >>> im(2*E)
    0
    >>> im(2*I + 17)
    2
    >>> im(x*I)
    re(x)
    >>> im(re(x) + y)
    im(y)
    >>> im(2 + 3*I)
    3

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Imaginary part of expression.

    See Also
    ========

    re
    """

    args: tTuple[Expr]

    is_extended_real = True
    unbranched = True  # implicitly works on the projection to C
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        if arg is S.NaN:
            return S.NaN
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif arg.is_extended_real:
            return S.Zero
        elif arg.is_imaginary or (I*arg).is_extended_real:
            return -I * arg
        elif arg.is_Matrix:
            return arg.as_real_imag()[1]
        elif arg.is_Function and isinstance(arg, conjugate):
            return -im(arg.args[0])
        else:
            included, reverted, excluded = [], [], []
            args = Add.make_args(arg)
            for term in args:
                coeff = term.as_coefficient(I)

                if coeff is not None:
                    if not coeff.is_extended_real:
                        reverted.append(coeff)
                    else:
                        excluded.append(coeff)
                elif term.has(I) or not term.is_extended_real:
                    # Try to do some advanced expansion.  If
                    # impossible, don't try to do im(arg) again
                    # (because this is what we are trying to do now).
                    real_imag = term.as_real_imag(ignore=arg)
                    if real_imag:
                        excluded.append(real_imag[1])
                    else:
                        included.append(term)

            if len(args) != len(included):
                a, b, c = (Add(*xs) for xs in [included, reverted, excluded])

                return cls(a) + re(b) + c

    def as_real_imag(self, deep=True, **hints):
        """
        Return the imaginary part with a zero real part.

        """
        return (self, S.Zero)

    def _eval_derivative(self, x):
        if x.is_extended_real or self.args[0].is_extended_real:
            return im(Derivative(self.args[0], x, evaluate=True))
        if x.is_imaginary or self.args[0].is_imaginary:
            return -I \
                * re(Derivative(self.args[0], x, evaluate=True))

    def _eval_rewrite_as_re(self, arg, **kwargs):
        return -I*(self.args[0] - re(self.args[0]))

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_is_zero(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_finite:
            return True

###############################################################################
############### SIGN, ABSOLUTE VALUE, ARGUMENT and CONJUGATION ################
###############################################################################

class sign(Function):
    """
    Returns the complex sign of an expression:

    Explanation
    ===========

    If the expression is real the sign will be:

        * $1$ if expression is positive
        * $0$ if expression is equal to zero
        * $-1$ if expression is negative

    If the expression is imaginary the sign will be:

        * $I$ if im(expression) is positive
        * $-I$ if im(expression) is negative

    Otherwise an unevaluated expression will be returned. When evaluated, the
    result (in general) will be ``cos(arg(expr)) + I*sin(arg(expr))``.

    Examples
    ========

    >>> from sympy import sign, I

    >>> sign(-1)
    -1
    >>> sign(0)
    0
    >>> sign(-3*I)
    -I
    >>> sign(1 + I)
    sign(1 + I)
    >>> _.evalf()
    0.707106781186548 + 0.707106781186548*I

    Parameters
    ==========

    arg : Expr
        Real or imaginary expression.

    Returns
    =======

    expr : Expr
        Complex sign of expression.

    See Also
    ========

    Abs, conjugate
    """

    is_complex = True
    _singularities = True

    def doit(self, **hints):
        s = super().doit()
        if s == self and self.args[0].is_zero is False:
            return self.args[0] / Abs(self.args[0])
        return s

    @classmethod
    def eval(cls, arg):
        # handle what we can
        if arg.is_Mul:
            c, args = arg.as_coeff_mul()
            unk = []
            s = sign(c)
            for a in args:
                if a.is_extended_negative:
                    s = -s
                elif a.is_extended_positive:
                    pass
                else:
                    if a.is_imaginary:
                        ai = im(a)
                        if ai.is_comparable:  # i.e. a = I*real
                            s *= I
                            if ai.is_extended_negative:
                                # can't use sign(ai) here since ai might not be
                                # a Number
                                s = -s
                        else:
                            unk.append(a)
                    else:
                        unk.append(a)
            if c is S.One and len(unk) == len(args):
                return None
            return s * cls(arg._new_rawargs(*unk))
        if arg is S.NaN:
            return S.NaN
        if arg.is_zero:  # it may be an Expr that is zero
            return S.Zero
        if arg.is_extended_positive:
            return S.One
        if arg.is_extended_negative:
            return S.NegativeOne
        if arg.is_Function:
            if isinstance(arg, sign):
                return arg
        if arg.is_imaginary:
            if arg.is_Pow and arg.exp is S.Half:
                # we catch this because non-trivial sqrt args are not expanded
                # e.g. sqrt(1-sqrt(2)) --x-->  to I*sqrt(sqrt(2) - 1)
                return I
            arg2 = -I * arg
            if arg2.is_extended_positive:
                return I
            if arg2.is_extended_negative:
                return -I

    def _eval_Abs(self):
        if fuzzy_not(self.args[0].is_zero):
            return S.One

    def _eval_conjugate(self):
        return sign(conjugate(self.args[0]))

    def _eval_derivative(self, x):
        if self.args[0].is_extended_real:
            from sympy.functions.special.delta_functions import DiracDelta
            return 2 * Derivative(self.args[0], x, evaluate=True) \
                * DiracDelta(self.args[0])
        elif self.args[0].is_imaginary:
            from sympy.functions.special.delta_functions import DiracDelta
            return 2 * Derivative(self.args[0], x, evaluate=True) \
                * DiracDelta(-I * self.args[0])

    def _eval_is_nonnegative(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_is_nonpositive(self):
        if self.args[0].is_nonpositive:
            return True

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def _eval_is_integer(self):
        return self.args[0].is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_power(self, other):
        if (
            fuzzy_not(self.args[0].is_zero) and
            other.is_integer and
            other.is_even
        ):
            return S.One

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg0 = self.args[0]
        x0 = arg0.subs(x, 0)
        if x0 != 0:
            return self.func(x0)
        if cdir != 0:
            cdir = arg0.dir(x, cdir)
        return -S.One if re(cdir) < 0 else S.One

    def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
        if arg.is_extended_real:
            return Piecewise((1, arg > 0), (-1, arg < 0), (0, True))

    def _eval_rewrite_as_Heaviside(self, arg, **kwargs):
        from sympy.functions.special.delta_functions import Heaviside
        if arg.is_extended_real:
            return Heaviside(arg) * 2 - 1

    def _eval_rewrite_as_Abs(self, arg, **kwargs):
        return Piecewise((0, Eq(arg, 0)), (arg / Abs(arg), True))

    def _eval_simplify(self, **kwargs):
        return self.func(factor_terms(self.args[0]))  # XXX include doit?


class Abs(Function):
    """
    Return the absolute value of the argument.

    Explanation
    ===========

    This is an extension of the built-in function ``abs()`` to accept symbolic
    values.  If you pass a SymPy expression to the built-in ``abs()``, it will
    pass it automatically to ``Abs()``.

    Examples
    ========

    >>> from sympy import Abs, Symbol, S, I
    >>> Abs(-1)
    1
    >>> x = Symbol('x', real=True)
    >>> Abs(-x)
    Abs(x)
    >>> Abs(x**2)
    x**2
    >>> abs(-x) # The Python built-in
    Abs(x)
    >>> Abs(3*x + 2*I)
    sqrt(9*x**2 + 4)
    >>> Abs(8*I)
    8

    Note that the Python built-in will return either an Expr or int depending on
    the argument::

        >>> type(abs(-1))
        <... 'int'>
        >>> type(abs(S.NegativeOne))
        <class 'sympy.core.numbers.One'>

    Abs will always return a SymPy object.

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Absolute value returned can be an expression or integer depending on
        input arg.

    See Also
    ========

    sign, conjugate
    """

    args: tTuple[Expr]

    is_extended_real = True
    is_extended_negative = False
    is_extended_nonnegative = True
    unbranched = True
    _singularities = True  # non-holomorphic

    def fdiff(self, argindex=1):
        """
        Get the first derivative of the argument to Abs().

        """
        if argindex == 1:
            return sign(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.simplify.simplify import signsimp

        if hasattr(arg, '_eval_Abs'):
            obj = arg._eval_Abs()
            if obj is not None:
                return obj
        if not isinstance(arg, Expr):
            raise TypeError("Bad argument type for Abs(): %s" % type(arg))

        # handle what we can
        arg = signsimp(arg, evaluate=False)
        n, d = arg.as_numer_denom()
        if d.free_symbols and not n.free_symbols:
            return cls(n)/cls(d)

        if arg.is_Mul:
            known = []
            unk = []
            for t in arg.args:
                if t.is_Pow and t.exp.is_integer and t.exp.is_negative:
                    bnew = cls(t.base)
                    if isinstance(bnew, cls):
                        unk.append(t)
                    else:
                        known.append(Pow(bnew, t.exp))
                else:
                    tnew = cls(t)
                    if isinstance(tnew, cls):
                        unk.append(t)
                    else:
                        known.append(tnew)
            known = Mul(*known)
            unk = cls(Mul(*unk), evaluate=False) if unk else S.One
            return known*unk
        if arg is S.NaN:
            return S.NaN
        if arg is S.ComplexInfinity:
            return oo
        from sympy.functions.elementary.exponential import exp, log

        if arg.is_Pow:
            base, exponent = arg.as_base_exp()
            if base.is_extended_real:
                if exponent.is_integer:
                    if exponent.is_even:
                        return arg
                    if base is S.NegativeOne:
                        return S.One
                    return Abs(base)**exponent
                if base.is_extended_nonnegative:
                    return base**re(exponent)
                if base.is_extended_negative:
                    return (-base)**re(exponent)*exp(-pi*im(exponent))
                return
            elif not base.has(Symbol): # complex base
                # express base**exponent as exp(exponent*log(base))
                a, b = log(base).as_real_imag()
                z = a + I*b
                return exp(re(exponent*z))
        if isinstance(arg, exp):
            return exp(re(arg.args[0]))
        if isinstance(arg, AppliedUndef):
            if arg.is_positive:
                return arg
            elif arg.is_negative:
                return -arg
            return
        if arg.is_Add and arg.has(oo, S.NegativeInfinity):
            if any(a.is_infinite for a in arg.as_real_imag()):
                return oo
        if arg.is_zero:
            return S.Zero
        if arg.is_extended_nonnegative:
            return arg
        if arg.is_extended_nonpositive:
            return -arg
        if arg.is_imaginary:
            arg2 = -I * arg
            if arg2.is_extended_nonnegative:
                return arg2
        if arg.is_extended_real:
            return
        # reject result if all new conjugates are just wrappers around
        # an expression that was already in the arg
        conj = signsimp(arg.conjugate(), evaluate=False)
        new_conj = conj.atoms(conjugate) - arg.atoms(conjugate)
        if new_conj and all(arg.has(i.args[0]) for i in new_conj):
            return
        if arg != conj and arg != -conj:
            ignore = arg.atoms(Abs)
            abs_free_arg = arg.xreplace({i: Dummy(real=True) for i in ignore})
            unk = [a for a in abs_free_arg.free_symbols if a.is_extended_real is None]
            if not unk or not all(conj.has(conjugate(u)) for u in unk):
                return sqrt(expand_mul(arg*conj))

    def _eval_is_real(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_integer(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_integer

    def _eval_is_extended_nonzero(self):
        return fuzzy_not(self._args[0].is_zero)

    def _eval_is_zero(self):
        return self._args[0].is_zero

    def _eval_is_extended_positive(self):
        return fuzzy_not(self._args[0].is_zero)

    def _eval_is_rational(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_rational

    def _eval_is_even(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_even

    def _eval_is_odd(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_odd

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_power(self, exponent):
        if self.args[0].is_extended_real and exponent.is_integer:
            if exponent.is_even:
                return self.args[0]**exponent
            elif exponent is not S.NegativeOne and exponent.is_Integer:
                return self.args[0]**(exponent - 1)*self
        return

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.functions.elementary.exponential import log
        direction = self.args[0].leadterm(x)[0]
        if direction.has(log(x)):
            direction = direction.subs(log(x), logx)
        s = self.args[0]._eval_nseries(x, n=n, logx=logx)
        return (sign(direction)*s).expand()

    def _eval_derivative(self, x):
        if self.args[0].is_extended_real or self.args[0].is_imaginary:
            return Derivative(self.args[0], x, evaluate=True) \
                * sign(conjugate(self.args[0]))
        rv = (re(self.args[0]) * Derivative(re(self.args[0]), x,
            evaluate=True) + im(self.args[0]) * Derivative(im(self.args[0]),
                x, evaluate=True)) / Abs(self.args[0])
        return rv.rewrite(sign)

    def _eval_rewrite_as_Heaviside(self, arg, **kwargs):
        # Note this only holds for real arg (since Heaviside is not defined
        # for complex arguments).
        from sympy.functions.special.delta_functions import Heaviside
        if arg.is_extended_real:
            return arg*(Heaviside(arg) - Heaviside(-arg))

    def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
        if arg.is_extended_real:
            return Piecewise((arg, arg >= 0), (-arg, True))
        elif arg.is_imaginary:
            return Piecewise((I*arg, I*arg >= 0), (-I*arg, True))

    def _eval_rewrite_as_sign(self, arg, **kwargs):
        return arg/sign(arg)

    def _eval_rewrite_as_conjugate(self, arg, **kwargs):
        return sqrt(arg*conjugate(arg))


class arg(Function):
    r"""
    Returns the argument (in radians) of a complex number. The argument is
    evaluated in consistent convention with ``atan2`` where the branch-cut is
    taken along the negative real axis and ``arg(z)`` is in the interval
    $(-\pi,\pi]$. For a positive number, the argument is always 0; the
    argument of a negative number is $\pi$; and the argument of 0
    is undefined and returns ``nan``. So the ``arg`` function will never nest
    greater than 3 levels since at the 4th application, the result must be
    nan; for a real number, nan is returned on the 3rd application.

    Examples
    ========

    >>> from sympy import arg, I, sqrt, Dummy
    >>> from sympy.abc import x
    >>> arg(2.0)
    0
    >>> arg(I)
    pi/2
    >>> arg(sqrt(2) + I*sqrt(2))
    pi/4
    >>> arg(sqrt(3)/2 + I/2)
    pi/6
    >>> arg(4 + 3*I)
    atan(3/4)
    >>> arg(0.8 + 0.6*I)
    0.643501108793284
    >>> arg(arg(arg(arg(x))))
    nan
    >>> real = Dummy(real=True)
    >>> arg(arg(arg(real)))
    nan

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    value : Expr
        Returns arc tangent of arg measured in radians.

    """

    is_extended_real = True
    is_real = True
    is_finite = True
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        a = arg
        for i in range(3):
            if isinstance(a, cls):
                a = a.args[0]
            else:
                if i == 2 and a.is_extended_real:
                    return S.NaN
                break
        else:
            return S.NaN
        from sympy.functions.elementary.exponential import exp, exp_polar
        if isinstance(arg, exp_polar):
            return periodic_argument(arg, oo)
        elif isinstance(arg, exp):
            i_ = im(arg.args[0])
            if i_.is_comparable:
                i_ %= 2*S.Pi
                if i_ > S.Pi:
                    i_ -= 2*S.Pi
                return i_

        if not arg.is_Atom:
            c, arg_ = factor_terms(arg).as_coeff_Mul()
            if arg_.is_Mul:
                arg_ = Mul(*[a if (sign(a) not in (-1, 1)) else
                    sign(a) for a in arg_.args])
            arg_ = sign(c)*arg_
        else:
            arg_ = arg
        if any(i.is_extended_positive is None for i in arg_.atoms(AppliedUndef)):
            return
        from sympy.functions.elementary.trigonometric import atan2
        x, y = arg_.as_real_imag()
        rv = atan2(y, x)
        if rv.is_number:
            return rv
        if arg_ != arg:
            return cls(arg_, evaluate=False)

    def _eval_derivative(self, t):
        x, y = self.args[0].as_real_imag()
        return (x * Derivative(y, t, evaluate=True) - y *
                    Derivative(x, t, evaluate=True)) / (x**2 + y**2)

    def _eval_rewrite_as_atan2(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import atan2
        x, y = self.args[0].as_real_imag()
        return atan2(y, x)


class conjugate(Function):
    """
    Returns the *complex conjugate* [1]_ of an argument.
    In mathematics, the complex conjugate of a complex number
    is given by changing the sign of the imaginary part.

    Thus, the conjugate of the complex number
    :math:`a + ib` (where $a$ and $b$ are real numbers) is :math:`a - ib`

    Examples
    ========

    >>> from sympy import conjugate, I
    >>> conjugate(2)
    2
    >>> conjugate(I)
    -I
    >>> conjugate(3 + 2*I)
    3 - 2*I
    >>> conjugate(5 - I)
    5 + I

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    arg : Expr
        Complex conjugate of arg as real, imaginary or mixed expression.

    See Also
    ========

    sign, Abs

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_conjugation
    """
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_conjugate()
        if obj is not None:
            return obj

    def inverse(self):
        return conjugate

    def _eval_Abs(self):
        return Abs(self.args[0], evaluate=True)

    def _eval_adjoint(self):
        return transpose(self.args[0])

    def _eval_conjugate(self):
        return self.args[0]

    def _eval_derivative(self, x):
        if x.is_real:
            return conjugate(Derivative(self.args[0], x, evaluate=True))
        elif x.is_imaginary:
            return -conjugate(Derivative(self.args[0], x, evaluate=True))

    def _eval_transpose(self):
        return adjoint(self.args[0])

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic


class transpose(Function):
    """
    Linear map transposition.

    Examples
    ========

    >>> from sympy import transpose, Matrix, MatrixSymbol
    >>> A = MatrixSymbol('A', 25, 9)
    >>> transpose(A)
    A.T
    >>> B = MatrixSymbol('B', 9, 22)
    >>> transpose(B)
    B.T
    >>> transpose(A*B)
    B.T*A.T
    >>> M = Matrix([[4, 5], [2, 1], [90, 12]])
    >>> M
    Matrix([
    [ 4,  5],
    [ 2,  1],
    [90, 12]])
    >>> transpose(M)
    Matrix([
    [4, 2, 90],
    [5, 1, 12]])

    Parameters
    ==========

    arg : Matrix
         Matrix or matrix expression to take the transpose of.

    Returns
    =======

    value : Matrix
        Transpose of arg.

    """

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_transpose()
        if obj is not None:
            return obj

    def _eval_adjoint(self):
        return conjugate(self.args[0])

    def _eval_conjugate(self):
        return adjoint(self.args[0])

    def _eval_transpose(self):
        return self.args[0]


class adjoint(Function):
    """
    Conjugate transpose or Hermite conjugation.

    Examples
    ========

    >>> from sympy import adjoint, MatrixSymbol
    >>> A = MatrixSymbol('A', 10, 5)
    >>> adjoint(A)
    Adjoint(A)

    Parameters
    ==========

    arg : Matrix
        Matrix or matrix expression to take the adjoint of.

    Returns
    =======

    value : Matrix
        Represents the conjugate transpose or Hermite
        conjugation of arg.

    """

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_adjoint()
        if obj is not None:
            return obj
        obj = arg._eval_transpose()
        if obj is not None:
            return conjugate(obj)

    def _eval_adjoint(self):
        return self.args[0]

    def _eval_conjugate(self):
        return transpose(self.args[0])

    def _eval_transpose(self):
        return conjugate(self.args[0])

    def _latex(self, printer, exp=None, *args):
        arg = printer._print(self.args[0])
        tex = r'%s^{\dagger}' % arg
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if printer._use_unicode:
            pform = pform**prettyForm('\N{DAGGER}')
        else:
            pform = pform**prettyForm('+')
        return pform

###############################################################################
############### HANDLING OF POLAR NUMBERS #####################################
###############################################################################


class polar_lift(Function):
    """
    Lift argument to the Riemann surface of the logarithm, using the
    standard branch.

    Examples
    ========

    >>> from sympy import Symbol, polar_lift, I
    >>> p = Symbol('p', polar=True)
    >>> x = Symbol('x')
    >>> polar_lift(4)
    4*exp_polar(0)
    >>> polar_lift(-4)
    4*exp_polar(I*pi)
    >>> polar_lift(-I)
    exp_polar(-I*pi/2)
    >>> polar_lift(I + 2)
    polar_lift(2 + I)

    >>> polar_lift(4*x)
    4*polar_lift(x)
    >>> polar_lift(4*p)
    4*p

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    periodic_argument
    """

    is_polar = True
    is_comparable = False  # Cannot be evalf'd.

    @classmethod
    def eval(cls, arg):
        from sympy.functions.elementary.complexes import arg as argument
        if arg.is_number:
            ar = argument(arg)
            # In general we want to affirm that something is known,
            # e.g. `not ar.has(argument) and not ar.has(atan)`
            # but for now we will just be more restrictive and
            # see that it has evaluated to one of the known values.
            if ar in (0, pi/2, -pi/2, pi):
                from sympy.functions.elementary.exponential import exp_polar
                return exp_polar(I*ar)*abs(arg)

        if arg.is_Mul:
            args = arg.args
        else:
            args = [arg]
        included = []
        excluded = []
        positive = []
        for arg in args:
            if arg.is_polar:
                included += [arg]
            elif arg.is_positive:
                positive += [arg]
            else:
                excluded += [arg]
        if len(excluded) < len(args):
            if excluded:
                return Mul(*(included + positive))*polar_lift(Mul(*excluded))
            elif included:
                return Mul(*(included + positive))
            else:
                from sympy.functions.elementary.exponential import exp_polar
                return Mul(*positive)*exp_polar(0)

    def _eval_evalf(self, prec):
        """ Careful! any evalf of polar numbers is flaky """
        return self.args[0]._eval_evalf(prec)

    def _eval_Abs(self):
        return Abs(self.args[0], evaluate=True)


class periodic_argument(Function):
    r"""
    Represent the argument on a quotient of the Riemann surface of the
    logarithm. That is, given a period $P$, always return a value in
    $(-P/2, P/2]$, by using $\exp(PI) = 1$.

    Examples
    ========

    >>> from sympy import exp_polar, periodic_argument
    >>> from sympy import I, pi
    >>> periodic_argument(exp_polar(10*I*pi), 2*pi)
    0
    >>> periodic_argument(exp_polar(5*I*pi), 4*pi)
    pi
    >>> from sympy import exp_polar, periodic_argument
    >>> from sympy import I, pi
    >>> periodic_argument(exp_polar(5*I*pi), 2*pi)
    pi
    >>> periodic_argument(exp_polar(5*I*pi), 3*pi)
    -pi
    >>> periodic_argument(exp_polar(5*I*pi), pi)
    0

    Parameters
    ==========

    ar : Expr
        A polar number.

    period : Expr
        The period $P$.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : Lift argument to the Riemann surface of the logarithm
    principal_branch
    """

    @classmethod
    def _getunbranched(cls, ar):
        from sympy.functions.elementary.exponential import exp_polar, log
        if ar.is_Mul:
            args = ar.args
        else:
            args = [ar]
        unbranched = 0
        for a in args:
            if not a.is_polar:
                unbranched += arg(a)
            elif isinstance(a, exp_polar):
                unbranched += a.exp.as_real_imag()[1]
            elif a.is_Pow:
                re, im = a.exp.as_real_imag()
                unbranched += re*unbranched_argument(
                    a.base) + im*log(abs(a.base))
            elif isinstance(a, polar_lift):
                unbranched += arg(a.args[0])
            else:
                return None
        return unbranched

    @classmethod
    def eval(cls, ar, period):
        # Our strategy is to evaluate the argument on the Riemann surface of the
        # logarithm, and then reduce.
        # NOTE evidently this means it is a rather bad idea to use this with
        # period != 2*pi and non-polar numbers.
        if not period.is_extended_positive:
            return None
        if period == oo and isinstance(ar, principal_branch):
            return periodic_argument(*ar.args)
        if isinstance(ar, polar_lift) and period >= 2*pi:
            return periodic_argument(ar.args[0], period)
        if ar.is_Mul:
            newargs = [x for x in ar.args if not x.is_positive]
            if len(newargs) != len(ar.args):
                return periodic_argument(Mul(*newargs), period)
        unbranched = cls._getunbranched(ar)
        if unbranched is None:
            return None
        from sympy.functions.elementary.trigonometric import atan, atan2
        if unbranched.has(periodic_argument, atan2, atan):
            return None
        if period == oo:
            return unbranched
        if period != oo:
            from sympy.functions.elementary.integers import ceiling
            n = ceiling(unbranched/period - S.Half)*period
            if not n.has(ceiling):
                return unbranched - n

    def _eval_evalf(self, prec):
        z, period = self.args
        if period == oo:
            unbranched = periodic_argument._getunbranched(z)
            if unbranched is None:
                return self
            return unbranched._eval_evalf(prec)
        ub = periodic_argument(z, oo)._eval_evalf(prec)
        from sympy.functions.elementary.integers import ceiling
        return (ub - ceiling(ub/period - S.Half)*period)._eval_evalf(prec)


def unbranched_argument(arg):
    '''
    Returns periodic argument of arg with period as infinity.

    Examples
    ========

    >>> from sympy import exp_polar, unbranched_argument
    >>> from sympy import I, pi
    >>> unbranched_argument(exp_polar(15*I*pi))
    15*pi
    >>> unbranched_argument(exp_polar(7*I*pi))
    7*pi

    See also
    ========

    periodic_argument
    '''
    return periodic_argument(arg, oo)


class principal_branch(Function):
    """
    Represent a polar number reduced to its principal branch on a quotient
    of the Riemann surface of the logarithm.

    Explanation
    ===========

    This is a function of two arguments. The first argument is a polar
    number `z`, and the second one a positive real number or infinity, `p`.
    The result is ``z mod exp_polar(I*p)``.

    Examples
    ========

    >>> from sympy import exp_polar, principal_branch, oo, I, pi
    >>> from sympy.abc import z
    >>> principal_branch(z, oo)
    z
    >>> principal_branch(exp_polar(2*pi*I)*3, 2*pi)
    3*exp_polar(0)
    >>> principal_branch(exp_polar(2*pi*I)*3*z, 2*pi)
    3*principal_branch(z, 2*pi)

    Parameters
    ==========

    x : Expr
        A polar number.

    period : Expr
        Positive real number or infinity.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : Lift argument to the Riemann surface of the logarithm
    periodic_argument
    """

    is_polar = True
    is_comparable = False  # cannot always be evalf'd

    @classmethod
    def eval(self, x, period):
        from sympy.functions.elementary.exponential import exp_polar
        if isinstance(x, polar_lift):
            return principal_branch(x.args[0], period)
        if period == oo:
            return x
        ub = periodic_argument(x, oo)
        barg = periodic_argument(x, period)
        if ub != barg and not ub.has(periodic_argument) \
                and not barg.has(periodic_argument):
            pl = polar_lift(x)

            def mr(expr):
                if not isinstance(expr, Symbol):
                    return polar_lift(expr)
                return expr
            pl = pl.replace(polar_lift, mr)
            # Recompute unbranched argument
            ub = periodic_argument(pl, oo)
            if not pl.has(polar_lift):
                if ub != barg:
                    res = exp_polar(I*(barg - ub))*pl
                else:
                    res = pl
                if not res.is_polar and not res.has(exp_polar):
                    res *= exp_polar(0)
                return res

        if not x.free_symbols:
            c, m = x, ()
        else:
            c, m = x.as_coeff_mul(*x.free_symbols)
        others = []
        for y in m:
            if y.is_positive:
                c *= y
            else:
                others += [y]
        m = tuple(others)
        arg = periodic_argument(c, period)
        if arg.has(periodic_argument):
            return None
        if arg.is_number and (unbranched_argument(c) != arg or
                              (arg == 0 and m != () and c != 1)):
            if arg == 0:
                return abs(c)*principal_branch(Mul(*m), period)
            return principal_branch(exp_polar(I*arg)*Mul(*m), period)*abs(c)
        if arg.is_number and ((abs(arg) < period/2) == True or arg == period/2) \
                and m == ():
            return exp_polar(arg*I)*abs(c)

    def _eval_evalf(self, prec):
        z, period = self.args
        p = periodic_argument(z, period)._eval_evalf(prec)
        if abs(p) > pi or p == -pi:
            return self  # Cannot evalf for this argument.
        from sympy.functions.elementary.exponential import exp
        return (abs(z)*exp(I*p))._eval_evalf(prec)


def _polarify(eq, lift, pause=False):
    from sympy.integrals.integrals import Integral
    if eq.is_polar:
        return eq
    if eq.is_number and not pause:
        return polar_lift(eq)
    if isinstance(eq, Symbol) and not pause and lift:
        return polar_lift(eq)
    elif eq.is_Atom:
        return eq
    elif eq.is_Add:
        r = eq.func(*[_polarify(arg, lift, pause=True) for arg in eq.args])
        if lift:
            return polar_lift(r)
        return r
    elif eq.is_Pow and eq.base == S.Exp1:
        return eq.func(S.Exp1, _polarify(eq.exp, lift, pause=False))
    elif eq.is_Function:
        return eq.func(*[_polarify(arg, lift, pause=False) for arg in eq.args])
    elif isinstance(eq, Integral):
        # Don't lift the integration variable
        func = _polarify(eq.function, lift, pause=pause)
        limits = []
        for limit in eq.args[1:]:
            var = _polarify(limit[0], lift=False, pause=pause)
            rest = _polarify(limit[1:], lift=lift, pause=pause)
            limits.append((var,) + rest)
        return Integral(*((func,) + tuple(limits)))
    else:
        return eq.func(*[_polarify(arg, lift, pause=pause)
                         if isinstance(arg, Expr) else arg for arg in eq.args])


def polarify(eq, subs=True, lift=False):
    """
    Turn all numbers in eq into their polar equivalents (under the standard
    choice of argument).

    Note that no attempt is made to guess a formal convention of adding
    polar numbers, expressions like $1 + x$ will generally not be altered.

    Note also that this function does not promote ``exp(x)`` to ``exp_polar(x)``.

    If ``subs`` is ``True``, all symbols which are not already polar will be
    substituted for polar dummies; in this case the function behaves much
    like :func:`~.posify`.

    If ``lift`` is ``True``, both addition statements and non-polar symbols are
    changed to their ``polar_lift()``ed versions.
    Note that ``lift=True`` implies ``subs=False``.

    Examples
    ========

    >>> from sympy import polarify, sin, I
    >>> from sympy.abc import x, y
    >>> expr = (-x)**y
    >>> expr.expand()
    (-x)**y
    >>> polarify(expr)
    ((_x*exp_polar(I*pi))**_y, {_x: x, _y: y})
    >>> polarify(expr)[0].expand()
    _x**_y*exp_polar(_y*I*pi)
    >>> polarify(x, lift=True)
    polar_lift(x)
    >>> polarify(x*(1+y), lift=True)
    polar_lift(x)*polar_lift(y + 1)

    Adds are treated carefully:

    >>> polarify(1 + sin((1 + I)*x))
    (sin(_x*polar_lift(1 + I)) + 1, {_x: x})
    """
    if lift:
        subs = False
    eq = _polarify(sympify(eq), lift)
    if not subs:
        return eq
    reps = {s: Dummy(s.name, polar=True) for s in eq.free_symbols}
    eq = eq.subs(reps)
    return eq, {r: s for s, r in reps.items()}


def _unpolarify(eq, exponents_only, pause=False):
    if not isinstance(eq, Basic) or eq.is_Atom:
        return eq

    if not pause:
        from sympy.functions.elementary.exponential import exp, exp_polar
        if isinstance(eq, exp_polar):
            return exp(_unpolarify(eq.exp, exponents_only))
        if isinstance(eq, principal_branch) and eq.args[1] == 2*pi:
            return _unpolarify(eq.args[0], exponents_only)
        if (
            eq.is_Add or eq.is_Mul or eq.is_Boolean or
            eq.is_Relational and (
                eq.rel_op in ('==', '!=') and 0 in eq.args or
                eq.rel_op not in ('==', '!='))
        ):
            return eq.func(*[_unpolarify(x, exponents_only) for x in eq.args])
        if isinstance(eq, polar_lift):
            return _unpolarify(eq.args[0], exponents_only)

    if eq.is_Pow:
        expo = _unpolarify(eq.exp, exponents_only)
        base = _unpolarify(eq.base, exponents_only,
            not (expo.is_integer and not pause))
        return base**expo

    if eq.is_Function and getattr(eq.func, 'unbranched', False):
        return eq.func(*[_unpolarify(x, exponents_only, exponents_only)
            for x in eq.args])

    return eq.func(*[_unpolarify(x, exponents_only, True) for x in eq.args])


def unpolarify(eq, subs=None, exponents_only=False):
    """
    If `p` denotes the projection from the Riemann surface of the logarithm to
    the complex line, return a simplified version `eq'` of `eq` such that
    `p(eq') = p(eq)`.
    Also apply the substitution subs in the end. (This is a convenience, since
    ``unpolarify``, in a certain sense, undoes :func:`polarify`.)

    Examples
    ========

    >>> from sympy import unpolarify, polar_lift, sin, I
    >>> unpolarify(polar_lift(I + 2))
    2 + I
    >>> unpolarify(sin(polar_lift(I + 7)))
    sin(7 + I)
    """
    if isinstance(eq, bool):
        return eq

    eq = sympify(eq)
    if subs is not None:
        return unpolarify(eq.subs(subs))
    changed = True
    pause = False
    if exponents_only:
        pause = True
    while changed:
        changed = False
        res = _unpolarify(eq, exponents_only, pause)
        if res != eq:
            changed = True
            eq = res
        if isinstance(res, bool):
            return res
    # Finally, replacing Exp(0) by 1 is always correct.
    # So is polar_lift(0) -> 0.
    from sympy.functions.elementary.exponential import exp_polar
    return res.subs({exp_polar(0): 1, polar_lift(0): 0})
