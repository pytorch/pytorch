from sympy.core import S, diff
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.complexes import im, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.utilities.misc import filldedent


###############################################################################
################################ DELTA FUNCTION ###############################
###############################################################################


class DiracDelta(DefinedFunction):
    r"""
    The DiracDelta function and its derivatives.

    Explanation
    ===========

    DiracDelta is not an ordinary function. It can be rigorously defined either
    as a distribution or as a measure.

    DiracDelta only makes sense in definite integrals, and in particular,
    integrals of the form ``Integral(f(x)*DiracDelta(x - x0), (x, a, b))``,
    where it equals ``f(x0)`` if ``a <= x0 <= b`` and ``0`` otherwise. Formally,
    DiracDelta acts in some ways like a function that is ``0`` everywhere except
    at ``0``, but in many ways it also does not. It can often be useful to treat
    DiracDelta in formal ways, building up and manipulating expressions with
    delta functions (which may eventually be integrated), but care must be taken
    to not treat it as a real function. SymPy's ``oo`` is similar. It only
    truly makes sense formally in certain contexts (such as integration limits),
    but SymPy allows its use everywhere, and it tries to be consistent with
    operations on it (like ``1/oo``), but it is easy to get into trouble and get
    wrong results if ``oo`` is treated too much like a number. Similarly, if
    DiracDelta is treated too much like a function, it is easy to get wrong or
    nonsensical results.

    DiracDelta function has the following properties:

    1) $\frac{d}{d x} \theta(x) = \delta(x)$
    2) $\int_{-\infty}^\infty \delta(x - a)f(x)\, dx = f(a)$ and $\int_{a-
       \epsilon}^{a+\epsilon} \delta(x - a)f(x)\, dx = f(a)$
    3) $\delta(x) = 0$ for all $x \neq 0$
    4) $\delta(g(x)) = \sum_i \frac{\delta(x - x_i)}{\|g'(x_i)\|}$ where $x_i$
       are the roots of $g$
    5) $\delta(-x) = \delta(x)$

    Derivatives of ``k``-th order of DiracDelta have the following properties:

    6) $\delta(x, k) = 0$ for all $x \neq 0$
    7) $\delta(-x, k) = -\delta(x, k)$ for odd $k$
    8) $\delta(-x, k) = \delta(x, k)$ for even $k$

    Examples
    ========

    >>> from sympy import DiracDelta, diff, pi
    >>> from sympy.abc import x, y

    >>> DiracDelta(x)
    DiracDelta(x)
    >>> DiracDelta(1)
    0
    >>> DiracDelta(-1)
    0
    >>> DiracDelta(pi)
    0
    >>> DiracDelta(x - 4).subs(x, 4)
    DiracDelta(0)
    >>> diff(DiracDelta(x))
    DiracDelta(x, 1)
    >>> diff(DiracDelta(x - 1), x, 2)
    DiracDelta(x - 1, 2)
    >>> diff(DiracDelta(x**2 - 1), x, 2)
    2*(2*x**2*DiracDelta(x**2 - 1, 2) + DiracDelta(x**2 - 1, 1))
    >>> DiracDelta(3*x).is_simple(x)
    True
    >>> DiracDelta(x**2).is_simple(x)
    False
    >>> DiracDelta((x**2 - 1)*y).expand(diracdelta=True, wrt=x)
    DiracDelta(x - 1)/(2*Abs(y)) + DiracDelta(x + 1)/(2*Abs(y))

    See Also
    ========

    Heaviside
    sympy.simplify.simplify.simplify, is_simple
    sympy.functions.special.tensor_functions.KroneckerDelta

    References
    ==========

    .. [1] https://mathworld.wolfram.com/DeltaFunction.html

    """

    is_real = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of a DiracDelta Function.

        Explanation
        ===========

        The difference between ``diff()`` and ``fdiff()`` is: ``diff()`` is the
        user-level function and ``fdiff()`` is an object method. ``fdiff()`` is
        a convenience method available in the ``Function`` class. It returns
        the derivative of the function without considering the chain rule.
        ``diff(function, x)`` calls ``Function._eval_derivative`` which in turn
        calls ``fdiff()`` internally to compute the derivative of the function.

        Examples
        ========

        >>> from sympy import DiracDelta, diff
        >>> from sympy.abc import x

        >>> DiracDelta(x).fdiff()
        DiracDelta(x, 1)

        >>> DiracDelta(x, 1).fdiff()
        DiracDelta(x, 2)

        >>> DiracDelta(x**2 - 1).fdiff()
        DiracDelta(x**2 - 1, 1)

        >>> diff(DiracDelta(x, 1)).fdiff()
        DiracDelta(x, 3)

        Parameters
        ==========

        argindex : integer
            degree of derivative

        """
        if argindex == 1:
            #I didn't know if there is a better way to handle default arguments
            k = 0
            if len(self.args) > 1:
                k = self.args[1]
            return self.func(self.args[0], k + 1)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg, k=S.Zero):
        """
        Returns a simplified form or a value of DiracDelta depending on the
        argument passed by the DiracDelta object.

        Explanation
        ===========

        The ``eval()`` method is automatically called when the ``DiracDelta``
        class is about to be instantiated and it returns either some simplified
        instance or the unevaluated instance depending on the argument passed.
        In other words, ``eval()`` method is not needed to be called explicitly,
        it is being called and evaluated once the object is called.

        Examples
        ========

        >>> from sympy import DiracDelta, S
        >>> from sympy.abc import x

        >>> DiracDelta(x)
        DiracDelta(x)

        >>> DiracDelta(-x, 1)
        -DiracDelta(x, 1)

        >>> DiracDelta(1)
        0

        >>> DiracDelta(5, 1)
        0

        >>> DiracDelta(0)
        DiracDelta(0)

        >>> DiracDelta(-1)
        0

        >>> DiracDelta(S.NaN)
        nan

        >>> DiracDelta(x - 100).subs(x, 5)
        0

        >>> DiracDelta(x - 100).subs(x, 100)
        DiracDelta(0)

        Parameters
        ==========

        k : integer
            order of derivative

        arg : argument passed to DiracDelta

        """
        if not k.is_Integer or k.is_negative:
            raise ValueError("Error: the second argument of DiracDelta must be \
            a non-negative integer, %s given instead." % (k,))
        if arg is S.NaN:
            return S.NaN
        if arg.is_nonzero:
            return S.Zero
        if fuzzy_not(im(arg).is_zero):
            raise ValueError(filldedent('''
                Function defined only for Real Values.
                Complex part: %s  found in %s .''' % (
                repr(im(arg)), repr(arg))))
        c, nc = arg.args_cnc()
        if c and c[0] is S.NegativeOne:
            # keep this fast and simple instead of using
            # could_extract_minus_sign
            if k.is_odd:
                return -cls(-arg, k)
            elif k.is_even:
                return cls(-arg, k) if k else cls(-arg)
        elif k.is_zero:
            return cls(arg, evaluate=False)

    def _eval_expand_diracdelta(self, **hints):
        """
        Compute a simplified representation of the function using
        property number 4. Pass ``wrt`` as a hint to expand the expression
        with respect to a particular variable.

        Explanation
        ===========

        ``wrt`` is:

        - a variable with respect to which a DiracDelta expression will
        get expanded.

        Examples
        ========

        >>> from sympy import DiracDelta
        >>> from sympy.abc import x, y

        >>> DiracDelta(x*y).expand(diracdelta=True, wrt=x)
        DiracDelta(x)/Abs(y)
        >>> DiracDelta(x*y).expand(diracdelta=True, wrt=y)
        DiracDelta(y)/Abs(x)

        >>> DiracDelta(x**2 + x - 2).expand(diracdelta=True, wrt=x)
        DiracDelta(x - 1)/3 + DiracDelta(x + 2)/3

        See Also
        ========

        is_simple, Diracdelta

        """
        wrt = hints.get('wrt', None)
        if wrt is None:
            free = self.free_symbols
            if len(free) == 1:
                wrt = free.pop()
            else:
                raise TypeError(filldedent('''
            When there is more than 1 free symbol or variable in the expression,
            the 'wrt' keyword is required as a hint to expand when using the
            DiracDelta hint.'''))

        if not self.args[0].has(wrt) or (len(self.args) > 1 and self.args[1] != 0 ):
            return self
        try:
            argroots = roots(self.args[0], wrt)
            result = 0
            valid = True
            darg = abs(diff(self.args[0], wrt))
            for r, m in argroots.items():
                if r.is_real is not False and m == 1:
                    result += self.func(wrt - r)/darg.subs(wrt, r)
                else:
                    # don't handle non-real and if m != 1 then
                    # a polynomial will have a zero in the derivative (darg)
                    # at r
                    valid = False
                    break
            if valid:
                return result
        except PolynomialError:
            pass
        return self

    def is_simple(self, x):
        """
        Tells whether the argument(args[0]) of DiracDelta is a linear
        expression in *x*.

        Examples
        ========

        >>> from sympy import DiracDelta, cos
        >>> from sympy.abc import x, y

        >>> DiracDelta(x*y).is_simple(x)
        True
        >>> DiracDelta(x*y).is_simple(y)
        True

        >>> DiracDelta(x**2 + x - 2).is_simple(x)
        False

        >>> DiracDelta(cos(x)).is_simple(x)
        False

        Parameters
        ==========

        x : can be a symbol

        See Also
        ========

        sympy.simplify.simplify.simplify, DiracDelta

        """
        p = self.args[0].as_poly(x)
        if p:
            return p.degree() == 1
        return False

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        """
        Represents DiracDelta in a piecewise form.

        Examples
        ========

        >>> from sympy import DiracDelta, Piecewise, Symbol
        >>> x = Symbol('x')

        >>> DiracDelta(x).rewrite(Piecewise)
        Piecewise((DiracDelta(0), Eq(x, 0)), (0, True))

        >>> DiracDelta(x - 5).rewrite(Piecewise)
        Piecewise((DiracDelta(0), Eq(x, 5)), (0, True))

        >>> DiracDelta(x**2 - 5).rewrite(Piecewise)
           Piecewise((DiracDelta(0), Eq(x**2, 5)), (0, True))

        >>> DiracDelta(x - 5, 4).rewrite(Piecewise)
        DiracDelta(x - 5, 4)

        """
        if len(args) == 1:
            return Piecewise((DiracDelta(0), Eq(args[0], 0)), (0, True))

    def _eval_rewrite_as_SingularityFunction(self, *args, **kwargs):
        """
        Returns the DiracDelta expression written in the form of Singularity
        Functions.

        """
        from sympy.solvers import solve
        from sympy.functions.special.singularity_functions import SingularityFunction
        if self == DiracDelta(0):
            return SingularityFunction(0, 0, -1)
        if self == DiracDelta(0, 1):
            return SingularityFunction(0, 0, -2)
        free = self.free_symbols
        if len(free) == 1:
            x = (free.pop())
            if len(args) == 1:
                return SingularityFunction(x, solve(args[0], x)[0], -1)
            return SingularityFunction(x, solve(args[0], x)[0], -args[1] - 1)
        else:
            # I don't know how to handle the case for DiracDelta expressions
            # having arguments with more than one variable.
            raise TypeError(filldedent('''
                rewrite(SingularityFunction) does not support
                arguments with more that one variable.'''))


###############################################################################
############################## HEAVISIDE FUNCTION #############################
###############################################################################


class Heaviside(DefinedFunction):
    r"""
    Heaviside step function.

    Explanation
    ===========

    The Heaviside step function has the following properties:

    1) $\frac{d}{d x} \theta(x) = \delta(x)$
    2) $\theta(x) = \begin{cases} 0 & \text{for}\: x < 0 \\ \frac{1}{2} &
       \text{for}\: x = 0 \\1 & \text{for}\: x > 0 \end{cases}$
    3) $\frac{d}{d x} \max(x, 0) = \theta(x)$

    Heaviside(x) is printed as $\theta(x)$ with the SymPy LaTeX printer.

    The value at 0 is set differently in different fields. SymPy uses 1/2,
    which is a convention from electronics and signal processing, and is
    consistent with solving improper integrals by Fourier transform and
    convolution.

    To specify a different value of Heaviside at ``x=0``, a second argument
    can be given. Using ``Heaviside(x, nan)`` gives an expression that will
    evaluate to nan for x=0.

    .. versionchanged:: 1.9 ``Heaviside(0)`` now returns 1/2 (before: undefined)

    Examples
    ========

    >>> from sympy import Heaviside, nan
    >>> from sympy.abc import x
    >>> Heaviside(9)
    1
    >>> Heaviside(-9)
    0
    >>> Heaviside(0)
    1/2
    >>> Heaviside(0, nan)
    nan
    >>> (Heaviside(x) + 1).replace(Heaviside(x), Heaviside(x, 1))
    Heaviside(x, 1) + 1

    See Also
    ========

    DiracDelta

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HeavisideStepFunction.html
    .. [2] https://dlmf.nist.gov/1.16#iv

    """

    is_real = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of a Heaviside Function.

        Examples
        ========

        >>> from sympy import Heaviside, diff
        >>> from sympy.abc import x

        >>> Heaviside(x).fdiff()
        DiracDelta(x)

        >>> Heaviside(x**2 - 1).fdiff()
        DiracDelta(x**2 - 1)

        >>> diff(Heaviside(x)).fdiff()
        DiracDelta(x, 1)

        Parameters
        ==========

        argindex : integer
            order of derivative

        """
        if argindex == 1:
            return DiracDelta(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def __new__(cls, arg, H0=S.Half, **options):
        if isinstance(H0, Heaviside) and len(H0.args) == 1:
            H0 = S.Half
        return super(cls, cls).__new__(cls, arg, H0, **options)

    @property
    def pargs(self):
        """Args without default S.Half"""
        args = self.args
        if args[1] is S.Half:
            args = args[:1]
        return args

    @classmethod
    def eval(cls, arg, H0=S.Half):
        """
        Returns a simplified form or a value of Heaviside depending on the
        argument passed by the Heaviside object.

        Explanation
        ===========

        The ``eval()`` method is automatically called when the ``Heaviside``
        class is about to be instantiated and it returns either some simplified
        instance or the unevaluated instance depending on the argument passed.
        In other words, ``eval()`` method is not needed to be called explicitly,
        it is being called and evaluated once the object is called.

        Examples
        ========

        >>> from sympy import Heaviside, S
        >>> from sympy.abc import x

        >>> Heaviside(x)
        Heaviside(x)

        >>> Heaviside(19)
        1

        >>> Heaviside(0)
        1/2

        >>> Heaviside(0, 1)
        1

        >>> Heaviside(-5)
        0

        >>> Heaviside(S.NaN)
        nan

        >>> Heaviside(x - 100).subs(x, 5)
        0

        >>> Heaviside(x - 100).subs(x, 105)
        1

        Parameters
        ==========

        arg : argument passed by Heaviside object

        H0 : value of Heaviside(0)

        """
        if arg.is_extended_negative:
            return S.Zero
        elif arg.is_extended_positive:
            return S.One
        elif arg.is_zero:
            return H0
        elif arg is S.NaN:
            return S.NaN
        elif fuzzy_not(im(arg).is_zero):
            raise ValueError("Function defined only for Real Values. Complex part: %s  found in %s ." % (repr(im(arg)), repr(arg)) )

    def _eval_rewrite_as_Piecewise(self, arg, H0=None, **kwargs):
        """
        Represents Heaviside in a Piecewise form.

        Examples
        ========

        >>> from sympy import Heaviside, Piecewise, Symbol, nan
        >>> x = Symbol('x')

        >>> Heaviside(x).rewrite(Piecewise)
        Piecewise((0, x < 0), (1/2, Eq(x, 0)), (1, True))

        >>> Heaviside(x,nan).rewrite(Piecewise)
        Piecewise((0, x < 0), (nan, Eq(x, 0)), (1, True))

        >>> Heaviside(x - 5).rewrite(Piecewise)
        Piecewise((0, x < 5), (1/2, Eq(x, 5)), (1, True))

        >>> Heaviside(x**2 - 1).rewrite(Piecewise)
        Piecewise((0, x**2 < 1), (1/2, Eq(x**2, 1)), (1, True))

        """
        if H0 == 0:
            return Piecewise((0, arg <= 0), (1, True))
        if H0 == 1:
            return Piecewise((0, arg < 0), (1, True))
        return Piecewise((0, arg < 0), (H0, Eq(arg, 0)), (1, True))

    def _eval_rewrite_as_sign(self, arg, H0=S.Half, **kwargs):
        """
        Represents the Heaviside function in the form of sign function.

        Explanation
        ===========

        The value of Heaviside(0) must be 1/2 for rewriting as sign to be
        strictly equivalent. For easier usage, we also allow this rewriting
        when Heaviside(0) is undefined.

        Examples
        ========

        >>> from sympy import Heaviside, Symbol, sign, nan
        >>> x = Symbol('x', real=True)
        >>> y = Symbol('y')

        >>> Heaviside(x).rewrite(sign)
        sign(x)/2 + 1/2

        >>> Heaviside(x, 0).rewrite(sign)
        Piecewise((sign(x)/2 + 1/2, Ne(x, 0)), (0, True))

        >>> Heaviside(x, nan).rewrite(sign)
        Piecewise((sign(x)/2 + 1/2, Ne(x, 0)), (nan, True))

        >>> Heaviside(x - 2).rewrite(sign)
        sign(x - 2)/2 + 1/2

        >>> Heaviside(x**2 - 2*x + 1).rewrite(sign)
        sign(x**2 - 2*x + 1)/2 + 1/2

        >>> Heaviside(y).rewrite(sign)
        Heaviside(y)

        >>> Heaviside(y**2 - 2*y + 1).rewrite(sign)
        Heaviside(y**2 - 2*y + 1)

        See Also
        ========

        sign

        """
        if arg.is_extended_real:
            pw1 = Piecewise(
                ((sign(arg) + 1)/2, Ne(arg, 0)),
                (Heaviside(0, H0=H0), True))
            pw2 = Piecewise(
                ((sign(arg) + 1)/2, Eq(Heaviside(0, H0=H0), S.Half)),
                (pw1, True))
            return pw2

    def _eval_rewrite_as_SingularityFunction(self, args, H0=S.Half, **kwargs):
        """
        Returns the Heaviside expression written in the form of Singularity
        Functions.

        """
        from sympy.solvers import solve
        from sympy.functions.special.singularity_functions import SingularityFunction
        if self == Heaviside(0):
            return SingularityFunction(0, 0, 0)
        free = self.free_symbols
        if len(free) == 1:
            x = (free.pop())
            return SingularityFunction(x, solve(args, x)[0], 0)
            # TODO
            # ((x - 5)**3*Heaviside(x - 5)).rewrite(SingularityFunction) should output
            # SingularityFunction(x, 5, 0) instead of (x - 5)**3*SingularityFunction(x, 5, 0)
        else:
            # I don't know how to handle the case for Heaviside expressions
            # having arguments with more than one variable.
            raise TypeError(filldedent('''
                rewrite(SingularityFunction) does not
                support arguments with more that one variable.'''))
