"""Fourier Series"""

from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence


__doctest_requires__ = {('fourier_series',): ['matplotlib']}


def fourier_cos_seq(func, limits, n):
    """Returns the cos sequence in a Fourier series"""
    from sympy.integrals import integrate
    x, L = limits[0], limits[2] - limits[1]
    cos_term = cos(2*n*pi*x / L)
    formula = 2 * cos_term * integrate(func * cos_term, limits) / L
    a0 = formula.subs(n, S.Zero) / 2
    return a0, SeqFormula(2 * cos_term * integrate(func * cos_term, limits)
                          / L, (n, 1, oo))


def fourier_sin_seq(func, limits, n):
    """Returns the sin sequence in a Fourier series"""
    from sympy.integrals import integrate
    x, L = limits[0], limits[2] - limits[1]
    sin_term = sin(2*n*pi*x / L)
    return SeqFormula(2 * sin_term * integrate(func * sin_term, limits)
                      / L, (n, 1, oo))


def _process_limits(func, limits):
    """
    Limits should be of the form (x, start, stop).
    x should be a symbol. Both start and stop should be bounded.

    Explanation
    ===========

    * If x is not given, x is determined from func.
    * If limits is None. Limit of the form (x, -pi, pi) is returned.

    Examples
    ========

    >>> from sympy.series.fourier import _process_limits as pari
    >>> from sympy.abc import x
    >>> pari(x**2, (x, -2, 2))
    (x, -2, 2)
    >>> pari(x**2, (-2, 2))
    (x, -2, 2)
    >>> pari(x**2, None)
    (x, -pi, pi)
    """
    def _find_x(func):
        free = func.free_symbols
        if len(free) == 1:
            return free.pop()
        elif not free:
            return Dummy('k')
        else:
            raise ValueError(
                " specify dummy variables for %s. If the function contains"
                " more than one free symbol, a dummy variable should be"
                " supplied explicitly e.g. FourierSeries(m*n**2, (n, -pi, pi))"
                % func)

    x, start, stop = None, None, None
    if limits is None:
        x, start, stop = _find_x(func), -pi, pi
    if is_sequence(limits, Tuple):
        if len(limits) == 3:
            x, start, stop = limits
        elif len(limits) == 2:
            x = _find_x(func)
            start, stop = limits

    if not isinstance(x, Symbol) or start is None or stop is None:
        raise ValueError('Invalid limits given: %s' % str(limits))

    unbounded = [S.NegativeInfinity, S.Infinity]
    if start in unbounded or stop in unbounded:
        raise ValueError("Both the start and end value should be bounded")

    return sympify((x, start, stop))


def finite_check(f, x, L):

    def check_fx(exprs, x):
        return x not in exprs.free_symbols

    def check_sincos(_expr, x, L):
        if isinstance(_expr, (sin, cos)):
            sincos_args = _expr.args[0]

            if sincos_args.match(a*(pi/L)*x + b) is not None:
                return True
            else:
                return False

    from sympy.simplify.fu import TR2, TR1, sincos_to_sum
    _expr = sincos_to_sum(TR2(TR1(f)))
    add_coeff = _expr.as_coeff_add()

    a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k != S.Zero, ])
    b = Wild('b', properties=[lambda k: x not in k.free_symbols, ])

    for s in add_coeff[1]:
        mul_coeffs = s.as_coeff_mul()[1]
        for t in mul_coeffs:
            if not (check_fx(t, x) or check_sincos(t, x, L)):
                return False, f

    return True, _expr


class FourierSeries(SeriesBase):
    r"""Represents Fourier sine/cosine series.

    Explanation
    ===========

    This class only represents a fourier series.
    No computation is performed.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    See Also
    ========

    sympy.series.fourier.fourier_series
    """
    def __new__(cls, *args):
        args = map(sympify, args)
        return Expr.__new__(cls, *args)

    @property
    def function(self):
        return self.args[0]

    @property
    def x(self):
        return self.args[1][0]

    @property
    def period(self):
        return (self.args[1][1], self.args[1][2])

    @property
    def a0(self):
        return self.args[2][0]

    @property
    def an(self):
        return self.args[2][1]

    @property
    def bn(self):
        return self.args[2][2]

    @property
    def interval(self):
        return Interval(0, oo)

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return oo

    @property
    def L(self):
        return abs(self.period[1] - self.period[0]) / 2

    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self

    def truncate(self, n=3):
        """
        Return the first n nonzero terms of the series.

        If ``n`` is None return an iterator.

        Parameters
        ==========

        n : int or None
            Amount of non-zero terms in approximation or None.

        Returns
        =======

        Expr or iterator :
            Approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.truncate(4)
        2*sin(x) - sin(2*x) + 2*sin(3*x)/3 - sin(4*x)/2

        See Also
        ========

        sympy.series.fourier.FourierSeries.sigma_approximation
        """
        if n is None:
            return iter(self)

        terms = []
        for t in self:
            if len(terms) == n:
                break
            if t is not S.Zero:
                terms.append(t)

        return Add(*terms)

    def sigma_approximation(self, n=3):
        r"""
        Return :math:`\sigma`-approximation of Fourier series with respect
        to order n.

        Explanation
        ===========

        Sigma approximation adjusts a Fourier summation to eliminate the Gibbs
        phenomenon which would otherwise occur at discontinuities.
        A sigma-approximated summation for a Fourier series of a T-periodical
        function can be written as

        .. math::
            s(\theta) = \frac{1}{2} a_0 + \sum _{k=1}^{m-1}
            \operatorname{sinc} \Bigl( \frac{k}{m} \Bigr) \cdot
            \left[ a_k \cos \Bigl( \frac{2\pi k}{T} \theta \Bigr)
            + b_k \sin \Bigl( \frac{2\pi k}{T} \theta \Bigr) \right],

        where :math:`a_0, a_k, b_k, k=1,\ldots,{m-1}` are standard Fourier
        series coefficients and
        :math:`\operatorname{sinc} \Bigl( \frac{k}{m} \Bigr)` is a Lanczos
        :math:`\sigma` factor (expressed in terms of normalized
        :math:`\operatorname{sinc}` function).

        Parameters
        ==========

        n : int
            Highest order of the terms taken into account in approximation.

        Returns
        =======

        Expr :
            Sigma approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.sigma_approximation(4)
        2*sin(x)*sinc(pi/4) - 2*sin(2*x)/pi + 2*sin(3*x)*sinc(3*pi/4)/3

        See Also
        ========

        sympy.series.fourier.FourierSeries.truncate

        Notes
        =====

        The behaviour of
        :meth:`~sympy.series.fourier.FourierSeries.sigma_approximation`
        is different from :meth:`~sympy.series.fourier.FourierSeries.truncate`
        - it takes all nonzero terms of degree smaller than n, rather than
        first n nonzero ones.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Gibbs_phenomenon
        .. [2] https://en.wikipedia.org/wiki/Sigma_approximation
        """
        terms = [sinc(pi * i / n) * t for i, t in enumerate(self[:n])
                 if t is not S.Zero]
        return Add(*terms)

    def shift(self, s):
        """
        Shift the function by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x) + s

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shift(1).truncate()
        -4*cos(x) + cos(2*x) + 1 + pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        a0 = self.a0 + s
        sfunc = self.function + s

        return self.func(sfunc, self.args[1], (a0, self.an, self.bn))

    def shiftx(self, s):
        """
        Shift x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x + s)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shiftx(1).truncate()
        -4*cos(x + 1) + cos(2*x + 2) + pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        an = self.an.subs(x, x + s)
        bn = self.bn.subs(x, x + s)
        sfunc = self.function.subs(x, x + s)

        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    def scale(self, s):
        """
        Scale the function by a term independent of x.

        Explanation
        ===========

        f(x) -> s * f(x)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scale(2).truncate()
        -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        an = self.an.coeff_mul(s)
        bn = self.bn.coeff_mul(s)
        a0 = self.a0 * s
        sfunc = self.args[0] * s

        return self.func(sfunc, self.args[1], (a0, an, bn))

    def scalex(self, s):
        """
        Scale x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(s*x)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scalex(2).truncate()
        -4*cos(2*x) + cos(4*x) + pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        an = self.an.subs(x, x * s)
        bn = self.bn.subs(x, x * s)
        sfunc = self.function.subs(x, x * s)

        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        for t in self:
            if t is not S.Zero:
                return t

    def _eval_term(self, pt):
        if pt == 0:
            return self.a0
        return self.an.coeff(pt) + self.bn.coeff(pt)

    def __neg__(self):
        return self.scale(-1)

    def __add__(self, other):
        if isinstance(other, FourierSeries):
            if self.period != other.period:
                raise ValueError("Both the series should have same periods")

            x, y = self.x, other.x
            function = self.function + other.function.subs(y, x)

            if self.x not in function.free_symbols:
                return function

            an = self.an + other.an
            bn = self.bn + other.bn
            a0 = self.a0 + other.a0

            return self.func(function, self.args[1], (a0, an, bn))

        return Add(self, other)

    def __sub__(self, other):
        return self.__add__(-other)


class FiniteFourierSeries(FourierSeries):
    r"""Represents Finite Fourier sine/cosine series.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    Parameters
    ==========

    f : Expr
        Expression for finding fourier_series

    limits : ( x, start, stop)
        x is the independent variable for the expression f
        (start, stop) is the period of the fourier series

    exprs: (a0, an, bn) or Expr
        a0 is the constant term a0 of the fourier series
        an is a dictionary of coefficients of cos terms
         an[k] = coefficient of cos(pi*(k/L)*x)
        bn is a dictionary of coefficients of sin terms
         bn[k] = coefficient of sin(pi*(k/L)*x)

        or exprs can be an expression to be converted to fourier form

    Methods
    =======

    This class is an extension of FourierSeries class.
    Please refer to sympy.series.fourier.FourierSeries for
    further information.

    See Also
    ========

    sympy.series.fourier.FourierSeries
    sympy.series.fourier.fourier_series
    """

    def __new__(cls, f, limits, exprs):
        f = sympify(f)
        limits = sympify(limits)
        exprs = sympify(exprs)

        if not (isinstance(exprs, Tuple) and len(exprs) == 3):  # exprs is not of form (a0, an, bn)
            # Converts the expression to fourier form
            c, e = exprs.as_coeff_add()
            from sympy.simplify.fu import TR10
            rexpr = c + Add(*[TR10(i) for i in e])
            a0, exp_ls = rexpr.expand(trig=False, power_base=False, power_exp=False, log=False).as_coeff_add()

            x = limits[0]
            L = abs(limits[2] - limits[1]) / 2

            a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k is not S.Zero, ])
            b = Wild('b', properties=[lambda k: x not in k.free_symbols, ])

            an = {}
            bn = {}

            # separates the coefficients of sin and cos terms in dictionaries an, and bn
            for p in exp_ls:
                t = p.match(b * cos(a * (pi / L) * x))
                q = p.match(b * sin(a * (pi / L) * x))
                if t:
                    an[t[a]] = t[b] + an.get(t[a], S.Zero)
                elif q:
                    bn[q[a]] = q[b] + bn.get(q[a], S.Zero)
                else:
                    a0 += p

            exprs = Tuple(a0, an, bn)

        return Expr.__new__(cls, f, limits, exprs)

    @property
    def interval(self):
        _length = 1 if self.a0 else 0
        _length += max(set(self.an.keys()).union(set(self.bn.keys()))) + 1
        return Interval(0, _length)

    @property
    def length(self):
        return self.stop - self.start

    def shiftx(self, s):
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        _expr = self.truncate().subs(x, x + s)
        sfunc = self.function.subs(x, x + s)

        return self.func(sfunc, self.args[1], _expr)

    def scale(self, s):
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        _expr = self.truncate() * s
        sfunc = self.function * s

        return self.func(sfunc, self.args[1], _expr)

    def scalex(self, s):
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        _expr = self.truncate().subs(x, x * s)
        sfunc = self.function.subs(x, x * s)

        return self.func(sfunc, self.args[1], _expr)

    def _eval_term(self, pt):
        if pt == 0:
            return self.a0

        _term = self.an.get(pt, S.Zero) * cos(pt * (pi / self.L) * self.x) \
                + self.bn.get(pt, S.Zero) * sin(pt * (pi / self.L) * self.x)
        return _term

    def __add__(self, other):
        if isinstance(other, FourierSeries):
            return other.__add__(fourier_series(self.function, self.args[1],\
                                                finite=False))
        elif isinstance(other, FiniteFourierSeries):
            if self.period != other.period:
                raise ValueError("Both the series should have same periods")

            x, y = self.x, other.x
            function = self.function + other.function.subs(y, x)

            if self.x not in function.free_symbols:
                return function

            return fourier_series(function, limits=self.args[1])


def fourier_series(f, limits=None, finite=True):
    r"""Computes the Fourier trigonometric series expansion.

    Explanation
    ===========

    Fourier trigonometric series of $f(x)$ over the interval $(a, b)$
    is defined as:

    .. math::
        \frac{a_0}{2} + \sum_{n=1}^{\infty}
        (a_n \cos(\frac{2n \pi x}{L}) + b_n \sin(\frac{2n \pi x}{L}))

    where the coefficients are:

    .. math::
        L = b - a

    .. math::
        a_0 = \frac{2}{L} \int_{a}^{b}{f(x) dx}

    .. math::
        a_n = \frac{2}{L} \int_{a}^{b}{f(x) \cos(\frac{2n \pi x}{L}) dx}

    .. math::
        b_n = \frac{2}{L} \int_{a}^{b}{f(x) \sin(\frac{2n \pi x}{L}) dx}

    The condition whether the function $f(x)$ given should be periodic
    or not is more than necessary, because it is sufficient to consider
    the series to be converging to $f(x)$ only in the given interval,
    not throughout the whole real line.

    This also brings a lot of ease for the computation because
    you do not have to make $f(x)$ artificially periodic by
    wrapping it with piecewise, modulo operations,
    but you can shape the function to look like the desired periodic
    function only in the interval $(a, b)$, and the computed series will
    automatically become the series of the periodic version of $f(x)$.

    This property is illustrated in the examples section below.

    Parameters
    ==========

    limits : (sym, start, end), optional
        *sym* denotes the symbol the series is computed with respect to.

        *start* and *end* denotes the start and the end of the interval
        where the fourier series converges to the given function.

        Default range is specified as $-\pi$ and $\pi$.

    Returns
    =======

    FourierSeries
        A symbolic object representing the Fourier trigonometric series.

    Examples
    ========

    Computing the Fourier series of $f(x) = x^2$:

    >>> from sympy import fourier_series, pi
    >>> from sympy.abc import x
    >>> f = x**2
    >>> s = fourier_series(f, (x, -pi, pi))
    >>> s1 = s.truncate(n=3)
    >>> s1
    -4*cos(x) + cos(2*x) + pi**2/3

    Shifting of the Fourier series:

    >>> s.shift(1).truncate()
    -4*cos(x) + cos(2*x) + 1 + pi**2/3
    >>> s.shiftx(1).truncate()
    -4*cos(x + 1) + cos(2*x + 2) + pi**2/3

    Scaling of the Fourier series:

    >>> s.scale(2).truncate()
    -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
    >>> s.scalex(2).truncate()
    -4*cos(2*x) + cos(4*x) + pi**2/3

    Computing the Fourier series of $f(x) = x$:

    This illustrates how truncating to the higher order gives better
    convergence.

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import fourier_series, pi, plot
        >>> from sympy.abc import x
        >>> f = x
        >>> s = fourier_series(f, (x, -pi, pi))
        >>> s1 = s.truncate(n = 3)
        >>> s2 = s.truncate(n = 5)
        >>> s3 = s.truncate(n = 7)
        >>> p = plot(f, s1, s2, s3, (x, -pi, pi), show=False, legend=True)

        >>> p[0].line_color = (0, 0, 0)
        >>> p[0].label = 'x'
        >>> p[1].line_color = (0.7, 0.7, 0.7)
        >>> p[1].label = 'n=3'
        >>> p[2].line_color = (0.5, 0.5, 0.5)
        >>> p[2].label = 'n=5'
        >>> p[3].line_color = (0.3, 0.3, 0.3)
        >>> p[3].label = 'n=7'

        >>> p.show()

    This illustrates how the series converges to different sawtooth
    waves if the different ranges are specified.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> s1 = fourier_series(x, (x, -1, 1)).truncate(10)
        >>> s2 = fourier_series(x, (x, -pi, pi)).truncate(10)
        >>> s3 = fourier_series(x, (x, 0, 1)).truncate(10)
        >>> p = plot(x, s1, s2, s3, (x, -5, 5), show=False, legend=True)

        >>> p[0].line_color = (0, 0, 0)
        >>> p[0].label = 'x'
        >>> p[1].line_color = (0.7, 0.7, 0.7)
        >>> p[1].label = '[-1, 1]'
        >>> p[2].line_color = (0.5, 0.5, 0.5)
        >>> p[2].label = '[-pi, pi]'
        >>> p[3].line_color = (0.3, 0.3, 0.3)
        >>> p[3].label = '[0, 1]'

        >>> p.show()

    Notes
    =====

    Computing Fourier series can be slow
    due to the integration required in computing
    an, bn.

    It is faster to compute Fourier series of a function
    by using shifting and scaling on an already
    computed Fourier series rather than computing
    again.

    e.g. If the Fourier series of ``x**2`` is known
    the Fourier series of ``x**2 - 1`` can be found by shifting by ``-1``.

    See Also
    ========

    sympy.series.fourier.FourierSeries

    References
    ==========

    .. [1] https://mathworld.wolfram.com/FourierSeries.html
    """
    f = sympify(f)

    limits = _process_limits(f, limits)
    x = limits[0]

    if x not in f.free_symbols:
        return f

    if finite:
        L = abs(limits[2] - limits[1]) / 2
        is_finite, res_f = finite_check(f, x, L)
        if is_finite:
            return FiniteFourierSeries(f, limits, res_f)

    n = Dummy('n')
    center = (limits[1] + limits[2]) / 2
    if center.is_zero:
        neg_f = f.subs(x, -x)
        if f == neg_f:
            a0, an = fourier_cos_seq(f, limits, n)
            bn = SeqFormula(0, (1, oo))
            return FourierSeries(f, limits, (a0, an, bn))
        elif f == -neg_f:
            a0 = S.Zero
            an = SeqFormula(0, (1, oo))
            bn = fourier_sin_seq(f, limits, n)
            return FourierSeries(f, limits, (a0, an, bn))
    a0, an = fourier_cos_seq(f, limits, n)
    bn = fourier_sin_seq(f, limits, n)
    return FourierSeries(f, limits, (a0, an, bn))
