from operator import gt, lt

from .libmp.backend import xrange

from .functions.functions import SpecialFunctions
from .functions.rszeta import RSCache
from .calculus.quadrature import QuadratureMethods
from .calculus.inverselaplace import LaplaceTransformInversionMethods
from .calculus.calculus import CalculusMethods
from .calculus.optimization import OptimizationMethods
from .calculus.odes import ODEMethods
from .matrices.matrices import MatrixMethods
from .matrices.calculus import MatrixCalculusMethods
from .matrices.linalg import LinearAlgebraMethods
from .matrices.eigen import Eigen
from .identification import IdentificationMethods
from .visualization import VisualizationMethods

from . import libmp

class Context(object):
    pass

class StandardBaseContext(Context,
    SpecialFunctions,
    RSCache,
    QuadratureMethods,
    LaplaceTransformInversionMethods,
    CalculusMethods,
    MatrixMethods,
    MatrixCalculusMethods,
    LinearAlgebraMethods,
    Eigen,
    IdentificationMethods,
    OptimizationMethods,
    ODEMethods,
    VisualizationMethods):

    NoConvergence = libmp.NoConvergence
    ComplexResult = libmp.ComplexResult

    def __init__(ctx):
        ctx._aliases = {}
        # Call those that need preinitialization (e.g. for wrappers)
        SpecialFunctions.__init__(ctx)
        RSCache.__init__(ctx)
        QuadratureMethods.__init__(ctx)
        LaplaceTransformInversionMethods.__init__(ctx)
        CalculusMethods.__init__(ctx)
        MatrixMethods.__init__(ctx)

    def _init_aliases(ctx):
        for alias, value in ctx._aliases.items():
            try:
                setattr(ctx, alias, getattr(ctx, value))
            except AttributeError:
                pass

    _fixed_precision = False

    # XXX
    verbose = False

    def warn(ctx, msg):
        print("Warning:", msg)

    def bad_domain(ctx, msg):
        raise ValueError(msg)

    def _re(ctx, x):
        if hasattr(x, "real"):
            return x.real
        return x

    def _im(ctx, x):
        if hasattr(x, "imag"):
            return x.imag
        return ctx.zero

    def _as_points(ctx, x):
        return x

    def fneg(ctx, x, **kwargs):
        return -ctx.convert(x)

    def fadd(ctx, x, y, **kwargs):
        return ctx.convert(x)+ctx.convert(y)

    def fsub(ctx, x, y, **kwargs):
        return ctx.convert(x)-ctx.convert(y)

    def fmul(ctx, x, y, **kwargs):
        return ctx.convert(x)*ctx.convert(y)

    def fdiv(ctx, x, y, **kwargs):
        return ctx.convert(x)/ctx.convert(y)

    def fsum(ctx, args, absolute=False, squared=False):
        if absolute:
            if squared:
                return sum((abs(x)**2 for x in args), ctx.zero)
            return sum((abs(x) for x in args), ctx.zero)
        if squared:
            return sum((x**2 for x in args), ctx.zero)
        return sum(args, ctx.zero)

    def fdot(ctx, xs, ys=None, conjugate=False):
        if ys is not None:
            xs = zip(xs, ys)
        if conjugate:
            cf = ctx.conj
            return sum((x*cf(y) for (x,y) in xs), ctx.zero)
        else:
            return sum((x*y for (x,y) in xs), ctx.zero)

    def fprod(ctx, args):
        prod = ctx.one
        for arg in args:
            prod *= arg
        return prod

    def nprint(ctx, x, n=6, **kwargs):
        """
        Equivalent to ``print(nstr(x, n))``.
        """
        print(ctx.nstr(x, n, **kwargs))

    def chop(ctx, x, tol=None):
        """
        Chops off small real or imaginary parts, or converts
        numbers close to zero to exact zeros. The input can be a
        single number or an iterable::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> chop(5+1e-10j, tol=1e-9)
            mpf('5.0')
            >>> nprint(chop([1.0, 1e-20, 3+1e-18j, -4, 2]))
            [1.0, 0.0, 3.0, -4.0, 2.0]

        The tolerance defaults to ``100*eps``.
        """
        if tol is None:
            tol = 100*ctx.eps
        try:
            x = ctx.convert(x)
            absx = abs(x)
            if abs(x) < tol:
                return ctx.zero
            if ctx._is_complex_type(x):
                #part_tol = min(tol, absx*tol)
                part_tol = max(tol, absx*tol)
                if abs(x.imag) < part_tol:
                    return x.real
                if abs(x.real) < part_tol:
                    return ctx.mpc(0, x.imag)
        except TypeError:
            if isinstance(x, ctx.matrix):
                return x.apply(lambda a: ctx.chop(a, tol))
            if hasattr(x, "__iter__"):
                return [ctx.chop(a, tol) for a in x]
        return x

    def almosteq(ctx, s, t, rel_eps=None, abs_eps=None):
        r"""
        Determine whether the difference between `s` and `t` is smaller
        than a given epsilon, either relatively or absolutely.

        Both a maximum relative difference and a maximum difference
        ('epsilons') may be specified. The absolute difference is
        defined as `|s-t|` and the relative difference is defined
        as `|s-t|/\max(|s|, |t|)`.

        If only one epsilon is given, both are set to the same value.
        If none is given, both epsilons are set to `2^{-p+m}` where
        `p` is the current working precision and `m` is a small
        integer. The default setting typically allows :func:`~mpmath.almosteq`
        to be used to check for mathematical equality
        in the presence of small rounding errors.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> almosteq(3.141592653589793, 3.141592653589790)
            True
            >>> almosteq(3.141592653589793, 3.141592653589700)
            False
            >>> almosteq(3.141592653589793, 3.141592653589700, 1e-10)
            True
            >>> almosteq(1e-20, 2e-20)
            True
            >>> almosteq(1e-20, 2e-20, rel_eps=0, abs_eps=0)
            False

        """
        t = ctx.convert(t)
        if abs_eps is None and rel_eps is None:
            rel_eps = abs_eps = ctx.ldexp(1, -ctx.prec+4)
        if abs_eps is None:
            abs_eps = rel_eps
        elif rel_eps is None:
            rel_eps = abs_eps
        diff = abs(s-t)
        if diff <= abs_eps:
            return True
        abss = abs(s)
        abst = abs(t)
        if abss < abst:
            err = diff/abst
        else:
            err = diff/abss
        return err <= rel_eps

    def arange(ctx, *args):
        r"""
        This is a generalized version of Python's :func:`~mpmath.range` function
        that accepts fractional endpoints and step sizes and
        returns a list of ``mpf`` instances. Like :func:`~mpmath.range`,
        :func:`~mpmath.arange` can be called with 1, 2 or 3 arguments:

        ``arange(b)``
            `[0, 1, 2, \ldots, x]`
        ``arange(a, b)``
            `[a, a+1, a+2, \ldots, x]`
        ``arange(a, b, h)``
            `[a, a+h, a+h, \ldots, x]`

        where `b-1 \le x < b` (in the third case, `b-h \le x < b`).

        Like Python's :func:`~mpmath.range`, the endpoint is not included. To
        produce ranges where the endpoint is included, :func:`~mpmath.linspace`
        is more convenient.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> arange(4)
            [mpf('0.0'), mpf('1.0'), mpf('2.0'), mpf('3.0')]
            >>> arange(1, 2, 0.25)
            [mpf('1.0'), mpf('1.25'), mpf('1.5'), mpf('1.75')]
            >>> arange(1, -1, -0.75)
            [mpf('1.0'), mpf('0.25'), mpf('-0.5')]

        """
        if not len(args) <= 3:
            raise TypeError('arange expected at most 3 arguments, got %i'
                            % len(args))
        if not len(args) >= 1:
            raise TypeError('arange expected at least 1 argument, got %i'
                            % len(args))
        # set default
        a = 0
        dt = 1
        # interpret arguments
        if len(args) == 1:
            b = args[0]
        elif len(args) >= 2:
            a = args[0]
            b = args[1]
        if len(args) == 3:
            dt = args[2]
        a, b, dt = ctx.mpf(a), ctx.mpf(b), ctx.mpf(dt)
        assert a + dt != a, 'dt is too small and would cause an infinite loop'
        # adapt code for sign of dt
        if a > b:
            if dt > 0:
                return []
            op = gt
        else:
            if dt < 0:
                return []
            op = lt
        # create list
        result = []
        i = 0
        t = a
        while 1:
            t = a + dt*i
            i += 1
            if op(t, b):
                result.append(t)
            else:
                break
        return result

    def linspace(ctx, *args, **kwargs):
        """
        ``linspace(a, b, n)`` returns a list of `n` evenly spaced
        samples from `a` to `b`. The syntax ``linspace(mpi(a,b), n)``
        is also valid.

        This function is often more convenient than :func:`~mpmath.arange`
        for partitioning an interval into subintervals, since
        the endpoint is included::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> linspace(1, 4, 4)
            [mpf('1.0'), mpf('2.0'), mpf('3.0'), mpf('4.0')]

        You may also provide the keyword argument ``endpoint=False``::

            >>> linspace(1, 4, 4, endpoint=False)
            [mpf('1.0'), mpf('1.75'), mpf('2.5'), mpf('3.25')]

        """
        if len(args) == 3:
            a = ctx.mpf(args[0])
            b = ctx.mpf(args[1])
            n = int(args[2])
        elif len(args) == 2:
            assert hasattr(args[0], '_mpi_')
            a = args[0].a
            b = args[0].b
            n = int(args[1])
        else:
            raise TypeError('linspace expected 2 or 3 arguments, got %i' \
                            % len(args))
        if n < 1:
            raise ValueError('n must be greater than 0')
        if not 'endpoint' in kwargs or kwargs['endpoint']:
            if n == 1:
                return [ctx.mpf(a)]
            step = (b - a) / ctx.mpf(n - 1)
            y = [i*step + a for i in xrange(n)]
            y[-1] = b
        else:
            step = (b - a) / ctx.mpf(n)
            y = [i*step + a for i in xrange(n)]
        return y

    def cos_sin(ctx, z, **kwargs):
        return ctx.cos(z, **kwargs), ctx.sin(z, **kwargs)

    def cospi_sinpi(ctx, z, **kwargs):
        return ctx.cospi(z, **kwargs), ctx.sinpi(z, **kwargs)

    def _default_hyper_maxprec(ctx, p):
        return int(1000 * p**0.25 + 4*p)

    _gcd = staticmethod(libmp.gcd)
    list_primes = staticmethod(libmp.list_primes)
    isprime = staticmethod(libmp.isprime)
    bernfrac = staticmethod(libmp.bernfrac)
    moebius = staticmethod(libmp.moebius)
    _ifac = staticmethod(libmp.ifac)
    _eulernum = staticmethod(libmp.eulernum)
    _stirling1 = staticmethod(libmp.stirling1)
    _stirling2 = staticmethod(libmp.stirling2)

    def sum_accurately(ctx, terms, check_step=1):
        prec = ctx.prec
        try:
            extraprec = 10
            while 1:
                ctx.prec = prec + extraprec + 5
                max_mag = ctx.ninf
                s = ctx.zero
                k = 0
                for term in terms():
                    s += term
                    if (not k % check_step) and term:
                        term_mag = ctx.mag(term)
                        max_mag = max(max_mag, term_mag)
                        sum_mag = ctx.mag(s)
                        if sum_mag - term_mag > ctx.prec:
                            break
                    k += 1
                cancellation = max_mag - sum_mag
                if cancellation != cancellation:
                    break
                if cancellation < extraprec or ctx._fixed_precision:
                    break
                extraprec += min(ctx.prec, cancellation)
            return s
        finally:
            ctx.prec = prec

    def mul_accurately(ctx, factors, check_step=1):
        prec = ctx.prec
        try:
            extraprec = 10
            while 1:
                ctx.prec = prec + extraprec + 5
                max_mag = ctx.ninf
                one = ctx.one
                s = one
                k = 0
                for factor in factors():
                    s *= factor
                    term = factor - one
                    if (not k % check_step):
                        term_mag = ctx.mag(term)
                        max_mag = max(max_mag, term_mag)
                        sum_mag = ctx.mag(s-one)
                        #if sum_mag - term_mag > ctx.prec:
                        #    break
                        if -term_mag > ctx.prec:
                            break
                    k += 1
                cancellation = max_mag - sum_mag
                if cancellation != cancellation:
                    break
                if cancellation < extraprec or ctx._fixed_precision:
                    break
                extraprec += min(ctx.prec, cancellation)
            return s
        finally:
            ctx.prec = prec

    def power(ctx, x, y):
        r"""Converts `x` and `y` to mpmath numbers and evaluates
        `x^y = \exp(y \log(x))`::

            >>> from mpmath import *
            >>> mp.dps = 30; mp.pretty = True
            >>> power(2, 0.5)
            1.41421356237309504880168872421

        This shows the leading few digits of a large Mersenne prime
        (performing the exact calculation ``2**43112609-1`` and
        displaying the result in Python would be very slow)::

            >>> power(2, 43112609)-1
            3.16470269330255923143453723949e+12978188
        """
        return ctx.convert(x) ** ctx.convert(y)

    def _zeta_int(ctx, n):
        return ctx.zeta(n)

    def maxcalls(ctx, f, N):
        """
        Return a wrapped copy of *f* that raises ``NoConvergence`` when *f*
        has been called more than *N* times::

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> f = maxcalls(sin, 10)
            >>> print(sum(f(n) for n in range(10)))
            1.95520948210738
            >>> f(10) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: maxcalls: function evaluated 10 times

        """
        counter = [0]
        def f_maxcalls_wrapped(*args, **kwargs):
            counter[0] += 1
            if counter[0] > N:
                raise ctx.NoConvergence("maxcalls: function evaluated %i times" % N)
            return f(*args, **kwargs)
        return f_maxcalls_wrapped

    def memoize(ctx, f):
        """
        Return a wrapped copy of *f* that caches computed values, i.e.
        a memoized copy of *f*. Values are only reused if the cached precision
        is equal to or higher than the working precision::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> f = memoize(maxcalls(sin, 1))
            >>> f(2)
            0.909297426825682
            >>> f(2)
            0.909297426825682
            >>> mp.dps = 25
            >>> f(2) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: maxcalls: function evaluated 1 times

        """
        f_cache = {}
        def f_cached(*args, **kwargs):
            if kwargs:
                key = args, tuple(kwargs.items())
            else:
                key = args
            prec = ctx.prec
            if key in f_cache:
                cprec, cvalue = f_cache[key]
                if cprec >= prec:
                    return +cvalue
            value = f(*args, **kwargs)
            f_cache[key] = (prec, value)
            return value
        f_cached.__name__ = f.__name__
        f_cached.__doc__ = f.__doc__
        return f_cached
