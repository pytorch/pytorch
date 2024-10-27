from ..libmp.backend import xrange
from .calculus import defun

try:
    iteritems = dict.iteritems
except AttributeError:
    iteritems = dict.items

#----------------------------------------------------------------------------#
#                                Differentiation                             #
#----------------------------------------------------------------------------#

@defun
def difference(ctx, s, n):
    r"""
    Given a sequence `(s_k)` containing at least `n+1` items, returns the
    `n`-th forward difference,

    .. math ::

        \Delta^n = \sum_{k=0}^{\infty} (-1)^{k+n} {n \choose k} s_k.
    """
    n = int(n)
    d = ctx.zero
    b = (-1) ** (n & 1)
    for k in xrange(n+1):
        d += b * s[k]
        b = (b * (k-n)) // (k+1)
    return d

def hsteps(ctx, f, x, n, prec, **options):
    singular = options.get('singular')
    addprec = options.get('addprec', 10)
    direction = options.get('direction', 0)
    workprec = (prec+2*addprec) * (n+1)
    orig = ctx.prec
    try:
        ctx.prec = workprec
        h = options.get('h')
        if h is None:
            if options.get('relative'):
                hextramag = int(ctx.mag(x))
            else:
                hextramag = 0
            h = ctx.ldexp(1, -prec-addprec-hextramag)
        else:
            h = ctx.convert(h)
        # Directed: steps x, x+h, ... x+n*h
        direction = options.get('direction', 0)
        if direction:
            h *= ctx.sign(direction)
            steps = xrange(n+1)
            norm = h
        # Central: steps x-n*h, x-(n-2)*h ..., x, ..., x+(n-2)*h, x+n*h
        else:
            steps = xrange(-n, n+1, 2)
            norm = (2*h)
        # Perturb
        if singular:
            x += 0.5*h
        values = [f(x+k*h) for k in steps]
        return values, norm, workprec
    finally:
        ctx.prec = orig


@defun
def diff(ctx, f, x, n=1, **options):
    r"""
    Numerically computes the derivative of `f`, `f'(x)`, or generally for
    an integer `n \ge 0`, the `n`-th derivative `f^{(n)}(x)`.
    A few basic examples are::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> diff(lambda x: x**2 + x, 1.0)
        3.0
        >>> diff(lambda x: x**2 + x, 1.0, 2)
        2.0
        >>> diff(lambda x: x**2 + x, 1.0, 3)
        0.0
        >>> nprint([diff(exp, 3, n) for n in range(5)])   # exp'(x) = exp(x)
        [20.0855, 20.0855, 20.0855, 20.0855, 20.0855]

    Even more generally, given a tuple of arguments `(x_1, \ldots, x_k)`
    and order `(n_1, \ldots, n_k)`, the partial derivative
    `f^{(n_1,\ldots,n_k)}(x_1,\ldots,x_k)` is evaluated. For example::

        >>> diff(lambda x,y: 3*x*y + 2*y - x, (0.25, 0.5), (0,1))
        2.75
        >>> diff(lambda x,y: 3*x*y + 2*y - x, (0.25, 0.5), (1,1))
        3.0

    **Options**

    The following optional keyword arguments are recognized:

    ``method``
        Supported methods are ``'step'`` or ``'quad'``: derivatives may be
        computed using either a finite difference with a small step
        size `h` (default), or numerical quadrature.
    ``direction``
        Direction of finite difference: can be -1 for a left
        difference, 0 for a central difference (default), or +1
        for a right difference; more generally can be any complex number.
    ``addprec``
        Extra precision for `h` used to account for the function's
        sensitivity to perturbations (default = 10).
    ``relative``
        Choose `h` relative to the magnitude of `x`, rather than an
        absolute value; useful for large or tiny `x` (default = False).
    ``h``
        As an alternative to ``addprec`` and ``relative``, manually
        select the step size `h`.
    ``singular``
        If True, evaluation exactly at the point `x` is avoided; this is
        useful for differentiating functions with removable singularities.
        Default = False.
    ``radius``
        Radius of integration contour (with ``method = 'quad'``).
        Default = 0.25. A larger radius typically is faster and more
        accurate, but it must be chosen so that `f` has no
        singularities within the radius from the evaluation point.

    A finite difference requires `n+1` function evaluations and must be
    performed at `(n+1)` times the target precision. Accordingly, `f` must
    support fast evaluation at high precision.

    With integration, a larger number of function evaluations is
    required, but not much extra precision is required. For high order
    derivatives, this method may thus be faster if f is very expensive to
    evaluate at high precision.

    **Further examples**

    The direction option is useful for computing left- or right-sided
    derivatives of nonsmooth functions::

        >>> diff(abs, 0, direction=0)
        0.0
        >>> diff(abs, 0, direction=1)
        1.0
        >>> diff(abs, 0, direction=-1)
        -1.0

    More generally, if the direction is nonzero, a right difference
    is computed where the step size is multiplied by sign(direction).
    For example, with direction=+j, the derivative from the positive
    imaginary direction will be computed::

        >>> diff(abs, 0, direction=j)
        (0.0 - 1.0j)

    With integration, the result may have a small imaginary part
    even even if the result is purely real::

        >>> diff(sqrt, 1, method='quad')    # doctest:+ELLIPSIS
        (0.5 - 4.59...e-26j)
        >>> chop(_)
        0.5

    Adding precision to obtain an accurate value::

        >>> diff(cos, 1e-30)
        0.0
        >>> diff(cos, 1e-30, h=0.0001)
        -9.99999998328279e-31
        >>> diff(cos, 1e-30, addprec=100)
        -1.0e-30

    """
    partial = False
    try:
        orders = list(n)
        x = list(x)
        partial = True
    except TypeError:
        pass
    if partial:
        x = [ctx.convert(_) for _ in x]
        return _partial_diff(ctx, f, x, orders, options)
    method = options.get('method', 'step')
    if n == 0 and method != 'quad' and not options.get('singular'):
        return f(ctx.convert(x))
    prec = ctx.prec
    try:
        if method == 'step':
            values, norm, workprec = hsteps(ctx, f, x, n, prec, **options)
            ctx.prec = workprec
            v = ctx.difference(values, n) / norm**n
        elif method == 'quad':
            ctx.prec += 10
            radius = ctx.convert(options.get('radius', 0.25))
            def g(t):
                rei = radius*ctx.expj(t)
                z = x + rei
                return f(z) / rei**n
            d = ctx.quadts(g, [0, 2*ctx.pi])
            v = d * ctx.factorial(n) / (2*ctx.pi)
        else:
            raise ValueError("unknown method: %r" % method)
    finally:
        ctx.prec = prec
    return +v

def _partial_diff(ctx, f, xs, orders, options):
    if not orders:
        return f()
    if not sum(orders):
        return f(*xs)
    i = 0
    for i in range(len(orders)):
        if orders[i]:
            break
    order = orders[i]
    def fdiff_inner(*f_args):
        def inner(t):
            return f(*(f_args[:i] + (t,) + f_args[i+1:]))
        return ctx.diff(inner, f_args[i], order, **options)
    orders[i] = 0
    return _partial_diff(ctx, fdiff_inner, xs, orders, options)

@defun
def diffs(ctx, f, x, n=None, **options):
    r"""
    Returns a generator that yields the sequence of derivatives

    .. math ::

        f(x), f'(x), f''(x), \ldots, f^{(k)}(x), \ldots

    With ``method='step'``, :func:`~mpmath.diffs` uses only `O(k)`
    function evaluations to generate the first `k` derivatives,
    rather than the roughly `O(k^2)` evaluations
    required if one calls :func:`~mpmath.diff` `k` separate times.

    With `n < \infty`, the generator stops as soon as the
    `n`-th derivative has been generated. If the exact number of
    needed derivatives is known in advance, this is further
    slightly more efficient.

    Options are the same as for :func:`~mpmath.diff`.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15
        >>> nprint(list(diffs(cos, 1, 5)))
        [0.540302, -0.841471, -0.540302, 0.841471, 0.540302, -0.841471]
        >>> for i, d in zip(range(6), diffs(cos, 1)):
        ...     print("%s %s" % (i, d))
        ...
        0 0.54030230586814
        1 -0.841470984807897
        2 -0.54030230586814
        3 0.841470984807897
        4 0.54030230586814
        5 -0.841470984807897

    """
    if n is None:
        n = ctx.inf
    else:
        n = int(n)
    if options.get('method', 'step') != 'step':
        k = 0
        while k < n + 1:
            yield ctx.diff(f, x, k, **options)
            k += 1
        return
    singular = options.get('singular')
    if singular:
        yield ctx.diff(f, x, 0, singular=True)
    else:
        yield f(ctx.convert(x))
    if n < 1:
        return
    if n == ctx.inf:
        A, B = 1, 2
    else:
        A, B = 1, n+1
    while 1:
        callprec = ctx.prec
        y, norm, workprec = hsteps(ctx, f, x, B, callprec, **options)
        for k in xrange(A, B):
            try:
                ctx.prec = workprec
                d = ctx.difference(y, k) / norm**k
            finally:
                ctx.prec = callprec
            yield +d
            if k >= n:
                return
        A, B = B, int(A*1.4+1)
        B = min(B, n)

def iterable_to_function(gen):
    gen = iter(gen)
    data = []
    def f(k):
        for i in xrange(len(data), k+1):
            data.append(next(gen))
        return data[k]
    return f

@defun
def diffs_prod(ctx, factors):
    r"""
    Given a list of `N` iterables or generators yielding
    `f_k(x), f'_k(x), f''_k(x), \ldots` for `k = 1, \ldots, N`,
    generate `g(x), g'(x), g''(x), \ldots` where
    `g(x) = f_1(x) f_2(x) \cdots f_N(x)`.

    At high precision and for large orders, this is typically more efficient
    than numerical differentiation if the derivatives of each `f_k(x)`
    admit direct computation.

    Note: This function does not increase the working precision internally,
    so guard digits may have to be added externally for full accuracy.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> f = lambda x: exp(x)*cos(x)*sin(x)
        >>> u = diffs(f, 1)
        >>> v = mp.diffs_prod([diffs(exp,1), diffs(cos,1), diffs(sin,1)])
        >>> next(u); next(v)
        1.23586333600241
        1.23586333600241
        >>> next(u); next(v)
        0.104658952245596
        0.104658952245596
        >>> next(u); next(v)
        -5.96999877552086
        -5.96999877552086
        >>> next(u); next(v)
        -12.4632923122697
        -12.4632923122697

    """
    N = len(factors)
    if N == 1:
        for c in factors[0]:
            yield c
    else:
        u = iterable_to_function(ctx.diffs_prod(factors[:N//2]))
        v = iterable_to_function(ctx.diffs_prod(factors[N//2:]))
        n = 0
        while 1:
            #yield sum(binomial(n,k)*u(n-k)*v(k) for k in xrange(n+1))
            s = u(n) * v(0)
            a = 1
            for k in xrange(1,n+1):
                a = a * (n-k+1) // k
                s += a * u(n-k) * v(k)
            yield s
            n += 1

def dpoly(n, _cache={}):
    """
    nth differentiation polynomial for exp (Faa di Bruno's formula).

    TODO: most exponents are zero, so maybe a sparse representation
    would be better.
    """
    if n in _cache:
        return _cache[n]
    if not _cache:
        _cache[0] = {(0,):1}
    R = dpoly(n-1)
    R = dict((c+(0,),v) for (c,v) in iteritems(R))
    Ra = {}
    for powers, count in iteritems(R):
        powers1 = (powers[0]+1,) + powers[1:]
        if powers1 in Ra:
            Ra[powers1] += count
        else:
            Ra[powers1] = count
    for powers, count in iteritems(R):
        if not sum(powers):
            continue
        for k,p in enumerate(powers):
            if p:
                powers2 = powers[:k] + (p-1,powers[k+1]+1) + powers[k+2:]
                if powers2 in Ra:
                    Ra[powers2] += p*count
                else:
                    Ra[powers2] = p*count
    _cache[n] = Ra
    return _cache[n]

@defun
def diffs_exp(ctx, fdiffs):
    r"""
    Given an iterable or generator yielding `f(x), f'(x), f''(x), \ldots`
    generate `g(x), g'(x), g''(x), \ldots` where `g(x) = \exp(f(x))`.

    At high precision and for large orders, this is typically more efficient
    than numerical differentiation if the derivatives of `f(x)`
    admit direct computation.

    Note: This function does not increase the working precision internally,
    so guard digits may have to be added externally for full accuracy.

    **Examples**

    The derivatives of the gamma function can be computed using
    logarithmic differentiation::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>>
        >>> def diffs_loggamma(x):
        ...     yield loggamma(x)
        ...     i = 0
        ...     while 1:
        ...         yield psi(i,x)
        ...         i += 1
        ...
        >>> u = diffs_exp(diffs_loggamma(3))
        >>> v = diffs(gamma, 3)
        >>> next(u); next(v)
        2.0
        2.0
        >>> next(u); next(v)
        1.84556867019693
        1.84556867019693
        >>> next(u); next(v)
        2.49292999190269
        2.49292999190269
        >>> next(u); next(v)
        3.44996501352367
        3.44996501352367

    """
    fn = iterable_to_function(fdiffs)
    f0 = ctx.exp(fn(0))
    yield f0
    i = 1
    while 1:
        s = ctx.mpf(0)
        for powers, c in iteritems(dpoly(i)):
            s += c*ctx.fprod(fn(k+1)**p for (k,p) in enumerate(powers) if p)
        yield s * f0
        i += 1

@defun
def differint(ctx, f, x, n=1, x0=0):
    r"""
    Calculates the Riemann-Liouville differintegral, or fractional
    derivative, defined by

    .. math ::

        \,_{x_0}{\mathbb{D}}^n_xf(x) = \frac{1}{\Gamma(m-n)} \frac{d^m}{dx^m}
        \int_{x_0}^{x}(x-t)^{m-n-1}f(t)dt

    where `f` is a given (presumably well-behaved) function,
    `x` is the evaluation point, `n` is the order, and `x_0` is
    the reference point of integration (`m` is an arbitrary
    parameter selected automatically).

    With `n = 1`, this is just the standard derivative `f'(x)`; with `n = 2`,
    the second derivative `f''(x)`, etc. With `n = -1`, it gives
    `\int_{x_0}^x f(t) dt`, with `n = -2`
    it gives `\int_{x_0}^x \left( \int_{x_0}^t f(u) du \right) dt`, etc.

    As `n` is permitted to be any number, this operator generalizes
    iterated differentiation and iterated integration to a single
    operator with a continuous order parameter.

    **Examples**

    There is an exact formula for the fractional derivative of a
    monomial `x^p`, which may be used as a reference. For example,
    the following gives a half-derivative (order 0.5)::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> x = mpf(3); p = 2; n = 0.5
        >>> differint(lambda t: t**p, x, n)
        7.81764019044672
        >>> gamma(p+1)/gamma(p-n+1) * x**(p-n)
        7.81764019044672

    Another useful test function is the exponential function, whose
    integration / differentiation formula easy generalizes
    to arbitrary order. Here we first compute a third derivative,
    and then a triply nested integral. (The reference point `x_0`
    is set to `-\infty` to avoid nonzero endpoint terms.)::

        >>> differint(lambda x: exp(pi*x), -1.5, 3)
        0.278538406900792
        >>> exp(pi*-1.5) * pi**3
        0.278538406900792
        >>> differint(lambda x: exp(pi*x), 3.5, -3, -inf)
        1922.50563031149
        >>> exp(pi*3.5) / pi**3
        1922.50563031149

    However, for noninteger `n`, the differentiation formula for the
    exponential function must be modified to give the same result as the
    Riemann-Liouville differintegral::

        >>> x = mpf(3.5)
        >>> c = pi
        >>> n = 1+2*j
        >>> differint(lambda x: exp(c*x), x, n)
        (-123295.005390743 + 140955.117867654j)
        >>> x**(-n) * exp(c)**x * (x*c)**n * gammainc(-n, 0, x*c) / gamma(-n)
        (-123295.005390743 + 140955.117867654j)


    """
    m = max(int(ctx.ceil(ctx.re(n)))+1, 1)
    r = m-n-1
    g = lambda x: ctx.quad(lambda t: (x-t)**r * f(t), [x0, x])
    return ctx.diff(g, x, m) / ctx.gamma(m-n)

@defun
def diffun(ctx, f, n=1, **options):
    r"""
    Given a function `f`, returns a function `g(x)` that evaluates the nth
    derivative `f^{(n)}(x)`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> cos2 = diffun(sin)
        >>> sin2 = diffun(sin, 4)
        >>> cos(1.3), cos2(1.3)
        (0.267498828624587, 0.267498828624587)
        >>> sin(1.3), sin2(1.3)
        (0.963558185417193, 0.963558185417193)

    The function `f` must support arbitrary precision evaluation.
    See :func:`~mpmath.diff` for additional details and supported
    keyword options.
    """
    if n == 0:
        return f
    def g(x):
        return ctx.diff(f, x, n, **options)
    return g

@defun
def taylor(ctx, f, x, n, **options):
    r"""
    Produces a degree-`n` Taylor polynomial around the point `x` of the
    given function `f`. The coefficients are returned as a list.

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nprint(chop(taylor(sin, 0, 5)))
        [0.0, 1.0, 0.0, -0.166667, 0.0, 0.00833333]

    The coefficients are computed using high-order numerical
    differentiation. The function must be possible to evaluate
    to arbitrary precision. See :func:`~mpmath.diff` for additional details
    and supported keyword options.

    Note that to evaluate the Taylor polynomial as an approximation
    of `f`, e.g. with :func:`~mpmath.polyval`, the coefficients must be reversed,
    and the point of the Taylor expansion must be subtracted from
    the argument:

        >>> p = taylor(exp, 2.0, 10)
        >>> polyval(p[::-1], 2.5 - 2.0)
        12.1824939606092
        >>> exp(2.5)
        12.1824939607035

    """
    gen = enumerate(ctx.diffs(f, x, n, **options))
    if options.get("chop", True):
        return [ctx.chop(d)/ctx.factorial(i) for i, d in gen]
    else:
        return [d/ctx.factorial(i) for i, d in gen]

@defun
def pade(ctx, a, L, M):
    r"""
    Computes a Pade approximation of degree `(L, M)` to a function.
    Given at least `L+M+1` Taylor coefficients `a` approximating
    a function `A(x)`, :func:`~mpmath.pade` returns coefficients of
    polynomials `P, Q` satisfying

    .. math ::

        P = \sum_{k=0}^L p_k x^k

        Q = \sum_{k=0}^M q_k x^k

        Q_0 = 1

        A(x) Q(x) = P(x) + O(x^{L+M+1})

    `P(x)/Q(x)` can provide a good approximation to an analytic function
    beyond the radius of convergence of its Taylor series (example
    from G.A. Baker 'Essentials of Pade Approximants' Academic Press,
    Ch.1A)::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> one = mpf(1)
        >>> def f(x):
        ...     return sqrt((one + 2*x)/(one + x))
        ...
        >>> a = taylor(f, 0, 6)
        >>> p, q = pade(a, 3, 3)
        >>> x = 10
        >>> polyval(p[::-1], x)/polyval(q[::-1], x)
        1.38169105566806
        >>> f(x)
        1.38169855941551

    """
    # To determine L+1 coefficients of P and M coefficients of Q
    # L+M+1 coefficients of A must be provided
    if len(a) < L+M+1:
        raise ValueError("L+M+1 Coefficients should be provided")

    if M == 0:
        if L == 0:
            return [ctx.one], [ctx.one]
        else:
            return a[:L+1], [ctx.one]

    # Solve first
    # a[L]*q[1] + ... + a[L-M+1]*q[M] = -a[L+1]
    # ...
    # a[L+M-1]*q[1] + ... + a[L]*q[M] = -a[L+M]
    A = ctx.matrix(M)
    for j in range(M):
        for i in range(min(M, L+j+1)):
            A[j, i] = a[L+j-i]
    v = -ctx.matrix(a[(L+1):(L+M+1)])
    x = ctx.lu_solve(A, v)
    q = [ctx.one] + list(x)
    # compute p
    p = [0]*(L+1)
    for i in range(L+1):
        s = a[i]
        for j in range(1, min(M,i) + 1):
            s += q[j]*a[i-j]
        p[i] = s
    return p, q
