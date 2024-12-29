import math

from ..libmp.backend import xrange

class QuadratureRule(object):
    """
    Quadrature rules are implemented using this class, in order to
    simplify the code and provide a common infrastructure
    for tasks such as error estimation and node caching.

    You can implement a custom quadrature rule by subclassing
    :class:`QuadratureRule` and implementing the appropriate
    methods. The subclass can then be used by :func:`~mpmath.quad` by
    passing it as the *method* argument.

    :class:`QuadratureRule` instances are supposed to be singletons.
    :class:`QuadratureRule` therefore implements instance caching
    in :func:`~mpmath.__new__`.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.standard_cache = {}
        self.transformed_cache = {}
        self.interval_count = {}

    def clear(self):
        """
        Delete cached node data.
        """
        self.standard_cache = {}
        self.transformed_cache = {}
        self.interval_count = {}

    def calc_nodes(self, degree, prec, verbose=False):
        r"""
        Compute nodes for the standard interval `[-1, 1]`. Subclasses
        should probably implement only this method, and use
        :func:`~mpmath.get_nodes` method to retrieve the nodes.
        """
        raise NotImplementedError

    def get_nodes(self, a, b, degree, prec, verbose=False):
        """
        Return nodes for given interval, degree and precision. The
        nodes are retrieved from a cache if already computed;
        otherwise they are computed by calling :func:`~mpmath.calc_nodes`
        and are then cached.

        Subclasses should probably not implement this method,
        but just implement :func:`~mpmath.calc_nodes` for the actual
        node computation.
        """
        key = (a, b, degree, prec)
        if key in self.transformed_cache:
            return self.transformed_cache[key]
        orig = self.ctx.prec
        try:
            self.ctx.prec = prec+20
            # Get nodes on standard interval
            if (degree, prec) in self.standard_cache:
                nodes = self.standard_cache[degree, prec]
            else:
                nodes = self.calc_nodes(degree, prec, verbose)
                self.standard_cache[degree, prec] = nodes
            # Transform to general interval
            nodes = self.transform_nodes(nodes, a, b, verbose)
            if key in self.interval_count:
                self.transformed_cache[key] = nodes
            else:
                self.interval_count[key] = True
        finally:
            self.ctx.prec = orig
        return nodes

    def transform_nodes(self, nodes, a, b, verbose=False):
        r"""
        Rescale standardized nodes (for `[-1, 1]`) to a general
        interval `[a, b]`. For a finite interval, a simple linear
        change of variables is used. Otherwise, the following
        transformations are used:

        .. math ::

            \lbrack a, \infty \rbrack : t = \frac{1}{x} + (a-1)

            \lbrack -\infty, b \rbrack : t = (b+1) - \frac{1}{x}

            \lbrack -\infty, \infty \rbrack : t = \frac{x}{\sqrt{1-x^2}}

        """
        ctx = self.ctx
        a = ctx.convert(a)
        b = ctx.convert(b)
        one = ctx.one
        if (a, b) == (-one, one):
            return nodes
        half = ctx.mpf(0.5)
        new_nodes = []
        if ctx.isinf(a) or ctx.isinf(b):
            if (a, b) == (ctx.ninf, ctx.inf):
                p05 = -half
                for x, w in nodes:
                    x2 = x*x
                    px1 = one-x2
                    spx1 = px1**p05
                    x = x*spx1
                    w *= spx1/px1
                    new_nodes.append((x, w))
            elif a == ctx.ninf:
                b1 = b+1
                for x, w in nodes:
                    u = 2/(x+one)
                    x = b1-u
                    w *= half*u**2
                    new_nodes.append((x, w))
            elif b == ctx.inf:
                a1 = a-1
                for x, w in nodes:
                    u = 2/(x+one)
                    x = a1+u
                    w *= half*u**2
                    new_nodes.append((x, w))
            elif a == ctx.inf or b == ctx.ninf:
                return [(x,-w) for (x,w) in self.transform_nodes(nodes, b, a, verbose)]
            else:
                raise NotImplementedError
        else:
            # Simple linear change of variables
            C = (b-a)/2
            D = (b+a)/2
            for x, w in nodes:
                new_nodes.append((D+C*x, C*w))
        return new_nodes

    def guess_degree(self, prec):
        """
        Given a desired precision `p` in bits, estimate the degree `m`
        of the quadrature required to accomplish full accuracy for
        typical integrals. By default, :func:`~mpmath.quad` will perform up
        to `m` iterations. The value of `m` should be a slight
        overestimate, so that "slightly bad" integrals can be dealt
        with automatically using a few extra iterations. On the
        other hand, it should not be too big, so :func:`~mpmath.quad` can
        quit within a reasonable amount of time when it is given
        an "unsolvable" integral.

        The default formula used by :func:`~mpmath.guess_degree` is tuned
        for both :class:`TanhSinh` and :class:`GaussLegendre`.
        The output is roughly as follows:

            +---------+---------+
            | `p`     | `m`     |
            +=========+=========+
            | 50      | 6       |
            +---------+---------+
            | 100     | 7       |
            +---------+---------+
            | 500     | 10      |
            +---------+---------+
            | 3000    | 12      |
            +---------+---------+

        This formula is based purely on a limited amount of
        experimentation and will sometimes be wrong.
        """
        # Expected degree
        # XXX: use mag
        g = int(4 + max(0, self.ctx.log(prec/30.0, 2)))
        # Reasonable "worst case"
        g += 2
        return g

    def estimate_error(self, results, prec, epsilon):
        r"""
        Given results from integrations `[I_1, I_2, \ldots, I_k]` done
        with a quadrature of rule of degree `1, 2, \ldots, k`, estimate
        the error of `I_k`.

        For `k = 2`, we estimate  `|I_{\infty}-I_2|` as `|I_2-I_1|`.

        For `k > 2`, we extrapolate `|I_{\infty}-I_k| \approx |I_{k+1}-I_k|`
        from `|I_k-I_{k-1}|` and `|I_k-I_{k-2}|` under the assumption
        that each degree increment roughly doubles the accuracy of
        the quadrature rule (this is true for both :class:`TanhSinh`
        and :class:`GaussLegendre`). The extrapolation formula is given
        by Borwein, Bailey & Girgensohn. Although not very conservative,
        this method seems to be very robust in practice.
        """
        if len(results) == 2:
            return abs(results[0]-results[1])
        try:
            if results[-1] == results[-2] == results[-3]:
                return self.ctx.zero
            D1 = self.ctx.log(abs(results[-1]-results[-2]), 10)
            D2 = self.ctx.log(abs(results[-1]-results[-3]), 10)
        except ValueError:
            return epsilon
        D3 = -prec
        D4 = min(0, max(D1**2/D2, 2*D1, D3))
        return self.ctx.mpf(10) ** int(D4)

    def summation(self, f, points, prec, epsilon, max_degree, verbose=False):
        """
        Main integration function. Computes the 1D integral over
        the interval specified by *points*. For each subinterval,
        performs quadrature of degree from 1 up to *max_degree*
        until :func:`~mpmath.estimate_error` signals convergence.

        :func:`~mpmath.summation` transforms each subintegration to
        the standard interval and then calls :func:`~mpmath.sum_next`.
        """
        ctx = self.ctx
        I = total_err = ctx.zero
        for i in xrange(len(points)-1):
            a, b = points[i], points[i+1]
            if a == b:
                continue
            # XXX: we could use a single variable transformation,
            # but this is not good in practice. We get better accuracy
            # by having 0 as an endpoint.
            if (a, b) == (ctx.ninf, ctx.inf):
                _f = f
                f = lambda x: _f(-x) + _f(x)
                a, b = (ctx.zero, ctx.inf)
            results = []
            err = ctx.zero
            for degree in xrange(1, max_degree+1):
                nodes = self.get_nodes(a, b, degree, prec, verbose)
                if verbose:
                    print("Integrating from %s to %s (degree %s of %s)" % \
                        (ctx.nstr(a), ctx.nstr(b), degree, max_degree))
                result = self.sum_next(f, nodes, degree, prec, results, verbose)
                results.append(result)
                if degree > 1:
                    err = self.estimate_error(results, prec, epsilon)
                    if verbose:
                        print("Estimated error:", ctx.nstr(err), " epsilon:", ctx.nstr(epsilon), " result: ", ctx.nstr(result))
                    if err <= epsilon:
                        break
            I += results[-1]
            total_err += err
        if total_err > epsilon:
            if verbose:
                print("Failed to reach full accuracy. Estimated error:", ctx.nstr(total_err))
        return I, total_err

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        r"""
        Evaluates the step sum `\sum w_k f(x_k)` where the *nodes* list
        contains the `(w_k, x_k)` pairs.

        :func:`~mpmath.summation` will supply the list *results* of
        values computed by :func:`~mpmath.sum_next` at previous degrees, in
        case the quadrature rule is able to reuse them.
        """
        return self.ctx.fdot((w, f(x)) for (x,w) in nodes)


class TanhSinh(QuadratureRule):
    r"""
    This class implements "tanh-sinh" or "doubly exponential"
    quadrature. This quadrature rule is based on the Euler-Maclaurin
    integral formula. By performing a change of variables involving
    nested exponentials / hyperbolic functions (hence the name), the
    derivatives at the endpoints vanish rapidly. Since the error term
    in the Euler-Maclaurin formula depends on the derivatives at the
    endpoints, a simple step sum becomes extremely accurate. In
    practice, this means that doubling the number of evaluation
    points roughly doubles the number of accurate digits.

    Comparison to Gauss-Legendre:
      * Initial computation of nodes is usually faster
      * Handles endpoint singularities better
      * Handles infinite integration intervals better
      * Is slower for smooth integrands once nodes have been computed

    The implementation of the tanh-sinh algorithm is based on the
    description given in Borwein, Bailey & Girgensohn, "Experimentation
    in Mathematics - Computational Paths to Discovery", A K Peters,
    2003, pages 312-313. In the present implementation, a few
    improvements have been made:

      * A more efficient scheme is used to compute nodes (exploiting
        recurrence for the exponential function)
      * The nodes are computed successively instead of all at once

    **References**

    * [Bailey]_
    * http://users.cs.dal.ca/~jborwein/tanh-sinh.pdf

    """

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        """
        Step sum for tanh-sinh quadrature of degree `m`. We exploit the
        fact that half of the abscissas at degree `m` are precisely the
        abscissas from degree `m-1`. Thus reusing the result from
        the previous level allows a 2x speedup.
        """
        h = self.ctx.mpf(2)**(-degree)
        # Abscissas overlap, so reusing saves half of the time
        if previous:
            S = previous[-1]/(h*2)
        else:
            S = self.ctx.zero
        S += self.ctx.fdot((w,f(x)) for (x,w) in nodes)
        return h*S

    def calc_nodes(self, degree, prec, verbose=False):
        r"""
        The abscissas and weights for tanh-sinh quadrature of degree
        `m` are given by

        .. math::

            x_k = \tanh(\pi/2 \sinh(t_k))

            w_k = \pi/2 \cosh(t_k) / \cosh(\pi/2 \sinh(t_k))^2

        where `t_k = t_0 + hk` for a step length `h \sim 2^{-m}`. The
        list of nodes is actually infinite, but the weights die off so
        rapidly that only a few are needed.
        """
        ctx = self.ctx
        nodes = []

        extra = 20
        ctx.prec += extra
        tol = ctx.ldexp(1, -prec-10)
        pi4 = ctx.pi/4

        # For simplicity, we work in steps h = 1/2^n, with the first point
        # offset so that we can reuse the sum from the previous degree

        # We define degree 1 to include the "degree 0" steps, including
        # the point x = 0. (It doesn't work well otherwise; not sure why.)
        t0 = ctx.ldexp(1, -degree)
        if degree == 1:
            #nodes.append((mpf(0), pi4))
            #nodes.append((-mpf(0), pi4))
            nodes.append((ctx.zero, ctx.pi/2))
            h = t0
        else:
            h = t0*2

        # Since h is fixed, we can compute the next exponential
        # by simply multiplying by exp(h)
        expt0 = ctx.exp(t0)
        a = pi4 * expt0
        b = pi4 / expt0
        udelta = ctx.exp(h)
        urdelta = 1/udelta

        for k in xrange(0, 20*2**degree+1):
            # Reference implementation:
            # t = t0 + k*h
            # x = tanh(pi/2 * sinh(t))
            # w = pi/2 * cosh(t) / cosh(pi/2 * sinh(t))**2

            # Fast implementation. Note that c = exp(pi/2 * sinh(t))
            c = ctx.exp(a-b)
            d = 1/c
            co = (c+d)/2
            si = (c-d)/2
            x = si / co
            w = (a+b) / co**2
            diff = abs(x-1)
            if diff <= tol:
                break

            nodes.append((x, w))
            nodes.append((-x, w))

            a *= udelta
            b *= urdelta

            if verbose and k % 300 == 150:
                # Note: the number displayed is rather arbitrary. Should
                # figure out how to print something that looks more like a
                # percentage
                print("Calculating nodes:", ctx.nstr(-ctx.log(diff, 10) / prec))

        ctx.prec -= extra
        return nodes


class GaussLegendre(QuadratureRule):
    r"""
    This class implements Gauss-Legendre quadrature, which is
    exceptionally efficient for polynomials and polynomial-like (i.e.
    very smooth) integrands.

    The abscissas and weights are given by roots and values of
    Legendre polynomials, which are the orthogonal polynomials
    on `[-1, 1]` with respect to the unit weight
    (see :func:`~mpmath.legendre`).

    In this implementation, we take the "degree" `m` of the quadrature
    to denote a Gauss-Legendre rule of degree `3 \cdot 2^m` (following
    Borwein, Bailey & Girgensohn). This way we get quadratic, rather
    than linear, convergence as the degree is incremented.

    Comparison to tanh-sinh quadrature:
      * Is faster for smooth integrands once nodes have been computed
      * Initial computation of nodes is usually slower
      * Handles endpoint singularities worse
      * Handles infinite integration intervals worse

    """

    def calc_nodes(self, degree, prec, verbose=False):
        r"""
        Calculates the abscissas and weights for Gauss-Legendre
        quadrature of degree of given degree (actually `3 \cdot 2^m`).
        """
        ctx = self.ctx
        # It is important that the epsilon is set lower than the
        # "real" epsilon
        epsilon = ctx.ldexp(1, -prec-8)
        # Fairly high precision might be required for accurate
        # evaluation of the roots
        orig = ctx.prec
        ctx.prec = int(prec*1.5)
        if degree == 1:
            x = ctx.sqrt(ctx.mpf(3)/5)
            w = ctx.mpf(5)/9
            nodes = [(-x,w),(ctx.zero,ctx.mpf(8)/9),(x,w)]
            ctx.prec = orig
            return nodes
        nodes = []
        n = 3*2**(degree-1)
        upto = n//2 + 1
        for j in xrange(1, upto):
            # Asymptotic formula for the roots
            r = ctx.mpf(math.cos(math.pi*(j-0.25)/(n+0.5)))
            # Newton iteration
            while 1:
                t1, t2 = 1, 0
                # Evaluates the Legendre polynomial using its defining
                # recurrence relation
                for j1 in xrange(1,n+1):
                    t3, t2, t1 = t2, t1, ((2*j1-1)*r*t1 - (j1-1)*t2)/j1
                t4 = n*(r*t1-t2)/(r**2-1)
                a = t1/t4
                r = r - a
                if abs(a) < epsilon:
                    break
            x = r
            w = 2/((1-r**2)*t4**2)
            if verbose  and j % 30 == 15:
                print("Computing nodes (%i of %i)" % (j, upto))
            nodes.append((x, w))
            nodes.append((-x, w))
        ctx.prec = orig
        return nodes

class QuadratureMethods(object):

    def __init__(ctx, *args, **kwargs):
        ctx._gauss_legendre = GaussLegendre(ctx)
        ctx._tanh_sinh = TanhSinh(ctx)

    def quad(ctx, f, *points, **kwargs):
        r"""
        Computes a single, double or triple integral over a given
        1D interval, 2D rectangle, or 3D cuboid. A basic example::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> quad(sin, [0, pi])
            2.0

        A basic 2D integral::

            >>> f = lambda x, y: cos(x+y/2)
            >>> quad(f, [-pi/2, pi/2], [0, pi])
            4.0

        **Interval format**

        The integration range for each dimension may be specified
        using a list or tuple. Arguments are interpreted as follows:

        ``quad(f, [x1, x2])`` -- calculates
        `\int_{x_1}^{x_2} f(x) \, dx`

        ``quad(f, [x1, x2], [y1, y2])`` -- calculates
        `\int_{x_1}^{x_2} \int_{y_1}^{y_2} f(x,y) \, dy \, dx`

        ``quad(f, [x1, x2], [y1, y2], [z1, z2])`` -- calculates
        `\int_{x_1}^{x_2} \int_{y_1}^{y_2} \int_{z_1}^{z_2} f(x,y,z)
        \, dz \, dy \, dx`

        Endpoints may be finite or infinite. An interval descriptor
        may also contain more than two points. In this
        case, the integration is split into subintervals, between
        each pair of consecutive points. This is useful for
        dealing with mid-interval discontinuities, or integrating
        over large intervals where the function is irregular or
        oscillates.

        **Options**

        :func:`~mpmath.quad` recognizes the following keyword arguments:

        *method*
            Chooses integration algorithm (described below).
        *error*
            If set to true, :func:`~mpmath.quad` returns `(v, e)` where `v` is the
            integral and `e` is the estimated error.
        *maxdegree*
            Maximum degree of the quadrature rule to try before
            quitting.
        *verbose*
            Print details about progress.

        **Algorithms**

        Mpmath presently implements two integration algorithms: tanh-sinh
        quadrature and Gauss-Legendre quadrature. These can be selected
        using *method='tanh-sinh'* or *method='gauss-legendre'* or by
        passing the classes *method=TanhSinh*, *method=GaussLegendre*.
        The functions :func:`~mpmath.quadts` and :func:`~mpmath.quadgl` are also available
        as shortcuts.

        Both algorithms have the property that doubling the number of
        evaluation points roughly doubles the accuracy, so both are ideal
        for high precision quadrature (hundreds or thousands of digits).

        At high precision, computing the nodes and weights for the
        integration can be expensive (more expensive than computing the
        function values). To make repeated integrations fast, nodes
        are automatically cached.

        The advantages of the tanh-sinh algorithm are that it tends to
        handle endpoint singularities well, and that the nodes are cheap
        to compute on the first run. For these reasons, it is used by
        :func:`~mpmath.quad` as the default algorithm.

        Gauss-Legendre quadrature often requires fewer function
        evaluations, and is therefore often faster for repeated use, but
        the algorithm does not handle endpoint singularities as well and
        the nodes are more expensive to compute. Gauss-Legendre quadrature
        can be a better choice if the integrand is smooth and repeated
        integrations are required (e.g. for multiple integrals).

        See the documentation for :class:`TanhSinh` and
        :class:`GaussLegendre` for additional details.

        **Examples of 1D integrals**

        Intervals may be infinite or half-infinite. The following two
        examples evaluate the limits of the inverse tangent function
        (`\int 1/(1+x^2) = \tan^{-1} x`), and the Gaussian integral
        `\int_{\infty}^{\infty} \exp(-x^2)\,dx = \sqrt{\pi}`::

            >>> mp.dps = 15
            >>> quad(lambda x: 2/(x**2+1), [0, inf])
            3.14159265358979
            >>> quad(lambda x: exp(-x**2), [-inf, inf])**2
            3.14159265358979

        Integrals can typically be resolved to high precision.
        The following computes 50 digits of `\pi` by integrating the
        area of the half-circle defined by `x^2 + y^2 \le 1`,
        `-1 \le x \le 1`, `y \ge 0`::

            >>> mp.dps = 50
            >>> 2*quad(lambda x: sqrt(1-x**2), [-1, 1])
            3.1415926535897932384626433832795028841971693993751

        One can just as well compute 1000 digits (output truncated)::

            >>> mp.dps = 1000
            >>> 2*quad(lambda x: sqrt(1-x**2), [-1, 1])  #doctest:+ELLIPSIS
            3.141592653589793238462643383279502884...216420199

        Complex integrals are supported. The following computes
        a residue at `z = 0` by integrating counterclockwise along the
        diamond-shaped path from `1` to `+i` to `-1` to `-i` to `1`::

            >>> mp.dps = 15
            >>> chop(quad(lambda z: 1/z, [1,j,-1,-j,1]))
            (0.0 + 6.28318530717959j)

        **Examples of 2D and 3D integrals**

        Here are several nice examples of analytically solvable
        2D integrals (taken from MathWorld [1]) that can be evaluated
        to high precision fairly rapidly by :func:`~mpmath.quad`::

            >>> mp.dps = 30
            >>> f = lambda x, y: (x-1)/((1-x*y)*log(x*y))
            >>> quad(f, [0, 1], [0, 1])
            0.577215664901532860606512090082
            >>> +euler
            0.577215664901532860606512090082

            >>> f = lambda x, y: 1/sqrt(1+x**2+y**2)
            >>> quad(f, [-1, 1], [-1, 1])
            3.17343648530607134219175646705
            >>> 4*log(2+sqrt(3))-2*pi/3
            3.17343648530607134219175646705

            >>> f = lambda x, y: 1/(1-x**2 * y**2)
            >>> quad(f, [0, 1], [0, 1])
            1.23370055013616982735431137498
            >>> pi**2 / 8
            1.23370055013616982735431137498

            >>> quad(lambda x, y: 1/(1-x*y), [0, 1], [0, 1])
            1.64493406684822643647241516665
            >>> pi**2 / 6
            1.64493406684822643647241516665

        Multiple integrals may be done over infinite ranges::

            >>> mp.dps = 15
            >>> print(quad(lambda x,y: exp(-x-y), [0, inf], [1, inf]))
            0.367879441171442
            >>> print(1/e)
            0.367879441171442

        For nonrectangular areas, one can call :func:`~mpmath.quad` recursively.
        For example, we can replicate the earlier example of calculating
        `\pi` by integrating over the unit-circle, and actually use double
        quadrature to actually measure the area circle::

            >>> f = lambda x: quad(lambda y: 1, [-sqrt(1-x**2), sqrt(1-x**2)])
            >>> quad(f, [-1, 1])
            3.14159265358979

        Here is a simple triple integral::

            >>> mp.dps = 15
            >>> f = lambda x,y,z: x*y/(1+z)
            >>> quad(f, [0,1], [0,1], [1,2], method='gauss-legendre')
            0.101366277027041
            >>> (log(3)-log(2))/4
            0.101366277027041

        **Singularities**

        Both tanh-sinh and Gauss-Legendre quadrature are designed to
        integrate smooth (infinitely differentiable) functions. Neither
        algorithm copes well with mid-interval singularities (such as
        mid-interval discontinuities in `f(x)` or `f'(x)`).
        The best solution is to split the integral into parts::

            >>> mp.dps = 15
            >>> quad(lambda x: abs(sin(x)), [0, 2*pi])   # Bad
            3.99900894176779
            >>> quad(lambda x: abs(sin(x)), [0, pi, 2*pi])  # Good
            4.0

        The tanh-sinh rule often works well for integrands having a
        singularity at one or both endpoints::

            >>> mp.dps = 15
            >>> quad(log, [0, 1], method='tanh-sinh')  # Good
            -1.0
            >>> quad(log, [0, 1], method='gauss-legendre')  # Bad
            -0.999932197413801

        However, the result may still be inaccurate for some functions::

            >>> quad(lambda x: 1/sqrt(x), [0, 1], method='tanh-sinh')
            1.99999999946942

        This problem is not due to the quadrature rule per se, but to
        numerical amplification of errors in the nodes. The problem can be
        circumvented by temporarily increasing the precision::

            >>> mp.dps = 30
            >>> a = quad(lambda x: 1/sqrt(x), [0, 1], method='tanh-sinh')
            >>> mp.dps = 15
            >>> +a
            2.0

        **Highly variable functions**

        For functions that are smooth (in the sense of being infinitely
        differentiable) but contain sharp mid-interval peaks or many
        "bumps", :func:`~mpmath.quad` may fail to provide full accuracy. For
        example, with default settings, :func:`~mpmath.quad` is able to integrate
        `\sin(x)` accurately over an interval of length 100 but not over
        length 1000::

            >>> quad(sin, [0, 100]); 1-cos(100)   # Good
            0.137681127712316
            0.137681127712316
            >>> quad(sin, [0, 1000]); 1-cos(1000)   # Bad
            -37.8587612408485
            0.437620923709297

        One solution is to break the integration into 10 intervals of
        length 100::

            >>> quad(sin, linspace(0, 1000, 10))   # Good
            0.437620923709297

        Another is to increase the degree of the quadrature::

            >>> quad(sin, [0, 1000], maxdegree=10)   # Also good
            0.437620923709297

        Whether splitting the interval or increasing the degree is
        more efficient differs from case to case. Another example is the
        function `1/(1+x^2)`, which has a sharp peak centered around
        `x = 0`::

            >>> f = lambda x: 1/(1+x**2)
            >>> quad(f, [-100, 100])   # Bad
            3.64804647105268
            >>> quad(f, [-100, 100], maxdegree=10)   # Good
            3.12159332021646
            >>> quad(f, [-100, 0, 100])   # Also good
            3.12159332021646

        **References**

        1. http://mathworld.wolfram.com/DoubleIntegral.html

        """
        rule = kwargs.get('method', 'tanh-sinh')
        if type(rule) is str:
            if rule == 'tanh-sinh':
                rule = ctx._tanh_sinh
            elif rule == 'gauss-legendre':
                rule = ctx._gauss_legendre
            else:
                raise ValueError("unknown quadrature rule: %s" % rule)
        else:
            rule = rule(ctx)
        verbose = kwargs.get('verbose')
        dim = len(points)
        orig = prec = ctx.prec
        epsilon = ctx.eps/8
        m = kwargs.get('maxdegree') or rule.guess_degree(prec)
        points = [ctx._as_points(p) for p in points]
        try:
            ctx.prec += 20
            if dim == 1:
                v, err = rule.summation(f, points[0], prec, epsilon, m, verbose)
            elif dim == 2:
                v, err = rule.summation(lambda x: \
                        rule.summation(lambda y: f(x,y), \
                        points[1], prec, epsilon, m)[0],
                    points[0], prec, epsilon, m, verbose)
            elif dim == 3:
                v, err = rule.summation(lambda x: \
                        rule.summation(lambda y: \
                            rule.summation(lambda z: f(x,y,z), \
                            points[2], prec, epsilon, m)[0],
                        points[1], prec, epsilon, m)[0],
                    points[0], prec, epsilon, m, verbose)
            else:
                raise NotImplementedError("quadrature must have dim 1, 2 or 3")
        finally:
            ctx.prec = orig
        if kwargs.get("error"):
            return +v, err
        return +v

    def quadts(ctx, *args, **kwargs):
        """
        Performs tanh-sinh quadrature. The call

            quadts(func, *points, ...)

        is simply a shortcut for:

            quad(func, *points, ..., method=TanhSinh)

        For example, a single integral and a double integral:

            quadts(lambda x: exp(cos(x)), [0, 1])
            quadts(lambda x, y: exp(cos(x+y)), [0, 1], [0, 1])

        See the documentation for quad for information about how points
        arguments and keyword arguments are parsed.

        See documentation for TanhSinh for algorithmic information about
        tanh-sinh quadrature.
        """
        kwargs['method'] = 'tanh-sinh'
        return ctx.quad(*args, **kwargs)

    def quadgl(ctx, *args, **kwargs):
        """
        Performs Gauss-Legendre quadrature. The call

            quadgl(func, *points, ...)

        is simply a shortcut for:

            quad(func, *points, ..., method=GaussLegendre)

        For example, a single integral and a double integral:

            quadgl(lambda x: exp(cos(x)), [0, 1])
            quadgl(lambda x, y: exp(cos(x+y)), [0, 1], [0, 1])

        See the documentation for quad for information about how points
        arguments and keyword arguments are parsed.

        See documentation for TanhSinh for algorithmic information about
        tanh-sinh quadrature.
        """
        kwargs['method'] = 'gauss-legendre'
        return ctx.quad(*args, **kwargs)

    def quadosc(ctx, f, interval, omega=None, period=None, zeros=None):
        r"""
        Calculates

        .. math ::

            I = \int_a^b f(x) dx

        where at least one of `a` and `b` is infinite and where
        `f(x) = g(x) \cos(\omega x  + \phi)` for some slowly
        decreasing function `g(x)`. With proper input, :func:`~mpmath.quadosc`
        can also handle oscillatory integrals where the oscillation
        rate is different from a pure sine or cosine wave.

        In the standard case when `|a| < \infty, b = \infty`,
        :func:`~mpmath.quadosc` works by evaluating the infinite series

        .. math ::

            I = \int_a^{x_1} f(x) dx +
            \sum_{k=1}^{\infty} \int_{x_k}^{x_{k+1}} f(x) dx

        where `x_k` are consecutive zeros (alternatively
        some other periodic reference point) of `f(x)`.
        Accordingly, :func:`~mpmath.quadosc` requires information about the
        zeros of `f(x)`. For a periodic function, you can specify
        the zeros by either providing the angular frequency `\omega`
        (*omega*) or the *period* `2 \pi/\omega`. In general, you can
        specify the `n`-th zero by providing the *zeros* arguments.
        Below is an example of each::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> f = lambda x: sin(3*x)/(x**2+1)
            >>> quadosc(f, [0,inf], omega=3)
            0.37833007080198
            >>> quadosc(f, [0,inf], period=2*pi/3)
            0.37833007080198
            >>> quadosc(f, [0,inf], zeros=lambda n: pi*n/3)
            0.37833007080198
            >>> (ei(3)*exp(-3)-exp(3)*ei(-3))/2  # Computed by Mathematica
            0.37833007080198

        Note that *zeros* was specified to multiply `n` by the
        *half-period*, not the full period. In theory, it does not matter
        whether each partial integral is done over a half period or a full
        period. However, if done over half-periods, the infinite series
        passed to :func:`~mpmath.nsum` becomes an *alternating series* and this
        typically makes the extrapolation much more efficient.

        Here is an example of an integration over the entire real line,
        and a half-infinite integration starting at `-\infty`::

            >>> quadosc(lambda x: cos(x)/(1+x**2), [-inf, inf], omega=1)
            1.15572734979092
            >>> pi/e
            1.15572734979092
            >>> quadosc(lambda x: cos(x)/x**2, [-inf, -1], period=2*pi)
            -0.0844109505595739
            >>> cos(1)+si(1)-pi/2
            -0.0844109505595738

        Of course, the integrand may contain a complex exponential just as
        well as a real sine or cosine::

            >>> quadosc(lambda x: exp(3*j*x)/(1+x**2), [-inf,inf], omega=3)
            (0.156410688228254 + 0.0j)
            >>> pi/e**3
            0.156410688228254
            >>> quadosc(lambda x: exp(3*j*x)/(2+x+x**2), [-inf,inf], omega=3)
            (0.00317486988463794 - 0.0447701735209082j)
            >>> 2*pi/sqrt(7)/exp(3*(j+sqrt(7))/2)
            (0.00317486988463794 - 0.0447701735209082j)

        **Non-periodic functions**

        If `f(x) = g(x) h(x)` for some function `h(x)` that is not
        strictly periodic, *omega* or *period* might not work, and it might
        be necessary to use *zeros*.

        A notable exception can be made for Bessel functions which, though not
        periodic, are "asymptotically periodic" in a sufficiently strong sense
        that the sum extrapolation will work out::

            >>> quadosc(j0, [0, inf], period=2*pi)
            1.0
            >>> quadosc(j1, [0, inf], period=2*pi)
            1.0

        More properly, one should provide the exact Bessel function zeros::

            >>> j0zero = lambda n: findroot(j0, pi*(n-0.25))
            >>> quadosc(j0, [0, inf], zeros=j0zero)
            1.0

        For an example where *zeros* becomes necessary, consider the
        complete Fresnel integrals

        .. math ::

            \int_0^{\infty} \cos x^2\,dx = \int_0^{\infty} \sin x^2\,dx
            = \sqrt{\frac{\pi}{8}}.

        Although the integrands do not decrease in magnitude as
        `x \to \infty`, the integrals are convergent since the oscillation
        rate increases (causing consecutive periods to asymptotically
        cancel out). These integrals are virtually impossible to calculate
        to any kind of accuracy using standard quadrature rules. However,
        if one provides the correct asymptotic distribution of zeros
        (`x_n \sim \sqrt{n}`), :func:`~mpmath.quadosc` works::

            >>> mp.dps = 30
            >>> f = lambda x: cos(x**2)
            >>> quadosc(f, [0,inf], zeros=lambda n:sqrt(pi*n))
            0.626657068657750125603941321203
            >>> f = lambda x: sin(x**2)
            >>> quadosc(f, [0,inf], zeros=lambda n:sqrt(pi*n))
            0.626657068657750125603941321203
            >>> sqrt(pi/8)
            0.626657068657750125603941321203

        (Interestingly, these integrals can still be evaluated if one
        places some other constant than `\pi` in the square root sign.)

        In general, if `f(x) \sim g(x) \cos(h(x))`, the zeros follow
        the inverse-function distribution `h^{-1}(x)`::

            >>> mp.dps = 15
            >>> f = lambda x: sin(exp(x))
            >>> quadosc(f, [1,inf], zeros=lambda n: log(n))
            -0.25024394235267
            >>> pi/2-si(e)
            -0.250243942352671

        **Non-alternating functions**

        If the integrand oscillates around a positive value, without
        alternating signs, the extrapolation might fail. A simple trick
        that sometimes works is to multiply or divide the frequency by 2::

            >>> f = lambda x: 1/x**2+sin(x)/x**4
            >>> quadosc(f, [1,inf], omega=1)  # Bad
            1.28642190869861
            >>> quadosc(f, [1,inf], omega=0.5)  # Perfect
            1.28652953559617
            >>> 1+(cos(1)+ci(1)+sin(1))/6
            1.28652953559617

        **Fast decay**

        :func:`~mpmath.quadosc` is primarily useful for slowly decaying
        integrands. If the integrand decreases exponentially or faster,
        :func:`~mpmath.quad` will likely handle it without trouble (and generally be
        much faster than :func:`~mpmath.quadosc`)::

            >>> quadosc(lambda x: cos(x)/exp(x), [0, inf], omega=1)
            0.5
            >>> quad(lambda x: cos(x)/exp(x), [0, inf])
            0.5

        """
        a, b = ctx._as_points(interval)
        a = ctx.convert(a)
        b = ctx.convert(b)
        if [omega, period, zeros].count(None) != 2:
            raise ValueError( \
                "must specify exactly one of omega, period, zeros")
        if a == ctx.ninf and b == ctx.inf:
            s1 = ctx.quadosc(f, [a, 0], omega=omega, zeros=zeros, period=period)
            s2 = ctx.quadosc(f, [0, b], omega=omega, zeros=zeros, period=period)
            return s1 + s2
        if a == ctx.ninf:
            if zeros:
                return ctx.quadosc(lambda x:f(-x), [-b,-a], lambda n: zeros(-n))
            else:
                return ctx.quadosc(lambda x:f(-x), [-b,-a], omega=omega, period=period)
        if b != ctx.inf:
            raise ValueError("quadosc requires an infinite integration interval")
        if not zeros:
            if omega:
                period = 2*ctx.pi/omega
            zeros = lambda n: n*period/2
        #for n in range(1,10):
        #    p = zeros(n)
        #    if p > a:
        #        break
        #if n >= 9:
        #    raise ValueError("zeros do not appear to be correctly indexed")
        n = 1
        s = ctx.quadgl(f, [a, zeros(n)])
        def term(k):
            return ctx.quadgl(f, [zeros(k), zeros(k+1)])
        s += ctx.nsum(term, [n, ctx.inf])
        return s

    def quadsubdiv(ctx, f, interval, tol=None, maxintervals=None, **kwargs):
        """
        Computes the integral of *f* over the interval or path specified
        by *interval*, using :func:`~mpmath.quad` together with adaptive
        subdivision of the interval.

        This function gives an accurate answer for some integrals where
        :func:`~mpmath.quad` fails::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> quad(lambda x: abs(sin(x)), [0, 2*pi])
            3.99900894176779
            >>> quadsubdiv(lambda x: abs(sin(x)), [0, 2*pi])
            4.0
            >>> quadsubdiv(sin, [0, 1000])
            0.437620923709297
            >>> quadsubdiv(lambda x: 1/(1+x**2), [-100, 100])
            3.12159332021646
            >>> quadsubdiv(lambda x: ceil(x), [0, 100])
            5050.0
            >>> quadsubdiv(lambda x: sin(x+exp(x)), [0,8])
            0.347400172657248

        The argument *maxintervals* can be set to limit the permissible
        subdivision::

            >>> quadsubdiv(lambda x: sin(x**2), [0,100], maxintervals=5, error=True)
            (-5.40487904307774, 5.011)
            >>> quadsubdiv(lambda x: sin(x**2), [0,100], maxintervals=100, error=True)
            (0.631417921866934, 1.10101120134116e-17)

        Subdivision does not guarantee a correct answer since, the error
        estimate on subintervals may be inaccurate::

            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, [0,1], error=True)
            (0.210802735500549, 1.0001111101e-17)
            >>> mp.dps = 20
            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, [0,1], error=True)
            (0.21080273550054927738, 2.200000001e-24)

        The second answer is correct. We can get an accurate result at lower
        precision by forcing a finer initial subdivision::

            >>> mp.dps = 15
            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, linspace(0,1,5))
            0.210802735500549

        The following integral is too oscillatory for convergence, but we can get a
        reasonable estimate::

            >>> v, err = fp.quadsubdiv(lambda x: fp.sin(1/x), [0,1], error=True)
            >>> round(v, 6), round(err, 6)
            (0.504067, 1e-06)
            >>> sin(1) - ci(1)
            0.504067061906928

        """
        queue = []
        for i in range(len(interval)-1):
            queue.append((interval[i], interval[i+1]))
        total = ctx.zero
        total_error = ctx.zero
        if maxintervals is None:
            maxintervals = 10 * ctx.prec
        count = 0
        quad_args = kwargs.copy()
        quad_args["verbose"] = False
        quad_args["error"] = True
        if tol is None:
            tol = +ctx.eps
        orig = ctx.prec
        try:
            ctx.prec += 5
            while queue:
                a, b = queue.pop()
                s, err = ctx.quad(f, [a, b], **quad_args)
                if kwargs.get("verbose"):
                    print("subinterval", count, a, b, err)
                if err < tol or count > maxintervals:
                    total += s
                    total_error += err
                else:
                    count += 1
                    if count == maxintervals and kwargs.get("verbose"):
                        print("warning: number of intervals exceeded maxintervals")
                    if a == -ctx.inf and b == ctx.inf:
                        m = 0
                    elif a == -ctx.inf:
                        m = min(b-1, 2*b)
                    elif b == ctx.inf:
                        m = max(a+1, 2*a)
                    else:
                        m = a + (b - a) / 2
                    queue.append((a, m))
                    queue.append((m, b))
        finally:
            ctx.prec = orig
        if kwargs.get("error"):
            return +total, +total_error
        else:
            return +total

if __name__ == '__main__':
    import doctest
    doctest.testmod()
