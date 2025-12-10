""" Riemann zeta and related function. """

from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import ArgumentIndexError, expand_mul, DefinedFunction
from sympy.core.logic import fuzzy_not
from sympy.core.numbers import pi, I, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.functions.elementary.complexes import re, unpolarify, Abs, polar_lift
from sympy.functions.elementary.exponential import log, exp_polar, exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polytools import Poly

###############################################################################
###################### LERCH TRANSCENDENT #####################################
###############################################################################


class lerchphi(DefinedFunction):
    r"""
    Lerch transcendent (Lerch phi function).

    Explanation
    ===========

    For $\operatorname{Re}(a) > 0$, $|z| < 1$ and $s \in \mathbb{C}$, the
    Lerch transcendent is defined as

    .. math :: \Phi(z, s, a) = \sum_{n=0}^\infty \frac{z^n}{(n + a)^s},

    where the standard branch of the argument is used for $n + a$,
    and by analytic continuation for other values of the parameters.

    A commonly used related function is the Lerch zeta function, defined by

    .. math:: L(q, s, a) = \Phi(e^{2\pi i q}, s, a).

    **Analytic Continuation and Branching Behavior**

    It can be shown that

    .. math:: \Phi(z, s, a) = z\Phi(z, s, a+1) + a^{-s}.

    This provides the analytic continuation to $\operatorname{Re}(a) \le 0$.

    Assume now $\operatorname{Re}(a) > 0$. The integral representation

    .. math:: \Phi_0(z, s, a) = \int_0^\infty \frac{t^{s-1} e^{-at}}{1 - ze^{-t}}
                                \frac{\mathrm{d}t}{\Gamma(s)}

    provides an analytic continuation to $\mathbb{C} - [1, \infty)$.
    Finally, for $x \in (1, \infty)$ we find

    .. math:: \lim_{\epsilon \to 0^+} \Phi_0(x + i\epsilon, s, a)
             -\lim_{\epsilon \to 0^+} \Phi_0(x - i\epsilon, s, a)
             = \frac{2\pi i \log^{s-1}{x}}{x^a \Gamma(s)},

    using the standard branch for both $\log{x}$ and
    $\log{\log{x}}$ (a branch of $\log{\log{x}}$ is needed to
    evaluate $\log{x}^{s-1}$).
    This concludes the analytic continuation. The Lerch transcendent is thus
    branched at $z \in \{0, 1, \infty\}$ and
    $a \in \mathbb{Z}_{\le 0}$. For fixed $z, a$ outside these
    branch points, it is an entire function of $s$.

    Examples
    ========

    The Lerch transcendent is a fairly general function, for this reason it does
    not automatically evaluate to simpler functions. Use ``expand_func()`` to
    achieve this.

    If $z=1$, the Lerch transcendent reduces to the Hurwitz zeta function:

    >>> from sympy import lerchphi, expand_func
    >>> from sympy.abc import z, s, a
    >>> expand_func(lerchphi(1, s, a))
    zeta(s, a)

    More generally, if $z$ is a root of unity, the Lerch transcendent
    reduces to a sum of Hurwitz zeta functions:

    >>> expand_func(lerchphi(-1, s, a))
    zeta(s, a/2)/2**s - zeta(s, a/2 + 1/2)/2**s

    If $a=1$, the Lerch transcendent reduces to the polylogarithm:

    >>> expand_func(lerchphi(z, s, 1))
    polylog(s, z)/z

    More generally, if $a$ is rational, the Lerch transcendent reduces
    to a sum of polylogarithms:

    >>> from sympy import S
    >>> expand_func(lerchphi(z, s, S(1)/2))
    2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z) -
                polylog(s, sqrt(z)*exp_polar(I*pi))/sqrt(z))
    >>> expand_func(lerchphi(z, s, S(3)/2))
    -2**s/z + 2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z) -
                          polylog(s, sqrt(z)*exp_polar(I*pi))/sqrt(z))/z

    The derivatives with respect to $z$ and $a$ can be computed in
    closed form:

    >>> lerchphi(z, s, a).diff(z)
    (-a*lerchphi(z, s, a) + lerchphi(z, s - 1, a))/z
    >>> lerchphi(z, s, a).diff(a)
    -s*lerchphi(z, s + 1, a)

    See Also
    ========

    polylog, zeta

    References
    ==========

    .. [1] Bateman, H.; Erdelyi, A. (1953), Higher Transcendental Functions,
           Vol. I, New York: McGraw-Hill. Section 1.11.
    .. [2] https://dlmf.nist.gov/25.14
    .. [3] https://en.wikipedia.org/wiki/Lerch_transcendent

    """

    def _eval_expand_func(self, **hints):
        z, s, a = self.args
        if z == 1:
            return zeta(s, a)
        if s.is_Integer and s <= 0:
            t = Dummy('t')
            p = Poly((t + a)**(-s), t)
            start = 1/(1 - t)
            res = S.Zero
            for c in reversed(p.all_coeffs()):
                res += c*start
                start = t*start.diff(t)
            return res.subs(t, z)

        if a.is_Rational:
            # See section 18 of
            #   Kelly B. Roach.  Hypergeometric Function Representations.
            #   In: Proceedings of the 1997 International Symposium on Symbolic and
            #   Algebraic Computation, pages 205-211, New York, 1997. ACM.
            # TODO should something be polarified here?
            add = S.Zero
            mul = S.One
            # First reduce a to the interaval (0, 1]
            if a > 1:
                n = floor(a)
                if n == a:
                    n -= 1
                a -= n
                mul = z**(-n)
                add = Add(*[-z**(k - n)/(a + k)**s for k in range(n)])
            elif a <= 0:
                n = floor(-a) + 1
                a += n
                mul = z**n
                add = Add(*[z**(n - 1 - k)/(a - k - 1)**s for k in range(n)])

            m, n = S([a.p, a.q])
            zet = exp_polar(2*pi*I/n)
            root = z**(1/n)
            up_zet = unpolarify(zet)
            addargs = []
            for k in range(n):
                p = polylog(s, zet**k*root)
                if isinstance(p, polylog):
                    p = p._eval_expand_func(**hints)
                addargs.append(p/(up_zet**k*root)**m)
            return add + mul*n**(s - 1)*Add(*addargs)

        # TODO use minpoly instead of ad-hoc methods when issue 5888 is fixed
        if isinstance(z, exp) and (z.args[0]/(pi*I)).is_Rational or z in [-1, I, -I]:
            # TODO reference?
            if z == -1:
                p, q = S([1, 2])
            elif z == I:
                p, q = S([1, 4])
            elif z == -I:
                p, q = S([-1, 4])
            else:
                arg = z.args[0]/(2*pi*I)
                p, q = S([arg.p, arg.q])
            return Add(*[exp(2*pi*I*k*p/q)/q**s*zeta(s, (k + a)/q)
                         for k in range(q)])

        return lerchphi(z, s, a)

    def fdiff(self, argindex=1):
        z, s, a = self.args
        if argindex == 3:
            return -s*lerchphi(z, s + 1, a)
        elif argindex == 1:
            return (lerchphi(z, s - 1, a) - a*lerchphi(z, s, a))/z
        else:
            raise ArgumentIndexError

    def _eval_rewrite_helper(self, target):
        res = self._eval_expand_func()
        if res.has(target):
            return res
        else:
            return self

    def _eval_rewrite_as_zeta(self, z, s, a, **kwargs):
        return self._eval_rewrite_helper(zeta)

    def _eval_rewrite_as_polylog(self, z, s, a, **kwargs):
        return self._eval_rewrite_helper(polylog)

###############################################################################
###################### POLYLOGARITHM ##########################################
###############################################################################


class polylog(DefinedFunction):
    r"""
    Polylogarithm function.

    Explanation
    ===========

    For $|z| < 1$ and $s \in \mathbb{C}$, the polylogarithm is
    defined by

    .. math:: \operatorname{Li}_s(z) = \sum_{n=1}^\infty \frac{z^n}{n^s},

    where the standard branch of the argument is used for $n$. It admits
    an analytic continuation which is branched at $z=1$ (notably not on the
    sheet of initial definition), $z=0$ and $z=\infty$.

    The name polylogarithm comes from the fact that for $s=1$, the
    polylogarithm is related to the ordinary logarithm (see examples), and that

    .. math:: \operatorname{Li}_{s+1}(z) =
                    \int_0^z \frac{\operatorname{Li}_s(t)}{t} \mathrm{d}t.

    The polylogarithm is a special case of the Lerch transcendent:

    .. math:: \operatorname{Li}_{s}(z) = z \Phi(z, s, 1).

    Examples
    ========

    For $z \in \{0, 1, -1\}$, the polylogarithm is automatically expressed
    using other functions:

    >>> from sympy import polylog
    >>> from sympy.abc import s
    >>> polylog(s, 0)
    0
    >>> polylog(s, 1)
    zeta(s)
    >>> polylog(s, -1)
    -dirichlet_eta(s)

    If $s$ is a negative integer, $0$ or $1$, the polylogarithm can be
    expressed using elementary functions. This can be done using
    ``expand_func()``:

    >>> from sympy import expand_func
    >>> from sympy.abc import z
    >>> expand_func(polylog(1, z))
    -log(1 - z)
    >>> expand_func(polylog(0, z))
    z/(1 - z)

    The derivative with respect to $z$ can be computed in closed form:

    >>> polylog(s, z).diff(z)
    polylog(s - 1, z)/z

    The polylogarithm can be expressed in terms of the lerch transcendent:

    >>> from sympy import lerchphi
    >>> polylog(s, z).rewrite(lerchphi)
    z*lerchphi(z, s, 1)

    See Also
    ========

    zeta, lerchphi

    """

    @classmethod
    def eval(cls, s, z):
        if z.is_number:
            if z is S.One:
                return zeta(s)
            elif z is S.NegativeOne:
                return -dirichlet_eta(s)
            elif z is S.Zero:
                return S.Zero
            elif s == 2:
                dilogtable = _dilogtable()
                if z in dilogtable:
                    return dilogtable[z]

        if z.is_zero:
            return S.Zero

        # Make an effort to determine if z is 1 to avoid replacing into
        # expression with singularity
        zone = z.equals(S.One)

        if zone:
            return zeta(s)
        elif zone is False:
            # For s = 0 or -1 use explicit formulas to evaluate, but
            # automatically expanding polylog(1, z) to -log(1-z) seems
            # undesirable for summation methods based on hypergeometric
            # functions
            if s is S.Zero:
                return z/(1 - z)
            elif s is S.NegativeOne:
                return z/(1 - z)**2
            if s.is_zero:
                return z/(1 - z)

        # polylog is branched, but not over the unit disk
        if z.has(exp_polar, polar_lift) and (zone or (Abs(z) <= S.One) == True):
            return cls(s, unpolarify(z))

    def fdiff(self, argindex=1):
        s, z = self.args
        if argindex == 2:
            return polylog(s - 1, z)/z
        raise ArgumentIndexError

    def _eval_rewrite_as_lerchphi(self, s, z, **kwargs):
        return z*lerchphi(z, s, 1)

    def _eval_expand_func(self, **hints):
        s, z = self.args
        if s == 1:
            return -log(1 - z)
        if s.is_Integer and s <= 0:
            u = Dummy('u')
            start = u/(1 - u)
            for _ in range(-s):
                start = u*start.diff(u)
            return expand_mul(start).subs(u, z)
        return polylog(s, z)

    def _eval_is_zero(self):
        z = self.args[1]
        if z.is_zero:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        nu, z = self.args

        z0 = z.subs(x, 0)
        if z0 is S.NaN:
            z0 = z.limit(x, 0, dir='-' if re(cdir).is_negative else '+')

        if z0.is_zero:
            # In case of powers less than 1, number of terms need to be computed
            # separately to avoid repeated callings of _eval_nseries with wrong n
            try:
                _, exp = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self

            if exp.is_positive:
                newn = ceiling(n/exp)
                o = Order(x**n, x)
                r = z._eval_nseries(x, n, logx, cdir).removeO()
                if r is S.Zero:
                    return o

                term = r
                s = [term]
                for k in range(2, newn):
                    term *= r
                    s.append(term/k**nu)
                return Add(*s) + o

        return super(polylog, self)._eval_nseries(x, n, logx, cdir)

###############################################################################
###################### HURWITZ GENERALIZED ZETA FUNCTION ######################
###############################################################################


class zeta(DefinedFunction):
    r"""
    Hurwitz zeta function (or Riemann zeta function).

    Explanation
    ===========

    For $\operatorname{Re}(a) > 0$ and $\operatorname{Re}(s) > 1$, this
    function is defined as

    .. math:: \zeta(s, a) = \sum_{n=0}^\infty \frac{1}{(n + a)^s},

    where the standard choice of argument for $n + a$ is used. For fixed
    $a$ not a nonpositive integer the Hurwitz zeta function admits a
    meromorphic continuation to all of $\mathbb{C}$; it is an unbranched
    function with a simple pole at $s = 1$.

    The Hurwitz zeta function is a special case of the Lerch transcendent:

    .. math:: \zeta(s, a) = \Phi(1, s, a).

    This formula defines an analytic continuation for all possible values of
    $s$ and $a$ (also $\operatorname{Re}(a) < 0$), see the documentation of
    :class:`lerchphi` for a description of the branching behavior.

    If no value is passed for $a$ a default value of $a = 1$ is assumed,
    yielding the Riemann zeta function.

    Examples
    ========

    For $a = 1$ the Hurwitz zeta function reduces to the famous Riemann
    zeta function:

    .. math:: \zeta(s, 1) = \zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s}.

    >>> from sympy import zeta
    >>> from sympy.abc import s
    >>> zeta(s, 1)
    zeta(s)
    >>> zeta(s)
    zeta(s)

    The Riemann zeta function can also be expressed using the Dirichlet eta
    function:

    >>> from sympy import dirichlet_eta
    >>> zeta(s).rewrite(dirichlet_eta)
    dirichlet_eta(s)/(1 - 2**(1 - s))

    The Riemann zeta function at nonnegative even and negative integer
    values is related to the Bernoulli numbers and polynomials:

    >>> zeta(2)
    pi**2/6
    >>> zeta(4)
    pi**4/90
    >>> zeta(0)
    -1/2
    >>> zeta(-1)
    -1/12
    >>> zeta(-4)
    0

    The specific formulae are:

    .. math:: \zeta(2n) = -\frac{(2\pi i)^{2n} B_{2n}}{2(2n)!}
    .. math:: \zeta(-n,a) = -\frac{B_{n+1}(a)}{n+1}

    No closed-form expressions are known at positive odd integers, but
    numerical evaluation is possible:

    >>> zeta(3).n()
    1.20205690315959

    The derivative of $\zeta(s, a)$ with respect to $a$ can be computed:

    >>> from sympy.abc import a
    >>> zeta(s, a).diff(a)
    -s*zeta(s + 1, a)

    However the derivative with respect to $s$ has no useful closed form
    expression:

    >>> zeta(s, a).diff(s)
    Derivative(zeta(s, a), s)

    The Hurwitz zeta function can be expressed in terms of the Lerch
    transcendent, :class:`~.lerchphi`:

    >>> from sympy import lerchphi
    >>> zeta(s, a).rewrite(lerchphi)
    lerchphi(1, s, a)

    See Also
    ========

    dirichlet_eta, lerchphi, polylog

    References
    ==========

    .. [1] https://dlmf.nist.gov/25.11
    .. [2] https://en.wikipedia.org/wiki/Hurwitz_zeta_function

    """

    @classmethod
    def eval(cls, s, a=None):
        if a is S.One:
            return cls(s)
        elif s is S.NaN or a is S.NaN:
            return S.NaN
        elif s is S.One:
            return S.ComplexInfinity
        elif s is S.Infinity:
            return S.One
        elif a is S.Infinity:
            return S.Zero

        sint = s.is_Integer
        if a is None:
            a = S.One
        if sint and s.is_nonpositive:
            return bernoulli(1-s, a) / (s-1)
        elif a is S.One:
            if sint and s.is_even:
                return -(2*pi*I)**s * bernoulli(s) / (2*factorial(s))
        elif sint and a.is_Integer and a.is_positive:
            return cls(s) - harmonic(a-1, s)
        elif a.is_Integer and a.is_nonpositive and \
                (s.is_integer is False or s.is_nonpositive is False):
            return S.NaN

    def _eval_rewrite_as_bernoulli(self, s, a=1, **kwargs):
        if a == 1 and s.is_integer and s.is_nonnegative and s.is_even:
            return -(2*pi*I)**s * bernoulli(s) / (2*factorial(s))
        return bernoulli(1-s, a) / (s-1)

    def _eval_rewrite_as_dirichlet_eta(self, s, a=1, **kwargs):
        if a != 1:
            return self
        s = self.args[0]
        return dirichlet_eta(s)/(1 - 2**(1 - s))

    def _eval_rewrite_as_lerchphi(self, s, a=1, **kwargs):
        return lerchphi(1, s, a)

    def _eval_is_finite(self):
        return fuzzy_not((self.args[0] - 1).is_zero)

    def _eval_expand_func(self, **hints):
        s = self.args[0]
        a = self.args[1] if len(self.args) > 1 else S.One
        if a.is_integer:
            if a.is_positive:
                return zeta(s) - harmonic(a-1, s)
            if a.is_nonpositive and (s.is_integer is False or
                    s.is_nonpositive is False):
                return S.NaN
        return self

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (1,)
        if argindex == 2:
            return -s*zeta(s + 1, a)
        else:
            raise ArgumentIndexError

    def _eval_as_leading_term(self, x, logx, cdir):
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (S.One,)

        try:
            c, e = a.leadterm(x)
        except NotImplementedError:
            return self

        if e.is_negative and not s.is_positive:
            raise NotImplementedError

        return super(zeta, self)._eval_as_leading_term(x, logx=logx, cdir=cdir)


class dirichlet_eta(DefinedFunction):
    r"""
    Dirichlet eta function.

    Explanation
    ===========

    For $\operatorname{Re}(s) > 0$ and $0 < x \le 1$, this function is defined as

    .. math:: \eta(s, a) = \sum_{n=0}^\infty \frac{(-1)^n}{(n+a)^s}.

    It admits a unique analytic continuation to all of $\mathbb{C}$ for any
    fixed $a$ not a nonpositive integer. It is an entire, unbranched function.

    It can be expressed using the Hurwitz zeta function as

    .. math:: \eta(s, a) = \zeta(s,a) - 2^{1-s} \zeta\left(s, \frac{a+1}{2}\right)

    and using the generalized Genocchi function as

    .. math:: \eta(s, a) = \frac{G(1-s, a)}{2(s-1)}.

    In both cases the limiting value of $\log2 - \psi(a) + \psi\left(\frac{a+1}{2}\right)$
    is used when $s = 1$.

    Examples
    ========

    >>> from sympy import dirichlet_eta, zeta
    >>> from sympy.abc import s
    >>> dirichlet_eta(s).rewrite(zeta)
    Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1 - s))*zeta(s), True))

    See Also
    ========

    zeta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_eta_function
    .. [2] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    def eval(cls, s, a=None):
        if a is S.One:
            return cls(s)
        if a is None:
            if s == 1:
                return log(2)
            z = zeta(s)
            if not z.has(zeta):
                return (1 - 2**(1-s)) * z
            return
        elif s == 1:
            from sympy.functions.special.gamma_functions import digamma
            return log(2) - digamma(a) + digamma((a+1)/2)
        z1 = zeta(s, a)
        z2 = zeta(s, (a+1)/2)
        if not z1.has(zeta) and not z2.has(zeta):
            return z1 - 2**(1-s) * z2

    def _eval_rewrite_as_zeta(self, s, a=1, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        if a == 1:
            return Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1-s)) * zeta(s), True))
        return Piecewise((log(2) - digamma(a) + digamma((a+1)/2), Eq(s, 1)),
                (zeta(s, a) - 2**(1-s) * zeta(s, (a+1)/2), True))

    def _eval_rewrite_as_genocchi(self, s, a=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((log(2) - digamma(a) + digamma((a+1)/2), Eq(s, 1)),
                (genocchi(1-s, a) / (2 * (s-1)), True))

    def _eval_evalf(self, prec):
        if all(i.is_number for i in self.args):
            return self.rewrite(zeta)._eval_evalf(prec)


class riemann_xi(DefinedFunction):
    r"""
    Riemann Xi function.

    Examples
    ========

    The Riemann Xi function is closely related to the Riemann zeta function.
    The zeros of Riemann Xi function are precisely the non-trivial zeros
    of the zeta function.

    >>> from sympy import riemann_xi, zeta
    >>> from sympy.abc import s
    >>> riemann_xi(s).rewrite(zeta)
    s*(s - 1)*gamma(s/2)*zeta(s)/(2*pi**(s/2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Riemann_Xi_function

    """


    @classmethod
    def eval(cls, s):
        from sympy.functions.special.gamma_functions import gamma
        z = zeta(s)
        if s in (S.Zero, S.One):
            return S.Half

        if not isinstance(z, zeta):
            return s*(s - 1)*gamma(s/2)*z/(2*pi**(s/2))

    def _eval_rewrite_as_zeta(self, s, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return s*(s - 1)*gamma(s/2)*zeta(s)/(2*pi**(s/2))


class stieltjes(DefinedFunction):
    r"""
    Represents Stieltjes constants, $\gamma_{k}$ that occur in
    Laurent Series expansion of the Riemann zeta function.

    Examples
    ========

    >>> from sympy import stieltjes
    >>> from sympy.abc import n, m
    >>> stieltjes(n)
    stieltjes(n)

    The zero'th stieltjes constant:

    >>> stieltjes(0)
    EulerGamma
    >>> stieltjes(0, 1)
    EulerGamma

    For generalized stieltjes constants:

    >>> stieltjes(n, m)
    stieltjes(n, m)

    Constants are only defined for integers >= 0:

    >>> stieltjes(-1)
    zoo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Stieltjes_constants

    """

    @classmethod
    def eval(cls, n, a=None):
        if a is not None:
            a = sympify(a)
            if a is S.NaN:
                return S.NaN
            if a.is_Integer and a.is_nonpositive:
                return S.ComplexInfinity

        if n.is_Number:
            if n is S.NaN:
                return S.NaN
            elif n < 0:
                return S.ComplexInfinity
            elif not n.is_Integer:
                return S.ComplexInfinity
            elif n is S.Zero and a in [None, 1]:
                return S.EulerGamma

        if n.is_extended_negative:
            return S.ComplexInfinity

        if n.is_zero and a in [None, 1]:
            return S.EulerGamma

        if n.is_integer == False:
            return S.ComplexInfinity


@cacheit
def _dilogtable():
    return {
        S.Half: pi**2/12 - log(2)**2/2,
        Integer(2) : pi**2/4 - I*pi*log(2),
        -(sqrt(5) - 1)/2 : -pi**2/15 + log((sqrt(5)-1)/2)**2/2,
        -(sqrt(5) + 1)/2 : -pi**2/10 - log((sqrt(5)+1)/2)**2,
        (3 - sqrt(5))/2 : pi**2/15 - log((sqrt(5)-1)/2)**2,
        (sqrt(5) - 1)/2 : pi**2/10 - log((sqrt(5)-1)/2)**2,
        I : I*S.Catalan - pi**2/48,
        -I : -I*S.Catalan - pi**2/48,
        1 - I : pi**2/16 - I*S.Catalan - pi*I/4*log(2),
        1 + I : pi**2/16 + I*S.Catalan + pi*I/4*log(2),
        (1 - I)/2 : -log(2)**2/8 + pi*I*log(2)/8 + 5*pi**2/96 - I*S.Catalan
    }
