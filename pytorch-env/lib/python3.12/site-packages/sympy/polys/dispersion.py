from sympy.core import S
from sympy.polys import Poly


def dispersionset(p, q=None, *gens, **args):
    r"""Compute the *dispersion set* of two polynomials.

    For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
    and `\deg g > 0` the dispersion set `\operatorname{J}(f, g)` is defined as:

    .. math::
        \operatorname{J}(f, g)
        & := \{a \in \mathbb{N}_0 | \gcd(f(x), g(x+a)) \neq 1\} \\
        &  = \{a \in \mathbb{N}_0 | \deg \gcd(f(x), g(x+a)) \geq 1\}

    For a single polynomial one defines `\operatorname{J}(f) := \operatorname{J}(f, f)`.

    Examples
    ========

    >>> from sympy import poly
    >>> from sympy.polys.dispersion import dispersion, dispersionset
    >>> from sympy.abc import x

    Dispersion set and dispersion of a simple polynomial:

    >>> fp = poly((x - 3)*(x + 3), x)
    >>> sorted(dispersionset(fp))
    [0, 6]
    >>> dispersion(fp)
    6

    Note that the definition of the dispersion is not symmetric:

    >>> fp = poly(x**4 - 3*x**2 + 1, x)
    >>> gp = fp.shift(-3)
    >>> sorted(dispersionset(fp, gp))
    [2, 3, 4]
    >>> dispersion(fp, gp)
    4
    >>> sorted(dispersionset(gp, fp))
    []
    >>> dispersion(gp, fp)
    -oo

    Computing the dispersion also works over field extensions:

    >>> from sympy import sqrt
    >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
    >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
    >>> sorted(dispersionset(fp, gp))
    [2]
    >>> sorted(dispersionset(gp, fp))
    [1, 4]

    We can even perform the computations for polynomials
    having symbolic coefficients:

    >>> from sympy.abc import a
    >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
    >>> sorted(dispersionset(fp))
    [0, 1]

    See Also
    ========

    dispersion

    References
    ==========

    .. [1] [ManWright94]_
    .. [2] [Koepf98]_
    .. [3] [Abramov71]_
    .. [4] [Man93]_
    """
    # Check for valid input
    same = False if q is not None else True
    if same:
        q = p

    p = Poly(p, *gens, **args)
    q = Poly(q, *gens, **args)

    if not p.is_univariate or not q.is_univariate:
        raise ValueError("Polynomials need to be univariate")

    # The generator
    if not p.gen == q.gen:
        raise ValueError("Polynomials must have the same generator")
    gen = p.gen

    # We define the dispersion of constant polynomials to be zero
    if p.degree() < 1 or q.degree() < 1:
        return {0}

    # Factor p and q over the rationals
    fp = p.factor_list()
    fq = q.factor_list() if not same else fp

    # Iterate over all pairs of factors
    J = set()
    for s, unused in fp[1]:
        for t, unused in fq[1]:
            m = s.degree()
            n = t.degree()
            if n != m:
                continue
            an = s.LC()
            bn = t.LC()
            if not (an - bn).is_zero:
                continue
            # Note that the roles of `s` and `t` below are switched
            # w.r.t. the original paper. This is for consistency
            # with the description in the book of W. Koepf.
            anm1 = s.coeff_monomial(gen**(m-1))
            bnm1 = t.coeff_monomial(gen**(n-1))
            alpha = (anm1 - bnm1) / S(n*bn)
            if not alpha.is_integer:
                continue
            if alpha < 0 or alpha in J:
                continue
            if n > 1 and not (s - t.shift(alpha)).is_zero:
                continue
            J.add(alpha)

    return J


def dispersion(p, q=None, *gens, **args):
    r"""Compute the *dispersion* of polynomials.

    For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
    and `\deg g > 0` the dispersion `\operatorname{dis}(f, g)` is defined as:

    .. math::
        \operatorname{dis}(f, g)
        & := \max\{ J(f,g) \cup \{0\} \} \\
        &  = \max\{ \{a \in \mathbb{N} | \gcd(f(x), g(x+a)) \neq 1\} \cup \{0\} \}

    and for a single polynomial `\operatorname{dis}(f) := \operatorname{dis}(f, f)`.
    Note that we make the definition `\max\{\} := -\infty`.

    Examples
    ========

    >>> from sympy import poly
    >>> from sympy.polys.dispersion import dispersion, dispersionset
    >>> from sympy.abc import x

    Dispersion set and dispersion of a simple polynomial:

    >>> fp = poly((x - 3)*(x + 3), x)
    >>> sorted(dispersionset(fp))
    [0, 6]
    >>> dispersion(fp)
    6

    Note that the definition of the dispersion is not symmetric:

    >>> fp = poly(x**4 - 3*x**2 + 1, x)
    >>> gp = fp.shift(-3)
    >>> sorted(dispersionset(fp, gp))
    [2, 3, 4]
    >>> dispersion(fp, gp)
    4
    >>> sorted(dispersionset(gp, fp))
    []
    >>> dispersion(gp, fp)
    -oo

    The maximum of an empty set is defined to be `-\infty`
    as seen in this example.

    Computing the dispersion also works over field extensions:

    >>> from sympy import sqrt
    >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
    >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
    >>> sorted(dispersionset(fp, gp))
    [2]
    >>> sorted(dispersionset(gp, fp))
    [1, 4]

    We can even perform the computations for polynomials
    having symbolic coefficients:

    >>> from sympy.abc import a
    >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
    >>> sorted(dispersionset(fp))
    [0, 1]

    See Also
    ========

    dispersionset

    References
    ==========

    .. [1] [ManWright94]_
    .. [2] [Koepf98]_
    .. [3] [Abramov71]_
    .. [4] [Man93]_
    """
    J = dispersionset(p, q, *gens, **args)
    if not J:
        # Definition for maximum of empty set
        j = S.NegativeInfinity
    else:
        j = max(J)
    return j
