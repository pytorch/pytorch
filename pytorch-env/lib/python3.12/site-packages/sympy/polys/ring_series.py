"""Power series evaluation and manipulation using sparse Polynomials

Implementing a new function
---------------------------

There are a few things to be kept in mind when adding a new function here::

    - The implementation should work on all possible input domains/rings.
      Special cases include the ``EX`` ring and a constant term in the series
      to be expanded. There can be two types of constant terms in the series:

        + A constant value or symbol.
        + A term of a multivariate series not involving the generator, with
          respect to which the series is to expanded.

      Strictly speaking, a generator of a ring should not be considered a
      constant. However, for series expansion both the cases need similar
      treatment (as the user does not care about inner details), i.e, use an
      addition formula to separate the constant part and the variable part (see
      rs_sin for reference).

    - All the algorithms used here are primarily designed to work for Taylor
      series (number of iterations in the algo equals the required order).
      Hence, it becomes tricky to get the series of the right order if a
      Puiseux series is input. Use rs_puiseux? in your function if your
      algorithm is not designed to handle fractional powers.

Extending rs_series
-------------------

To make a function work with rs_series you need to do two things::

    - Many sure it works with a constant term (as explained above).
    - If the series contains constant terms, you might need to extend its ring.
      You do so by adding the new terms to the rings as generators.
      ``PolyRing.compose`` and ``PolyRing.add_gens`` are two functions that do
      so and need to be called every time you expand a series containing a
      constant term.

Look at rs_sin and rs_series for further reference.

"""

from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,
                                   monomial_ldiv)
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational
from sympy.core.intfunc import igcd
from sympy.functions import sin, cos, tan, atan, exp, atanh, tanh, log, ceiling
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math


def _invert_monoms(p1):
    """
    Compute ``x**n * p1(1/x)`` for a univariate polynomial ``p1`` in ``x``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _invert_monoms
    >>> R, x = ring('x', ZZ)
    >>> p = x**2 + 2*x + 3
    >>> _invert_monoms(p)
    3*x**2 + 2*x + 1

    See Also
    ========

    sympy.polys.densebasic.dup_reverse
    """
    terms = list(p1.items())
    terms.sort()
    deg = p1.degree()
    R = p1.ring
    p = R.zero
    cv = p1.listcoeffs()
    mv = p1.listmonoms()
    for mvi, cvi in zip(mv, cv):
        p[(deg - mvi[0],)] = cvi
    return p

def _giant_steps(target):
    """Return a list of precision steps for the Newton's method"""
    res = giant_steps(2, target)
    if res[0] != 2:
        res = [2] + res
    return res

def rs_trunc(p1, x, prec):
    """
    Truncate the series in the ``x`` variable with precision ``prec``,
    that is, modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_trunc
    >>> R, x = ring('x', QQ)
    >>> p = x**10 + x**5 + x + 1
    >>> rs_trunc(p, x, 12)
    x**10 + x**5 + x + 1
    >>> rs_trunc(p, x, 10)
    x**5 + x + 1
    """
    R = p1.ring
    p = R.zero
    i = R.gens.index(x)
    for exp1 in p1:
        if exp1[i] >= prec:
            continue
        p[exp1] = p1[exp1]
    return p

def rs_is_puiseux(p, x):
    """
    Test if ``p`` is Puiseux series in ``x``.

    Raise an exception if it has a negative power in ``x``.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_is_puiseux
    >>> R, x = ring('x', QQ)
    >>> p = x**QQ(2,5) + x**QQ(2,3) + x
    >>> rs_is_puiseux(p, x)
    True
    """
    index = p.ring.gens.index(x)
    for k in p:
        if k[index] != int(k[index]):
            return True
        if k[index] < 0:
            raise ValueError('The series is not regular in %s' % x)
    return False

def rs_puiseux(f, p, x, prec):
    """
    Return the puiseux series for `f(p, x, prec)`.

    To be used when function ``f`` is implemented only for regular series.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_puiseux, rs_exp
    >>> R, x = ring('x', QQ)
    >>> p = x**QQ(2,5) + x**QQ(2,3) + x
    >>> rs_puiseux(rs_exp,p, x, 1)
    1/2*x**(4/5) + x**(2/3) + x**(2/5) + 1
    """
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            num, den = power.as_numer_denom()
            n = int(n*den // igcd(n, den))
        elif power != int(power):
            den = power.denominator
            n = int(n*den // igcd(n, den))
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, x, prec*n)
        n1 = QQ(1, n)
        if isinstance(r, tuple):
            r = tuple([pow_xin(rx, index, n1) for rx in r])
        else:
            r = pow_xin(r, index, n1)
    else:
        r = f(p, x, prec)
    return r

def rs_puiseux2(f, p, q, x, prec):
    """
    Return the puiseux series for `f(p, q, x, prec)`.

    To be used when function ``f`` is implemented only for regular series.
    """
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            num, den = power.as_numer_denom()
            n = n*den // igcd(n, den)
        elif power != int(power):
            den = power.denominator
            n = n*den // igcd(n, den)
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, q, x, prec*n)
        n1 = QQ(1, n)
        r = pow_xin(r, index, n1)
    else:
        r = f(p, q, x, prec)
    return r

def rs_mul(p1, p2, x, prec):
    """
    Return the product of the given two series, modulo ``O(x**prec)``.

    ``x`` is the series variable or its position in the generators.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_mul
    >>> R, x = ring('x', QQ)
    >>> p1 = x**2 + 2*x + 1
    >>> p2 = x + 1
    >>> rs_mul(p1, p2, x, 3)
    3*x**2 + 3*x + 1
    """
    R = p1.ring
    p = R.zero
    if R.__class__ != p2.ring.__class__ or R != p2.ring:
        raise ValueError('p1 and p2 must have the same ring')
    iv = R.gens.index(x)
    if not isinstance(p2, PolyElement):
        raise ValueError('p2 must be a polynomial')
    if R == p2.ring:
        get = p.get
        items2 = list(p2.items())
        items2.sort(key=lambda e: e[0][iv])
        if R.ngens == 1:
            for exp1, v1 in p1.items():
                for exp2, v2 in items2:
                    exp = exp1[0] + exp2[0]
                    if exp < prec:
                        exp = (exp, )
                        p[exp] = get(exp, 0) + v1*v2
                    else:
                        break
        else:
            monomial_mul = R.monomial_mul
            for exp1, v1 in p1.items():
                for exp2, v2 in items2:
                    if exp1[iv] + exp2[iv] < prec:
                        exp = monomial_mul(exp1, exp2)
                        p[exp] = get(exp, 0) + v1*v2
                    else:
                        break

    p.strip_zero()
    return p

def rs_square(p1, x, prec):
    """
    Square the series modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_square
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + 2*x + 1
    >>> rs_square(p, x, 3)
    6*x**2 + 4*x + 1
    """
    R = p1.ring
    p = R.zero
    iv = R.gens.index(x)
    get = p.get
    items = list(p1.items())
    items.sort(key=lambda e: e[0][iv])
    monomial_mul = R.monomial_mul
    for i in range(len(items)):
        exp1, v1 = items[i]
        for j in range(i):
            exp2, v2 = items[j]
            if exp1[iv] + exp2[iv] < prec:
                exp = monomial_mul(exp1, exp2)
                p[exp] = get(exp, 0) + v1*v2
            else:
                break
    p = p.imul_num(2)
    get = p.get
    for expv, v in p1.items():
        if 2*expv[iv] < prec:
            e2 = monomial_mul(expv, expv)
            p[e2] = get(e2, 0) + v**2
    p.strip_zero()
    return p

def rs_pow(p1, n, x, prec):
    """
    Return ``p1**n`` modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_pow
    >>> R, x = ring('x', QQ)
    >>> p = x + 1
    >>> rs_pow(p, 4, x, 3)
    6*x**2 + 4*x + 1
    """
    R = p1.ring
    if isinstance(n, Rational):
        np = int(n.p)
        nq = int(n.q)
        if nq != 1:
            res = rs_nth_root(p1, nq, x, prec)
            if np != 1:
                res = rs_pow(res, np, x, prec)
        else:
            res = rs_pow(p1, np, x, prec)
        return res

    n = as_int(n)
    if n == 0:
        if p1:
            return R(1)
        else:
            raise ValueError('0**0 is undefined')
    if n < 0:
        p1 = rs_pow(p1, -n, x, prec)
        return rs_series_inversion(p1, x, prec)
    if n == 1:
        return rs_trunc(p1, x, prec)
    if n == 2:
        return rs_square(p1, x, prec)
    if n == 3:
        p2 = rs_square(p1, x, prec)
        return rs_mul(p1, p2, x, prec)
    p = R(1)
    while 1:
        if n & 1:
            p = rs_mul(p1, p, x, prec)
            n -= 1
            if not n:
                break
        p1 = rs_square(p1, x, prec)
        n = n // 2
    return p

def rs_subs(p, rules, x, prec):
    """
    Substitution with truncation according to the mapping in ``rules``.

    Return a series with precision ``prec`` in the generator ``x``

    Note that substitutions are not done one after the other

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_subs
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x**2 + y**2
    >>> rs_subs(p, {x: x+ y, y: x+ 2*y}, x, 3)
    2*x**2 + 6*x*y + 5*y**2
    >>> (x + y)**2 + (x + 2*y)**2
    2*x**2 + 6*x*y + 5*y**2

    which differs from

    >>> rs_subs(rs_subs(p, {x: x+ y}, x, 3), {y: x+ 2*y}, x, 3)
    5*x**2 + 12*x*y + 8*y**2

    Parameters
    ----------
    p : :class:`~.PolyElement` Input series.
    rules : ``dict`` with substitution mappings.
    x : :class:`~.PolyElement` in which the series truncation is to be done.
    prec : :class:`~.Integer` order of the series after truncation.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_subs
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_subs(x**2+y**2, {y: (x+y)**2}, x, 3)
     6*x**2*y**2 + x**2 + 4*x*y**3 + y**4
    """
    R = p.ring
    ngens = R.ngens
    d = R(0)
    for i in range(ngens):
        d[(i, 1)] = R.gens[i]
    for var in rules:
        d[(R.index(var), 1)] = rules[var]
    p1 = R(0)
    p_keys = sorted(p.keys())
    for expv in p_keys:
        p2 = R(1)
        for i in range(ngens):
            power = expv[i]
            if power == 0:
                continue
            if (i, power) not in d:
                q, r = divmod(power, 2)
                if r == 0 and (i, q) in d:
                    d[(i, power)] = rs_square(d[(i, q)], x, prec)
                elif (i, power - 1) in d:
                    d[(i, power)] = rs_mul(d[(i, power - 1)], d[(i, 1)],
                                           x, prec)
                else:
                    d[(i, power)] = rs_pow(d[(i, 1)], power, x, prec)
            p2 = rs_mul(p2, d[(i, power)], x, prec)
        p1 += p2*p[expv]
    return p1

def _has_constant_term(p, x):
    """
    Check if ``p`` has a constant term in ``x``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _has_constant_term
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + x + 1
    >>> _has_constant_term(p, x)
    True
    """
    R = p.ring
    iv = R.gens.index(x)
    zm = R.zero_monom
    a = [0]*R.ngens
    a[iv] = 1
    miv = tuple(a)
    for expv in p:
        if monomial_min(expv, miv) == zm:
            return True
    return False

def _get_constant_term(p, x):
    """Return constant term in p with respect to x

    Note that it is not simply `p[R.zero_monom]` as there might be multiple
    generators in the ring R. We want the `x`-free term which can contain other
    generators.
    """
    R = p.ring
    i = R.gens.index(x)
    zm = R.zero_monom
    a = [0]*R.ngens
    a[i] = 1
    miv = tuple(a)
    c = 0
    for expv in p:
        if monomial_min(expv, miv) == zm:
            c += R({expv: p[expv]})
    return c

def _check_series_var(p, x, name):
    index = p.ring.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m < 0:
        raise PoleError("Asymptotic expansion of %s around [oo] not "
                        "implemented." % name)
    return index, m

def _series_inversion1(p, x, prec):
    """
    Univariate series inversion ``1/p`` modulo ``O(x**prec)``.

    The Newton method is used.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _series_inversion1
    >>> R, x = ring('x', QQ)
    >>> p = x + 1
    >>> _series_inversion1(p, x, 4)
    -x**3 + x**2 - x + 1
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(_series_inversion1, p, x, prec)
    R = p.ring
    zm = R.zero_monom
    c = p[zm]

    # giant_steps does not seem to work with PythonRational numbers with 1 as
    # denominator. This makes sure such a number is converted to integer.
    if prec == int(prec):
        prec = int(prec)

    if zm not in p:
        raise ValueError("No constant term in series")
    if _has_constant_term(p - c, x):
        raise ValueError("p cannot contain a constant term depending on "
                         "parameters")
    one = R(1)
    if R.domain is EX:
        one = 1
    if c != one:
        # TODO add check that it is a unit
        p1 = R(1)/c
    else:
        p1 = R(1)
    for precx in _giant_steps(prec):
        t = 1 - rs_mul(p1, p, x, precx)
        p1 = p1 + rs_mul(p1, t, x, precx)
    return p1

def rs_series_inversion(p, x, prec):
    """
    Multivariate series inversion ``1/p`` modulo ``O(x**prec)``.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_inversion
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_series_inversion(1 + x*y**2, x, 4)
    -x**3*y**6 + x**2*y**4 - x*y**2 + 1
    >>> rs_series_inversion(1 + x*y**2, y, 4)
    -x*y**2 + 1
    >>> rs_series_inversion(x + x**2, x, 4)
    x**3 - x**2 + x - 1 + x**(-1)
    """
    R = p.ring
    if p == R.zero:
        raise ZeroDivisionError
    zm = R.zero_monom
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m:
        p = mul_xin(p, index, -m)
        prec = prec + m
    if zm not in p:
        raise NotImplementedError("No constant term in series")

    if _has_constant_term(p - p[zm], x):
        raise NotImplementedError("p - p[0] must not have a constant term in "
                                  "the series variables")
    r = _series_inversion1(p, x, prec)
    if m != 0:
        r = mul_xin(r, index, -m)
    return r

def _coefficient_t(p, t):
    r"""Coefficient of `x_i**j` in p, where ``t`` = (i, j)"""
    i, j = t
    R = p.ring
    expv1 = [0]*R.ngens
    expv1[i] = j
    expv1 = tuple(expv1)
    p1 = R(0)
    for expv in p:
        if expv[i] == j:
            p1[monomial_div(expv, expv1)] = p[expv]
    return p1

def rs_series_reversion(p, x, n, y):
    r"""
    Reversion of a series.

    ``p`` is a series with ``O(x**n)`` of the form $p = ax + f(x)$
    where $a$ is a number different from 0.

    $f(x) = \sum_{k=2}^{n-1} a_kx_k$

    Parameters
    ==========

      a_k : Can depend polynomially on other variables, not indicated.
      x : Variable with name x.
      y : Variable with name y.

    Returns
    =======

    Solve $p = y$, that is, given $ax + f(x) - y = 0$,
    find the solution $x = r(y)$ up to $O(y^n)$.

    Algorithm
    =========

    If $r_i$ is the solution at order $i$, then:
    $ar_i + f(r_i) - y = O\left(y^{i + 1}\right)$

    and if $r_{i + 1}$ is the solution at order $i + 1$, then:
    $ar_{i + 1} + f(r_{i + 1}) - y = O\left(y^{i + 2}\right)$

    We have, $r_{i + 1} = r_i + e$, such that,
    $ae + f(r_i) = O\left(y^{i + 2}\right)$
    or $e = -f(r_i)/a$

    So we use the recursion relation:
    $r_{i + 1} = r_i - f(r_i)/a$
    with the boundary condition: $r_1 = y$

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_reversion, rs_trunc
    >>> R, x, y, a, b = ring('x, y, a, b', QQ)
    >>> p = x - x**2 - 2*b*x**2 + 2*a*b*x**2
    >>> p1 = rs_series_reversion(p, x, 3, y); p1
    -2*y**2*a*b + 2*y**2*b + y**2 + y
    >>> rs_trunc(p.compose(x, p1), y, 3)
    y
    """
    if rs_is_puiseux(p, x):
        raise NotImplementedError
    R = p.ring
    nx = R.gens.index(x)
    y = R(y)
    ny = R.gens.index(y)
    if _has_constant_term(p, x):
        raise ValueError("p must not contain a constant term in the series "
                         "variable")
    a = _coefficient_t(p, (nx, 1))
    zm = R.zero_monom
    assert zm in a and len(a) == 1
    a = a[zm]
    r = y/a
    for i in range(2, n):
        sp = rs_subs(p, {x: r}, y, i + 1)
        sp = _coefficient_t(sp, (ny, i))*y**i
        r -= sp/a
    return r

def rs_series_from_list(p, c, x, prec, concur=1):
    """
    Return a series `sum c[n]*p**n` modulo `O(x**prec)`.

    It reduces the number of multiplications by summing concurrently.

    `ax = [1, p, p**2, .., p**(J - 1)]`
    `s = sum(c[i]*ax[i]` for i in `range(r, (r + 1)*J))*p**((K - 1)*J)`
    with `K >= (n + 1)/J`

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_from_list, rs_trunc
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + x + 1
    >>> c = [1, 2, 3]
    >>> rs_series_from_list(p, c, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> rs_trunc(1 + 2*p + 3*p**2, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> pc = R.from_list(list(reversed(c)))
    >>> rs_trunc(pc.compose(x, p), x, 4)
    6*x**3 + 11*x**2 + 8*x + 6

    """

    # TODO: Add this when it is documented in Sphinx
    """
    See Also
    ========

    sympy.polys.rings.PolyRing.compose

    """
    R = p.ring
    n = len(c)
    if not concur:
        q = R(1)
        s = c[0]*q
        for i in range(1, n):
            q = rs_mul(q, p, x, prec)
            s += c[i]*q
        return s
    J = int(math.sqrt(n) + 1)
    K, r = divmod(n, J)
    if r:
        K += 1
    ax = [R(1)]
    q = R(1)
    if len(p) < 20:
        for i in range(1, J):
            q = rs_mul(q, p, x, prec)
            ax.append(q)
    else:
        for i in range(1, J):
            if i % 2 == 0:
                q = rs_square(ax[i//2], x, prec)
            else:
                q = rs_mul(q, p, x, prec)
            ax.append(q)
    # optimize using rs_square
    pj = rs_mul(ax[-1], p, x, prec)
    b = R(1)
    s = R(0)
    for k in range(K - 1):
        r = J*k
        s1 = c[r]
        for j in range(1, J):
            s1 += c[r + j]*ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
        b = rs_mul(b, pj, x, prec)
        if not b:
            break
    k = K - 1
    r = J*k
    if r < n:
        s1 = c[r]*R(1)
        for j in range(1, J):
            if r + j >= n:
                break
            s1 += c[r + j]*ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
    return s

def rs_diff(p, x):
    """
    Return partial derivative of ``p`` with respect to ``x``.

    Parameters
    ==========

    x : :class:`~.PolyElement` with respect to which ``p`` is differentiated.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_diff
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x**2*y**3
    >>> rs_diff(p, x)
    2*x*y**3 + 1
    """
    R = p.ring
    n = R.gens.index(x)
    p1 = R.zero
    mn = [0]*R.ngens
    mn[n] = 1
    mn = tuple(mn)
    for expv in p:
        if expv[n]:
            e = monomial_ldiv(expv, mn)
            p1[e] = R.domain_new(p[expv]*expv[n])
    return p1

def rs_integrate(p, x):
    """
    Integrate ``p`` with respect to ``x``.

    Parameters
    ==========

    x : :class:`~.PolyElement` with respect to which ``p`` is integrated.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_integrate
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x**2*y**3
    >>> rs_integrate(p, x)
    1/3*x**3*y**3 + 1/2*x**2
    """
    R = p.ring
    p1 = R.zero
    n = R.gens.index(x)
    mn = [0]*R.ngens
    mn[n] = 1
    mn = tuple(mn)

    for expv in p:
        e = monomial_mul(expv, mn)
        p1[e] = R.domain_new(p[expv]/(expv[n] + 1))
    return p1

def rs_fun(p, f, *args):
    r"""
    Function of a multivariate series computed by substitution.

    The case with f method name is used to compute `rs\_tan` and `rs\_nth\_root`
    of a multivariate series:

        `rs\_fun(p, tan, iv, prec)`

        tan series is first computed for a dummy variable _x,
        i.e, `rs\_tan(\_x, iv, prec)`. Then we substitute _x with p to get the
        desired series

    Parameters
    ==========

    p : :class:`~.PolyElement` The multivariate series to be expanded.
    f : `ring\_series` function to be applied on `p`.
    args[-2] : :class:`~.PolyElement` with respect to which, the series is to be expanded.
    args[-1] : Required order of the expanded series.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_fun, _tan1
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x*y + x**2*y + x**3*y**2
    >>> rs_fun(p, _tan1, x, 4)
    1/3*x**3*y**3 + 2*x**3*y**2 + x**3*y + 1/3*x**3 + x**2*y + x*y + x
    """
    _R = p.ring
    R1, _x = ring('_x', _R.domain)
    h = int(args[-1])
    args1 = args[:-2] + (_x, h)
    zm = _R.zero_monom
    # separate the constant term of the series
    # compute the univariate series f(_x, .., 'x', sum(nv))
    if zm in p:
        x1 = _x + p[zm]
        p1 = p - p[zm]
    else:
        x1 = _x
        p1 = p
    if isinstance(f, str):
        q = getattr(x1, f)(*args1)
    else:
        q = f(x1, *args1)
    a = sorted(q.items())
    c = [0]*h
    for x in a:
        c[x[0][0]] = x[1]
    p1 = rs_series_from_list(p1, c, args[-2], args[-1])
    return p1

def mul_xin(p, i, n):
    r"""
    Return `p*x_i**n`.

    `x\_i` is the ith variable in ``p``.
    """
    R = p.ring
    q = R(0)
    for k, v in p.items():
        k1 = list(k)
        k1[i] += n
        q[tuple(k1)] = v
    return q

def pow_xin(p, i, n):
    """
    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import pow_xin
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x**QQ(2,5) + x + x**QQ(2,3)
    >>> index = p.ring.gens.index(x)
    >>> pow_xin(p, index, 15)
    x**15 + x**10 + x**6
    """
    R = p.ring
    q = R(0)
    for k, v in p.items():
        k1 = list(k)
        k1[i] *= n
        q[tuple(k1)] = v
    return q

def _nth_root1(p, n, x, prec):
    """
    Univariate series expansion of the nth root of ``p``.

    The Newton method is used.
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux2(_nth_root1, p, n, x, prec)
    R = p.ring
    zm = R.zero_monom
    if zm not in p:
        raise NotImplementedError('No constant term in series')
    n = as_int(n)
    assert p[zm] == 1
    p1 = R(1)
    if p == 1:
        return p
    if n == 0:
        return R(1)
    if n == 1:
        return p
    if n < 0:
        n = -n
        sign = 1
    else:
        sign = 0
    for precx in _giant_steps(prec):
        tmp = rs_pow(p1, n + 1, x, precx)
        tmp = rs_mul(tmp, p, x, precx)
        p1 += p1/n - tmp/n
    if sign:
        return p1
    else:
        return _series_inversion1(p1, x, prec)

def rs_nth_root(p, n, x, prec):
    """
    Multivariate series expansion of the nth root of ``p``.

    Parameters
    ==========

    p : Expr
        The polynomial to computer the root of.
    n : integer
        The order of the root to be computed.
    x : :class:`~.PolyElement`
    prec : integer
        Order of the expanded series.

    Notes
    =====

    The result of this function is dependent on the ring over which the
    polynomial has been defined. If the answer involves a root of a constant,
    make sure that the polynomial is over a real field. It cannot yet handle
    roots of symbols.

    Examples
    ========

    >>> from sympy.polys.domains import QQ, RR
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_nth_root
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_nth_root(1 + x + x*y, -3, x, 3)
    2/9*x**2*y**2 + 4/9*x**2*y + 2/9*x**2 - 1/3*x*y - 1/3*x + 1
    >>> R, x, y = ring('x, y', RR)
    >>> rs_nth_root(3 + x + x*y, 3, x, 2)
    0.160249952256379*x*y + 0.160249952256379*x + 1.44224957030741
    """
    if n == 0:
        if p == 0:
            raise ValueError('0**0 expression')
        else:
            return p.ring(1)
    if n == 1:
        return rs_trunc(p, x, prec)
    R = p.ring
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    p = mul_xin(p, index, -m)
    prec -= m

    if _has_constant_term(p - 1, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = c_expr**QQ(1, n)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(c_expr**(QQ(1, n)))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        else:
            try:                              # RealElement doesn't support
                const = R(c**Rational(1, n))  # exponentiation with mpq object
            except ValueError:                # as exponent
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        res = rs_nth_root(p/c, n, x, prec)*const
    else:
        res = _nth_root1(p, n, x, prec)
    if m:
        m = QQ(m, n)
        res = mul_xin(res, index, m)
    return res

def rs_log(p, x, prec):
    """
    The Logarithm of ``p`` modulo ``O(x**prec)``.

    Notes
    =====

    Truncation of ``integral dx p**-1*d p/dx`` is used.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_log
    >>> R, x = ring('x', QQ)
    >>> rs_log(1 + x, x, 8)
    1/7*x**7 - 1/6*x**6 + 1/5*x**5 - 1/4*x**4 + 1/3*x**3 - 1/2*x**2 + x
    >>> rs_log(x**QQ(3, 2) + 1, x, 5)
    1/3*x**(9/2) - 1/2*x**3 + x**(3/2)
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_log, p, x, prec)
    R = p.ring
    if p == 1:
        return R.zero
    c = _get_constant_term(p, x)
    if c:
        const = 0
        if c == 1:
            pass
        else:
            c_expr = c.as_expr()
            if R.domain is EX:
                const = log(c_expr)
            elif isinstance(c, PolyElement):
                try:
                    const = R(log(c_expr))
                except ValueError:
                    R = R.add_gens([log(c_expr)])
                    p = p.set_ring(R)
                    x = x.set_ring(R)
                    c = c.set_ring(R)
                    const = R(log(c_expr))
            else:
                try:
                    const = R(log(c))
                except ValueError:
                    raise DomainError("The given series cannot be expanded in "
                        "this domain.")

        dlog = p.diff(x)
        dlog = rs_mul(dlog, _series_inversion1(p, x, prec), x, prec - 1)
        return rs_integrate(dlog, x) + const
    else:
        raise NotImplementedError

def rs_LambertW(p, x, prec):
    """
    Calculate the series expansion of the principal branch of the Lambert W
    function.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_LambertW
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_LambertW(x + x*y, x, 3)
    -x**2*y**2 - 2*x**2*y - x**2 + x*y + x

    See Also
    ========

    LambertW
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_LambertW, p, x, prec)
    R = p.ring
    p1 = R(0)
    if _has_constant_term(p, x):
        raise NotImplementedError("Polynomial must not have constant term in "
                                  "the series variables")
    if x in R.gens:
        for precx in _giant_steps(prec):
            e = rs_exp(p1, x, precx)
            p2 = rs_mul(e, p1, x, precx) - p
            p3 = rs_mul(e, p1 + 1, x, precx)
            p3 = rs_series_inversion(p3, x, precx)
            tmp = rs_mul(p2, p3, x, precx)
            p1 -= tmp
        return p1
    else:
        raise NotImplementedError

def _exp1(p, x, prec):
    r"""Helper function for `rs\_exp`. """
    R = p.ring
    p1 = R(1)
    for precx in _giant_steps(prec):
        pt = p - rs_log(p1, x, precx)
        tmp = rs_mul(pt, p1, x, precx)
        p1 += tmp
    return p1

def rs_exp(p, x, prec):
    """
    Exponentiation of a series modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_exp
    >>> R, x = ring('x', QQ)
    >>> rs_exp(x**2, x, 7)
    1/6*x**6 + 1/2*x**4 + x**2 + 1
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_exp, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            const = exp(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(exp(c_expr))
            except ValueError:
                R = R.add_gens([exp(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                const = R(exp(c_expr))
        else:
            try:
                const = R(exp(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        p1 = p - c

    # Makes use of SymPy functions to evaluate the values of the cos/sin
    # of the constant term.
        return const*rs_exp(p1, x, prec)

    if len(p) > 20:
        return _exp1(p, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(prec):
        c.append(one/n)
        k += 1
        n *= k

    r = rs_series_from_list(p, c, x, prec)
    return r

def _atan(p, iv, prec):
    """
    Expansion using formula.

    Faster on very small and univariate series.
    """
    R = p.ring
    mo = R(-1)
    c = [-mo]
    p2 = rs_square(p, iv, prec)
    for k in range(1, prec):
        c.append(mo**k/(2*k + 1))
    s = rs_series_from_list(p2, c, iv, prec)
    s = rs_mul(s, p, iv, prec)
    return s

def rs_atan(p, x, prec):
    """
    The arctangent of a series

    Return the series expansion of the atan of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_atan
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_atan(x + x*y, x, 4)
    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x

    See Also
    ========

    atan
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atan, p, x, prec)
    R = p.ring
    const = 0
    if _has_constant_term(p, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = atan(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(atan(c_expr))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        else:
            try:
                const = R(atan(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")

    # Instead of using a closed form formula, we differentiate atan(p) to get
    # `1/(1+p**2) * dp`, whose series expansion is much easier to calculate.
    # Finally we integrate to get back atan
    dp = p.diff(x)
    p1 = rs_square(p, x, prec) + R(1)
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_asin(p, x, prec):
    """
    Arcsine of a series

    Return the series expansion of the asin of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_asin
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_asin(x, x, 8)
    5/112*x**7 + 3/40*x**5 + 1/6*x**3 + x

    See Also
    ========

    asin
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_asin, p, x, prec)
    if _has_constant_term(p, x):
        raise NotImplementedError("Polynomial must not have constant term in "
                                  "series variables")
    R = p.ring
    if x in R.gens:
        # get a good value
        if len(p) > 20:
            dp = rs_diff(p, x)
            p1 = 1 - rs_square(p, x, prec - 1)
            p1 = rs_nth_root(p1, -2, x, prec - 1)
            p1 = rs_mul(dp, p1, x, prec - 1)
            return rs_integrate(p1, x)
        one = R(1)
        c = [0, one, 0]
        for k in range(3, prec, 2):
            c.append((k - 2)**2*c[-2]/(k*(k - 1)))
            c.append(0)
        return rs_series_from_list(p, c, x, prec)

    else:
        raise NotImplementedError

def _tan1(p, x, prec):
    r"""
    Helper function of :func:`rs_tan`.

    Return the series expansion of tan of a univariate series using Newton's
    method. It takes advantage of the fact that series expansion of atan is
    easier than that of tan.

    Consider `f(x) = y - \arctan(x)`
    Let r be a root of f(x) found using Newton's method.
    Then `f(r) = 0`
    Or `y = \arctan(x)` where `x = \tan(y)` as required.
    """
    R = p.ring
    p1 = R(0)
    for precx in _giant_steps(prec):
        tmp = p - rs_atan(p1, x, precx)
        tmp = rs_mul(tmp, 1 + rs_square(p1, x, precx), x, precx)
        p1 += tmp
    return p1

def rs_tan(p, x, prec):
    """
    Tangent of a series.

    Return the series expansion of the tan of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_tan
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_tan(x + x*y, x, 4)
    1/3*x**3*y**3 + x**3*y**2 + x**3*y + 1/3*x**3 + x*y + x

   See Also
   ========

   _tan1, tan
   """
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_tan, p, x, prec)
        return r
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            const = tan(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(tan(c_expr))
            except ValueError:
                R = R.add_gens([tan(c_expr, )])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                const = R(tan(c_expr))
        else:
            try:
                const = R(tan(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        p1 = p - c

    # Makes use of SymPy functions to evaluate the values of the cos/sin
    # of the constant term.
        t2 = rs_tan(p1, x, prec)
        t = rs_series_inversion(1 - const*t2, x, prec)
        return rs_mul(const + t2, t, x, prec)

    if R.ngens == 1:
        return _tan1(p, x, prec)
    else:
        return rs_fun(p, rs_tan, x, prec)

def rs_cot(p, x, prec):
    """
    Cotangent of a series

    Return the series expansion of the cot of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cot
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_cot(x, x, 6)
    -2/945*x**5 - 1/45*x**3 - 1/3*x + x**(-1)

    See Also
    ========

    cot
    """
    # It can not handle series like `p = x + x*y` where the coefficient of the
    # linear term in the series variable is symbolic.
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_cot, p, x, prec)
        return r
    i, m = _check_series_var(p, x, 'cot')
    prec1 = prec + 2*m
    c, s = rs_cos_sin(p, x, prec1)
    s = mul_xin(s, i, -m)
    s = rs_series_inversion(s, x, prec1)
    res = rs_mul(c, s, x, prec1)
    res = mul_xin(res, i, -m)
    res = rs_trunc(res, x, prec)
    return res

def rs_sin(p, x, prec):
    """
    Sine of a series

    Return the series expansion of the sin of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_sin
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_sin(x + x*y, x, 4)
    -1/6*x**3*y**3 - 1/2*x**3*y**2 - 1/2*x**3*y - 1/6*x**3 + x*y + x
    >>> rs_sin(x**QQ(3, 2) + x*y**QQ(7, 5), x, 4)
    -1/2*x**(7/2)*y**(14/5) - 1/6*x**3*y**(21/5) + x**(3/2) + x*y**(7/5)

    See Also
    ========

    sin
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sin, p, x, prec)
    R = x.ring
    if not p:
        return R(0)
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            t1, t2 = sin(c_expr), cos(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                t1, t2 = R(sin(c_expr)), R(cos(c_expr))
            except ValueError:
                R = R.add_gens([sin(c_expr), cos(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        else:
            try:
                t1, t2 = R(sin(c)), R(cos(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        p1 = p - c

    # Makes use of SymPy cos, sin functions to evaluate the values of the
    # cos/sin of the constant term.
        return rs_sin(p1, x, prec)*t2 + rs_cos(p1, x, prec)*t1

    # Series is calculated in terms of tan as its evaluation is fast.
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return rs_mul(p1, 2*t, x, prec)
    one = R(1)
    n = 1
    c = [0]
    for k in range(2, prec + 2, 2):
        c.append(one/n)
        c.append(0)
        n *= -k*(k + 1)
    return rs_series_from_list(p, c, x, prec)

def rs_cos(p, x, prec):
    """
    Cosine of a series

    Return the series expansion of the cos of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cos
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_cos(x + x*y, x, 4)
    -1/2*x**2*y**2 - x**2*y - 1/2*x**2 + 1
    >>> rs_cos(x + x*y, x, 4)/x**QQ(7, 5)
    -1/2*x**(3/5)*y**2 - x**(3/5)*y - 1/2*x**(3/5) + x**(-7/5)

    See Also
    ========

    cos
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            _, _ = sin(c_expr), cos(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                _, _ = R(sin(c_expr)), R(cos(c_expr))
            except ValueError:
                R = R.add_gens([sin(c_expr), cos(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
        else:
            try:
                _, _ = R(sin(c)), R(cos(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        p1 = p - c

    # Makes use of SymPy cos, sin functions to evaluate the values of the
    # cos/sin of the constant term.
        p_cos = rs_cos(p1, x, prec)
        p_sin = rs_sin(p1, x, prec)
        R = R.compose(p_cos.ring).compose(p_sin.ring)
        p_cos.set_ring(R)
        p_sin.set_ring(R)
        t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        return p_cos*t2 - p_sin*t1

    # Series is calculated in terms of tan as its evaluation is fast.
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1+t2, x, prec)
        return rs_mul(p1, 1 - t2, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(2, prec + 2, 2):
        c.append(one/n)
        c.append(0)
        n *= -k*(k - 1)
    return rs_series_from_list(p, c, x, prec)

def rs_cos_sin(p, x, prec):
    r"""
    Return the tuple ``(rs_cos(p, x, prec)`, `rs_sin(p, x, prec))``.

    Is faster than calling rs_cos and rs_sin separately
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos_sin, p, x, prec)
    t = rs_tan(p/2, x, prec)
    t2 = rs_square(t, x, prec)
    p1 = rs_series_inversion(1 + t2, x, prec)
    return (rs_mul(p1, 1 - t2, x, prec), rs_mul(p1, 2*t, x, prec))

def _atanh(p, x, prec):
    """
    Expansion using formula

    Faster for very small and univariate series
    """
    R = p.ring
    one = R(1)
    c = [one]
    p2 = rs_square(p, x, prec)
    for k in range(1, prec):
        c.append(one/(2*k + 1))
    s = rs_series_from_list(p2, c, x, prec)
    s = rs_mul(s, p, x, prec)
    return s

def rs_atanh(p, x, prec):
    """
    Hyperbolic arctangent of a series

    Return the series expansion of the atanh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_atanh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_atanh(x + x*y, x, 4)
    1/3*x**3*y**3 + x**3*y**2 + x**3*y + 1/3*x**3 + x*y + x

    See Also
    ========

    atanh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atanh, p, x, prec)
    R = p.ring
    const = 0
    if _has_constant_term(p, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = atanh(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(atanh(c_expr))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        else:
            try:
                const = R(atanh(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")

    # Instead of using a closed form formula, we differentiate atanh(p) to get
    # `1/(1-p**2) * dp`, whose series expansion is much easier to calculate.
    # Finally we integrate to get back atanh
    dp = rs_diff(p, x)
    p1 = - rs_square(p, x, prec) + 1
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_sinh(p, x, prec):
    """
    Hyperbolic sine of a series

    Return the series expansion of the sinh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_sinh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_sinh(x + x*y, x, 4)
    1/6*x**3*y**3 + 1/2*x**3*y**2 + 1/2*x**3*y + 1/6*x**3 + x*y + x

    See Also
    ========

    sinh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sinh, p, x, prec)
    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t - t1)/2

def rs_cosh(p, x, prec):
    """
    Hyperbolic cosine of a series

    Return the series expansion of the cosh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cosh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_cosh(x + x*y, x, 4)
    1/2*x**2*y**2 + x**2*y + 1/2*x**2 + 1

    See Also
    ========

    cosh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cosh, p, x, prec)
    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t + t1)/2

def _tanh(p, x, prec):
    r"""
    Helper function of :func:`rs_tanh`

    Return the series expansion of tanh of a univariate series using Newton's
    method. It takes advantage of the fact that series expansion of atanh is
    easier than that of tanh.

    See Also
    ========

    _tanh
    """
    R = p.ring
    p1 = R(0)
    for precx in _giant_steps(prec):
        tmp = p - rs_atanh(p1, x, precx)
        tmp = rs_mul(tmp, 1 - rs_square(p1, x, prec), x, precx)
        p1 += tmp
    return p1

def rs_tanh(p, x, prec):
    """
    Hyperbolic tangent of a series

    Return the series expansion of the tanh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_tanh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_tanh(x + x*y, x, 4)
    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x

    See Also
    ========

    tanh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_tanh, p, x, prec)
    R = p.ring
    const = 0
    if _has_constant_term(p, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = tanh(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(tanh(c_expr))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        else:
            try:
                const = R(tanh(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        p1 = p - c
        t1 = rs_tanh(p1, x, prec)
        t = rs_series_inversion(1 + const*t1, x, prec)
        return rs_mul(const + t1, t, x, prec)

    if R.ngens == 1:
        return _tanh(p, x, prec)
    else:
        return rs_fun(p, _tanh, x, prec)

def rs_newton(p, x, prec):
    """
    Compute the truncated Newton sum of the polynomial ``p``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_newton
    >>> R, x = ring('x', QQ)
    >>> p = x**2 - 2
    >>> rs_newton(p, x, 5)
    8*x**4 + 4*x**2 + 2
    """
    deg = p.degree()
    p1 = _invert_monoms(p)
    p2 = rs_series_inversion(p1, x, prec)
    p3 = rs_mul(p1.diff(x), p2, x, prec)
    res = deg - p3*x
    return res

def rs_hadamard_exp(p1, inverse=False):
    """
    Return ``sum f_i/i!*x**i`` from ``sum f_i*x**i``,
    where ``x`` is the first variable.

    If ``invers=True`` return ``sum f_i*i!*x**i``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_hadamard_exp
    >>> R, x = ring('x', QQ)
    >>> p = 1 + x + x**2 + x**3
    >>> rs_hadamard_exp(p)
    1/6*x**3 + 1/2*x**2 + x + 1
    """
    R = p1.ring
    if R.domain != QQ:
        raise NotImplementedError
    p = R.zero
    if not inverse:
        for exp1, v1 in p1.items():
            p[exp1] = v1/int(ifac(exp1[0]))
    else:
        for exp1, v1 in p1.items():
            p[exp1] = v1*int(ifac(exp1[0]))
    return p

def rs_compose_add(p1, p2):
    """
    compute the composed sum ``prod(p2(x - beta) for beta root of p1)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_compose_add
    >>> R, x = ring('x', QQ)
    >>> f = x**2 - 2
    >>> g = x**2 - 3
    >>> rs_compose_add(f, g)
    x**4 - 10*x**2 + 1

    References
    ==========

    .. [1] A. Bostan, P. Flajolet, B. Salvy and E. Schost
           "Fast Computation with Two Algebraic Numbers",
           (2002) Research Report 4579, Institut
           National de Recherche en Informatique et en Automatique
    """
    R = p1.ring
    x = R.gens[0]
    prec = p1.degree()*p2.degree() + 1
    np1 = rs_newton(p1, x, prec)
    np1e = rs_hadamard_exp(np1)
    np2 = rs_newton(p2, x, prec)
    np2e = rs_hadamard_exp(np2)
    np3e = rs_mul(np1e, np2e, x, prec)
    np3 = rs_hadamard_exp(np3e, True)
    np3a = (np3[(0,)] - np3)/x
    q = rs_integrate(np3a, x)
    q = rs_exp(q, x, prec)
    q = _invert_monoms(q)
    q = q.primitive()[1]
    dp = p1.degree()*p2.degree() - q.degree()
    # `dp` is the multiplicity of the zeroes of the resultant;
    # these zeroes are missed in this computation so they are put here.
    # if p1 and p2 are monic irreducible polynomials,
    # there are zeroes in the resultant
    # if and only if p1 = p2 ; in fact in that case p1 and p2 have a
    # root in common, so gcd(p1, p2) != 1; being p1 and p2 irreducible
    # this means p1 = p2
    if dp:
        q = q*x**dp
    return q


_convert_func = {
        'sin': 'rs_sin',
        'cos': 'rs_cos',
        'exp': 'rs_exp',
        'tan': 'rs_tan',
        'log': 'rs_log'
        }

def rs_min_pow(expr, series_rs, a):
    """Find the minimum power of `a` in the series expansion of expr"""
    series = 0
    n = 2
    while series == 0:
        series = _rs_series(expr, series_rs, a, n)
        n *= 2
    R = series.ring
    a = R(a)
    i = R.gens.index(a)
    return min(series, key=lambda t: t[i])[i]


def _rs_series(expr, series_rs, a, prec):
    # TODO Use _parallel_dict_from_expr instead of sring as sring is
    # inefficient. For details, read the todo in sring.
    args = expr.args
    R = series_rs.ring

    # expr does not contain any function to be expanded
    if not any(arg.has(Function) for arg in args) and not expr.is_Function:
        return series_rs

    if not expr.has(a):
        return series_rs

    elif expr.is_Function:
        arg = args[0]
        if len(args) > 1:
            raise NotImplementedError
        R1, series = sring(arg, domain=QQ, expand=False, series=True)
        series_inner = _rs_series(arg, series, a, prec)

        # Why do we need to compose these three rings?
        #
        # We want to use a simple domain (like ``QQ`` or ``RR``) but they don't
        # support symbolic coefficients. We need a ring that for example lets
        # us have `sin(1)` and `cos(1)` as coefficients if we are expanding
        # `sin(x + 1)`. The ``EX`` domain allows all symbolic coefficients, but
        # that makes it very complex and hence slow.
        #
        # To solve this problem, we add only those symbolic elements as
        # generators to our ring, that we need. Here, series_inner might
        # involve terms like `sin(4)`, `exp(a)`, etc, which are not there in
        # R1 or R. Hence, we compose these three rings to create one that has
        # the generators of all three.
        R = R.compose(R1).compose(series_inner.ring)
        series_inner = series_inner.set_ring(R)
        series = eval(_convert_func[str(expr.func)])(series_inner,
            R(a), prec)
        return series

    elif expr.is_Mul:
        n = len(args)
        for arg in args:    # XXX Looks redundant
            if not arg.is_Number:
                R1, _ = sring(arg, expand=False, series=True)
                R = R.compose(R1)
        min_pows = list(map(rs_min_pow, args, [R(arg) for arg in args],
            [a]*len(args)))
        sum_pows = sum(min_pows)
        series = R(1)

        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, prec - sum_pows +
                min_pows[i])
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series *= _series
        series = rs_trunc(series, R(a), prec)
        return series

    elif expr.is_Add:
        n = len(args)
        series = R(0)
        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, prec)
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series += _series
        return series

    elif expr.is_Pow:
        R1, _ = sring(expr.base, domain=QQ, expand=False, series=True)
        R = R.compose(R1)
        series_inner = _rs_series(expr.base, R(expr.base), a, prec)
        return rs_pow(series_inner, expr.exp, series_inner.ring(a), prec)

    # The `is_constant` method is buggy hence we check it at the end.
    # See issue #9786 for details.
    elif isinstance(expr, Expr) and expr.is_constant():
        return sring(expr, domain=QQ, expand=False, series=True)[1]

    else:
        raise NotImplementedError

def rs_series(expr, a, prec):
    """Return the series expansion of an expression about 0.

    Parameters
    ==========

    expr : :class:`Expr`
    a : :class:`Symbol` with respect to which expr is to be expanded
    prec : order of the series expansion

    Currently supports multivariate Taylor series expansion. This is much
    faster that SymPy's series method as it uses sparse polynomial operations.

    It automatically creates the simplest ring required to represent the series
    expansion through repeated calls to sring.

    Examples
    ========

    >>> from sympy.polys.ring_series import rs_series
    >>> from sympy import sin, cos, exp, tan, symbols, QQ
    >>> a, b, c = symbols('a, b, c')
    >>> rs_series(sin(a) + exp(a), a, 5)
    1/24*a**4 + 1/2*a**2 + 2*a + 1
    >>> series = rs_series(tan(a + b)*cos(a + c), a, 2)
    >>> series.as_expr()
    -a*sin(c)*tan(b) + a*cos(c)*tan(b)**2 + a*cos(c) + cos(c)*tan(b)
    >>> series = rs_series(exp(a**QQ(1,3) + a**QQ(2, 5)), a, 1)
    >>> series.as_expr()
    a**(11/15) + a**(4/5)/2 + a**(2/5) + a**(2/3)/2 + a**(1/3) + 1

    """
    R, series = sring(expr, domain=QQ, expand=False, series=True)
    if a not in R.symbols:
        R = R.add_gens([a, ])
    series = series.set_ring(R)
    series = _rs_series(expr, series, a, prec)
    R = series.ring
    gen = R(a)
    prec_got = series.degree(gen) + 1

    if prec_got >= prec:
        return rs_trunc(series, gen, prec)
    else:
        # increase the requested number of terms to get the desired
        # number keep increasing (up to 9) until the received order
        # is different than the original order and then predict how
        # many additional terms are needed
        for more in range(1, 9):
            p1 = _rs_series(expr, series, a, prec=prec + more)
            gen = gen.set_ring(p1.ring)
            new_prec = p1.degree(gen) + 1
            if new_prec != prec_got:
                prec_do = ceiling(prec + (prec - prec_got)*more/(new_prec -
                    prec_got))
                p1 = _rs_series(expr, series, a, prec=prec_do)
                while p1.degree(gen) + 1 < prec:
                    p1 = _rs_series(expr, series, a, prec=prec_do)
                    gen = gen.set_ring(p1.ring)
                    prec_do *= 2
                break
            else:
                break
        else:
            raise ValueError('Could not calculate %s terms for %s'
                             % (str(prec), expr))
        return rs_trunc(p1, gen, prec)
