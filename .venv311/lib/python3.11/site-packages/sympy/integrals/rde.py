"""
Algorithms for solving the Risch differential equation.

Given a differential field K of characteristic 0 that is a simple
monomial extension of a base field k and f, g in K, the Risch
Differential Equation problem is to decide if there exist y in K such
that Dy + f*y == g and to find one if there are some.  If t is a
monomial over k and the coefficients of f and g are in k(t), then y is
in k(t), and the outline of the algorithm here is given as:

1. Compute the normal part n of the denominator of y.  The problem is
then reduced to finding y' in k<t>, where y == y'/n.
2. Compute the special part s of the denominator of y.   The problem is
then reduced to finding y'' in k[t], where y == y''/(n*s)
3. Bound the degree of y''.
4. Reduce the equation Dy + f*y == g to a similar equation with f, g in
k[t].
5. Find the solutions in k[t] of bounded degree of the reduced equation.

See Chapter 6 of "Symbolic Integration I: Transcendental Functions" by
Manuel Bronstein.  See also the docstring of risch.py.
"""

from operator import mul
from functools import reduce

from sympy.core import oo
from sympy.core.symbol import Dummy

from sympy.polys import Poly, gcd, ZZ, cancel

from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt

from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
    splitfactor, NonElementaryIntegralException, DecrementLevel, recognize_log_derivative)

# TODO: Add messages to NonElementaryIntegralException errors


def order_at(a, p, t):
    """
    Computes the order of a at p, with respect to t.

    Explanation
    ===========

    For a, p in k[t], the order of a at p is defined as nu_p(a) = max({n
    in Z+ such that p**n|a}), where a != 0.  If a == 0, nu_p(a) = +oo.

    To compute the order at a rational function, a/b, use the fact that
    nu_p(a/b) == nu_p(a) - nu_p(b).
    """
    if a.is_zero:
        return oo
    if p == Poly(t, t):
        return a.as_poly(t).ET()[0][0]

    # Uses binary search for calculating the power. power_list collects the tuples
    # (p^k,k) where each k is some power of 2. After deciding the largest k
    # such that k is power of 2 and p^k|a the loop iteratively calculates
    # the actual power.
    power_list = []
    p1 = p
    r = a.rem(p1)
    tracks_power = 1
    while r.is_zero:
        power_list.append((p1,tracks_power))
        p1 = p1*p1
        tracks_power *= 2
        r = a.rem(p1)
    n = 0
    product = Poly(1, t)
    while len(power_list) != 0:
        final = power_list.pop()
        productf = product*final[0]
        r = a.rem(productf)
        if r.is_zero:
            n += final[1]
            product = productf
    return n


def order_at_oo(a, d, t):
    """
    Computes the order of a/d at oo (infinity), with respect to t.

    For f in k(t), the order or f at oo is defined as deg(d) - deg(a), where
    f == a/d.
    """
    if a.is_zero:
        return oo
    return d.degree(t) - a.degree(t)


def weak_normalizer(a, d, DE, z=None):
    """
    Weak normalization.

    Explanation
    ===========

    Given a derivation D on k[t] and f == a/d in k(t), return q in k[t]
    such that f - Dq/q is weakly normalized with respect to t.

    f in k(t) is said to be "weakly normalized" with respect to t if
    residue_p(f) is not a positive integer for any normal irreducible p
    in k[t] such that f is in R_p (Definition 6.1.1).  If f has an
    elementary integral, this is equivalent to no logarithm of
    integral(f) whose argument depends on t has a positive integer
    coefficient, where the arguments of the logarithms not in k(t) are
    in k[t].

    Returns (q, f - Dq/q)
    """
    z = z or Dummy('z')
    dn, ds = splitfactor(d, DE)

    # Compute d1, where dn == d1*d2**2*...*dn**n is a square-free
    # factorization of d.
    g = gcd(dn, dn.diff(DE.t))
    d_sqf_part = dn.quo(g)
    d1 = d_sqf_part.quo(gcd(d_sqf_part, g))

    a1, b = gcdex_diophantine(d.quo(d1).as_poly(DE.t), d1.as_poly(DE.t),
        a.as_poly(DE.t))
    r = (a - Poly(z, DE.t)*derivation(d1, DE)).as_poly(DE.t).resultant(
        d1.as_poly(DE.t))
    r = Poly(r, z)

    if not r.expr.has(z):
        return (Poly(1, DE.t), (a, d))

    N = [i for i in r.real_roots() if i in ZZ and i > 0]

    q = reduce(mul, [gcd(a - Poly(n, DE.t)*derivation(d1, DE), d1) for n in N],
        Poly(1, DE.t))

    dq = derivation(q, DE)
    sn = q*a - d*dq
    sd = q*d
    sn, sd = sn.cancel(sd, include=True)

    return (q, (sn, sd))


def normal_denom(fa, fd, ga, gd, DE):
    """
    Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g in k(t) with f weakly
    normalized with respect to t, either raise NonElementaryIntegralException,
    in which case the equation Dy + f*y == g has no solution in k(t), or the
    quadruplet (a, b, c, h) such that a, h in k[t], b, c in k<t>, and for any
    solution y in k(t) of Dy + f*y == g, q = y*h in k<t> satisfies
    a*Dq + b*q == c.

    This constitutes step 1 in the outline given in the rde.py docstring.
    """
    dn, ds = splitfactor(fd, DE)
    en, es = splitfactor(gd, DE)

    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))

    a = dn*h
    c = a*h
    if c.div(en)[1]:
        # en does not divide dn*h**2
        raise NonElementaryIntegralException
    ca = c*ga
    ca, cd = ca.cancel(gd, include=True)

    ba = a*fa - dn*derivation(h, DE)*fd
    ba, bd = ba.cancel(fd, include=True)

    # (dn*h, dn*h*f - dn*Dh, dn*h**2*g, h)
    return (a, (ba, bd), (ca, cd), h)


def special_denom(a, ba, bd, ca, cd, DE, case='auto'):
    """
    Special part of the denominator.

    Explanation
    ===========

    case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the
    hyperexponential (resp. hypertangent) case, given a derivation D on
    k[t] and a in k[t], b, c, in k<t> with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the quadruplet (A, B, C, 1/h) such that
    A, B, C, h in k[t] and for any solution q in k<t> of a*Dq + b*q == c,
    r = qh in k[t] satisfies A*Dr + B*r == C.

    For ``case == 'primitive'``, k<t> == k[t], so it returns (a, b, c, 1) in
    this case.

    This constitutes step 2 of the outline given in the rde.py docstring.
    """
    # TODO: finish writing this and write tests

    if case == 'auto':
        case = DE.case

    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t**2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        B = ba.to_field().quo(bd)
        C = ca.to_field().quo(cd)
        return (a, B, C, Poly(1, DE.t))
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base'}, not %s." % case)

    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = order_at(ca, p, DE.t) - order_at(cd, p, DE.t)

    n = min(0, nc - min(0, nb))
    if not nb:
        # Possible cancellation.
        from .prde import parametric_log_deriv
        if case == 'exp':
            dcoeff = DE.d.quo(Poly(DE.t, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                alphaa, alphad = frac_in(-ba.eval(0)/bd.eval(0)/a.eval(0), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    Q, m, z = A
                    if Q == 1:
                        n = min(n, m)

        elif case == 'tan':
            dcoeff = DE.d.quo(Poly(DE.t**2+1, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                alphaa, alphad = frac_in(im(-ba.eval(sqrt(-1))/bd.eval(sqrt(-1))/a.eval(sqrt(-1))), DE.t)
                betaa, betad = frac_in(re(-ba.eval(sqrt(-1))/bd.eval(sqrt(-1))/a.eval(sqrt(-1))), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)

                if recognize_log_derivative(Poly(2, DE.t)*betaa, betad, DE):
                    A = parametric_log_deriv(alphaa*Poly(sqrt(-1), DE.t)*betad+alphad*betaa, alphad*betad, etaa, etad, DE)
                    if A is not None:
                        Q, m, z = A
                        if Q == 1:
                            n = min(n, m)
    N = max(0, -nb, n - nc)
    pN = p**N
    pn = p**-n

    A = a*pN
    B = ba*pN.quo(bd) + Poly(n, DE.t)*a*derivation(p, DE).quo(p)*pN
    C = (ca*pN*pn).quo(cd)
    h = pn

    # (a*p**N, (b + n*a*Dp/p)*p**N, c*p**(N - n), p**-n)
    return (A, B, C, h)


def bound_degree(a, b, cQ, DE, case='auto', parametric=False):
    """
    Bound on polynomial solutions.

    Explanation
    ===========

    Given a derivation D on k[t] and ``a``, ``b``, ``c`` in k[t] with ``a != 0``, return
    n in ZZ such that deg(q) <= n for any solution q in k[t] of
    a*Dq + b*q == c, when parametric=False, or deg(q) <= n for any solution
    c1, ..., cm in Const(k) and q in k[t] of a*Dq + b*q == Sum(ci*gi, (i, 1, m))
    when parametric=True.

    For ``parametric=False``, ``cQ`` is ``c``, a ``Poly``; for ``parametric=True``, ``cQ`` is Q ==
    [q1, ..., qm], a list of Polys.

    This constitutes step 3 of the outline given in the rde.py docstring.
    """
    # TODO: finish writing this and write tests

    if case == 'auto':
        case = DE.case

    da = a.degree(DE.t)
    db = b.degree(DE.t)

    # The parametric and regular cases are identical, except for this part
    if parametric:
        dc = max(i.degree(DE.t) for i in cQ)
    else:
        dc = cQ.degree(DE.t)

    alpha = cancel(-b.as_poly(DE.t).LC().as_expr()/
        a.as_poly(DE.t).LC().as_expr())

    if case == 'base':
        n = max(0, dc - max(db, da - 1))
        if db == da - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)

    elif case == 'primitive':
        if db > da:
            n = max(0, dc - db)
        else:
            n = max(0, dc - da + 1)

        etaa, etad = frac_in(DE.d, DE.T[DE.level - 1])

        t1 = DE.t
        with DecrementLevel(DE):
            alphaa, alphad = frac_in(alpha, DE.t)
            if db == da - 1:
                from .prde import limited_integrate
                # if alpha == m*Dt + Dz for z in k and m in ZZ:
                try:
                    (za, zd), m = limited_integrate(alphaa, alphad, [(etaa, etad)],
                        DE)
                except NonElementaryIntegralException:
                    pass
                else:
                    if len(m) != 1:
                        raise ValueError("Length of m should be 1")
                    n = max(n, m[0])

            elif db == da:
                # if alpha == Dz/z for z in k*:
                    # beta = -lc(a*Dz + b*z)/(z*lc(a))
                    # if beta == m*Dt + Dw for w in k and m in ZZ:
                        # n = max(n, m)
                from .prde import is_log_deriv_k_t_radical_in_field
                A = is_log_deriv_k_t_radical_in_field(alphaa, alphad, DE)
                if A is not None:
                    aa, z = A
                    if aa == 1:
                        beta = -(a*derivation(z, DE).as_poly(t1) +
                            b*z.as_poly(t1)).LC()/(z.as_expr()*a.LC())
                        betaa, betad = frac_in(beta, DE.t)
                        from .prde import limited_integrate
                        try:
                            (za, zd), m = limited_integrate(betaa, betad,
                                [(etaa, etad)], DE)
                        except NonElementaryIntegralException:
                            pass
                        else:
                            if len(m) != 1:
                                raise ValueError("Length of m should be 1")
                            n = max(n, m[0].as_expr())

    elif case == 'exp':
        from .prde import parametric_log_deriv

        n = max(0, dc - max(db, da))
        if da == db:
            etaa, etad = frac_in(DE.d.quo(Poly(DE.t, DE.t)), DE.T[DE.level - 1])
            with DecrementLevel(DE):
                alphaa, alphad = frac_in(alpha, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    # if alpha == m*Dt/t + Dz/z for z in k* and m in ZZ:
                        # n = max(n, m)
                    a, m, z = A
                    if a == 1:
                        n = max(n, m)

    elif case in ('tan', 'other_nonlinear'):
        delta = DE.d.degree(DE.t)
        lam = DE.d.LC()
        alpha = cancel(alpha/lam)
        n = max(0, dc - max(da + delta - 1, db))
        if db == da + delta - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)

    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'other_nonlinear', 'base'}, not %s." % case)

    return n


def spde(a, b, c, n, DE):
    """
    Rothstein's Special Polynomial Differential Equation algorithm.

    Explanation
    ===========

    Given a derivation D on k[t], an integer n and ``a``,``b``,``c`` in k[t] with
    ``a != 0``, either raise NonElementaryIntegralException, in which case the
    equation a*Dq + b*q == c has no solution of degree at most ``n`` in
    k[t], or return the tuple (B, C, m, alpha, beta) such that B, C,
    alpha, beta in k[t], m in ZZ, and any solution q in k[t] of degree
    at most n of a*Dq + b*q == c must be of the form
    q == alpha*h + beta, where h in k[t], deg(h) <= m, and Dh + B*h == C.

    This constitutes step 4 of the outline given in the rde.py docstring.
    """
    zero = Poly(0, DE.t)

    alpha = Poly(1, DE.t)
    beta = Poly(0, DE.t)

    while True:
        if c.is_zero:
            return (zero, zero, 0, zero, beta)  # -1 is more to the point
        if (n < 0) is True:
            raise NonElementaryIntegralException

        g = a.gcd(b)
        if not c.rem(g).is_zero:  # g does not divide c
            raise NonElementaryIntegralException

        a, b, c = a.quo(g), b.quo(g), c.quo(g)

        if a.degree(DE.t) == 0:
            b = b.to_field().quo(a)
            c = c.to_field().quo(a)
            return (b, c, n, alpha, beta)

        r, z = gcdex_diophantine(b, a, c)
        b += derivation(a, DE)
        c = z - derivation(r, DE)
        n -= a.degree(DE.t)

        beta += alpha * r
        alpha *= a

def no_cancel_b_large(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) large enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with ``b != 0`` and either D == d/dt or
    deg(b) > max(0, deg(D) - 1), either raise NonElementaryIntegralException, in
    which case the equation ``Dq + b*q == c`` has no solution of degree at
    most n in k[t], or a solution q in k[t] of this equation with
    ``deg(q) < n``.
    """
    q = Poly(0, DE.t)

    while not c.is_zero:
        m = c.degree(DE.t) - b.degree(DE.t)
        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        p = Poly(c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC()*DE.t**m, DE.t,
            expand=False)
        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


def no_cancel_b_small(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) small enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with deg(b) < deg(D) - 1 and either D == d/dt or
    deg(D) >= 2, either raise NonElementaryIntegralException, in which case the
    equation Dq + b*q == c has no solution of degree at most n in k[t],
    or a solution q in k[t] of this equation with deg(q) <= n, or the
    tuple (h, b0, c0) such that h in k[t], b0, c0, in k, and for any
    solution q in k[t] of degree at most n of Dq + bq == c, y == q - h
    is a solution in k of Dy + b0*y == c0.
    """
    q = Poly(0, DE.t)

    while not c.is_zero:
        if n == 0:
            m = 0
        else:
            m = c.degree(DE.t) - DE.d.degree(DE.t) + 1

        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        if m > 0:
            p = Poly(c.as_poly(DE.t).LC()/(m*DE.d.as_poly(DE.t).LC())*DE.t**m,
                DE.t, expand=False)
        else:
            if b.degree(DE.t) != c.degree(DE.t):
                raise NonElementaryIntegralException
            if b.degree(DE.t) == 0:
                return (q, b.as_poly(DE.T[DE.level - 1]),
                    c.as_poly(DE.T[DE.level - 1]))
            p = Poly(c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC(), DE.t,
                expand=False)

        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


# TODO: better name for this function
def no_cancel_equal(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) == deg(D) - 1

    Explanation
    ===========

    Given a derivation D on k[t] with deg(D) >= 2, n either an integer
    or +oo, and b, c in k[t] with deg(b) == deg(D) - 1, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c has
    no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n, or the tuple (h, m, C) such that h
    in k[t], m in ZZ, and C in k[t], and for any solution q in k[t] of
    degree at most n of Dq + b*q == c, y == q - h is a solution in k[t]
    of degree at most m of Dy + b*y == C.
    """
    q = Poly(0, DE.t)
    lc = cancel(-b.as_poly(DE.t).LC()/DE.d.as_poly(DE.t).LC())
    if lc.is_Integer and lc.is_positive:
        M = lc
    else:
        M = -1

    while not c.is_zero:
        m = max(M, c.degree(DE.t) - DE.d.degree(DE.t) + 1)

        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        u = cancel(m*DE.d.as_poly(DE.t).LC() + b.as_poly(DE.t).LC())
        if u.is_zero:
            return (q, m, c)
        if m > 0:
            p = Poly(c.as_poly(DE.t).LC()/u*DE.t**m, DE.t, expand=False)
        else:
            if c.degree(DE.t) != DE.d.degree(DE.t) - 1:
                raise NonElementaryIntegralException
            else:
                p = c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC()

        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


def cancel_primitive(b, c, n, DE):
    """
    Poly Risch Differential Equation - Cancellation: Primitive case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.
    """
    # Delayed imports
    from .prde import is_log_deriv_k_t_radical_in_field
    with DecrementLevel(DE):
        ba, bd = frac_in(b, DE.t)
        A = is_log_deriv_k_t_radical_in_field(ba, bd, DE)
        if A is not None:
            n, z = A
            if n == 1:  # b == Dz/z
                raise NotImplementedError("is_deriv_in_field() is required to "
                    " solve this problem.")
                # if z*c == Dp for p in k[t] and deg(p) <= n:
                #     return p/z
                # else:
                #     raise NonElementaryIntegralException

    if c.is_zero:
        return c  # return 0

    if n < c.degree(DE.t):
        raise NonElementaryIntegralException

    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException
        with DecrementLevel(DE):
            a2a, a2d = frac_in(c.LC(), DE.t)
            sa, sd = rischDE(ba, bd, a2a, a2d, DE)
        stm = Poly(sa.as_expr()/sd.as_expr()*DE.t**m, DE.t, expand=False)
        q += stm
        n = m - 1
        c -= b*stm + derivation(stm, DE)

    return q


def cancel_exp(b, c, n, DE):
    """
    Poly Risch Differential Equation - Cancellation: Hyperexponential case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt/t in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.
    """
    from .prde import parametric_log_deriv
    eta = DE.d.quo(Poly(DE.t, DE.t)).as_expr()

    with DecrementLevel(DE):
        etaa, etad = frac_in(eta, DE.t)
        ba, bd = frac_in(b, DE.t)
        A = parametric_log_deriv(ba, bd, etaa, etad, DE)
        if A is not None:
            a, m, z = A
            if a == 1:
                raise NotImplementedError("is_deriv_in_field() is required to "
                    "solve this problem.")
                # if c*z*t**m == Dp for p in k<t> and q = p/(z*t**m) in k[t] and
                # deg(q) <= n:
                #     return q
                # else:
                #     raise NonElementaryIntegralException

    if c.is_zero:
        return c  # return 0

    if n < c.degree(DE.t):
        raise NonElementaryIntegralException

    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException
        # a1 = b + m*Dt/t
        a1 = b.as_expr()
        with DecrementLevel(DE):
            # TODO: Write a dummy function that does this idiom
            a1a, a1d = frac_in(a1, DE.t)
            a1a = a1a*etad + etaa*a1d*Poly(m, DE.t)
            a1d = a1d*etad

            a2a, a2d = frac_in(c.LC(), DE.t)

            sa, sd = rischDE(a1a, a1d, a2a, a2d, DE)
        stm = Poly(sa.as_expr()/sd.as_expr()*DE.t**m, DE.t, expand=False)
        q += stm
        n = m - 1
        c -= b*stm + derivation(stm, DE)  # deg(c) becomes smaller
    return q


def solve_poly_rde(b, cQ, n, DE, parametric=False):
    """
    Solve a Polynomial Risch Differential Equation with degree bound ``n``.

    This constitutes step 4 of the outline given in the rde.py docstring.

    For parametric=False, cQ is c, a Poly; for parametric=True, cQ is Q ==
    [q1, ..., qm], a list of Polys.
    """
    # No cancellation
    if not b.is_zero and (DE.case == 'base' or
            b.degree(DE.t) > max(0, DE.d.degree(DE.t) - 1)):

        if parametric:
            # Delayed imports
            from .prde import prde_no_cancel_b_large
            return prde_no_cancel_b_large(b, cQ, n, DE)
        return no_cancel_b_large(b, cQ, n, DE)

    elif (b.is_zero or b.degree(DE.t) < DE.d.degree(DE.t) - 1) and \
            (DE.case == 'base' or DE.d.degree(DE.t) >= 2):

        if parametric:
            from .prde import prde_no_cancel_b_small
            return prde_no_cancel_b_small(b, cQ, n, DE)

        R = no_cancel_b_small(b, cQ, n, DE)

        if isinstance(R, Poly):
            return R
        else:
            # XXX: Might k be a field? (pg. 209)
            h, b0, c0 = R
            with DecrementLevel(DE):
                b0, c0 = b0.as_poly(DE.t), c0.as_poly(DE.t)
                if b0 is None:  # See above comment
                    raise ValueError("b0 should be a non-Null value")
                if c0 is  None:
                    raise ValueError("c0 should be a non-Null value")
                y = solve_poly_rde(b0, c0, n, DE).as_poly(DE.t)
            return h + y

    elif DE.d.degree(DE.t) >= 2 and b.degree(DE.t) == DE.d.degree(DE.t) - 1 and \
            n > -b.as_poly(DE.t).LC()/DE.d.as_poly(DE.t).LC():

        # TODO: Is this check necessary, and if so, what should it do if it fails?
        # b comes from the first element returned from spde()
        if not b.as_poly(DE.t).LC().is_number:
            raise TypeError("Result should be a number")

        if parametric:
            raise NotImplementedError("prde_no_cancel_b_equal() is not yet "
                "implemented.")

        R = no_cancel_equal(b, cQ, n, DE)

        if isinstance(R, Poly):
            return R
        else:
            h, m, C = R
            # XXX: Or should it be rischDE()?
            y = solve_poly_rde(b, C, m, DE)
            return h + y

    else:
        # Cancellation
        if b.is_zero:
            raise NotImplementedError("Remaining cases for Poly (P)RDE are "
            "not yet implemented (is_deriv_in_field() required).")
        else:
            if DE.case == 'exp':
                if parametric:
                    raise NotImplementedError("Parametric RDE cancellation "
                        "hyperexponential case is not yet implemented.")
                return cancel_exp(b, cQ, n, DE)

            elif DE.case == 'primitive':
                if parametric:
                    raise NotImplementedError("Parametric RDE cancellation "
                        "primitive case is not yet implemented.")
                return cancel_primitive(b, cQ, n, DE)

            else:
                raise NotImplementedError("Other Poly (P)RDE cancellation "
                    "cases are not yet implemented (%s)." % DE.case)

        if parametric:
            raise NotImplementedError("Remaining cases for Poly PRDE not yet "
                "implemented.")
        raise NotImplementedError("Remaining cases for Poly RDE not yet "
            "implemented.")


def rischDE(fa, fd, ga, gd, DE):
    """
    Solve a Risch Differential Equation: Dy + f*y == g.

    Explanation
    ===========

    See the outline in the docstring of rde.py for more information
    about the procedure used.  Either raise NonElementaryIntegralException, in
    which case there is no solution y in the given differential field,
    or return y in k(t) satisfying Dy + f*y == g, or raise
    NotImplementedError, in which case, the algorithms necessary to
    solve the given Risch Differential Equation have not yet been
    implemented.
    """
    _, (fa, fd) = weak_normalizer(fa, fd, DE)
    a, (ba, bd), (ca, cd), hn = normal_denom(fa, fd, ga, gd, DE)
    A, B, C, hs = special_denom(a, ba, bd, ca, cd, DE)
    try:
        # Until this is fully implemented, use oo.  Note that this will almost
        # certainly cause non-termination in spde() (unless A == 1), and
        # *might* lead to non-termination in the next step for a nonelementary
        # integral (I don't know for certain yet).  Fortunately, spde() is
        # currently written recursively, so this will just give
        # RuntimeError: maximum recursion depth exceeded.
        n = bound_degree(A, B, C, DE)
    except NotImplementedError:
        # Useful for debugging:
        # import warnings
        # warnings.warn("rischDE: Proceeding with n = oo; may cause "
        #     "non-termination.")
        n = oo

    B, C, m, alpha, beta = spde(A, B, C, n, DE)
    if C.is_zero:
        y = C
    else:
        y = solve_poly_rde(B, C, m, DE)

    return (alpha*y + beta, hn*hs)
