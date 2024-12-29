r"""
This module is intended for solving recurrences or, in other words,
difference equations. Currently supported are linear, inhomogeneous
equations with polynomial or rational coefficients.

The solutions are obtained among polynomials, rational functions,
hypergeometric terms, or combinations of hypergeometric term which
are pairwise dissimilar.

``rsolve_X`` functions were meant as a low level interface
for ``rsolve`` which would use Mathematica's syntax.

Given a recurrence relation:

    .. math:: a_{k}(n) y(n+k) + a_{k-1}(n) y(n+k-1) +
              ... + a_{0}(n) y(n) = f(n)

where `k > 0` and `a_{i}(n)` are polynomials in `n`. To use
``rsolve_X`` we need to put all coefficients in to a list ``L`` of
`k+1` elements the following way:

    ``L = [a_{0}(n), ..., a_{k-1}(n), a_{k}(n)]``

where ``L[i]``, for `i=0, \ldots, k`, maps to
`a_{i}(n) y(n+i)` (`y(n+i)` is implicit).

For example if we would like to compute `m`-th Bernoulli polynomial
up to a constant (example was taken from rsolve_poly docstring),
then we would use `b(n+1) - b(n) = m n^{m-1}` recurrence, which
has solution `b(n) = B_m + C`.

Then ``L = [-1, 1]`` and `f(n) = m n^(m-1)` and finally for `m=4`:

>>> from sympy import Symbol, bernoulli, rsolve_poly
>>> n = Symbol('n', integer=True)

>>> rsolve_poly([-1, 1], 4*n**3, n)
C0 + n**4 - 2*n**3 + n**2

>>> bernoulli(4, n)
n**4 - 2*n**3 + n**2 - 1/30

For the sake of completeness, `f(n)` can be:

    [1] a polynomial               -> rsolve_poly
    [2] a rational function        -> rsolve_ratio
    [3] a hypergeometric function  -> rsolve_hyper
"""
from collections import defaultdict

from sympy.concrete import product
from sympy.core.singleton import S
from sympy.core.numbers import Rational, I
from sympy.core.symbol import Symbol, Wild, Dummy
from sympy.core.relational import Equality
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify

from sympy.simplify import simplify, hypersimp, hypersimilar  # type: ignore
from sympy.solvers import solve, solve_undetermined_coeffs
from sympy.polys import Poly, quo, gcd, lcm, roots, resultant
from sympy.functions import binomial, factorial, FallingFactorial, RisingFactorial
from sympy.matrices import Matrix, casoratian
from sympy.utilities.iterables import numbered_symbols


def rsolve_poly(coeffs, f, n, shift=0, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order
    `k` with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f`, where `f` is a polynomial, we seek for
    all polynomial solutions over field `K` of characteristic zero.

    The algorithm performs two basic steps:

        (1) Compute degree `N` of the general polynomial solution.
        (2) Find all polynomials of degree `N` or less
            of `\operatorname{L} y = f`.

    There are two methods for computing the polynomial solutions.
    If the degree bound is relatively small, i.e. it's smaller than
    or equal to the order of the recurrence, then naive method of
    undetermined coefficients is being used. This gives a system
    of algebraic equations with `N+1` unknowns.

    In the other case, the algorithm performs transformation of the
    initial equation to an equivalent one for which the system of
    algebraic equations has only `r` indeterminates. This method is
    quite sophisticated (in comparison with the naive one) and was
    invented together by Abramov, Bronstein and Petkovsek.

    It is possible to generalize the algorithm implemented here to
    the case of linear q-difference and differential equations.

    Lets say that we would like to compute `m`-th Bernoulli polynomial
    up to a constant. For this we can use `b(n+1) - b(n) = m n^{m-1}`
    recurrence, which has solution `b(n) = B_m + C`. For example:

    >>> from sympy import Symbol, rsolve_poly
    >>> n = Symbol('n', integer=True)

    >>> rsolve_poly([-1, 1], 4*n**3, n)
    C0 + n**4 - 2*n**3 + n**2

    References
    ==========

    .. [1] S. A. Abramov, M. Bronstein and M. Petkovsek, On polynomial
           solutions of linear operator equations, in: T. Levelt, ed.,
           Proc. ISSAC '95, ACM Press, New York, 1995, 290-296.

    .. [2] M. Petkovsek, Hypergeometric solutions of linear recurrences
           with polynomial coefficients, J. Symbolic Computation,
           14 (1992), 243-264.

    .. [3] M. Petkovsek, H. S. Wilf, D. Zeilberger, A = B, 1996.

    """
    f = sympify(f)

    if not f.is_polynomial(n):
        return None

    homogeneous = f.is_zero

    r = len(coeffs) - 1

    coeffs = [Poly(coeff, n) for coeff in coeffs]

    polys = [Poly(0, n)]*(r + 1)
    terms = [(S.Zero, S.NegativeInfinity)]*(r + 1)

    for i in range(r + 1):
        for j in range(i, r + 1):
            polys[i] += coeffs[j]*(binomial(j, i).as_poly(n))

        if not polys[i].is_zero:
            (exp,), coeff = polys[i].LT()
            terms[i] = (coeff, exp)

    d = b = terms[0][1]

    for i in range(1, r + 1):
        if terms[i][1] > d:
            d = terms[i][1]

        if terms[i][1] - i > b:
            b = terms[i][1] - i

    d, b = int(d), int(b)

    x = Dummy('x')

    degree_poly = S.Zero

    for i in range(r + 1):
        if terms[i][1] - i == b:
            degree_poly += terms[i][0]*FallingFactorial(x, i)

    nni_roots = list(roots(degree_poly, x, filter='Z',
        predicate=lambda r: r >= 0).keys())

    if nni_roots:
        N = [max(nni_roots)]
    else:
        N = []

    if homogeneous:
        N += [-b - 1]
    else:
        N += [f.as_poly(n).degree() - b, -b - 1]

    N = int(max(N))

    if N < 0:
        if homogeneous:
            if hints.get('symbols', False):
                return (S.Zero, [])
            else:
                return S.Zero
        else:
            return None

    if N <= r:
        C = []
        y = E = S.Zero

        for i in range(N + 1):
            C.append(Symbol('C' + str(i + shift)))
            y += C[i] * n**i

        for i in range(r + 1):
            E += coeffs[i].as_expr()*y.subs(n, n + i)

        solutions = solve_undetermined_coeffs(E - f, C, n)

        if solutions is not None:
            _C = C
            C = [c for c in C if (c not in solutions)]
            result = y.subs(solutions)
        else:
            return None  # TBD
    else:
        A = r
        U = N + A + b + 1

        nni_roots = list(roots(polys[r], filter='Z',
            predicate=lambda r: r >= 0).keys())

        if nni_roots != []:
            a = max(nni_roots) + 1
        else:
            a = S.Zero

        def _zero_vector(k):
            return [S.Zero] * k

        def _one_vector(k):
            return [S.One] * k

        def _delta(p, k):
            B = S.One
            D = p.subs(n, a + k)

            for i in range(1, k + 1):
                B *= Rational(i - k - 1, i)
                D += B * p.subs(n, a + k - i)

            return D

        alpha = {}

        for i in range(-A, d + 1):
            I = _one_vector(d + 1)

            for k in range(1, d + 1):
                I[k] = I[k - 1] * (x + i - k + 1)/k

            alpha[i] = S.Zero

            for j in range(A + 1):
                for k in range(d + 1):
                    B = binomial(k, i + j)
                    D = _delta(polys[j].as_expr(), k)

                    alpha[i] += I[k]*B*D

        V = Matrix(U, A, lambda i, j: int(i == j))

        if homogeneous:
            for i in range(A, U):
                v = _zero_vector(A)

                for k in range(1, A + b + 1):
                    if i - k < 0:
                        break

                    B = alpha[k - A].subs(x, i - k)

                    for j in range(A):
                        v[j] += B * V[i - k, j]

                denom = alpha[-A].subs(x, i)

                for j in range(A):
                    V[i, j] = -v[j] / denom
        else:
            G = _zero_vector(U)

            for i in range(A, U):
                v = _zero_vector(A)
                g = S.Zero

                for k in range(1, A + b + 1):
                    if i - k < 0:
                        break

                    B = alpha[k - A].subs(x, i - k)

                    for j in range(A):
                        v[j] += B * V[i - k, j]

                    g += B * G[i - k]

                denom = alpha[-A].subs(x, i)

                for j in range(A):
                    V[i, j] = -v[j] / denom

                G[i] = (_delta(f, i - A) - g) / denom

        P, Q = _one_vector(U), _zero_vector(A)

        for i in range(1, U):
            P[i] = (P[i - 1] * (n - a - i + 1)/i).expand()

        for i in range(A):
            Q[i] = Add(*[(v*p).expand() for v, p in zip(V[:, i], P)])

        if not homogeneous:
            h = Add(*[(g*p).expand() for g, p in zip(G, P)])

        C = [Symbol('C' + str(i + shift)) for i in range(A)]

        g = lambda i: Add(*[c*_delta(q, i) for c, q in zip(C, Q)])

        if homogeneous:
            E = [g(i) for i in range(N + 1, U)]
        else:
            E = [g(i) + _delta(h, i) for i in range(N + 1, U)]

        if E != []:
            solutions = solve(E, *C)

            if not solutions:
                if homogeneous:
                    if hints.get('symbols', False):
                        return (S.Zero, [])
                    else:
                        return S.Zero
                else:
                    return None
        else:
            solutions = {}

        if homogeneous:
            result = S.Zero
        else:
            result = h

        _C = C[:]
        for c, q in list(zip(C, Q)):
            if c in solutions:
                s = solutions[c]*q
                C.remove(c)
            else:
                s = c*q

            result += s.expand()

    if C != _C:
        # renumber so they are contiguous
        result = result.xreplace(dict(zip(C, _C)))
        C = _C[:len(C)]

    if hints.get('symbols', False):
        return (result, C)
    else:
        return result


def rsolve_ratio(coeffs, f, n, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order `k`
    with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f`, where `f` is a polynomial, we seek
    for all rational solutions over field `K` of characteristic zero.

    This procedure accepts only polynomials, however if you are
    interested in solving recurrence with rational coefficients
    then use ``rsolve`` which will pre-process the given equation
    and run this procedure with polynomial arguments.

    The algorithm performs two basic steps:

        (1) Compute polynomial `v(n)` which can be used as universal
            denominator of any rational solution of equation
            `\operatorname{L} y = f`.

        (2) Construct new linear difference equation by substitution
            `y(n) = u(n)/v(n)` and solve it for `u(n)` finding all its
            polynomial solutions. Return ``None`` if none were found.

    The algorithm implemented here is a revised version of the original
    Abramov's algorithm, developed in 1989. The new approach is much
    simpler to implement and has better overall efficiency. This
    method can be easily adapted to the q-difference equations case.

    Besides finding rational solutions alone, this functions is
    an important part of Hyper algorithm where it is used to find
    a particular solution for the inhomogeneous part of a recurrence.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.solvers.recurr import rsolve_ratio
    >>> rsolve_ratio([-2*x**3 + x**2 + 2*x - 1, 2*x**3 + x**2 - 6*x,
    ... - 2*x**3 - 11*x**2 - 18*x - 9, 2*x**3 + 13*x**2 + 22*x + 8], 0, x)
    C0*(2*x - 3)/(2*(x**2 - 1))

    References
    ==========

    .. [1] S. A. Abramov, Rational solutions of linear difference
           and q-difference equations with polynomial coefficients,
           in: T. Levelt, ed., Proc. ISSAC '95, ACM Press, New York,
           1995, 285-289

    See Also
    ========

    rsolve_hyper
    """
    f = sympify(f)

    if not f.is_polynomial(n):
        return None

    coeffs = list(map(sympify, coeffs))

    r = len(coeffs) - 1

    A, B = coeffs[r], coeffs[0]
    A = A.subs(n, n - r).expand()

    h = Dummy('h')

    res = resultant(A, B.subs(n, n + h), n)

    if not res.is_polynomial(h):
        p, q = res.as_numer_denom()
        res = quo(p, q, h)

    nni_roots = list(roots(res, h, filter='Z',
        predicate=lambda r: r >= 0).keys())

    if not nni_roots:
        return rsolve_poly(coeffs, f, n, **hints)
    else:
        C, numers = S.One, [S.Zero]*(r + 1)

        for i in range(int(max(nni_roots)), -1, -1):
            d = gcd(A, B.subs(n, n + i), n)

            A = quo(A, d, n)
            B = quo(B, d.subs(n, n - i), n)

            C *= Mul(*[d.subs(n, n - j) for j in range(i + 1)])

        denoms = [C.subs(n, n + i) for i in range(r + 1)]

        for i in range(r + 1):
            g = gcd(coeffs[i], denoms[i], n)

            numers[i] = quo(coeffs[i], g, n)
            denoms[i] = quo(denoms[i], g, n)

        for i in range(r + 1):
            numers[i] *= Mul(*(denoms[:i] + denoms[i + 1:]))

        result = rsolve_poly(numers, f * Mul(*denoms), n, **hints)

        if result is not None:
            if hints.get('symbols', False):
                return (simplify(result[0] / C), result[1])
            else:
                return simplify(result / C)
        else:
            return None


def rsolve_hyper(coeffs, f, n, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order `k`
    with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f` we seek for all hypergeometric solutions
    over field `K` of characteristic zero.

    The inhomogeneous part can be either hypergeometric or a sum
    of a fixed number of pairwise dissimilar hypergeometric terms.

    The algorithm performs three basic steps:

        (1) Group together similar hypergeometric terms in the
            inhomogeneous part of `\operatorname{L} y = f`, and find
            particular solution using Abramov's algorithm.

        (2) Compute generating set of `\operatorname{L}` and find basis
            in it, so that all solutions are linearly independent.

        (3) Form final solution with the number of arbitrary
            constants equal to dimension of basis of `\operatorname{L}`.

    Term `a(n)` is hypergeometric if it is annihilated by first order
    linear difference equations with polynomial coefficients or, in
    simpler words, if consecutive term ratio is a rational function.

    The output of this procedure is a linear combination of fixed
    number of hypergeometric terms. However the underlying method
    can generate larger class of solutions - D'Alembertian terms.

    Note also that this method not only computes the kernel of the
    inhomogeneous equation, but also reduces in to a basis so that
    solutions generated by this procedure are linearly independent

    Examples
    ========

    >>> from sympy.solvers import rsolve_hyper
    >>> from sympy.abc import x

    >>> rsolve_hyper([-1, -1, 1], 0, x)
    C0*(1/2 - sqrt(5)/2)**x + C1*(1/2 + sqrt(5)/2)**x

    >>> rsolve_hyper([-1, 1], 1 + x, x)
    C0 + x*(x + 1)/2

    References
    ==========

    .. [1] M. Petkovsek, Hypergeometric solutions of linear recurrences
           with polynomial coefficients, J. Symbolic Computation,
           14 (1992), 243-264.

    .. [2] M. Petkovsek, H. S. Wilf, D. Zeilberger, A = B, 1996.
    """
    coeffs = list(map(sympify, coeffs))

    f = sympify(f)

    r, kernel, symbols = len(coeffs) - 1, [], set()

    if not f.is_zero:
        if f.is_Add:
            similar = {}

            for g in f.expand().args:
                if not g.is_hypergeometric(n):
                    return None

                for h in similar.keys():
                    if hypersimilar(g, h, n):
                        similar[h] += g
                        break
                else:
                    similar[g] = S.Zero

            inhomogeneous = [g + h for g, h in similar.items()]
        elif f.is_hypergeometric(n):
            inhomogeneous = [f]
        else:
            return None

        for i, g in enumerate(inhomogeneous):
            coeff, polys = S.One, coeffs[:]
            denoms = [S.One]*(r + 1)

            s = hypersimp(g, n)

            for j in range(1, r + 1):
                coeff *= s.subs(n, n + j - 1)

                p, q = coeff.as_numer_denom()

                polys[j] *= p
                denoms[j] = q

            for j in range(r + 1):
                polys[j] *= Mul(*(denoms[:j] + denoms[j + 1:]))

            # FIXME: The call to rsolve_ratio below should suffice (rsolve_poly
            # call can be removed) but the XFAIL test_rsolve_ratio_missed must
            # be fixed first.
            R = rsolve_ratio(polys, Mul(*denoms), n, symbols=True)
            if R is not None:
                R, syms = R
                if syms:
                    R = R.subs(zip(syms, [0]*len(syms)))
            else:
                R = rsolve_poly(polys, Mul(*denoms), n)

            if R:
                inhomogeneous[i] *= R
            else:
                return None

            result = Add(*inhomogeneous)
            result = simplify(result)
    else:
        result = S.Zero

    Z = Dummy('Z')

    p, q = coeffs[0], coeffs[r].subs(n, n - r + 1)

    p_factors = list(roots(p, n).keys())
    q_factors = list(roots(q, n).keys())

    factors = [(S.One, S.One)]

    for p in p_factors:
        for q in q_factors:
            if p.is_integer and q.is_integer and p <= q:
                continue
            else:
                factors += [(n - p, n - q)]

    p = [(n - p, S.One) for p in p_factors]
    q = [(S.One, n - q) for q in q_factors]

    factors = p + factors + q

    for A, B in factors:
        polys, degrees = [], []
        D = A*B.subs(n, n + r - 1)

        for i in range(r + 1):
            a = Mul(*[A.subs(n, n + j) for j in range(i)])
            b = Mul(*[B.subs(n, n + j) for j in range(i, r)])

            poly = quo(coeffs[i]*a*b, D, n)
            polys.append(poly.as_poly(n))

            if not poly.is_zero:
                degrees.append(polys[i].degree())

        if degrees:
            d, poly = max(degrees), S.Zero
        else:
            return None

        for i in range(r + 1):
            coeff = polys[i].nth(d)

            if coeff is not S.Zero:
                poly += coeff * Z**i

        for z in roots(poly, Z).keys():
            if z.is_zero:
                continue

            recurr_coeffs = [polys[i].as_expr()*z**i for i in range(r + 1)]
            if d == 0 and 0 != Add(*[recurr_coeffs[j]*j for j in range(1, r + 1)]):
                # faster inline check (than calling rsolve_poly) for a
                # constant solution to a constant coefficient recurrence.
                sol = [Symbol("C" + str(len(symbols)))]
            else:
                sol, syms = rsolve_poly(recurr_coeffs, 0, n, len(symbols), symbols=True)
                sol = sol.collect(syms)
                sol = [sol.coeff(s) for s in syms]

            for C in sol:
                ratio = z * A * C.subs(n, n + 1) / B / C
                ratio = simplify(ratio)
                # If there is a nonnegative root in the denominator of the ratio,
                # this indicates that the term y(n_root) is zero, and one should
                # start the product with the term y(n_root + 1).
                n0 = 0
                for n_root in roots(ratio.as_numer_denom()[1], n).keys():
                    if n_root.has(I):
                        return None
                    elif (n0 < (n_root + 1)) == True:
                        n0 = n_root + 1
                K = product(ratio, (n, n0, n - 1))
                if K.has(factorial, FallingFactorial, RisingFactorial):
                    K = simplify(K)

                if casoratian(kernel + [K], n, zero=False) != 0:
                    kernel.append(K)

    kernel.sort(key=default_sort_key)
    sk = list(zip(numbered_symbols('C'), kernel))

    for C, ker in sk:
        result += C * ker

    if hints.get('symbols', False):
        # XXX: This returns the symbols in a non-deterministic order
        symbols |= {s for s, k in sk}
        return (result, list(symbols))
    else:
        return result


def rsolve(f, y, init=None):
    r"""
    Solve univariate recurrence with rational coefficients.

    Given `k`-th order linear recurrence `\operatorname{L} y = f`,
    or equivalently:

    .. math:: a_{k}(n) y(n+k) + a_{k-1}(n) y(n+k-1) +
              \cdots + a_{0}(n) y(n) = f(n)

    where `a_{i}(n)`, for `i=0, \ldots, k`, are polynomials or rational
    functions in `n`, and `f` is a hypergeometric function or a sum
    of a fixed number of pairwise dissimilar hypergeometric terms in
    `n`, finds all solutions or returns ``None``, if none were found.

    Initial conditions can be given as a dictionary in two forms:

        (1) ``{  n_0  : v_0,   n_1  : v_1, ...,   n_m  : v_m}``
        (2) ``{y(n_0) : v_0, y(n_1) : v_1, ..., y(n_m) : v_m}``

    or as a list ``L`` of values:

        ``L = [v_0, v_1, ..., v_m]``

    where ``L[i] = v_i``, for `i=0, \ldots, m`, maps to `y(n_i)`.

    Examples
    ========

    Lets consider the following recurrence:

    .. math:: (n - 1) y(n + 2) - (n^2 + 3 n - 2) y(n + 1) +
              2 n (n + 1) y(n) = 0

    >>> from sympy import Function, rsolve
    >>> from sympy.abc import n
    >>> y = Function('y')

    >>> f = (n - 1)*y(n + 2) - (n**2 + 3*n - 2)*y(n + 1) + 2*n*(n + 1)*y(n)

    >>> rsolve(f, y(n))
    2**n*C0 + C1*factorial(n)

    >>> rsolve(f, y(n), {y(0):0, y(1):3})
    3*2**n - 3*factorial(n)

    See Also
    ========

    rsolve_poly, rsolve_ratio, rsolve_hyper

    """
    if isinstance(f, Equality):
        f = f.lhs - f.rhs

    n = y.args[0]
    k = Wild('k', exclude=(n,))

    # Preprocess user input to allow things like
    # y(n) + a*(y(n + 1) + y(n - 1))/2
    f = f.expand().collect(y.func(Wild('m', integer=True)))

    h_part = defaultdict(list)
    i_part = []
    for g in Add.make_args(f):
        coeff, dep = g.as_coeff_mul(y.func)
        if not dep:
            i_part.append(coeff)
            continue
        for h in dep:
            if h.is_Function and h.func == y.func:
                result = h.args[0].match(n + k)
                if result is not None:
                    h_part[int(result[k])].append(coeff)
                    continue
            raise ValueError(
                "'%s(%s + k)' expected, got '%s'" % (y.func, n, h))
    for k in h_part:
        h_part[k] = Add(*h_part[k])
    h_part.default_factory = lambda: 0
    i_part = Add(*i_part)

    for k, coeff in h_part.items():
        h_part[k] = simplify(coeff)

    common = S.One

    if not i_part.is_zero and not i_part.is_hypergeometric(n) and \
       not (i_part.is_Add and all((x.is_hypergeometric(n) for x in i_part.expand().args))):
        raise ValueError("The independent term should be a sum of hypergeometric functions, got '%s'" % i_part)

    for coeff in h_part.values():
        if coeff.is_rational_function(n):
            if not coeff.is_polynomial(n):
                common = lcm(common, coeff.as_numer_denom()[1], n)
        else:
            raise ValueError(
                "Polynomial or rational function expected, got '%s'" % coeff)

    i_numer, i_denom = i_part.as_numer_denom()

    if i_denom.is_polynomial(n):
        common = lcm(common, i_denom, n)

    if common is not S.One:
        for k, coeff in h_part.items():
            numer, denom = coeff.as_numer_denom()
            h_part[k] = numer*quo(common, denom, n)

        i_part = i_numer*quo(common, i_denom, n)

    K_min = min(h_part.keys())

    if K_min < 0:
        K = abs(K_min)

        H_part = defaultdict(lambda: S.Zero)
        i_part = i_part.subs(n, n + K).expand()
        common = common.subs(n, n + K).expand()

        for k, coeff in h_part.items():
            H_part[k + K] = coeff.subs(n, n + K).expand()
    else:
        H_part = h_part

    K_max = max(H_part.keys())
    coeffs = [H_part[i] for i in range(K_max + 1)]

    result = rsolve_hyper(coeffs, -i_part, n, symbols=True)

    if result is None:
        return None

    solution, symbols = result

    if init in ({}, []):
        init = None

    if symbols and init is not None:
        if isinstance(init, list):
            init = {i: init[i] for i in range(len(init))}

        equations = []

        for k, v in init.items():
            try:
                i = int(k)
            except TypeError:
                if k.is_Function and k.func == y.func:
                    i = int(k.args[0])
                else:
                    raise ValueError("Integer or term expected, got '%s'" % k)

            eq = solution.subs(n, i) - v
            if eq.has(S.NaN):
                eq = solution.limit(n, i) - v
            equations.append(eq)

        result = solve(equations, *symbols)

        if not result:
            return None
        else:
            solution = solution.subs(result)

    return solution
