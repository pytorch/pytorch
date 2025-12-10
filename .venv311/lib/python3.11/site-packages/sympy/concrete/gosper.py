"""Gosper's algorithm for hypergeometric summation. """

from sympy.core import S, Dummy, symbols
from sympy.polys import Poly, parallel_poly_from_expr, factor
from sympy.utilities.iterables import is_sequence


def gosper_normal(f, g, n, polys=True):
    r"""
    Compute the Gosper's normal form of ``f`` and ``g``.

    Explanation
    ===========

    Given relatively prime univariate polynomials ``f`` and ``g``,
    rewrite their quotient to a normal form defined as follows:

    .. math::
        \frac{f(n)}{g(n)} = Z \cdot \frac{A(n) C(n+1)}{B(n) C(n)}

    where ``Z`` is an arbitrary constant and ``A``, ``B``, ``C`` are
    monic polynomials in ``n`` with the following properties:

    1. `\gcd(A(n), B(n+h)) = 1 \forall h \in \mathbb{N}`
    2. `\gcd(B(n), C(n+1)) = 1`
    3. `\gcd(A(n), C(n)) = 1`

    This normal form, or rational factorization in other words, is a
    crucial step in Gosper's algorithm and in solving of difference
    equations. It can be also used to decide if two hypergeometric
    terms are similar or not.

    This procedure will return a tuple containing elements of this
    factorization in the form ``(Z*A, B, C)``.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_normal
    >>> from sympy.abc import n

    >>> gosper_normal(4*n+5, 2*(4*n+1)*(2*n+3), n, polys=False)
    (1/4, n + 3/2, n + 1/4)

    """
    (p, q), opt = parallel_poly_from_expr(
        (f, g), n, field=True, extension=True)

    a, A = p.LC(), p.monic()
    b, B = q.LC(), q.monic()

    C, Z = A.one, a/b
    h = Dummy('h')

    D = Poly(n + h, n, h, domain=opt.domain)

    R = A.resultant(B.compose(D))
    roots = {r for r in R.ground_roots().keys() if r.is_Integer and r >= 0}
    for i in sorted(roots):
        d = A.gcd(B.shift(+i))

        A = A.quo(d)
        B = B.quo(d.shift(-i))

        for j in range(1, i + 1):
            C *= d.shift(-j)

    A = A.mul_ground(Z)

    if not polys:
        A = A.as_expr()
        B = B.as_expr()
        C = C.as_expr()

    return A, B, C


def gosper_term(f, n):
    r"""
    Compute Gosper's hypergeometric term for ``f``.

    Explanation
    ===========

    Suppose ``f`` is a hypergeometric term such that:

    .. math::
        s_n = \sum_{k=0}^{n-1} f_k

    and `f_k` does not depend on `n`. Returns a hypergeometric
    term `g_n` such that `g_{n+1} - g_n = f_n`.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_term
    >>> from sympy import factorial
    >>> from sympy.abc import n

    >>> gosper_term((4*n + 1)*factorial(n)/factorial(2*n + 1), n)
    (-n - 1/2)/(n + 1/4)

    """
    from sympy.simplify import hypersimp
    r = hypersimp(f, n)

    if r is None:
        return None    # 'f' is *not* a hypergeometric term

    p, q = r.as_numer_denom()

    A, B, C = gosper_normal(p, q, n)
    B = B.shift(-1)

    N = S(A.degree())
    M = S(B.degree())
    K = S(C.degree())

    if (N != M) or (A.LC() != B.LC()):
        D = {K - max(N, M)}
    elif not N:
        D = {K - N + 1, S.Zero}
    else:
        D = {K - N + 1, (B.nth(N - 1) - A.nth(N - 1))/A.LC()}

    for d in set(D):
        if not d.is_Integer or d < 0:
            D.remove(d)

    if not D:
        return None    # 'f(n)' is *not* Gosper-summable

    d = max(D)

    coeffs = symbols('c:%s' % (d + 1), cls=Dummy)
    domain = A.get_domain().inject(*coeffs)

    x = Poly(coeffs, n, domain=domain)
    H = A*x.shift(1) - B*x - C

    from sympy.solvers.solvers import solve
    solution = solve(H.coeffs(), coeffs)

    if solution is None:
        return None    # 'f(n)' is *not* Gosper-summable

    x = x.as_expr().subs(solution)

    for coeff in coeffs:
        if coeff not in solution:
            x = x.subs(coeff, 0)

    if x.is_zero:
        return None    # 'f(n)' is *not* Gosper-summable
    else:
        return B.as_expr()*x/C.as_expr()


def gosper_sum(f, k):
    r"""
    Gosper's hypergeometric summation algorithm.

    Explanation
    ===========

    Given a hypergeometric term ``f`` such that:

    .. math ::
        s_n = \sum_{k=0}^{n-1} f_k

    and `f(n)` does not depend on `n`, returns `g_{n} - g(0)` where
    `g_{n+1} - g_n = f_n`, or ``None`` if `s_n` cannot be expressed
    in closed form as a sum of hypergeometric terms.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_sum
    >>> from sympy import factorial
    >>> from sympy.abc import n, k

    >>> f = (4*k + 1)*factorial(k)/factorial(2*k + 1)
    >>> gosper_sum(f, (k, 0, n))
    (-factorial(n) + 2*factorial(2*n + 1))/factorial(2*n + 1)
    >>> _.subs(n, 2) == sum(f.subs(k, i) for i in [0, 1, 2])
    True
    >>> gosper_sum(f, (k, 3, n))
    (-60*factorial(n) + factorial(2*n + 1))/(60*factorial(2*n + 1))
    >>> _.subs(n, 5) == sum(f.subs(k, i) for i in [3, 4, 5])
    True

    References
    ==========

    .. [1] Marko Petkovsek, Herbert S. Wilf, Doron Zeilberger, A = B,
           AK Peters, Ltd., Wellesley, MA, USA, 1997, pp. 73--100

    """
    indefinite = False

    if is_sequence(k):
        k, a, b = k
    else:
        indefinite = True

    g = gosper_term(f, k)

    if g is None:
        return None

    if indefinite:
        result = f*g
    else:
        result = (f*(g + 1)).subs(k, b) - (f*g).subs(k, a)

        if result is S.NaN:
            try:
                result = (f*(g + 1)).limit(k, b) - (f*g).limit(k, a)
            except NotImplementedError:
                result = None

    return factor(result)
