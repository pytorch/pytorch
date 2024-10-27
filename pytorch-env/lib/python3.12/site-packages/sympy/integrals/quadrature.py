from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                    hermite_poly, jacobi_poly)
from sympy.polys.rootoftools import RootOf


def gauss_legendre(n, n_digits):
    r"""
    Computes the Gauss-Legendre quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Legendre quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2}{\left(1-x_i^2\right) \left(P'_n(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_legendre
    >>> x, w = gauss_legendre(3, 5)
    >>> x
    [-0.7746, 0, 0.7746]
    >>> w
    [0.55556, 0.88889, 0.55556]
    >>> x, w = gauss_legendre(4, 5)
    >>> x
    [-0.86114, -0.33998, 0.33998, 0.86114]
    >>> w
    [0.34785, 0.65215, 0.65215, 0.34785]

    See Also
    ========

    gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/legendre_rule/legendre_rule.html
    """
    x = Dummy("x")
    p = legendre_poly(n, x, polys=True)
    pd = p.diff(x)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((2/((1-r**2) * pd.subs(x, r)**2)).n(n_digits))
    return xi, w


def gauss_laguerre(n, n_digits):
    r"""
    Computes the Gauss-Laguerre quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Laguerre quadrature approximates the integral:

    .. math::
        \int_0^{\infty} e^{-x} f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)


    The nodes `x_i` of an order `n` quadrature rule are the roots of `L_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{x_i}{(n+1)^2 \left(L_{n+1}(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : The ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_laguerre
    >>> x, w = gauss_laguerre(3, 5)
    >>> x
    [0.41577, 2.2943, 6.2899]
    >>> w
    [0.71109, 0.27852, 0.010389]
    >>> x, w = gauss_laguerre(6, 5)
    >>> x
    [0.22285, 1.1889, 2.9927, 5.7751, 9.8375, 15.983]
    >>> w
    [0.45896, 0.417, 0.11337, 0.010399, 0.00026102, 8.9855e-7]

    See Also
    ========

    gauss_legendre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/laguerre_rule/laguerre_rule.html
    """
    x = Dummy("x")
    p = laguerre_poly(n, x, polys=True)
    p1 = laguerre_poly(n+1, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((r/((n+1)**2 * p1.subs(x, r)**2)).n(n_digits))
    return xi, w


def gauss_hermite(n, n_digits):
    r"""
    Computes the Gauss-Hermite quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Hermite quadrature approximates the integral:

    .. math::
        \int_{-\infty}^{\infty} e^{-x^2} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `H_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2^{n-1} n! \sqrt{\pi}}{n^2 \left(H_{n-1}(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : The ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_hermite
    >>> x, w = gauss_hermite(3, 5)
    >>> x
    [-1.2247, 0, 1.2247]
    >>> w
    [0.29541, 1.1816, 0.29541]

    >>> x, w = gauss_hermite(6, 5)
    >>> x
    [-2.3506, -1.3358, -0.43608, 0.43608, 1.3358, 2.3506]
    >>> w
    [0.00453, 0.15707, 0.72463, 0.72463, 0.15707, 0.00453]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_gen_laguerre, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss-Hermite_Quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/hermite_rule/hermite_rule.html
    .. [3] https://people.sc.fsu.edu/~jburkardt/cpp_src/gen_hermite_rule/gen_hermite_rule.html
    """
    x = Dummy("x")
    p = hermite_poly(n, x, polys=True)
    p1 = hermite_poly(n-1, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append(((2**(n-1) * factorial(n) * sqrt(pi)) /
                 (n**2 * p1.subs(x, r)**2)).n(n_digits))
    return xi, w


def gauss_gen_laguerre(n, alpha, n_digits):
    r"""
    Computes the generalized Gauss-Laguerre quadrature [1]_ points and weights.

    Explanation
    ===========

    The generalized Gauss-Laguerre quadrature approximates the integral:

    .. math::
        \int_{0}^\infty x^{\alpha} e^{-x} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of
    `L^{\alpha}_n` and the weights `w_i` are given by:

    .. math::
        w_i = \frac{\Gamma(\alpha+n)}
                {n \Gamma(n) L^{\alpha}_{n-1}(x_i) L^{\alpha+1}_{n-1}(x_i)}

    Parameters
    ==========

    n :
        The order of quadrature.

    alpha :
        The exponent of the singularity, `\alpha > -1`.

    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.integrals.quadrature import gauss_gen_laguerre
    >>> x, w = gauss_gen_laguerre(3, -S.Half, 5)
    >>> x
    [0.19016, 1.7845, 5.5253]
    >>> w
    [1.4493, 0.31413, 0.00906]

    >>> x, w = gauss_gen_laguerre(4, 3*S.Half, 5)
    >>> x
    [0.97851, 2.9904, 6.3193, 11.712]
    >>> w
    [0.53087, 0.67721, 0.11895, 0.0023152]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/gen_laguerre_rule/gen_laguerre_rule.html
    """
    x = Dummy("x")
    p = laguerre_poly(n, x, alpha=alpha, polys=True)
    p1 = laguerre_poly(n-1, x, alpha=alpha, polys=True)
    p2 = laguerre_poly(n-1, x, alpha=alpha+1, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((gamma(alpha+n) /
                 (n*gamma(n)*p1.subs(x, r)*p2.subs(x, r))).n(n_digits))
    return xi, w


def gauss_chebyshev_t(n, n_digits):
    r"""
    Computes the Gauss-Chebyshev quadrature [1]_ points and weights of
    the first kind.

    Explanation
    ===========

    The Gauss-Chebyshev quadrature of the first kind approximates the integral:

    .. math::
        \int_{-1}^{1} \frac{1}{\sqrt{1-x^2}} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `T_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{\pi}{n}

    Parameters
    ==========

    n :
        The order of quadrature.

    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_chebyshev_t
    >>> x, w = gauss_chebyshev_t(3, 5)
    >>> x
    [0.86602, 0, -0.86602]
    >>> w
    [1.0472, 1.0472, 1.0472]

    >>> x, w = gauss_chebyshev_t(6, 5)
    >>> x
    [0.96593, 0.70711, 0.25882, -0.25882, -0.70711, -0.96593]
    >>> w
    [0.5236, 0.5236, 0.5236, 0.5236, 0.5236, 0.5236]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/chebyshev1_rule/chebyshev1_rule.html
    """
    xi = []
    w = []
    for i in range(1, n+1):
        xi.append((cos((2*i-S.One)/(2*n)*S.Pi)).n(n_digits))
        w.append((S.Pi/n).n(n_digits))
    return xi, w


def gauss_chebyshev_u(n, n_digits):
    r"""
    Computes the Gauss-Chebyshev quadrature [1]_ points and weights of
    the second kind.

    Explanation
    ===========

    The Gauss-Chebyshev quadrature of the second kind approximates the
    integral:

    .. math::
        \int_{-1}^{1} \sqrt{1-x^2} f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `U_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{\pi}{n+1} \sin^2 \left(\frac{i}{n+1}\pi\right)

    Parameters
    ==========

    n : the order of quadrature

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_chebyshev_u
    >>> x, w = gauss_chebyshev_u(3, 5)
    >>> x
    [0.70711, 0, -0.70711]
    >>> w
    [0.3927, 0.7854, 0.3927]

    >>> x, w = gauss_chebyshev_u(6, 5)
    >>> x
    [0.90097, 0.62349, 0.22252, -0.22252, -0.62349, -0.90097]
    >>> w
    [0.084489, 0.27433, 0.42658, 0.42658, 0.27433, 0.084489]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre, gauss_chebyshev_t, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/chebyshev2_rule/chebyshev2_rule.html
    """
    xi = []
    w = []
    for i in range(1, n+1):
        xi.append((cos(i/(n+S.One)*S.Pi)).n(n_digits))
        w.append((S.Pi/(n+S.One)*sin(i*S.Pi/(n+S.One))**2).n(n_digits))
    return xi, w


def gauss_jacobi(n, alpha, beta, n_digits):
    r"""
    Computes the Gauss-Jacobi quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Jacobi quadrature of the first kind approximates the integral:

    .. math::
        \int_{-1}^1 (1-x)^\alpha (1+x)^\beta f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of
    `P^{(\alpha,\beta)}_n` and the weights `w_i` are given by:

    .. math::
        w_i = -\frac{2n+\alpha+\beta+2}{n+\alpha+\beta+1}
              \frac{\Gamma(n+\alpha+1)\Gamma(n+\beta+1)}
              {\Gamma(n+\alpha+\beta+1)(n+1)!}
              \frac{2^{\alpha+\beta}}{P'_n(x_i)
              P^{(\alpha,\beta)}_{n+1}(x_i)}

    Parameters
    ==========

    n : the order of quadrature

    alpha : the first parameter of the Jacobi Polynomial, `\alpha > -1`

    beta : the second parameter of the Jacobi Polynomial, `\beta > -1`

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.integrals.quadrature import gauss_jacobi
    >>> x, w = gauss_jacobi(3, S.Half, -S.Half, 5)
    >>> x
    [-0.90097, -0.22252, 0.62349]
    >>> w
    [1.7063, 1.0973, 0.33795]

    >>> x, w = gauss_jacobi(6, 1, 1, 5)
    >>> x
    [-0.87174, -0.5917, -0.2093, 0.2093, 0.5917, 0.87174]
    >>> w
    [0.050584, 0.22169, 0.39439, 0.39439, 0.22169, 0.050584]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre,
    gauss_chebyshev_t, gauss_chebyshev_u, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/jacobi_rule/jacobi_rule.html
    .. [3] https://people.sc.fsu.edu/~jburkardt/cpp_src/gegenbauer_rule/gegenbauer_rule.html
    """
    x = Dummy("x")
    p = jacobi_poly(n, alpha, beta, x, polys=True)
    pd = p.diff(x)
    pn = jacobi_poly(n+1, alpha, beta, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((
            - (2*n+alpha+beta+2) / (n+alpha+beta+S.One) *
            (gamma(n+alpha+1)*gamma(n+beta+1)) /
            (gamma(n+alpha+beta+S.One)*gamma(n+2)) *
            2**(alpha+beta) / (pd.subs(x, r) * pn.subs(x, r))).n(n_digits))
    return xi, w


def gauss_lobatto(n, n_digits):
    r"""
    Computes the Gauss-Lobatto quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Lobatto quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P'_(n-1)`
    and the weights `w_i` are given by:

    .. math::
        &w_i = \frac{2}{n(n-1) \left[P_{n-1}(x_i)\right]^2},\quad x\neq\pm 1\\
        &w_i = \frac{2}{n(n-1)},\quad x=\pm 1

    Parameters
    ==========

    n : the order of quadrature

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_lobatto
    >>> x, w = gauss_lobatto(3, 5)
    >>> x
    [-1, 0, 1]
    >>> w
    [0.33333, 1.3333, 0.33333]
    >>> x, w = gauss_lobatto(4, 5)
    >>> x
    [-1, -0.44721, 0.44721, 1]
    >>> w
    [0.16667, 0.83333, 0.83333, 0.16667]

    See Also
    ========

    gauss_legendre,gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
    .. [2] https://web.archive.org/web/20200118141346/http://people.math.sfu.ca/~cbm/aands/page_888.htm
    """
    x = Dummy("x")
    p = legendre_poly(n-1, x, polys=True)
    pd = p.diff(x)
    xi = []
    w = []
    for r in pd.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((2/(n*(n-1) * p.subs(x, r)**2)).n(n_digits))

    xi.insert(0, -1)
    xi.append(1)
    w.insert(0, (S(2)/(n*(n-1))).n(n_digits))
    w.append((S(2)/(n*(n-1))).n(n_digits))
    return xi, w
