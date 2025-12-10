"""Square-free decomposition algorithms and related tools. """


from sympy.polys.densearith import (
    dup_neg, dmp_neg,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_quo, dmp_quo,
    dup_mul_ground, dmp_mul_ground)
from sympy.polys.densebasic import (
    dup_strip,
    dup_LC, dmp_ground_LC,
    dmp_zero_p,
    dmp_ground,
    dup_degree, dmp_degree, dmp_degree_in, dmp_degree_list,
    dmp_raise, dmp_inject,
    dup_convert)
from sympy.polys.densetools import (
    dup_diff, dmp_diff, dmp_diff_in,
    dup_shift, dmp_shift,
    dup_monic, dmp_ground_monic,
    dup_primitive, dmp_ground_primitive)
from sympy.polys.euclidtools import (
    dup_inner_gcd, dmp_inner_gcd,
    dup_gcd, dmp_gcd,
    dmp_resultant, dmp_primitive)
from sympy.polys.galoistools import (
    gf_sqf_list, gf_sqf_part)
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    DomainError)


def _dup_check_degrees(f, result):
    """Sanity check the degrees of a computed factorization in K[x]."""
    deg = sum(k * dup_degree(fac) for (fac, k) in result)
    assert deg == dup_degree(f)


def _dmp_check_degrees(f, u, result):
    """Sanity check the degrees of a computed factorization in K[X]."""
    degs = [0] * (u + 1)
    for fac, k in result:
        degs_fac = dmp_degree_list(fac, u)
        degs = [d1 + k * d2 for d1, d2 in zip(degs, degs_fac)]
    assert tuple(degs) == dmp_degree_list(f, u)


def dup_sqf_p(f, K):
    """
    Return ``True`` if ``f`` is a square-free polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_p(x**2 - 2*x + 1)
    False
    >>> R.dup_sqf_p(x**2 - 1)
    True

    """
    if not f:
        return True
    else:
        return not dup_degree(dup_gcd(f, dup_diff(f, 1, K), K))


def dmp_sqf_p(f, u, K):
    """
    Return ``True`` if ``f`` is a square-free polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqf_p(x**2 + 2*x*y + y**2)
    False
    >>> R.dmp_sqf_p(x**2 + y**2)
    True

    """
    if dmp_zero_p(f, u):
        return True

    for i in range(u+1):

        fp = dmp_diff_in(f, 1, i, u, K)

        if dmp_zero_p(fp, u):
            continue

        gcd = dmp_gcd(f, fp, u, K)

        if dmp_degree_in(gcd, i, u) != 0:
            return False

    return True


def dup_sqf_norm(f, K):
    r"""
    Find a shift of `f` in `K[x]` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x)=f(x-sa)`, `r(x)=\text{Norm}(g(x))` and
    `r` is a square-free polynomial over `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\mathbb{Q}(\sqrt{3})`
    and rings `K[x]` and `k[x]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import sqrt

    >>> K = QQ.algebraic_field(sqrt(3))
    >>> R, x = ring("x", K)
    >>> _, X = ring("x", QQ)

    We can now find a square free norm for a shift of `f`:

    >>> f = x**2 - 1
    >>> s, g, r = R.dup_sqf_norm(f)

    The choice of shift `s` is arbitrary and the particular values returned for
    `g` and `r` are determined by `s`.

    >>> s == 1
    True
    >>> g == x**2 - 2*sqrt(3)*x + 2
    True
    >>> r == X**4 - 8*X**2 + 4
    True

    The invariants are:

    >>> g == f.shift(-s*K.unit)
    True
    >>> g.norm() == r
    True
    >>> r.is_squarefree
    True

    Explanation
    ===========

    This is part of Trager's algorithm for factorizing polynomials over
    algebraic number fields. In particular this function is algorithm
    ``sqfr_norm`` from [Trager76]_.

    See Also
    ========

    dmp_sqf_norm:
        Analogous function for multivariate polynomials over ``k(a)``.
    dmp_norm:
        Computes the norm of `f` directly without any shift.
    dup_ext_factor:
        Function implementing Trager's algorithm that uses this.
    sympy.polys.polytools.sqf_norm:
        High-level interface for using this function.
    """
    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    s, g = 0, dmp_raise(K.mod.to_list(), 1, 0, K.dom)

    while True:
        h, _ = dmp_inject(f, 0, K, front=True)
        r = dmp_resultant(g, h, 1, K.dom)

        if dup_sqf_p(r, K.dom):
            break
        else:
            f, s = dup_shift(f, -K.unit, K), s + 1

    return s, f, r


def _dmp_sqf_norm_shifts(f, u, K):
    """Generate a sequence of candidate shifts for dmp_sqf_norm."""
    #
    # We want to find a minimal shift if possible because shifting high degree
    # variables can be expensive e.g. x**10 -> (x + 1)**10. We try a few easy
    # cases first before the final infinite loop that is guaranteed to give
    # only finitely many bad shifts (see Trager76 for proof of this in the
    # univariate case).
    #

    # First the trivial shift [0, 0, ...]
    n = u + 1
    s0 = [0] * n
    yield s0, f

    # Shift in multiples of the generator of the extension field K
    a = K.unit

    # Variables of degree > 0 ordered by increasing degree
    d = dmp_degree_list(f, u)
    var_indices = [i for di, i in sorted(zip(d, range(u+1))) if di > 0]

    # Now try [1, 0, 0, ...], [0, 1, 0, ...]
    for i in var_indices:
        s1 = s0.copy()
        s1[i] = 1
        a1 = [-a*s1i for s1i in s1]
        f1 = dmp_shift(f, a1, u, K)
        yield s1, f1

    # Now try [1, 1, 1, ...], [2, 2, 2, ...]
    j = 0
    while True:
        j += 1
        sj = [j] * n
        aj = [-a*j] * n
        fj = dmp_shift(f, aj, u, K)
        yield sj, fj


def dmp_sqf_norm(f, u, K):
    r"""
    Find a shift of ``f`` in ``K[X]`` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x_1,x_2,\cdots)=f(x_1-s_1 a, x_2 - s_2 a,
    \cdots)`, `r(x)=\text{Norm}(g(x))` and `r` is a square-free polynomial over
    `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\mathbb{Q}(i)` and rings
    `K[x,y]` and `k[x,y]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import I

    >>> K = QQ.algebraic_field(I)
    >>> R, x, y = ring("x,y", K)
    >>> _, X, Y = ring("x,y", QQ)

    We can now find a square free norm for a shift of `f`:

    >>> f = x*y + y**2
    >>> s, g, r = R.dmp_sqf_norm(f)

    The choice of shifts ``s`` is arbitrary and the particular values returned
    for ``g`` and ``r`` are determined by ``s``.

    >>> s
    [0, 1]
    >>> g == x*y - I*x + y**2 - 2*I*y - 1
    True
    >>> r == X**2*Y**2 + X**2 + 2*X*Y**3 + 2*X*Y + Y**4 + 2*Y**2 + 1
    True

    The required invariants are:

    >>> g == f.shift_list([-si*K.unit for si in s])
    True
    >>> g.norm() == r
    True
    >>> r.is_squarefree
    True

    Explanation
    ===========

    This is part of Trager's algorithm for factorizing polynomials over
    algebraic number fields. In particular this function is a multivariate
    generalization of algorithm ``sqfr_norm`` from [Trager76]_.

    See Also
    ========

    dup_sqf_norm:
        Analogous function for univariate polynomials over ``k(a)``.
    dmp_norm:
        Computes the norm of `f` directly without any shift.
    dmp_ext_factor:
        Function implementing Trager's algorithm that uses this.
    sympy.polys.polytools.sqf_norm:
        High-level interface for using this function.
    """
    if not u:
        s, g, r = dup_sqf_norm(f, K)
        return [s], g, r

    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    g = dmp_raise(K.mod.to_list(), u + 1, 0, K.dom)

    for s, f in _dmp_sqf_norm_shifts(f, u, K):

        h, _ = dmp_inject(f, u, K, front=True)
        r = dmp_resultant(g, h, u + 1, K.dom)

        if dmp_sqf_p(r, u, K.dom):
            break

    return s, f, r


def dmp_norm(f, u, K):
    r"""
    Norm of ``f`` in ``K[X]``, often not square-free.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Examples
    ========

    We first define the algebraic number field `K = k(a) = \mathbb{Q}(\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.sqfreetools import dmp_norm
    >>> k = QQ
    >>> K = k.algebraic_field(sqrt(2))

    We can now compute the norm of a polynomial `p` in `K[x,y]`:

    >>> p = [[K(1)], [K(1),K.unit]]                  # x + y + sqrt(2)
    >>> N = [[k(1)], [k(2),k(0)], [k(1),k(0),k(-2)]] # x**2 + 2*x*y + y**2 - 2
    >>> dmp_norm(p, 1, K) == N
    True

    In higher level functions that is:

    >>> from sympy import expand, roots, minpoly
    >>> from sympy.abc import x, y
    >>> from math import prod
    >>> a = sqrt(2)
    >>> e = (x + y + a)
    >>> e.as_poly([x, y], extension=a).norm()
    Poly(x**2 + 2*x*y + y**2 - 2, x, y, domain='QQ')

    This is equal to the product of the expressions `x + y + a_i` where the
    `a_i` are the conjugates of `a`:

    >>> pa = minpoly(a)
    >>> pa
    _x**2 - 2
    >>> rs = roots(pa, multiple=True)
    >>> rs
    [sqrt(2), -sqrt(2)]
    >>> n = prod(e.subs(a, r) for r in rs)
    >>> n
    (x + y - sqrt(2))*(x + y + sqrt(2))
    >>> expand(n)
    x**2 + 2*x*y + y**2 - 2

    Explanation
    ===========

    Given an algebraic number field `K = k(a)` any element `b` of `K` can be
    represented as polynomial function `b=g(a)` where `g` is in `k[x]`. If the
    minimal polynomial of `a` over `k` is `p_a` then the roots `a_1`, `a_2`,
    `\cdots` of `p_a(x)` are the conjugates of `a`. The norm of `b` is the
    product `g(a1) \times g(a2) \times \cdots` and is an element of `k`.

    As in [Trager76]_ we extend this norm to multivariate polynomials over `K`.
    If `b(x)` is a polynomial in `k(a)[X]` then we can think of `b` as being
    alternately a function `g_X(a)` where `g_X` is an element of `k[X][y]` i.e.
    a polynomial function with coefficients that are elements of `k[X]`. Then
    the norm of `b` is the product `g_X(a1) \times g_X(a2) \times \cdots` and
    will be an element of `k[X]`.

    See Also
    ========

    dmp_sqf_norm:
        Compute a shift of `f` so that the `\text{Norm}(f)` is square-free.
    sympy.polys.polytools.Poly.norm:
        Higher-level function that calls this.
    """
    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    g = dmp_raise(K.mod.to_list(), u + 1, 0, K.dom)
    h, _ = dmp_inject(f, u, K, front=True)

    return dmp_resultant(g, h, u + 1, K.dom)


def dup_gf_sqf_part(f, K):
    """Compute square-free part of ``f`` in ``GF(p)[x]``. """
    f = dup_convert(f, K, K.dom)
    g = gf_sqf_part(f, K.mod, K.dom)
    return dup_convert(g, K.dom, K)


def dmp_gf_sqf_part(f, u, K):
    """Compute square-free part of ``f`` in ``GF(p)[X]``. """
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_sqf_part(f, K):
    """
    Returns square-free part of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_part(x**3 - 3*x - 2)
    x**2 - x - 2

    See Also
    ========

    sympy.polys.polytools.Poly.sqf_part
    """
    if K.is_FiniteField:
        return dup_gf_sqf_part(f, K)

    if not f:
        return f

    if K.is_negative(dup_LC(f, K)):
        f = dup_neg(f, K)

    gcd = dup_gcd(f, dup_diff(f, 1, K), K)
    sqf = dup_quo(f, gcd, K)

    if K.is_Field:
        return dup_monic(sqf, K)
    else:
        return dup_primitive(sqf, K)[1]


def dmp_sqf_part(f, u, K):
    """
    Returns square-free part of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqf_part(x**3 + 2*x**2*y + x*y**2)
    x**2 + x*y

    """
    if not u:
        return dup_sqf_part(f, K)

    if K.is_FiniteField:
        return dmp_gf_sqf_part(f, u, K)

    if dmp_zero_p(f, u):
        return f

    if K.is_negative(dmp_ground_LC(f, u, K)):
        f = dmp_neg(f, u, K)

    gcd = f
    for i in range(u+1):
        gcd = dmp_gcd(gcd, dmp_diff_in(f, 1, i, u, K), u, K)
    sqf = dmp_quo(f, gcd, u, K)

    if K.is_Field:
        return dmp_ground_monic(sqf, u, K)
    else:
        return dmp_ground_primitive(sqf, u, K)[1]


def dup_gf_sqf_list(f, K, all=False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[x]``. """
    f_orig = f

    f = dup_convert(f, K, K.dom)

    coeff, factors = gf_sqf_list(f, K.mod, K.dom, all=all)

    for i, (f, k) in enumerate(factors):
        factors[i] = (dup_convert(f, K.dom, K), k)

    _dup_check_degrees(f_orig, factors)

    return K.convert(coeff, K.dom), factors


def dmp_gf_sqf_list(f, u, K, all=False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[X]``. """
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_sqf_list(f, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Uses Yun's algorithm from [Yun76]_.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list(f)
    (2, [(x + 1, 2), (x + 2, 3)])
    >>> R.dup_sqf_list(f, all=True)
    (2, [(1, 1), (x + 1, 2), (x + 2, 3)])

    See Also
    ========

    dmp_sqf_list:
        Corresponding function for multivariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.

    References
    ==========

    [Yun76]_
    """
    if K.is_FiniteField:
        return dup_gf_sqf_list(f, K, all=all)

    f_orig = f

    if K.is_Field:
        coeff = dup_LC(f, K)
        f = dup_monic(f, K)
    else:
        coeff, f = dup_primitive(f, K)

        if K.is_negative(dup_LC(f, K)):
            f = dup_neg(f, K)
            coeff = -coeff

    if dup_degree(f) <= 0:
        return coeff, []

    result, i = [], 1

    h = dup_diff(f, 1, K)
    g, p, q = dup_inner_gcd(f, h, K)

    while True:
        d = dup_diff(p, 1, K)
        h = dup_sub(q, d, K)

        if not h:
            result.append((p, i))
            break

        g, p, q = dup_inner_gcd(p, h, K)

        if all or dup_degree(g) > 0:
            result.append((g, i))

        i += 1

    _dup_check_degrees(f_orig, result)

    return coeff, result


def dup_sqf_list_include(f, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list_include(f)
    [(2, 1), (x + 1, 2), (x + 2, 3)]
    >>> R.dup_sqf_list_include(f, all=True)
    [(2, 1), (x + 1, 2), (x + 2, 3)]

    """
    coeff, factors = dup_sqf_list(f, K, all=all)

    if factors and factors[0][1] == 1:
        g = dup_mul_ground(factors[0][0], coeff, K)
        return [(g, 1)] + factors[1:]
    else:
        g = dup_strip([coeff])
        return [(g, 1)] + factors


def dmp_sqf_list(f, u, K, all=False):
    """
    Return square-free decomposition of a polynomial in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list(f)
    (1, [(x + y, 2), (x, 3)])
    >>> R.dmp_sqf_list(f, all=True)
    (1, [(1, 1), (x + y, 2), (x, 3)])

    Explanation
    ===========

    Uses Yun's algorithm for univariate polynomials from [Yun76]_ recursively.
    The multivariate polynomial is treated as a univariate polynomial in its
    leading variable. Then Yun's algorithm computes the square-free
    factorization of the primitive and the content is factored recursively.

    It would be better to use a dedicated algorithm for multivariate
    polynomials instead.

    See Also
    ========

    dup_sqf_list:
        Corresponding function for univariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.
    """
    if not u:
        return dup_sqf_list(f, K, all=all)

    if K.is_FiniteField:
        return dmp_gf_sqf_list(f, u, K, all=all)

    f_orig = f

    if K.is_Field:
        coeff = dmp_ground_LC(f, u, K)
        f = dmp_ground_monic(f, u, K)
    else:
        coeff, f = dmp_ground_primitive(f, u, K)

        if K.is_negative(dmp_ground_LC(f, u, K)):
            f = dmp_neg(f, u, K)
            coeff = -coeff

    deg = dmp_degree(f, u)
    if deg < 0:
        return coeff, []

    # Yun's algorithm requires the polynomial to be primitive as a univariate
    # polynomial in its main variable.
    content, f = dmp_primitive(f, u, K)

    result = {}

    if deg != 0:

        h = dmp_diff(f, 1, u, K)
        g, p, q = dmp_inner_gcd(f, h, u, K)

        i = 1

        while True:
            d = dmp_diff(p, 1, u, K)
            h = dmp_sub(q, d, u, K)

            if dmp_zero_p(h, u):
                result[i] = p
                break

            g, p, q = dmp_inner_gcd(p, h, u, K)

            if all or dmp_degree(g, u) > 0:
                result[i] = g

            i += 1

    coeff_content, result_content = dmp_sqf_list(content, u-1, K, all=all)

    coeff *= coeff_content

    # Combine factors of the content and primitive part that have the same
    # multiplicity to produce a list in ascending order of multiplicity.
    for fac, i in result_content:
        fac = [fac]
        if i in result:
            result[i] = dmp_mul(result[i], fac, u, K)
        else:
            result[i] = fac

    result = [(result[i], i) for i in sorted(result)]

    _dmp_check_degrees(f_orig, u, result)

    return coeff, result


def dmp_sqf_list_include(f, u, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list_include(f)
    [(1, 1), (x + y, 2), (x, 3)]
    >>> R.dmp_sqf_list_include(f, all=True)
    [(1, 1), (x + y, 2), (x, 3)]

    """
    if not u:
        return dup_sqf_list_include(f, K, all=all)

    coeff, factors = dmp_sqf_list(f, u, K, all=all)

    if factors and factors[0][1] == 1:
        g = dmp_mul_ground(factors[0][0], coeff, u, K)
        return [(g, 1)] + factors[1:]
    else:
        g = dmp_ground(coeff, u)
        return [(g, 1)] + factors


def dup_gff_list(f, K):
    """
    Compute greatest factorial factorization of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_gff_list(x**5 + 2*x**4 - x**3 - 2*x**2)
    [(x, 1), (x + 2, 4)]

    """
    if not f:
        raise ValueError("greatest factorial factorization doesn't exist for a zero polynomial")

    f = dup_monic(f, K)

    if not dup_degree(f):
        return []
    else:
        g = dup_gcd(f, dup_shift(f, K.one, K), K)
        H = dup_gff_list(g, K)

        for i, (h, k) in enumerate(H):
            g = dup_mul(g, dup_shift(h, -K(k), K), K)
            H[i] = (h, k + 1)

        f = dup_quo(f, g, K)

        if not dup_degree(f):
            return H
        else:
            return [(f, 1)] + H


def dmp_gff_list(f, u, K):
    """
    Compute greatest factorial factorization of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_gff_list(f, K)
    else:
        raise MultivariatePolynomialError(f)
