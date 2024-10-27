"""High-level polynomials manipulation functions. """


from sympy.core import S, Basic, symbols, Dummy
from sympy.polys.polyerrors import (
    PolificationFailed, ComputationFailed,
    MultivariatePolynomialError, OptionError)
from sympy.polys.polyoptions import allowed_flags, build_options
from sympy.polys.polytools import poly_from_expr, Poly
from sympy.polys.specialpolys import (
    symmetric_poly, interpolating_poly)
from sympy.polys.rings import sring
from sympy.utilities import numbered_symbols, take, public

@public
def symmetrize(F, *gens, **args):
    r"""
    Rewrite a polynomial in terms of elementary symmetric polynomials.

    A symmetric polynomial is a multivariate polynomial that remains invariant
    under any variable permutation, i.e., if `f = f(x_1, x_2, \dots, x_n)`,
    then `f = f(x_{i_1}, x_{i_2}, \dots, x_{i_n})`, where
    `(i_1, i_2, \dots, i_n)` is a permutation of `(1, 2, \dots, n)` (an
    element of the group `S_n`).

    Returns a tuple of symmetric polynomials ``(f1, f2, ..., fn)`` such that
    ``f = f1 + f2 + ... + fn``.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import symmetrize
    >>> from sympy.abc import x, y

    >>> symmetrize(x**2 + y**2)
    (-2*x*y + (x + y)**2, 0)

    >>> symmetrize(x**2 + y**2, formal=True)
    (s1**2 - 2*s2, 0, [(s1, x + y), (s2, x*y)])

    >>> symmetrize(x**2 - y**2)
    (-2*x*y + (x + y)**2, -2*y**2)

    >>> symmetrize(x**2 - y**2, formal=True)
    (s1**2 - 2*s2, -2*y**2, [(s1, x + y), (s2, x*y)])

    """
    allowed_flags(args, ['formal', 'symbols'])

    iterable = True

    if not hasattr(F, '__iter__'):
        iterable = False
        F = [F]

    R, F = sring(F, *gens, **args)
    gens = R.symbols

    opt = build_options(gens, args)
    symbols = opt.symbols
    symbols = [next(symbols) for i in range(len(gens))]

    result = []

    for f in F:
        p, r, m = f.symmetrize()
        result.append((p.as_expr(*symbols), r.as_expr(*gens)))

    polys = [(s, g.as_expr()) for s, (_, g) in zip(symbols, m)]

    if not opt.formal:
        for i, (sym, non_sym) in enumerate(result):
            result[i] = (sym.subs(polys), non_sym)

    if not iterable:
        result, = result

    if not opt.formal:
        return result
    else:
        if iterable:
            return result, polys
        else:
            return result + (polys,)


@public
def horner(f, *gens, **args):
    """
    Rewrite a polynomial in Horner form.

    Among other applications, evaluation of a polynomial at a point is optimal
    when it is applied using the Horner scheme ([1]).

    Examples
    ========

    >>> from sympy.polys.polyfuncs import horner
    >>> from sympy.abc import x, y, a, b, c, d, e

    >>> horner(9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5)
    x*(x*(x*(9*x + 8) + 7) + 6) + 5

    >>> horner(a*x**4 + b*x**3 + c*x**2 + d*x + e)
    e + x*(d + x*(c + x*(a*x + b)))

    >>> f = 4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y

    >>> horner(f, wrt=x)
    x*(x*y*(4*y + 2) + y*(2*y + 1))

    >>> horner(f, wrt=y)
    y*(x*y*(4*x + 2) + x*(2*x + 1))

    References
    ==========
    [1] - https://en.wikipedia.org/wiki/Horner_scheme

    """
    allowed_flags(args, [])

    try:
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        return exc.expr

    form, gen = S.Zero, F.gen

    if F.is_univariate:
        for coeff in F.all_coeffs():
            form = form*gen + coeff
    else:
        F, gens = Poly(F, gen), gens[1:]

        for coeff in F.all_coeffs():
            form = form*gen + horner(coeff, *gens, **args)

    return form


@public
def interpolate(data, x):
    """
    Construct an interpolating polynomial for the data points
    evaluated at point x (which can be symbolic or numeric).

    Examples
    ========

    >>> from sympy.polys.polyfuncs import interpolate
    >>> from sympy.abc import a, b, x

    A list is interpreted as though it were paired with a range starting
    from 1:

    >>> interpolate([1, 4, 9, 16], x)
    x**2

    This can be made explicit by giving a list of coordinates:

    >>> interpolate([(1, 1), (2, 4), (3, 9)], x)
    x**2

    The (x, y) coordinates can also be given as keys and values of a
    dictionary (and the points need not be equispaced):

    >>> interpolate([(-1, 2), (1, 2), (2, 5)], x)
    x**2 + 1
    >>> interpolate({-1: 2, 1: 2, 2: 5}, x)
    x**2 + 1

    If the interpolation is going to be used only once then the
    value of interest can be passed instead of passing a symbol:

    >>> interpolate([1, 4, 9], 5)
    25

    Symbolic coordinates are also supported:

    >>> [(i,interpolate((a, b), i)) for i in range(1, 4)]
    [(1, a), (2, b), (3, -a + 2*b)]
    """
    n = len(data)

    if isinstance(data, dict):
        if x in data:
            return S(data[x])
        X, Y = list(zip(*data.items()))
    else:
        if isinstance(data[0], tuple):
            X, Y = list(zip(*data))
            if x in X:
                return S(Y[X.index(x)])
        else:
            if x in range(1, n + 1):
                return S(data[x - 1])
            Y = list(data)
            X = list(range(1, n + 1))

    try:
        return interpolating_poly(n, x, X, Y).expand()
    except ValueError:
        d = Dummy()
        return interpolating_poly(n, d, X, Y).expand().subs(d, x)


@public
def rational_interpolate(data, degnum, X=symbols('x')):
    """
    Returns a rational interpolation, where the data points are element of
    any integral domain.

    The first argument  contains the data (as a list of coordinates). The
    ``degnum`` argument is the degree in the numerator of the rational
    function. Setting it too high will decrease the maximal degree in the
    denominator for the same amount of data.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import rational_interpolate

    >>> data = [(1, -210), (2, -35), (3, 105), (4, 231), (5, 350), (6, 465)]
    >>> rational_interpolate(data, 2)
    (105*x**2 - 525)/(x + 1)

    Values do not need to be integers:

    >>> from sympy import sympify
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = sympify("[-1, 0, 2, 22/5, 7, 68/7]")
    >>> rational_interpolate(zip(x, y), 2)
    (3*x**2 - 7*x + 2)/(x + 1)

    The symbol for the variable can be changed if needed:
    >>> from sympy import symbols
    >>> z = symbols('z')
    >>> rational_interpolate(data, 2, X=z)
    (105*z**2 - 525)/(z + 1)

    References
    ==========

    .. [1] Algorithm is adapted from:
           http://axiom-wiki.newsynthesis.org/RationalInterpolation

    """
    from sympy.matrices.dense import ones

    xdata, ydata = list(zip(*data))

    k = len(xdata) - degnum - 1
    if k < 0:
        raise OptionError("Too few values for the required degree.")
    c = ones(degnum + k + 1, degnum + k + 2)
    for j in range(max(degnum, k)):
        for i in range(degnum + k + 1):
            c[i, j + 1] = c[i, j]*xdata[i]
    for j in range(k + 1):
        for i in range(degnum + k + 1):
            c[i, degnum + k + 1 - j] = -c[i, k - j]*ydata[i]
    r = c.nullspace()[0]
    return (sum(r[i] * X**i for i in range(degnum + 1))
            / sum(r[i + degnum + 1] * X**i for i in range(k + 1)))


@public
def viete(f, roots=None, *gens, **args):
    """
    Generate Viete's formulas for ``f``.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import viete
    >>> from sympy import symbols

    >>> x, a, b, c, r1, r2 = symbols('x,a:c,r1:3')

    >>> viete(a*x**2 + b*x + c, [r1, r2], x)
    [(r1 + r2, -b/a), (r1*r2, c/a)]

    """
    allowed_flags(args, [])

    if isinstance(roots, Basic):
        gens, roots = (roots,) + gens, None

    try:
        f, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('viete', 1, exc)

    if f.is_multivariate:
        raise MultivariatePolynomialError(
            "multivariate polynomials are not allowed")

    n = f.degree()

    if n < 1:
        raise ValueError(
            "Cannot derive Viete's formulas for a constant polynomial")

    if roots is None:
        roots = numbered_symbols('r', start=1)

    roots = take(roots, n)

    if n != len(roots):
        raise ValueError("required %s roots, got %s" % (n, len(roots)))

    lc, coeffs = f.LC(), f.all_coeffs()
    result, sign = [], -1

    for i, coeff in enumerate(coeffs[1:]):
        poly = symmetric_poly(i + 1, roots)
        coeff = sign*(coeff/lc)
        result.append((poly, coeff))
        sign = -sign

    return result
