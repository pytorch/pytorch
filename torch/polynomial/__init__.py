import torch
from torch._C import _add_docstr, _special  # type: ignore[attr-defined]

__all__ = [
    "polyadd",
    "polycompanion",
    "polyder",
    "polydiv",
    "polydomain",
    "polyfit",
    "polyfromroots",
    "polygrid2d",
    "polygrid3d",
    "polyint",
    "polyline",
    "polymul",
    "polymulx",
    "polyone",
    "polypow",
    "polyroots",
    "polysub",
    "polytrim",
    "polyval",
    "polyval2d",
    "polyval3d",
    "polyvalfromroots",
    "polyvander",
    "polyvander2d",
    "polyvander3d",
]

Tensor = torch.Tensor

polyadd = _add_docstr(
    _polynomial.polynomial_polyadd,
    r"""
polyadd(input, *, out=None) -> Tensor

Adds one polynomial to another.

Returns the sum of two polynomials :math:`c_{1} + c_{2}`. The arguments are tensors of coefficients from lowest order term to highest order term, i.e.,

    .. math::
        [1, 2, 3] = 1 + 2x + 3x^{2}.
""" + """
Args:
    c1: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.
    c2: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.

Returns:
    coefficients representing the sum of two polynomials :math:`c_{1}` and :math:`c_{2}`.
""",
)

polycompanion = _add_docstr(
    _polynomial.special_polycompanion,
    r"""
polycompanion(input, *, out=None) -> Tensor

Returns the companion matrix of `c`.

The companion matrix for a power series cannot be made symmetric by scaling the basis, so this function differs from those for the orthogonal polynomials.
""" + """
Args:
    c: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.

Returns:
    companion matrix of shape :math:`(degree, degree)`.
""",
)

polyder = _add_docstr(
    _polynomial.special_polyder,
    r"""
polyder(input, *, out=None) -> Tensor

Differentiate a polynomial.

Returns the polynomial coefficients `c` differentiated `m` times along `dim`. At each iteration the result is multiplied by `scaling_factor` (for use in a linear change of variable). The argument `c` is a tensor of coefficients from lowest order term to highest order term, i.e.,

.. math::
    [1, 2, 3] = 1 + 2x + 3x^{2}.

while

.. math::
    [[1, 2], [1, 2]] = 1 + 1x + 2y + 2xy

if `dim = 0` is :math:`x` and `dim = 1` is :math`y`.
""" + """
Args:
    c: polynomial coefficients. If `c` is multi-dimensional `dim` corresponds to different variables with the degree in each dimension given by the corresponding index.
    order: number of derivatives taken, must be non-negative. Default: 1.
    scaling_factor: each differentiation is multiplied by `scaling_factor`. The end result is multiplication by :math:`scaling_factor^{m}`. This is for use in a linear change of variable. Default: 1.
    dim: Dimension over which the derivative is taken. Default: 0.

Returns:
    polynomial coefficients of the derivative.

Examples:
    >>> c = torch.tensor([1, 2, 3, 4]) # 1 + 2x + 3x^{2} + 4x^{3}
    >>> polyder(c) # d/dx(c) = 2 + 6x + 12x^{2}
    tensor([ 2,  6, 12])
    >>> polyder(c, 3) # d^{3}/dx^{3}(c) = 24
    tensor([24])
    >>> polyder(c, scaling_factor = -1) # d/-dx(c) = -2 - 6x - 12x^{2}
    tensor([ -2,  -6, -12])
    >>> polyder(c, 2, scaling_factor = -1) # d^{2}/-dx^{2}(c) = 6 + 24x
    tensor([ 6, 24])
""",
)

polydiv = _add_docstr(
    _polynomial.special_polyder,
    r"""
polydiv(input, *, out=None) -> Tensor

Divides one polynomial by another.

Returns the quotient and remainder of two polynomials :math:`\\frac{c1}{c2}`. The arguments are tensors of coefficients from lowest order term to highest order term, i.e.,

.. math::
    [1, 2, 3] = 1 + 2x + 3x^{2}.
""" + """
Args:
    c1: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.
    c2: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.

Returns:
    coefficients representing the quotient and the remainder of two polynomials :math:`c_{1}` and :math:`c_{2}`.

Examples:
    >>> c1 = torch.tensor([1, 2, 3])
    >>> c2 = torch.tensor([3, 2, 1])
    >>> polydiv(c1, c2)
    (tensor([3.]), tensor([-8., -4.]))
    >>> polydiv(c2, c1)
    (tensor([0.3333]), tensor([2.6667, 1.3333]))
""",
)

polydomain = _add_docstr(
    _polynomial.special_polydomain,
    r"""
polydomain(input, *, out=None) -> Tensor
""" + """
""",
)

polyfit = _add_docstr(
    _polynomial.special_polyfit,
    r"""
Least-squares fit of a polynomial to data.

Return the coefficients of a polynomial of `degree` that is the least squares fit to the data values `y` given at points `x`. If `y` is one-dimensional the returned coefficients will also be one-dimensional. If `y` is two-dimensional multiple fits are done, one for each column of `y`, and the resulting coefficients are stored in the corresponding columns of a two-dimensional tensor.

The fitted polynomials are in the form:

.. math::
    p(x) = c_{0} + c_{1}x + \\ldots + c_{n}x^{n},

where :math:`n` is `degree`.

Note:
    The solution is the coefficients of the polynomial :math:`p` that minimizes the sum of the weighted squared errors:

    where the :math:`w_{j}` are the weights. This problem is solved by setting up the typically over-determined matrix equation:
    
    where :math:`V` is the weighted pseudo Vandermonde matrix of :math:`x`, :math:`c` are the coefficients to be solved for, :math:`w` are the weights, and :math:`y` are the observed values. This equation is then solved using the singular value decomposition of :math:`V`.

Note:
    If some of the singular values of :math:`V` are so small that they are neglected, a `RankWarning` exception is raised. This means that the coefficient values may be poorly determined. Fitting to a lower order polynomial will usually get rid of the warning (but may not be what you want, of course; if you have independent reasons for choosing the degree which isn’t working, you may have to: a.) reconsider those reasons, and/or b.) reconsider the quality of your data). The `rcond` parameter can also be set to a value smaller than its default, but the resulting fit may be spurious and have large contributions from roundoff error.

Note:
    Polynomial fits using double precision tend to “fail” at about (polynomial) degree 20. Fits using Chebyshev or Legendre series are generally better conditioned, but much can still depend on the distribution of the sample points and the smoothness of the data. If the quality of the fit is inadequate, splines may be a good alternative.
""" + """
Args:
    x: :math:`x`-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    y: :math:`y`-coordinates of the sample points. Several sets of sample points sharing the same :math`x`-coordinates can be (independently) fit with one call to `polyfit` by passing in for `y` a two-dimensional tensor that contains one data set per column.
    degree: Degrees of the fitting polynomials. If `degree` is a scalar all terms up to and including the `degree` term are included in the fit. A tensor specifying the degrees of the terms to include may be used instead.
    rcond: Relative condition number of the fit.  Singular values smaller than `rcond`, relative to the largest singular value, will be ignored. The default value is `len(x) * epsilon`, where `epsilon` is the relative precision of `torch.float`.
    weights: If not `None`, the weight, :math:`w_{i}`, applies to the unsquared residual :math:`y_{i} - \\hat{y}_{i}` at :math:`x_{i}`. Ideally the weights are chosen so that the errors of the products :math:`w_{i}y_{i}` all have the same variance. When using inverse-variance weighting, use :math:`w_{i} = \\frac{1}{\\sigma(y_{i})}`. The default value is `None`.
        
Returns:
    A named tuple (`coefficients`, `residuals`, `rank`, `singular_values`).
""",
)

polyfromroots = _add_docstr(
    _polynomial.special_polyfromroots,
    r"""
polyfromroots(input, *, out=None) -> Tensor

Generate a monic polynomial with given roots.

Return the coefficients of the polynomial:

.. math::
    p(x) = (x - r_{0}) \\times \\ldot \\times (x - r_{n})

where :math:`r_{n}` are the `roots`. If a zero has multiplicity :math:`n`, then it must appear in `roots` :math:`n` times. For instance, if 2 is a root of multiplicity three and 3 is a root of multiplicity 2, then `roots` looks something like `tensor([2, 2, 2, 3, 3])`. The roots can appear in any order.

If the returned coefficients are `c`, then:

.. math::
    p(x) = c_{0} + c_{1}x + \\ldots +  x^{n}

The coefficient of the last term is 1 for monic polynomials in this form.
""" + """
Args:
    roots: sequence containing the roots.

Returns:
    one-dimensional tensor of the polynomial’s coefficients If all the roots are real, then `out` is also real, otherwise it is complex.

Note:
    The coefficients are determined by multiplying together linear factors of the form ``(x - r_i)``, i.e.

    .. math::
        p(x) = (x - r_0) (x - r_1) ... (x - r_n)

    where ``n == len(roots) - 1``; note that this implies that ``1`` is always returned for :math:`a_n`.

Example:
    >>> polyfromroots(torch.tensor([-1, 0, 1]))
    torch.tensor([ 0., -1.,  0.,  1.])
    >>> j = complex(0, 1)
    >>> polyfromroots(torch.tensor([-j, j]))
    torch.tensor([1.+0.j,  0.+0.j,  1.+0.j])
""",
)

polygrid2d = _add_docstr(
    _polynomial.special_polygrid2d,
    r"""
polygrid2d(input, *, out=None) -> Tensor

Evaluates a two-dimensional polynomial on the Cartesian product of :math:`x` and :math:`y`.

This function returns the values:

.. math::
    p(a, b) = \\sum_{i, j} c_{i, j} a^{i} b^{j}

where the points :math:`(a, b)` consist of all pairs formed by taking :math:`a` from :math:`x` and :math:`b` from :math:`y`. The resulting points form a grid with :math:`x` in the first dimension and :math:`y` in the second dimension.

If :math:`c` has fewer than two dimensions, ones are implicitly appended to its shape to make it two-dimensional. The shape of the result will be c.shape[2:] + x.shape + y.shape.
""" + """
Args:
    x: one-dimensional series.
    y: one-dimensional series.
    c: coefficients ordered so that the coefficients for terms of degree :math:`(i, j)` are contained in :math:`c[i, j]`. If :math:`c` has a dimensions greater than two the remaining indices enumerate multiple sets of coefficients.

Returns:
    values of the three-dimensional polynomial at points in the Cartesian product of :math:`x` and :math:`y`.
""",
)

polygrid3d = _add_docstr(
    _polynomial.special_polygrid3d,
    r"""
polygrid3d(input, *, out=None) -> Tensor

Evaluates a three-dimensional polynomial on the Cartesian product of :math:`x`, :math:`y`, and  :math:`z`.

This function returns the values:

.. math::
    p(a,b,c) = \\sum_{i, j, k} c_{i, j, k} a^{i} b^{j} c^{k}

where the points :math:`(a, b, c)` consist of all triples formed by taking :math:`a` from :math:`x`, :math:`b` from :math:`y`, and :math:`c` from :math:`z`. The resulting points form a grid with :math:`x` in the first dimension, `y` in the second dimension, and `z` in the third dimension.

If :math:`c` has fewer than three dimensions, ones are implicitly appended to its shape to make it two-dimensional. The shape of the result will be c.shape[3:] + x.shape + y.shape + z.shape.
""" + """
Args:
    x: one-dimensional series.
    y: one-dimensional series.
    z: one-dimensional series.
    c: coefficients ordered so that the coefficients for terms of degree :math:`(i, j, k)` are contained in :math:`c[i, j, k]`. If :math:`c` has a dimensions greater than three the remaining indices enumerate multiple sets of coefficients.

Returns:
    values of the three-dimensional polynomial at points in the Cartesian product of :math:`x`, :math:`y`, and :math:`z`.
""",
)

polyint = _add_docstr(
    _polynomial.special_polyint,
    r"""
polyint(input, *, out=None) -> Tensor

Integrate a polynomial.

Returns the polynomial coefficients `c` integrated `m` times from `lower_bound` along `dim`. At each iteration the resulting series is multiplied by `scaling_factor` and an integration constant, `k`, is added. The scaling factor is for use in a linear change of variable. The argument `c` is a tensor of coefficients, from low to high degree along each axis:

.. math::
    [1, 2, 3] = 1 + 2x + 3x^{2}`

while

.. math::
    [[1, 2], [1, 2]] = 1 + 1x + 2y + 2xy

if `dim = 0` is :math:`x` and `dim = 1` is :math`y`.

Note:
    Note that the result of each integration is multiplied by `scaling_factor`. Why is this important to note? Say one is making a linear change of variable in an integral relative to `x`. Then

    dx = \\frac{du}{a}, so one will need to set scl equal to  - perhaps not what one would have first thought.
""" + """
Args:
    c: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.
    m: order of integration, must be positive. Default: 1.
    k: Integration constants. The value of the first integral at zero is the first value in the list, the value of the second integral at zero is the second value, etc. If k == [] (the default), all constants are set to zero. If m == 1, a single scalar can be given instead of a list.
    lower_bound: The lower bound of the integral. Default: 0.
    scaling_factor: Following each integration the result is multiplied by `scaling_factor` before the integration constant is added. Default: 1.
    dim: Dimension over which the integral is taken. Default: 0.

Returns:
    coefficient tensor of the integral.
""",
)

polyline = _add_docstr(
    _polynomial.special_polyline,
    r"""
polyline(input, *, out=None) -> Tensor

Returns a tensor representing a linear polynomial.
""" + """
Args:
    offset, scale: the "y-intercept" and "slope" of the line, respectively.

Returns:
    this module's representation of the linear polynomial ``off + scl*x``.

Examples:
    >>> polyline(1, -1)
    torch.tensor([ 1, -1])
    >>> polyval(torch.tensor(1), polyline(1, -1))
    0
""",
)

polymul = _add_docstr(
    _polynomial.special_polymul,
    r"""
polymul(input, *, out=None) -> Tensor

Multiply one polynomial by another.
""" + """
Args:
    c1: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.
    c2: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.

Returns:
    the coefficients of their product.
""",
)

polymulx = _add_docstr(
    _polynomial.special_polymulx,
    r"""
polymulx(input, *, out=None) -> Tensor

Multiply a polynomial by x.

Multiply the polynomial `c` by x, where x is the independent variable.
""" + """
Args:
    c: one-dimensional tensor of polynomial coefficients ordered from low to high.

Returns:
    Tensor representing the result of the multiplication.
""",
)

polyone = _add_docstr(
    _polynomial.special_polyone,
    r"""
polyone(input, *, out=None) -> Tensor
""" + """
""",
)

polypow = _add_docstr(
    _polynomial.special_polypow,
    r"""
polypow(input, *, out=None) -> Tensor

Raise a polynomial to a power.

Returns the polynomial `c` raised to the power `pow`. The argument `c` is a sequence of coefficients ordered from low to high. i.e., [1,2,3] is the series  ``1 + 2*x + 3*x**2.``
""" + """
Args:
    coefficients: one-dimensional tensor of tensor of series coefficients ordered from low to high degree.
    exponent : power to which the series will be raised
    maximum_exponent: Maximum power allowed. This is mainly to limit growth of the series to unmanageable size. Default is 16

Returns:
    power series of power.
""",
)

polyroots = _add_docstr(
    _polynomial.special_polyroots,
    r"""
polyroots(input, *, out=None) -> Tensor

Compute the roots of a polynomial.

    Return the roots (a.k.a. “zeros”) of the polynomial:

    .. math::
        p(x) = \\sum_i c_{i}x^{i}.

    Note:
        The root estimates are obtained as the eigenvalues of the companion matrix, roots far from the origin of the complex plane may have large errors due to the numerical instability of the power series for such values. Roots with multiplicity greater than 1 will also show larger errors as the value of the series near such points is relatively insensitive to errors in the roots. Isolated roots near the origin can be improved by a few iterations of Newton’s method.
""" + """
Args:
    c: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.

Returns:
    roots of the polynomial.
""",
)

polysub = _add_docstr(
    _polynomial.special_polysub,
    r"""
polysub(input, *, out=None) -> Tensor

Subtracts one polynomial from another.

Returns the difference of two polynomials :math:`c_{1} - c_{2}`. The arguments are sequences of coefficients from lowest order term to highest order term, i.e.,

.. math::
    [1, 2, 3] = 1 + 2x + 3x^{2}.
""" + """
Args:
    c1: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.
    c2: one-dimensional polynomial coefficients ordered from lowest order term to highest order term.

Returns:
    coefficients representing the difference of two polynomials :math:`c_{1}` and :math:`c_{2}`.
""",
)

polytrim = _add_docstr(
    _polynomial.special_polytrim,
    r"""
polytrim(input, *, out=None) -> Tensor
""" + """
""",
)

polyval = _add_docstr(
    _polynomial.special_polyval,
    r"""
polyval(input, *, out=None) -> Tensor

Evaluates a polynomial at :math:`x`.

If :math:`c` is of length :math:`n + 1`, this function returns the value:

.. math::
    p(x) = c_{0} + c_{1} x + ... + c_{n} x^{n}

If :math:`c` is a one-dimensional tensor, then :math:`p(x)` will have the same shape as :math:`x`. If :math:`c` is multi-dimensional, then the shape of the result depends on the value of `expand`. If `expand` is true the shape will be c.size()[1:] + x.size(). If `expand` is `False` the shape will be c.size()[1:].

Note:
    Trailing zeros in the coefficients will be used in the evaluation, so they should be avoided if efficiency is a concern.
""" + """
Args:
    x: points.
    c: one-dimensional polynomial coefficients ordered so that the coefficients for terms of degree :math:`n` are contained in :math:`c[n]`. If :math:`c` is multi-dimensional the remaining indices enumerate multiple polynomials. In the two-dimensional case the coefficients may be thought of as stored in the columns of :math:`c`.
    expand: if true, the shape of the coefficient tensor is extended with ones on the right, one for each dimension of :math:`x`. The result is that every column of coefficients in :math:`c` is evaluated for every element of :math:`x`. If false, :math:`x` is broadcast over the columns of :math:`c` for the evaluation. This keyword is useful when :math:`c` is multi-dimensional. The default value is ``True``.

Returns:
    the shape of the returned tensor is described above.
""",
)

polyval2d = _add_docstr(
    _polynomial.special_polyval2d,
    r"""
polyval2d(input, *, out=None) -> Tensor

Evaluate a two-dimensional polynomial at points (x, y).

This function returns the value

.. math::
    p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

If `c` has fewer than two dimensions, ones are implicitly appended to its shape to make it two-dimensional. The shape of the result will be c.size()[2:] + x.size().
""" + """
Args:
    x, y: two-dimensional series evaluated at the points :math:`(x, y)`, where `x` and `y` must have the same shape.
    c: coefficients ordered so that the coefficient of the term of multi-degree :math:`i`, :math:`j` is contained in :math:`c_ij`. If `c` has a dimension greater than two the remaining indices enumerate multiple sets of coefficients.

Returns:
    values of the two dimensional polynomial at points formed with pairs of corresponding values from `x` and `y`.
""",
)

polyval3d = _add_docstr(
    _polynomial.special_polyval3d,
    r"""
polyval3d(input, *, out=None) -> Tensor

Evaluate a three-dimensional polynomial at points (x, y, z).

This function returns the values:

.. math::
    p(x, y, z) = \\sum_{i, j, k} c_{i, j, k}x^{i}y^{j}z^{k}

The parameters `x`, `y`, and `z` are converted to tensors only if they are tuples or a lists, otherwise they are treated as a scalars and they must have the same shape after conversion. In either case, either `x`, `y`, and `z` or their elements must support multiplication and addition both with themselves and with the elements of `c`.

If `c` has fewer than 3 dimensions, ones are implicitly appended to its shape to make it three-dimensional. The shape of the result will be c.shape[3:] + x.shape.
""" + """
Args:
    x, y, z: The three dimensional series is evaluated at the points `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If any of `x`, `y`, or `z` is a list or tuple, it is first converted to an ndarray, otherwise it is left unchanged and if it isn't an ndarray it is  treated as a scalar.
    c: Array of coefficients ordered so that the coefficient of the term of multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension greater than 3 the remaining indices enumerate multiple sets of coefficients.

Returns:
    The values of the multidimensional polynomial on points formed with triples of corresponding values from `x`, `y`, and `z`.
""",
)

polyvalfromroots = _add_docstr(
    _polynomial.special_polyvalfromroots,
    r"""
polyvalfromroots(input, *, out=None) -> Tensor

Evaluate a polynomial specified by its roots at points x.

If `r` is of length `N`, this function returns the value

.. math::
    p(x) = \\prod_{n=1}^{N} (x - r_n)

The parameter `x` is converted to an tensor only if it is a tuple or a list, otherwise it is treated as a scalar. In either case, either `x` or its elements must support multiplication and addition both with themselves and with the elements of `r`.

If `r` is a one-dimensional tensor, then `p(x)` will have the same shape as `x`.  If `r` is multidimensional, then the shape of the result depends on the value of `tensor`. If `tensor is ``True`` the shape will be r.shape[1:] + x.shape; that is, each polynomial is evaluated at every value of `x`. If `tensor` is ``False``, the shape will be r.shape[1:]; that is, each polynomial is evaluated only for the corresponding broadcast value of `x`. Note that scalars have shape (,).
""" + """
Args:
    x: If `x` is a list or tuple, it is converted to an ndarray, otherwise it is left unchanged and treated as a scalar. In either case, `x` or its elements must support addition and multiplication with with themselves and with the elements of `r`.
    r: Array of roots. If `r` is multidimensional the first index is the root index, while the remaining indices enumerate multiple polynomials. For instance, in the two dimensional case the roots of each polynomial may be thought of as stored in the columns of `r`.
    expand : If True, the shape of the roots tensor is extended with ones on the right, one for each dimension of `x`. Scalars have dimension 0 for this action. The result is that every column of coefficients in `r` is evaluated for every element of `x`. If False, `x` is broadcast over the columns of `r` for the evaluation.  This keyword is useful when `r` is multidimensional. The default value is True.

Returns:
    The shape of the returned tensor is described above.
""",
)

polyvander = _add_docstr(
    _polynomial.special_polyvander,
    r"""
polyvander(input, *, out=None) -> Tensor

Vandermonde matrix of given degree.

Returns the Vandermonde matrix of degree :math:`degree` and sample points :math:`x`. The Vandermonde matrix is defined by:

.. math::
    V[..., i] = x^{i},

where:

.. math::
    0 <= i <= deg.

The leading indices of :math:`V` index the elements of :math:`x` and the last index is the power of :math:`x`.

If :math:`c` is a one-dimensional tensor of coefficients of length :math:`n + 1` and :math:`V` is the tensor ``V = polyvander(x, n)``, then ``torch.dot(V, c)`` and ``polyval(x, c)`` are the same up to roundoff. This equivalence is useful both for least squares fitting and for the evaluation of a large number of polynomials of the same degree and sample points.
""" + """
Args:
    x: Array of points. The dtype is converted to float64 or complex128 depending on whether any of the elements are complex. If `x` is scalar it is converted to a one-dimensional tensor.
    degree: Degree of the resulting matrix.

Returns:
    The Vandermonde matrix. The shape of the returned matrix is ``x.shape + (deg + 1,)``, where the last index is the power of `x`. The dtype will be the same as the converted `x`.
""",
)

polyvander2d = _add_docstr(
    _polynomial.special_polyvander2d,
    r"""
polyvander2d(input, *, out=None) -> Tensor

Pseudo-Vandermonde matrix of given degrees.

Returns the pseudo-Vandermonde matrix of degrees `deg` and sample points `(x, y)`. The pseudo-Vandermonde matrix is defined by:

.. math::
    V[..., (deg[1] + 1)*i + j] = x^i * y^j,

where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of `V` index the points `(x, y)` and the last index encodes the powers of `x` and `y`.

If ``V = polyvander2d(x, y, [xdeg, ydeg])``, then the columns of `V` correspond to the elements of a two-dimensional coefficient tensor `c` of shape (xdeg + 1, ydeg + 1) in the order

.. math::
    c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

and ``torch.p.dot(V, c.flat)`` and ``polyval2d(x, y, c)`` will be the same up to roundoff. This equivalence is useful both for least squares fitting and for the evaluation of a large number of two-dimensional polynomials of the same degrees and sample points.
""" + """
Args:
    x, y : Arrays of point coordinates, all of the same shape. The dtypes will be converted to either float64 or complex128 depending on whether any of the elements are complex. Scalars are converted to one-dimensional arrays. degrees : List of maximum degrees of the form [x_deg, y_deg].

Returns:
    The shape of the returned matrix is ``x.shape + (order,)``, where :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same as the converted `x` and `y`.
""",
)

polyvander3d = _add_docstr(
    _polynomial.special_polyvander3d,
    r"""
polyvander3d(input, *, out=None) -> Tensor

Pseudo-Vandermonde matrix of given degrees.

Returns the pseudo-Vandermonde matrix of degrees ``degrees`` and sample points ``(x, y, z)``. If ``l, m, n`` are the given degrees in ``x, y, z``, then the pseudo-Vandermonde matrix is defined by:

.. math::
    V[..., (m+1)(n+1)i + (n+1)j + k] = x^i * y^j * z^k,

where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading indices of ``V`` index the points ``(x, y, z)`` and the last index encodes the powers of ``x``, ``y``, and ``z``.

If ``V = polyvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns of ``V`` correspond to the elements of a three-dimensional coefficient tensor ``c`` of shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order:

.. math::
    c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

and ``torch.dot(V, c.flat)`` and ``polyval3d(x, y, z, c)`` will be the same up to roundoff. This equivalence is useful both for least squares fitting and for the evaluation of a large number of three-dimensional polynomials of the same degrees and sample points.
""" + """
Args:
    x, y, z: Arrays of point coordinates, all of the same shape. The dtypes will be converted to either float64 or complex128 depending on whether any of the elements are complex. Scalars are converted to one-dimensional arrays.
    degrees: List of maximum degrees of the form [x_deg, y_deg, z_deg].

Returns:
    the shape of the returned matrix is ``x.shape + (order,)``, where :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`. The dtype will be the same as the converted `x`, `y`, and `z`.
""",
)
