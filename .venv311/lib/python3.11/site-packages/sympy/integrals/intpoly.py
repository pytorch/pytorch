"""
Module to implement integration of uni/bivariate polynomials over
2D Polytopes and uni/bi/trivariate polynomials over 3D Polytopes.

Uses evaluation techniques as described in Chin et al. (2015) [1].


References
===========

.. [1] Chin, Eric B., Jean B. Lasserre, and N. Sukumar. "Numerical integration
of homogeneous functions on convex and nonconvex polygons and polyhedra."
Computational Mechanics 56.6 (2015): 967-981

PDF link : http://dilbert.engr.ucdavis.edu/~suku/quadrature/cls-integration.pdf
"""

from functools import cmp_to_key

from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify


def polytope_integrate(poly, expr=None, *, clockwise=False, max_degree=None):
    """Integrates polynomials over 2/3-Polytopes.

    Explanation
    ===========

    This function accepts the polytope in ``poly`` and the function in ``expr``
    (uni/bi/trivariate polynomials are implemented) and returns
    the exact integral of ``expr`` over ``poly``.

    Parameters
    ==========

    poly : The input Polygon.

    expr : The input polynomial.

    clockwise : Binary value to sort input points of 2-Polytope clockwise.(Optional)

    max_degree : The maximum degree of any monomial of the input polynomial.(Optional)

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Point, Polygon
    >>> from sympy.integrals.intpoly import polytope_integrate
    >>> polygon = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    >>> polys = [1, x, y, x*y, x**2*y, x*y**2]
    >>> expr = x*y
    >>> polytope_integrate(polygon, expr)
    1/4
    >>> polytope_integrate(polygon, polys, max_degree=3)
    {1: 1, x: 1/2, y: 1/2, x*y: 1/4, x*y**2: 1/6, x**2*y: 1/6}
    """
    if clockwise:
        if isinstance(poly, Polygon):
            poly = Polygon(*point_sort(poly.vertices), evaluate=False)
        else:
            raise TypeError("clockwise=True works for only 2-Polytope"
                            "V-representation input")

    if isinstance(poly, Polygon):
        # For Vertex Representation(2D case)
        hp_params = hyperplane_parameters(poly)
        facets = poly.sides
    elif len(poly[0]) == 2:
        # For Hyperplane Representation(2D case)
        plen = len(poly)
        if len(poly[0][0]) == 2:
            intersections = [intersection(poly[(i - 1) % plen], poly[i],
                                          "plane2D")
                             for i in range(0, plen)]
            hp_params = poly
            lints = len(intersections)
            facets = [Segment2D(intersections[i],
                                intersections[(i + 1) % lints])
                      for i in range(lints)]
        else:
            raise NotImplementedError("Integration for H-representation 3D"
                                      "case not implemented yet.")
    else:
        # For Vertex Representation(3D case)
        vertices = poly[0]
        facets = poly[1:]
        hp_params = hyperplane_parameters(facets, vertices)

        if max_degree is None:
            if expr is None:
                raise TypeError('Input expression must be a valid SymPy expression')
            return main_integrate3d(expr, facets, vertices, hp_params)

    if max_degree is not None:
        result = {}
        if expr is not None:
            f_expr = []
            for e in expr:
                _ = decompose(e)
                if len(_) == 1 and not _.popitem()[0]:
                    f_expr.append(e)
                elif Poly(e).total_degree() <= max_degree:
                    f_expr.append(e)
            expr = f_expr

        if not isinstance(expr, list) and expr is not None:
            raise TypeError('Input polynomials must be list of expressions')

        if len(hp_params[0][0]) == 3:
            result_dict = main_integrate3d(0, facets, vertices, hp_params,
                                           max_degree)
        else:
            result_dict = main_integrate(0, facets, hp_params, max_degree)

        if expr is None:
            return result_dict

        for poly in expr:
            poly = _sympify(poly)
            if poly not in result:
                if poly.is_zero:
                    result[S.Zero] = S.Zero
                    continue
                integral_value = S.Zero
                monoms = decompose(poly, separate=True)
                for monom in monoms:
                    monom = nsimplify(monom)
                    coeff, m = strip(monom)
                    integral_value += result_dict[m] * coeff
                result[poly] = integral_value
        return result

    if expr is None:
        raise TypeError('Input expression must be a valid SymPy expression')

    return main_integrate(expr, facets, hp_params)


def strip(monom):
    if monom.is_zero:
        return S.Zero, S.Zero
    elif monom.is_number:
        return monom, S.One
    else:
        coeff = LC(monom)
        return coeff, monom / coeff

def _polynomial_integrate(polynomials, facets, hp_params):
    dims = (x, y)
    dim_length = len(dims)
    integral_value = S.Zero
    for deg in polynomials:
        poly_contribute = S.Zero
        facet_count = 0
        for hp in hp_params:
            value_over_boundary = integration_reduction(facets,
                                                        facet_count,
                                                        hp[0], hp[1],
                                                        polynomials[deg],
                                                        dims, deg)
            poly_contribute += value_over_boundary * (hp[1] / norm(hp[0]))
            facet_count += 1
        poly_contribute /= (dim_length + deg)
        integral_value += poly_contribute

    return integral_value


def main_integrate3d(expr, facets, vertices, hp_params, max_degree=None):
    """Function to translate the problem of integrating uni/bi/tri-variate
    polynomials over a 3-Polytope to integrating over its faces.
    This is done using Generalized Stokes' Theorem and Euler's Theorem.

    Parameters
    ==========

    expr :
        The input polynomial.
    facets :
        Faces of the 3-Polytope(expressed as indices of `vertices`).
    vertices :
        Vertices that constitute the Polytope.
    hp_params :
        Hyperplane Parameters of the facets.
    max_degree : optional
        Max degree of constituent monomial in given list of polynomial.

    Examples
    ========

    >>> from sympy.integrals.intpoly import main_integrate3d, \
    hyperplane_parameters
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> vertices = cube[0]
    >>> faces = cube[1:]
    >>> hp_params = hyperplane_parameters(faces, vertices)
    >>> main_integrate3d(1, faces, vertices, hp_params)
    -125
    """
    result = {}
    dims = (x, y, z)
    dim_length = len(dims)
    if max_degree:
        grad_terms = gradient_terms(max_degree, 3)
        flat_list = [term for z_terms in grad_terms
                     for x_term in z_terms
                     for term in x_term]

        for term in flat_list:
            result[term[0]] = 0

        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            x0 = vertices[facets[facet_count][0]]

            for i, monom in enumerate(flat_list):
                #  Every monomial is a tuple :
                #  (term, x_degree, y_degree, z_degree, value over boundary)
                expr, x_d, y_d, z_d, z_index, y_index, x_index, _ = monom
                degree = x_d + y_d + z_d
                if b.is_zero:
                    value_over_face = S.Zero
                else:
                    value_over_face = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, expr, degree, dims,
                                                      x_index, y_index,
                                                      z_index, x0, grad_terms,
                                                      i, vertices, hp)
                monom[7] = value_over_face
                result[expr] += value_over_face * \
                    (b / norm(a)) / (dim_length + x_d + y_d + z_d)
        return result
    else:
        integral_value = S.Zero
        polynomials = decompose(expr)
        for deg in polynomials:
            poly_contribute = S.Zero
            facet_count = 0
            for i, facet in enumerate(facets):
                hp = hp_params[i]
                if hp[1].is_zero:
                    continue
                pi = polygon_integrate(facet, hp, i, facets, vertices, expr, deg)
                poly_contribute += pi *\
                    (hp[1] / norm(tuple(hp[0])))
                facet_count += 1
            poly_contribute /= (dim_length + deg)
            integral_value += poly_contribute
    return integral_value


def main_integrate(expr, facets, hp_params, max_degree=None):
    """Function to translate the problem of integrating univariate/bivariate
    polynomials over a 2-Polytope to integrating over its boundary facets.
    This is done using Generalized Stokes's Theorem and Euler's Theorem.

    Parameters
    ==========

    expr :
        The input polynomial.
    facets :
        Facets(Line Segments) of the 2-Polytope.
    hp_params :
        Hyperplane Parameters of the facets.
    max_degree : optional
        The maximum degree of any monomial of the input polynomial.

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import main_integrate,\
    hyperplane_parameters
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> hp_params = hyperplane_parameters(triangle)
    >>> main_integrate(x**2 + y**2, facets, hp_params)
    325/6
    """
    dims = (x, y)
    dim_length = len(dims)
    result = {}

    if max_degree:
        grad_terms = [[0, 0, 0, 0]] + gradient_terms(max_degree)

        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            x0 = facets[facet_count].points[0]

            for i, monom in enumerate(grad_terms):
                #  Every monomial is a tuple :
                #  (term, x_degree, y_degree, value over boundary)
                m, x_d, y_d, _ = monom
                value = result.get(m, None)
                degree = S.Zero
                if b.is_zero:
                    value_over_boundary = S.Zero
                else:
                    degree = x_d + y_d
                    value_over_boundary = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, m, degree, dims, x_d,
                                                      y_d, max_degree, x0,
                                                      grad_terms, i)
                monom[3] = value_over_boundary
                if value is not None:
                    result[m] += value_over_boundary * \
                                        (b / norm(a)) / (dim_length + degree)
                else:
                    result[m] = value_over_boundary * \
                                (b / norm(a)) / (dim_length + degree)
        return result
    else:
        if not isinstance(expr, list):
            polynomials = decompose(expr)
            return _polynomial_integrate(polynomials, facets, hp_params)
        else:
            return {e: _polynomial_integrate(decompose(e), facets, hp_params) for e in expr}


def polygon_integrate(facet, hp_param, index, facets, vertices, expr, degree):
    """Helper function to integrate the input uni/bi/trivariate polynomial
    over a certain face of the 3-Polytope.

    Parameters
    ==========

    facet :
        Particular face of the 3-Polytope over which ``expr`` is integrated.
    index :
        The index of ``facet`` in ``facets``.
    facets :
        Faces of the 3-Polytope(expressed as indices of `vertices`).
    vertices :
        Vertices that constitute the facet.
    expr :
        The input polynomial.
    degree :
        Degree of ``expr``.

    Examples
    ========

    >>> from sympy.integrals.intpoly import polygon_integrate
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                 (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                 [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                 [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> facet = cube[1]
    >>> facets = cube[1:]
    >>> vertices = cube[0]
    >>> polygon_integrate(facet, [(0, 1, 0), 5], 0, facets, vertices, 1, 0)
    -25
    """
    expr = S(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = vertices[facet[0]]
    facet_len = len(facet)
    for i, fac in enumerate(facet):
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])
        result += distance_to_side(x0, side, hp_param[0]) *\
            lineseg_integrate(facet, i, side, expr, degree)
    if not expr.is_number:
        expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] +\
            diff(expr, z) * x0[2]
        result += polygon_integrate(facet, hp_param, index, facets, vertices,
                                    expr, degree - 1)
    result /= (degree + 2)
    return result


def distance_to_side(point, line_seg, A):
    """Helper function to compute the signed distance between given 3D point
    and a line segment.

    Parameters
    ==========

    point : 3D Point
    line_seg : Line Segment

    Examples
    ========

    >>> from sympy.integrals.intpoly import distance_to_side
    >>> point = (0, 0, 0)
    >>> distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0))
    -sqrt(2)/2
    """
    x1, x2 = line_seg
    rev_normal = [-1 * S(i)/norm(A) for i in A]
    vector = [x2[i] - x1[i] for i in range(0, 3)]
    vector = [vector[i]/norm(vector) for i in range(0, 3)]

    n_side = cross_product((0, 0, 0), rev_normal, vector)
    vectorx0 = [line_seg[0][i] - point[i] for i in range(0, 3)]
    dot_product = sum(vectorx0[i] * n_side[i] for i in range(0, 3))

    return dot_product


def lineseg_integrate(polygon, index, line_seg, expr, degree):
    """Helper function to compute the line integral of ``expr`` over ``line_seg``.

    Parameters
    ===========

    polygon :
        Face of a 3-Polytope.
    index :
        Index of line_seg in polygon.
    line_seg :
        Line Segment.

    Examples
    ========

    >>> from sympy.integrals.intpoly import lineseg_integrate
    >>> polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]
    >>> line_seg = [(0, 5, 0), (5, 5, 0)]
    >>> lineseg_integrate(polygon, 0, line_seg, 1, 0)
    5
    """
    expr = _sympify(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = line_seg[0]
    distance = norm(tuple([line_seg[1][i] - line_seg[0][i] for i in
                           range(3)]))
    if isinstance(expr, Expr):
        expr_dict = {x: line_seg[1][0],
                     y: line_seg[1][1],
                     z: line_seg[1][2]}
        result += distance * expr.subs(expr_dict)
    else:
        result += distance * expr

    expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] +\
        diff(expr, z) * x0[2]

    result += lineseg_integrate(polygon, index, line_seg, expr, degree - 1)
    result /= (degree + 1)
    return result


def integration_reduction(facets, index, a, b, expr, dims, degree):
    """Helper method for main_integrate. Returns the value of the input
    expression evaluated over the polytope facet referenced by a given index.

    Parameters
    ===========

    facets :
        List of facets of the polytope.
    index :
        Index referencing the facet to integrate the expression over.
    a :
        Hyperplane parameter denoting direction.
    b :
        Hyperplane parameter denoting distance.
    expr :
        The expression to integrate over the facet.
    dims :
        List of symbols denoting axes.
    degree :
        Degree of the homogeneous polynomial.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import integration_reduction,\
    hyperplane_parameters
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> a, b = hyperplane_parameters(triangle)[0]
    >>> integration_reduction(facets, 0, a, b, 1, (x, y), 0)
    5
    """
    expr = _sympify(expr)
    if expr.is_zero:
        return expr

    value = S.Zero
    x0 = facets[index].points[0]
    m = len(facets)
    gens = (x, y)

    inner_product = diff(expr, gens[0]) * x0[0] + diff(expr, gens[1]) * x0[1]

    if inner_product != 0:
        value += integration_reduction(facets, index, a, b,
                                       inner_product, dims, degree - 1)

    value += left_integral2D(m, index, facets, x0, expr, gens)

    return value/(len(dims) + degree - 1)


def left_integral2D(m, index, facets, x0, expr, gens):
    """Computes the left integral of Eq 10 in Chin et al.
    For the 2D case, the integral is just an evaluation of the polynomial
    at the intersection of two facets which is multiplied by the distance
    between the first point of facet and that intersection.

    Parameters
    ==========

    m :
        No. of hyperplanes.
    index :
        Index of facet to find intersections with.
    facets :
        List of facets(Line Segments in 2D case).
    x0 :
        First point on facet referenced by index.
    expr :
        Input polynomial
    gens :
        Generators which generate the polynomial

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import left_integral2D
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> left_integral2D(3, 0, facets, facets[0].points[0], 1, (x, y))
    5
    """
    value = S.Zero
    for j in range(m):
        intersect = ()
        if j in ((index - 1) % m, (index + 1) % m):
            intersect = intersection(facets[index], facets[j], "segment2D")
        if intersect:
            distance_origin = norm(tuple(map(lambda x, y: x - y,
                                             intersect, x0)))
            if is_vertex(intersect):
                if isinstance(expr, Expr):
                    if len(gens) == 3:
                        expr_dict = {gens[0]: intersect[0],
                                     gens[1]: intersect[1],
                                     gens[2]: intersect[2]}
                    else:
                        expr_dict = {gens[0]: intersect[0],
                                     gens[1]: intersect[1]}
                    value += distance_origin * expr.subs(expr_dict)
                else:
                    value += distance_origin * expr
    return value


def integration_reduction_dynamic(facets, index, a, b, expr, degree, dims,
                                  x_index, y_index, max_index, x0,
                                  monomial_values, monom_index, vertices=None,
                                  hp_param=None):
    """The same integration_reduction function which uses a dynamic
    programming approach to compute terms by using the values of the integral
    of previously computed terms.

    Parameters
    ==========

    facets :
        Facets of the Polytope.
    index :
        Index of facet to find intersections with.(Used in left_integral()).
    a, b :
        Hyperplane parameters.
    expr :
        Input monomial.
    degree :
        Total degree of ``expr``.
    dims :
        Tuple denoting axes variables.
    x_index :
        Exponent of 'x' in ``expr``.
    y_index :
        Exponent of 'y' in ``expr``.
    max_index :
        Maximum exponent of any monomial in ``monomial_values``.
    x0 :
        First point on ``facets[index]``.
    monomial_values :
        List of monomial values constituting the polynomial.
    monom_index :
        Index of monomial whose integration is being found.
    vertices : optional
        Coordinates of vertices constituting the 3-Polytope.
    hp_param : optional
        Hyperplane Parameter of the face of the facets[index].

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import (integration_reduction_dynamic, \
            hyperplane_parameters)
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> a, b = hyperplane_parameters(triangle)[0]
    >>> x0 = facets[0].points[0]
    >>> monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5],\
                           [y, 0, 1, 15], [x, 1, 0, None]]
    >>> integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1, 0, 1,\
                                      x0, monomial_values, 3)
    25/2
    """
    value = S.Zero
    m = len(facets)

    if expr == S.Zero:
        return expr

    if len(dims) == 2:
        if not expr.is_number:
            _, x_degree, y_degree, _ = monomial_values[monom_index]
            x_index = monom_index - max_index + \
                x_index - 2 if x_degree > 0 else 0
            y_index = monom_index - 1 if y_degree > 0 else 0
            x_value, y_value =\
                monomial_values[x_index][3], monomial_values[y_index][3]

            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1]

        value += left_integral2D(m, index, facets, x0, expr, dims)
    else:
        # For 3D use case the max_index contains the z_degree of the term
        z_index = max_index
        if not expr.is_number:
            x_degree, y_degree, z_degree = y_index,\
                                           z_index - x_index - y_index, x_index
            x_value = monomial_values[z_index - 1][y_index - 1][x_index][7]\
                if x_degree > 0 else 0
            y_value = monomial_values[z_index - 1][y_index][x_index][7]\
                if y_degree > 0 else 0
            z_value = monomial_values[z_index - 1][y_index][x_index - 1][7]\
                if z_degree > 0 else 0

            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1] \
                + z_degree * z_value * x0[2]

        value += left_integral3D(facets, index, expr,
                                 vertices, hp_param, degree)
    return value / (len(dims) + degree - 1)


def left_integral3D(facets, index, expr, vertices, hp_param, degree):
    """Computes the left integral of Eq 10 in Chin et al.

    Explanation
    ===========

    For the 3D case, this is the sum of the integral values over constituting
    line segments of the face (which is accessed by facets[index]) multiplied
    by the distance between the first point of facet and that line segment.

    Parameters
    ==========

    facets :
        List of faces of the 3-Polytope.
    index :
        Index of face over which integral is to be calculated.
    expr :
        Input polynomial.
    vertices :
        List of vertices that constitute the 3-Polytope.
    hp_param :
        The hyperplane parameters of the face.
    degree :
        Degree of the ``expr``.

    Examples
    ========

    >>> from sympy.integrals.intpoly import left_integral3D
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                 (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                 [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                 [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> facets = cube[1:]
    >>> vertices = cube[0]
    >>> left_integral3D(facets, 3, 1, vertices, ([0, -1, 0], -5), 0)
    -50
    """
    value = S.Zero
    facet = facets[index]
    x0 = vertices[facet[0]]
    facet_len = len(facet)
    for i, fac in enumerate(facet):
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])
        value += distance_to_side(x0, side, hp_param[0]) * \
            lineseg_integrate(facet, i, side, expr, degree)
    return value


def gradient_terms(binomial_power=0, no_of_gens=2):
    """Returns a list of all the possible monomials between
    0 and y**binomial_power for 2D case and z**binomial_power
    for 3D case.

    Parameters
    ==========

    binomial_power :
        Power upto which terms are generated.
    no_of_gens :
        Denotes whether terms are being generated for 2D or 3D case.

    Examples
    ========

    >>> from sympy.integrals.intpoly import gradient_terms
    >>> gradient_terms(2)
    [[1, 0, 0, 0], [y, 0, 1, 0], [y**2, 0, 2, 0], [x, 1, 0, 0],
    [x*y, 1, 1, 0], [x**2, 2, 0, 0]]
    >>> gradient_terms(2, 3)
    [[[[1, 0, 0, 0, 0, 0, 0, 0]]], [[[y, 0, 1, 0, 1, 0, 0, 0],
    [z, 0, 0, 1, 1, 0, 1, 0]], [[x, 1, 0, 0, 1, 1, 0, 0]]],
    [[[y**2, 0, 2, 0, 2, 0, 0, 0], [y*z, 0, 1, 1, 2, 0, 1, 0],
    [z**2, 0, 0, 2, 2, 0, 2, 0]], [[x*y, 1, 1, 0, 2, 1, 0, 0],
    [x*z, 1, 0, 1, 2, 1, 1, 0]], [[x**2, 2, 0, 0, 2, 2, 0, 0]]]]
    """
    if no_of_gens == 2:
        count = 0
        terms = [None] * int((binomial_power ** 2 + 3 * binomial_power + 2) / 2)
        for x_count in range(0, binomial_power + 1):
            for y_count in range(0, binomial_power - x_count + 1):
                terms[count] = [x**x_count*y**y_count,
                                x_count, y_count, 0]
                count += 1
    else:
        terms = [[[[x ** x_count * y ** y_count *
                    z ** (z_count - y_count - x_count),
                    x_count, y_count, z_count - y_count - x_count,
                    z_count, x_count, z_count - y_count - x_count, 0]
                 for y_count in range(z_count - x_count, -1, -1)]
                 for x_count in range(0, z_count + 1)]
                 for z_count in range(0, binomial_power + 1)]
    return terms


def hyperplane_parameters(poly, vertices=None):
    """A helper function to return the hyperplane parameters
    of which the facets of the polytope are a part of.

    Parameters
    ==========

    poly :
        The input 2/3-Polytope.
    vertices :
        Vertex indices of 3-Polytope.

    Examples
    ========

    >>> from sympy import Point, Polygon
    >>> from sympy.integrals.intpoly import hyperplane_parameters
    >>> hyperplane_parameters(Polygon(Point(0, 3), Point(5, 3), Point(1, 1)))
    [((0, 1), 3), ((1, -2), -1), ((-2, -1), -3)]
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> hyperplane_parameters(cube[1:], cube[0])
    [([0, -1, 0], -5), ([0, 0, -1], -5), ([-1, 0, 0], -5),
    ([0, 1, 0], 0), ([1, 0, 0], 0), ([0, 0, 1], 0)]
    """
    if isinstance(poly, Polygon):
        vertices = list(poly.vertices) + [poly.vertices[0]]  # Close the polygon
        params = [None] * (len(vertices) - 1)

        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]

            a1 = v1[1] - v2[1]
            a2 = v2[0] - v1[0]
            b = v2[0] * v1[1] - v2[1] * v1[0]

            factor = gcd_list([a1, a2, b])

            b = S(b) / factor
            a = (S(a1) / factor, S(a2) / factor)
            params[i] = (a, b)
    else:
        params = [None] * len(poly)
        for i, polygon in enumerate(poly):
            v1, v2, v3 = [vertices[vertex] for vertex in polygon[:3]]
            normal = cross_product(v1, v2, v3)
            b = sum(normal[j] * v1[j] for j in range(0, 3))
            fac = gcd_list(normal)
            if fac.is_zero:
                fac = 1
            normal = [j / fac for j in normal]
            b = b / fac
            params[i] = (normal, b)
    return params


def cross_product(v1, v2, v3):
    """Returns the cross-product of vectors (v2 - v1) and (v3 - v1)
    That is : (v2 - v1) X (v3 - v1)
    """
    v2 = [v2[j] - v1[j] for j in range(0, 3)]
    v3 = [v3[j] - v1[j] for j in range(0, 3)]
    return [v3[2] * v2[1] - v3[1] * v2[2],
            v3[0] * v2[2] - v3[2] * v2[0],
            v3[1] * v2[0] - v3[0] * v2[1]]


def best_origin(a, b, lineseg, expr):
    """Helper method for polytope_integrate. Currently not used in the main
    algorithm.

    Explanation
    ===========

    Returns a point on the lineseg whose vector inner product with the
    divergence of `expr` yields an expression with the least maximum
    total power.

    Parameters
    ==========

    a :
        Hyperplane parameter denoting direction.
    b :
        Hyperplane parameter denoting distance.
    lineseg :
        Line segment on which to find the origin.
    expr :
        The expression which determines the best point.

    Algorithm(currently works only for 2D use case)
    ===============================================

    1 > Firstly, check for edge cases. Here that would refer to vertical
        or horizontal lines.

    2 > If input expression is a polynomial containing more than one generator
        then find out the total power of each of the generators.

        x**2 + 3 + x*y + x**4*y**5 ---> {x: 7, y: 6}

        If expression is a constant value then pick the first boundary point
        of the line segment.

    3 > First check if a point exists on the line segment where the value of
        the highest power generator becomes 0. If not check if the value of
        the next highest becomes 0. If none becomes 0 within line segment
        constraints then pick the first boundary point of the line segment.
        Actually, any point lying on the segment can be picked as best origin
        in the last case.

    Examples
    ========

    >>> from sympy.integrals.intpoly import best_origin
    >>> from sympy.abc import x, y
    >>> from sympy import Point, Segment2D
    >>> l = Segment2D(Point(0, 3), Point(1, 1))
    >>> expr = x**3*y**7
    >>> best_origin((2, 1), 3, l, expr)
    (0, 3.0)
    """
    a1, b1 = lineseg.points[0]

    def x_axis_cut(ls):
        """Returns the point where the input line segment
        intersects the x-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        p, q = ls.points
        if p.y.is_zero:
            return tuple(p)
        elif q.y.is_zero:
            return tuple(q)
        elif p.y/q.y < S.Zero:
            return p.y * (p.x - q.x)/(q.y - p.y) + p.x, S.Zero
        else:
            return ()

    def y_axis_cut(ls):
        """Returns the point where the input line segment
        intersects the y-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        p, q = ls.points
        if p.x.is_zero:
            return tuple(p)
        elif q.x.is_zero:
            return tuple(q)
        elif p.x/q.x < S.Zero:
            return S.Zero, p.x * (p.y - q.y)/(q.x - p.x) + p.y
        else:
            return ()

    gens = (x, y)
    power_gens = {}

    for i in gens:
        power_gens[i] = S.Zero

    if len(gens) > 1:
        # Special case for vertical and horizontal lines
        if len(gens) == 2:
            if a[0] == 0:
                if y_axis_cut(lineseg):
                    return S.Zero, b/a[1]
                else:
                    return a1, b1
            elif a[1] == 0:
                if x_axis_cut(lineseg):
                    return b/a[0], S.Zero
                else:
                    return a1, b1

        if isinstance(expr, Expr):  # Find the sum total of power of each
            if expr.is_Add:         # generator and store in a dictionary.
                for monomial in expr.args:
                    if monomial.is_Pow:
                        if monomial.args[0] in gens:
                            power_gens[monomial.args[0]] += monomial.args[1]
                    else:
                        for univariate in monomial.args:
                            term_type = len(univariate.args)
                            if term_type == 0 and univariate in gens:
                                power_gens[univariate] += 1
                            elif term_type == 2 and univariate.args[0] in gens:
                                power_gens[univariate.args[0]] +=\
                                           univariate.args[1]
            elif expr.is_Mul:
                for term in expr.args:
                    term_type = len(term.args)
                    if term_type == 0 and term in gens:
                        power_gens[term] += 1
                    elif term_type == 2 and term.args[0] in gens:
                        power_gens[term.args[0]] += term.args[1]
            elif expr.is_Pow:
                power_gens[expr.args[0]] = expr.args[1]
            elif expr.is_Symbol:
                power_gens[expr] += 1
        else:  # If `expr` is a constant take first vertex of the line segment.
            return a1, b1

        #  TODO : This part is quite hacky. Should be made more robust with
        #  TODO : respect to symbol names and scalable w.r.t higher dimensions.
        power_gens = sorted(power_gens.items(), key=lambda k: str(k[0]))
        if power_gens[0][1] >= power_gens[1][1]:
            if y_axis_cut(lineseg):
                x0 = (S.Zero, b / a[1])
            elif x_axis_cut(lineseg):
                x0 = (b / a[0], S.Zero)
            else:
                x0 = (a1, b1)
        else:
            if x_axis_cut(lineseg):
                x0 = (b/a[0], S.Zero)
            elif y_axis_cut(lineseg):
                x0 = (S.Zero, b/a[1])
            else:
                x0 = (a1, b1)
    else:
        x0 = (b/a[0])
    return x0


def decompose(expr, separate=False):
    """Decomposes an input polynomial into homogeneous ones of
    smaller or equal degree.

    Explanation
    ===========

    Returns a dictionary with keys as the degree of the smaller
    constituting polynomials. Values are the constituting polynomials.

    Parameters
    ==========

    expr : Expr
        Polynomial(SymPy expression).
    separate : bool
        If True then simply return a list of the constituent monomials
        If not then break up the polynomial into constituent homogeneous
        polynomials.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import decompose
    >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5)
    {1: x + y, 2: x**2 + x*y, 5: x**3*y**2 + y**5}
    >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5, True)
    {x, x**2, y, y**5, x*y, x**3*y**2}
    """
    poly_dict = {}

    if isinstance(expr, Expr) and not expr.is_number:
        if expr.is_Symbol:
            poly_dict[1] = expr
        elif expr.is_Add:
            symbols = expr.atoms(Symbol)
            degrees = [(sum(degree_list(monom, *symbols)), monom)
                       for monom in expr.args]
            if separate:
                return {monom[1] for monom in degrees}
            else:
                for monom in degrees:
                    degree, term = monom
                    if poly_dict.get(degree):
                        poly_dict[degree] += term
                    else:
                        poly_dict[degree] = term
        elif expr.is_Pow:
            _, degree = expr.args
            poly_dict[degree] = expr
        else:  # Now expr can only be of `Mul` type
            degree = 0
            for term in expr.args:
                term_type = len(term.args)
                if term_type == 0 and term.is_Symbol:
                    degree += 1
                elif term_type == 2:
                    degree += term.args[1]
            poly_dict[degree] = expr
    else:
        poly_dict[0] = expr

    if separate:
        return set(poly_dict.values())
    return poly_dict


def point_sort(poly, normal=None, clockwise=True):
    """Returns the same polygon with points sorted in clockwise or
    anti-clockwise order.

    Note that it's necessary for input points to be sorted in some order
    (clockwise or anti-clockwise) for the integration algorithm to work.
    As a convention algorithm has been implemented keeping clockwise
    orientation in mind.

    Parameters
    ==========

    poly:
        2D or 3D Polygon.
    normal : optional
        The normal of the plane which the 3-Polytope is a part of.
    clockwise : bool, optional
        Returns points sorted in clockwise order if True and
        anti-clockwise if False.

    Examples
    ========

    >>> from sympy.integrals.intpoly import point_sort
    >>> from sympy import Point
    >>> point_sort([Point(0, 0), Point(1, 0), Point(1, 1)])
    [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]
    """
    pts = poly.vertices if isinstance(poly, Polygon) else poly
    n = len(pts)
    if n < 2:
        return list(pts)

    order = S.One if clockwise else S.NegativeOne
    dim = len(pts[0])
    if dim == 2:
        center = Point(sum((vertex.x for vertex in pts)) / n,
                        sum((vertex.y for vertex in pts)) / n)
    else:
        center = Point(sum((vertex.x for vertex in pts)) / n,
                        sum((vertex.y for vertex in pts)) / n,
                        sum((vertex.z for vertex in pts)) / n)

    def compare(a, b):
        if a.x - center.x >= S.Zero and b.x - center.x < S.Zero:
            return -order
        elif a.x - center.x < 0 and b.x - center.x >= 0:
            return order
        elif a.x - center.x == 0 and b.x - center.x == 0:
            if a.y - center.y >= 0 or b.y - center.y >= 0:
                return -order if a.y > b.y else order
            return -order if b.y > a.y else order

        det = (a.x - center.x) * (b.y - center.y) -\
              (b.x - center.x) * (a.y - center.y)
        if det < 0:
            return -order
        elif det > 0:
            return order

        first = (a.x - center.x) * (a.x - center.x) +\
                (a.y - center.y) * (a.y - center.y)
        second = (b.x - center.x) * (b.x - center.x) +\
                 (b.y - center.y) * (b.y - center.y)
        return -order if first > second else order

    def compare3d(a, b):
        det = cross_product(center, a, b)
        dot_product = sum(det[i] * normal[i] for i in range(0, 3))
        if dot_product < 0:
            return -order
        elif dot_product > 0:
            return order

    return sorted(pts, key=cmp_to_key(compare if dim==2 else compare3d))


def norm(point):
    """Returns the Euclidean norm of a point from origin.

    Parameters
    ==========

    point:
        This denotes a point in the dimension_al spac_e.

    Examples
    ========

    >>> from sympy.integrals.intpoly import norm
    >>> from sympy import Point
    >>> norm(Point(2, 7))
    sqrt(53)
    """
    half = S.Half
    if isinstance(point, (list, tuple)):
        return sum(coord ** 2 for coord in point) ** half
    elif isinstance(point, Point):
        if isinstance(point, Point2D):
            return (point.x ** 2 + point.y ** 2) ** half
        else:
            return (point.x ** 2 + point.y ** 2 + point.z) ** half
    elif isinstance(point, dict):
        return sum(i**2 for i in point.values()) ** half


def intersection(geom_1, geom_2, intersection_type):
    """Returns intersection between geometric objects.

    Explanation
    ===========

    Note that this function is meant for use in integration_reduction and
    at that point in the calling function the lines denoted by the segments
    surely intersect within segment boundaries. Coincident lines are taken
    to be non-intersecting. Also, the hyperplane intersection for 2D case is
    also implemented.

    Parameters
    ==========

    geom_1, geom_2:
        The input line segments.

    Examples
    ========

    >>> from sympy.integrals.intpoly import intersection
    >>> from sympy import Point, Segment2D
    >>> l1 = Segment2D(Point(1, 1), Point(3, 5))
    >>> l2 = Segment2D(Point(2, 0), Point(2, 5))
    >>> intersection(l1, l2, "segment2D")
    (2, 3)
    >>> p1 = ((-1, 0), 0)
    >>> p2 = ((0, 1), 1)
    >>> intersection(p1, p2, "plane2D")
    (0, 1)
    """
    if intersection_type[:-2] == "segment":
        if intersection_type == "segment2D":
            x1, y1 = geom_1.points[0]
            x2, y2 = geom_1.points[1]
            x3, y3 = geom_2.points[0]
            x4, y4 = geom_2.points[1]
        elif intersection_type == "segment3D":
            x1, y1, z1 = geom_1.points[0]
            x2, y2, z2 = geom_1.points[1]
            x3, y3, z3 = geom_2.points[0]
            x4, y4, z4 = geom_2.points[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom:
            t1 = x1 * y2 - y1 * x2
            t2 = x3 * y4 - x4 * y3
            return (S(t1 * (x3 - x4) - t2 * (x1 - x2)) / denom,
                    S(t1 * (y3 - y4) - t2 * (y1 - y2)) / denom)
    if intersection_type[:-2] == "plane":
        if intersection_type == "plane2D":  # Intersection of hyperplanes
            a1x, a1y = geom_1[0]
            a2x, a2y = geom_2[0]
            b1, b2 = geom_1[1], geom_2[1]

            denom = a1x * a2y - a2x * a1y
            if denom:
                return (S(b1 * a2y - b2 * a1y) / denom,
                        S(b2 * a1x - b1 * a2x) / denom)


def is_vertex(ent):
    """If the input entity is a vertex return True.

    Parameter
    =========

    ent :
        Denotes a geometric entity representing a point.

    Examples
    ========

    >>> from sympy import Point
    >>> from sympy.integrals.intpoly import is_vertex
    >>> is_vertex((2, 3))
    True
    >>> is_vertex((2, 3, 6))
    True
    >>> is_vertex(Point(2, 3))
    True
    """
    if isinstance(ent, tuple):
        if len(ent) in [2, 3]:
            return True
    elif isinstance(ent, Point):
        return True
    return False


def plot_polytope(poly):
    """Plots the 2D polytope using the functions written in plotting
    module which in turn uses matplotlib backend.

    Parameter
    =========

    poly:
        Denotes a 2-Polytope.
    """
    from sympy.plotting.plot import Plot, List2DSeries

    xl = [vertex.x for vertex in poly.vertices]
    yl = [vertex.y for vertex in poly.vertices]

    xl.append(poly.vertices[0].x)  # Closing the polygon
    yl.append(poly.vertices[0].y)

    l2ds = List2DSeries(xl, yl)
    p = Plot(l2ds, axes='label_axes=True')
    p.show()


def plot_polynomial(expr):
    """Plots the polynomial using the functions written in
    plotting module which in turn uses matplotlib backend.

    Parameter
    =========

    expr:
        Denotes a polynomial(SymPy expression).
    """
    from sympy.plotting.plot import plot3d, plot
    gens = expr.free_symbols
    if len(gens) == 2:
        plot3d(expr)
    else:
        plot(expr)
