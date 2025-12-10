from sympy.core import Lambda, Symbol, symbols
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r, R3_c, R3_s, R2_origin
from sympy.diffgeom import (Manifold, Patch, CoordSystem, Commutator, Differential, TensorProduct,
        WedgeProduct, BaseCovarDerivativeOp, CovarDerivativeOp, LieDerivative,
        covariant_order, contravariant_order, twoform_to_matrix, metric_to_Christoffel_1st,
        metric_to_Christoffel_2nd, metric_to_Riemann_components,
        metric_to_Ricci_components, intcurve_diffequ, intcurve_series)
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin
from sympy.matrices import Matrix
from sympy.testing.pytest import raises, nocache_fail
from sympy.testing.pytest import warns_deprecated_sympy

TP = TensorProduct


def test_coordsys_transform():
    # test inverse transforms
    p, q, r, s = symbols('p q r s')
    rel = {('first', 'second'): [(p, q), (q, -p)]}
    R2_pq = CoordSystem('first', R2_origin, [p, q], rel)
    R2_rs = CoordSystem('second', R2_origin, [r, s], rel)
    r, s = R2_rs.symbols
    assert R2_rs.transform(R2_pq) == Matrix([[-s], [r]])

    # inverse transform impossible case
    a, b = symbols('a b', positive=True)
    rel = {('first', 'second'): [(a,), (-a,)]}
    R2_a = CoordSystem('first', R2_origin, [a], rel)
    R2_b = CoordSystem('second', R2_origin, [b], rel)
    # This transformation is uninvertible because there is no positive a, b satisfying a = -b
    with raises(NotImplementedError):
        R2_b.transform(R2_a)

    # inverse transform ambiguous case
    c, d = symbols('c d')
    rel = {('first', 'second'): [(c,), (c**2,)]}
    R2_c = CoordSystem('first', R2_origin, [c], rel)
    R2_d = CoordSystem('second', R2_origin, [d], rel)
    # The transform method should throw if it finds multiple inverses for a coordinate transformation.
    with raises(ValueError):
        R2_d.transform(R2_c)

    # test indirect transformation
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): [(a, b), (2*a, 3*b)],
        ('C2', 'C3'): [(c, d), (3*c, 2*d)]}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c/2, d/3])
    assert C1.transform(C3) == Matrix([6*a, 6*b])
    assert C3.transform(C1) == Matrix([e/6, f/6])
    assert C3.transform(C2) == Matrix([e/3, f/2])

    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): [(a, b), (2*a, 3*b + 1)],
        ('C3', 'C2'): [(e, f), (-e - 2, 2*f)]}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c/2, (d - 1)/3])
    assert C1.transform(C3) == Matrix([-2*a - 2, (3*b + 1)/2])
    assert C3.transform(C1) == Matrix([-e/2 - 1, (2*f - 1)/3])
    assert C3.transform(C2) == Matrix([-e - 2, 2*f])

    # old signature uses Lambda
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): Lambda((a, b), (2*a, 3*b + 1)),
        ('C3', 'C2'): Lambda((e, f), (-e - 2, 2*f))}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c/2, (d - 1)/3])
    assert C1.transform(C3) == Matrix([-2*a - 2, (3*b + 1)/2])
    assert C3.transform(C1) == Matrix([-e/2 - 1, (2*f - 1)/3])
    assert C3.transform(C2) == Matrix([-e - 2, 2*f])


def test_R2():
    x0, y0, r0, theta0 = symbols('x0, y0, r0, theta0', real=True)
    point_r = R2_r.point([x0, y0])
    point_p = R2_p.point([r0, theta0])

    # r**2 = x**2 + y**2
    assert (R2.r**2 - R2.x**2 - R2.y**2).rcall(point_r) == 0
    assert trigsimp( (R2.r**2 - R2.x**2 - R2.y**2).rcall(point_p) ) == 0
    assert trigsimp(R2.e_r(R2.x**2 + R2.y**2).rcall(point_p).doit()) == 2*r0

    # polar->rect->polar == Id
    a, b = symbols('a b', positive=True)
    m = Matrix([[a], [b]])

    #TODO assert m == R2_r.transform(R2_p, R2_p.transform(R2_r, [a, b])).applyfunc(simplify)
    assert m == R2_p.transform(R2_r, R2_r.transform(R2_p, m)).applyfunc(simplify)

    # deprecated method
    with warns_deprecated_sympy():
        assert m == R2_p.coord_tuple_transform_to(
            R2_r, R2_r.coord_tuple_transform_to(R2_p, m)).applyfunc(simplify)


def test_R3():
    a, b, c = symbols('a b c', positive=True)
    m = Matrix([[a], [b], [c]])

    assert m == R3_c.transform(R3_r, R3_r.transform(R3_c, m)).applyfunc(simplify)
    #TODO assert m == R3_r.transform(R3_c, R3_c.transform(R3_r, m)).applyfunc(simplify)
    assert m == R3_s.transform(
        R3_r, R3_r.transform(R3_s, m)).applyfunc(simplify)
    #TODO assert m == R3_r.transform(R3_s, R3_s.transform(R3_r, m)).applyfunc(simplify)
    assert m == R3_s.transform(
        R3_c, R3_c.transform(R3_s, m)).applyfunc(simplify)
    #TODO assert m == R3_c.transform(R3_s, R3_s.transform(R3_c, m)).applyfunc(simplify)

    with warns_deprecated_sympy():
        assert m == R3_c.coord_tuple_transform_to(
            R3_r, R3_r.coord_tuple_transform_to(R3_c, m)).applyfunc(simplify)
        #TODO assert m == R3_r.coord_tuple_transform_to(R3_c, R3_c.coord_tuple_transform_to(R3_r, m)).applyfunc(simplify)
        assert m == R3_s.coord_tuple_transform_to(
            R3_r, R3_r.coord_tuple_transform_to(R3_s, m)).applyfunc(simplify)
        #TODO assert m == R3_r.coord_tuple_transform_to(R3_s, R3_s.coord_tuple_transform_to(R3_r, m)).applyfunc(simplify)
        assert m == R3_s.coord_tuple_transform_to(
            R3_c, R3_c.coord_tuple_transform_to(R3_s, m)).applyfunc(simplify)
        #TODO assert m == R3_c.coord_tuple_transform_to(R3_s, R3_s.coord_tuple_transform_to(R3_c, m)).applyfunc(simplify)


def test_CoordinateSymbol():
    x, y = R2_r.symbols
    r, theta = R2_p.symbols
    assert y.rewrite(R2_p) == r*sin(theta)


def test_point():
    x, y = symbols('x, y')
    p = R2_r.point([x, y])
    assert p.free_symbols == {x, y}
    assert p.coords(R2_r) == p.coords() == Matrix([x, y])
    assert p.coords(R2_p) == Matrix([sqrt(x**2 + y**2), atan2(y, x)])


def test_commutator():
    assert Commutator(R2.e_x, R2.e_y) == 0
    assert Commutator(R2.x*R2.e_x, R2.x*R2.e_x) == 0
    assert Commutator(R2.x*R2.e_x, R2.x*R2.e_y) == R2.x*R2.e_y
    c = Commutator(R2.e_x, R2.e_r)
    assert c(R2.x) == R2.y*(R2.x**2 + R2.y**2)**(-1)*sin(R2.theta)


def test_differential():
    xdy = R2.x*R2.dy
    dxdy = Differential(xdy)
    assert xdy.rcall(None) == xdy
    assert dxdy(R2.e_x, R2.e_y) == 1
    assert dxdy(R2.e_x, R2.x*R2.e_y) == R2.x
    assert Differential(dxdy) == 0


def test_products():
    assert TensorProduct(
        R2.dx, R2.dy)(R2.e_x, R2.e_y) == R2.dx(R2.e_x)*R2.dy(R2.e_y) == 1
    assert TensorProduct(R2.dx, R2.dy)(None, R2.e_y) == R2.dx
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x, None) == R2.dy
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x) == R2.dy
    assert TensorProduct(R2.x, R2.dx) == R2.x*R2.dx
    assert TensorProduct(
        R2.e_x, R2.e_y)(R2.x, R2.y) == R2.e_x(R2.x) * R2.e_y(R2.y) == 1
    assert TensorProduct(R2.e_x, R2.e_y)(None, R2.y) == R2.e_x
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x, None) == R2.e_y
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x) == R2.e_y
    assert TensorProduct(R2.x, R2.e_x) == R2.x * R2.e_x
    assert TensorProduct(
        R2.dx, R2.e_y)(R2.e_x, R2.y) == R2.dx(R2.e_x) * R2.e_y(R2.y) == 1
    assert TensorProduct(R2.dx, R2.e_y)(None, R2.y) == R2.dx
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x, None) == R2.e_y
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x) == R2.e_y
    assert TensorProduct(R2.x, R2.e_x) == R2.x * R2.e_x
    assert TensorProduct(
        R2.e_x, R2.dy)(R2.x, R2.e_y) == R2.e_x(R2.x) * R2.dy(R2.e_y) == 1
    assert TensorProduct(R2.e_x, R2.dy)(None, R2.e_y) == R2.e_x
    assert TensorProduct(R2.e_x, R2.dy)(R2.x, None) == R2.dy
    assert TensorProduct(R2.e_x, R2.dy)(R2.x) == R2.dy
    assert TensorProduct(R2.e_y,R2.e_x)(R2.x**2 + R2.y**2,R2.x**2 + R2.y**2) == 4*R2.x*R2.y

    assert WedgeProduct(R2.dx, R2.dy)(R2.e_x, R2.e_y) == 1
    assert WedgeProduct(R2.e_x, R2.e_y)(R2.x, R2.y) == 1


def test_lie_derivative():
    assert LieDerivative(R2.e_x, R2.y) == R2.e_x(R2.y) == 0
    assert LieDerivative(R2.e_x, R2.x) == R2.e_x(R2.x) == 1
    assert LieDerivative(R2.e_x, R2.e_x) == Commutator(R2.e_x, R2.e_x) == 0
    assert LieDerivative(R2.e_x, R2.e_r) == Commutator(R2.e_x, R2.e_r)
    assert LieDerivative(R2.e_x + R2.e_y, R2.x) == 1
    assert LieDerivative(
        R2.e_x, TensorProduct(R2.dx, R2.dy))(R2.e_x, R2.e_y) == 0


@nocache_fail
def test_covar_deriv():
    ch = metric_to_Christoffel_2nd(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    cvd = BaseCovarDerivativeOp(R2_r, 0, ch)
    assert cvd(R2.x) == 1
    # This line fails if the cache is disabled:
    assert cvd(R2.x*R2.e_x) == R2.e_x
    cvd = CovarDerivativeOp(R2.x*R2.e_x, ch)
    assert cvd(R2.x) == R2.x
    assert cvd(R2.x*R2.e_x) == R2.x*R2.e_x


def test_intcurve_diffequ():
    t = symbols('t')
    start_point = R2_r.point([1, 0])
    vector_field = -R2.y*R2.e_x + R2.x*R2.e_y
    equations, init_cond = intcurve_diffequ(vector_field, t, start_point)
    assert str(equations) == '[f_1(t) + Derivative(f_0(t), t), -f_0(t) + Derivative(f_1(t), t)]'
    assert str(init_cond) == '[f_0(0) - 1, f_1(0)]'
    equations, init_cond = intcurve_diffequ(vector_field, t, start_point, R2_p)
    assert str(
        equations) == '[Derivative(f_0(t), t), Derivative(f_1(t), t) - 1]'
    assert str(init_cond) == '[f_0(0) - 1, f_1(0)]'


def test_helpers_and_coordinate_dependent():
    one_form = R2.dr + R2.dx
    two_form = Differential(R2.x*R2.dr + R2.r*R2.dx)
    three_form = Differential(
        R2.y*two_form) + Differential(R2.x*Differential(R2.r*R2.dr))
    metric = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dy, R2.dy)
    metric_ambig = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dr, R2.dr)
    misform_a = TensorProduct(R2.dr, R2.dr) + R2.dr
    misform_b = R2.dr**4
    misform_c = R2.dx*R2.dy
    twoform_not_sym = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dx, R2.dy)
    twoform_not_TP = WedgeProduct(R2.dx, R2.dy)

    one_vector = R2.e_x + R2.e_y
    two_vector = TensorProduct(R2.e_x, R2.e_y)
    three_vector = TensorProduct(R2.e_x, R2.e_y, R2.e_x)
    two_wp = WedgeProduct(R2.e_x,R2.e_y)

    assert covariant_order(one_form) == 1
    assert covariant_order(two_form) == 2
    assert covariant_order(three_form) == 3
    assert covariant_order(two_form + metric) == 2
    assert covariant_order(two_form + metric_ambig) == 2
    assert covariant_order(two_form + twoform_not_sym) == 2
    assert covariant_order(two_form + twoform_not_TP) == 2

    assert contravariant_order(one_vector) == 1
    assert contravariant_order(two_vector) == 2
    assert contravariant_order(three_vector) == 3
    assert contravariant_order(two_vector + two_wp) == 2

    raises(ValueError, lambda: covariant_order(misform_a))
    raises(ValueError, lambda: covariant_order(misform_b))
    raises(ValueError, lambda: covariant_order(misform_c))

    assert twoform_to_matrix(metric) == Matrix([[1, 0], [0, 1]])
    assert twoform_to_matrix(twoform_not_sym) == Matrix([[1, 0], [1, 0]])
    assert twoform_to_matrix(twoform_not_TP) == Matrix([[0, -1], [1, 0]])

    raises(ValueError, lambda: twoform_to_matrix(one_form))
    raises(ValueError, lambda: twoform_to_matrix(three_form))
    raises(ValueError, lambda: twoform_to_matrix(metric_ambig))

    raises(ValueError, lambda: metric_to_Christoffel_1st(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Christoffel_2nd(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Riemann_components(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Ricci_components(twoform_not_sym))


def test_correct_arguments():
    raises(ValueError, lambda: R2.e_x(R2.e_x))
    raises(ValueError, lambda: R2.e_x(R2.dx))

    raises(ValueError, lambda: Commutator(R2.e_x, R2.x))
    raises(ValueError, lambda: Commutator(R2.dx, R2.e_x))

    raises(ValueError, lambda: Differential(Differential(R2.e_x)))

    raises(ValueError, lambda: R2.dx(R2.x))

    raises(ValueError, lambda: LieDerivative(R2.dx, R2.dx))
    raises(ValueError, lambda: LieDerivative(R2.x, R2.dx))

    raises(ValueError, lambda: CovarDerivativeOp(R2.dx, []))
    raises(ValueError, lambda: CovarDerivativeOp(R2.x, []))

    a = Symbol('a')
    raises(ValueError, lambda: intcurve_series(R2.dx, a, R2_r.point([1, 2])))
    raises(ValueError, lambda: intcurve_series(R2.x, a, R2_r.point([1, 2])))

    raises(ValueError, lambda: intcurve_diffequ(R2.dx, a, R2_r.point([1, 2])))
    raises(ValueError, lambda: intcurve_diffequ(R2.x, a, R2_r.point([1, 2])))

    raises(ValueError, lambda: contravariant_order(R2.e_x + R2.dx))
    raises(ValueError, lambda: covariant_order(R2.e_x + R2.dx))

    raises(ValueError, lambda: contravariant_order(R2.e_x*R2.e_y))
    raises(ValueError, lambda: covariant_order(R2.dx*R2.dy))

def test_simplify():
    x, y = R2_r.coord_functions()
    dx, dy = R2_r.base_oneforms()
    ex, ey = R2_r.base_vectors()
    assert simplify(x) == x
    assert simplify(x*y) == x*y
    assert simplify(dx*dy) == dx*dy
    assert simplify(ex*ey) == ex*ey
    assert ((1-x)*dx)/(1-x)**2 == dx/(1-x)


def test_issue_17917():
    X = R2.x*R2.e_x - R2.y*R2.e_y
    Y = (R2.x**2 + R2.y**2)*R2.e_x - R2.x*R2.y*R2.e_y
    assert LieDerivative(X, Y).expand() == (
        R2.x**2*R2.e_x - 3*R2.y**2*R2.e_x - R2.x*R2.y*R2.e_y)

def test_deprecations():
    m = Manifold('M', 2)
    p = Patch('P', m)
    with warns_deprecated_sympy():
        CoordSystem('Car2d', p, names=['x', 'y'])

    with warns_deprecated_sympy():
        c = CoordSystem('Car2d', p, ['x', 'y'])

    with warns_deprecated_sympy():
        list(m.patches)

    with warns_deprecated_sympy():
        list(c.transforms)
