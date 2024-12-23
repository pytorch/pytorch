from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r
from sympy.diffgeom import intcurve_series, Differential, WedgeProduct
from sympy.core import symbols, Function, Derivative
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin, cos
from sympy.matrices import Matrix

# Most of the functionality is covered in the
# test_functional_diffgeom_ch* tests which are based on the
# example from the paper of Sussman and Wisdom.
# If they do not cover something, additional tests are added in other test
# functions.

# From "Functional Differential Geometry" as of 2011
# by Sussman and Wisdom.


def test_functional_diffgeom_ch2():
    x0, y0, r0, theta0 = symbols('x0, y0, r0, theta0', real=True)
    x, y = symbols('x, y', real=True)
    f = Function('f')

    assert (R2_p.point_to_coords(R2_r.point([x0, y0])) ==
           Matrix([sqrt(x0**2 + y0**2), atan2(y0, x0)]))
    assert (R2_r.point_to_coords(R2_p.point([r0, theta0])) ==
           Matrix([r0*cos(theta0), r0*sin(theta0)]))

    assert R2_p.jacobian(R2_r, [r0, theta0]) == Matrix(
        [[cos(theta0), -r0*sin(theta0)], [sin(theta0), r0*cos(theta0)]])

    field = f(R2.x, R2.y)
    p1_in_rect = R2_r.point([x0, y0])
    p1_in_polar = R2_p.point([sqrt(x0**2 + y0**2), atan2(y0, x0)])
    assert field.rcall(p1_in_rect) == f(x0, y0)
    assert field.rcall(p1_in_polar) == f(x0, y0)

    p_r = R2_r.point([x0, y0])
    p_p = R2_p.point([r0, theta0])
    assert R2.x(p_r) == x0
    assert R2.x(p_p) == r0*cos(theta0)
    assert R2.r(p_p) == r0
    assert R2.r(p_r) == sqrt(x0**2 + y0**2)
    assert R2.theta(p_r) == atan2(y0, x0)

    h = R2.x*R2.r**2 + R2.y**3
    assert h.rcall(p_r) == x0*(x0**2 + y0**2) + y0**3
    assert h.rcall(p_p) == r0**3*sin(theta0)**3 + r0**3*cos(theta0)


def test_functional_diffgeom_ch3():
    x0, y0 = symbols('x0, y0', real=True)
    x, y, t = symbols('x, y, t', real=True)
    f = Function('f')
    b1 = Function('b1')
    b2 = Function('b2')
    p_r = R2_r.point([x0, y0])

    s_field = f(R2.x, R2.y)
    v_field = b1(R2.x)*R2.e_x + b2(R2.y)*R2.e_y
    assert v_field.rcall(s_field).rcall(p_r).doit() == b1(
        x0)*Derivative(f(x0, y0), x0) + b2(y0)*Derivative(f(x0, y0), y0)

    assert R2.e_x(R2.r**2).rcall(p_r) == 2*x0
    v = R2.e_x + 2*R2.e_y
    s = R2.r**2 + 3*R2.x
    assert v.rcall(s).rcall(p_r).doit() == 2*x0 + 4*y0 + 3

    circ = -R2.y*R2.e_x + R2.x*R2.e_y
    series = intcurve_series(circ, t, R2_r.point([1, 0]), coeffs=True)
    series_x, series_y = zip(*series)
    assert all(
        term == cos(t).taylor_term(i, t) for i, term in enumerate(series_x))
    assert all(
        term == sin(t).taylor_term(i, t) for i, term in enumerate(series_y))


def test_functional_diffgeom_ch4():
    x0, y0, theta0 = symbols('x0, y0, theta0', real=True)
    x, y, r, theta = symbols('x, y, r, theta', real=True)
    r0 = symbols('r0', positive=True)
    f = Function('f')
    b1 = Function('b1')
    b2 = Function('b2')
    p_r = R2_r.point([x0, y0])
    p_p = R2_p.point([r0, theta0])

    f_field = b1(R2.x, R2.y)*R2.dx + b2(R2.x, R2.y)*R2.dy
    assert f_field.rcall(R2.e_x).rcall(p_r) == b1(x0, y0)
    assert f_field.rcall(R2.e_y).rcall(p_r) == b2(x0, y0)

    s_field_r = f(R2.x, R2.y)
    df = Differential(s_field_r)
    assert df(R2.e_x).rcall(p_r).doit() == Derivative(f(x0, y0), x0)
    assert df(R2.e_y).rcall(p_r).doit() == Derivative(f(x0, y0), y0)

    s_field_p = f(R2.r, R2.theta)
    df = Differential(s_field_p)
    assert trigsimp(df(R2.e_x).rcall(p_p).doit()) == (
        cos(theta0)*Derivative(f(r0, theta0), r0) -
        sin(theta0)*Derivative(f(r0, theta0), theta0)/r0)
    assert trigsimp(df(R2.e_y).rcall(p_p).doit()) == (
        sin(theta0)*Derivative(f(r0, theta0), r0) +
        cos(theta0)*Derivative(f(r0, theta0), theta0)/r0)

    assert R2.dx(R2.e_x).rcall(p_r) == 1
    assert R2.dx(R2.e_x) == 1
    assert R2.dx(R2.e_y).rcall(p_r) == 0
    assert R2.dx(R2.e_y) == 0

    circ = -R2.y*R2.e_x + R2.x*R2.e_y
    assert R2.dx(circ).rcall(p_r).doit() == -y0
    assert R2.dy(circ).rcall(p_r) == x0
    assert R2.dr(circ).rcall(p_r) == 0
    assert simplify(R2.dtheta(circ).rcall(p_r)) == 1

    assert (circ - R2.e_theta).rcall(s_field_r).rcall(p_r) == 0


def test_functional_diffgeom_ch6():
    u0, u1, u2, v0, v1, v2, w0, w1, w2 = symbols('u0:3, v0:3, w0:3', real=True)

    u = u0*R2.e_x + u1*R2.e_y
    v = v0*R2.e_x + v1*R2.e_y
    wp = WedgeProduct(R2.dx, R2.dy)
    assert wp(u, v) == u0*v1 - u1*v0

    u = u0*R3_r.e_x + u1*R3_r.e_y + u2*R3_r.e_z
    v = v0*R3_r.e_x + v1*R3_r.e_y + v2*R3_r.e_z
    w = w0*R3_r.e_x + w1*R3_r.e_y + w2*R3_r.e_z
    wp = WedgeProduct(R3_r.dx, R3_r.dy, R3_r.dz)
    assert wp(
        u, v, w) == Matrix(3, 3, [u0, u1, u2, v0, v1, v2, w0, w1, w2]).det()

    a, b, c = symbols('a, b, c', cls=Function)
    a_f = a(R3_r.x, R3_r.y, R3_r.z)
    b_f = b(R3_r.x, R3_r.y, R3_r.z)
    c_f = c(R3_r.x, R3_r.y, R3_r.z)
    theta = a_f*R3_r.dx + b_f*R3_r.dy + c_f*R3_r.dz
    dtheta = Differential(theta)
    da = Differential(a_f)
    db = Differential(b_f)
    dc = Differential(c_f)
    expr = dtheta - WedgeProduct(
        da, R3_r.dx) - WedgeProduct(db, R3_r.dy) - WedgeProduct(dc, R3_r.dz)
    assert expr.rcall(R3_r.e_x, R3_r.e_y) == 0
