from sympy.core.function import Derivative
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.simplify import simplify
from sympy.core.symbol import symbols
from sympy.core import S
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.vector import Dot
from sympy.vector.operators import curl, divergence, gradient, Gradient, Divergence, Cross
from sympy.vector.deloperator import Del
from sympy.vector.functions import (is_conservative, is_solenoidal,
                                    scalar_potential, directional_derivative,
                                    laplacian, scalar_potential_difference)
from sympy.testing.pytest import raises

C = CoordSys3D('C')
i, j, k = C.base_vectors()
x, y, z = C.base_scalars()
delop = Del()
a, b, c, q = symbols('a b c q')


def test_del_operator():
    # Tests for curl

    assert delop ^ Vector.zero == Vector.zero
    assert ((delop ^ Vector.zero).doit() == Vector.zero ==
            curl(Vector.zero))
    assert delop.cross(Vector.zero) == delop ^ Vector.zero
    assert (delop ^ i).doit() == Vector.zero
    assert delop.cross(2*y**2*j, doit=True) == Vector.zero
    assert delop.cross(2*y**2*j) == delop ^ 2*y**2*j
    v = x*y*z * (i + j + k)
    assert ((delop ^ v).doit() ==
            (-x*y + x*z)*i + (x*y - y*z)*j + (-x*z + y*z)*k ==
            curl(v))
    assert delop ^ v == delop.cross(v)
    assert (delop.cross(2*x**2*j) ==
            (Derivative(0, C.y) - Derivative(2*C.x**2, C.z))*C.i +
            (-Derivative(0, C.x) + Derivative(0, C.z))*C.j +
            (-Derivative(0, C.y) + Derivative(2*C.x**2, C.x))*C.k)
    assert (delop.cross(2*x**2*j, doit=True) == 4*x*k ==
            curl(2*x**2*j))

    #Tests for divergence
    assert delop & Vector.zero is S.Zero == divergence(Vector.zero)
    assert (delop & Vector.zero).doit() is S.Zero
    assert delop.dot(Vector.zero) == delop & Vector.zero
    assert (delop & i).doit() is S.Zero
    assert (delop & x**2*i).doit() == 2*x == divergence(x**2*i)
    assert (delop.dot(v, doit=True) == x*y + y*z + z*x ==
            divergence(v))
    assert delop & v == delop.dot(v)
    assert delop.dot(1/(x*y*z) * (i + j + k), doit=True) == \
           - 1 / (x*y*z**2) - 1 / (x*y**2*z) - 1 / (x**2*y*z)
    v = x*i + y*j + z*k
    assert (delop & v == Derivative(C.x, C.x) +
            Derivative(C.y, C.y) + Derivative(C.z, C.z))
    assert delop.dot(v, doit=True) == 3 == divergence(v)
    assert delop & v == delop.dot(v)
    assert simplify((delop & v).doit()) == 3

    #Tests for gradient
    assert (delop.gradient(0, doit=True) == Vector.zero ==
            gradient(0))
    assert delop.gradient(0) == delop(0)
    assert (delop(S.Zero)).doit() == Vector.zero
    assert (delop(x) == (Derivative(C.x, C.x))*C.i +
            (Derivative(C.x, C.y))*C.j + (Derivative(C.x, C.z))*C.k)
    assert (delop(x)).doit() == i == gradient(x)
    assert (delop(x*y*z) ==
            (Derivative(C.x*C.y*C.z, C.x))*C.i +
            (Derivative(C.x*C.y*C.z, C.y))*C.j +
            (Derivative(C.x*C.y*C.z, C.z))*C.k)
    assert (delop.gradient(x*y*z, doit=True) ==
            y*z*i + z*x*j + x*y*k ==
            gradient(x*y*z))
    assert delop(x*y*z) == delop.gradient(x*y*z)
    assert (delop(2*x**2)).doit() == 4*x*i
    assert ((delop(a*sin(y) / x)).doit() ==
            -a*sin(y)/x**2 * i + a*cos(y)/x * j)

    #Tests for directional derivative
    assert (Vector.zero & delop)(a) is S.Zero
    assert ((Vector.zero & delop)(a)).doit() is S.Zero
    assert ((v & delop)(Vector.zero)).doit() == Vector.zero
    assert ((v & delop)(S.Zero)).doit() is S.Zero
    assert ((i & delop)(x)).doit() == 1
    assert ((j & delop)(y)).doit() == 1
    assert ((k & delop)(z)).doit() == 1
    assert ((i & delop)(x*y*z)).doit() == y*z
    assert ((v & delop)(x)).doit() == x
    assert ((v & delop)(x*y*z)).doit() == 3*x*y*z
    assert (v & delop)(x + y + z) == C.x + C.y + C.z
    assert ((v & delop)(x + y + z)).doit() == x + y + z
    assert ((v & delop)(v)).doit() == v
    assert ((i & delop)(v)).doit() == i
    assert ((j & delop)(v)).doit() == j
    assert ((k & delop)(v)).doit() == k
    assert ((v & delop)(Vector.zero)).doit() == Vector.zero

    # Tests for laplacian on scalar fields
    assert laplacian(x*y*z) is S.Zero
    assert laplacian(x**2) == S(2)
    assert laplacian(x**2*y**2*z**2) == \
                    2*y**2*z**2 + 2*x**2*z**2 + 2*x**2*y**2
    A = CoordSys3D('A', transformation="spherical", variable_names=["r", "theta", "phi"])
    B = CoordSys3D('B', transformation='cylindrical', variable_names=["r", "theta", "z"])
    assert laplacian(A.r + A.theta + A.phi) == 2/A.r + cos(A.theta)/(A.r**2*sin(A.theta))
    assert laplacian(B.r + B.theta + B.z) == 1/B.r

    # Tests for laplacian on vector fields
    assert laplacian(x*y*z*(i + j + k)) == Vector.zero
    assert laplacian(x*y**2*z*(i + j + k)) == \
                            2*x*z*i + 2*x*z*j + 2*x*z*k


def test_product_rules():
    """
    Tests the six product rules defined with respect to the Del
    operator

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Del

    """

    #Define the scalar and vector functions
    f = 2*x*y*z
    g = x*y + y*z + z*x
    u = x**2*i + 4*j - y**2*z*k
    v = 4*i + x*y*z*k

    # First product rule
    lhs = delop(f * g, doit=True)
    rhs = (f * delop(g) + g * delop(f)).doit()
    assert simplify(lhs) == simplify(rhs)

    # Second product rule
    lhs = delop(u & v).doit()
    rhs = ((u ^ (delop ^ v)) + (v ^ (delop ^ u)) + \
          ((u & delop)(v)) + ((v & delop)(u))).doit()
    assert simplify(lhs) == simplify(rhs)

    # Third product rule
    lhs = (delop & (f*v)).doit()
    rhs = ((f * (delop & v)) + (v & (delop(f)))).doit()
    assert simplify(lhs) == simplify(rhs)

    # Fourth product rule
    lhs = (delop & (u ^ v)).doit()
    rhs = ((v & (delop ^ u)) - (u & (delop ^ v))).doit()
    assert simplify(lhs) == simplify(rhs)

    # Fifth product rule
    lhs = (delop ^ (f * v)).doit()
    rhs = (((delop(f)) ^ v) + (f * (delop ^ v))).doit()
    assert simplify(lhs) == simplify(rhs)

    # Sixth product rule
    lhs = (delop ^ (u ^ v)).doit()
    rhs = (u * (delop & v) - v * (delop & u) +
           (v & delop)(u) - (u & delop)(v)).doit()
    assert simplify(lhs) == simplify(rhs)


P = C.orient_new_axis('P', q, C.k)  # type: ignore
scalar_field = 2*x**2*y*z
grad_field = gradient(scalar_field)
vector_field = y**2*i + 3*x*j + 5*y*z*k
curl_field = curl(vector_field)


def test_conservative():
    assert is_conservative(Vector.zero) is True
    assert is_conservative(i) is True
    assert is_conservative(2 * i + 3 * j + 4 * k) is True
    assert (is_conservative(y*z*i + x*z*j + x*y*k) is
            True)
    assert is_conservative(x * j) is False
    assert is_conservative(grad_field) is True
    assert is_conservative(curl_field) is False
    assert (is_conservative(4*x*y*z*i + 2*x**2*z*j) is
            False)
    assert is_conservative(z*P.i + P.x*k) is True


def test_solenoidal():
    assert is_solenoidal(Vector.zero) is True
    assert is_solenoidal(i) is True
    assert is_solenoidal(2 * i + 3 * j + 4 * k) is True
    assert (is_solenoidal(y*z*i + x*z*j + x*y*k) is
            True)
    assert is_solenoidal(y * j) is False
    assert is_solenoidal(grad_field) is False
    assert is_solenoidal(curl_field) is True
    assert is_solenoidal((-2*y + 3)*k) is True
    assert is_solenoidal(cos(q)*i + sin(q)*j + cos(q)*P.k) is True
    assert is_solenoidal(z*P.i + P.x*k) is True


def test_directional_derivative():
    assert directional_derivative(C.x*C.y*C.z, 3*C.i + 4*C.j + C.k) == C.x*C.y + 4*C.x*C.z + 3*C.y*C.z
    assert directional_derivative(5*C.x**2*C.z, 3*C.i + 4*C.j + C.k) == 5*C.x**2 + 30*C.x*C.z
    assert directional_derivative(5*C.x**2*C.z, 4*C.j) is S.Zero

    D = CoordSys3D("D", "spherical", variable_names=["r", "theta", "phi"],
                   vector_names=["e_r", "e_theta", "e_phi"])
    r, theta, phi = D.base_scalars()
    e_r, e_theta, e_phi = D.base_vectors()
    assert directional_derivative(r**2*e_r, e_r) == 2*r*e_r
    assert directional_derivative(5*r**2*phi, 3*e_r + 4*e_theta + e_phi) == 5*r**2 + 30*r*phi


def test_scalar_potential():
    assert scalar_potential(Vector.zero, C) == 0
    assert scalar_potential(i, C) == x
    assert scalar_potential(j, C) == y
    assert scalar_potential(k, C) == z
    assert scalar_potential(y*z*i + x*z*j + x*y*k, C) == x*y*z
    assert scalar_potential(grad_field, C) == scalar_field
    assert scalar_potential(z*P.i + P.x*k, C) == x*z*cos(q) + y*z*sin(q)
    assert scalar_potential(z*P.i + P.x*k, P) == P.x*P.z
    raises(ValueError, lambda: scalar_potential(x*j, C))


def test_scalar_potential_difference():
    point1 = C.origin.locate_new('P1', 1*i + 2*j + 3*k)
    point2 = C.origin.locate_new('P2', 4*i + 5*j + 6*k)
    genericpointC = C.origin.locate_new('RP', x*i + y*j + z*k)
    genericpointP = P.origin.locate_new('PP', P.x*P.i + P.y*P.j + P.z*P.k)
    assert scalar_potential_difference(S.Zero, C, point1, point2) == 0
    assert (scalar_potential_difference(scalar_field, C, C.origin,
                                        genericpointC) ==
            scalar_field)
    assert (scalar_potential_difference(grad_field, C, C.origin,
                                        genericpointC) ==
            scalar_field)
    assert scalar_potential_difference(grad_field, C, point1, point2) == 948
    assert (scalar_potential_difference(y*z*i + x*z*j +
                                        x*y*k, C, point1,
                                        genericpointC) ==
            x*y*z - 6)
    potential_diff_P = (2*P.z*(P.x*sin(q) + P.y*cos(q))*
                        (P.x*cos(q) - P.y*sin(q))**2)
    assert (scalar_potential_difference(grad_field, P, P.origin,
                                        genericpointP).simplify() ==
            potential_diff_P.simplify())


def test_differential_operators_curvilinear_system():
    A = CoordSys3D('A', transformation="spherical", variable_names=["r", "theta", "phi"])
    B = CoordSys3D('B', transformation='cylindrical', variable_names=["r", "theta", "z"])
    # Test for spherical coordinate system and gradient
    assert gradient(3*A.r + 4*A.theta) == 3*A.i + 4/A.r*A.j
    assert gradient(3*A.r*A.phi + 4*A.theta) == 3*A.phi*A.i + 4/A.r*A.j + (3/sin(A.theta))*A.k
    assert gradient(0*A.r + 0*A.theta+0*A.phi) == Vector.zero
    assert gradient(A.r*A.theta*A.phi) == A.theta*A.phi*A.i + A.phi*A.j + (A.theta/sin(A.theta))*A.k
    # Test for spherical coordinate system and divergence
    assert divergence(A.r * A.i + A.theta * A.j + A.phi * A.k) == \
           (sin(A.theta)*A.r + cos(A.theta)*A.r*A.theta)/(sin(A.theta)*A.r**2) + 3 + 1/(sin(A.theta)*A.r)
    assert divergence(3*A.r*A.phi*A.i + A.theta*A.j + A.r*A.theta*A.phi*A.k) == \
           (sin(A.theta)*A.r + cos(A.theta)*A.r*A.theta)/(sin(A.theta)*A.r**2) + 9*A.phi + A.theta/sin(A.theta)
    assert divergence(Vector.zero) == 0
    assert divergence(0*A.i + 0*A.j + 0*A.k) == 0
    # Test for spherical coordinate system and curl
    assert curl(A.r*A.i + A.theta*A.j + A.phi*A.k) == \
           (cos(A.theta)*A.phi/(sin(A.theta)*A.r))*A.i + (-A.phi/A.r)*A.j + A.theta/A.r*A.k
    assert curl(A.r*A.j + A.phi*A.k) == (cos(A.theta)*A.phi/(sin(A.theta)*A.r))*A.i + (-A.phi/A.r)*A.j + 2*A.k

    # Test for cylindrical coordinate system and gradient
    assert gradient(0*B.r + 0*B.theta+0*B.z) == Vector.zero
    assert gradient(B.r*B.theta*B.z) == B.theta*B.z*B.i + B.z*B.j + B.r*B.theta*B.k
    assert gradient(3*B.r) == 3*B.i
    assert gradient(2*B.theta) == 2/B.r * B.j
    assert gradient(4*B.z) == 4*B.k
    # Test for cylindrical coordinate system and divergence
    assert divergence(B.r*B.i + B.theta*B.j + B.z*B.k) == 3 + 1/B.r
    assert divergence(B.r*B.j + B.z*B.k) == 1
    # Test for cylindrical coordinate system and curl
    assert curl(B.r*B.j + B.z*B.k) == 2*B.k
    assert curl(3*B.i + 2/B.r*B.j + 4*B.k) == Vector.zero

def test_mixed_coordinates():
    # gradient
    a = CoordSys3D('a')
    b = CoordSys3D('b')
    c = CoordSys3D('c')
    assert gradient(a.x*b.y) == b.y*a.i + a.x*b.j
    assert gradient(3*cos(q)*a.x*b.x+a.y*(a.x+(cos(q)+b.x))) ==\
           (a.y + 3*b.x*cos(q))*a.i + (a.x + b.x + cos(q))*a.j + (3*a.x*cos(q) + a.y)*b.i
    # Some tests need further work:
    # assert gradient(a.x*(cos(a.x+b.x))) == (cos(a.x + b.x))*a.i + a.x*Gradient(cos(a.x + b.x))
    # assert gradient(cos(a.x + b.x)*cos(a.x + b.z)) == Gradient(cos(a.x + b.x)*cos(a.x + b.z))
    assert gradient(a.x**b.y) == Gradient(a.x**b.y)
    # assert gradient(cos(a.x+b.y)*a.z) == None
    assert gradient(cos(a.x*b.y)) == Gradient(cos(a.x*b.y))
    assert gradient(3*cos(q)*a.x*b.x*a.z*a.y+ b.y*b.z + cos(a.x+a.y)*b.z) == \
           (3*a.y*a.z*b.x*cos(q) - b.z*sin(a.x + a.y))*a.i + \
           (3*a.x*a.z*b.x*cos(q) - b.z*sin(a.x + a.y))*a.j + (3*a.x*a.y*b.x*cos(q))*a.k + \
           (3*a.x*a.y*a.z*cos(q))*b.i + b.z*b.j + (b.y + cos(a.x + a.y))*b.k
    # divergence
    assert divergence(a.i*a.x+a.j*a.y+a.z*a.k + b.i*b.x+b.j*b.y+b.z*b.k + c.i*c.x+c.j*c.y+c.z*c.k) == S(9)
    # assert divergence(3*a.i*a.x*cos(a.x+b.z) + a.j*b.x*c.z) == None
    assert divergence(3*a.i*a.x*a.z + b.j*b.x*c.z + 3*a.j*a.z*a.y) == \
            6*a.z + b.x*Dot(b.j, c.k)
    assert divergence(3*cos(q)*a.x*b.x*b.i*c.x) == \
        3*a.x*b.x*cos(q)*Dot(b.i, c.i) + 3*a.x*c.x*cos(q) + 3*b.x*c.x*cos(q)*Dot(b.i, a.i)
    assert divergence(a.x*b.x*c.x*Cross(a.x*a.i, a.y*b.j)) ==\
           a.x*b.x*c.x*Divergence(Cross(a.x*a.i, a.y*b.j)) + \
           b.x*c.x*Dot(Cross(a.x*a.i, a.y*b.j), a.i) + \
           a.x*c.x*Dot(Cross(a.x*a.i, a.y*b.j), b.i) + \
           a.x*b.x*Dot(Cross(a.x*a.i, a.y*b.j), c.i)
    assert divergence(a.x*b.x*c.x*(a.x*a.i + b.x*b.i)) == \
                4*a.x*b.x*c.x +\
                a.x**2*c.x*Dot(a.i, b.i) +\
                a.x**2*b.x*Dot(a.i, c.i) +\
                b.x**2*c.x*Dot(b.i, a.i) +\
                a.x*b.x**2*Dot(b.i, c.i)
