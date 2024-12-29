from sympy import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
                                     RigidBody)
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
                                     kinetic_energy, linear_momentum,
                                     outer, potential_energy, msubs,
                                     find_dynamicsymbols, Lagrangian)

from sympy.physics.mechanics.functions import (
    center_of_mass, _validate_coordinates, _parse_linear_solver)
from sympy.testing.pytest import raises, warns_deprecated_sympy


q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
N = ReferenceFrame('N')
A = N.orientnew('A', 'Axis', [q1, N.z])
B = A.orientnew('B', 'Axis', [q2, A.x])
C = B.orientnew('C', 'Axis', [q3, B.y])


def test_linear_momentum():
    N = ReferenceFrame('N')
    Ac = Point('Ac')
    Ac.set_vel(N, 25 * N.y)
    I = outer(N.x, N.x)
    A = RigidBody('A', Ac, N, 20, (I, Ac))
    P = Point('P')
    Pa = Particle('Pa', P, 1)
    Pa.point.set_vel(N, 10 * N.x)
    raises(TypeError, lambda: linear_momentum(A, A, Pa))
    raises(TypeError, lambda: linear_momentum(N, N, Pa))
    assert linear_momentum(N, A, Pa) == 10 * N.x + 500 * N.y


def test_angular_momentum_and_linear_momentum():
    """A rod with length 2l, centroidal inertia I, and mass M along with a
    particle of mass m fixed to the end of the rod rotate with an angular rate
    of omega about point O which is fixed to the non-particle end of the rod.
    The rod's reference frame is A and the inertial frame is N."""
    m, M, l, I = symbols('m, M, l, I')
    omega = dynamicsymbols('omega')
    N = ReferenceFrame('N')
    a = ReferenceFrame('a')
    O = Point('O')
    Ac = O.locatenew('Ac', l * N.x)
    P = Ac.locatenew('P', l * N.x)
    O.set_vel(N, 0 * N.x)
    a.set_ang_vel(N, omega * N.z)
    Ac.v2pt_theory(O, N, a)
    P.v2pt_theory(O, N, a)
    Pa = Particle('Pa', P, m)
    A = RigidBody('A', Ac, a, M, (I * outer(N.z, N.z), Ac))
    expected = 2 * m * omega * l * N.y + M * l * omega * N.y
    assert linear_momentum(N, A, Pa) == expected
    raises(TypeError, lambda: angular_momentum(N, N, A, Pa))
    raises(TypeError, lambda: angular_momentum(O, O, A, Pa))
    raises(TypeError, lambda: angular_momentum(O, N, O, Pa))
    expected = (I + M * l**2 + 4 * m * l**2) * omega * N.z
    assert angular_momentum(O, N, A, Pa) == expected


def test_kinetic_energy():
    m, M, l1 = symbols('m M l1')
    omega = dynamicsymbols('omega')
    N = ReferenceFrame('N')
    O = Point('O')
    O.set_vel(N, 0 * N.x)
    Ac = O.locatenew('Ac', l1 * N.x)
    P = Ac.locatenew('P', l1 * N.x)
    a = ReferenceFrame('a')
    a.set_ang_vel(N, omega * N.z)
    Ac.v2pt_theory(O, N, a)
    P.v2pt_theory(O, N, a)
    Pa = Particle('Pa', P, m)
    I = outer(N.z, N.z)
    A = RigidBody('A', Ac, a, M, (I, Ac))
    raises(TypeError, lambda: kinetic_energy(Pa, Pa, A))
    raises(TypeError, lambda: kinetic_energy(N, N, A))
    assert 0 == (kinetic_energy(N, Pa, A) - (M*l1**2*omega**2/2
            + 2*l1**2*m*omega**2 + omega**2/2)).expand()


def test_potential_energy():
    m, M, l1, g, h, H = symbols('m M l1 g h H')
    omega = dynamicsymbols('omega')
    N = ReferenceFrame('N')
    O = Point('O')
    O.set_vel(N, 0 * N.x)
    Ac = O.locatenew('Ac', l1 * N.x)
    P = Ac.locatenew('P', l1 * N.x)
    a = ReferenceFrame('a')
    a.set_ang_vel(N, omega * N.z)
    Ac.v2pt_theory(O, N, a)
    P.v2pt_theory(O, N, a)
    Pa = Particle('Pa', P, m)
    I = outer(N.z, N.z)
    A = RigidBody('A', Ac, a, M, (I, Ac))
    Pa.potential_energy = m * g * h
    A.potential_energy = M * g * H
    assert potential_energy(A, Pa) == m * g * h + M * g * H


def test_Lagrangian():
    M, m, g, h = symbols('M m g h')
    N = ReferenceFrame('N')
    O = Point('O')
    O.set_vel(N, 0 * N.x)
    P = O.locatenew('P', 1 * N.x)
    P.set_vel(N, 10 * N.x)
    Pa = Particle('Pa', P, 1)
    Ac = O.locatenew('Ac', 2 * N.y)
    Ac.set_vel(N, 5 * N.y)
    a = ReferenceFrame('a')
    a.set_ang_vel(N, 10 * N.z)
    I = outer(N.z, N.z)
    A = RigidBody('A', Ac, a, 20, (I, Ac))
    Pa.potential_energy = m * g * h
    A.potential_energy = M * g * h
    raises(TypeError, lambda: Lagrangian(A, A, Pa))
    raises(TypeError, lambda: Lagrangian(N, N, Pa))


def test_msubs():
    a, b = symbols('a, b')
    x, y, z = dynamicsymbols('x, y, z')
    # Test simple substitution
    expr = Matrix([[a*x + b, x*y.diff() + y],
                   [x.diff().diff(), z + sin(z.diff())]])
    sol = Matrix([[a + b, y],
                  [x.diff().diff(), 1]])
    sd = {x: 1, z: 1, z.diff(): 0, y.diff(): 0}
    assert msubs(expr, sd) == sol
    # Test smart substitution
    expr = cos(x + y)*tan(x + y) + b*x.diff()
    sd = {x: 0, y: pi/2, x.diff(): 1}
    assert msubs(expr, sd, smart=True) == b + 1
    N = ReferenceFrame('N')
    v = x*N.x + y*N.y
    d = x*(N.x|N.x) + y*(N.y|N.y)
    v_sol = 1*N.y
    d_sol = 1*(N.y|N.y)
    sd = {x: 0, y: 1}
    assert msubs(v, sd) == v_sol
    assert msubs(d, sd) == d_sol


def test_find_dynamicsymbols():
    a, b = symbols('a, b')
    x, y, z = dynamicsymbols('x, y, z')
    expr = Matrix([[a*x + b, x*y.diff() + y],
                   [x.diff().diff(), z + sin(z.diff())]])
    # Test finding all dynamicsymbols
    sol = {x, y.diff(), y, x.diff().diff(), z, z.diff()}
    assert find_dynamicsymbols(expr) == sol
    # Test finding all but those in sym_list
    exclude_list = [x, y, z]
    sol = {y.diff(), x.diff().diff(), z.diff()}
    assert find_dynamicsymbols(expr, exclude=exclude_list) == sol
    # Test finding all dynamicsymbols in a vector with a given reference frame
    d, e, f = dynamicsymbols('d, e, f')
    A = ReferenceFrame('A')
    v = d * A.x + e * A.y + f * A.z
    sol = {d, e, f}
    assert find_dynamicsymbols(v, reference_frame=A) == sol
    # Test if a ValueError is raised on supplying only a vector as input
    raises(ValueError, lambda: find_dynamicsymbols(v))


# This function tests the center_of_mass() function
# that was added in PR #14758 to compute the center of
# mass of a system of bodies.
def test_center_of_mass():
    a = ReferenceFrame('a')
    m = symbols('m', real=True)
    p1 = Particle('p1', Point('p1_pt'), S.One)
    p2 = Particle('p2', Point('p2_pt'), S(2))
    p3 = Particle('p3', Point('p3_pt'), S(3))
    p4 = Particle('p4', Point('p4_pt'), m)
    b_f = ReferenceFrame('b_f')
    b_cm = Point('b_cm')
    mb = symbols('mb')
    b = RigidBody('b', b_cm, b_f, mb, (outer(b_f.x, b_f.x), b_cm))
    p2.point.set_pos(p1.point, a.x)
    p3.point.set_pos(p1.point, a.x + a.y)
    p4.point.set_pos(p1.point, a.y)
    b.masscenter.set_pos(p1.point, a.y + a.z)
    point_o=Point('o')
    point_o.set_pos(p1.point, center_of_mass(p1.point, p1, p2, p3, p4, b))
    expr = 5/(m + mb + 6)*a.x + (m + mb + 3)/(m + mb + 6)*a.y + mb/(m + mb + 6)*a.z
    assert point_o.pos_from(p1.point)-expr == 0


def test_validate_coordinates():
    q1, q2, q3, u1, u2, u3, ua1, ua2, ua3 = dynamicsymbols('q1:4 u1:4 ua1:4')
    s1, s2, s3 = symbols('s1:4')
    # Test normal
    _validate_coordinates([q1, q2, q3], [u1, u2, u3],
                          u_auxiliary=[ua1, ua2, ua3])
    # Test not equal number of coordinates and speeds
    _validate_coordinates([q1, q2])
    _validate_coordinates([q1, q2], [u1])
    _validate_coordinates(speeds=[u1, u2])
    # Test duplicate
    _validate_coordinates([q1, q2, q2], [u1, u2, u3], check_duplicates=False)
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q2], [u1, u2, u3]))
    _validate_coordinates([q1, q2, q3], [u1, u2, u2], check_duplicates=False)
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [u1, u2, u2], check_duplicates=True))
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [q1, u2, u3], check_duplicates=True))
    _validate_coordinates([q1, q2, q3], [u1, u2, u3], check_duplicates=False,
                          u_auxiliary=[u1, ua2, ua2])
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [u1, u2, u3], u_auxiliary=[u1, ua2, ua3]))
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [u1, u2, u3], u_auxiliary=[q1, ua2, ua3]))
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [u1, u2, u3], u_auxiliary=[ua1, ua2, ua2]))
    # Test is_dynamicsymbols
    _validate_coordinates([q1 + q2, q3], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([q1 + q2, q3]))
    _validate_coordinates([s1, q1, q2], [0, u1, u2], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates(
        [s1, q1, q2], [0, u1, u2], is_dynamicsymbols=True))
    _validate_coordinates([s1 + s2 + s3, q1], [0, u1], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates(
        [s1 + s2 + s3, q1], [0, u1], is_dynamicsymbols=True))
    _validate_coordinates(u_auxiliary=[s1, ua1], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates(u_auxiliary=[s1, ua1]))
    # Test normal function
    t = dynamicsymbols._t
    a = symbols('a')
    f1, f2 = symbols('f1:3', cls=Function)
    _validate_coordinates([f1(a), f2(a)], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([f1(a), f2(a)]))
    raises(ValueError, lambda: _validate_coordinates(speeds=[f1(a), f2(a)]))
    dynamicsymbols._t = a
    _validate_coordinates([f1(a), f2(a)])
    raises(ValueError, lambda: _validate_coordinates([f1(t), f2(t)]))
    dynamicsymbols._t = t


def test_parse_linear_solver():
    A, b = Matrix(3, 3, symbols('a:9')), Matrix(3, 2, symbols('b:6'))
    assert _parse_linear_solver(Matrix.LUsolve) == Matrix.LUsolve  # Test callable
    assert _parse_linear_solver('LU')(A, b) == Matrix.LUsolve(A, b)


def test_deprecated_moved_functions():
    from sympy.physics.mechanics.functions import (
        inertia, inertia_of_point_mass, gravity)
    N = ReferenceFrame('N')
    with warns_deprecated_sympy():
        assert inertia(N, 0, 1, 0, 1) == (N.x | N.y) + (N.y | N.x) + (N.y | N.y)
    with warns_deprecated_sympy():
        assert inertia_of_point_mass(1, N.x + N.y, N) == (
            (N.x | N.x) + (N.y | N.y) + 2 * (N.z | N.z) -
            (N.x | N.y) - (N.y | N.x))
    p = Particle('P')
    with warns_deprecated_sympy():
        assert gravity(-2 * N.z, p) == [(p.masscenter, -2 * p.mass * N.z)]
