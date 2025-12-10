from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia, Inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy import expand, zeros, simplify, symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy


def test_rigidbody_default():
    # Test default
    b = RigidBody('B')
    I = inertia(b.frame, *symbols('B_ixx B_iyy B_izz B_ixy B_iyz B_izx'))
    assert b.name == 'B'
    assert b.mass == symbols('B_mass')
    assert b.masscenter.name == 'B_masscenter'
    assert b.inertia == (I, b.masscenter)
    assert b.central_inertia == I
    assert b.frame.name == 'B_frame'
    assert b.__str__() == 'B'
    assert b.__repr__() == (
        "RigidBody('B', masscenter=B_masscenter, frame=B_frame, mass=B_mass, "
        "inertia=Inertia(dyadic=B_ixx*(B_frame.x|B_frame.x) + "
        "B_ixy*(B_frame.x|B_frame.y) + B_izx*(B_frame.x|B_frame.z) + "
        "B_ixy*(B_frame.y|B_frame.x) + B_iyy*(B_frame.y|B_frame.y) + "
        "B_iyz*(B_frame.y|B_frame.z) + B_izx*(B_frame.z|B_frame.x) + "
        "B_iyz*(B_frame.z|B_frame.y) + B_izz*(B_frame.z|B_frame.z), "
        "point=B_masscenter))")


def test_rigidbody():
    m, m2, v1, v2, v3, omega = symbols('m m2 v1 v2 v3 omega')
    A = ReferenceFrame('A')
    A2 = ReferenceFrame('A2')
    P = Point('P')
    P2 = Point('P2')
    I = Dyadic(0)
    I2 = Dyadic(0)
    B = RigidBody('B', P, A, m, (I, P))
    assert B.mass == m
    assert B.frame == A
    assert B.masscenter == P
    assert B.inertia == (I, B.masscenter)

    B.mass = m2
    B.frame = A2
    B.masscenter = P2
    B.inertia = (I2, B.masscenter)
    raises(TypeError, lambda: RigidBody(P, P, A, m, (I, P)))
    raises(TypeError, lambda: RigidBody('B', P, P, m, (I, P)))
    raises(TypeError, lambda: RigidBody('B', P, A, m, (P, P)))
    raises(TypeError, lambda: RigidBody('B', P, A, m, (I, I)))
    assert B.__str__() == 'B'
    assert B.mass == m2
    assert B.frame == A2
    assert B.masscenter == P2
    assert B.inertia == (I2, B.masscenter)
    assert isinstance(B.inertia, Inertia)

    # Testing linear momentum function assuming A2 is the inertial frame
    N = ReferenceFrame('N')
    P2.set_vel(N, v1 * N.x + v2 * N.y + v3 * N.z)
    assert B.linear_momentum(N) == m2 * (v1 * N.x + v2 * N.y + v3 * N.z)


def test_rigidbody2():
    M, v, r, omega, g, h = dynamicsymbols('M v r omega g h')
    N = ReferenceFrame('N')
    b = ReferenceFrame('b')
    b.set_ang_vel(N, omega * b.x)
    P = Point('P')
    I = outer(b.x, b.x)
    Inertia_tuple = (I, P)
    B = RigidBody('B', P, b, M, Inertia_tuple)
    P.set_vel(N, v * b.x)
    assert B.angular_momentum(P, N) == omega * b.x
    O = Point('O')
    O.set_vel(N, v * b.x)
    P.set_pos(O, r * b.y)
    assert B.angular_momentum(O, N) == omega * b.x - M*v*r*b.z
    B.potential_energy = M * g * h
    assert B.potential_energy == M * g * h
    assert expand(2 * B.kinetic_energy(N)) == omega**2 + M * v**2


def test_rigidbody3():
    q1, q2, q3, q4 = dynamicsymbols('q1:5')
    p1, p2, p3 = symbols('p1:4')
    m = symbols('m')

    A = ReferenceFrame('A')
    B = A.orientnew('B', 'axis', [q1, A.x])
    O = Point('O')
    O.set_vel(A, q2*A.x + q3*A.y + q4*A.z)
    P = O.locatenew('P', p1*B.x + p2*B.y + p3*B.z)
    P.v2pt_theory(O, A, B)
    I = outer(B.x, B.x)

    rb1 = RigidBody('rb1', P, B, m, (I, P))
    # I_S/O = I_S/S* + I_S*/O
    rb2 = RigidBody('rb2', P, B, m,
                    (I + inertia_of_point_mass(m, P.pos_from(O), B), O))

    assert rb1.central_inertia == rb2.central_inertia
    assert rb1.angular_momentum(O, A) == rb2.angular_momentum(O, A)


def test_pendulum_angular_momentum():
    """Consider a pendulum of length OA = 2a, of mass m as a rigid body of
    center of mass G (OG = a) which turn around (O,z). The angle between the
    reference frame R and the rod is q.  The inertia of the body is I =
    (G,0,ma^2/3,ma^2/3). """

    m, a = symbols('m, a')
    q = dynamicsymbols('q')

    R = ReferenceFrame('R')
    R1 = R.orientnew('R1', 'Axis', [q, R.z])
    R1.set_ang_vel(R, q.diff() * R.z)

    I = inertia(R1, 0, m * a**2 / 3, m * a**2 / 3)

    O = Point('O')

    A = O.locatenew('A', 2*a * R1.x)
    G = O.locatenew('G', a * R1.x)

    S = RigidBody('S', G, R1, m, (I, G))

    O.set_vel(R, 0)
    A.v2pt_theory(O, R, R1)
    G.v2pt_theory(O, R, R1)

    assert (4 * m * a**2 / 3 * q.diff() * R.z -
            S.angular_momentum(O, R).express(R)) == 0


def test_rigidbody_inertia():
    N = ReferenceFrame('N')
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    Io = inertia(N, Ix, Iy, Iz)
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    R = RigidBody('R', o, N, m, (Io, p))
    I_check = inertia(N, Ix - b ** 2 * m, Iy - a ** 2 * m,
                      Iz - m * (a ** 2 + b ** 2), m * a * b)
    assert isinstance(R.inertia, Inertia)
    assert R.inertia == (Io, p)
    assert R.central_inertia == I_check
    R.central_inertia = Io
    assert R.inertia == (Io, o)
    assert R.central_inertia == Io
    R.inertia = (Io, p)
    assert R.inertia == (Io, p)
    assert R.central_inertia == I_check
    # parse Inertia object
    R.inertia = Inertia(Io, o)
    assert R.inertia == (Io, o)


def test_parallel_axis():
    N = ReferenceFrame('N')
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    Io = inertia(N, Ix, Iy, Iz)
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    R = RigidBody('R', o, N, m, (Io, o))
    Ip = R.parallel_axis(p)
    Ip_expected = inertia(N, Ix + m * b**2, Iy + m * a**2,
                          Iz + m * (a**2 + b**2), ixy=-m * a * b)
    assert Ip == Ip_expected
    # Reference frame from which the parallel axis is viewed should not matter
    A = ReferenceFrame('A')
    A.orient_axis(N, N.z, 1)
    assert simplify(
        (R.parallel_axis(p, A) - Ip_expected).to_matrix(A)) == zeros(3, 3)


def test_deprecated_set_potential_energy():
    m, g, h = symbols('m g h')
    A = ReferenceFrame('A')
    P = Point('P')
    I = Dyadic(0)
    B = RigidBody('B', P, A, m, (I, P))
    with warns_deprecated_sympy():
        B.set_potential_energy(m*g*h)
