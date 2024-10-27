from sympy import (Symbol, symbols, sin, cos, Matrix, zeros,
                                simplify)
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises, warns_deprecated_sympy


def test_default():
    with warns_deprecated_sympy():
        body = Body('body')
    assert body.name == 'body'
    assert body.loads == []
    point = Point('body_masscenter')
    point.set_vel(body.frame, 0)
    com = body.masscenter
    frame = body.frame
    assert com.vel(frame) == point.vel(frame)
    assert body.mass == Symbol('body_mass')
    ixx, iyy, izz = symbols('body_ixx body_iyy body_izz')
    ixy, iyz, izx = symbols('body_ixy body_iyz body_izx')
    assert body.inertia == (inertia(body.frame, ixx, iyy, izz, ixy, iyz, izx),
                            body.masscenter)


def test_custom_rigid_body():
    # Body with RigidBody.
    rigidbody_masscenter = Point('rigidbody_masscenter')
    rigidbody_mass = Symbol('rigidbody_mass')
    rigidbody_frame = ReferenceFrame('rigidbody_frame')
    body_inertia = inertia(rigidbody_frame, 1, 0, 0)
    with warns_deprecated_sympy():
        rigid_body = Body('rigidbody_body', rigidbody_masscenter,
                          rigidbody_mass, rigidbody_frame, body_inertia)
    com = rigid_body.masscenter
    frame = rigid_body.frame
    rigidbody_masscenter.set_vel(rigidbody_frame, 0)
    assert com.vel(frame) == rigidbody_masscenter.vel(frame)
    assert com.pos_from(com) == rigidbody_masscenter.pos_from(com)

    assert rigid_body.mass == rigidbody_mass
    assert rigid_body.inertia == (body_inertia, rigidbody_masscenter)

    assert rigid_body.is_rigidbody

    assert hasattr(rigid_body, 'masscenter')
    assert hasattr(rigid_body, 'mass')
    assert hasattr(rigid_body, 'frame')
    assert hasattr(rigid_body, 'inertia')


def test_particle_body():
    #  Body with Particle
    particle_masscenter = Point('particle_masscenter')
    particle_mass = Symbol('particle_mass')
    particle_frame = ReferenceFrame('particle_frame')
    with warns_deprecated_sympy():
        particle_body = Body('particle_body', particle_masscenter,
                             particle_mass, particle_frame)
    com = particle_body.masscenter
    frame = particle_body.frame
    particle_masscenter.set_vel(particle_frame, 0)
    assert com.vel(frame) == particle_masscenter.vel(frame)
    assert com.pos_from(com) == particle_masscenter.pos_from(com)

    assert particle_body.mass == particle_mass
    assert not hasattr(particle_body, "_inertia")
    assert hasattr(particle_body, 'frame')
    assert hasattr(particle_body, 'masscenter')
    assert hasattr(particle_body, 'mass')
    assert particle_body.inertia == (Dyadic(0), particle_body.masscenter)
    assert particle_body.central_inertia == Dyadic(0)
    assert not particle_body.is_rigidbody

    particle_body.central_inertia = inertia(particle_frame, 1, 1, 1)
    assert particle_body.central_inertia == inertia(particle_frame, 1, 1, 1)
    assert particle_body.is_rigidbody

    with warns_deprecated_sympy():
        particle_body = Body('particle_body', mass=particle_mass)
    assert not particle_body.is_rigidbody
    point = particle_body.masscenter.locatenew('point', particle_body.x)
    point_inertia = particle_mass * inertia(particle_body.frame, 0, 1, 1)
    particle_body.inertia = (point_inertia, point)
    assert particle_body.inertia == (point_inertia, point)
    assert particle_body.central_inertia == Dyadic(0)
    assert particle_body.is_rigidbody


def test_particle_body_add_force():
    #  Body with Particle
    particle_masscenter = Point('particle_masscenter')
    particle_mass = Symbol('particle_mass')
    particle_frame = ReferenceFrame('particle_frame')
    with warns_deprecated_sympy():
        particle_body = Body('particle_body', particle_masscenter,
                             particle_mass, particle_frame)

    a = Symbol('a')
    force_vector = a * particle_body.frame.x
    particle_body.apply_force(force_vector, particle_body.masscenter)
    assert len(particle_body.loads) == 1
    point = particle_body.masscenter.locatenew(
        particle_body._name + '_point0', 0)
    point.set_vel(particle_body.frame, 0)
    force_point = particle_body.loads[0][0]

    frame = particle_body.frame
    assert force_point.vel(frame) == point.vel(frame)
    assert force_point.pos_from(force_point) == point.pos_from(force_point)

    assert particle_body.loads[0][1] == force_vector


def test_body_add_force():
    # Body with RigidBody.
    rigidbody_masscenter = Point('rigidbody_masscenter')
    rigidbody_mass = Symbol('rigidbody_mass')
    rigidbody_frame = ReferenceFrame('rigidbody_frame')
    body_inertia = inertia(rigidbody_frame, 1, 0, 0)
    with warns_deprecated_sympy():
        rigid_body = Body('rigidbody_body', rigidbody_masscenter,
                          rigidbody_mass, rigidbody_frame, body_inertia)

    l = Symbol('l')
    Fa = Symbol('Fa')
    point = rigid_body.masscenter.locatenew(
        'rigidbody_body_point0',
        l * rigid_body.frame.x)
    point.set_vel(rigid_body.frame, 0)
    force_vector = Fa * rigid_body.frame.z
    # apply_force with point
    rigid_body.apply_force(force_vector, point)
    assert len(rigid_body.loads) == 1
    force_point = rigid_body.loads[0][0]
    frame = rigid_body.frame
    assert force_point.vel(frame) == point.vel(frame)
    assert force_point.pos_from(force_point) == point.pos_from(force_point)
    assert rigid_body.loads[0][1] == force_vector
    # apply_force without point
    rigid_body.apply_force(force_vector)
    assert len(rigid_body.loads) == 2
    assert rigid_body.loads[1][1] == force_vector
    # passing something else than point
    raises(TypeError, lambda: rigid_body.apply_force(force_vector,  0))
    raises(TypeError, lambda: rigid_body.apply_force(0))

def test_body_add_torque():
    with warns_deprecated_sympy():
        body = Body('body')
    torque_vector = body.frame.x
    body.apply_torque(torque_vector)

    assert len(body.loads) == 1
    assert body.loads[0] == (body.frame, torque_vector)
    raises(TypeError, lambda: body.apply_torque(0))

def test_body_masscenter_vel():
    with warns_deprecated_sympy():
        A = Body('A')
    N = ReferenceFrame('N')
    with warns_deprecated_sympy():
        B = Body('B', frame=N)
    A.masscenter.set_vel(N, N.z)
    assert A.masscenter_vel(B) == N.z
    assert A.masscenter_vel(N) == N.z

def test_body_ang_vel():
    with warns_deprecated_sympy():
        A = Body('A')
    N = ReferenceFrame('N')
    with warns_deprecated_sympy():
        B = Body('B', frame=N)
    A.frame.set_ang_vel(N, N.y)
    assert A.ang_vel_in(B) == N.y
    assert B.ang_vel_in(A) == -N.y
    assert A.ang_vel_in(N) == N.y

def test_body_dcm():
    with warns_deprecated_sympy():
        A = Body('A')
        B = Body('B')
    A.frame.orient_axis(B.frame, B.frame.z, 10)
    assert A.dcm(B) == Matrix([[cos(10), sin(10), 0], [-sin(10), cos(10), 0], [0, 0, 1]])
    assert A.dcm(B.frame) == Matrix([[cos(10), sin(10), 0], [-sin(10), cos(10), 0], [0, 0, 1]])

def test_body_axis():
    N = ReferenceFrame('N')
    with warns_deprecated_sympy():
        B = Body('B', frame=N)
    assert B.x == N.x
    assert B.y == N.y
    assert B.z == N.z

def test_apply_force_multiple_one_point():
    a, b = symbols('a b')
    P = Point('P')
    with warns_deprecated_sympy():
        B = Body('B')
    f1 = a*B.x
    f2 = b*B.y
    B.apply_force(f1, P)
    assert B.loads == [(P, f1)]
    B.apply_force(f2, P)
    assert B.loads == [(P, f1+f2)]

def test_apply_force():
    f, g = symbols('f g')
    q, x, v1, v2 = dynamicsymbols('q x v1 v2')
    P1 = Point('P1')
    P2 = Point('P2')
    with warns_deprecated_sympy():
        B1 = Body('B1')
        B2 = Body('B2')
    N = ReferenceFrame('N')

    P1.set_vel(B1.frame, v1*B1.x)
    P2.set_vel(B2.frame, v2*B2.x)
    force = f*q*N.z # time varying force

    B1.apply_force(force, P1, B2, P2) #applying equal and opposite force on moving points
    assert B1.loads == [(P1, force)]
    assert B2.loads == [(P2, -force)]

    g1 = B1.mass*g*N.y
    g2 = B2.mass*g*N.y

    B1.apply_force(g1) #applying gravity on B1 masscenter
    B2.apply_force(g2) #applying gravity on B2 masscenter

    assert B1.loads == [(P1,force), (B1.masscenter, g1)]
    assert B2.loads == [(P2, -force), (B2.masscenter, g2)]

    force2 = x*N.x

    B1.apply_force(force2, reaction_body=B2) #Applying time varying force on masscenter

    assert B1.loads == [(P1, force), (B1.masscenter, force2+g1)]
    assert B2.loads == [(P2, -force), (B2.masscenter, -force2+g2)]

def test_apply_torque():
    t = symbols('t')
    q = dynamicsymbols('q')
    with warns_deprecated_sympy():
        B1 = Body('B1')
        B2 = Body('B2')
    N = ReferenceFrame('N')
    torque = t*q*N.x

    B1.apply_torque(torque, B2) #Applying equal and opposite torque
    assert B1.loads == [(B1.frame, torque)]
    assert B2.loads == [(B2.frame, -torque)]

    torque2 = t*N.y
    B1.apply_torque(torque2)
    assert B1.loads == [(B1.frame, torque+torque2)]

def test_clear_load():
    a = symbols('a')
    P = Point('P')
    with warns_deprecated_sympy():
        B = Body('B')
    force = a*B.z
    B.apply_force(force, P)
    assert B.loads == [(P, force)]
    B.clear_loads()
    assert B.loads == []

def test_remove_load():
    P1 = Point('P1')
    P2 = Point('P2')
    with warns_deprecated_sympy():
        B = Body('B')
    f1 = B.x
    f2 = B.y
    B.apply_force(f1, P1)
    B.apply_force(f2, P2)
    assert B.loads == [(P1, f1), (P2, f2)]
    B.remove_load(P2)
    assert B.loads == [(P1, f1)]
    B.apply_torque(f1.cross(f2))
    assert B.loads == [(P1, f1), (B.frame, f1.cross(f2))]
    B.remove_load()
    assert B.loads == [(P1, f1)]

def test_apply_loads_on_multi_degree_freedom_holonomic_system():
    """Example based on: https://pydy.readthedocs.io/en/latest/examples/multidof-holonomic.html"""
    with warns_deprecated_sympy():
        W = Body('W') #Wall
        B = Body('B') #Block
        P = Body('P') #Pendulum
        b = Body('b') #bob
    q1, q2 = dynamicsymbols('q1 q2') #generalized coordinates
    k, c, g, kT = symbols('k c g kT') #constants
    F, T = dynamicsymbols('F T') #Specified forces

    #Applying forces
    B.apply_force(F*W.x)
    W.apply_force(k*q1*W.x, reaction_body=B) #Spring force
    W.apply_force(c*q1.diff()*W.x, reaction_body=B) #dampner
    P.apply_force(P.mass*g*W.y)
    b.apply_force(b.mass*g*W.y)

    #Applying torques
    P.apply_torque(kT*q2*W.z, reaction_body=b)
    P.apply_torque(T*W.z)

    assert B.loads == [(B.masscenter, (F - k*q1 - c*q1.diff())*W.x)]
    assert P.loads == [(P.masscenter, P.mass*g*W.y), (P.frame, (T + kT*q2)*W.z)]
    assert b.loads == [(b.masscenter, b.mass*g*W.y), (b.frame, -kT*q2*W.z)]
    assert W.loads == [(W.masscenter, (c*q1.diff() + k*q1)*W.x)]


def test_parallel_axis():
    N = ReferenceFrame('N')
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    Io = inertia(N, Ix, Iy, Iz)
    # Test RigidBody
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    with warns_deprecated_sympy():
        R = Body('R', masscenter=o, frame=N, mass=m, central_inertia=Io)
    Ip = R.parallel_axis(p)
    Ip_expected = inertia(N, Ix + m * b**2, Iy + m * a**2,
                          Iz + m * (a**2 + b**2), ixy=-m * a * b)
    assert Ip == Ip_expected
    # Reference frame from which the parallel axis is viewed should not matter
    A = ReferenceFrame('A')
    A.orient_axis(N, N.z, 1)
    assert simplify(
        (R.parallel_axis(p, A) - Ip_expected).to_matrix(A)) == zeros(3, 3)
    # Test Particle
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    with warns_deprecated_sympy():
        P = Body('P', masscenter=o, mass=m, frame=N)
    Ip = P.parallel_axis(p, N)
    Ip_expected = inertia(N, m * b ** 2, m * a ** 2, m * (a ** 2 + b ** 2),
                          ixy=-m * a * b)
    assert not P.is_rigidbody
    assert Ip == Ip_expected
