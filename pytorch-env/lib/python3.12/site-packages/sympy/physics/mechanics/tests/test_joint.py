from sympy.core.function import expand_mul
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy import Matrix, simplify, eye, zeros
from sympy.core.symbol import symbols
from sympy.physics.mechanics import (
    dynamicsymbols, RigidBody, Particle, JointsMethod, PinJoint, PrismaticJoint,
    CylindricalJoint, PlanarJoint, SphericalJoint, WeldJoint, Body)
from sympy.physics.mechanics.joint import Joint
from sympy.physics.vector import Vector, ReferenceFrame, Point
from sympy.testing.pytest import raises, warns_deprecated_sympy


t = dynamicsymbols._t # type: ignore


def _generate_body(interframe=False):
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    P = RigidBody('P', frame=N)
    C = RigidBody('C', frame=A)
    if interframe:
        Pint, Cint = ReferenceFrame('P_int'), ReferenceFrame('C_int')
        Pint.orient_axis(N, N.x, pi)
        Cint.orient_axis(A, A.y, -pi / 2)
        return N, A, P, C, Pint, Cint
    return N, A, P, C


def test_Joint():
    parent = RigidBody('parent')
    child = RigidBody('child')
    raises(TypeError, lambda: Joint('J', parent, child))


def test_coordinate_generation():
    q, u, qj, uj = dynamicsymbols('q u q_J u_J')
    q0j, q1j, q2j, q3j, u0j, u1j, u2j, u3j = dynamicsymbols('q0:4_J u0:4_J')
    q0, q1, q2, q3, u0, u1, u2, u3 = dynamicsymbols('q0:4 u0:4')
    _, _, P, C = _generate_body()
    # Using PinJoint to access Joint's coordinate generation method
    J = PinJoint('J', P, C)
    # Test single given
    assert J._fill_coordinate_list(q, 1) == Matrix([q])
    assert J._fill_coordinate_list([u], 1) == Matrix([u])
    assert J._fill_coordinate_list([u], 1, offset=2) == Matrix([u])
    # Test None
    assert J._fill_coordinate_list(None, 1) == Matrix([qj])
    assert J._fill_coordinate_list([None], 1) == Matrix([qj])
    assert J._fill_coordinate_list([q0, None, None], 3) == Matrix(
        [q0, q1j, q2j])
    # Test autofill
    assert J._fill_coordinate_list(None, 3) == Matrix([q0j, q1j, q2j])
    assert J._fill_coordinate_list([], 3) == Matrix([q0j, q1j, q2j])
    # Test offset
    assert J._fill_coordinate_list([], 3, offset=1) == Matrix([q1j, q2j, q3j])
    assert J._fill_coordinate_list([q1, None, q3], 3, offset=1) == Matrix(
        [q1, q2j, q3])
    assert J._fill_coordinate_list(None, 2, offset=2) == Matrix([q2j, q3j])
    # Test label
    assert J._fill_coordinate_list(None, 1, 'u') == Matrix([uj])
    assert J._fill_coordinate_list([], 3, 'u') == Matrix([u0j, u1j, u2j])
    # Test single numbering
    assert J._fill_coordinate_list(None, 1, number_single=True) == Matrix([q0j])
    assert J._fill_coordinate_list([], 1, 'u', 2, True) == Matrix([u2j])
    assert J._fill_coordinate_list([], 3, 'q') == Matrix([q0j, q1j, q2j])
    # Test invalid number of coordinates supplied
    raises(ValueError, lambda: J._fill_coordinate_list([q0, q1], 1))
    raises(ValueError, lambda: J._fill_coordinate_list([u0, u1, None], 2, 'u'))
    raises(ValueError, lambda: J._fill_coordinate_list([q0, q1], 3))
    # Test incorrect coordinate type
    raises(TypeError, lambda: J._fill_coordinate_list([q0, symbols('q1')], 2))
    raises(TypeError, lambda: J._fill_coordinate_list([q0 + q1, q1], 2))
    # Test if derivative as generalized speed is allowed
    _, _, P, C = _generate_body()
    PinJoint('J', P, C, q1, q1.diff(t))
    # Test duplicate coordinates
    _, _, P, C = _generate_body()
    raises(ValueError, lambda: SphericalJoint('J', P, C, [q1j, None, None]))
    raises(ValueError, lambda: SphericalJoint('J', P, C, speeds=[u0, u0, u1]))


def test_pin_joint():
    P = RigidBody('P')
    C = RigidBody('C')
    l, m = symbols('l m')
    q, u = dynamicsymbols('q_J, u_J')
    Pj = PinJoint('J', P, C)
    assert Pj.name == 'J'
    assert Pj.parent == P
    assert Pj.child == C
    assert Pj.coordinates == Matrix([q])
    assert Pj.speeds == Matrix([u])
    assert Pj.kdes == Matrix([u - q.diff(t)])
    assert Pj.joint_axis == P.frame.x
    assert Pj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Pj.parent_point.pos_from(P.masscenter) == Vector(0)
    assert Pj.parent_point.pos_from(Pj._child_point) == Vector(0)
    assert C.masscenter.pos_from(P.masscenter) == Vector(0)
    assert Pj.parent_interframe == P.frame
    assert Pj.child_interframe == C.frame
    assert Pj.__str__() == 'PinJoint: J  parent: P  child: C'

    P1 = RigidBody('P1')
    C1 = RigidBody('C1')
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P1.frame, P1.y, pi / 2)
    J1 = PinJoint('J1', P1, C1, parent_point=l*P1.frame.x,
                  child_point=m*C1.frame.y, joint_axis=P1.frame.z,
                  parent_interframe=Pint)
    assert J1._joint_axis == P1.frame.z
    assert J1._child_point.pos_from(C1.masscenter) == m * C1.frame.y
    assert J1._parent_point.pos_from(P1.masscenter) == l * P1.frame.x
    assert J1._parent_point.pos_from(J1._child_point) == Vector(0)
    assert (P1.masscenter.pos_from(C1.masscenter) ==
            -l*P1.frame.x + m*C1.frame.y)
    assert J1.parent_interframe == Pint
    assert J1.child_interframe == C1.frame

    q, u = dynamicsymbols('q, u')
    N, A, P, C, Pint, Cint = _generate_body(True)
    parent_point = P.masscenter.locatenew('parent_point', N.x + N.y)
    child_point = C.masscenter.locatenew('child_point', C.y + C.z)
    J = PinJoint('J', P, C, q, u, parent_point=parent_point,
                 child_point=child_point, parent_interframe=Pint,
                 child_interframe=Cint, joint_axis=N.z)
    assert J.joint_axis == N.z
    assert J.parent_point.vel(N) == 0
    assert J.parent_point == parent_point
    assert J.child_point == child_point
    assert J.child_point.pos_from(P.masscenter) == N.x + N.y
    assert J.parent_point.pos_from(C.masscenter) == C.y + C.z
    assert C.masscenter.pos_from(P.masscenter) == N.x + N.y - C.y - C.z
    assert C.masscenter.vel(N).express(N) == (u * sin(q) - u * cos(q)) * N.x + (
            -u * sin(q) - u * cos(q)) * N.y
    assert J.parent_interframe == Pint
    assert J.child_interframe == Cint


def test_particle_compatibility():
    m, l = symbols('m l')
    C_frame = ReferenceFrame('C')
    P = Particle('P')
    C = Particle('C', mass=m)
    q, u = dynamicsymbols('q, u')
    J = PinJoint('J', P, C, q, u, child_interframe=C_frame,
                 child_point=l * C_frame.y)
    assert J.child_interframe == C_frame
    assert J.parent_interframe.name == 'J_P_frame'
    assert C.masscenter.pos_from(P.masscenter) == -l * C_frame.y
    assert C_frame.dcm(J.parent_interframe) == Matrix([[1, 0, 0],
                                                       [0, cos(q), sin(q)],
                                                       [0, -sin(q), cos(q)]])
    assert C.masscenter.vel(J.parent_interframe) == -l * u * C_frame.z
    # Test with specified joint axis
    P_frame = ReferenceFrame('P')
    C_frame = ReferenceFrame('C')
    P = Particle('P')
    C = Particle('C', mass=m)
    q, u = dynamicsymbols('q, u')
    J = PinJoint('J', P, C, q, u, parent_interframe=P_frame,
                 child_interframe=C_frame, child_point=l * C_frame.y,
                 joint_axis=P_frame.z)
    assert J.joint_axis == J.parent_interframe.z
    assert C_frame.dcm(J.parent_interframe) == Matrix([[cos(q), sin(q), 0],
                                                       [-sin(q), cos(q), 0],
                                                       [0, 0, 1]])
    assert P.masscenter.vel(J.parent_interframe) == 0
    assert C.masscenter.vel(J.parent_interframe) == l * u * C_frame.x
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1:4 u1:4')
    qdot_to_u = {qi.diff(t): ui for qi, ui in ((q1, u1), (q2, u2), (q3, u3))}
    # Test compatibility for prismatic joint
    P, C = Particle('P'), Particle('C')
    J = PrismaticJoint('J', P, C, q, u)
    assert J.parent_interframe.dcm(J.child_interframe) == eye(3)
    assert C.masscenter.pos_from(P.masscenter) == q * J.parent_interframe.x
    assert P.masscenter.vel(J.parent_interframe) == 0
    assert C.masscenter.vel(J.parent_interframe) == u * J.parent_interframe.x
    # Test compatibility for cylindrical joint
    P, C = Particle('P'), Particle('C')
    P_frame = ReferenceFrame('P_frame')
    J = CylindricalJoint('J', P, C, q1, q2, u1, u2, parent_interframe=P_frame,
                         parent_point=l * P_frame.x, joint_axis=P_frame.y)
    assert J.parent_interframe.dcm(J.child_interframe) == Matrix([
        [cos(q1), 0, sin(q1)], [0, 1, 0], [-sin(q1), 0, cos(q1)]])
    assert C.masscenter.pos_from(P.masscenter) == l * P_frame.x + q2 * P_frame.y
    assert C.masscenter.vel(J.parent_interframe) == u2 * P_frame.y
    assert P.masscenter.vel(J.child_interframe).xreplace(qdot_to_u) == (
        -u2 * P_frame.y - l * u1 * P_frame.z)
    # Test compatibility for planar joint
    P, C = Particle('P'), Particle('C')
    C_frame = ReferenceFrame('C_frame')
    J = PlanarJoint('J', P, C, q1, [q2, q3], u1, [u2, u3],
                    child_interframe=C_frame, child_point=l * C_frame.z)
    P_frame = J.parent_interframe
    assert J.parent_interframe.dcm(J.child_interframe) == Matrix([
        [1, 0, 0], [0, cos(q1), -sin(q1)], [0, sin(q1), cos(q1)]])
    assert C.masscenter.pos_from(P.masscenter) == (
        -l * C_frame.z + q2 * P_frame.y + q3 * P_frame.z)
    assert C.masscenter.vel(J.parent_interframe) == (
        l * u1 * C_frame.y + u2 * P_frame.y + u3 * P_frame.z)
    # Test compatibility for weld joint
    P, C = Particle('P'), Particle('C')
    C_frame, P_frame = ReferenceFrame('C_frame'), ReferenceFrame('P_frame')
    J = WeldJoint('J', P, C, parent_interframe=P_frame,
                  child_interframe=C_frame, parent_point=l * P_frame.x,
                  child_point=l * C_frame.y)
    assert P_frame.dcm(C_frame) == eye(3)
    assert C.masscenter.pos_from(P.masscenter) == l * P_frame.x - l * C_frame.y
    assert C.masscenter.vel(J.parent_interframe) == 0


def test_body_compatibility():
    m, l = symbols('m l')
    C_frame = ReferenceFrame('C')
    with warns_deprecated_sympy():
        P = Body('P')
        C = Body('C', mass=m, frame=C_frame)
    q, u = dynamicsymbols('q, u')
    PinJoint('J', P, C, q, u, child_point=l * C_frame.y)
    assert C.frame == C_frame
    assert P.frame.name == 'P_frame'
    assert C.masscenter.pos_from(P.masscenter) == -l * C.y
    assert C.frame.dcm(P.frame) == Matrix([[1, 0, 0],
                                           [0, cos(q), sin(q)],
                                           [0, -sin(q), cos(q)]])
    assert C.masscenter.vel(P.frame) == -l * u * C.z


def test_pin_joint_double_pendulum():
    q1, q2 = dynamicsymbols('q1 q2')
    u1, u2 = dynamicsymbols('u1 u2')
    m, l = symbols('m l')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = RigidBody('C', frame=N)  # ceiling
    PartP = RigidBody('P', frame=A, mass=m)
    PartR = RigidBody('R', frame=B, mass=m)

    J1 = PinJoint('J1', C, PartP, speeds=u1, coordinates=q1,
                  child_point=-l*A.x, joint_axis=C.frame.z)
    J2 = PinJoint('J2', PartP, PartR, speeds=u2, coordinates=q2,
                  child_point=-l*B.x, joint_axis=PartP.frame.z)

    # Check orientation
    assert N.dcm(A) == Matrix([[cos(q1), -sin(q1), 0],
                               [sin(q1), cos(q1), 0], [0, 0, 1]])
    assert A.dcm(B) == Matrix([[cos(q2), -sin(q2), 0],
                               [sin(q2), cos(q2), 0], [0, 0, 1]])
    assert simplify(N.dcm(B)) == Matrix([[cos(q1 + q2), -sin(q1 + q2), 0],
                                                 [sin(q1 + q2), cos(q1 + q2), 0],
                                                 [0, 0, 1]])

    # Check Angular Velocity
    assert A.ang_vel_in(N) == u1 * N.z
    assert B.ang_vel_in(A) == u2 * A.z
    assert B.ang_vel_in(N) == u1 * N.z + u2 * A.z

    # Check kde
    assert J1.kdes == Matrix([u1 - q1.diff(t)])
    assert J2.kdes == Matrix([u2 - q2.diff(t)])

    # Check Linear Velocity
    assert PartP.masscenter.vel(N) == l*u1*A.y
    assert PartR.masscenter.vel(A) == l*u2*B.y
    assert PartR.masscenter.vel(N) == l*u1*A.y + l*(u1 + u2)*B.y


def test_pin_joint_chaos_pendulum():
    mA, mB, lA, lB, h = symbols('mA, mB, lA, lB, h')
    theta, phi, omega, alpha = dynamicsymbols('theta phi omega alpha')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    lA = (lB - h / 2) / 2
    lC = (lB/2 + h/4)
    rod = RigidBody('rod', frame=A, mass=mA)
    plate = RigidBody('plate', mass=mB, frame=B)
    C = RigidBody('C', frame=N)
    J1 = PinJoint('J1', C, rod, coordinates=theta, speeds=omega,
                  child_point=lA*A.z, joint_axis=N.y)
    J2 = PinJoint('J2', rod, plate, coordinates=phi, speeds=alpha,
                  parent_point=lC*A.z, joint_axis=A.z)

    # Check orientation
    assert A.dcm(N) == Matrix([[cos(theta), 0, -sin(theta)],
                               [0, 1, 0],
                               [sin(theta), 0, cos(theta)]])
    assert A.dcm(B) == Matrix([[cos(phi), -sin(phi), 0],
                               [sin(phi), cos(phi), 0],
                               [0, 0, 1]])
    assert B.dcm(N) == Matrix([
        [cos(phi)*cos(theta), sin(phi), -sin(theta)*cos(phi)],
        [-sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta)],
        [sin(theta), 0, cos(theta)]])

    # Check Angular Velocity
    assert A.ang_vel_in(N) == omega*N.y
    assert A.ang_vel_in(B) == -alpha*A.z
    assert N.ang_vel_in(B) == -omega*N.y - alpha*A.z

    # Check kde
    assert J1.kdes == Matrix([omega - theta.diff(t)])
    assert J2.kdes == Matrix([alpha - phi.diff(t)])

    # Check pos of masscenters
    assert C.masscenter.pos_from(rod.masscenter) == lA*A.z
    assert rod.masscenter.pos_from(plate.masscenter) == - lC * A.z

    # Check Linear Velocities
    assert rod.masscenter.vel(N) == (h/4 - lB/2)*omega*A.x
    assert plate.masscenter.vel(N) == ((h/4 - lB/2)*omega +
                                       (h/4 + lB/2)*omega)*A.x


def test_pin_joint_interframe():
    q, u = dynamicsymbols('q, u')
    # Check not connected
    N, A, P, C = _generate_body()
    Pint, Cint = ReferenceFrame('Pint'), ReferenceFrame('Cint')
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint))
    raises(ValueError, lambda: PinJoint('J', P, C, child_interframe=Cint))
    # Check not fixed interframe
    Pint.orient_axis(N, N.z, q)
    Cint.orient_axis(A, A.z, q)
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint))
    raises(ValueError, lambda: PinJoint('J', P, C, child_interframe=Cint))
    # Check only parent_interframe
    N, A, P, C = _generate_body()
    Pint = ReferenceFrame('Pint')
    Pint.orient_body_fixed(N, (pi / 4, pi, pi / 3), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=N.x, child_point=-C.y,
             parent_interframe=Pint, joint_axis=Pint.x)
    assert simplify(N.dcm(A)) - Matrix([
        [-1 / 2, sqrt(3) * cos(q) / 2, -sqrt(3) * sin(q) / 2],
        [sqrt(6) / 4, sqrt(2) * (2 * sin(q) + cos(q)) / 4,
         sqrt(2) * (-sin(q) + 2 * cos(q)) / 4],
        [sqrt(6) / 4, sqrt(2) * (-2 * sin(q) + cos(q)) / 4,
         -sqrt(2) * (sin(q) + 2 * cos(q)) / 4]]) == zeros(3)
    assert A.ang_vel_in(N) == u * Pint.x
    assert C.masscenter.pos_from(P.masscenter) == N.x + A.y
    assert C.masscenter.vel(N) == u * A.z
    assert P.masscenter.vel(Pint) == Vector(0)
    assert C.masscenter.vel(Pint) == u * A.z
    # Check only child_interframe
    N, A, P, C = _generate_body()
    Cint = ReferenceFrame('Cint')
    Cint.orient_body_fixed(A, (2 * pi / 3, -pi, pi / 2), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=-N.z, child_point=C.x,
             child_interframe=Cint, joint_axis=P.x + P.z)
    assert simplify(N.dcm(A)) == Matrix([
        [-sqrt(2) * sin(q) / 2,
         -sqrt(3) * (cos(q) - 1) / 4 - cos(q) / 4 - S(1) / 4,
         sqrt(3) * (cos(q) + 1) / 4 - cos(q) / 4 + S(1) / 4],
        [cos(q), (sqrt(2) + sqrt(6)) * -sin(q) / 4,
         (-sqrt(2) + sqrt(6)) * sin(q) / 4],
        [sqrt(2) * sin(q) / 2,
         sqrt(3) * (cos(q) + 1) / 4 + cos(q) / 4 - S(1) / 4,
         sqrt(3) * (1 - cos(q)) / 4 + cos(q) / 4 + S(1) / 4]])
    assert A.ang_vel_in(N) == sqrt(2) * u / 2 * N.x + sqrt(2) * u / 2 * N.z
    assert C.masscenter.pos_from(P.masscenter) == - N.z - A.x
    assert C.masscenter.vel(N).simplify() == (
        -sqrt(6) - sqrt(2)) * u / 4 * A.y + (
               -sqrt(2) + sqrt(6)) * u / 4 * A.z
    assert C.masscenter.vel(Cint) == Vector(0)
    # Check combination
    N, A, P, C = _generate_body()
    Pint, Cint = ReferenceFrame('Pint'), ReferenceFrame('Cint')
    Pint.orient_body_fixed(N, (-pi / 2, pi, pi / 2), 'xyz')
    Cint.orient_body_fixed(A, (2 * pi / 3, -pi, pi / 2), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=N.x - N.y, child_point=-C.z,
             parent_interframe=Pint, child_interframe=Cint,
             joint_axis=Pint.x + Pint.z)
    assert simplify(N.dcm(A)) == Matrix([
        [cos(q), (sqrt(2) + sqrt(6)) * -sin(q) / 4,
         (-sqrt(2) + sqrt(6)) * sin(q) / 4],
        [-sqrt(2) * sin(q) / 2,
         -sqrt(3) * (cos(q) + 1) / 4 - cos(q) / 4 + S(1) / 4,
         sqrt(3) * (cos(q) - 1) / 4 - cos(q) / 4 - S(1) / 4],
        [sqrt(2) * sin(q) / 2,
         sqrt(3) * (cos(q) - 1) / 4 + cos(q) / 4 + S(1) / 4,
         -sqrt(3) * (cos(q) + 1) / 4 + cos(q) / 4 - S(1) / 4]])
    assert A.ang_vel_in(N) == sqrt(2) * u / 2 * Pint.x + sqrt(
        2) * u / 2 * Pint.z
    assert C.masscenter.pos_from(P.masscenter) == N.x - N.y + A.z
    N_v_C = (-sqrt(2) + sqrt(6)) * u / 4 * A.x
    assert C.masscenter.vel(N).simplify() == N_v_C
    assert C.masscenter.vel(Pint).simplify() == N_v_C
    assert C.masscenter.vel(Cint) == Vector(0)


def test_pin_joint_joint_axis():
    q, u = dynamicsymbols('q, u')
    # Check parent as reference
    N, A, P, C, Pint, Cint = _generate_body(True)
    pin = PinJoint('J', P, C, q, u, parent_interframe=Pint,
                   child_interframe=Cint, joint_axis=P.y)
    assert pin.joint_axis == P.y
    assert N.dcm(A) == Matrix([[sin(q), 0, cos(q)], [0, -1, 0],
                               [cos(q), 0, -sin(q)]])
    # Check parent_interframe as reference
    N, A, P, C, Pint, Cint = _generate_body(True)
    pin = PinJoint('J', P, C, q, u, parent_interframe=Pint,
                   child_interframe=Cint, joint_axis=Pint.y)
    assert pin.joint_axis == Pint.y
    assert N.dcm(A) == Matrix([[-sin(q), 0, cos(q)], [0, -1, 0],
                               [cos(q), 0, sin(q)]])
    # Check combination of joint_axis with interframes supplied as vectors (2x)
    N, A, P, C = _generate_body()
    pin = PinJoint('J', P, C, q, u, parent_interframe=N.z,
                   child_interframe=-C.z, joint_axis=N.z)
    assert pin.joint_axis == N.z
    assert N.dcm(A) == Matrix([[-cos(q), -sin(q), 0], [-sin(q), cos(q), 0],
                               [0, 0, -1]])
    N, A, P, C = _generate_body()
    pin = PinJoint('J', P, C, q, u, parent_interframe=N.z,
                   child_interframe=-C.z, joint_axis=N.x)
    assert pin.joint_axis == N.x
    assert N.dcm(A) == Matrix([[-1, 0, 0], [0, cos(q), sin(q)],
                               [0, sin(q), -cos(q)]])
    # Check time varying axis
    N, A, P, C, Pint, Cint = _generate_body(True)
    raises(ValueError, lambda: PinJoint('J', P, C,
                                        joint_axis=cos(q) * N.x + sin(q) * N.y))
    # Check joint_axis provided in child frame
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=C.x))
    # Check some invalid combinations
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=P.x + C.y))
    raises(ValueError, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=Pint.x + C.y))
    raises(ValueError, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=P.x + Cint.y))
    # Check valid special combination
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, parent_interframe=Pint, child_interframe=Cint,
             joint_axis=Pint.x + P.y)
    # Check invalid zero vector
    raises(Exception, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=Vector(0)))
    raises(Exception, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=P.y + Pint.y))


def test_pin_joint_arbitrary_axis():
    q, u = dynamicsymbols('q_J, u_J')

    # When the bodies are attached though masscenters but axes are opposite.
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, child_interframe=-A.x)

    assert (-A.x).angle_between(N.x) == 0
    assert -A.x.express(N) == N.x
    assert A.dcm(N) == Matrix([[-1, 0, 0],
                               [0, -cos(q), -sin(q)],
                               [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u*N.x
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    assert C.masscenter.pos_from(P.masscenter) == 0
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == 0
    assert C.masscenter.vel(N) == 0

    # When axes are different and parent joint is at masscenter but child joint
    # is at a unit vector from child masscenter.
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, child_interframe=A.y, child_point=A.x)

    assert A.y.angle_between(N.x) == 0  # Axis are aligned
    assert A.y.express(N) == N.x
    assert A.dcm(N) == Matrix([[0, -cos(q), -sin(q)],
                               [1, 0, 0],
                               [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u*N.x
    assert A.ang_vel_in(N).express(A) == u * A.y
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    assert A.ang_vel_in(N).cross(A.y) == 0
    assert C.masscenter.vel(N) == u*A.z
    assert C.masscenter.pos_from(P.masscenter) == -A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            cos(q)*N.y + sin(q)*N.z)
    assert C.masscenter.vel(N).angle_between(A.x) == pi/2

    # Similar to previous case but wrt parent body
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_interframe=N.y, parent_point=N.x)

    assert N.y.angle_between(A.x) == 0  # Axis are aligned
    assert N.y.express(A) == A.x
    assert A.dcm(N) == Matrix([[0, 1, 0],
                               [-cos(q), 0, sin(q)],
                               [sin(q), 0, cos(q)]])
    assert A.ang_vel_in(N) == u*N.y
    assert A.ang_vel_in(N).express(A) == u*A.x
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    angle = A.ang_vel_in(N).angle_between(A.x)
    assert angle.xreplace({u: 1}) == 0
    assert C.masscenter.vel(N) == 0
    assert C.masscenter.pos_from(P.masscenter) == N.x

    # Both joint pos id defined but different axes
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_point=N.x, child_point=A.x,
             child_interframe=A.x + A.y)
    assert expand_mul(N.x.angle_between(A.x + A.y)) == 0  # Axis are aligned
    assert (A.x + A.y).express(N).simplify() == sqrt(2)*N.x
    assert simplify(A.dcm(N)) == Matrix([
        [sqrt(2)/2, -sqrt(2)*cos(q)/2, -sqrt(2)*sin(q)/2],
        [sqrt(2)/2, sqrt(2)*cos(q)/2, sqrt(2)*sin(q)/2],
        [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u*N.x
    assert (A.ang_vel_in(N).express(A).simplify() ==
            (u*A.x + u*A.y)/sqrt(2))
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    angle = A.ang_vel_in(N).angle_between(A.x + A.y)
    assert angle.xreplace({u: 1}) == 0
    assert C.masscenter.vel(N).simplify() == (u * A.z)/sqrt(2)
    assert C.masscenter.pos_from(P.masscenter) == N.x - A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            (1 - sqrt(2)/2)*N.x + sqrt(2)*cos(q)/2*N.y +
            sqrt(2)*sin(q)/2*N.z)
    assert (C.masscenter.vel(N).express(N).simplify() ==
            -sqrt(2)*u*sin(q)/2*N.y + sqrt(2)*u*cos(q)/2*N.z)
    assert C.masscenter.vel(N).angle_between(A.x) == pi/2

    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_point=N.x, child_point=A.x,
             child_interframe=A.x + A.y - A.z)
    assert expand_mul(N.x.angle_between(A.x + A.y - A.z)) == 0  # Axis aligned
    assert (A.x + A.y - A.z).express(N).simplify() == sqrt(3)*N.x
    assert simplify(A.dcm(N)) == Matrix([
        [sqrt(3)/3, -sqrt(6)*sin(q + pi/4)/3,
         sqrt(6)*cos(q + pi/4)/3],
        [sqrt(3)/3, sqrt(6)*cos(q + pi/12)/3,
         sqrt(6)*sin(q + pi/12)/3],
        [-sqrt(3)/3, sqrt(6)*cos(q + 5*pi/12)/3,
         sqrt(6)*sin(q + 5*pi/12)/3]])
    assert A.ang_vel_in(N) == u*N.x
    assert A.ang_vel_in(N).express(A).simplify() == (u*A.x + u*A.y -
                                                     u*A.z)/sqrt(3)
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    angle = A.ang_vel_in(N).angle_between(A.x + A.y-A.z)
    assert angle.xreplace({u: 1}).simplify() == 0
    assert C.masscenter.vel(N).simplify() == (u*A.y + u*A.z)/sqrt(3)
    assert C.masscenter.pos_from(P.masscenter) == N.x - A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            (1 - sqrt(3)/3)*N.x + sqrt(6)*sin(q + pi/4)/3*N.y -
            sqrt(6)*cos(q + pi/4)/3*N.z)
    assert (C.masscenter.vel(N).express(N).simplify() ==
            sqrt(6)*u*cos(q + pi/4)/3*N.y +
            sqrt(6)*u*sin(q + pi/4)/3*N.z)
    assert C.masscenter.vel(N).angle_between(A.x) == pi/2

    N, A, P, C = _generate_body()
    m, n = symbols('m n')
    PinJoint('J', P, C, parent_point=m * N.x, child_point=n * A.x,
             child_interframe=A.x + A.y - A.z,
             parent_interframe=N.x - N.y + N.z)
    angle = (N.x - N.y + N.z).angle_between(A.x + A.y - A.z)
    assert expand_mul(angle) == 0  # Axis are aligned
    assert ((A.x-A.y+A.z).express(N).simplify() ==
            (-4*cos(q)/3 - S(1)/3)*N.x + (S(1)/3 - 4*sin(q + pi/6)/3)*N.y +
            (4*cos(q + pi/3)/3 - S(1)/3)*N.z)
    assert simplify(A.dcm(N)) == Matrix([
        [S(1)/3 - 2*cos(q)/3, -2*sin(q + pi/6)/3 - S(1)/3,
         2*cos(q + pi/3)/3 + S(1)/3],
        [2*cos(q + pi/3)/3 + S(1)/3, 2*cos(q)/3 - S(1)/3,
         2*sin(q + pi/6)/3 + S(1)/3],
        [-2*sin(q + pi/6)/3 - S(1)/3, 2*cos(q + pi/3)/3 + S(1)/3,
         2*cos(q)/3 - S(1)/3]])
    assert (A.ang_vel_in(N) - (u*N.x - u*N.y + u*N.z)/sqrt(3)).simplify()
    assert A.ang_vel_in(N).express(A).simplify() == (u*A.x + u*A.y -
                                                     u*A.z)/sqrt(3)
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    angle = A.ang_vel_in(N).angle_between(A.x+A.y-A.z)
    assert angle.xreplace({u: 1}).simplify() == 0
    assert (C.masscenter.vel(N).simplify() ==
            sqrt(3)*n*u/3*A.y + sqrt(3)*n*u/3*A.z)
    assert C.masscenter.pos_from(P.masscenter) == m*N.x - n*A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            (m + n*(2*cos(q) - 1)/3)*N.x + n*(2*sin(q + pi/6) +
            1)/3*N.y - n*(2*cos(q + pi/3) + 1)/3*N.z)
    assert (C.masscenter.vel(N).express(N).simplify() ==
            - 2*n*u*sin(q)/3*N.x + 2*n*u*cos(q + pi/6)/3*N.y +
            2*n*u*sin(q + pi/3)/3*N.z)
    assert C.masscenter.vel(N).dot(N.x - N.y + N.z).simplify() == 0


def test_create_aligned_frame_pi():
    N, A, P, C = _generate_body()
    f = Joint._create_aligned_interframe(P, -P.x, P.x)
    assert f.z == P.z
    f = Joint._create_aligned_interframe(P, -P.y, P.y)
    assert f.x == P.x
    f = Joint._create_aligned_interframe(P, -P.z, P.z)
    assert f.y == P.y
    f = Joint._create_aligned_interframe(P, -P.x - P.y, P.x + P.y)
    assert f.z == P.z
    f = Joint._create_aligned_interframe(P, -P.y - P.z, P.y + P.z)
    assert f.x == P.x
    f = Joint._create_aligned_interframe(P, -P.x - P.z, P.x + P.z)
    assert f.y == P.y
    f = Joint._create_aligned_interframe(P, -P.x - P.y - P.z, P.x + P.y + P.z)
    assert f.y - f.z == P.y - P.z


def test_pin_joint_axis():
    q, u = dynamicsymbols('q u')
    # Test default joint axis
    N, A, P, C, Pint, Cint = _generate_body(True)
    J = PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint)
    assert J.joint_axis == Pint.x
    # Test for the same joint axis expressed in different frames
    N_R_A = Matrix([[0, sin(q), cos(q)],
                    [0, -cos(q), sin(q)],
                    [1, 0, 0]])
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint,
             joint_axis=N.z)
    assert N.dcm(A) == N_R_A
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint,
             joint_axis=-Pint.z)
    assert N.dcm(A) == N_R_A
    # Test time varying joint axis
    N, A, P, C, Pint, Cint = _generate_body(True)
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=q * N.z))


def test_locate_joint_pos():
    # Test Vector and default
    N, A, P, C = _generate_body()
    joint = PinJoint('J', P, C, parent_point=N.y + N.z)
    assert joint.parent_point.name == 'J_P_joint'
    assert joint.parent_point.pos_from(P.masscenter) == N.y + N.z
    assert joint.child_point == C.masscenter
    # Test Point objects
    N, A, P, C = _generate_body()
    parent_point = P.masscenter.locatenew('p', N.y + N.z)
    joint = PinJoint('J', P, C, parent_point=parent_point,
                     child_point=C.masscenter)
    assert joint.parent_point == parent_point
    assert joint.child_point == C.masscenter
    # Check invalid type
    N, A, P, C = _generate_body()
    raises(TypeError,
           lambda: PinJoint('J', P, C, parent_point=N.x.to_matrix(N)))
    # Test time varying positions
    q = dynamicsymbols('q')
    N, A, P, C = _generate_body()
    raises(ValueError, lambda: PinJoint('J', P, C, parent_point=q * N.x))
    N, A, P, C = _generate_body()
    child_point = C.masscenter.locatenew('p', q * A.y)
    raises(ValueError, lambda: PinJoint('J', P, C, child_point=child_point))
    # Test undefined position
    child_point = Point('p')
    raises(ValueError, lambda: PinJoint('J', P, C, child_point=child_point))


def test_locate_joint_frame():
    # Test rotated frame and default
    N, A, P, C = _generate_body()
    parent_interframe = ReferenceFrame('int_frame')
    parent_interframe.orient_axis(N, N.z, 1)
    joint = PinJoint('J', P, C, parent_interframe=parent_interframe)
    assert joint.parent_interframe == parent_interframe
    assert joint.parent_interframe.ang_vel_in(N) == 0
    assert joint.child_interframe == A
    # Test time varying orientations
    q = dynamicsymbols('q')
    N, A, P, C = _generate_body()
    parent_interframe = ReferenceFrame('int_frame')
    parent_interframe.orient_axis(N, N.z, q)
    raises(ValueError,
           lambda: PinJoint('J', P, C, parent_interframe=parent_interframe))
    # Test undefined frame
    N, A, P, C = _generate_body()
    child_interframe = ReferenceFrame('int_frame')
    child_interframe.orient_axis(N, N.z, 1)  # Defined with respect to parent
    raises(ValueError,
           lambda: PinJoint('J', P, C, child_interframe=child_interframe))


def test_prismatic_joint():
    _, _, P, C = _generate_body()
    q, u = dynamicsymbols('q_S, u_S')
    S = PrismaticJoint('S', P, C)
    assert S.name == 'S'
    assert S.parent == P
    assert S.child == C
    assert S.coordinates == Matrix([q])
    assert S.speeds == Matrix([u])
    assert S.kdes == Matrix([u - q.diff(t)])
    assert S.joint_axis == P.frame.x
    assert S.child_point.pos_from(C.masscenter) == Vector(0)
    assert S.parent_point.pos_from(P.masscenter) == Vector(0)
    assert S.parent_point.pos_from(S.child_point) == - q * P.frame.x
    assert P.masscenter.pos_from(C.masscenter) == - q * P.frame.x
    assert C.masscenter.vel(P.frame) == u * P.frame.x
    assert P.frame.ang_vel_in(C.frame) == 0
    assert C.frame.ang_vel_in(P.frame) == 0
    assert S.__str__() == 'PrismaticJoint: S  parent: P  child: C'

    N, A, P, C = _generate_body()
    l, m = symbols('l m')
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P.frame, P.y, pi / 2)
    S = PrismaticJoint('S', P, C, parent_point=l * P.frame.x,
                       child_point=m * C.frame.y, joint_axis=P.frame.z,
                       parent_interframe=Pint)

    assert S.joint_axis == P.frame.z
    assert S.child_point.pos_from(C.masscenter) == m * C.frame.y
    assert S.parent_point.pos_from(P.masscenter) == l * P.frame.x
    assert S.parent_point.pos_from(S.child_point) == - q * P.frame.z
    assert P.masscenter.pos_from(C.masscenter) == - l * N.x - q * N.z + m * A.y
    assert C.masscenter.vel(P.frame) == u * P.frame.z
    assert P.masscenter.vel(Pint) == Vector(0)
    assert C.frame.ang_vel_in(P.frame) == 0
    assert P.frame.ang_vel_in(C.frame) == 0

    _, _, P, C = _generate_body()
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P.frame, P.y, pi / 2)
    S = PrismaticJoint('S', P, C, parent_point=l * P.frame.z,
                       child_point=m * C.frame.x, joint_axis=P.frame.z,
                       parent_interframe=Pint)
    assert S.joint_axis == P.frame.z
    assert S.child_point.pos_from(C.masscenter) == m * C.frame.x
    assert S.parent_point.pos_from(P.masscenter) == l * P.frame.z
    assert S.parent_point.pos_from(S.child_point) == - q * P.frame.z
    assert P.masscenter.pos_from(C.masscenter) == (-l - q)*P.frame.z + m*C.frame.x
    assert C.masscenter.vel(P.frame) == u * P.frame.z
    assert C.frame.ang_vel_in(P.frame) == 0
    assert P.frame.ang_vel_in(C.frame) == 0


def test_prismatic_joint_arbitrary_axis():
    q, u = dynamicsymbols('q_S, u_S')

    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, child_interframe=-A.x)

    assert (-A.x).angle_between(N.x) == 0
    assert -A.x.express(N) == N.x
    assert A.dcm(N) == Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert C.masscenter.pos_from(P.masscenter) == q * N.x
    assert C.masscenter.pos_from(P.masscenter).express(A).simplify() == -q * A.x
    assert C.masscenter.vel(N) == u * N.x
    assert C.masscenter.vel(N).express(A) == -u * A.x
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    #When axes are different and parent joint is at masscenter but child joint is at a unit vector from
    #child masscenter.
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, child_interframe=A.y, child_point=A.x)

    assert A.y.angle_between(N.x) == 0 #Axis are aligned
    assert A.y.express(N) == N.x
    assert A.dcm(N) == Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert C.masscenter.vel(N) == u * N.x
    assert C.masscenter.vel(N).express(A) == u * A.y
    assert C.masscenter.pos_from(P.masscenter) == q*N.x - A.x
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == q*N.x + N.y
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    #Similar to previous case but wrt parent body
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_interframe=N.y, parent_point=N.x)

    assert N.y.angle_between(A.x) == 0 #Axis are aligned
    assert N.y.express(A) ==  A.x
    assert A.dcm(N) == Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert C.masscenter.vel(N) == u * N.y
    assert C.masscenter.vel(N).express(A) == u * A.x
    assert C.masscenter.pos_from(P.masscenter) == N.x + q*N.y
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    #Both joint pos is defined but different axes
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_point=N.x, child_point=A.x,
                   child_interframe=A.x + A.y)
    assert N.x.angle_between(A.x + A.y) == 0 #Axis are aligned
    assert (A.x + A.y).express(N) == sqrt(2)*N.x
    assert A.dcm(N) == Matrix([[sqrt(2)/2, -sqrt(2)/2, 0], [sqrt(2)/2, sqrt(2)/2, 0], [0, 0, 1]])
    assert C.masscenter.pos_from(P.masscenter) == (q + 1)*N.x - A.x
    assert C.masscenter.pos_from(P.masscenter).express(N) == (q - sqrt(2)/2 + 1)*N.x + sqrt(2)/2*N.y
    assert C.masscenter.vel(N).express(A) == u * (A.x + A.y)/sqrt(2)
    assert C.masscenter.vel(N) == u*N.x
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_point=N.x, child_point=A.x,
                   child_interframe=A.x + A.y - A.z)
    assert N.x.angle_between(A.x + A.y - A.z).simplify() == 0 #Axis are aligned
    assert ((A.x + A.y - A.z).express(N) - sqrt(3)*N.x).simplify() == 0
    assert simplify(A.dcm(N)) == Matrix([[sqrt(3)/3, -sqrt(3)/3, sqrt(3)/3],
                                                 [sqrt(3)/3, sqrt(3)/6 + S(1)/2, S(1)/2 - sqrt(3)/6],
                                                 [-sqrt(3)/3, S(1)/2 - sqrt(3)/6, sqrt(3)/6 + S(1)/2]])
    assert C.masscenter.pos_from(P.masscenter) == (q + 1)*N.x - A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N) -
        ((q - sqrt(3)/3 + 1)*N.x + sqrt(3)/3*N.y - sqrt(3)/3*N.z)).simplify() == 0
    assert C.masscenter.vel(N) == u*N.x
    assert (C.masscenter.vel(N).express(A) - (
        sqrt(3)*u/3*A.x + sqrt(3)*u/3*A.y - sqrt(3)*u/3*A.z)).simplify()
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    N, A, P, C = _generate_body()
    m, n = symbols('m n')
    PrismaticJoint('S', P, C, parent_point=m*N.x, child_point=n*A.x,
                   child_interframe=A.x + A.y - A.z,
                   parent_interframe=N.x - N.y + N.z)
    # 0 angle means that the axis are aligned
    assert (N.x-N.y+N.z).angle_between(A.x+A.y-A.z).simplify() == 0
    assert ((A.x+A.y-A.z).express(N) - (N.x - N.y + N.z)).simplify() == 0
    assert simplify(A.dcm(N)) == Matrix([[-S(1)/3, -S(2)/3, S(2)/3],
                                                 [S(2)/3, S(1)/3, S(2)/3],
                                                 [-S(2)/3, S(2)/3, S(1)/3]])
    assert (C.masscenter.pos_from(P.masscenter) - (
        (m + sqrt(3)*q/3)*N.x - sqrt(3)*q/3*N.y + sqrt(3)*q/3*N.z - n*A.x)
            ).express(N).simplify() == 0
    assert (C.masscenter.pos_from(P.masscenter).express(N) - (
        (m + n/3 + sqrt(3)*q/3)*N.x + (2*n/3 - sqrt(3)*q/3)*N.y +
        (-2*n/3 + sqrt(3)*q/3)*N.z)).simplify() == 0
    assert (C.masscenter.vel(N).express(N) - (
        sqrt(3)*u/3*N.x - sqrt(3)*u/3*N.y + sqrt(3)*u/3*N.z)).simplify() == 0
    assert (C.masscenter.vel(N).express(A) -
            (sqrt(3)*u/3*A.x + sqrt(3)*u/3*A.y - sqrt(3)*u/3*A.z)).simplify() == 0
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0


def test_cylindrical_joint():
    N, A, P, C = _generate_body()
    q0_def, q1_def, u0_def, u1_def = dynamicsymbols('q0:2_J, u0:2_J')
    Cj = CylindricalJoint('J', P, C)
    assert Cj.name == 'J'
    assert Cj.parent == P
    assert Cj.child == C
    assert Cj.coordinates == Matrix([q0_def, q1_def])
    assert Cj.speeds == Matrix([u0_def, u1_def])
    assert Cj.rotation_coordinate == q0_def
    assert Cj.translation_coordinate == q1_def
    assert Cj.rotation_speed == u0_def
    assert Cj.translation_speed == u1_def
    assert Cj.kdes == Matrix([u0_def - q0_def.diff(t), u1_def - q1_def.diff(t)])
    assert Cj.joint_axis == N.x
    assert Cj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(P.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(Cj._child_point) == -q1_def * N.x
    assert C.masscenter.pos_from(P.masscenter) == q1_def * N.x
    assert Cj.child_point.vel(N) == u1_def * N.x
    assert A.ang_vel_in(N) == u0_def * N.x
    assert Cj.parent_interframe == N
    assert Cj.child_interframe == A
    assert Cj.__str__() == 'CylindricalJoint: J  parent: P  child: C'

    q0, q1, u0, u1 = dynamicsymbols('q0:2, u0:2')
    l, m = symbols('l, m')
    N, A, P, C, Pint, Cint = _generate_body(True)
    Cj = CylindricalJoint('J', P, C, rotation_coordinate=q0, rotation_speed=u0,
                          translation_speed=u1, parent_point=m * N.x,
                          child_point=l * A.y, parent_interframe=Pint,
                          child_interframe=Cint, joint_axis=2 * N.z)
    assert Cj.coordinates == Matrix([q0, q1_def])
    assert Cj.speeds == Matrix([u0, u1])
    assert Cj.rotation_coordinate == q0
    assert Cj.translation_coordinate == q1_def
    assert Cj.rotation_speed == u0
    assert Cj.translation_speed == u1
    assert Cj.kdes == Matrix([u0 - q0.diff(t), u1 - q1_def.diff(t)])
    assert Cj.joint_axis == 2 * N.z
    assert Cj.child_point.pos_from(C.masscenter) == l * A.y
    assert Cj.parent_point.pos_from(P.masscenter) == m * N.x
    assert Cj.parent_point.pos_from(Cj._child_point) == -q1_def * N.z
    assert C.masscenter.pos_from(
        P.masscenter) == m * N.x + q1_def * N.z - l * A.y
    assert C.masscenter.vel(N) == u1 * N.z - u0 * l * A.z
    assert A.ang_vel_in(N) == u0 * N.z


def test_planar_joint():
    N, A, P, C = _generate_body()
    q0_def, q1_def, q2_def = dynamicsymbols('q0:3_J')
    u0_def, u1_def, u2_def = dynamicsymbols('u0:3_J')
    Cj = PlanarJoint('J', P, C)
    assert Cj.name == 'J'
    assert Cj.parent == P
    assert Cj.child == C
    assert Cj.coordinates == Matrix([q0_def, q1_def, q2_def])
    assert Cj.speeds == Matrix([u0_def, u1_def, u2_def])
    assert Cj.rotation_coordinate == q0_def
    assert Cj.planar_coordinates == Matrix([q1_def, q2_def])
    assert Cj.rotation_speed == u0_def
    assert Cj.planar_speeds == Matrix([u1_def, u2_def])
    assert Cj.kdes == Matrix([u0_def - q0_def.diff(t), u1_def - q1_def.diff(t),
                              u2_def - q2_def.diff(t)])
    assert Cj.rotation_axis == N.x
    assert Cj.planar_vectors == [N.y, N.z]
    assert Cj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(P.masscenter) == Vector(0)
    r_P_C = q1_def * N.y + q2_def * N.z
    assert Cj.parent_point.pos_from(Cj.child_point) == -r_P_C
    assert C.masscenter.pos_from(P.masscenter) == r_P_C
    assert Cj.child_point.vel(N) == u1_def * N.y + u2_def * N.z
    assert A.ang_vel_in(N) == u0_def * N.x
    assert Cj.parent_interframe == N
    assert Cj.child_interframe == A
    assert Cj.__str__() == 'PlanarJoint: J  parent: P  child: C'

    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    l, m = symbols('l, m')
    N, A, P, C, Pint, Cint = _generate_body(True)
    Cj = PlanarJoint('J', P, C, rotation_coordinate=q0,
                     planar_coordinates=[q1, q2], planar_speeds=[u1, u2],
                     parent_point=m * N.x, child_point=l * A.y,
                     parent_interframe=Pint, child_interframe=Cint)
    assert Cj.coordinates == Matrix([q0, q1, q2])
    assert Cj.speeds == Matrix([u0_def, u1, u2])
    assert Cj.rotation_coordinate == q0
    assert Cj.planar_coordinates == Matrix([q1, q2])
    assert Cj.rotation_speed == u0_def
    assert Cj.planar_speeds == Matrix([u1, u2])
    assert Cj.kdes == Matrix([u0_def - q0.diff(t), u1 - q1.diff(t),
                              u2 - q2.diff(t)])
    assert Cj.rotation_axis == Pint.x
    assert Cj.planar_vectors == [Pint.y, Pint.z]
    assert Cj.child_point.pos_from(C.masscenter) == l * A.y
    assert Cj.parent_point.pos_from(P.masscenter) == m * N.x
    assert Cj.parent_point.pos_from(Cj.child_point) == q1 * N.y + q2 * N.z
    assert C.masscenter.pos_from(
        P.masscenter) == m * N.x - q1 * N.y - q2 * N.z - l * A.y
    assert C.masscenter.vel(N) == -u1 * N.y - u2 * N.z + u0_def * l * A.x
    assert A.ang_vel_in(N) == u0_def * N.x


def test_planar_joint_advanced():
    # Tests whether someone is able to just specify two normals, which will form
    # the rotation axis seen from the parent and child body.
    # This specific example is a block on a slope, which has that same slope of
    # 30 degrees, so in the zero configuration the frames of the parent and
    # child are actually aligned.
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    l1, l2 = symbols('l1:3')
    N, A, P, C = _generate_body()
    J = PlanarJoint('J', P, C, q0, [q1, q2], u0, [u1, u2],
                    parent_point=l1 * N.z,
                    child_point=-l2 * C.z,
                    parent_interframe=N.z + N.y / sqrt(3),
                    child_interframe=A.z + A.y / sqrt(3))
    assert J.rotation_axis.express(N) == (N.z + N.y / sqrt(3)).normalize()
    assert J.rotation_axis.express(A) == (A.z + A.y / sqrt(3)).normalize()
    assert J.rotation_axis.angle_between(N.z) == pi / 6
    assert N.dcm(A).xreplace({q0: 0, q1: 0, q2: 0}) == eye(3)
    N_R_A = Matrix([
        [cos(q0), -sqrt(3) * sin(q0) / 2, sin(q0) / 2],
        [sqrt(3) * sin(q0) / 2, 3 * cos(q0) / 4 + 1 / 4,
         sqrt(3) * (1 - cos(q0)) / 4],
        [-sin(q0) / 2, sqrt(3) * (1 - cos(q0)) / 4, cos(q0) / 4 + 3 / 4]])
    # N.dcm(A) == N_R_A did not work
    assert simplify(N.dcm(A) - N_R_A) == zeros(3)


def test_spherical_joint():
    N, A, P, C = _generate_body()
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3_S, u0:3_S')
    S = SphericalJoint('S', P, C)
    assert S.name == 'S'
    assert S.parent == P
    assert S.child == C
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    assert S.kdes == Matrix([u0 - q0.diff(t), u1 - q1.diff(t), u2 - q2.diff(t)])
    assert S.child_point.pos_from(C.masscenter) == Vector(0)
    assert S.parent_point.pos_from(P.masscenter) == Vector(0)
    assert S.parent_point.pos_from(S.child_point) == Vector(0)
    assert P.masscenter.pos_from(C.masscenter) == Vector(0)
    assert C.masscenter.vel(N) == Vector(0)
    assert N.ang_vel_in(A) == (-u0 * cos(q1) * cos(q2) - u1 * sin(q2)) * A.x + (
            u0 * sin(q2) * cos(q1) - u1 * cos(q2)) * A.y + (
                   -u0 * sin(q1) - u2) * A.z
    assert A.ang_vel_in(N) == (u0 * cos(q1) * cos(q2) + u1 * sin(q2)) * A.x + (
            -u0 * sin(q2) * cos(q1) + u1 * cos(q2)) * A.y + (
                   u0 * sin(q1) + u2) * A.z
    assert S.__str__() == 'SphericalJoint: S  parent: P  child: C'
    assert S._rot_type == 'BODY'
    assert S._rot_order == 123
    assert S._amounts is None


def test_spherical_joint_speeds_as_derivative_terms():
    # This tests checks whether the system remains valid if the user chooses to
    # pass the derivative of the generalized coordinates as generalized speeds
    q0, q1, q2 = dynamicsymbols('q0:3')
    u0, u1, u2 = dynamicsymbols('q0:3', 1)
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2])
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    assert S.kdes == Matrix([0, 0, 0])
    assert N.ang_vel_in(A) == (-u0 * cos(q1) * cos(q2) - u1 * sin(q2)) * A.x + (
        u0 * sin(q2) * cos(q1) - u1 * cos(q2)) * A.y + (
               -u0 * sin(q1) - u2) * A.z


def test_spherical_joint_coords():
    q0s, q1s, q2s, u0s, u1s, u2s = dynamicsymbols('q0:3_S, u0:3_S')
    q0, q1, q2, q3, u0, u1, u2, u4 = dynamicsymbols('q0:4, u0:4')
    # Test coordinates as list
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, [q0, q1, q2], [u0, u1, u2])
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    # Test coordinates as Matrix
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, Matrix([q0, q1, q2]),
                       Matrix([u0, u1, u2]))
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    # Test too few generalized coordinates
    N, A, P, C = _generate_body()
    raises(ValueError,
           lambda: SphericalJoint('S', P, C, Matrix([q0, q1]), Matrix([u0])))
    # Test too many generalized coordinates
    raises(ValueError, lambda: SphericalJoint(
        'S', P, C, Matrix([q0, q1, q2, q3]), Matrix([u0, u1, u2])))
    raises(ValueError, lambda: SphericalJoint(
        'S', P, C, Matrix([q0, q1, q2]), Matrix([u0, u1, u2, u4])))


def test_spherical_joint_orient_body():
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    N_R_A = Matrix([
        [-sin(q1), -sin(q2) * cos(q1), cos(q1) * cos(q2)],
        [-sin(q0) * cos(q1), sin(q0) * sin(q1) * sin(q2) - cos(q0) * cos(q2),
         -sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0)],
        [cos(q0) * cos(q1), -sin(q0) * cos(q2) - sin(q1) * sin(q2) * cos(q0),
         -sin(q0) * sin(q2) + sin(q1) * cos(q0) * cos(q2)]])
    N_w_A = Matrix([[-u0 * sin(q1) - u2],
                    [-u0 * sin(q2) * cos(q1) + u1 * cos(q2)],
                    [u0 * cos(q1) * cos(q2) + u1 * sin(q2)]])
    N_v_Co = Matrix([
        [-sqrt(2) * (u0 * cos(q2 + pi / 4) * cos(q1) + u1 * sin(q2 + pi / 4))],
        [-u0 * sin(q1) - u2], [-u0 * sin(q1) - u2]])
    # Test default rot_type='BODY', rot_order=123
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.y, child_point=-A.y + A.z,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='body', rot_order=123)
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - N_R_A) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A) - N_w_A) == zeros(3, 1)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == N_v_Co
    # Test change of amounts
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.y, child_point=-A.y + A.z,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='BODY', amounts=(q1, q0, q2), rot_order=123)
    switch_order = lambda expr: expr.xreplace(
        {q0: q1, q1: q0, q2: q2, u0: u1, u1: u0, u2: u2})
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - switch_order(N_R_A)) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A) - switch_order(N_w_A)
                            ) == zeros(3, 1)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == switch_order(N_v_Co)
    # Test different rot_order
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.y, child_point=-A.y + A.z,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='BodY', rot_order='yxz')
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 'yxz'
    assert simplify(N.dcm(A) - Matrix([
        [-sin(q0) * cos(q1), sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0),
         sin(q0) * sin(q1) * sin(q2) + cos(q0) * cos(q2)],
        [-sin(q1), -cos(q1) * cos(q2), -sin(q2) * cos(q1)],
        [cos(q0) * cos(q1), -sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2),
         sin(q0) * cos(q2) - sin(q1) * sin(q2) * cos(q0)]])) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A) - Matrix([
        [u0 * sin(q1) - u2], [u0 * cos(q1) * cos(q2) - u1 * sin(q2)],
        [u0 * sin(q2) * cos(q1) + u1 * cos(q2)]])) == zeros(3, 1)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == Matrix([
        [-sqrt(2) * (u0 * sin(q2 + pi / 4) * cos(q1) + u1 * cos(q2 + pi / 4))],
        [u0 * sin(q1) - u2], [u0 * sin(q1) - u2]])


def test_spherical_joint_orient_space():
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    N_R_A = Matrix([
        [-sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2),
         sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0), cos(q1) * cos(q2)],
        [-sin(q0) * cos(q2) + sin(q1) * sin(q2) * cos(q0),
         -sin(q0) * sin(q1) * sin(q2) - cos(q0) * cos(q2), -sin(q2) * cos(q1)],
        [cos(q0) * cos(q1), -sin(q0) * cos(q1), sin(q1)]])
    N_w_A = Matrix([
        [u1 * sin(q0) - u2 * cos(q0) * cos(q1)],
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)], [u0 - u2 * sin(q1)]])
    N_v_Co = Matrix([
        [u0 - u2 * sin(q1)], [u0 - u2 * sin(q1)],
        [sqrt(2) * (-u1 * sin(q0 + pi / 4) + u2 * cos(q0 + pi / 4) * cos(q1))]])
    # Test default rot_type='BODY', rot_order=123
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.z, child_point=-A.x + A.y,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='space', rot_order=123)
    assert S._rot_type.upper() == 'SPACE'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - N_R_A) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A)) == N_w_A
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == N_v_Co
    # Test change of amounts
    switch_order = lambda expr: expr.xreplace(
        {q0: q1, q1: q0, q2: q2, u0: u1, u1: u0, u2: u2})
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.z, child_point=-A.x + A.y,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='SPACE', amounts=(q1, q0, q2), rot_order=123)
    assert S._rot_type.upper() == 'SPACE'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - switch_order(N_R_A)) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A)) == switch_order(N_w_A)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == switch_order(N_v_Co)
    # Test different rot_order
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.z, child_point=-A.x + A.y,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='SPaCe', rot_order='zxy')
    assert S._rot_type.upper() == 'SPACE'
    assert S._rot_order == 'zxy'
    assert simplify(N.dcm(A) - Matrix([
        [-sin(q2) * cos(q1), -sin(q0) * cos(q2) + sin(q1) * sin(q2) * cos(q0),
         sin(q0) * sin(q1) * sin(q2) + cos(q0) * cos(q2)],
        [-sin(q1), -cos(q0) * cos(q1), -sin(q0) * cos(q1)],
        [cos(q1) * cos(q2), -sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2),
         -sin(q0) * sin(q1) * cos(q2) + sin(q2) * cos(q0)]]))
    assert simplify(A.ang_vel_in(N).to_matrix(A) - Matrix([
        [-u0 + u2 * sin(q1)], [-u1 * sin(q0) + u2 * cos(q0) * cos(q1)],
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)]])) == zeros(3, 1)
    assert simplify(C.masscenter.vel(N).to_matrix(A) - Matrix([
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)],
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)],
        [u0 + u1 * sin(q0) - u2 * sin(q1) -
         u2 * cos(q0) * cos(q1)]])) == zeros(3, 1)


def test_weld_joint():
    _, _, P, C = _generate_body()
    W = WeldJoint('W', P, C)
    assert W.name == 'W'
    assert W.parent == P
    assert W.child == C
    assert W.coordinates == Matrix()
    assert W.speeds == Matrix()
    assert W.kdes == Matrix(1, 0, []).T
    assert P.frame.dcm(C.frame) == eye(3)
    assert W.child_point.pos_from(C.masscenter) == Vector(0)
    assert W.parent_point.pos_from(P.masscenter) == Vector(0)
    assert W.parent_point.pos_from(W.child_point) == Vector(0)
    assert P.masscenter.pos_from(C.masscenter) == Vector(0)
    assert C.masscenter.vel(P.frame) == Vector(0)
    assert P.frame.ang_vel_in(C.frame) == 0
    assert C.frame.ang_vel_in(P.frame) == 0
    assert W.__str__() == 'WeldJoint: W  parent: P  child: C'

    N, A, P, C = _generate_body()
    l, m = symbols('l m')
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P.frame, P.y, pi / 2)
    W = WeldJoint('W', P, C, parent_point=l * P.frame.x,
                  child_point=m * C.frame.y, parent_interframe=Pint)

    assert W.child_point.pos_from(C.masscenter) == m * C.frame.y
    assert W.parent_point.pos_from(P.masscenter) == l * P.frame.x
    assert W.parent_point.pos_from(W.child_point) == Vector(0)
    assert P.masscenter.pos_from(C.masscenter) == - l * N.x + m * A.y
    assert C.masscenter.vel(P.frame) == Vector(0)
    assert P.masscenter.vel(Pint) == Vector(0)
    assert C.frame.ang_vel_in(P.frame) == 0
    assert P.frame.ang_vel_in(C.frame) == 0
    assert P.x == A.z

    with warns_deprecated_sympy():
        JointsMethod(P, W)  # Tests #10770


def test_deprecated_parent_child_axis():
    q, u = dynamicsymbols('q_J, u_J')
    N, A, P, C = _generate_body()
    with warns_deprecated_sympy():
        PinJoint('J', P, C, child_axis=-A.x)
    assert (-A.x).angle_between(N.x) == 0
    assert -A.x.express(N) == N.x
    assert A.dcm(N) == Matrix([[-1, 0, 0],
                               [0, -cos(q), -sin(q)],
                               [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u * N.x
    assert A.ang_vel_in(N).magnitude() == sqrt(u ** 2)

    N, A, P, C = _generate_body()
    with warns_deprecated_sympy():
        PrismaticJoint('J', P, C, parent_axis=P.x + P.y)
    assert (A.x).angle_between(N.x + N.y) == 0
    assert A.x.express(N) == (N.x + N.y) / sqrt(2)
    assert A.dcm(N) == Matrix([[sqrt(2) / 2, sqrt(2) / 2, 0],
                               [-sqrt(2) / 2, sqrt(2) / 2, 0], [0, 0, 1]])
    assert A.ang_vel_in(N) == Vector(0)


def test_deprecated_joint_pos():
    N, A, P, C = _generate_body()
    with warns_deprecated_sympy():
        pin = PinJoint('J', P, C, parent_joint_pos=N.x + N.y,
                       child_joint_pos=C.y - C.z)
    assert pin.parent_point.pos_from(P.masscenter) == N.x + N.y
    assert pin.child_point.pos_from(C.masscenter) == C.y - C.z

    N, A, P, C = _generate_body()
    with warns_deprecated_sympy():
        slider = PrismaticJoint('J', P, C, parent_joint_pos=N.z + N.y,
                                child_joint_pos=C.y - C.x)
    assert slider.parent_point.pos_from(P.masscenter) == N.z + N.y
    assert slider.child_point.pos_from(C.masscenter) == C.y - C.x
