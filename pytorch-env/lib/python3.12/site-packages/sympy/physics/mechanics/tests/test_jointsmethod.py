from sympy.core.function import expand
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.mechanics import (
    PinJoint, JointsMethod, RigidBody, Particle, Body, KanesMethod,
    PrismaticJoint, LagrangesMethod, inertia)
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy import zeros
from sympy.utilities.lambdify import lambdify
from sympy.solvers.solvers import solve


t = dynamicsymbols._t # type: ignore


def test_jointsmethod():
    with warns_deprecated_sympy():
        P = Body('P')
        C = Body('C')
    Pin = PinJoint('P1', P, C)
    C_ixx, g = symbols('C_ixx g')
    q, u = dynamicsymbols('q_P1, u_P1')
    P.apply_force(g*P.y)
    with warns_deprecated_sympy():
        method = JointsMethod(P, Pin)
    assert method.frame == P.frame
    assert method.bodies == [C, P]
    assert method.loads == [(P.masscenter, g*P.frame.y)]
    assert method.q == Matrix([q])
    assert method.u == Matrix([u])
    assert method.kdes == Matrix([u - q.diff()])
    soln = method.form_eoms()
    assert soln == Matrix([[-C_ixx*u.diff()]])
    assert method.forcing_full == Matrix([[u], [0]])
    assert method.mass_matrix_full == Matrix([[1, 0], [0, C_ixx]])
    assert isinstance(method.method, KanesMethod)


def test_rigid_body_particle_compatibility():
    l, m, g = symbols('l m g')
    C = RigidBody('C')
    b = Particle('b', mass=m)
    b_frame = ReferenceFrame('b_frame')
    q, u = dynamicsymbols('q u')
    P = PinJoint('P', C, b, coordinates=q, speeds=u, child_interframe=b_frame,
                 child_point=-l * b_frame.x, joint_axis=C.z)
    with warns_deprecated_sympy():
        method = JointsMethod(C, P)
    method.loads.append((b.masscenter, m * g * C.x))
    method.form_eoms()
    rhs = method.rhs()
    assert rhs[1] == -g*sin(q)/l


def test_jointmethod_duplicate_coordinates_speeds():
    with warns_deprecated_sympy():
        P = Body('P')
        C = Body('C')
        T = Body('T')
    q, u = dynamicsymbols('q u')
    P1 = PinJoint('P1', P, C, q)
    P2 = PrismaticJoint('P2', C, T, q)
    with warns_deprecated_sympy():
        raises(ValueError, lambda: JointsMethod(P, P1, P2))

    P1 = PinJoint('P1', P, C, speeds=u)
    P2 = PrismaticJoint('P2', C, T, speeds=u)
    with warns_deprecated_sympy():
        raises(ValueError, lambda: JointsMethod(P, P1, P2))

    P1 = PinJoint('P1', P, C, q, u)
    P2 = PrismaticJoint('P2', C, T, q, u)
    with warns_deprecated_sympy():
        raises(ValueError, lambda: JointsMethod(P, P1, P2))

def test_complete_simple_double_pendulum():
    q1, q2 = dynamicsymbols('q1 q2')
    u1, u2 = dynamicsymbols('u1 u2')
    m, l, g = symbols('m l g')
    with warns_deprecated_sympy():
        C = Body('C')  # ceiling
        PartP = Body('P', mass=m)
        PartR = Body('R', mass=m)
    J1 = PinJoint('J1', C, PartP, speeds=u1, coordinates=q1,
                  child_point=-l*PartP.x, joint_axis=C.z)
    J2 = PinJoint('J2', PartP, PartR, speeds=u2, coordinates=q2,
                  child_point=-l*PartR.x, joint_axis=PartP.z)

    PartP.apply_force(m*g*C.x)
    PartR.apply_force(m*g*C.x)

    with warns_deprecated_sympy():
        method = JointsMethod(C, J1, J2)
    method.form_eoms()

    assert expand(method.mass_matrix_full) == Matrix([[1, 0, 0, 0],
                                                      [0, 1, 0, 0],
                                                      [0, 0, 2*l**2*m*cos(q2) + 3*l**2*m, l**2*m*cos(q2) + l**2*m],
                                                      [0, 0, l**2*m*cos(q2) + l**2*m, l**2*m]])
    assert trigsimp(method.forcing_full) == trigsimp(Matrix([[u1], [u2], [-g*l*m*(sin(q1 + q2) + sin(q1)) -
                                           g*l*m*sin(q1) + l**2*m*(2*u1 + u2)*u2*sin(q2)],
                                          [-g*l*m*sin(q1 + q2) - l**2*m*u1**2*sin(q2)]]))

def test_two_dof_joints():
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    m, c1, c2, k1, k2 = symbols('m c1 c2 k1 k2')
    with warns_deprecated_sympy():
        W = Body('W')
        B1 = Body('B1', mass=m)
        B2 = Body('B2', mass=m)
    J1 = PrismaticJoint('J1', W, B1, coordinates=q1, speeds=u1)
    J2 = PrismaticJoint('J2', B1, B2, coordinates=q2, speeds=u2)
    W.apply_force(k1*q1*W.x, reaction_body=B1)
    W.apply_force(c1*u1*W.x, reaction_body=B1)
    B1.apply_force(k2*q2*W.x, reaction_body=B2)
    B1.apply_force(c2*u2*W.x, reaction_body=B2)
    with warns_deprecated_sympy():
        method = JointsMethod(W, J1, J2)
    method.form_eoms()
    MM = method.mass_matrix
    forcing = method.forcing
    rhs = MM.LUsolve(forcing)
    assert expand(rhs[0]) == expand((-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2)/m)
    assert expand(rhs[1]) == expand((k1 * q1 + c1 * u1 - 2 * k2 * q2 - 2 *
                                    c2 * u2) / m)

def test_simple_pedulum():
    l, m, g = symbols('l m g')
    with warns_deprecated_sympy():
        C = Body('C')
        b = Body('b', mass=m)
    q = dynamicsymbols('q')
    P = PinJoint('P', C, b, speeds=q.diff(t), coordinates=q,
                 child_point=-l * b.x, joint_axis=C.z)
    b.potential_energy = - m * g * l * cos(q)
    with warns_deprecated_sympy():
        method = JointsMethod(C, P)
    method.form_eoms(LagrangesMethod)
    rhs = method.rhs()
    assert rhs[1] == -g*sin(q)/l

def test_chaos_pendulum():
    #https://www.pydy.org/examples/chaos_pendulum.html
    mA, mB, lA, lB, IAxx, IBxx, IByy, IBzz, g = symbols('mA, mB, lA, lB, IAxx, IBxx, IByy, IBzz, g')
    theta, phi, omega, alpha = dynamicsymbols('theta phi omega alpha')

    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    with warns_deprecated_sympy():
        rod = Body('rod', mass=mA, frame=A,
                   central_inertia=inertia(A, IAxx, IAxx, 0))
        plate = Body('plate', mass=mB, frame=B,
                     central_inertia=inertia(B, IBxx, IByy, IBzz))
        C = Body('C')
    J1 = PinJoint('J1', C, rod, coordinates=theta, speeds=omega,
                  child_point=-lA * rod.z, joint_axis=C.y)
    J2 = PinJoint('J2', rod, plate, coordinates=phi, speeds=alpha,
                  parent_point=(lB - lA) * rod.z, joint_axis=rod.z)

    rod.apply_force(mA*g*C.z)
    plate.apply_force(mB*g*C.z)

    with warns_deprecated_sympy():
        method = JointsMethod(C, J1, J2)
    method.form_eoms()

    MM = method.mass_matrix
    forcing = method.forcing
    rhs = MM.LUsolve(forcing)
    xd = (-2 * IBxx * alpha * omega * sin(phi) * cos(phi) + 2 * IByy * alpha * omega * sin(phi) *
            cos(phi) - g * lA * mA * sin(theta) - g * lB * mB * sin(theta)) / (IAxx + IBxx *
                sin(phi)**2 + IByy * cos(phi)**2 + lA**2 * mA + lB**2 * mB)
    assert (rhs[0] - xd).simplify() == 0
    xd = (IBxx - IByy) * omega**2 * sin(phi) * cos(phi) / IBzz
    assert (rhs[1] - xd).simplify() == 0

def test_four_bar_linkage_with_manual_constraints():
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1:4, u1:4')
    l1, l2, l3, l4, rho = symbols('l1:5, rho')

    N = ReferenceFrame('N')
    inertias = [inertia(N, 0, 0, rho * l ** 3 / 12) for l in (l1, l2, l3, l4)]
    with warns_deprecated_sympy():
        link1 = Body('Link1', frame=N, mass=rho * l1,
                     central_inertia=inertias[0])
        link2 = Body('Link2', mass=rho * l2, central_inertia=inertias[1])
        link3 = Body('Link3', mass=rho * l3, central_inertia=inertias[2])
        link4 = Body('Link4', mass=rho * l4, central_inertia=inertias[3])

    joint1 = PinJoint(
        'J1', link1, link2, coordinates=q1, speeds=u1, joint_axis=link1.z,
        parent_point=l1 / 2 * link1.x, child_point=-l2 / 2 * link2.x)
    joint2 = PinJoint(
        'J2', link2, link3, coordinates=q2, speeds=u2, joint_axis=link2.z,
        parent_point=l2 / 2 * link2.x, child_point=-l3 / 2 * link3.x)
    joint3 = PinJoint(
        'J3', link3, link4, coordinates=q3, speeds=u3, joint_axis=link3.z,
        parent_point=l3 / 2 * link3.x, child_point=-l4 / 2 * link4.x)

    loop = link4.masscenter.pos_from(link1.masscenter) \
           + l1 / 2 * link1.x + l4 / 2 * link4.x

    fh = Matrix([loop.dot(link1.x), loop.dot(link1.y)])

    with warns_deprecated_sympy():
        method = JointsMethod(link1, joint1, joint2, joint3)

    t = dynamicsymbols._t
    qdots = solve(method.kdes, [q1.diff(t), q2.diff(t), q3.diff(t)])
    fhd = fh.diff(t).subs(qdots)

    kane = KanesMethod(method.frame, q_ind=[q1], u_ind=[u1],
                       q_dependent=[q2, q3], u_dependent=[u2, u3],
                       kd_eqs=method.kdes, configuration_constraints=fh,
                       velocity_constraints=fhd, forcelist=method.loads,
                       bodies=method.bodies)
    fr, frs = kane.kanes_equations()
    assert fr == zeros(1)

    # Numerically check the mass- and forcing-matrix
    p = Matrix([l1, l2, l3, l4, rho])
    q = Matrix([q1, q2, q3])
    u = Matrix([u1, u2, u3])
    eval_m = lambdify((q, p), kane.mass_matrix)
    eval_f = lambdify((q, u, p), kane.forcing)
    eval_fhd = lambdify((q, u, p), fhd)

    p_vals = [0.13, 0.24, 0.21, 0.34, 997]
    q_vals = [2.1, 0.6655470375077588, 2.527408138024188]  # Satisfies fh
    u_vals = [0.2, -0.17963733938852067, 0.1309060540601612]  # Satisfies fhd
    mass_check = Matrix([[3.452709815256506e+01, 7.003948798374735e+00,
                          -4.939690970641498e+00],
                         [-2.203792703880936e-14, 2.071702479957077e-01,
                          2.842917573033711e-01],
                         [-1.300000000000123e-01, -8.836934896046506e-03,
                          1.864891330060847e-01]])
    forcing_check = Matrix([[-0.031211821321648],
                            [-0.00066022608181],
                            [0.001813559741243]])
    eps = 1e-10
    assert all(abs(x) < eps for x in eval_fhd(q_vals, u_vals, p_vals))
    assert all(abs(x) < eps for x in
               (Matrix(eval_m(q_vals, p_vals)) - mass_check))
    assert all(abs(x) < eps for x in
               (Matrix(eval_f(q_vals, u_vals, p_vals)) - forcing_check))
