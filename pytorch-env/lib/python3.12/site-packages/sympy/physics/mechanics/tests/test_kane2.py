from sympy import cos, Matrix, sin, zeros, tan, pi, symbols
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import (cross, dot, dynamicsymbols,
                                     find_dynamicsymbols, KanesMethod, inertia,
                                     inertia_of_point_mass, Point,
                                     ReferenceFrame, RigidBody)


def test_aux_dep():
    # This test is about rolling disc dynamics, comparing the results found
    # with KanesMethod to those found when deriving the equations "manually"
    # with SymPy.
    # The terms Fr, Fr*, and Fr*_steady are all compared between the two
    # methods. Here, Fr*_steady refers to the generalized inertia forces for an
    # equilibrium configuration.
    # Note: comparing to the test of test_rolling_disc() in test_kane.py, this
    # test also tests auxiliary speeds and configuration and motion constraints
    #, seen in  the generalized dependent coordinates q[3], and depend speeds
    # u[3], u[4] and u[5].


    # First, manual derivation of Fr, Fr_star, Fr_star_steady.

    # Symbols for time and constant parameters.
    # Symbols for contact forces: Fx, Fy, Fz.
    t, r, m, g, I, J = symbols('t r m g I J')
    Fx, Fy, Fz = symbols('Fx Fy Fz')

    # Configuration variables and their time derivatives:
    # q[0] -- yaw
    # q[1] -- lean
    # q[2] -- spin
    # q[3] -- dot(-r*B.z, A.z) -- distance from ground plane to disc center in
    #         A.z direction
    # Generalized speeds and their time derivatives:
    # u[0] -- disc angular velocity component, disc fixed x direction
    # u[1] -- disc angular velocity component, disc fixed y direction
    # u[2] -- disc angular velocity component, disc fixed z direction
    # u[3] -- disc velocity component, A.x direction
    # u[4] -- disc velocity component, A.y direction
    # u[5] -- disc velocity component, A.z direction
    # Auxiliary generalized speeds:
    # ua[0] -- contact point auxiliary generalized speed, A.x direction
    # ua[1] -- contact point auxiliary generalized speed, A.y direction
    # ua[2] -- contact point auxiliary generalized speed, A.z direction
    q = dynamicsymbols('q:4')
    qd = [qi.diff(t) for qi in q]
    u = dynamicsymbols('u:6')
    ud = [ui.diff(t) for ui in u]
    ud_zero = dict(zip(ud, [0.]*len(ud)))
    ua = dynamicsymbols('ua:3')
    ua_zero = dict(zip(ua, [0.]*len(ua))) # noqa:F841

    # Reference frames:
    # Yaw intermediate frame: A.
    # Lean intermediate frame: B.
    # Disc fixed frame: C.
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q[0], N.z])
    B = A.orientnew('B', 'Axis', [q[1], A.x])
    C = B.orientnew('C', 'Axis', [q[2], B.y])

    # Angular velocity and angular acceleration of disc fixed frame
    # u[0], u[1] and u[2] are generalized independent speeds.
    C.set_ang_vel(N, u[0]*B.x + u[1]*B.y + u[2]*B.z)
    C.set_ang_acc(N, C.ang_vel_in(N).diff(t, B)
                   + cross(B.ang_vel_in(N), C.ang_vel_in(N)))

    # Velocity and acceleration of points:
    # Disc-ground contact point: P.
    # Center of disc: O, defined from point P with depend coordinate: q[3]
    # u[3], u[4] and u[5] are generalized dependent speeds.
    P = Point('P')
    P.set_vel(N, ua[0]*A.x + ua[1]*A.y + ua[2]*A.z)
    O = P.locatenew('O', q[3]*A.z + r*sin(q[1])*A.y)
    O.set_vel(N, u[3]*A.x + u[4]*A.y + u[5]*A.z)
    O.set_acc(N, O.vel(N).diff(t, A) + cross(A.ang_vel_in(N), O.vel(N)))

    # Kinematic differential equations:
    # Two equalities: one is w_c_n_qd = C.ang_vel_in(N) in three coordinates
    #                 directions of B, for qd0, qd1 and qd2.
    #                 the other is v_o_n_qd = O.vel(N) in A.z direction for qd3.
    # Then, solve for dq/dt's in terms of u's: qd_kd.
    w_c_n_qd = qd[0]*A.z + qd[1]*B.x + qd[2]*B.y
    v_o_n_qd = O.pos_from(P).diff(t, A) + cross(A.ang_vel_in(N), O.pos_from(P))
    kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] +
                      [dot(v_o_n_qd - O.vel(N), A.z)])
    qd_kd = solve(kindiffs, qd) # noqa:F841

    # Values of generalized speeds during a steady turn for later substitution
    # into the Fr_star_steady.
    steady_conditions = solve(kindiffs.subs({qd[1] : 0, qd[3] : 0}), u)
    steady_conditions.update({qd[1] : 0, qd[3] : 0})

    # Partial angular velocities and velocities.
    partial_w_C = [C.ang_vel_in(N).diff(ui, N) for ui in u + ua]
    partial_v_O = [O.vel(N).diff(ui, N) for ui in u + ua]
    partial_v_P = [P.vel(N).diff(ui, N) for ui in u + ua]

    # Configuration constraint: f_c, the projection of radius r in A.z direction
    #                                is q[3].
    # Velocity constraints: f_v, for u3, u4 and u5.
    # Acceleration constraints: f_a.
    f_c = Matrix([dot(-r*B.z, A.z) - q[3]])
    f_v = Matrix([dot(O.vel(N) - (P.vel(N) + cross(C.ang_vel_in(N),
        O.pos_from(P))), ai).expand() for ai in A])
    v_o_n = cross(C.ang_vel_in(N), O.pos_from(P))
    a_o_n = v_o_n.diff(t, A) + cross(A.ang_vel_in(N), v_o_n)
    f_a = Matrix([dot(O.acc(N) - a_o_n, ai) for ai in A]) # noqa:F841

    # Solve for constraint equations in the form of
    #                           u_dependent = A_rs * [u_i; u_aux].
    # First, obtain constraint coefficient matrix:  M_v * [u; ua] = 0;
    # Second, taking u[0], u[1], u[2] as independent,
    #         taking u[3], u[4], u[5] as dependent,
    #         rearranging the matrix of M_v to be A_rs for u_dependent.
    # Third, u_aux ==0 for u_dep, and resulting dictionary of u_dep_dict.
    M_v = zeros(3, 9)
    for i in range(3):
        for j, ui in enumerate(u + ua):
            M_v[i, j] = f_v[i].diff(ui)

    M_v_i = M_v[:, :3]
    M_v_d = M_v[:, 3:6]
    M_v_aux = M_v[:, 6:]
    M_v_i_aux = M_v_i.row_join(M_v_aux)
    A_rs = - M_v_d.inv() * M_v_i_aux

    u_dep = A_rs[:, :3] * Matrix(u[:3])
    u_dep_dict = dict(zip(u[3:], u_dep))

    # Active forces: F_O acting on point O; F_P acting on point P.
    # Generalized active forces (unconstrained): Fr_u = F_point * pv_point.
    F_O = m*g*A.z
    F_P = Fx * A.x + Fy * A.y + Fz * A.z
    Fr_u = Matrix([dot(F_O, pv_o) + dot(F_P, pv_p) for pv_o, pv_p in
            zip(partial_v_O, partial_v_P)])

    # Inertia force: R_star_O.
    # Inertia of disc: I_C_O, where J is a inertia component about principal axis.
    # Inertia torque: T_star_C.
    # Generalized inertia forces (unconstrained): Fr_star_u.
    R_star_O = -m*O.acc(N)
    I_C_O = inertia(B, I, J, I)
    T_star_C = -(dot(I_C_O, C.ang_acc_in(N)) \
                 + cross(C.ang_vel_in(N), dot(I_C_O, C.ang_vel_in(N))))
    Fr_star_u = Matrix([dot(R_star_O, pv) + dot(T_star_C, pav) for pv, pav in
                        zip(partial_v_O, partial_w_C)])

    # Form nonholonomic Fr: Fr_c, and nonholonomic Fr_star: Fr_star_c.
    # Also, nonholonomic Fr_star in steady turning condition: Fr_star_steady.
    Fr_c = Fr_u[:3, :].col_join(Fr_u[6:, :]) + A_rs.T * Fr_u[3:6, :]
    Fr_star_c = Fr_star_u[:3, :].col_join(Fr_star_u[6:, :])\
                + A_rs.T * Fr_star_u[3:6, :]
    Fr_star_steady = Fr_star_c.subs(ud_zero).subs(u_dep_dict)\
            .subs(steady_conditions).subs({q[3]: -r*cos(q[1])}).expand()


    # Second, using KaneMethod in mechanics for fr, frstar and frstar_steady.

    # Rigid Bodies: disc, with inertia I_C_O.
    iner_tuple = (I_C_O, O)
    disc = RigidBody('disc', O, C, m, iner_tuple)
    bodyList = [disc]

    # Generalized forces: Gravity: F_o; Auxiliary forces: F_p.
    F_o = (O, F_O)
    F_p = (P, F_P)
    forceList = [F_o,  F_p]

    # KanesMethod.
    kane = KanesMethod(
        N, q_ind= q[:3], u_ind= u[:3], kd_eqs=kindiffs,
        q_dependent=q[3:], configuration_constraints = f_c,
        u_dependent=u[3:], velocity_constraints= f_v,
        u_auxiliary=ua
        )

    # fr, frstar, frstar_steady and kdd(kinematic differential equations).
    (fr, frstar)= kane.kanes_equations(bodyList, forceList)
    frstar_steady = frstar.subs(ud_zero).subs(u_dep_dict).subs(steady_conditions)\
                    .subs({q[3]: -r*cos(q[1])}).expand()
    kdd = kane.kindiffdict()

    assert Matrix(Fr_c).expand() == fr.expand()
    assert Matrix(Fr_star_c.subs(kdd)).expand() == frstar.expand()
    # These Matrices have some Integer(0) and some Float(0). Running under
    # SymEngine gives different types of zero.
    assert (simplify(Matrix(Fr_star_steady).expand()).xreplace({0:0.0}) ==
            simplify(frstar_steady.expand()).xreplace({0:0.0}))

    syms_in_forcing = find_dynamicsymbols(kane.forcing)
    for qdi in qd:
        assert qdi not in syms_in_forcing


def test_non_central_inertia():
    # This tests that the calculation of Fr* does not depend the point
    # about which the inertia of a rigid body is defined. This test solves
    # exercises 8.12, 8.17 from Kane 1985.

    # Declare symbols
    q1, q2, q3 = dynamicsymbols('q1:4')
    q1d, q2d, q3d = dynamicsymbols('q1:4', level=1)
    u1, u2, u3, u4, u5 = dynamicsymbols('u1:6')
    u_prime, R, M, g, e, f, theta = symbols('u\' R, M, g, e, f, theta')
    a, b, mA, mB, IA, J, K, t = symbols('a b mA mB IA J K t')
    Q1, Q2, Q3 = symbols('Q1, Q2 Q3')
    IA22, IA23, IA33 = symbols('IA22 IA23 IA33')

    # Reference Frames
    F = ReferenceFrame('F')
    P = F.orientnew('P', 'axis', [-theta, F.y])
    A = P.orientnew('A', 'axis', [q1, P.x])
    A.set_ang_vel(F, u1*A.x + u3*A.z)
    # define frames for wheels
    B = A.orientnew('B', 'axis', [q2, A.z])
    C = A.orientnew('C', 'axis', [q3, A.z])
    B.set_ang_vel(A, u4 * A.z)
    C.set_ang_vel(A, u5 * A.z)

    # define points D, S*, Q on frame A and their velocities
    pD = Point('D')
    pD.set_vel(A, 0)
    # u3 will not change v_D_F since wheels are still assumed to roll without slip.
    pD.set_vel(F, u2 * A.y)

    pS_star = pD.locatenew('S*', e*A.y)
    pQ = pD.locatenew('Q', f*A.y - R*A.x)
    for p in [pS_star, pQ]:
        p.v2pt_theory(pD, F, A)

    # masscenters of bodies A, B, C
    pA_star = pD.locatenew('A*', a*A.y)
    pB_star = pD.locatenew('B*', b*A.z)
    pC_star = pD.locatenew('C*', -b*A.z)
    for p in [pA_star, pB_star, pC_star]:
        p.v2pt_theory(pD, F, A)

    # points of B, C touching the plane P
    pB_hat = pB_star.locatenew('B^', -R*A.x)
    pC_hat = pC_star.locatenew('C^', -R*A.x)
    pB_hat.v2pt_theory(pB_star, F, B)
    pC_hat.v2pt_theory(pC_star, F, C)

    # the velocities of B^, C^ are zero since B, C are assumed to roll without slip
    kde = [q1d - u1, q2d - u4, q3d - u5]
    vc = [dot(p.vel(F), A.y) for p in [pB_hat, pC_hat]]

    # inertias of bodies A, B, C
    # IA22, IA23, IA33 are not specified in the problem statement, but are
    # necessary to define an inertia object. Although the values of
    # IA22, IA23, IA33 are not known in terms of the variables given in the
    # problem statement, they do not appear in the general inertia terms.
    inertia_A = inertia(A, IA, IA22, IA33, 0, IA23, 0)
    inertia_B = inertia(B, K, K, J)
    inertia_C = inertia(C, K, K, J)

    # define the rigid bodies A, B, C
    rbA = RigidBody('rbA', pA_star, A, mA, (inertia_A, pA_star))
    rbB = RigidBody('rbB', pB_star, B, mB, (inertia_B, pB_star))
    rbC = RigidBody('rbC', pC_star, C, mB, (inertia_C, pC_star))

    km = KanesMethod(F, q_ind=[q1, q2, q3], u_ind=[u1, u2], kd_eqs=kde,
                     u_dependent=[u4, u5], velocity_constraints=vc,
                     u_auxiliary=[u3])

    forces = [(pS_star, -M*g*F.x), (pQ, Q1*A.x + Q2*A.y + Q3*A.z)]
    bodies = [rbA, rbB, rbC]
    fr, fr_star = km.kanes_equations(bodies, forces)
    vc_map = solve(vc, [u4, u5])

    # KanesMethod returns the negative of Fr, Fr* as defined in Kane1985.
    fr_star_expected = Matrix([
            -(IA + 2*J*b**2/R**2 + 2*K +
              mA*a**2 + 2*mB*b**2) * u1.diff(t) - mA*a*u1*u2,
            -(mA + 2*mB +2*J/R**2) * u2.diff(t) + mA*a*u1**2,
            0])
    t = trigsimp(fr_star.subs(vc_map).subs({u3: 0})).doit().expand()
    assert ((fr_star_expected - t).expand() == zeros(3, 1))

    # define inertias of rigid bodies A, B, C about point D
    # I_S/O = I_S/S* + I_S*/O
    bodies2 = []
    for rb, I_star in zip([rbA, rbB, rbC], [inertia_A, inertia_B, inertia_C]):
        I = I_star + inertia_of_point_mass(rb.mass,
                                           rb.masscenter.pos_from(pD),
                                           rb.frame)
        bodies2.append(RigidBody('', rb.masscenter, rb.frame, rb.mass,
                                 (I, pD)))
    fr2, fr_star2 = km.kanes_equations(bodies2, forces)

    t = trigsimp(fr_star2.subs(vc_map).subs({u3: 0})).doit()
    assert (fr_star_expected - t).expand() == zeros(3, 1)

def test_sub_qdot():
    # This test solves exercises 8.12, 8.17 from Kane 1985 and defines
    # some velocities in terms of q, qdot.

    ## --- Declare symbols ---
    q1, q2, q3 = dynamicsymbols('q1:4')
    q1d, q2d, q3d = dynamicsymbols('q1:4', level=1)
    u1, u2, u3 = dynamicsymbols('u1:4')
    u_prime, R, M, g, e, f, theta = symbols('u\' R, M, g, e, f, theta')
    a, b, mA, mB, IA, J, K, t = symbols('a b mA mB IA J K t')
    IA22, IA23, IA33 = symbols('IA22 IA23 IA33')
    Q1, Q2, Q3 = symbols('Q1 Q2 Q3')

    # --- Reference Frames ---
    F = ReferenceFrame('F')
    P = F.orientnew('P', 'axis', [-theta, F.y])
    A = P.orientnew('A', 'axis', [q1, P.x])
    A.set_ang_vel(F, u1*A.x + u3*A.z)
    # define frames for wheels
    B = A.orientnew('B', 'axis', [q2, A.z])
    C = A.orientnew('C', 'axis', [q3, A.z])

    ## --- define points D, S*, Q on frame A and their velocities ---
    pD = Point('D')
    pD.set_vel(A, 0)
    # u3 will not change v_D_F since wheels are still assumed to roll w/o slip
    pD.set_vel(F, u2 * A.y)

    pS_star = pD.locatenew('S*', e*A.y)
    pQ = pD.locatenew('Q', f*A.y - R*A.x)
    # masscenters of bodies A, B, C
    pA_star = pD.locatenew('A*', a*A.y)
    pB_star = pD.locatenew('B*', b*A.z)
    pC_star = pD.locatenew('C*', -b*A.z)
    for p in [pS_star, pQ, pA_star, pB_star, pC_star]:
        p.v2pt_theory(pD, F, A)

    # points of B, C touching the plane P
    pB_hat = pB_star.locatenew('B^', -R*A.x)
    pC_hat = pC_star.locatenew('C^', -R*A.x)
    pB_hat.v2pt_theory(pB_star, F, B)
    pC_hat.v2pt_theory(pC_star, F, C)

    # --- relate qdot, u ---
    # the velocities of B^, C^ are zero since B, C are assumed to roll w/o slip
    kde = [dot(p.vel(F), A.y) for p in [pB_hat, pC_hat]]
    kde += [u1 - q1d]
    kde_map = solve(kde, [q1d, q2d, q3d])
    for k, v in list(kde_map.items()):
        kde_map[k.diff(t)] = v.diff(t)

    # inertias of bodies A, B, C
    # IA22, IA23, IA33 are not specified in the problem statement, but are
    # necessary to define an inertia object. Although the values of
    # IA22, IA23, IA33 are not known in terms of the variables given in the
    # problem statement, they do not appear in the general inertia terms.
    inertia_A = inertia(A, IA, IA22, IA33, 0, IA23, 0)
    inertia_B = inertia(B, K, K, J)
    inertia_C = inertia(C, K, K, J)

    # define the rigid bodies A, B, C
    rbA = RigidBody('rbA', pA_star, A, mA, (inertia_A, pA_star))
    rbB = RigidBody('rbB', pB_star, B, mB, (inertia_B, pB_star))
    rbC = RigidBody('rbC', pC_star, C, mB, (inertia_C, pC_star))

    ## --- use kanes method ---
    km = KanesMethod(F, [q1, q2, q3], [u1, u2], kd_eqs=kde, u_auxiliary=[u3])

    forces = [(pS_star, -M*g*F.x), (pQ, Q1*A.x + Q2*A.y + Q3*A.z)]
    bodies = [rbA, rbB, rbC]

    # Q2 = -u_prime * u2 * Q1 / sqrt(u2**2 + f**2 * u1**2)
    # -u_prime * R * u2 / sqrt(u2**2 + f**2 * u1**2) = R / Q1 * Q2
    fr_expected = Matrix([
            f*Q3 + M*g*e*sin(theta)*cos(q1),
            Q2 + M*g*sin(theta)*sin(q1),
            e*M*g*cos(theta) - Q1*f - Q2*R])
             #Q1 * (f - u_prime * R * u2 / sqrt(u2**2 + f**2 * u1**2)))])
    fr_star_expected = Matrix([
            -(IA + 2*J*b**2/R**2 + 2*K +
              mA*a**2 + 2*mB*b**2) * u1.diff(t) - mA*a*u1*u2,
            -(mA + 2*mB +2*J/R**2) * u2.diff(t) + mA*a*u1**2,
            0])

    fr, fr_star = km.kanes_equations(bodies, forces)
    assert (fr.expand() == fr_expected.expand())
    assert ((fr_star_expected - trigsimp(fr_star)).expand() == zeros(3, 1))

def test_sub_qdot2():
    # This test solves exercises 8.3 from Kane 1985 and defines
    # all velocities in terms of q, qdot. We check that the generalized active
    # forces are correctly computed if u terms are only defined in the
    # kinematic differential equations.
    #
    # This functionality was added in PR 8948. Without qdot/u substitution, the
    # KanesMethod constructor will fail during the constraint initialization as
    # the B matrix will be poorly formed and inversion of the dependent part
    # will fail.

    g, m, Px, Py, Pz, R, t = symbols('g m Px Py Pz R t')
    q = dynamicsymbols('q:5')
    qd = dynamicsymbols('q:5', level=1)
    u = dynamicsymbols('u:5')

    ## Define inertial, intermediate, and rigid body reference frames
    A = ReferenceFrame('A')
    B_prime = A.orientnew('B_prime', 'Axis', [q[0], A.z])
    B = B_prime.orientnew('B', 'Axis', [pi/2 - q[1], B_prime.x])
    C = B.orientnew('C', 'Axis', [q[2], B.z])

    ## Define points of interest and their velocities
    pO = Point('O')
    pO.set_vel(A, 0)

    # R is the point in plane H that comes into contact with disk C.
    pR = pO.locatenew('R', q[3]*A.x + q[4]*A.y)
    pR.set_vel(A, pR.pos_from(pO).diff(t, A))
    pR.set_vel(B, 0)

    # C^ is the point in disk C that comes into contact with plane H.
    pC_hat = pR.locatenew('C^', 0)
    pC_hat.set_vel(C, 0)

    # C* is the point at the center of disk C.
    pCs = pC_hat.locatenew('C*', R*B.y)
    pCs.set_vel(C, 0)
    pCs.set_vel(B, 0)

    # calculate velocites of points C* and C^ in frame A
    pCs.v2pt_theory(pR, A, B) # points C* and R are fixed in frame B
    pC_hat.v2pt_theory(pCs, A, C) # points C* and C^ are fixed in frame C

    ## Define forces on each point of the system
    R_C_hat = Px*A.x + Py*A.y + Pz*A.z
    R_Cs = -m*g*A.z
    forces = [(pC_hat, R_C_hat), (pCs, R_Cs)]

    ## Define kinematic differential equations
    # let ui = omega_C_A & bi (i = 1, 2, 3)
    # u4 = qd4, u5 = qd5
    u_expr = [C.ang_vel_in(A) & uv for uv in B]
    u_expr += qd[3:]
    kde = [ui - e for ui, e in zip(u, u_expr)]
    km1 = KanesMethod(A, q, u, kde)
    fr1, _ = km1.kanes_equations([], forces)

    ## Calculate generalized active forces if we impose the condition that the
    # disk C is rolling without slipping
    u_indep = u[:3]
    u_dep = list(set(u) - set(u_indep))
    vc = [pC_hat.vel(A) & uv for uv in [A.x, A.y]]
    km2 = KanesMethod(A, q, u_indep, kde,
                      u_dependent=u_dep, velocity_constraints=vc)
    fr2, _ = km2.kanes_equations([], forces)

    fr1_expected = Matrix([
        -R*g*m*sin(q[1]),
        -R*(Px*cos(q[0]) + Py*sin(q[0]))*tan(q[1]),
        R*(Px*cos(q[0]) + Py*sin(q[0])),
        Px,
        Py])
    fr2_expected = Matrix([
        -R*g*m*sin(q[1]),
        0,
        0])
    assert (trigsimp(fr1.expand()) == trigsimp(fr1_expected.expand()))
    assert (trigsimp(fr2.expand()) == trigsimp(fr2_expected.expand()))
