from sympy import symbols, Matrix, cos, sin, atan, sqrt, Rational
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point,\
    dot, cross, inertia, KanesMethod, Particle, RigidBody, Lagrangian,\
    LagrangesMethod
from sympy.testing.pytest import slow


@slow
def test_linearize_rolling_disc_kane():
    # Symbols for time and constant parameters
    t, r, m, g, v = symbols('t r m g v')

    # Configuration variables and their time derivatives
    q1, q2, q3, q4, q5, q6 = q = dynamicsymbols('q1:7')
    q1d, q2d, q3d, q4d, q5d, q6d = qd = [qi.diff(t) for qi in q]

    # Generalized speeds and their time derivatives
    u = dynamicsymbols('u:6')
    u1, u2, u3, u4, u5, u6 = u = dynamicsymbols('u1:7')
    u1d, u2d, u3d, u4d, u5d, u6d = [ui.diff(t) for ui in u]

    # Reference frames
    N = ReferenceFrame('N')                   # Inertial frame
    NO = Point('NO')                          # Inertial origin
    A = N.orientnew('A', 'Axis', [q1, N.z])   # Yaw intermediate frame
    B = A.orientnew('B', 'Axis', [q2, A.x])   # Lean intermediate frame
    C = B.orientnew('C', 'Axis', [q3, B.y])   # Disc fixed frame
    CO = NO.locatenew('CO', q4*N.x + q5*N.y + q6*N.z)      # Disc center

    # Disc angular velocity in N expressed using time derivatives of coordinates
    w_c_n_qd = C.ang_vel_in(N)
    w_b_n_qd = B.ang_vel_in(N)

    # Inertial angular velocity and angular acceleration of disc fixed frame
    C.set_ang_vel(N, u1*B.x + u2*B.y + u3*B.z)

    # Disc center velocity in N expressed using time derivatives of coordinates
    v_co_n_qd = CO.pos_from(NO).dt(N)

    # Disc center velocity in N expressed using generalized speeds
    CO.set_vel(N, u4*C.x + u5*C.y + u6*C.z)

    # Disc Ground Contact Point
    P = CO.locatenew('P', r*B.z)
    P.v2pt_theory(CO, N, C)

    # Configuration constraint
    f_c = Matrix([q6 - dot(CO.pos_from(P), N.z)])

    # Velocity level constraints
    f_v = Matrix([dot(P.vel(N), uv) for uv in C])

    # Kinematic differential equations
    kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] +
                        [dot(v_co_n_qd - CO.vel(N), uv) for uv in N])
    qdots = solve(kindiffs, qd)

    # Set angular velocity of remaining frames
    B.set_ang_vel(N, w_b_n_qd.subs(qdots))
    C.set_ang_acc(N, C.ang_vel_in(N).dt(B) + cross(B.ang_vel_in(N), C.ang_vel_in(N)))

    # Active forces
    F_CO = m*g*A.z

    # Create inertia dyadic of disc C about point CO
    I = (m * r**2) / 4
    J = (m * r**2) / 2
    I_C_CO = inertia(C, I, J, I)

    Disc = RigidBody('Disc', CO, C, m, (I_C_CO, CO))
    BL = [Disc]
    FL = [(CO, F_CO)]
    KM = KanesMethod(N, [q1, q2, q3, q4, q5], [u1, u2, u3], kd_eqs=kindiffs,
            q_dependent=[q6], configuration_constraints=f_c,
            u_dependent=[u4, u5, u6], velocity_constraints=f_v)
    (fr, fr_star) = KM.kanes_equations(BL, FL)

    # Test generalized form equations
    linearizer = KM.to_linearizer()
    assert linearizer.f_c == f_c
    assert linearizer.f_v == f_v
    assert linearizer.f_a == f_v.diff(t).subs(KM.kindiffdict())
    sol = solve(linearizer.f_0 + linearizer.f_1, qd)
    for qi in qdots.keys():
        assert sol[qi] == qdots[qi]
    assert simplify(linearizer.f_2 + linearizer.f_3 - fr - fr_star) == Matrix([0, 0, 0])

    # Perform the linearization
    # Precomputed operating point
    q_op = {q6: -r*cos(q2)}
    u_op = {u1: 0,
            u2: sin(q2)*q1d + q3d,
            u3: cos(q2)*q1d,
            u4: -r*(sin(q2)*q1d + q3d)*cos(q3),
            u5: 0,
            u6: -r*(sin(q2)*q1d + q3d)*sin(q3)}
    qd_op = {q2d: 0,
             q4d: -r*(sin(q2)*q1d + q3d)*cos(q1),
             q5d: -r*(sin(q2)*q1d + q3d)*sin(q1),
             q6d: 0}
    ud_op = {u1d: 4*g*sin(q2)/(5*r) + sin(2*q2)*q1d**2/2 + 6*cos(q2)*q1d*q3d/5,
             u2d: 0,
             u3d: 0,
             u4d: r*(sin(q2)*sin(q3)*q1d*q3d + sin(q3)*q3d**2),
             u5d: r*(4*g*sin(q2)/(5*r) + sin(2*q2)*q1d**2/2 + 6*cos(q2)*q1d*q3d/5),
             u6d: -r*(sin(q2)*cos(q3)*q1d*q3d + cos(q3)*q3d**2)}

    A, B = linearizer.linearize(op_point=[q_op, u_op, qd_op, ud_op], A_and_B=True, simplify=True)

    upright_nominal = {q1d: 0, q2: 0, m: 1, r: 1, g: 1}

    # Precomputed solution
    A_sol = Matrix([[0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [sin(q1)*q3d, 0, 0, 0, 0, -sin(q1), -cos(q1), 0],
                    [-cos(q1)*q3d, 0, 0, 0, 0, cos(q1), -sin(q1), 0],
                    [0, Rational(4, 5), 0, 0, 0, 0, 0, 6*q3d/5],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -2*q3d, 0, 0]])
    B_sol = Matrix([])

    # Check that linearization is correct
    assert A.subs(upright_nominal) == A_sol
    assert B.subs(upright_nominal) == B_sol

    # Check eigenvalues at critical speed are all zero:
    assert sympify(A.subs(upright_nominal).subs(q3d, 1/sqrt(3))).eigenvals() == {0: 8}

    # Check whether alternative solvers work
    # symengine doesn't support method='GJ'
    linearizer = KM.to_linearizer(linear_solver='GJ')
    A, B = linearizer.linearize(op_point=[q_op, u_op, qd_op, ud_op],
                                A_and_B=True, simplify=True)
    assert A.subs(upright_nominal) == A_sol
    assert B.subs(upright_nominal) == B_sol

def test_linearize_pendulum_kane_minimal():
    q1 = dynamicsymbols('q1')                     # angle of pendulum
    u1 = dynamicsymbols('u1')                     # Angular velocity
    q1d = dynamicsymbols('q1', 1)                 # Angular velocity
    L, m, t = symbols('L, m, t')
    g = 9.8

    # Compose world frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)

    # A.x is along the pendulum
    A = N.orientnew('A', 'axis', [q1, N.z])
    A.set_ang_vel(N, u1*N.z)

    # Locate point P relative to the origin N*
    P = pN.locatenew('P', L*A.x)
    P.v2pt_theory(pN, N, A)
    pP = Particle('pP', P, m)

    # Create Kinematic Differential Equations
    kde = Matrix([q1d - u1])

    # Input the force resultant at P
    R = m*g*N.x

    # Solve for eom with kanes method
    KM = KanesMethod(N, q_ind=[q1], u_ind=[u1], kd_eqs=kde)
    (fr, frstar) = KM.kanes_equations([pP], [(P, R)])

    # Linearize
    A, B, inp_vec = KM.linearize(A_and_B=True, simplify=True)

    assert A == Matrix([[0, 1], [-9.8*cos(q1)/L, 0]])
    assert B == Matrix([])

def test_linearize_pendulum_kane_nonminimal():
    # Create generalized coordinates and speeds for this non-minimal realization
    # q1, q2 = N.x and N.y coordinates of pendulum
    # u1, u2 = N.x and N.y velocities of pendulum
    q1, q2 = dynamicsymbols('q1:3')
    q1d, q2d = dynamicsymbols('q1:3', level=1)
    u1, u2 = dynamicsymbols('u1:3')
    u1d, u2d = dynamicsymbols('u1:3', level=1)
    L, m, t = symbols('L, m, t')
    g = 9.8

    # Compose world frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)

    # A.x is along the pendulum
    theta1 = atan(q2/q1)
    A = N.orientnew('A', 'axis', [theta1, N.z])

    # Locate the pendulum mass
    P = pN.locatenew('P1', q1*N.x + q2*N.y)
    pP = Particle('pP', P, m)

    # Calculate the kinematic differential equations
    kde = Matrix([q1d - u1,
                  q2d - u2])
    dq_dict = solve(kde, [q1d, q2d])

    # Set velocity of point P
    P.set_vel(N, P.pos_from(pN).dt(N).subs(dq_dict))

    # Configuration constraint is length of pendulum
    f_c = Matrix([P.pos_from(pN).magnitude() - L])

    # Velocity constraint is that the velocity in the A.x direction is
    # always zero (the pendulum is never getting longer).
    f_v = Matrix([P.vel(N).express(A).dot(A.x)])
    f_v.simplify()

    # Acceleration constraints is the time derivative of the velocity constraint
    f_a = f_v.diff(t)
    f_a.simplify()

    # Input the force resultant at P
    R = m*g*N.x

    # Derive the equations of motion using the KanesMethod class.
    KM = KanesMethod(N, q_ind=[q2], u_ind=[u2], q_dependent=[q1],
            u_dependent=[u1], configuration_constraints=f_c,
            velocity_constraints=f_v, acceleration_constraints=f_a, kd_eqs=kde)
    (fr, frstar) = KM.kanes_equations([pP], [(P, R)])

    # Set the operating point to be straight down, and non-moving
    q_op = {q1: L, q2: 0}
    u_op = {u1: 0, u2: 0}
    ud_op = {u1d: 0, u2d: 0}

    A, B, inp_vec = KM.linearize(op_point=[q_op, u_op, ud_op], A_and_B=True,
                                 simplify=True)

    assert A.expand() == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])


    # symengine doesn't support method='GJ'
    A, B, inp_vec = KM.linearize(op_point=[q_op, u_op, ud_op], A_and_B=True,
                                simplify=True, linear_solver='GJ')

    assert A.expand() == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])

    A, B, inp_vec = KM.linearize(op_point=[q_op, u_op, ud_op],
                                 A_and_B=True,
                                 simplify=True,
                                 linear_solver=lambda A, b: A.LUsolve(b))

    assert A.expand() == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])


def test_linearize_pendulum_lagrange_minimal():
    q1 = dynamicsymbols('q1')                     # angle of pendulum
    q1d = dynamicsymbols('q1', 1)                 # Angular velocity
    L, m, t = symbols('L, m, t')
    g = 9.8

    # Compose world frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)

    # A.x is along the pendulum
    A = N.orientnew('A', 'axis', [q1, N.z])
    A.set_ang_vel(N, q1d*N.z)

    # Locate point P relative to the origin N*
    P = pN.locatenew('P', L*A.x)
    P.v2pt_theory(pN, N, A)
    pP = Particle('pP', P, m)

    # Solve for eom with Lagranges method
    Lag = Lagrangian(N, pP)
    LM = LagrangesMethod(Lag, [q1], forcelist=[(P, m*g*N.x)], frame=N)
    LM.form_lagranges_equations()

    # Linearize
    A, B, inp_vec = LM.linearize([q1], [q1d], A_and_B=True)

    assert simplify(A) == Matrix([[0, 1], [-9.8*cos(q1)/L, 0]])
    assert B == Matrix([])

    # Check an alternative solver
    A, B, inp_vec = LM.linearize([q1], [q1d], A_and_B=True, linear_solver='GJ')

    assert simplify(A) == Matrix([[0, 1], [-9.8*cos(q1)/L, 0]])
    assert B == Matrix([])


def test_linearize_pendulum_lagrange_nonminimal():
    q1, q2 = dynamicsymbols('q1:3')
    q1d, q2d = dynamicsymbols('q1:3', level=1)
    L, m, t = symbols('L, m, t')
    g = 9.8
    # Compose World Frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)
    # A.x is along the pendulum
    theta1 = atan(q2/q1)
    A = N.orientnew('A', 'axis', [theta1, N.z])
    # Create point P, the pendulum mass
    P = pN.locatenew('P1', q1*N.x + q2*N.y)
    P.set_vel(N, P.pos_from(pN).dt(N))
    pP = Particle('pP', P, m)
    # Constraint Equations
    f_c = Matrix([q1**2 + q2**2 - L**2])
    # Calculate the lagrangian, and form the equations of motion
    Lag = Lagrangian(N, pP)
    LM = LagrangesMethod(Lag, [q1, q2], hol_coneqs=f_c, forcelist=[(P, m*g*N.x)], frame=N)
    LM.form_lagranges_equations()
    # Compose operating point
    op_point = {q1: L, q2: 0, q1d: 0, q2d: 0, q1d.diff(t): 0, q2d.diff(t): 0}
    # Solve for multiplier operating point
    lam_op = LM.solve_multipliers(op_point=op_point)
    op_point.update(lam_op)
    # Perform the Linearization
    A, B, inp_vec = LM.linearize([q2], [q2d], [q1], [q1d],
            op_point=op_point, A_and_B=True)
    assert simplify(A) == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])

    # Check if passing a function to linear_solver works
    A, B, inp_vec = LM.linearize([q2], [q2d], [q1], [q1d], op_point=op_point,
                                 A_and_B=True, linear_solver=lambda A, b:
                                 A.LUsolve(b))
    assert simplify(A) == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])

def test_linearize_rolling_disc_lagrange():
    q1, q2, q3 = q = dynamicsymbols('q1 q2 q3')
    q1d, q2d, q3d = qd = dynamicsymbols('q1 q2 q3', 1)
    r, m, g = symbols('r m g')

    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])

    C = Point('C')
    C.set_vel(N, 0)
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)

    I = inertia(L, m / 4 * r**2, m / 2 * r**2, m / 4 * r**2)
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    BodyD.potential_energy = - m * g * r * cos(q2)

    Lag = Lagrangian(N, BodyD)
    l = LagrangesMethod(Lag, q)
    l.form_lagranges_equations()

    # Linearize about steady-state upright rolling
    op_point = {q1: 0, q2: 0, q3: 0,
                q1d: 0, q2d: 0,
                q1d.diff(): 0, q2d.diff(): 0, q3d.diff(): 0}
    A = l.linearize(q_ind=q, qd_ind=qd, op_point=op_point, A_and_B=True)[0]
    sol = Matrix([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, -6*q3d, 0],
                  [0, -4*g/(5*r), 0, 6*q3d/5, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

    assert A == sol
