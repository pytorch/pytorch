from sympy import (zeros, Matrix, symbols, lambdify, sqrt, pi,
                                simplify)
from sympy.physics.mechanics import (dynamicsymbols, cross, inertia, RigidBody,
                                     ReferenceFrame, KanesMethod)


def _create_rolling_disc():
    # Define symbols and coordinates
    t = dynamicsymbols._t
    q1, q2, q3, q4, q5, u1, u2, u3, u4, u5 = dynamicsymbols('q1:6 u1:6')
    g, r, m = symbols('g r m')
    # Define bodies and frames
    ground = RigidBody('ground')
    disc = RigidBody('disk', mass=m)
    disc.inertia = (m * r ** 2 / 4 * inertia(disc.frame, 1, 2, 1),
                    disc.masscenter)
    ground.masscenter.set_vel(ground.frame, 0)
    disc.masscenter.set_vel(disc.frame, 0)
    int_frame = ReferenceFrame('int_frame')
    # Orient frames
    int_frame.orient_body_fixed(ground.frame, (q1, q2, 0), 'zxy')
    disc.frame.orient_axis(int_frame, int_frame.y, q3)
    g_w_d = disc.frame.ang_vel_in(ground.frame)
    disc.frame.set_ang_vel(ground.frame,
                           u1 * disc.x + u2 * disc.y + u3 * disc.z)
    # Define points
    cp = ground.masscenter.locatenew('contact_point',
                                     q4 * ground.x + q5 * ground.y)
    cp.set_vel(ground.frame, u4 * ground.x + u5 * ground.y)
    disc.masscenter.set_pos(cp, r * int_frame.z)
    disc.masscenter.set_vel(ground.frame, cross(
        disc.frame.ang_vel_in(ground.frame), disc.masscenter.pos_from(cp)))
    # Define kinematic differential equations
    kdes = [g_w_d.dot(disc.x) - u1, g_w_d.dot(disc.y) - u2,
            g_w_d.dot(disc.z) - u3, q4.diff(t) - u4, q5.diff(t) - u5]
    # Define nonholonomic constraints
    v0 = cp.vel(ground.frame) + cross(
        disc.frame.ang_vel_in(int_frame), cp.pos_from(disc.masscenter))
    fnh = [v0.dot(ground.x), v0.dot(ground.y)]
    # Define loads
    loads = [(disc.masscenter, -disc.mass * g * ground.z)]
    bodies = [disc]
    return {
        'frame': ground.frame,
        'q_ind': [q1, q2, q3, q4, q5],
        'u_ind': [u1, u2, u3],
        'u_dep': [u4, u5],
        'kdes': kdes,
        'fnh': fnh,
        'bodies': bodies,
        'loads': loads
    }


def _verify_rolling_disc_numerically(kane, all_zero=False):
    q, u, p = dynamicsymbols('q1:6'), dynamicsymbols('u1:6'), symbols('g r m')
    eval_sys = lambdify((q, u, p), (kane.mass_matrix_full, kane.forcing_full),
                        cse=True)
    solve_sys = lambda q, u, p: Matrix.LUsolve(
        *(Matrix(mat) for mat in eval_sys(q, u, p)))
    solve_u_dep = lambdify((q, u[:3], p), kane._Ars * Matrix(u[:3]), cse=True)
    eps = 1e-10
    p_vals = (9.81, 0.26, 3.43)
    # First numeric test
    q_vals = (0.3, 0.1, 1.97, -0.35, 2.27)
    u_vals = [-0.2, 1.3, 0.15]
    u_vals.extend(solve_u_dep(q_vals, u_vals, p_vals)[:2, 0])
    expected = Matrix([
        0.126603940595934, 0.215942571601660, 1.28736069604936,
        0.319764288376543, 0.0989146857254898, -0.925848952664489,
        -0.0181350656532944, 2.91695398184589, -0.00992793421754526,
        0.0412861634829171])
    assert all(abs(x) < eps for x in
               (solve_sys(q_vals, u_vals, p_vals) - expected))
    # Second numeric test
    q_vals = (3.97, -0.28, 8.2, -0.35, 2.27)
    u_vals = [-0.25, -2.2, 0.62]
    u_vals.extend(solve_u_dep(q_vals, u_vals, p_vals)[:2, 0])
    expected = Matrix([
        0.0259159090798597, 0.668041660387416, -2.19283799213811,
        0.385441810852219, 0.420109283790573, 1.45030568179066,
        -0.0110924422400793, -8.35617840186040, -0.154098542632173,
        -0.146102664410010])
    assert all(abs(x) < eps for x in
               (solve_sys(q_vals, u_vals, p_vals) - expected))
    if all_zero:
        q_vals = (0, 0, 0, 0, 0)
        u_vals = (0, 0, 0, 0, 0)
        assert solve_sys(q_vals, u_vals, p_vals) == zeros(10, 1)


def test_kane_rolling_disc_lu():
    props = _create_rolling_disc()
    kane = KanesMethod(props['frame'], props['q_ind'], props['u_ind'],
                       props['kdes'], u_dependent=props['u_dep'],
                       velocity_constraints=props['fnh'],
                       bodies=props['bodies'], forcelist=props['loads'],
                       explicit_kinematics=False, constraint_solver='LU')
    kane.kanes_equations()
    _verify_rolling_disc_numerically(kane)


def test_kane_rolling_disc_kdes_callable():
    props = _create_rolling_disc()
    kane = KanesMethod(
        props['frame'], props['q_ind'], props['u_ind'], props['kdes'],
        u_dependent=props['u_dep'], velocity_constraints=props['fnh'],
        bodies=props['bodies'], forcelist=props['loads'],
        explicit_kinematics=False,
        kd_eqs_solver=lambda A, b: simplify(A.LUsolve(b)))
    q, u, p = dynamicsymbols('q1:6'), dynamicsymbols('u1:6'), symbols('g r m')
    qd = dynamicsymbols('q1:6', 1)
    eval_kdes = lambdify((q, qd, u, p), tuple(kane.kindiffdict().items()))
    eps = 1e-10
    # Test with only zeros. If 'LU' would be used this would result in nan.
    p_vals = (9.81, 0.25, 3.5)
    zero_vals = (0, 0, 0, 0, 0)
    assert all(abs(qdi - fui) < eps for qdi, fui in
               eval_kdes(zero_vals, zero_vals, zero_vals, p_vals))
    # Test with some arbitrary values
    q_vals = tuple(map(float, (pi / 6, pi / 3, pi / 2, 0.42, 0.62)))
    qd_vals = tuple(map(float, (4, 1 / 3, 4 - 2 * sqrt(3),
                                0.25 * (2 * sqrt(3) - 3),
                                0.25 * (2 - sqrt(3)))))
    u_vals = tuple(map(float, (-2, 4, 1 / 3, 0.25 * (-3 + 2 * sqrt(3)),
                               0.25 * (-sqrt(3) + 2))))
    assert all(abs(qdi - fui) < eps for qdi, fui in
               eval_kdes(q_vals, qd_vals, u_vals, p_vals))
