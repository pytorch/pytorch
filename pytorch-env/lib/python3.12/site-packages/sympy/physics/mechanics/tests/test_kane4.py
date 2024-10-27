from sympy import (cos, sin, Matrix, symbols)
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
                                        KanesMethod, Particle)

def test_replace_qdots_in_force():
    # Test PR 16700 "Replaces qdots with us in force-list in kanes.py"
    # The new functionality allows one to specify forces in qdots which will
    # automatically be replaced with u:s which are defined by the kde supplied
    # to KanesMethod. The test case is the double pendulum with interacting
    # forces in the example of chapter 4.7 "CONTRIBUTING INTERACTION FORCES"
    # in Ref. [1]. Reference list at end test function.

    q1, q2 = dynamicsymbols('q1, q2')
    qd1, qd2 = dynamicsymbols('q1, q2', level=1)
    u1, u2 = dynamicsymbols('u1, u2')

    l, m = symbols('l, m')

    N = ReferenceFrame('N') # Inertial frame
    A = N.orientnew('A', 'Axis', (q1, N.z)) # Rod A frame
    B = A.orientnew('B', 'Axis', (q2, N.z)) # Rod B frame

    O = Point('O') # Origo
    O.set_vel(N, 0)

    P = O.locatenew('P', ( l * A.x )) # Point @ end of rod A
    P.v2pt_theory(O, N, A)

    Q = P.locatenew('Q', ( l * B.x )) # Point @ end of rod B
    Q.v2pt_theory(P, N, B)

    Ap = Particle('Ap', P, m)
    Bp = Particle('Bp', Q, m)

    # The forces are specified below. sigma is the torsional spring stiffness
    # and delta is the viscous damping coefficient acting between the two
    # bodies. Here, we specify the viscous damper as function of qdots prior
    # forming the kde. In more complex systems it not might be obvious which
    # kde is most efficient, why it is convenient to specify viscous forces in
    # qdots independently of the kde.
    sig, delta = symbols('sigma, delta')
    Ta = (sig * q2 + delta * qd2) * N.z
    forces = [(A, Ta), (B, -Ta)]

    # Try different kdes.
    kde1 = [u1 - qd1, u2 - qd2]
    kde2 = [u1 - qd1, u2 - (qd1 + qd2)]

    KM1 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde1)
    fr1, fstar1 = KM1.kanes_equations([Ap, Bp], forces)

    KM2 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde2)
    fr2, fstar2 = KM2.kanes_equations([Ap, Bp], forces)

    # Check EOM for KM2:
    # Mass and force matrix from p.6 in Ref. [2] with added forces from
    # example of chapter 4.7 in [1] and without gravity.
    forcing_matrix_expected = Matrix( [ [ m * l**2 * sin(q2) * u2**2 + sig * q2
                                        + delta * (u2 - u1)],
                                        [ m * l**2 * sin(q2) * -u1**2 - sig * q2
                                        - delta * (u2 - u1)] ] )
    mass_matrix_expected = Matrix( [ [ 2 * m * l**2, m * l**2 * cos(q2) ],
                                    [ m * l**2 * cos(q2), m * l**2 ] ] )

    assert (KM2.mass_matrix.expand() == mass_matrix_expected.expand())
    assert (KM2.forcing.expand() == forcing_matrix_expected.expand())

    # Check fr1 with reference fr_expected from [1] with u:s instead of qdots.
    fr1_expected = Matrix([ 0, -(sig*q2 + delta * u2) ])
    assert fr1.expand() == fr1_expected.expand()

    # Check fr2
    fr2_expected = Matrix([sig * q2 + delta * (u2 - u1),
                            - sig * q2 - delta * (u2 - u1)])
    assert fr2.expand() == fr2_expected.expand()

    # Specifying forces in u:s should stay the same:
    Ta = (sig * q2 + delta * u2) * N.z
    forces = [(A, Ta), (B, -Ta)]
    KM1 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde1)
    fr1, fstar1 = KM1.kanes_equations([Ap, Bp], forces)

    assert fr1.expand() == fr1_expected.expand()

    Ta = (sig * q2 + delta * (u2-u1)) * N.z
    forces = [(A, Ta), (B, -Ta)]
    KM2 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde2)
    fr2, fstar2 = KM2.kanes_equations([Ap, Bp], forces)

    assert fr2.expand() == fr2_expected.expand()

    # Test if we have a qubic qdot force:
    Ta = (sig * q2 + delta * qd2**3) * N.z
    forces = [(A, Ta), (B, -Ta)]

    KM1 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde1)
    fr1, fstar1 = KM1.kanes_equations([Ap, Bp], forces)

    fr1_cubic_expected = Matrix([ 0, -(sig*q2 + delta * u2**3) ])

    assert fr1.expand() == fr1_cubic_expected.expand()

    KM2 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde2)
    fr2, fstar2 = KM2.kanes_equations([Ap, Bp], forces)

    fr2_cubic_expected = Matrix([sig * q2 + delta * (u2 - u1)**3,
                            - sig * q2 - delta * (u2 - u1)**3])

    assert fr2.expand() == fr2_cubic_expected.expand()

    # References:
    # [1] T.R. Kane, D. a Levinson, Dynamics Theory and Applications, 2005.
    # [2] Arun K Banerjee, Flexible Multibody Dynamics:Efficient Formulations
    #     and Applications, John Wiley and Sons, Ltd, 2016.
    #     doi:http://dx.doi.org/10.1002/9781119015635.
