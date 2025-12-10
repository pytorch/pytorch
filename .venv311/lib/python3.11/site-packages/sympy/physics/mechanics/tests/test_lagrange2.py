from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics import ReferenceFrame, Point, Particle
from sympy.physics.mechanics import LagrangesMethod, Lagrangian

### This test asserts that a system with more than one external forces
### is accurately formed with Lagrange method (see issue #8626)

def test_lagrange_2forces():
    ### Equations for two damped springs in series with two forces

    ### generalized coordinates
    q1, q2 = dynamicsymbols('q1, q2')
    ### generalized speeds
    q1d, q2d = dynamicsymbols('q1, q2', 1)

    ### Mass, spring strength, friction coefficient
    m, k, nu = symbols('m, k, nu')

    N = ReferenceFrame('N')
    O = Point('O')

    ### Two points
    P1 = O.locatenew('P1', q1 * N.x)
    P1.set_vel(N, q1d * N.x)
    P2 = O.locatenew('P1', q2 * N.x)
    P2.set_vel(N, q2d * N.x)

    pP1 = Particle('pP1', P1, m)
    pP1.potential_energy = k * q1**2 / 2

    pP2 = Particle('pP2', P2, m)
    pP2.potential_energy = k * (q1 - q2)**2 / 2

    #### Friction forces
    forcelist = [(P1, - nu * q1d * N.x),
                 (P2, - nu * q2d * N.x)]
    lag = Lagrangian(N, pP1, pP2)

    l_method = LagrangesMethod(lag, (q1, q2), forcelist=forcelist, frame=N)
    l_method.form_lagranges_equations()

    eq1 = l_method.eom[0]
    assert eq1.diff(q1d) == nu
    eq2 = l_method.eom[1]
    assert eq2.diff(q2d) == nu
