import sympy.physics.mechanics.models as models
from sympy import (cos, sin, Matrix, symbols, zeros)
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols)


def test_multi_mass_spring_damper_inputs():

    c0, k0, m0 = symbols("c0 k0 m0")
    g = symbols("g")
    v0, x0, f0 = dynamicsymbols("v0 x0 f0")

    kane1 = models.multi_mass_spring_damper(1)
    massmatrix1 = Matrix([[m0]])
    forcing1 = Matrix([[-c0*v0 - k0*x0]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == Matrix([0])
    assert simplify(forcing1 - kane1.forcing) == Matrix([0])

    kane2 = models.multi_mass_spring_damper(1, True)
    massmatrix2 = Matrix([[m0]])
    forcing2 = Matrix([[-c0*v0 + g*m0 - k0*x0]])
    assert simplify(massmatrix2 - kane2.mass_matrix) == Matrix([0])
    assert simplify(forcing2 - kane2.forcing) == Matrix([0])

    kane3 = models.multi_mass_spring_damper(1, True, True)
    massmatrix3 = Matrix([[m0]])
    forcing3 = Matrix([[-c0*v0 + g*m0 - k0*x0 + f0]])
    assert simplify(massmatrix3 - kane3.mass_matrix) == Matrix([0])
    assert simplify(forcing3 - kane3.forcing) == Matrix([0])

    kane4 = models.multi_mass_spring_damper(1, False, True)
    massmatrix4 = Matrix([[m0]])
    forcing4 = Matrix([[-c0*v0 - k0*x0 + f0]])
    assert simplify(massmatrix4 - kane4.mass_matrix) == Matrix([0])
    assert simplify(forcing4 - kane4.forcing) == Matrix([0])


def test_multi_mass_spring_damper_higher_order():
    c0, k0, m0 = symbols("c0 k0 m0")
    c1, k1, m1 = symbols("c1 k1 m1")
    c2, k2, m2 = symbols("c2 k2 m2")
    v0, x0 = dynamicsymbols("v0 x0")
    v1, x1 = dynamicsymbols("v1 x1")
    v2, x2 = dynamicsymbols("v2 x2")

    kane1 = models.multi_mass_spring_damper(3)
    massmatrix1 = Matrix([[m0 + m1 + m2, m1 + m2, m2],
                          [m1 + m2, m1 + m2, m2],
                          [m2, m2, m2]])
    forcing1 = Matrix([[-c0*v0 - k0*x0],
                       [-c1*v1 - k1*x1],
                       [-c2*v2 - k2*x2]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(3)
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0, 0])


def test_n_link_pendulum_on_cart_inputs():
    l0, m0 = symbols("l0 m0")
    m1 = symbols("m1")
    g = symbols("g")
    q0, q1, F, T1 = dynamicsymbols("q0 q1 F T1")
    u0, u1 = dynamicsymbols("u0 u1")

    kane1 = models.n_link_pendulum_on_cart(1)
    massmatrix1 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    forcing1 = Matrix([[-l0*m1*u1**2*sin(q1) + F], [g*l0*m1*sin(q1)]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(2)
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0])

    kane2 = models.n_link_pendulum_on_cart(1, False)
    massmatrix2 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    forcing2 = Matrix([[-l0*m1*u1**2*sin(q1)], [g*l0*m1*sin(q1)]])
    assert simplify(massmatrix2 - kane2.mass_matrix) == zeros(2)
    assert simplify(forcing2 - kane2.forcing) == Matrix([0, 0])

    kane3 = models.n_link_pendulum_on_cart(1, False, True)
    massmatrix3 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    forcing3 = Matrix([[-l0*m1*u1**2*sin(q1)], [g*l0*m1*sin(q1) + T1]])
    assert simplify(massmatrix3 - kane3.mass_matrix) == zeros(2)
    assert simplify(forcing3 - kane3.forcing) == Matrix([0, 0])

    kane4 = models.n_link_pendulum_on_cart(1, True, False)
    massmatrix4 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    forcing4 = Matrix([[-l0*m1*u1**2*sin(q1) + F], [g*l0*m1*sin(q1)]])
    assert simplify(massmatrix4 - kane4.mass_matrix) == zeros(2)
    assert simplify(forcing4 - kane4.forcing) == Matrix([0, 0])


def test_n_link_pendulum_on_cart_higher_order():
    l0, m0 = symbols("l0 m0")
    l1, m1 = symbols("l1 m1")
    m2 = symbols("m2")
    g = symbols("g")
    q0, q1, q2 = dynamicsymbols("q0 q1 q2")
    u0, u1, u2 = dynamicsymbols("u0 u1 u2")
    F, T1 = dynamicsymbols("F T1")

    kane1 = models.n_link_pendulum_on_cart(2)
    massmatrix1 = Matrix([[m0 + m1 + m2, -l0*m1*cos(q1) - l0*m2*cos(q1),
                           -l1*m2*cos(q2)],
                          [-l0*m1*cos(q1) - l0*m2*cos(q1), l0**2*m1 + l0**2*m2,
                           l0*l1*m2*(sin(q1)*sin(q2) + cos(q1)*cos(q2))],
                          [-l1*m2*cos(q2),
                           l0*l1*m2*(sin(q1)*sin(q2) + cos(q1)*cos(q2)),
                           l1**2*m2]])
    forcing1 = Matrix([[-l0*m1*u1**2*sin(q1) - l0*m2*u1**2*sin(q1) -
                        l1*m2*u2**2*sin(q2) + F],
                       [g*l0*m1*sin(q1) + g*l0*m2*sin(q1) -
                        l0*l1*m2*(sin(q1)*cos(q2) - sin(q2)*cos(q1))*u2**2],
                       [g*l1*m2*sin(q2) - l0*l1*m2*(-sin(q1)*cos(q2) +
                                                    sin(q2)*cos(q1))*u1**2]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(3)
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0, 0])
