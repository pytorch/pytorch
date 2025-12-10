from sympy.core.function import expand
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.sets.sets import Interval
from sympy.simplify.simplify import simplify
from sympy.physics.continuum_mechanics.beam import Beam
from sympy.functions import SingularityFunction, Piecewise, meijerg, Abs, log
from sympy.testing.pytest import raises
from sympy.physics.units import meter, newton, kilo, giga, milli
from sympy.physics.continuum_mechanics.beam import Beam3D
from sympy.geometry import Circle, Polygon, Point2D, Triangle
from sympy.core.sympify import sympify

x = Symbol('x')
y = Symbol('y')
R1, R2 = symbols('R1, R2')


def test_Beam():
    E = Symbol('E')
    E_1 = Symbol('E_1')
    I = Symbol('I')
    I_1 = Symbol('I_1')
    A = Symbol('A')

    b = Beam(1, E, I)
    assert b.length == 1
    assert b.elastic_modulus == E
    assert b.second_moment == I
    assert b.variable == x

    # Test the length setter
    b.length = 4
    assert b.length == 4

    # Test the E setter
    b.elastic_modulus = E_1
    assert b.elastic_modulus == E_1

    # Test the I setter
    b.second_moment = I_1
    assert b.second_moment is I_1

    # Test the variable setter
    b.variable = y
    assert b.variable is y

    # Test for all boundary conditions.
    b.bc_deflection = [(0, 2)]
    b.bc_slope = [(0, 1)]
    b.bc_bending_moment = [(0, 5)]
    b.bc_shear_force = [(2, 1)]
    assert b.boundary_conditions == {'deflection': [(0, 2)], 'slope': [(0, 1)],
                                     'bending_moment': [(0, 5)], 'shear_force': [(2, 1)]}

    # Test for shear force boundary condition method
    b.bc_shear_force.extend([(1, 1), (2, 3)])
    sf_bcs = b.bc_shear_force
    assert sf_bcs == [(2, 1), (1, 1), (2, 3)]

    # Test for slope boundary condition method
    b.bc_bending_moment.extend([(1, 3), (5, 3)])
    bm_bcs = b.bc_bending_moment
    assert bm_bcs == [(0, 5), (1, 3), (5, 3)]

    # Test for slope boundary condition method
    b.bc_slope.extend([(4, 3), (5, 0)])
    s_bcs = b.bc_slope
    assert s_bcs == [(0, 1), (4, 3), (5, 0)]

    # Test for deflection boundary condition method
    b.bc_deflection.extend([(4, 3), (5, 0)])
    d_bcs = b.bc_deflection
    assert d_bcs == [(0, 2), (4, 3), (5, 0)]

    # Test for updated boundary conditions
    bcs_new = b.boundary_conditions
    assert bcs_new == {
        'deflection': [(0, 2), (4, 3), (5, 0)],
        'slope': [(0, 1), (4, 3), (5, 0)],
        'bending_moment': [(0, 5), (1, 3), (5, 3)],
        'shear_force': [(2, 1), (1, 1), (2, 3)]}

    b1 = Beam(30, E, I)
    b1.apply_load(-8, 0, -1)
    b1.apply_load(R1, 10, -1)
    b1.apply_load(R2, 30, -1)
    b1.apply_load(120, 30, -2)
    b1.bc_deflection = [(10, 0), (30, 0)]
    b1.solve_for_reaction_loads(R1, R2)

    # Test for finding reaction forces
    p = b1.reaction_loads
    q = {R1: 6, R2: 2}
    assert p == q

    # Test for load distribution function.
    p = b1.load
    q = -8*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 10, -1) \
    + 120*SingularityFunction(x, 30, -2) + 2*SingularityFunction(x, 30, -1)
    assert p == q

    # Test for shear force distribution function
    p = b1.shear_force()
    q = 8*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 10, 0) \
    - 120*SingularityFunction(x, 30, -1) - 2*SingularityFunction(x, 30, 0)
    assert p == q

    # Test for shear stress distribution function
    p = b1.shear_stress()
    q = (8*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 10, 0) \
    - 120*SingularityFunction(x, 30, -1) \
    - 2*SingularityFunction(x, 30, 0))/A
    assert p==q

    # Test for bending moment distribution function
    p = b1.bending_moment()
    q = 8*SingularityFunction(x, 0, 1) - 6*SingularityFunction(x, 10, 1) \
    - 120*SingularityFunction(x, 30, 0) - 2*SingularityFunction(x, 30, 1)
    assert p == q

    # Test for slope distribution function
    p = b1.slope()
    q = -4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2) \
    + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) \
    + Rational(4000, 3)
    assert p == q/(E*I)

    # Test for deflection distribution function
    p = b1.deflection()
    q = x*Rational(4000, 3) - 4*SingularityFunction(x, 0, 3)/3 \
    + SingularityFunction(x, 10, 3) + 60*SingularityFunction(x, 30, 2) \
    + SingularityFunction(x, 30, 3)/3 - 12000
    assert p == q/(E*I)

    # Test using symbols
    l = Symbol('l')
    w0 = Symbol('w0')
    w2 = Symbol('w2')
    a1 = Symbol('a1')
    c = Symbol('c')
    c1 = Symbol('c1')
    d = Symbol('d')
    e = Symbol('e')
    f = Symbol('f')

    b2 = Beam(l, E, I)

    b2.apply_load(w0, a1, 1)
    b2.apply_load(w2, c1, -1)

    b2.bc_deflection = [(c, d)]
    b2.bc_slope = [(e, f)]

    # Test for load distribution function.
    p = b2.load
    q = w0*SingularityFunction(x, a1, 1) + w2*SingularityFunction(x, c1, -1)
    assert p == q

    # Test for shear force distribution function
    p = b2.shear_force()
    q = -w0*SingularityFunction(x, a1, 2)/2 \
    - w2*SingularityFunction(x, c1, 0)
    assert p == q

    # Test for shear stress distribution function
    p = b2.shear_stress()
    q = (-w0*SingularityFunction(x, a1, 2)/2 \
    - w2*SingularityFunction(x, c1, 0))/A
    assert p == q

    # Test for bending moment distribution function
    p = b2.bending_moment()
    q = -w0*SingularityFunction(x, a1, 3)/6 - w2*SingularityFunction(x, c1, 1)
    assert p == q

    # Test for slope distribution function
    p = b2.slope()
    q = (w0*SingularityFunction(x, a1, 4)/24 + w2*SingularityFunction(x, c1, 2)/2)/(E*I) + (E*I*f - w0*SingularityFunction(e, a1, 4)/24 - w2*SingularityFunction(e, c1, 2)/2)/(E*I)
    assert expand(p) == expand(q)

    # Test for deflection distribution function
    p = b2.deflection()
    q = x*(E*I*f - w0*SingularityFunction(e, a1, 4)/24 \
    - w2*SingularityFunction(e, c1, 2)/2)/(E*I) \
    + (w0*SingularityFunction(x, a1, 5)/120 \
    + w2*SingularityFunction(x, c1, 3)/6)/(E*I) \
    + (E*I*(-c*f + d) + c*w0*SingularityFunction(e, a1, 4)/24 \
    + c*w2*SingularityFunction(e, c1, 2)/2 \
    - w0*SingularityFunction(c, a1, 5)/120 \
    - w2*SingularityFunction(c, c1, 3)/6)/(E*I)
    assert simplify(p - q) == 0

    b3 = Beam(9, E, I, 2)
    b3.apply_load(value=-2, start=2, order=2, end=3)
    b3.bc_slope.append((0, 2))
    C3 = symbols('C3')
    C4 = symbols('C4')

    p = b3.load
    q = -2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) \
    + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)
    assert p == q

    p = b3.shear_force()
    q = 2*SingularityFunction(x, 2, 3)/3 - 2*SingularityFunction(x, 3, 1) \
    - 2*SingularityFunction(x, 3, 2) - 2*SingularityFunction(x, 3, 3)/3
    assert p == q

    p = b3.shear_stress()
    q = SingularityFunction(x, 2, 3)/3 - 1*SingularityFunction(x, 3, 1) \
    - 1*SingularityFunction(x, 3, 2) - 1*SingularityFunction(x, 3, 3)/3
    assert p == q

    p = b3.slope()
    q = 2 - (SingularityFunction(x, 2, 5)/30 - SingularityFunction(x, 3, 3)/3 \
    - SingularityFunction(x, 3, 4)/6 - SingularityFunction(x, 3, 5)/30)/(E*I)
    assert p == q

    p = b3.deflection()
    q = 2*x - (SingularityFunction(x, 2, 6)/180 \
    - SingularityFunction(x, 3, 4)/12 - SingularityFunction(x, 3, 5)/30 \
    - SingularityFunction(x, 3, 6)/180)/(E*I)
    assert p == q + C4

    b4 = Beam(4, E, I, 3)
    b4.apply_load(-3, 0, 0, end=3)

    p = b4.load
    q = -3*SingularityFunction(x, 0, 0) + 3*SingularityFunction(x, 3, 0)
    assert p == q

    p = b4.shear_force()
    q = 3*SingularityFunction(x, 0, 1) \
    - 3*SingularityFunction(x, 3, 1)
    assert p == q

    p = b4.shear_stress()
    q = SingularityFunction(x, 0, 1) - SingularityFunction(x, 3, 1)
    assert p == q

    p = b4.slope()
    q = -3*SingularityFunction(x, 0, 3)/6 + 3*SingularityFunction(x, 3, 3)/6
    assert p == q/(E*I) + C3

    p = b4.deflection()
    q = -3*SingularityFunction(x, 0, 4)/24 + 3*SingularityFunction(x, 3, 4)/24
    assert p == q/(E*I) + C3*x + C4

    # can't use end with point loads
    raises(ValueError, lambda: b4.apply_load(-3, 0, -1, end=3))
    with raises(TypeError):
        b4.variable = 1


def test_insufficient_bconditions():
    # Test cases when required number of boundary conditions
    # are not provided to solve the integration constants.
    L = symbols('L', positive=True)
    E, I, P, a3, a4 = symbols('E I P a3 a4')

    b = Beam(L, E, I, base_char='a')
    b.apply_load(R2, L, -1)
    b.apply_load(R1, 0, -1)
    b.apply_load(-P, L/2, -1)
    b.solve_for_reaction_loads(R1, R2)

    p = b.slope()
    q = P*SingularityFunction(x, 0, 2)/4 - P*SingularityFunction(x, L/2, 2)/2 + P*SingularityFunction(x, L, 2)/4
    assert p == q/(E*I) + a3

    p = b.deflection()
    q = P*SingularityFunction(x, 0, 3)/12 - P*SingularityFunction(x, L/2, 3)/6 + P*SingularityFunction(x, L, 3)/12
    assert p == q/(E*I) + a3*x + a4

    b.bc_deflection = [(0, 0)]
    p = b.deflection()
    q = a3*x + P*SingularityFunction(x, 0, 3)/12 - P*SingularityFunction(x, L/2, 3)/6 + P*SingularityFunction(x, L, 3)/12
    assert p == q/(E*I)

    b.bc_deflection = [(0, 0), (L, 0)]
    p = b.deflection()
    q = -L**2*P*x/16 + P*SingularityFunction(x, 0, 3)/12 - P*SingularityFunction(x, L/2, 3)/6 + P*SingularityFunction(x, L, 3)/12
    assert p == q/(E*I)


def test_statically_indeterminate():
    E = Symbol('E')
    I = Symbol('I')
    M1, M2 = symbols('M1, M2')
    F = Symbol('F')
    l = Symbol('l', positive=True)

    b5 = Beam(l, E, I)
    b5.bc_deflection = [(0, 0),(l, 0)]
    b5.bc_slope = [(0, 0),(l, 0)]

    b5.apply_load(R1, 0, -1)
    b5.apply_load(M1, 0, -2)
    b5.apply_load(R2, l, -1)
    b5.apply_load(M2, l, -2)
    b5.apply_load(-F, l/2, -1)

    b5.solve_for_reaction_loads(R1, R2, M1, M2)
    p = b5.reaction_loads
    q = {R1: F/2, R2: F/2, M1: -F*l/8, M2: F*l/8}
    assert p == q


def test_beam_units():
    E = Symbol('E')
    I = Symbol('I')
    R1, R2 = symbols('R1, R2')

    kN = kilo*newton
    gN = giga*newton

    b = Beam(8*meter, 200*gN/meter**2, 400*1000000*(milli*meter)**4)
    b.apply_load(5*kN, 2*meter, -1)
    b.apply_load(R1, 0*meter, -1)
    b.apply_load(R2, 8*meter, -1)
    b.apply_load(10*kN/meter, 4*meter, 0, end=8*meter)
    b.bc_deflection = [(0*meter, 0*meter), (8*meter, 0*meter)]
    b.solve_for_reaction_loads(R1, R2)
    assert b.reaction_loads == {R1: -13750*newton, R2: -31250*newton}

    b = Beam(3*meter, E*newton/meter**2, I*meter**4)
    b.apply_load(8*kN, 1*meter, -1)
    b.apply_load(R1, 0*meter, -1)
    b.apply_load(R2, 3*meter, -1)
    b.apply_load(12*kN*meter, 2*meter, -2)
    b.bc_deflection = [(0*meter, 0*meter), (3*meter, 0*meter)]
    b.solve_for_reaction_loads(R1, R2)
    assert b.reaction_loads == {R1: newton*Rational(-28000, 3), R2: newton*Rational(4000, 3)}
    assert b.deflection().subs(x, 1*meter) == 62000*meter/(9*E*I)


def test_variable_moment():
    E = Symbol('E')
    I = Symbol('I')

    b = Beam(4, E, 2*(4 - x))
    b.apply_load(20, 4, -1)
    R, M = symbols('R, M')
    b.apply_load(R, 0, -1)
    b.apply_load(M, 0, -2)
    b.bc_deflection = [(0, 0)]
    b.bc_slope = [(0, 0)]
    b.solve_for_reaction_loads(R, M)
    assert b.slope().expand() == ((10*x*SingularityFunction(x, 0, 0)
        - 10*(x - 4)*SingularityFunction(x, 4, 0))/E).expand()
    assert b.deflection().expand() == ((5*x**2*SingularityFunction(x, 0, 0)
        - 10*Piecewise((0, Abs(x)/4 < 1), (x**2*meijerg(((-1, 1), ()), ((), (-2, 0)), x/4), True))
        + 40*SingularityFunction(x, 4, 1))/E).expand()

    b = Beam(4, E - x, I)
    b.apply_load(20, 4, -1)
    R, M = symbols('R, M')
    b.apply_load(R, 0, -1)
    b.apply_load(M, 0, -2)
    b.bc_deflection = [(0, 0)]
    b.bc_slope = [(0, 0)]
    b.solve_for_reaction_loads(R, M)
    assert b.slope().expand() == ((-80*(-log(-E) + log(-E + x))*SingularityFunction(x, 0, 0)
        + 80*(-log(-E + 4) + log(-E + x))*SingularityFunction(x, 4, 0) + 20*(-E*log(-E)
        + E*log(-E + x) + x)*SingularityFunction(x, 0, 0) - 20*(-E*log(-E + 4) + E*log(-E + x)
        + x - 4)*SingularityFunction(x, 4, 0))/I).expand()


def test_composite_beam():
    E = Symbol('E')
    I = Symbol('I')
    b1 = Beam(2, E, 1.5*I)
    b2 = Beam(2, E, I)
    b = b1.join(b2, "fixed")
    b.apply_load(-20, 0, -1)
    b.apply_load(80, 0, -2)
    b.apply_load(20, 4, -1)
    b.bc_slope = [(0, 0)]
    b.bc_deflection = [(0, 0)]
    assert b.length == 4
    assert b.second_moment == Piecewise((1.5*I, x <= 2), (I, x <= 4))
    assert b.slope().subs(x, 4) == 120.0/(E*I)
    assert b.slope().subs(x, 2) == 80.0/(E*I)
    assert int(b.deflection().subs(x, 4).args[0]) == -302  # Coefficient of 1/(E*I)

    l = symbols('l', positive=True)
    R1, M1, R2, R3, P = symbols('R1 M1 R2 R3 P')
    b1 = Beam(2*l, E, I)
    b2 = Beam(2*l, E, I)
    b = b1.join(b2,"hinge")
    b.apply_load(M1, 0, -2)
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, l, -1)
    b.apply_load(R3, 4*l, -1)
    b.apply_load(P, 3*l, -1)
    b.bc_slope = [(0, 0)]
    b.bc_deflection = [(0, 0), (l, 0), (4*l, 0)]
    b.solve_for_reaction_loads(M1, R1, R2, R3)
    assert b.reaction_loads == {R3: -P/2, R2: P*Rational(-5, 4), M1: -P*l/4, R1: P*Rational(3, 4)}
    assert b.slope().subs(x, 3*l) == -7*P*l**2/(48*E*I)
    assert b.deflection().subs(x, 2*l) == 7*P*l**3/(24*E*I)
    assert b.deflection().subs(x, 3*l) == 5*P*l**3/(16*E*I)

    # When beams having same second moment are joined.
    b1 = Beam(2, 500, 10)
    b2 = Beam(2, 500, 10)
    b = b1.join(b2, "fixed")
    b.apply_load(M1, 0, -2)
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, 1, -1)
    b.apply_load(R3, 4, -1)
    b.apply_load(10, 3, -1)
    b.bc_slope = [(0, 0)]
    b.bc_deflection = [(0, 0), (1, 0), (4, 0)]
    b.solve_for_reaction_loads(M1, R1, R2, R3)
    assert b.slope() == -2*SingularityFunction(x, 0, 1)/5625 + SingularityFunction(x, 0, 2)/1875\
                - 133*SingularityFunction(x, 1, 2)/135000 + SingularityFunction(x, 3, 2)/1000\
                - 37*SingularityFunction(x, 4, 2)/67500
    assert b.deflection() == -SingularityFunction(x, 0, 2)/5625 + SingularityFunction(x, 0, 3)/5625\
                    - 133*SingularityFunction(x, 1, 3)/405000 + SingularityFunction(x, 3, 3)/3000\
                    - 37*SingularityFunction(x, 4, 3)/202500


def test_point_cflexure():
    E = Symbol('E')
    I = Symbol('I')
    b = Beam(10, E, I)
    b.apply_load(-4, 0, -1)
    b.apply_load(-46, 6, -1)
    b.apply_load(10, 2, -1)
    b.apply_load(20, 4, -1)
    b.apply_load(3, 6, 0)
    assert b.point_cflexure() == [Rational(10, 3)]

    E = Symbol('E')
    I = Symbol('I')
    b = Beam(15, E, I)
    r0 = b.apply_support(0, type='pin')
    r10 = b.apply_support(10, type='pin')
    r15, m15 = b.apply_support(15, type='fixed')
    b.apply_rotation_hinge(12)
    b.apply_load(-10, 5, -1)
    b.apply_load(-5, 10, 0, 15)
    b.solve_for_reaction_loads(r0, r10, r15, m15)
    assert b.point_cflexure() == [Rational(1200, 163), 12, Rational(163, 12)]

    E = Symbol('E')
    I = Symbol('I')
    b = Beam(15, E, I)
    r0 = b.apply_support(0, type='pin')
    r10 = b.apply_support(10, type='pin')
    r15, m15 = b.apply_support(15, type='fixed')
    b.apply_rotation_hinge(5)
    b.apply_rotation_hinge(12)
    b.apply_load(-10, 5, -1)
    b.apply_load(-5, 10, 0, 15)
    b.solve_for_reaction_loads(r0, r10, r15, m15)
    with raises(NotImplementedError):
        b.point_cflexure()

def test_remove_load():
    E = Symbol('E')
    I = Symbol('I')
    b = Beam(4, E, I)

    try:
        b.remove_load(2, 1, -1)
    # As no load is applied on beam, ValueError should be returned.
    except ValueError:
        assert True
    else:
        assert False

    b.apply_load(-3, 0, -2)
    b.apply_load(4, 2, -1)
    b.apply_load(-2, 2, 2, end = 3)
    b.remove_load(-2, 2, 2, end = 3)
    assert b.load == -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1)
    assert b.applied_loads == [(-3, 0, -2, None), (4, 2, -1, None)]

    try:
        b.remove_load(1, 2, -1)
    # As load of this magnitude was never applied at
    # this position, method should return a ValueError.
    except ValueError:
        assert True
    else:
        assert False

    b.remove_load(-3, 0, -2)
    b.remove_load(4, 2, -1)
    assert b.load == 0
    assert b.applied_loads == []


def test_apply_support():
    E = Symbol('E')
    I = Symbol('I')

    b = Beam(4, E, I)
    b.apply_support(0, "cantilever")
    b.apply_load(20, 4, -1)
    M_0, R_0 = symbols('M_0, R_0')
    b.solve_for_reaction_loads(R_0, M_0)
    assert simplify(b.slope()) == simplify((80*SingularityFunction(x, 0, 1) - 10*SingularityFunction(x, 0, 2)
                + 10*SingularityFunction(x, 4, 2))/(E*I))
    assert simplify(b.deflection()) == simplify((40*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 0, 3)/3
                + 10*SingularityFunction(x, 4, 3)/3)/(E*I))

    b = Beam(30, E, I)
    p0 = b.apply_support(10, "pin")
    p1 = b.apply_support(30, "roller")
    b.apply_load(-8, 0, -1)
    b.apply_load(120, 30, -2)
    b.solve_for_reaction_loads(p0, p1)
    assert b.slope() == (-4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2)
            + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) + Rational(4000, 3))/(E*I)
    assert b.deflection() == (x*Rational(4000, 3) - 4*SingularityFunction(x, 0, 3)/3 + SingularityFunction(x, 10, 3)
            + 60*SingularityFunction(x, 30, 2) + SingularityFunction(x, 30, 3)/3 - 12000)/(E*I)
    R_10 = Symbol('R_10')
    R_30 = Symbol('R_30')
    assert p0 == R_10
    assert b.reaction_loads == {R_10: 6, R_30: 2}
    assert b.reaction_loads[p0] == 6

    b = Beam(8, E, I)
    p0, m0 = b.apply_support(0, "fixed")
    p1 = b.apply_support(8, "roller")
    b.apply_load(-5, 0, 0, 8)
    b.solve_for_reaction_loads(p0, m0, p1)
    R_0 = Symbol('R_0')
    M_0 = Symbol('M_0')
    R_8 = Symbol('R_8')
    assert p0 == R_0
    assert m0 == M_0
    assert p1 == R_8
    assert b.reaction_loads == {R_0: 25, M_0: -40, R_8: 15}
    assert b.reaction_loads[m0] == -40

    P = Symbol('P', positive=True)
    L = Symbol('L', positive=True)
    b = Beam(L, E, I)
    b.apply_support(0, type='fixed')
    b.apply_support(L, type='fixed')
    b.apply_load(-P, L/2, -1)
    R_0, R_L, M_0, M_L = symbols('R_0, R_L, M_0, M_L')
    b.solve_for_reaction_loads(R_0, R_L, M_0, M_L)
    assert b.reaction_loads == {R_0: P/2, R_L: P/2, M_0: -L*P/8, M_L: L*P/8}

def test_apply_rotation_hinge():
    b = Beam(15, 20, 20)
    r0, m0 = b.apply_support(0, type='fixed')
    r10 = b.apply_support(10, type='pin')
    r15 = b.apply_support(15, type='pin')
    p7 = b.apply_rotation_hinge(7)
    p12 = b.apply_rotation_hinge(12)
    b.apply_load(-10, 7, -1)
    b.apply_load(-2, 10, 0, 15)
    b.solve_for_reaction_loads(r0, m0, r10, r15)
    R_0, M_0, R_10, R_15, P_7, P_12 = symbols('R_0, M_0, R_10, R_15, P_7, P_12')
    expected_reactions = {R_0: 20/3, M_0: -140/3, R_10: 31/3, R_15: 3}
    expected_rotations = {P_7: 2281/2160, P_12: -5137/5184}
    reaction_symbols = [r0, m0, r10, r15]
    rotation_symbols = [p7, p12]
    tolerance = 1e-6
    assert all(abs(b.reaction_loads[r] - expected_reactions[r]) < tolerance for r in reaction_symbols)
    assert all(abs(b.rotation_jumps[r] - expected_rotations[r]) < tolerance for r in rotation_symbols)
    expected_bending_moment = (140 * SingularityFunction(x, 0, 0) / 3 - 20 * SingularityFunction(x, 0, 1) / 3
        - 11405 * SingularityFunction(x, 7, -1) / 27 + 10 * SingularityFunction(x, 7, 1)
        - 31 * SingularityFunction(x, 10, 1) / 3 + SingularityFunction(x, 10, 2)
        + 128425 * SingularityFunction(x, 12, -1) / 324 - 3 * SingularityFunction(x, 15, 1)
        - SingularityFunction(x, 15, 2))
    assert b.bending_moment().expand() == expected_bending_moment.expand()
    expected_slope = (-7*SingularityFunction(x, 0, 1)/60 + SingularityFunction(x, 0, 2)/120
        + 2281*SingularityFunction(x, 7, 0)/2160 - SingularityFunction(x, 7, 2)/80
        + 31*SingularityFunction(x, 10, 2)/2400 - SingularityFunction(x, 10, 3)/1200
        - 5137*SingularityFunction(x, 12, 0)/5184 + 3*SingularityFunction(x, 15, 2)/800
        + SingularityFunction(x, 15, 3)/1200)
    assert b.slope().expand() == expected_slope.expand()
    expected_deflection = (-7 * SingularityFunction(x, 0, 2) / 120 + SingularityFunction(x, 0, 3) / 360
        + 2281 * SingularityFunction(x, 7, 1) / 2160 - SingularityFunction(x, 7, 3) / 240
        + 31 * SingularityFunction(x, 10, 3) / 7200 - SingularityFunction(x, 10, 4) / 4800
        - 5137 * SingularityFunction(x, 12, 1) / 5184 + SingularityFunction(x, 15, 3) / 800
        + SingularityFunction(x, 15, 4) / 4800)
    assert b.deflection().expand() == expected_deflection.expand()

    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    b = Beam(10, E, I)
    r0, m0 = b.apply_support(0, type="fixed")
    r10 = b.apply_support(10, type="pin")
    b.apply_rotation_hinge(6)
    b.apply_load(F, 8, -1)
    b.solve_for_reaction_loads(r0, m0, r10)
    assert b.reaction_loads == {R_0: -F/2, M_0: 3*F, R_10: -F/2}
    assert (b.bending_moment() == -3*F*SingularityFunction(x, 0, 0) + F*SingularityFunction(x, 0, 1)/2
            + 17*F*SingularityFunction(x, 6, -1) - F*SingularityFunction(x, 8, 1)
            + F*SingularityFunction(x, 10, 1)/2)
    expected_deflection = -(-3*F*SingularityFunction(x, 0, 2)/2 + F*SingularityFunction(x, 0, 3)/12
            + 17*F*SingularityFunction(x, 6, 1) - F*SingularityFunction(x, 8, 3)/6
            + F*SingularityFunction(x, 10, 3)/12)/(E*I)
    assert b.deflection().expand() == expected_deflection.expand()

    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    l1 = Symbol('l1', positive=True)
    l2 = Symbol('l2', positive=True)
    l3 = Symbol('l3', positive=True)
    L = l1 + l2 + l3
    b = Beam(L, E, I)
    r0, m0 = b.apply_support(0, type="fixed")
    r1 = b.apply_support(L, type="pin")
    b.apply_rotation_hinge(l1)
    b.apply_load(F, l1+l2, -1)
    b.solve_for_reaction_loads(r0, m0, r1)
    assert b.reaction_loads[r0] == -F*l3/(l2 + l3)
    assert b.reaction_loads[m0] == F*l1*l3/(l2 + l3)
    assert b.reaction_loads[r1] == -F*l2/(l2 + l3)
    expected_bending_moment = (-F*l1*l3*SingularityFunction(x, 0, 0)/(l2 + l3)
            + F*l2*SingularityFunction(x, l1 + l2 + l3, 1)/(l2 + l3)
            + F*l3*SingularityFunction(x, 0, 1)/(l2 + l3) - F*SingularityFunction(x, l1 + l2, 1)
            - (-2*F*l1**3*l3 - 3*F*l1**2*l2*l3 - 3*F*l1**2*l3**2 + F*l2**3*l3 + 3*F*l2**2*l3**2 + 2*F*l2*l3**3)
            *SingularityFunction(x, l1, -1)/(6*l2**2 + 12*l2*l3 + 6*l3**2))
    assert simplify(b.bending_moment().expand()) == simplify(expected_bending_moment.expand())

def test_apply_sliding_hinge():
    b = Beam(13, 20, 20)
    r0, m0 = b.apply_support(0, type="fixed")
    w8 = b.apply_sliding_hinge(8)
    r13 = b.apply_support(13, type="pin")
    b.apply_load(-10, 5, -1)
    b.solve_for_reaction_loads(r0, m0, r13)
    R_0, M_0, R_13, W_8 = symbols('R_0, M_0, R_13 W_8')
    assert b.reaction_loads == {R_0: 10, M_0: -50, R_13: 0}
    tolerance = 1e-6
    assert abs(b.deflection_jumps[w8] - 85/24) < tolerance
    assert (b.bending_moment() == 50*SingularityFunction(x, 0, 0) - 10*SingularityFunction(x, 0, 1)
            + 10*SingularityFunction(x, 5, 1) - 4250*SingularityFunction(x, 8, -2)/3)
    assert (b.deflection() == -SingularityFunction(x, 0, 2)/16 + SingularityFunction(x, 0, 3)/240
            - SingularityFunction(x, 5, 3)/240 + 85*SingularityFunction(x, 8, 0)/24)

    E = Symbol('E')
    I = Symbol('I')
    I2 = Symbol('I2')
    b1 = Beam(5, E, I)
    b2 = Beam(8, E, I2)
    b = b1.join(b2)
    r0, m0 = b.apply_support(0, type="fixed")
    b.apply_sliding_hinge(8)
    r13 = b.apply_support(13, type="pin")
    b.apply_load(-10, 5, -1)
    b.solve_for_reaction_loads(r0, m0, r13)
    W_8 = Symbol('W_8')
    assert b.deflection_jumps == {W_8: 4250/(3*E*I2)}

    E = Symbol('E')
    I = Symbol('I')
    q = Symbol('q')
    l1 = Symbol('l1', positive=True)
    l2 = Symbol('l2', positive=True)
    l3 = Symbol('l3', positive=True)
    L = l1 + l2 + l3
    b = Beam(L, E, I)
    r0 = b.apply_support(0, type="pin")
    r3 = b.apply_support(l1, type="pin")
    b.apply_sliding_hinge(l1 + l2)
    r10 = b.apply_support(L, type="pin")
    b.apply_load(q, 0, 0, l1)
    b.solve_for_reaction_loads(r0, r3, r10)
    assert (b.bending_moment() == l1*q*SingularityFunction(x, 0, 1)/2 + l1*q*SingularityFunction(x, l1, 1)/2
            - q*SingularityFunction(x, 0, 2)/2 + q*SingularityFunction(x, l1, 2)/2
            + (-l1**3*l2*q/24 - l1**3*l3*q/24)*SingularityFunction(x, l1 + l2, -2))
    assert b.deflection() ==(l1**3*q*x/24 - l1*q*SingularityFunction(x, 0, 3)/12
                             - l1*q*SingularityFunction(x, l1, 3)/12 + q*SingularityFunction(x, 0, 4)/24
                             - q*SingularityFunction(x, l1, 4)/24
                             + (l1**3*l2*q/24 + l1**3*l3*q/24)*SingularityFunction(x, l1 + l2, 0))/(E*I)

def test_max_shear_force():
    E = Symbol('E')
    I = Symbol('I')

    b = Beam(3, E, I)
    R, M = symbols('R, M')
    b.apply_load(R, 0, -1)
    b.apply_load(M, 0, -2)
    b.apply_load(2, 3, -1)
    b.apply_load(4, 2, -1)
    b.apply_load(2, 2, 0, end=3)
    b.solve_for_reaction_loads(R, M)
    assert b.max_shear_force() == (Interval(0, 2), 8)

    l = symbols('l', positive=True)
    P = Symbol('P')
    b = Beam(l, E, I)
    R1, R2 = symbols('R1, R2')
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, l, -1)
    b.apply_load(P, 0, 0, end=l)
    b.solve_for_reaction_loads(R1, R2)
    max_shear = b.max_shear_force()
    assert max_shear[0] == 0
    assert simplify(max_shear[1] - (l*Abs(P)/2)) == 0


def test_max_bmoment():
    E = Symbol('E')
    I = Symbol('I')
    l, P = symbols('l, P', positive=True)

    b = Beam(l, E, I)
    R1, R2 = symbols('R1, R2')
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, l, -1)
    b.apply_load(P, l/2, -1)
    b.solve_for_reaction_loads(R1, R2)
    b.reaction_loads
    assert b.max_bmoment() == (l/2, P*l/4)

    b = Beam(l, E, I)
    R1, R2 = symbols('R1, R2')
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, l, -1)
    b.apply_load(P, 0, 0, end=l)
    b.solve_for_reaction_loads(R1, R2)
    assert b.max_bmoment() == (l/2, P*l**2/8)


def test_max_deflection():
    E, I, l, F = symbols('E, I, l, F', positive=True)
    b = Beam(l, E, I)
    b.bc_deflection = [(0, 0),(l, 0)]
    b.bc_slope = [(0, 0),(l, 0)]
    b.apply_load(F/2, 0, -1)
    b.apply_load(-F*l/8, 0, -2)
    b.apply_load(F/2, l, -1)
    b.apply_load(F*l/8, l, -2)
    b.apply_load(-F, l/2, -1)
    assert b.max_deflection() == (l/2, F*l**3/(192*E*I))

def test_solve_for_ild_reactions():
    E = Symbol('E')
    I = Symbol('I')
    b = Beam(10, E, I)
    b.apply_support(0, type="pin")
    b.apply_support(10, type="pin")
    R_0, R_10 = symbols('R_0, R_10')
    b.solve_for_ild_reactions(1, R_0, R_10)
    a = b.ild_variable
    assert b.ild_reactions == {R_0: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/10
                                    - SingularityFunction(a, 10, 1)/10,
                               R_10: -SingularityFunction(a, 0, 1)/10 + SingularityFunction(a, 10, 0)
                                     + SingularityFunction(a, 10, 1)/10}

    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    L = Symbol('L', positive=True)
    b = Beam(L, E, I)
    b.apply_support(L, type="fixed")
    b.apply_load(F, 0, -1)
    R_L, M_L = symbols('R_L, M_L')
    b.solve_for_ild_reactions(F, R_L, M_L)
    a = b.ild_variable
    assert b.ild_reactions == {R_L: -F*SingularityFunction(a, 0, 0) + F*SingularityFunction(a, L, 0) - F,
                               M_L: -F*L*SingularityFunction(a, 0, 0) - F*L + F*SingularityFunction(a, 0, 1)
                                    - F*SingularityFunction(a, L, 1)}

    E = Symbol('E')
    I = Symbol('I')
    b = Beam(20, E, I)
    r0 = b.apply_support(0, type="pin")
    r5 = b.apply_support(5, type="pin")
    r10 = b.apply_support(10, type="pin")
    r20, m20 = b.apply_support(20, type="fixed")
    b.solve_for_ild_reactions(1, r0, r5, r10, r20, m20)
    a = b.ild_variable
    assert b.ild_reactions[r0].subs(a, 4) == -Rational(59, 475)
    assert b.ild_reactions[r5].subs(a, 4) == -Rational(2296, 2375)
    assert b.ild_reactions[r10].subs(a, 4) == Rational(243, 2375)
    assert b.ild_reactions[r20].subs(a, 12) == -Rational(83, 475)
    assert b.ild_reactions[m20].subs(a, 12) == -Rational(264, 475)

def test_solve_for_ild_shear():
    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    L1 = Symbol('L1', positive=True)
    L2 = Symbol('L2', positive=True)
    b = Beam(L1 + L2, E, I)
    r0 = b.apply_support(0, type="pin")
    rL = b.apply_support(L1 + L2, type="pin")
    b.solve_for_ild_reactions(F, r0, rL)
    b.solve_for_ild_shear(L1, F, r0, rL)
    a = b.ild_variable
    expected_shear = (-F*L1*SingularityFunction(a, 0, 0)/(L1 + L2) - F*L2*SingularityFunction(a, 0, 0)/(L1 + L2)
                      - F*SingularityFunction(-a, 0, 0) + F*SingularityFunction(a, L1 + L2, 0) + F
                      + F*SingularityFunction(a, 0, 1)/(L1 + L2) - F*SingularityFunction(a, L1 + L2, 1)/(L1 + L2)
                      - (-F*L1*SingularityFunction(a, 0, 0)/(L1 + L2) + F*L1*SingularityFunction(a, L1 + L2, 0)/(L1 + L2)
                         - F*L2*SingularityFunction(a, 0, 0)/(L1 + L2) + F*L2*SingularityFunction(a, L1 + L2, 0)/(L1 + L2)
                         + 2*F)*SingularityFunction(a, L1, 0))
    assert b.ild_shear.expand() == expected_shear.expand()

    E = Symbol('E')
    I = Symbol('I')
    b = Beam(20, E, I)
    r0 = b.apply_support(0, type="pin")
    r5 = b.apply_support(5, type="pin")
    r10 = b.apply_support(10, type="pin")
    r20, m20 = b.apply_support(20, type="fixed")
    b.solve_for_ild_reactions(1, r0, r5, r10, r20, m20)
    b.solve_for_ild_shear(6, 1, r0, r5, r10, r20, m20)
    a = b.ild_variable
    assert b.ild_shear.subs(a, 12) == Rational(96, 475)
    assert b.ild_shear.subs(a, 4) == -Rational(216, 2375)

def test_solve_for_ild_moment():
    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    L1 = Symbol('L1', positive=True)
    L2 = Symbol('L2', positive=True)
    b = Beam(L1 + L2, E, I)
    r0 = b.apply_support(0, type="pin")
    rL = b.apply_support(L1 + L2, type="pin")
    a = b.ild_variable
    b.solve_for_ild_reactions(F, r0, rL)
    b.solve_for_ild_moment(L1, F, r0, rL)
    assert b.ild_moment.subs(a, 3).subs(L1, 5).subs(L2, 5) == -3*F/2

    E = Symbol('E')
    I = Symbol('I')
    b = Beam(20, E, I)
    r0 = b.apply_support(0, type="pin")
    r5 = b.apply_support(5, type="pin")
    r10 = b.apply_support(10, type="pin")
    r20, m20 = b.apply_support(20, type="fixed")
    b.solve_for_ild_reactions(1, r0, r5, r10, r20, m20)
    b.solve_for_ild_moment(5, 1, r0, r5, r10, r20, m20)
    assert b.ild_moment.subs(a, 12) == -Rational(96, 475)
    assert b.ild_moment.subs(a, 4) == Rational(36, 95)

def test_ild_with_rotation_hinge():
    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    L1 = Symbol('L1', positive=True)
    L2 = Symbol('L2', positive=True)
    L3 = Symbol('L3', positive=True)
    b = Beam(L1 + L2 + L3, E, I)
    r0 = b.apply_support(0, type="pin")
    r1 = b.apply_support(L1 + L2, type="pin")
    r2 = b.apply_support(L1 + L2 + L3, type="pin")
    b.apply_rotation_hinge(L1 + L2)
    b.solve_for_ild_reactions(F, r0, r1, r2)
    a = b.ild_variable
    assert b.ild_reactions[r0].subs(a, 4).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -3*F/5
    assert b.ild_reactions[r0].subs(a, -10).subs(L1, 5).subs(L2, 5).subs(L3, 10) == 0
    assert b.ild_reactions[r0].subs(a, 25).subs(L1, 5).subs(L2, 5).subs(L3, 10) == 0
    assert b.ild_reactions[r1].subs(a, 4).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -2*F/5
    assert b.ild_reactions[r2].subs(a, 18).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -4*F/5
    b.solve_for_ild_shear(L1, F, r0, r1, r2)
    assert b.ild_shear.subs(a, 7).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -3*F/10
    assert b.ild_shear.subs(a, 70).subs(L1, 5).subs(L2, 5).subs(L3, 10) == 0
    b.solve_for_ild_moment(L1, F, r0, r1, r2)
    assert b.ild_moment.subs(a, 1).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -F/2
    assert b.ild_moment.subs(a, 8).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -F

def test_ild_with_sliding_hinge():
    b = Beam(13, 200, 200)
    r0 = b.apply_support(0, type="pin")
    r6 = b.apply_support(6, type="pin")
    r13, m13 = b.apply_support(13, type="fixed")
    w3 = b.apply_sliding_hinge(3)
    b.solve_for_ild_reactions(1, r0, r6, r13, m13)
    a = b.ild_variable
    assert b.ild_reactions[r0].subs(a, 3) == -1
    assert b.ild_reactions[r6].subs(a, 3) == Rational(9, 14)
    assert b.ild_reactions[r13].subs(a, 9) == -Rational(207, 343)
    assert b.ild_reactions[m13].subs(a, 9) == -Rational(60, 49)
    assert b.ild_reactions[m13].subs(a, 15) == 0
    assert b.ild_reactions[m13].subs(a, -3) == 0
    assert b.ild_deflection_jumps[w3].subs(a, 9) == -Rational(9, 35000)
    b.solve_for_ild_shear(7, 1, r0, r6, r13, m13)
    assert b.ild_shear.subs(a, 8) == -Rational(200, 343)
    b.solve_for_ild_moment(8, 1, r0, r6, r13, m13)
    assert b.ild_moment.subs(a, 3) == -Rational(12, 7)

def test_Beam3D():
    l, E, G, I, A = symbols('l, E, G, I, A')
    R1, R2, R3, R4 = symbols('R1, R2, R3, R4')

    b = Beam3D(l, E, G, I, A)
    m, q = symbols('m, q')
    b.apply_load(q, 0, 0, dir="y")
    b.apply_moment_load(m, 0, 0, dir="z")
    b.bc_slope = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    b.bc_deflection = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    b.solve_slope_deflection()

    assert b.polar_moment() == 2*I
    assert b.shear_force() == [0, -q*x, 0]
    assert b.shear_stress() == [0, -q*x/A, 0]
    assert b.axial_stress() == 0
    assert b.bending_moment() == [0, 0, -m*x + q*x**2/2]
    expected_deflection = (x*(A*G*q*x**3/4 + A*G*x**2*(-l*(A*G*l*(l*q - 2*m) +
        12*E*I*q)/(A*G*l**2 + 12*E*I)/2 - m) + 3*E*I*l*(A*G*l*(l*q - 2*m) +
        12*E*I*q)/(A*G*l**2 + 12*E*I) + x*(-A*G*l**2*q/2 +
        3*A*G*l**2*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(A*G*l**2 + 12*E*I)/4 +
        A*G*l*m*Rational(3, 2) - 3*E*I*q))/(6*A*E*G*I))
    dx, dy, dz = b.deflection()
    assert dx == dz == 0
    assert simplify(dy - expected_deflection) == 0

    b2 = Beam3D(30, E, G, I, A, x)
    b2.apply_load(50, start=0, order=0, dir="y")
    b2.bc_deflection = [(0, [0, 0, 0]), (30, [0, 0, 0])]
    b2.apply_load(R1, start=0, order=-1, dir="y")
    b2.apply_load(R2, start=30, order=-1, dir="y")
    b2.solve_for_reaction_loads(R1, R2)
    assert b2.reaction_loads == {R1: -750, R2: -750}

    b2.solve_slope_deflection()
    assert b2.slope() == [0, 0, 25*x**3/(3*E*I) - 375*x**2/(E*I) + 3750*x/(E*I)]
    expected_deflection = 25*x**4/(12*E*I) - 125*x**3/(E*I) + 1875*x**2/(E*I) - \
        25*x**2/(A*G) + 750*x/(A*G)
    dx, dy, dz = b2.deflection()
    assert dx == dz == 0
    assert dy == expected_deflection

    # Test for solve_for_reaction_loads
    b3 = Beam3D(30, E, G, I, A, x)
    b3.apply_load(8, start=0, order=0, dir="y")
    b3.apply_load(9*x, start=0, order=0, dir="z")
    b3.apply_load(R1, start=0, order=-1, dir="y")
    b3.apply_load(R2, start=30, order=-1, dir="y")
    b3.apply_load(R3, start=0, order=-1, dir="z")
    b3.apply_load(R4, start=30, order=-1, dir="z")
    b3.solve_for_reaction_loads(R1, R2, R3, R4)
    assert b3.reaction_loads == {R1: -120, R2: -120, R3: -1350, R4: -2700}


def test_polar_moment_Beam3D():
    l, E, G, A, I1, I2 = symbols('l, E, G, A, I1, I2')
    I = [I1, I2]

    b = Beam3D(l, E, G, I, A)
    assert b.polar_moment() == I1 + I2


def test_parabolic_loads():

    E, I, L = symbols('E, I, L', positive=True, real=True)
    R, M, P = symbols('R, M, P', real=True)

    # cantilever beam fixed at x=0 and parabolic distributed loading across
    # length of beam
    beam = Beam(L, E, I)

    beam.bc_deflection.append((0, 0))
    beam.bc_slope.append((0, 0))
    beam.apply_load(R, 0, -1)
    beam.apply_load(M, 0, -2)

    # parabolic load
    beam.apply_load(1, 0, 2)

    beam.solve_for_reaction_loads(R, M)

    assert beam.reaction_loads[R] == -L**3/3

    # cantilever beam fixed at x=0 and parabolic distributed loading across
    # first half of beam
    beam = Beam(2*L, E, I)

    beam.bc_deflection.append((0, 0))
    beam.bc_slope.append((0, 0))
    beam.apply_load(R, 0, -1)
    beam.apply_load(M, 0, -2)

    # parabolic load from x=0 to x=L
    beam.apply_load(1, 0, 2, end=L)

    beam.solve_for_reaction_loads(R, M)

    # result should be the same as the prior example
    assert beam.reaction_loads[R] == -L**3/3

    # check constant load
    beam = Beam(2*L, E, I)
    beam.apply_load(P, 0, 0, end=L)
    loading = beam.load.xreplace({L: 10, E: 20, I: 30, P: 40})
    assert loading.xreplace({x: 5}) == 40
    assert loading.xreplace({x: 15}) == 0

    # check ramp load
    beam = Beam(2*L, E, I)
    beam.apply_load(P, 0, 1, end=L)
    assert beam.load == (P*SingularityFunction(x, 0, 1) -
                         P*SingularityFunction(x, L, 1) -
                         P*L*SingularityFunction(x, L, 0))

    # check higher order load: x**8 load from x=0 to x=L
    beam = Beam(2*L, E, I)
    beam.apply_load(P, 0, 8, end=L)
    loading = beam.load.xreplace({L: 10, E: 20, I: 30, P: 40})
    assert loading.xreplace({x: 5}) == 40*5**8
    assert loading.xreplace({x: 15}) == 0


def test_cross_section():
    I = Symbol('I')
    l = Symbol('l')
    E = Symbol('E')
    C3, C4 = symbols('C3, C4')
    a, c, g, h, r, n = symbols('a, c, g, h, r, n')

    # test for second_moment and cross_section setter
    b0 = Beam(l, E, I)
    assert b0.second_moment == I
    assert b0.cross_section == None
    b0.cross_section = Circle((0, 0), 5)
    assert b0.second_moment == pi*Rational(625, 4)
    assert b0.cross_section == Circle((0, 0), 5)
    b0.second_moment = 2*n - 6
    assert b0.second_moment == 2*n-6
    assert b0.cross_section == None
    with raises(ValueError):
        b0.second_moment = Circle((0, 0), 5)

    # beam with a circular cross-section
    b1 = Beam(50, E, Circle((0, 0), r))
    assert b1.cross_section == Circle((0, 0), r)
    assert b1.second_moment == pi*r*Abs(r)**3/4

    b1.apply_load(-10, 0, -1)
    b1.apply_load(R1, 5, -1)
    b1.apply_load(R2, 50, -1)
    b1.apply_load(90, 45, -2)
    b1.solve_for_reaction_loads(R1, R2)
    assert b1.load == (-10*SingularityFunction(x, 0, -1) + 82*SingularityFunction(x, 5, -1)/S(9)
                         + 90*SingularityFunction(x, 45, -2) + 8*SingularityFunction(x, 50, -1)/9)
    assert b1.bending_moment() == (10*SingularityFunction(x, 0, 1) - 82*SingularityFunction(x, 5, 1)/9
                                     - 90*SingularityFunction(x, 45, 0) - 8*SingularityFunction(x, 50, 1)/9)
    q = (-5*SingularityFunction(x, 0, 2) + 41*SingularityFunction(x, 5, 2)/S(9)
           + 90*SingularityFunction(x, 45, 1) + 4*SingularityFunction(x, 50, 2)/S(9))/(pi*E*r*Abs(r)**3)
    assert b1.slope() == C3 + 4*q
    q = (-5*SingularityFunction(x, 0, 3)/3 + 41*SingularityFunction(x, 5, 3)/27 + 45*SingularityFunction(x, 45, 2)
           + 4*SingularityFunction(x, 50, 3)/27)/(pi*E*r*Abs(r)**3)
    assert b1.deflection() == C3*x + C4 + 4*q

    # beam with a recatangular cross-section
    b2 = Beam(20, E, Polygon((0, 0), (a, 0), (a, c), (0, c)))
    assert b2.cross_section == Polygon((0, 0), (a, 0), (a, c), (0, c))
    assert b2.second_moment == a*c**3/12
    # beam with a triangular cross-section
    b3 = Beam(15, E, Triangle((0, 0), (g, 0), (g/2, h)))
    assert b3.cross_section == Triangle(Point2D(0, 0), Point2D(g, 0), Point2D(g/2, h))
    assert b3.second_moment == g*h**3/36

    # composite beam
    b = b2.join(b3, "fixed")
    b.apply_load(-30, 0, -1)
    b.apply_load(65, 0, -2)
    b.apply_load(40, 0, -1)
    b.bc_slope = [(0, 0)]
    b.bc_deflection = [(0, 0)]

    assert b.second_moment == Piecewise((a*c**3/12, x <= 20), (g*h**3/36, x <= 35))
    assert b.cross_section == None
    assert b.length == 35
    assert b.slope().subs(x, 7) == 8400/(E*a*c**3)
    assert b.slope().subs(x, 25) == 52200/(E*g*h**3) + 39600/(E*a*c**3)
    assert b.deflection().subs(x, 30) == -537000/(E*g*h**3) - 712000/(E*a*c**3)

def test_max_shear_force_Beam3D():
    x = symbols('x')
    b = Beam3D(20, 40, 21, 100, 25)
    b.apply_load(15, start=0, order=0, dir="z")
    b.apply_load(12*x, start=0, order=0, dir="y")
    b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
    assert b.max_shear_force() == [(0, 0), (20, 2400), (20, 300)]

def test_max_bending_moment_Beam3D():
    x = symbols('x')
    b = Beam3D(20, 40, 21, 100, 25)
    b.apply_load(15, start=0, order=0, dir="z")
    b.apply_load(12*x, start=0, order=0, dir="y")
    b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
    assert b.max_bmoment() == [(0, 0), (20, 3000), (20, 16000)]

def test_max_deflection_Beam3D():
    x = symbols('x')
    b = Beam3D(20, 40, 21, 100, 25)
    b.apply_load(15, start=0, order=0, dir="z")
    b.apply_load(12*x, start=0, order=0, dir="y")
    b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
    b.solve_slope_deflection()
    c = sympify("495/14")
    p = sympify("-10 + 10*sqrt(10793)/43")
    q = sympify("(10 - 10*sqrt(10793)/43)**3/160 - 20/7 + (10 - 10*sqrt(10793)/43)**4/6400 + 20*sqrt(10793)/301 + 27*(10 - 10*sqrt(10793)/43)**2/560")
    assert b.max_deflection() == [(0, 0), (10, c), (p, q)]

def test_torsion_Beam3D():
    x = symbols('x')
    b = Beam3D(20, 40, 21, 100, 25)
    b.apply_moment_load(15, 5, -2, dir='x')
    b.apply_moment_load(25, 10, -2, dir='x')
    b.apply_moment_load(-5, 20, -2, dir='x')
    b.solve_for_torsion()
    assert b.angular_deflection().subs(x, 3) == sympify("1/40")
    assert b.angular_deflection().subs(x, 9) == sympify("17/280")
    assert b.angular_deflection().subs(x, 12) == sympify("53/840")
    assert b.angular_deflection().subs(x, 17) == sympify("2/35")
    assert b.angular_deflection().subs(x, 20) == sympify("3/56")
