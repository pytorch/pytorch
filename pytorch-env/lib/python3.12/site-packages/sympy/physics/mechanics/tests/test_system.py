from sympy import symbols, Matrix, atan, zeros
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, Particle, Point,
                                     ReferenceFrame, SymbolicSystem)
from sympy.testing.pytest import raises

# This class is going to be tested using a simple pendulum set up in x and y
# coordinates
x, y, u, v, lam = dynamicsymbols('x y u v lambda')
m, l, g = symbols('m l g')

# Set up the different forms the equations can take
#       [1] Explicit form where the kinematics and dynamics are combined
#           x' = F(x, t, r, p)
#
#       [2] Implicit form where the kinematics and dynamics are combined
#           M(x, p) x' = F(x, t, r, p)
#
#       [3] Implicit form where the kinematics and dynamics are separate
#           M(q, p) u' = F(q, u, t, r, p)
#           q' = G(q, u, t, r, p)
dyn_implicit_mat = Matrix([[1, 0, -x/m],
                           [0, 1, -y/m],
                           [0, 0, l**2/m]])

dyn_implicit_rhs = Matrix([0, 0, u**2 + v**2 - g*y])

comb_implicit_mat = Matrix([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, -x/m],
                            [0, 0, 0, 1, -y/m],
                            [0, 0, 0, 0, l**2/m]])

comb_implicit_rhs = Matrix([u, v, 0, 0, u**2 + v**2 - g*y])

kin_explicit_rhs = Matrix([u, v])

comb_explicit_rhs = comb_implicit_mat.LUsolve(comb_implicit_rhs)

# Set up a body and load to pass into the system
theta = atan(x/y)
N = ReferenceFrame('N')
A = N.orientnew('A', 'Axis', [theta, N.z])
O = Point('O')
P = O.locatenew('P', l * A.x)

Pa = Particle('Pa', P, m)

bodies = [Pa]
loads = [(P, g * m * N.x)]

# Set up some output equations to be given to SymbolicSystem
# Change to make these fit the pendulum
PE = symbols("PE")
out_eqns = {PE: m*g*(l+y)}

# Set up remaining arguments that can be passed to SymbolicSystem
alg_con = [2]
alg_con_full = [4]
coordinates = (x, y, lam)
speeds = (u, v)
states = (x, y, u, v, lam)
coord_idxs = (0, 1)
speed_idxs = (2, 3)


def test_form_1():
    symsystem1 = SymbolicSystem(states, comb_explicit_rhs,
                                alg_con=alg_con_full, output_eqns=out_eqns,
                                coord_idxs=coord_idxs, speed_idxs=speed_idxs,
                                bodies=bodies, loads=loads)

    assert symsystem1.coordinates == Matrix([x, y])
    assert symsystem1.speeds == Matrix([u, v])
    assert symsystem1.states == Matrix([x, y, u, v, lam])

    assert symsystem1.alg_con == [4]

    inter = comb_explicit_rhs
    assert simplify(symsystem1.comb_explicit_rhs - inter) == zeros(5, 1)

    assert set(symsystem1.dynamic_symbols()) == {y, v, lam, u, x}
    assert type(symsystem1.dynamic_symbols()) == tuple
    assert set(symsystem1.constant_symbols()) == {l, g, m}
    assert type(symsystem1.constant_symbols()) == tuple

    assert symsystem1.output_eqns == out_eqns

    assert symsystem1.bodies == (Pa,)
    assert symsystem1.loads == ((P, g * m * N.x),)


def test_form_2():
    symsystem2 = SymbolicSystem(coordinates, comb_implicit_rhs, speeds=speeds,
                                mass_matrix=comb_implicit_mat,
                                alg_con=alg_con_full, output_eqns=out_eqns,
                                bodies=bodies, loads=loads)

    assert symsystem2.coordinates == Matrix([x, y, lam])
    assert symsystem2.speeds == Matrix([u, v])
    assert symsystem2.states == Matrix([x, y, lam, u, v])

    assert symsystem2.alg_con == [4]

    inter = comb_implicit_rhs
    assert simplify(symsystem2.comb_implicit_rhs - inter) == zeros(5, 1)
    assert simplify(symsystem2.comb_implicit_mat-comb_implicit_mat) == zeros(5)

    assert set(symsystem2.dynamic_symbols()) == {y, v, lam, u, x}
    assert type(symsystem2.dynamic_symbols()) == tuple
    assert set(symsystem2.constant_symbols()) == {l, g, m}
    assert type(symsystem2.constant_symbols()) == tuple

    inter = comb_explicit_rhs
    symsystem2.compute_explicit_form()
    assert simplify(symsystem2.comb_explicit_rhs - inter) == zeros(5, 1)


    assert symsystem2.output_eqns == out_eqns

    assert symsystem2.bodies == (Pa,)
    assert symsystem2.loads == ((P, g * m * N.x),)


def test_form_3():
    symsystem3 = SymbolicSystem(states, dyn_implicit_rhs,
                                mass_matrix=dyn_implicit_mat,
                                coordinate_derivatives=kin_explicit_rhs,
                                alg_con=alg_con, coord_idxs=coord_idxs,
                                speed_idxs=speed_idxs, bodies=bodies,
                                loads=loads)

    assert symsystem3.coordinates == Matrix([x, y])
    assert symsystem3.speeds == Matrix([u, v])
    assert symsystem3.states == Matrix([x, y, u, v, lam])

    assert symsystem3.alg_con == [4]

    inter1 = kin_explicit_rhs
    inter2 = dyn_implicit_rhs
    assert simplify(symsystem3.kin_explicit_rhs - inter1) == zeros(2, 1)
    assert simplify(symsystem3.dyn_implicit_mat - dyn_implicit_mat) == zeros(3)
    assert simplify(symsystem3.dyn_implicit_rhs - inter2) == zeros(3, 1)

    inter = comb_implicit_rhs
    assert simplify(symsystem3.comb_implicit_rhs - inter) == zeros(5, 1)
    assert simplify(symsystem3.comb_implicit_mat-comb_implicit_mat) == zeros(5)

    inter = comb_explicit_rhs
    symsystem3.compute_explicit_form()
    assert simplify(symsystem3.comb_explicit_rhs - inter) == zeros(5, 1)

    assert set(symsystem3.dynamic_symbols()) == {y, v, lam, u, x}
    assert type(symsystem3.dynamic_symbols()) == tuple
    assert set(symsystem3.constant_symbols()) == {l, g, m}
    assert type(symsystem3.constant_symbols()) == tuple

    assert symsystem3.output_eqns == {}

    assert symsystem3.bodies == (Pa,)
    assert symsystem3.loads == ((P, g * m * N.x),)


def test_property_attributes():
    symsystem = SymbolicSystem(states, comb_explicit_rhs,
                               alg_con=alg_con_full, output_eqns=out_eqns,
                               coord_idxs=coord_idxs, speed_idxs=speed_idxs,
                               bodies=bodies, loads=loads)

    with raises(AttributeError):
        symsystem.bodies = 42
    with raises(AttributeError):
        symsystem.coordinates = 42
    with raises(AttributeError):
        symsystem.dyn_implicit_rhs = 42
    with raises(AttributeError):
        symsystem.comb_implicit_rhs = 42
    with raises(AttributeError):
        symsystem.loads = 42
    with raises(AttributeError):
        symsystem.dyn_implicit_mat = 42
    with raises(AttributeError):
        symsystem.comb_implicit_mat = 42
    with raises(AttributeError):
        symsystem.kin_explicit_rhs = 42
    with raises(AttributeError):
        symsystem.comb_explicit_rhs = 42
    with raises(AttributeError):
        symsystem.speeds = 42
    with raises(AttributeError):
        symsystem.states = 42
    with raises(AttributeError):
        symsystem.alg_con = 42


def test_not_specified_errors():
    """This test will cover errors that arise from trying to access attributes
    that were not specified upon object creation or were specified on creation
    and the user tries to recalculate them."""
    # Trying to access form 2 when form 1 given
    # Trying to access form 3 when form 2 given

    symsystem1 = SymbolicSystem(states, comb_explicit_rhs)

    with raises(AttributeError):
        symsystem1.comb_implicit_mat
    with raises(AttributeError):
        symsystem1.comb_implicit_rhs
    with raises(AttributeError):
        symsystem1.dyn_implicit_mat
    with raises(AttributeError):
        symsystem1.dyn_implicit_rhs
    with raises(AttributeError):
        symsystem1.kin_explicit_rhs
    with raises(AttributeError):
        symsystem1.compute_explicit_form()

    symsystem2 = SymbolicSystem(coordinates, comb_implicit_rhs, speeds=speeds,
                                mass_matrix=comb_implicit_mat)

    with raises(AttributeError):
        symsystem2.dyn_implicit_mat
    with raises(AttributeError):
        symsystem2.dyn_implicit_rhs
    with raises(AttributeError):
        symsystem2.kin_explicit_rhs

    # Attribute error when trying to access coordinates and speeds when only the
    # states were given.
    with raises(AttributeError):
        symsystem1.coordinates
    with raises(AttributeError):
        symsystem1.speeds

    # Attribute error when trying to access bodies and loads when they are not
    # given
    with raises(AttributeError):
        symsystem1.bodies
    with raises(AttributeError):
        symsystem1.loads

    # Attribute error when trying to access comb_explicit_rhs before it was
    # calculated
    with raises(AttributeError):
        symsystem2.comb_explicit_rhs
