from sympy.physics.continuum_mechanics.arch import Arch
from sympy import Symbol, simplify

x = Symbol('x')
t = Symbol('t')

def test_arch_init():
    a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
    assert a.get_loads == {'distributed': {}, 'concentrated': {}}
    assert a.reaction_force == {Symbol('R_A_x'):0, Symbol('R_A_y'):0, Symbol('R_B_x'):0, Symbol('R_B_y'):0}
    assert a.supports == {'left':'hinge', 'right':'hinge'}
    assert a.left_support == (0,0)
    assert a.right_support == (10,0)
    assert a.get_shape_eqn == 5 - ((x-5)**2)/5

    a = Arch((0,0),(10,1),crown_x=6)
    a.change_support_type(left_support='roller')
    a.add_member(0.5)
    assert a.supports == {'left':'roller', 'right':'hinge'}
    assert simplify(a.get_shape_eqn) == simplify(9/5 - (x - 6)**2/20)

def test_arch_support():
    a = Arch((0,0),(40,0),crown_x=20,crown_y=12)
    a.apply_load(-1,'C',8,150,angle=270)
    a.apply_load(0,'D',start=20,end=40,mag=-4)
    a.solve()
    assert abs(a.reaction_force[Symbol("R_A_x")] - 83.33333333333333) < 10e-12
    assert abs(a.reaction_force[Symbol("R_B_y")] - 90.00000000000000) < 10e-12
    assert abs(a.reaction_force[Symbol("R_B_x")] + 83.33333333333333) < 10e-12
    assert abs(a.reaction_force[Symbol("R_A_y")] - 140.00000000000000) < 10e-12

def test_arch_member():
    a = Arch((0,0),(40,0),crown_x=20,crown_y=15)
    a.change_support_type(right_support='roller')
    a.add_member(0)
    a.apply_load(-1,'D',start=12,mag=3,angle=270)
    a.apply_load(-1,'E',start=6,mag=4,angle=270)
    a.apply_load(-1,'C',start=30,mag=5,angle=270)
    a.solve()
    assert a.reaction_force[Symbol("R_A_x")] == 0
    assert abs(a.reaction_force[Symbol("R_A_y")] - 6.750000000000000) < 10e-12
    assert a.reaction_force[Symbol("R_B_x")] == 0
    assert abs(a.reaction_force[Symbol("R_B_y")] - 5.250000000000000) < 10e-12

def test_symbol_magnitude():
    a = Arch((0,0),(16,0),crown_x=8,crown_y=5)
    a.apply_load(0,'C',start=3,end=5,mag=t)
    a.solve()
    assert a.reaction_force[Symbol("R_A_x")] == -(4*t)/5
    assert a.reaction_force[Symbol("R_A_y")] == -(3*t)/2
    assert a.reaction_force[Symbol("R_B_x")] == (4*t)/5
    assert a.reaction_force[Symbol("R_B_y")] == -t/2
    assert a.bending_moment_at(4) == -5*t/2

def test_forces():
    a = Arch((0,0),(40,0),crown_x=20,crown_y=12)
    a.apply_load(-1,'C',8,150,angle=270)
    a.apply_load(0,'D',start=20,end=40,mag=-4)
    a.solve()
    assert abs(a.axial_force_at(7.999999999999999)-149.430523405935) < 1e-12
    assert abs(a.shear_force_at(7.999999999999999)-64.9227473161196) < 1e-12
