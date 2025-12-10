from sympy.physics.continuum_mechanics.cable import Cable
from sympy.core.symbol import Symbol


def test_cable():
    c = Cable(('A', 0, 10), ('B', 10, 10))
    assert c.supports == {'A': [0, 10], 'B': [10, 10]}
    assert c.left_support == [0, 10]
    assert c.right_support == [10, 10]
    assert c.loads == {'distributed': {}, 'point_load': {}}
    assert c.loads_position == {}
    assert c.length == 0
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    # tests for change_support method
    c.change_support('A', ('C', 12, 3))
    assert c.supports == {'B': [10, 10], 'C': [12, 3]}
    assert c.left_support == [10, 10]
    assert c.right_support == [12, 3]
    assert c.reaction_loads == {Symbol("R_B_x"): 0, Symbol("R_B_y"): 0, Symbol("R_C_x"): 0, Symbol("R_C_y"): 0}

    c.change_support('C', ('A', 0, 10))

    # tests for apply_load method for point loads
    c.apply_load(-1, ('X', 2, 5, 3, 30))
    c.apply_load(-1, ('Y', 5, 8, 5, 60))
    assert c.loads == {'distributed': {}, 'point_load': {'X': [3, 30], 'Y': [5, 60]}}
    assert c.loads_position == {'X': [2, 5], 'Y': [5, 8]}
    assert c.length == 0
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    # tests for remove_loads method
    c.remove_loads('X')
    assert c.loads == {'distributed': {}, 'point_load': {'Y': [5, 60]}}
    assert c.loads_position == {'Y': [5, 8]}
    assert c.length == 0
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    c.remove_loads('Y')

    #tests for apply_load method for distributed load
    c.apply_load(0, ('Z', 9))
    assert c.loads == {'distributed': {'Z': 9}, 'point_load': {}}
    assert c.loads_position == {}
    assert c.length == 0
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    # tests for apply_length method
    c.apply_length(20)
    assert c.length == 20

    del c
    # tests for solve method
    # for point loads
    c = Cable(("A", 0, 10), ("B", 5.5, 8))
    c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
    c.apply_load(-1, ('X', 4, 6, 8, 270))
    c.solve()
    #assert c.tension == {Symbol("Z_X"): 4.79150773600774, Symbol("X_B"): 6.78571428571429, Symbol("A_Z"): 6.89488895397307}
    assert abs(c.tension[Symbol("A_Z")] - 6.89488895397307) < 10e-12
    assert abs(c.tension[Symbol("Z_X")] - 4.79150773600774) < 10e-12
    assert abs(c.tension[Symbol("X_B")] - 6.78571428571429) < 10e-12
    #assert c.reaction_loads == {Symbol("R_A_x"): -4.06504065040650, Symbol("R_A_y"): 5.56910569105691, Symbol("R_B_x"): 4.06504065040650, Symbol("R_B_y"): 5.43089430894309}
    assert abs(c.reaction_loads[Symbol("R_A_x")] + 4.06504065040650) < 10e-12
    assert abs(c.reaction_loads[Symbol("R_A_y")] - 5.56910569105691) < 10e-12
    assert abs(c.reaction_loads[Symbol("R_B_x")] - 4.06504065040650) < 10e-12
    assert abs(c.reaction_loads[Symbol("R_B_y")] - 5.43089430894309) < 10e-12
    assert abs(c.length - 8.25609584845190) < 10e-12

    del c
    # tests for solve method
    # for distributed loads
    c=Cable(("A", 0, 40),("B", 100, 20))
    c.apply_load(0, ("X", 850))
    c.solve(58.58, 0)

    # assert c.tension['distributed'] == 36456.8485*sqrt(0.000543529004799705*(X + 0.00135624381275735)**2 + 1)
    assert abs(c.tension_at(0) - 61717.4130533677) < 10e-11
    assert abs(c.tension_at(40) - 39738.0809048449) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_A_x")] - 36465.0000000000) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_A_y")] + 49793.0000000000) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_B_x")] - 44399.9537590861) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_B_y")] - 42868.2071025955 ) < 10e-11
