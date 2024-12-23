from sympy.core.symbol import Symbol, symbols
from sympy.physics.continuum_mechanics.truss import Truss
from sympy import sqrt


def test_truss():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    AB, BC, AC = symbols('AB, BC, AC')
    P = Symbol('P')

    t = Truss()
    assert t.nodes == []
    assert t.node_labels == []
    assert t.node_positions == []
    assert t.members == {}
    assert t.loads == {}
    assert t.supports == {}
    assert t.reaction_loads == {}
    assert t.internal_forces == {}

    # testing the add_node method
    t.add_node((A, 0, 0), (B, 2, 2), (C, 3, 0))
    assert t.nodes == [(A, 0, 0), (B, 2, 2), (C, 3, 0)]
    assert t.node_labels == [A, B, C]
    assert t.node_positions == [(0, 0), (2, 2), (3, 0)]
    assert t.loads == {}
    assert t.supports == {}
    assert t.reaction_loads == {}

    # testing the remove_node method
    t.remove_node(C)
    assert t.nodes == [(A, 0, 0), (B, 2, 2)]
    assert t.node_labels == [A, B]
    assert t.node_positions == [(0, 0), (2, 2)]
    assert t.loads == {}
    assert t.supports == {}

    t.add_node((C, 3, 0))

    # testing the add_member method
    t.add_member((AB, A, B), (BC, B, C), (AC, A, C))
    assert t.members == {AB: [A, B], BC: [B, C], AC: [A, C]}
    assert t.internal_forces == {AB: 0, BC: 0, AC: 0}

    # testing the remove_member method
    t.remove_member(BC)
    assert t.members == {AB: [A, B], AC: [A, C]}
    assert t.internal_forces == {AB: 0, AC: 0}

    t.add_member((BC, B, C))

    D, CD = symbols('D, CD')

    # testing the change_label methods
    t.change_node_label((B, D))
    assert t.nodes == [(A, 0, 0), (D, 2, 2), (C, 3, 0)]
    assert t.node_labels == [A, D, C]
    assert t.loads == {}
    assert t.supports == {}
    assert t.members == {AB: [A, D], BC: [D, C], AC: [A, C]}

    t.change_member_label((BC, CD))
    assert t.members == {AB: [A, D], CD: [D, C], AC: [A, C]}
    assert t.internal_forces == {AB: 0, CD: 0, AC: 0}


    # testing the apply_load method
    t.apply_load((A, P, 90), (A, P/4, 90), (A, 2*P,45), (D, P/2, 90))
    assert t.loads == {A: [[P, 90], [P/4, 90], [2*P, 45]], D: [[P/2, 90]]}
    assert t.loads[A] == [[P, 90], [P/4, 90], [2*P, 45]]

    # testing the remove_load method
    t.remove_load((A, P/4, 90))
    assert t.loads == {A: [[P, 90], [2*P, 45]], D: [[P/2, 90]]}
    assert t.loads[A] == [[P, 90], [2*P, 45]]

    # testing the apply_support method
    t.apply_support((A, "pinned"), (D, "roller"))
    assert t.supports == {A: 'pinned', D: 'roller'}
    assert t.reaction_loads == {}
    assert t.loads == {A: [[P, 90], [2*P, 45], [Symbol('R_A_x'), 0], [Symbol('R_A_y'), 90]],  D: [[P/2, 90], [Symbol('R_D_y'), 90]]}

    # testing the remove_support method
    t.remove_support(A)
    assert t.supports == {D: 'roller'}
    assert t.reaction_loads == {}
    assert t.loads == {A: [[P, 90], [2*P, 45]], D: [[P/2, 90], [Symbol('R_D_y'), 90]]}

    t.apply_support((A, "pinned"))

    # testing the solve method
    t.solve()
    assert t.reaction_loads['R_A_x'] == -sqrt(2)*P
    assert t.reaction_loads['R_A_y'] == -sqrt(2)*P - P
    assert t.reaction_loads['R_D_y'] == -P/2
    assert t.internal_forces[AB]/P == 0
    assert t.internal_forces[CD] == 0
    assert t.internal_forces[AC] == 0
