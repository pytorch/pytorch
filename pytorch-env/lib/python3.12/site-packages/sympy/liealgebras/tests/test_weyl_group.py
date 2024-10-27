from sympy.liealgebras.weyl_group import WeylGroup
from sympy.matrices import Matrix

def test_weyl_group():
    c = WeylGroup("A3")
    assert c.matrix_form('r1*r2') == Matrix([[0, 0, 1, 0], [1, 0, 0, 0],
        [0, 1, 0, 0], [0, 0, 0, 1]])
    assert c.generators() == ['r1', 'r2', 'r3']
    assert c.group_order() == 24.0
    assert c.group_name() == "S4: the symmetric group acting on 4 elements."
    assert c.coxeter_diagram() == "0---0---0\n1   2   3"
    assert c.element_order('r1*r2*r3') == 4
    assert c.element_order('r1*r3*r2*r3') == 3
    d = WeylGroup("B5")
    assert d.group_order() == 3840
    assert d.element_order('r1*r2*r4*r5') == 12
    assert d.matrix_form('r2*r3') ==  Matrix([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    assert d.element_order('r1*r2*r1*r3*r5') == 6
    e = WeylGroup("D5")
    assert e.element_order('r2*r3*r5') == 4
    assert e.matrix_form('r2*r3*r5') == Matrix([[1, 0, 0, 0, 0], [0, 0, 0, 0, -1],
        [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0]])
    f = WeylGroup("G2")
    assert f.element_order('r1*r2*r1*r2') == 3
    assert f.element_order('r2*r1*r1*r2') == 1

    assert f.matrix_form('r1*r2*r1*r2') == Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    g = WeylGroup("F4")
    assert g.matrix_form('r2*r3') == Matrix([[1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, -1], [0, 0, 1, 0]])

    assert g.element_order('r2*r3') == 4
    h = WeylGroup("E6")
    assert h.group_order() == 51840
