from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix



def test_type_D():
    c = CartanType("D4")
    m = Matrix(4, 4, [2, -1, 0, 0, -1, 2, -1, -1, 0, -1, 2, 0, 0, -1, 0, 2])
    assert c.cartan_matrix() == m
    assert c.basis() == 6
    assert c.lie_algebra() == "so(8)"
    assert c.roots() == 24
    assert c.simple_root(3) == [0, 0, 1, -1]
    diag = "    3\n    0\n    |\n    |\n0---0---0\n1   2   4"
    assert diag == c.dynkin_diagram()
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 1, 0, 0],
            3: [1, 0, -1, 0], 4: [1, 0, 1, 0], 5: [1, 0, 0, -1], 6: [1, 0, 0, 1],
            7: [0, 1, -1, 0], 8: [0, 1, 1, 0], 9: [0, 1, 0, -1], 10: [0, 1, 0, 1],
            11: [0, 0, 1, -1], 12: [0, 0, 1, 1]}
