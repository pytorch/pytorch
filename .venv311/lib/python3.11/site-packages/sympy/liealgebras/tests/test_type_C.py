from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

def test_type_C():
    c = CartanType("C4")
    m = Matrix(4, 4, [2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -2, 2])
    assert c.cartan_matrix() == m
    assert c.dimension() == 4
    assert c.simple_root(4) == [0, 0, 0, 2]
    assert c.roots() == 32
    assert c.basis() == 36
    assert c.lie_algebra() == "sp(8)"
    t = CartanType(['C', 3])
    assert t.dimension() == 3
    diag = "0---0---0=<=0\n1   2   3   4"
    assert c.dynkin_diagram() == diag
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 1, 0, 0],
            3: [1, 0, -1, 0], 4: [1, 0, 1, 0], 5: [1, 0, 0, -1],
            6: [1, 0, 0, 1], 7: [0, 1, -1, 0], 8: [0, 1, 1, 0],
            9: [0, 1, 0, -1], 10: [0, 1, 0, 1], 11: [0, 0, 1, -1],
            12: [0, 0, 1, 1], 13: [2, 0, 0, 0], 14: [0, 2, 0, 0], 15: [0, 0, 2, 0],
            16: [0, 0, 0, 2]}
