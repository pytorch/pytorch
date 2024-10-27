from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

def test_type_A():
    c = CartanType("A3")
    m = Matrix(3, 3, [2, -1, 0, -1, 2, -1, 0, -1, 2])
    assert m == c.cartan_matrix()
    assert c.basis() == 8
    assert c.roots() == 12
    assert c.dimension() == 4
    assert c.simple_root(1) == [1, -1, 0, 0]
    assert c.highest_root() == [1, 0, 0, -1]
    assert c.lie_algebra() == "su(4)"
    diag = "0---0---0\n1   2   3"
    assert c.dynkin_diagram() == diag
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 0, -1, 0],
            3: [1, 0, 0, -1], 4: [0, 1, -1, 0], 5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
