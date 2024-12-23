from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

def test_type_B():
    c = CartanType("B3")
    m = Matrix(3, 3, [2, -1, 0, -1, 2, -2, 0, -1, 2])
    assert m == c.cartan_matrix()
    assert c.dimension() == 3
    assert c.roots() == 18
    assert c.simple_root(3) == [0, 0, 1]
    assert c.basis() == 3
    assert c.lie_algebra() == "so(6)"
    diag = "0---0=>=0\n1   2   3"
    assert c.dynkin_diagram() == diag
    assert c.positive_roots() ==  {1: [1, -1, 0], 2: [1, 1, 0], 3: [1, 0, -1],
            4: [1, 0, 1], 5: [0, 1, -1], 6: [0, 1, 1], 7: [1, 0, 0],
            8: [0, 1, 0], 9: [0, 0, 1]}
