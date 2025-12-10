from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix
from sympy.core.backend import S

def test_type_F():
    c = CartanType("F4")
    m = Matrix(4, 4, [2, -1, 0, 0, -1, 2, -2, 0, 0, -1, 2, -1, 0, 0, -1, 2])
    assert c.cartan_matrix() == m
    assert c.dimension() == 4
    assert c.simple_root(1) == [1, -1, 0, 0]
    assert c.simple_root(2) == [0, 1, -1, 0]
    assert c.simple_root(3) == [0, 0, 0, 1]
    assert c.simple_root(4) == [-S.Half, -S.Half, -S.Half, -S.Half]
    assert c.roots() == 48
    assert c.basis() == 52
    diag = "0---0=>=0---0\n" + "   ".join(str(i) for i in range(1, 5))
    assert c.dynkin_diagram() == diag
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 1, 0, 0], 3: [1, 0, -1, 0],
            4: [1, 0, 1, 0], 5: [1, 0, 0, -1], 6: [1, 0, 0, 1], 7: [0, 1, -1, 0],
            8: [0, 1, 1, 0], 9: [0, 1, 0, -1], 10: [0, 1, 0, 1], 11: [0, 0, 1, -1],
            12: [0, 0, 1, 1], 13: [1, 0, 0, 0], 14: [0, 1, 0, 0], 15: [0, 0, 1, 0],
            16: [0, 0, 0, 1], 17: [S.Half, S.Half, S.Half, S.Half], 18: [S.Half, -S.Half, S.Half, S.Half],
            19: [S.Half, S.Half, -S.Half, S.Half], 20: [S.Half, S.Half, S.Half, -S.Half], 21: [S.Half, S.Half, -S.Half, -S.Half],
            22: [S.Half, -S.Half, S.Half, -S.Half], 23: [S.Half, -S.Half, -S.Half, S.Half], 24: [S.Half, -S.Half, -S.Half, -S.Half]}
