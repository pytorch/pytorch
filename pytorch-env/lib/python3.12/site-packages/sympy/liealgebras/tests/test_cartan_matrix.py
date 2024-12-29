from sympy.liealgebras.cartan_matrix import CartanMatrix
from sympy.matrices import Matrix

def test_CartanMatrix():
    c = CartanMatrix("A3")
    m = Matrix(3, 3, [2, -1, 0, -1, 2, -1, 0, -1, 2])
    assert c == m
    a = CartanMatrix(["G",2])
    mt = Matrix(2, 2, [2, -1, -3, 2])
    assert a == mt
