from sympy.testing.pytest import raises
from sympy.matrices.exceptions import NonSquareMatrixError, NonInvertibleMatrixError

from sympy import Matrix, Rational


def test_lll():
    A = Matrix([[1, 0, 0, 0, -20160],
                [0, 1, 0, 0, 33768],
                [0, 0, 1, 0, 39578],
                [0, 0, 0, 1, 47757]])
    L = Matrix([[ 10, -3,  -2,  8,  -4],
                [  3, -9,   8,  1, -11],
                [ -3, 13,  -9, -3,  -9],
                [-12, -7, -11,  9,  -1]])
    T = Matrix([[ 10, -3,  -2,  8],
                [  3, -9,   8,  1],
                [ -3, 13,  -9, -3],
                [-12, -7, -11,  9]])
    assert A.lll() == L
    assert A.lll_transform() == (L, T)
    assert T * A == L


def test_matrix_inv_mod():
    A = Matrix(2, 1, [1, 0])
    raises(NonSquareMatrixError, lambda: A.inv_mod(2))
    A = Matrix(2, 2, [1, 0, 0, 0])
    raises(NonInvertibleMatrixError, lambda: A.inv_mod(2))
    A = Matrix(2, 2, [1, 2, 3, 4])
    Ai = Matrix(2, 2, [1, 1, 0, 1])
    assert A.inv_mod(3) == Ai
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert A.inv_mod(2) == A
    A = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    raises(NonInvertibleMatrixError, lambda: A.inv_mod(5))
    A = Matrix(3, 3, [5, 1, 3, 2, 6, 0, 2, 1, 1])
    Ai = Matrix(3, 3, [6, 8, 0, 1, 5, 6, 5, 6, 4])
    assert A.inv_mod(9) == Ai
    A = Matrix(3, 3, [1, 6, -3, 4, 1, -5, 3, -5, 5])
    Ai = Matrix(3, 3, [4, 3, 3, 1, 2, 5, 1, 5, 1])
    assert A.inv_mod(6) == Ai
    A = Matrix(3, 3, [1, 6, 1, 4, 1, 5, 3, 2, 5])
    Ai = Matrix(3, 3, [6, 0, 3, 6, 6, 4, 1, 6, 1])
    assert A.inv_mod(7) == Ai
    A = Matrix([[1, 2], [3, Rational(3,4)]])
    raises(ValueError, lambda: A.inv_mod(2))
    A = Matrix([[1, 2], [3, 4]])
    raises(TypeError, lambda: A.inv_mod(Rational(1, 2)))
    # https://github.com/sympy/sympy/issues/27663
    M = Matrix([
        [2, 3, 1, 4],
        [1, 5, 3, 2],
        [3, 2, 4, 1],
        [4, 1, 2, 5],
    ])
    assert M.inv_mod(26) == Matrix([
        [7, 21, 10, 10],
        [1, 7, 19, 3],
        [14, 1, 15, 1],
        [25, 23, 3, 12],
    ])
