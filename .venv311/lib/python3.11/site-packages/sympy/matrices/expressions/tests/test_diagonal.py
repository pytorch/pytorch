from sympy.matrices.expressions import MatrixSymbol
from sympy.matrices.expressions.diagonal import DiagonalMatrix, DiagonalOf, DiagMatrix, diagonalize_vector
from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import Identity
from sympy.testing.pytest import raises


n = Symbol('n')
m = Symbol('m')


def test_DiagonalMatrix():
    x = MatrixSymbol('x', n, m)
    D = DiagonalMatrix(x)
    assert D.diagonal_length is None
    assert D.shape == (n, m)

    x = MatrixSymbol('x', n, n)
    D = DiagonalMatrix(x)
    assert D.diagonal_length == n
    assert D.shape == (n, n)
    assert D[1, 2] == 0
    assert D[1, 1] == x[1, 1]
    i = Symbol('i')
    j = Symbol('j')
    x = MatrixSymbol('x', 3, 3)
    ij = DiagonalMatrix(x)[i, j]
    assert ij != 0
    assert ij.subs({i:0, j:0}) == x[0, 0]
    assert ij.subs({i:0, j:1}) == 0
    assert ij.subs({i:1, j:1}) == x[1, 1]
    assert ask(Q.diagonal(D))  # affirm that D is diagonal

    x = MatrixSymbol('x', n, 3)
    D = DiagonalMatrix(x)
    assert D.diagonal_length == 3
    assert D.shape == (n, 3)
    assert D[2, m] == KroneckerDelta(2, m)*x[2, m]
    assert D[3, m] == 0
    raises(IndexError, lambda: D[m, 3])

    x = MatrixSymbol('x', 3, n)
    D = DiagonalMatrix(x)
    assert D.diagonal_length == 3
    assert D.shape == (3, n)
    assert D[m, 2] == KroneckerDelta(m, 2)*x[m, 2]
    assert D[m, 3] == 0
    raises(IndexError, lambda: D[3, m])

    x = MatrixSymbol('x', n, m)
    D = DiagonalMatrix(x)
    assert D.diagonal_length is None
    assert D.shape == (n, m)
    assert D[m, 4] != 0

    x = MatrixSymbol('x', 3, 4)
    assert [DiagonalMatrix(x)[i] for i in range(12)] == [
        x[0, 0], 0, 0, 0, 0, x[1, 1], 0, 0, 0, 0, x[2, 2], 0]

    # shape is retained, issue 12427
    assert (
        DiagonalMatrix(MatrixSymbol('x', 3, 4))*
        DiagonalMatrix(MatrixSymbol('x', 4, 2))).shape == (3, 2)


def test_DiagonalOf():
    x = MatrixSymbol('x', n, n)
    d = DiagonalOf(x)
    assert d.shape == (n, 1)
    assert d.diagonal_length == n
    assert d[2, 0] == d[2] == x[2, 2]

    x = MatrixSymbol('x', n, m)
    d = DiagonalOf(x)
    assert d.shape == (None, 1)
    assert d.diagonal_length is None
    assert d[2, 0] == d[2] == x[2, 2]

    d = DiagonalOf(MatrixSymbol('x', 4, 3))
    assert d.shape == (3, 1)
    d = DiagonalOf(MatrixSymbol('x', n, 3))
    assert d.shape == (3, 1)
    d = DiagonalOf(MatrixSymbol('x', 3, n))
    assert d.shape == (3, 1)
    x = MatrixSymbol('x', n, m)
    assert [DiagonalOf(x)[i] for i in range(4)] ==[
        x[0, 0], x[1, 1], x[2, 2], x[3, 3]]


def test_DiagMatrix():
    x = MatrixSymbol('x', n, 1)
    d = DiagMatrix(x)
    assert d.shape == (n, n)
    assert d[0, 1] == 0
    assert d[0, 0] == x[0, 0]

    a = MatrixSymbol('a', 1, 1)
    d = diagonalize_vector(a)
    assert isinstance(d, MatrixSymbol)
    assert a == d
    assert diagonalize_vector(Identity(3)) == Identity(3)
    assert DiagMatrix(Identity(3)).doit() == Identity(3)
    assert isinstance(DiagMatrix(Identity(3)), DiagMatrix)

    # A diagonal matrix is equal to its transpose:
    assert DiagMatrix(x).T == DiagMatrix(x)
    assert diagonalize_vector(x.T) == DiagMatrix(x)

    dx = DiagMatrix(x)
    assert dx[0, 0] == x[0, 0]
    assert dx[1, 1] == x[1, 0]
    assert dx[0, 1] == 0
    assert dx[0, m] == x[0, 0]*KroneckerDelta(0, m)

    z = MatrixSymbol('z', 1, n)
    dz = DiagMatrix(z)
    assert dz[0, 0] == z[0, 0]
    assert dz[1, 1] == z[0, 1]
    assert dz[0, 1] == 0
    assert dz[0, m] == z[0, m]*KroneckerDelta(0, m)

    v = MatrixSymbol('v', 3, 1)
    dv = DiagMatrix(v)
    assert dv.as_explicit() == Matrix([
        [v[0, 0], 0, 0],
        [0, v[1, 0], 0],
        [0, 0, v[2, 0]],
    ])

    v = MatrixSymbol('v', 1, 3)
    dv = DiagMatrix(v)
    assert dv.as_explicit() == Matrix([
        [v[0, 0], 0, 0],
        [0, v[0, 1], 0],
        [0, 0, v[0, 2]],
    ])

    dv = DiagMatrix(3*v)
    assert dv.args == (3*v,)
    assert dv.doit() == 3*DiagMatrix(v)
    assert isinstance(dv.doit(), MatMul)

    a = MatrixSymbol("a", 3, 1).as_explicit()
    expr = DiagMatrix(a)
    result = Matrix([
        [a[0, 0], 0, 0],
        [0, a[1, 0], 0],
        [0, 0, a[2, 0]],
    ])
    assert expr.doit() == result
    expr = DiagMatrix(a.T)
    assert expr.doit() == result
