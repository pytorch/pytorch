from sympy.matrices import Matrix
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.solvers import solve


def test_columnspace_one():
    m = Matrix([[ 1,  2,  0,  2,  5],
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    basis = m.columnspace()
    assert basis[0] == Matrix([1, -2, 0, 3])
    assert basis[1] == Matrix([2, -5, -3, 6])
    assert basis[2] == Matrix([2, -1, 4, -7])

    assert len(basis) == 3
    assert Matrix.hstack(m, *basis).columnspace() == basis


def test_rowspace():
    m = Matrix([[ 1,  2,  0,  2,  5],
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    basis = m.rowspace()
    assert basis[0] == Matrix([[1, 2, 0, 2, 5]])
    assert basis[1] == Matrix([[0, -1, 1, 3, 2]])
    assert basis[2] == Matrix([[0, 0, 0, 5, 5]])

    assert len(basis) == 3


def test_nullspace_one():
    m = Matrix([[ 1,  2,  0,  2,  5],
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    basis = m.nullspace()
    assert basis[0] == Matrix([-2, 1, 1, 0, 0])
    assert basis[1] == Matrix([-1, -1, 0, -1, 1])
    # make sure the null space is really gets zeroed
    assert all(e.is_zero for e in m*basis[0])
    assert all(e.is_zero for e in m*basis[1])

def test_nullspace_second():
    # first test reduced row-ech form
    R = Rational

    M = Matrix([[5, 7, 2,  1],
                [1, 6, 2, -1]])
    out, tmp = M.rref()
    assert out == Matrix([[1, 0, -R(2)/23, R(13)/23],
                          [0, 1,  R(8)/23, R(-6)/23]])

    M = Matrix([[-5, -1,  4, -3, -1],
                [ 1, -1, -1,  1,  0],
                [-1,  0,  0,  0,  0],
                [ 4,  1, -4,  3,  1],
                [-2,  0,  2, -2, -1]])
    assert M*M.nullspace()[0] == Matrix(5, 1, [0]*5)

    M = Matrix([[ 1,  3, 0,  2,  6, 3, 1],
                [-2, -6, 0, -2, -8, 3, 1],
                [ 3,  9, 0,  0,  6, 6, 2],
                [-1, -3, 0,  1,  0, 9, 3]])
    out, tmp = M.rref()
    assert out == Matrix([[1, 3, 0, 0, 2, 0, 0],
                          [0, 0, 0, 1, 2, 0, 0],
                          [0, 0, 0, 0, 0, 1, R(1)/3],
                          [0, 0, 0, 0, 0, 0, 0]])

    # now check the vectors
    basis = M.nullspace()
    assert basis[0] == Matrix([-3, 1, 0, 0, 0, 0, 0])
    assert basis[1] == Matrix([0, 0, 1, 0, 0, 0, 0])
    assert basis[2] == Matrix([-2, 0, 0, -2, 1, 0, 0])
    assert basis[3] == Matrix([0, 0, 0, 0, 0, R(-1)/3, 1])

    # issue 4797; just see that we can do it when rows > cols
    M = Matrix([[1, 2], [2, 4], [3, 6]])
    assert M.nullspace()


def test_columnspace_second():
    M = Matrix([[ 1,  2,  0,  2,  5],
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    # now check the vectors
    basis = M.columnspace()
    assert basis[0] == Matrix([1, -2, 0, 3])
    assert basis[1] == Matrix([2, -5, -3, 6])
    assert basis[2] == Matrix([2, -1, 4, -7])

    #check by columnspace definition
    a, b, c, d, e = symbols('a b c d e')
    X = Matrix([a, b, c, d, e])
    for i in range(len(basis)):
        eq=M*X-basis[i]
        assert len(solve(eq, X)) != 0

    #check if rank-nullity theorem holds
    assert M.rank() == len(basis)
    assert len(M.nullspace()) + len(M.columnspace()) == M.cols
