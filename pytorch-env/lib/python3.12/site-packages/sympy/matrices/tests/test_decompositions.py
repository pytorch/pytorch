from sympy.core.function import expand_mul
from sympy.core.numbers import I, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import Abs
from sympy.simplify.simplify import simplify
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices import Matrix, zeros, eye, SparseMatrix
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, slow
from sympy.testing.matrices import allclose


def test_LUdecomp():
    testmat = Matrix([[0, 2, 5, 3],
                      [3, 3, 7, 4],
                      [8, 4, 0, 2],
                      [-2, 6, 3, 4]])
    L, U, p = testmat.LUdecomposition()
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(4)

    testmat = Matrix([[6, -2, 7, 4],
                      [0, 3, 6, 7],
                      [1, -2, 7, 4],
                      [-9, 2, 6, 3]])
    L, U, p = testmat.LUdecomposition()
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(4)

    # non-square
    testmat = Matrix([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]])
    L, U, p = testmat.LUdecomposition(rankcheck=False)
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(4, 3)

    # square and singular
    testmat = Matrix([[1, 2, 3],
                      [2, 4, 6],
                      [4, 5, 6]])
    L, U, p = testmat.LUdecomposition(rankcheck=False)
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(3)

    M = Matrix(((1, x, 1), (2, y, 0), (y, 0, z)))
    L, U, p = M.LUdecomposition()
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - M == zeros(3)

    mL = Matrix((
        (1, 0, 0),
        (2, 3, 0),
    ))
    assert mL.is_lower is True
    assert mL.is_upper is False
    mU = Matrix((
        (1, 2, 3),
        (0, 4, 5),
    ))
    assert mU.is_lower is False
    assert mU.is_upper is True

    # test FF LUdecomp
    M = Matrix([[1, 3, 3],
                [3, 2, 6],
                [3, 2, 2]])
    P, L, Dee, U = M.LUdecompositionFF()
    assert P*M == L*Dee.inv()*U

    M = Matrix([[1,  2, 3,  4],
                [3, -1, 2,  3],
                [3,  1, 3, -2],
                [6, -1, 0,  2]])
    P, L, Dee, U = M.LUdecompositionFF()
    assert P*M == L*Dee.inv()*U

    M = Matrix([[0, 0, 1],
                [2, 3, 0],
                [3, 1, 4]])
    P, L, Dee, U = M.LUdecompositionFF()
    assert P*M == L*Dee.inv()*U

    # issue 15794
    M = Matrix(
        [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
    )
    raises(ValueError, lambda : M.LUdecomposition_Simple(rankcheck=True))

def test_singular_value_decompositionD():
    A = Matrix([[1, 2], [2, 1]])
    U, S, V = A.singular_value_decomposition()
    assert U * S * V.T == A
    assert U.T * U == eye(U.cols)
    assert V.T * V == eye(V.cols)

    B = Matrix([[1, 2]])
    U, S, V = B.singular_value_decomposition()

    assert U * S * V.T == B
    assert U.T * U == eye(U.cols)
    assert V.T * V == eye(V.cols)

    C = Matrix([
        [1, 0, 0, 0, 2],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
    ])

    U, S, V = C.singular_value_decomposition()

    assert U * S * V.T == C
    assert U.T * U == eye(U.cols)
    assert V.T * V == eye(V.cols)

    D = Matrix([[Rational(1, 3), sqrt(2)], [0, Rational(1, 4)]])
    U, S, V = D.singular_value_decomposition()
    assert simplify(U.T * U) == eye(U.cols)
    assert simplify(V.T * V) == eye(V.cols)
    assert simplify(U * S * V.T) == D


def test_QR():
    A = Matrix([[1, 2], [2, 3]])
    Q, S = A.QRdecomposition()
    R = Rational
    assert Q == Matrix([
        [  5**R(-1, 2),  (R(2)/5)*(R(1)/5)**R(-1, 2)],
        [2*5**R(-1, 2), (-R(1)/5)*(R(1)/5)**R(-1, 2)]])
    assert S == Matrix([[5**R(1, 2), 8*5**R(-1, 2)], [0, (R(1)/5)**R(1, 2)]])
    assert Q*S == A
    assert Q.T * Q == eye(2)

    A = Matrix([[1, 1, 1], [1, 1, 3], [2, 3, 4]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[12, 0, -51], [6, 0, 167], [-4, 0, 24]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    x = Symbol('x')
    A = Matrix([x])
    Q, R = A.QRdecomposition()
    assert Q == Matrix([x / Abs(x)])
    assert R == Matrix([Abs(x)])

    A = Matrix([[x, 0], [0, x]])
    Q, R = A.QRdecomposition()
    assert Q == x / Abs(x) * Matrix([[1, 0], [0, 1]])
    assert R == Abs(x) * Matrix([[1, 0], [0, 1]])


def test_QR_non_square():
    # Narrow (cols < rows) matrices
    A = Matrix([[9, 0, 26], [12, 0, -7], [0, 4, 4], [0, -3, -3]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix(2, 1, [1, 2])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    # Wide (cols > rows) matrices
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, 2, 3, 4], [1, 4, 9, 16], [1, 8, 27, 64]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix(1, 2, [1, 2])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

def test_QR_trivial():
    # Rank deficient matrices
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    # Zero rank matrices
    A = Matrix([[0, 0, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0], [0, 0, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0], [0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    # Rank deficient matrices with zero norm from beginning columns
    A = Matrix([[0, 0, 0], [1, 2, 3]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [0, 0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [0, 0, 0, 0], [2, 4, 6, 8]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 3]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R


def test_QR_float():
    A = Matrix([[1, 1], [1, 1.01]])
    Q, R = A.QRdecomposition()
    assert allclose(Q * R, A)
    assert allclose(Q * Q.T, Matrix.eye(2))
    assert allclose(Q.T * Q, Matrix.eye(2))

    A = Matrix([[1, 1], [1, 1.001]])
    Q, R = A.QRdecomposition()
    assert allclose(Q * R, A)
    assert allclose(Q * Q.T, Matrix.eye(2))
    assert allclose(Q.T * Q, Matrix.eye(2))


def test_LUdecomposition_Simple_iszerofunc():
    # Test if callable passed to matrices.LUdecomposition_Simple() as iszerofunc keyword argument is used inside
    # matrices.LUdecomposition_Simple()
    magic_string = "I got passed in!"
    def goofyiszero(value):
        raise ValueError(magic_string)

    try:
        lu, p = Matrix([[1, 0], [0, 1]]).LUdecomposition_Simple(iszerofunc=goofyiszero)
    except ValueError as err:
        assert magic_string == err.args[0]
        return

    assert False

def test_LUdecomposition_iszerofunc():
    # Test if callable passed to matrices.LUdecomposition() as iszerofunc keyword argument is used inside
    # matrices.LUdecomposition_Simple()
    magic_string = "I got passed in!"
    def goofyiszero(value):
        raise ValueError(magic_string)

    try:
        l, u, p = Matrix([[1, 0], [0, 1]]).LUdecomposition(iszerofunc=goofyiszero)
    except ValueError as err:
        assert magic_string == err.args[0]
        return

    assert False

def test_LDLdecomposition():
    raises(NonSquareMatrixError, lambda: Matrix((1, 2)).LDLdecomposition())
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).LDLdecomposition())
    raises(ValueError, lambda: Matrix(((5 + I, 0), (0, 1))).LDLdecomposition())
    raises(ValueError, lambda: Matrix(((1, 5), (5, 1))).LDLdecomposition())
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).LDLdecomposition(hermitian=False))
    A = Matrix(((1, 5), (5, 1)))
    L, D = A.LDLdecomposition(hermitian=False)
    assert L * D * L.T == A
    A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L, D = A.LDLdecomposition()
    assert L * D * L.T == A
    assert L.is_lower
    assert L == Matrix([[1, 0, 0], [ Rational(3, 5), 1, 0], [Rational(-1, 5), Rational(1, 3), 1]])
    assert D.is_diagonal()
    assert D == Matrix([[25, 0, 0], [0, 9, 0], [0, 0, 9]])
    A = Matrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    L, D = A.LDLdecomposition()
    assert expand_mul(L * D * L.H) == A
    assert L.expand() == Matrix([[1, 0, 0], [I/2, 1, 0], [S.Half - I/2, 0, 1]])
    assert D.expand() == Matrix(((4, 0, 0), (0, 1, 0), (0, 0, 9)))

    raises(NonSquareMatrixError, lambda: SparseMatrix((1, 2)).LDLdecomposition())
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).LDLdecomposition())
    raises(ValueError, lambda: SparseMatrix(((5 + I, 0), (0, 1))).LDLdecomposition())
    raises(ValueError, lambda: SparseMatrix(((1, 5), (5, 1))).LDLdecomposition())
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).LDLdecomposition(hermitian=False))
    A = SparseMatrix(((1, 5), (5, 1)))
    L, D = A.LDLdecomposition(hermitian=False)
    assert L * D * L.T == A
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L, D = A.LDLdecomposition()
    assert L * D * L.T == A
    assert L.is_lower
    assert L == Matrix([[1, 0, 0], [ Rational(3, 5), 1, 0], [Rational(-1, 5), Rational(1, 3), 1]])
    assert D.is_diagonal()
    assert D == Matrix([[25, 0, 0], [0, 9, 0], [0, 0, 9]])
    A = SparseMatrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    L, D = A.LDLdecomposition()
    assert expand_mul(L * D * L.H) == A
    assert L == Matrix(((1, 0, 0), (I/2, 1, 0), (S.Half - I/2, 0, 1)))
    assert D == Matrix(((4, 0, 0), (0, 1, 0), (0, 0, 9)))

def test_pinv_succeeds_with_rank_decomposition_method():
    # Test rank decomposition method of pseudoinverse succeeding
    As = [Matrix([
        [61, 89, 55, 20, 71, 0],
        [62, 96, 85, 85, 16, 0],
        [69, 56, 17,  4, 54, 0],
        [10, 54, 91, 41, 71, 0],
        [ 7, 30, 10, 48, 90, 0],
        [0,0,0,0,0,0]])]
    for A in As:
        A_pinv = A.pinv(method="RD")
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA

def test_rank_decomposition():
    a = Matrix(0, 0, [])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a

    a = Matrix(1, 1, [5])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a

    a = Matrix(3, 3, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a

    a = Matrix([
        [0, 0, 1, 2, 2, -5, 3],
        [-1, 5, 2, 2, 1, -7, 5],
        [0, 0, -2, -3, -3, 8, -5],
        [-1, 5, 0, -1, -2, 1, 0]])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a


@slow
def test_upper_hessenberg_decomposition():
    A = Matrix([
        [1, 0, sqrt(3)],
        [sqrt(2), Rational(1, 2), 2],
        [1, Rational(1, 4), 3],
    ])
    H, P = A.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert (simplify(P * H * P.H)) == A


    B = Matrix([
        [1, 2, 10],
        [8, 2, 5],
        [3, 12, 34],
    ])
    H, P = B.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == B

    C = Matrix([
        [1, sqrt(2), 2, 3],
        [0, 5, 3, 4],
        [1, 1, 4, sqrt(5)],
        [0, 2, 2, 3]
    ])

    H, P = C.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == C

    D = Matrix([
        [1, 2, 3],
        [-3, 5, 6],
        [4, -8, 9],
    ])
    H, P = D.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == D

    E = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ])

    H, P = E.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == E
