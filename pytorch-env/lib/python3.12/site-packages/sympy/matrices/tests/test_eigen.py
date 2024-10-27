from sympy.core.evalf import N
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices import eye, Matrix
from sympy.core.singleton import S
from sympy.testing.pytest import raises, XFAIL
from sympy.matrices.exceptions import NonSquareMatrixError, MatrixError
from sympy.matrices.expressions.fourier import DFT
from sympy.simplify.simplify import simplify
from sympy.matrices.immutable import ImmutableMatrix
from sympy.testing.pytest import slow
from sympy.testing.matrices import allclose


def test_eigen():
    R = Rational
    M = Matrix.eye(3)
    assert M.eigenvals(multiple=False) == {S.One: 3}
    assert M.eigenvals(multiple=True) == [1, 1, 1]

    assert M.eigenvects() == (
        [(1, 3, [Matrix([1, 0, 0]),
                 Matrix([0, 1, 0]),
                 Matrix([0, 0, 1])])])

    assert M.left_eigenvects() == (
        [(1, 3, [Matrix([[1, 0, 0]]),
                 Matrix([[0, 1, 0]]),
                 Matrix([[0, 0, 1]])])])

    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])

    assert M.eigenvals() == {2*S.One: 1, -S.One: 1, S.Zero: 1}

    assert M.eigenvects() == (
        [
            (-1, 1, [Matrix([-1, 1, 0])]),
            ( 0, 1, [Matrix([0, -1, 1])]),
            ( 2, 1, [Matrix([R(2, 3), R(1, 3), 1])])
        ])

    assert M.left_eigenvects() == (
        [
            (-1, 1, [Matrix([[-2, 1, 1]])]),
            (0, 1, [Matrix([[-1, -1, 1]])]),
            (2, 1, [Matrix([[1, 1, 1]])])
        ])

    a = Symbol('a')
    M = Matrix([[a, 0],
                [0, 1]])

    assert M.eigenvals() == {a: 1, S.One: 1}

    M = Matrix([[1, -1],
                [1,  3]])
    assert M.eigenvects() == ([(2, 2, [Matrix(2, 1, [-1, 1])])])
    assert M.left_eigenvects() == ([(2, 2, [Matrix([[1, 1]])])])

    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a = R(15, 2)
    b = 3*33**R(1, 2)
    c = R(13, 2)
    d = (R(33, 8) + 3*b/8)
    e = (R(33, 8) - 3*b/8)

    def NS(e, n):
        return str(N(e, n))
    r = [
        (a - b/2, 1, [Matrix([(12 + 24/(c - b/2))/((c - b/2)*e) + 3/(c - b/2),
                              (6 + 12/(c - b/2))/e, 1])]),
        (      0, 1, [Matrix([1, -2, 1])]),
        (a + b/2, 1, [Matrix([(12 + 24/(c + b/2))/((c + b/2)*d) + 3/(c + b/2),
                              (6 + 12/(c + b/2))/d, 1])]),
    ]
    r1 = [(NS(r[i][0], 2), NS(r[i][1], 2),
        [NS(j, 2) for j in r[i][2][0]]) for i in range(len(r))]
    r = M.eigenvects()
    r2 = [(NS(r[i][0], 2), NS(r[i][1], 2),
        [NS(j, 2) for j in r[i][2][0]]) for i in range(len(r))]
    assert sorted(r1) == sorted(r2)

    eps = Symbol('eps', real=True)

    M = Matrix([[abs(eps), I*eps    ],
                [-I*eps,   abs(eps) ]])

    assert M.eigenvects() == (
        [
            ( 0, 1, [Matrix([[-I*eps/abs(eps)], [1]])]),
            ( 2*abs(eps), 1, [ Matrix([[I*eps/abs(eps)], [1]]) ] ),
        ])

    assert M.left_eigenvects() == (
        [
            (0, 1, [Matrix([[I*eps/Abs(eps), 1]])]),
            (2*Abs(eps), 1, [Matrix([[-I*eps/Abs(eps), 1]])])
        ])

    M = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])
    M._eigenvects = M.eigenvects(simplify=False)
    assert max(i.q for i in M._eigenvects[0][2][0]) > 1
    M._eigenvects = M.eigenvects(simplify=True)
    assert max(i.q for i in M._eigenvects[0][2][0]) == 1

    M = Matrix([[Rational(1, 4), 1], [1, 1]])
    assert M.eigenvects() == [
        (Rational(5, 8) - sqrt(73)/8, 1, [Matrix([[-sqrt(73)/8 - Rational(3, 8)], [1]])]),
        (Rational(5, 8) + sqrt(73)/8, 1, [Matrix([[Rational(-3, 8) + sqrt(73)/8], [1]])])]

    # issue 10719
    assert Matrix([]).eigenvals() == {}
    assert Matrix([]).eigenvals(multiple=True) == []
    assert Matrix([]).eigenvects() == []

    # issue 15119
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 2], [0, 4], [0, 0]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 0], [3, 4], [5, 6]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 2, 3], [0, 5, 6]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 0, 0], [4, 5, 0]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 2, 3], [0, 5, 6]]).eigenvals(
               error_when_incomplete = False))
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 0, 0], [4, 5, 0]]).eigenvals(
               error_when_incomplete = False))

    m = Matrix([[1, 2], [3, 4]])
    assert isinstance(m.eigenvals(simplify=True, multiple=False), dict)
    assert isinstance(m.eigenvals(simplify=True, multiple=True), list)
    assert isinstance(m.eigenvals(simplify=lambda x: x, multiple=False), dict)
    assert isinstance(m.eigenvals(simplify=lambda x: x, multiple=True), list)


def test_float_eigenvals():
    m = Matrix([[1, .6, .6], [.6, .9, .9], [.9, .6, .6]])
    evals = [
        Rational(5, 4) - sqrt(385)/20,
        sqrt(385)/20 + Rational(5, 4),
        S.Zero]

    n_evals = m.eigenvals(rational=True, multiple=True)
    n_evals = sorted(n_evals)
    s_evals = [x.evalf() for x in evals]
    s_evals = sorted(s_evals)

    for x, y in zip(n_evals, s_evals):
        assert abs(x-y) < 10**-9


@XFAIL
def test_eigen_vects():
    m = Matrix(2, 2, [1, 0, 0, I])
    raises(NotImplementedError, lambda: m.is_diagonalizable(True))
    # !!! bug because of eigenvects() or roots(x**2 + (-1 - I)*x + I, x)
    # see issue 5292
    assert not m.is_diagonalizable(True)
    raises(MatrixError, lambda: m.diagonalize(True))
    (P, D) = m.diagonalize(True)

def test_issue_8240():
    # Eigenvalues of large triangular matrices
    x, y = symbols('x y')
    n = 200

    diagonal_variables = [Symbol('x%s' % i) for i in range(n)]
    M = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        M[i][i] = diagonal_variables[i]
    M = Matrix(M)

    eigenvals = M.eigenvals()
    assert len(eigenvals) == n
    for i in range(n):
        assert eigenvals[diagonal_variables[i]] == 1

    eigenvals = M.eigenvals(multiple=True)
    assert set(eigenvals) == set(diagonal_variables)

    # with multiplicity
    M = Matrix([[x, 0, 0], [1, y, 0], [2, 3, x]])
    eigenvals = M.eigenvals()
    assert eigenvals == {x: 2, y: 1}

    eigenvals = M.eigenvals(multiple=True)
    assert len(eigenvals) == 3
    assert eigenvals.count(x) == 2
    assert eigenvals.count(y) == 1


def test_eigenvals():
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
    assert M.eigenvals() == {2*S.One: 1, -S.One: 1, S.Zero: 1}

    m = Matrix([
        [3,  0,  0, 0, -3],
        [0, -3, -3, 0,  3],
        [0,  3,  0, 3,  0],
        [0,  0,  3, 0,  3],
        [3,  0,  0, 3,  0]])

    # XXX Used dry-run test because arbitrary symbol that appears in
    # CRootOf may not be unique.
    assert m.eigenvals()


def test_eigenvects():
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
    vecs = M.eigenvects()
    for val, mult, vec_list in vecs:
        assert len(vec_list) == 1
        assert M*vec_list[0] == val*vec_list[0]


def test_left_eigenvects():
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
    vecs = M.left_eigenvects()
    for val, mult, vec_list in vecs:
        assert len(vec_list) == 1
        assert vec_list[0]*M == val*vec_list[0]


@slow
def test_bidiagonalize():
    M = Matrix([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    assert M.bidiagonalize() == M
    assert M.bidiagonalize(upper=False) == M
    assert M.bidiagonalize() == M
    assert M.bidiagonal_decomposition() == (M, M, M)
    assert M.bidiagonal_decomposition(upper=False) == (M, M, M)
    assert M.bidiagonalize() == M

    import random
    #Real Tests
    for real_test in range(2):
        test_values = []
        row = 2
        col = 2
        for _ in range(row * col):
            value = random.randint(-1000000000, 1000000000)
            test_values = test_values + [value]
        # L     -> Lower Bidiagonalization
        # M     -> Mutable Matrix
        # N     -> Immutable Matrix
        # 0     -> Bidiagonalized form
        # 1,2,3 -> Bidiagonal_decomposition matrices
        # 4     -> Product of 1 2 3
        M = Matrix(row, col, test_values)
        N = ImmutableMatrix(M)

        N1, N2, N3 = N.bidiagonal_decomposition()
        M1, M2, M3 = M.bidiagonal_decomposition()
        M0 = M.bidiagonalize()
        N0 = N.bidiagonalize()

        N4 = N1 * N2 * N3
        M4 = M1 * M2 * M3

        N2.simplify()
        N4.simplify()
        N0.simplify()

        M0.simplify()
        M2.simplify()
        M4.simplify()

        LM0 = M.bidiagonalize(upper=False)
        LM1, LM2, LM3 = M.bidiagonal_decomposition(upper=False)
        LN0 = N.bidiagonalize(upper=False)
        LN1, LN2, LN3 = N.bidiagonal_decomposition(upper=False)

        LN4 = LN1 * LN2 * LN3
        LM4 = LM1 * LM2 * LM3

        LN2.simplify()
        LN4.simplify()
        LN0.simplify()

        LM0.simplify()
        LM2.simplify()
        LM4.simplify()

        assert M == M4
        assert M2 == M0
        assert N == N4
        assert N2 == N0
        assert M == LM4
        assert LM2 == LM0
        assert N == LN4
        assert LN2 == LN0

    #Complex Tests
    for complex_test in range(2):
        test_values = []
        size = 2
        for _ in range(size * size):
            real = random.randint(-1000000000, 1000000000)
            comp = random.randint(-1000000000, 1000000000)
            value = real + comp * I
            test_values = test_values + [value]
        M = Matrix(size, size, test_values)
        N = ImmutableMatrix(M)
        # L     -> Lower Bidiagonalization
        # M     -> Mutable Matrix
        # N     -> Immutable Matrix
        # 0     -> Bidiagonalized form
        # 1,2,3 -> Bidiagonal_decomposition matrices
        # 4     -> Product of 1 2 3
        N1, N2, N3 = N.bidiagonal_decomposition()
        M1, M2, M3 = M.bidiagonal_decomposition()
        M0 = M.bidiagonalize()
        N0 = N.bidiagonalize()

        N4 = N1 * N2 * N3
        M4 = M1 * M2 * M3

        N2.simplify()
        N4.simplify()
        N0.simplify()

        M0.simplify()
        M2.simplify()
        M4.simplify()

        LM0 = M.bidiagonalize(upper=False)
        LM1, LM2, LM3 = M.bidiagonal_decomposition(upper=False)
        LN0 = N.bidiagonalize(upper=False)
        LN1, LN2, LN3 = N.bidiagonal_decomposition(upper=False)

        LN4 = LN1 * LN2 * LN3
        LM4 = LM1 * LM2 * LM3

        LN2.simplify()
        LN4.simplify()
        LN0.simplify()

        LM0.simplify()
        LM2.simplify()
        LM4.simplify()

        assert M == M4
        assert M2 == M0
        assert N == N4
        assert N2 == N0
        assert M == LM4
        assert LM2 == LM0
        assert N == LN4
        assert LN2 == LN0

    M = Matrix(18, 8, range(1, 145))
    M = M.applyfunc(lambda i: Float(i))
    assert M.bidiagonal_decomposition()[1] == M.bidiagonalize()
    assert M.bidiagonal_decomposition(upper=False)[1] == M.bidiagonalize(upper=False)
    a, b, c = M.bidiagonal_decomposition()
    diff = a * b * c - M
    assert abs(max(diff)) < 10**-12


def test_diagonalize():
    m = Matrix(2, 2, [0, -1, 1, 0])
    raises(MatrixError, lambda: m.diagonalize(reals_only=True))
    P, D = m.diagonalize()
    assert D.is_diagonal()
    assert D == Matrix([
                 [-I, 0],
                 [ 0, I]])

    # make sure we use floats out if floats are passed in
    m = Matrix(2, 2, [0, .5, .5, 0])
    P, D = m.diagonalize()
    assert all(isinstance(e, Float) for e in D.values())
    assert all(isinstance(e, Float) for e in P.values())

    _, D2 = m.diagonalize(reals_only=True)
    assert D == D2

    m = Matrix(
        [[0, 1, 0, 0], [1, 0, 0, 0.002], [0.002, 0, 0, 1], [0, 0, 1, 0]])
    P, D = m.diagonalize()
    assert allclose(P*D, m*P)


def test_is_diagonalizable():
    a, b, c = symbols('a b c')
    m = Matrix(2, 2, [a, c, c, b])
    assert m.is_symmetric()
    assert m.is_diagonalizable()
    assert not Matrix(2, 2, [1, 1, 0, 1]).is_diagonalizable()

    m = Matrix(2, 2, [0, -1, 1, 0])
    assert m.is_diagonalizable()
    assert not m.is_diagonalizable(reals_only=True)


def test_jordan_form():
    m = Matrix(3, 2, [-3, 1, -3, 20, 3, 10])
    raises(NonSquareMatrixError, lambda: m.jordan_form())

    # the next two tests test the cases where the old
    # algorithm failed due to the fact that the block structure can
    # *NOT* be determined  from algebraic and geometric multiplicity alone
    # This can be seen most easily when one lets compute the J.c.f. of a matrix that
    # is in J.c.f already.
    m = Matrix(4, 4, [2, 1, 0, 0,
                    0, 2, 1, 0,
                    0, 0, 2, 0,
                    0, 0, 0, 2
    ])
    P, J = m.jordan_form()
    assert m == J

    m = Matrix(4, 4, [2, 1, 0, 0,
                    0, 2, 0, 0,
                    0, 0, 2, 1,
                    0, 0, 0, 2
    ])
    P, J = m.jordan_form()
    assert m == J

    A = Matrix([[ 2,  4,  1,  0],
                [-4,  2,  0,  1],
                [ 0,  0,  2,  4],
                [ 0,  0, -4,  2]])
    P, J = A.jordan_form()
    assert simplify(P*J*P.inv()) == A

    assert Matrix(1, 1, [1]).jordan_form() == (Matrix([1]), Matrix([1]))
    assert Matrix(1, 1, [1]).jordan_form(calc_transform=False) == Matrix([1])

    # If we have eigenvalues in CRootOf form, raise errors
    m = Matrix([[3, 0, 0, 0, -3], [0, -3, -3, 0, 3], [0, 3, 0, 3, 0], [0, 0, 3, 0, 3], [3, 0, 0, 3, 0]])
    raises(MatrixError, lambda: m.jordan_form())

    # make sure that if the input has floats, the output does too
    m = Matrix([
        [                0.6875, 0.125 + 0.1875*sqrt(3)],
        [0.125 + 0.1875*sqrt(3),                 0.3125]])
    P, J = m.jordan_form()
    assert all(isinstance(x, Float) or x == 0 for x in P)
    assert all(isinstance(x, Float) or x == 0 for x in J)


def test_singular_values():
    x = Symbol('x', real=True)

    A = Matrix([[0, 1*I], [2, 0]])
    # if singular values can be sorted, they should be in decreasing order
    assert A.singular_values() == [2, 1]

    A = eye(3)
    A[1, 1] = x
    A[2, 2] = 5
    vals = A.singular_values()
    # since Abs(x) cannot be sorted, test set equality
    assert set(vals) == {5, 1, Abs(x)}

    A = Matrix([[sin(x), cos(x)], [-cos(x), sin(x)]])
    vals = [sv.trigsimp() for sv in A.singular_values()]
    assert vals == [S.One, S.One]

    A = Matrix([
        [2, 4],
        [1, 3],
        [0, 0],
        [0, 0]
        ])
    assert A.singular_values() == \
        [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221))]
    assert A.T.singular_values() == \
        [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221)), 0, 0]

def test___eq__():
    assert (Matrix(
        [[0, 1, 1],
        [1, 0, 0],
        [1, 1, 1]]) == {}) is False


def test_definite():
    # Examples from Gilbert Strang, "Introduction to Linear Algebra"
    # Positive definite matrices
    m = Matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    m = Matrix([[5, 4], [4, 5]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # Positive semidefinite matrices
    m = Matrix([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    m = Matrix([[1, 2], [2, 4]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # Examples from Mathematica documentation
    # Non-hermitian positive definite matrices
    m = Matrix([[2, 3], [4, 8]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # Hermetian matrices
    m = Matrix([[1, 2*I], [-I, 4]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # Symbolic matrices examples
    a = Symbol('a', positive=True)
    b = Symbol('b', negative=True)
    m = Matrix([[a, 0, 0], [0, a, 0], [0, 0, a]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    m = Matrix([[b, 0, 0], [0, b, 0], [0, 0, b]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == False
    assert m.is_negative_definite == True
    assert m.is_negative_semidefinite == True
    assert m.is_indefinite == False

    m = Matrix([[a, 0], [0, b]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == False
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == True

    m = Matrix([
        [0.0228202735623867, 0.00518748979085398,
         -0.0743036351048907, -0.00709135324903921],
        [0.00518748979085398, 0.0349045359786350,
         0.0830317991056637, 0.00233147902806909],
        [-0.0743036351048907, 0.0830317991056637,
         1.15859676366277, 0.340359081555988],
        [-0.00709135324903921, 0.00233147902806909,
         0.340359081555988, 0.928147644848199]
    ])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_indefinite == False

    # test for issue 19547: https://github.com/sympy/sympy/issues/19547
    m = Matrix([
        [0, 0, 0],
        [0, 1, 2],
        [0, 2, 1]
    ])
    assert not m.is_positive_definite
    assert not m.is_positive_semidefinite


def test_positive_semidefinite_cholesky():
    from sympy.matrices.eigen import _is_positive_semidefinite_cholesky

    m = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[0, 0, 0], [0, 5, -10*I], [0, 10*I, 5]])
    assert _is_positive_semidefinite_cholesky(m) == False
    m = Matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    assert _is_positive_semidefinite_cholesky(m) == False
    m = Matrix([[0, 1], [1, 0]])
    assert _is_positive_semidefinite_cholesky(m) == False

    # https://www.value-at-risk.net/cholesky-factorization/
    m = Matrix([[4, -2, -6], [-2, 10, 9], [-6, 9, 14]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[9, -3, 3], [-3, 2, 1], [3, 1, 6]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[4, -2, 2], [-2, 1, -1], [2, -1, 5]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[1, 2, -1], [2, 5, 1], [-1, 1, 9]])
    assert _is_positive_semidefinite_cholesky(m) == False


def test_issue_20582():
    A = Matrix([
        [5, -5, -3, 2, -7],
        [-2, -5, 0, 2, 1],
        [-2, -7, -5, -2, -6],
        [7, 10, 3, 9, -2],
        [4, -10, 3, -8, -4]
    ])
    # XXX Used dry-run test because arbitrary symbol that appears in
    # CRootOf may not be unique.
    assert A.eigenvects()

def test_issue_19210():
    t = Symbol('t')
    H = Matrix([[3, 0, 0, 0], [0, 1 , 2, 0], [0, 2, 2, 0], [0, 0, 0, 4]])
    A = (-I * H * t).jordan_form()
    assert A == (Matrix([
                    [0, 1,                  0,                0],
                    [0, 0, -4/(-1 + sqrt(17)), 4/(1 + sqrt(17))],
                    [0, 0,                  1,                1],
                    [1, 0,                  0,                0]]), Matrix([
                    [-4*I*t,      0,                         0,                         0],
                    [     0, -3*I*t,                         0,                         0],
                    [     0,      0, t*(-3*I/2 + sqrt(17)*I/2),                         0],
                    [     0,      0,                         0, t*(-sqrt(17)*I/2 - 3*I/2)]]))


def test_issue_20275():
    # XXX We use complex expansions because complex exponentials are not
    # recognized by polys.domains
    A = DFT(3).as_explicit().expand(complex=True)
    eigenvects = A.eigenvects()
    assert eigenvects[0] == (
        -1, 1,
        [Matrix([[1 - sqrt(3)], [1], [1]])]
    )
    assert eigenvects[1] == (
        1, 1,
        [Matrix([[1 + sqrt(3)], [1], [1]])]
    )
    assert eigenvects[2] == (
        -I, 1,
        [Matrix([[0], [-1], [1]])]
    )

    A = DFT(4).as_explicit().expand(complex=True)
    eigenvects = A.eigenvects()
    assert eigenvects[0] == (
        -1, 1,
        [Matrix([[-1], [1], [1], [1]])]
    )
    assert eigenvects[1] == (
        1, 2,
        [Matrix([[1], [0], [1], [0]]), Matrix([[2], [1], [0], [1]])]
    )
    assert eigenvects[2] == (
        -I, 1,
        [Matrix([[0], [-1], [0], [1]])]
    )

    # XXX We skip test for some parts of eigenvectors which are very
    # complicated and fragile under expression tree changes
    A = DFT(5).as_explicit().expand(complex=True)
    eigenvects = A.eigenvects()
    assert eigenvects[0] == (
        -1, 1,
        [Matrix([[1 - sqrt(5)], [1], [1], [1], [1]])]
    )
    assert eigenvects[1] == (
        1, 2,
        [Matrix([[S(1)/2 + sqrt(5)/2], [0], [1], [1], [0]]),
         Matrix([[S(1)/2 + sqrt(5)/2], [1], [0], [0], [1]])]
    )


def test_issue_20752():
    b = symbols('b', nonzero=True)
    m = Matrix([[0, 0, 0], [0, b, 0], [0, 0, b]])
    assert m.is_positive_semidefinite is None


def test_issue_25282():
    dd = sd = [0] * 11 + [1]
    ds = [2, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    ss = ds.copy()
    ss[8] = 2

    def rotate(x, i):
        return x[i:] + x[:i]

    mat = []
    for i in range(12):
        mat.append(rotate(ss, i) + rotate(sd, i))
    for i in range(12):
        mat.append(rotate(ds, i) + rotate(dd, i))

    assert sum(Matrix(mat).eigenvals().values()) == 24
