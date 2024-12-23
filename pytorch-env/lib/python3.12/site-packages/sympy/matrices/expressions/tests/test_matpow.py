from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.powsimp import powsimp
from sympy.testing.pytest import raises
from sympy.core.expr import unchanged
from sympy.core import symbols, S
from sympy.matrices import Identity, MatrixSymbol, ImmutableMatrix, ZeroMatrix, OneMatrix, Matrix
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.expressions import MatPow, MatAdd, MatMul
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixElement

n, m, l, k = symbols('n m l k', integer=True)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
E = MatrixSymbol('E', m, n)


def test_entry_matrix():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    assert MatPow(X, 0)[0, 0] == 1
    assert MatPow(X, 0)[0, 1] == 0
    assert MatPow(X, 1)[0, 0] == 1
    assert MatPow(X, 1)[0, 1] == 2
    assert MatPow(X, 2)[0, 0] == 7


def test_entry_symbol():
    from sympy.concrete import Sum
    assert MatPow(C, 0)[0, 0] == 1
    assert MatPow(C, 0)[0, 1] == 0
    assert MatPow(C, 1)[0, 0] == C[0, 0]
    assert isinstance(MatPow(C, 2)[0, 0], Sum)
    assert isinstance(MatPow(C, n)[0, 0], MatrixElement)


def test_as_explicit_symbol():
    X = MatrixSymbol('X', 2, 2)
    assert MatPow(X, 0).as_explicit() == ImmutableMatrix(Identity(2))
    assert MatPow(X, 1).as_explicit() == X.as_explicit()
    assert MatPow(X, 2).as_explicit() == (X.as_explicit())**2
    assert MatPow(X, n).as_explicit() == ImmutableMatrix([
        [(X ** n)[0, 0], (X ** n)[0, 1]],
        [(X ** n)[1, 0], (X ** n)[1, 1]],
    ])

    a = MatrixSymbol("a", 3, 1)
    b = MatrixSymbol("b", 3, 1)
    c = MatrixSymbol("c", 3, 1)

    expr = (a.T*b)**S.Half
    assert expr.as_explicit() == Matrix([[sqrt(a[0, 0]*b[0, 0] + a[1, 0]*b[1, 0] + a[2, 0]*b[2, 0])]])

    expr = c*(a.T*b)**S.Half
    m = sqrt(a[0, 0]*b[0, 0] + a[1, 0]*b[1, 0] + a[2, 0]*b[2, 0])
    assert expr.as_explicit() == Matrix([[c[0, 0]*m], [c[1, 0]*m], [c[2, 0]*m]])

    expr = (a*b.T)**S.Half
    denom = sqrt(a[0, 0]*b[0, 0] + a[1, 0]*b[1, 0] + a[2, 0]*b[2, 0])
    expected = (a*b.T).as_explicit()/denom
    assert expr.as_explicit() == expected

    expr = X**-1
    det = X[0, 0]*X[1, 1] - X[1, 0]*X[0, 1]
    expected = Matrix([[X[1, 1], -X[0, 1]], [-X[1, 0], X[0, 0]]])/det
    assert expr.as_explicit() == expected

    expr = X**m
    assert expr.as_explicit() == X.as_explicit()**m


def test_as_explicit_matrix():
    A = ImmutableMatrix([[1, 2], [3, 4]])
    assert MatPow(A, 0).as_explicit() == ImmutableMatrix(Identity(2))
    assert MatPow(A, 1).as_explicit() == A
    assert MatPow(A, 2).as_explicit() == A**2
    assert MatPow(A, -1).as_explicit() == A.inv()
    assert MatPow(A, -2).as_explicit() == (A.inv())**2
    # less expensive than testing on a 2x2
    A = ImmutableMatrix([4])
    assert MatPow(A, S.Half).as_explicit() == A**S.Half


def test_doit_symbol():
    assert MatPow(C, 0).doit() == Identity(n)
    assert MatPow(C, 1).doit() == C
    assert MatPow(C, -1).doit() == C.I
    for r in [2, S.Half, S.Pi, n]:
        assert MatPow(C, r).doit() == MatPow(C, r)


def test_doit_matrix():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    assert MatPow(X, 0).doit() == ImmutableMatrix(Identity(2))
    assert MatPow(X, 1).doit() == X
    assert MatPow(X, 2).doit() == X**2
    assert MatPow(X, -1).doit() == X.inv()
    assert MatPow(X, -2).doit() == (X.inv())**2
    # less expensive than testing on a 2x2
    assert MatPow(ImmutableMatrix([4]), S.Half).doit() == ImmutableMatrix([2])
    X = ImmutableMatrix([[0, 2], [0, 4]]) # det() == 0
    raises(ValueError, lambda: MatPow(X,-1).doit())
    raises(ValueError, lambda: MatPow(X,-2).doit())


def test_nonsquare():
    A = MatrixSymbol('A', 2, 3)
    B = ImmutableMatrix([[1, 2, 3], [4, 5, 6]])
    for r in [-1, 0, 1, 2, S.Half, S.Pi, n]:
        raises(NonSquareMatrixError, lambda: MatPow(A, r))
        raises(NonSquareMatrixError, lambda: MatPow(B, r))


def test_doit_equals_pow(): #17179
    X = ImmutableMatrix ([[1,0],[0,1]])
    assert MatPow(X, n).doit() == X**n == X


def test_doit_nested_MatrixExpr():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[2, 3], [4, 5]])
    assert MatPow(MatMul(X, Y), 2).doit() == (X*Y)**2
    assert MatPow(MatAdd(X, Y), 2).doit() == (X + Y)**2


def test_identity_power():
    k = Identity(n)
    assert MatPow(k, 4).doit() == k
    assert MatPow(k, n).doit() == k
    assert MatPow(k, -3).doit() == k
    assert MatPow(k, 0).doit() == k
    l = Identity(3)
    assert MatPow(l, n).doit() == l
    assert MatPow(l, -1).doit() == l
    assert MatPow(l, 0).doit() == l


def test_zero_power():
    z1 = ZeroMatrix(n, n)
    assert MatPow(z1, 3).doit() == z1
    raises(ValueError, lambda:MatPow(z1, -1).doit())
    assert MatPow(z1, 0).doit() == Identity(n)
    assert MatPow(z1, n).doit() == z1
    raises(ValueError, lambda:MatPow(z1, -2).doit())
    z2 = ZeroMatrix(4, 4)
    assert MatPow(z2, n).doit() == z2
    raises(ValueError, lambda:MatPow(z2, -3).doit())
    assert MatPow(z2, 2).doit() == z2
    assert MatPow(z2, 0).doit() == Identity(4)
    raises(ValueError, lambda:MatPow(z2, -1).doit())


def test_OneMatrix_power():
    o = OneMatrix(3, 3)
    assert o ** 0 == Identity(3)
    assert o ** 1 == o
    assert o * o == o ** 2 == 3 * o
    assert o * o * o == o ** 3 == 9 * o

    o = OneMatrix(n, n)
    assert o * o == o ** 2 == n * o
    # powsimp necessary as n ** (n - 2) * n does not produce n ** (n - 1)
    assert powsimp(o ** (n - 1) * o) == o ** n == n ** (n - 1) * o


def test_transpose_power():
    from sympy.matrices.expressions.transpose import Transpose as TP

    assert (C*D).T**5 == ((C*D)**5).T == (D.T * C.T)**5
    assert ((C*D).T**5).T == (C*D)**5

    assert (C.T.I.T)**7 == C**-7
    assert (C.T**l).T**k == C**(l*k)

    assert ((E.T * A.T)**5).T == (A*E)**5
    assert ((A*E).T**5).T**7 == (A*E)**35
    assert TP(TP(C**2 * D**3)**5).doit() == (C**2 * D**3)**5

    assert ((D*C)**-5).T**-5 == ((D*C)**25).T
    assert (((D*C)**l).T**k).T == (D*C)**(l*k)


def test_Inverse():
    assert Inverse(MatPow(C, 0)).doit() == Identity(n)
    assert Inverse(MatPow(C, 1)).doit() == Inverse(C)
    assert Inverse(MatPow(C, 2)).doit() == MatPow(C, -2)
    assert Inverse(MatPow(C, -1)).doit() == C

    assert MatPow(Inverse(C), 0).doit() == Identity(n)
    assert MatPow(Inverse(C), 1).doit() == Inverse(C)
    assert MatPow(Inverse(C), 2).doit() == MatPow(C, -2)
    assert MatPow(Inverse(C), -1).doit() == C


def test_combine_powers():
    assert (C ** 1) ** 1 == C
    assert (C ** 2) ** 3 == MatPow(C, 6)
    assert (C ** -2) ** -3 == MatPow(C, 6)
    assert (C ** -1) ** -1 == C
    assert (((C ** 2) ** 3) ** 4) ** 5 == MatPow(C, 120)
    assert (C ** n) ** n == C ** (n ** 2)


def test_unchanged():
    assert unchanged(MatPow, C, 0)
    assert unchanged(MatPow, C, 1)
    assert unchanged(MatPow, Inverse(C), -1)
    assert unchanged(Inverse, MatPow(C, -1), -1)
    assert unchanged(MatPow, MatPow(C, -1), -1)
    assert unchanged(MatPow, MatPow(C, 1), 1)


def test_no_exponentiation():
    # if this passes, Pow.as_numer_denom should recognize
    # MatAdd as exponent
    raises(NotImplementedError, lambda: 3**(-2*C))
