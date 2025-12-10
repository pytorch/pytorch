from sympy.concrete.summations import Sum
from sympy.core.exprtools import gcd_terms
from sympy.core.function import (diff, expand)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol, Str)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.polys.polytools import factor

from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,
                    Function)
from sympy.functions import sin, cos, tan, sqrt, cbrt, exp
from sympy.simplify import simplify
from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,
        MatPow, Matrix, MatrixExpr, MatrixSymbol,
        SparseMatrix, Transpose, Adjoint, MatrixSet)
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.expressions.determinant import Determinant, det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.testing.pytest import raises, XFAIL, skip
from importlib.metadata import version

n, m, l, k, p = symbols('n m l k p', integer=True)
x = symbols('x')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
E = MatrixSymbol('E', m, n)
w = MatrixSymbol('w', n, 1)


def test_matrix_symbol_creation():
    assert MatrixSymbol('A', 2, 2)
    assert MatrixSymbol('A', 0, 0)
    raises(ValueError, lambda: MatrixSymbol('A', -1, 2))
    raises(ValueError, lambda: MatrixSymbol('A', 2.0, 2))
    raises(ValueError, lambda: MatrixSymbol('A', 2j, 2))
    raises(ValueError, lambda: MatrixSymbol('A', 2, -1))
    raises(ValueError, lambda: MatrixSymbol('A', 2, 2.0))
    raises(ValueError, lambda: MatrixSymbol('A', 2, 2j))

    n = symbols('n')
    assert MatrixSymbol('A', n, n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: MatrixSymbol('A', n, n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: MatrixSymbol('A', n, n))


def test_matexpr_properties():
    assert A.shape == (n, m)
    assert (A * B).shape == (n, l)
    assert A[0, 1].indices == (0, 1)
    assert A[0, 0].symbol == A
    assert A[0, 0].symbol.name == 'A'


def test_matexpr():
    assert (x*A).shape == A.shape
    assert (x*A).__class__ == MatMul
    assert 2*A - A - A == ZeroMatrix(*A.shape)
    assert (A*B).shape == (n, l)


def test_matexpr_subs():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    C = MatrixSymbol('C', m, l)

    assert A.subs(n, m).shape == (m, m)
    assert (A*B).subs(B, C) == A*C
    assert (A*B).subs(l, n).is_square

    W = MatrixSymbol("W", 3, 3)
    X = MatrixSymbol("X", 2, 2)
    Y = MatrixSymbol("Y", 1, 2)
    Z = MatrixSymbol("Z", n, 2)
    # no restrictions on Symbol replacement
    assert X.subs(X, Y) == Y
    # it might be better to just change the name
    y = Str('y')
    assert X.subs(Str("X"), y).args == (y, 2, 2)
    # it's ok to introduce a wider matrix
    assert X[1, 1].subs(X, W) == W[1, 1]
    # but for a given MatrixExpression, only change
    # name if indexing on the new shape is valid.
    # Here, X is 2,2; Y is 1,2 and Y[1, 1] is out
    # of range so an error is raised
    raises(IndexError, lambda: X[1, 1].subs(X, Y))
    # here, [0, 1] is in range so the subs succeeds
    assert X[0, 1].subs(X, Y) == Y[0, 1]
    # and here the size of n will accept any index
    # in the first position
    assert W[2, 1].subs(W, Z) == Z[2, 1]
    # but not in the second position
    raises(IndexError, lambda: W[2, 2].subs(W, Z))
    # any matrix should raise if invalid
    raises(IndexError, lambda: W[2, 2].subs(W, zeros(2)))

    A = SparseMatrix([[1, 2], [3, 4]])
    B = Matrix([[1, 2], [3, 4]])
    C, D = MatrixSymbol('C', 2, 2), MatrixSymbol('D', 2, 2)

    assert (C*D).subs({C: A, D: B}) == MatMul(A, B)


def test_addition():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, m)

    assert isinstance(A + B, MatAdd)
    assert (A + B).shape == A.shape
    assert isinstance(A - A + 2*B, MatMul)

    raises(TypeError, lambda: A + 1)
    raises(TypeError, lambda: 5 + A)
    raises(TypeError, lambda: 5 - A)

    assert A + ZeroMatrix(n, m) - A == ZeroMatrix(n, m)
    raises(TypeError, lambda: ZeroMatrix(n, m) + S.Zero)


def test_multiplication():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    C = MatrixSymbol('C', n, n)

    assert (2*A*B).shape == (n, l)
    assert (A*0*B) == ZeroMatrix(n, l)
    assert (2*A).shape == A.shape

    assert A * ZeroMatrix(m, m) * B == ZeroMatrix(n, l)

    assert C * Identity(n) * C.I == Identity(n)

    assert B/2 == S.Half*B
    raises(NotImplementedError, lambda: 2/B)

    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    assert Identity(n) * (A + B) == A + B

    assert A**2*A == A**3
    assert A**2*(A.I)**3 == A.I
    assert A**3*(A.I)**2 == A


def test_MatPow():
    A = MatrixSymbol('A', n, n)

    AA = MatPow(A, 2)
    assert AA.exp == 2
    assert AA.base == A
    assert (A**n).exp == n

    assert A**0 == Identity(n)
    assert A**1 == A
    assert A**2 == AA
    assert A**-1 == Inverse(A)
    assert (A**-1)**-1 == A
    assert (A**2)**3 == A**6
    assert A**S.Half == sqrt(A)
    assert A**Rational(1, 3) == cbrt(A)
    raises(NonSquareMatrixError, lambda: MatrixSymbol('B', 3, 2)**2)


def test_MatrixSymbol():
    n, m, t = symbols('n,m,t')
    X = MatrixSymbol('X', n, m)
    assert X.shape == (n, m)
    raises(TypeError, lambda: MatrixSymbol('X', n, m)(t))  # issue 5855
    assert X.doit() == X


def test_dense_conversion():
    X = MatrixSymbol('X', 2, 2)
    assert ImmutableMatrix(X) == ImmutableMatrix(2, 2, lambda i, j: X[i, j])
    assert Matrix(X) == Matrix(2, 2, lambda i, j: X[i, j])


def test_free_symbols():
    assert (C*D).free_symbols == {C, D}


def test_zero_matmul():
    assert isinstance(S.Zero * MatrixSymbol('X', 2, 2), MatrixExpr)


def test_matadd_simplify():
    A = MatrixSymbol('A', 1, 1)
    assert simplify(MatAdd(A, ImmutableMatrix([[sin(x)**2 + cos(x)**2]]))) == \
        MatAdd(A, Matrix([[1]]))


def test_matmul_simplify():
    A = MatrixSymbol('A', 1, 1)
    assert simplify(MatMul(A, ImmutableMatrix([[sin(x)**2 + cos(x)**2]]))) == \
        MatMul(A, Matrix([[1]]))


def test_invariants():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    X = MatrixSymbol('X', n, n)
    objs = [Identity(n), ZeroMatrix(m, n), A, MatMul(A, B), MatAdd(A, A),
            Transpose(A), Adjoint(A), Inverse(X), MatPow(X, 2), MatPow(X, -1),
            MatPow(X, 0)]
    for obj in objs:
        assert obj == obj.__class__(*obj.args)


def test_matexpr_indexing():
    A = MatrixSymbol('A', n, m)
    A[1, 2]
    A[l, k]
    A[l + 1, k + 1]
    A = MatrixSymbol('A', 2, 1)
    for i in range(-2, 2):
        for j in range(-1, 1):
            A[i, j]


def test_single_indexing():
    A = MatrixSymbol('A', 2, 3)
    assert A[1] == A[0, 1]
    assert A[int(1)] == A[0, 1]
    assert A[3] == A[1, 0]
    assert list(A[:2, :2]) == [A[0, 0], A[0, 1], A[1, 0], A[1, 1]]
    raises(IndexError, lambda: A[6])
    raises(IndexError, lambda: A[n])
    B = MatrixSymbol('B', n, m)
    raises(IndexError, lambda: B[1])
    B = MatrixSymbol('B', n, 3)
    assert B[3] == B[1, 0]


def test_MatrixElement_commutative():
    assert A[0, 1]*A[1, 0] == A[1, 0]*A[0, 1]


def test_MatrixSymbol_determinant():
    A = MatrixSymbol('A', 4, 4)
    assert A.as_explicit().det() == A[0, 0]*A[1, 1]*A[2, 2]*A[3, 3] - \
        A[0, 0]*A[1, 1]*A[2, 3]*A[3, 2] - A[0, 0]*A[1, 2]*A[2, 1]*A[3, 3] + \
        A[0, 0]*A[1, 2]*A[2, 3]*A[3, 1] + A[0, 0]*A[1, 3]*A[2, 1]*A[3, 2] - \
        A[0, 0]*A[1, 3]*A[2, 2]*A[3, 1] - A[0, 1]*A[1, 0]*A[2, 2]*A[3, 3] + \
        A[0, 1]*A[1, 0]*A[2, 3]*A[3, 2] + A[0, 1]*A[1, 2]*A[2, 0]*A[3, 3] - \
        A[0, 1]*A[1, 2]*A[2, 3]*A[3, 0] - A[0, 1]*A[1, 3]*A[2, 0]*A[3, 2] + \
        A[0, 1]*A[1, 3]*A[2, 2]*A[3, 0] + A[0, 2]*A[1, 0]*A[2, 1]*A[3, 3] - \
        A[0, 2]*A[1, 0]*A[2, 3]*A[3, 1] - A[0, 2]*A[1, 1]*A[2, 0]*A[3, 3] + \
        A[0, 2]*A[1, 1]*A[2, 3]*A[3, 0] + A[0, 2]*A[1, 3]*A[2, 0]*A[3, 1] - \
        A[0, 2]*A[1, 3]*A[2, 1]*A[3, 0] - A[0, 3]*A[1, 0]*A[2, 1]*A[3, 2] + \
        A[0, 3]*A[1, 0]*A[2, 2]*A[3, 1] + A[0, 3]*A[1, 1]*A[2, 0]*A[3, 2] - \
        A[0, 3]*A[1, 1]*A[2, 2]*A[3, 0] - A[0, 3]*A[1, 2]*A[2, 0]*A[3, 1] + \
        A[0, 3]*A[1, 2]*A[2, 1]*A[3, 0]

    B = MatrixSymbol('B', 4, 4)
    assert Determinant(A + B).doit() == det(A + B) == (A + B).det()


def test_MatrixElement_diff():
    assert (A[3, 0]*A[0, 0]).diff(A[0, 0]) == A[3, 0]


def test_MatrixElement_doit():
    u = MatrixSymbol('u', 2, 1)
    v = ImmutableMatrix([3, 5])
    assert u[0, 0].subs(u, v).doit() == v[0, 0]


def test_identity_powers():
    M = Identity(n)
    assert MatPow(M, 3).doit() == M**3
    assert M**n == M
    assert MatPow(M, 0).doit() == M**2
    assert M**-2 == M
    assert MatPow(M, -2).doit() == M**0
    N = Identity(3)
    assert MatPow(N, 2).doit() == N**n
    assert MatPow(N, 3).doit() == N
    assert MatPow(N, -2).doit() == N**4
    assert MatPow(N, 2).doit() == N**0


def test_Zero_power():
    z1 = ZeroMatrix(n, n)
    assert z1**4 == z1
    raises(ValueError, lambda:z1**-2)
    assert z1**0 == Identity(n)
    assert MatPow(z1, 2).doit() == z1**2
    raises(ValueError, lambda:MatPow(z1, -2).doit())
    z2 = ZeroMatrix(3, 3)
    assert MatPow(z2, 4).doit() == z2**4
    raises(ValueError, lambda:z2**-3)
    assert z2**3 == MatPow(z2, 3).doit()
    assert z2**0 == Identity(3)
    raises(ValueError, lambda:MatPow(z2, -1).doit())


def test_matrixelement_diff():
    dexpr = diff((D*w)[k,0], w[p,0])

    assert w[k, p].diff(w[k, p]) == 1
    assert w[k, p].diff(w[0, 0]) == KroneckerDelta(0, k, (0, n-1))*KroneckerDelta(0, p, (0, 0))
    _i_1 = Dummy("_i_1")
    assert dexpr.dummy_eq(Sum(KroneckerDelta(_i_1, p, (0, n-1))*D[k, _i_1], (_i_1, 0, n - 1)))
    assert dexpr.doit() == D[k, p]


def test_MatrixElement_with_values():
    x, y, z, w = symbols("x y z w")
    M = Matrix([[x, y], [z, w]])
    i, j = symbols("i, j")
    Mij = M[i, j]
    assert isinstance(Mij, MatrixElement)
    Ms = SparseMatrix([[2, 3], [4, 5]])
    msij = Ms[i, j]
    assert isinstance(msij, MatrixElement)
    for oi, oj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert Mij.subs({i: oi, j: oj}) == M[oi, oj]
        assert msij.subs({i: oi, j: oj}) == Ms[oi, oj]
    A = MatrixSymbol("A", 2, 2)
    assert A[0, 0].subs(A, M) == x
    assert A[i, j].subs(A, M) == M[i, j]
    assert M[i, j].subs(M, A) == A[i, j]

    assert isinstance(M[3*i - 2, j], MatrixElement)
    assert M[3*i - 2, j].subs({i: 1, j: 0}) == M[1, 0]
    assert isinstance(M[i, 0], MatrixElement)
    assert M[i, 0].subs(i, 0) == M[0, 0]
    assert M[0, i].subs(i, 1) == M[0, 1]

    assert M[i, j].diff(x) == Matrix([[1, 0], [0, 0]])[i, j]

    raises(ValueError, lambda: M[i, 2])
    raises(ValueError, lambda: M[i, -1])
    raises(ValueError, lambda: M[2, i])
    raises(ValueError, lambda: M[-1, i])


def test_inv():
    B = MatrixSymbol('B', 3, 3)
    assert B.inv() == B**-1

    # https://github.com/sympy/sympy/issues/19162
    X = MatrixSymbol('X', 1, 1).as_explicit()
    assert X.inv() == Matrix([[1/X[0, 0]]])

    X = MatrixSymbol('X', 2, 2).as_explicit()
    detX = X[0, 0]*X[1, 1] - X[0, 1]*X[1, 0]
    invX = Matrix([[ X[1, 1], -X[0, 1]],
                   [-X[1, 0],  X[0, 0]]]) / detX
    assert X.inv() == invX


@XFAIL
def test_factor_expand():
    A = MatrixSymbol("A", n, n)
    B = MatrixSymbol("B", n, n)
    expr1 = (A + B)*(C + D)
    expr2 = A*C + B*C + A*D + B*D
    assert expr1 != expr2
    assert expand(expr1) == expr2
    assert factor(expr2) == expr1

    expr = B**(-1)*(A**(-1)*B**(-1) - A**(-1)*C*B**(-1))**(-1)*A**(-1)
    I = Identity(n)
    # Ideally we get the first, but we at least don't want a wrong answer
    assert factor(expr) in [I - C, B**-1*(A**-1*(I - C)*B**-1)**-1*A**-1]

def test_numpy_conversion():
    try:
        from numpy import array, array_equal
    except ImportError:
        skip('NumPy must be available to test creating matrices from ndarrays')
    A = MatrixSymbol('A', 2, 2)
    np_array = array([[MatrixElement(A, 0, 0), MatrixElement(A, 0, 1)],
    [MatrixElement(A, 1, 0), MatrixElement(A, 1, 1)]])
    assert array_equal(array(A), np_array)
    assert array_equal(array(A, copy=True), np_array)
    if(int(version('numpy').split('.')[0]) >= 2): #run this test only if numpy is new enough that copy variable is passed properly.
        raises(TypeError, lambda: array(A, copy=False))

def test_issue_2749():
    A = MatrixSymbol("A", 5, 2)
    assert (A.T * A).I.as_explicit() == Matrix([[(A.T * A).I[0, 0], (A.T * A).I[0, 1]], \
    [(A.T * A).I[1, 0], (A.T * A).I[1, 1]]])


def test_issue_2750():
    x = MatrixSymbol('x', 1, 1)
    assert (x.T*x).as_explicit()**-1 == Matrix([[x[0, 0]**(-2)]])


def test_issue_7842():
    A = MatrixSymbol('A', 3, 1)
    B = MatrixSymbol('B', 2, 1)
    assert Eq(A, B) == False
    assert Eq(A[1,0], B[1, 0]).func is Eq
    A = ZeroMatrix(2, 3)
    B = ZeroMatrix(2, 3)
    assert Eq(A, B) == True


def test_issue_21195():
    t = symbols('t')
    x = Function('x')(t)
    dx = x.diff(t)
    exp1 = cos(x) + cos(x)*dx
    exp2 = sin(x) + tan(x)*(dx.diff(t))
    exp3 = sin(x)*sin(t)*(dx.diff(t)).diff(t)
    A = Matrix([[exp1], [exp2], [exp3]])
    B = Matrix([[exp1.diff(x)], [exp2.diff(x)], [exp3.diff(x)]])
    assert A.diff(x) == B


def test_issue_24859():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 3, 2)
    J = A*B
    Jinv = Matrix(J).adjugate()
    u = MatrixSymbol('u', 2, 3)
    Jk = Jinv.subs(A, A + x*u)

    expected = B[0, 1]*u[1, 0] + B[1, 1]*u[1, 1] + B[2, 1]*u[1, 2]
    assert Jk[0, 0].diff(x) == expected
    assert diff(Jk[0, 0], x).doit() == expected


def test_MatMul_postprocessor():
    z = zeros(2)
    z1 = ZeroMatrix(2, 2)
    assert Mul(0, z) == Mul(z, 0) in [z, z1]

    M = Matrix([[1, 2], [3, 4]])
    Mx = Matrix([[x, 2*x], [3*x, 4*x]])
    assert Mul(x, M) == Mul(M, x) == Mx

    A = MatrixSymbol("A", 2, 2)
    assert Mul(A, M) == MatMul(A, M)
    assert Mul(M, A) == MatMul(M, A)
    # Scalars should be absorbed into constant matrices
    a = Mul(x, M, A)
    b = Mul(M, x, A)
    c = Mul(M, A, x)
    assert a == b == c == MatMul(Mx, A)
    a = Mul(x, A, M)
    b = Mul(A, x, M)
    c = Mul(A, M, x)
    assert a == b == c == MatMul(A, Mx)
    assert Mul(M, M) == M**2
    assert Mul(A, M, M) == MatMul(A, M**2)
    assert Mul(M, M, A) == MatMul(M**2, A)
    assert Mul(M, A, M) == MatMul(M, A, M)

    assert Mul(A, x, M, M, x) == MatMul(A, Mx**2)


@XFAIL
def test_MatAdd_postprocessor_xfail():
    # This is difficult to get working because of the way that Add processes
    # its args.
    z = zeros(2)
    assert Add(z, S.NaN) == Add(S.NaN, z)


def test_MatAdd_postprocessor():
    # Some of these are nonsensical, but we do not raise errors for Add
    # because that breaks algorithms that want to replace matrices with dummy
    # symbols.

    z = zeros(2)

    assert Add(0, z) == Add(z, 0) == z

    a = Add(S.Infinity, z)
    assert a == Add(z, S.Infinity)
    assert isinstance(a, Add)
    assert a.args == (S.Infinity, z)

    a = Add(S.ComplexInfinity, z)
    assert a == Add(z, S.ComplexInfinity)
    assert isinstance(a, Add)
    assert a.args == (S.ComplexInfinity, z)

    a = Add(z, S.NaN)
    # assert a == Add(S.NaN, z) # See the XFAIL above
    assert isinstance(a, Add)
    assert a.args == (S.NaN, z)

    M = Matrix([[1, 2], [3, 4]])
    a = Add(x, M)
    assert a == Add(M, x)
    assert isinstance(a, Add)
    assert a.args == (x, M)

    A = MatrixSymbol("A", 2, 2)
    assert Add(A, M) == Add(M, A) == A + M

    # Scalars should be absorbed into constant matrices (producing an error)
    a = Add(x, M, A)
    assert a == Add(M, x, A) == Add(M, A, x) == Add(x, A, M) == Add(A, x, M) == Add(A, M, x)
    assert isinstance(a, Add)
    assert a.args == (x, A + M)

    assert Add(M, M) == 2*M
    assert Add(M, A, M) == Add(M, M, A) == Add(A, M, M) == A + 2*M

    a = Add(A, x, M, M, x)
    assert isinstance(a, Add)
    assert a.args == (2*x, A + 2*M)


def test_simplify_matrix_expressions():
    # Various simplification functions
    assert type(gcd_terms(C*D + D*C)) == MatAdd
    a = gcd_terms(2*C*D + 4*D*C)
    assert type(a) == MatAdd
    assert a.args == (2*C*D, 4*D*C)


def test_exp():
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    expr1 = exp(A)*exp(B)
    expr2 = exp(B)*exp(A)
    assert expr1 != expr2
    assert expr1 - expr2 != 0
    assert not isinstance(expr1, exp)
    assert not isinstance(expr2, exp)


def test_invalid_args():
    raises(SympifyError, lambda: MatrixSymbol(1, 2, 'A'))


def test_matrixsymbol_from_symbol():
    # The label should be preserved during doit and subs
    A_label = Symbol('A', complex=True)
    A = MatrixSymbol(A_label, 2, 2)

    A_1 = A.doit()
    A_2 = A.subs(2, 3)
    assert A_1.args == A.args
    assert A_2.args[0] == A.args[0]


def test_as_explicit():
    Z = MatrixSymbol('Z', 2, 3)
    assert Z.as_explicit() == ImmutableMatrix([
        [Z[0, 0], Z[0, 1], Z[0, 2]],
        [Z[1, 0], Z[1, 1], Z[1, 2]],
    ])
    raises(ValueError, lambda: A.as_explicit())


def test_MatrixSet():
    M = MatrixSet(2, 2, set=S.Reals)
    assert M.shape == (2, 2)
    assert M.set == S.Reals
    X = Matrix([[1, 2], [3, 4]])
    assert X in M
    X = ZeroMatrix(2, 2)
    assert X in M
    raises(TypeError, lambda: A in M)
    raises(TypeError, lambda: 1 in M)
    M = MatrixSet(n, m, set=S.Reals)
    assert A in M
    raises(TypeError, lambda: C in M)
    raises(TypeError, lambda: X in M)
    M = MatrixSet(2, 2, set={1, 2, 3})
    X = Matrix([[1, 2], [3, 4]])
    Y = Matrix([[1, 2]])
    assert (X in M) == S.false
    assert (Y in M) == S.false
    raises(ValueError, lambda: MatrixSet(2, -2, S.Reals))
    raises(ValueError, lambda: MatrixSet(2.4, -1, S.Reals))
    raises(TypeError, lambda: MatrixSet(2, 2, (1, 2, 3)))


def test_matrixsymbol_solving():
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    Z = ZeroMatrix(2, 2)
    assert -(-A + B) - A + B == Z
    assert (-(-A + B) - A + B).simplify() == Z
    assert (-(-A + B) - A + B).expand() == Z
    assert (-(-A + B) - A + B - Z).simplify() == Z
    assert (-(-A + B) - A + B - Z).expand() == Z
    assert (A*(A + B) + B*(A.T + B.T)).expand() == A**2 + A*B + B*A.T + B*B.T
