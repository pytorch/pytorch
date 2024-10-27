from sympy.core.add import Add
from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.core.relational import Eq
from sympy.concrete.summations import Sum
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import (
    ZeroMatrix, GenericZeroMatrix, Identity, GenericIdentity, OneMatrix)
from sympy.matrices.expressions.matmul import MatMul
from sympy.testing.pytest import raises


def test_zero_matrix_creation():
    assert unchanged(ZeroMatrix, 2, 2)
    assert unchanged(ZeroMatrix, 0, 0)
    raises(ValueError, lambda: ZeroMatrix(-1, 2))
    raises(ValueError, lambda: ZeroMatrix(2.0, 2))
    raises(ValueError, lambda: ZeroMatrix(2j, 2))
    raises(ValueError, lambda: ZeroMatrix(2, -1))
    raises(ValueError, lambda: ZeroMatrix(2, 2.0))
    raises(ValueError, lambda: ZeroMatrix(2, 2j))

    n = symbols('n')
    assert unchanged(ZeroMatrix, n, n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: ZeroMatrix(n, n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: ZeroMatrix(n, n))


def test_generic_zero_matrix():
    z = GenericZeroMatrix()
    n = symbols('n', integer=True)
    A = MatrixSymbol("A", n, n)

    assert z == z
    assert z != A
    assert A != z

    assert z.is_ZeroMatrix

    raises(TypeError, lambda: z.shape)
    raises(TypeError, lambda: z.rows)
    raises(TypeError, lambda: z.cols)

    assert MatAdd() == z
    assert MatAdd(z, A) == MatAdd(A)
    # Make sure it is hashable
    hash(z)


def test_identity_matrix_creation():
    assert Identity(2)
    assert Identity(0)
    raises(ValueError, lambda: Identity(-1))
    raises(ValueError, lambda: Identity(2.0))
    raises(ValueError, lambda: Identity(2j))

    n = symbols('n')
    assert Identity(n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: Identity(n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: Identity(n))


def test_generic_identity():
    I = GenericIdentity()
    n = symbols('n', integer=True)
    A = MatrixSymbol("A", n, n)

    assert I == I
    assert I != A
    assert A != I

    assert I.is_Identity
    assert I**-1 == I

    raises(TypeError, lambda: I.shape)
    raises(TypeError, lambda: I.rows)
    raises(TypeError, lambda: I.cols)

    assert MatMul() == I
    assert MatMul(I, A) == MatMul(A)
    # Make sure it is hashable
    hash(I)


def test_one_matrix_creation():
    assert OneMatrix(2, 2)
    assert OneMatrix(0, 0)
    assert Eq(OneMatrix(1, 1), Identity(1))
    raises(ValueError, lambda: OneMatrix(-1, 2))
    raises(ValueError, lambda: OneMatrix(2.0, 2))
    raises(ValueError, lambda: OneMatrix(2j, 2))
    raises(ValueError, lambda: OneMatrix(2, -1))
    raises(ValueError, lambda: OneMatrix(2, 2.0))
    raises(ValueError, lambda: OneMatrix(2, 2j))

    n = symbols('n')
    assert OneMatrix(n, n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: OneMatrix(n, n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: OneMatrix(n, n))


def test_ZeroMatrix():
    n, m = symbols('n m', integer=True)
    A = MatrixSymbol('A', n, m)
    Z = ZeroMatrix(n, m)

    assert A + Z == A
    assert A*Z.T == ZeroMatrix(n, n)
    assert Z*A.T == ZeroMatrix(n, n)
    assert A - A == ZeroMatrix(*A.shape)

    assert Z

    assert Z.transpose() == ZeroMatrix(m, n)
    assert Z.conjugate() == Z
    assert Z.adjoint() == ZeroMatrix(m, n)
    assert re(Z) == Z
    assert im(Z) == Z

    assert ZeroMatrix(n, n)**0 == Identity(n)
    assert ZeroMatrix(3, 3).as_explicit() == ImmutableDenseMatrix.zeros(3, 3)


def test_ZeroMatrix_doit():
    n = symbols('n', integer=True)
    Znn = ZeroMatrix(Add(n, n, evaluate=False), n)
    assert isinstance(Znn.rows, Add)
    assert Znn.doit() == ZeroMatrix(2*n, n)
    assert isinstance(Znn.doit().rows, Mul)


def test_OneMatrix():
    n, m = symbols('n m', integer=True)
    A = MatrixSymbol('A', n, m)
    U = OneMatrix(n, m)

    assert U.shape == (n, m)
    assert isinstance(A + U, Add)
    assert U.transpose() == OneMatrix(m, n)
    assert U.conjugate() == U
    assert U.adjoint() == OneMatrix(m, n)
    assert re(U) == U
    assert im(U) == ZeroMatrix(n, m)

    assert OneMatrix(n, n) ** 0 == Identity(n)

    U = OneMatrix(n, n)
    assert U[1, 2] == 1

    U = OneMatrix(2, 3)
    assert U.as_explicit() == ImmutableDenseMatrix.ones(2, 3)


def test_OneMatrix_doit():
    n = symbols('n', integer=True)
    Unn = OneMatrix(Add(n, n, evaluate=False), n)
    assert isinstance(Unn.rows, Add)
    assert Unn.doit() == OneMatrix(2 * n, n)
    assert isinstance(Unn.doit().rows, Mul)


def test_OneMatrix_mul():
    n, m, k = symbols('n m k', integer=True)
    w = MatrixSymbol('w', n, 1)
    assert OneMatrix(n, m) * OneMatrix(m, k) == OneMatrix(n, k) * m
    assert w * OneMatrix(1, 1) == w
    assert OneMatrix(1, 1) * w.T == w.T


def test_Identity():
    n, m = symbols('n m', integer=True)
    A = MatrixSymbol('A', n, m)
    i, j = symbols('i j')

    In = Identity(n)
    Im = Identity(m)

    assert A*Im == A
    assert In*A == A

    assert In.transpose() == In
    assert In.inverse() == In
    assert In.conjugate() == In
    assert In.adjoint() == In
    assert re(In) == In
    assert im(In) == ZeroMatrix(n, n)

    assert In[i, j] != 0
    assert Sum(In[i, j], (i, 0, n-1), (j, 0, n-1)).subs(n,3).doit() == 3
    assert Sum(Sum(In[i, j], (i, 0, n-1)), (j, 0, n-1)).subs(n,3).doit() == 3

    # If range exceeds the limit `(0, n-1)`, do not remove `Piecewise`:
    expr = Sum(In[i, j], (i, 0, n-1))
    assert expr.doit() == 1
    expr = Sum(In[i, j], (i, 0, n-2))
    assert expr.doit().dummy_eq(
        Piecewise(
            (1, (j >= 0) & (j <= n-2)),
            (0, True)
        )
    )
    expr = Sum(In[i, j], (i, 1, n-1))
    assert expr.doit().dummy_eq(
        Piecewise(
            (1, (j >= 1) & (j <= n-1)),
            (0, True)
        )
    )
    assert Identity(3).as_explicit() == ImmutableDenseMatrix.eye(3)


def test_Identity_doit():
    n = symbols('n', integer=True)
    Inn = Identity(Add(n, n, evaluate=False))
    assert isinstance(Inn.rows, Add)
    assert Inn.doit() == Identity(2*n)
    assert isinstance(Inn.doit().rows, Mul)
