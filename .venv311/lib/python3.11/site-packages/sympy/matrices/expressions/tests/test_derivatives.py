"""
Some examples have been taken from:

http://www.math.uwaterloo.ca/~hwolkowi//matrixcookbook.pdf
"""
from sympy import KroneckerProduct
from sympy.combinatorics import Permutation
from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct, hadamard_product)
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import OneMatrix
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import (Identity, ZeroMatrix)
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.matrices.expressions import hadamard_power
from sympy.tensor.array.expressions.array_expressions import ArrayAdd, ArrayTensorProduct, PermuteDims

i, j, k = symbols("i j k")
m, n = symbols("m n")

X = MatrixSymbol("X", k, k)
x = MatrixSymbol("x", k, 1)
y = MatrixSymbol("y", k, 1)

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)
c = MatrixSymbol("c", k, 1)
d = MatrixSymbol("d", k, 1)


KDelta = lambda i, j: KroneckerDelta(i, j, (0, k-1))


def _check_derivative_with_explicit_matrix(expr, x, diffexpr, dim=2):
    # TODO: this is commented because it slows down the tests.
    return

    expr = expr.xreplace({k: dim})
    x = x.xreplace({k: dim})
    diffexpr = diffexpr.xreplace({k: dim})

    expr = expr.as_explicit()
    x = x.as_explicit()
    diffexpr = diffexpr.as_explicit()

    assert expr.diff(x).reshape(*diffexpr.shape).tomatrix() == diffexpr


def test_matrix_derivative_by_scalar():
    assert A.diff(i) == ZeroMatrix(k, k)
    assert (A*(X + B)*c).diff(i) == ZeroMatrix(k, 1)
    assert x.diff(i) == ZeroMatrix(k, 1)
    assert (x.T*y).diff(i) == ZeroMatrix(1, 1)
    assert (x*x.T).diff(i) == ZeroMatrix(k, k)
    assert (x + y).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_power(x, 2).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_power(x, i).diff(i).dummy_eq(
        HadamardProduct(x.applyfunc(log), HadamardPower(x, i)))
    assert hadamard_product(x, y).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_product(i*OneMatrix(k, 1), x, y).diff(i) == hadamard_product(x, y)
    assert (i*x).diff(i) == x
    assert (sin(i)*A*B*x).diff(i) == cos(i)*A*B*x
    assert x.applyfunc(sin).diff(i) == ZeroMatrix(k, 1)
    assert Trace(i**2*X).diff(i) == 2*i*Trace(X)

    mu = symbols("mu")
    expr = (2*mu*x)
    assert expr.diff(x) == 2*mu*Identity(k)


def test_one_matrix():
    assert MatMul(x.T, OneMatrix(k, 1)).diff(x) == OneMatrix(k, 1)


def test_matrix_derivative_non_matrix_result():
    # This is a 4-dimensional array:
    I = Identity(k)
    AdA = PermuteDims(ArrayTensorProduct(I, I), Permutation(3)(1, 2))
    assert A.diff(A) == AdA
    assert A.T.diff(A) == PermuteDims(ArrayTensorProduct(I, I), Permutation(3)(1, 2, 3))
    assert (2*A).diff(A) == PermuteDims(ArrayTensorProduct(2*I, I), Permutation(3)(1, 2))
    assert MatAdd(A, A).diff(A) == ArrayAdd(AdA, AdA)
    assert (A + B).diff(A) == AdA


def test_matrix_derivative_trivial_cases():
    # Cookbook example 33:
    # TODO: find a way to represent a four-dimensional zero-array:
    assert X.diff(A) == ArrayDerivative(X, A)


def test_matrix_derivative_with_inverse():

    # Cookbook example 61:
    expr = a.T*Inverse(X)*b
    assert expr.diff(X) == -Inverse(X).T*a*b.T*Inverse(X).T

    # Cookbook example 62:
    expr = Determinant(Inverse(X))
    # Not implemented yet:
    # assert expr.diff(X) == -Determinant(X.inv())*(X.inv()).T

    # Cookbook example 63:
    expr = Trace(A*Inverse(X)*B)
    assert expr.diff(X) == -(X**(-1)*B*A*X**(-1)).T

    # Cookbook example 64:
    expr = Trace(Inverse(X + A))
    assert expr.diff(X) == -(Inverse(X + A)).T**2


def test_matrix_derivative_vectors_and_scalars():

    assert x.diff(x) == Identity(k)
    assert x[i, 0].diff(x[m, 0]).doit() == KDelta(m, i)

    assert x.T.diff(x) == Identity(k)

    # Cookbook example 69:
    expr = x.T*a
    assert expr.diff(x) == a
    assert expr[0, 0].diff(x[m, 0]).doit() == a[m, 0]
    expr = a.T*x
    assert expr.diff(x) == a

    # Cookbook example 70:
    expr = a.T*X*b
    assert expr.diff(X) == a*b.T

    # Cookbook example 71:
    expr = a.T*X.T*b
    assert expr.diff(X) == b*a.T

    # Cookbook example 72:
    expr = a.T*X*a
    assert expr.diff(X) == a*a.T
    expr = a.T*X.T*a
    assert expr.diff(X) == a*a.T

    # Cookbook example 77:
    expr = b.T*X.T*X*c
    assert expr.diff(X) == X*b*c.T + X*c*b.T

    # Cookbook example 78:
    expr = (B*x + b).T*C*(D*x + d)
    assert expr.diff(x) == B.T*C*(D*x + d) + D.T*C.T*(B*x + b)

    # Cookbook example 81:
    expr = x.T*B*x
    assert expr.diff(x) == B*x + B.T*x

    # Cookbook example 82:
    expr = b.T*X.T*D*X*c
    assert expr.diff(X) == D.T*X*b*c.T + D*X*c*b.T

    # Cookbook example 83:
    expr = (X*b + c).T*D*(X*b + c)
    assert expr.diff(X) == D*(X*b + c)*b.T + D.T*(X*b + c)*b.T
    assert str(expr[0, 0].diff(X[m, n]).doit()) == \
        'b[n, 0]*Sum((c[_i_1, 0] + Sum(X[_i_1, _i_3]*b[_i_3, 0], (_i_3, 0, k - 1)))*D[_i_1, m], (_i_1, 0, k - 1)) + Sum((c[_i_2, 0] + Sum(X[_i_2, _i_4]*b[_i_4, 0], (_i_4, 0, k - 1)))*D[m, _i_2]*b[n, 0], (_i_2, 0, k - 1))'

    # See https://github.com/sympy/sympy/issues/16504#issuecomment-1018339957
    expr = x*x.T*x
    I = Identity(k)
    assert expr.diff(x) == KroneckerProduct(I, x.T*x) + 2*x*x.T


def test_matrix_derivatives_of_traces():

    expr = Trace(A)*A
    I = Identity(k)
    assert expr.diff(A) == ArrayAdd(ArrayTensorProduct(I, A), PermuteDims(ArrayTensorProduct(Trace(A)*I, I), Permutation(3)(1, 2)))
    assert expr[i, j].diff(A[m, n]).doit() == (
        KDelta(i, m)*KDelta(j, n)*Trace(A) +
        KDelta(m, n)*A[i, j]
    )

    ## First order:

    # Cookbook example 99:
    expr = Trace(X)
    assert expr.diff(X) == Identity(k)
    assert expr.rewrite(Sum).diff(X[m, n]).doit() == KDelta(m, n)

    # Cookbook example 100:
    expr = Trace(X*A)
    assert expr.diff(X) == A.T
    assert expr.rewrite(Sum).diff(X[m, n]).doit() == A[n, m]

    # Cookbook example 101:
    expr = Trace(A*X*B)
    assert expr.diff(X) == A.T*B.T
    assert expr.rewrite(Sum).diff(X[m, n]).doit().dummy_eq((A.T*B.T)[m, n])

    # Cookbook example 102:
    expr = Trace(A*X.T*B)
    assert expr.diff(X) == B*A

    # Cookbook example 103:
    expr = Trace(X.T*A)
    assert expr.diff(X) == A

    # Cookbook example 104:
    expr = Trace(A*X.T)
    assert expr.diff(X) == A

    # Cookbook example 105:
    # TODO: TensorProduct is not supported
    #expr = Trace(TensorProduct(A, X))
    #assert expr.diff(X) == Trace(A)*Identity(k)

    ## Second order:

    # Cookbook example 106:
    expr = Trace(X**2)
    assert expr.diff(X) == 2*X.T

    # Cookbook example 107:
    expr = Trace(X**2*B)
    assert expr.diff(X) == (X*B + B*X).T
    expr = Trace(MatMul(X, X, B))
    assert expr.diff(X) == (X*B + B*X).T

    # Cookbook example 108:
    expr = Trace(X.T*B*X)
    assert expr.diff(X) == B*X + B.T*X

    # Cookbook example 109:
    expr = Trace(B*X*X.T)
    assert expr.diff(X) == B*X + B.T*X

    # Cookbook example 110:
    expr = Trace(X*X.T*B)
    assert expr.diff(X) == B*X + B.T*X

    # Cookbook example 111:
    expr = Trace(X*B*X.T)
    assert expr.diff(X) == X*B.T + X*B

    # Cookbook example 112:
    expr = Trace(B*X.T*X)
    assert expr.diff(X) == X*B.T + X*B

    # Cookbook example 113:
    expr = Trace(X.T*X*B)
    assert expr.diff(X) == X*B.T + X*B

    # Cookbook example 114:
    expr = Trace(A*X*B*X)
    assert expr.diff(X) == A.T*X.T*B.T + B.T*X.T*A.T

    # Cookbook example 115:
    expr = Trace(X.T*X)
    assert expr.diff(X) == 2*X
    expr = Trace(X*X.T)
    assert expr.diff(X) == 2*X

    # Cookbook example 116:
    expr = Trace(B.T*X.T*C*X*B)
    assert expr.diff(X) == C.T*X*B*B.T + C*X*B*B.T

    # Cookbook example 117:
    expr = Trace(X.T*B*X*C)
    assert expr.diff(X) == B*X*C + B.T*X*C.T

    # Cookbook example 118:
    expr = Trace(A*X*B*X.T*C)
    assert expr.diff(X) == A.T*C.T*X*B.T + C*A*X*B

    # Cookbook example 119:
    expr = Trace((A*X*B + C)*(A*X*B + C).T)
    assert expr.diff(X) == 2*A.T*(A*X*B + C)*B.T

    # Cookbook example 120:
    # TODO: no support for TensorProduct.
    # expr = Trace(TensorProduct(X, X))
    # expr = Trace(X)*Trace(X)
    # expr.diff(X) == 2*Trace(X)*Identity(k)

    # Higher Order

    # Cookbook example 121:
    expr = Trace(X**k)
    #assert expr.diff(X) == k*(X**(k-1)).T

    # Cookbook example 122:
    expr = Trace(A*X**k)
    #assert expr.diff(X) == # Needs indices

    # Cookbook example 123:
    expr = Trace(B.T*X.T*C*X*X.T*C*X*B)
    assert expr.diff(X) == C*X*X.T*C*X*B*B.T + C.T*X*B*B.T*X.T*C.T*X + C*X*B*B.T*X.T*C*X + C.T*X*X.T*C.T*X*B*B.T

    # Other

    # Cookbook example 124:
    expr = Trace(A*X**(-1)*B)
    assert expr.diff(X) == -Inverse(X).T*A.T*B.T*Inverse(X).T

    # Cookbook example 125:
    expr = Trace(Inverse(X.T*C*X)*A)
    # Warning: result in the cookbook is equivalent if B and C are symmetric:
    assert expr.diff(X) == - X.inv().T*A.T*X.inv()*C.inv().T*X.inv().T - X.inv().T*A*X.inv()*C.inv()*X.inv().T

    # Cookbook example 126:
    expr = Trace((X.T*C*X).inv()*(X.T*B*X))
    assert expr.diff(X) == -2*C*X*(X.T*C*X).inv()*X.T*B*X*(X.T*C*X).inv() + 2*B*X*(X.T*C*X).inv()

    # Cookbook example 127:
    expr = Trace((A + X.T*C*X).inv()*(X.T*B*X))
    # Warning: result in the cookbook is equivalent if B and C are symmetric:
    assert expr.diff(X) == B*X*Inverse(A + X.T*C*X) - C*X*Inverse(A + X.T*C*X)*X.T*B*X*Inverse(A + X.T*C*X) - C.T*X*Inverse(A.T + (C*X).T*X)*X.T*B.T*X*Inverse(A.T + (C*X).T*X) + B.T*X*Inverse(A.T + (C*X).T*X)


def test_derivatives_of_complicated_matrix_expr():
    expr = a.T*(A*X*(X.T*B + X*A) + B.T*X.T*(a*b.T*(X*D*X.T + X*(X.T*B + A*X)*D*B - X.T*C.T*A)*B + B*(X*D.T + B*A*X*A.T - 3*X*D))*B + 42*X*B*X.T*A.T*(X + X.T))*b
    result = (B*(B*A*X*A.T - 3*X*D + X*D.T) + a*b.T*(X*(A*X + X.T*B)*D*B + X*D*X.T - X.T*C.T*A)*B)*B*b*a.T*B.T + B**2*b*a.T*B.T*X.T*a*b.T*X*D + 42*A*X*B.T*X.T*a*b.T + B*D*B**3*b*a.T*B.T*X.T*a*b.T*X + B*b*a.T*A*X + a*b.T*(42*X + 42*X.T)*A*X*B.T + b*a.T*X*B*a*b.T*B.T**2*X*D.T + b*a.T*X*B*a*b.T*B.T**3*D.T*(B.T*X + X.T*A.T) + 42*b*a.T*X*B*X.T*A.T + A.T*(42*X + 42*X.T)*b*a.T*X*B + A.T*B.T**2*X*B*a*b.T*B.T*A + A.T*a*b.T*(A.T*X.T + B.T*X) + A.T*X.T*b*a.T*X*B*a*b.T*B.T**3*D.T + B.T*X*B*a*b.T*B.T*D - 3*B.T*X*B*a*b.T*B.T*D.T - C.T*A*B**2*b*a.T*B.T*X.T*a*b.T + X.T*A.T*a*b.T*A.T
    assert expr.diff(X) == result


def test_mixed_deriv_mixed_expressions():

    expr = 3*Trace(A)
    assert expr.diff(A) == 3*Identity(k)

    expr = k
    deriv = expr.diff(A)
    assert isinstance(deriv, ZeroMatrix)
    assert deriv == ZeroMatrix(k, k)

    expr = Trace(A)**2
    assert expr.diff(A) == (2*Trace(A))*Identity(k)

    expr = Trace(A)*A
    I = Identity(k)
    assert expr.diff(A) == ArrayAdd(ArrayTensorProduct(I, A), PermuteDims(ArrayTensorProduct(Trace(A)*I, I), Permutation(3)(1, 2)))

    expr = Trace(Trace(A)*A)
    assert expr.diff(A) == (2*Trace(A))*Identity(k)

    expr = Trace(Trace(Trace(A)*A)*A)
    assert expr.diff(A) == (3*Trace(A)**2)*Identity(k)


def test_derivatives_matrix_norms():

    expr = x.T*y
    assert expr.diff(x) == y
    assert expr[0, 0].diff(x[m, 0]).doit() == y[m, 0]

    expr = (x.T*y)**S.Half
    assert expr.diff(x) == y/(2*sqrt(x.T*y))

    expr = (x.T*x)**S.Half
    assert expr.diff(x) == x*(x.T*x)**Rational(-1, 2)

    expr = (c.T*a*x.T*b)**S.Half
    assert expr.diff(x) == b*a.T*c/sqrt(c.T*a*x.T*b)/2

    expr = (c.T*a*x.T*b)**Rational(1, 3)
    assert expr.diff(x) == b*a.T*c*(c.T*a*x.T*b)**Rational(-2, 3)/3

    expr = (a.T*X*b)**S.Half
    assert expr.diff(X) == a/(2*sqrt(a.T*X*b))*b.T

    expr = d.T*x*(a.T*X*b)**S.Half*y.T*c
    assert expr.diff(X) == a/(2*sqrt(a.T*X*b))*x.T*d*y.T*c*b.T


def test_derivatives_elementwise_applyfunc():

    expr = x.applyfunc(tan)
    assert expr.diff(x).dummy_eq(
        DiagMatrix(x.applyfunc(lambda x: tan(x)**2 + 1)))
    assert expr[i, 0].diff(x[m, 0]).doit() == (tan(x[i, 0])**2 + 1)*KDelta(i, m)
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    expr = (i**2*x).applyfunc(sin)
    assert expr.diff(i).dummy_eq(
        HadamardProduct((2*i)*x, (i**2*x).applyfunc(cos)))
    assert expr[i, 0].diff(i).doit() == 2*i*x[i, 0]*cos(i**2*x[i, 0])
    _check_derivative_with_explicit_matrix(expr, i, expr.diff(i))

    expr = (log(i)*A*B).applyfunc(sin)
    assert expr.diff(i).dummy_eq(
        HadamardProduct(A*B/i, (log(i)*A*B).applyfunc(cos)))
    _check_derivative_with_explicit_matrix(expr, i, expr.diff(i))

    expr = A*x.applyfunc(exp)
    # TODO: restore this result (currently returning the transpose):
    #  assert expr.diff(x).dummy_eq(DiagMatrix(x.applyfunc(exp))*A.T)
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    expr = x.T*A*x + k*y.applyfunc(sin).T*x
    assert expr.diff(x).dummy_eq(A.T*x + A*x + k*y.applyfunc(sin))
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    expr = x.applyfunc(sin).T*y
    # TODO: restore (currently returning the transpose):
    #  assert expr.diff(x).dummy_eq(DiagMatrix(x.applyfunc(cos))*y)
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    expr = (a.T * X * b).applyfunc(sin)
    assert expr.diff(X).dummy_eq(a*(a.T*X*b).applyfunc(cos)*b.T)
    _check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    expr = a.T * X.applyfunc(sin) * b
    assert expr.diff(X).dummy_eq(
        DiagMatrix(a)*X.applyfunc(cos)*DiagMatrix(b))
    _check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    expr = a.T * (A*X*B).applyfunc(sin) * b
    assert expr.diff(X).dummy_eq(
        A.T*DiagMatrix(a)*(A*X*B).applyfunc(cos)*DiagMatrix(b)*B.T)
    _check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    expr = a.T * (A*X*b).applyfunc(sin) * b.T
    # TODO: not implemented
    #assert expr.diff(X) == ...
    #_check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    expr = a.T*A*X.applyfunc(sin)*B*b
    assert expr.diff(X).dummy_eq(
        HadamardProduct(A.T * a * b.T * B.T, X.applyfunc(cos)))

    expr = a.T * (A*X.applyfunc(sin)*B).applyfunc(log) * b
    # TODO: wrong
    # assert expr.diff(X) == A.T*DiagMatrix(a)*(A*X.applyfunc(sin)*B).applyfunc(Lambda(k, 1/k))*DiagMatrix(b)*B.T

    expr = a.T * (X.applyfunc(sin)).applyfunc(log) * b
    # TODO: wrong
    # assert expr.diff(X) == DiagMatrix(a)*X.applyfunc(sin).applyfunc(Lambda(k, 1/k))*DiagMatrix(b)


def test_derivatives_of_hadamard_expressions():

    # Hadamard Product

    expr = hadamard_product(a, x, b)
    assert expr.diff(x) == DiagMatrix(hadamard_product(b, a))

    expr = a.T*hadamard_product(A, X, B)*b
    assert expr.diff(X) == HadamardProduct(a*b.T, A, B)

    # Hadamard Power

    expr = hadamard_power(x, 2)
    assert expr.diff(x).doit() == 2*DiagMatrix(x)

    expr = hadamard_power(x.T, 2)
    assert expr.diff(x).doit() == 2*DiagMatrix(x)

    expr = hadamard_power(x, S.Half)
    assert expr.diff(x) == S.Half*DiagMatrix(hadamard_power(x, Rational(-1, 2)))

    expr = hadamard_power(a.T*X*b, 2)
    assert expr.diff(X) == 2*a*a.T*X*b*b.T

    expr = hadamard_power(a.T*X*b, S.Half)
    assert expr.diff(X) == a/(2*sqrt(a.T*X*b))*b.T
