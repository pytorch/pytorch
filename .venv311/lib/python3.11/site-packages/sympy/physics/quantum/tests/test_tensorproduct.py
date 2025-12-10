from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.core.expr import unchanged
from sympy.matrices import Matrix, SparseMatrix, ImmutableMatrix
from sympy.testing.pytest import warns_deprecated_sympy

from sympy.physics.quantum.commutator import Commutator as Comm
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.tensorproduct import TensorProduct as TP
from sympy.physics.quantum.tensorproduct import tensor_product_simp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qubit import Qubit, QubitBra
from sympy.physics.quantum.operator import OuterProduct, Operator
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr

A = Operator('A')
B = Operator('B')
C = Operator('C')
D = Operator('D')
x = symbols('x')
y = symbols('y', integer=True, positive=True)

mat1 = Matrix([[1, 2*I], [1 + I, 3]])
mat2 = Matrix([[2*I, 3], [4*I, 2]])


def test_sparse_matrices():
    spm = SparseMatrix.diag(1, 0)
    assert unchanged(TensorProduct, spm, spm)


def test_tensor_product_dagger():
    assert Dagger(TensorProduct(I*A, B)) == \
        -I*TensorProduct(Dagger(A), Dagger(B))
    assert Dagger(TensorProduct(mat1, mat2)) == \
        TensorProduct(Dagger(mat1), Dagger(mat2))


def test_tensor_product_abstract():

    assert TP(x*A, 2*B) == x*2*TP(A, B)
    assert TP(A, B) != TP(B, A)
    assert TP(A, B).is_commutative is False
    assert isinstance(TP(A, B), TP)
    assert TP(A, B).subs(A, C) == TP(C, B)


def test_tensor_product_expand():
    assert TP(A + B, B + C).expand(tensorproduct=True) == \
        TP(A, B) + TP(A, C) + TP(B, B) + TP(B, C)
    #Tests for fix of issue #24142
    assert TP(A-B, B-A).expand(tensorproduct=True) == \
        TP(A, B) - TP(A, A) - TP(B, B) + TP(B, A)
    assert TP(2*A + B, A + B).expand(tensorproduct=True) == \
        2 * TP(A, A) + 2 * TP(A, B) + TP(B, A) + TP(B, B)
    assert TP(2 * A * B + A, A + B).expand(tensorproduct=True) == \
        2 * TP(A*B, A) + 2 * TP(A*B, B) + TP(A, A) + TP(A, B)


def test_tensor_product_commutator():
    assert TP(Comm(A, B), C).doit().expand(tensorproduct=True) == \
        TP(A*B, C) - TP(B*A, C)
    assert Comm(TP(A, B), TP(B, C)).doit() == \
        TP(A, B)*TP(B, C) - TP(B, C)*TP(A, B)


def test_tensor_product_simp():
    with warns_deprecated_sympy():
        assert tensor_product_simp(TP(A, B)*TP(B, C)) == TP(A*B, B*C)
        # tests for Pow-expressions
        assert TP(A, B)**y == TP(A**y, B**y)
        assert tensor_product_simp(TP(A, B)**y) == TP(A**y, B**y)
        assert tensor_product_simp(x*TP(A, B)**2) == x*TP(A**2,B**2)
        assert tensor_product_simp(x*(TP(A, B)**2)*TP(C,D)) == x*TP(A**2*C,B**2*D)
        assert tensor_product_simp(TP(A,B)-TP(C,D)**y) == TP(A,B)-TP(C**y,D**y)


def test_issue_5923():
    # most of the issue regarding sympification of args has been handled
    # and is tested internally by the use of args_cnc through the quantum
    # module, but the following is a test from the issue that used to raise.
    assert TensorProduct(1, Qubit('1')*Qubit('1').dual) == \
        TensorProduct(1, OuterProduct(Qubit(1), QubitBra(1)))


def test_eval_trace():
    # This test includes tests with dependencies between TensorProducts
    #and density operators. Since, the test is more to test the behavior of
    #TensorProducts it remains here

    # Density with simple tensor products as args
    t = TensorProduct(A, B)
    d = Density([t, 1.0])
    tr = Tr(d)
    assert tr.doit() == 1.0*Tr(A*Dagger(A))*Tr(B*Dagger(B))

    ## partial trace with simple tensor products as args
    t = TensorProduct(A, B, C)
    d = Density([t, 1.0])
    tr = Tr(d, [1])
    assert tr.doit() == 1.0*A*Dagger(A)*Tr(B*Dagger(B))*C*Dagger(C)

    tr = Tr(d, [0, 2])
    assert tr.doit() == 1.0*Tr(A*Dagger(A))*B*Dagger(B)*Tr(C*Dagger(C))

    # Density with multiple Tensorproducts as states
    t2 = TensorProduct(A, B)
    t3 = TensorProduct(C, D)

    d = Density([t2, 0.5], [t3, 0.5])
    t = Tr(d)
    assert t.doit() == (0.5*Tr(A*Dagger(A))*Tr(B*Dagger(B)) +
                        0.5*Tr(C*Dagger(C))*Tr(D*Dagger(D)))

    t = Tr(d, [0])
    assert t.doit() == (0.5*Tr(A*Dagger(A))*B*Dagger(B) +
                        0.5*Tr(C*Dagger(C))*D*Dagger(D))

    #Density with mixed states
    d = Density([t2 + t3, 1.0])
    t = Tr(d)
    assert t.doit() == ( 1.0*Tr(A*Dagger(A))*Tr(B*Dagger(B)) +
                        1.0*Tr(A*Dagger(C))*Tr(B*Dagger(D)) +
                        1.0*Tr(C*Dagger(A))*Tr(D*Dagger(B)) +
                        1.0*Tr(C*Dagger(C))*Tr(D*Dagger(D)))

    t = Tr(d, [1] )
    assert t.doit() == ( 1.0*A*Dagger(A)*Tr(B*Dagger(B)) +
                        1.0*A*Dagger(C)*Tr(B*Dagger(D)) +
                        1.0*C*Dagger(A)*Tr(D*Dagger(B)) +
                        1.0*C*Dagger(C)*Tr(D*Dagger(D)))


def test_pr24993():
    from sympy.matrices.expressions.kronecker import matrix_kronecker_product
    from sympy.physics.quantum.matrixutils    import matrix_tensor_product
    X = Matrix([[0, 1], [1, 0]])
    Xi = ImmutableMatrix(X)
    assert TensorProduct(Xi, Xi) == TensorProduct(X, X)
    assert TensorProduct(Xi, Xi) == matrix_tensor_product(X, X)
    assert TensorProduct(Xi, Xi) == matrix_kronecker_product(X, X)
