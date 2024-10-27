from sympy.physics.matrices import msigma, mgamma, minkowski_tensor, pat_matrix, mdft
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.testing.pytest import warns_deprecated_sympy


def test_parallel_axis_theorem():
    # This tests the parallel axis theorem matrix by comparing to test
    # matrices.

    # First case, 1 in all directions.
    mat1 = Matrix(((2, -1, -1), (-1, 2, -1), (-1, -1, 2)))
    assert pat_matrix(1, 1, 1, 1) == mat1
    assert pat_matrix(2, 1, 1, 1) == 2*mat1

    # Second case, 1 in x, 0 in all others
    mat2 = Matrix(((0, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert pat_matrix(1, 1, 0, 0) == mat2
    assert pat_matrix(2, 1, 0, 0) == 2*mat2

    # Third case, 1 in y, 0 in all others
    mat3 = Matrix(((1, 0, 0), (0, 0, 0), (0, 0, 1)))
    assert pat_matrix(1, 0, 1, 0) == mat3
    assert pat_matrix(2, 0, 1, 0) == 2*mat3

    # Fourth case, 1 in z, 0 in all others
    mat4 = Matrix(((1, 0, 0), (0, 1, 0), (0, 0, 0)))
    assert pat_matrix(1, 0, 0, 1) == mat4
    assert pat_matrix(2, 0, 0, 1) == 2*mat4


def test_Pauli():
    #this and the following test are testing both Pauli and Dirac matrices
    #and also that the general Matrix class works correctly in a real world
    #situation
    sigma1 = msigma(1)
    sigma2 = msigma(2)
    sigma3 = msigma(3)

    assert sigma1 == sigma1
    assert sigma1 != sigma2

    # sigma*I -> I*sigma    (see #354)
    assert sigma1*sigma2 == sigma3*I
    assert sigma3*sigma1 == sigma2*I
    assert sigma2*sigma3 == sigma1*I

    assert sigma1*sigma1 == eye(2)
    assert sigma2*sigma2 == eye(2)
    assert sigma3*sigma3 == eye(2)

    assert sigma1*2*sigma1 == 2*eye(2)
    assert sigma1*sigma3*sigma1 == -sigma3


def test_Dirac():
    gamma0 = mgamma(0)
    gamma1 = mgamma(1)
    gamma2 = mgamma(2)
    gamma3 = mgamma(3)
    gamma5 = mgamma(5)

    # gamma*I -> I*gamma    (see #354)
    assert gamma5 == gamma0 * gamma1 * gamma2 * gamma3 * I
    assert gamma1 * gamma2 + gamma2 * gamma1 == zeros(4)
    assert gamma0 * gamma0 == eye(4) * minkowski_tensor[0, 0]
    assert gamma2 * gamma2 != eye(4) * minkowski_tensor[0, 0]
    assert gamma2 * gamma2 == eye(4) * minkowski_tensor[2, 2]

    assert mgamma(5, True) == \
        mgamma(0, True)*mgamma(1, True)*mgamma(2, True)*mgamma(3, True)*I

def test_mdft():
    with warns_deprecated_sympy():
        assert mdft(1) == Matrix([[1]])
    with warns_deprecated_sympy():
        assert mdft(2) == 1/sqrt(2)*Matrix([[1,1],[1,-1]])
    with warns_deprecated_sympy():
        assert mdft(4) == Matrix([[S.Half,  S.Half,  S.Half, S.Half],
                                  [S.Half, -I/2, Rational(-1,2),  I/2],
                                  [S.Half, Rational(-1,2),  S.Half, Rational(-1,2)],
                                  [S.Half,  I/2, Rational(-1,2), -I/2]])
