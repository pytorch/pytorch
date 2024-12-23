# Test Matrix/DomainMatrix interaction.


from sympy import GF, ZZ, QQ, EXRAW
from sympy.polys.matrices import DomainMatrix, DM

from sympy import (
    Matrix,
    MutableMatrix,
    ImmutableMatrix,
    SparseMatrix,
    MutableDenseMatrix,
    ImmutableDenseMatrix,
    MutableSparseMatrix,
    ImmutableSparseMatrix,
)
from sympy import symbols, S, sqrt

from sympy.testing.pytest import raises


x, y = symbols('x y')


MATRIX_TYPES = (
    Matrix,
    MutableMatrix,
    ImmutableMatrix,
    SparseMatrix,
    MutableDenseMatrix,
    ImmutableDenseMatrix,
    MutableSparseMatrix,
    ImmutableSparseMatrix,
)
IMMUTABLE = (
    ImmutableMatrix,
    ImmutableDenseMatrix,
    ImmutableSparseMatrix,
)


def DMs(items, domain):
    return DM(items, domain).to_sparse()


def test_Matrix_rep_domain():

    for Mat in MATRIX_TYPES:

        M = Mat([[1, 2], [3, 4]])
        assert M._rep == DMs([[1, 2], [3, 4]], ZZ)
        assert (M / 2)._rep == DMs([[(1,2), 1], [(3,2), 2]], QQ)
        if not isinstance(M, IMMUTABLE):
            M[0, 0] = x
            assert M._rep == DMs([[x, 2], [3, 4]], EXRAW)

        M = Mat([[S(1)/2, 2], [3, 4]])
        assert M._rep == DMs([[(1,2), 2], [3, 4]], QQ)
        if not isinstance(M, IMMUTABLE):
            M[0, 0] = x
            assert M._rep == DMs([[x, 2], [3, 4]], EXRAW)

        dM = DMs([[1, 2], [3, 4]], ZZ)
        assert Mat._fromrep(dM)._rep == dM

    # XXX: This is not intended. Perhaps it should be coerced to EXRAW?
    # The private _fromrep method is never called like this but perhaps it
    # should be guarded.
    #
    # It is not clear how to integrate domains other than ZZ, QQ and EXRAW with
    # the rest of Matrix or if the public type for this needs to be something
    # different from Matrix somehow.
    K = QQ.algebraic_field(sqrt(2))
    dM = DM([[1, 2], [3, 4]], K)
    assert Mat._fromrep(dM)._rep.domain == K


def test_Matrix_to_DM():

    M = Matrix([[1, 2], [3, 4]])
    assert M.to_DM() == DMs([[1, 2], [3, 4]], ZZ)
    assert M.to_DM() is not M._rep
    assert M.to_DM(field=True) == DMs([[1, 2], [3, 4]], QQ)
    assert M.to_DM(domain=QQ) == DMs([[1, 2], [3, 4]], QQ)
    assert M.to_DM(domain=QQ[x]) == DMs([[1, 2], [3, 4]], QQ[x])
    assert M.to_DM(domain=GF(3)) == DMs([[1, 2], [0, 1]], GF(3))

    M = Matrix([[1, 2], [3, 4]])
    M[0, 0] = x
    assert M._rep.domain == EXRAW
    M[0, 0] = 1
    assert M.to_DM() == DMs([[1, 2], [3, 4]], ZZ)

    M = Matrix([[S(1)/2, 2], [3, 4]])
    assert M.to_DM() == DMs([[QQ(1,2), 2], [3, 4]], QQ)

    M = Matrix([[x, 2], [3, 4]])
    assert M.to_DM() == DMs([[x, 2], [3, 4]], ZZ[x])
    assert M.to_DM(field=True) == DMs([[x, 2], [3, 4]], ZZ.frac_field(x))

    M = Matrix([[1/x, 2], [3, 4]])
    assert M.to_DM() == DMs([[1/x, 2], [3, 4]], ZZ.frac_field(x))

    M = Matrix([[1, sqrt(2)], [3, 4]])
    K = QQ.algebraic_field(sqrt(2))
    sqrt2 = K.from_sympy(sqrt(2)) # XXX: Maybe K(sqrt(2)) should work
    M_K = DomainMatrix([[K(1), sqrt2], [K(3), K(4)]], (2, 2), K)
    assert M.to_DM() == DMs([[1, sqrt(2)], [3, 4]], EXRAW)
    assert M.to_DM(extension=True) == M_K.to_sparse()

    # Options cannot be used with the domain parameter
    M = Matrix([[1, 2], [3, 4]])
    raises(TypeError, lambda: M.to_DM(domain=QQ, field=True))
