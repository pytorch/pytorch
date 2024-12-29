from sympy import ZZ, Matrix
from sympy.polys.matrices import DM, DomainMatrix
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM

import pytest

zeros = lambda shape, K: DomainMatrix.zeros(shape, K).to_dense()
eye = lambda n, K: DomainMatrix.eye(n, K).to_dense()


#
# DomainMatrix.nullspace can have a divided answer or can return an undivided
# uncanonical answer. The uncanonical answer is not unique but we can make it
# unique by making it primitive (remove gcd). The tests here all show the
# primitive form. We test two things:
#
#   A.nullspace().primitive()[1] == answer.
#   A.nullspace(divide_last=True) == _divide_last(answer).
#
# The nullspace as returned by DomainMatrix and related classes is the
# transpose of the nullspace as returned by Matrix. Matrix returns a list of
# of column vectors whereas DomainMatrix returns a matrix whose rows are the
# nullspace vectors.
#


NULLSPACE_EXAMPLES = [

    (
        'zz_1',
         DM([[ 1, 2, 3]], ZZ),
         DM([[-2, 1, 0],
             [-3, 0, 1]], ZZ),
    ),

    (
        'zz_2',
         zeros((0, 0), ZZ),
         zeros((0, 0), ZZ),
    ),

    (
        'zz_3',
        zeros((2, 0), ZZ),
        zeros((0, 0), ZZ),
    ),

    (
        'zz_4',
        zeros((0, 2), ZZ),
        eye(2, ZZ),
    ),

    (
        'zz_5',
        zeros((2, 2), ZZ),
        eye(2, ZZ),
    ),

    (
        'zz_6',
        DM([[1, 2],
            [3, 4]], ZZ),
        zeros((0, 2), ZZ),
    ),

    (
        'zz_7',
        DM([[1, 1],
            [1, 1]], ZZ),
        DM([[-1, 1]], ZZ),
    ),

    (
        'zz_8',
        DM([[1],
            [1]], ZZ),
        zeros((0, 1), ZZ),
    ),

    (
        'zz_9',
        DM([[1, 1]], ZZ),
        DM([[-1, 1]], ZZ),
    ),

    (
        'zz_10',
        DM([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]], ZZ),
        DM([[ 0,  0, 1,  0,  0, 0, 0, 0, 0, 0],
            [-1,  0, 0,  0,  0, 0, 1, 0, 0, 0],
            [ 0, -1, 0,  0,  0, 0, 0, 1, 0, 0],
            [ 0,  0, 0, -1,  0, 0, 0, 0, 1, 0],
            [ 0,  0, 0,  0, -1, 0, 0, 0, 0, 1]], ZZ),
    ),

]


def _to_DM(A, ans):
    """Convert the answer to DomainMatrix."""
    if isinstance(A, DomainMatrix):
        return A.to_dense()
    elif isinstance(A, DDM):
        return DomainMatrix(list(A), A.shape, A.domain).to_dense()
    elif isinstance(A, SDM):
        return DomainMatrix(dict(A), A.shape, A.domain).to_dense()
    else:
        assert False # pragma: no cover


def _divide_last(null):
    """Normalize the nullspace by the rightmost non-zero entry."""
    null = null.to_field()

    if null.is_zero_matrix:
        return null

    rows = []
    for i in range(null.shape[0]):
        for j in reversed(range(null.shape[1])):
            if null[i, j]:
                rows.append(null[i, :] / null[i, j])
                break
        else:
            assert False # pragma: no cover

    return DomainMatrix.vstack(*rows)


def _check_primitive(null, null_ans):
    """Check that the primitive of the answer matches."""
    null = _to_DM(null, null_ans)
    cont, null_prim = null.primitive()
    assert null_prim == null_ans


def _check_divided(null, null_ans):
    """Check the divided answer."""
    null = _to_DM(null, null_ans)
    null_ans_norm = _divide_last(null_ans)
    assert null == null_ans_norm


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_Matrix_nullspace(name, A, A_null):
    A = A.to_Matrix()

    A_null_cols = A.nullspace()

    # We have to patch up the case where the nullspace is empty
    if A_null_cols:
        A_null_found = Matrix.hstack(*A_null_cols)
    else:
        A_null_found = Matrix.zeros(A.cols, 0)

    A_null_found = A_null_found.to_DM().to_field().to_dense()

    # The Matrix result is the transpose of DomainMatrix result.
    A_null_found = A_null_found.transpose()

    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_dense_nullspace(name, A, A_null):
    A = A.to_field().to_dense()
    A_null_found = A.nullspace(divide_last=True)
    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_sparse_nullspace(name, A, A_null):
    A = A.to_field().to_sparse()
    A_null_found = A.nullspace(divide_last=True)
    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_ddm_nullspace(name, A, A_null):
    A = A.to_field().to_ddm()
    A_null_found, _ = A.nullspace()
    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_sdm_nullspace(name, A, A_null):
    A = A.to_field().to_sdm()
    A_null_found, _ = A.nullspace()
    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_dense_nullspace_fracfree(name, A, A_null):
    A = A.to_dense()
    A_null_found = A.nullspace()
    _check_primitive(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_sparse_nullspace_fracfree(name, A, A_null):
    A = A.to_sparse()
    A_null_found = A.nullspace()
    _check_primitive(A_null_found, A_null)
