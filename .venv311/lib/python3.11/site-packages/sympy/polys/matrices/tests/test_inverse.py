from sympy import ZZ, Matrix
from sympy.polys.matrices import DM, DomainMatrix
from sympy.polys.matrices.dense import ddm_iinv
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.matrices.exceptions import NonInvertibleMatrixError

import pytest
from sympy.testing.pytest import raises
from sympy.core.numbers import all_close

from sympy.abc import x


# Examples are given as adjugate matrix and determinant adj_det should match
# these exactly but inv_den only matches after cancel_denom.


INVERSE_EXAMPLES = [

    (
        'zz_1',
        DomainMatrix([], (0, 0), ZZ),
        DomainMatrix([], (0, 0), ZZ),
        ZZ(1),
    ),

    (
        'zz_2',
        DM([[2]], ZZ),
        DM([[1]], ZZ),
        ZZ(2),
    ),

    (
        'zz_3',
        DM([[2, 0],
            [0, 2]], ZZ),
        DM([[2, 0],
            [0, 2]], ZZ),
        ZZ(4),
    ),

    (
        'zz_4',
        DM([[1, 2],
            [3, 4]], ZZ),
        DM([[ 4, -2],
            [-3,  1]], ZZ),
        ZZ(-2),
    ),

    (
        'zz_5',
        DM([[2, 2, 0],
            [0, 2, 2],
            [0, 0, 2]], ZZ),
        DM([[4, -4, 4],
            [0, 4, -4],
            [0, 0,  4]], ZZ),
        ZZ(8),
    ),

    (
        'zz_6',
        DM([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], ZZ),
        DM([[-3,   6, -3],
            [ 6, -12,  6],
            [-3,   6, -3]], ZZ),
        ZZ(0),
    ),
]


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_Matrix_inv(name, A, A_inv, den):

    def _check(**kwargs):
        if den != 0:
            assert A.inv(**kwargs) == A_inv
        else:
            raises(NonInvertibleMatrixError, lambda: A.inv(**kwargs))

    K = A.domain
    A = A.to_Matrix()
    A_inv = A_inv.to_Matrix() / K.to_sympy(den)
    _check()
    for method in ['GE', 'LU', 'ADJ', 'CH', 'LDL', 'QR']:
        _check(method=method)


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dm_inv_den(name, A, A_inv, den):
    if den != 0:
        A_inv_f, den_f = A.inv_den()
        assert A_inv_f.cancel_denom(den_f) == A_inv.cancel_denom(den)
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv_den())


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dm_inv(name, A, A_inv, den):
    A = A.to_field()
    if den != 0:
        A_inv = A_inv.to_field() / den
        assert A.inv() == A_inv
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv())


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_ddm_inv(name, A, A_inv, den):
    A = A.to_field().to_ddm()
    if den != 0:
        A_inv = (A_inv.to_field() / den).to_ddm()
        assert A.inv() == A_inv
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv())


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_sdm_inv(name, A, A_inv, den):
    A = A.to_field().to_sdm()
    if den != 0:
        A_inv = (A_inv.to_field() / den).to_sdm()
        assert A.inv() == A_inv
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv())


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dense_ddm_iinv(name, A, A_inv, den):
    A = A.to_field().to_ddm().copy()
    K = A.domain
    A_result = A.copy()
    if den != 0:
        A_inv = (A_inv.to_field() / den).to_ddm()
        ddm_iinv(A_result, A, K)
        assert A_result == A_inv
    else:
        raises(DMNonInvertibleMatrixError, lambda: ddm_iinv(A_result, A, K))


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_Matrix_adjugate(name, A, A_inv, den):
    A = A.to_Matrix()
    A_inv = A_inv.to_Matrix()
    assert A.adjugate() == A_inv
    for method in ["bareiss", "berkowitz", "bird", "laplace", "lu"]:
        assert A.adjugate(method=method) == A_inv


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dm_adj_det(name, A, A_inv, den):
    assert A.adj_det() == (A_inv, den)


def test_inverse_inexact():

    M = Matrix([[x-0.3, -0.06, -0.22],
                [-0.46, x-0.48, -0.41],
                [-0.14, -0.39, x-0.64]])

    Mn = Matrix([[1.0*x**2 - 1.12*x + 0.1473, 0.06*x + 0.0474, 0.22*x - 0.081],
                 [0.46*x - 0.237, 1.0*x**2 - 0.94*x + 0.1612, 0.41*x - 0.0218],
                 [0.14*x + 0.1122, 0.39*x - 0.1086, 1.0*x**2 - 0.78*x + 0.1164]])

    d = 1.0*x**3 - 1.42*x**2 + 0.4249*x - 0.0546540000000002

    Mi = Mn / d

    M_dm = M.to_DM()
    M_dmd = M_dm.to_dense()
    M_dm_num, M_dm_den = M_dm.inv_den()
    M_dmd_num, M_dmd_den = M_dmd.inv_den()

    # XXX: We don't check M_dm().to_field().inv() which currently uses division
    # and produces a more complicate result from gcd cancellation failing.
    # DomainMatrix.inv() over RR(x) should be changed to clear denominators and
    # use DomainMatrix.inv_den().

    Minvs = [
        M.inv(),
        (M_dm_num.to_field() / M_dm_den).to_Matrix(),
        (M_dmd_num.to_field() / M_dmd_den).to_Matrix(),
        M_dm_num.to_Matrix() / M_dm_den.as_expr(),
        M_dmd_num.to_Matrix() / M_dmd_den.as_expr(),
    ]

    for Minv in Minvs:
        for Mi1, Mi2 in zip(Minv.flat(), Mi.flat()):
            assert all_close(Mi2, Mi1)
