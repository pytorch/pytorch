from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.domains import ZZ, QQ
from sympy import Matrix
import pytest


FFLU_EXAMPLES = [
    (
        'zz_2x3',
        DM([[1, 2, 3], [4, 5, 6]], ZZ),
        DM([[1, 0], [0, 1]], ZZ),
        DM([[1, 0], [4, -3]], ZZ),
        DM([[1, 0], [0, -3]], ZZ),
        DM([[1, 2, 3], [0, -3, -6]], ZZ),
    ),

    (
        'zz_2x2',
        DM([[4, 3], [6, 3]], ZZ),
        DM([[1, 0], [0, 1]], ZZ),
        DM([[1, 0], [6, -6]], ZZ),
        DM([[4, 0], [0, -3]], ZZ),
        DM([[4, 3], [0, -3]], ZZ),
    ),

    (
        'zz_3x2',
        DM([[1, 2], [3, 4], [5, 6]], ZZ),
        DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ZZ),
        DM([[1, 0, 0], [3, 1, 0], [5, 2, 1]], ZZ),
        DM([[1, 0], [0, -2]], ZZ),
        DM([[1, 2], [0, -2], [0, 0]], ZZ),
    ),

    (
        'zz_3x3',
        DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ZZ),
        DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ZZ),
        DM([[1, 0, 0], [4, 1, 0], [7, 2, 1]], ZZ),
        DM([[1, 0, 0], [0, -3, 0], [0, 0, 0]], ZZ),
        DM([[1, 2, 3], [0, -3, -6], [0, 0, 0]], ZZ),
    ),

    (
        'zz_zero',
        DM([[0, 0, 0], [0, 0, 0], [0, 0, 0]], ZZ),
        DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ZZ),
        DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ZZ),
        DM([[0, 0, 0], [0, 0, 0], [0, 0, 0]], ZZ),
        DM([[0, 0, 0], [0, 0, 0], [0, 0, 0]], ZZ),
    ),

    (
        'zz_empty',
        DM([], ZZ),
        DM([], ZZ),
        DM([], ZZ),
        DM([], ZZ),
        DM([], ZZ),
    ),

    (
        'zz_empty_0x2',
        DomainMatrix([], (0, 2), ZZ),
        DomainMatrix([], (0, 0), ZZ),
        DomainMatrix([], (0, 0), ZZ),
        DomainMatrix([], (0, 0), ZZ),
        DomainMatrix([], (0, 2), ZZ)
    ),

    (

        'zz_empty_2x0',
        DomainMatrix([[], []], (2, 0), ZZ),
        DomainMatrix.eye((2, 2), ZZ),
        DomainMatrix.eye((2, 2), ZZ),
        DomainMatrix.eye((2, 2), ZZ),
        DomainMatrix([[], []], (2, 0), ZZ)

    ),

    (
        'zz_negative',
        DM([[-1, -2], [-3, -4]], ZZ),
        DM([[1, 0], [0, 1]], ZZ),
        DM([[-1, 0], [-3, -2]], ZZ),
        DM([[-1, 0], [0, 2]], ZZ),
        DM([[-1, -2], [0, -2]], ZZ),
    ),

    (
        'zz_mixed_signs',
        DM([[1, -2], [-3, 4]], ZZ),
        DM([[1, 0], [0, 1]], ZZ),
        DM([[1, 0], [-3, 1]], ZZ),
        DM([[1, 0], [0, -2]], ZZ),
        DM([[1, -2], [0, -2]], ZZ),
    ),

    (
        'zz_upper_triangular',
        DM([[1, 2, 3], [0, 4, 5], [0, 0, 6]], ZZ),
        DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ZZ),
        DM([[1, 0, 0], [0, 4, 0], [0, 0, 24]], ZZ),
        DM([[1, 0, 0], [0, 4, 0], [0, 0, 96]], ZZ),
        DM([[1, 2, 3], [0, 4, 5], [0, 0, 24]], ZZ),
    ),

    (
        'zz_lower_triangular',
        DM([[1, 0, 0], [2, 3, 0], [4, 5, 6]], ZZ),
        DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ZZ),
        DM([[1, 0, 0], [2, 3, 0], [4, 5, 18]], ZZ),
        DM([[1, 0, 0], [0, 3, 0], [0, 0, 54]], ZZ),
        DM([[1, 0, 0], [0, 3, 0], [0, 0, 18]], ZZ),
    ),

    (
        'zz_diagonal',
        DM([[2, 0, 0], [0, 3, 0], [0, 0, 4]], ZZ),
        DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ZZ),
        DM([[2, 0, 0], [0, 6, 0], [0, 0, 24]], ZZ),
        DM([[2, 0, 0], [0, 12, 0], [0, 0, 144]], ZZ),
        DM([[2, 0, 0], [0, 6, 0], [0, 0, 24]], ZZ)

    ),

    (
        'rank_deficient_3x3',
        DM([[1, 2, 3], [2, 4, 6], [3, 6, 9]], ZZ),
        DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ZZ),
        DM([[1, 0, 0], [2, 1, 0], [3, 0, 1]], ZZ),
        DM([[1, 0, 0], [0, 0, 0], [0, 0, 0]], ZZ),
        DM([[1, 2, 3], [0, 0, 0], [0, 0, 0]], ZZ),
    ),

    (
        'zz_1x1',
        DM([[5]], ZZ),
        DM([[1]], ZZ),
        DM([[5]], ZZ),
        DM([[5]], ZZ),
        DM([[5]], ZZ),
    ),

    (
        'zz_nx1_2rows',
        DM([[81], [54]], ZZ),
        DM([[1, 0], [0, 1]], ZZ),
        DM([[81, 0], [54, 81]], ZZ),
        DM([[81, 0], [0, 81]], ZZ),
        DM([[81], [0]], ZZ),
    ),

    (
        'zz_nx2_3rows',
        DM([[2, 7], [7, 45], [25, 84]], ZZ),
        DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ZZ),
        DM([[2, 0, 0], [7, 82, 0], [25, 41, 41]], ZZ),
        DM([[2, 0, 0], [0, 82, 0], [0, 0, 41]], ZZ),
        DM([[2, 7], [0, 82], [0, 0]], ZZ),
    ),

    (

        'zz_1x2',
        DM([[0, 28]], ZZ),
        DM([[1]], ZZ),
        DM([[28]], ZZ),
        DM([[28]], ZZ),
        DM([[0, 28]], ZZ)
    ),

    (
        'zz_nx3_4rows',
        DM([[84, 30, 9], [20, 59, 13], [53, 46, 81], [63, 48, 29]], ZZ),
        DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], ZZ),
        DM([[84, 0, 0, 0], [20, 365904, 0, 0], [53, 303411, 303411, 0], [63, 303411, 303411, 303411]], ZZ),
        DM([[84, 0, 0, 0], [0, 365904, 0, 0], [0, 0, 1321658316, 0], [0, 0, 0, 303411]], ZZ),
        DM([[84, 30, 9], [0, 365904, 13], [0, 0, 1321658316], [0, 0, 0]], ZZ),
    ),

    (
        'fflu_row_swap',
        DM([[0, 1, 2], [3, 4, 5], [6, 7, 8]], ZZ),
        DM([[0, 1, 0], [1, 0, 0], [0, 0, 1]], ZZ),
        DM([[3, 0, 0], [0, 3, 0], [6, -3, 1]], ZZ),
        DM([[3, 0, 0], [0, 9, 0], [0, 0, 3]], ZZ),
        DM([[3, 4, 5], [0, 3, 6], [0, 0, 0]], ZZ)
    ),
]


def _check_fflu(A, P, L, D, U):
    P_field = P.to_field().to_dense()
    L_field = L.to_field().to_dense()
    D_field = D.to_field().to_dense()
    U_field = U.to_field().to_dense()
    m, n = A.shape
    assert P_field.shape == (m, m)
    assert L_field.shape == (m, m)
    assert D_field.shape == (m, m)
    assert U_field.shape == (m, n)
    assert L_field.is_lower
    assert D_field.is_diagonal
    di, d = D.inv_den()
    assert P.matmul(A).rmul(d) == L.matmul(di).matmul(U)
    assert U_field.is_upper


def _to_DM(A, ans):
    if isinstance(A, DomainMatrix):
        return A
    elif isinstance(A, Matrix):
        return A.to_DM(ans.domain)
    return DomainMatrix(A.to_list(), A.shape, A.domain)


def _check_fflu_result(result, A, P_ans, L_ans, D_ans, U_ans):
    P, L, D, U = result
    P = _to_DM(P, P_ans)
    L = _to_DM(L, L_ans)
    D = _to_DM(D, D_ans)
    U = _to_DM(U, U_ans)
    A = _to_DM(A, P_ans)
    m, n = A.shape
    assert P.shape == (m, m)
    assert L.shape == (m, m)
    assert D.shape == (m, m)
    assert U.shape == (m, n)
    assert L.is_lower
    assert D.is_diagonal
    di, d = D.inv_den()
    assert P.matmul(A).rmul(d) == L.matmul(di).matmul(U)
    assert U.is_upper


@pytest.mark.parametrize('name, A, P_ans, L_ans, D_ans, U_ans', FFLU_EXAMPLES)
def test_dm_dense_fflu(name, A, P_ans, L_ans, D_ans, U_ans):
    A = A.to_dense()
    _check_fflu_result(A.fflu(), A, P_ans, L_ans, D_ans, U_ans)


@pytest.mark.parametrize('name, A, P_ans, L_ans, D_ans, U_ans', FFLU_EXAMPLES)
def test_dm_sparse_fflu(name, A, P_ans, L_ans, D_ans, U_ans):
    A = A.to_sparse()
    _check_fflu_result(A.fflu(), A, P_ans, L_ans, D_ans, U_ans)


@pytest.mark.parametrize('name, A, P_ans, L_ans, D_ans, U_ans', FFLU_EXAMPLES)
def test_ddm_fflu(name, A, P_ans, L_ans, D_ans, U_ans):
    A = A.to_ddm()
    _check_fflu_result(A.fflu(), A, P_ans, L_ans, D_ans, U_ans)


@pytest.mark.parametrize('name, A, P_ans, L_ans, D_ans, U_ans', FFLU_EXAMPLES)
def test_sdm_fflu(name, A, P_ans, L_ans, D_ans, U_ans):
    A = A.to_sdm()
    _check_fflu_result(A.fflu(), A, P_ans, L_ans, D_ans, U_ans)


@pytest.mark.parametrize('name, A, P_ans, L_ans, D_ans, U_ans', FFLU_EXAMPLES)
def test_dfm_fflu(name, A, P_ans, L_ans, D_ans, U_ans):
    pytest.importorskip('flint')
    if A.domain not in (ZZ, QQ) and not A.domain.is_FF:
        pytest.skip("Domain not supported by DFM")
    A = A.to_dfm()
    _check_fflu_result(A.fflu(), A, P_ans, L_ans, D_ans, U_ans)


def test_fflu_empty_matrix():
    A = DomainMatrix([], (0, 0), ZZ)
    P, L, D, U = A.fflu()
    assert P.shape == (0, 0)
    assert L.shape == (0, 0)
    assert D.shape == (0, 0)
    assert U.shape == (0, 0)


def test_fflu_properties():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    P, L, D, U = A.fflu()
    assert P.shape == (2, 2)
    assert L.shape == (2, 2)
    assert D.shape == (2, 2)
    assert U.shape == (2, 2)
    assert L.is_lower
    assert U.is_upper
    assert D.is_diagonal
    di, d = D.inv_den()
    assert P.matmul(A).rmul(d) == L.matmul(di).matmul(U)


def test_fflu_rank_deficient():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(2), ZZ(4)]], (2, 2), ZZ)
    P, L, D, U = A.fflu()
    assert P.shape == (2, 2)
    assert L.shape == (2, 2)
    assert D.shape == (2, 2)
    assert U.shape == (2, 2)
    assert U.getitem_sympy(1, 1) == 0
