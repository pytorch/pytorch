from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices import DM
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMRankError, DMValueError, DMShapeError, DMDomainError
from sympy.polys.matrices.lll import _ddm_lll, ddm_lll, ddm_lll_transform
from sympy.testing.pytest import raises


def test_lll():
    normal_test_data = [
        (
            DM([[1, 0, 0, 0, -20160],
                [0, 1, 0, 0, 33768],
                [0, 0, 1, 0, 39578],
                [0, 0, 0, 1, 47757]], ZZ),
            DM([[10, -3, -2, 8, -4],
                [3, -9, 8, 1, -11],
                [-3, 13, -9, -3, -9],
                [-12, -7, -11, 9, -1]], ZZ)
        ),
        (
            DM([[20, 52, 3456],
                [14, 31, -1],
                [34, -442, 0]], ZZ),
            DM([[14, 31, -1],
                [188, -101, -11],
                [236, 13, 3443]], ZZ)
        ),
        (
            DM([[34, -1, -86, 12],
                [-54, 34, 55, 678],
                [23, 3498, 234, 6783],
                [87, 49, 665, 11]], ZZ),
            DM([[34, -1, -86, 12],
                [291, 43, 149, 83],
                [-54, 34, 55, 678],
                [-189, 3077, -184, -223]], ZZ)
        )
    ]
    delta = QQ(5, 6)
    for basis_dm, reduced_dm in normal_test_data:
        reduced = _ddm_lll(basis_dm.rep.to_ddm(), delta=delta)[0]
        assert reduced == reduced_dm.rep.to_ddm()

        reduced = ddm_lll(basis_dm.rep.to_ddm(), delta=delta)
        assert reduced == reduced_dm.rep.to_ddm()

        reduced, transform = _ddm_lll(basis_dm.rep.to_ddm(), delta=delta, return_transform=True)
        assert reduced == reduced_dm.rep.to_ddm()
        assert transform.matmul(basis_dm.rep.to_ddm()) == reduced_dm.rep.to_ddm()

        reduced, transform = ddm_lll_transform(basis_dm.rep.to_ddm(), delta=delta)
        assert reduced == reduced_dm.rep.to_ddm()
        assert transform.matmul(basis_dm.rep.to_ddm()) == reduced_dm.rep.to_ddm()

        reduced = basis_dm.rep.lll(delta=delta)
        assert reduced == reduced_dm.rep

        reduced, transform = basis_dm.rep.lll_transform(delta=delta)
        assert reduced == reduced_dm.rep
        assert transform.matmul(basis_dm.rep) == reduced_dm.rep

        reduced = basis_dm.rep.to_sdm().lll(delta=delta)
        assert reduced == reduced_dm.rep.to_sdm()

        reduced, transform = basis_dm.rep.to_sdm().lll_transform(delta=delta)
        assert reduced == reduced_dm.rep.to_sdm()
        assert transform.matmul(basis_dm.rep.to_sdm()) == reduced_dm.rep.to_sdm()

        reduced = basis_dm.lll(delta=delta)
        assert reduced == reduced_dm

        reduced, transform = basis_dm.lll_transform(delta=delta)
        assert reduced == reduced_dm
        assert transform.matmul(basis_dm) == reduced_dm


def test_lll_linear_dependent():
    linear_dependent_test_data = [
        DM([[0, -1, -2, -3],
            [1, 0, -1, -2],
            [2, 1, 0, -1],
            [3, 2, 1, 0]], ZZ),
        DM([[1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 2, 3, 6]], ZZ),
        DM([[3, -5, 1],
            [4, 6, 0],
            [10, -4, 2]], ZZ)
    ]
    for not_basis in linear_dependent_test_data:
        raises(DMRankError, lambda: _ddm_lll(not_basis.rep.to_ddm()))
        raises(DMRankError, lambda: ddm_lll(not_basis.rep.to_ddm()))
        raises(DMRankError, lambda: not_basis.rep.lll())
        raises(DMRankError, lambda: not_basis.rep.to_sdm().lll())
        raises(DMRankError, lambda: not_basis.lll())
        raises(DMRankError, lambda: _ddm_lll(not_basis.rep.to_ddm(), return_transform=True))
        raises(DMRankError, lambda: ddm_lll_transform(not_basis.rep.to_ddm()))
        raises(DMRankError, lambda: not_basis.rep.lll_transform())
        raises(DMRankError, lambda: not_basis.rep.to_sdm().lll_transform())
        raises(DMRankError, lambda: not_basis.lll_transform())


def test_lll_wrong_delta():
    dummy_matrix = DomainMatrix.ones((3, 3), ZZ)
    for wrong_delta in [QQ(-1, 4), QQ(0, 1), QQ(1, 4), QQ(1, 1), QQ(100, 1)]:
        raises(DMValueError, lambda: _ddm_lll(dummy_matrix.rep, delta=wrong_delta))
        raises(DMValueError, lambda: ddm_lll(dummy_matrix.rep, delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.lll(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.to_sdm().lll(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.lll(delta=wrong_delta))
        raises(DMValueError, lambda: _ddm_lll(dummy_matrix.rep, delta=wrong_delta, return_transform=True))
        raises(DMValueError, lambda: ddm_lll_transform(dummy_matrix.rep, delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.lll_transform(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.to_sdm().lll_transform(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.lll_transform(delta=wrong_delta))


def test_lll_wrong_shape():
    wrong_shape_matrix = DomainMatrix.ones((4, 3), ZZ)
    raises(DMShapeError, lambda: _ddm_lll(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: ddm_lll(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.lll())
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.to_sdm().lll())
    raises(DMShapeError, lambda: wrong_shape_matrix.lll())
    raises(DMShapeError, lambda: _ddm_lll(wrong_shape_matrix.rep, return_transform=True))
    raises(DMShapeError, lambda: ddm_lll_transform(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.lll_transform())
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.to_sdm().lll_transform())
    raises(DMShapeError, lambda: wrong_shape_matrix.lll_transform())


def test_lll_wrong_domain():
    wrong_domain_matrix = DomainMatrix.ones((3, 3), QQ)
    raises(DMDomainError, lambda: _ddm_lll(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: ddm_lll(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.lll())
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.to_sdm().lll())
    raises(DMDomainError, lambda: wrong_domain_matrix.lll())
    raises(DMDomainError, lambda: _ddm_lll(wrong_domain_matrix.rep, return_transform=True))
    raises(DMDomainError, lambda: ddm_lll_transform(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.lll_transform())
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.to_sdm().lll_transform())
    raises(DMDomainError, lambda: wrong_domain_matrix.lll_transform())
