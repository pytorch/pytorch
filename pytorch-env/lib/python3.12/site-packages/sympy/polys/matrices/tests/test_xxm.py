#
# Test basic features of DDM, SDM and DFM.
#
# These three types are supposed to be interchangeable, so we should use the
# same tests for all of them for the most part.
#
# The tests here cover the basic part of the inerface that the three types
# should expose and that DomainMatrix should mostly rely on.
#
# More in-depth tests of the heavier algorithms like rref etc should go in
# their own test files.
#
# Any new methods added to the DDM, SDM or DFM classes should be tested here
# and added to all classes.
#

from sympy.external.gmpy import GROUND_TYPES

from sympy import ZZ, QQ, GF, ZZ_I, symbols

from sympy.polys.matrices.exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError,
    DMShapeError,
)

from sympy.polys.matrices.domainmatrix import DM, DomainMatrix, DDM, SDM, DFM

from sympy.testing.pytest import raises, skip
import pytest


def test_XXM_constructors():
    """Test the DDM, etc constructors."""

    lol = [
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6)],
    ]
    dod = {
        0: {0: ZZ(1), 1: ZZ(2)},
        1: {0: ZZ(3), 1: ZZ(4)},
        2: {0: ZZ(5), 1: ZZ(6)},
    }

    lol_0x0 = []
    lol_0x2 = []
    lol_2x0 = [[], []]
    dod_0x0 = {}
    dod_0x2 = {}
    dod_2x0 = {}

    lol_bad = [
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6), ZZ(7)],
    ]
    dod_bad = {
        0: {0: ZZ(1), 1: ZZ(2)},
        1: {0: ZZ(3), 1: ZZ(4)},
        2: {0: ZZ(5), 1: ZZ(6), 2: ZZ(7)},
    }

    XDM_dense = [DDM]
    XDM_sparse = [SDM]

    if GROUND_TYPES == 'flint':
        XDM_dense.append(DFM)

    for XDM in XDM_dense:

        A = XDM(lol, (3, 2), ZZ)
        assert A.rows == 3
        assert A.cols == 2
        assert A.domain == ZZ
        assert A.shape == (3, 2)
        if XDM is not DFM:
            assert ZZ.of_type(A[0][0]) is True
        else:
            assert ZZ.of_type(A.rep[0, 0]) is True

        Adm = DomainMatrix(lol, (3, 2), ZZ)
        if XDM is DFM:
            assert Adm.rep == A
            assert Adm.rep.to_ddm() != A
        elif GROUND_TYPES == 'flint':
            assert Adm.rep.to_ddm() == A
            assert Adm.rep != A
        else:
            assert Adm.rep == A
            assert Adm.rep.to_ddm() == A

        assert XDM(lol_0x0, (0, 0), ZZ).shape == (0, 0)
        assert XDM(lol_0x2, (0, 2), ZZ).shape == (0, 2)
        assert XDM(lol_2x0, (2, 0), ZZ).shape == (2, 0)
        raises(DMBadInputError, lambda: XDM(lol, (2, 3), ZZ))
        raises(DMBadInputError, lambda: XDM(lol_bad, (3, 2), ZZ))
        raises(DMBadInputError, lambda: XDM(dod, (3, 2), ZZ))

    for XDM in XDM_sparse:

        A = XDM(dod, (3, 2), ZZ)
        assert A.rows == 3
        assert A.cols == 2
        assert A.domain == ZZ
        assert A.shape == (3, 2)
        assert ZZ.of_type(A[0][0]) is True

        assert DomainMatrix(dod, (3, 2), ZZ).rep == A

        assert XDM(dod_0x0, (0, 0), ZZ).shape == (0, 0)
        assert XDM(dod_0x2, (0, 2), ZZ).shape == (0, 2)
        assert XDM(dod_2x0, (2, 0), ZZ).shape == (2, 0)
        raises(DMBadInputError, lambda: XDM(dod, (2, 3), ZZ))
        raises(DMBadInputError, lambda: XDM(lol, (3, 2), ZZ))
        raises(DMBadInputError, lambda: XDM(dod_bad, (3, 2), ZZ))

    raises(DMBadInputError, lambda: DomainMatrix(lol, (2, 3), ZZ))
    raises(DMBadInputError, lambda: DomainMatrix(lol_bad, (3, 2), ZZ))
    raises(DMBadInputError, lambda: DomainMatrix(dod_bad, (3, 2), ZZ))


def test_XXM_eq():
    """Test equality for DDM, SDM, DFM and DomainMatrix."""

    lol1 = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    dod1 = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}

    lol2 = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(5)]]
    dod2 = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(5)}}

    A1_ddm = DDM(lol1, (2, 2), ZZ)
    A1_sdm = SDM(dod1, (2, 2), ZZ)
    A1_dm_d = DomainMatrix(lol1, (2, 2), ZZ)
    A1_dm_s = DomainMatrix(dod1, (2, 2), ZZ)

    A2_ddm = DDM(lol2, (2, 2), ZZ)
    A2_sdm = SDM(dod2, (2, 2), ZZ)
    A2_dm_d = DomainMatrix(lol2, (2, 2), ZZ)
    A2_dm_s = DomainMatrix(dod2, (2, 2), ZZ)

    A1_all = [A1_ddm, A1_sdm, A1_dm_d, A1_dm_s]
    A2_all = [A2_ddm, A2_sdm, A2_dm_d, A2_dm_s]

    if GROUND_TYPES == 'flint':

        A1_dfm = DFM([[1, 2], [3, 4]], (2, 2), ZZ)
        A2_dfm = DFM([[1, 2], [3, 5]], (2, 2), ZZ)

        A1_all.append(A1_dfm)
        A2_all.append(A2_dfm)

    for n, An in enumerate(A1_all):
        for m, Am in enumerate(A1_all):
            if n == m:
                assert (An == Am) is True
                assert (An != Am) is False
            else:
                assert (An == Am) is False
                assert (An != Am) is True

    for n, An in enumerate(A2_all):
        for m, Am in enumerate(A2_all):
            if n == m:
                assert (An == Am) is True
                assert (An != Am) is False
            else:
                assert (An == Am) is False
                assert (An != Am) is True

    for n, A1 in enumerate(A1_all):
        for m, A2 in enumerate(A2_all):
            assert (A1 == A2) is False
            assert (A1 != A2) is True


def test_to_XXM():
    """Test to_ddm etc. for DDM, SDM, DFM and DomainMatrix."""

    lol = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    dod = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}

    A_ddm = DDM(lol, (2, 2), ZZ)
    A_sdm = SDM(dod, (2, 2), ZZ)
    A_dm_d = DomainMatrix(lol, (2, 2), ZZ)
    A_dm_s = DomainMatrix(dod, (2, 2), ZZ)

    A_all = [A_ddm, A_sdm, A_dm_d, A_dm_s]

    if GROUND_TYPES == 'flint':
        A_dfm = DFM(lol, (2, 2), ZZ)
        A_all.append(A_dfm)

    for A in A_all:
        assert A.to_ddm() == A_ddm
        assert A.to_sdm() == A_sdm
        if GROUND_TYPES != 'flint':
            raises(NotImplementedError, lambda: A.to_dfm())
            assert A.to_dfm_or_ddm() == A_ddm

        # Add e.g. DDM.to_DM()?
        # assert A.to_DM() == A_dm

    if GROUND_TYPES == 'flint':
        for A in A_all:
            assert A.to_dfm() == A_dfm
            for K in [ZZ, QQ, GF(5), ZZ_I]:
                if isinstance(A, DFM) and not DFM._supports_domain(K):
                    raises(NotImplementedError, lambda: A.convert_to(K))
                else:
                    A_K = A.convert_to(K)
                    if DFM._supports_domain(K):
                        A_dfm_K = A_dfm.convert_to(K)
                        assert A_K.to_dfm() == A_dfm_K
                        assert A_K.to_dfm_or_ddm() == A_dfm_K
                    else:
                        raises(NotImplementedError, lambda: A_K.to_dfm())
                        assert A_K.to_dfm_or_ddm() == A_ddm.convert_to(K)


def test_DFM_domains():
    """Test which domains are supported by DFM."""

    x, y = symbols('x, y')

    if GROUND_TYPES in ('python', 'gmpy'):

        supported = []
        flint_funcs = {}
        not_supported = [ZZ, QQ, GF(5), QQ[x], QQ[x,y]]

    elif GROUND_TYPES == 'flint':

        import flint
        supported = [ZZ, QQ]
        flint_funcs = {
            ZZ: flint.fmpz_mat,
            QQ: flint.fmpq_mat,
        }
        not_supported = [
            # This could be supported but not yet implemented in SymPy:
            GF(5),
            # Other domains could be supported but not implemented as matrices
            # in python-flint:
            QQ[x],
            QQ[x,y],
            QQ.frac_field(x,y),
            # Others would potentially never be supported by python-flint:
            ZZ_I,
        ]

    else:
        assert False, "Unknown GROUND_TYPES: %s" % GROUND_TYPES

    for domain in supported:
        assert DFM._supports_domain(domain) is True
        assert DFM._get_flint_func(domain) == flint_funcs[domain]
    for domain in not_supported:
        assert DFM._supports_domain(domain) is False
        raises(NotImplementedError, lambda: DFM._get_flint_func(domain))


def _DM(lol, typ, K):
    """Make a DM of type typ over K from lol."""
    A = DM(lol, K)

    if typ == 'DDM':
        return A.to_ddm()
    elif typ == 'SDM':
        return A.to_sdm()
    elif typ == 'DFM':
        if GROUND_TYPES != 'flint':
            skip("DFM not supported in this ground type")
        return A.to_dfm()
    else:
        assert False, "Unknown type %s" % typ


def _DMZ(lol, typ):
    """Make a DM of type typ over ZZ from lol."""
    return _DM(lol, typ, ZZ)


def _DMQ(lol, typ):
    """Make a DM of type typ over QQ from lol."""
    return _DM(lol, typ, QQ)


def DM_ddm(lol, K):
    """Make a DDM over K from lol."""
    return _DM(lol, 'DDM', K)


def DM_sdm(lol, K):
    """Make a SDM over K from lol."""
    return _DM(lol, 'SDM', K)


def DM_dfm(lol, K):
    """Make a DFM over K from lol."""
    return _DM(lol, 'DFM', K)


def DMZ_ddm(lol):
    """Make a DDM from lol."""
    return _DMZ(lol, 'DDM')


def DMZ_sdm(lol):
    """Make a SDM from lol."""
    return _DMZ(lol, 'SDM')


def DMZ_dfm(lol):
    """Make a DFM from lol."""
    return _DMZ(lol, 'DFM')


def DMQ_ddm(lol):
    """Make a DDM from lol."""
    return _DMQ(lol, 'DDM')


def DMQ_sdm(lol):
    """Make a SDM from lol."""
    return _DMQ(lol, 'SDM')


def DMQ_dfm(lol):
    """Make a DFM from lol."""
    return _DMQ(lol, 'DFM')


DM_all = [DM_ddm, DM_sdm, DM_dfm]
DMZ_all = [DMZ_ddm, DMZ_sdm, DMZ_dfm]
DMQ_all = [DMQ_ddm, DMQ_sdm, DMQ_dfm]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XDM_getitem(DM):
    """Test getitem for DDM, etc."""

    lol = [[0, 1], [2, 0]]
    A = DM(lol)
    m, n = A.shape

    indices = [-3, -2, -1, 0, 1, 2]

    for i in indices:
        for j in indices:
            if -2 <= i < m and -2 <= j < n:
                assert A.getitem(i, j) == ZZ(lol[i][j])
            else:
                raises(IndexError, lambda: A.getitem(i, j))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XDM_setitem(DM):
    """Test setitem for DDM, etc."""

    A = DM([[0, 1, 2], [3, 4, 5]])

    A.setitem(0, 0, ZZ(6))
    assert A == DM([[6, 1, 2], [3, 4, 5]])

    A.setitem(0, 1, ZZ(7))
    assert A == DM([[6, 7, 2], [3, 4, 5]])

    A.setitem(0, 2, ZZ(8))
    assert A == DM([[6, 7, 8], [3, 4, 5]])

    A.setitem(0, -1, ZZ(9))
    assert A == DM([[6, 7, 9], [3, 4, 5]])

    A.setitem(0, -2, ZZ(10))
    assert A == DM([[6, 10, 9], [3, 4, 5]])

    A.setitem(0, -3, ZZ(11))
    assert A == DM([[11, 10, 9], [3, 4, 5]])

    raises(IndexError, lambda: A.setitem(0, 3, ZZ(12)))
    raises(IndexError, lambda: A.setitem(0, -4, ZZ(13)))

    A.setitem(1, 0, ZZ(14))
    assert A == DM([[11, 10, 9], [14, 4, 5]])

    A.setitem(1, 1, ZZ(15))
    assert A == DM([[11, 10, 9], [14, 15, 5]])

    A.setitem(-1, 1, ZZ(16))
    assert A == DM([[11, 10, 9], [14, 16, 5]])

    A.setitem(-2, 1, ZZ(17))
    assert A == DM([[11, 17, 9], [14, 16, 5]])

    raises(IndexError, lambda: A.setitem(2, 0, ZZ(18)))
    raises(IndexError, lambda: A.setitem(-3, 0, ZZ(19)))

    A.setitem(1, 2, ZZ(0))
    assert A == DM([[11, 17, 9], [14, 16, 0]])

    A.setitem(1, -2, ZZ(0))
    assert A == DM([[11, 17, 9], [14, 0, 0]])

    A.setitem(1, -3, ZZ(0))
    assert A == DM([[11, 17, 9], [0, 0, 0]])

    A.setitem(0, 0, ZZ(0))
    assert A == DM([[0, 17, 9], [0, 0, 0]])

    A.setitem(0, -1, ZZ(0))
    assert A == DM([[0, 17, 0], [0, 0, 0]])

    A.setitem(0, 0, ZZ(0))
    assert A == DM([[0, 17, 0], [0, 0, 0]])

    A.setitem(0, -2, ZZ(0))
    assert A == DM([[0, 0, 0], [0, 0, 0]])

    A.setitem(0, -3, ZZ(1))
    assert A == DM([[1, 0, 0], [0, 0, 0]])


class _Sliced:
    def __getitem__(self, item):
        return item


_slice = _Sliced()


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_extract_slice(DM):
    A = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A.extract_slice(*_slice[:,:]) == A
    assert A.extract_slice(*_slice[1:,:]) == DM([[4, 5, 6], [7, 8, 9]])
    assert A.extract_slice(*_slice[1:,1:]) == DM([[5, 6], [8, 9]])
    assert A.extract_slice(*_slice[1:,:-1]) == DM([[4, 5], [7, 8]])
    assert A.extract_slice(*_slice[1:,:-1:2]) == DM([[4], [7]])
    assert A.extract_slice(*_slice[:,::2]) == DM([[1, 3], [4, 6], [7, 9]])
    assert A.extract_slice(*_slice[::2,:]) == DM([[1, 2, 3], [7, 8, 9]])
    assert A.extract_slice(*_slice[::2,::2]) == DM([[1, 3], [7, 9]])
    assert A.extract_slice(*_slice[::2,::-2]) == DM([[3, 1], [9, 7]])
    assert A.extract_slice(*_slice[::-2,::2]) == DM([[7, 9], [1, 3]])
    assert A.extract_slice(*_slice[::-2,::-2]) == DM([[9, 7], [3, 1]])
    assert A.extract_slice(*_slice[:,::-1]) == DM([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
    assert A.extract_slice(*_slice[::-1,:]) == DM([[7, 8, 9], [4, 5, 6], [1, 2, 3]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_extract(DM):

    A = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    assert A.extract([0, 1, 2], [0, 1, 2]) == A
    assert A.extract([1, 2], [1, 2]) == DM([[5, 6], [8, 9]])
    assert A.extract([1, 2], [0, 1]) == DM([[4, 5], [7, 8]])
    assert A.extract([1, 2], [0, 2]) == DM([[4, 6], [7, 9]])
    assert A.extract([1, 2], [0]) == DM([[4], [7]])
    assert A.extract([1, 2], []) == DM([[1]]).zeros((2, 0), ZZ)
    assert A.extract([], [0, 1, 2]) == DM([[1]]).zeros((0, 3), ZZ)

    raises(IndexError, lambda: A.extract([1, 2], [0, 3]))
    raises(IndexError, lambda: A.extract([1, 2], [0, -4]))
    raises(IndexError, lambda: A.extract([3, 1], [0, 1]))
    raises(IndexError, lambda: A.extract([-4, 2], [3, 1]))

    B = DM([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert B.extract([1, 2], [1, 2]) == DM([[0, 0], [0, 0]])


def test_XXM_str():

    A = DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)

    assert str(A) == \
        'DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
    assert str(A.to_ddm()) == \
        '[[1, 2, 3], [4, 5, 6], [7, 8, 9]]'
    assert str(A.to_sdm()) == \
        '{0: {0: 1, 1: 2, 2: 3}, 1: {0: 4, 1: 5, 2: 6}, 2: {0: 7, 1: 8, 2: 9}}'

    assert repr(A) == \
        'DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
    assert repr(A.to_ddm()) == \
        'DDM([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
    assert repr(A.to_sdm()) == \
        'SDM({0: {0: 1, 1: 2, 2: 3}, 1: {0: 4, 1: 5, 2: 6}, 2: {0: 7, 1: 8, 2: 9}}, (3, 3), ZZ)'

    B = DomainMatrix({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3)}}, (2, 2), ZZ)

    assert str(B) == \
        'DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3}}, (2, 2), ZZ)'
    assert str(B.to_ddm()) == \
        '[[1, 2], [3, 0]]'
    assert str(B.to_sdm()) == \
        '{0: {0: 1, 1: 2}, 1: {0: 3}}'

    assert repr(B) == \
        'DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3}}, (2, 2), ZZ)'

    if GROUND_TYPES != 'gmpy':
        assert repr(B.to_ddm()) == \
            'DDM([[1, 2], [3, 0]], (2, 2), ZZ)'
        assert repr(B.to_sdm()) == \
            'SDM({0: {0: 1, 1: 2}, 1: {0: 3}}, (2, 2), ZZ)'
    else:
        assert repr(B.to_ddm()) == \
            'DDM([[mpz(1), mpz(2)], [mpz(3), mpz(0)]], (2, 2), ZZ)'
        assert repr(B.to_sdm()) == \
            'SDM({0: {0: mpz(1), 1: mpz(2)}, 1: {0: mpz(3)}}, (2, 2), ZZ)'

    if GROUND_TYPES == 'flint':

        assert str(A.to_dfm()) == \
            '[[1, 2, 3], [4, 5, 6], [7, 8, 9]]'
        assert str(B.to_dfm()) == \
            '[[1, 2], [3, 0]]'

        assert repr(A.to_dfm()) == \
            'DFM([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
        assert repr(B.to_dfm()) == \
            'DFM([[1, 2], [3, 0]], (2, 2), ZZ)'


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_list(DM):
    T = type(DM([[0]]))

    lol = [[1, 2, 4], [4, 5, 6]]
    lol_ZZ = [[ZZ(1), ZZ(2), ZZ(4)], [ZZ(4), ZZ(5), ZZ(6)]]
    lol_ZZ_bad = [[ZZ(1), ZZ(2), ZZ(4)], [ZZ(4), ZZ(5), ZZ(6), ZZ(7)]]

    assert T.from_list(lol_ZZ, (2, 3), ZZ) == DM(lol)
    raises(DMBadInputError, lambda: T.from_list(lol_ZZ_bad, (3, 2), ZZ))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_list(DM):
    lol = [[1, 2, 4], [4, 5, 6]]
    assert DM(lol).to_list() == [[ZZ(1), ZZ(2), ZZ(4)], [ZZ(4), ZZ(5), ZZ(6)]]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_list_flat(DM):
    lol = [[1, 2, 4], [4, 5, 6]]
    assert DM(lol).to_list_flat() == [ZZ(1), ZZ(2), ZZ(4), ZZ(4), ZZ(5), ZZ(6)]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_list_flat(DM):
    T = type(DM([[0]]))
    flat = [ZZ(1), ZZ(2), ZZ(4), ZZ(4), ZZ(5), ZZ(6)]
    assert T.from_list_flat(flat, (2, 3), ZZ) == DM([[1, 2, 4], [4, 5, 6]])
    raises(DMBadInputError, lambda: T.from_list_flat(flat, (3, 3), ZZ))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_flat_nz(DM):
    M = DM([[1, 2, 0], [0, 0, 0], [0, 0, 3]])
    elements = [ZZ(1), ZZ(2), ZZ(3)]
    indices = ((0, 0), (0, 1), (2, 2))
    assert M.to_flat_nz() == (elements, (indices, M.shape))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_flat_nz(DM):
    T = type(DM([[0]]))
    elements = [ZZ(1), ZZ(2), ZZ(3)]
    indices = ((0, 0), (0, 1), (2, 2))
    data = (indices, (3, 3))
    result = DM([[1, 2, 0], [0, 0, 0], [0, 0, 3]])
    assert T.from_flat_nz(elements, data, ZZ) == result
    raises(DMBadInputError, lambda: T.from_flat_nz(elements, (indices, (2, 3)), ZZ))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_dod(DM):
    dod = {0: {0: ZZ(1), 2: ZZ(4)}, 1: {0: ZZ(4), 1: ZZ(5), 2: ZZ(6)}}
    assert DM([[1, 0, 4], [4, 5, 6]]).to_dod() == dod


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_dod(DM):
    T = type(DM([[0]]))
    dod = {0: {0: ZZ(1), 2: ZZ(4)}, 1: {0: ZZ(4), 1: ZZ(5), 2: ZZ(6)}}
    assert T.from_dod(dod, (2, 3), ZZ) == DM([[1, 0, 4], [4, 5, 6]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_dok(DM):
    dod = {(0, 0): ZZ(1), (0, 2): ZZ(4),
           (1, 0): ZZ(4), (1, 1): ZZ(5), (1, 2): ZZ(6)}
    assert DM([[1, 0, 4], [4, 5, 6]]).to_dok() == dod


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_dok(DM):
    T = type(DM([[0]]))
    dod = {(0, 0): ZZ(1), (0, 2): ZZ(4),
           (1, 0): ZZ(4), (1, 1): ZZ(5), (1, 2): ZZ(6)}
    assert T.from_dok(dod, (2, 3), ZZ) == DM([[1, 0, 4], [4, 5, 6]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_iter_values(DM):
    values = [ZZ(1), ZZ(4), ZZ(4), ZZ(5), ZZ(6)]
    assert sorted(DM([[1, 0, 4], [4, 5, 6]]).iter_values()) == values


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_iter_items(DM):
    items = [((0, 0), ZZ(1)), ((0, 2), ZZ(4)),
             ((1, 0), ZZ(4)), ((1, 1), ZZ(5)), ((1, 2), ZZ(6))]
    assert sorted(DM([[1, 0, 4], [4, 5, 6]]).iter_items()) == items


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_ddm(DM):
    T = type(DM([[0]]))
    ddm = DDM([[1, 2, 4], [4, 5, 6]], (2, 3), ZZ)
    assert T.from_ddm(ddm) == DM([[1, 2, 4], [4, 5, 6]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_zeros(DM):
    T = type(DM([[0]]))
    assert T.zeros((2, 3), ZZ) == DM([[0, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_ones(DM):
    T = type(DM([[0]]))
    assert T.ones((2, 3), ZZ) == DM([[1, 1, 1], [1, 1, 1]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_eye(DM):
    T = type(DM([[0]]))
    assert T.eye(3, ZZ) == DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert T.eye((3, 2), ZZ) == DM([[1, 0], [0, 1], [0, 0]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_diag(DM):
    T = type(DM([[0]]))
    assert T.diag([1, 2, 3], ZZ) == DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_transpose(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    assert A.transpose() == DM([[1, 4], [2, 5], [3, 6]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_add(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[2, 4, 6], [8, 10, 12]])
    assert A.add(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_sub(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[0, 0, 0], [0, 0, 0]])
    assert A.sub(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_mul(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    b = ZZ(2)
    assert A.mul(b) == DM([[2, 4, 6], [8, 10, 12]])
    assert A.rmul(b) == DM([[2, 4, 6], [8, 10, 12]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_matmul(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2], [3, 4], [5, 6]])
    C = DM([[22, 28], [49, 64]])
    assert A.matmul(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_mul_elementwise(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[1, 4, 9], [16, 25, 36]])
    assert A.mul_elementwise(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_neg(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[-1, -2, -3], [-4, -5, -6]])
    assert A.neg() == C


@pytest.mark.parametrize('DM', DM_all)
def test_XXM_convert_to(DM):
    A = DM([[1, 2, 3], [4, 5, 6]], ZZ)
    B = DM([[1, 2, 3], [4, 5, 6]], QQ)
    assert A.convert_to(QQ) == B
    assert B.convert_to(ZZ) == A


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_scc(DM):
    A = DM([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1]])
    assert A.scc() == [[0, 1], [2], [3, 5], [4]]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_hstack(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[7, 8], [9, 10]])
    C = DM([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]])
    ABC = DM([[1, 2, 3, 7, 8, 1, 2, 3, 7, 8],
              [4, 5, 6, 9, 10, 4, 5, 6, 9, 10]])
    assert A.hstack(B) == C
    assert A.hstack(B, C) == ABC


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_vstack(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[7, 8, 9]])
    C = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ABC = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A.vstack(B) == C
    assert A.vstack(B, C) == ABC


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_applyfunc(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[2, 4, 6], [8, 10, 12]])
    assert A.applyfunc(lambda x: 2*x, ZZ) == B


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_is_upper(DM):
    assert DM([[1, 2, 3], [0, 5, 6]]).is_upper() is True
    assert DM([[1, 2, 3], [4, 5, 6]]).is_upper() is False


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_is_lower(DM):
    assert DM([[1, 0, 0], [4, 5, 0]]).is_lower() is True
    assert DM([[1, 2, 3], [4, 5, 6]]).is_lower() is False


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_is_diagonal(DM):
    assert DM([[1, 0, 0], [0, 5, 0]]).is_diagonal() is True
    assert DM([[1, 2, 3], [4, 5, 6]]).is_diagonal() is False


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_diagonal(DM):
    assert DM([[1, 0, 0], [0, 5, 0]]).diagonal() == [1, 5]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_is_zero_matrix(DM):
    assert DM([[0, 0, 0], [0, 0, 0]]).is_zero_matrix() is True
    assert DM([[1, 0, 0], [0, 0, 0]]).is_zero_matrix() is False


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_det_ZZ(DM):
    assert DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).det() == 0
    assert DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).det() == -3


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_det_QQ(DM):
    dM1 = DM([[(1,2), (2,3)], [(3,4), (4,5)]])
    assert dM1.det() == QQ(-1,10)


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_inv_QQ(DM):
    dM1 = DM([[(1,2), (2,3)], [(3,4), (4,5)]])
    dM2 = DM([[(-8,1), (20,3)], [(15,2), (-5,1)]])
    assert dM1.inv() == dM2
    assert dM1.matmul(dM2) == DM([[1, 0], [0, 1]])

    dM3 = DM([[(1,2), (2,3)], [(1,4), (1,3)]])
    raises(DMNonInvertibleMatrixError, lambda: dM3.inv())

    dM4 = DM([[(1,2), (2,3), (3,4)], [(1,4), (1,3), (1,2)]])
    raises(DMNonSquareMatrixError, lambda: dM4.inv())


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_inv_ZZ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    # XXX: Maybe this should return a DM over QQ instead?
    # XXX: Handle unimodular matrices?
    raises(DMDomainError, lambda: dM1.inv())


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_charpoly_ZZ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    assert dM1.charpoly() == [1, -16, -12, 3]


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_charpoly_QQ(DM):
    dM1 = DM([[(1,2), (2,3)], [(3,4), (4,5)]])
    assert dM1.charpoly() == [QQ(1,1), QQ(-13,10), QQ(-1,10)]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_lu_solve_ZZ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    dM2 = DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    raises(DMDomainError, lambda: dM1.lu_solve(dM2))


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_lu_solve_QQ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    dM2 = DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dM3 = DM([[(-2,3),(-4,3),(1,1)],[(-2,3),(11,3),(-2,1)],[(1,1),(-2,1),(1,1)]])
    assert dM1.lu_solve(dM2) == dM3 == dM1.inv()

    dM4 = DM([[1, 2, 3], [4, 5, 6]])
    dM5 = DM([[1, 0], [0, 1], [0, 0]])
    raises(DMShapeError, lambda: dM4.lu_solve(dM5))


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_nullspace_QQ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # XXX: Change the signature to just return the nullspace. Possibly
    # returning the rank or nullity makes sense but the list of nonpivots is
    # not useful.
    assert dM1.nullspace() == (DM([[1, -2, 1]]), [2])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_lll(DM):
    M = DM([[1, 2, 3], [4, 5, 20]])
    M_lll = DM([[1, 2, 3], [-1, -5, 5]])
    T = DM([[1, 0], [-5, 1]])
    assert M.lll() == M_lll
    assert M.lll_transform() == (M_lll, T)
    assert T.matmul(M) == M_lll
