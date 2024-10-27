from sympy.external.gmpy import GROUND_TYPES

from sympy import Integer, Rational, S, sqrt, Matrix, symbols
from sympy import FF, ZZ, QQ, QQ_I, EXRAW

from sympy.polys.matrices.domainmatrix import DomainMatrix, DomainScalar, DM
from sympy.polys.matrices.exceptions import (
    DMBadInputError, DMDomainError, DMShapeError, DMFormatError, DMNotAField,
    DMNonSquareMatrixError, DMNonInvertibleMatrixError,
)
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM

from sympy.testing.pytest import raises


def test_DM():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DM([[1, 2], [3, 4]], ZZ)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == ZZ


def test_DomainMatrix_init():
    lol = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    dod = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    ddm = DDM(lol, (2, 2), ZZ)
    sdm = SDM(dod, (2, 2), ZZ)

    A = DomainMatrix(lol, (2, 2), ZZ)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    A = DomainMatrix(dod, (2, 2), ZZ)
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    raises(TypeError, lambda: DomainMatrix(ddm, (2, 2), ZZ))
    raises(TypeError, lambda: DomainMatrix(sdm, (2, 2), ZZ))
    raises(TypeError, lambda: DomainMatrix(Matrix([[1]]), (1, 1), ZZ))

    for fmt, rep in [('sparse', sdm), ('dense', ddm)]:
        if fmt == 'dense' and GROUND_TYPES == 'flint':
            rep = rep.to_dfm()
        A = DomainMatrix(lol, (2, 2), ZZ, fmt=fmt)
        assert A.rep == rep
        A = DomainMatrix(dod, (2, 2), ZZ, fmt=fmt)
        assert A.rep == rep

    raises(ValueError, lambda: DomainMatrix(lol, (2, 2), ZZ, fmt='invalid'))

    raises(DMBadInputError, lambda: DomainMatrix([[ZZ(1), ZZ(2)]], (2, 2), ZZ))


def test_DomainMatrix_from_rep():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DomainMatrix.from_rep(ddm)
    # XXX: Should from_rep convert to DFM?
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    sdm = SDM({0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    A = DomainMatrix.from_rep(sdm)
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    A = DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    raises(TypeError, lambda: DomainMatrix.from_rep(A))


def test_DomainMatrix_from_list():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DomainMatrix.from_list([[1, 2], [3, 4]], ZZ)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    dom = FF(7)
    ddm = DDM([[dom(1), dom(2)], [dom(3), dom(4)]], (2, 2), dom)
    A = DomainMatrix.from_list([[1, 2], [3, 4]], dom)
    # Not a DFM because FF(7) is not supported by DFM
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == dom

    ddm = DDM([[QQ(1, 2), QQ(3, 1)], [QQ(1, 4), QQ(5, 1)]], (2, 2), QQ)
    A = DomainMatrix.from_list([[(1, 2), (3, 1)], [(1, 4), (5, 1)]], QQ)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == QQ


def test_DomainMatrix_from_list_sympy():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DomainMatrix.from_list_sympy(2, 2, [[1, 2], [3, 4]])
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    K = QQ.algebraic_field(sqrt(2))
    ddm = DDM(
        [[K.convert(1 + sqrt(2)), K.convert(2 + sqrt(2))],
         [K.convert(3 + sqrt(2)), K.convert(4 + sqrt(2))]],
        (2, 2),
        K
    )
    A = DomainMatrix.from_list_sympy(
        2, 2, [[1 + sqrt(2), 2 + sqrt(2)], [3 + sqrt(2), 4 + sqrt(2)]],
        extension=True)
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == K


def test_DomainMatrix_from_dict_sympy():
    sdm = SDM({0: {0: QQ(1, 2)}, 1: {1: QQ(2, 3)}}, (2, 2), QQ)
    sympy_dict = {0: {0: Rational(1, 2)}, 1: {1: Rational(2, 3)}}
    A = DomainMatrix.from_dict_sympy(2, 2, sympy_dict)
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == QQ

    fds = DomainMatrix.from_dict_sympy
    raises(DMBadInputError, lambda: fds(2, 2, {3: {0: Rational(1, 2)}}))
    raises(DMBadInputError, lambda: fds(2, 2, {0: {3: Rational(1, 2)}}))


def test_DomainMatrix_from_Matrix():
    sdm = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
    A = DomainMatrix.from_Matrix(Matrix([[1, 2], [3, 4]]))
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    K = QQ.algebraic_field(sqrt(2))
    sdm = SDM(
        {0: {0: K.convert(1 + sqrt(2)), 1: K.convert(2 + sqrt(2))},
         1: {0: K.convert(3 + sqrt(2)), 1: K.convert(4 + sqrt(2))}},
        (2, 2),
        K
    )
    A = DomainMatrix.from_Matrix(
        Matrix([[1 + sqrt(2), 2 + sqrt(2)], [3 + sqrt(2), 4 + sqrt(2)]]),
        extension=True)
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == K

    A = DomainMatrix.from_Matrix(Matrix([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]]), fmt='dense')
    ddm = DDM([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]], (2, 2), QQ)

    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == QQ


def test_DomainMatrix_eq():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A == A
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(1)]], (2, 2), ZZ)
    assert A != B
    C = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    assert A != C


def test_DomainMatrix_unify_eq():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B1 = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    B2 = DomainMatrix([[QQ(1), QQ(3)], [QQ(3), QQ(4)]], (2, 2), QQ)
    B3 = DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    assert A.unify_eq(B1) is True
    assert A.unify_eq(B2) is False
    assert A.unify_eq(B3) is False


def test_DomainMatrix_get_domain():
    K, items = DomainMatrix.get_domain([1, 2, 3, 4])
    assert items == [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
    assert K == ZZ

    K, items = DomainMatrix.get_domain([1, 2, 3, Rational(1, 2)])
    assert items == [QQ(1), QQ(2), QQ(3), QQ(1, 2)]
    assert K == QQ


def test_DomainMatrix_convert_to():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = A.convert_to(QQ)
    assert Aq == DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)


def test_DomainMatrix_choose_domain():
    A = [[1, 2], [3, 0]]
    assert DM(A, QQ).choose_domain() == DM(A, ZZ)
    assert DM(A, QQ).choose_domain(field=True) == DM(A, QQ)
    assert DM(A, ZZ).choose_domain(field=True) == DM(A, QQ)

    x = symbols('x')
    B = [[1, x], [x**2, x**3]]
    assert DM(B, QQ[x]).choose_domain(field=True) == DM(B, ZZ.frac_field(x))


def test_DomainMatrix_to_flat_nz():
    Adm = DM([[1, 2], [3, 0]], ZZ)
    Addm = Adm.rep.to_ddm()
    Asdm = Adm.rep.to_sdm()
    for A in [Adm, Addm, Asdm]:
        elems, data = A.to_flat_nz()
        assert A.from_flat_nz(elems, data, A.domain) == A
        elemsq = [QQ(e) for e in elems]
        assert A.from_flat_nz(elemsq, data, QQ) == A.convert_to(QQ)
        elems2 = [2*e for e in elems]
        assert A.from_flat_nz(elems2, data, A.domain) == 2*A


def test_DomainMatrix_to_sympy():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_sympy() == A.convert_to(EXRAW)


def test_DomainMatrix_to_field():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = A.to_field()
    assert Aq == DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)


def test_DomainMatrix_to_sparse():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A_sparse = A.to_sparse()
    assert A_sparse.rep == {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}


def test_DomainMatrix_to_dense():
    A = DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)
    A_dense = A.to_dense()
    ddm = DDM([[1, 2], [3, 4]], (2, 2), ZZ)
    if GROUND_TYPES != 'flint':
        assert A_dense.rep == ddm
    else:
        assert A_dense.rep == ddm.to_dfm()


def test_DomainMatrix_unify():
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    assert Az.unify(Az) == (Az, Az)
    assert Az.unify(Aq) == (Aq, Aq)
    assert Aq.unify(Az) == (Aq, Aq)
    assert Aq.unify(Aq) == (Aq, Aq)

    As = DomainMatrix({0: {1: ZZ(1)}, 1:{0:ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    assert As.unify(As) == (As, As)
    assert Ad.unify(Ad) == (Ad, Ad)

    Bs, Bd = As.unify(Ad, fmt='dense')
    assert Bs.rep == DDM([[0, 1], [2, 0]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Bd.rep == DDM([[1, 2],[3, 4]], (2, 2), ZZ).to_dfm_or_ddm()

    Bs, Bd = As.unify(Ad, fmt='sparse')
    assert Bs.rep == SDM({0: {1: 1}, 1: {0: 2}}, (2, 2), ZZ)
    assert Bd.rep == SDM({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

    raises(ValueError, lambda: As.unify(Ad, fmt='invalid'))


def test_DomainMatrix_to_Matrix():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A_Matrix = Matrix([[1, 2], [3, 4]])
    assert A.to_Matrix() == A_Matrix
    assert A.to_sparse().to_Matrix() == A_Matrix
    assert A.convert_to(QQ).to_Matrix() == A_Matrix
    assert A.convert_to(QQ.algebraic_field(sqrt(2))).to_Matrix() == A_Matrix


def test_DomainMatrix_to_list():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_list() == [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]


def test_DomainMatrix_to_list_flat():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_list_flat() == [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]


def test_DomainMatrix_flat():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.flat() == [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]


def test_DomainMatrix_from_list_flat():
    nums = [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    assert DomainMatrix.from_list_flat(nums, (2, 2), ZZ) == A
    assert DDM.from_list_flat(nums, (2, 2), ZZ) == A.rep.to_ddm()
    assert SDM.from_list_flat(nums, (2, 2), ZZ) == A.rep.to_sdm()

    assert A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)

    raises(DMBadInputError, DomainMatrix.from_list_flat, nums, (2, 3), ZZ)
    raises(DMBadInputError, DDM.from_list_flat, nums, (2, 3), ZZ)
    raises(DMBadInputError, SDM.from_list_flat, nums, (2, 3), ZZ)


def test_DomainMatrix_to_dod():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_dod() == {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(4)]], (2, 2), ZZ)
    assert A.to_dod() == {0: {0: ZZ(1)}, 1: {1: ZZ(4)}}


def test_DomainMatrix_from_dod():
    items = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    A = DM([[1, 2], [3, 4]], ZZ)
    assert DomainMatrix.from_dod(items, (2, 2), ZZ) == A.to_sparse()
    assert A.from_dod_like(items) == A
    assert A.from_dod_like(items, QQ) == A.convert_to(QQ)


def test_DomainMatrix_to_dok():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_dok() == {(0, 0):ZZ(1), (0, 1):ZZ(2), (1, 0):ZZ(3), (1, 1):ZZ(4)}
    A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(4)]], (2, 2), ZZ)
    dok = {(0, 0):ZZ(1), (1, 1):ZZ(4)}
    assert A.to_dok() == dok
    assert A.to_dense().to_dok() == dok
    assert A.to_sparse().to_dok() == dok
    assert A.rep.to_ddm().to_dok() == dok
    assert A.rep.to_sdm().to_dok() == dok


def test_DomainMatrix_from_dok():
    items = {(0, 0): ZZ(1), (1, 1): ZZ(2)}
    A = DM([[1, 0], [0, 2]], ZZ)
    assert DomainMatrix.from_dok(items, (2, 2), ZZ) == A.to_sparse()
    assert DDM.from_dok(items, (2, 2), ZZ) == A.rep.to_ddm()
    assert SDM.from_dok(items, (2, 2), ZZ) == A.rep.to_sdm()


def test_DomainMatrix_repr():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert repr(A) == 'DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)'


def test_DomainMatrix_transpose():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    AT = DomainMatrix([[ZZ(1), ZZ(3)], [ZZ(2), ZZ(4)]], (2, 2), ZZ)
    assert A.transpose() == AT


def test_DomainMatrix_is_zero_matrix():
    A = DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    B = DomainMatrix([[ZZ(0)]], (1, 1), ZZ)
    assert A.is_zero_matrix is False
    assert B.is_zero_matrix is True


def test_DomainMatrix_is_upper():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(0), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.is_upper is True
    assert B.is_upper is False


def test_DomainMatrix_is_lower():
    A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.is_lower is True
    assert B.is_lower is False


def test_DomainMatrix_is_diagonal():
    A = DM([[1, 0], [0, 4]], ZZ)
    B = DM([[1, 2], [3, 4]], ZZ)
    assert A.is_diagonal is A.to_sparse().is_diagonal is True
    assert B.is_diagonal is B.to_sparse().is_diagonal is False


def test_DomainMatrix_is_square():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)], [ZZ(5), ZZ(6)]], (3, 2), ZZ)
    assert A.is_square is True
    assert B.is_square is False


def test_DomainMatrix_diagonal():
    A = DM([[1, 2], [3, 4]], ZZ)
    assert A.diagonal() == A.to_sparse().diagonal() == [ZZ(1), ZZ(4)]
    A = DM([[1, 2], [3, 4], [5, 6]], ZZ)
    assert A.diagonal() == A.to_sparse().diagonal() == [ZZ(1), ZZ(4)]
    A = DM([[1, 2, 3], [4, 5, 6]], ZZ)
    assert A.diagonal() == A.to_sparse().diagonal() == [ZZ(1), ZZ(5)]


def test_DomainMatrix_rank():
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(6), QQ(8)]], (3, 2), QQ)
    assert A.rank() == 2


def test_DomainMatrix_add():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert A + A == A.add(A) == B

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    L = [[2, 3], [3, 4]]
    raises(TypeError, lambda: A + L)
    raises(TypeError, lambda: L + A)

    A1 = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A1 + A2)
    raises(DMShapeError, lambda: A2 + A1)
    raises(DMShapeError, lambda: A1.add(A2))
    raises(DMShapeError, lambda: A2.add(A1))

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Asum = DomainMatrix([[QQ(2), QQ(4)], [QQ(6), QQ(8)]], (2, 2), QQ)
    assert Az + Aq == Asum
    assert Aq + Az == Asum
    raises(DMDomainError, lambda: Az.add(Aq))
    raises(DMDomainError, lambda: Aq.add(Az))

    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    Asd = As + Ad
    Ads = Ad + As
    assert Asd == DomainMatrix([[1, 3], [5, 4]], (2, 2), ZZ)
    assert Asd.rep == DDM([[1, 3], [5, 4]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Ads == DomainMatrix([[1, 3], [5, 4]], (2, 2), ZZ)
    assert Ads.rep == DDM([[1, 3], [5, 4]], (2, 2), ZZ).to_dfm_or_ddm()
    raises(DMFormatError, lambda: As.add(Ad))


def test_DomainMatrix_sub():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(0), ZZ(0)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    assert A - A == A.sub(A) == B

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    L = [[2, 3], [3, 4]]
    raises(TypeError, lambda: A - L)
    raises(TypeError, lambda: L - A)

    A1 = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A1 - A2)
    raises(DMShapeError, lambda: A2 - A1)
    raises(DMShapeError, lambda: A1.sub(A2))
    raises(DMShapeError, lambda: A2.sub(A1))

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Adiff = DomainMatrix([[QQ(0), QQ(0)], [QQ(0), QQ(0)]], (2, 2), QQ)
    assert Az - Aq == Adiff
    assert Aq - Az == Adiff
    raises(DMDomainError, lambda: Az.sub(Aq))
    raises(DMDomainError, lambda: Aq.sub(Az))

    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    Asd = As - Ad
    Ads = Ad - As
    assert Asd == DomainMatrix([[-1, -1], [-1, -4]], (2, 2), ZZ)
    assert Asd.rep == DDM([[-1, -1], [-1, -4]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Asd == -Ads
    assert Asd.rep == -Ads.rep


def test_DomainMatrix_neg():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aneg = DomainMatrix([[ZZ(-1), ZZ(-2)], [ZZ(-3), ZZ(-4)]], (2, 2), ZZ)
    assert -A == A.neg() == Aneg


def test_DomainMatrix_mul():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(7), ZZ(10)], [ZZ(15), ZZ(22)]], (2, 2), ZZ)
    assert A*A == A.matmul(A) == A2

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    L = [[1, 2], [3, 4]]
    raises(TypeError, lambda: A * L)
    raises(TypeError, lambda: L * A)

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Aprod = DomainMatrix([[QQ(7), QQ(10)], [QQ(15), QQ(22)]], (2, 2), QQ)
    assert Az * Aq == Aprod
    assert Aq * Az == Aprod
    raises(DMDomainError, lambda: Az.matmul(Aq))
    raises(DMDomainError, lambda: Aq.matmul(Az))

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    AA = DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    x = ZZ(2)
    assert A * x == x * A == A.mul(x) == AA

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    AA = DomainMatrix.zeros((2, 2), ZZ)
    x = ZZ(0)
    assert A * x == x * A == A.mul(x).to_sparse() == AA

    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    Asd = As * Ad
    Ads = Ad * As
    assert Asd == DomainMatrix([[3, 4], [2, 4]], (2, 2), ZZ)
    assert Asd.rep == DDM([[3, 4], [2, 4]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Ads == DomainMatrix([[4, 1], [8, 3]], (2, 2), ZZ)
    assert Ads.rep == DDM([[4, 1], [8, 3]], (2, 2), ZZ).to_dfm_or_ddm()


def test_DomainMatrix_mul_elementwise():
    A = DomainMatrix([[ZZ(2), ZZ(2)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(4), ZZ(0)], [ZZ(3), ZZ(0)]], (2, 2), ZZ)
    C = DomainMatrix([[ZZ(8), ZZ(0)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    assert A.mul_elementwise(B) == C
    assert B.mul_elementwise(A) == C


def test_DomainMatrix_pow():
    eye = DomainMatrix.eye(2, ZZ)
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(7), ZZ(10)], [ZZ(15), ZZ(22)]], (2, 2), ZZ)
    A3 = DomainMatrix([[ZZ(37), ZZ(54)], [ZZ(81), ZZ(118)]], (2, 2), ZZ)
    assert A**0 == A.pow(0) == eye
    assert A**1 == A.pow(1) == A
    assert A**2 == A.pow(2) == A2
    assert A**3 == A.pow(3) == A3

    raises(TypeError, lambda: A ** Rational(1, 2))
    raises(NotImplementedError, lambda: A ** -1)
    raises(NotImplementedError, lambda: A.pow(-1))

    A = DomainMatrix.zeros((2, 1), ZZ)
    raises(DMNonSquareMatrixError, lambda: A ** 1)


def test_DomainMatrix_clear_denoms():
    A = DM([[(1,2),(1,3)],[(1,4),(1,5)]], QQ)

    den_Z = DomainScalar(ZZ(60), ZZ)
    Anum_Z = DM([[30, 20], [15, 12]], ZZ)
    Anum_Q = Anum_Z.convert_to(QQ)

    assert A.clear_denoms() == (den_Z, Anum_Q)
    assert A.clear_denoms(convert=True) == (den_Z, Anum_Z)
    assert A * den_Z == Anum_Q
    assert A == Anum_Q / den_Z


def test_DomainMatrix_clear_denoms_rowwise():
    A = DM([[(1,2),(1,3)],[(1,4),(1,5)]], QQ)

    den_Z = DM([[6, 0], [0, 20]], ZZ).to_sparse()
    Anum_Z = DM([[3, 2], [5, 4]], ZZ)
    Anum_Q = DM([[3, 2], [5, 4]], QQ)

    assert A.clear_denoms_rowwise() == (den_Z, Anum_Q)
    assert A.clear_denoms_rowwise(convert=True) == (den_Z, Anum_Z)
    assert den_Z * A == Anum_Q
    assert A == den_Z.to_field().inv() * Anum_Q

    A = DM([[(1,2),(1,3),0,0],[0,0,0,0], [(1,4),(1,5),(1,6),(1,7)]], QQ)
    den_Z = DM([[6, 0, 0], [0, 1, 0], [0, 0, 420]], ZZ).to_sparse()
    Anum_Z = DM([[3, 2, 0, 0], [0, 0, 0, 0], [105, 84, 70, 60]], ZZ)
    Anum_Q = Anum_Z.convert_to(QQ)

    assert A.clear_denoms_rowwise() == (den_Z, Anum_Q)
    assert A.clear_denoms_rowwise(convert=True) == (den_Z, Anum_Z)
    assert den_Z * A == Anum_Q
    assert A == den_Z.to_field().inv() * Anum_Q


def test_DomainMatrix_cancel_denom():
    A = DM([[2, 4], [6, 8]], ZZ)
    assert A.cancel_denom(ZZ(1)) == (DM([[2, 4], [6, 8]], ZZ), ZZ(1))
    assert A.cancel_denom(ZZ(3)) == (DM([[2, 4], [6, 8]], ZZ), ZZ(3))
    assert A.cancel_denom(ZZ(4)) == (DM([[1, 2], [3, 4]], ZZ), ZZ(2))

    A = DM([[1, 2], [3, 4]], ZZ)
    assert A.cancel_denom(ZZ(2)) == (A, ZZ(2))
    assert A.cancel_denom(ZZ(-2)) == (-A, ZZ(2))

    # Test canonicalization of denominator over Gaussian rationals.
    A = DM([[1, 2], [3, 4]], QQ_I)
    assert A.cancel_denom(QQ_I(0,2)) == (QQ_I(0,-1)*A, QQ_I(2))

    raises(ZeroDivisionError, lambda: A.cancel_denom(ZZ(0)))


def test_DomainMatrix_cancel_denom_elementwise():
    A = DM([[2, 4], [6, 8]], ZZ)
    numers, denoms = A.cancel_denom_elementwise(ZZ(1))
    assert numers == DM([[2, 4], [6, 8]], ZZ)
    assert denoms == DM([[1, 1], [1, 1]], ZZ)
    numers, denoms = A.cancel_denom_elementwise(ZZ(4))
    assert numers == DM([[1, 1], [3, 2]], ZZ)
    assert denoms == DM([[2, 1], [2, 1]], ZZ)

    raises(ZeroDivisionError, lambda: A.cancel_denom_elementwise(ZZ(0)))


def test_DomainMatrix_content_primitive():
    A = DM([[2, 4], [6, 8]], ZZ)
    A_primitive = DM([[1, 2], [3, 4]], ZZ)
    A_content = ZZ(2)
    assert A.content() == A_content
    assert A.primitive() == (A_content, A_primitive)


def test_DomainMatrix_scc():
    Ad = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)],
                       [ZZ(0), ZZ(1), ZZ(0)],
                       [ZZ(2), ZZ(0), ZZ(4)]], (3, 3), ZZ)
    As = Ad.to_sparse()
    Addm = Ad.rep
    Asdm = As.rep
    for A in [Ad, As, Addm, Asdm]:
        assert Ad.scc() == [[1], [0, 2]]

    A = DM([[ZZ(1), ZZ(2), ZZ(3)]], ZZ)
    raises(DMNonSquareMatrixError, lambda: A.scc())


def test_DomainMatrix_rref():
    # More tests in test_rref.py
    A = DomainMatrix([], (0, 1), QQ)
    assert A.rref() == (A, ())

    A = DomainMatrix([[QQ(1)]], (1, 1), QQ)
    assert A.rref() == (A, (0,))

    A = DomainMatrix([[QQ(0)]], (1, 1), QQ)
    assert A.rref() == (A, ())

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Ar, pivots = A.rref()
    assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    assert pivots == (0, 1)

    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Ar, pivots = A.rref()
    assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    assert pivots == (0, 1)

    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(0), QQ(4)]], (2, 2), QQ)
    Ar, pivots = A.rref()
    assert Ar == DomainMatrix([[QQ(0), QQ(1)], [QQ(0), QQ(0)]], (2, 2), QQ)
    assert pivots == (1,)

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Ar, pivots = Az.rref()
    assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    assert pivots == (0, 1)

    methods = ('auto', 'GJ', 'FF', 'CD', 'GJ_dense', 'FF_dense', 'CD_dense')
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    for method in methods:
        Ar, pivots = Az.rref(method=method)
        assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
        assert pivots == (0, 1)

    raises(ValueError, lambda: Az.rref(method='foo'))
    raises(ValueError, lambda: Az.rref_den(method='foo'))


def test_DomainMatrix_columnspace():
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ)
    Acol = DomainMatrix([[QQ(1), QQ(1)], [QQ(2), QQ(3)]], (2, 2), QQ)
    assert A.columnspace() == Acol

    Az = DomainMatrix([[ZZ(1), ZZ(-1), ZZ(1)], [ZZ(2), ZZ(-2), ZZ(3)]], (2, 3), ZZ)
    raises(DMNotAField, lambda: Az.columnspace())

    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ, fmt='sparse')
    Acol = DomainMatrix({0: {0: QQ(1), 1: QQ(1)}, 1: {0: QQ(2), 1: QQ(3)}}, (2, 2), QQ)
    assert A.columnspace() == Acol


def test_DomainMatrix_rowspace():
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ)
    assert A.rowspace() == A

    Az = DomainMatrix([[ZZ(1), ZZ(-1), ZZ(1)], [ZZ(2), ZZ(-2), ZZ(3)]], (2, 3), ZZ)
    raises(DMNotAField, lambda: Az.rowspace())

    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ, fmt='sparse')
    assert A.rowspace() == A


def test_DomainMatrix_nullspace():
    A = DomainMatrix([[QQ(1), QQ(1)], [QQ(1), QQ(1)]], (2, 2), QQ)
    Anull = DomainMatrix([[QQ(-1), QQ(1)]], (1, 2), QQ)
    assert A.nullspace() == Anull

    A = DomainMatrix([[ZZ(1), ZZ(1)], [ZZ(1), ZZ(1)]], (2, 2), ZZ)
    Anull = DomainMatrix([[ZZ(-1), ZZ(1)]], (1, 2), ZZ)
    assert A.nullspace() == Anull

    raises(DMNotAField, lambda: A.nullspace(divide_last=True))

    A = DomainMatrix([[ZZ(2), ZZ(2)], [ZZ(2), ZZ(2)]], (2, 2), ZZ)
    Anull = DomainMatrix([[ZZ(-2), ZZ(2)]], (1, 2), ZZ)

    Arref, den, pivots = A.rref_den()
    assert den == ZZ(2)
    assert Arref.nullspace_from_rref() == Anull
    assert Arref.nullspace_from_rref(pivots) == Anull
    assert Arref.to_sparse().nullspace_from_rref() == Anull.to_sparse()
    assert Arref.to_sparse().nullspace_from_rref(pivots) == Anull.to_sparse()


def test_DomainMatrix_solve():
    # XXX: Maybe the _solve method should be changed...
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    particular = DomainMatrix([[1, 0]], (1, 2), QQ)
    nullspace = DomainMatrix([[-2, 1]], (1, 2), QQ)
    assert A._solve(b) == (particular, nullspace)

    b3 = DomainMatrix([[QQ(1)], [QQ(1)], [QQ(1)]], (3, 1), QQ)
    raises(DMShapeError, lambda: A._solve(b3))

    bz = DomainMatrix([[ZZ(1)], [ZZ(1)]], (2, 1), ZZ)
    raises(DMNotAField, lambda: A._solve(bz))


def test_DomainMatrix_inv():
    A = DomainMatrix([], (0, 0), QQ)
    assert A.inv() == A

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Ainv = DomainMatrix([[QQ(-2), QQ(1)], [QQ(3, 2), QQ(-1, 2)]], (2, 2), QQ)
    assert A.inv() == Ainv

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    raises(DMNotAField, lambda: Az.inv())

    Ans = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMNonSquareMatrixError, lambda: Ans.inv())

    Aninv = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(6)]], (2, 2), QQ)
    raises(DMNonInvertibleMatrixError, lambda: Aninv.inv())


def test_DomainMatrix_det():
    A = DomainMatrix([], (0, 0), ZZ)
    assert A.det() == 1

    A = DomainMatrix([[1]], (1, 1), ZZ)
    assert A.det() == 1

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.det() == ZZ(-2)

    A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(3), ZZ(5)]], (3, 3), ZZ)
    assert A.det() == ZZ(-1)

    A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(2), ZZ(5)]], (3, 3), ZZ)
    assert A.det() == ZZ(0)

    Ans = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMNonSquareMatrixError, lambda: Ans.det())

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    assert A.det() == QQ(-2)


def test_DomainMatrix_eval_poly():
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    p = [ZZ(1), ZZ(2), ZZ(3)]
    result = DomainMatrix([[ZZ(12), ZZ(14)], [ZZ(21), ZZ(33)]], (2, 2), ZZ)
    assert dM.eval_poly(p) == result == p[0]*dM**2 + p[1]*dM + p[2]*dM**0
    assert dM.eval_poly([]) == dM.zeros(dM.shape, dM.domain)
    assert dM.eval_poly([ZZ(2)]) == 2*dM.eye(2, dM.domain)

    dM2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMNonSquareMatrixError, lambda: dM2.eval_poly([ZZ(1)]))


def test_DomainMatrix_eval_poly_mul():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    p = [ZZ(1), ZZ(2), ZZ(3)]
    result = DomainMatrix([[ZZ(40)], [ZZ(87)]], (2, 1), ZZ)
    assert A.eval_poly_mul(p, b) == result == p[0]*A**2*b + p[1]*A*b + p[2]*b

    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    dM1 = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMNonSquareMatrixError, lambda: dM1.eval_poly_mul([ZZ(1)], b))
    b1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: dM.eval_poly_mul([ZZ(1)], b1))
    bq = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    raises(DMDomainError, lambda: dM.eval_poly_mul([ZZ(1)], bq))


def _check_solve_den(A, b, xnum, xden):
    # Examples for solve_den, solve_den_charpoly, solve_den_rref should use
    # this so that all methods and types are tested.

    case1 = (A, xnum, b)
    case2 = (A.to_sparse(), xnum.to_sparse(), b.to_sparse())

    for Ai, xnum_i, b_i in [case1, case2]:
        # The key invariant for solve_den:
        assert Ai*xnum_i == xden*b_i

        # solve_den_rref can differ at least by a minus sign
        answers = [(xnum_i, xden), (-xnum_i, -xden)]
        assert Ai.solve_den(b) in answers
        assert Ai.solve_den(b, method='rref') in answers
        assert Ai.solve_den_rref(b) in answers

        # charpoly can only be used if A is square and guarantees to return the
        # actual determinant as a denominator.
        m, n = Ai.shape
        if m == n:
            assert Ai.solve_den(b_i, method='charpoly') == (xnum_i, xden)
            assert Ai.solve_den_charpoly(b_i) == (xnum_i, xden)
        else:
            raises(DMNonSquareMatrixError, lambda: Ai.solve_den_charpoly(b))
            raises(DMNonSquareMatrixError, lambda: Ai.solve_den(b, method='charpoly'))


def test_DomainMatrix_solve_den():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    result = DomainMatrix([[ZZ(0)], [ZZ(-1)]], (2, 1), ZZ)
    den = ZZ(-2)
    _check_solve_den(A, b, result, den)

    A = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(1), ZZ(2), ZZ(4)],
        [ZZ(1), ZZ(3), ZZ(5)]], (3, 3), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)], [ZZ(3)]], (3, 1), ZZ)
    result = DomainMatrix([[ZZ(2)], [ZZ(0)], [ZZ(-1)]], (3, 1), ZZ)
    den = ZZ(-1)
    _check_solve_den(A, b, result, den)

    A = DomainMatrix([[ZZ(2)], [ZZ(2)]], (2, 1), ZZ)
    b = DomainMatrix([[ZZ(3)], [ZZ(3)]], (2, 1), ZZ)
    result = DomainMatrix([[ZZ(3)]], (1, 1), ZZ)
    den = ZZ(2)
    _check_solve_den(A, b, result, den)


def test_DomainMatrix_solve_den_charpoly():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    A1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMNonSquareMatrixError, lambda: A1.solve_den_charpoly(b))
    b1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A.solve_den_charpoly(b1))
    bq = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    raises(DMDomainError, lambda: A.solve_den_charpoly(bq))


def test_DomainMatrix_solve_den_charpoly_check():
    # Test check
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(2), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(3)]], (2, 1), ZZ)
    raises(DMNonInvertibleMatrixError, lambda: A.solve_den_charpoly(b))
    adjAb = DomainMatrix([[ZZ(-2)], [ZZ(1)]], (2, 1), ZZ)
    assert A.adjugate() * b == adjAb
    assert A.solve_den_charpoly(b, check=False) == (adjAb, ZZ(0))


def test_DomainMatrix_solve_den_errors():
    A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMShapeError, lambda: A.solve_den(b))
    raises(DMShapeError, lambda: A.solve_den_rref(b))

    A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    b = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A.solve_den(b))
    raises(DMShapeError, lambda: A.solve_den_rref(b))

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A.solve_den(b1))

    A = DomainMatrix([[ZZ(2)]], (1, 1), ZZ)
    b = DomainMatrix([[ZZ(2)]], (1, 1), ZZ)
    raises(DMBadInputError, lambda: A.solve_den(b1, method='invalid'))

    A = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMNonSquareMatrixError, lambda: A.solve_den_charpoly(b))


def test_DomainMatrix_solve_den_rref_underdetermined():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(1), ZZ(2)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(1)]], (2, 1), ZZ)
    raises(DMNonInvertibleMatrixError, lambda: A.solve_den(b))
    raises(DMNonInvertibleMatrixError, lambda: A.solve_den_rref(b))


def test_DomainMatrix_adj_poly_det():
    A = DM([[ZZ(1), ZZ(2), ZZ(3)],
            [ZZ(4), ZZ(5), ZZ(6)],
            [ZZ(7), ZZ(8), ZZ(9)]], ZZ)
    p, detA = A.adj_poly_det()
    assert p == [ZZ(1), ZZ(-15), ZZ(-18)]
    assert A.adjugate() == p[0]*A**2 + p[1]*A**1 + p[2]*A**0 == A.eval_poly(p)
    assert A.det() == detA

    A = DM([[ZZ(1), ZZ(2), ZZ(3)],
            [ZZ(7), ZZ(8), ZZ(9)]], ZZ)
    raises(DMNonSquareMatrixError, lambda: A.adj_poly_det())


def test_DomainMatrix_inv_den():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    den = ZZ(-2)
    result = DomainMatrix([[ZZ(4), ZZ(-2)], [ZZ(-3), ZZ(1)]], (2, 2), ZZ)
    assert A.inv_den() == (result, den)


def test_DomainMatrix_adjugate():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    result = DomainMatrix([[ZZ(4), ZZ(-2)], [ZZ(-3), ZZ(1)]], (2, 2), ZZ)
    assert A.adjugate() == result


def test_DomainMatrix_adj_det():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    adjA = DomainMatrix([[ZZ(4), ZZ(-2)], [ZZ(-3), ZZ(1)]], (2, 2), ZZ)
    assert A.adj_det() == (adjA, ZZ(-2))


def test_DomainMatrix_lu():
    A = DomainMatrix([], (0, 0), QQ)
    assert A.lu() == (A, A, [])

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(3), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(1), QQ(2)], [QQ(0), QQ(-2)]], (2, 2), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(3), QQ(4)], [QQ(0), QQ(2)]], (2, 2), QQ)
    swaps = [(0, 1)]
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(2), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(1), QQ(2)], [QQ(0), QQ(0)]], (2, 2), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(0), QQ(4)]], (2, 2), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(0), QQ(2)], [QQ(0), QQ(4)]], (2, 2), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]], (2, 3), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(4), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(1), QQ(2), QQ(3)], [QQ(0), QQ(-3), QQ(-6)]], (2, 3), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    L = DomainMatrix([
        [QQ(1), QQ(0), QQ(0)],
        [QQ(3), QQ(1), QQ(0)],
        [QQ(5), QQ(2), QQ(1)]], (3, 3), QQ)
    U = DomainMatrix([[QQ(1), QQ(2)], [QQ(0), QQ(-2)], [QQ(0), QQ(0)]], (3, 2), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 2]]
    L = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
    U = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    to_dom = lambda rows, dom: [[dom(e) for e in row] for row in rows]
    A = DomainMatrix(to_dom(A, QQ), (4, 4), QQ)
    L = DomainMatrix(to_dom(L, QQ), (4, 4), QQ)
    U = DomainMatrix(to_dom(U, QQ), (4, 4), QQ)
    assert A.lu() == (L, U, [])

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    raises(DMNotAField, lambda: A.lu())


def test_DomainMatrix_lu_solve():
    # Base case
    A = b = x = DomainMatrix([], (0, 0), QQ)
    assert A.lu_solve(b) == x

    # Basic example
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DomainMatrix([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Example with swaps
    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DomainMatrix([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Non-invertible
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # Overdetermined, consistent
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    x = DomainMatrix([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Overdetermined, inconsistent
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)], [QQ(4)]], (3, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # Underdetermined
    A = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    b = DomainMatrix([[QQ(1)]], (1, 1), QQ)
    raises(NotImplementedError, lambda: A.lu_solve(b))

    # Non-field
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMNotAField, lambda: A.lu_solve(b))

    # Shape mismatch
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMShapeError, lambda: A.lu_solve(b))


def test_DomainMatrix_charpoly():
    A = DomainMatrix([], (0, 0), ZZ)
    p = [ZZ(1)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[1]], (1, 1), ZZ)
    p = [ZZ(1), ZZ(-1)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    p = [ZZ(1), ZZ(-5), ZZ(-2)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    p = [ZZ(1), ZZ(-15), ZZ(-18), ZZ(0)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[ZZ(0), ZZ(1), ZZ(0)],
                      [ZZ(1), ZZ(0), ZZ(1)],
                      [ZZ(0), ZZ(1), ZZ(0)]], (3, 3), ZZ)
    p = [ZZ(1), ZZ(0), ZZ(-2), ZZ(0)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DM([[17, 0, 30,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [69, 0,  0,  0,  0, 86, 0,  0, 0, 0],
            [23, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0, 13,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0, 32, 0, 0],
            [ 0, 0,  0,  0, 37, 67, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0]], ZZ)
    p = ZZ.map([1, -17, -2070, 0, -771420, 0, 0, 0, 0, 0, 0])
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    Ans = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMNonSquareMatrixError, lambda: Ans.charpoly())


def test_DomainMatrix_charpoly_factor_list():
    A = DomainMatrix([], (0, 0), ZZ)
    assert A.charpoly_factor_list() == []

    A = DM([[1]], ZZ)
    assert A.charpoly_factor_list() == [
        ([ZZ(1), ZZ(-1)], 1)
    ]

    A = DM([[1, 2], [3, 4]], ZZ)
    assert A.charpoly_factor_list() == [
        ([ZZ(1), ZZ(-5), ZZ(-2)], 1)
    ]

    A = DM([[1, 2, 0], [3, 4, 0], [0, 0, 1]], ZZ)
    assert A.charpoly_factor_list() == [
        ([ZZ(1), ZZ(-1)], 1),
        ([ZZ(1), ZZ(-5), ZZ(-2)], 1)
    ]


def test_DomainMatrix_eye():
    A = DomainMatrix.eye(3, QQ)
    assert A.rep == SDM.eye((3, 3), QQ)
    assert A.shape == (3, 3)
    assert A.domain == QQ


def test_DomainMatrix_zeros():
    A = DomainMatrix.zeros((1, 2), QQ)
    assert A.rep == SDM.zeros((1, 2), QQ)
    assert A.shape == (1, 2)
    assert A.domain == QQ


def test_DomainMatrix_ones():
    A = DomainMatrix.ones((2, 3), QQ)
    if GROUND_TYPES != 'flint':
        assert A.rep == DDM.ones((2, 3), QQ)
    else:
        assert A.rep == SDM.ones((2, 3), QQ).to_dfm()
    assert A.shape == (2, 3)
    assert A.domain == QQ


def test_DomainMatrix_diag():
    A = DomainMatrix({0:{0:ZZ(2)}, 1:{1:ZZ(3)}}, (2, 2), ZZ)
    assert DomainMatrix.diag([ZZ(2), ZZ(3)], ZZ) == A

    A = DomainMatrix({0:{0:ZZ(2)}, 1:{1:ZZ(3)}}, (3, 4), ZZ)
    assert DomainMatrix.diag([ZZ(2), ZZ(3)], ZZ, (3, 4)) == A


def test_DomainMatrix_hstack():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
    C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)

    AB = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(5), ZZ(6)],
        [ZZ(3), ZZ(4), ZZ(7), ZZ(8)]], (2, 4), ZZ)
    ABC = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(5), ZZ(6), ZZ(9), ZZ(10)],
        [ZZ(3), ZZ(4), ZZ(7), ZZ(8), ZZ(11), ZZ(12)]], (2, 6), ZZ)
    assert A.hstack(B) == AB
    assert A.hstack(B, C) == ABC


def test_DomainMatrix_vstack():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
    C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)

    AB = DomainMatrix([
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8)]], (4, 2), ZZ)
    ABC = DomainMatrix([
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8)],
        [ZZ(9), ZZ(10)],
        [ZZ(11), ZZ(12)]], (6, 2), ZZ)
    assert A.vstack(B) == AB
    assert A.vstack(B, C) == ABC


def test_DomainMatrix_applyfunc():
    A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    B = DomainMatrix([[ZZ(2), ZZ(4)]], (1, 2), ZZ)
    assert A.applyfunc(lambda x: 2*x) == B


def test_DomainMatrix_scalarmul():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    lamda = DomainScalar(QQ(3)/QQ(2), QQ)
    assert A * lamda == DomainMatrix([[QQ(3, 2), QQ(3)], [QQ(9, 2), QQ(6)]], (2, 2), QQ)
    assert A * 2 == DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert 2 * A == DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert A * DomainScalar(ZZ(0), ZZ) == DomainMatrix({}, (2, 2), ZZ)
    assert A * DomainScalar(ZZ(1), ZZ) == A

    raises(TypeError, lambda: A * 1.5)


def test_DomainMatrix_truediv():
    A = DomainMatrix.from_Matrix(Matrix([[1, 2], [3, 4]]))
    lamda = DomainScalar(QQ(3)/QQ(2), QQ)
    assert A / lamda == DomainMatrix({0: {0: QQ(2, 3), 1: QQ(4, 3)}, 1: {0: QQ(2), 1: QQ(8, 3)}}, (2, 2), QQ)
    b = DomainScalar(ZZ(1), ZZ)
    assert A / b == DomainMatrix({0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)

    assert A / 1 == DomainMatrix({0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)
    assert A / 2 == DomainMatrix({0: {0: QQ(1, 2), 1: QQ(1)}, 1: {0: QQ(3, 2), 1: QQ(2)}}, (2, 2), QQ)

    raises(ZeroDivisionError, lambda: A / 0)
    raises(TypeError, lambda: A / 1.5)
    raises(ZeroDivisionError, lambda: A / DomainScalar(ZZ(0), ZZ))

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_field() / 2 == DomainMatrix([[QQ(1, 2), QQ(1)], [QQ(3, 2), QQ(2)]], (2, 2), QQ)
    assert A / 2 == DomainMatrix([[QQ(1, 2), QQ(1)], [QQ(3, 2), QQ(2)]], (2, 2), QQ)
    assert A.to_field() / QQ(2,3) == DomainMatrix([[QQ(3, 2), QQ(3)], [QQ(9, 2), QQ(6)]], (2, 2), QQ)


def test_DomainMatrix_getitem():
    dM = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)

    assert dM[1:,:-2] == DomainMatrix([[ZZ(4)], [ZZ(7)]], (2, 1), ZZ)
    assert dM[2,:-2] == DomainMatrix([[ZZ(7)]], (1, 1), ZZ)
    assert dM[:-2,:-2] == DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    assert dM[:-1,0:2] == DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(4), ZZ(5)]], (2, 2), ZZ)
    assert dM[:, -1] == DomainMatrix([[ZZ(3)], [ZZ(6)], [ZZ(9)]], (3, 1), ZZ)
    assert dM[-1, :] == DomainMatrix([[ZZ(7), ZZ(8), ZZ(9)]], (1, 3), ZZ)
    assert dM[::-1, :] == DomainMatrix([
                            [ZZ(7), ZZ(8), ZZ(9)],
                            [ZZ(4), ZZ(5), ZZ(6)],
                            [ZZ(1), ZZ(2), ZZ(3)]], (3, 3), ZZ)

    raises(IndexError, lambda: dM[4, :-2])
    raises(IndexError, lambda: dM[:-2, 4])

    assert dM[1, 2] == DomainScalar(ZZ(6), ZZ)
    assert dM[-2, 2] == DomainScalar(ZZ(6), ZZ)
    assert dM[1, -2] == DomainScalar(ZZ(5), ZZ)
    assert dM[-1, -3] == DomainScalar(ZZ(7), ZZ)

    raises(IndexError, lambda: dM[3, 3])
    raises(IndexError, lambda: dM[1, 4])
    raises(IndexError, lambda: dM[-1, -4])

    dM = DomainMatrix({0: {0: ZZ(1)}}, (10, 10), ZZ)
    assert dM[5, 5] == DomainScalar(ZZ(0), ZZ)
    assert dM[0, 0] == DomainScalar(ZZ(1), ZZ)

    dM = DomainMatrix({1: {0: 1}}, (2,1), ZZ)
    assert dM[0:, 0] == DomainMatrix({1: {0: 1}}, (2, 1), ZZ)
    raises(IndexError, lambda: dM[3, 0])

    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    assert dM[:2,:2] == DomainMatrix({}, (2, 2), ZZ)
    assert dM[2:,2:] == DomainMatrix({0: {0: 1}, 2: {2: 1}}, (3, 3), ZZ)
    assert dM[3:,3:] == DomainMatrix({1: {1: 1}}, (2, 2), ZZ)
    assert dM[2:, 6:] == DomainMatrix({}, (3, 0), ZZ)


def test_DomainMatrix_getitem_sympy():
    dM = DomainMatrix({2: {2: ZZ(2)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    val1 = dM.getitem_sympy(0, 0)
    assert val1 is S.Zero
    val2 = dM.getitem_sympy(2, 2)
    assert val2 == 2 and isinstance(val2, Integer)


def test_DomainMatrix_extract():
    dM1 = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    dM2 = DomainMatrix([
        [ZZ(1), ZZ(3)],
        [ZZ(7), ZZ(9)]], (2, 2), ZZ)
    assert dM1.extract([0, 2], [0, 2]) == dM2
    assert dM1.to_sparse().extract([0, 2], [0, 2]) == dM2.to_sparse()
    assert dM1.extract([0, -1], [0, -1]) == dM2
    assert dM1.to_sparse().extract([0, -1], [0, -1]) == dM2.to_sparse()

    dM3 = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(2)],
        [ZZ(4), ZZ(5), ZZ(5)],
        [ZZ(4), ZZ(5), ZZ(5)]], (3, 3), ZZ)
    assert dM1.extract([0, 1, 1], [0, 1, 1]) == dM3
    assert dM1.to_sparse().extract([0, 1, 1], [0, 1, 1]) == dM3.to_sparse()

    empty = [
        ([], [], (0, 0)),
        ([1], [], (1, 0)),
        ([], [1], (0, 1)),
    ]
    for rows, cols, size in empty:
        assert dM1.extract(rows, cols) == DomainMatrix.zeros(size, ZZ).to_dense()
        assert dM1.to_sparse().extract(rows, cols) == DomainMatrix.zeros(size, ZZ)

    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    bad_indices = [([2], [0]), ([0], [2]), ([-3], [0]), ([0], [-3])]
    for rows, cols in bad_indices:
        raises(IndexError, lambda: dM.extract(rows, cols))
        raises(IndexError, lambda: dM.to_sparse().extract(rows, cols))


def test_DomainMatrix_setitem():
    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    dM[2, 2] = ZZ(2)
    assert dM == DomainMatrix({2: {2: ZZ(2)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    def setitem(i, j, val):
        dM[i, j] = val
    raises(TypeError, lambda: setitem(2, 2, QQ(1, 2)))
    raises(NotImplementedError, lambda: setitem(slice(1, 2), 2, ZZ(1)))


def test_DomainMatrix_pickling():
    import pickle
    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    assert pickle.loads(pickle.dumps(dM)) == dM
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert pickle.loads(pickle.dumps(dM)) == dM
