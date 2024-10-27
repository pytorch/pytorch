"""
Tests for the basic functionality of the SDM class.
"""

from itertools import product

from sympy.core.singleton import S
from sympy.external.gmpy import GROUND_TYPES
from sympy.testing.pytest import raises

from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
                                             DMShapeError)


def test_SDM():
    A = SDM({0:{0:ZZ(1)}}, (2, 2), ZZ)
    assert A.domain == ZZ
    assert A.shape == (2, 2)
    assert dict(A) == {0:{0:ZZ(1)}}

    raises(DMBadInputError, lambda: SDM({5:{1:ZZ(0)}}, (2, 2), ZZ))
    raises(DMBadInputError, lambda: SDM({0:{5:ZZ(0)}}, (2, 2), ZZ))


def test_DDM_str():
    sdm = SDM({0:{0:ZZ(1)}, 1:{1:ZZ(1)}}, (2, 2), ZZ)
    assert str(sdm) == '{0: {0: 1}, 1: {1: 1}}'
    if GROUND_TYPES == 'gmpy': # pragma: no cover
        assert repr(sdm) == 'SDM({0: {0: mpz(1)}, 1: {1: mpz(1)}}, (2, 2), ZZ)'
    else:        # pragma: no cover
        assert repr(sdm) == 'SDM({0: {0: 1}, 1: {1: 1}}, (2, 2), ZZ)'


def test_SDM_new():
    A = SDM({0:{0:ZZ(1)}}, (2, 2), ZZ)
    B = A.new({}, (2, 2), ZZ)
    assert B == SDM({}, (2, 2), ZZ)


def test_SDM_copy():
    A = SDM({0:{0:ZZ(1)}}, (2, 2), ZZ)
    B = A.copy()
    assert A == B
    A[0][0] = ZZ(2)
    assert A != B


def test_SDM_from_list():
    A = SDM.from_list([[ZZ(0), ZZ(1)], [ZZ(1), ZZ(0)]], (2, 2), ZZ)
    assert A == SDM({0:{1:ZZ(1)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)

    raises(DMBadInputError, lambda: SDM.from_list([[ZZ(0)], [ZZ(0), ZZ(1)]], (2, 2), ZZ))
    raises(DMBadInputError, lambda: SDM.from_list([[ZZ(0), ZZ(1)]], (2, 2), ZZ))


def test_SDM_to_list():
    A = SDM({0:{1: ZZ(1)}}, (2, 2), ZZ)
    assert A.to_list() == [[ZZ(0), ZZ(1)], [ZZ(0), ZZ(0)]]

    A = SDM({}, (0, 2), ZZ)
    assert A.to_list() == []

    A = SDM({}, (2, 0), ZZ)
    assert A.to_list() == [[], []]


def test_SDM_to_list_flat():
    A = SDM({0:{1: ZZ(1)}}, (2, 2), ZZ)
    assert A.to_list_flat() == [ZZ(0), ZZ(1), ZZ(0), ZZ(0)]


def test_SDM_to_dok():
    A = SDM({0:{1: ZZ(1)}}, (2, 2), ZZ)
    assert A.to_dok() == {(0, 1): ZZ(1)}


def test_SDM_from_ddm():
    A = DDM([[ZZ(1), ZZ(0)], [ZZ(1), ZZ(0)]], (2, 2), ZZ)
    B = SDM.from_ddm(A)
    assert B.domain == ZZ
    assert B.shape == (2, 2)
    assert dict(B) == {0:{0:ZZ(1)}, 1:{0:ZZ(1)}}


def test_SDM_to_ddm():
    A = SDM({0:{1: ZZ(1)}}, (2, 2), ZZ)
    B = DDM([[ZZ(0), ZZ(1)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    assert A.to_ddm() == B


def test_SDM_to_sdm():
    A = SDM({0:{1: ZZ(1)}}, (2, 2), ZZ)
    assert A.to_sdm() == A


def test_SDM_getitem():
    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    assert A.getitem(0, 0) == ZZ.zero
    assert A.getitem(0, 1) == ZZ.one
    assert A.getitem(1, 0) == ZZ.zero
    assert A.getitem(-2, -2) == ZZ.zero
    assert A.getitem(-2, -1) == ZZ.one
    assert A.getitem(-1, -2) == ZZ.zero
    raises(IndexError, lambda: A.getitem(2, 0))
    raises(IndexError, lambda: A.getitem(0, 2))


def test_SDM_setitem():
    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    A.setitem(0, 0, ZZ(1))
    assert A == SDM({0:{0:ZZ(1), 1:ZZ(1)}}, (2, 2), ZZ)
    A.setitem(1, 0, ZZ(1))
    assert A == SDM({0:{0:ZZ(1), 1:ZZ(1)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
    A.setitem(1, 0, ZZ(0))
    assert A == SDM({0:{0:ZZ(1), 1:ZZ(1)}}, (2, 2), ZZ)
    # Repeat the above test so that this time the row is empty
    A.setitem(1, 0, ZZ(0))
    assert A == SDM({0:{0:ZZ(1), 1:ZZ(1)}}, (2, 2), ZZ)
    A.setitem(0, 0, ZZ(0))
    assert A == SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    # This time the row is there but column is empty
    A.setitem(0, 0, ZZ(0))
    assert A == SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    raises(IndexError, lambda: A.setitem(2, 0, ZZ(1)))
    raises(IndexError, lambda: A.setitem(0, 2, ZZ(1)))


def test_SDM_extract_slice():
    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    B = A.extract_slice(slice(1, 2), slice(1, 2))
    assert B == SDM({0:{0:ZZ(4)}}, (1, 1), ZZ)


def test_SDM_extract():
    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    B = A.extract([1], [1])
    assert B == SDM({0:{0:ZZ(4)}}, (1, 1), ZZ)
    B = A.extract([1, 0], [1, 0])
    assert B == SDM({0:{0:ZZ(4), 1:ZZ(3)}, 1:{0:ZZ(2), 1:ZZ(1)}}, (2, 2), ZZ)
    B = A.extract([1, 1], [1, 1])
    assert B == SDM({0:{0:ZZ(4), 1:ZZ(4)}, 1:{0:ZZ(4), 1:ZZ(4)}}, (2, 2), ZZ)
    B = A.extract([-1], [-1])
    assert B == SDM({0:{0:ZZ(4)}}, (1, 1), ZZ)

    A = SDM({}, (2, 2), ZZ)
    B = A.extract([0, 1, 0], [0, 0])
    assert B == SDM({}, (3, 2), ZZ)

    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    assert A.extract([], []) == SDM.zeros((0, 0), ZZ)
    assert A.extract([1], []) == SDM.zeros((1, 0), ZZ)
    assert A.extract([], [1]) == SDM.zeros((0, 1), ZZ)

    raises(IndexError, lambda: A.extract([2], [0]))
    raises(IndexError, lambda: A.extract([0], [2]))
    raises(IndexError, lambda: A.extract([-3], [0]))
    raises(IndexError, lambda: A.extract([0], [-3]))


def test_SDM_zeros():
    A = SDM.zeros((2, 2), ZZ)
    assert A.domain == ZZ
    assert A.shape == (2, 2)
    assert dict(A) == {}

def test_SDM_ones():
    A = SDM.ones((1, 2), QQ)
    assert A.domain == QQ
    assert A.shape == (1, 2)
    assert dict(A) == {0:{0:QQ(1), 1:QQ(1)}}

def test_SDM_eye():
    A = SDM.eye((2, 2), ZZ)
    assert A.domain == ZZ
    assert A.shape == (2, 2)
    assert dict(A) == {0:{0:ZZ(1)}, 1:{1:ZZ(1)}}


def test_SDM_diag():
    A = SDM.diag([ZZ(1), ZZ(2)], ZZ, (2, 3))
    assert A == SDM({0:{0:ZZ(1)}, 1:{1:ZZ(2)}}, (2, 3), ZZ)


def test_SDM_transpose():
    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(1), 1:ZZ(3)}, 1:{0:ZZ(2), 1:ZZ(4)}}, (2, 2), ZZ)
    assert A.transpose() == B

    A = SDM({0:{1:ZZ(2)}}, (2, 2), ZZ)
    B = SDM({1:{0:ZZ(2)}}, (2, 2), ZZ)
    assert A.transpose() == B

    A = SDM({0:{1:ZZ(2)}}, (1, 2), ZZ)
    B = SDM({1:{0:ZZ(2)}}, (2, 1), ZZ)
    assert A.transpose() == B


def test_SDM_mul():
    A = SDM({0:{0:ZZ(2)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(4)}}, (2, 2), ZZ)
    assert A*ZZ(2) == B
    assert ZZ(2)*A == B

    raises(TypeError, lambda: A*QQ(1, 2))
    raises(TypeError, lambda: QQ(1, 2)*A)


def test_SDM_mul_elementwise():
    A = SDM({0:{0:ZZ(2), 1:ZZ(2)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(4)}, 1:{0:ZZ(3)}}, (2, 2), ZZ)
    C = SDM({0:{0:ZZ(8)}}, (2, 2), ZZ)
    assert A.mul_elementwise(B) == C
    assert B.mul_elementwise(A) == C

    Aq = A.convert_to(QQ)
    A1 = SDM({0:{0:ZZ(1)}}, (1, 1), ZZ)

    raises(DMDomainError, lambda: Aq.mul_elementwise(B))
    raises(DMShapeError, lambda: A1.mul_elementwise(B))


def test_SDM_matmul():
    A = SDM({0:{0:ZZ(2)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(4)}}, (2, 2), ZZ)
    assert A.matmul(A) == A*A == B

    C = SDM({0:{0:ZZ(2)}}, (2, 2), QQ)
    raises(DMDomainError, lambda: A.matmul(C))

    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(7), 1:ZZ(10)}, 1:{0:ZZ(15), 1:ZZ(22)}}, (2, 2), ZZ)
    assert A.matmul(A) == A*A == B

    A22 = SDM({0:{0:ZZ(4)}}, (2, 2), ZZ)
    A32 = SDM({0:{0:ZZ(2)}}, (3, 2), ZZ)
    A23 = SDM({0:{0:ZZ(4)}}, (2, 3), ZZ)
    A33 = SDM({0:{0:ZZ(8)}}, (3, 3), ZZ)
    A22 = SDM({0:{0:ZZ(8)}}, (2, 2), ZZ)
    assert A32.matmul(A23) == A33
    assert A23.matmul(A32) == A22
    # XXX: @ not supported by SDM...
    #assert A32.matmul(A23) == A32 @ A23 == A33
    #assert A23.matmul(A32) == A23 @ A32 == A22
    #raises(DMShapeError, lambda: A23 @ A22)
    raises(DMShapeError, lambda: A23.matmul(A22))

    A = SDM({0: {0: ZZ(-1), 1: ZZ(1)}}, (1, 2), ZZ)
    B = SDM({0: {0: ZZ(-1)}, 1: {0: ZZ(-1)}}, (2, 1), ZZ)
    assert A.matmul(B) == A*B == SDM({}, (1, 1), ZZ)


def test_matmul_exraw():

    def dm(d):
        result = {}
        for i, row in d.items():
            row = {j:val for j, val in row.items() if val}
            if row:
                result[i] = row
        return SDM(result, (2, 2), EXRAW)

    values = [S.NegativeInfinity, S.NegativeOne, S.Zero, S.One, S.Infinity]
    for a, b, c, d in product(*[values]*4):
        Ad = dm({0: {0:a, 1:b}, 1: {0:c, 1:d}})
        Ad2 = dm({0: {0:a*a + b*c, 1:a*b + b*d}, 1:{0:c*a + d*c, 1: c*b + d*d}})
        assert Ad * Ad == Ad2


def test_SDM_add():
    A = SDM({0:{1:ZZ(1)}, 1:{0:ZZ(2), 1:ZZ(3)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(1)}, 1:{0:ZZ(-2), 1:ZZ(3)}}, (2, 2), ZZ)
    C = SDM({0:{0:ZZ(1), 1:ZZ(1)}, 1:{1:ZZ(6)}}, (2, 2), ZZ)
    assert A.add(B) == B.add(A) == A + B == B + A == C

    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(1)}, 1:{0:ZZ(-2), 1:ZZ(3)}}, (2, 2), ZZ)
    C = SDM({0:{0:ZZ(1), 1:ZZ(1)}, 1:{0:ZZ(-2), 1:ZZ(3)}}, (2, 2), ZZ)
    assert A.add(B) == B.add(A) == A + B == B + A == C

    raises(TypeError, lambda: A + [])


def test_SDM_sub():
    A = SDM({0:{1:ZZ(1)}, 1:{0:ZZ(2), 1:ZZ(3)}}, (2, 2), ZZ)
    B = SDM({0:{0:ZZ(1)}, 1:{0:ZZ(-2), 1:ZZ(3)}}, (2, 2), ZZ)
    C = SDM({0:{0:ZZ(-1), 1:ZZ(1)}, 1:{0:ZZ(4)}}, (2, 2), ZZ)
    assert A.sub(B) == A - B == C

    raises(TypeError, lambda: A - [])


def test_SDM_neg():
    A = SDM({0:{1:ZZ(1)}, 1:{0:ZZ(2), 1:ZZ(3)}}, (2, 2), ZZ)
    B = SDM({0:{1:ZZ(-1)}, 1:{0:ZZ(-2), 1:ZZ(-3)}}, (2, 2), ZZ)
    assert A.neg() == -A == B


def test_SDM_convert_to():
    A = SDM({0:{1:ZZ(1)}, 1:{0:ZZ(2), 1:ZZ(3)}}, (2, 2), ZZ)
    B = SDM({0:{1:QQ(1)}, 1:{0:QQ(2), 1:QQ(3)}}, (2, 2), QQ)
    C = A.convert_to(QQ)
    assert C == B
    assert C.domain == QQ

    D = A.convert_to(ZZ)
    assert D == A
    assert D.domain == ZZ


def test_SDM_hstack():
    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    B = SDM({1:{1:ZZ(1)}}, (2, 2), ZZ)
    AA = SDM({0:{1:ZZ(1), 3:ZZ(1)}}, (2, 4), ZZ)
    AB = SDM({0:{1:ZZ(1)}, 1:{3:ZZ(1)}}, (2, 4), ZZ)
    assert SDM.hstack(A) == A
    assert SDM.hstack(A, A) == AA
    assert SDM.hstack(A, B) == AB


def test_SDM_vstack():
    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    B = SDM({1:{1:ZZ(1)}}, (2, 2), ZZ)
    AA = SDM({0:{1:ZZ(1)}, 2:{1:ZZ(1)}}, (4, 2), ZZ)
    AB = SDM({0:{1:ZZ(1)}, 3:{1:ZZ(1)}}, (4, 2), ZZ)
    assert SDM.vstack(A) == A
    assert SDM.vstack(A, A) == AA
    assert SDM.vstack(A, B) == AB


def test_SDM_applyfunc():
    A = SDM({0:{1:ZZ(1)}}, (2, 2), ZZ)
    B = SDM({0:{1:ZZ(2)}}, (2, 2), ZZ)
    assert A.applyfunc(lambda x: 2*x, ZZ) == B


def test_SDM_inv():
    A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    B = SDM({0:{0:QQ(-2), 1:QQ(1)}, 1:{0:QQ(3, 2), 1:QQ(-1, 2)}}, (2, 2), QQ)
    assert A.inv() == B


def test_SDM_det():
    A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    assert A.det() == QQ(-2)


def test_SDM_lu():
    A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    L = SDM({0:{0:QQ(1)}, 1:{0:QQ(3), 1:QQ(1)}}, (2, 2), QQ)
    #U = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(-2)}}, (2, 2), QQ)
    #swaps = []
    # This doesn't quite work. U has some nonzero elements in the lower part.
    #assert A.lu() == (L, U, swaps)
    assert A.lu()[0] == L


def test_SDM_lu_solve():
    A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    b = SDM({0:{0:QQ(1)}, 1:{0:QQ(2)}}, (2, 1), QQ)
    x = SDM({1:{0:QQ(1, 2)}}, (2, 1), QQ)
    assert A.matmul(x) == b
    assert A.lu_solve(b) == x


def test_SDM_charpoly():
    A = SDM({0:{0:ZZ(1), 1:ZZ(2)}, 1:{0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    assert A.charpoly() == [ZZ(1), ZZ(-5), ZZ(-2)]


def test_SDM_nullspace():
    # More tests are in test_nullspace.py
    A = SDM({0:{0:QQ(1), 1:QQ(1)}}, (2, 2), QQ)
    assert A.nullspace()[0] == SDM({0:{0:QQ(-1), 1:QQ(1)}}, (1, 2), QQ)


def test_SDM_rref():
    # More tests are in test_rref.py

    A = SDM({0:{0:QQ(1), 1:QQ(2)},
             1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
    A_rref = SDM({0:{0:QQ(1)}, 1:{1:QQ(1)}}, (2, 2), QQ)
    assert A.rref() == (A_rref, [0, 1])

    A = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(2)},
             1: {0: QQ(3),           2: QQ(4)}}, (2, 3), ZZ)
    A_rref = SDM({0: {0: QQ(1,1), 2: QQ(4,3)},
                  1: {1: QQ(1,1), 2: QQ(1,3)}}, (2, 3), QQ)
    assert A.rref() == (A_rref, [0, 1])


def test_SDM_particular():
    A = SDM({0:{0:QQ(1)}}, (2, 2), QQ)
    Apart = SDM.zeros((1, 2), QQ)
    assert A.particular() == Apart


def test_SDM_is_zero_matrix():
    A = SDM({0: {0: QQ(1)}}, (2, 2), QQ)
    Azero = SDM.zeros((1, 2), QQ)
    assert A.is_zero_matrix() is False
    assert Azero.is_zero_matrix() is True


def test_SDM_is_upper():
    A = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)},
                       1: {1: QQ(5), 2: QQ(6), 3: QQ(7)},
                                 2: {2: QQ(8), 3: QQ(9)}}, (3, 4), QQ)
    B = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)},
                       1: {1: QQ(5), 2: QQ(6), 3: QQ(7)},
                       2: {1: QQ(7), 2: QQ(8), 3: QQ(9)}}, (3, 4), QQ)
    assert A.is_upper() is True
    assert B.is_upper() is False


def test_SDM_is_lower():
    A = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)},
                       1: {1: QQ(5), 2: QQ(6), 3: QQ(7)},
                                 2: {2: QQ(8), 3: QQ(9)}}, (3, 4), QQ
            ).transpose()
    B = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)},
                       1: {1: QQ(5), 2: QQ(6), 3: QQ(7)},
                       2: {1: QQ(7), 2: QQ(8), 3: QQ(9)}}, (3, 4), QQ
            ).transpose()
    assert A.is_lower() is True
    assert B.is_lower() is False
