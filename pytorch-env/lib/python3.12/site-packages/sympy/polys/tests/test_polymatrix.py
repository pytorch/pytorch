from sympy.testing.pytest import raises

from sympy.polys.polymatrix import PolyMatrix
from sympy.polys import Poly

from sympy.core.singleton import S
from sympy.matrices.dense import Matrix
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ

from sympy.abc import x, y


def _test_polymatrix():
    pm1 = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(x**3, x), Poly(-1 + x, x)]])
    v1 = PolyMatrix([[1, 0], [-1, 0]], ring='ZZ[x]')
    m1 = PolyMatrix([[1, 0], [-1, 0]], ring='ZZ[x]')
    A = PolyMatrix([[Poly(x**2 + x, x), Poly(0, x)], \
                    [Poly(x**3 - x + 1, x), Poly(0, x)]])
    B = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(-x**2, x), Poly(x, x)]])
    assert A.ring == ZZ[x]
    assert isinstance(pm1*v1, PolyMatrix)
    assert pm1*v1 == A
    assert pm1*m1 == A
    assert v1*pm1 == B

    pm2 = PolyMatrix([[Poly(x**2, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**2, x, domain='QQ'), \
                    Poly(x**3, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**3, x, domain='QQ')]])
    assert pm2.ring == QQ[x]
    v2 = PolyMatrix([1, 0, 0, 0, 0, 0], ring='ZZ[x]')
    m2 = PolyMatrix([1, 0, 0, 0, 0, 0], ring='ZZ[x]')
    C = PolyMatrix([[Poly(x**2, x, domain='QQ')]])
    assert pm2*v2 == C
    assert pm2*m2 == C

    pm3 = PolyMatrix([[Poly(x**2, x), S.One]], ring='ZZ[x]')
    v3 = S.Half*pm3
    assert v3 == PolyMatrix([[Poly(S.Half*x**2, x, domain='QQ'), S.Half]], ring='QQ[x]')
    assert pm3*S.Half == v3
    assert v3.ring == QQ[x]

    pm4 = PolyMatrix([[Poly(x**2, x, domain='ZZ'), Poly(-x**2, x, domain='ZZ')]])
    v4 = PolyMatrix([1, -1], ring='ZZ[x]')
    assert pm4*v4 == PolyMatrix([[Poly(2*x**2, x, domain='ZZ')]])

    assert len(PolyMatrix(ring=ZZ[x])) == 0
    assert PolyMatrix([1, 0, 0, 1], x)/(-1) == PolyMatrix([-1, 0, 0, -1], x)


def test_polymatrix_constructor():
    M1 = PolyMatrix([[x, y]], ring=QQ[x,y])
    assert M1.ring == QQ[x,y]
    assert M1.domain == QQ
    assert M1.gens == (x, y)
    assert M1.shape == (1, 2)
    assert M1.rows == 1
    assert M1.cols == 2
    assert len(M1) == 2
    assert list(M1) == [Poly(x, (x, y), domain=QQ), Poly(y, (x, y), domain=QQ)]

    M2 = PolyMatrix([[x, y]], ring=QQ[x][y])
    assert M2.ring == QQ[x][y]
    assert M2.domain == QQ[x]
    assert M2.gens == (y,)
    assert M2.shape == (1, 2)
    assert M2.rows == 1
    assert M2.cols == 2
    assert len(M2) == 2
    assert list(M2) == [Poly(x, (y,), domain=QQ[x]), Poly(y, (y,), domain=QQ[x])]

    assert PolyMatrix([[x, y]], y) == PolyMatrix([[x, y]], ring=ZZ.frac_field(x)[y])
    assert PolyMatrix([[x, y]], ring='ZZ[x,y]') == PolyMatrix([[x, y]], ring=ZZ[x,y])

    assert PolyMatrix([[x, y]], (x, y)) == PolyMatrix([[x, y]], ring=QQ[x,y])
    assert PolyMatrix([[x, y]], x, y) == PolyMatrix([[x, y]], ring=QQ[x,y])
    assert PolyMatrix([x, y]) == PolyMatrix([[x], [y]], ring=QQ[x,y])
    assert PolyMatrix(1, 2, [x, y]) == PolyMatrix([[x, y]], ring=QQ[x,y])
    assert PolyMatrix(1, 2, lambda i,j: [x,y][j]) == PolyMatrix([[x, y]], ring=QQ[x,y])
    assert PolyMatrix(0, 2, [], x, y).shape == (0, 2)
    assert PolyMatrix(2, 0, [], x, y).shape == (2, 0)
    assert PolyMatrix([[], []], x, y).shape == (2, 0)
    assert PolyMatrix(ring=QQ[x,y]) == PolyMatrix(0, 0, [], ring=QQ[x,y]) == PolyMatrix([], ring=QQ[x,y])
    raises(TypeError, lambda: PolyMatrix())
    raises(TypeError, lambda: PolyMatrix(1))

    assert PolyMatrix([Poly(x), Poly(y)]) == PolyMatrix([[x], [y]], ring=ZZ[x,y])

    # XXX: Maybe a bug in parallel_poly_from_expr (x lost from gens and domain):
    assert PolyMatrix([Poly(y, x), 1]) == PolyMatrix([[y], [1]], ring=QQ[y])


def test_polymatrix_eq():
    assert (PolyMatrix([x]) == PolyMatrix([x])) is True
    assert (PolyMatrix([y]) == PolyMatrix([x])) is False
    assert (PolyMatrix([x]) != PolyMatrix([x])) is False
    assert (PolyMatrix([y]) != PolyMatrix([x])) is True

    assert PolyMatrix([[x, y]]) != PolyMatrix([x, y]) == PolyMatrix([[x], [y]])

    assert PolyMatrix([x], ring=QQ[x]) != PolyMatrix([x], ring=ZZ[x])

    assert PolyMatrix([x]) != Matrix([x])
    assert PolyMatrix([x]).to_Matrix() == Matrix([x])

    assert PolyMatrix([1], x) == PolyMatrix([1], x)
    assert PolyMatrix([1], x) != PolyMatrix([1], y)


def test_polymatrix_from_Matrix():
    assert PolyMatrix.from_Matrix(Matrix([1, 2]), x) == PolyMatrix([1, 2], x, ring=QQ[x])
    assert PolyMatrix.from_Matrix(Matrix([1]), ring=QQ[x]) == PolyMatrix([1], x)
    pmx = PolyMatrix([1, 2], x)
    pmy = PolyMatrix([1, 2], y)
    assert pmx != pmy
    assert pmx.set_gens(y) == pmy


def test_polymatrix_repr():
    assert repr(PolyMatrix([[1, 2]], x)) == 'PolyMatrix([[1, 2]], ring=QQ[x])'
    assert repr(PolyMatrix(0, 2, [], x)) == 'PolyMatrix(0, 2, [], ring=QQ[x])'


def test_polymatrix_getitem():
    M = PolyMatrix([[1, 2], [3, 4]], x)
    assert M[:, :] == M
    assert M[0, :] == PolyMatrix([[1, 2]], x)
    assert M[:, 0] == PolyMatrix([1, 3], x)
    assert M[0, 0] == Poly(1, x, domain=QQ)
    assert M[0] == Poly(1, x, domain=QQ)
    assert M[:2] == [Poly(1, x, domain=QQ), Poly(2, x, domain=QQ)]


def test_polymatrix_arithmetic():
    M = PolyMatrix([[1, 2], [3, 4]], x)
    assert M + M == PolyMatrix([[2, 4], [6, 8]], x)
    assert M - M == PolyMatrix([[0, 0], [0, 0]], x)
    assert -M == PolyMatrix([[-1, -2], [-3, -4]], x)
    raises(TypeError, lambda: M + 1)
    raises(TypeError, lambda: M - 1)
    raises(TypeError, lambda: 1 + M)
    raises(TypeError, lambda: 1 - M)

    assert M * M == PolyMatrix([[7, 10], [15, 22]], x)
    assert 2 * M == PolyMatrix([[2, 4], [6, 8]], x)
    assert M * 2 == PolyMatrix([[2, 4], [6, 8]], x)
    assert S(2) * M == PolyMatrix([[2, 4], [6, 8]], x)
    assert M * S(2) == PolyMatrix([[2, 4], [6, 8]], x)
    raises(TypeError, lambda: [] * M)
    raises(TypeError, lambda: M * [])
    M2 = PolyMatrix([[1, 2]], ring=ZZ[x])
    assert S.Half * M2 == PolyMatrix([[S.Half, 1]], ring=QQ[x])
    assert M2 * S.Half == PolyMatrix([[S.Half, 1]], ring=QQ[x])

    assert M / 2 == PolyMatrix([[S(1)/2, 1], [S(3)/2, 2]], x)
    assert M / Poly(2, x) == PolyMatrix([[S(1)/2, 1], [S(3)/2, 2]], x)
    raises(TypeError, lambda: M / [])


def test_polymatrix_manipulations():
    M1 = PolyMatrix([[1, 2], [3, 4]], x)
    assert M1.transpose() == PolyMatrix([[1, 3], [2, 4]], x)
    M2 = PolyMatrix([[5, 6], [7, 8]], x)
    assert M1.row_join(M2) == PolyMatrix([[1, 2, 5, 6], [3, 4, 7, 8]], x)
    assert M1.col_join(M2) == PolyMatrix([[1, 2], [3, 4], [5, 6], [7, 8]], x)
    assert M1.applyfunc(lambda e: 2*e) == PolyMatrix([[2, 4], [6, 8]], x)


def test_polymatrix_ones_zeros():
    assert PolyMatrix.zeros(1, 2, x) == PolyMatrix([[0, 0]], x)
    assert PolyMatrix.eye(2, x) == PolyMatrix([[1, 0], [0, 1]], x)


def test_polymatrix_rref():
    M = PolyMatrix([[1, 2], [3, 4]], x)
    assert M.rref() == (PolyMatrix.eye(2, x), (0, 1))
    raises(ValueError, lambda: PolyMatrix([1, 2], ring=ZZ[x]).rref())
    raises(ValueError, lambda: PolyMatrix([1, x], ring=QQ[x]).rref())


def test_polymatrix_nullspace():
    M = PolyMatrix([[1, 2], [3, 6]], x)
    assert M.nullspace() == [PolyMatrix([-2, 1], x)]
    raises(ValueError, lambda: PolyMatrix([1, 2], ring=ZZ[x]).nullspace())
    raises(ValueError, lambda: PolyMatrix([1, x], ring=QQ[x]).nullspace())
    assert M.rank() == 1
