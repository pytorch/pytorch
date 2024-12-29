from sympy.matrices.sparsetools import _doktocsr, _csrtodok, banded
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices import SparseMatrix
from sympy.testing.pytest import raises


def test_doktocsr():
    a = SparseMatrix([[1, 2, 0, 0], [0, 3, 9, 0], [0, 1, 4, 0]])
    b = SparseMatrix(4, 6, [10, 20, 0, 0, 0, 0, 0, 30, 0, 40, 0, 0, 0, 0, 50,
        60, 70, 0, 0, 0, 0, 0, 0, 80])
    c = SparseMatrix(4, 4, [0, 0, 0, 0, 0, 12, 0, 2, 15, 0, 12, 0, 0, 0, 0, 4])
    d = SparseMatrix(10, 10, {(1, 1): 12, (3, 5): 7, (7, 8): 12})
    e = SparseMatrix([[0, 0, 0], [1, 0, 2], [3, 0, 0]])
    f = SparseMatrix(7, 8, {(2, 3): 5, (4, 5):12})
    assert _doktocsr(a) == [[1, 2, 3, 9, 1, 4], [0, 1, 1, 2, 1, 2],
        [0, 2, 4, 6], [3, 4]]
    assert _doktocsr(b) == [[10, 20, 30, 40, 50, 60, 70, 80],
        [0, 1, 1, 3, 2, 3, 4, 5], [0, 2, 4, 7, 8], [4, 6]]
    assert _doktocsr(c) == [[12, 2, 15, 12, 4], [1, 3, 0, 2, 3],
        [0, 0, 2, 4, 5], [4, 4]]
    assert _doktocsr(d) == [[12, 7, 12], [1, 5, 8],
        [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3], [10, 10]]
    assert _doktocsr(e) == [[1, 2, 3], [0, 2, 0], [0, 0, 2, 3], [3, 3]]
    assert _doktocsr(f) == [[5, 12], [3, 5], [0, 0, 0, 1, 1, 2, 2, 2], [7, 8]]


def test_csrtodok():
    h = [[5, 7, 5], [2, 1, 3], [0, 1, 1, 3], [3, 4]]
    g = [[12, 5, 4], [2, 4, 2], [0, 1, 2, 3], [3, 7]]
    i = [[1, 3, 12], [0, 2, 4], [0, 2, 3], [2, 5]]
    j = [[11, 15, 12, 15], [2, 4, 1, 2], [0, 1, 1, 2, 3, 4], [5, 8]]
    k = [[1, 3], [2, 1], [0, 1, 1, 2], [3, 3]]
    m = _csrtodok(h)
    assert isinstance(m, SparseMatrix)
    assert m == SparseMatrix(3, 4,
        {(0, 2): 5, (2, 1): 7, (2, 3): 5})
    assert _csrtodok(g) == SparseMatrix(3, 7,
        {(0, 2): 12, (1, 4): 5, (2, 2): 4})
    assert _csrtodok(i) == SparseMatrix([[1, 0, 3, 0, 0], [0, 0, 0, 0, 12]])
    assert _csrtodok(j) == SparseMatrix(5, 8,
        {(0, 2): 11, (2, 4): 15, (3, 1): 12, (4, 2): 15})
    assert _csrtodok(k) == SparseMatrix(3, 3, {(0, 2): 1, (2, 1): 3})


def test_banded():
    raises(TypeError, lambda: banded())
    raises(TypeError, lambda: banded(1))
    raises(TypeError, lambda: banded(1, 2))
    raises(TypeError, lambda: banded(1, 2, 3))
    raises(TypeError, lambda: banded(1, 2, 3, 4))
    raises(ValueError, lambda: banded({0: (1, 2)}, rows=1))
    raises(ValueError, lambda: banded({0: (1, 2)}, cols=1))
    raises(ValueError, lambda: banded(1, {0: (1, 2)}))
    raises(ValueError, lambda: banded(2, 1, {0: (1, 2)}))
    raises(ValueError, lambda: banded(1, 2, {0: (1, 2)}))

    assert isinstance(banded(2, 4, {}), SparseMatrix)
    assert banded(2, 4, {}) == zeros(2, 4)
    assert banded({0: 0, 1: 0}) == zeros(0)
    assert banded({0: Matrix([1, 2])}) == Matrix([1, 2])
    assert banded({1: [1, 2, 3, 0], -1: [4, 5, 6]}) == \
        banded({1: (1, 2, 3), -1: (4, 5, 6)}) == \
        Matrix([
        [0, 1, 0, 0],
        [4, 0, 2, 0],
        [0, 5, 0, 3],
        [0, 0, 6, 0]])
    assert banded(3, 4, {-1: 1, 0: 2, 1: 3}) == \
        Matrix([
        [2, 3, 0, 0],
        [1, 2, 3, 0],
        [0, 1, 2, 3]])
    s = lambda d: (1 + d)**2
    assert banded(5, {0: s, 2: s}) == \
        Matrix([
        [1, 0, 1,  0,  0],
        [0, 4, 0,  4,  0],
        [0, 0, 9,  0,  9],
        [0, 0, 0, 16,  0],
        [0, 0, 0,  0, 25]])
    assert banded(2, {0: 1}) == \
        Matrix([
        [1, 0],
        [0, 1]])
    assert banded(2, 3, {0: 1}) == \
        Matrix([
        [1, 0, 0],
        [0, 1, 0]])
    vert = Matrix([1, 2, 3])
    assert banded({0: vert}, cols=3) == \
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [3, 2, 1],
        [0, 3, 2],
        [0, 0, 3]])
    assert banded(4, {0: ones(2)}) == \
        Matrix([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]])
    raises(ValueError, lambda: banded({0: 2, 1: ones(2)}, rows=5))
    assert banded({0: 2, 2: (ones(2),)*3}) == \
        Matrix([
        [2, 0, 1, 1, 0, 0, 0, 0],
        [0, 2, 1, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 1, 1, 0, 0],
        [0, 0, 0, 2, 1, 1, 0, 0],
        [0, 0, 0, 0, 2, 0, 1, 1],
        [0, 0, 0, 0, 0, 2, 1, 1]])
    raises(ValueError, lambda: banded({0: (2,)*5, 1: (ones(2),)*3}))
    u2 = Matrix([[1, 1], [0, 1]])
    assert banded({0: (2,)*5, 1: (u2,)*3}) == \
        Matrix([
        [2, 1, 1, 0, 0, 0, 0],
        [0, 2, 1, 0, 0, 0, 0],
        [0, 0, 2, 1, 1, 0, 0],
        [0, 0, 0, 2, 1, 0, 0],
        [0, 0, 0, 0, 2, 1, 1],
        [0, 0, 0, 0, 0, 0, 1]])
    assert banded({0:(0, ones(2)), 2: 2}) == \
        Matrix([
        [0, 0, 2],
        [0, 1, 1],
        [0, 1, 1]])
    raises(ValueError, lambda: banded({0: (0, ones(2)), 1: 2}))
    assert banded({0: 1}, cols=3) == banded({0: 1}, rows=3) == eye(3)
    assert banded({1: 1}, rows=3) == Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]])
