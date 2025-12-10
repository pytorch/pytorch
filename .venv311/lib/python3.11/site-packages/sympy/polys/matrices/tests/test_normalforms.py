from sympy.testing.pytest import raises

from sympy.core.symbol import Symbol
from sympy.polys.matrices.normalforms import (
    invariant_factors,
    smith_normal_form,
    smith_normal_decomp,
    is_smith_normal_form,
    hermite_normal_form,
    _hermite_normal_form,
    _hermite_normal_form_modulo_D
)
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.matrices.exceptions import DMDomainError, DMShapeError


def test_is_smith_normal_form():

    snf_examples = [
        DM([[0, 0], [0, 0]], ZZ),
        DM([[1, 0], [0, 0]], ZZ),
        DM([[1, 0], [0, 1]], ZZ),
        DM([[1, 0], [0, 2]], ZZ),
    ]

    non_snf_examples = [
        DM([[0, 1], [0, 0]], ZZ),
        DM([[0, 0], [0, 1]], ZZ),
        DM([[2, 0], [0, 3]], ZZ),
    ]

    for m in snf_examples:
        assert is_smith_normal_form(m) is True

    for m in non_snf_examples:
        assert is_smith_normal_form(m) is False


def test_smith_normal():

    m = DM([
        [12, 6, 4, 8],
        [3, 9, 6, 12],
        [2, 16, 14, 28],
        [20, 10, 10, 20]], ZZ)

    smf = DM([
        [1, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 30, 0],
        [0, 0, 0, 0]], ZZ)

    s = DM([
        [0, 1, -1, 0],
        [1, -4, 0, 0],
        [0, -2, 3, 0],
        [-2, 2, -1, 1]], ZZ)

    t = DM([
        [1, 1, 10, 0],
        [0, -1, -2, 0],
        [0, 1, 3, -2],
        [0, 0, 0, 1]], ZZ)

    assert smith_normal_form(m).to_dense() == smf
    assert smith_normal_decomp(m) == (smf, s, t)
    assert is_smith_normal_form(smf)
    assert smf == s * m * t

    m00 = DomainMatrix.zeros((0, 0), ZZ).to_dense()
    m01 = DomainMatrix.zeros((0, 1), ZZ).to_dense()
    m10 = DomainMatrix.zeros((1, 0), ZZ).to_dense()
    i11 = DM([[1]], ZZ)

    assert smith_normal_form(m00) == m00.to_sparse()
    assert smith_normal_form(m01) == m01.to_sparse()
    assert smith_normal_form(m10) == m10.to_sparse()
    assert smith_normal_form(i11) == i11.to_sparse()

    assert smith_normal_decomp(m00) == (m00, m00, m00)
    assert smith_normal_decomp(m01) == (m01, m00, i11)
    assert smith_normal_decomp(m10) == (m10, i11, m00)
    assert smith_normal_decomp(i11) == (i11, i11, i11)

    x = Symbol('x')
    m = DM([[x-1,  1, -1],
            [  0,  x, -1],
            [  0, -1,  x]], QQ[x])
    dx = m.domain.gens[0]
    assert invariant_factors(m) == (1, dx-1, dx**2-1)

    zr = DomainMatrix([], (0, 2), ZZ)
    zc = DomainMatrix([[], []], (2, 0), ZZ)
    assert smith_normal_form(zr).to_dense() == zr
    assert smith_normal_form(zc).to_dense() == zc

    assert smith_normal_form(DM([[2, 4]], ZZ)).to_dense() == DM([[2, 0]], ZZ)
    assert smith_normal_form(DM([[0, -2]], ZZ)).to_dense() == DM([[2, 0]], ZZ)
    assert smith_normal_form(DM([[0], [-2]], ZZ)).to_dense() == DM([[2], [0]], ZZ)

    assert smith_normal_decomp(DM([[0, -2]], ZZ)) == (
        DM([[2, 0]], ZZ), DM([[-1]], ZZ), DM([[0, 1], [1, 0]], ZZ)
    )
    assert smith_normal_decomp(DM([[0], [-2]], ZZ)) == (
        DM([[2], [0]], ZZ), DM([[0, -1], [1, 0]], ZZ), DM([[1]], ZZ)
    )

    m =   DM([[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0]], ZZ)
    snf = DM([[1, 0, 0, 0], [0, 6, 0, 0], [0, 0, 0, 0]], ZZ)
    s = DM([[1, 0, 1], [2, 0, 3], [0, 1, 0]], ZZ)
    t = DM([[1, -2, 0, 0], [0, 0, 0, 1], [-1, 3, 0, 0], [0, 0, 1, 0]], ZZ)

    assert smith_normal_form(m).to_dense() == snf
    assert smith_normal_decomp(m) == (snf, s, t)
    assert is_smith_normal_form(snf)
    assert snf == s * m * t

    raises(ValueError, lambda: smith_normal_form(DM([[1]], ZZ[x])))


def test_hermite_normal():
    m = DM([[2, 7, 17, 29, 41], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]], ZZ)
    hnf = DM([[1, 0, 0], [0, 2, 1], [0, 0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf
    assert hermite_normal_form(m, D=ZZ(2)) == hnf
    assert hermite_normal_form(m, D=ZZ(2), check_rank=True) == hnf

    m = m.transpose()
    hnf = DM([[37, 0, 19], [222, -6, 113], [48, 0, 25], [0, 2, 1], [0, 0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf
    raises(DMShapeError, lambda: _hermite_normal_form_modulo_D(m, ZZ(96)))
    raises(DMDomainError, lambda: _hermite_normal_form_modulo_D(m, QQ(96)))

    m = DM([[8, 28, 68, 116, 164], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]], ZZ)
    hnf = DM([[4, 0, 0], [0, 2, 1], [0, 0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf
    assert hermite_normal_form(m, D=ZZ(8)) == hnf
    assert hermite_normal_form(m, D=ZZ(8), check_rank=True) == hnf

    m = DM([[10, 8, 6, 30, 2], [45, 36, 27, 18, 9], [5, 4, 3, 2, 1]], ZZ)
    hnf = DM([[26, 2], [0, 9], [0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf

    m = DM([[2, 7], [0, 0], [0, 0]], ZZ)
    hnf = DM([[1], [0], [0]], ZZ)
    assert hermite_normal_form(m) == hnf

    m = DM([[-2, 1], [0, 1]], ZZ)
    hnf = DM([[2, 1], [0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf

    m = DomainMatrix([[QQ(1)]], (1, 1), QQ)
    raises(DMDomainError, lambda: hermite_normal_form(m))
    raises(DMDomainError, lambda: _hermite_normal_form(m))
    raises(DMDomainError, lambda: _hermite_normal_form_modulo_D(m, ZZ(1)))
