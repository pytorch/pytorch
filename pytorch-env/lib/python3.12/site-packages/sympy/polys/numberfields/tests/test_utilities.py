from sympy.abc import x
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.utilities import (
    AlgIntPowers, coeff_search, extract_fundamental_discriminant,
    isolate, supplement_a_subspace,
)
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.testing.pytest import raises


def test_AlgIntPowers_01():
    T = Poly(cyclotomic_poly(5))
    zeta_pow = AlgIntPowers(T)
    raises(ValueError, lambda: zeta_pow[-1])
    for e in range(10):
        a = e % 5
        if a < 4:
            c = zeta_pow[e]
            assert c[a] == 1 and all(c[i] == 0 for i in range(4) if i != a)
        else:
            assert zeta_pow[e] == [-1] * 4


def test_AlgIntPowers_02():
    T = Poly(x**3 + 2*x**2 + 3*x + 4)
    m = 7
    theta_pow = AlgIntPowers(T, m)
    for e in range(10):
        computed = theta_pow[e]
        coeffs = (Poly(x)**e % T + Poly(x**3)).rep.to_list()[1:]
        expected = [c % m for c in reversed(coeffs)]
        assert computed == expected


def test_coeff_search():
    C = []
    search = coeff_search(2, 1)
    for i, c in enumerate(search):
        C.append(c)
        if i == 12:
            break
    assert C == [[1, 1], [1, 0], [1, -1], [0, 1], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2], [1, 2], [1, -2], [0, 2], [3, 3]]


def test_extract_fundamental_discriminant():
    # To extract, integer must be 0 or 1 mod 4.
    raises(ValueError, lambda: extract_fundamental_discriminant(2))
    raises(ValueError, lambda: extract_fundamental_discriminant(3))
    # Try many cases, of different forms:
    cases = (
        (0, {}, {0: 1}),
        (1, {}, {}),
        (8, {2: 3}, {}),
        (-8, {2: 3, -1: 1}, {}),
        (12, {2: 2, 3: 1}, {}),
        (36, {}, {2: 1, 3: 1}),
        (45, {5: 1}, {3: 1}),
        (48, {2: 2, 3: 1}, {2: 1}),
        (1125, {5: 1}, {3: 1, 5: 1}),
    )
    for a, D_expected, F_expected in cases:
        D, F = extract_fundamental_discriminant(a)
        assert D == D_expected
        assert F == F_expected


def test_supplement_a_subspace_1():
    M = DM([[1, 7, 0], [2, 3, 4]], QQ).transpose()

    # First supplement over QQ:
    B = supplement_a_subspace(M)
    assert B[:, :2] == M
    assert B[:, 2] == DomainMatrix.eye(3, QQ).to_dense()[:, 0]

    # Now supplement over FF(7):
    M = M.convert_to(FF(7))
    B = supplement_a_subspace(M)
    assert B[:, :2] == M
    # When we work mod 7, first col of M goes to [1, 0, 0],
    # so the supplementary vector cannot equal this, as it did
    # when we worked over QQ. Instead, we get the second std basis vector:
    assert B[:, 2] == DomainMatrix.eye(3, FF(7)).to_dense()[:, 1]


def test_supplement_a_subspace_2():
    M = DM([[1, 0, 0], [2, 0, 0]], QQ).transpose()
    with raises(DMRankError):
        supplement_a_subspace(M)


def test_IntervalPrinter():
    ip = IntervalPrinter()
    assert ip.doprint(x**Rational(1, 3)) == "x**(mpi('1/3'))"
    assert ip.doprint(sqrt(x)) == "x**(mpi('1/2'))"


def test_isolate():
    assert isolate(1) == (1, 1)
    assert isolate(S.Half) == (S.Half, S.Half)

    assert isolate(sqrt(2)) == (1, 2)
    assert isolate(-sqrt(2)) == (-2, -1)

    assert isolate(sqrt(2), eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    assert isolate(-sqrt(2), eps=Rational(1, 100)) == (Rational(-17, 12), Rational(-24, 17))

    raises(NotImplementedError, lambda: isolate(I))
