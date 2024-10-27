"""Tests for sparse distributed modules. """

from sympy.polys.distributedmodules import (
    sdm_monomial_mul, sdm_monomial_deg, sdm_monomial_divides,
    sdm_add, sdm_LM, sdm_LT, sdm_mul_term, sdm_zero, sdm_deg,
    sdm_LC, sdm_from_dict,
    sdm_spoly, sdm_ecart, sdm_nf_mora, sdm_groebner,
    sdm_from_vector, sdm_to_vector, sdm_monomial_lcm
)

from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ

from sympy.abc import x, y, z


def test_sdm_monomial_mul():
    assert sdm_monomial_mul((1, 1, 0), (1, 3)) == (1, 2, 3)


def test_sdm_monomial_deg():
    assert sdm_monomial_deg((5, 2, 1)) == 3


def test_sdm_monomial_lcm():
    assert sdm_monomial_lcm((1, 2, 3), (1, 5, 0)) == (1, 5, 3)


def test_sdm_monomial_divides():
    assert sdm_monomial_divides((1, 0, 0), (1, 0, 0)) is True
    assert sdm_monomial_divides((1, 0, 0), (1, 2, 1)) is True
    assert sdm_monomial_divides((5, 1, 1), (5, 2, 1)) is True

    assert sdm_monomial_divides((1, 0, 0), (2, 0, 0)) is False
    assert sdm_monomial_divides((1, 1, 0), (1, 0, 0)) is False
    assert sdm_monomial_divides((5, 1, 2), (5, 0, 1)) is False


def test_sdm_LC():
    assert sdm_LC([((1, 2, 3), QQ(5))], QQ) == QQ(5)


def test_sdm_from_dict():
    dic = {(1, 2, 1, 1): QQ(1), (1, 1, 2, 1): QQ(1), (1, 0, 2, 1): QQ(1),
           (1, 0, 0, 3): QQ(1), (1, 1, 1, 0): QQ(1)}
    assert sdm_from_dict(dic, grlex) == \
        [((1, 2, 1, 1), QQ(1)), ((1, 1, 2, 1), QQ(1)),
         ((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1))]

# TODO test to_dict?


def test_sdm_add():
    assert sdm_add([((1, 1, 1), QQ(1))], [((2, 0, 0), QQ(1))], lex, QQ) == \
        [((2, 0, 0), QQ(1)), ((1, 1, 1), QQ(1))]
    assert sdm_add([((1, 1, 1), QQ(1))], [((1, 1, 1), QQ(-1))], lex, QQ) == []
    assert sdm_add([((1, 0, 0), QQ(1))], [((1, 0, 0), QQ(2))], lex, QQ) == \
        [((1, 0, 0), QQ(3))]
    assert sdm_add([((1, 0, 1), QQ(1))], [((1, 1, 0), QQ(1))], lex, QQ) == \
        [((1, 1, 0), QQ(1)), ((1, 0, 1), QQ(1))]


def test_sdm_LM():
    dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(1), (4, 0, 1): QQ(1)}
    assert sdm_LM(sdm_from_dict(dic, lex)) == (4, 0, 1)


def test_sdm_LT():
    dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(2), (4, 0, 1): QQ(3)}
    assert sdm_LT(sdm_from_dict(dic, lex)) == ((4, 0, 1), QQ(3))


def test_sdm_mul_term():
    assert sdm_mul_term([((1, 0, 0), QQ(1))], ((0, 0), QQ(0)), lex, QQ) == []
    assert sdm_mul_term([], ((1, 0), QQ(1)), lex, QQ) == []
    assert sdm_mul_term([((1, 0, 0), QQ(1))], ((1, 0), QQ(1)), lex, QQ) == \
        [((1, 1, 0), QQ(1))]
    f = [((2, 0, 1), QQ(4)), ((1, 1, 0), QQ(3))]
    assert sdm_mul_term(f, ((1, 1), QQ(2)), lex, QQ) == \
        [((2, 1, 2), QQ(8)), ((1, 2, 1), QQ(6))]


def test_sdm_zero():
    assert sdm_zero() == []


def test_sdm_deg():
    assert sdm_deg([((1, 2, 3), 1), ((10, 0, 1), 1), ((2, 3, 4), 4)]) == 7


def test_sdm_spoly():
    f = [((2, 1, 1), QQ(1)), ((1, 0, 1), QQ(1))]
    g = [((2, 3, 0), QQ(1))]
    h = [((1, 2, 3), QQ(1))]
    assert sdm_spoly(f, h, lex, QQ) == []
    assert sdm_spoly(f, g, lex, QQ) == [((1, 2, 1), QQ(1))]


def test_sdm_ecart():
    assert sdm_ecart([((1, 2, 3), 1), ((1, 0, 1), 1)]) == 0
    assert sdm_ecart([((2, 2, 1), 1), ((1, 5, 1), 1)]) == 3


def test_sdm_nf_mora():
    f = sdm_from_dict({(1, 2, 1, 1): QQ(1), (1, 1, 2, 1): QQ(1),
                (1, 0, 2, 1): QQ(1), (1, 0, 0, 3): QQ(1), (1, 1, 1, 0): QQ(1)},
        grlex)
    f1 = sdm_from_dict({(1, 1, 1, 0): QQ(1), (1, 0, 2, 0): QQ(1),
                        (1, 0, 0, 0): QQ(-1)}, grlex)
    f2 = sdm_from_dict({(1, 1, 1, 0): QQ(1)}, grlex)
    (id0, id1, id2) = [sdm_from_dict({(i, 0, 0, 0): QQ(1)}, grlex)
                       for i in range(3)]

    assert sdm_nf_mora(f, [f1, f2], grlex, QQ, phantom=(id0, [id1, id2])) == \
        ([((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1)),
          ((1, 1, 0, 1), QQ(1))],
         [((1, 1, 0, 1), QQ(-1)), ((0, 0, 0, 0), QQ(1))])
    assert sdm_nf_mora(f, [f2, f1], grlex, QQ, phantom=(id0, [id2, id1])) == \
        ([((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1))],
         [((2, 1, 0, 1), QQ(-1)), ((2, 0, 1, 1), QQ(-1)), ((0, 0, 0, 0), QQ(1))])

    f = sdm_from_vector([x*z, y**2 + y*z - z, y], lex, QQ, gens=[x, y, z])
    f1 = sdm_from_vector([x, y, 1], lex, QQ, gens=[x, y, z])
    f2 = sdm_from_vector([x*y, z, z**2], lex, QQ, gens=[x, y, z])
    assert sdm_nf_mora(f, [f1, f2], lex, QQ) == \
        sdm_nf_mora(f, [f2, f1], lex, QQ) == \
        [((1, 0, 1, 1), QQ(1)), ((1, 0, 0, 1), QQ(-1)), ((0, 1, 1, 0), QQ(-1)),
         ((0, 1, 0, 1), QQ(1))]


def test_conversion():
    f = [x**2 + y**2, 2*z]
    g = [((1, 0, 0, 1), QQ(2)), ((0, 2, 0, 0), QQ(1)), ((0, 0, 2, 0), QQ(1))]
    assert sdm_to_vector(g, [x, y, z], QQ) == f
    assert sdm_from_vector(f, lex, QQ) == g
    assert sdm_from_vector(
        [x, 1], lex, QQ) == [((1, 0), QQ(1)), ((0, 1), QQ(1))]
    assert sdm_to_vector([((1, 1, 0, 0), 1)], [x, y, z], QQ, n=3) == [0, x, 0]
    assert sdm_from_vector([0, 0], lex, QQ, gens=[x, y]) == sdm_zero()


def test_nontrivial():
    gens = [x, y, z]

    def contains(I, f):
        S = [sdm_from_vector([g], lex, QQ, gens=gens) for g in I]
        G = sdm_groebner(S, sdm_nf_mora, lex, QQ)
        return sdm_nf_mora(sdm_from_vector([f], lex, QQ, gens=gens),
                           G, lex, QQ) == sdm_zero()

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**3)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4 + y**3 + 2*z*y*x)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y*z)
    assert contains([x, 1 + x + y, 5 - 7*y], 1)
    assert contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**3)
    assert not contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**2 + y**2)

    # compare local order
    assert not contains([x*(1 + x + y), y*(1 + z)], x)
    assert not contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_local():
    igrlex = InverseOrder(grlex)
    gens = [x, y, z]

    def contains(I, f):
        S = [sdm_from_vector([g], igrlex, QQ, gens=gens) for g in I]
        G = sdm_groebner(S, sdm_nf_mora, igrlex, QQ)
        return sdm_nf_mora(sdm_from_vector([f], lex, QQ, gens=gens),
                           G, lex, QQ) == sdm_zero()
    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_uncovered_line():
    gens = [x, y]
    f1 = sdm_zero()
    f2 = sdm_from_vector([x, 0], lex, QQ, gens=gens)
    f3 = sdm_from_vector([0, y], lex, QQ, gens=gens)

    assert sdm_spoly(f1, f2, lex, QQ) == sdm_zero()
    assert sdm_spoly(f3, f2, lex, QQ) == sdm_zero()


def test_chain_criterion():
    gens = [x]
    f1 = sdm_from_vector([1, x], grlex, QQ, gens=gens)
    f2 = sdm_from_vector([0, x - 2], grlex, QQ, gens=gens)
    assert len(sdm_groebner([f1, f2], sdm_nf_mora, grlex, QQ)) == 2
