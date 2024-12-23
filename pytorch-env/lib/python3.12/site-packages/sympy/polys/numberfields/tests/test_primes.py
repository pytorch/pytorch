from math import prod

from sympy import QQ, ZZ
from sympy.abc import x, theta
from sympy.ntheory import factorint
from sympy.ntheory.residue_ntheory import n_order
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.matrices import DomainMatrix
from sympy.polys.numberfields.basis import round_two
from sympy.polys.numberfields.exceptions import StructureError
from sympy.polys.numberfields.modules import PowerBasis, to_col
from sympy.polys.numberfields.primes import (
    prime_decomp, _two_elt_rep,
    _check_formal_conditions_for_maximal_order,
)
from sympy.testing.pytest import raises


def test_check_formal_conditions_for_maximal_order():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = A.submodule_from_matrix(DomainMatrix.eye(4, ZZ)[:, :-1])
    # Is a direct submodule of a power basis, but lacks 1 as first generator:
    raises(StructureError, lambda: _check_formal_conditions_for_maximal_order(B))
    # Is not a direct submodule of a power basis:
    raises(StructureError, lambda: _check_formal_conditions_for_maximal_order(C))
    # Is direct submod of pow basis, and starts with 1, but not sq/max rank/HNF:
    raises(StructureError, lambda: _check_formal_conditions_for_maximal_order(D))


def test_two_elt_rep():
    ell = 7
    T = Poly(cyclotomic_poly(ell))
    ZK, dK = round_two(T)
    for p in [29, 13, 11, 5]:
        P = prime_decomp(p, T)
        for Pi in P:
            # We have Pi in two-element representation, and, because we are
            # looking at a cyclotomic field, this was computed by the "easy"
            # method that just factors T mod p. We will now convert this to
            # a set of Z-generators, then convert that back into a two-element
            # rep. The latter need not be identical to the two-elt rep we
            # already have, but it must have the same HNF.
            H = p*ZK + Pi.alpha*ZK
            gens = H.basis_element_pullbacks()
            # Note: we could supply f = Pi.f, but prefer to test behavior without it.
            b = _two_elt_rep(gens, ZK, p)
            if b != Pi.alpha:
                H2 = p*ZK + b*ZK
                assert H2 == H


def test_valuation_at_prime_ideal():
    p = 7
    T = Poly(cyclotomic_poly(p))
    ZK, dK = round_two(T)
    P = prime_decomp(p, T, dK=dK, ZK=ZK)
    assert len(P) == 1
    P0 = P[0]
    v = P0.valuation(p*ZK)
    assert v == P0.e
    # Test easy 0 case:
    assert P0.valuation(5*ZK) == 0


def test_decomp_1():
    # All prime decompositions in cyclotomic fields are in the "easy case,"
    # since the index is unity.
    # Here we check the ramified prime.
    T = Poly(cyclotomic_poly(7))
    raises(ValueError, lambda: prime_decomp(7))
    P = prime_decomp(7, T)
    assert len(P) == 1
    P0 = P[0]
    assert P0.e == 6
    assert P0.f == 1
    # Test powers:
    assert P0**0 == P0.ZK
    assert P0**1 == P0
    assert P0**6 == 7 * P0.ZK


def test_decomp_2():
    # More easy cyclotomic cases, but here we check unramified primes.
    ell = 7
    T = Poly(cyclotomic_poly(ell))
    for p in [29, 13, 11, 5]:
        f_exp = n_order(p, ell)
        g_exp = (ell - 1) // f_exp
        P = prime_decomp(p, T)
        assert len(P) == g_exp
        for Pi in P:
            assert Pi.e == 1
            assert Pi.f == f_exp


def test_decomp_3():
    T = Poly(x ** 2 - 35)
    rad = {}
    ZK, dK = round_two(T, radicals=rad)
    # 35 is 3 mod 4, so field disc is 4*5*7, and theory says each of the
    # rational primes 2, 5, 7 should be the square of a prime ideal.
    for p in [2, 5, 7]:
        P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=rad.get(p))
        assert len(P) == 1
        assert P[0].e == 2
        assert P[0]**2 == p*ZK


def test_decomp_4():
    T = Poly(x ** 2 - 21)
    rad = {}
    ZK, dK = round_two(T, radicals=rad)
    # 21 is 1 mod 4, so field disc is 3*7, and theory says the
    # rational primes 3, 7 should be the square of a prime ideal.
    for p in [3, 7]:
        P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=rad.get(p))
        assert len(P) == 1
        assert P[0].e == 2
        assert P[0]**2 == p*ZK


def test_decomp_5():
    # Here is our first test of the "hard case" of prime decomposition.
    # We work in a quadratic extension Q(sqrt(d)) where d is 1 mod 4, and
    # we consider the factorization of the rational prime 2, which divides
    # the index.
    # Theory says the form of p's factorization depends on the residue of
    # d mod 8, so we consider both cases, d = 1 mod 8 and d = 5 mod 8.
    for d in [-7, -3]:
        T = Poly(x ** 2 - d)
        rad = {}
        ZK, dK = round_two(T, radicals=rad)
        p = 2
        P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=rad.get(p))
        if d % 8 == 1:
            assert len(P) == 2
            assert all(P[i].e == 1 and P[i].f == 1 for i in range(2))
            assert prod(Pi**Pi.e for Pi in P) == p * ZK
        else:
            assert d % 8 == 5
            assert len(P) == 1
            assert P[0].e == 1
            assert P[0].f == 2
            assert P[0].as_submodule() == p * ZK


def test_decomp_6():
    # Another case where 2 divides the index. This is Dedekind's example of
    # an essential discriminant divisor. (See Cohen, Exercise 6.10.)
    T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    rad = {}
    ZK, dK = round_two(T, radicals=rad)
    p = 2
    P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=rad.get(p))
    assert len(P) == 3
    assert all(Pi.e == Pi.f == 1 for Pi in P)
    assert prod(Pi**Pi.e for Pi in P) == p*ZK


def test_decomp_7():
    # Try working through an AlgebraicField
    T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    K = QQ.alg_field_from_poly(T)
    p = 2
    P = K.primes_above(p)
    ZK = K.maximal_order()
    assert len(P) == 3
    assert all(Pi.e == Pi.f == 1 for Pi in P)
    assert prod(Pi**Pi.e for Pi in P) == p*ZK


def test_decomp_8():
    # This time we consider various cubics, and try factoring all primes
    # dividing the index.
    cases = (
        x ** 3 + 3 * x ** 2 - 4 * x + 4,
        x ** 3 + 3 * x ** 2 + 3 * x - 3,
        x ** 3 + 5 * x ** 2 - x + 3,
        x ** 3 + 5 * x ** 2 - 5 * x - 5,
        x ** 3 + 3 * x ** 2 + 5,
        x ** 3 + 6 * x ** 2 + 3 * x - 1,
        x ** 3 + 6 * x ** 2 + 4,
        x ** 3 + 7 * x ** 2 + 7 * x - 7,
        x ** 3 + 7 * x ** 2 - x + 5,
        x ** 3 + 7 * x ** 2 - 5 * x + 5,
        x ** 3 + 4 * x ** 2 - 3 * x + 7,
        x ** 3 + 8 * x ** 2 + 5 * x - 1,
        x ** 3 + 8 * x ** 2 - 2 * x + 6,
        x ** 3 + 6 * x ** 2 - 3 * x + 8,
        x ** 3 + 9 * x ** 2 + 6 * x - 8,
        x ** 3 + 15 * x ** 2 - 9 * x + 13,
    )
    def display(T, p, radical, P, I, J):
        """Useful for inspection, when running test manually."""
        print('=' * 20)
        print(T, p, radical)
        for Pi in P:
            print(f'  ({Pi!r})')
        print("I: ", I)
        print("J: ", J)
        print(f'Equal: {I == J}')
    inspect = False
    for g in cases:
        T = Poly(g)
        rad = {}
        ZK, dK = round_two(T, radicals=rad)
        dT = T.discriminant()
        f_squared = dT // dK
        F = factorint(f_squared)
        for p in F:
            radical = rad.get(p)
            P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=radical)
            I = prod(Pi**Pi.e for Pi in P)
            J = p * ZK
            if inspect:
                display(T, p, radical, P, I, J)
            assert I == J


def test_PrimeIdeal_eq():
    # `==` should fail on objects of different types, so even a completely
    # inert PrimeIdeal should test unequal to the rational prime it divides.
    T = Poly(cyclotomic_poly(7))
    P0 = prime_decomp(5, T)[0]
    assert P0.f == 6
    assert P0.as_submodule() == 5 * P0.ZK
    assert P0 != 5


def test_PrimeIdeal_add():
    T = Poly(cyclotomic_poly(7))
    P0 = prime_decomp(7, T)[0]
    # Adding ideals computes their GCD, so adding the ramified prime dividing
    # 7 to 7 itself should reproduce this prime (as a submodule).
    assert P0 + 7 * P0.ZK == P0.as_submodule()


def test_str():
    # Without alias:
    k = QQ.alg_field_from_poly(Poly(x**2 + 7))
    frp = k.primes_above(2)[0]
    assert str(frp) == '(2, 3*_x/2 + 1/2)'

    frp = k.primes_above(3)[0]
    assert str(frp) == '(3)'

    # With alias:
    k = QQ.alg_field_from_poly(Poly(x ** 2 + 7), alias='alpha')
    frp = k.primes_above(2)[0]
    assert str(frp) == '(2, 3*alpha/2 + 1/2)'

    frp = k.primes_above(3)[0]
    assert str(frp) == '(3)'


def test_repr():
    T = Poly(x**2 + 7)
    ZK, dK = round_two(T)
    P = prime_decomp(2, T, dK=dK, ZK=ZK)
    assert repr(P[0]) == '[ (2, (3*x + 1)/2) e=1, f=1 ]'
    assert P[0].repr(field_gen=theta) == '[ (2, (3*theta + 1)/2) e=1, f=1 ]'
    assert P[0].repr(field_gen=theta, just_gens=True) == '(2, (3*theta + 1)/2)'


def test_PrimeIdeal_reduce():
    k = QQ.alg_field_from_poly(Poly(x ** 3 + x ** 2 - 2 * x + 8))
    Zk = k.maximal_order()
    P = k.primes_above(2)
    frp = P[2]

    # reduce_element
    a = Zk.parent(to_col([23, 20, 11]), denom=6)
    a_bar_expected = Zk.parent(to_col([11, 5, 2]), denom=6)
    a_bar = frp.reduce_element(a)
    assert a_bar == a_bar_expected

    # reduce_ANP
    a = k([QQ(11, 6), QQ(20, 6), QQ(23, 6)])
    a_bar_expected = k([QQ(2, 6), QQ(5, 6), QQ(11, 6)])
    a_bar = frp.reduce_ANP(a)
    assert a_bar == a_bar_expected

    # reduce_alg_num
    a = k.to_alg_num(a)
    a_bar_expected = k.to_alg_num(a_bar_expected)
    a_bar = frp.reduce_alg_num(a)
    assert a_bar == a_bar_expected


def test_issue_23402():
    k = QQ.alg_field_from_poly(Poly(x ** 3 + x ** 2 - 2 * x + 8))
    P = k.primes_above(3)
    assert P[0].alpha.equiv(0)
