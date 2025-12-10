"""Tests for computing Galois groups. """

from sympy.abc import x
from sympy.combinatorics.galois import (
    S1TransitiveSubgroups, S2TransitiveSubgroups, S3TransitiveSubgroups,
    S4TransitiveSubgroups, S5TransitiveSubgroups, S6TransitiveSubgroups,
)
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.numberfields.galoisgroups import (
    tschirnhausen_transformation,
    galois_group,
    _galois_group_degree_4_root_approx,
    _galois_group_degree_5_hybrid,
)
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises


def test_tschirnhausen_transformation():
    for T in [
        Poly(x**2 - 2),
        Poly(x**2 + x + 1),
        Poly(x**4 + 1),
        Poly(x**4 - x**3 + x**2 - x + 1),
    ]:
        _, U = tschirnhausen_transformation(T)
        assert U.degree() == T.degree()
        assert U.is_monic
        assert U.is_irreducible
        K = QQ.alg_field_from_poly(T)
        L = QQ.alg_field_from_poly(U)
        assert field_isomorphism(K.ext, L.ext) is not None


# Test polys are from:
# Cohen, H. *A Course in Computational Algebraic Number Theory*.
test_polys_by_deg = {
    # Degree 1
    1: [
        (x, S1TransitiveSubgroups.S1, True)
    ],
    # Degree 2
    2: [
        (x**2 + x + 1, S2TransitiveSubgroups.S2, False)
    ],
    # Degree 3
    3: [
        (x**3 + x**2 - 2*x - 1, S3TransitiveSubgroups.A3, True),
        (x**3 + 2, S3TransitiveSubgroups.S3, False),
    ],
    # Degree 4
    4: [
        (x**4 + x**3 + x**2 + x + 1, S4TransitiveSubgroups.C4, False),
        (x**4 + 1, S4TransitiveSubgroups.V, True),
        (x**4 - 2, S4TransitiveSubgroups.D4, False),
        (x**4 + 8*x + 12, S4TransitiveSubgroups.A4, True),
        (x**4 + x + 1, S4TransitiveSubgroups.S4, False),
    ],
    # Degree 5
    5: [
        (x**5 + x**4 - 4*x**3 - 3*x**2 + 3*x + 1, S5TransitiveSubgroups.C5, True),
        (x**5 - 5*x + 12, S5TransitiveSubgroups.D5, True),
        (x**5 + 2, S5TransitiveSubgroups.M20, False),
        (x**5 + 20*x + 16, S5TransitiveSubgroups.A5, True),
        (x**5 - x + 1, S5TransitiveSubgroups.S5, False),
    ],
    # Degree 6
    6: [
        (x**6 + x**5 + x**4 + x**3 + x**2 + x + 1, S6TransitiveSubgroups.C6, False),
        (x**6 + 108, S6TransitiveSubgroups.S3, False),
        (x**6 + 2, S6TransitiveSubgroups.D6, False),
        (x**6 - 3*x**2 - 1, S6TransitiveSubgroups.A4, True),
        (x**6 + 3*x**3 + 3, S6TransitiveSubgroups.G18, False),
        (x**6 - 3*x**2 + 1, S6TransitiveSubgroups.A4xC2, False),
        (x**6 - 4*x**2 - 1, S6TransitiveSubgroups.S4p, True),
        (x**6 - 3*x**5 + 6*x**4 - 7*x**3 + 2*x**2 + x - 4, S6TransitiveSubgroups.S4m, False),
        (x**6 + 2*x**3 - 2, S6TransitiveSubgroups.G36m, False),
        (x**6 + 2*x**2 + 2, S6TransitiveSubgroups.S4xC2, False),
        (x**6 + 10*x**5 + 55*x**4 + 140*x**3 + 175*x**2 + 170*x + 25, S6TransitiveSubgroups.PSL2F5, True),
        (x**6 + 10*x**5 + 55*x**4 + 140*x**3 + 175*x**2 - 3019*x + 25, S6TransitiveSubgroups.PGL2F5, False),
        (x**6 + 6*x**4 + 2*x**3 + 9*x**2 + 6*x - 4, S6TransitiveSubgroups.G36p, True),
        (x**6 + 2*x**4 + 2*x**3 + x**2 + 2*x + 2, S6TransitiveSubgroups.G72, False),
        (x**6 + 24*x - 20, S6TransitiveSubgroups.A6, True),
        (x**6 + x + 1, S6TransitiveSubgroups.S6, False),
    ],
}


def test_galois_group():
    """
    Try all the test polys.
    """
    for deg in range(1, 7):
        polys = test_polys_by_deg[deg]
        for T, G, alt in polys:
            assert galois_group(T, by_name=True) == (G, alt)


def test_galois_group_degree_out_of_bounds():
    raises(ValueError, lambda: galois_group(Poly(0, x)))
    raises(ValueError, lambda: galois_group(Poly(1, x)))
    raises(ValueError, lambda: galois_group(Poly(x ** 7 + 1)))


def test_galois_group_not_by_name():
    """
    Check at least one polynomial of each supported degree, to see that
    conversion from name to group works.
    """
    for deg in range(1, 7):
        T, G_name, _ = test_polys_by_deg[deg][0]
        G, _ = galois_group(T)
        assert G == G_name.get_perm_group()


def test_galois_group_not_monic_over_ZZ():
    """
    Check that we can work with polys that are not monic over ZZ.
    """
    for deg in range(1, 7):
        T, G, alt = test_polys_by_deg[deg][0]
        assert galois_group(T/2, by_name=True) == (G, alt)


def test__galois_group_degree_4_root_approx():
    for T, G, alt in test_polys_by_deg[4]:
        assert _galois_group_degree_4_root_approx(Poly(T)) == (G, alt)


def test__galois_group_degree_5_hybrid():
    for T, G, alt in test_polys_by_deg[5]:
        assert _galois_group_degree_5_hybrid(Poly(T)) == (G, alt)


def test_AlgebraicField_galois_group():
    k = QQ.alg_field_from_poly(Poly(x**4 + 1))
    G, _ = k.galois_group(by_name=True)
    assert G == S4TransitiveSubgroups.V

    k = QQ.alg_field_from_poly(Poly(x**4 - 2))
    G, _ = k.galois_group(by_name=True)
    assert G == S4TransitiveSubgroups.D4
