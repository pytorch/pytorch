from hypothesis import given
from hypothesis import strategies as st
from sympy.abc import x
from sympy.polys.polytools import Poly


def polys(*, nonzero=False, domain="ZZ"):
    # This is a simple strategy, but sufficient the tests below
    elems = {"ZZ": st.integers(), "QQ": st.fractions()}
    coeff_st = st.lists(elems[domain])
    if nonzero:
        coeff_st = coeff_st.filter(any)
    return st.builds(Poly, coeff_st, st.just(x), domain=st.just(domain))


@given(f=polys(), g=polys(), r=polys())
def test_gcd_hypothesis(f, g, r):
    gcd_1 = f.gcd(g)
    gcd_2 = g.gcd(f)
    assert gcd_1 == gcd_2

    # multiply by r
    gcd_3 = g.gcd(f + r * g)
    assert gcd_1 == gcd_3


@given(f_z=polys(), g_z=polys(nonzero=True))
def test_poly_hypothesis_integers(f_z, g_z):
    remainder_z = f_z.rem(g_z)
    assert g_z.degree() >= remainder_z.degree() or remainder_z.degree() == 0


@given(f_q=polys(domain="QQ"), g_q=polys(nonzero=True, domain="QQ"))
def test_poly_hypothesis_rationals(f_q, g_q):
    remainder_q = f_q.rem(g_q)
    assert g_q.degree() >= remainder_q.degree() or remainder_q.degree() == 0
