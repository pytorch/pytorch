from mpmath.libmp import *
from mpmath import mpf, mp

from random import randint, choice, seed

all_modes = [round_floor, round_ceiling, round_down, round_up, round_nearest]

fb = from_bstr
fi = from_int
ff = from_float


def test_div_1_3():
    a = fi(1)
    b = fi(3)
    c = fi(-1)

    # floor rounds down, ceiling rounds up
    assert mpf_div(a, b, 7, round_floor)   == fb('0.01010101')
    assert mpf_div(a, b, 7, round_ceiling) == fb('0.01010110')
    assert mpf_div(a, b, 7, round_down)    == fb('0.01010101')
    assert mpf_div(a, b, 7, round_up)      == fb('0.01010110')
    assert mpf_div(a, b, 7, round_nearest) == fb('0.01010101')

    # floor rounds up, ceiling rounds down
    assert mpf_div(c, b, 7, round_floor)   == fb('-0.01010110')
    assert mpf_div(c, b, 7, round_ceiling) == fb('-0.01010101')
    assert mpf_div(c, b, 7, round_down)    == fb('-0.01010101')
    assert mpf_div(c, b, 7, round_up)      == fb('-0.01010110')
    assert mpf_div(c, b, 7, round_nearest) == fb('-0.01010101')

def test_mpf_divi_1_3():
    a = 1
    b = fi(3)
    c = -1
    assert mpf_rdiv_int(a, b, 7, round_floor)   == fb('0.01010101')
    assert mpf_rdiv_int(a, b, 7, round_ceiling) == fb('0.01010110')
    assert mpf_rdiv_int(a, b, 7, round_down)    == fb('0.01010101')
    assert mpf_rdiv_int(a, b, 7, round_up)      == fb('0.01010110')
    assert mpf_rdiv_int(a, b, 7, round_nearest) == fb('0.01010101')
    assert mpf_rdiv_int(c, b, 7, round_floor)   == fb('-0.01010110')
    assert mpf_rdiv_int(c, b, 7, round_ceiling) == fb('-0.01010101')
    assert mpf_rdiv_int(c, b, 7, round_down)    == fb('-0.01010101')
    assert mpf_rdiv_int(c, b, 7, round_up)      == fb('-0.01010110')
    assert mpf_rdiv_int(c, b, 7, round_nearest) == fb('-0.01010101')


def test_div_300():

    q = fi(1000000)
    a = fi(300499999)    # a/q is a little less than a half-integer
    b = fi(300500000)    # b/q exactly a half-integer
    c = fi(300500001)    # c/q is a little more than a half-integer

    # Check nearest integer rounding (prec=9 as 2**8 < 300 < 2**9)

    assert mpf_div(a, q, 9, round_down) == fi(300)
    assert mpf_div(b, q, 9, round_down) == fi(300)
    assert mpf_div(c, q, 9, round_down) == fi(300)
    assert mpf_div(a, q, 9, round_up) == fi(301)
    assert mpf_div(b, q, 9, round_up) == fi(301)
    assert mpf_div(c, q, 9, round_up) == fi(301)

    # Nearest even integer is down
    assert mpf_div(a, q, 9, round_nearest) == fi(300)
    assert mpf_div(b, q, 9, round_nearest) == fi(300)
    assert mpf_div(c, q, 9, round_nearest) == fi(301)

    # Nearest even integer is up
    a = fi(301499999)
    b = fi(301500000)
    c = fi(301500001)
    assert mpf_div(a, q, 9, round_nearest) == fi(301)
    assert mpf_div(b, q, 9, round_nearest) == fi(302)
    assert mpf_div(c, q, 9, round_nearest) == fi(302)


def test_tight_integer_division():
    # Test that integer division at tightest possible precision is exact
    N = 100
    seed(1)
    for i in range(N):
        a = choice([1, -1]) * randint(1, 1<<randint(10, 100))
        b = choice([1, -1]) * randint(1, 1<<randint(10, 100))
        p = a * b
        width = bitcount(abs(b)) - trailing(b)
        a = fi(a); b = fi(b); p = fi(p)
        for mode in all_modes:
            assert mpf_div(p, a, width, mode) == b


def test_epsilon_rounding():
    # Verify that mpf_div uses infinite precision; this result will
    # appear to be exactly 0.101 to a near-sighted algorithm

    a = fb('0.101' + ('0'*200) + '1')
    b = fb('1.10101')
    c = mpf_mul(a, b, 250, round_floor) # exact
    assert mpf_div(c, b, bitcount(a[1]), round_floor) == a # exact

    assert mpf_div(c, b, 2, round_down) == fb('0.10')
    assert mpf_div(c, b, 3, round_down) == fb('0.101')
    assert mpf_div(c, b, 2, round_up) == fb('0.11')
    assert mpf_div(c, b, 3, round_up) == fb('0.110')
    assert mpf_div(c, b, 2, round_floor) == fb('0.10')
    assert mpf_div(c, b, 3, round_floor) == fb('0.101')
    assert mpf_div(c, b, 2, round_ceiling) == fb('0.11')
    assert mpf_div(c, b, 3, round_ceiling) == fb('0.110')

    # The same for negative numbers
    a = fb('-0.101' + ('0'*200) + '1')
    b = fb('1.10101')
    c = mpf_mul(a, b, 250, round_floor)
    assert mpf_div(c, b, bitcount(a[1]), round_floor) == a

    assert mpf_div(c, b, 2, round_down) == fb('-0.10')
    assert mpf_div(c, b, 3, round_up) == fb('-0.110')

    # Floor goes up, ceiling goes down
    assert mpf_div(c, b, 2, round_floor) == fb('-0.11')
    assert mpf_div(c, b, 3, round_floor) == fb('-0.110')
    assert mpf_div(c, b, 2, round_ceiling) == fb('-0.10')
    assert mpf_div(c, b, 3, round_ceiling) == fb('-0.101')


def test_mod():
    mp.dps = 15
    assert mpf(234) % 1 == 0
    assert mpf(-3) % 256 == 253
    assert mpf(0.25) % 23490.5 == 0.25
    assert mpf(0.25) % -23490.5 == -23490.25
    assert mpf(-0.25) % 23490.5 == 23490.25
    assert mpf(-0.25) % -23490.5 == -0.25
    # Check that these cases are handled efficiently
    assert mpf('1e10000000000') % 1 == 0
    assert mpf('1.23e-1000000000') % 1 == mpf('1.23e-1000000000')
    # test __rmod__
    assert 3 % mpf('1.75') == 1.25

def test_div_negative_rnd_bug():
    mp.dps = 15
    assert (-3) / mpf('0.1531879017645047') == mpf('-19.583791966887116')
    assert mpf('-2.6342475750861301') / mpf('0.35126216427941814') == mpf('-7.4993775104985909')
