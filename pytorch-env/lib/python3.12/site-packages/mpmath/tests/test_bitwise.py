"""
Test bit-level integer and mpf operations
"""

from mpmath import *
from mpmath.libmp import *

def test_bitcount():
    assert bitcount(0) == 0
    assert bitcount(1) == 1
    assert bitcount(7) == 3
    assert bitcount(8) == 4
    assert bitcount(2**100) == 101
    assert bitcount(2**100-1) == 100

def test_trailing():
    assert trailing(0) == 0
    assert trailing(1) == 0
    assert trailing(2) == 1
    assert trailing(7) == 0
    assert trailing(8) == 3
    assert trailing(2**100) == 100
    assert trailing(2**100-1) == 0

def test_round_down():
    assert from_man_exp(0, -4, 4, round_down)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_down)[:3] == (0, 15, 0)
    assert from_man_exp(0xf1, -4, 4, round_down)[:3] == (0, 15, 0)
    assert from_man_exp(0xff, -4, 4, round_down)[:3] == (0, 15, 0)
    assert from_man_exp(-0xf0, -4, 4, round_down)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf1, -4, 4, round_down)[:3] == (1, 15, 0)
    assert from_man_exp(-0xff, -4, 4, round_down)[:3] == (1, 15, 0)

def test_round_up():
    assert from_man_exp(0, -4, 4, round_up)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_up)[:3] == (0, 15, 0)
    assert from_man_exp(0xf1, -4, 4, round_up)[:3] == (0, 1, 4)
    assert from_man_exp(0xff, -4, 4, round_up)[:3] == (0, 1, 4)
    assert from_man_exp(-0xf0, -4, 4, round_up)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf1, -4, 4, round_up)[:3] == (1, 1, 4)
    assert from_man_exp(-0xff, -4, 4, round_up)[:3] == (1, 1, 4)

def test_round_floor():
    assert from_man_exp(0, -4, 4, round_floor)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_floor)[:3] == (0, 15, 0)
    assert from_man_exp(0xf1, -4, 4, round_floor)[:3] == (0, 15, 0)
    assert from_man_exp(0xff, -4, 4, round_floor)[:3] == (0, 15, 0)
    assert from_man_exp(-0xf0, -4, 4, round_floor)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf1, -4, 4, round_floor)[:3] == (1, 1, 4)
    assert from_man_exp(-0xff, -4, 4, round_floor)[:3] == (1, 1, 4)

def test_round_ceiling():
    assert from_man_exp(0, -4, 4, round_ceiling)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_ceiling)[:3] == (0, 15, 0)
    assert from_man_exp(0xf1, -4, 4, round_ceiling)[:3] == (0, 1, 4)
    assert from_man_exp(0xff, -4, 4, round_ceiling)[:3] == (0, 1, 4)
    assert from_man_exp(-0xf0, -4, 4, round_ceiling)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf1, -4, 4, round_ceiling)[:3] == (1, 15, 0)
    assert from_man_exp(-0xff, -4, 4, round_ceiling)[:3] == (1, 15, 0)

def test_round_nearest():
    assert from_man_exp(0, -4, 4, round_nearest)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_nearest)[:3] == (0, 15, 0)
    assert from_man_exp(0xf7, -4, 4, round_nearest)[:3] == (0, 15, 0)
    assert from_man_exp(0xf8, -4, 4, round_nearest)[:3] == (0, 1, 4)    # 1111.1000 -> 10000.0
    assert from_man_exp(0xf9, -4, 4, round_nearest)[:3] == (0, 1, 4)    # 1111.1001 -> 10000.0
    assert from_man_exp(0xe8, -4, 4, round_nearest)[:3] == (0, 7, 1)    # 1110.1000 -> 1110.0
    assert from_man_exp(0xe9, -4, 4, round_nearest)[:3] == (0, 15, 0)     # 1110.1001 -> 1111.0
    assert from_man_exp(-0xf0, -4, 4, round_nearest)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf7, -4, 4, round_nearest)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf8, -4, 4, round_nearest)[:3] == (1, 1, 4)
    assert from_man_exp(-0xf9, -4, 4, round_nearest)[:3] == (1, 1, 4)
    assert from_man_exp(-0xe8, -4, 4, round_nearest)[:3] == (1, 7, 1)
    assert from_man_exp(-0xe9, -4, 4, round_nearest)[:3] == (1, 15, 0)

def test_rounding_bugs():
    # 1 less than power-of-two cases
    assert from_man_exp(72057594037927935, -56, 53, round_up) == (0, 1, 0, 1)
    assert from_man_exp(73786976294838205979, -65, 53, round_nearest) == (0, 1, 1, 1)
    assert from_man_exp(31, 0, 4, round_up) == (0, 1, 5, 1)
    assert from_man_exp(-31, 0, 4, round_floor) == (1, 1, 5, 1)
    assert from_man_exp(255, 0, 7, round_up) == (0, 1, 8, 1)
    assert from_man_exp(-255, 0, 7, round_floor) == (1, 1, 8, 1)

def test_rounding_issue_200():
    a = from_man_exp(9867,-100)
    b = from_man_exp(9867,-200)
    c = from_man_exp(-1,0)
    z = (1, 1023, -10, 10)
    assert mpf_add(a, c, 10, 'd') == z
    assert mpf_add(b, c, 10, 'd') == z
    assert mpf_add(c, a, 10, 'd') == z
    assert mpf_add(c, b, 10, 'd') == z

def test_perturb():
    a = fone
    b = from_float(0.99999999999999989)
    c = from_float(1.0000000000000002)
    assert mpf_perturb(a, 0, 53, round_nearest) == a
    assert mpf_perturb(a, 1, 53, round_nearest) == a
    assert mpf_perturb(a, 0, 53, round_up) == c
    assert mpf_perturb(a, 0, 53, round_ceiling) == c
    assert mpf_perturb(a, 0, 53, round_down) == a
    assert mpf_perturb(a, 0, 53, round_floor) == a
    assert mpf_perturb(a, 1, 53, round_up) == a
    assert mpf_perturb(a, 1, 53, round_ceiling) == a
    assert mpf_perturb(a, 1, 53, round_down) == b
    assert mpf_perturb(a, 1, 53, round_floor) == b
    a = mpf_neg(a)
    b = mpf_neg(b)
    c = mpf_neg(c)
    assert mpf_perturb(a, 0, 53, round_nearest) == a
    assert mpf_perturb(a, 1, 53, round_nearest) == a
    assert mpf_perturb(a, 0, 53, round_up) == a
    assert mpf_perturb(a, 0, 53, round_floor) == a
    assert mpf_perturb(a, 0, 53, round_down) == b
    assert mpf_perturb(a, 0, 53, round_ceiling) == b
    assert mpf_perturb(a, 1, 53, round_up) == c
    assert mpf_perturb(a, 1, 53, round_floor) == c
    assert mpf_perturb(a, 1, 53, round_down) == a
    assert mpf_perturb(a, 1, 53, round_ceiling) == a

def test_add_exact():
    ff = from_float
    assert mpf_add(ff(3.0), ff(2.5)) == ff(5.5)
    assert mpf_add(ff(3.0), ff(-2.5)) == ff(0.5)
    assert mpf_add(ff(-3.0), ff(2.5)) == ff(-0.5)
    assert mpf_add(ff(-3.0), ff(-2.5)) == ff(-5.5)
    assert mpf_sub(mpf_add(fone, ff(1e-100)), fone) == ff(1e-100)
    assert mpf_sub(mpf_add(ff(1e-100), fone), fone) == ff(1e-100)
    assert mpf_sub(mpf_add(fone, ff(-1e-100)), fone) == ff(-1e-100)
    assert mpf_sub(mpf_add(ff(-1e-100), fone), fone) == ff(-1e-100)
    assert mpf_add(fone, fzero) == fone
    assert mpf_add(fzero, fone) == fone
    assert mpf_add(fzero, fzero) == fzero

def test_long_exponent_shifts():
    mp.dps = 15
    # Check for possible bugs due to exponent arithmetic overflow
    # in a C implementation
    x = mpf(1)
    for p in [32, 64]:
        a = ldexp(1,2**(p-1))
        b = ldexp(1,2**p)
        c = ldexp(1,2**(p+1))
        d = ldexp(1,-2**(p-1))
        e = ldexp(1,-2**p)
        f = ldexp(1,-2**(p+1))
        assert (x+a) == a
        assert (x+b) == b
        assert (x+c) == c
        assert (x+d) == x
        assert (x+e) == x
        assert (x+f) == x
        assert (a+x) == a
        assert (b+x) == b
        assert (c+x) == c
        assert (d+x) == x
        assert (e+x) == x
        assert (f+x) == x
        assert (x-a) == -a
        assert (x-b) == -b
        assert (x-c) == -c
        assert (x-d) == x
        assert (x-e) == x
        assert (x-f) == x
        assert (a-x) == a
        assert (b-x) == b
        assert (c-x) == c
        assert (d-x) == -x
        assert (e-x) == -x
        assert (f-x) == -x

def test_float_rounding():
    mp.prec = 64
    for x in [mpf(1), mpf(1)+eps, mpf(1)-eps, -mpf(1)+eps, -mpf(1)-eps]:
        fa = float(x)
        fb = float(fadd(x,0,prec=53,rounding='n'))
        assert fa == fb
        z = mpc(x,x)
        ca = complex(z)
        cb = complex(fadd(z,0,prec=53,rounding='n'))
        assert ca == cb
        for rnd in ['n', 'd', 'u', 'f', 'c']:
            fa = to_float(x._mpf_, rnd=rnd)
            fb = to_float(fadd(x,0,prec=53,rounding=rnd)._mpf_, rnd=rnd)
            assert fa == fb
    mp.prec = 53
