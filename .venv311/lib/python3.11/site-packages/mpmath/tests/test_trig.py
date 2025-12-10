from mpmath import *
from mpmath.libmp import *

def test_trig_misc_hard():
    mp.prec = 53
    # Worst-case input for an IEEE double, from a paper by Kahan
    x = ldexp(6381956970095103,797)
    assert cos(x) == mpf('-4.6871659242546277e-19')
    assert sin(x) == 1

    mp.prec = 150
    a = mpf(10**50)
    mp.prec = 53
    assert sin(a).ae(-0.7896724934293100827)
    assert cos(a).ae(-0.6135286082336635622)

    # Check relative accuracy close to x = zero
    assert sin(1e-100) == 1e-100  # when rounding to nearest
    assert sin(1e-6).ae(9.999999999998333e-007, rel_eps=2e-15, abs_eps=0)
    assert sin(1e-6j).ae(1.0000000000001666e-006j, rel_eps=2e-15, abs_eps=0)
    assert sin(-1e-6j).ae(-1.0000000000001666e-006j, rel_eps=2e-15, abs_eps=0)
    assert cos(1e-100) == 1
    assert cos(1e-6).ae(0.9999999999995)
    assert cos(-1e-6j).ae(1.0000000000005)
    assert tan(1e-100) == 1e-100
    assert tan(1e-6).ae(1.0000000000003335e-006, rel_eps=2e-15, abs_eps=0)
    assert tan(1e-6j).ae(9.9999999999966644e-007j, rel_eps=2e-15, abs_eps=0)
    assert tan(-1e-6j).ae(-9.9999999999966644e-007j, rel_eps=2e-15, abs_eps=0)

def test_trig_near_zero():
    mp.dps = 15

    for r in [round_nearest, round_down, round_up, round_floor, round_ceiling]:
        assert sin(0, rounding=r) == 0
        assert cos(0, rounding=r) == 1

    a = mpf('1e-100')
    b = mpf('-1e-100')

    assert sin(a, rounding=round_nearest) == a
    assert sin(a, rounding=round_down) < a
    assert sin(a, rounding=round_floor) < a
    assert sin(a, rounding=round_up) >= a
    assert sin(a, rounding=round_ceiling) >= a
    assert sin(b, rounding=round_nearest) == b
    assert sin(b, rounding=round_down) > b
    assert sin(b, rounding=round_floor) <= b
    assert sin(b, rounding=round_up) <= b
    assert sin(b, rounding=round_ceiling) > b

    assert cos(a, rounding=round_nearest) == 1
    assert cos(a, rounding=round_down) < 1
    assert cos(a, rounding=round_floor) < 1
    assert cos(a, rounding=round_up) == 1
    assert cos(a, rounding=round_ceiling) == 1
    assert cos(b, rounding=round_nearest) == 1
    assert cos(b, rounding=round_down) < 1
    assert cos(b, rounding=round_floor) < 1
    assert cos(b, rounding=round_up) == 1
    assert cos(b, rounding=round_ceiling) == 1


def test_trig_near_n_pi():

    mp.dps = 15
    a = [n*pi for n in [1, 2, 6, 11, 100, 1001, 10000, 100001]]
    mp.dps = 135
    a.append(10**100 * pi)
    mp.dps = 15

    assert sin(a[0]) == mpf('1.2246467991473531772e-16')
    assert sin(a[1]) == mpf('-2.4492935982947063545e-16')
    assert sin(a[2]) == mpf('-7.3478807948841190634e-16')
    assert sin(a[3]) == mpf('4.8998251578625894243e-15')
    assert sin(a[4]) == mpf('1.9643867237284719452e-15')
    assert sin(a[5]) == mpf('-8.8632615209684813458e-15')
    assert sin(a[6]) == mpf('-4.8568235395684898392e-13')
    assert sin(a[7]) == mpf('3.9087342299491231029e-11')
    assert sin(a[8]) == mpf('-1.369235466754566993528e-36')

    r = round_nearest
    assert cos(a[0], rounding=r) == -1
    assert cos(a[1], rounding=r) == 1
    assert cos(a[2], rounding=r) == 1
    assert cos(a[3], rounding=r) == -1
    assert cos(a[4], rounding=r) == 1
    assert cos(a[5], rounding=r) == -1
    assert cos(a[6], rounding=r) == 1
    assert cos(a[7], rounding=r) == -1
    assert cos(a[8], rounding=r) == 1

    r = round_up
    assert cos(a[0], rounding=r) == -1
    assert cos(a[1], rounding=r) == 1
    assert cos(a[2], rounding=r) == 1
    assert cos(a[3], rounding=r) == -1
    assert cos(a[4], rounding=r) == 1
    assert cos(a[5], rounding=r) == -1
    assert cos(a[6], rounding=r) == 1
    assert cos(a[7], rounding=r) == -1
    assert cos(a[8], rounding=r) == 1

    r = round_down
    assert cos(a[0], rounding=r) > -1
    assert cos(a[1], rounding=r) < 1
    assert cos(a[2], rounding=r) < 1
    assert cos(a[3], rounding=r) > -1
    assert cos(a[4], rounding=r) < 1
    assert cos(a[5], rounding=r) > -1
    assert cos(a[6], rounding=r) < 1
    assert cos(a[7], rounding=r) > -1
    assert cos(a[8], rounding=r) < 1

    r = round_floor
    assert cos(a[0], rounding=r) == -1
    assert cos(a[1], rounding=r) < 1
    assert cos(a[2], rounding=r) < 1
    assert cos(a[3], rounding=r) == -1
    assert cos(a[4], rounding=r) < 1
    assert cos(a[5], rounding=r) == -1
    assert cos(a[6], rounding=r) < 1
    assert cos(a[7], rounding=r) == -1
    assert cos(a[8], rounding=r) < 1

    r = round_ceiling
    assert cos(a[0], rounding=r) > -1
    assert cos(a[1], rounding=r) == 1
    assert cos(a[2], rounding=r) == 1
    assert cos(a[3], rounding=r) > -1
    assert cos(a[4], rounding=r) == 1
    assert cos(a[5], rounding=r) > -1
    assert cos(a[6], rounding=r) == 1
    assert cos(a[7], rounding=r) > -1
    assert cos(a[8], rounding=r) == 1

    mp.dps = 15
