from itertools import permutations

from sympy.external.ntheory import (bit_scan1, remove, bit_scan0, is_fermat_prp,
                                    is_euler_prp, is_strong_prp, gcdext, _lucas_sequence,
                                    is_fibonacci_prp, is_lucas_prp, is_selfridge_prp,
                                    is_strong_lucas_prp, is_strong_selfridge_prp,
                                    is_bpsw_prp, is_strong_bpsw_prp)
from sympy.testing.pytest import raises


def test_bit_scan1():
    assert bit_scan1(0) is None
    assert bit_scan1(1) == 0
    assert bit_scan1(-1) == 0
    assert bit_scan1(2) == 1
    assert bit_scan1(7) == 0
    assert bit_scan1(-7) == 0
    for i in range(100):
        assert bit_scan1(1 << i) == i
        assert bit_scan1((1 << i) * 31337) == i
    for i in range(500):
        n = (1 << 500) + (1 << i)
        assert bit_scan1(n) == i
    assert bit_scan1(1 << 1000001) == 1000001
    assert bit_scan1((1 << 273956)*7**37) == 273956
    # issue 12709
    for i in range(1, 10):
        big = 1 << i
        assert bit_scan1(-big) == bit_scan1(big)


def test_bit_scan0():
    assert bit_scan0(-1) is None
    assert bit_scan0(0) == 0
    assert bit_scan0(1) == 1
    assert bit_scan0(-2) == 0


def test_remove():
    raises(ValueError, lambda: remove(1, 1))
    assert remove(0, 3) == (0, 0)
    for f in range(2, 10):
        for y in range(2, 1000):
            for z in [1, 17, 101, 1009]:
                assert remove(z*f**y, f) == (z, y)


def test_gcdext():
    assert gcdext(0, 0) == (0, 0, 0)
    assert gcdext(3, 0) == (3, 1, 0)
    assert gcdext(0, 4) == (4, 0, 1)

    for n in range(1, 10):
        assert gcdext(n, 1) == gcdext(-n, 1) == (1, 0, 1)
        assert gcdext(n, -1) == gcdext(-n, -1) == (1, 0, -1)
        assert gcdext(n, n) == gcdext(-n, n) == (n, 0, 1)
        assert gcdext(n, -n) == gcdext(-n, -n) == (n, 0, -1)

    for n in range(2, 10):
        assert gcdext(1, n) == gcdext(1, -n) == (1, 1, 0)
        assert gcdext(-1, n) == gcdext(-1, -n) == (1, -1, 0)

    for a, b in permutations([2**5, 3, 5, 7**2, 11], 2):
        g, x, y = gcdext(a, b)
        assert g == a*x + b*y == 1


def test_is_fermat_prp():
    # invalid input
    raises(ValueError, lambda: is_fermat_prp(0, 10))
    raises(ValueError, lambda: is_fermat_prp(5, 1))

    # n = 1
    assert not is_fermat_prp(1, 3)

    # n is prime
    assert is_fermat_prp(2, 4)
    assert is_fermat_prp(3, 2)
    assert is_fermat_prp(11, 3)
    assert is_fermat_prp(2**31-1, 5)

    # A001567
    pseudorpime = [341, 561, 645, 1105, 1387, 1729, 1905, 2047,
                   2465, 2701, 2821, 3277, 4033, 4369, 4371, 4681]
    for n in pseudorpime:
        assert is_fermat_prp(n, 2)

    # A020136
    pseudorpime = [15, 85, 91, 341, 435, 451, 561, 645, 703, 1105,
                   1247, 1271, 1387, 1581, 1695, 1729, 1891, 1905]
    for n in pseudorpime:
        assert is_fermat_prp(n, 4)


def test_is_euler_prp():
    # invalid input
    raises(ValueError, lambda: is_euler_prp(0, 10))
    raises(ValueError, lambda: is_euler_prp(5, 1))

    # n = 1
    assert not is_euler_prp(1, 3)

    # n is prime
    assert is_euler_prp(2, 4)
    assert is_euler_prp(3, 2)
    assert is_euler_prp(11, 3)
    assert is_euler_prp(2**31-1, 5)

    # A047713
    pseudorpime = [561, 1105, 1729, 1905, 2047, 2465, 3277, 4033,
                   4681, 6601, 8321, 8481, 10585, 12801, 15841]
    for n in pseudorpime:
        assert is_euler_prp(n, 2)

    # A048950
    pseudorpime = [121, 703, 1729, 1891, 2821, 3281, 7381, 8401,
                   8911, 10585, 12403, 15457, 15841, 16531, 18721]
    for n in pseudorpime:
        assert is_euler_prp(n, 3)


def test_is_strong_prp():
    # invalid input
    raises(ValueError, lambda: is_strong_prp(0, 10))
    raises(ValueError, lambda: is_strong_prp(5, 1))

    # n = 1
    assert not is_strong_prp(1, 3)

    # n is prime
    assert is_strong_prp(2, 4)
    assert is_strong_prp(3, 2)
    assert is_strong_prp(11, 3)
    assert is_strong_prp(2**31-1, 5)

    # A001262
    pseudorpime = [2047, 3277, 4033, 4681, 8321, 15841, 29341,
                   42799, 49141, 52633, 65281, 74665, 80581]
    for n in pseudorpime:
        assert is_strong_prp(n, 2)

    # A020229
    pseudorpime = [121, 703, 1891, 3281, 8401, 8911, 10585, 12403,
                   16531, 18721, 19345, 23521, 31621, 44287, 47197]
    for n in pseudorpime:
        assert is_strong_prp(n, 3)


def test_lucas_sequence():
    def lucas_u(P, Q, length):
        array = [0] * length
        array[1] = 1
        for k in range(2, length):
            array[k] = P * array[k - 1] - Q * array[k - 2]
        return array

    def lucas_v(P, Q, length):
        array = [0] * length
        array[0] = 2
        array[1] = P
        for k in range(2, length):
            array[k] = P * array[k - 1] - Q * array[k - 2]
        return array

    length = 20
    for P in range(-10, 10):
        for Q in range(-10, 10):
            D = P**2 - 4*Q
            if D == 0:
                continue
            us = lucas_u(P, Q, length)
            vs = lucas_v(P, Q, length)
            for n in range(3, 100, 2):
                for k in range(length):
                    U, V, Qk = _lucas_sequence(n, P, Q, k)
                    assert U == us[k] % n
                    assert V == vs[k] % n
                    assert pow(Q, k, n) == Qk


def test_is_fibonacci_prp():
    # invalid input
    raises(ValueError, lambda: is_fibonacci_prp(3, 2, 1))
    raises(ValueError, lambda: is_fibonacci_prp(3, -5, 1))
    raises(ValueError, lambda: is_fibonacci_prp(3, 5, 2))
    raises(ValueError, lambda: is_fibonacci_prp(0, 5, -1))

    # n = 1
    assert not is_fibonacci_prp(1, 3, 1)

    # n is prime
    assert is_fibonacci_prp(2, 5, 1)
    assert is_fibonacci_prp(3, 6, -1)
    assert is_fibonacci_prp(11, 7, 1)
    assert is_fibonacci_prp(2**31-1, 8, -1)

    # A005845
    pseudorpime = [705, 2465, 2737, 3745, 4181, 5777, 6721,
                   10877, 13201, 15251, 24465, 29281, 34561]
    for n in pseudorpime:
        assert is_fibonacci_prp(n, 1, -1)


def test_is_lucas_prp():
    # invalid input
    raises(ValueError, lambda: is_lucas_prp(3, 2, 1))
    raises(ValueError, lambda: is_lucas_prp(0, 5, -1))
    raises(ValueError, lambda: is_lucas_prp(15, 3, 1))

    # n = 1
    assert not is_lucas_prp(1, 3, 1)

    # n is prime
    assert is_lucas_prp(2, 5, 2)
    assert is_lucas_prp(3, 6, -1)
    assert is_lucas_prp(11, 7, 5)
    assert is_lucas_prp(2**31-1, 8, -3)

    # A081264
    pseudorpime = [323, 377, 1891, 3827, 4181, 5777, 6601, 6721,
                   8149, 10877, 11663, 13201, 13981, 15251, 17119]
    for n in pseudorpime:
        assert is_lucas_prp(n, 1, -1)


def test_is_selfridge_prp():
    # invalid input
    raises(ValueError, lambda: is_selfridge_prp(0))

    # n = 1
    assert not is_selfridge_prp(1)

    # n is prime
    assert is_selfridge_prp(2)
    assert is_selfridge_prp(3)
    assert is_selfridge_prp(11)
    assert is_selfridge_prp(2**31-1)

    # A217120
    pseudorpime = [323, 377, 1159, 1829, 3827, 5459, 5777, 9071,
                   9179, 10877, 11419, 11663, 13919, 14839, 16109]
    for n in pseudorpime:
        assert is_selfridge_prp(n)


def test_is_strong_lucas_prp():
    # invalid input
    raises(ValueError, lambda: is_strong_lucas_prp(3, 2, 1))
    raises(ValueError, lambda: is_strong_lucas_prp(0, 5, -1))
    raises(ValueError, lambda: is_strong_lucas_prp(15, 3, 1))

    # n = 1
    assert not is_strong_lucas_prp(1, 3, 1)

    # n is prime
    assert is_strong_lucas_prp(2, 5, 2)
    assert is_strong_lucas_prp(3, 6, -1)
    assert is_strong_lucas_prp(11, 7, 5)
    assert is_strong_lucas_prp(2**31-1, 8, -3)


def test_is_strong_selfridge_prp():
    # invalid input
    raises(ValueError, lambda: is_strong_selfridge_prp(0))

    # n = 1
    assert not is_strong_selfridge_prp(1)

    # n is prime
    assert is_strong_selfridge_prp(2)
    assert is_strong_selfridge_prp(3)
    assert is_strong_selfridge_prp(11)
    assert is_strong_selfridge_prp(2**31-1)

    # A217255
    pseudorpime = [5459, 5777, 10877, 16109, 18971, 22499, 24569,
                   25199, 40309, 58519, 75077, 97439, 100127, 113573]
    for n in pseudorpime:
        assert is_strong_selfridge_prp(n)


def test_is_bpsw_prp():
    # invalid input
    raises(ValueError, lambda: is_bpsw_prp(0))

    # n = 1
    assert not is_bpsw_prp(1)

    # n is prime
    assert is_bpsw_prp(2)
    assert is_bpsw_prp(3)
    assert is_bpsw_prp(11)
    assert is_bpsw_prp(2**31-1)


def test_is_strong_bpsw_prp():
    # invalid input
    raises(ValueError, lambda: is_strong_bpsw_prp(0))

    # n = 1
    assert not is_strong_bpsw_prp(1)

    # n is prime
    assert is_strong_bpsw_prp(2)
    assert is_strong_bpsw_prp(3)
    assert is_strong_bpsw_prp(11)
    assert is_strong_bpsw_prp(2**31-1)
