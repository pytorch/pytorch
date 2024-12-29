from bisect import bisect, bisect_left

from sympy.functions.combinatorial.numbers import mobius, totient
from sympy.ntheory.generate import (sieve, Sieve)

from sympy.ntheory import isprime, randprime, nextprime, prevprime, \
    primerange, primepi, prime, primorial, composite, compositepi
from sympy.ntheory.generate import cycle_length, _primepi
from sympy.ntheory.primetest import mr
from sympy.testing.pytest import raises

def test_prime():
    assert prime(1) == 2
    assert prime(2) == 3
    assert prime(5) == 11
    assert prime(11) == 31
    assert prime(57) == 269
    assert prime(296) == 1949
    assert prime(559) == 4051
    assert prime(3000) == 27449
    assert prime(4096) == 38873
    assert prime(9096) == 94321
    assert prime(25023) == 287341
    assert prime(10000000) == 179424673 # issue #20951
    assert prime(99999999) == 2038074739
    raises(ValueError, lambda: prime(0))
    sieve.extend(3000)
    assert prime(401) == 2749
    raises(ValueError, lambda: prime(-1))


def test__primepi():
    assert _primepi(-1) == 0
    assert _primepi(1) == 0
    assert _primepi(2) == 1
    assert _primepi(5) == 3
    assert _primepi(11) == 5
    assert _primepi(57) == 16
    assert _primepi(296) == 62
    assert _primepi(559) == 102
    assert _primepi(3000) == 430
    assert _primepi(4096) == 564
    assert _primepi(9096) == 1128
    assert _primepi(25023) == 2763
    assert _primepi(10**8) == 5761455
    assert _primepi(253425253) == 13856396
    assert _primepi(8769575643) == 401464322
    sieve.extend(3000)
    assert _primepi(2000) == 303


def test_composite():
    from sympy.ntheory.generate import sieve
    sieve._reset()
    assert composite(1) == 4
    assert composite(2) == 6
    assert composite(5) == 10
    assert composite(11) == 20
    assert composite(41) == 58
    assert composite(57) == 80
    assert composite(296) == 370
    assert composite(559) == 684
    assert composite(3000) == 3488
    assert composite(4096) == 4736
    assert composite(9096) == 10368
    assert composite(25023) == 28088
    sieve.extend(3000)
    assert composite(1957) == 2300
    assert composite(2568) == 2998
    raises(ValueError, lambda: composite(0))


def test_compositepi():
    assert compositepi(1) == 0
    assert compositepi(2) == 0
    assert compositepi(5) == 1
    assert compositepi(11) == 5
    assert compositepi(57) == 40
    assert compositepi(296) == 233
    assert compositepi(559) == 456
    assert compositepi(3000) == 2569
    assert compositepi(4096) == 3531
    assert compositepi(9096) == 7967
    assert compositepi(25023) == 22259
    assert compositepi(10**8) == 94238544
    assert compositepi(253425253) == 239568856
    assert compositepi(8769575643) == 8368111320
    sieve.extend(3000)
    assert compositepi(2321) == 1976


def test_generate():
    from sympy.ntheory.generate import sieve
    sieve._reset()
    assert nextprime(-4) == 2
    assert nextprime(2) == 3
    assert nextprime(5) == 7
    assert nextprime(12) == 13
    assert prevprime(3) == 2
    assert prevprime(7) == 5
    assert prevprime(13) == 11
    assert prevprime(19) == 17
    assert prevprime(20) == 19

    sieve.extend_to_no(9)
    assert sieve._list[-1] == 23

    assert sieve._list[-1] < 31
    assert 31 in sieve

    assert nextprime(90) == 97
    assert nextprime(10**40) == (10**40 + 121)
    primelist = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                 79, 83, 89, 97, 101, 103, 107, 109, 113,
                 127, 131, 137, 139, 149, 151, 157, 163,
                 167, 173, 179, 181, 191, 193, 197, 199,
                 211, 223, 227, 229, 233, 239, 241, 251,
                 257, 263, 269, 271, 277, 281, 283, 293]
    for i in range(len(primelist) - 2):
        for j in range(2, len(primelist) - i):
            assert nextprime(primelist[i], j) == primelist[i + j]
            if 3 < i:
                assert nextprime(primelist[i] - 1, j) == primelist[i + j - 1]
    raises(ValueError, lambda: nextprime(2, 0))
    raises(ValueError, lambda: nextprime(2, -1))
    assert prevprime(97) == 89
    assert prevprime(10**40) == (10**40 - 17)

    raises(ValueError, lambda: Sieve(0))
    raises(ValueError, lambda: Sieve(-1))
    for sieve_interval in [1, 10, 11, 1_000_000]:
        s = Sieve(sieve_interval=sieve_interval)
        for head in range(s._list[-1] + 1, (s._list[-1] + 1)**2, 2):
            for tail in range(head + 1, (s._list[-1] + 1)**2):
                A = list(s._primerange(head, tail))
                B = primelist[bisect(primelist, head):bisect_left(primelist, tail)]
                assert A == B
        for k in range(s._list[-1], primelist[-1] - 1, 2):
            s = Sieve(sieve_interval=sieve_interval)
            s.extend(k)
            assert list(s._list) == primelist[:bisect(primelist, k)]
            s.extend(primelist[-1])
            assert list(s._list) == primelist

    assert list(sieve.primerange(10, 1)) == []
    assert list(sieve.primerange(5, 9)) == [5, 7]
    sieve._reset(prime=True)
    assert list(sieve.primerange(2, 13)) == [2, 3, 5, 7, 11]
    assert list(sieve.primerange(13)) == [2, 3, 5, 7, 11]
    assert list(sieve.primerange(8)) == [2, 3, 5, 7]
    assert list(sieve.primerange(-2)) == []
    assert list(sieve.primerange(29)) == [2, 3, 5, 7, 11, 13, 17, 19, 23]
    assert list(sieve.primerange(34)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    assert list(sieve.totientrange(5, 15)) == [4, 2, 6, 4, 6, 4, 10, 4, 12, 6]
    sieve._reset(totient=True)
    assert list(sieve.totientrange(3, 13)) == [2, 2, 4, 2, 6, 4, 6, 4, 10, 4]
    assert list(sieve.totientrange(900, 1000)) == [totient(x) for x in range(900, 1000)]
    assert list(sieve.totientrange(0, 1)) == []
    assert list(sieve.totientrange(1, 2)) == [1]

    assert list(sieve.mobiusrange(5, 15)) == [-1, 1, -1, 0, 0, 1, -1, 0, -1, 1]
    sieve._reset(mobius=True)
    assert list(sieve.mobiusrange(3, 13)) == [-1, 0, -1, 1, -1, 0, 0, 1, -1, 0]
    assert list(sieve.mobiusrange(1050, 1100)) == [mobius(x) for x in range(1050, 1100)]
    assert list(sieve.mobiusrange(0, 1)) == []
    assert list(sieve.mobiusrange(1, 2)) == [1]

    assert list(primerange(10, 1)) == []
    assert list(primerange(2, 7)) == [2, 3, 5]
    assert list(primerange(2, 10)) == [2, 3, 5, 7]
    assert list(primerange(1050, 1100)) == [1051, 1061,
        1063, 1069, 1087, 1091, 1093, 1097]
    s = Sieve()
    for i in range(30, 2350, 376):
        for j in range(2, 5096, 1139):
            A = list(s.primerange(i, i + j))
            B = list(primerange(i, i + j))
            assert A == B
    s = Sieve()
    sieve._reset(prime=True)
    sieve.extend(13)
    for i in range(200):
        for j in range(i, 200):
            A = list(s.primerange(i, j))
            B = list(primerange(i, j))
            assert A == B
    sieve.extend(1000)
    for a, b in [(901, 1103), # a < 1000 < b < 1000**2
                 (806, 1002007), # a < 1000 < 1000**2 < b
                 (2000, 30001), # 1000 < a < b < 1000**2
                 (100005, 1010001), # 1000 < a < 1000**2 < b
                 (1003003, 1005000), # 1000**2 < a < b
                 ]:
        assert list(primerange(a, b)) == list(s.primerange(a, b))
    sieve._reset(prime=True)
    sieve.extend(100000)
    assert len(sieve._list) == len(set(sieve._list))
    s = Sieve()
    assert s[10] == 29

    assert nextprime(2, 2) == 5

    raises(ValueError, lambda: totient(0))

    raises(ValueError, lambda: primorial(0))

    assert mr(1, [2]) is False

    func = lambda i: (i**2 + 1) % 51
    assert next(cycle_length(func, 4)) == (6, 3)
    assert list(cycle_length(func, 4, values=True)) == \
        [4, 17, 35, 2, 5, 26, 14, 44, 50, 2, 5, 26, 14]
    assert next(cycle_length(func, 4, nmax=5)) == (5, None)
    assert list(cycle_length(func, 4, nmax=5, values=True)) == \
        [4, 17, 35, 2, 5]
    sieve.extend(3000)
    assert nextprime(2968) == 2969
    assert prevprime(2930) == 2927
    raises(ValueError, lambda: prevprime(1))
    raises(ValueError, lambda: prevprime(-4))


def test_randprime():
    assert randprime(10, 1) is None
    assert randprime(3, -3) is None
    assert randprime(2, 3) == 2
    assert randprime(1, 3) == 2
    assert randprime(3, 5) == 3
    raises(ValueError, lambda: randprime(-12, -2))
    raises(ValueError, lambda: randprime(-10, 0))
    raises(ValueError, lambda: randprime(20, 22))
    raises(ValueError, lambda: randprime(0, 2))
    raises(ValueError, lambda: randprime(1, 2))
    for a in [100, 300, 500, 250000]:
        for b in [100, 300, 500, 250000]:
            p = randprime(a, a + b)
            assert a <= p < (a + b) and isprime(p)


def test_primorial():
    assert primorial(1) == 2
    assert primorial(1, nth=0) == 1
    assert primorial(2) == 6
    assert primorial(2, nth=0) == 2
    assert primorial(4, nth=0) == 6


def test_search():
    assert 2 in sieve
    assert 2.1 not in sieve
    assert 1 not in sieve
    assert 2**1000 not in sieve
    raises(ValueError, lambda: sieve.search(1))


def test_sieve_slice():
    assert sieve[5] == 11
    assert list(sieve[5:10]) == [sieve[x] for x in range(5, 10)]
    assert list(sieve[5:10:2]) == [sieve[x] for x in range(5, 10, 2)]
    assert list(sieve[1:5]) == [2, 3, 5, 7]
    raises(IndexError, lambda: sieve[:5])
    raises(IndexError, lambda: sieve[0])
    raises(IndexError, lambda: sieve[0:5])

def test_sieve_iter():
    values = []
    for value in sieve:
        if value > 7:
            break
        values.append(value)
    assert values == list(sieve[1:5])


def test_sieve_repr():
    assert "sieve" in repr(sieve)
    assert "prime" in repr(sieve)


def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert primepi(0) == 0
