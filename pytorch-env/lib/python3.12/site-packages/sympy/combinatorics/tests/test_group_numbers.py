from sympy.combinatorics.group_numbers import (is_nilpotent_number,
    is_abelian_number, is_cyclic_number, _holder_formula, groups_count)
from sympy.ntheory.factor_ import factorint
from sympy.ntheory.generate import prime
from sympy.testing.pytest import raises
from sympy import randprime


def test_is_nilpotent_number():
    assert is_nilpotent_number(21) == False
    assert is_nilpotent_number(randprime(1, 30)**12) == True
    raises(ValueError, lambda: is_nilpotent_number(-5))

    A056867	= [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19,
               23, 25, 27, 29, 31, 32, 33, 35, 37, 41, 43, 45,
               47, 49, 51, 53, 59, 61, 64, 65, 67, 69, 71, 73,
               77, 79, 81, 83, 85, 87, 89, 91, 95, 97, 99]
    for n in range(1, 100):
        assert is_nilpotent_number(n) == (n in A056867)


def test_is_abelian_number():
    assert is_abelian_number(4) == True
    assert is_abelian_number(randprime(1, 2000)**2) == True
    assert is_abelian_number(randprime(1000, 100000)) == True
    assert is_abelian_number(60) == False
    assert is_abelian_number(24) == False
    raises(ValueError, lambda: is_abelian_number(-5))

    A051532 = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 23, 25,
               29, 31, 33, 35, 37, 41, 43, 45, 47, 49, 51, 53,
               59, 61, 65, 67, 69, 71, 73, 77, 79, 83, 85, 87,
               89, 91, 95, 97, 99]
    for n in range(1, 100):
        assert is_abelian_number(n) == (n in A051532)


A003277 = [1, 2, 3, 5, 7, 11, 13, 15, 17, 19, 23, 29,
           31, 33, 35, 37, 41, 43, 47, 51, 53, 59, 61,
           65, 67, 69, 71, 73, 77, 79, 83, 85, 87, 89,
           91, 95, 97]


def test_is_cyclic_number():
    assert is_cyclic_number(15) == True
    assert is_cyclic_number(randprime(1, 2000)**2) == False
    assert is_cyclic_number(randprime(1000, 100000)) == True
    assert is_cyclic_number(4) == False
    raises(ValueError, lambda: is_cyclic_number(-5))

    for n in range(1, 100):
        assert is_cyclic_number(n) == (n in A003277)


def test_holder_formula():
    # semiprime
    assert _holder_formula({3, 5}) == 1
    assert _holder_formula({5, 11}) == 2
    # n in A003277 is always 1
    for n in A003277:
        assert _holder_formula(set(factorint(n).keys())) == 1
    # otherwise
    assert _holder_formula({2, 3, 5, 7}) == 12


def test_groups_count():
    A000001 = [0, 1, 1, 1, 2, 1, 2, 1, 5, 2, 2, 1, 5, 1,
               2, 1, 14, 1, 5, 1, 5, 2, 2, 1, 15, 2, 2,
               5, 4, 1, 4, 1, 51, 1, 2, 1, 14, 1, 2, 2,
               14, 1, 6, 1, 4, 2, 2, 1, 52, 2, 5, 1, 5,
               1, 15, 2, 13, 2, 2, 1, 13, 1, 2, 4, 267,
               1, 4, 1, 5, 1, 4, 1, 50, 1, 2, 3, 4, 1,
               6, 1, 52, 15, 2, 1, 15, 1, 2, 1, 12, 1,
               10, 1, 4, 2]
    for n in range(1, len(A000001)):
        try:
            assert groups_count(n) == A000001[n]
        except ValueError:
            pass

    A000679 = [1, 1, 2, 5, 14, 51, 267, 2328, 56092, 10494213, 49487367289]
    for e in range(1, len(A000679)):
        assert groups_count(2**e) == A000679[e]

    A090091 = [1, 1, 2, 5, 15, 67, 504, 9310, 1396077, 5937876645]
    for e in range(1, len(A090091)):
        assert groups_count(3**e) == A090091[e]

    A090130 = [1, 1, 2, 5, 15, 77, 684, 34297]
    for e in range(1, len(A090130)):
        assert groups_count(5**e) == A090130[e]

    A090140 = [1, 1, 2, 5, 15, 83, 860, 113147]
    for e in range(1, len(A090140)):
        assert groups_count(7**e) == A090140[e]

    A232105 = [51, 67, 77, 83, 87, 97, 101, 107, 111, 125, 131,
               145, 149, 155, 159, 173, 183, 193, 203, 207, 217]
    for i in range(len(A232105)):
        assert groups_count(prime(i+1)**5) == A232105[i]

    A232106 = [267, 504, 684, 860, 1192, 1476, 1944, 2264, 2876,
               4068, 4540, 6012, 7064, 7664, 8852, 10908, 13136]
    for i in range(len(A232106)):
        assert groups_count(prime(i+1)**6) == A232106[i]

    A232107 = [2328, 9310, 34297, 113147, 750735, 1600573,
               5546909, 9380741, 23316851, 71271069, 98488755]
    for i in range(len(A232107)):
        assert groups_count(prime(i+1)**7) == A232107[i]
