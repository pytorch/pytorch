from sympy.ntheory import count_digits, digits, is_palindromic
from sympy.core.intfunc import num_digits

from sympy.testing.pytest import raises


def test_num_digits():
    # depending on whether one rounds up or down or uses log or log10,
    # one or more of these will fail if you don't check for the off-by
    # one condition
    assert num_digits(2, 2) == 2
    assert num_digits(2**48 - 1, 2) == 48
    assert num_digits(1000, 10) == 4
    assert num_digits(125, 5) == 4
    assert num_digits(100, 16) == 2
    assert num_digits(-1000, 10) == 4
    # if changes are made to the function, this structured test over
    # this range will expose problems
    for base in range(2, 100):
        for e in range(1, 100):
            n = base**e
            assert num_digits(n, base) == e + 1
            assert num_digits(n + 1, base) == e + 1
            assert num_digits(n - 1, base) == e


def test_digits():
    assert all(digits(n, 2)[1:] == [int(d) for d in format(n, 'b')]
                for n in range(20))
    assert all(digits(n, 8)[1:] == [int(d) for d in format(n, 'o')]
                for n in range(20))
    assert all(digits(n, 16)[1:] == [int(d, 16) for d in format(n, 'x')]
                for n in range(20))
    assert digits(2345, 34) == [34, 2, 0, 33]
    assert digits(384753, 71) == [71, 1, 5, 23, 4]
    assert digits(93409, 10) == [10, 9, 3, 4, 0, 9]
    assert digits(-92838, 11) == [-11, 6, 3, 8, 2, 9]
    assert digits(35, 10) == [10, 3, 5]
    assert digits(35, 10, 3) == [10, 0, 3, 5]
    assert digits(-35, 10, 4) == [-10, 0, 0, 3, 5]
    raises(ValueError, lambda: digits(2, 2, 1))


def test_count_digits():
    assert count_digits(55, 2) == {1: 5, 0: 1}
    assert count_digits(55, 10) == {5: 2}
    n = count_digits(123)
    assert n[4] == 0 and type(n[4]) is int


def test_is_palindromic():
    assert is_palindromic(-11)
    assert is_palindromic(11)
    assert is_palindromic(0o121, 8)
    assert not is_palindromic(123)
