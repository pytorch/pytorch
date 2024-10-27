from sympy.core import S, Rational
from sympy.combinatorics.schur_number import schur_partition, SchurNumber
from sympy.core.random import _randint
from sympy.testing.pytest import raises
from sympy.core.symbol import symbols


def _sum_free_test(subset):
    """
    Checks if subset is sum-free(There are no x,y,z in the subset such that
    x + y = z)
    """
    for i in subset:
        for j in subset:
            assert (i + j in subset) is False


def test_schur_partition():
    raises(ValueError, lambda: schur_partition(S.Infinity))
    raises(ValueError, lambda: schur_partition(-1))
    raises(ValueError, lambda: schur_partition(0))
    assert schur_partition(2) == [[1, 2]]

    random_number_generator = _randint(1000)
    for _ in range(5):
        n = random_number_generator(1, 1000)
        result = schur_partition(n)
        t = 0
        numbers = []
        for item in result:
            _sum_free_test(item)
            """
            Checks if the occurrence of all numbers is exactly one
            """
            t += len(item)
            for l in item:
                assert (l in numbers) is False
                numbers.append(l)
        assert n == t

    x = symbols("x")
    raises(ValueError, lambda: schur_partition(x))

def test_schur_number():
    first_known_schur_numbers = {1: 1, 2: 4, 3: 13, 4: 44, 5: 160}
    for k in first_known_schur_numbers:
        assert SchurNumber(k) == first_known_schur_numbers[k]

    assert SchurNumber(S.Infinity) == S.Infinity
    assert SchurNumber(0) == 0
    raises(ValueError, lambda: SchurNumber(0.5))

    n = symbols("n")
    assert SchurNumber(n).lower_bound() == 3**n/2 - Rational(1, 2)
    assert SchurNumber(8).lower_bound() == 5039
