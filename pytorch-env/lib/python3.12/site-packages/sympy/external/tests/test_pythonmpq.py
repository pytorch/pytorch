"""
test_pythonmpq.py

Test the PythonMPQ class for consistency with gmpy2's mpq type. If gmpy2 is
installed run the same tests for both.
"""
from fractions import Fraction
from decimal import Decimal
import pickle
from typing import Callable, List, Tuple, Type

from sympy.testing.pytest import raises

from sympy.external.pythonmpq import PythonMPQ

#
# If gmpy2 is installed then run the tests for both mpq and PythonMPQ.
# That should ensure consistency between the implementation here and mpq.
#
rational_types: List[Tuple[Callable, Type, Callable, Type]]
rational_types = [(PythonMPQ, PythonMPQ, int, int)]
try:
    from gmpy2 import mpq, mpz
    rational_types.append((mpq, type(mpq(1)), mpz, type(mpz(1))))
except ImportError:
    pass


def test_PythonMPQ():
    #
    # Test PythonMPQ and also mpq if gmpy/gmpy2 is installed.
    #
    for Q, TQ, Z, TZ in rational_types:

        def check_Q(q):
            assert isinstance(q, TQ)
            assert isinstance(q.numerator, TZ)
            assert isinstance(q.denominator, TZ)
            return q.numerator, q.denominator

        # Check construction from different types
        assert check_Q(Q(3)) == (3, 1)
        assert check_Q(Q(3, 5)) == (3, 5)
        assert check_Q(Q(Q(3, 5))) == (3, 5)
        assert check_Q(Q(0.5)) == (1, 2)
        assert check_Q(Q('0.5')) == (1, 2)
        assert check_Q(Q(Fraction(3, 5))) == (3, 5)

        # https://github.com/aleaxit/gmpy/issues/327
        if Q is PythonMPQ:
            assert check_Q(Q(Decimal('0.6'))) == (3, 5)

        # Invalid types
        raises(TypeError, lambda: Q([]))
        raises(TypeError, lambda: Q([], []))

        # Check normalisation of signs
        assert check_Q(Q(2, 3)) == (2, 3)
        assert check_Q(Q(-2, 3)) == (-2, 3)
        assert check_Q(Q(2, -3)) == (-2, 3)
        assert check_Q(Q(-2, -3)) == (2, 3)

        # Check gcd calculation
        assert check_Q(Q(12, 8)) == (3, 2)

        # __int__/__float__
        assert int(Q(5, 3)) == 1
        assert int(Q(-5, 3)) == -1
        assert float(Q(5, 2)) == 2.5
        assert float(Q(-5, 2)) == -2.5

        # __str__/__repr__
        assert str(Q(2, 1)) == "2"
        assert str(Q(1, 2)) == "1/2"
        if Q is PythonMPQ:
            assert repr(Q(2, 1)) == "MPQ(2,1)"
            assert repr(Q(1, 2)) == "MPQ(1,2)"
        else:
            assert repr(Q(2, 1)) == "mpq(2,1)"
            assert repr(Q(1, 2)) == "mpq(1,2)"

        # __bool__
        assert bool(Q(1, 2)) is True
        assert bool(Q(0)) is False

        # __eq__/__ne__
        assert (Q(2, 3) == Q(2, 3)) is True
        assert (Q(2, 3) == Q(2, 5)) is False
        assert (Q(2, 3) != Q(2, 3)) is False
        assert (Q(2, 3) != Q(2, 5)) is True

        # __hash__
        assert hash(Q(3, 5)) == hash(Fraction(3, 5))

        # __reduce__
        q = Q(2, 3)
        assert pickle.loads(pickle.dumps(q)) == q

        # __ge__/__gt__/__le__/__lt__
        assert (Q(1, 3) < Q(2, 3)) is True
        assert (Q(2, 3) < Q(2, 3)) is False
        assert (Q(2, 3) < Q(1, 3)) is False
        assert (Q(-2, 3) < Q(1, 3)) is True
        assert (Q(1, 3) < Q(-2, 3)) is False

        assert (Q(1, 3) <= Q(2, 3)) is True
        assert (Q(2, 3) <= Q(2, 3)) is True
        assert (Q(2, 3) <= Q(1, 3)) is False
        assert (Q(-2, 3) <= Q(1, 3)) is True
        assert (Q(1, 3) <= Q(-2, 3)) is False

        assert (Q(1, 3) > Q(2, 3)) is False
        assert (Q(2, 3) > Q(2, 3)) is False
        assert (Q(2, 3) > Q(1, 3)) is True
        assert (Q(-2, 3) > Q(1, 3)) is False
        assert (Q(1, 3) > Q(-2, 3)) is True

        assert (Q(1, 3) >= Q(2, 3)) is False
        assert (Q(2, 3) >= Q(2, 3)) is True
        assert (Q(2, 3) >= Q(1, 3)) is True
        assert (Q(-2, 3) >= Q(1, 3)) is False
        assert (Q(1, 3) >= Q(-2, 3)) is True

        # __abs__/__pos__/__neg__
        assert abs(Q(2, 3)) == abs(Q(-2, 3)) == Q(2, 3)
        assert +Q(2, 3) == Q(2, 3)
        assert -Q(2, 3) == Q(-2, 3)

        # __add__/__radd__
        assert Q(2, 3) + Q(5, 7) == Q(29, 21)
        assert Q(2, 3) + 1 == Q(5, 3)
        assert 1 + Q(2, 3) == Q(5, 3)
        raises(TypeError, lambda: [] + Q(1))
        raises(TypeError, lambda: Q(1) + [])

        # __sub__/__rsub__
        assert Q(2, 3) - Q(5, 7) == Q(-1, 21)
        assert Q(2, 3) - 1 == Q(-1, 3)
        assert 1 - Q(2, 3) == Q(1, 3)
        raises(TypeError, lambda: [] - Q(1))
        raises(TypeError, lambda: Q(1) - [])

        # __mul__/__rmul__
        assert Q(2, 3) * Q(5, 7) == Q(10, 21)
        assert Q(2, 3) * 1 == Q(2, 3)
        assert 1 * Q(2, 3) == Q(2, 3)
        raises(TypeError, lambda: [] * Q(1))
        raises(TypeError, lambda: Q(1) * [])

        # __pow__/__rpow__
        assert Q(2, 3) ** 2 == Q(4, 9)
        assert Q(2, 3) ** 1 == Q(2, 3)
        assert Q(-2, 3) ** 2 == Q(4, 9)
        assert Q(-2, 3) ** -1 == Q(-3, 2)
        if Q is PythonMPQ:
            raises(TypeError, lambda: 1 ** Q(2, 3))
            raises(TypeError, lambda: Q(1, 4) ** Q(1, 2))
        raises(TypeError, lambda: [] ** Q(1))
        raises(TypeError, lambda: Q(1) ** [])

        # __div__/__rdiv__
        assert Q(2, 3) / Q(5, 7) == Q(14, 15)
        assert Q(2, 3) / 1 == Q(2, 3)
        assert 1 / Q(2, 3) == Q(3, 2)
        raises(TypeError, lambda: [] / Q(1))
        raises(TypeError, lambda: Q(1) / [])
        raises(ZeroDivisionError, lambda: Q(1, 2) / Q(0))

        # __divmod__
        if Q is PythonMPQ:
            raises(TypeError, lambda: Q(2, 3) // Q(1, 3))
            raises(TypeError, lambda: Q(2, 3) % Q(1, 3))
            raises(TypeError, lambda: 1 // Q(1, 3))
            raises(TypeError, lambda: 1 % Q(1, 3))
            raises(TypeError, lambda: Q(2, 3) // 1)
            raises(TypeError, lambda: Q(2, 3) % 1)
