"""
PythonMPQ: Rational number type based on Python integers.

This class is intended as a pure Python fallback for when gmpy2 is not
installed. If gmpy2 is installed then its mpq type will be used instead. The
mpq type is around 20x faster. We could just use the stdlib Fraction class
here but that is slower:

    from fractions import Fraction
    from sympy.external.pythonmpq import PythonMPQ
    nums = range(1000)
    dens = range(5, 1005)
    rats = [Fraction(n, d) for n, d in zip(nums, dens)]
    sum(rats) # <--- 24 milliseconds
    rats = [PythonMPQ(n, d) for n, d in zip(nums, dens)]
    sum(rats) # <---  7 milliseconds

Both mpq and Fraction have some awkward features like the behaviour of
division with // and %:

    >>> from fractions import Fraction
    >>> Fraction(2, 3) % Fraction(1, 4)
    1/6

For the QQ domain we do not want this behaviour because there should be no
remainder when dividing rational numbers. SymPy does not make use of this
aspect of mpq when gmpy2 is installed. Since this class is a fallback for that
case we do not bother implementing e.g. __mod__ so that we can be sure we
are not using it when gmpy2 is installed either.
"""

from __future__ import annotations
import operator
from math import gcd
from decimal import Decimal
from fractions import Fraction
import sys
from typing import Type


# Used for __hash__
_PyHASH_MODULUS = sys.hash_info.modulus
_PyHASH_INF = sys.hash_info.inf


class PythonMPQ:
    """Rational number implementation that is intended to be compatible with
    gmpy2's mpq.

    Also slightly faster than fractions.Fraction.

    PythonMPQ should be treated as immutable although no effort is made to
    prevent mutation (since that might slow down calculations).
    """
    __slots__ = ('numerator', 'denominator')

    def __new__(cls, numerator, denominator=None):
        """Construct PythonMPQ with gcd computation and checks"""
        if denominator is not None:
            #
            # PythonMPQ(n, d): require n and d to be int and d != 0
            #
            if isinstance(numerator, int) and isinstance(denominator, int):
                # This is the slow part:
                divisor = gcd(numerator, denominator)
                numerator //= divisor
                denominator //= divisor
                return cls._new_check(numerator, denominator)
        else:
            #
            # PythonMPQ(q)
            #
            # Here q can be PythonMPQ, int, Decimal, float, Fraction or str
            #
            if isinstance(numerator, int):
                return cls._new(numerator, 1)
            elif isinstance(numerator, PythonMPQ):
                return cls._new(numerator.numerator, numerator.denominator)

            # Let Fraction handle Decimal/float conversion and str parsing
            if isinstance(numerator, (Decimal, float, str)):
                numerator = Fraction(numerator)
            if isinstance(numerator, Fraction):
                return cls._new(numerator.numerator, numerator.denominator)
        #
        # Reject everything else. This is more strict than mpq which allows
        # things like mpq(Fraction, Fraction) or mpq(Decimal, any). The mpq
        # behaviour is somewhat inconsistent so we choose to accept only a
        # more strict subset of what mpq allows.
        #
        raise TypeError("PythonMPQ() requires numeric or string argument")

    @classmethod
    def _new_check(cls, numerator, denominator):
        """Construct PythonMPQ, check divide by zero and canonicalize signs"""
        if not denominator:
            raise ZeroDivisionError(f'Zero divisor {numerator}/{denominator}')
        elif denominator < 0:
            numerator = -numerator
            denominator = -denominator
        return cls._new(numerator, denominator)

    @classmethod
    def _new(cls, numerator, denominator):
        """Construct PythonMPQ efficiently (no checks)"""
        obj = super().__new__(cls)
        obj.numerator = numerator
        obj.denominator = denominator
        return obj

    def __int__(self):
        """Convert to int (truncates towards zero)"""
        p, q = self.numerator, self.denominator
        if p < 0:
            return -(-p//q)
        return p//q

    def __float__(self):
        """Convert to float (approximately)"""
        return self.numerator / self.denominator

    def __bool__(self):
        """True/False if nonzero/zero"""
        return bool(self.numerator)

    def __eq__(self, other):
        """Compare equal with PythonMPQ, int, float, Decimal or Fraction"""
        if isinstance(other, PythonMPQ):
            return (self.numerator == other.numerator
                and self.denominator == other.denominator)
        elif isinstance(other, self._compatible_types):
            return self.__eq__(PythonMPQ(other))
        else:
            return NotImplemented

    def __hash__(self):
        """hash - same as mpq/Fraction"""
        try:
            dinv = pow(self.denominator, -1, _PyHASH_MODULUS)
        except ValueError:
            hash_ = _PyHASH_INF
        else:
            hash_ = hash(hash(abs(self.numerator)) * dinv)
        result = hash_ if self.numerator >= 0 else -hash_
        return -2 if result == -1 else result

    def __reduce__(self):
        """Deconstruct for pickling"""
        return type(self), (self.numerator, self.denominator)

    def __str__(self):
        """Convert to string"""
        if self.denominator != 1:
            return f"{self.numerator}/{self.denominator}"
        else:
            return f"{self.numerator}"

    def __repr__(self):
        """Convert to string"""
        return f"MPQ({self.numerator},{self.denominator})"

    def _cmp(self, other, op):
        """Helper for lt/le/gt/ge"""
        if not isinstance(other, self._compatible_types):
            return NotImplemented
        lhs = self.numerator * other.denominator
        rhs = other.numerator * self.denominator
        return op(lhs, rhs)

    def __lt__(self, other):
        """self < other"""
        return self._cmp(other, operator.lt)

    def __le__(self, other):
        """self <= other"""
        return self._cmp(other, operator.le)

    def __gt__(self, other):
        """self > other"""
        return self._cmp(other, operator.gt)

    def __ge__(self, other):
        """self >= other"""
        return self._cmp(other, operator.ge)

    def __abs__(self):
        """abs(q)"""
        return self._new(abs(self.numerator), self.denominator)

    def __pos__(self):
        """+q"""
        return self

    def __neg__(self):
        """-q"""
        return self._new(-self.numerator, self.denominator)

    def __add__(self, other):
        """q1 + q2"""
        if isinstance(other, PythonMPQ):
            #
            # This is much faster than the naive method used in the stdlib
            # fractions module. Not sure where this method comes from
            # though...
            #
            # Compare timings for something like:
            #   nums = range(1000)
            #   rats = [PythonMPQ(n, d) for n, d in zip(nums[:-5], nums[5:])]
            #   sum(rats) # <-- time this
            #
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            g = gcd(aq, bq)
            if g == 1:
                p = ap*bq + aq*bp
                q = bq*aq
            else:
                q1, q2 = aq//g, bq//g
                p, q = ap*q2 + bp*q1, q1*q2
                g2 = gcd(p, g)
                p, q = (p // g2), q * (g // g2)

        elif isinstance(other, int):
            p = self.numerator + self.denominator * other
            q = self.denominator
        else:
            return NotImplemented

        return self._new(p, q)

    def __radd__(self, other):
        """z1 + q2"""
        if isinstance(other, int):
            p = self.numerator + self.denominator * other
            q = self.denominator
            return self._new(p, q)
        else:
            return NotImplemented

    def __sub__(self ,other):
        """q1 - q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            g = gcd(aq, bq)
            if g == 1:
                p = ap*bq - aq*bp
                q = bq*aq
            else:
                q1, q2 = aq//g, bq//g
                p, q = ap*q2 - bp*q1, q1*q2
                g2 = gcd(p, g)
                p, q = (p // g2), q * (g // g2)
        elif isinstance(other, int):
            p = self.numerator - self.denominator*other
            q = self.denominator
        else:
            return NotImplemented

        return self._new(p, q)

    def __rsub__(self, other):
        """z1 - q2"""
        if isinstance(other, int):
            p = self.denominator * other - self.numerator
            q = self.denominator
            return self._new(p, q)
        else:
            return NotImplemented

    def __mul__(self, other):
        """q1 * q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            x1 = gcd(ap, bq)
            x2 = gcd(bp, aq)
            p, q = ((ap//x1)*(bp//x2), (aq//x2)*(bq//x1))
        elif isinstance(other, int):
            x = gcd(other, self.denominator)
            p = self.numerator*(other//x)
            q = self.denominator//x
        else:
            return NotImplemented

        return self._new(p, q)

    def __rmul__(self, other):
        """z1 * q2"""
        if isinstance(other, int):
            x = gcd(self.denominator, other)
            p = self.numerator*(other//x)
            q = self.denominator//x
            return self._new(p, q)
        else:
            return NotImplemented

    def __pow__(self, exp):
        """q ** z"""
        p, q = self.numerator, self.denominator

        if exp < 0:
            p, q, exp = q, p, -exp

        return self._new_check(p**exp, q**exp)

    def __truediv__(self, other):
        """q1 / q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            x1 = gcd(ap, bp)
            x2 = gcd(bq, aq)
            p, q = ((ap//x1)*(bq//x2), (aq//x2)*(bp//x1))
        elif isinstance(other, int):
            x = gcd(other, self.numerator)
            p = self.numerator//x
            q = self.denominator*(other//x)
        else:
            return NotImplemented

        return self._new_check(p, q)

    def __rtruediv__(self, other):
        """z / q"""
        if isinstance(other, int):
            x = gcd(self.numerator, other)
            p = self.denominator*(other//x)
            q = self.numerator//x
            return self._new_check(p, q)
        else:
            return NotImplemented

    _compatible_types: tuple[Type, ...] = ()

#
# These are the types that PythonMPQ will interoperate with for operations
# and comparisons such as ==, + etc. We define this down here so that we can
# include PythonMPQ in the list as well.
#
PythonMPQ._compatible_types = (PythonMPQ, int, Decimal, Fraction)
