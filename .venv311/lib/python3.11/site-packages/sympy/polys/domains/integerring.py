"""Implementation of :class:`IntegerRing` class. """

from sympy.external.gmpy import MPZ, GROUND_TYPES

from sympy.core.numbers import int_valued
from sympy.polys.domains.groundtypes import (
    SymPyInteger,
    factorial,
    gcdex, gcd, lcm, sqrt, is_square, sqrtrem,
)

from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.ring import Ring
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

import math

@public
class IntegerRing(Ring, CharacteristicZero, SimpleDomain):
    r"""The domain ``ZZ`` representing the integers `\mathbb{Z}`.

    The :py:class:`IntegerRing` class represents the ring of integers as a
    :py:class:`~.Domain` in the domain system. :py:class:`IntegerRing` is a
    super class of :py:class:`PythonIntegerRing` and
    :py:class:`GMPYIntegerRing` one of which will be the implementation for
    :ref:`ZZ` depending on whether or not ``gmpy`` or ``gmpy2`` is installed.

    See also
    ========

    Domain
    """

    rep = 'ZZ'
    alias = 'ZZ'
    dtype = MPZ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)


    is_IntegerRing = is_ZZ = True
    is_Numerical = True
    is_PID = True

    has_assoc_Ring = True
    has_assoc_Field = True

    def __init__(self):
        """Allow instantiation of this domain. """

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, IntegerRing):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute a hash value for this domain. """
        return hash('ZZ')

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(int(a))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return MPZ(a.p)
        elif int_valued(a):
            return MPZ(int(a))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def get_field(self):
        r"""Return the associated field of fractions :ref:`QQ`

        Returns
        =======

        :ref:`QQ`:
            The associated field of fractions :ref:`QQ`, a
            :py:class:`~.Domain` representing the rational numbers
            `\mathbb{Q}`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> ZZ.get_field()
        QQ
        """
        from sympy.polys.domains import QQ
        return QQ

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`.

        Parameters
        ==========

        *extension : One or more :py:class:`~.Expr`.
            Generators of the extension. These should be expressions that are
            algebraic over `\mathbb{Q}`.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            If provided, this will be used as the alias symbol for the
            primitive element of the returned :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicField`
            A :py:class:`~.Domain` representing the algebraic field extension.

        Examples
        ========

        >>> from sympy import ZZ, sqrt
        >>> ZZ.algebraic_field(sqrt(2))
        QQ<sqrt(2)>
        """
        return self.get_field().algebraic_field(*extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        """Convert a :py:class:`~.ANP` object to :ref:`ZZ`.

        See :py:meth:`~.Domain.convert`.
        """
        if a.is_ground:
            return K1.convert(a.LC(), K0.dom)

    def log(self, a, b):
        r"""Logarithm of *a* to the base *b*.

        Parameters
        ==========

        a: number
        b: number

        Returns
        =======

        $\\lfloor\log(a, b)\\rfloor$:
            Floor of the logarithm of *a* to the base *b*

        Examples
        ========

        >>> from sympy import ZZ
        >>> ZZ.log(ZZ(8), ZZ(2))
        3
        >>> ZZ.log(ZZ(9), ZZ(2))
        3

        Notes
        =====

        This function uses ``math.log`` which is based on ``float`` so it will
        fail for large integer arguments.
        """
        return self.dtype(int(math.log(int(a), b)))

    def from_FF(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        return MPZ(K0.to_int(a))

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        return MPZ(K0.to_int(a))

    def from_ZZ(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        return MPZ(a)

    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        return MPZ(a)

    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to GMPY's ``mpz``. """
        return MPZ(K0.to_int(a))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpz`` to GMPY's ``mpz``. """
        return a

    def from_QQ_gmpy(K1, a, K0):
        """Convert GMPY ``mpq`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return a.numerator

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to GMPY's ``mpz``. """
        p, q = K0.to_rational(a)

        if q == 1:
            # XXX: If MPZ is flint.fmpz and p is a gmpy2.mpz, then we need
            # to convert via int because fmpz and mpz do not know about each
            # other.
            return MPZ(int(p))

    def from_GaussianIntegerRing(K1, a, K0):
        if a.y == 0:
            return a.x

    def from_EX(K1, a, K0):
        """Convert ``Expression`` to GMPY's ``mpz``. """
        if a.is_Integer:
            return K1.from_sympy(a)

    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        h, s, t = gcdex(a, b)
        # XXX: This conditional logic should be handled somewhere else.
        if GROUND_TYPES == 'gmpy':
            return s, t, h
        else:
            return h, s, t

    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        return gcd(a, b)

    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        return lcm(a, b)

    def sqrt(self, a):
        """Compute square root of ``a``. """
        return sqrt(a)

    def is_square(self, a):
        """Return ``True`` if ``a`` is a square.

        Explanation
        ===========
        An integer is a square if and only if there exists an integer
        ``b`` such that ``b * b == a``.
        """
        return is_square(a)

    def exsqrt(self, a):
        """Non-negative square root of ``a`` if ``a`` is a square.

        See also
        ========
        is_square
        """
        if a < 0:
            return None
        root, rem = sqrtrem(a)
        if rem != 0:
            return None
        return root

    def factorial(self, a):
        """Compute factorial of ``a``. """
        return factorial(a)


ZZ = IntegerRing()
