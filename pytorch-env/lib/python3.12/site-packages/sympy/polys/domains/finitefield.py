"""Implementation of :class:`FiniteField` class. """

import operator

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

from sympy.core.numbers import int_valued
from sympy.polys.domains.field import Field

from sympy.polys.domains.modularinteger import ModularIntegerFactory
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.galoistools import gf_zassenhaus, gf_irred_p_rabin
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
from sympy.polys.domains.groundtypes import SymPyInteger


if GROUND_TYPES == 'flint':
    __doctest_skip__ = ['FiniteField']


if GROUND_TYPES == 'flint':
    import flint
    # Don't use python-flint < 0.5.0 because nmod was missing some features in
    # previous versions of python-flint and fmpz_mod was not yet added.
    _major, _minor, *_ = flint.__version__.split('.')
    if (int(_major), int(_minor)) < (0, 5):
        flint = None
else:
    flint = None


def _modular_int_factory(mod, dom, symmetric, self):

    # Use flint if available
    if flint is not None:

        nmod = flint.nmod
        fmpz_mod_ctx = flint.fmpz_mod_ctx
        index = operator.index

        try:
            mod = dom.convert(mod)
        except CoercionFailed:
            raise ValueError('modulus must be an integer, got %s' % mod)

        # mod might be e.g. Integer
        try:
            fmpz_mod_ctx(mod)
        except TypeError:
            mod = index(mod)

        # flint's nmod is only for moduli up to 2^64-1 (on a 64-bit machine)
        try:
            nmod(0, mod)
        except OverflowError:
            # Use fmpz_mod
            fctx = fmpz_mod_ctx(mod)

            def ctx(x):
                try:
                    return fctx(x)
                except TypeError:
                    # x might be Integer
                    return fctx(index(x))
        else:
            # Use nmod
            def ctx(x):
                try:
                    return nmod(x, mod)
                except TypeError:
                    return nmod(index(x), mod)

        return ctx

    # Use the Python implementation
    return ModularIntegerFactory(mod, dom, symmetric, self)


@public
@doctest_depends_on(modules=['python', 'gmpy'])
class FiniteField(Field, SimpleDomain):
    r"""Finite field of prime order :ref:`GF(p)`

    A :ref:`GF(p)` domain represents a `finite field`_ `\mathbb{F}_p` of prime
    order as :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    A :py:class:`~.Poly` created from an expression with integer
    coefficients will have the domain :ref:`ZZ`. However, if the ``modulus=p``
    option is given then the domain will be a finite field instead.

    >>> from sympy import Poly, Symbol
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + 1)
    >>> p
    Poly(x**2 + 1, x, domain='ZZ')
    >>> p.domain
    ZZ
    >>> p2 = Poly(x**2 + 1, modulus=2)
    >>> p2
    Poly(x**2 + 1, x, modulus=2)
    >>> p2.domain
    GF(2)

    It is possible to factorise a polynomial over :ref:`GF(p)` using the
    modulus argument to :py:func:`~.factor` or by specifying the domain
    explicitly. The domain can also be given as a string.

    >>> from sympy import factor, GF
    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, modulus=2)
    (x + 1)**2
    >>> factor(x**2 + 1, domain=GF(2))
    (x + 1)**2
    >>> factor(x**2 + 1, domain='GF(2)')
    (x + 1)**2

    It is also possible to use :ref:`GF(p)` with the :py:func:`~.cancel`
    and :py:func:`~.gcd` functions.

    >>> from sympy import cancel, gcd
    >>> cancel((x**2 + 1)/(x + 1))
    (x**2 + 1)/(x + 1)
    >>> cancel((x**2 + 1)/(x + 1), domain=GF(2))
    x + 1
    >>> gcd(x**2 + 1, x + 1)
    1
    >>> gcd(x**2 + 1, x + 1, domain=GF(2))
    x + 1

    When using the domain directly :ref:`GF(p)` can be used as a constructor
    to create instances which then support the operations ``+,-,*,**,/``

    >>> from sympy import GF
    >>> K = GF(5)
    >>> K
    GF(5)
    >>> x = K(3)
    >>> y = K(2)
    >>> x
    3 mod 5
    >>> y
    2 mod 5
    >>> x * y
    1 mod 5
    >>> x / y
    4 mod 5

    Notes
    =====

    It is also possible to create a :ref:`GF(p)` domain of **non-prime**
    order but the resulting ring is **not** a field: it is just the ring of
    the integers modulo ``n``.

    >>> K = GF(9)
    >>> z = K(3)
    >>> z
    3 mod 9
    >>> z**2
    0 mod 9

    It would be good to have a proper implementation of prime power fields
    (``GF(p**n)``) but these are not yet implemented in SymPY.

    .. _finite field: https://en.wikipedia.org/wiki/Finite_field
    """

    rep = 'FF'
    alias = 'FF'

    is_FiniteField = is_FF = True
    is_Numerical = True

    has_assoc_Ring = False
    has_assoc_Field = True

    dom = None
    mod = None

    def __init__(self, mod, symmetric=True):
        from sympy.polys.domains import ZZ
        dom = ZZ

        if mod <= 0:
            raise ValueError('modulus must be a positive integer, got %s' % mod)

        self.dtype = _modular_int_factory(mod, dom, symmetric, self)
        self.zero = self.dtype(0)
        self.one = self.dtype(1)
        self.dom = dom
        self.mod = mod
        self.sym = symmetric
        self._tp = type(self.zero)

    @property
    def tp(self):
        return self._tp

    def __str__(self):
        return 'GF(%s)' % self.mod

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.mod, self.dom))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, FiniteField) and \
            self.mod == other.mod and self.dom == other.dom

    def characteristic(self):
        """Return the characteristic of this domain. """
        return self.mod

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(self.to_int(a))

    def from_sympy(self, a):
        """Convert SymPy's Integer to SymPy's ``Integer``. """
        if a.is_Integer:
            return self.dtype(self.dom.dtype(int(a)))
        elif int_valued(a):
            return self.dtype(self.dom.dtype(int(a)))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def to_int(self, a):
        """Convert ``val`` to a Python ``int`` object. """
        aval = int(a)
        if self.sym and aval > self.mod // 2:
            aval -= self.mod
        return aval

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return bool(a)

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return True

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return False

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return not a

    def from_FF(K1, a, K0=None):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ(int(a), K0.dom))

    def from_FF_python(K1, a, K0=None):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_python(int(a), K0.dom))

    def from_ZZ(K1, a, K0=None):
        """Convert Python's ``int`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_python(a, K0))

    def from_ZZ_python(K1, a, K0=None):
        """Convert Python's ``int`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_python(a, K0))

    def from_QQ(K1, a, K0=None):
        """Convert Python's ``Fraction`` to ``dtype``. """
        if a.denominator == 1:
            return K1.from_ZZ_python(a.numerator)

    def from_QQ_python(K1, a, K0=None):
        """Convert Python's ``Fraction`` to ``dtype``. """
        if a.denominator == 1:
            return K1.from_ZZ_python(a.numerator)

    def from_FF_gmpy(K1, a, K0=None):
        """Convert ``ModularInteger(mpz)`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_gmpy(a.val, K0.dom))

    def from_ZZ_gmpy(K1, a, K0=None):
        """Convert GMPY's ``mpz`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_gmpy(a, K0))

    def from_QQ_gmpy(K1, a, K0=None):
        """Convert GMPY's ``mpq`` to ``dtype``. """
        if a.denominator == 1:
            return K1.from_ZZ_gmpy(a.numerator)

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to ``dtype``. """
        p, q = K0.to_rational(a)

        if q == 1:
            return K1.dtype(K1.dom.dtype(p))

    def is_square(self, a):
        """Returns True if ``a`` is a quadratic residue modulo p. """
        # a is not a square <=> x**2-a is irreducible
        poly = [int(x) for x in [self.one, self.zero, -a]]
        return not gf_irred_p_rabin(poly, self.mod, self.dom)

    def exsqrt(self, a):
        """Square root modulo p of ``a`` if it is a quadratic residue.

        Explanation
        ===========
        Always returns the square root that is no larger than ``p // 2``.
        """
        # x**2-a is not square-free if a=0 or the field is characteristic 2
        if self.mod == 2 or a == 0:
            return a
        # Otherwise, use square-free factorization routine to factorize x**2-a
        poly = [int(x) for x in [self.one, self.zero, -a]]
        for factor in gf_zassenhaus(poly, self.mod, self.dom):
            if len(factor) == 2 and factor[1] <= self.mod // 2:
                return self.dtype(factor[1])
        return None


FF = GF = FiniteField
