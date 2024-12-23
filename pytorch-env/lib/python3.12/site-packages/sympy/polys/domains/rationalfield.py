"""Implementation of :class:`RationalField` class. """


from sympy.external.gmpy import MPQ

from sympy.polys.domains.groundtypes import SymPyRational, is_square, sqrtrem

from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class RationalField(Field, CharacteristicZero, SimpleDomain):
    r"""Abstract base class for the domain :ref:`QQ`.

    The :py:class:`RationalField` class represents the field of rational
    numbers $\mathbb{Q}$ as a :py:class:`~.Domain` in the domain system.
    :py:class:`RationalField` is a superclass of
    :py:class:`PythonRationalField` and :py:class:`GMPYRationalField` one of
    which will be the implementation for :ref:`QQ` depending on whether either
    of ``gmpy`` or ``gmpy2`` is installed or not.

    See also
    ========

    Domain
    """

    rep = 'QQ'
    alias = 'QQ'

    is_RationalField = is_QQ = True
    is_Numerical = True

    has_assoc_Ring = True
    has_assoc_Field = True

    dtype = MPQ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)

    def __init__(self):
        pass

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, RationalField):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Returns hash code of ``self``. """
        return hash('QQ')

    def get_ring(self):
        """Returns ring associated with ``self``. """
        from sympy.polys.domains import ZZ
        return ZZ

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyRational(int(a.numerator), int(a.denominator))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Rational:
            return MPQ(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            return MPQ(*map(int, RR.to_rational(a)))
        else:
            raise CoercionFailed("expected `Rational` object, got %s" % a)

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`.

        Parameters
        ==========

        *extension : One or more :py:class:`~.Expr`
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

        >>> from sympy import QQ, sqrt
        >>> QQ.algebraic_field(sqrt(2))
        QQ<sqrt(2)>
        """
        from sympy.polys.domains import AlgebraicField
        return AlgebraicField(self, *extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        """Convert a :py:class:`~.ANP` object to :ref:`QQ`.

        See :py:meth:`~.Domain.convert`
        """
        if a.is_ground:
            return K1.convert(a.LC(), K0.dom)

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return MPQ(a)

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return MPQ(a)

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return MPQ(a.numerator, a.denominator)

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return MPQ(a.numerator, a.denominator)

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return MPQ(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianElement`` object to ``dtype``. """
        if a.y == 0:
            return MPQ(a.x)

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return MPQ(*map(int, K0.to_rational(a)))

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__truediv__``.  """
        return MPQ(a) / MPQ(b)

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__truediv__``. """
        return MPQ(a) / MPQ(b)

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies nothing.  """
        return self.zero

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__truediv__``. """
        return MPQ(a) / MPQ(b), self.zero

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numerator

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denominator

    def is_square(self, a):
        """Return ``True`` if ``a`` is a square.

        Explanation
        ===========
        A rational number is a square if and only if there exists
        a rational number ``b`` such that ``b * b == a``.
        """
        return is_square(a.numerator) and is_square(a.denominator)

    def exsqrt(self, a):
        """Non-negative square root of ``a`` if ``a`` is a square.

        See also
        ========
        is_square
        """
        if a.numerator < 0:  # denominator is always positive
            return None
        p_sqrt, p_rem = sqrtrem(a.numerator)
        if p_rem != 0:
            return None
        q_sqrt, q_rem = sqrtrem(a.denominator)
        if q_rem != 0:
            return None
        return MPQ(p_sqrt, q_sqrt)

QQ = RationalField()
