"""Implementation of :class:`RealField` class. """


from sympy.external.gmpy import SYMPY_INTS
from sympy.core.numbers import Float
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.mpelements import MPContext
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class RealField(Field, CharacteristicZero, SimpleDomain):
    """Real numbers up to the given precision. """

    rep = 'RR'

    is_RealField = is_RR = True

    is_Exact = False
    is_Numerical = True
    is_PID = False

    has_assoc_Ring = False
    has_assoc_Field = True

    _default_precision = 53

    @property
    def has_default_precision(self):
        return self.precision == self._default_precision

    @property
    def precision(self):
        return self._context.prec

    @property
    def dps(self):
        return self._context.dps

    @property
    def tolerance(self):
        return self._context.tolerance

    def __init__(self, prec=_default_precision, dps=None, tol=None):
        context = MPContext(prec, dps, tol, True)
        context._parent = self
        self._context = context

        self._dtype = context.mpf
        self.zero = self.dtype(0)
        self.one = self.dtype(1)

    @property
    def tp(self):
        # XXX: Domain treats tp as an alis of dtype. Here we need to two
        # separate things: dtype is a callable to make/convert instances.
        # We use tp with isinstance to check if an object is an instance
        # of the domain already.
        return self._dtype

    def dtype(self, arg):
        # XXX: This is needed because mpmath does not recognise fmpz.
        # It might be better to add conversion routines to mpmath and if that
        # happens then this can be removed.
        if isinstance(arg, SYMPY_INTS):
            arg = int(arg)
        return self._dtype(arg)

    def __eq__(self, other):
        return (isinstance(other, RealField)
           and self.precision == other.precision
           and self.tolerance == other.tolerance)

    def __hash__(self):
        return hash((self.__class__.__name__, self._dtype, self.precision, self.tolerance))

    def to_sympy(self, element):
        """Convert ``element`` to SymPy number. """
        return Float(element, self.dps)

    def from_sympy(self, expr):
        """Convert SymPy's number to ``dtype``. """
        number = expr.evalf(n=self.dps)

        if number.is_Number:
            return self.dtype(number)
        else:
            raise CoercionFailed("expected real number, got %s" % expr)

    def from_ZZ(self, element, base):
        return self.dtype(element)

    def from_ZZ_python(self, element, base):
        return self.dtype(element)

    def from_ZZ_gmpy(self, element, base):
        return self.dtype(int(element))

    # XXX: We need to convert the denominators to int here because mpmath does
    # not recognise mpz. Ideally mpmath would handle this and if it changed to
    # do so then the calls to int here could be removed.

    def from_QQ(self, element, base):
        return self.dtype(element.numerator) / int(element.denominator)

    def from_QQ_python(self, element, base):
        return self.dtype(element.numerator) / int(element.denominator)

    def from_QQ_gmpy(self, element, base):
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_AlgebraicField(self, element, base):
        return self.from_sympy(base.to_sympy(element).evalf(self.dps))

    def from_RealField(self, element, base):
        if self == base:
            return element
        else:
            return self.dtype(element)

    def from_ComplexField(self, element, base):
        if not element.imag:
            return self.dtype(element.real)

    def to_rational(self, element, limit=True):
        """Convert a real number to rational number. """
        return self._context.to_rational(element, limit)

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self

    def get_exact(self):
        """Returns an exact domain associated with ``self``. """
        from sympy.polys.domains import QQ
        return QQ

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        return self.one

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        return a*b

    def almosteq(self, a, b, tolerance=None):
        """Check if ``a`` and ``b`` are almost equal. """
        return self._context.almosteq(a, b, tolerance)

    def is_square(self, a):
        """Returns ``True`` if ``a >= 0`` and ``False`` otherwise. """
        return a >= 0

    def exsqrt(self, a):
        """Non-negative square root for ``a >= 0`` and ``None`` otherwise.

        Explanation
        ===========
        The square root may be slightly inaccurate due to floating point
        rounding error.
        """
        return a ** 0.5 if a >= 0 else None


RR = RealField()
