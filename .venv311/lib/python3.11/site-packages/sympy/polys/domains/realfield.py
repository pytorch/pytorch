"""Implementation of :class:`RealField` class. """


from sympy.external.gmpy import SYMPY_INTS, MPQ
from sympy.core.numbers import Float
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

from mpmath import MPContext
from mpmath.libmp import to_rational as _mpmath_to_rational


def to_rational(s, max_denom, limit=True):

    p, q = _mpmath_to_rational(s._mpf_)

    # Needed for GROUND_TYPES=flint if gmpy2 is installed because mpmath's
    # to_rational() function returns a gmpy2.mpz instance and if MPQ is
    # flint.fmpq then MPQ(p, q) will fail.
    p = int(p)
    q = int(q)

    if not limit or q <= max_denom:
        return p, q

    p0, q0, p1, q1 = 0, 1, 1, 0
    n, d = p, q

    while True:
        a = n//d
        q2 = q0 + a*q1
        if q2 > max_denom:
            break
        p0, q0, p1, q1 = p1, q1, p0 + a*p1, q2
        n, d = d, n - a*d

    k = (max_denom - q0)//q1

    number = MPQ(p, q)
    bound1 = MPQ(p0 + k*p1, q0 + k*q1)
    bound2 = MPQ(p1, q1)

    if not bound2 or not bound1:
        return p, q
    elif abs(bound2 - number) <= abs(bound1 - number):
        return bound2.numerator, bound2.denominator
    else:
        return bound1.numerator, bound1.denominator


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
        return self._tolerance

    def __init__(self, prec=None, dps=None, tol=None):
        # XXX: The tol parameter is ignored but is kept for now for backwards
        # compatibility.

        context = MPContext()

        if prec is None and dps is None:
            context.prec = self._default_precision
        elif dps is None:
            context.prec = prec
        elif prec is None:
            context.dps = dps
        else:
            raise TypeError("Cannot set both prec and dps")

        self._context = context

        self._dtype = context.mpf
        self.zero = self.dtype(0)
        self.one = self.dtype(1)

        # Only max_denom here is used for anything and is only used for
        # to_rational.
        self._max_denom = max(2**context.prec // 200, 99)
        self._tolerance = self.one / self._max_denom

    @property
    def tp(self):
        # XXX: Domain treats tp as an alias of dtype. Here we need to two
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
        return isinstance(other, RealField) and self.precision == other.precision

    def __hash__(self):
        return hash((self.__class__.__name__, self._dtype, self.precision))

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
        return self.dtype(element)

    def from_ComplexField(self, element, base):
        if not element.imag:
            return self.dtype(element.real)

    def to_rational(self, element, limit=True):
        """Convert a real number to rational number. """
        return to_rational(element, self._max_denom, limit=limit)

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
