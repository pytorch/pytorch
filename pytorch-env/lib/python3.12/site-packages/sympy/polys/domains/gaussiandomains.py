"""Domains of Gaussian type."""

from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring


class GaussianElement(DomainElement):
    """Base class for elements of Gaussian type domains."""
    base: Domain
    _parent: Domain

    __slots__ = ('x', 'y')

    def __new__(cls, x, y=0):
        conv = cls.base.convert
        return cls.new(conv(x), conv(y))

    @classmethod
    def new(cls, x, y):
        """Create a new GaussianElement of the same domain."""
        obj = super().__new__(cls)
        obj.x = x
        obj.y = y
        return obj

    def parent(self):
        """The domain that this is an element of (ZZ_I or QQ_I)"""
        return self._parent

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        else:
            return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, GaussianElement):
            return NotImplemented
        return [self.y, self.x] < [other.y, other.x]

    def __pos__(self):
        return self

    def __neg__(self):
        return self.new(-self.x, -self.y)

    def __repr__(self):
        return "%s(%s, %s)" % (self._parent.rep, self.x, self.y)

    def __str__(self):
        return str(self._parent.to_sympy(self))

    @classmethod
    def _get_xy(cls, other):
        if not isinstance(other, cls):
            try:
                other = cls._parent.convert(other)
            except CoercionFailed:
                return None, None
        return other.x, other.y

    def __add__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x + x, self.y + y)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x - x, self.y - y)
        else:
            return NotImplemented

    def __rsub__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(x - self.x, y - self.y)
        else:
            return NotImplemented

    def __mul__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x*x - self.y*y, self.x*y + self.y*x)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __pow__(self, exp):
        if exp == 0:
            return self.new(1, 0)
        if exp < 0:
            self, exp = 1/self, -exp
        if exp == 1:
            return self
        pow2 = self
        prod = self if exp % 2 else self._parent.one
        exp //= 2
        while exp:
            pow2 *= pow2
            if exp % 2:
                prod *= pow2
            exp //= 2
        return prod

    def __bool__(self):
        return bool(self.x) or bool(self.y)

    def quadrant(self):
        """Return quadrant index 0-3.

        0 is included in quadrant 0.
        """
        if self.y > 0:
            return 0 if self.x > 0 else 1
        elif self.y < 0:
            return 2 if self.x < 0 else 3
        else:
            return 0 if self.x >= 0 else 2

    def __rdivmod__(self, other):
        try:
            other = self._parent.convert(other)
        except CoercionFailed:
            return NotImplemented
        else:
            return other.__divmod__(self)

    def __rtruediv__(self, other):
        try:
            other = QQ_I.convert(other)
        except CoercionFailed:
            return NotImplemented
        else:
            return other.__truediv__(self)

    def __floordiv__(self, other):
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[0]

    def __rfloordiv__(self, other):
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[0]

    def __mod__(self, other):
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[1]

    def __rmod__(self, other):
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[1]


class GaussianInteger(GaussianElement):
    """Gaussian integer: domain element for :ref:`ZZ_I`

        >>> from sympy import ZZ_I
        >>> z = ZZ_I(2, 3)
        >>> z
        (2 + 3*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianInteger'>
    """
    base = ZZ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        return QQ_I.convert(self)/other

    def __divmod__(self, other):
        if not other:
            raise ZeroDivisionError('divmod({}, 0)'.format(self))
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented

        # multiply self and other by x - I*y
        # self/other == (a + I*b)/c
        a, b = self.x*x + self.y*y, -self.x*y + self.y*x
        c = x*x + y*y

        # find integers qx and qy such that
        # |a - qx*c| <= c/2 and |b - qy*c| <= c/2
        qx = (2*a + c) // (2*c)  # -c <= 2*a - qx*2*c < c
        qy = (2*b + c) // (2*c)

        q = GaussianInteger(qx, qy)
        # |self/other - q| < 1 since
        # |a/c - qx|**2 + |b/c - qy|**2 <= 1/4 + 1/4 < 1

        return q, self - q*other  # |r| < |other|


class GaussianRational(GaussianElement):
    """Gaussian rational: domain element for :ref:`QQ_I`

        >>> from sympy import QQ_I, QQ
        >>> z = QQ_I(QQ(2, 3), QQ(4, 5))
        >>> z
        (2/3 + 4/5*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianRational'>
    """
    base = QQ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        if not other:
            raise ZeroDivisionError('{} / 0'.format(self))
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented
        c = x*x + y*y

        return GaussianRational((self.x*x + self.y*y)/c,
                                (-self.x*y + self.y*x)/c)

    def __divmod__(self, other):
        try:
            other = self._parent.convert(other)
        except CoercionFailed:
            return NotImplemented
        if not other:
            raise ZeroDivisionError('{} % 0'.format(self))
        else:
            return self/other, QQ_I.zero


class GaussianDomain():
    """Base class for Gaussian domains."""
    dom = None  # type: Domain

    is_Numerical = True
    is_Exact = True

    has_assoc_Ring = True
    has_assoc_Field = True

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        conv = self.dom.to_sympy
        return conv(a.x) + I*conv(a.y)

    def from_sympy(self, a):
        """Convert a SymPy object to ``self.dtype``."""
        r, b = a.as_coeff_Add()
        x = self.dom.from_sympy(r)  # may raise CoercionFailed
        if not b:
            return self.new(x, 0)
        r, b = b.as_coeff_Mul()
        y = self.dom.from_sympy(r)
        if b is I:
            return self.new(x, y)
        else:
            raise CoercionFailed("{} is not Gaussian".format(a))

    def inject(self, *gens):
        """Inject generators into this domain. """
        return self.poly_ring(*gens)

    def canonical_unit(self, d):
        unit = self.units[-d.quadrant()]  # - for inverse power
        return unit

    def is_negative(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_positive(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_nonnegative(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_nonpositive(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY mpz to ``self.dtype``."""
        return K1(a)

    def from_ZZ(K1, a, K0):
        """Convert a ZZ_python element to ``self.dtype``."""
        return K1(a)

    def from_ZZ_python(K1, a, K0):
        """Convert a ZZ_python element to ``self.dtype``."""
        return K1(a)

    def from_QQ(K1, a, K0):
        """Convert a GMPY mpq to ``self.dtype``."""
        return K1(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY mpq to ``self.dtype``."""
        return K1(a)

    def from_QQ_python(K1, a, K0):
        """Convert a QQ_python element to ``self.dtype``."""
        return K1(a)

    def from_AlgebraicField(K1, a, K0):
        """Convert an element from ZZ<I> or QQ<I> to ``self.dtype``."""
        if K0.ext.args[0] == I:
            return K1.from_sympy(K0.to_sympy(a))


class GaussianIntegerRing(GaussianDomain, Ring):
    r"""Ring of Gaussian integers ``ZZ_I``

    The :ref:`ZZ_I` domain represents the `Gaussian integers`_ `\mathbb{Z}[i]`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of integers and ``I`` (`\sqrt{-1}`)
    will have the domain :ref:`ZZ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I)
    >>> p
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> p.domain
    ZZ_I

    The :ref:`ZZ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian integers.

    >>> from sympy import factor
    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, domain='ZZ_I')
    (x - I)*(x + I)

    The corresponding `field of fractions`_ is the domain of the Gaussian
    rationals :ref:`QQ_I`. Conversely :ref:`ZZ_I` is the `ring of integers`_
    of :ref:`QQ_I`.

    >>> from sympy import ZZ_I, QQ_I
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`ZZ_I` can be used as a constructor.

    >>> ZZ_I(3, 4)
    (3 + 4*I)
    >>> ZZ_I(5)
    (5 + 0*I)

    The domain elements of :ref:`ZZ_I` are instances of
    :py:class:`~.GaussianInteger` which support the rings operations
    ``+,-,*,**``.

    >>> z1 = ZZ_I(5, 1)
    >>> z2 = ZZ_I(2, 3)
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 3*I)
    >>> z1 + z2
    (7 + 4*I)
    >>> z1 * z2
    (7 + 17*I)
    >>> z1 ** 2
    (24 + 10*I)

    Both floor (``//``) and modulo (``%``) division work with
    :py:class:`~.GaussianInteger` (see the :py:meth:`~.Domain.div` method).

    >>> z3, z4 = ZZ_I(5), ZZ_I(1, 3)
    >>> z3 // z4  # floor division
    (1 + -1*I)
    >>> z3 % z4   # modulo division (remainder)
    (1 + -2*I)
    >>> (z3//z4)*z4 + z3%z4 == z3
    True

    True division (``/``) in :ref:`ZZ_I` gives an element of :ref:`QQ_I`. The
    :py:meth:`~.Domain.exquo` method can be used to divide in :ref:`ZZ_I` when
    exact division is possible.

    >>> z1 / z2
    (1 + -1*I)
    >>> ZZ_I.exquo(z1, z2)
    (1 + -1*I)
    >>> z3 / z4
    (1/2 + -3/2*I)
    >>> ZZ_I.exquo(z3, z4)
    Traceback (most recent call last):
        ...
    ExactQuotientFailed: (1 + 3*I) does not divide (5 + 0*I) in ZZ_I

    The :py:meth:`~.Domain.gcd` method can be used to compute the `gcd`_ of any
    two elements.

    >>> ZZ_I.gcd(ZZ_I(10), ZZ_I(2))
    (2 + 0*I)
    >>> ZZ_I.gcd(ZZ_I(5), ZZ_I(2, 1))
    (2 + 1*I)

    .. _Gaussian integers: https://en.wikipedia.org/wiki/Gaussian_integer
    .. _gcd: https://en.wikipedia.org/wiki/Greatest_common_divisor

    """
    dom = ZZ
    dtype = GaussianInteger
    zero = dtype(ZZ(0), ZZ(0))
    one = dtype(ZZ(1), ZZ(0))
    imag_unit = dtype(ZZ(0), ZZ(1))
    units = (one, imag_unit, -one, -imag_unit)  # powers of i

    rep = 'ZZ_I'

    is_GaussianRing = True
    is_ZZ_I = True

    def __init__(self):  # override Domain.__init__
        """For constructing ZZ_I."""

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, GaussianIntegerRing):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute hash code of ``self``. """
        return hash('ZZ_I')

    @property
    def has_CharacteristicZero(self):
        return True

    def characteristic(self):
        return 0

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self

    def get_field(self):
        """Returns a field associated with ``self``. """
        return QQ_I

    def normalize(self, d, *args):
        """Return first quadrant element associated with ``d``.

        Also multiply the other arguments by the same power of i.
        """
        unit = self.canonical_unit(d)
        d *= unit
        args = tuple(a*unit for a in args)
        return (d,) + args if args else d

    def gcd(self, a, b):
        """Greatest common divisor of a and b over ZZ_I."""
        while b:
            a, b = b, a % b
        return self.normalize(a)

    def lcm(self, a, b):
        """Least common multiple of a and b over ZZ_I."""
        return (a * b) // self.gcd(a, b)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to ZZ_I."""
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to ZZ_I."""
        return K1.new(ZZ.convert(a.x), ZZ.convert(a.y))

ZZ_I = GaussianInteger._parent = GaussianIntegerRing()


class GaussianRationalField(GaussianDomain, Field):
    r"""Field of Gaussian rationals ``QQ_I``

    The :ref:`QQ_I` domain represents the `Gaussian rationals`_ `\mathbb{Q}(i)`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of rationals and ``I`` (`\sqrt{-1}`)
    will have the domain :ref:`QQ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I/2)
    >>> p
    Poly(x**2 + I/2, x, domain='QQ_I')
    >>> p.domain
    QQ_I

    The polys option ``gaussian=True`` can be used to specify that the domain
    should be :ref:`QQ_I` even if the coefficients do not contain ``I`` or are
    all integers.

    >>> Poly(x**2)
    Poly(x**2, x, domain='ZZ')
    >>> Poly(x**2 + I)
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> Poly(x**2/2)
    Poly(1/2*x**2, x, domain='QQ')
    >>> Poly(x**2, gaussian=True)
    Poly(x**2, x, domain='QQ_I')
    >>> Poly(x**2 + I, gaussian=True)
    Poly(x**2 + I, x, domain='QQ_I')
    >>> Poly(x**2/2, gaussian=True)
    Poly(1/2*x**2, x, domain='QQ_I')

    The :ref:`QQ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian rationals.

    >>> from sympy import factor, QQ_I
    >>> factor(x**2/4 + 1)
    (x**2 + 4)/4
    >>> factor(x**2/4 + 1, domain='QQ_I')
    (x - 2*I)*(x + 2*I)/4
    >>> factor(x**2/4 + 1, domain=QQ_I)
    (x - 2*I)*(x + 2*I)/4

    It is also possible to specify the :ref:`QQ_I` domain explicitly with
    polys functions like :py:func:`~.apart`.

    >>> from sympy import apart
    >>> apart(1/(1 + x**2))
    1/(x**2 + 1)
    >>> apart(1/(1 + x**2), domain=QQ_I)
    I/(2*(x + I)) - I/(2*(x - I))

    The corresponding `ring of integers`_ is the domain of the Gaussian
    integers :ref:`ZZ_I`. Conversely :ref:`QQ_I` is the `field of fractions`_
    of :ref:`ZZ_I`.

    >>> from sympy import ZZ_I, QQ_I, QQ
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`QQ_I` can be used as a constructor.

    >>> QQ_I(3, 4)
    (3 + 4*I)
    >>> QQ_I(5)
    (5 + 0*I)
    >>> QQ_I(QQ(2, 3), QQ(4, 5))
    (2/3 + 4/5*I)

    The domain elements of :ref:`QQ_I` are instances of
    :py:class:`~.GaussianRational` which support the field operations
    ``+,-,*,**,/``.

    >>> z1 = QQ_I(5, 1)
    >>> z2 = QQ_I(2, QQ(1, 2))
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 1/2*I)
    >>> z1 + z2
    (7 + 3/2*I)
    >>> z1 * z2
    (19/2 + 9/2*I)
    >>> z2 ** 2
    (15/4 + 2*I)

    True division (``/``) in :ref:`QQ_I` gives an element of :ref:`QQ_I` and
    is always exact.

    >>> z1 / z2
    (42/17 + -2/17*I)
    >>> QQ_I.exquo(z1, z2)
    (42/17 + -2/17*I)
    >>> z1 == (z1/z2)*z2
    True

    Both floor (``//``) and modulo (``%``) division can be used with
    :py:class:`~.GaussianRational` (see :py:meth:`~.Domain.div`)
    but division is always exact so there is no remainder.

    >>> z1 // z2
    (42/17 + -2/17*I)
    >>> z1 % z2
    (0 + 0*I)
    >>> QQ_I.div(z1, z2)
    ((42/17 + -2/17*I), (0 + 0*I))
    >>> (z1//z2)*z2 + z1%z2 == z1
    True

    .. _Gaussian rationals: https://en.wikipedia.org/wiki/Gaussian_rational
    """
    dom = QQ
    dtype = GaussianRational
    zero = dtype(QQ(0), QQ(0))
    one = dtype(QQ(1), QQ(0))
    imag_unit = dtype(QQ(0), QQ(1))
    units = (one, imag_unit, -one, -imag_unit)  # powers of i

    rep = 'QQ_I'

    is_GaussianField = True
    is_QQ_I = True

    def __init__(self):  # override Domain.__init__
        """For constructing QQ_I."""

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, GaussianRationalField):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute hash code of ``self``. """
        return hash('QQ_I')

    @property
    def has_CharacteristicZero(self):
        return True

    def characteristic(self):
        return 0

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return ZZ_I

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def as_AlgebraicField(self):
        """Get equivalent domain as an ``AlgebraicField``. """
        return AlgebraicField(self.dom, I)

    def numer(self, a):
        """Get the numerator of ``a``."""
        ZZ_I = self.get_ring()
        return ZZ_I.convert(a * self.denom(a))

    def denom(self, a):
        """Get the denominator of ``a``."""
        ZZ = self.dom.get_ring()
        QQ = self.dom
        ZZ_I = self.get_ring()
        denom_ZZ = ZZ.lcm(QQ.denom(a.x), QQ.denom(a.y))
        return ZZ_I(denom_ZZ, ZZ.zero)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to QQ_I."""
        return K1.new(a.x, a.y)

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to QQ_I."""
        return a

    def from_ComplexField(K1, a, K0):
        """Convert a ComplexField element to QQ_I."""
        return K1.new(QQ.convert(a.real), QQ.convert(a.imag))


QQ_I = GaussianRational._parent = GaussianRationalField()
