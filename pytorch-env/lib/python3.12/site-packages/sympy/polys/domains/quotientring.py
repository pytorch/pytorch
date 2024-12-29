"""Implementation of :class:`QuotientRing` class."""


from sympy.polys.agca.modules import FreeModuleQuotientRing
from sympy.polys.domains.ring import Ring
from sympy.polys.polyerrors import NotReversible, CoercionFailed
from sympy.utilities import public

# TODO
# - successive quotients (when quotient ideals are implemented)
# - poly rings over quotients?
# - division by non-units in integral domains?

@public
class QuotientRingElement:
    """
    Class representing elements of (commutative) quotient rings.

    Attributes:

    - ring - containing ring
    - data - element of ring.ring (i.e. base ring) representing self
    """

    def __init__(self, ring, data):
        self.ring = ring
        self.data = data

    def __str__(self):
        from sympy.printing.str import sstr
        data = self.ring.ring.to_sympy(self.data)
        return sstr(data) + " + " + str(self.ring.base_ideal)

    __repr__ = __str__

    def __bool__(self):
        return not self.ring.is_zero(self)

    def __add__(self, om):
        if not isinstance(om, self.__class__) or om.ring != self.ring:
            try:
                om = self.ring.convert(om)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring(self.data + om.data)

    __radd__ = __add__

    def __neg__(self):
        return self.ring(self.data*self.ring.ring.convert(-1))

    def __sub__(self, om):
        return self.__add__(-om)

    def __rsub__(self, om):
        return (-self).__add__(om)

    def __mul__(self, o):
        if not isinstance(o, self.__class__):
            try:
                o = self.ring.convert(o)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring(self.data*o.data)

    __rmul__ = __mul__

    def __rtruediv__(self, o):
        return self.ring.revert(self)*o

    def __truediv__(self, o):
        if not isinstance(o, self.__class__):
            try:
                o = self.ring.convert(o)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring.revert(o)*self

    def __pow__(self, oth):
        if oth < 0:
            return self.ring.revert(self) ** -oth
        return self.ring(self.data ** oth)

    def __eq__(self, om):
        if not isinstance(om, self.__class__) or om.ring != self.ring:
            return False
        return self.ring.is_zero(self - om)

    def __ne__(self, om):
        return not self == om


class QuotientRing(Ring):
    """
    Class representing (commutative) quotient rings.

    You should not usually instantiate this by hand, instead use the constructor
    from the base ring in the construction.

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> I = QQ.old_poly_ring(x).ideal(x**3 + 1)
    >>> QQ.old_poly_ring(x).quotient_ring(I)
    QQ[x]/<x**3 + 1>

    Shorter versions are possible:

    >>> QQ.old_poly_ring(x)/I
    QQ[x]/<x**3 + 1>

    >>> QQ.old_poly_ring(x)/[x**3 + 1]
    QQ[x]/<x**3 + 1>

    Attributes:

    - ring - the base ring
    - base_ideal - the ideal used to form the quotient
    """

    has_assoc_Ring = True
    has_assoc_Field = False
    dtype = QuotientRingElement

    def __init__(self, ring, ideal):
        if not ideal.ring == ring:
            raise ValueError('Ideal must belong to %s, got %s' % (ring, ideal))
        self.ring = ring
        self.base_ideal = ideal
        self.zero = self(self.ring.zero)
        self.one = self(self.ring.one)

    def __str__(self):
        return str(self.ring) + "/" + str(self.base_ideal)

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.ring, self.base_ideal))

    def new(self, a):
        """Construct an element of ``self`` domain from ``a``. """
        if not isinstance(a, self.ring.dtype):
            a = self.ring(a)
        # TODO optionally disable reduction?
        return self.dtype(self, self.base_ideal.reduce_element(a))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, QuotientRing) and \
            self.ring == other.ring and self.base_ideal == other.base_ideal

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.ring.convert(a, K0))

    from_ZZ_python = from_ZZ
    from_QQ_python = from_ZZ_python
    from_ZZ_gmpy = from_ZZ_python
    from_QQ_gmpy = from_ZZ_python
    from_RealField = from_ZZ_python
    from_GlobalPolynomialRing = from_ZZ_python
    from_FractionField = from_ZZ_python

    def from_sympy(self, a):
        return self(self.ring.from_sympy(a))

    def to_sympy(self, a):
        return self.ring.to_sympy(a.data)

    def from_QuotientRing(self, a, K0):
        if K0 == self:
            return a

    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. ``K[X]``. """
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        """Returns a fraction field, i.e. ``K(X)``. """
        raise NotImplementedError('nested domains not allowed')

    def revert(self, a):
        """
        Compute a**(-1), if possible.
        """
        I = self.ring.ideal(a.data) + self.base_ideal
        try:
            return self(I.in_terms_of_generators(1)[0])
        except ValueError:  # 1 not in I
            raise NotReversible('%s not a unit in %r' % (a, self))

    def is_zero(self, a):
        return self.base_ideal.contains(a.data)

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        (QQ[x]/<x**2 + 1>)**2
        """
        return FreeModuleQuotientRing(self, rank)
