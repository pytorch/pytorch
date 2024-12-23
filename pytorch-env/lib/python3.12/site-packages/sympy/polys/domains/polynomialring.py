"""Implementation of :class:`PolynomialRing` class. """


from sympy.polys.domains.ring import Ring
from sympy.polys.domains.compositedomain import CompositeDomain

from sympy.polys.polyerrors import CoercionFailed, GeneratorsError
from sympy.utilities import public

@public
class PolynomialRing(Ring, CompositeDomain):
    """A class for representing multivariate polynomial rings. """

    is_PolynomialRing = is_Poly = True

    has_assoc_Ring  = True
    has_assoc_Field = True

    def __init__(self, domain_or_ring, symbols=None, order=None):
        from sympy.polys.rings import PolyRing

        if isinstance(domain_or_ring, PolyRing) and symbols is None and order is None:
            ring = domain_or_ring
        else:
            ring = PolyRing(symbols, domain_or_ring, order)

        self.ring = ring
        self.dtype = ring.dtype

        self.gens = ring.gens
        self.ngens = ring.ngens
        self.symbols = ring.symbols
        self.domain = ring.domain


        if symbols:
            if ring.domain.is_Field and ring.domain.is_Exact and len(symbols)==1:
                self.is_PID = True

        # TODO: remove this
        self.dom = self.domain

    def new(self, element):
        return self.ring.ring_new(element)

    @property
    def zero(self):
        return self.ring.zero

    @property
    def one(self):
        return self.ring.one

    @property
    def order(self):
        return self.ring.order

    def __str__(self):
        return str(self.domain) + '[' + ','.join(map(str, self.symbols)) + ']'

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype.ring, self.domain, self.symbols))

    def __eq__(self, other):
        """Returns `True` if two domains are equivalent. """
        return isinstance(other, PolynomialRing) and \
            (self.dtype.ring, self.domain, self.symbols) == \
            (other.dtype.ring, other.domain, other.symbols)

    def is_unit(self, a):
        """Returns ``True`` if ``a`` is a unit of ``self``"""
        if not a.is_ground:
            return False
        K = self.domain
        return K.is_unit(K.convert_from(a, self))

    def canonical_unit(self, a):
        u = self.domain.canonical_unit(a.LC)
        return self.ring.ground_new(u)

    def to_sympy(self, a):
        """Convert `a` to a SymPy object. """
        return a.as_expr()

    def from_sympy(self, a):
        """Convert SymPy's expression to `dtype`. """
        return self.ring.from_expr(a)

    def from_ZZ(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY `mpz` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY `mpq` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a `GaussianInteger` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a `GaussianRational` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_ComplexField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert an algebraic number to ``dtype``. """
        if K1.domain != K0:
            a = K1.domain.convert_from(a, K0)
        if a is not None:
            return K1.new(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        try:
            return a.set_ring(K1.ring)
        except (CoercionFailed, GeneratorsError):
            return None

    def from_FractionField(K1, a, K0):
        """Convert a rational function to ``dtype``. """
        if K1.domain == K0:
            return K1.ring.from_list([a])

        q, r = K0.numer(a).div(K0.denom(a))

        if r.is_zero:
            return K1.from_PolynomialRing(q, K0.field.ring.to_domain())
        else:
            return None

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert from old poly ring to ``dtype``. """
        if K1.symbols == K0.gens:
            ad = a.to_dict()
            if K1.domain != K0.domain:
                ad = {m: K1.domain.convert(c) for m, c in ad.items()}
            return K1(ad)
        elif a.is_ground and K0.domain == K1:
            return K1.convert_from(a.to_list()[0], K0.domain)

    def get_field(self):
        """Returns a field associated with `self`. """
        return self.ring.to_field().to_domain()

    def is_positive(self, a):
        """Returns True if `LC(a)` is positive. """
        return self.domain.is_positive(a.LC)

    def is_negative(self, a):
        """Returns True if `LC(a)` is negative. """
        return self.domain.is_negative(a.LC)

    def is_nonpositive(self, a):
        """Returns True if `LC(a)` is non-positive. """
        return self.domain.is_nonpositive(a.LC)

    def is_nonnegative(self, a):
        """Returns True if `LC(a)` is non-negative. """
        return self.domain.is_nonnegative(a.LC)

    def gcdex(self, a, b):
        """Extended GCD of `a` and `b`. """
        return a.gcdex(b)

    def gcd(self, a, b):
        """Returns GCD of `a` and `b`. """
        return a.gcd(b)

    def lcm(self, a, b):
        """Returns LCM of `a` and `b`. """
        return a.lcm(b)

    def factorial(self, a):
        """Returns factorial of `a`. """
        return self.dtype(self.domain.factorial(a))
