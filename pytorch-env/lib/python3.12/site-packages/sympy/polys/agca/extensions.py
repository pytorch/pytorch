"""Finite extensions of ring domains."""

from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import (CoercionFailed, NotInvertible,
        GeneratorsError)
from sympy.polys.polytools import Poly
from sympy.printing.defaults import DefaultPrinting


class ExtensionElement(DomainElement, DefaultPrinting):
    """
    Element of a finite extension.

    A class of univariate polynomials modulo the ``modulus``
    of the extension ``ext``. It is represented by the
    unique polynomial ``rep`` of lowest degree. Both
    ``rep`` and the representation ``mod`` of ``modulus``
    are of class DMP.

    """
    __slots__ = ('rep', 'ext')

    def __init__(self, rep, ext):
        self.rep = rep
        self.ext = ext

    def parent(f):
        return f.ext

    def as_expr(f):
        return f.ext.to_sympy(f)

    def __bool__(f):
        return bool(f.rep)

    def __pos__(f):
        return f

    def __neg__(f):
        return ExtElem(-f.rep, f.ext)

    def _get_rep(f, g):
        if isinstance(g, ExtElem):
            if g.ext == f.ext:
                return g.rep
            else:
                return None
        else:
            try:
                g = f.ext.convert(g)
                return g.rep
            except CoercionFailed:
                return None

    def __add__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(f.rep + rep, f.ext)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(f.rep - rep, f.ext)
        else:
            return NotImplemented

    def __rsub__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(rep - f.rep, f.ext)
        else:
            return NotImplemented

    def __mul__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem((f.rep * rep) % f.ext.mod, f.ext)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def _divcheck(f):
        """Raise if division is not implemented for this divisor"""
        if not f:
            raise NotInvertible('Zero divisor')
        elif f.ext.is_Field:
            return True
        elif f.rep.is_ground and f.ext.domain.is_unit(f.rep.LC()):
            return True
        else:
            # Some cases like (2*x + 2)/2 over ZZ will fail here. It is
            # unclear how to implement division in general if the ground
            # domain is not a field so for now it was decided to restrict the
            # implementation to division by invertible constants.
            msg = (f"Can not invert {f} in {f.ext}. "
                    "Only division by invertible constants is implemented.")
            raise NotImplementedError(msg)

    def inverse(f):
        """Multiplicative inverse.

        Raises
        ======

        NotInvertible
            If the element is a zero divisor.

        """
        f._divcheck()

        if f.ext.is_Field:
            invrep = f.rep.invert(f.ext.mod)
        else:
            R = f.ext.ring
            invrep = R.exquo(R.one, f.rep)

        return ExtElem(invrep, f.ext)

    def __truediv__(f, g):
        rep = f._get_rep(g)
        if rep is None:
            return NotImplemented
        g = ExtElem(rep, f.ext)

        try:
            ginv = g.inverse()
        except NotInvertible:
            raise ZeroDivisionError(f"{f} / {g}")

        return f * ginv

    __floordiv__ = __truediv__

    def __rtruediv__(f, g):
        try:
            g = f.ext.convert(g)
        except CoercionFailed:
            return NotImplemented
        return g / f

    __rfloordiv__ = __rtruediv__

    def __mod__(f, g):
        rep = f._get_rep(g)
        if rep is None:
            return NotImplemented
        g = ExtElem(rep, f.ext)

        try:
            g._divcheck()
        except NotInvertible:
            raise ZeroDivisionError(f"{f} % {g}")

        # Division where defined is always exact so there is no remainder
        return f.ext.zero

    def __rmod__(f, g):
        try:
            g = f.ext.convert(g)
        except CoercionFailed:
            return NotImplemented
        return g % f

    def __pow__(f, n):
        if not isinstance(n, int):
            raise TypeError("exponent of type 'int' expected")
        if n < 0:
            try:
                f, n = f.inverse(), -n
            except NotImplementedError:
                raise ValueError("negative powers are not defined")

        b = f.rep
        m = f.ext.mod
        r = f.ext.one.rep
        while n > 0:
            if n % 2:
                r = (r*b) % m
            b = (b*b) % m
            n //= 2

        return ExtElem(r, f.ext)

    def __eq__(f, g):
        if isinstance(g, ExtElem):
            return f.rep == g.rep and f.ext == g.ext
        else:
            return NotImplemented

    def __ne__(f, g):
        return not f == g

    def __hash__(f):
        return hash((f.rep, f.ext))

    def __str__(f):
        from sympy.printing.str import sstr
        return sstr(f.as_expr())

    __repr__ = __str__

    @property
    def is_ground(f):
        return f.rep.is_ground

    def to_ground(f):
        [c] = f.rep.to_list()
        return c

ExtElem = ExtensionElement


class MonogenicFiniteExtension(Domain):
    r"""
    Finite extension generated by an integral element.

    The generator is defined by a monic univariate
    polynomial derived from the argument ``mod``.

    A shorter alias is ``FiniteExtension``.

    Examples
    ========

    Quadratic integer ring $\mathbb{Z}[\sqrt2]$:

    >>> from sympy import Symbol, Poly
    >>> from sympy.polys.agca.extensions import FiniteExtension
    >>> x = Symbol('x')
    >>> R = FiniteExtension(Poly(x**2 - 2)); R
    ZZ[x]/(x**2 - 2)
    >>> R.rank
    2
    >>> R(1 + x)*(3 - 2*x)
    x - 1

    Finite field $GF(5^3)$ defined by the primitive
    polynomial $x^3 + x^2 + 2$ (over $\mathbb{Z}_5$).

    >>> F = FiniteExtension(Poly(x**3 + x**2 + 2, modulus=5)); F
    GF(5)[x]/(x**3 + x**2 + 2)
    >>> F.basis
    (1, x, x**2)
    >>> F(x + 3)/(x**2 + 2)
    -2*x**2 + x + 2

    Function field of an elliptic curve:

    >>> t = Symbol('t')
    >>> FiniteExtension(Poly(t**2 - x**3 - x + 1, t, field=True))
    ZZ(x)[t]/(t**2 - x**3 - x + 1)

    """
    is_FiniteExtension = True

    dtype = ExtensionElement

    def __init__(self, mod):
        if not (isinstance(mod, Poly) and mod.is_univariate):
            raise TypeError("modulus must be a univariate Poly")

        # Using auto=True (default) potentially changes the ground domain to a
        # field whereas auto=False raises if division is not exact.  We'll let
        # the caller decide whether or not they want to put the ground domain
        # over a field. In most uses mod is already monic.
        mod = mod.monic(auto=False)

        self.rank = mod.degree()
        self.modulus = mod
        self.mod = mod.rep  # DMP representation

        self.domain = dom = mod.domain
        self.ring = dom.old_poly_ring(*mod.gens)

        self.zero = self.convert(self.ring.zero)
        self.one = self.convert(self.ring.one)

        gen = self.ring.gens[0]
        self.symbol = self.ring.symbols[0]
        self.generator = self.convert(gen)
        self.basis = tuple(self.convert(gen**i) for i in range(self.rank))

        # XXX: It might be necessary to check mod.is_irreducible here
        self.is_Field = self.domain.is_Field

    def new(self, arg):
        rep = self.ring.convert(arg)
        return ExtElem(rep % self.mod, self)

    def __eq__(self, other):
        if not isinstance(other, FiniteExtension):
            return False
        return self.modulus == other.modulus

    def __hash__(self):
        return hash((self.__class__.__name__, self.modulus))

    def __str__(self):
        return "%s/(%s)" % (self.ring, self.modulus.as_expr())

    __repr__ = __str__

    @property
    def has_CharacteristicZero(self):
        return self.domain.has_CharacteristicZero

    def characteristic(self):
        return self.domain.characteristic()

    def convert(self, f, base=None):
        rep = self.ring.convert(f, base)
        return ExtElem(rep % self.mod, self)

    def convert_from(self, f, base):
        rep = self.ring.convert(f, base)
        return ExtElem(rep % self.mod, self)

    def to_sympy(self, f):
        return self.ring.to_sympy(f.rep)

    def from_sympy(self, f):
        return self.convert(f)

    def set_domain(self, K):
        mod = self.modulus.set_domain(K)
        return self.__class__(mod)

    def drop(self, *symbols):
        if self.symbol in symbols:
            raise GeneratorsError('Can not drop generator from FiniteExtension')
        K = self.domain.drop(*symbols)
        return self.set_domain(K)

    def quo(self, f, g):
        return self.exquo(f, g)

    def exquo(self, f, g):
        rep = self.ring.exquo(f.rep, g.rep)
        return ExtElem(rep % self.mod, self)

    def is_negative(self, a):
        return False

    def is_unit(self, a):
        if self.is_Field:
            return bool(a)
        elif a.is_ground:
            return self.domain.is_unit(a.to_ground())

FiniteExtension = MonogenicFiniteExtension
