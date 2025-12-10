"""Implementation of :class:`ExpressionDomain` class. """


from sympy.core import sympify, SympifyError
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyutils import PicklableWithSlots
from sympy.utilities import public

eflags = {"deep": False, "mul": True, "power_exp": False, "power_base": False,
              "basic": False, "multinomial": False, "log": False}

@public
class ExpressionDomain(Field, CharacteristicZero, SimpleDomain):
    """A class for arbitrary expressions. """

    is_SymbolicDomain = is_EX = True

    class Expression(DomainElement, PicklableWithSlots):
        """An arbitrary expression. """

        __slots__ = ('ex',)

        def __init__(self, ex):
            if not isinstance(ex, self.__class__):
                self.ex = sympify(ex)
            else:
                self.ex = ex.ex

        def __repr__(f):
            return 'EX(%s)' % repr(f.ex)

        def __str__(f):
            return 'EX(%s)' % str(f.ex)

        def __hash__(self):
            return hash((self.__class__.__name__, self.ex))

        def parent(self):
            return EX

        def as_expr(f):
            return f.ex

        def numer(f):
            return f.__class__(f.ex.as_numer_denom()[0])

        def denom(f):
            return f.__class__(f.ex.as_numer_denom()[1])

        def simplify(f, ex):
            return f.__class__(ex.cancel().expand(**eflags))

        def __abs__(f):
            return f.__class__(abs(f.ex))

        def __neg__(f):
            return f.__class__(-f.ex)

        def _to_ex(f, g):
            try:
                return f.__class__(g)
            except SympifyError:
                return None

        def __lt__(f, g):
            return f.ex.sort_key() < g.ex.sort_key()

        def __add__(f, g):
            g = f._to_ex(g)

            if g is None:
                return NotImplemented
            elif g == EX.zero:
                return f
            elif f == EX.zero:
                return g
            else:
                return f.simplify(f.ex + g.ex)

        def __radd__(f, g):
            return f.simplify(f.__class__(g).ex + f.ex)

        def __sub__(f, g):
            g = f._to_ex(g)

            if g is None:
                return NotImplemented
            elif g == EX.zero:
                return f
            elif f == EX.zero:
                return -g
            else:
                return f.simplify(f.ex - g.ex)

        def __rsub__(f, g):
            return f.simplify(f.__class__(g).ex - f.ex)

        def __mul__(f, g):
            g = f._to_ex(g)

            if g is None:
                return NotImplemented

            if EX.zero in (f, g):
                return EX.zero
            elif f.ex.is_Number and g.ex.is_Number:
                return f.__class__(f.ex*g.ex)

            return f.simplify(f.ex*g.ex)

        def __rmul__(f, g):
            return f.simplify(f.__class__(g).ex*f.ex)

        def __pow__(f, n):
            n = f._to_ex(n)

            if n is not None:
                return f.simplify(f.ex**n.ex)
            else:
                return NotImplemented

        def __truediv__(f, g):
            g = f._to_ex(g)

            if g is not None:
                return f.simplify(f.ex/g.ex)
            else:
                return NotImplemented

        def __rtruediv__(f, g):
            return f.simplify(f.__class__(g).ex/f.ex)

        def __eq__(f, g):
            return f.ex == f.__class__(g).ex

        def __ne__(f, g):
            return not f == g

        def __bool__(f):
            return not f.ex.is_zero

        def gcd(f, g):
            from sympy.polys import gcd
            return f.__class__(gcd(f.ex, f.__class__(g).ex))

        def lcm(f, g):
            from sympy.polys import lcm
            return f.__class__(lcm(f.ex, f.__class__(g).ex))

    dtype = Expression

    zero = Expression(0)
    one = Expression(1)

    rep = 'EX'

    has_assoc_Ring = False
    has_assoc_Field = True

    def __init__(self):
        pass

    def __eq__(self, other):
        if isinstance(other, ExpressionDomain):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        return hash("EX")

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return a.as_expr()

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        return self.dtype(a)

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_AlgebraicField(K1, a, K0):
        """Convert an ``ANP`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ComplexField(K1, a, K0):
        """Convert a mpmath ``mpc`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_PolynomialRing(K1, a, K0):
        """Convert a ``DMP`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_FractionField(K1, a, K0):
        """Convert a ``DMF`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ExpressionDomain(K1, a, K0):
        """Convert a ``EX`` object to ``dtype``. """
        return a

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self  # XXX: EX is not a ring but we don't have much choice here.

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return a.ex.as_coeff_mul()[0].is_positive

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return a.ex.could_extract_minus_sign()

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return a.ex.as_coeff_mul()[0].is_nonpositive

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return a.ex.as_coeff_mul()[0].is_nonnegative

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numer()

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denom()

    def gcd(self, a, b):
        return self(1)

    def lcm(self, a, b):
        return a.lcm(b)


EX = ExpressionDomain()
