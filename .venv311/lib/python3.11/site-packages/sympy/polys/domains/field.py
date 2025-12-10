"""Implementation of :class:`Field` class. """


from sympy.polys.domains.ring import Ring
from sympy.polys.polyerrors import NotReversible, DomainError
from sympy.utilities import public

@public
class Field(Ring):
    """Represents a field domain. """

    is_Field = True
    is_PID = True

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        raise DomainError('there is no ring associated with %s' % self)

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__truediv__``.  """
        return a / b

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__truediv__``. """
        return a / b

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies nothing.  """
        return self.zero

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__truediv__``. """
        return a / b, self.zero

    def gcd(self, a, b):
        """
        Returns GCD of ``a`` and ``b``.

        This definition of GCD over fields allows to clear denominators
        in `primitive()`.

        Examples
        ========

        >>> from sympy.polys.domains import QQ
        >>> from sympy import S, gcd, primitive
        >>> from sympy.abc import x

        >>> QQ.gcd(QQ(2, 3), QQ(4, 9))
        2/9
        >>> gcd(S(2)/3, S(4)/9)
        2/9
        >>> primitive(2*x/3 + S(4)/9)
        (2/9, 3*x + 2)

        """
        try:
            ring = self.get_ring()
        except DomainError:
            return self.one

        p = ring.gcd(self.numer(a), self.numer(b))
        q = ring.lcm(self.denom(a), self.denom(b))

        return self.convert(p, ring)/q

    def gcdex(self, a, b):
        """
        Returns x, y, g such that a * x + b * y == g == gcd(a, b)
        """
        d = self.gcd(a, b)

        if a == self.zero:
            if b == self.zero:
                return self.zero, self.one, self.zero
            else:
                return self.zero, d/b, d
        else:
            return d/a, self.zero, d

    def lcm(self, a, b):
        """
        Returns LCM of ``a`` and ``b``.

        >>> from sympy.polys.domains import QQ
        >>> from sympy import S, lcm

        >>> QQ.lcm(QQ(2, 3), QQ(4, 9))
        4/3
        >>> lcm(S(2)/3, S(4)/9)
        4/3

        """

        try:
            ring = self.get_ring()
        except DomainError:
            return a*b

        p = ring.lcm(self.numer(a), self.numer(b))
        q = ring.gcd(self.denom(a), self.denom(b))

        return self.convert(p, ring)/q

    def revert(self, a):
        """Returns ``a**(-1)`` if possible. """
        if a:
            return 1/a
        else:
            raise NotReversible('zero is not reversible')

    def is_unit(self, a):
        """Return true if ``a`` is a invertible"""
        return bool(a)
