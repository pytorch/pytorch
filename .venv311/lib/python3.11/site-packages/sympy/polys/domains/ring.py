"""Implementation of :class:`Ring` class. """


from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import ExactQuotientFailed, NotInvertible, NotReversible

from sympy.utilities import public

@public
class Ring(Domain):
    """Represents a ring domain. """

    is_Ring = True

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__floordiv__``.  """
        if a % b:
            raise ExactQuotientFailed(a, b, self)
        else:
            return a // b

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__floordiv__``. """
        return a // b

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies ``__mod__``.  """
        return a % b

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__divmod__``. """
        return divmod(a, b)

    def invert(self, a, b):
        """Returns inversion of ``a mod b``. """
        s, t, h = self.gcdex(a, b)

        if self.is_one(h):
            return s % b
        else:
            raise NotInvertible("zero divisor")

    def revert(self, a):
        """Returns ``a**(-1)`` if possible. """
        if self.is_one(a) or self.is_one(-a):
            return a
        else:
            raise NotReversible('only units are reversible in a ring')

    def is_unit(self, a):
        try:
            self.revert(a)
            return True
        except NotReversible:
            return False

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a

    def denom(self, a):
        """Returns denominator of `a`. """
        return self.one

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over self.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2)
        QQ[x]**2
        """
        raise NotImplementedError

    def ideal(self, *gens):
        """
        Generate an ideal of ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x**2)
        <x**2>
        """
        from sympy.polys.agca.ideals import ModuleImplementedIdeal
        return ModuleImplementedIdeal(self, self.free_module(1).submodule(
            *[[x] for x in gens]))

    def quotient_ring(self, e):
        """
        Form a quotient ring of ``self``.

        Here ``e`` can be an ideal or an iterable.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).quotient_ring(QQ.old_poly_ring(x).ideal(x**2))
        QQ[x]/<x**2>
        >>> QQ.old_poly_ring(x).quotient_ring([x**2])
        QQ[x]/<x**2>

        The division operator has been overloaded for this:

        >>> QQ.old_poly_ring(x)/[x**2]
        QQ[x]/<x**2>
        """
        from sympy.polys.agca.ideals import Ideal
        from sympy.polys.domains.quotientring import QuotientRing
        if not isinstance(e, Ideal):
            e = self.ideal(*e)
        return QuotientRing(self, e)

    def __truediv__(self, e):
        return self.quotient_ring(e)
