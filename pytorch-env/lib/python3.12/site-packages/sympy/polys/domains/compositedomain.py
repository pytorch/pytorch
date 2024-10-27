"""Implementation of :class:`CompositeDomain` class. """


from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import GeneratorsError

from sympy.utilities import public

@public
class CompositeDomain(Domain):
    """Base class for composite domains, e.g. ZZ[x], ZZ(X). """

    is_Composite = True

    gens, ngens, symbols, domain = [None]*4

    def inject(self, *symbols):
        """Inject generators into this domain.  """
        if not (set(self.symbols) & set(symbols)):
            return self.__class__(self.domain, self.symbols + symbols, self.order)
        else:
            raise GeneratorsError("common generators in %s and %s" % (self.symbols, symbols))

    def drop(self, *symbols):
        """Drop generators from this domain. """
        symset = set(symbols)
        newsyms = tuple(s for s in self.symbols if s not in symset)
        domain = self.domain.drop(*symbols)
        if not newsyms:
            return domain
        else:
            return self.__class__(domain, newsyms, self.order)

    def set_domain(self, domain):
        """Set the ground domain of this domain. """
        return self.__class__(domain, self.symbols, self.order)

    @property
    def is_Exact(self):
        """Returns ``True`` if this domain is exact. """
        return self.domain.is_Exact

    def get_exact(self):
        """Returns an exact version of this domain. """
        return self.set_domain(self.domain.get_exact())

    @property
    def has_CharacteristicZero(self):
        return self.domain.has_CharacteristicZero

    def characteristic(self):
        return self.domain.characteristic()
