"""

Module for the DomainScalar class.

A DomainScalar represents an element which is in a particular
Domain. The idea is that the DomainScalar class provides the
convenience routines for unifying elements with different domains.

It assists in Scalar Multiplication and getitem for DomainMatrix.

"""
from ..constructor import construct_domain

from sympy.polys.domains import Domain, ZZ


class DomainScalar:
    r"""
    docstring
    """

    def __new__(cls, element, domain):
        if not isinstance(domain, Domain):
            raise TypeError("domain should be of type Domain")
        if not domain.of_type(element):
            raise TypeError("element %s should be in domain %s" % (element, domain))
        return cls.new(element, domain)

    @classmethod
    def new(cls, element, domain):
        obj = super().__new__(cls)
        obj.element = element
        obj.domain = domain
        return obj

    def __repr__(self):
        return repr(self.element)

    @classmethod
    def from_sympy(cls, expr):
        [domain, [element]] = construct_domain([expr])
        return cls.new(element, domain)

    def to_sympy(self):
        return self.domain.to_sympy(self.element)

    def to_domain(self, domain):
        element = domain.convert_from(self.element, self.domain)
        return self.new(element, domain)

    def convert_to(self, domain):
        return self.to_domain(domain)

    def unify(self, other):
        domain = self.domain.unify(other.domain)
        return self.to_domain(domain), other.to_domain(domain)

    def __bool__(self):
        return bool(self.element)

    def __add__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        return self.new(self.element + other.element, self.domain)

    def __sub__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        return self.new(self.element - other.element, self.domain)

    def __mul__(self, other):
        if not isinstance(other, DomainScalar):
            if isinstance(other, int):
                other = DomainScalar(ZZ(other), ZZ)
            else:
                return NotImplemented

        self, other = self.unify(other)
        return self.new(self.element * other.element, self.domain)

    def __floordiv__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        return self.new(self.domain.quo(self.element, other.element), self.domain)

    def __mod__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        return self.new(self.domain.rem(self.element, other.element), self.domain)

    def __divmod__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        q, r = self.domain.div(self.element, other.element)
        return (self.new(q, self.domain), self.new(r, self.domain))

    def __pow__(self, n):
        if not isinstance(n, int):
            return NotImplemented
        return self.new(self.element**n, self.domain)

    def __pos__(self):
        return self.new(+self.element, self.domain)

    def __neg__(self):
        return self.new(-self.element, self.domain)

    def __eq__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        return self.element == other.element and self.domain == other.domain

    def is_zero(self):
        return self.element == self.domain.zero

    def is_one(self):
        return self.element == self.domain.one
