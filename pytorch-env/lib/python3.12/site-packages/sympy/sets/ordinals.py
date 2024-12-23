from sympy.core import Basic, Integer
import operator


class OmegaPower(Basic):
    """
    Represents ordinal exponential and multiplication terms one of the
    building blocks of the :class:`Ordinal` class.
    In ``OmegaPower(a, b)``, ``a`` represents exponent and ``b`` represents multiplicity.
    """
    def __new__(cls, a, b):
        if isinstance(b, int):
            b = Integer(b)
        if not isinstance(b, Integer) or b <= 0:
            raise TypeError("multiplicity must be a positive integer")

        if not isinstance(a, Ordinal):
            a = Ordinal.convert(a)

        return Basic.__new__(cls, a, b)

    @property
    def exp(self):
        return self.args[0]

    @property
    def mult(self):
        return self.args[1]

    def _compare_term(self, other, op):
        if self.exp == other.exp:
            return op(self.mult, other.mult)
        else:
            return op(self.exp, other.exp)

    def __eq__(self, other):
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self.args == other.args

    def __hash__(self):
        return Basic.__hash__(self)

    def __lt__(self, other):
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self._compare_term(other, operator.lt)


class Ordinal(Basic):
    """
    Represents ordinals in Cantor normal form.

    Internally, this class is just a list of instances of OmegaPower.

    Examples
    ========
    >>> from sympy import Ordinal, OmegaPower
    >>> from sympy.sets.ordinals import omega
    >>> w = omega
    >>> w.is_limit_ordinal
    True
    >>> Ordinal(OmegaPower(w + 1, 1), OmegaPower(3, 2))
    w**(w + 1) + w**3*2
    >>> 3 + w
    w
    >>> (w + 1) * w
    w**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ordinal_arithmetic
    """
    def __new__(cls, *terms):
        obj = super().__new__(cls, *terms)
        powers = [i.exp for i in obj.args]
        if not all(powers[i] >= powers[i+1] for i in range(len(powers) - 1)):
            raise ValueError("powers must be in decreasing order")
        return obj

    @property
    def terms(self):
        return self.args

    @property
    def leading_term(self):
        if self == ord0:
            raise ValueError("ordinal zero has no leading term")
        return self.terms[0]

    @property
    def trailing_term(self):
        if self == ord0:
            raise ValueError("ordinal zero has no trailing term")
        return self.terms[-1]

    @property
    def is_successor_ordinal(self):
        try:
            return self.trailing_term.exp == ord0
        except ValueError:
            return False

    @property
    def is_limit_ordinal(self):
        try:
            return not self.trailing_term.exp == ord0
        except ValueError:
            return False

    @property
    def degree(self):
        return self.leading_term.exp

    @classmethod
    def convert(cls, integer_value):
        if integer_value == 0:
            return ord0
        return Ordinal(OmegaPower(0, integer_value))

    def __eq__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        return self.terms == other.terms

    def __hash__(self):
        return hash(self.args)

    def __lt__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        for term_self, term_other in zip(self.terms, other.terms):
            if term_self != term_other:
                return term_self < term_other
        return len(self.terms) < len(other.terms)

    def __le__(self, other):
        return (self == other or self < other)

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __str__(self):
        net_str = ""
        plus_count = 0
        if self == ord0:
            return 'ord0'
        for i in self.terms:
            if plus_count:
                net_str += " + "

            if i.exp == ord0:
                net_str += str(i.mult)
            elif i.exp == 1:
                net_str += 'w'
            elif len(i.exp.terms) > 1 or i.exp.is_limit_ordinal:
                net_str += 'w**(%s)'%i.exp
            else:
                net_str += 'w**%s'%i.exp

            if not i.mult == 1 and not i.exp == ord0:
                net_str += '*%s'%i.mult

            plus_count += 1
        return(net_str)

    __repr__ = __str__

    def __add__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        if other == ord0:
            return self
        a_terms = list(self.terms)
        b_terms = list(other.terms)
        r = len(a_terms) - 1
        b_exp = other.degree
        while r >= 0 and a_terms[r].exp < b_exp:
            r -= 1
        if r < 0:
            terms = b_terms
        elif a_terms[r].exp == b_exp:
            sum_term = OmegaPower(b_exp, a_terms[r].mult + other.leading_term.mult)
            terms = a_terms[:r] + [sum_term] + b_terms[1:]
        else:
            terms = a_terms[:r+1] + b_terms
        return Ordinal(*terms)

    def __radd__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        return other + self

    def __mul__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        if ord0 in (self, other):
            return ord0
        a_exp = self.degree
        a_mult = self.leading_term.mult
        summation = []
        if other.is_limit_ordinal:
            for arg in other.terms:
                summation.append(OmegaPower(a_exp + arg.exp, arg.mult))

        else:
            for arg in other.terms[:-1]:
                summation.append(OmegaPower(a_exp + arg.exp, arg.mult))
            b_mult = other.trailing_term.mult
            summation.append(OmegaPower(a_exp, a_mult*b_mult))
            summation += list(self.terms[1:])
        return Ordinal(*summation)

    def __rmul__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        return other * self

    def __pow__(self, other):
        if not self == omega:
            return NotImplemented
        return Ordinal(OmegaPower(other, 1))


class OrdinalZero(Ordinal):
    """The ordinal zero.

    OrdinalZero can be imported as ``ord0``.
    """
    pass


class OrdinalOmega(Ordinal):
    """The ordinal omega which forms the base of all ordinals in cantor normal form.

    OrdinalOmega can be imported as ``omega``.

    Examples
    ========

    >>> from sympy.sets.ordinals import omega
    >>> omega + omega
    w*2
    """
    def __new__(cls):
        return Ordinal.__new__(cls)

    @property
    def terms(self):
        return (OmegaPower(1, 1),)


ord0 = OrdinalZero()
omega = OrdinalOmega()
