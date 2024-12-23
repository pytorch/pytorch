"""
Module defining unit prefixe class and some constants.

Constant dict for SI and binary prefixes are defined as PREFIXES and
BIN_PREFIXES.
"""
from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.core.singleton import S

class Prefix(Expr):
    """
    This class represent prefixes, with their name, symbol and factor.

    Prefixes are used to create derived units from a given unit. They should
    always be encapsulated into units.

    The factor is constructed from a base (default is 10) to some power, and
    it gives the total multiple or fraction. For example the kilometer km
    is constructed from the meter (factor 1) and the kilo (10 to the power 3,
    i.e. 1000). The base can be changed to allow e.g. binary prefixes.

    A prefix multiplied by something will always return the product of this
    other object times the factor, except if the other object:

    - is a prefix and they can be combined into a new prefix;
    - defines multiplication with prefixes (which is the case for the Unit
      class).
    """
    _op_priority = 13.0
    is_commutative = True

    def __new__(cls, name, abbrev, exponent, base=sympify(10), latex_repr=None):

        name = sympify(name)
        abbrev = sympify(abbrev)
        exponent = sympify(exponent)
        base = sympify(base)

        obj = Expr.__new__(cls, name, abbrev, exponent, base)
        obj._name = name
        obj._abbrev = abbrev
        obj._scale_factor = base**exponent
        obj._exponent = exponent
        obj._base = base
        obj._latex_repr = latex_repr
        return obj

    @property
    def name(self):
        return self._name

    @property
    def abbrev(self):
        return self._abbrev

    @property
    def scale_factor(self):
        return self._scale_factor

    def _latex(self, printer):
        if self._latex_repr is None:
            return r'\text{%s}' % self._abbrev
        return self._latex_repr

    @property
    def base(self):
        return self._base

    def __str__(self):
        return str(self._abbrev)

    def __repr__(self):
        if self.base == 10:
            return "Prefix(%r, %r, %r)" % (
                str(self.name), str(self.abbrev), self._exponent)
        else:
            return "Prefix(%r, %r, %r, %r)" % (
                str(self.name), str(self.abbrev), self._exponent, self.base)

    def __mul__(self, other):
        from sympy.physics.units import Quantity
        if not isinstance(other, (Quantity, Prefix)):
            return super().__mul__(other)

        fact = self.scale_factor * other.scale_factor

        if isinstance(other, Prefix):
            if fact == 1:
                return S.One
            # simplify prefix
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == fact:
                    return PREFIXES[p]
            return fact

        return self.scale_factor * other

    def __truediv__(self, other):
        if not hasattr(other, "scale_factor"):
            return super().__truediv__(other)

        fact = self.scale_factor / other.scale_factor

        if fact == 1:
            return S.One
        elif isinstance(other, Prefix):
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == fact:
                    return PREFIXES[p]
            return fact

        return self.scale_factor / other

    def __rtruediv__(self, other):
        if other == 1:
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == 1 / self.scale_factor:
                    return PREFIXES[p]
        return other / self.scale_factor


def prefix_unit(unit, prefixes):
    """
    Return a list of all units formed by unit and the given prefixes.

    You can use the predefined PREFIXES or BIN_PREFIXES, but you can also
    pass as argument a subdict of them if you do not want all prefixed units.

        >>> from sympy.physics.units.prefixes import (PREFIXES,
        ...                                                 prefix_unit)
        >>> from sympy.physics.units import m
        >>> pref = {"m": PREFIXES["m"], "c": PREFIXES["c"], "d": PREFIXES["d"]}
        >>> prefix_unit(m, pref)  # doctest: +SKIP
        [millimeter, centimeter, decimeter]
    """

    from sympy.physics.units.quantities import Quantity
    from sympy.physics.units import UnitSystem

    prefixed_units = []

    for prefix in prefixes.values():
        quantity = Quantity(
                "%s%s" % (prefix.name, unit.name),
                abbrev=("%s%s" % (prefix.abbrev, unit.abbrev)),
                is_prefixed=True,
           )
        UnitSystem._quantity_dimensional_equivalence_map_global[quantity] = unit
        UnitSystem._quantity_scale_factors_global[quantity] = (prefix.scale_factor, unit)
        prefixed_units.append(quantity)

    return prefixed_units


yotta = Prefix('yotta', 'Y', 24)
zetta = Prefix('zetta', 'Z', 21)
exa = Prefix('exa', 'E', 18)
peta = Prefix('peta', 'P', 15)
tera = Prefix('tera', 'T', 12)
giga = Prefix('giga', 'G', 9)
mega = Prefix('mega', 'M', 6)
kilo = Prefix('kilo', 'k', 3)
hecto = Prefix('hecto', 'h', 2)
deca = Prefix('deca', 'da', 1)
deci = Prefix('deci', 'd', -1)
centi = Prefix('centi', 'c', -2)
milli = Prefix('milli', 'm', -3)
micro = Prefix('micro', 'mu', -6, latex_repr=r"\mu")
nano = Prefix('nano', 'n', -9)
pico = Prefix('pico', 'p', -12)
femto = Prefix('femto', 'f', -15)
atto = Prefix('atto', 'a', -18)
zepto = Prefix('zepto', 'z', -21)
yocto = Prefix('yocto', 'y', -24)


# https://physics.nist.gov/cuu/Units/prefixes.html
PREFIXES = {
    'Y': yotta,
    'Z': zetta,
    'E': exa,
    'P': peta,
    'T': tera,
    'G': giga,
    'M': mega,
    'k': kilo,
    'h': hecto,
    'da': deca,
    'd': deci,
    'c': centi,
    'm': milli,
    'mu': micro,
    'n': nano,
    'p': pico,
    'f': femto,
    'a': atto,
    'z': zepto,
    'y': yocto,
}


kibi = Prefix('kibi', 'Y', 10, 2)
mebi = Prefix('mebi', 'Y', 20, 2)
gibi = Prefix('gibi', 'Y', 30, 2)
tebi = Prefix('tebi', 'Y', 40, 2)
pebi = Prefix('pebi', 'Y', 50, 2)
exbi = Prefix('exbi', 'Y', 60, 2)


# https://physics.nist.gov/cuu/Units/binary.html
BIN_PREFIXES = {
    'Ki': kibi,
    'Mi': mebi,
    'Gi': gibi,
    'Ti': tebi,
    'Pi': pebi,
    'Ei': exbi,
}
