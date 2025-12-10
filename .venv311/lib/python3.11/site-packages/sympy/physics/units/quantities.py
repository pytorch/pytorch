"""
Physical quantities.
"""

from sympy.core.expr import AtomicExpr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.prefixes import Prefix


class Quantity(AtomicExpr):
    """
    Physical quantity: can be a unit of measure, a constant or a generic quantity.
    """

    is_commutative = True
    is_real = True
    is_number = False
    is_nonzero = True
    is_physical_constant = False
    _diff_wrt = True

    def __new__(cls, name, abbrev=None,
                latex_repr=None, pretty_unicode_repr=None,
                pretty_ascii_repr=None, mathml_presentation_repr=None,
                is_prefixed=False,
                **assumptions):

        if not isinstance(name, Symbol):
            name = Symbol(name)

        if abbrev is None:
            abbrev = name
        elif isinstance(abbrev, str):
            abbrev = Symbol(abbrev)

        # HACK: These are here purely for type checking. They actually get assigned below.
        cls._is_prefixed = is_prefixed

        obj = AtomicExpr.__new__(cls, name, abbrev)
        obj._name = name
        obj._abbrev = abbrev
        obj._latex_repr = latex_repr
        obj._unicode_repr = pretty_unicode_repr
        obj._ascii_repr = pretty_ascii_repr
        obj._mathml_repr = mathml_presentation_repr
        obj._is_prefixed = is_prefixed
        return obj

    def set_global_dimension(self, dimension):
        _QuantityMapper._quantity_dimension_global[self] = dimension

    def set_global_relative_scale_factor(self, scale_factor, reference_quantity):
        """
        Setting a scale factor that is valid across all unit system.
        """
        from sympy.physics.units import UnitSystem
        scale_factor = sympify(scale_factor)
        if isinstance(scale_factor, Prefix):
            self._is_prefixed = True
        # replace all prefixes by their ratio to canonical units:
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Prefix),
            lambda x: x.scale_factor
        )
        scale_factor = sympify(scale_factor)
        UnitSystem._quantity_scale_factors_global[self] = (scale_factor, reference_quantity)
        UnitSystem._quantity_dimensional_equivalence_map_global[self] = reference_quantity

    @property
    def name(self):
        return self._name

    @property
    def dimension(self):
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_default_unit_system()
        return unit_system.get_quantity_dimension(self)

    @property
    def abbrev(self):
        """
        Symbol representing the unit name.

        Prepend the abbreviation with the prefix symbol if it is defines.
        """
        return self._abbrev

    @property
    def scale_factor(self):
        """
        Overall magnitude of the quantity as compared to the canonical units.
        """
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_default_unit_system()
        return unit_system.get_quantity_scale_factor(self)

    def _eval_is_positive(self):
        return True

    def _eval_is_constant(self):
        return True

    def _eval_Abs(self):
        return self

    def _eval_subs(self, old, new):
        if isinstance(new, Quantity) and self != old:
            return self

    def _latex(self, printer):
        if self._latex_repr:
            return self._latex_repr
        else:
            return r'\text{{{}}}'.format(self.args[1] \
                          if len(self.args) >= 2 else self.args[0])

    def convert_to(self, other, unit_system="SI"):
        """
        Convert the quantity to another quantity of same dimensions.

        Examples
        ========

        >>> from sympy.physics.units import speed_of_light, meter, second
        >>> speed_of_light
        speed_of_light
        >>> speed_of_light.convert_to(meter/second)
        299792458*meter/second

        >>> from sympy.physics.units import liter
        >>> liter.convert_to(meter**3)
        meter**3/1000
        """
        from .util import convert_to
        return convert_to(self, other, unit_system)

    @property
    def free_symbols(self):
        """Return free symbols from quantity."""
        return set()

    @property
    def is_prefixed(self):
        """Whether or not the quantity is prefixed. Eg. `kilogram` is prefixed, but `gram` is not."""
        return self._is_prefixed

class PhysicalConstant(Quantity):
    """Represents a physical constant, eg. `speed_of_light` or `avogadro_constant`."""

    is_physical_constant = True
