"""
Definition of physical dimensions.

Unit systems will be constructed on top of these dimensions.

Most of the examples in the doc use MKS system and are presented from the
computer point of view: from a human point, adding length to time is not legal
in MKS but it is in natural system; for a computer in natural system there is
no time dimension (but a velocity dimension instead) - in the basis - so the
question of adding time to length has no meaning.
"""

from __future__ import annotations

import collections
from functools import reduce

from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.core.expr import Expr
from sympy.core.power import Pow


class _QuantityMapper:

    _quantity_scale_factors_global: dict[Expr, Expr] = {}
    _quantity_dimensional_equivalence_map_global: dict[Expr, Expr] = {}
    _quantity_dimension_global: dict[Expr, Expr] = {}

    def __init__(self, *args, **kwargs):
        self._quantity_dimension_map = {}
        self._quantity_scale_factors = {}

    def set_quantity_dimension(self, quantity, dimension):
        """
        Set the dimension for the quantity in a unit system.

        If this relation is valid in every unit system, use
        ``quantity.set_global_dimension(dimension)`` instead.
        """
        from sympy.physics.units import Quantity
        dimension = sympify(dimension)
        if not isinstance(dimension, Dimension):
            if dimension == 1:
                dimension = Dimension(1)
            else:
                raise ValueError("expected dimension or 1")
        elif isinstance(dimension, Quantity):
            dimension = self.get_quantity_dimension(dimension)
        self._quantity_dimension_map[quantity] = dimension

    def set_quantity_scale_factor(self, quantity, scale_factor):
        """
        Set the scale factor of a quantity relative to another quantity.

        It should be used only once per quantity to just one other quantity,
        the algorithm will then be able to compute the scale factors to all
        other quantities.

        In case the scale factor is valid in every unit system, please use
        ``quantity.set_global_relative_scale_factor(scale_factor)`` instead.
        """
        from sympy.physics.units import Quantity
        from sympy.physics.units.prefixes import Prefix
        scale_factor = sympify(scale_factor)
        # replace all prefixes by their ratio to canonical units:
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Prefix),
            lambda x: x.scale_factor
        )
        # replace all quantities by their ratio to canonical units:
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Quantity),
            lambda x: self.get_quantity_scale_factor(x)
        )
        self._quantity_scale_factors[quantity] = scale_factor

    def get_quantity_dimension(self, unit):
        from sympy.physics.units import Quantity
        # First look-up the local dimension map, then the global one:
        if unit in self._quantity_dimension_map:
            return self._quantity_dimension_map[unit]
        if unit in self._quantity_dimension_global:
            return self._quantity_dimension_global[unit]
        if unit in self._quantity_dimensional_equivalence_map_global:
            dep_unit = self._quantity_dimensional_equivalence_map_global[unit]
            if isinstance(dep_unit, Quantity):
                return self.get_quantity_dimension(dep_unit)
            else:
                return Dimension(self.get_dimensional_expr(dep_unit))
        if isinstance(unit, Quantity):
            return Dimension(unit.name)
        else:
            return Dimension(1)

    def get_quantity_scale_factor(self, unit):
        if unit in self._quantity_scale_factors:
            return self._quantity_scale_factors[unit]
        if unit in self._quantity_scale_factors_global:
            mul_factor, other_unit = self._quantity_scale_factors_global[unit]
            return mul_factor*self.get_quantity_scale_factor(other_unit)
        return S.One


class Dimension(Expr):
    """
    This class represent the dimension of a physical quantities.

    The ``Dimension`` constructor takes as parameters a name and an optional
    symbol.

    For example, in classical mechanics we know that time is different from
    temperature and dimensions make this difference (but they do not provide
    any measure of these quantities.

        >>> from sympy.physics.units import Dimension
        >>> length = Dimension('length')
        >>> length
        Dimension(length)
        >>> time = Dimension('time')
        >>> time
        Dimension(time)

    Dimensions can be composed using multiplication, division and
    exponentiation (by a number) to give new dimensions. Addition and
    subtraction is defined only when the two objects are the same dimension.

        >>> velocity = length / time
        >>> velocity
        Dimension(length/time)

    It is possible to use a dimension system object to get the dimensionsal
    dependencies of a dimension, for example the dimension system used by the
    SI units convention can be used:

        >>> from sympy.physics.units.systems.si import dimsys_SI
        >>> dimsys_SI.get_dimensional_dependencies(velocity)
        {Dimension(length, L): 1, Dimension(time, T): -1}
        >>> length + length
        Dimension(length)
        >>> l2 = length**2
        >>> l2
        Dimension(length**2)
        >>> dimsys_SI.get_dimensional_dependencies(l2)
        {Dimension(length, L): 2}

    """

    _op_priority = 13.0

    # XXX: This doesn't seem to be used anywhere...
    _dimensional_dependencies = {}  # type: ignore

    is_commutative = True
    is_number = False
    # make sqrt(M**2) --> M
    is_positive = True
    is_real = True

    def __new__(cls, name, symbol=None):

        if isinstance(name, str):
            name = Symbol(name)
        else:
            name = sympify(name)

        if not isinstance(name, Expr):
            raise TypeError("Dimension name needs to be a valid math expression")

        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        elif symbol is not None:
            assert isinstance(symbol, Symbol)

        obj = Expr.__new__(cls, name)

        obj._name = name
        obj._symbol = symbol
        return obj

    @property
    def name(self):
        return self._name

    @property
    def symbol(self):
        return self._symbol

    def __str__(self):
        """
        Display the string representation of the dimension.
        """
        if self.symbol is None:
            return "Dimension(%s)" % (self.name)
        else:
            return "Dimension(%s, %s)" % (self.name, self.symbol)

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return self

    def __add__(self, other):
        from sympy.physics.units.quantities import Quantity
        other = sympify(other)
        if isinstance(other, Basic):
            if other.has(Quantity):
                raise TypeError("cannot sum dimension and quantity")
            if isinstance(other, Dimension) and self == other:
                return self
            return super().__add__(other)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # there is no notion of ordering (or magnitude) among dimension,
        # subtraction is equivalent to addition when the operation is legal
        return self + other

    def __rsub__(self, other):
        # there is no notion of ordering (or magnitude) among dimension,
        # subtraction is equivalent to addition when the operation is legal
        return self + other

    def __pow__(self, other):
        return self._eval_power(other)

    def _eval_power(self, other):
        other = sympify(other)
        return Dimension(self.name**other)

    def __mul__(self, other):
        from sympy.physics.units.quantities import Quantity
        if isinstance(other, Basic):
            if other.has(Quantity):
                raise TypeError("cannot sum dimension and quantity")
            if isinstance(other, Dimension):
                return Dimension(self.name*other.name)
            if not other.free_symbols:  # other.is_number cannot be used
                return self
            return super().__mul__(other)
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self*Pow(other, -1)

    def __rtruediv__(self, other):
        return other * pow(self, -1)

    @classmethod
    def _from_dimensional_dependencies(cls, dependencies):
        return reduce(lambda x, y: x * y, (
            d**e for d, e in dependencies.items()
        ), 1)

    def has_integer_powers(self, dim_sys):
        """
        Check if the dimension object has only integer powers.

        All the dimension powers should be integers, but rational powers may
        appear in intermediate steps. This method may be used to check that the
        final result is well-defined.
        """

        return all(dpow.is_Integer for dpow in dim_sys.get_dimensional_dependencies(self).values())


# Create dimensions according to the base units in MKSA.
# For other unit systems, they can be derived by transforming the base
# dimensional dependency dictionary.


class DimensionSystem(Basic, _QuantityMapper):
    r"""
    DimensionSystem represents a coherent set of dimensions.

    The constructor takes three parameters:

    - base dimensions;
    - derived dimensions: these are defined in terms of the base dimensions
      (for example velocity is defined from the division of length by time);
    - dependency of dimensions: how the derived dimensions depend
      on the base dimensions.

    Optionally either the ``derived_dims`` or the ``dimensional_dependencies``
    may be omitted.
    """

    def __new__(cls, base_dims, derived_dims=(), dimensional_dependencies={}):
        dimensional_dependencies = dict(dimensional_dependencies)

        def parse_dim(dim):
            if isinstance(dim, str):
                dim = Dimension(Symbol(dim))
            elif isinstance(dim, Dimension):
                pass
            elif isinstance(dim, Symbol):
                dim = Dimension(dim)
            else:
                raise TypeError("%s wrong type" % dim)
            return dim

        base_dims = [parse_dim(i) for i in base_dims]
        derived_dims = [parse_dim(i) for i in derived_dims]

        for dim in base_dims:
            if (dim in dimensional_dependencies
                and (len(dimensional_dependencies[dim]) != 1 or
                dimensional_dependencies[dim].get(dim, None) != 1)):
                raise IndexError("Repeated value in base dimensions")
            dimensional_dependencies[dim] = Dict({dim: 1})

        def parse_dim_name(dim):
            if isinstance(dim, Dimension):
                return dim
            elif isinstance(dim, str):
                return Dimension(Symbol(dim))
            elif isinstance(dim, Symbol):
                return Dimension(dim)
            else:
                raise TypeError("unrecognized type %s for %s" % (type(dim), dim))

        for dim in dimensional_dependencies.keys():
            dim = parse_dim(dim)
            if (dim not in derived_dims) and (dim not in base_dims):
                derived_dims.append(dim)

        def parse_dict(d):
            return Dict({parse_dim_name(i): j for i, j in d.items()})

        # Make sure everything is a SymPy type:
        dimensional_dependencies = {parse_dim_name(i): parse_dict(j) for i, j in
                                    dimensional_dependencies.items()}

        for dim in derived_dims:
            if dim in base_dims:
                raise ValueError("Dimension %s both in base and derived" % dim)
            if dim not in dimensional_dependencies:
                # TODO: should this raise a warning?
                dimensional_dependencies[dim] = Dict({dim: 1})

        base_dims.sort(key=default_sort_key)
        derived_dims.sort(key=default_sort_key)

        base_dims = Tuple(*base_dims)
        derived_dims = Tuple(*derived_dims)
        dimensional_dependencies = Dict({i: Dict(j) for i, j in dimensional_dependencies.items()})
        obj = Basic.__new__(cls, base_dims, derived_dims, dimensional_dependencies)
        return obj

    @property
    def base_dims(self):
        return self.args[0]

    @property
    def derived_dims(self):
        return self.args[1]

    @property
    def dimensional_dependencies(self):
        return self.args[2]

    def _get_dimensional_dependencies_for_name(self, dimension):
        if isinstance(dimension, str):
            dimension = Dimension(Symbol(dimension))
        elif not isinstance(dimension, Dimension):
            dimension = Dimension(dimension)

        if dimension.name.is_Symbol:
            # Dimensions not included in the dependencies are considered
            # as base dimensions:
            return dict(self.dimensional_dependencies.get(dimension, {dimension: 1}))

        if dimension.name.is_number or dimension.name.is_NumberSymbol:
            return {}

        get_for_name = self._get_dimensional_dependencies_for_name

        if dimension.name.is_Mul:
            ret = collections.defaultdict(int)
            dicts = [get_for_name(i) for i in dimension.name.args]
            for d in dicts:
                for k, v in d.items():
                    ret[k] += v
            return {k: v for (k, v) in ret.items() if v != 0}

        if dimension.name.is_Add:
            dicts = [get_for_name(i) for i in dimension.name.args]
            if all(d == dicts[0] for d in dicts[1:]):
                return dicts[0]
            raise TypeError("Only equivalent dimensions can be added or subtracted.")

        if dimension.name.is_Pow:
            dim_base = get_for_name(dimension.name.base)
            dim_exp = get_for_name(dimension.name.exp)
            if dim_exp == {} or dimension.name.exp.is_Symbol:
                return {k: v * dimension.name.exp for (k, v) in dim_base.items()}
            else:
                raise TypeError("The exponent for the power operator must be a Symbol or dimensionless.")

        if dimension.name.is_Function:
            args = (Dimension._from_dimensional_dependencies(
                get_for_name(arg)) for arg in dimension.name.args)
            result = dimension.name.func(*args)

            dicts = [get_for_name(i) for i in dimension.name.args]

            if isinstance(result, Dimension):
                return self.get_dimensional_dependencies(result)
            elif result.func == dimension.name.func:
                if isinstance(dimension.name, TrigonometricFunction):
                    if dicts[0] in ({}, {Dimension('angle'): 1}):
                        return {}
                    else:
                        raise TypeError("The input argument for the function {} must be dimensionless or have dimensions of angle.".format(dimension.func))
                else:
                    if all(item == {} for item in dicts):
                        return {}
                    else:
                        raise TypeError("The input arguments for the function {} must be dimensionless.".format(dimension.func))
            else:
                return get_for_name(result)

        raise TypeError("Type {} not implemented for get_dimensional_dependencies".format(type(dimension.name)))

    def get_dimensional_dependencies(self, name, mark_dimensionless=False):
        dimdep = self._get_dimensional_dependencies_for_name(name)
        if mark_dimensionless and dimdep == {}:
            return {Dimension(1): 1}
        return dict(dimdep.items())

    def equivalent_dims(self, dim1, dim2):
        deps1 = self.get_dimensional_dependencies(dim1)
        deps2 = self.get_dimensional_dependencies(dim2)
        return deps1 == deps2

    def extend(self, new_base_dims, new_derived_dims=(), new_dim_deps=None):
        deps = dict(self.dimensional_dependencies)
        if new_dim_deps:
            deps.update(new_dim_deps)

        new_dim_sys = DimensionSystem(
            tuple(self.base_dims) + tuple(new_base_dims),
            tuple(self.derived_dims) + tuple(new_derived_dims),
            deps
        )
        new_dim_sys._quantity_dimension_map.update(self._quantity_dimension_map)
        new_dim_sys._quantity_scale_factors.update(self._quantity_scale_factors)
        return new_dim_sys

    def is_dimensionless(self, dimension):
        """
        Check if the dimension object really has a dimension.

        A dimension should have at least one component with non-zero power.
        """
        if dimension.name == 1:
            return True
        return self.get_dimensional_dependencies(dimension) == {}

    @property
    def list_can_dims(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        List all canonical dimension names.
        """
        dimset = set()
        for i in self.base_dims:
            dimset.update(set(self.get_dimensional_dependencies(i).keys()))
        return tuple(sorted(dimset, key=str))

    @property
    def inv_can_transf_matrix(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Compute the inverse transformation matrix from the base to the
        canonical dimension basis.

        It corresponds to the matrix where columns are the vector of base
        dimensions in canonical basis.

        This matrix will almost never be used because dimensions are always
        defined with respect to the canonical basis, so no work has to be done
        to get them in this basis. Nonetheless if this matrix is not square
        (or not invertible) it means that we have chosen a bad basis.
        """
        matrix = reduce(lambda x, y: x.row_join(y),
                        [self.dim_can_vector(d) for d in self.base_dims])
        return matrix

    @property
    def can_transf_matrix(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Return the canonical transformation matrix from the canonical to the
        base dimension basis.

        It is the inverse of the matrix computed with inv_can_transf_matrix().
        """

        #TODO: the inversion will fail if the system is inconsistent, for
        #      example if the matrix is not a square
        return reduce(lambda x, y: x.row_join(y),
                      [self.dim_can_vector(d) for d in sorted(self.base_dims, key=str)]
                      ).inv()

    def dim_can_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Dimensional representation in terms of the canonical base dimensions.
        """

        vec = []
        for d in self.list_can_dims:
            vec.append(self.get_dimensional_dependencies(dim).get(d, 0))
        return Matrix(vec)

    def dim_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.


        Vector representation in terms of the base dimensions.
        """
        return self.can_transf_matrix * Matrix(self.dim_can_vector(dim))

    def print_dim_base(self, dim):
        """
        Give the string expression of a dimension in term of the basis symbols.
        """
        dims = self.dim_vector(dim)
        symbols = [i.symbol if i.symbol is not None else i.name for i in self.base_dims]
        res = S.One
        for (s, p) in zip(symbols, dims):
            res *= s**p
        return res

    @property
    def dim(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Give the dimension of the system.

        That is return the number of dimensions forming the basis.
        """
        return len(self.base_dims)

    @property
    def is_consistent(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Check if the system is well defined.
        """

        # not enough or too many base dimensions compared to independent
        # dimensions
        # in vector language: the set of vectors do not form a basis
        return self.inv_can_transf_matrix.is_square
