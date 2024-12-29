"""Computations with ideals of polynomial rings."""

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable


class Ideal(IntegerPowerable):
    """
    Abstract base class for ideals.

    Do not instantiate - use explicit constructors in the ring class instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> QQ.old_poly_ring(x).ideal(x+1)
    <x + 1>

    Attributes

    - ring - the ring this ideal belongs to

    Non-implemented methods:

    - _contains_elem
    - _contains_ideal
    - _quotient
    - _intersect
    - _union
    - _product
    - is_whole_ring
    - is_zero
    - is_prime, is_maximal, is_primary, is_radical
    - is_principal
    - height, depth
    - radical

    Methods that likely should be overridden in subclasses:

    - reduce_element
    """

    def _contains_elem(self, x):
        """Implementation of element containment."""
        raise NotImplementedError

    def _contains_ideal(self, I):
        """Implementation of ideal containment."""
        raise NotImplementedError

    def _quotient(self, J):
        """Implementation of ideal quotient."""
        raise NotImplementedError

    def _intersect(self, J):
        """Implementation of ideal intersection."""
        raise NotImplementedError

    def is_whole_ring(self):
        """Return True if ``self`` is the whole ring."""
        raise NotImplementedError

    def is_zero(self):
        """Return True if ``self`` is the zero ideal."""
        raise NotImplementedError

    def _equals(self, J):
        """Implementation of ideal equality."""
        return self._contains_ideal(J) and J._contains_ideal(self)

    def is_prime(self):
        """Return True if ``self`` is a prime ideal."""
        raise NotImplementedError

    def is_maximal(self):
        """Return True if ``self`` is a maximal ideal."""
        raise NotImplementedError

    def is_radical(self):
        """Return True if ``self`` is a radical ideal."""
        raise NotImplementedError

    def is_primary(self):
        """Return True if ``self`` is a primary ideal."""
        raise NotImplementedError

    def is_principal(self):
        """Return True if ``self`` is a principal ideal."""
        raise NotImplementedError

    def radical(self):
        """Compute the radical of ``self``."""
        raise NotImplementedError

    def depth(self):
        """Compute the depth of ``self``."""
        raise NotImplementedError

    def height(self):
        """Compute the height of ``self``."""
        raise NotImplementedError

    # TODO more

    # non-implemented methods end here

    def __init__(self, ring):
        self.ring = ring

    def _check_ideal(self, J):
        """Helper to check ``J`` is an ideal of our ring."""
        if not isinstance(J, Ideal) or J.ring != self.ring:
            raise ValueError(
                'J must be an ideal of %s, got %s' % (self.ring, J))

    def contains(self, elem):
        """
        Return True if ``elem`` is an element of this ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x+1, x-1).contains(3)
        True
        >>> QQ.old_poly_ring(x).ideal(x**2, x**3).contains(x)
        False
        """
        return self._contains_elem(self.ring.convert(elem))

    def subset(self, other):
        """
        Returns True if ``other`` is is a subset of ``self``.

        Here ``other`` may be an ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x+1)
        >>> I.subset([x**2 - 1, x**2 + 2*x + 1])
        True
        >>> I.subset([x**2 + 1, x + 1])
        False
        >>> I.subset(QQ.old_poly_ring(x).ideal(x**2 - 1))
        True
        """
        if isinstance(other, Ideal):
            return self._contains_ideal(other)
        return all(self._contains_elem(x) for x in other)

    def quotient(self, J, **opts):
        r"""
        Compute the ideal quotient of ``self`` by ``J``.

        That is, if ``self`` is the ideal `I`, compute the set
        `I : J = \{x \in R | xJ \subset I \}`.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> R = QQ.old_poly_ring(x, y)
        >>> R.ideal(x*y).quotient(R.ideal(x))
        <y>
        """
        self._check_ideal(J)
        return self._quotient(J, **opts)

    def intersect(self, J):
        """
        Compute the intersection of self with ideal J.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> R = QQ.old_poly_ring(x, y)
        >>> R.ideal(x).intersect(R.ideal(y))
        <x*y>
        """
        self._check_ideal(J)
        return self._intersect(J)

    def saturate(self, J):
        r"""
        Compute the ideal saturation of ``self`` by ``J``.

        That is, if ``self`` is the ideal `I`, compute the set
        `I : J^\infty = \{x \in R | xJ^n \subset I \text{ for some } n\}`.
        """
        raise NotImplementedError
        # Note this can be implemented using repeated quotient

    def union(self, J):
        """
        Compute the ideal generated by the union of ``self`` and ``J``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x**2 - 1).union(QQ.old_poly_ring(x).ideal((x+1)**2)) == QQ.old_poly_ring(x).ideal(x+1)
        True
        """
        self._check_ideal(J)
        return self._union(J)

    def product(self, J):
        r"""
        Compute the ideal product of ``self`` and ``J``.

        That is, compute the ideal generated by products `xy`, for `x` an element
        of ``self`` and `y \in J`.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x, y).ideal(x).product(QQ.old_poly_ring(x, y).ideal(y))
        <x*y>
        """
        self._check_ideal(J)
        return self._product(J)

    def reduce_element(self, x):
        """
        Reduce the element ``x`` of our ring modulo the ideal ``self``.

        Here "reduce" has no specific meaning: it could return a unique normal
        form, simplify the expression a bit, or just do nothing.
        """
        return x

    def __add__(self, e):
        if not isinstance(e, Ideal):
            R = self.ring.quotient_ring(self)
            if isinstance(e, R.dtype):
                return e
            if isinstance(e, R.ring.dtype):
                return R(e)
            return R.convert(e)
        self._check_ideal(e)
        return self.union(e)

    __radd__ = __add__

    def __mul__(self, e):
        if not isinstance(e, Ideal):
            try:
                e = self.ring.ideal(e)
            except CoercionFailed:
                return NotImplemented
        self._check_ideal(e)
        return self.product(e)

    __rmul__ = __mul__

    def _zeroth_power(self):
        return self.ring.ideal(1)

    def _first_power(self):
        # Raising to any power but 1 returns a new instance. So we mult by 1
        # here so that the first power is no exception.
        return self * 1

    def __eq__(self, e):
        if not isinstance(e, Ideal) or e.ring != self.ring:
            return False
        return self._equals(e)

    def __ne__(self, e):
        return not (self == e)


class ModuleImplementedIdeal(Ideal):
    """
    Ideal implementation relying on the modules code.

    Attributes:

    - _module - the underlying module
    """

    def __init__(self, ring, module):
        Ideal.__init__(self, ring)
        self._module = module

    def _contains_elem(self, x):
        return self._module.contains([x])

    def _contains_ideal(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.is_submodule(J._module)

    def _intersect(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.intersect(J._module))

    def _quotient(self, J, **opts):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.module_quotient(J._module, **opts)

    def _union(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.union(J._module))

    @property
    def gens(self):
        """
        Return generators for ``self``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x, y
        >>> list(QQ.old_poly_ring(x, y).ideal(x, y, x**2 + y).gens)
        [DMP_Python([[1], []], QQ), DMP_Python([[1, 0]], QQ), DMP_Python([[1], [], [1, 0]], QQ)]
        """
        return (x[0] for x in self._module.gens)

    def is_zero(self):
        """
        Return True if ``self`` is the zero ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x).is_zero()
        False
        >>> QQ.old_poly_ring(x).ideal().is_zero()
        True
        """
        return self._module.is_zero()

    def is_whole_ring(self):
        """
        Return True if ``self`` is the whole ring, i.e. one generator is a unit.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ, ilex
        >>> QQ.old_poly_ring(x).ideal(x).is_whole_ring()
        False
        >>> QQ.old_poly_ring(x).ideal(3).is_whole_ring()
        True
        >>> QQ.old_poly_ring(x, order=ilex).ideal(2 + x).is_whole_ring()
        True
        """
        return self._module.is_full_module()

    def __repr__(self):
        from sympy.printing.str import sstr
        gens = [self.ring.to_sympy(x) for [x] in self._module.gens]
        return '<' + ','.join(sstr(g) for g in gens) + '>'

    # NOTE this is the only method using the fact that the module is a SubModule
    def _product(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.submodule(
            *[[x*y] for [x] in self._module.gens for [y] in J._module.gens]))

    def in_terms_of_generators(self, e):
        """
        Express ``e`` in terms of the generators of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x**2 + 1, x)
        >>> I.in_terms_of_generators(1)  # doctest: +SKIP
        [DMP_Python([1], QQ), DMP_Python([-1, 0], QQ)]
        """
        return self._module.in_terms_of_generators([e])

    def reduce_element(self, x, **options):
        return self._module.reduce_element([x], **options)[0]
