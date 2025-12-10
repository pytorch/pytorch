"""
Computations with modules over polynomial rings.

This module implements various classes that encapsulate groebner basis
computations for modules. Most of them should not be instantiated by hand.
Instead, use the constructing routines on objects you already have.

For example, to construct a free module over ``QQ[x, y]``, call
``QQ[x, y].free_module(rank)`` instead of the ``FreeModule`` constructor.
In fact ``FreeModule`` is an abstract base class that should not be
instantiated, the ``free_module`` method instead returns the implementing class
``FreeModulePolyRing``.

In general, the abstract base classes implement most functionality in terms of
a few non-implemented methods. The concrete base classes supply only these
non-implemented methods. They may also supply new implementations of the
convenience methods, for example if there are faster algorithms available.
"""


from copy import copy
from functools import reduce

from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable

# TODO
# - module saturation
# - module quotient/intersection for quotient rings
# - free resoltutions / syzygies
# - finding small/minimal generating sets
# - ...

##########################################################################
## Abstract base classes #################################################
##########################################################################


class Module:
    """
    Abstract base class for modules.

    Do not instantiate - use ring explicit constructors instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> QQ.old_poly_ring(x).free_module(2)
    QQ[x]**2

    Attributes:

    - dtype - type of elements
    - ring - containing ring

    Non-implemented methods:

    - submodule
    - quotient_module
    - is_zero
    - is_submodule
    - multiply_ideal

    The method convert likely needs to be changed in subclasses.
    """

    def __init__(self, ring):
        self.ring = ring

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into internal representation of this module.

        If ``M`` is not None, it should be a module containing it.
        """
        if not isinstance(elem, self.dtype):
            raise CoercionFailed
        return elem

    def submodule(self, *gens):
        """Generate a submodule."""
        raise NotImplementedError

    def quotient_module(self, other):
        """Generate a quotient module."""
        raise NotImplementedError

    def __truediv__(self, e):
        if not isinstance(e, Module):
            e = self.submodule(*e)
        return self.quotient_module(e)

    def contains(self, elem):
        """Return True if ``elem`` is an element of this module."""
        try:
            self.convert(elem)
            return True
        except CoercionFailed:
            return False

    def __contains__(self, elem):
        return self.contains(elem)

    def subset(self, other):
        """
        Returns True if ``other`` is is a subset of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.subset([(1, x), (x, 2)])
        True
        >>> F.subset([(1/x, x), (x, 2)])
        False
        """
        return all(self.contains(x) for x in other)

    def __eq__(self, other):
        return self.is_submodule(other) and other.is_submodule(self)

    def __ne__(self, other):
        return not (self == other)

    def is_zero(self):
        """Returns True if ``self`` is a zero module."""
        raise NotImplementedError

    def is_submodule(self, other):
        """Returns True if ``other`` is a submodule of ``self``."""
        raise NotImplementedError

    def multiply_ideal(self, other):
        """
        Multiply ``self`` by the ideal ``other``.
        """
        raise NotImplementedError

    def __mul__(self, e):
        if not isinstance(e, Ideal):
            try:
                e = self.ring.ideal(e)
            except (CoercionFailed, NotImplementedError):
                return NotImplemented
        return self.multiply_ideal(e)

    __rmul__ = __mul__

    def identity_hom(self):
        """Return the identity homomorphism on ``self``."""
        raise NotImplementedError


class ModuleElement:
    """
    Base class for module element wrappers.

    Use this class to wrap primitive data types as module elements. It stores
    a reference to the containing module, and implements all the arithmetic
    operators.

    Attributes:

    - module - containing module
    - data - internal data

    Methods that likely need change in subclasses:

    - add
    - mul
    - div
    - eq
    """

    def __init__(self, module, data):
        self.module = module
        self.data = data

    def add(self, d1, d2):
        """Add data ``d1`` and ``d2``."""
        return d1 + d2

    def mul(self, m, d):
        """Multiply module data ``m`` by coefficient d."""
        return m * d

    def div(self, m, d):
        """Divide module data ``m`` by coefficient d."""
        return m / d

    def eq(self, d1, d2):
        """Return true if d1 and d2 represent the same element."""
        return d1 == d2

    def __add__(self, om):
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return NotImplemented
        return self.__class__(self.module, self.add(self.data, om.data))

    __radd__ = __add__

    def __neg__(self):
        return self.__class__(self.module, self.mul(self.data,
                       self.module.ring.convert(-1)))

    def __sub__(self, om):
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return NotImplemented
        return self.__add__(-om)

    def __rsub__(self, om):
        return (-self).__add__(om)

    def __mul__(self, o):
        if not isinstance(o, self.module.ring.dtype):
            try:
                o = self.module.ring.convert(o)
            except CoercionFailed:
                return NotImplemented
        return self.__class__(self.module, self.mul(self.data, o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if not isinstance(o, self.module.ring.dtype):
            try:
                o = self.module.ring.convert(o)
            except CoercionFailed:
                return NotImplemented
        return self.__class__(self.module, self.div(self.data, o))

    def __eq__(self, om):
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return False
        return self.eq(self.data, om.data)

    def __ne__(self, om):
        return not self == om

##########################################################################
## Free Modules ##########################################################
##########################################################################


class FreeModuleElement(ModuleElement):
    """Element of a free module. Data stored as a tuple."""

    def add(self, d1, d2):
        return tuple(x + y for x, y in zip(d1, d2))

    def mul(self, d, p):
        return tuple(x * p for x in d)

    def div(self, d, p):
        return tuple(x / p for x in d)

    def __repr__(self):
        from sympy.printing.str import sstr
        data = self.data
        if any(isinstance(x, DMP) for x in data):
            data = [self.module.ring.to_sympy(x) for x in data]
        return '[' + ', '.join(sstr(x) for x in data) + ']'

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, idx):
        return self.data[idx]


class FreeModule(Module):
    """
    Abstract base class for free modules.

    Additional attributes:

    - rank - rank of the free module

    Non-implemented methods:

    - submodule
    """

    dtype = FreeModuleElement

    def __init__(self, ring, rank):
        Module.__init__(self, ring)
        self.rank = rank

    def __repr__(self):
        return repr(self.ring) + "**" + repr(self.rank)

    def is_submodule(self, other):
        """
        Returns True if ``other`` is a submodule of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([2, x])
        >>> F.is_submodule(F)
        True
        >>> F.is_submodule(M)
        True
        >>> M.is_submodule(F)
        False
        """
        if isinstance(other, SubModule):
            return other.container == self
        if isinstance(other, FreeModule):
            return other.ring == self.ring and other.rank == self.rank
        return False

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        This method is called implicitly whenever computations involve elements
        not in the internal representation.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.convert([1, 0])
        [1, 0]
        """
        if isinstance(elem, FreeModuleElement):
            if elem.module is self:
                return elem
            if elem.module.rank != self.rank:
                raise CoercionFailed
            return FreeModuleElement(self,
                     tuple(self.ring.convert(x, elem.module.ring) for x in elem.data))
        elif iterable(elem):
            tpl = tuple(self.ring.convert(x) for x in elem)
            if len(tpl) != self.rank:
                raise CoercionFailed
            return FreeModuleElement(self, tpl)
        elif _aresame(elem, 0):
            return FreeModuleElement(self, (self.ring.convert(0),)*self.rank)
        else:
            raise CoercionFailed

    def is_zero(self):
        """
        Returns True if ``self`` is a zero module.

        (If, as this implementation assumes, the coefficient ring is not the
        zero ring, then this is equivalent to the rank being zero.)

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(0).is_zero()
        True
        >>> QQ.old_poly_ring(x).free_module(1).is_zero()
        False
        """
        return self.rank == 0

    def basis(self):
        """
        Return a set of basis elements.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(3).basis()
        ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        """
        from sympy.matrices import eye
        M = eye(self.rank)
        return tuple(self.convert(M.row(i)) for i in range(self.rank))

    def quotient_module(self, submodule):
        """
        Return a quotient module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2)
        >>> M.quotient_module(M.submodule([1, x], [x, 2]))
        QQ[x]**2/<[1, x], [x, 2]>

        Or more conicisely, using the overloaded division operator:

        >>> QQ.old_poly_ring(x).free_module(2) / [[1, x], [x, 2]]
        QQ[x]**2/<[1, x], [x, 2]>
        """
        return QuotientModule(self.ring, self, submodule)

    def multiply_ideal(self, other):
        """
        Multiply ``self`` by the ideal ``other``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x)
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.multiply_ideal(I)
        <[x, 0], [0, x]>
        """
        return self.submodule(*self.basis()).multiply_ideal(other)

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).identity_hom()
        Matrix([
        [1, 0], : QQ[x]**2 -> QQ[x]**2
        [0, 1]])
        """
        from sympy.polys.agca.homomorphisms import homomorphism
        return homomorphism(self, self, self.basis())


class FreeModulePolyRing(FreeModule):
    """
    Free module over a generalized polynomial ring.

    Do not instantiate this, use the constructor method of the ring instead:

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x).free_module(3)
    >>> F
    QQ[x]**3
    >>> F.contains([x, 1, 0])
    True
    >>> F.contains([1/x, 0, 1])
    False
    """

    def __init__(self, ring, rank):
        from sympy.polys.domains.old_polynomialring import PolynomialRingBase
        FreeModule.__init__(self, ring, rank)
        if not isinstance(ring, PolynomialRingBase):
            raise NotImplementedError('This implementation only works over '
                                      + 'polynomial rings, got %s' % ring)
        if not isinstance(ring.dom, Field):
            raise NotImplementedError('Ground domain must be a field, '
                                      + 'got %s' % ring.dom)

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x, y).free_module(2).submodule([x, x + y])
        >>> M
        <[x, x + y]>
        >>> M.contains([2*x, 2*x + 2*y])
        True
        >>> M.contains([x, y])
        False
        """
        return SubModulePolyRing(gens, self, **opts)


class FreeModuleQuotientRing(FreeModule):
    """
    Free module over a quotient ring.

    Do not instantiate this, use the constructor method of the ring instead:

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(3)
    >>> F
    (QQ[x]/<x**2 + 1>)**3

    Attributes

    - quot - the quotient module `R^n / IR^n`, where `R/I` is our ring
    """

    def __init__(self, ring, rank):
        from sympy.polys.domains.quotientring import QuotientRing
        FreeModule.__init__(self, ring, rank)
        if not isinstance(ring, QuotientRing):
            raise NotImplementedError('This implementation only works over '
                             + 'quotient rings, got %s' % ring)
        F = self.ring.ring.free_module(self.rank)
        self.quot = F / (self.ring.base_ideal*F)

    def __repr__(self):
        return "(" + repr(self.ring) + ")" + "**" + repr(self.rank)

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> M = (QQ.old_poly_ring(x, y)/[x**2 - y**2]).free_module(2).submodule([x, x + y])
        >>> M
        <[x + <x**2 - y**2>, x + y + <x**2 - y**2>]>
        >>> M.contains([y**2, x**2 + x*y])
        True
        >>> M.contains([x, y])
        False
        """
        return SubModuleQuotientRing(gens, self, **opts)

    def lift(self, elem):
        """
        Lift the element ``elem`` of self to the module self.quot.

        Note that self.quot is the same set as self, just as an R-module
        and not as an R/I-module, so this makes sense.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        >>> e = F.convert([1, 0])
        >>> e
        [1 + <x**2 + 1>, 0 + <x**2 + 1>]
        >>> L = F.quot
        >>> l = F.lift(e)
        >>> l
        [1, 0] + <[x**2 + 1, 0], [0, x**2 + 1]>
        >>> L.contains(l)
        True
        """
        return self.quot.convert([x.data for x in elem])

    def unlift(self, elem):
        """
        Push down an element of self.quot to self.

        This undoes ``lift``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        >>> e = F.convert([1, 0])
        >>> l = F.lift(e)
        >>> e == l
        False
        >>> e == F.unlift(l)
        True
        """
        return self.convert(elem.data)

##########################################################################
## Submodules and subquotients ###########################################
##########################################################################


class SubModule(Module):
    """
    Base class for submodules.

    Attributes:

    - container - containing module
    - gens - generators (subset of containing module)
    - rank - rank of containing module

    Non-implemented methods:

    - _contains
    - _syzygies
    - _in_terms_of_generators
    - _intersect
    - _module_quotient

    Methods that likely need change in subclasses:

    - reduce_element
    """

    def __init__(self, gens, container):
        Module.__init__(self, container.ring)
        self.gens = tuple(container.convert(x) for x in gens)
        self.container = container
        self.rank = container.rank
        self.ring = container.ring
        self.dtype = container.dtype

    def __repr__(self):
        return "<" + ", ".join(repr(x) for x in self.gens) + ">"

    def _contains(self, other):
        """Implementation of containment.
           Other is guaranteed to be FreeModuleElement."""
        raise NotImplementedError

    def _syzygies(self):
        """Implementation of syzygy computation wrt self generators."""
        raise NotImplementedError

    def _in_terms_of_generators(self, e):
        """Implementation of expression in terms of generators."""
        raise NotImplementedError

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal represantition.

        Mostly called implicitly.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([1, x])
        >>> M.convert([2, 2*x])
        [2, 2*x]
        """
        if isinstance(elem, self.container.dtype) and elem.module is self:
            return elem
        r = copy(self.container.convert(elem, M))
        r.module = self
        if not self._contains(r):
            raise CoercionFailed
        return r

    def _intersect(self, other):
        """Implementation of intersection.
           Other is guaranteed to be a submodule of same free module."""
        raise NotImplementedError

    def _module_quotient(self, other):
        """Implementation of quotient.
           Other is guaranteed to be a submodule of same free module."""
        raise NotImplementedError

    def intersect(self, other, **options):
        """
        Returns the intersection of ``self`` with submodule ``other``.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x, y).free_module(2)
        >>> F.submodule([x, x]).intersect(F.submodule([y, y]))
        <[x*y, x*y]>

        Some implementation allow further options to be passed. Currently, to
        only one implemented is ``relations=True``, in which case the function
        will return a triple ``(res, rela, relb)``, where ``res`` is the
        intersection module, and ``rela`` and ``relb`` are lists of coefficient
        vectors, expressing the generators of ``res`` in terms of the
        generators of ``self`` (``rela``) and ``other`` (``relb``).

        >>> F.submodule([x, x]).intersect(F.submodule([y, y]), relations=True)
        (<[x*y, x*y]>, [(DMP_Python([[1, 0]], QQ),)], [(DMP_Python([[1], []], QQ),)])

        The above result says: the intersection module is generated by the
        single element `(-xy, -xy) = -y (x, x) = -x (y, y)`, where
        `(x, x)` and `(y, y)` respectively are the unique generators of
        the two modules being intersected.
        """
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        return self._intersect(other, **options)

    def module_quotient(self, other, **options):
        r"""
        Returns the module quotient of ``self`` by submodule ``other``.

        That is, if ``self`` is the module `M` and ``other`` is `N`, then
        return the ideal `\{f \in R | fN \subset M\}`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x, y
        >>> F = QQ.old_poly_ring(x, y).free_module(2)
        >>> S = F.submodule([x*y, x*y])
        >>> T = F.submodule([x, x])
        >>> S.module_quotient(T)
        <y>

        Some implementations allow further options to be passed. Currently, the
        only one implemented is ``relations=True``, which may only be passed
        if ``other`` is principal. In this case the function
        will return a pair ``(res, rel)`` where ``res`` is the ideal, and
        ``rel`` is a list of coefficient vectors, expressing the generators of
        the ideal, multiplied by the generator of ``other`` in terms of
        generators of ``self``.

        >>> S.module_quotient(T, relations=True)
        (<y>, [[DMP_Python([[1]], QQ)]])

        This means that the quotient ideal is generated by the single element
        `y`, and that `y (x, x) = 1 (xy, xy)`, `(x, x)` and `(xy, xy)` being
        the generators of `T` and `S`, respectively.
        """
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        return self._module_quotient(other, **options)

    def union(self, other):
        """
        Returns the module generated by the union of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(1)
        >>> M = F.submodule([x**2 + x]) # <x(x+1)>
        >>> N = F.submodule([x**2 - 1]) # <(x-1)(x+1)>
        >>> M.union(N) == F.submodule([x+1])
        True
        """
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        return self.__class__(self.gens + other.gens, self.container)

    def is_zero(self):
        """
        Return True if ``self`` is a zero module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_zero()
        False
        >>> F.submodule([0, 0]).is_zero()
        True
        """
        return all(x == 0 for x in self.gens)

    def submodule(self, *gens):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([x, 1])
        >>> M.submodule([x**2, x])
        <[x**2, x]>
        """
        if not self.subset(gens):
            raise ValueError('%s not a subset of %s' % (gens, self))
        return self.__class__(gens, self.container)

    def is_full_module(self):
        """
        Return True if ``self`` is the entire free module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_full_module()
        False
        >>> F.submodule([1, 1], [1, 2]).is_full_module()
        True
        """
        return all(self.contains(x) for x in self.container.basis())

    def is_submodule(self, other):
        """
        Returns True if ``other`` is a submodule of ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([2, x])
        >>> N = M.submodule([2*x, x**2])
        >>> M.is_submodule(M)
        True
        >>> M.is_submodule(N)
        True
        >>> N.is_submodule(M)
        False
        """
        if isinstance(other, SubModule):
            return self.container == other.container and \
                all(self.contains(x) for x in other.gens)
        if isinstance(other, (FreeModule, QuotientModule)):
            return self.container == other and self.is_full_module()
        return False

    def syzygy_module(self, **opts):
        r"""
        Compute the syzygy module of the generators of ``self``.

        Suppose `M` is generated by `f_1, \ldots, f_n` over the ring
        `R`. Consider the homomorphism `\phi: R^n \to M`, given by
        sending `(r_1, \ldots, r_n) \to r_1 f_1 + \cdots + r_n f_n`.
        The syzygy module is defined to be the kernel of `\phi`.

        Examples
        ========

        The syzygy module is zero iff the generators generate freely a free
        submodule:

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([1, 0], [1, 1]).syzygy_module().is_zero()
        True

        A slightly more interesting example:

        >>> M = QQ.old_poly_ring(x, y).free_module(2).submodule([x, 2*x], [y, 2*y])
        >>> S = QQ.old_poly_ring(x, y).free_module(2).submodule([y, -x])
        >>> M.syzygy_module() == S
        True
        """
        F = self.ring.free_module(len(self.gens))
        # NOTE we filter out zero syzygies. This is for convenience of the
        # _syzygies function and not meant to replace any real "generating set
        # reduction" algorithm
        return F.submodule(*[x for x in self._syzygies() if F.convert(x) != 0],
                           **opts)

    def in_terms_of_generators(self, e):
        """
        Express element ``e`` of ``self`` in terms of the generators.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([1, 0], [1, 1])
        >>> M.in_terms_of_generators([x, x**2])  # doctest: +SKIP
        [DMP_Python([-1, 1, 0], QQ), DMP_Python([1, 0, 0], QQ)]
        """
        try:
            e = self.convert(e)
        except CoercionFailed:
            raise ValueError('%s is not an element of %s' % (e, self))
        return self._in_terms_of_generators(e)

    def reduce_element(self, x):
        """
        Reduce the element ``x`` of our ring modulo the ideal ``self``.

        Here "reduce" has no specific meaning, it could return a unique normal
        form, simplify the expression a bit, or just do nothing.
        """
        return x

    def quotient_module(self, other, **opts):
        """
        Return a quotient module.

        This is the same as taking a submodule of a quotient of the containing
        module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> S1 = F.submodule([x, 1])
        >>> S2 = F.submodule([x**2, x])
        >>> S1.quotient_module(S2)
        <[x, 1] + <[x**2, x]>>

        Or more coincisely, using the overloaded division operator:

        >>> F.submodule([x, 1]) / [(x**2, x)]
        <[x, 1] + <[x**2, x]>>
        """
        if not self.is_submodule(other):
            raise ValueError('%s not a submodule of %s' % (other, self))
        return SubQuotientModule(self.gens,
                self.container.quotient_module(other), **opts)

    def __add__(self, oth):
        return self.container.quotient_module(self).convert(oth)

    __radd__ = __add__

    def multiply_ideal(self, I):
        """
        Multiply ``self`` by the ideal ``I``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x**2)
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([1, 1])
        >>> I*M
        <[x**2, x**2]>
        """
        return self.submodule(*[x*g for [x] in I._module.gens for g in self.gens])

    def inclusion_hom(self):
        """
        Return a homomorphism representing the inclusion map of ``self``.

        That is, the natural map from ``self`` to ``self.container``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([x, x]).inclusion_hom()
        Matrix([
        [1, 0], : <[x, x]> -> QQ[x]**2
        [0, 1]])
        """
        return self.container.identity_hom().restrict_domain(self)

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([x, x]).identity_hom()
        Matrix([
        [1, 0], : <[x, x]> -> <[x, x]>
        [0, 1]])
        """
        return self.container.identity_hom().restrict_domain(
            self).restrict_codomain(self)


class SubQuotientModule(SubModule):
    """
    Submodule of a quotient module.

    Equivalently, quotient module of a submodule.

    Do not instantiate this, instead use the submodule or quotient_module
    constructing methods:

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> S = F.submodule([1, 0], [1, x])
    >>> Q = F/[(1, 0)]
    >>> S/[(1, 0)] == Q.submodule([5, x])
    True

    Attributes:

    - base - base module we are quotient of
    - killed_module - submodule used to form the quotient
    """
    def __init__(self, gens, container, **opts):
        SubModule.__init__(self, gens, container)
        self.killed_module = self.container.killed_module
        # XXX it is important for some code below that the generators of base
        #     are in this particular order!
        self.base = self.container.base.submodule(
            *[x.data for x in self.gens], **opts).union(self.killed_module)

    def _contains(self, elem):
        return self.base.contains(elem.data)

    def _syzygies(self):
        # let N = self.killed_module be generated by e_1, ..., e_r
        # let F = self.base be generated by f_1, ..., f_s and e_1, ..., e_r
        # Then self = F/N.
        # Let phi: R**s --> self be the evident surjection.
        # Similarly psi: R**(s + r) --> F.
        # We need to find generators for ker(phi). Let chi: R**s --> F be the
        # evident lift of phi. For X in R**s, phi(X) = 0 iff chi(X) is
        # contained in N, iff there exists Y in R**r such that
        # psi(X, Y) = 0.
        # Hence if alpha: R**(s + r) --> R**s is the projection map, then
        # ker(phi) = alpha ker(psi).
        return [X[:len(self.gens)] for X in self.base._syzygies()]

    def _in_terms_of_generators(self, e):
        return self.base._in_terms_of_generators(e.data)[:len(self.gens)]

    def is_full_module(self):
        """
        Return True if ``self`` is the entire free module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_full_module()
        False
        >>> F.submodule([1, 1], [1, 2]).is_full_module()
        True
        """
        return self.base.is_full_module()

    def quotient_hom(self):
        """
        Return the quotient homomorphism to self.

        That is, return the natural map from ``self.base`` to ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = (QQ.old_poly_ring(x).free_module(2) / [(1, x)]).submodule([1, 0])
        >>> M.quotient_hom()
        Matrix([
        [1, 0], : <[1, 0], [1, x]> -> <[1, 0] + <[1, x]>, [1, x] + <[1, x]>>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(self.killed_module)


_subs0 = lambda x: x[0]
_subs1 = lambda x: x[1:]


class ModuleOrder(ProductOrder):
    """A product monomial order with a zeroth term as module index."""

    def __init__(self, o1, o2, TOP):
        if TOP:
            ProductOrder.__init__(self, (o2, _subs1), (o1, _subs0))
        else:
            ProductOrder.__init__(self, (o1, _subs0), (o2, _subs1))


class SubModulePolyRing(SubModule):
    """
    Submodule of a free module over a generalized polynomial ring.

    Do not instantiate this, use the constructor method of FreeModule instead:

    >>> from sympy.abc import x, y
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x, y).free_module(2)
    >>> F.submodule([x, y], [1, 0])
    <[x, y], [1, 0]>

    Attributes:

    - order - monomial order used
    """

    #self._gb - cached groebner basis
    #self._gbe - cached groebner basis relations

    def __init__(self, gens, container, order="lex", TOP=True):
        SubModule.__init__(self, gens, container)
        if not isinstance(container, FreeModulePolyRing):
            raise NotImplementedError('This implementation is for submodules of '
                             + 'FreeModulePolyRing, got %s' % container)
        self.order = ModuleOrder(monomial_key(order), self.ring.order, TOP)
        self._gb = None
        self._gbe = None

    def __eq__(self, other):
        if isinstance(other, SubModulePolyRing) and self.order != other.order:
            return False
        return SubModule.__eq__(self, other)

    def _groebner(self, extended=False):
        """Returns a standard basis in sdm form."""
        from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_mora
        if self._gbe is None and extended:
            gb, gbe = sdm_groebner(
                [self.ring._vector_to_sdm(x, self.order) for x in self.gens],
                sdm_nf_mora, self.order, self.ring.dom, extended=True)
            self._gb, self._gbe = tuple(gb), tuple(gbe)
        if self._gb is None:
            self._gb = tuple(sdm_groebner(
                             [self.ring._vector_to_sdm(x, self.order) for x in self.gens],
               sdm_nf_mora, self.order, self.ring.dom))
        if extended:
            return self._gb, self._gbe
        else:
            return self._gb

    def _groebner_vec(self, extended=False):
        """Returns a standard basis in element form."""
        if not extended:
            return [FreeModuleElement(self,
                        tuple(self.ring._sdm_to_vector(x, self.rank)))
                    for x in self._groebner()]
        gb, gbe = self._groebner(extended=True)
        return ([self.convert(self.ring._sdm_to_vector(x, self.rank))
                 for x in gb],
                [self.ring._sdm_to_vector(x, len(self.gens)) for x in gbe])

    def _contains(self, x):
        from sympy.polys.distributedmodules import sdm_zero, sdm_nf_mora
        return sdm_nf_mora(self.ring._vector_to_sdm(x, self.order),
                           self._groebner(), self.order, self.ring.dom) == \
            sdm_zero()

    def _syzygies(self):
        """Compute syzygies. See [SCA, algorithm 2.5.4]."""
        # NOTE if self.gens is a standard basis, this can be done more
        #      efficiently using Schreyer's theorem

        # First bullet point
        k = len(self.gens)
        r = self.rank
        zero = self.ring.convert(0)
        one = self.ring.convert(1)
        Rkr = self.ring.free_module(r + k)
        newgens = []
        for j, f in enumerate(self.gens):
            m = [0]*(r + k)
            for i, v in enumerate(f):
                m[i] = v
            for i in range(k):
                m[r + i] = one if j == i else zero
            m = FreeModuleElement(Rkr, tuple(m))
            newgens.append(m)
        # Note: we need *descending* order on module index, and TOP=False to
        #       get an elimination order
        F = Rkr.submodule(*newgens, order='ilex', TOP=False)

        # Second bullet point: standard basis of F
        G = F._groebner_vec()

        # Third bullet point: G0 = G intersect the new k components
        G0 = [x[r:] for x in G if all(y == zero for y in x[:r])]

        # Fourth and fifth bullet points: we are done
        return G0

    def _in_terms_of_generators(self, e):
        """Expression in terms of generators. See [SCA, 2.8.1]."""
        # NOTE: if gens is a standard basis, this can be done more efficiently
        M = self.ring.free_module(self.rank).submodule(*((e,) + self.gens))
        S = M.syzygy_module(
            order="ilex", TOP=False)  # We want decreasing order!
        G = S._groebner_vec()
        # This list cannot not be empty since e is an element
        e = [x for x in G if self.ring.is_unit(x[0])][0]
        return [-x/e[0] for x in e[1:]]

    def reduce_element(self, x, NF=None):
        """
        Reduce the element ``x`` of our container modulo ``self``.

        This applies the normal form ``NF`` to ``x``. If ``NF`` is passed
        as none, the default Mora normal form is used (which is not unique!).
        """
        from sympy.polys.distributedmodules import sdm_nf_mora
        if NF is None:
            NF = sdm_nf_mora
        return self.container.convert(self.ring._sdm_to_vector(NF(
            self.ring._vector_to_sdm(x, self.order), self._groebner(),
            self.order, self.ring.dom),
            self.rank))

    def _intersect(self, other, relations=False):
        # See: [SCA, section 2.8.2]
        fi = self.gens
        hi = other.gens
        r = self.rank
        ci = [[0]*(2*r) for _ in range(r)]
        for k in range(r):
            ci[k][k] = 1
            ci[k][r + k] = 1
        di = [list(f) + [0]*r for f in fi]
        ei = [[0]*r + list(h) for h in hi]
        syz = self.ring.free_module(2*r).submodule(*(ci + di + ei))._syzygies()
        nonzero = [x for x in syz if any(y != self.ring.zero for y in x[:r])]
        res = self.container.submodule(*([-y for y in x[:r]] for x in nonzero))
        reln1 = [x[r:r + len(fi)] for x in nonzero]
        reln2 = [x[r + len(fi):] for x in nonzero]
        if relations:
            return res, reln1, reln2
        return res

    def _module_quotient(self, other, relations=False):
        # See: [SCA, section 2.8.4]
        if relations and len(other.gens) != 1:
            raise NotImplementedError
        if len(other.gens) == 0:
            return self.ring.ideal(1)
        elif len(other.gens) == 1:
            # We do some trickery. Let f be the (vector!) generating ``other``
            # and f1, .., fn be the (vectors) generating self.
            # Consider the submodule of R^{r+1} generated by (f, 1) and
            # {(fi, 0) | i}. Then the intersection with the last module
            # component yields the quotient.
            g1 = list(other.gens[0]) + [1]
            gi = [list(x) + [0] for x in self.gens]
            # NOTE: We *need* to use an elimination order
            M = self.ring.free_module(self.rank + 1).submodule(*([g1] + gi),
                                            order='ilex', TOP=False)
            if not relations:
                return self.ring.ideal(*[x[-1] for x in M._groebner_vec() if
                                         all(y == self.ring.zero for y in x[:-1])])
            else:
                G, R = M._groebner_vec(extended=True)
                indices = [i for i, x in enumerate(G) if
                           all(y == self.ring.zero for y in x[:-1])]
                return (self.ring.ideal(*[G[i][-1] for i in indices]),
                        [[-x for x in R[i][1:]] for i in indices])
        # For more generators, we use I : <h1, .., hn> = intersection of
        #                                    {I : <hi> | i}
        # TODO this can be done more efficiently
        return reduce(lambda x, y: x.intersect(y),
            (self._module_quotient(self.container.submodule(x)) for x in other.gens))


class SubModuleQuotientRing(SubModule):
    """
    Class for submodules of free modules over quotient rings.

    Do not instantiate this. Instead use the submodule methods.

    >>> from sympy.abc import x, y
    >>> from sympy import QQ
    >>> M = (QQ.old_poly_ring(x, y)/[x**2 - y**2]).free_module(2).submodule([x, x + y])
    >>> M
    <[x + <x**2 - y**2>, x + y + <x**2 - y**2>]>
    >>> M.contains([y**2, x**2 + x*y])
    True
    >>> M.contains([x, y])
    False

    Attributes:

    - quot - the subquotient of `R^n/IR^n` generated by lifts of our generators
    """

    def __init__(self, gens, container):
        SubModule.__init__(self, gens, container)
        self.quot = self.container.quot.submodule(
            *[self.container.lift(x) for x in self.gens])

    def _contains(self, elem):
        return self.quot._contains(self.container.lift(elem))

    def _syzygies(self):
        return [tuple(self.ring.convert(y, self.quot.ring) for y in x)
                for x in self.quot._syzygies()]

    def _in_terms_of_generators(self, elem):
        return [self.ring.convert(x, self.quot.ring) for x in
            self.quot._in_terms_of_generators(self.container.lift(elem))]

##########################################################################
## Quotient Modules ######################################################
##########################################################################


class QuotientModuleElement(ModuleElement):
    """Element of a quotient module."""

    def eq(self, d1, d2):
        """Equality comparison."""
        return self.module.killed_module.contains(d1 - d2)

    def __repr__(self):
        return repr(self.data) + " + " + repr(self.module.killed_module)


class QuotientModule(Module):
    """
    Class for quotient modules.

    Do not instantiate this directly. For subquotients, see the
    SubQuotientModule class.

    Attributes:

    - base - the base module we are a quotient of
    - killed_module - the submodule used to form the quotient
    - rank of the base
    """

    dtype = QuotientModuleElement

    def __init__(self, ring, base, submodule):
        Module.__init__(self, ring)
        if not base.is_submodule(submodule):
            raise ValueError('%s is not a submodule of %s' % (submodule, base))
        self.base = base
        self.killed_module = submodule
        self.rank = base.rank

    def __repr__(self):
        return repr(self.base) + "/" + repr(self.killed_module)

    def is_zero(self):
        """
        Return True if ``self`` is a zero module.

        This happens if and only if the base module is the same as the
        submodule being killed.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> (F/[(1, 0)]).is_zero()
        False
        >>> (F/[(1, 0), (0, 1)]).is_zero()
        True
        """
        return self.base == self.killed_module

    def is_submodule(self, other):
        """
        Return True if ``other`` is a submodule of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> Q = QQ.old_poly_ring(x).free_module(2) / [(x, x)]
        >>> S = Q.submodule([1, 0])
        >>> Q.is_submodule(S)
        True
        >>> S.is_submodule(Q)
        False
        """
        if isinstance(other, QuotientModule):
            return self.killed_module == other.killed_module and \
                self.base.is_submodule(other.base)
        if isinstance(other, SubQuotientModule):
            return other.container == self
        return False

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        This is the same as taking a quotient of a submodule of the base
        module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> Q = QQ.old_poly_ring(x).free_module(2) / [(x, x)]
        >>> Q.submodule([x, 0])
        <[x, 0] + <[x, x]>>
        """
        return SubQuotientModule(gens, self, **opts)

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        This method is called implicitly whenever computations involve elements
        not in the internal representation.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> F.convert([1, 0])
        [1, 0] + <[1, 2], [1, x]>
        """
        if isinstance(elem, QuotientModuleElement):
            if elem.module is self:
                return elem
            if self.killed_module.is_submodule(elem.module.killed_module):
                return QuotientModuleElement(self, self.base.convert(elem.data))
            raise CoercionFailed
        return QuotientModuleElement(self, self.base.convert(elem))

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> M.identity_hom()
        Matrix([
        [1, 0], : QQ[x]**2/<[1, 2], [1, x]> -> QQ[x]**2/<[1, 2], [1, x]>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(
            self.killed_module).quotient_domain(self.killed_module)

    def quotient_hom(self):
        """
        Return the quotient homomorphism to ``self``.

        That is, return a homomorphism representing the natural map from
        ``self.base`` to ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> M.quotient_hom()
        Matrix([
        [1, 0], : QQ[x]**2 -> QQ[x]**2/<[1, 2], [1, x]>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(
            self.killed_module)
