from typing import Any, Callable
from functools import reduce
from collections import defaultdict
import inspect

from sympy.core.kind import Kind, UndefinedKind, NumberKind
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import (FuzzyBool, fuzzy_bool, fuzzy_or, fuzzy_and,
    fuzzy_not)
from sympy.core.numbers import Float, Integer
from sympy.core.operations import LatticeOp
from sympy.core.parameters import global_parameters
from sympy.core.relational import Eq, Ne, is_lt
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import And, Or, Not, Xor, true, false
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import (iproduct, sift, roundrobin, iterable,
                                       subsets)
from sympy.utilities.misc import func_name, filldedent

from mpmath import mpi, mpf

from mpmath.libmp.libmpf import prec_to_dps


tfn = defaultdict(lambda: None, {
    True: S.true,
    S.true: S.true,
    False: S.false,
    S.false: S.false})


@sympify_method_args
class Set(Basic, EvalfMixin):
    """
    The base class for any kind of set.

    Explanation
    ===========

    This is not meant to be used directly as a container of items. It does not
    behave like the builtin ``set``; see :class:`FiniteSet` for that.

    Real intervals are represented by the :class:`Interval` class and unions of
    sets by the :class:`Union` class. The empty set is represented by the
    :class:`EmptySet` class and available as a singleton as ``S.EmptySet``.
    """

    __slots__ = ()

    is_number = False
    is_iterable = False
    is_interval = False

    is_FiniteSet = False
    is_Interval = False
    is_ProductSet = False
    is_Union = False
    is_Intersection: FuzzyBool = None
    is_UniversalSet: FuzzyBool = None
    is_Complement: FuzzyBool = None
    is_ComplexRegion = False

    is_empty: FuzzyBool = None
    is_finite_set: FuzzyBool = None

    @property  # type: ignore
    @deprecated(
        """
        The is_EmptySet attribute of Set objects is deprecated.
        Use 's is S.EmptySet" or 's.is_empty' instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-is-emptyset",
    )
    def is_EmptySet(self):
        return None

    @staticmethod
    def _infimum_key(expr):
        """
        Return infimum (if possible) else S.Infinity.
        """
        try:
            infimum = expr.inf
            assert infimum.is_comparable
            infimum = infimum.evalf()  # issue #18505
        except (NotImplementedError,
                AttributeError, AssertionError, ValueError):
            infimum = S.Infinity
        return infimum

    def union(self, other):
        """
        Returns the union of ``self`` and ``other``.

        Examples
        ========

        As a shortcut it is possible to use the ``+`` operator:

        >>> from sympy import Interval, FiniteSet
        >>> Interval(0, 1).union(Interval(2, 3))
        Union(Interval(0, 1), Interval(2, 3))
        >>> Interval(0, 1) + Interval(2, 3)
        Union(Interval(0, 1), Interval(2, 3))
        >>> Interval(1, 2, True, True) + FiniteSet(2, 3)
        Union({3}, Interval.Lopen(1, 2))

        Similarly it is possible to use the ``-`` operator for set differences:

        >>> Interval(0, 2) - Interval(0, 1)
        Interval.Lopen(1, 2)
        >>> Interval(1, 3) - FiniteSet(2)
        Union(Interval.Ropen(1, 2), Interval.Lopen(2, 3))

        """
        return Union(self, other)

    def intersect(self, other):
        """
        Returns the intersection of 'self' and 'other'.

        Examples
        ========

        >>> from sympy import Interval

        >>> Interval(1, 3).intersect(Interval(1, 2))
        Interval(1, 2)

        >>> from sympy import imageset, Lambda, symbols, S
        >>> n, m = symbols('n m')
        >>> a = imageset(Lambda(n, 2*n), S.Integers)
        >>> a.intersect(imageset(Lambda(m, 2*m + 1), S.Integers))
        EmptySet

        """
        return Intersection(self, other)

    def intersection(self, other):
        """
        Alias for :meth:`intersect()`
        """
        return self.intersect(other)

    def is_disjoint(self, other):
        """
        Returns True if ``self`` and ``other`` are disjoint.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 2).is_disjoint(Interval(1, 2))
        False
        >>> Interval(0, 2).is_disjoint(Interval(3, 4))
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Disjoint_sets
        """
        return self.intersect(other) == S.EmptySet

    def isdisjoint(self, other):
        """
        Alias for :meth:`is_disjoint()`
        """
        return self.is_disjoint(other)

    def complement(self, universe):
        r"""
        The complement of 'self' w.r.t the given universe.

        Examples
        ========

        >>> from sympy import Interval, S
        >>> Interval(0, 1).complement(S.Reals)
        Union(Interval.open(-oo, 0), Interval.open(1, oo))

        >>> Interval(0, 1).complement(S.UniversalSet)
        Complement(UniversalSet, Interval(0, 1))

        """
        return Complement(universe, self)

    def _complement(self, other):
        # this behaves as other - self
        if isinstance(self, ProductSet) and isinstance(other, ProductSet):
            # If self and other are disjoint then other - self == self
            if len(self.sets) != len(other.sets):
                return other

            # There can be other ways to represent this but this gives:
            # (A x B) - (C x D) = ((A - C) x B) U (A x (B - D))
            overlaps = []
            pairs = list(zip(self.sets, other.sets))
            for n in range(len(pairs)):
                sets = (o if i != n else o-s for i, (s, o) in enumerate(pairs))
                overlaps.append(ProductSet(*sets))
            return Union(*overlaps)

        elif isinstance(other, Interval):
            if isinstance(self, (Interval, FiniteSet)):
                return Intersection(other, self.complement(S.Reals))

        elif isinstance(other, Union):
            return Union(*(o - self for o in other.args))

        elif isinstance(other, Complement):
            return Complement(other.args[0], Union(other.args[1], self), evaluate=False)

        elif other is S.EmptySet:
            return S.EmptySet

        elif isinstance(other, FiniteSet):
            sifted = sift(other, lambda x: fuzzy_bool(self.contains(x)))
            # ignore those that are contained in self
            return Union(FiniteSet(*(sifted[False])),
                Complement(FiniteSet(*(sifted[None])), self, evaluate=False)
                if sifted[None] else S.EmptySet)

    def symmetric_difference(self, other):
        """
        Returns symmetric difference of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy import Interval, S
        >>> Interval(1, 3).symmetric_difference(S.Reals)
        Union(Interval.open(-oo, 1), Interval.open(3, oo))
        >>> Interval(1, 10).symmetric_difference(S.Reals)
        Union(Interval.open(-oo, 1), Interval.open(10, oo))

        >>> from sympy import S, EmptySet
        >>> S.Reals.symmetric_difference(EmptySet)
        Reals

        References
        ==========
        .. [1] https://en.wikipedia.org/wiki/Symmetric_difference

        """
        return SymmetricDifference(self, other)

    def _symmetric_difference(self, other):
        return Union(Complement(self, other), Complement(other, self))

    @property
    def inf(self):
        """
        The infimum of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).inf
        0
        >>> Union(Interval(0, 1), Interval(2, 3)).inf
        0

        """
        return self._inf

    @property
    def _inf(self):
        raise NotImplementedError("(%s)._inf" % self)

    @property
    def sup(self):
        """
        The supremum of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).sup
        1
        >>> Union(Interval(0, 1), Interval(2, 3)).sup
        3

        """
        return self._sup

    @property
    def _sup(self):
        raise NotImplementedError("(%s)._sup" % self)

    def contains(self, other):
        """
        Returns a SymPy value indicating whether ``other`` is contained
        in ``self``: ``true`` if it is, ``false`` if it is not, else
        an unevaluated ``Contains`` expression (or, as in the case of
        ConditionSet and a union of FiniteSet/Intervals, an expression
        indicating the conditions for containment).

        Examples
        ========

        >>> from sympy import Interval, S
        >>> from sympy.abc import x

        >>> Interval(0, 1).contains(0.5)
        True

        As a shortcut it is possible to use the ``in`` operator, but that
        will raise an error unless an affirmative true or false is not
        obtained.

        >>> Interval(0, 1).contains(x)
        (0 <= x) & (x <= 1)
        >>> x in Interval(0, 1)
        Traceback (most recent call last):
        ...
        TypeError: did not evaluate to a bool: None

        The result of 'in' is a bool, not a SymPy value

        >>> 1 in Interval(0, 2)
        True
        >>> _ is S.true
        False
        """
        from .contains import Contains
        other = sympify(other, strict=True)

        c = self._contains(other)
        if isinstance(c, Contains):
            return c
        if c is None:
            return Contains(other, self, evaluate=False)
        b = tfn[c]
        if b is None:
            return c
        return b

    def _contains(self, other):
        """Test if ``other`` is an element of the set ``self``.

        This is an internal method that is expected to be overridden by
        subclasses of ``Set`` and will be called by the public
        :func:`Set.contains` method or the :class:`Contains` expression.

        Parameters
        ==========

        other: Sympified :class:`Basic` instance
            The object whose membership in ``self`` is to be tested.

        Returns
        =======

        Symbolic :class:`Boolean` or ``None``.

        A return value of ``None`` indicates that it is unknown whether
        ``other`` is contained in ``self``. Returning ``None`` from here
        ensures that ``self.contains(other)`` or ``Contains(self, other)`` will
        return an unevaluated :class:`Contains` expression.

        If not ``None`` then the returned value is a :class:`Boolean` that is
        logically equivalent to the statement that ``other`` is an element of
        ``self``. Usually this would be either ``S.true`` or ``S.false`` but
        not always.
        """
        raise NotImplementedError(f"{type(self).__name__}._contains")

    def is_subset(self, other):
        """
        Returns True if ``self`` is a subset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_subset(Interval(0, 1, left_open=True))
        False

        """
        if not isinstance(other, Set):
            raise ValueError("Unknown argument '%s'" % other)

        # Handle the trivial cases
        if self == other:
            return True
        is_empty = self.is_empty
        if is_empty is True:
            return True
        elif fuzzy_not(is_empty) and other.is_empty:
            return False
        if self.is_finite_set is False and other.is_finite_set:
            return False

        # Dispatch on subclass rules
        ret = self._eval_is_subset(other)
        if ret is not None:
            return ret
        ret = other._eval_is_superset(self)
        if ret is not None:
            return ret

        # Use pairwise rules from multiple dispatch
        from sympy.sets.handlers.issubset import is_subset_sets
        ret = is_subset_sets(self, other)
        if ret is not None:
            return ret

        # Fall back on computing the intersection
        # XXX: We shouldn't do this. A query like this should be handled
        # without evaluating new Set objects. It should be the other way round
        # so that the intersect method uses is_subset for evaluation.
        if self.intersect(other) == self:
            return True

    def _eval_is_subset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    def _eval_is_superset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    # This should be deprecated:
    def issubset(self, other):
        """
        Alias for :meth:`is_subset()`
        """
        return self.is_subset(other)

    def is_proper_subset(self, other):
        """
        Returns True if ``self`` is a proper subset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_proper_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_proper_subset(Interval(0, 1))
        False

        """
        if isinstance(other, Set):
            return self != other and self.is_subset(other)
        else:
            raise ValueError("Unknown argument '%s'" % other)

    def is_superset(self, other):
        """
        Returns True if ``self`` is a superset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_superset(Interval(0, 1))
        False
        >>> Interval(0, 1).is_superset(Interval(0, 1, left_open=True))
        True

        """
        if isinstance(other, Set):
            return other.is_subset(self)
        else:
            raise ValueError("Unknown argument '%s'" % other)

    # This should be deprecated:
    def issuperset(self, other):
        """
        Alias for :meth:`is_superset()`
        """
        return self.is_superset(other)

    def is_proper_superset(self, other):
        """
        Returns True if ``self`` is a proper superset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).is_proper_superset(Interval(0, 0.5))
        True
        >>> Interval(0, 1).is_proper_superset(Interval(0, 1))
        False

        """
        if isinstance(other, Set):
            return self != other and self.is_superset(other)
        else:
            raise ValueError("Unknown argument '%s'" % other)

    def _eval_powerset(self):
        from .powerset import PowerSet
        return PowerSet(self)

    def powerset(self):
        """
        Find the Power set of ``self``.

        Examples
        ========

        >>> from sympy import EmptySet, FiniteSet, Interval

        A power set of an empty set:

        >>> A = EmptySet
        >>> A.powerset()
        {EmptySet}

        A power set of a finite set:

        >>> A = FiniteSet(1, 2)
        >>> a, b, c = FiniteSet(1), FiniteSet(2), FiniteSet(1, 2)
        >>> A.powerset() == FiniteSet(a, b, c, EmptySet)
        True

        A power set of an interval:

        >>> Interval(1, 2).powerset()
        PowerSet(Interval(1, 2))

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Power_set

        """
        return self._eval_powerset()

    @property
    def measure(self):
        """
        The (Lebesgue) measure of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).measure
        1
        >>> Union(Interval(0, 1), Interval(2, 3)).measure
        2

        """
        return self._measure

    @property
    def kind(self):
        """
        The kind of a Set

        Explanation
        ===========

        Any :class:`Set` will have kind :class:`SetKind` which is
        parametrised by the kind of the elements of the set. For example
        most sets are sets of numbers and will have kind
        ``SetKind(NumberKind)``. If elements of sets are different in kind than
        their kind will ``SetKind(UndefinedKind)``. See
        :class:`sympy.core.kind.Kind` for an explanation of the kind system.

        Examples
        ========

        >>> from sympy import Interval, Matrix, FiniteSet, EmptySet, ProductSet, PowerSet

        >>> FiniteSet(Matrix([1, 2])).kind
        SetKind(MatrixKind(NumberKind))

        >>> Interval(1, 2).kind
        SetKind(NumberKind)

        >>> EmptySet.kind
        SetKind()

        A :class:`sympy.sets.powerset.PowerSet` is a set of sets:

        >>> PowerSet({1, 2, 3}).kind
        SetKind(SetKind(NumberKind))

        A :class:`ProductSet` represents the set of tuples of elements of
        other sets. Its kind is :class:`sympy.core.containers.TupleKind`
        parametrised by the kinds of the elements of those sets:

        >>> p = ProductSet(FiniteSet(1, 2), FiniteSet(3, 4))
        >>> list(p)
        [(1, 3), (2, 3), (1, 4), (2, 4)]
        >>> p.kind
        SetKind(TupleKind(NumberKind, NumberKind))

        When all elements of the set do not have same kind, the kind
        will be returned as ``SetKind(UndefinedKind)``:

        >>> FiniteSet(0, Matrix([1, 2])).kind
        SetKind(UndefinedKind)

        The kind of the elements of a set are given by the ``element_kind``
        attribute of ``SetKind``:

        >>> Interval(1, 2).kind.element_kind
        NumberKind

        See Also
        ========

        NumberKind
        sympy.core.kind.UndefinedKind
        sympy.core.containers.TupleKind
        MatrixKind
        sympy.matrices.expressions.sets.MatrixSet
        sympy.sets.conditionset.ConditionSet
        Rationals
        Naturals
        Integers
        sympy.sets.fancysets.ImageSet
        sympy.sets.fancysets.Range
        sympy.sets.fancysets.ComplexRegion
        sympy.sets.powerset.PowerSet
        sympy.sets.sets.ProductSet
        sympy.sets.sets.Interval
        sympy.sets.sets.Union
        sympy.sets.sets.Intersection
        sympy.sets.sets.Complement
        sympy.sets.sets.EmptySet
        sympy.sets.sets.UniversalSet
        sympy.sets.sets.FiniteSet
        sympy.sets.sets.SymmetricDifference
        sympy.sets.sets.DisjointUnion
        """
        return self._kind()

    @property
    def boundary(self):
        """
        The boundary or frontier of a set.

        Explanation
        ===========

        A point x is on the boundary of a set S if

        1.  x is in the closure of S.
            I.e. Every neighborhood of x contains a point in S.
        2.  x is not in the interior of S.
            I.e. There does not exist an open set centered on x contained
            entirely within S.

        There are the points on the outer rim of S.  If S is open then these
        points need not actually be contained within S.

        For example, the boundary of an interval is its start and end points.
        This is true regardless of whether or not the interval is open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).boundary
        {0, 1}
        >>> Interval(0, 1, True, False).boundary
        {0, 1}
        """
        return self._boundary

    @property
    def is_open(self):
        """
        Property method to check whether a set is open.

        Explanation
        ===========

        A set is open if and only if it has an empty intersection with its
        boundary. In particular, a subset A of the reals is open if and only
        if each one of its points is contained in an open interval that is a
        subset of A.

        Examples
        ========
        >>> from sympy import S
        >>> S.Reals.is_open
        True
        >>> S.Rationals.is_open
        False
        """
        return Intersection(self, self.boundary).is_empty

    @property
    def is_closed(self):
        """
        A property method to check whether a set is closed.

        Explanation
        ===========

        A set is closed if its complement is an open set. The closedness of a
        subset of the reals is determined with respect to R and its standard
        topology.

        Examples
        ========
        >>> from sympy import Interval
        >>> Interval(0, 1).is_closed
        True
        """
        return self.boundary.is_subset(self)

    @property
    def closure(self):
        """
        Property method which returns the closure of a set.
        The closure is defined as the union of the set itself and its
        boundary.

        Examples
        ========
        >>> from sympy import S, Interval
        >>> S.Reals.closure
        Reals
        >>> Interval(0, 1).closure
        Interval(0, 1)
        """
        return self + self.boundary

    @property
    def interior(self):
        """
        Property method which returns the interior of a set.
        The interior of a set S consists all points of S that do not
        belong to the boundary of S.

        Examples
        ========
        >>> from sympy import Interval
        >>> Interval(0, 1).interior
        Interval.open(0, 1)
        >>> Interval(0, 1).boundary.interior
        EmptySet
        """
        return self - self.boundary

    @property
    def _boundary(self):
        raise NotImplementedError()

    @property
    def _measure(self):
        raise NotImplementedError("(%s)._measure" % self)

    def _kind(self):
        return SetKind(UndefinedKind)

    def _eval_evalf(self, prec):
        dps = prec_to_dps(prec)
        return self.func(*[arg.evalf(n=dps) for arg in self.args])

    @sympify_return([('other', 'Set')], NotImplemented)
    def __add__(self, other):
        return self.union(other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __or__(self, other):
        return self.union(other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __and__(self, other):
        return self.intersect(other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __mul__(self, other):
        return ProductSet(self, other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __xor__(self, other):
        return SymmetricDifference(self, other)

    @sympify_return([('exp', Expr)], NotImplemented)
    def __pow__(self, exp):
        if not (exp.is_Integer and exp >= 0):
            raise ValueError("%s: Exponent must be a positive Integer" % exp)
        return ProductSet(*[self]*exp)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __sub__(self, other):
        return Complement(self, other)

    def __contains__(self, other):
        other = _sympify(other)
        c = self._contains(other)
        b = tfn[c]
        if b is None:
            # x in y must evaluate to T or F; to entertain a None
            # result with Set use y.contains(x)
            raise TypeError('did not evaluate to a bool: %r' % c)
        return b


class ProductSet(Set):
    """
    Represents a Cartesian Product of Sets.

    Explanation
    ===========

    Returns a Cartesian product given several sets as either an iterable
    or individual arguments.

    Can use ``*`` operator on any sets for convenient shorthand.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet, ProductSet
    >>> I = Interval(0, 5); S = FiniteSet(1, 2, 3)
    >>> ProductSet(I, S)
    ProductSet(Interval(0, 5), {1, 2, 3})

    >>> (2, 2) in ProductSet(I, S)
    True

    >>> Interval(0, 1) * Interval(0, 1) # The unit square
    ProductSet(Interval(0, 1), Interval(0, 1))

    >>> coin = FiniteSet('H', 'T')
    >>> set(coin**2)
    {(H, H), (H, T), (T, H), (T, T)}

    The Cartesian product is not commutative or associative e.g.:

    >>> I*S == S*I
    False
    >>> (I*I)*I == I*(I*I)
    False

    Notes
    =====

    - Passes most operations down to the argument sets

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cartesian_product
    """
    is_ProductSet = True

    def __new__(cls, *sets, **assumptions):
        if len(sets) == 1 and iterable(sets[0]) and not isinstance(sets[0], (Set, set)):
            sympy_deprecation_warning(
                """
ProductSet(iterable) is deprecated. Use ProductSet(*iterable) instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-productset-iterable",
            )
            sets = tuple(sets[0])

        sets = [sympify(s) for s in sets]

        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Arguments to ProductSet should be of type Set")

        # Nullary product of sets is *not* the empty set
        if len(sets) == 0:
            return FiniteSet(())

        if S.EmptySet in sets:
            return S.EmptySet

        return Basic.__new__(cls, *sets, **assumptions)

    @property
    def sets(self):
        return self.args

    def flatten(self):
        def _flatten(sets):
            for s in sets:
                if s.is_ProductSet:
                    yield from _flatten(s.sets)
                else:
                    yield s
        return ProductSet(*_flatten(self.sets))



    def _contains(self, element):
        """
        ``in`` operator for ProductSets.

        Examples
        ========

        >>> from sympy import Interval
        >>> (2, 3) in Interval(0, 5) * Interval(0, 5)
        True

        >>> (10, 10) in Interval(0, 5) * Interval(0, 5)
        False

        Passes operation on to constituent sets
        """
        if element.is_Symbol:
            return None

        if not isinstance(element, Tuple) or len(element) != len(self.sets):
            return S.false

        return And(*[s.contains(e) for s, e in zip(self.sets, element)])

    def as_relational(self, *symbols):
        symbols = [_sympify(s) for s in symbols]
        if len(symbols) != len(self.sets) or not all(
                i.is_Symbol for i in symbols):
            raise ValueError(
                'number of symbols must match the number of sets')
        return And(*[s.as_relational(i) for s, i in zip(self.sets, symbols)])

    @property
    def _boundary(self):
        return Union(*(ProductSet(*(b + b.boundary if i != j else b.boundary
                                for j, b in enumerate(self.sets)))
                                for i, a in enumerate(self.sets)))

    @property
    def is_iterable(self):
        """
        A property method which tests whether a set is iterable or not.
        Returns True if set is iterable, otherwise returns False.

        Examples
        ========

        >>> from sympy import FiniteSet, Interval
        >>> I = Interval(0, 1)
        >>> A = FiniteSet(1, 2, 3, 4, 5)
        >>> I.is_iterable
        False
        >>> A.is_iterable
        True

        """
        return all(set.is_iterable for set in self.sets)

    def __iter__(self):
        """
        A method which implements is_iterable property method.
        If self.is_iterable returns True (both constituent sets are iterable),
        then return the Cartesian Product. Otherwise, raise TypeError.
        """
        return iproduct(*self.sets)

    @property
    def is_empty(self):
        return fuzzy_or(s.is_empty for s in self.sets)

    @property
    def is_finite_set(self):
        all_finite = fuzzy_and(s.is_finite_set for s in self.sets)
        return fuzzy_or([self.is_empty, all_finite])

    @property
    def _measure(self):
        measure = 1
        for s in self.sets:
            measure *= s.measure
        return measure

    def _kind(self):
        return SetKind(TupleKind(*(i.kind.element_kind for i in self.args)))

    def __len__(self):
        return reduce(lambda a, b: a*b, (len(s) for s in self.args))

    def __bool__(self):
        return all(self.sets)


class Interval(Set):
    """
    Represents a real interval as a Set.

    Usage:
        Returns an interval with end points ``start`` and ``end``.

        For ``left_open=True`` (default ``left_open`` is ``False``) the interval
        will be open on the left. Similarly, for ``right_open=True`` the interval
        will be open on the right.

    Examples
    ========

    >>> from sympy import Symbol, Interval
    >>> Interval(0, 1)
    Interval(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Lopen(0, 1)
    Interval.Lopen(0, 1)
    >>> Interval.open(0, 1)
    Interval.open(0, 1)

    >>> a = Symbol('a', real=True)
    >>> Interval(0, a)
    Interval(0, a)

    Notes
    =====
    - Only real end points are supported
    - ``Interval(a, b)`` with $a > b$ will return the empty set
    - Use the ``evalf()`` method to turn an Interval into an mpmath
      ``mpi`` interval instance

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_%28mathematics%29
    """
    is_Interval = True

    def __new__(cls, start, end, left_open=False, right_open=False):

        start = _sympify(start)
        end = _sympify(end)
        left_open = _sympify(left_open)
        right_open = _sympify(right_open)

        if not all(isinstance(a, (type(true), type(false)))
            for a in [left_open, right_open]):
            raise NotImplementedError(
                "left_open and right_open can have only true/false values, "
                "got %s and %s" % (left_open, right_open))

        # Only allow real intervals
        if fuzzy_not(fuzzy_and(i.is_extended_real for i in (start, end, end-start))):
            raise ValueError("Non-real intervals are not supported")

        # evaluate if possible
        if is_lt(end, start):
            return S.EmptySet
        elif (end - start).is_negative:
            return S.EmptySet

        if end == start and (left_open or right_open):
            return S.EmptySet
        if end == start and not (left_open or right_open):
            if start is S.Infinity or start is S.NegativeInfinity:
                return S.EmptySet
            return FiniteSet(end)

        # Make sure infinite interval end points are open.
        if start is S.NegativeInfinity:
            left_open = true
        if end is S.Infinity:
            right_open = true
        if start == S.Infinity or end == S.NegativeInfinity:
            return S.EmptySet

        return Basic.__new__(cls, start, end, left_open, right_open)

    @property
    def start(self):
        """
        The left end point of the interval.

        This property takes the same value as the ``inf`` property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).start
        0

        """
        return self._args[0]

    @property
    def end(self):
        """
        The right end point of the interval.

        This property takes the same value as the ``sup`` property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).end
        1

        """
        return self._args[1]

    @property
    def left_open(self):
        """
        True if interval is left-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, left_open=True).left_open
        True
        >>> Interval(0, 1, left_open=False).left_open
        False

        """
        return self._args[2]

    @property
    def right_open(self):
        """
        True if interval is right-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, right_open=True).right_open
        True
        >>> Interval(0, 1, right_open=False).right_open
        False

        """
        return self._args[3]

    @classmethod
    def open(cls, a, b):
        """Return an interval including neither boundary."""
        return cls(a, b, True, True)

    @classmethod
    def Lopen(cls, a, b):
        """Return an interval not including the left boundary."""
        return cls(a, b, True, False)

    @classmethod
    def Ropen(cls, a, b):
        """Return an interval not including the right boundary."""
        return cls(a, b, False, True)

    @property
    def _inf(self):
        return self.start

    @property
    def _sup(self):
        return self.end

    @property
    def left(self):
        return self.start

    @property
    def right(self):
        return self.end

    @property
    def is_empty(self):
        if self.left_open or self.right_open:
            cond = self.start >= self.end  # One/both bounds open
        else:
            cond = self.start > self.end  # Both bounds closed
        return fuzzy_bool(cond)

    @property
    def is_finite_set(self):
        return self.measure.is_zero

    def _complement(self, other):
        if other == S.Reals:
            a = Interval(S.NegativeInfinity, self.start,
                         True, not self.left_open)
            b = Interval(self.end, S.Infinity, not self.right_open, True)
            return Union(a, b)

        if isinstance(other, FiniteSet):
            nums = [m for m in other.args if m.is_number]
            if nums == []:
                return None

        return Set._complement(self, other)

    @property
    def _boundary(self):
        finite_points = [p for p in (self.start, self.end)
                         if abs(p) != S.Infinity]
        return FiniteSet(*finite_points)

    def _contains(self, other):
        if (not isinstance(other, Expr) or other is S.NaN
            or other.is_real is False or other.has(S.ComplexInfinity)):
                # if an expression has zoo it will be zoo or nan
                # and neither of those is real
                return false

        if self.start is S.NegativeInfinity and self.end is S.Infinity:
            if other.is_real is not None:
                return tfn[other.is_real]

        d = Dummy()
        return self.as_relational(d).subs(d, other)

    def as_relational(self, x):
        """Rewrite an interval in terms of inequalities and logic operators."""
        x = sympify(x)
        if self.right_open:
            right = x < self.end
        else:
            right = x <= self.end
        if self.left_open:
            left = self.start < x
        else:
            left = self.start <= x
        return And(left, right)

    @property
    def _measure(self):
        return self.end - self.start

    def _kind(self):
        return SetKind(NumberKind)

    def to_mpi(self, prec=53):
        return mpi(mpf(self.start._eval_evalf(prec)),
            mpf(self.end._eval_evalf(prec)))

    def _eval_evalf(self, prec):
        return Interval(self.left._evalf(prec), self.right._evalf(prec),
            left_open=self.left_open, right_open=self.right_open)

    def _is_comparable(self, other):
        is_comparable = self.start.is_comparable
        is_comparable &= self.end.is_comparable
        is_comparable &= other.start.is_comparable
        is_comparable &= other.end.is_comparable

        return is_comparable

    @property
    def is_left_unbounded(self):
        """Return ``True`` if the left endpoint is negative infinity. """
        return self.left is S.NegativeInfinity or self.left == Float("-inf")

    @property
    def is_right_unbounded(self):
        """Return ``True`` if the right endpoint is positive infinity. """
        return self.right is S.Infinity or self.right == Float("+inf")

    def _eval_Eq(self, other):
        if not isinstance(other, Interval):
            if isinstance(other, FiniteSet):
                return false
            elif isinstance(other, Set):
                return None
            return false


class Union(Set, LatticeOp):
    """
    Represents a union of sets as a :class:`Set`.

    Examples
    ========

    >>> from sympy import Union, Interval
    >>> Union(Interval(1, 2), Interval(3, 4))
    Union(Interval(1, 2), Interval(3, 4))

    The Union constructor will always try to merge overlapping intervals,
    if possible. For example:

    >>> Union(Interval(1, 2), Interval(2, 3))
    Interval(1, 3)

    See Also
    ========

    Intersection

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Union_%28set_theory%29
    """
    is_Union = True

    @property
    def identity(self):
        return S.EmptySet

    @property
    def zero(self):
        return S.UniversalSet

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # flatten inputs to merge intersections and iterables
        args = _sympify(args)

        # Reduce sets using known rules
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_union(args)

        args = list(ordered(args, Set._infimum_key))

        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj

    @property
    def args(self):
        return self._args

    def _complement(self, universe):
        # DeMorgan's Law
        return Intersection(s.complement(universe) for s in self.args)

    @property
    def _inf(self):
        # We use Min so that sup is meaningful in combination with symbolic
        # interval end points.
        return Min(*[set.inf for set in self.args])

    @property
    def _sup(self):
        # We use Max so that sup is meaningful in combination with symbolic
        # end points.
        return Max(*[set.sup for set in self.args])

    @property
    def is_empty(self):
        return fuzzy_and(set.is_empty for set in self.args)

    @property
    def is_finite_set(self):
        return fuzzy_and(set.is_finite_set for set in self.args)

    @property
    def _measure(self):
        # Measure of a union is the sum of the measures of the sets minus
        # the sum of their pairwise intersections plus the sum of their
        # triple-wise intersections minus ... etc...

        # Sets is a collection of intersections and a set of elementary
        # sets which made up those intersections (called "sos" for set of sets)
        # An example element might of this list might be:
        #    ( {A,B,C}, A.intersect(B).intersect(C) )

        # Start with just elementary sets (  ({A}, A), ({B}, B), ... )
        # Then get and subtract (  ({A,B}, (A int B), ... ) while non-zero
        sets = [(FiniteSet(s), s) for s in self.args]
        measure = 0
        parity = 1
        while sets:
            # Add up the measure of these sets and add or subtract it to total
            measure += parity * sum(inter.measure for sos, inter in sets)

            # For each intersection in sets, compute the intersection with every
            # other set not already part of the intersection.
            sets = ((sos + FiniteSet(newset), newset.intersect(intersection))
                    for sos, intersection in sets for newset in self.args
                    if newset not in sos)

            # Clear out sets with no measure
            sets = [(sos, inter) for sos, inter in sets if inter.measure != 0]

            # Clear out duplicates
            sos_list = []
            sets_list = []
            for _set in sets:
                if _set[0] in sos_list:
                    continue
                else:
                    sos_list.append(_set[0])
                    sets_list.append(_set)
            sets = sets_list

            # Flip Parity - next time subtract/add if we added/subtracted here
            parity *= -1
        return measure

    def _kind(self):
        kinds = tuple(arg.kind for arg in self.args if arg is not S.EmptySet)
        if not kinds:
            return SetKind()
        elif all(i == kinds[0] for i in kinds):
            return kinds[0]
        else:
            return SetKind(UndefinedKind)

    @property
    def _boundary(self):
        def boundary_of_set(i):
            """ The boundary of set i minus interior of all other sets """
            b = self.args[i].boundary
            for j, a in enumerate(self.args):
                if j != i:
                    b = b - a.interior
            return b
        return Union(*map(boundary_of_set, range(len(self.args))))

    def _contains(self, other):
        return Or(*[s.contains(other) for s in self.args])

    def is_subset(self, other):
        return fuzzy_and(s.is_subset(other) for s in self.args)

    def as_relational(self, symbol):
        """Rewrite a Union in terms of equalities and logic operators. """
        if (len(self.args) == 2 and
                all(isinstance(i, Interval) for i in self.args)):
            # optimization to give 3 args as (x > 1) & (x < 5) & Ne(x, 3)
            # instead of as 4, ((1 <= x) & (x < 3)) | ((x <= 5) & (3 < x))
            # XXX: This should be ideally be improved to handle any number of
            # intervals and also not to assume that the intervals are in any
            # particular sorted order.
            a, b = self.args
            if a.sup == b.inf and a.right_open and b.left_open:
                mincond = symbol > a.inf if a.left_open else symbol >= a.inf
                maxcond = symbol < b.sup if b.right_open else symbol <= b.sup
                necond = Ne(symbol, a.sup)
                return And(necond, mincond, maxcond)
        return Or(*[i.as_relational(symbol) for i in self.args])

    @property
    def is_iterable(self):
        return all(arg.is_iterable for arg in self.args)

    def __iter__(self):
        return roundrobin(*(iter(arg) for arg in self.args))


class Intersection(Set, LatticeOp):
    """
    Represents an intersection of sets as a :class:`Set`.

    Examples
    ========

    >>> from sympy import Intersection, Interval
    >>> Intersection(Interval(1, 3), Interval(2, 4))
    Interval(2, 3)

    We often use the .intersect method

    >>> Interval(1,3).intersect(Interval(2,4))
    Interval(2, 3)

    See Also
    ========

    Union

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Intersection_%28set_theory%29
    """
    is_Intersection = True

    @property
    def identity(self):
        return S.UniversalSet

    @property
    def zero(self):
        return S.EmptySet

    def __new__(cls, *args , evaluate=None):
        if evaluate is None:
            evaluate = global_parameters.evaluate

        # flatten inputs to merge intersections and iterables
        args = list(ordered(set(_sympify(args))))

        # Reduce sets using known rules
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_intersection(args)

        args = list(ordered(args, Set._infimum_key))

        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj

    @property
    def args(self):
        return self._args

    @property
    def is_iterable(self):
        return any(arg.is_iterable for arg in self.args)

    @property
    def is_finite_set(self):
        if fuzzy_or(arg.is_finite_set for arg in self.args):
            return True

    def _kind(self):
        kinds = tuple(arg.kind for arg in self.args if arg is not S.UniversalSet)
        if not kinds:
            return SetKind(UndefinedKind)
        elif all(i == kinds[0] for i in kinds):
            return kinds[0]
        else:
            return SetKind()

    @property
    def _inf(self):
        raise NotImplementedError()

    @property
    def _sup(self):
        raise NotImplementedError()

    def _contains(self, other):
        return And(*[set.contains(other) for set in self.args])

    def __iter__(self):
        sets_sift = sift(self.args, lambda x: x.is_iterable)

        completed = False
        candidates = sets_sift[True] + sets_sift[None]

        finite_candidates, others = [], []
        for candidate in candidates:
            length = None
            try:
                length = len(candidate)
            except TypeError:
                others.append(candidate)

            if length is not None:
                finite_candidates.append(candidate)
        finite_candidates.sort(key=len)

        for s in finite_candidates + others:
            other_sets = set(self.args) - {s}
            other = Intersection(*other_sets, evaluate=False)
            completed = True
            for x in s:
                try:
                    if x in other:
                        yield x
                except TypeError:
                    completed = False
            if completed:
                return

        if not completed:
            if not candidates:
                raise TypeError("None of the constituent sets are iterable")
            raise TypeError(
                "The computation had not completed because of the "
                "undecidable set membership is found in every candidates.")

    @staticmethod
    def _handle_finite_sets(args):
        '''Simplify intersection of one or more FiniteSets and other sets'''

        # First separate the FiniteSets from the others
        fs_args, others = sift(args, lambda x: x.is_FiniteSet, binary=True)

        # Let the caller handle intersection of non-FiniteSets
        if not fs_args:
            return

        # Convert to Python sets and build the set of all elements
        fs_sets = [set(fs) for fs in fs_args]
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Extract elements that are definitely in or definitely not in the
        # intersection. Here we check contains for all of args.
        definite = set()
        for e in all_elements:
            inall = fuzzy_and(s.contains(e) for s in args)
            if inall is True:
                definite.add(e)
            if inall is not None:
                for s in fs_sets:
                    s.discard(e)

        # At this point all elements in all of fs_sets are possibly in the
        # intersection. In some cases this is because they are definitely in
        # the intersection of the finite sets but it's not clear if they are
        # members of others. We might have {m, n}, {m}, and Reals where we
        # don't know if m or n is real. We want to remove n here but it is
        # possibly in because it might be equal to m. So what we do now is
        # extract the elements that are definitely in the remaining finite
        # sets iteratively until we end up with {n}, {}. At that point if we
        # get any empty set all remaining elements are discarded.

        fs_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Need fuzzy containment testing
        fs_symsets = [FiniteSet(*s) for s in fs_sets]

        while fs_elements:
            for e in fs_elements:
                infs = fuzzy_and(s.contains(e) for s in fs_symsets)
                if infs is True:
                    definite.add(e)
                if infs is not None:
                    for n, s in enumerate(fs_sets):
                        # Update Python set and FiniteSet
                        if e in s:
                            s.remove(e)
                            fs_symsets[n] = FiniteSet(*s)
                    fs_elements.remove(e)
                    break
            # If we completed the for loop without removing anything we are
            # done so quit the outer while loop
            else:
                break

        # If any of the sets of remainder elements is empty then we discard
        # all of them for the intersection.
        if not all(fs_sets):
            fs_sets = [set()]

        # Here we fold back the definitely included elements into each fs.
        # Since they are definitely included they must have been members of
        # each FiniteSet to begin with. We could instead fold these in with a
        # Union at the end to get e.g. {3}|({x}&{y}) rather than {3,x}&{3,y}.
        if definite:
            fs_sets = [fs | definite for fs in fs_sets]

        if fs_sets == [set()]:
            return S.EmptySet

        sets = [FiniteSet(*s) for s in fs_sets]

        # Any set in others is redundant if it contains all the elements that
        # are in the finite sets so we don't need it in the Intersection
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())
        is_redundant = lambda o: all(fuzzy_bool(o.contains(e)) for e in all_elements)
        others = [o for o in others if not is_redundant(o)]

        if others:
            rest = Intersection(*others)
            # XXX: Maybe this shortcut should be at the beginning. For large
            # FiniteSets it could much more efficient to process the other
            # sets first...
            if rest is S.EmptySet:
                return S.EmptySet
            # Flatten the Intersection
            if rest.is_Intersection:
                sets.extend(rest.args)
            else:
                sets.append(rest)

        if len(sets) == 1:
            return sets[0]
        else:
            return Intersection(*sets, evaluate=False)

    def as_relational(self, symbol):
        """Rewrite an Intersection in terms of equalities and logic operators"""
        return And(*[set.as_relational(symbol) for set in self.args])


class Complement(Set):
    r"""Represents the set difference or relative complement of a set with
    another set.

    $$A - B = \{x \in A \mid x \notin B\}$$


    Examples
    ========

    >>> from sympy import Complement, FiniteSet
    >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
    {0, 2}

    See Also
    =========

    Intersection, Union

    References
    ==========

    .. [1] https://mathworld.wolfram.com/ComplementSet.html
    """

    is_Complement = True

    def __new__(cls, a, b, evaluate=True):
        a, b = map(_sympify, (a, b))
        if evaluate:
            return Complement.reduce(a, b)

        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        """
        Simplify a :class:`Complement`.

        """
        if B == S.UniversalSet or A.is_subset(B):
            return S.EmptySet

        if isinstance(B, Union):
            return Intersection(*(s.complement(A) for s in B.args))

        result = B._complement(A)
        if result is not None:
            return result
        else:
            return Complement(A, B, evaluate=False)

    def _contains(self, other):
        A = self.args[0]
        B = self.args[1]
        return And(A.contains(other), Not(B.contains(other)))

    def as_relational(self, symbol):
        """Rewrite a complement in terms of equalities and logic
        operators"""
        A, B = self.args

        A_rel = A.as_relational(symbol)
        B_rel = Not(B.as_relational(symbol))

        return And(A_rel, B_rel)

    def _kind(self):
        return self.args[0].kind

    @property
    def is_iterable(self):
        if self.args[0].is_iterable:
            return True

    @property
    def is_finite_set(self):
        A, B = self.args
        a_finite = A.is_finite_set
        if a_finite is True:
            return True
        elif a_finite is False and B.is_finite_set:
            return False

    def __iter__(self):
        A, B = self.args
        for a in A:
            if a not in B:
                    yield a
            else:
                continue


class EmptySet(Set, metaclass=Singleton):
    """
    Represents the empty set. The empty set is available as a singleton
    as ``S.EmptySet``.

    Examples
    ========

    >>> from sympy import S, Interval
    >>> S.EmptySet
    EmptySet

    >>> Interval(1, 2).intersect(S.EmptySet)
    EmptySet

    See Also
    ========

    UniversalSet

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Empty_set
    """
    is_empty = True
    is_finite_set = True
    is_FiniteSet = True

    @property  # type: ignore
    @deprecated(
        """
        The is_EmptySet attribute of Set objects is deprecated.
        Use 's is S.EmptySet" or 's.is_empty' instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-is-emptyset",
    )
    def is_EmptySet(self):
        return True

    @property
    def _measure(self):
        return 0

    def _contains(self, other):
        return false

    def as_relational(self, symbol):
        return false

    def __len__(self):
        return 0

    def __iter__(self):
        return iter([])

    def _eval_powerset(self):
        return FiniteSet(self)

    @property
    def _boundary(self):
        return self

    def _complement(self, other):
        return other

    def _kind(self):
        return SetKind()

    def _symmetric_difference(self, other):
        return other


class UniversalSet(Set, metaclass=Singleton):
    """
    Represents the set of all things.
    The universal set is available as a singleton as ``S.UniversalSet``.

    Examples
    ========

    >>> from sympy import S, Interval
    >>> S.UniversalSet
    UniversalSet

    >>> Interval(1, 2).intersect(S.UniversalSet)
    Interval(1, 2)

    See Also
    ========

    EmptySet

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Universal_set
    """

    is_UniversalSet = True
    is_empty = False
    is_finite_set = False

    def _complement(self, other):
        return S.EmptySet

    def _symmetric_difference(self, other):
        return other

    @property
    def _measure(self):
        return S.Infinity

    def _kind(self):
        return SetKind(UndefinedKind)

    def _contains(self, other):
        return true

    def as_relational(self, symbol):
        return true

    @property
    def _boundary(self):
        return S.EmptySet


class FiniteSet(Set):
    """
    Represents a finite set of Sympy expressions.

    Examples
    ========

    >>> from sympy import FiniteSet, Symbol, Interval, Naturals0
    >>> FiniteSet(1, 2, 3, 4)
    {1, 2, 3, 4}
    >>> 3 in FiniteSet(1, 2, 3, 4)
    True
    >>> FiniteSet(1, (1, 2), Symbol('x'))
    {1, x, (1, 2)}
    >>> FiniteSet(Interval(1, 2), Naturals0, {1, 2})
    FiniteSet({1, 2}, Interval(1, 2), Naturals0)
    >>> members = [1, 2, 3, 4]
    >>> f = FiniteSet(*members)
    >>> f
    {1, 2, 3, 4}
    >>> f - FiniteSet(2)
    {1, 3, 4}
    >>> f + FiniteSet(2, 5)
    {1, 2, 3, 4, 5}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Finite_set
    """
    is_FiniteSet = True
    is_iterable = True
    is_empty = False
    is_finite_set = True

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        if evaluate:
            args = list(map(sympify, args))

            if len(args) == 0:
                return S.EmptySet
        else:
            args = list(map(sympify, args))

        # keep the form of the first canonical arg
        dargs = {}
        for i in reversed(list(ordered(args))):
            if i.is_Symbol:
                dargs[i] = i
            else:
                try:
                    dargs[i.as_dummy()] = i
                except TypeError:
                    # e.g. i = class without args like `Interval`
                    dargs[i] = i
        _args_set = set(dargs.values())
        args = list(ordered(_args_set, Set._infimum_key))
        obj = Basic.__new__(cls, *args)
        obj._args_set = _args_set
        return obj


    def __iter__(self):
        return iter(self.args)

    def _complement(self, other):
        if isinstance(other, Interval):
            # Splitting in sub-intervals is only done for S.Reals;
            # other cases that need splitting will first pass through
            # Set._complement().
            nums, syms = [], []
            for m in self.args:
                if m.is_number and m.is_real:
                    nums.append(m)
                elif m.is_real == False:
                    pass  # drop non-reals
                else:
                    syms.append(m)  # various symbolic expressions
            if other == S.Reals and nums != []:
                nums.sort()
                intervals = []  # Build up a list of intervals between the elements
                intervals += [Interval(S.NegativeInfinity, nums[0], True, True)]
                for a, b in zip(nums[:-1], nums[1:]):
                    intervals.append(Interval(a, b, True, True))  # both open
                intervals.append(Interval(nums[-1], S.Infinity, True, True))
                if syms != []:
                    return Complement(Union(*intervals, evaluate=False),
                            FiniteSet(*syms), evaluate=False)
                else:
                    return Union(*intervals, evaluate=False)
            elif nums == []:  # no splitting necessary or possible:
                if syms:
                    return Complement(other, FiniteSet(*syms), evaluate=False)
                else:
                    return other

        elif isinstance(other, FiniteSet):
            unk = []
            for i in self:
                c = sympify(other.contains(i))
                if c is not S.true and c is not S.false:
                    unk.append(i)
            unk = FiniteSet(*unk)
            if unk == self:
                return
            not_true = []
            for i in other:
                c = sympify(self.contains(i))
                if c is not S.true:
                    not_true.append(i)
            return Complement(FiniteSet(*not_true), unk)

        return Set._complement(self, other)

    def _contains(self, other):
        """
        Tests whether an element, other, is in the set.

        Explanation
        ===========

        The actual test is for mathematical equality (as opposed to
        syntactical equality). In the worst case all elements of the
        set must be checked.

        Examples
        ========

        >>> from sympy import FiniteSet
        >>> 1 in FiniteSet(1, 2)
        True
        >>> 5 in FiniteSet(1, 2)
        False

        """
        if other in self._args_set:
            return S.true
        else:
            # evaluate=True is needed to override evaluate=False context;
            # we need Eq to do the evaluation
            return Or(*[Eq(e, other, evaluate=True) for e in self.args])

    def _eval_is_subset(self, other):
        return fuzzy_and(other._contains(e) for e in self.args)

    @property
    def _boundary(self):
        return self

    @property
    def _inf(self):
        return Min(*self)

    @property
    def _sup(self):
        return Max(*self)

    @property
    def measure(self):
        return 0

    def _kind(self):
        if not self.args:
            return SetKind()
        elif all(i.kind == self.args[0].kind for i in self.args):
            return SetKind(self.args[0].kind)
        else:
            return SetKind(UndefinedKind)

    def __len__(self):
        return len(self.args)

    def as_relational(self, symbol):
        """Rewrite a FiniteSet in terms of equalities and logic operators. """
        return Or(*[Eq(symbol, elem) for elem in self])

    def compare(self, other):
        return (hash(self) - hash(other))

    def _eval_evalf(self, prec):
        dps = prec_to_dps(prec)
        return FiniteSet(*[elem.evalf(n=dps) for elem in self])

    def _eval_simplify(self, **kwargs):
        from sympy.simplify import simplify
        return FiniteSet(*[simplify(elem, **kwargs) for elem in self])

    @property
    def _sorted_args(self):
        return self.args

    def _eval_powerset(self):
        return self.func(*[self.func(*s) for s in subsets(self.args)])

    def _eval_rewrite_as_PowerSet(self, *args, **kwargs):
        """Rewriting method for a finite set to a power set."""
        from .powerset import PowerSet

        is2pow = lambda n: bool(n and not n & (n - 1))
        if not is2pow(len(self)):
            return None

        fs_test = lambda arg: isinstance(arg, Set) and arg.is_FiniteSet
        if not all(fs_test(arg) for arg in args):
            return None

        biggest = max(args, key=len)
        for arg in subsets(biggest.args):
            arg_set = FiniteSet(*arg)
            if arg_set not in args:
                return None
        return PowerSet(biggest)

    def __ge__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return other.is_subset(self)

    def __gt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_superset(other)

    def __le__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_subset(other)

    def __lt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_subset(other)

    def __eq__(self, other):
        if isinstance(other, (set, frozenset)):
            return self._args_set == other
        return super().__eq__(other)

    __hash__ : Callable[[Basic], Any] = Basic.__hash__

_sympy_converter[set] = lambda x: FiniteSet(*x)
_sympy_converter[frozenset] = lambda x: FiniteSet(*x)


class SymmetricDifference(Set):
    """Represents the set of elements which are in either of the
    sets and not in their intersection.

    Examples
    ========

    >>> from sympy import SymmetricDifference, FiniteSet
    >>> SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(3, 4, 5))
    {1, 2, 4, 5}

    See Also
    ========

    Complement, Union

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_difference
    """

    is_SymmetricDifference = True

    def __new__(cls, a, b, evaluate=True):
        if evaluate:
            return SymmetricDifference.reduce(a, b)

        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        result = B._symmetric_difference(A)
        if result is not None:
            return result
        else:
            return SymmetricDifference(A, B, evaluate=False)

    def as_relational(self, symbol):
        """Rewrite a symmetric_difference in terms of equalities and
        logic operators"""
        A, B = self.args

        A_rel = A.as_relational(symbol)
        B_rel = B.as_relational(symbol)

        return Xor(A_rel, B_rel)

    @property
    def is_iterable(self):
        if all(arg.is_iterable for arg in self.args):
            return True

    def __iter__(self):

        args = self.args
        union = roundrobin(*(iter(arg) for arg in args))

        for item in union:
            count = 0
            for s in args:
                if item in s:
                    count += 1

            if count % 2 == 1:
                yield item



class DisjointUnion(Set):
    """ Represents the disjoint union (also known as the external disjoint union)
    of a finite number of sets.

    Examples
    ========

    >>> from sympy import DisjointUnion, FiniteSet, Interval, Union, Symbol
    >>> A = FiniteSet(1, 2, 3)
    >>> B = Interval(0, 5)
    >>> DisjointUnion(A, B)
    DisjointUnion({1, 2, 3}, Interval(0, 5))
    >>> DisjointUnion(A, B).rewrite(Union)
    Union(ProductSet({1, 2, 3}, {0}), ProductSet(Interval(0, 5), {1}))
    >>> C = FiniteSet(Symbol('x'), Symbol('y'), Symbol('z'))
    >>> DisjointUnion(C, C)
    DisjointUnion({x, y, z}, {x, y, z})
    >>> DisjointUnion(C, C).rewrite(Union)
    ProductSet({x, y, z}, {0, 1})

    References
    ==========

    https://en.wikipedia.org/wiki/Disjoint_union
    """

    def __new__(cls, *sets):
        dj_collection = []
        for set_i in sets:
            if isinstance(set_i, Set):
                dj_collection.append(set_i)
            else:
                raise TypeError("Invalid input: '%s', input args \
                    to DisjointUnion must be Sets" % set_i)
        obj = Basic.__new__(cls, *dj_collection)
        return obj

    @property
    def sets(self):
        return self.args

    @property
    def is_empty(self):
        return fuzzy_and(s.is_empty for s in self.sets)

    @property
    def is_finite_set(self):
        all_finite = fuzzy_and(s.is_finite_set for s in self.sets)
        return fuzzy_or([self.is_empty, all_finite])

    @property
    def is_iterable(self):
        if self.is_empty:
            return False
        iter_flag = True
        for set_i in self.sets:
            if not set_i.is_empty:
                iter_flag = iter_flag and set_i.is_iterable
        return iter_flag

    def _eval_rewrite_as_Union(self, *sets, **kwargs):
        """
        Rewrites the disjoint union as the union of (``set`` x {``i``})
        where ``set`` is the element in ``sets`` at index = ``i``
        """

        dj_union = S.EmptySet
        index = 0
        for set_i in sets:
            if isinstance(set_i, Set):
                cross = ProductSet(set_i, FiniteSet(index))
                dj_union = Union(dj_union, cross)
                index = index + 1
        return dj_union

    def _contains(self, element):
        """
        ``in`` operator for DisjointUnion

        Examples
        ========

        >>> from sympy import Interval, DisjointUnion
        >>> D = DisjointUnion(Interval(0, 1), Interval(0, 2))
        >>> (0.5, 0) in D
        True
        >>> (0.5, 1) in D
        True
        >>> (1.5, 0) in D
        False
        >>> (1.5, 1) in D
        True

        Passes operation on to constituent sets
        """
        if not isinstance(element, Tuple) or len(element) != 2:
            return S.false

        if not element[1].is_Integer:
            return S.false

        if element[1] >= len(self.sets) or element[1] < 0:
            return S.false

        return self.sets[element[1]]._contains(element[0])

    def _kind(self):
        if not self.args:
            return SetKind()
        elif all(i.kind == self.args[0].kind for i in self.args):
            return self.args[0].kind
        else:
            return SetKind(UndefinedKind)

    def __iter__(self):
        if self.is_iterable:

            iters = []
            for i, s in enumerate(self.sets):
                iters.append(iproduct(s, {Integer(i)}))

            return iter(roundrobin(*iters))
        else:
            raise ValueError("'%s' is not iterable." % self)

    def __len__(self):
        """
        Returns the length of the disjoint union, i.e., the number of elements in the set.

        Examples
        ========

        >>> from sympy import FiniteSet, DisjointUnion, EmptySet
        >>> D1 = DisjointUnion(FiniteSet(1, 2, 3, 4), EmptySet, FiniteSet(3, 4, 5))
        >>> len(D1)
        7
        >>> D2 = DisjointUnion(FiniteSet(3, 5, 7), EmptySet, FiniteSet(3, 5, 7))
        >>> len(D2)
        6
        >>> D3 = DisjointUnion(EmptySet, EmptySet)
        >>> len(D3)
        0

        Adds up the lengths of the constituent sets.
        """

        if self.is_finite_set:
            size = 0
            for set in self.sets:
                size += len(set)
            return size
        else:
            raise ValueError("'%s' is not a finite set." % self)


def imageset(*args):
    r"""
    Return an image of the set under transformation ``f``.

    Explanation
    ===========

    If this function cannot compute the image, it returns an
    unevaluated ImageSet object.

    .. math::
        \{ f(x) \mid x \in \mathrm{self} \}

    Examples
    ========

    >>> from sympy import S, Interval, imageset, sin, Lambda
    >>> from sympy.abc import x

    >>> imageset(x, 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(lambda x: 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(Lambda(x, sin(x)), Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))

    >>> imageset(sin, Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))
    >>> imageset(lambda y: x + y, Interval(-2, 1))
    ImageSet(Lambda(y, x + y), Interval(-2, 1))

    Expressions applied to the set of Integers are simplified
    to show as few negatives as possible and linear expressions
    are converted to a canonical form. If this is not desirable
    then the unevaluated ImageSet should be used.

    >>> imageset(x, -2*x + 5, S.Integers)
    ImageSet(Lambda(x, 2*x + 1), Integers)

    See Also
    ========

    sympy.sets.fancysets.ImageSet

    """
    from .fancysets import ImageSet
    from .setexpr import set_function

    if len(args) < 2:
        raise ValueError('imageset expects at least 2 args, got: %s' % len(args))

    if isinstance(args[0], (Symbol, tuple)) and len(args) > 2:
        f = Lambda(args[0], args[1])
        set_list = args[2:]
    else:
        f = args[0]
        set_list = args[1:]

    if isinstance(f, Lambda):
        pass
    elif callable(f):
        nargs = getattr(f, 'nargs', {})
        if nargs:
            if len(nargs) != 1:
                raise NotImplementedError(filldedent('''
                    This function can take more than 1 arg
                    but the potentially complicated set input
                    has not been analyzed at this point to
                    know its dimensions. TODO
                    '''))
            N = nargs.args[0]
            if N == 1:
                s = 'x'
            else:
                s = [Symbol('x%i' % i) for i in range(1, N + 1)]
        else:
            s = inspect.signature(f).parameters

        dexpr = _sympify(f(*[Dummy() for i in s]))
        var = tuple(uniquely_named_symbol(
            Symbol(i), dexpr) for i in s)
        f = Lambda(var, f(*var))
    else:
        raise TypeError(filldedent('''
            expecting lambda, Lambda, or FunctionClass,
            not \'%s\'.''' % func_name(f)))

    if any(not isinstance(s, Set) for s in set_list):
        name = [func_name(s) for s in set_list]
        raise ValueError(
            'arguments after mapping should be sets, not %s' % name)

    if len(set_list) == 1:
        set = set_list[0]
        try:
            # TypeError if arg count != set dimensions
            r = set_function(f, set)
            if r is None:
                raise TypeError
            if not r:
                return r
        except TypeError:
            r = ImageSet(f, set)
        if isinstance(r, ImageSet):
            f, set = r.args

        if f.variables[0] == f.expr:
            return set

        if isinstance(set, ImageSet):
            # XXX: Maybe this should just be:
            # f2 = set.lambda
            # fun = Lambda(f2.signature, f(*f2.expr))
            # return imageset(fun, *set.base_sets)
            if len(set.lamda.variables) == 1 and len(f.variables) == 1:
                x = set.lamda.variables[0]
                y = f.variables[0]
                return imageset(
                    Lambda(x, f.expr.subs(y, set.lamda.expr)), *set.base_sets)

        if r is not None:
            return r

    return ImageSet(f, *set_list)


def is_function_invertible_in_set(func, setv):
    """
    Checks whether function ``func`` is invertible when the domain is
    restricted to set ``setv``.
    """
    # Functions known to always be invertible:
    if func in (exp, log):
        return True
    u = Dummy("u")
    fdiff = func(u).diff(u)
    # monotonous functions:
    # TODO: check subsets (`func` in `setv`)
    if (fdiff > 0) == True or (fdiff < 0) == True:
        return True
    # TODO: support more
    return None


def simplify_union(args):
    """
    Simplify a :class:`Union` using known rules.

    Explanation
    ===========

    We first start with global rules like 'Merge all FiniteSets'

    Then we iterate through all pairs and ask the constituent sets if they
    can simplify themselves with any other constituent.  This process depends
    on ``union_sets(a, b)`` functions.
    """
    from sympy.sets.handlers.union import union_sets

    # ===== Global Rules =====
    if not args:
        return S.EmptySet

    for arg in args:
        if not isinstance(arg, Set):
            raise TypeError("Input args to Union must be Sets")

    # Merge all finite sets
    finite_sets = [x for x in args if x.is_FiniteSet]
    if len(finite_sets) > 1:
        a = (x for set in finite_sets for x in set)
        finite_set = FiniteSet(*a)
        args = [finite_set] + [x for x in args if not x.is_FiniteSet]

    # ===== Pair-wise Rules =====
    # Here we depend on rules built into the constituent sets
    args = set(args)
    new_args = True
    while new_args:
        for s in args:
            new_args = False
            for t in args - {s}:
                new_set = union_sets(s, t)
                # This returns None if s does not know how to intersect
                # with t. Returns the newly intersected set otherwise
                if new_set is not None:
                    if not isinstance(new_set, set):
                        new_set = {new_set}
                    new_args = (args - {s, t}).union(new_set)
                    break
            if new_args:
                args = new_args
                break

    if len(args) == 1:
        return args.pop()
    else:
        return Union(*args, evaluate=False)


def simplify_intersection(args):
    """
    Simplify an intersection using known rules.

    Explanation
    ===========

    We first start with global rules like
    'if any empty sets return empty set' and 'distribute any unions'

    Then we iterate through all pairs and ask the constituent sets if they
    can simplify themselves with any other constituent
    """

    # ===== Global Rules =====
    if not args:
        return S.UniversalSet

    for arg in args:
        if not isinstance(arg, Set):
            raise TypeError("Input args to Union must be Sets")

    # If any EmptySets return EmptySet
    if S.EmptySet in args:
        return S.EmptySet

    # Handle Finite sets
    rv = Intersection._handle_finite_sets(args)

    if rv is not None:
        return rv

    # If any of the sets are unions, return a Union of Intersections
    for s in args:
        if s.is_Union:
            other_sets = set(args) - {s}
            if len(other_sets) > 0:
                other = Intersection(*other_sets)
                return Union(*(Intersection(arg, other) for arg in s.args))
            else:
                return Union(*s.args)

    for s in args:
        if s.is_Complement:
            args.remove(s)
            other_sets = args + [s.args[0]]
            return Complement(Intersection(*other_sets), s.args[1])

    from sympy.sets.handlers.intersection import intersection_sets

    # At this stage we are guaranteed not to have any
    # EmptySets, FiniteSets, or Unions in the intersection

    # ===== Pair-wise Rules =====
    # Here we depend on rules built into the constituent sets
    args = set(args)
    new_args = True
    while new_args:
        for s in args:
            new_args = False
            for t in args - {s}:
                new_set = intersection_sets(s, t)
                # This returns None if s does not know how to intersect
                # with t. Returns the newly intersected set otherwise

                if new_set is not None:
                    new_args = (args - {s, t}).union({new_set})
                    break
            if new_args:
                args = new_args
                break

    if len(args) == 1:
        return args.pop()
    else:
        return Intersection(*args, evaluate=False)


def _handle_finite_sets(op, x, y, commutative):
    # Handle finite sets:
    fs_args, other = sift([x, y], lambda x: isinstance(x, FiniteSet), binary=True)
    if len(fs_args) == 2:
        return FiniteSet(*[op(i, j) for i in fs_args[0] for j in fs_args[1]])
    elif len(fs_args) == 1:
        sets = [_apply_operation(op, other[0], i, commutative) for i in fs_args[0]]
        return Union(*sets)
    else:
        return None


def _apply_operation(op, x, y, commutative):
    from .fancysets import ImageSet
    d = Dummy('d')

    out = _handle_finite_sets(op, x, y, commutative)
    if out is None:
        out = op(x, y)

    if out is None and commutative:
        out = op(y, x)
    if out is None:
        _x, _y = symbols("x y")
        if isinstance(x, Set) and not isinstance(y, Set):
            out = ImageSet(Lambda(d, op(d, y)), x).doit()
        elif not isinstance(x, Set) and isinstance(y, Set):
            out = ImageSet(Lambda(d, op(x, d)), y).doit()
        else:
            out = ImageSet(Lambda((_x, _y), op(_x, _y)), x, y)
    return out


def set_add(x, y):
    from sympy.sets.handlers.add import _set_add
    return _apply_operation(_set_add, x, y, commutative=True)


def set_sub(x, y):
    from sympy.sets.handlers.add import _set_sub
    return _apply_operation(_set_sub, x, y, commutative=False)


def set_mul(x, y):
    from sympy.sets.handlers.mul import _set_mul
    return _apply_operation(_set_mul, x, y, commutative=True)


def set_div(x, y):
    from sympy.sets.handlers.mul import _set_div
    return _apply_operation(_set_div, x, y, commutative=False)


def set_pow(x, y):
    from sympy.sets.handlers.power import _set_pow
    return _apply_operation(_set_pow, x, y, commutative=False)


def set_function(f, x):
    from sympy.sets.handlers.functions import _set_function
    return _set_function(f, x)


class SetKind(Kind):
    """
    SetKind is kind for all Sets

    Every instance of Set will have kind ``SetKind`` parametrised by the kind
    of the elements of the ``Set``. The kind of the elements might be
    ``NumberKind``, or ``TupleKind`` or something else. When not all elements
    have the same kind then the kind of the elements will be given as
    ``UndefinedKind``.

    Parameters
    ==========

    element_kind: Kind (optional)
        The kind of the elements of the set. In a well defined set all elements
        will have the same kind. Otherwise the kind should
        :class:`sympy.core.kind.UndefinedKind`. The ``element_kind`` argument is optional but
        should only be omitted in the case of ``EmptySet`` whose kind is simply
        ``SetKind()``

    Examples
    ========

    >>> from sympy import Interval
    >>> Interval(1, 2).kind
    SetKind(NumberKind)
    >>> Interval(1,2).kind.element_kind
    NumberKind

    See Also
    ========

    sympy.core.kind.NumberKind
    sympy.matrices.kind.MatrixKind
    sympy.core.containers.TupleKind
    """
    def __new__(cls, element_kind=None):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        if not self.element_kind:
            return "SetKind()"
        else:
            return "SetKind(%s)" % self.element_kind
