from functools import reduce
from itertools import product

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.intfunc import igcd
from sympy.core.numbers import oo, Rational
from sympy.core.relational import Eq, is_eq
from sympy.core.kind import NumberKind
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.logic.boolalg import And, Or
from .sets import tfn, Set, Interval, Union, FiniteSet, ProductSet, SetKind
from sympy.utilities.misc import filldedent


class Rationals(Set, metaclass=Singleton):
    """
    Represents the rational numbers. This set is also available as
    the singleton ``S.Rationals``.

    Examples
    ========

    >>> from sympy import S
    >>> S.Half in S.Rationals
    True
    >>> iterable = iter(S.Rationals)
    >>> [next(iterable) for i in range(12)]
    [0, 1, -1, 1/2, 2, -1/2, -2, 1/3, 3, -1/3, -3, 2/3]
    """

    is_iterable = True
    _inf = S.NegativeInfinity
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        return tfn[other.is_rational]

    def __iter__(self):
        yield S.Zero
        yield S.One
        yield S.NegativeOne
        d = 2
        while True:
            for n in range(d):
                if igcd(n, d) == 1:
                    yield Rational(n, d)
                    yield Rational(d, n)
                    yield Rational(-n, d)
                    yield Rational(-d, n)
            d += 1

    @property
    def _boundary(self):
        return S.Reals

    def _kind(self):
        return SetKind(NumberKind)


class Naturals(Set, metaclass=Singleton):
    """
    Represents the natural numbers (or counting numbers) which are all
    positive integers starting from 1. This set is also available as
    the singleton ``S.Naturals``.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Naturals)
    >>> next(iterable)
    1
    >>> next(iterable)
    2
    >>> next(iterable)
    3
    >>> pprint(S.Naturals.intersect(Interval(0, 10)))
    {1, 2, ..., 10}

    See Also
    ========

    Naturals0 : non-negative integers (i.e. includes 0, too)
    Integers : also includes negative integers
    """

    is_iterable = True
    _inf = S.One
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_positive and other.is_integer:
            return S.true
        elif other.is_integer is False or other.is_positive is False:
            return S.false

    def _eval_is_subset(self, other):
        return Range(1, oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(1, oo).is_superset(other)

    def __iter__(self):
        i = self._inf
        while True:
            yield i
            i = i + 1

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        return And(Eq(floor(x), x), x >= self.inf, x < oo)

    def _kind(self):
        return SetKind(NumberKind)


class Naturals0(Naturals):
    """Represents the whole numbers which are all the non-negative integers,
    inclusive of zero.

    See Also
    ========

    Naturals : positive integers; does not include 0
    Integers : also includes the negative integers
    """
    _inf = S.Zero

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_integer and other.is_nonnegative:
            return S.true
        elif other.is_integer is False or other.is_nonnegative is False:
            return S.false

    def _eval_is_subset(self, other):
        return Range(oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(oo).is_superset(other)


class Integers(Set, metaclass=Singleton):
    """
    Represents all integers: positive, negative and zero. This set is also
    available as the singleton ``S.Integers``.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Integers)
    >>> next(iterable)
    0
    >>> next(iterable)
    1
    >>> next(iterable)
    -1
    >>> next(iterable)
    2

    >>> pprint(S.Integers.intersect(Interval(-4, 4)))
    {-4, -3, ..., 4}

    See Also
    ========

    Naturals0 : non-negative integers
    Integers : positive and negative integers and zero
    """

    is_iterable = True
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        return tfn[other.is_integer]

    def __iter__(self):
        yield S.Zero
        i = S.One
        while True:
            yield i
            yield -i
            i = i + 1

    @property
    def _inf(self):
        return S.NegativeInfinity

    @property
    def _sup(self):
        return S.Infinity

    @property
    def _boundary(self):
        return self

    def _kind(self):
        return SetKind(NumberKind)

    def as_relational(self, x):
        return And(Eq(floor(x), x), -oo < x, x < oo)

    def _eval_is_subset(self, other):
        return Range(-oo, oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(-oo, oo).is_superset(other)


class Reals(Interval, metaclass=Singleton):
    """
    Represents all real numbers
    from negative infinity to positive infinity,
    including all integer, rational and irrational numbers.
    This set is also available as the singleton ``S.Reals``.


    Examples
    ========

    >>> from sympy import S, Rational, pi, I
    >>> 5 in S.Reals
    True
    >>> Rational(-1, 2) in S.Reals
    True
    >>> pi in S.Reals
    True
    >>> 3*I in S.Reals
    False
    >>> S.Reals.contains(pi)
    True


    See Also
    ========

    ComplexRegion
    """
    @property
    def start(self):
        return S.NegativeInfinity

    @property
    def end(self):
        return S.Infinity

    @property
    def left_open(self):
        return True

    @property
    def right_open(self):
        return True

    def __eq__(self, other):
        return other == Interval(S.NegativeInfinity, S.Infinity)

    def __hash__(self):
        return hash(Interval(S.NegativeInfinity, S.Infinity))


class ImageSet(Set):
    """
    Image of a set under a mathematical function. The transformation
    must be given as a Lambda function which has as many arguments
    as the elements of the set upon which it operates, e.g. 1 argument
    when acting on the set of integers or 2 arguments when acting on
    a complex region.

    This function is not normally called directly, but is called
    from ``imageset``.


    Examples
    ========

    >>> from sympy import Symbol, S, pi, Dummy, Lambda
    >>> from sympy import FiniteSet, ImageSet, Interval

    >>> x = Symbol('x')
    >>> N = S.Naturals
    >>> squares = ImageSet(Lambda(x, x**2), N) # {x**2 for x in N}
    >>> 4 in squares
    True
    >>> 5 in squares
    False

    >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
    {1, 4, 9}

    >>> square_iterable = iter(squares)
    >>> for i in range(4):
    ...     next(square_iterable)
    1
    4
    9
    16

    If you want to get value for `x` = 2, 1/2 etc. (Please check whether the
    `x` value is in ``base_set`` or not before passing it as args)

    >>> squares.lamda(2)
    4
    >>> squares.lamda(S(1)/2)
    1/4

    >>> n = Dummy('n')
    >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
    >>> dom = Interval(-1, 1)
    >>> dom.intersect(solutions)
    {0}

    See Also
    ========

    sympy.sets.sets.imageset
    """
    def __new__(cls, flambda, *sets):
        if not isinstance(flambda, Lambda):
            raise ValueError('First argument must be a Lambda')

        signature = flambda.signature

        if len(signature) != len(sets):
            raise ValueError('Incompatible signature')

        sets = [_sympify(s) for s in sets]

        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Set arguments to ImageSet should of type Set")

        if not all(cls._check_sig(sg, st) for sg, st in zip(signature, sets)):
            raise ValueError("Signature %s does not match sets %s" % (signature, sets))

        if flambda is S.IdentityFunction and len(sets) == 1:
            return sets[0]

        if not set(flambda.variables) & flambda.expr.free_symbols:
            is_empty = fuzzy_or(s.is_empty for s in sets)
            if is_empty == True:
                return S.EmptySet
            elif is_empty == False:
                return FiniteSet(flambda.expr)

        return Basic.__new__(cls, flambda, *sets)

    lamda = property(lambda self: self.args[0])
    base_sets = property(lambda self: self.args[1:])

    @property
    def base_set(self):
        # XXX: Maybe deprecate this? It is poorly defined in handling
        # the multivariate case...
        sets = self.base_sets
        if len(sets) == 1:
            return sets[0]
        else:
            return ProductSet(*sets).flatten()

    @property
    def base_pset(self):
        return ProductSet(*self.base_sets)

    @classmethod
    def _check_sig(cls, sig_i, set_i):
        if sig_i.is_symbol:
            return True
        elif isinstance(set_i, ProductSet):
            sets = set_i.sets
            if len(sig_i) != len(sets):
                return False
            # Recurse through the signature for nested tuples:
            return all(cls._check_sig(ts, ps) for ts, ps in zip(sig_i, sets))
        else:
            # XXX: Need a better way of checking whether a set is a set of
            # Tuples or not. For example a FiniteSet can contain Tuples
            # but so can an ImageSet or a ConditionSet. Others like
            # Integers, Reals etc can not contain Tuples. We could just
            # list the possibilities here... Current code for e.g.
            # _contains probably only works for ProductSet.
            return True # Give the benefit of the doubt

    def __iter__(self):
        already_seen = set()
        for i in self.base_pset:
            val = self.lamda(*i)
            if val in already_seen:
                continue
            else:
                already_seen.add(val)
                yield val

    def _is_multivariate(self):
        return len(self.lamda.variables) > 1

    def _contains(self, other):
        from sympy.solvers.solveset import _solveset_multi

        def get_symsetmap(signature, base_sets):
            '''Attempt to get a map of symbols to base_sets'''
            queue = list(zip(signature, base_sets))
            symsetmap = {}
            for sig, base_set in queue:
                if sig.is_symbol:
                    symsetmap[sig] = base_set
                elif base_set.is_ProductSet:
                    sets = base_set.sets
                    if len(sig) != len(sets):
                        raise ValueError("Incompatible signature")
                    # Recurse
                    queue.extend(zip(sig, sets))
                else:
                    # If we get here then we have something like sig = (x, y) and
                    # base_set = {(1, 2), (3, 4)}. For now we give up.
                    return None

            return symsetmap

        def get_equations(expr, candidate):
            '''Find the equations relating symbols in expr and candidate.'''
            queue = [(expr, candidate)]
            for e, c in queue:
                if not isinstance(e, Tuple):
                    yield Eq(e, c)
                elif not isinstance(c, Tuple) or len(e) != len(c):
                    yield False
                    return
                else:
                    queue.extend(zip(e, c))

        # Get the basic objects together:
        other = _sympify(other)
        expr = self.lamda.expr
        sig = self.lamda.signature
        variables = self.lamda.variables
        base_sets = self.base_sets

        # Use dummy symbols for ImageSet parameters so they don't match
        # anything in other
        rep = {v: Dummy(v.name) for v in variables}
        variables = [v.subs(rep) for v in variables]
        sig = sig.subs(rep)
        expr = expr.subs(rep)

        # Map the parts of other to those in the Lambda expr
        equations = []
        for eq in get_equations(expr, other):
            # Unsatisfiable equation?
            if eq is False:
                return S.false
            equations.append(eq)

        # Map the symbols in the signature to the corresponding domains
        symsetmap = get_symsetmap(sig, base_sets)
        if symsetmap is None:
            # Can't factor the base sets to a ProductSet
            return None

        # Which of the variables in the Lambda signature need to be solved for?
        symss = (eq.free_symbols for eq in equations)
        variables = set(variables) & reduce(set.union, symss, set())

        # Use internal multivariate solveset
        variables = tuple(variables)
        base_sets = [symsetmap[v] for v in variables]
        solnset = _solveset_multi(equations, variables, base_sets)
        if solnset is None:
            return None
        return tfn[fuzzy_not(solnset.is_empty)]

    @property
    def is_iterable(self):
        return all(s.is_iterable for s in self.base_sets)

    def doit(self, **hints):
        from sympy.sets.setexpr import SetExpr
        f = self.lamda
        sig = f.signature
        if len(sig) == 1 and sig[0].is_symbol and isinstance(f.expr, Expr):
            base_set = self.base_sets[0]
            return SetExpr(base_set)._eval_func(f).set
        if all(s.is_FiniteSet for s in self.base_sets):
            return FiniteSet(*(f(*a) for a in product(*self.base_sets)))
        return self

    def _kind(self):
        return SetKind(self.lamda.expr.kind)


class Range(Set):
    """
    Represents a range of integers. Can be called as ``Range(stop)``,
    ``Range(start, stop)``, or ``Range(start, stop, step)``; when ``step`` is
    not given it defaults to 1.

    ``Range(stop)`` is the same as ``Range(0, stop, 1)`` and the stop value
    (just as for Python ranges) is not included in the Range values.

        >>> from sympy import Range
        >>> list(Range(3))
        [0, 1, 2]

    The step can also be negative:

        >>> list(Range(10, 0, -2))
        [10, 8, 6, 4, 2]

    The stop value is made canonical so equivalent ranges always
    have the same args:

        >>> Range(0, 10, 3)
        Range(0, 12, 3)

    Infinite ranges are allowed. ``oo`` and ``-oo`` are never included in the
    set (``Range`` is always a subset of ``Integers``). If the starting point
    is infinite, then the final value is ``stop - step``. To iterate such a
    range, it needs to be reversed:

        >>> from sympy import oo
        >>> r = Range(-oo, 1)
        >>> r[-1]
        0
        >>> next(iter(r))
        Traceback (most recent call last):
        ...
        TypeError: Cannot iterate over Range with infinite start
        >>> next(iter(r.reversed))
        0

    Although ``Range`` is a :class:`Set` (and supports the normal set
    operations) it maintains the order of the elements and can
    be used in contexts where ``range`` would be used.

        >>> from sympy import Interval
        >>> Range(0, 10, 2).intersect(Interval(3, 7))
        Range(4, 8, 2)
        >>> list(_)
        [4, 6]

    Although slicing of a Range will always return a Range -- possibly
    empty -- an empty set will be returned from any intersection that
    is empty:

        >>> Range(3)[:0]
        Range(0, 0, 1)
        >>> Range(3).intersect(Interval(4, oo))
        EmptySet
        >>> Range(3).intersect(Range(4, oo))
        EmptySet

    Range will accept symbolic arguments but has very limited support
    for doing anything other than displaying the Range:

        >>> from sympy import Symbol, pprint
        >>> from sympy.abc import i, j, k
        >>> Range(i, j, k).start
        i
        >>> Range(i, j, k).inf
        Traceback (most recent call last):
        ...
        ValueError: invalid method for symbolic range

    Better success will be had when using integer symbols:

        >>> n = Symbol('n', integer=True)
        >>> r = Range(n, n + 20, 3)
        >>> r.inf
        n
        >>> pprint(r)
        {n, n + 3, ..., n + 18}
    """

    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], range):
                raise TypeError(
                    'use sympify(%s) to convert range to Range' % args[0])

        # expand range
        slc = slice(*args)

        if slc.step == 0:
            raise ValueError("step cannot be 0")

        start, stop, step = slc.start or 0, slc.stop, slc.step or 1
        try:
            ok = []
            for w in (start, stop, step):
                w = sympify(w)
                if w in [S.NegativeInfinity, S.Infinity] or (
                        w.has(Symbol) and w.is_integer != False):
                    ok.append(w)
                elif not w.is_Integer:
                    if w.is_infinite:
                        raise ValueError('infinite symbols not allowed')
                    raise ValueError
                else:
                    ok.append(w)
        except ValueError:
            raise ValueError(filldedent('''
    Finite arguments to Range must be integers; `imageset` can define
    other cases, e.g. use `imageset(i, i/10, Range(3))` to give
    [0, 1/10, 1/5].'''))
        start, stop, step = ok

        null = False
        if any(i.has(Symbol) for i in (start, stop, step)):
            dif = stop - start
            n = dif/step
            if n.is_Rational:
                if dif == 0:
                    null = True
                else:  # (x, x + 5, 2) or (x, 3*x, x)
                    n = floor(n)
                    end = start + n*step
                    if dif.is_Rational:  # (x, x + 5, 2)
                        if (end - stop).is_negative:
                            end += step
                    else:  # (x, 3*x, x)
                        if (end/stop - 1).is_negative:
                            end += step
            elif n.is_extended_negative:
                null = True
            else:
                end = stop  # other methods like sup and reversed must fail
        elif start.is_infinite:
            span = step*(stop - start)
            if span is S.NaN or span <= 0:
                null = True
            elif step.is_Integer and stop.is_infinite and abs(step) != 1:
                raise ValueError(filldedent('''
                    Step size must be %s in this case.''' % (1 if step > 0 else -1)))
            else:
                end = stop
        else:
            oostep = step.is_infinite
            if oostep:
                step = S.One if step > 0 else S.NegativeOne
            n = ceiling((stop - start)/step)
            if n <= 0:
                null = True
            elif oostep:
                step = S.One  # make it canonical
                end = start + step
            else:
                end = start + n*step
        if null:
            start = end = S.Zero
            step = S.One
        return Basic.__new__(cls, start, end, step)

    start = property(lambda self: self.args[0])
    stop = property(lambda self: self.args[1])
    step = property(lambda self: self.args[2])

    @property
    def reversed(self):
        """Return an equivalent Range in the opposite order.

        Examples
        ========

        >>> from sympy import Range
        >>> Range(10).reversed
        Range(9, -1, -1)
        """
        if self.has(Symbol):
            n = (self.stop - self.start)/self.step
            if not n.is_extended_positive or not all(
                    i.is_integer or i.is_infinite for i in self.args):
                raise ValueError('invalid method for symbolic range')
        if self.start == self.stop:
            return self
        return self.func(
            self.stop - self.step, self.start - self.step, -self.step)

    def _kind(self):
        return SetKind(NumberKind)

    def _contains(self, other):
        if self.start == self.stop:
            return S.false
        if other.is_infinite:
            return S.false
        if not other.is_integer:
            return tfn[other.is_integer]
        if self.has(Symbol):
            n = (self.stop - self.start)/self.step
            if not n.is_extended_positive or not all(
                    i.is_integer or i.is_infinite for i in self.args):
                return
        else:
            n = self.size
        if self.start.is_finite:
            ref = self.start
        elif self.stop.is_finite:
            ref = self.stop
        else:  # both infinite; step is +/- 1 (enforced by __new__)
            return S.true
        if n == 1:
            return Eq(other, self[0])
        res = (ref - other) % self.step
        if res == S.Zero:
            if self.has(Symbol):
                d = Dummy('i')
                return self.as_relational(d).subs(d, other)
            return And(other >= self.inf, other <= self.sup)
        elif res.is_Integer:  # off sequence
            return S.false
        else:  # symbolic/unsimplified residue modulo step
            return None

    def __iter__(self):
        n = self.size  # validate
        if not (n.has(S.Infinity) or n.has(S.NegativeInfinity) or n.is_Integer):
            raise TypeError("Cannot iterate over symbolic Range")
        if self.start in [S.NegativeInfinity, S.Infinity]:
            raise TypeError("Cannot iterate over Range with infinite start")
        elif self.start != self.stop:
            i = self.start
            if n.is_infinite:
                while True:
                    yield i
                    i += self.step
            else:
                for _ in range(n):
                    yield i
                    i += self.step

    @property
    def is_iterable(self):
        # Check that size can be determined, used by __iter__
        dif = self.stop - self.start
        n = dif/self.step
        if not (n.has(S.Infinity) or n.has(S.NegativeInfinity) or n.is_Integer):
            return False
        if self.start in [S.NegativeInfinity, S.Infinity]:
            return False
        if not (n.is_extended_nonnegative and all(i.is_integer for i in self.args)):
            return False
        return True

    def __len__(self):
        rv = self.size
        if rv is S.Infinity:
            raise ValueError('Use .size to get the length of an infinite Range')
        return int(rv)

    @property
    def size(self):
        if self.start == self.stop:
            return S.Zero
        dif = self.stop - self.start
        n = dif/self.step
        if n.is_infinite:
            return S.Infinity
        if  n.is_extended_nonnegative and all(i.is_integer for i in self.args):
            return abs(floor(n))
        raise ValueError('Invalid method for symbolic Range')

    @property
    def is_finite_set(self):
        if self.start.is_integer and self.stop.is_integer:
            return True
        return self.size.is_finite

    @property
    def is_empty(self):
        try:
            return self.size.is_zero
        except ValueError:
            return None

    def __bool__(self):
        # this only distinguishes between definite null range
        # and non-null/unknown null; getting True doesn't mean
        # that it actually is not null
        b = is_eq(self.start, self.stop)
        if b is None:
            raise ValueError('cannot tell if Range is null or not')
        return not bool(b)

    def __getitem__(self, i):
        ooslice = "cannot slice from the end with an infinite value"
        zerostep = "slice step cannot be zero"
        infinite = "slicing not possible on range with infinite start"
        # if we had to take every other element in the following
        # oo, ..., 6, 4, 2, 0
        # we might get oo, ..., 4, 0 or oo, ..., 6, 2
        ambiguous = "cannot unambiguously re-stride from the end " + \
            "with an infinite value"
        if isinstance(i, slice):
            if self.size.is_finite:  # validates, too
                if self.start == self.stop:
                    return Range(0)
                start, stop, step = i.indices(self.size)
                n = ceiling((stop - start)/step)
                if n <= 0:
                    return Range(0)
                canonical_stop = start + n*step
                end = canonical_stop - step
                ss = step*self.step
                return Range(self[start], self[end] + ss, ss)
            else:  # infinite Range
                start = i.start
                stop = i.stop
                if i.step == 0:
                    raise ValueError(zerostep)
                step = i.step or 1
                ss = step*self.step
                #---------------------
                # handle infinite Range
                #   i.e. Range(-oo, oo) or Range(oo, -oo, -1)
                # --------------------
                if self.start.is_infinite and self.stop.is_infinite:
                    raise ValueError(infinite)
                #---------------------
                # handle infinite on right
                #   e.g. Range(0, oo) or Range(0, -oo, -1)
                # --------------------
                if self.stop.is_infinite:
                    # start and stop are not interdependent --
                    # they only depend on step --so we use the
                    # equivalent reversed values
                    return self.reversed[
                        stop if stop is None else -stop + 1:
                        start if start is None else -start:
                        step].reversed
                #---------------------
                # handle infinite on the left
                #   e.g. Range(oo, 0, -1) or Range(-oo, 0)
                # --------------------
                # consider combinations of
                # start/stop {== None, < 0, == 0, > 0} and
                # step {< 0, > 0}
                if start is None:
                    if stop is None:
                        if step < 0:
                            return Range(self[-1], self.start, ss)
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step < 0:
                            return Range(self[-1], self[stop], ss)
                        else:  # > 0
                            return Range(self.start, self[stop], ss)
                    elif stop == 0:
                        if step > 0:
                            return Range(0)
                        else:  # < 0
                            raise ValueError(ooslice)
                    elif stop == 1:
                        if step > 0:
                            raise ValueError(ooslice)  # infinite singleton
                        else:  # < 0
                            raise ValueError(ooslice)
                    else:  # > 1
                        raise ValueError(ooslice)
                elif start < 0:
                    if stop is None:
                        if step < 0:
                            return Range(self[start], self.start, ss)
                        else:  # > 0
                            return Range(self[start], self.stop, ss)
                    elif stop < 0:
                        return Range(self[start], self[stop], ss)
                    elif stop == 0:
                        if step < 0:
                            raise ValueError(ooslice)
                        else:  # > 0
                            return Range(0)
                    elif stop > 0:
                        raise ValueError(ooslice)
                elif start == 0:
                    if stop is None:
                        if step < 0:
                            raise ValueError(ooslice)  # infinite singleton
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step > 1:
                            raise ValueError(ambiguous)
                        elif step == 1:
                            return Range(self.start, self[stop], ss)
                        else:  # < 0
                            return Range(0)
                    else:  # >= 0
                        raise ValueError(ooslice)
                elif start > 0:
                    raise ValueError(ooslice)
        else:
            if self.start == self.stop:
                raise IndexError('Range index out of range')
            if not (all(i.is_integer or i.is_infinite
                    for i in self.args) and ((self.stop - self.start)/
                    self.step).is_extended_positive):
                raise ValueError('Invalid method for symbolic Range')
            if i == 0:
                if self.start.is_infinite:
                    raise ValueError(ooslice)
                return self.start
            if i == -1:
                if self.stop.is_infinite:
                    raise ValueError(ooslice)
                return self.stop - self.step
            n = self.size  # must be known for any other index
            rv = (self.stop if i < 0 else self.start) + i*self.step
            if rv.is_infinite:
                raise ValueError(ooslice)
            val = (rv - self.start)/self.step
            rel = fuzzy_or([val.is_infinite,
                            fuzzy_and([val.is_nonnegative, (n-val).is_nonnegative])])
            if rel:
                return rv
            if rel is None:
                raise ValueError('Invalid method for symbolic Range')
            raise IndexError("Range index out of range")

    @property
    def _inf(self):
        if not self:
            return S.EmptySet.inf
        if self.has(Symbol):
            if all(i.is_integer or i.is_infinite for i in self.args):
                dif = self.stop - self.start
                if self.step.is_positive and dif.is_positive:
                    return self.start
                elif self.step.is_negative and dif.is_negative:
                    return self.stop - self.step
            raise ValueError('invalid method for symbolic range')
        if self.step > 0:
            return self.start
        else:
            return self.stop - self.step

    @property
    def _sup(self):
        if not self:
            return S.EmptySet.sup
        if self.has(Symbol):
            if all(i.is_integer or i.is_infinite for i in self.args):
                dif = self.stop - self.start
                if self.step.is_positive and dif.is_positive:
                    return self.stop - self.step
                elif self.step.is_negative and dif.is_negative:
                    return self.start
            raise ValueError('invalid method for symbolic range')
        if self.step > 0:
            return self.stop - self.step
        else:
            return self.start

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        """Rewrite a Range in terms of equalities and logic operators. """
        if self.start.is_infinite:
            assert not self.stop.is_infinite  # by instantiation
            a = self.reversed.start
        else:
            a = self.start
        step = self.step
        in_seq = Eq(Mod(x - a, step), 0)
        ints = And(Eq(Mod(a, 1), 0), Eq(Mod(step, 1), 0))
        n = (self.stop - self.start)/self.step
        if n == 0:
            return S.EmptySet.as_relational(x)
        if n == 1:
            return And(Eq(x, a), ints)
        try:
            a, b = self.inf, self.sup
        except ValueError:
            a = None
        if a is not None:
            range_cond = And(
                x > a if a.is_infinite else x >= a,
                x < b if b.is_infinite else x <= b)
        else:
            a, b = self.start, self.stop - self.step
            range_cond = Or(
                And(self.step >= 1, x > a if a.is_infinite else x >= a,
                x < b if b.is_infinite else x <= b),
                And(self.step <= -1, x < a if a.is_infinite else x <= a,
                x > b if b.is_infinite else x >= b))
        return And(in_seq, ints, range_cond)


_sympy_converter[range] = lambda r: Range(r.start, r.stop, r.step)

def normalize_theta_set(theta):
    r"""
    Normalize a Real Set `theta` in the interval `[0, 2\pi)`. It returns
    a normalized value of theta in the Set. For Interval, a maximum of
    one cycle $[0, 2\pi]$, is returned i.e. for theta equal to $[0, 10\pi]$,
    returned normalized value would be $[0, 2\pi)$. As of now intervals
    with end points as non-multiples of ``pi`` is not supported.

    Raises
    ======

    NotImplementedError
        The algorithms for Normalizing theta Set are not yet
        implemented.
    ValueError
        The input is not valid, i.e. the input is not a real set.
    RuntimeError
        It is a bug, please report to the github issue tracker.

    Examples
    ========

    >>> from sympy.sets.fancysets import normalize_theta_set
    >>> from sympy import Interval, FiniteSet, pi
    >>> normalize_theta_set(Interval(9*pi/2, 5*pi))
    Interval(pi/2, pi)
    >>> normalize_theta_set(Interval(-3*pi/2, pi/2))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-pi/2, pi/2))
    Union(Interval(0, pi/2), Interval.Ropen(3*pi/2, 2*pi))
    >>> normalize_theta_set(Interval(-4*pi, 3*pi))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-3*pi/2, -pi/2))
    Interval(pi/2, 3*pi/2)
    >>> normalize_theta_set(FiniteSet(0, pi, 3*pi))
    {0, pi}

    """
    from sympy.functions.elementary.trigonometric import _pi_coeff

    if theta.is_Interval:
        interval_len = theta.measure
        # one complete circle
        if interval_len >= 2*S.Pi:
            if interval_len == 2*S.Pi and theta.left_open and theta.right_open:
                k = _pi_coeff(theta.start)
                return Union(Interval(0, k*S.Pi, False, True),
                        Interval(k*S.Pi, 2*S.Pi, True, True))
            return Interval(0, 2*S.Pi, False, True)

        k_start, k_end = _pi_coeff(theta.start), _pi_coeff(theta.end)

        if k_start is None or k_end is None:
            raise NotImplementedError("Normalizing theta without pi as coefficient is "
                                    "not yet implemented")
        new_start = k_start*S.Pi
        new_end = k_end*S.Pi

        if new_start > new_end:
            return Union(Interval(S.Zero, new_end, False, theta.right_open),
                         Interval(new_start, 2*S.Pi, theta.left_open, True))
        else:
            return Interval(new_start, new_end, theta.left_open, theta.right_open)

    elif theta.is_FiniteSet:
        new_theta = []
        for element in theta:
            k = _pi_coeff(element)
            if k is None:
                raise NotImplementedError('Normalizing theta without pi as '
                                          'coefficient, is not Implemented.')
            else:
                new_theta.append(k*S.Pi)
        return FiniteSet(*new_theta)

    elif theta.is_Union:
        return Union(*[normalize_theta_set(interval) for interval in theta.args])

    elif theta.is_subset(S.Reals):
        raise NotImplementedError("Normalizing theta when, it is of type %s is not "
                                  "implemented" % type(theta))
    else:
        raise ValueError(" %s is not a real set" % (theta))


class ComplexRegion(Set):
    r"""
    Represents the Set of all Complex Numbers. It can represent a
    region of Complex Plane in both the standard forms Polar and
    Rectangular coordinates.

    * Polar Form
      Input is in the form of the ProductSet or Union of ProductSets
      of the intervals of ``r`` and ``theta``, and use the flag ``polar=True``.

      .. math:: Z = \{z \in \mathbb{C} \mid z = r\times (\cos(\theta) + I\sin(\theta)), r \in [\texttt{r}], \theta \in [\texttt{theta}]\}

    * Rectangular Form
      Input is in the form of the ProductSet or Union of ProductSets
      of interval of x and y, the real and imaginary parts of the Complex numbers in a plane.
      Default input type is in rectangular form.

    .. math:: Z = \{z \in \mathbb{C} \mid z = x + Iy, x \in [\operatorname{re}(z)], y \in [\operatorname{im}(z)]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, Interval, S, I, Union
    >>> a = Interval(2, 3)
    >>> b = Interval(4, 6)
    >>> c1 = ComplexRegion(a*b)  # Rectangular Form
    >>> c1
    CartesianComplexRegion(ProductSet(Interval(2, 3), Interval(4, 6)))

    * c1 represents the rectangular region in complex plane
      surrounded by the coordinates (2, 4), (3, 4), (3, 6) and
      (2, 6), of the four vertices.

    >>> c = Interval(1, 8)
    >>> c2 = ComplexRegion(Union(a*b, b*c))
    >>> c2
    CartesianComplexRegion(Union(ProductSet(Interval(2, 3), Interval(4, 6)), ProductSet(Interval(4, 6), Interval(1, 8))))

    * c2 represents the Union of two rectangular regions in complex
      plane. One of them surrounded by the coordinates of c1 and
      other surrounded by the coordinates (4, 1), (6, 1), (6, 8) and
      (4, 8).

    >>> 2.5 + 4.5*I in c1
    True
    >>> 2.5 + 6.5*I in c1
    False

    >>> r = Interval(0, 1)
    >>> theta = Interval(0, 2*S.Pi)
    >>> c2 = ComplexRegion(r*theta, polar=True)  # Polar Form
    >>> c2  # unit Disk
    PolarComplexRegion(ProductSet(Interval(0, 1), Interval.Ropen(0, 2*pi)))

    * c2 represents the region in complex plane inside the
      Unit Disk centered at the origin.

    >>> 0.5 + 0.5*I in c2
    True
    >>> 1 + 2*I in c2
    False

    >>> unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    >>> upper_half_unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    >>> intersection = unit_disk.intersect(upper_half_unit_disk)
    >>> intersection
    PolarComplexRegion(ProductSet(Interval(0, 1), Interval(0, pi)))
    >>> intersection == upper_half_unit_disk
    True

    See Also
    ========

    CartesianComplexRegion
    PolarComplexRegion
    Complexes

    """
    is_ComplexRegion = True

    def __new__(cls, sets, polar=False):
        if polar is False:
            return CartesianComplexRegion(sets)
        elif polar is True:
            return PolarComplexRegion(sets)
        else:
            raise ValueError("polar should be either True or False")

    @property
    def sets(self):
        """
        Return raw input sets to the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.sets
        ProductSet(Interval(2, 3), Interval(4, 5))
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.sets
        Union(ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        return self.args[0]

    @property
    def psets(self):
        """
        Return a tuple of sets (ProductSets) input of the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)),)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        if self.sets.is_ProductSet:
            psets = ()
            psets = psets + (self.sets, )
        else:
            psets = self.sets.args
        return psets

    @property
    def a_interval(self):
        """
        Return the union of intervals of `x` when, self is in
        rectangular form, or the union of intervals of `r` when
        self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.a_interval
        Interval(2, 3)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.a_interval
        Union(Interval(2, 3), Interval(4, 5))

        """
        a_interval = []
        for element in self.psets:
            a_interval.append(element.args[0])

        a_interval = Union(*a_interval)
        return a_interval

    @property
    def b_interval(self):
        """
        Return the union of intervals of `y` when, self is in
        rectangular form, or the union of intervals of `theta`
        when self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.b_interval
        Interval(4, 5)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.b_interval
        Interval(1, 7)

        """
        b_interval = []
        for element in self.psets:
            b_interval.append(element.args[1])

        b_interval = Union(*b_interval)
        return b_interval

    @property
    def _measure(self):
        """
        The measure of self.sets.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, S
        >>> a, b = Interval(2, 5), Interval(4, 8)
        >>> c = Interval(0, 2*S.Pi)
        >>> c1 = ComplexRegion(a*b)
        >>> c1.measure
        12
        >>> c2 = ComplexRegion(a*c, polar=True)
        >>> c2.measure
        6*pi

        """
        return self.sets._measure

    def _kind(self):
        return self.args[0].kind

    @classmethod
    def from_real(cls, sets):
        """
        Converts given subset of real numbers to a complex region.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion
        >>> unit = Interval(0,1)
        >>> ComplexRegion.from_real(unit)
        CartesianComplexRegion(ProductSet(Interval(0, 1), {0}))

        """
        if not sets.is_subset(S.Reals):
            raise ValueError("sets must be a subset of the real line")

        return CartesianComplexRegion(sets * FiniteSet(0))

    def _contains(self, other):
        from sympy.functions import arg, Abs

        isTuple = isinstance(other, Tuple)
        if isTuple and len(other) != 2:
            raise ValueError('expecting Tuple of length 2')

        # If the other is not an Expression, and neither a Tuple
        if not isinstance(other, (Expr, Tuple)):
            return S.false

        # self in rectangular form
        if not self.polar:
            re, im = other if isTuple else other.as_real_imag()
            return tfn[fuzzy_or(fuzzy_and([
                pset.args[0]._contains(re),
                pset.args[1]._contains(im)])
                for pset in self.psets)]

        # self in polar form
        elif self.polar:
            if other.is_zero:
                # ignore undefined complex argument
                return tfn[fuzzy_or(pset.args[0]._contains(S.Zero)
                    for pset in self.psets)]
            if isTuple:
                r, theta = other
            else:
                r, theta = Abs(other), arg(other)
            if theta.is_real and theta.is_number:
                # angles in psets are normalized to [0, 2pi)
                theta %= 2*S.Pi
                return tfn[fuzzy_or(fuzzy_and([
                    pset.args[0]._contains(r),
                    pset.args[1]._contains(theta)])
                    for pset in self.psets)]


class CartesianComplexRegion(ComplexRegion):
    r"""
    Set representing a square region of the complex plane.

    .. math:: Z = \{z \in \mathbb{C} \mid z = x + Iy, x \in [\operatorname{re}(z)], y \in [\operatorname{im}(z)]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, I, Interval
    >>> region = ComplexRegion(Interval(1, 3) * Interval(4, 6))
    >>> 2 + 5*I in region
    True
    >>> 5*I in region
    False

    See also
    ========

    ComplexRegion
    PolarComplexRegion
    Complexes
    """

    polar = False
    variables = symbols('x, y', cls=Dummy)

    def __new__(cls, sets):

        if sets == S.Reals*S.Reals:
            return S.Complexes

        if all(_a.is_FiniteSet for _a in sets.args) and (len(sets.args) == 2):

            # ** ProductSet of FiniteSets in the Complex Plane. **
            # For Cases like ComplexRegion({2, 4}*{3}), It
            # would return {2 + 3*I, 4 + 3*I}

            # FIXME: This should probably be handled with something like:
            # return ImageSet(Lambda((x, y), x+I*y), sets).rewrite(FiniteSet)
            complex_num = []
            for x in sets.args[0]:
                for y in sets.args[1]:
                    complex_num.append(x + S.ImaginaryUnit*y)
            return FiniteSet(*complex_num)
        else:
            return Set.__new__(cls, sets)

    @property
    def expr(self):
        x, y = self.variables
        return x + S.ImaginaryUnit*y


class PolarComplexRegion(ComplexRegion):
    r"""
    Set representing a polar region of the complex plane.

    .. math:: Z = \{z \in \mathbb{C} \mid z = r\times (\cos(\theta) + I\sin(\theta)), r \in [\texttt{r}], \theta \in [\texttt{theta}]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, Interval, oo, pi, I
    >>> rset = Interval(0, oo)
    >>> thetaset = Interval(0, pi)
    >>> upper_half_plane = ComplexRegion(rset * thetaset, polar=True)
    >>> 1 + I in upper_half_plane
    True
    >>> 1 - I in upper_half_plane
    False

    See also
    ========

    ComplexRegion
    CartesianComplexRegion
    Complexes

    """

    polar = True
    variables = symbols('r, theta', cls=Dummy)

    def __new__(cls, sets):

        new_sets = []
        # sets is Union of ProductSets
        if not sets.is_ProductSet:
            for k in sets.args:
                new_sets.append(k)
        # sets is ProductSets
        else:
            new_sets.append(sets)
        # Normalize input theta
        for k, v in enumerate(new_sets):
            new_sets[k] = ProductSet(v.args[0],
                                     normalize_theta_set(v.args[1]))
        sets = Union(*new_sets)
        return Set.__new__(cls, sets)

    @property
    def expr(self):
        r, theta = self.variables
        return r*(cos(theta) + S.ImaginaryUnit*sin(theta))


class Complexes(CartesianComplexRegion, metaclass=Singleton):
    """
    The :class:`Set` of all complex numbers

    Examples
    ========

    >>> from sympy import S, I
    >>> S.Complexes
    Complexes
    >>> 1 + I in S.Complexes
    True

    See also
    ========

    Reals
    ComplexRegion

    """

    is_empty = False
    is_finite_set = False

    # Override property from superclass since Complexes has no args
    @property
    def sets(self):
        return ProductSet(S.Reals, S.Reals)

    def __new__(cls):
        return Set.__new__(cls)
