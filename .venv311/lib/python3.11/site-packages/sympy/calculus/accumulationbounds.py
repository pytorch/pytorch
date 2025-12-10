from sympy.core import Add, Mul, Pow, S
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import _sympifyit, oo, zoo
from sympy.core.relational import is_le, is_lt, is_ge, is_gt
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import And
from sympy.multipledispatch import dispatch
from sympy.series.order import Order
from sympy.sets.sets import FiniteSet


class AccumulationBounds(Expr):
    r"""An accumulation bounds.

    # Note AccumulationBounds has an alias: AccumBounds

    AccumulationBounds represent an interval `[a, b]`, which is always closed
    at the ends. Here `a` and `b` can be any value from extended real numbers.

    The intended meaning of AccummulationBounds is to give an approximate
    location of the accumulation points of a real function at a limit point.

    Let `a` and `b` be reals such that `a \le b`.

    `\left\langle a, b\right\rangle = \{x \in \mathbb{R} \mid a \le x \le b\}`

    `\left\langle -\infty, b\right\rangle = \{x \in \mathbb{R} \mid x \le b\} \cup \{-\infty, \infty\}`

    `\left\langle a, \infty \right\rangle = \{x \in \mathbb{R} \mid a \le x\} \cup \{-\infty, \infty\}`

    `\left\langle -\infty, \infty \right\rangle = \mathbb{R} \cup \{-\infty, \infty\}`

    ``oo`` and ``-oo`` are added to the second and third definition respectively,
    since if either ``-oo`` or ``oo`` is an argument, then the other one should
    be included (though not as an end point). This is forced, since we have,
    for example, ``1/AccumBounds(0, 1) = AccumBounds(1, oo)``, and the limit at
    `0` is not one-sided. As `x` tends to `0-`, then `1/x \rightarrow -\infty`, so `-\infty`
    should be interpreted as belonging to ``AccumBounds(1, oo)`` though it need
    not appear explicitly.

    In many cases it suffices to know that the limit set is bounded.
    However, in some other cases more exact information could be useful.
    For example, all accumulation values of `\cos(x) + 1` are non-negative.
    (``AccumBounds(-1, 1) + 1 = AccumBounds(0, 2)``)

    A AccumulationBounds object is defined to be real AccumulationBounds,
    if its end points are finite reals.

    Let `X`, `Y` be real AccumulationBounds, then their sum, difference,
    product are defined to be the following sets:

    `X + Y = \{ x+y \mid x \in X \cap y \in Y\}`

    `X - Y = \{ x-y \mid x \in X \cap y \in Y\}`

    `X \times Y = \{ x \times y \mid x \in X \cap y \in Y\}`

    When an AccumBounds is raised to a negative power, if 0 is contained
    between the bounds then an infinite range is returned, otherwise if an
    endpoint is 0 then a semi-infinite range with consistent sign will be returned.

    AccumBounds in expressions behave a lot like Intervals but the
    semantics are not necessarily the same. Division (or exponentiation
    to a negative integer power) could be handled with *intervals* by
    returning a union of the results obtained after splitting the
    bounds between negatives and positives, but that is not done with
    AccumBounds. In addition, bounds are assumed to be independent of
    each other; if the same bound is used in more than one place in an
    expression, the result may not be the supremum or infimum of the
    expression (see below). Finally, when a boundary is ``1``,
    exponentiation to the power of ``oo`` yields ``oo``, neither
    ``1`` nor ``nan``.

    Examples
    ========

    >>> from sympy import AccumBounds, sin, exp, log, pi, E, S, oo
    >>> from sympy.abc import x

    >>> AccumBounds(0, 1) + AccumBounds(1, 2)
    AccumBounds(1, 3)

    >>> AccumBounds(0, 1) - AccumBounds(0, 2)
    AccumBounds(-2, 1)

    >>> AccumBounds(-2, 3)*AccumBounds(-1, 1)
    AccumBounds(-3, 3)

    >>> AccumBounds(1, 2)*AccumBounds(3, 5)
    AccumBounds(3, 10)

    The exponentiation of AccumulationBounds is defined
    as follows:

    If 0 does not belong to `X` or `n > 0` then

    `X^n = \{ x^n \mid x \in X\}`

    >>> AccumBounds(1, 4)**(S(1)/2)
    AccumBounds(1, 2)

    otherwise, an infinite or semi-infinite result is obtained:

    >>> 1/AccumBounds(-1, 1)
    AccumBounds(-oo, oo)
    >>> 1/AccumBounds(0, 2)
    AccumBounds(1/2, oo)
    >>> 1/AccumBounds(-oo, 0)
    AccumBounds(-oo, 0)

    A boundary of 1 will always generate all nonnegatives:

    >>> AccumBounds(1, 2)**oo
    AccumBounds(0, oo)
    >>> AccumBounds(0, 1)**oo
    AccumBounds(0, oo)

    If the exponent is itself an AccumulationBounds or is not an
    integer then unevaluated results will be returned unless the base
    values are positive:

    >>> AccumBounds(2, 3)**AccumBounds(-1, 2)
    AccumBounds(1/3, 9)
    >>> AccumBounds(-2, 3)**AccumBounds(-1, 2)
    AccumBounds(-2, 3)**AccumBounds(-1, 2)

    >>> AccumBounds(-2, -1)**(S(1)/2)
    sqrt(AccumBounds(-2, -1))

    Note: `\left\langle a, b\right\rangle^2` is not same as `\left\langle a, b\right\rangle \times \left\langle a, b\right\rangle`

    >>> AccumBounds(-1, 1)**2
    AccumBounds(0, 1)

    >>> AccumBounds(1, 3) < 4
    True

    >>> AccumBounds(1, 3) < -1
    False

    Some elementary functions can also take AccumulationBounds as input.
    A function `f` evaluated for some real AccumulationBounds `\left\langle a, b \right\rangle`
    is defined as `f(\left\langle a, b\right\rangle) = \{ f(x) \mid a \le x \le b \}`

    >>> sin(AccumBounds(pi/6, pi/3))
    AccumBounds(1/2, sqrt(3)/2)

    >>> exp(AccumBounds(0, 1))
    AccumBounds(1, E)

    >>> log(AccumBounds(1, E))
    AccumBounds(0, 1)

    Some symbol in an expression can be substituted for a AccumulationBounds
    object. But it does not necessarily evaluate the AccumulationBounds for
    that expression.

    The same expression can be evaluated to different values depending upon
    the form it is used for substitution since each instance of an
    AccumulationBounds is considered independent. For example:

    >>> (x**2 + 2*x + 1).subs(x, AccumBounds(-1, 1))
    AccumBounds(-1, 4)

    >>> ((x + 1)**2).subs(x, AccumBounds(-1, 1))
    AccumBounds(0, 4)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_arithmetic

    .. [2] https://fab.cba.mit.edu/classes/S62.12/docs/Hickey_interval.pdf

    Notes
    =====

    Do not use ``AccumulationBounds`` for floating point interval arithmetic
    calculations, use ``mpmath.iv`` instead.
    """

    is_extended_real = True
    is_number = False

    def __new__(cls, min, max) -> Expr: # type: ignore

        min = _sympify(min)
        max = _sympify(max)

        # Only allow real intervals (use symbols with 'is_extended_real=True').
        if not min.is_extended_real or not max.is_extended_real:
            raise ValueError("Only real AccumulationBounds are supported")

        if max == min:
            return max

        # Make sure that the created AccumBounds object will be valid.
        if max.is_number and min.is_number:
            bad = max.is_comparable and min.is_comparable and max < min
        else:
            bad = (max - min).is_extended_negative
        if bad:
            raise ValueError(
                "Lower limit should be smaller than upper limit")

        return Basic.__new__(cls, min, max)

    # setting the operation priority
    _op_priority = 11.0

    def _eval_is_real(self):
        if self.min.is_real and self.max.is_real:
            return True

    @property
    def min(self):
        """
        Returns the minimum possible value attained by AccumulationBounds
        object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).min
        1

        """
        return self.args[0]

    @property
    def max(self):
        """
        Returns the maximum possible value attained by AccumulationBounds
        object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).max
        3

        """
        return self.args[1]

    @property
    def delta(self):
        """
        Returns the difference of maximum possible value attained by
        AccumulationBounds object and minimum possible value attained
        by AccumulationBounds object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).delta
        2

        """
        return self.max - self.min

    @property
    def mid(self):
        """
        Returns the mean of maximum possible value attained by
        AccumulationBounds object and minimum possible value
        attained by AccumulationBounds object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).mid
        2

        """
        return (self.min + self.max) / 2

    @_sympifyit('other', NotImplemented)
    def _eval_power(self, other):
        return self.__pow__(other)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                return AccumBounds(
                    Add(self.min, other.min),
                    Add(self.max, other.max))
            if other is S.Infinity and self.min is S.NegativeInfinity or \
                    other is S.NegativeInfinity and self.max is S.Infinity:
                return AccumBounds(-oo, oo)
            elif other.is_extended_real:
                if self.min is S.NegativeInfinity and self.max is S.Infinity:
                    return AccumBounds(-oo, oo)
                elif self.min is S.NegativeInfinity:
                    return AccumBounds(-oo, self.max + other)
                elif self.max is S.Infinity:
                    return AccumBounds(self.min + other, oo)
                else:
                    return AccumBounds(Add(self.min, other), Add(self.max, other))
            return Add(self, other, evaluate=False)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return AccumBounds(-self.max, -self.min)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                return AccumBounds(
                    Add(self.min, -other.max),
                    Add(self.max, -other.min))
            if other is S.NegativeInfinity and self.min is S.NegativeInfinity or \
                    other is S.Infinity and self.max is S.Infinity:
                return AccumBounds(-oo, oo)
            elif other.is_extended_real:
                if self.min is S.NegativeInfinity and self.max is S.Infinity:
                    return AccumBounds(-oo, oo)
                elif self.min is S.NegativeInfinity:
                    return AccumBounds(-oo, self.max - other)
                elif self.max is S.Infinity:
                    return AccumBounds(self.min - other, oo)
                else:
                    return AccumBounds(
                        Add(self.min, -other),
                        Add(self.max, -other))
            return Add(self, -other, evaluate=False)
        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        return self.__neg__() + other

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if self.args == (-oo, oo):
            return self
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                if other.args == (-oo, oo):
                    return other
                v = set()
                for a in self.args:
                    vi = other*a
                    v.update(vi.args or (vi,))
                return AccumBounds(Min(*v), Max(*v))
            if other is S.Infinity:
                if self.min.is_zero:
                    return AccumBounds(0, oo)
                if self.max.is_zero:
                    return AccumBounds(-oo, 0)
            if other is S.NegativeInfinity:
                if self.min.is_zero:
                    return AccumBounds(-oo, 0)
                if self.max.is_zero:
                    return AccumBounds(0, oo)
            if other.is_extended_real:
                if other.is_zero:
                    if self.max is S.Infinity:
                        return AccumBounds(0, oo)
                    if self.min is S.NegativeInfinity:
                        return AccumBounds(-oo, 0)
                    return S.Zero
                if other.is_extended_positive:
                    return AccumBounds(
                        Mul(self.min, other),
                        Mul(self.max, other))
                elif other.is_extended_negative:
                    return AccumBounds(
                        Mul(self.max, other),
                        Mul(self.min, other))
            if isinstance(other, Order):
                return other
            return Mul(self, other, evaluate=False)
        return NotImplemented

    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                if other.min.is_positive or other.max.is_negative:
                    return self * AccumBounds(1/other.max, 1/other.min)

                if (self.min.is_extended_nonpositive and self.max.is_extended_nonnegative and
                    other.min.is_extended_nonpositive and other.max.is_extended_nonnegative):
                    if self.min.is_zero and other.min.is_zero:
                        return AccumBounds(0, oo)
                    if self.max.is_zero and other.min.is_zero:
                        return AccumBounds(-oo, 0)
                    return AccumBounds(-oo, oo)

                if self.max.is_extended_negative:
                    if other.min.is_extended_negative:
                        if other.max.is_zero:
                            return AccumBounds(self.max / other.min, oo)
                        if other.max.is_extended_positive:
                            # if we were dealing with intervals we would return
                            # Union(Interval(-oo, self.max/other.max),
                            #       Interval(self.max/other.min, oo))
                            return AccumBounds(-oo, oo)

                    if other.min.is_zero and other.max.is_extended_positive:
                        return AccumBounds(-oo, self.max / other.max)

                if self.min.is_extended_positive:
                    if other.min.is_extended_negative:
                        if other.max.is_zero:
                            return AccumBounds(-oo, self.min / other.min)
                        if other.max.is_extended_positive:
                            # if we were dealing with intervals we would return
                            # Union(Interval(-oo, self.min/other.min),
                            #       Interval(self.min/other.max, oo))
                            return AccumBounds(-oo, oo)

                    if other.min.is_zero and other.max.is_extended_positive:
                        return AccumBounds(self.min / other.max, oo)

            elif other.is_extended_real:
                if other in (S.Infinity, S.NegativeInfinity):
                    if self == AccumBounds(-oo, oo):
                        return AccumBounds(-oo, oo)
                    if self.max is S.Infinity:
                        return AccumBounds(Min(0, other), Max(0, other))
                    if self.min is S.NegativeInfinity:
                        return AccumBounds(Min(0, -other), Max(0, -other))
                if other.is_extended_positive:
                    return AccumBounds(self.min / other, self.max / other)
                elif other.is_extended_negative:
                    return AccumBounds(self.max / other, self.min / other)
            if (1 / other) is S.ComplexInfinity:
                return Mul(self, 1 / other, evaluate=False)
            else:
                return Mul(self, 1 / other)

        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rtruediv__(self, other):
        if isinstance(other, Expr):
            if other.is_extended_real:
                if other.is_zero:
                    return S.Zero
                if (self.min.is_extended_nonpositive and self.max.is_extended_nonnegative):
                    if self.min.is_zero:
                        if other.is_extended_positive:
                            return AccumBounds(Mul(other, 1 / self.max), oo)
                        if other.is_extended_negative:
                            return AccumBounds(-oo, Mul(other, 1 / self.max))
                    if self.max.is_zero:
                        if other.is_extended_positive:
                            return AccumBounds(-oo, Mul(other, 1 / self.min))
                        if other.is_extended_negative:
                            return AccumBounds(Mul(other, 1 / self.min), oo)
                    return AccumBounds(-oo, oo)
                else:
                    return AccumBounds(Min(other / self.min, other / self.max),
                                       Max(other / self.min, other / self.max))
            return Mul(other, 1 / self, evaluate=False)
        else:
            return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __pow__(self, other):
        if isinstance(other, Expr):
            if other is S.Infinity:
                if self.min.is_extended_nonnegative:
                    if self.max < 1:
                        return S.Zero
                    if self.min > 1:
                        return S.Infinity
                    return AccumBounds(0, oo)
                elif self.max.is_extended_negative:
                    if self.min > -1:
                        return S.Zero
                    if self.max < -1:
                        return zoo
                    return S.NaN
                else:
                    if self.min > -1:
                        if self.max < 1:
                            return S.Zero
                        return AccumBounds(0, oo)
                    return AccumBounds(-oo, oo)

            if other is S.NegativeInfinity:
                return (1/self)**oo

            # generically true
            if (self.max - self.min).is_nonnegative:
                # well defined
                if self.min.is_nonnegative:
                    # no 0 to worry about
                    if other.is_nonnegative:
                        # no infinity to worry about
                        return self.func(self.min**other, self.max**other)

            if other.is_zero:
                return S.One  # x**0 = 1

            if other.is_Integer or other.is_integer:
                if self.min.is_extended_positive:
                    return AccumBounds(
                        Min(self.min**other, self.max**other),
                        Max(self.min**other, self.max**other))
                elif self.max.is_extended_negative:
                    return AccumBounds(
                        Min(self.max**other, self.min**other),
                        Max(self.max**other, self.min**other))

                if other % 2 == 0:
                    if other.is_extended_negative:
                        if self.min.is_zero:
                            return AccumBounds(self.max**other, oo)
                        if self.max.is_zero:
                            return AccumBounds(self.min**other, oo)
                        return (1/self)**(-other)
                    return AccumBounds(
                        S.Zero, Max(self.min**other, self.max**other))
                elif other % 2 == 1:
                    if other.is_extended_negative:
                        if self.min.is_zero:
                            return AccumBounds(self.max**other, oo)
                        if self.max.is_zero:
                            return AccumBounds(-oo, self.min**other)
                        return (1/self)**(-other)
                    return AccumBounds(self.min**other, self.max**other)

            # non-integer exponent
            # 0**neg or neg**frac yields complex
            if (other.is_number or other.is_rational) and (
                    self.min.is_extended_nonnegative or (
                    other.is_extended_nonnegative and
                    self.min.is_extended_nonnegative)):
                num, den = other.as_numer_denom()
                if num is S.One:
                    return AccumBounds(*[i**(1/den) for i in self.args])

                elif den is not S.One:  # e.g. if other is not Float
                    return (self**num)**(1/den)  # ok for non-negative base

            if isinstance(other, AccumBounds):
                if (self.min.is_extended_positive or
                        self.min.is_extended_nonnegative and
                        other.min.is_extended_nonnegative):
                    p = [self**i for i in other.args]
                    if not any(i.is_Pow for i in p):
                        a = [j for i in p for j in i.args or (i,)]
                        try:
                            return self.func(min(a), max(a))
                        except TypeError:  # can't sort
                            pass

            return Pow(self, other, evaluate=False)

        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rpow__(self, other):
        if other.is_real and other.is_extended_nonnegative and (
                self.max - self.min).is_extended_positive:
            if other is S.One:
                return S.One
            if other.is_extended_positive:
                a, b = [other**i for i in self.args]
                if min(a, b) != a:
                    a, b = b, a
                return self.func(a, b)
            if other.is_zero:
                if self.min.is_zero:
                    return self.func(0, 1)
                if self.min.is_extended_positive:
                    return S.Zero

        return Pow(other, self, evaluate=False)

    def __abs__(self):
        if self.max.is_extended_negative:
            return self.__neg__()
        elif self.min.is_extended_negative:
            return AccumBounds(S.Zero, Max(abs(self.min), self.max))
        else:
            return self


    def __contains__(self, other):
        """
        Returns ``True`` if other is contained in self, where other
        belongs to extended real numbers, ``False`` if not contained,
        otherwise TypeError is raised.

        Examples
        ========

        >>> from sympy import AccumBounds, oo
        >>> 1 in AccumBounds(-1, 3)
        True

        -oo and oo go together as limits (in AccumulationBounds).

        >>> -oo in AccumBounds(1, oo)
        True

        >>> oo in AccumBounds(-oo, 0)
        True

        """
        other = _sympify(other)

        if other in (S.Infinity, S.NegativeInfinity):
            if self.min is S.NegativeInfinity or self.max is S.Infinity:
                return True
            return False

        rv = And(self.min <= other, self.max >= other)
        if rv not in (True, False):
            raise TypeError("input failed to evaluate")
        return rv

    def intersection(self, other):
        """
        Returns the intersection of 'self' and 'other'.
        Here other can be an instance of :py:class:`~.FiniteSet` or AccumulationBounds.

        Parameters
        ==========

        other : AccumulationBounds
            Another AccumulationBounds object with which the intersection
            has to be computed.

        Returns
        =======

        AccumulationBounds
            Intersection of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy import AccumBounds, FiniteSet
        >>> AccumBounds(1, 3).intersection(AccumBounds(2, 4))
        AccumBounds(2, 3)

        >>> AccumBounds(1, 3).intersection(AccumBounds(4, 6))
        EmptySet

        >>> AccumBounds(1, 4).intersection(FiniteSet(1, 2, 5))
        {1, 2}

        """
        if not isinstance(other, (AccumBounds, FiniteSet)):
            raise TypeError(
                "Input must be AccumulationBounds or FiniteSet object")

        if isinstance(other, FiniteSet):
            fin_set = S.EmptySet
            for i in other:
                if i in self:
                    fin_set = fin_set + FiniteSet(i)
            return fin_set

        if self.max < other.min or self.min > other.max:
            return S.EmptySet

        if self.min <= other.min:
            if self.max <= other.max:
                return AccumBounds(other.min, self.max)
            if self.max > other.max:
                return other

        if other.min <= self.min:
            if other.max < self.max:
                return AccumBounds(self.min, other.max)
            if other.max > self.max:
                return self

    def union(self, other):
        # TODO : Devise a better method for Union of AccumBounds
        # this method is not actually correct and
        # can be made better
        if not isinstance(other, AccumBounds):
            raise TypeError(
                "Input must be AccumulationBounds or FiniteSet object")

        if self.min <= other.min and self.max >= other.min:
            return AccumBounds(self.min, Max(self.max, other.max))

        if other.min <= self.min and other.max >= self.min:
            return AccumBounds(other.min, Max(self.max, other.max))


@dispatch(AccumulationBounds, AccumulationBounds) # type: ignore # noqa:F811
def _eval_is_le(lhs, rhs): # noqa:F811
    if is_le(lhs.max, rhs.min):
        return True
    if is_gt(lhs.min, rhs.max):
        return False


@dispatch(AccumulationBounds, Basic) # type: ignore # noqa:F811
def _eval_is_le(lhs, rhs): # noqa: F811

    """
    Returns ``True `` if range of values attained by ``lhs`` AccumulationBounds
    object is greater than the range of values attained by ``rhs``,
    where ``rhs`` may be any value of type AccumulationBounds object or
    extended real number value, ``False`` if ``rhs`` satisfies
    the same property, else an unevaluated :py:class:`~.Relational`.

    Examples
    ========

    >>> from sympy import AccumBounds, oo
    >>> AccumBounds(1, 3) > AccumBounds(4, oo)
    False
    >>> AccumBounds(1, 4) > AccumBounds(3, 4)
    AccumBounds(1, 4) > AccumBounds(3, 4)
    >>> AccumBounds(1, oo) > -1
    True

    """
    if not rhs.is_extended_real:
            raise TypeError(
                "Invalid comparison of %s %s" %
                (type(rhs), rhs))
    elif rhs.is_comparable:
        if is_le(lhs.max, rhs):
            return True
        if is_gt(lhs.min, rhs):
            return False


@dispatch(AccumulationBounds, AccumulationBounds)
def _eval_is_ge(lhs, rhs): # noqa:F811
    if is_ge(lhs.min, rhs.max):
        return True
    if is_lt(lhs.max, rhs.min):
        return False


@dispatch(AccumulationBounds, Expr)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa: F811
    """
    Returns ``True`` if range of values attained by ``lhs`` AccumulationBounds
    object is less that the range of values attained by ``rhs``, where
    other may be any value of type AccumulationBounds object or extended
    real number value, ``False`` if ``rhs`` satisfies the same
    property, else an unevaluated :py:class:`~.Relational`.

    Examples
    ========

    >>> from sympy import AccumBounds, oo
    >>> AccumBounds(1, 3) >= AccumBounds(4, oo)
    False
    >>> AccumBounds(1, 4) >= AccumBounds(3, 4)
    AccumBounds(1, 4) >= AccumBounds(3, 4)
    >>> AccumBounds(1, oo) >= 1
    True
    """

    if not rhs.is_extended_real:
        raise TypeError(
            "Invalid comparison of %s %s" %
            (type(rhs), rhs))
    elif rhs.is_comparable:
        if is_ge(lhs.min, rhs):
            return True
        if is_lt(lhs.max, rhs):
            return False


@dispatch(Expr, AccumulationBounds)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811
    if not lhs.is_extended_real:
        raise TypeError(
            "Invalid comparison of %s %s" %
            (type(lhs), lhs))
    elif lhs.is_comparable:
        if is_le(rhs.max, lhs):
            return True
        if is_gt(rhs.min, lhs):
            return False


@dispatch(AccumulationBounds, AccumulationBounds)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811
    if is_ge(lhs.min, rhs.max):
        return True
    if is_lt(lhs.max, rhs.min):
        return False

# setting an alias for AccumulationBounds
AccumBounds = AccumulationBounds
