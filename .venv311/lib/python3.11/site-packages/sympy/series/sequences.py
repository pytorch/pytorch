from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import call_highest_priority
from sympy.core.parameters import global_parameters
from sympy.core.function import AppliedUndef, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S, Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.sympify import sympify
from sympy.matrices import Matrix
from sympy.polys import lcm, factor
from sympy.sets.sets import Interval, Intersection
from sympy.tensor.indexed import Idx
from sympy.utilities.iterables import flatten, is_sequence, iterable


###############################################################################
#                            SEQUENCES                                        #
###############################################################################


class SeqBase(Basic):
    """Base class for sequences"""

    is_commutative = True
    _op_priority = 15

    @staticmethod
    def _start_key(expr):
        """Return start (if possible) else S.Infinity.

        adapted from Set._infimum_key
        """
        try:
            start = expr.start
        except NotImplementedError:
            start = S.Infinity
        return start

    def _intersect_interval(self, other):
        """Returns start and stop.

        Takes intersection over the two intervals.
        """
        interval = Intersection(self.interval, other.interval)
        return interval.inf, interval.sup

    @property
    def gen(self):
        """Returns the generator for the sequence"""
        raise NotImplementedError("(%s).gen" % self)

    @property
    def interval(self):
        """The interval on which the sequence is defined"""
        raise NotImplementedError("(%s).interval" % self)

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        raise NotImplementedError("(%s).start" % self)

    @property
    def stop(self):
        """The ending point of the sequence. This point is included"""
        raise NotImplementedError("(%s).stop" % self)

    @property
    def length(self):
        """Length of the sequence"""
        raise NotImplementedError("(%s).length" % self)

    @property
    def variables(self):
        """Returns a tuple of variables that are bounded"""
        return ()

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n, m
        >>> SeqFormula(m*n**2, (n, 0, 5)).free_symbols
        {m}
        """
        return ({j for i in self.args for j in i.free_symbols
                   .difference(self.variables)})

    @cacheit
    def coeff(self, pt):
        """Returns the coefficient at point pt"""
        if pt < self.start or pt > self.stop:
            raise IndexError("Index %s out of bounds %s" % (pt, self.interval))
        return self._eval_coeff(pt)

    def _eval_coeff(self, pt):
        raise NotImplementedError("The _eval_coeff method should be added to"
                                  "%s to return coefficient so it is available"
                                  "when coeff calls it."
                                  % self.func)

    def _ith_point(self, i):
        """Returns the i'th point of a sequence.

        Explanation
        ===========

        If start point is negative infinity, point is returned from the end.
        Assumes the first point to be indexed zero.

        Examples
        =========

        >>> from sympy import oo
        >>> from sympy.series.sequences import SeqPer

        bounded

        >>> SeqPer((1, 2, 3), (-10, 10))._ith_point(0)
        -10
        >>> SeqPer((1, 2, 3), (-10, 10))._ith_point(5)
        -5

        End is at infinity

        >>> SeqPer((1, 2, 3), (0, oo))._ith_point(5)
        5

        Starts at negative infinity

        >>> SeqPer((1, 2, 3), (-oo, 0))._ith_point(5)
        -5
        """
        if self.start is S.NegativeInfinity:
            initial = self.stop
        else:
            initial = self.start

        if self.start is S.NegativeInfinity:
            step = -1
        else:
            step = 1

        return initial + i*step

    def _add(self, other):
        """
        Should only be used internally.

        Explanation
        ===========

        self._add(other) returns a new, term-wise added sequence if self
        knows how to add with other, otherwise it returns ``None``.

        ``other`` should only be a sequence object.

        Used within :class:`SeqAdd` class.
        """
        return None

    def _mul(self, other):
        """
        Should only be used internally.

        Explanation
        ===========

        self._mul(other) returns a new, term-wise multiplied sequence if self
        knows how to multiply with other, otherwise it returns ``None``.

        ``other`` should only be a sequence object.

        Used within :class:`SeqMul` class.
        """
        return None

    def coeff_mul(self, other):
        """
        Should be used when ``other`` is not a sequence. Should be
        defined to define custom behaviour.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2).coeff_mul(2)
        SeqFormula(2*n**2, (n, 0, oo))

        Notes
        =====

        '*' defines multiplication of sequences with sequences only.
        """
        return Mul(self, other)

    def __add__(self, other):
        """Returns the term-wise addition of 'self' and 'other'.

        ``other`` should be a sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) + SeqFormula(n**3)
        SeqFormula(n**3 + n**2, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot add sequence and %s' % type(other))
        return SeqAdd(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Returns the term-wise subtraction of ``self`` and ``other``.

        ``other`` should be a sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) - (SeqFormula(n))
        SeqFormula(n**2 - n, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot subtract sequence and %s' % type(other))
        return SeqAdd(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        """Negates the sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> -SeqFormula(n**2)
        SeqFormula(-n**2, (n, 0, oo))
        """
        return self.coeff_mul(-1)

    def __mul__(self, other):
        """Returns the term-wise multiplication of 'self' and 'other'.

        ``other`` should be a sequence. For ``other`` not being a
        sequence see :func:`coeff_mul` method.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) * (SeqFormula(n))
        SeqFormula(n**3, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot multiply sequence and %s' % type(other))
        return SeqMul(self, other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self * other

    def __iter__(self):
        for i in range(self.length):
            pt = self._ith_point(i)
            yield self.coeff(pt)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.coeff(index)
        elif isinstance(index, slice):
            start, stop = index.start, index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            return [self.coeff(self._ith_point(i)) for i in
                    range(start, stop, index.step or 1)]

    def find_linear_recurrence(self,n,d=None,gfvar=None):
        r"""
        Finds the shortest linear recurrence that satisfies the first n
        terms of sequence of order `\leq` ``n/2`` if possible.
        If ``d`` is specified, find shortest linear recurrence of order
        `\leq` min(d, n/2) if possible.
        Returns list of coefficients ``[b(1), b(2), ...]`` corresponding to the
        recurrence relation ``x(n) = b(1)*x(n-1) + b(2)*x(n-2) + ...``
        Returns ``[]`` if no recurrence is found.
        If gfvar is specified, also returns ordinary generating function as a
        function of gfvar.

        Examples
        ========

        >>> from sympy import sequence, sqrt, oo, lucas
        >>> from sympy.abc import n, x, y
        >>> sequence(n**2).find_linear_recurrence(10, 2)
        []
        >>> sequence(n**2).find_linear_recurrence(10)
        [3, -3, 1]
        >>> sequence(2**n).find_linear_recurrence(10)
        [2]
        >>> sequence(23*n**4+91*n**2).find_linear_recurrence(10)
        [5, -10, 10, -5, 1]
        >>> sequence(sqrt(5)*(((1 + sqrt(5))/2)**n - (-(1 + sqrt(5))/2)**(-n))/5).find_linear_recurrence(10)
        [1, 1]
        >>> sequence(x+y*(-2)**(-n), (n, 0, oo)).find_linear_recurrence(30)
        [1/2, 1/2]
        >>> sequence(3*5**n + 12).find_linear_recurrence(20,gfvar=x)
        ([6, -5], 3*(5 - 21*x)/((x - 1)*(5*x - 1)))
        >>> sequence(lucas(n)).find_linear_recurrence(15,gfvar=x)
        ([1, 1], (x - 2)/(x**2 + x - 1))
        """
        from sympy.simplify import simplify
        x = [simplify(expand(t)) for t in self[:n]]
        lx = len(x)
        if d is None:
            r = lx//2
        else:
            r = min(d,lx//2)
        coeffs = []
        for l in range(1, r+1):
            l2 = 2*l
            mlist = []
            for k in range(l):
                mlist.append(x[k:k+l])
            m = Matrix(mlist)
            if m.det() != 0:
                y = simplify(m.LUsolve(Matrix(x[l:l2])))
                if lx == l2:
                    coeffs = flatten(y[::-1])
                    break
                mlist = []
                for k in range(l,lx-l):
                    mlist.append(x[k:k+l])
                m = Matrix(mlist)
                if m*y == Matrix(x[l2:]):
                    coeffs = flatten(y[::-1])
                    break
        if gfvar is None:
            return coeffs
        else:
            l = len(coeffs)
            if l == 0:
                return [], None
            else:
                n, d = x[l-1]*gfvar**(l-1), 1 - coeffs[l-1]*gfvar**l
                for i in range(l-1):
                    n += x[i]*gfvar**i
                    for j in range(l-i-1):
                        n -= coeffs[i]*x[j]*gfvar**(i+j+1)
                    d -= coeffs[i]*gfvar**(i+1)
                return coeffs, simplify(factor(n)/factor(d))

class EmptySequence(SeqBase, metaclass=Singleton):
    """Represents an empty sequence.

    The empty sequence is also available as a singleton as
    ``S.EmptySequence``.

    Examples
    ========

    >>> from sympy import EmptySequence, SeqPer
    >>> from sympy.abc import x
    >>> EmptySequence
    EmptySequence
    >>> SeqPer((1, 2), (x, 0, 10)) + EmptySequence
    SeqPer((1, 2), (x, 0, 10))
    >>> SeqPer((1, 2)) * EmptySequence
    EmptySequence
    >>> EmptySequence.coeff_mul(-1)
    EmptySequence
    """

    @property
    def interval(self):
        return S.EmptySet

    @property
    def length(self):
        return S.Zero

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        return self

    def __iter__(self):
        return iter([])


class SeqExpr(SeqBase):
    """Sequence expression class.

    Various sequences should inherit from this class.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExpr
    >>> from sympy.abc import x
    >>> from sympy import Tuple
    >>> s = SeqExpr(Tuple(1, 2, 3), Tuple(x, 0, 10))
    >>> s.gen
    (1, 2, 3)
    >>> s.interval
    Interval(0, 10)
    >>> s.length
    11

    See Also
    ========

    sympy.series.sequences.SeqPer
    sympy.series.sequences.SeqFormula
    """

    @property
    def gen(self):
        return self.args[0]

    @property
    def interval(self):
        return Interval(self.args[1][1], self.args[1][2])

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return self.stop - self.start + 1

    @property
    def variables(self):
        return (self.args[1][0],)


class SeqPer(SeqExpr):
    """
    Represents a periodic sequence.

    The elements are repeated after a given period.

    Examples
    ========

    >>> from sympy import SeqPer, oo
    >>> from sympy.abc import k

    >>> s = SeqPer((1, 2, 3), (0, 5))
    >>> s.periodical
    (1, 2, 3)
    >>> s.period
    3

    For value at a particular point

    >>> s.coeff(3)
    1

    supports slicing

    >>> s[:]
    [1, 2, 3, 1, 2, 3]

    iterable

    >>> list(s)
    [1, 2, 3, 1, 2, 3]

    sequence starts from negative infinity

    >>> SeqPer((1, 2, 3), (-oo, 0))[0:6]
    [1, 2, 3, 1, 2, 3]

    Periodic formulas

    >>> SeqPer((k, k**2, k**3), (k, 0, oo))[0:6]
    [0, 1, 8, 3, 16, 125]

    See Also
    ========

    sympy.series.sequences.SeqFormula
    """

    def __new__(cls, periodical, limits=None):
        periodical = sympify(periodical)

        def _find_x(periodical):
            free = periodical.free_symbols
            if len(periodical.free_symbols) == 1:
                return free.pop()
            else:
                return Dummy('k')

        x, start, stop = None, None, None
        if limits is None:
            x, start, stop = _find_x(periodical), 0, S.Infinity
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(periodical)
                start, stop = limits

        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))

        if start is S.NegativeInfinity and stop is S.Infinity:
                raise ValueError("Both the start and end value"
                                 "cannot be unbounded")

        limits = sympify((x, start, stop))

        if is_sequence(periodical, Tuple):
            periodical = sympify(tuple(flatten(periodical)))
        else:
            raise ValueError("invalid period %s should be something "
                             "like e.g (1, 2) " % periodical)

        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence

        return Basic.__new__(cls, periodical, limits)

    @property
    def period(self):
        return len(self.gen)

    @property
    def periodical(self):
        return self.gen

    def _eval_coeff(self, pt):
        if self.start is S.NegativeInfinity:
            idx = (self.stop - pt) % self.period
        else:
            idx = (pt - self.start) % self.period
        return self.periodical[idx].subs(self.variables[0], pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        if isinstance(other, SeqPer):
            per1, lper1 = self.periodical, self.period
            per2, lper2 = other.periodical, other.period

            per_length = lcm(lper1, lper2)

            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 + ele2)

            start, stop = self._intersect_interval(other)
            return SeqPer(new_per, (self.variables[0], start, stop))

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        if isinstance(other, SeqPer):
            per1, lper1 = self.periodical, self.period
            per2, lper2 = other.periodical, other.period

            per_length = lcm(lper1, lper2)

            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 * ele2)

            start, stop = self._intersect_interval(other)
            return SeqPer(new_per, (self.variables[0], start, stop))

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        coeff = sympify(coeff)
        per = [x * coeff for x in self.periodical]
        return SeqPer(per, self.args[1])


class SeqFormula(SeqExpr):
    """
    Represents sequence based on a formula.

    Elements are generated using a formula.

    Examples
    ========

    >>> from sympy import SeqFormula, oo, Symbol
    >>> n = Symbol('n')
    >>> s = SeqFormula(n**2, (n, 0, 5))
    >>> s.formula
    n**2

    For value at a particular point

    >>> s.coeff(3)
    9

    supports slicing

    >>> s[:]
    [0, 1, 4, 9, 16, 25]

    iterable

    >>> list(s)
    [0, 1, 4, 9, 16, 25]

    sequence starts from negative infinity

    >>> SeqFormula(n**2, (-oo, 0))[0:6]
    [0, 1, 4, 9, 16, 25]

    See Also
    ========

    sympy.series.sequences.SeqPer
    """

    def __new__(cls, formula, limits=None):
        formula = sympify(formula)

        def _find_x(formula):
            free = formula.free_symbols
            if len(free) == 1:
                return free.pop()
            elif not free:
                return Dummy('k')
            else:
                raise ValueError(
                    " specify dummy variables for %s. If the formula contains"
                    " more than one free symbol, a dummy variable should be"
                    " supplied explicitly e.g., SeqFormula(m*n**2, (n, 0, 5))"
                    % formula)

        x, start, stop = None, None, None
        if limits is None:
            x, start, stop = _find_x(formula), 0, S.Infinity
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(formula)
                start, stop = limits

        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))

        if start is S.NegativeInfinity and stop is S.Infinity:
                raise ValueError("Both the start and end value "
                                 "cannot be unbounded")
        limits = sympify((x, start, stop))

        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence

        return Basic.__new__(cls, formula, limits)

    @property
    def formula(self):
        return self.gen

    def _eval_coeff(self, pt):
        d = self.variables[0]
        return self.formula.subs(d, pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        if isinstance(other, SeqFormula):
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            formula = form1 + form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        if isinstance(other, SeqFormula):
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            formula = form1 * form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        coeff = sympify(coeff)
        formula = self.formula * coeff
        return SeqFormula(formula, self.args[1])

    def expand(self, *args, **kwargs):
        return SeqFormula(expand(self.formula, *args, **kwargs), self.args[1])

class RecursiveSeq(SeqBase):
    """
    A finite degree recursive sequence.

    Explanation
    ===========

    That is, a sequence a(n) that depends on a fixed, finite number of its
    previous values. The general form is

        a(n) = f(a(n - 1), a(n - 2), ..., a(n - d))

    for some fixed, positive integer d, where f is some function defined by a
    SymPy expression.

    Parameters
    ==========

    recurrence : SymPy expression defining recurrence
        This is *not* an equality, only the expression that the nth term is
        equal to. For example, if :code:`a(n) = f(a(n - 1), ..., a(n - d))`,
        then the expression should be :code:`f(a(n - 1), ..., a(n - d))`.

    yn : applied undefined function
        Represents the nth term of the sequence as e.g. :code:`y(n)` where
        :code:`y` is an undefined function and `n` is the sequence index.

    n : symbolic argument
        The name of the variable that the recurrence is in, e.g., :code:`n` if
        the recurrence function is :code:`y(n)`.

    initial : iterable with length equal to the degree of the recurrence
        The initial values of the recurrence.

    start : start value of sequence (inclusive)

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy.series.sequences import RecursiveSeq
    >>> y = Function("y")
    >>> n = symbols("n")
    >>> fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])

    >>> fib.coeff(3) # Value at a particular point
    2

    >>> fib[:6] # supports slicing
    [0, 1, 1, 2, 3, 5]

    >>> fib.recurrence # inspect recurrence
    Eq(y(n), y(n - 2) + y(n - 1))

    >>> fib.degree # automatically determine degree
    2

    >>> for x in zip(range(10), fib): # supports iteration
    ...     print(x)
    (0, 0)
    (1, 1)
    (2, 1)
    (3, 2)
    (4, 3)
    (5, 5)
    (6, 8)
    (7, 13)
    (8, 21)
    (9, 34)

    See Also
    ========

    sympy.series.sequences.SeqFormula

    """

    def __new__(cls, recurrence, yn, n, initial=None, start=0):
        if not isinstance(yn, AppliedUndef):
            raise TypeError("recurrence sequence must be an applied undefined function"
                            ", found `{}`".format(yn))

        if not isinstance(n, Basic) or not n.is_symbol:
            raise TypeError("recurrence variable must be a symbol"
                            ", found `{}`".format(n))

        if yn.args != (n,):
            raise TypeError("recurrence sequence does not match symbol")

        y = yn.func

        k = Wild("k", exclude=(n,))
        degree = 0

        # Find all applications of y in the recurrence and check that:
        #   1. The function y is only being used with a single argument; and
        #   2. All arguments are n + k for constant negative integers k.

        prev_ys = recurrence.find(y)
        for prev_y in prev_ys:
            if len(prev_y.args) != 1:
                raise TypeError("Recurrence should be in a single variable")

            shift = prev_y.args[0].match(n + k)[k]
            if not (shift.is_constant() and shift.is_integer and shift < 0):
                raise TypeError("Recurrence should have constant,"
                                " negative, integer shifts"
                                " (found {})".format(prev_y))

            if -shift > degree:
                degree = -shift

        if not initial:
            initial = [Dummy("c_{}".format(k)) for k in range(degree)]

        if len(initial) != degree:
            raise ValueError("Number of initial terms must equal degree")

        degree = Integer(degree)
        start = sympify(start)

        initial = Tuple(*(sympify(x) for x in initial))

        seq = Basic.__new__(cls, recurrence, yn, n, initial, start)

        seq.cache = {y(start + k): init for k, init in enumerate(initial)}
        seq.degree = degree

        return seq

    @property
    def _recurrence(self):
        """Equation defining recurrence."""
        return self.args[0]

    @property
    def recurrence(self):
        """Equation defining recurrence."""
        return Eq(self.yn, self.args[0])

    @property
    def yn(self):
        """Applied function representing the nth term"""
        return self.args[1]

    @property
    def y(self):
        """Undefined function for the nth term of the sequence"""
        return self.yn.func

    @property
    def n(self):
        """Sequence index symbol"""
        return self.args[2]

    @property
    def initial(self):
        """The initial values of the sequence"""
        return self.args[3]

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        return self.args[4]

    @property
    def stop(self):
        """The ending point of the sequence. (oo)"""
        return S.Infinity

    @property
    def interval(self):
        """Interval on which sequence is defined."""
        return (self.start, S.Infinity)

    def _eval_coeff(self, index):
        if index - self.start < len(self.cache):
            return self.cache[self.y(index)]

        for current in range(len(self.cache), index + 1):
            # Use xreplace over subs for performance.
            # See issue #10697.
            seq_index = self.start + current
            current_recurrence = self._recurrence.xreplace({self.n: seq_index})
            new_term = current_recurrence.xreplace(self.cache)

            self.cache[self.y(seq_index)] = new_term

        return self.cache[self.y(self.start + current)]

    def __iter__(self):
        index = self.start
        while True:
            yield self._eval_coeff(index)
            index += 1


def sequence(seq, limits=None):
    """
    Returns appropriate sequence object.

    Explanation
    ===========

    If ``seq`` is a SymPy sequence, returns :class:`SeqPer` object
    otherwise returns :class:`SeqFormula` object.

    Examples
    ========

    >>> from sympy import sequence
    >>> from sympy.abc import n
    >>> sequence(n**2, (n, 0, 5))
    SeqFormula(n**2, (n, 0, 5))
    >>> sequence((1, 2, 3), (n, 0, 5))
    SeqPer((1, 2, 3), (n, 0, 5))

    See Also
    ========

    sympy.series.sequences.SeqPer
    sympy.series.sequences.SeqFormula
    """
    seq = sympify(seq)

    if is_sequence(seq, Tuple):
        return SeqPer(seq, limits)
    else:
        return SeqFormula(seq, limits)


###############################################################################
#                            OPERATIONS                                       #
###############################################################################


class SeqExprOp(SeqBase):
    """
    Base class for operations on sequences.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExprOp, sequence
    >>> from sympy.abc import n
    >>> s1 = sequence(n**2, (n, 0, 10))
    >>> s2 = sequence((1, 2, 3), (n, 5, 10))
    >>> s = SeqExprOp(s1, s2)
    >>> s.gen
    (n**2, (1, 2, 3))
    >>> s.interval
    Interval(5, 10)
    >>> s.length
    6

    See Also
    ========

    sympy.series.sequences.SeqAdd
    sympy.series.sequences.SeqMul
    """
    @property
    def gen(self):
        """Generator for the sequence.

        returns a tuple of generators of all the argument sequences.
        """
        return tuple(a.gen for a in self.args)

    @property
    def interval(self):
        """Sequence is defined on the intersection
        of all the intervals of respective sequences
        """
        return Intersection(*(a.interval for a in self.args))

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def variables(self):
        """Cumulative of all the bound variables"""
        return tuple(flatten([a.variables for a in self.args]))

    @property
    def length(self):
        return self.stop - self.start + 1


class SeqAdd(SeqExprOp):
    """Represents term-wise addition of sequences.

    Rules:
        * The interval on which sequence is defined is the intersection
          of respective intervals of sequences.
        * Anything + :class:`EmptySequence` remains unchanged.
        * Other rules are defined in ``_add`` methods of sequence classes.

    Examples
    ========

    >>> from sympy import EmptySequence, oo, SeqAdd, SeqPer, SeqFormula
    >>> from sympy.abc import n
    >>> SeqAdd(SeqPer((1, 2), (n, 0, oo)), EmptySequence)
    SeqPer((1, 2), (n, 0, oo))
    >>> SeqAdd(SeqPer((1, 2), (n, 0, 5)), SeqPer((1, 2), (n, 6, 10)))
    EmptySequence
    >>> SeqAdd(SeqPer((1, 2), (n, 0, oo)), SeqFormula(n**2, (n, 0, oo)))
    SeqAdd(SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo)))
    >>> SeqAdd(SeqFormula(n**3), SeqFormula(n**2))
    SeqFormula(n**3 + n**2, (n, 0, oo))

    See Also
    ========

    sympy.series.sequences.SeqMul
    """

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # flatten inputs
        args = list(args)

        # adapted from sympy.sets.sets.Union
        def _flatten(arg):
            if isinstance(arg, SeqBase):
                if isinstance(arg, SeqAdd):
                    return sum(map(_flatten, arg.args), [])
                else:
                    return [arg]
            if iterable(arg):
                return sum(map(_flatten, arg), [])
            raise TypeError("Input must be Sequences or "
                            " iterables of Sequences")
        args = _flatten(args)

        args = [a for a in args if a is not S.EmptySequence]

        # Addition of no sequences is EmptySequence
        if not args:
            return S.EmptySequence

        if Intersection(*(a.interval for a in args)) is S.EmptySet:
            return S.EmptySequence

        # reduce using known rules
        if evaluate:
            return SeqAdd.reduce(args)

        args = list(ordered(args, SeqBase._start_key))

        return Basic.__new__(cls, *args)

    @staticmethod
    def reduce(args):
        """Simplify :class:`SeqAdd` using known rules.

        Iterates through all pairs and ask the constituent
        sequences if they can simplify themselves with any other constituent.

        Notes
        =====

        adapted from ``Union.reduce``

        """
        new_args = True
        while new_args:
            for id1, s in enumerate(args):
                new_args = False
                for id2, t in enumerate(args):
                    if id1 == id2:
                        continue
                    new_seq = s._add(t)
                    # This returns None if s does not know how to add
                    # with t. Returns the newly added sequence otherwise
                    if new_seq is not None:
                        new_args = [a for a in args if a not in (s, t)]
                        new_args.append(new_seq)
                        break
                if new_args:
                    args = new_args
                    break

        if len(args) == 1:
            return args.pop()
        else:
            return SeqAdd(args, evaluate=False)

    def _eval_coeff(self, pt):
        """adds up the coefficients of all the sequences at point pt"""
        return sum(a.coeff(pt) for a in self.args)


class SeqMul(SeqExprOp):
    r"""Represents term-wise multiplication of sequences.

    Explanation
    ===========

    Handles multiplication of sequences only. For multiplication
    with other objects see :func:`SeqBase.coeff_mul`.

    Rules:
        * The interval on which sequence is defined is the intersection
          of respective intervals of sequences.
        * Anything \* :class:`EmptySequence` returns :class:`EmptySequence`.
        * Other rules are defined in ``_mul`` methods of sequence classes.

    Examples
    ========

    >>> from sympy import EmptySequence, oo, SeqMul, SeqPer, SeqFormula
    >>> from sympy.abc import n
    >>> SeqMul(SeqPer((1, 2), (n, 0, oo)), EmptySequence)
    EmptySequence
    >>> SeqMul(SeqPer((1, 2), (n, 0, 5)), SeqPer((1, 2), (n, 6, 10)))
    EmptySequence
    >>> SeqMul(SeqPer((1, 2), (n, 0, oo)), SeqFormula(n**2))
    SeqMul(SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo)))
    >>> SeqMul(SeqFormula(n**3), SeqFormula(n**2))
    SeqFormula(n**5, (n, 0, oo))

    See Also
    ========

    sympy.series.sequences.SeqAdd
    """

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # flatten inputs
        args = list(args)

        # adapted from sympy.sets.sets.Union
        def _flatten(arg):
            if isinstance(arg, SeqBase):
                if isinstance(arg, SeqMul):
                    return sum(map(_flatten, arg.args), [])
                else:
                    return [arg]
            elif iterable(arg):
                return sum(map(_flatten, arg), [])
            raise TypeError("Input must be Sequences or "
                            " iterables of Sequences")
        args = _flatten(args)

        # Multiplication of no sequences is EmptySequence
        if not args:
            return S.EmptySequence

        if Intersection(*(a.interval for a in args)) is S.EmptySet:
            return S.EmptySequence

        # reduce using known rules
        if evaluate:
            return SeqMul.reduce(args)

        args = list(ordered(args, SeqBase._start_key))

        return Basic.__new__(cls, *args)

    @staticmethod
    def reduce(args):
        """Simplify a :class:`SeqMul` using known rules.

        Explanation
        ===========

        Iterates through all pairs and ask the constituent
        sequences if they can simplify themselves with any other constituent.

        Notes
        =====

        adapted from ``Union.reduce``

        """
        new_args = True
        while new_args:
            for id1, s in enumerate(args):
                new_args = False
                for id2, t in enumerate(args):
                    if id1 == id2:
                        continue
                    new_seq = s._mul(t)
                    # This returns None if s does not know how to multiply
                    # with t. Returns the newly multiplied sequence otherwise
                    if new_seq is not None:
                        new_args = [a for a in args if a not in (s, t)]
                        new_args.append(new_seq)
                        break
                if new_args:
                    args = new_args
                    break

        if len(args) == 1:
            return args.pop()
        else:
            return SeqMul(args, evaluate=False)

    def _eval_coeff(self, pt):
        """multiplies the coefficients of all the sequences at point pt"""
        val = 1
        for a in self.args:
            val *= a.coeff(pt)
        return val
