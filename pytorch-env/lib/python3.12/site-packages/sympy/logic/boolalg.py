"""
Boolean algebra module for SymPy
"""

from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent


def as_Boolean(e):
    """Like ``bool``, return the Boolean value of an expression, e,
    which can be any instance of :py:class:`~.Boolean` or ``bool``.

    Examples
    ========

    >>> from sympy import true, false, nan
    >>> from sympy.logic.boolalg import as_Boolean
    >>> from sympy.abc import x
    >>> as_Boolean(0) is false
    True
    >>> as_Boolean(1) is true
    True
    >>> as_Boolean(x)
    x
    >>> as_Boolean(2)
    Traceback (most recent call last):
    ...
    TypeError: expecting bool or Boolean, not `2`.
    >>> as_Boolean(nan)
    Traceback (most recent call last):
    ...
    TypeError: expecting bool or Boolean, not `nan`.

    """
    from sympy.core.symbol import Symbol
    if e == True:
        return true
    if e == False:
        return false
    if isinstance(e, Symbol):
        z = e.is_zero
        if z is None:
            return e
        return false if z else true
    if isinstance(e, Boolean):
        return e
    raise TypeError('expecting bool or Boolean, not `%s`.' % e)


@sympify_method_args
class Boolean(Basic):
    """A Boolean object is an object for which logic operations make sense."""

    __slots__ = ()

    kind = BooleanKind

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __and__(self, other):
        return And(self, other)

    __rand__ = __and__

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __or__(self, other):
        return Or(self, other)

    __ror__ = __or__

    def __invert__(self):
        """Overloading for ~"""
        return Not(self)

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __rshift__(self, other):
        return Implies(self, other)

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __lshift__(self, other):
        return Implies(other, self)

    __rrshift__ = __lshift__
    __rlshift__ = __rshift__

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __xor__(self, other):
        return Xor(self, other)

    __rxor__ = __xor__

    def equals(self, other):
        """
        Returns ``True`` if the given formulas have the same truth table.
        For two formulas to be equal they must have the same literals.

        Examples
        ========

        >>> from sympy.abc import A, B, C
        >>> from sympy import And, Or, Not
        >>> (A >> B).equals(~B >> ~A)
        True
        >>> Not(And(A, B, C)).equals(And(Not(A), Not(B), Not(C)))
        False
        >>> Not(And(A, Not(A))).equals(Or(B, Not(B)))
        False

        """
        from sympy.logic.inference import satisfiable
        from sympy.core.relational import Relational

        if self.has(Relational) or other.has(Relational):
            raise NotImplementedError('handling of relationals')
        return self.atoms() == other.atoms() and \
            not satisfiable(Not(Equivalent(self, other)))

    def to_nnf(self, simplify=True):
        # override where necessary
        return self

    def as_set(self):
        """
        Rewrites Boolean expression in terms of real sets.

        Examples
        ========

        >>> from sympy import Symbol, Eq, Or, And
        >>> x = Symbol('x', real=True)
        >>> Eq(x, 0).as_set()
        {0}
        >>> (x > 0).as_set()
        Interval.open(0, oo)
        >>> And(-2 < x, x < 2).as_set()
        Interval.open(-2, 2)
        >>> Or(x < -2, 2 < x).as_set()
        Union(Interval.open(-oo, -2), Interval.open(2, oo))

        """
        from sympy.calculus.util import periodicity
        from sympy.core.relational import Relational

        free = self.free_symbols
        if len(free) == 1:
            x = free.pop()
            if x.kind is NumberKind:
                reps = {}
                for r in self.atoms(Relational):
                    if periodicity(r, x) not in (0, None):
                        s = r._eval_as_set()
                        if s in (S.EmptySet, S.UniversalSet, S.Reals):
                            reps[r] = s.as_relational(x)
                            continue
                        raise NotImplementedError(filldedent('''
                            as_set is not implemented for relationals
                            with periodic solutions
                            '''))
                new = self.subs(reps)
                if new.func != self.func:
                    return new.as_set()  # restart with new obj
                else:
                    return new._eval_as_set()

            return self._eval_as_set()
        else:
            raise NotImplementedError("Sorry, as_set has not yet been"
                                      " implemented for multivariate"
                                      " expressions")

    @property
    def binary_symbols(self):
        from sympy.core.relational import Eq, Ne
        return set().union(*[i.binary_symbols for i in self.args
                           if i.is_Boolean or i.is_Symbol
                           or isinstance(i, (Eq, Ne))])

    def _eval_refine(self, assumptions):
        from sympy.assumptions import ask
        ret = ask(self, assumptions)
        if ret is True:
            return true
        elif ret is False:
            return false
        return None


class BooleanAtom(Boolean):
    """
    Base class of :py:class:`~.BooleanTrue` and :py:class:`~.BooleanFalse`.
    """
    is_Boolean = True
    is_Atom = True
    _op_priority = 11  # higher than Expr

    def simplify(self, *a, **kw):
        return self

    def expand(self, *a, **kw):
        return self

    @property
    def canonical(self):
        return self

    def _noop(self, other=None):
        raise TypeError('BooleanAtom not allowed in this context.')

    __add__ = _noop
    __radd__ = _noop
    __sub__ = _noop
    __rsub__ = _noop
    __mul__ = _noop
    __rmul__ = _noop
    __pow__ = _noop
    __rpow__ = _noop
    __truediv__ = _noop
    __rtruediv__ = _noop
    __mod__ = _noop
    __rmod__ = _noop
    _eval_power = _noop

    # /// drop when Py2 is no longer supported
    def __lt__(self, other):
        raise TypeError(filldedent('''
            A Boolean argument can only be used in
            Eq and Ne; all other relationals expect
            real expressions.
        '''))

    __le__ = __lt__
    __gt__ = __lt__
    __ge__ = __lt__
    # \\\

    def _eval_simplify(self, **kwargs):
        return self


class BooleanTrue(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of ``True``, a singleton that can be accessed via ``S.true``.

    This is the SymPy version of ``True``, for use in the logic module. The
    primary advantage of using ``true`` instead of ``True`` is that shorthand Boolean
    operations like ``~`` and ``>>`` will work as expected on this class, whereas with
    True they act bitwise on 1. Functions in the logic module will return this
    class when they evaluate to true.

    Notes
    =====

    There is liable to be some confusion as to when ``True`` should
    be used and when ``S.true`` should be used in various contexts
    throughout SymPy. An important thing to remember is that
    ``sympify(True)`` returns ``S.true``. This means that for the most
    part, you can just use ``True`` and it will automatically be converted
    to ``S.true`` when necessary, similar to how you can generally use 1
    instead of ``S.One``.

    The rule of thumb is:

    "If the boolean in question can be replaced by an arbitrary symbolic
    ``Boolean``, like ``Or(x, y)`` or ``x > 1``, use ``S.true``.
    Otherwise, use ``True``"

    In other words, use ``S.true`` only on those contexts where the
    boolean is being used as a symbolic representation of truth.
    For example, if the object ends up in the ``.args`` of any expression,
    then it must necessarily be ``S.true`` instead of ``True``, as
    elements of ``.args`` must be ``Basic``. On the other hand,
    ``==`` is not a symbolic operation in SymPy, since it always returns
    ``True`` or ``False``, and does so in terms of structural equality
    rather than mathematical, so it should return ``True``. The assumptions
    system should use ``True`` and ``False``. Aside from not satisfying
    the above rule of thumb, the assumptions system uses a three-valued logic
    (``True``, ``False``, ``None``), whereas ``S.true`` and ``S.false``
    represent a two-valued logic. When in doubt, use ``True``.

    "``S.true == True is True``."

    While "``S.true is True``" is ``False``, "``S.true == True``"
    is ``True``, so if there is any doubt over whether a function or
    expression will return ``S.true`` or ``True``, just use ``==``
    instead of ``is`` to do the comparison, and it will work in either
    case.  Finally, for boolean flags, it's better to just use ``if x``
    instead of ``if x is True``. To quote PEP 8:

    Do not compare boolean values to ``True`` or ``False``
    using ``==``.

    * Yes:   ``if greeting:``
    * No:    ``if greeting == True:``
    * Worse: ``if greeting is True:``

    Examples
    ========

    >>> from sympy import sympify, true, false, Or
    >>> sympify(True)
    True
    >>> _ is True, _ is true
    (False, True)

    >>> Or(true, false)
    True
    >>> _ is true
    True

    Python operators give a boolean result for true but a
    bitwise result for True

    >>> ~true, ~True  # doctest: +SKIP
    (False, -2)
    >>> true >> true, True >> True
    (True, 0)

    See Also
    ========

    sympy.logic.boolalg.BooleanFalse

    """
    def __bool__(self):
        return True

    def __hash__(self):
        return hash(True)

    def __eq__(self, other):
        if other is True:
            return True
        if other is False:
            return False
        return super().__eq__(other)

    @property
    def negated(self):
        return false

    def as_set(self):
        """
        Rewrite logic operators and relationals in terms of real sets.

        Examples
        ========

        >>> from sympy import true
        >>> true.as_set()
        UniversalSet

        """
        return S.UniversalSet


class BooleanFalse(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of ``False``, a singleton that can be accessed via ``S.false``.

    This is the SymPy version of ``False``, for use in the logic module. The
    primary advantage of using ``false`` instead of ``False`` is that shorthand
    Boolean operations like ``~`` and ``>>`` will work as expected on this class,
    whereas with ``False`` they act bitwise on 0. Functions in the logic module
    will return this class when they evaluate to false.

    Notes
    ======

    See the notes section in :py:class:`sympy.logic.boolalg.BooleanTrue`

    Examples
    ========

    >>> from sympy import sympify, true, false, Or
    >>> sympify(False)
    False
    >>> _ is False, _ is false
    (False, True)

    >>> Or(true, false)
    True
    >>> _ is true
    True

    Python operators give a boolean result for false but a
    bitwise result for False

    >>> ~false, ~False  # doctest: +SKIP
    (True, -1)
    >>> false >> false, False >> False
    (True, 0)

    See Also
    ========

    sympy.logic.boolalg.BooleanTrue

    """
    def __bool__(self):
        return False

    def __hash__(self):
        return hash(False)

    def __eq__(self, other):
        if other is True:
            return False
        if other is False:
            return True
        return super().__eq__(other)

    @property
    def negated(self):
        return true

    def as_set(self):
        """
        Rewrite logic operators and relationals in terms of real sets.

        Examples
        ========

        >>> from sympy import false
        >>> false.as_set()
        EmptySet
        """
        return S.EmptySet


true = BooleanTrue()
false = BooleanFalse()
# We want S.true and S.false to work, rather than S.BooleanTrue and
# S.BooleanFalse, but making the class and instance names the same causes some
# major issues (like the inability to import the class directly from this
# file).
S.true = true
S.false = false

_sympy_converter[bool] = lambda x: true if x else false


class BooleanFunction(Application, Boolean):
    """Boolean function is a function that lives in a boolean space
    It is used as base class for :py:class:`~.And`, :py:class:`~.Or`,
    :py:class:`~.Not`, etc.
    """
    is_Boolean = True

    def _eval_simplify(self, **kwargs):
        rv = simplify_univariate(self)
        if not isinstance(rv, BooleanFunction):
            return rv.simplify(**kwargs)
        rv = rv.func(*[a.simplify(**kwargs) for a in rv.args])
        return simplify_logic(rv)

    def simplify(self, **kwargs):
        from sympy.simplify.simplify import simplify
        return simplify(self, **kwargs)

    def __lt__(self, other):
        raise TypeError(filldedent('''
            A Boolean argument can only be used in
            Eq and Ne; all other relationals expect
            real expressions.
        '''))
    __le__ = __lt__
    __ge__ = __lt__
    __gt__ = __lt__

    @classmethod
    def binary_check_and_simplify(self, *args):
        return [as_Boolean(i) for i in args]

    def to_nnf(self, simplify=True):
        return self._to_nnf(*self.args, simplify=simplify)

    def to_anf(self, deep=True):
        return self._to_anf(*self.args, deep=deep)

    @classmethod
    def _to_nnf(cls, *args, **kwargs):
        simplify = kwargs.get('simplify', True)
        argset = set()
        for arg in args:
            if not is_literal(arg):
                arg = arg.to_nnf(simplify)
            if simplify:
                if isinstance(arg, cls):
                    arg = arg.args
                else:
                    arg = (arg,)
                for a in arg:
                    if Not(a) in argset:
                        return cls.zero
                    argset.add(a)
            else:
                argset.add(arg)
        return cls(*argset)

    @classmethod
    def _to_anf(cls, *args, **kwargs):
        deep = kwargs.get('deep', True)
        new_args = []
        for arg in args:
            if deep:
                if not is_literal(arg) or isinstance(arg, Not):
                    arg = arg.to_anf(deep=deep)
            new_args.append(arg)
        return cls(*new_args, remove_true=False)

    # the diff method below is copied from Expr class
    def diff(self, *symbols, **assumptions):
        assumptions.setdefault("evaluate", True)
        return Derivative(self, *symbols, **assumptions)

    def _eval_derivative(self, x):
        if x in self.binary_symbols:
            from sympy.core.relational import Eq
            from sympy.functions.elementary.piecewise import Piecewise
            return Piecewise(
                (0, Eq(self.subs(x, 0), self.subs(x, 1))),
                (1, True))
        elif x in self.free_symbols:
            # not implemented, see https://www.encyclopediaofmath.org/
            # index.php/Boolean_differential_calculus
            pass
        else:
            return S.Zero


class And(LatticeOp, BooleanFunction):
    """
    Logical AND function.

    It evaluates its arguments in order, returning false immediately
    when an argument is false and true if they are all true.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import And
    >>> x & y
    x & y

    Notes
    =====

    The ``&`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise
    and. Hence, ``And(a, b)`` and ``a & b`` will produce different results if
    ``a`` and ``b`` are integers.

    >>> And(x, y).subs(x, 1)
    y

    """
    zero = false
    identity = true

    nargs = None

    @classmethod
    def _new_args_filter(cls, args):
        args = BooleanFunction.binary_check_and_simplify(*args)
        args = LatticeOp._new_args_filter(args, And)
        newargs = []
        rel = set()
        for x in ordered(args):
            if x.is_Relational:
                c = x.canonical
                if c in rel:
                    continue
                elif c.negated.canonical in rel:
                    return [false]
                else:
                    rel.add(c)
            newargs.append(x)
        return newargs

    def _eval_subs(self, old, new):
        args = []
        bad = None
        for i in self.args:
            try:
                i = i.subs(old, new)
            except TypeError:
                # store TypeError
                if bad is None:
                    bad = i
                continue
            if i == False:
                return false
            elif i != True:
                args.append(i)
        if bad is not None:
            # let it raise
            bad.subs(old, new)
        # If old is And, replace the parts of the arguments with new if all
        # are there
        if isinstance(old, And):
            old_set = set(old.args)
            if old_set.issubset(args):
                args = set(args) - old_set
                args.add(new)

        return self.func(*args)

    def _eval_simplify(self, **kwargs):
        from sympy.core.relational import Equality, Relational
        from sympy.solvers.solveset import linear_coeffs
        # standard simplify
        rv = super()._eval_simplify(**kwargs)
        if not isinstance(rv, And):
            return rv

        # simplify args that are equalities involving
        # symbols so x == 0 & x == y -> x==0 & y == 0
        Rel, nonRel = sift(rv.args, lambda i: isinstance(i, Relational),
                           binary=True)
        if not Rel:
            return rv
        eqs, other = sift(Rel, lambda i: isinstance(i, Equality), binary=True)

        measure = kwargs['measure']
        if eqs:
            ratio = kwargs['ratio']
            reps = {}
            sifted = {}
            # group by length of free symbols
            sifted = sift(ordered([
                (i.free_symbols, i) for i in eqs]),
                lambda x: len(x[0]))
            eqs = []
            nonlineqs = []
            while 1 in sifted:
                for free, e in sifted.pop(1):
                    x = free.pop()
                    if (e.lhs != x or x in e.rhs.free_symbols) and x not in reps:
                        try:
                            m, b = linear_coeffs(
                                Add(e.lhs, -e.rhs, evaluate=False), x)
                            enew = e.func(x, -b/m)
                            if measure(enew) <= ratio*measure(e):
                                e = enew
                            else:
                                eqs.append(e)
                                continue
                        except ValueError:
                            pass
                    if x in reps:
                        eqs.append(e.subs(x, reps[x]))
                    elif e.lhs == x and x not in e.rhs.free_symbols:
                        reps[x] = e.rhs
                        eqs.append(e)
                    else:
                        # x is not yet identified, but may be later
                        nonlineqs.append(e)
                resifted = defaultdict(list)
                for k in sifted:
                    for f, e in sifted[k]:
                        e = e.xreplace(reps)
                        f = e.free_symbols
                        resifted[len(f)].append((f, e))
                sifted = resifted
            for k in sifted:
                eqs.extend([e for f, e in sifted[k]])
            nonlineqs = [ei.subs(reps) for ei in nonlineqs]
            other = [ei.subs(reps) for ei in other]
            rv = rv.func(*([i.canonical for i in (eqs + nonlineqs + other)] + nonRel))
        patterns = _simplify_patterns_and()
        threeterm_patterns = _simplify_patterns_and3()
        return _apply_patternbased_simplification(rv, patterns,
                                                  measure, false,
                                                  threeterm_patterns=threeterm_patterns)

    def _eval_as_set(self):
        from sympy.sets.sets import Intersection
        return Intersection(*[arg.as_set() for arg in self.args])

    def _eval_rewrite_as_Nor(self, *args, **kwargs):
        return Nor(*[Not(arg) for arg in self.args])

    def to_anf(self, deep=True):
        if deep:
            result = And._to_anf(*self.args, deep=deep)
            return distribute_xor_over_and(result)
        return self


class Or(LatticeOp, BooleanFunction):
    """
    Logical OR function

    It evaluates its arguments in order, returning true immediately
    when an  argument is true, and false if they are all false.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Or
    >>> x | y
    x | y

    Notes
    =====

    The ``|`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise
    or. Hence, ``Or(a, b)`` and ``a | b`` will return different things if
    ``a`` and ``b`` are integers.

    >>> Or(x, y).subs(x, 0)
    y

    """
    zero = true
    identity = false

    @classmethod
    def _new_args_filter(cls, args):
        newargs = []
        rel = []
        args = BooleanFunction.binary_check_and_simplify(*args)
        for x in args:
            if x.is_Relational:
                c = x.canonical
                if c in rel:
                    continue
                nc = c.negated.canonical
                if any(r == nc for r in rel):
                    return [true]
                rel.append(c)
            newargs.append(x)
        return LatticeOp._new_args_filter(newargs, Or)

    def _eval_subs(self, old, new):
        args = []
        bad = None
        for i in self.args:
            try:
                i = i.subs(old, new)
            except TypeError:
                # store TypeError
                if bad is None:
                    bad = i
                continue
            if i == True:
                return true
            elif i != False:
                args.append(i)
        if bad is not None:
            # let it raise
            bad.subs(old, new)
        # If old is Or, replace the parts of the arguments with new if all
        # are there
        if isinstance(old, Or):
            old_set = set(old.args)
            if old_set.issubset(args):
                args = set(args) - old_set
                args.add(new)

        return self.func(*args)

    def _eval_as_set(self):
        from sympy.sets.sets import Union
        return Union(*[arg.as_set() for arg in self.args])

    def _eval_rewrite_as_Nand(self, *args, **kwargs):
        return Nand(*[Not(arg) for arg in self.args])

    def _eval_simplify(self, **kwargs):
        from sympy.core.relational import Le, Ge, Eq
        lege = self.atoms(Le, Ge)
        if lege:
            reps = {i: self.func(
                Eq(i.lhs, i.rhs), i.strict) for i in lege}
            return self.xreplace(reps)._eval_simplify(**kwargs)
        # standard simplify
        rv = super()._eval_simplify(**kwargs)
        if not isinstance(rv, Or):
            return rv
        patterns = _simplify_patterns_or()
        return _apply_patternbased_simplification(rv, patterns,
                                                  kwargs['measure'], true)

    def to_anf(self, deep=True):
        args = range(1, len(self.args) + 1)
        args = (combinations(self.args, j) for j in args)
        args = chain.from_iterable(args)  # powerset
        args = (And(*arg) for arg in args)
        args = (to_anf(x, deep=deep) if deep else x for x in args)
        return Xor(*list(args), remove_true=False)


class Not(BooleanFunction):
    """
    Logical Not function (negation)


    Returns ``true`` if the statement is ``false`` or ``False``.
    Returns ``false`` if the statement is ``true`` or ``True``.

    Examples
    ========

    >>> from sympy import Not, And, Or
    >>> from sympy.abc import x, A, B
    >>> Not(True)
    False
    >>> Not(False)
    True
    >>> Not(And(True, False))
    True
    >>> Not(Or(True, False))
    False
    >>> Not(And(And(True, x), Or(x, False)))
    ~x
    >>> ~x
    ~x
    >>> Not(And(Or(A, B), Or(~A, ~B)))
    ~((A | B) & (~A | ~B))

    Notes
    =====

    - The ``~`` operator is provided as a convenience, but note that its use
      here is different from its normal use in Python, which is bitwise
      not. In particular, ``~a`` and ``Not(a)`` will be different if ``a`` is
      an integer. Furthermore, since bools in Python subclass from ``int``,
      ``~True`` is the same as ``~1`` which is ``-2``, which has a boolean
      value of True.  To avoid this issue, use the SymPy boolean types
      ``true`` and ``false``.

    - As of Python 3.12, the bitwise not operator ``~`` used on a
      Python ``bool`` is deprecated and will emit a warning.

    >>> from sympy import true
    >>> ~True  # doctest: +SKIP
    -2
    >>> ~true
    False

    """

    is_Not = True

    @classmethod
    def eval(cls, arg):
        if isinstance(arg, Number) or arg in (True, False):
            return false if arg else true
        if arg.is_Not:
            return arg.args[0]
        # Simplify Relational objects.
        if arg.is_Relational:
            return arg.negated

    def _eval_as_set(self):
        """
        Rewrite logic operators and relationals in terms of real sets.

        Examples
        ========

        >>> from sympy import Not, Symbol
        >>> x = Symbol('x')
        >>> Not(x > 0).as_set()
        Interval(-oo, 0)
        """
        return self.args[0].as_set().complement(S.Reals)

    def to_nnf(self, simplify=True):
        if is_literal(self):
            return self

        expr = self.args[0]

        func, args = expr.func, expr.args

        if func == And:
            return Or._to_nnf(*[Not(arg) for arg in args], simplify=simplify)

        if func == Or:
            return And._to_nnf(*[Not(arg) for arg in args], simplify=simplify)

        if func == Implies:
            a, b = args
            return And._to_nnf(a, Not(b), simplify=simplify)

        if func == Equivalent:
            return And._to_nnf(Or(*args), Or(*[Not(arg) for arg in args]),
                               simplify=simplify)

        if func == Xor:
            result = []
            for i in range(1, len(args)+1, 2):
                for neg in combinations(args, i):
                    clause = [Not(s) if s in neg else s for s in args]
                    result.append(Or(*clause))
            return And._to_nnf(*result, simplify=simplify)

        if func == ITE:
            a, b, c = args
            return And._to_nnf(Or(a, Not(c)), Or(Not(a), Not(b)), simplify=simplify)

        raise ValueError("Illegal operator %s in expression" % func)

    def to_anf(self, deep=True):
        return Xor._to_anf(true, self.args[0], deep=deep)


class Xor(BooleanFunction):
    """
    Logical XOR (exclusive OR) function.


    Returns True if an odd number of the arguments are True and the rest are
    False.

    Returns False if an even number of the arguments are True and the rest are
    False.

    Examples
    ========

    >>> from sympy.logic.boolalg import Xor
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> Xor(True, False)
    True
    >>> Xor(True, True)
    False
    >>> Xor(True, False, True, True, False)
    True
    >>> Xor(True, False, True, False)
    False
    >>> x ^ y
    x ^ y

    Notes
    =====

    The ``^`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise xor. In
    particular, ``a ^ b`` and ``Xor(a, b)`` will be different if ``a`` and
    ``b`` are integers.

    >>> Xor(x, y).subs(y, 0)
    x

    """
    def __new__(cls, *args, remove_true=True, **kwargs):
        argset = set()
        obj = super().__new__(cls, *args, **kwargs)
        for arg in obj._args:
            if isinstance(arg, Number) or arg in (True, False):
                if arg:
                    arg = true
                else:
                    continue
            if isinstance(arg, Xor):
                for a in arg.args:
                    argset.remove(a) if a in argset else argset.add(a)
            elif arg in argset:
                argset.remove(arg)
            else:
                argset.add(arg)
        rel = [(r, r.canonical, r.negated.canonical)
               for r in argset if r.is_Relational]
        odd = False  # is number of complimentary pairs odd? start 0 -> False
        remove = []
        for i, (r, c, nc) in enumerate(rel):
            for j in range(i + 1, len(rel)):
                rj, cj = rel[j][:2]
                if cj == nc:
                    odd = not odd
                    break
                elif cj == c:
                    break
            else:
                continue
            remove.append((r, rj))
        if odd:
            argset.remove(true) if true in argset else argset.add(true)
        for a, b in remove:
            argset.remove(a)
            argset.remove(b)
        if len(argset) == 0:
            return false
        elif len(argset) == 1:
            return argset.pop()
        elif True in argset and remove_true:
            argset.remove(True)
            return Not(Xor(*argset))
        else:
            obj._args = tuple(ordered(argset))
            obj._argset = frozenset(argset)
            return obj

    # XXX: This should be cached on the object rather than using cacheit
    # Maybe it can be computed in __new__?
    @property  # type: ignore
    @cacheit
    def args(self):
        return tuple(ordered(self._argset))

    def to_nnf(self, simplify=True):
        args = []
        for i in range(0, len(self.args)+1, 2):
            for neg in combinations(self.args, i):
                clause = [Not(s) if s in neg else s for s in self.args]
                args.append(Or(*clause))
        return And._to_nnf(*args, simplify=simplify)

    def _eval_rewrite_as_Or(self, *args, **kwargs):
        a = self.args
        return Or(*[_convert_to_varsSOP(x, self.args)
                    for x in _get_odd_parity_terms(len(a))])

    def _eval_rewrite_as_And(self, *args, **kwargs):
        a = self.args
        return And(*[_convert_to_varsPOS(x, self.args)
                     for x in _get_even_parity_terms(len(a))])

    def _eval_simplify(self, **kwargs):
        # as standard simplify uses simplify_logic which writes things as
        # And and Or, we only simplify the partial expressions before using
        # patterns
        rv = self.func(*[a.simplify(**kwargs) for a in self.args])
        rv = rv.to_anf()
        if not isinstance(rv, Xor):  # This shouldn't really happen here
            return rv
        patterns = _simplify_patterns_xor()
        return _apply_patternbased_simplification(rv, patterns,
                                                  kwargs['measure'], None)

    def _eval_subs(self, old, new):
        # If old is Xor, replace the parts of the arguments with new if all
        # are there
        if isinstance(old, Xor):
            old_set = set(old.args)
            if old_set.issubset(self.args):
                args = set(self.args) - old_set
                args.add(new)
                return self.func(*args)


class Nand(BooleanFunction):
    """
    Logical NAND function.

    It evaluates its arguments in order, giving True immediately if any
    of them are False, and False if they are all True.

    Returns True if any of the arguments are False
    Returns False if all arguments are True

    Examples
    ========

    >>> from sympy.logic.boolalg import Nand
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> Nand(False, True)
    True
    >>> Nand(True, True)
    False
    >>> Nand(x, y)
    ~(x & y)

    """
    @classmethod
    def eval(cls, *args):
        return Not(And(*args))


class Nor(BooleanFunction):
    """
    Logical NOR function.

    It evaluates its arguments in order, giving False immediately if any
    of them are True, and True if they are all False.

    Returns False if any argument is True
    Returns True if all arguments are False

    Examples
    ========

    >>> from sympy.logic.boolalg import Nor
    >>> from sympy import symbols
    >>> x, y = symbols('x y')

    >>> Nor(True, False)
    False
    >>> Nor(True, True)
    False
    >>> Nor(False, True)
    False
    >>> Nor(False, False)
    True
    >>> Nor(x, y)
    ~(x | y)

    """
    @classmethod
    def eval(cls, *args):
        return Not(Or(*args))


class Xnor(BooleanFunction):
    """
    Logical XNOR function.

    Returns False if an odd number of the arguments are True and the rest are
    False.

    Returns True if an even number of the arguments are True and the rest are
    False.

    Examples
    ========

    >>> from sympy.logic.boolalg import Xnor
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> Xnor(True, False)
    False
    >>> Xnor(True, True)
    True
    >>> Xnor(True, False, True, True, False)
    False
    >>> Xnor(True, False, True, False)
    True

    """
    @classmethod
    def eval(cls, *args):
        return Not(Xor(*args))


class Implies(BooleanFunction):
    r"""
    Logical implication.

    A implies B is equivalent to if A then B. Mathematically, it is written
    as `A \Rightarrow B` and is equivalent to `\neg A \vee B` or ``~A | B``.

    Accepts two Boolean arguments; A and B.
    Returns False if A is True and B is False
    Returns True otherwise.

    Examples
    ========

    >>> from sympy.logic.boolalg import Implies
    >>> from sympy import symbols
    >>> x, y = symbols('x y')

    >>> Implies(True, False)
    False
    >>> Implies(False, False)
    True
    >>> Implies(True, True)
    True
    >>> Implies(False, True)
    True
    >>> x >> y
    Implies(x, y)
    >>> y << x
    Implies(x, y)

    Notes
    =====

    The ``>>`` and ``<<`` operators are provided as a convenience, but note
    that their use here is different from their normal use in Python, which is
    bit shifts. Hence, ``Implies(a, b)`` and ``a >> b`` will return different
    things if ``a`` and ``b`` are integers.  In particular, since Python
    considers ``True`` and ``False`` to be integers, ``True >> True`` will be
    the same as ``1 >> 1``, i.e., 0, which has a truth value of False.  To
    avoid this issue, use the SymPy objects ``true`` and ``false``.

    >>> from sympy import true, false
    >>> True >> False
    1
    >>> true >> false
    False

    """
    @classmethod
    def eval(cls, *args):
        try:
            newargs = []
            for x in args:
                if isinstance(x, Number) or x in (0, 1):
                    newargs.append(bool(x))
                else:
                    newargs.append(x)
            A, B = newargs
        except ValueError:
            raise ValueError(
                "%d operand(s) used for an Implies "
                "(pairs are required): %s" % (len(args), str(args)))
        if A in (True, False) or B in (True, False):
            return Or(Not(A), B)
        elif A == B:
            return true
        elif A.is_Relational and B.is_Relational:
            if A.canonical == B.canonical:
                return true
            if A.negated.canonical == B.canonical:
                return B
        else:
            return Basic.__new__(cls, *args)

    def to_nnf(self, simplify=True):
        a, b = self.args
        return Or._to_nnf(Not(a), b, simplify=simplify)

    def to_anf(self, deep=True):
        a, b = self.args
        return Xor._to_anf(true, a, And(a, b), deep=deep)


class Equivalent(BooleanFunction):
    """
    Equivalence relation.

    ``Equivalent(A, B)`` is True iff A and B are both True or both False.

    Returns True if all of the arguments are logically equivalent.
    Returns False otherwise.

    For two arguments, this is equivalent to :py:class:`~.Xnor`.

    Examples
    ========

    >>> from sympy.logic.boolalg import Equivalent, And
    >>> from sympy.abc import x
    >>> Equivalent(False, False, False)
    True
    >>> Equivalent(True, False, False)
    False
    >>> Equivalent(x, And(x, True))
    True

    """
    def __new__(cls, *args, **options):
        from sympy.core.relational import Relational
        args = [_sympify(arg) for arg in args]

        argset = set(args)
        for x in args:
            if isinstance(x, Number) or x in [True, False]:  # Includes 0, 1
                argset.discard(x)
                argset.add(bool(x))
        rel = []
        for r in argset:
            if isinstance(r, Relational):
                rel.append((r, r.canonical, r.negated.canonical))
        remove = []
        for i, (r, c, nc) in enumerate(rel):
            for j in range(i + 1, len(rel)):
                rj, cj = rel[j][:2]
                if cj == nc:
                    return false
                elif cj == c:
                    remove.append((r, rj))
                    break
        for a, b in remove:
            argset.remove(a)
            argset.remove(b)
            argset.add(True)
        if len(argset) <= 1:
            return true
        if True in argset:
            argset.discard(True)
            return And(*argset)
        if False in argset:
            argset.discard(False)
            return And(*[Not(arg) for arg in argset])
        _args = frozenset(argset)
        obj = super().__new__(cls, _args)
        obj._argset = _args
        return obj

    # XXX: This should be cached on the object rather than using cacheit
    # Maybe it can be computed in __new__?
    @property  # type: ignore
    @cacheit
    def args(self):
        return tuple(ordered(self._argset))

    def to_nnf(self, simplify=True):
        args = []
        for a, b in zip(self.args, self.args[1:]):
            args.append(Or(Not(a), b))
        args.append(Or(Not(self.args[-1]), self.args[0]))
        return And._to_nnf(*args, simplify=simplify)

    def to_anf(self, deep=True):
        a = And(*self.args)
        b = And(*[to_anf(Not(arg), deep=False) for arg in self.args])
        b = distribute_xor_over_and(b)
        return Xor._to_anf(a, b, deep=deep)


class ITE(BooleanFunction):
    """
    If-then-else clause.

    ``ITE(A, B, C)`` evaluates and returns the result of B if A is true
    else it returns the result of C. All args must be Booleans.

    From a logic gate perspective, ITE corresponds to a 2-to-1 multiplexer,
    where A is the select signal.

    Examples
    ========

    >>> from sympy.logic.boolalg import ITE, And, Xor, Or
    >>> from sympy.abc import x, y, z
    >>> ITE(True, False, True)
    False
    >>> ITE(Or(True, False), And(True, True), Xor(True, True))
    True
    >>> ITE(x, y, z)
    ITE(x, y, z)
    >>> ITE(True, x, y)
    x
    >>> ITE(False, x, y)
    y
    >>> ITE(x, y, y)
    y

    Trying to use non-Boolean args will generate a TypeError:

    >>> ITE(True, [], ())
    Traceback (most recent call last):
    ...
    TypeError: expecting bool, Boolean or ITE, not `[]`

    """
    def __new__(cls, *args, **kwargs):
        from sympy.core.relational import Eq, Ne
        if len(args) != 3:
            raise ValueError('expecting exactly 3 args')
        a, b, c = args
        # check use of binary symbols
        if isinstance(a, (Eq, Ne)):
            # in this context, we can evaluate the Eq/Ne
            # if one arg is a binary symbol and the other
            # is true/false
            b, c = map(as_Boolean, (b, c))
            bin_syms = set().union(*[i.binary_symbols for i in (b, c)])
            if len(set(a.args) - bin_syms) == 1:
                # one arg is a binary_symbols
                _a = a
                if a.lhs is true:
                    a = a.rhs
                elif a.rhs is true:
                    a = a.lhs
                elif a.lhs is false:
                    a = Not(a.rhs)
                elif a.rhs is false:
                    a = Not(a.lhs)
                else:
                    # binary can only equal True or False
                    a = false
                if isinstance(_a, Ne):
                    a = Not(a)
        else:
            a, b, c = BooleanFunction.binary_check_and_simplify(
                a, b, c)
        rv = None
        if kwargs.get('evaluate', True):
            rv = cls.eval(a, b, c)
        if rv is None:
            rv = BooleanFunction.__new__(cls, a, b, c, evaluate=False)
        return rv

    @classmethod
    def eval(cls, *args):
        from sympy.core.relational import Eq, Ne
        # do the args give a singular result?
        a, b, c = args
        if isinstance(a, (Ne, Eq)):
            _a = a
            if true in a.args:
                a = a.lhs if a.rhs is true else a.rhs
            elif false in a.args:
                a = Not(a.lhs) if a.rhs is false else Not(a.rhs)
            else:
                _a = None
            if _a is not None and isinstance(_a, Ne):
                a = Not(a)
        if a is true:
            return b
        if a is false:
            return c
        if b == c:
            return b
        else:
            # or maybe the results allow the answer to be expressed
            # in terms of the condition
            if b is true and c is false:
                return a
            if b is false and c is true:
                return Not(a)
        if [a, b, c] != args:
            return cls(a, b, c, evaluate=False)

    def to_nnf(self, simplify=True):
        a, b, c = self.args
        return And._to_nnf(Or(Not(a), b), Or(a, c), simplify=simplify)

    def _eval_as_set(self):
        return self.to_nnf().as_set()

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        return Piecewise((args[1], args[0]), (args[2], True))


class Exclusive(BooleanFunction):
    """
    True if only one or no argument is true.

    ``Exclusive(A, B, C)`` is equivalent to ``~(A & B) & ~(A & C) & ~(B & C)``.

    For two arguments, this is equivalent to :py:class:`~.Xor`.

    Examples
    ========

    >>> from sympy.logic.boolalg import Exclusive
    >>> Exclusive(False, False, False)
    True
    >>> Exclusive(False, True, False)
    True
    >>> Exclusive(False, True, True)
    False

    """
    @classmethod
    def eval(cls, *args):
        and_args = []
        for a, b in combinations(args, 2):
            and_args.append(Not(And(a, b)))
        return And(*and_args)


# end class definitions. Some useful methods


def conjuncts(expr):
    """Return a list of the conjuncts in ``expr``.

    Examples
    ========

    >>> from sympy.logic.boolalg import conjuncts
    >>> from sympy.abc import A, B
    >>> conjuncts(A & B)
    frozenset({A, B})
    >>> conjuncts(A | B)
    frozenset({A | B})

    """
    return And.make_args(expr)


def disjuncts(expr):
    """Return a list of the disjuncts in ``expr``.

    Examples
    ========

    >>> from sympy.logic.boolalg import disjuncts
    >>> from sympy.abc import A, B
    >>> disjuncts(A | B)
    frozenset({A, B})
    >>> disjuncts(A & B)
    frozenset({A & B})

    """
    return Or.make_args(expr)


def distribute_and_over_or(expr):
    """
    Given a sentence ``expr`` consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in CNF.

    Examples
    ========

    >>> from sympy.logic.boolalg import distribute_and_over_or, And, Or, Not
    >>> from sympy.abc import A, B, C
    >>> distribute_and_over_or(Or(A, And(Not(B), Not(C))))
    (A | ~B) & (A | ~C)

    """
    return _distribute((expr, And, Or))


def distribute_or_over_and(expr):
    """
    Given a sentence ``expr`` consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in DNF.

    Note that the output is NOT simplified.

    Examples
    ========

    >>> from sympy.logic.boolalg import distribute_or_over_and, And, Or, Not
    >>> from sympy.abc import A, B, C
    >>> distribute_or_over_and(And(Or(Not(A), B), C))
    (B & C) | (C & ~A)

    """
    return _distribute((expr, Or, And))


def distribute_xor_over_and(expr):
    """
    Given a sentence ``expr`` consisting of conjunction and
    exclusive disjunctions of literals, return an
    equivalent exclusive disjunction.

    Note that the output is NOT simplified.

    Examples
    ========

    >>> from sympy.logic.boolalg import distribute_xor_over_and, And, Xor, Not
    >>> from sympy.abc import A, B, C
    >>> distribute_xor_over_and(And(Xor(Not(A), B), C))
    (B & C) ^ (C & ~A)
    """
    return _distribute((expr, Xor, And))


def _distribute(info):
    """
    Distributes ``info[1]`` over ``info[2]`` with respect to ``info[0]``.
    """
    if isinstance(info[0], info[2]):
        for arg in info[0].args:
            if isinstance(arg, info[1]):
                conj = arg
                break
        else:
            return info[0]
        rest = info[2](*[a for a in info[0].args if a is not conj])
        return info[1](*list(map(_distribute,
                                 [(info[2](c, rest), info[1], info[2])
                                  for c in conj.args])), remove_true=False)
    elif isinstance(info[0], info[1]):
        return info[1](*list(map(_distribute,
                                 [(x, info[1], info[2])
                                  for x in info[0].args])),
                       remove_true=False)
    else:
        return info[0]


def to_anf(expr, deep=True):
    r"""
    Converts expr to Algebraic Normal Form (ANF).

    ANF is a canonical normal form, which means that two
    equivalent formulas will convert to the same ANF.

    A logical expression is in ANF if it has the form

    .. math:: 1 \oplus a \oplus b \oplus ab \oplus abc

    i.e. it can be:
        - purely true,
        - purely false,
        - conjunction of variables,
        - exclusive disjunction.

    The exclusive disjunction can only contain true, variables
    or conjunction of variables. No negations are permitted.

    If ``deep`` is ``False``, arguments of the boolean
    expression are considered variables, i.e. only the
    top-level expression is converted to ANF.

    Examples
    ========
    >>> from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent
    >>> from sympy.logic.boolalg import to_anf
    >>> from sympy.abc import A, B, C
    >>> to_anf(Not(A))
    A ^ True
    >>> to_anf(And(Or(A, B), Not(C)))
    A ^ B ^ (A & B) ^ (A & C) ^ (B & C) ^ (A & B & C)
    >>> to_anf(Implies(Not(A), Equivalent(B, C)), deep=False)
    True ^ ~A ^ (~A & (Equivalent(B, C)))

    """
    expr = sympify(expr)

    if is_anf(expr):
        return expr
    return expr.to_anf(deep=deep)


def to_nnf(expr, simplify=True):
    """
    Converts ``expr`` to Negation Normal Form (NNF).

    A logical expression is in NNF if it
    contains only :py:class:`~.And`, :py:class:`~.Or` and :py:class:`~.Not`,
    and :py:class:`~.Not` is applied only to literals.
    If ``simplify`` is ``True``, the result contains no redundant clauses.

    Examples
    ========

    >>> from sympy.abc import A, B, C, D
    >>> from sympy.logic.boolalg import Not, Equivalent, to_nnf
    >>> to_nnf(Not((~A & ~B) | (C & D)))
    (A | B) & (~C | ~D)
    >>> to_nnf(Equivalent(A >> B, B >> A))
    (A | ~B | (A & ~B)) & (B | ~A | (B & ~A))

    """
    if is_nnf(expr, simplify):
        return expr
    return expr.to_nnf(simplify)


def to_cnf(expr, simplify=False, force=False):
    """
    Convert a propositional logical sentence ``expr`` to conjunctive normal
    form: ``((A | ~B | ...) & (B | C | ...) & ...)``.
    If ``simplify`` is ``True``, ``expr`` is evaluated to its simplest CNF
    form using the Quine-McCluskey algorithm; this may take a long
    time. If there are more than 8 variables the ``force`` flag must be set
    to ``True`` to simplify (default is ``False``).

    Examples
    ========

    >>> from sympy.logic.boolalg import to_cnf
    >>> from sympy.abc import A, B, D
    >>> to_cnf(~(A | B) | D)
    (D | ~A) & (D | ~B)
    >>> to_cnf((A | B) & (A | ~A), True)
    A | B

    """
    expr = sympify(expr)
    if not isinstance(expr, BooleanFunction):
        return expr

    if simplify:
        if not force and len(_find_predicates(expr)) > 8:
            raise ValueError(filldedent('''
            To simplify a logical expression with more
            than 8 variables may take a long time and requires
            the use of `force=True`.'''))
        return simplify_logic(expr, 'cnf', True, force=force)

    # Don't convert unless we have to
    if is_cnf(expr):
        return expr

    expr = eliminate_implications(expr)
    res = distribute_and_over_or(expr)

    return res


def to_dnf(expr, simplify=False, force=False):
    """
    Convert a propositional logical sentence ``expr`` to disjunctive normal
    form: ``((A & ~B & ...) | (B & C & ...) | ...)``.
    If ``simplify`` is ``True``, ``expr`` is evaluated to its simplest DNF form using
    the Quine-McCluskey algorithm; this may take a long
    time. If there are more than 8 variables, the ``force`` flag must be set to
    ``True`` to simplify (default is ``False``).

    Examples
    ========

    >>> from sympy.logic.boolalg import to_dnf
    >>> from sympy.abc import A, B, C
    >>> to_dnf(B & (A | C))
    (A & B) | (B & C)
    >>> to_dnf((A & B) | (A & ~B) | (B & C) | (~B & C), True)
    A | C

    """
    expr = sympify(expr)
    if not isinstance(expr, BooleanFunction):
        return expr

    if simplify:
        if not force and len(_find_predicates(expr)) > 8:
            raise ValueError(filldedent('''
            To simplify a logical expression with more
            than 8 variables may take a long time and requires
            the use of `force=True`.'''))
        return simplify_logic(expr, 'dnf', True, force=force)

    # Don't convert unless we have to
    if is_dnf(expr):
        return expr

    expr = eliminate_implications(expr)
    return distribute_or_over_and(expr)


def is_anf(expr):
    r"""
    Checks if ``expr``  is in Algebraic Normal Form (ANF).

    A logical expression is in ANF if it has the form

    .. math:: 1 \oplus a \oplus b \oplus ab \oplus abc

    i.e. it is purely true, purely false, conjunction of
    variables or exclusive disjunction. The exclusive
    disjunction can only contain true, variables or
    conjunction of variables. No negations are permitted.

    Examples
    ========

    >>> from sympy.logic.boolalg import And, Not, Xor, true, is_anf
    >>> from sympy.abc import A, B, C
    >>> is_anf(true)
    True
    >>> is_anf(A)
    True
    >>> is_anf(And(A, B, C))
    True
    >>> is_anf(Xor(A, Not(B)))
    False

    """
    expr = sympify(expr)

    if is_literal(expr) and not isinstance(expr, Not):
        return True

    if isinstance(expr, And):
        for arg in expr.args:
            if not arg.is_Symbol:
                return False
        return True

    elif isinstance(expr, Xor):
        for arg in expr.args:
            if isinstance(arg, And):
                for a in arg.args:
                    if not a.is_Symbol:
                        return False
            elif is_literal(arg):
                if isinstance(arg, Not):
                    return False
            else:
                return False
        return True

    else:
        return False


def is_nnf(expr, simplified=True):
    """
    Checks if ``expr`` is in Negation Normal Form (NNF).

    A logical expression is in NNF if it
    contains only :py:class:`~.And`, :py:class:`~.Or` and :py:class:`~.Not`,
    and :py:class:`~.Not` is applied only to literals.
    If ``simplified`` is ``True``, checks if result contains no redundant clauses.

    Examples
    ========

    >>> from sympy.abc import A, B, C
    >>> from sympy.logic.boolalg import Not, is_nnf
    >>> is_nnf(A & B | ~C)
    True
    >>> is_nnf((A | ~A) & (B | C))
    False
    >>> is_nnf((A | ~A) & (B | C), False)
    True
    >>> is_nnf(Not(A & B) | C)
    False
    >>> is_nnf((A >> B) & (B >> A))
    False

    """

    expr = sympify(expr)
    if is_literal(expr):
        return True

    stack = [expr]

    while stack:
        expr = stack.pop()
        if expr.func in (And, Or):
            if simplified:
                args = expr.args
                for arg in args:
                    if Not(arg) in args:
                        return False
            stack.extend(expr.args)

        elif not is_literal(expr):
            return False

    return True


def is_cnf(expr):
    """
    Test whether or not an expression is in conjunctive normal form.

    Examples
    ========

    >>> from sympy.logic.boolalg import is_cnf
    >>> from sympy.abc import A, B, C
    >>> is_cnf(A | B | C)
    True
    >>> is_cnf(A & B & C)
    True
    >>> is_cnf((A & B) | C)
    False

    """
    return _is_form(expr, And, Or)


def is_dnf(expr):
    """
    Test whether or not an expression is in disjunctive normal form.

    Examples
    ========

    >>> from sympy.logic.boolalg import is_dnf
    >>> from sympy.abc import A, B, C
    >>> is_dnf(A | B | C)
    True
    >>> is_dnf(A & B & C)
    True
    >>> is_dnf((A & B) | C)
    True
    >>> is_dnf(A & (B | C))
    False

    """
    return _is_form(expr, Or, And)


def _is_form(expr, function1, function2):
    """
    Test whether or not an expression is of the required form.

    """
    expr = sympify(expr)

    vals = function1.make_args(expr) if isinstance(expr, function1) else [expr]
    for lit in vals:
        if isinstance(lit, function2):
            vals2 = function2.make_args(lit) if isinstance(lit, function2) else [lit]
            for l in vals2:
                if is_literal(l) is False:
                    return False
        elif is_literal(lit) is False:
            return False

    return True


def eliminate_implications(expr):
    """
    Change :py:class:`~.Implies` and :py:class:`~.Equivalent` into
    :py:class:`~.And`, :py:class:`~.Or`, and :py:class:`~.Not`.
    That is, return an expression that is equivalent to ``expr``, but has only
    ``&``, ``|``, and ``~`` as logical
    operators.

    Examples
    ========

    >>> from sympy.logic.boolalg import Implies, Equivalent, \
         eliminate_implications
    >>> from sympy.abc import A, B, C
    >>> eliminate_implications(Implies(A, B))
    B | ~A
    >>> eliminate_implications(Equivalent(A, B))
    (A | ~B) & (B | ~A)
    >>> eliminate_implications(Equivalent(A, B, C))
    (A | ~C) & (B | ~A) & (C | ~B)

    """
    return to_nnf(expr, simplify=False)


def is_literal(expr):
    """
    Returns True if expr is a literal, else False.

    Examples
    ========

    >>> from sympy import Or, Q
    >>> from sympy.abc import A, B
    >>> from sympy.logic.boolalg import is_literal
    >>> is_literal(A)
    True
    >>> is_literal(~A)
    True
    >>> is_literal(Q.zero(A))
    True
    >>> is_literal(A + B)
    True
    >>> is_literal(Or(A, B))
    False

    """
    from sympy.assumptions import AppliedPredicate

    if isinstance(expr, Not):
        return is_literal(expr.args[0])
    elif expr in (True, False) or isinstance(expr, AppliedPredicate) or expr.is_Atom:
        return True
    elif not isinstance(expr, BooleanFunction) and all(
            (isinstance(expr, AppliedPredicate) or a.is_Atom) for a in expr.args):
        return True
    return False


def to_int_repr(clauses, symbols):
    """
    Takes clauses in CNF format and puts them into an integer representation.

    Examples
    ========

    >>> from sympy.logic.boolalg import to_int_repr
    >>> from sympy.abc import x, y
    >>> to_int_repr([x | y, y], [x, y]) == [{1, 2}, {2}]
    True

    """

    # Convert the symbol list into a dict
    symbols = dict(zip(symbols, range(1, len(symbols) + 1)))

    def append_symbol(arg, symbols):
        if isinstance(arg, Not):
            return -symbols[arg.args[0]]
        else:
            return symbols[arg]

    return [{append_symbol(arg, symbols) for arg in Or.make_args(c)}
            for c in clauses]


def term_to_integer(term):
    """
    Return an integer corresponding to the base-2 digits given by *term*.

    Parameters
    ==========

    term : a string or list of ones and zeros

    Examples
    ========

    >>> from sympy.logic.boolalg import term_to_integer
    >>> term_to_integer([1, 0, 0])
    4
    >>> term_to_integer('100')
    4

    """

    return int(''.join(list(map(str, list(term)))), 2)


integer_to_term = ibin  # XXX could delete?


def truth_table(expr, variables, input=True):
    """
    Return a generator of all possible configurations of the input variables,
    and the result of the boolean expression for those values.

    Parameters
    ==========

    expr : Boolean expression

    variables : list of variables

    input : bool (default ``True``)
        Indicates whether to return the input combinations.

    Examples
    ========

    >>> from sympy.logic.boolalg import truth_table
    >>> from sympy.abc import x,y
    >>> table = truth_table(x >> y, [x, y])
    >>> for t in table:
    ...     print('{0} -> {1}'.format(*t))
    [0, 0] -> True
    [0, 1] -> True
    [1, 0] -> False
    [1, 1] -> True

    >>> table = truth_table(x | y, [x, y])
    >>> list(table)
    [([0, 0], False), ([0, 1], True), ([1, 0], True), ([1, 1], True)]

    If ``input`` is ``False``, ``truth_table`` returns only a list of truth values.
    In this case, the corresponding input values of variables can be
    deduced from the index of a given output.

    >>> from sympy.utilities.iterables import ibin
    >>> vars = [y, x]
    >>> values = truth_table(x >> y, vars, input=False)
    >>> values = list(values)
    >>> values
    [True, False, True, True]

    >>> for i, value in enumerate(values):
    ...     print('{0} -> {1}'.format(list(zip(
    ...     vars, ibin(i, len(vars)))), value))
    [(y, 0), (x, 0)] -> True
    [(y, 0), (x, 1)] -> False
    [(y, 1), (x, 0)] -> True
    [(y, 1), (x, 1)] -> True

    """
    variables = [sympify(v) for v in variables]

    expr = sympify(expr)
    if not isinstance(expr, BooleanFunction) and not is_literal(expr):
        return

    table = product((0, 1), repeat=len(variables))
    for term in table:
        value = expr.xreplace(dict(zip(variables, term)))

        if input:
            yield list(term), value
        else:
            yield value


def _check_pair(minterm1, minterm2):
    """
    Checks if a pair of minterms differs by only one bit. If yes, returns
    index, else returns `-1`.
    """
    # Early termination seems to be faster than list comprehension,
    # at least for large examples.
    index = -1
    for x, i in enumerate(minterm1):  # zip(minterm1, minterm2) is slower
        if i != minterm2[x]:
            if index == -1:
                index = x
            else:
                return -1
    return index


def _convert_to_varsSOP(minterm, variables):
    """
    Converts a term in the expansion of a function from binary to its
    variable form (for SOP).
    """
    temp = [variables[n] if val == 1 else Not(variables[n])
            for n, val in enumerate(minterm) if val != 3]
    return And(*temp)


def _convert_to_varsPOS(maxterm, variables):
    """
    Converts a term in the expansion of a function from binary to its
    variable form (for POS).
    """
    temp = [variables[n] if val == 0 else Not(variables[n])
            for n, val in enumerate(maxterm) if val != 3]
    return Or(*temp)


def _convert_to_varsANF(term, variables):
    """
    Converts a term in the expansion of a function from binary to its
    variable form (for ANF).

    Parameters
    ==========

    term : list of 1's and 0's (complementation pattern)
    variables : list of variables

    """
    temp = [variables[n] for n, t in enumerate(term) if t == 1]

    if not temp:
        return true

    return And(*temp)


def _get_odd_parity_terms(n):
    """
    Returns a list of lists, with all possible combinations of n zeros and ones
    with an odd number of ones.
    """
    return [e for e in [ibin(i, n) for i in range(2**n)] if sum(e) % 2 == 1]


def _get_even_parity_terms(n):
    """
    Returns a list of lists, with all possible combinations of n zeros and ones
    with an even number of ones.
    """
    return [e for e in [ibin(i, n) for i in range(2**n)] if sum(e) % 2 == 0]


def _simplified_pairs(terms):
    """
    Reduces a set of minterms, if possible, to a simplified set of minterms
    with one less variable in the terms using QM method.
    """
    if not terms:
        return []

    simplified_terms = []
    todo = list(range(len(terms)))

    # Count number of ones as _check_pair can only potentially match if there
    # is at most a difference of a single one
    termdict = defaultdict(list)
    for n, term in enumerate(terms):
        ones = sum(1 for t in term if t == 1)
        termdict[ones].append(n)

    variables = len(terms[0])
    for k in range(variables):
        for i in termdict[k]:
            for j in termdict[k+1]:
                index = _check_pair(terms[i], terms[j])
                if index != -1:
                    # Mark terms handled
                    todo[i] = todo[j] = None
                    # Copy old term
                    newterm = terms[i][:]
                    # Set differing position to don't care
                    newterm[index] = 3
                    # Add if not already there
                    if newterm not in simplified_terms:
                        simplified_terms.append(newterm)

    if simplified_terms:
        # Further simplifications only among the new terms
        simplified_terms = _simplified_pairs(simplified_terms)

    # Add remaining, non-simplified, terms
    simplified_terms.extend([terms[i] for i in todo if i is not None])
    return simplified_terms


def _rem_redundancy(l1, terms):
    """
    After the truth table has been sufficiently simplified, use the prime
    implicant table method to recognize and eliminate redundant pairs,
    and return the essential arguments.
    """

    if not terms:
        return []

    nterms = len(terms)
    nl1 = len(l1)

    # Create dominating matrix
    dommatrix = [[0]*nl1 for n in range(nterms)]
    colcount = [0]*nl1
    rowcount = [0]*nterms
    for primei, prime in enumerate(l1):
        for termi, term in enumerate(terms):
            # Check prime implicant covering term
            if all(t == 3 or t == mt for t, mt in zip(prime, term)):
                dommatrix[termi][primei] = 1
                colcount[primei] += 1
                rowcount[termi] += 1

    # Keep track if anything changed
    anythingchanged = True
    # Then, go again
    while anythingchanged:
        anythingchanged = False

        for rowi in range(nterms):
            # Still non-dominated?
            if rowcount[rowi]:
                row = dommatrix[rowi]
                for row2i in range(nterms):
                    # Still non-dominated?
                    if rowi != row2i and rowcount[rowi] and (rowcount[rowi] <= rowcount[row2i]):
                        row2 = dommatrix[row2i]
                        if all(row2[n] >= row[n] for n in range(nl1)):
                            # row2 dominating row, remove row2
                            rowcount[row2i] = 0
                            anythingchanged = True
                            for primei, prime in enumerate(row2):
                                if prime:
                                    # Make corresponding entry 0
                                    dommatrix[row2i][primei] = 0
                                    colcount[primei] -= 1

        colcache = {}

        for coli in range(nl1):
            # Still non-dominated?
            if colcount[coli]:
                if coli in colcache:
                    col = colcache[coli]
                else:
                    col = [dommatrix[i][coli] for i in range(nterms)]
                    colcache[coli] = col
                for col2i in range(nl1):
                    # Still non-dominated?
                    if coli != col2i and colcount[col2i] and (colcount[coli] >= colcount[col2i]):
                        if col2i in colcache:
                            col2 = colcache[col2i]
                        else:
                            col2 = [dommatrix[i][col2i] for i in range(nterms)]
                            colcache[col2i] = col2
                        if all(col[n] >= col2[n] for n in range(nterms)):
                            # col dominating col2, remove col2
                            colcount[col2i] = 0
                            anythingchanged = True
                            for termi, term in enumerate(col2):
                                if term and dommatrix[termi][col2i]:
                                    # Make corresponding entry 0
                                    dommatrix[termi][col2i] = 0
                                    rowcount[termi] -= 1

        if not anythingchanged:
            # Heuristically select the prime implicant covering most terms
            maxterms = 0
            bestcolidx = -1
            for coli in range(nl1):
                s = colcount[coli]
                if s > maxterms:
                    bestcolidx = coli
                    maxterms = s

            # In case we found a prime implicant covering at least two terms
            if bestcolidx != -1 and maxterms > 1:
                for primei, prime in enumerate(l1):
                    if primei != bestcolidx:
                        for termi, term in enumerate(colcache[bestcolidx]):
                            if term and dommatrix[termi][primei]:
                                # Make corresponding entry 0
                                dommatrix[termi][primei] = 0
                                anythingchanged = True
                                rowcount[termi] -= 1
                                colcount[primei] -= 1

    return [l1[i] for i in range(nl1) if colcount[i]]


def _input_to_binlist(inputlist, variables):
    binlist = []
    bits = len(variables)
    for val in inputlist:
        if isinstance(val, int):
            binlist.append(ibin(val, bits))
        elif isinstance(val, dict):
            nonspecvars = list(variables)
            for key in val.keys():
                nonspecvars.remove(key)
            for t in product((0, 1), repeat=len(nonspecvars)):
                d = dict(zip(nonspecvars, t))
                d.update(val)
                binlist.append([d[v] for v in variables])
        elif isinstance(val, (list, tuple)):
            if len(val) != bits:
                raise ValueError("Each term must contain {bits} bits as there are"
                                 "\n{bits} variables (or be an integer)."
                                 "".format(bits=bits))
            binlist.append(list(val))
        else:
            raise TypeError("A term list can only contain lists,"
                            " ints or dicts.")
    return binlist


def SOPform(variables, minterms, dontcares=None):
    """
    The SOPform function uses simplified_pairs and a redundant group-
    eliminating algorithm to convert the list of all input combos that
    generate '1' (the minterms) into the smallest sum-of-products form.

    The variables must be given as the first argument.

    Return a logical :py:class:`~.Or` function (i.e., the "sum of products" or
    "SOP" form) that gives the desired outcome. If there are inputs that can
    be ignored, pass them as a list, too.

    The result will be one of the (perhaps many) functions that satisfy
    the conditions.

    Examples
    ========

    >>> from sympy.logic import SOPform
    >>> from sympy import symbols
    >>> w, x, y, z = symbols('w x y z')
    >>> minterms = [[0, 0, 0, 1], [0, 0, 1, 1],
    ...             [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
    >>> dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    >>> SOPform([w, x, y, z], minterms, dontcares)
    (y & z) | (~w & ~x)

    The terms can also be represented as integers:

    >>> minterms = [1, 3, 7, 11, 15]
    >>> dontcares = [0, 2, 5]
    >>> SOPform([w, x, y, z], minterms, dontcares)
    (y & z) | (~w & ~x)

    They can also be specified using dicts, which does not have to be fully
    specified:

    >>> minterms = [{w: 0, x: 1}, {y: 1, z: 1, x: 0}]
    >>> SOPform([w, x, y, z], minterms)
    (x & ~w) | (y & z & ~x)

    Or a combination:

    >>> minterms = [4, 7, 11, [1, 1, 1, 1]]
    >>> dontcares = [{w : 0, x : 0, y: 0}, 5]
    >>> SOPform([w, x, y, z], minterms, dontcares)
    (w & y & z) | (~w & ~y) | (x & z & ~w)

    See also
    ========

    POSform

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Quine-McCluskey_algorithm
    .. [2] https://en.wikipedia.org/wiki/Don%27t-care_term

    """
    if not minterms:
        return false

    variables = tuple(map(sympify, variables))


    minterms = _input_to_binlist(minterms, variables)
    dontcares = _input_to_binlist((dontcares or []), variables)
    for d in dontcares:
        if d in minterms:
            raise ValueError('%s in minterms is also in dontcares' % d)

    return _sop_form(variables, minterms, dontcares)


def _sop_form(variables, minterms, dontcares):
    new = _simplified_pairs(minterms + dontcares)
    essential = _rem_redundancy(new, minterms)
    return Or(*[_convert_to_varsSOP(x, variables) for x in essential])


def POSform(variables, minterms, dontcares=None):
    """
    The POSform function uses simplified_pairs and a redundant-group
    eliminating algorithm to convert the list of all input combinations
    that generate '1' (the minterms) into the smallest product-of-sums form.

    The variables must be given as the first argument.

    Return a logical :py:class:`~.And` function (i.e., the "product of sums"
    or "POS" form) that gives the desired outcome. If there are inputs that can
    be ignored, pass them as a list, too.

    The result will be one of the (perhaps many) functions that satisfy
    the conditions.

    Examples
    ========

    >>> from sympy.logic import POSform
    >>> from sympy import symbols
    >>> w, x, y, z = symbols('w x y z')
    >>> minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1],
    ...             [1, 0, 1, 1], [1, 1, 1, 1]]
    >>> dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    >>> POSform([w, x, y, z], minterms, dontcares)
    z & (y | ~w)

    The terms can also be represented as integers:

    >>> minterms = [1, 3, 7, 11, 15]
    >>> dontcares = [0, 2, 5]
    >>> POSform([w, x, y, z], minterms, dontcares)
    z & (y | ~w)

    They can also be specified using dicts, which does not have to be fully
    specified:

    >>> minterms = [{w: 0, x: 1}, {y: 1, z: 1, x: 0}]
    >>> POSform([w, x, y, z], minterms)
    (x | y) & (x | z) & (~w | ~x)

    Or a combination:

    >>> minterms = [4, 7, 11, [1, 1, 1, 1]]
    >>> dontcares = [{w : 0, x : 0, y: 0}, 5]
    >>> POSform([w, x, y, z], minterms, dontcares)
    (w | x) & (y | ~w) & (z | ~y)

    See also
    ========

    SOPform

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Quine-McCluskey_algorithm
    .. [2] https://en.wikipedia.org/wiki/Don%27t-care_term

    """
    if not minterms:
        return false

    variables = tuple(map(sympify, variables))
    minterms = _input_to_binlist(minterms, variables)
    dontcares = _input_to_binlist((dontcares or []), variables)
    for d in dontcares:
        if d in minterms:
            raise ValueError('%s in minterms is also in dontcares' % d)

    maxterms = []
    for t in product((0, 1), repeat=len(variables)):
        t = list(t)
        if (t not in minterms) and (t not in dontcares):
            maxterms.append(t)

    new = _simplified_pairs(maxterms + dontcares)
    essential = _rem_redundancy(new, maxterms)
    return And(*[_convert_to_varsPOS(x, variables) for x in essential])


def ANFform(variables, truthvalues):
    """
    The ANFform function converts the list of truth values to
    Algebraic Normal Form (ANF).

    The variables must be given as the first argument.

    Return True, False, logical :py:class:`~.And` function (i.e., the
    "Zhegalkin monomial") or logical :py:class:`~.Xor` function (i.e.,
    the "Zhegalkin polynomial"). When True and False
    are represented by 1 and 0, respectively, then
    :py:class:`~.And` is multiplication and :py:class:`~.Xor` is addition.

    Formally a "Zhegalkin monomial" is the product (logical
    And) of a finite set of distinct variables, including
    the empty set whose product is denoted 1 (True).
    A "Zhegalkin polynomial" is the sum (logical Xor) of a
    set of Zhegalkin monomials, with the empty set denoted
    by 0 (False).

    Parameters
    ==========

    variables : list of variables
    truthvalues : list of 1's and 0's (result column of truth table)

    Examples
    ========
    >>> from sympy.logic.boolalg import ANFform
    >>> from sympy.abc import x, y
    >>> ANFform([x], [1, 0])
    x ^ True
    >>> ANFform([x, y], [0, 1, 1, 1])
    x ^ y ^ (x & y)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Zhegalkin_polynomial

    """

    n_vars = len(variables)
    n_values = len(truthvalues)

    if n_values != 2 ** n_vars:
        raise ValueError("The number of truth values must be equal to 2^%d, "
                         "got %d" % (n_vars, n_values))

    variables = tuple(map(sympify, variables))

    coeffs = anf_coeffs(truthvalues)
    terms = []

    for i, t in enumerate(product((0, 1), repeat=n_vars)):
        if coeffs[i] == 1:
            terms.append(t)

    return Xor(*[_convert_to_varsANF(x, variables) for x in terms],
               remove_true=False)


def anf_coeffs(truthvalues):
    """
    Convert a list of truth values of some boolean expression
    to the list of coefficients of the polynomial mod 2 (exclusive
    disjunction) representing the boolean expression in ANF
    (i.e., the "Zhegalkin polynomial").

    There are `2^n` possible Zhegalkin monomials in `n` variables, since
    each monomial is fully specified by the presence or absence of
    each variable.

    We can enumerate all the monomials. For example, boolean
    function with four variables ``(a, b, c, d)`` can contain
    up to `2^4 = 16` monomials. The 13-th monomial is the
    product ``a & b & d``, because 13 in binary is 1, 1, 0, 1.

    A given monomial's presence or absence in a polynomial corresponds
    to that monomial's coefficient being 1 or 0 respectively.

    Examples
    ========
    >>> from sympy.logic.boolalg import anf_coeffs, bool_monomial, Xor
    >>> from sympy.abc import a, b, c
    >>> truthvalues = [0, 1, 1, 0, 0, 1, 0, 1]
    >>> coeffs = anf_coeffs(truthvalues)
    >>> coeffs
    [0, 1, 1, 0, 0, 0, 1, 0]
    >>> polynomial = Xor(*[
    ...     bool_monomial(k, [a, b, c])
    ...     for k, coeff in enumerate(coeffs) if coeff == 1
    ... ])
    >>> polynomial
    b ^ c ^ (a & b)

    """

    s = '{:b}'.format(len(truthvalues))
    n = len(s) - 1

    if len(truthvalues) != 2**n:
        raise ValueError("The number of truth values must be a power of two, "
                         "got %d" % len(truthvalues))

    coeffs = [[v] for v in truthvalues]

    for i in range(n):
        tmp = []
        for j in range(2 ** (n-i-1)):
            tmp.append(coeffs[2*j] +
                list(map(lambda x, y: x^y, coeffs[2*j], coeffs[2*j+1])))
        coeffs = tmp

    return coeffs[0]


def bool_minterm(k, variables):
    """
    Return the k-th minterm.

    Minterms are numbered by a binary encoding of the complementation
    pattern of the variables. This convention assigns the value 1 to
    the direct form and 0 to the complemented form.

    Parameters
    ==========

    k : int or list of 1's and 0's (complementation pattern)
    variables : list of variables

    Examples
    ========

    >>> from sympy.logic.boolalg import bool_minterm
    >>> from sympy.abc import x, y, z
    >>> bool_minterm([1, 0, 1], [x, y, z])
    x & z & ~y
    >>> bool_minterm(6, [x, y, z])
    x & y & ~z

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Canonical_normal_form#Indexing_minterms

    """
    if isinstance(k, int):
        k = ibin(k, len(variables))
    variables = tuple(map(sympify, variables))
    return _convert_to_varsSOP(k, variables)


def bool_maxterm(k, variables):
    """
    Return the k-th maxterm.

    Each maxterm is assigned an index based on the opposite
    conventional binary encoding used for minterms. The maxterm
    convention assigns the value 0 to the direct form and 1 to
    the complemented form.

    Parameters
    ==========

    k : int or list of 1's and 0's (complementation pattern)
    variables : list of variables

    Examples
    ========
    >>> from sympy.logic.boolalg import bool_maxterm
    >>> from sympy.abc import x, y, z
    >>> bool_maxterm([1, 0, 1], [x, y, z])
    y | ~x | ~z
    >>> bool_maxterm(6, [x, y, z])
    z | ~x | ~y

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Canonical_normal_form#Indexing_maxterms

    """
    if isinstance(k, int):
        k = ibin(k, len(variables))
    variables = tuple(map(sympify, variables))
    return _convert_to_varsPOS(k, variables)


def bool_monomial(k, variables):
    """
    Return the k-th monomial.

    Monomials are numbered by a binary encoding of the presence and
    absences of the variables. This convention assigns the value
    1 to the presence of variable and 0 to the absence of variable.

    Each boolean function can be uniquely represented by a
    Zhegalkin Polynomial (Algebraic Normal Form). The Zhegalkin
    Polynomial of the boolean function with `n` variables can contain
    up to `2^n` monomials. We can enumerate all the monomials.
    Each monomial is fully specified by the presence or absence
    of each variable.

    For example, boolean function with four variables ``(a, b, c, d)``
    can contain up to `2^4 = 16` monomials. The 13-th monomial is the
    product ``a & b & d``, because 13 in binary is 1, 1, 0, 1.

    Parameters
    ==========

    k : int or list of 1's and 0's
    variables : list of variables

    Examples
    ========
    >>> from sympy.logic.boolalg import bool_monomial
    >>> from sympy.abc import x, y, z
    >>> bool_monomial([1, 0, 1], [x, y, z])
    x & z
    >>> bool_monomial(6, [x, y, z])
    x & y

    """
    if isinstance(k, int):
        k = ibin(k, len(variables))
    variables = tuple(map(sympify, variables))
    return _convert_to_varsANF(k, variables)


def _find_predicates(expr):
    """Helper to find logical predicates in BooleanFunctions.

    A logical predicate is defined here as anything within a BooleanFunction
    that is not a BooleanFunction itself.

    """
    if not isinstance(expr, BooleanFunction):
        return {expr}
    return set().union(*(map(_find_predicates, expr.args)))


def simplify_logic(expr, form=None, deep=True, force=False, dontcare=None):
    """
    This function simplifies a boolean function to its simplified version
    in SOP or POS form. The return type is an :py:class:`~.Or` or
    :py:class:`~.And` object in SymPy.

    Parameters
    ==========

    expr : Boolean

    form : string (``'cnf'`` or ``'dnf'``) or ``None`` (default).
        If ``'cnf'`` or ``'dnf'``, the simplest expression in the corresponding
        normal form is returned; if ``None``, the answer is returned
        according to the form with fewest args (in CNF by default).

    deep : bool (default ``True``)
        Indicates whether to recursively simplify any
        non-boolean functions contained within the input.

    force : bool (default ``False``)
        As the simplifications require exponential time in the number
        of variables, there is by default a limit on expressions with
        8 variables. When the expression has more than 8 variables
        only symbolical simplification (controlled by ``deep``) is
        made. By setting ``force`` to ``True``, this limit is removed. Be
        aware that this can lead to very long simplification times.

    dontcare : Boolean
        Optimize expression under the assumption that inputs where this
        expression is true are don't care. This is useful in e.g. Piecewise
        conditions, where later conditions do not need to consider inputs that
        are converted by previous conditions. For example, if a previous
        condition is ``And(A, B)``, the simplification of expr can be made
        with don't cares for ``And(A, B)``.

    Examples
    ========

    >>> from sympy.logic import simplify_logic
    >>> from sympy.abc import x, y, z
    >>> b = (~x & ~y & ~z) | ( ~x & ~y & z)
    >>> simplify_logic(b)
    ~x & ~y
    >>> simplify_logic(x | y, dontcare=y)
    x

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Don%27t-care_term

    """

    if form not in (None, 'cnf', 'dnf'):
        raise ValueError("form can be cnf or dnf only")
    expr = sympify(expr)
    # check for quick exit if form is given: right form and all args are
    # literal and do not involve Not
    if form:
        form_ok = False
        if form == 'cnf':
            form_ok = is_cnf(expr)
        elif form == 'dnf':
            form_ok = is_dnf(expr)

        if form_ok and all(is_literal(a)
                for a in expr.args):
            return expr
    from sympy.core.relational import Relational
    if deep:
        variables = expr.atoms(Relational)
        from sympy.simplify.simplify import simplify
        s = tuple(map(simplify, variables))
        expr = expr.xreplace(dict(zip(variables, s)))
    if not isinstance(expr, BooleanFunction):
        return expr
    # Replace Relationals with Dummys to possibly
    # reduce the number of variables
    repl = {}
    undo = {}
    from sympy.core.symbol import Dummy
    variables = expr.atoms(Relational)
    if dontcare is not None:
        dontcare = sympify(dontcare)
        variables.update(dontcare.atoms(Relational))
    while variables:
        var = variables.pop()
        if var.is_Relational:
            d = Dummy()
            undo[d] = var
            repl[var] = d
            nvar = var.negated
            if nvar in variables:
                repl[nvar] = Not(d)
                variables.remove(nvar)

    expr = expr.xreplace(repl)

    if dontcare is not None:
        dontcare = dontcare.xreplace(repl)

    # Get new variables after replacing
    variables = _find_predicates(expr)
    if not force and len(variables) > 8:
        return expr.xreplace(undo)
    if dontcare is not None:
        # Add variables from dontcare
        dcvariables = _find_predicates(dontcare)
        variables.update(dcvariables)
        # if too many restore to variables only
        if not force and len(variables) > 8:
            variables = _find_predicates(expr)
            dontcare = None
    # group into constants and variable values
    c, v = sift(ordered(variables), lambda x: x in (True, False), binary=True)
    variables = c + v
    # standardize constants to be 1 or 0 in keeping with truthtable
    c = [1 if i == True else 0 for i in c]
    truthtable = _get_truthtable(v, expr, c)
    if dontcare is not None:
        dctruthtable = _get_truthtable(v, dontcare, c)
        truthtable = [t for t in truthtable if t not in dctruthtable]
    else:
        dctruthtable = []
    big = len(truthtable) >= (2 ** (len(variables) - 1))
    if form == 'dnf' or form is None and big:
        return _sop_form(variables, truthtable, dctruthtable).xreplace(undo)
    return POSform(variables, truthtable, dctruthtable).xreplace(undo)


def _get_truthtable(variables, expr, const):
    """ Return a list of all combinations leading to a True result for ``expr``.
    """
    _variables = variables.copy()
    def _get_tt(inputs):
        if _variables:
            v = _variables.pop()
            tab = [[i[0].xreplace({v: false}), [0] + i[1]] for i in inputs if i[0] is not false]
            tab.extend([[i[0].xreplace({v: true}), [1] + i[1]] for i in inputs if i[0] is not false])
            return _get_tt(tab)
        return inputs
    res = [const + k[1] for k in _get_tt([[expr, []]]) if k[0]]
    if res == [[]]:
        return []
    else:
        return res


def _finger(eq):
    """
    Assign a 5-item fingerprint to each symbol in the equation:
    [
    # of times it appeared as a Symbol;
    # of times it appeared as a Not(symbol);
    # of times it appeared as a Symbol in an And or Or;
    # of times it appeared as a Not(Symbol) in an And or Or;
    a sorted tuple of tuples, (i, j, k), where i is the number of arguments
    in an And or Or with which it appeared as a Symbol, and j is
    the number of arguments that were Not(Symbol); k is the number
    of times that (i, j) was seen.
    ]

    Examples
    ========

    >>> from sympy.logic.boolalg import _finger as finger
    >>> from sympy import And, Or, Not, Xor, to_cnf, symbols
    >>> from sympy.abc import a, b, x, y
    >>> eq = Or(And(Not(y), a), And(Not(y), b), And(x, y))
    >>> dict(finger(eq))
    {(0, 0, 1, 0, ((2, 0, 1),)): [x],
    (0, 0, 1, 0, ((2, 1, 1),)): [a, b],
    (0, 0, 1, 2, ((2, 0, 1),)): [y]}
    >>> dict(finger(x & ~y))
    {(0, 1, 0, 0, ()): [y], (1, 0, 0, 0, ()): [x]}

    In the following, the (5, 2, 6) means that there were 6 Or
    functions in which a symbol appeared as itself amongst 5 arguments in
    which there were also 2 negated symbols, e.g. ``(a0 | a1 | a2 | ~a3 | ~a4)``
    is counted once for a0, a1 and a2.

    >>> dict(finger(to_cnf(Xor(*symbols('a:5')))))
    {(0, 0, 8, 8, ((5, 0, 1), (5, 2, 6), (5, 4, 1))): [a0, a1, a2, a3, a4]}

    The equation must not have more than one level of nesting:

    >>> dict(finger(And(Or(x, y), y)))
    {(0, 0, 1, 0, ((2, 0, 1),)): [x], (1, 0, 1, 0, ((2, 0, 1),)): [y]}
    >>> dict(finger(And(Or(x, And(a, x)), y)))
    Traceback (most recent call last):
    ...
    NotImplementedError: unexpected level of nesting

    So y and x have unique fingerprints, but a and b do not.
    """
    f = eq.free_symbols
    d = dict(list(zip(f, [[0]*4 + [defaultdict(int)] for fi in f])))
    for a in eq.args:
        if a.is_Symbol:
            d[a][0] += 1
        elif a.is_Not:
            d[a.args[0]][1] += 1
        else:
            o = len(a.args), sum(isinstance(ai, Not) for ai in a.args)
            for ai in a.args:
                if ai.is_Symbol:
                    d[ai][2] += 1
                    d[ai][-1][o] += 1
                elif ai.is_Not:
                    d[ai.args[0]][3] += 1
                else:
                    raise NotImplementedError('unexpected level of nesting')
    inv = defaultdict(list)
    for k, v in ordered(iter(d.items())):
        v[-1] = tuple(sorted([i + (j,) for i, j in v[-1].items()]))
        inv[tuple(v)].append(k)
    return inv


def bool_map(bool1, bool2):
    """
    Return the simplified version of *bool1*, and the mapping of variables
    that makes the two expressions *bool1* and *bool2* represent the same
    logical behaviour for some correspondence between the variables
    of each.
    If more than one mappings of this sort exist, one of them
    is returned.

    For example, ``And(x, y)`` is logically equivalent to ``And(a, b)`` for
    the mapping ``{x: a, y: b}`` or ``{x: b, y: a}``.
    If no such mapping exists, return ``False``.

    Examples
    ========

    >>> from sympy import SOPform, bool_map, Or, And, Not, Xor
    >>> from sympy.abc import w, x, y, z, a, b, c, d
    >>> function1 = SOPform([x, z, y],[[1, 0, 1], [0, 0, 1]])
    >>> function2 = SOPform([a, b, c],[[1, 0, 1], [1, 0, 0]])
    >>> bool_map(function1, function2)
    (y & ~z, {y: a, z: b})

    The results are not necessarily unique, but they are canonical. Here,
    ``(w, z)`` could be ``(a, d)`` or ``(d, a)``:

    >>> eq =  Or(And(Not(y), w), And(Not(y), z), And(x, y))
    >>> eq2 = Or(And(Not(c), a), And(Not(c), d), And(b, c))
    >>> bool_map(eq, eq2)
    ((x & y) | (w & ~y) | (z & ~y), {w: a, x: b, y: c, z: d})
    >>> eq = And(Xor(a, b), c, And(c,d))
    >>> bool_map(eq, eq.subs(c, x))
    (c & d & (a | b) & (~a | ~b), {a: a, b: b, c: d, d: x})

    """

    def match(function1, function2):
        """Return the mapping that equates variables between two
        simplified boolean expressions if possible.

        By "simplified" we mean that a function has been denested
        and is either an And (or an Or) whose arguments are either
        symbols (x), negated symbols (Not(x)), or Or (or an And) whose
        arguments are only symbols or negated symbols. For example,
        ``And(x, Not(y), Or(w, Not(z)))``.

        Basic.match is not robust enough (see issue 4835) so this is
        a workaround that is valid for simplified boolean expressions
        """

        # do some quick checks
        if function1.__class__ != function2.__class__:
            return None  # maybe simplification makes them the same?
        if len(function1.args) != len(function2.args):
            return None  # maybe simplification makes them the same?
        if function1.is_Symbol:
            return {function1: function2}

        # get the fingerprint dictionaries
        f1 = _finger(function1)
        f2 = _finger(function2)

        # more quick checks
        if len(f1) != len(f2):
            return False

        # assemble the match dictionary if possible
        matchdict = {}
        for k in f1.keys():
            if k not in f2:
                return False
            if len(f1[k]) != len(f2[k]):
                return False
            for i, x in enumerate(f1[k]):
                matchdict[x] = f2[k][i]
        return matchdict

    a = simplify_logic(bool1)
    b = simplify_logic(bool2)
    m = match(a, b)
    if m:
        return a, m
    return m


def _apply_patternbased_simplification(rv, patterns, measure,
                                       dominatingvalue,
                                       replacementvalue=None,
                                       threeterm_patterns=None):
    """
    Replace patterns of Relational

    Parameters
    ==========

    rv : Expr
        Boolean expression

    patterns : tuple
        Tuple of tuples, with (pattern to simplify, simplified pattern) with
        two terms.

    measure : function
        Simplification measure.

    dominatingvalue : Boolean or ``None``
        The dominating value for the function of consideration.
        For example, for :py:class:`~.And` ``S.false`` is dominating.
        As soon as one expression is ``S.false`` in :py:class:`~.And`,
        the whole expression is ``S.false``.

    replacementvalue : Boolean or ``None``, optional
        The resulting value for the whole expression if one argument
        evaluates to ``dominatingvalue``.
        For example, for :py:class:`~.Nand` ``S.false`` is dominating, but
        in this case the resulting value is ``S.true``. Default is ``None``.
        If ``replacementvalue`` is ``None`` and ``dominatingvalue`` is not
        ``None``, ``replacementvalue = dominatingvalue``.

    threeterm_patterns : tuple, optional
        Tuple of tuples, with (pattern to simplify, simplified pattern) with
        three terms.

    """
    from sympy.core.relational import Relational, _canonical

    if replacementvalue is None and dominatingvalue is not None:
        replacementvalue = dominatingvalue
    # Use replacement patterns for Relationals
    Rel, nonRel = sift(rv.args, lambda i: isinstance(i, Relational),
                       binary=True)
    if len(Rel) <= 1:
        return rv
    Rel, nonRealRel = sift(Rel, lambda i: not any(s.is_real is False
                                                  for s in i.free_symbols),
                           binary=True)
    Rel = [i.canonical for i in Rel]

    if threeterm_patterns and len(Rel) >= 3:
        Rel = _apply_patternbased_threeterm_simplification(Rel,
                            threeterm_patterns, rv.func, dominatingvalue,
                            replacementvalue, measure)

    Rel = _apply_patternbased_twoterm_simplification(Rel, patterns,
                    rv.func, dominatingvalue, replacementvalue, measure)

    rv = rv.func(*([_canonical(i) for i in ordered(Rel)]
                 + nonRel + nonRealRel))
    return rv


def _apply_patternbased_twoterm_simplification(Rel, patterns, func,
                                               dominatingvalue,
                                               replacementvalue,
                                               measure):
    """ Apply pattern-based two-term simplification."""
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core.relational import Ge, Gt, _Inequality
    changed = True
    while changed and len(Rel) >= 2:
        changed = False
        # Use only < or <=
        Rel = [r.reversed if isinstance(r, (Ge, Gt)) else r for r in Rel]
        # Sort based on ordered
        Rel = list(ordered(Rel))
        # Eq and Ne must be tested reversed as well
        rtmp = [(r, ) if isinstance(r, _Inequality) else (r, r.reversed) for r in Rel]
        # Create a list of possible replacements
        results = []
        # Try all combinations of possibly reversed relational
        for ((i, pi), (j, pj)) in combinations(enumerate(rtmp), 2):
            for pattern, simp in patterns:
                res = []
                for p1, p2 in product(pi, pj):
                    # use SymPy matching
                    oldexpr = Tuple(p1, p2)
                    tmpres = oldexpr.match(pattern)
                    if tmpres:
                        res.append((tmpres, oldexpr))

                if res:
                    for tmpres, oldexpr in res:
                        # we have a matching, compute replacement
                        np = simp.xreplace(tmpres)
                        if np == dominatingvalue:
                            # if dominatingvalue, the whole expression
                            # will be replacementvalue
                            return [replacementvalue]
                        # add replacement
                        if not isinstance(np, ITE) and not np.has(Min, Max):
                            # We only want to use ITE and Min/Max replacements if
                            # they simplify to a relational
                            costsaving = measure(func(*oldexpr.args)) - measure(np)
                            if costsaving > 0:
                                results.append((costsaving, ([i, j], np)))
        if results:
            # Sort results based on complexity
            results = sorted(results,
                                           key=lambda pair: pair[0], reverse=True)
            # Replace the one providing most simplification
            replacement = results[0][1]
            idx, newrel = replacement
            idx.sort()
            # Remove the old relationals
            for index in reversed(idx):
                del Rel[index]
            if dominatingvalue is None or newrel != Not(dominatingvalue):
                # Insert the new one (no need to insert a value that will
                # not affect the result)
                if newrel.func == func:
                    for a in newrel.args:
                        Rel.append(a)
                else:
                    Rel.append(newrel)
            # We did change something so try again
            changed = True
    return Rel


def _apply_patternbased_threeterm_simplification(Rel, patterns, func,
                                                 dominatingvalue,
                                                 replacementvalue,
                                                 measure):
    """ Apply pattern-based three-term simplification."""
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core.relational import Le, Lt, _Inequality
    changed = True
    while changed and len(Rel) >= 3:
        changed = False
        # Use only > or >=
        Rel = [r.reversed if isinstance(r, (Le, Lt)) else r for r in Rel]
        # Sort based on ordered
        Rel = list(ordered(Rel))
        # Create a list of possible replacements
        results = []
        # Eq and Ne must be tested reversed as well
        rtmp = [(r, ) if isinstance(r, _Inequality) else (r, r.reversed) for r in Rel]
        # Try all combinations of possibly reversed relational
        for ((i, pi), (j, pj), (k, pk)) in permutations(enumerate(rtmp), 3):
            for pattern, simp in patterns:
                res = []
                for p1, p2, p3 in product(pi, pj, pk):
                    # use SymPy matching
                    oldexpr = Tuple(p1, p2, p3)
                    tmpres = oldexpr.match(pattern)
                    if tmpres:
                        res.append((tmpres, oldexpr))

                if res:
                    for tmpres, oldexpr in res:
                        # we have a matching, compute replacement
                        np = simp.xreplace(tmpres)
                        if np == dominatingvalue:
                            # if dominatingvalue, the whole expression
                            # will be replacementvalue
                            return [replacementvalue]
                        # add replacement
                        if not isinstance(np, ITE) and not np.has(Min, Max):
                            # We only want to use ITE and Min/Max replacements if
                            # they simplify to a relational
                            costsaving = measure(func(*oldexpr.args)) - measure(np)
                            if costsaving > 0:
                                results.append((costsaving, ([i, j, k], np)))
        if results:
            # Sort results based on complexity
            results = sorted(results,
                                           key=lambda pair: pair[0], reverse=True)
            # Replace the one providing most simplification
            replacement = results[0][1]
            idx, newrel = replacement
            idx.sort()
            # Remove the old relationals
            for index in reversed(idx):
                del Rel[index]
            if dominatingvalue is None or newrel != Not(dominatingvalue):
                # Insert the new one (no need to insert a value that will
                # not affect the result)
                if newrel.func == func:
                    for a in newrel.args:
                        Rel.append(a)
                else:
                    Rel.append(newrel)
            # We did change something so try again
            changed = True
    return Rel


@cacheit
def _simplify_patterns_and():
    """ Two-term patterns for And."""

    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.miscellaneous import Min, Max
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    # Relationals patterns should be in alphabetical order
    # (pattern1, pattern2, simplified)
    # Do not use Ge, Gt
    _matchers_and = ((Tuple(Eq(a, b), Lt(a, b)), false),
                     #(Tuple(Eq(a, b), Lt(b, a)), S.false),
                     #(Tuple(Le(b, a), Lt(a, b)), S.false),
                     #(Tuple(Lt(b, a), Le(a, b)), S.false),
                     (Tuple(Lt(b, a), Lt(a, b)), false),
                     (Tuple(Eq(a, b), Le(b, a)), Eq(a, b)),
                     #(Tuple(Eq(a, b), Le(a, b)), Eq(a, b)),
                     #(Tuple(Le(b, a), Lt(b, a)), Gt(a, b)),
                     (Tuple(Le(b, a), Le(a, b)), Eq(a, b)),
                     #(Tuple(Le(b, a), Ne(a, b)), Gt(a, b)),
                     #(Tuple(Lt(b, a), Ne(a, b)), Gt(a, b)),
                     (Tuple(Le(a, b), Lt(a, b)), Lt(a, b)),
                     (Tuple(Le(a, b), Ne(a, b)), Lt(a, b)),
                     (Tuple(Lt(a, b), Ne(a, b)), Lt(a, b)),
                     # Sign
                     (Tuple(Eq(a, b), Eq(a, -b)), And(Eq(a, S.Zero), Eq(b, S.Zero))),
                     # Min/Max/ITE
                     (Tuple(Le(b, a), Le(c, a)), Ge(a, Max(b, c))),
                     (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, Ge(a, b), Gt(a, c))),
                     (Tuple(Lt(b, a), Lt(c, a)), Gt(a, Max(b, c))),
                     (Tuple(Le(a, b), Le(a, c)), Le(a, Min(b, c))),
                     (Tuple(Le(a, b), Lt(a, c)), ITE(b < c, Le(a, b), Lt(a, c))),
                     (Tuple(Lt(a, b), Lt(a, c)), Lt(a, Min(b, c))),
                     (Tuple(Le(a, b), Le(c, a)), ITE(Eq(b, c), Eq(a, b), ITE(b < c, false, And(Le(a, b), Ge(a, c))))),
                     (Tuple(Le(c, a), Le(a, b)), ITE(Eq(b, c), Eq(a, b), ITE(b < c, false, And(Le(a, b), Ge(a, c))))),
                     (Tuple(Lt(a, b), Lt(c, a)), ITE(b < c, false, And(Lt(a, b), Gt(a, c)))),
                     (Tuple(Lt(c, a), Lt(a, b)), ITE(b < c, false, And(Lt(a, b), Gt(a, c)))),
                     (Tuple(Le(a, b), Lt(c, a)), ITE(b <= c, false, And(Le(a, b), Gt(a, c)))),
                     (Tuple(Le(c, a), Lt(a, b)), ITE(b <= c, false, And(Lt(a, b), Ge(a, c)))),
                     (Tuple(Eq(a, b), Eq(a, c)), ITE(Eq(b, c), Eq(a, b), false)),
                     (Tuple(Lt(a, b), Lt(-b, a)), ITE(b > 0, Lt(Abs(a), b), false)),
                     (Tuple(Le(a, b), Le(-b, a)), ITE(b >= 0, Le(Abs(a), b), false)),
                     )
    return _matchers_and


@cacheit
def _simplify_patterns_and3():
    """ Three-term patterns for And."""

    from sympy.core import Wild
    from sympy.core.relational import Eq, Ge, Gt

    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    # Relationals patterns should be in alphabetical order
    # (pattern1, pattern2, pattern3, simplified)
    # Do not use Le, Lt
    _matchers_and = ((Tuple(Ge(a, b), Ge(b, c), Gt(c, a)), false),
                     (Tuple(Ge(a, b), Gt(b, c), Gt(c, a)), false),
                     (Tuple(Gt(a, b), Gt(b, c), Gt(c, a)), false),
                     # (Tuple(Ge(c, a), Gt(a, b), Gt(b, c)), S.false),
                     # Lower bound relations
                     # Commented out combinations that does not simplify
                     (Tuple(Ge(a, b), Ge(a, c), Ge(b, c)), And(Ge(a, b), Ge(b, c))),
                     (Tuple(Ge(a, b), Ge(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))),
                     # (Tuple(Ge(a, b), Gt(a, c), Ge(b, c)), And(Ge(a, b), Ge(b, c))),
                     (Tuple(Ge(a, b), Gt(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))),
                     # (Tuple(Gt(a, b), Ge(a, c), Ge(b, c)), And(Gt(a, b), Ge(b, c))),
                     (Tuple(Ge(a, c), Gt(a, b), Gt(b, c)), And(Gt(a, b), Gt(b, c))),
                     (Tuple(Ge(b, c), Gt(a, b), Gt(a, c)), And(Gt(a, b), Ge(b, c))),
                     (Tuple(Gt(a, b), Gt(a, c), Gt(b, c)), And(Gt(a, b), Gt(b, c))),
                     # Upper bound relations
                     # Commented out combinations that does not simplify
                     (Tuple(Ge(b, a), Ge(c, a), Ge(b, c)), And(Ge(c, a), Ge(b, c))),
                     (Tuple(Ge(b, a), Ge(c, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))),
                     # (Tuple(Ge(b, a), Gt(c, a), Ge(b, c)), And(Gt(c, a), Ge(b, c))),
                     (Tuple(Ge(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))),
                     # (Tuple(Gt(b, a), Ge(c, a), Ge(b, c)), And(Ge(c, a), Ge(b, c))),
                     (Tuple(Ge(c, a), Gt(b, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))),
                     (Tuple(Ge(b, c), Gt(b, a), Gt(c, a)), And(Gt(c, a), Ge(b, c))),
                     (Tuple(Gt(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))),
                     # Circular relation
                     (Tuple(Ge(a, b), Ge(b, c), Ge(c, a)), And(Eq(a, b), Eq(b, c))),
                     )
    return _matchers_and


@cacheit
def _simplify_patterns_or():
    """ Two-term patterns for Or."""

    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.miscellaneous import Min, Max
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    # Relationals patterns should be in alphabetical order
    # (pattern1, pattern2, simplified)
    # Do not use Ge, Gt
    _matchers_or = ((Tuple(Le(b, a), Le(a, b)), true),
                    #(Tuple(Le(b, a), Lt(a, b)), true),
                    (Tuple(Le(b, a), Ne(a, b)), true),
                    #(Tuple(Le(a, b), Lt(b, a)), true),
                    #(Tuple(Le(a, b), Ne(a, b)), true),
                    #(Tuple(Eq(a, b), Le(b, a)), Ge(a, b)),
                    #(Tuple(Eq(a, b), Lt(b, a)), Ge(a, b)),
                    (Tuple(Eq(a, b), Le(a, b)), Le(a, b)),
                    (Tuple(Eq(a, b), Lt(a, b)), Le(a, b)),
                    #(Tuple(Le(b, a), Lt(b, a)), Ge(a, b)),
                    (Tuple(Lt(b, a), Lt(a, b)), Ne(a, b)),
                    (Tuple(Lt(b, a), Ne(a, b)), Ne(a, b)),
                    (Tuple(Le(a, b), Lt(a, b)), Le(a, b)),
                    #(Tuple(Lt(a, b), Ne(a, b)), Ne(a, b)),
                    (Tuple(Eq(a, b), Ne(a, c)), ITE(Eq(b, c), true, Ne(a, c))),
                    (Tuple(Ne(a, b), Ne(a, c)), ITE(Eq(b, c), Ne(a, b), true)),
                    # Min/Max/ITE
                    (Tuple(Le(b, a), Le(c, a)), Ge(a, Min(b, c))),
                    #(Tuple(Ge(b, a), Ge(c, a)), Ge(Min(b, c), a)),
                    (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, Lt(c, a), Le(b, a))),
                    (Tuple(Lt(b, a), Lt(c, a)), Gt(a, Min(b, c))),
                    #(Tuple(Gt(b, a), Gt(c, a)), Gt(Min(b, c), a)),
                    (Tuple(Le(a, b), Le(a, c)), Le(a, Max(b, c))),
                    #(Tuple(Le(b, a), Le(c, a)), Le(Max(b, c), a)),
                    (Tuple(Le(a, b), Lt(a, c)), ITE(b >= c, Le(a, b), Lt(a, c))),
                    (Tuple(Lt(a, b), Lt(a, c)), Lt(a, Max(b, c))),
                    #(Tuple(Lt(b, a), Lt(c, a)), Lt(Max(b, c), a)),
                    (Tuple(Le(a, b), Le(c, a)), ITE(b >= c, true, Or(Le(a, b), Ge(a, c)))),
                    (Tuple(Le(c, a), Le(a, b)), ITE(b >= c, true, Or(Le(a, b), Ge(a, c)))),
                    (Tuple(Lt(a, b), Lt(c, a)), ITE(b > c, true, Or(Lt(a, b), Gt(a, c)))),
                    (Tuple(Lt(c, a), Lt(a, b)), ITE(b > c, true, Or(Lt(a, b), Gt(a, c)))),
                    (Tuple(Le(a, b), Lt(c, a)), ITE(b >= c, true, Or(Le(a, b), Gt(a, c)))),
                    (Tuple(Le(c, a), Lt(a, b)), ITE(b >= c, true, Or(Lt(a, b), Ge(a, c)))),
                    (Tuple(Lt(b, a), Lt(a, -b)), ITE(b >= 0, Gt(Abs(a), b), true)),
                    (Tuple(Le(b, a), Le(a, -b)), ITE(b > 0, Ge(Abs(a), b), true)),
                    )
    return _matchers_or


@cacheit
def _simplify_patterns_xor():
    """ Two-term patterns for Xor."""

    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    # Relationals patterns should be in alphabetical order
    # (pattern1, pattern2, simplified)
    # Do not use Ge, Gt
    _matchers_xor = (#(Tuple(Le(b, a), Lt(a, b)), true),
                     #(Tuple(Lt(b, a), Le(a, b)), true),
                     #(Tuple(Eq(a, b), Le(b, a)), Gt(a, b)),
                     #(Tuple(Eq(a, b), Lt(b, a)), Ge(a, b)),
                     (Tuple(Eq(a, b), Le(a, b)), Lt(a, b)),
                     (Tuple(Eq(a, b), Lt(a, b)), Le(a, b)),
                     (Tuple(Le(a, b), Lt(a, b)), Eq(a, b)),
                     (Tuple(Le(a, b), Le(b, a)), Ne(a, b)),
                     (Tuple(Le(b, a), Ne(a, b)), Le(a, b)),
                     # (Tuple(Lt(b, a), Lt(a, b)), Ne(a, b)),
                     (Tuple(Lt(b, a), Ne(a, b)), Lt(a, b)),
                     # (Tuple(Le(a, b), Lt(a, b)), Eq(a, b)),
                     # (Tuple(Le(a, b), Ne(a, b)), Ge(a, b)),
                     # (Tuple(Lt(a, b), Ne(a, b)), Gt(a, b)),
                     # Min/Max/ITE
                     (Tuple(Le(b, a), Le(c, a)),
                      And(Ge(a, Min(b, c)), Lt(a, Max(b, c)))),
                     (Tuple(Le(b, a), Lt(c, a)),
                      ITE(b > c, And(Gt(a, c), Lt(a, b)),
                          And(Ge(a, b), Le(a, c)))),
                     (Tuple(Lt(b, a), Lt(c, a)),
                      And(Gt(a, Min(b, c)), Le(a, Max(b, c)))),
                     (Tuple(Le(a, b), Le(a, c)),
                      And(Le(a, Max(b, c)), Gt(a, Min(b, c)))),
                     (Tuple(Le(a, b), Lt(a, c)),
                      ITE(b < c, And(Lt(a, c), Gt(a, b)),
                          And(Le(a, b), Ge(a, c)))),
                     (Tuple(Lt(a, b), Lt(a, c)),
                      And(Lt(a, Max(b, c)), Ge(a, Min(b, c)))),
                     )
    return _matchers_xor


def simplify_univariate(expr):
    """return a simplified version of univariate boolean expression, else ``expr``"""
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.core.relational import Eq, Ne
    if not isinstance(expr, BooleanFunction):
        return expr
    if expr.atoms(Eq, Ne):
        return expr
    c = expr
    free = c.free_symbols
    if len(free) != 1:
        return c
    x = free.pop()
    ok, i = Piecewise((0, c), evaluate=False
            )._intervals(x, err_on_Eq=True)
    if not ok:
        return c
    if not i:
        return false
    args = []
    for a, b, _, _ in i:
        if a is S.NegativeInfinity:
            if b is S.Infinity:
                c = true
            else:
                if c.subs(x, b) == True:
                    c = (x <= b)
                else:
                    c = (x < b)
        else:
            incl_a = (c.subs(x, a) == True)
            incl_b = (c.subs(x, b) == True)
            if incl_a and incl_b:
                if b.is_infinite:
                    c = (x >= a)
                else:
                    c = And(a <= x, x <= b)
            elif incl_a:
                c = And(a <= x, x < b)
            elif incl_b:
                if b.is_infinite:
                    c = (x > a)
                else:
                    c = And(a < x, x <= b)
            else:
                c = And(a < x, x < b)
        args.append(c)
    return Or(*args)


# Classes corresponding to logic gates
# Used in gateinputcount method
BooleanGates = (And, Or, Xor, Nand, Nor, Not, Xnor, ITE)

def gateinputcount(expr):
    """
    Return the total number of inputs for the logic gates realizing the
    Boolean expression.

    Returns
    =======

    int
        Number of gate inputs

    Note
    ====

    Not all Boolean functions count as gate here, only those that are
    considered to be standard gates. These are: :py:class:`~.And`,
    :py:class:`~.Or`, :py:class:`~.Xor`, :py:class:`~.Not`, and
    :py:class:`~.ITE` (multiplexer). :py:class:`~.Nand`, :py:class:`~.Nor`,
    and :py:class:`~.Xnor` will be evaluated to ``Not(And())`` etc.

    Examples
    ========

    >>> from sympy.logic import And, Or, Nand, Not, gateinputcount
    >>> from sympy.abc import x, y, z
    >>> expr = And(x, y)
    >>> gateinputcount(expr)
    2
    >>> gateinputcount(Or(expr, z))
    4

    Note that ``Nand`` is automatically evaluated to ``Not(And())`` so

    >>> gateinputcount(Nand(x, y, z))
    4
    >>> gateinputcount(Not(And(x, y, z)))
    4

    Although this can be avoided by using ``evaluate=False``

    >>> gateinputcount(Nand(x, y, z, evaluate=False))
    3

    Also note that a comparison will count as a Boolean variable:

    >>> gateinputcount(And(x > z, y >= 2))
    2

    As will a symbol:
    >>> gateinputcount(x)
    0

    """
    if not isinstance(expr, Boolean):
        raise TypeError("Expression must be Boolean")
    if isinstance(expr, BooleanGates):
        return len(expr.args) + sum(gateinputcount(x) for x in expr.args)
    return 0
