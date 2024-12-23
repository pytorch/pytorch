from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Lambda, BadSignatureError
from sympy.core.logic import fuzzy_bool
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import And, as_Boolean
from sympy.utilities.iterables import sift, flatten, has_dups
from sympy.utilities.exceptions import sympy_deprecation_warning
from .contains import Contains
from .sets import Set, Union, FiniteSet, SetKind


adummy = Dummy('conditionset')


class ConditionSet(Set):
    r"""
    Set of elements which satisfies a given condition.

    .. math:: \{x \mid \textrm{condition}(x) = \texttt{True}, x \in S\}

    Examples
    ========

    >>> from sympy import Symbol, S, ConditionSet, pi, Eq, sin, Interval
    >>> from sympy.abc import x, y, z

    >>> sin_sols = ConditionSet(x, Eq(sin(x), 0), Interval(0, 2*pi))
    >>> 2*pi in sin_sols
    True
    >>> pi/2 in sin_sols
    False
    >>> 3*pi in sin_sols
    False
    >>> 5 in ConditionSet(x, x**2 > 4, S.Reals)
    True

    If the value is not in the base set, the result is false:

    >>> 5 in ConditionSet(x, x**2 > 4, Interval(2, 4))
    False

    Notes
    =====

    Symbols with assumptions should be avoided or else the
    condition may evaluate without consideration of the set:

    >>> n = Symbol('n', negative=True)
    >>> cond = (n > 0); cond
    False
    >>> ConditionSet(n, cond, S.Integers)
    EmptySet

    Only free symbols can be changed by using `subs`:

    >>> c = ConditionSet(x, x < 1, {x, z})
    >>> c.subs(x, y)
    ConditionSet(x, x < 1, {y, z})

    To check if ``pi`` is in ``c`` use:

    >>> pi in c
    False

    If no base set is specified, the universal set is implied:

    >>> ConditionSet(x, x < 1).base_set
    UniversalSet

    Only symbols or symbol-like expressions can be used:

    >>> ConditionSet(x + 1, x + 1 < 1, S.Integers)
    Traceback (most recent call last):
    ...
    ValueError: non-symbol dummy not recognized in condition

    When the base set is a ConditionSet, the symbols will be
    unified if possible with preference for the outermost symbols:

    >>> ConditionSet(x, x < y, ConditionSet(z, z + y < 2, S.Integers))
    ConditionSet(x, (x < y) & (x + y < 2), Integers)

    """
    def __new__(cls, sym, condition, base_set=S.UniversalSet):
        sym = _sympify(sym)
        flat = flatten([sym])
        if has_dups(flat):
            raise BadSignatureError("Duplicate symbols detected")
        base_set = _sympify(base_set)
        if not isinstance(base_set, Set):
            raise TypeError(
                'base set should be a Set object, not %s' % base_set)
        condition = _sympify(condition)

        if isinstance(condition, FiniteSet):
            condition_orig = condition
            temp = (Eq(lhs, 0) for lhs in condition)
            condition = And(*temp)
            sympy_deprecation_warning(
                f"""
Using a set for the condition in ConditionSet is deprecated. Use a boolean
instead.

In this case, replace

    {condition_orig}

with

    {condition}
""",
                deprecated_since_version='1.5',
                active_deprecations_target="deprecated-conditionset-set",
                )

        condition = as_Boolean(condition)

        if condition is S.true:
            return base_set

        if condition is S.false:
            return S.EmptySet

        if base_set is S.EmptySet:
            return S.EmptySet

        # no simple answers, so now check syms
        for i in flat:
            if not getattr(i, '_diff_wrt', False):
                raise ValueError('`%s` is not symbol-like' % i)

        if base_set.contains(sym) is S.false:
            raise TypeError('sym `%s` is not in base_set `%s`' % (sym, base_set))

        know = None
        if isinstance(base_set, FiniteSet):
            sifted = sift(
                base_set, lambda _: fuzzy_bool(condition.subs(sym, _)))
            if sifted[None]:
                know = FiniteSet(*sifted[True])
                base_set = FiniteSet(*sifted[None])
            else:
                return FiniteSet(*sifted[True])

        if isinstance(base_set, cls):
            s, c, b = base_set.args
            def sig(s):
                return cls(s, Eq(adummy, 0)).as_dummy().sym
            sa, sb = map(sig, (sym, s))
            if sa != sb:
                raise BadSignatureError('sym does not match sym of base set')
            reps = dict(zip(flatten([sym]), flatten([s])))
            if s == sym:
                condition = And(condition, c)
                base_set = b
            elif not c.free_symbols & sym.free_symbols:
                reps = {v: k for k, v in reps.items()}
                condition = And(condition, c.xreplace(reps))
                base_set = b
            elif not condition.free_symbols & s.free_symbols:
                sym = sym.xreplace(reps)
                condition = And(condition.xreplace(reps), c)
                base_set = b

        # flatten ConditionSet(Contains(ConditionSet())) expressions
        if isinstance(condition, Contains) and (sym == condition.args[0]):
            if isinstance(condition.args[1], Set):
                return condition.args[1].intersect(base_set)

        rv = Basic.__new__(cls, sym, condition, base_set)
        return rv if know is None else Union(know, rv)

    sym = property(lambda self: self.args[0])
    condition = property(lambda self: self.args[1])
    base_set = property(lambda self: self.args[2])

    @property
    def free_symbols(self):
        cond_syms = self.condition.free_symbols - self.sym.free_symbols
        return cond_syms | self.base_set.free_symbols

    @property
    def bound_symbols(self):
        return flatten([self.sym])

    def _contains(self, other):
        def ok_sig(a, b):
            tuples = [isinstance(i, Tuple) for i in (a, b)]
            c = tuples.count(True)
            if c == 1:
                return False
            if c == 0:
                return True
            return len(a) == len(b) and all(
                ok_sig(i, j) for i, j in zip(a, b))
        if not ok_sig(self.sym, other):
            return S.false

        # try doing base_cond first and return
        # False immediately if it is False
        base_cond = Contains(other, self.base_set)
        if base_cond is S.false:
            return S.false

        # Substitute other into condition. This could raise e.g. for
        # ConditionSet(x, 1/x >= 0, Reals).contains(0)
        lamda = Lambda((self.sym,), self.condition)
        try:
            lambda_cond = lamda(other)
        except TypeError:
            return None
        else:
            return And(base_cond, lambda_cond)

    def as_relational(self, other):
        f = Lambda(self.sym, self.condition)
        if isinstance(self.sym, Tuple):
            f = f(*other)
        else:
            f = f(other)
        return And(f, self.base_set.contains(other))

    def _eval_subs(self, old, new):
        sym, cond, base = self.args
        dsym = sym.subs(old, adummy)
        insym = dsym.has(adummy)
        # prioritize changing a symbol in the base
        newbase = base.subs(old, new)
        if newbase != base:
            if not insym:
                cond = cond.subs(old, new)
            return self.func(sym, cond, newbase)
        if insym:
            pass  # no change of bound symbols via subs
        elif getattr(new, '_diff_wrt', False):
            cond = cond.subs(old, new)
        else:
            pass  # let error about the symbol raise from __new__
        return self.func(sym, cond, base)

    def _kind(self):
        return SetKind(self.sym.kind)
