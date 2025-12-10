from sympy.core.decorators import _sympifyit
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.singleton import S
from sympy.core.sympify import _sympify

from .sets import Set, FiniteSet, SetKind


class PowerSet(Set):
    r"""A symbolic object representing a power set.

    Parameters
    ==========

    arg : Set
        The set to take power of.

    evaluate : bool
        The flag to control evaluation.

        If the evaluation is disabled for finite sets, it can take
        advantage of using subset test as a membership test.

    Notes
    =====

    Power set `\mathcal{P}(S)` is defined as a set containing all the
    subsets of `S`.

    If the set `S` is a finite set, its power set would have
    `2^{\left| S \right|}` elements, where `\left| S \right|` denotes
    the cardinality of `S`.

    Examples
    ========

    >>> from sympy import PowerSet, S, FiniteSet

    A power set of a finite set:

    >>> PowerSet(FiniteSet(1, 2, 3))
    PowerSet({1, 2, 3})

    A power set of an empty set:

    >>> PowerSet(S.EmptySet)
    PowerSet(EmptySet)
    >>> PowerSet(PowerSet(S.EmptySet))
    PowerSet(PowerSet(EmptySet))

    A power set of an infinite set:

    >>> PowerSet(S.Reals)
    PowerSet(Reals)

    Evaluating the power set of a finite set to its explicit form:

    >>> PowerSet(FiniteSet(1, 2, 3)).rewrite(FiniteSet)
    FiniteSet(EmptySet, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3})

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Power_set

    .. [2] https://en.wikipedia.org/wiki/Axiom_of_power_set
    """
    def __new__(cls, arg, evaluate=None):
        if evaluate is None:
            evaluate=global_parameters.evaluate

        arg = _sympify(arg)

        if not isinstance(arg, Set):
            raise ValueError('{} must be a set.'.format(arg))

        return super().__new__(cls, arg)

    @property
    def arg(self):
        return self.args[0]

    def _eval_rewrite_as_FiniteSet(self, *args, **kwargs):
        arg = self.arg
        if arg.is_FiniteSet:
            return arg.powerset()
        return None

    @_sympifyit('other', NotImplemented)
    def _contains(self, other):
        if not isinstance(other, Set):
            return None

        return fuzzy_bool(self.arg.is_superset(other))

    def _eval_is_subset(self, other):
        if isinstance(other, PowerSet):
            return self.arg.is_subset(other.arg)

    def __len__(self):
        return 2 ** len(self.arg)

    def __iter__(self):
        found = [S.EmptySet]
        yield S.EmptySet

        for x in self.arg:
            temp = []
            x = FiniteSet(x)
            for y in found:
                new = x + y
                yield new
                temp.append(new)
            found.extend(temp)

    @property
    def kind(self):
        return SetKind(self.arg.kind)
