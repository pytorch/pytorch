"""
General binary relations.
"""
from typing import Optional

from sympy.core.singleton import S
from sympy.assumptions import AppliedPredicate, ask, Predicate, Q  # type: ignore
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.boolalg import conjuncts, Not

__all__ = ["BinaryRelation", "AppliedBinaryRelation"]


class BinaryRelation(Predicate):
    """
    Base class for all binary relational predicates.

    Explanation
    ===========

    Binary relation takes two arguments and returns ``AppliedBinaryRelation``
    instance. To evaluate it to boolean value, use :obj:`~.ask()` or
    :obj:`~.refine()` function.

    You can add support for new types by registering the handler to dispatcher.
    See :obj:`~.Predicate()` for more information about predicate dispatching.

    Examples
    ========

    Applying and evaluating to boolean value:

    >>> from sympy import Q, ask, sin, cos
    >>> from sympy.abc import x
    >>> Q.eq(sin(x)**2+cos(x)**2, 1)
    Q.eq(sin(x)**2 + cos(x)**2, 1)
    >>> ask(_)
    True

    You can define a new binary relation by subclassing and dispatching.
    Here, we define a relation $R$ such that $x R y$ returns true if
    $x = y + 1$.

    >>> from sympy import ask, Number, Q
    >>> from sympy.assumptions import BinaryRelation
    >>> class MyRel(BinaryRelation):
    ...     name = "R"
    ...     is_reflexive = False
    >>> Q.R = MyRel()
    >>> @Q.R.register(Number, Number)
    ... def _(n1, n2, assumptions):
    ...     return ask(Q.zero(n1 - n2 - 1), assumptions)
    >>> Q.R(2, 1)
    Q.R(2, 1)

    Now, we can use ``ask()`` to evaluate it to boolean value.

    >>> ask(Q.R(2, 1))
    True
    >>> ask(Q.R(1, 2))
    False

    ``Q.R`` returns ``False`` with minimum cost if two arguments have same
    structure because it is antireflexive relation [1] by
    ``is_reflexive = False``.

    >>> ask(Q.R(x, x))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Reflexive_relation
    """

    is_reflexive: Optional[bool] = None
    is_symmetric: Optional[bool] = None

    def __call__(self, *args):
        if not len(args) == 2:
            raise ValueError("Binary relation takes two arguments, but got %s." % len(args))
        return AppliedBinaryRelation(self, *args)

    @property
    def reversed(self):
        if self.is_symmetric:
            return self
        return None

    @property
    def negated(self):
        return None

    def _compare_reflexive(self, lhs, rhs):
        # quick exit for structurally same arguments
        # do not check != here because it cannot catch the
        # equivalent arguments with different structures.

        # reflexivity does not hold to NaN
        if lhs is S.NaN or rhs is S.NaN:
            return None

        reflexive = self.is_reflexive
        if reflexive is None:
            pass
        elif reflexive and (lhs == rhs):
            return True
        elif not reflexive and (lhs == rhs):
            return False
        return None

    def eval(self, args, assumptions=True):
        # quick exit for structurally same arguments
        ret = self._compare_reflexive(*args)
        if ret is not None:
            return ret

        # don't perform simplify on args here. (done by AppliedBinaryRelation._eval_ask)
        # evaluate by multipledispatch
        lhs, rhs = args
        ret = self.handler(lhs, rhs, assumptions=assumptions)
        if ret is not None:
            return ret

        # check reversed order if the relation is reflexive
        if self.is_reflexive:
            types = (type(lhs), type(rhs))
            if self.handler.dispatch(*types) is not self.handler.dispatch(*reversed(types)):
                ret = self.handler(rhs, lhs, assumptions=assumptions)

        return ret


class AppliedBinaryRelation(AppliedPredicate):
    """
    The class of expressions resulting from applying ``BinaryRelation``
    to the arguments.

    """

    @property
    def lhs(self):
        """The left-hand side of the relation."""
        return self.arguments[0]

    @property
    def rhs(self):
        """The right-hand side of the relation."""
        return self.arguments[1]

    @property
    def reversed(self):
        """
        Try to return the relationship with sides reversed.
        """
        revfunc = self.function.reversed
        if revfunc is None:
            return self
        return revfunc(self.rhs, self.lhs)

    @property
    def reversedsign(self):
        """
        Try to return the relationship with signs reversed.
        """
        revfunc = self.function.reversed
        if revfunc is None:
            return self
        if not any(side.kind is BooleanKind for side in self.arguments):
            return revfunc(-self.lhs, -self.rhs)
        return self

    @property
    def negated(self):
        neg_rel = self.function.negated
        if neg_rel is None:
            return Not(self, evaluate=False)
        return neg_rel(*self.arguments)

    def _eval_ask(self, assumptions):
        conj_assumps = set()
        binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
        for a in conjuncts(assumptions):
            if a.func in binrelpreds:
                conj_assumps.add(binrelpreds[type(a)](*a.args))
            else:
                conj_assumps.add(a)

        # After CNF in assumptions module is modified to take polyadic
        # predicate, this will be removed
        if any(rel in conj_assumps for rel in (self, self.reversed)):
            return True
        neg_rels = (self.negated, self.reversed.negated, Not(self, evaluate=False),
            Not(self.reversed, evaluate=False))
        if any(rel in conj_assumps for rel in neg_rels):
            return False

        # evaluation using multipledispatching
        ret = self.function.eval(self.arguments, assumptions)
        if ret is not None:
            return ret

        # simplify the args and try again
        args = tuple(a.simplify() for a in self.arguments)
        return self.function.eval(args, assumptions)

    def __bool__(self):
        ret = ask(self)
        if ret is None:
            raise TypeError("Cannot determine truth value of %s" % self)
        return ret
