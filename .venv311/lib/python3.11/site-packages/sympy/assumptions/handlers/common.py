"""
This module defines base class for handlers and some core handlers:
``Q.commutative`` and ``Q.is_true``.
"""

from sympy.assumptions import Q, ask, AppliedPredicate
from sympy.core import Basic, Symbol
from sympy.core.logic import _fuzzy_group, fuzzy_and, fuzzy_or
from sympy.core.numbers import NaN, Number
from sympy.logic.boolalg import (And, BooleanTrue, BooleanFalse, conjuncts,
    Equivalent, Implies, Not, Or)
from sympy.utilities.exceptions import sympy_deprecation_warning

from ..predicates.common import CommutativePredicate, IsTruePredicate


class AskHandler:
    """Base class that all Ask Handlers must inherit."""
    def __new__(cls, *args, **kwargs):
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. The AskHandler class should
            be replaced with the multipledispatch handler of Predicate
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
        )
        return super().__new__(cls, *args, **kwargs)


class CommonHandler(AskHandler):
    # Deprecated
    """Defines some useful methods common to most Handlers. """

    @staticmethod
    def AlwaysTrue(expr, assumptions):
        return True

    @staticmethod
    def AlwaysFalse(expr, assumptions):
        return False

    @staticmethod
    def AlwaysNone(expr, assumptions):
        return None

    NaN = AlwaysFalse


# CommutativePredicate

@CommutativePredicate.register(Symbol)
def _(expr, assumptions):
    """Objects are expected to be commutative unless otherwise stated"""
    assumps = conjuncts(assumptions)
    if expr.is_commutative is not None:
        return expr.is_commutative and not ~Q.commutative(expr) in assumps
    if Q.commutative(expr) in assumps:
        return True
    elif ~Q.commutative(expr) in assumps:
        return False
    return True

@CommutativePredicate.register(Basic)
def _(expr, assumptions):
    for arg in expr.args:
        if not ask(Q.commutative(arg), assumptions):
            return False
    return True

@CommutativePredicate.register(Number)
def _(expr, assumptions):
    return True

@CommutativePredicate.register(NaN)
def _(expr, assumptions):
    return True


# IsTruePredicate

@IsTruePredicate.register(bool)
def _(expr, assumptions):
    return expr

@IsTruePredicate.register(BooleanTrue)
def _(expr, assumptions):
    return True

@IsTruePredicate.register(BooleanFalse)
def _(expr, assumptions):
    return False

@IsTruePredicate.register(AppliedPredicate)
def _(expr, assumptions):
    return ask(expr, assumptions)

@IsTruePredicate.register(Not)
def _(expr, assumptions):
    arg = expr.args[0]
    if arg.is_Symbol:
        # symbol used as abstract boolean object
        return None
    value = ask(arg, assumptions=assumptions)
    if value in (True, False):
        return not value
    else:
        return None

@IsTruePredicate.register(Or)
def _(expr, assumptions):
    result = False
    for arg in expr.args:
        p = ask(arg, assumptions=assumptions)
        if p is True:
            return True
        if p is None:
            result = None
    return result

@IsTruePredicate.register(And)
def _(expr, assumptions):
    result = True
    for arg in expr.args:
        p = ask(arg, assumptions=assumptions)
        if p is False:
            return False
        if p is None:
            result = None
    return result

@IsTruePredicate.register(Implies)
def _(expr, assumptions):
    p, q = expr.args
    return ask(~p | q, assumptions=assumptions)

@IsTruePredicate.register(Equivalent)
def _(expr, assumptions):
    p, q = expr.args
    pt = ask(p, assumptions=assumptions)
    if pt is None:
        return None
    qt = ask(q, assumptions=assumptions)
    if qt is None:
        return None
    return pt == qt


#### Helper methods
def test_closed_group(expr, assumptions, key):
    """
    Test for membership in a group with respect
    to the current operation.
    """
    return _fuzzy_group(
        (ask(key(a), assumptions) for a in expr.args), quick_exit=True)

def ask_all(*queries, assumptions):
    return fuzzy_and(
        (ask(query, assumptions) for query in queries))

def ask_any(*queries, assumptions):
    return fuzzy_or(
        (ask(query, assumptions) for query in queries))
