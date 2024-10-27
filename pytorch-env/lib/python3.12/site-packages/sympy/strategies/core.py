""" Generic SymPy-Independent Strategies """
from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout


_S = TypeVar('_S')
_T = TypeVar('_T')


def identity(x: _T) -> _T:
    return x


def exhaust(rule: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """ Apply a rule repeatedly until it has no effect """
    def exhaustive_rl(expr: _T) -> _T:
        new, old = rule(expr), expr
        while new != old:
            new, old = rule(new), new
        return new
    return exhaustive_rl


def memoize(rule: Callable[[_S], _T]) -> Callable[[_S], _T]:
    """Memoized version of a rule

    Notes
    =====

    This cache can grow infinitely, so it is not recommended to use this
    than ``functools.lru_cache`` unless you need very heavy computation.
    """
    cache: dict[_S, _T] = {}

    def memoized_rl(expr: _S) -> _T:
        if expr in cache:
            return cache[expr]
        else:
            result = rule(expr)
            cache[expr] = result
            return result
    return memoized_rl


def condition(
    cond: Callable[[_T], bool], rule: Callable[[_T], _T]
) -> Callable[[_T], _T]:
    """ Only apply rule if condition is true """
    def conditioned_rl(expr: _T) -> _T:
        if cond(expr):
            return rule(expr)
        return expr
    return conditioned_rl


def chain(*rules: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """
    Compose a sequence of rules so that they apply to the expr sequentially
    """
    def chain_rl(expr: _T) -> _T:
        for rule in rules:
            expr = rule(expr)
        return expr
    return chain_rl


def debug(rule, file=None):
    """ Print out before and after expressions each time rule is used """
    if file is None:
        file = stdout

    def debug_rl(*args, **kwargs):
        expr = args[0]
        result = rule(*args, **kwargs)
        if result != expr:
            file.write("Rule: %s\n" % rule.__name__)
            file.write("In:   %s\nOut:  %s\n\n" % (expr, result))
        return result
    return debug_rl


def null_safe(rule: Callable[[_T], _T | None]) -> Callable[[_T], _T]:
    """ Return original expr if rule returns None """
    def null_safe_rl(expr: _T) -> _T:
        result = rule(expr)
        if result is None:
            return expr
        return result
    return null_safe_rl


def tryit(rule: Callable[[_T], _T], exception) -> Callable[[_T], _T]:
    """ Return original expr if rule raises exception """
    def try_rl(expr: _T) -> _T:
        try:
            return rule(expr)
        except exception:
            return expr
    return try_rl


def do_one(*rules: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """ Try each of the rules until one works. Then stop. """
    def do_one_rl(expr: _T) -> _T:
        for rl in rules:
            result = rl(expr)
            if result != expr:
                return result
        return expr
    return do_one_rl


def switch(
    key: Callable[[_S], _T],
    ruledict: Mapping[_T, Callable[[_S], _S]]
) -> Callable[[_S], _S]:
    """ Select a rule based on the result of key called on the function """
    def switch_rl(expr: _S) -> _S:
        rl = ruledict.get(key(expr), identity)
        return rl(expr)
    return switch_rl


# XXX Untyped default argument for minimize function
# where python requires SupportsRichComparison type
def _identity(x):
    return x


def minimize(
    *rules: Callable[[_S], _T],
    objective=_identity
) -> Callable[[_S], _T]:
    """ Select result of rules that minimizes objective

    >>> from sympy.strategies import minimize
    >>> inc = lambda x: x + 1
    >>> dec = lambda x: x - 1
    >>> rl = minimize(inc, dec)
    >>> rl(4)
    3

    >>> rl = minimize(inc, dec, objective=lambda x: -x)  # maximize
    >>> rl(4)
    5
    """
    def minrule(expr: _S) -> _T:
        return min([rule(expr) for rule in rules], key=objective)
    return minrule
