from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher

class FinitePredicate(Predicate):
    """
    Finite number predicate.

    Explanation
    ===========

    ``Q.finite(x)`` is true if ``x`` is a number but neither an infinity
    nor a ``NaN``. In other words, ``ask(Q.finite(x))`` is true for all
    numerical ``x`` having a bounded absolute value.

    Examples
    ========

    >>> from sympy import Q, ask, S, oo, I, zoo
    >>> from sympy.abc import x
    >>> ask(Q.finite(oo))
    False
    >>> ask(Q.finite(-oo))
    False
    >>> ask(Q.finite(zoo))
    False
    >>> ask(Q.finite(1))
    True
    >>> ask(Q.finite(2 + 3*I))
    True
    >>> ask(Q.finite(x), Q.positive(x))
    True
    >>> print(ask(Q.finite(S.NaN)))
    None

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Finite

    """
    name = 'finite'
    handler = Dispatcher(
        "FiniteHandler",
        doc=("Handler for Q.finite. Test that an expression is bounded respect"
        " to all its variables.")
    )


class InfinitePredicate(Predicate):
    """
    Infinite number predicate.

    ``Q.infinite(x)`` is true iff the absolute value of ``x`` is
    infinity.

    """
    # TODO: Add examples
    name = 'infinite'
    handler = Dispatcher(
        "InfiniteHandler",
        doc="""Handler for Q.infinite key."""
    )


class PositiveInfinitePredicate(Predicate):
    """
    Positive infinity predicate.

    ``Q.positive_infinite(x)`` is true iff ``x`` is positive infinity ``oo``.
    """
    name = 'positive_infinite'
    handler = Dispatcher("PositiveInfiniteHandler")


class NegativeInfinitePredicate(Predicate):
    """
    Negative infinity predicate.

    ``Q.negative_infinite(x)`` is true iff ``x`` is negative infinity ``-oo``.
    """
    name = 'negative_infinite'
    handler = Dispatcher("NegativeInfiniteHandler")
