from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher


class PrimePredicate(Predicate):
    """
    Prime number predicate.

    Explanation
    ===========

    ``ask(Q.prime(x))`` is true iff ``x`` is a natural number greater
    than 1 that has no positive divisors other than ``1`` and the
    number itself.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> ask(Q.prime(0))
    False
    >>> ask(Q.prime(1))
    False
    >>> ask(Q.prime(2))
    True
    >>> ask(Q.prime(20))
    False
    >>> ask(Q.prime(-3))
    False

    """
    name = 'prime'
    handler = Dispatcher(
        "PrimeHandler",
        doc=("Handler for key 'prime'. Test that an expression represents a prime"
        " number. When the expression is an exact number, the result (when True)"
        " is subject to the limitations of isprime() which is used to return the "
        "result.")
    )


class CompositePredicate(Predicate):
    """
    Composite number predicate.

    Explanation
    ===========

    ``ask(Q.composite(x))`` is true iff ``x`` is a positive integer and has
    at least one positive divisor other than ``1`` and the number itself.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> ask(Q.composite(0))
    False
    >>> ask(Q.composite(1))
    False
    >>> ask(Q.composite(2))
    False
    >>> ask(Q.composite(20))
    True

    """
    name = 'composite'
    handler = Dispatcher("CompositeHandler", doc="Handler for key 'composite'.")


class EvenPredicate(Predicate):
    """
    Even number predicate.

    Explanation
    ===========

    ``ask(Q.even(x))`` is true iff ``x`` belongs to the set of even
    integers.

    Examples
    ========

    >>> from sympy import Q, ask, pi
    >>> ask(Q.even(0))
    True
    >>> ask(Q.even(2))
    True
    >>> ask(Q.even(3))
    False
    >>> ask(Q.even(pi))
    False

    """
    name = 'even'
    handler = Dispatcher("EvenHandler", doc="Handler for key 'even'.")


class OddPredicate(Predicate):
    """
    Odd number predicate.

    Explanation
    ===========

    ``ask(Q.odd(x))`` is true iff ``x`` belongs to the set of odd numbers.

    Examples
    ========

    >>> from sympy import Q, ask, pi
    >>> ask(Q.odd(0))
    False
    >>> ask(Q.odd(2))
    False
    >>> ask(Q.odd(3))
    True
    >>> ask(Q.odd(pi))
    False

    """
    name = 'odd'
    handler = Dispatcher(
        "OddHandler",
        doc=("Handler for key 'odd'. Test that an expression represents an odd"
        " number.")
    )
