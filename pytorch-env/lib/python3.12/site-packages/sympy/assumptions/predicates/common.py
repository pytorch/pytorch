from sympy.assumptions import Predicate, AppliedPredicate, Q
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.multipledispatch import Dispatcher


class CommutativePredicate(Predicate):
    """
    Commutative predicate.

    Explanation
    ===========

    ``ask(Q.commutative(x))`` is true iff ``x`` commutes with any other
    object with respect to multiplication operation.

    """
    # TODO: Add examples
    name = 'commutative'
    handler = Dispatcher("CommutativeHandler", doc="Handler for key 'commutative'.")


binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}

class IsTruePredicate(Predicate):
    """
    Generic predicate.

    Explanation
    ===========

    ``ask(Q.is_true(x))`` is true iff ``x`` is true. This only makes
    sense if ``x`` is a boolean object.

    Examples
    ========

    >>> from sympy import ask, Q
    >>> from sympy.abc import x, y
    >>> ask(Q.is_true(True))
    True

    Wrapping another applied predicate just returns the applied predicate.

    >>> Q.is_true(Q.even(x))
    Q.even(x)

    Wrapping binary relation classes in SymPy core returns applied binary
    relational predicates.

    >>> from sympy import Eq, Gt
    >>> Q.is_true(Eq(x, y))
    Q.eq(x, y)
    >>> Q.is_true(Gt(x, y))
    Q.gt(x, y)

    Notes
    =====

    This class is designed to wrap the boolean objects so that they can
    behave as if they are applied predicates. Consequently, wrapping another
    applied predicate is unnecessary and thus it just returns the argument.
    Also, binary relation classes in SymPy core have binary predicates to
    represent themselves and thus wrapping them with ``Q.is_true`` converts them
    to these applied predicates.

    """
    name = 'is_true'
    handler = Dispatcher(
        "IsTrueHandler",
        doc="Wrapper allowing to query the truth value of a boolean expression."
    )

    def __call__(self, arg):
        # No need to wrap another predicate
        if isinstance(arg, AppliedPredicate):
            return arg
        # Convert relational predicates instead of wrapping them
        if getattr(arg, "is_Relational", False):
            pred = binrelpreds[type(arg)]
            return pred(*arg.args)
        return super().__call__(arg)
