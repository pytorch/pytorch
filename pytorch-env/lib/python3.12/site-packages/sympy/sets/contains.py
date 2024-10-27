from sympy.core import S
from sympy.core.sympify import sympify
from sympy.core.relational import Eq, Ne
from sympy.core.parameters import global_parameters
from sympy.logic.boolalg import Boolean
from sympy.utilities.misc import func_name
from .sets import Set


class Contains(Boolean):
    """
    Asserts that x is an element of the set S.

    Examples
    ========

    >>> from sympy import Symbol, Integer, S, Contains
    >>> Contains(Integer(2), S.Integers)
    True
    >>> Contains(Integer(-2), S.Naturals)
    False
    >>> i = Symbol('i', integer=True)
    >>> Contains(i, S.Naturals)
    Contains(i, Naturals)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Element_%28mathematics%29
    """
    def __new__(cls, x, s, evaluate=None):
        x = sympify(x)
        s = sympify(s)

        if evaluate is None:
            evaluate = global_parameters.evaluate

        if not isinstance(s, Set):
            raise TypeError('expecting Set, not %s' % func_name(s))

        if evaluate:
            # _contains can return symbolic booleans that would be returned by
            # s.contains(x) but here for Contains(x, s) we only evaluate to
            # true, false or return the unevaluated Contains.
            result = s._contains(x)

            if isinstance(result, Boolean):
                if result in (S.true, S.false):
                    return result
            elif result is not None:
                raise TypeError("_contains() should return Boolean or None")

        return super().__new__(cls, x, s)

    @property
    def binary_symbols(self):
        return set().union(*[i.binary_symbols
            for i in self.args[1].args
            if i.is_Boolean or i.is_Symbol or
            isinstance(i, (Eq, Ne))])

    def as_set(self):
        return self.args[1]
