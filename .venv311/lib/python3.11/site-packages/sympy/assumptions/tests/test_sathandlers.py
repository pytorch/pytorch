from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.logic.boolalg import (And, Or)

from sympy.assumptions.sathandlers import (ClassFactRegistry, allargs,
    anyarg, exactlyonearg,)

x, y, z = symbols('x y z')


def test_class_handler_registry():
    my_handler_registry = ClassFactRegistry()

    # The predicate doesn't matter here, so just pass
    @my_handler_registry.register(Mul)
    def fact1(expr):
        pass
    @my_handler_registry.multiregister(Expr)
    def fact2(expr):
        pass

    assert my_handler_registry[Basic] == (frozenset(), frozenset())
    assert my_handler_registry[Expr] == (frozenset(), frozenset({fact2}))
    assert my_handler_registry[Mul] == (frozenset({fact1}), frozenset({fact2}))


def test_allargs():
    assert allargs(x, Q.zero(x), x*y) == And(Q.zero(x), Q.zero(y))
    assert allargs(x, Q.positive(x) | Q.negative(x), x*y) == And(Q.positive(x) | Q.negative(x), Q.positive(y) | Q.negative(y))


def test_anyarg():
    assert anyarg(x, Q.zero(x), x*y) == Or(Q.zero(x), Q.zero(y))
    assert anyarg(x, Q.positive(x) & Q.negative(x), x*y) == \
        Or(Q.positive(x) & Q.negative(x), Q.positive(y) & Q.negative(y))


def test_exactlyonearg():
    assert exactlyonearg(x, Q.zero(x), x*y) == \
        Or(Q.zero(x) & ~Q.zero(y), Q.zero(y) & ~Q.zero(x))
    assert exactlyonearg(x, Q.zero(x), x*y*z) == \
        Or(Q.zero(x) & ~Q.zero(y) & ~Q.zero(z), Q.zero(y)
        & ~Q.zero(x) & ~Q.zero(z), Q.zero(z) & ~Q.zero(x) & ~Q.zero(y))
    assert exactlyonearg(x, Q.positive(x) | Q.negative(x), x*y) == \
        Or((Q.positive(x) | Q.negative(x)) &
        ~(Q.positive(y) | Q.negative(y)), (Q.positive(y) | Q.negative(y)) &
        ~(Q.positive(x) | Q.negative(x)))
