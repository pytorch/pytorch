from sympy.core.numbers import Rational
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.exponential import exp


def test_equal():
    b = Symbol("b")
    a = Symbol("a")
    e1 = a + b
    e2 = 2*a*b
    e3 = a**3*b**2
    e4 = a*b + b*a
    assert not e1 == e2
    assert not e1 == e2
    assert e1 != e2
    assert e2 == e4
    assert e2 != e3
    assert not e2 == e3

    x = Symbol("x")
    e1 = exp(x + 1/x)
    y = Symbol("x")
    e2 = exp(y + 1/y)
    assert e1 == e2
    assert not e1 != e2
    y = Symbol("y")
    e2 = exp(y + 1/y)
    assert not e1 == e2
    assert e1 != e2

    e5 = Rational(3) + 2*x - x - x
    assert e5 == 3
    assert 3 == e5
    assert e5 != 4
    assert 4 != e5
    assert e5 != 3 + x
    assert 3 + x != e5


def test_expevalbug():
    x = Symbol("x")
    e1 = exp(1*x)
    e3 = exp(x)
    assert e1 == e3


def test_cmp_bug1():
    class T:
        pass

    t = T()
    x = Symbol("x")

    assert not (x == t)
    assert (x != t)


def test_cmp_bug2():
    class T:
        pass

    t = T()

    assert not (Symbol == t)
    assert (Symbol != t)


def test_cmp_issue_4357():
    """ Check that Basic subclasses can be compared with sympifiable objects.

    https://github.com/sympy/sympy/issues/4357
    """
    assert not (Symbol == 1)
    assert (Symbol != 1)
    assert not (Symbol == 'x')
    assert (Symbol != 'x')


def test_dummy_eq():
    x = Symbol('x')
    y = Symbol('y')

    u = Dummy('u')

    assert (u**2 + 1).dummy_eq(x**2 + 1) is True
    assert ((u**2 + 1) == (x**2 + 1)) is False

    assert (u**2 + y).dummy_eq(x**2 + y, x) is True
    assert (u**2 + y).dummy_eq(x**2 + y, y) is False
