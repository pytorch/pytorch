from sympy.core.numbers import E
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import Order
from sympy.abc import x, y


def test_sin():
    e = sin(x).lseries(x)
    assert next(e) == x
    assert next(e) == -x**3/6
    assert next(e) == x**5/120


def test_cos():
    e = cos(x).lseries(x)
    assert next(e) == 1
    assert next(e) == -x**2/2
    assert next(e) == x**4/24


def test_exp():
    e = exp(x).lseries(x)
    assert next(e) == 1
    assert next(e) == x
    assert next(e) == x**2/2
    assert next(e) == x**3/6


def test_exp2():
    e = exp(cos(x)).lseries(x)
    assert next(e) == E
    assert next(e) == -E*x**2/2
    assert next(e) == E*x**4/6
    assert next(e) == -31*E*x**6/720


def test_simple():
    assert list(x.lseries()) == [x]
    assert list(S.One.lseries(x)) == [1]
    assert not next((x/(x + y)).lseries(y)).has(Order)


def test_issue_5183():
    s = (x + 1/x).lseries()
    assert list(s) == [1/x, x]
    assert next((x + x**2).lseries()) == x
    assert next(((1 + x)**7).lseries(x)) == 1
    assert next((sin(x + y)).series(x, n=3).lseries(y)) == x
    # it would be nice if all terms were grouped, but in the
    # following case that would mean that all the terms would have
    # to be known since, for example, every term has a constant in it.
    s = ((1 + x)**7).series(x, 1, n=None)
    assert [next(s) for i in range(2)] == [128, -448 + 448*x]


def test_issue_6999():
    s = tanh(x).lseries(x, 1)
    assert next(s) == tanh(1)
    assert next(s) == x - (x - 1)*tanh(1)**2 - 1
    assert next(s) == -(x - 1)**2*tanh(1) + (x - 1)**2*tanh(1)**3
    assert next(s) == -(x - 1)**3*tanh(1)**4 - (x - 1)**3/3 + \
        4*(x - 1)**3*tanh(1)**2/3
