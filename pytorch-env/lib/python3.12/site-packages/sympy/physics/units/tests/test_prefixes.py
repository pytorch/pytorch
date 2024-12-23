from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.physics.units import Quantity, length, meter, W
from sympy.physics.units.prefixes import PREFIXES, Prefix, prefix_unit, kilo, \
    kibi
from sympy.physics.units.systems import SI

x = Symbol('x')


def test_prefix_operations():
    m = PREFIXES['m']
    k = PREFIXES['k']
    M = PREFIXES['M']

    dodeca = Prefix('dodeca', 'dd', 1, base=12)

    assert m * k is S.One
    assert m * W == W / 1000
    assert k * k == M
    assert 1 / m == k
    assert k / m == M

    assert dodeca * dodeca == 144
    assert 1 / dodeca == S.One / 12
    assert k / dodeca == S(1000) / 12
    assert dodeca / dodeca is S.One

    m = Quantity("fake_meter")
    SI.set_quantity_dimension(m, S.One)
    SI.set_quantity_scale_factor(m, S.One)

    assert dodeca * m == 12 * m
    assert dodeca / m == 12 / m

    expr1 = kilo * 3
    assert isinstance(expr1, Mul)
    assert expr1.args == (3, kilo)

    expr2 = kilo * x
    assert isinstance(expr2, Mul)
    assert expr2.args == (x, kilo)

    expr3 = kilo / 3
    assert isinstance(expr3, Mul)
    assert expr3.args == (Rational(1, 3), kilo)
    assert expr3.args == (S.One/3, kilo)

    expr4 = kilo / x
    assert isinstance(expr4, Mul)
    assert expr4.args == (1/x, kilo)


def test_prefix_unit():
    m = Quantity("fake_meter", abbrev="m")
    m.set_global_relative_scale_factor(1, meter)

    pref = {"m": PREFIXES["m"], "c": PREFIXES["c"], "d": PREFIXES["d"]}

    q1 = Quantity("millifake_meter", abbrev="mm")
    q2 = Quantity("centifake_meter", abbrev="cm")
    q3 = Quantity("decifake_meter", abbrev="dm")

    SI.set_quantity_dimension(q1, length)

    SI.set_quantity_scale_factor(q1, PREFIXES["m"])
    SI.set_quantity_scale_factor(q1, PREFIXES["c"])
    SI.set_quantity_scale_factor(q1, PREFIXES["d"])

    res = [q1, q2, q3]

    prefs = prefix_unit(m, pref)
    assert set(prefs) == set(res)
    assert {v.abbrev for v in prefs} == set(symbols("mm,cm,dm"))


def test_bases():
    assert kilo.base == 10
    assert kibi.base == 2


def test_repr():
    assert eval(repr(kilo)) == kilo
    assert eval(repr(kibi)) == kibi
