from sympy.strategies.branch.tools import canon
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S


def posdec(x):
    if isinstance(x, Integer) and x > 0:
        yield x - 1
    else:
        yield x


def branch5(x):
    if isinstance(x, Integer):
        if 0 < x < 5:
            yield x - 1
        elif 5 < x < 10:
            yield x + 1
        elif x == 5:
            yield x + 1
            yield x - 1
        else:
            yield x


def test_zero_ints():
    expr = Basic(S(2), Basic(S(5), S(3)), S(8))
    expected = {Basic(S(0), Basic(S(0), S(0)), S(0))}

    brl = canon(posdec)
    assert set(brl(expr)) == expected


def test_split5():
    expr = Basic(S(2), Basic(S(5), S(3)), S(8))
    expected = {
        Basic(S(0), Basic(S(0), S(0)), S(10)),
        Basic(S(0), Basic(S(10), S(0)), S(10))}

    brl = canon(branch5)
    assert set(brl(expr)) == expected
