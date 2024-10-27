from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.strategies.branch.traverse import top_down, sall
from sympy.strategies.branch.core import do_one, identity


def inc(x):
    if isinstance(x, Integer):
        yield x + 1


def test_top_down_easy():
    expr = Basic(S(1), S(2))
    expected = Basic(S(2), S(3))
    brl = top_down(inc)

    assert set(brl(expr)) == {expected}


def test_top_down_big_tree():
    expr = Basic(S(1), Basic(S(2)), Basic(S(3), Basic(S(4)), S(5)))
    expected = Basic(S(2), Basic(S(3)), Basic(S(4), Basic(S(5)), S(6)))
    brl = top_down(inc)

    assert set(brl(expr)) == {expected}


def test_top_down_harder_function():
    def split5(x):
        if x == 5:
            yield x - 1
            yield x + 1

    expr = Basic(Basic(S(5), S(6)), S(1))
    expected = {Basic(Basic(S(4), S(6)), S(1)), Basic(Basic(S(6), S(6)), S(1))}
    brl = top_down(split5)

    assert set(brl(expr)) == expected


def test_sall():
    expr = Basic(S(1), S(2))
    expected = Basic(S(2), S(3))
    brl = sall(inc)

    assert list(brl(expr)) == [expected]

    expr = Basic(S(1), S(2), Basic(S(3), S(4)))
    expected = Basic(S(2), S(3), Basic(S(3), S(4)))
    brl = sall(do_one(inc, identity))

    assert list(brl(expr)) == [expected]
