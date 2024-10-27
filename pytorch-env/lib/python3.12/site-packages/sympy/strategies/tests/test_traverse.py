from sympy.strategies.traverse import (
    top_down, bottom_up, sall, top_down_once, bottom_up_once, basic_fns)
from sympy.strategies.rl import rebuild
from sympy.strategies.util import expr_fns
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Str, Symbol
from sympy.abc import x, y, z


def zero_symbols(expression):
    return S.Zero if isinstance(expression, Symbol) else expression


def test_sall():
    zero_onelevel = sall(zero_symbols)

    assert zero_onelevel(Basic(x, y, Basic(x, z))) == \
        Basic(S(0), S(0), Basic(x, z))


def test_bottom_up():
    _test_global_traversal(bottom_up)
    _test_stop_on_non_basics(bottom_up)


def test_top_down():
    _test_global_traversal(top_down)
    _test_stop_on_non_basics(top_down)


def _test_global_traversal(trav):
    zero_all_symbols = trav(zero_symbols)

    assert zero_all_symbols(Basic(x, y, Basic(x, z))) == \
        Basic(S(0), S(0), Basic(S(0), S(0)))


def _test_stop_on_non_basics(trav):
    def add_one_if_can(expr):
        try:
            return expr + 1
        except TypeError:
            return expr

    expr = Basic(S(1), Str('a'), Basic(S(2), Str('b')))
    expected = Basic(S(2), Str('a'), Basic(S(3), Str('b')))
    rl = trav(add_one_if_can)

    assert rl(expr) == expected


class Basic2(Basic):
    pass


def rl(x):
    if x.args and not isinstance(x.args[0], Integer):
        return Basic2(*x.args)
    return x


def test_top_down_once():
    top_rl = top_down_once(rl)

    assert top_rl(Basic(S(1.0), S(2.0), Basic(S(3), S(4)))) == \
        Basic2(S(1.0), S(2.0), Basic(S(3), S(4)))


def test_bottom_up_once():
    bottom_rl = bottom_up_once(rl)

    assert bottom_rl(Basic(S(1), S(2), Basic(S(3.0), S(4.0)))) == \
        Basic(S(1), S(2), Basic2(S(3.0), S(4.0)))


def test_expr_fns():
    expr = x + y**3
    e = bottom_up(lambda v: v + 1, expr_fns)(expr)
    b = bottom_up(lambda v: Basic.__new__(Add, v, S(1)), basic_fns)(expr)

    assert rebuild(b) == e
