from __future__ import annotations
from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.strategies.core import (
    null_safe, exhaust, memoize, condition,
    chain, tryit, do_one, debug, switch, minimize)
from io import StringIO


def posdec(x: int) -> int:
    if x > 0:
        return x - 1
    return x


def inc(x: int) -> int:
    return x + 1


def dec(x: int) -> int:
    return x - 1


def test_null_safe():
    def rl(expr: int) -> int | None:
        if expr == 1:
            return 2
        return None

    safe_rl = null_safe(rl)
    assert rl(1) == safe_rl(1)
    assert rl(3) is None
    assert safe_rl(3) == 3


def test_exhaust():
    sink = exhaust(posdec)
    assert sink(5) == 0
    assert sink(10) == 0


def test_memoize():
    rl = memoize(posdec)
    assert rl(5) == posdec(5)
    assert rl(5) == posdec(5)
    assert rl(-2) == posdec(-2)


def test_condition():
    rl = condition(lambda x: x % 2 == 0, posdec)
    assert rl(5) == 5
    assert rl(4) == 3


def test_chain():
    rl = chain(posdec, posdec)
    assert rl(5) == 3
    assert rl(1) == 0


def test_tryit():
    def rl(expr: Basic) -> Basic:
        assert False

    safe_rl = tryit(rl, AssertionError)
    assert safe_rl(S(1)) == S(1)


def test_do_one():
    rl = do_one(posdec, posdec)
    assert rl(5) == 4

    def rl1(x: int) -> int:
        if x == 1:
            return 2
        return x

    def rl2(x: int) -> int:
        if x == 2:
            return 3
        return x

    rule = do_one(rl1, rl2)
    assert rule(1) == 2
    assert rule(rule(1)) == 3


def test_debug():
    file = StringIO()
    rl = debug(posdec, file)
    rl(5)
    log = file.getvalue()
    file.close()

    assert posdec.__name__ in log
    assert '5' in log
    assert '4' in log


def test_switch():
    def key(x: int) -> int:
        return x % 3

    rl = switch(key, {0: inc, 1: dec})
    assert rl(3) == 4
    assert rl(4) == 3
    assert rl(5) == 5


def test_minimize():
    def key(x: int) -> int:
        return -x

    rl = minimize(inc, dec)
    assert rl(4) == 3

    rl = minimize(inc, dec, objective=key)
    assert rl(4) == 5
