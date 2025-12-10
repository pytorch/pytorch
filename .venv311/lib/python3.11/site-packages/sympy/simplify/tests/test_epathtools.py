"""Tests for tools for manipulation of expressions using paths. """

from sympy.simplify.epathtools import epath, EPath
from sympy.testing.pytest import raises

from sympy.core.numbers import E
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.abc import x, y, z, t


def test_epath_select():
    expr = [((x, 1, t), 2), ((3, y, 4), z)]

    assert epath("/*", expr) == [((x, 1, t), 2), ((3, y, 4), z)]
    assert epath("/*/*", expr) == [(x, 1, t), 2, (3, y, 4), z]
    assert epath("/*/*/*", expr) == [x, 1, t, 3, y, 4]
    assert epath("/*/*/*/*", expr) == []

    assert epath("/[:]", expr) == [((x, 1, t), 2), ((3, y, 4), z)]
    assert epath("/[:]/[:]", expr) == [(x, 1, t), 2, (3, y, 4), z]
    assert epath("/[:]/[:]/[:]", expr) == [x, 1, t, 3, y, 4]
    assert epath("/[:]/[:]/[:]/[:]", expr) == []

    assert epath("/*/[:]", expr) == [(x, 1, t), 2, (3, y, 4), z]

    assert epath("/*/[0]", expr) == [(x, 1, t), (3, y, 4)]
    assert epath("/*/[1]", expr) == [2, z]
    assert epath("/*/[2]", expr) == []

    assert epath("/*/int", expr) == [2]
    assert epath("/*/Symbol", expr) == [z]
    assert epath("/*/tuple", expr) == [(x, 1, t), (3, y, 4)]
    assert epath("/*/__iter__?", expr) == [(x, 1, t), (3, y, 4)]

    assert epath("/*/int|tuple", expr) == [(x, 1, t), 2, (3, y, 4)]
    assert epath("/*/Symbol|tuple", expr) == [(x, 1, t), (3, y, 4), z]
    assert epath("/*/int|Symbol|tuple", expr) == [(x, 1, t), 2, (3, y, 4), z]

    assert epath("/*/int|__iter__?", expr) == [(x, 1, t), 2, (3, y, 4)]
    assert epath("/*/Symbol|__iter__?", expr) == [(x, 1, t), (3, y, 4), z]
    assert epath(
        "/*/int|Symbol|__iter__?", expr) == [(x, 1, t), 2, (3, y, 4), z]

    assert epath("/*/[0]/int", expr) == [1, 3, 4]
    assert epath("/*/[0]/Symbol", expr) == [x, t, y]

    assert epath("/*/[0]/int[1:]", expr) == [1, 4]
    assert epath("/*/[0]/Symbol[1:]", expr) == [t, y]

    assert epath("/Symbol", x + y + z + 1) == [x, y, z]
    assert epath("/*/*/Symbol", t + sin(x + 1) + cos(x + y + E)) == [x, x, y]


def test_epath_apply():
    expr = [((x, 1, t), 2), ((3, y, 4), z)]
    func = lambda expr: expr**2

    assert epath("/*", expr, list) == [[(x, 1, t), 2], [(3, y, 4), z]]

    assert epath("/*/[0]", expr, list) == [([x, 1, t], 2), ([3, y, 4], z)]
    assert epath("/*/[1]", expr, func) == [((x, 1, t), 4), ((3, y, 4), z**2)]
    assert epath("/*/[2]", expr, list) == expr

    assert epath("/*/[0]/int", expr, func) == [((x, 1, t), 2), ((9, y, 16), z)]
    assert epath("/*/[0]/Symbol", expr, func) == [((x**2, 1, t**2), 2),
                 ((3, y**2, 4), z)]
    assert epath(
        "/*/[0]/int[1:]", expr, func) == [((x, 1, t), 2), ((3, y, 16), z)]
    assert epath("/*/[0]/Symbol[1:]", expr, func) == [((x, 1, t**2),
                 2), ((3, y**2, 4), z)]

    assert epath("/Symbol", x + y + z + 1, func) == x**2 + y**2 + z**2 + 1
    assert epath("/*/*/Symbol", t + sin(x + 1) + cos(x + y + E), func) == \
        t + sin(x**2 + 1) + cos(x**2 + y**2 + E)


def test_EPath():
    assert EPath("/*/[0]")._path == "/*/[0]"
    assert EPath(EPath("/*/[0]"))._path == "/*/[0]"
    assert isinstance(epath("/*/[0]"), EPath) is True

    assert repr(EPath("/*/[0]")) == "EPath('/*/[0]')"

    raises(ValueError, lambda: EPath(""))
    raises(ValueError, lambda: EPath("/"))
    raises(ValueError, lambda: EPath("/|x"))
    raises(ValueError, lambda: EPath("/["))
    raises(ValueError, lambda: EPath("/[0]%"))

    raises(NotImplementedError, lambda: EPath("Symbol"))
