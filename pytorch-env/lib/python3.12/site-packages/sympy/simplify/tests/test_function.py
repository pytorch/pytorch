""" Unit tests for Hyper_Function"""
from sympy.core import symbols, Dummy, Tuple, S, Rational
from sympy.functions import hyper

from sympy.simplify.hyperexpand import Hyper_Function

def test_attrs():
    a, b = symbols('a, b', cls=Dummy)
    f = Hyper_Function([2, a], [b])
    assert f.ap == Tuple(2, a)
    assert f.bq == Tuple(b)
    assert f.args == (Tuple(2, a), Tuple(b))
    assert f.sizes == (2, 1)

def test_call():
    a, b, x = symbols('a, b, x', cls=Dummy)
    f = Hyper_Function([2, a], [b])
    assert f(x) == hyper([2, a], [b], x)

def test_has():
    a, b, c = symbols('a, b, c', cls=Dummy)
    f = Hyper_Function([2, -a], [b])
    assert f.has(a)
    assert f.has(Tuple(b))
    assert not f.has(c)

def test_eq():
    assert Hyper_Function([1], []) == Hyper_Function([1], [])
    assert (Hyper_Function([1], []) != Hyper_Function([1], [])) is False
    assert Hyper_Function([1], []) != Hyper_Function([2], [])
    assert Hyper_Function([1], []) != Hyper_Function([1, 2], [])
    assert Hyper_Function([1], []) != Hyper_Function([1], [2])

def test_gamma():
    assert Hyper_Function([2, 3], [-1]).gamma == 0
    assert Hyper_Function([-2, -3], [-1]).gamma == 2
    n = Dummy(integer=True)
    assert Hyper_Function([-1, n, 1], []).gamma == 1
    assert Hyper_Function([-1, -n, 1], []).gamma == 1
    p = Dummy(integer=True, positive=True)
    assert Hyper_Function([-1, p, 1], []).gamma == 1
    assert Hyper_Function([-1, -p, 1], []).gamma == 2

def test_suitable_origin():
    assert Hyper_Function((S.Half,), (Rational(3, 2),))._is_suitable_origin() is True
    assert Hyper_Function((S.Half,), (S.Half,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (Rational(-1, 2),))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (0,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (-1, 1,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half, 0), (1,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half, 1),
            (2, Rational(-2, 3)))._is_suitable_origin() is True
    assert Hyper_Function((S.Half, 1),
            (2, Rational(-2, 3), Rational(3, 2)))._is_suitable_origin() is True
