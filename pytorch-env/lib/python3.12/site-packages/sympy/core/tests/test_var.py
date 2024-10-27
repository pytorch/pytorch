from sympy.core.function import (Function, FunctionClass)
from sympy.core.symbol import (Symbol, var)
from sympy.testing.pytest import raises

def test_var():
    ns = {"var": var, "raises": raises}
    eval("var('a')", ns)
    assert ns["a"] == Symbol("a")

    eval("var('b bb cc zz _x')", ns)
    assert ns["b"] == Symbol("b")
    assert ns["bb"] == Symbol("bb")
    assert ns["cc"] == Symbol("cc")
    assert ns["zz"] == Symbol("zz")
    assert ns["_x"] == Symbol("_x")

    v = eval("var(['d', 'e', 'fg'])", ns)
    assert ns['d'] == Symbol('d')
    assert ns['e'] == Symbol('e')
    assert ns['fg'] == Symbol('fg')

# check return value
    assert v != ['d', 'e', 'fg']
    assert v == [Symbol('d'), Symbol('e'), Symbol('fg')]


def test_var_return():
    ns = {"var": var, "raises": raises}
    "raises(ValueError, lambda: var(''))"
    v2 = eval("var('q')", ns)
    v3 = eval("var('q p')", ns)

    assert v2 == Symbol('q')
    assert v3 == (Symbol('q'), Symbol('p'))


def test_var_accepts_comma():
    ns = {"var": var}
    v1 = eval("var('x y z')", ns)
    v2 = eval("var('x,y,z')", ns)
    v3 = eval("var('x,y z')", ns)

    assert v1 == v2
    assert v1 == v3


def test_var_keywords():
    ns = {"var": var}
    eval("var('x y', real=True)", ns)
    assert ns['x'].is_real and ns['y'].is_real


def test_var_cls():
    ns = {"var": var, "Function": Function}
    eval("var('f', cls=Function)", ns)

    assert isinstance(ns['f'], FunctionClass)

    eval("var('g,h', cls=Function)", ns)

    assert isinstance(ns['g'], FunctionClass)
    assert isinstance(ns['h'], FunctionClass)
