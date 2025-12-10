from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import (Symbol, symbols)

from sympy.testing.pytest import XFAIL

class SymbolInMulOnce(Symbol):
    # Test class for a symbol that can only appear once in a `Mul` expression.
    pass


Basic._constructor_postprocessor_mapping[SymbolInMulOnce] = {
    "Mul": [lambda x: x],
    "Pow": [lambda x: x.base if isinstance(x.base, SymbolInMulOnce) else x],
    "Add": [lambda x: x],
}


def _postprocess_SymbolRemovesOtherSymbols(expr):
    args = tuple(i for i in expr.args if not isinstance(i, Symbol) or isinstance(i, SymbolRemovesOtherSymbols))
    if args == expr.args:
        return expr
    return Mul.fromiter(args)


class SymbolRemovesOtherSymbols(Symbol):
    # Test class for a symbol that removes other symbols in `Mul`.
    pass

Basic._constructor_postprocessor_mapping[SymbolRemovesOtherSymbols] = {
    "Mul": [_postprocess_SymbolRemovesOtherSymbols],
}

class SubclassSymbolInMulOnce(SymbolInMulOnce):
    pass

class SubclassSymbolRemovesOtherSymbols(SymbolRemovesOtherSymbols):
    pass


def test_constructor_postprocessors1():
    x = SymbolInMulOnce("x")
    y = SymbolInMulOnce("y")
    assert isinstance(3*x, Mul)
    assert (3*x).args == (3, x)
    assert x*x == x
    assert 3*x*x == 3*x
    assert 2*x*x + x == 3*x
    assert x**3*y*y == x*y
    assert x**5 + y*x**3 == x + x*y

    w = SymbolRemovesOtherSymbols("w")
    assert x*w == w
    assert (3*w).args == (3, w)
    assert set((w + x).args) == {x, w}

def test_constructor_postprocessors2():
    x = SubclassSymbolInMulOnce("x")
    y = SubclassSymbolInMulOnce("y")
    assert isinstance(3*x, Mul)
    assert (3*x).args == (3, x)
    assert x*x == x
    assert 3*x*x == 3*x
    assert 2*x*x + x == 3*x
    assert x**3*y*y == x*y
    assert x**5 + y*x**3 == x + x*y

    w = SubclassSymbolRemovesOtherSymbols("w")
    assert x*w == w
    assert (3*w).args == (3, w)
    assert set((w + x).args) == {x, w}


@XFAIL
def test_subexpression_postprocessors():
    # The postprocessors used to work with subexpressions, but the
    # functionality was removed. See #15948.
    a = symbols("a")
    x = SymbolInMulOnce("x")
    w = SymbolRemovesOtherSymbols("w")
    assert 3*a*w**2 == 3*w**2
    assert 3*a*x**3*w**2 == 3*w**2

    x = SubclassSymbolInMulOnce("x")
    w = SubclassSymbolRemovesOtherSymbols("w")
    assert 3*a*w**2 == 3*w**2
    assert 3*a*x**3*w**2 == 3*w**2
