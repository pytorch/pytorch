from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
        dotedges, dotprint)
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing.repr import srepr
from sympy.abc import x


def test_purestr():
    assert purestr(Symbol('x')) == "Symbol('x')"
    assert purestr(Basic(S(1), S(2))) == "Basic(Integer(1), Integer(2))"
    assert purestr(Float(2)) == "Float('2.0', precision=53)"

    assert purestr(Symbol('x'), with_args=True) == ("Symbol('x')", ())
    assert purestr(Basic(S(1), S(2)), with_args=True) == \
            ('Basic(Integer(1), Integer(2))', ('Integer(1)', 'Integer(2)'))
    assert purestr(Float(2), with_args=True) == \
        ("Float('2.0', precision=53)", ())


def test_styleof():
    styles = [(Basic, {'color': 'blue', 'shape': 'ellipse'}),
              (Expr,  {'color': 'black'})]
    assert styleof(Basic(S(1)), styles) == {'color': 'blue', 'shape': 'ellipse'}

    assert styleof(x + 1, styles) == {'color': 'black', 'shape': 'ellipse'}


def test_attrprint():
    assert attrprint({'color': 'blue', 'shape': 'ellipse'}) == \
           '"color"="blue", "shape"="ellipse"'

def test_dotnode():

    assert dotnode(x, repeat=False) == \
        '"Symbol(\'x\')" ["color"="black", "label"="x", "shape"="ellipse"];'
    assert dotnode(x+2, repeat=False) == \
        '"Add(Integer(2), Symbol(\'x\'))" ' \
        '["color"="black", "label"="Add", "shape"="ellipse"];', \
        dotnode(x+2,repeat=0)

    assert dotnode(x + x**2, repeat=False) == \
        '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))" ' \
        '["color"="black", "label"="Add", "shape"="ellipse"];'
    assert dotnode(x + x**2, repeat=True) == \
        '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))_()" ' \
        '["color"="black", "label"="Add", "shape"="ellipse"];'

def test_dotedges():
    assert sorted(dotedges(x+2, repeat=False)) == [
        '"Add(Integer(2), Symbol(\'x\'))" -> "Integer(2)";',
        '"Add(Integer(2), Symbol(\'x\'))" -> "Symbol(\'x\')";'
    ]
    assert sorted(dotedges(x + 2, repeat=True)) == [
        '"Add(Integer(2), Symbol(\'x\'))_()" -> "Integer(2)_(0,)";',
        '"Add(Integer(2), Symbol(\'x\'))_()" -> "Symbol(\'x\')_(1,)";'
    ]

def test_dotprint():
    text = dotprint(x+2, repeat=False)
    assert all(e in text for e in dotedges(x+2, repeat=False))
    assert all(
        n in text for n in [dotnode(expr, repeat=False)
        for expr in (x, Integer(2), x+2)])
    assert 'digraph' in text

    text = dotprint(x+x**2, repeat=False)
    assert all(e in text for e in dotedges(x+x**2, repeat=False))
    assert all(
        n in text for n in [dotnode(expr, repeat=False)
        for expr in (x, Integer(2), x**2)])
    assert 'digraph' in text

    text = dotprint(x+x**2, repeat=True)
    assert all(e in text for e in dotedges(x+x**2, repeat=True))
    assert all(
        n in text for n in [dotnode(expr, pos=())
        for expr in [x + x**2]])

    text = dotprint(x**x, repeat=True)
    assert all(e in text for e in dotedges(x**x, repeat=True))
    assert all(
        n in text for n in [dotnode(x, pos=(0,)), dotnode(x, pos=(1,))])
    assert 'digraph' in text

def test_dotprint_depth():
    text = dotprint(3*x+2, depth=1)
    assert dotnode(3*x+2) in text
    assert dotnode(x) not in text
    text = dotprint(3*x+2)
    assert "depth" not in text

def test_Matrix_and_non_basics():
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    n = Symbol('n')
    assert dotprint(MatrixSymbol('X', n, n)) == \
"""digraph{

# Graph style
"ordering"="out"
"rankdir"="TD"

#########
# Nodes #
#########

"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" ["color"="black", "label"="MatrixSymbol", "shape"="ellipse"];
"Str('X')_(0,)" ["color"="blue", "label"="X", "shape"="ellipse"];
"Symbol('n')_(1,)" ["color"="black", "label"="n", "shape"="ellipse"];
"Symbol('n')_(2,)" ["color"="black", "label"="n", "shape"="ellipse"];

#########
# Edges #
#########

"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Str('X')_(0,)";
"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(1,)";
"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(2,)";
}"""


def test_labelfunc():
    text = dotprint(x + 2, labelfunc=srepr)
    assert "Symbol('x')" in text
    assert "Integer(2)" in text


def test_commutative():
    x, y = symbols('x y', commutative=False)
    assert dotprint(x + y) == dotprint(y + x)
    assert dotprint(x*y) != dotprint(y*x)
