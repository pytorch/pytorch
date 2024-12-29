from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import symbols
from sympy.core.singleton import S
from sympy.core.function import expand, Function
from sympy.core.numbers import I
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor
from sympy.core.traversal import preorder_traversal, use, postorder_traversal, iterargs, iterfreeargs
from sympy.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy.testing.pytest import warns_deprecated_sympy
from sympy.utilities.iterables import capture

b1 = Basic()
b2 = Basic(b1)
b3 = Basic(b2)
b21 = Basic(b2, b1)


def test_preorder_traversal():
    expr = Basic(b21, b3)
    assert list(
        preorder_traversal(expr)) == [expr, b21, b2, b1, b1, b3, b2, b1]
    assert list(preorder_traversal(('abc', ('d', 'ef')))) == [
        ('abc', ('d', 'ef')), 'abc', ('d', 'ef'), 'd', 'ef']

    result = []
    pt = preorder_traversal(expr)
    for i in pt:
        result.append(i)
        if i == b2:
            pt.skip()
    assert result == [expr, b21, b2, b1, b3, b2]

    w, x, y, z = symbols('w:z')
    expr = z + w*(x + y)
    assert list(preorder_traversal([expr], keys=default_sort_key)) == \
        [[w*(x + y) + z], w*(x + y) + z, z, w*(x + y), w, x + y, x, y]
    assert list(preorder_traversal((x + y)*z, keys=True)) == \
        [z*(x + y), z, x + y, x, y]


def test_use():
    x, y = symbols('x y')

    assert use(0, expand) == 0

    f = (x + y)**2*x + 1

    assert use(f, expand, level=0) == x**3 + 2*x**2*y + x*y**2 + + 1
    assert use(f, expand, level=1) == x**3 + 2*x**2*y + x*y**2 + + 1
    assert use(f, expand, level=2) == 1 + x*(2*x*y + x**2 + y**2)
    assert use(f, expand, level=3) == (x + y)**2*x + 1

    f = (x**2 + 1)**2 - 1
    kwargs = {'gaussian': True}

    assert use(f, factor, level=0, kwargs=kwargs) == x**2*(x**2 + 2)
    assert use(f, factor, level=1, kwargs=kwargs) == (x + I)**2*(x - I)**2 - 1
    assert use(f, factor, level=2, kwargs=kwargs) == (x + I)**2*(x - I)**2 - 1
    assert use(f, factor, level=3, kwargs=kwargs) == (x**2 + 1)**2 - 1


def test_postorder_traversal():
    x, y, z, w = symbols('x y z w')
    expr = z + w*(x + y)
    expected = [z, w, x, y, x + y, w*(x + y), w*(x + y) + z]
    assert list(postorder_traversal(expr, keys=default_sort_key)) == expected
    assert list(postorder_traversal(expr, keys=True)) == expected

    expr = Piecewise((x, x < 1), (x**2, True))
    expected = [
        x, 1, x, x < 1, ExprCondPair(x, x < 1),
        2, x, x**2, S.true,
        ExprCondPair(x**2, True), Piecewise((x, x < 1), (x**2, True))
    ]
    assert list(postorder_traversal(expr, keys=default_sort_key)) == expected
    assert list(postorder_traversal(
        [expr], keys=default_sort_key)) == expected + [[expr]]

    assert list(postorder_traversal(Integral(x**2, (x, 0, 1)),
        keys=default_sort_key)) == [
            2, x, x**2, 0, 1, x, Tuple(x, 0, 1),
            Integral(x**2, Tuple(x, 0, 1))
        ]
    assert list(postorder_traversal(('abc', ('d', 'ef')))) == [
        'abc', 'd', 'ef', ('d', 'ef'), ('abc', ('d', 'ef'))]


def test_iterargs():
    f = Function('f')
    x = symbols('x')
    assert list(iterfreeargs(Integral(f(x), (f(x), 1)))) == [
        Integral(f(x), (f(x), 1)), 1]
    assert list(iterargs(Integral(f(x), (f(x), 1)))) == [
        Integral(f(x), (f(x), 1)), f(x), (f(x), 1), x, f(x), 1, x]

def test_deprecated_imports():
    x = symbols('x')

    with warns_deprecated_sympy():
        from sympy.core.basic import preorder_traversal
        preorder_traversal(x)
    with warns_deprecated_sympy():
        from sympy.simplify.simplify import bottom_up
        bottom_up(x, lambda x: x)
    with warns_deprecated_sympy():
        from sympy.simplify.simplify import walk
        walk(x, lambda x: x)
    with warns_deprecated_sympy():
        from sympy.simplify.traversaltools import use
        use(x, lambda x: x)
    with warns_deprecated_sympy():
        from sympy.utilities.iterables import postorder_traversal
        postorder_traversal(x)
    with warns_deprecated_sympy():
        from sympy.utilities.iterables import interactive_traversal
        capture(lambda: interactive_traversal(x))
