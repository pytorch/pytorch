from sympy.diffgeom import Manifold, Patch, CoordSystem, Point
from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.testing.pytest import warns_deprecated_sympy

m = Manifold('m', 2)
p = Patch('p', m)
a, b = symbols('a b')
cs = CoordSystem('cs', p, [a, b])
x, y = symbols('x y')
f = Function('f')
s1, s2 = cs.coord_functions()
v1, v2 = cs.base_vectors()
f1, f2 = cs.base_oneforms()

def test_point():
    point = Point(cs, [x, y])
    assert point != Point(cs, [2, y])
    #TODO assert point.subs(x, 2) == Point(cs, [2, y])
    #TODO assert point.free_symbols == set([x, y])

def test_subs():
    assert s1.subs(s1, s2) == s2
    assert v1.subs(v1, v2) == v2
    assert f1.subs(f1, f2) == f2
    assert (x*f(s1) + y).subs(s1, s2) == x*f(s2) + y
    assert (f(s1)*v1).subs(v1, v2) == f(s1)*v2
    assert (y*f(s1)*f1).subs(f1, f2) == y*f(s1)*f2

def test_deprecated():
    with warns_deprecated_sympy():
        cs_wname = CoordSystem('cs', p, ['a', 'b'])
        assert cs_wname == cs_wname.func(*cs_wname.args)
