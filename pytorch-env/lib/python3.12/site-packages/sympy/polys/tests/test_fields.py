"""Test sparse rational functions. """

from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex

from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt

def test_FracField___init__():
    F1 = FracField("x,y", ZZ, lex)
    F2 = FracField("x,y", ZZ, lex)
    F3 = FracField("x,y,z", ZZ, lex)

    assert F1.x == F1.gens[0]
    assert F1.y == F1.gens[1]
    assert F1.x == F2.x
    assert F1.y == F2.y
    assert F1.x != F3.x
    assert F1.y != F3.y

def test_FracField___hash__():
    F, x, y, z = field("x,y,z", QQ)
    assert hash(F)

def test_FracField___eq__():
    assert field("x,y,z", QQ)[0] == field("x,y,z", QQ)[0]
    assert field("x,y,z", QQ)[0] is field("x,y,z", QQ)[0]

    assert field("x,y,z", QQ)[0] != field("x,y,z", ZZ)[0]
    assert field("x,y,z", QQ)[0] is not field("x,y,z", ZZ)[0]

    assert field("x,y,z", ZZ)[0] != field("x,y,z", QQ)[0]
    assert field("x,y,z", ZZ)[0] is not field("x,y,z", QQ)[0]

    assert field("x,y,z", QQ)[0] != field("x,y", QQ)[0]
    assert field("x,y,z", QQ)[0] is not field("x,y", QQ)[0]

    assert field("x,y", QQ)[0] != field("x,y,z", QQ)[0]
    assert field("x,y", QQ)[0] is not field("x,y,z", QQ)[0]

def test_sfield():
    x = symbols("x")

    F = FracField((E, exp(exp(x)), exp(x)), ZZ, lex)
    e, exex, ex = F.gens
    assert sfield(exp(x)*exp(exp(x) + 1 + log(exp(x) + 3)/2)**2/(exp(x) + 3)) \
        == (F, e**2*exex**2*ex)

    F = FracField((x, exp(1/x), log(x), x**QQ(1, 3)), ZZ, lex)
    _, ex, lg, x3 = F.gens
    assert sfield(((x-3)*log(x)+4*x**2)*exp(1/x+log(x)/3)/x**2) == \
        (F, (4*F.x**2*ex + F.x*ex*lg - 3*ex*lg)/x3**5)

    F = FracField((x, log(x), sqrt(x + log(x))), ZZ, lex)
    _, lg, srt = F.gens
    assert sfield((x + 1) / (x * (x + log(x))**QQ(3, 2)) - 1/(x * log(x)**2)) \
        == (F, (F.x*lg**2 - F.x*srt + lg**2 - lg*srt)/
            (F.x**2*lg**2*srt + F.x*lg**3*srt))

def test_FracElement___hash__():
    F, x, y, z = field("x,y,z", QQ)
    assert hash(x*y/z)

def test_FracElement_copy():
    F, x, y, z = field("x,y,z", ZZ)

    f = x*y/3*z
    g = f.copy()

    assert f == g
    g.numer[(1, 1, 1)] = 7
    assert f != g

def test_FracElement_as_expr():
    F, x, y, z = field("x,y,z", ZZ)
    f = (3*x**2*y - x*y*z)/(7*z**3 + 1)

    X, Y, Z = F.symbols
    g = (3*X**2*Y - X*Y*Z)/(7*Z**3 + 1)

    assert f != g
    assert f.as_expr() == g

    X, Y, Z = symbols("x,y,z")
    g = (3*X**2*Y - X*Y*Z)/(7*Z**3 + 1)

    assert f != g
    assert f.as_expr(X, Y, Z) == g

    raises(ValueError, lambda: f.as_expr(X))

def test_FracElement_from_expr():
    x, y, z = symbols("x,y,z")
    F, X, Y, Z = field((x, y, z), ZZ)

    f = F.from_expr(1)
    assert f == 1 and isinstance(f, F.dtype)

    f = F.from_expr(Rational(3, 7))
    assert f == F(3)/7 and isinstance(f, F.dtype)

    f = F.from_expr(x)
    assert f == X and isinstance(f, F.dtype)

    f = F.from_expr(Rational(3,7)*x)
    assert f == X*Rational(3, 7) and isinstance(f, F.dtype)

    f = F.from_expr(1/x)
    assert f == 1/X and isinstance(f, F.dtype)

    f = F.from_expr(x*y*z)
    assert f == X*Y*Z and isinstance(f, F.dtype)

    f = F.from_expr(x*y/z)
    assert f == X*Y/Z and isinstance(f, F.dtype)

    f = F.from_expr(x*y*z + x*y + x)
    assert f == X*Y*Z + X*Y + X and isinstance(f, F.dtype)

    f = F.from_expr((x*y*z + x*y + x)/(x*y + 7))
    assert f == (X*Y*Z + X*Y + X)/(X*Y + 7) and isinstance(f, F.dtype)

    f = F.from_expr(x**3*y*z + x**2*y**7 + 1)
    assert f == X**3*Y*Z + X**2*Y**7 + 1 and isinstance(f, F.dtype)

    raises(ValueError, lambda: F.from_expr(2**x))
    raises(ValueError, lambda: F.from_expr(7*x + sqrt(2)))

    assert isinstance(ZZ[2**x].get_field().convert(2**(-x)),
        FracElement)
    assert isinstance(ZZ[x**2].get_field().convert(x**(-6)),
        FracElement)
    assert isinstance(ZZ[exp(Rational(1, 3))].get_field().convert(E),
        FracElement)


def test_FracField_nested():
    a, b, x = symbols('a b x')
    F1 = ZZ.frac_field(a, b)
    F2 = F1.frac_field(x)
    frac = F2(a + b)
    assert frac.numer == F1.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F1(a + b)]
    assert frac.denom == F1.poly_ring(x)(1)

    F3 = ZZ.poly_ring(a, b)
    F4 = F3.frac_field(x)
    frac = F4(a + b)
    assert frac.numer == F3.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F3(a + b)]
    assert frac.denom == F3.poly_ring(x)(1)

    frac = F2(F3(a + b))
    assert frac.numer == F1.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F1(a + b)]
    assert frac.denom == F1.poly_ring(x)(1)

    frac = F4(F1(a + b))
    assert frac.numer == F3.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F3(a + b)]
    assert frac.denom == F3.poly_ring(x)(1)


def test_FracElement__lt_le_gt_ge__():
    F, x, y = field("x,y", ZZ)

    assert F(1) < 1/x < 1/x**2 < 1/x**3
    assert F(1) <= 1/x <= 1/x**2 <= 1/x**3

    assert -7/x < 1/x < 3/x < y/x < 1/x**2
    assert -7/x <= 1/x <= 3/x <= y/x <= 1/x**2

    assert 1/x**3 > 1/x**2 > 1/x > F(1)
    assert 1/x**3 >= 1/x**2 >= 1/x >= F(1)

    assert 1/x**2 > y/x > 3/x > 1/x > -7/x
    assert 1/x**2 >= y/x >= 3/x >= 1/x >= -7/x

def test_FracElement___neg__():
    F, x,y = field("x,y", QQ)

    f = (7*x - 9)/y
    g = (-7*x + 9)/y

    assert -f == g
    assert -g == f

def test_FracElement___add__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y
    assert f + g == g + f == (x + y)/(x*y)

    assert x + F.ring.gens[0] == F.ring.gens[0] + x == 2*x

    F, x,y = field("x,y", ZZ)
    assert x + 3 == 3 + x
    assert x + QQ(3,7) == QQ(3,7) + x == (7*x + 3)/7

    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)

    f = (u*v + x)/(y + u*v)
    assert dict(f.numer) == {(1, 0, 0, 0): 1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0): u*v}

    Ruv, u,v = ring("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Ruv)

    f = (u*v + x)/(y + u*v)
    assert dict(f.numer) == {(1, 0, 0, 0): 1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0): u*v}

def test_FracElement___sub__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y
    assert f - g == (-x + y)/(x*y)

    assert x - F.ring.gens[0] == F.ring.gens[0] - x == 0

    F, x,y = field("x,y", ZZ)
    assert x - 3 == -(3 - x)
    assert x - QQ(3,7) == -(QQ(3,7) - x) == (7*x - 3)/7

    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)

    f = (u*v - x)/(y - u*v)
    assert dict(f.numer) == {(1, 0, 0, 0):-1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0):-u*v}

    Ruv, u,v = ring("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Ruv)

    f = (u*v - x)/(y - u*v)
    assert dict(f.numer) == {(1, 0, 0, 0):-1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0):-u*v}

def test_FracElement___mul__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y
    assert f*g == g*f == 1/(x*y)

    assert x*F.ring.gens[0] == F.ring.gens[0]*x == x**2

    F, x,y = field("x,y", ZZ)
    assert x*3 == 3*x
    assert x*QQ(3,7) == QQ(3,7)*x == x*Rational(3, 7)

    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)

    f = ((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)
    assert dict(f.numer) == {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    assert dict(f.denom) == {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u*v, (0, 0, 0, 0): -1}

    Ruv, u,v = ring("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Ruv)

    f = ((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)
    assert dict(f.numer) == {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    assert dict(f.denom) == {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u*v, (0, 0, 0, 0): -1}

def test_FracElement___truediv__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y
    assert f/g == y/x

    assert x/F.ring.gens[0] == F.ring.gens[0]/x == 1

    F, x,y = field("x,y", ZZ)
    assert x*3 == 3*x
    assert x/QQ(3,7) == (QQ(3,7)/x)**-1 == x*Rational(7, 3)

    raises(ZeroDivisionError, lambda: x/0)
    raises(ZeroDivisionError, lambda: 1/(x - x))
    raises(ZeroDivisionError, lambda: x/(x - x))

    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)

    f = (u*v)/(x*y)
    assert dict(f.numer) == {(0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(1, 1, 0, 0): 1}

    g = (x*y)/(u*v)
    assert dict(g.numer) == {(1, 1, 0, 0): 1}
    assert dict(g.denom) == {(0, 0, 0, 0): u*v}

    Ruv, u,v = ring("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Ruv)

    f = (u*v)/(x*y)
    assert dict(f.numer) == {(0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(1, 1, 0, 0): 1}

    g = (x*y)/(u*v)
    assert dict(g.numer) == {(1, 1, 0, 0): 1}
    assert dict(g.denom) == {(0, 0, 0, 0): u*v}

def test_FracElement___pow__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y

    assert f**3 == 1/x**3
    assert g**3 == 1/y**3

    assert (f*g)**3 == 1/(x**3*y**3)
    assert (f*g)**-3 == (x*y)**3

    raises(ZeroDivisionError, lambda: (x - x)**-3)

def test_FracElement_diff():
    F, x,y,z = field("x,y,z", ZZ)

    assert ((x**2 + y)/(z + 1)).diff(x) == 2*x/(z + 1)

@XFAIL
def test_FracElement___call__():
    F, x,y,z = field("x,y,z", ZZ)
    f = (x**2 + 3*y)/z

    r = f(1, 1, 1)
    assert r == 4 and not isinstance(r, FracElement)
    raises(ZeroDivisionError, lambda: f(1, 1, 0))

def test_FracElement_evaluate():
    F, x,y,z = field("x,y,z", ZZ)
    Fyz = field("y,z", ZZ)[0]
    f = (x**2 + 3*y)/z

    assert f.evaluate(x, 0) == 3*Fyz.y/Fyz.z
    raises(ZeroDivisionError, lambda: f.evaluate(z, 0))

def test_FracElement_subs():
    F, x,y,z = field("x,y,z", ZZ)
    f = (x**2 + 3*y)/z

    assert f.subs(x, 0) == 3*y/z
    raises(ZeroDivisionError, lambda: f.subs(z, 0))

def test_FracElement_compose():
    pass

def test_FracField_index():
    a = symbols("a")
    F, x, y, z = field('x y z', QQ)
    assert F.index(x) == 0
    assert F.index(y) == 1

    raises(ValueError, lambda: F.index(1))
    raises(ValueError, lambda: F.index(a))
    pass
