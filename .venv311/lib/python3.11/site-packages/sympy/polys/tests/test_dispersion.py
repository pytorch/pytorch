from sympy.core import Symbol, S, oo
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import poly
from sympy.polys.dispersion import dispersion, dispersionset


def test_dispersion():
    x = Symbol("x")
    a = Symbol("a")

    fp = poly(S.Zero, x)
    assert sorted(dispersionset(fp)) == [0]

    fp = poly(S(2), x)
    assert sorted(dispersionset(fp)) == [0]

    fp = poly(x + 1, x)
    assert sorted(dispersionset(fp)) == [0]
    assert dispersion(fp) == 0

    fp = poly((x + 1)*(x + 2), x)
    assert sorted(dispersionset(fp)) == [0, 1]
    assert dispersion(fp) == 1

    fp = poly(x*(x + 3), x)
    assert sorted(dispersionset(fp)) == [0, 3]
    assert dispersion(fp) == 3

    fp = poly((x - 3)*(x + 3), x)
    assert sorted(dispersionset(fp)) == [0, 6]
    assert dispersion(fp) == 6

    fp = poly(x**4 - 3*x**2 + 1, x)
    gp = fp.shift(-3)
    assert sorted(dispersionset(fp, gp)) == [2, 3, 4]
    assert dispersion(fp, gp) == 4
    assert sorted(dispersionset(gp, fp)) == []
    assert dispersion(gp, fp) is -oo

    fp = poly(x*(3*x**2+a)*(x-2536)*(x**3+a), x)
    gp = fp.as_expr().subs(x, x-345).as_poly(x)
    assert sorted(dispersionset(fp, gp)) == [345, 2881]
    assert sorted(dispersionset(gp, fp)) == [2191]

    gp = poly((x-2)**2*(x-3)**3*(x-5)**3, x)
    assert sorted(dispersionset(gp)) == [0, 1, 2, 3]
    assert sorted(dispersionset(gp, (gp+4)**2)) == [1, 2]

    fp = poly(x*(x+2)*(x-1), x)
    assert sorted(dispersionset(fp)) == [0, 1, 2, 3]

    fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
    gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
    assert sorted(dispersionset(fp, gp)) == [2]
    assert sorted(dispersionset(gp, fp)) == [1, 4]

    # There are some difficulties if we compute over Z[a]
    # and alpha happens to lie in Z[a] instead of simply Z.
    # Hence we can not decide if alpha is indeed integral
    # in general.

    fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
    assert sorted(dispersionset(fp)) == [0, 1]

    # For any specific value of a, the dispersion is 3*a
    # but the algorithm can not find this in general.
    # This is the point where the resultant based Ansatz
    # is superior to the current one.
    fp = poly(a**2*x**3 + (a**3 + a**2 + a + 1)*x, x)
    gp = fp.as_expr().subs(x, x - 3*a).as_poly(x)
    assert sorted(dispersionset(fp, gp)) == []

    fpa = fp.as_expr().subs(a, 2).as_poly(x)
    gpa = gp.as_expr().subs(a, 2).as_poly(x)
    assert sorted(dispersionset(fpa, gpa)) == [6]

    # Work with Expr instead of Poly
    f = (x + 1)*(x + 2)
    assert sorted(dispersionset(f)) == [0, 1]
    assert dispersion(f) == 1

    f = x**4 - 3*x**2 + 1
    g = x**4 - 12*x**3 + 51*x**2 - 90*x + 55
    assert sorted(dispersionset(f, g)) == [2, 3, 4]
    assert dispersion(f, g) == 4

    # Work with Expr and specify a generator
    f = (x + 1)*(x + 2)
    assert sorted(dispersionset(f, None, x)) == [0, 1]
    assert dispersion(f, None, x) == 1

    f = x**4 - 3*x**2 + 1
    g = x**4 - 12*x**3 + 51*x**2 - 90*x + 55
    assert sorted(dispersionset(f, g, x)) == [2, 3, 4]
    assert dispersion(f, g, x) == 4
