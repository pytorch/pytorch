from itertools import product

from sympy.core.function import (Function, diff)
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.calculus.finite_diff import (
    apply_finite_diff, differentiate_finite, finite_diff_weights,
    _as_finite_diff
)
from sympy.testing.pytest import raises, warns_deprecated_sympy


def test_apply_finite_diff():
    x, h = symbols('x h')
    f = Function('f')
    assert (apply_finite_diff(1, [x-h, x+h], [f(x-h), f(x+h)], x) -
            (f(x+h)-f(x-h))/(2*h)).simplify() == 0

    assert (apply_finite_diff(1, [5, 6, 7], [f(5), f(6), f(7)], 5) -
            (Rational(-3, 2)*f(5) + 2*f(6) - S.Half*f(7))).simplify() == 0
    raises(ValueError, lambda: apply_finite_diff(1, [x, h], [f(x)]))


def test_finite_diff_weights():

    d = finite_diff_weights(1, [5, 6, 7], 5)
    assert d[1][2] == [Rational(-3, 2), 2, Rational(-1, 2)]

    # Table 1, p. 702 in doi:10.1090/S0025-5718-1988-0935077-0
    # --------------------------------------------------------
    xl = [0, 1, -1, 2, -2, 3, -3, 4, -4]

    # d holds all coefficients
    d = finite_diff_weights(4, xl, S.Zero)

    # Zeroeth derivative
    for i in range(5):
        assert d[0][i] == [S.One] + [S.Zero]*8

    # First derivative
    assert d[1][0] == [S.Zero]*9
    assert d[1][2] == [S.Zero, S.Half, Rational(-1, 2)] + [S.Zero]*6
    assert d[1][4] == [S.Zero, Rational(2, 3), Rational(-2, 3), Rational(-1, 12), Rational(1, 12)] + [S.Zero]*4
    assert d[1][6] == [S.Zero, Rational(3, 4), Rational(-3, 4), Rational(-3, 20), Rational(3, 20),
                       Rational(1, 60), Rational(-1, 60)] + [S.Zero]*2
    assert d[1][8] == [S.Zero, Rational(4, 5), Rational(-4, 5), Rational(-1, 5), Rational(1, 5),
                       Rational(4, 105), Rational(-4, 105), Rational(-1, 280), Rational(1, 280)]

    # Second derivative
    for i in range(2):
        assert d[2][i] == [S.Zero]*9
    assert d[2][2] == [-S(2), S.One, S.One] + [S.Zero]*6
    assert d[2][4] == [Rational(-5, 2), Rational(4, 3), Rational(4, 3), Rational(-1, 12), Rational(-1, 12)] + [S.Zero]*4
    assert d[2][6] == [Rational(-49, 18), Rational(3, 2), Rational(3, 2), Rational(-3, 20), Rational(-3, 20),
                       Rational(1, 90), Rational(1, 90)] + [S.Zero]*2
    assert d[2][8] == [Rational(-205, 72), Rational(8, 5), Rational(8, 5), Rational(-1, 5), Rational(-1, 5),
                       Rational(8, 315), Rational(8, 315), Rational(-1, 560), Rational(-1, 560)]

    # Third derivative
    for i in range(3):
        assert d[3][i] == [S.Zero]*9
    assert d[3][4] == [S.Zero, -S.One, S.One, S.Half, Rational(-1, 2)] + [S.Zero]*4
    assert d[3][6] == [S.Zero, Rational(-13, 8), Rational(13, 8), S.One, -S.One,
                       Rational(-1, 8), Rational(1, 8)] + [S.Zero]*2
    assert d[3][8] == [S.Zero, Rational(-61, 30), Rational(61, 30), Rational(169, 120), Rational(-169, 120),
                       Rational(-3, 10), Rational(3, 10), Rational(7, 240), Rational(-7, 240)]

    # Fourth derivative
    for i in range(4):
        assert d[4][i] == [S.Zero]*9
    assert d[4][4] == [S(6), -S(4), -S(4), S.One, S.One] + [S.Zero]*4
    assert d[4][6] == [Rational(28, 3), Rational(-13, 2), Rational(-13, 2), S(2), S(2),
                       Rational(-1, 6), Rational(-1, 6)] + [S.Zero]*2
    assert d[4][8] == [Rational(91, 8), Rational(-122, 15), Rational(-122, 15), Rational(169, 60), Rational(169, 60),
                       Rational(-2, 5), Rational(-2, 5), Rational(7, 240), Rational(7, 240)]

    # Table 2, p. 703 in doi:10.1090/S0025-5718-1988-0935077-0
    # --------------------------------------------------------
    xl = [[j/S(2) for j in list(range(-i*2+1, 0, 2))+list(range(1, i*2+1, 2))]
          for i in range(1, 5)]

    # d holds all coefficients
    d = [finite_diff_weights({0: 1, 1: 2, 2: 4, 3: 4}[i], xl[i], 0) for
         i in range(4)]

    # Zeroth derivative
    assert d[0][0][1] == [S.Half, S.Half]
    assert d[1][0][3] == [Rational(-1, 16), Rational(9, 16), Rational(9, 16), Rational(-1, 16)]
    assert d[2][0][5] == [Rational(3, 256), Rational(-25, 256), Rational(75, 128), Rational(75, 128),
                          Rational(-25, 256), Rational(3, 256)]
    assert d[3][0][7] == [Rational(-5, 2048), Rational(49, 2048), Rational(-245, 2048), Rational(1225, 2048),
                          Rational(1225, 2048), Rational(-245, 2048), Rational(49, 2048), Rational(-5, 2048)]

    # First derivative
    assert d[0][1][1] == [-S.One, S.One]
    assert d[1][1][3] == [Rational(1, 24), Rational(-9, 8), Rational(9, 8), Rational(-1, 24)]
    assert d[2][1][5] == [Rational(-3, 640), Rational(25, 384), Rational(-75, 64),
                          Rational(75, 64), Rational(-25, 384), Rational(3, 640)]
    assert d[3][1][7] == [Rational(5, 7168), Rational(-49, 5120),
                          Rational(245, 3072), Rational(-1225, 1024),
                          Rational(1225, 1024), Rational(-245, 3072),
                          Rational(49, 5120), Rational(-5, 7168)]

    # Reasonably the rest of the table is also correct... (testing of that
    # deemed excessive at the moment)
    raises(ValueError, lambda: finite_diff_weights(-1, [1, 2]))
    raises(ValueError, lambda: finite_diff_weights(1.2, [1, 2]))
    x = symbols('x')
    raises(ValueError, lambda: finite_diff_weights(x, [1, 2]))


def test_as_finite_diff():
    x = symbols('x')
    f = Function('f')
    dx = Function('dx')

    _as_finite_diff(f(x).diff(x), [x-2, x-1, x, x+1, x+2])

    # Use of undefined functions in ``points``
    df_true = -f(x+dx(x)/2-dx(x+dx(x)/2)/2) / dx(x+dx(x)/2) \
              + f(x+dx(x)/2+dx(x+dx(x)/2)/2) / dx(x+dx(x)/2)
    df_test = diff(f(x), x).as_finite_difference(points=dx(x), x0=x+dx(x)/2)
    assert (df_test - df_true).simplify() == 0


def test_differentiate_finite():
    x, y, h = symbols('x y h')
    f = Function('f')
    with warns_deprecated_sympy():
        res0 = differentiate_finite(f(x, y) + exp(42), x, y, evaluate=True)
    xm, xp, ym, yp = [v + sign*S.Half for v, sign in product([x, y], [-1, 1])]
    ref0 = f(xm, ym) + f(xp, yp) - f(xm, yp) - f(xp, ym)
    assert (res0 - ref0).simplify() == 0

    g = Function('g')
    with warns_deprecated_sympy():
        res1 = differentiate_finite(f(x)*g(x) + 42, x, evaluate=True)
    ref1 = (-f(x - S.Half) + f(x + S.Half))*g(x) + \
           (-g(x - S.Half) + g(x + S.Half))*f(x)
    assert (res1 - ref1).simplify() == 0

    res2 = differentiate_finite(f(x) + x**3 + 42, x, points=[x-1, x+1])
    ref2 = (f(x + 1) + (x + 1)**3 - f(x - 1) - (x - 1)**3)/2
    assert (res2 - ref2).simplify() == 0
    raises(TypeError, lambda: differentiate_finite(f(x)*g(x), x,
                                                   pints=[x-1, x+1]))

    res3 = differentiate_finite(f(x)*g(x).diff(x), x)
    ref3 = (-g(x) + g(x + 1))*f(x + S.Half) - (g(x) - g(x - 1))*f(x - S.Half)
    assert res3 == ref3

    res4 = differentiate_finite(f(x)*g(x).diff(x).diff(x), x)
    ref4 = -((g(x - Rational(3, 2)) - 2*g(x - S.Half) + g(x + S.Half))*f(x - S.Half)) \
           + (g(x - S.Half) - 2*g(x + S.Half) + g(x + Rational(3, 2)))*f(x + S.Half)
    assert res4 == ref4

    res5_expr = f(x).diff(x)*g(x).diff(x)
    res5 = differentiate_finite(res5_expr, points=[x-h, x, x+h])
    ref5 = (-2*f(x)/h + f(-h + x)/(2*h) + 3*f(h + x)/(2*h))*(-2*g(x)/h + g(-h + x)/(2*h) \
           + 3*g(h + x)/(2*h))/(2*h) - (2*f(x)/h - 3*f(-h + x)/(2*h) - \
           f(h + x)/(2*h))*(2*g(x)/h - 3*g(-h + x)/(2*h) - g(h + x)/(2*h))/(2*h)
    assert res5 == ref5

    res6 = res5.limit(h, 0).doit()
    ref6 = diff(res5_expr, x)
    assert res6 == ref6
