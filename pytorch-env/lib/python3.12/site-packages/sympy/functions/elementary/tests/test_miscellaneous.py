import itertools as it

from sympy.core.expr import unchanged
from sympy.core.function import Function
from sympy.core.numbers import I, oo, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import (sqrt, cbrt, root, Min,
                                                      Max, real_root, Rem)
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.delta_functions import Heaviside

from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, skip, ignore_warnings

def test_Min():
    from sympy.abc import x, y, z
    n = Symbol('n', negative=True)
    n_ = Symbol('n_', negative=True)
    nn = Symbol('nn', nonnegative=True)
    nn_ = Symbol('nn_', nonnegative=True)
    p = Symbol('p', positive=True)
    p_ = Symbol('p_', positive=True)
    np = Symbol('np', nonpositive=True)
    np_ = Symbol('np_', nonpositive=True)
    r = Symbol('r', real=True)

    assert Min(5, 4) == 4
    assert Min(-oo, -oo) is -oo
    assert Min(-oo, n) is -oo
    assert Min(n, -oo) is -oo
    assert Min(-oo, np) is -oo
    assert Min(np, -oo) is -oo
    assert Min(-oo, 0) is -oo
    assert Min(0, -oo) is -oo
    assert Min(-oo, nn) is -oo
    assert Min(nn, -oo) is -oo
    assert Min(-oo, p) is -oo
    assert Min(p, -oo) is -oo
    assert Min(-oo, oo) is -oo
    assert Min(oo, -oo) is -oo
    assert Min(n, n) == n
    assert unchanged(Min, n, np)
    assert Min(np, n) == Min(n, np)
    assert Min(n, 0) == n
    assert Min(0, n) == n
    assert Min(n, nn) == n
    assert Min(nn, n) == n
    assert Min(n, p) == n
    assert Min(p, n) == n
    assert Min(n, oo) == n
    assert Min(oo, n) == n
    assert Min(np, np) == np
    assert Min(np, 0) == np
    assert Min(0, np) == np
    assert Min(np, nn) == np
    assert Min(nn, np) == np
    assert Min(np, p) == np
    assert Min(p, np) == np
    assert Min(np, oo) == np
    assert Min(oo, np) == np
    assert Min(0, 0) == 0
    assert Min(0, nn) == 0
    assert Min(nn, 0) == 0
    assert Min(0, p) == 0
    assert Min(p, 0) == 0
    assert Min(0, oo) == 0
    assert Min(oo, 0) == 0
    assert Min(nn, nn) == nn
    assert unchanged(Min, nn, p)
    assert Min(p, nn) == Min(nn, p)
    assert Min(nn, oo) == nn
    assert Min(oo, nn) == nn
    assert Min(p, p) == p
    assert Min(p, oo) == p
    assert Min(oo, p) == p
    assert Min(oo, oo) is oo

    assert Min(n, n_).func is Min
    assert Min(nn, nn_).func is Min
    assert Min(np, np_).func is Min
    assert Min(p, p_).func is Min

    # lists
    assert Min() is S.Infinity
    assert Min(x) == x
    assert Min(x, y) == Min(y, x)
    assert Min(x, y, z) == Min(z, y, x)
    assert Min(x, Min(y, z)) == Min(z, y, x)
    assert Min(x, Max(y, -oo)) == Min(x, y)
    assert Min(p, oo, n, p, p, p_) == n
    assert Min(p_, n_, p) == n_
    assert Min(n, oo, -7, p, p, 2) == Min(n, -7)
    assert Min(2, x, p, n, oo, n_, p, 2, -2, -2) == Min(-2, x, n, n_)
    assert Min(0, x, 1, y) == Min(0, x, y)
    assert Min(1000, 100, -100, x, p, n) == Min(n, x, -100)
    assert unchanged(Min, sin(x), cos(x))
    assert Min(sin(x), cos(x)) == Min(cos(x), sin(x))
    assert Min(cos(x), sin(x)).subs(x, 1) == cos(1)
    assert Min(cos(x), sin(x)).subs(x, S.Half) == sin(S.Half)
    raises(ValueError, lambda: Min(cos(x), sin(x)).subs(x, I))
    raises(ValueError, lambda: Min(I))
    raises(ValueError, lambda: Min(I, x))
    raises(ValueError, lambda: Min(S.ComplexInfinity, x))

    assert Min(1, x).diff(x) == Heaviside(1 - x)
    assert Min(x, 1).diff(x) == Heaviside(1 - x)
    assert Min(0, -x, 1 - 2*x).diff(x) == -Heaviside(x + Min(0, -2*x + 1)) \
        - 2*Heaviside(2*x + Min(0, -x) - 1)

    # issue 7619
    f = Function('f')
    assert Min(1, 2*Min(f(1), 2))  # doesn't fail

    # issue 7233
    e = Min(0, x)
    assert e.n().args == (0, x)

    # issue 8643
    m = Min(n, p_, n_, r)
    assert m.is_positive is False
    assert m.is_nonnegative is False
    assert m.is_negative is True

    m = Min(p, p_)
    assert m.is_positive is True
    assert m.is_nonnegative is True
    assert m.is_negative is False

    m = Min(p, nn_, p_)
    assert m.is_positive is None
    assert m.is_nonnegative is True
    assert m.is_negative is False

    m = Min(nn, p, r)
    assert m.is_positive is None
    assert m.is_nonnegative is None
    assert m.is_negative is None


def test_Max():
    from sympy.abc import x, y, z
    n = Symbol('n', negative=True)
    n_ = Symbol('n_', negative=True)
    nn = Symbol('nn', nonnegative=True)
    p = Symbol('p', positive=True)
    p_ = Symbol('p_', positive=True)
    r = Symbol('r', real=True)

    assert Max(5, 4) == 5

    # lists

    assert Max() is S.NegativeInfinity
    assert Max(x) == x
    assert Max(x, y) == Max(y, x)
    assert Max(x, y, z) == Max(z, y, x)
    assert Max(x, Max(y, z)) == Max(z, y, x)
    assert Max(x, Min(y, oo)) == Max(x, y)
    assert Max(n, -oo, n_, p, 2) == Max(p, 2)
    assert Max(n, -oo, n_, p) == p
    assert Max(2, x, p, n, -oo, S.NegativeInfinity, n_, p, 2) == Max(2, x, p)
    assert Max(0, x, 1, y) == Max(1, x, y)
    assert Max(r, r + 1, r - 1) == 1 + r
    assert Max(1000, 100, -100, x, p, n) == Max(p, x, 1000)
    assert Max(cos(x), sin(x)) == Max(sin(x), cos(x))
    assert Max(cos(x), sin(x)).subs(x, 1) == sin(1)
    assert Max(cos(x), sin(x)).subs(x, S.Half) == cos(S.Half)
    raises(ValueError, lambda: Max(cos(x), sin(x)).subs(x, I))
    raises(ValueError, lambda: Max(I))
    raises(ValueError, lambda: Max(I, x))
    raises(ValueError, lambda: Max(S.ComplexInfinity, 1))
    assert Max(n, -oo, n_,  p, 2) == Max(p, 2)
    assert Max(n, -oo, n_,  p, 1000) == Max(p, 1000)

    assert Max(1, x).diff(x) == Heaviside(x - 1)
    assert Max(x, 1).diff(x) == Heaviside(x - 1)
    assert Max(x**2, 1 + x, 1).diff(x) == \
        2*x*Heaviside(x**2 - Max(1, x + 1)) \
        + Heaviside(x - Max(1, x**2) + 1)

    e = Max(0, x)
    assert e.n().args == (0, x)

    # issue 8643
    m = Max(p, p_, n, r)
    assert m.is_positive is True
    assert m.is_nonnegative is True
    assert m.is_negative is False

    m = Max(n, n_)
    assert m.is_positive is False
    assert m.is_nonnegative is False
    assert m.is_negative is True

    m = Max(n, n_, r)
    assert m.is_positive is None
    assert m.is_nonnegative is None
    assert m.is_negative is None

    m = Max(n, nn, r)
    assert m.is_positive is None
    assert m.is_nonnegative is True
    assert m.is_negative is False


def test_minmax_assumptions():
    r = Symbol('r', real=True)
    a = Symbol('a', real=True, algebraic=True)
    t = Symbol('t', real=True, transcendental=True)
    q = Symbol('q', rational=True)
    p = Symbol('p', irrational=True)
    n = Symbol('n', rational=True, integer=False)
    i = Symbol('i', integer=True)
    o = Symbol('o', odd=True)
    e = Symbol('e', even=True)
    k = Symbol('k', prime=True)
    reals = [r, a, t, q, p, n, i, o, e, k]

    for ext in (Max, Min):
        for x, y in it.product(reals, repeat=2):

            # Must be real
            assert ext(x, y).is_real

            # Algebraic?
            if x.is_algebraic and y.is_algebraic:
                assert ext(x, y).is_algebraic
            elif x.is_transcendental and y.is_transcendental:
                assert ext(x, y).is_transcendental
            else:
                assert ext(x, y).is_algebraic is None

            # Rational?
            if x.is_rational and y.is_rational:
                assert ext(x, y).is_rational
            elif x.is_irrational and y.is_irrational:
                assert ext(x, y).is_irrational
            else:
                assert ext(x, y).is_rational is None

            # Integer?
            if x.is_integer and y.is_integer:
                assert ext(x, y).is_integer
            elif x.is_noninteger and y.is_noninteger:
                assert ext(x, y).is_noninteger
            else:
                assert ext(x, y).is_integer is None

            # Odd?
            if x.is_odd and y.is_odd:
                assert ext(x, y).is_odd
            elif x.is_odd is False and y.is_odd is False:
                assert ext(x, y).is_odd is False
            else:
                assert ext(x, y).is_odd is None

            # Even?
            if x.is_even and y.is_even:
                assert ext(x, y).is_even
            elif x.is_even is False and y.is_even is False:
                assert ext(x, y).is_even is False
            else:
                assert ext(x, y).is_even is None

            # Prime?
            if x.is_prime and y.is_prime:
                assert ext(x, y).is_prime
            elif x.is_prime is False and y.is_prime is False:
                assert ext(x, y).is_prime is False
            else:
                assert ext(x, y).is_prime is None


def test_issue_8413():
    x = Symbol('x', real=True)
    # we can't evaluate in general because non-reals are not
    # comparable: Min(floor(3.2 + I), 3.2 + I) -> ValueError
    assert Min(floor(x), x) == floor(x)
    assert Min(ceiling(x), x) == x
    assert Max(floor(x), x) == x
    assert Max(ceiling(x), x) == ceiling(x)


def test_root():
    from sympy.abc import x
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True)

    assert root(2, 2) == sqrt(2)
    assert root(2, 1) == 2
    assert root(2, 3) == 2**Rational(1, 3)
    assert root(2, 3) == cbrt(2)
    assert root(2, -5) == 2**Rational(4, 5)/2

    assert root(-2, 1) == -2

    assert root(-2, 2) == sqrt(2)*I
    assert root(-2, 1) == -2

    assert root(x, 2) == sqrt(x)
    assert root(x, 1) == x
    assert root(x, 3) == x**Rational(1, 3)
    assert root(x, 3) == cbrt(x)
    assert root(x, -5) == x**Rational(-1, 5)

    assert root(x, n) == x**(1/n)
    assert root(x, -n) == x**(-1/n)

    assert root(x, n, k) == (-1)**(2*k/n)*x**(1/n)


def test_real_root():
    assert real_root(-8, 3) == -2
    assert real_root(-16, 4) == root(-16, 4)
    r = root(-7, 4)
    assert real_root(r) == r
    r1 = root(-1, 3)
    r2 = r1**2
    r3 = root(-1, 4)
    assert real_root(r1 + r2 + r3) == -1 + r2 + r3
    assert real_root(root(-2, 3)) == -root(2, 3)
    assert real_root(-8., 3) == -2.0
    x = Symbol('x')
    n = Symbol('n')
    g = real_root(x, n)
    assert g.subs({"x": -8, "n": 3}) == -2
    assert g.subs({"x": 8, "n": 3}) == 2
    # give principle root if there is no real root -- if this is not desired
    # then maybe a Root class is needed to raise an error instead
    assert g.subs({"x": I, "n": 3}) == cbrt(I)
    assert g.subs({"x": -8, "n": 2}) == sqrt(-8)
    assert g.subs({"x": I, "n": 2}) == sqrt(I)


def test_issue_11463():
    numpy = import_module('numpy')
    if not numpy:
        skip("numpy not installed.")
    x = Symbol('x')
    f = lambdify(x, real_root((log(x/(x-2))), 3), 'numpy')
    # numpy.select evaluates all options before considering conditions,
    # so it raises a warning about root of negative number which does
    # not affect the outcome. This warning is suppressed here
    with ignore_warnings(RuntimeWarning):
        assert f(numpy.array(-1)) < -1


def test_rewrite_MaxMin_as_Heaviside():
    from sympy.abc import x
    assert Max(0, x).rewrite(Heaviside) == x*Heaviside(x)
    assert Max(3, x).rewrite(Heaviside) == x*Heaviside(x - 3) + \
        3*Heaviside(-x + 3)
    assert Max(0, x+2, 2*x).rewrite(Heaviside) == \
        2*x*Heaviside(2*x)*Heaviside(x - 2) + \
        (x + 2)*Heaviside(-x + 2)*Heaviside(x + 2)

    assert Min(0, x).rewrite(Heaviside) == x*Heaviside(-x)
    assert Min(3, x).rewrite(Heaviside) == x*Heaviside(-x + 3) + \
        3*Heaviside(x - 3)
    assert Min(x, -x, -2).rewrite(Heaviside) == \
        x*Heaviside(-2*x)*Heaviside(-x - 2) - \
        x*Heaviside(2*x)*Heaviside(x - 2) \
        - 2*Heaviside(-x + 2)*Heaviside(x + 2)


def test_rewrite_MaxMin_as_Piecewise():
    from sympy.core.symbol import symbols
    from sympy.functions.elementary.piecewise import Piecewise
    x, y, z, a, b = symbols('x y z a b', real=True)
    vx, vy, va = symbols('vx vy va')
    assert Max(a, b).rewrite(Piecewise) == Piecewise((a, a >= b), (b, True))
    assert Max(x, y, z).rewrite(Piecewise) == Piecewise((x, (x >= y) & (x >= z)), (y, y >= z), (z, True))
    assert Max(x, y, a, b).rewrite(Piecewise) == Piecewise((a, (a >= b) & (a >= x) & (a >= y)),
        (b, (b >= x) & (b >= y)), (x, x >= y), (y, True))
    assert Min(a, b).rewrite(Piecewise) == Piecewise((a, a <= b), (b, True))
    assert Min(x, y, z).rewrite(Piecewise) == Piecewise((x, (x <= y) & (x <= z)), (y, y <= z), (z, True))
    assert Min(x,  y, a, b).rewrite(Piecewise) ==  Piecewise((a, (a <= b) & (a <= x) & (a <= y)),
        (b, (b <= x) & (b <= y)), (x, x <= y), (y, True))

    # Piecewise rewriting of Min/Max does also takes place for not explicitly real arguments
    assert Max(vx, vy).rewrite(Piecewise) == Piecewise((vx, vx >= vy), (vy, True))
    assert Min(va, vx, vy).rewrite(Piecewise) == Piecewise((va, (va <= vx) & (va <= vy)), (vx, vx <= vy), (vy, True))


def test_issue_11099():
    from sympy.abc import x, y
    # some fixed value tests
    fixed_test_data = {x: -2, y: 3}
    assert Min(x, y).evalf(subs=fixed_test_data) == \
        Min(x, y).subs(fixed_test_data).evalf()
    assert Max(x, y).evalf(subs=fixed_test_data) == \
        Max(x, y).subs(fixed_test_data).evalf()
    # randomly generate some test data
    from sympy.core.random import randint
    for i in range(20):
        random_test_data = {x: randint(-100, 100), y: randint(-100, 100)}
        assert Min(x, y).evalf(subs=random_test_data) == \
            Min(x, y).subs(random_test_data).evalf()
        assert Max(x, y).evalf(subs=random_test_data) == \
            Max(x, y).subs(random_test_data).evalf()


def test_issue_12638():
    from sympy.abc import a, b, c
    assert Min(a, b, c, Max(a, b)) == Min(a, b, c)
    assert Min(a, b, Max(a, b, c)) == Min(a, b)
    assert Min(a, b, Max(a, c)) == Min(a, b)

def test_issue_21399():
    from sympy.abc import a, b, c
    assert Max(Min(a, b), Min(a, b, c)) == Min(a, b)


def test_instantiation_evaluation():
    from sympy.abc import v, w, x, y, z
    assert Min(1, Max(2, x)) == 1
    assert Max(3, Min(2, x)) == 3
    assert Min(Max(x, y), Max(x, z)) == Max(x, Min(y, z))
    assert set(Min(Max(w, x), Max(y, z)).args) == {
        Max(w, x), Max(y, z)}
    assert Min(Max(x, y), Max(x, z), w) == Min(
        w, Max(x, Min(y, z)))
    A, B = Min, Max
    for i in range(2):
        assert A(x, B(x, y)) == x
        assert A(x, B(y, A(x, w, z))) == A(x, B(y, A(w, z)))
        A, B = B, A
    assert Min(w, Max(x, y), Max(v, x, z)) == Min(
        w, Max(x, Min(y, Max(v, z))))

def test_rewrite_as_Abs():
    from itertools import permutations
    from sympy.functions.elementary.complexes import Abs
    from sympy.abc import x, y, z, w
    def test(e):
        free = e.free_symbols
        a = e.rewrite(Abs)
        assert not a.has(Min, Max)
        for i in permutations(range(len(free))):
            reps = dict(zip(free, i))
            assert a.xreplace(reps) == e.xreplace(reps)
    test(Min(x, y))
    test(Max(x, y))
    test(Min(x, y, z))
    test(Min(Max(w, x), Max(y, z)))

def test_issue_14000():
    assert isinstance(sqrt(4, evaluate=False), Pow) == True
    assert isinstance(cbrt(3.5, evaluate=False), Pow) == True
    assert isinstance(root(16, 4, evaluate=False), Pow) == True

    assert sqrt(4, evaluate=False) == Pow(4, S.Half, evaluate=False)
    assert cbrt(3.5, evaluate=False) == Pow(3.5, Rational(1, 3), evaluate=False)
    assert root(4, 2, evaluate=False) == Pow(4, S.Half, evaluate=False)

    assert root(16, 4, 2, evaluate=False).has(Pow) == True
    assert real_root(-8, 3, evaluate=False).has(Pow) == True

def test_issue_6899():
    from sympy.core.function import Lambda
    x = Symbol('x')
    eqn = Lambda(x, x)
    assert eqn.func(*eqn.args) == eqn

def test_Rem():
    from sympy.abc import x, y
    assert Rem(5, 3) == 2
    assert Rem(-5, 3) == -2
    assert Rem(5, -3) == 2
    assert Rem(-5, -3) == -2
    assert Rem(x**3, y) == Rem(x**3, y)
    assert Rem(Rem(-5, 3) + 3, 3) == 1


def test_minmax_no_evaluate():
    from sympy import evaluate
    p = Symbol('p', positive=True)

    assert Max(1, 3) == 3
    assert Max(1, 3).args == ()
    assert Max(0, p) == p
    assert Max(0, p).args == ()
    assert Min(0, p) == 0
    assert Min(0, p).args == ()

    assert Max(1, 3, evaluate=False) != 3
    assert Max(1, 3, evaluate=False).args == (1, 3)
    assert Max(0, p, evaluate=False) != p
    assert Max(0, p, evaluate=False).args == (0, p)
    assert Min(0, p, evaluate=False) != 0
    assert Min(0, p, evaluate=False).args == (0, p)

    with evaluate(False):
        assert Max(1, 3) != 3
        assert Max(1, 3).args == (1, 3)
        assert Max(0, p) != p
        assert Max(0, p).args == (0, p)
        assert Min(0, p) != 0
        assert Min(0, p).args == (0, p)
