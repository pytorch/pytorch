"""Tests for algorithms for computing symbolic roots of polynomials. """

from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import (conjugate, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.polys.domains.integerring import ZZ
from sympy.sets.sets import Interval
from sympy.simplify.powsimp import powsimp

from sympy.polys import Poly, cyclotomic_poly, intervals, nroots, rootof

from sympy.polys.polyroots import (root_factors, roots_linear,
    roots_quadratic, roots_cubic, roots_quartic, roots_quintic,
    roots_cyclotomic, roots_binomial, preprocess_roots, roots)

from sympy.polys.orthopolys import legendre_poly
from sympy.polys.polyerrors import PolynomialError, \
    UnsolvableFactorError
from sympy.polys.polyutils import _nsort

from sympy.testing.pytest import raises, slow
from sympy.core.random import verify_numerically
import mpmath
from itertools import product



a, b, c, d, e, q, t, x, y, z = symbols('a,b,c,d,e,q,t,x,y,z')


def _check(roots):
    # this is the desired invariant for roots returned
    # by all_roots. It is trivially true for linear
    # polynomials.
    nreal = sum(1 if i.is_real else 0 for i in roots)
    assert sorted(roots[:nreal]) == list(roots[:nreal])
    for ix in range(nreal, len(roots), 2):
        if not (
                roots[ix + 1] == roots[ix] or
                roots[ix + 1] == conjugate(roots[ix])):
            return False
    return True


def test_roots_linear():
    assert roots_linear(Poly(2*x + 1, x)) == [Rational(-1, 2)]


def test_roots_quadratic():
    assert roots_quadratic(Poly(2*x**2, x)) == [0, 0]
    assert roots_quadratic(Poly(2*x**2 + 3*x, x)) == [Rational(-3, 2), 0]
    assert roots_quadratic(Poly(2*x**2 + 3, x)) == [-I*sqrt(6)/2, I*sqrt(6)/2]
    assert roots_quadratic(Poly(2*x**2 + 4*x + 3, x)) == [-1 - I*sqrt(2)/2, -1 + I*sqrt(2)/2]
    _check(Poly(2*x**2 + 4*x + 3, x).all_roots())

    f = x**2 + (2*a*e + 2*c*e)/(a - c)*x + (d - b + a*e**2 - c*e**2)/(a - c)
    assert roots_quadratic(Poly(f, x)) == \
        [-e*(a + c)/(a - c) - sqrt(a*b + c*d - a*d - b*c + 4*a*c*e**2)/(a - c),
         -e*(a + c)/(a - c) + sqrt(a*b + c*d - a*d - b*c + 4*a*c*e**2)/(a - c)]

    # check for simplification
    f = Poly(y*x**2 - 2*x - 2*y, x)
    assert roots_quadratic(f) == \
        [-sqrt(2*y**2 + 1)/y + 1/y, sqrt(2*y**2 + 1)/y + 1/y]
    f = Poly(x**2 + (-y**2 - 2)*x + y**2 + 1, x)
    assert roots_quadratic(f) == \
        [1,y**2 + 1]

    f = Poly(sqrt(2)*x**2 - 1, x)
    r = roots_quadratic(f)
    assert r == _nsort(r)

    # issue 8255
    f = Poly(-24*x**2 - 180*x + 264)
    assert [w.n(2) for w in f.all_roots(radicals=True)] == \
           [w.n(2) for w in f.all_roots(radicals=False)]
    for _a, _b, _c in product((-2, 2), (-2, 2), (0, -1)):
        f = Poly(_a*x**2 + _b*x + _c)
        roots = roots_quadratic(f)
        assert roots == _nsort(roots)


def test_issue_7724():
    eq = Poly(x**4*I + x**2 + I, x)
    assert roots(eq) == {
        sqrt(I/2 + sqrt(5)*I/2): 1,
        sqrt(-sqrt(5)*I/2 + I/2): 1,
        -sqrt(I/2 + sqrt(5)*I/2): 1,
        -sqrt(-sqrt(5)*I/2 + I/2): 1}


def test_issue_8438():
    p = Poly([1, y, -2, -3], x).as_expr()
    roots = roots_cubic(Poly(p, x), x)
    z = Rational(-3, 2) - I*7/2  # this will fail in code given in commit msg
    post = [r.subs(y, z) for r in roots]
    assert set(post) == \
    set(roots_cubic(Poly(p.subs(y, z), x)))
    # /!\ if p is not made an expression, this is *very* slow
    assert all(p.subs({y: z, x: i}).n(2, chop=True) == 0 for i in post)


def test_issue_8285():
    roots = (Poly(4*x**8 - 1, x)*Poly(x**2 + 1)).all_roots()
    assert _check(roots)
    f = Poly(x**4 + 5*x**2 + 6, x)
    ro = [rootof(f, i) for i in range(4)]
    roots = Poly(x**4 + 5*x**2 + 6, x).all_roots()
    assert roots == ro
    assert _check(roots)
    # more than 2 complex roots from which to identify the
    # imaginary ones
    roots = Poly(2*x**8 - 1).all_roots()
    assert _check(roots)
    assert len(Poly(2*x**10 - 1).all_roots()) == 10  # doesn't fail


def test_issue_8289():
    roots = (Poly(x**2 + 2)*Poly(x**4 + 2)).all_roots()
    assert _check(roots)
    roots = Poly(x**6 + 3*x**3 + 2, x).all_roots()
    assert _check(roots)
    roots = Poly(x**6 - x + 1).all_roots()
    assert _check(roots)
    # all imaginary roots with multiplicity of 2
    roots = Poly(x**4 + 4*x**2 + 4, x).all_roots()
    assert _check(roots)


def test_issue_14291():
    assert Poly(((x - 1)**2 + 1)*((x - 1)**2 + 2)*(x - 1)
        ).all_roots() == [1, 1 - I, 1 + I, 1 - sqrt(2)*I, 1 + sqrt(2)*I]
    p = x**4 + 10*x**2 + 1
    ans = [rootof(p, i) for i in range(4)]
    assert Poly(p).all_roots() == ans
    _check(ans)


def test_issue_13340():
    eq = Poly(y**3 + exp(x)*y + x, y, domain='EX')
    roots_d = roots(eq)
    assert len(roots_d) == 3


def test_issue_14522():
    eq = Poly(x**4 + x**3*(16 + 32*I) + x**2*(-285 + 386*I) + x*(-2824 - 448*I) - 2058 - 6053*I, x)
    roots_eq = roots(eq)
    assert all(eq(r) == 0 for r in roots_eq)


def test_issue_15076():
    sol = roots_quartic(Poly(t**4 - 6*t**2 + t/x - 3, t))
    assert sol[0].has(x)


def test_issue_16589():
    eq = Poly(x**4 - 8*sqrt(2)*x**3 + 4*x**3 - 64*sqrt(2)*x**2 + 1024*x, x)
    roots_eq = roots(eq)
    assert 0 in roots_eq


def test_roots_cubic():
    assert roots_cubic(Poly(2*x**3, x)) == [0, 0, 0]
    assert roots_cubic(Poly(x**3 - 3*x**2 + 3*x - 1, x)) == [1, 1, 1]

    # valid for arbitrary y (issue 21263)
    r = root(y, 3)
    assert roots_cubic(Poly(x**3 - y, x)) == [r,
        r*(-S.Half + sqrt(3)*I/2),
        r*(-S.Half - sqrt(3)*I/2)]
    # simpler form when y is negative
    assert roots_cubic(Poly(x**3 - -1, x)) == \
        [-1, S.Half - I*sqrt(3)/2, S.Half + I*sqrt(3)/2]
    assert roots_cubic(Poly(2*x**3 - 3*x**2 - 3*x - 1, x))[0] == \
         S.Half + 3**Rational(1, 3)/2 + 3**Rational(2, 3)/2
    eq = -x**3 + 2*x**2 + 3*x - 2
    assert roots(eq, trig=True, multiple=True) == \
           roots_cubic(Poly(eq, x), trig=True) == [
        Rational(2, 3) + 2*sqrt(13)*cos(acos(8*sqrt(13)/169)/3)/3,
        -2*sqrt(13)*sin(-acos(8*sqrt(13)/169)/3 + pi/6)/3 + Rational(2, 3),
        -2*sqrt(13)*cos(-acos(8*sqrt(13)/169)/3 + pi/3)/3 + Rational(2, 3),
        ]


def test_roots_quartic():
    assert roots_quartic(Poly(x**4, x)) == [0, 0, 0, 0]
    assert roots_quartic(Poly(x**4 + x**3, x)) in [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ]
    assert roots_quartic(Poly(x**4 - x**3, x)) in [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    lhs = roots_quartic(Poly(x**4 + x, x))
    rhs = [S.Half + I*sqrt(3)/2, S.Half - I*sqrt(3)/2, S.Zero, -S.One]

    assert sorted(lhs, key=hash) == sorted(rhs, key=hash)

    # test of all branches of roots quartic
    for i, (a, b, c, d) in enumerate([(1, 2, 3, 0),
                                      (3, -7, -9, 9),
                                      (1, 2, 3, 4),
                                      (1, 2, 3, 4),
                                      (-7, -3, 3, -6),
                                      (-3, 5, -6, -4),
                                      (6, -5, -10, -3)]):
        if i == 2:
            c = -a*(a**2/S(8) - b/S(2))
        elif i == 3:
            d = a*(a*(a**2*Rational(3, 256) - b/S(16)) + c/S(4))
        eq = x**4 + a*x**3 + b*x**2 + c*x + d
        ans = roots_quartic(Poly(eq, x))
        assert all(eq.subs(x, ai).n(chop=True) == 0 for ai in ans)

    # not all symbolic quartics are unresolvable
    eq = Poly(q*x + q/4 + x**4 + x**3 + 2*x**2 - Rational(1, 3), x)
    sol = roots_quartic(eq)
    assert all(verify_numerically(eq.subs(x, i), 0) for i in sol)
    z = symbols('z', negative=True)
    eq = x**4 + 2*x**3 + 3*x**2 + x*(z + 11) + 5
    zans = roots_quartic(Poly(eq, x))
    assert all(verify_numerically(eq.subs(((x, i), (z, -1))), 0) for i in zans)
    # but some are (see also issue 4989)
    # it's ok if the solution is not Piecewise, but the tests below should pass
    eq = Poly(y*x**4 + x**3 - x + z, x)
    ans = roots_quartic(eq)
    assert all(type(i) == Piecewise for i in ans)
    reps = (
        {"y": Rational(-1, 3), "z": Rational(-1, 4)},  # 4 real
        {"y": Rational(-1, 3), "z": Rational(-1, 2)},  # 2 real
        {"y": Rational(-1, 3), "z": -2})  # 0 real
    for rep in reps:
        sol = roots_quartic(Poly(eq.subs(rep), x))
        assert all(verify_numerically(w.subs(rep) - s, 0) for w, s in zip(ans, sol))


def test_issue_21287():
    assert not any(isinstance(i, Piecewise) for i in roots_quartic(
        Poly(x**4 - x**2*(3 + 5*I) + 2*x*(-1 + I) - 1 + 3*I, x)))


def test_roots_quintic():
    eqs = (x**5 - 2,
            (x/2 + 1)**5 - 5*(x/2 + 1) + 12,
            x**5 - 110*x**3 - 55*x**2 + 2310*x + 979)
    for eq in eqs:
        roots = roots_quintic(Poly(eq))
        assert len(roots) == 5
        assert all(eq.subs(x, r.n(10)).n(chop = 1e-5) == 0 for r in roots)


def test_roots_cyclotomic():
    assert roots_cyclotomic(cyclotomic_poly(1, x, polys=True)) == [1]
    assert roots_cyclotomic(cyclotomic_poly(2, x, polys=True)) == [-1]
    assert roots_cyclotomic(cyclotomic_poly(
        3, x, polys=True)) == [Rational(-1, 2) - I*sqrt(3)/2, Rational(-1, 2) + I*sqrt(3)/2]
    assert roots_cyclotomic(cyclotomic_poly(4, x, polys=True)) == [-I, I]
    assert roots_cyclotomic(cyclotomic_poly(
        6, x, polys=True)) == [S.Half - I*sqrt(3)/2, S.Half + I*sqrt(3)/2]

    assert roots_cyclotomic(cyclotomic_poly(7, x, polys=True)) == [
        -cos(pi/7) - I*sin(pi/7),
        -cos(pi/7) + I*sin(pi/7),
        -cos(pi*Rational(3, 7)) - I*sin(pi*Rational(3, 7)),
        -cos(pi*Rational(3, 7)) + I*sin(pi*Rational(3, 7)),
        cos(pi*Rational(2, 7)) - I*sin(pi*Rational(2, 7)),
        cos(pi*Rational(2, 7)) + I*sin(pi*Rational(2, 7)),
    ]

    assert roots_cyclotomic(cyclotomic_poly(8, x, polys=True)) == [
        -sqrt(2)/2 - I*sqrt(2)/2,
        -sqrt(2)/2 + I*sqrt(2)/2,
        sqrt(2)/2 - I*sqrt(2)/2,
        sqrt(2)/2 + I*sqrt(2)/2,
    ]

    assert roots_cyclotomic(cyclotomic_poly(12, x, polys=True)) == [
        -sqrt(3)/2 - I/2,
        -sqrt(3)/2 + I/2,
        sqrt(3)/2 - I/2,
        sqrt(3)/2 + I/2,
    ]

    assert roots_cyclotomic(
        cyclotomic_poly(1, x, polys=True), factor=True) == [1]
    assert roots_cyclotomic(
        cyclotomic_poly(2, x, polys=True), factor=True) == [-1]

    assert roots_cyclotomic(cyclotomic_poly(3, x, polys=True), factor=True) == \
        [-root(-1, 3), -1 + root(-1, 3)]
    assert roots_cyclotomic(cyclotomic_poly(4, x, polys=True), factor=True) == \
        [-I, I]
    assert roots_cyclotomic(cyclotomic_poly(5, x, polys=True), factor=True) == \
        [-root(-1, 5), -root(-1, 5)**3, root(-1, 5)**2, -1 - root(-1, 5)**2 + root(-1, 5) + root(-1, 5)**3]

    assert roots_cyclotomic(cyclotomic_poly(6, x, polys=True), factor=True) == \
        [1 - root(-1, 3), root(-1, 3)]


def test_roots_binomial():
    assert roots_binomial(Poly(5*x, x)) == [0]
    assert roots_binomial(Poly(5*x**4, x)) == [0, 0, 0, 0]
    assert roots_binomial(Poly(5*x + 2, x)) == [Rational(-2, 5)]

    A = 10**Rational(3, 4)/10

    assert roots_binomial(Poly(5*x**4 + 2, x)) == \
        [-A - A*I, -A + A*I, A - A*I, A + A*I]
    _check(roots_binomial(Poly(x**8 - 2)))

    a1 = Symbol('a1', nonnegative=True)
    b1 = Symbol('b1', nonnegative=True)

    r0 = roots_quadratic(Poly(a1*x**2 + b1, x))
    r1 = roots_binomial(Poly(a1*x**2 + b1, x))

    assert powsimp(r0[0]) == powsimp(r1[0])
    assert powsimp(r0[1]) == powsimp(r1[1])
    for a, b, s, n in product((1, 2), (1, 2), (-1, 1), (2, 3, 4, 5)):
        if a == b and a != 1:  # a == b == 1 is sufficient
            continue
        p = Poly(a*x**n + s*b)
        ans = roots_binomial(p)
        assert ans == _nsort(ans)

    # issue 8813
    assert roots(Poly(2*x**3 - 16*y**3, x)) == {
        2*y*(Rational(-1, 2) - sqrt(3)*I/2): 1,
        2*y: 1,
        2*y*(Rational(-1, 2) + sqrt(3)*I/2): 1}


def test_roots_preprocessing():
    f = a*y*x**2 + y - b

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1
    assert poly == Poly(a*y*x**2 + y - b, x)

    f = c**3*x**3 + c**2*x**2 + c*x + a

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1/c
    assert poly == Poly(x**3 + x**2 + x + a, x)

    f = c**3*x**3 + c**2*x**2 + a

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1/c
    assert poly == Poly(x**3 + x**2 + a, x)

    f = c**3*x**3 + c*x + a

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1/c
    assert poly == Poly(x**3 + x + a, x)

    f = c**3*x**3 + a

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1/c
    assert poly == Poly(x**3 + a, x)

    E, F, J, L = symbols("E,F,J,L")

    f = -21601054687500000000*E**8*J**8/L**16 + \
        508232812500000000*F*x*E**7*J**7/L**14 - \
        4269543750000000*E**6*F**2*J**6*x**2/L**12 + \
        16194716250000*E**5*F**3*J**5*x**3/L**10 - \
        27633173750*E**4*F**4*J**4*x**4/L**8 + \
        14840215*E**3*F**5*J**3*x**5/L**6 + \
        54794*E**2*F**6*J**2*x**6/(5*L**4) - \
        1153*E*J*F**7*x**7/(80*L**2) + \
        633*F**8*x**8/160000

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 20*E*J/(F*L**2)
    assert poly == 633*x**8 - 115300*x**7 + 4383520*x**6 + 296804300*x**5 - 27633173750*x**4 + \
        809735812500*x**3 - 10673859375000*x**2 + 63529101562500*x - 135006591796875

    f = Poly(-y**2 + x**2*exp(x), y, domain=ZZ[x, exp(x)])
    g = Poly(-y**2 + exp(x), y, domain=ZZ[exp(x)])

    assert preprocess_roots(f) == (x, g)


def test_roots0():
    assert roots(1, x) == {}
    assert roots(x, x) == {S.Zero: 1}
    assert roots(x**9, x) == {S.Zero: 9}
    assert roots(((x - 2)*(x + 3)*(x - 4)).expand(), x) == {-S(3): 1, S(2): 1, S(4): 1}

    assert roots(2*x + 1, x) == {Rational(-1, 2): 1}
    assert roots((2*x + 1)**2, x) == {Rational(-1, 2): 2}
    assert roots((2*x + 1)**5, x) == {Rational(-1, 2): 5}
    assert roots((2*x + 1)**10, x) == {Rational(-1, 2): 10}

    assert roots(x**4 - 1, x) == {I: 1, S.One: 1, -S.One: 1, -I: 1}
    assert roots((x**4 - 1)**2, x) == {I: 2, S.One: 2, -S.One: 2, -I: 2}

    assert roots(((2*x - 3)**2).expand(), x) == {Rational( 3, 2): 2}
    assert roots(((2*x + 3)**2).expand(), x) == {Rational(-3, 2): 2}

    assert roots(((2*x - 3)**3).expand(), x) == {Rational( 3, 2): 3}
    assert roots(((2*x + 3)**3).expand(), x) == {Rational(-3, 2): 3}

    assert roots(((2*x - 3)**5).expand(), x) == {Rational( 3, 2): 5}
    assert roots(((2*x + 3)**5).expand(), x) == {Rational(-3, 2): 5}

    assert roots(((a*x - b)**5).expand(), x) == { b/a: 5}
    assert roots(((a*x + b)**5).expand(), x) == {-b/a: 5}

    assert roots(x**2 + (-a - 1)*x + a, x) == {a: 1, S.One: 1}

    assert roots(x**4 - 2*x**2 + 1, x) == {S.One: 2, S.NegativeOne: 2}

    assert roots(x**6 - 4*x**4 + 4*x**3 - x**2, x) == \
        {S.One: 2, -1 - sqrt(2): 1, S.Zero: 2, -1 + sqrt(2): 1}

    assert roots(x**8 - 1, x) == {
        sqrt(2)/2 + I*sqrt(2)/2: 1,
        sqrt(2)/2 - I*sqrt(2)/2: 1,
        -sqrt(2)/2 + I*sqrt(2)/2: 1,
        -sqrt(2)/2 - I*sqrt(2)/2: 1,
        S.One: 1, -S.One: 1, I: 1, -I: 1
    }

    f = -2016*x**2 - 5616*x**3 - 2056*x**4 + 3324*x**5 + 2176*x**6 - \
        224*x**7 - 384*x**8 - 64*x**9

    assert roots(f) == {S.Zero: 2, -S(2): 2, S(2): 1, Rational(-7, 2): 1,
                Rational(-3, 2): 1, Rational(-1, 2): 1, Rational(3, 2): 1}

    assert roots((a + b + c)*x - (a + b + c + d), x) == {(a + b + c + d)/(a + b + c): 1}

    assert roots(x**3 + x**2 - x + 1, x, cubics=False) == {}
    assert roots(((x - 2)*(
        x + 3)*(x - 4)).expand(), x, cubics=False) == {-S(3): 1, S(2): 1, S(4): 1}
    assert roots(((x - 2)*(x + 3)*(x - 4)*(x - 5)).expand(), x, cubics=False) == \
        {-S(3): 1, S(2): 1, S(4): 1, S(5): 1}
    assert roots(x**3 + 2*x**2 + 4*x + 8, x) == {-S(2): 1, -2*I: 1, 2*I: 1}
    assert roots(x**3 + 2*x**2 + 4*x + 8, x, cubics=True) == \
        {-2*I: 1, 2*I: 1, -S(2): 1}
    assert roots((x**2 - x)*(x**3 + 2*x**2 + 4*x + 8), x ) == \
        {S.One: 1, S.Zero: 1, -S(2): 1, -2*I: 1, 2*I: 1}

    r1_2, r1_3 = S.Half, Rational(1, 3)

    x0 = (3*sqrt(33) + 19)**r1_3
    x1 = 4/x0/3
    x2 = x0/3
    x3 = sqrt(3)*I/2
    x4 = x3 - r1_2
    x5 = -x3 - r1_2
    assert roots(x**3 + x**2 - x + 1, x, cubics=True) == {
        -x1 - x2 - r1_3: 1,
        -x1/x4 - x2*x4 - r1_3: 1,
        -x1/x5 - x2*x5 - r1_3: 1,
    }

    f = (x**2 + 2*x + 3).subs(x, 2*x**2 + 3*x).subs(x, 5*x - 4)

    r13_20, r1_20 = [ Rational(*r)
        for r in ((13, 20), (1, 20)) ]

    s2 = sqrt(2)
    assert roots(f, x) == {
        r13_20 + r1_20*sqrt(1 - 8*I*s2): 1,
        r13_20 - r1_20*sqrt(1 - 8*I*s2): 1,
        r13_20 + r1_20*sqrt(1 + 8*I*s2): 1,
        r13_20 - r1_20*sqrt(1 + 8*I*s2): 1,
    }

    f = x**4 + x**3 + x**2 + x + 1

    r1_4, r1_8, r5_8 = [ Rational(*r) for r in ((1, 4), (1, 8), (5, 8)) ]

    assert roots(f, x) == {
        -r1_4 + r1_4*5**r1_2 + I*(r5_8 + r1_8*5**r1_2)**r1_2: 1,
        -r1_4 + r1_4*5**r1_2 - I*(r5_8 + r1_8*5**r1_2)**r1_2: 1,
        -r1_4 - r1_4*5**r1_2 + I*(r5_8 - r1_8*5**r1_2)**r1_2: 1,
        -r1_4 - r1_4*5**r1_2 - I*(r5_8 - r1_8*5**r1_2)**r1_2: 1,
    }

    f = z**3 + (-2 - y)*z**2 + (1 + 2*y - 2*x**2)*z - y + 2*x**2

    assert roots(f, z) == {
        S.One: 1,
        S.Half + S.Half*y + S.Half*sqrt(1 - 2*y + y**2 + 8*x**2): 1,
        S.Half + S.Half*y - S.Half*sqrt(1 - 2*y + y**2 + 8*x**2): 1,
    }

    assert roots(a*b*c*x**3 + 2*x**2 + 4*x + 8, x, cubics=False) == {}
    assert roots(a*b*c*x**3 + 2*x**2 + 4*x + 8, x, cubics=True) != {}

    assert roots(x**4 - 1, x, filter='Z') == {S.One: 1, -S.One: 1}
    assert roots(x**4 - 1, x, filter='I') == {I: 1, -I: 1}

    assert roots((x - 1)*(x + 1), x) == {S.One: 1, -S.One: 1}
    assert roots(
        (x - 1)*(x + 1), x, predicate=lambda r: r.is_positive) == {S.One: 1}

    assert roots(x**4 - 1, x, filter='Z', multiple=True) == [-S.One, S.One]
    assert roots(x**4 - 1, x, filter='I', multiple=True) == [I, -I]

    ar, br = symbols('a, b', real=True)
    p = x**2*(ar-br)**2 + 2*x*(br-ar) + 1
    assert roots(p, x, filter='R') == {1/(ar - br): 2}

    assert roots(x**3, x, multiple=True) == [S.Zero, S.Zero, S.Zero]
    assert roots(1234, x, multiple=True) == []

    f = x**6 - x**5 + x**4 - x**3 + x**2 - x + 1

    assert roots(f) == {
        -I*sin(pi/7) + cos(pi/7): 1,
        -I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 1,
        -I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 1,
        I*sin(pi/7) + cos(pi/7): 1,
        I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 1,
        I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 1,
    }

    g = ((x**2 + 1)*f**2).expand()

    assert roots(g) == {
        -I*sin(pi/7) + cos(pi/7): 2,
        -I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 2,
        -I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 2,
        I*sin(pi/7) + cos(pi/7): 2,
        I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 2,
        I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 2,
        -I: 1, I: 1,
    }

    r = roots(x**3 + 40*x + 64)
    real_root = [rx for rx in r if rx.is_real][0]
    cr = 108 + 6*sqrt(1074)
    assert real_root == -2*root(cr, 3)/3 + 20/root(cr, 3)

    eq = Poly((7 + 5*sqrt(2))*x**3 + (-6 - 4*sqrt(2))*x**2 + (-sqrt(2) - 1)*x + 2, x, domain='EX')
    assert roots(eq) == {-1 + sqrt(2): 1, -2 + 2*sqrt(2): 1, -sqrt(2) + 1: 1}

    eq = Poly(41*x**5 + 29*sqrt(2)*x**5 - 153*x**4 - 108*sqrt(2)*x**4 +
    175*x**3 + 125*sqrt(2)*x**3 - 45*x**2 - 30*sqrt(2)*x**2 - 26*sqrt(2)*x -
    26*x + 24, x, domain='EX')
    assert roots(eq) == {-sqrt(2) + 1: 1, -2 + 2*sqrt(2): 1, -1 + sqrt(2): 1,
                         -4 + 4*sqrt(2): 1, -3 + 3*sqrt(2): 1}

    eq = Poly(x**3 - 2*x**2 + 6*sqrt(2)*x**2 - 8*sqrt(2)*x + 23*x - 14 +
            14*sqrt(2), x, domain='EX')
    assert roots(eq) == {-2*sqrt(2) + 2: 1, -2*sqrt(2) + 1: 1, -2*sqrt(2) - 1: 1}

    assert roots(Poly((x + sqrt(2))**3 - 7, x, domain='EX')) == \
        {-sqrt(2) + root(7, 3)*(-S.Half - sqrt(3)*I/2): 1,
         -sqrt(2) + root(7, 3)*(-S.Half + sqrt(3)*I/2): 1,
         -sqrt(2) + root(7, 3): 1}

def test_roots_slow():
    """Just test that calculating these roots does not hang. """
    a, b, c, d, x = symbols("a,b,c,d,x")

    f1 = x**2*c + (a/b) + x*c*d - a
    f2 = x**2*(a + b*(c - d)*a) + x*a*b*c/(b*d - d) + (a*d - c/d)

    assert list(roots(f1, x).values()) == [1, 1]
    assert list(roots(f2, x).values()) == [1, 1]

    (zz, yy, xx, zy, zx, yx, k) = symbols("zz,yy,xx,zy,zx,yx,k")

    e1 = (zz - k)*(yy - k)*(xx - k) + zy*yx*zx + zx - zy - yx
    e2 = (zz - k)*yx*yx + zx*(yy - k)*zx + zy*zy*(xx - k)

    assert list(roots(e1 - e2, k).values()) == [1, 1, 1]

    f = x**3 + 2*x**2 + 8
    R = list(roots(f).keys())

    assert not any(i for i in [f.subs(x, ri).n(chop=True) for ri in R])


def test_roots_inexact():
    R1 = roots(x**2 + x + 1, x, multiple=True)
    R2 = roots(x**2 + x + 1.0, x, multiple=True)

    for r1, r2 in zip(R1, R2):
        assert abs(r1 - r2) < 1e-12

    f = x**4 + 3.0*sqrt(2.0)*x**3 - (78.0 + 24.0*sqrt(3.0))*x**2 \
        + 144.0*(2*sqrt(3.0) + 9.0)

    R1 = roots(f, multiple=True)
    R2 = (-12.7530479110482, -3.85012393732929,
          4.89897948556636, 7.46155167569183)

    for r1, r2 in zip(R1, R2):
        assert abs(r1 - r2) < 1e-10


def test_roots_preprocessed():
    E, F, J, L = symbols("E,F,J,L")

    f = -21601054687500000000*E**8*J**8/L**16 + \
        508232812500000000*F*x*E**7*J**7/L**14 - \
        4269543750000000*E**6*F**2*J**6*x**2/L**12 + \
        16194716250000*E**5*F**3*J**5*x**3/L**10 - \
        27633173750*E**4*F**4*J**4*x**4/L**8 + \
        14840215*E**3*F**5*J**3*x**5/L**6 + \
        54794*E**2*F**6*J**2*x**6/(5*L**4) - \
        1153*E*J*F**7*x**7/(80*L**2) + \
        633*F**8*x**8/160000

    assert roots(f, x) == {}

    R1 = roots(f.evalf(), x, multiple=True)
    R2 = [-1304.88375606366, 97.1168816800648, 186.946430171876, 245.526792947065,
          503.441004174773, 791.549343830097, 1273.16678129348, 1850.10650616851]

    w = Wild('w')
    p = w*E*J/(F*L**2)

    assert len(R1) == len(R2)

    for r1, r2 in zip(R1, R2):
        match = r1.match(p)
        assert match is not None and abs(match[w] - r2) < 1e-10


def test_roots_strict():
    assert roots(x**2 - 2*x + 1, strict=False) == {1: 2}
    assert roots(x**2 - 2*x + 1, strict=True) == {1: 2}

    assert roots(x**6 - 2*x**5 - x**2 + 3*x - 2, strict=False) == {2: 1}
    raises(UnsolvableFactorError, lambda: roots(x**6 - 2*x**5 - x**2 + 3*x - 2, strict=True))


def test_roots_mixed():
    f = -1936 - 5056*x - 7592*x**2 + 2704*x**3 - 49*x**4

    _re, _im = intervals(f, all=True)
    _nroots = nroots(f)
    _sroots = roots(f, multiple=True)

    _re = [ Interval(a, b) for (a, b), _ in _re ]
    _im = [ Interval(re(a), re(b))*Interval(im(a), im(b)) for (a, b),
            _ in _im ]

    _intervals = _re + _im
    _sroots = [ r.evalf() for r in _sroots ]

    _nroots = sorted(_nroots, key=lambda x: x.sort_key())
    _sroots = sorted(_sroots, key=lambda x: x.sort_key())

    for _roots in (_nroots, _sroots):
        for i, r in zip(_intervals, _roots):
            if r.is_real:
                assert r in i
            else:
                assert (re(r), im(r)) in i


def test_root_factors():
    assert root_factors(Poly(1, x)) == [Poly(1, x)]
    assert root_factors(Poly(x, x)) == [Poly(x, x)]

    assert root_factors(x**2 - 1, x) == [x + 1, x - 1]
    assert root_factors(x**2 - y, x) == [x - sqrt(y), x + sqrt(y)]

    assert root_factors((x**4 - 1)**2) == \
        [x + 1, x + 1, x - 1, x - 1, x - I, x - I, x + I, x + I]

    assert root_factors(Poly(x**4 - 1, x), filter='Z') == \
        [Poly(x + 1, x), Poly(x - 1, x), Poly(x**2 + 1, x)]
    assert root_factors(8*x**2 + 12*x**4 + 6*x**6 + x**8, x, filter='Q') == \
        [x, x, x**6 + 6*x**4 + 12*x**2 + 8]


@slow
def test_nroots1():
    n = 64
    p = legendre_poly(n, x, polys=True)

    raises(mpmath.mp.NoConvergence, lambda: p.nroots(n=3, maxsteps=5))

    roots = p.nroots(n=3)
    # The order of roots matters. They are ordered from smallest to the
    # largest.
    assert [str(r) for r in roots] == \
            ['-0.999', '-0.996', '-0.991', '-0.983', '-0.973', '-0.961',
            '-0.946', '-0.930', '-0.911', '-0.889', '-0.866', '-0.841',
            '-0.813', '-0.784', '-0.753', '-0.720', '-0.685', '-0.649',
            '-0.611', '-0.572', '-0.531', '-0.489', '-0.446', '-0.402',
            '-0.357', '-0.311', '-0.265', '-0.217', '-0.170', '-0.121',
            '-0.0730', '-0.0243', '0.0243', '0.0730', '0.121', '0.170',
            '0.217', '0.265', '0.311', '0.357', '0.402', '0.446', '0.489',
            '0.531', '0.572', '0.611', '0.649', '0.685', '0.720', '0.753',
            '0.784', '0.813', '0.841', '0.866', '0.889', '0.911', '0.930',
            '0.946', '0.961', '0.973', '0.983', '0.991', '0.996', '0.999']

def test_nroots2():
    p = Poly(x**5 + 3*x + 1, x)

    roots = p.nroots(n=3)
    # The order of roots matters. The roots are ordered by their real
    # components (if they agree, then by their imaginary components),
    # with real roots appearing first.
    assert [str(r) for r in roots] == \
            ['-0.332', '-0.839 - 0.944*I', '-0.839 + 0.944*I',
                '1.01 - 0.937*I', '1.01 + 0.937*I']

    roots = p.nroots(n=5)
    assert [str(r) for r in roots] == \
            ['-0.33199', '-0.83907 - 0.94385*I', '-0.83907 + 0.94385*I',
              '1.0051 - 0.93726*I', '1.0051 + 0.93726*I']


def test_roots_composite():
    assert len(roots(Poly(y**3 + y**2*sqrt(x) + y + x, y, composite=True))) == 3


def test_issue_19113():
    eq = cos(x)**3 - cos(x) + 1
    raises(PolynomialError, lambda: roots(eq))


def test_issue_17454():
    assert roots([1, -3*(-4 - 4*I)**2/8 + 12*I, 0], multiple=True) == [0, 0]


def test_issue_20913():
    assert Poly(x + 9671406556917067856609794, x).real_roots() == [-9671406556917067856609794]
    assert Poly(x**3 + 4, x).real_roots() == [-2**(S(2)/3)]


def test_issue_22768():
    e = Rational(1, 3)
    r = (-1/a)**e*(a + 1)**(5*e)
    assert roots(Poly(a*x**3 + (a + 1)**5, x)) == {
        r: 1,
        -r*(1 + sqrt(3)*I)/2: 1,
        r*(-1 + sqrt(3)*I)/2: 1}
