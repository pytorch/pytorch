"""Tests for the implementation of RootOf class and related tools. """

from sympy.polys.polytools import Poly
import sympy.polys.rootoftools as rootoftools
from sympy.polys.rootoftools import (rootof, RootOf, CRootOf, RootSum,
    _pure_key_dict as D)

from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    GeneratorsNeeded,
    PolynomialError,
)

from sympy.core.function import (Function, Lambda)
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import tan
from sympy.integrals.integrals import Integral
from sympy.polys.orthopolys import legendre_poly
from sympy.solvers.solvers import solve


from sympy.testing.pytest import raises, slow
from sympy.core.expr import unchanged

from sympy.abc import a, b, x, y, z, r


def test_CRootOf___new__():
    assert rootof(x, 0) == 0
    assert rootof(x, -1) == 0

    assert rootof(x, S.Zero) == 0

    assert rootof(x - 1, 0) == 1
    assert rootof(x - 1, -1) == 1

    assert rootof(x + 1, 0) == -1
    assert rootof(x + 1, -1) == -1

    assert rootof(x**2 + 2*x + 3, 0) == -1 - I*sqrt(2)
    assert rootof(x**2 + 2*x + 3, 1) == -1 + I*sqrt(2)
    assert rootof(x**2 + 2*x + 3, -1) == -1 + I*sqrt(2)
    assert rootof(x**2 + 2*x + 3, -2) == -1 - I*sqrt(2)

    r = rootof(x**2 + 2*x + 3, 0, radicals=False)
    assert isinstance(r, RootOf) is True

    r = rootof(x**2 + 2*x + 3, 1, radicals=False)
    assert isinstance(r, RootOf) is True

    r = rootof(x**2 + 2*x + 3, -1, radicals=False)
    assert isinstance(r, RootOf) is True

    r = rootof(x**2 + 2*x + 3, -2, radicals=False)
    assert isinstance(r, RootOf) is True

    assert rootof((x - 1)*(x + 1), 0, radicals=False) == -1
    assert rootof((x - 1)*(x + 1), 1, radicals=False) == 1
    assert rootof((x - 1)*(x + 1), -1, radicals=False) == 1
    assert rootof((x - 1)*(x + 1), -2, radicals=False) == -1

    assert rootof((x - 1)*(x + 1), 0, radicals=True) == -1
    assert rootof((x - 1)*(x + 1), 1, radicals=True) == 1
    assert rootof((x - 1)*(x + 1), -1, radicals=True) == 1
    assert rootof((x - 1)*(x + 1), -2, radicals=True) == -1

    assert rootof((x - 1)*(x**3 + x + 3), 0) == rootof(x**3 + x + 3, 0)
    assert rootof((x - 1)*(x**3 + x + 3), 1) == 1
    assert rootof((x - 1)*(x**3 + x + 3), 2) == rootof(x**3 + x + 3, 1)
    assert rootof((x - 1)*(x**3 + x + 3), 3) == rootof(x**3 + x + 3, 2)
    assert rootof((x - 1)*(x**3 + x + 3), -1) == rootof(x**3 + x + 3, 2)
    assert rootof((x - 1)*(x**3 + x + 3), -2) == rootof(x**3 + x + 3, 1)
    assert rootof((x - 1)*(x**3 + x + 3), -3) == 1
    assert rootof((x - 1)*(x**3 + x + 3), -4) == rootof(x**3 + x + 3, 0)

    assert rootof(x**4 + 3*x**3, 0) == -3
    assert rootof(x**4 + 3*x**3, 1) == 0
    assert rootof(x**4 + 3*x**3, 2) == 0
    assert rootof(x**4 + 3*x**3, 3) == 0

    raises(GeneratorsNeeded, lambda: rootof(0, 0))
    raises(GeneratorsNeeded, lambda: rootof(1, 0))

    raises(PolynomialError, lambda: rootof(Poly(0, x), 0))
    raises(PolynomialError, lambda: rootof(Poly(1, x), 0))
    raises(PolynomialError, lambda: rootof(x - y, 0))
    # issue 8617
    raises(PolynomialError, lambda: rootof(exp(x), 0))

    raises(NotImplementedError, lambda: rootof(x**3 - x + sqrt(2), 0))
    raises(NotImplementedError, lambda: rootof(x**3 - x + I, 0))

    raises(IndexError, lambda: rootof(x**2 - 1, -4))
    raises(IndexError, lambda: rootof(x**2 - 1, -3))
    raises(IndexError, lambda: rootof(x**2 - 1, 2))
    raises(IndexError, lambda: rootof(x**2 - 1, 3))
    raises(ValueError, lambda: rootof(x**2 - 1, x))

    assert rootof(Poly(x - y, x), 0) == y

    assert rootof(Poly(x**2 - y, x), 0) == -sqrt(y)
    assert rootof(Poly(x**2 - y, x), 1) == sqrt(y)

    assert rootof(Poly(x**3 - y, x), 0) == y**Rational(1, 3)

    assert rootof(y*x**3 + y*x + 2*y, x, 0) == -1
    raises(NotImplementedError, lambda: rootof(x**3 + x + 2*y, x, 0))

    assert rootof(x**3 + x + 1, 0).is_commutative is True


def test_CRootOf_attributes():
    r = rootof(x**3 + x + 3, 0)
    assert r.is_number
    assert r.free_symbols == set()
    # if the following assertion fails then multivariate polynomials
    # are apparently supported and the RootOf.free_symbols routine
    # should be changed to return whatever symbols would not be
    # the PurePoly dummy symbol
    raises(NotImplementedError, lambda: rootof(Poly(x**3 + y*x + 1, x), 0))


def test_CRootOf___eq__():
    assert (rootof(x**3 + x + 3, 0) == rootof(x**3 + x + 3, 0)) is True
    assert (rootof(x**3 + x + 3, 0) == rootof(x**3 + x + 3, 1)) is False
    assert (rootof(x**3 + x + 3, 1) == rootof(x**3 + x + 3, 1)) is True
    assert (rootof(x**3 + x + 3, 1) == rootof(x**3 + x + 3, 2)) is False
    assert (rootof(x**3 + x + 3, 2) == rootof(x**3 + x + 3, 2)) is True

    assert (rootof(x**3 + x + 3, 0) == rootof(y**3 + y + 3, 0)) is True
    assert (rootof(x**3 + x + 3, 0) == rootof(y**3 + y + 3, 1)) is False
    assert (rootof(x**3 + x + 3, 1) == rootof(y**3 + y + 3, 1)) is True
    assert (rootof(x**3 + x + 3, 1) == rootof(y**3 + y + 3, 2)) is False
    assert (rootof(x**3 + x + 3, 2) == rootof(y**3 + y + 3, 2)) is True


def test_CRootOf___eval_Eq__():
    f = Function('f')
    eq = x**3 + x + 3
    r = rootof(eq, 2)
    r1 = rootof(eq, 1)
    assert Eq(r, r1) is S.false
    assert Eq(r, r) is S.true
    assert unchanged(Eq, r, x)
    assert Eq(r, 0) is S.false
    assert Eq(r, S.Infinity) is S.false
    assert Eq(r, I) is S.false
    assert unchanged(Eq, r, f(0))
    sol = solve(eq)
    for s in sol:
        if s.is_real:
            assert Eq(r, s) is S.false
    r = rootof(eq, 0)
    for s in sol:
        if s.is_real:
            assert Eq(r, s) is S.true
    eq = x**3 + x + 1
    sol = solve(eq)
    assert [Eq(rootof(eq, i), j) for i in range(3) for j in sol
        ].count(True) == 3
    assert Eq(rootof(eq, 0), 1 + S.ImaginaryUnit) == False


def test_CRootOf_is_real():
    assert rootof(x**3 + x + 3, 0).is_real is True
    assert rootof(x**3 + x + 3, 1).is_real is False
    assert rootof(x**3 + x + 3, 2).is_real is False


def test_CRootOf_is_complex():
    assert rootof(x**3 + x + 3, 0).is_complex is True


def test_CRootOf_is_algebraic():
    assert rootof(x**3 + x + 3, 0).is_algebraic is True
    assert rootof(x**3 + x + 3, 1).is_algebraic is True
    assert rootof(x**3 + x + 3, 2).is_algebraic is True


def test_CRootOf_subs():
    assert rootof(x**3 + x + 1, 0).subs(x, y) == rootof(y**3 + y + 1, 0)


def test_CRootOf_diff():
    assert rootof(x**3 + x + 1, 0).diff(x) == 0
    assert rootof(x**3 + x + 1, 0).diff(y) == 0

@slow
def test_CRootOf_evalf():
    real = rootof(x**3 + x + 3, 0).evalf(n=20)

    assert real.epsilon_eq(Float("-1.2134116627622296341"))

    re, im = rootof(x**3 + x + 3, 1).evalf(n=20).as_real_imag()

    assert re.epsilon_eq( Float("0.60670583138111481707"))
    assert im.epsilon_eq(-Float("1.45061224918844152650"))

    re, im = rootof(x**3 + x + 3, 2).evalf(n=20).as_real_imag()

    assert re.epsilon_eq(Float("0.60670583138111481707"))
    assert im.epsilon_eq(Float("1.45061224918844152650"))

    p = legendre_poly(4, x, polys=True)
    roots = [str(r.n(17)) for r in p.real_roots()]
    # magnitudes are given by
    # sqrt(3/S(7) - 2*sqrt(6/S(5))/7)
    #   and
    # sqrt(3/S(7) + 2*sqrt(6/S(5))/7)
    assert roots == [
            "-0.86113631159405258",
            "-0.33998104358485626",
             "0.33998104358485626",
             "0.86113631159405258",
             ]

    re = rootof(x**5 - 5*x + 12, 0).evalf(n=20)
    assert re.epsilon_eq(Float("-1.84208596619025438271"))

    re, im = rootof(x**5 - 5*x + 12, 1).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("-0.351854240827371999559"))
    assert im.epsilon_eq(Float("-1.709561043370328882010"))

    re, im = rootof(x**5 - 5*x + 12, 2).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("-0.351854240827371999559"))
    assert im.epsilon_eq(Float("+1.709561043370328882010"))

    re, im = rootof(x**5 - 5*x + 12, 3).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("+1.272897223922499190910"))
    assert im.epsilon_eq(Float("-0.719798681483861386681"))

    re, im = rootof(x**5 - 5*x + 12, 4).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("+1.272897223922499190910"))
    assert im.epsilon_eq(Float("+0.719798681483861386681"))

    # issue 6393
    assert str(rootof(x**5 + 2*x**4 + x**3 - 68719476736, 0).n(3)) == '147.'
    eq = (531441*x**11 + 3857868*x**10 + 13730229*x**9 + 32597882*x**8 +
        55077472*x**7 + 60452000*x**6 + 32172064*x**5 - 4383808*x**4 -
        11942912*x**3 - 1506304*x**2 + 1453312*x + 512)
    a, b = rootof(eq, 1).n(2).as_real_imag()
    c, d = rootof(eq, 2).n(2).as_real_imag()
    assert a == c
    assert b < d
    assert b == -d
    # issue 6451
    r = rootof(legendre_poly(64, x), 7)
    assert r.n(2) == r.n(100).n(2)
    # issue 9019
    r0 = rootof(x**2 + 1, 0, radicals=False)
    r1 = rootof(x**2 + 1, 1, radicals=False)
    assert r0.n(4) == Float(-1.0, 4) * I
    assert r1.n(4) == Float(1.0, 4) * I

    # make sure verification is used in case a max/min traps the "root"
    assert str(rootof(4*x**5 + 16*x**3 + 12*x**2 + 7, 0).n(3)) == '-0.976'

    # watch out for UnboundLocalError
    c = CRootOf(90720*x**6 - 4032*x**4 + 84*x**2 - 1, 0)
    assert c._eval_evalf(2)  # doesn't fail

    # watch out for imaginary parts that don't want to evaluate
    assert str(RootOf(x**16 + 32*x**14 + 508*x**12 + 5440*x**10 +
        39510*x**8 + 204320*x**6 + 755548*x**4 + 1434496*x**2 +
        877969, 10).n(2)) == '-3.4*I'
    assert abs(RootOf(x**4 + 10*x**2 + 1, 0).n(2)) < 0.4

    # check reset and args
    r = [RootOf(x**3 + x + 3, i) for i in range(3)]
    r[0]._reset()
    for ri in r:
        i = ri._get_interval()
        ri.n(2)
        assert i != ri._get_interval()
        ri._reset()
        assert i == ri._get_interval()
        assert i == i.func(*i.args)


def test_issue_24978():
    # Irreducible poly with negative leading coeff is normalized
    # (factor of -1 is extracted), before being stored as CRootOf.poly.
    f = -x**2 + 2
    r = CRootOf(f, 0)
    assert r.poly.as_expr() == x**2 - 2
    # An action that prompts calculation of an interval puts r.poly in
    # the cache.
    r.n()
    assert r.poly in rootoftools._reals_cache


def test_CRootOf_evalf_caching_bug():
    r = rootof(x**5 - 5*x + 12, 1)
    r.n()
    a = r._get_interval()
    r = rootof(x**5 - 5*x + 12, 1)
    r.n()
    b = r._get_interval()
    assert a == b


def test_CRootOf_real_roots():
    assert Poly(x**5 + x + 1).real_roots() == [rootof(x**3 - x**2 + 1, 0)]
    assert Poly(x**5 + x + 1).real_roots(radicals=False) == [rootof(
        x**3 - x**2 + 1, 0)]

    # https://github.com/sympy/sympy/issues/20902
    p = Poly(-3*x**4 - 10*x**3 - 12*x**2 - 6*x - 1, x, domain='ZZ')
    assert CRootOf.real_roots(p) == [S(-1), S(-1), S(-1), S(-1)/3]

    # with real algebraic coefficients
    assert Poly(x**3 + sqrt(2)*x**2 - 1, x, extension=True).real_roots() == [
        rootof(x**6 - 2*x**4 - 2*x**3 + 1, 0)
    ]
    assert Poly(x**5 + sqrt(2) * x**3 - 1, x, extension=True).real_roots() == [
        rootof(x**10 - 2*x**6 - 2*x**5 + 1, 0)
    ]
    r = rootof(y**5 + y**3 - 1, 0)
    assert Poly(x**5 + r*x - 1, x, extension=True).real_roots() ==\
    [
        rootof(x**25 - 5*x**20 + x**17 + 10*x**15 - 3*x**12 -
               10*x**10 + 3*x**7 + 6*x**5 - x**2 - 1, 0)
    ]
    # roots with multiplicity
    assert Poly((x-1) * (x-sqrt(2))**2, x, extension=True).real_roots() ==\
    [
        S(1), sqrt(2), sqrt(2)
    ]


def test_CRootOf_all_roots():
    assert Poly(x**5 + x + 1).all_roots() == [
        rootof(x**3 - x**2 + 1, 0),
        Rational(-1, 2) - sqrt(3)*I/2,
        Rational(-1, 2) + sqrt(3)*I/2,
        rootof(x**3 - x**2 + 1, 1),
        rootof(x**3 - x**2 + 1, 2),
    ]

    assert Poly(x**5 + x + 1).all_roots(radicals=False) == [
        rootof(x**3 - x**2 + 1, 0),
        rootof(x**2 + x + 1, 0, radicals=False),
        rootof(x**2 + x + 1, 1, radicals=False),
        rootof(x**3 - x**2 + 1, 1),
        rootof(x**3 - x**2 + 1, 2),
    ]

    # with real algebraic coefficients
    assert Poly(x**3 + sqrt(2)*x**2 - 1, x, extension=True).all_roots() ==\
    [
        rootof(x**6 - 2*x**4 - 2*x**3 + 1, 0),
        rootof(x**6 - 2*x**4 - 2*x**3 + 1, 2),
        rootof(x**6 - 2*x**4 - 2*x**3 + 1, 3)
    ]
    # roots with multiplicity
    assert Poly((x-1) * (x-sqrt(2))**2 * (x-I) * (x+I), x, extension=True).all_roots() ==\
    [
        S(1), sqrt(2), sqrt(2), -I, I
    ]

    # imaginary algebraic coeffs (gaussian domain)
    assert Poly(x**2 - I/2, x, extension=True).all_roots() ==\
    [
        S(1)/2 + I/2,
        -S(1)/2 - I/2
    ]


def test_CRootOf_eval_rational():
    p = legendre_poly(4, x, polys=True)
    roots = [r.eval_rational(n=18) for r in p.real_roots()]
    for root in roots:
        assert isinstance(root, Rational)
    roots = [str(root.n(17)) for root in roots]
    assert roots == [
            "-0.86113631159405258",
            "-0.33998104358485626",
             "0.33998104358485626",
             "0.86113631159405258",
             ]


def test_CRootOf_lazy():
    # irreducible poly with both real and complex roots:
    f = Poly(x**3 + 2*x + 2)

    # real root:
    CRootOf.clear_cache()
    r = CRootOf(f, 0)
    # Not yet in cache, after construction:
    assert r.poly not in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache
    r.evalf()
    # In cache after evaluation:
    assert r.poly in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache

    # complex root:
    CRootOf.clear_cache()
    r = CRootOf(f, 1)
    # Not yet in cache, after construction:
    assert r.poly not in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache
    r.evalf()
    # In cache after evaluation:
    assert r.poly in rootoftools._reals_cache
    assert r.poly in rootoftools._complexes_cache

    # composite poly with both real and complex roots:
    f = Poly((x**2 - 2)*(x**2 + 1))

    # real root:
    CRootOf.clear_cache()
    r = CRootOf(f, 0)
    # In cache immediately after construction:
    assert r.poly in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache

    # complex root:
    CRootOf.clear_cache()
    r = CRootOf(f, 2)
    # In cache immediately after construction:
    assert r.poly in rootoftools._reals_cache
    assert r.poly in rootoftools._complexes_cache


def test_RootSum___new__():
    f = x**3 + x + 3

    g = Lambda(r, log(r*x))
    s = RootSum(f, g)

    assert isinstance(s, RootSum) is True

    assert RootSum(f**2, g) == 2*RootSum(f, g)
    assert RootSum((x - 7)*f**3, g) == log(7*x) + 3*RootSum(f, g)

    # issue 5571
    assert hash(RootSum((x - 7)*f**3, g)) == hash(log(7*x) + 3*RootSum(f, g))

    raises(MultivariatePolynomialError, lambda: RootSum(x**3 + x + y))
    raises(ValueError, lambda: RootSum(x**2 + 3, lambda x: x))

    assert RootSum(f, exp) == RootSum(f, Lambda(x, exp(x)))
    assert RootSum(f, log) == RootSum(f, Lambda(x, log(x)))

    assert isinstance(RootSum(f, auto=False), RootSum) is True

    assert RootSum(f) == 0
    assert RootSum(f, Lambda(x, x)) == 0
    assert RootSum(f, Lambda(x, x**2)) == -2

    assert RootSum(f, Lambda(x, 1)) == 3
    assert RootSum(f, Lambda(x, 2)) == 6

    assert RootSum(f, auto=False).is_commutative is True

    assert RootSum(f, Lambda(x, 1/(x + x**2))) == Rational(11, 3)
    assert RootSum(f, Lambda(x, y/(x + x**2))) == Rational(11, 3)*y

    assert RootSum(x**2 - 1, Lambda(x, 3*x**2), x) == 6
    assert RootSum(x**2 - y, Lambda(x, 3*x**2), x) == 6*y

    assert RootSum(x**2 - 1, Lambda(x, z*x**2), x) == 2*z
    assert RootSum(x**2 - y, Lambda(x, z*x**2), x) == 2*z*y

    assert RootSum(
        x**2 - 1, Lambda(x, exp(x)), quadratic=True) == exp(-1) + exp(1)

    assert RootSum(x**3 + a*x + a**3, tan, x) == \
        RootSum(x**3 + x + 1, Lambda(x, tan(a*x)))
    assert RootSum(a**3*x**3 + a*x + 1, tan, x) == \
        RootSum(x**3 + x + 1, Lambda(x, tan(x/a)))


def test_RootSum_free_symbols():
    assert RootSum(x**3 + x + 3, Lambda(r, exp(r))).free_symbols == set()
    assert RootSum(x**3 + x + 3, Lambda(r, exp(a*r))).free_symbols == {a}
    assert RootSum(
        x**3 + x + y, Lambda(r, exp(a*r)), x).free_symbols == {a, y}


def test_RootSum___eq__():
    f = Lambda(x, exp(x))

    assert (RootSum(x**3 + x + 1, f) == RootSum(x**3 + x + 1, f)) is True
    assert (RootSum(x**3 + x + 1, f) == RootSum(y**3 + y + 1, f)) is True

    assert (RootSum(x**3 + x + 1, f) == RootSum(x**3 + x + 2, f)) is False
    assert (RootSum(x**3 + x + 1, f) == RootSum(y**3 + y + 2, f)) is False


def test_RootSum_doit():
    rs = RootSum(x**2 + 1, exp)

    assert isinstance(rs, RootSum) is True
    assert rs.doit() == exp(-I) + exp(I)

    rs = RootSum(x**2 + a, exp, x)

    assert isinstance(rs, RootSum) is True
    assert rs.doit() == exp(-sqrt(-a)) + exp(sqrt(-a))


def test_RootSum_evalf():
    rs = RootSum(x**2 + 1, exp)

    assert rs.evalf(n=20, chop=True).epsilon_eq(Float("1.0806046117362794348"))
    assert rs.evalf(n=15, chop=True).epsilon_eq(Float("1.08060461173628"))

    rs = RootSum(x**2 + a, exp, x)

    assert rs.evalf() == rs


def test_RootSum_diff():
    f = x**3 + x + 3

    g = Lambda(r, exp(r*x))
    h = Lambda(r, r*exp(r*x))

    assert RootSum(f, g).diff(x) == RootSum(f, h)


def test_RootSum_subs():
    f = x**3 + x + 3
    g = Lambda(r, exp(r*x))

    F = y**3 + y + 3
    G = Lambda(r, exp(r*y))

    assert RootSum(f, g).subs(y, 1) == RootSum(f, g)
    assert RootSum(f, g).subs(x, y) == RootSum(F, G)


def test_RootSum_rational():
    assert RootSum(
        z**5 - z + 1, Lambda(z, z/(x - z))) == (4*x - 5)/(x**5 - x + 1)

    f = 161*z**3 + 115*z**2 + 19*z + 1
    g = Lambda(z, z*log(
        -3381*z**4/4 - 3381*z**3/4 - 625*z**2/2 - z*Rational(125, 2) - 5 + exp(x)))

    assert RootSum(f, g).diff(x) == -(
        (5*exp(2*x) - 6*exp(x) + 4)*exp(x)/(exp(3*x) - exp(2*x) + 1))/7


def test_RootSum_independent():
    f = (x**3 - a)**2*(x**4 - b)**3

    g = Lambda(x, 5*tan(x) + 7)
    h = Lambda(x, tan(x))

    r0 = RootSum(x**3 - a, h, x)
    r1 = RootSum(x**4 - b, h, x)

    assert RootSum(f, g, x).as_ordered_terms() == [10*r0, 15*r1, 126]


def test_issue_7876():
    l1 = Poly(x**6 - x + 1, x).all_roots()
    l2 = [rootof(x**6 - x + 1, i) for i in range(6)]
    assert frozenset(l1) == frozenset(l2)


def test_issue_8316():
    f = Poly(7*x**8 - 9)
    assert len(f.all_roots()) == 8
    f = Poly(7*x**8 - 10)
    assert len(f.all_roots()) == 8


def test__imag_count():
    from sympy.polys.rootoftools import _imag_count_of_factor
    def imag_count(p):
        return sum(_imag_count_of_factor(f)*m for f, m in
        p.factor_list()[1])
    assert imag_count(Poly(x**6 + 10*x**2 + 1)) == 2
    assert imag_count(Poly(x**2)) == 0
    assert imag_count(Poly([1]*3 + [-1], x)) == 0
    assert imag_count(Poly(x**3 + 1)) == 0
    assert imag_count(Poly(x**2 + 1)) == 2
    assert imag_count(Poly(x**2 - 1)) == 0
    assert imag_count(Poly(x**4 - 1)) == 2
    assert imag_count(Poly(x**4 + 1)) == 0
    assert imag_count(Poly([1, 2, 3], x)) == 0
    assert imag_count(Poly(x**3 + x + 1)) == 0
    assert imag_count(Poly(x**4 + x + 1)) == 0
    def q(r1, r2, p):
        return Poly(((x - r1)*(x - r2)).subs(x, x**p), x)
    assert imag_count(q(-1, -2, 2)) == 4
    assert imag_count(q(-1, 2, 2)) == 2
    assert imag_count(q(1, 2, 2)) == 0
    assert imag_count(q(1, 2, 4)) == 4
    assert imag_count(q(-1, 2, 4)) == 2
    assert imag_count(q(-1, -2, 4)) == 0


def test_RootOf_is_imaginary():
    r = RootOf(x**4 + 4*x**2 + 1, 1)
    i = r._get_interval()
    assert r.is_imaginary and i.ax*i.bx <= 0


def test_is_disjoint():
    eq = x**3 + 5*x + 1
    ir = rootof(eq, 0)._get_interval()
    ii = rootof(eq, 1)._get_interval()
    assert ir.is_disjoint(ii)
    assert ii.is_disjoint(ir)


def test_pure_key_dict():
    p = D()
    assert (x in p) is False
    assert (1 in p) is False
    p[x] = 1
    assert x in p
    assert y in p
    assert p[y] == 1
    raises(KeyError, lambda: p[1])
    def dont(k):
        p[k] = 2
    raises(ValueError, lambda: dont(1))


@slow
def test_eval_approx_relative():
    CRootOf.clear_cache()
    t = [CRootOf(x**3 + 10*x + 1, i) for i in range(3)]
    assert [i.eval_rational(1e-1) for i in t] == [
        Rational(-21, 220), Rational(15, 256) - I*805/256,
        Rational(15, 256) + I*805/256]
    t[0]._reset()
    assert [i.eval_rational(1e-1, 1e-4) for i in t] == [
        Rational(-21, 220), Rational(3275, 65536) - I*414645/131072,
        Rational(3275, 65536) + I*414645/131072]
    assert S(t[0]._get_interval().dx) < 1e-1
    assert S(t[1]._get_interval().dx) < 1e-1
    assert S(t[1]._get_interval().dy) < 1e-4
    assert S(t[2]._get_interval().dx) < 1e-1
    assert S(t[2]._get_interval().dy) < 1e-4
    t[0]._reset()
    assert [i.eval_rational(1e-4, 1e-4) for i in t] == [
        Rational(-2001, 20020), Rational(6545, 131072) - I*414645/131072,
        Rational(6545, 131072) + I*414645/131072]
    assert S(t[0]._get_interval().dx) < 1e-4
    assert S(t[1]._get_interval().dx) < 1e-4
    assert S(t[1]._get_interval().dy) < 1e-4
    assert S(t[2]._get_interval().dx) < 1e-4
    assert S(t[2]._get_interval().dy) < 1e-4
    # in the following, the actual relative precision is
    # less than tested, but it should never be greater
    t[0]._reset()
    assert [i.eval_rational(n=2) for i in t] == [
        Rational(-202201, 2024022), Rational(104755, 2097152) - I*6634255/2097152,
        Rational(104755, 2097152) + I*6634255/2097152]
    assert abs(S(t[0]._get_interval().dx)/t[0]) < 1e-2
    assert abs(S(t[1]._get_interval().dx)/t[1]).n() < 1e-2
    assert abs(S(t[1]._get_interval().dy)/t[1]).n() < 1e-2
    assert abs(S(t[2]._get_interval().dx)/t[2]).n() < 1e-2
    assert abs(S(t[2]._get_interval().dy)/t[2]).n() < 1e-2
    t[0]._reset()
    assert [i.eval_rational(n=3) for i in t] == [
        Rational(-202201, 2024022), Rational(1676045, 33554432) - I*106148135/33554432,
        Rational(1676045, 33554432) + I*106148135/33554432]
    assert abs(S(t[0]._get_interval().dx)/t[0]) < 1e-3
    assert abs(S(t[1]._get_interval().dx)/t[1]).n() < 1e-3
    assert abs(S(t[1]._get_interval().dy)/t[1]).n() < 1e-3
    assert abs(S(t[2]._get_interval().dx)/t[2]).n() < 1e-3
    assert abs(S(t[2]._get_interval().dy)/t[2]).n() < 1e-3

    t[0]._reset()
    a = [i.eval_approx(2) for i in t]
    assert [str(i) for i in a] == [
        '-0.10', '0.05 - 3.2*I', '0.05 + 3.2*I']
    assert all(abs(((a[i] - t[i])/t[i]).n()) < 1e-2 for i in range(len(a)))


def test_issue_15920():
    r = rootof(x**5 - x + 1, 0)
    p = Integral(x, (x, 1, y))
    assert unchanged(Eq, r, p)


def test_issue_19113():
    eq = y**3 - y + 1
    # generator is a canonical x in RootOf
    assert str(Poly(eq).real_roots()) == '[CRootOf(x**3 - x + 1, 0)]'
    assert str(Poly(eq.subs(y, tan(y))).real_roots()
        ) == '[CRootOf(x**3 - x + 1, 0)]'
    assert str(Poly(eq.subs(y, tan(x))).real_roots()
        ) == '[CRootOf(x**3 - x + 1, 0)]'
