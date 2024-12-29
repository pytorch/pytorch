"""Tests for tools for manipulating of large commutative expressions. """

from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import (decompose_power, Factors, Term, _gcd_terms,
                                  gcd_terms, factor_terms, factor_nc, _mask_nc,
                                  _monotonic_sign)
from sympy.core.mul import _keep_coeff as _keep_coeff
from sympy.simplify.cse_opts import sub_pre
from sympy.testing.pytest import raises

from sympy.abc import a, b, t, x, y, z


def test_decompose_power():
    assert decompose_power(x) == (x, 1)
    assert decompose_power(x**2) == (x, 2)
    assert decompose_power(x**(2*y)) == (x**y, 2)
    assert decompose_power(x**(2*y/3)) == (x**(y/3), 2)
    assert decompose_power(x**(y*Rational(2, 3))) == (x**(y/3), 2)


def test_Factors():
    assert Factors() == Factors({}) == Factors(S.One)
    assert Factors().as_expr() is S.One
    assert Factors({x: 2, y: 3, sin(x): 4}).as_expr() == x**2*y**3*sin(x)**4
    assert Factors(S.Infinity) == Factors({oo: 1})
    assert Factors(S.NegativeInfinity) == Factors({oo: 1, -1: 1})
    # issue #18059:
    assert Factors((x**2)**S.Half).as_expr() == (x**2)**S.Half

    a = Factors({x: 5, y: 3, z: 7})
    b = Factors({      y: 4, z: 3, t: 10})

    assert a.mul(b) == a*b == Factors({x: 5, y: 7, z: 10, t: 10})

    assert a.div(b) == divmod(a, b) == \
        (Factors({x: 5, z: 4}), Factors({y: 1, t: 10}))
    assert a.quo(b) == a/b == Factors({x: 5, z: 4})
    assert a.rem(b) == a % b == Factors({y: 1, t: 10})

    assert a.pow(3) == a**3 == Factors({x: 15, y: 9, z: 21})
    assert b.pow(3) == b**3 == Factors({y: 12, z: 9, t: 30})

    assert a.gcd(b) == Factors({y: 3, z: 3})
    assert a.lcm(b) == Factors({x: 5, y: 4, z: 7, t: 10})

    a = Factors({x: 4, y: 7, t: 7})
    b = Factors({z: 1, t: 3})

    assert a.normal(b) == (Factors({x: 4, y: 7, t: 4}), Factors({z: 1}))

    assert Factors(sqrt(2)*x).as_expr() == sqrt(2)*x

    assert Factors(-I)*I == Factors()
    assert Factors({S.NegativeOne: S(3)})*Factors({S.NegativeOne: S.One, I: S(5)}) == \
        Factors(I)
    assert Factors(sqrt(I)*I) == Factors(I**(S(3)/2)) == Factors({I: S(3)/2})
    assert Factors({I: S(3)/2}).as_expr() == I**(S(3)/2)

    assert Factors(S(2)**x).div(S(3)**x) == \
        (Factors({S(2): x}), Factors({S(3): x}))
    assert Factors(2**(2*x + 2)).div(S(8)) == \
        (Factors({S(2): 2*x + 2}), Factors({S(8): S.One}))

    # coverage
    # /!\ things break if this is not True
    assert Factors({S.NegativeOne: Rational(3, 2)}) == Factors({I: S.One, S.NegativeOne: S.One})
    assert Factors({I: S.One, S.NegativeOne: Rational(1, 3)}).as_expr() == I*(-1)**Rational(1, 3)

    assert Factors(-1.) == Factors({S.NegativeOne: S.One, S(1.): 1})
    assert Factors(-2.) == Factors({S.NegativeOne: S.One, S(2.): 1})
    assert Factors((-2.)**x) == Factors({S(-2.): x})
    assert Factors(S(-2)) == Factors({S.NegativeOne: S.One, S(2): 1})
    assert Factors(S.Half) == Factors({S(2): -S.One})
    assert Factors(Rational(3, 2)) == Factors({S(3): S.One, S(2): S.NegativeOne})
    assert Factors({I: S.One}) == Factors(I)
    assert Factors({-1.0: 2, I: 1}) == Factors({S(1.0): 1, I: 1})
    assert Factors({S.NegativeOne: Rational(-3, 2)}).as_expr() == I
    A = symbols('A', commutative=False)
    assert Factors(2*A**2) == Factors({S(2): 1, A**2: 1})
    assert Factors(I) == Factors({I: S.One})
    assert Factors(x).normal(S(2)) == (Factors(x), Factors(S(2)))
    assert Factors(x).normal(S.Zero) == (Factors(), Factors(S.Zero))
    raises(ZeroDivisionError, lambda: Factors(x).div(S.Zero))
    assert Factors(x).mul(S(2)) == Factors(2*x)
    assert Factors(x).mul(S.Zero).is_zero
    assert Factors(x).mul(1/x).is_one
    assert Factors(x**sqrt(2)**3).as_expr() == x**(2*sqrt(2))
    assert Factors(x)**Factors(S(2)) == Factors(x**2)
    assert Factors(x).gcd(S.Zero) == Factors(x)
    assert Factors(x).lcm(S.Zero).is_zero
    assert Factors(S.Zero).div(x) == (Factors(S.Zero), Factors())
    assert Factors(x).div(x) == (Factors(), Factors())
    assert Factors({x: .2})/Factors({x: .2}) == Factors()
    assert Factors(x) != Factors()
    assert Factors(S.Zero).normal(x) == (Factors(S.Zero), Factors())
    n, d = x**(2 + y), x**2
    f = Factors(n)
    assert f.div(d) == f.normal(d) == (Factors(x**y), Factors())
    assert f.gcd(d) == Factors()
    d = x**y
    assert f.div(d) == f.normal(d) == (Factors(x**2), Factors())
    assert f.gcd(d) == Factors(d)
    n = d = 2**x
    f = Factors(n)
    assert f.div(d) == f.normal(d) == (Factors(), Factors())
    assert f.gcd(d) == Factors(d)
    n, d = 2**x, 2**y
    f = Factors(n)
    assert f.div(d) == f.normal(d) == (Factors({S(2): x}), Factors({S(2): y}))
    assert f.gcd(d) == Factors()

    # extraction of constant only
    n = x**(x + 3)
    assert Factors(n).normal(x**-3) == (Factors({x: x + 6}), Factors({}))
    assert Factors(n).normal(x**3) == (Factors({x: x}), Factors({}))
    assert Factors(n).normal(x**4) == (Factors({x: x}), Factors({x: 1}))
    assert Factors(n).normal(x**(y - 3)) == \
        (Factors({x: x + 6}), Factors({x: y}))
    assert Factors(n).normal(x**(y + 3)) == (Factors({x: x}), Factors({x: y}))
    assert Factors(n).normal(x**(y + 4)) == \
        (Factors({x: x}), Factors({x: y + 1}))

    assert Factors(n).div(x**-3) == (Factors({x: x + 6}), Factors({}))
    assert Factors(n).div(x**3) == (Factors({x: x}), Factors({}))
    assert Factors(n).div(x**4) == (Factors({x: x}), Factors({x: 1}))
    assert Factors(n).div(x**(y - 3)) == \
        (Factors({x: x + 6}), Factors({x: y}))
    assert Factors(n).div(x**(y + 3)) == (Factors({x: x}), Factors({x: y}))
    assert Factors(n).div(x**(y + 4)) == \
        (Factors({x: x}), Factors({x: y + 1}))

    assert Factors(3 * x / 2) == Factors({3: 1, 2: -1, x: 1})
    assert Factors(x * x / y) == Factors({x: 2, y: -1})
    assert Factors(27 * x / y**9) == Factors({27: 1, x: 1, y: -9})


def test_Term():
    a = Term(4*x*y**2/z/t**3)
    b = Term(2*x**3*y**5/t**3)

    assert a == Term(4, Factors({x: 1, y: 2}), Factors({z: 1, t: 3}))
    assert b == Term(2, Factors({x: 3, y: 5}), Factors({t: 3}))

    assert a.as_expr() == 4*x*y**2/z/t**3
    assert b.as_expr() == 2*x**3*y**5/t**3

    assert a.inv() == \
        Term(S.One/4, Factors({z: 1, t: 3}), Factors({x: 1, y: 2}))
    assert b.inv() == Term(S.Half, Factors({t: 3}), Factors({x: 3, y: 5}))

    assert a.mul(b) == a*b == \
        Term(8, Factors({x: 4, y: 7}), Factors({z: 1, t: 6}))
    assert a.quo(b) == a/b == Term(2, Factors({}), Factors({x: 2, y: 3, z: 1}))

    assert a.pow(3) == a**3 == \
        Term(64, Factors({x: 3, y: 6}), Factors({z: 3, t: 9}))
    assert b.pow(3) == b**3 == Term(8, Factors({x: 9, y: 15}), Factors({t: 9}))

    assert a.pow(-3) == a**(-3) == \
        Term(S.One/64, Factors({z: 3, t: 9}), Factors({x: 3, y: 6}))
    assert b.pow(-3) == b**(-3) == \
        Term(S.One/8, Factors({t: 9}), Factors({x: 9, y: 15}))

    assert a.gcd(b) == Term(2, Factors({x: 1, y: 2}), Factors({t: 3}))
    assert a.lcm(b) == Term(4, Factors({x: 3, y: 5}), Factors({z: 1, t: 3}))

    a = Term(4*x*y**2/z/t**3)
    b = Term(2*x**3*y**5*t**7)

    assert a.mul(b) == Term(8, Factors({x: 4, y: 7, t: 4}), Factors({z: 1}))

    assert Term((2*x + 2)**3) == Term(8, Factors({x + 1: 3}), Factors({}))
    assert Term((2*x + 2)*(3*x + 6)**2) == \
        Term(18, Factors({x + 1: 1, x + 2: 2}), Factors({}))


def test_gcd_terms():
    f = 2*(x + 1)*(x + 4)/(5*x**2 + 5) + (2*x + 2)*(x + 5)/(x**2 + 1)/5 + \
        (2*x + 2)*(x + 6)/(5*x**2 + 5)

    assert _gcd_terms(f) == ((Rational(6, 5))*((1 + x)/(1 + x**2)), 5 + x, 1)
    assert _gcd_terms(Add.make_args(f)) == \
        ((Rational(6, 5))*((1 + x)/(1 + x**2)), 5 + x, 1)

    newf = (Rational(6, 5))*((1 + x)*(5 + x)/(1 + x**2))
    assert gcd_terms(f) == newf
    args = Add.make_args(f)
    # non-Basic sequences of terms treated as terms of Add
    assert gcd_terms(list(args)) == newf
    assert gcd_terms(tuple(args)) == newf
    assert gcd_terms(set(args)) == newf
    # but a Basic sequence is treated as a container
    assert gcd_terms(Tuple(*args)) != newf
    assert gcd_terms(Basic(Tuple(S(1), 3*y + 3*x*y), Tuple(S(1), S(3)))) == \
        Basic(Tuple(S(1), 3*y*(x + 1)), Tuple(S(1), S(3)))
    # but we shouldn't change keys of a dictionary or some may be lost
    assert gcd_terms(Dict((x*(1 + y), S(2)), (x + x*y, y + x*y))) == \
        Dict({x*(y + 1): S(2), x + x*y: y*(1 + x)})

    assert gcd_terms((2*x + 2)**3 + (2*x + 2)**2) == 4*(x + 1)**2*(2*x + 3)

    assert gcd_terms(0) == 0
    assert gcd_terms(1) == 1
    assert gcd_terms(x) == x
    assert gcd_terms(2 + 2*x) == Mul(2, 1 + x, evaluate=False)
    arg = x*(2*x + 4*y)
    garg = 2*x*(x + 2*y)
    assert gcd_terms(arg) == garg
    assert gcd_terms(sin(arg)) == sin(garg)

    # issue 6139-like
    alpha, alpha1, alpha2, alpha3 = symbols('alpha:4')
    a = alpha**2 - alpha*x**2 + alpha + x**3 - x*(alpha + 1)
    rep = (alpha, (1 + sqrt(5))/2 + alpha1*x + alpha2*x**2 + alpha3*x**3)
    s = (a/(x - alpha)).subs(*rep).series(x, 0, 1)
    assert simplify(collect(s, x)) == -sqrt(5)/2 - Rational(3, 2) + O(x)

    # issue 5917
    assert _gcd_terms([S.Zero, S.Zero]) == (0, 0, 1)
    assert _gcd_terms([2*x + 4]) == (2, x + 2, 1)

    eq = x/(x + 1/x)
    assert gcd_terms(eq, fraction=False) == eq
    eq = x/2/y + 1/x/y
    assert gcd_terms(eq, fraction=True, clear=True) == \
        (x**2 + 2)/(2*x*y)
    assert gcd_terms(eq, fraction=True, clear=False) == \
        (x**2/2 + 1)/(x*y)
    assert gcd_terms(eq, fraction=False, clear=True) == \
        (x + 2/x)/(2*y)
    assert gcd_terms(eq, fraction=False, clear=False) == \
        (x/2 + 1/x)/y


def test_factor_terms():
    A = Symbol('A', commutative=False)
    assert factor_terms(9*(x + x*y + 1) + (3*x + 3)**(2 + 2*x)) == \
        9*x*y + 9*x + _keep_coeff(S(3), x + 1)**_keep_coeff(S(2), x + 1) + 9
    assert factor_terms(9*(x + x*y + 1) + (3)**(2 + 2*x)) == \
        _keep_coeff(S(9), 3**(2*x) + x*y + x + 1)
    assert factor_terms(3**(2 + 2*x) + a*3**(2 + 2*x)) == \
        9*3**(2*x)*(a + 1)
    assert factor_terms(x + x*A) == \
        x*(1 + A)
    assert factor_terms(sin(x + x*A)) == \
        sin(x*(1 + A))
    assert factor_terms((3*x + 3)**((2 + 2*x)/3)) == \
        _keep_coeff(S(3), x + 1)**_keep_coeff(Rational(2, 3), x + 1)
    assert factor_terms(x + (x*y + x)**(3*x + 3)) == \
        x + (x*(y + 1))**_keep_coeff(S(3), x + 1)
    assert factor_terms(a*(x + x*y) + b*(x*2 + y*x*2)) == \
        x*(a + 2*b)*(y + 1)
    i = Integral(x, (x, 0, oo))
    assert factor_terms(i) == i

    assert factor_terms(x/2 + y) == x/2 + y
    # fraction doesn't apply to integer denominators
    assert factor_terms(x/2 + y, fraction=True) == x/2 + y
    # clear *does* apply to the integer denominators
    assert factor_terms(x/2 + y, clear=True) == Mul(S.Half, x + 2*y, evaluate=False)

    # check radical extraction
    eq = sqrt(2) + sqrt(10)
    assert factor_terms(eq) == eq
    assert factor_terms(eq, radical=True) == sqrt(2)*(1 + sqrt(5))
    eq = root(-6, 3) + root(6, 3)
    assert factor_terms(eq, radical=True) == 6**(S.One/3)*(1 + (-1)**(S.One/3))

    eq = [x + x*y]
    ans = [x*(y + 1)]
    for c in [list, tuple, set]:
        assert factor_terms(c(eq)) == c(ans)
    assert factor_terms(Tuple(x + x*y)) == Tuple(x*(y + 1))
    assert factor_terms(Interval(0, 1)) == Interval(0, 1)
    e = 1/sqrt(a/2 + 1)
    assert factor_terms(e, clear=False) == 1/sqrt(a/2 + 1)
    assert factor_terms(e, clear=True) == sqrt(2)/sqrt(a + 2)

    eq = x/(x + 1/x) + 1/(x**2 + 1)
    assert factor_terms(eq, fraction=False) == eq
    assert factor_terms(eq, fraction=True) == 1

    assert factor_terms((1/(x**3 + x**2) + 2/x**2)*y) == \
        y*(2 + 1/(x + 1))/x**2

    # if not True, then processesing for this in factor_terms is not necessary
    assert gcd_terms(-x - y) == -x - y
    assert factor_terms(-x - y) == Mul(-1, x + y, evaluate=False)

    # if not True, then "special" processesing in factor_terms is not necessary
    assert gcd_terms(exp(Mul(-1, x + 1))) == exp(-x - 1)
    e = exp(-x - 2) + x
    assert factor_terms(e) == exp(Mul(-1, x + 2, evaluate=False)) + x
    assert factor_terms(e, sign=False) == e
    assert factor_terms(exp(-4*x - 2) - x) == -x + exp(Mul(-2, 2*x + 1, evaluate=False))

    # sum/integral tests
    for F in (Sum, Integral):
        assert factor_terms(F(x, (y, 1, 10))) == x * F(1, (y, 1, 10))
        assert factor_terms(F(x, (y, 1, 10)) + x) == x * (1 + F(1, (y, 1, 10)))
        assert factor_terms(F(x*y + x*y**2, (y, 1, 10))) == x*F(y*(y + 1), (y, 1, 10))

    # expressions involving Pow terms with base 0
    assert factor_terms(0**(x - 2) - 1) == 0**(x - 2) - 1
    assert factor_terms(0**(x + 2) - 1) == 0**(x + 2) - 1
    assert factor_terms((0**(x + 2) - 1).subs(x,-2)) == 0


def test_xreplace():
    e = Mul(2, 1 + x, evaluate=False)
    assert e.xreplace({}) == e
    assert e.xreplace({y: x}) == e


def test_factor_nc():
    x, y = symbols('x,y')
    k = symbols('k', integer=True)
    n, m, o = symbols('n,m,o', commutative=False)

    # mul and multinomial expansion is needed
    from sympy.core.function import _mexpand
    e = x*(1 + y)**2
    assert _mexpand(e) == x + x*2*y + x*y**2

    def factor_nc_test(e):
        ex = _mexpand(e)
        assert ex.is_Add
        f = factor_nc(ex)
        assert not f.is_Add and _mexpand(f) == ex

    factor_nc_test(x*(1 + y))
    factor_nc_test(n*(x + 1))
    factor_nc_test(n*(x + m))
    factor_nc_test((x + m)*n)
    factor_nc_test(n*m*(x*o + n*o*m)*n)
    s = Sum(x, (x, 1, 2))
    factor_nc_test(x*(1 + s))
    factor_nc_test(x*(1 + s)*s)
    factor_nc_test(x*(1 + sin(s)))
    factor_nc_test((1 + n)**2)

    factor_nc_test((x + n)*(x + m)*(x + y))
    factor_nc_test(x*(n*m + 1))
    factor_nc_test(x*(n*m + x))
    factor_nc_test(x*(x*n*m + 1))
    factor_nc_test(n*(m/x + o))
    factor_nc_test(m*(n + o/2))
    factor_nc_test(x*n*(x*m + 1))
    factor_nc_test(x*(m*n + x*n*m))
    factor_nc_test(n*(1 - m)*n**2)

    factor_nc_test((n + m)**2)
    factor_nc_test((n - m)*(n + m)**2)
    factor_nc_test((n + m)**2*(n - m))
    factor_nc_test((m - n)*(n + m)**2*(n - m))

    assert factor_nc(n*(n + n*m)) == n**2*(1 + m)
    assert factor_nc(m*(m*n + n*m*n**2)) == m*(m + n*m*n)*n
    eq = m*sin(n) - sin(n)*m
    assert factor_nc(eq) == eq

    # for coverage:
    from sympy.physics.secondquant import Commutator
    from sympy.polys.polytools import factor
    eq = 1 + x*Commutator(m, n)
    assert factor_nc(eq) == eq
    eq = x*Commutator(m, n) + x*Commutator(m, o)*Commutator(m, n)
    assert factor(eq) == x*(1 + Commutator(m, o))*Commutator(m, n)

    # issue 6534
    assert (2*n + 2*m).factor() == 2*(n + m)

    # issue 6701
    _n = symbols('nz', zero=False, commutative=False)
    assert factor_nc(_n**k + _n**(k + 1)) == _n**k*(1 + _n)
    assert factor_nc((m*n)**k + (m*n)**(k + 1)) == (1 + m*n)*(m*n)**k

    # issue 6918
    assert factor_nc(-n*(2*x**2 + 2*x)) == -2*n*x*(x + 1)


def test_issue_6360():
    a, b = symbols("a b")
    apb = a + b
    eq = apb + apb**2*(-2*a - 2*b)
    assert factor_terms(sub_pre(eq)) == a + b - 2*(a + b)**3


def test_issue_7903():
    a = symbols(r'a', real=True)
    t = exp(I*cos(a)) + exp(-I*sin(a))
    assert t.simplify()

def test_issue_8263():
    F, G = symbols('F, G', commutative=False, cls=Function)
    x, y = symbols('x, y')
    expr, dummies, _ = _mask_nc(F(x)*G(y) - G(y)*F(x))
    for v in dummies.values():
        assert not v.is_commutative
    assert not expr.is_zero

def test_monotonic_sign():
    F = _monotonic_sign
    x = symbols('x')
    assert F(x) is None
    assert F(-x) is None
    assert F(Dummy(prime=True)) == 2
    assert F(Dummy(prime=True, odd=True)) == 3
    assert F(Dummy(composite=True)) == 4
    assert F(Dummy(composite=True, odd=True)) == 9
    assert F(Dummy(positive=True, integer=True)) == 1
    assert F(Dummy(positive=True, even=True)) == 2
    assert F(Dummy(positive=True, even=True, prime=False)) == 4
    assert F(Dummy(negative=True, integer=True)) == -1
    assert F(Dummy(negative=True, even=True)) == -2
    assert F(Dummy(zero=True)) == 0
    assert F(Dummy(nonnegative=True)) == 0
    assert F(Dummy(nonpositive=True)) == 0

    assert F(Dummy(positive=True) + 1).is_positive
    assert F(Dummy(positive=True, integer=True) - 1).is_nonnegative
    assert F(Dummy(positive=True) - 1) is None
    assert F(Dummy(negative=True) + 1) is None
    assert F(Dummy(negative=True, integer=True) - 1).is_nonpositive
    assert F(Dummy(negative=True) - 1).is_negative
    assert F(-Dummy(positive=True) + 1) is None
    assert F(-Dummy(positive=True, integer=True) - 1).is_negative
    assert F(-Dummy(positive=True) - 1).is_negative
    assert F(-Dummy(negative=True) + 1).is_positive
    assert F(-Dummy(negative=True, integer=True) - 1).is_nonnegative
    assert F(-Dummy(negative=True) - 1) is None
    x = Dummy(negative=True)
    assert F(x**3).is_nonpositive
    assert F(x**3 + log(2)*x - 1).is_negative
    x = Dummy(positive=True)
    assert F(-x**3).is_nonpositive

    p = Dummy(positive=True)
    assert F(1/p).is_positive
    assert F(p/(p + 1)).is_positive
    p = Dummy(nonnegative=True)
    assert F(p/(p + 1)).is_nonnegative
    p = Dummy(positive=True)
    assert F(-1/p).is_negative
    p = Dummy(nonpositive=True)
    assert F(p/(-p + 1)).is_nonpositive

    p = Dummy(positive=True, integer=True)
    q = Dummy(positive=True, integer=True)
    assert F(-2/p/q).is_negative
    assert F(-2/(p - 1)/q) is None

    assert F((p - 1)*q + 1).is_positive
    assert F(-(p - 1)*q - 1).is_negative

def test_issue_17256():
    from sympy.sets.fancysets import Range
    x = Symbol('x')
    s1 = Sum(x + 1, (x, 1, 9))
    s2 = Sum(x + 1, (x, Range(1, 10)))
    a = Symbol('a')
    r1 = s1.xreplace({x:a})
    r2 = s2.xreplace({x:a})

    assert r1.doit() == r2.doit()
    s1 = Sum(x + 1, (x, 0, 9))
    s2 = Sum(x + 1, (x, Range(10)))
    a = Symbol('a')
    r1 = s1.xreplace({x:a})
    r2 = s2.xreplace({x:a})
    assert r1 == r2

def test_issue_21623():
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    M = MatrixSymbol('X', 2, 2)
    assert gcd_terms(M[0,0], 1) == M[0,0]
