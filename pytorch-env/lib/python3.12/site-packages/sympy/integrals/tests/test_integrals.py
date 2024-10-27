import math
from sympy.concrete.summations import (Sum, summation)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, Lambda, diff)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, I, Rational, nan, oo, pi, zoo, all_close)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, im, polar_lift, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (acosh, asinh, cosh, coth, csch, sinh, tanh, sech)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan, sec)
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import (Ci, Ei, Si, erf, erfc, erfi, fresnelc, li)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import lerchphi
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import (Idx, IndexedBase)
from sympy.core.expr import unchanged
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import Integral
from sympy.integrals.risch import NonElementaryIntegral
from sympy.physics import units
from sympy.testing.pytest import raises, slow, warns_deprecated_sympy, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically


x, y, z, a, b, c, d, e, s, t, x_1, x_2 = symbols('x y z a b c d e s t x_1 x_2')
n = Symbol('n', integer=True)
f = Function('f')


def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)


def test_poly_deprecated():
    p = Poly(2*x, x)
    assert p.integrate(x) == Poly(x**2, x, domain='QQ')
    # The stacklevel is based on Integral(Poly)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        integrate(p, x)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        Integral(p, (x,))


@slow
def test_principal_value():
    g = 1 / x
    assert Integral(g, (x, -oo, oo)).principal_value() == 0
    assert Integral(g, (y, -oo, oo)).principal_value() == oo * sign(1 / x)
    raises(ValueError, lambda: Integral(g, (x)).principal_value())
    raises(ValueError, lambda: Integral(g).principal_value())

    l = 1 / ((x ** 3) - 1)
    assert Integral(l, (x, -oo, oo)).principal_value().together() == -sqrt(3)*pi/3
    raises(ValueError, lambda: Integral(l, (x, -oo, 1)).principal_value())

    d = 1 / (x ** 2 - 1)
    assert Integral(d, (x, -oo, oo)).principal_value() == 0
    assert Integral(d, (x, -2, 2)).principal_value() == -log(3)

    v = x / (x ** 2 - 1)
    assert Integral(v, (x, -oo, oo)).principal_value() == 0
    assert Integral(v, (x, -2, 2)).principal_value() == 0

    s = x ** 2 / (x ** 2 - 1)
    assert Integral(s, (x, -oo, oo)).principal_value() is oo
    assert Integral(s, (x, -2, 2)).principal_value() == -log(3) + 4

    f = 1 / ((x ** 2 - 1) * (1 + x ** 2))
    assert Integral(f, (x, -oo, oo)).principal_value() == -pi / 2
    assert Integral(f, (x, -2, 2)).principal_value() == -atan(2) - log(3) / 2


def diff_test(i):
    """Return the set of symbols, s, which were used in testing that
    i.diff(s) agrees with i.doit().diff(s). If there is an error then
    the assertion will fail, causing the test to fail."""
    syms = i.free_symbols
    for s in syms:
        assert (i.diff(s).doit() - i.doit().diff(s)).expand() == 0
    return syms


def test_improper_integral():
    assert integrate(log(x), (x, 0, 1)) == -1
    assert integrate(x**(-2), (x, 1, oo)) == 1
    assert integrate(1/(1 + exp(x)), (x, 0, oo)) == log(2)


def test_constructor():
    # this is shared by Sum, so testing Integral's constructor
    # is equivalent to testing Sum's
    s1 = Integral(n, n)
    assert s1.limits == (Tuple(n),)
    s2 = Integral(n, (n,))
    assert s2.limits == (Tuple(n),)
    s3 = Integral(Sum(x, (x, 1, y)))
    assert s3.limits == (Tuple(y),)
    s4 = Integral(n, Tuple(n,))
    assert s4.limits == (Tuple(n),)

    s5 = Integral(n, (n, Interval(1, 2)))
    assert s5.limits == (Tuple(n, 1, 2),)

    # Testing constructor with inequalities:
    s6 = Integral(n, n > 10)
    assert s6.limits == (Tuple(n, 10, oo),)
    s7 = Integral(n, (n > 2) & (n < 5))
    assert s7.limits == (Tuple(n, 2, 5),)


def test_basics():

    assert Integral(0, x) != 0
    assert Integral(x, (x, 1, 1)) != 0
    assert Integral(oo, x) != oo
    assert Integral(S.NaN, x) is S.NaN

    assert diff(Integral(y, y), x) == 0
    assert diff(Integral(x, (x, 0, 1)), x) == 0
    assert diff(Integral(x, x), x) == x
    assert diff(Integral(t, (t, 0, x)), x) == x

    e = (t + 1)**2
    assert diff(integrate(e, (t, 0, x)), x) == \
        diff(Integral(e, (t, 0, x)), x).doit().expand() == \
        ((1 + x)**2).expand()
    assert diff(integrate(e, (t, 0, x)), t) == \
        diff(Integral(e, (t, 0, x)), t) == 0
    assert diff(integrate(e, (t, 0, x)), a) == \
        diff(Integral(e, (t, 0, x)), a) == 0
    assert diff(integrate(e, t), a) == diff(Integral(e, t), a) == 0

    assert integrate(e, (t, a, x)).diff(x) == \
        Integral(e, (t, a, x)).diff(x).doit().expand()
    assert Integral(e, (t, a, x)).diff(x).doit() == ((1 + x)**2)
    assert integrate(e, (t, x, a)).diff(x).doit() == (-(1 + x)**2).expand()

    assert integrate(t**2, (t, x, 2*x)).diff(x) == 7*x**2

    assert Integral(x, x).atoms() == {x}
    assert Integral(f(x), (x, 0, 1)).atoms() == {S.Zero, S.One, x}

    assert diff_test(Integral(x, (x, 3*y))) == {y}
    assert diff_test(Integral(x, (a, 3*y))) == {x, y}

    assert integrate(x, (x, oo, oo)) == 0 #issue 8171
    assert integrate(x, (x, -oo, -oo)) == 0

    # sum integral of terms
    assert integrate(y + x + exp(x), x) == x*y + x**2/2 + exp(x)

    assert Integral(x).is_commutative
    n = Symbol('n', commutative=False)
    assert Integral(n + x, x).is_commutative is False


def test_diff_wrt():
    class Test(Expr):
        _diff_wrt = True
        is_commutative = True

    t = Test()
    assert integrate(t + 1, t) == t**2/2 + t
    assert integrate(t + 1, (t, 0, 1)) == Rational(3, 2)

    raises(ValueError, lambda: integrate(x + 1, x + 1))
    raises(ValueError, lambda: integrate(x + 1, (x + 1, 0, 1)))


def test_basics_multiple():
    assert diff_test(Integral(x, (x, 3*x, 5*y), (y, x, 2*x))) == {x}
    assert diff_test(Integral(x, (x, 5*y), (y, x, 2*x))) == {x}
    assert diff_test(Integral(x, (x, 5*y), (y, y, 2*x))) == {x, y}
    assert diff_test(Integral(y, y, x)) == {x, y}
    assert diff_test(Integral(y*x, x, y)) == {x, y}
    assert diff_test(Integral(x + y, y, (y, 1, x))) == {x}
    assert diff_test(Integral(x + y, (x, x, y), (y, y, x))) == {x, y}


def test_conjugate_transpose():
    A, B = symbols("A B", commutative=False)

    x = Symbol("x", complex=True)
    p = Integral(A*B, (x,))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()

    x = Symbol("x", real=True)
    p = Integral(A*B, (x,))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()


def test_integration():
    assert integrate(0, (t, 0, x)) == 0
    assert integrate(3, (t, 0, x)) == 3*x
    assert integrate(t, (t, 0, x)) == x**2/2
    assert integrate(3*t, (t, 0, x)) == 3*x**2/2
    assert integrate(3*t**2, (t, 0, x)) == x**3
    assert integrate(1/t, (t, 1, x)) == log(x)
    assert integrate(-1/t**2, (t, 1, x)) == 1/x - 1
    assert integrate(t**2 + 5*t - 8, (t, 0, x)) == x**3/3 + 5*x**2/2 - 8*x
    assert integrate(x**2, x) == x**3/3
    assert integrate((3*t*x)**5, x) == (3*t)**5 * x**6 / 6

    b = Symbol("b")
    c = Symbol("c")
    assert integrate(a*t, (t, 0, x)) == a*x**2/2
    assert integrate(a*t**4, (t, 0, x)) == a*x**5/5
    assert integrate(a*t**2 + b*t + c, (t, 0, x)) == a*x**3/3 + b*x**2/2 + c*x


def test_multiple_integration():
    assert integrate((x**2)*(y**2), (x, 0, 1), (y, -1, 2)) == Rational(1)
    assert integrate((y**2)*(x**2), x, y) == Rational(1, 9)*(x**3)*(y**3)
    assert integrate(1/(x + 3)/(1 + x)**3, x) == \
        log(3 + x)*Rational(-1, 8) + log(1 + x)*Rational(1, 8) + x/(4 + 8*x + 4*x**2)
    assert integrate(sin(x*y)*y, (x, 0, 1), (y, 0, 1)) == -sin(1) + 1


def test_issue_3532():
    assert integrate(exp(-x), (x, 0, oo)) == 1


def test_issue_3560():
    assert integrate(sqrt(x)**3, x) == 2*sqrt(x)**5/5
    assert integrate(sqrt(x), x) == 2*sqrt(x)**3/3
    assert integrate(1/sqrt(x)**3, x) == -2/sqrt(x)


def test_issue_18038():
    raises(AttributeError, lambda: integrate((x, x)))


def test_integrate_poly():
    p = Poly(x + x**2*y + y**3, x, y)

    # The stacklevel is based on Integral(Poly)
    with warns_deprecated_sympy():
        qx = Integral(p, x)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        qx = integrate(p, x)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        qy = integrate(p, y)

    assert isinstance(qx, Poly) is True
    assert isinstance(qy, Poly) is True

    assert qx.gens == (x, y)
    assert qy.gens == (x, y)

    assert qx.as_expr() == x**2/2 + x**3*y/3 + x*y**3
    assert qy.as_expr() == x*y + x**2*y**2/2 + y**4/4


def test_integrate_poly_definite():
    p = Poly(x + x**2*y + y**3, x, y)

    with warns_deprecated_sympy():
        Qx = Integral(p, (x, 0, 1))
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        Qx = integrate(p, (x, 0, 1))
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        Qy = integrate(p, (y, 0, pi))

    assert isinstance(Qx, Poly) is True
    assert isinstance(Qy, Poly) is True

    assert Qx.gens == (y,)
    assert Qy.gens == (x,)

    assert Qx.as_expr() == S.Half + y/3 + y**3
    assert Qy.as_expr() == pi**4/4 + pi*x + pi**2*x**2/2


def test_integrate_omit_var():
    y = Symbol('y')

    assert integrate(x) == x**2/2

    raises(ValueError, lambda: integrate(2))
    raises(ValueError, lambda: integrate(x*y))


def test_integrate_poly_accurately():
    y = Symbol('y')
    assert integrate(x*sin(y), x) == x**2*sin(y)/2

    # when passed to risch_norman, this will be a CPU hog, so this really
    # checks, that integrated function is recognized as polynomial
    assert integrate(x**1000*sin(y), x) == x**1001*sin(y)/1001


def test_issue_3635():
    y = Symbol('y')
    assert integrate(x**2, y) == x**2*y
    assert integrate(x**2, (y, -1, 1)) == 2*x**2

# works in SymPy and py.test but hangs in `setup.py test`


def test_integrate_linearterm_pow():
    # check integrate((a*x+b)^c, x)  --  issue 3499
    y = Symbol('y', positive=True)
    # TODO: Remove conds='none' below, let the assumption take care of it.
    assert integrate(x**y, x, conds='none') == x**(y + 1)/(y + 1)
    assert integrate((exp(y)*x + 1/y)**(1 + sin(y)), x, conds='none') == \
        exp(-y)*(exp(y)*x + 1/y)**(2 + sin(y)) / (2 + sin(y))


def test_issue_3618():
    assert integrate(pi*sqrt(x), x) == 2*pi*sqrt(x)**3/3
    assert integrate(pi*sqrt(x) + E*sqrt(x)**3, x) == \
        2*pi*sqrt(x)**3/3 + 2*E *sqrt(x)**5/5


def test_issue_3623():
    assert integrate(cos((n + 1)*x), x) == Piecewise(
        (sin(x*(n + 1))/(n + 1), Ne(n + 1, 0)), (x, True))
    assert integrate(cos((n - 1)*x), x) == Piecewise(
        (sin(x*(n - 1))/(n - 1), Ne(n - 1, 0)), (x, True))
    assert integrate(cos((n + 1)*x) + cos((n - 1)*x), x) == \
        Piecewise((sin(x*(n - 1))/(n - 1), Ne(n - 1, 0)), (x, True)) + \
        Piecewise((sin(x*(n + 1))/(n + 1), Ne(n + 1, 0)), (x, True))


def test_issue_3664():
    n = Symbol('n', integer=True, nonzero=True)
    assert integrate(-1./2 * x * sin(n * pi * x/2), [x, -2, 0]) == \
        2.0*cos(pi*n)/(pi*n)
    assert integrate(x * sin(n * pi * x/2) * Rational(-1, 2), [x, -2, 0]) == \
        2*cos(pi*n)/(pi*n)


def test_issue_3679():
    # definite integration of rational functions gives wrong answers
    assert NS(Integral(1/(x**2 - 8*x + 17), (x, 2, 4))) == '1.10714871779409'


def test_issue_3686():  # remove this when fresnel integrals are implemented
    from sympy.core.function import expand_func
    from sympy.functions.special.error_functions import fresnels
    assert expand_func(integrate(sin(x**2), x)) == \
        sqrt(2)*sqrt(pi)*fresnels(sqrt(2)*x/sqrt(pi))/2


def test_integrate_units():
    m = units.m
    s = units.s
    assert integrate(x * m/s, (x, 1*s, 5*s)) == 12*m*s


def test_transcendental_functions():
    assert integrate(LambertW(2*x), x) == \
        -x + x*LambertW(2*x) + x/LambertW(2*x)


def test_log_polylog():
    assert integrate(log(1 - x)/x, (x, 0, 1)) == -pi**2/6
    assert integrate(log(x)*(1 - x)**(-1), (x, 0, 1)) == -pi**2/6


def test_issue_3740():
    f = 4*log(x) - 2*log(x)**2
    fid = diff(integrate(f, x), x)
    assert abs(f.subs(x, 42).evalf() - fid.subs(x, 42).evalf()) < 1e-10


def test_issue_3788():
    assert integrate(1/(1 + x**2), x) == atan(x)


def test_issue_3952():
    f = sin(x)
    assert integrate(f, x) == -cos(x)
    raises(ValueError, lambda: integrate(f, 2*x))


def test_issue_4516():
    assert integrate(2**x - 2*x, x) == 2**x/log(2) - x**2


def test_issue_7450():
    ans = integrate(exp(-(1 + I)*x), (x, 0, oo))
    assert re(ans) == S.Half and im(ans) == Rational(-1, 2)


def test_issue_8623():
    assert integrate((1 + cos(2*x)) / (3 - 2*cos(2*x)), (x, 0, pi)) == -pi/2 + sqrt(5)*pi/2
    assert integrate((1 + cos(2*x))/(3 - 2*cos(2*x))) == -x/2 + sqrt(5)*(atan(sqrt(5)*tan(x)) + \
        pi*floor((x - pi/2)/pi))/2


def test_issue_9569():
    assert integrate(1 / (2 - cos(x)), (x, 0, pi)) == pi/sqrt(3)
    assert integrate(1/(2 - cos(x))) == 2*sqrt(3)*(atan(sqrt(3)*tan(x/2)) + pi*floor((x/2 - pi/2)/pi))/3


def test_issue_13733():
    s = Symbol('s', positive=True)
    pz = exp(-(z - y)**2/(2*s*s))/sqrt(2*pi*s*s)
    pzgx = integrate(pz, (z, x, oo))
    assert integrate(pzgx, (x, 0, oo)) == sqrt(2)*s*exp(-y**2/(2*s**2))/(2*sqrt(pi)) + \
        y*erf(sqrt(2)*y/(2*s))/2 + y/2


def test_issue_13749():
    assert integrate(1 / (2 + cos(x)), (x, 0, pi)) == pi/sqrt(3)
    assert integrate(1/(2 + cos(x))) == 2*sqrt(3)*(atan(sqrt(3)*tan(x/2)/3) + pi*floor((x/2 - pi/2)/pi))/3


def test_issue_18133():
    assert integrate(exp(x)/(1 + x)**2, x) == NonElementaryIntegral(exp(x)/(x + 1)**2, x)


def test_issue_21741():
    a = 4e6
    b = 2.5e-7
    r = Piecewise((b*I*exp(-a*I*pi*t*y)*exp(-a*I*pi*x*z)/(pi*x), Ne(x, 0)),
                  (z*exp(-a*I*pi*t*y), True))
    fun = E**((-2*I*pi*(z*x+t*y))/(500*10**(-9)))
    assert all_close(integrate(fun, z), r)


def test_matrices():
    M = Matrix(2, 2, lambda i, j: (i + j + 1)*sin((i + j + 1)*x))

    assert integrate(M, x) == Matrix([
        [-cos(x), -cos(2*x)],
        [-cos(2*x), -cos(3*x)],
    ])


def test_integrate_functions():
    # issue 4111
    assert integrate(f(x), x) == Integral(f(x), x)
    assert integrate(f(x), (x, 0, 1)) == Integral(f(x), (x, 0, 1))
    assert integrate(f(x)*diff(f(x), x), x) == f(x)**2/2
    assert integrate(diff(f(x), x) / f(x), x) == log(f(x))


def test_integrate_derivatives():
    assert integrate(Derivative(f(x), x), x) == f(x)
    assert integrate(Derivative(f(y), y), x) == x*Derivative(f(y), y)
    assert integrate(Derivative(f(x), x)**2, x) == \
        Integral(Derivative(f(x), x)**2, x)


def test_transform():
    a = Integral(x**2 + 1, (x, -1, 2))
    fx = x
    fy = 3*y + 1
    assert a.doit() == a.transform(fx, fy).doit()
    assert a.transform(fx, fy).transform(fy, fx) == a
    fx = 3*x + 1
    fy = y
    assert a.transform(fx, fy).transform(fy, fx) == a
    a = Integral(sin(1/x), (x, 0, 1))
    assert a.transform(x, 1/y) == Integral(sin(y)/y**2, (y, 1, oo))
    assert a.transform(x, 1/y).transform(y, 1/x) == a
    a = Integral(exp(-x**2), (x, -oo, oo))
    assert a.transform(x, 2*y) == Integral(2*exp(-4*y**2), (y, -oo, oo))
    # < 3 arg limit handled properly
    assert Integral(x, x).transform(x, a*y).doit() == \
        Integral(y*a**2, y).doit()
    _3 = S(3)
    assert Integral(x, (x, 0, -_3)).transform(x, 1/y).doit() == \
        Integral(-1/x**3, (x, -oo, -1/_3)).doit()
    assert Integral(x, (x, 0, _3)).transform(x, 1/y) == \
        Integral(y**(-3), (y, 1/_3, oo))
    # issue 8400
    i = Integral(x + y, (x, 1, 2), (y, 1, 2))
    assert i.transform(x, (x + 2*y, x)).doit() == \
        i.transform(x, (x + 2*z, x)).doit() == 3

    i = Integral(x, (x, a, b))
    assert i.transform(x, 2*s) == Integral(4*s, (s, a/2, b/2))
    raises(ValueError, lambda: i.transform(x, 1))
    raises(ValueError, lambda: i.transform(x, s*t))
    raises(ValueError, lambda: i.transform(x, -s))
    raises(ValueError, lambda: i.transform(x, (s, t)))
    raises(ValueError, lambda: i.transform(2*x, 2*s))

    i = Integral(x**2, (x, 1, 2))
    raises(ValueError, lambda: i.transform(x**2, s))

    am = Symbol('a', negative=True)
    bp = Symbol('b', positive=True)
    i = Integral(x, (x, bp, am))
    i.transform(x, 2*s)
    assert i.transform(x, 2*s) == Integral(-4*s, (s, am/2, bp/2))

    i = Integral(x, (x, a))
    assert i.transform(x, 2*s) == Integral(4*s, (s, a/2))


def test_issue_4052():
    f = S.Half*asin(x) + x*sqrt(1 - x**2)/2

    assert integrate(cos(asin(x)), x) == f
    assert integrate(sin(acos(x)), x) == f


@slow
def test_evalf_integrals():
    assert NS(Integral(x, (x, 2, 5)), 15) == '10.5000000000000'
    gauss = Integral(exp(-x**2), (x, -oo, oo))
    assert NS(gauss, 15) == '1.77245385090552'
    assert NS(gauss**2 - pi + E*Rational(
        1, 10**20), 15) in ('2.71828182845904e-20', '2.71828182845905e-20')
    # A monster of an integral from http://mathworld.wolfram.com/DefiniteIntegral.html
    t = Symbol('t')
    a = 8*sqrt(3)/(1 + 3*t**2)
    b = 16*sqrt(2)*(3*t + 1)*sqrt(4*t**2 + t + 1)**3
    c = (3*t**2 + 1)*(11*t**2 + 2*t + 3)**2
    d = sqrt(2)*(249*t**2 + 54*t + 65)/(11*t**2 + 2*t + 3)**2
    f = a - b/c - d
    assert NS(Integral(f, (t, 0, 1)), 50) == \
        NS((3*sqrt(2) - 49*pi + 162*atan(sqrt(2)))/12, 50)
    # http://mathworld.wolfram.com/VardisIntegral.html
    assert NS(Integral(log(log(1/x))/(1 + x + x**2), (x, 0, 1)), 15) == \
        NS('pi/sqrt(3) * log(2*pi**(5/6) / gamma(1/6))', 15)
    # http://mathworld.wolfram.com/AhmedsIntegral.html
    assert NS(Integral(atan(sqrt(x**2 + 2))/(sqrt(x**2 + 2)*(x**2 + 1)), (x,
              0, 1)), 15) == NS(5*pi**2/96, 15)
    # http://mathworld.wolfram.com/AbelsIntegral.html
    assert NS(Integral(x/((exp(pi*x) - exp(
        -pi*x))*(x**2 + 1)), (x, 0, oo)), 15) == NS('log(2)/2-1/4', 15)
    # Complex part trimming
    # http://mathworld.wolfram.com/VardisIntegral.html
    assert NS(Integral(log(log(sin(x)/cos(x))), (x, pi/4, pi/2)), 15, chop=True) == \
        NS('pi/4*log(4*pi**3/gamma(1/4)**4)', 15)
    #
    # Endpoints causing trouble (rounding error in integration points -> complex log)
    assert NS(
        2 + Integral(log(2*cos(x/2)), (x, -pi, pi)), 17, chop=True) == NS(2, 17)
    assert NS(
        2 + Integral(log(2*cos(x/2)), (x, -pi, pi)), 20, chop=True) == NS(2, 20)
    assert NS(
        2 + Integral(log(2*cos(x/2)), (x, -pi, pi)), 22, chop=True) == NS(2, 22)
    # Needs zero handling
    assert NS(pi - 4*Integral(
        'sqrt(1-x**2)', (x, 0, 1)), 15, maxn=30, chop=True) in ('0.0', '0')
    # Oscillatory quadrature
    a = Integral(sin(x)/x**2, (x, 1, oo)).evalf(maxn=15)
    assert 0.49 < a < 0.51
    assert NS(
        Integral(sin(x)/x**2, (x, 1, oo)), quad='osc') == '0.504067061906928'
    assert NS(Integral(
        cos(pi*x + 1)/x, (x, -oo, -1)), quad='osc') == '0.276374705640365'
    # indefinite integrals aren't evaluated
    assert NS(Integral(x, x)) == 'Integral(x, x)'
    assert NS(Integral(x, (x, y))) == 'Integral(x, (x, y))'


def test_evalf_issue_939():
    # https://github.com/sympy/sympy/issues/4038

    # The output form of an integral may differ by a step function between
    # revisions, making this test a bit useless. This can't be said about
    # other two tests. For now, all values of this evaluation are used here,
    # but in future this should be reconsidered.
    assert NS(integrate(1/(x**5 + 1), x).subs(x, 4), chop=True) in \
        ['-0.000976138910649103', '0.965906660135753', '1.93278945918216']

    assert NS(Integral(1/(x**5 + 1), (x, 2, 4))) == '0.0144361088886740'
    assert NS(
        integrate(1/(x**5 + 1), (x, 2, 4)), chop=True) == '0.0144361088886740'


def test_double_previously_failing_integrals():
    # Double integrals not implemented <- Sure it is!
    res = integrate(sqrt(x) + x*y, (x, 1, 2), (y, -1, 1))
    # Old numerical test
    assert NS(res, 15) == '2.43790283299492'
    # Symbolic test
    assert res == Rational(-4, 3) + 8*sqrt(2)/3
    # double integral + zero detection
    assert integrate(sin(x + x*y), (x, -1, 1), (y, -1, 1)) is S.Zero


def test_integrate_SingularityFunction():
    in_1 = SingularityFunction(x, a, 3) + SingularityFunction(x, 5, -1)
    out_1 = SingularityFunction(x, a, 4)/4 + SingularityFunction(x, 5, 0)
    assert integrate(in_1, x) == out_1

    in_2 = 10*SingularityFunction(x, 4, 0) - 5*SingularityFunction(x, -6, -2)
    out_2 = 10*SingularityFunction(x, 4, 1) - 5*SingularityFunction(x, -6, -1)
    assert integrate(in_2, x) == out_2

    in_3 = 2*x**2*y -10*SingularityFunction(x, -4, 7) - 2*SingularityFunction(y, 10, -2)
    out_3_1 = 2*x**3*y/3 - 2*x*SingularityFunction(y, 10, -2) - 5*SingularityFunction(x, -4, 8)/4
    out_3_2 = x**2*y**2 - 10*y*SingularityFunction(x, -4, 7) - 2*SingularityFunction(y, 10, -1)
    assert integrate(in_3, x) == out_3_1
    assert integrate(in_3, y) == out_3_2

    assert unchanged(Integral, in_3, (x,))
    assert Integral(in_3, x) == Integral(in_3, (x,))
    assert Integral(in_3, x).doit() == out_3_1

    in_4 = 10*SingularityFunction(x, -4, 7) - 2*SingularityFunction(x, 10, -2)
    out_4 = 5*SingularityFunction(x, -4, 8)/4 - 2*SingularityFunction(x, 10, -1)
    assert integrate(in_4, (x, -oo, x)) == out_4

    assert integrate(SingularityFunction(x, 5, -1), x) == SingularityFunction(x, 5, 0)
    assert integrate(SingularityFunction(x, 0, -1), (x, -oo, oo)) == 1
    assert integrate(5*SingularityFunction(x, 5, -1), (x, -oo, oo)) == 5
    assert integrate(SingularityFunction(x, 5, -1) * f(x), (x, -oo, oo)) == f(5)


def test_integrate_DiracDelta():
    # This is here to check that deltaintegrate is being called, but also
    # to test definite integrals. More tests are in test_deltafunctions.py
    assert integrate(DiracDelta(x) * f(x), (x, -oo, oo)) == f(0)
    assert integrate(DiracDelta(x)**2, (x, -oo, oo)) == DiracDelta(0)
    # issue 4522
    assert integrate(integrate((4 - 4*x + x*y - 4*y) * \
        DiracDelta(x)*DiracDelta(y - 1), (x, 0, 1)), (y, 0, 1)) == 0
    # issue 5729
    p = exp(-(x**2 + y**2))/pi
    assert integrate(p*DiracDelta(x - 10*y), (x, -oo, oo), (y, -oo, oo)) == \
        integrate(p*DiracDelta(x - 10*y), (y, -oo, oo), (x, -oo, oo)) == \
        integrate(p*DiracDelta(10*x - y), (x, -oo, oo), (y, -oo, oo)) == \
        integrate(p*DiracDelta(10*x - y), (y, -oo, oo), (x, -oo, oo)) == \
        1/sqrt(101*pi)


def test_integrate_returns_piecewise():
    assert integrate(x**y, x) == Piecewise(
        (x**(y + 1)/(y + 1), Ne(y, -1)), (log(x), True))
    assert integrate(x**y, y) == Piecewise(
        (x**y/log(x), Ne(log(x), 0)), (y, True))
    assert integrate(exp(n*x), x) == Piecewise(
        (exp(n*x)/n, Ne(n, 0)), (x, True))
    assert integrate(x*exp(n*x), x) == Piecewise(
        ((n*x - 1)*exp(n*x)/n**2, Ne(n**2, 0)), (x**2/2, True))
    assert integrate(x**(n*y), x) == Piecewise(
        (x**(n*y + 1)/(n*y + 1), Ne(n*y, -1)), (log(x), True))
    assert integrate(x**(n*y), y) == Piecewise(
        (x**(n*y)/(n*log(x)), Ne(n*log(x), 0)), (y, True))
    assert integrate(cos(n*x), x) == Piecewise(
        (sin(n*x)/n, Ne(n, 0)), (x, True))
    assert integrate(cos(n*x)**2, x) == Piecewise(
        ((n*x/2 + sin(n*x)*cos(n*x)/2)/n, Ne(n, 0)), (x, True))
    assert integrate(x*cos(n*x), x) == Piecewise(
        (x*sin(n*x)/n + cos(n*x)/n**2, Ne(n, 0)), (x**2/2, True))
    assert integrate(sin(n*x), x) == Piecewise(
        (-cos(n*x)/n, Ne(n, 0)), (0, True))
    assert integrate(sin(n*x)**2, x) == Piecewise(
        ((n*x/2 - sin(n*x)*cos(n*x)/2)/n, Ne(n, 0)), (0, True))
    assert integrate(x*sin(n*x), x) == Piecewise(
        (-x*cos(n*x)/n + sin(n*x)/n**2, Ne(n, 0)), (0, True))
    assert integrate(exp(x*y), (x, 0, z)) == Piecewise(
        (exp(y*z)/y - 1/y, (y > -oo) & (y < oo) & Ne(y, 0)), (z, True))
    # https://github.com/sympy/sympy/issues/23707
    assert integrate(exp(t)*exp(-t*sqrt(x - y)), t) == Piecewise(
        (-exp(t)/(sqrt(x - y)*exp(t*sqrt(x - y)) - exp(t*sqrt(x - y))),
        Ne(x, y + 1)), (t, True))


def test_integrate_max_min():
    x = symbols('x', real=True)
    assert integrate(Min(x, 2), (x, 0, 3)) == 4
    assert integrate(Max(x**2, x**3), (x, 0, 2)) == Rational(49, 12)
    assert integrate(Min(exp(x), exp(-x))**2, x) == Piecewise( \
        (exp(2*x)/2, x <= 0), (1 - exp(-2*x)/2, True))
    # issue 7907
    c = symbols('c', extended_real=True)
    int1 = integrate(Max(c, x)*exp(-x**2), (x, -oo, oo))
    int2 = integrate(c*exp(-x**2), (x, -oo, c))
    int3 = integrate(x*exp(-x**2), (x, c, oo))
    assert int1 == int2 + int3 == sqrt(pi)*c*erf(c)/2 + \
        sqrt(pi)*c/2 + exp(-c**2)/2


def test_integrate_Abs_sign():
    assert integrate(Abs(x), (x, -2, 1)) == Rational(5, 2)
    assert integrate(Abs(x), (x, 0, 1)) == S.Half
    assert integrate(Abs(x + 1), (x, 0, 1)) == Rational(3, 2)
    assert integrate(Abs(x**2 - 1), (x, -2, 2)) == 4
    assert integrate(Abs(x**2 - 3*x), (x, -15, 15)) == 2259
    assert integrate(sign(x), (x, -1, 2)) == 1
    assert integrate(sign(x)*sin(x), (x, -pi, pi)) == 4
    assert integrate(sign(x - 2) * x**2, (x, 0, 3)) == Rational(11, 3)

    t, s = symbols('t s', real=True)
    assert integrate(Abs(t), t) == Piecewise(
        (-t**2/2, t <= 0), (t**2/2, True))
    assert integrate(Abs(2*t - 6), t) == Piecewise(
        (-t**2 + 6*t, t <= 3), (t**2 - 6*t + 18, True))
    assert (integrate(abs(t - s**2), (t, 0, 2)) ==
        2*s**2*Min(2, s**2) - 2*s**2 - Min(2, s**2)**2 + 2)
    assert integrate(exp(-Abs(t)), t) == Piecewise(
        (exp(t), t <= 0), (2 - exp(-t), True))
    assert integrate(sign(2*t - 6), t) == Piecewise(
        (-t, t < 3), (t - 6, True))
    assert integrate(2*t*sign(t**2 - 1), t) == Piecewise(
        (t**2, t < -1), (-t**2 + 2, t < 1), (t**2, True))
    assert integrate(sign(t), (t, s + 1)) == Piecewise(
        (s + 1, s + 1 > 0), (-s - 1, s + 1 < 0), (0, True))


def test_subs1():
    e = Integral(exp(x - y), x)
    assert e.subs(y, 3) == Integral(exp(x - 3), x)
    e = Integral(exp(x - y), (x, 0, 1))
    assert e.subs(y, 3) == Integral(exp(x - 3), (x, 0, 1))
    f = Lambda(x, exp(-x**2))
    conv = Integral(f(x - y)*f(y), (y, -oo, oo))
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo))


def test_subs2():
    e = Integral(exp(x - y), x, t)
    assert e.subs(y, 3) == Integral(exp(x - 3), x, t)
    e = Integral(exp(x - y), (x, 0, 1), (t, 0, 1))
    assert e.subs(y, 3) == Integral(exp(x - 3), (x, 0, 1), (t, 0, 1))
    f = Lambda(x, exp(-x**2))
    conv = Integral(f(x - y)*f(y), (y, -oo, oo), (t, 0, 1))
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo), (t, 0, 1))


def test_subs3():
    e = Integral(exp(x - y), (x, 0, y), (t, y, 1))
    assert e.subs(y, 3) == Integral(exp(x - 3), (x, 0, 3), (t, 3, 1))
    f = Lambda(x, exp(-x**2))
    conv = Integral(f(x - y)*f(y), (y, -oo, oo), (t, x, 1))
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo), (t, 0, 1))


def test_subs4():
    e = Integral(exp(x), (x, 0, y), (t, y, 1))
    assert e.subs(y, 3) == Integral(exp(x), (x, 0, 3), (t, 3, 1))
    f = Lambda(x, exp(-x**2))
    conv = Integral(f(y)*f(y), (y, -oo, oo), (t, x, 1))
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo), (t, 0, 1))


def test_subs5():
    e = Integral(exp(-x**2), (x, -oo, oo))
    assert e.subs(x, 5) == e
    e = Integral(exp(-x**2 + y), x)
    assert e.subs(y, 5) == Integral(exp(-x**2 + 5), x)
    e = Integral(exp(-x**2 + y), (x, x))
    assert e.subs(x, 5) == Integral(exp(y - x**2), (x, 5))
    assert e.subs(y, 5) == Integral(exp(-x**2 + 5), x)
    e = Integral(exp(-x**2 + y), (y, -oo, oo), (x, -oo, oo))
    assert e.subs(x, 5) == e
    assert e.subs(y, 5) == e
    # Test evaluation of antiderivatives
    e = Integral(exp(-x**2), (x, x))
    assert e.subs(x, 5) == Integral(exp(-x**2), (x, 5))
    e = Integral(exp(x), x)
    assert (e.subs(x,1) - e.subs(x,0) - Integral(exp(x), (x, 0, 1))
        ).doit().is_zero


def test_subs6():
    a, b = symbols('a b')
    e = Integral(x*y, (x, f(x), f(y)))
    assert e.subs(x, 1) == Integral(x*y, (x, f(1), f(y)))
    assert e.subs(y, 1) == Integral(x, (x, f(x), f(1)))
    e = Integral(x*y, (x, f(x), f(y)), (y, f(x), f(y)))
    assert e.subs(x, 1) == Integral(x*y, (x, f(1), f(y)), (y, f(1), f(y)))
    assert e.subs(y, 1) == Integral(x*y, (x, f(x), f(y)), (y, f(x), f(1)))
    e = Integral(x*y, (x, f(x), f(a)), (y, f(x), f(a)))
    assert e.subs(a, 1) == Integral(x*y, (x, f(x), f(1)), (y, f(x), f(1)))


def test_subs7():
    e = Integral(x, (x, 1, y), (y, 1, 2))
    assert e.subs({x: 1, y: 2}) == e
    e = Integral(sin(x) + sin(y), (x, sin(x), sin(y)),
                                  (y, 1, 2))
    assert e.subs(sin(y), 1) == e
    assert e.subs(sin(x), 1) == Integral(sin(x) + sin(y), (x, 1, sin(y)),
                                         (y, 1, 2))

def test_expand():
    e = Integral(f(x)+f(x**2), (x, 1, y))
    assert e.expand() == Integral(f(x), (x, 1, y)) + Integral(f(x**2), (x, 1, y))
    e = Integral(f(x)+f(x**2), (x, 1, oo))
    assert e.expand() == e
    assert e.expand(force=True) == Integral(f(x), (x, 1, oo)) + \
           Integral(f(x**2), (x, 1, oo))


def test_integration_variable():
    raises(ValueError, lambda: Integral(exp(-x**2), 3))
    raises(ValueError, lambda: Integral(exp(-x**2), (3, -oo, oo)))


def test_expand_integral():
    assert Integral(cos(x**2)*(sin(x**2) + 1), (x, 0, 1)).expand() == \
        Integral(cos(x**2)*sin(x**2), (x, 0, 1)) + \
        Integral(cos(x**2), (x, 0, 1))
    assert Integral(cos(x**2)*(sin(x**2) + 1), x).expand() == \
        Integral(cos(x**2)*sin(x**2), x) + \
        Integral(cos(x**2), x)


def test_as_sum_midpoint1():
    e = Integral(sqrt(x**3 + 1), (x, 2, 10))
    assert e.as_sum(1, method="midpoint") == 8*sqrt(217)
    assert e.as_sum(2, method="midpoint") == 4*sqrt(65) + 12*sqrt(57)
    assert e.as_sum(3, method="midpoint") == 8*sqrt(217)/3 + \
        8*sqrt(3081)/27 + 8*sqrt(52809)/27
    assert e.as_sum(4, method="midpoint") == 2*sqrt(730) + \
        4*sqrt(7) + 4*sqrt(86) + 6*sqrt(14)
    assert abs(e.as_sum(4, method="midpoint").n() - e.n()) < 0.5

    e = Integral(sqrt(x**3 + y**3), (x, 2, 10), (y, 0, 10))
    raises(NotImplementedError, lambda: e.as_sum(4))


def test_as_sum_midpoint2():
    e = Integral((x + y)**2, (x, 0, 1))
    n = Symbol('n', positive=True, integer=True)
    assert e.as_sum(1, method="midpoint").expand() == Rational(1, 4) + y + y**2
    assert e.as_sum(2, method="midpoint").expand() == Rational(5, 16) + y + y**2
    assert e.as_sum(3, method="midpoint").expand() == Rational(35, 108) + y + y**2
    assert e.as_sum(4, method="midpoint").expand() == Rational(21, 64) + y + y**2
    assert e.as_sum(n, method="midpoint").expand() == \
        y**2 + y + Rational(1, 3) - 1/(12*n**2)


def test_as_sum_left():
    e = Integral((x + y)**2, (x, 0, 1))
    assert e.as_sum(1, method="left").expand() == y**2
    assert e.as_sum(2, method="left").expand() == Rational(1, 8) + y/2 + y**2
    assert e.as_sum(3, method="left").expand() == Rational(5, 27) + y*Rational(2, 3) + y**2
    assert e.as_sum(4, method="left").expand() == Rational(7, 32) + y*Rational(3, 4) + y**2
    assert e.as_sum(n, method="left").expand() == \
        y**2 + y + Rational(1, 3) - y/n - 1/(2*n) + 1/(6*n**2)
    assert e.as_sum(10, method="left", evaluate=False).has(Sum)


def test_as_sum_right():
    e = Integral((x + y)**2, (x, 0, 1))
    assert e.as_sum(1, method="right").expand() == 1 + 2*y + y**2
    assert e.as_sum(2, method="right").expand() == Rational(5, 8) + y*Rational(3, 2) + y**2
    assert e.as_sum(3, method="right").expand() == Rational(14, 27) + y*Rational(4, 3) + y**2
    assert e.as_sum(4, method="right").expand() == Rational(15, 32) + y*Rational(5, 4) + y**2
    assert e.as_sum(n, method="right").expand() == \
        y**2 + y + Rational(1, 3) + y/n + 1/(2*n) + 1/(6*n**2)


def test_as_sum_trapezoid():
    e = Integral((x + y)**2, (x, 0, 1))
    assert e.as_sum(1, method="trapezoid").expand() == y**2 + y + S.Half
    assert e.as_sum(2, method="trapezoid").expand() == y**2 + y + Rational(3, 8)
    assert e.as_sum(3, method="trapezoid").expand() == y**2 + y + Rational(19, 54)
    assert e.as_sum(4, method="trapezoid").expand() == y**2 + y + Rational(11, 32)
    assert e.as_sum(n, method="trapezoid").expand() == \
        y**2 + y + Rational(1, 3) + 1/(6*n**2)
    assert Integral(sign(x), (x, 0, 1)).as_sum(1, 'trapezoid') == S.Half


def test_as_sum_raises():
    e = Integral((x + y)**2, (x, 0, 1))
    raises(ValueError, lambda: e.as_sum(-1))
    raises(ValueError, lambda: e.as_sum(0))
    raises(ValueError, lambda: Integral(x).as_sum(3))
    raises(ValueError, lambda: e.as_sum(oo))
    raises(ValueError, lambda: e.as_sum(3, method='xxxx2'))


def test_nested_doit():
    e = Integral(Integral(x, x), x)
    f = Integral(x, x, x)
    assert e.doit() == f.doit()


def test_issue_4665():
    # Allow only upper or lower limit evaluation
    e = Integral(x**2, (x, None, 1))
    f = Integral(x**2, (x, 1, None))
    assert e.doit() == Rational(1, 3)
    assert f.doit() == Rational(-1, 3)
    assert Integral(x*y, (x, None, y)).subs(y, t) == Integral(x*t, (x, None, t))
    assert Integral(x*y, (x, y, None)).subs(y, t) == Integral(x*t, (x, t, None))
    assert integrate(x**2, (x, None, 1)) == Rational(1, 3)
    assert integrate(x**2, (x, 1, None)) == Rational(-1, 3)
    assert integrate("x**2", ("x", "1", None)) == Rational(-1, 3)


def test_integral_reconstruct():
    e = Integral(x**2, (x, -1, 1))
    assert e == Integral(*e.args)


def test_doit_integrals():
    e = Integral(Integral(2*x), (x, 0, 1))
    assert e.doit() == Rational(1, 3)
    assert e.doit(deep=False) == Rational(1, 3)
    f = Function('f')
    # doesn't matter if the integral can't be performed
    assert Integral(f(x), (x, 1, 1)).doit() == 0
    # doesn't matter if the limits can't be evaluated
    assert Integral(0, (x, 1, Integral(f(x), x))).doit() == 0
    assert Integral(x, (a, 0)).doit() == 0
    limits = ((a, 1, exp(x)), (x, 0))
    assert Integral(a, *limits).doit() == Rational(1, 4)
    assert Integral(a, *list(reversed(limits))).doit() == 0


def test_issue_4884():
    assert integrate(sqrt(x)*(1 + x)) == \
        Piecewise(
            (2*sqrt(x)*(x + 1)**2/5 - 2*sqrt(x)*(x + 1)/15 - 4*sqrt(x)/15,
            Abs(x + 1) > 1),
            (2*I*sqrt(-x)*(x + 1)**2/5 - 2*I*sqrt(-x)*(x + 1)/15 -
            4*I*sqrt(-x)/15, True))
    assert integrate(x**x*(1 + log(x))) == x**x

def test_issue_18153():
    assert integrate(x**n*log(x),x) == \
    Piecewise(
        (n*x*x**n*log(x)/(n**2 + 2*n + 1) +
    x*x**n*log(x)/(n**2 + 2*n + 1) - x*x**n/(n**2 + 2*n + 1)
    , Ne(n, -1)), (log(x)**2/2, True)
    )


def test_is_number():
    from sympy.abc import x, y, z
    assert Integral(x).is_number is False
    assert Integral(1, x).is_number is False
    assert Integral(1, (x, 1)).is_number is True
    assert Integral(1, (x, 1, 2)).is_number is True
    assert Integral(1, (x, 1, y)).is_number is False
    assert Integral(1, (x, y)).is_number is False
    assert Integral(x, y).is_number is False
    assert Integral(x, (y, 1, x)).is_number is False
    assert Integral(x, (y, 1, 2)).is_number is False
    assert Integral(x, (x, 1, 2)).is_number is True
    # `foo.is_number` should always be equivalent to `not foo.free_symbols`
    # in each of these cases, there are pseudo-free symbols
    i = Integral(x, (y, 1, 1))
    assert i.is_number is False and i.n() == 0
    i = Integral(x, (y, z, z))
    assert i.is_number is False and i.n() == 0
    i = Integral(1, (y, z, z + 2))
    assert i.is_number is False and i.n() == 2.0

    assert Integral(x*y, (x, 1, 2), (y, 1, 3)).is_number is True
    assert Integral(x*y, (x, 1, 2), (y, 1, z)).is_number is False
    assert Integral(x, (x, 1)).is_number is True
    assert Integral(x, (x, 1, Integral(y, (y, 1, 2)))).is_number is True
    assert Integral(Sum(z, (z, 1, 2)), (x, 1, 2)).is_number is True
    # it is possible to get a false negative if the integrand is
    # actually an unsimplified zero, but this is true of is_number in general.
    assert Integral(sin(x)**2 + cos(x)**2 - 1, x).is_number is False
    assert Integral(f(x), (x, 0, 1)).is_number is True


def test_free_symbols():
    from sympy.abc import x, y, z
    assert Integral(0, x).free_symbols == {x}
    assert Integral(x).free_symbols == {x}
    assert Integral(x, (x, None, y)).free_symbols == {y}
    assert Integral(x, (x, y, None)).free_symbols == {y}
    assert Integral(x, (x, 1, y)).free_symbols == {y}
    assert Integral(x, (x, y, 1)).free_symbols == {y}
    assert Integral(x, (x, x, y)).free_symbols == {x, y}
    assert Integral(x, x, y).free_symbols == {x, y}
    assert Integral(x, (x, 1, 2)).free_symbols == set()
    assert Integral(x, (y, 1, 2)).free_symbols == {x}
    # pseudo-free in this case
    assert Integral(x, (y, z, z)).free_symbols == {x, z}
    assert Integral(x, (y, 1, 2), (y, None, None)
        ).free_symbols == {x, y}
    assert Integral(x, (y, 1, 2), (x, 1, y)
        ).free_symbols == {y}
    assert Integral(2, (y, 1, 2), (y, 1, x), (x, 1, 2)
        ).free_symbols == set()
    assert Integral(2, (y, x, 2), (y, 1, x), (x, 1, 2)
        ).free_symbols == set()
    assert Integral(2, (x, 1, 2), (y, x, 2), (y, 1, 2)
        ).free_symbols == {x}
    assert Integral(f(x), (f(x), 1, y)).free_symbols == {y}
    assert Integral(f(x), (f(x), 1, x)).free_symbols == {x}


def test_is_zero():
    from sympy.abc import x, m
    assert Integral(0, (x, 1, x)).is_zero
    assert Integral(1, (x, 1, 1)).is_zero
    assert Integral(1, (x, 1, 2), (y, 2)).is_zero is False
    assert Integral(x, (m, 0)).is_zero
    assert Integral(x + m, (m, 0)).is_zero is None
    i = Integral(m, (m, 1, exp(x)), (x, 0))
    assert i.is_zero is None
    assert Integral(m, (x, 0), (m, 1, exp(x))).is_zero is True

    assert Integral(x, (x, oo, oo)).is_zero # issue 8171
    assert Integral(x, (x, -oo, -oo)).is_zero

    # this is zero but is beyond the scope of what is_zero
    # should be doing
    assert Integral(sin(x), (x, 0, 2*pi)).is_zero is None


def test_series():
    from sympy.abc import x
    i = Integral(cos(x), (x, x))
    e = i.lseries(x)
    assert i.nseries(x, n=8).removeO() == Add(*[next(e) for j in range(4)])


def test_trig_nonelementary_integrals():
    x = Symbol('x')
    assert integrate((1 + sin(x))/x, x) == log(x) + Si(x)
    # next one comes out as log(x) + log(x**2)/2 + Ci(x)
    # so not hardcoding this log ugliness
    assert integrate((cos(x) + 2)/x, x).has(Ci)


def test_issue_4403():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z', positive=True)
    assert integrate(sqrt(x**2 + z**2), x) == \
        z**2*asinh(x/z)/2 + x*sqrt(x**2 + z**2)/2
    assert integrate(sqrt(x**2 - z**2), x) == \
        x*sqrt(x**2 - z**2)/2 - z**2*log(x + sqrt(x**2 - z**2))/2

    x = Symbol('x', real=True)
    y = Symbol('y', positive=True)
    assert integrate(1/(x**2 + y**2)**S('3/2'), x) == \
        x/(y**2*sqrt(x**2 + y**2))
    # If y is real and nonzero, we get x*Abs(y)/(y**3*sqrt(x**2 + y**2)),
    # which results from sqrt(1 + x**2/y**2) = sqrt(x**2 + y**2)/|y|.


def test_issue_4403_2():
    assert integrate(sqrt(-x**2 - 4), x) == \
        -2*atan(x/sqrt(-4 - x**2)) + x*sqrt(-4 - x**2)/2


def test_issue_4100():
    R = Symbol('R', positive=True)
    assert integrate(sqrt(R**2 - x**2), (x, 0, R)) == pi*R**2/4


def test_issue_5167():
    from sympy.abc import w, x, y, z
    f = Function('f')
    assert Integral(Integral(f(x), x), x) == Integral(f(x), x, x)
    assert Integral(f(x)).args == (f(x), Tuple(x))
    assert Integral(Integral(f(x))).args == (f(x), Tuple(x), Tuple(x))
    assert Integral(Integral(f(x)), y).args == (f(x), Tuple(x), Tuple(y))
    assert Integral(Integral(f(x), z), y).args == (f(x), Tuple(z), Tuple(y))
    assert Integral(Integral(Integral(f(x), x), y), z).args == \
        (f(x), Tuple(x), Tuple(y), Tuple(z))
    assert integrate(Integral(f(x), x), x) == Integral(f(x), x, x)
    assert integrate(Integral(f(x), y), x) == y*Integral(f(x), x)
    assert integrate(Integral(f(x), x), y) in [Integral(y*f(x), x), y*Integral(f(x), x)]
    assert integrate(Integral(2, x), x) == x**2
    assert integrate(Integral(2, x), y) == 2*x*y
    # don't re-order given limits
    assert Integral(1, x, y).args != Integral(1, y, x).args
    # do as many as possible
    assert Integral(f(x), y, x, y, x).doit() == y**2*Integral(f(x), x, x)/2
    assert Integral(f(x), (x, 1, 2), (w, 1, x), (z, 1, y)).doit() == \
        y*(x - 1)*Integral(f(x), (x, 1, 2)) - (x - 1)*Integral(f(x), (x, 1, 2))


def test_issue_4890():
    z = Symbol('z', positive=True)
    assert integrate(exp(-log(x)**2), x) == \
        sqrt(pi)*exp(Rational(1, 4))*erf(log(x) - S.Half)/2
    assert integrate(exp(log(x)**2), x) == \
        sqrt(pi)*exp(Rational(-1, 4))*erfi(log(x)+S.Half)/2
    assert integrate(exp(-z*log(x)**2), x) == \
        sqrt(pi)*exp(1/(4*z))*erf(sqrt(z)*log(x) - 1/(2*sqrt(z)))/(2*sqrt(z))


def test_issue_4551():
    assert not integrate(1/(x*sqrt(1 - x**2)), x).has(Integral)


def test_issue_4376():
    n = Symbol('n', integer=True, positive=True)
    assert simplify(integrate(n*(x**(1/n) - 1), (x, 0, S.Half)) -
                (n**2 - 2**(1/n)*n**2 - n*2**(1/n))/(2**(1 + 1/n) + n*2**(1 + 1/n))) == 0


def test_issue_4517():
    assert integrate((sqrt(x) - x**3)/x**Rational(1, 3), x) == \
        6*x**Rational(7, 6)/7 - 3*x**Rational(11, 3)/11


def test_issue_4527():
    k, m = symbols('k m', integer=True)
    assert integrate(sin(k*x)*sin(m*x), (x, 0, pi)).simplify() == \
        Piecewise((0, Eq(k, 0) | Eq(m, 0)),
                  (-pi/2, Eq(k, -m) | (Eq(k, 0) & Eq(m, 0))),
                  (pi/2, Eq(k, m) | (Eq(k, 0) & Eq(m, 0))),
                  (0, True))
    # Should be possible to further simplify to:
    # Piecewise(
    #    (0, Eq(k, 0) | Eq(m, 0)),
    #    (-pi/2, Eq(k, -m)),
    #    (pi/2, Eq(k, m)),
    #    (0, True))
    assert integrate(sin(k*x)*sin(m*x), (x,)) == Piecewise(
        (0, And(Eq(k, 0), Eq(m, 0))),
        (-x*sin(m*x)**2/2 - x*cos(m*x)**2/2 + sin(m*x)*cos(m*x)/(2*m), Eq(k, -m)),
        (x*sin(m*x)**2/2 + x*cos(m*x)**2/2 - sin(m*x)*cos(m*x)/(2*m), Eq(k, m)),
        (m*sin(k*x)*cos(m*x)/(k**2 - m**2) -
         k*sin(m*x)*cos(k*x)/(k**2 - m**2), True))


def test_issue_4199():
    ypos = Symbol('y', positive=True)
    # TODO: Remove conds='none' below, let the assumption take care of it.
    assert integrate(exp(-I*2*pi*ypos*x)*x, (x, -oo, oo), conds='none') == \
        Integral(exp(-I*2*pi*ypos*x)*x, (x, -oo, oo))


def test_issue_3940():
    a, b, c, d = symbols('a:d', positive=True)
    assert integrate(exp(-x**2 + I*c*x), x) == \
        -sqrt(pi)*exp(-c**2/4)*erf(I*c/2 - x)/2
    assert integrate(exp(a*x**2 + b*x + c), x) == \
        sqrt(pi)*exp(c - b**2/(4*a))*erfi((2*a*x + b)/(2*sqrt(a)))/(2*sqrt(a))

    from sympy.core.function import expand_mul
    from sympy.abc import k
    assert expand_mul(integrate(exp(-x**2)*exp(I*k*x), (x, -oo, oo))) == \
        sqrt(pi)*exp(-k**2/4)
    a, d = symbols('a d', positive=True)
    assert expand_mul(integrate(exp(-a*x**2 + 2*d*x), (x, -oo, oo))) == \
        sqrt(pi)*exp(d**2/a)/sqrt(a)


def test_issue_5413():
    # Note that this is not the same as testing ratint() because integrate()
    # pulls out the coefficient.
    assert integrate(-a/(a**2 + x**2), x) == I*log(-I*a + x)/2 - I*log(I*a + x)/2


def test_issue_4892a():
    A, z = symbols('A z')
    c = Symbol('c', nonzero=True)
    P1 = -A*exp(-z)
    P2 = -A/(c*t)*(sin(x)**2 + cos(y)**2)

    h1 = -sin(x)**2 - cos(y)**2
    h2 = -sin(x)**2 + sin(y)**2 - 1

    # there is still some non-deterministic behavior in integrate
    # or trigsimp which permits one of the following
    assert integrate(c*(P2 - P1), t) in [
        c*(-A*(-h1)*log(c*t)/c + A*t*exp(-z)),
        c*(-A*(-h2)*log(c*t)/c + A*t*exp(-z)),
        c*( A* h1 *log(c*t)/c + A*t*exp(-z)),
        c*( A* h2 *log(c*t)/c + A*t*exp(-z)),
        (A*c*t - A*(-h1)*log(t)*exp(z))*exp(-z),
        (A*c*t - A*(-h2)*log(t)*exp(z))*exp(-z),
    ]


def test_issue_4892b():
    # Issues relating to issue 4596 are making the actual result of this hard
    # to test.  The answer should be something like
    #
    # (-sin(y) + sqrt(-72 + 48*cos(y) - 8*cos(y)**2)/2)*log(x + sqrt(-72 +
    # 48*cos(y) - 8*cos(y)**2)/(2*(3 - cos(y)))) + (-sin(y) - sqrt(-72 +
    # 48*cos(y) - 8*cos(y)**2)/2)*log(x - sqrt(-72 + 48*cos(y) -
    # 8*cos(y)**2)/(2*(3 - cos(y)))) + x**2*sin(y)/2 + 2*x*cos(y)

    expr = (sin(y)*x**3 + 2*cos(y)*x**2 + 12)/(x**2 + 2)
    assert trigsimp(factor(integrate(expr, x).diff(x) - expr)) == 0


def test_issue_5178():
    assert integrate(sin(x)*f(y, z), (x, 0, pi), (y, 0, pi), (z, 0, pi)) == \
        2*Integral(f(y, z), (y, 0, pi), (z, 0, pi))


def test_integrate_series():
    f = sin(x).series(x, 0, 10)
    g = x**2/2 - x**4/24 + x**6/720 - x**8/40320 + x**10/3628800 + O(x**11)

    assert integrate(f, x) == g
    assert diff(integrate(f, x), x) == f

    assert integrate(O(x**5), x) == O(x**6)


def test_atom_bug():
    from sympy.integrals.heurisch import heurisch
    assert heurisch(meijerg([], [], [1], [], x), x) is None


def test_limit_bug():
    z = Symbol('z', zero=False)
    assert integrate(sin(x*y*z), (x, 0, pi), (y, 0, pi)).together() == \
        (log(z) - Ci(pi**2*z) + EulerGamma + 2*log(pi))/z


def test_issue_4703():
    g = Function('g')
    assert integrate(exp(x)*g(x), x).has(Integral)


def test_issue_1888():
    f = Function('f')
    assert integrate(f(x).diff(x)**2, x).has(Integral)

# The following tests work using meijerint.


def test_issue_3558():
    assert integrate(cos(x*y), (x, -pi/2, pi/2), (y, 0, pi)) == 2*Si(pi**2/2)


def test_issue_4422():
    assert integrate(1/sqrt(16 + 4*x**2), x) == asinh(x/2) / 2


def test_issue_4493():
    assert simplify(integrate(x*sqrt(1 + 2*x), x)) == \
        sqrt(2*x + 1)*(6*x**2 + x - 1)/15


def test_issue_4737():
    assert integrate(sin(x)/x, (x, -oo, oo)) == pi
    assert integrate(sin(x)/x, (x, 0, oo)) == pi/2
    assert integrate(sin(x)/x, x) == Si(x)


def test_issue_4992():
    # Note: psi in _check_antecedents becomes NaN.
    from sympy.core.function import expand_func
    a = Symbol('a', positive=True)
    assert simplify(expand_func(integrate(exp(-x)*log(x)*x**a, (x, 0, oo)))) == \
        (a*polygamma(0, a) + 1)*gamma(a)


def test_issue_4487():
    from sympy.functions.special.gamma_functions import lowergamma
    assert simplify(integrate(exp(-x)*x**y, x)) == lowergamma(y + 1, x)


def test_issue_4215():
    x = Symbol("x")
    assert integrate(1/(x**2), (x, -1, 1)) is oo


def test_issue_4400():
    n = Symbol('n', integer=True, positive=True)
    assert integrate((x**n)*log(x), x) == \
        n*x*x**n*log(x)/(n**2 + 2*n + 1) + x*x**n*log(x)/(n**2 + 2*n + 1) - \
        x*x**n/(n**2 + 2*n + 1)


def test_issue_6253():
    # Note: this used to raise NotImplementedError
    # Note: psi in _check_antecedents becomes NaN.
    assert integrate((sqrt(1 - x) + sqrt(1 + x))**2/x, x, meijerg=True) == \
        Integral((sqrt(-x + 1) + sqrt(x + 1))**2/x, x)


def test_issue_4153():
    assert integrate(1/(1 + x + y + z), (x, 0, 1), (y, 0, 1), (z, 0, 1)) in [
        -12*log(3) - 3*log(6)/2 + 3*log(8)/2 + 5*log(2) + 7*log(4),
        6*log(2) + 8*log(4) - 27*log(3)/2, 22*log(2) - 27*log(3)/2,
        -12*log(3) - 3*log(6)/2 + 47*log(2)/2]


def test_issue_4326():
    R, b, h = symbols('R b h')
    # It doesn't matter if we can do the integral.  Just make sure the result
    # doesn't contain nan.  This is really a test against _eval_interval.
    e = integrate(((h*(x - R + b))/b)*sqrt(R**2 - x**2), (x, R - b, R))
    assert not e.has(nan)
    # See that it evaluates
    assert not e.has(Integral)


def test_powers():
    assert integrate(2**x + 3**x, x) == 2**x/log(2) + 3**x/log(3)


def test_manual_option():
    raises(ValueError, lambda: integrate(1/x, x, manual=True, meijerg=True))
    # an example of a function that manual integration cannot handle
    assert integrate(log(1+x)/x, (x, 0, 1), manual=True).has(Integral)


def test_meijerg_option():
    raises(ValueError, lambda: integrate(1/x, x, meijerg=True, risch=True))
    # an example of a function that meijerg integration cannot handle
    assert integrate(tan(x), x, meijerg=True) == Integral(tan(x), x)


def test_risch_option():
    # risch=True only allowed on indefinite integrals
    raises(ValueError, lambda: integrate(1/log(x), (x, 0, oo), risch=True))
    assert integrate(exp(-x**2), x, risch=True) == NonElementaryIntegral(exp(-x**2), x)
    assert integrate(log(1/x)*y, x, y, risch=True) == y**2*(x*log(1/x)/2 + x/2)
    assert integrate(erf(x), x, risch=True) == Integral(erf(x), x)
    # TODO: How to test risch=False?


@slow
def test_heurisch_option():
    raises(ValueError, lambda: integrate(1/x, x, risch=True, heurisch=True))
    # an integral that heurisch can handle
    assert integrate(exp(x**2), x, heurisch=True) == sqrt(pi)*erfi(x)/2
    # an integral that heurisch currently cannot handle
    assert integrate(exp(x)/x, x, heurisch=True) == Integral(exp(x)/x, x)
    # an integral where heurisch currently hangs, issue 15471
    assert integrate(log(x)*cos(log(x))/x**Rational(3, 4), x, heurisch=False) == (
        -128*x**Rational(1, 4)*sin(log(x))/289 + 240*x**Rational(1, 4)*cos(log(x))/289 +
        (16*x**Rational(1, 4)*sin(log(x))/17 + 4*x**Rational(1, 4)*cos(log(x))/17)*log(x))


def test_issue_6828():
    f = 1/(1.08*x**2 - 4.3)
    g = integrate(f, x).diff(x)
    assert verify_numerically(f, g, tol=1e-12)


def test_issue_4803():
    x_max = Symbol("x_max")
    assert integrate(y/pi*exp(-(x_max - x)/cos(a)), x) == \
        y*exp((x - x_max)/cos(a))*cos(a)/pi


def test_issue_4234():
    assert integrate(1/sqrt(1 + tan(x)**2)) == tan(x)/sqrt(1 + tan(x)**2)


def test_issue_4492():
    assert simplify(integrate(x**2 * sqrt(5 - x**2), x)).factor(
        deep=True) == Piecewise(
        (I*(2*x**5 - 15*x**3 + 25*x - 25*sqrt(x**2 - 5)*acosh(sqrt(5)*x/5)) /
            (8*sqrt(x**2 - 5)), (x > sqrt(5)) | (x < -sqrt(5))),
        ((2*x**5 - 15*x**3 + 25*x - 25*sqrt(5 - x**2)*asin(sqrt(5)*x/5)) /
            (-8*sqrt(-x**2 + 5)), True))


def test_issue_2708():
    # This test needs to use an integration function that can
    # not be evaluated in closed form.  Update as needed.
    f = 1/(a + z + log(z))
    integral_f = NonElementaryIntegral(f, (z, 2, 3))
    assert Integral(f, (z, 2, 3)).doit() == integral_f
    assert integrate(f + exp(z), (z, 2, 3)) == integral_f - exp(2) + exp(3)
    assert integrate(2*f + exp(z), (z, 2, 3)) == \
        2*integral_f - exp(2) + exp(3)
    assert integrate(exp(1.2*n*s*z*(-t + z)/t), (z, 0, x)) == \
        NonElementaryIntegral(exp(-1.2*n*s*z)*exp(1.2*n*s*z**2/t),
                                  (z, 0, x))


def test_issue_2884():
    f = (4.000002016020*x + 4.000002016020*y + 4.000006024032)*exp(10.0*x)
    e = integrate(f, (x, 0.1, 0.2))
    assert str(e) == '1.86831064982608*y + 2.16387491480008'


def test_issue_8368i():
    from sympy.functions.elementary.complexes import arg, Abs
    assert integrate(exp(-s*x)*cosh(x), (x, 0, oo)) == \
        Piecewise(
            (   pi*Piecewise(
                    (   -s/(pi*(-s**2 + 1)),
                        Abs(s**2) < 1),
                    (   1/(pi*s*(1 - 1/s**2)),
                        Abs(s**(-2)) < 1),
                    (   meijerg(
                            ((S.Half,), (0, 0)),
                            ((0, S.Half), (0,)),
                            polar_lift(s)**2),
                        True)
                ),
                s**2 > 1
            ),
            (
                Integral(exp(-s*x)*cosh(x), (x, 0, oo)),
                True))
    assert integrate(exp(-s*x)*sinh(x), (x, 0, oo)) == \
        Piecewise(
            (   -1/(s + 1)/2 - 1/(-s + 1)/2,
                And(
                    Abs(s) > 1,
                    Abs(arg(s)) < pi/2,
                    Abs(arg(s)) <= pi/2
                    )),
            (   Integral(exp(-s*x)*sinh(x), (x, 0, oo)),
                True))


def test_issue_8901():
    assert integrate(sinh(1.0*x)) == 1.0*cosh(1.0*x)
    assert integrate(tanh(1.0*x)) == 1.0*x - 1.0*log(tanh(1.0*x) + 1)
    assert integrate(tanh(x)) == x - log(tanh(x) + 1)


@slow
def test_issue_8945():
    assert integrate(sin(x)**3/x, (x, 0, 1)) == -Si(3)/4 + 3*Si(1)/4
    assert integrate(sin(x)**3/x, (x, 0, oo)) == pi/4
    assert integrate(cos(x)**2/x**2, x) == -Si(2*x) - cos(2*x)/(2*x) - 1/(2*x)


@slow
def test_issue_7130():
    i, L, a, b = symbols('i L a b')
    integrand = (cos(pi*i*x/L)**2 / (a + b*x)).rewrite(exp)
    assert x not in integrate(integrand, (x, 0, L)).free_symbols


def test_issue_10567():
    a, b, c, t = symbols('a b c t')
    vt = Matrix([a*t, b, c])
    assert integrate(vt, t) == Integral(vt, t).doit()
    assert integrate(vt, t) == Matrix([[a*t**2/2], [b*t], [c*t]])


def test_issue_11742():
    assert integrate(sqrt(-x**2 + 8*x + 48), (x, 4, 12)) == 16*pi


def test_issue_11856():
    t = symbols('t')
    assert integrate(sinc(pi*t), t) == Si(pi*t)/pi


@slow
def test_issue_11876():
    assert integrate(sqrt(log(1/x)), (x, 0, 1)) == sqrt(pi)/2


def test_issue_4950():
    assert integrate((-60*exp(x) - 19.2*exp(4*x))*exp(4*x), x) ==\
        -2.4*exp(8*x) - 12.0*exp(5*x)


def test_issue_4968():
    assert integrate(sin(log(x**2))) == x*sin(log(x**2))/5 - 2*x*cos(log(x**2))/5


def test_singularities():
    assert integrate(1/x**2, (x, -oo, oo)) is oo
    assert integrate(1/x**2, (x, -1, 1)) is oo
    assert integrate(1/(x - 1)**2, (x, -2, 2)) is oo

    assert integrate(1/x**2, (x, 1, -1)) is -oo
    assert integrate(1/(x - 1)**2, (x, 2, -2)) is -oo


def test_issue_12645():
    x, y = symbols('x y', real=True)
    assert (integrate(sin(x*x*x + y*y),
                      (x, -sqrt(pi - y*y), sqrt(pi - y*y)),
                      (y, -sqrt(pi), sqrt(pi)))
                == Integral(sin(x**3 + y**2),
                            (x, -sqrt(-y**2 + pi), sqrt(-y**2 + pi)),
                            (y, -sqrt(pi), sqrt(pi))))


def test_issue_12677():
    assert integrate(sin(x) / (cos(x)**3), (x, 0, pi/6)) == Rational(1, 6)


def test_issue_14078():
    assert integrate((cos(3*x)-cos(x))/x, (x, 0, oo)) == -log(3)


def test_issue_14064():
    assert integrate(1/cosh(x), (x, 0, oo)) == pi/2


def test_issue_14027():
    assert integrate(1/(1 + exp(x - S.Half)/(1 + exp(x))), x) == \
        x - exp(S.Half)*log(exp(x) + exp(S.Half)/(1 + exp(S.Half)))/(exp(S.Half) + E)


def test_issue_8170():
    assert integrate(tan(x), (x, 0, pi/2)) is S.Infinity


def test_issue_8440_14040():
    assert integrate(1/x, (x, -1, 1)) is S.NaN
    assert integrate(1/(x + 1), (x, -2, 3)) is S.NaN


def test_issue_14096():
    assert integrate(1/(x + y)**2, (x, 0, 1)) == -1/(y + 1) + 1/y
    assert integrate(1/(1 + x + y + z)**2, (x, 0, 1), (y, 0, 1), (z, 0, 1)) == \
        -4*log(4) - 6*log(2) + 9*log(3)


def test_issue_14144():
    assert Abs(integrate(1/sqrt(1 - x**3), (x, 0, 1)).n() - 1.402182) < 1e-6
    assert Abs(integrate(sqrt(1 - x**3), (x, 0, 1)).n() - 0.841309) < 1e-6


def test_issue_14375():
    # This raised a TypeError. The antiderivative has exp_polar, which
    # may be possible to unpolarify, so the exact output is not asserted here.
    assert integrate(exp(I*x)*log(x), x).has(Ei)


def test_issue_14437():
    f = Function('f')(x, y, z)
    assert integrate(f, (x, 0, 1), (y, 0, 2), (z, 0, 3)) == \
                Integral(f, (x, 0, 1), (y, 0, 2), (z, 0, 3))


def test_issue_14470():
    assert integrate(1/sqrt(exp(x) + 1), x) == log(sqrt(exp(x) + 1) - 1) - log(sqrt(exp(x) + 1) + 1)


def test_issue_14877():
    f = exp(1 - exp(x**2)*x + 2*x**2)*(2*x**3 + x)/(1 - exp(x**2)*x)**2
    assert integrate(f, x) == \
        -exp(2*x**2 - x*exp(x**2) + 1)/(x*exp(3*x**2) - exp(2*x**2))


def test_issue_14782():
    f = sqrt(-x**2 + 1)*(-x**2 + x)
    assert integrate(f, [x, -1, 1]) == - pi / 8


@slow
def test_issue_14782_slow():
    f = sqrt(-x**2 + 1)*(-x**2 + x)
    assert integrate(f, [x, 0, 1]) == S.One / 3 - pi / 16


def test_issue_12081():
    f = x**(Rational(-3, 2))*exp(-x)
    assert integrate(f, [x, 0, oo]) is oo


def test_issue_15285():
    y = 1/x - 1
    f = 4*y*exp(-2*y)/x**2
    assert integrate(f, [x, 0, 1]) == 1


def test_issue_15432():
    assert integrate(x**n * exp(-x) * log(x), (x, 0, oo)).gammasimp() == Piecewise(
        (gamma(n + 1)*polygamma(0, n) + gamma(n + 1)/n, re(n) + 1 > 0),
        (Integral(x**n*exp(-x)*log(x), (x, 0, oo)), True))


def test_issue_15124():
    omega = IndexedBase('omega')
    m, p = symbols('m p', cls=Idx)
    assert integrate(exp(x*I*(omega[m] + omega[p])), x, conds='none') == \
        -I*exp(I*x*omega[m])*exp(I*x*omega[p])/(omega[m] + omega[p])


def test_issue_15218():
    with warns_deprecated_sympy():
        Integral(Eq(x, y))
    with warns_deprecated_sympy():
        assert Integral(Eq(x, y), x) == Eq(Integral(x, x), Integral(y, x))
    with warns_deprecated_sympy():
        assert Integral(Eq(x, y), x).doit() == Eq(x**2/2, x*y)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        # The warning is made in the ExprWithLimits superclass. The stacklevel
        # is correct for integrate(Eq) but not Eq.integrate
        assert Eq(x, y).integrate(x) == Eq(x**2/2, x*y)

    # These are not deprecated because they are definite integrals
    assert integrate(Eq(x, y), (x, 0, 1)) == Eq(S.Half, y)
    assert Eq(x, y).integrate((x, 0, 1)) == Eq(S.Half, y)


def test_issue_15292():
    res = integrate(exp(-x**2*cos(2*t)) * cos(x**2*sin(2*t)), (x, 0, oo))
    assert isinstance(res, Piecewise)
    assert gammasimp((res - sqrt(pi)/2 * cos(t)).subs(t, pi/6)) == 0


def test_issue_4514():
    assert integrate(sin(2*x)/sin(x), x) == 2*sin(x)


def test_issue_15457():
    x, a, b = symbols('x a b', real=True)
    definite = integrate(exp(Abs(x-2)), (x, a, b))
    indefinite = integrate(exp(Abs(x-2)), x)
    assert definite.subs({a: 1, b: 3}) == -2 + 2*E
    assert indefinite.subs(x, 3) - indefinite.subs(x, 1) == -2 + 2*E
    assert definite.subs({a: -3, b: -1}) == -exp(3) + exp(5)
    assert indefinite.subs(x, -1) - indefinite.subs(x, -3) == -exp(3) + exp(5)


def test_issue_15431():
    assert integrate(x*exp(x)*log(x), x) == \
        (x*exp(x) - exp(x))*log(x) - exp(x) + Ei(x)


def test_issue_15640_log_substitutions():
    f = x/log(x)
    F = Ei(2*log(x))
    assert integrate(f, x) == F and F.diff(x) == f
    f = x**3/log(x)**2
    F = -x**4/log(x) + 4*Ei(4*log(x))
    assert integrate(f, x) == F and F.diff(x) == f
    f = sqrt(log(x))/x**2
    F = -sqrt(pi)*erfc(sqrt(log(x)))/2 - sqrt(log(x))/x
    assert integrate(f, x) == F and F.diff(x) == f


def test_issue_15509():
    from sympy.vector import CoordSys3D
    N = CoordSys3D('N')
    x = N.x
    assert integrate(cos(a*x + b), (x, x_1, x_2), heurisch=True) == Piecewise(
        (-sin(a*x_1 + b)/a + sin(a*x_2 + b)/a, (a > -oo) & (a < oo) & Ne(a, 0)), \
            (-x_1*cos(b) + x_2*cos(b), True))


def test_issue_4311_fast():
    x = symbols('x', real=True)
    assert integrate(x*abs(9-x**2), x) == Piecewise(
        (x**4/4 - 9*x**2/2, x <= -3),
        (-x**4/4 + 9*x**2/2 - Rational(81, 2), x <= 3),
        (x**4/4 - 9*x**2/2, True))


def test_integrate_with_complex_constants():
    K = Symbol('K', positive=True)
    x = Symbol('x', real=True)
    m = Symbol('m', real=True)
    t = Symbol('t', real=True)
    assert integrate(exp(-I*K*x**2+m*x), x) == sqrt(pi)*exp(-I*m**2
                    /(4*K))*erfi((-2*I*K*x + m)/(2*sqrt(K)*sqrt(-I)))/(2*sqrt(K)*sqrt(-I))
    assert integrate(1/(1 + I*x**2), x) == (-I*(sqrt(-I)*log(x - I*sqrt(-I))/2
            - sqrt(-I)*log(x + I*sqrt(-I))/2))
    assert integrate(exp(-I*x**2), x) == sqrt(pi)*erf(sqrt(I)*x)/(2*sqrt(I))

    assert integrate((1/(exp(I*t)-2)), t) == -t/2 - I*log(exp(I*t) - 2)/2
    assert integrate((1/(exp(I*t)-2)), (t, 0, 2*pi)) == -pi


def test_issue_14241():
    x = Symbol('x')
    n = Symbol('n', positive=True, integer=True)
    assert integrate(n * x ** (n - 1) / (x + 1), x) == \
           n**2*x**n*lerchphi(x*exp_polar(I*pi), 1, n)*gamma(n)/gamma(n + 1)


def test_issue_13112():
    assert integrate(sin(t)**2 / (5 - 4*cos(t)), [t, 0, 2*pi]) == pi / 4


def test_issue_14709b():
    h = Symbol('h', positive=True)
    i = integrate(x*acos(1 - 2*x/h), (x, 0, h))
    assert i == 5*h**2*pi/16


def test_issue_8614():
    x = Symbol('x')
    t = Symbol('t')
    assert integrate(exp(t)/t, (t, -oo, x)) == Ei(x)
    assert integrate((exp(-x) - exp(-2*x))/x, (x, 0, oo)) == log(2)


@slow
def test_issue_15494():
    s = symbols('s', positive=True)

    integrand = (exp(s/2) - 2*exp(1.6*s) + exp(s))*exp(s)
    solution = integrate(integrand, s)
    assert solution != S.NaN
    # Not sure how to test this properly as it is a symbolic expression with floats
    # assert str(solution) == '0.666666666666667*exp(1.5*s) + 0.5*exp(2.0*s) - 0.769230769230769*exp(2.6*s)'
    # Maybe
    assert abs(solution.subs(s, 1) - (-3.67440080236188)) <= 1e-8

    integrand = (exp(s/2) - 2*exp(S(8)/5*s) + exp(s))*exp(s)
    assert integrate(integrand, s) == -10*exp(13*s/5)/13 + 2*exp(3*s/2)/3 + exp(2*s)/2


def test_li_integral():
    y = Symbol('y')
    assert Integral(li(y*x**2), x).doit() == Piecewise((x*li(x**2*y) - \
        x*Ei(3*log(x**2*y)/2)/sqrt(x**2*y),
        Ne(y, 0)), (0, True))


def test_issue_17473():
    x = Symbol('x')
    n = Symbol('n')
    h = S.Half
    ans = x**(n + 1)*gamma(h + h/n)*hyper((h + h/n,),
        (3*h, 3*h + h/n), -x**(2*n)/4)/(2*n*gamma(3*h + h/n))
    got = integrate(sin(x**n), x)
    assert got == ans
    _x = Symbol('x', zero=False)
    reps = {x: _x}
    assert integrate(sin(_x**n), _x) == ans.xreplace(reps).expand()


def test_issue_17671():
    assert integrate(log(log(x)) / x**2, [x, 1, oo]) == -EulerGamma
    assert integrate(log(log(x)) / x**3, [x, 1, oo]) == -log(2)/2 - EulerGamma/2
    assert integrate(log(log(x)) / x**10, [x, 1, oo]) == -log(9)/9 - EulerGamma/9


def test_issue_2975():
    w = Symbol('w')
    C = Symbol('C')
    y = Symbol('y')
    assert integrate(1/(y**2+C)**(S(3)/2), (y, -w/2, w/2)) == w/(C**(S(3)/2)*sqrt(1 + w**2/(4*C)))


def test_issue_7827():
    x, n, M = symbols('x n M')
    N = Symbol('N', integer=True)
    assert integrate(summation(x*n, (n, 1, N)), x) == x**2*(N**2/4 + N/4)
    assert integrate(summation(x*sin(n), (n,1,N)), x) == \
        Sum(x**2*sin(n)/2, (n, 1, N))
    assert integrate(summation(sin(n*x), (n,1,N)), x) == \
        Sum(Piecewise((-cos(n*x)/n, Ne(n, 0)), (0, True)), (n, 1, N))
    assert integrate(integrate(summation(sin(n*x), (n,1,N)), x), x) == \
        Piecewise((Sum(Piecewise((-sin(n*x)/n**2, Ne(n, 0)), (-x/n, True)),
        (n, 1, N)), (n > -oo) & (n < oo) & Ne(n, 0)), (0, True))
    assert integrate(Sum(x, (n, 1, M)), x) == M*x**2/2
    raises(ValueError, lambda: integrate(Sum(x, (x, y, n)), y))
    raises(ValueError, lambda: integrate(Sum(x, (x, 1, n)), n))
    raises(ValueError, lambda: integrate(Sum(x, (x, 1, y)), x))


def test_issue_4231():
    f = (1 + 2*x + sqrt(x + log(x))*(1 + 3*x) + x**2)/(x*(x + sqrt(x + log(x)))*sqrt(x + log(x)))
    assert integrate(f, x) == 2*sqrt(x + log(x)) + 2*log(x + sqrt(x + log(x)))


def test_issue_17841():
    f = diff(1/(x**2+x+I), x)
    assert integrate(f, x) == 1/(x**2 + x + I)


def test_issue_21034():
    x = Symbol('x', real=True, nonzero=True)
    f1 = x*(-x**4/asin(5)**4 - x*sinh(x + log(asin(5))) + 5)
    f2 = (x + cosh(cos(4)))/(x*(x + 1/(12*x)))

    assert integrate(f1, x) == \
        -x**6/(6*asin(5)**4) - x**2*cosh(x + log(asin(5))) + 5*x**2/2 + 2*x*sinh(x + log(asin(5))) - 2*cosh(x + log(asin(5)))

    assert integrate(f2, x) == \
        log(x**2 + S(1)/12)/2 + 2*sqrt(3)*cosh(cos(4))*atan(2*sqrt(3)*x)


def test_issue_4187():
    assert integrate(log(x)*exp(-x), x) == Ei(-x) - exp(-x)*log(x)
    assert integrate(log(x)*exp(-x), (x, 0, oo)) == -EulerGamma


def test_issue_5547():
    L = Symbol('L')
    z = Symbol('z')
    r0 = Symbol('r0')
    R0 = Symbol('R0')

    assert integrate(r0**2*cos(z)**2, (z, -L/2, L/2)) == -r0**2*(-L/4 -
                    sin(L/2)*cos(L/2)/2) + r0**2*(L/4 + sin(L/2)*cos(L/2)/2)

    assert integrate(r0**2*cos(R0*z)**2, (z, -L/2, L/2)) == Piecewise(
        (-r0**2*(-L*R0/4 - sin(L*R0/2)*cos(L*R0/2)/2)/R0 +
         r0**2*(L*R0/4 + sin(L*R0/2)*cos(L*R0/2)/2)/R0, (R0 > -oo) & (R0 < oo) & Ne(R0, 0)),
        (L*r0**2, True))

    w = 2*pi*z/L

    sol = sqrt(2)*sqrt(L)*r0**2*fresnelc(sqrt(2)*sqrt(L))*gamma(S.One/4)/(16*gamma(S(5)/4)) + L*r0**2/2

    assert integrate(r0**2*cos(w*z)**2, (z, -L/2, L/2)) == sol


def test_issue_15810():
    assert integrate(1/(2**(2*x/3) + 1), (x, 0, oo)) == Rational(3, 2)


def test_issue_21024():
    x = Symbol('x', real=True, nonzero=True)
    f = log(x)*log(4*x) + log(3*x + exp(2))
    F = x*log(x)**2 + x*(1 - 2*log(2)) + (-2*x + 2*x*log(2))*log(x) + \
        (x + exp(2)/6)*log(3*x + exp(2)) + exp(2)*log(3*x + exp(2))/6
    assert F == integrate(f, x)

    f = (x + exp(3))/x**2
    F = log(x) - exp(3)/x
    assert F == integrate(f, x)

    f = (x**2 + exp(5))/x
    F = x**2/2 + exp(5)*log(x)
    assert F == integrate(f, x)

    f = x/(2*x + tanh(1))
    F = x/2 - log(2*x + tanh(1))*tanh(1)/4
    assert F == integrate(f, x)

    f = x - sinh(4)/x
    F = x**2/2 - log(x)*sinh(4)
    assert F == integrate(f, x)

    f = log(x + exp(5)/x)
    F = x*log(x + exp(5)/x) - x + 2*exp(Rational(5, 2))*atan(x*exp(Rational(-5, 2)))
    assert F == integrate(f, x)

    f = x**5/(x + E)
    F = x**5/5 - E*x**4/4 + x**3*exp(2)/3 - x**2*exp(3)/2 + x*exp(4) - exp(5)*log(x + E)
    assert F == integrate(f, x)

    f = 4*x/(x + sinh(5))
    F = 4*x - 4*log(x + sinh(5))*sinh(5)
    assert F == integrate(f, x)

    f = x**2/(2*x + sinh(2))
    F = x**2/4 - x*sinh(2)/4 + log(2*x + sinh(2))*sinh(2)**2/8
    assert F == integrate(f, x)

    f = -x**2/(x + E)
    F = -x**2/2 + E*x - exp(2)*log(x + E)
    assert F == integrate(f, x)

    f = (2*x + 3)*exp(5)/x
    F = 2*x*exp(5) + 3*exp(5)*log(x)
    assert F == integrate(f, x)

    f = x + 2 + cosh(3)/x
    F = x**2/2 + 2*x + log(x)*cosh(3)
    assert F == integrate(f, x)

    f = x - tanh(1)/x**3
    F = x**2/2 + tanh(1)/(2*x**2)
    assert F == integrate(f, x)

    f = (3*x - exp(6))/x
    F = 3*x - exp(6)*log(x)
    assert F == integrate(f, x)

    f = x**4/(x + exp(5))**2 + x
    F = x**3/3 + x**2*(Rational(1, 2) - exp(5)) + 3*x*exp(10) - 4*exp(15)*log(x + exp(5)) - exp(20)/(x + exp(5))
    assert F == integrate(f, x)

    f = x*(x + exp(10)/x**2) + x
    F = x**3/3 + x**2/2 + exp(10)*log(x)
    assert F == integrate(f, x)

    f = x + x/(5*x + sinh(3))
    F = x**2/2 + x/5 - log(5*x + sinh(3))*sinh(3)/25
    assert F == integrate(f, x)

    f = (x + exp(3))/(2*x**2 + 2*x)
    F = exp(3)*log(x)/2 - exp(3)*log(x + 1)/2 + log(x + 1)/2
    assert F == integrate(f, x).expand()

    f = log(x + 4*sinh(4))
    F = x*log(x + 4*sinh(4)) - x + 4*log(x + 4*sinh(4))*sinh(4)
    assert F == integrate(f, x)

    f = -x + 20*(exp(-5) - atan(4)/x)**3*sin(4)/x
    F = (-x**2*exp(15)/2 + 20*log(x)*sin(4) - (-180*x**2*exp(5)*sin(4)*atan(4) + 90*x*exp(10)*sin(4)*atan(4)**2 - \
        20*exp(15)*sin(4)*atan(4)**3)/(3*x**3))*exp(-15)
    assert F == integrate(f, x)

    f = 2*x**2*exp(-4) + 6/x
    F_true = (2*x**3/3 + 6*exp(4)*log(x))*exp(-4)
    assert F_true == integrate(f, x)


def test_issue_21721():
    a = Symbol('a')
    assert integrate(1/(pi*(1+(x-a)**2)),(x,-oo,oo)).expand() == \
    -Heaviside(im(a) - 1, 0) + Heaviside(im(a) + 1, 0)


def test_issue_21831():
    theta = symbols('theta')
    assert integrate(cos(3*theta)/(5-4*cos(theta)), (theta, 0, 2*pi)) == pi/12
    integrand = cos(2*theta)/(5 - 4*cos(theta))
    assert integrate(integrand, (theta, 0, 2*pi)) == pi/6


@slow
def test_issue_22033_integral():
    assert integrate((x**2 - Rational(1, 4))**2 * sqrt(1 - x**2), (x, -1, 1)) == pi/32


@slow
def test_issue_21671():
    assert integrate(1,(z,x**2+y**2,2-x**2-y**2),(y,-sqrt(1-x**2),sqrt(1-x**2)),(x,-1,1)) == pi
    assert integrate(-4*(1 - x**2)**(S(3)/2)/3 + 2*sqrt(1 - x**2)*(2 - 2*x**2), (x, -1, 1)) == pi


def test_issue_18527():
    # The manual integrator can not currently solve this. Assert that it does
    # not give an incorrect result involving Abs when x has real assumptions.
    xr = symbols('xr', real=True)
    expr = (cos(x)/(4+(sin(x))**2))
    res_real = integrate(expr.subs(x, xr), xr, manual=True).subs(xr, x)
    assert integrate(expr, x, manual=True) == res_real == Integral(expr, x)


def test_issue_23718():
    f = 1/(b*cos(x) + a*sin(x))
    Fpos = (-log(-a/b + tan(x/2) - sqrt(a**2 + b**2)/b)/sqrt(a**2 + b**2)
            +log(-a/b + tan(x/2) + sqrt(a**2 + b**2)/b)/sqrt(a**2 + b**2))
    F = Piecewise(
        # XXX: The zoo case here is for a=b=0 so it should just be zoo or maybe
        # it doesn't really need to be included at all given that the original
        # integrand is really undefined in that case anyway.
        (zoo*(-log(tan(x/2) - 1) + log(tan(x/2) + 1)),  Eq(a, 0) & Eq(b, 0)),
        (log(tan(x/2))/a,                               Eq(b, 0)),
        (-I/(-I*b*sin(x) + b*cos(x)),                   Eq(a, -I*b)),
        (I/(I*b*sin(x) + b*cos(x)),                     Eq(a,  I*b)),
        (Fpos,                                          True),
    )
    assert integrate(f, x) == F

    ap, bp = symbols('a, b', positive=True)
    rep = {a: ap, b: bp}
    assert integrate(f.subs(rep), x) == Fpos.subs(rep)


def test_issue_23566():
    i = integrate(1/sqrt(x**2-1), (x, -2, -1))
    assert i == -log(2 - sqrt(3))
    assert math.isclose(i.n(), 1.31695789692482)


def test_pr_23583():
    # This result from meijerg is wrong. Check whether new result is correct when this test fail.
    assert integrate(1/sqrt((x - I)**2-1)) == Piecewise((acosh(x - I), Abs((x - I)**2) > 1), (-I*asin(x - I), True))


def test_issue_7264():
    assert integrate(exp(x)*sqrt(1 + exp(2*x))) == sqrt(exp(2*x) + 1)*exp(x)/2 + asinh(exp(x))/2


def test_issue_11254a():
    assert integrate(sech(x), (x, 0, 1)) == 2*atan(tanh(S.Half))


def test_issue_11254b():
    assert integrate(csch(x), x) == log(tanh(x/2))
    assert integrate(csch(x), (x, 0, 1)) == oo


def test_issue_11254d():
    # (sech(x)**2).rewrite(sinh)
    assert integrate(-1/sinh(x + I*pi/2, evaluate=False)**2, x) == -2/(exp(2*x) + 1)
    assert integrate(cosh(x)**(-2), x) == 2*tanh(x/2)/(tanh(x/2)**2 + 1)


def test_issue_22863():
    i = integrate((3*x**3-x**2+2*x-4)/sqrt(x**2-3*x+2), (x, 0, 1))
    assert i == -101*sqrt(2)/8 - 135*log(3 - 2*sqrt(2))/16
    assert math.isclose(i.n(), -2.98126694400554)


def test_issue_9723():
    assert integrate(sqrt(x + sqrt(x))) == \
        2*sqrt(sqrt(x) + x)*(sqrt(x)/12 + x/3 - S(1)/8) + log(2*sqrt(x) + 2*sqrt(sqrt(x) + x) + 1)/8
    assert integrate(sqrt(2*x+3+sqrt(4*x+5))**3) == \
        sqrt(2*x + sqrt(4*x + 5) + 3) * \
           (9*x/10 + 11*(4*x + 5)**(S(3)/2)/40 + sqrt(4*x + 5)/40 + (4*x + 5)**2/10 + S(11)/10)/2


def test_issue_23704():
    # XXX: This is testing that an exception is not raised in risch Ideally
    # manualintegrate (manual=True) would be able to compute this but
    # manualintegrate is very slow for this example so we don't test that here.
    assert (integrate(log(x)/x**2/(c*x**2+b*x+a),x, risch=True)
        == NonElementaryIntegral(log(x)/(a*x**2 + b*x**3 + c*x**4), x))


def test_exp_substitution():
    assert integrate(1/sqrt(1-exp(2*x))) == log(sqrt(1 - exp(2*x)) - 1)/2 - log(sqrt(1 - exp(2*x)) + 1)/2


def test_hyperbolic():
    assert integrate(coth(x)) == x - log(tanh(x) + 1) + log(tanh(x))
    assert integrate(sech(x)) == 2*atan(tanh(x/2))
    assert integrate(csch(x)) == log(tanh(x/2))


def test_nested_pow():
    assert integrate(sqrt(x**2)) == x*sqrt(x**2)/2
    assert integrate(sqrt(x**(S(5)/3))) == 6*x*sqrt(x**(S(5)/3))/11
    assert integrate(1/sqrt(x**2)) == x*log(x)/sqrt(x**2)
    assert integrate(x*sqrt(x**(-4))) == x**2*sqrt(x**-4)*log(x)


def test_sqrt_quadratic():
    assert integrate(1/sqrt(3*x**2+4*x+5)) == sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/3
    assert integrate(1/sqrt(-3*x**2+4*x+5)) == sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/3
    assert integrate(1/sqrt(3*x**2+4*x-5)) == sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/3
    assert integrate(1/sqrt(4*x**2-4*x+1)) == (x - S.Half)*log(x - S.Half)/(2*sqrt((x - S.Half)**2))
    assert integrate(1/sqrt(a+b*x+c*x**2), x) == \
        Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(c, 0) & Ne(a - b**2/(4*c), 0)),
                  ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), Ne(c, 0)),
                  (2*sqrt(a + b*x)/b, Ne(b, 0)), (x/sqrt(a), True))

    assert integrate((7*x+6)/sqrt(3*x**2+4*x+5)) == \
           7*sqrt(3*x**2 + 4*x + 5)/3 + 4*sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/9
    assert integrate((7*x+6)/sqrt(-3*x**2+4*x+5)) == \
           -7*sqrt(-3*x**2 + 4*x + 5)/3 + 32*sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/9
    assert integrate((7*x+6)/sqrt(3*x**2+4*x-5)) == \
           7*sqrt(3*x**2 + 4*x - 5)/3 + 4*sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/9
    assert integrate((d+e*x)/sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((-b*e/(2*c) + d) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   e*sqrt(a + b*x + c*x**2)/c, Ne(c, 0)),
                  ((2*d*sqrt(a + b*x) + 2*e*(-a*sqrt(a + b*x) + (a + b*x)**(S(3)/2)/3)/b)/b, Ne(b, 0)),
                  ((d*x + e*x**2/2)/sqrt(a), True))

    assert integrate((3*x**3-x**2+2*x-4)/sqrt(x**2-3*x+2)) == \
           sqrt(x**2 - 3*x + 2)*(x**2 + 13*x/4 + S(101)/8) + 135*log(2*x + 2*sqrt(x**2 - 3*x + 2) - 3)/16

    assert integrate(sqrt(53225*x**2-66732*x+23013)) == \
           (x/2 - S(16683)/53225)*sqrt(53225*x**2 - 66732*x + 23013) + \
           111576969*sqrt(2129)*asinh(53225*x/10563 - S(11122)/3521)/1133160250
    assert integrate(sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((a/2 - b**2/(8*c)) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   (b/(4*c) + x/2)*sqrt(a + b*x + c*x**2), Ne(c, 0)),
                  (2*(a + b*x)**(S(3)/2)/(3*b), Ne(b, 0)),
                  (sqrt(a)*x, True))

    assert integrate(x*sqrt(x**2+2*x+4)) == \
        (x**2/3 + x/6 + S(5)/6)*sqrt(x**2 + 2*x + 4) - 3*asinh(sqrt(3)*(x + 1)/3)/2


def test_mul_pow_derivative():
    assert integrate(x*sec(x)*tan(x)) == x*sec(x) - log(tan(x) + sec(x))
    assert integrate(x*sec(x)**2, x) == x*tan(x) + log(cos(x))
    assert integrate(x**3*Derivative(f(x), (x, 4))) == \
           x**3*Derivative(f(x), (x, 3)) - 3*x**2*Derivative(f(x), (x, 2)) + 6*x*Derivative(f(x), x) - 6*f(x)


def test_issue_20782():
    fun1 = Piecewise((0, x < 0.0), (1, True))
    fun2 = -Piecewise((0, x < 1.0), (1, True))
    fun_sum = fun1 + fun2
    L = (x, -float('Inf'), 1)

    assert integrate(fun1, L) == 1
    assert integrate(fun2, L) == 0
    assert integrate(-fun1, L) == -1
    assert integrate(-fun2, L) == 0.
    assert integrate(fun_sum, L) == 1.
    assert integrate(-fun_sum, L) == -1.


def test_issue_20781():
    P = lambda a: Piecewise((0, x < a), (1, x >= a))
    f = lambda a: P(int(a)) + P(float(a))
    L = (x, -float('Inf'), x)
    f1 = integrate(f(1), L)
    assert f1 == 2*x - Min(1.0, x) - Min(x, Max(1.0, 1, evaluate=False))
    # XXX is_zero is True for S(0) and Float(0) and this is baked into
    # the code more deeply than the issue of Float(0) != S(0)
    assert integrate(f(0), (x, -float('Inf'), x)
        ) == 2*x - 2*Min(0, x)


@slow
def test_issue_19427():
    # <https://github.com/sympy/sympy/issues/19427>
    x = Symbol("x")

    # Have always been okay:
    assert integrate((x ** 4) * sqrt(1 - x ** 2), (x, -1, 1)) == pi / 16
    assert integrate((-2 * x ** 2) * sqrt(1 - x ** 2), (x, -1, 1)) == -pi / 4
    assert integrate((1) * sqrt(1 - x ** 2), (x, -1, 1)) == pi / 2

    # Sum of the above, used to incorrectly return 0 for a while:
    assert integrate((x ** 4 - 2 * x ** 2 + 1) * sqrt(1 - x ** 2), (x, -1, 1)) == 5 * pi / 16


def test_issue_23942():
    I1 = Integral(1/sqrt(a*(1 + x)**3 + (1 + x)**2), (x, 0, z))
    assert I1.series(a, 1, n=1) == Integral(1/sqrt(x**3 + 4*x**2 + 5*x + 2), (x, 0, z)) + O(a - 1, (a, 1))
    I2 = Integral(1/sqrt(a*(4 - x)**4 + (5 + x)**2), (x, 0, z))
    assert I2.series(a, 2, n=1) == Integral(1/sqrt(2*x**4 - 32*x**3 + 193*x**2 - 502*x + 537), (x, 0, z)) + O(a - 2, (a, 2))


def test_issue_25886():
    # https://github.com/sympy/sympy/issues/25886
    f = (1-x)*exp(0.937098661j*x)
    F_exp = (1.0*(-1.0671234968289*I*y
             + 1.13875255748434
             + 1.0671234968289*I)*exp(0.937098661*I*y)
            - 1.13875255748434*exp(0.937098661*I))
    F = integrate(f, (x, y, 1.0))
    assert F.is_same(F_exp, math.isclose)


def test_old_issues():
    # https://github.com/sympy/sympy/issues/5212
    I1 = integrate(cos(log(x**2))/x)
    assert I1 == sin(log(x**2))/2
    # https://github.com/sympy/sympy/issues/5462
    I2 = integrate(1/(x**2+y**2)**(Rational(3,2)),x)
    assert I2 == x/(y**3*sqrt(x**2/y**2 + 1))
    # https://github.com/sympy/sympy/issues/6278
    I3 = integrate(1/(cos(x)+2),(x,0,2*pi))
    assert I3 == 2*sqrt(3)*pi/3
