from sympy.core.evalf import N
from sympy.core.function import (Derivative, Function, PoleError, Subs)
from sympy.core.numbers import (E, Float, Rational, oo, pi, I)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral, integrate
from sympy.series.order import O
from sympy.series.series import series
from sympy.abc import x, y, n, k
from sympy.testing.pytest import raises
from sympy.core import EulerGamma


def test_sin():
    e1 = sin(x).series(x, 0)
    e2 = series(sin(x), x, 0)
    assert e1 == e2


def test_cos():
    e1 = cos(x).series(x, 0)
    e2 = series(cos(x), x, 0)
    assert e1 == e2


def test_exp():
    e1 = exp(x).series(x, 0)
    e2 = series(exp(x), x, 0)
    assert e1 == e2


def test_exp2():
    e1 = exp(cos(x)).series(x, 0)
    e2 = series(exp(cos(x)), x, 0)
    assert e1 == e2


def test_issue_5223():
    assert series(1, x) == 1
    assert next(S.Zero.lseries(x)) == 0
    assert cos(x).series() == cos(x).series(x)
    raises(ValueError, lambda: cos(x + y).series())
    raises(ValueError, lambda: x.series(dir=""))

    assert (cos(x).series(x, 1) -
            cos(x + 1).series(x).subs(x, x - 1)).removeO() == 0
    e = cos(x).series(x, 1, n=None)
    assert [next(e) for i in range(2)] == [cos(1), -((x - 1)*sin(1))]
    e = cos(x).series(x, 1, n=None, dir='-')
    assert [next(e) for i in range(2)] == [cos(1), (1 - x)*sin(1)]
    # the following test is exact so no need for x -> x - 1 replacement
    assert abs(x).series(x, 1, dir='-') == x
    assert exp(x).series(x, 1, dir='-', n=3).removeO() == \
        E - E*(-x + 1) + E*(-x + 1)**2/2

    D = Derivative
    assert D(x**2 + x**3*y**2, x, 2, y, 1).series(x).doit() == 12*x*y
    assert next(D(cos(x), x).lseries()) == D(1, x)
    assert D(
        exp(x), x).series(n=3) == D(1, x) + D(x, x) + D(x**2/2, x) + D(x**3/6, x) + O(x**3)

    assert Integral(x, (x, 1, 3), (y, 1, x)).series(x) == -4 + 4*x

    assert (1 + x + O(x**2)).getn() == 2
    assert (1 + x).getn() is None

    raises(PoleError, lambda: ((1/sin(x))**oo).series())
    logx = Symbol('logx')
    assert ((sin(x))**y).nseries(x, n=1, logx=logx) == \
        exp(y*logx) + O(x*exp(y*logx), x)

    assert sin(1/x).series(x, oo, n=5) == 1/x - 1/(6*x**3) + O(x**(-5), (x, oo))
    assert abs(x).series(x, oo, n=5, dir='+') == x
    assert abs(x).series(x, -oo, n=5, dir='-') == -x
    assert abs(-x).series(x, oo, n=5, dir='+') == x
    assert abs(-x).series(x, -oo, n=5, dir='-') == -x

    assert exp(x*log(x)).series(n=3) == \
        1 + x*log(x) + x**2*log(x)**2/2 + O(x**3*log(x)**3)
    # XXX is this right? If not, fix "ngot > n" handling in expr.
    p = Symbol('p', positive=True)
    assert exp(sqrt(p)**3*log(p)).series(n=3) == \
        1 + p**S('3/2')*log(p) + O(p**3*log(p)**3)

    assert exp(sin(x)*log(x)).series(n=2) == 1 + x*log(x) + O(x**2*log(x)**2)


def test_issue_6350():
    expr = integrate(exp(k*(y**3 - 3*y)), (y, 0, oo), conds='none')
    assert expr.series(k, 0, 3) == -(-1)**(S(2)/3)*sqrt(3)*gamma(S(1)/3)**2*gamma(S(2)/3)/(6*pi*k**(S(1)/3)) - \
        sqrt(3)*k*gamma(-S(2)/3)*gamma(-S(1)/3)/(6*pi) - \
        (-1)**(S(1)/3)*sqrt(3)*k**(S(1)/3)*gamma(-S(1)/3)*gamma(S(1)/3)*gamma(S(2)/3)/(6*pi) - \
        (-1)**(S(2)/3)*sqrt(3)*k**(S(5)/3)*gamma(S(1)/3)**2*gamma(S(2)/3)/(4*pi) - \
        (-1)**(S(1)/3)*sqrt(3)*k**(S(7)/3)*gamma(-S(1)/3)*gamma(S(1)/3)*gamma(S(2)/3)/(8*pi) + O(k**3)


def test_issue_11313():
    assert Integral(cos(x), x).series(x) == sin(x).series(x)
    assert Derivative(sin(x), x).series(x, n=3).doit() == cos(x).series(x, n=3)

    assert Derivative(x**3, x).as_leading_term(x) == 3*x**2
    assert Derivative(x**3, y).as_leading_term(x) == 0
    assert Derivative(sin(x), x).as_leading_term(x) == 1
    assert Derivative(cos(x), x).as_leading_term(x) == -x

    # This result is equivalent to zero, zero is not return because
    # `Expr.series` doesn't currently detect an `x` in its `free_symbol`s.
    assert Derivative(1, x).as_leading_term(x) == Derivative(1, x)

    assert Derivative(exp(x), x).series(x).doit() == exp(x).series(x)
    assert 1 + Integral(exp(x), x).series(x) == exp(x).series(x)

    assert Derivative(log(x), x).series(x).doit() == (1/x).series(x)
    assert Integral(log(x), x).series(x) == Integral(log(x), x).doit().series(x).removeO()


def test_series_of_Subs():
    from sympy.abc import z

    subs1 = Subs(sin(x), x, y)
    subs2 = Subs(sin(x) * cos(z), x, y)
    subs3 = Subs(sin(x * z), (x, z), (y, x))

    assert subs1.series(x) == subs1
    subs1_series = (Subs(x, x, y) + Subs(-x**3/6, x, y) +
        Subs(x**5/120, x, y) + O(y**6))
    assert subs1.series() == subs1_series
    assert subs1.series(y) == subs1_series
    assert subs1.series(z) == subs1
    assert subs2.series(z) == (Subs(z**4*sin(x)/24, x, y) +
        Subs(-z**2*sin(x)/2, x, y) + Subs(sin(x), x, y) + O(z**6))
    assert subs3.series(x).doit() == subs3.doit().series(x)
    assert subs3.series(z).doit() == sin(x*y)

    raises(ValueError, lambda: Subs(x + 2*y, y, z).series())
    assert Subs(x + y, y, z).series(x).doit() == x + z


def test_issue_3978():
    f = Function('f')
    assert f(x).series(x, 0, 3, dir='-') == \
            f(0) + x*Subs(Derivative(f(x), x), x, 0) + \
            x**2*Subs(Derivative(f(x), x, x), x, 0)/2 + O(x**3)
    assert f(x).series(x, 0, 3) == \
            f(0) + x*Subs(Derivative(f(x), x), x, 0) + \
            x**2*Subs(Derivative(f(x), x, x), x, 0)/2 + O(x**3)
    assert f(x**2).series(x, 0, 3) == \
            f(0) + x**2*Subs(Derivative(f(x), x), x, 0) + O(x**3)
    assert f(x**2+1).series(x, 0, 3) == \
            f(1) + x**2*Subs(Derivative(f(x), x), x, 1) + O(x**3)

    class TestF(Function):
        pass

    assert TestF(x).series(x, 0, 3) ==  TestF(0) + \
            x*Subs(Derivative(TestF(x), x), x, 0) + \
            x**2*Subs(Derivative(TestF(x), x, x), x, 0)/2 + O(x**3)

from sympy.series.acceleration import richardson, shanks
from sympy.concrete.summations import Sum
from sympy.core.numbers import Integer


def test_acceleration():
    e = (1 + 1/n)**n
    assert round(richardson(e, n, 10, 20).evalf(), 10) == round(E.evalf(), 10)

    A = Sum(Integer(-1)**(k + 1) / k, (k, 1, n))
    assert round(shanks(A, n, 25).evalf(), 4) == round(log(2).evalf(), 4)
    assert round(shanks(A, n, 25, 5).evalf(), 10) == round(log(2).evalf(), 10)


def test_issue_5852():
    assert series(1/cos(x/log(x)), x, 0) == 1 + x**2/(2*log(x)**2) + \
        5*x**4/(24*log(x)**4) + O(x**6)


def test_issue_4583():
    assert cos(1 + x + x**2).series(x, 0, 5) == cos(1) - x*sin(1) + \
        x**2*(-sin(1) - cos(1)/2) + x**3*(-cos(1) + sin(1)/6) + \
        x**4*(-11*cos(1)/24 + sin(1)/2) + O(x**5)


def test_issue_6318():
    eq = (1/x)**Rational(2, 3)
    assert (eq + 1).as_leading_term(x) == eq


def test_x_is_base_detection():
    eq = (x**2)**Rational(2, 3)
    assert eq.series() == x**Rational(4, 3)


def test_issue_7203():
    assert series(cos(x), x, pi, 3) == \
        -1 + (x - pi)**2/2 + O((x - pi)**3, (x, pi))


def test_exp_product_positive_factors():
    a, b = symbols('a, b', positive=True)
    x = a * b
    assert series(exp(x), x, n=8) == 1 + a*b + a**2*b**2/2 + \
        a**3*b**3/6 + a**4*b**4/24 + a**5*b**5/120 + a**6*b**6/720 + \
        a**7*b**7/5040 + O(a**8*b**8, a, b)


def test_issue_8805():
    assert series(1, n=8) == 1


def test_issue_9173():
    p0,p1,p2,p3,b0,b1,b2=symbols('p0 p1 p2 p3 b0 b1 b2')
    Q=(p0+(p1+(p2+p3/y)/y)/y)/(1+((p3/(b0*y)+(b0*p2-b1*p3)/b0**2)/y+\
       (b0**2*p1-b0*b1*p2-p3*(b0*b2-b1**2))/b0**3)/y)

    series = Q.series(y,n=3)

    assert series == y*(b0*p2/p3+b0*(-p2/p3+b1/b0))+y**2*(b0*p1/p3+b0*p2*\
            (-p2/p3+b1/b0)/p3+b0*(-p1/p3+(p2/p3-b1/b0)**2+b1*p2/(b0*p3)+\
            b2/b0-b1**2/b0**2))+b0+O(y**3)
    assert series.simplify() == b2*y**2 + b1*y + b0 + O(y**3)


def test_issue_9549():
    y = (x**2 + x + 1) / (x**3 + x**2)
    assert series(y, x, oo) == x**(-5) - 1/x**4 + x**(-3) + 1/x + O(x**(-6), (x, oo))


def test_issue_10761():
    assert series(1/(x**-2 + x**-3), x, 0) == x**3 - x**4 + x**5 + O(x**6)


def test_issue_12578():
    y = (1 - 1/(x/2 - 1/(2*x))**4)**(S(1)/8)
    assert y.series(x, 0, n=17) == 1 - 2*x**4 - 8*x**6 - 34*x**8 - 152*x**10 - 714*x**12 - \
        3472*x**14 - 17318*x**16 + O(x**17)


def test_issue_12791():
    beta = symbols('beta', positive=True)
    theta, varphi = symbols('theta varphi', real=True)

    expr = (-beta**2*varphi*sin(theta) + beta**2*cos(theta) + \
        beta*varphi*sin(theta) - beta*cos(theta) - beta + 1)/(beta*cos(theta) - 1)**2

    sol = (0.5/(0.5*cos(theta) - 1.0)**2 - 0.25*cos(theta)/(0.5*cos(theta) - 1.0)**2
        + (beta - 0.5)*(-0.25*varphi*sin(2*theta) - 1.5*cos(theta)
        + 0.25*cos(2*theta) + 1.25)/((0.5*cos(theta) - 1.0)**2*(0.5*cos(theta) - 1.0))
        + 0.25*varphi*sin(theta)/(0.5*cos(theta) - 1.0)**2
        + O((beta - S.Half)**2, (beta, S.Half)))

    assert expr.series(beta, 0.5, 2).trigsimp() == sol


def test_issue_14384():
    x, a = symbols('x a')
    assert series(x**a, x) == x**a
    assert series(x**(-2*a), x) == x**(-2*a)
    assert series(exp(a*log(x)), x) == exp(a*log(x))
    raises(PoleError, lambda: series(x**I, x))
    raises(PoleError, lambda: series(x**(I + 1), x))
    raises(PoleError, lambda: series(exp(I*log(x)), x))


def test_issue_14885():
    assert series(x**Rational(-3, 2)*exp(x), x, 0) == (x**Rational(-3, 2) + 1/sqrt(x) +
        sqrt(x)/2 + x**Rational(3, 2)/6 + x**Rational(5, 2)/24 + x**Rational(7, 2)/120 +
        x**Rational(9, 2)/720 + x**Rational(11, 2)/5040 + O(x**6))


def test_issue_15539():
    assert series(atan(x), x, -oo) == (-1/(5*x**5) + 1/(3*x**3) - 1/x - pi/2
        + O(x**(-6), (x, -oo)))
    assert series(atan(x), x, oo) == (-1/(5*x**5) + 1/(3*x**3) - 1/x + pi/2
        + O(x**(-6), (x, oo)))


def test_issue_7259():
    assert series(LambertW(x), x) == x - x**2 + 3*x**3/2 - 8*x**4/3 + 125*x**5/24 + O(x**6)
    assert series(LambertW(x**2), x, n=8) == x**2 - x**4 + 3*x**6/2 + O(x**8)
    assert series(LambertW(sin(x)), x, n=4) == x - x**2 + 4*x**3/3 + O(x**4)

def test_issue_11884():
    assert cos(x).series(x, 1, n=1) == cos(1) + O(x - 1, (x, 1))


def test_issue_18008():
    y = x*(1 + x*(1 - x))/((1 + x*(1 - x)) - (1 - x)*(1 - x))
    assert y.series(x, oo, n=4) == -9/(32*x**3) - 3/(16*x**2) - 1/(8*x) + S(1)/4 + x/2 + \
        O(x**(-4), (x, oo))


def test_issue_18842():
    f = log(x/(1 - x))
    assert f.series(x, 0.491, n=1).removeO().nsimplify() ==  \
        -S(180019443780011)/5000000000000000


def test_issue_19534():
    dt = symbols('dt', real=True)
    expr = 16*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0)/45 + \
            49*dt*(-0.049335189898860408029*dt*(2.0*dt + 1.0) + \
            0.29601113939316244817*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) - \
            0.12564355335492979587*dt*(0.074074074074074074074*dt*(2.0*dt + 1.0) + \
            0.2962962962962962963*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) + \
            0.96296296296296296296*dt + 1.0) + 0.051640768506639183825*dt + \
            dt*(1/2 - sqrt(21)/14) + 1.0)/180 + 49*dt*(-0.23637909581542530626*dt*(2.0*dt + 1.0) - \
            0.74817562366625959291*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) + \
            0.88085458023927036857*dt*(0.074074074074074074074*dt*(2.0*dt + 1.0) + \
            0.2962962962962962963*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) + \
            0.96296296296296296296*dt + 1.0) + \
            2.1165151389911680013*dt*(-0.049335189898860408029*dt*(2.0*dt + 1.0) + \
            0.29601113939316244817*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) - \
            0.12564355335492979587*dt*(0.074074074074074074074*dt*(2.0*dt + 1.0) + \
            0.2962962962962962963*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) + \
            0.96296296296296296296*dt + 1.0) + 0.22431393315265061193*dt + 1.0) - \
            1.1854881643947648988*dt + dt*(sqrt(21)/14 + 1/2) + 1.0)/180 + \
            dt*(0.66666666666666666667*dt*(2.0*dt + 1.0) + \
            6.0173399699313066769*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) - \
            4.1117044797036320069*dt*(0.074074074074074074074*dt*(2.0*dt + 1.0) + \
            0.2962962962962962963*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) + \
            0.96296296296296296296*dt + 1.0) - \
            7.0189140975801991157*dt*(-0.049335189898860408029*dt*(2.0*dt + 1.0) + \
            0.29601113939316244817*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) - \
            0.12564355335492979587*dt*(0.074074074074074074074*dt*(2.0*dt + 1.0) + \
            0.2962962962962962963*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) + \
            0.96296296296296296296*dt + 1.0) + 0.22431393315265061193*dt + 1.0) + \
            0.94010945196161777522*dt*(-0.23637909581542530626*dt*(2.0*dt + 1.0) - \
            0.74817562366625959291*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) + \
            0.88085458023927036857*dt*(0.074074074074074074074*dt*(2.0*dt + 1.0) + \
            0.2962962962962962963*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) + \
            0.96296296296296296296*dt + 1.0) + \
            2.1165151389911680013*dt*(-0.049335189898860408029*dt*(2.0*dt + 1.0) + \
            0.29601113939316244817*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) - \
            0.12564355335492979587*dt*(0.074074074074074074074*dt*(2.0*dt + 1.0) + \
            0.2962962962962962963*dt*(0.125*dt*(2.0*dt + 1.0) + 0.875*dt + 1.0) + \
            0.96296296296296296296*dt + 1.0) + 0.22431393315265061193*dt + 1.0) - \
            0.35816132904077632692*dt + 1.0) + 5.5065024887242400038*dt + 1.0)/20 + dt/20 + 1

    assert N(expr.series(dt, 0, 8), 20) == (
            - Float('0.00092592592592592596126289', precision=70) * dt**7
            + Float('0.0027777777777777783174695', precision=70) * dt**6
            + Float('0.016666666666666656027029', precision=70) * dt**5
            + Float('0.083333333333333300951828', precision=70) * dt**4
            + Float('0.33333333333333337034077', precision=70) * dt**3
            + Float('1.0', precision=70) * dt**2
            + Float('1.0', precision=70) * dt
            + Float('1.0', precision=70)
        )


def test_issue_11407():
    a, b, c, x = symbols('a b c x')
    assert series(sqrt(a + b + c*x), x, 0, 1) == sqrt(a + b) + O(x)
    assert series(sqrt(a + b + c + c*x), x, 0, 1) == sqrt(a + b + c) + O(x)


def test_issue_14037():
    assert (sin(x**50)/x**51).series(x, n=0) == 1/x + O(1, x)


def test_issue_20551():
    expr = (exp(x)/x).series(x, n=None)
    terms = [ next(expr) for i in range(3) ]
    assert terms == [1/x, 1, x/2]


def test_issue_20697():
    p_0, p_1, p_2, p_3, b_0, b_1, b_2 = symbols('p_0 p_1 p_2 p_3 b_0 b_1 b_2')
    Q = (p_0 + (p_1 + (p_2 + p_3/y)/y)/y)/(1 + ((p_3/(b_0*y) + (b_0*p_2\
        - b_1*p_3)/b_0**2)/y + (b_0**2*p_1 - b_0*b_1*p_2 - p_3*(b_0*b_2\
        - b_1**2))/b_0**3)/y)
    assert Q.series(y, n=3).ratsimp() == b_2*y**2 + b_1*y + b_0 + O(y**3)


def test_issue_21245():
    fi = (1 + sqrt(5))/2
    assert (1/(1 - x - x**2)).series(x, 1/fi, 1).factor() == \
        (-37*sqrt(5) - 83 + 13*sqrt(5)*x + 29*x + O((x - 2/(1 + sqrt(5)))**2, (x\
            , 2/(1 + sqrt(5)))))/((2*sqrt(5) + 5)**2*(x + sqrt(5)*x - 2))



def test_issue_21938():
    expr = sin(1/x + exp(-x)) - sin(1/x)
    assert expr.series(x, oo) == (1/(24*x**4) - 1/(2*x**2) + 1 + O(x**(-6), (x, oo)))*exp(-x)


def test_issue_23432():
    expr = 1/sqrt(1 - x**2)
    result = expr.series(x, 0.5)
    assert result.is_Add and len(result.args) == 7


def test_issue_23727():
    res = series(sqrt(1 - x**2), x, 0.1)
    assert res.is_Add == True


def test_issue_24266():
    #type1: exp(f(x))
    assert (exp(-I*pi*(2*x+1))).series(x, 0, 3) == -1 + 2*I*pi*x + 2*pi**2*x**2 + O(x**3)
    assert (exp(-I*pi*(2*x+1))*gamma(1+x)).series(x, 0, 3) == -1 + x*(EulerGamma + 2*I*pi) + \
        x**2*(-EulerGamma**2/2 + 23*pi**2/12 - 2*EulerGamma*I*pi) + O(x**3)

    #type2: c**f(x)
    assert ((2*I)**(-I*pi*(2*x+1))).series(x, 0, 2) == exp(pi**2/2 - I*pi*log(2)) + \
          x*(pi**2*exp(pi**2/2 - I*pi*log(2)) - 2*I*pi*exp(pi**2/2 - I*pi*log(2))*log(2)) + O(x**2)
    assert ((2)**(-I*pi*(2*x+1))).series(x, 0, 2) == exp(-I*pi*log(2)) - 2*I*pi*x*exp(-I*pi*log(2))*log(2) + O(x**2)

    #type3: f(y)**g(x)
    assert ((y)**(I*pi*(2*x+1))).series(x, 0, 2) == exp(I*pi*log(y)) + 2*I*pi*x*exp(I*pi*log(y))*log(y) + O(x**2)
    assert ((I*y)**(I*pi*(2*x+1))).series(x, 0, 2) == exp(I*pi*log(I*y)) + 2*I*pi*x*exp(I*pi*log(I*y))*log(I*y) + O(x**2)


def test_issue_26856():
    raises(ValueError, lambda: (2**x).series(x, oo, -1))
