from itertools import product
from sympy.core.function import (Subs, count_ops, diff, expand)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (asec, acsc)
from sympy.functions.elementary.trigonometric import (acot, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import (exptrigsimp, trigsimp)

from sympy.testing.pytest import XFAIL

from sympy.abc import x, y



def test_trigsimp1():
    x, y = symbols('x,y')

    assert trigsimp(1 - sin(x)**2) == cos(x)**2
    assert trigsimp(1 - cos(x)**2) == sin(x)**2
    assert trigsimp(sin(x)**2 + cos(x)**2) == 1
    assert trigsimp(1 + tan(x)**2) == 1/cos(x)**2
    assert trigsimp(1/cos(x)**2 - 1) == tan(x)**2
    assert trigsimp(1/cos(x)**2 - tan(x)**2) == 1
    assert trigsimp(1 + cot(x)**2) == 1/sin(x)**2
    assert trigsimp(1/sin(x)**2 - 1) == 1/tan(x)**2
    assert trigsimp(1/sin(x)**2 - cot(x)**2) == 1

    assert trigsimp(5*cos(x)**2 + 5*sin(x)**2) == 5
    assert trigsimp(5*cos(x/2)**2 + 2*sin(x/2)**2) == 3*cos(x)/2 + Rational(7, 2)

    assert trigsimp(sin(x)/cos(x)) == tan(x)
    assert trigsimp(2*tan(x)*cos(x)) == 2*sin(x)
    assert trigsimp(cot(x)**3*sin(x)**3) == cos(x)**3
    assert trigsimp(y*tan(x)**2/sin(x)**2) == y/cos(x)**2
    assert trigsimp(cot(x)/cos(x)) == 1/sin(x)

    assert trigsimp(sin(x + y) + sin(x - y)) == 2*sin(x)*cos(y)
    assert trigsimp(sin(x + y) - sin(x - y)) == 2*sin(y)*cos(x)
    assert trigsimp(cos(x + y) + cos(x - y)) == 2*cos(x)*cos(y)
    assert trigsimp(cos(x + y) - cos(x - y)) == -2*sin(x)*sin(y)
    assert trigsimp(tan(x + y) - tan(x)/(1 - tan(x)*tan(y))) == \
        sin(y)/(-sin(y)*tan(x) + cos(y))  # -tan(y)/(tan(x)*tan(y) - 1)

    assert trigsimp(sinh(x + y) + sinh(x - y)) == 2*sinh(x)*cosh(y)
    assert trigsimp(sinh(x + y) - sinh(x - y)) == 2*sinh(y)*cosh(x)
    assert trigsimp(cosh(x + y) + cosh(x - y)) == 2*cosh(x)*cosh(y)
    assert trigsimp(cosh(x + y) - cosh(x - y)) == 2*sinh(x)*sinh(y)
    assert trigsimp(tanh(x + y) - tanh(x)/(1 + tanh(x)*tanh(y))) == \
        sinh(y)/(sinh(y)*tanh(x) + cosh(y))

    assert trigsimp(cos(0.12345)**2 + sin(0.12345)**2) == 1.0
    e = 2*sin(x)**2 + 2*cos(x)**2
    assert trigsimp(log(e)) == log(2)


def test_trigsimp1a():
    assert trigsimp(sin(2)**2*cos(3)*exp(2)/cos(2)**2) == tan(2)**2*cos(3)*exp(2)
    assert trigsimp(tan(2)**2*cos(3)*exp(2)*cos(2)**2) == sin(2)**2*cos(3)*exp(2)
    assert trigsimp(cot(2)*cos(3)*exp(2)*sin(2)) == cos(3)*exp(2)*cos(2)
    assert trigsimp(tan(2)*cos(3)*exp(2)/sin(2)) == cos(3)*exp(2)/cos(2)
    assert trigsimp(cot(2)*cos(3)*exp(2)/cos(2)) == cos(3)*exp(2)/sin(2)
    assert trigsimp(cot(2)*cos(3)*exp(2)*tan(2)) == cos(3)*exp(2)
    assert trigsimp(sinh(2)*cos(3)*exp(2)/cosh(2)) == tanh(2)*cos(3)*exp(2)
    assert trigsimp(tanh(2)*cos(3)*exp(2)*cosh(2)) == sinh(2)*cos(3)*exp(2)
    assert trigsimp(coth(2)*cos(3)*exp(2)*sinh(2)) == cosh(2)*cos(3)*exp(2)
    assert trigsimp(tanh(2)*cos(3)*exp(2)/sinh(2)) == cos(3)*exp(2)/cosh(2)
    assert trigsimp(coth(2)*cos(3)*exp(2)/cosh(2)) == cos(3)*exp(2)/sinh(2)
    assert trigsimp(coth(2)*cos(3)*exp(2)*tanh(2)) == cos(3)*exp(2)


def test_trigsimp2():
    x, y = symbols('x,y')
    assert trigsimp(cos(x)**2*sin(y)**2 + cos(x)**2*cos(y)**2 + sin(x)**2,
            recursive=True) == 1
    assert trigsimp(sin(x)**2*sin(y)**2 + sin(x)**2*cos(y)**2 + cos(x)**2,
            recursive=True) == 1
    assert trigsimp(
        Subs(x, x, sin(y)**2 + cos(y)**2)) == Subs(x, x, 1)


def test_issue_4373():
    x = Symbol("x")
    assert abs(trigsimp(2.0*sin(x)**2 + 2.0*cos(x)**2) - 2.0) < 1e-10


def test_trigsimp3():
    x, y = symbols('x,y')
    assert trigsimp(sin(x)/cos(x)) == tan(x)
    assert trigsimp(sin(x)**2/cos(x)**2) == tan(x)**2
    assert trigsimp(sin(x)**3/cos(x)**3) == tan(x)**3
    assert trigsimp(sin(x)**10/cos(x)**10) == tan(x)**10

    assert trigsimp(cos(x)/sin(x)) == 1/tan(x)
    assert trigsimp(cos(x)**2/sin(x)**2) == 1/tan(x)**2
    assert trigsimp(cos(x)**10/sin(x)**10) == 1/tan(x)**10

    assert trigsimp(tan(x)) == trigsimp(sin(x)/cos(x))


def test_issue_4661():
    a, x, y = symbols('a x y')
    eq = -4*sin(x)**4 + 4*cos(x)**4 - 8*cos(x)**2
    assert trigsimp(eq) == -4
    n = sin(x)**6 + 4*sin(x)**4*cos(x)**2 + 5*sin(x)**2*cos(x)**4 + 2*cos(x)**6
    d = -sin(x)**2 - 2*cos(x)**2
    assert simplify(n/d) == -1
    assert trigsimp(-2*cos(x)**2 + cos(x)**4 - sin(x)**4) == -1
    eq = (- sin(x)**3/4)*cos(x) + (cos(x)**3/4)*sin(x) - sin(2*x)*cos(2*x)/8
    assert trigsimp(eq) == 0


def test_issue_4494():
    a, b = symbols('a b')
    eq = sin(a)**2*sin(b)**2 + cos(a)**2*cos(b)**2*tan(a)**2 + cos(a)**2
    assert trigsimp(eq) == 1


def test_issue_5948():
    a, x, y = symbols('a x y')
    assert trigsimp(diff(integrate(cos(x)/sin(x)**7, x), x)) == \
           cos(x)/sin(x)**7


def test_issue_4775():
    a, x, y = symbols('a x y')
    assert trigsimp(sin(x)*cos(y)+cos(x)*sin(y)) == sin(x + y)
    assert trigsimp(sin(x)*cos(y)+cos(x)*sin(y)+3) == sin(x + y) + 3


def test_issue_4280():
    a, x, y = symbols('a x y')
    assert trigsimp(cos(x)**2 + cos(y)**2*sin(x)**2 + sin(y)**2*sin(x)**2) == 1
    assert trigsimp(a**2*sin(x)**2 + a**2*cos(y)**2*cos(x)**2 + a**2*cos(x)**2*sin(y)**2) == a**2
    assert trigsimp(a**2*cos(y)**2*sin(x)**2 + a**2*sin(y)**2*sin(x)**2) == a**2*sin(x)**2


def test_issue_3210():
    eqs = (sin(2)*cos(3) + sin(3)*cos(2),
        -sin(2)*sin(3) + cos(2)*cos(3),
        sin(2)*cos(3) - sin(3)*cos(2),
        sin(2)*sin(3) + cos(2)*cos(3),
        sin(2)*sin(3) + cos(2)*cos(3) + cos(2),
        sinh(2)*cosh(3) + sinh(3)*cosh(2),
        sinh(2)*sinh(3) + cosh(2)*cosh(3),
        )
    assert [trigsimp(e) for e in eqs] == [
        sin(5),
        cos(5),
        -sin(1),
        cos(1),
        cos(1) + cos(2),
        sinh(5),
        cosh(5),
        ]


def test_trigsimp_issues():
    a, x, y = symbols('a x y')

    # issue 4625 - factor_terms works, too
    assert trigsimp(sin(x)**3 + cos(x)**2*sin(x)) == sin(x)

    # issue 5948
    assert trigsimp(diff(integrate(cos(x)/sin(x)**3, x), x)) == \
        cos(x)/sin(x)**3
    assert trigsimp(diff(integrate(sin(x)/cos(x)**3, x), x)) == \
        sin(x)/cos(x)**3

    # check integer exponents
    e = sin(x)**y/cos(x)**y
    assert trigsimp(e) == e
    assert trigsimp(e.subs(y, 2)) == tan(x)**2
    assert trigsimp(e.subs(x, 1)) == tan(1)**y

    # check for multiple patterns
    assert (cos(x)**2/sin(x)**2*cos(y)**2/sin(y)**2).trigsimp() == \
        1/tan(x)**2/tan(y)**2
    assert trigsimp(cos(x)/sin(x)*cos(x+y)/sin(x+y)) == \
        1/(tan(x)*tan(x + y))

    eq = cos(2)*(cos(3) + 1)**2/(cos(3) - 1)**2
    assert trigsimp(eq) == eq.factor()  # factor makes denom (-1 + cos(3))**2
    assert trigsimp(cos(2)*(cos(3) + 1)**2*(cos(3) - 1)**2) == \
        cos(2)*sin(3)**4

    # issue 6789; this generates an expression that formerly caused
    # trigsimp to hang
    assert cot(x).equals(tan(x)) is False

    # nan or the unchanged expression is ok, but not sin(1)
    z = cos(x)**2 + sin(x)**2 - 1
    z1 = tan(x)**2 - 1/cot(x)**2
    n = (1 + z1/z)
    assert trigsimp(sin(n)) != sin(1)
    eq = x*(n - 1) - x*n
    assert trigsimp(eq) is S.NaN
    assert trigsimp(eq, recursive=True) is S.NaN
    assert trigsimp(1).is_Integer

    assert trigsimp(-sin(x)**4 - 2*sin(x)**2*cos(x)**2 - cos(x)**4) == -1


def test_trigsimp_issue_2515():
    x = Symbol('x')
    assert trigsimp(x*cos(x)*tan(x)) == x*sin(x)
    assert trigsimp(-sin(x) + cos(x)*tan(x)) == 0


def test_trigsimp_issue_3826():
    assert trigsimp(tan(2*x).expand(trig=True)) == tan(2*x)


def test_trigsimp_issue_4032():
    n = Symbol('n', integer=True, positive=True)
    assert trigsimp(2**(n/2)*cos(pi*n/4)/2 + 2**(n - 1)/2) == \
        2**(n/2)*cos(pi*n/4)/2 + 2**n/4


def test_trigsimp_issue_7761():
    assert trigsimp(cosh(pi/4)) == cosh(pi/4)


def test_trigsimp_noncommutative():
    x, y = symbols('x,y')
    A, B = symbols('A,B', commutative=False)

    assert trigsimp(A - A*sin(x)**2) == A*cos(x)**2
    assert trigsimp(A - A*cos(x)**2) == A*sin(x)**2
    assert trigsimp(A*sin(x)**2 + A*cos(x)**2) == A
    assert trigsimp(A + A*tan(x)**2) == A/cos(x)**2
    assert trigsimp(A/cos(x)**2 - A) == A*tan(x)**2
    assert trigsimp(A/cos(x)**2 - A*tan(x)**2) == A
    assert trigsimp(A + A*cot(x)**2) == A/sin(x)**2
    assert trigsimp(A/sin(x)**2 - A) == A/tan(x)**2
    assert trigsimp(A/sin(x)**2 - A*cot(x)**2) == A

    assert trigsimp(y*A*cos(x)**2 + y*A*sin(x)**2) == y*A

    assert trigsimp(A*sin(x)/cos(x)) == A*tan(x)
    assert trigsimp(A*tan(x)*cos(x)) == A*sin(x)
    assert trigsimp(A*cot(x)**3*sin(x)**3) == A*cos(x)**3
    assert trigsimp(y*A*tan(x)**2/sin(x)**2) == y*A/cos(x)**2
    assert trigsimp(A*cot(x)/cos(x)) == A/sin(x)

    assert trigsimp(A*sin(x + y) + A*sin(x - y)) == 2*A*sin(x)*cos(y)
    assert trigsimp(A*sin(x + y) - A*sin(x - y)) == 2*A*sin(y)*cos(x)
    assert trigsimp(A*cos(x + y) + A*cos(x - y)) == 2*A*cos(x)*cos(y)
    assert trigsimp(A*cos(x + y) - A*cos(x - y)) == -2*A*sin(x)*sin(y)

    assert trigsimp(A*sinh(x + y) + A*sinh(x - y)) == 2*A*sinh(x)*cosh(y)
    assert trigsimp(A*sinh(x + y) - A*sinh(x - y)) == 2*A*sinh(y)*cosh(x)
    assert trigsimp(A*cosh(x + y) + A*cosh(x - y)) == 2*A*cosh(x)*cosh(y)
    assert trigsimp(A*cosh(x + y) - A*cosh(x - y)) == 2*A*sinh(x)*sinh(y)

    assert trigsimp(A*cos(0.12345)**2 + A*sin(0.12345)**2) == 1.0*A


def test_hyperbolic_simp():
    x, y = symbols('x,y')

    assert trigsimp(sinh(x)**2 + 1) == cosh(x)**2
    assert trigsimp(cosh(x)**2 - 1) == sinh(x)**2
    assert trigsimp(cosh(x)**2 - sinh(x)**2) == 1
    assert trigsimp(1 - tanh(x)**2) == 1/cosh(x)**2
    assert trigsimp(1 - 1/cosh(x)**2) == tanh(x)**2
    assert trigsimp(tanh(x)**2 + 1/cosh(x)**2) == 1
    assert trigsimp(coth(x)**2 - 1) == 1/sinh(x)**2
    assert trigsimp(1/sinh(x)**2 + 1) == 1/tanh(x)**2
    assert trigsimp(coth(x)**2 - 1/sinh(x)**2) == 1

    assert trigsimp(5*cosh(x)**2 - 5*sinh(x)**2) == 5
    assert trigsimp(5*cosh(x/2)**2 - 2*sinh(x/2)**2) == 3*cosh(x)/2 + Rational(7, 2)

    assert trigsimp(sinh(x)/cosh(x)) == tanh(x)
    assert trigsimp(tanh(x)) == trigsimp(sinh(x)/cosh(x))
    assert trigsimp(cosh(x)/sinh(x)) == 1/tanh(x)
    assert trigsimp(2*tanh(x)*cosh(x)) == 2*sinh(x)
    assert trigsimp(coth(x)**3*sinh(x)**3) == cosh(x)**3
    assert trigsimp(y*tanh(x)**2/sinh(x)**2) == y/cosh(x)**2
    assert trigsimp(coth(x)/cosh(x)) == 1/sinh(x)

    for a in (pi/6*I, pi/4*I, pi/3*I):
        assert trigsimp(sinh(a)*cosh(x) + cosh(a)*sinh(x)) == sinh(x + a)
        assert trigsimp(-sinh(a)*cosh(x) + cosh(a)*sinh(x)) == sinh(x - a)

    e = 2*cosh(x)**2 - 2*sinh(x)**2
    assert trigsimp(log(e)) == log(2)

    # issue 19535:
    assert trigsimp(sqrt(cosh(x)**2 - 1)) == sqrt(sinh(x)**2)

    assert trigsimp(cosh(x)**2*cosh(y)**2 - cosh(x)**2*sinh(y)**2 - sinh(x)**2,
            recursive=True) == 1
    assert trigsimp(sinh(x)**2*sinh(y)**2 - sinh(x)**2*cosh(y)**2 + cosh(x)**2,
            recursive=True) == 1

    assert abs(trigsimp(2.0*cosh(x)**2 - 2.0*sinh(x)**2) - 2.0) < 1e-10

    assert trigsimp(sinh(x)**2/cosh(x)**2) == tanh(x)**2
    assert trigsimp(sinh(x)**3/cosh(x)**3) == tanh(x)**3
    assert trigsimp(sinh(x)**10/cosh(x)**10) == tanh(x)**10
    assert trigsimp(cosh(x)**3/sinh(x)**3) == 1/tanh(x)**3

    assert trigsimp(cosh(x)/sinh(x)) == 1/tanh(x)
    assert trigsimp(cosh(x)**2/sinh(x)**2) == 1/tanh(x)**2
    assert trigsimp(cosh(x)**10/sinh(x)**10) == 1/tanh(x)**10

    assert trigsimp(x*cosh(x)*tanh(x)) == x*sinh(x)
    assert trigsimp(-sinh(x) + cosh(x)*tanh(x)) == 0

    assert tan(x) != 1/cot(x)  # cot doesn't auto-simplify

    assert trigsimp(tan(x) - 1/cot(x)) == 0
    assert trigsimp(3*tanh(x)**7 - 2/coth(x)**7) == tanh(x)**7


def test_trigsimp_groebner():
    from sympy.simplify.trigsimp import trigsimp_groebner

    c = cos(x)
    s = sin(x)
    ex = (4*s*c + 12*s + 5*c**3 + 21*c**2 + 23*c + 15)/(
        -s*c**2 + 2*s*c + 15*s + 7*c**3 + 31*c**2 + 37*c + 21)
    resnum = (5*s - 5*c + 1)
    resdenom = (8*s - 6*c)
    results = [resnum/resdenom, (-resnum)/(-resdenom)]
    assert trigsimp_groebner(ex) in results
    assert trigsimp_groebner(s/c, hints=[tan]) == tan(x)
    assert trigsimp_groebner(c*s) == c*s
    assert trigsimp((-s + 1)/c + c/(-s + 1),
                    method='groebner') == 2/c
    assert trigsimp((-s + 1)/c + c/(-s + 1),
                    method='groebner', polynomial=True) == 2/c

    # Test quick=False works
    assert trigsimp_groebner(ex, hints=[2]) in results
    assert trigsimp_groebner(ex, hints=[int(2)]) in results

    # test "I"
    assert trigsimp_groebner(sin(I*x)/cos(I*x), hints=[tanh]) == I*tanh(x)

    # test hyperbolic / sums
    assert trigsimp_groebner((tanh(x)+tanh(y))/(1+tanh(x)*tanh(y)),
                             hints=[(tanh, x, y)]) == tanh(x + y)


def test_issue_2827_trigsimp_methods():
    measure1 = lambda expr: len(str(expr))
    measure2 = lambda expr: -count_ops(expr)
                                       # Return the most complicated result
    expr = (x + 1)/(x + sin(x)**2 + cos(x)**2)
    ans = Matrix([1])
    M = Matrix([expr])
    assert trigsimp(M, method='fu', measure=measure1) == ans
    assert trigsimp(M, method='fu', measure=measure2) != ans
    # all methods should work with Basic expressions even if they
    # aren't Expr
    M = Matrix.eye(1)
    assert all(trigsimp(M, method=m) == M for m in
        'fu matching groebner old'.split())
    # watch for E in exptrigsimp, not only exp()
    eq = 1/sqrt(E) + E
    assert exptrigsimp(eq) == eq

def test_issue_15129_trigsimp_methods():
    t1 = Matrix([sin(Rational(1, 50)), cos(Rational(1, 50)), 0])
    t2 = Matrix([sin(Rational(1, 25)), cos(Rational(1, 25)), 0])
    t3 = Matrix([cos(Rational(1, 25)), sin(Rational(1, 25)), 0])
    r1 = t1.dot(t2)
    r2 = t1.dot(t3)
    assert trigsimp(r1) == cos(Rational(1, 50))
    assert trigsimp(r2) == sin(Rational(3, 50))

def test_exptrigsimp():
    def valid(a, b):
        from sympy.core.random import verify_numerically as tn
        if not (tn(a, b) and a == b):
            return False
        return True

    assert exptrigsimp(exp(x) + exp(-x)) == 2*cosh(x)
    assert exptrigsimp(exp(x) - exp(-x)) == 2*sinh(x)
    assert exptrigsimp((2*exp(x)-2*exp(-x))/(exp(x)+exp(-x))) == 2*tanh(x)
    assert exptrigsimp((2*exp(2*x)-2)/(exp(2*x)+1)) == 2*tanh(x)
    e = [cos(x) + I*sin(x), cos(x) - I*sin(x),
         cosh(x) - sinh(x), cosh(x) + sinh(x)]
    ok = [exp(I*x), exp(-I*x), exp(-x), exp(x)]
    assert all(valid(i, j) for i, j in zip(
        [exptrigsimp(ei) for ei in e], ok))

    ue = [cos(x) + sin(x), cos(x) - sin(x),
          cosh(x) + I*sinh(x), cosh(x) - I*sinh(x)]
    assert [exptrigsimp(ei) == ei for ei in ue]

    res = []
    ok = [y*tanh(1), 1/(y*tanh(1)), I*y*tan(1), -I/(y*tan(1)),
        y*tanh(x), 1/(y*tanh(x)), I*y*tan(x), -I/(y*tan(x)),
        y*tanh(1 + I), 1/(y*tanh(1 + I))]
    for a in (1, I, x, I*x, 1 + I):
        w = exp(a)
        eq = y*(w - 1/w)/(w + 1/w)
        res.append(simplify(eq))
        res.append(simplify(1/eq))
    assert all(valid(i, j) for i, j in zip(res, ok))

    for a in range(1, 3):
        w = exp(a)
        e = w + 1/w
        s = simplify(e)
        assert s == exptrigsimp(e)
        assert valid(s, 2*cosh(a))
        e = w - 1/w
        s = simplify(e)
        assert s == exptrigsimp(e)
        assert valid(s, 2*sinh(a))

def test_exptrigsimp_noncommutative():
    a,b = symbols('a b', commutative=False)
    x = Symbol('x', commutative=True)
    assert exp(a + x) == exptrigsimp(exp(a)*exp(x))
    p = exp(a)*exp(b) - exp(b)*exp(a)
    assert p == exptrigsimp(p) != 0

def test_powsimp_on_numbers():
    assert 2**(Rational(1, 3) - 2) == 2**Rational(1, 3)/4


@XFAIL
def test_issue_6811_fail():
    # from doc/src/modules/physics/mechanics/examples.rst, the current `eq`
    # at Line 576 (in different variables) was formerly the equivalent and
    # shorter expression given below...it would be nice to get the short one
    # back again
    xp, y, x, z = symbols('xp, y, x, z')
    eq = 4*(-19*sin(x)*y + 5*sin(3*x)*y + 15*cos(2*x)*z - 21*z)*xp/(9*cos(x) - 5*cos(3*x))
    assert trigsimp(eq) == -2*(2*cos(x)*tan(x)*y + 3*z)*xp/cos(x)


def test_Piecewise():
    e1 = x*(x + y) - y*(x + y)
    e2 = sin(x)**2 + cos(x)**2
    e3 = expand((x + y)*y/x)
    # s1 = simplify(e1)
    s2 = simplify(e2)
    # s3 = simplify(e3)

    # trigsimp tries not to touch non-trig containing args
    assert trigsimp(Piecewise((e1, e3 < e2), (e3, True))) == \
        Piecewise((e1, e3 < s2), (e3, True))


def test_issue_21594():
    assert simplify(exp(Rational(1,2)) + exp(Rational(-1,2))) == cosh(S.Half)*2


def test_trigsimp_old():
    x, y = symbols('x,y')

    assert trigsimp(1 - sin(x)**2, old=True) == cos(x)**2
    assert trigsimp(1 - cos(x)**2, old=True) == sin(x)**2
    assert trigsimp(sin(x)**2 + cos(x)**2, old=True) == 1
    assert trigsimp(1 + tan(x)**2, old=True) == 1/cos(x)**2
    assert trigsimp(1/cos(x)**2 - 1, old=True) == tan(x)**2
    assert trigsimp(1/cos(x)**2 - tan(x)**2, old=True) == 1
    assert trigsimp(1 + cot(x)**2, old=True) == 1/sin(x)**2
    assert trigsimp(1/sin(x)**2 - cot(x)**2, old=True) == 1

    assert trigsimp(5*cos(x)**2 + 5*sin(x)**2, old=True) == 5

    assert trigsimp(sin(x)/cos(x), old=True) == tan(x)
    assert trigsimp(2*tan(x)*cos(x), old=True) == 2*sin(x)
    assert trigsimp(cot(x)**3*sin(x)**3, old=True) == cos(x)**3
    assert trigsimp(y*tan(x)**2/sin(x)**2, old=True) == y/cos(x)**2
    assert trigsimp(cot(x)/cos(x), old=True) == 1/sin(x)

    assert trigsimp(sin(x + y) + sin(x - y), old=True) == 2*sin(x)*cos(y)
    assert trigsimp(sin(x + y) - sin(x - y), old=True) == 2*sin(y)*cos(x)
    assert trigsimp(cos(x + y) + cos(x - y), old=True) == 2*cos(x)*cos(y)
    assert trigsimp(cos(x + y) - cos(x - y), old=True) == -2*sin(x)*sin(y)

    assert trigsimp(sinh(x + y) + sinh(x - y), old=True) == 2*sinh(x)*cosh(y)
    assert trigsimp(sinh(x + y) - sinh(x - y), old=True) == 2*sinh(y)*cosh(x)
    assert trigsimp(cosh(x + y) + cosh(x - y), old=True) == 2*cosh(x)*cosh(y)
    assert trigsimp(cosh(x + y) - cosh(x - y), old=True) == 2*sinh(x)*sinh(y)

    assert trigsimp(cos(0.12345)**2 + sin(0.12345)**2, old=True) == 1.0

    assert trigsimp(sin(x)/cos(x), old=True, method='combined') == tan(x)
    assert trigsimp(sin(x)/cos(x), old=True, method='groebner') == sin(x)/cos(x)
    assert trigsimp(sin(x)/cos(x), old=True, method='groebner', hints=[tan]) == tan(x)

    assert trigsimp(1-sin(sin(x)**2+cos(x)**2)**2, old=True, deep=True) == cos(1)**2


def test_trigsimp_inverse():
    alpha = symbols('alpha')
    s, c = sin(alpha), cos(alpha)

    for finv in [asin, acos, asec, acsc, atan, acot]:
        f = finv.inverse(None)
        assert alpha == trigsimp(finv(f(alpha)), inverse=True)

    # test atan2(cos, sin), atan2(sin, cos), etc...
    for a, b in [[c, s], [s, c]]:
        for i, j in product([-1, 1], repeat=2):
            angle = atan2(i*b, j*a)
            angle_inverted = trigsimp(angle, inverse=True)
            assert angle_inverted != angle  # assures simplification happened
            assert sin(angle_inverted) == trigsimp(sin(angle))
            assert cos(angle_inverted) == trigsimp(cos(angle))
