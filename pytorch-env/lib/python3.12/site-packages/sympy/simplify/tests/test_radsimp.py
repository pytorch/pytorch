from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.series.order import O
from sympy.simplify.radsimp import (collect, collect_const, fraction, radsimp, rcollect)

from sympy.core.expr import unchanged
from sympy.core.mul import _unevaluated_Mul as umul
from sympy.simplify.radsimp import (_unevaluated_Add,
    collect_sqrt, fraction_expand, collect_abs)
from sympy.testing.pytest import raises

from sympy.abc import x, y, z, a, b, c, d


def test_radsimp():
    r2 = sqrt(2)
    r3 = sqrt(3)
    r5 = sqrt(5)
    r7 = sqrt(7)
    assert fraction(radsimp(1/r2)) == (sqrt(2), 2)
    assert radsimp(1/(1 + r2)) == \
        -1 + sqrt(2)
    assert radsimp(1/(r2 + r3)) == \
        -sqrt(2) + sqrt(3)
    assert fraction(radsimp(1/(1 + r2 + r3))) == \
        (-sqrt(6) + sqrt(2) + 2, 4)
    assert fraction(radsimp(1/(r2 + r3 + r5))) == \
        (-sqrt(30) + 2*sqrt(3) + 3*sqrt(2), 12)
    assert fraction(radsimp(1/(1 + r2 + r3 + r5))) == (
        (-34*sqrt(10) - 26*sqrt(15) - 55*sqrt(3) - 61*sqrt(2) + 14*sqrt(30) +
        93 + 46*sqrt(6) + 53*sqrt(5), 71))
    assert fraction(radsimp(1/(r2 + r3 + r5 + r7))) == (
        (-50*sqrt(42) - 133*sqrt(5) - 34*sqrt(70) - 145*sqrt(3) + 22*sqrt(105)
        + 185*sqrt(2) + 62*sqrt(30) + 135*sqrt(7), 215))
    z = radsimp(1/(1 + r2/3 + r3/5 + r5 + r7))
    assert len((3616791619821680643598*z).args) == 16
    assert radsimp(1/z) == 1/z
    assert radsimp(1/z, max_terms=20).expand() == 1 + r2/3 + r3/5 + r5 + r7
    assert radsimp(1/(r2*3)) == \
        sqrt(2)/6
    assert radsimp(1/(r2*a + r3 + r5 + r7)) == (
        (8*sqrt(2)*a**7 - 8*sqrt(7)*a**6 - 8*sqrt(5)*a**6 - 8*sqrt(3)*a**6 -
        180*sqrt(2)*a**5 + 8*sqrt(30)*a**5 + 8*sqrt(42)*a**5 + 8*sqrt(70)*a**5
        - 24*sqrt(105)*a**4 + 84*sqrt(3)*a**4 + 100*sqrt(5)*a**4 +
        116*sqrt(7)*a**4 - 72*sqrt(70)*a**3 - 40*sqrt(42)*a**3 -
        8*sqrt(30)*a**3 + 782*sqrt(2)*a**3 - 462*sqrt(3)*a**2 -
        302*sqrt(7)*a**2 - 254*sqrt(5)*a**2 + 120*sqrt(105)*a**2 -
        795*sqrt(2)*a - 62*sqrt(30)*a + 82*sqrt(42)*a + 98*sqrt(70)*a -
        118*sqrt(105) + 59*sqrt(7) + 295*sqrt(5) + 531*sqrt(3))/(16*a**8 -
        480*a**6 + 3128*a**4 - 6360*a**2 + 3481))
    assert radsimp(1/(r2*a + r2*b + r3 + r7)) == (
        (sqrt(2)*a*(a + b)**2 - 5*sqrt(2)*a + sqrt(42)*a + sqrt(2)*b*(a +
        b)**2 - 5*sqrt(2)*b + sqrt(42)*b - sqrt(7)*(a + b)**2 - sqrt(3)*(a +
        b)**2 - 2*sqrt(3) + 2*sqrt(7))/(2*a**4 + 8*a**3*b + 12*a**2*b**2 -
        20*a**2 + 8*a*b**3 - 40*a*b + 2*b**4 - 20*b**2 + 8))
    assert radsimp(1/(r2*a + r2*b + r2*c + r2*d)) == \
        sqrt(2)/(2*a + 2*b + 2*c + 2*d)
    assert radsimp(1/(1 + r2*a + r2*b + r2*c + r2*d)) == (
        (sqrt(2)*a + sqrt(2)*b + sqrt(2)*c + sqrt(2)*d - 1)/(2*a**2 + 4*a*b +
        4*a*c + 4*a*d + 2*b**2 + 4*b*c + 4*b*d + 2*c**2 + 4*c*d + 2*d**2 - 1))
    assert radsimp((y**2 - x)/(y - sqrt(x))) == \
        sqrt(x) + y
    assert radsimp(-(y**2 - x)/(y - sqrt(x))) == \
        -(sqrt(x) + y)
    assert radsimp(1/(1 - I + a*I)) == \
        (-I*a + 1 + I)/(a**2 - 2*a + 2)
    assert radsimp(1/((-x + y)*(x - sqrt(y)))) == \
        (-x - sqrt(y))/((x - y)*(x**2 - y))
    e = (3 + 3*sqrt(2))*x*(3*x - 3*sqrt(y))
    assert radsimp(e) == x*(3 + 3*sqrt(2))*(3*x - 3*sqrt(y))
    assert radsimp(1/e) == (
        (-9*x + 9*sqrt(2)*x - 9*sqrt(y) + 9*sqrt(2)*sqrt(y))/(9*x*(9*x**2 -
        9*y)))
    assert radsimp(1 + 1/(1 + sqrt(3))) == \
        Mul(S.Half, -1 + sqrt(3), evaluate=False) + 1
    A = symbols("A", commutative=False)
    assert radsimp(x**2 + sqrt(2)*x**2 - sqrt(2)*x*A) == \
        x**2 + sqrt(2)*x**2 - sqrt(2)*x*A
    assert radsimp(1/sqrt(5 + 2 * sqrt(6))) == -sqrt(2) + sqrt(3)
    assert radsimp(1/sqrt(5 + 2 * sqrt(6))**3) == -(-sqrt(3) + sqrt(2))**3

    # issue 6532
    assert fraction(radsimp(1/sqrt(x))) == (sqrt(x), x)
    assert fraction(radsimp(1/sqrt(2*x + 3))) == (sqrt(2*x + 3), 2*x + 3)
    assert fraction(radsimp(1/sqrt(2*(x + 3)))) == (sqrt(2*x + 6), 2*x + 6)

    # issue 5994
    e = S('-(2 + 2*sqrt(2) + 4*2**(1/4))/'
        '(1 + 2**(3/4) + 3*2**(1/4) + 3*sqrt(2))')
    assert radsimp(e).expand() == -2*2**Rational(3, 4) - 2*2**Rational(1, 4) + 2 + 2*sqrt(2)

    # issue 5986 (modifications to radimp didn't initially recognize this so
    # the test is included here)
    assert radsimp(1/(-sqrt(5)/2 - S.Half + (-sqrt(5)/2 - S.Half)**2)) == 1

    # from issue 5934
    eq = (
        (-240*sqrt(2)*sqrt(sqrt(5) + 5)*sqrt(8*sqrt(5) + 40) -
        360*sqrt(2)*sqrt(-8*sqrt(5) + 40)*sqrt(-sqrt(5) + 5) -
        120*sqrt(10)*sqrt(-8*sqrt(5) + 40)*sqrt(-sqrt(5) + 5) +
        120*sqrt(2)*sqrt(-sqrt(5) + 5)*sqrt(8*sqrt(5) + 40) +
        120*sqrt(2)*sqrt(-8*sqrt(5) + 40)*sqrt(sqrt(5) + 5) +
        120*sqrt(10)*sqrt(-sqrt(5) + 5)*sqrt(8*sqrt(5) + 40) +
        120*sqrt(10)*sqrt(-8*sqrt(5) + 40)*sqrt(sqrt(5) + 5))/(-36000 -
        7200*sqrt(5) + (12*sqrt(10)*sqrt(sqrt(5) + 5) +
        24*sqrt(10)*sqrt(-sqrt(5) + 5))**2))
    assert radsimp(eq) is S.NaN  # it's 0/0

    # work with normal form
    e = 1/sqrt(sqrt(7)/7 + 2*sqrt(2) + 3*sqrt(3) + 5*sqrt(5)) + 3
    assert radsimp(e) == (
        -sqrt(sqrt(7) + 14*sqrt(2) + 21*sqrt(3) +
        35*sqrt(5))*(-11654899*sqrt(35) - 1577436*sqrt(210) - 1278438*sqrt(15)
        - 1346996*sqrt(10) + 1635060*sqrt(6) + 5709765 + 7539830*sqrt(14) +
        8291415*sqrt(21))/1300423175 + 3)

    # obey power rules
    base = sqrt(3) - sqrt(2)
    assert radsimp(1/base**3) == (sqrt(3) + sqrt(2))**3
    assert radsimp(1/(-base)**3) == -(sqrt(2) + sqrt(3))**3
    assert radsimp(1/(-base)**x) == (-base)**(-x)
    assert radsimp(1/base**x) == (sqrt(2) + sqrt(3))**x
    assert radsimp(root(1/(-1 - sqrt(2)), -x)) == (-1)**(-1/x)*(1 + sqrt(2))**(1/x)

    # recurse
    e = cos(1/(1 + sqrt(2)))
    assert radsimp(e) == cos(-sqrt(2) + 1)
    assert radsimp(e/2) == cos(-sqrt(2) + 1)/2
    assert radsimp(1/e) == 1/cos(-sqrt(2) + 1)
    assert radsimp(2/e) == 2/cos(-sqrt(2) + 1)
    assert fraction(radsimp(e/sqrt(x))) == (sqrt(x)*cos(-sqrt(2)+1), x)

    # test that symbolic denominators are not processed
    r = 1 + sqrt(2)
    assert radsimp(x/r, symbolic=False) == -x*(-sqrt(2) + 1)
    assert radsimp(x/(y + r), symbolic=False) == x/(y + 1 + sqrt(2))
    assert radsimp(x/(y + r)/r, symbolic=False) == \
        -x*(-sqrt(2) + 1)/(y + 1 + sqrt(2))

    # issue 7408
    eq = sqrt(x)/sqrt(y)
    assert radsimp(eq) == umul(sqrt(x), sqrt(y), 1/y)
    assert radsimp(eq, symbolic=False) == eq

    # issue 7498
    assert radsimp(sqrt(x)/sqrt(y)**3) == umul(sqrt(x), sqrt(y**3), 1/y**3)

    # for coverage
    eq = sqrt(x)/y**2
    assert radsimp(eq) == eq


def test_radsimp_issue_3214():
    c, p = symbols('c p', positive=True)
    s = sqrt(c**2 - p**2)
    b = (c + I*p - s)/(c + I*p + s)
    assert radsimp(b) == -I*(c + I*p - sqrt(c**2 - p**2))**2/(2*c*p)


def test_collect_1():
    """Collect with respect to Symbol"""
    x, y, z, n = symbols('x,y,z,n')
    assert collect(1, x) == 1
    assert collect( x + y*x, x ) == x * (1 + y)
    assert collect( x + x**2, x ) == x + x**2
    assert collect( x**2 + y*x**2, x ) == (x**2)*(1 + y)
    assert collect( x**2 + y*x, x ) == x*y + x**2
    assert collect( 2*x**2 + y*x**2 + 3*x*y, [x] ) == x**2*(2 + y) + 3*x*y
    assert collect( 2*x**2 + y*x**2 + 3*x*y, [y] ) == 2*x**2 + y*(x**2 + 3*x)

    assert collect( ((1 + y + x)**4).expand(), x) == ((1 + y)**4).expand() + \
        x*(4*(1 + y)**3).expand() + x**2*(6*(1 + y)**2).expand() + \
        x**3*(4*(1 + y)).expand() + x**4
    # symbols can be given as any iterable
    expr = x + y
    assert collect(expr, expr.free_symbols) == expr
    assert collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x, x, exact=None
        ) == x*exp(x) + 3*x + (y + 2)*sin(x)
    assert collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x + y*x +
        y*x*exp(x), x, exact=None
        ) == x*exp(x)*(y + 1) + (3 + y)*x + (y + 2)*sin(x)


def test_collect_2():
    """Collect with respect to a sum"""
    a, b, x = symbols('a,b,x')
    assert collect(a*(cos(x) + sin(x)) + b*(cos(x) + sin(x)),
        sin(x) + cos(x)) == (a + b)*(cos(x) + sin(x))


def test_collect_3():
    """Collect with respect to a product"""
    a, b, c = symbols('a,b,c')
    f = Function('f')
    x, y, z, n = symbols('x,y,z,n')

    assert collect(-x/8 + x*y, -x) == x*(y - Rational(1, 8))

    assert collect( 1 + x*(y**2), x*y ) == 1 + x*(y**2)
    assert collect( x*y + a*x*y, x*y) == x*y*(1 + a)
    assert collect( 1 + x*y + a*x*y, x*y) == 1 + x*y*(1 + a)
    assert collect(a*x*f(x) + b*(x*f(x)), x*f(x)) == x*(a + b)*f(x)

    assert collect(a*x*log(x) + b*(x*log(x)), x*log(x)) == x*(a + b)*log(x)
    assert collect(a*x**2*log(x)**2 + b*(x*log(x))**2, x*log(x)) == \
        x**2*log(x)**2*(a + b)

    # with respect to a product of three symbols
    assert collect(y*x*z + a*x*y*z, x*y*z) == (1 + a)*x*y*z


def test_collect_4():
    """Collect with respect to a power"""
    a, b, c, x = symbols('a,b,c,x')

    assert collect(a*x**c + b*x**c, x**c) == x**c*(a + b)
    # issue 6096: 2 stays with c (unless c is integer or x is positive0
    assert collect(a*x**(2*c) + b*x**(2*c), x**c) == x**(2*c)*(a + b)


def test_collect_5():
    """Collect with respect to a tuple"""
    a, x, y, z, n = symbols('a,x,y,z,n')
    assert collect(x**2*y**4 + z*(x*y**2)**2 + z + a*z, [x*y**2, z]) in [
        z*(1 + a + x**2*y**4) + x**2*y**4,
        z*(1 + a) + x**2*y**4*(1 + z) ]
    assert collect((1 + (x + y) + (x + y)**2).expand(),
                   [x, y]) == 1 + y + x*(1 + 2*y) + x**2 + y**2


def test_collect_pr19431():
    """Unevaluated collect with respect to a product"""
    a = symbols('a')
    assert collect(a**2*(a**2 + 1), a**2, evaluate=False)[a**2] == (a**2 + 1)


def test_collect_D():
    D = Derivative
    f = Function('f')
    x, a, b = symbols('x,a,b')
    fx = D(f(x), x)
    fxx = D(f(x), x, x)

    assert collect(a*fx + b*fx, fx) == (a + b)*fx
    assert collect(a*D(fx, x) + b*D(fx, x), fx) == (a + b)*D(fx, x)
    assert collect(a*fxx + b*fxx, fx) == (a + b)*D(fx, x)
    # issue 4784
    assert collect(5*f(x) + 3*fx, fx) == 5*f(x) + 3*fx
    assert collect(f(x) + f(x)*diff(f(x), x) + x*diff(f(x), x)*f(x), f(x).diff(x)) == \
        (x*f(x) + f(x))*D(f(x), x) + f(x)
    assert collect(f(x) + f(x)*diff(f(x), x) + x*diff(f(x), x)*f(x), f(x).diff(x), exact=True) == \
        (x*f(x) + f(x))*D(f(x), x) + f(x)
    assert collect(1/f(x) + 1/f(x)*diff(f(x), x) + x*diff(f(x), x)/f(x), f(x).diff(x), exact=True) == \
        (1/f(x) + x/f(x))*D(f(x), x) + 1/f(x)
    e = (1 + x*fx + fx)/f(x)
    assert collect(e.expand(), fx) == fx*(x/f(x) + 1/f(x)) + 1/f(x)


def test_collect_func():
    f = ((x + a + 1)**3).expand()

    assert collect(f, x) == a**3 + 3*a**2 + 3*a + x**3 + x**2*(3*a + 3) + \
        x*(3*a**2 + 6*a + 3) + 1
    assert collect(f, x, factor) == x**3 + 3*x**2*(a + 1) + 3*x*(a + 1)**2 + \
        (a + 1)**3

    assert collect(f, x, evaluate=False) == {
        S.One: a**3 + 3*a**2 + 3*a + 1,
        x: 3*a**2 + 6*a + 3, x**2: 3*a + 3,
        x**3: 1
    }

    assert collect(f, x, factor, evaluate=False) == {
        S.One: (a + 1)**3, x: 3*(a + 1)**2,
        x**2: umul(S(3), a + 1), x**3: 1}


def test_collect_order():
    a, b, x, t = symbols('a,b,x,t')

    assert collect(t + t*x + t*x**2 + O(x**3), t) == t*(1 + x + x**2 + O(x**3))
    assert collect(t + t*x + x**2 + O(x**3), t) == \
        t*(1 + x + O(x**3)) + x**2 + O(x**3)

    f = a*x + b*x + c*x**2 + d*x**2 + O(x**3)
    g = x*(a + b) + x**2*(c + d) + O(x**3)

    assert collect(f, x) == g
    assert collect(f, x, distribute_order_term=False) == g

    f = sin(a + b).series(b, 0, 10)

    assert collect(f, [sin(a), cos(a)]) == \
        sin(a)*cos(b).series(b, 0, 10) + cos(a)*sin(b).series(b, 0, 10)
    assert collect(f, [sin(a), cos(a)], distribute_order_term=False) == \
        sin(a)*cos(b).series(b, 0, 10).removeO() + \
        cos(a)*sin(b).series(b, 0, 10).removeO() + O(b**10)


def test_rcollect():
    assert rcollect((x**2*y + x*y + x + y)/(x + y), y) == \
        (x + y*(1 + x + x**2))/(x + y)
    assert rcollect(sqrt(-((x + 1)*(y + 1))), z) == sqrt(-((x + 1)*(y + 1)))


def test_collect_D_0():
    D = Derivative
    f = Function('f')
    x, a, b = symbols('x,a,b')
    fxx = D(f(x), x, x)

    assert collect(a*fxx + b*fxx, fxx) == (a + b)*fxx


def test_collect_Wild():
    """Collect with respect to functions with Wild argument"""
    a, b, x, y = symbols('a b x y')
    f = Function('f')
    w1 = Wild('.1')
    w2 = Wild('.2')
    assert collect(f(x) + a*f(x), f(w1)) == (1 + a)*f(x)
    assert collect(f(x, y) + a*f(x, y), f(w1)) == f(x, y) + a*f(x, y)
    assert collect(f(x, y) + a*f(x, y), f(w1, w2)) == (1 + a)*f(x, y)
    assert collect(f(x, y) + a*f(x, y), f(w1, w1)) == f(x, y) + a*f(x, y)
    assert collect(f(x, x) + a*f(x, x), f(w1, w1)) == (1 + a)*f(x, x)
    assert collect(a*(x + 1)**y + (x + 1)**y, w1**y) == (1 + a)*(x + 1)**y
    assert collect(a*(x + 1)**y + (x + 1)**y, w1**b) == \
        a*(x + 1)**y + (x + 1)**y
    assert collect(a*(x + 1)**y + (x + 1)**y, (x + 1)**w2) == \
        (1 + a)*(x + 1)**y
    assert collect(a*(x + 1)**y + (x + 1)**y, w1**w2) == (1 + a)*(x + 1)**y


def test_collect_const():
    # coverage not provided by above tests
    assert collect_const(2*sqrt(3) + 4*a*sqrt(5)) == \
        2*(2*sqrt(5)*a + sqrt(3))  # let the primitive reabsorb
    assert collect_const(2*sqrt(3) + 4*a*sqrt(5), sqrt(3)) == \
        2*sqrt(3) + 4*a*sqrt(5)
    assert collect_const(sqrt(2)*(1 + sqrt(2)) + sqrt(3) + x*sqrt(2)) == \
        sqrt(2)*(x + 1 + sqrt(2)) + sqrt(3)

    # issue 5290
    assert collect_const(2*x + 2*y + 1, 2) == \
        collect_const(2*x + 2*y + 1) == \
        Add(S.One, Mul(2, x + y, evaluate=False), evaluate=False)
    assert collect_const(-y - z) == Mul(-1, y + z, evaluate=False)
    assert collect_const(2*x - 2*y - 2*z, 2) == \
        Mul(2, x - y - z, evaluate=False)
    assert collect_const(2*x - 2*y - 2*z, -2) == \
        _unevaluated_Add(2*x, Mul(-2, y + z, evaluate=False))

    # this is why the content_primitive is used
    eq = (sqrt(15 + 5*sqrt(2))*x + sqrt(3 + sqrt(2))*y)*2
    assert collect_sqrt(eq + 2) == \
        2*sqrt(sqrt(2) + 3)*(sqrt(5)*x + y) + 2

    # issue 16296
    assert collect_const(a + b + x/2 + y/2) == a + b + Mul(S.Half, x + y, evaluate=False)


def test_issue_13143():
    f = Function('f')
    fx = f(x).diff(x)
    e = f(x) + fx + f(x)*fx
    # collect function before derivative
    assert collect(e, Wild('w')) == f(x)*(fx + 1) + fx
    e = f(x) + f(x)*fx + x*fx*f(x)
    assert collect(e, fx) == (x*f(x) + f(x))*fx + f(x)
    assert collect(e, f(x)) == (x*fx + fx + 1)*f(x)
    e = f(x) + fx + f(x)*fx
    assert collect(e, [f(x), fx]) == f(x)*(1 + fx) + fx
    assert collect(e, [fx, f(x)]) == fx*(1 + f(x)) + f(x)


def test_issue_6097():
    assert collect(a*y**(2.0*x) + b*y**(2.0*x), y**x) == (a + b)*(y**x)**2.0
    assert collect(a*2**(2.0*x) + b*2**(2.0*x), 2**x) == (a + b)*(2**x)**2.0


def test_fraction_expand():
    eq = (x + y)*y/x
    assert eq.expand(frac=True) == fraction_expand(eq) == (x*y + y**2)/x
    assert eq.expand() == y + y**2/x


def test_fraction():
    x, y, z = map(Symbol, 'xyz')
    A = Symbol('A', commutative=False)

    assert fraction(S.Half) == (1, 2)

    assert fraction(x) == (x, 1)
    assert fraction(1/x) == (1, x)
    assert fraction(x/y) == (x, y)
    assert fraction(x/2) == (x, 2)

    assert fraction(x*y/z) == (x*y, z)
    assert fraction(x/(y*z)) == (x, y*z)

    assert fraction(1/y**2) == (1, y**2)
    assert fraction(x/y**2) == (x, y**2)

    assert fraction((x**2 + 1)/y) == (x**2 + 1, y)
    assert fraction(x*(y + 1)/y**7) == (x*(y + 1), y**7)

    assert fraction(exp(-x), exact=True) == (exp(-x), 1)
    assert fraction((1/(x + y))/2, exact=True) == (1, Mul(2,(x + y), evaluate=False))

    assert fraction(x*A/y) == (x*A, y)
    assert fraction(x*A**-1/y) == (x*A**-1, y)

    n = symbols('n', negative=True)
    assert fraction(exp(n)) == (1, exp(-n))
    assert fraction(exp(-n)) == (exp(-n), 1)

    p = symbols('p', positive=True)
    assert fraction(exp(-p)*log(p), exact=True) == (exp(-p)*log(p), 1)

    m = Mul(1, 1, S.Half, evaluate=False)
    assert fraction(m) == (1, 2)
    assert fraction(m, exact=True) == (Mul(1, 1, evaluate=False), 2)

    m = Mul(1, 1, S.Half, S.Half, Pow(1, -1, evaluate=False), evaluate=False)
    assert fraction(m) == (1, 4)
    assert fraction(m, exact=True) == \
            (Mul(1, 1, evaluate=False), Mul(2, 2, 1, evaluate=False))


def test_issue_5615():
    aA, Re, a, b, D = symbols('aA Re a b D')
    e = ((D**3*a + b*aA**3)/Re).expand()
    assert collect(e, [aA**3/Re, a]) == e


def test_issue_5933():
    from sympy.geometry.polygon import (Polygon, RegularPolygon)
    from sympy.simplify.radsimp import denom
    x = Polygon(*RegularPolygon((0, 0), 1, 5).vertices).centroid.x
    assert abs(denom(x).n()) > 1e-12
    assert abs(denom(radsimp(x))) > 1e-12  # in case simplify didn't handle it


def test_issue_14608():
    a, b = symbols('a b', commutative=False)
    x, y = symbols('x y')
    raises(AttributeError, lambda: collect(a*b + b*a, a))
    assert collect(x*y + y*(x+1), a) == x*y + y*(x+1)
    assert collect(x*y + y*(x+1) + a*b + b*a, y) == y*(2*x + 1) + a*b + b*a


def test_collect_abs():
    s = abs(x) + abs(y)
    assert collect_abs(s) == s
    assert unchanged(Mul, abs(x), abs(y))
    ans = Abs(x*y)
    assert isinstance(ans, Abs)
    assert collect_abs(abs(x)*abs(y)) == ans
    assert collect_abs(1 + exp(abs(x)*abs(y))) == 1 + exp(ans)

    # See https://github.com/sympy/sympy/issues/12910
    p = Symbol('p', positive=True)
    assert collect_abs(p/abs(1-p)).is_commutative is True


def test_issue_19149():
    eq = exp(3*x/4)
    assert collect(eq, exp(x)) == eq

def test_issue_19719():
    a, b = symbols('a, b')
    expr = a**2 * (b + 1) + (7 + 1/b)/a
    collected = collect(expr, (a**2, 1/a), evaluate=False)
    # Would return {_Dummy_20**(-2): b + 1, 1/a: 7 + 1/b} without xreplace
    assert collected == {a**2: b + 1, 1/a: 7 + 1/b}


def test_issue_21355():
    assert radsimp(1/(x + sqrt(x**2))) == 1/(x + sqrt(x**2))
    assert radsimp(1/(x - sqrt(x**2))) == 1/(x - sqrt(x**2))
