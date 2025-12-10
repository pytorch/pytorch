from sympy.concrete.summations import Sum
from sympy.core.function import (Derivative, diff)
from sympy.core.numbers import (Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import (RisingFactorial, binomial, factorial)
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root, gegenbauer, hermite, hermite_prob, jacobi, jacobi_normalized, laguerre, legendre)
from sympy.polys.orthopolys import laguerre_poly
from sympy.polys.polyroots import roots

from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises


x = Symbol('x')


def test_jacobi():
    n = Symbol("n")
    a = Symbol("a")
    b = Symbol("b")

    assert jacobi(0, a, b, x) == 1
    assert jacobi(1, a, b, x) == a/2 - b/2 + x*(a/2 + b/2 + 1)

    assert jacobi(n, a, a, x) == RisingFactorial(
        a + 1, n)*gegenbauer(n, a + S.Half, x)/RisingFactorial(2*a + 1, n)
    assert jacobi(n, a, -a, x) == ((-1)**a*(-x + 1)**(-a/2)*(x + 1)**(a/2)*assoc_legendre(n, a, x)*
                                   factorial(-a + n)*gamma(a + n + 1)/(factorial(a + n)*gamma(n + 1)))
    assert jacobi(n, -b, b, x) == ((-x + 1)**(b/2)*(x + 1)**(-b/2)*assoc_legendre(n, b, x)*
                                   gamma(-b + n + 1)/gamma(n + 1))
    assert jacobi(n, 0, 0, x) == legendre(n, x)
    assert jacobi(n, S.Half, S.Half, x) == RisingFactorial(
        Rational(3, 2), n)*chebyshevu(n, x)/factorial(n + 1)
    assert jacobi(n, Rational(-1, 2), Rational(-1, 2), x) == RisingFactorial(
        S.Half, n)*chebyshevt(n, x)/factorial(n)

    X = jacobi(n, a, b, x)
    assert isinstance(X, jacobi)

    assert jacobi(n, a, b, -x) == (-1)**n*jacobi(n, b, a, x)
    assert jacobi(n, a, b, 0) == 2**(-n)*gamma(a + n + 1)*hyper(
        (-b - n, -n), (a + 1,), -1)/(factorial(n)*gamma(a + 1))
    assert jacobi(n, a, b, 1) == RisingFactorial(a + 1, n)/factorial(n)

    m = Symbol("m", positive=True)
    assert jacobi(m, a, b, oo) == oo*RisingFactorial(a + b + m + 1, m)
    assert unchanged(jacobi, n, a, b, oo)

    assert conjugate(jacobi(m, a, b, x)) == \
        jacobi(m, conjugate(a), conjugate(b), conjugate(x))

    _k = Dummy('k')
    assert diff(jacobi(n, a, b, x), n) == Derivative(jacobi(n, a, b, x), n)
    assert diff(jacobi(n, a, b, x), a).dummy_eq(Sum((jacobi(n, a, b, x) +
        (2*_k + a + b + 1)*RisingFactorial(_k + b + 1, -_k + n)*jacobi(_k, a,
        b, x)/((-_k + n)*RisingFactorial(_k + a + b + 1, -_k + n)))/(_k + a
        + b + n + 1), (_k, 0, n - 1)))
    assert diff(jacobi(n, a, b, x), b).dummy_eq(Sum(((-1)**(-_k + n)*(2*_k +
        a + b + 1)*RisingFactorial(_k + a + 1, -_k + n)*jacobi(_k, a, b, x)/
        ((-_k + n)*RisingFactorial(_k + a + b + 1, -_k + n)) + jacobi(n, a,
        b, x))/(_k + a + b + n + 1), (_k, 0, n - 1)))
    assert diff(jacobi(n, a, b, x), x) == \
        (a/2 + b/2 + n/2 + S.Half)*jacobi(n - 1, a + 1, b + 1, x)

    assert jacobi_normalized(n, a, b, x) == \
           (jacobi(n, a, b, x)/sqrt(2**(a + b + 1)*gamma(a + n + 1)*gamma(b + n + 1)
                                    /((a + b + 2*n + 1)*factorial(n)*gamma(a + b + n + 1))))

    raises(ValueError, lambda: jacobi(-2.1, a, b, x))
    raises(ValueError, lambda: jacobi(Dummy(positive=True, integer=True), 1, 2, oo))

    assert jacobi(n, a, b, x).rewrite(Sum).dummy_eq(Sum((S.Half - x/2)
        **_k*RisingFactorial(-n, _k)*RisingFactorial(_k + a + 1, -_k + n)*
        RisingFactorial(a + b + n + 1, _k)/factorial(_k), (_k, 0, n))/factorial(n))
    assert jacobi(n, a, b, x).rewrite("polynomial").dummy_eq(Sum((S.Half - x/2)
        **_k*RisingFactorial(-n, _k)*RisingFactorial(_k + a + 1, -_k + n)*
        RisingFactorial(a + b + n + 1, _k)/factorial(_k), (_k, 0, n))/factorial(n))
    raises(ArgumentIndexError, lambda: jacobi(n, a, b, x).fdiff(5))


def test_gegenbauer():
    n = Symbol("n")
    a = Symbol("a")

    assert gegenbauer(0, a, x) == 1
    assert gegenbauer(1, a, x) == 2*a*x
    assert gegenbauer(2, a, x) == -a + x**2*(2*a**2 + 2*a)
    assert gegenbauer(3, a, x) == \
        x**3*(4*a**3/3 + 4*a**2 + a*Rational(8, 3)) + x*(-2*a**2 - 2*a)

    assert gegenbauer(-1, a, x) == 0
    assert gegenbauer(n, S.Half, x) == legendre(n, x)
    assert gegenbauer(n, 1, x) == chebyshevu(n, x)
    assert gegenbauer(n, -1, x) == 0

    X = gegenbauer(n, a, x)
    assert isinstance(X, gegenbauer)

    assert gegenbauer(n, a, -x) == (-1)**n*gegenbauer(n, a, x)
    assert gegenbauer(n, a, 0) == 2**n*sqrt(pi) * \
        gamma(a + n/2)/(gamma(a)*gamma(-n/2 + S.Half)*gamma(n + 1))
    assert gegenbauer(n, a, 1) == gamma(2*a + n)/(gamma(2*a)*gamma(n + 1))

    assert gegenbauer(n, Rational(3, 4), -1) is zoo
    assert gegenbauer(n, Rational(1, 4), -1) == (sqrt(2)*cos(pi*(n + S.One/4))*
                      gamma(n + S.Half)/(sqrt(pi)*gamma(n + 1)))

    m = Symbol("m", positive=True)
    assert gegenbauer(m, a, oo) == oo*RisingFactorial(a, m)
    assert unchanged(gegenbauer, n, a, oo)

    assert conjugate(gegenbauer(n, a, x)) == gegenbauer(n, conjugate(a), conjugate(x))

    _k = Dummy('k')

    assert diff(gegenbauer(n, a, x), n) == Derivative(gegenbauer(n, a, x), n)
    assert diff(gegenbauer(n, a, x), a).dummy_eq(Sum((2*(-1)**(-_k + n) + 2)*
        (_k + a)*gegenbauer(_k, a, x)/((-_k + n)*(_k + 2*a + n)) + ((2*_k +
        2)/((_k + 2*a)*(2*_k + 2*a + 1)) + 2/(_k + 2*a + n))*gegenbauer(n, a
        , x), (_k, 0, n - 1)))
    assert diff(gegenbauer(n, a, x), x) == 2*a*gegenbauer(n - 1, a + 1, x)

    assert gegenbauer(n, a, x).rewrite(Sum).dummy_eq(
        Sum((-1)**_k*(2*x)**(-2*_k + n)*RisingFactorial(a, -_k + n)
        /(factorial(_k)*factorial(-2*_k + n)), (_k, 0, floor(n/2))))
    assert gegenbauer(n, a, x).rewrite("polynomial").dummy_eq(
        Sum((-1)**_k*(2*x)**(-2*_k + n)*RisingFactorial(a, -_k + n)
        /(factorial(_k)*factorial(-2*_k + n)), (_k, 0, floor(n/2))))

    raises(ArgumentIndexError, lambda: gegenbauer(n, a, x).fdiff(4))


def test_legendre():
    assert legendre(0, x) == 1
    assert legendre(1, x) == x
    assert legendre(2, x) == ((3*x**2 - 1)/2).expand()
    assert legendre(3, x) == ((5*x**3 - 3*x)/2).expand()
    assert legendre(4, x) == ((35*x**4 - 30*x**2 + 3)/8).expand()
    assert legendre(5, x) == ((63*x**5 - 70*x**3 + 15*x)/8).expand()
    assert legendre(6, x) == ((231*x**6 - 315*x**4 + 105*x**2 - 5)/16).expand()

    assert legendre(10, -1) == 1
    assert legendre(11, -1) == -1
    assert legendre(10, 1) == 1
    assert legendre(11, 1) == 1
    assert legendre(10, 0) != 0
    assert legendre(11, 0) == 0

    assert legendre(-1, x) == 1
    k = Symbol('k')
    assert legendre(5 - k, x).subs(k, 2) == ((5*x**3 - 3*x)/2).expand()

    assert roots(legendre(4, x), x) == {
        sqrt(Rational(3, 7) - Rational(2, 35)*sqrt(30)): 1,
        -sqrt(Rational(3, 7) - Rational(2, 35)*sqrt(30)): 1,
        sqrt(Rational(3, 7) + Rational(2, 35)*sqrt(30)): 1,
        -sqrt(Rational(3, 7) + Rational(2, 35)*sqrt(30)): 1,
    }

    n = Symbol("n")

    X = legendre(n, x)
    assert isinstance(X, legendre)
    assert unchanged(legendre, n, x)

    assert legendre(n, 0) == sqrt(pi)/(gamma(S.Half - n/2)*gamma(n/2 + 1))
    assert legendre(n, 1) == 1
    assert legendre(n, oo) is oo
    assert legendre(-n, x) == legendre(n - 1, x)
    assert legendre(n, -x) == (-1)**n*legendre(n, x)
    assert unchanged(legendre, -n + k, x)

    assert conjugate(legendre(n, x)) == legendre(n, conjugate(x))

    assert diff(legendre(n, x), x) == \
        n*(x*legendre(n, x) - legendre(n - 1, x))/(x**2 - 1)
    assert diff(legendre(n, x), n) == Derivative(legendre(n, x), n)

    _k = Dummy('k')
    assert legendre(n, x).rewrite(Sum).dummy_eq(Sum((-1)**_k*(S.Half -
            x/2)**_k*(x/2 + S.Half)**(-_k + n)*binomial(n, _k)**2, (_k, 0, n)))
    assert legendre(n, x).rewrite("polynomial").dummy_eq(Sum((-1)**_k*(S.Half -
            x/2)**_k*(x/2 + S.Half)**(-_k + n)*binomial(n, _k)**2, (_k, 0, n)))
    raises(ArgumentIndexError, lambda: legendre(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: legendre(n, x).fdiff(3))


def test_assoc_legendre():
    Plm = assoc_legendre
    Q = sqrt(1 - x**2)

    assert Plm(0, 0, x) == 1
    assert Plm(1, 0, x) == x
    assert Plm(1, 1, x) == -Q
    assert Plm(2, 0, x) == (3*x**2 - 1)/2
    assert Plm(2, 1, x) == -3*x*Q
    assert Plm(2, 2, x) == 3*Q**2
    assert Plm(3, 0, x) == (5*x**3 - 3*x)/2
    assert Plm(3, 1, x).expand() == (( 3*(1 - 5*x**2)/2 ).expand() * Q).expand()
    assert Plm(3, 2, x) == 15*x * Q**2
    assert Plm(3, 3, x) == -15 * Q**3

    # negative m
    assert Plm(1, -1, x) == -Plm(1, 1, x)/2
    assert Plm(2, -2, x) == Plm(2, 2, x)/24
    assert Plm(2, -1, x) == -Plm(2, 1, x)/6
    assert Plm(3, -3, x) == -Plm(3, 3, x)/720
    assert Plm(3, -2, x) == Plm(3, 2, x)/120
    assert Plm(3, -1, x) == -Plm(3, 1, x)/12

    n = Symbol("n")
    m = Symbol("m")
    X = Plm(n, m, x)
    assert isinstance(X, assoc_legendre)

    assert Plm(n, 0, x) == legendre(n, x)
    assert Plm(n, m, 0) == 2**m*sqrt(pi)/(gamma(-m/2 - n/2 +
                           S.Half)*gamma(-m/2 + n/2 + 1))

    assert diff(Plm(m, n, x), x) == (m*x*assoc_legendre(m, n, x) -
                (m + n)*assoc_legendre(m - 1, n, x))/(x**2 - 1)

    _k = Dummy('k')
    assert Plm(m, n, x).rewrite(Sum).dummy_eq(
            (1 - x**2)**(n/2)*Sum((-1)**_k*2**(-m)*x**(-2*_k + m - n)*factorial
             (-2*_k + 2*m)/(factorial(_k)*factorial(-_k + m)*factorial(-2*_k + m
              - n)), (_k, 0, floor(m/2 - n/2))))
    assert Plm(m, n, x).rewrite("polynomial").dummy_eq(
            (1 - x**2)**(n/2)*Sum((-1)**_k*2**(-m)*x**(-2*_k + m - n)*factorial
             (-2*_k + 2*m)/(factorial(_k)*factorial(-_k + m)*factorial(-2*_k + m
              - n)), (_k, 0, floor(m/2 - n/2))))
    assert conjugate(assoc_legendre(n, m, x)) == \
        assoc_legendre(n, conjugate(m), conjugate(x))
    raises(ValueError, lambda: Plm(0, 1, x))
    raises(ValueError, lambda: Plm(-1, 1, x))
    raises(ArgumentIndexError, lambda: Plm(n, m, x).fdiff(1))
    raises(ArgumentIndexError, lambda: Plm(n, m, x).fdiff(2))
    raises(ArgumentIndexError, lambda: Plm(n, m, x).fdiff(4))


def test_chebyshev():
    assert chebyshevt(0, x) == 1
    assert chebyshevt(1, x) == x
    assert chebyshevt(2, x) == 2*x**2 - 1
    assert chebyshevt(3, x) == 4*x**3 - 3*x

    for n in range(1, 4):
        for k in range(n):
            z = chebyshevt_root(n, k)
            assert chebyshevt(n, z) == 0
        raises(ValueError, lambda: chebyshevt_root(n, n))

    for n in range(1, 4):
        for k in range(n):
            z = chebyshevu_root(n, k)
            assert chebyshevu(n, z) == 0
        raises(ValueError, lambda: chebyshevu_root(n, n))

    n = Symbol("n")
    X = chebyshevt(n, x)
    assert isinstance(X, chebyshevt)
    assert unchanged(chebyshevt, n, x)
    assert chebyshevt(n, -x) == (-1)**n*chebyshevt(n, x)
    assert chebyshevt(-n, x) == chebyshevt(n, x)

    assert chebyshevt(n, 0) == cos(pi*n/2)
    assert chebyshevt(n, 1) == 1
    assert chebyshevt(n, oo) is oo

    assert conjugate(chebyshevt(n, x)) == chebyshevt(n, conjugate(x))

    assert diff(chebyshevt(n, x), x) == n*chebyshevu(n - 1, x)

    X = chebyshevu(n, x)
    assert isinstance(X, chebyshevu)

    y = Symbol('y')
    assert chebyshevu(n, -x) == (-1)**n*chebyshevu(n, x)
    assert chebyshevu(-n, x) == -chebyshevu(n - 2, x)
    assert unchanged(chebyshevu, -n + y, x)

    assert chebyshevu(n, 0) == cos(pi*n/2)
    assert chebyshevu(n, 1) == n + 1
    assert chebyshevu(n, oo) is oo

    assert conjugate(chebyshevu(n, x)) == chebyshevu(n, conjugate(x))

    assert diff(chebyshevu(n, x), x) == \
        (-x*chebyshevu(n, x) + (n + 1)*chebyshevt(n + 1, x))/(x**2 - 1)

    _k = Dummy('k')
    assert chebyshevt(n, x).rewrite(Sum).dummy_eq(Sum(x**(-2*_k + n)
                    *(x**2 - 1)**_k*binomial(n, 2*_k), (_k, 0, floor(n/2))))
    assert chebyshevt(n, x).rewrite("polynomial").dummy_eq(Sum(x**(-2*_k + n)
                    *(x**2 - 1)**_k*binomial(n, 2*_k), (_k, 0, floor(n/2))))
    assert chebyshevu(n, x).rewrite(Sum).dummy_eq(Sum((-1)**_k*(2*x)
                    **(-2*_k + n)*factorial(-_k + n)/(factorial(_k)*
                       factorial(-2*_k + n)), (_k, 0, floor(n/2))))
    assert chebyshevu(n, x).rewrite("polynomial").dummy_eq(Sum((-1)**_k*(2*x)
                    **(-2*_k + n)*factorial(-_k + n)/(factorial(_k)*
                       factorial(-2*_k + n)), (_k, 0, floor(n/2))))
    raises(ArgumentIndexError, lambda: chebyshevt(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: chebyshevt(n, x).fdiff(3))
    raises(ArgumentIndexError, lambda: chebyshevu(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: chebyshevu(n, x).fdiff(3))


def test_hermite():
    assert hermite(0, x) == 1
    assert hermite(1, x) == 2*x
    assert hermite(2, x) == 4*x**2 - 2
    assert hermite(3, x) == 8*x**3 - 12*x
    assert hermite(4, x) == 16*x**4 - 48*x**2 + 12
    assert hermite(6, x) == 64*x**6 - 480*x**4 + 720*x**2 - 120

    n = Symbol("n")
    assert unchanged(hermite, n, x)
    assert hermite(n, -x) == (-1)**n*hermite(n, x)
    assert unchanged(hermite, -n, x)

    assert hermite(n, 0) == 2**n*sqrt(pi)/gamma(S.Half - n/2)
    assert hermite(n, oo) is oo

    assert conjugate(hermite(n, x)) == hermite(n, conjugate(x))

    _k = Dummy('k')
    assert hermite(n, x).rewrite(Sum).dummy_eq(factorial(n)*Sum((-1)
        **_k*(2*x)**(-2*_k + n)/(factorial(_k)*factorial(-2*_k + n)), (_k,
        0, floor(n/2))))
    assert hermite(n, x).rewrite("polynomial").dummy_eq(factorial(n)*Sum((-1)
        **_k*(2*x)**(-2*_k + n)/(factorial(_k)*factorial(-2*_k + n)), (_k,
        0, floor(n/2))))

    assert diff(hermite(n, x), x) == 2*n*hermite(n - 1, x)
    assert diff(hermite(n, x), n) == Derivative(hermite(n, x), n)
    raises(ArgumentIndexError, lambda: hermite(n, x).fdiff(3))

    assert hermite(n, x).rewrite(hermite_prob) == \
            sqrt(2)**n * hermite_prob(n, x*sqrt(2))


def test_hermite_prob():
    assert hermite_prob(0, x) == 1
    assert hermite_prob(1, x) == x
    assert hermite_prob(2, x) == x**2 - 1
    assert hermite_prob(3, x) == x**3 - 3*x
    assert hermite_prob(4, x) == x**4 - 6*x**2 + 3
    assert hermite_prob(6, x) == x**6 - 15*x**4 + 45*x**2 - 15

    n = Symbol("n")
    assert unchanged(hermite_prob, n, x)
    assert hermite_prob(n, -x) == (-1)**n*hermite_prob(n, x)
    assert unchanged(hermite_prob, -n, x)

    assert hermite_prob(n, 0) == sqrt(pi)/gamma(S.Half - n/2)
    assert hermite_prob(n, oo) is oo

    assert conjugate(hermite_prob(n, x)) == hermite_prob(n, conjugate(x))

    _k = Dummy('k')
    assert hermite_prob(n, x).rewrite(Sum).dummy_eq(factorial(n) *
        Sum((-S.Half)**_k * x**(n-2*_k) / (factorial(_k) * factorial(n-2*_k)),
        (_k, 0, floor(n/2))))
    assert hermite_prob(n, x).rewrite("polynomial").dummy_eq(factorial(n) *
        Sum((-S.Half)**_k * x**(n-2*_k) / (factorial(_k) * factorial(n-2*_k)),
        (_k, 0, floor(n/2))))

    assert diff(hermite_prob(n, x), x) == n*hermite_prob(n-1, x)
    assert diff(hermite_prob(n, x), n) == Derivative(hermite_prob(n, x), n)
    raises(ArgumentIndexError, lambda: hermite_prob(n, x).fdiff(3))

    assert hermite_prob(n, x).rewrite(hermite) == \
            sqrt(2)**(-n) * hermite(n, x/sqrt(2))


def test_laguerre():
    n = Symbol("n")
    m = Symbol("m", negative=True)

    # Laguerre polynomials:
    assert laguerre(0, x) == 1
    assert laguerre(1, x) == -x + 1
    assert laguerre(2, x) == x**2/2 - 2*x + 1
    assert laguerre(3, x) == -x**3/6 + 3*x**2/2 - 3*x + 1
    assert laguerre(-2, x) == (x + 1)*exp(x)

    X = laguerre(n, x)
    assert isinstance(X, laguerre)

    assert laguerre(n, 0) == 1
    assert laguerre(n, oo) == (-1)**n*oo
    assert laguerre(n, -oo) is oo

    assert conjugate(laguerre(n, x)) == laguerre(n, conjugate(x))

    _k = Dummy('k')

    assert laguerre(n, x).rewrite(Sum).dummy_eq(
        Sum(x**_k*RisingFactorial(-n, _k)/factorial(_k)**2, (_k, 0, n)))
    assert laguerre(n, x).rewrite("polynomial").dummy_eq(
        Sum(x**_k*RisingFactorial(-n, _k)/factorial(_k)**2, (_k, 0, n)))
    assert laguerre(m, x).rewrite(Sum).dummy_eq(
        exp(x)*Sum((-x)**_k*RisingFactorial(m + 1, _k)/factorial(_k)**2,
            (_k, 0, -m - 1)))
    assert laguerre(m, x).rewrite("polynomial").dummy_eq(
        exp(x)*Sum((-x)**_k*RisingFactorial(m + 1, _k)/factorial(_k)**2,
            (_k, 0, -m - 1)))

    assert diff(laguerre(n, x), x) == -assoc_laguerre(n - 1, 1, x)

    k = Symbol('k')
    assert laguerre(-n, x) == exp(x)*laguerre(n - 1, -x)
    assert laguerre(-3, x) == exp(x)*laguerre(2, -x)
    assert unchanged(laguerre, -n + k, x)

    raises(ValueError, lambda: laguerre(-2.1, x))
    raises(ValueError, lambda: laguerre(Rational(5, 2), x))
    raises(ArgumentIndexError, lambda: laguerre(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: laguerre(n, x).fdiff(3))


def test_assoc_laguerre():
    n = Symbol("n")
    m = Symbol("m")
    alpha = Symbol("alpha")

    # generalized Laguerre polynomials:
    assert assoc_laguerre(0, alpha, x) == 1
    assert assoc_laguerre(1, alpha, x) == -x + alpha + 1
    assert assoc_laguerre(2, alpha, x).expand() == \
        (x**2/2 - (alpha + 2)*x + (alpha + 2)*(alpha + 1)/2).expand()
    assert assoc_laguerre(3, alpha, x).expand() == \
        (-x**3/6 + (alpha + 3)*x**2/2 - (alpha + 2)*(alpha + 3)*x/2 +
        (alpha + 1)*(alpha + 2)*(alpha + 3)/6).expand()

    # Test the lowest 10 polynomials with laguerre_poly, to make sure it works:
    for i in range(10):
        assert assoc_laguerre(i, 0, x).expand() == laguerre_poly(i, x)

    X = assoc_laguerre(n, m, x)
    assert isinstance(X, assoc_laguerre)

    assert assoc_laguerre(n, 0, x) == laguerre(n, x)
    assert assoc_laguerre(n, alpha, 0) == binomial(alpha + n, alpha)
    p = Symbol("p", positive=True)
    assert assoc_laguerre(p, alpha, oo) == (-1)**p*oo
    assert assoc_laguerre(p, alpha, -oo) is oo

    assert diff(assoc_laguerre(n, alpha, x), x) == \
        -assoc_laguerre(n - 1, alpha + 1, x)
    _k = Dummy('k')
    assert diff(assoc_laguerre(n, alpha, x), alpha).dummy_eq(
        Sum(assoc_laguerre(_k, alpha, x)/(-alpha + n), (_k, 0, n - 1)))

    assert conjugate(assoc_laguerre(n, alpha, x)) == \
        assoc_laguerre(n, conjugate(alpha), conjugate(x))

    assert assoc_laguerre(n, alpha, x).rewrite(Sum).dummy_eq(
            gamma(alpha + n + 1)*Sum(x**_k*RisingFactorial(-n, _k)/
            (factorial(_k)*gamma(_k + alpha + 1)), (_k, 0, n))/factorial(n))
    assert assoc_laguerre(n, alpha, x).rewrite("polynomial").dummy_eq(
            gamma(alpha + n + 1)*Sum(x**_k*RisingFactorial(-n, _k)/
            (factorial(_k)*gamma(_k + alpha + 1)), (_k, 0, n))/factorial(n))
    raises(ValueError, lambda: assoc_laguerre(-2.1, alpha, x))
    raises(ArgumentIndexError, lambda: assoc_laguerre(n, alpha, x).fdiff(1))
    raises(ArgumentIndexError, lambda: assoc_laguerre(n, alpha, x).fdiff(4))
