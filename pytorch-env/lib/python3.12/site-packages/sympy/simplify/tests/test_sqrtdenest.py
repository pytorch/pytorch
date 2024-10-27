from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
    _subsets as subsets, _sqrt_numeric_denest)

r2, r3, r5, r6, r7, r10, r15, r29 = [sqrt(x) for x in (2, 3, 5, 6, 7, 10,
                                          15, 29)]


def test_sqrtdenest():
    d = {sqrt(5 + 2 * r6): r2 + r3,
        sqrt(5. + 2 * r6): sqrt(5. + 2 * r6),
        sqrt(5. + 4*sqrt(5 + 2 * r6)): sqrt(5.0 + 4*r2 + 4*r3),
        sqrt(r2): sqrt(r2),
        sqrt(5 + r7): sqrt(5 + r7),
        sqrt(3 + sqrt(5 + 2*r7)):
         3*r2*(5 + 2*r7)**Rational(1, 4)/(2*sqrt(6 + 3*r7)) +
         r2*sqrt(6 + 3*r7)/(2*(5 + 2*r7)**Rational(1, 4)),
        sqrt(3 + 2*r3): 3**Rational(3, 4)*(r6/2 + 3*r2/2)/3}
    for i in d:
        assert sqrtdenest(i) == d[i], i


def test_sqrtdenest2():
    assert sqrtdenest(sqrt(16 - 2*r29 + 2*sqrt(55 - 10*r29))) == \
        r5 + sqrt(11 - 2*r29)
    e = sqrt(-r5 + sqrt(-2*r29 + 2*sqrt(-10*r29 + 55) + 16))
    assert sqrtdenest(e) == root(-2*r29 + 11, 4)
    r = sqrt(1 + r7)
    assert sqrtdenest(sqrt(1 + r)) == sqrt(1 + r)
    e = sqrt(((1 + sqrt(1 + 2*sqrt(3 + r2 + r5)))**2).expand())
    assert sqrtdenest(e) == 1 + sqrt(1 + 2*sqrt(r2 + r5 + 3))

    assert sqrtdenest(sqrt(5*r3 + 6*r2)) == \
        sqrt(2)*root(3, 4) + root(3, 4)**3

    assert sqrtdenest(sqrt(((1 + r5 + sqrt(1 + r3))**2).expand())) == \
        1 + r5 + sqrt(1 + r3)

    assert sqrtdenest(sqrt(((1 + r5 + r7 + sqrt(1 + r3))**2).expand())) == \
        1 + sqrt(1 + r3) + r5 + r7

    e = sqrt(((1 + cos(2) + cos(3) + sqrt(1 + r3))**2).expand())
    assert sqrtdenest(e) == cos(3) + cos(2) + 1 + sqrt(1 + r3)

    e = sqrt(-2*r10 + 2*r2*sqrt(-2*r10 + 11) + 14)
    assert sqrtdenest(e) == sqrt(-2*r10 - 2*r2 + 4*r5 + 14)

    # check that the result is not more complicated than the input
    z = sqrt(-2*r29 + cos(2) + 2*sqrt(-10*r29 + 55) + 16)
    assert sqrtdenest(z) == z

    assert sqrtdenest(sqrt(r6 + sqrt(15))) == sqrt(r6 + sqrt(15))

    z = sqrt(15 - 2*sqrt(31) + 2*sqrt(55 - 10*r29))
    assert sqrtdenest(z) == z


def test_sqrtdenest_rec():
    assert sqrtdenest(sqrt(-4*sqrt(14) - 2*r6 + 4*sqrt(21) + 33)) == \
        -r2 + r3 + 2*r7
    assert sqrtdenest(sqrt(-28*r7 - 14*r5 + 4*sqrt(35) + 82)) == \
        -7 + r5 + 2*r7
    assert sqrtdenest(sqrt(6*r2/11 + 2*sqrt(22)/11 + 6*sqrt(11)/11 + 2)) == \
        sqrt(11)*(r2 + 3 + sqrt(11))/11
    assert sqrtdenest(sqrt(468*r3 + 3024*r2 + 2912*r6 + 19735)) == \
        9*r3 + 26 + 56*r6
    z = sqrt(-490*r3 - 98*sqrt(115) - 98*sqrt(345) - 2107)
    assert sqrtdenest(z) == sqrt(-1)*(7*r5 + 7*r15 + 7*sqrt(23))
    z = sqrt(-4*sqrt(14) - 2*r6 + 4*sqrt(21) + 34)
    assert sqrtdenest(z) == z
    assert sqrtdenest(sqrt(-8*r2 - 2*r5 + 18)) == -r10 + 1 + r2 + r5
    assert sqrtdenest(sqrt(8*r2 + 2*r5 - 18)) == \
        sqrt(-1)*(-r10 + 1 + r2 + r5)
    assert sqrtdenest(sqrt(8*r2/3 + 14*r5/3 + Rational(154, 9))) == \
        -r10/3 + r2 + r5 + 3
    assert sqrtdenest(sqrt(sqrt(2*r6 + 5) + sqrt(2*r7 + 8))) == \
        sqrt(1 + r2 + r3 + r7)
    assert sqrtdenest(sqrt(4*r15 + 8*r5 + 12*r3 + 24)) == 1 + r3 + r5 + r15

    w = 1 + r2 + r3 + r5 + r7
    assert sqrtdenest(sqrt((w**2).expand())) == w
    z = sqrt((w**2).expand() + 1)
    assert sqrtdenest(z) == z

    z = sqrt(2*r10 + 6*r2 + 4*r5 + 12 + 10*r15 + 30*r3)
    assert sqrtdenest(z) == z


def test_issue_6241():
    z = sqrt( -320 + 32*sqrt(5) + 64*r15)
    assert sqrtdenest(z) == z


def test_sqrtdenest3():
    z = sqrt(13 - 2*r10 + 2*r2*sqrt(-2*r10 + 11))
    assert sqrtdenest(z) == -1 + r2 + r10
    assert sqrtdenest(z, max_iter=1) == -1 + sqrt(2) + sqrt(10)
    z = sqrt(sqrt(r2 + 2) + 2)
    assert sqrtdenest(z) == z
    assert sqrtdenest(sqrt(-2*r10 + 4*r2*sqrt(-2*r10 + 11) + 20)) == \
        sqrt(-2*r10 - 4*r2 + 8*r5 + 20)
    assert sqrtdenest(sqrt((112 + 70*r2) + (46 + 34*r2)*r5)) == \
        r10 + 5 + 4*r2 + 3*r5
    z = sqrt(5 + sqrt(2*r6 + 5)*sqrt(-2*r29 + 2*sqrt(-10*r29 + 55) + 16))
    r = sqrt(-2*r29 + 11)
    assert sqrtdenest(z) == sqrt(r2*r + r3*r + r10 + r15 + 5)

    n = sqrt(2*r6/7 + 2*r7/7 + 2*sqrt(42)/7 + 2)
    d = sqrt(16 - 2*r29 + 2*sqrt(55 - 10*r29))
    assert sqrtdenest(n/d) == r7*(1 + r6 + r7)/(Mul(7, (sqrt(-2*r29 + 11) + r5),
                                                    evaluate=False))


def test_sqrtdenest4():
    # see Denest_en.pdf in https://github.com/sympy/sympy/issues/3192
    z = sqrt(8 - r2*sqrt(5 - r5) - sqrt(3)*(1 + r5))
    z1 = sqrtdenest(z)
    c = sqrt(-r5 + 5)
    z1 = ((-r15*c - r3*c + c + r5*c - r6 - r2 + r10 + sqrt(30))/4).expand()
    assert sqrtdenest(z) == z1

    z = sqrt(2*r2*sqrt(r2 + 2) + 5*r2 + 4*sqrt(r2 + 2) + 8)
    assert sqrtdenest(z) == r2 + sqrt(r2 + 2) + 2

    w = 2 + r2 + r3 + (1 + r3)*sqrt(2 + r2 + 5*r3)
    z = sqrt((w**2).expand())
    assert sqrtdenest(z) == w.expand()


def test_sqrt_symbolic_denest():
    x = Symbol('x')
    z = sqrt(((1 + sqrt(sqrt(2 + x) + 3))**2).expand())
    assert sqrtdenest(z) == sqrt((1 + sqrt(sqrt(2 + x) + 3))**2)
    z = sqrt(((1 + sqrt(sqrt(2 + cos(1)) + 3))**2).expand())
    assert sqrtdenest(z) == 1 + sqrt(sqrt(2 + cos(1)) + 3)
    z = ((1 + cos(2))**4 + 1).expand()
    assert sqrtdenest(z) == z
    z = sqrt(((1 + sqrt(sqrt(2 + cos(3*x)) + 3))**2 + 1).expand())
    assert sqrtdenest(z) == z
    c = cos(3)
    c2 = c**2
    assert sqrtdenest(sqrt(2*sqrt(1 + r3)*c + c2 + 1 + r3*c2)) == \
        -1 - sqrt(1 + r3)*c
    ra = sqrt(1 + r3)
    z = sqrt(20*ra*sqrt(3 + 3*r3) + 12*r3*ra*sqrt(3 + 3*r3) + 64*r3 + 112)
    assert sqrtdenest(z) == z


def test_issue_5857():
    from sympy.abc import x, y
    z = sqrt(1/(4*r3 + 7) + 1)
    ans = (r2 + r6)/(r3 + 2)
    assert sqrtdenest(z) == ans
    assert sqrtdenest(1 + z) == 1 + ans
    assert sqrtdenest(Integral(z + 1, (x, 1, 2))) == \
        Integral(1 + ans, (x, 1, 2))
    assert sqrtdenest(x + sqrt(y)) == x + sqrt(y)
    ans = (r2 + r6)/(r3 + 2)
    assert sqrtdenest(z) == ans
    assert sqrtdenest(1 + z) == 1 + ans
    assert sqrtdenest(Integral(z + 1, (x, 1, 2))) == \
        Integral(1 + ans, (x, 1, 2))
    assert sqrtdenest(x + sqrt(y)) == x + sqrt(y)


def test_subsets():
    assert subsets(1) == [[1]]
    assert subsets(4) == [
        [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0],
        [0, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1],
        [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]]


def test_issue_5653():
    assert sqrtdenest(
        sqrt(2 + sqrt(2 + sqrt(2)))) == sqrt(2 + sqrt(2 + sqrt(2)))

def test_issue_12420():
    assert sqrtdenest((3 - sqrt(2)*sqrt(4 + 3*I) + 3*I)/2) == I
    e = 3 - sqrt(2)*sqrt(4 + I) + 3*I
    assert sqrtdenest(e) == e

def test_sqrt_ratcomb():
    assert sqrtdenest(sqrt(1 + r3) + sqrt(3 + 3*r3) - sqrt(10 + 6*r3)) == 0

def test_issue_18041():
    e = -sqrt(-2 + 2*sqrt(3)*I)
    assert sqrtdenest(e) == -1 - sqrt(3)*I

def test_issue_19914():
    a = Integer(-8)
    b = Integer(-1)
    r = Integer(63)
    d2 = a*a - b*b*r

    assert _sqrt_numeric_denest(a, b, r, d2) == \
        sqrt(14)*I/2 + 3*sqrt(2)*I/2
    assert sqrtdenest(sqrt(-8-sqrt(63))) == sqrt(14)*I/2 + 3*sqrt(2)*I/2
