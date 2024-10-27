"""Tests for minimal polynomials. """

from sympy.core.function import expand
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (AlgebraicNumber, I, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.solvers.solveset import nonlinsolve
from sympy.geometry import Circle, intersection
from sympy.testing.pytest import raises, slow
from sympy.sets.sets import FiniteSet
from sympy.geometry.point import Point2D
from sympy.polys.numberfields.minpoly import (
    minimal_polynomial,
    _choose_factor,
    _minpoly_op_algebraic_element,
    _separate_sq,
    _minpoly_groebner,
)
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import (
    NotAlgebraic,
    GeneratorsError,
)

from sympy.polys.domains import QQ
from sympy.polys.rootoftools import rootof
from sympy.polys.polytools import degree

from sympy.abc import x, y, z

Q = Rational


def test_minimal_polynomial():
    assert minimal_polynomial(-7, x) == x + 7
    assert minimal_polynomial(-1, x) == x + 1
    assert minimal_polynomial( 0, x) == x
    assert minimal_polynomial( 1, x) == x - 1
    assert minimal_polynomial( 7, x) == x - 7

    assert minimal_polynomial(sqrt(2), x) == x**2 - 2
    assert minimal_polynomial(sqrt(5), x) == x**2 - 5
    assert minimal_polynomial(sqrt(6), x) == x**2 - 6

    assert minimal_polynomial(2*sqrt(2), x) == x**2 - 8
    assert minimal_polynomial(3*sqrt(5), x) == x**2 - 45
    assert minimal_polynomial(4*sqrt(6), x) == x**2 - 96

    assert minimal_polynomial(2*sqrt(2) + 3, x) == x**2 - 6*x + 1
    assert minimal_polynomial(3*sqrt(5) + 6, x) == x**2 - 12*x - 9
    assert minimal_polynomial(4*sqrt(6) + 7, x) == x**2 - 14*x - 47

    assert minimal_polynomial(2*sqrt(2) - 3, x) == x**2 + 6*x + 1
    assert minimal_polynomial(3*sqrt(5) - 6, x) == x**2 + 12*x - 9
    assert minimal_polynomial(4*sqrt(6) - 7, x) == x**2 + 14*x - 47

    assert minimal_polynomial(sqrt(1 + sqrt(6)), x) == x**4 - 2*x**2 - 5
    assert minimal_polynomial(sqrt(I + sqrt(6)), x) == x**8 - 10*x**4 + 49

    assert minimal_polynomial(2*I + sqrt(2 + I), x) == x**4 + 4*x**2 + 8*x + 37

    assert minimal_polynomial(sqrt(2) + sqrt(3), x) == x**4 - 10*x**2 + 1
    assert minimal_polynomial(
        sqrt(2) + sqrt(3) + sqrt(6), x) == x**4 - 22*x**2 - 48*x - 23

    a = 1 - 9*sqrt(2) + 7*sqrt(3)

    assert minimal_polynomial(
        1/a, x) == 392*x**4 - 1232*x**3 + 612*x**2 + 4*x - 1
    assert minimal_polynomial(
        1/sqrt(a), x) == 392*x**8 - 1232*x**6 + 612*x**4 + 4*x**2 - 1

    raises(NotAlgebraic, lambda: minimal_polynomial(oo, x))
    raises(NotAlgebraic, lambda: minimal_polynomial(2**y, x))
    raises(NotAlgebraic, lambda: minimal_polynomial(sin(1), x))

    assert minimal_polynomial(sqrt(2)).dummy_eq(x**2 - 2)
    assert minimal_polynomial(sqrt(2), x) == x**2 - 2

    assert minimal_polynomial(sqrt(2), polys=True) == Poly(x**2 - 2)
    assert minimal_polynomial(sqrt(2), x, polys=True) == Poly(x**2 - 2, domain='QQ')
    assert minimal_polynomial(sqrt(2), x, polys=True, compose=False) == Poly(x**2 - 2, domain='QQ')

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(3))

    assert minimal_polynomial(a, x) == x**2 - 2
    assert minimal_polynomial(b, x) == x**2 - 3

    assert minimal_polynomial(a, x, polys=True) == Poly(x**2 - 2, domain='QQ')
    assert minimal_polynomial(b, x, polys=True) == Poly(x**2 - 3, domain='QQ')

    assert minimal_polynomial(sqrt(a/2 + 17), x) == 2*x**4 - 68*x**2 + 577
    assert minimal_polynomial(sqrt(b/2 + 17), x) == 4*x**4 - 136*x**2 + 1153

    a, b = sqrt(2)/3 + 7, AlgebraicNumber(sqrt(2)/3 + 7)

    f = 81*x**8 - 2268*x**6 - 4536*x**5 + 22644*x**4 + 63216*x**3 - \
        31608*x**2 - 189648*x + 141358

    assert minimal_polynomial(sqrt(a) + sqrt(sqrt(a)), x) == f
    assert minimal_polynomial(sqrt(b) + sqrt(sqrt(b)), x) == f

    assert minimal_polynomial(
        a**Q(3, 2), x) == 729*x**4 - 506898*x**2 + 84604519

    # issue 5994
    eq = S('''
        -1/(800*sqrt(-1/240 + 1/(18000*(-1/17280000 +
        sqrt(15)*I/28800000)**(1/3)) + 2*(-1/17280000 +
        sqrt(15)*I/28800000)**(1/3)))''')
    assert minimal_polynomial(eq, x) == 8000*x**2 - 1

    ex = (sqrt(5)*sqrt(I)/(5*sqrt(1 + 125*I))
            + 25*sqrt(5)/(I**Q(5,2)*(1 + 125*I)**Q(3,2))
            + 3125*sqrt(5)/(I**Q(11,2)*(1 + 125*I)**Q(3,2))
            + 5*I*sqrt(1 - I/125))
    mp = minimal_polynomial(ex, x)
    assert mp == 25*x**4 + 5000*x**2 + 250016

    ex = 1 + sqrt(2) + sqrt(3)
    mp = minimal_polynomial(ex, x)
    assert mp == x**4 - 4*x**3 - 4*x**2 + 16*x - 8

    ex = 1/(1 + sqrt(2) + sqrt(3))
    mp = minimal_polynomial(ex, x)
    assert mp == 8*x**4 - 16*x**3 + 4*x**2 + 4*x - 1

    p = (expand((1 + sqrt(2) - 2*sqrt(3) + sqrt(7))**3))**Rational(1, 3)
    mp = minimal_polynomial(p, x)
    assert mp == x**8 - 8*x**7 - 56*x**6 + 448*x**5 + 480*x**4 - 5056*x**3 + 1984*x**2 + 7424*x - 3008
    p = expand((1 + sqrt(2) - 2*sqrt(3) + sqrt(7))**3)
    mp = minimal_polynomial(p, x)
    assert mp == x**8 - 512*x**7 - 118208*x**6 + 31131136*x**5 + 647362560*x**4 - 56026611712*x**3 + 116994310144*x**2 + 404854931456*x - 27216576512

    assert minimal_polynomial(S("-sqrt(5)/2 - 1/2 + (-sqrt(5)/2 - 1/2)**2"), x) == x - 1
    a = 1 + sqrt(2)
    assert minimal_polynomial((a*sqrt(2) + a)**3, x) == x**2 - 198*x + 1

    p = 1/(1 + sqrt(2) + sqrt(3))
    assert minimal_polynomial(p, x, compose=False) == 8*x**4 - 16*x**3 + 4*x**2 + 4*x - 1

    p = 2/(1 + sqrt(2) + sqrt(3))
    assert minimal_polynomial(p, x, compose=False) == x**4 - 4*x**3 + 2*x**2 + 4*x - 2

    assert minimal_polynomial(1 + sqrt(2)*I, x, compose=False) == x**2 - 2*x + 3
    assert minimal_polynomial(1/(1 + sqrt(2)) + 1, x, compose=False) == x**2 - 2
    assert minimal_polynomial(sqrt(2)*I + I*(1 + sqrt(2)), x,
            compose=False) ==  x**4 + 18*x**2 + 49

    # minimal polynomial of I
    assert minimal_polynomial(I, x, domain=QQ.algebraic_field(I)) == x - I
    K = QQ.algebraic_field(I*(sqrt(2) + 1))
    assert minimal_polynomial(I, x, domain=K) == x - I
    assert minimal_polynomial(I, x, domain=QQ) == x**2 + 1
    assert minimal_polynomial(I, x, domain='QQ(y)') == x**2 + 1

    #issue 11553
    assert minimal_polynomial(GoldenRatio, x) == x**2 - x - 1
    assert minimal_polynomial(TribonacciConstant + 3, x) == x**3 - 10*x**2 + 32*x - 34
    assert minimal_polynomial(GoldenRatio, x, domain=QQ.algebraic_field(sqrt(5))) == \
            2*x - sqrt(5) - 1
    assert minimal_polynomial(TribonacciConstant, x, domain=QQ.algebraic_field(cbrt(19 - 3*sqrt(33)))) == \
    48*x - 19*(19 - 3*sqrt(33))**Rational(2, 3) - 3*sqrt(33)*(19 - 3*sqrt(33))**Rational(2, 3) \
    - 16*(19 - 3*sqrt(33))**Rational(1, 3) - 16

    # AlgebraicNumber with an alias.
    # Wester H24
    phi = AlgebraicNumber(S.GoldenRatio.expand(func=True), alias='phi')
    assert minimal_polynomial(phi, x) == x**2 - x - 1


def test_minimal_polynomial_issue_19732():
    # https://github.com/sympy/sympy/issues/19732
    expr = (-280898097948878450887044002323982963174671632174995451265117559518123750720061943079105185551006003416773064305074191140286225850817291393988597615/(-488144716373031204149459129212782509078221364279079444636386844223983756114492222145074506571622290776245390771587888364089507840000000*sqrt(238368341569)*sqrt(S(11918417078450)/63568729
    - 24411360*sqrt(238368341569)/63568729) +
    238326799225996604451373809274348704114327860564921529846705817404208077866956345381951726531296652901169111729944612727047670549086208000000*sqrt(S(11918417078450)/63568729
        - 24411360*sqrt(238368341569)/63568729)) -
    180561807339168676696180573852937120123827201075968945871075967679148461189459480842956689723484024031016208588658753107/(-59358007109636562851035004992802812513575019937126272896569856090962677491318275291141463850327474176000000*sqrt(238368341569)*sqrt(S(11918417078450)/63568729
        - 24411360*sqrt(238368341569)/63568729) +
        28980348180319251787320809875930301310576055074938369007463004788921613896002936637780993064387310446267596800000*sqrt(S(11918417078450)/63568729
            - 24411360*sqrt(238368341569)/63568729)))
    poly = (2151288870990266634727173620565483054187142169311153766675688628985237817262915166497766867289157986631135400926544697981091151416655364879773546003475813114962656742744975460025956167152918469472166170500512008351638710934022160294849059721218824490226159355197136265032810944357335461128949781377875451881300105989490353140886315677977149440000000000000000000000*x**4
            - 5773274155644072033773937864114266313663195672820501581692669271302387257492905909558846459600429795784309388968498783843631580008547382703258503404023153694528041873101120067477617592651525155101107144042679962433039557235772239171616433004024998230222455940044709064078962397144550855715640331680262171410099614469231080995436488414164502751395405398078353242072696360734131090111239998110773292915337556205692674790561090109440000000000000*x**2
            + 211295968822207088328287206509522887719741955693091053353263782924470627623790749534705683380138972642560898936171035770539616881000369889020398551821767092685775598633794696371561234818461806577723412581353857653829324364446419444210520602157621008010129702779407422072249192199762604318993590841636967747488049176548615614290254356975376588506729604345612047361483789518445332415765213187893207704958013682516462853001964919444736320672860140355089)
    assert minimal_polynomial(expr, x) == poly


def test_minimal_polynomial_hi_prec():
    p = 1/sqrt(1 - 9*sqrt(2) + 7*sqrt(3) + Rational(1, 10)**30)
    mp = minimal_polynomial(p, x)
    # checked with Wolfram Alpha
    assert mp.coeff(x**6) == -1232000000000000000000000000001223999999999999999999999999999987999999999999999999999999999996000000000000000000000000000000


def test_minimal_polynomial_sq():
    from sympy.core.add import Add
    from sympy.core.function import expand_multinomial
    p = expand_multinomial((1 + 5*sqrt(2) + 2*sqrt(3))**3)
    mp = minimal_polynomial(p**Rational(1, 3), x)
    assert mp == x**4 - 4*x**3 - 118*x**2 + 244*x + 1321
    p = expand_multinomial((1 + sqrt(2) - 2*sqrt(3) + sqrt(7))**3)
    mp = minimal_polynomial(p**Rational(1, 3), x)
    assert mp == x**8 - 8*x**7 - 56*x**6 + 448*x**5 + 480*x**4 - 5056*x**3 + 1984*x**2 + 7424*x - 3008
    p = Add(*[sqrt(i) for i in range(1, 12)])
    mp = minimal_polynomial(p, x)
    assert mp.subs({x: 0}) == -71965773323122507776


def test_minpoly_compose():
    # issue 6868
    eq = S('''
        -1/(800*sqrt(-1/240 + 1/(18000*(-1/17280000 +
        sqrt(15)*I/28800000)**(1/3)) + 2*(-1/17280000 +
        sqrt(15)*I/28800000)**(1/3)))''')
    mp = minimal_polynomial(eq + 3, x)
    assert mp == 8000*x**2 - 48000*x + 71999

    # issue 5888
    assert minimal_polynomial(exp(I*pi/8), x) == x**8 + 1

    mp = minimal_polynomial(sin(pi/7) + sqrt(2), x)
    assert mp == 4096*x**12 - 63488*x**10 + 351488*x**8 - 826496*x**6 + \
        770912*x**4 - 268432*x**2 + 28561
    mp = minimal_polynomial(cos(pi/7) + sqrt(2), x)
    assert mp == 64*x**6 - 64*x**5 - 432*x**4 + 304*x**3 + 712*x**2 - \
            232*x - 239
    mp = minimal_polynomial(exp(I*pi/7) + sqrt(2), x)
    assert mp == x**12 - 2*x**11 - 9*x**10 + 16*x**9 + 43*x**8 - 70*x**7 - 97*x**6 + 126*x**5 + 211*x**4 - 212*x**3 - 37*x**2 + 142*x + 127

    mp = minimal_polynomial(sin(pi/7) + sqrt(2), x)
    assert mp == 4096*x**12 - 63488*x**10 + 351488*x**8 - 826496*x**6 + \
        770912*x**4 - 268432*x**2 + 28561
    mp = minimal_polynomial(cos(pi/7) + sqrt(2), x)
    assert mp == 64*x**6 - 64*x**5 - 432*x**4 + 304*x**3 + 712*x**2 - \
            232*x - 239
    mp = minimal_polynomial(exp(I*pi/7) + sqrt(2), x)
    assert mp == x**12 - 2*x**11 - 9*x**10 + 16*x**9 + 43*x**8 - 70*x**7 - 97*x**6 + 126*x**5 + 211*x**4 - 212*x**3 - 37*x**2 + 142*x + 127

    mp = minimal_polynomial(exp(I*pi*Rational(2, 7)), x)
    assert mp == x**6 + x**5 + x**4 + x**3 + x**2 + x + 1
    mp = minimal_polynomial(exp(I*pi*Rational(2, 15)), x)
    assert mp == x**8 - x**7 + x**5 - x**4 + x**3 - x + 1
    mp = minimal_polynomial(cos(pi*Rational(2, 7)), x)
    assert mp == 8*x**3 + 4*x**2 - 4*x - 1
    mp = minimal_polynomial(sin(pi*Rational(2, 7)), x)
    ex = (5*cos(pi*Rational(2, 7)) - 7)/(9*cos(pi/7) - 5*cos(pi*Rational(3, 7)))
    mp = minimal_polynomial(ex, x)
    assert mp == x**3 + 2*x**2 - x - 1
    assert minimal_polynomial(-1/(2*cos(pi/7)), x) == x**3 + 2*x**2 - x - 1
    assert minimal_polynomial(sin(pi*Rational(2, 15)), x) == \
            256*x**8 - 448*x**6 + 224*x**4 - 32*x**2 + 1
    assert minimal_polynomial(sin(pi*Rational(5, 14)), x) == 8*x**3 - 4*x**2 - 4*x + 1
    assert minimal_polynomial(cos(pi/15), x) == 16*x**4 + 8*x**3 - 16*x**2 - 8*x + 1

    ex = rootof(x**3 +x*4 + 1, 0)
    mp = minimal_polynomial(ex, x)
    assert mp == x**3 + 4*x + 1
    mp = minimal_polynomial(ex + 1, x)
    assert mp == x**3 - 3*x**2 + 7*x - 4
    assert minimal_polynomial(exp(I*pi/3), x) == x**2 - x + 1
    assert minimal_polynomial(exp(I*pi/4), x) == x**4 + 1
    assert minimal_polynomial(exp(I*pi/6), x) == x**4 - x**2 + 1
    assert minimal_polynomial(exp(I*pi/9), x) == x**6 - x**3 + 1
    assert minimal_polynomial(exp(I*pi/10), x) == x**8 - x**6 + x**4 - x**2 + 1
    assert minimal_polynomial(sin(pi/9), x) == 64*x**6 - 96*x**4 + 36*x**2 - 3
    assert minimal_polynomial(sin(pi/11), x) == 1024*x**10 - 2816*x**8 + \
            2816*x**6 - 1232*x**4 + 220*x**2 - 11
    assert minimal_polynomial(sin(pi/21), x) == 4096*x**12 - 11264*x**10 + \
           11264*x**8 - 4992*x**6 + 960*x**4 - 64*x**2 + 1
    assert minimal_polynomial(cos(pi/9), x) == 8*x**3 - 6*x - 1

    ex = 2**Rational(1, 3)*exp(2*I*pi/3)
    assert minimal_polynomial(ex, x) == x**3 - 2

    raises(NotAlgebraic, lambda: minimal_polynomial(cos(pi*sqrt(2)), x))
    raises(NotAlgebraic, lambda: minimal_polynomial(sin(pi*sqrt(2)), x))
    raises(NotAlgebraic, lambda: minimal_polynomial(exp(1.618*I*pi), x))
    raises(NotAlgebraic, lambda: minimal_polynomial(exp(I*pi*sqrt(2)), x))

    # issue 5934
    ex = 1/(-36000 - 7200*sqrt(5) + (12*sqrt(10)*sqrt(sqrt(5) + 5) +
        24*sqrt(10)*sqrt(-sqrt(5) + 5))**2) + 1
    raises(ZeroDivisionError, lambda: minimal_polynomial(ex, x))

    ex = sqrt(1 + 2**Rational(1,3)) + sqrt(1 + 2**Rational(1,4)) + sqrt(2)
    mp = minimal_polynomial(ex, x)
    assert degree(mp) == 48 and mp.subs({x:0}) == -16630256576

    ex = tan(pi/5, evaluate=False)
    mp = minimal_polynomial(ex, x)
    assert mp == x**4 - 10*x**2 + 5
    assert mp.subs(x, tan(pi/5)).is_zero

    ex = tan(pi/6, evaluate=False)
    mp = minimal_polynomial(ex, x)
    assert mp == 3*x**2 - 1
    assert mp.subs(x, tan(pi/6)).is_zero

    ex = tan(pi/10, evaluate=False)
    mp = minimal_polynomial(ex, x)
    assert mp == 5*x**4 - 10*x**2 + 1
    assert mp.subs(x, tan(pi/10)).is_zero

    raises(NotAlgebraic, lambda: minimal_polynomial(tan(pi*sqrt(2)), x))


def test_minpoly_issue_7113():
    # see discussion in https://github.com/sympy/sympy/pull/2234
    from sympy.simplify.simplify import nsimplify
    r = nsimplify(pi, tolerance=0.000000001)
    mp = minimal_polynomial(r, x)
    assert mp == 1768292677839237920489538677417507171630859375*x**109 - \
    2734577732179183863586489182929671773182898498218854181690460140337930774573792597743853652058046464


def test_minpoly_issue_23677():
    r1 = CRootOf(4000000*x**3 - 239960000*x**2 + 4782399900*x - 31663998001, 0)
    r2 = CRootOf(4000000*x**3 - 239960000*x**2 + 4782399900*x - 31663998001, 1)
    num = (7680000000000000000*r1**4*r2**4 - 614323200000000000000*r1**4*r2**3
            + 18458112576000000000000*r1**4*r2**2 - 246896663036160000000000*r1**4*r2
            + 1240473830323209600000000*r1**4 - 614323200000000000000*r1**3*r2**4
            - 1476464424954240000000000*r1**3*r2**2 - 99225501687553535904000000*r1**3
            + 18458112576000000000000*r1**2*r2**4 - 1476464424954240000000000*r1**2*r2**3
            - 593391458458356671712000000*r1**2*r2 + 2981354896834339226880720000*r1**2
            - 246896663036160000000000*r1*r2**4 - 593391458458356671712000000*r1*r2**2
            - 39878756418031796275267195200*r1 + 1240473830323209600000000*r2**4
            - 99225501687553535904000000*r2**3 + 2981354896834339226880720000*r2**2 -
            39878756418031796275267195200*r2 + 200361370275616536577343808012)
    mp = (x**3 + 59426520028417434406408556687919*x**2 +
        1161475464966574421163316896737773190861975156439163671112508400*x +
        7467465541178623874454517208254940823818304424383315270991298807299003671748074773558707779600)
    assert minimal_polynomial(num, x) == mp


def test_minpoly_issue_7574():
    ex = -(-1)**Rational(1, 3) + (-1)**Rational(2,3)
    assert minimal_polynomial(ex, x) == x + 1


def test_choose_factor():
    # Test that this does not enter an infinite loop:
    bad_factors = [Poly(x-2, x), Poly(x+2, x)]
    raises(NotImplementedError, lambda: _choose_factor(bad_factors, x, sqrt(3)))


def test_minpoly_fraction_field():
    assert minimal_polynomial(1/x, y) == -x*y + 1
    assert minimal_polynomial(1 / (x + 1), y) == (x + 1)*y - 1

    assert minimal_polynomial(sqrt(x), y) == y**2 - x
    assert minimal_polynomial(sqrt(x + 1), y) == y**2 - x - 1
    assert minimal_polynomial(sqrt(x) / x, y) == x*y**2 - 1
    assert minimal_polynomial(sqrt(2) * sqrt(x), y) == y**2 - 2 * x
    assert minimal_polynomial(sqrt(2) + sqrt(x), y) == \
        y**4 + (-2*x - 4)*y**2 + x**2 - 4*x + 4

    assert minimal_polynomial(x**Rational(1,3), y) == y**3 - x
    assert minimal_polynomial(x**Rational(1,3) + sqrt(x), y) == \
        y**6 - 3*x*y**4 - 2*x*y**3 + 3*x**2*y**2 - 6*x**2*y - x**3 + x**2

    assert minimal_polynomial(sqrt(x) / z, y) == z**2*y**2 - x
    assert minimal_polynomial(sqrt(x) / (z + 1), y) == (z**2 + 2*z + 1)*y**2 - x

    assert minimal_polynomial(1/x, y, polys=True) == Poly(-x*y + 1, y, domain='ZZ(x)')
    assert minimal_polynomial(1 / (x + 1), y, polys=True) == \
        Poly((x + 1)*y - 1, y, domain='ZZ(x)')
    assert minimal_polynomial(sqrt(x), y, polys=True) == Poly(y**2 - x, y, domain='ZZ(x)')
    assert minimal_polynomial(sqrt(x) / z, y, polys=True) == \
        Poly(z**2*y**2 - x, y, domain='ZZ(x, z)')

    # this is (sqrt(1 + x**3)/x).integrate(x).diff(x) - sqrt(1 + x**3)/x
    a = sqrt(x)/sqrt(1 + x**(-3)) - sqrt(x**3 + 1)/x + 1/(x**Rational(5, 2)* \
        (1 + x**(-3))**Rational(3, 2)) + 1/(x**Rational(11, 2)*(1 + x**(-3))**Rational(3, 2))

    assert minimal_polynomial(a, y) == y

    raises(NotAlgebraic, lambda: minimal_polynomial(exp(x), y))
    raises(GeneratorsError, lambda: minimal_polynomial(sqrt(x), x))
    raises(GeneratorsError, lambda: minimal_polynomial(sqrt(x) - y, x))
    raises(NotImplementedError, lambda: minimal_polynomial(sqrt(x), y, compose=False))

@slow
def test_minpoly_fraction_field_slow():
    assert minimal_polynomial(minimal_polynomial(sqrt(x**Rational(1,5) - 1),
        y).subs(y, sqrt(x**Rational(1,5) - 1)), z) == z

def test_minpoly_domain():
    assert minimal_polynomial(sqrt(2), x, domain=QQ.algebraic_field(sqrt(2))) == \
        x - sqrt(2)
    assert minimal_polynomial(sqrt(8), x, domain=QQ.algebraic_field(sqrt(2))) == \
        x - 2*sqrt(2)
    assert minimal_polynomial(sqrt(Rational(3,2)), x,
        domain=QQ.algebraic_field(sqrt(2))) == 2*x**2 - 3

    raises(NotAlgebraic, lambda: minimal_polynomial(y, x, domain=QQ))


def test_issue_14831():
    a = -2*sqrt(2)*sqrt(12*sqrt(2) + 17)
    assert minimal_polynomial(a, x) == x**2 + 16*x - 8
    e = (-3*sqrt(12*sqrt(2) + 17) + 12*sqrt(2) +
         17 - 2*sqrt(2)*sqrt(12*sqrt(2) + 17))
    assert minimal_polynomial(e, x) == x


def test_issue_18248():
    assert nonlinsolve([x*y**3-sqrt(2)/3, x*y**6-4/(9*(sqrt(3)))],x,y) == \
            FiniteSet((sqrt(3)/2, sqrt(6)/3), (sqrt(3)/2, -sqrt(6)/6 - sqrt(2)*I/2),
            (sqrt(3)/2, -sqrt(6)/6 + sqrt(2)*I/2))


def test_issue_13230():
    c1 = Circle(Point2D(3, sqrt(5)), 5)
    c2 = Circle(Point2D(4, sqrt(7)), 6)
    assert intersection(c1, c2) == [Point2D(-1 + (-sqrt(7) + sqrt(5))*(-2*sqrt(7)/29
    + 9*sqrt(5)/29 + sqrt(196*sqrt(35) + 1941)/29), -2*sqrt(7)/29 + 9*sqrt(5)/29
    + sqrt(196*sqrt(35) + 1941)/29), Point2D(-1 + (-sqrt(7) + sqrt(5))*(-sqrt(196*sqrt(35)
    + 1941)/29 - 2*sqrt(7)/29 + 9*sqrt(5)/29), -sqrt(196*sqrt(35) + 1941)/29 - 2*sqrt(7)/29 + 9*sqrt(5)/29)]

def test_issue_19760():
    e = 1/(sqrt(1 + sqrt(2)) - sqrt(2)*sqrt(1 + sqrt(2))) + 1
    mp_expected = x**4 - 4*x**3 + 4*x**2 - 2

    for comp in (True, False):
        mp = Poly(minimal_polynomial(e, compose=comp))
        assert mp(x) == mp_expected, "minimal_polynomial(e, compose=%s) = %s; %s expected" % (comp, mp(x), mp_expected)


def test_issue_20163():
    assert apart(1/(x**6+1), extension=[sqrt(3), I]) == \
        (sqrt(3) + I)/(2*x + sqrt(3) + I)/6 + \
        (sqrt(3) - I)/(2*x + sqrt(3) - I)/6 - \
        (sqrt(3) - I)/(2*x - sqrt(3) + I)/6 - \
        (sqrt(3) + I)/(2*x - sqrt(3) - I)/6 + \
        I/(x + I)/6 - I/(x - I)/6


def test_issue_22559():
    alpha = AlgebraicNumber(sqrt(2))
    assert minimal_polynomial(alpha**3, x) == x**2 - 8


def test_issue_22561():
    a = AlgebraicNumber(sqrt(2) + sqrt(3), [S(1) / 2, 0, S(-9) / 2, 0], gen=x)
    assert a.as_expr() == sqrt(2)
    assert minimal_polynomial(a, x) == x**2 - 2
    assert minimal_polynomial(a**3, x) == x**2 - 8


def test_separate_sq_not_impl():
    raises(NotImplementedError, lambda: _separate_sq(x**(S(1)/3) + x))


def test_minpoly_op_algebraic_element_not_impl():
    raises(NotImplementedError,
           lambda: _minpoly_op_algebraic_element(Pow, sqrt(2), sqrt(3), x, QQ))


def test_minpoly_groebner():
    assert _minpoly_groebner(S(2)/3, x, Poly) == 3*x - 2
    assert _minpoly_groebner(
        (sqrt(2) + 3)*(sqrt(2) + 1), x, Poly) == x**2 - 10*x - 7
    assert _minpoly_groebner((sqrt(2) + 3)**(S(1)/3)*(sqrt(2) + 1)**(S(1)/3),
                             x, Poly) == x**6 - 10*x**3 - 7
    assert _minpoly_groebner((sqrt(2) + 3)**(-S(1)/3)*(sqrt(2) + 1)**(S(1)/3),
                             x, Poly) == 7*x**6 - 2*x**3 - 1
    raises(NotAlgebraic, lambda: _minpoly_groebner(pi**2, x, Poly))
