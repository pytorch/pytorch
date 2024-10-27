from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.parameters import evaluate
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.hyperbolic import (cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, cot, csc, sec, sin, tan)
from sympy.simplify.powsimp import powsimp
from sympy.simplify.fu import (
    L, TR1, TR10, TR10i, TR11, _TR11, TR12, TR12i, TR13, TR14, TR15, TR16,
    TR111, TR2, TR2i, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TRmorrie, _TR56 as T,
    TRpower, hyper_as_trig, fu, process_common_addends, trig_split,
    as_f_sign_1)
from sympy.core.random import verify_numerically
from sympy.abc import a, b, c, x, y, z


def test_TR1():
    assert TR1(2*csc(x) + sec(x)) == 1/cos(x) + 2/sin(x)


def test_TR2():
    assert TR2(tan(x)) == sin(x)/cos(x)
    assert TR2(cot(x)) == cos(x)/sin(x)
    assert TR2(tan(tan(x) - sin(x)/cos(x))) == 0


def test_TR2i():
    # just a reminder that ratios of powers only simplify if both
    # numerator and denominator satisfy the condition that each
    # has a positive base or an integer exponent; e.g. the following,
    # at y=-1, x=1/2 gives sqrt(2)*I != -sqrt(2)*I
    assert powsimp(2**x/y**x) != (2/y)**x

    assert TR2i(sin(x)/cos(x)) == tan(x)
    assert TR2i(sin(x)*sin(y)/cos(x)) == tan(x)*sin(y)
    assert TR2i(1/(sin(x)/cos(x))) == 1/tan(x)
    assert TR2i(1/(sin(x)*sin(y)/cos(x))) == 1/tan(x)/sin(y)
    assert TR2i(sin(x)/2/(cos(x) + 1)) == sin(x)/(cos(x) + 1)/2

    assert TR2i(sin(x)/2/(cos(x) + 1), half=True) == tan(x/2)/2
    assert TR2i(sin(1)/(cos(1) + 1), half=True) == tan(S.Half)
    assert TR2i(sin(2)/(cos(2) + 1), half=True) == tan(1)
    assert TR2i(sin(4)/(cos(4) + 1), half=True) == tan(2)
    assert TR2i(sin(5)/(cos(5) + 1), half=True) == tan(5*S.Half)
    assert TR2i((cos(1) + 1)/sin(1), half=True) == 1/tan(S.Half)
    assert TR2i((cos(2) + 1)/sin(2), half=True) == 1/tan(1)
    assert TR2i((cos(4) + 1)/sin(4), half=True) == 1/tan(2)
    assert TR2i((cos(5) + 1)/sin(5), half=True) == 1/tan(5*S.Half)
    assert TR2i((cos(1) + 1)**(-a)*sin(1)**a, half=True) == tan(S.Half)**a
    assert TR2i((cos(2) + 1)**(-a)*sin(2)**a, half=True) == tan(1)**a
    assert TR2i((cos(4) + 1)**(-a)*sin(4)**a, half=True) == (cos(4) + 1)**(-a)*sin(4)**a
    assert TR2i((cos(5) + 1)**(-a)*sin(5)**a, half=True) == (cos(5) + 1)**(-a)*sin(5)**a
    assert TR2i((cos(1) + 1)**a*sin(1)**(-a), half=True) == tan(S.Half)**(-a)
    assert TR2i((cos(2) + 1)**a*sin(2)**(-a), half=True) == tan(1)**(-a)
    assert TR2i((cos(4) + 1)**a*sin(4)**(-a), half=True) == (cos(4) + 1)**a*sin(4)**(-a)
    assert TR2i((cos(5) + 1)**a*sin(5)**(-a), half=True) == (cos(5) + 1)**a*sin(5)**(-a)

    i = symbols('i', integer=True)
    assert TR2i(((cos(5) + 1)**i*sin(5)**(-i)), half=True) == tan(5*S.Half)**(-i)
    assert TR2i(1/((cos(5) + 1)**i*sin(5)**(-i)), half=True) == tan(5*S.Half)**i


def test_TR3():
    assert TR3(cos(y - x*(y - x))) == cos(x*(x - y) + y)
    assert cos(pi/2 + x) == -sin(x)
    assert cos(30*pi/2 + x) == -cos(x)

    for f in (cos, sin, tan, cot, csc, sec):
        i = f(pi*Rational(3, 7))
        j = TR3(i)
        assert verify_numerically(i, j) and i.func != j.func

    with evaluate(False):
        eq = cos(9*pi/22)
    assert eq.has(9*pi) and TR3(eq) == sin(pi/11)


def test_TR4():
    for i in [0, pi/6, pi/4, pi/3, pi/2]:
        with evaluate(False):
            eq = cos(i)
        assert isinstance(eq, cos) and TR4(eq) == cos(i)


def test__TR56():
    h = lambda x: 1 - x
    assert T(sin(x)**3, sin, cos, h, 4, False) == sin(x)*(-cos(x)**2 + 1)
    assert T(sin(x)**10, sin, cos, h, 4, False) == sin(x)**10
    assert T(sin(x)**6, sin, cos, h, 6, False) == (-cos(x)**2 + 1)**3
    assert T(sin(x)**6, sin, cos, h, 6, True) == sin(x)**6
    assert T(sin(x)**8, sin, cos, h, 10, True) == (-cos(x)**2 + 1)**4

    # issue 17137
    assert T(sin(x)**I, sin, cos, h, 4, True) == sin(x)**I
    assert T(sin(x)**(2*I + 1), sin, cos, h, 4, True) == sin(x)**(2*I + 1)


def test_TR5():
    assert TR5(sin(x)**2) == -cos(x)**2 + 1
    assert TR5(sin(x)**-2) == sin(x)**(-2)
    assert TR5(sin(x)**4) == (-cos(x)**2 + 1)**2


def test_TR6():
    assert TR6(cos(x)**2) == -sin(x)**2 + 1
    assert TR6(cos(x)**-2) == cos(x)**(-2)
    assert TR6(cos(x)**4) == (-sin(x)**2 + 1)**2


def test_TR7():
    assert TR7(cos(x)**2) == cos(2*x)/2 + S.Half
    assert TR7(cos(x)**2 + 1) == cos(2*x)/2 + Rational(3, 2)


def test_TR8():
    assert TR8(cos(2)*cos(3)) == cos(5)/2 + cos(1)/2
    assert TR8(cos(2)*sin(3)) == sin(5)/2 + sin(1)/2
    assert TR8(sin(2)*sin(3)) == -cos(5)/2 + cos(1)/2
    assert TR8(sin(1)*sin(2)*sin(3)) == sin(4)/4 - sin(6)/4 + sin(2)/4
    assert TR8(cos(2)*cos(3)*cos(4)*cos(5)) == \
        cos(4)/4 + cos(10)/8 + cos(2)/8 + cos(8)/8 + cos(14)/8 + \
        cos(6)/8 + Rational(1, 8)
    assert TR8(cos(2)*cos(3)*cos(4)*cos(5)*cos(6)) == \
        cos(10)/8 + cos(4)/8 + 3*cos(2)/16 + cos(16)/16 + cos(8)/8 + \
        cos(14)/16 + cos(20)/16 + cos(12)/16 + Rational(1, 16) + cos(6)/8
    assert TR8(sin(pi*Rational(3, 7))**2*cos(pi*Rational(3, 7))**2/(16*sin(pi/7)**2)) == Rational(1, 64)

def test_TR9():
    a = S.Half
    b = 3*a
    assert TR9(a) == a
    assert TR9(cos(1) + cos(2)) == 2*cos(a)*cos(b)
    assert TR9(cos(1) - cos(2)) == 2*sin(a)*sin(b)
    assert TR9(sin(1) - sin(2)) == -2*sin(a)*cos(b)
    assert TR9(sin(1) + sin(2)) == 2*sin(b)*cos(a)
    assert TR9(cos(1) + 2*sin(1) + 2*sin(2)) == cos(1) + 4*sin(b)*cos(a)
    assert TR9(cos(4) + cos(2) + 2*cos(1)*cos(3)) == 4*cos(1)*cos(3)
    assert TR9((cos(4) + cos(2))/cos(3)/2 + cos(3)) == 2*cos(1)*cos(2)
    assert TR9(cos(3) + cos(4) + cos(5) + cos(6)) == \
        4*cos(S.Half)*cos(1)*cos(Rational(9, 2))
    assert TR9(cos(3) + cos(3)*cos(2)) == cos(3) + cos(2)*cos(3)
    assert TR9(-cos(y) + cos(x*y)) == -2*sin(x*y/2 - y/2)*sin(x*y/2 + y/2)
    assert TR9(-sin(y) + sin(x*y)) == 2*sin(x*y/2 - y/2)*cos(x*y/2 + y/2)
    c = cos(x)
    s = sin(x)
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for a in ((c, s), (s, c), (cos(x), cos(x*y)), (sin(x), sin(x*y))):
            args = zip(si, a)
            ex = Add(*[Mul(*ai) for ai in args])
            t = TR9(ex)
            assert not (a[0].func == a[1].func and (
                not verify_numerically(ex, t.expand(trig=True)) or t.is_Add)
                or a[1].func != a[0].func and ex != t)


def test_TR10():
    assert TR10(cos(a + b)) == -sin(a)*sin(b) + cos(a)*cos(b)
    assert TR10(sin(a + b)) == sin(a)*cos(b) + sin(b)*cos(a)
    assert TR10(sin(a + b + c)) == \
        (-sin(a)*sin(b) + cos(a)*cos(b))*sin(c) + \
        (sin(a)*cos(b) + sin(b)*cos(a))*cos(c)
    assert TR10(cos(a + b + c)) == \
        (-sin(a)*sin(b) + cos(a)*cos(b))*cos(c) - \
        (sin(a)*cos(b) + sin(b)*cos(a))*sin(c)


def test_TR10i():
    assert TR10i(cos(1)*cos(3) + sin(1)*sin(3)) == cos(2)
    assert TR10i(cos(1)*cos(3) - sin(1)*sin(3)) == cos(4)
    assert TR10i(cos(1)*sin(3) - sin(1)*cos(3)) == sin(2)
    assert TR10i(cos(1)*sin(3) + sin(1)*cos(3)) == sin(4)
    assert TR10i(cos(1)*sin(3) + sin(1)*cos(3) + 7) == sin(4) + 7
    assert TR10i(cos(1)*sin(3) + sin(1)*cos(3) + cos(3)) == cos(3) + sin(4)
    assert TR10i(2*cos(1)*sin(3) + 2*sin(1)*cos(3) + cos(3)) == \
        2*sin(4) + cos(3)
    assert TR10i(cos(2)*cos(3) + sin(2)*(cos(1)*sin(2) + cos(2)*sin(1))) == \
        cos(1)
    eq = (cos(2)*cos(3) + sin(2)*(
        cos(1)*sin(2) + cos(2)*sin(1)))*cos(5) + sin(1)*sin(5)
    assert TR10i(eq) == TR10i(eq.expand()) == cos(4)
    assert TR10i(sqrt(2)*cos(x)*x + sqrt(6)*sin(x)*x) == \
        2*sqrt(2)*x*sin(x + pi/6)
    assert TR10i(cos(x)/sqrt(6) + sin(x)/sqrt(2) +
            cos(x)/sqrt(6)/3 + sin(x)/sqrt(2)/3) == 4*sqrt(6)*sin(x + pi/6)/9
    assert TR10i(cos(x)/sqrt(6) + sin(x)/sqrt(2) +
            cos(y)/sqrt(6)/3 + sin(y)/sqrt(2)/3) == \
        sqrt(6)*sin(x + pi/6)/3 + sqrt(6)*sin(y + pi/6)/9
    assert TR10i(cos(x) + sqrt(3)*sin(x) + 2*sqrt(3)*cos(x + pi/6)) == 4*cos(x)
    assert TR10i(cos(x) + sqrt(3)*sin(x) +
            2*sqrt(3)*cos(x + pi/6) + 4*sin(x)) == 4*sqrt(2)*sin(x + pi/4)
    assert TR10i(cos(2)*sin(3) + sin(2)*cos(4)) == \
        sin(2)*cos(4) + sin(3)*cos(2)

    A = Symbol('A', commutative=False)
    assert TR10i(sqrt(2)*cos(x)*A + sqrt(6)*sin(x)*A) == \
        2*sqrt(2)*sin(x + pi/6)*A


    c = cos(x)
    s = sin(x)
    h = sin(y)
    r = cos(y)
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for argsi in ((c*r, s*h), (c*h, s*r)): # explicit 2-args
            args = zip(si, argsi)
            ex = Add(*[Mul(*ai) for ai in args])
            t = TR10i(ex)
            assert not (ex - t.expand(trig=True) or t.is_Add)

    c = cos(x)
    s = sin(x)
    h = sin(pi/6)
    r = cos(pi/6)
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for argsi in ((c*r, s*h), (c*h, s*r)): # induced
            args = zip(si, argsi)
            ex = Add(*[Mul(*ai) for ai in args])
            t = TR10i(ex)
            assert not (ex - t.expand(trig=True) or t.is_Add)


def test_TR11():

    assert TR11(sin(2*x)) == 2*sin(x)*cos(x)
    assert TR11(sin(4*x)) == 4*((-sin(x)**2 + cos(x)**2)*sin(x)*cos(x))
    assert TR11(sin(x*Rational(4, 3))) == \
        4*((-sin(x/3)**2 + cos(x/3)**2)*sin(x/3)*cos(x/3))

    assert TR11(cos(2*x)) == -sin(x)**2 + cos(x)**2
    assert TR11(cos(4*x)) == \
        (-sin(x)**2 + cos(x)**2)**2 - 4*sin(x)**2*cos(x)**2

    assert TR11(cos(2)) == cos(2)

    assert TR11(cos(pi*Rational(3, 7)), pi*Rational(2, 7)) == -cos(pi*Rational(2, 7))**2 + sin(pi*Rational(2, 7))**2
    assert TR11(cos(4), 2) == -sin(2)**2 + cos(2)**2
    assert TR11(cos(6), 2) == cos(6)
    assert TR11(sin(x)/cos(x/2), x/2) == 2*sin(x/2)

def test__TR11():

    assert _TR11(sin(x/3)*sin(2*x)*sin(x/4)/(cos(x/6)*cos(x/8))) == \
        4*sin(x/8)*sin(x/6)*sin(2*x),_TR11(sin(x/3)*sin(2*x)*sin(x/4)/(cos(x/6)*cos(x/8)))
    assert _TR11(sin(x/3)/cos(x/6)) == 2*sin(x/6)

    assert _TR11(cos(x/6)/sin(x/3)) == 1/(2*sin(x/6))
    assert _TR11(sin(2*x)*cos(x/8)/sin(x/4)) == sin(2*x)/(2*sin(x/8)), _TR11(sin(2*x)*cos(x/8)/sin(x/4))
    assert _TR11(sin(x)/sin(x/2)) == 2*cos(x/2)


def test_TR12():
    assert TR12(tan(x + y)) == (tan(x) + tan(y))/(-tan(x)*tan(y) + 1)
    assert TR12(tan(x + y + z)) ==\
        (tan(z) + (tan(x) + tan(y))/(-tan(x)*tan(y) + 1))/(
        1 - (tan(x) + tan(y))*tan(z)/(-tan(x)*tan(y) + 1))
    assert TR12(tan(x*y)) == tan(x*y)


def test_TR13():
    assert TR13(tan(3)*tan(2)) == -tan(2)/tan(5) - tan(3)/tan(5) + 1
    assert TR13(cot(3)*cot(2)) == 1 + cot(3)*cot(5) + cot(2)*cot(5)
    assert TR13(tan(1)*tan(2)*tan(3)) == \
        (-tan(2)/tan(5) - tan(3)/tan(5) + 1)*tan(1)
    assert TR13(tan(1)*tan(2)*cot(3)) == \
        (-tan(2)/tan(3) + 1 - tan(1)/tan(3))*cot(3)


def test_L():
    assert L(cos(x) + sin(x)) == 2


def test_fu():

    assert fu(sin(50)**2 + cos(50)**2 + sin(pi/6)) == Rational(3, 2)
    assert fu(sqrt(6)*cos(x) + sqrt(2)*sin(x)) == 2*sqrt(2)*sin(x + pi/3)


    eq = sin(x)**4 - cos(y)**2 + sin(y)**2 + 2*cos(x)**2
    assert fu(eq) == cos(x)**4 - 2*cos(y)**2 + 2

    assert fu(S.Half - cos(2*x)/2) == sin(x)**2

    assert fu(sin(a)*(cos(b) - sin(b)) + cos(a)*(sin(b) + cos(b))) == \
        sqrt(2)*sin(a + b + pi/4)

    assert fu(sqrt(3)*cos(x)/2 + sin(x)/2) == sin(x + pi/3)

    assert fu(1 - sin(2*x)**2/4 - sin(y)**2 - cos(x)**4) == \
        -cos(x)**2 + cos(y)**2

    assert fu(cos(pi*Rational(4, 9))) == sin(pi/18)
    assert fu(cos(pi/9)*cos(pi*Rational(2, 9))*cos(pi*Rational(3, 9))*cos(pi*Rational(4, 9))) == Rational(1, 16)

    assert fu(
        tan(pi*Rational(7, 18)) + tan(pi*Rational(5, 18)) - sqrt(3)*tan(pi*Rational(5, 18))*tan(pi*Rational(7, 18))) == \
        -sqrt(3)

    assert fu(tan(1)*tan(2)) == tan(1)*tan(2)

    expr = Mul(*[cos(2**i) for i in range(10)])
    assert fu(expr) == sin(1024)/(1024*sin(1))

    # issue #18059:
    assert fu(cos(x) + sqrt(sin(x)**2)) == cos(x) + sqrt(sin(x)**2)

    assert fu((-14*sin(x)**3 + 35*sin(x) + 6*sqrt(3)*cos(x)**3 + 9*sqrt(3)*cos(x))/((cos(2*x) + 4))) == \
        7*sin(x) + 3*sqrt(3)*cos(x)


def test_objective():
    assert fu(sin(x)/cos(x), measure=lambda x: x.count_ops()) == \
            tan(x)
    assert fu(sin(x)/cos(x), measure=lambda x: -x.count_ops()) == \
            sin(x)/cos(x)


def test_process_common_addends():
    # this tests that the args are not evaluated as they are given to do
    # and that key2 works when key1 is False
    do = lambda x: Add(*[i**(i%2) for i in x.args])
    assert process_common_addends(Add(*[1, 2, 3, 4], evaluate=False), do,
        key2=lambda x: x%2, key1=False) == 1**1 + 3**1 + 2**0 + 4**0


def test_trig_split():
    assert trig_split(cos(x), cos(y)) == (1, 1, 1, x, y, True)
    assert trig_split(2*cos(x), -2*cos(y)) == (2, 1, -1, x, y, True)
    assert trig_split(cos(x)*sin(y), cos(y)*sin(y)) == \
        (sin(y), 1, 1, x, y, True)

    assert trig_split(cos(x), -sqrt(3)*sin(x), two=True) == \
        (2, 1, -1, x, pi/6, False)
    assert trig_split(cos(x), sin(x), two=True) == \
        (sqrt(2), 1, 1, x, pi/4, False)
    assert trig_split(cos(x), -sin(x), two=True) == \
        (sqrt(2), 1, -1, x, pi/4, False)
    assert trig_split(sqrt(2)*cos(x), -sqrt(6)*sin(x), two=True) == \
        (2*sqrt(2), 1, -1, x, pi/6, False)
    assert trig_split(-sqrt(6)*cos(x), -sqrt(2)*sin(x), two=True) == \
        (-2*sqrt(2), 1, 1, x, pi/3, False)
    assert trig_split(cos(x)/sqrt(6), sin(x)/sqrt(2), two=True) == \
        (sqrt(6)/3, 1, 1, x, pi/6, False)
    assert trig_split(-sqrt(6)*cos(x)*sin(y),
            -sqrt(2)*sin(x)*sin(y), two=True) == \
        (-2*sqrt(2)*sin(y), 1, 1, x, pi/3, False)

    assert trig_split(cos(x), sin(x)) is None
    assert trig_split(cos(x), sin(z)) is None
    assert trig_split(2*cos(x), -sin(x)) is None
    assert trig_split(cos(x), -sqrt(3)*sin(x)) is None
    assert trig_split(cos(x)*cos(y), sin(x)*sin(z)) is None
    assert trig_split(cos(x)*cos(y), sin(x)*sin(y)) is None
    assert trig_split(-sqrt(6)*cos(x), sqrt(2)*sin(x)*sin(y), two=True) is \
        None

    assert trig_split(sqrt(3)*sqrt(x), cos(3), two=True) is None
    assert trig_split(sqrt(3)*root(x, 3), sin(3)*cos(2), two=True) is None
    assert trig_split(cos(5)*cos(6), cos(7)*sin(5), two=True) is None


def test_TRmorrie():
    assert TRmorrie(7*Mul(*[cos(i) for i in range(10)])) == \
        7*sin(12)*sin(16)*cos(5)*cos(7)*cos(9)/(64*sin(1)*sin(3))
    assert TRmorrie(x) == x
    assert TRmorrie(2*x) == 2*x
    e = cos(pi/7)*cos(pi*Rational(2, 7))*cos(pi*Rational(4, 7))
    assert TR8(TRmorrie(e)) == Rational(-1, 8)
    e = Mul(*[cos(2**i*pi/17) for i in range(1, 17)])
    assert TR8(TR3(TRmorrie(e))) == Rational(1, 65536)
    # issue 17063
    eq = cos(x)/cos(x/2)
    assert TRmorrie(eq) == eq
    # issue #20430
    eq = cos(x/2)*sin(x/2)*cos(x)**3
    assert TRmorrie(eq) == sin(2*x)*cos(x)**2/4


def test_TRpower():
    assert TRpower(1/sin(x)**2) == 1/sin(x)**2
    assert TRpower(cos(x)**3*sin(x/2)**4) == \
        (3*cos(x)/4 + cos(3*x)/4)*(-cos(x)/2 + cos(2*x)/8 + Rational(3, 8))
    for k in range(2, 8):
        assert verify_numerically(sin(x)**k, TRpower(sin(x)**k))
        assert verify_numerically(cos(x)**k, TRpower(cos(x)**k))


def test_hyper_as_trig():
    from sympy.simplify.fu import _osborne, _osbornei

    eq = sinh(x)**2 + cosh(x)**2
    t, f = hyper_as_trig(eq)
    assert f(fu(t)) == cosh(2*x)
    e, f = hyper_as_trig(tanh(x + y))
    assert f(TR12(e)) == (tanh(x) + tanh(y))/(tanh(x)*tanh(y) + 1)

    d = Dummy()
    assert _osborne(sinh(x), d) == I*sin(x*d)
    assert _osborne(tanh(x), d) == I*tan(x*d)
    assert _osborne(coth(x), d) == cot(x*d)/I
    assert _osborne(cosh(x), d) == cos(x*d)
    assert _osborne(sech(x), d) == sec(x*d)
    assert _osborne(csch(x), d) == csc(x*d)/I
    for func in (sinh, cosh, tanh, coth, sech, csch):
        h = func(pi)
        assert _osbornei(_osborne(h, d), d) == h
    # /!\ the _osborne functions are not meant to work
    # in the o(i(trig, d), d) direction so we just check
    # that they work as they are supposed to work
    assert _osbornei(cos(x*y + z), y) == cosh(x + z*I)
    assert _osbornei(sin(x*y + z), y) == sinh(x + z*I)/I
    assert _osbornei(tan(x*y + z), y) == tanh(x + z*I)/I
    assert _osbornei(cot(x*y + z), y) == coth(x + z*I)*I
    assert _osbornei(sec(x*y + z), y) == sech(x + z*I)
    assert _osbornei(csc(x*y + z), y) == csch(x + z*I)*I


def test_TR12i():
    ta, tb, tc = [tan(i) for i in (a, b, c)]
    assert TR12i((ta + tb)/(-ta*tb + 1)) == tan(a + b)
    assert TR12i((ta + tb)/(ta*tb - 1)) == -tan(a + b)
    assert TR12i((-ta - tb)/(ta*tb - 1)) == tan(a + b)
    eq = (ta + tb)/(-ta*tb + 1)**2*(-3*ta - 3*tc)/(2*(ta*tc - 1))
    assert TR12i(eq.expand()) == \
        -3*tan(a + b)*tan(a + c)/(tan(a) + tan(b) - 1)/2
    assert TR12i(tan(x)/sin(x)) == tan(x)/sin(x)
    eq = (ta + cos(2))/(-ta*tb + 1)
    assert TR12i(eq) == eq
    eq = (ta + tb + 2)**2/(-ta*tb + 1)
    assert TR12i(eq) == eq
    eq = ta/(-ta*tb + 1)
    assert TR12i(eq) == eq
    eq = (((ta + tb)*(a + 1)).expand())**2/(ta*tb - 1)
    assert TR12i(eq) == -(a + 1)**2*tan(a + b)


def test_TR14():
    eq = (cos(x) - 1)*(cos(x) + 1)
    ans = -sin(x)**2
    assert TR14(eq) == ans
    assert TR14(1/eq) == 1/ans
    assert TR14((cos(x) - 1)**2*(cos(x) + 1)**2) == ans**2
    assert TR14((cos(x) - 1)**2*(cos(x) + 1)**3) == ans**2*(cos(x) + 1)
    assert TR14((cos(x) - 1)**3*(cos(x) + 1)**2) == ans**2*(cos(x) - 1)
    eq = (cos(x) - 1)**y*(cos(x) + 1)**y
    assert TR14(eq) == eq
    eq = (cos(x) - 2)**y*(cos(x) + 1)
    assert TR14(eq) == eq
    eq = (tan(x) - 2)**2*(cos(x) + 1)
    assert TR14(eq) == eq
    i = symbols('i', integer=True)
    assert TR14((cos(x) - 1)**i*(cos(x) + 1)**i) == ans**i
    assert TR14((sin(x) - 1)**i*(sin(x) + 1)**i) == (-cos(x)**2)**i
    # could use extraction in this case
    eq = (cos(x) - 1)**(i + 1)*(cos(x) + 1)**i
    assert TR14(eq) in [(cos(x) - 1)*ans**i, eq]

    assert TR14((sin(x) - 1)*(sin(x) + 1)) == -cos(x)**2
    p1 = (cos(x) + 1)*(cos(x) - 1)
    p2 = (cos(y) - 1)*2*(cos(y) + 1)
    p3 = (3*(cos(y) - 1))*(3*(cos(y) + 1))
    assert TR14(p1*p2*p3*(x - 1)) == -18*((x - 1)*sin(x)**2*sin(y)**4)


def test_TR15_16_17():
    assert TR15(1 - 1/sin(x)**2) == -cot(x)**2
    assert TR16(1 - 1/cos(x)**2) == -tan(x)**2
    assert TR111(1 - 1/tan(x)**2) == 1 - cot(x)**2


def test_as_f_sign_1():
    assert as_f_sign_1(x + 1) == (1, x, 1)
    assert as_f_sign_1(x - 1) == (1, x, -1)
    assert as_f_sign_1(-x + 1) == (-1, x, -1)
    assert as_f_sign_1(-x - 1) == (-1, x, 1)
    assert as_f_sign_1(2*x + 2) == (2, x, 1)
    assert as_f_sign_1(x*y - y) == (y, x, -1)
    assert as_f_sign_1(-x*y + y) == (-y, x, -1)


def test_issue_25590():
    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)

    assert TR8(2*cos(x)*sin(x)*B*A) == sin(2*x)*B*A
    assert TR13(tan(2)*tan(3)*B*A) == (-tan(2)/tan(5) - tan(3)/tan(5) + 1)*B*A

    # XXX The result may not be optimal than
    # sin(2*x)*B*A + cos(x)**2 and may change in the future
    assert (2*cos(x)*sin(x)*B*A + cos(x)**2).simplify() == sin(2*x)*B*A + cos(2*x)/2 + S.One/2
