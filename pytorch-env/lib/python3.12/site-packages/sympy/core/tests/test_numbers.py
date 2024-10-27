import numbers as nums
import decimal
from sympy.concrete.summations import Sum
from sympy.core import (EulerGamma, Catalan, TribonacciConstant,
    GoldenRatio)
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import (mpf_norm, seterr,
    Integer, I, pi, comp, Rational, E, nan,
    oo, AlgebraicNumber, Number, Float, zoo, equal_valued,
    int_valued, all_close)
from sympy.core.intfunc import (igcd, igcdex, igcd2, igcd_lehmer,
    ilcm, integer_nthroot, isqrt, integer_log, mod_inverse)
from sympy.core.power import Pow
from sympy.core.relational import Ge, Gt, Le, Lt
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.integers import floor
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.polys.domains.realfield import RealField
from sympy.printing.latex import latex
from sympy.printing.repr import srepr
from sympy.simplify import simplify
from sympy.polys.domains.groundtypes import PythonRational
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow
from sympy import Add

from mpmath import mpf
import mpmath
from sympy.core import numbers
t = Symbol('t', real=False)

_ninf = float(-oo)
_inf = float(oo)


def same_and_same_prec(a, b):
    # stricter matching for Floats
    return a == b and a._prec == b._prec


def test_seterr():
    seterr(divide=True)
    raises(ValueError, lambda: S.Zero/S.Zero)
    seterr(divide=False)
    assert S.Zero / S.Zero is S.NaN


def test_mod():
    x = S.Half
    y = Rational(3, 4)
    z = Rational(5, 18043)

    assert x % x == 0
    assert x % y == S.Half
    assert x % z == Rational(3, 36086)
    assert y % x == Rational(1, 4)
    assert y % y == 0
    assert y % z == Rational(9, 72172)
    assert z % x == Rational(5, 18043)
    assert z % y == Rational(5, 18043)
    assert z % z == 0

    a = Float(2.6)

    assert (a % .2) == 0.0
    assert (a % 2).round(15) == 0.6
    assert (a % 0.5).round(15) == 0.1

    p = Symbol('p', infinite=True)

    assert oo % oo is nan
    assert zoo % oo is nan
    assert 5 % oo is nan
    assert p % 5 is nan

    # In these two tests, if the precision of m does
    # not match the precision of the ans, then it is
    # likely that the change made now gives an answer
    # with degraded accuracy.
    r = Rational(500, 41)
    f = Float('.36', 3)
    m = r % f
    ans = Float(r % Rational(f), 3)
    assert m == ans and m._prec == ans._prec
    f = Float('8.36', 3)
    m = f % r
    ans = Float(Rational(f) % r, 3)
    assert m == ans and m._prec == ans._prec

    s = S.Zero

    assert s % float(1) == 0.0

    # No rounding required since these numbers can be represented
    # exactly.
    assert Rational(3, 4) % Float(1.1) == 0.75
    assert Float(1.5) % Rational(5, 4) == 0.25
    assert Rational(5, 4).__rmod__(Float('1.5')) == 0.25
    assert Float('1.5').__rmod__(Float('2.75')) == Float('1.25')
    assert 2.75 % Float('1.5') == Float('1.25')

    a = Integer(7)
    b = Integer(4)

    assert type(a % b) == Integer
    assert a % b == Integer(3)
    assert Integer(1) % Rational(2, 3) == Rational(1, 3)
    assert Rational(7, 5) % Integer(1) == Rational(2, 5)
    assert Integer(2) % 1.5 == 0.5

    assert Integer(3).__rmod__(Integer(10)) == Integer(1)
    assert Integer(10) % 4 == Integer(2)
    assert 15 % Integer(4) == Integer(3)


def test_divmod():
    x = Symbol("x")
    assert divmod(S(12), S(8)) == Tuple(1, 4)
    assert divmod(-S(12), S(8)) == Tuple(-2, 4)
    assert divmod(S.Zero, S.One) == Tuple(0, 0)
    raises(ZeroDivisionError, lambda: divmod(S.Zero, S.Zero))
    raises(ZeroDivisionError, lambda: divmod(S.One, S.Zero))
    assert divmod(S(12), 8) == Tuple(1, 4)
    assert divmod(12, S(8)) == Tuple(1, 4)
    assert S(1024)//x == 1024//x == floor(1024/x)

    assert divmod(S("2"), S("3/2")) == Tuple(S("1"), S("1/2"))
    assert divmod(S("3/2"), S("2")) == Tuple(S("0"), S("3/2"))
    assert divmod(S("2"), S("3.5")) == Tuple(S("0"), S("2."))
    assert divmod(S("3.5"), S("2")) == Tuple(S("1"), S("1.5"))
    assert divmod(S("2"), S("1/3")) == Tuple(S("6"), S("0"))
    assert divmod(S("1/3"), S("2")) == Tuple(S("0"), S("1/3"))
    assert divmod(S("2"), S("1/10")) == Tuple(S("20"), S("0"))
    assert divmod(S("2"), S(".1"))[0] == 19
    assert divmod(S("0.1"), S("2")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("2"), 2) == Tuple(S("1"), S("0"))
    assert divmod(2, S("2")) == Tuple(S("1"), S("0"))
    assert divmod(S("2"), 1.5) == Tuple(S("1"), S("0.5"))
    assert divmod(1.5, S("2")) == Tuple(S("0"), S("1.5"))
    assert divmod(0.3, S("2")) == Tuple(S("0"), S("0.3"))
    assert divmod(S("3/2"), S("3.5")) == Tuple(S("0"), S(3/2))
    assert divmod(S("3.5"), S("3/2")) == Tuple(S("2"), S("0.5"))
    assert divmod(S("3/2"), S("1/3")) == Tuple(S("4"), S("1/6"))
    assert divmod(S("1/3"), S("3/2")) == Tuple(S("0"), S("1/3"))
    assert divmod(S("3/2"), S("0.1"))[0] == 14
    assert divmod(S("0.1"), S("3/2")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("3/2"), 2) == Tuple(S("0"), S("3/2"))
    assert divmod(2, S("3/2")) == Tuple(S("1"), S("1/2"))
    assert divmod(S("3/2"), 1.5) == Tuple(S("1"), S("0."))
    assert divmod(1.5, S("3/2")) == Tuple(S("1"), S("0."))
    assert divmod(S("3/2"), 0.3) == Tuple(S("5"), S("0."))
    assert divmod(0.3, S("3/2")) == Tuple(S("0"), S("0.3"))
    assert divmod(S("1/3"), S("3.5")) == (0, 1/3)
    assert divmod(S("3.5"), S("0.1")) == Tuple(S("35"), S("0."))
    assert divmod(S("0.1"), S("3.5")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("3.5"), 2) == Tuple(S("1"), S("1.5"))
    assert divmod(2, S("3.5")) == Tuple(S("0"), S("2."))
    assert divmod(S("3.5"), 1.5) == Tuple(S("2"), S("0.5"))
    assert divmod(1.5, S("3.5")) == Tuple(S("0"), S("1.5"))
    assert divmod(0.3, S("3.5")) == Tuple(S("0"), S("0.3"))
    assert divmod(S("0.1"), S("1/3")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("1/3"), 2) == Tuple(S("0"), S("1/3"))
    assert divmod(2, S("1/3")) == Tuple(S("6"), S("0"))
    assert divmod(S("1/3"), 1.5) == (0, 1/3)
    assert divmod(0.3, S("1/3")) == (0, 0.3)
    assert divmod(S("0.1"), 2) == (0, 0.1)
    assert divmod(2, S("0.1"))[0] == 19
    assert divmod(S("0.1"), 1.5) == (0, 0.1)
    assert divmod(1.5, S("0.1")) == Tuple(S("15"), S("0."))
    assert divmod(S("0.1"), 0.3) == Tuple(S("0"), S("0.1"))

    assert str(divmod(S("2"), 0.3)) == '(6, 0.2)'
    assert str(divmod(S("3.5"), S("1/3"))) == '(10, 0.166666666666667)'
    assert str(divmod(S("3.5"), 0.3)) == '(11, 0.2)'
    assert str(divmod(S("1/3"), S("0.1"))) == '(3, 0.0333333333333333)'
    assert str(divmod(1.5, S("1/3"))) == '(4, 0.166666666666667)'
    assert str(divmod(S("1/3"), 0.3)) == '(1, 0.0333333333333333)'
    assert str(divmod(0.3, S("0.1"))) == '(2, 0.1)'

    assert divmod(-3, S(2)) == (-2, 1)
    assert divmod(S(-3), S(2)) == (-2, 1)
    assert divmod(S(-3), 2) == (-2, 1)

    assert divmod(oo, 1) == (S.NaN, S.NaN)
    assert divmod(S.NaN, 1) == (S.NaN, S.NaN)
    assert divmod(1, S.NaN) == (S.NaN, S.NaN)
    ans = [(-1, oo), (-1, oo), (0, 0), (0, 1), (0, 2)]
    OO = float('inf')
    ANS = [tuple(map(float, i)) for i in ans]
    assert [divmod(i, oo) for i in range(-2, 3)] == ans
    ans = [(0, -2), (0, -1), (0, 0), (-1, -oo), (-1, -oo)]
    ANS = [tuple(map(float, i)) for i in ans]
    assert [divmod(i, -oo) for i in range(-2, 3)] == ans
    assert [divmod(i, -OO) for i in range(-2, 3)] == ANS

    # sympy's divmod gives an Integer for the quotient rather than a float
    dmod = lambda a, b: tuple([j if i else int(j) for i, j in enumerate(divmod(a, b))])
    for a in (4, 4., 4.25, 0, 0., -4, -4. -4.25):
        for b in (2, 2., 2.5, -2, -2., -2.5):
            assert divmod(S(a), S(b)) == dmod(a, b)


def test_igcd():
    assert igcd(0, 0) == 0
    assert igcd(0, 1) == 1
    assert igcd(1, 0) == 1
    assert igcd(0, 7) == 7
    assert igcd(7, 0) == 7
    assert igcd(7, 1) == 1
    assert igcd(1, 7) == 1
    assert igcd(-1, 0) == 1
    assert igcd(0, -1) == 1
    assert igcd(-1, -1) == 1
    assert igcd(-1, 7) == 1
    assert igcd(7, -1) == 1
    assert igcd(8, 2) == 2
    assert igcd(4, 8) == 4
    assert igcd(8, 16) == 8
    assert igcd(7, -3) == 1
    assert igcd(-7, 3) == 1
    assert igcd(-7, -3) == 1
    assert igcd(*[10, 20, 30]) == 10
    raises(TypeError, lambda: igcd())
    raises(TypeError, lambda: igcd(2))
    raises(ValueError, lambda: igcd(0, None))
    raises(ValueError, lambda: igcd(1, 2.2))
    for args in permutations((45.1, 1, 30)):
        raises(ValueError, lambda: igcd(*args))
    for args in permutations((1, 2, None)):
        raises(ValueError, lambda: igcd(*args))


def test_igcd_lehmer():
    a, b = fibonacci(10001), fibonacci(10000)
    # len(str(a)) == 2090
    # small divisors, long Euclidean sequence
    assert igcd_lehmer(a, b) == 1
    c = fibonacci(100)
    assert igcd_lehmer(a*c, b*c) == c
    # big divisor
    assert igcd_lehmer(a, 10**1000) == 1
    # swapping argument
    assert igcd_lehmer(1, 2) == igcd_lehmer(2, 1)


def test_igcd2():
    # short loop
    assert igcd2(2**100 - 1, 2**99 - 1) == 1
    # Lehmer's algorithm
    a, b = int(fibonacci(10001)), int(fibonacci(10000))
    assert igcd2(a, b) == 1


def test_ilcm():
    assert ilcm(0, 0) == 0
    assert ilcm(1, 0) == 0
    assert ilcm(0, 1) == 0
    assert ilcm(1, 1) == 1
    assert ilcm(2, 1) == 2
    assert ilcm(8, 2) == 8
    assert ilcm(8, 6) == 24
    assert ilcm(8, 7) == 56
    assert ilcm(*[10, 20, 30]) == 60
    raises(ValueError, lambda: ilcm(8.1, 7))
    raises(ValueError, lambda: ilcm(8, 7.1))
    raises(TypeError, lambda: ilcm(8))


def test_igcdex():
    assert igcdex(2, 3) == (-1, 1, 1)
    assert igcdex(10, 12) == (-1, 1, 2)
    assert igcdex(100, 2004) == (-20, 1, 4)
    assert igcdex(0, 0) == (0, 1, 0)
    assert igcdex(1, 0) == (1, 0, 1)


def _strictly_equal(a, b):
    return (a.p, a.q, type(a.p), type(a.q)) == \
           (b.p, b.q, type(b.p), type(b.q))


def _test_rational_new(cls):
    """
    Tests that are common between Integer and Rational.
    """
    assert cls(0) is S.Zero
    assert cls(1) is S.One
    assert cls(-1) is S.NegativeOne
    # These look odd, but are similar to int():
    assert cls('1') is S.One
    assert cls('-1') is S.NegativeOne

    i = Integer(10)
    assert _strictly_equal(i, cls('10'))
    assert _strictly_equal(i, cls('10'))
    assert _strictly_equal(i, cls(int(10)))
    assert _strictly_equal(i, cls(i))

    raises(TypeError, lambda: cls(Symbol('x')))


def test_Integer_new():
    """
    Test for Integer constructor
    """
    _test_rational_new(Integer)

    assert _strictly_equal(Integer(0.9), S.Zero)
    assert _strictly_equal(Integer(10.5), Integer(10))
    raises(ValueError, lambda: Integer("10.5"))
    assert Integer(Rational('1.' + '9'*20)) == 1


def test_Rational_new():
    """"
    Test for Rational constructor
    """
    _test_rational_new(Rational)

    n1 = S.Half
    assert n1 == Rational(Integer(1), 2)
    assert n1 == Rational(Integer(1), Integer(2))
    assert n1 == Rational(1, Integer(2))
    assert n1 == Rational(S.Half)
    assert 1 == Rational(n1, n1)
    assert Rational(3, 2) == Rational(S.Half, Rational(1, 3))
    assert Rational(3, 1) == Rational(1, Rational(1, 3))
    n3_4 = Rational(3, 4)
    assert Rational('3/4') == n3_4
    assert -Rational('-3/4') == n3_4
    assert Rational('.76').limit_denominator(4) == n3_4
    assert Rational(19, 25).limit_denominator(4) == n3_4
    assert Rational('19/25').limit_denominator(4) == n3_4
    assert Rational(1.0, 3) == Rational(1, 3)
    assert Rational(1, 3.0) == Rational(1, 3)
    assert Rational(Float(0.5)) == S.Half
    assert Rational('1e2/1e-2') == Rational(10000)
    assert Rational('1 234') == Rational(1234)
    assert Rational('1/1 234') == Rational(1, 1234)
    assert Rational(-1, 0) is S.ComplexInfinity
    assert Rational(1, 0) is S.ComplexInfinity
    # Make sure Rational doesn't lose precision on Floats
    assert Rational(pi.evalf(100)).evalf(100) == pi.evalf(100)
    raises(TypeError, lambda: Rational('3**3'))
    raises(TypeError, lambda: Rational('1/2 + 2/3'))

    # handle fractions.Fraction instances
    try:
        import fractions
        assert Rational(fractions.Fraction(1, 2)) == S.Half
    except ImportError:
        pass

    assert Rational(PythonRational(2, 6)) == Rational(1, 3)

    assert Rational(2, 4, gcd=1).q == 4
    n = Rational(2, -4, gcd=1)
    assert n.q == 4
    assert n.p == -2

def test_issue_24543():
    for p in ('1.5', 1.5, 2):
        for q in ('1.5', 1.5, 2):
            assert Rational(p, q).as_numer_denom() == Rational('%s/%s'%(p,q)).as_numer_denom()

    assert Rational('0.5', '100') == Rational(1, 200)


def test_Number_new():
    """"
    Test for Number constructor
    """
    # Expected behavior on numbers and strings
    assert Number(1) is S.One
    assert Number(2).__class__ is Integer
    assert Number(-622).__class__ is Integer
    assert Number(5, 3).__class__ is Rational
    assert Number(5.3).__class__ is Float
    assert Number('1') is S.One
    assert Number('2').__class__ is Integer
    assert Number('-622').__class__ is Integer
    assert Number('5/3').__class__ is Rational
    assert Number('5.3').__class__ is Float
    raises(ValueError, lambda: Number('cos'))
    raises(TypeError, lambda: Number(cos))
    a = Rational(3, 5)
    assert Number(a) is a  # Check idempotence on Numbers
    u = ['inf', '-inf', 'nan', 'iNF', '+inf']
    v = [oo, -oo, nan, oo, oo]
    for i, a in zip(u, v):
        assert Number(i) is a, (i, Number(i), a)


def test_Number_cmp():
    n1 = Number(1)
    n2 = Number(2)
    n3 = Number(-3)

    assert n1 < n2
    assert n1 <= n2
    assert n3 < n1
    assert n2 > n3
    assert n2 >= n3

    raises(TypeError, lambda: n1 < S.NaN)
    raises(TypeError, lambda: n1 <= S.NaN)
    raises(TypeError, lambda: n1 > S.NaN)
    raises(TypeError, lambda: n1 >= S.NaN)


def test_Rational_cmp():
    n1 = Rational(1, 4)
    n2 = Rational(1, 3)
    n3 = Rational(2, 4)
    n4 = Rational(2, -4)
    n5 = Rational(0)
    n6 = Rational(1)
    n7 = Rational(3)
    n8 = Rational(-3)

    assert n8 < n5
    assert n5 < n6
    assert n6 < n7
    assert n8 < n7
    assert n7 > n8
    assert (n1 + 1)**n2 < 2
    assert ((n1 + n6)/n7) < 1

    assert n4 < n3
    assert n2 < n3
    assert n1 < n2
    assert n3 > n1
    assert not n3 < n1
    assert not (Rational(-1) > 0)
    assert Rational(-1) < 0

    raises(TypeError, lambda: n1 < S.NaN)
    raises(TypeError, lambda: n1 <= S.NaN)
    raises(TypeError, lambda: n1 > S.NaN)
    raises(TypeError, lambda: n1 >= S.NaN)


def test_Float():
    def eq(a, b):
        t = Float("1.0E-15")
        return (-t < a - b < t)

    zeros = (0, S.Zero, 0., Float(0))
    for i, j in permutations(zeros[:-1], 2):
        assert i == j
    for i, j in permutations(zeros[-2:], 2):
        assert i == j
    for z in zeros:
        assert z in zeros
    assert S.Zero.is_zero

    a = Float(2) ** Float(3)
    assert eq(a.evalf(), Float(8))
    assert eq((pi ** -1).evalf(), Float("0.31830988618379067"))
    a = Float(2) ** Float(4)
    assert eq(a.evalf(), Float(16))
    assert (S(.3) == S(.5)) is False

    mpf = (0, 5404319552844595, -52, 53)
    x_str = Float((0, '13333333333333', -52, 53))
    x_0xstr = Float((0, '0x13333333333333', -52, 53))
    x2_str = Float((0, '26666666666666', -53, 54))
    x_hex = Float((0, int(0x13333333333333), -52, 53))
    x_dec = Float(mpf)
    assert x_str == x_0xstr == x_hex == x_dec == Float(1.2)
    # x2_str was entered slightly malformed in that the mantissa
    # was even -- it should be odd and the even part should be
    # included with the exponent, but this is resolved by normalization
    # ONLY IF REQUIREMENTS of mpf_norm are met: the bitcount must
    # be exact: double the mantissa ==> increase bc by 1
    assert Float(1.2)._mpf_ == mpf
    assert x2_str._mpf_ == mpf

    assert Float((0, int(0), -123, -1)) is S.NaN
    assert Float((0, int(0), -456, -2)) is S.Infinity
    assert Float((1, int(0), -789, -3)) is S.NegativeInfinity
    # if you don't give the full signature, it's not special
    assert Float((0, int(0), -123)) == Float(0)
    assert Float((0, int(0), -456)) == Float(0)
    assert Float((1, int(0), -789)) == Float(0)

    raises(ValueError, lambda: Float((0, 7, 1, 3), ''))

    assert Float('0.0').is_finite is True
    assert Float('0.0').is_negative is False
    assert Float('0.0').is_positive is False
    assert Float('0.0').is_infinite is False
    assert Float('0.0').is_zero is True

    # rationality properties
    # if the integer test fails then the use of intlike
    # should be removed from gamma_functions.py
    assert Float(1).is_integer is None
    assert Float(1).is_rational is None
    assert Float(1).is_irrational is None
    assert sqrt(2).n(15).is_rational is None
    assert sqrt(2).n(15).is_irrational is None

    # do not automatically evalf
    def teq(a):
        assert (a.evalf() == a) is False
        assert (a.evalf() != a) is True
        assert (a == a.evalf()) is False
        assert (a != a.evalf()) is True

    teq(pi)
    teq(2*pi)
    teq(cos(0.1, evaluate=False))

    # long integer
    i = 12345678901234567890
    assert same_and_same_prec(Float(12, ''), Float('12', ''))
    assert same_and_same_prec(Float(Integer(i), ''), Float(i, ''))
    assert same_and_same_prec(Float(i, ''), Float(str(i), 20))
    assert same_and_same_prec(Float(str(i)), Float(i, ''))
    assert same_and_same_prec(Float(i), Float(i, ''))

    # inexact floats (repeating binary = denom not multiple of 2)
    # cannot have precision greater than 15
    assert Float(.125, 22)._prec == 76
    assert Float(2.0, 22)._prec == 76
    # only default prec is equal, even for exactly representable float
    assert Float(.125, 22) != .125
    #assert Float(2.0, 22) == 2
    assert float(Float('.12500000000000001', '')) == .125
    raises(ValueError, lambda: Float(.12500000000000001, ''))

    # allow spaces
    assert Float('123 456.123 456') == Float('123456.123456')
    assert Integer('123 456') == Integer('123456')
    assert Rational('123 456.123 456') == Rational('123456.123456')
    assert Float(' .3e2') == Float('0.3e2')
    # but treat them as strictly ass underscore between digits: only 1
    raises(ValueError, lambda: Float('1  2'))

    # allow underscore between digits
    assert Float('1_23.4_56') == Float('123.456')
    # assert Float('1_23.4_5_6', 12) == Float('123.456', 12)
    # ...but not in all cases (per Py 3.6)
    raises(ValueError, lambda: Float('1_'))
    raises(ValueError, lambda: Float('1__2'))
    raises(ValueError, lambda: Float('_1'))
    raises(ValueError, lambda: Float('_inf'))

    # allow auto precision detection
    assert Float('.1', '') == Float(.1, 1)
    assert Float('.125', '') == Float(.125, 3)
    assert Float('.100', '') == Float(.1, 3)
    assert Float('2.0', '') == Float('2', 2)

    raises(ValueError, lambda: Float("12.3d-4", ""))
    raises(ValueError, lambda: Float(12.3, ""))
    raises(ValueError, lambda: Float('.'))
    raises(ValueError, lambda: Float('-.'))

    zero = Float('0.0')
    assert Float('-0') == zero
    assert Float('.0') == zero
    assert Float('-.0') == zero
    assert Float('-0.0') == zero
    assert Float(0.0) == zero
    assert Float(0) == zero
    assert Float(0, '') == Float('0', '')
    assert Float(1) == Float(1.0)
    assert Float(S.Zero) == zero
    assert Float(S.One) == Float(1.0)

    assert Float(decimal.Decimal('0.1'), 3) == Float('.1', 3)
    assert Float(decimal.Decimal('nan')) is S.NaN
    assert Float(decimal.Decimal('Infinity')) is S.Infinity
    assert Float(decimal.Decimal('-Infinity')) is S.NegativeInfinity

    assert '{:.3f}'.format(Float(4.236622)) == '4.237'
    assert '{:.35f}'.format(Float(pi.n(40), 40)) == \
        '3.14159265358979323846264338327950288'

    # unicode
    assert Float('0.73908513321516064100000000') == \
        Float('0.73908513321516064100000000')
    assert Float('0.73908513321516064100000000', 28) == \
        Float('0.73908513321516064100000000', 28)

    # binary precision
    # Decimal value 0.1 cannot be expressed precisely as a base 2 fraction
    a = Float(S.One/10, dps=15)
    b = Float(S.One/10, dps=16)
    p = Float(S.One/10, precision=53)
    q = Float(S.One/10, precision=54)
    assert a._mpf_ == p._mpf_
    assert not a._mpf_ == q._mpf_
    assert not b._mpf_ == q._mpf_

    # Precision specifying errors
    raises(ValueError, lambda: Float("1.23", dps=3, precision=10))
    raises(ValueError, lambda: Float("1.23", dps="", precision=10))
    raises(ValueError, lambda: Float("1.23", dps=3, precision=""))
    raises(ValueError, lambda: Float("1.23", dps="", precision=""))

    # from NumberSymbol
    assert same_and_same_prec(Float(pi, 32), pi.evalf(32))
    assert same_and_same_prec(Float(Catalan), Catalan.evalf())

    # oo and nan
    u = ['inf', '-inf', 'nan', 'iNF', '+inf']
    v = [oo, -oo, nan, oo, oo]
    for i, a in zip(u, v):
        assert Float(i) is a


def test_zero_not_false():
    # https://github.com/sympy/sympy/issues/20796
    assert (S(0.0) == S.false) is False
    assert (S.false == S(0.0)) is False
    assert (S(0) == S.false) is False
    assert (S.false == S(0)) is False


@conserve_mpmath_dps
def test_float_mpf():
    import mpmath
    mpmath.mp.dps = 100
    mp_pi = mpmath.pi()

    assert Float(mp_pi, 100) == Float(mp_pi._mpf_, 100) == pi.evalf(100)

    mpmath.mp.dps = 15

    assert Float(mp_pi, 100) == Float(mp_pi._mpf_, 100) == pi.evalf(100)


def test_Float_RealElement():
    repi = RealField(dps=100)(pi.evalf(100))
    # We still have to pass the precision because Float doesn't know what
    # RealElement is, but make sure it keeps full precision from the result.
    assert Float(repi, 100) == pi.evalf(100)


def test_Float_default_to_highprec_from_str():
    s = str(pi.evalf(128))
    assert same_and_same_prec(Float(s), Float(s, ''))


def test_Float_eval():
    a = Float(3.2)
    assert (a**2).is_Float


def test_Float_issue_2107():
    a = Float(0.1, 10)
    b = Float("0.1", 10)

    assert a - a == 0
    assert a + (-a) == 0
    assert S.Zero + a - a == 0
    assert S.Zero + a + (-a) == 0

    assert b - b == 0
    assert b + (-b) == 0
    assert S.Zero + b - b == 0
    assert S.Zero + b + (-b) == 0


def test_issue_14289():
    from sympy.polys.numberfields import to_number_field

    a = 1 - sqrt(2)
    b = to_number_field(a)
    assert b.as_expr() == a
    assert b.minpoly(a).expand() == 0


def test_Float_from_tuple():
    a = Float((0, '1L', 0, 1))
    b = Float((0, '1', 0, 1))
    assert a == b


def test_Infinity():
    assert oo != 1
    assert 1*oo is oo
    assert 1 != oo
    assert oo != -oo
    assert oo != Symbol("x")**3
    assert oo + 1 is oo
    assert 2 + oo is oo
    assert 3*oo + 2 is oo
    assert S.Half**oo == 0
    assert S.Half**(-oo) is oo
    assert -oo*3 is -oo
    assert oo + oo is oo
    assert -oo + oo*(-5) is -oo
    assert 1/oo == 0
    assert 1/(-oo) == 0
    assert 8/oo == 0
    assert oo % 2 is nan
    assert 2 % oo is nan
    assert oo/oo is nan
    assert oo/-oo is nan
    assert -oo/oo is nan
    assert -oo/-oo is nan
    assert oo - oo is nan
    assert oo - -oo is oo
    assert -oo - oo is -oo
    assert -oo - -oo is nan
    assert oo + -oo is nan
    assert -oo + oo is nan
    assert oo + oo is oo
    assert -oo + oo is nan
    assert oo + -oo is nan
    assert -oo + -oo is -oo
    assert oo*oo is oo
    assert -oo*oo is -oo
    assert oo*-oo is -oo
    assert -oo*-oo is oo
    assert oo/0 is oo
    assert -oo/0 is -oo
    assert 0/oo == 0
    assert 0/-oo == 0
    assert oo*0 is nan
    assert -oo*0 is nan
    assert 0*oo is nan
    assert 0*-oo is nan
    assert oo + 0 is oo
    assert -oo + 0 is -oo
    assert 0 + oo is oo
    assert 0 + -oo is -oo
    assert oo - 0 is oo
    assert -oo - 0 is -oo
    assert 0 - oo is -oo
    assert 0 - -oo is oo
    assert oo/2 is oo
    assert -oo/2 is -oo
    assert oo/-2 is -oo
    assert -oo/-2 is oo
    assert oo*2 is oo
    assert -oo*2 is -oo
    assert oo*-2 is -oo
    assert 2/oo == 0
    assert 2/-oo == 0
    assert -2/oo == 0
    assert -2/-oo == 0
    assert 2*oo is oo
    assert 2*-oo is -oo
    assert -2*oo is -oo
    assert -2*-oo is oo
    assert 2 + oo is oo
    assert 2 - oo is -oo
    assert -2 + oo is oo
    assert -2 - oo is -oo
    assert 2 + -oo is -oo
    assert 2 - -oo is oo
    assert -2 + -oo is -oo
    assert -2 - -oo is oo
    assert S(2) + oo is oo
    assert S(2) - oo is -oo
    assert oo/I == -oo*I
    assert -oo/I == oo*I
    assert oo*float(1) == _inf and (oo*float(1)) is oo
    assert -oo*float(1) == _ninf and (-oo*float(1)) is -oo
    assert oo/float(1) == _inf and (oo/float(1)) is oo
    assert -oo/float(1) == _ninf and (-oo/float(1)) is -oo
    assert oo*float(-1) == _ninf and (oo*float(-1)) is -oo
    assert -oo*float(-1) == _inf and (-oo*float(-1)) is oo
    assert oo/float(-1) == _ninf and (oo/float(-1)) is -oo
    assert -oo/float(-1) == _inf and (-oo/float(-1)) is oo
    assert oo + float(1) == _inf and (oo + float(1)) is oo
    assert -oo + float(1) == _ninf and (-oo + float(1)) is -oo
    assert oo - float(1) == _inf and (oo - float(1)) is oo
    assert -oo - float(1) == _ninf and (-oo - float(1)) is -oo
    assert float(1)*oo == _inf and (float(1)*oo) is oo
    assert float(1)*-oo == _ninf and (float(1)*-oo) is -oo
    assert float(1)/oo == 0
    assert float(1)/-oo == 0
    assert float(-1)*oo == _ninf and (float(-1)*oo) is -oo
    assert float(-1)*-oo == _inf and (float(-1)*-oo) is oo
    assert float(-1)/oo == 0
    assert float(-1)/-oo == 0
    assert float(1) + oo is oo
    assert float(1) + -oo is -oo
    assert float(1) - oo is -oo
    assert float(1) - -oo is oo
    assert oo == float(oo)
    assert (oo != float(oo)) is False
    assert type(float(oo)) is float
    assert -oo == float(-oo)
    assert (-oo != float(-oo)) is False
    assert type(float(-oo)) is float

    assert Float('nan') is nan
    assert nan*1.0 is nan
    assert -1.0*nan is nan
    assert nan*oo is nan
    assert nan*-oo is nan
    assert nan/oo is nan
    assert nan/-oo is nan
    assert nan + oo is nan
    assert nan + -oo is nan
    assert nan - oo is nan
    assert nan - -oo is nan
    assert -oo * S.Zero is nan

    assert oo*nan is nan
    assert -oo*nan is nan
    assert oo/nan is nan
    assert -oo/nan is nan
    assert oo + nan is nan
    assert -oo + nan is nan
    assert oo - nan is nan
    assert -oo - nan is nan
    assert S.Zero * oo is nan
    assert oo.is_Rational is False
    assert isinstance(oo, Rational) is False

    assert S.One/oo == 0
    assert -S.One/oo == 0
    assert S.One/-oo == 0
    assert -S.One/-oo == 0
    assert S.One*oo is oo
    assert -S.One*oo is -oo
    assert S.One*-oo is -oo
    assert -S.One*-oo is oo
    assert S.One/nan is nan
    assert S.One - -oo is oo
    assert S.One + nan is nan
    assert S.One - nan is nan
    assert nan - S.One is nan
    assert nan/S.One is nan
    assert -oo - S.One is -oo


def test_Infinity_2():
    x = Symbol('x')
    assert oo*x != oo
    assert oo*(pi - 1) is oo
    assert oo*(1 - pi) is -oo

    assert (-oo)*x != -oo
    assert (-oo)*(pi - 1) is -oo
    assert (-oo)*(1 - pi) is oo

    assert (-1)**S.NaN is S.NaN
    assert oo - _inf is S.NaN
    assert oo + _ninf is S.NaN
    assert oo*0 is S.NaN
    assert oo/_inf is S.NaN
    assert oo/_ninf is S.NaN
    assert oo**S.NaN is S.NaN
    assert -oo + _inf is S.NaN
    assert -oo - _ninf is S.NaN
    assert -oo*S.NaN is S.NaN
    assert -oo*0 is S.NaN
    assert -oo/_inf is S.NaN
    assert -oo/_ninf is S.NaN
    assert -oo/S.NaN is S.NaN
    assert abs(-oo) is oo
    assert all((-oo)**i is S.NaN for i in (oo, -oo, S.NaN))
    assert (-oo)**3 is -oo
    assert (-oo)**2 is oo
    assert abs(S.ComplexInfinity) is oo


def test_Mul_Infinity_Zero():
    assert Float(0)*_inf is nan
    assert Float(0)*_ninf is nan
    assert Float(0)*_inf is nan
    assert Float(0)*_ninf is nan
    assert _inf*Float(0) is nan
    assert _ninf*Float(0) is nan
    assert _inf*Float(0) is nan
    assert _ninf*Float(0) is nan


def test_Div_By_Zero():
    assert 1/S.Zero is zoo
    assert 1/Float(0) is zoo
    assert 0/S.Zero is nan
    assert 0/Float(0) is nan
    assert S.Zero/0 is nan
    assert Float(0)/0 is nan
    assert -1/S.Zero is zoo
    assert -1/Float(0) is zoo


@_both_exp_pow
def test_Infinity_inequations():
    assert oo > pi
    assert not (oo < pi)
    assert exp(-3) < oo

    assert _inf > pi
    assert not (_inf < pi)
    assert exp(-3) < _inf

    raises(TypeError, lambda: oo < I)
    raises(TypeError, lambda: oo <= I)
    raises(TypeError, lambda: oo > I)
    raises(TypeError, lambda: oo >= I)
    raises(TypeError, lambda: -oo < I)
    raises(TypeError, lambda: -oo <= I)
    raises(TypeError, lambda: -oo > I)
    raises(TypeError, lambda: -oo >= I)

    raises(TypeError, lambda: I < oo)
    raises(TypeError, lambda: I <= oo)
    raises(TypeError, lambda: I > oo)
    raises(TypeError, lambda: I >= oo)
    raises(TypeError, lambda: I < -oo)
    raises(TypeError, lambda: I <= -oo)
    raises(TypeError, lambda: I > -oo)
    raises(TypeError, lambda: I >= -oo)

    assert oo > -oo and oo >= -oo
    assert (oo < -oo) == False and (oo <= -oo) == False
    assert -oo < oo and -oo <= oo
    assert (-oo > oo) == False and (-oo >= oo) == False

    assert (oo < oo) == False  # issue 7775
    assert (oo > oo) == False
    assert (-oo > -oo) == False and (-oo < -oo) == False
    assert oo >= oo and oo <= oo and -oo >= -oo and -oo <= -oo
    assert (-oo < -_inf) ==  False
    assert (oo > _inf) == False
    assert -oo >= -_inf
    assert oo <= _inf

    x = Symbol('x')
    b = Symbol('b', finite=True, real=True)
    assert (x < oo) == Lt(x, oo)  # issue 7775
    assert b < oo and b > -oo and b <= oo and b >= -oo
    assert oo > b and oo >= b and (oo < b) == False and (oo <= b) == False
    assert (-oo > b) == False and (-oo >= b) == False and -oo < b and -oo <= b
    assert (oo < x) == Lt(oo, x) and (oo > x) == Gt(oo, x)
    assert (oo <= x) == Le(oo, x) and (oo >= x) == Ge(oo, x)
    assert (-oo < x) == Lt(-oo, x) and (-oo > x) == Gt(-oo, x)
    assert (-oo <= x) == Le(-oo, x) and (-oo >= x) == Ge(-oo, x)


def test_NaN():
    assert nan is nan
    assert nan != 1
    assert 1*nan is nan
    assert 1 != nan
    assert -nan is nan
    assert oo != Symbol("x")**3
    assert 2 + nan is nan
    assert 3*nan + 2 is nan
    assert -nan*3 is nan
    assert nan + nan is nan
    assert -nan + nan*(-5) is nan
    assert 8/nan is nan
    raises(TypeError, lambda: nan > 0)
    raises(TypeError, lambda: nan < 0)
    raises(TypeError, lambda: nan >= 0)
    raises(TypeError, lambda: nan <= 0)
    raises(TypeError, lambda: 0 < nan)
    raises(TypeError, lambda: 0 > nan)
    raises(TypeError, lambda: 0 <= nan)
    raises(TypeError, lambda: 0 >= nan)
    assert nan**0 == 1  # as per IEEE 754
    assert 1**nan is nan # IEEE 754 is not the best choice for symbolic work
    # test Pow._eval_power's handling of NaN
    assert Pow(nan, 0, evaluate=False)**2 == 1
    for n in (1, 1., S.One, S.NegativeOne, Float(1)):
        assert n + nan is nan
        assert n - nan is nan
        assert nan + n is nan
        assert nan - n is nan
        assert n/nan is nan
        assert nan/n is nan


def test_special_numbers():
    assert isinstance(S.NaN, Number) is True
    assert isinstance(S.Infinity, Number) is True
    assert isinstance(S.NegativeInfinity, Number) is True

    assert S.NaN.is_number is True
    assert S.Infinity.is_number is True
    assert S.NegativeInfinity.is_number is True
    assert S.ComplexInfinity.is_number is True

    assert isinstance(S.NaN, Rational) is False
    assert isinstance(S.Infinity, Rational) is False
    assert isinstance(S.NegativeInfinity, Rational) is False

    assert S.NaN.is_rational is not True
    assert S.Infinity.is_rational is not True
    assert S.NegativeInfinity.is_rational is not True


def test_powers():
    assert integer_nthroot(1, 2) == (1, True)
    assert integer_nthroot(1, 5) == (1, True)
    assert integer_nthroot(2, 1) == (2, True)
    assert integer_nthroot(2, 2) == (1, False)
    assert integer_nthroot(2, 5) == (1, False)
    assert integer_nthroot(4, 2) == (2, True)
    assert integer_nthroot(123**25, 25) == (123, True)
    assert integer_nthroot(123**25 + 1, 25) == (123, False)
    assert integer_nthroot(123**25 - 1, 25) == (122, False)
    assert integer_nthroot(1, 1) == (1, True)
    assert integer_nthroot(0, 1) == (0, True)
    assert integer_nthroot(0, 3) == (0, True)
    assert integer_nthroot(10000, 1) == (10000, True)
    assert integer_nthroot(4, 2) == (2, True)
    assert integer_nthroot(16, 2) == (4, True)
    assert integer_nthroot(26, 2) == (5, False)
    assert integer_nthroot(1234567**7, 7) == (1234567, True)
    assert integer_nthroot(1234567**7 + 1, 7) == (1234567, False)
    assert integer_nthroot(1234567**7 - 1, 7) == (1234566, False)
    b = 25**1000
    assert integer_nthroot(b, 1000) == (25, True)
    assert integer_nthroot(b + 1, 1000) == (25, False)
    assert integer_nthroot(b - 1, 1000) == (24, False)
    c = 10**400
    c2 = c**2
    assert integer_nthroot(c2, 2) == (c, True)
    assert integer_nthroot(c2 + 1, 2) == (c, False)
    assert integer_nthroot(c2 - 1, 2) == (c - 1, False)
    assert integer_nthroot(2, 10**10) == (1, False)

    p, r = integer_nthroot(int(factorial(10000)), 100)
    assert p % (10**10) == 5322420655
    assert not r

    # Test that this is fast
    assert integer_nthroot(2, 10**10) == (1, False)

    # output should be int if possible
    assert type(integer_nthroot(2**61, 2)[0]) is int


def test_integer_nthroot_overflow():
    assert integer_nthroot(10**(50*50), 50) == (10**50, True)
    assert integer_nthroot(10**100000, 10000) == (10**10, True)


def test_integer_log():
    raises(ValueError, lambda: integer_log(2, 1))
    raises(ValueError, lambda: integer_log(0, 2))
    raises(ValueError, lambda: integer_log(1.1, 2))
    raises(ValueError, lambda: integer_log(1, 2.2))

    assert integer_log(1, 2) == (0, True)
    assert integer_log(1, 3) == (0, True)
    assert integer_log(2, 3) == (0, False)
    assert integer_log(3, 3) == (1, True)
    assert integer_log(3*2, 3) == (1, False)
    assert integer_log(3**2, 3) == (2, True)
    assert integer_log(3*4, 3) == (2, False)
    assert integer_log(3**3, 3) == (3, True)
    assert integer_log(27, 5) == (2, False)
    assert integer_log(2, 3) == (0, False)
    assert integer_log(-4, 2) == (2, False)
    assert integer_log(-16, 4) == (0, False)
    assert integer_log(-4, -2) == (2, False)
    assert integer_log(4, -2) == (2, True)
    assert integer_log(-8, -2) == (3, True)
    assert integer_log(8, -2) == (3, False)
    assert integer_log(-9, 3) == (0, False)
    assert integer_log(-9, -3) == (2, False)
    assert integer_log(9, -3) == (2, True)
    assert integer_log(-27, -3) == (3, True)
    assert integer_log(27, -3) == (3, False)


def test_isqrt():
    from math import sqrt as _sqrt
    limit = 4503599761588223
    assert int(_sqrt(limit)) == integer_nthroot(limit, 2)[0]
    assert int(_sqrt(limit + 1)) != integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + 1) == integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + S.Half) == integer_nthroot(limit, 2)[0]
    assert isqrt(limit + 1 + S.Half) == integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + 2 + S.Half) == integer_nthroot(limit + 2, 2)[0]

    # Regression tests for https://github.com/sympy/sympy/issues/17034
    assert isqrt(4503599761588224) == 67108864
    assert isqrt(9999999999999999) == 99999999

    # Other corner cases, especially involving non-integers.
    raises(ValueError, lambda: isqrt(-1))
    raises(ValueError, lambda: isqrt(-10**1000))
    raises(ValueError, lambda: isqrt(Rational(-1, 2)))

    tiny = Rational(1, 10**1000)
    raises(ValueError, lambda: isqrt(-tiny))
    assert isqrt(1-tiny) == 0
    assert isqrt(4503599761588224-tiny) == 67108864
    assert isqrt(10**100 - tiny) == 10**50 - 1


def test_powers_Integer():
    """Test Integer._eval_power"""
    # check infinity
    assert S.One ** S.Infinity is S.NaN
    assert S.NegativeOne** S.Infinity is S.NaN
    assert S(2) ** S.Infinity is S.Infinity
    assert S(-2)** S.Infinity == zoo
    assert S(0) ** S.Infinity is S.Zero

    # check Nan
    assert S.One ** S.NaN is S.NaN
    assert S.NegativeOne ** S.NaN is S.NaN

    # check for exact roots
    assert S.NegativeOne ** Rational(6, 5) == - (-1)**(S.One/5)
    assert sqrt(S(4)) == 2
    assert sqrt(S(-4)) == I * 2
    assert S(16) ** Rational(1, 4) == 2
    assert S(-16) ** Rational(1, 4) == 2 * (-1)**Rational(1, 4)
    assert S(9) ** Rational(3, 2) == 27
    assert S(-9) ** Rational(3, 2) == -27*I
    assert S(27) ** Rational(2, 3) == 9
    assert S(-27) ** Rational(2, 3) == 9 * (S.NegativeOne ** Rational(2, 3))
    assert (-2) ** Rational(-2, 1) == Rational(1, 4)

    # not exact roots
    assert sqrt(-3) == I*sqrt(3)
    assert (3) ** (Rational(3, 2)) == 3 * sqrt(3)
    assert (-3) ** (Rational(3, 2)) == - 3 * sqrt(-3)
    assert (-3) ** (Rational(5, 2)) == 9 * I * sqrt(3)
    assert (-3) ** (Rational(7, 2)) == - I * 27 * sqrt(3)
    assert (2) ** (Rational(3, 2)) == 2 * sqrt(2)
    assert (2) ** (Rational(-3, 2)) == sqrt(2) / 4
    assert (81) ** (Rational(2, 3)) == 9 * (S(3) ** (Rational(2, 3)))
    assert (-81) ** (Rational(2, 3)) == 9 * (S(-3) ** (Rational(2, 3)))
    assert (-3) ** Rational(-7, 3) == \
        -(-1)**Rational(2, 3)*3**Rational(2, 3)/27
    assert (-3) ** Rational(-2, 3) == \
        -(-1)**Rational(1, 3)*3**Rational(1, 3)/3

    # join roots
    assert sqrt(6) + sqrt(24) == 3*sqrt(6)
    assert sqrt(2) * sqrt(3) == sqrt(6)

    # separate symbols & constansts
    x = Symbol("x")
    assert sqrt(49 * x) == 7 * sqrt(x)
    assert sqrt((3 - sqrt(pi)) ** 2) == 3 - sqrt(pi)

    # check that it is fast for big numbers
    assert (2**64 + 1) ** Rational(4, 3)
    assert (2**64 + 1) ** Rational(17, 25)

    # negative rational power and negative base
    assert (-3) ** Rational(-7, 3) == \
        -(-1)**Rational(2, 3)*3**Rational(2, 3)/27
    assert (-3) ** Rational(-2, 3) == \
        -(-1)**Rational(1, 3)*3**Rational(1, 3)/3
    assert (-2) ** Rational(-10, 3) == \
        (-1)**Rational(2, 3)*2**Rational(2, 3)/16
    assert abs(Pow(-2, Rational(-10, 3)).n() -
        Pow(-2, Rational(-10, 3), evaluate=False).n()) < 1e-16

    # negative base and rational power with some simplification
    assert (-8) ** Rational(2, 5) == \
        2*(-1)**Rational(2, 5)*2**Rational(1, 5)
    assert (-4) ** Rational(9, 5) == \
        -8*(-1)**Rational(4, 5)*2**Rational(3, 5)

    assert S(1234).factors() == {617: 1, 2: 1}
    assert Rational(2*3, 3*5*7).factors() == {2: 1, 5: -1, 7: -1}

    # test that eval_power factors numbers bigger than
    # the current limit in factor_trial_division (2**15)
    from sympy.ntheory.generate import nextprime
    n = nextprime(2**15)
    assert sqrt(n**2) == n
    assert sqrt(n**3) == n*sqrt(n)
    assert sqrt(4*n) == 2*sqrt(n)

    # check that factors of base with powers sharing gcd with power are removed
    assert (2**4*3)**Rational(1, 6) == 2**Rational(2, 3)*3**Rational(1, 6)
    assert (2**4*3)**Rational(5, 6) == 8*2**Rational(1, 3)*3**Rational(5, 6)

    # check that bases sharing a gcd are exptracted
    assert 2**Rational(1, 3)*3**Rational(1, 4)*6**Rational(1, 5) == \
        2**Rational(8, 15)*3**Rational(9, 20)
    assert sqrt(8)*24**Rational(1, 3)*6**Rational(1, 5) == \
        4*2**Rational(7, 10)*3**Rational(8, 15)
    assert sqrt(8)*(-24)**Rational(1, 3)*(-6)**Rational(1, 5) == \
        4*(-3)**Rational(8, 15)*2**Rational(7, 10)
    assert 2**Rational(1, 3)*2**Rational(8, 9) == 2*2**Rational(2, 9)
    assert 2**Rational(2, 3)*6**Rational(1, 3) == 2*3**Rational(1, 3)
    assert 2**Rational(2, 3)*6**Rational(8, 9) == \
        2*2**Rational(5, 9)*3**Rational(8, 9)
    assert (-2)**Rational(2, S(3))*(-4)**Rational(1, S(3)) == -2*2**Rational(1, 3)
    assert 3*Pow(3, 2, evaluate=False) == 3**3
    assert 3*Pow(3, Rational(-1, 3), evaluate=False) == 3**Rational(2, 3)
    assert (-2)**Rational(1, 3)*(-3)**Rational(1, 4)*(-5)**Rational(5, 6) == \
        -(-1)**Rational(5, 12)*2**Rational(1, 3)*3**Rational(1, 4) * \
        5**Rational(5, 6)

    assert Integer(-2)**Symbol('', even=True) == \
        Integer(2)**Symbol('', even=True)
    assert (-1)**Float(.5) == 1.0*I


def test_powers_Rational():
    """Test Rational._eval_power"""
    # check infinity
    assert S.Half ** S.Infinity == 0
    assert Rational(3, 2) ** S.Infinity is S.Infinity
    assert Rational(-1, 2) ** S.Infinity == 0
    assert Rational(-3, 2) ** S.Infinity == zoo

    # check Nan
    assert Rational(3, 4) ** S.NaN is S.NaN
    assert Rational(-2, 3) ** S.NaN is S.NaN

    # exact roots on numerator
    assert sqrt(Rational(4, 3)) == 2 * sqrt(3) / 3
    assert Rational(4, 3) ** Rational(3, 2) == 8 * sqrt(3) / 9
    assert sqrt(Rational(-4, 3)) == I * 2 * sqrt(3) / 3
    assert Rational(-4, 3) ** Rational(3, 2) == - I * 8 * sqrt(3) / 9
    assert Rational(27, 2) ** Rational(1, 3) == 3 * (2 ** Rational(2, 3)) / 2
    assert Rational(5**3, 8**3) ** Rational(4, 3) == Rational(5**4, 8**4)

    # exact root on denominator
    assert sqrt(Rational(1, 4)) == S.Half
    assert sqrt(Rational(1, -4)) == I * S.Half
    assert sqrt(Rational(3, 4)) == sqrt(3) / 2
    assert sqrt(Rational(3, -4)) == I * sqrt(3) / 2
    assert Rational(5, 27) ** Rational(1, 3) == (5 ** Rational(1, 3)) / 3

    # not exact roots
    assert sqrt(S.Half) == sqrt(2) / 2
    assert sqrt(Rational(-4, 7)) == I * sqrt(Rational(4, 7))
    assert Rational(-3, 2)**Rational(-7, 3) == \
        -4*(-1)**Rational(2, 3)*2**Rational(1, 3)*3**Rational(2, 3)/27
    assert Rational(-3, 2)**Rational(-2, 3) == \
        -(-1)**Rational(1, 3)*2**Rational(2, 3)*3**Rational(1, 3)/3
    assert Rational(-3, 2)**Rational(-10, 3) == \
        8*(-1)**Rational(2, 3)*2**Rational(1, 3)*3**Rational(2, 3)/81
    assert abs(Pow(Rational(-2, 3), Rational(-7, 4)).n() -
        Pow(Rational(-2, 3), Rational(-7, 4), evaluate=False).n()) < 1e-16

    # negative integer power and negative rational base
    assert Rational(-2, 3) ** Rational(-2, 1) == Rational(9, 4)

    a = Rational(1, 10)
    assert a**Float(a, 2) == Float(a, 2)**Float(a, 2)
    assert Rational(-2, 3)**Symbol('', even=True) == \
        Rational(2, 3)**Symbol('', even=True)


def test_powers_Float():
    assert str((S('-1/10')**S('3/10')).n()) == str(Float(-.1)**(.3))


def test_lshift_Integer():
    assert Integer(0) << Integer(2) == Integer(0)
    assert Integer(0) << 2 == Integer(0)
    assert 0 << Integer(2) == Integer(0)

    assert Integer(0b11) << Integer(0) == Integer(0b11)
    assert Integer(0b11) << 0 == Integer(0b11)
    assert 0b11 << Integer(0) == Integer(0b11)

    assert Integer(0b11) << Integer(2) == Integer(0b11 << 2)
    assert Integer(0b11) << 2 == Integer(0b11 << 2)
    assert 0b11 << Integer(2) == Integer(0b11 << 2)

    assert Integer(-0b11) << Integer(2) == Integer(-0b11 << 2)
    assert Integer(-0b11) << 2 == Integer(-0b11 << 2)
    assert -0b11 << Integer(2) == Integer(-0b11 << 2)

    raises(TypeError, lambda: Integer(2) << 0.0)
    raises(TypeError, lambda: 0.0 << Integer(2))
    raises(ValueError, lambda: Integer(1) << Integer(-1))


def test_rshift_Integer():
    assert Integer(0) >> Integer(2) == Integer(0)
    assert Integer(0) >> 2 == Integer(0)
    assert 0 >> Integer(2) == Integer(0)

    assert Integer(0b11) >> Integer(0) == Integer(0b11)
    assert Integer(0b11) >> 0 == Integer(0b11)
    assert 0b11 >> Integer(0) == Integer(0b11)

    assert Integer(0b11) >> Integer(2) == Integer(0)
    assert Integer(0b11) >> 2 == Integer(0)
    assert 0b11 >> Integer(2) == Integer(0)

    assert Integer(-0b11) >> Integer(2) == Integer(-1)
    assert Integer(-0b11) >> 2 == Integer(-1)
    assert -0b11 >> Integer(2) == Integer(-1)

    assert Integer(0b1100) >> Integer(2) == Integer(0b1100 >> 2)
    assert Integer(0b1100) >> 2 == Integer(0b1100 >> 2)
    assert 0b1100 >> Integer(2) == Integer(0b1100 >> 2)

    assert Integer(-0b1100) >> Integer(2) == Integer(-0b1100 >> 2)
    assert Integer(-0b1100) >> 2 == Integer(-0b1100 >> 2)
    assert -0b1100 >> Integer(2) == Integer(-0b1100 >> 2)

    raises(TypeError, lambda: Integer(0b10) >> 0.0)
    raises(TypeError, lambda: 0.0 >> Integer(2))
    raises(ValueError, lambda: Integer(1) >> Integer(-1))


def test_and_Integer():
    assert Integer(0b01010101) & Integer(0b10101010) == Integer(0)
    assert Integer(0b01010101) & 0b10101010 == Integer(0)
    assert 0b01010101 & Integer(0b10101010) == Integer(0)

    assert Integer(0b01010101) & Integer(0b11011011) == Integer(0b01010001)
    assert Integer(0b01010101) & 0b11011011 == Integer(0b01010001)
    assert 0b01010101 & Integer(0b11011011) == Integer(0b01010001)

    assert -Integer(0b01010101) & Integer(0b11011011) == Integer(-0b01010101 & 0b11011011)
    assert Integer(-0b01010101) & 0b11011011 == Integer(-0b01010101 & 0b11011011)
    assert -0b01010101 & Integer(0b11011011) == Integer(-0b01010101 & 0b11011011)

    assert Integer(0b01010101) & -Integer(0b11011011) == Integer(0b01010101 & -0b11011011)
    assert Integer(0b01010101) & -0b11011011 == Integer(0b01010101 & -0b11011011)
    assert 0b01010101 & Integer(-0b11011011) == Integer(0b01010101 & -0b11011011)

    raises(TypeError, lambda: Integer(2) & 0.0)
    raises(TypeError, lambda: 0.0 & Integer(2))


def test_xor_Integer():
    assert Integer(0b01010101) ^ Integer(0b11111111) == Integer(0b10101010)
    assert Integer(0b01010101) ^ 0b11111111 == Integer(0b10101010)
    assert 0b01010101 ^ Integer(0b11111111) == Integer(0b10101010)

    assert Integer(0b01010101) ^ Integer(0b11011011) == Integer(0b10001110)
    assert Integer(0b01010101) ^ 0b11011011 == Integer(0b10001110)
    assert 0b01010101 ^ Integer(0b11011011) == Integer(0b10001110)

    assert -Integer(0b01010101) ^ Integer(0b11011011) == Integer(-0b01010101 ^ 0b11011011)
    assert Integer(-0b01010101) ^ 0b11011011 == Integer(-0b01010101 ^ 0b11011011)
    assert -0b01010101 ^ Integer(0b11011011) == Integer(-0b01010101 ^ 0b11011011)

    assert Integer(0b01010101) ^ -Integer(0b11011011) == Integer(0b01010101 ^ -0b11011011)
    assert Integer(0b01010101) ^ -0b11011011 == Integer(0b01010101 ^ -0b11011011)
    assert 0b01010101 ^ Integer(-0b11011011) == Integer(0b01010101 ^ -0b11011011)

    raises(TypeError, lambda: Integer(2) ^ 0.0)
    raises(TypeError, lambda: 0.0 ^ Integer(2))


def test_or_Integer():
    assert Integer(0b01010101) | Integer(0b10101010) == Integer(0b11111111)
    assert Integer(0b01010101) | 0b10101010 == Integer(0b11111111)
    assert 0b01010101 | Integer(0b10101010) == Integer(0b11111111)

    assert Integer(0b01010101) | Integer(0b11011011) == Integer(0b11011111)
    assert Integer(0b01010101) | 0b11011011 == Integer(0b11011111)
    assert 0b01010101 | Integer(0b11011011) == Integer(0b11011111)

    assert -Integer(0b01010101) | Integer(0b11011011) == Integer(-0b01010101 | 0b11011011)
    assert Integer(-0b01010101) | 0b11011011 == Integer(-0b01010101 | 0b11011011)
    assert -0b01010101 | Integer(0b11011011) == Integer(-0b01010101 | 0b11011011)

    assert Integer(0b01010101) | -Integer(0b11011011) == Integer(0b01010101 | -0b11011011)
    assert Integer(0b01010101) | -0b11011011 == Integer(0b01010101 | -0b11011011)
    assert 0b01010101 | Integer(-0b11011011) == Integer(0b01010101 | -0b11011011)

    raises(TypeError, lambda: Integer(2) | 0.0)
    raises(TypeError, lambda: 0.0 | Integer(2))


def test_invert_Integer():
    assert ~Integer(0b01010101) == Integer(-0b01010110)
    assert ~Integer(0b01010101) == Integer(~0b01010101)
    assert ~(~Integer(0b01010101)) == Integer(0b01010101)


def test_abs1():
    assert Rational(1, 6) != Rational(-1, 6)
    assert abs(Rational(1, 6)) == abs(Rational(-1, 6))


def test_accept_int():
    assert not Float(4) == 4
    assert Float(4) != 4
    assert Float(4) == 4.0


def test_dont_accept_str():
    assert Float("0.2") != "0.2"
    assert not (Float("0.2") == "0.2")


def test_int():
    a = Rational(5)
    assert int(a) == 5
    a = Rational(9, 10)
    assert int(a) == int(-a) == 0
    assert 1/(-1)**Rational(2, 3) == -(-1)**Rational(1, 3)
    # issue 10368
    a = Rational(32442016954, 78058255275)
    assert type(int(a)) is type(int(-a)) is int


def test_int_NumberSymbols():
    assert int(Catalan) == 0
    assert int(EulerGamma) == 0
    assert int(pi) == 3
    assert int(E) == 2
    assert int(GoldenRatio) == 1
    assert int(TribonacciConstant) == 1
    for i in [Catalan, E, EulerGamma, GoldenRatio, TribonacciConstant, pi]:
        a, b = i.approximation_interval(Integer)
        ia = int(i)
        assert ia == a
        assert isinstance(ia, int)
        assert b == a + 1
        assert a.is_Integer and b.is_Integer


def test_real_bug():
    x = Symbol("x")
    assert str(2.0*x*x) in ["(2.0*x)*x", "2.0*x**2", "2.00000000000000*x**2"]
    assert str(2.1*x*x) != "(2.0*x)*x"


def test_bug_sqrt():
    assert ((sqrt(Rational(2)) + 1)*(sqrt(Rational(2)) - 1)).expand() == 1


def test_pi_Pi():
    "Test that pi (instance) is imported, but Pi (class) is not"
    from sympy import pi  # noqa
    with raises(ImportError):
        from sympy import Pi  # noqa


def test_no_len():
    # there should be no len for numbers
    raises(TypeError, lambda: len(Rational(2)))
    raises(TypeError, lambda: len(Rational(2, 3)))
    raises(TypeError, lambda: len(Integer(2)))


def test_issue_3321():
    assert sqrt(Rational(1, 5)) == Rational(1, 5)**S.Half
    assert 5 * sqrt(Rational(1, 5)) == sqrt(5)


def test_issue_3692():
    assert ((-1)**Rational(1, 6)).expand(complex=True) == I/2 + sqrt(3)/2
    assert ((-5)**Rational(1, 6)).expand(complex=True) == \
        5**Rational(1, 6)*I/2 + 5**Rational(1, 6)*sqrt(3)/2
    assert ((-64)**Rational(1, 6)).expand(complex=True) == I + sqrt(3)


def test_issue_3423():
    x = Symbol("x")
    assert sqrt(x - 1).as_base_exp() == (x - 1, S.Half)
    assert sqrt(x - 1) != I*sqrt(1 - x)


def test_issue_3449():
    x = Symbol("x")
    assert sqrt(x - 1).subs(x, 5) == 2


def test_issue_13890():
    x = Symbol("x")
    e = (-x/4 - S.One/12)**x - 1
    f = simplify(e)
    a = Rational(9, 5)
    assert abs(e.subs(x,a).evalf() - f.subs(x,a).evalf()) < 1e-15


def test_Integer_factors():
    def F(i):
        return Integer(i).factors()

    assert F(1) == {}
    assert F(2) == {2: 1}
    assert F(3) == {3: 1}
    assert F(4) == {2: 2}
    assert F(5) == {5: 1}
    assert F(6) == {2: 1, 3: 1}
    assert F(7) == {7: 1}
    assert F(8) == {2: 3}
    assert F(9) == {3: 2}
    assert F(10) == {2: 1, 5: 1}
    assert F(11) == {11: 1}
    assert F(12) == {2: 2, 3: 1}
    assert F(13) == {13: 1}
    assert F(14) == {2: 1, 7: 1}
    assert F(15) == {3: 1, 5: 1}
    assert F(16) == {2: 4}
    assert F(17) == {17: 1}
    assert F(18) == {2: 1, 3: 2}
    assert F(19) == {19: 1}
    assert F(20) == {2: 2, 5: 1}
    assert F(21) == {3: 1, 7: 1}
    assert F(22) == {2: 1, 11: 1}
    assert F(23) == {23: 1}
    assert F(24) == {2: 3, 3: 1}
    assert F(25) == {5: 2}
    assert F(26) == {2: 1, 13: 1}
    assert F(27) == {3: 3}
    assert F(28) == {2: 2, 7: 1}
    assert F(29) == {29: 1}
    assert F(30) == {2: 1, 3: 1, 5: 1}
    assert F(31) == {31: 1}
    assert F(32) == {2: 5}
    assert F(33) == {3: 1, 11: 1}
    assert F(34) == {2: 1, 17: 1}
    assert F(35) == {5: 1, 7: 1}
    assert F(36) == {2: 2, 3: 2}
    assert F(37) == {37: 1}
    assert F(38) == {2: 1, 19: 1}
    assert F(39) == {3: 1, 13: 1}
    assert F(40) == {2: 3, 5: 1}
    assert F(41) == {41: 1}
    assert F(42) == {2: 1, 3: 1, 7: 1}
    assert F(43) == {43: 1}
    assert F(44) == {2: 2, 11: 1}
    assert F(45) == {3: 2, 5: 1}
    assert F(46) == {2: 1, 23: 1}
    assert F(47) == {47: 1}
    assert F(48) == {2: 4, 3: 1}
    assert F(49) == {7: 2}
    assert F(50) == {2: 1, 5: 2}
    assert F(51) == {3: 1, 17: 1}


def test_Rational_factors():
    def F(p, q, visual=None):
        return Rational(p, q).factors(visual=visual)

    assert F(2, 3) == {2: 1, 3: -1}
    assert F(2, 9) == {2: 1, 3: -2}
    assert F(2, 15) == {2: 1, 3: -1, 5: -1}
    assert F(6, 10) == {3: 1, 5: -1}


def test_issue_4107():
    assert pi*(E + 10) + pi*(-E - 10) != 0
    assert pi*(E + 10**10) + pi*(-E - 10**10) != 0
    assert pi*(E + 10**20) + pi*(-E - 10**20) != 0
    assert pi*(E + 10**80) + pi*(-E - 10**80) != 0

    assert (pi*(E + 10) + pi*(-E - 10)).expand() == 0
    assert (pi*(E + 10**10) + pi*(-E - 10**10)).expand() == 0
    assert (pi*(E + 10**20) + pi*(-E - 10**20)).expand() == 0
    assert (pi*(E + 10**80) + pi*(-E - 10**80)).expand() == 0


def test_IntegerInteger():
    a = Integer(4)
    b = Integer(a)

    assert a == b


def test_Rational_gcd_lcm_cofactors():
    assert Integer(4).gcd(2) == Integer(2)
    assert Integer(4).lcm(2) == Integer(4)
    assert Integer(4).gcd(Integer(2)) == Integer(2)
    assert Integer(4).lcm(Integer(2)) == Integer(4)
    a, b = 720**99911, 480**12342
    assert Integer(a).lcm(b) == a*b/Integer(a).gcd(b)

    assert Integer(4).gcd(3) == Integer(1)
    assert Integer(4).lcm(3) == Integer(12)
    assert Integer(4).gcd(Integer(3)) == Integer(1)
    assert Integer(4).lcm(Integer(3)) == Integer(12)

    assert Rational(4, 3).gcd(2) == Rational(2, 3)
    assert Rational(4, 3).lcm(2) == Integer(4)
    assert Rational(4, 3).gcd(Integer(2)) == Rational(2, 3)
    assert Rational(4, 3).lcm(Integer(2)) == Integer(4)

    assert Integer(4).gcd(Rational(2, 9)) == Rational(2, 9)
    assert Integer(4).lcm(Rational(2, 9)) == Integer(4)

    assert Rational(4, 3).gcd(Rational(2, 9)) == Rational(2, 9)
    assert Rational(4, 3).lcm(Rational(2, 9)) == Rational(4, 3)
    assert Rational(4, 5).gcd(Rational(2, 9)) == Rational(2, 45)
    assert Rational(4, 5).lcm(Rational(2, 9)) == Integer(4)
    assert Rational(5, 9).lcm(Rational(3, 7)) == Rational(Integer(5).lcm(3),Integer(9).gcd(7))

    assert Integer(4).cofactors(2) == (Integer(2), Integer(2), Integer(1))
    assert Integer(4).cofactors(Integer(2)) == \
        (Integer(2), Integer(2), Integer(1))

    assert Integer(4).gcd(Float(2.0)) == Float(1.0)
    assert Integer(4).lcm(Float(2.0)) == Float(8.0)
    assert Integer(4).cofactors(Float(2.0)) == (Float(1.0), Float(4.0), Float(2.0))

    assert S.Half.gcd(Float(2.0)) == Float(1.0)
    assert S.Half.lcm(Float(2.0)) == Float(1.0)
    assert S.Half.cofactors(Float(2.0)) == \
        (Float(1.0), Float(0.5), Float(2.0))


def test_Float_gcd_lcm_cofactors():
    assert Float(2.0).gcd(Integer(4)) == Float(1.0)
    assert Float(2.0).lcm(Integer(4)) == Float(8.0)
    assert Float(2.0).cofactors(Integer(4)) == (Float(1.0), Float(2.0), Float(4.0))

    assert Float(2.0).gcd(S.Half) == Float(1.0)
    assert Float(2.0).lcm(S.Half) == Float(1.0)
    assert Float(2.0).cofactors(S.Half) == \
        (Float(1.0), Float(2.0), Float(0.5))


def test_issue_4611():
    assert abs(pi._evalf(50) - 3.14159265358979) < 1e-10
    assert abs(E._evalf(50) - 2.71828182845905) < 1e-10
    assert abs(Catalan._evalf(50) - 0.915965594177219) < 1e-10
    assert abs(EulerGamma._evalf(50) - 0.577215664901533) < 1e-10
    assert abs(GoldenRatio._evalf(50) - 1.61803398874989) < 1e-10
    assert abs(TribonacciConstant._evalf(50) - 1.83928675521416) < 1e-10

    x = Symbol("x")
    assert (pi + x).evalf() == pi.evalf() + x
    assert (E + x).evalf() == E.evalf() + x
    assert (Catalan + x).evalf() == Catalan.evalf() + x
    assert (EulerGamma + x).evalf() == EulerGamma.evalf() + x
    assert (GoldenRatio + x).evalf() == GoldenRatio.evalf() + x
    assert (TribonacciConstant + x).evalf() == TribonacciConstant.evalf() + x


@conserve_mpmath_dps
def test_conversion_to_mpmath():
    assert mpmath.mpmathify(Integer(1)) == mpmath.mpf(1)
    assert mpmath.mpmathify(S.Half) == mpmath.mpf(0.5)
    assert mpmath.mpmathify(Float('1.23', 15)) == mpmath.mpf('1.23')

    assert mpmath.mpmathify(I) == mpmath.mpc(1j)

    assert mpmath.mpmathify(1 + 2*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1.0 + 2*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1 + 2.0*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1.0 + 2.0*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(S.Half + S.Half*I) == mpmath.mpc(0.5 + 0.5j)

    assert mpmath.mpmathify(2*I) == mpmath.mpc(2j)
    assert mpmath.mpmathify(2.0*I) == mpmath.mpc(2j)
    assert mpmath.mpmathify(S.Half*I) == mpmath.mpc(0.5j)

    mpmath.mp.dps = 100
    assert mpmath.mpmathify(pi.evalf(100) + pi.evalf(100)*I) == mpmath.pi + mpmath.pi*mpmath.j
    assert mpmath.mpmathify(pi.evalf(100)*I) == mpmath.pi*mpmath.j


def test_relational():
    # real
    x = S(.1)
    assert (x != cos) is True
    assert (x == cos) is False

    # rational
    x = Rational(1, 3)
    assert (x != cos) is True
    assert (x == cos) is False

    # integer defers to rational so these tests are omitted

    # number symbol
    x = pi
    assert (x != cos) is True
    assert (x == cos) is False


def test_Integer_as_index():
    assert 'hello'[Integer(2):] == 'llo'


def test_Rational_int():
    assert int( Rational(7, 5)) == 1
    assert int( S.Half) == 0
    assert int(Rational(-1, 2)) == 0
    assert int(-Rational(7, 5)) == -1


def test_zoo():
    b = Symbol('b', finite=True)
    nz = Symbol('nz', nonzero=True)
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    im = Symbol('i', imaginary=True)
    c = Symbol('c', complex=True)
    pb = Symbol('pb', positive=True)
    nb = Symbol('nb', negative=True)
    imb = Symbol('ib', imaginary=True, finite=True)
    for i in [I, S.Infinity, S.NegativeInfinity, S.Zero, S.One, S.Pi, S.Half, S(3), log(3),
              b, nz, p, n, im, pb, nb, imb, c]:
        if i.is_finite and (i.is_real or i.is_imaginary):
            assert i + zoo is zoo
            assert i - zoo is zoo
            assert zoo + i is zoo
            assert zoo - i is zoo
        elif i.is_finite is not False:
            assert (i + zoo).is_Add
            assert (i - zoo).is_Add
            assert (zoo + i).is_Add
            assert (zoo - i).is_Add
        else:
            assert (i + zoo) is S.NaN
            assert (i - zoo) is S.NaN
            assert (zoo + i) is S.NaN
            assert (zoo - i) is S.NaN

        if fuzzy_not(i.is_zero) and (i.is_extended_real or i.is_imaginary):
            assert i*zoo is zoo
            assert zoo*i is zoo
        elif i.is_zero:
            assert i*zoo is S.NaN
            assert zoo*i is S.NaN
        else:
            assert (i*zoo).is_Mul
            assert (zoo*i).is_Mul

        if fuzzy_not((1/i).is_zero) and (i.is_real or i.is_imaginary):
            assert zoo/i is zoo
        elif (1/i).is_zero:
            assert zoo/i is S.NaN
        elif i.is_zero:
            assert zoo/i is zoo
        else:
            assert (zoo/i).is_Mul

    assert (I*oo).is_Mul  # allow directed infinity
    assert zoo + zoo is S.NaN
    assert zoo * zoo is zoo
    assert zoo - zoo is S.NaN
    assert zoo/zoo is S.NaN
    assert zoo**zoo is S.NaN
    assert zoo**0 is S.One
    assert zoo**2 is zoo
    assert 1/zoo is S.Zero

    assert Mul.flatten([S.NegativeOne, oo, S(0)]) == ([S.NaN], [], None)


def test_issue_4122():
    x = Symbol('x', nonpositive=True)
    assert oo + x is oo
    x = Symbol('x', extended_nonpositive=True)
    assert (oo + x).is_Add
    x = Symbol('x', finite=True)
    assert (oo + x).is_Add  # x could be imaginary
    x = Symbol('x', nonnegative=True)
    assert oo + x is oo
    x = Symbol('x', extended_nonnegative=True)
    assert oo + x is oo
    x = Symbol('x', finite=True, real=True)
    assert oo + x is oo

    # similarly for negative infinity
    x = Symbol('x', nonnegative=True)
    assert -oo + x is -oo
    x = Symbol('x', extended_nonnegative=True)
    assert (-oo + x).is_Add
    x = Symbol('x', finite=True)
    assert (-oo + x).is_Add
    x = Symbol('x', nonpositive=True)
    assert -oo + x is -oo
    x = Symbol('x', extended_nonpositive=True)
    assert -oo + x is -oo
    x = Symbol('x', finite=True, real=True)
    assert -oo + x is -oo


def test_GoldenRatio_expand():
    assert GoldenRatio.expand(func=True) == S.Half + sqrt(5)/2


def test_TribonacciConstant_expand():
        assert TribonacciConstant.expand(func=True) == \
          (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3


def test_as_content_primitive():
    assert S.Zero.as_content_primitive() == (1, 0)
    assert S.Half.as_content_primitive() == (S.Half, 1)
    assert (Rational(-1, 2)).as_content_primitive() == (S.Half, -1)
    assert S(3).as_content_primitive() == (3, 1)
    assert S(3.1).as_content_primitive() == (1, 3.1)


def test_hashing_sympy_integers():
    # Test for issue 5072
    assert {Integer(3)} == {int(3)}
    assert hash(Integer(4)) == hash(int(4))


def test_rounding_issue_4172():
    assert int((E**100).round()) == \
        26881171418161354484126255515800135873611119
    assert int((pi**100).round()) == \
        51878483143196131920862615246303013562686760680406
    assert int((Rational(1)/EulerGamma**100).round()) == \
        734833795660954410469466


@XFAIL
def test_mpmath_issues():
    from mpmath.libmp.libmpf import _normalize
    import mpmath.libmp as mlib
    rnd = mlib.round_nearest
    mpf = (0, int(0), -123, -1, 53, rnd)  # nan
    assert _normalize(mpf, 53) != (0, int(0), 0, 0)
    mpf = (0, int(0), -456, -2, 53, rnd)  # +inf
    assert _normalize(mpf, 53) != (0, int(0), 0, 0)
    mpf = (1, int(0), -789, -3, 53, rnd)  # -inf
    assert _normalize(mpf, 53) != (0, int(0), 0, 0)

    from mpmath.libmp.libmpf import fnan
    assert mlib.mpf_eq(fnan, fnan)


def test_Catalan_EulerGamma_prec():
    n = GoldenRatio
    f = Float(n.n(), 5)
    assert f._mpf_ == (0, int(212079), -17, 18)
    assert f._prec == 20
    assert n._as_mpf_val(20) == f._mpf_

    n = EulerGamma
    f = Float(n.n(), 5)
    assert f._mpf_ == (0, int(302627), -19, 19)
    assert f._prec == 20
    assert n._as_mpf_val(20) == f._mpf_


def test_Catalan_rewrite():
    k = Dummy('k', integer=True, nonnegative=True)
    assert Catalan.rewrite(Sum).dummy_eq(
            Sum((-1)**k/(2*k + 1)**2, (k, 0, oo)))
    assert Catalan.rewrite() == Catalan


def test_bool_eq():
    assert 0 == False
    assert S(0) == False
    assert S(0) != S.false
    assert 1 == True
    assert S.One == True
    assert S.One != S.true


def test_Float_eq():
    # Floats with different precision should not compare equal
    assert Float(.5, 10) != Float(.5, 11) != Float(.5, 1)
    # but floats that aren't exact in base-2 still
    # don't compare the same because they have different
    # underlying mpf values
    assert Float(.12, 3) != Float(.12, 4)
    assert Float(.12, 3) != .12
    assert 0.12 != Float(.12, 3)
    assert Float('.12', 22) != .12
    # issue 11707
    # but Float/Rational -- except for 0 --
    # are exact so Rational(x) = Float(y) only if
    # Rational(x) == Rational(Float(y))
    assert Float('1.1') != Rational(11, 10)
    assert Rational(11, 10) != Float('1.1')
    # coverage
    assert not Float(3) == 2
    assert not Float(3) == Float(2)
    assert not Float(3) == 3
    assert not Float(2**2) == S.Half
    assert Float(2**2) == 4.0
    assert not Float(2**-2) == 1
    assert Float(2**-1) == 0.5
    assert not Float(2*3) == 3
    assert not Float(2*3) == 0.5
    assert Float(2*3) == 6.0
    assert not Float(2*3) == 6
    assert not Float(2*3) == 8
    assert not Float(.75) == Rational(3, 4)
    assert Float(.75) == 0.75
    assert Float(5/18) == 5/18
    # 4473
    assert Float(2.) != 3
    assert not Float((0,1,-3)) == S.One/8
    assert Float((0,1,-3)) == 1/8
    assert Float((0,1,-3)) != S.One/9
    # 16196
    assert not 2 == Float(2)  # unlike Python
    assert t**2 != t**2.0


def test_issue_6640():
    from mpmath.libmp.libmpf import finf, fninf
    # fnan is not included because Float no longer returns fnan,
    # but otherwise, the same sort of test could apply
    assert Float(finf).is_zero is False
    assert Float(fninf).is_zero is False
    assert bool(Float(0)) is False


def test_issue_6349():
    assert Float('23.e3', '')._prec == 10
    assert Float('23e3', '')._prec == 20
    assert Float('23000', '')._prec == 20
    assert Float('-23000', '')._prec == 20


def test_mpf_norm():
    assert mpf_norm((1, 0, 1, 0), 10) == mpf('0')._mpf_
    assert Float._new((1, 0, 1, 0), 10)._mpf_ == mpf('0')._mpf_


def test_latex():
    assert latex(pi) == r"\pi"
    assert latex(E) == r"e"
    assert latex(GoldenRatio) == r"\phi"
    assert latex(TribonacciConstant) == r"\text{TribonacciConstant}"
    assert latex(EulerGamma) == r"\gamma"
    assert latex(oo) == r"\infty"
    assert latex(-oo) == r"-\infty"
    assert latex(zoo) == r"\tilde{\infty}"
    assert latex(nan) == r"\text{NaN}"
    assert latex(I) == r"i"


def test_issue_7742():
    assert -oo % 1 is nan


def test_simplify_AlgebraicNumber():
    A = AlgebraicNumber
    e = 3**(S.One/6)*(3 + (135 + 78*sqrt(3))**Rational(2, 3))/(45 + 26*sqrt(3))**(S.One/3)
    assert simplify(A(e)) == A(12)  # wester test_C20

    e = (41 + 29*sqrt(2))**(S.One/5)
    assert simplify(A(e)) == A(1 + sqrt(2))  # wester test_C21

    e = (3 + 4*I)**Rational(3, 2)
    assert simplify(A(e)) == A(2 + 11*I)  # issue 4401


def test_Float_idempotence():
    x = Float('1.23', '')
    y = Float(x)
    z = Float(x, 15)
    assert same_and_same_prec(y, x)
    assert not same_and_same_prec(z, x)
    x = Float(10**20)
    y = Float(x)
    z = Float(x, 15)
    assert same_and_same_prec(y, x)
    assert not same_and_same_prec(z, x)


def test_comp1():
    # sqrt(2) = 1.414213 5623730950...
    a = sqrt(2).n(7)
    assert comp(a, 1.4142129) is False
    assert comp(a, 1.4142130)
    #                  ...
    assert comp(a, 1.4142141)
    assert comp(a, 1.4142142) is False
    assert comp(sqrt(2).n(2), '1.4')
    assert comp(sqrt(2).n(2), Float(1.4, 2), '')
    assert comp(sqrt(2).n(2), 1.4, '')
    assert comp(sqrt(2).n(2), Float(1.4, 3), '') is False
    assert comp(sqrt(2) + sqrt(3)*I, 1.4 + 1.7*I, .1)
    assert not comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*0.89, .1)
    assert comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*0.90, .1)
    assert comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*1.07, .1)
    assert not comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*1.08, .1)
    assert [(i, j)
            for i in range(130, 150)
            for j in range(170, 180)
            if comp((sqrt(2)+ I*sqrt(3)).n(3), i/100. + I*j/100.)] == [
        (141, 173), (142, 173)]
    raises(ValueError, lambda: comp(t, '1'))
    raises(ValueError, lambda: comp(t, 1))
    assert comp(0, 0.0)
    assert comp(.5, S.Half)
    assert comp(2 + sqrt(2), 2.0 + sqrt(2))
    assert not comp(0, 1)
    assert not comp(2, sqrt(2))
    assert not comp(2 + I, 2.0 + sqrt(2))
    assert not comp(2.0 + sqrt(2), 2 + I)
    assert not comp(2.0 + sqrt(2), sqrt(3))
    assert comp(1/pi.n(4), 0.3183, 1e-5)
    assert not comp(1/pi.n(4), 0.3183, 8e-6)


def test_issue_9491():
    assert oo**zoo is nan


def test_issue_10063():
    assert 2**Float(3) == Float(8)


def test_issue_10020():
    assert oo**I is S.NaN
    assert oo**(1 + I) is S.ComplexInfinity
    assert oo**(-1 + I) is S.Zero
    assert (-oo)**I is S.NaN
    assert (-oo)**(-1 + I) is S.Zero
    assert oo**t == Pow(oo, t, evaluate=False)
    assert (-oo)**t == Pow(-oo, t, evaluate=False)


def test_invert_numbers():
    assert S(2).invert(5) == 3
    assert S(2).invert(Rational(5, 2)) == S.Half
    assert S(2).invert(5.) == S.Half
    assert S(2).invert(S(5)) == 3
    assert S(2.).invert(5) == 0.5
    assert S(sqrt(2)).invert(5) == 1/sqrt(2)
    assert S(sqrt(2)).invert(sqrt(3)) == 1/sqrt(2)


def test_mod_inverse():
    assert mod_inverse(3, 11) == 4
    assert mod_inverse(5, 11) == 9
    assert mod_inverse(21124921, 521512) == 7713
    assert mod_inverse(124215421, 5125) == 2981
    assert mod_inverse(214, 12515) == 1579
    assert mod_inverse(5823991, 3299) == 1442
    assert mod_inverse(123, 44) == 39
    assert mod_inverse(2, 5) == 3
    assert mod_inverse(-2, 5) == 2
    assert mod_inverse(2, -5) == -2
    assert mod_inverse(-2, -5) == -3
    assert mod_inverse(-3, -7) == -5
    x = Symbol('x')
    assert S(2).invert(x) == S.Half
    raises(TypeError, lambda: mod_inverse(2, x))
    raises(ValueError, lambda: mod_inverse(2, S.Half))
    raises(ValueError, lambda: mod_inverse(2, cos(1)**2 + sin(1)**2))


def test_golden_ratio_rewrite_as_sqrt():
    assert GoldenRatio.rewrite(sqrt) == S.Half + sqrt(5)*S.Half


def test_tribonacci_constant_rewrite_as_sqrt():
    assert TribonacciConstant.rewrite(sqrt) == \
      (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3


def test_comparisons_with_unknown_type():
    class Foo:
        """
        Class that is unaware of Basic, and relies on both classes returning
        the NotImplemented singleton for equivalence to evaluate to False.

        """

    ni, nf, nr = Integer(3), Float(1.0), Rational(1, 3)
    foo = Foo()

    for n in ni, nf, nr, oo, -oo, zoo, nan:
        assert n != foo
        assert foo != n
        assert not n == foo
        assert not foo == n
        raises(TypeError, lambda: n < foo)
        raises(TypeError, lambda: foo > n)
        raises(TypeError, lambda: n > foo)
        raises(TypeError, lambda: foo < n)
        raises(TypeError, lambda: n <= foo)
        raises(TypeError, lambda: foo >= n)
        raises(TypeError, lambda: n >= foo)
        raises(TypeError, lambda: foo <= n)

    class Bar:
        """
        Class that considers itself equal to any instance of Number except
        infinities and nans, and relies on SymPy types returning the
        NotImplemented singleton for symmetric equality relations.

        """
        def __eq__(self, other):
            if other in (oo, -oo, zoo, nan):
                return False
            if isinstance(other, Number):
                return True
            return NotImplemented

        def __ne__(self, other):
            return not self == other

    bar = Bar()

    for n in ni, nf, nr:
        assert n == bar
        assert bar == n
        assert not n != bar
        assert not bar != n

    for n in oo, -oo, zoo, nan:
        assert n != bar
        assert bar != n
        assert not n == bar
        assert not bar == n

    for n in ni, nf, nr, oo, -oo, zoo, nan:
        raises(TypeError, lambda: n < bar)
        raises(TypeError, lambda: bar > n)
        raises(TypeError, lambda: n > bar)
        raises(TypeError, lambda: bar < n)
        raises(TypeError, lambda: n <= bar)
        raises(TypeError, lambda: bar >= n)
        raises(TypeError, lambda: n >= bar)
        raises(TypeError, lambda: bar <= n)


def test_NumberSymbol_comparison():
    from sympy.core.tests.test_relational import rel_check
    rpi = Rational('905502432259640373/288230376151711744')
    fpi = Float(float(pi))
    assert rel_check(rpi, fpi)


def test_Integer_precision():
    # Make sure Integer inputs for keyword args work
    assert Float('1.0', dps=Integer(15))._prec == 53
    assert Float('1.0', precision=Integer(15))._prec == 15
    assert type(Float('1.0', precision=Integer(15))._prec) == int
    assert sympify(srepr(Float('1.0', precision=15))) == Float('1.0', precision=15)


def test_numpy_to_float():
    from sympy.testing.pytest import skip
    from sympy.external import import_module
    np = import_module('numpy')
    if not np:
        skip('numpy not installed. Abort numpy tests.')

    def check_prec_and_relerr(npval, ratval):
        prec = np.finfo(npval).nmant + 1
        x = Float(npval)
        assert x._prec == prec
        y = Float(ratval, precision=prec)
        assert abs((x - y)/y) < 2**(-(prec + 1))

    check_prec_and_relerr(np.float16(2.0/3), Rational(2, 3))
    check_prec_and_relerr(np.float32(2.0/3), Rational(2, 3))
    check_prec_and_relerr(np.float64(2.0/3), Rational(2, 3))
    # extended precision, on some arch/compilers:
    x = np.longdouble(2)/3
    check_prec_and_relerr(x, Rational(2, 3))
    y = Float(x, precision=10)
    assert same_and_same_prec(y, Float(Rational(2, 3), precision=10))

    raises(TypeError, lambda: Float(np.complex64(1+2j)))
    raises(TypeError, lambda: Float(np.complex128(1+2j)))


def test_Integer_ceiling_floor():
    a = Integer(4)

    assert a.floor() == a
    assert a.ceiling() == a


def test_ComplexInfinity():
    assert zoo.floor() is zoo
    assert zoo.ceiling() is zoo
    assert zoo**zoo is S.NaN


def test_Infinity_floor_ceiling_power():
    assert oo.floor() is oo
    assert oo.ceiling() is oo
    assert oo**S.NaN is S.NaN
    assert oo**zoo is S.NaN


def test_One_power():
    assert S.One**12 is S.One
    assert S.NegativeOne**S.NaN is S.NaN


def test_NegativeInfinity():
    assert (-oo).floor() is -oo
    assert (-oo).ceiling() is -oo
    assert (-oo)**11 is -oo
    assert (-oo)**12 is oo


def test_issue_6133():
    raises(TypeError, lambda: (-oo < None))
    raises(TypeError, lambda: (S(-2) < None))
    raises(TypeError, lambda: (oo < None))
    raises(TypeError, lambda: (oo > None))
    raises(TypeError, lambda: (S(2) < None))


def test_abc():
    x = numbers.Float(5)
    assert(isinstance(x, nums.Number))
    assert(isinstance(x, numbers.Number))
    assert(isinstance(x, nums.Real))
    y = numbers.Rational(1, 3)
    assert(isinstance(y, nums.Number))
    assert(y.numerator == 1)
    assert(y.denominator == 3)
    assert(isinstance(y, nums.Rational))
    z = numbers.Integer(3)
    assert(isinstance(z, nums.Number))
    assert(isinstance(z, numbers.Number))
    assert(isinstance(z, nums.Rational))
    assert(isinstance(z, numbers.Rational))
    assert(isinstance(z, nums.Integral))


def test_floordiv():
    assert S(2)//S.Half == 4


def test_negation():
    assert -S.Zero is S.Zero
    assert -Float(0) is not S.Zero and -Float(0) == 0.0


def test_exponentiation_of_0():
    x = Symbol('x')
    assert 0**-x == zoo**x
    assert unchanged(Pow, 0, x)
    x = Symbol('x', zero=True)
    assert 0**-x == S.One
    assert 0**x == S.One


def test_int_valued():
    x = Symbol('x')
    assert int_valued(x) == False
    assert int_valued(S.Half) == False
    assert int_valued(S.One) == True
    assert int_valued(Float(1)) == True
    assert int_valued(Float(1.1)) == False
    assert int_valued(pi) == False


def test_equal_valued():
    x = Symbol('x')

    equal_values = [
        [1, 1.0, S(1), S(1.0), S(1).n(5)],
        [2, 2.0, S(2), S(2.0), S(2).n(5)],
        [-1, -1.0, -S(1), -S(1.0), -S(1).n(5)],
        [0.5, S(0.5), S(1)/2],
        [-0.5, -S(0.5), -S(1)/2],
        [0, 0.0, S(0), S(0.0), S(0).n()],
        [pi], [pi.n()],           # <-- not equal
        [S(1)/10], [0.1, S(0.1)], # <-- not equal
        [S(0.1).n(5)],
        [oo],
        [cos(x/2)], [cos(0.5*x)], # <-- no recursion
    ]

    for m, values_m in enumerate(equal_values):
        for value_i in values_m:

            # All values in same list equal
            for value_j in values_m:
                assert equal_valued(value_i, value_j) is True

            # Not equal to anything in any other list:
            for n, values_n in enumerate(equal_values):
                if n == m:
                    continue
                for value_j in values_n:
                    assert equal_valued(value_i, value_j) is False


def test_all_close():
    x = Symbol('x')
    assert all_close(2, 2) is True
    assert all_close(2, 2.0000) is True
    assert all_close(2, 2.0001) is False
    assert all_close(1/3, 1/3.0001) is False
    assert all_close(1/3, 1/3.0001, 1e-3, 1e-3) is True
    assert all_close(1/3, Rational(1, 3)) is True
    assert all_close(0.1*exp(0.2*x), exp(x/5)/10) is True
    # The expressions should be structurally the same:
    assert all_close(1.4142135623730951, sqrt(2)) is False
    assert all_close(1.4142135623730951, sqrt(2).evalf()) is True
    assert all_close(x + 1e-20, x) is False
    # We should be able to match terms of an Add/Mul in any order
    assert all_close(Add(1, 2, evaluate=False), Add(2, 1, evaluate=False))
