from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath

def mpc_ae(a, b, eps=eps):
    res = True
    res = res and a.real.ae(b.real, eps)
    res = res and a.imag.ae(b.imag, eps)
    return res

#----------------------------------------------------------------------------
# Constants and functions
#

tpi = "3.1415926535897932384626433832795028841971693993751058209749445923078\
1640628620899862803482534211706798"
te = "2.71828182845904523536028747135266249775724709369995957496696762772407\
663035354759457138217852516642743"
tdegree = "0.017453292519943295769236907684886127134428718885417254560971914\
4017100911460344944368224156963450948221"
teuler = "0.5772156649015328606065120900824024310421593359399235988057672348\
84867726777664670936947063291746749516"
tln2 = "0.693147180559945309417232121458176568075500134360255254120680009493\
393621969694715605863326996418687542"
tln10 = "2.30258509299404568401799145468436420760110148862877297603332790096\
757260967735248023599720508959829834"
tcatalan = "0.91596559417721901505460351493238411077414937428167213426649811\
9621763019776254769479356512926115106249"
tkhinchin = "2.6854520010653064453097148354817956938203822939944629530511523\
4555721885953715200280114117493184769800"
tglaisher = "1.2824271291006226368753425688697917277676889273250011920637400\
2174040630885882646112973649195820237439420646"
tapery = "1.2020569031595942853997381615114499907649862923404988817922715553\
4183820578631309018645587360933525815"
tphi = "1.618033988749894848204586834365638117720309179805762862135448622705\
26046281890244970720720418939113748475"
tmertens = "0.26149721284764278375542683860869585905156664826119920619206421\
3924924510897368209714142631434246651052"
ttwinprime = "0.660161815846869573927812110014555778432623360284733413319448\
423335405642304495277143760031413839867912"

def test_constants():
    for prec in [3, 7, 10, 15, 20, 37, 80, 100, 29]:
        mp.dps = prec
        assert pi == mpf(tpi)
        assert e == mpf(te)
        assert degree == mpf(tdegree)
        assert euler == mpf(teuler)
        assert ln2 == mpf(tln2)
        assert ln10 == mpf(tln10)
        assert catalan == mpf(tcatalan)
        assert khinchin == mpf(tkhinchin)
        assert glaisher == mpf(tglaisher)
        assert phi == mpf(tphi)
        if prec < 50:
            assert mertens == mpf(tmertens)
            assert twinprime == mpf(ttwinprime)
    mp.dps = 15
    assert pi >= -1
    assert pi > 2
    assert pi > 3
    assert pi < 4

def test_exact_sqrts():
    for i in range(20000):
        assert sqrt(mpf(i*i)) == i
    random.seed(1)
    for prec in [100, 300, 1000, 10000]:
        mp.dps = prec
        for i in range(20):
            A = random.randint(10**(prec//2-2), 10**(prec//2-1))
            assert sqrt(mpf(A*A)) == A
    mp.dps = 15
    for i in range(100):
        for a in [1, 8, 25, 112307]:
            assert sqrt(mpf((a*a, 2*i))) == mpf((a, i))
            assert sqrt(mpf((a*a, -2*i))) == mpf((a, -i))

def test_sqrt_rounding():
    for i in [2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]:
        i = from_int(i)
        for dps in [7, 15, 83, 106, 2000]:
            mp.dps = dps
            a = mpf_pow_int(mpf_sqrt(i, mp.prec, round_down), 2, mp.prec, round_down)
            b = mpf_pow_int(mpf_sqrt(i, mp.prec, round_up), 2, mp.prec, round_up)
            assert mpf_lt(a, i)
            assert mpf_gt(b, i)
    random.seed(1234)
    prec = 100
    for rnd in [round_down, round_nearest, round_ceiling]:
        for i in range(100):
            a = mpf_rand(prec)
            b = mpf_mul(a, a)
            assert mpf_sqrt(b, prec, rnd) == a
    # Test some extreme cases
    mp.dps = 100
    a = mpf(9) + 1e-90
    b = mpf(9) - 1e-90
    mp.dps = 15
    assert sqrt(a, rounding='d') == 3
    assert sqrt(a, rounding='n') == 3
    assert sqrt(a, rounding='u') > 3
    assert sqrt(b, rounding='d') < 3
    assert sqrt(b, rounding='n') == 3
    assert sqrt(b, rounding='u') == 3
    # A worst case, from the MPFR test suite
    assert sqrt(mpf('7.0503726185518891')) == mpf('2.655253776675949')

def test_float_sqrt():
    mp.dps = 15
    # These should round identically
    for x in [0, 1e-7, 0.1, 0.5, 1, 2, 3, 4, 5, 0.333, 76.19]:
        assert sqrt(mpf(x)) == float(x)**0.5
    assert sqrt(-1) == 1j
    assert sqrt(-2).ae(cmath.sqrt(-2))
    assert sqrt(-3).ae(cmath.sqrt(-3))
    assert sqrt(-100).ae(cmath.sqrt(-100))
    assert sqrt(1j).ae(cmath.sqrt(1j))
    assert sqrt(-1j).ae(cmath.sqrt(-1j))
    assert sqrt(math.pi + math.e*1j).ae(cmath.sqrt(math.pi + math.e*1j))
    assert sqrt(math.pi - math.e*1j).ae(cmath.sqrt(math.pi - math.e*1j))

def test_hypot():
    assert hypot(0, 0) == 0
    assert hypot(0, 0.33) == mpf(0.33)
    assert hypot(0.33, 0) == mpf(0.33)
    assert hypot(-0.33, 0) == mpf(0.33)
    assert hypot(3, 4) == mpf(5)

def test_exact_cbrt():
    for i in range(0, 20000, 200):
        assert cbrt(mpf(i*i*i)) == i
    random.seed(1)
    for prec in [100, 300, 1000, 10000]:
        mp.dps = prec
        A = random.randint(10**(prec//2-2), 10**(prec//2-1))
        assert cbrt(mpf(A*A*A)) == A
    mp.dps = 15

def test_exp():
    assert exp(0) == 1
    assert exp(10000).ae(mpf('8.8068182256629215873e4342'))
    assert exp(-10000).ae(mpf('1.1354838653147360985e-4343'))
    a = exp(mpf((1, 8198646019315405, -53, 53)))
    assert(a.bc == bitcount(a.man))
    mp.prec = 67
    a = exp(mpf((1, 1781864658064754565, -60, 61)))
    assert(a.bc == bitcount(a.man))
    mp.prec = 53
    assert exp(ln2 * 10).ae(1024)
    assert exp(2+2j).ae(cmath.exp(2+2j))

def test_issue_73():
    mp.dps = 512
    a = exp(-1)
    b = exp(1)
    mp.dps = 15
    assert (+a).ae(0.36787944117144233)
    assert (+b).ae(2.7182818284590451)

def test_log():
    mp.dps = 15
    assert log(1) == 0
    for x in [0.5, 1.5, 2.0, 3.0, 100, 10**50, 1e-50]:
        assert log(x).ae(math.log(x))
        assert log(x, x) == 1
    assert log(1024, 2) == 10
    assert log(10**1234, 10) == 1234
    assert log(2+2j).ae(cmath.log(2+2j))
    # Accuracy near 1
    assert (log(0.6+0.8j).real*10**17).ae(2.2204460492503131)
    assert (log(0.6-0.8j).real*10**17).ae(2.2204460492503131)
    assert (log(0.8-0.6j).real*10**17).ae(2.2204460492503131)
    assert (log(1+1e-8j).real*10**16).ae(0.5)
    assert (log(1-1e-8j).real*10**16).ae(0.5)
    assert (log(-1+1e-8j).real*10**16).ae(0.5)
    assert (log(-1-1e-8j).real*10**16).ae(0.5)
    assert (log(1j+1e-8).real*10**16).ae(0.5)
    assert (log(1j-1e-8).real*10**16).ae(0.5)
    assert (log(-1j+1e-8).real*10**16).ae(0.5)
    assert (log(-1j-1e-8).real*10**16).ae(0.5)
    assert (log(1+1e-40j).real*10**80).ae(0.5)
    assert (log(1j+1e-40).real*10**80).ae(0.5)
    # Huge
    assert log(ldexp(1.234,10**20)).ae(log(2)*1e20)
    assert log(ldexp(1.234,10**200)).ae(log(2)*1e200)
    # Some special values
    assert log(mpc(0,0)) == mpc(-inf,0)
    assert isnan(log(mpc(nan,0)).real)
    assert isnan(log(mpc(nan,0)).imag)
    assert isnan(log(mpc(0,nan)).real)
    assert isnan(log(mpc(0,nan)).imag)
    assert isnan(log(mpc(nan,1)).real)
    assert isnan(log(mpc(nan,1)).imag)
    assert isnan(log(mpc(1,nan)).real)
    assert isnan(log(mpc(1,nan)).imag)

def test_trig_hyperb_basic():
    for x in (list(range(100)) + list(range(-100,0))):
        t = x / 4.1
        assert cos(mpf(t)).ae(math.cos(t))
        assert sin(mpf(t)).ae(math.sin(t))
        assert tan(mpf(t)).ae(math.tan(t))
        assert cosh(mpf(t)).ae(math.cosh(t))
        assert sinh(mpf(t)).ae(math.sinh(t))
        assert tanh(mpf(t)).ae(math.tanh(t))
    assert sin(1+1j).ae(cmath.sin(1+1j))
    assert sin(-4-3.6j).ae(cmath.sin(-4-3.6j))
    assert cos(1+1j).ae(cmath.cos(1+1j))
    assert cos(-4-3.6j).ae(cmath.cos(-4-3.6j))

def test_degrees():
    assert cos(0*degree) == 1
    assert cos(90*degree).ae(0)
    assert cos(180*degree).ae(-1)
    assert cos(270*degree).ae(0)
    assert cos(360*degree).ae(1)
    assert sin(0*degree) == 0
    assert sin(90*degree).ae(1)
    assert sin(180*degree).ae(0)
    assert sin(270*degree).ae(-1)
    assert sin(360*degree).ae(0)

def random_complexes(N):
    random.seed(1)
    a = []
    for i in range(N):
        x1 = random.uniform(-10, 10)
        y1 = random.uniform(-10, 10)
        x2 = random.uniform(-10, 10)
        y2 = random.uniform(-10, 10)
        z1 = complex(x1, y1)
        z2 = complex(x2, y2)
        a.append((z1, z2))
    return a

def test_complex_powers():
    for dps in [15, 30, 100]:
        # Check accuracy for complex square root
        mp.dps = dps
        a = mpc(1j)**0.5
        assert a.real == a.imag == mpf(2)**0.5 / 2
    mp.dps = 15
    random.seed(1)
    for (z1, z2) in random_complexes(100):
        assert (mpc(z1)**mpc(z2)).ae(z1**z2, 1e-12)
    assert (e**(-pi*1j)).ae(-1)
    mp.dps = 50
    assert (e**(-pi*1j)).ae(-1)
    mp.dps = 15

def test_complex_sqrt_accuracy():
    def test_mpc_sqrt(lst):
        for a, b in lst:
            z = mpc(a + j*b)
            assert mpc_ae(sqrt(z*z), z)
            z = mpc(-a + j*b)
            assert mpc_ae(sqrt(z*z), -z)
            z = mpc(a - j*b)
            assert mpc_ae(sqrt(z*z), z)
            z = mpc(-a - j*b)
            assert mpc_ae(sqrt(z*z), -z)
    random.seed(2)
    N = 10
    mp.dps = 30
    dps = mp.dps
    test_mpc_sqrt([(random.uniform(0, 10),random.uniform(0, 10)) for i in range(N)])
    test_mpc_sqrt([(i + 0.1, (i + 0.2)*10**i) for i in range(N)])
    mp.dps = 15

def test_atan():
    mp.dps = 15
    assert atan(-2.3).ae(math.atan(-2.3))
    assert atan(1e-50) == 1e-50
    assert atan(1e50).ae(pi/2)
    assert atan(-1e-50) == -1e-50
    assert atan(-1e50).ae(-pi/2)
    assert atan(10**1000).ae(pi/2)
    for dps in [25, 70, 100, 300, 1000]:
        mp.dps = dps
        assert (4*atan(1)).ae(pi)
    mp.dps = 15
    pi2 = pi/2
    assert atan(mpc(inf,-1)).ae(pi2)
    assert atan(mpc(inf,0)).ae(pi2)
    assert atan(mpc(inf,1)).ae(pi2)
    assert atan(mpc(1,inf)).ae(pi2)
    assert atan(mpc(0,inf)).ae(pi2)
    assert atan(mpc(-1,inf)).ae(-pi2)
    assert atan(mpc(-inf,1)).ae(-pi2)
    assert atan(mpc(-inf,0)).ae(-pi2)
    assert atan(mpc(-inf,-1)).ae(-pi2)
    assert atan(mpc(-1,-inf)).ae(-pi2)
    assert atan(mpc(0,-inf)).ae(-pi2)
    assert atan(mpc(1,-inf)).ae(pi2)

def test_atan2():
    mp.dps = 15
    assert atan2(1,1).ae(pi/4)
    assert atan2(1,-1).ae(3*pi/4)
    assert atan2(-1,-1).ae(-3*pi/4)
    assert atan2(-1,1).ae(-pi/4)
    assert atan2(-1,0).ae(-pi/2)
    assert atan2(1,0).ae(pi/2)
    assert atan2(0,0) == 0
    assert atan2(inf,0).ae(pi/2)
    assert atan2(-inf,0).ae(-pi/2)
    assert isnan(atan2(inf,inf))
    assert isnan(atan2(-inf,inf))
    assert isnan(atan2(inf,-inf))
    assert isnan(atan2(3,nan))
    assert isnan(atan2(nan,3))
    assert isnan(atan2(0,nan))
    assert isnan(atan2(nan,0))
    assert atan2(0,inf) == 0
    assert atan2(0,-inf).ae(pi)
    assert atan2(10,inf) == 0
    assert atan2(-10,inf) == 0
    assert atan2(-10,-inf).ae(-pi)
    assert atan2(10,-inf).ae(pi)
    assert atan2(inf,10).ae(pi/2)
    assert atan2(inf,-10).ae(pi/2)
    assert atan2(-inf,10).ae(-pi/2)
    assert atan2(-inf,-10).ae(-pi/2)

def test_areal_inverses():
    assert asin(mpf(0)) == 0
    assert asinh(mpf(0)) == 0
    assert acosh(mpf(1)) == 0
    assert isinstance(asin(mpf(0.5)), mpf)
    assert isinstance(asin(mpf(2.0)), mpc)
    assert isinstance(acos(mpf(0.5)), mpf)
    assert isinstance(acos(mpf(2.0)), mpc)
    assert isinstance(atanh(mpf(0.1)), mpf)
    assert isinstance(atanh(mpf(1.1)), mpc)

    random.seed(1)
    for i in range(50):
        x = random.uniform(0, 1)
        assert asin(mpf(x)).ae(math.asin(x))
        assert acos(mpf(x)).ae(math.acos(x))

        x = random.uniform(-10, 10)
        assert asinh(mpf(x)).ae(cmath.asinh(x).real)
        assert isinstance(asinh(mpf(x)), mpf)
        x = random.uniform(1, 10)
        assert acosh(mpf(x)).ae(cmath.acosh(x).real)
        assert isinstance(acosh(mpf(x)), mpf)
        x = random.uniform(-10, 0.999)
        assert isinstance(acosh(mpf(x)), mpc)

        x = random.uniform(-1, 1)
        assert atanh(mpf(x)).ae(cmath.atanh(x).real)
        assert isinstance(atanh(mpf(x)), mpf)

    dps = mp.dps
    mp.dps = 300
    assert isinstance(asin(0.5), mpf)
    mp.dps = 1000
    assert asin(1).ae(pi/2)
    assert asin(-1).ae(-pi/2)
    mp.dps = dps

def test_invhyperb_inaccuracy():
    mp.dps = 15
    assert (asinh(1e-5)*10**5).ae(0.99999999998333333)
    assert (asinh(1e-10)*10**10).ae(1)
    assert (asinh(1e-50)*10**50).ae(1)
    assert (asinh(-1e-5)*10**5).ae(-0.99999999998333333)
    assert (asinh(-1e-10)*10**10).ae(-1)
    assert (asinh(-1e-50)*10**50).ae(-1)
    assert asinh(10**20).ae(46.744849040440862)
    assert asinh(-10**20).ae(-46.744849040440862)
    assert (tanh(1e-10)*10**10).ae(1)
    assert (tanh(-1e-10)*10**10).ae(-1)
    assert (atanh(1e-10)*10**10).ae(1)
    assert (atanh(-1e-10)*10**10).ae(-1)

def test_complex_functions():
    for x in (list(range(10)) + list(range(-10,0))):
        for y in (list(range(10)) + list(range(-10,0))):
            z = complex(x, y)/4.3 + 0.01j
            assert exp(mpc(z)).ae(cmath.exp(z))
            assert log(mpc(z)).ae(cmath.log(z))
            assert cos(mpc(z)).ae(cmath.cos(z))
            assert sin(mpc(z)).ae(cmath.sin(z))
            assert tan(mpc(z)).ae(cmath.tan(z))
            assert sinh(mpc(z)).ae(cmath.sinh(z))
            assert cosh(mpc(z)).ae(cmath.cosh(z))
            assert tanh(mpc(z)).ae(cmath.tanh(z))

def test_complex_inverse_functions():
    mp.dps = 15
    iv.dps = 15
    for (z1, z2) in random_complexes(30):
        # apparently cmath uses a different branch, so we
        # can't use it for comparison
        assert sinh(asinh(z1)).ae(z1)
        #
        assert acosh(z1).ae(cmath.acosh(z1))
        assert atanh(z1).ae(cmath.atanh(z1))
        assert atan(z1).ae(cmath.atan(z1))
        # the reason we set a big eps here is that the cmath
        # functions are inaccurate
        assert asin(z1).ae(cmath.asin(z1), rel_eps=1e-12)
        assert acos(z1).ae(cmath.acos(z1), rel_eps=1e-12)
        one = mpf(1)
    for i in range(-9, 10, 3):
        for k in range(-9, 10, 3):
            a = 0.9*j*10**k + 0.8*one*10**i
            b = cos(acos(a))
            assert b.ae(a)
            b = sin(asin(a))
            assert b.ae(a)
    one = mpf(1)
    err = 2*10**-15
    for i in range(-9, 9, 3):
        for k in range(-9, 9, 3):
            a = -0.9*10**k + j*0.8*one*10**i
            b = cosh(acosh(a))
            assert b.ae(a, err)
            b = sinh(asinh(a))
            assert b.ae(a, err)

def test_reciprocal_functions():
    assert sec(3).ae(-1.01010866590799375)
    assert csc(3).ae(7.08616739573718592)
    assert cot(3).ae(-7.01525255143453347)
    assert sech(3).ae(0.0993279274194332078)
    assert csch(3).ae(0.0998215696688227329)
    assert coth(3).ae(1.00496982331368917)
    assert asec(3).ae(1.23095941734077468)
    assert acsc(3).ae(0.339836909454121937)
    assert acot(3).ae(0.321750554396642193)
    assert asech(0.5).ae(1.31695789692481671)
    assert acsch(3).ae(0.327450150237258443)
    assert acoth(3).ae(0.346573590279972655)
    assert acot(0).ae(1.5707963267948966192)
    assert acoth(0).ae(1.5707963267948966192j)

def test_ldexp():
    mp.dps = 15
    assert ldexp(mpf(2.5), 0) == 2.5
    assert ldexp(mpf(2.5), -1) == 1.25
    assert ldexp(mpf(2.5), 2) == 10
    assert ldexp(mpf('inf'), 3) == mpf('inf')

def test_frexp():
    mp.dps = 15
    assert frexp(0) == (0.0, 0)
    assert frexp(9) == (0.5625, 4)
    assert frexp(1) == (0.5, 1)
    assert frexp(0.2) == (0.8, -2)
    assert frexp(1000) == (0.9765625, 10)

def test_aliases():
    assert ln(7) == log(7)
    assert log10(3.75) == log(3.75,10)
    assert degrees(5.6) == 5.6 / degree
    assert radians(5.6) == 5.6 * degree
    assert power(-1,0.5) == j
    assert fmod(25,7) == 4.0 and isinstance(fmod(25,7), mpf)

def test_arg_sign():
    assert arg(3) == 0
    assert arg(-3).ae(pi)
    assert arg(j).ae(pi/2)
    assert arg(-j).ae(-pi/2)
    assert arg(0) == 0
    assert isnan(atan2(3,nan))
    assert isnan(atan2(nan,3))
    assert isnan(atan2(0,nan))
    assert isnan(atan2(nan,0))
    assert isnan(atan2(nan,nan))
    assert arg(inf) == 0
    assert arg(-inf).ae(pi)
    assert isnan(arg(nan))
    #assert arg(inf*j).ae(pi/2)
    assert sign(0) == 0
    assert sign(3) == 1
    assert sign(-3) == -1
    assert sign(inf) == 1
    assert sign(-inf) == -1
    assert isnan(sign(nan))
    assert sign(j) == j
    assert sign(-3*j) == -j
    assert sign(1+j).ae((1+j)/sqrt(2))

def test_misc_bugs():
    # test that this doesn't raise an exception
    mp.dps = 1000
    log(1302)
    mp.dps = 15

def test_arange():
    assert arange(10) == [mpf('0.0'), mpf('1.0'), mpf('2.0'), mpf('3.0'),
                          mpf('4.0'), mpf('5.0'), mpf('6.0'), mpf('7.0'),
                          mpf('8.0'), mpf('9.0')]
    assert arange(-5, 5) == [mpf('-5.0'), mpf('-4.0'), mpf('-3.0'),
                             mpf('-2.0'), mpf('-1.0'), mpf('0.0'),
                             mpf('1.0'), mpf('2.0'), mpf('3.0'), mpf('4.0')]
    assert arange(0, 1, 0.1) == [mpf('0.0'), mpf('0.10000000000000001'),
                                 mpf('0.20000000000000001'),
                                 mpf('0.30000000000000004'),
                                 mpf('0.40000000000000002'),
                                 mpf('0.5'), mpf('0.60000000000000009'),
                                 mpf('0.70000000000000007'),
                                 mpf('0.80000000000000004'),
                                 mpf('0.90000000000000002')]
    assert arange(17, -9, -3) == [mpf('17.0'), mpf('14.0'), mpf('11.0'),
                                  mpf('8.0'), mpf('5.0'), mpf('2.0'),
                                  mpf('-1.0'), mpf('-4.0'), mpf('-7.0')]
    assert arange(0.2, 0.1, -0.1) == [mpf('0.20000000000000001')]
    assert arange(0) == []
    assert arange(1000, -1) == []
    assert arange(-1.23, 3.21, -0.0000001) == []

def test_linspace():
    assert linspace(2, 9, 7) == [mpf('2.0'), mpf('3.166666666666667'),
        mpf('4.3333333333333339'), mpf('5.5'), mpf('6.666666666666667'),
        mpf('7.8333333333333339'), mpf('9.0')]
    assert linspace(2, 9, 7, endpoint=0) == [mpf('2.0'), mpf('3.0'), mpf('4.0'),
        mpf('5.0'), mpf('6.0'), mpf('7.0'), mpf('8.0')]
    assert linspace(2, 7, 1) == [mpf(2)]

def test_float_cbrt():
    mp.dps = 30
    for a in arange(0,10,0.1):
        assert cbrt(a*a*a).ae(a, eps)
    assert cbrt(-1).ae(0.5 + j*sqrt(3)/2)
    one_third = mpf(1)/3
    for a in arange(0,10,2.7) + [0.1 + 10**5]:
        a = mpc(a + 1.1j)
        r1 = cbrt(a)
        mp.dps += 10
        r2 = pow(a, one_third)
        mp.dps -= 10
        assert r1.ae(r2, eps)
    mp.dps = 100
    for n in range(100, 301, 100):
        w = 10**n + j*10**-3
        z = w*w*w
        r = cbrt(z)
        assert mpc_ae(r, w, eps)
    mp.dps = 15

def test_root():
    mp.dps = 30
    random.seed(1)
    a = random.randint(0, 10000)
    p = a*a*a
    r = nthroot(mpf(p), 3)
    assert r == a
    for n in range(4, 10):
        p = p*a
        assert nthroot(mpf(p), n) == a
    mp.dps = 40
    for n in range(10, 5000, 100):
        for a in [random.random()*10000, random.random()*10**100]:
            r = nthroot(a, n)
            r1 = pow(a, mpf(1)/n)
            assert r.ae(r1)
            r = nthroot(a, -n)
            r1 = pow(a, -mpf(1)/n)
            assert r.ae(r1)
    # XXX: this is broken right now
    # tests for nthroot rounding
    for rnd in ['nearest', 'up', 'down']:
        mp.rounding = rnd
        for n in [-5, -3, 3, 5]:
            prec = 50
            for i in range(10):
                mp.prec = prec
                a = rand()
                mp.prec = 2*prec
                b = a**n
                mp.prec = prec
                r = nthroot(b, n)
                assert r == a
    mp.dps = 30
    for n in range(3, 21):
        a = (random.random() + j*random.random())
        assert nthroot(a, n).ae(pow(a, mpf(1)/n))
        assert mpc_ae(nthroot(a, n), pow(a, mpf(1)/n))
        a = (random.random()*10**100 + j*random.random())
        r = nthroot(a, n)
        mp.dps += 4
        r1 = pow(a, mpf(1)/n)
        mp.dps -= 4
        assert r.ae(r1)
        assert mpc_ae(r, r1, eps)
        r = nthroot(a, -n)
        mp.dps += 4
        r1 = pow(a, -mpf(1)/n)
        mp.dps -= 4
        assert r.ae(r1)
        assert mpc_ae(r, r1, eps)
    mp.dps = 15
    assert nthroot(4, 1) == 4
    assert nthroot(4, 0) == 1
    assert nthroot(4, -1) == 0.25
    assert nthroot(inf, 1) == inf
    assert nthroot(inf, 2) == inf
    assert nthroot(inf, 3) == inf
    assert nthroot(inf, -1) == 0
    assert nthroot(inf, -2) == 0
    assert nthroot(inf, -3) == 0
    assert nthroot(j, 1) == j
    assert nthroot(j, 0) == 1
    assert nthroot(j, -1) == -j
    assert isnan(nthroot(nan, 1))
    assert isnan(nthroot(nan, 0))
    assert isnan(nthroot(nan, -1))
    assert isnan(nthroot(inf, 0))
    assert root(2,3) == nthroot(2,3)
    assert root(16,4,0) == 2
    assert root(16,4,1) == 2j
    assert root(16,4,2) == -2
    assert root(16,4,3) == -2j
    assert root(16,4,4) == 2
    assert root(-125,3,1) == -5

def test_issue_136():
    for dps in [20, 80]:
        mp.dps = dps
        r = nthroot(mpf('-1e-20'), 4)
        assert r.ae(mpf(10)**(-5) * (1 + j) * mpf(2)**(-0.5))
    mp.dps = 80
    assert nthroot('-1e-3', 4).ae(mpf(10)**(-3./4) * (1 + j)/sqrt(2))
    assert nthroot('-1e-6', 4).ae((1 + j)/(10 * sqrt(20)))
    # Check that this doesn't take eternity to compute
    mp.dps = 20
    assert nthroot('-1e100000000', 4).ae((1+j)*mpf('1e25000000')/sqrt(2))
    mp.dps = 15

def test_mpcfun_real_imag():
    mp.dps = 15
    x = mpf(0.3)
    y = mpf(0.4)
    assert exp(mpc(x,0)) == exp(x)
    assert exp(mpc(0,y)) == mpc(cos(y),sin(y))
    assert cos(mpc(x,0)) == cos(x)
    assert sin(mpc(x,0)) == sin(x)
    assert cos(mpc(0,y)) == cosh(y)
    assert sin(mpc(0,y)) == mpc(0,sinh(y))
    assert cospi(mpc(x,0)) == cospi(x)
    assert sinpi(mpc(x,0)) == sinpi(x)
    assert cospi(mpc(0,y)).ae(cosh(pi*y))
    assert sinpi(mpc(0,y)).ae(mpc(0,sinh(pi*y)))
    c, s = cospi_sinpi(mpc(x,0))
    assert c == cospi(x)
    assert s == sinpi(x)
    c, s = cospi_sinpi(mpc(0,y))
    assert c.ae(cosh(pi*y))
    assert s.ae(mpc(0,sinh(pi*y)))
    c, s = cos_sin(mpc(x,0))
    assert c == cos(x)
    assert s == sin(x)
    c, s = cos_sin(mpc(0,y))
    assert c == cosh(y)
    assert s == mpc(0,sinh(y))

def test_perturbation_rounding():
    mp.dps = 100
    a = pi/10**50
    b = -pi/10**50
    c = 1 + a
    d = 1 + b
    mp.dps = 15
    assert exp(a) == 1
    assert exp(a, rounding='c') > 1
    assert exp(b, rounding='c') == 1
    assert exp(a, rounding='f') == 1
    assert exp(b, rounding='f') < 1
    assert cos(a) == 1
    assert cos(a, rounding='c') == 1
    assert cos(b, rounding='c') == 1
    assert cos(a, rounding='f') < 1
    assert cos(b, rounding='f') < 1
    for f in [sin, atan, asinh, tanh]:
        assert f(a) == +a
        assert f(a, rounding='c') > a
        assert f(a, rounding='f') < a
        assert f(b) == +b
        assert f(b, rounding='c') > b
        assert f(b, rounding='f') < b
    for f in [asin, tan, sinh, atanh]:
        assert f(a) == +a
        assert f(b) == +b
        assert f(a, rounding='c') > a
        assert f(b, rounding='c') > b
        assert f(a, rounding='f') < a
        assert f(b, rounding='f') < b
    assert ln(c) == +a
    assert ln(d) == +b
    assert ln(c, rounding='c') > a
    assert ln(c, rounding='f') < a
    assert ln(d, rounding='c') > b
    assert ln(d, rounding='f') < b
    assert cosh(a) == 1
    assert cosh(b) == 1
    assert cosh(a, rounding='c') > 1
    assert cosh(b, rounding='c') > 1
    assert cosh(a, rounding='f') == 1
    assert cosh(b, rounding='f') == 1

def test_integer_parts():
    assert floor(3.2) == 3
    assert ceil(3.2) == 4
    assert floor(3.2+5j) == 3+5j
    assert ceil(3.2+5j) == 4+5j

def test_complex_parts():
    assert fabs('3') == 3
    assert fabs(3+4j) == 5
    assert re(3) == 3
    assert re(1+4j) == 1
    assert im(3) == 0
    assert im(1+4j) == 4
    assert conj(3) == 3
    assert conj(3+4j) == 3-4j
    assert mpf(3).conjugate() == 3

def test_cospi_sinpi():
    assert sinpi(0) == 0
    assert sinpi(0.5) == 1
    assert sinpi(1) == 0
    assert sinpi(1.5) == -1
    assert sinpi(2) == 0
    assert sinpi(2.5) == 1
    assert sinpi(-0.5) == -1
    assert cospi(0) == 1
    assert cospi(0.5) == 0
    assert cospi(1) == -1
    assert cospi(1.5) == 0
    assert cospi(2) == 1
    assert cospi(2.5) == 0
    assert cospi(-0.5) == 0
    assert cospi(100000000000.25).ae(sqrt(2)/2)
    a = cospi(2+3j)
    assert a.real.ae(cos((2+3j)*pi).real)
    assert a.imag == 0
    b = sinpi(2+3j)
    assert b.imag.ae(sin((2+3j)*pi).imag)
    assert b.real == 0
    mp.dps = 35
    x1 = mpf(10000) - mpf('1e-15')
    x2 = mpf(10000) + mpf('1e-15')
    x3 = mpf(10000.5) - mpf('1e-15')
    x4 = mpf(10000.5) + mpf('1e-15')
    x5 = mpf(10001) - mpf('1e-15')
    x6 = mpf(10001) + mpf('1e-15')
    x7 = mpf(10001.5) - mpf('1e-15')
    x8 = mpf(10001.5) + mpf('1e-15')
    mp.dps = 15
    M = 10**15
    assert (sinpi(x1)*M).ae(-pi)
    assert (sinpi(x2)*M).ae(pi)
    assert (cospi(x3)*M).ae(pi)
    assert (cospi(x4)*M).ae(-pi)
    assert (sinpi(x5)*M).ae(pi)
    assert (sinpi(x6)*M).ae(-pi)
    assert (cospi(x7)*M).ae(-pi)
    assert (cospi(x8)*M).ae(pi)
    assert 0.999 < cospi(x1, rounding='d') < 1
    assert 0.999 < cospi(x2, rounding='d') < 1
    assert 0.999 < sinpi(x3, rounding='d') < 1
    assert 0.999 < sinpi(x4, rounding='d') < 1
    assert -1 < cospi(x5, rounding='d') < -0.999
    assert -1 < cospi(x6, rounding='d') < -0.999
    assert -1 < sinpi(x7, rounding='d') < -0.999
    assert -1 < sinpi(x8, rounding='d') < -0.999
    assert (sinpi(1e-15)*M).ae(pi)
    assert (sinpi(-1e-15)*M).ae(-pi)
    assert cospi(1e-15) == 1
    assert cospi(1e-15, rounding='d') < 1

def test_expj():
    assert expj(0) == 1
    assert expj(1).ae(exp(j))
    assert expj(j).ae(exp(-1))
    assert expj(1+j).ae(exp(j*(1+j)))
    assert expjpi(0) == 1
    assert expjpi(1).ae(exp(j*pi))
    assert expjpi(j).ae(exp(-pi))
    assert expjpi(1+j).ae(exp(j*pi*(1+j)))
    assert expjpi(-10**15 * j).ae('2.22579818340535731e+1364376353841841')

def test_sinc():
    assert sinc(0) == sincpi(0) == 1
    assert sinc(inf) == sincpi(inf) == 0
    assert sinc(-inf) == sincpi(-inf) == 0
    assert sinc(2).ae(0.45464871341284084770)
    assert sinc(2+3j).ae(0.4463290318402435457-2.7539470277436474940j)
    assert sincpi(2) == 0
    assert sincpi(1.5).ae(-0.212206590789193781)

def test_fibonacci():
    mp.dps = 15
    assert [fibonacci(n) for n in range(-5, 10)] == \
        [5, -3, 2, -1, 1, 0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert fib(2.5).ae(1.4893065462657091)
    assert fib(3+4j).ae(-5248.51130728372 - 14195.962288353j)
    assert fib(1000).ae(4.3466557686937455e+208)
    assert str(fib(10**100)) == '6.24499112864607e+2089876402499787337692720892375554168224592399182109535392875613974104853496745963277658556235103534'
    mp.dps = 2100
    a = fib(10000)
    assert a % 10**10 == 9947366875
    mp.dps = 15
    assert fibonacci(inf) == inf
    assert fib(3+0j) == 2

def test_call_with_dps():
    mp.dps = 15
    assert abs(exp(1, dps=30)-e(dps=35)) < 1e-29

def test_tanh():
    mp.dps = 15
    assert tanh(0) == 0
    assert tanh(inf) == 1
    assert tanh(-inf) == -1
    assert isnan(tanh(nan))
    assert tanh(mpc('inf', '0')) == 1

def test_atanh():
    mp.dps = 15
    assert atanh(0) == 0
    assert atanh(0.5).ae(0.54930614433405484570)
    assert atanh(-0.5).ae(-0.54930614433405484570)
    assert atanh(1) == inf
    assert atanh(-1) == -inf
    assert isnan(atanh(nan))
    assert isinstance(atanh(1), mpf)
    assert isinstance(atanh(-1), mpf)
    # Limits at infinity
    jpi2 = j*pi/2
    assert atanh(inf).ae(-jpi2)
    assert atanh(-inf).ae(jpi2)
    assert atanh(mpc(inf,-1)).ae(-jpi2)
    assert atanh(mpc(inf,0)).ae(-jpi2)
    assert atanh(mpc(inf,1)).ae(jpi2)
    assert atanh(mpc(1,inf)).ae(jpi2)
    assert atanh(mpc(0,inf)).ae(jpi2)
    assert atanh(mpc(-1,inf)).ae(jpi2)
    assert atanh(mpc(-inf,1)).ae(jpi2)
    assert atanh(mpc(-inf,0)).ae(jpi2)
    assert atanh(mpc(-inf,-1)).ae(-jpi2)
    assert atanh(mpc(-1,-inf)).ae(-jpi2)
    assert atanh(mpc(0,-inf)).ae(-jpi2)
    assert atanh(mpc(1,-inf)).ae(-jpi2)

def test_expm1():
    mp.dps = 15
    assert expm1(0) == 0
    assert expm1(3).ae(exp(3)-1)
    assert expm1(inf) == inf
    assert expm1(1e-50).ae(1e-50)
    assert (expm1(1e-10)*1e10).ae(1.00000000005)

def test_log1p():
    mp.dps = 15
    assert log1p(0) == 0
    assert log1p(3).ae(log(1+3))
    assert log1p(inf) == inf
    assert log1p(1e-50).ae(1e-50)
    assert (log1p(1e-10)*1e10).ae(0.99999999995)

def test_powm1():
    mp.dps = 15
    assert powm1(2,3) == 7
    assert powm1(-1,2) == 0
    assert powm1(-1,0) == 0
    assert powm1(-2,0) == 0
    assert powm1(3+4j,0) == 0
    assert powm1(0,1) == -1
    assert powm1(0,0) == 0
    assert powm1(1,0) == 0
    assert powm1(1,2) == 0
    assert powm1(1,3+4j) == 0
    assert powm1(1,5) == 0
    assert powm1(j,4) == 0
    assert powm1(-j,4) == 0
    assert (powm1(2,1e-100)*1e100).ae(ln2)
    assert powm1(2,'1e-100000000000') != 0
    assert (powm1(fadd(1,1e-100,exact=True), 5)*1e100).ae(5)

def test_unitroots():
    assert unitroots(1) == [1]
    assert unitroots(2) == [1, -1]
    a, b, c = unitroots(3)
    assert a == 1
    assert b.ae(-0.5 + 0.86602540378443864676j)
    assert c.ae(-0.5 - 0.86602540378443864676j)
    assert unitroots(1, primitive=True) == [1]
    assert unitroots(2, primitive=True) == [-1]
    assert unitroots(3, primitive=True) == unitroots(3)[1:]
    assert unitroots(4, primitive=True) == [j, -j]
    assert len(unitroots(17, primitive=True)) == 16
    assert len(unitroots(16, primitive=True)) == 8

def test_cyclotomic():
    mp.dps = 15
    assert [cyclotomic(n,1) for n in range(31)] == [1,0,2,3,2,5,1,7,2,3,1,11,1,13,1,1,2,17,1,19,1,1,1,23,1,5,1,3,1,29,1]
    assert [cyclotomic(n,-1) for n in range(31)] == [1,-2,0,1,2,1,3,1,2,1,5,1,1,1,7,1,2,1,3,1,1,1,11,1,1,1,13,1,1,1,1]
    assert [cyclotomic(n,j) for n in range(21)] == [1,-1+j,1+j,j,0,1,-j,j,2,-j,1,j,3,1,-j,1,2,1,j,j,5]
    assert [cyclotomic(n,-j) for n in range(21)] == [1,-1-j,1-j,-j,0,1,j,-j,2,j,1,-j,3,1,j,1,2,1,-j,-j,5]
    assert cyclotomic(1624,j) == 1
    assert cyclotomic(33600,j) == 1
    u = sqrt(j, prec=500)
    assert cyclotomic(8, u).ae(0)
    assert cyclotomic(30, u).ae(5.8284271247461900976)
    assert cyclotomic(2040, u).ae(1)
    assert cyclotomic(0,2.5) == 1
    assert cyclotomic(1,2.5) == 2.5-1
    assert cyclotomic(2,2.5) == 2.5+1
    assert cyclotomic(3,2.5) == 2.5**2 + 2.5 + 1
    assert cyclotomic(7,2.5) == 406.234375
