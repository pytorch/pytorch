import pytest
from mpmath import *

def ae(a, b):
    return abs(a-b) < 10**(-mp.dps+5)

def test_basic_integrals():
    for prec in [15, 30, 100]:
        mp.dps = prec
        assert ae(quadts(lambda x: x**3 - 3*x**2, [-2, 4]), -12)
        assert ae(quadgl(lambda x: x**3 - 3*x**2, [-2, 4]), -12)
        assert ae(quadts(sin, [0, pi]), 2)
        assert ae(quadts(sin, [0, 2*pi]), 0)
        assert ae(quadts(exp, [-inf, -1]), 1/e)
        assert ae(quadts(lambda x: exp(-x), [0, inf]), 1)
        assert ae(quadts(lambda x: exp(-x*x), [-inf, inf]), sqrt(pi))
        assert ae(quadts(lambda x: 1/(1+x*x), [-1, 1]), pi/2)
        assert ae(quadts(lambda x: 1/(1+x*x), [-inf, inf]), pi)
        assert ae(quadts(lambda x: 2*sqrt(1-x*x), [-1, 1]), pi)
    mp.dps = 15

def test_multiple_intervals():
    y,err = quad(lambda x: sign(x), [-0.5, 0.9, 1], maxdegree=2, error=True)
    assert abs(y-0.5) < 2*err

def test_quad_symmetry():
    assert quadts(sin, [-1, 1]) == 0
    assert quadgl(sin, [-1, 1]) == 0

def test_quad_infinite_mirror():
    # Check mirrored infinite interval
    assert ae(quad(lambda x: exp(-x*x), [inf,-inf]), -sqrt(pi))
    assert ae(quad(lambda x: exp(x), [0,-inf]), -1)

def test_quadgl_linear():
    assert quadgl(lambda x: x, [0, 1], maxdegree=1).ae(0.5)

def test_complex_integration():
    assert quadts(lambda x: x, [0, 1+j]).ae(j)

def test_quadosc():
    mp.dps = 15
    assert quadosc(lambda x: sin(x)/x, [0, inf], period=2*pi).ae(pi/2)

# Double integrals
def test_double_trivial():
    assert ae(quadts(lambda x, y: x, [0, 1], [0, 1]), 0.5)
    assert ae(quadts(lambda x, y: x, [-1, 1], [-1, 1]), 0.0)

def test_double_1():
    assert ae(quadts(lambda x, y: cos(x+y/2), [-pi/2, pi/2], [0, pi]), 4)

def test_double_2():
    assert ae(quadts(lambda x, y: (x-1)/((1-x*y)*log(x*y)), [0, 1], [0, 1]), euler)

def test_double_3():
    assert ae(quadts(lambda x, y: 1/sqrt(1+x*x+y*y), [-1, 1], [-1, 1]), 4*log(2+sqrt(3))-2*pi/3)

def test_double_4():
    assert ae(quadts(lambda x, y: 1/(1-x*x * y*y), [0, 1], [0, 1]), pi**2 / 8)

def test_double_5():
    assert ae(quadts(lambda x, y: 1/(1-x*y), [0, 1], [0, 1]), pi**2 / 6)

def test_double_6():
    assert ae(quadts(lambda x, y: exp(-(x+y)), [0, inf], [0, inf]), 1)

def test_double_7():
    assert ae(quadts(lambda x, y: exp(-x*x-y*y), [-inf, inf], [-inf, inf]), pi)


# Test integrals from "Experimentation in Mathematics" by Borwein,
# Bailey & Girgensohn
def test_expmath_integrals():
    for prec in [15, 30, 50]:
        mp.dps = prec
        assert ae(quadts(lambda x: x/sinh(x), [0, inf]),                    pi**2 / 4)
        assert ae(quadts(lambda x: log(x)**2 / (1+x**2), [0, inf]),         pi**3 / 8)
        assert ae(quadts(lambda x: (1+x**2)/(1+x**4), [0, inf]),            pi/sqrt(2))
        assert ae(quadts(lambda x: log(x)/cosh(x)**2, [0, inf]),            log(pi)-2*log(2)-euler)
        assert ae(quadts(lambda x: log(1+x**3)/(1-x+x**2), [0, inf]),       2*pi*log(3)/sqrt(3))
        assert ae(quadts(lambda x: log(x)**2 / (x**2+x+1), [0, 1]),         8*pi**3 / (81*sqrt(3)))
        assert ae(quadts(lambda x: log(cos(x))**2, [0, pi/2]),              pi/2 * (log(2)**2+pi**2/12))
        assert ae(quadts(lambda x: x**2 / sin(x)**2, [0, pi/2]),            pi*log(2))
        assert ae(quadts(lambda x: x**2/sqrt(exp(x)-1), [0, inf]),          4*pi*(log(2)**2 + pi**2/12))
        assert ae(quadts(lambda x: x*exp(-x)*sqrt(1-exp(-2*x)), [0, inf]),  pi*(1+2*log(2))/8)
    mp.dps = 15

# Do not reach full accuracy
@pytest.mark.xfail
def test_expmath_fail():
    assert ae(quadts(lambda x: sqrt(tan(x)), [0, pi/2]),          pi*sqrt(2)/2)
    assert ae(quadts(lambda x: atan(x)/(x*sqrt(1-x**2)), [0, 1]), pi*log(1+sqrt(2))/2)
    assert ae(quadts(lambda x: log(1+x**2)/x**2, [0, 1]),         pi/2-log(2))
    assert ae(quadts(lambda x: x**2/((1+x**4)*sqrt(1-x**4)), [0, 1]),     pi/8)
