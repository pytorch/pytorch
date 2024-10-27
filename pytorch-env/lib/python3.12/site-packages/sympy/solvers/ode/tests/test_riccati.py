from sympy.core.random import randint
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.simplify.ratsimp import ratsimp
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import slow
from sympy.solvers.ode.riccati import (riccati_normal, riccati_inverse_normal,
    riccati_reduced, match_riccati, inverse_transform_poly, limit_at_inf,
    check_necessary_conds, val_at_inf, construct_c_case_1,
    construct_c_case_2, construct_c_case_3, construct_d_case_4,
    construct_d_case_5, construct_d_case_6, rational_laurent_series,
    solve_riccati)

f = Function('f')
x = symbols('x')

# These are the functions used to generate the tests
# SHOULD NOT BE USED DIRECTLY IN TESTS

def rand_rational(maxint):
    return Rational(randint(-maxint, maxint), randint(1, maxint))


def rand_poly(x, degree, maxint):
    return Poly([rand_rational(maxint) for _ in range(degree+1)], x)


def rand_rational_function(x, degree, maxint):
    degnum = randint(1, degree)
    degden = randint(1, degree)
    num = rand_poly(x, degnum, maxint)
    den = rand_poly(x, degden, maxint)
    while den == Poly(0, x):
        den = rand_poly(x, degden, maxint)
    return num / den


def find_riccati_ode(ratfunc, x, yf):
    y = ratfunc
    yp = y.diff(x)
    q1 = rand_rational_function(x, 1, 3)
    q2 = rand_rational_function(x, 1, 3)
    while q2 == 0:
        q2 = rand_rational_function(x, 1, 3)
    q0 = ratsimp(yp - q1*y - q2*y**2)
    eq = Eq(yf.diff(), q0 + q1*yf + q2*yf**2)
    sol = Eq(yf, y)
    assert checkodesol(eq, sol) == (True, 0)
    return eq, q0, q1, q2


# Testing functions start

def test_riccati_transformation():
    """
    This function tests the transformation of the
    solution of a Riccati ODE to the solution of
    its corresponding normal Riccati ODE.

    Each test case 4 values -

    1. w - The solution to be transformed
    2. b1 - The coefficient of f(x) in the ODE.
    3. b2 - The coefficient of f(x)**2 in the ODE.
    4. y - The solution to the normal Riccati ODE.
    """
    tests = [
    (
        x/(x - 1),
        (x**2 + 7)/3*x,
        x,
        -x**2/(x - 1) - x*(x**2/3 + S(7)/3)/2 - 1/(2*x)
    ),
    (
        (2*x + 3)/(2*x + 2),
        (3 - 3*x)/(x + 1),
        5*x,
        -5*x*(2*x + 3)/(2*x + 2) - (3 - 3*x)/(Mul(2, x + 1, evaluate=False)) - 1/(2*x)
    ),
    (
        -1/(2*x**2 - 1),
        0,
        (2 - x)/(4*x - 2),
        (2 - x)/((4*x - 2)*(2*x**2 - 1)) - (4*x - 2)*(Mul(-4, 2 - x, evaluate=False)/(4*x - \
        2)**2 - 1/(4*x - 2))/(Mul(2, 2 - x, evaluate=False))
    ),
    (
        x,
        (8*x - 12)/(12*x + 9),
        x**3/(6*x - 9),
        -x**4/(6*x - 9) - (8*x - 12)/(Mul(2, 12*x + 9, evaluate=False)) - (6*x - 9)*(-6*x**3/(6*x \
        - 9)**2 + 3*x**2/(6*x - 9))/(2*x**3)
    )]
    for w, b1, b2, y in tests:
        assert y == riccati_normal(w, x, b1, b2)
        assert w == riccati_inverse_normal(y, x, b1, b2).cancel()

    # Test bp parameter in riccati_inverse_normal
    tests = [
    (
        (-2*x - 1)/(2*x**2 + 2*x - 2),
        -2/x,
        (-x - 1)/(4*x),
        8*x**2*(1/(4*x) + (-x - 1)/(4*x**2))/(-x - 1)**2 + 4/(-x - 1),
        -2*x*(-1/(4*x) - (-x - 1)/(4*x**2))/(-x - 1) - (-2*x - 1)*(-x - 1)/(4*x*(2*x**2 + 2*x \
        - 2)) + 1/x
    ),
    (
        3/(2*x**2),
        -2/x,
        (-x - 1)/(4*x),
        8*x**2*(1/(4*x) + (-x - 1)/(4*x**2))/(-x - 1)**2 + 4/(-x - 1),
        -2*x*(-1/(4*x) - (-x - 1)/(4*x**2))/(-x - 1) + 1/x - Mul(3, -x - 1, evaluate=False)/(8*x**3)
    )]
    for w, b1, b2, bp, y in tests:
        assert y == riccati_normal(w, x, b1, b2)
        assert w == riccati_inverse_normal(y, x, b1, b2, bp).cancel()


def test_riccati_reduced():
    """
    This function tests the transformation of a
    Riccati ODE to its normal Riccati ODE.

    Each test case 2 values -

    1. eq - A Riccati ODE.
    2. normal_eq - The normal Riccati ODE of eq.
    """
    tests = [
    (
        f(x).diff(x) - x**2 - x*f(x) - x*f(x)**2,

        f(x).diff(x) + f(x)**2 + x**3 - x**2/4 - 3/(4*x**2)
    ),
    (
        6*x/(2*x + 9) + f(x).diff(x) - (x + 1)*f(x)**2/x,

        -3*x**2*(1/x + (-x - 1)/x**2)**2/(4*(-x - 1)**2) + Mul(6, \
        -x - 1, evaluate=False)/(2*x + 9) + f(x)**2 + f(x).diff(x) \
        - (-1 + (x + 1)/x)/(x*(-x - 1))
    ),
    (
        f(x)**2 + f(x).diff(x) - (x - 1)*f(x)/(-x - S(1)/2),

        -(2*x - 2)**2/(4*(2*x + 1)**2) + (2*x - 2)/(2*x + 1)**2 + \
        f(x)**2 + f(x).diff(x) - 1/(2*x + 1)
    ),
    (
        f(x).diff(x) - f(x)**2/x,

        f(x)**2 + f(x).diff(x) + 1/(4*x**2)
    ),
    (
        -3*(-x**2 - x + 1)/(x**2 + 6*x + 1) + f(x).diff(x) + f(x)**2/x,

        f(x)**2 + f(x).diff(x) + (3*x**2/(x**2 + 6*x + 1) + 3*x/(x**2 \
        + 6*x + 1) - 3/(x**2 + 6*x + 1))/x + 1/(4*x**2)
    ),
    (
        6*x/(2*x + 9) + f(x).diff(x) - (x + 1)*f(x)/x,

        False
    ),
    (
        f(x)*f(x).diff(x) - 1/x + f(x)/3 + f(x)**2/(x**2 - 2),

        False
    )]
    for eq, normal_eq in tests:
        assert normal_eq == riccati_reduced(eq, f, x)


def test_match_riccati():
    """
    This function tests if an ODE is Riccati or not.

    Each test case has 5 values -

    1. eq - The Riccati ODE.
    2. match - Boolean indicating if eq is a Riccati ODE.
    3. b0 -
    4. b1 - Coefficient of f(x) in eq.
    5. b2 - Coefficient of f(x)**2 in eq.
    """
    tests = [
    # Test Rational Riccati ODEs
    (
        f(x).diff(x) - (405*x**3 - 882*x**2 - 78*x + 92)/(243*x**4 \
        - 945*x**3 + 846*x**2 + 180*x - 72) - 2 - f(x)**2/(3*x + 1) \
        - (S(1)/3 - x)*f(x)/(S(1)/3 - 3*x/2),

        True,

        45*x**3/(27*x**4 - 105*x**3 + 94*x**2 + 20*x - 8) - 98*x**2/ \
        (27*x**4 - 105*x**3 + 94*x**2 + 20*x - 8) - 26*x/(81*x**4 - \
        315*x**3 + 282*x**2 + 60*x - 24) + 2 + 92/(243*x**4 - 945*x**3 \
        + 846*x**2 + 180*x - 72),

        Mul(-1, 2 - 6*x, evaluate=False)/(9*x - 2),

        1/(3*x + 1)
    ),
    (
        f(x).diff(x) + 4*x/27 - (x/3 - 1)*f(x)**2 - (2*x/3 + \
        1)*f(x)/(3*x + 2) - S(10)/27 - (265*x**2 + 423*x + 162) \
        /(324*x**3 + 216*x**2),

        True,

        -4*x/27 + S(10)/27 + 3/(6*x**3 + 4*x**2) + 47/(36*x**2 \
        + 24*x) + 265/(324*x + 216),

        Mul(-1, -2*x - 3, evaluate=False)/(9*x + 6),

        x/3 - 1
    ),
    (
        f(x).diff(x) - (304*x**5 - 745*x**4 + 631*x**3 - 876*x**2 \
        + 198*x - 108)/(36*x**6 - 216*x**5 + 477*x**4 - 567*x**3 + \
        360*x**2 - 108*x) - S(17)/9 - (x - S(3)/2)*f(x)/(x/2 - \
        S(3)/2) - (x/3 - 3)*f(x)**2/(3*x),

        True,

        304*x**4/(36*x**5 - 216*x**4 + 477*x**3 - 567*x**2 + 360*x - \
        108) - 745*x**3/(36*x**5 - 216*x**4 + 477*x**3 - 567*x**2 + \
        360*x - 108) + 631*x**2/(36*x**5 - 216*x**4 + 477*x**3 - 567* \
        x**2 + 360*x - 108) - 292*x/(12*x**5 - 72*x**4 + 159*x**3 - \
        189*x**2 + 120*x - 36) + S(17)/9 - 12/(4*x**6 - 24*x**5 + \
        53*x**4 - 63*x**3 + 40*x**2 - 12*x) + 22/(4*x**5 - 24*x**4 \
        + 53*x**3 - 63*x**2 + 40*x - 12),

        Mul(-1, 3 - 2*x, evaluate=False)/(x - 3),

        Mul(-1, 9 - x, evaluate=False)/(9*x)
    ),
    # Test Non-Rational Riccati ODEs
    (
        f(x).diff(x) - x**(S(3)/2)/(x**(S(1)/2) - 2) + x**2*f(x) + \
        x*f(x)**2/(x**(S(3)/4)),
        False, 0, 0, 0
    ),
    (
        f(x).diff(x) - sin(x**2) + exp(x)*f(x) + log(x)*f(x)**2,
        False, 0, 0, 0
    ),
    (
        f(x).diff(x) - tanh(x + sqrt(x)) + f(x) + x**4*f(x)**2,
        False, 0, 0, 0
    ),
    # Test Non-Riccati ODEs
    (
        (1 - x**2)*f(x).diff(x, 2) - 2*x*f(x).diff(x) + 20*f(x),
        False, 0, 0, 0
    ),
    (
        f(x).diff(x) - x**2 + x**3*f(x) + (x**2/(x + 1))*f(x)**3,
        False, 0, 0, 0
    ),
    (
        f(x).diff(x)*f(x)**2 + (x**2 - 1)/(x**3 + 1)*f(x) + 1/(2*x \
        + 3) + f(x)**2,
        False, 0, 0, 0
    )]
    for eq, res, b0, b1, b2 in tests:
        match, funcs = match_riccati(eq, f, x)
        assert match == res
        if res:
            assert [b0, b1, b2] == funcs


def test_val_at_inf():
    """
    This function tests the valuation of rational
    function at oo.

    Each test case has 3 values -

    1. num - Numerator of rational function.
    2. den - Denominator of rational function.
    3. val_inf - Valuation of rational function at oo
    """
    tests = [
    # degree(denom) > degree(numer)
    (
        Poly(10*x**3 + 8*x**2 - 13*x + 6, x),
        Poly(-13*x**10 - x**9 + 5*x**8 + 7*x**7 + 10*x**6 + 6*x**5 - 7*x**4 + 11*x**3 - 8*x**2 + 5*x + 13, x),
         7
    ),
    (
        Poly(1, x),
        Poly(-9*x**4 + 3*x**3 + 15*x**2 - 6*x - 14, x),
         4
    ),
    # degree(denom) == degree(numer)
    (
        Poly(-6*x**3 - 8*x**2 + 8*x - 6, x),
        Poly(-5*x**3 + 12*x**2 - 6*x - 9, x),
         0
    ),
    # degree(denom) < degree(numer)
    (
        Poly(12*x**8 - 12*x**7 - 11*x**6 + 8*x**5 + 3*x**4 - x**3 + x**2 - 11*x, x),
        Poly(-14*x**2 + x, x),
         -6
    ),
    (
        Poly(5*x**6 + 9*x**5 - 11*x**4 - 9*x**3 + x**2 - 4*x + 4, x),
        Poly(15*x**4 + 3*x**3 - 8*x**2 + 15*x + 12, x),
         -2
    )]
    for num, den, val in tests:
        assert val_at_inf(num, den, x) == val


def test_necessary_conds():
    """
    This function tests the necessary conditions for
    a Riccati ODE to have a rational particular solution.
    """
    # Valuation at Infinity is an odd negative integer
    assert check_necessary_conds(-3, [1, 2, 4]) == False
    # Valuation at Infinity is a positive integer lesser than 2
    assert check_necessary_conds(1, [1, 2, 4]) == False
    # Multiplicity of a pole is an odd integer greater than 1
    assert check_necessary_conds(2, [3, 1, 6]) == False
    # All values are correct
    assert check_necessary_conds(-10, [1, 2, 8, 12]) == True


def test_inverse_transform_poly():
    """
    This function tests the substitution x -> 1/x
    in rational functions represented using Poly.
    """
    fns = [
    (15*x**3 - 8*x**2 - 2*x - 6)/(18*x + 6),

    (180*x**5 + 40*x**4 + 80*x**3 + 30*x**2 - 60*x - 80)/(180*x**3 - 150*x**2 + 75*x + 12),

    (-15*x**5 - 36*x**4 + 75*x**3 - 60*x**2 - 80*x - 60)/(80*x**4 + 60*x**3 + 60*x**2 + 60*x - 80),

    (60*x**7 + 24*x**6 - 15*x**5 - 20*x**4 + 30*x**2 + 100*x - 60)/(240*x**2 - 20*x - 30),

    (30*x**6 - 12*x**5 + 15*x**4 - 15*x**2 + 10*x + 60)/(3*x**10 - 45*x**9 + 15*x**5 + 15*x**4 - 5*x**3 \
    + 15*x**2 + 45*x - 15)
    ]
    for f in fns:
        num, den = [Poly(e, x) for e in f.as_numer_denom()]
        num, den = inverse_transform_poly(num, den, x)
        assert f.subs(x, 1/x).cancel() == num/den


def test_limit_at_inf():
    """
    This function tests the limit at oo of a
    rational function.

    Each test case has 3 values -

    1. num - Numerator of rational function.
    2. den - Denominator of rational function.
    3. limit_at_inf - Limit of rational function at oo
    """
    tests = [
    # deg(denom) > deg(numer)
    (
        Poly(-12*x**2 + 20*x + 32, x),
        Poly(32*x**3 + 72*x**2 + 3*x - 32, x),
        0
    ),
    # deg(denom) < deg(numer)
    (
        Poly(1260*x**4 - 1260*x**3 - 700*x**2 - 1260*x + 1400, x),
        Poly(6300*x**3 - 1575*x**2 + 756*x - 540, x),
        oo
    ),
    # deg(denom) < deg(numer), one of the leading coefficients is negative
    (
        Poly(-735*x**8 - 1400*x**7 + 1680*x**6 - 315*x**5 - 600*x**4 + 840*x**3 - 525*x**2 \
        + 630*x + 3780, x),
        Poly(1008*x**7 - 2940*x**6 - 84*x**5 + 2940*x**4 - 420*x**3 + 1512*x**2 + 105*x + 168, x),
        -oo
    ),
    # deg(denom) == deg(numer)
    (
        Poly(105*x**7 - 960*x**6 + 60*x**5 + 60*x**4 - 80*x**3 + 45*x**2 + 120*x + 15, x),
        Poly(735*x**7 + 525*x**6 + 720*x**5 + 720*x**4 - 8400*x**3 - 2520*x**2 + 2800*x + 280, x),
        S(1)/7
    ),
    (
        Poly(288*x**4 - 450*x**3 + 280*x**2 - 900*x - 90, x),
        Poly(607*x**4 + 840*x**3 - 1050*x**2 + 420*x + 420, x),
        S(288)/607
    )]
    for num, den, lim in tests:
        assert limit_at_inf(num, den, x) == lim


def test_construct_c_case_1():
    """
    This function tests the Case 1 in the step
    to calculate coefficients of c-vectors.

    Each test case has 4 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. pole - Pole of a(x) for which c-vector is being
       calculated.
    4. c - The c-vector for the pole.
    """
    tests = [
    (
        Poly(-3*x**3 + 3*x**2 + 4*x - 5, x, extension=True),
        Poly(4*x**8 + 16*x**7 + 9*x**5 + 12*x**4 + 6*x**3 + 12*x**2, x, extension=True),
        S(0),
        [[S(1)/2 + sqrt(6)*I/6], [S(1)/2 - sqrt(6)*I/6]]
    ),
    (
        Poly(1200*x**3 + 1440*x**2 + 816*x + 560, x, extension=True),
        Poly(128*x**5 - 656*x**4 + 1264*x**3 - 1125*x**2 + 385*x + 49, x, extension=True),
        S(7)/4,
        [[S(1)/2 + sqrt(16367978)/634], [S(1)/2 - sqrt(16367978)/634]]
    ),
    (
        Poly(4*x + 2, x, extension=True),
        Poly(18*x**4 + (2 - 18*sqrt(3))*x**3 + (14 - 11*sqrt(3))*x**2 + (4 - 6*sqrt(3))*x \
            + 8*sqrt(3) + 16, x, domain='QQ<sqrt(3)>'),
        (S(1) + sqrt(3))/2,
        [[S(1)/2 + sqrt(Mul(4, 2*sqrt(3) + 4, evaluate=False)/(19*sqrt(3) + 44) + 1)/2], \
            [S(1)/2 - sqrt(Mul(4, 2*sqrt(3) + 4, evaluate=False)/(19*sqrt(3) + 44) + 1)/2]]
    )]
    for num, den, pole, c in tests:
        assert construct_c_case_1(num, den, x, pole) == c


def test_construct_c_case_2():
    """
    This function tests the Case 2 in the step
    to calculate coefficients of c-vectors.

    Each test case has 5 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. pole - Pole of a(x) for which c-vector is being
       calculated.
    4. mul - The multiplicity of the pole.
    5. c - The c-vector for the pole.
    """
    tests = [
    # Testing poles with multiplicity 2
    (
        Poly(1, x, extension=True),
        Poly((x - 1)**2*(x - 2), x, extension=True),
        1, 2,
        [[-I*(-1 - I)/2], [I*(-1 + I)/2]]
    ),
    (
        Poly(3*x**5 - 12*x**4 - 7*x**3 + 1, x, extension=True),
        Poly((3*x - 1)**2*(x + 2)**2, x, extension=True),
        S(1)/3, 2,
        [[-S(89)/98], [-S(9)/98]]
    ),
    # Testing poles with multiplicity 4
    (
        Poly(x**3 - x**2 + 4*x, x, extension=True),
        Poly((x - 2)**4*(x + 5)**2, x, extension=True),
        2, 4,
        [[7*sqrt(3)*(S(60)/343 - 4*sqrt(3)/7)/12, 2*sqrt(3)/7], \
        [-7*sqrt(3)*(S(60)/343 + 4*sqrt(3)/7)/12, -2*sqrt(3)/7]]
    ),
    (
        Poly(3*x**5 + x**4 + 3, x, extension=True),
        Poly((4*x + 1)**4*(x + 2), x, extension=True),
        -S(1)/4, 4,
        [[128*sqrt(439)*(-sqrt(439)/128 - S(55)/14336)/439, sqrt(439)/256], \
        [-128*sqrt(439)*(sqrt(439)/128 - S(55)/14336)/439, -sqrt(439)/256]]
    ),
    # Testing poles with multiplicity 6
    (
        Poly(x**3 + 2, x, extension=True),
        Poly((3*x - 1)**6*(x**2 + 1), x, extension=True),
        S(1)/3, 6,
        [[27*sqrt(66)*(-sqrt(66)/54 - S(131)/267300)/22, -2*sqrt(66)/1485, sqrt(66)/162], \
        [-27*sqrt(66)*(sqrt(66)/54 - S(131)/267300)/22, 2*sqrt(66)/1485, -sqrt(66)/162]]
    ),
    (
        Poly(x**2 + 12, x, extension=True),
        Poly((x - sqrt(2))**6, x, extension=True),
        sqrt(2), 6,
        [[sqrt(14)*(S(6)/7 - 3*sqrt(14))/28, sqrt(7)/7, sqrt(14)], \
        [-sqrt(14)*(S(6)/7 + 3*sqrt(14))/28, -sqrt(7)/7, -sqrt(14)]]
    )]
    for num, den, pole, mul, c in tests:
        assert construct_c_case_2(num, den, x, pole, mul) == c


def test_construct_c_case_3():
    """
    This function tests the Case 3 in the step
    to calculate coefficients of c-vectors.
    """
    assert construct_c_case_3() == [[1]]


def test_construct_d_case_4():
    """
    This function tests the Case 4 in the step
    to calculate coefficients of the d-vector.

    Each test case has 4 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. mul - Multiplicity of oo as a pole.
    4. d - The d-vector.
    """
    tests = [
    # Tests with multiplicity at oo = 2
    (
        Poly(-x**5 - 2*x**4 + 4*x**3 + 2*x + 5, x, extension=True),
        Poly(9*x**3 - 2*x**2 + 10*x - 2, x, extension=True),
        2,
        [[10*I/27, I/3, -3*I*(S(158)/243 - I/3)/2], \
        [-10*I/27, -I/3, 3*I*(S(158)/243 + I/3)/2]]
    ),
    (
        Poly(-x**6 + 9*x**5 + 5*x**4 + 6*x**3 + 5*x**2 + 6*x + 7, x, extension=True),
        Poly(x**4 + 3*x**3 + 12*x**2 - x + 7, x, extension=True),
        2,
        [[-6*I, I, -I*(17 - I)/2], [6*I, -I, I*(17 + I)/2]]
    ),
    # Tests with multiplicity at oo = 4
    (
        Poly(-2*x**6 - x**5 - x**4 - 2*x**3 - x**2 - 3*x - 3, x, extension=True),
        Poly(3*x**2 + 10*x + 7, x, extension=True),
        4,
        [[269*sqrt(6)*I/288, -17*sqrt(6)*I/36, sqrt(6)*I/3, -sqrt(6)*I*(S(16969)/2592 \
        - 2*sqrt(6)*I/3)/4], [-269*sqrt(6)*I/288, 17*sqrt(6)*I/36, -sqrt(6)*I/3, \
        sqrt(6)*I*(S(16969)/2592 + 2*sqrt(6)*I/3)/4]]
    ),
    (
        Poly(-3*x**5 - 3*x**4 - 3*x**3 - x**2 - 1, x, extension=True),
        Poly(12*x - 2, x, extension=True),
        4,
        [[41*I/192, 7*I/24, I/2, -I*(-S(59)/6912 - I)], \
        [-41*I/192, -7*I/24, -I/2, I*(-S(59)/6912 + I)]]
    ),
    # Tests with multiplicity at oo = 4
    (
        Poly(-x**7 - x**5 - x**4 - x**2 - x, x, extension=True),
        Poly(x + 2, x, extension=True),
        6,
        [[-5*I/2, 2*I, -I, I, -I*(-9 - 3*I)/2], [5*I/2, -2*I, I, -I, I*(-9 + 3*I)/2]]
    ),
    (
        Poly(-x**7 - x**6 - 2*x**5 - 2*x**4 - x**3 - x**2 + 2*x - 2, x, extension=True),
        Poly(2*x - 2, x, extension=True),
        6,
        [[3*sqrt(2)*I/4, 3*sqrt(2)*I/4, sqrt(2)*I/2, sqrt(2)*I/2, -sqrt(2)*I*(-S(7)/8 - \
        3*sqrt(2)*I/2)/2], [-3*sqrt(2)*I/4, -3*sqrt(2)*I/4, -sqrt(2)*I/2, -sqrt(2)*I/2, \
        sqrt(2)*I*(-S(7)/8 + 3*sqrt(2)*I/2)/2]]
    )]
    for num, den, mul, d in tests:
        ser = rational_laurent_series(num, den, x, oo, mul, 1)
        assert construct_d_case_4(ser, mul//2) == d


def test_construct_d_case_5():
    """
    This function tests the Case 5 in the step
    to calculate coefficients of the d-vector.

    Each test case has 3 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. d - The d-vector.
    """
    tests = [
    (
        Poly(2*x**3 + x**2 + x - 2, x, extension=True),
        Poly(9*x**3 + 5*x**2 + 2*x - 1, x, extension=True),
        [[sqrt(2)/3, -sqrt(2)/108], [-sqrt(2)/3, sqrt(2)/108]]
    ),
    (
        Poly(3*x**5 + x**4 - x**3 + x**2 - 2*x - 2, x, domain='ZZ'),
        Poly(9*x**5 + 7*x**4 + 3*x**3 + 2*x**2 + 5*x + 7, x, domain='ZZ'),
        [[sqrt(3)/3, -2*sqrt(3)/27], [-sqrt(3)/3, 2*sqrt(3)/27]]
    ),
    (
        Poly(x**2 - x + 1, x, domain='ZZ'),
        Poly(3*x**2 + 7*x + 3, x, domain='ZZ'),
        [[sqrt(3)/3, -5*sqrt(3)/9], [-sqrt(3)/3, 5*sqrt(3)/9]]
    )]
    for num, den, d in tests:
        # Multiplicity of oo is 0
        ser = rational_laurent_series(num, den, x, oo, 0, 1)
        assert construct_d_case_5(ser) == d


def test_construct_d_case_6():
    """
    This function tests the Case 6 in the step
    to calculate coefficients of the d-vector.

    Each test case has 3 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. d - The d-vector.
    """
    tests = [
    (
        Poly(-2*x**2 - 5, x, domain='ZZ'),
        Poly(4*x**4 + 2*x**2 + 10*x + 2, x, domain='ZZ'),
        [[S(1)/2 + I/2], [S(1)/2 - I/2]]
    ),
    (
        Poly(-2*x**3 - 4*x**2 - 2*x - 5, x, domain='ZZ'),
        Poly(x**6 - x**5 + 2*x**4 - 4*x**3 - 5*x**2 - 5*x + 9, x, domain='ZZ'),
        [[1], [0]]
    ),
    (
        Poly(-5*x**3 + x**2 + 11*x + 12, x, domain='ZZ'),
        Poly(6*x**8 - 26*x**7 - 27*x**6 - 10*x**5 - 44*x**4 - 46*x**3 - 34*x**2 \
        - 27*x - 42, x, domain='ZZ'),
        [[1], [0]]
    )]
    for num, den, d in tests:
        assert construct_d_case_6(num, den, x) == d


def test_rational_laurent_series():
    """
    This function tests the computation of coefficients
    of Laurent series of a rational function.

    Each test case has 5 values -

    1. num - Numerator of the rational function.
    2. den - Denominator of the rational function.
    3. x0 - Point about which Laurent series is to
       be calculated.
    4. mul - Multiplicity of x0 if x0 is a pole of
       the rational function (0 otherwise).
    5. n - Number of terms upto which the series
       is to be calculated.
    """
    tests = [
    # Laurent series about simple pole (Multiplicity = 1)
    (
        Poly(x**2 - 3*x + 9, x, extension=True),
        Poly(x**2 - x, x, extension=True),
        S(1), 1, 6,
        {1: 7, 0: -8, -1: 9, -2: -9, -3: 9, -4: -9}
    ),
    # Laurent series about multiple pole (Multiplicity > 1)
    (
        Poly(64*x**3 - 1728*x + 1216, x, extension=True),
        Poly(64*x**4 - 80*x**3 - 831*x**2 + 1809*x - 972, x, extension=True),
        S(9)/8, 2, 3,
        {0: S(32177152)/46521675, 2: S(1019)/984, -1: S(11947565056)/28610830125, \
        1: S(209149)/75645}
    ),
    (
        Poly(1, x, extension=True),
        Poly(x**5 + (-4*sqrt(2) - 1)*x**4 + (4*sqrt(2) + 12)*x**3 + (-12 - 8*sqrt(2))*x**2 \
        + (4 + 8*sqrt(2))*x - 4, x, extension=True),
        sqrt(2), 4, 6,
        {4: 1 + sqrt(2), 3: -3 - 2*sqrt(2), 2: Mul(-1, -3 - 2*sqrt(2), evaluate=False)/(-1 \
        + sqrt(2)), 1: (-3 - 2*sqrt(2))/(-1 + sqrt(2))**2, 0: Mul(-1, -3 - 2*sqrt(2), evaluate=False \
        )/(-1 + sqrt(2))**3, -1: (-3 - 2*sqrt(2))/(-1 + sqrt(2))**4}
    ),
    # Laurent series about oo
    (
        Poly(x**5 - 4*x**3 + 6*x**2 + 10*x - 13, x, extension=True),
        Poly(x**2 - 5, x, extension=True),
        oo, 3, 6,
        {3: 1, 2: 0, 1: 1, 0: 6, -1: 15, -2: 17}
    ),
    # Laurent series at x0 where x0 is not a pole of the function
    # Using multiplicity as 0 (as x0 will not be a pole)
    (
        Poly(3*x**3 + 6*x**2 - 2*x + 5, x, extension=True),
        Poly(9*x**4 - x**3 - 3*x**2 + 4*x + 4, x, extension=True),
        S(2)/5, 0, 1,
        {0: S(3345)/3304, -1: S(399325)/2729104, -2: S(3926413375)/4508479808, \
        -3: S(-5000852751875)/1862002160704, -4: S(-6683640101653125)/6152055138966016}
    ),
    (
        Poly(-7*x**2 + 2*x - 4, x, extension=True),
        Poly(7*x**5 + 9*x**4 + 8*x**3 + 3*x**2 + 6*x + 9, x, extension=True),
        oo, 0, 6,
        {0: 0, -2: 0, -5: -S(71)/49, -1: 0, -3: -1, -4: S(11)/7}
    )]
    for num, den, x0, mul, n, ser in tests:
        assert ser == rational_laurent_series(num, den, x, x0, mul, n)


def check_dummy_sol(eq, solse, dummy_sym):
    """
    Helper function to check if actual solution
    matches expected solution if actual solution
    contains dummy symbols.
    """
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs
    _, funcs = match_riccati(eq, f, x)

    sols = solve_riccati(f(x), x, *funcs)
    C1 = Dummy('C1')
    sols = [sol.subs(C1, dummy_sym) for sol in sols]

    assert all(x[0] for x in checkodesol(eq, sols))
    assert all(s1.dummy_eq(s2, dummy_sym) for s1, s2 in zip(sols, solse))


def test_solve_riccati():
    """
    This function tests the computation of rational
    particular solutions for a Riccati ODE.

    Each test case has 2 values -

    1. eq - Riccati ODE to be solved.
    2. sol - Expected solution to the equation.

    Some examples have been taken from the paper - "Statistical Investigation of
    First-Order Algebraic ODEs and their Rational General Solutions" by
    Georg Grasegger, N. Thieu Vo, Franz Winkler

    https://www3.risc.jku.at/publications/download/risc_5197/RISCReport15-19.pdf
    """
    C0 = Dummy('C0')
    # Type: 1st Order Rational Riccati, dy/dx = a + b*y + c*y**2,
    # a, b, c are rational functions of x

    tests = [
    # a(x) is a constant
    (
        Eq(f(x).diff(x) + f(x)**2 - 2, 0),
        [Eq(f(x), sqrt(2)), Eq(f(x), -sqrt(2))]
    ),
    # a(x) is a constant
    (
        f(x)**2 + f(x).diff(x) + 4*f(x)/x + 2/x**2,
        [Eq(f(x), (-2*C0 - x)/(C0*x + x**2))]
    ),
    # a(x) is a constant
    (
        2*x**2*f(x).diff(x) - x*(4*f(x) + f(x).diff(x) - 4) + (f(x) - 1)*f(x),
        [Eq(f(x), (C0 + 2*x**2)/(C0 + x))]
    ),
    # Pole with multiplicity 1
    (
        Eq(f(x).diff(x), -f(x)**2 - 2/(x**3 - x**2)),
        [Eq(f(x), 1/(x**2 - x))]
    ),
    # One pole of multiplicity 2
    (
        x**2 - (2*x + 1/x)*f(x) + f(x)**2 + f(x).diff(x),
        [Eq(f(x), (C0*x + x**3 + 2*x)/(C0 + x**2)), Eq(f(x), x)]
    ),
    (
        x**4*f(x).diff(x) + x**2 - x*(2*f(x)**2 + f(x).diff(x)) + f(x),
        [Eq(f(x), (C0*x**2 + x)/(C0 + x**2)), Eq(f(x), x**2)]
    ),
    # Multiple poles of multiplicity 2
    (
        -f(x)**2 + f(x).diff(x) + (15*x**2 - 20*x + 7)/((x - 1)**2*(2*x \
            - 1)**2),
        [Eq(f(x), (9*C0*x - 6*C0 - 15*x**5 + 60*x**4 - 94*x**3 + 72*x**2 \
        - 30*x + 6)/(6*C0*x**2 - 9*C0*x + 3*C0 + 6*x**6 - 29*x**5 + \
        57*x**4 - 58*x**3 + 30*x**2 - 6*x)), Eq(f(x), (3*x - 2)/(2*x**2 \
        - 3*x + 1))]
    ),
    # Regression: Poles with even multiplicity > 2 fixed
    (
        f(x)**2 + f(x).diff(x) - (4*x**6 - 8*x**5 + 12*x**4 + 4*x**3 + \
            7*x**2 - 20*x + 4)/(4*x**4),
        [Eq(f(x), (2*x**5 - 2*x**4 - x**3 + 4*x**2 + 3*x - 2)/(2*x**4 \
            - 2*x**2))]
    ),
    # Regression: Poles with even multiplicity > 2 fixed
    (
        Eq(f(x).diff(x), (-x**6 + 15*x**4 - 40*x**3 + 45*x**2 - 24*x + 4)/\
            (x**12 - 12*x**11 + 66*x**10 - 220*x**9 + 495*x**8 - 792*x**7 + 924*x**6 - \
            792*x**5 + 495*x**4 - 220*x**3 + 66*x**2 - 12*x + 1) + f(x)**2 + f(x)),
        [Eq(f(x), 1/(x**6 - 6*x**5 + 15*x**4 - 20*x**3 + 15*x**2 - 6*x + 1))]
    ),
    # More than 2 poles with multiplicity 2
    # Regression: Fixed mistake in necessary conditions
    (
        Eq(f(x).diff(x), x*f(x) + 2*x + (3*x - 2)*f(x)**2/(4*x + 2) + \
            (8*x**2 - 7*x + 26)/(16*x**3 - 24*x**2 + 8) - S(3)/2),
        [Eq(f(x), (1 - 4*x)/(2*x - 2))]
    ),
    # Regression: Fixed mistake in necessary conditions
    (
        Eq(f(x).diff(x), (-12*x**2 - 48*x - 15)/(24*x**3 - 40*x**2 + 8*x + 8) \
            + 3*f(x)**2/(6*x + 2)),
        [Eq(f(x), (2*x + 1)/(2*x - 2))]
    ),
    # Imaginary poles
    (
        f(x).diff(x) + (3*x**2 + 1)*f(x)**2/x + (6*x**2 - x + 3)*f(x)/(x*(x \
            - 1)) + (3*x**2 - 2*x + 2)/(x*(x - 1)**2),
        [Eq(f(x), (-C0 - x**3 + x**2 - 2*x)/(C0*x - C0 + x**4 - x**3 + x**2 \
            - x)), Eq(f(x), -1/(x - 1))],
    ),
    # Imaginary coefficients in equation
    (
        f(x).diff(x) - 2*I*(f(x)**2 + 1)/x,
        [Eq(f(x), (-I*C0 + I*x**4)/(C0 + x**4)), Eq(f(x), -I)]
    ),
    # Regression: linsolve returning empty solution
    # Large value of m (> 10)
    (
        Eq(f(x).diff(x), x*f(x)/(S(3)/2 - 2*x) + (x/2 - S(1)/3)*f(x)**2/\
            (2*x/3 - S(1)/2) - S(5)/4 + (281*x**2 - 1260*x + 756)/(16*x**3 - 12*x**2)),
        [Eq(f(x), (9 - x)/x), Eq(f(x), (40*x**14 + 28*x**13 + 420*x**12 + 2940*x**11 + \
            18480*x**10 + 103950*x**9 + 519750*x**8 + 2286900*x**7 + 8731800*x**6 + 28378350*\
            x**5 + 76403250*x**4 + 163721250*x**3 + 261954000*x**2 + 278326125*x + 147349125)/\
            ((24*x**14 + 140*x**13 + 840*x**12 + 4620*x**11 + 23100*x**10 + 103950*x**9 + \
            415800*x**8 + 1455300*x**7 + 4365900*x**6 + 10914750*x**5 + 21829500*x**4 + 32744250\
            *x**3 + 32744250*x**2 + 16372125*x)))]
    ),
    # Regression: Fixed bug due to a typo in paper
    (
        Eq(f(x).diff(x), 18*x**3 + 18*x**2 + (-x/2 - S(1)/2)*f(x)**2 + 6),
        [Eq(f(x), 6*x)]
    ),
    # Regression: Fixed bug due to a typo in paper
    (
        Eq(f(x).diff(x), -3*x**3/4 + 15*x/2 + (x/3 - S(4)/3)*f(x)**2 \
            + 9 + (1 - x)*f(x)/x + 3/x),
        [Eq(f(x), -3*x/2 - 3)]
    )]
    for eq, sol in tests:
        check_dummy_sol(eq, sol, C0)


@slow
def test_solve_riccati_slow():
    """
    This function tests the computation of rational
    particular solutions for a Riccati ODE.

    Each test case has 2 values -

    1. eq - Riccati ODE to be solved.
    2. sol - Expected solution to the equation.
    """
    C0 = Dummy('C0')
    tests = [
    # Very large values of m (989 and 991)
    (
        Eq(f(x).diff(x), (1 - x)*f(x)/(x - 3) + (2 - 12*x)*f(x)**2/(2*x - 9) + \
            (54924*x**3 - 405264*x**2 + 1084347*x - 1087533)/(8*x**4 - 132*x**3 + 810*x**2 - \
            2187*x + 2187) + 495),
        [Eq(f(x), (18*x + 6)/(2*x - 9))]
    )]
    for eq, sol in tests:
        check_dummy_sol(eq, sol, C0)
