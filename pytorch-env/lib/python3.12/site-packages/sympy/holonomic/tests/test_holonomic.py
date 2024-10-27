from sympy.holonomic import (DifferentialOperator, HolonomicFunction,
                             DifferentialOperators, from_hyper,
                             from_meijerg, expr_to_holonomic)
from sympy.holonomic.recurrence import RecurrenceOperators, HolonomicSequence
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import besselj
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import (Ci, Si, erf, erfc)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.simplify.hyperexpand import hyperexpand
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR


def test_DifferentialOperator():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    assert Dx == R.derivative_operator
    assert Dx == DifferentialOperator([R.base.zero, R.base.one], R)
    assert x * Dx + x**2 * Dx**2 == DifferentialOperator([0, x, x**2], R)
    assert (x**2 + 1) + Dx + x * \
        Dx**5 == DifferentialOperator([x**2 + 1, 1, 0, 0, 0, x], R)
    assert (x * Dx + x**2 + 1 - Dx * (x**3 + x))**3 == (-48 * x**6) + \
        (-57 * x**7) * Dx + (-15 * x**8) * Dx**2 + (-x**9) * Dx**3
    p = (x * Dx**2 + (x**2 + 3) * Dx**5) * (Dx + x**2)
    q = (2 * x) + (4 * x**2) * Dx + (x**3) * Dx**2 + \
        (20 * x**2 + x + 60) * Dx**3 + (10 * x**3 + 30 * x) * Dx**4 + \
        (x**4 + 3 * x**2) * Dx**5 + (x**2 + 3) * Dx**6
    assert p == q


def test_HolonomicFunction_addition():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx**2 * x, x)
    q = HolonomicFunction((2) * Dx + (x) * Dx**2, x)
    assert p == q
    p = HolonomicFunction(x * Dx + 1, x)
    q = HolonomicFunction(Dx + 1, x)
    r = HolonomicFunction((x - 2) + (x**2 - 2) * Dx + (x**2 - x) * Dx**2, x)
    assert p + q == r
    p = HolonomicFunction(x * Dx + Dx**2 * (x**2 + 2), x)
    q = HolonomicFunction(Dx - 3, x)
    r = HolonomicFunction((-54 * x**2 - 126 * x - 150) + (-135 * x**3 - 252 * x**2 - 270 * x + 140) * Dx +\
                 (-27 * x**4 - 24 * x**2 + 14 * x - 150) * Dx**2 + \
                 (9 * x**4 + 15 * x**3 + 38 * x**2 + 30 * x +40) * Dx**3, x)
    assert p + q == r
    p = HolonomicFunction(Dx**5 - 1, x)
    q = HolonomicFunction(x**3 + Dx, x)
    r = HolonomicFunction((-x**18 + 45*x**14 - 525*x**10 + 1575*x**6 - x**3 - 630*x**2) + \
        (-x**15 + 30*x**11 - 195*x**7 + 210*x**3 - 1)*Dx + (x**18 - 45*x**14 + 525*x**10 - \
        1575*x**6 + x**3 + 630*x**2)*Dx**5 + (x**15 - 30*x**11 + 195*x**7 - 210*x**3 + \
        1)*Dx**6, x)
    assert p+q == r

    p = x**2 + 3*x + 8
    q = x**3 - 7*x + 5
    p = p*Dx - p.diff()
    q = q*Dx - q.diff()
    r = HolonomicFunction(p, x) + HolonomicFunction(q, x)
    s = HolonomicFunction((6*x**2 + 18*x + 14) + (-4*x**3 - 18*x**2 - 62*x + 10)*Dx +\
        (x**4 + 6*x**3 + 31*x**2 - 10*x - 71)*Dx**2, x)
    assert r == s


def test_HolonomicFunction_multiplication():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx+x+x*Dx**2, x)
    q = HolonomicFunction(x*Dx+Dx*x+Dx**2, x)
    r = HolonomicFunction((8*x**6 + 4*x**4 + 6*x**2 + 3) + (24*x**5 - 4*x**3 + 24*x)*Dx + \
        (8*x**6 + 20*x**4 + 12*x**2 + 2)*Dx**2 + (8*x**5 + 4*x**3 + 4*x)*Dx**3 + \
        (2*x**4 + x**2)*Dx**4, x)
    assert p*q == r
    p = HolonomicFunction(Dx**2+1, x)
    q = HolonomicFunction(Dx-1, x)
    r = HolonomicFunction((2) + (-2)*Dx + (1)*Dx**2, x)
    assert p*q == r
    p = HolonomicFunction(Dx**2+1+x+Dx, x)
    q = HolonomicFunction((Dx*x-1)**2, x)
    r = HolonomicFunction((4*x**7 + 11*x**6 + 16*x**5 + 4*x**4 - 6*x**3 - 7*x**2 - 8*x - 2) + \
        (8*x**6 + 26*x**5 + 24*x**4 - 3*x**3 - 11*x**2 - 6*x - 2)*Dx + \
        (8*x**6 + 18*x**5 + 15*x**4 - 3*x**3 - 6*x**2 - 6*x - 2)*Dx**2 + (8*x**5 + \
            10*x**4 + 6*x**3 - 2*x**2 - 4*x)*Dx**3 + (4*x**5 + 3*x**4 - x**2)*Dx**4, x)
    assert p*q == r
    p = HolonomicFunction(x*Dx**2-1, x)
    q = HolonomicFunction(Dx*x-x, x)
    r = HolonomicFunction((x - 3) + (-2*x + 2)*Dx + (x)*Dx**2, x)
    assert p*q == r


def test_HolonomicFunction_power():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx+x+x*Dx**2, x)
    a = HolonomicFunction(Dx, x)
    for n in range(10):
        assert a == p**n
        a *= p


def test_addition_initial_condition():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx-1, x, 0, [3])
    q = HolonomicFunction(Dx**2+1, x, 0, [1, 0])
    r = HolonomicFunction(-1 + Dx - Dx**2 + Dx**3, x, 0, [4, 3, 2])
    assert p + q == r
    p = HolonomicFunction(Dx - x + Dx**2, x, 0, [1, 2])
    q = HolonomicFunction(Dx**2 + x, x, 0, [1, 0])
    r = HolonomicFunction((-x**4 - x**3/4 - x**2 + Rational(1, 4)) + (x**3 + x**2/4 + x*Rational(3, 4) + 1)*Dx + \
        (x*Rational(-3, 2) + Rational(7, 4))*Dx**2 + (x**2 - x*Rational(7, 4) + Rational(1, 4))*Dx**3 + (x**2 + x/4 + S.Half)*Dx**4, x, 0, [2, 2, -2, 2])
    assert p + q == r
    p = HolonomicFunction(Dx**2 + 4*x*Dx + x**2, x, 0, [3, 4])
    q = HolonomicFunction(Dx**2 + 1, x, 0, [1, 1])
    r = HolonomicFunction((x**6 + 2*x**4 - 5*x**2 - 6) + (4*x**5 + 36*x**3 - 32*x)*Dx + \
         (x**6 + 3*x**4 + 5*x**2 - 9)*Dx**2 + (4*x**5 + 36*x**3 - 32*x)*Dx**3 + (x**4 + \
            10*x**2 - 3)*Dx**4, x, 0, [4, 5, -1, -17])
    assert p + q == r
    q = HolonomicFunction(Dx**3 + x, x, 2, [3, 0, 1])
    p = HolonomicFunction(Dx - 1, x, 2, [1])
    r = HolonomicFunction((-x**2 - x + 1) + (x**2 + x)*Dx + (-x - 2)*Dx**3 + \
        (x + 1)*Dx**4, x, 2, [4, 1, 2, -5 ])
    assert p + q == r
    p = expr_to_holonomic(sin(x))
    q = expr_to_holonomic(1/x, x0=1)
    r = HolonomicFunction((x**2 + 6) + (x**3 + 2*x)*Dx + (x**2 + 6)*Dx**2 + (x**3 + 2*x)*Dx**3, \
        x, 1, [sin(1) + 1, -1 + cos(1), -sin(1) + 2])
    assert p + q == r
    C_1 = symbols('C_1')
    p = expr_to_holonomic(sqrt(x))
    q = expr_to_holonomic(sqrt(x**2-x))
    r = (p + q).to_expr().subs(C_1, -I/2).expand()
    assert r == I*sqrt(x)*sqrt(-x + 1) + sqrt(x)


def test_multiplication_initial_condition():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx**2 + x*Dx - 1, x, 0, [3, 1])
    q = HolonomicFunction(Dx**2 + 1, x, 0, [1, 1])
    r = HolonomicFunction((x**4 + 14*x**2 + 60) + 4*x*Dx + (x**4 + 9*x**2 + 20)*Dx**2 + \
        (2*x**3 + 18*x)*Dx**3 + (x**2 + 10)*Dx**4, x, 0, [3, 4, 2, 3])
    assert p * q == r
    p = HolonomicFunction(Dx**2 + x, x, 0, [1, 0])
    q = HolonomicFunction(Dx**3 - x**2, x, 0, [3, 3, 3])
    r = HolonomicFunction((x**8 - 37*x**7/27 - 10*x**6/27 - 164*x**5/9 - 184*x**4/9 + \
        160*x**3/27 + 404*x**2/9 + 8*x + Rational(40, 3)) + (6*x**7 - 128*x**6/9 - 98*x**5/9 - 28*x**4/9 + \
        8*x**3/9 + 28*x**2 + x*Rational(40, 9) - 40)*Dx + (3*x**6 - 82*x**5/9 + 76*x**4/9 + 4*x**3/3 + \
        220*x**2/9 - x*Rational(80, 3))*Dx**2 + (-2*x**6 + 128*x**5/27 - 2*x**4/3 -80*x**2/9 + Rational(200, 9))*Dx**3 + \
        (3*x**5 - 64*x**4/9 - 28*x**3/9 + 6*x**2 - x*Rational(20, 9) - Rational(20, 3))*Dx**4 + (-4*x**3 + 64*x**2/9 + \
            x*Rational(8, 3))*Dx**5 + (x**4 - 64*x**3/27 - 4*x**2/3 + Rational(20, 9))*Dx**6, x, 0, [3, 3, 3, -3, -12, -24])
    assert p * q == r
    p = HolonomicFunction(Dx - 1, x, 0, [2])
    q = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])
    r = HolonomicFunction(2 -2*Dx + Dx**2, x, 0, [0, 2])
    assert p * q == r
    q = HolonomicFunction(x*Dx**2 + 1 + 2*Dx, x, 0,[0, 1])
    r = HolonomicFunction((x - 1) + (-2*x + 2)*Dx + x*Dx**2, x, 0, [0, 2])
    assert p * q == r
    p = HolonomicFunction(Dx**2 - 1, x, 0, [1, 3])
    q = HolonomicFunction(Dx**3 + 1, x, 0, [1, 2, 1])
    r = HolonomicFunction(6*Dx + 3*Dx**2 + 2*Dx**3 - 3*Dx**4 + Dx**6, x, 0, [1, 5, 14, 17, 17, 2])
    assert p * q == r
    p = expr_to_holonomic(sin(x))
    q = expr_to_holonomic(1/x, x0=1)
    r = HolonomicFunction(x + 2*Dx + x*Dx**2, x, 1, [sin(1), -sin(1) + cos(1)])
    assert p * q == r
    p = expr_to_holonomic(sqrt(x))
    q = expr_to_holonomic(sqrt(x**2-x))
    r = (p * q).to_expr()
    assert r == I*x*sqrt(-x + 1)


def test_HolonomicFunction_composition():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx-1, x).composition(x**2+x)
    r = HolonomicFunction((-2*x - 1) + Dx, x)
    assert p == r
    p = HolonomicFunction(Dx**2+1, x).composition(x**5+x**2+1)
    r = HolonomicFunction((125*x**12 + 150*x**9 + 60*x**6 + 8*x**3) + (-20*x**3 - 2)*Dx + \
        (5*x**4 + 2*x)*Dx**2, x)
    assert p == r
    p = HolonomicFunction(Dx**2*x+x, x).composition(2*x**3+x**2+1)
    r = HolonomicFunction((216*x**9 + 324*x**8 + 180*x**7 + 152*x**6 + 112*x**5 + \
        36*x**4 + 4*x**3) + (24*x**4 + 16*x**3 + 3*x**2 - 6*x - 1)*Dx + (6*x**5 + 5*x**4 + \
        x**3 + 3*x**2 + x)*Dx**2, x)
    assert p == r
    p = HolonomicFunction(Dx**2+1, x).composition(1-x**2)
    r = HolonomicFunction((4*x**3) - Dx + x*Dx**2, x)
    assert p == r
    p = HolonomicFunction(Dx**2+1, x).composition(x - 2/(x**2 + 1))
    r = HolonomicFunction((x**12 + 6*x**10 + 12*x**9 + 15*x**8 + 48*x**7 + 68*x**6 + \
        72*x**5 + 111*x**4 + 112*x**3 + 54*x**2 + 12*x + 1) + (12*x**8 + 32*x**6 + \
        24*x**4 - 4)*Dx + (x**12 + 6*x**10 + 4*x**9 + 15*x**8 + 16*x**7 + 20*x**6 + 24*x**5+ \
        15*x**4 + 16*x**3 + 6*x**2 + 4*x + 1)*Dx**2, x)
    assert p == r


def test_from_hyper():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = hyper([1, 1], [Rational(3, 2)], x**2/4)
    q = HolonomicFunction((4*x) + (5*x**2 - 8)*Dx + (x**3 - 4*x)*Dx**2, x, 1, [2*sqrt(3)*pi/9, -4*sqrt(3)*pi/27 + Rational(4, 3)])
    r = from_hyper(p)
    assert r == q
    p = from_hyper(hyper([1], [Rational(3, 2)], x**2/4))
    q = HolonomicFunction(-x + (-x**2/2 + 2)*Dx + x*Dx**2, x)
    # x0 = 1
    y0 = '[sqrt(pi)*exp(1/4)*erf(1/2), -sqrt(pi)*exp(1/4)*erf(1/2)/2 + 1]'
    assert sstr(p.y0) == y0
    assert q.annihilator == p.annihilator


def test_from_meijerg():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = from_meijerg(meijerg(([], [Rational(3, 2)]), ([S.Half], [S.Half, 1]), x))
    q = HolonomicFunction(x/2 - Rational(1, 4) + (-x**2 + x/4)*Dx + x**2*Dx**2 + x**3*Dx**3, x, 1, \
        [1/sqrt(pi), 1/(2*sqrt(pi)), -1/(4*sqrt(pi))])
    assert p == q
    p = from_meijerg(meijerg(([], []), ([0], []), x))
    q = HolonomicFunction(1 + Dx, x, 0, [1])
    assert p == q
    p = from_meijerg(meijerg(([1], []), ([S.Half], [0]), x))
    q = HolonomicFunction((x + S.Half)*Dx + x*Dx**2, x, 1, [sqrt(pi)*erf(1), exp(-1)])
    assert p == q
    p = from_meijerg(meijerg(([0], [1]), ([0], []), 2*x**2))
    q = HolonomicFunction((3*x**2 - 1)*Dx + x**3*Dx**2, x, 1, [-exp(Rational(-1, 2)) + 1, -exp(Rational(-1, 2))])
    assert p == q


def test_to_Sequence():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    n = symbols('n', integer=True)
    _, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    p = HolonomicFunction(x**2*Dx**4 + x + Dx, x).to_sequence()
    q = [(HolonomicSequence(1 + (n + 2)*Sn**2 + (n**4 + 6*n**3 + 11*n**2 + 6*n)*Sn**3), 0, 1)]
    assert p == q
    p = HolonomicFunction(x**2*Dx**4 + x**3 + Dx**2, x).to_sequence()
    q = [(HolonomicSequence(1 + (n**4 + 14*n**3 + 72*n**2 + 163*n + 140)*Sn**5), 0, 0)]
    assert p == q
    p = HolonomicFunction(x**3*Dx**4 + 1 + Dx**2, x).to_sequence()
    q = [(HolonomicSequence(1 + (n**4 - 2*n**3 - n**2 + 2*n)*Sn + (n**2 + 3*n + 2)*Sn**2), 0, 0)]
    assert p == q
    p = HolonomicFunction(3*x**3*Dx**4 + 2*x*Dx + x*Dx**3, x).to_sequence()
    q = [(HolonomicSequence(2*n + (3*n**4 - 6*n**3 - 3*n**2 + 6*n)*Sn + (n**3 + 3*n**2 + 2*n)*Sn**2), 0, 1)]
    assert p == q


def test_to_Sequence_Initial_Coniditons():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    n = symbols('n', integer=True)
    _, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    p = HolonomicFunction(Dx - 1, x, 0, [1]).to_sequence()
    q = [(HolonomicSequence(-1 + (n + 1)*Sn, 1), 0)]
    assert p == q
    p = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).to_sequence()
    q = [(HolonomicSequence(1 + (n**2 + 3*n + 2)*Sn**2, [0, 1]), 0)]
    assert p == q
    p = HolonomicFunction(Dx**2 + 1 + x**3*Dx, x, 0, [2, 3]).to_sequence()
    q = [(HolonomicSequence(n + Sn**2 + (n**2 + 7*n + 12)*Sn**4, [2, 3, -1, Rational(-1, 2), Rational(1, 12)]), 1)]
    assert p == q
    p = HolonomicFunction(x**3*Dx**5 + 1 + Dx, x).to_sequence()
    q = [(HolonomicSequence(1 + (n + 1)*Sn + (n**5 - 5*n**3 + 4*n)*Sn**2), 0, 3)]
    assert p == q
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    p = expr_to_holonomic(log(1+x**2))
    q = [(HolonomicSequence(n**2 + (n**2 + 2*n)*Sn**2, [0, 0, C_2]), 0, 1)]
    assert p.to_sequence() == q
    p = p.diff()
    q = [(HolonomicSequence((n + 2) + (n + 2)*Sn**2, [C_0, 0]), 1, 0)]
    assert p.to_sequence() == q
    p = expr_to_holonomic(erf(x) + x).to_sequence()
    q = [(HolonomicSequence((2*n**2 - 2*n) + (n**3 + 2*n**2 - n - 2)*Sn**2, [0, 1 + 2/sqrt(pi), 0, C_3]), 0, 2)]
    assert p == q

def test_series():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx**2 + 2*x*Dx, x, 0, [0, 1]).series(n=10)
    q = x - x**3/3 + x**5/10 - x**7/42 + x**9/216 + O(x**10)
    assert p == q
    p = HolonomicFunction(Dx - 1, x).composition(x**2, 0, [1])  # e^(x**2)
    q = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0])  # cos(x)
    r = (p * q).series(n=10)  # expansion of cos(x) * exp(x**2)
    s = 1 + x**2/2 + x**4/24 - 31*x**6/720 - 179*x**8/8064 + O(x**10)
    assert r == s
    t = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1])  # log(1 + x)
    r = (p * t + q).series(n=10)
    s = 1 + x - x**2 + 4*x**3/3 - 17*x**4/24 + 31*x**5/30 - 481*x**6/720 +\
     71*x**7/105 - 20159*x**8/40320 + 379*x**9/840 + O(x**10)
    assert r == s
    p = HolonomicFunction((6+6*x-3*x**2) - (10*x-3*x**2-3*x**3)*Dx + \
        (4-6*x**3+2*x**4)*Dx**2, x, 0, [0, 1]).series(n=7)
    q = x + x**3/6 - 3*x**4/16 + x**5/20 - 23*x**6/960 + O(x**7)
    assert p == q
    p = HolonomicFunction((6+6*x-3*x**2) - (10*x-3*x**2-3*x**3)*Dx + \
        (4-6*x**3+2*x**4)*Dx**2, x, 0, [1, 0]).series(n=7)
    q = 1 - 3*x**2/4 - x**3/4 - 5*x**4/32 - 3*x**5/40 - 17*x**6/384 + O(x**7)
    assert p == q
    p = expr_to_holonomic(erf(x) + x).series(n=10)
    C_3 = symbols('C_3')
    q = (erf(x) + x).series(n=10)
    assert p.subs(C_3, -2/(3*sqrt(pi))) == q
    assert expr_to_holonomic(sqrt(x**3 + x)).series(n=10) == sqrt(x**3 + x).series(n=10)
    assert expr_to_holonomic((2*x - 3*x**2)**Rational(1, 3)).series() == ((2*x - 3*x**2)**Rational(1, 3)).series()
    assert  expr_to_holonomic(sqrt(x**2-x)).series() == (sqrt(x**2-x)).series()
    assert expr_to_holonomic(cos(x)**2/x**2, y0={-2: [1, 0, -1]}).series(n=10) == (cos(x)**2/x**2).series(n=10)
    assert expr_to_holonomic(cos(x)**2/x**2, x0=1).series(n=10).together() == (cos(x)**2/x**2).series(n=10, x0=1).together()
    assert expr_to_holonomic(cos(x-1)**2/(x-1)**2, x0=1, y0={-2: [1, 0, -1]}).series(n=10) \
        == (cos(x-1)**2/(x-1)**2).series(x0=1, n=10)

def test_evalf_euler():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')

    # log(1+x)
    p = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1])

    # path taken is a straight line from 0 to 1, on the real axis
    r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    s = '0.699525841805253'  # approx. equal to log(2) i.e. 0.693147180559945
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # path taken is a triangle 0-->1+i-->2
    r = [0.1 + 0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1+0.1*I)
    for i in range(10):
        r.append(r[-1]+0.1-0.1*I)

    # close to the exact solution 1.09861228866811
    # imaginary part also close to zero
    s = '1.07530466271334 - 0.0251200594793912*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # sin(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])
    s = '0.905546532085401 - 6.93889390390723e-18*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # computing sin(pi/2) using this method
    # using a linear path from 0 to pi/2
    r = [0.1]
    for i in range(14):
        r.append(r[-1] + 0.1)
    r.append(pi/2)
    s = '1.08016557252834' # close to 1.0 (exact solution)
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # trying different path, a rectangle (0-->i-->pi/2 + i-->pi/2)
    # computing the same value sin(pi/2) using different path
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(15):
        r.append(r[-1]+0.1)
    r.append(pi/2+I)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    # close to 1.0
    s = '0.976882381836257 - 1.65557671738537e-16*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # cos(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0])
    # compute cos(pi) along 0-->pi
    r = [0.05]
    for i in range(61):
        r.append(r[-1]+0.05)
    r.append(pi)
    # close to -1 (exact answer)
    s = '-1.08140824719196'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # a rectangular path (0 -> i -> 2+i -> 2)
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(20):
        r.append(r[-1]+0.1)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    p = HolonomicFunction(Dx**2 + 1, x, 0, [1,1]).evalf(r, method='Euler')
    s = '0.501421652861245 - 3.88578058618805e-16*I'
    assert sstr(p[-1]) == s

def test_evalf_rk4():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')

    # log(1+x)
    p = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1])

    # path taken is a straight line from 0 to 1, on the real axis
    r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    s = '0.693146363174626'  # approx. equal to log(2) i.e. 0.693147180559945
    assert sstr(p.evalf(r)[-1]) == s

    # path taken is a triangle 0-->1+i-->2
    r = [0.1 + 0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1+0.1*I)
    for i in range(10):
        r.append(r[-1]+0.1-0.1*I)

    # close to the exact solution 1.09861228866811
    # imaginary part also close to zero
    s = '1.098616 + 1.36083e-7*I'
    assert sstr(p.evalf(r)[-1].n(7)) == s

    # sin(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])
    s = '0.90929463522785 + 1.52655665885959e-16*I'
    assert sstr(p.evalf(r)[-1]) == s

    # computing sin(pi/2) using this method
    # using a linear path from 0 to pi/2
    r = [0.1]
    for i in range(14):
        r.append(r[-1] + 0.1)
    r.append(pi/2)
    s = '0.999999895088917' # close to 1.0 (exact solution)
    assert sstr(p.evalf(r)[-1]) == s

    # trying different path, a rectangle (0-->i-->pi/2 + i-->pi/2)
    # computing the same value sin(pi/2) using different path
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(15):
        r.append(r[-1]+0.1)
    r.append(pi/2+I)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    # close to 1.0
    s = '1.00000003415141 + 6.11940487991086e-16*I'
    assert sstr(p.evalf(r)[-1]) == s

    # cos(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0])
    # compute cos(pi) along 0-->pi
    r = [0.05]
    for i in range(61):
        r.append(r[-1]+0.05)
    r.append(pi)
    # close to -1 (exact answer)
    s = '-0.999999993238714'
    assert sstr(p.evalf(r)[-1]) == s

    # a rectangular path (0 -> i -> 2+i -> 2)
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(20):
        r.append(r[-1]+0.1)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    p = HolonomicFunction(Dx**2 + 1, x, 0, [1,1]).evalf(r)
    s = '0.493152791638442 - 1.41553435639707e-15*I'
    assert sstr(p[-1]) == s


def test_expr_to_holonomic():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = expr_to_holonomic((sin(x)/x)**2)
    q = HolonomicFunction(8*x + (4*x**2 + 6)*Dx + 6*x*Dx**2 + x**2*Dx**3, x, 0, \
        [1, 0, Rational(-2, 3)])
    assert p == q
    p = expr_to_holonomic(1/(1+x**2)**2)
    q = HolonomicFunction(4*x + (x**2 + 1)*Dx, x, 0, [1])
    assert p == q
    p = expr_to_holonomic(exp(x)*sin(x)+x*log(1+x))
    q = HolonomicFunction((2*x**3 + 10*x**2 + 20*x + 18) + (-2*x**4 - 10*x**3 - 20*x**2 \
        - 18*x)*Dx + (2*x**5 + 6*x**4 + 7*x**3 + 8*x**2 + 10*x - 4)*Dx**2 + \
        (-2*x**5 - 5*x**4 - 2*x**3 + 2*x**2 - x + 4)*Dx**3 + (x**5 + 2*x**4 - x**3 - \
        7*x**2/2 + x + Rational(5, 2))*Dx**4, x, 0, [0, 1, 4, -1])
    assert p == q
    p = expr_to_holonomic(x*exp(x)+cos(x)+1)
    q = HolonomicFunction((-x - 3)*Dx + (x + 2)*Dx**2 + (-x - 3)*Dx**3 + (x + 2)*Dx**4, x, \
        0, [2, 1, 1, 3])
    assert p == q
    assert (x*exp(x)+cos(x)+1).series(n=10) == p.series(n=10)
    p = expr_to_holonomic(log(1 + x)**2 + 1)
    q = HolonomicFunction(Dx + (3*x + 3)*Dx**2 + (x**2 + 2*x + 1)*Dx**3, x, 0, [1, 0, 2])
    assert p == q
    p = expr_to_holonomic(erf(x)**2 + x)
    q = HolonomicFunction((8*x**4 - 2*x**2 + 2)*Dx**2 + (6*x**3 - x/2)*Dx**3 + \
        (x**2+ Rational(1, 4))*Dx**4, x, 0, [0, 1, 8/pi, 0])
    assert p == q
    p = expr_to_holonomic(cosh(x)*x)
    q = HolonomicFunction((-x**2 + 2) -2*x*Dx + x**2*Dx**2, x, 0, [0, 1])
    assert p == q
    p = expr_to_holonomic(besselj(2, x))
    q = HolonomicFunction((x**2 - 4) + x*Dx + x**2*Dx**2, x, 0, [0, 0])
    assert p == q
    p = expr_to_holonomic(besselj(0, x) + exp(x))
    q = HolonomicFunction((-x**2 - x/2 + S.Half) + (x**2 - x/2 - Rational(3, 2))*Dx + (-x**2 + x/2 + 1)*Dx**2 +\
        (x**2 + x/2)*Dx**3, x, 0, [2, 1, S.Half])
    assert p == q
    p = expr_to_holonomic(sin(x)**2/x)
    q = HolonomicFunction(4 + 4*x*Dx + 3*Dx**2 + x*Dx**3, x, 0, [0, 1, 0])
    assert p == q
    p = expr_to_holonomic(sin(x)**2/x, x0=2)
    q = HolonomicFunction((4) + (4*x)*Dx + (3)*Dx**2 + (x)*Dx**3, x, 2, [sin(2)**2/2,
        sin(2)*cos(2) - sin(2)**2/4, -3*sin(2)**2/4 + cos(2)**2 - sin(2)*cos(2)])
    assert p == q
    p = expr_to_holonomic(log(x)/2 - Ci(2*x)/2 + Ci(2)/2)
    q = HolonomicFunction(4*Dx + 4*x*Dx**2 + 3*Dx**3 + x*Dx**4, x, 0, \
        [-log(2)/2 - EulerGamma/2 + Ci(2)/2, 0, 1, 0])
    assert p == q
    p = p.to_expr()
    q = log(x)/2 - Ci(2*x)/2 + Ci(2)/2
    assert p == q
    p = expr_to_holonomic(x**S.Half, x0=1)
    q = HolonomicFunction(x*Dx - S.Half, x, 1, [1])
    assert p == q
    p = expr_to_holonomic(sqrt(1 + x**2))
    q = HolonomicFunction((-x) + (x**2 + 1)*Dx, x, 0, [1])
    assert p == q
    assert (expr_to_holonomic(sqrt(x) + sqrt(2*x)).to_expr()-\
        (sqrt(x) + sqrt(2*x))).simplify() == 0
    assert expr_to_holonomic(3*x+2*sqrt(x)).to_expr() == 3*x+2*sqrt(x)
    p = expr_to_holonomic((x**4+x**3+5*x**2+3*x+2)/x**2, lenics=3)
    q = HolonomicFunction((-2*x**4 - x**3 + 3*x + 4) + (x**5 + x**4 + 5*x**3 + 3*x**2 + \
        2*x)*Dx, x, 0, {-2: [2, 3, 5]})
    assert p == q
    p = expr_to_holonomic(1/(x-1)**2, lenics=3, x0=1)
    q = HolonomicFunction((2) + (x - 1)*Dx, x, 1, {-2: [1, 0, 0]})
    assert p == q
    a = symbols("a")
    p = expr_to_holonomic(sqrt(a*x), x=x)
    assert p.to_expr() == sqrt(a)*sqrt(x)

def test_to_hyper():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx - 2, x, 0, [3]).to_hyper()
    q = 3 * hyper([], [], 2*x)
    assert p == q
    p = hyperexpand(HolonomicFunction((1 + x) * Dx - 3, x, 0, [2]).to_hyper()).expand()
    q = 2*x**3 + 6*x**2 + 6*x + 2
    assert p == q
    p = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1]).to_hyper()
    q = -x**2*hyper((2, 2, 1), (3, 2), -x)/2 + x
    assert p == q
    p = HolonomicFunction(2*x*Dx + Dx**2, x, 0, [0, 2/sqrt(pi)]).to_hyper()
    q = 2*x*hyper((S.Half,), (Rational(3, 2),), -x**2)/sqrt(pi)
    assert p == q
    p = hyperexpand(HolonomicFunction(2*x*Dx + Dx**2, x, 0, [1, -2/sqrt(pi)]).to_hyper())
    q = erfc(x)
    assert p.rewrite(erfc) == q
    p =  hyperexpand(HolonomicFunction((x**2 - 1) + x*Dx + x**2*Dx**2,
        x, 0, [0, S.Half]).to_hyper())
    q = besselj(1, x)
    assert p == q
    p = hyperexpand(HolonomicFunction(x*Dx**2 + Dx + x, x, 0, [1, 0]).to_hyper())
    q = besselj(0, x)
    assert p == q

def test_to_expr():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx - 1, x, 0, [1]).to_expr()
    q = exp(x)
    assert p == q
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0]).to_expr()
    q = cos(x)
    assert p == q
    p = HolonomicFunction(Dx**2 - 1, x, 0, [1, 0]).to_expr()
    q = cosh(x)
    assert p == q
    p = HolonomicFunction(2 + (4*x - 1)*Dx + \
        (x**2 - x)*Dx**2, x, 0, [1, 2]).to_expr().expand()
    q = 1/(x**2 - 2*x + 1)
    assert p == q
    p = expr_to_holonomic(sin(x)**2/x).integrate((x, 0, x)).to_expr()
    q = (sin(x)**2/x).integrate((x, 0, x))
    assert p == q
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    p = expr_to_holonomic(log(1+x**2)).to_expr()
    q = C_2*log(x**2 + 1)
    assert p == q
    p = expr_to_holonomic(log(1+x**2)).diff().to_expr()
    q = C_0*x/(x**2 + 1)
    assert p == q
    p = expr_to_holonomic(erf(x) + x).to_expr()
    q = 3*C_3*x - 3*sqrt(pi)*C_3*erf(x)/2 + x + 2*x/sqrt(pi)
    assert p == q
    p = expr_to_holonomic(sqrt(x), x0=1).to_expr()
    assert p == sqrt(x)
    assert expr_to_holonomic(sqrt(x)).to_expr() == sqrt(x)
    p = expr_to_holonomic(sqrt(1 + x**2)).to_expr()
    assert p == sqrt(1+x**2)
    p = expr_to_holonomic((2*x**2 + 1)**Rational(2, 3)).to_expr()
    assert p == (2*x**2 + 1)**Rational(2, 3)
    p = expr_to_holonomic(sqrt(-x**2+2*x)).to_expr()
    assert p == sqrt(x)*sqrt(-x + 2)
    p = expr_to_holonomic((-2*x**3+7*x)**Rational(2, 3)).to_expr()
    q = x**Rational(2, 3)*(-2*x**2 + 7)**Rational(2, 3)
    assert p == q
    p = from_hyper(hyper((-2, -3), (S.Half, ), x))
    s = hyperexpand(hyper((-2, -3), (S.Half, ), x))
    D_0 = Symbol('D_0')
    C_0 = Symbol('C_0')
    assert (p.to_expr().subs({C_0:1, D_0:0}) - s).simplify() == 0
    p.y0 = {0: [1], S.Half: [0]}
    assert p.to_expr() == s
    assert expr_to_holonomic(x**5).to_expr() == x**5
    assert expr_to_holonomic(2*x**3-3*x**2).to_expr().expand() == \
        2*x**3-3*x**2
    a = symbols("a")
    p = (expr_to_holonomic(1.4*x)*expr_to_holonomic(a*x, x)).to_expr()
    q = 1.4*a*x**2
    assert p == q
    p = (expr_to_holonomic(1.4*x)+expr_to_holonomic(a*x, x)).to_expr()
    q = x*(a + 1.4)
    assert p == q
    p = (expr_to_holonomic(1.4*x)+expr_to_holonomic(x)).to_expr()
    assert p == 2.4*x


def test_integrate():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = expr_to_holonomic(sin(x)**2/x, x0=1).integrate((x, 2, 3))
    q = '0.166270406994788'
    assert sstr(p) == q
    p = expr_to_holonomic(sin(x)).integrate((x, 0, x)).to_expr()
    q = 1 - cos(x)
    assert p == q
    p = expr_to_holonomic(sin(x)).integrate((x, 0, 3))
    q = 1 - cos(3)
    assert p == q
    p = expr_to_holonomic(sin(x)/x, x0=1).integrate((x, 1, 2))
    q = '0.659329913368450'
    assert sstr(p) == q
    p = expr_to_holonomic(sin(x)**2/x, x0=1).integrate((x, 1, 0))
    q = '-0.423690480850035'
    assert sstr(p) == q
    p = expr_to_holonomic(sin(x)/x)
    assert p.integrate(x).to_expr() == Si(x)
    assert p.integrate((x, 0, 2)) == Si(2)
    p = expr_to_holonomic(sin(x)**2/x)
    q = p.to_expr()
    assert p.integrate(x).to_expr() == q.integrate((x, 0, x))
    assert p.integrate((x, 0, 1)) == q.integrate((x, 0, 1))
    assert expr_to_holonomic(1/x, x0=1).integrate(x).to_expr() == log(x)
    p = expr_to_holonomic((x + 1)**3*exp(-x), x0=-1).integrate(x).to_expr()
    q = (-x**3 - 6*x**2 - 15*x + 6*exp(x + 1) - 16)*exp(-x)
    assert p == q
    p = expr_to_holonomic(cos(x)**2/x**2, y0={-2: [1, 0, -1]}).integrate(x).to_expr()
    q = -Si(2*x) - cos(x)**2/x
    assert p == q
    p = expr_to_holonomic(sqrt(x**2+x)).integrate(x).to_expr()
    q = (x**Rational(3, 2)*(2*x**2 + 3*x + 1) - x*sqrt(x + 1)*asinh(sqrt(x)))/(4*x*sqrt(x + 1))
    assert p == q
    p = expr_to_holonomic(sqrt(x**2+1)).integrate(x).to_expr()
    q = (sqrt(x**2+1)).integrate(x)
    assert (p-q).simplify() == 0
    p = expr_to_holonomic(1/x**2, y0={-2:[1, 0, 0]})
    r = expr_to_holonomic(1/x**2, lenics=3)
    assert p == r
    q = expr_to_holonomic(cos(x)**2)
    assert (r*q).integrate(x).to_expr() == -Si(2*x) - cos(x)**2/x


def test_diff():
    x, y = symbols('x, y')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(x*Dx**2 + 1, x, 0, [0, 1])
    assert p.diff().to_expr() == p.to_expr().diff().simplify()
    p = HolonomicFunction(Dx**2 - 1, x, 0, [1, 0])
    assert p.diff(x, 2).to_expr() == p.to_expr()
    p = expr_to_holonomic(Si(x))
    assert p.diff().to_expr() == sin(x)/x
    assert p.diff(y) == 0
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    q = Si(x)
    assert p.diff(x).to_expr() == q.diff()
    assert p.diff(x, 2).to_expr().subs(C_0, Rational(-1, 3)).cancel() == q.diff(x, 2).cancel()
    assert p.diff(x, 3).series().subs({C_3: Rational(-1, 3), C_0: 0}) == q.diff(x, 3).series()


def test_extended_domain_in_expr_to_holonomic():
    x = symbols('x')
    p = expr_to_holonomic(1.2*cos(3.1*x))
    assert p.to_expr() == 1.2*cos(3.1*x)
    assert sstr(p.integrate(x).to_expr()) == '0.387096774193548*sin(3.1*x)'
    _, Dx = DifferentialOperators(RR.old_poly_ring(x), 'Dx')
    p = expr_to_holonomic(1.1329138213*x)
    q = HolonomicFunction((-1.1329138213) + (1.1329138213*x)*Dx, x, 0, {1: [1.1329138213]})
    assert p == q
    assert p.to_expr() == 1.1329138213*x
    assert sstr(p.integrate((x, 1, 2))) == sstr((1.1329138213*x).integrate((x, 1, 2)))
    y, z = symbols('y, z')
    p = expr_to_holonomic(sin(x*y*z), x=x)
    assert p.to_expr() == sin(x*y*z)
    assert p.integrate(x).to_expr() == (-cos(x*y*z) + 1)/(y*z)
    p = expr_to_holonomic(sin(x*y + z), x=x).integrate(x).to_expr()
    q = (cos(z) - cos(x*y + z))/y
    assert p == q
    a = symbols('a')
    p = expr_to_holonomic(a*x, x)
    assert p.to_expr() == a*x
    assert p.integrate(x).to_expr() == a*x**2/2
    D_2, C_1 = symbols("D_2, C_1")
    p = expr_to_holonomic(x) + expr_to_holonomic(1.2*cos(x))
    p = p.to_expr().subs(D_2, 0)
    assert p - x - 1.2*cos(1.0*x) == 0
    p = expr_to_holonomic(x) * expr_to_holonomic(1.2*cos(x))
    p = p.to_expr().subs(C_1, 0)
    assert p - 1.2*x*cos(1.0*x) == 0


def test_to_meijerg():
    x = symbols('x')
    assert hyperexpand(expr_to_holonomic(sin(x)).to_meijerg()) == sin(x)
    assert hyperexpand(expr_to_holonomic(cos(x)).to_meijerg()) == cos(x)
    assert hyperexpand(expr_to_holonomic(exp(x)).to_meijerg()) == exp(x)
    assert hyperexpand(expr_to_holonomic(log(x)).to_meijerg()).simplify() == log(x)
    assert expr_to_holonomic(4*x**2/3 + 7).to_meijerg() == 4*x**2/3 + 7
    assert hyperexpand(expr_to_holonomic(besselj(2, x), lenics=3).to_meijerg()) == besselj(2, x)
    p = hyper((Rational(-1, 2), -3), (), x)
    assert from_hyper(p).to_meijerg() == hyperexpand(p)
    p = hyper((S.One, S(3)), (S(2), ), x)
    assert (hyperexpand(from_hyper(p).to_meijerg()) - hyperexpand(p)).expand() == 0
    p = from_hyper(hyper((-2, -3), (S.Half, ), x))
    s = hyperexpand(hyper((-2, -3), (S.Half, ), x))
    C_0 = Symbol('C_0')
    C_1 = Symbol('C_1')
    D_0 = Symbol('D_0')
    assert (hyperexpand(p.to_meijerg()).subs({C_0:1, D_0:0}) - s).simplify() == 0
    p.y0 = {0: [1], S.Half: [0]}
    assert (hyperexpand(p.to_meijerg()) - s).simplify() == 0
    p = expr_to_holonomic(besselj(S.Half, x), initcond=False)
    assert (p.to_expr() - (D_0*sin(x) + C_0*cos(x) + C_1*sin(x))/sqrt(x)).simplify() == 0
    p = expr_to_holonomic(besselj(S.Half, x), y0={Rational(-1, 2): [sqrt(2)/sqrt(pi), sqrt(2)/sqrt(pi)]})
    assert (p.to_expr() - besselj(S.Half, x) - besselj(Rational(-1, 2), x)).simplify() == 0


def test_gaussian():
    mu, x = symbols("mu x")
    sd = symbols("sd", positive=True)
    Q = QQ[mu, sd].get_field()
    e = sqrt(2)*exp(-(-mu + x)**2/(2*sd**2))/(2*sqrt(pi)*sd)
    h1 = expr_to_holonomic(e, x, domain=Q)

    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h2 = HolonomicFunction((-mu/sd**2 + x/sd**2) + (1)*Dx, x)

    assert h1 == h2


def test_beta():
    a, b, x = symbols("a b x", positive=True)
    e = x**(a - 1)*(-x + 1)**(b - 1)/beta(a, b)
    Q = QQ[a, b].get_field()
    h1 = expr_to_holonomic(e, x, domain=Q)

    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h2 = HolonomicFunction((a + x*(-a - b + 2) - 1) + (x**2 - x)*Dx, x)

    assert h1 == h2


def test_gamma():
    a, b, x = symbols("a b x", positive=True)
    e = b**(-a)*x**(a - 1)*exp(-x/b)/gamma(a)
    Q = QQ[a, b].get_field()
    h1 = expr_to_holonomic(e, x, domain=Q)

    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h2 = HolonomicFunction((-a + 1 + x/b) + (x)*Dx, x)

    assert h1 == h2


def test_symbolic_power():
    x, n = symbols("x n")
    Q = QQ[n].get_field()
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h1 = HolonomicFunction((-1) + (x)*Dx, x) ** -n
    h2 = HolonomicFunction((n) + (x)*Dx, x)

    assert h1 == h2


def test_negative_power():
    x = symbols("x")
    _, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    h1 = HolonomicFunction((-1) + (x)*Dx, x) ** -2
    h2 = HolonomicFunction((2) + (x)*Dx, x)

    assert h1 == h2


def test_expr_in_power():
    x, n = symbols("x n")
    Q = QQ[n].get_field()
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h1 = HolonomicFunction((-1) + (x)*Dx, x) ** (n - 3)
    h2 = HolonomicFunction((-n + 3) + (x)*Dx, x)

    assert h1 == h2


def test_DifferentialOperatorEqPoly():
    x = symbols('x', integer=True)
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    do = DifferentialOperator([x**2, R.base.zero, R.base.zero], R)
    do2 = DifferentialOperator([x**2, 1, x], R)
    assert not do == do2

    # polynomial comparison issue, see https://github.com/sympy/sympy/pull/15799
    # should work once that is solved
    # p = do.listofpoly[0]
    # assert do == p

    p2 = do2.listofpoly[0]
    assert not do2 == p2


def test_DifferentialOperatorPow():
    x = symbols('x', integer=True)
    R, _ = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    do = DifferentialOperator([x**2, R.base.zero, R.base.zero], R)
    a = DifferentialOperator([R.base.one], R)
    for n in range(10):
        assert a == do**n
        a *= do
