from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, pi, Rational, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan
from sympy.matrices.dense import eye
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.simplify.simplify import simplify
from sympy.core.containers import Tuple
from sympy.matrices import ImmutableMatrix, Matrix, ShapeError
from sympy.physics.control import (TransferFunction, Series, Parallel,
    Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback,
    StateSpace, gbt, bilinear, forward_diff, backward_diff, phase_margin, gain_margin)
from sympy.testing.pytest import raises

a, x, b, c, s, g, d, p, k, tau, zeta, wn, T = symbols('a, x, b, c, s, g, d, p, k,\
    tau, zeta, wn, T')
a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = symbols('a0:4,\
    b0:4, c0:4, d0:4')
TF1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
TF2 = TransferFunction(k, 1, s)
TF3 = TransferFunction(a2*p - s, a2*s + p, s)


def test_TransferFunction_construction():
    tf = TransferFunction(s + 1, s**2 + s + 1, s)
    assert tf.num == (s + 1)
    assert tf.den == (s**2 + s + 1)
    assert tf.args == (s + 1, s**2 + s + 1, s)

    tf1 = TransferFunction(s + 4, s - 5, s)
    assert tf1.num == (s + 4)
    assert tf1.den == (s - 5)
    assert tf1.args == (s + 4, s - 5, s)

    # using different polynomial variables.
    tf2 = TransferFunction(p + 3, p**2 - 9, p)
    assert tf2.num == (p + 3)
    assert tf2.den == (p**2 - 9)
    assert tf2.args == (p + 3, p**2 - 9, p)

    tf3 = TransferFunction(p**3 + 5*p**2 + 4, p**4 + 3*p + 1, p)
    assert tf3.args == (p**3 + 5*p**2 + 4, p**4 + 3*p + 1, p)

    # no pole-zero cancellation on its own.
    tf4 = TransferFunction((s + 3)*(s - 1), (s - 1)*(s + 5), s)
    assert tf4.den == (s - 1)*(s + 5)
    assert tf4.args == ((s + 3)*(s - 1), (s - 1)*(s + 5), s)

    tf4_ = TransferFunction(p + 2, p + 2, p)
    assert tf4_.args == (p + 2, p + 2, p)

    tf5 = TransferFunction(s - 1, 4 - p, s)
    assert tf5.args == (s - 1, 4 - p, s)

    tf5_ = TransferFunction(s - 1, s - 1, s)
    assert tf5_.args == (s - 1, s - 1, s)

    tf6 = TransferFunction(5, 6, s)
    assert tf6.num == 5
    assert tf6.den == 6
    assert tf6.args == (5, 6, s)

    tf6_ = TransferFunction(1/2, 4, s)
    assert tf6_.num == 0.5
    assert tf6_.den == 4
    assert tf6_.args == (0.500000000000000, 4, s)

    tf7 = TransferFunction(3*s**2 + 2*p + 4*s, 8*p**2 + 7*s, s)
    tf8 = TransferFunction(3*s**2 + 2*p + 4*s, 8*p**2 + 7*s, p)
    assert not tf7 == tf8

    tf7_ = TransferFunction(a0*s + a1*s**2 + a2*s**3, b0*p - b1*s, s)
    tf8_ = TransferFunction(a0*s + a1*s**2 + a2*s**3, b0*p - b1*s, s)
    assert tf7_ == tf8_
    assert -(-tf7_) == tf7_ == -(-(-(-tf7_)))

    tf9 = TransferFunction(a*s**3 + b*s**2 + g*s + d, d*p + g*p**2 + g*s, s)
    assert tf9.args == (a*s**3 + b*s**2 + d + g*s, d*p + g*p**2 + g*s, s)

    tf10 = TransferFunction(p**3 + d, g*s**2 + d*s + a, p)
    tf10_ = TransferFunction(p**3 + d, g*s**2 + d*s + a, p)
    assert tf10.args == (d + p**3, a + d*s + g*s**2, p)
    assert tf10_ == tf10

    tf11 = TransferFunction(a1*s + a0, b2*s**2 + b1*s + b0, s)
    assert tf11.num == (a0 + a1*s)
    assert tf11.den == (b0 + b1*s + b2*s**2)
    assert tf11.args == (a0 + a1*s, b0 + b1*s + b2*s**2, s)

    # when just the numerator is 0, leave the denominator alone.
    tf12 = TransferFunction(0, p**2 - p + 1, p)
    assert tf12.args == (0, p**2 - p + 1, p)

    tf13 = TransferFunction(0, 1, s)
    assert tf13.args == (0, 1, s)

    # float exponents
    tf14 = TransferFunction(a0*s**0.5 + a2*s**0.6 - a1, a1*p**(-8.7), s)
    assert tf14.args == (a0*s**0.5 - a1 + a2*s**0.6, a1*p**(-8.7), s)

    tf15 = TransferFunction(a2**2*p**(1/4) + a1*s**(-4/5), a0*s - p, p)
    assert tf15.args == (a1*s**(-0.8) + a2**2*p**0.25, a0*s - p, p)

    omega_o, k_p, k_o, k_i = symbols('omega_o, k_p, k_o, k_i')
    tf18 = TransferFunction((k_p + k_o*s + k_i/s), s**2 + 2*omega_o*s + omega_o**2, s)
    assert tf18.num == k_i/s + k_o*s + k_p
    assert tf18.args == (k_i/s + k_o*s + k_p, omega_o**2 + 2*omega_o*s + s**2, s)

    # ValueError when denominator is zero.
    raises(ValueError, lambda: TransferFunction(4, 0, s))
    raises(ValueError, lambda: TransferFunction(s, 0, s))
    raises(ValueError, lambda: TransferFunction(0, 0, s))

    raises(TypeError, lambda: TransferFunction(Matrix([1, 2, 3]), s, s))

    raises(TypeError, lambda: TransferFunction(s**2 + 2*s - 1, s + 3, 3))
    raises(TypeError, lambda: TransferFunction(p + 1, 5 - p, 4))
    raises(TypeError, lambda: TransferFunction(3, 4, 8))


def test_TransferFunction_functions():
    # classmethod from_rational_expression
    expr_1 = Mul(0, Pow(s, -1, evaluate=False), evaluate=False)
    expr_2 = s/0
    expr_3 = (p*s**2 + 5*s)/(s + 1)**3
    expr_4 = 6
    expr_5 = ((2 + 3*s)*(5 + 2*s))/((9 + 3*s)*(5 + 2*s**2))
    expr_6 = (9*s**4 + 4*s**2 + 8)/((s + 1)*(s + 9))
    tf = TransferFunction(s + 1, s**2 + 2, s)
    delay = exp(-s/tau)
    expr_7 = delay*tf.to_expr()
    H1 = TransferFunction.from_rational_expression(expr_7, s)
    H2 = TransferFunction(s + 1, (s**2 + 2)*exp(s/tau), s)
    expr_8 = Add(2,  3*s/(s**2 + 1), evaluate=False)

    assert TransferFunction.from_rational_expression(expr_1) == TransferFunction(0, s, s)
    raises(ZeroDivisionError, lambda: TransferFunction.from_rational_expression(expr_2))
    raises(ValueError, lambda: TransferFunction.from_rational_expression(expr_3))
    assert TransferFunction.from_rational_expression(expr_3, s) == TransferFunction((p*s**2 + 5*s), (s + 1)**3, s)
    assert TransferFunction.from_rational_expression(expr_3, p) == TransferFunction((p*s**2 + 5*s), (s + 1)**3, p)
    raises(ValueError, lambda: TransferFunction.from_rational_expression(expr_4))
    assert TransferFunction.from_rational_expression(expr_4, s) == TransferFunction(6, 1, s)
    assert TransferFunction.from_rational_expression(expr_5, s) == \
        TransferFunction((2 + 3*s)*(5 + 2*s), (9 + 3*s)*(5 + 2*s**2), s)
    assert TransferFunction.from_rational_expression(expr_6, s) == \
        TransferFunction((9*s**4 + 4*s**2 + 8), (s + 1)*(s + 9), s)
    assert H1 == H2
    assert TransferFunction.from_rational_expression(expr_8, s) == \
        TransferFunction(2*s**2 + 3*s + 2, s**2 + 1, s)

    # classmethod from_coeff_lists
    tf1 = TransferFunction.from_coeff_lists([1, 2], [3, 4, 5], s)
    num2 = [p**2, 2*p]
    den2 = [p**3, p + 1, 4]
    tf2 = TransferFunction.from_coeff_lists(num2, den2, s)
    num3 = [1, 2, 3]
    den3 = [0, 0]

    assert tf1 == TransferFunction(s + 2, 3*s**2 + 4*s + 5, s)
    assert tf2 == TransferFunction(p**2*s + 2*p, p**3*s**2 + s*(p + 1) + 4, s)
    raises(ZeroDivisionError, lambda: TransferFunction.from_coeff_lists(num3, den3, s))

    # classmethod from_zpk
    zeros = [4]
    poles = [-1+2j, -1-2j]
    gain = 3
    tf1 = TransferFunction.from_zpk(zeros, poles, gain, s)

    assert tf1 == TransferFunction(3*s - 12, (s + 1.0 - 2.0*I)*(s + 1.0 + 2.0*I), s)

    # explicitly cancel poles and zeros.
    tf0 = TransferFunction(s**5 + s**3 + s, s - s**2, s)
    a = TransferFunction(-(s**4 + s**2 + 1), s - 1, s)
    assert tf0.simplify() == simplify(tf0) == a

    tf1 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
    b = TransferFunction(p + 3, p + 5, p)
    assert tf1.simplify() == simplify(tf1) == b

    # expand the numerator and the denominator.
    G1 = TransferFunction((1 - s)**2, (s**2 + 1)**2, s)
    G2 = TransferFunction(1, -3, p)
    c = (a2*s**p + a1*s**s + a0*p**p)*(p**s + s**p)
    d = (b0*s**s + b1*p**s)*(b2*s*p + p**p)
    e = a0*p**p*p**s + a0*p**p*s**p + a1*p**s*s**s + a1*s**p*s**s + a2*p**s*s**p + a2*s**(2*p)
    f = b0*b2*p*s*s**s + b0*p**p*s**s + b1*b2*p*p**s*s + b1*p**p*p**s
    g = a1*a2*s*s**p + a1*p*s + a2*b1*p*s*s**p + b1*p**2*s
    G3 = TransferFunction(c, d, s)
    G4 = TransferFunction(a0*s**s - b0*p**p, (a1*s + b1*s*p)*(a2*s**p + p), p)

    assert G1.expand() == TransferFunction(s**2 - 2*s + 1, s**4 + 2*s**2 + 1, s)
    assert tf1.expand() == TransferFunction(p**2 + 2*p - 3, p**2 + 4*p - 5, p)
    assert G2.expand() == G2
    assert G3.expand() == TransferFunction(e, f, s)
    assert G4.expand() == TransferFunction(a0*s**s - b0*p**p, g, p)

    # purely symbolic polynomials.
    p1 = a1*s + a0
    p2 = b2*s**2 + b1*s + b0
    SP1 = TransferFunction(p1, p2, s)
    expect1 = TransferFunction(2.0*s + 1.0, 5.0*s**2 + 4.0*s + 3.0, s)
    expect1_ = TransferFunction(2*s + 1, 5*s**2 + 4*s + 3, s)
    assert SP1.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}) == expect1_
    assert SP1.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}).evalf() == expect1
    assert expect1_.evalf() == expect1

    c1, d0, d1, d2 = symbols('c1, d0:3')
    p3, p4 = c1*p, d2*p**3 + d1*p**2 - d0
    SP2 = TransferFunction(p3, p4, p)
    expect2 = TransferFunction(2.0*p, 5.0*p**3 + 2.0*p**2 - 3.0, p)
    expect2_ = TransferFunction(2*p, 5*p**3 + 2*p**2 - 3, p)
    assert SP2.subs({c1: 2, d0: 3, d1: 2, d2: 5}) == expect2_
    assert SP2.subs({c1: 2, d0: 3, d1: 2, d2: 5}).evalf() == expect2
    assert expect2_.evalf() == expect2

    SP3 = TransferFunction(a0*p**3 + a1*s**2 - b0*s + b1, a1*s + p, s)
    expect3 = TransferFunction(2.0*p**3 + 4.0*s**2 - s + 5.0, p + 4.0*s, s)
    expect3_ = TransferFunction(2*p**3 + 4*s**2 - s + 5, p + 4*s, s)
    assert SP3.subs({a0: 2, a1: 4, b0: 1, b1: 5}) == expect3_
    assert SP3.subs({a0: 2, a1: 4, b0: 1, b1: 5}).evalf() == expect3
    assert expect3_.evalf() == expect3

    SP4 = TransferFunction(s - a1*p**3, a0*s + p, p)
    expect4 = TransferFunction(7.0*p**3 + s, p - s, p)
    expect4_ = TransferFunction(7*p**3 + s, p - s, p)
    assert SP4.subs({a0: -1, a1: -7}) == expect4_
    assert SP4.subs({a0: -1, a1: -7}).evalf() == expect4
    assert expect4_.evalf() == expect4

    # evaluate the transfer function at particular frequencies.
    assert tf1.eval_frequency(wn) == wn**2/(wn**2 + 4*wn - 5) + 2*wn/(wn**2 + 4*wn - 5) - 3/(wn**2 + 4*wn - 5)
    assert G1.eval_frequency(1 + I) == S(3)/25 + S(4)*I/25
    assert G4.eval_frequency(S(5)/3) == \
        a0*s**s/(a1*a2*s**(S(8)/3) + S(5)*a1*s/3 + 5*a2*b1*s**(S(8)/3)/3 + S(25)*b1*s/9) - 5*3**(S(1)/3)*5**(S(2)/3)*b0/(9*a1*a2*s**(S(8)/3) + 15*a1*s + 15*a2*b1*s**(S(8)/3) + 25*b1*s)

    # Low-frequency (or DC) gain.
    assert tf0.dc_gain() == 1
    assert tf1.dc_gain() == Rational(3, 5)
    assert SP2.dc_gain() == 0
    assert expect4.dc_gain() == -1
    assert expect2_.dc_gain() == 0
    assert TransferFunction(1, s, s).dc_gain() == oo

    # Poles of a transfer function.
    tf_ = TransferFunction(x**3 - k, k, x)
    _tf = TransferFunction(k, x**4 - k, x)
    TF_ = TransferFunction(x**2, x**10 + x + x**2, x)
    _TF = TransferFunction(x**10 + x + x**2, x**2, x)
    assert G1.poles() == [I, I, -I, -I]
    assert G2.poles() == []
    assert tf1.poles() == [-5, 1]
    assert expect4_.poles() == [s]
    assert SP4.poles() == [-a0*s]
    assert expect3.poles() == [-0.25*p]
    assert str(expect2.poles()) == str([0.729001428685125, -0.564500714342563 - 0.710198984796332*I, -0.564500714342563 + 0.710198984796332*I])
    assert str(expect1.poles()) == str([-0.4 - 0.66332495807108*I, -0.4 + 0.66332495807108*I])
    assert _tf.poles() == [k**(Rational(1, 4)), -k**(Rational(1, 4)), I*k**(Rational(1, 4)), -I*k**(Rational(1, 4))]
    assert TF_.poles() == [CRootOf(x**9 + x + 1, 0), 0, CRootOf(x**9 + x + 1, 1), CRootOf(x**9 + x + 1, 2),
        CRootOf(x**9 + x + 1, 3), CRootOf(x**9 + x + 1, 4), CRootOf(x**9 + x + 1, 5), CRootOf(x**9 + x + 1, 6),
        CRootOf(x**9 + x + 1, 7), CRootOf(x**9 + x + 1, 8)]
    raises(NotImplementedError, lambda: TransferFunction(x**2, a0*x**10 + x + x**2, x).poles())

    # Stability of a transfer function.
    q, r = symbols('q, r', negative=True)
    t = symbols('t', positive=True)
    TF_ = TransferFunction(s**2 + a0 - a1*p, q*s - r, s)
    stable_tf = TransferFunction(s**2 + a0 - a1*p, q*s - 1, s)
    stable_tf_ = TransferFunction(s**2 + a0 - a1*p, q*s - t, s)

    assert G1.is_stable() is False
    assert G2.is_stable() is True
    assert tf1.is_stable() is False   # as one pole is +ve, and the other is -ve.
    assert expect2.is_stable() is False
    assert expect1.is_stable() is True
    assert stable_tf.is_stable() is True
    assert stable_tf_.is_stable() is True
    assert TF_.is_stable() is False
    assert expect4_.is_stable() is None   # no assumption provided for the only pole 's'.
    assert SP4.is_stable() is None

    # Zeros of a transfer function.
    assert G1.zeros() == [1, 1]
    assert G2.zeros() == []
    assert tf1.zeros() == [-3, 1]
    assert expect4_.zeros() == [7**(Rational(2, 3))*(-s)**(Rational(1, 3))/7, -7**(Rational(2, 3))*(-s)**(Rational(1, 3))/14 -
        sqrt(3)*7**(Rational(2, 3))*I*(-s)**(Rational(1, 3))/14, -7**(Rational(2, 3))*(-s)**(Rational(1, 3))/14 + sqrt(3)*7**(Rational(2, 3))*I*(-s)**(Rational(1, 3))/14]
    assert SP4.zeros() == [(s/a1)**(Rational(1, 3)), -(s/a1)**(Rational(1, 3))/2 - sqrt(3)*I*(s/a1)**(Rational(1, 3))/2,
        -(s/a1)**(Rational(1, 3))/2 + sqrt(3)*I*(s/a1)**(Rational(1, 3))/2]
    assert str(expect3.zeros()) == str([0.125 - 1.11102430216445*sqrt(-0.405063291139241*p**3 - 1.0),
        1.11102430216445*sqrt(-0.405063291139241*p**3 - 1.0) + 0.125])
    assert tf_.zeros() == [k**(Rational(1, 3)), -k**(Rational(1, 3))/2 - sqrt(3)*I*k**(Rational(1, 3))/2,
        -k**(Rational(1, 3))/2 + sqrt(3)*I*k**(Rational(1, 3))/2]
    assert _TF.zeros() == [CRootOf(x**9 + x + 1, 0), 0, CRootOf(x**9 + x + 1, 1), CRootOf(x**9 + x + 1, 2),
        CRootOf(x**9 + x + 1, 3), CRootOf(x**9 + x + 1, 4), CRootOf(x**9 + x + 1, 5), CRootOf(x**9 + x + 1, 6),
        CRootOf(x**9 + x + 1, 7), CRootOf(x**9 + x + 1, 8)]
    raises(NotImplementedError, lambda: TransferFunction(a0*x**10 + x + x**2, x**2, x).zeros())

    # negation of TF.
    tf2 = TransferFunction(s + 3, s**2 - s**3 + 9, s)
    tf3 = TransferFunction(-3*p + 3, 1 - p, p)
    assert -tf2 == TransferFunction(-s - 3, s**2 - s**3 + 9, s)
    assert -tf3 == TransferFunction(3*p - 3, 1 - p, p)

    # taking power of a TF.
    tf4 = TransferFunction(p + 4, p - 3, p)
    tf5 = TransferFunction(s**2 + 1, 1 - s, s)
    expect2 = TransferFunction((s**2 + 1)**3, (1 - s)**3, s)
    expect1 = TransferFunction((p + 4)**2, (p - 3)**2, p)
    assert (tf4*tf4).doit() == tf4**2 == pow(tf4, 2) == expect1
    assert (tf5*tf5*tf5).doit() == tf5**3 == pow(tf5, 3) == expect2
    assert tf5**0 == pow(tf5, 0) == TransferFunction(1, 1, s)
    assert Series(tf4).doit()**-1 == tf4**-1 == pow(tf4, -1) == TransferFunction(p - 3, p + 4, p)
    assert (tf5*tf5).doit()**-1 == tf5**-2 == pow(tf5, -2) == TransferFunction((1 - s)**2, (s**2 + 1)**2, s)

    raises(ValueError, lambda: tf4**(s**2 + s - 1))
    raises(ValueError, lambda: tf5**s)
    raises(ValueError, lambda: tf4**tf5)

    # SymPy's own functions.
    tf = TransferFunction(s - 1, s**2 - 2*s + 1, s)
    tf6 = TransferFunction(s + p, p**2 - 5, s)
    assert factor(tf) == TransferFunction(s - 1, (s - 1)**2, s)
    assert tf.num.subs(s, 2) == tf.den.subs(s, 2) == 1
    # subs & xreplace
    assert tf.subs(s, 2) == TransferFunction(s - 1, s**2 - 2*s + 1, s)
    assert tf6.subs(p, 3) == TransferFunction(s + 3, 4, s)
    assert tf3.xreplace({p: s}) == TransferFunction(-3*s + 3, 1 - s, s)
    raises(TypeError, lambda: tf3.xreplace({p: exp(2)}))
    assert tf3.subs(p, exp(2)) == tf3

    tf7 = TransferFunction(a0*s**p + a1*p**s, a2*p - s, s)
    assert tf7.xreplace({s: k}) == TransferFunction(a0*k**p + a1*p**k, a2*p - k, k)
    assert tf7.subs(s, k) == TransferFunction(a0*s**p + a1*p**s, a2*p - s, s)

    # Conversion to Expr with to_expr()
    tf8 = TransferFunction(a0*s**5 + 5*s**2 + 3, s**6 - 3, s)
    tf9 = TransferFunction((5 + s), (5 + s)*(6 + s), s)
    tf10 = TransferFunction(0, 1, s)
    tf11 = TransferFunction(1, 1, s)
    assert tf8.to_expr() == Mul((a0*s**5 + 5*s**2 + 3), Pow((s**6 - 3), -1, evaluate=False), evaluate=False)
    assert tf9.to_expr() == Mul((s + 5), Pow((5 + s)*(6 + s), -1, evaluate=False), evaluate=False)
    assert tf10.to_expr() == Mul(S(0), Pow(1, -1, evaluate=False), evaluate=False)
    assert tf11.to_expr() == Pow(1, -1, evaluate=False)

def test_TransferFunction_addition_and_subtraction():
    tf1 = TransferFunction(s + 6, s - 5, s)
    tf2 = TransferFunction(s + 3, s + 1, s)
    tf3 = TransferFunction(s + 1, s**2 + s + 1, s)
    tf4 = TransferFunction(p, 2 - p, p)

    # addition
    assert tf1 + tf2 == Parallel(tf1, tf2)
    assert tf3 + tf1 == Parallel(tf3, tf1)
    assert -tf1 + tf2 + tf3 == Parallel(-tf1, tf2, tf3)
    assert tf1 + (tf2 + tf3) == Parallel(tf1, tf2, tf3)

    c = symbols("c", commutative=False)
    raises(ValueError, lambda: tf1 + Matrix([1, 2, 3]))
    raises(ValueError, lambda: tf2 + c)
    raises(ValueError, lambda: tf3 + tf4)
    raises(ValueError, lambda: tf1 + (s - 1))
    raises(ValueError, lambda: tf1 + 8)
    raises(ValueError, lambda: (1 - p**3) + tf1)

    # subtraction
    assert tf1 - tf2 == Parallel(tf1, -tf2)
    assert tf3 - tf2 == Parallel(tf3, -tf2)
    assert -tf1 - tf3 == Parallel(-tf1, -tf3)
    assert tf1 - tf2 + tf3 == Parallel(tf1, -tf2, tf3)

    raises(ValueError, lambda: tf1 - Matrix([1, 2, 3]))
    raises(ValueError, lambda: tf3 - tf4)
    raises(ValueError, lambda: tf1 - (s - 1))
    raises(ValueError, lambda: tf1 - 8)
    raises(ValueError, lambda: (s + 5) - tf2)
    raises(ValueError, lambda: (1 + p**4) - tf1)


def test_TransferFunction_multiplication_and_division():
    G1 = TransferFunction(s + 3, -s**3 + 9, s)
    G2 = TransferFunction(s + 1, s - 5, s)
    G3 = TransferFunction(p, p**4 - 6, p)
    G4 = TransferFunction(p + 4, p - 5, p)
    G5 = TransferFunction(s + 6, s - 5, s)
    G6 = TransferFunction(s + 3, s + 1, s)
    G7 = TransferFunction(1, 1, s)

    # multiplication
    assert G1*G2 == Series(G1, G2)
    assert -G1*G5 == Series(-G1, G5)
    assert -G2*G5*-G6 == Series(-G2, G5, -G6)
    assert -G1*-G2*-G5*-G6 == Series(-G1, -G2, -G5, -G6)
    assert G3*G4 == Series(G3, G4)
    assert (G1*G2)*-(G5*G6) == \
        Series(G1, G2, TransferFunction(-1, 1, s), Series(G5, G6))
    assert G1*G2*(G5 + G6) == Series(G1, G2, Parallel(G5, G6))

    # division - See ``test_Feedback_functions()`` for division by Parallel objects.
    assert G5/G6 == Series(G5, pow(G6, -1))
    assert -G3/G4 == Series(-G3, pow(G4, -1))
    assert (G5*G6)/G7 == Series(G5, G6, pow(G7, -1))

    c = symbols("c", commutative=False)
    raises(ValueError, lambda: G3 * Matrix([1, 2, 3]))
    raises(ValueError, lambda: G1 * c)
    raises(ValueError, lambda: G3 * G5)
    raises(ValueError, lambda: G5 * (s - 1))
    raises(ValueError, lambda: 9 * G5)

    raises(ValueError, lambda: G3 / Matrix([1, 2, 3]))
    raises(ValueError, lambda: G6 / 0)
    raises(ValueError, lambda: G3 / G5)
    raises(ValueError, lambda: G5 / 2)
    raises(ValueError, lambda: G5 / s**2)
    raises(ValueError, lambda: (s - 4*s**2) / G2)
    raises(ValueError, lambda: 0 / G4)
    raises(ValueError, lambda: G7 / (1 + G6))
    raises(ValueError, lambda: G7 / (G5 * G6))
    raises(ValueError, lambda: G7 / (G7 + (G5 + G6)))


def test_TransferFunction_is_proper():
    omega_o, zeta, tau = symbols('omega_o, zeta, tau')
    G1 = TransferFunction(omega_o**2, s**2 + p*omega_o*zeta*s + omega_o**2, omega_o)
    G2 = TransferFunction(tau - s**3, tau + p**4, tau)
    G3 = TransferFunction(a*b*s**3 + s**2 - a*p + s, b - s*p**2, p)
    G4 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
    assert G1.is_proper
    assert G2.is_proper
    assert G3.is_proper
    assert not G4.is_proper


def test_TransferFunction_is_strictly_proper():
    omega_o, zeta, tau = symbols('omega_o, zeta, tau')
    tf1 = TransferFunction(omega_o**2, s**2 + p*omega_o*zeta*s + omega_o**2, omega_o)
    tf2 = TransferFunction(tau - s**3, tau + p**4, tau)
    tf3 = TransferFunction(a*b*s**3 + s**2 - a*p + s, b - s*p**2, p)
    tf4 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
    assert not tf1.is_strictly_proper
    assert not tf2.is_strictly_proper
    assert tf3.is_strictly_proper
    assert not tf4.is_strictly_proper


def test_TransferFunction_is_biproper():
    tau, omega_o, zeta = symbols('tau, omega_o, zeta')
    tf1 = TransferFunction(omega_o**2, s**2 + p*omega_o*zeta*s + omega_o**2, omega_o)
    tf2 = TransferFunction(tau - s**3, tau + p**4, tau)
    tf3 = TransferFunction(a*b*s**3 + s**2 - a*p + s, b - s*p**2, p)
    tf4 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
    assert tf1.is_biproper
    assert tf2.is_biproper
    assert not tf3.is_biproper
    assert not tf4.is_biproper


def test_Series_construction():
    tf = TransferFunction(a0*s**3 + a1*s**2 - a2*s, b0*p**4 + b1*p**3 - b2*s*p, s)
    tf2 = TransferFunction(a2*p - s, a2*s + p, s)
    tf3 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf4 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    inp = Function('X_d')(s)
    out = Function('X')(s)

    s0 = Series(tf, tf2)
    assert s0.args == (tf, tf2)
    assert s0.var == s

    s1 = Series(Parallel(tf, -tf2), tf2)
    assert s1.args == (Parallel(tf, -tf2), tf2)
    assert s1.var == s

    tf3_ = TransferFunction(inp, 1, s)
    tf4_ = TransferFunction(-out, 1, s)
    s2 = Series(tf, Parallel(tf3_, tf4_), tf2)
    assert s2.args == (tf, Parallel(tf3_, tf4_), tf2)

    s3 = Series(tf, tf2, tf4)
    assert s3.args == (tf, tf2, tf4)

    s4 = Series(tf3_, tf4_)
    assert s4.args == (tf3_, tf4_)
    assert s4.var == s

    s6 = Series(tf2, tf4, Parallel(tf2, -tf), tf4)
    assert s6.args == (tf2, tf4, Parallel(tf2, -tf), tf4)

    s7 = Series(tf, tf2)
    assert s0 == s7
    assert not s0 == s2

    raises(ValueError, lambda: Series(tf, tf3))
    raises(ValueError, lambda: Series(tf, tf2, tf3, tf4))
    raises(ValueError, lambda: Series(-tf3, tf2))
    raises(TypeError, lambda: Series(2, tf, tf4))
    raises(TypeError, lambda: Series(s**2 + p*s, tf3, tf2))
    raises(TypeError, lambda: Series(tf3, Matrix([1, 2, 3, 4])))


def test_MIMOSeries_construction():
    tf_1 = TransferFunction(a0*s**3 + a1*s**2 - a2*s, b0*p**4 + b1*p**3 - b2*s*p, s)
    tf_2 = TransferFunction(a2*p - s, a2*s + p, s)
    tf_3 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)

    tfm_1 = TransferFunctionMatrix([[tf_1, tf_2, tf_3], [-tf_3, -tf_2, tf_1]])
    tfm_2 = TransferFunctionMatrix([[-tf_2], [-tf_2], [-tf_3]])
    tfm_3 = TransferFunctionMatrix([[-tf_3]])
    tfm_4 = TransferFunctionMatrix([[TF3], [TF2], [-TF1]])
    tfm_5 = TransferFunctionMatrix.from_Matrix(Matrix([1/p]), p)

    s8 = MIMOSeries(tfm_2, tfm_1)
    assert s8.args == (tfm_2, tfm_1)
    assert s8.var == s
    assert s8.shape == (s8.num_outputs, s8.num_inputs) == (2, 1)

    s9 = MIMOSeries(tfm_3, tfm_2, tfm_1)
    assert s9.args == (tfm_3, tfm_2, tfm_1)
    assert s9.var == s
    assert s9.shape == (s9.num_outputs, s9.num_inputs) == (2, 1)

    s11 = MIMOSeries(tfm_3, MIMOParallel(-tfm_2, -tfm_4), tfm_1)
    assert s11.args == (tfm_3, MIMOParallel(-tfm_2, -tfm_4), tfm_1)
    assert s11.shape == (s11.num_outputs, s11.num_inputs) == (2, 1)

    # arg cannot be empty tuple.
    raises(ValueError, lambda: MIMOSeries())

    # arg cannot contain SISO as well as MIMO systems.
    raises(TypeError, lambda: MIMOSeries(tfm_1, tf_1))

    # for all the adjacent transfer function matrices:
    # no. of inputs of first TFM must be equal to the no. of outputs of the second TFM.
    raises(ValueError, lambda: MIMOSeries(tfm_1, tfm_2, -tfm_1))

    # all the TFMs must use the same complex variable.
    raises(ValueError, lambda: MIMOSeries(tfm_3, tfm_5))

    # Number or expression not allowed in the arguments.
    raises(TypeError, lambda: MIMOSeries(2, tfm_2, tfm_3))
    raises(TypeError, lambda: MIMOSeries(s**2 + p*s, -tfm_2, tfm_3))
    raises(TypeError, lambda: MIMOSeries(Matrix([1/p]), tfm_3))


def test_Series_functions():
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    assert tf1*tf2*tf3 == Series(tf1, tf2, tf3) == Series(Series(tf1, tf2), tf3) \
        == Series(tf1, Series(tf2, tf3))
    assert tf1*(tf2 + tf3) == Series(tf1, Parallel(tf2, tf3))
    assert tf1*tf2 + tf5 == Parallel(Series(tf1, tf2), tf5)
    assert tf1*tf2 - tf5 == Parallel(Series(tf1, tf2), -tf5)
    assert tf1*tf2 + tf3 + tf5 == Parallel(Series(tf1, tf2), tf3, tf5)
    assert tf1*tf2 - tf3 - tf5 == Parallel(Series(tf1, tf2), -tf3, -tf5)
    assert tf1*tf2 - tf3 + tf5 == Parallel(Series(tf1, tf2), -tf3, tf5)
    assert tf1*tf2 + tf3*tf5 == Parallel(Series(tf1, tf2), Series(tf3, tf5))
    assert tf1*tf2 - tf3*tf5 == Parallel(Series(tf1, tf2), Series(TransferFunction(-1, 1, s), Series(tf3, tf5)))
    assert tf2*tf3*(tf2 - tf1)*tf3 == Series(tf2, tf3, Parallel(tf2, -tf1), tf3)
    assert -tf1*tf2 == Series(-tf1, tf2)
    assert -(tf1*tf2) == Series(TransferFunction(-1, 1, s), Series(tf1, tf2))
    raises(ValueError, lambda: tf1*tf2*tf4)
    raises(ValueError, lambda: tf1*(tf2 - tf4))
    raises(ValueError, lambda: tf3*Matrix([1, 2, 3]))

    # evaluate=True -> doit()
    assert Series(tf1, tf2, evaluate=True) == Series(tf1, tf2).doit() == \
        TransferFunction(k, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Series(tf1, tf2, Parallel(tf1, -tf3), evaluate=True) == Series(tf1, tf2, Parallel(tf1, -tf3)).doit() == \
        TransferFunction(k*(a2*s + p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2)), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)**2, s)
    assert Series(tf2, tf1, -tf3, evaluate=True) == Series(tf2, tf1, -tf3).doit() == \
        TransferFunction(k*(-a2*p + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert not Series(tf1, -tf2, evaluate=False) == Series(tf1, -tf2).doit()

    assert Series(Parallel(tf1, tf2), Parallel(tf2, -tf3)).doit() == \
        TransferFunction((k*(s**2 + 2*s*wn*zeta + wn**2) + 1)*(-a2*p + k*(a2*s + p) + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Series(-tf1, -tf2, -tf3).doit() == \
        TransferFunction(k*(-a2*p + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert -Series(tf1, tf2, tf3).doit() == \
        TransferFunction(-k*(a2*p - s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Series(tf2, tf3, Parallel(tf2, -tf1), tf3).doit() == \
        TransferFunction(k*(a2*p - s)**2*(k*(s**2 + 2*s*wn*zeta + wn**2) - 1), (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2), s)

    assert Series(tf1, tf2).rewrite(TransferFunction) == TransferFunction(k, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Series(tf2, tf1, -tf3).rewrite(TransferFunction) == \
        TransferFunction(k*(-a2*p + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)

    S1 = Series(Parallel(tf1, tf2), Parallel(tf2, -tf3))
    assert S1.is_proper
    assert not S1.is_strictly_proper
    assert S1.is_biproper

    S2 = Series(tf1, tf2, tf3)
    assert S2.is_proper
    assert S2.is_strictly_proper
    assert not S2.is_biproper

    S3 = Series(tf1, -tf2, Parallel(tf1, -tf3))
    assert S3.is_proper
    assert S3.is_strictly_proper
    assert not S3.is_biproper


def test_MIMOSeries_functions():
    tfm1 = TransferFunctionMatrix([[TF1, TF2, TF3], [-TF3, -TF2, TF1]])
    tfm2 = TransferFunctionMatrix([[-TF1], [-TF2], [-TF3]])
    tfm3 = TransferFunctionMatrix([[-TF1]])
    tfm4 = TransferFunctionMatrix([[-TF2, -TF3], [-TF1, TF2]])
    tfm5 = TransferFunctionMatrix([[TF2, -TF2], [-TF3, -TF2]])
    tfm6 = TransferFunctionMatrix([[-TF3], [TF1]])
    tfm7 = TransferFunctionMatrix([[TF1], [-TF2]])

    assert tfm1*tfm2 + tfm6 == MIMOParallel(MIMOSeries(tfm2, tfm1), tfm6)
    assert tfm1*tfm2 + tfm7 + tfm6 == MIMOParallel(MIMOSeries(tfm2, tfm1), tfm7, tfm6)
    assert tfm1*tfm2 - tfm6 - tfm7 == MIMOParallel(MIMOSeries(tfm2, tfm1), -tfm6, -tfm7)
    assert tfm4*tfm5 + (tfm4 - tfm5) == MIMOParallel(MIMOSeries(tfm5, tfm4), tfm4, -tfm5)
    assert tfm4*-tfm6 + (-tfm4*tfm6) == MIMOParallel(MIMOSeries(-tfm6, tfm4), MIMOSeries(tfm6, -tfm4))

    raises(ValueError, lambda: tfm1*tfm2 + TF1)
    raises(TypeError, lambda: tfm1*tfm2 + a0)
    raises(TypeError, lambda: tfm4*tfm6 - (s - 1))
    raises(TypeError, lambda: tfm4*-tfm6 - 8)
    raises(TypeError, lambda: (-1 + p**5) + tfm1*tfm2)

    # Shape criteria.

    raises(TypeError, lambda: -tfm1*tfm2 + tfm4)
    raises(TypeError, lambda: tfm1*tfm2 - tfm4 + tfm5)
    raises(TypeError, lambda: tfm1*tfm2 - tfm4*tfm5)

    assert tfm1*tfm2*-tfm3 == MIMOSeries(-tfm3, tfm2, tfm1)
    assert (tfm1*-tfm2)*tfm3 == MIMOSeries(tfm3, -tfm2, tfm1)

    # Multiplication of a Series object with a SISO TF not allowed.

    raises(ValueError, lambda: tfm4*tfm5*TF1)
    raises(TypeError, lambda: tfm4*tfm5*a1)
    raises(TypeError, lambda: tfm4*-tfm5*(s - 2))
    raises(TypeError, lambda: tfm5*tfm4*9)
    raises(TypeError, lambda: (-p**3 + 1)*tfm5*tfm4)

    # Transfer function matrix in the arguments.
    assert (MIMOSeries(tfm2, tfm1, evaluate=True) == MIMOSeries(tfm2, tfm1).doit()
        == TransferFunctionMatrix(((TransferFunction(-k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (-a2*p + s)*(a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2)**2 - (a2*s + p)**2,
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2, s),),
        (TransferFunction(k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (-a2*p + s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + (a2*p - s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2),
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2, s),))))

    # doit() should not cancel poles and zeros.
    mat_1 = Matrix([[1/(1+s), (1+s)/(1+s**2+2*s)**3]])
    mat_2 = Matrix([[(1+s)], [(1+s**2+2*s)**3/(1+s)]])
    tm_1, tm_2 = TransferFunctionMatrix.from_Matrix(mat_1, s), TransferFunctionMatrix.from_Matrix(mat_2, s)
    assert (MIMOSeries(tm_2, tm_1).doit()
        == TransferFunctionMatrix(((TransferFunction(2*(s + 1)**2*(s**2 + 2*s + 1)**3, (s + 1)**2*(s**2 + 2*s + 1)**3, s),),)))
    assert MIMOSeries(tm_2, tm_1).doit().simplify() == TransferFunctionMatrix(((TransferFunction(2, 1, s),),))

    # calling doit() will expand the internal Series and Parallel objects.
    assert (MIMOSeries(-tfm3, -tfm2, tfm1, evaluate=True)
        == MIMOSeries(-tfm3, -tfm2, tfm1).doit()
        == TransferFunctionMatrix(((TransferFunction(k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (a2*p - s)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (a2*s + p)**2,
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**3, s),),
        (TransferFunction(-k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (-a2*p + s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + (a2*p - s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2),
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**3, s),))))
    assert (MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5, evaluate=True)
        == MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5).doit()
        == TransferFunctionMatrix(((TransferFunction(-k*(-a2*s - p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2)), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s), TransferFunction(k*(-a2*p - \
            k*(a2*s + p) + s), a2*s + p, s)), (TransferFunction(-k*(-a2*s - p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2)), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s), \
            TransferFunction((-a2*p + s)*(-a2*p - k*(a2*s + p) + s), (a2*s + p)**2, s)))) == MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5).rewrite(TransferFunctionMatrix))


def test_Parallel_construction():
    tf = TransferFunction(a0*s**3 + a1*s**2 - a2*s, b0*p**4 + b1*p**3 - b2*s*p, s)
    tf2 = TransferFunction(a2*p - s, a2*s + p, s)
    tf3 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf4 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    inp = Function('X_d')(s)
    out = Function('X')(s)

    p0 = Parallel(tf, tf2)
    assert p0.args == (tf, tf2)
    assert p0.var == s

    p1 = Parallel(Series(tf, -tf2), tf2)
    assert p1.args == (Series(tf, -tf2), tf2)
    assert p1.var == s

    tf3_ = TransferFunction(inp, 1, s)
    tf4_ = TransferFunction(-out, 1, s)
    p2 = Parallel(tf, Series(tf3_, -tf4_), tf2)
    assert p2.args == (tf, Series(tf3_, -tf4_), tf2)

    p3 = Parallel(tf, tf2, tf4)
    assert p3.args == (tf, tf2, tf4)

    p4 = Parallel(tf3_, tf4_)
    assert p4.args == (tf3_, tf4_)
    assert p4.var == s

    p5 = Parallel(tf, tf2)
    assert p0 == p5
    assert not p0 == p1

    p6 = Parallel(tf2, tf4, Series(tf2, -tf4))
    assert p6.args == (tf2, tf4, Series(tf2, -tf4))

    p7 = Parallel(tf2, tf4, Series(tf2, -tf), tf4)
    assert p7.args == (tf2, tf4, Series(tf2, -tf), tf4)

    raises(ValueError, lambda: Parallel(tf, tf3))
    raises(ValueError, lambda: Parallel(tf, tf2, tf3, tf4))
    raises(ValueError, lambda: Parallel(-tf3, tf4))
    raises(TypeError, lambda: Parallel(2, tf, tf4))
    raises(TypeError, lambda: Parallel(s**2 + p*s, tf3, tf2))
    raises(TypeError, lambda: Parallel(tf3, Matrix([1, 2, 3, 4])))


def test_MIMOParallel_construction():
    tfm1 = TransferFunctionMatrix([[TF1], [TF2], [TF3]])
    tfm2 = TransferFunctionMatrix([[-TF3], [TF2], [TF1]])
    tfm3 = TransferFunctionMatrix([[TF1]])
    tfm4 = TransferFunctionMatrix([[TF2], [TF1], [TF3]])
    tfm5 = TransferFunctionMatrix([[TF1, TF2], [TF2, TF1]])
    tfm6 = TransferFunctionMatrix([[TF2, TF1], [TF1, TF2]])
    tfm7 = TransferFunctionMatrix.from_Matrix(Matrix([[1/p]]), p)

    p8 = MIMOParallel(tfm1, tfm2)
    assert p8.args == (tfm1, tfm2)
    assert p8.var == s
    assert p8.shape == (p8.num_outputs, p8.num_inputs) == (3, 1)

    p9 = MIMOParallel(MIMOSeries(tfm3, tfm1), tfm2)
    assert p9.args == (MIMOSeries(tfm3, tfm1), tfm2)
    assert p9.var == s
    assert p9.shape == (p9.num_outputs, p9.num_inputs) == (3, 1)

    p10 = MIMOParallel(tfm1, MIMOSeries(tfm3, tfm4), tfm2)
    assert p10.args == (tfm1, MIMOSeries(tfm3, tfm4), tfm2)
    assert p10.var == s
    assert p10.shape == (p10.num_outputs, p10.num_inputs) == (3, 1)

    p11 = MIMOParallel(tfm2, tfm1, tfm4)
    assert p11.args == (tfm2, tfm1, tfm4)
    assert p11.shape == (p11.num_outputs, p11.num_inputs) == (3, 1)

    p12 = MIMOParallel(tfm6, tfm5)
    assert p12.args == (tfm6, tfm5)
    assert p12.shape == (p12.num_outputs, p12.num_inputs) == (2, 2)

    p13 = MIMOParallel(tfm2, tfm4, MIMOSeries(-tfm3, tfm4), -tfm4)
    assert p13.args == (tfm2, tfm4, MIMOSeries(-tfm3, tfm4), -tfm4)
    assert p13.shape == (p13.num_outputs, p13.num_inputs) == (3, 1)

    # arg cannot be empty tuple.
    raises(TypeError, lambda: MIMOParallel(()))

    # arg cannot contain SISO as well as MIMO systems.
    raises(TypeError, lambda: MIMOParallel(tfm1, tfm2, TF1))

    # all TFMs must have same shapes.
    raises(TypeError, lambda: MIMOParallel(tfm1, tfm3, tfm4))

    # all TFMs must be using the same complex variable.
    raises(ValueError, lambda: MIMOParallel(tfm3, tfm7))

    # Number or expression not allowed in the arguments.
    raises(TypeError, lambda: MIMOParallel(2, tfm1, tfm4))
    raises(TypeError, lambda: MIMOParallel(s**2 + p*s, -tfm4, tfm2))


def test_Parallel_functions():
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    assert tf1 + tf2 + tf3 == Parallel(tf1, tf2, tf3)
    assert tf1 + tf2 + tf3 + tf5 == Parallel(tf1, tf2, tf3, tf5)
    assert tf1 + tf2 - tf3 - tf5 == Parallel(tf1, tf2, -tf3, -tf5)
    assert tf1 + tf2*tf3 == Parallel(tf1, Series(tf2, tf3))
    assert tf1 - tf2*tf3 == Parallel(tf1, -Series(tf2,tf3))
    assert -tf1 - tf2 == Parallel(-tf1, -tf2)
    assert -(tf1 + tf2) == Series(TransferFunction(-1, 1, s), Parallel(tf1, tf2))
    assert (tf2 + tf3)*tf1 == Series(Parallel(tf2, tf3), tf1)
    assert (tf1 + tf2)*(tf3*tf5) == Series(Parallel(tf1, tf2), tf3, tf5)
    assert -(tf2 + tf3)*-tf5 == Series(TransferFunction(-1, 1, s), Parallel(tf2, tf3), -tf5)
    assert tf2 + tf3 + tf2*tf1 + tf5 == Parallel(tf2, tf3, Series(tf2, tf1), tf5)
    assert tf2 + tf3 + tf2*tf1 - tf3 == Parallel(tf2, tf3, Series(tf2, tf1), -tf3)
    assert (tf1 + tf2 + tf5)*(tf3 + tf5) == Series(Parallel(tf1, tf2, tf5), Parallel(tf3, tf5))
    raises(ValueError, lambda: tf1 + tf2 + tf4)
    raises(ValueError, lambda: tf1 - tf2*tf4)
    raises(ValueError, lambda: tf3 + Matrix([1, 2, 3]))

    # evaluate=True -> doit()
    assert Parallel(tf1, tf2, evaluate=True) == Parallel(tf1, tf2).doit() == \
        TransferFunction(k*(s**2 + 2*s*wn*zeta + wn**2) + 1, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Parallel(tf1, tf2, Series(-tf1, tf3), evaluate=True) == \
        Parallel(tf1, tf2, Series(-tf1, tf3)).doit() == TransferFunction(k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)**2 + \
            (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2) + (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), (a2*s + p)*(s**2 + \
                2*s*wn*zeta + wn**2)**2, s)
    assert Parallel(tf2, tf1, -tf3, evaluate=True) == Parallel(tf2, tf1, -tf3).doit() == \
        TransferFunction(a2*s + k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2) \
            , (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert not Parallel(tf1, -tf2, evaluate=False) == Parallel(tf1, -tf2).doit()

    assert Parallel(Series(tf1, tf2), Series(tf2, tf3)).doit() == \
        TransferFunction(k*(a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2) + k*(a2*s + p), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Parallel(-tf1, -tf2, -tf3).doit() == \
        TransferFunction(-a2*s - k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) - p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2), \
            (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert -Parallel(tf1, tf2, tf3).doit() == \
        TransferFunction(-a2*s - k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) - p - (a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2), \
            (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Parallel(tf2, tf3, Series(tf2, -tf1), tf3).doit() == \
        TransferFunction(k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) - k*(a2*s + p) + (2*a2*p - 2*s)*(s**2 + 2*s*wn*zeta \
            + wn**2), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)

    assert Parallel(tf1, tf2).rewrite(TransferFunction) == \
        TransferFunction(k*(s**2 + 2*s*wn*zeta + wn**2) + 1, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Parallel(tf2, tf1, -tf3).rewrite(TransferFunction) == \
        TransferFunction(a2*s + k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + \
             wn**2), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)

    assert Parallel(tf1, Parallel(tf2, tf3)) == Parallel(tf1, tf2, tf3) == Parallel(Parallel(tf1, tf2), tf3)

    P1 = Parallel(Series(tf1, tf2), Series(tf2, tf3))
    assert P1.is_proper
    assert not P1.is_strictly_proper
    assert P1.is_biproper

    P2 = Parallel(tf1, -tf2, -tf3)
    assert P2.is_proper
    assert not P2.is_strictly_proper
    assert P2.is_biproper

    P3 = Parallel(tf1, -tf2, Series(tf1, tf3))
    assert P3.is_proper
    assert not P3.is_strictly_proper
    assert P3.is_biproper


def test_MIMOParallel_functions():
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    tfm1 = TransferFunctionMatrix([[TF1], [TF2], [TF3]])
    tfm2 = TransferFunctionMatrix([[-TF2], [tf5], [-TF1]])
    tfm3 = TransferFunctionMatrix([[tf5], [-tf5], [TF2]])
    tfm4 = TransferFunctionMatrix([[TF2, -tf5], [TF1, tf5]])
    tfm5 = TransferFunctionMatrix([[TF1, TF2], [TF3, -tf5]])
    tfm6 = TransferFunctionMatrix([[-TF2]])
    tfm7 = TransferFunctionMatrix([[tf4], [-tf4], [tf4]])

    assert tfm1 + tfm2 + tfm3 == MIMOParallel(tfm1, tfm2, tfm3) == MIMOParallel(MIMOParallel(tfm1, tfm2), tfm3)
    assert tfm2 - tfm1 - tfm3 == MIMOParallel(tfm2, -tfm1, -tfm3)
    assert tfm2 - tfm3 + (-tfm1*tfm6*-tfm6) == MIMOParallel(tfm2, -tfm3, MIMOSeries(-tfm6, tfm6, -tfm1))
    assert tfm1 + tfm1 - (-tfm1*tfm6) == MIMOParallel(tfm1, tfm1, -MIMOSeries(tfm6, -tfm1))
    assert tfm2 - tfm3 - tfm1 + tfm2 == MIMOParallel(tfm2, -tfm3, -tfm1, tfm2)
    assert tfm1 + tfm2 - tfm3 - tfm1 == MIMOParallel(tfm1, tfm2, -tfm3, -tfm1)
    raises(ValueError, lambda: tfm1 + tfm2 + TF2)
    raises(TypeError, lambda: tfm1 - tfm2 - a1)
    raises(TypeError, lambda: tfm2 - tfm3 - (s - 1))
    raises(TypeError, lambda: -tfm3 - tfm2 - 9)
    raises(TypeError, lambda: (1 - p**3) - tfm3 - tfm2)
    # All TFMs must use the same complex var. tfm7 uses 'p'.
    raises(ValueError, lambda: tfm3 - tfm2 - tfm7)
    raises(ValueError, lambda: tfm2 - tfm1 + tfm7)
    # (tfm1 +/- tfm2) has (3, 1) shape while tfm4 has (2, 2) shape.
    raises(TypeError, lambda: tfm1 + tfm2 + tfm4)
    raises(TypeError, lambda: (tfm1 - tfm2) - tfm4)

    assert (tfm1 + tfm2)*tfm6 == MIMOSeries(tfm6, MIMOParallel(tfm1, tfm2))
    assert (tfm2 - tfm3)*tfm6*-tfm6 == MIMOSeries(-tfm6, tfm6, MIMOParallel(tfm2, -tfm3))
    assert (tfm2 - tfm1 - tfm3)*(tfm6 + tfm6) == MIMOSeries(MIMOParallel(tfm6, tfm6), MIMOParallel(tfm2, -tfm1, -tfm3))
    raises(ValueError, lambda: (tfm4 + tfm5)*TF1)
    raises(TypeError, lambda: (tfm2 - tfm3)*a2)
    raises(TypeError, lambda: (tfm3 + tfm2)*(s - 6))
    raises(TypeError, lambda: (tfm1 + tfm2 + tfm3)*0)
    raises(TypeError, lambda: (1 - p**3)*(tfm1 + tfm3))

    # (tfm3 - tfm2) has (3, 1) shape while tfm4*tfm5 has (2, 2) shape.
    raises(ValueError, lambda: (tfm3 - tfm2)*tfm4*tfm5)
    # (tfm1 - tfm2) has (3, 1) shape while tfm5 has (2, 2) shape.
    raises(ValueError, lambda: (tfm1 - tfm2)*tfm5)

    # TFM in the arguments.
    assert (MIMOParallel(tfm1, tfm2, evaluate=True) == MIMOParallel(tfm1, tfm2).doit()
    == MIMOParallel(tfm1, tfm2).rewrite(TransferFunctionMatrix)
    == TransferFunctionMatrix(((TransferFunction(-k*(s**2 + 2*s*wn*zeta + wn**2) + 1, s**2 + 2*s*wn*zeta + wn**2, s),), \
        (TransferFunction(-a0 + a1*s**2 + a2*s + k*(a0 + s), a0 + s, s),), (TransferFunction(-a2*s - p + (a2*p - s)* \
        (s**2 + 2*s*wn*zeta + wn**2), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s),))))


def test_Feedback_construction():
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)
    tf6 = TransferFunction(s - p, p + s, p)

    f1 = Feedback(TransferFunction(1, 1, s), tf1*tf2*tf3)
    assert f1.args == (TransferFunction(1, 1, s), Series(tf1, tf2, tf3), -1)
    assert f1.sys1 == TransferFunction(1, 1, s)
    assert f1.sys2 == Series(tf1, tf2, tf3)
    assert f1.var == s

    f2 = Feedback(tf1, tf2*tf3)
    assert f2.args == (tf1, Series(tf2, tf3), -1)
    assert f2.sys1 == tf1
    assert f2.sys2 == Series(tf2, tf3)
    assert f2.var == s

    f3 = Feedback(tf1*tf2, tf5)
    assert f3.args == (Series(tf1, tf2), tf5, -1)
    assert f3.sys1 == Series(tf1, tf2)

    f4 = Feedback(tf4, tf6)
    assert f4.args == (tf4, tf6, -1)
    assert f4.sys1 == tf4
    assert f4.var == p

    f5 = Feedback(tf5, TransferFunction(1, 1, s))
    assert f5.args == (tf5, TransferFunction(1, 1, s), -1)
    assert f5.var == s
    assert f5 == Feedback(tf5)  # When sys2 is not passed explicitly, it is assumed to be unit tf.

    f6 = Feedback(TransferFunction(1, 1, p), tf4)
    assert f6.args == (TransferFunction(1, 1, p), tf4, -1)
    assert f6.var == p

    f7 = -Feedback(tf4*tf6, TransferFunction(1, 1, p))
    assert f7.args == (Series(TransferFunction(-1, 1, p), Series(tf4, tf6)), -TransferFunction(1, 1, p), -1)
    assert f7.sys1 == Series(TransferFunction(-1, 1, p), Series(tf4, tf6))

    # denominator can't be a Parallel instance
    raises(TypeError, lambda: Feedback(tf1, tf2 + tf3))
    raises(TypeError, lambda: Feedback(tf1, Matrix([1, 2, 3])))
    raises(TypeError, lambda: Feedback(TransferFunction(1, 1, s), s - 1))
    raises(TypeError, lambda: Feedback(1, 1))
    # raises(ValueError, lambda: Feedback(TransferFunction(1, 1, s), TransferFunction(1, 1, s)))
    raises(ValueError, lambda: Feedback(tf2, tf4*tf5))
    raises(ValueError, lambda: Feedback(tf2, tf1, 1.5))  # `sign` can only be -1 or 1
    raises(ValueError, lambda: Feedback(tf1, -tf1**-1))  # denominator can't be zero
    raises(ValueError, lambda: Feedback(tf4, tf5))  # Both systems should use the same `var`


def test_Feedback_functions():
    tf = TransferFunction(1, 1, s)
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)
    tf6 = TransferFunction(s - p, p + s, p)

    assert (tf1*tf2*tf3 / tf3*tf5) == Series(tf1, tf2, tf3, pow(tf3, -1), tf5)
    assert (tf1*tf2*tf3) / (tf3*tf5) == Series((tf1*tf2*tf3).doit(), pow((tf3*tf5).doit(),-1))
    assert tf / (tf + tf1) == Feedback(tf, tf1)
    assert tf / (tf + tf1*tf2*tf3) == Feedback(tf, tf1*tf2*tf3)
    assert tf1 / (tf + tf1*tf2*tf3) == Feedback(tf1, tf2*tf3)
    assert (tf1*tf2) / (tf + tf1*tf2) == Feedback(tf1*tf2, tf)
    assert (tf1*tf2) / (tf + tf1*tf2*tf5) == Feedback(tf1*tf2, tf5)
    assert (tf1*tf2) / (tf + tf1*tf2*tf5*tf3) in (Feedback(tf1*tf2, tf5*tf3), Feedback(tf1*tf2, tf3*tf5))
    assert tf4 / (TransferFunction(1, 1, p) + tf4*tf6) == Feedback(tf4, tf6)
    assert tf5 / (tf + tf5) == Feedback(tf5, tf)

    raises(TypeError, lambda: tf1*tf2*tf3 / (1 + tf1*tf2*tf3))
    raises(ValueError, lambda: tf2*tf3 / (tf + tf2*tf3*tf4))

    assert Feedback(tf, tf1*tf2*tf3).doit() == \
        TransferFunction((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), k*(a2*p - s) + \
        (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Feedback(tf, tf1*tf2*tf3).sensitivity == \
        1/(k*(a2*p - s)/((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)) + 1)
    assert Feedback(tf1, tf2*tf3).doit() == \
        TransferFunction((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), (k*(a2*p - s) + \
        (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2))*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Feedback(tf1, tf2*tf3).sensitivity == \
        1/(k*(a2*p - s)/((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)) + 1)
    assert Feedback(tf1*tf2, tf5).doit() == \
        TransferFunction(k*(a0 + s)*(s**2 + 2*s*wn*zeta + wn**2), (k*(-a0 + a1*s**2 + a2*s) + \
        (a0 + s)*(s**2 + 2*s*wn*zeta + wn**2))*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Feedback(tf1*tf2, tf5, 1).sensitivity == \
        1/(-k*(-a0 + a1*s**2 + a2*s)/((a0 + s)*(s**2 + 2*s*wn*zeta + wn**2)) + 1)
    assert Feedback(tf4, tf6).doit() == \
        TransferFunction(p*(p + s)*(a0*p + p**a1 - s), p*(p*(p + s) + (-p + s)*(a0*p + p**a1 - s)), p)
    assert -Feedback(tf4*tf6, TransferFunction(1, 1, p)).doit() == \
        TransferFunction(-p*(-p + s)*(p + s)*(a0*p + p**a1 - s), p*(p + s)*(p*(p + s) + (-p + s)*(a0*p + p**a1 - s)), p)
    assert Feedback(tf, tf).doit() == TransferFunction(1, 2, s)

    assert Feedback(tf1, tf2*tf5).rewrite(TransferFunction) == \
        TransferFunction((a0 + s)*(s**2 + 2*s*wn*zeta + wn**2), (k*(-a0 + a1*s**2 + a2*s) + \
        (a0 + s)*(s**2 + 2*s*wn*zeta + wn**2))*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Feedback(TransferFunction(1, 1, p), tf4).rewrite(TransferFunction) == \
        TransferFunction(p, a0*p + p + p**a1 - s, p)


def test_Feedback_as_TransferFunction():
    # Solves issue https://github.com/sympy/sympy/issues/26161
    tf1 = TransferFunction(s+1, 1, s)
    tf2 = TransferFunction(s+2, 1, s)
    fd1 = Feedback(tf1, tf2, -1) # Negative Feedback system
    fd2 = Feedback(tf1, tf2, 1) # Positive Feedback system
    unit = TransferFunction(1, 1, s)

    # Checking the type
    assert isinstance(fd1, TransferFunction)
    assert isinstance(fd1, Feedback)

    # Testing the numerator and denominator
    assert fd1.num == tf1
    assert fd2.num == tf1
    assert fd1.den == Parallel(unit, Series(tf2, tf1))
    assert fd2.den == Parallel(unit, -Series(tf2, tf1))

    # Testing the Series and Parallel Combination with Feedback and TransferFunction
    s1 = Series(tf1, fd1)
    p1 = Parallel(tf1, fd1)
    assert tf1 * fd1 == s1
    assert tf1 + fd1 == p1
    assert s1.doit() == TransferFunction((s + 1)**2, (s + 1)*(s + 2) + 1, s)
    assert p1.doit() == TransferFunction(s + (s + 1)*((s + 1)*(s + 2) + 1) + 1, (s + 1)*(s + 2) + 1, s)

    # Testing the use of Feedback and TransferFunction with Feedback
    fd3 = Feedback(tf1*fd1, tf2, -1)
    assert fd3 == Feedback(Series(tf1, fd1), tf2)
    assert fd3.num == tf1 * fd1
    assert fd3.den == Parallel(unit, Series(tf2, Series(tf1, fd1)))

    # Testing the use of Feedback and TransferFunction with TransferFunction
    tf3 = TransferFunction(tf1*fd1, tf2, s)
    assert tf3 == TransferFunction(Series(tf1, fd1), tf2, s)
    assert tf3.num == tf1*fd1

def test_issue_26161():
    # Issue https://github.com/sympy/sympy/issues/26161
    Ib, Is, m, h, l2, l1 = symbols('I_b, I_s, m, h, l2, l1',
                                            real=True, nonnegative=True)
    KD, KP, v = symbols('K_D, K_P, v', real=True)

    tau1_sq = (Ib + m * h ** 2) / m / g / h
    tau2 = l2 / v
    tau3 = v / (l1 + l2)
    K = v ** 2 / g / (l1 + l2)

    Gtheta = TransferFunction(-K * (tau2 * s + 1), tau1_sq * s ** 2 - 1, s)
    Gdelta = TransferFunction(1, Is * s ** 2 + c * s, s)
    Gpsi = TransferFunction(1, tau3 * s, s)
    Dcont = TransferFunction(KD * s, 1, s)
    PIcont = TransferFunction(KP, s, s)
    Gunity = TransferFunction(1, 1, s)

    Ginner = Feedback(Dcont * Gdelta, Gtheta)
    Gouter = Feedback(PIcont * Ginner * Gpsi, Gunity)
    assert Gouter == Feedback(Series(PIcont, Series(Ginner, Gpsi)), Gunity)
    assert Gouter.num == Series(PIcont, Series(Ginner, Gpsi))
    assert Gouter.den == Parallel(Gunity, Series(Gunity, Series(PIcont, Series(Ginner, Gpsi))))
    expr = (KD*KP*g*s**3*v**2*(l1 + l2)*(Is*s**2 + c*s)**2*(-g*h*m + s**2*(Ib + h**2*m))*(-KD*g*h*m*s*v**2*(l2*s + v) + \
            g*v*(l1 + l2)*(Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m))))/((s**2*v*(Is*s**2 + c*s)*(-KD*g*h*m*s*v**2* \
            (l2*s + v) + g*v*(l1 + l2)*(Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m)))*(KD*KP*g*s*v*(l1 + l2)**2* \
            (Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m)) + s**2*v*(Is*s**2 + c*s)*(-KD*g*h*m*s*v**2*(l2*s + v) + \
            g*v*(l1 + l2)*(Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m))))/(l1 + l2)))

    assert (Gouter.to_expr() - expr).simplify() == 0


def test_MIMOFeedback_construction():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s**3 - 1, s)
    tf3 = TransferFunction(s, s + 1, s)
    tf4 = TransferFunction(s, s**2 + 1, s)

    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf3], [tf4, tf1]])
    tfm_3 = TransferFunctionMatrix([[tf3, tf4], [tf1, tf2]])

    f1 = MIMOFeedback(tfm_1, tfm_2)
    assert f1.args == (tfm_1, tfm_2, -1)
    assert f1.sys1 == tfm_1
    assert f1.sys2 == tfm_2
    assert f1.var == s
    assert f1.sign == -1
    assert -(-f1) == f1

    f2 = MIMOFeedback(tfm_2, tfm_1, 1)
    assert f2.args == (tfm_2, tfm_1, 1)
    assert f2.sys1 == tfm_2
    assert f2.sys2 == tfm_1
    assert f2.var == s
    assert f2.sign == 1

    f3 = MIMOFeedback(tfm_1, MIMOSeries(tfm_3, tfm_2))
    assert f3.args == (tfm_1, MIMOSeries(tfm_3, tfm_2), -1)
    assert f3.sys1 == tfm_1
    assert f3.sys2 == MIMOSeries(tfm_3, tfm_2)
    assert f3.var == s
    assert f3.sign == -1

    mat = Matrix([[1, 1/s], [0, 1]])
    sys1 = controller = TransferFunctionMatrix.from_Matrix(mat, s)
    f4 = MIMOFeedback(sys1, controller)
    assert f4.args == (sys1, controller, -1)
    assert f4.sys1 == f4.sys2 == sys1


def test_MIMOFeedback_errors():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s**3 - 1, s)
    tf3 = TransferFunction(s, s - 1, s)
    tf4 = TransferFunction(s, s**2 + 1, s)
    tf5 = TransferFunction(1, 1, s)
    tf6 = TransferFunction(-1, s - 1, s)

    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf3], [tf4, tf1]])
    tfm_3 = TransferFunctionMatrix.from_Matrix(eye(2), var=s)
    tfm_4 = TransferFunctionMatrix([[tf1, tf5], [tf5, tf5]])
    tfm_5 = TransferFunctionMatrix([[-tf3, tf3], [tf3, tf6]])
    # tfm_4 is inverse of tfm_5. Therefore tfm_5*tfm_4 = I
    tfm_6 = TransferFunctionMatrix([[-tf3]])
    tfm_7 = TransferFunctionMatrix([[tf3, tf4]])

    # Unsupported Types
    raises(TypeError, lambda: MIMOFeedback(tf1, tf2))
    raises(TypeError, lambda: MIMOFeedback(MIMOParallel(tfm_1, tfm_2), tfm_3))
    # Shape Errors
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_6, 1))
    raises(ValueError, lambda: MIMOFeedback(tfm_7, tfm_7))
    # sign not 1/-1
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_2, -2))
    # Non-Invertible Systems
    raises(ValueError, lambda: MIMOFeedback(tfm_5, tfm_4, 1))
    raises(ValueError, lambda: MIMOFeedback(tfm_4, -tfm_5))
    raises(ValueError, lambda: MIMOFeedback(tfm_3, tfm_3, 1))
    # Variable not same in both the systems
    tfm_8 = TransferFunctionMatrix.from_Matrix(eye(2), var=p)
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_8, 1))


def test_MIMOFeedback_functions():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s - 1, s)
    tf3 = TransferFunction(1, 1, s)
    tf4 = TransferFunction(-1, s - 1, s)

    tfm_1 = TransferFunctionMatrix.from_Matrix(eye(2), var=s)
    tfm_2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf3]])
    tfm_3 = TransferFunctionMatrix([[-tf2, tf2], [tf2, tf4]])
    tfm_4 = TransferFunctionMatrix([[tf1, tf2], [-tf2, tf1]])

    # sensitivity, doit(), rewrite()
    F_1 = MIMOFeedback(tfm_2, tfm_3)
    F_2 = MIMOFeedback(tfm_2, MIMOSeries(tfm_4, -tfm_1), 1)

    assert F_1.sensitivity == Matrix([[S.Half, 0], [0, S.Half]])
    assert F_2.sensitivity == Matrix([[(-2*s**4 + s**2)/(s**2 - s + 1),
        (2*s**3 - s**2)/(s**2 - s + 1)], [-s**2, s]])

    assert F_1.doit() == \
        TransferFunctionMatrix(((TransferFunction(1, 2*s, s),
        TransferFunction(1, 2, s)), (TransferFunction(1, 2, s),
        TransferFunction(1, 2, s)))) == F_1.rewrite(TransferFunctionMatrix)
    assert F_2.doit(cancel=False, expand=True) == \
        TransferFunctionMatrix(((TransferFunction(-s**5 + 2*s**4 - 2*s**3 + s**2, s**5 - 2*s**4 + 3*s**3 - 2*s**2 + s, s),
        TransferFunction(-2*s**4 + 2*s**3, s**2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(-s**2 + s, 1, s))))
    assert F_2.doit(cancel=False) == \
        TransferFunctionMatrix(((TransferFunction(s*(2*s**3 - s**2)*(s**2 - s + 1) + \
        (-2*s**4 + s**2)*(s**2 - s + 1), s*(s**2 - s + 1)**2, s), TransferFunction(-2*s**4 + 2*s**3, s**2 - s + 1, s)),
        (TransferFunction(0, 1, s), TransferFunction(-s**2 + s, 1, s))))
    assert F_2.doit() == \
        TransferFunctionMatrix(((TransferFunction(s*(-2*s**2 + s*(2*s - 1) + 1), s**2 - s + 1, s),
        TransferFunction(-2*s**3*(s - 1), s**2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(s*(1 - s), 1, s))))
    assert F_2.doit(expand=True) == \
        TransferFunctionMatrix(((TransferFunction(-s**2 + s, s**2 - s + 1, s), TransferFunction(-2*s**4 + 2*s**3, s**2 - s + 1, s)),
        (TransferFunction(0, 1, s), TransferFunction(-s**2 + s, 1, s))))

    assert -(F_1.doit()) == (-F_1).doit()  # First negating then calculating vs calculating then negating.


def test_TransferFunctionMatrix_construction():
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)

    tfm3_ = TransferFunctionMatrix([[-TF3]])
    assert tfm3_.shape == (tfm3_.num_outputs, tfm3_.num_inputs) == (1, 1)
    assert tfm3_.args == Tuple(Tuple(Tuple(-TF3)))
    assert tfm3_.var == s

    tfm5 = TransferFunctionMatrix([[TF1, -TF2], [TF3, tf5]])
    assert tfm5.shape == (tfm5.num_outputs, tfm5.num_inputs) == (2, 2)
    assert tfm5.args == Tuple(Tuple(Tuple(TF1, -TF2), Tuple(TF3, tf5)))
    assert tfm5.var == s

    tfm7 = TransferFunctionMatrix([[TF1, TF2], [TF3, -tf5], [-tf5, TF2]])
    assert tfm7.shape == (tfm7.num_outputs, tfm7.num_inputs) == (3, 2)
    assert tfm7.args == Tuple(Tuple(Tuple(TF1, TF2), Tuple(TF3, -tf5), Tuple(-tf5, TF2)))
    assert tfm7.var == s

    # all transfer functions will use the same complex variable. tf4 uses 'p'.
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1], [TF2], [tf4]]))
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1, tf4], [TF3, tf5]]))

    # length of all the lists in the TFM should be equal.
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1], [TF3, tf5]]))
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1, TF3], [tf5]]))

    # lists should only support transfer functions in them.
    raises(TypeError, lambda: TransferFunctionMatrix([[TF1, TF2], [TF3, Matrix([1, 2])]]))
    raises(TypeError, lambda: TransferFunctionMatrix([[TF1, Matrix([1, 2])], [TF3, TF2]]))

    # `arg` should strictly be nested list of TransferFunction
    raises(ValueError, lambda: TransferFunctionMatrix([TF1, TF2, tf5]))
    raises(ValueError, lambda: TransferFunctionMatrix([TF1]))

def test_TransferFunctionMatrix_functions():
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    #  Classmethod (from_matrix)

    mat_1 = ImmutableMatrix([
        [s*(s + 1)*(s - 3)/(s**4 + 1), 2],
        [p, p*(s + 1)/(s*(s**1 + 1))]
        ])
    mat_2 = ImmutableMatrix([[(2*s + 1)/(s**2 - 9)]])
    mat_3 = ImmutableMatrix([[1, 2], [3, 4]])
    assert TransferFunctionMatrix.from_Matrix(mat_1, s) == \
        TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)],
        [TransferFunction(p, 1, s), TransferFunction(p, s, s)]])
    assert TransferFunctionMatrix.from_Matrix(mat_2, s) == \
        TransferFunctionMatrix([[TransferFunction(2*s + 1, s**2 - 9, s)]])
    assert TransferFunctionMatrix.from_Matrix(mat_3, p) == \
        TransferFunctionMatrix([[TransferFunction(1, 1, p), TransferFunction(2, 1, p)],
        [TransferFunction(3, 1, p), TransferFunction(4, 1, p)]])

    #  Negating a TFM

    tfm1 = TransferFunctionMatrix([[TF1], [TF2]])
    assert -tfm1 == TransferFunctionMatrix([[-TF1], [-TF2]])

    tfm2 = TransferFunctionMatrix([[TF1, TF2, TF3], [tf5, -TF1, -TF3]])
    assert -tfm2 == TransferFunctionMatrix([[-TF1, -TF2, -TF3], [-tf5, TF1, TF3]])

    # subs()

    H_1 = TransferFunctionMatrix.from_Matrix(mat_1, s)
    H_2 = TransferFunctionMatrix([[TransferFunction(a*p*s, k*s**2, s), TransferFunction(p*s, k*(s**2 - a), s)]])
    assert H_1.subs(p, 1) == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]])
    assert H_1.subs({p: 1}) == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]])
    assert H_1.subs({p: 1, s: 1}) == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]]) # This should ignore `s` as it is `var`
    assert H_2.subs(p, 2) == TransferFunctionMatrix([[TransferFunction(2*a*s, k*s**2, s), TransferFunction(2*s, k*(-a + s**2), s)]])
    assert H_2.subs(k, 1) == TransferFunctionMatrix([[TransferFunction(a*p*s, s**2, s), TransferFunction(p*s, -a + s**2, s)]])
    assert H_2.subs(a, 0) == TransferFunctionMatrix([[TransferFunction(0, k*s**2, s), TransferFunction(p*s, k*s**2, s)]])
    assert H_2.subs({p: 1, k: 1, a: a0}) == TransferFunctionMatrix([[TransferFunction(a0*s, s**2, s), TransferFunction(s, -a0 + s**2, s)]])

    # eval_frequency()
    assert H_2.eval_frequency(S(1)/2 + I) == Matrix([[2*a*p/(5*k) - 4*I*a*p/(5*k), I*p/(-a*k - 3*k/4 + I*k) + p/(-2*a*k - 3*k/2 + 2*I*k)]])

    # transpose()

    assert H_1.transpose() == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(p, 1, s)], [TransferFunction(2, 1, s), TransferFunction(p, s, s)]])
    assert H_2.transpose() == TransferFunctionMatrix([[TransferFunction(a*p*s, k*s**2, s)], [TransferFunction(p*s, k*(-a + s**2), s)]])
    assert H_1.transpose().transpose() == H_1
    assert H_2.transpose().transpose() == H_2

    # elem_poles()

    assert H_1.elem_poles() == [[[-sqrt(2)/2 - sqrt(2)*I/2, -sqrt(2)/2 + sqrt(2)*I/2, sqrt(2)/2 - sqrt(2)*I/2, sqrt(2)/2 + sqrt(2)*I/2], []],
        [[], [0]]]
    assert H_2.elem_poles() == [[[0, 0], [sqrt(a), -sqrt(a)]]]
    assert tfm2.elem_poles() == [[[wn*(-zeta + sqrt((zeta - 1)*(zeta + 1))), wn*(-zeta - sqrt((zeta - 1)*(zeta + 1)))], [], [-p/a2]],
        [[-a0], [wn*(-zeta + sqrt((zeta - 1)*(zeta + 1))), wn*(-zeta - sqrt((zeta - 1)*(zeta + 1)))], [-p/a2]]]

    # elem_zeros()

    assert H_1.elem_zeros() == [[[-1, 0, 3], []], [[], []]]
    assert H_2.elem_zeros() == [[[0], [0]]]
    assert tfm2.elem_zeros() == [[[], [], [a2*p]],
        [[-a2/(2*a1) - sqrt(4*a0*a1 + a2**2)/(2*a1), -a2/(2*a1) + sqrt(4*a0*a1 + a2**2)/(2*a1)], [], [a2*p]]]

    # doit()

    H_3 = TransferFunctionMatrix([[Series(TransferFunction(1, s**3 - 3, s), TransferFunction(s**2 - 2*s + 5, 1, s), TransferFunction(1, s, s))]])
    H_4 = TransferFunctionMatrix([[Parallel(TransferFunction(s**3 - 3, 4*s**4 - s**2 - 2*s + 5, s), TransferFunction(4 - s**3, 4*s**4 - s**2 - 2*s + 5, s))]])

    assert H_3.doit() == TransferFunctionMatrix([[TransferFunction(s**2 - 2*s + 5, s*(s**3 - 3), s)]])
    assert H_4.doit() == TransferFunctionMatrix([[TransferFunction(1, 4*s**4 - s**2 - 2*s + 5, s)]])

    # _flat()

    assert H_1._flat() == [TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s), TransferFunction(p, 1, s), TransferFunction(p, s, s)]
    assert H_2._flat() == [TransferFunction(a*p*s, k*s**2, s), TransferFunction(p*s, k*(-a + s**2), s)]
    assert H_3._flat() == [Series(TransferFunction(1, s**3 - 3, s), TransferFunction(s**2 - 2*s + 5, 1, s), TransferFunction(1, s, s))]
    assert H_4._flat() == [Parallel(TransferFunction(s**3 - 3, 4*s**4 - s**2 - 2*s + 5, s), TransferFunction(4 - s**3, 4*s**4 - s**2 - 2*s + 5, s))]

    # evalf()

    assert H_1.evalf() == \
        TransferFunctionMatrix(((TransferFunction(s*(s - 3.0)*(s + 1.0), s**4 + 1.0, s), TransferFunction(2.0, 1, s)), (TransferFunction(1.0*p, 1, s), TransferFunction(p, s, s))))
    assert H_2.subs({a:3.141, p:2.88, k:2}).evalf() == \
        TransferFunctionMatrix(((TransferFunction(4.5230399999999999494093572138808667659759521484375, s, s),
        TransferFunction(2.87999999999999989341858963598497211933135986328125*s, 2.0*s**2 - 6.282000000000000028421709430404007434844970703125, s)),))

    # simplify()

    H_5 = TransferFunctionMatrix([[TransferFunction(s**5 + s**3 + s, s - s**2, s),
        TransferFunction((s + 3)*(s - 1), (s - 1)*(s + 5), s)]])

    assert H_5.simplify() == simplify(H_5) == \
        TransferFunctionMatrix(((TransferFunction(-s**4 - s**2 - 1, s - 1, s), TransferFunction(s + 3, s + 5, s)),))

    # expand()

    assert (H_1.expand()
            == TransferFunctionMatrix(((TransferFunction(s**3 - 2*s**2 - 3*s, s**4 + 1, s), TransferFunction(2, 1, s)),
            (TransferFunction(p, 1, s), TransferFunction(p, s, s)))))
    assert H_5.expand() == \
        TransferFunctionMatrix(((TransferFunction(s**5 + s**3 + s, -s**2 + s, s), TransferFunction(s**2 + 2*s - 3, s**2 + 4*s - 5, s)),))

def test_TransferFunction_gbt():
    # simple transfer function, e.g. ohms law
    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = gbt(tf, T, 0.5)
    # discretized transfer function with coefs from tf.gbt()
    tf_test_bilinear = TransferFunction(s * numZ[0] + numZ[1], s * denZ[0] + denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s * T/(2*(a + b*T/2)) + T/(2*(a + b*T/2)), s + (-a + b*T/2)/(a + b*T/2), s)

    assert S.Zero == (tf_test_bilinear.simplify()-tf_test_manual.simplify()).simplify().num

    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = gbt(tf, T, 0)
    # discretized transfer function with coefs from tf.gbt()
    tf_test_forward = TransferFunction(numZ[0], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(T/a, s + (-a + b*T)/a, s)

    assert S.Zero == (tf_test_forward.simplify()-tf_test_manual.simplify()).simplify().num

    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = gbt(tf, T, 1)
    # discretized transfer function with coefs from tf.gbt()
    tf_test_backward = TransferFunction(s*numZ[0], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s * T/(a + b*T), s - a/(a + b*T), s)

    assert S.Zero == (tf_test_backward.simplify()-tf_test_manual.simplify()).simplify().num

    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = gbt(tf, T, 0.3)
    # discretized transfer function with coefs from tf.gbt()
    tf_test_gbt = TransferFunction(s*numZ[0]+numZ[1], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s*3*T/(10*(a + 3*b*T/10)) + 7*T/(10*(a + 3*b*T/10)), s + (-a + 7*b*T/10)/(a + 3*b*T/10), s)

    assert S.Zero == (tf_test_gbt.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_bilinear():
    # simple transfer function, e.g. ohms law
    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = bilinear(tf, T)
    # discretized transfer function with coefs from tf.bilinear()
    tf_test_bilinear = TransferFunction(s*numZ[0]+numZ[1], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s * T/(2*(a + b*T/2)) + T/(2*(a + b*T/2)), s + (-a + b*T/2)/(a + b*T/2), s)

    assert S.Zero == (tf_test_bilinear.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_forward_diff():
    # simple transfer function, e.g. ohms law
    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = forward_diff(tf, T)
    # discretized transfer function with coefs from tf.forward_diff()
    tf_test_forward = TransferFunction(numZ[0], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(T/a, s + (-a + b*T)/a, s)

    assert S.Zero == (tf_test_forward.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_backward_diff():
    # simple transfer function, e.g. ohms law
    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = backward_diff(tf, T)
    # discretized transfer function with coefs from tf.backward_diff()
    tf_test_backward = TransferFunction(s*numZ[0]+numZ[1], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s * T/(a + b*T), s - a/(a + b*T), s)

    assert S.Zero == (tf_test_backward.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_phase_margin():
    # Test for phase margin
    tf1 = TransferFunction(10, p**3 + 1, p)
    tf2 = TransferFunction(s**2, 10, s)
    tf3 = TransferFunction(1, a*s+b, s)
    tf4 = TransferFunction((s + 1)*exp(s/tau), s**2 + 2, s)
    tf_m = TransferFunctionMatrix([[tf2],[tf3]])

    assert phase_margin(tf1) == -180 + 180*atan(3*sqrt(11))/pi
    assert phase_margin(tf2) == 0

    raises(NotImplementedError, lambda: phase_margin(tf4))
    raises(ValueError, lambda: phase_margin(tf3))
    raises(ValueError, lambda: phase_margin(MIMOSeries(tf_m)))

def test_TransferFunction_gain_margin():
    # Test for gain margin
    tf1 = TransferFunction(s**2, 5*(s+1)*(s-5)*(s-10), s)
    tf2 = TransferFunction(s**2 + 2*s + 1, 1, s)
    tf3 = TransferFunction(1, a*s+b, s)
    tf4 = TransferFunction((s + 1)*exp(s/tau), s**2 + 2, s)
    tf_m = TransferFunctionMatrix([[tf2],[tf3]])

    assert gain_margin(tf1) == -20*log(S(7)/540)/log(10)
    assert gain_margin(tf2) == oo

    raises(NotImplementedError, lambda: gain_margin(tf4))
    raises(ValueError, lambda: gain_margin(tf3))
    raises(ValueError, lambda: gain_margin(MIMOSeries(tf_m)))


def test_StateSpace_construction():
    # using different numbers for a SISO system.
    A1 = Matrix([[0, 1], [1, 0]])
    B1 = Matrix([1, 0])
    C1 = Matrix([[0, 1]])
    D1 = Matrix([0])
    ss1 = StateSpace(A1, B1, C1, D1)

    assert ss1.state_matrix == Matrix([[0, 1], [1, 0]])
    assert ss1.input_matrix == Matrix([1, 0])
    assert ss1.output_matrix == Matrix([[0, 1]])
    assert ss1.feedforward_matrix == Matrix([0])
    assert ss1.args == (Matrix([[0, 1], [1, 0]]), Matrix([[1], [0]]), Matrix([[0, 1]]), Matrix([[0]]))

    # using different symbols for a SISO system.
    ss2 = StateSpace(Matrix([a0]), Matrix([a1]),
                    Matrix([a2]), Matrix([a3]))

    assert ss2.state_matrix == Matrix([[a0]])
    assert ss2.input_matrix == Matrix([[a1]])
    assert ss2.output_matrix == Matrix([[a2]])
    assert ss2.feedforward_matrix == Matrix([[a3]])
    assert ss2.args == (Matrix([[a0]]), Matrix([[a1]]), Matrix([[a2]]), Matrix([[a3]]))

    # using different numbers for a MIMO system.
    ss3 = StateSpace(Matrix([[-1.5, -2], [1, 0]]),
                    Matrix([[0.5, 0], [0, 1]]),
                    Matrix([[0, 1], [0, 2]]),
                    Matrix([[2, 2], [1, 1]]))

    assert ss3.state_matrix == Matrix([[-1.5, -2], [1,  0]])
    assert ss3.input_matrix == Matrix([[0.5, 0], [0, 1]])
    assert ss3.output_matrix == Matrix([[0, 1], [0, 2]])
    assert ss3.feedforward_matrix == Matrix([[2, 2], [1, 1]])
    assert ss3.args == (Matrix([[-1.5, -2],
                                [1,  0]]),
                        Matrix([[0.5, 0],
                                [0, 1]]),
                        Matrix([[0, 1],
                                [0, 2]]),
                        Matrix([[2, 2],
                                [1, 1]]))

    # using different symbols for a MIMO system.
    A4 = Matrix([[a0, a1], [a2, a3]])
    B4 = Matrix([[b0, b1], [b2, b3]])
    C4 = Matrix([[c0, c1], [c2, c3]])
    D4 = Matrix([[d0, d1], [d2, d3]])
    ss4 = StateSpace(A4, B4, C4, D4)

    assert ss4.state_matrix == Matrix([[a0, a1], [a2, a3]])
    assert ss4.input_matrix == Matrix([[b0, b1], [b2, b3]])
    assert ss4.output_matrix == Matrix([[c0, c1], [c2, c3]])
    assert ss4.feedforward_matrix == Matrix([[d0, d1], [d2, d3]])
    assert ss4.args == (Matrix([[a0, a1],
                                [a2, a3]]),
                        Matrix([[b0, b1],
                                [b2, b3]]),
                        Matrix([[c0, c1],
                                [c2, c3]]),
                        Matrix([[d0, d1],
                                [d2, d3]]))

    # using less matrices. Rest will be filled with a minimum of zeros.
    ss5 = StateSpace()
    assert ss5.args == (Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[0]]))

    A6 = Matrix([[0, 1], [1, 0]])
    B6 = Matrix([1, 1])
    ss6 = StateSpace(A6, B6)

    assert ss6.state_matrix == Matrix([[0, 1], [1, 0]])
    assert ss6.input_matrix ==  Matrix([1, 1])
    assert ss6.output_matrix == Matrix([[0, 0]])
    assert ss6.feedforward_matrix == Matrix([[0]])
    assert ss6.args == (Matrix([[0, 1],
                                [1, 0]]),
                        Matrix([[1],
                                [1]]),
                        Matrix([[0, 0]]),
                        Matrix([[0]]))

    # Check if the system is SISO or MIMO.
    # If system is not SISO, then it is definitely MIMO.

    assert ss1.is_SISO == True
    assert ss2.is_SISO == True
    assert ss3.is_SISO == False
    assert ss4.is_SISO == False
    assert ss5.is_SISO == True
    assert ss6.is_SISO == True

    # ShapeError if matrices do not fit.
    raises(ShapeError, lambda: StateSpace(Matrix([s, (s+1)**2]), Matrix([s+1]),
                                          Matrix([s**2 - 1]), Matrix([2*s])))
    raises(ShapeError, lambda: StateSpace(Matrix([s]), Matrix([s+1, s**3 + 1]),
                                          Matrix([s**2 - 1]), Matrix([2*s])))
    raises(ShapeError, lambda: StateSpace(Matrix([s]), Matrix([s+1]),
                                          Matrix([[s**2 - 1], [s**2 + 2*s + 1]]), Matrix([2*s])))
    raises(ShapeError, lambda: StateSpace(Matrix([[-s, -s], [s, 0]]),
                                                Matrix([[s/2, 0], [0, s]]),
                                                Matrix([[0, s]]),
                                                Matrix([[2*s, 2*s], [s, s]])))

    # TypeError if arguments are not sympy matrices.
    raises(TypeError, lambda: StateSpace(s**2, s+1, 2*s, 1))
    raises(TypeError, lambda: StateSpace(Matrix([2, 0.5]), Matrix([-1]),
                                         Matrix([1]), 0))
def test_StateSpace_add():
    A1 = Matrix([[4, 1],[2, -3]])
    B1 = Matrix([[5, 2],[-3, -3]])
    C1 = Matrix([[2, -4],[0, 1]])
    D1 = Matrix([[3, 2],[1, -1]])
    ss1 = StateSpace(A1, B1, C1, D1)

    A2 = Matrix([[-3, 4, 2],[-1, -3, 0],[2, 5, 3]])
    B2 = Matrix([[1, 4],[-3, -3],[-2, 1]])
    C2 = Matrix([[4, 2, -3],[1, 4, 3]])
    D2 = Matrix([[-2, 4],[0, 1]])
    ss2 = StateSpace(A2, B2, C2, D2)
    ss3 = StateSpace()
    ss4 = StateSpace(Matrix([1]), Matrix([2]), Matrix([3]), Matrix([4]))

    expected_add = \
        StateSpace(
        Matrix([
        [4,  1,  0,  0, 0],
        [2, -3,  0,  0, 0],
        [0,  0, -3,  4, 2],
        [0,  0, -1, -3, 0],
        [0,  0,  2,  5, 3]]),
        Matrix([
        [ 5,  2],
        [-3, -3],
        [ 1,  4],
        [-3, -3],
        [-2,  1]]),
        Matrix([
        [2, -4, 4, 2, -3],
        [0,  1, 1, 4,  3]]),
        Matrix([
        [1, 6],
        [1, 0]]))

    expected_mul = \
        StateSpace(
        Matrix([
        [ -3,   4,  2, 0,  0],
        [ -1,  -3,  0, 0,  0],
        [  2,   5,  3, 0,  0],
        [ 22,  18, -9, 4,  1],
        [-15, -18,  0, 2, -3]]),
        Matrix([
        [  1,   4],
        [ -3,  -3],
        [ -2,   1],
        [-10,  22],
        [  6, -15]]),
        Matrix([
        [14, 14, -3, 2, -4],
        [ 3, -2, -6, 0,  1]]),
        Matrix([
        [-6, 14],
        [-2,  3]]))

    assert ss1 + ss2 == expected_add
    assert ss1*ss2 == expected_mul
    assert ss3 + 1/2 == StateSpace(Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[0.5]]))
    assert ss4*1.5 == StateSpace(Matrix([[1]]), Matrix([[2]]), Matrix([[4.5]]), Matrix([[6.0]]))
    assert 1.5*ss4 == StateSpace(Matrix([[1]]), Matrix([[3.0]]), Matrix([[3]]), Matrix([[6.0]]))
    raises(ShapeError, lambda: ss1 + ss3)
    raises(ShapeError, lambda: ss2*ss4)

def test_StateSpace_negation():
    A = Matrix([[a0, a1], [a2, a3]])
    B = Matrix([[b0, b1], [b2, b3]])
    C = Matrix([[c0, c1], [c1, c2], [c2, c3]])
    D = Matrix([[d0, d1], [d1, d2], [d2, d3]])
    SS = StateSpace(A, B, C, D)
    SS_neg = -SS

    state_mat = Matrix([[-1, 1], [1, -1]])
    input_mat = Matrix([1, -1])
    output_mat = Matrix([[-1, 1]])
    feedforward_mat = Matrix([1])
    system = StateSpace(state_mat, input_mat, output_mat, feedforward_mat)

    assert SS_neg == \
        StateSpace(Matrix([[a0, a1],
                           [a2, a3]]),
                   Matrix([[b0, b1],
                           [b2, b3]]),
                   Matrix([[-c0, -c1],
                           [-c1, -c2],
                           [-c2, -c3]]),
                   Matrix([[-d0, -d1],
                           [-d1, -d2],
                           [-d2, -d3]]))
    assert -system == \
        StateSpace(Matrix([[-1,  1],
                           [ 1, -1]]),
                   Matrix([[ 1],[-1]]),
                   Matrix([[1, -1]]),
                   Matrix([[-1]]))
    assert -SS_neg == SS
    assert -(-(-(-system))) == system

def test_SymPy_substitution_functions():
    # subs
    ss1 = StateSpace(Matrix([s]), Matrix([(s + 1)**2]), Matrix([s**2 - 1]), Matrix([2*s]))
    ss2 = StateSpace(Matrix([s + p]), Matrix([(s + 1)*(p - 1)]), Matrix([p**3 - s**3]), Matrix([s - p]))

    assert ss1.subs({s:5}) == StateSpace(Matrix([[5]]), Matrix([[36]]), Matrix([[24]]), Matrix([[10]]))
    assert ss2.subs({p:1}) == StateSpace(Matrix([[s + 1]]), Matrix([[0]]), Matrix([[1 - s**3]]), Matrix([[s - 1]]))

    # xreplace
    assert ss1.xreplace({s:p}) == \
        StateSpace(Matrix([[p]]), Matrix([[(p + 1)**2]]), Matrix([[p**2 - 1]]), Matrix([[2*p]]))
    assert ss2.xreplace({s:a, p:b}) == \
        StateSpace(Matrix([[a + b]]), Matrix([[(a + 1)*(b - 1)]]), Matrix([[-a**3 + b**3]]), Matrix([[a - b]]))

    # evalf
    p1 = a1*s + a0
    p2 = b2*s**2 + b1*s + b0
    G = StateSpace(Matrix([p1]), Matrix([p2]))
    expect = StateSpace(Matrix([[2*s + 1]]), Matrix([[5*s**2 + 4*s + 3]]), Matrix([[0]]), Matrix([[0]]))
    expect_ = StateSpace(Matrix([[2.0*s + 1.0]]), Matrix([[5.0*s**2 + 4.0*s + 3.0]]), Matrix([[0]]), Matrix([[0]]))
    assert G.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}) == expect
    assert G.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}).evalf() == expect_
    assert expect.evalf() == expect_

def test_conversion():
    # StateSpace to TransferFunction for SISO
    A1 = Matrix([[-5, -1], [3, -1]])
    B1 = Matrix([2, 5])
    C1 = Matrix([[1, 2]])
    D1 = Matrix([0])
    H1 = StateSpace(A1, B1, C1, D1)
    tm1 = H1.rewrite(TransferFunction)
    tm2 = (-H1).rewrite(TransferFunction)

    tf1 = tm1[0][0]
    tf2 = tm2[0][0]

    assert tf1 == TransferFunction(12*s + 59, s**2 + 6*s + 8, s)
    assert tf2.num == -tf1.num
    assert tf2.den == tf1.den

    # StateSpace to TransferFunction for MIMO
    A2 = Matrix([[-1.5, -2, 3], [1, 0, 1], [2, 1, 1]])
    B2 = Matrix([[0.5, 0, 1], [0, 1, 2], [2, 2, 3]])
    C2 = Matrix([[0, 1, 0], [0, 2, 1], [1, 0, 2]])
    D2 = Matrix([[2, 2, 0], [1, 1, 1], [3, 2, 1]])
    H2 = StateSpace(A2, B2, C2, D2)
    tm3 = H2.rewrite(TransferFunction)

    # outputs for input i obtained at Index i-1. Consider input 1
    assert tm3[0][0] == TransferFunction(2.0*s**3 + 1.0*s**2 - 10.5*s + 4.5, 1.0*s**3 + 0.5*s**2 - 6.5*s - 2.5, s)
    assert tm3[0][1] == TransferFunction(2.0*s**3 + 2.0*s**2 - 10.5*s - 3.5, 1.0*s**3 + 0.5*s**2 - 6.5*s - 2.5, s)
    assert tm3[0][2] == TransferFunction(2.0*s**2 + 5.0*s - 0.5, 1.0*s**3 + 0.5*s**2 - 6.5*s - 2.5, s)

    # TransferFunction to StateSpace
    SS = TF1.rewrite(StateSpace)
    assert SS == \
        StateSpace(Matrix([[     0,          1],
                           [-wn**2, -2*wn*zeta]]),
                   Matrix([[0],
                           [1]]),
                   Matrix([[1, 0]]),
                   Matrix([[0]]))
    assert SS.rewrite(TransferFunction)[0][0] == TF1

    # Transfer function has to be proper
    raises(ValueError, lambda: TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s).rewrite(StateSpace))


def test_StateSpace_functions():
    # https://in.mathworks.com/help/control/ref/statespacemodel.obsv.html

    A_mat = Matrix([[-1.5, -2], [1, 0]])
    B_mat = Matrix([0.5, 0])
    C_mat = Matrix([[0, 1]])
    D_mat = Matrix([1])
    SS1 = StateSpace(A_mat, B_mat, C_mat, D_mat)
    SS2 = StateSpace(Matrix([[1, 1], [4, -2]]),Matrix([[0, 1], [0, 2]]),Matrix([[-1, 1], [1, -1]]))
    SS3 = StateSpace(Matrix([[1, 1], [4, -2]]),Matrix([[1, -1], [1, -1]]))

    # Observability
    assert SS1.is_observable() == True
    assert SS2.is_observable() == False
    assert SS1.observability_matrix() == Matrix([[0, 1], [1, 0]])
    assert SS2.observability_matrix() == Matrix([[-1,  1], [ 1, -1], [ 3, -3], [-3,  3]])
    assert SS1.observable_subspace() == [Matrix([[0], [1]]), Matrix([[1], [0]])]
    assert SS2.observable_subspace() == [Matrix([[-1], [ 1], [ 3], [-3]])]

    # Controllability
    assert SS1.is_controllable() == True
    assert SS3.is_controllable() == False
    assert SS1.controllability_matrix() ==  Matrix([[0.5, -0.75], [  0,   0.5]])
    assert SS3.controllability_matrix() == Matrix([[1, -1, 2, -2], [1, -1, 2, -2]])
    assert SS1.controllable_subspace() == [Matrix([[0.5], [  0]]), Matrix([[-0.75], [  0.5]])]
    assert SS3.controllable_subspace() == [Matrix([[1], [1]])]

    # Append
    A1 = Matrix([[0, 1], [1, 0]])
    B1 = Matrix([[0], [1]])
    C1 = Matrix([[0, 1]])
    D1 = Matrix([[0]])
    ss1 = StateSpace(A1, B1, C1, D1)
    ss2 = StateSpace(Matrix([[1, 0], [0, 1]]), Matrix([[1], [0]]), Matrix([[1, 0]]), Matrix([[1]]))
    ss3 = ss1.append(ss2)

    assert ss3.num_states == ss1.num_states + ss2.num_states
    assert ss3.num_inputs == ss1.num_inputs + ss2.num_inputs
    assert ss3.num_outputs == ss1.num_outputs + ss2.num_outputs
    assert ss3.state_matrix == Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert ss3.input_matrix == Matrix([[0, 0], [1, 0], [0, 1], [0, 0]])
    assert ss3.output_matrix == Matrix([[0, 1, 0, 0], [0, 0, 1, 0]])
    assert ss3.feedforward_matrix == Matrix([[0, 0], [0, 1]])
