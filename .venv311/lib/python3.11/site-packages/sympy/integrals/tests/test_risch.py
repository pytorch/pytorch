"""Most of these tests come from the examples in Bronstein's book."""
from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
    derivation, splitfactor, splitfactor_sqf, canonical_representation,
    hermite_reduce, polynomial_reduce, residue_reduce, residue_reduce_to_basic,
    integrate_primitive, integrate_hyperexponential_polynomial,
    integrate_hyperexponential, integrate_hypertangent_polynomial,
    integrate_nonlinear_no_specials, integer_powers, DifferentialExtension,
    risch_integrate, DecrementLevel, NonElementaryIntegral, recognize_log_derivative,
    recognize_derivative, laurent_series)
from sympy.testing.pytest import raises

from sympy.abc import x, t, nu, z, a, y
t0, t1, t2 = symbols('t:3')
i = Symbol('i')

def test_gcdex_diophantine():
    assert gcdex_diophantine(Poly(x**4 - 2*x**3 - 6*x**2 + 12*x + 15),
    Poly(x**3 + x**2 - 4*x - 4), Poly(x**2 - 1)) == \
        (Poly((-x**2 + 4*x - 3)/5), Poly((x**3 - 7*x**2 + 16*x - 10)/5))
    assert gcdex_diophantine(Poly(x**3 + 6*x + 7), Poly(x**2 + 3*x + 2), Poly(x + 1)) == \
        (Poly(1/13, x, domain='QQ'), Poly(-1/13*x + 3/13, x, domain='QQ'))


def test_frac_in():
    assert frac_in(Poly((x + 1)/x*t, t), x) == \
        (Poly(t*x + t, x), Poly(x, x))
    assert frac_in((x + 1)/x*t, x) == \
        (Poly(t*x + t, x), Poly(x, x))
    assert frac_in((Poly((x + 1)/x*t, t), Poly(t + 1, t)), x) == \
        (Poly(t*x + t, x), Poly((1 + t)*x, x))
    raises(ValueError, lambda: frac_in((x + 1)/log(x)*t, x))
    assert frac_in(Poly((2 + 2*x + x*(1 + x))/(1 + x)**2, t), x, cancel=True) == \
        (Poly(x + 2, x), Poly(x + 1, x))


def test_as_poly_1t():
    assert as_poly_1t(2/t + t, t, z) in [
        Poly(t + 2*z, t, z), Poly(t + 2*z, z, t)]
    assert as_poly_1t(2/t + 3/t**2, t, z) in [
        Poly(2*z + 3*z**2, t, z), Poly(2*z + 3*z**2, z, t)]
    assert as_poly_1t(2/((exp(2) + 1)*t), t, z) in [
        Poly(2/(exp(2) + 1)*z, t, z), Poly(2/(exp(2) + 1)*z, z, t)]
    assert as_poly_1t(2/((exp(2) + 1)*t) + t, t, z) in [
        Poly(t + 2/(exp(2) + 1)*z, t, z), Poly(t + 2/(exp(2) + 1)*z, z, t)]
    assert as_poly_1t(S.Zero, t, z) == Poly(0, t, z)


def test_derivation():
    p = Poly(4*x**4*t**5 + (-4*x**3 - 4*x**4)*t**4 + (-3*x**2 + 2*x**3)*t**3 +
        (2*x + 7*x**2 + 2*x**3)*t**2 + (1 - 4*x - 4*x**2)*t - 1 + 2*x, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - 3/(2*x)*t + 1/(2*x), t)]})
    assert derivation(p, DE) == Poly(-20*x**4*t**6 + (2*x**3 + 16*x**4)*t**5 +
        (21*x**2 + 12*x**3)*t**4 + (x*Rational(7, 2) - 25*x**2 - 12*x**3)*t**3 +
        (-5 - x*Rational(15, 2) + 7*x**2)*t**2 - (3 - 8*x - 10*x**2 - 4*x**3)/(2*x)*t +
        (1 - 4*x**2)/(2*x), t)
    assert derivation(Poly(1, t), DE) == Poly(0, t)
    assert derivation(Poly(t, t), DE) == DE.d
    assert derivation(Poly(t**2 + 1/x*t + (1 - 2*x)/(4*x**2), t), DE) == \
        Poly(-2*t**3 - 4/x*t**2 - (5 - 2*x)/(2*x**2)*t - (1 - 2*x)/(2*x**3), t, domain='ZZ(x)')
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(t, t)]})
    assert derivation(Poly(x*t*t1, t), DE) == Poly(t*t1 + x*t*t1 + t, t)
    assert derivation(Poly(x*t*t1, t), DE, coefficientD=True) == \
        Poly((1 + t1)*t, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert derivation(Poly(x, x), DE) == Poly(1, x)
    # Test basic option
    assert derivation((x + 1)/(x - 1), DE, basic=True) == -2/(1 - 2*x + x**2)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert derivation((t + 1)/(t - 1), DE, basic=True) == -2*t/(1 - 2*t + t**2)
    assert derivation(t + 1, DE, basic=True) == t


def test_splitfactor():
    p = Poly(4*x**4*t**5 + (-4*x**3 - 4*x**4)*t**4 + (-3*x**2 + 2*x**3)*t**3 +
        (2*x + 7*x**2 + 2*x**3)*t**2 + (1 - 4*x - 4*x**2)*t - 1 + 2*x, t, field=True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - 3/(2*x)*t + 1/(2*x), t)]})
    assert splitfactor(p, DE) == (Poly(4*x**4*t**3 + (-8*x**3 - 4*x**4)*t**2 +
        (4*x**2 + 8*x**3)*t - 4*x**2, t, domain='ZZ(x)'),
        Poly(t**2 + 1/x*t + (1 - 2*x)/(4*x**2), t, domain='ZZ(x)'))
    assert splitfactor(Poly(x, t), DE) == (Poly(x, t), Poly(1, t))
    r = Poly(-4*x**4*z**2 + 4*x**6*z**2 - z*x**3 - 4*x**5*z**3 + 4*x**3*z**3 + x**4 + z*x**5 - x**6, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    assert splitfactor(r, DE, coefficientD=True) == \
        (Poly(x*z - x**2 - z*x**3 + x**4, t), Poly(-x**2 + 4*x**2*z**2, t))
    assert splitfactor_sqf(r, DE, coefficientD=True) == \
        (((Poly(x*z - x**2 - z*x**3 + x**4, t), 1),), ((Poly(-x**2 + 4*x**2*z**2, t), 1),))
    assert splitfactor(Poly(0, t), DE) == (Poly(0, t), Poly(1, t))
    assert splitfactor_sqf(Poly(0, t), DE) == (((Poly(0, t), 1),), ())


def test_canonical_representation():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    assert canonical_representation(Poly(x - t, t), Poly(t**2, t), DE) == \
        (Poly(0, t, domain='ZZ[x]'), (Poly(0, t, domain='QQ[x]'),
        Poly(1, t, domain='ZZ')), (Poly(-t + x, t, domain='QQ[x]'),
        Poly(t**2, t)))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert canonical_representation(Poly(t**5 + t**3 + x**2*t + 1, t),
    Poly((t**2 + 1)**3, t), DE) == \
        (Poly(0, t, domain='ZZ[x]'), (Poly(t**5 + t**3 + x**2*t + 1, t, domain='QQ[x]'),
         Poly(t**6 + 3*t**4 + 3*t**2 + 1, t, domain='QQ')),
        (Poly(0, t, domain='QQ[x]'), Poly(1, t, domain='QQ')))


def test_hermite_reduce():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})

    assert hermite_reduce(Poly(x - t, t), Poly(t**2, t), DE) == \
        ((Poly(-x, t, domain='QQ[x]'), Poly(t, t, domain='QQ[x]')),
         (Poly(0, t, domain='QQ[x]'), Poly(1, t, domain='QQ[x]')),
         (Poly(-x, t, domain='QQ[x]'), Poly(1, t, domain='QQ[x]')))

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - t/x - (1 - nu**2/x**2), t)]})

    assert hermite_reduce(
            Poly(x**2*t**5 + x*t**4 - nu**2*t**3 - x*(x**2 + 1)*t**2 - (x**2 - nu**2)*t - x**5/4, t),
            Poly(x**2*t**4 + x**2*(x**2 + 2)*t**2 + x**2 + x**4 + x**6/4, t), DE) == \
        ((Poly(-x**2 - 4, t, domain='ZZ(x,nu)'), Poly(4*t**2 + 2*x**2 + 4, t, domain='ZZ(x,nu)')),
         (Poly((-2*nu**2 - x**4)*t - (2*x**3 + 2*x), t, domain='ZZ(x,nu)'),
          Poly(2*x**2*t**2 + x**4 + 2*x**2, t, domain='ZZ(x,nu)')),
         (Poly(x*t + 1, t, domain='ZZ(x,nu)'), Poly(x, t, domain='ZZ(x,nu)')))

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})

    a = Poly((-2 + 3*x)*t**3 + (-1 + x)*t**2 + (-4*x + 2*x**2)*t + x**2, t)
    d = Poly(x*t**6 - 4*x**2*t**5 + 6*x**3*t**4 - 4*x**4*t**3 + x**5*t**2, t)

    assert hermite_reduce(a, d, DE) == \
        ((Poly(3*t**2 + t + 3*x, t, domain='ZZ(x)'),
          Poly(3*t**4 - 9*x*t**3 + 9*x**2*t**2 - 3*x**3*t, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')))

    assert hermite_reduce(
            Poly(-t**2 + 2*t + 2, t, domain='ZZ(x)'),
            Poly(-x*t**2 + 2*x*t - x, t, domain='ZZ(x)'), DE) == \
        ((Poly(3, t, domain='ZZ(x)'), Poly(t - 1, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(1, t, domain='ZZ(x)'), Poly(x, t, domain='ZZ(x)')))

    assert hermite_reduce(
            Poly(-x**2*t**6 + (-1 - 2*x**3 + x**4)*t**3 + (-3 - 3*x**4)*t**2 -
                2*x*t - x - 3*x**2, t, domain='ZZ(x)'),
            Poly(x**4*t**6 - 2*x**2*t**3 + 1, t, domain='ZZ(x)'), DE) == \
        ((Poly(x**3*t + x**4 + 1, t, domain='ZZ(x)'), Poly(x**3*t**3 - x, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(-1, t, domain='ZZ(x)'), Poly(x**2, t, domain='ZZ(x)')))

    assert hermite_reduce(
            Poly((-2 + 3*x)*t**3 + (-1 + x)*t**2 + (-4*x + 2*x**2)*t + x**2, t),
            Poly(x*t**6 - 4*x**2*t**5 + 6*x**3*t**4 - 4*x**4*t**3 + x**5*t**2, t), DE) == \
        ((Poly(3*t**2 + t + 3*x, t, domain='ZZ(x)'),
          Poly(3*t**4 - 9*x*t**3 + 9*x**2*t**2 - 3*x**3*t, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')))


def test_polynomial_reduce():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    assert polynomial_reduce(Poly(1 + x*t + t**2, t), DE) == \
        (Poly(t, t), Poly(x*t, t))
    assert polynomial_reduce(Poly(0, t), DE) == \
        (Poly(0, t), Poly(0, t))


def test_laurent_series():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1, t)]})
    a = Poly(36, t)
    d = Poly((t - 2)*(t**2 - 1)**2, t)
    F = Poly(t**2 - 1, t)
    n = 2
    assert laurent_series(a, d, F, n, DE) == \
        (Poly(-3*t**3 + 3*t**2 - 6*t - 8, t), Poly(t**5 + t**4 - 2*t**3 - 2*t**2 + t + 1, t),
        [Poly(-3*t**3 - 6*t**2, t, domain='QQ'), Poly(2*t**6 + 6*t**5 - 8*t**3, t, domain='QQ')])


def test_recognize_derivative():
    DE = DifferentialExtension(extension={'D': [Poly(1, t)]})
    a = Poly(36, t)
    d = Poly((t - 2)*(t**2 - 1)**2, t)
    assert recognize_derivative(a, d, DE) == False
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    a = Poly(2, t)
    d = Poly(t**2 - 1, t)
    assert recognize_derivative(a, d, DE) == False
    assert recognize_derivative(Poly(x*t, t), Poly(1, t), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert recognize_derivative(Poly(t, t), Poly(1, t), DE) == True


def test_recognize_log_derivative():

    a = Poly(2*x**2 + 4*x*t - 2*t - x**2*t, t)
    d = Poly((2*x + t)*(t + x**2), t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert recognize_log_derivative(a, d, DE, z) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    assert recognize_log_derivative(Poly(t + 1, t), Poly(t + x, t), DE) == True
    assert recognize_log_derivative(Poly(2, t), Poly(t**2 - 1, t), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert recognize_log_derivative(Poly(1, x), Poly(x**2 - 2, x), DE) == False
    assert recognize_log_derivative(Poly(1, x), Poly(x**2 + x, x), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert recognize_log_derivative(Poly(1, t), Poly(t**2 - 2, t), DE) == False
    assert recognize_log_derivative(Poly(1, t), Poly(t**2 + t, t), DE) == False


def test_residue_reduce():
    a = Poly(2*t**2 - t - x**2, t)
    d = Poly(t**3 - x**2*t, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)], 'Tfuncs': [log]})
    assert residue_reduce(a, d, DE, z, invert=False) == \
        ([(Poly(z**2 - Rational(1, 4), z, domain='ZZ(x)'),
          Poly((1 + 3*x*z - 6*z**2 - 2*x**2 + 4*x**2*z**2)*t - x*z + x**2 +
              2*x**2*z**2 - 2*z*x**3, t, domain='ZZ(z, x)'))], False)
    assert residue_reduce(a, d, DE, z, invert=True) == \
        ([(Poly(z**2 - Rational(1, 4), z, domain='ZZ(x)'), Poly(t + 2*x*z, t))], False)
    assert residue_reduce(Poly(-2/x, t), Poly(t**2 - 1, t,), DE, z, invert=False) == \
        ([(Poly(z**2 - 1, z, domain='QQ'), Poly(-2*z*t/x - 2/x, t, domain='ZZ(z,x)'))], True)
    ans = residue_reduce(Poly(-2/x, t), Poly(t**2 - 1, t), DE, z, invert=True)
    assert ans == ([(Poly(z**2 - 1, z, domain='QQ'), Poly(t + z, t))], True)
    assert residue_reduce_to_basic(ans[0], DE, z) == -log(-1 + log(x)) + log(1 + log(x))

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - t/x - (1 - nu**2/x**2), t)]})
    # TODO: Skip or make faster
    assert residue_reduce(Poly((-2*nu**2 - x**4)/(2*x**2)*t - (1 + x**2)/x, t),
    Poly(t**2 + 1 + x**2/2, t), DE, z) == \
        ([(Poly(z + S.Half, z, domain='QQ'), Poly(t**2 + 1 + x**2/2, t,
            domain='ZZ(x,nu)'))], True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    assert residue_reduce(Poly(-2*x*t + 1 - x**2, t),
    Poly(t**2 + 2*x*t + 1 + x**2, t), DE, z) == \
        ([(Poly(z**2 + Rational(1, 4), z), Poly(t + x + 2*z, t))], True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert residue_reduce(Poly(t, t), Poly(t + sqrt(2), t), DE, z) == \
        ([(Poly(z - 1, z, domain='QQ'), Poly(t + sqrt(2), t))], True)


def test_integrate_hyperexponential():
    # TODO: Add tests for integrate_hyperexponential() from the book
    a = Poly((1 + 2*t1 + t1**2 + 2*t1**3)*t**2 + (1 + t1**2)*t + 1 + t1**2, t)
    d = Poly(1, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t1**2, t1),
        Poly(t*(1 + t1**2), t)], 'Tfuncs': [tan, Lambda(i, exp(tan(i)))]})
    assert integrate_hyperexponential(a, d, DE) == \
        (exp(2*tan(x))*tan(x) + exp(tan(x)), 1 + t1**2, True)
    a = Poly((t1**3 + (x + 1)*t1**2 + t1 + x + 2)*t, t)
    assert integrate_hyperexponential(a, d, DE) == \
        ((x + tan(x))*exp(tan(x)), 0, True)

    a = Poly(t, t)
    d = Poly(1, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2*x*t, t)],
        'Tfuncs': [Lambda(i, exp(x**2))]})

    assert integrate_hyperexponential(a, d, DE) == \
        (0, NonElementaryIntegral(exp(x**2), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)], 'Tfuncs': [exp]})
    assert integrate_hyperexponential(a, d, DE) == (exp(x), 0, True)

    a = Poly(25*t**6 - 10*t**5 + 7*t**4 - 8*t**3 + 13*t**2 + 2*t - 1, t)
    d = Poly(25*t**6 + 35*t**4 + 11*t**2 + 1, t)
    assert integrate_hyperexponential(a, d, DE) == \
        (-(11 - 10*exp(x))/(5 + 25*exp(2*x)) + log(1 + exp(2*x)), -1, True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(t0*t, t)],
        'Tfuncs': [exp, Lambda(i, exp(exp(i)))]})
    assert integrate_hyperexponential(Poly(2*t0*t**2, t), Poly(1, t), DE) == (exp(2*exp(x)), 0, True)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(-t0*t, t)],
        'Tfuncs': [exp, Lambda(i, exp(-exp(i)))]})
    assert integrate_hyperexponential(Poly(-27*exp(9) - 162*t0*exp(9) +
    27*x*t0*exp(9), t), Poly((36*exp(18) + x**2*exp(18) - 12*x*exp(18))*t, t), DE) == \
        (27*exp(exp(x))/(-6*exp(9) + x*exp(9)), 0, True)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)], 'Tfuncs': [exp]})
    assert integrate_hyperexponential(Poly(x**2/2*t, t), Poly(1, t), DE) == \
        ((2 - 2*x + x**2)*exp(x)/2, 0, True)
    assert integrate_hyperexponential(Poly(1 + t, t), Poly(t, t), DE) == \
        (-exp(-x), 1, True)  # x - exp(-x)
    assert integrate_hyperexponential(Poly(x, t), Poly(t + 1, t), DE) == \
        (0, NonElementaryIntegral(x/(1 + exp(x)), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t0), Poly(2*x*t1, t1)],
        'Tfuncs': [log, Lambda(i, exp(i**2))]})

    elem, nonelem, b = integrate_hyperexponential(Poly((8*x**7 - 12*x**5 + 6*x**3 - x)*t1**4 +
        (8*t0*x**7 - 8*t0*x**6 - 4*t0*x**5 + 2*t0*x**3 + 2*t0*x**2 - t0*x +
        24*x**8 - 36*x**6 - 4*x**5 + 22*x**4 + 4*x**3 - 7*x**2 - x + 1)*t1**3
        + (8*t0*x**8 - 4*t0*x**6 - 16*t0*x**5 - 2*t0*x**4 + 12*t0*x**3 +
        t0*x**2 - 2*t0*x + 24*x**9 - 36*x**7 - 8*x**6 + 22*x**5 + 12*x**4 -
        7*x**3 - 6*x**2 + x + 1)*t1**2 + (8*t0*x**8 - 8*t0*x**6 - 16*t0*x**5 +
        6*t0*x**4 + 10*t0*x**3 - 2*t0*x**2 - t0*x + 8*x**10 - 12*x**8 - 4*x**7
        + 2*x**6 + 12*x**5 + 3*x**4 - 9*x**3 - x**2 + 2*x)*t1 + 8*t0*x**7 -
        12*t0*x**6 - 4*t0*x**5 + 8*t0*x**4 - t0*x**2 - 4*x**7 + 4*x**6 +
        4*x**5 - 4*x**4 - x**3 + x**2, t1), Poly((8*x**7 - 12*x**5 + 6*x**3 -
        x)*t1**4 + (24*x**8 + 8*x**7 - 36*x**6 - 12*x**5 + 18*x**4 + 6*x**3 -
        3*x**2 - x)*t1**3 + (24*x**9 + 24*x**8 - 36*x**7 - 36*x**6 + 18*x**5 +
        18*x**4 - 3*x**3 - 3*x**2)*t1**2 + (8*x**10 + 24*x**9 - 12*x**8 -
        36*x**7 + 6*x**6 + 18*x**5 - x**4 - 3*x**3)*t1 + 8*x**10 - 12*x**8 +
        6*x**6 - x**4, t1), DE)

    assert factor(elem) == -((x - 1)*log(x)/((x + exp(x**2))*(2*x**2 - 1)))
    assert (nonelem, b) == (NonElementaryIntegral(exp(x**2)/(exp(x**2) + 1), x), False)

def test_integrate_hyperexponential_polynomial():
    # Without proper cancellation within integrate_hyperexponential_polynomial(),
    # this will take a long time to complete, and will return a complicated
    # expression
    p = Poly((-28*x**11*t0 - 6*x**8*t0 + 6*x**9*t0 - 15*x**8*t0**2 +
        15*x**7*t0**2 + 84*x**10*t0**2 - 140*x**9*t0**3 - 20*x**6*t0**3 +
        20*x**7*t0**3 - 15*x**6*t0**4 + 15*x**5*t0**4 + 140*x**8*t0**4 -
        84*x**7*t0**5 - 6*x**4*t0**5 + 6*x**5*t0**5 + x**3*t0**6 - x**4*t0**6 +
        28*x**6*t0**6 - 4*x**5*t0**7 + x**9 - x**10 + 4*x**12)/(-8*x**11*t0 +
        28*x**10*t0**2 - 56*x**9*t0**3 + 70*x**8*t0**4 - 56*x**7*t0**5 +
        28*x**6*t0**6 - 8*x**5*t0**7 + x**4*t0**8 + x**12)*t1**2 +
        (-28*x**11*t0 - 12*x**8*t0 + 12*x**9*t0 - 30*x**8*t0**2 +
        30*x**7*t0**2 + 84*x**10*t0**2 - 140*x**9*t0**3 - 40*x**6*t0**3 +
        40*x**7*t0**3 - 30*x**6*t0**4 + 30*x**5*t0**4 + 140*x**8*t0**4 -
        84*x**7*t0**5 - 12*x**4*t0**5 + 12*x**5*t0**5 - 2*x**4*t0**6 +
        2*x**3*t0**6 + 28*x**6*t0**6 - 4*x**5*t0**7 + 2*x**9 - 2*x**10 +
        4*x**12)/(-8*x**11*t0 + 28*x**10*t0**2 - 56*x**9*t0**3 +
        70*x**8*t0**4 - 56*x**7*t0**5 + 28*x**6*t0**6 - 8*x**5*t0**7 +
        x**4*t0**8 + x**12)*t1 + (-2*x**2*t0 + 2*x**3*t0 + x*t0**2 -
        x**2*t0**2 + x**3 - x**4)/(-4*x**5*t0 + 6*x**4*t0**2 - 4*x**3*t0**3 +
        x**2*t0**4 + x**6), t1, z, expand=False)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t0), Poly(2*x*t1, t1)]})
    assert integrate_hyperexponential_polynomial(p, DE, z) == (
        Poly((x - t0)*t1**2 + (-2*t0 + 2*x)*t1, t1), Poly(-2*x*t0 + x**2 +
        t0**2, t1), True)

    DE = DifferentialExtension(extension={'D':[Poly(1, x), Poly(t0, t0)]})
    assert integrate_hyperexponential_polynomial(Poly(0, t0), DE, z) == (
        Poly(0, t0), Poly(1, t0), True)


def test_integrate_hyperexponential_returns_piecewise():
    a, b = symbols('a b')
    DE = DifferentialExtension(a**x, x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(x*log(a))/log(a), Ne(log(a), 0)), (x, True)), 0, True)
    DE = DifferentialExtension(a**(b*x), x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(b*x*log(a))/(b*log(a)), Ne(b*log(a), 0)), (x, True)), 0, True)
    DE = DifferentialExtension(exp(a*x), x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(a*x)/a, Ne(a, 0)), (x, True)), 0, True)
    DE = DifferentialExtension(x*exp(a*x), x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        ((a*x - 1)*exp(a*x)/a**2, Ne(a**2, 0)), (x**2/2, True)), 0, True)
    DE = DifferentialExtension(x**2*exp(a*x), x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        ((x**2*a**2 - 2*a*x + 2)*exp(a*x)/a**3, Ne(a**3, 0)),
        (x**3/3, True)), 0, True)
    DE = DifferentialExtension(x**y + z, y)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(log(x)*y)/log(x), Ne(log(x), 0)), (y, True)), z, True)
    DE = DifferentialExtension(x**y + z + x**(2*y), y)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        ((exp(2*log(x)*y)*log(x) +
            2*exp(log(x)*y)*log(x))/(2*log(x)**2), Ne(2*log(x)**2, 0)),
            (2*y, True),
        ), z, True)
    # TODO: Add a test where two different parts of the extension use a
    # Piecewise, like y**x + z**x.


def test_issue_13947():
    a, t, s = symbols('a t s')
    assert risch_integrate(2**(-pi)/(2**t + 1), t) == \
        2**(-pi)*t - 2**(-pi)*log(2**t + 1)/log(2)
    assert risch_integrate(a**(t - s)/(a**t + 1), t) == \
        exp(-s*log(a))*log(a**t + 1)/log(a)


def test_integrate_primitive():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)],
        'Tfuncs': [log]})
    assert integrate_primitive(Poly(t, t), Poly(1, t), DE) == (x*log(x), -1, True)
    assert integrate_primitive(Poly(x, t), Poly(t, t), DE) == (0, NonElementaryIntegral(x/log(x), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(1/(x + 1), t2)],
        'Tfuncs': [log, Lambda(i, log(i + 1))]})
    assert integrate_primitive(Poly(t1, t2), Poly(t2, t2), DE) == \
        (0, NonElementaryIntegral(log(x)/log(1 + x), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(1/(x*t1), t2)],
        'Tfuncs': [log, Lambda(i, log(log(i)))]})
    assert integrate_primitive(Poly(t2, t2), Poly(t1, t2), DE) == \
        (0, NonElementaryIntegral(log(log(x))/log(x), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t0)],
        'Tfuncs': [log]})
    assert integrate_primitive(Poly(x**2*t0**3 + (3*x**2 + x)*t0**2 + (3*x**2
    + 2*x)*t0 + x**2 + x, t0), Poly(x**2*t0**4 + 4*x**2*t0**3 + 6*x**2*t0**2 +
    4*x**2*t0 + x**2, t0), DE) == \
        (-1/(log(x) + 1), NonElementaryIntegral(1/(log(x) + 1), x), False)

def test_integrate_hypertangent_polynomial():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert integrate_hypertangent_polynomial(Poly(t**2 + x*t + 1, t), DE) == \
        (Poly(t, t), Poly(x/2, t))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(a*(t**2 + 1), t)]})
    assert integrate_hypertangent_polynomial(Poly(t**5, t), DE) == \
        (Poly(1/(4*a)*t**4 - 1/(2*a)*t**2, t), Poly(1/(2*a), t))


def test_integrate_nonlinear_no_specials():
    a, d, = Poly(x**2*t**5 + x*t**4 - nu**2*t**3 - x*(x**2 + 1)*t**2 - (x**2 -
    nu**2)*t - x**5/4, t), Poly(x**2*t**4 + x**2*(x**2 + 2)*t**2 + x**2 + x**4 + x**6/4, t)
    # f(x) == phi_nu(x), the logarithmic derivative of J_v, the Bessel function,
    # which has no specials (see Chapter 5, note 4 of Bronstein's book).
    f = Function('phi_nu')
    DE = DifferentialExtension(extension={'D': [Poly(1, x),
        Poly(-t**2 - t/x - (1 - nu**2/x**2), t)], 'Tfuncs': [f]})
    assert integrate_nonlinear_no_specials(a, d, DE) == \
        (-log(1 + f(x)**2 + x**2/2)/2 + (- 4 - x**2)/(4 + 2*x**2 + 4*f(x)**2), True)
    assert integrate_nonlinear_no_specials(Poly(t, t), Poly(1, t), DE) == \
        (0, False)


def test_integer_powers():
    assert integer_powers([x, x/2, x**2 + 1, x*Rational(2, 3)]) == [
            (x/6, [(x, 6), (x/2, 3), (x*Rational(2, 3), 4)]),
            (1 + x**2, [(1 + x**2, 1)])]


def test_DifferentialExtension_exp():
    assert DifferentialExtension(exp(x) + exp(x**2), x)._important_attrs == \
        (Poly(t1 + t0, t1), Poly(1, t1), [Poly(1, x,), Poly(t0, t0),
        Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i)),
        Lambda(i, exp(i**2))], [], [None, 'exp', 'exp'], [None, x, x**2])
    assert DifferentialExtension(exp(x) + exp(2*x), x)._important_attrs == \
        (Poly(t0**2 + t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0, t0)], [x, t0],
        [Lambda(i, exp(i))], [], [None, 'exp'], [None, x])
    assert DifferentialExtension(exp(x) + exp(x/2), x)._important_attrs == \
        (Poly(t0**2 + t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0/2, t0)],
        [x, t0], [Lambda(i, exp(i/2))], [], [None, 'exp'], [None, x/2])
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x + x**2), x)._important_attrs == \
        (Poly((1 + t0)*t1 + t0, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0),
        Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i)),
        Lambda(i, exp(i**2))], [], [None, 'exp', 'exp'], [None, x, x**2])
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x + x**2 + 1), x)._important_attrs == \
        (Poly((1 + S.Exp1*t0)*t1 + t0, t1), Poly(1, t1), [Poly(1, x),
        Poly(t0, t0), Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i)),
        Lambda(i, exp(i**2))], [], [None, 'exp', 'exp'], [None, x, x**2])
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x/2 + x**2), x)._important_attrs == \
        (Poly((t0 + 1)*t1 + t0**2, t1), Poly(1, t1), [Poly(1, x),
        Poly(t0/2, t0), Poly(2*x*t1, t1)], [x, t0, t1],
        [Lambda(i, exp(i/2)), Lambda(i, exp(i**2))],
        [(exp(x/2), sqrt(exp(x)))], [None, 'exp', 'exp'], [None, x/2, x**2])
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x/2 + x**2 + 3), x)._important_attrs == \
        (Poly((t0*exp(3) + 1)*t1 + t0**2, t1), Poly(1, t1), [Poly(1, x),
        Poly(t0/2, t0), Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i/2)),
        Lambda(i, exp(i**2))], [(exp(x/2), sqrt(exp(x)))], [None, 'exp', 'exp'],
        [None, x/2, x**2])
    assert DifferentialExtension(sqrt(exp(x)), x)._important_attrs == \
        (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0/2, t0)], [x, t0],
        [Lambda(i, exp(i/2))], [(exp(x/2), sqrt(exp(x)))], [None, 'exp'], [None, x/2])

    assert DifferentialExtension(exp(x/2), x)._important_attrs == \
        (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0/2, t0)], [x, t0],
        [Lambda(i, exp(i/2))], [], [None, 'exp'], [None, x/2])


def test_DifferentialExtension_log():
    assert DifferentialExtension(log(x)*log(x + 1)*log(2*x**2 + 2*x), x)._important_attrs == \
        (Poly(t0*t1**2 + (t0*log(2) + t0**2)*t1, t1), Poly(1, t1),
        [Poly(1, x), Poly(1/x, t0),
        Poly(1/(x + 1), t1, expand=False)], [x, t0, t1],
        [Lambda(i, log(i)), Lambda(i, log(i + 1))], [], [None, 'log', 'log'],
        [None, x, x + 1])
    assert DifferentialExtension(x**x*log(x), x)._important_attrs == \
        (Poly(t0*t1, t1), Poly(1, t1), [Poly(1, x), Poly(1/x, t0),
        Poly((1 + t0)*t1, t1)], [x, t0, t1], [Lambda(i, log(i)),
        Lambda(i, exp(t0*i))], [(exp(x*log(x)), x**x)], [None, 'log', 'exp'],
        [None, x, t0*x])


def test_DifferentialExtension_symlog():
    # See comment on test_risch_integrate below
    assert DifferentialExtension(log(x**x), x)._important_attrs == \
        (Poly(t0*x, t1), Poly(1, t1), [Poly(1, x), Poly(1/x, t0), Poly((t0 +
            1)*t1, t1)], [x, t0, t1], [Lambda(i, log(i)), Lambda(i, exp(i*t0))],
            [(exp(x*log(x)), x**x)], [None, 'log', 'exp'], [None, x, t0*x])
    assert DifferentialExtension(log(x**y), x)._important_attrs == \
        (Poly(y*t0, t0), Poly(1, t0), [Poly(1, x), Poly(1/x, t0)], [x, t0],
        [Lambda(i, log(i))], [(y*log(x), log(x**y))], [None, 'log'],
        [None, x])
    assert DifferentialExtension(log(sqrt(x)), x)._important_attrs == \
        (Poly(t0, t0), Poly(2, t0), [Poly(1, x), Poly(1/x, t0)], [x, t0],
        [Lambda(i, log(i))], [(log(x)/2, log(sqrt(x)))], [None, 'log'],
        [None, x])


def test_DifferentialExtension_handle_first():
    assert DifferentialExtension(exp(x)*log(x), x, handle_first='log')._important_attrs == \
        (Poly(t0*t1, t1), Poly(1, t1), [Poly(1, x), Poly(1/x, t0),
        Poly(t1, t1)], [x, t0, t1], [Lambda(i, log(i)), Lambda(i, exp(i))],
        [], [None, 'log', 'exp'], [None, x, x])
    assert DifferentialExtension(exp(x)*log(x), x, handle_first='exp')._important_attrs == \
        (Poly(t0*t1, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0),
        Poly(1/x, t1)], [x, t0, t1], [Lambda(i, exp(i)), Lambda(i, log(i))],
        [], [None, 'exp', 'log'], [None, x, x])

    # This one must have the log first, regardless of what we set it to
    # (because the log is inside of the exponential: x**x == exp(x*log(x)))
    assert DifferentialExtension(-x**x*log(x)**2 + x**x - x**x/x, x,
    handle_first='exp')._important_attrs == \
        DifferentialExtension(-x**x*log(x)**2 + x**x - x**x/x, x,
        handle_first='log')._important_attrs == \
        (Poly((-1 + x - x*t0**2)*t1, t1), Poly(x, t1),
            [Poly(1, x), Poly(1/x, t0), Poly((1 + t0)*t1, t1)], [x, t0, t1],
            [Lambda(i, log(i)), Lambda(i, exp(t0*i))], [(exp(x*log(x)), x**x)],
            [None, 'log', 'exp'], [None, x, t0*x])


def test_DifferentialExtension_all_attrs():
    # Test 'unimportant' attributes
    DE = DifferentialExtension(exp(x)*log(x), x, handle_first='exp')
    assert DE.f == exp(x)*log(x)
    assert DE.newf == t0*t1
    assert DE.x == x
    assert DE.cases == ['base', 'exp', 'primitive']
    assert DE.case == 'primitive'

    assert DE.level == -1
    assert DE.t == t1 == DE.T[DE.level]
    assert DE.d == Poly(1/x, t1) == DE.D[DE.level]
    raises(ValueError, lambda: DE.increment_level())
    DE.decrement_level()
    assert DE.level == -2
    assert DE.t == t0 == DE.T[DE.level]
    assert DE.d == Poly(t0, t0) == DE.D[DE.level]
    assert DE.case == 'exp'
    DE.decrement_level()
    assert DE.level == -3
    assert DE.t == x == DE.T[DE.level] == DE.x
    assert DE.d == Poly(1, x) == DE.D[DE.level]
    assert DE.case == 'base'
    raises(ValueError, lambda: DE.decrement_level())
    DE.increment_level()
    DE.increment_level()
    assert DE.level == -1
    assert DE.t == t1 == DE.T[DE.level]
    assert DE.d == Poly(1/x, t1) == DE.D[DE.level]
    assert DE.case == 'primitive'

    # Test methods
    assert DE.indices('log') == [2]
    assert DE.indices('exp') == [1]


def test_DifferentialExtension_extension_flag():
    raises(ValueError, lambda: DifferentialExtension(extension={'T': [x, t]}))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert DE._important_attrs == (None, None, [Poly(1, x), Poly(t, t)], [x, t],
        None, None, None, None)
    assert DE.d == Poly(t, t)
    assert DE.t == t
    assert DE.level == -1
    assert DE.cases == ['base', 'exp']
    assert DE.x == x
    assert DE.case == 'exp'

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)],
        'exts': [None, 'exp'], 'extargs': [None, x]})
    assert DE._important_attrs == (None, None, [Poly(1, x), Poly(t, t)], [x, t],
        None, None, [None, 'exp'], [None, x])
    raises(ValueError, lambda: DifferentialExtension())


def test_DifferentialExtension_misc():
    # Odd ends
    assert DifferentialExtension(sin(y)*exp(x), x)._important_attrs == \
        (Poly(sin(y)*t0, t0, domain='ZZ[sin(y)]'), Poly(1, t0, domain='ZZ'),
        [Poly(1, x, domain='ZZ'), Poly(t0, t0, domain='ZZ')], [x, t0],
        [Lambda(i, exp(i))], [], [None, 'exp'], [None, x])
    raises(NotImplementedError, lambda: DifferentialExtension(sin(x), x))
    assert DifferentialExtension(10**x, x)._important_attrs == \
        (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(log(10)*t0, t0)], [x, t0],
        [Lambda(i, exp(i*log(10)))], [(exp(x*log(10)), 10**x)], [None, 'exp'],
        [None, x*log(10)])
    assert DifferentialExtension(log(x) + log(x**2), x)._important_attrs in [
        (Poly(3*t0, t0), Poly(2, t0), [Poly(1, x), Poly(2/x, t0)], [x, t0],
        [Lambda(i, log(i**2))], [], [None, ], [], [1], [x**2]),
        (Poly(3*t0, t0), Poly(1, t0), [Poly(1, x), Poly(1/x, t0)], [x, t0],
        [Lambda(i, log(i))], [], [None, 'log'], [None, x])]
    assert DifferentialExtension(S.Zero, x)._important_attrs == \
        (Poly(0, x), Poly(1, x), [Poly(1, x)], [x], [], [], [None], [None])
    assert DifferentialExtension(tan(atan(x).rewrite(log)), x)._important_attrs == \
        (Poly(x, x), Poly(1, x), [Poly(1, x)], [x], [], [], [None], [None])


def test_DifferentialExtension_Rothstein():
    # Rothstein's integral
    f = (2581284541*exp(x) + 1757211400)/(39916800*exp(3*x) +
    119750400*exp(x)**2 + 119750400*exp(x) + 39916800)*exp(1/(exp(x) + 1) - 10*x)
    assert DifferentialExtension(f, x)._important_attrs == \
        (Poly((1757211400 + 2581284541*t0)*t1, t1), Poly(39916800 +
        119750400*t0 + 119750400*t0**2 + 39916800*t0**3, t1),
        [Poly(1, x), Poly(t0, t0), Poly(-(10 + 21*t0 + 10*t0**2)/(1 + 2*t0 +
        t0**2)*t1, t1, domain='ZZ(t0)')], [x, t0, t1],
        [Lambda(i, exp(i)), Lambda(i, exp(1/(t0 + 1) - 10*i))], [],
        [None, 'exp', 'exp'], [None, x, 1/(t0 + 1) - 10*x])


class _TestingException(Exception):
    """Dummy Exception class for testing."""
    pass


def test_DecrementLevel():
    DE = DifferentialExtension(x*log(exp(x) + 1), x)
    assert DE.level == -1
    assert DE.t == t1
    assert DE.d == Poly(t0/(t0 + 1), t1)
    assert DE.case == 'primitive'

    with DecrementLevel(DE):
        assert DE.level == -2
        assert DE.t == t0
        assert DE.d == Poly(t0, t0)
        assert DE.case == 'exp'

        with DecrementLevel(DE):
            assert DE.level == -3
            assert DE.t == x
            assert DE.d == Poly(1, x)
            assert DE.case == 'base'

        assert DE.level == -2
        assert DE.t == t0
        assert DE.d == Poly(t0, t0)
        assert DE.case == 'exp'

    assert DE.level == -1
    assert DE.t == t1
    assert DE.d == Poly(t0/(t0 + 1), t1)
    assert DE.case == 'primitive'

    # Test that __exit__ is called after an exception correctly
    try:
        with DecrementLevel(DE):
            raise _TestingException
    except _TestingException:
        pass
    else:
        raise AssertionError("Did not raise.")

    assert DE.level == -1
    assert DE.t == t1
    assert DE.d == Poly(t0/(t0 + 1), t1)
    assert DE.case == 'primitive'


def test_risch_integrate():
    assert risch_integrate(t0*exp(x), x) == t0*exp(x)
    assert risch_integrate(sin(x), x, rewrite_complex=True) == -exp(I*x)/2 - exp(-I*x)/2

    # From my GSoC writeup
    assert risch_integrate((1 + 2*x**2 + x**4 + 2*x**3*exp(2*x**2))/
    (x**4*exp(x**2) + 2*x**2*exp(x**2) + exp(x**2)), x) == \
        NonElementaryIntegral(exp(-x**2), x) + exp(x**2)/(1 + x**2)


    assert risch_integrate(0, x) == 0

    # also tests prde_cancel()
    e1 = log(x/exp(x) + 1)
    ans1 = risch_integrate(e1, x)
    assert ans1 == (x*log(x*exp(-x) + 1) + NonElementaryIntegral((x**2 - x)/(x + exp(x)), x))
    assert cancel(diff(ans1, x) - e1) == 0

    # also tests issue #10798
    e2 = (log(-1/y)/2 - log(1/y)/2)/y - (log(1 - 1/y)/2 - log(1 + 1/y)/2)/y
    ans2 = risch_integrate(e2, y)
    assert ans2 == log(1/y)*log(1 - 1/y)/2 - log(1/y)*log(1 + 1/y)/2 + \
            NonElementaryIntegral((I*pi*y**2 - 2*y*log(1/y) - I*pi)/(2*y**3 - 2*y), y)
    assert expand_log(cancel(diff(ans2, y) - e2), force=True) == 0

    # These are tested here in addition to in test_DifferentialExtension above
    # (symlogs) to test that backsubs works correctly.  The integrals should be
    # written in terms of the original logarithms in the integrands.

    # XXX: Unfortunately, making backsubs work on this one is a little
    # trickier, because x**x is converted to exp(x*log(x)), and so log(x**x)
    # is converted to x*log(x). (x**2*log(x)).subs(x*log(x), log(x**x)) is
    # smart enough, the issue is that these splits happen at different places
    # in the algorithm.  Maybe a heuristic is in order
    assert risch_integrate(log(x**x), x) == x**2*log(x)/2 - x**2/4

    assert risch_integrate(log(x**y), x) == x*log(x**y) - x*y
    assert risch_integrate(log(sqrt(x)), x) == x*log(sqrt(x)) - x/2


def test_risch_integrate_float():
    assert risch_integrate((-60*exp(x) - 19.2*exp(4*x))*exp(4*x), x) == -2.4*exp(8*x) - 12.0*exp(5*x)


def test_NonElementaryIntegral():
    assert isinstance(risch_integrate(exp(x**2), x), NonElementaryIntegral)
    assert isinstance(risch_integrate(x**x*log(x), x), NonElementaryIntegral)
    # Make sure methods of Integral still give back a NonElementaryIntegral
    assert isinstance(NonElementaryIntegral(x**x*t0, x).subs(t0, log(x)), NonElementaryIntegral)


def test_xtothex():
    a = risch_integrate(x**x, x)
    assert a == NonElementaryIntegral(x**x, x)
    assert isinstance(a, NonElementaryIntegral)


def test_DifferentialExtension_equality():
    DE1 = DE2 = DifferentialExtension(log(x), x)
    assert DE1 == DE2


def test_DifferentialExtension_printing():
    DE = DifferentialExtension(exp(2*x**2) + log(exp(x**2) + 1), x)
    assert repr(DE) == ("DifferentialExtension(dict([('f', exp(2*x**2) + log(exp(x**2) + 1)), "
        "('x', x), ('T', [x, t0, t1]), ('D', [Poly(1, x, domain='ZZ'), Poly(2*x*t0, t0, domain='ZZ[x]'), "
        "Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')]), ('fa', Poly(t1 + t0**2, t1, domain='ZZ[t0]')), "
        "('fd', Poly(1, t1, domain='ZZ')), ('Tfuncs', [Lambda(i, exp(i**2)), Lambda(i, log(t0 + 1))]), "
        "('backsubs', []), ('exts', [None, 'exp', 'log']), ('extargs', [None, x**2, t0 + 1]), "
        "('cases', ['base', 'exp', 'primitive']), ('case', 'primitive'), ('t', t1), "
        "('d', Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')), ('newf', t0**2 + t1), ('level', -1), "
        "('dummy', False)]))")

    assert str(DE) == ("DifferentialExtension({fa=Poly(t1 + t0**2, t1, domain='ZZ[t0]'), "
        "fd=Poly(1, t1, domain='ZZ'), D=[Poly(1, x, domain='ZZ'), Poly(2*x*t0, t0, domain='ZZ[x]'), "
        "Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')]})")


def test_issue_23948():
    f = (
        ( (-2*x**5 + 28*x**4 - 144*x**3 + 324*x**2 - 270*x)*log(x)**2
         +(-4*x**6 + 56*x**5 - 288*x**4 + 648*x**3 - 540*x**2)*log(x)
         +(2*x**5 - 28*x**4 + 144*x**3 - 324*x**2 + 270*x)*exp(x)
         +(2*x**5 - 28*x**4 + 144*x**3 - 324*x**2 + 270*x)*log(5)
         -2*x**7 + 26*x**6 - 116*x**5 + 180*x**4 + 54*x**3 - 270*x**2
        )*log(-log(x)**2 - 2*x*log(x) + exp(x) + log(5) - x**2 - x)**2
       +( (4*x**5 - 44*x**4 + 168*x**3 - 216*x**2 - 108*x + 324)*log(x)
         +(-2*x**5 + 24*x**4 - 108*x**3 + 216*x**2 - 162*x)*exp(x)
         +4*x**6 - 42*x**5 + 144*x**4 - 108*x**3 - 324*x**2 + 486*x
        )*log(-log(x)**2 - 2*x*log(x) + exp(x) + log(5) - x**2 - x)
    )/(x*exp(x)**2*log(x)**2 + 2*x**2*exp(x)**2*log(x) - x*exp(x)**3
       +(-x*log(5) + x**3 + x**2)*exp(x)**2)

    F = ((x**4 - 12*x**3 + 54*x**2 - 108*x + 81)*exp(-2*x)
        *log(-x**2 - 2*x*log(x) - x + exp(x) - log(x)**2 + log(5))**2)

    assert risch_integrate(f, x) == F
