from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.physics.hydrogen import R_nl, E_nl, E_nl_dirac, Psi_nlm
from sympy.testing.pytest import raises

n, r, Z = symbols('n r Z')


def feq(a, b, max_relative_error=1e-12, max_absolute_error=1e-12):
    a = float(a)
    b = float(b)
    # if the numbers are close enough (absolutely), then they are equal
    if abs(a - b) < max_absolute_error:
        return True
    # if not, they can still be equal if their relative error is small
    if abs(b) > abs(a):
        relative_error = abs((a - b)/b)
    else:
        relative_error = abs((a - b)/a)
    return relative_error <= max_relative_error


def test_wavefunction():
    a = 1/Z
    R = {
        (1, 0): 2*sqrt(1/a**3) * exp(-r/a),
        (2, 0): sqrt(1/(2*a**3)) * exp(-r/(2*a)) * (1 - r/(2*a)),
        (2, 1): S.Half * sqrt(1/(6*a**3)) * exp(-r/(2*a)) * r/a,
        (3, 0): Rational(2, 3) * sqrt(1/(3*a**3)) * exp(-r/(3*a)) *
        (1 - 2*r/(3*a) + Rational(2, 27) * (r/a)**2),
        (3, 1): Rational(4, 27) * sqrt(2/(3*a**3)) * exp(-r/(3*a)) *
        (1 - r/(6*a)) * r/a,
        (3, 2): Rational(2, 81) * sqrt(2/(15*a**3)) * exp(-r/(3*a)) * (r/a)**2,
        (4, 0): Rational(1, 4) * sqrt(1/a**3) * exp(-r/(4*a)) *
        (1 - 3*r/(4*a) + Rational(1, 8) * (r/a)**2 - Rational(1, 192) * (r/a)**3),
        (4, 1): Rational(1, 16) * sqrt(5/(3*a**3)) * exp(-r/(4*a)) *
        (1 - r/(4*a) + Rational(1, 80) * (r/a)**2) * (r/a),
        (4, 2): Rational(1, 64) * sqrt(1/(5*a**3)) * exp(-r/(4*a)) *
        (1 - r/(12*a)) * (r/a)**2,
        (4, 3): Rational(1, 768) * sqrt(1/(35*a**3)) * exp(-r/(4*a)) * (r/a)**3,
    }
    for n, l in R:
        assert simplify(R_nl(n, l, r, Z) - R[(n, l)]) == 0


def test_norm():
    # Maximum "n" which is tested:
    n_max = 2  # it works, but is slow, for n_max > 2
    for n in range(n_max + 1):
        for l in range(n):
            assert integrate(R_nl(n, l, r)**2 * r**2, (r, 0, oo)) == 1

def test_psi_nlm():
    r=S('r')
    phi=S('phi')
    theta=S('theta')
    assert (Psi_nlm(1, 0, 0, r, phi, theta) == exp(-r) / sqrt(pi))
    assert (Psi_nlm(2, 1, -1, r, phi, theta)) == S.Half * exp(-r / (2)) * r \
        * (sin(theta) * exp(-I * phi) / (4 * sqrt(pi)))
    assert (Psi_nlm(3, 2, 1, r, phi, theta, 2) == -sqrt(2) * sin(theta) \
         * exp(I * phi) * cos(theta) / (4 * sqrt(pi)) * S(2) / 81 \
        * sqrt(2 * 2 ** 3) * exp(-2 * r / (3)) * (r * 2) ** 2)

def test_hydrogen_energies():
    assert E_nl(n, Z) == -Z**2/(2*n**2)
    assert E_nl(n) == -1/(2*n**2)

    assert E_nl(1, 47) == -S(47)**2/(2*1**2)
    assert E_nl(2, 47) == -S(47)**2/(2*2**2)

    assert E_nl(1) == -S.One/(2*1**2)
    assert E_nl(2) == -S.One/(2*2**2)
    assert E_nl(3) == -S.One/(2*3**2)
    assert E_nl(4) == -S.One/(2*4**2)
    assert E_nl(100) == -S.One/(2*100**2)

    raises(ValueError, lambda: E_nl(0))


def test_hydrogen_energies_relat():
    # First test exact formulas for small "c" so that we get nice expressions:
    assert E_nl_dirac(2, 0, Z=1, c=1) == 1/sqrt(2) - 1
    assert simplify(E_nl_dirac(2, 0, Z=1, c=2) - ( (8*sqrt(3) + 16)
                / sqrt(16*sqrt(3) + 32) - 4)) == 0
    assert simplify(E_nl_dirac(2, 0, Z=1, c=3) - ( (54*sqrt(2) + 81)
                / sqrt(108*sqrt(2) + 162) - 9)) == 0

    # Now test for almost the correct speed of light, without floating point
    # numbers:
    assert simplify(E_nl_dirac(2, 0, Z=1, c=137) - ( (352275361 + 10285412 *
        sqrt(1173)) / sqrt(704550722 + 20570824 * sqrt(1173)) - 18769)) == 0
    assert simplify(E_nl_dirac(2, 0, Z=82, c=137) - ( (352275361 + 2571353 *
        sqrt(12045)) / sqrt(704550722 + 5142706*sqrt(12045)) - 18769)) == 0

    # Test using exact speed of light, and compare against the nonrelativistic
    # energies:
    for n in range(1, 5):
        for l in range(n):
            assert feq(E_nl_dirac(n, l), E_nl(n), 1e-5, 1e-5)
            if l > 0:
                assert feq(E_nl_dirac(n, l, False), E_nl(n), 1e-5, 1e-5)

    Z = 2
    for n in range(1, 5):
        for l in range(n):
            assert feq(E_nl_dirac(n, l, Z=Z), E_nl(n, Z), 1e-4, 1e-4)
            if l > 0:
                assert feq(E_nl_dirac(n, l, False, Z), E_nl(n, Z), 1e-4, 1e-4)

    Z = 3
    for n in range(1, 5):
        for l in range(n):
            assert feq(E_nl_dirac(n, l, Z=Z), E_nl(n, Z), 1e-3, 1e-3)
            if l > 0:
                assert feq(E_nl_dirac(n, l, False, Z), E_nl(n, Z), 1e-3, 1e-3)

    # Test the exceptions:
    raises(ValueError, lambda: E_nl_dirac(0, 0))
    raises(ValueError, lambda: E_nl_dirac(1, -1))
    raises(ValueError, lambda: E_nl_dirac(1, 0, False))
