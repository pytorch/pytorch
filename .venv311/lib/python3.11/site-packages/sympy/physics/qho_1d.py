from sympy.core import S, pi, Rational
from sympy.functions import hermite, sqrt, exp, factorial, Abs
from sympy.physics.quantum.constants import hbar


def psi_n(n, x, m, omega):
    """
    Returns the wavefunction psi_{n} for the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        the "nodal" quantum number.  Corresponds to the number of nodes in the
        wavefunction.  ``n >= 0``
    x :
        x coordinate.
    m :
        Mass of the particle.
    omega :
        Angular frequency of the oscillator.

    Examples
    ========

    >>> from sympy.physics.qho_1d import psi_n
    >>> from sympy.abc import m, x, omega
    >>> psi_n(0, x, m, omega)
    (m*omega)**(1/4)*exp(-m*omega*x**2/(2*hbar))/(hbar**(1/4)*pi**(1/4))

    """

    # sympify arguments
    n, x, m, omega = map(S, [n, x, m, omega])
    nu = m * omega / hbar
    # normalization coefficient
    C = (nu/pi)**Rational(1, 4) * sqrt(1/(2**n*factorial(n)))

    return C * exp(-nu* x**2 /2) * hermite(n, sqrt(nu)*x)


def E_n(n, omega):
    """
    Returns the Energy of the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    omega :
        The harmonic oscillator angular frequency.

    Notes
    =====

    The unit of the returned value matches the unit of hw, since the energy is
    calculated as:

        E_n = hbar * omega*(n + 1/2)

    Examples
    ========

    >>> from sympy.physics.qho_1d import E_n
    >>> from sympy.abc import x, omega
    >>> E_n(x, omega)
    hbar*omega*(x + 1/2)
    """

    return hbar * omega * (n + S.Half)


def coherent_state(n, alpha):
    """
    Returns <n|alpha> for the coherent states of 1D harmonic oscillator.
    See https://en.wikipedia.org/wiki/Coherent_states

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    alpha :
        The eigen value of annihilation operator.
    """

    return exp(- Abs(alpha)**2/2)*(alpha**n)/sqrt(factorial(n))
