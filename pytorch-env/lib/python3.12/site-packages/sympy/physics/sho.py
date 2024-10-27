from sympy.core import S, pi, Rational
from sympy.functions import assoc_laguerre, sqrt, exp, factorial, factorial2


def R_nl(n, l, nu, r):
    """
    Returns the radial wavefunction R_{nl} for a 3d isotropic harmonic
    oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.  Corresponds to the number of nodes in
        the wavefunction.  ``n >= 0``
    l :
        The quantum number for orbital angular momentum.
    nu :
        mass-scaled frequency: nu = m*omega/(2*hbar) where `m` is the mass
        and `omega` the frequency of the oscillator.
        (in atomic units ``nu == omega/2``)
    r :
        Radial coordinate.

    Examples
    ========

    >>> from sympy.physics.sho import R_nl
    >>> from sympy.abc import r, nu, l
    >>> R_nl(0, 0, 1, r)
    2*2**(3/4)*exp(-r**2)/pi**(1/4)
    >>> R_nl(1, 0, 1, r)
    4*2**(1/4)*sqrt(3)*(3/2 - 2*r**2)*exp(-r**2)/(3*pi**(1/4))

    l, nu and r may be symbolic:

    >>> R_nl(0, 0, nu, r)
    2*2**(3/4)*sqrt(nu**(3/2))*exp(-nu*r**2)/pi**(1/4)
    >>> R_nl(0, l, 1, r)
    r**l*sqrt(2**(l + 3/2)*2**(l + 2)/factorial2(2*l + 1))*exp(-r**2)/pi**(1/4)

    The normalization of the radial wavefunction is:

    >>> from sympy import Integral, oo
    >>> Integral(R_nl(0, 0, 1, r)**2*r**2, (r, 0, oo)).n()
    1.00000000000000
    >>> Integral(R_nl(1, 0, 1, r)**2*r**2, (r, 0, oo)).n()
    1.00000000000000
    >>> Integral(R_nl(1, 1, 1, r)**2*r**2, (r, 0, oo)).n()
    1.00000000000000

    """
    n, l, nu, r = map(S, [n, l, nu, r])

    # formula uses n >= 1 (instead of nodal n >= 0)
    n = n + 1
    C = sqrt(
            ((2*nu)**(l + Rational(3, 2))*2**(n + l + 1)*factorial(n - 1))/
            (sqrt(pi)*(factorial2(2*n + 2*l - 1)))
    )
    return C*r**(l)*exp(-nu*r**2)*assoc_laguerre(n - 1, l + S.Half, 2*nu*r**2)


def E_nl(n, l, hw):
    """
    Returns the Energy of an isotropic harmonic oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    l :
        The orbital angular momentum.
    hw :
        The harmonic oscillator parameter.

    Notes
    =====

    The unit of the returned value matches the unit of hw, since the energy is
    calculated as:

        E_nl = (2*n + l + 3/2)*hw

    Examples
    ========

    >>> from sympy.physics.sho import E_nl
    >>> from sympy import symbols
    >>> x, y, z = symbols('x, y, z')
    >>> E_nl(x, y, z)
    z*(2*x + y + 3/2)
    """
    return (2*n + l + Rational(3, 2))*hw
