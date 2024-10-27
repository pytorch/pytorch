from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.polynomials import assoc_laguerre
from sympy.functions.special.spherical_harmonics import Ynm


def R_nl(n, l, r, Z=1):
    """
    Returns the Hydrogen radial wavefunction R_{nl}.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    r :
        Radial coordinate.
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)

    Everything is in Hartree atomic units.

    Examples
    ========

    >>> from sympy.physics.hydrogen import R_nl
    >>> from sympy.abc import r, Z
    >>> R_nl(1, 0, r, Z)
    2*sqrt(Z**3)*exp(-Z*r)
    >>> R_nl(2, 0, r, Z)
    sqrt(2)*(-Z*r + 2)*sqrt(Z**3)*exp(-Z*r/2)/4
    >>> R_nl(2, 1, r, Z)
    sqrt(6)*Z*r*sqrt(Z**3)*exp(-Z*r/2)/12

    For Hydrogen atom, you can just use the default value of Z=1:

    >>> R_nl(1, 0, r)
    2*exp(-r)
    >>> R_nl(2, 0, r)
    sqrt(2)*(2 - r)*exp(-r/2)/4
    >>> R_nl(3, 0, r)
    2*sqrt(3)*(2*r**2/9 - 2*r + 3)*exp(-r/3)/27

    For Silver atom, you would use Z=47:

    >>> R_nl(1, 0, r, Z=47)
    94*sqrt(47)*exp(-47*r)
    >>> R_nl(2, 0, r, Z=47)
    47*sqrt(94)*(2 - 47*r)*exp(-47*r/2)/4
    >>> R_nl(3, 0, r, Z=47)
    94*sqrt(141)*(4418*r**2/9 - 94*r + 3)*exp(-47*r/3)/27

    The normalization of the radial wavefunction is:

    >>> from sympy import integrate, oo
    >>> integrate(R_nl(1, 0, r)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 0, r)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 1, r)**2 * r**2, (r, 0, oo))
    1

    It holds for any atomic number:

    >>> integrate(R_nl(1, 0, r, Z=2)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 0, r, Z=3)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 1, r, Z=4)**2 * r**2, (r, 0, oo))
    1

    """
    # sympify arguments
    n, l, r, Z = map(S, [n, l, r, Z])
    # radial quantum number
    n_r = n - l - 1
    # rescaled "r"
    a = 1/Z  # Bohr radius
    r0 = 2 * r / (n * a)
    # normalization coefficient
    C = sqrt((S(2)/(n*a))**3 * factorial(n_r) / (2*n*factorial(n + l)))
    # This is an equivalent normalization coefficient, that can be found in
    # some books. Both coefficients seem to be the same fast:
    # C =  S(2)/n**2 * sqrt(1/a**3 * factorial(n_r) / (factorial(n+l)))
    return C * r0**l * assoc_laguerre(n_r, 2*l + 1, r0).expand() * exp(-r0/2)


def Psi_nlm(n, l, m, r, phi, theta, Z=1):
    """
    Returns the Hydrogen wave function psi_{nlm}. It's the product of
    the radial wavefunction R_{nl} and the spherical harmonic Y_{l}^{m}.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    m : integer
        ``m`` is the Magnetic Quantum Number with values
        ranging from ``-l`` to ``l``.
    r :
        radial coordinate
    phi :
        azimuthal angle
    theta :
        polar angle
    Z :
        atomic number (1 for Hydrogen, 2 for Helium, ...)

    Everything is in Hartree atomic units.

    Examples
    ========

    >>> from sympy.physics.hydrogen import Psi_nlm
    >>> from sympy import Symbol
    >>> r=Symbol("r", positive=True)
    >>> phi=Symbol("phi", real=True)
    >>> theta=Symbol("theta", real=True)
    >>> Z=Symbol("Z", positive=True, integer=True, nonzero=True)
    >>> Psi_nlm(1,0,0,r,phi,theta,Z)
    Z**(3/2)*exp(-Z*r)/sqrt(pi)
    >>> Psi_nlm(2,1,1,r,phi,theta,Z)
    -Z**(5/2)*r*exp(I*phi)*exp(-Z*r/2)*sin(theta)/(8*sqrt(pi))

    Integrating the absolute square of a hydrogen wavefunction psi_{nlm}
    over the whole space leads 1.

    The normalization of the hydrogen wavefunctions Psi_nlm is:

    >>> from sympy import integrate, conjugate, pi, oo, sin
    >>> wf=Psi_nlm(2,1,1,r,phi,theta,Z)
    >>> abs_sqrd=wf*conjugate(wf)
    >>> jacobi=r**2*sin(theta)
    >>> integrate(abs_sqrd*jacobi, (r,0,oo), (phi,0,2*pi), (theta,0,pi))
    1
    """

    # sympify arguments
    n, l, m, r, phi, theta, Z = map(S, [n, l, m, r, phi, theta, Z])
    # check if values for n,l,m make physically sense
    if n.is_integer and n < 1:
        raise ValueError("'n' must be positive integer")
    if l.is_integer and not (n > l):
        raise ValueError("'n' must be greater than 'l'")
    if m.is_integer and not (abs(m) <= l):
        raise ValueError("|'m'| must be less or equal 'l'")
    # return the hydrogen wave function
    return R_nl(n, l, r, Z)*Ynm(l, m, theta, phi).expand(func=True)


def E_nl(n, Z=1):
    """
    Returns the energy of the state (n, l) in Hartree atomic units.

    The energy does not depend on "l".

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)

    Examples
    ========

    >>> from sympy.physics.hydrogen import E_nl
    >>> from sympy.abc import n, Z
    >>> E_nl(n, Z)
    -Z**2/(2*n**2)
    >>> E_nl(1)
    -1/2
    >>> E_nl(2)
    -1/8
    >>> E_nl(3)
    -1/18
    >>> E_nl(3, 47)
    -2209/18

    """
    n, Z = S(n), S(Z)
    if n.is_integer and (n < 1):
        raise ValueError("'n' must be positive integer")
    return -Z**2/(2*n**2)


def E_nl_dirac(n, l, spin_up=True, Z=1, c=Float("137.035999037")):
    """
    Returns the relativistic energy of the state (n, l, spin) in Hartree atomic
    units.

    The energy is calculated from the Dirac equation. The rest mass energy is
    *not* included.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    spin_up :
        True if the electron spin is up (default), otherwise down
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)
    c :
        Speed of light in atomic units. Default value is 137.035999037,
        taken from https://arxiv.org/abs/1012.3627

    Examples
    ========

    >>> from sympy.physics.hydrogen import E_nl_dirac
    >>> E_nl_dirac(1, 0)
    -0.500006656595360

    >>> E_nl_dirac(2, 0)
    -0.125002080189006
    >>> E_nl_dirac(2, 1)
    -0.125000416028342
    >>> E_nl_dirac(2, 1, False)
    -0.125002080189006

    >>> E_nl_dirac(3, 0)
    -0.0555562951740285
    >>> E_nl_dirac(3, 1)
    -0.0555558020932949
    >>> E_nl_dirac(3, 1, False)
    -0.0555562951740285
    >>> E_nl_dirac(3, 2)
    -0.0555556377366884
    >>> E_nl_dirac(3, 2, False)
    -0.0555558020932949

    """
    n, l, Z, c = map(S, [n, l, Z, c])
    if not (l >= 0):
        raise ValueError("'l' must be positive or zero")
    if not (n > l):
        raise ValueError("'n' must be greater than 'l'")
    if (l == 0 and spin_up is False):
        raise ValueError("Spin must be up for l==0.")
    # skappa is sign*kappa, where sign contains the correct sign
    if spin_up:
        skappa = -l - 1
    else:
        skappa = -l
    beta = sqrt(skappa**2 - Z**2/c**2)
    return c**2/sqrt(1 + Z**2/(n + skappa + beta)**2/c**2) - c**2
