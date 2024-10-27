#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The module implements routines to model the polarization of optical fields
and can be used to calculate the effects of polarization optical elements on
the fields.

- Jones vectors.

- Stokes vectors.

- Jones matrices.

- Mueller matrices.

Examples
========

We calculate a generic Jones vector:

>>> from sympy import symbols, pprint, zeros, simplify
>>> from sympy.physics.optics.polarization import (jones_vector, stokes_vector,
...     half_wave_retarder, polarizing_beam_splitter, jones_2_stokes)

>>> psi, chi, p, I0 = symbols("psi, chi, p, I0", real=True)
>>> x0 = jones_vector(psi, chi)
>>> pprint(x0, use_unicode=True)
⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤
⎢                                ⎥
⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦

And the more general Stokes vector:

>>> s0 = stokes_vector(psi, chi, p, I0)
>>> pprint(s0, use_unicode=True)
⎡          I₀          ⎤
⎢                      ⎥
⎢I₀⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥
⎢                      ⎥
⎢I₀⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥
⎢                      ⎥
⎣    I₀⋅p⋅sin(2⋅χ)     ⎦

We calculate how the Jones vector is modified by a half-wave plate:

>>> alpha = symbols("alpha", real=True)
>>> HWP = half_wave_retarder(alpha)
>>> x1 = simplify(HWP*x0)

We calculate the very common operation of passing a beam through a half-wave
plate and then through a polarizing beam-splitter. We do this by putting this
Jones vector as the first entry of a two-Jones-vector state that is transformed
by a 4x4 Jones matrix modelling the polarizing beam-splitter to get the
transmitted and reflected Jones vectors:

>>> PBS = polarizing_beam_splitter()
>>> X1 = zeros(4, 1)
>>> X1[:2, :] = x1
>>> X2 = PBS*X1
>>> transmitted_port = X2[:2, :]
>>> reflected_port = X2[2:, :]

This allows us to calculate how the power in both ports depends on the initial
polarization:

>>> transmitted_power = jones_2_stokes(transmitted_port)[0]
>>> reflected_power = jones_2_stokes(reflected_port)[0]
>>> print(transmitted_power)
cos(-2*alpha + chi + psi)**2/2 + cos(2*alpha + chi - psi)**2/2


>>> print(reflected_power)
sin(-2*alpha + chi + psi)**2/2 + sin(2*alpha + chi - psi)**2/2

Please see the description of the individual functions for further
details and examples.

References
==========

.. [1] https://en.wikipedia.org/wiki/Jones_calculus
.. [2] https://en.wikipedia.org/wiki/Mueller_calculus
.. [3] https://en.wikipedia.org/wiki/Stokes_parameters

"""

from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct


def jones_vector(psi, chi):
    """A Jones vector corresponding to a polarization ellipse with `psi` tilt,
    and `chi` circularity.

    Parameters
    ==========

    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the `x` axis.

    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.


    Returns
    =======

    Matrix :
        A Jones vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> psi, chi = symbols("psi, chi", real=True)

    A general Jones vector.

    >>> pprint(jones_vector(psi, chi), use_unicode=True)
    ⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤
    ⎢                                ⎥
    ⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦

    Horizontal polarization.

    >>> pprint(jones_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎣0⎦

    Vertical polarization.

    >>> pprint(jones_vector(pi/2, 0), use_unicode=True)
    ⎡0⎤
    ⎢ ⎥
    ⎣1⎦

    Diagonal polarization.

    >>> pprint(jones_vector(pi/4, 0), use_unicode=True)
    ⎡√2⎤
    ⎢──⎥
    ⎢2 ⎥
    ⎢  ⎥
    ⎢√2⎥
    ⎢──⎥
    ⎣2 ⎦

    Anti-diagonal polarization.

    >>> pprint(jones_vector(-pi/4, 0), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢-√2 ⎥
    ⎢────⎥
    ⎣ 2  ⎦

    Right-hand circular polarization.

    >>> pprint(jones_vector(0, pi/4), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢√2⋅ⅈ⎥
    ⎢────⎥
    ⎣ 2  ⎦

    Left-hand circular polarization.

    >>> pprint(jones_vector(0, -pi/4), use_unicode=True)
    ⎡  √2  ⎤
    ⎢  ──  ⎥
    ⎢  2   ⎥
    ⎢      ⎥
    ⎢-√2⋅ⅈ ⎥
    ⎢──────⎥
    ⎣  2   ⎦

    """
    return Matrix([-I*sin(chi)*sin(psi) + cos(chi)*cos(psi),
                   I*sin(chi)*cos(psi) + sin(psi)*cos(chi)])


def stokes_vector(psi, chi, p=1, I=1):
    """A Stokes vector corresponding to a polarization ellipse with ``psi``
    tilt, and ``chi`` circularity.

    Parameters
    ==========

    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the ``x`` axis.
    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.
    p : numeric type or SymPy Symbol
        The degree of polarization.
    I : numeric type or SymPy Symbol
        The intensity of the field.


    Returns
    =======

    Matrix :
        A Stokes vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import stokes_vector
    >>> psi, chi, p, I = symbols("psi, chi, p, I", real=True)
    >>> pprint(stokes_vector(psi, chi, p, I), use_unicode=True)
    ⎡          I          ⎤
    ⎢                     ⎥
    ⎢I⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥
    ⎢                     ⎥
    ⎢I⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥
    ⎢                     ⎥
    ⎣    I⋅p⋅sin(2⋅χ)     ⎦


    Horizontal polarization

    >>> pprint(stokes_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    Vertical polarization

    >>> pprint(stokes_vector(pi/2, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣0 ⎦

    Diagonal polarization

    >>> pprint(stokes_vector(pi/4, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎣0⎦

    Anti-diagonal polarization

    >>> pprint(stokes_vector(-pi/4, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎣0 ⎦

    Right-hand circular polarization

    >>> pprint(stokes_vector(0, pi/4), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣1⎦

    Left-hand circular polarization

    >>> pprint(stokes_vector(0, -pi/4), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣-1⎦

    Unpolarized light

    >>> pprint(stokes_vector(0, 0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    """
    S0 = I
    S1 = I*p*cos(2*psi)*cos(2*chi)
    S2 = I*p*sin(2*psi)*cos(2*chi)
    S3 = I*p*sin(2*chi)
    return Matrix([S0, S1, S2, S3])


def jones_2_stokes(e):
    """Return the Stokes vector for a Jones vector ``e``.

    Parameters
    ==========

    e : SymPy Matrix
        A Jones vector.

    Returns
    =======

    SymPy Matrix
        A Jones vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> from sympy.physics.optics.polarization import jones_2_stokes
    >>> H = jones_vector(0, 0)
    >>> V = jones_vector(pi/2, 0)
    >>> D = jones_vector(pi/4, 0)
    >>> A = jones_vector(-pi/4, 0)
    >>> R = jones_vector(0, pi/4)
    >>> L = jones_vector(0, -pi/4)
    >>> pprint([jones_2_stokes(e) for e in [H, V, D, A, R, L]],
    ...         use_unicode=True)
    ⎡⎡1⎤  ⎡1 ⎤  ⎡1⎤  ⎡1 ⎤  ⎡1⎤  ⎡1 ⎤⎤
    ⎢⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥⎥
    ⎢⎢1⎥  ⎢-1⎥  ⎢0⎥  ⎢0 ⎥  ⎢0⎥  ⎢0 ⎥⎥
    ⎢⎢ ⎥, ⎢  ⎥, ⎢ ⎥, ⎢  ⎥, ⎢ ⎥, ⎢  ⎥⎥
    ⎢⎢0⎥  ⎢0 ⎥  ⎢1⎥  ⎢-1⎥  ⎢0⎥  ⎢0 ⎥⎥
    ⎢⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥⎥
    ⎣⎣0⎦  ⎣0 ⎦  ⎣0⎦  ⎣0 ⎦  ⎣1⎦  ⎣-1⎦⎦

    """
    ex, ey = e
    return Matrix([Abs(ex)**2 + Abs(ey)**2,
                   Abs(ex)**2 - Abs(ey)**2,
                   2*re(ex*ey.conjugate()),
                   -2*im(ex*ey.conjugate())])


def linear_polarizer(theta=0):
    """A linear polarizer Jones matrix with transmission axis at
    an angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the transmission axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the polarizer.

    Examples
    ========

    A generic polarizer.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import linear_polarizer
    >>> theta = symbols("theta", real=True)
    >>> J = linear_polarizer(theta)
    >>> pprint(J, use_unicode=True)
    ⎡      2                     ⎤
    ⎢   cos (θ)     sin(θ)⋅cos(θ)⎥
    ⎢                            ⎥
    ⎢                     2      ⎥
    ⎣sin(θ)⋅cos(θ)     sin (θ)   ⎦


    """
    M = Matrix([[cos(theta)**2, sin(theta)*cos(theta)],
                [sin(theta)*cos(theta), sin(theta)**2]])
    return M


def phase_retarder(theta=0, delta=0):
    """A phase retarder Jones matrix with retardance ``delta`` at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.
    delta : numeric type or SymPy Symbol
        The phase difference between the fast and slow axes of the
        transmitted light.

    Returns
    =======

    SymPy Matrix :
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic retarder.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import phase_retarder
    >>> theta, delta = symbols("theta, delta", real=True)
    >>> R = phase_retarder(theta, delta)
    >>> pprint(R, use_unicode=True)
    ⎡                          -ⅈ⋅δ               -ⅈ⋅δ               ⎤
    ⎢                          ─────              ─────              ⎥
    ⎢⎛ ⅈ⋅δ    2         2   ⎞    2    ⎛     ⅈ⋅δ⎞    2                ⎥
    ⎢⎝ℯ   ⋅sin (θ) + cos (θ)⎠⋅ℯ       ⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥
    ⎢                                                                ⎥
    ⎢            -ⅈ⋅δ                                           -ⅈ⋅δ ⎥
    ⎢            ─────                                          ─────⎥
    ⎢⎛     ⅈ⋅δ⎞    2                  ⎛ ⅈ⋅δ    2         2   ⎞    2  ⎥
    ⎣⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝ℯ   ⋅cos (θ) + sin (θ)⎠⋅ℯ     ⎦

    """
    R = Matrix([[cos(theta)**2 + exp(I*delta)*sin(theta)**2,
                (1-exp(I*delta))*cos(theta)*sin(theta)],
                [(1-exp(I*delta))*cos(theta)*sin(theta),
                sin(theta)**2 + exp(I*delta)*cos(theta)**2]])
    return R*exp(-I*delta/2)


def half_wave_retarder(theta):
    """A half-wave retarder Jones matrix at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic half-wave plate.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import half_wave_retarder
    >>> theta= symbols("theta", real=True)
    >>> HWP = half_wave_retarder(theta)
    >>> pprint(HWP, use_unicode=True)
    ⎡   ⎛     2         2   ⎞                        ⎤
    ⎢-ⅈ⋅⎝- sin (θ) + cos (θ)⎠    -2⋅ⅈ⋅sin(θ)⋅cos(θ)  ⎥
    ⎢                                                ⎥
    ⎢                             ⎛   2         2   ⎞⎥
    ⎣   -2⋅ⅈ⋅sin(θ)⋅cos(θ)     -ⅈ⋅⎝sin (θ) - cos (θ)⎠⎦

    """
    return phase_retarder(theta, pi)


def quarter_wave_retarder(theta):
    """A quarter-wave retarder Jones matrix at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic quarter-wave plate.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import quarter_wave_retarder
    >>> theta= symbols("theta", real=True)
    >>> QWP = quarter_wave_retarder(theta)
    >>> pprint(QWP, use_unicode=True)
    ⎡                       -ⅈ⋅π            -ⅈ⋅π               ⎤
    ⎢                       ─────           ─────              ⎥
    ⎢⎛     2         2   ⎞    4               4                ⎥
    ⎢⎝ⅈ⋅sin (θ) + cos (θ)⎠⋅ℯ       (1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥
    ⎢                                                          ⎥
    ⎢         -ⅈ⋅π                                        -ⅈ⋅π ⎥
    ⎢         ─────                                       ─────⎥
    ⎢           4                  ⎛   2           2   ⎞    4  ⎥
    ⎣(1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝sin (θ) + ⅈ⋅cos (θ)⎠⋅ℯ     ⎦

    """
    return phase_retarder(theta, pi/2)


def transmissive_filter(T):
    """An attenuator Jones matrix with transmittance ``T``.

    Parameters
    ==========

    T : numeric type or SymPy Symbol
        The transmittance of the attenuator.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the filter.

    Examples
    ========

    A generic filter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import transmissive_filter
    >>> T = symbols("T", real=True)
    >>> NDF = transmissive_filter(T)
    >>> pprint(NDF, use_unicode=True)
    ⎡√T  0 ⎤
    ⎢      ⎥
    ⎣0   √T⎦

    """
    return Matrix([[sqrt(T), 0], [0, sqrt(T)]])


def reflective_filter(R):
    """A reflective filter Jones matrix with reflectance ``R``.

    Parameters
    ==========

    R : numeric type or SymPy Symbol
        The reflectance of the filter.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the filter.

    Examples
    ========

    A generic filter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import reflective_filter
    >>> R = symbols("R", real=True)
    >>> pprint(reflective_filter(R), use_unicode=True)
    ⎡√R   0 ⎤
    ⎢       ⎥
    ⎣0   -√R⎦

    """
    return Matrix([[sqrt(R), 0], [0, -sqrt(R)]])


def mueller_matrix(J):
    """The Mueller matrix corresponding to Jones matrix `J`.

    Parameters
    ==========

    J : SymPy Matrix
        A Jones matrix.

    Returns
    =======

    SymPy Matrix
        The corresponding Mueller matrix.

    Examples
    ========

    Generic optical components.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import (mueller_matrix,
    ...     linear_polarizer, half_wave_retarder, quarter_wave_retarder)
    >>> theta = symbols("theta", real=True)

    A linear_polarizer

    >>> pprint(mueller_matrix(linear_polarizer(theta)), use_unicode=True)
    ⎡            cos(2⋅θ)      sin(2⋅θ)     ⎤
    ⎢  1/2       ────────      ────────    0⎥
    ⎢               2             2         ⎥
    ⎢                                       ⎥
    ⎢cos(2⋅θ)  cos(4⋅θ)   1    sin(4⋅θ)     ⎥
    ⎢────────  ──────── + ─    ────────    0⎥
    ⎢   2         4       4       4         ⎥
    ⎢                                       ⎥
    ⎢sin(2⋅θ)    sin(4⋅θ)    1   cos(4⋅θ)   ⎥
    ⎢────────    ────────    ─ - ────────  0⎥
    ⎢   2           4        4      4       ⎥
    ⎢                                       ⎥
    ⎣   0           0             0        0⎦

    A half-wave plate

    >>> pprint(mueller_matrix(half_wave_retarder(theta)), use_unicode=True)
    ⎡1              0                           0               0 ⎤
    ⎢                                                             ⎥
    ⎢        4           2                                        ⎥
    ⎢0  8⋅sin (θ) - 8⋅sin (θ) + 1           sin(4⋅θ)            0 ⎥
    ⎢                                                             ⎥
    ⎢                                     4           2           ⎥
    ⎢0          sin(4⋅θ)           - 8⋅sin (θ) + 8⋅sin (θ) - 1  0 ⎥
    ⎢                                                             ⎥
    ⎣0              0                           0               -1⎦

    A quarter-wave plate

    >>> pprint(mueller_matrix(quarter_wave_retarder(theta)), use_unicode=True)
    ⎡1       0             0            0    ⎤
    ⎢                                        ⎥
    ⎢   cos(4⋅θ)   1    sin(4⋅θ)             ⎥
    ⎢0  ──────── + ─    ────────    -sin(2⋅θ)⎥
    ⎢      2       2       2                 ⎥
    ⎢                                        ⎥
    ⎢     sin(4⋅θ)    1   cos(4⋅θ)           ⎥
    ⎢0    ────────    ─ - ────────  cos(2⋅θ) ⎥
    ⎢        2        2      2               ⎥
    ⎢                                        ⎥
    ⎣0    sin(2⋅θ)     -cos(2⋅θ)        0    ⎦

    """
    A = Matrix([[1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 1, 1, 0],
                [0, -I, I, 0]])

    return simplify(A*TensorProduct(J, J.conjugate())*A.inv())


def polarizing_beam_splitter(Tp=1, Rs=1, Ts=0, Rp=0, phia=0, phib=0):
    r"""A polarizing beam splitter Jones matrix at angle `theta`.

    Parameters
    ==========

    J : SymPy Matrix
        A Jones matrix.
    Tp : numeric type or SymPy Symbol
        The transmissivity of the P-polarized component.
    Rs : numeric type or SymPy Symbol
        The reflectivity of the S-polarized component.
    Ts : numeric type or SymPy Symbol
        The transmissivity of the S-polarized component.
    Rp : numeric type or SymPy Symbol
        The reflectivity of the P-polarized component.
    phia : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode a.
    phib : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode b.


    Returns
    =======

    SymPy Matrix
        A 4x4 matrix representing the PBS. This matrix acts on a 4x1 vector
        whose first two entries are the Jones vector on one of the PBS ports,
        and the last two entries the Jones vector on the other port.

    Examples
    ========

    Generic polarizing beam-splitter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import polarizing_beam_splitter
    >>> Ts, Rs, Tp, Rp = symbols(r"Ts, Rs, Tp, Rp", positive=True)
    >>> phia, phib = symbols("phi_a, phi_b", real=True)
    >>> PBS = polarizing_beam_splitter(Tp, Rs, Ts, Rp, phia, phib)
    >>> pprint(PBS, use_unicode=False)
    [   ____                           ____                    ]
    [ \/ Tp            0           I*\/ Rp           0         ]
    [                                                          ]
    [                  ____                       ____  I*phi_a]
    [   0            \/ Ts            0      -I*\/ Rs *e       ]
    [                                                          ]
    [    ____                         ____                     ]
    [I*\/ Rp           0            \/ Tp            0         ]
    [                                                          ]
    [               ____  I*phi_b                    ____      ]
    [   0      -I*\/ Rs *e            0            \/ Ts       ]

    """
    PBS = Matrix([[sqrt(Tp), 0, I*sqrt(Rp), 0],
                  [0, sqrt(Ts), 0, -I*sqrt(Rs)*exp(I*phia)],
                  [I*sqrt(Rp), 0, sqrt(Tp), 0],
                  [0, -I*sqrt(Rs)*exp(I*phib), 0, sqrt(Ts)]])
    return PBS
