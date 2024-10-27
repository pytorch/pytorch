# -*- coding: utf-8 -*-
r"""
Wigner, Clebsch-Gordan, Racah, and Gaunt coefficients

Collection of functions for calculating Wigner 3j, 6j, 9j,
Clebsch-Gordan, Racah as well as Gaunt coefficients exactly, all
evaluating to a rational number times the square root of a rational
number [Rasch03]_.

Please see the description of the individual functions for further
details and examples.

References
==========

.. [Regge58] 'Symmetry Properties of Clebsch-Gordan Coefficients',
  T. Regge, Nuovo Cimento, Volume 10, pp. 544 (1958)
.. [Regge59] 'Symmetry Properties of Racah Coefficients',
  T. Regge, Nuovo Cimento, Volume 11, pp. 116 (1959)
.. [Edmonds74] A. R. Edmonds. Angular momentum in quantum mechanics.
  Investigations in physics, 4.; Investigations in physics, no. 4.
  Princeton, N.J., Princeton University Press, 1957.
.. [Rasch03] J. Rasch and A. C. H. Yu, 'Efficient Storage Scheme for
  Pre-calculated Wigner 3j, 6j and Gaunt Coefficients', SIAM
  J. Sci. Comput. Volume 25, Issue 4, pp. 1416-1428 (2003)
.. [Liberatodebrito82] 'FORTRAN program for the integral of three
  spherical harmonics', A. Liberato de Brito,
  Comput. Phys. Commun., Volume 25, pp. 81-85 (1982)
.. [Homeier96] 'Some Properties of the Coupling Coefficients of Real
  Spherical Harmonics and Their Relation to Gaunt Coefficients',
  H. H. H. Homeier and E. O. Steinborn J. Mol. Struct., Volume 368,
  pp. 31-37 (1996)

Credits and Copyright
=====================

This code was taken from Sage with the permission of all authors:

https://groups.google.com/forum/#!topic/sage-devel/M4NZdu-7O38

Authors
=======

- Jens Rasch (2009-03-24): initial version for Sage

- Jens Rasch (2009-05-31): updated to sage-4.0

- Oscar Gerardo Lazo Arjona (2017-06-18): added Wigner D matrices

- Phil Adam LeMaitre (2022-09-19): added real Gaunt coefficient

Copyright (C) 2008 Jens Rasch <jyr2000@gmail.com>

"""
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.numbers import int_valued
from sympy.core.function import Function
from sympy.core.numbers import (Float, I, Integer, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.utilities.misc import as_int

# This list of precomputed factorials is needed to massively
# accelerate future calculations of the various coefficients
_Factlist = [1]


def _calc_factlist(nn):
    r"""
    Function calculates a list of precomputed factorials in order to
    massively accelerate future calculations of the various
    coefficients.

    Parameters
    ==========

    nn : integer
        Highest factorial to be computed.

    Returns
    =======

    list of integers :
        The list of precomputed factorials.

    Examples
    ========

    Calculate list of factorials::

        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    """
    if nn >= len(_Factlist):
        for ii in range(len(_Factlist), int(nn + 1)):
            _Factlist.append(_Factlist[ii - 1] * ii)
    return _Factlist[:int(nn) + 1]


def _int_or_halfint(value):
    """return Python int unless value is half-int (then return float)"""
    if isinstance(value, int):
        return value
    elif type(value) is float:
        if value.is_integer():
            return int(value)  # an int
        if (2*value).is_integer():
            return value  # a float
    elif isinstance(value, Rational):
        if value.q == 2:
            return value.p/value.q  # a float
        elif value.q == 1:
            return value.p  # an int
    elif isinstance(value, Float):
        return _int_or_halfint(float(value))
    raise ValueError("expecting integer or half-integer, got %s" % value)


def wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    Calculate the Wigner 3j symbol `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.

    Parameters
    ==========

    j_1, j_2, j_3, m_1, m_2, m_3 :
        Integer or half integer.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0

    It is an error to have arguments that are not integer or half
    integer values::

        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer

    Notes
    =====

    The Wigner 3j symbol obeys the following symmetry rules:

    - invariant under any permutation of the columns (with the
      exception of a sign change where `J:=j_1+j_2+j_3`):

      .. math::

         \begin{aligned}
         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
          &=\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\
          &=\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\
          &=(-1)^J \operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)
         \end{aligned}

    - invariant under space inflection, i.e.

      .. math::

         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
         =(-1)^J \operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 additional symmetries based on
      the work by [Regge58]_

    - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation

    - zero for `m_1 + m_2 + m_3 \neq 0`

    - zero for violating any one of the conditions
         `m_1  \in \{-|j_1|, \ldots, |j_1|\}`,
         `m_2  \in \{-|j_2|, \ldots, |j_2|\}`,
         `m_3  \in \{-|j_3|, \ldots, |j_3|\}`

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """

    j_1, j_2, j_3, m_1, m_2, m_3 = map(_int_or_halfint,
                                       [j_1, j_2, j_3, m_1, m_2, m_3])

    if m_1 + m_2 + m_3 != 0:
        return S.Zero
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return S.Zero
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return S.Zero
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return S.Zero
    if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
        return S.Zero
    if not (int_valued(j_1 - m_1) and \
            int_valued(j_2 - m_2) and \
            int_valued(j_3 - m_3)):
        return S.Zero

    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2),
                  j_3 + abs(m_3))
    _calc_factlist(int(maxfact))

    argsqrt = Integer(_Factlist[int(j_1 + j_2 - j_3)] *
                     _Factlist[int(j_1 - j_2 + j_3)] *
                     _Factlist[int(-j_1 + j_2 + j_3)] *
                     _Factlist[int(j_1 - m_1)] *
                     _Factlist[int(j_1 + m_1)] *
                     _Factlist[int(j_2 - m_2)] *
                     _Factlist[int(j_2 + m_2)] *
                     _Factlist[int(j_3 - m_3)] *
                     _Factlist[int(j_3 + m_3)]) / \
        _Factlist[int(j_1 + j_2 + j_3 + 1)]

    ressqrt = sqrt(argsqrt)
    if ressqrt.is_complex or ressqrt.is_infinite:
        ressqrt = ressqrt.as_real_imag()[0]

    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * \
            _Factlist[int(ii + j_3 - j_1 - m_2)] * \
            _Factlist[int(j_2 + m_2 - ii)] * \
            _Factlist[int(j_1 - ii - m_1)] * \
            _Factlist[int(ii + j_3 - j_2 + m_1)] * \
            _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + Integer((-1) ** ii) / den

    prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    res = ressqrt * sumres * prefid
    return res


def clebsch_gordan(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    Calculates the Clebsch-Gordan coefficient.
    `\left\langle j_1 m_1 \; j_2 m_2 | j_3 m_3 \right\rangle`.

    The reference for this function is [Edmonds74]_.

    Parameters
    ==========

    j_1, j_2, j_3, m_1, m_2, m_3 :
        Integer or half integer.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.physics.wigner import clebsch_gordan
    >>> clebsch_gordan(S(3)/2, S(1)/2, 2, S(3)/2, S(1)/2, 2)
    1
    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, S(3)/2, -S(1)/2, 1)
    sqrt(3)/2
    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, -S(1)/2, S(1)/2, 0)
    -sqrt(2)/2

    Notes
    =====

    The Clebsch-Gordan coefficient will be evaluated via its relation
    to Wigner 3j symbols:

    .. math::

        \left\langle j_1 m_1 \; j_2 m_2 | j_3 m_3 \right\rangle
        =(-1)^{j_1-j_2+m_3} \sqrt{2j_3+1}
        \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,-m_3)

    See also the documentation on Wigner 3j symbols which exhibit much
    higher symmetry relations than the Clebsch-Gordan coefficient.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """
    res = (-1) ** sympify(j_1 - j_2 + m_3) * sqrt(2 * j_3 + 1) * \
        wigner_3j(j_1, j_2, j_3, m_1, m_2, -m_3)
    return res


def _big_delta_coeff(aa, bb, cc, prec=None):
    r"""
    Calculates the Delta coefficient of the 3 angular momenta for
    Racah symbols. Also checks that the differences are of integer
    value.

    Parameters
    ==========

    aa :
        First angular momentum, integer or half integer.
    bb :
        Second angular momentum, integer or half integer.
    cc :
        Third angular momentum, integer or half integer.
    prec :
        Precision of the ``sqrt()`` calculation.

    Returns
    =======

    double : Value of the Delta coefficient.

    Examples
    ========

        sage: from sage.functions.wigner import _big_delta_coeff
        sage: _big_delta_coeff(1,1,1)
        1/2*sqrt(1/6)
    """

    # the triangle test will only pass if a) all 3 values are ints or
    # b) 1 is an int and the other two are half-ints
    if not int_valued(aa + bb - cc):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if not int_valued(aa + cc - bb):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if not int_valued(bb + cc - aa):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if (aa + bb - cc) < 0:
        return S.Zero
    if (aa + cc - bb) < 0:
        return S.Zero
    if (bb + cc - aa) < 0:
        return S.Zero

    maxfact = max(aa + bb - cc, aa + cc - bb, bb + cc - aa, aa + bb + cc + 1)
    _calc_factlist(maxfact)

    argsqrt = Integer(_Factlist[int(aa + bb - cc)] *
                     _Factlist[int(aa + cc - bb)] *
                     _Factlist[int(bb + cc - aa)]) / \
        Integer(_Factlist[int(aa + bb + cc + 1)])

    ressqrt = sqrt(argsqrt)
    if prec:
        ressqrt = ressqrt.evalf(prec).as_real_imag()[0]
    return ressqrt


def racah(aa, bb, cc, dd, ee, ff, prec=None):
    r"""
    Calculate the Racah symbol `W(a,b,c,d;e,f)`.

    Parameters
    ==========

    a, ..., f :
        Integer or half integer.
    prec :
        Precision, default: ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import racah
    >>> racah(3,3,3,3,3,3)
    -1/14

    Notes
    =====

    The Racah symbol is related to the Wigner 6j symbol:

    .. math::

       \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
       =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)

    Please see the 6j symbol for its much richer symmetries and for
    additional properties.

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 6j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """
    prefac = _big_delta_coeff(aa, bb, ee, prec) * \
        _big_delta_coeff(cc, dd, ee, prec) * \
        _big_delta_coeff(aa, cc, ff, prec) * \
        _big_delta_coeff(bb, dd, ff, prec)
    if prefac == 0:
        return S.Zero
    imin = max(aa + bb + ee, cc + dd + ee, aa + cc + ff, bb + dd + ff)
    imax = min(aa + bb + cc + dd, aa + dd + ee + ff, bb + cc + ee + ff)

    maxfact = max(imax + 1, aa + bb + cc + dd, aa + dd + ee + ff,
                 bb + cc + ee + ff)
    _calc_factlist(maxfact)

    sumres = 0
    for kk in range(int(imin), int(imax) + 1):
        den = _Factlist[int(kk - aa - bb - ee)] * \
            _Factlist[int(kk - cc - dd - ee)] * \
            _Factlist[int(kk - aa - cc - ff)] * \
            _Factlist[int(kk - bb - dd - ff)] * \
            _Factlist[int(aa + bb + cc + dd - kk)] * \
            _Factlist[int(aa + dd + ee + ff - kk)] * \
            _Factlist[int(bb + cc + ee + ff - kk)]
        sumres = sumres + Integer((-1) ** kk * _Factlist[kk + 1]) / den

    res = prefac * sumres * (-1) ** int(aa + bb + cc + dd)
    return res


def wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6, prec=None):
    r"""
    Calculate the Wigner 6j symbol `\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)`.

    Parameters
    ==========

    j_1, ..., j_6 :
        Integer or half integer.
    prec :
        Precision, default: ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_6j
    >>> wigner_6j(3,3,3,3,3,3)
    -1/14
    >>> wigner_6j(5,5,5,5,5,5)
    1/52

    It is an error to have arguments that are not integer or half
    integer values or do not fulfill the triangle relation::

        sage: wigner_6j(2.5,2.5,2.5,2.5,2.5,2.5)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation
        sage: wigner_6j(0.5,0.5,1.1,0.5,0.5,1.1)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation

    Notes
    =====

    The Wigner 6j symbol is related to the Racah symbol but exhibits
    more symmetries as detailed below.

    .. math::

       \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
        =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)

    The Wigner 6j symbol obeys the following symmetry rules:

    - Wigner 6j symbols are left invariant under any permutation of
      the columns:

      .. math::

         \begin{aligned}
         \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
          &=\operatorname{Wigner6j}(j_3,j_1,j_2,j_6,j_4,j_5) \\
          &=\operatorname{Wigner6j}(j_2,j_3,j_1,j_5,j_6,j_4) \\
          &=\operatorname{Wigner6j}(j_3,j_2,j_1,j_6,j_5,j_4) \\
          &=\operatorname{Wigner6j}(j_1,j_3,j_2,j_4,j_6,j_5) \\
          &=\operatorname{Wigner6j}(j_2,j_1,j_3,j_5,j_4,j_6)
         \end{aligned}

    - They are invariant under the exchange of the upper and lower
      arguments in each of any two columns, i.e.

      .. math::

         \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
          =\operatorname{Wigner6j}(j_1,j_5,j_6,j_4,j_2,j_3)
          =\operatorname{Wigner6j}(j_4,j_2,j_6,j_1,j_5,j_3)
          =\operatorname{Wigner6j}(j_4,j_5,j_3,j_1,j_2,j_6)

    - additional 6 symmetries [Regge59]_ giving rise to 144 symmetries
      in total

    - only non-zero if any triple of `j`'s fulfill a triangle relation

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 6j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    """
    res = (-1) ** int(j_1 + j_2 + j_4 + j_5) * \
        racah(j_1, j_2, j_5, j_4, j_3, j_6, prec)
    return res


def wigner_9j(j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9, prec=None):
    r"""
    Calculate the Wigner 9j symbol
    `\operatorname{Wigner9j}(j_1,j_2,j_3,j_4,j_5,j_6,j_7,j_8,j_9)`.

    Parameters
    ==========

    j_1, ..., j_9 :
        Integer or half integer.
    prec : precision, default
        ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_9j
    >>> wigner_9j(1,1,1, 1,1,1, 1,1,0, prec=64)
    0.05555555555555555555555555555555555555555555555555555555555555555

    >>> wigner_9j(1/2,1/2,0, 1/2,3/2,1, 0,1,1, prec=64)
    0.1666666666666666666666666666666666666666666666666666666666666667

    It is an error to have arguments that are not integer or half
    integer values or do not fulfill the triangle relation::

        sage: wigner_9j(0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5,prec=64)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation
        sage: wigner_9j(1,1,1, 0.5,1,1.5, 0.5,1,2.5,prec=64)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.
    """
    imax = int(min(j_1 + j_9, j_2 + j_6, j_4 + j_8) * 2)
    imin = imax % 2
    sumres = 0
    for kk in range(imin, int(imax) + 1, 2):
        sumres = sumres + (kk + 1) * \
            racah(j_1, j_2, j_9, j_6, j_3, kk / 2, prec) * \
            racah(j_4, j_6, j_8, j_2, j_5, kk / 2, prec) * \
            racah(j_1, j_4, j_9, j_8, j_7, kk / 2, prec)
    return sumres


def gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    r"""
    Calculate the Gaunt coefficient.

    Explanation
    ===========

    The Gaunt coefficient is defined as the integral over three
    spherical harmonics:

    .. math::

        \begin{aligned}
        \operatorname{Gaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\int Y_{l_1,m_1}(\Omega)
         Y_{l_2,m_2}(\Omega) Y_{l_3,m_3}(\Omega) \,d\Omega \\
        &=\sqrt{\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\pi}}
         \operatorname{Wigner3j}(l_1,l_2,l_3,0,0,0)
         \operatorname{Wigner3j}(l_1,l_2,l_3,m_1,m_2,m_3)
        \end{aligned}

    Parameters
    ==========

    l_1, l_2, l_3, m_1, m_2, m_3 :
        Integer.
    prec - precision, default: ``None``.
        Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import gaunt
    >>> gaunt(1,0,1,1,0,-1)
    -1/(2*sqrt(pi))
    >>> gaunt(1000,1000,1200,9,3,-12).n(64)
    0.006895004219221134484332976156744208248842039317638217822322799675

    It is an error to use non-integer values for `l` and `m`::

        sage: gaunt(1.2,0,1.2,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer
        sage: gaunt(1,0,1,1.1,0,-1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer

    Notes
    =====

    The Gaunt coefficient obeys the following symmetry rules:

    - invariant under any permutation of the columns

      .. math::
        \begin{aligned}
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          &=Y(l_3,l_1,l_2,m_3,m_1,m_2) \\
          &=Y(l_2,l_3,l_1,m_2,m_3,m_1) \\
          &=Y(l_3,l_2,l_1,m_3,m_2,m_1) \\
          &=Y(l_1,l_3,l_2,m_1,m_3,m_2) \\
          &=Y(l_2,l_1,l_3,m_2,m_1,m_3)
        \end{aligned}

    - invariant under space inflection, i.e.

      .. math::
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          =Y(l_1,l_2,l_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 Regge symmetries as inherited
      for the `3j` symbols [Regge58]_

    - zero for `l_1`, `l_2`, `l_3` not fulfilling triangle relation

    - zero for violating any one of the conditions: `l_1 \ge |m_1|`,
      `l_2 \ge |m_2|`, `l_3 \ge |m_3|`

    - non-zero only for an even sum of the `l_i`, i.e.
      `L = l_1 + l_2 + l_3 = 2n` for `n` in `\mathbb{N}`

    Algorithms
    ==========

    This function uses the algorithm of [Liberatodebrito82]_ to
    calculate the value of the Gaunt coefficient exactly. Note that
    the formula contains alternating sums over large factorials and is
    therefore unsuitable for finite precision arithmetic and only
    useful for a computer algebra system [Rasch03]_.

    Authors
    =======

    Jens Rasch (2009-03-24): initial version for Sage.
    """
    l_1, l_2, l_3, m_1, m_2, m_3 = [
        as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]

    if l_1 + l_2 - l_3 < 0:
        return S.Zero
    if l_1 - l_2 + l_3 < 0:
        return S.Zero
    if -l_1 + l_2 + l_3 < 0:
        return S.Zero
    if (m_1 + m_2 + m_3) != 0:
        return S.Zero
    if (abs(m_1) > l_1) or (abs(m_2) > l_2) or (abs(m_3) > l_3):
        return S.Zero
    bigL, remL = divmod(l_1 + l_2 + l_3, 2)
    if remL % 2:
        return S.Zero

    imin = max(-l_3 + l_1 + m_2, -l_3 + l_2 - m_1, 0)
    imax = min(l_2 + m_2, l_1 - m_1, l_1 + l_2 - l_3)

    _calc_factlist(max(l_1 + l_2 + l_3 + 1, imax + 1))

    ressqrt = sqrt((2 * l_1 + 1) * (2 * l_2 + 1) * (2 * l_3 + 1) * \
        _Factlist[l_1 - m_1] * _Factlist[l_1 + m_1] * _Factlist[l_2 - m_2] * \
        _Factlist[l_2 + m_2] * _Factlist[l_3 - m_3] * _Factlist[l_3 + m_3] / \
        (4*pi))

    prefac = Integer(_Factlist[bigL] * _Factlist[l_2 - l_1 + l_3] *
                     _Factlist[l_1 - l_2 + l_3] * _Factlist[l_1 + l_2 - l_3])/ \
        _Factlist[2 * bigL + 1]/ \
        (_Factlist[bigL - l_1] *
         _Factlist[bigL - l_2] * _Factlist[bigL - l_3])

    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * _Factlist[ii + l_3 - l_1 - m_2] * \
            _Factlist[l_2 + m_2 - ii] * _Factlist[l_1 - ii - m_1] * \
            _Factlist[ii + l_3 - l_2 + m_1] * _Factlist[l_1 + l_2 - l_3 - ii]
        sumres = sumres + Integer((-1) ** ii) / den

    res = ressqrt * prefac * sumres * Integer((-1) ** (bigL + l_3 + m_1 - m_2))
    if prec is not None:
        res = res.n(prec)
    return res


def real_gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    r"""
    Calculate the real Gaunt coefficient.

    Explanation
    ===========

    The real Gaunt coefficient is defined as the integral over three
    real spherical harmonics:

    .. math::
        \begin{aligned}
        \operatorname{RealGaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\int Z^{m_1}_{l_1}(\Omega)
         Z^{m_2}_{l_2}(\Omega) Z^{m_3}_{l_3}(\Omega) \,d\Omega \\
        \end{aligned}

    Alternatively, it can be defined in terms of the standard Gaunt
    coefficient by relating the real spherical harmonics to the standard
    spherical harmonics via a unitary transformation `U`, i.e.
    `Z^{m}_{l}(\Omega)=\sum_{m'}U^{m}_{m'}Y^{m'}_{l}(\Omega)` [Homeier96]_.
    The real Gaunt coefficient is then defined as

    .. math::
        \begin{aligned}
        \operatorname{RealGaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\int Z^{m_1}_{l_1}(\Omega)
         Z^{m_2}_{l_2}(\Omega) Z^{m_3}_{l_3}(\Omega) \,d\Omega \\
        &=\sum_{m'_1 m'_2 m'_3} U^{m_1}_{m'_1}U^{m_2}_{m'_2}U^{m_3}_{m'_3}
         \operatorname{Gaunt}(l_1,l_2,l_3,m'_1,m'_2,m'_3)
        \end{aligned}

    The unitary matrix `U` has components

    .. math::
        \begin{aligned}
        U^m_{m'} = \delta_{|m||m'|}*(\delta_{m'0}\delta_{m0} + \frac{1}{\sqrt{2}}\big[\Theta(m)
        \big(\delta_{m'm}+(-1)^{m'}\delta_{m'-m}\big)+i\Theta(-m)\big((-1)^{-m}
        \delta_{m'-m}-\delta_{m'm}*(-1)^{m'-m}\big)\big])
        \end{aligned}

    where `\delta_{ij}` is the Kronecker delta symbol and `\Theta` is a step
    function defined as

    .. math::
        \begin{aligned}
        \Theta(x) = \begin{cases} 1 \,\text{for}\, x > 0 \\ 0 \,\text{for}\, x \leq 0 \end{cases}
        \end{aligned}

    Parameters
    ==========

    l_1, l_2, l_3, m_1, m_2, m_3 :
        Integer.

    prec - precision, default: ``None``.
        Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import real_gaunt
    >>> real_gaunt(2,2,4,-1,-1,0)
    -2/(7*sqrt(pi))
    >>> real_gaunt(10,10,20,-9,-9,0).n(64)
    -0.00002480019791932209313156167176797577821140084216297395518482071448

    It is an error to use non-integer values for `l` and `m`::
        real_gaunt(2.8,0.5,1.3,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer
        real_gaunt(2,2,4,0.7,1,-3.4)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer

    Notes
    =====

    The real Gaunt coefficient inherits from the standard Gaunt coefficient,
    the invariance under any permutation of the pairs `(l_i, m_i)` and the
    requirement that the sum of the `l_i` be even to yield a non-zero value.
    It also obeys the following symmetry rules:

    - zero for `l_1`, `l_2`, `l_3` not fulfiling the condition
      `l_1 \in \{l_{\text{max}}, l_{\text{max}}-2, \ldots, l_{\text{min}}\}`,
      where `l_{\text{max}} = l_2+l_3`,

      .. math::
          \begin{aligned}
          l_{\text{min}} = \begin{cases} \kappa(l_2, l_3, m_2, m_3) & \text{if}\,
          \kappa(l_2, l_3, m_2, m_3) + l_{\text{max}}\, \text{is even} \\
          \kappa(l_2, l_3, m_2, m_3)+1 & \text{if}\, \kappa(l_2, l_3, m_2, m_3) +
          l_{\text{max}}\, \text{is odd}\end{cases}
          \end{aligned}

      and `\kappa(l_2, l_3, m_2, m_3) = \max{\big(|l_2-l_3|, \min{\big(|m_2+m_3|,
      |m_2-m_3|\big)}\big)}`

    - zero for an odd number of negative `m_i`

    Algorithms
    ==========

    This function uses the algorithms of [Homeier96]_ and [Rasch03]_ to
    calculate the value of the real Gaunt coefficient exactly. Note that
    the formula used in [Rasch03]_ contains alternating sums over large
    factorials and is therefore unsuitable for finite precision arithmetic
    and only useful for a computer algebra system [Rasch03]_. However, this
    function can in principle use any algorithm that computes the Gaunt
    coefficient, so it is suitable for finite precision arithmetic in so far
    as the algorithm which computes the Gaunt coefficient is.
    """
    l_1, l_2, l_3, m_1, m_2, m_3 = [
        as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]

    # check for quick exits
    if sum(1 for i in (m_1, m_2, m_3) if i < 0) % 2:
        return S.Zero  # odd number of negative m
    if (l_1 + l_2 + l_3) % 2:
        return S.Zero  # sum of l is odd
    lmax = l_2 + l_3
    lmin = max(abs(l_2 - l_3), min(abs(m_2 + m_3), abs(m_2 - m_3)))
    if (lmin + lmax) % 2:
        lmin += 1
    if lmin not in range(lmax, lmin - 2, -2):
        return S.Zero

    kron_del = lambda i, j: 1 if i == j else 0
    s = lambda e: -1 if e % 2 else 1  #  (-1)**e to give +/-1, avoiding float when e<0
    A = lambda a, b: (-kron_del(a, b)*s(a-b) + kron_del(a, -b)*
                      s(b)) if b < 0 else 0
    B = lambda a, b: (kron_del(a, b) + kron_del(a, -b)*s(a)) if b > 0 else 0
    C = lambda a, b: kron_del(abs(a), abs(b))*(kron_del(a, 0)*kron_del(b, 0) +
                                          (B(a, b) + I*A(a, b))/sqrt(2))
    ugnt = 0
    for i in range(-l_1, l_1+1):
        U1 = C(i, m_1)
        for j in range(-l_2, l_2+1):
            U2 = C(j, m_2)
            U3 = C(-i-j, m_3)
            ugnt = ugnt + re(U1*U2*U3)*gaunt(l_1, l_2, l_3, i, j, -i-j)

    if prec is not None:
        ugnt = ugnt.n(prec)
    return ugnt


class Wigner3j(Function):

    def doit(self, **hints):
        if all(obj.is_number for obj in self.args):
            return wigner_3j(*self.args)
        else:
            return self

def dot_rot_grad_Ynm(j, p, l, m, theta, phi):
    r"""
    Returns dot product of rotational gradients of spherical harmonics.

    Explanation
    ===========

    This function returns the right hand side of the following expression:

    .. math ::
        \vec{R}Y{_j^{p}} \cdot \vec{R}Y{_l^{m}} = (-1)^{m+p}
        \sum\limits_{k=|l-j|}^{l+j}Y{_k^{m+p}}  * \alpha_{l,m,j,p,k} *
        \frac{1}{2} (k^2-j^2-l^2+k-j-l)


    Arguments
    =========

    j, p, l, m .... indices in spherical harmonics (expressions or integers)
    theta, phi .... angle arguments in spherical harmonics

    Example
    =======

    >>> from sympy import symbols
    >>> from sympy.physics.wigner import dot_rot_grad_Ynm
    >>> theta, phi = symbols("theta phi")
    >>> dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit()
    3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))

    """
    j = sympify(j)
    p = sympify(p)
    l = sympify(l)
    m = sympify(m)
    theta = sympify(theta)
    phi = sympify(phi)
    k = Dummy("k")

    def alpha(l,m,j,p,k):
        return sqrt((2*l+1)*(2*j+1)*(2*k+1)/(4*pi)) * \
                Wigner3j(j, l, k, S.Zero, S.Zero, S.Zero) * \
                Wigner3j(j, l, k, p, m, -m-p)

    return (S.NegativeOne)**(m+p) * Sum(Ynm(k, m+p, theta, phi) * alpha(l,m,j,p,k) / 2 \
        *(k**2-j**2-l**2+k-j-l), (k, abs(l-j), l+j))


def wigner_d_small(J, beta):
    """Return the small Wigner d matrix for angular momentum J.

    Explanation
    ===========

    J : An integer, half-integer, or SymPy symbol for the total angular
        momentum of the angular momentum space being rotated.
    beta : A real number representing the Euler angle of rotation about
        the so-called line of nodes. See [Edmonds74]_.

    Returns
    =======

    A matrix representing the corresponding Euler angle rotation( in the basis
    of eigenvectors of `J_z`).

    .. math ::
        \\mathcal{d}_{\\beta} = \\exp\\big( \\frac{i\\beta}{\\hbar} J_y\\big)

    The components are calculated using the general form [Edmonds74]_,
    equation 4.1.15.

    Examples
    ========

    >>> from sympy import Integer, symbols, pi, pprint
    >>> from sympy.physics.wigner import wigner_d_small
    >>> half = 1/Integer(2)
    >>> beta = symbols("beta", real=True)
    >>> pprint(wigner_d_small(half, beta), use_unicode=True)
    ⎡   ⎛β⎞      ⎛β⎞⎤
    ⎢cos⎜─⎟   sin⎜─⎟⎥
    ⎢   ⎝2⎠      ⎝2⎠⎥
    ⎢               ⎥
    ⎢    ⎛β⎞     ⎛β⎞⎥
    ⎢-sin⎜─⎟  cos⎜─⎟⎥
    ⎣    ⎝2⎠     ⎝2⎠⎦

    >>> pprint(wigner_d_small(2*half, beta), use_unicode=True)
    ⎡        2⎛β⎞              ⎛β⎞    ⎛β⎞           2⎛β⎞     ⎤
    ⎢     cos ⎜─⎟        √2⋅sin⎜─⎟⋅cos⎜─⎟        sin ⎜─⎟     ⎥
    ⎢         ⎝2⎠              ⎝2⎠    ⎝2⎠            ⎝2⎠     ⎥
    ⎢                                                        ⎥
    ⎢       ⎛β⎞    ⎛β⎞       2⎛β⎞      2⎛β⎞        ⎛β⎞    ⎛β⎞⎥
    ⎢-√2⋅sin⎜─⎟⋅cos⎜─⎟  - sin ⎜─⎟ + cos ⎜─⎟  √2⋅sin⎜─⎟⋅cos⎜─⎟⎥
    ⎢       ⎝2⎠    ⎝2⎠        ⎝2⎠       ⎝2⎠        ⎝2⎠    ⎝2⎠⎥
    ⎢                                                        ⎥
    ⎢        2⎛β⎞               ⎛β⎞    ⎛β⎞          2⎛β⎞     ⎥
    ⎢     sin ⎜─⎟        -√2⋅sin⎜─⎟⋅cos⎜─⎟       cos ⎜─⎟     ⎥
    ⎣         ⎝2⎠               ⎝2⎠    ⎝2⎠           ⎝2⎠     ⎦

    From table 4 in [Edmonds74]_

    >>> pprint(wigner_d_small(half, beta).subs({beta:pi/2}), use_unicode=True)
    ⎡ √2   √2⎤
    ⎢ ──   ──⎥
    ⎢ 2    2 ⎥
    ⎢        ⎥
    ⎢-√2   √2⎥
    ⎢────  ──⎥
    ⎣ 2    2 ⎦

    >>> pprint(wigner_d_small(2*half, beta).subs({beta:pi/2}),
    ... use_unicode=True)
    ⎡       √2      ⎤
    ⎢1/2    ──   1/2⎥
    ⎢       2       ⎥
    ⎢               ⎥
    ⎢-√2         √2 ⎥
    ⎢────   0    ── ⎥
    ⎢ 2          2  ⎥
    ⎢               ⎥
    ⎢      -√2      ⎥
    ⎢1/2   ────  1/2⎥
    ⎣       2       ⎦

    >>> pprint(wigner_d_small(3*half, beta).subs({beta:pi/2}),
    ... use_unicode=True)
    ⎡ √2    √6    √6   √2⎤
    ⎢ ──    ──    ──   ──⎥
    ⎢ 4     4     4    4 ⎥
    ⎢                    ⎥
    ⎢-√6   -√2    √2   √6⎥
    ⎢────  ────   ──   ──⎥
    ⎢ 4     4     4    4 ⎥
    ⎢                    ⎥
    ⎢ √6   -√2   -√2   √6⎥
    ⎢ ──   ────  ────  ──⎥
    ⎢ 4     4     4    4 ⎥
    ⎢                    ⎥
    ⎢-√2    √6   -√6   √2⎥
    ⎢────   ──   ────  ──⎥
    ⎣ 4     4     4    4 ⎦

    >>> pprint(wigner_d_small(4*half, beta).subs({beta:pi/2}),
    ... use_unicode=True)
    ⎡             √6            ⎤
    ⎢1/4   1/2    ──   1/2   1/4⎥
    ⎢             4             ⎥
    ⎢                           ⎥
    ⎢-1/2  -1/2   0    1/2   1/2⎥
    ⎢                           ⎥
    ⎢ √6                     √6 ⎥
    ⎢ ──    0    -1/2   0    ── ⎥
    ⎢ 4                      4  ⎥
    ⎢                           ⎥
    ⎢-1/2  1/2    0    -1/2  1/2⎥
    ⎢                           ⎥
    ⎢             √6            ⎥
    ⎢1/4   -1/2   ──   -1/2  1/4⎥
    ⎣             4             ⎦

    """
    M = [J-i for i in range(2*J+1)]
    d = zeros(2*J+1)
    for i, Mi in enumerate(M):
        for j, Mj in enumerate(M):

            # We get the maximum and minimum value of sigma.
            sigmamax = min([J-Mi, J-Mj])
            sigmamin = max([0, -Mi-Mj])

            dij = sqrt(factorial(J+Mi)*factorial(J-Mi) /
                       factorial(J+Mj)/factorial(J-Mj))
            terms = [(-1)**(J-Mi-s) *
                     binomial(J+Mj, J-Mi-s) *
                     binomial(J-Mj, s) *
                     cos(beta/2)**(2*s+Mi+Mj) *
                     sin(beta/2)**(2*J-2*s-Mj-Mi)
                     for s in range(sigmamin, sigmamax+1)]

            d[i, j] = dij*Add(*terms)

    return ImmutableMatrix(d)


def wigner_d(J, alpha, beta, gamma):
    """Return the Wigner D matrix for angular momentum J.

    Explanation
    ===========

    J :
        An integer, half-integer, or SymPy symbol for the total angular
        momentum of the angular momentum space being rotated.
    alpha, beta, gamma - Real numbers representing the Euler.
        Angles of rotation about the so-called vertical, line of nodes, and
        figure axes. See [Edmonds74]_.

    Returns
    =======

    A matrix representing the corresponding Euler angle rotation( in the basis
    of eigenvectors of `J_z`).

    .. math ::
        \\mathcal{D}_{\\alpha \\beta \\gamma} =
        \\exp\\big( \\frac{i\\alpha}{\\hbar} J_z\\big)
        \\exp\\big( \\frac{i\\beta}{\\hbar} J_y\\big)
        \\exp\\big( \\frac{i\\gamma}{\\hbar} J_z\\big)

    The components are calculated using the general form [Edmonds74]_,
    equation 4.1.12.

    Examples
    ========

    The simplest possible example:

    >>> from sympy.physics.wigner import wigner_d
    >>> from sympy import Integer, symbols, pprint
    >>> half = 1/Integer(2)
    >>> alpha, beta, gamma = symbols("alpha, beta, gamma", real=True)
    >>> pprint(wigner_d(half, alpha, beta, gamma), use_unicode=True)
    ⎡  ⅈ⋅α  ⅈ⋅γ             ⅈ⋅α  -ⅈ⋅γ         ⎤
    ⎢  ───  ───             ───  ─────        ⎥
    ⎢   2    2     ⎛β⎞       2     2      ⎛β⎞ ⎥
    ⎢ ℯ   ⋅ℯ   ⋅cos⎜─⎟     ℯ   ⋅ℯ     ⋅sin⎜─⎟ ⎥
    ⎢              ⎝2⎠                    ⎝2⎠ ⎥
    ⎢                                         ⎥
    ⎢  -ⅈ⋅α   ⅈ⋅γ          -ⅈ⋅α   -ⅈ⋅γ        ⎥
    ⎢  ─────  ───          ─────  ─────       ⎥
    ⎢    2     2     ⎛β⎞     2      2      ⎛β⎞⎥
    ⎢-ℯ     ⋅ℯ   ⋅sin⎜─⎟  ℯ     ⋅ℯ     ⋅cos⎜─⎟⎥
    ⎣                ⎝2⎠                   ⎝2⎠⎦

    """
    d = wigner_d_small(J, beta)
    M = [J-i for i in range(2*J+1)]
    D = [[exp(I*Mi*alpha)*d[i, j]*exp(I*Mj*gamma)
          for j, Mj in enumerate(M)] for i, Mi in enumerate(M)]
    return ImmutableMatrix(D)
