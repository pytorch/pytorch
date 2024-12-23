r"""A module for special angle formulas for trigonometric functions

TODO
====

This module should be developed in the future to contain direct square root
representation of

.. math
    F(\frac{n}{m} \pi)

for every

- $m \in \{ 3, 5, 17, 257, 65537 \}$
- $n \in \mathbb{N}$, $0 \le n < m$
- $F \in \{\sin, \cos, \tan, \csc, \sec, \cot\}$

Without multi-step rewrites
(e.g. $\tan \to \cos/\sin \to \cos/\sqrt \to \ sqrt$)
or using chebyshev identities
(e.g. $\cos \to \cos + \cos^2 + \cdots \to \sqrt{} + \sqrt{}^2 + \cdots $),
which are trivial to implement in sympy,
and had used to give overly complicated expressions.

The reference can be found below, if anyone may need help implementing them.

References
==========

.. [*] Gottlieb, Christian. (1999). The Simple and straightforward construction
   of the regular 257-gon. The Mathematical Intelligencer. 21. 31-37.
   10.1007/BF03024829.
.. [*] https://resources.wolframcloud.com/FunctionRepository/resources/Cos2PiOverFermatPrime
"""
from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.intfunc import igcdex
from sympy.core.numbers import Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit


def migcdex(*x: int) -> tuple[tuple[int, ...], int]:
    r"""Compute extended gcd for multiple integers.

    Explanation
    ===========

    Given the integers $x_1, \cdots, x_n$ and
    an extended gcd for multiple arguments are defined as a solution
    $(y_1, \cdots, y_n), g$ for the diophantine equation
    $x_1 y_1 + \cdots + x_n y_n = g$ such that
    $g = \gcd(x_1, \cdots, x_n)$.

    Examples
    ========

    >>> from sympy.functions.elementary._trigonometric_special import migcdex
    >>> migcdex()
    ((), 0)
    >>> migcdex(4)
    ((1,), 4)
    >>> migcdex(4, 6)
    ((-1, 1), 2)
    >>> migcdex(6, 10, 15)
    ((1, 1, -1), 1)
    """
    if not x:
        return (), 0

    if len(x) == 1:
        return (1,), x[0]

    if len(x) == 2:
        u, v, h = igcdex(x[0], x[1])
        return (u, v), h

    y, g = migcdex(*x[1:])
    u, v, h = igcdex(x[0], g)
    return (u, *(v * i for i in y)), h


def ipartfrac(*denoms: int) -> tuple[int, ...]:
    r"""Compute the partial fraction decomposition.

    Explanation
    ===========

    Given a rational number $\frac{1}{q_1 \cdots q_n}$ where all
    $q_1, \cdots, q_n$ are pairwise coprime,

    A partial fraction decomposition is defined as

    .. math::
        \frac{1}{q_1 \cdots q_n} = \frac{p_1}{q_1} + \cdots + \frac{p_n}{q_n}

    And it can be derived from solving the following diophantine equation for
    the $p_1, \cdots, p_n$

    .. math::
        1 = p_1 \prod_{i \ne 1}q_i + \cdots + p_n \prod_{i \ne n}q_i

    Where $q_1, \cdots, q_n$ being pairwise coprime implies
    $\gcd(\prod_{i \ne 1}q_i, \cdots, \prod_{i \ne n}q_i) = 1$,
    which guarantees the existence of the solution.

    It is sufficient to compute partial fraction decomposition only
    for numerator $1$ because partial fraction decomposition for any
    $\frac{n}{q_1 \cdots q_n}$ can be easily computed by multiplying
    the result by $n$ afterwards.

    Parameters
    ==========

    denoms : int
        The pairwise coprime integer denominators $q_i$ which defines the
        rational number $\frac{1}{q_1 \cdots q_n}$

    Returns
    =======

    tuple[int, ...]
        The list of numerators which semantically corresponds to $p_i$ of the
        partial fraction decomposition
        $\frac{1}{q_1 \cdots q_n} = \frac{p_1}{q_1} + \cdots + \frac{p_n}{q_n}$

    Examples
    ========

    >>> from sympy import Rational, Mul
    >>> from sympy.functions.elementary._trigonometric_special import ipartfrac

    >>> denoms = 2, 3, 5
    >>> numers = ipartfrac(2, 3, 5)
    >>> numers
    (1, 7, -14)

    >>> Rational(1, Mul(*denoms))
    1/30
    >>> out = 0
    >>> for n, d in zip(numers, denoms):
    ...    out += Rational(n, d)
    >>> out
    1/30
    """
    if not denoms:
        return ()

    def mul(x: int, y: int) -> int:
        return x * y

    denom = reduce(mul, denoms)
    a = [denom // x for x in denoms]
    h, _ = migcdex(*a)
    return h


def fermat_coords(n: int) -> list[int] | None:
    """If n can be factored in terms of Fermat primes with
    multiplicity of each being 1, return those primes, else
    None
    """
    primes = []
    for p in [3, 5, 17, 257, 65537]:
        quotient, remainder = divmod(n, p)
        if remainder == 0:
            n = quotient
            primes.append(p)
            if n == 1:
                return primes
    return None


@cacheit
def cos_3() -> Expr:
    r"""Computes $\cos \frac{\pi}{3}$ in square roots"""
    return S.Half


@cacheit
def cos_5() -> Expr:
    r"""Computes $\cos \frac{\pi}{5}$ in square roots"""
    return (sqrt(5) + 1) / 4


@cacheit
def cos_17() -> Expr:
    r"""Computes $\cos \frac{\pi}{17}$ in square roots"""
    return sqrt(
        (15 + sqrt(17)) / 32 + sqrt(2) * (sqrt(17 - sqrt(17)) +
        sqrt(sqrt(2) * (-8 * sqrt(17 + sqrt(17)) - (1 - sqrt(17))
        * sqrt(17 - sqrt(17))) + 6 * sqrt(17) + 34)) / 32)


@cacheit
def cos_257() -> Expr:
    r"""Computes $\cos \frac{\pi}{257}$ in square roots

    References
    ==========

    .. [*] https://math.stackexchange.com/questions/516142/how-does-cos2-pi-257-look-like-in-real-radicals
    .. [*] https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html
    """
    def f1(a: Expr, b: Expr) -> tuple[Expr, Expr]:
        return (a + sqrt(a**2 + b)) / 2, (a - sqrt(a**2 + b)) / 2

    def f2(a: Expr, b: Expr) -> Expr:
        return (a - sqrt(a**2 + b))/2

    t1, t2 = f1(S.NegativeOne, Integer(256))
    z1, z3 = f1(t1, Integer(64))
    z2, z4 = f1(t2, Integer(64))
    y1, y5 = f1(z1, 4*(5 + t1 + 2*z1))
    y6, y2 = f1(z2, 4*(5 + t2 + 2*z2))
    y3, y7 = f1(z3, 4*(5 + t1 + 2*z3))
    y8, y4 = f1(z4, 4*(5 + t2 + 2*z4))
    x1, x9 = f1(y1, -4*(t1 + y1 + y3 + 2*y6))
    x2, x10 = f1(y2, -4*(t2 + y2 + y4 + 2*y7))
    x3, x11 = f1(y3, -4*(t1 + y3 + y5 + 2*y8))
    x4, x12 = f1(y4, -4*(t2 + y4 + y6 + 2*y1))
    x5, x13 = f1(y5, -4*(t1 + y5 + y7 + 2*y2))
    x6, x14 = f1(y6, -4*(t2 + y6 + y8 + 2*y3))
    x15, x7 = f1(y7, -4*(t1 + y7 + y1 + 2*y4))
    x8, x16 = f1(y8, -4*(t2 + y8 + y2 + 2*y5))
    v1 = f2(x1, -4*(x1 + x2 + x3 + x6))
    v2 = f2(x2, -4*(x2 + x3 + x4 + x7))
    v3 = f2(x8, -4*(x8 + x9 + x10 + x13))
    v4 = f2(x9, -4*(x9 + x10 + x11 + x14))
    v5 = f2(x10, -4*(x10 + x11 + x12 + x15))
    v6 = f2(x16, -4*(x16 + x1 + x2 + x5))
    u1 = -f2(-v1, -4*(v2 + v3))
    u2 = -f2(-v4, -4*(v5 + v6))
    w1 = -2*f2(-u1, -4*u2)
    return sqrt(sqrt(2)*sqrt(w1 + 4)/8 + S.Half)


def cos_table() -> dict[int, Callable[[], Expr]]:
    r"""Lazily evaluated table for $\cos \frac{\pi}{n}$ in square roots for
    $n \in \{3, 5, 17, 257, 65537\}$.

    Notes
    =====

    65537 is the only other known Fermat prime and it is nearly impossible to
    build in the current SymPy due to performance issues.

    References
    ==========

    https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html
    """
    return {
        3: cos_3,
        5: cos_5,
        17: cos_17,
        257: cos_257
    }
