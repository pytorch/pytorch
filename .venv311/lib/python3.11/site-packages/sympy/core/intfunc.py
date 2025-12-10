"""
The routines here were removed from numbers.py, power.py,
digits.py and factor_.py so they could be imported into core
without raising circular import errors.

Although the name 'intfunc' was chosen to represent functions that
work with integers, it can also be thought of as containing
internal/core functions that are needed by the classes of the core.
"""

import math
import sys
from functools import lru_cache

from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import (gcd as number_gcd, lcm as number_lcm, sqrt,
                                 iroot, bit_scan1, gcdext)
from sympy.utilities.misc import as_int, filldedent


def num_digits(n, base=10):
    """Return the number of digits needed to express n in give base.

    Examples
    ========

    >>> from sympy.core.intfunc import num_digits
    >>> num_digits(10)
    2
    >>> num_digits(10, 2)  # 1010 -> 4 digits
    4
    >>> num_digits(-100, 16)  # -64 -> 2 digits
    2


    Parameters
    ==========

    n: integer
        The number whose digits are counted.

    b: integer
        The base in which digits are computed.

    See Also
    ========
    sympy.ntheory.digits.digits, sympy.ntheory.digits.count_digits
    """
    if base < 0:
        raise ValueError('base must be int greater than 1')
    if not n:
        return 1
    e, t = integer_log(abs(n), base)
    return 1 + e


def integer_log(n, b):
    r"""
    Returns ``(e, bool)`` where e is the largest nonnegative integer
    such that :math:`|n| \geq |b^e|` and ``bool`` is True if $n = b^e$.

    Examples
    ========

    >>> from sympy import integer_log
    >>> integer_log(125, 5)
    (3, True)
    >>> integer_log(17, 9)
    (1, False)

    If the base is positive and the number negative the
    return value will always be the same except for 2:

    >>> integer_log(-4, 2)
    (2, False)
    >>> integer_log(-16, 4)
    (0, False)

    When the base is negative, the returned value
    will only be True if the parity of the exponent is
    correct for the sign of the base:

    >>> integer_log(4, -2)
    (2, True)
    >>> integer_log(8, -2)
    (3, False)
    >>> integer_log(-8, -2)
    (3, True)
    >>> integer_log(-4, -2)
    (2, False)

    See Also
    ========
    integer_nthroot
    sympy.ntheory.primetest.is_square
    sympy.ntheory.factor_.multiplicity
    sympy.ntheory.factor_.perfect_power
    """
    n = as_int(n)
    b = as_int(b)

    if b < 0:
        e, t = integer_log(abs(n), -b)
        # (-2)**3 == -8
        # (-2)**2 = 4
        t = t and e % 2 == (n < 0)
        return e, t
    if b <= 1:
        raise ValueError('base must be 2 or more')
    if n < 0:
        if b != 2:
            return 0, False
        e, t = integer_log(-n, b)
        return e, False
    if n == 0:
        raise ValueError('n cannot be 0')

    if n < b:
        return 0, n == 1
    if b == 2:
        e = n.bit_length() - 1
        return e, trailing(n) == e
    t = trailing(b)
    if 2**t == b:
        e = int(n.bit_length() - 1)//t
        n_ = 1 << (t*e)
        return e, n_ == n

    d = math.floor(math.log10(n) / math.log10(b))
    n_ = b ** d
    while n_ <= n:  # this will iterate 0, 1 or 2 times
        d += 1
        n_ *= b
    return d - (n_ > n), (n_ == n or n_//b == n)


def trailing(n):
    """Count the number of trailing zero digits in the binary
    representation of n, i.e. determine the largest power of 2
    that divides n.

    Examples
    ========

    >>> from sympy import trailing
    >>> trailing(128)
    7
    >>> trailing(63)
    0

    See Also
    ========
    sympy.ntheory.factor_.multiplicity

    """
    if not n:
        return 0
    return bit_scan1(int(n))


@lru_cache(1024)
def igcd(*args):
    """Computes nonnegative integer greatest common divisor.

    Explanation
    ===========

    The algorithm is based on the well known Euclid's algorithm [1]_. To
    improve speed, ``igcd()`` has its own caching mechanism.
    If you do not need the cache mechanism, using ``sympy.external.gmpy.gcd``.

    Examples
    ========

    >>> from sympy import igcd
    >>> igcd(2, 4)
    2
    >>> igcd(5, 10, 15)
    5

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euclidean_algorithm

    """
    if len(args) < 2:
        raise TypeError("igcd() takes at least 2 arguments (%s given)" % len(args))
    return int(number_gcd(*map(as_int, args)))


igcd2 = math.gcd


def igcd_lehmer(a, b):
    r"""Computes greatest common divisor of two integers.

    Explanation
    ===========

    Euclid's algorithm for the computation of the greatest
    common divisor ``gcd(a, b)``  of two (positive) integers
    $a$ and $b$ is based on the division identity
    $$ a = q \times b + r$$,
    where the quotient  $q$  and the remainder  $r$  are integers
    and  $0 \le r < b$. Then each common divisor of  $a$  and  $b$
    divides  $r$, and it follows that  ``gcd(a, b) == gcd(b, r)``.
    The algorithm works by constructing the sequence
    r0, r1, r2, ..., where  r0 = a, r1 = b,  and each  rn
    is the remainder from the division of the two preceding
    elements.

    In Python, ``q = a // b``  and  ``r = a % b``  are obtained by the
    floor division and the remainder operations, respectively.
    These are the most expensive arithmetic operations, especially
    for large  a  and  b.

    Lehmer's algorithm [1]_ is based on the observation that the quotients
    ``qn = r(n-1) // rn``  are in general small integers even
    when  a  and  b  are very large. Hence the quotients can be
    usually determined from a relatively small number of most
    significant bits.

    The efficiency of the algorithm is further enhanced by not
    computing each long remainder in Euclid's sequence. The remainders
    are linear combinations of  a  and  b  with integer coefficients
    derived from the quotients. The coefficients can be computed
    as far as the quotients can be determined from the chosen
    most significant parts of  a  and  b. Only then a new pair of
    consecutive remainders is computed and the algorithm starts
    anew with this pair.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lehmer%27s_GCD_algorithm

    """
    a, b = abs(as_int(a)), abs(as_int(b))
    if a < b:
        a, b = b, a

    # The algorithm works by using one or two digit division
    # whenever possible. The outer loop will replace the
    # pair (a, b) with a pair of shorter consecutive elements
    # of the Euclidean gcd sequence until a and b
    # fit into two Python (long) int digits.
    nbits = 2 * sys.int_info.bits_per_digit

    while a.bit_length() > nbits and b != 0:
        # Quotients are mostly small integers that can
        # be determined from most significant bits.
        n = a.bit_length() - nbits
        x, y = int(a >> n), int(b >> n)  # most significant bits

        # Elements of the Euclidean gcd sequence are linear
        # combinations of a and b with integer coefficients.
        # Compute the coefficients of consecutive pairs
        #     a' = A*a + B*b, b' = C*a + D*b
        # using small integer arithmetic as far as possible.
        A, B, C, D = 1, 0, 0, 1  # initial values

        while True:
            # The coefficients alternate in sign while looping.
            # The inner loop combines two steps to keep track
            # of the signs.

            # At this point we have
            #   A > 0, B <= 0, C <= 0, D > 0,
            #   x' = x + B <= x < x" = x + A,
            #   y' = y + C <= y < y" = y + D,
            # and
            #   x'*N <= a' < x"*N, y'*N <= b' < y"*N,
            # where N = 2**n.

            # Now, if y' > 0, and x"//y' and x'//y" agree,
            # then their common value is equal to  q = a'//b'.
            # In addition,
            #   x'%y" = x' - q*y" < x" - q*y' = x"%y',
            # and
            #   (x'%y")*N < a'%b' < (x"%y')*N.

            # On the other hand, we also have  x//y == q,
            # and therefore
            #   x'%y" = x + B - q*(y + D) = x%y + B',
            #   x"%y' = x + A - q*(y + C) = x%y + A',
            # where
            #    B' = B - q*D < 0, A' = A - q*C > 0.

            if y + C <= 0:
                break
            q = (x + A) // (y + C)

            # Now  x'//y" <= q, and equality holds if
            #   x' - q*y" = (x - q*y) + (B - q*D) >= 0.
            # This is a minor optimization to avoid division.
            x_qy, B_qD = x - q * y, B - q * D
            if x_qy + B_qD < 0:
                break

            # Next step in the Euclidean sequence.
            x, y = y, x_qy
            A, B, C, D = C, D, A - q * C, B_qD

            # At this point the signs of the coefficients
            # change and their roles are interchanged.
            #   A <= 0, B > 0, C > 0, D < 0,
            #   x' = x + A <= x < x" = x + B,
            #   y' = y + D < y < y" = y + C.

            if y + D <= 0:
                break
            q = (x + B) // (y + D)
            x_qy, A_qC = x - q * y, A - q * C
            if x_qy + A_qC < 0:
                break

            x, y = y, x_qy
            A, B, C, D = C, D, A_qC, B - q * D
            # Now the conditions on top of the loop
            # are again satisfied.
            #   A > 0, B < 0, C < 0, D > 0.

        if B == 0:
            # This can only happen when y == 0 in the beginning
            # and the inner loop does nothing.
            # Long division is forced.
            a, b = b, a % b
            continue

        # Compute new long arguments using the coefficients.
        a, b = A * a + B * b, C * a + D * b

    # Small divisors. Finish with the standard algorithm.
    while b:
        a, b = b, a % b

    return a


def ilcm(*args):
    """Computes integer least common multiple.

    Examples
    ========

    >>> from sympy import ilcm
    >>> ilcm(5, 10)
    10
    >>> ilcm(7, 3)
    21
    >>> ilcm(5, 10, 15)
    30

    """
    if len(args) < 2:
        raise TypeError("ilcm() takes at least 2 arguments (%s given)" % len(args))
    return int(number_lcm(*map(as_int, args)))


def igcdex(a, b):
    """Returns x, y, g such that g = x*a + y*b = gcd(a, b).

    Examples
    ========

    >>> from sympy.core.intfunc import igcdex
    >>> igcdex(2, 3)
    (-1, 1, 1)
    >>> igcdex(10, 12)
    (-1, 1, 2)

    >>> x, y, g = igcdex(100, 2004)
    >>> x, y, g
    (-20, 1, 4)
    >>> x*100 + y*2004
    4

    """
    g, x, y = gcdext(int(a), int(b))
    return x, y, g


def mod_inverse(a, m):
    r"""
    Return the number $c$ such that, $a \times c = 1 \pmod{m}$
    where $c$ has the same sign as $m$. If no such value exists,
    a ValueError is raised.

    Examples
    ========

    >>> from sympy import mod_inverse, S

    Suppose we wish to find multiplicative inverse $x$ of
    3 modulo 11. This is the same as finding $x$ such
    that $3x = 1 \pmod{11}$. One value of x that satisfies
    this congruence is 4. Because $3 \times 4 = 12$ and $12 = 1 \pmod{11}$.
    This is the value returned by ``mod_inverse``:

    >>> mod_inverse(3, 11)
    4
    >>> mod_inverse(-3, 11)
    7

    When there is a common factor between the numerators of
    `a` and `m` the inverse does not exist:

    >>> mod_inverse(2, 4)
    Traceback (most recent call last):
    ...
    ValueError: inverse of 2 mod 4 does not exist

    >>> mod_inverse(S(2)/7, S(5)/2)
    7/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
    .. [2] https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    """
    c = None
    try:
        a, m = as_int(a), as_int(m)
        if m != 1 and m != -1:
            x, _, g = igcdex(a, m)
            if g == 1:
                c = x % m
    except ValueError:
        a, m = sympify(a), sympify(m)
        if not (a.is_number and m.is_number):
            raise TypeError(
                filldedent(
                    """
                Expected numbers for arguments; symbolic `mod_inverse`
                is not implemented
                but symbolic expressions can be handled with the
                similar function,
                sympy.polys.polytools.invert"""
                )
            )
        big = m > 1
        if big not in (S.true, S.false):
            raise ValueError("m > 1 did not evaluate; try to simplify %s" % m)
        elif big:
            c = 1 / a
    if c is None:
        raise ValueError("inverse of %s (mod %s) does not exist" % (a, m))
    return c


def isqrt(n):
    r""" Return the largest integer less than or equal to `\sqrt{n}`.

    Parameters
    ==========

    n : non-negative integer

    Returns
    =======

    int : `\left\lfloor\sqrt{n}\right\rfloor`

    Raises
    ======

    ValueError
        If n is negative.
    TypeError
        If n is of a type that cannot be compared to ``int``.
        Therefore, a TypeError is raised for ``str``, but not for ``float``.

    Examples
    ========

    >>> from sympy.core.intfunc import isqrt
    >>> isqrt(0)
    0
    >>> isqrt(9)
    3
    >>> isqrt(10)
    3
    >>> isqrt("30")
    Traceback (most recent call last):
        ...
    TypeError: '<' not supported between instances of 'str' and 'int'
    >>> from sympy.core.numbers import Rational
    >>> isqrt(Rational(-1, 2))
    Traceback (most recent call last):
        ...
    ValueError: n must be nonnegative

    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    return int(sqrt(int(n)))


def integer_nthroot(y, n):
    """
    Return a tuple containing x = floor(y**(1/n))
    and a boolean indicating whether the result is exact (that is,
    whether x**n == y).

    Examples
    ========

    >>> from sympy import integer_nthroot
    >>> integer_nthroot(16, 2)
    (4, True)
    >>> integer_nthroot(26, 2)
    (5, False)

    To simply determine if a number is a perfect square, the is_square
    function should be used:

    >>> from sympy.ntheory.primetest import is_square
    >>> is_square(26)
    False

    See Also
    ========
    sympy.ntheory.primetest.is_square
    integer_log
    """
    x, b = iroot(as_int(y), as_int(n))
    return int(x), b
