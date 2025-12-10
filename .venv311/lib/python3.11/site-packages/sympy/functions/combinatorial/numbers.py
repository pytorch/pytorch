"""
This module implements some special functions that commonly appear in
combinatorial contexts (e.g. in power series); in particular,
sequences of rational numbers such as Bernoulli and Fibonacci numbers.

Factorials, binomial coefficients and related functions are located in
the separate 'factorials' module.
"""
from __future__ import annotations
from math import prod
from collections import defaultdict
from typing import Callable

from sympy.core import S, Symbol, Add, Dummy
from sympy.core.cache import cacheit
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, DefinedFunction, expand_mul
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import E, I, pi, oo, Rational, Integer
from sympy.core.relational import Eq, is_le, is_gt, is_lt
from sympy.external.gmpy import SYMPY_INTS, remove, lcm, legendre, jacobi, kronecker
from sympy.functions.combinatorial.factorials import (binomial,
    factorial, subfactorial)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.ntheory.factor_ import (factorint, _divisor_sigma, is_carmichael,
                                   find_carmichael_numbers_in_range, find_first_n_carmichaels)
from sympy.ntheory.generate import _primepi
from sympy.ntheory.partitions_ import _partition, _partition_rec
from sympy.ntheory.primetest import isprime, is_square
from sympy.polys.appellseqs import bernoulli_poly, euler_poly, genocchi_poly
from sympy.polys.polytools import cancel
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import multiset, multiset_derangements, iterable
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int

from mpmath import mp, workprec
from mpmath.libmp import ifib as _ifib


def _product(a, b):
    return prod(range(a, b + 1))


# Dummy symbol used for computing polynomial sequences
_sym = Symbol('x')


#----------------------------------------------------------------------------#
#                                                                            #
#                           Carmichael numbers                               #
#                                                                            #
#----------------------------------------------------------------------------#

class carmichael(DefinedFunction):
    r"""
    Carmichael Numbers:

    Certain cryptographic algorithms make use of big prime numbers.
    However, checking whether a big number is prime is not so easy.
    Randomized prime number checking tests exist that offer a high degree of
    confidence of accurate determination at low cost, such as the Fermat test.

    Let 'a' be a random number between $2$ and $n - 1$, where $n$ is the
    number whose primality we are testing. Then, $n$ is probably prime if it
    satisfies the modular arithmetic congruence relation:

    .. math :: a^{n-1} = 1 \pmod{n}

    (where mod refers to the modulo operation)

    If a number passes the Fermat test several times, then it is prime with a
    high probability.

    Unfortunately, certain composite numbers (non-primes) still pass the Fermat
    test with every number smaller than themselves.
    These numbers are called Carmichael numbers.

    A Carmichael number will pass a Fermat primality test to every base $b$
    relatively prime to the number, even though it is not actually prime.
    This makes tests based on Fermat's Little Theorem less effective than
    strong probable prime tests such as the Baillie-PSW primality test and
    the Miller-Rabin primality test.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import find_first_n_carmichaels, find_carmichael_numbers_in_range
    >>> find_first_n_carmichaels(5)
    [561, 1105, 1729, 2465, 2821]
    >>> find_carmichael_numbers_in_range(0, 562)
    [561]
    >>> find_carmichael_numbers_in_range(0,1000)
    [561]
    >>> find_carmichael_numbers_in_range(0,2000)
    [561, 1105, 1729]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_number
    .. [2] https://en.wikipedia.org/wiki/Fermat_primality_test
    .. [3] https://www.jstor.org/stable/23248683?seq=1#metadata_info_tab_contents
    """

    @staticmethod
    def is_perfect_square(n):
        sympy_deprecation_warning(
        """
is_perfect_square is just a wrapper around sympy.ntheory.primetest.is_square
so use that directly instead.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return is_square(n)

    @staticmethod
    def divides(p, n):
        sympy_deprecation_warning(
        """
        divides can be replaced by directly testing n % p == 0.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return n % p == 0

    @staticmethod
    def is_prime(n):
        sympy_deprecation_warning(
        """
is_prime is just a wrapper around sympy.ntheory.primetest.isprime so use that
directly instead.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return isprime(n)

    @staticmethod
    def is_carmichael(n):
        sympy_deprecation_warning(
        """
is_carmichael is just a wrapper around sympy.ntheory.factor_.is_carmichael so use that
directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return is_carmichael(n)

    @staticmethod
    def find_carmichael_numbers_in_range(x, y):
        sympy_deprecation_warning(
        """
find_carmichael_numbers_in_range is just a wrapper around sympy.ntheory.factor_.find_carmichael_numbers_in_range so use that
directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return find_carmichael_numbers_in_range(x, y)

    @staticmethod
    def find_first_n_carmichaels(n):
        sympy_deprecation_warning(
        """
find_first_n_carmichaels is just a wrapper around sympy.ntheory.factor_.find_first_n_carmichaels so use that
directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return find_first_n_carmichaels(n)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Fibonacci numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class fibonacci(DefinedFunction):
    r"""
    Fibonacci numbers / Fibonacci polynomials

    The Fibonacci numbers are the integer sequence defined by the
    initial terms `F_0 = 0`, `F_1 = 1` and the two-term recurrence
    relation `F_n = F_{n-1} + F_{n-2}`.  This definition
    extended to arbitrary real and complex arguments using
    the formula

    .. math :: F_z = \frac{\phi^z - \cos(\pi z) \phi^{-z}}{\sqrt 5}

    The Fibonacci polynomials are defined by `F_1(x) = 1`,
    `F_2(x) = x`, and `F_n(x) = x*F_{n-1}(x) + F_{n-2}(x)` for `n > 2`.
    For all positive integers `n`, `F_n(1) = F_n`.

    * ``fibonacci(n)`` gives the `n^{th}` Fibonacci number, `F_n`
    * ``fibonacci(n, x)`` gives the `n^{th}` Fibonacci polynomial in `x`, `F_n(x)`

    Examples
    ========

    >>> from sympy import fibonacci, Symbol

    >>> [fibonacci(x) for x in range(11)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    >>> fibonacci(5, Symbol('t'))
    t**4 + 3*t**2 + 1

    See Also
    ========

    bell, bernoulli, catalan, euler, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fibonacci_number
    .. [2] https://mathworld.wolfram.com/FibonacciNumber.html

    """

    @staticmethod
    def _fib(n):
        return _ifib(n)

    @staticmethod
    @recurrence_memo([None, S.One, _sym])
    def _fibpoly(n, prev):
        return (prev[-2] + _sym*prev[-1]).expand()

    @classmethod
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            if sym is None:
                n = int(n)
                if n < 0:
                    return S.NegativeOne**(n + 1) * fibonacci(-n)
                else:
                    return Integer(cls._fib(n))
            else:
                if n < 1:
                    raise ValueError("Fibonacci polynomials are defined "
                       "only for positive integer indices.")
                return cls._fibpoly(n).subs(_sym, sym)

    def _eval_rewrite_as_tractable(self, n, **kwargs):
        from sympy.functions import sqrt, cos
        return (S.GoldenRatio**n - cos(S.Pi*n)/S.GoldenRatio**n)/sqrt(5)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        return 2**(-n)*sqrt(5)*((1 + sqrt(5))**n - (-sqrt(5) + 1)**n) / 5

    def _eval_rewrite_as_GoldenRatio(self,n, **kwargs):
        return (S.GoldenRatio**n - 1/(-S.GoldenRatio)**n)/(2*S.GoldenRatio-1)


#----------------------------------------------------------------------------#
#                                                                            #
#                               Lucas numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class lucas(DefinedFunction):
    """
    Lucas numbers

    Lucas numbers satisfy a recurrence relation similar to that of
    the Fibonacci sequence, in which each term is the sum of the
    preceding two. They are generated by choosing the initial
    values `L_0 = 2` and `L_1 = 1`.

    * ``lucas(n)`` gives the `n^{th}` Lucas number

    Examples
    ========

    >>> from sympy import lucas

    >>> [lucas(x) for x in range(11)]
    [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas_number
    .. [2] https://mathworld.wolfram.com/LucasNumber.html

    """

    @classmethod
    def eval(cls, n):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            return fibonacci(n + 1) + fibonacci(n - 1)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        return 2**(-n)*((1 + sqrt(5))**n + (-sqrt(5) + 1)**n)


#----------------------------------------------------------------------------#
#                                                                            #
#                             Tribonacci numbers                             #
#                                                                            #
#----------------------------------------------------------------------------#


class tribonacci(DefinedFunction):
    r"""
    Tribonacci numbers / Tribonacci polynomials

    The Tribonacci numbers are the integer sequence defined by the
    initial terms `T_0 = 0`, `T_1 = 1`, `T_2 = 1` and the three-term
    recurrence relation `T_n = T_{n-1} + T_{n-2} + T_{n-3}`.

    The Tribonacci polynomials are defined by `T_0(x) = 0`, `T_1(x) = 1`,
    `T_2(x) = x^2`, and `T_n(x) = x^2 T_{n-1}(x) + x T_{n-2}(x) + T_{n-3}(x)`
    for `n > 2`.  For all positive integers `n`, `T_n(1) = T_n`.

    * ``tribonacci(n)`` gives the `n^{th}` Tribonacci number, `T_n`
    * ``tribonacci(n, x)`` gives the `n^{th}` Tribonacci polynomial in `x`, `T_n(x)`

    Examples
    ========

    >>> from sympy import tribonacci, Symbol

    >>> [tribonacci(x) for x in range(11)]
    [0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149]
    >>> tribonacci(5, Symbol('t'))
    t**8 + 3*t**5 + 3*t**2

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalizations_of_Fibonacci_numbers#Tribonacci_numbers
    .. [2] https://mathworld.wolfram.com/TribonacciNumber.html
    .. [3] https://oeis.org/A000073

    """

    @staticmethod
    @recurrence_memo([S.Zero, S.One, S.One])
    def _trib(n, prev):
        return (prev[-3] + prev[-2] + prev[-1])

    @staticmethod
    @recurrence_memo([S.Zero, S.One, _sym**2])
    def _tribpoly(n, prev):
        return (prev[-3] + _sym*prev[-2] + _sym**2*prev[-1]).expand()

    @classmethod
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            n = int(n)
            if n < 0:
                raise ValueError("Tribonacci polynomials are defined "
                       "only for non-negative integer indices.")
            if sym is None:
                return Integer(cls._trib(n))
            else:
                return cls._tribpoly(n).subs(_sym, sym)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        w = (-1 + S.ImaginaryUnit * sqrt(3)) / 2
        a = (1 + cbrt(19 + 3*sqrt(33)) + cbrt(19 - 3*sqrt(33))) / 3
        b = (1 + w*cbrt(19 + 3*sqrt(33)) + w**2*cbrt(19 - 3*sqrt(33))) / 3
        c = (1 + w**2*cbrt(19 + 3*sqrt(33)) + w*cbrt(19 - 3*sqrt(33))) / 3
        Tn = (a**(n + 1)/((a - b)*(a - c))
            + b**(n + 1)/((b - a)*(b - c))
            + c**(n + 1)/((c - a)*(c - b)))
        return Tn

    def _eval_rewrite_as_TribonacciConstant(self, n, **kwargs):
        from sympy.functions.elementary.integers import floor
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        b = cbrt(586 + 102*sqrt(33))
        Tn = 3 * b * S.TribonacciConstant**n / (b**2 - 2*b + 4)
        return floor(Tn + S.Half)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Bernoulli numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class bernoulli(DefinedFunction):
    r"""
    Bernoulli numbers / Bernoulli polynomials / Bernoulli function

    The Bernoulli numbers are a sequence of rational numbers
    defined by `B_0 = 1` and the recursive relation (`n > 0`):

    .. math :: n+1 = \sum_{k=0}^n \binom{n+1}{k} B_k

    They are also commonly defined by their exponential generating
    function, which is `\frac{x}{1 - e^{-x}}`. For odd indices > 1,
    the Bernoulli numbers are zero.

    The Bernoulli polynomials satisfy the analogous formula:

    .. math :: B_n(x) = \sum_{k=0}^n (-1)^k \binom{n}{k} B_k x^{n-k}

    Bernoulli numbers and Bernoulli polynomials are related as
    `B_n(1) = B_n`.

    The generalized Bernoulli function `\operatorname{B}(s, a)`
    is defined for any complex `s` and `a`, except where `a` is a
    nonpositive integer and `s` is not a nonnegative integer. It is
    an entire function of `s` for fixed `a`, related to the Hurwitz
    zeta function by

    .. math:: \operatorname{B}(s, a) = \begin{cases}
              -s \zeta(1-s, a) & s \ne 0 \\ 1 & s = 0 \end{cases}

    When `s` is a nonnegative integer this function reduces to the
    Bernoulli polynomials: `\operatorname{B}(n, x) = B_n(x)`. When
    `a` is omitted it is assumed to be 1, yielding the (ordinary)
    Bernoulli function which interpolates the Bernoulli numbers and is
    related to the Riemann zeta function.

    We compute Bernoulli numbers using Ramanujan's formula:

    .. math :: B_n = \frac{A(n) - S(n)}{\binom{n+3}{n}}

    where:

    .. math :: A(n) = \begin{cases} \frac{n+3}{3} &
        n \equiv 0\ \text{or}\ 2 \pmod{6} \\
        -\frac{n+3}{6} & n \equiv 4 \pmod{6} \end{cases}

    and:

    .. math :: S(n) = \sum_{k=1}^{[n/6]} \binom{n+3}{n-6k} B_{n-6k}

    This formula is similar to the sum given in the definition, but
    cuts `\frac{2}{3}` of the terms. For Bernoulli polynomials, we use
    Appell sequences.

    For `n` a nonnegative integer and `s`, `a`, `x` arbitrary complex numbers,

    * ``bernoulli(n)`` gives the nth Bernoulli number, `B_n`
    * ``bernoulli(s)`` gives the Bernoulli function `\operatorname{B}(s)`
    * ``bernoulli(n, x)`` gives the nth Bernoulli polynomial in `x`, `B_n(x)`
    * ``bernoulli(s, a)`` gives the generalized Bernoulli function
      `\operatorname{B}(s, a)`

    .. versionchanged:: 1.12
        ``bernoulli(1)`` gives `+\frac{1}{2}` instead of `-\frac{1}{2}`.
        This choice of value confers several theoretical advantages [5]_,
        including the extension to complex parameters described above
        which this function now implements. The previous behavior, defined
        only for nonnegative integers `n`, can be obtained with
        ``(-1)**n*bernoulli(n)``.

    Examples
    ========

    >>> from sympy import bernoulli
    >>> from sympy.abc import x
    >>> [bernoulli(n) for n in range(11)]
    [1, 1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> bernoulli(1000001)
    0
    >>> bernoulli(3, x)
    x**3 - 3*x**2/2 + x/2

    See Also
    ========

    andre, bell, catalan, euler, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.polys.appellseqs.bernoulli_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_number
    .. [2] https://en.wikipedia.org/wiki/Bernoulli_polynomial
    .. [3] https://mathworld.wolfram.com/BernoulliNumber.html
    .. [4] https://mathworld.wolfram.com/BernoulliPolynomial.html
    .. [5] Peter Luschny, "The Bernoulli Manifesto",
           https://luschny.de/math/zeta/The-Bernoulli-Manifesto.html
    .. [6] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    args: tuple[Integer]

    # Calculates B_n for positive even n
    @staticmethod
    def _calc_bernoulli(n):
        s = 0
        a = int(binomial(n + 3, n - 6))
        for j in range(1, n//6 + 1):
            s += a * bernoulli(n - 6*j)
            # Avoid computing each binomial coefficient from scratch
            a *= _product(n - 6 - 6*j + 1, n - 6*j)
            a //= _product(6*j + 4, 6*j + 9)
        if n % 6 == 4:
            s = -Rational(n + 3, 6) - s
        else:
            s = Rational(n + 3, 3) - s
        return s / binomial(n + 3, n)

    # We implement a specialized memoization scheme to handle each
    # case modulo 6 separately
    _cache = {0: S.One, 1: Rational(1, 2), 2: Rational(1, 6), 4: Rational(-1, 30)}
    _highest = {0: 0, 1: 1, 2: 2, 4: 4}

    @classmethod
    def eval(cls, n, x=None):
        if x is S.One:
            return cls(n)
        elif n.is_zero:
            return S.One
        elif n.is_integer is False or n.is_nonnegative is False:
            if x is not None and x.is_Integer and x.is_nonpositive:
                return S.NaN
            return
        # Bernoulli numbers
        elif x is None:
            if n is S.One:
                return S.Half
            elif n.is_odd and (n-1).is_positive:
                return S.Zero
            elif n.is_Number:
                n = int(n)
                # Use mpmath for enormous Bernoulli numbers
                if n > 500:
                    p, q = mp.bernfrac(n)
                    return Rational(int(p), int(q))
                case = n % 6
                highest_cached = cls._highest[case]
                if n <= highest_cached:
                    return cls._cache[n]
                # To avoid excessive recursion when, say, bernoulli(1000) is
                # requested, calculate and cache the entire sequence ... B_988,
                # B_994, B_1000 in increasing order
                for i in range(highest_cached + 6, n + 6, 6):
                    b = cls._calc_bernoulli(i)
                    cls._cache[i] = b
                    cls._highest[case] = i
                return b
        # Bernoulli polynomials
        elif n.is_Number:
            return bernoulli_poly(n, x)

    def _eval_rewrite_as_zeta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        return Piecewise((1, Eq(n, 0)), (-n * zeta(1-n, x), True))

    def _eval_evalf(self, prec):
        if not all(x.is_number for x in self.args):
            return
        n = self.args[0]._to_mpmath(prec)
        x = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        with workprec(prec):
            if n == 0:
                res = mp.mpf(1)
            elif n == 1:
                res = x - mp.mpf(0.5)
            elif mp.isint(n) and n >= 0:
                res = mp.bernoulli(n) if x == 1 else mp.bernpoly(n, x)
            else:
                res = -n * mp.zeta(1-n, x)
        return Expr._from_mpmath(res, prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                                Bell numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class bell(DefinedFunction):
    r"""
    Bell numbers / Bell polynomials

    The Bell numbers satisfy `B_0 = 1` and

    .. math:: B_n = \sum_{k=0}^{n-1} \binom{n-1}{k} B_k.

    They are also given by:

    .. math:: B_n = \frac{1}{e} \sum_{k=0}^{\infty} \frac{k^n}{k!}.

    The Bell polynomials are given by `B_0(x) = 1` and

    .. math:: B_n(x) = x \sum_{k=1}^{n-1} \binom{n-1}{k-1} B_{k-1}(x).

    The second kind of Bell polynomials (are sometimes called "partial" Bell
    polynomials or incomplete Bell polynomials) are defined as

    .. math:: B_{n,k}(x_1, x_2,\dotsc x_{n-k+1}) =
            \sum_{j_1+j_2+j_2+\dotsb=k \atop j_1+2j_2+3j_2+\dotsb=n}
                \frac{n!}{j_1!j_2!\dotsb j_{n-k+1}!}
                \left(\frac{x_1}{1!} \right)^{j_1}
                \left(\frac{x_2}{2!} \right)^{j_2} \dotsb
                \left(\frac{x_{n-k+1}}{(n-k+1)!} \right) ^{j_{n-k+1}}.

    * ``bell(n)`` gives the `n^{th}` Bell number, `B_n`.
    * ``bell(n, x)`` gives the `n^{th}` Bell polynomial, `B_n(x)`.
    * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
      `B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})`.

    Notes
    =====

    Not to be confused with Bernoulli numbers and Bernoulli polynomials,
    which use the same notation.

    Examples
    ========

    >>> from sympy import bell, Symbol, symbols

    >>> [bell(n) for n in range(11)]
    [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    >>> bell(30)
    846749014511809332450147
    >>> bell(4, Symbol('t'))
    t**4 + 6*t**3 + 7*t**2 + t
    >>> bell(6, 2, symbols('x:6')[1:])
    6*x1*x5 + 15*x2*x4 + 10*x3**2

    See Also
    ========

    bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bell_number
    .. [2] https://mathworld.wolfram.com/BellNumber.html
    .. [3] https://mathworld.wolfram.com/BellPolynomial.html

    """

    @staticmethod
    @recurrence_memo([1, 1])
    def _bell(n, prev):
        s = 1
        a = 1
        for k in range(1, n):
            a = a * (n - k) // k
            s += a * prev[k]
        return s

    @staticmethod
    @recurrence_memo([S.One, _sym])
    def _bell_poly(n, prev):
        s = 1
        a = 1
        for k in range(2, n + 1):
            a = a * (n - k + 1) // (k - 1)
            s += a * prev[k - 1]
        return expand_mul(_sym * s)

    @staticmethod
    def _bell_incomplete_poly(n, k, symbols):
        r"""
        The second kind of Bell polynomials (incomplete Bell polynomials).

        Calculated by recurrence formula:

        .. math:: B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1}) =
                \sum_{m=1}^{n-k+1}
                \x_m \binom{n-1}{m-1} B_{n-m,k-1}(x_1, x_2, \dotsc, x_{n-m-k})

        where
            `B_{0,0} = 1;`
            `B_{n,0} = 0; for n \ge 1`
            `B_{0,k} = 0; for k \ge 1`

        """
        if (n == 0) and (k == 0):
            return S.One
        elif (n == 0) or (k == 0):
            return S.Zero
        s = S.Zero
        a = S.One
        for m in range(1, n - k + 2):
            s += a * bell._bell_incomplete_poly(
                n - m, k - 1, symbols) * symbols[m - 1]
            a = a * (n - m) / m
        return expand_mul(s)

    @classmethod
    def eval(cls, n, k_sym=None, symbols=None):
        if n is S.Infinity:
            if k_sym is None:
                return S.Infinity
            else:
                raise ValueError("Bell polynomial is not defined")

        if n.is_negative or n.is_integer is False:
            raise ValueError("a non-negative integer expected")

        if n.is_Integer and n.is_nonnegative:
            if k_sym is None:
                return Integer(cls._bell(int(n)))
            elif symbols is None:
                return cls._bell_poly(int(n)).subs(_sym, k_sym)
            else:
                r = cls._bell_incomplete_poly(int(n), int(k_sym), symbols)
                return r

    def _eval_rewrite_as_Sum(self, n, k_sym=None, symbols=None, **kwargs):
        from sympy.concrete.summations import Sum
        if (k_sym is not None) or (symbols is not None):
            return self

        # Dobinski's formula
        if not n.is_nonnegative:
            return self
        k = Dummy('k', integer=True, nonnegative=True)
        return 1 / E * Sum(k**n / factorial(k), (k, 0, S.Infinity))


#----------------------------------------------------------------------------#
#                                                                            #
#                              Harmonic numbers                              #
#                                                                            #
#----------------------------------------------------------------------------#


class harmonic(DefinedFunction):
    r"""
    Harmonic numbers

    The nth harmonic number is given by `\operatorname{H}_{n} =
    1 + \frac{1}{2} + \frac{1}{3} + \ldots + \frac{1}{n}`.

    More generally:

    .. math:: \operatorname{H}_{n,m} = \sum_{k=1}^{n} \frac{1}{k^m}

    As `n \rightarrow \infty`, `\operatorname{H}_{n,m} \rightarrow \zeta(m)`,
    the Riemann zeta function.

    * ``harmonic(n)`` gives the nth harmonic number, `\operatorname{H}_n`

    * ``harmonic(n, m)`` gives the nth generalized harmonic number
      of order `m`, `\operatorname{H}_{n,m}`, where
      ``harmonic(n) == harmonic(n, 1)``

    This function can be extended to complex `n` and `m` where `n` is not a
    negative integer or `m` is a nonpositive integer as

    .. math:: \operatorname{H}_{n,m} = \begin{cases} \zeta(m) - \zeta(m, n+1)
            & m \ne 1 \\ \psi(n+1) + \gamma & m = 1 \end{cases}

    Examples
    ========

    >>> from sympy import harmonic, oo

    >>> [harmonic(n) for n in range(6)]
    [0, 1, 3/2, 11/6, 25/12, 137/60]
    >>> [harmonic(n, 2) for n in range(6)]
    [0, 1, 5/4, 49/36, 205/144, 5269/3600]
    >>> harmonic(oo, 2)
    pi**2/6

    >>> from sympy import Symbol, Sum
    >>> n = Symbol("n")

    >>> harmonic(n).rewrite(Sum)
    Sum(1/_k, (_k, 1, n))

    We can evaluate harmonic numbers for all integral and positive
    rational arguments:

    >>> from sympy import S, expand_func, simplify
    >>> harmonic(8)
    761/280
    >>> harmonic(11)
    83711/27720

    >>> H = harmonic(1/S(3))
    >>> H
    harmonic(1/3)
    >>> He = expand_func(H)
    >>> He
    -log(6) - sqrt(3)*pi/6 + 2*Sum(log(sin(_k*pi/3))*cos(2*_k*pi/3), (_k, 1, 1))
                           + 3*Sum(1/(3*_k + 1), (_k, 0, 0))
    >>> He.doit()
    -log(6) - sqrt(3)*pi/6 - log(sqrt(3)/2) + 3
    >>> H = harmonic(25/S(7))
    >>> He = simplify(expand_func(H).doit())
    >>> He
    log(sin(2*pi/7)**(2*cos(16*pi/7))/(14*sin(pi/7)**(2*cos(pi/7))*cos(pi/14)**(2*sin(pi/14)))) + pi*tan(pi/14)/2 + 30247/9900
    >>> He.n(40)
    1.983697455232980674869851942390639915940
    >>> harmonic(25/S(7)).n(40)
    1.983697455232980674869851942390639915940

    We can rewrite harmonic numbers in terms of polygamma functions:

    >>> from sympy import digamma, polygamma
    >>> m = Symbol("m", integer=True, positive=True)

    >>> harmonic(n).rewrite(digamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n).rewrite(polygamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n,3).rewrite(polygamma)
    polygamma(2, n + 1)/2 + zeta(3)

    >>> simplify(harmonic(n,m).rewrite(polygamma))
    Piecewise((polygamma(0, n + 1) + EulerGamma, Eq(m, 1)),
    (-(-1)**m*polygamma(m - 1, n + 1)/factorial(m - 1) + zeta(m), True))

    Integer offsets in the argument can be pulled out:

    >>> from sympy import expand_func

    >>> expand_func(harmonic(n+4))
    harmonic(n) + 1/(n + 4) + 1/(n + 3) + 1/(n + 2) + 1/(n + 1)

    >>> expand_func(harmonic(n-4))
    harmonic(n) - 1/(n - 1) - 1/(n - 2) - 1/(n - 3) - 1/n

    Some limits can be computed as well:

    >>> from sympy import limit, oo

    >>> limit(harmonic(n), n, oo)
    oo

    >>> limit(harmonic(n, 2), n, oo)
    pi**2/6

    >>> limit(harmonic(n, 3), n, oo)
    zeta(3)

    For `m > 1`, `H_{n,m}` tends to `\zeta(m)` in the limit of infinite `n`:

    >>> m = Symbol("m", positive=True)
    >>> limit(harmonic(n, m+1), n, oo)
    zeta(m + 1)

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Harmonic_number
    .. [2] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber/
    .. [3] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber2/

    """

    # This prevents redundant recalculations and speeds up harmonic number computations.
    harmonic_cache: dict[Integer, Callable[[int], Rational]] = {}

    @classmethod
    def eval(cls, n, m=None):
        from sympy.functions.special.zeta_functions import zeta
        if m is S.One:
            return cls(n)
        if m is None:
            m = S.One
        if n.is_zero:
            return S.Zero
        elif m.is_zero:
            return n
        elif n is S.Infinity:
            if m.is_negative:
                return S.NaN
            elif is_le(m, S.One):
                return S.Infinity
            elif is_gt(m, S.One):
                return zeta(m)
        elif m.is_Integer and m.is_nonpositive:
            return (bernoulli(1-m, n+1) - bernoulli(1-m)) / (1-m)
        elif n.is_Integer:
            if n.is_negative and (m.is_integer is False or m.is_nonpositive is False):
                return S.ComplexInfinity if m is S.One else S.NaN
            if n.is_nonnegative:
                if m.is_Integer:
                    if m not in cls.harmonic_cache:
                        @recurrence_memo([0])
                        def f(n, prev):
                            return prev[-1] + S.One / n**m
                        cls.harmonic_cache[m] = f
                    return cls.harmonic_cache[m](int(n))
                return Add(*(k**(-m) for k in range(1, int(n) + 1)))

    def _eval_rewrite_as_polygamma(self, n, m=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import gamma, polygamma
        if m.is_integer and m.is_positive:
            return Piecewise((polygamma(0, n+1) + S.EulerGamma, Eq(m, 1)),
                    (S.NegativeOne**m * (polygamma(m-1, 1) - polygamma(m-1, n+1)) /
                    gamma(m), True))

    def _eval_rewrite_as_digamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)

    def _eval_rewrite_as_trigamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)

    def _eval_rewrite_as_Sum(self, n, m=None, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k", integer=True)
        if m is None:
            m = S.One
        return Sum(k**(-m), (k, 1, n))

    def _eval_rewrite_as_zeta(self, n, m=S.One, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((digamma(n + 1) + S.EulerGamma, Eq(m, 1)),
                         (zeta(m) - zeta(m, n+1), True))

    def _eval_expand_func(self, **hints):
        from sympy.concrete.summations import Sum
        n = self.args[0]
        m = self.args[1] if len(self.args) == 2 else 1

        if m == S.One:
            if n.is_Add:
                off = n.args[0]
                nnew = n - off
                if off.is_Integer and off.is_positive:
                    result = [S.One/(nnew + i) for i in range(off, 0, -1)] + [harmonic(nnew)]
                    return Add(*result)
                elif off.is_Integer and off.is_negative:
                    result = [-S.One/(nnew + i) for i in range(0, off, -1)] + [harmonic(nnew)]
                    return Add(*result)

            if n.is_Rational:
                # Expansions for harmonic numbers at general rational arguments (u + p/q)
                # Split n as u + p/q with p < q
                p, q = n.as_numer_denom()
                u = p // q
                p = p - u * q
                if u.is_nonnegative and p.is_positive and q.is_positive and p < q:
                    from sympy.functions.elementary.exponential import log
                    from sympy.functions.elementary.integers import floor
                    from sympy.functions.elementary.trigonometric import sin, cos, cot
                    k = Dummy("k")
                    t1 = q * Sum(1 / (q * k + p), (k, 0, u))
                    t2 = 2 * Sum(cos((2 * pi * p * k) / S(q)) *
                                   log(sin((pi * k) / S(q))),
                                   (k, 1, floor((q - 1) / S(2))))
                    t3 = (pi / 2) * cot((pi * p) / q) + log(2 * q)
                    return t1 + t2 - t3

        return self

    def _eval_rewrite_as_tractable(self, n, m=1, limitvar=None, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import polygamma
        pg = self.rewrite(polygamma)
        if not isinstance(pg, harmonic):
            return pg.rewrite("tractable", deep=True)
        arg = m - S.One
        if arg.is_nonzero:
            return (zeta(m) - zeta(m, n+1)).rewrite("tractable", deep=True)

    def _eval_evalf(self, prec):
        if not all(x.is_number for x in self.args):
            return
        n = self.args[0]._to_mpmath(prec)
        m = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        if mp.isint(n) and n < 0:
            return S.NaN
        with workprec(prec):
            if m == 1:
                res = mp.harmonic(n)
            else:
                res = mp.zeta(m) - mp.zeta(m, n+1)
        return Expr._from_mpmath(res, prec)

    def fdiff(self, argindex=1):
        from sympy.functions.special.zeta_functions import zeta
        if len(self.args) == 2:
            n, m = self.args
        else:
            n, m = self.args + (1,)
        if argindex == 1:
            return m * zeta(m+1, n+1)
        else:
            raise ArgumentIndexError


#----------------------------------------------------------------------------#
#                                                                            #
#                           Euler numbers                                    #
#                                                                            #
#----------------------------------------------------------------------------#


class euler(DefinedFunction):
    r"""
    Euler numbers / Euler polynomials / Euler function

    The Euler numbers are given by:

    .. math:: E_{2n} = I \sum_{k=1}^{2n+1} \sum_{j=0}^k \binom{k}{j}
        \frac{(-1)^j (k-2j)^{2n+1}}{2^k I^k k}

    .. math:: E_{2n+1} = 0

    Euler numbers and Euler polynomials are related by

    .. math:: E_n = 2^n E_n\left(\frac{1}{2}\right).

    We compute symbolic Euler polynomials using Appell sequences,
    but numerical evaluation of the Euler polynomial is computed
    more efficiently (and more accurately) using the mpmath library.

    The Euler polynomials are special cases of the generalized Euler function,
    related to the Genocchi function as

    .. math:: \operatorname{E}(s, a) = -\frac{\operatorname{G}(s+1, a)}{s+1}

    with the limit of `\psi\left(\frac{a+1}{2}\right) - \psi\left(\frac{a}{2}\right)`
    being taken when `s = -1`. The (ordinary) Euler function interpolating
    the Euler numbers is then obtained as
    `\operatorname{E}(s) = 2^s \operatorname{E}\left(s, \frac{1}{2}\right)`.

    * ``euler(n)`` gives the nth Euler number `E_n`.
    * ``euler(s)`` gives the Euler function `\operatorname{E}(s)`.
    * ``euler(n, x)`` gives the nth Euler polynomial `E_n(x)`.
    * ``euler(s, a)`` gives the generalized Euler function `\operatorname{E}(s, a)`.

    Examples
    ========

    >>> from sympy import euler, Symbol, S
    >>> [euler(n) for n in range(10)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0]
    >>> [2**n*euler(n,1) for n in range(10)]
    [1, 1, 0, -2, 0, 16, 0, -272, 0, 7936]
    >>> n = Symbol("n")
    >>> euler(n + 2*n)
    euler(3*n)

    >>> x = Symbol("x")
    >>> euler(n, x)
    euler(n, x)

    >>> euler(0, x)
    1
    >>> euler(1, x)
    x - 1/2
    >>> euler(2, x)
    x**2 - x
    >>> euler(3, x)
    x**3 - 3*x**2/2 + 1/4
    >>> euler(4, x)
    x**4 - 2*x**3 + x

    >>> euler(12, S.Half)
    2702765/4096
    >>> euler(12)
    2702765

    See Also
    ========

    andre, bell, bernoulli, catalan, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.polys.appellseqs.euler_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler_numbers
    .. [2] https://mathworld.wolfram.com/EulerNumber.html
    .. [3] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [4] https://mathworld.wolfram.com/AlternatingPermutation.html

    """

    @classmethod
    def eval(cls, n, x=None):
        if n.is_zero:
            return S.One
        elif n is S.NegativeOne:
            if x is None:
                return S.Pi/2
            from sympy.functions.special.gamma_functions import digamma
            return digamma((x+1)/2) - digamma(x/2)
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        # Euler numbers
        elif x is None:
            if n.is_odd and n.is_positive:
                return S.Zero
            elif n.is_Number:
                from mpmath import mp
                n = n._to_mpmath(mp.prec)
                res = mp.eulernum(n, exact=True)
                return Integer(res)
        # Euler polynomials
        elif n.is_Number:
            from sympy.core.evalf import pure_complex
            n = int(n)
            reim = pure_complex(x, or_real=True)
            if reim and all(a.is_Float or a.is_Integer for a in reim) \
                    and any(a.is_Float for a in reim):
                from mpmath import mp
                prec = min([a._prec for a in reim if a.is_Float])
                with workprec(prec):
                    res = mp.eulerpoly(n, x)
                return Expr._from_mpmath(res, prec)
            return euler_poly(n, x)

    def _eval_rewrite_as_Sum(self, n, x=None, **kwargs):
        from sympy.concrete.summations import Sum
        if x is None and n.is_even:
            k = Dummy("k", integer=True)
            j = Dummy("j", integer=True)
            n = n / 2
            Em = (S.ImaginaryUnit * Sum(Sum(binomial(k, j) * (S.NegativeOne**j *
                                                              (k - 2*j)**(2*n + 1)) /
                  (2**k*S.ImaginaryUnit**k * k), (j, 0, k)), (k, 1, 2*n + 1)))
            return Em
        if x:
            k = Dummy("k", integer=True)
            return Sum(binomial(n, k)*euler(k)/2**k*(x - S.Half)**(n - k), (k, 0, n))

    def _eval_rewrite_as_genocchi(self, n, x=None, **kwargs):
        if x is None:
            return Piecewise((S.Pi/2, Eq(n, -1)),
                             (-2**n * genocchi(n+1, S.Half) / (n+1), True))
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((digamma((x+1)/2) - digamma(x/2), Eq(n, -1)),
                         (-genocchi(n+1, x) / (n+1), True))

    def _eval_evalf(self, prec):
        if not all(i.is_number for i in self.args):
            return
        from mpmath import mp
        m, x = (self.args[0], None) if len(self.args) == 1 else self.args
        m = m._to_mpmath(prec)
        if x is not None:
            x = x._to_mpmath(prec)
        with workprec(prec):
            if mp.isint(m) and m >= 0:
                res = mp.eulernum(m) if x is None else mp.eulerpoly(m, x)
            else:
                if m == -1:
                    res = mp.pi if x is None else mp.digamma((x+1)/2) - mp.digamma(x/2)
                else:
                    y = 0.5 if x is None else x
                    res = 2 * (mp.zeta(-m, y) - 2**(m+1) * mp.zeta(-m, (y+1)/2))
                if x is None:
                    res *= 2**m
        return Expr._from_mpmath(res, prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                              Catalan numbers                               #
#                                                                            #
#----------------------------------------------------------------------------#


class catalan(DefinedFunction):
    r"""
    Catalan numbers

    The `n^{th}` catalan number is given by:

    .. math :: C_n = \frac{1}{n+1} \binom{2n}{n}

    * ``catalan(n)`` gives the `n^{th}` Catalan number, `C_n`

    Examples
    ========

    >>> from sympy import (Symbol, binomial, gamma, hyper,
    ...     catalan, diff, combsimp, Rational, I)

    >>> [catalan(i) for i in range(1,10)]
    [1, 2, 5, 14, 42, 132, 429, 1430, 4862]

    >>> n = Symbol("n", integer=True)

    >>> catalan(n)
    catalan(n)

    Catalan numbers can be transformed into several other, identical
    expressions involving other mathematical functions

    >>> catalan(n).rewrite(binomial)
    binomial(2*n, n)/(n + 1)

    >>> catalan(n).rewrite(gamma)
    4**n*gamma(n + 1/2)/(sqrt(pi)*gamma(n + 2))

    >>> catalan(n).rewrite(hyper)
    hyper((-n, 1 - n), (2,), 1)

    For some non-integer values of n we can get closed form
    expressions by rewriting in terms of gamma functions:

    >>> catalan(Rational(1, 2)).rewrite(gamma)
    8/(3*pi)

    We can differentiate the Catalan numbers C(n) interpreted as a
    continuous real function in n:

    >>> diff(catalan(n), n)
    (polygamma(0, n + 1/2) - polygamma(0, n + 2) + log(4))*catalan(n)

    As a more advanced example consider the following ratio
    between consecutive numbers:

    >>> combsimp((catalan(n + 1)/catalan(n)).rewrite(binomial))
    2*(2*n + 1)/(n + 2)

    The Catalan numbers can be generalized to complex numbers:

    >>> catalan(I).rewrite(gamma)
    4**I*gamma(1/2 + I)/(sqrt(pi)*gamma(2 + I))

    and evaluated with arbitrary precision:

    >>> catalan(I).evalf(20)
    0.39764993382373624267 - 0.020884341620842555705*I

    See Also
    ========

    andre, bell, bernoulli, euler, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.functions.combinatorial.factorials.binomial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Catalan_number
    .. [2] https://mathworld.wolfram.com/CatalanNumber.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/CatalanNumber/
    .. [4] http://geometer.org/mathcircles/catalan.pdf

    """

    @classmethod
    def eval(cls, n):
        from sympy.functions.special.gamma_functions import gamma
        if (n.is_Integer and n.is_nonnegative) or \
           (n.is_noninteger and n.is_negative):
            return 4**n*gamma(n + S.Half)/(gamma(S.Half)*gamma(n + 2))

        if (n.is_integer and n.is_negative):
            if (n + 1).is_negative:
                return S.Zero
            if (n + 1).is_zero:
                return Rational(-1, 2)

    def fdiff(self, argindex=1):
        from sympy.functions.elementary.exponential import log
        from sympy.functions.special.gamma_functions import polygamma
        n = self.args[0]
        return catalan(n)*(polygamma(0, n + S.Half) - polygamma(0, n + 2) + log(4))

    def _eval_rewrite_as_binomial(self, n, **kwargs):
        return binomial(2*n, n)/(n + 1)

    def _eval_rewrite_as_factorial(self, n, **kwargs):
        return factorial(2*n) / (factorial(n+1) * factorial(n))

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        # The gamma function allows to generalize Catalan numbers to complex n
        return 4**n*gamma(n + S.Half)/(gamma(S.Half)*gamma(n + 2))

    def _eval_rewrite_as_hyper(self, n, **kwargs):
        from sympy.functions.special.hyper import hyper
        return hyper([1 - n, -n], [2], 1)

    def _eval_rewrite_as_Product(self, n, **kwargs):
        from sympy.concrete.products import Product
        if not (n.is_integer and n.is_nonnegative):
            return self
        k = Dummy('k', integer=True, positive=True)
        return Product((n + k) / k, (k, 2, n))

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_is_composite(self):
        if self.args[0].is_integer and (self.args[0] - 3).is_positive:
            return True

    def _eval_evalf(self, prec):
        from sympy.functions.special.gamma_functions import gamma
        if self.args[0].is_number:
            return self.rewrite(gamma)._eval_evalf(prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Genocchi numbers                                 #
#                                                                            #
#----------------------------------------------------------------------------#


class genocchi(DefinedFunction):
    r"""
    Genocchi numbers / Genocchi polynomials / Genocchi function

    The Genocchi numbers are a sequence of integers `G_n` that satisfy the
    relation:

    .. math:: \frac{-2t}{1 + e^{-t}} = \sum_{n=0}^\infty \frac{G_n t^n}{n!}

    They are related to the Bernoulli numbers by

    .. math:: G_n = 2 (1 - 2^n) B_n

    and generalize like the Bernoulli numbers to the Genocchi polynomials and
    function as

    .. math:: \operatorname{G}(s, a) = 2 \left(\operatorname{B}(s, a) -
              2^s \operatorname{B}\left(s, \frac{a+1}{2}\right)\right)

    .. versionchanged:: 1.12
        ``genocchi(1)`` gives `-1` instead of `1`.

    Examples
    ========

    >>> from sympy import genocchi, Symbol
    >>> [genocchi(n) for n in range(9)]
    [0, -1, -1, 0, 1, 0, -3, 0, 17]
    >>> n = Symbol('n', integer=True, positive=True)
    >>> genocchi(2*n + 1)
    0
    >>> x = Symbol('x')
    >>> genocchi(4, x)
    -4*x**3 + 6*x**2 - 1

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, partition, tribonacci
    sympy.polys.appellseqs.genocchi_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Genocchi_number
    .. [2] https://mathworld.wolfram.com/GenocchiNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    def eval(cls, n, x=None):
        if x is S.One:
            return cls(n)
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        # Genocchi numbers
        elif x is None:
            if n.is_odd and (n-1).is_positive:
                return S.Zero
            elif n.is_Number:
                return 2 * (1-S(2)**n) * bernoulli(n)
        # Genocchi polynomials
        elif n.is_Number:
            return genocchi_poly(n, x)

    def _eval_rewrite_as_bernoulli(self, n, x=1, **kwargs):
        if x == 1 and n.is_integer and n.is_nonnegative:
            return 2 * (1-S(2)**n) * bernoulli(n)
        return 2 * (bernoulli(n, x) - 2**n * bernoulli(n, (x+1) / 2))

    def _eval_rewrite_as_dirichlet_eta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import dirichlet_eta
        return -2*n * dirichlet_eta(1-n, x)

    def _eval_is_integer(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            return True

    def _eval_is_negative(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_odd:
                return fuzzy_not((n-1).is_positive)
            return (n/2).is_odd

    def _eval_is_positive(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_zero or n.is_odd:
                return False
            return (n/2).is_even

    def _eval_is_even(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_even:
                return n.is_zero
            return (n-1).is_positive

    def _eval_is_odd(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_even:
                return fuzzy_not(n.is_zero)
            return fuzzy_not((n-1).is_positive)

    def _eval_is_prime(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        # only G_6 = -3 and G_8 = 17 are prime,
        # but SymPy does not consider negatives as prime
        # so only n=8 is tested
        return (n-8).is_zero

    def _eval_evalf(self, prec):
        if all(i.is_number for i in self.args):
            return self.rewrite(bernoulli)._eval_evalf(prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                              Andre numbers                                 #
#                                                                            #
#----------------------------------------------------------------------------#


class andre(DefinedFunction):
    r"""
    Andre numbers / Andre function

    The Andre number `\mathcal{A}_n` is Luschny's name for half the number of
    *alternating permutations* on `n` elements, where a permutation is alternating
    if adjacent elements alternately compare "greater" and "smaller" going from
    left to right. For example, `2 < 3 > 1 < 4` is an alternating permutation.

    This sequence is A000111 in the OEIS, which assigns the names *up/down numbers*
    and *Euler zigzag numbers*. It satisfies a recurrence relation similar to that
    for the Catalan numbers, with `\mathcal{A}_0 = 1` and

    .. math:: 2 \mathcal{A}_{n+1} = \sum_{k=0}^n \binom{n}{k} \mathcal{A}_k \mathcal{A}_{n-k}

    The Bernoulli and Euler numbers are signed transformations of the odd- and
    even-indexed elements of this sequence respectively:

    .. math :: \operatorname{B}_{2k} = \frac{2k \mathcal{A}_{2k-1}}{(-4)^k - (-16)^k}

    .. math :: \operatorname{E}_{2k} = (-1)^k \mathcal{A}_{2k}

    Like the Bernoulli and Euler numbers, the Andre numbers are interpolated by the
    entire Andre function:

    .. math :: \mathcal{A}(s) = (-i)^{s+1} \operatorname{Li}_{-s}(i) +
            i^{s+1} \operatorname{Li}_{-s}(-i) = \\ \frac{2 \Gamma(s+1)}{(2\pi)^{s+1}}
            (\zeta(s+1, 1/4) - \zeta(s+1, 3/4) \cos{\pi s})

    Examples
    ========

    >>> from sympy import andre, euler, bernoulli
    >>> [andre(n) for n in range(11)]
    [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]
    >>> [(-1)**k * andre(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [euler(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [andre(2*k-1) * (2*k) / ((-4)**k - (-16)**k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]
    >>> [bernoulli(2*k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]

    See Also
    ========

    bernoulli, catalan, euler, sympy.polys.appellseqs.andre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [2] https://mathworld.wolfram.com/EulerZigzagNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743
    """

    @classmethod
    def eval(cls, n):
        if n is S.NaN:
            return S.NaN
        elif n is S.Infinity:
            return S.Infinity
        if n.is_zero:
            return S.One
        elif n == -1:
            return -log(2)
        elif n == -2:
            return -2*S.Catalan
        elif n.is_Integer:
            if n.is_nonnegative and n.is_even:
                return abs(euler(n))
            elif n.is_odd:
                from sympy.functions.special.zeta_functions import zeta
                m = -n-1
                return I**m * Rational(1-2**m, 4**m) * zeta(-n)

    def _eval_rewrite_as_zeta(self, s, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        from sympy.functions.special.gamma_functions import gamma
        from sympy.functions.special.zeta_functions import zeta
        return 2 * gamma(s+1) / (2*pi)**(s+1) * \
                (zeta(s+1, S.One/4) - cos(pi*s) * zeta(s+1, S(3)/4))

    def _eval_rewrite_as_polylog(self, s, **kwargs):
        from sympy.functions.special.zeta_functions import polylog
        return (-I)**(s+1) * polylog(-s, I) + I**(s+1) * polylog(-s, -I)

    def _eval_is_integer(self):
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_evalf(self, prec):
        if not self.args[0].is_number:
            return
        s = self.args[0]._to_mpmath(prec+12)
        with workprec(prec+12):
            sp, cp = mp.sinpi(s/2), mp.cospi(s/2)
            res = 2*mp.dirichlet(-s, (-sp, cp, sp, -cp))
        return Expr._from_mpmath(res, prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Partition numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#

class partition(DefinedFunction):
    r"""
    Partition numbers

    The Partition numbers are a sequence of integers `p_n` that represent the
    number of distinct ways of representing `n` as a sum of natural numbers
    (with order irrelevant). The generating function for `p_n` is given by:

    .. math:: \sum_{n=0}^\infty p_n x^n = \prod_{k=1}^\infty (1 - x^k)^{-1}

    Examples
    ========

    >>> from sympy import partition, Symbol
    >>> [partition(n) for n in range(9)]
    [1, 1, 2, 3, 5, 7, 11, 15, 22]
    >>> n = Symbol('n', integer=True, negative=True)
    >>> partition(n)
    0

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_(number_theory%29
    .. [2] https://en.wikipedia.org/wiki/Pentagonal_number_theorem

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_negative is True:
            return S.Zero
        if n.is_zero is True or n is S.One:
            return S.One
        if n.is_Integer is True:
            return S(_partition(as_int(n)))

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative is True:
            return True

    def _eval_Mod(self, q):
        # Ramanujan's congruences
        n = self.args[0]
        for p, rem in [(5, 4), (7, 5), (11, 6)]:
            if q == p and n % q == rem:
                return S.Zero


class divisor_sigma(DefinedFunction):
    r"""
    Calculate the divisor function `\sigma_k(n)` for positive integer n

    ``divisor_sigma(n, k)`` is equal to ``sum([x**k for x in divisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k(n) = \prod_{i=1}^\omega (1+p_i^k+p_i^{2k}+\cdots
        + p_i^{m_ik}).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> divisor_sigma(18, 0)
    6
    >>> divisor_sigma(39, 1)
    56
    >>> divisor_sigma(12, 2)
    210
    >>> divisor_sigma(37)
    38

    See Also
    ========

    sympy.ntheory.factor_.divisor_count, totient, sympy.ntheory.factor_.divisors, sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Divisor_function

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n, k=S.One):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        if n.is_prime is True:
            return 1 + n**k
        if n is S.One:
            return S.One
        if n.is_Integer is True:
            if k.is_zero is True:
                return Mul(*[e + 1 for e in factorint(n).values()])
            if k.is_Integer is True:
                return S(_divisor_sigma(as_int(n), as_int(k)))
            if k.is_zero is False:
                return Mul(*[cancel((p**(k*(e + 1)) - 1) / (p**k - 1)) for p, e in factorint(n).items()])


class udivisor_sigma(DefinedFunction):
    r"""
    Calculate the unitary divisor function `\sigma_k^*(n)` for positive integer n

    ``udivisor_sigma(n, k)`` is equal to ``sum([x**k for x in udivisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k^*(n) = \prod_{i=1}^\omega (1+ p_i^{m_ik}).

    Parameters
    ==========

    k : power of divisors in the sum

        for k = 0, 1:
        ``udivisor_sigma(n, 0)`` is equal to ``udivisor_count(n)``
        ``udivisor_sigma(n, 1)`` is equal to ``sum(udivisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import udivisor_sigma
    >>> udivisor_sigma(18, 0)
    4
    >>> udivisor_sigma(74, 1)
    114
    >>> udivisor_sigma(36, 3)
    47450
    >>> udivisor_sigma(111)
    152

    See Also
    ========

    sympy.ntheory.factor_.divisor_count, totient, sympy.ntheory.factor_.divisors,
    sympy.ntheory.factor_.udivisors, sympy.ntheory.factor_.udivisor_count, divisor_sigma,
    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n, k=S.One):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        if n.is_prime is True:
            return 1 + n**k
        if n.is_Integer:
            return Mul(*[1+p**(k*e) for p, e in factorint(n).items()])


class legendre_symbol(DefinedFunction):
    r"""
    Returns the Legendre symbol `(a / p)`.

    For an integer ``a`` and an odd prime ``p``, the Legendre symbol is
    defined as

    .. math ::
        \genfrac(){}{}{a}{p} = \begin{cases}
             0 & \text{if } p \text{ divides } a\\
             1 & \text{if } a \text{ is a quadratic residue modulo } p\\
            -1 & \text{if } a \text{ is a quadratic nonresidue modulo } p
        \end{cases}

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import legendre_symbol
    >>> [legendre_symbol(i, 7) for i in range(7)]
    [0, 1, 1, -1, 1, -1, -1]
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]

    See Also
    ========

    sympy.ntheory.residue_ntheory.is_quad_residue, jacobi_symbol

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, a, p):
        if a.is_integer is False:
            raise TypeError("a should be an integer")
        if p.is_integer is False:
            raise TypeError("p should be an integer")
        if p.is_prime is False or p.is_odd is False:
            raise ValueError("p should be an odd prime integer")
        if (a % p).is_zero is True:
            return S.Zero
        if a is S.One:
            return S.One
        if a.is_Integer is True and p.is_Integer is True:
            return S(legendre(as_int(a), as_int(p)))


class jacobi_symbol(DefinedFunction):
    r"""
    Returns the Jacobi symbol `(m / n)`.

    For any integer ``m`` and any positive odd integer ``n`` the Jacobi symbol
    is defined as the product of the Legendre symbols corresponding to the
    prime factors of ``n``:

    .. math ::
        \genfrac(){}{}{m}{n} =
            \genfrac(){}{}{m}{p^{1}}^{\alpha_1}
            \genfrac(){}{}{m}{p^{2}}^{\alpha_2}
            ...
            \genfrac(){}{}{m}{p^{k}}^{\alpha_k}
            \text{ where } n =
                p_1^{\alpha_1}
                p_2^{\alpha_2}
                ...
                p_k^{\alpha_k}

    Like the Legendre symbol, if the Jacobi symbol `\genfrac(){}{}{m}{n} = -1`
    then ``m`` is a quadratic nonresidue modulo ``n``.

    But, unlike the Legendre symbol, if the Jacobi symbol
    `\genfrac(){}{}{m}{n} = 1` then ``m`` may or may not be a quadratic residue
    modulo ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import jacobi_symbol, legendre_symbol
    >>> from sympy import S
    >>> jacobi_symbol(45, 77)
    -1
    >>> jacobi_symbol(60, 121)
    1

    The relationship between the ``jacobi_symbol`` and ``legendre_symbol`` can
    be demonstrated as follows:

    >>> L = legendre_symbol
    >>> S(45).factors()
    {3: 2, 5: 1}
    >>> jacobi_symbol(7, 45) == L(7, 3)**2 * L(7, 5)**1
    True

    See Also
    ========

    sympy.ntheory.residue_ntheory.is_quad_residue, legendre_symbol

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, m, n):
        if m.is_integer is False:
            raise TypeError("m should be an integer")
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False or n.is_odd is False:
            raise ValueError("n should be an odd positive integer")
        if m is S.One or n is S.One:
            return S.One
        if (m % n).is_zero is True:
            return S.Zero
        if m.is_Integer is True and n.is_Integer is True:
            return S(jacobi(as_int(m), as_int(n)))


class kronecker_symbol(DefinedFunction):
    r"""
    Returns the Kronecker symbol `(a / n)`.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import kronecker_symbol
    >>> kronecker_symbol(45, 77)
    -1
    >>> kronecker_symbol(13, -120)
    1

    See Also
    ========

    jacobi_symbol, legendre_symbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_symbol

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, a, n):
        if a.is_integer is False:
            raise TypeError("a should be an integer")
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if a is S.One or n is S.One:
            return S.One
        if a.is_Integer is True and n.is_Integer is True:
            return S(kronecker(as_int(a), as_int(n)))


class mobius(DefinedFunction):
    """
    Mobius function maps natural number to {-1, 0, 1}

    It is defined as follows:
        1) `1` if `n = 1`.
        2) `0` if `n` has a squared prime factor.
        3) `(-1)^k` if `n` is a square-free positive integer with `k`
           number of prime factors.

    It is an important multiplicative function in number theory
    and combinatorics.  It has applications in mathematical series,
    algebraic number theory and also physics (Fermion operator has very
    concrete realization with Mobius Function model).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import mobius
    >>> mobius(13*7)
    1
    >>> mobius(1)
    1
    >>> mobius(13*7*5)
    -1
    >>> mobius(13**2)
    0

    Even in the case of a symbol, if it clearly contains a squared prime factor, it will be zero.

    >>> from sympy import Symbol
    >>> n = Symbol("n", integer=True, positive=True)
    >>> mobius(4*n)
    0
    >>> mobius(n**2)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_function
    .. [2] Thomas Koshy "Elementary Number Theory with Applications"
    .. [3] https://oeis.org/A008683

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:
            return S.NegativeOne
        if n is S.One:
            return S.One
        result = None
        for m, e in (_.as_base_exp() for _ in Mul.make_args(n)):
            if m.is_integer is True and m.is_positive is True and \
               e.is_integer is True and e.is_positive is True:
                lt = is_lt(S.One, e) # 1 < e
                if lt is True:
                    result = S.Zero
                elif m.is_Integer is True:
                    factors = factorint(m)
                    if any(v > 1 for v in factors.values()):
                        result = S.Zero
                    elif lt is False:
                        s = S.NegativeOne if len(factors) % 2 else S.One
                        if result is None:
                            result = s
                        else:
                            result *= s
            else:
                return
        return result


class primenu(DefinedFunction):
    r"""
    Calculate the number of distinct prime factors for a positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primenu(n)`` or `\nu(n)` is:

    .. math ::
        \nu(n) = k.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primenu
    >>> primenu(1)
    0
    >>> primenu(30)
    3

    See Also
    ========

    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html
    .. [2] https://oeis.org/A001221

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:
            return S.One
        if n is S.One:
            return S.Zero
        if n.is_Integer is True:
            return S(len(factorint(n)))


class primeomega(DefinedFunction):
    r"""
    Calculate the number of prime factors counting multiplicities for a
    positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primeomega(n)``  or `\Omega(n)` is:

    .. math ::
        \Omega(n) = \sum_{i=1}^k m_i.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primeomega
    >>> primeomega(1)
    0
    >>> primeomega(20)
    3

    See Also
    ========

    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html
    .. [2] https://oeis.org/A001222

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:
            return S.One
        if n is S.One:
            return S.Zero
        if n.is_Integer is True:
            return S(sum(factorint(n).values()))


class totient(DefinedFunction):
    r"""
    Calculate the Euler totient function phi(n)

    ``totient(n)`` or `\phi(n)` is the number of positive integers `\leq` n
    that are relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> totient(1)
    1
    >>> totient(25)
    20
    >>> totient(45) == totient(5)*totient(9)
    True

    See Also
    ========

    sympy.ntheory.factor_.divisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%27s_totient_function
    .. [2] https://mathworld.wolfram.com/TotientFunction.html
    .. [3] https://oeis.org/A000010

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n is S.One:
            return S.One
        if n.is_prime is True:
            return n - 1
        if isinstance(n, Dict):
            return S(prod(p**(k-1)*(p-1) for p, k in n.items()))
        if n.is_Integer is True:
            return S(prod(p**(k-1)*(p-1) for p, k in factorint(n).items()))


class reduced_totient(DefinedFunction):
    r"""
    Calculate the Carmichael reduced totient function lambda(n)

    ``reduced_totient(n)`` or `\lambda(n)` is the smallest m > 0 such that
    `k^m \equiv 1 \mod n` for all k relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import reduced_totient
    >>> reduced_totient(1)
    1
    >>> reduced_totient(8)
    2
    >>> reduced_totient(30)
    4

    See Also
    ========

    totient

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_function
    .. [2] https://mathworld.wolfram.com/CarmichaelFunction.html
    .. [3] https://oeis.org/A002322

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n is S.One:
            return S.One
        if n.is_prime is True:
            return n - 1
        if isinstance(n, Dict):
            t = 1
            if 2 in n:
                t = (1 << (n[2] - 2)) if 2 < n[2] else n[2]
            return S(lcm(int(t), *(int(p-1)*int(p)**int(k-1) for p, k in n.items() if p != 2)))
        if n.is_Integer is True:
            n, t = remove(int(n), 2)
            if not t:
                t = 1
            elif 2 < t:
                t = 1 << (t - 2)
            return S(lcm(t, *((p-1)*p**(k-1) for p, k in factorint(n).items())))


class primepi(DefinedFunction):
    r""" Represents the prime counting function pi(n) = the number
    of prime numbers less than or equal to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primepi
    >>> from sympy import prime, prevprime, isprime
    >>> primepi(25)
    9

    So there are 9 primes less than or equal to 25. Is 25 prime?

    >>> isprime(25)
    False

    It is not. So the first prime less than 25 must be the
    9th prime:

    >>> prevprime(25) == prime(9)
    True

    See Also
    ========

    sympy.ntheory.primetest.isprime : Test if n is prime
    sympy.ntheory.generate.primerange : Generate all primes in a given range
    sympy.ntheory.generate.prime : Return the nth prime

    References
    ==========

    .. [1] https://oeis.org/A000720

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n is S.Infinity:
            return S.Infinity
        if n is S.NegativeInfinity:
            return S.Zero
        if n.is_real is False:
            raise TypeError("n should be a real")
        if is_lt(n, S(2)) is True:
            return S.Zero
        try:
            n = int(n)
        except TypeError:
            return
        return S(_primepi(n))


#######################################################################
###
### Functions for enumerating partitions, permutations and combinations
###
#######################################################################


class _MultisetHistogram(tuple):
    __slots__ = ()


_N = -1
_ITEMS = -2
_M = slice(None, _ITEMS)


def _multiset_histogram(n):
    """Return tuple used in permutation and combination counting. Input
    is a dictionary giving items with counts as values or a sequence of
    items (which need not be sorted).

    The data is stored in a class deriving from tuple so it is easily
    recognized and so it can be converted easily to a list.
    """
    if isinstance(n, dict):  # item: count
        if not all(isinstance(v, int) and v >= 0 for v in n.values()):
            raise ValueError
        tot = sum(n.values())
        items = sum(1 for k in n if n[k] > 0)
        return _MultisetHistogram([n[k] for k in n if n[k] > 0] + [items, tot])
    else:
        n = list(n)
        s = set(n)
        lens = len(s)
        lenn = len(n)
        if lens == lenn:
            n = [1]*lenn + [lenn, lenn]
            return _MultisetHistogram(n)
        m = dict(zip(s, range(lens)))
        d = dict(zip(range(lens), (0,)*lens))
        for i in n:
            d[m[i]] += 1
        return _multiset_histogram(d)


def nP(n, k=None, replacement=False):
    """Return the number of permutations of ``n`` items taken ``k`` at a time.

    Possible values for ``n``:

        integer - set of length ``n``

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    If ``k`` is None then the total of all permutations of length 0
    through the number of items represented by ``n`` will be returned.

    If ``replacement`` is True then a given item can appear more than once
    in the ``k`` items. (For example, for 'ab' permutations of 2 would
    include 'aa', 'ab', 'ba' and 'bb'.) The multiplicity of elements in
    ``n`` is ignored when ``replacement`` is True but the total number
    of elements is considered since no element can appear more times than
    the number of elements in ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nP
    >>> from sympy.utilities.iterables import multiset_permutations, multiset
    >>> nP(3, 2)
    6
    >>> nP('abc', 2) == nP(multiset('abc'), 2) == 6
    True
    >>> nP('aab', 2)
    3
    >>> nP([1, 2, 2], 2)
    3
    >>> [nP(3, i) for i in range(4)]
    [1, 3, 6, 6]
    >>> nP(3) == sum(_)
    True

    When ``replacement`` is True, each item can have multiplicity
    equal to the length represented by ``n``:

    >>> nP('aabc', replacement=True)
    121
    >>> [len(list(multiset_permutations('aaaabbbbcccc', i))) for i in range(5)]
    [1, 3, 9, 27, 81]
    >>> sum(_)
    121

    See Also
    ========
    sympy.utilities.iterables.multiset_permutations

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Permutation

    """
    try:
        n = as_int(n)
    except ValueError:
        return Integer(_nP(_multiset_histogram(n), k, replacement))
    return Integer(_nP(n, k, replacement))


@cacheit
def _nP(n, k=None, replacement=False):

    if k == 0:
        return 1
    if isinstance(n, SYMPY_INTS):  # n different items
        # assert n >= 0
        if k is None:
            return sum(_nP(n, i, replacement) for i in range(n + 1))
        elif replacement:
            return n**k
        elif k > n:
            return 0
        elif k == n:
            return factorial(k)
        elif k == 1:
            return n
        else:
            # assert k >= 0
            return _product(n - k + 1, n)
    elif isinstance(n, _MultisetHistogram):
        if k is None:
            return sum(_nP(n, i, replacement) for i in range(n[_N] + 1))
        elif replacement:
            return n[_ITEMS]**k
        elif k == n[_N]:
            return factorial(k)/prod([factorial(i) for i in n[_M] if i > 1])
        elif k > n[_N]:
            return 0
        elif k == 1:
            return n[_ITEMS]
        else:
            # assert k >= 0
            tot = 0
            n = list(n)
            for i in range(len(n[_M])):
                if not n[i]:
                    continue
                n[_N] -= 1
                if n[i] == 1:
                    n[i] = 0
                    n[_ITEMS] -= 1
                    tot += _nP(_MultisetHistogram(n), k - 1)
                    n[_ITEMS] += 1
                    n[i] = 1
                else:
                    n[i] -= 1
                    tot += _nP(_MultisetHistogram(n), k - 1)
                    n[i] += 1
                n[_N] += 1
            return tot


@cacheit
def _AOP_product(n):
    """for n = (m1, m2, .., mk) return the coefficients of the polynomial,
    prod(sum(x**i for i in range(nj + 1)) for nj in n); i.e. the coefficients
    of the product of AOPs (all-one polynomials) or order given in n.  The
    resulting coefficient corresponding to x**r is the number of r-length
    combinations of sum(n) elements with multiplicities given in n.
    The coefficients are given as a default dictionary (so if a query is made
    for a key that is not present, 0 will be returned).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import _AOP_product
    >>> from sympy.abc import x
    >>> n = (2, 2, 3)  # e.g. aabbccc
    >>> prod = ((x**2 + x + 1)*(x**2 + x + 1)*(x**3 + x**2 + x + 1)).expand()
    >>> c = _AOP_product(n); dict(c)
    {0: 1, 1: 3, 2: 6, 3: 8, 4: 8, 5: 6, 6: 3, 7: 1}
    >>> [c[i] for i in range(8)] == [prod.coeff(x, i) for i in range(8)]
    True

    The generating poly used here is the same as that listed in
    https://tinyurl.com/cep849r, but in a refactored form.

    """

    n = list(n)
    ord = sum(n)
    need = (ord + 2)//2
    rv = [1]*(n.pop() + 1)
    rv.extend((0,) * (need - len(rv)))
    rv = rv[:need]
    while n:
        ni = n.pop()
        N = ni + 1
        was = rv[:]
        for i in range(1, min(N, len(rv))):
            rv[i] += rv[i - 1]
        for i in range(N, need):
            rv[i] += rv[i - 1] - was[i - N]
    rev = list(reversed(rv))
    if ord % 2:
        rv = rv + rev
    else:
        rv[-1:] = rev
    d = defaultdict(int)
    for i, r in enumerate(rv):
        d[i] = r
    return d


def nC(n, k=None, replacement=False):
    """Return the number of combinations of ``n`` items taken ``k`` at a time.

    Possible values for ``n``:

        integer - set of length ``n``

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    If ``k`` is None then the total of all combinations of length 0
    through the number of items represented in ``n`` will be returned.

    If ``replacement`` is True then a given item can appear more than once
    in the ``k`` items. (For example, for 'ab' sets of 2 would include 'aa',
    'ab', and 'bb'.) The multiplicity of elements in ``n`` is ignored when
    ``replacement`` is True but the total number of elements is considered
    since no element can appear more times than the number of elements in
    ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nC
    >>> from sympy.utilities.iterables import multiset_combinations
    >>> nC(3, 2)
    3
    >>> nC('abc', 2)
    3
    >>> nC('aab', 2)
    2

    When ``replacement`` is True, each item can have multiplicity
    equal to the length represented by ``n``:

    >>> nC('aabc', replacement=True)
    35
    >>> [len(list(multiset_combinations('aaaabbbbcccc', i))) for i in range(5)]
    [1, 3, 6, 10, 15]
    >>> sum(_)
    35

    If there are ``k`` items with multiplicities ``m_1, m_2, ..., m_k``
    then the total of all combinations of length 0 through ``k`` is the
    product, ``(m_1 + 1)*(m_2 + 1)*...*(m_k + 1)``. When the multiplicity
    of each item is 1 (i.e., k unique items) then there are 2**k
    combinations. For example, if there are 4 unique items, the total number
    of combinations is 16:

    >>> sum(nC(4, i) for i in range(5))
    16

    See Also
    ========

    sympy.utilities.iterables.multiset_combinations

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Combination
    .. [2] https://tinyurl.com/cep849r

    """

    if isinstance(n, SYMPY_INTS):
        if k is None:
            if not replacement:
                return 2**n
            return sum(nC(n, i, replacement) for i in range(n + 1))
        if k < 0:
            raise ValueError("k cannot be negative")
        if replacement:
            return binomial(n + k - 1, k)
        return binomial(n, k)
    if isinstance(n, _MultisetHistogram):
        N = n[_N]
        if k is None:
            if not replacement:
                return prod(m + 1 for m in n[_M])
            return sum(nC(n, i, replacement) for i in range(N + 1))
        elif replacement:
            return nC(n[_ITEMS], k, replacement)
        # assert k >= 0
        elif k in (1, N - 1):
            return n[_ITEMS]
        elif k in (0, N):
            return 1
        return _AOP_product(tuple(n[_M]))[k]
    else:
        return nC(_multiset_histogram(n), k, replacement)


def _eval_stirling1(n, k):
    if n == k == 0:
        return S.One
    if 0 in (n, k):
        return S.Zero

    # some special values
    if n == k:
        return S.One
    elif k == n - 1:
        return binomial(n, 2)
    elif k == n - 2:
        return (3*n - 1)*binomial(n, 3)/4
    elif k == n - 3:
        return binomial(n, 2)*binomial(n, 4)

    return _stirling1(n, k)


@cacheit
def _stirling1(n, k):
    row = [0, 1]+[0]*(k-1) # for n = 1
    for i in range(2, n+1):
        for j in range(min(k,i), 0, -1):
            row[j] = (i-1) * row[j] + row[j-1]
    return Integer(row[k])


def _eval_stirling2(n, k):
    if n == k == 0:
        return S.One
    if 0 in (n, k):
        return S.Zero

    # some special values
    if n == k:
        return S.One
    elif k == n - 1:
        return binomial(n, 2)
    elif k == 1:
        return S.One
    elif k == 2:
        return Integer(2**(n - 1) - 1)

    return _stirling2(n, k)


@cacheit
def _stirling2(n, k):
    row = [0, 1]+[0]*(k-1) # for n = 1
    for i in range(2, n+1):
        for j in range(min(k,i), 0, -1):
            row[j] = j * row[j] + row[j-1]
    return Integer(row[k])


def stirling(n, k, d=None, kind=2, signed=False):
    r"""Return Stirling number $S(n, k)$ of the first or second (default) kind.

    The sum of all Stirling numbers of the second kind for $k = 1$
    through $n$ is ``bell(n)``. The recurrence relationship for these numbers
    is:

    .. math :: {0 \brace 0} = 1; {n \brace 0} = {0 \brace k} = 0;

    .. math :: {{n+1} \brace k} = j {n \brace k} + {n \brace {k-1}}

    where $j$ is:
        $n$ for Stirling numbers of the first kind,
        $-n$ for signed Stirling numbers of the first kind,
        $k$ for Stirling numbers of the second kind.

    The first kind of Stirling number counts the number of permutations of
    ``n`` distinct items that have ``k`` cycles; the second kind counts the
    ways in which ``n`` distinct items can be partitioned into ``k`` parts.
    If ``d`` is given, the "reduced Stirling number of the second kind" is
    returned: $S^{d}(n, k) = S(n - d + 1, k - d + 1)$ with $n \ge k \ge d$.
    (This counts the ways to partition $n$ consecutive integers into $k$
    groups with no pairwise difference less than $d$. See example below.)

    To obtain the signed Stirling numbers of the first kind, use keyword
    ``signed=True``. Using this keyword automatically sets ``kind`` to 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import stirling, bell
    >>> from sympy.combinatorics import Permutation
    >>> from sympy.utilities.iterables import multiset_partitions, permutations

    First kind (unsigned by default):

    >>> [stirling(6, i, kind=1) for i in range(7)]
    [0, 120, 274, 225, 85, 15, 1]
    >>> perms = list(permutations(range(4)))
    >>> [sum(Permutation(p).cycles == i for p in perms) for i in range(5)]
    [0, 6, 11, 6, 1]
    >>> [stirling(4, i, kind=1) for i in range(5)]
    [0, 6, 11, 6, 1]

    First kind (signed):

    >>> [stirling(4, i, signed=True) for i in range(5)]
    [0, -6, 11, -6, 1]

    Second kind:

    >>> [stirling(10, i) for i in range(12)]
    [0, 1, 511, 9330, 34105, 42525, 22827, 5880, 750, 45, 1, 0]
    >>> sum(_) == bell(10)
    True
    >>> len(list(multiset_partitions(range(4), 2))) == stirling(4, 2)
    True

    Reduced second kind:

    >>> from sympy import subsets, oo
    >>> def delta(p):
    ...    if len(p) == 1:
    ...        return oo
    ...    return min(abs(i[0] - i[1]) for i in subsets(p, 2))
    >>> parts = multiset_partitions(range(5), 3)
    >>> d = 2
    >>> sum(1 for p in parts if all(delta(i) >= d for i in p))
    7
    >>> stirling(5, 3, 2)
    7

    See Also
    ========
    sympy.utilities.iterables.multiset_partitions


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind
    .. [2] https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind

    """
    # TODO: make this a class like bell()

    n = as_int(n)
    k = as_int(k)
    if n < 0:
        raise ValueError('n must be nonnegative')
    if k > n:
        return S.Zero
    if d:
        # assert k >= d
        # kind is ignored -- only kind=2 is supported
        return _eval_stirling2(n - d + 1, k - d + 1)
    elif signed:
        # kind is ignored -- only kind=1 is supported
        return S.NegativeOne**(n - k)*_eval_stirling1(n, k)

    if kind == 1:
        return _eval_stirling1(n, k)
    elif kind == 2:
        return _eval_stirling2(n, k)
    else:
        raise ValueError('kind must be 1 or 2, not %s' % k)


@cacheit
def _nT(n, k):
    """Return the partitions of ``n`` items into ``k`` parts. This
    is used by ``nT`` for the case when ``n`` is an integer."""
    # really quick exits
    if k > n or k < 0:
        return 0
    if k in (1, n):
        return 1
    if k == 0:
        return 0
    # exits that could be done below but this is quicker
    if k == 2:
        return n//2
    d = n - k
    if d <= 3:
        return d
    # quick exit
    if 3*k >= n:  # or, equivalently, 2*k >= d
        # all the information needed in this case
        # will be in the cache needed to calculate
        # partition(d), so...
        # update cache
        tot = _partition_rec(d)
        # and correct for values not needed
        if d - k > 0:
            tot -= sum(_partition_rec.fetch_item(slice(d - k)))
        return tot
    # regular exit
    # nT(n, k) = Sum(nT(n - k, m), (m, 1, k));
    # calculate needed nT(i, j) values
    p = [1]*d
    for i in range(2, k + 1):
        for m  in range(i + 1, d):
            p[m] += p[m - i]
        d -= 1
    # if p[0] were appended to the end of p then the last
    # k values of p are the nT(n, j) values for 0 < j < k in reverse
    # order p[-1] = nT(n, 1), p[-2] = nT(n, 2), etc.... Instead of
    # putting the 1 from p[0] there, however, it is simply added to
    # the sum below which is valid for 1 < k <= n//2
    return (1 + sum(p[1 - k:]))


def nT(n, k=None):
    """Return the number of ``k``-sized partitions of ``n`` items.

    Possible values for ``n``:

        integer - ``n`` identical items

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    Note: the convention for ``nT`` is different than that of ``nC`` and
    ``nP`` in that
    here an integer indicates ``n`` *identical* items instead of a set of
    length ``n``; this is in keeping with the ``partitions`` function which
    treats its integer-``n`` input like a list of ``n`` 1s. One can use
    ``range(n)`` for ``n`` to indicate ``n`` distinct items.

    If ``k`` is None then the total number of ways to partition the elements
    represented in ``n`` will be returned.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nT

    Partitions of the given multiset:

    >>> [nT('aabbc', i) for i in range(1, 7)]
    [1, 8, 11, 5, 1, 0]
    >>> nT('aabbc') == sum(_)
    True

    >>> [nT("mississippi", i) for i in range(1, 12)]
    [1, 74, 609, 1521, 1768, 1224, 579, 197, 50, 9, 1]

    Partitions when all items are identical:

    >>> [nT(5, i) for i in range(1, 6)]
    [1, 2, 2, 1, 1]
    >>> nT('1'*5) == sum(_)
    True

    When all items are different:

    >>> [nT(range(5), i) for i in range(1, 6)]
    [1, 15, 25, 10, 1]
    >>> nT(range(5)) == sum(_)
    True

    Partitions of an integer expressed as a sum of positive integers:

    >>> from sympy import partition
    >>> partition(4)
    5
    >>> nT(4, 1) + nT(4, 2) + nT(4, 3) + nT(4, 4)
    5
    >>> nT('1'*4)
    5

    See Also
    ========
    sympy.utilities.iterables.partitions
    sympy.utilities.iterables.multiset_partitions
    sympy.functions.combinatorial.numbers.partition

    References
    ==========

    .. [1] https://web.archive.org/web/20210507012732/https://teaching.csse.uwa.edu.au/units/CITS7209/partition.pdf

    """

    if isinstance(n, SYMPY_INTS):
        # n identical items
        if k is None:
            return partition(n)
        if isinstance(k, SYMPY_INTS):
            n = as_int(n)
            k = as_int(k)
            return Integer(_nT(n, k))
    if not isinstance(n, _MultisetHistogram):
        try:
            # if n contains hashable items there is some
            # quick handling that can be done
            u = len(set(n))
            if u <= 1:
                return nT(len(n), k)
            elif u == len(n):
                n = range(u)
            raise TypeError
        except TypeError:
            n = _multiset_histogram(n)
    N = n[_N]
    if k is None and N == 1:
        return 1
    if k in (1, N):
        return 1
    if k == 2 or N == 2 and k is None:
        m, r = divmod(N, 2)
        rv = sum(nC(n, i) for i in range(1, m + 1))
        if not r:
            rv -= nC(n, m)//2
        if k is None:
            rv += 1  # for k == 1
        return rv
    if N == n[_ITEMS]:
        # all distinct
        if k is None:
            return bell(N)
        return stirling(N, k)
    m = MultisetPartitionTraverser()
    if k is None:
        return m.count_partitions(n[_M])
    # MultisetPartitionTraverser does not have a range-limited count
    # method, so need to enumerate and count
    tot = 0
    for discard in m.enum_range(n[_M], k-1, k):
        tot += 1
    return tot


#-----------------------------------------------------------------------------#
#                                                                             #
#                          Motzkin numbers                                    #
#                                                                             #
#-----------------------------------------------------------------------------#


class motzkin(DefinedFunction):
    """
    The nth Motzkin number is the number
    of ways of drawing non-intersecting chords
    between n points on a circle (not necessarily touching
    every point by a chord). The Motzkin numbers are named
    after Theodore Motzkin and have diverse applications
    in geometry, combinatorics and number theory.

    Motzkin numbers are the integer sequence defined by the
    initial terms `M_0 = 1`, `M_1 = 1` and the two-term recurrence relation
    `M_n = \frac{2*n + 1}{n + 2} * M_{n-1} + \frac{3n - 3}{n + 2} * M_{n-2}`.


    Examples
    ========

    >>> from sympy import motzkin

    >>> motzkin.is_motzkin(5)
    False
    >>> motzkin.find_motzkin_numbers_in_range(2,300)
    [2, 4, 9, 21, 51, 127]
    >>> motzkin.find_motzkin_numbers_in_range(2,900)
    [2, 4, 9, 21, 51, 127, 323, 835]
    >>> motzkin.find_first_n_motzkins(10)
    [1, 1, 2, 4, 9, 21, 51, 127, 323, 835]


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Motzkin_number
    .. [2] https://mathworld.wolfram.com/MotzkinNumber.html

    """

    @staticmethod
    def is_motzkin(n):
        try:
            n = as_int(n)
        except ValueError:
            return False
        if n > 0:
            if n in (1, 2):
                return True

            tn1 = 1
            tn = 2
            i = 3
            while tn < n:
                a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
                i += 1
                tn1 = tn
                tn = a

            if tn == n:
                return True
            else:
                return False

        else:
            return False

    @staticmethod
    def find_motzkin_numbers_in_range(x, y):
        if 0 <= x <= y:
            motzkins = []
            if x <= 1 <= y:
                motzkins.append(1)
            tn1 = 1
            tn = 2
            i = 3
            while tn <= y:
                if tn >= x:
                    motzkins.append(tn)
                a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
                i += 1
                tn1 = tn
                tn = int(a)

            return motzkins

        else:
            raise ValueError('The provided range is not valid. This condition should satisfy x <= y')

    @staticmethod
    def find_first_n_motzkins(n):
        try:
            n = as_int(n)
        except ValueError:
            raise ValueError('The provided number must be a positive integer')
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        motzkins = [1]
        if n >= 1:
            motzkins.append(1)
        tn1 = 1
        tn = 2
        i = 3
        while i <= n:
            motzkins.append(tn)
            a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
            i += 1
            tn1 = tn
            tn = int(a)

        return motzkins

    @staticmethod
    @recurrence_memo([S.One, S.One])
    def _motzkin(n, prev):
        return ((2*n + 1)*prev[-1] + (3*n - 3)*prev[-2]) // (n + 2)

    @classmethod
    def eval(cls, n):
        try:
            n = as_int(n)
        except ValueError:
            raise ValueError('The provided number must be a positive integer')
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        return Integer(cls._motzkin(n - 1))


def nD(i=None, brute=None, *, n=None, m=None):
    """return the number of derangements for: ``n`` unique items, ``i``
    items (as a sequence or multiset), or multiplicities, ``m`` given
    as a sequence or multiset.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements as enum
    >>> from sympy.functions.combinatorial.numbers import nD

    A derangement ``d`` of sequence ``s`` has all ``d[i] != s[i]``:

    >>> set([''.join(i) for i in enum('abc')])
    {'bca', 'cab'}
    >>> nD('abc')
    2

    Input as iterable or dictionary (multiset form) is accepted:

    >>> assert nD([1, 2, 2, 3, 3, 3]) == nD({1: 1, 2: 2, 3: 3})

    By default, a brute-force enumeration and count of multiset permutations
    is only done if there are fewer than 9 elements. There may be cases when
    there is high multiplicity with few unique elements that will benefit
    from a brute-force enumeration, too. For this reason, the `brute`
    keyword (default None) is provided. When False, the brute-force
    enumeration will never be used. When True, it will always be used.

    >>> nD('1111222233', brute=True)
    44

    For convenience, one may specify ``n`` distinct items using the
    ``n`` keyword:

    >>> assert nD(n=3) == nD('abc') == 2

    Since the number of derangments depends on the multiplicity of the
    elements and not the elements themselves, it may be more convenient
    to give a list or multiset of multiplicities using keyword ``m``:

    >>> assert nD('abc') == nD(m=(1,1,1)) == nD(m={1:3}) == 2

    """
    from sympy.integrals.integrals import integrate
    from sympy.functions.special.polynomials import laguerre
    from sympy.abc import x
    def ok(x):
        if not isinstance(x, SYMPY_INTS):
            raise TypeError('expecting integer values')
        if x < 0:
            raise ValueError('value must not be negative')
        return True

    if (i, n, m).count(None) != 2:
        raise ValueError('enter only 1 of i, n, or m')
    if i is not None:
        if isinstance(i, SYMPY_INTS):
            raise TypeError('items must be a list or dictionary')
        if not i:
            return S.Zero
        if type(i) is not dict:
            s = list(i)
            ms = multiset(s)
        elif type(i) is dict:
            all(ok(_) for _ in i.values())
            ms = {k: v for k, v in i.items() if v}
            s = None
        if not ms:
            return S.Zero
        N = sum(ms.values())
        counts = multiset(ms.values())
        nkey = len(ms)
    elif n is not None:
        ok(n)
        if not n:
            return S.Zero
        return subfactorial(n)
    elif m is not None:
        if isinstance(m, dict):
            all(ok(i) and ok(j) for i, j in m.items())
            counts = {k: v for k, v in m.items() if k*v}
        elif iterable(m) or isinstance(m, str):
            m = list(m)
            all(ok(i) for i in m)
            counts = multiset([i for i in m if i])
        else:
            raise TypeError('expecting iterable')
        if not counts:
            return S.Zero
        N = sum(k*v for k, v in counts.items())
        nkey = sum(counts.values())
        s = None
    big = int(max(counts))
    if big == 1:  # no repetition
        return subfactorial(nkey)
    nval = len(counts)
    if big*2 > N:
        return S.Zero
    if big*2 == N:
        if nkey == 2 and nval == 1:
            return S.One  # aaabbb
        if nkey - 1 == big:  # one element repeated
            return factorial(big)  # e.g. abc part of abcddd
    if N < 9 and brute is None or brute:
        # for all possibilities, this was found to be faster
        if s is None:
            s = []
            i = 0
            for m, v in counts.items():
                for j in range(v):
                    s.extend([i]*m)
                    i += 1
        return Integer(sum(1 for i in multiset_derangements(s)))
    from sympy.functions.elementary.exponential import exp
    return Integer(abs(integrate(exp(-x)*Mul(*[
        laguerre(i, x)**m for i, m in counts.items()]), (x, 0, oo))))
