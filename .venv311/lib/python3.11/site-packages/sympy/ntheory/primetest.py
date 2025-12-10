"""
Primality testing

"""

from itertools import count

from sympy.core.sympify import sympify
from sympy.external.gmpy import (gmpy as _gmpy, gcd, jacobi,
                                 is_square as gmpy_is_square,
                                 bit_scan1, is_fermat_prp, is_euler_prp,
                                 is_selfridge_prp, is_strong_selfridge_prp,
                                 is_strong_bpsw_prp)
from sympy.external.ntheory import _lucas_sequence
from sympy.utilities.misc import as_int, filldedent

# Note: This list should be updated whenever new Mersenne primes are found.
# Refer: https://www.mersenne.org/
MERSENNE_PRIME_EXPONENTS = (2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203,
 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049,
 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583,
 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933,
 136279841)


def is_fermat_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{n-1} \equiv 1 \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Fermat pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_fermat_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_fermat_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    341
    561
    645

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fermat_pseudoprime
    """
    n, a = as_int(n), as_int(a)
    if a == 1:
        return n == 2 or bool(n % 2)
    return is_fermat_prp(n, a)


def is_euler_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{(n-1)/2} \equiv \pm 1 \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Euler pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_euler_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_euler_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    341
    561

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler_pseudoprime
    """
    n, a = as_int(n), as_int(a)
    if a < 1:
        raise ValueError("a should be an integer greater than 0")
    if n < 1:
        raise ValueError("n should be an integer greater than 0")
    if n == 1:
        return False
    if a == 1:
        return n == 2 or bool(n % 2)  # (prime or odd composite)
    if n % 2 == 0:
        return n == 2
    if gcd(n, a) != 1:
        raise ValueError("The two numbers should be relatively prime")
    return pow(a, (n - 1) // 2, n) in [1, n - 1]


def is_euler_jacobi_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{(n-1)/2} \equiv \left(\frac{a}{n}\right) \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Euler-Jacobi pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_euler_jacobi_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_euler_jacobi_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    561

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Jacobi_pseudoprime
    """
    n, a = as_int(n), as_int(a)
    if a == 1:
        return n == 2 or bool(n % 2)
    return is_euler_prp(n, a)


def is_square(n, prep=True):
    """Return True if n == a * a for some integer a, else False.
    If n is suspected of *not* being a square then this is a
    quick method of confirming that it is not.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_square
    >>> is_square(25)
    True
    >>> is_square(2)
    False

    References
    ==========

    .. [1]  https://mersenneforum.org/showpost.php?p=110896

    See Also
    ========
    sympy.core.intfunc.isqrt
    """
    if prep:
        n = as_int(n)
        if n < 0:
            return False
        if n in (0, 1):
            return True
    return gmpy_is_square(n)


def _test(n, base, s, t):
    """Miller-Rabin strong pseudoprime test for one base.
    Return False if n is definitely composite, True if n is
    probably prime, with a probability greater than 3/4.

    """
    # do the Fermat test
    b = pow(base, t, n)
    if b == 1 or b == n - 1:
        return True
    for _ in range(s - 1):
        b = pow(b, 2, n)
        if b == n - 1:
            return True
        # see I. Niven et al. "An Introduction to Theory of Numbers", page 78
        if b == 1:
            return False
    return False


def mr(n, bases):
    """Perform a Miller-Rabin strong pseudoprime test on n using a
    given list of bases/witnesses.

    References
    ==========

    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 135-138

    A list of thresholds and the bases they require are here:
    https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Deterministic_variants

    Examples
    ========

    >>> from sympy.ntheory.primetest import mr
    >>> mr(1373651, [2, 3])
    False
    >>> mr(479001599, [31, 73])
    True

    """
    from sympy.polys.domains import ZZ

    n = as_int(n)
    if n < 2 or (n > 2 and n % 2 == 0):
        return False
    # remove powers of 2 from n-1 (= t * 2**s)
    s = bit_scan1(n - 1)
    t = n >> s
    for base in bases:
        # Bases >= n are wrapped, bases < 2 are invalid
        if base >= n:
            base %= n
        if base >= 2:
            base = ZZ(base)
            if not _test(n, base, s, t):
                return False
    return True


def _lucas_extrastrong_params(n):
    """Calculates the "extra strong" parameters (D, P, Q) for n.

    Parameters
    ==========

    n : int
        positive odd integer

    Returns
    =======

    D, P, Q: "extra strong" parameters.
             ``(0, 0, 0)`` if we find a nontrivial divisor of ``n``.

    Examples
    ========

    >>> from sympy.ntheory.primetest import _lucas_extrastrong_params
    >>> _lucas_extrastrong_params(101)
    (12, 4, 1)
    >>> _lucas_extrastrong_params(15)
    (0, 0, 0)

    References
    ==========
    .. [1] OEIS A217719: Extra Strong Lucas Pseudoprimes
           https://oeis.org/A217719
    .. [2] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    """
    for P in count(3):
        D = P**2 - 4
        j = jacobi(D, n)
        if j == -1:
            return (D, P, 1)
        elif j == 0 and D % n:
            return (0, 0, 0)


def is_lucas_prp(n):
    """Standard Lucas compositeness test with Selfridge parameters.  Returns
    False if n is definitely composite, and True if n is a Lucas probable
    prime.

    This is typically used in combination with the Miller-Rabin test.

    References
    ==========
    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    .. [2] OEIS A217120: Lucas Pseudoprimes
           https://oeis.org/A217120
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_lucas_prp
    >>> for i in range(10000):
    ...     if is_lucas_prp(i) and not isprime(i):
    ...         print(i)
    323
    377
    1159
    1829
    3827
    5459
    5777
    9071
    9179
    """
    n = as_int(n)
    if n < 2:
        return False
    return is_selfridge_prp(n)


def is_strong_lucas_prp(n):
    """Strong Lucas compositeness test with Selfridge parameters.  Returns
    False if n is definitely composite, and True if n is a strong Lucas
    probable prime.

    This is often used in combination with the Miller-Rabin test, and
    in particular, when combined with M-R base 2 creates the strong BPSW test.

    References
    ==========
    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    .. [2] OEIS A217255: Strong Lucas Pseudoprimes
           https://oeis.org/A217255
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime
    .. [4] https://en.wikipedia.org/wiki/Baillie-PSW_primality_test

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_strong_lucas_prp
    >>> for i in range(20000):
    ...     if is_strong_lucas_prp(i) and not isprime(i):
    ...        print(i)
    5459
    5777
    10877
    16109
    18971
    """
    n = as_int(n)
    if n < 2:
        return False
    return is_strong_selfridge_prp(n)


def is_extra_strong_lucas_prp(n):
    """Extra Strong Lucas compositeness test.  Returns False if n is
    definitely composite, and True if n is an "extra strong" Lucas probable
    prime.

    The parameters are selected using P = 3, Q = 1, then incrementing P until
    (D|n) == -1.  The test itself is as defined in [1]_, from the
    Mo and Jones preprint.  The parameter selection and test are the same as
    used in OEIS A217719, Perl's Math::Prime::Util, and the Lucas pseudoprime
    page on Wikipedia.

    It is 20-50% faster than the strong test.

    Because of the different parameters selected, there is no relationship
    between the strong Lucas pseudoprimes and extra strong Lucas pseudoprimes.
    In particular, one is not a subset of the other.

    References
    ==========
    .. [1] Jon Grantham, Frobenius Pseudoprimes,
           Math. Comp. Vol 70, Number 234 (2001), pp. 873-891,
           https://doi.org/10.1090%2FS0025-5718-00-01197-2
    .. [2] OEIS A217719: Extra Strong Lucas Pseudoprimes
           https://oeis.org/A217719
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_extra_strong_lucas_prp
    >>> for i in range(20000):
    ...     if is_extra_strong_lucas_prp(i) and not isprime(i):
    ...        print(i)
    989
    3239
    5777
    10877
    """
    # Implementation notes:
    #   1) the parameters differ from Thomas R. Nicely's.  His parameter
    #      selection leads to pseudoprimes that overlap M-R tests, and
    #      contradict Baillie and Wagstaff's suggestion of (D|n) = -1.
    #   2) The MathWorld page as of June 2013 specifies Q=-1.  The Lucas
    #      sequence must have Q=1.  See Grantham theorem 2.3, any of the
    #      references on the MathWorld page, or run it and see Q=-1 is wrong.
    n = as_int(n)
    if n == 2:
        return True
    if n < 2 or (n % 2) == 0:
        return False
    if gmpy_is_square(n):
        return False

    D, P, Q = _lucas_extrastrong_params(n)
    if D == 0:
        return False

    # remove powers of 2 from n+1 (= k * 2**s)
    s = bit_scan1(n + 1)
    k = (n + 1) >> s

    U, V, _ = _lucas_sequence(n, P, Q, k)

    if U == 0 and (V == 2 or V == n - 2):
        return True
    for _ in range(1, s):
        if V == 0:
            return True
        V = (V*V - 2) % n
    return False


def proth_test(n):
    r""" Test if the Proth number `n = k2^m + 1` is prime. where k is a positive odd number and `2^m > k`.

    Parameters
    ==========

    n : Integer
        ``n`` is Proth number

    Returns
    =======

    bool : If ``True``, then ``n`` is the Proth prime

    Raises
    ======

    ValueError
        If ``n`` is not Proth number.

    Examples
    ========

    >>> from sympy.ntheory.primetest import proth_test
    >>> proth_test(41)
    True
    >>> proth_test(57)
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Proth_prime

    """
    n = as_int(n)
    if n < 3:
        raise ValueError("n is not Proth number")
    m = bit_scan1(n - 1)
    k = n >> m
    if m < k.bit_length():
        raise ValueError("n is not Proth number")
    if n % 3 == 0:
        return n == 3
    if k % 3: # n % 12 == 5
        return pow(3, n >> 1, n) == n - 1
    # If `n` is a square number, then `jacobi(a, n) = 1` for any `a`
    if gmpy_is_square(n):
        return False
    # `a` may be chosen at random.
    # In any case, we want to find `a` such that `jacobi(a, n) = -1`.
    for a in range(5, n):
        j = jacobi(a, n)
        if j == -1:
            return pow(a, n >> 1, n) == n - 1
        if j == 0:
            return False


def _lucas_lehmer_primality_test(p):
    r""" Test if the Mersenne number `M_p = 2^p-1` is prime.

    Parameters
    ==========

    p : int
        ``p`` is an odd prime number

    Returns
    =======

    bool : If ``True``, then `M_p` is the Mersenne prime

    Examples
    ========

    >>> from sympy.ntheory.primetest import _lucas_lehmer_primality_test
    >>> _lucas_lehmer_primality_test(5) # 2**5 - 1 = 31 is prime
    True
    >>> _lucas_lehmer_primality_test(11) # 2**11 - 1 = 2047 is not prime
    False

    See Also
    ========

    is_mersenne_prime

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas%E2%80%93Lehmer_primality_test

    """
    v = 4
    m = 2**p - 1
    for _ in range(p - 2):
        v = pow(v, 2, m) - 2
    return v == 0


def is_mersenne_prime(n):
    """Returns True if  ``n`` is a Mersenne prime, else False.

    A Mersenne prime is a prime number having the form `2^i - 1`.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import is_mersenne_prime
    >>> is_mersenne_prime(6)
    False
    >>> is_mersenne_prime(127)
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/MersennePrime.html

    """
    n = as_int(n)
    if n < 1:
        return False
    if n & (n + 1):
        # n is not Mersenne number
        return False
    p = n.bit_length()
    if p in MERSENNE_PRIME_EXPONENTS:
        return True
    if p < 65_000_000 or not isprime(p):
        # According to GIMPS, verification was completed on September 19, 2023 for p less than 65 million.
        # https://www.mersenne.org/report_milestones/
        # If p is composite number, then n=2**p-1 is composite number.
        return False
    result = _lucas_lehmer_primality_test(p)
    if result:
        raise ValueError(filldedent('''
            This Mersenne Prime, 2^%s - 1, should
            be added to SymPy's known values.''' % p))
    return result


_MR_BASES_32 = [15591, 2018, 166, 7429, 8064, 16045, 10503, 4399, 1949, 1295,
                2776, 3620, 560, 3128, 5212, 2657, 2300, 2021, 4652, 1471,
                9336, 4018, 2398, 20462, 10277, 8028, 2213, 6219, 620, 3763,
                4852, 5012, 3185, 1333, 6227,5298, 1074, 2391, 5113, 7061,
                803, 1269, 3875, 422, 751, 580, 4729, 10239, 746, 2951, 556,
                2206, 3778, 481, 1522, 3476, 481, 2487, 3266, 5633, 488, 3373,
                6441, 3344, 17, 15105, 1490, 4154, 2036, 1882, 1813, 467,
                3307, 14042, 6371, 658, 1005, 903, 737, 1887, 7447, 1888,
                2848, 1784, 7559, 3400, 951, 13969, 4304, 177, 41, 19875,
                3110, 13221, 8726, 571, 7043, 6943, 1199, 352, 6435, 165,
                1169, 3315, 978, 233, 3003, 2562, 2994, 10587, 10030, 2377,
                1902, 5354, 4447, 1555, 263, 27027, 2283, 305, 669, 1912, 601,
                6186, 429, 1930, 14873, 1784, 1661, 524, 3577, 236, 2360,
                6146, 2850, 55637, 1753, 4178, 8466, 222, 2579, 2743, 2031,
                2226, 2276, 374, 2132, 813, 23788, 1610, 4422, 5159, 1725,
                3597, 3366, 14336, 579, 165, 1375, 10018, 12616, 9816, 1371,
                536, 1867, 10864, 857, 2206, 5788, 434, 8085, 17618, 727,
                3639, 1595, 4944, 2129, 2029, 8195, 8344, 6232, 9183, 8126,
                1870, 3296, 7455, 8947, 25017, 541, 19115, 368, 566, 5674,
                411, 522, 1027, 8215, 2050, 6544, 10049, 614, 774, 2333, 3007,
                35201, 4706, 1152, 1785, 1028, 1540, 3743, 493, 4474, 2521,
                26845, 8354, 864, 18915, 5465, 2447, 42, 4511, 1660, 166,
                1249, 6259, 2553, 304, 272, 7286, 73, 6554, 899, 2816, 5197,
                13330, 7054, 2818, 3199, 811, 922, 350, 7514, 4452, 3449,
                2663, 4708, 418, 1621, 1171, 3471, 88, 11345, 412, 1559, 194]


def isprime(n):
    """
    Test if n is a prime number (True) or not (False). For n < 2^64 the
    answer is definitive; larger n values have a small probability of actually
    being pseudoprimes.

    Negative numbers (e.g. -2) are not considered prime.

    The first step is looking for trivial factors, which if found enables
    a quick return.  Next, if the sieve is large enough, use bisection search
    on the sieve.  For small numbers, a set of deterministic Miller-Rabin
    tests are performed with bases that are known to have no counterexamples
    in their range.  Finally if the number is larger than 2^64, a strong
    BPSW test is performed.  While this is a probable prime test and we
    believe counterexamples exist, there are no known counterexamples.

    Examples
    ========

    >>> from sympy.ntheory import isprime
    >>> isprime(13)
    True
    >>> isprime(15)
    False

    Notes
    =====

    This routine is intended only for integer input, not numerical
    expressions which may represent numbers. Floats are also
    rejected as input because they represent numbers of limited
    precision. While it is tempting to permit 7.0 to represent an
    integer there are errors that may "pass silently" if this is
    allowed:

    >>> from sympy import Float, S
    >>> int(1e3) == 1e3 == 10**3
    True
    >>> int(1e23) == 1e23
    True
    >>> int(1e23) == 10**23
    False

    >>> near_int = 1 + S(1)/10**19
    >>> near_int == int(near_int)
    False
    >>> n = Float(near_int, 10)  # truncated by precision
    >>> n % 1 == 0
    True
    >>> n = Float(near_int, 20)
    >>> n % 1 == 0
    False

    See Also
    ========

    sympy.ntheory.generate.primerange : Generates all primes in a given range
    sympy.functions.combinatorial.numbers.primepi : Return the number of primes less than or equal to n
    sympy.ntheory.generate.prime : Return the nth prime

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Strong_pseudoprime
    .. [2] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    .. [3] https://en.wikipedia.org/wiki/Baillie-PSW_primality_test
    """
    n = as_int(n)

    # Step 1, do quick composite testing via trial division.  The individual
    # modulo tests benchmark faster than one or two primorial igcds for me.
    # The point here is just to speedily handle small numbers and many
    # composites.  Step 2 only requires that n <= 2 get handled here.
    if n in [2, 3, 5]:
        return True
    if n < 2 or (n % 2) == 0 or (n % 3) == 0 or (n % 5) == 0:
        return False
    if n < 49:
        return True
    if (n %  7) == 0 or (n % 11) == 0 or (n % 13) == 0 or (n % 17) == 0 or \
       (n % 19) == 0 or (n % 23) == 0 or (n % 29) == 0 or (n % 31) == 0 or \
       (n % 37) == 0 or (n % 41) == 0 or (n % 43) == 0 or (n % 47) == 0:
        return False
    if n < 2809:
        return True
    if n < 65077:
        # There are only five Euler pseudoprimes with a least prime factor greater than 47
        return pow(2, n >> 1, n) in [1, n - 1] and n not in [8321, 31621, 42799, 49141, 49981]

    # bisection search on the sieve if the sieve is large enough
    from sympy.ntheory.generate import sieve as s
    if n <= s._list[-1]:
        l, u = s.search(n)
        return l == u
    from sympy.ntheory.factor_ import factor_cache
    if (ret := factor_cache.get(n)) is not None:
        return ret == n

    # If we have GMPY2, skip straight to step 3 and do a strong BPSW test.
    # This should be a bit faster than our step 2, and for large values will
    # be a lot faster than our step 3 (C+GMP vs. Python).
    if _gmpy is not None:
        return is_strong_bpsw_prp(n)


    # Step 2: deterministic Miller-Rabin testing for numbers < 2^64.  See:
    #    https://miller-rabin.appspot.com/
    # for lists.  We have made sure the M-R routine will successfully handle
    # bases larger than n, so we can use the minimal set.
    # In September 2015 deterministic numbers were extended to over 2^81.
    #    https://arxiv.org/pdf/1509.00864.pdf
    #    https://oeis.org/A014233
    if n < 341531:
        return mr(n, [9345883071009581737])
    if n < 4296595241:
        # Michal Forisek and Jakub Jancina,
        # Fast Primality Testing for Integers That Fit into a Machine Word
        # https://ceur-ws.org/Vol-1326/020-Forisek.pdf
        h = ((n >> 16) ^ n) * 0x45d9f3b
        h = ((h >> 16) ^ h) * 0x45d9f3b
        h = ((h >> 16) ^ h) & 255
        return mr(n, [_MR_BASES_32[h]])
    if n < 350269456337:
        return mr(n, [4230279247111683200, 14694767155120705706, 16641139526367750375])
    if n < 55245642489451:
        return mr(n, [2, 141889084524735, 1199124725622454117, 11096072698276303650])
    if n < 7999252175582851:
        return mr(n, [2, 4130806001517, 149795463772692060, 186635894390467037, 3967304179347715805])
    if n < 585226005592931977:
        return mr(n, [2, 123635709730000, 9233062284813009, 43835965440333360, 761179012939631437, 1263739024124850375])
    if n < 18446744073709551616:
        return mr(n, [2, 325, 9375, 28178, 450775, 9780504, 1795265022])
    if n < 318665857834031151167461:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    if n < 3317044064679887385961981:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])

    # We could do this instead at any point:
    #if n < 18446744073709551616:
    #   return mr(n, [2]) and is_extra_strong_lucas_prp(n)

    # Here are tests that are safe for MR routines that don't understand
    # large bases.
    #if n < 9080191:
    #    return mr(n, [31, 73])
    #if n < 19471033:
    #    return mr(n, [2, 299417])
    #if n < 38010307:
    #    return mr(n, [2, 9332593])
    #if n < 316349281:
    #    return mr(n, [11000544, 31481107])
    #if n < 4759123141:
    #    return mr(n, [2, 7, 61])
    #if n < 105936894253:
    #    return mr(n, [2, 1005905886, 1340600841])
    #if n < 31858317218647:
    #    return mr(n, [2, 642735, 553174392, 3046413974])
    #if n < 3071837692357849:
    #    return mr(n, [2, 75088, 642735, 203659041, 3613982119])
    #if n < 18446744073709551616:
    #    return mr(n, [2, 325, 9375, 28178, 450775, 9780504, 1795265022])

    # Step 3: BPSW.
    #
    #  Time for isprime(10**2000 + 4561), no gmpy or gmpy2 installed
    #     44.0s   old isprime using 46 bases
    #      5.3s   strong BPSW + one random base
    #      4.3s   extra strong BPSW + one random base
    #      4.1s   strong BPSW
    #      3.2s   extra strong BPSW

    # Classic BPSW from page 1401 of the paper.  See alternate ideas below.
    return is_strong_bpsw_prp(n)

    # Using extra strong test, which is somewhat faster
    #return mr(n, [2]) and is_extra_strong_lucas_prp(n)

    # Add a random M-R base
    #import random
    #return mr(n, [2, random.randint(3, n-1)]) and is_strong_lucas_prp(n)


def is_gaussian_prime(num):
    r"""Test if num is a Gaussian prime number.

    References
    ==========

    .. [1] https://oeis.org/wiki/Gaussian_primes
    """

    num = sympify(num)
    a, b = num.as_real_imag()
    a = as_int(a, strict=False)
    b = as_int(b, strict=False)
    if a == 0:
        b = abs(b)
        return isprime(b) and b % 4 == 3
    elif b == 0:
        a = abs(a)
        return isprime(a) and a % 4 == 3
    return isprime(a**2 + b**2)
