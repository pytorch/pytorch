from mpmath.libmp import (fzero, from_int, from_rational,
    fone, fhalf, bitcount, to_int, mpf_mul, mpf_div, mpf_sub,
    mpf_add, mpf_sqrt, mpf_pi, mpf_cosh_sinh, mpf_cos, mpf_sin)
from .residue_ntheory import _sqrt_mod_prime_power, is_quad_residue
from sympy.utilities.decorator import deprecated
from sympy.utilities.memoization import recurrence_memo

import math
from itertools import count

def _pre():
    maxn = 10**5
    global _factor, _totient
    _factor = [0]*maxn
    _totient = [1]*maxn
    lim = int(maxn**0.5) + 5
    for i in range(2, lim):
        if _factor[i] == 0:
            for j in range(i*i, maxn, i):
                if _factor[j] == 0:
                    _factor[j] = i
    for i in range(2, maxn):
        if _factor[i] == 0:
            _factor[i] = i
            _totient[i] = i-1
            continue
        x = _factor[i]
        y = i//x
        if y % x == 0:
            _totient[i] = _totient[y]*x
        else:
            _totient[i] = _totient[y]*(x - 1)

def _a(n, k, prec):
    """ Compute the inner sum in HRR formula [1]_

    References
    ==========

    .. [1] https://msp.org/pjm/1956/6-1/pjm-v6-n1-p18-p.pdf

    """
    if k == 1:
        return fone

    k1 = k
    e = 0
    p = _factor[k]
    while k1 % p == 0:
        k1 //= p
        e += 1
    k2 = k//k1 # k2 = p^e
    v = 1 - 24*n
    pi = mpf_pi(prec)

    if k1 == 1:
        # k  = p^e
        if p == 2:
            mod = 8*k
            v = mod + v % mod
            v = (v*pow(9, k - 1, mod)) % mod
            m = _sqrt_mod_prime_power(v, 2, e + 3)[0]
            arg = mpf_div(mpf_mul(
                from_int(4*m), pi, prec), from_int(mod), prec)
            return mpf_mul(mpf_mul(
                from_int((-1)**e*(2 - (m % 4))),
                mpf_sqrt(from_int(k), prec), prec),
                mpf_sin(arg, prec), prec)
        if p == 3:
            mod = 3*k
            v = mod + v % mod
            if e > 1:
                v = (v*pow(64, k//3 - 1, mod)) % mod
            m = _sqrt_mod_prime_power(v, 3, e + 1)[0]
            arg = mpf_div(mpf_mul(from_int(4*m), pi, prec),
                from_int(mod), prec)
            return mpf_mul(mpf_mul(
                from_int(2*(-1)**(e + 1)*(3 - 2*(m % 3))),
                mpf_sqrt(from_int(k//3), prec), prec),
                mpf_sin(arg, prec), prec)
        v = k + v % k
        jacobi3 = -1 if k % 12 in [5, 7] else 1
        if v % p == 0:
            if e == 1:
                return mpf_mul(
                    from_int(jacobi3),
                    mpf_sqrt(from_int(k), prec), prec)
            return fzero
        if not is_quad_residue(v, p):
            return fzero
        _phi = p**(e - 1)*(p - 1)
        v = (v*pow(576, _phi - 1, k))
        m = _sqrt_mod_prime_power(v, p, e)[0]
        arg = mpf_div(
            mpf_mul(from_int(4*m), pi, prec),
            from_int(k), prec)
        return mpf_mul(mpf_mul(
            from_int(2*jacobi3),
            mpf_sqrt(from_int(k), prec), prec),
            mpf_cos(arg, prec), prec)

    if p != 2 or e >= 3:
        d1, d2 = math.gcd(k1, 24), math.gcd(k2, 24)
        e = 24//(d1*d2)
        n1 = ((d2*e*n + (k2**2 - 1)//d1)*
            pow(e*k2*k2*d2, _totient[k1] - 1, k1)) % k1
        n2 = ((d1*e*n + (k1**2 - 1)//d2)*
            pow(e*k1*k1*d1, _totient[k2] - 1, k2)) % k2
        return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)
    if e == 2:
        n1 = ((8*n + 5)*pow(128, _totient[k1] - 1, k1)) % k1
        n2 = (4 + ((n - 2 - (k1**2 - 1)//8)*(k1**2)) % 4) % 4
        return mpf_mul(mpf_mul(
            from_int(-1),
            _a(n1, k1, prec), prec),
            _a(n2, k2, prec))
    n1 = ((8*n + 1)*pow(32, _totient[k1] - 1, k1)) % k1
    n2 = (2 + (n - (k1**2 - 1)//8) % 2) % 2
    return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)

def _d(n, j, prec, sq23pi, sqrt8):
    """
    Compute the sinh term in the outer sum of the HRR formula.
    The constants sqrt(2/3*pi) and sqrt(8) must be precomputed.
    """
    j = from_int(j)
    pi = mpf_pi(prec)
    a = mpf_div(sq23pi, j, prec)
    b = mpf_sub(from_int(n), from_rational(1, 24, prec), prec)
    c = mpf_sqrt(b, prec)
    ch, sh = mpf_cosh_sinh(mpf_mul(a, c), prec)
    D = mpf_div(
        mpf_sqrt(j, prec),
        mpf_mul(mpf_mul(sqrt8, b), pi), prec)
    E = mpf_sub(mpf_mul(a, ch), mpf_div(sh, c, prec), prec)
    return mpf_mul(D, E)


@recurrence_memo([1, 1])
def _partition_rec(n: int, prev) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int
        nonnegative integer

    """
    v = 0
    penta = 0 # pentagonal number: 1, 5, 12, ...
    for i in count():
        penta += 3*i + 1
        np = n - penta
        if np < 0:
            break
        s = prev[np]
        np -= i + 1
        # np = n - gp where gp = generalized pentagonal: 2, 7, 15, ...
        if 0 <= np:
            s += prev[np]
        v += -s if i % 2 else s
    return v


def _partition(n: int) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int

    """
    if n < 0:
        return 0
    if (n <= 200_000 and n - _partition_rec.cache_length() < 70 or
            _partition_rec.cache_length() == 2 and n < 14_400):
        # There will be 2*10**5 elements created here
        # and n elements created by partition, so in case we
        # are going to be working with small n, we just
        # use partition to calculate (and cache) the values
        # since lookup is used there while summation, using
        # _factor and _totient, will be used below. But we
        # only do so if n is relatively close to the length
        # of the cache since doing 1 calculation here is about
        # the same as adding 70 elements to the cache. In addition,
        # the startup here costs about the same as calculating the first
        # 14,400 values via partition, so we delay startup here unless n
        # is smaller than that.
        return _partition_rec(n)
    if '_factor' not in globals():
        _pre()
    # Estimate number of bits in p(n). This formula could be tidied
    pbits = int((
        math.pi*(2*n/3.)**0.5 -
        math.log(4*n))/math.log(10) + 1) * \
        math.log2(10)
    prec = p = int(pbits*1.1 + 100)

    # find the number of terms needed so rounded sum will be accurate
    # using Rademacher's bound M(n, N) for the remainder after a partial
    # sum of N terms (https://arxiv.org/pdf/1205.5991.pdf, (1.8))
    c1 = 44*math.pi**2/(225*math.sqrt(3))
    c2 = math.pi*math.sqrt(2)/75
    c3 = math.pi*math.sqrt(2/3)
    def _M(n, N):
        sqrt = math.sqrt
        return c1/sqrt(N) + c2*sqrt(N/(n - 1))*math.sinh(c3*sqrt(n)/N)
    big = max(9, math.ceil(n**0.5))  # should be too large (for n > 65, ceil should work)
    assert _M(n, big) < 0.5  # else double big until too large
    while big > 40 and _M(n, big) < 0.5:
        big //= 2
    small = big
    big = small*2
    while big - small > 1:
        N = (big + small)//2
        if (er := _M(n, N)) < 0.5:
            big = N
        elif er >= 0.5:
            small = N
    M = big  # done with function M; now have value

    # sanity check for expected size of answer
    if M > 10**5:  # i.e. M > maxn
        raise ValueError("Input too big")  # i.e. n > 149832547102

    # calculate it
    s = fzero
    sq23pi = mpf_mul(mpf_sqrt(from_rational(2, 3, p), p), mpf_pi(p), p)
    sqrt8 = mpf_sqrt(from_int(8), p)
    for q in range(1, M):
        a = _a(n, q, p)
        d = _d(n, q, p, sq23pi, sqrt8)
        s = mpf_add(s, mpf_mul(a, d), prec)
        # On average, the terms decrease rapidly in magnitude.
        # Dynamically reducing the precision greatly improves
        # performance.
        p = bitcount(abs(to_int(d))) + 50
    return int(to_int(mpf_add(s, fhalf, prec)))


@deprecated("""\
The `sympy.ntheory.partitions_.npartitions` has been moved to `sympy.functions.combinatorial.numbers.partition`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def npartitions(n, verbose=False):
    """
    Calculate the partition function P(n), i.e. the number of ways that
    n can be written as a sum of positive integers.

    .. deprecated:: 1.13

        The ``npartitions`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.partition`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    P(n) is computed using the Hardy-Ramanujan-Rademacher formula [1]_.


    The correctness of this implementation has been tested through $10^{10}$.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import partition
    >>> partition(25)
    1958

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PartitionFunctionP.html

    """
    from sympy.functions.combinatorial.numbers import partition as func_partition
    return func_partition(n)
