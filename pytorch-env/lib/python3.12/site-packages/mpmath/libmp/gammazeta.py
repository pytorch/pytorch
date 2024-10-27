"""
-----------------------------------------------------------------------
This module implements gamma- and zeta-related functions:

* Bernoulli numbers
* Factorials
* The gamma function
* Polygamma functions
* Harmonic numbers
* The Riemann zeta function
* Constants related to these functions

-----------------------------------------------------------------------
"""

import math
import sys

from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy

from .libintmath import list_primes, ifac, ifac2, moebius

from .libmpf import (\
    round_floor, round_ceiling, round_down, round_up,
    round_nearest, round_fast,
    lshift, sqrt_fixed, isqrt_fast,
    fzero, fone, fnone, fhalf, ftwo, finf, fninf, fnan,
    from_int, to_int, to_fixed, from_man_exp, from_rational,
    mpf_pos, mpf_neg, mpf_abs, mpf_add, mpf_sub,
    mpf_mul, mpf_mul_int, mpf_div, mpf_sqrt, mpf_pow_int,
    mpf_rdiv_int,
    mpf_perturb, mpf_le, mpf_lt, mpf_gt, mpf_shift,
    negative_rnd, reciprocal_rnd,
    bitcount, to_float, mpf_floor, mpf_sign, ComplexResult
)

from .libelefun import (\
    constant_memo,
    def_mpf_constant,
    mpf_pi, pi_fixed, ln2_fixed, log_int_fixed, mpf_ln2,
    mpf_exp, mpf_log, mpf_pow, mpf_cosh,
    mpf_cos_sin, mpf_cosh_sinh, mpf_cos_sin_pi, mpf_cos_pi, mpf_sin_pi,
    ln_sqrt2pi_fixed, mpf_ln_sqrt2pi, sqrtpi_fixed, mpf_sqrtpi,
    cos_sin_fixed, exp_fixed
)

from .libmpc import (\
    mpc_zero, mpc_one, mpc_half, mpc_two,
    mpc_abs, mpc_shift, mpc_pos, mpc_neg,
    mpc_add, mpc_sub, mpc_mul, mpc_div,
    mpc_add_mpf, mpc_mul_mpf, mpc_div_mpf, mpc_mpf_div,
    mpc_mul_int, mpc_pow_int,
    mpc_log, mpc_exp, mpc_pow,
    mpc_cos_pi, mpc_sin_pi,
    mpc_reciprocal, mpc_square,
    mpc_sub_mpf
)



# Catalan's constant is computed using Lupas's rapidly convergent series
# (listed on http://mathworld.wolfram.com/CatalansConstant.html)
#            oo
#            ___       n-1  8n     2                   3    2
#        1  \      (-1)    2   (40n  - 24n + 3) [(2n)!] (n!)
#  K =  ---  )     -----------------------------------------
#       64  /___               3               2
#                             n  (2n-1) [(4n)!]
#           n = 1

@constant_memo
def catalan_fixed(prec):
    prec = prec + 20
    a = one = MPZ_ONE << prec
    s, t, n = 0, 1, 1
    while t:
        a *= 32 * n**3 * (2*n-1)
        a //= (3-16*n+16*n**2)**2
        t = a * (-1)**(n-1) * (40*n**2-24*n+3) // (n**3 * (2*n-1))
        s += t
        n += 1
    return s >> (20 + 6)

# Khinchin's constant is relatively difficult to compute. Here
# we use the rational zeta series

#                    oo                2*n-1
#                   ___                ___
#                   \   ` zeta(2*n)-1  \   ` (-1)^(k+1)
#  log(K)*log(2) =   )    ------------  )    ----------
#                   /___.      n       /___.      k
#                   n = 1              k = 1

# which adds half a digit per term. The essential trick for achieving
# reasonable efficiency is to recycle both the values of the zeta
# function (essentially Bernoulli numbers) and the partial terms of
# the inner sum.

# An alternative might be to use K = 2*exp[1/log(2) X] where

#      / 1     1       [ pi*x*(1-x^2) ]
#  X = |    ------ log [ ------------ ].
#      / 0  x(1+x)     [  sin(pi*x)   ]

# and integrate numerically. In practice, this seems to be slightly
# slower than the zeta series at high precision.

@constant_memo
def khinchin_fixed(prec):
    wp = int(prec + prec**0.5 + 15)
    s = MPZ_ZERO
    fac = from_int(4)
    t = ONE = MPZ_ONE << wp
    pi = mpf_pi(wp)
    pipow = twopi2 = mpf_shift(mpf_mul(pi, pi, wp), 2)
    n = 1
    while 1:
        zeta2n = mpf_abs(mpf_bernoulli(2*n, wp))
        zeta2n = mpf_mul(zeta2n, pipow, wp)
        zeta2n = mpf_div(zeta2n, fac, wp)
        zeta2n = to_fixed(zeta2n, wp)
        term = (((zeta2n - ONE) * t) // n) >> wp
        if term < 100:
            break
        #if not n % 10:
        #    print n, math.log(int(abs(term)))
        s += term
        t += ONE//(2*n+1) - ONE//(2*n)
        n += 1
        fac = mpf_mul_int(fac, (2*n)*(2*n-1), wp)
        pipow = mpf_mul(pipow, twopi2, wp)
    s = (s << wp) // ln2_fixed(wp)
    K = mpf_exp(from_man_exp(s, -wp), wp)
    K = to_fixed(K, prec)
    return K


# Glaisher's constant is defined as A = exp(1/2 - zeta'(-1)).
# One way to compute it would be to perform direct numerical
# differentiation, but computing arbitrary Riemann zeta function
# values at high precision is expensive. We instead use the formula

#     A = exp((6 (-zeta'(2))/pi^2 + log 2 pi + gamma)/12)

# and compute zeta'(2) from the series representation

#              oo
#              ___
#             \     log k
#  -zeta'(2) = )    -----
#             /___     2
#                    k
#            k = 2

# This series converges exceptionally slowly, but can be accelerated
# using Euler-Maclaurin formula. The important insight is that the
# E-M integral can be done in closed form and that the high order
# are given by

#    n  /       \
#   d   | log x |   a + b log x
#   --- | ----- | = -----------
#     n |   2   |      2 + n
#   dx  \  x    /     x

# where a and b are integers given by a simple recurrence. Note
# that just one logarithm is needed. However, lots of integer
# logarithms are required for the initial summation.

# This algorithm could possibly be turned into a faster algorithm
# for general evaluation of zeta(s) or zeta'(s); this should be
# looked into.

@constant_memo
def glaisher_fixed(prec):
    wp = prec + 30
    # Number of direct terms to sum before applying the Euler-Maclaurin
    # formula to the tail. TODO: choose more intelligently
    N = int(0.33*prec + 5)
    ONE = MPZ_ONE << wp
    # Euler-Maclaurin, step 1: sum log(k)/k**2 for k from 2 to N-1
    s = MPZ_ZERO
    for k in range(2, N):
        #print k, N
        s += log_int_fixed(k, wp) // k**2
    logN = log_int_fixed(N, wp)
    #logN = to_fixed(mpf_log(from_int(N), wp+20), wp)
    # E-M step 2: integral of log(x)/x**2 from N to inf
    s += (ONE + logN) // N
    # E-M step 3: endpoint correction term f(N)/2
    s += logN // (N**2 * 2)
    # E-M step 4: the series of derivatives
    pN = N**3
    a = 1
    b = -2
    j = 3
    fac = from_int(2)
    k = 1
    while 1:
        # D(2*k-1) * B(2*k) / fac(2*k) [D(n) = nth derivative]
        D = ((a << wp) + b*logN) // pN
        D = from_man_exp(D, -wp)
        B = mpf_bernoulli(2*k, wp)
        term = mpf_mul(B, D, wp)
        term = mpf_div(term, fac, wp)
        term = to_fixed(term, wp)
        if abs(term) < 100:
            break
        #if not k % 10:
        #    print k, math.log(int(abs(term)), 10)
        s -= term
        # Advance derivative twice
        a, b, pN, j = b-a*j, -j*b, pN*N, j+1
        a, b, pN, j = b-a*j, -j*b, pN*N, j+1
        k += 1
        fac = mpf_mul_int(fac, (2*k)*(2*k-1), wp)
    # A = exp((6*s/pi**2 + log(2*pi) + euler)/12)
    pi = pi_fixed(wp)
    s *= 6
    s = (s << wp) // (pi**2 >> wp)
    s += euler_fixed(wp)
    s += to_fixed(mpf_log(from_man_exp(2*pi, -wp), wp), wp)
    s //= 12
    A = mpf_exp(from_man_exp(s, -wp), wp)
    return to_fixed(A, prec)

# Apery's constant can be computed using the very rapidly convergent
# series
#              oo
#              ___              2                      10
#             \         n  205 n  + 250 n + 77     (n!)
#  zeta(3) =   )    (-1)   -------------------  ----------
#             /___               64                      5
#             n = 0                             ((2n+1)!)

@constant_memo
def apery_fixed(prec):
    prec += 20
    d = MPZ_ONE << prec
    term = MPZ(77) << prec
    n = 1
    s = MPZ_ZERO
    while term:
        s += term
        d *= (n**10)
        d //= (((2*n+1)**5) * (2*n)**5)
        term = (-1)**n * (205*(n**2) + 250*n + 77) * d
        n += 1
    return s >> (20 + 6)

"""
Euler's constant (gamma) is computed using the Brent-McMillan formula,
gamma ~= I(n)/J(n) - log(n), where

   I(n) = sum_{k=0,1,2,...} (n**k / k!)**2 * H(k)
   J(n) = sum_{k=0,1,2,...} (n**k / k!)**2
   H(k) = 1 + 1/2 + 1/3 + ... + 1/k

The error is bounded by O(exp(-4n)). Choosing n to be a power
of two, 2**p, the logarithm becomes particularly easy to calculate.[1]

We use the formulation of Algorithm 3.9 in [2] to make the summation
more efficient.

Reference:
[1] Xavier Gourdon & Pascal Sebah, The Euler constant: gamma
http://numbers.computation.free.fr/Constants/Gamma/gamma.pdf

[2] [BorweinBailey]_
"""

@constant_memo
def euler_fixed(prec):
    extra = 30
    prec += extra
    # choose p such that exp(-4*(2**p)) < 2**-n
    p = int(math.log((prec/4) * math.log(2), 2)) + 1
    n = 2**p
    A = U = -p*ln2_fixed(prec)
    B = V = MPZ_ONE << prec
    k = 1
    while 1:
        B = B*n**2//k**2
        A = (A*n**2//k + B)//k
        U += A
        V += B
        if max(abs(A), abs(B)) < 100:
            break
        k += 1
    return (U<<(prec-extra))//V

# Use zeta accelerated formulas for the Mertens and twin
# prime constants; see
# http://mathworld.wolfram.com/MertensConstant.html
# http://mathworld.wolfram.com/TwinPrimesConstant.html

@constant_memo
def mertens_fixed(prec):
    wp = prec + 20
    m = 2
    s = mpf_euler(wp)
    while 1:
        t = mpf_zeta_int(m, wp)
        if t == fone:
            break
        t = mpf_log(t, wp)
        t = mpf_mul_int(t, moebius(m), wp)
        t = mpf_div(t, from_int(m), wp)
        s = mpf_add(s, t)
        m += 1
    return to_fixed(s, prec)

@constant_memo
def twinprime_fixed(prec):
    def I(n):
        return sum(moebius(d)<<(n//d) for d in xrange(1,n+1) if not n%d)//n
    wp = 2*prec + 30
    res = fone
    primes = [from_rational(1,p,wp) for p in [2,3,5,7]]
    ppowers = [mpf_mul(p,p,wp) for p in primes]
    n = 2
    while 1:
        a = mpf_zeta_int(n, wp)
        for i in range(4):
            a = mpf_mul(a, mpf_sub(fone, ppowers[i]), wp)
            ppowers[i] = mpf_mul(ppowers[i], primes[i], wp)
        a = mpf_pow_int(a, -I(n), wp)
        if mpf_pos(a, prec+10, 'n') == fone:
            break
        #from libmpf import to_str
        #print n, to_str(mpf_sub(fone, a), 6)
        res = mpf_mul(res, a, wp)
        n += 1
    res = mpf_mul(res, from_int(3*15*35), wp)
    res = mpf_div(res, from_int(4*16*36), wp)
    return to_fixed(res, prec)


mpf_euler = def_mpf_constant(euler_fixed)
mpf_apery = def_mpf_constant(apery_fixed)
mpf_khinchin = def_mpf_constant(khinchin_fixed)
mpf_glaisher = def_mpf_constant(glaisher_fixed)
mpf_catalan = def_mpf_constant(catalan_fixed)
mpf_mertens = def_mpf_constant(mertens_fixed)
mpf_twinprime = def_mpf_constant(twinprime_fixed)


#-----------------------------------------------------------------------#
#                                                                       #
#                          Bernoulli numbers                            #
#                                                                       #
#-----------------------------------------------------------------------#

MAX_BERNOULLI_CACHE = 3000


r"""
Small Bernoulli numbers and factorials are used in numerous summations,
so it is critical for speed that sequential computation is fast and that
values are cached up to a fairly high threshold.

On the other hand, we also want to support fast computation of isolated
large numbers. Currently, no such acceleration is provided for integer
factorials (though it is for large floating-point factorials, which are
computed via gamma if the precision is low enough).

For sequential computation of Bernoulli numbers, we use Ramanujan's formula

                           / n + 3 \
  B   =  (A(n) - S(n))  /  |       |
   n                       \   n   /

where A(n) = (n+3)/3 when n = 0 or 2 (mod 6), A(n) = -(n+3)/6
when n = 4 (mod 6), and

         [n/6]
          ___
         \      /  n + 3  \
  S(n) =  )     |         | * B
         /___   \ n - 6*k /    n-6*k
         k = 1

For isolated large Bernoulli numbers, we use the Riemann zeta function
to calculate a numerical value for B_n. The von Staudt-Clausen theorem
can then be used to optionally find the exact value of the
numerator and denominator.
"""

bernoulli_cache = {}
f3 = from_int(3)
f6 = from_int(6)

def bernoulli_size(n):
    """Accurately estimate the size of B_n (even n > 2 only)"""
    lgn = math.log(n,2)
    return int(2.326 + 0.5*lgn + n*(lgn - 4.094))

BERNOULLI_PREC_CUTOFF = bernoulli_size(MAX_BERNOULLI_CACHE)

def mpf_bernoulli(n, prec, rnd=None):
    """Computation of Bernoulli numbers (numerically)"""
    if n < 2:
        if n < 0:
            raise ValueError("Bernoulli numbers only defined for n >= 0")
        if n == 0:
            return fone
        if n == 1:
            return mpf_neg(fhalf)
    # For odd n > 1, the Bernoulli numbers are zero
    if n & 1:
        return fzero
    # If precision is extremely high, we can save time by computing
    # the Bernoulli number at a lower precision that is sufficient to
    # obtain the exact fraction, round to the exact fraction, and
    # convert the fraction back to an mpf value at the original precision
    if prec > BERNOULLI_PREC_CUTOFF and prec > bernoulli_size(n)*1.1 + 1000:
        p, q = bernfrac(n)
        return from_rational(p, q, prec, rnd or round_floor)
    if n > MAX_BERNOULLI_CACHE:
        return mpf_bernoulli_huge(n, prec, rnd)
    wp = prec + 30
    # Reuse nearby precisions
    wp += 32 - (prec & 31)
    cached = bernoulli_cache.get(wp)
    if cached:
        numbers, state = cached
        if n in numbers:
            if not rnd:
                return numbers[n]
            return mpf_pos(numbers[n], prec, rnd)
        m, bin, bin1 = state
        if n - m > 10:
            return mpf_bernoulli_huge(n, prec, rnd)
    else:
        if n > 10:
            return mpf_bernoulli_huge(n, prec, rnd)
        numbers = {0:fone}
        m, bin, bin1 = state = [2, MPZ(10), MPZ_ONE]
        bernoulli_cache[wp] = (numbers, state)
    while m <= n:
        #print m
        case = m % 6
        # Accurately estimate size of B_m so we can use
        # fixed point math without using too much precision
        szbm = bernoulli_size(m)
        s = 0
        sexp = max(0, szbm)  - wp
        if m < 6:
            a = MPZ_ZERO
        else:
            a = bin1
        for j in xrange(1, m//6+1):
            usign, uman, uexp, ubc = u = numbers[m-6*j]
            if usign:
                uman = -uman
            s += lshift(a*uman, uexp-sexp)
            # Update inner binomial coefficient
            j6 = 6*j
            a *= ((m-5-j6)*(m-4-j6)*(m-3-j6)*(m-2-j6)*(m-1-j6)*(m-j6))
            a //= ((4+j6)*(5+j6)*(6+j6)*(7+j6)*(8+j6)*(9+j6))
        if case == 0: b = mpf_rdiv_int(m+3, f3, wp)
        if case == 2: b = mpf_rdiv_int(m+3, f3, wp)
        if case == 4: b = mpf_rdiv_int(-m-3, f6, wp)
        s = from_man_exp(s, sexp, wp)
        b = mpf_div(mpf_sub(b, s, wp), from_int(bin), wp)
        numbers[m] = b
        m += 2
        # Update outer binomial coefficient
        bin = bin * ((m+2)*(m+3)) // (m*(m-1))
        if m > 6:
            bin1 = bin1 * ((2+m)*(3+m)) // ((m-7)*(m-6))
        state[:] = [m, bin, bin1]
    return numbers[n]

def mpf_bernoulli_huge(n, prec, rnd=None):
    wp = prec + 10
    piprec = wp + int(math.log(n,2))
    v = mpf_gamma_int(n+1, wp)
    v = mpf_mul(v, mpf_zeta_int(n, wp), wp)
    v = mpf_mul(v, mpf_pow_int(mpf_pi(piprec), -n, wp))
    v = mpf_shift(v, 1-n)
    if not n & 3:
        v = mpf_neg(v)
    return mpf_pos(v, prec, rnd or round_fast)

def bernfrac(n):
    r"""
    Returns a tuple of integers `(p, q)` such that `p/q = B_n` exactly,
    where `B_n` denotes the `n`-th Bernoulli number. The fraction is
    always reduced to lowest terms. Note that for `n > 1` and `n` odd,
    `B_n = 0`, and `(0, 1)` is returned.

    **Examples**

    The first few Bernoulli numbers are exactly::

        >>> from mpmath import *
        >>> for n in range(15):
        ...     p, q = bernfrac(n)
        ...     print("%s %s/%s" % (n, p, q))
        ...
        0 1/1
        1 -1/2
        2 1/6
        3 0/1
        4 -1/30
        5 0/1
        6 1/42
        7 0/1
        8 -1/30
        9 0/1
        10 5/66
        11 0/1
        12 -691/2730
        13 0/1
        14 7/6

    This function works for arbitrarily large `n`::

        >>> p, q = bernfrac(10**4)
        >>> print(q)
        2338224387510
        >>> print(len(str(p)))
        27692
        >>> mp.dps = 15
        >>> print(mpf(p) / q)
        -9.04942396360948e+27677
        >>> print(bernoulli(10**4))
        -9.04942396360948e+27677

    .. note ::

        :func:`~mpmath.bernoulli` computes a floating-point approximation
        directly, without computing the exact fraction first.
        This is much faster for large `n`.

    **Algorithm**

    :func:`~mpmath.bernfrac` works by computing the value of `B_n` numerically
    and then using the von Staudt-Clausen theorem [1] to reconstruct
    the exact fraction. For large `n`, this is significantly faster than
    computing `B_1, B_2, \ldots, B_2` recursively with exact arithmetic.
    The implementation has been tested for `n = 10^m` up to `m = 6`.

    In practice, :func:`~mpmath.bernfrac` appears to be about three times
    slower than the specialized program calcbn.exe [2]

    **References**

    1. MathWorld, von Staudt-Clausen Theorem:
       http://mathworld.wolfram.com/vonStaudt-ClausenTheorem.html

    2. The Bernoulli Number Page:
       http://www.bernoulli.org/

    """
    n = int(n)
    if n < 3:
        return [(1, 1), (-1, 2), (1, 6)][n]
    if n & 1:
        return (0, 1)
    q = 1
    for k in list_primes(n+1):
        if not (n % (k-1)):
            q *= k
    prec = bernoulli_size(n) + int(math.log(q,2)) + 20
    b = mpf_bernoulli(n, prec)
    p = mpf_mul(b, from_int(q))
    pint = to_int(p, round_nearest)
    return (pint, q)


#-----------------------------------------------------------------------#
#                                                                       #
#                         Polygamma functions                           #
#                                                                       #
#-----------------------------------------------------------------------#

r"""
For all polygamma (psi) functions, we use the Euler-Maclaurin summation
formula. It looks slightly different in the m = 0 and m > 0 cases.

For m = 0, we have
                                 oo
                                ___   B
       (0)                1    \       2 k    -2 k
    psi   (z)  ~ log z + --- -  )    ------  z
                         2 z   /___  (2 k)!
                               k = 1

Experiment shows that the minimum term of the asymptotic series
reaches 2^(-p) when Re(z) > 0.11*p. So we simply use the recurrence
for psi (equivalent, in fact, to summing to the first few terms
directly before applying E-M) to obtain z large enough.

Since, very crudely, log z ~= 1 for Re(z) > 1, we can use
fixed-point arithmetic  (if z is extremely large, log(z) itself
is a sufficient approximation, so we can stop there already).

For Re(z) << 0, we could use recurrence, but this is of course
inefficient for large negative z, so there we use the
reflection formula instead.

For m > 0, we have

                  N - 1
                   ___
  ~~~(m)       [  \          1    ]         1            1
  psi   (z)  ~ [   )     -------- ] +  ---------- +  -------- +
               [  /___        m+1 ]           m+1           m
                  k = 1  (z+k)    ]    2 (z+N)       m (z+N)

      oo
     ___    B
    \        2 k   (m+1) (m+2) ... (m+2k-1)
  +  )     ------  ------------------------
    /___   (2 k)!            m + 2 k
    k = 1               (z+N)

where ~~~ denotes the function rescaled by 1/((-1)^(m+1) m!).

Here again N is chosen to make z+N large enough for the minimum
term in the last series to become smaller than eps.

TODO: the current estimation of N for m > 0 is *very suboptimal*.

TODO: implement the reflection formula for m > 0, Re(z) << 0.
It is generally a combination of multiple cotangents. Need to
figure out a reasonably simple way to generate these formulas
on the fly.

TODO: maybe use exact algorithms to compute psi for integral
and certain rational arguments, as this can be much more
efficient. (On the other hand, the availability of these
special values provides a convenient way to test the general
algorithm.)
"""

# Harmonic numbers are just shifted digamma functions
# We should calculate these exactly when x is an integer
# and when doing so is faster.

def mpf_harmonic(x, prec, rnd):
    if x in (fzero, fnan, finf):
        return x
    a = mpf_psi0(mpf_add(fone, x, prec+5), prec)
    return mpf_add(a, mpf_euler(prec+5, rnd), prec, rnd)

def mpc_harmonic(z, prec, rnd):
    if z[1] == fzero:
        return (mpf_harmonic(z[0], prec, rnd), fzero)
    a = mpc_psi0(mpc_add_mpf(z, fone, prec+5), prec)
    return mpc_add_mpf(a, mpf_euler(prec+5, rnd), prec, rnd)

def mpf_psi0(x, prec, rnd=round_fast):
    """
    Computation of the digamma function (psi function of order 0)
    of a real argument.
    """
    sign, man, exp, bc = x
    wp = prec + 10
    if not man:
        if x == finf: return x
        if x == fninf or x == fnan: return fnan
    if x == fzero or (exp >= 0 and sign):
        raise ValueError("polygamma pole")
    # Near 0 -- fixed-point arithmetic becomes bad
    if exp+bc < -5:
        v = mpf_psi0(mpf_add(x, fone, prec, rnd), prec, rnd)
        return mpf_sub(v, mpf_div(fone, x, wp, rnd), prec, rnd)
    # Reflection formula
    if sign and exp+bc > 3:
        c, s = mpf_cos_sin_pi(x, wp)
        q = mpf_mul(mpf_div(c, s, wp), mpf_pi(wp), wp)
        p = mpf_psi0(mpf_sub(fone, x, wp), wp)
        return mpf_sub(p, q, prec, rnd)
    # The logarithmic term is accurate enough
    if (not sign) and bc + exp > wp:
        return mpf_log(mpf_sub(x, fone, wp), prec, rnd)
    # Initial recurrence to obtain a large enough x
    m = to_int(x)
    n = int(0.11*wp) + 2
    s = MPZ_ZERO
    x = to_fixed(x, wp)
    one = MPZ_ONE << wp
    if m < n:
        for k in xrange(m, n):
            s -= (one << wp) // x
            x += one
    x -= one
    # Logarithmic term
    s += to_fixed(mpf_log(from_man_exp(x, -wp, wp), wp), wp)
    # Endpoint term in Euler-Maclaurin expansion
    s += (one << wp) // (2*x)
    # Euler-Maclaurin remainder sum
    x2 = (x*x) >> wp
    t = one
    prev = 0
    k = 1
    while 1:
        t = (t*x2) >> wp
        bsign, bman, bexp, bbc = mpf_bernoulli(2*k, wp)
        offset = (bexp + 2*wp)
        if offset >= 0: term = (bman << offset) // (t*(2*k))
        else:           term = (bman >> (-offset)) // (t*(2*k))
        if k & 1: s -= term
        else:     s += term
        if k > 2 and term >= prev:
            break
        prev = term
        k += 1
    return from_man_exp(s, -wp, wp, rnd)

def mpc_psi0(z, prec, rnd=round_fast):
    """
    Computation of the digamma function (psi function of order 0)
    of a complex argument.
    """
    re, im = z
    # Fall back to the real case
    if im == fzero:
        return (mpf_psi0(re, prec, rnd), fzero)
    wp = prec + 20
    sign, man, exp, bc = re
    # Reflection formula
    if sign and exp+bc > 3:
        c = mpc_cos_pi(z, wp)
        s = mpc_sin_pi(z, wp)
        q = mpc_mul_mpf(mpc_div(c, s, wp), mpf_pi(wp), wp)
        p = mpc_psi0(mpc_sub(mpc_one, z, wp), wp)
        return mpc_sub(p, q, prec, rnd)
    # Just the logarithmic term
    if (not sign) and bc + exp > wp:
        return mpc_log(mpc_sub(z, mpc_one, wp), prec, rnd)
    # Initial recurrence to obtain a large enough z
    w = to_int(re)
    n = int(0.11*wp) + 2
    s = mpc_zero
    if w < n:
        for k in xrange(w, n):
            s = mpc_sub(s, mpc_reciprocal(z, wp), wp)
            z = mpc_add_mpf(z, fone, wp)
    z = mpc_sub(z, mpc_one, wp)
    # Logarithmic and endpoint term
    s = mpc_add(s, mpc_log(z, wp), wp)
    s = mpc_add(s, mpc_div(mpc_half, z, wp), wp)
    # Euler-Maclaurin remainder sum
    z2 = mpc_square(z, wp)
    t = mpc_one
    prev = mpc_zero
    szprev = fzero
    k = 1
    eps = mpf_shift(fone, -wp+2)
    while 1:
        t = mpc_mul(t, z2, wp)
        bern = mpf_bernoulli(2*k, wp)
        term = mpc_mpf_div(bern, mpc_mul_int(t, 2*k, wp), wp)
        s = mpc_sub(s, term, wp)
        szterm = mpc_abs(term, 10)
        if k > 2 and (mpf_le(szterm, eps) or mpf_le(szprev, szterm)):
            break
        prev = term
        szprev = szterm
        k += 1
    return s

# Currently unoptimized
def mpf_psi(m, x, prec, rnd=round_fast):
    """
    Computation of the polygamma function of arbitrary integer order
    m >= 0, for a real argument x.
    """
    if m == 0:
        return mpf_psi0(x, prec, rnd=round_fast)
    return mpc_psi(m, (x, fzero), prec, rnd)[0]

def mpc_psi(m, z, prec, rnd=round_fast):
    """
    Computation of the polygamma function of arbitrary integer order
    m >= 0, for a complex argument z.
    """
    if m == 0:
        return mpc_psi0(z, prec, rnd)
    re, im = z
    wp = prec + 20
    sign, man, exp, bc = re
    if not im[1]:
        if im in (finf, fninf, fnan):
            return (fnan, fnan)
    if not man:
        if re == finf and im == fzero:
            return (fzero, fzero)
        if re == fnan:
            return (fnan, fnan)
    # Recurrence
    w = to_int(re)
    n = int(0.4*wp + 4*m)
    s = mpc_zero
    if w < n:
        for k in xrange(w, n):
            t = mpc_pow_int(z, -m-1, wp)
            s = mpc_add(s, t, wp)
            z = mpc_add_mpf(z, fone, wp)
    zm = mpc_pow_int(z, -m, wp)
    z2 = mpc_pow_int(z, -2, wp)
    # 1/m*(z+N)^m
    integral_term = mpc_div_mpf(zm, from_int(m), wp)
    s = mpc_add(s, integral_term, wp)
    # 1/2*(z+N)^(-(m+1))
    s = mpc_add(s, mpc_mul_mpf(mpc_div(zm, z, wp), fhalf, wp), wp)
    a = m + 1
    b = 2
    k = 1
    # Important: we want to sum up to the *relative* error,
    # not the absolute error, because psi^(m)(z) might be tiny
    magn = mpc_abs(s, 10)
    magn = magn[2]+magn[3]
    eps = mpf_shift(fone, magn-wp+2)
    while 1:
        zm = mpc_mul(zm, z2, wp)
        bern = mpf_bernoulli(2*k, wp)
        scal = mpf_mul_int(bern, a, wp)
        scal = mpf_div(scal, from_int(b), wp)
        term = mpc_mul_mpf(zm, scal, wp)
        s = mpc_add(s, term, wp)
        szterm = mpc_abs(term, 10)
        if k > 2 and mpf_le(szterm, eps):
            break
        #print k, to_str(szterm, 10), to_str(eps, 10)
        a *= (m+2*k)*(m+2*k+1)
        b *= (2*k+1)*(2*k+2)
        k += 1
    # Scale and sign factor
    v = mpc_mul_mpf(s, mpf_gamma(from_int(m+1), wp), prec, rnd)
    if not (m & 1):
        v = mpf_neg(v[0]), mpf_neg(v[1])
    return v


#-----------------------------------------------------------------------#
#                                                                       #
#                         Riemann zeta function                         #
#                                                                       #
#-----------------------------------------------------------------------#

r"""
We use zeta(s) = eta(s) / (1 - 2**(1-s)) and Borwein's approximation

                  n-1
                  ___       k
             -1  \      (-1)  (d_k - d_n)
  eta(s) ~= ----  )     ------------------
             d_n /___              s
                 k = 0      (k + 1)
where
             k
             ___                i
            \     (n + i - 1)! 4
  d_k  =  n  )    ---------------.
            /___   (n - i)! (2i)!
            i = 0

If s = a + b*I, the absolute error for eta(s) is bounded by

    3 (1 + 2|b|)
    ------------ * exp(|b| pi/2)
               n
    (3+sqrt(8))

Disregarding the linear term, we have approximately,

  log(err) ~= log(exp(1.58*|b|)) - log(5.8**n)
  log(err) ~= 1.58*|b| - log(5.8)*n
  log(err) ~= 1.58*|b| - 1.76*n
  log2(err) ~= 2.28*|b| - 2.54*n

So for p bits, we should choose n > (p + 2.28*|b|) / 2.54.

References:
-----------

Peter Borwein, "An Efficient Algorithm for the Riemann Zeta Function"
http://www.cecm.sfu.ca/personal/pborwein/PAPERS/P117.ps

http://en.wikipedia.org/wiki/Dirichlet_eta_function
"""

borwein_cache = {}

def borwein_coefficients(n):
    if n in borwein_cache:
        return borwein_cache[n]
    ds = [MPZ_ZERO] * (n+1)
    d = MPZ_ONE
    s = ds[0] = MPZ_ONE
    for i in range(1, n+1):
        d = d * 4 * (n+i-1) * (n-i+1)
        d //= ((2*i) * ((2*i)-1))
        s += d
        ds[i] = s
    borwein_cache[n] = ds
    return ds

ZETA_INT_CACHE_MAX_PREC = 1000
zeta_int_cache = {}

def mpf_zeta_int(s, prec, rnd=round_fast):
    """
    Optimized computation of zeta(s) for an integer s.
    """
    wp = prec + 20
    s = int(s)
    if s in zeta_int_cache and zeta_int_cache[s][0] >= wp:
        return mpf_pos(zeta_int_cache[s][1], prec, rnd)
    if s < 2:
        if s == 1:
            raise ValueError("zeta(1) pole")
        if not s:
            return mpf_neg(fhalf)
        return mpf_div(mpf_bernoulli(-s+1, wp), from_int(s-1), prec, rnd)
    # 2^-s term vanishes?
    if s >= wp:
        return mpf_perturb(fone, 0, prec, rnd)
    # 5^-s term vanishes?
    elif s >= wp*0.431:
        t = one = 1 << wp
        t += 1 << (wp - s)
        t += one // (MPZ_THREE ** s)
        t += 1 << max(0, wp - s*2)
        return from_man_exp(t, -wp, prec, rnd)
    else:
        # Fast enough to sum directly?
        # Even better, we use the Euler product (idea stolen from pari)
        m = (float(wp)/(s-1) + 1)
        if m < 30:
            needed_terms = int(2.0**m + 1)
            if needed_terms < int(wp/2.54 + 5) / 10:
                t = fone
                for k in list_primes(needed_terms):
                    #print k, needed_terms
                    powprec = int(wp - s*math.log(k,2))
                    if powprec < 2:
                        break
                    a = mpf_sub(fone, mpf_pow_int(from_int(k), -s, powprec), wp)
                    t = mpf_mul(t, a, wp)
                return mpf_div(fone, t, wp)
    # Use Borwein's algorithm
    n = int(wp/2.54 + 5)
    d = borwein_coefficients(n)
    t = MPZ_ZERO
    s = MPZ(s)
    for k in xrange(n):
        t += (((-1)**k * (d[k] - d[n])) << wp) // (k+1)**s
    t = (t << wp) // (-d[n])
    t = (t << wp) // ((1 << wp) - (1 << (wp+1-s)))
    if (s in zeta_int_cache and zeta_int_cache[s][0] < wp) or (s not in zeta_int_cache):
        zeta_int_cache[s] = (wp, from_man_exp(t, -wp-wp))
    return from_man_exp(t, -wp-wp, prec, rnd)

def mpf_zeta(s, prec, rnd=round_fast, alt=0):
    sign, man, exp, bc = s
    if not man:
        if s == fzero:
            if alt:
                return fhalf
            else:
                return mpf_neg(fhalf)
        if s == finf:
            return fone
        return fnan
    wp = prec + 20
    # First term vanishes?
    if (not sign) and (exp + bc > (math.log(wp,2) + 2)):
        return mpf_perturb(fone, alt, prec, rnd)
    # Optimize for integer arguments
    elif exp >= 0:
        if alt:
            if s == fone:
                return mpf_ln2(prec, rnd)
            z = mpf_zeta_int(to_int(s), wp, negative_rnd[rnd])
            q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
            return mpf_mul(z, q, prec, rnd)
        else:
            return mpf_zeta_int(to_int(s), prec, rnd)
    # Negative: use the reflection formula
    # Borwein only proves the accuracy bound for x >= 1/2. However, based on
    # tests, the accuracy without reflection is quite good even some distance
    # to the left of 1/2. XXX: verify this.
    if sign:
        # XXX: could use the separate refl. formula for Dirichlet eta
        if alt:
            q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
            return mpf_mul(mpf_zeta(s, wp), q, prec, rnd)
        # XXX: -1 should be done exactly
        y = mpf_sub(fone, s, 10*wp)
        a = mpf_gamma(y, wp)
        b = mpf_zeta(y, wp)
        c = mpf_sin_pi(mpf_shift(s, -1), wp)
        wp2 = wp + max(0,exp+bc)
        pi = mpf_pi(wp+wp2)
        d = mpf_div(mpf_pow(mpf_shift(pi, 1), s, wp2), pi, wp2)
        return mpf_mul(a,mpf_mul(b,mpf_mul(c,d,wp),wp),prec,rnd)

    # Near pole
    r = mpf_sub(fone, s, wp)
    asign, aman, aexp, abc = mpf_abs(r)
    pole_dist = -2*(aexp+abc)
    if pole_dist > wp:
        if alt:
            return mpf_ln2(prec, rnd)
        else:
            q = mpf_neg(mpf_div(fone, r, wp))
            return mpf_add(q, mpf_euler(wp), prec, rnd)
    else:
        wp += max(0, pole_dist)

    t = MPZ_ZERO
    #wp += 16 - (prec & 15)
    # Use Borwein's algorithm
    n = int(wp/2.54 + 5)
    d = borwein_coefficients(n)
    t = MPZ_ZERO
    sf = to_fixed(s, wp)
    ln2 = ln2_fixed(wp)
    for k in xrange(n):
        u = (-sf*log_int_fixed(k+1, wp, ln2)) >> wp
        #esign, eman, eexp, ebc = mpf_exp(u, wp)
        #offset = eexp + wp
        #if offset >= 0:
        #    w = ((d[k] - d[n]) * eman) << offset
        #else:
        #    w = ((d[k] - d[n]) * eman) >> (-offset)
        eman = exp_fixed(u, wp, ln2)
        w = (d[k] - d[n]) * eman
        if k & 1:
            t -= w
        else:
            t += w
    t = t // (-d[n])
    t = from_man_exp(t, -wp, wp)
    if alt:
        return mpf_pos(t, prec, rnd)
    else:
        q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
        return mpf_div(t, q, prec, rnd)

def mpc_zeta(s, prec, rnd=round_fast, alt=0, force=False):
    re, im = s
    if im == fzero:
        return mpf_zeta(re, prec, rnd, alt), fzero

    # slow for large s
    if (not force) and mpf_gt(mpc_abs(s, 10), from_int(prec)):
        raise NotImplementedError

    wp = prec + 20

    # Near pole
    r = mpc_sub(mpc_one, s, wp)
    asign, aman, aexp, abc = mpc_abs(r, 10)
    pole_dist = -2*(aexp+abc)
    if pole_dist > wp:
        if alt:
            q = mpf_ln2(wp)
            y = mpf_mul(q, mpf_euler(wp), wp)
            g = mpf_shift(mpf_mul(q, q, wp), -1)
            g = mpf_sub(y, g)
            z = mpc_mul_mpf(r, mpf_neg(g), wp)
            z = mpc_add_mpf(z, q, wp)
            return mpc_pos(z, prec, rnd)
        else:
            q = mpc_neg(mpc_div(mpc_one, r, wp))
            q = mpc_add_mpf(q, mpf_euler(wp), wp)
            return mpc_pos(q, prec, rnd)
    else:
        wp += max(0, pole_dist)

    # Reflection formula. To be rigorous, we should reflect to the left of
    # re = 1/2 (see comments for mpf_zeta), but this leads to unnecessary
    # slowdown for interesting values of s
    if mpf_lt(re, fzero):
        # XXX: could use the separate refl. formula for Dirichlet eta
        if alt:
            q = mpc_sub(mpc_one, mpc_pow(mpc_two, mpc_sub(mpc_one, s, wp),
                wp), wp)
            return mpc_mul(mpc_zeta(s, wp), q, prec, rnd)
        # XXX: -1 should be done exactly
        y = mpc_sub(mpc_one, s, 10*wp)
        a = mpc_gamma(y, wp)
        b = mpc_zeta(y, wp)
        c = mpc_sin_pi(mpc_shift(s, -1), wp)
        rsign, rman, rexp, rbc = re
        isign, iman, iexp, ibc = im
        mag = max(rexp+rbc, iexp+ibc)
        wp2 = wp + max(0, mag)
        pi = mpf_pi(wp+wp2)
        pi2 = (mpf_shift(pi, 1), fzero)
        d = mpc_div_mpf(mpc_pow(pi2, s, wp2), pi, wp2)
        return mpc_mul(a,mpc_mul(b,mpc_mul(c,d,wp),wp),prec,rnd)
    n = int(wp/2.54 + 5)
    n += int(0.9*abs(to_int(im)))
    d = borwein_coefficients(n)
    ref = to_fixed(re, wp)
    imf = to_fixed(im, wp)
    tre = MPZ_ZERO
    tim = MPZ_ZERO
    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << (2*wp)
    critical_line = re == fhalf
    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp-1)
    wp2 = wp+wp
    for k in xrange(n):
        log = log_int_fixed(k+1, wp, ln2)
        # A square root is much cheaper than an exp
        if critical_line:
            w = one_2wp // isqrt_fast((k+1) << wp2)
        else:
            w = exp_fixed((-ref*log) >> wp, wp)
        if k & 1:
            w *= (d[n] - d[k])
        else:
            w *= (d[k] - d[n])
        wre, wim = cos_sin_fixed((-imf*log)>>wp, wp, pi2)
        tre += (w * wre) >> wp
        tim += (w * wim) >> wp
    tre //= (-d[n])
    tim //= (-d[n])
    tre = from_man_exp(tre, -wp, wp)
    tim = from_man_exp(tim, -wp, wp)
    if alt:
        return mpc_pos((tre, tim), prec, rnd)
    else:
        q = mpc_sub(mpc_one, mpc_pow(mpc_two, r, wp), wp)
        return mpc_div((tre, tim), q, prec, rnd)

def mpf_altzeta(s, prec, rnd=round_fast):
    return mpf_zeta(s, prec, rnd, 1)

def mpc_altzeta(s, prec, rnd=round_fast):
    return mpc_zeta(s, prec, rnd, 1)

# Not optimized currently
mpf_zetasum = None


def pow_fixed(x, n, wp):
    if n == 1:
        return x
    y = MPZ_ONE << wp
    while n:
        if n & 1:
            y = (y*x) >> wp
            n -= 1
        x = (x*x) >> wp
        n //= 2
    return y

# TODO: optimize / cleanup interface / unify with list_primes
sieve_cache = []
primes_cache = []
mult_cache = []

def primesieve(n):
    global sieve_cache, primes_cache, mult_cache
    if n < len(sieve_cache):
        sieve = sieve_cache#[:n+1]
        primes = primes_cache[:primes_cache.index(max(sieve))+1]
        mult = mult_cache#[:n+1]
        return sieve, primes, mult
    sieve = [0] * (n+1)
    mult = [0] * (n+1)
    primes = list_primes(n)
    for p in primes:
        #sieve[p::p] = p
        for k in xrange(p,n+1,p):
            sieve[k] = p
    for i, p in enumerate(sieve):
        if i >= 2:
            m = 1
            n = i // p
            while not n % p:
                n //= p
                m += 1
            mult[i] = m
    sieve_cache = sieve
    primes_cache = primes
    mult_cache = mult
    return sieve, primes, mult

def zetasum_sieved(critical_line, sre, sim, a, n, wp):
    if a < 1:
        raise ValueError("a cannot be less than 1")
    sieve, primes, mult = primesieve(a+n)
    basic_powers = {}
    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << (2*wp)
    wp2 = wp+wp
    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp-1)
    for p in primes:
        if p*2 > a+n:
            break
        log = log_int_fixed(p, wp, ln2)
        cos, sin = cos_sin_fixed((-sim*log)>>wp, wp, pi2)
        if critical_line:
            u = one_2wp // isqrt_fast(p<<wp2)
        else:
            u = exp_fixed((-sre*log)>>wp, wp)
        pre = (u*cos) >> wp
        pim = (u*sin) >> wp
        basic_powers[p] = [(pre, pim)]
        tre, tim = pre, pim
        for m in range(1,int(math.log(a+n,p)+0.01)+1):
            tre, tim = ((pre*tre-pim*tim)>>wp), ((pim*tre+pre*tim)>>wp)
            basic_powers[p].append((tre,tim))
    xre = MPZ_ZERO
    xim = MPZ_ZERO
    if a == 1:
        xre += one
    aa = max(a,2)
    for k in xrange(aa, a+n+1):
        p = sieve[k]
        if p in basic_powers:
            m = mult[k]
            tre, tim = basic_powers[p][m-1]
            while 1:
                k //= p**m
                if k == 1:
                    break
                p = sieve[k]
                m = mult[k]
                pre, pim = basic_powers[p][m-1]
                tre, tim = ((pre*tre-pim*tim)>>wp), ((pim*tre+pre*tim)>>wp)
        else:
            log = log_int_fixed(k, wp, ln2)
            cos, sin = cos_sin_fixed((-sim*log)>>wp, wp, pi2)
            if critical_line:
                u = one_2wp // isqrt_fast(k<<wp2)
            else:
                u = exp_fixed((-sre*log)>>wp, wp)
            tre = (u*cos) >> wp
            tim = (u*sin) >> wp
        xre += tre
        xim += tim
    return xre, xim

# Set to something large to disable
ZETASUM_SIEVE_CUTOFF = 10

def mpc_zetasum(s, a, n, derivatives, reflect, prec):
    """
    Fast version of mp._zetasum, assuming s = complex, a = integer.
    """

    wp = prec + 10
    derivatives = list(derivatives)
    have_derivatives = derivatives != [0]
    have_one_derivative = len(derivatives) == 1

    # parse s
    sre, sim = s
    critical_line = (sre == fhalf)
    sre = to_fixed(sre, wp)
    sim = to_fixed(sim, wp)

    if a > 0 and n > ZETASUM_SIEVE_CUTOFF and not have_derivatives \
            and not reflect and (n < 4e7 or sys.maxsize > 2**32):
        re, im = zetasum_sieved(critical_line, sre, sim, a, n, wp)
        xs = [(from_man_exp(re, -wp, prec, 'n'), from_man_exp(im, -wp, prec, 'n'))]
        return xs, []

    maxd = max(derivatives)
    if not have_one_derivative:
        derivatives = range(maxd+1)

    # x_d = 0, y_d = 0
    xre = [MPZ_ZERO for d in derivatives]
    xim = [MPZ_ZERO for d in derivatives]
    if reflect:
        yre = [MPZ_ZERO for d in derivatives]
        yim = [MPZ_ZERO for d in derivatives]
    else:
        yre = yim = []

    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << (2*wp)

    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp-1)
    wp2 = wp+wp

    for w in xrange(a, a+n+1):
        log = log_int_fixed(w, wp, ln2)
        cos, sin = cos_sin_fixed((-sim*log)>>wp, wp, pi2)
        if critical_line:
            u = one_2wp // isqrt_fast(w<<wp2)
        else:
            u = exp_fixed((-sre*log)>>wp, wp)
        xterm_re = (u * cos) >> wp
        xterm_im = (u * sin) >> wp
        if reflect:
            reciprocal = (one_2wp // (u*w))
            yterm_re = (reciprocal * cos) >> wp
            yterm_im = (reciprocal * sin) >> wp

        if have_derivatives:
            if have_one_derivative:
                log = pow_fixed(log, maxd, wp)
                xre[0] += (xterm_re * log) >> wp
                xim[0] += (xterm_im * log) >> wp
                if reflect:
                    yre[0] += (yterm_re * log) >> wp
                    yim[0] += (yterm_im * log) >> wp
            else:
                t = MPZ_ONE << wp
                for d in derivatives:
                    xre[d] += (xterm_re * t) >> wp
                    xim[d] += (xterm_im * t) >> wp
                    if reflect:
                        yre[d] += (yterm_re * t) >> wp
                        yim[d] += (yterm_im * t) >> wp
                    t = (t * log) >> wp
        else:
            xre[0] += xterm_re
            xim[0] += xterm_im
            if reflect:
                yre[0] += yterm_re
                yim[0] += yterm_im
    if have_derivatives:
        if have_one_derivative:
            if maxd % 2:
                xre[0] = -xre[0]
                xim[0] = -xim[0]
                if reflect:
                    yre[0] = -yre[0]
                    yim[0] = -yim[0]
        else:
            xre = [(-1)**d * xre[d] for d in derivatives]
            xim = [(-1)**d * xim[d] for d in derivatives]
            if reflect:
                yre = [(-1)**d * yre[d] for d in derivatives]
                yim = [(-1)**d * yim[d] for d in derivatives]
    xs = [(from_man_exp(xa, -wp, prec, 'n'), from_man_exp(xb, -wp, prec, 'n'))
        for (xa, xb) in zip(xre, xim)]
    ys = [(from_man_exp(ya, -wp, prec, 'n'), from_man_exp(yb, -wp, prec, 'n'))
        for (ya, yb) in zip(yre, yim)]
    return xs, ys


#-----------------------------------------------------------------------#
#                                                                       #
#              The gamma function  (NEW IMPLEMENTATION)                 #
#                                                                       #
#-----------------------------------------------------------------------#

# Higher means faster, but more precomputation time
MAX_GAMMA_TAYLOR_PREC = 5000
# Need to derive higher bounds for Taylor series to go higher
assert MAX_GAMMA_TAYLOR_PREC < 15000

# Use Stirling's series if abs(x) > beta*prec
# Important: must be large enough for convergence!
GAMMA_STIRLING_BETA = 0.2

SMALL_FACTORIAL_CACHE_SIZE = 150

gamma_taylor_cache = {}
gamma_stirling_cache = {}

small_factorial_cache = [from_int(ifac(n)) for \
    n in range(SMALL_FACTORIAL_CACHE_SIZE+1)]

def zeta_array(N, prec):
    """
    zeta(n) = A * pi**n / n! + B

    where A is a rational number (A = Bernoulli number
    for n even) and B is an infinite sum over powers of exp(2*pi).
    (B = 0 for n even).

    TODO: this is currently only used for gamma, but could
    be very useful elsewhere.
    """
    extra = 30
    wp = prec+extra
    zeta_values = [MPZ_ZERO] * (N+2)
    pi = pi_fixed(wp)
    # STEP 1:
    one = MPZ_ONE << wp
    zeta_values[0] = -one//2
    f_2pi = mpf_shift(mpf_pi(wp),1)
    exp_2pi_k = exp_2pi = mpf_exp(f_2pi, wp)
    # Compute exponential series
    # Store values of 1/(exp(2*pi*k)-1),
    # exp(2*pi*k)/(exp(2*pi*k)-1)**2, 1/(exp(2*pi*k)-1)**2
    # pi*k*exp(2*pi*k)/(exp(2*pi*k)-1)**2
    exps3 = []
    k = 1
    while 1:
        tp = wp - 9*k
        if tp < 1:
            break
        # 1/(exp(2*pi*k-1)
        q1 = mpf_div(fone, mpf_sub(exp_2pi_k, fone, tp), tp)
        # pi*k*exp(2*pi*k)/(exp(2*pi*k)-1)**2
        q2 = mpf_mul(exp_2pi_k, mpf_mul(q1,q1,tp), tp)
        q1 = to_fixed(q1, wp)
        q2 = to_fixed(q2, wp)
        q2 = (k * q2 * pi) >> wp
        exps3.append((q1, q2))
        # Multiply for next round
        exp_2pi_k = mpf_mul(exp_2pi_k, exp_2pi, wp)
        k += 1
    # Exponential sum
    for n in xrange(3, N+1, 2):
        s = MPZ_ZERO
        k = 1
        for e1, e2 in exps3:
            if n%4 == 3:
                t = e1 // k**n
            else:
                U = (n-1)//4
                t = (e1 + e2//U) // k**n
            if not t:
                break
            s += t
            k += 1
        zeta_values[n] = -2*s
    # Even zeta values
    B = [mpf_abs(mpf_bernoulli(k,wp)) for k in xrange(N+2)]
    pi_pow = fpi = mpf_pow_int(mpf_shift(mpf_pi(wp), 1), 2, wp)
    pi_pow = mpf_div(pi_pow, from_int(4), wp)
    for n in xrange(2,N+2,2):
        z = mpf_mul(B[n], pi_pow, wp)
        zeta_values[n] = to_fixed(z, wp)
        pi_pow = mpf_mul(pi_pow, fpi, wp)
        pi_pow = mpf_div(pi_pow, from_int((n+1)*(n+2)), wp)
    # Zeta sum
    reciprocal_pi = (one << wp) // pi
    for n in xrange(3, N+1, 4):
        U = (n-3)//4
        s = zeta_values[4*U+4]*(4*U+7)//4
        for k in xrange(1, U+1):
            s -= (zeta_values[4*k] * zeta_values[4*U+4-4*k]) >> wp
        zeta_values[n] += (2*s*reciprocal_pi) >> wp
    for n in xrange(5, N+1, 4):
        U = (n-1)//4
        s = zeta_values[4*U+2]*(2*U+1)
        for k in xrange(1, 2*U+1):
            s += ((-1)**k*2*k* zeta_values[2*k] * zeta_values[4*U+2-2*k])>>wp
        zeta_values[n] += ((s*reciprocal_pi)>>wp)//(2*U)
    return [x>>extra for x in zeta_values]

def gamma_taylor_coefficients(inprec):
    """
    Gives the Taylor coefficients of 1/gamma(1+x) as
    a list of fixed-point numbers. Enough coefficients are returned
    to ensure that the series converges to the given precision
    when x is in [0.5, 1.5].
    """
    # Reuse nearby cache values (small case)
    if inprec < 400:
        prec = inprec + (10-(inprec%10))
    elif inprec < 1000:
        prec = inprec + (30-(inprec%30))
    else:
        prec = inprec
    if prec in gamma_taylor_cache:
        return gamma_taylor_cache[prec], prec

    # Experimentally determined bounds
    if prec < 1000:
        N = int(prec**0.76 + 2)
    else:
        # Valid to at least 15000 bits
        N = int(prec**0.787 + 2)

    # Reuse higher precision values
    for cprec in gamma_taylor_cache:
        if cprec > prec:
            coeffs = [x>>(cprec-prec) for x in gamma_taylor_cache[cprec][-N:]]
            if inprec < 1000:
                gamma_taylor_cache[prec] = coeffs
            return coeffs, prec

    # Cache at a higher precision (large case)
    if prec > 1000:
        prec = int(prec * 1.2)

    wp = prec + 20
    A = [0] * N
    A[0] = MPZ_ZERO
    A[1] = MPZ_ONE << wp
    A[2] = euler_fixed(wp)
    # SLOW, reference implementation
    #zeta_values = [0,0]+[to_fixed(mpf_zeta_int(k,wp),wp) for k in xrange(2,N)]
    zeta_values = zeta_array(N, wp)
    for k in xrange(3, N):
        a = (-A[2]*A[k-1])>>wp
        for j in xrange(2,k):
            a += ((-1)**j * zeta_values[j] * A[k-j]) >> wp
        a //= (1-k)
        A[k] = a
    A = [a>>20 for a in A]
    A = A[::-1]
    A = A[:-1]
    gamma_taylor_cache[prec] = A
    #return A, prec
    return gamma_taylor_coefficients(inprec)

def gamma_fixed_taylor(xmpf, x, wp, prec, rnd, type):
    # Determine nearest multiple of N/2
    #n = int(x >> (wp-1))
    #steps = (n-1)>>1
    nearest_int = ((x >> (wp-1)) + MPZ_ONE) >> 1
    one = MPZ_ONE << wp
    coeffs, cwp = gamma_taylor_coefficients(wp)
    if nearest_int > 0:
        r = one
        for i in xrange(nearest_int-1):
            x -= one
            r = (r*x) >> wp
        x -= one
        p = MPZ_ZERO
        for c in coeffs:
            p = c + ((x*p)>>wp)
        p >>= (cwp-wp)
        if type == 0:
            return from_man_exp((r<<wp)//p, -wp, prec, rnd)
        if type == 2:
            return mpf_shift(from_rational(p, (r<<wp), prec, rnd), wp)
        if type == 3:
            return mpf_log(mpf_abs(from_man_exp((r<<wp)//p, -wp)), prec, rnd)
    else:
        r = one
        for i in xrange(-nearest_int):
            r = (r*x) >> wp
            x += one
        p = MPZ_ZERO
        for c in coeffs:
            p = c + ((x*p)>>wp)
        p >>= (cwp-wp)
        if wp - bitcount(abs(x)) > 10:
            # pass very close to 0, so do floating-point multiply
            g = mpf_add(xmpf, from_int(-nearest_int))  # exact
            r = from_man_exp(p*r,-wp-wp)
            r = mpf_mul(r, g, wp)
            if type == 0:
                return mpf_div(fone, r, prec, rnd)
            if type == 2:
                return mpf_pos(r, prec, rnd)
            if type == 3:
                return mpf_log(mpf_abs(mpf_div(fone, r, wp)), prec, rnd)
        else:
            r = from_man_exp(x*p*r,-3*wp)
            if type == 0: return mpf_div(fone, r, prec, rnd)
            if type == 2: return mpf_pos(r, prec, rnd)
            if type == 3: return mpf_neg(mpf_log(mpf_abs(r), prec, rnd))

def stirling_coefficient(n):
    if n in gamma_stirling_cache:
        return gamma_stirling_cache[n]
    p, q = bernfrac(n)
    q *= MPZ(n*(n-1))
    gamma_stirling_cache[n] = p, q, bitcount(abs(p)), bitcount(q)
    return gamma_stirling_cache[n]

def real_stirling_series(x, prec):
    """
    Sums the rational part of Stirling's expansion,

    log(sqrt(2*pi)) - z + 1/(12*z) - 1/(360*z^3) + ...

    """
    t = (MPZ_ONE<<(prec+prec)) // x   # t = 1/x
    u = (t*t)>>prec                  # u = 1/x**2
    s = ln_sqrt2pi_fixed(prec) - x
    # Add initial terms of Stirling's series
    s += t//12;            t = (t*u)>>prec
    s -= t//360;           t = (t*u)>>prec
    s += t//1260;          t = (t*u)>>prec
    s -= t//1680;          t = (t*u)>>prec
    if not t: return s
    s += t//1188;          t = (t*u)>>prec
    s -= 691*t//360360;    t = (t*u)>>prec
    s += t//156;           t = (t*u)>>prec
    if not t: return s
    s -= 3617*t//122400;   t = (t*u)>>prec
    s += 43867*t//244188;  t = (t*u)>>prec
    s -= 174611*t//125400;  t = (t*u)>>prec
    if not t: return s
    k = 22
    # From here on, the coefficients are growing, so we
    # have to keep t at a roughly constant size
    usize = bitcount(abs(u))
    tsize = bitcount(abs(t))
    texp = 0
    while 1:
        p, q, pb, qb = stirling_coefficient(k)
        term_mag = tsize + pb + texp
        shift = -texp
        m = pb - term_mag
        if m > 0 and shift < m:
            p >>= m
            shift -= m
        m = tsize - term_mag
        if m > 0 and shift < m:
            w = t >> m
            shift -= m
        else:
            w = t
        term = (t*p//q) >> shift
        if not term:
            break
        s += term
        t = (t*u) >> usize
        texp -= (prec - usize)
        k += 2
    return s

def complex_stirling_series(x, y, prec):
    # t = 1/z
    _m = (x*x + y*y) >> prec
    tre = (x << prec) // _m
    tim = (-y << prec) // _m
    # u = 1/z**2
    ure = (tre*tre - tim*tim) >> prec
    uim = tim*tre >> (prec-1)
    # s = log(sqrt(2*pi)) - z
    sre = ln_sqrt2pi_fixed(prec) - x
    sim = -y

    # Add initial terms of Stirling's series
    sre += tre//12; sim += tim//12;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= tre//360; sim -= tim//360;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre += tre//1260; sim += tim//1260;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= tre//1680; sim -= tim//1680;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    if abs(tre) + abs(tim) < 5: return sre, sim
    sre += tre//1188; sim += tim//1188;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= 691*tre//360360; sim -= 691*tim//360360;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre += tre//156; sim += tim//156;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    if abs(tre) + abs(tim) < 5: return sre, sim
    sre -= 3617*tre//122400; sim -= 3617*tim//122400;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre += 43867*tre//244188; sim += 43867*tim//244188;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= 174611*tre//125400; sim -= 174611*tim//125400;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    if abs(tre) + abs(tim) < 5: return sre, sim

    k = 22
    # From here on, the coefficients are growing, so we
    # have to keep t at a roughly constant size
    usize = bitcount(max(abs(ure), abs(uim)))
    tsize = bitcount(max(abs(tre), abs(tim)))
    texp = 0
    while 1:
        p, q, pb, qb = stirling_coefficient(k)
        term_mag = tsize + pb + texp
        shift = -texp
        m = pb - term_mag
        if m > 0 and shift < m:
            p >>= m
            shift -= m
        m = tsize - term_mag
        if m > 0 and shift < m:
            wre = tre >> m
            wim = tim >> m
            shift -= m
        else:
            wre = tre
            wim = tim
        termre = (tre*p//q) >> shift
        termim = (tim*p//q) >> shift
        if abs(termre) + abs(termim) < 5:
            break
        sre += termre
        sim += termim
        tre, tim = ((tre*ure - tim*uim)>>usize), \
            ((tre*uim + tim*ure)>>usize)
        texp -= (prec - usize)
        k += 2
    return sre, sim


def mpf_gamma(x, prec, rnd='d', type=0):
    """
    This function implements multipurpose evaluation of the gamma
    function, G(x), as well as the following versions of the same:

    type = 0 -- G(x)                    [standard gamma function]
    type = 1 -- G(x+1) = x*G(x+1) = x!  [factorial]
    type = 2 -- 1/G(x)                  [reciprocal gamma function]
    type = 3 -- log(|G(x)|)             [log-gamma function, real part]
    """

    # Specal values
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            if type == 1: return fone
            if type == 2: return fzero
            raise ValueError("gamma function pole")
        if x == finf:
            if type == 2: return fzero
            return finf
        return fnan

    # First of all, for log gamma, numbers can be well beyond the fixed-point
    # range, so we must take care of huge numbers before e.g. trying
    # to convert x to the nearest integer
    if type == 3:
        wp = prec+20
        if exp+bc > wp and not sign:
            return mpf_sub(mpf_mul(x, mpf_log(x, wp), wp), x, prec, rnd)

    # We strongly want to special-case small integers
    is_integer = exp >= 0
    if is_integer:
        # Poles
        if sign:
            if type == 2:
                return fzero
            raise ValueError("gamma function pole")
        # n = x
        n = man << exp
        if n < SMALL_FACTORIAL_CACHE_SIZE:
            if type == 0:
                return mpf_pos(small_factorial_cache[n-1], prec, rnd)
            if type == 1:
                return mpf_pos(small_factorial_cache[n], prec, rnd)
            if type == 2:
                return mpf_div(fone, small_factorial_cache[n-1], prec, rnd)
            if type == 3:
                return mpf_log(small_factorial_cache[n-1], prec, rnd)
    else:
        # floor(abs(x))
        n = int(man >> (-exp))

    # Estimate size and precision
    # Estimate log(gamma(|x|),2) as x*log(x,2)
    mag = exp + bc
    gamma_size = n*mag

    if type == 3:
        wp = prec + 20
    else:
        wp = prec + bitcount(gamma_size) + 20

    # Very close to 0, pole
    if mag < -wp:
        if type == 0:
            return mpf_sub(mpf_div(fone,x, wp),mpf_shift(fone,-wp),prec,rnd)
        if type == 1: return mpf_sub(fone, x, prec, rnd)
        if type == 2: return mpf_add(x, mpf_shift(fone,mag-wp), prec, rnd)
        if type == 3: return mpf_neg(mpf_log(mpf_abs(x), prec, rnd))

    # From now on, we assume having a gamma function
    if type == 1:
        return mpf_gamma(mpf_add(x, fone), prec, rnd, 0)

    # Special case integers (those not small enough to be caught above,
    # but still small enough for an exact factorial to be faster
    # than an approximate algorithm), and half-integers
    if exp >= -1:
        if is_integer:
            if gamma_size < 10*wp:
                if type == 0:
                    return from_int(ifac(n-1), prec, rnd)
                if type == 2:
                    return from_rational(MPZ_ONE, ifac(n-1), prec, rnd)
                if type == 3:
                    return mpf_log(from_int(ifac(n-1)), prec, rnd)
        # half-integer
        if n < 100 or gamma_size < 10*wp:
            if sign:
                w = sqrtpi_fixed(wp)
                if n % 2: f = ifac2(2*n+1)
                else:     f = -ifac2(2*n+1)
                if type == 0:
                    return mpf_shift(from_rational(w, f, prec, rnd), -wp+n+1)
                if type == 2:
                    return mpf_shift(from_rational(f, w, prec, rnd), wp-n-1)
                if type == 3:
                    return mpf_log(mpf_shift(from_rational(w, abs(f),
                        prec, rnd), -wp+n+1), prec, rnd)
            elif n == 0:
                if type == 0: return mpf_sqrtpi(prec, rnd)
                if type == 2: return mpf_div(fone, mpf_sqrtpi(wp), prec, rnd)
                if type == 3: return mpf_log(mpf_sqrtpi(wp), prec, rnd)
            else:
                w = sqrtpi_fixed(wp)
                w = from_man_exp(w * ifac2(2*n-1), -wp-n)
                if type == 0: return mpf_pos(w, prec, rnd)
                if type == 2: return mpf_div(fone, w, prec, rnd)
                if type == 3: return mpf_log(mpf_abs(w), prec, rnd)

    # Convert to fixed point
    offset = exp + wp
    if offset >= 0: absxman = man << offset
    else:           absxman = man >> (-offset)

    # For log gamma, provide accurate evaluation for x = 1+eps and 2+eps
    if type == 3 and not sign:
        one = MPZ_ONE << wp
        one_dist = abs(absxman-one)
        two_dist = abs(absxman-2*one)
        cancellation = (wp - bitcount(min(one_dist, two_dist)))
        if cancellation > 10:
            xsub1 = mpf_sub(fone, x)
            xsub2 = mpf_sub(ftwo, x)
            xsub1mag = xsub1[2]+xsub1[3]
            xsub2mag = xsub2[2]+xsub2[3]
            if xsub1mag < -wp:
                return mpf_mul(mpf_euler(wp), mpf_sub(fone, x), prec, rnd)
            if xsub2mag < -wp:
                return mpf_mul(mpf_sub(fone, mpf_euler(wp)),
                    mpf_sub(x, ftwo), prec, rnd)
            # Proceed but increase precision
            wp += max(-xsub1mag, -xsub2mag)
            offset = exp + wp
            if offset >= 0: absxman = man << offset
            else:           absxman = man >> (-offset)

    # Use Taylor series if appropriate
    n_for_stirling = int(GAMMA_STIRLING_BETA*wp)
    if n < max(100, n_for_stirling) and wp < MAX_GAMMA_TAYLOR_PREC:
        if sign:
            absxman = -absxman
        return gamma_fixed_taylor(x, absxman, wp, prec, rnd, type)

    # Use Stirling's series
    # First ensure that |x| is large enough for rapid convergence
    xorig = x

    # Argument reduction
    r = 0
    if n < n_for_stirling:
        r = one = MPZ_ONE << wp
        d = n_for_stirling - n
        for k in xrange(d):
            r = (r * absxman) >> wp
            absxman += one
        x = xabs = from_man_exp(absxman, -wp)
        if sign:
            x = mpf_neg(x)
    else:
        xabs = mpf_abs(x)

    # Asymptotic series
    y = real_stirling_series(absxman, wp)
    u = to_fixed(mpf_log(xabs, wp), wp)
    u = ((absxman - (MPZ_ONE<<(wp-1))) * u) >> wp
    y += u
    w = from_man_exp(y, -wp)

    # Compute final value
    if sign:
        # Reflection formula
        A = mpf_mul(mpf_sin_pi(xorig, wp), xorig, wp)
        B = mpf_neg(mpf_pi(wp))
        if type == 0 or type == 2:
            A = mpf_mul(A, mpf_exp(w, wp))
            if r:
                B = mpf_mul(B, from_man_exp(r, -wp), wp)
            if type == 0:
                return mpf_div(B, A, prec, rnd)
            if type == 2:
                return mpf_div(A, B, prec, rnd)
        if type == 3:
            if r:
                B = mpf_mul(B, from_man_exp(r, -wp), wp)
            A = mpf_add(mpf_log(mpf_abs(A), wp), w, wp)
            return mpf_sub(mpf_log(mpf_abs(B), wp), A, prec, rnd)
    else:
        if type == 0:
            if r:
                return mpf_div(mpf_exp(w, wp),
                    from_man_exp(r, -wp), prec, rnd)
            return mpf_exp(w, prec, rnd)
        if type == 2:
            if r:
                return mpf_div(from_man_exp(r, -wp),
                    mpf_exp(w, wp), prec, rnd)
            return mpf_exp(mpf_neg(w), prec, rnd)
        if type == 3:
            if r:
                return mpf_sub(w, mpf_log(from_man_exp(r,-wp), wp), prec, rnd)
            return mpf_pos(w, prec, rnd)


def mpc_gamma(z, prec, rnd='d', type=0):
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b

    if b == fzero:
        # Imaginary part on negative half-axis for log-gamma function
        if type == 3 and asign:
            re = mpf_gamma(a, prec, rnd, 3)
            n = (-aman) >> (-aexp)
            im = mpf_mul_int(mpf_pi(prec+10), n, prec, rnd)
            return re, im
        return mpf_gamma(a, prec, rnd, type), fzero

    # Some kind of complex inf/nan
    if (not aman and aexp) or (not bman and bexp):
        return (fnan, fnan)

    # Initial working precision
    wp = prec + 20

    amag = aexp+abc
    bmag = bexp+bbc
    if aman:
        mag = max(amag, bmag)
    else:
        mag = bmag

    # Close to 0
    if mag < -8:
        if mag < -wp:
            # 1/gamma(z) = z + euler*z^2 + O(z^3)
            v = mpc_add(z, mpc_mul_mpf(mpc_mul(z,z,wp),mpf_euler(wp),wp), wp)
            if type == 0: return mpc_reciprocal(v, prec, rnd)
            if type == 1: return mpc_div(z, v, prec, rnd)
            if type == 2: return mpc_pos(v, prec, rnd)
            if type == 3: return mpc_log(mpc_reciprocal(v, prec), prec, rnd)
        elif type != 1:
            wp += (-mag)

    # Handle huge log-gamma values; must do this before converting to
    # a fixed-point value. TODO: determine a precise cutoff of validity
    # depending on amag and bmag
    if type == 3 and mag > wp and ((not asign) or (bmag >= amag)):
        return mpc_sub(mpc_mul(z, mpc_log(z, wp), wp), z, prec, rnd)

    # From now on, we assume having a gamma function
    if type == 1:
        return mpc_gamma((mpf_add(a, fone), b), prec, rnd, 0)

    an = abs(to_int(a))
    bn = abs(to_int(b))
    absn = max(an, bn)
    gamma_size = absn*mag
    if type == 3:
        pass
    else:
        wp += bitcount(gamma_size)

    # Reflect to the right half-plane. Note that Stirling's expansion
    # is valid in the left half-plane too, as long as we're not too close
    # to the real axis, but in order to use this argument reduction
    # in the negative direction must be implemented.
    #need_reflection = asign and ((bmag < 0) or (amag-bmag > 4))
    need_reflection = asign
    zorig = z
    if need_reflection:
        z = mpc_neg(z)
        asign, aman, aexp, abc = a = z[0]
        bsign, bman, bexp, bbc = b = z[1]

    # Imaginary part very small compared to real one?
    yfinal = 0
    balance_prec = 0
    if bmag < -10:
        # Check z ~= 1 and z ~= 2 for loggamma
        if type == 3:
            zsub1 = mpc_sub_mpf(z, fone)
            if zsub1[0] == fzero:
                cancel1 = -bmag
            else:
                cancel1 = -max(zsub1[0][2]+zsub1[0][3], bmag)
            if cancel1 > wp:
                pi = mpf_pi(wp)
                x = mpc_mul_mpf(zsub1, pi, wp)
                x = mpc_mul(x, x, wp)
                x = mpc_div_mpf(x, from_int(12), wp)
                y = mpc_mul_mpf(zsub1, mpf_neg(mpf_euler(wp)), wp)
                yfinal = mpc_add(x, y, wp)
                if not need_reflection:
                    return mpc_pos(yfinal, prec, rnd)
            elif cancel1 > 0:
                wp += cancel1
            zsub2 = mpc_sub_mpf(z, ftwo)
            if zsub2[0] == fzero:
                cancel2 = -bmag
            else:
                cancel2 = -max(zsub2[0][2]+zsub2[0][3], bmag)
            if cancel2 > wp:
                pi = mpf_pi(wp)
                t = mpf_sub(mpf_mul(pi, pi), from_int(6))
                x = mpc_mul_mpf(mpc_mul(zsub2, zsub2, wp), t, wp)
                x = mpc_div_mpf(x, from_int(12), wp)
                y = mpc_mul_mpf(zsub2, mpf_sub(fone, mpf_euler(wp)), wp)
                yfinal = mpc_add(x, y, wp)
                if not need_reflection:
                    return mpc_pos(yfinal, prec, rnd)
            elif cancel2 > 0:
                wp += cancel2
        if bmag < -wp:
            # Compute directly from the real gamma function.
            pp = 2*(wp+10)
            aabs = mpf_abs(a)
            eps = mpf_shift(fone, amag-wp)
            x1 = mpf_gamma(aabs, pp, type=type)
            x2 = mpf_gamma(mpf_add(aabs, eps), pp, type=type)
            xprime = mpf_div(mpf_sub(x2, x1, pp), eps, pp)
            y = mpf_mul(b, xprime, prec, rnd)
            yfinal = (x1, y)
            # Note: we still need to use the reflection formula for
            # near-poles, and the correct branch of the log-gamma function
            if not need_reflection:
                return mpc_pos(yfinal, prec, rnd)
        else:
            balance_prec += (-bmag)

    wp += balance_prec
    n_for_stirling = int(GAMMA_STIRLING_BETA*wp)
    need_reduction = absn < n_for_stirling

    afix = to_fixed(a, wp)
    bfix = to_fixed(b, wp)

    r = 0
    if not yfinal:
        zprered = z
        # Argument reduction
        if absn < n_for_stirling:
            absn = complex(an, bn)
            d = int((1 + n_for_stirling**2 - bn**2)**0.5 - an)
            rre = one = MPZ_ONE << wp
            rim = MPZ_ZERO
            for k in xrange(d):
                rre, rim = ((afix*rre-bfix*rim)>>wp), ((afix*rim + bfix*rre)>>wp)
                afix += one
            r = from_man_exp(rre, -wp), from_man_exp(rim, -wp)
            a = from_man_exp(afix, -wp)
            z = a, b

        yre, yim = complex_stirling_series(afix, bfix, wp)
        # (z-1/2)*log(z) + S
        lre, lim = mpc_log(z, wp)
        lre = to_fixed(lre, wp)
        lim = to_fixed(lim, wp)
        yre = ((lre*afix - lim*bfix)>>wp) - (lre>>1) + yre
        yim = ((lre*bfix + lim*afix)>>wp) - (lim>>1) + yim
        y = from_man_exp(yre, -wp), from_man_exp(yim, -wp)

        if r and type == 3:
            # If re(z) > 0 and abs(z) <= 4, the branches of loggamma(z)
            # and log(gamma(z)) coincide. Otherwise, use the zeroth order
            # Stirling expansion to compute the correct imaginary part.
            y = mpc_sub(y, mpc_log(r, wp), wp)
            zfa = to_float(zprered[0])
            zfb = to_float(zprered[1])
            zfabs = math.hypot(zfa,zfb)
            #if not (zfa > 0.0 and zfabs <= 4):
            yfb = to_float(y[1])
            u = math.atan2(zfb, zfa)
            if zfabs <= 0.5:
                gi = 0.577216*zfb - u
            else:
                gi = -zfb - 0.5*u + zfa*u + zfb*math.log(zfabs)
            n = int(math.floor((gi-yfb)/(2*math.pi)+0.5))
            y = (y[0], mpf_add(y[1], mpf_mul_int(mpf_pi(wp), 2*n, wp), wp))

    if need_reflection:
        if type == 0 or type == 2:
            A = mpc_mul(mpc_sin_pi(zorig, wp), zorig, wp)
            B = (mpf_neg(mpf_pi(wp)), fzero)
            if yfinal:
                if type == 2:
                    A = mpc_div(A, yfinal, wp)
                else:
                    A = mpc_mul(A, yfinal, wp)
            else:
                A = mpc_mul(A, mpc_exp(y, wp), wp)
            if r:
                B = mpc_mul(B, r, wp)
            if type == 0: return mpc_div(B, A, prec, rnd)
            if type == 2: return mpc_div(A, B, prec, rnd)

        # Reflection formula for the log-gamma function with correct branch
        # http://functions.wolfram.com/GammaBetaErf/LogGamma/16/01/01/0006/
        # LogGamma[z] == -LogGamma[-z] - Log[-z] +
        # Sign[Im[z]] Floor[Re[z]] Pi I + Log[Pi] -
        #      Log[Sin[Pi (z - Floor[Re[z]])]] -
        # Pi I (1 - Abs[Sign[Im[z]]]) Abs[Floor[Re[z]]]
        if type == 3:
            if yfinal:
                s1 = mpc_neg(yfinal)
            else:
                s1 = mpc_neg(y)
            # s -= log(-z)
            s1 = mpc_sub(s1, mpc_log(mpc_neg(zorig), wp), wp)
            # floor(re(z))
            rezfloor = mpf_floor(zorig[0])
            imzsign = mpf_sign(zorig[1])
            pi = mpf_pi(wp)
            t = mpf_mul(pi, rezfloor)
            t = mpf_mul_int(t, imzsign, wp)
            s1 = (s1[0], mpf_add(s1[1], t, wp))
            s1 = mpc_add_mpf(s1, mpf_log(pi, wp), wp)
            t = mpc_sin_pi(mpc_sub_mpf(zorig, rezfloor), wp)
            t = mpc_log(t, wp)
            s1 = mpc_sub(s1, t, wp)
            # Note: may actually be unused, because we fall back
            # to the mpf_ function for real arguments
            if not imzsign:
                t = mpf_mul(pi, mpf_floor(rezfloor), wp)
                s1 = (s1[0], mpf_sub(s1[1], t, wp))
            return mpc_pos(s1, prec, rnd)
    else:
        if type == 0:
            if r:
                return mpc_div(mpc_exp(y, wp), r, prec, rnd)
            return mpc_exp(y, prec, rnd)
        if type == 2:
            if r:
                return mpc_div(r, mpc_exp(y, wp), prec, rnd)
            return mpc_exp(mpc_neg(y), prec, rnd)
        if type == 3:
            return mpc_pos(y, prec, rnd)

def mpf_factorial(x, prec, rnd='d'):
    return mpf_gamma(x, prec, rnd, 1)

def mpc_factorial(x, prec, rnd='d'):
    return mpc_gamma(x, prec, rnd, 1)

def mpf_rgamma(x, prec, rnd='d'):
    return mpf_gamma(x, prec, rnd, 2)

def mpc_rgamma(x, prec, rnd='d'):
    return mpc_gamma(x, prec, rnd, 2)

def mpf_loggamma(x, prec, rnd='d'):
    sign, man, exp, bc = x
    if sign:
        raise ComplexResult
    return mpf_gamma(x, prec, rnd, 3)

def mpc_loggamma(z, prec, rnd='d'):
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero and asign:
        re = mpf_gamma(a, prec, rnd, 3)
        n = (-aman) >> (-aexp)
        im = mpf_mul_int(mpf_pi(prec+10), n, prec, rnd)
        return re, im
    return mpc_gamma(z, prec, rnd, 3)

def mpf_gamma_int(n, prec, rnd=round_fast):
    if n < SMALL_FACTORIAL_CACHE_SIZE:
        return mpf_pos(small_factorial_cache[n-1], prec, rnd)
    return mpf_gamma(from_int(n), prec, rnd)
