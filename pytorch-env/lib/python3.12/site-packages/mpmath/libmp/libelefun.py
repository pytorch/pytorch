"""
This module implements computation of elementary transcendental
functions (powers, logarithms, trigonometric and hyperbolic
functions, inverse trigonometric and hyperbolic) for real
floating-point numbers.

For complex and interval implementations of the same functions,
see libmpc and libmpi.

"""

import math
from bisect import bisect

from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND

from .libmpf import (
    round_floor, round_ceiling, round_down, round_up,
    round_nearest, round_fast,
    ComplexResult,
    bitcount, bctable, lshift, rshift, giant_steps, sqrt_fixed,
    from_int, to_int, from_man_exp, to_fixed, to_float, from_float,
    from_rational, normalize,
    fzero, fone, fnone, fhalf, finf, fninf, fnan,
    mpf_cmp, mpf_sign, mpf_abs,
    mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul, mpf_div, mpf_shift,
    mpf_rdiv_int, mpf_pow_int, mpf_sqrt,
    reciprocal_rnd, negative_rnd, mpf_perturb,
    isqrt_fast
)

from .libintmath import ifib


#-------------------------------------------------------------------------------
# Tuning parameters
#-------------------------------------------------------------------------------

# Cutoff for computing exp from cosh+sinh. This reduces the
# number of terms by half, but also requires a square root which
# is expensive with the pure-Python square root code.
if BACKEND == 'python':
    EXP_COSH_CUTOFF = 600
else:
    EXP_COSH_CUTOFF = 400
# Cutoff for using more than 2 series
EXP_SERIES_U_CUTOFF = 1500

# Also basically determined by sqrt
if BACKEND == 'python':
    COS_SIN_CACHE_PREC = 400
else:
    COS_SIN_CACHE_PREC = 200
COS_SIN_CACHE_STEP = 8
cos_sin_cache = {}

# Number of integer logarithms to cache (for zeta sums)
MAX_LOG_INT_CACHE = 2000
log_int_cache = {}

LOG_TAYLOR_PREC = 2500  # Use Taylor series with caching up to this prec
LOG_TAYLOR_SHIFT = 9    # Cache log values in steps of size 2^-N
log_taylor_cache = {}
# prec/size ratio of x for fastest convergence in AGM formula
LOG_AGM_MAG_PREC_RATIO = 20

ATAN_TAYLOR_PREC = 3000  # Same as for log
ATAN_TAYLOR_SHIFT = 7   # steps of size 2^-N
atan_taylor_cache = {}


# ~= next power of two + 20
cache_prec_steps = [22,22]
for k in xrange(1, bitcount(LOG_TAYLOR_PREC)+1):
    cache_prec_steps += [min(2**k,LOG_TAYLOR_PREC)+20] * 2**(k-1)


#----------------------------------------------------------------------------#
#                                                                            #
#                   Elementary mathematical constants                        #
#                                                                            #
#----------------------------------------------------------------------------#

def constant_memo(f):
    """
    Decorator for caching computed values of mathematical
    constants. This decorator should be applied to a
    function taking a single argument prec as input and
    returning a fixed-point value with the given precision.
    """
    f.memo_prec = -1
    f.memo_val = None
    def g(prec, **kwargs):
        memo_prec = f.memo_prec
        if prec <= memo_prec:
            return f.memo_val >> (memo_prec-prec)
        newprec = int(prec*1.05+10)
        f.memo_val = f(newprec, **kwargs)
        f.memo_prec = newprec
        return f.memo_val >> (newprec-prec)
    g.__name__ = f.__name__
    g.__doc__ = f.__doc__
    return g

def def_mpf_constant(fixed):
    """
    Create a function that computes the mpf value for a mathematical
    constant, given a function that computes the fixed-point value.

    Assumptions: the constant is positive and has magnitude ~= 1;
    the fixed-point function rounds to floor.
    """
    def f(prec, rnd=round_fast):
        wp = prec + 20
        v = fixed(wp)
        if rnd in (round_up, round_ceiling):
            v += 1
        return normalize(0, v, -wp, bitcount(v), prec, rnd)
    f.__doc__ = fixed.__doc__
    return f

def bsp_acot(q, a, b, hyperbolic):
    if b - a == 1:
        a1 = MPZ(2*a + 3)
        if hyperbolic or a&1:
            return MPZ_ONE, a1 * q**2, a1
        else:
            return -MPZ_ONE, a1 * q**2, a1
    m = (a+b)//2
    p1, q1, r1 = bsp_acot(q, a, m, hyperbolic)
    p2, q2, r2 = bsp_acot(q, m, b, hyperbolic)
    return q2*p1 + r1*p2, q1*q2, r1*r2

# the acoth(x) series converges like the geometric series for x^2
# N = ceil(p*log(2)/(2*log(x)))
def acot_fixed(a, prec, hyperbolic):
    """
    Compute acot(a) or acoth(a) for an integer a with binary splitting; see
    http://numbers.computation.free.fr/Constants/Algorithms/splitting.html
    """
    N = int(0.35 * prec/math.log(a) + 20)
    p, q, r = bsp_acot(a, 0,N, hyperbolic)
    return ((p+q)<<prec)//(q*a)

def machin(coefs, prec, hyperbolic=False):
    """
    Evaluate a Machin-like formula, i.e., a linear combination of
    acot(n) or acoth(n) for specific integer values of n, using fixed-
    point arithmetic. The input should be a list [(c, n), ...], giving
    c*acot[h](n) + ...
    """
    extraprec = 10
    s = MPZ_ZERO
    for a, b in coefs:
        s += MPZ(a) * acot_fixed(MPZ(b), prec+extraprec, hyperbolic)
    return (s >> extraprec)

# Logarithms of integers are needed for various computations involving
# logarithms, powers, radix conversion, etc

@constant_memo
def ln2_fixed(prec):
    """
    Computes ln(2). This is done with a hyperbolic Machin-type formula,
    with binary splitting at high precision.
    """
    return machin([(18, 26), (-2, 4801), (8, 8749)], prec, True)

@constant_memo
def ln10_fixed(prec):
    """
    Computes ln(10). This is done with a hyperbolic Machin-type formula.
    """
    return machin([(46, 31), (34, 49), (20, 161)], prec, True)


r"""
For computation of pi, we use the Chudnovsky series:

             oo
             ___        k
      1     \       (-1)  (6 k)! (A + B k)
    ----- =  )     -----------------------
    12 pi   /___               3  3k+3/2
                    (3 k)! (k!)  C
            k = 0

where A, B, and C are certain integer constants. This series adds roughly
14 digits per term. Note that C^(3/2) can be extracted so that the
series contains only rational terms. This makes binary splitting very
efficient.

The recurrence formulas for the binary splitting were taken from
ftp://ftp.gmplib.org/pub/src/gmp-chudnovsky.c

Previously, Machin's formula was used at low precision and the AGM iteration
was used at high precision. However, the Chudnovsky series is essentially as
fast as the Machin formula at low precision and in practice about 3x faster
than the AGM at high precision (despite theoretically having a worse
asymptotic complexity), so there is no reason not to use it in all cases.

"""

# Constants in Chudnovsky's series
CHUD_A = MPZ(13591409)
CHUD_B = MPZ(545140134)
CHUD_C = MPZ(640320)
CHUD_D = MPZ(12)

def bs_chudnovsky(a, b, level, verbose):
    """
    Computes the sum from a to b of the series in the Chudnovsky
    formula. Returns g, p, q where p/q is the sum as an exact
    fraction and g is a temporary value used to save work
    for recursive calls.
    """
    if b-a == 1:
        g = MPZ((6*b-5)*(2*b-1)*(6*b-1))
        p = b**3 * CHUD_C**3 // 24
        q = (-1)**b * g * (CHUD_A+CHUD_B*b)
    else:
        if verbose and level < 4:
            print("  binary splitting", a, b)
        mid = (a+b)//2
        g1, p1, q1 = bs_chudnovsky(a, mid, level+1, verbose)
        g2, p2, q2 = bs_chudnovsky(mid, b, level+1, verbose)
        p = p1*p2
        g = g1*g2
        q = q1*p2 + q2*g1
    return g, p, q

@constant_memo
def pi_fixed(prec, verbose=False, verbose_base=None):
    """
    Compute floor(pi * 2**prec) as a big integer.

    This is done using Chudnovsky's series (see comments in
    libelefun.py for details).
    """
    # The Chudnovsky series gives 14.18 digits per term
    N = int(prec/3.3219280948/14.181647462 + 2)
    if verbose:
        print("binary splitting with N =", N)
    g, p, q = bs_chudnovsky(0, N, 0, verbose)
    sqrtC = isqrt_fast(CHUD_C<<(2*prec))
    v = p*CHUD_C*sqrtC//((q+CHUD_A*p)*CHUD_D)
    return v

def degree_fixed(prec):
    return pi_fixed(prec)//180

def bspe(a, b):
    """
    Sum series for exp(1)-1 between a, b, returning the result
    as an exact fraction (p, q).
    """
    if b-a == 1:
        return MPZ_ONE, MPZ(b)
    m = (a+b)//2
    p1, q1 = bspe(a, m)
    p2, q2 = bspe(m, b)
    return p1*q2+p2, q1*q2

@constant_memo
def e_fixed(prec):
    """
    Computes exp(1). This is done using the ordinary Taylor series for
    exp, with binary splitting. For a description of the algorithm,
    see:

        http://numbers.computation.free.fr/Constants/
            Algorithms/splitting.html
    """
    # Slight overestimate of N needed for 1/N! < 2**(-prec)
    # This could be tightened for large N.
    N = int(1.1*prec/math.log(prec) + 20)
    p, q = bspe(0,N)
    return ((p+q)<<prec)//q

@constant_memo
def phi_fixed(prec):
    """
    Computes the golden ratio, (1+sqrt(5))/2
    """
    prec += 10
    a = isqrt_fast(MPZ_FIVE<<(2*prec)) + (MPZ_ONE << prec)
    return a >> 11

mpf_phi    = def_mpf_constant(phi_fixed)
mpf_pi     = def_mpf_constant(pi_fixed)
mpf_e      = def_mpf_constant(e_fixed)
mpf_degree = def_mpf_constant(degree_fixed)
mpf_ln2    = def_mpf_constant(ln2_fixed)
mpf_ln10   = def_mpf_constant(ln10_fixed)


@constant_memo
def ln_sqrt2pi_fixed(prec):
    wp = prec + 10
    # ln(sqrt(2*pi)) = ln(2*pi)/2
    return to_fixed(mpf_log(mpf_shift(mpf_pi(wp), 1), wp), prec-1)

@constant_memo
def sqrtpi_fixed(prec):
    return sqrt_fixed(pi_fixed(prec), prec)

mpf_sqrtpi   = def_mpf_constant(sqrtpi_fixed)
mpf_ln_sqrt2pi   = def_mpf_constant(ln_sqrt2pi_fixed)


#----------------------------------------------------------------------------#
#                                                                            #
#                                    Powers                                  #
#                                                                            #
#----------------------------------------------------------------------------#

def mpf_pow(s, t, prec, rnd=round_fast):
    """
    Compute s**t. Raises ComplexResult if s is negative and t is
    fractional.
    """
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if ssign and texp < 0:
        raise ComplexResult("negative number raised to a fractional power")
    if texp >= 0:
        return mpf_pow_int(s, (-1)**tsign * (tman<<texp), prec, rnd)
    # s**(n/2) = sqrt(s)**n
    if texp == -1:
        if tman == 1:
            if tsign:
                return mpf_div(fone, mpf_sqrt(s, prec+10,
                    reciprocal_rnd[rnd]), prec, rnd)
            return mpf_sqrt(s, prec, rnd)
        else:
            if tsign:
                return mpf_pow_int(mpf_sqrt(s, prec+10,
                    reciprocal_rnd[rnd]), -tman, prec, rnd)
            return mpf_pow_int(mpf_sqrt(s, prec+10, rnd), tman, prec, rnd)
    # General formula: s**t = exp(t*log(s))
    # TODO: handle rnd direction of the logarithm carefully
    c = mpf_log(s, prec+10, rnd)
    return mpf_exp(mpf_mul(t, c), prec, rnd)

def int_pow_fixed(y, n, prec):
    """n-th power of a fixed point number with precision prec

       Returns the power in the form man, exp,
       man * 2**exp ~= y**n
    """
    if n == 2:
        return (y*y), 0
    bc = bitcount(y)
    exp = 0
    workprec = 2 * (prec + 4*bitcount(n) + 4)
    _, pm, pe, pbc = fone
    while 1:
        if n & 1:
            pm = pm*y
            pe = pe+exp
            pbc += bc - 2
            pbc = pbc + bctable[int(pm >> pbc)]
            if pbc > workprec:
                pm = pm >> (pbc-workprec)
                pe += pbc - workprec
                pbc = workprec
            n -= 1
            if not n:
                break
        y = y*y
        exp = exp+exp
        bc = bc + bc - 2
        bc = bc + bctable[int(y >> bc)]
        if bc > workprec:
            y = y >> (bc-workprec)
            exp += bc - workprec
            bc = workprec
        n = n // 2
    return pm, pe

# froot(s, n, prec, rnd) computes the real n-th root of a
# positive mpf tuple s.
# To compute the root we start from a 50-bit estimate for r
# generated with ordinary floating-point arithmetic, and then refine
# the value to full accuracy using the iteration

#            1  /                     y       \
#   r     = --- | (n-1)  * r   +  ----------  |
#    n+1     n  \           n     r_n**(n-1)  /

# which is simply Newton's method applied to the equation r**n = y.
# With giant_steps(start, prec+extra) = [p0,...,pm, prec+extra]
# and y = man * 2**-shift  one has
# (man * 2**exp)**(1/n) =
# y**(1/n) * 2**(start-prec/n) * 2**(p0-start) * ... * 2**(prec+extra-pm) *
# 2**((exp+shift-(n-1)*prec)/n -extra))
# The last factor is accounted for in the last line of froot.

def nthroot_fixed(y, n, prec, exp1):
    start = 50
    try:
        y1 = rshift(y, prec - n*start)
        r = MPZ(int(y1**(1.0/n)))
    except OverflowError:
        y1 = from_int(y1, start)
        fn = from_int(n)
        fn = mpf_rdiv_int(1, fn, start)
        r = mpf_pow(y1, fn, start)
        r = to_int(r)
    extra = 10
    extra1 = n
    prevp = start
    for p in giant_steps(start, prec+extra):
        pm, pe = int_pow_fixed(r, n-1, prevp)
        r2 = rshift(pm, (n-1)*prevp - p - pe - extra1)
        B = lshift(y, 2*p-prec+extra1)//r2
        r = (B + (n-1) * lshift(r, p-prevp))//n
        prevp = p
    return r

def mpf_nthroot(s, n, prec, rnd=round_fast):
    """nth-root of a positive number

    Use the Newton method when faster, otherwise use x**(1/n)
    """
    sign, man, exp, bc = s
    if sign:
        raise ComplexResult("nth root of a negative number")
    if not man:
        if s == fnan:
            return fnan
        if s == fzero:
            if n > 0:
                return fzero
            if n == 0:
                return fone
            return finf
        # Infinity
        if not n:
            return fnan
        if n < 0:
            return fzero
        return finf
    flag_inverse = False
    if n < 2:
        if n == 0:
            return fone
        if n == 1:
            return mpf_pos(s, prec, rnd)
        if n == -1:
            return mpf_div(fone, s, prec, rnd)
        # n < 0
        rnd = reciprocal_rnd[rnd]
        flag_inverse = True
        extra_inverse = 5
        prec += extra_inverse
        n = -n
    if n > 20 and (n >= 20000 or prec < int(233 + 28.3 * n**0.62)):
        prec2 = prec + 10
        fn = from_int(n)
        nth = mpf_rdiv_int(1, fn, prec2)
        r = mpf_pow(s, nth, prec2, rnd)
        s = normalize(r[0], r[1], r[2], r[3], prec, rnd)
        if flag_inverse:
            return mpf_div(fone, s, prec-extra_inverse, rnd)
        else:
            return s
    # Convert to a fixed-point number with prec2 bits.
    prec2 = prec + 2*n - (prec%n)
    # a few tests indicate that
    # for 10 < n < 10**4 a bit more precision is needed
    if n > 10:
        prec2 += prec2//10
        prec2 = prec2 - prec2%n
    # Mantissa may have more bits than we need. Trim it down.
    shift = bc - prec2
    # Adjust exponents to make prec2 and exp+shift multiples of n.
    sign1 = 0
    es = exp+shift
    if es < 0:
        sign1 = 1
        es = -es
    if sign1:
        shift += es%n
    else:
        shift -= es%n
    man = rshift(man, shift)
    extra = 10
    exp1 = ((exp+shift-(n-1)*prec2)//n) - extra
    rnd_shift = 0
    if flag_inverse:
        if rnd == 'u' or rnd == 'c':
            rnd_shift = 1
    else:
        if rnd == 'd' or rnd == 'f':
            rnd_shift = 1
    man = nthroot_fixed(man+rnd_shift, n, prec2, exp1)
    s = from_man_exp(man, exp1, prec, rnd)
    if flag_inverse:
        return mpf_div(fone, s, prec-extra_inverse, rnd)
    else:
        return s

def mpf_cbrt(s, prec, rnd=round_fast):
    """cubic root of a positive number"""
    return mpf_nthroot(s, 3, prec, rnd)

#----------------------------------------------------------------------------#
#                                                                            #
#                                Logarithms                                  #
#                                                                            #
#----------------------------------------------------------------------------#


def log_int_fixed(n, prec, ln2=None):
    """
    Fast computation of log(n), caching the value for small n,
    intended for zeta sums.
    """
    if n in log_int_cache:
        value, vprec = log_int_cache[n]
        if vprec >= prec:
            return value >> (vprec - prec)
    wp = prec + 10
    if wp <= LOG_TAYLOR_SHIFT:
        if ln2 is None:
            ln2 = ln2_fixed(wp)
        r = bitcount(n)
        x = n << (wp-r)
        v = log_taylor_cached(x, wp) + r*ln2
    else:
        v = to_fixed(mpf_log(from_int(n), wp+5), wp)
    if n < MAX_LOG_INT_CACHE:
        log_int_cache[n] = (v, wp)
    return v >> (wp-prec)

def agm_fixed(a, b, prec):
    """
    Fixed-point computation of agm(a,b), assuming
    a, b both close to unit magnitude.
    """
    i = 0
    while 1:
        anew = (a+b)>>1
        if i > 4 and abs(a-anew) < 8:
            return a
        b = isqrt_fast(a*b)
        a = anew
        i += 1
    return a

def log_agm(x, prec):
    """
    Fixed-point computation of -log(x) = log(1/x), suitable
    for large precision. It is required that 0 < x < 1. The
    algorithm used is the Sasaki-Kanada formula

        -log(x) = pi/agm(theta2(x)^2,theta3(x)^2). [1]

    For faster convergence in the theta functions, x should
    be chosen closer to 0.

    Guard bits must be added by the caller.

    HYPOTHESIS: if x = 2^(-n), n bits need to be added to
    account for the truncation to a fixed-point number,
    and this is the only significant cancellation error.

    The number of bits lost to roundoff is small and can be
    considered constant.

    [1] Richard P. Brent, "Fast Algorithms for High-Precision
        Computation of Elementary Functions (extended abstract)",
        http://wwwmaths.anu.edu.au/~brent/pd/RNC7-Brent.pdf

    """
    x2 = (x*x) >> prec
    # Compute jtheta2(x)**2
    s = a = b = x2
    while a:
        b = (b*x2) >> prec
        a = (a*b) >> prec
        s += a
    s += (MPZ_ONE<<prec)
    s = (s*s)>>(prec-2)
    s = (s*isqrt_fast(x<<prec))>>prec
    # Compute jtheta3(x)**2
    t = a = b = x
    while a:
        b = (b*x2) >> prec
        a = (a*b) >> prec
        t += a
    t = (MPZ_ONE<<prec) + (t<<1)
    t = (t*t)>>prec
    # Final formula
    p = agm_fixed(s, t, prec)
    return (pi_fixed(prec) << prec) // p

def log_taylor(x, prec, r=0):
    """
    Fixed-point calculation of log(x). It is assumed that x is close
    enough to 1 for the Taylor series to converge quickly. Convergence
    can be improved by specifying r > 0 to compute
    log(x^(1/2^r))*2^r, at the cost of performing r square roots.

    The caller must provide sufficient guard bits.
    """
    for i in xrange(r):
        x = isqrt_fast(x<<prec)
    one = MPZ_ONE << prec
    v = ((x-one)<<prec)//(x+one)
    sign = v < 0
    if sign:
        v = -v
    v2 = (v*v) >> prec
    v4 = (v2*v2) >> prec
    s0 = v
    s1 = v//3
    v = (v*v4) >> prec
    k = 5
    while v:
        s0 += v // k
        k += 2
        s1 += v // k
        v = (v*v4) >> prec
        k += 2
    s1 = (s1*v2) >> prec
    s = (s0+s1) << (1+r)
    if sign:
        return -s
    return s

def log_taylor_cached(x, prec):
    """
    Fixed-point computation of log(x), assuming x in (0.5, 2)
    and prec <= LOG_TAYLOR_PREC.
    """
    n = x >> (prec-LOG_TAYLOR_SHIFT)
    cached_prec = cache_prec_steps[prec]
    dprec = cached_prec - prec
    if (n, cached_prec) in log_taylor_cache:
        a, log_a = log_taylor_cache[n, cached_prec]
    else:
        a = n << (cached_prec - LOG_TAYLOR_SHIFT)
        log_a = log_taylor(a, cached_prec, 8)
        log_taylor_cache[n, cached_prec] = (a, log_a)
    a >>= dprec
    log_a >>= dprec
    u = ((x - a) << prec) // a
    v = (u << prec) // ((MPZ_TWO << prec) + u)
    v2 = (v*v) >> prec
    v4 = (v2*v2) >> prec
    s0 = v
    s1 = v//3
    v = (v*v4) >> prec
    k = 5
    while v:
        s0 += v//k
        k += 2
        s1 += v//k
        v = (v*v4) >> prec
        k += 2
    s1 = (s1*v2) >> prec
    s = (s0+s1) << 1
    return log_a + s

def mpf_log(x, prec, rnd=round_fast):
    """
    Compute the natural logarithm of the mpf value x. If x is negative,
    ComplexResult is raised.
    """
    sign, man, exp, bc = x
    #------------------------------------------------------------------
    # Handle special values
    if not man:
        if x == fzero: return fninf
        if x == finf: return finf
        if x == fnan: return fnan
    if sign:
        raise ComplexResult("logarithm of a negative number")
    wp = prec + 20
    #------------------------------------------------------------------
    # Handle log(2^n) = log(n)*2.
    # Here we catch the only possible exact value, log(1) = 0
    if man == 1:
        if not exp:
            return fzero
        return from_man_exp(exp*ln2_fixed(wp), -wp, prec, rnd)
    mag = exp+bc
    abs_mag = abs(mag)
    #------------------------------------------------------------------
    # Handle x = 1+eps, where log(x) ~ x. We need to check for
    # cancellation when moving to fixed-point math and compensate
    # by increasing the precision. Note that abs_mag in (0, 1) <=>
    # 0.5 < x < 2 and x != 1
    if abs_mag <= 1:
        # Calculate t = x-1 to measure distance from 1 in bits
        tsign = 1-abs_mag
        if tsign:
            tman = (MPZ_ONE<<bc) - man
        else:
            tman = man - (MPZ_ONE<<(bc-1))
        tbc = bitcount(tman)
        cancellation = bc - tbc
        if cancellation > wp:
            t = normalize(tsign, tman, abs_mag-bc, tbc, tbc, 'n')
            return mpf_perturb(t, tsign, prec, rnd)
        else:
            wp += cancellation
        # TODO: if close enough to 1, we could use Taylor series
        # even in the AGM precision range, since the Taylor series
        # converges rapidly
    #------------------------------------------------------------------
    # Another special case:
    # n*log(2) is a good enough approximation
    if abs_mag > 10000:
        if bitcount(abs_mag) > wp:
            return from_man_exp(exp*ln2_fixed(wp), -wp, prec, rnd)
    #------------------------------------------------------------------
    # General case.
    # Perform argument reduction using log(x) = log(x*2^n) - n*log(2):
    # If we are in the Taylor precision range, choose magnitude 0 or 1.
    # If we are in the AGM precision range, choose magnitude -m for
    # some large m; benchmarking on one machine showed m = prec/20 to be
    # optimal between 1000 and 100,000 digits.
    if wp <= LOG_TAYLOR_PREC:
        m = log_taylor_cached(lshift(man, wp-bc), wp)
        if mag:
            m += mag*ln2_fixed(wp)
    else:
        optimal_mag = -wp//LOG_AGM_MAG_PREC_RATIO
        n = optimal_mag - mag
        x = mpf_shift(x, n)
        wp += (-optimal_mag)
        m = -log_agm(to_fixed(x, wp), wp)
        m -= n*ln2_fixed(wp)
    return from_man_exp(m, -wp, prec, rnd)

def mpf_log_hypot(a, b, prec, rnd):
    """
    Computes log(sqrt(a^2+b^2)) accurately.
    """
    # If either a or b is inf/nan/0, assume it to be a
    if not b[1]:
        a, b = b, a
    # a is inf/nan/0
    if not a[1]:
        # both are inf/nan/0
        if not b[1]:
            if a == b == fzero:
                return fninf
            if fnan in (a, b):
                return fnan
            # at least one term is (+/- inf)^2
            return finf
        # only a is inf/nan/0
        if a == fzero:
            # log(sqrt(0+b^2)) = log(|b|)
            return mpf_log(mpf_abs(b), prec, rnd)
        if a == fnan:
            return fnan
        return finf
    # Exact
    a2 = mpf_mul(a,a)
    b2 = mpf_mul(b,b)
    extra = 20
    # Not exact
    h2 = mpf_add(a2, b2, prec+extra)
    cancelled = mpf_add(h2, fnone, 10)
    mag_cancelled = cancelled[2]+cancelled[3]
    # Just redo the sum exactly if necessary (could be smarter
    # and avoid memory allocation when a or b is precisely 1
    # and the other is tiny...)
    if cancelled == fzero or mag_cancelled < -extra//2:
        h2 = mpf_add(a2, b2, prec+extra-min(a2[2],b2[2]))
    return mpf_shift(mpf_log(h2, prec, rnd), -1)


#----------------------------------------------------------------------
# Inverse tangent
#

def atan_newton(x, prec):
    if prec >= 100:
        r = math.atan(int((x>>(prec-53)))/2.0**53)
    else:
        r = math.atan(int(x)/2.0**prec)
    prevp = 50
    r = MPZ(int(r * 2.0**53) >> (53-prevp))
    extra_p = 50
    for wp in giant_steps(prevp, prec):
        wp += extra_p
        r = r << (wp-prevp)
        cos, sin = cos_sin_fixed(r, wp)
        tan = (sin << wp) // cos
        a = ((tan-rshift(x, prec-wp)) << wp) // ((MPZ_ONE<<wp) + ((tan**2)>>wp))
        r = r - a
        prevp = wp
    return rshift(r, prevp-prec)

def atan_taylor_get_cached(n, prec):
    # Taylor series with caching wins up to huge precisions
    # To avoid unnecessary precomputation at low precision, we
    # do it in steps
    # Round to next power of 2
    prec2 = (1<<(bitcount(prec-1))) + 20
    dprec = prec2 - prec
    if (n, prec2) in atan_taylor_cache:
        a, atan_a = atan_taylor_cache[n, prec2]
    else:
        a = n << (prec2 - ATAN_TAYLOR_SHIFT)
        atan_a = atan_newton(a, prec2)
        atan_taylor_cache[n, prec2] = (a, atan_a)
    return (a >> dprec), (atan_a >> dprec)

def atan_taylor(x, prec):
    n = (x >> (prec-ATAN_TAYLOR_SHIFT))
    a, atan_a = atan_taylor_get_cached(n, prec)
    d = x - a
    s0 = v = (d << prec) // ((a**2 >> prec) + (a*d >> prec) + (MPZ_ONE << prec))
    v2 = (v**2 >> prec)
    v4 = (v2 * v2) >> prec
    s1 = v//3
    v = (v * v4) >> prec
    k = 5
    while v:
        s0 += v // k
        k += 2
        s1 += v // k
        v = (v * v4) >> prec
        k += 2
    s1 = (s1 * v2) >> prec
    s = s0 - s1
    return atan_a + s

def atan_inf(sign, prec, rnd):
    if not sign:
        return mpf_shift(mpf_pi(prec, rnd), -1)
    return mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))

def mpf_atan(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero: return fzero
        if x == finf: return atan_inf(0, prec, rnd)
        if x == fninf: return atan_inf(1, prec, rnd)
        return fnan
    mag = exp + bc
    # Essentially infinity
    if mag > prec+20:
        return atan_inf(sign, prec, rnd)
    # Essentially ~ x
    if -mag > prec+20:
        return mpf_perturb(x, 1-sign, prec, rnd)
    wp = prec + 30 + abs(mag)
    # For large x, use atan(x) = pi/2 - atan(1/x)
    if mag >= 2:
        x = mpf_rdiv_int(1, x, wp)
        reciprocal = True
    else:
        reciprocal = False
    t = to_fixed(x, wp)
    if sign:
        t = -t
    if wp < ATAN_TAYLOR_PREC:
        a = atan_taylor(t, wp)
    else:
        a = atan_newton(t, wp)
    if reciprocal:
        a = ((pi_fixed(wp)>>1)+1) - a
    if sign:
        a = -a
    return from_man_exp(a, -wp, prec, rnd)

# TODO: cleanup the special cases
def mpf_atan2(y, x, prec, rnd=round_fast):
    xsign, xman, xexp, xbc = x
    ysign, yman, yexp, ybc = y
    if not yman:
        if y == fzero and x != fnan:
            if mpf_sign(x) >= 0:
                return fzero
            return mpf_pi(prec, rnd)
        if y in (finf, fninf):
            if x in (finf, fninf):
                return fnan
            # pi/2
            if y == finf:
                return mpf_shift(mpf_pi(prec, rnd), -1)
            # -pi/2
            return mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))
        return fnan
    if ysign:
        return mpf_neg(mpf_atan2(mpf_neg(y), x, prec, negative_rnd[rnd]))
    if not xman:
        if x == fnan:
            return fnan
        if x == finf:
            return fzero
        if x == fninf:
            return mpf_pi(prec, rnd)
        if y == fzero:
            return fzero
        return mpf_shift(mpf_pi(prec, rnd), -1)
    tquo = mpf_atan(mpf_div(y, x, prec+4), prec+4)
    if xsign:
        return mpf_add(mpf_pi(prec+4), tquo, prec, rnd)
    else:
        return mpf_pos(tquo, prec, rnd)

def mpf_asin(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if bc+exp > 0 and x not in (fone, fnone):
        raise ComplexResult("asin(x) is real only for -1 <= x <= 1")
    # asin(x) = 2*atan(x/(1+sqrt(1-x**2)))
    wp = prec + 15
    a = mpf_mul(x, x)
    b = mpf_add(fone, mpf_sqrt(mpf_sub(fone, a, wp), wp), wp)
    c = mpf_div(x, b, wp)
    return mpf_shift(mpf_atan(c, prec, rnd), 1)

def mpf_acos(x, prec, rnd=round_fast):
    # acos(x) = 2*atan(sqrt(1-x**2)/(1+x))
    sign, man, exp, bc = x
    if bc + exp > 0:
        if x not in (fone, fnone):
            raise ComplexResult("acos(x) is real only for -1 <= x <= 1")
        if x == fnone:
            return mpf_pi(prec, rnd)
    wp = prec + 15
    a = mpf_mul(x, x)
    b = mpf_sqrt(mpf_sub(fone, a, wp), wp)
    c = mpf_div(b, mpf_add(fone, x, wp), wp)
    return mpf_shift(mpf_atan(c, prec, rnd), 1)

def mpf_asinh(x, prec, rnd=round_fast):
    wp = prec + 20
    sign, man, exp, bc = x
    mag = exp+bc
    if mag < -8:
        if mag < -wp:
            return mpf_perturb(x, 1-sign, prec, rnd)
        wp += (-mag)
    # asinh(x) = log(x+sqrt(x**2+1))
    # use reflection symmetry to avoid cancellation
    q = mpf_sqrt(mpf_add(mpf_mul(x, x), fone, wp), wp)
    q = mpf_add(mpf_abs(x), q, wp)
    if sign:
        return mpf_neg(mpf_log(q, prec, negative_rnd[rnd]))
    else:
        return mpf_log(q, prec, rnd)

def mpf_acosh(x, prec, rnd=round_fast):
    # acosh(x) = log(x+sqrt(x**2-1))
    wp = prec + 15
    if mpf_cmp(x, fone) == -1:
        raise ComplexResult("acosh(x) is real only for x >= 1")
    q = mpf_sqrt(mpf_add(mpf_mul(x,x), fnone, wp), wp)
    return mpf_log(mpf_add(x, q, wp), prec, rnd)

def mpf_atanh(x, prec, rnd=round_fast):
    # atanh(x) = log((1+x)/(1-x))/2
    sign, man, exp, bc = x
    if (not man) and exp:
        if x in (fzero, fnan):
            return x
        raise ComplexResult("atanh(x) is real only for -1 <= x <= 1")
    mag = bc + exp
    if mag > 0:
        if mag == 1 and man == 1:
            return [finf, fninf][sign]
        raise ComplexResult("atanh(x) is real only for -1 <= x <= 1")
    wp = prec + 15
    if mag < -8:
        if mag < -wp:
            return mpf_perturb(x, sign, prec, rnd)
        wp += (-mag)
    a = mpf_add(x, fone, wp)
    b = mpf_sub(fone, x, wp)
    return mpf_shift(mpf_log(mpf_div(a, b, wp), prec, rnd), -1)

def mpf_fibonacci(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fninf:
            return fnan
        return x
    # F(2^n) ~= 2^(2^n)
    size = abs(exp+bc)
    if exp >= 0:
        # Exact
        if size < 10 or size <= bitcount(prec):
            return from_int(ifib(to_int(x)), prec, rnd)
    # Use the modified Binet formula
    wp = prec + size + 20
    a = mpf_phi(wp)
    b = mpf_add(mpf_shift(a, 1), fnone, wp)
    u = mpf_pow(a, x, wp)
    v = mpf_cos_pi(x, wp)
    v = mpf_div(v, u, wp)
    u = mpf_sub(u, v, wp)
    u = mpf_div(u, b, prec, rnd)
    return u


#-------------------------------------------------------------------------------
# Exponential-type functions
#-------------------------------------------------------------------------------

def exponential_series(x, prec, type=0):
    """
    Taylor series for cosh/sinh or cos/sin.

    type = 0 -- returns exp(x)  (slightly faster than cosh+sinh)
    type = 1 -- returns (cosh(x), sinh(x))
    type = 2 -- returns (cos(x), sin(x))
    """
    if x < 0:
        x = -x
        sign = 1
    else:
        sign = 0
    r = int(0.5*prec**0.5)
    xmag = bitcount(x) - prec
    r = max(0, xmag + r)
    extra = 10 + 2*max(r,-xmag)
    wp = prec + extra
    x <<= (extra - r)
    one = MPZ_ONE << wp
    alt = (type == 2)
    if prec < EXP_SERIES_U_CUTOFF:
        x2 = a = (x*x) >> wp
        x4 = (x2*x2) >> wp
        s0 = s1 = MPZ_ZERO
        k = 2
        while a:
            a //= (k-1)*k; s0 += a; k += 2
            a //= (k-1)*k; s1 += a; k += 2
            a = (a*x4) >> wp
        s1 = (x2*s1) >> wp
        if alt:
            c = s1 - s0 + one
        else:
            c = s1 + s0 + one
    else:
        u = int(0.3*prec**0.35)
        x2 = a = (x*x) >> wp
        xpowers = [one, x2]
        for i in xrange(1, u):
            xpowers.append((xpowers[-1]*x2)>>wp)
        sums = [MPZ_ZERO] * u
        k = 2
        while a:
            for i in xrange(u):
                a //= (k-1)*k
                if alt and k & 2: sums[i] -= a
                else:             sums[i] += a
                k += 2
            a = (a*xpowers[-1]) >> wp
        for i in xrange(1, u):
            sums[i] = (sums[i]*xpowers[i]) >> wp
        c = sum(sums) + one
    if type == 0:
        s = isqrt_fast(c*c - (one<<wp))
        if sign:
            v = c - s
        else:
            v = c + s
        for i in xrange(r):
            v = (v*v) >> wp
        return v >> extra
    else:
        # Repeatedly apply the double-angle formula
        # cosh(2*x) = 2*cosh(x)^2 - 1
        # cos(2*x) = 2*cos(x)^2 - 1
        pshift = wp-1
        for i in xrange(r):
            c = ((c*c) >> pshift) - one
        # With the abs, this is the same for sinh and sin
        s = isqrt_fast(abs((one<<wp) - c*c))
        if sign:
            s = -s
        return (c>>extra), (s>>extra)

def exp_basecase(x, prec):
    """
    Compute exp(x) as a fixed-point number. Works for any x,
    but for speed should have |x| < 1. For an arbitrary number,
    use exp(x) = exp(x-m*log(2)) * 2^m where m = floor(x/log(2)).
    """
    if prec > EXP_COSH_CUTOFF:
        return exponential_series(x, prec, 0)
    r = int(prec**0.5)
    prec += r
    s0 = s1 = (MPZ_ONE << prec)
    k = 2
    a = x2 = (x*x) >> prec
    while a:
        a //= k; s0 += a; k += 1
        a //= k; s1 += a; k += 1
        a = (a*x2) >> prec
    s1 = (s1*x) >> prec
    s = s0 + s1
    u = r
    while r:
        s = (s*s) >> prec
        r -= 1
    return s >> u

def exp_expneg_basecase(x, prec):
    """
    Computation of exp(x), exp(-x)
    """
    if prec > EXP_COSH_CUTOFF:
        cosh, sinh = exponential_series(x, prec, 1)
        return cosh+sinh, cosh-sinh
    a = exp_basecase(x, prec)
    b = (MPZ_ONE << (prec+prec)) // a
    return a, b

def cos_sin_basecase(x, prec):
    """
    Compute cos(x), sin(x) as fixed-point numbers, assuming x
    in [0, pi/2). For an arbitrary number, use x' = x - m*(pi/2)
    where m = floor(x/(pi/2)) along with quarter-period symmetries.
    """
    if prec > COS_SIN_CACHE_PREC:
        return exponential_series(x, prec, 2)
    precs = prec - COS_SIN_CACHE_STEP
    t = x >> precs
    n = int(t)
    if n not in cos_sin_cache:
        w = t<<(10+COS_SIN_CACHE_PREC-COS_SIN_CACHE_STEP)
        cos_t, sin_t = exponential_series(w, 10+COS_SIN_CACHE_PREC, 2)
        cos_sin_cache[n] = (cos_t>>10), (sin_t>>10)
    cos_t, sin_t = cos_sin_cache[n]
    offset = COS_SIN_CACHE_PREC - prec
    cos_t >>= offset
    sin_t >>= offset
    x -= t << precs
    cos = MPZ_ONE << prec
    sin = x
    k = 2
    a = -((x*x) >> prec)
    while a:
        a //= k; cos += a; k += 1; a = (a*x) >> prec
        a //= k; sin += a; k += 1; a = -((a*x) >> prec)
    return ((cos*cos_t-sin*sin_t) >> prec), ((sin*cos_t+cos*sin_t) >> prec)

def mpf_exp(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if man:
        mag = bc + exp
        wp = prec + 14
        if sign:
            man = -man
        # TODO: the best cutoff depends on both x and the precision.
        if prec > 600 and exp >= 0:
            # Need about log2(exp(n)) ~= 1.45*mag extra precision
            e = mpf_e(wp+int(1.45*mag))
            return mpf_pow_int(e, man<<exp, prec, rnd)
        if mag < -wp:
            return mpf_perturb(fone, sign, prec, rnd)
        # |x| >= 2
        if mag > 1:
            # For large arguments: exp(2^mag*(1+eps)) =
            # exp(2^mag)*exp(2^mag*eps) = exp(2^mag)*(1 + 2^mag*eps + ...)
            # so about mag extra bits is required.
            wpmod = wp + mag
            offset = exp + wpmod
            if offset >= 0:
                t = man << offset
            else:
                t = man >> (-offset)
            lg2 = ln2_fixed(wpmod)
            n, t = divmod(t, lg2)
            n = int(n)
            t >>= mag
        else:
            offset = exp + wp
            if offset >= 0:
                t = man << offset
            else:
                t = man >> (-offset)
            n = 0
        man = exp_basecase(t, wp)
        return from_man_exp(man, n-wp, prec, rnd)
    if not exp:
        return fone
    if x == fninf:
        return fzero
    return x


def mpf_cosh_sinh(x, prec, rnd=round_fast, tanh=0):
    """Simultaneously compute (cosh(x), sinh(x)) for real x"""
    sign, man, exp, bc = x
    if (not man) and exp:
        if tanh:
            if x == finf: return fone
            if x == fninf: return fnone
            return fnan
        if x == finf: return (finf, finf)
        if x == fninf: return (finf, fninf)
        return fnan, fnan
    mag = exp+bc
    wp = prec+14
    if mag < -4:
        # Extremely close to 0, sinh(x) ~= x and cosh(x) ~= 1
        if mag < -wp:
            if tanh:
                return mpf_perturb(x, 1-sign, prec, rnd)
            cosh = mpf_perturb(fone, 0, prec, rnd)
            sinh = mpf_perturb(x, sign, prec, rnd)
            return cosh, sinh
        # Fix for cancellation when computing sinh
        wp += (-mag)
    # Does exp(-2*x) vanish?
    if mag > 10:
        if 3*(1<<(mag-1)) > wp:
            # XXX: rounding
            if tanh:
                return mpf_perturb([fone,fnone][sign], 1-sign, prec, rnd)
            c = s = mpf_shift(mpf_exp(mpf_abs(x), prec, rnd), -1)
            if sign:
                s = mpf_neg(s)
            return c, s
    # |x| > 1
    if mag > 1:
        wpmod = wp + mag
        offset = exp + wpmod
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        lg2 = ln2_fixed(wpmod)
        n, t = divmod(t, lg2)
        n = int(n)
        t >>= mag
    else:
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        n = 0
    a, b = exp_expneg_basecase(t, wp)
    # TODO: optimize division precision
    cosh = a + (b>>(2*n))
    sinh = a - (b>>(2*n))
    if sign:
        sinh = -sinh
    if tanh:
        man = (sinh << wp) // cosh
        return from_man_exp(man, -wp, prec, rnd)
    else:
        cosh = from_man_exp(cosh, n-wp-1, prec, rnd)
        sinh = from_man_exp(sinh, n-wp-1, prec, rnd)
        return cosh, sinh


def mod_pi2(man, exp, mag, wp):
    # Reduce to standard interval
    if mag > 0:
        i = 0
        while 1:
            cancellation_prec = 20 << i
            wpmod = wp + mag + cancellation_prec
            pi2 = pi_fixed(wpmod-1)
            pi4 = pi2 >> 1
            offset = wpmod + exp
            if offset >= 0:
                t = man << offset
            else:
                t = man >> (-offset)
            n, y = divmod(t, pi2)
            if y > pi4:
                small = pi2 - y
            else:
                small = y
            if small >> (wp+mag-10):
                n = int(n)
                t = y >> mag
                wp = wpmod - mag
                break
            i += 1
    else:
        wp += (-mag)
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        n = 0
    return t, n, wp


def mpf_cos_sin(x, prec, rnd=round_fast, which=0, pi=False):
    """
    which:
    0 -- return cos(x), sin(x)
    1 -- return cos(x)
    2 -- return sin(x)
    3 -- return tan(x)

    if pi=True, compute for pi*x
    """
    sign, man, exp, bc = x
    if not man:
        if exp:
            c, s = fnan, fnan
        else:
            c, s = fone, fzero
        if which == 0: return c, s
        if which == 1: return c
        if which == 2: return s
        if which == 3: return s

    mag = bc + exp
    wp = prec + 10

    # Extremely small?
    if mag < 0:
        if mag < -wp:
            if pi:
                x = mpf_mul(x, mpf_pi(wp))
            c = mpf_perturb(fone, 1, prec, rnd)
            s = mpf_perturb(x, 1-sign, prec, rnd)
            if which == 0: return c, s
            if which == 1: return c
            if which == 2: return s
            if which == 3: return mpf_perturb(x, sign, prec, rnd)
    if pi:
        if exp >= -1:
            if exp == -1:
                c = fzero
                s = (fone, fnone)[bool(man & 2) ^ sign]
            elif exp == 0:
                c, s = (fnone, fzero)
            else:
                c, s = (fone, fzero)
            if which == 0: return c, s
            if which == 1: return c
            if which == 2: return s
            if which == 3: return mpf_div(s, c, prec, rnd)
        # Subtract nearest half-integer (= mod by pi/2)
        n = ((man >> (-exp-2)) + 1) >> 1
        man = man - (n << (-exp-1))
        mag2 = bitcount(man) + exp
        wp = prec + 10 - mag2
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        t = (t*pi_fixed(wp)) >> wp
    else:
        t, n, wp = mod_pi2(man, exp, mag, wp)
    c, s = cos_sin_basecase(t, wp)
    m = n & 3
    if   m == 1: c, s = -s, c
    elif m == 2: c, s = -c, -s
    elif m == 3: c, s = s, -c
    if sign:
        s = -s
    if which == 0:
        c = from_man_exp(c, -wp, prec, rnd)
        s = from_man_exp(s, -wp, prec, rnd)
        return c, s
    if which == 1:
        return from_man_exp(c, -wp, prec, rnd)
    if which == 2:
        return from_man_exp(s, -wp, prec, rnd)
    if which == 3:
        return from_rational(s, c, prec, rnd)

def mpf_cos(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 1)
def mpf_sin(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 2)
def mpf_tan(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 3)
def mpf_cos_sin_pi(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 0, 1)
def mpf_cos_pi(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 1, 1)
def mpf_sin_pi(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 2, 1)
def mpf_cosh(x, prec, rnd=round_fast): return mpf_cosh_sinh(x, prec, rnd)[0]
def mpf_sinh(x, prec, rnd=round_fast): return mpf_cosh_sinh(x, prec, rnd)[1]
def mpf_tanh(x, prec, rnd=round_fast): return mpf_cosh_sinh(x, prec, rnd, tanh=1)


# Low-overhead fixed-point versions

def cos_sin_fixed(x, prec, pi2=None):
    if pi2 is None:
        pi2 = pi_fixed(prec-1)
    n, t = divmod(x, pi2)
    n = int(n)
    c, s = cos_sin_basecase(t, prec)
    m = n & 3
    if m == 0: return c, s
    if m == 1: return -s, c
    if m == 2: return -c, -s
    if m == 3: return s, -c

def exp_fixed(x, prec, ln2=None):
    if ln2 is None:
        ln2 = ln2_fixed(prec)
    n, t = divmod(x, ln2)
    n = int(n)
    v = exp_basecase(t, prec)
    if n >= 0:
        return v << n
    else:
        return v >> (-n)


if BACKEND == 'sage':
    try:
        import sage.libs.mpmath.ext_libmp as _lbmp
        mpf_sqrt = _lbmp.mpf_sqrt
        mpf_exp = _lbmp.mpf_exp
        mpf_log = _lbmp.mpf_log
        mpf_cos = _lbmp.mpf_cos
        mpf_sin = _lbmp.mpf_sin
        mpf_pow = _lbmp.mpf_pow
        exp_fixed = _lbmp.exp_fixed
        cos_sin_fixed = _lbmp.cos_sin_fixed
        log_int_fixed = _lbmp.log_int_fixed
    except (ImportError, AttributeError):
        print("Warning: Sage imports in libelefun failed")
