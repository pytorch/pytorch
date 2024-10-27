"""
Low-level functions for complex arithmetic.
"""

import sys

from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND

from .libmpf import (\
    round_floor, round_ceiling, round_down, round_up,
    round_nearest, round_fast, bitcount,
    bctable, normalize, normalize1, reciprocal_rnd, rshift, lshift, giant_steps,
    negative_rnd,
    to_str, to_fixed, from_man_exp, from_float, to_float, from_int, to_int,
    fzero, fone, ftwo, fhalf, finf, fninf, fnan, fnone,
    mpf_abs, mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul,
    mpf_div, mpf_mul_int, mpf_shift, mpf_sqrt, mpf_hypot,
    mpf_rdiv_int, mpf_floor, mpf_ceil, mpf_nint, mpf_frac,
    mpf_sign, mpf_hash,
    ComplexResult
)

from .libelefun import (\
    mpf_pi, mpf_exp, mpf_log, mpf_cos_sin, mpf_cosh_sinh, mpf_tan, mpf_pow_int,
    mpf_log_hypot,
    mpf_cos_sin_pi, mpf_phi,
    mpf_cos, mpf_sin, mpf_cos_pi, mpf_sin_pi,
    mpf_atan, mpf_atan2, mpf_cosh, mpf_sinh, mpf_tanh,
    mpf_asin, mpf_acos, mpf_acosh, mpf_nthroot, mpf_fibonacci
)

# An mpc value is a (real, imag) tuple
mpc_one = fone, fzero
mpc_zero = fzero, fzero
mpc_two = ftwo, fzero
mpc_half = (fhalf, fzero)

_infs = (finf, fninf)
_infs_nan = (finf, fninf, fnan)

def mpc_is_inf(z):
    """Check if either real or imaginary part is infinite"""
    re, im = z
    if re in _infs: return True
    if im in _infs: return True
    return False

def mpc_is_infnan(z):
    """Check if either real or imaginary part is infinite or nan"""
    re, im = z
    if re in _infs_nan: return True
    if im in _infs_nan: return True
    return False

def mpc_to_str(z, dps, **kwargs):
    re, im = z
    rs = to_str(re, dps)
    if im[0]:
        return rs + " - " + to_str(mpf_neg(im), dps, **kwargs) + "j"
    else:
        return rs + " + " + to_str(im, dps, **kwargs) + "j"

def mpc_to_complex(z, strict=False, rnd=round_fast):
    re, im = z
    return complex(to_float(re, strict, rnd), to_float(im, strict, rnd))

def mpc_hash(z):
    if sys.version_info >= (3, 2):
        re, im = z
        h = mpf_hash(re) + sys.hash_info.imag * mpf_hash(im)
        # Need to reduce either module 2^32 or 2^64
        h = h % (2**sys.hash_info.width)
        return int(h)
    else:
        try:
            return hash(mpc_to_complex(z, strict=True))
        except OverflowError:
            return hash(z)

def mpc_conjugate(z, prec, rnd=round_fast):
    re, im = z
    return re, mpf_neg(im, prec, rnd)

def mpc_is_nonzero(z):
    return z != mpc_zero

def mpc_add(z, w, prec, rnd=round_fast):
    a, b = z
    c, d = w
    return mpf_add(a, c, prec, rnd), mpf_add(b, d, prec, rnd)

def mpc_add_mpf(z, x, prec, rnd=round_fast):
    a, b = z
    return mpf_add(a, x, prec, rnd), b

def mpc_sub(z, w, prec=0, rnd=round_fast):
    a, b = z
    c, d = w
    return mpf_sub(a, c, prec, rnd), mpf_sub(b, d, prec, rnd)

def mpc_sub_mpf(z, p, prec=0, rnd=round_fast):
    a, b = z
    return mpf_sub(a, p, prec, rnd), b

def mpc_pos(z, prec, rnd=round_fast):
    a, b = z
    return mpf_pos(a, prec, rnd), mpf_pos(b, prec, rnd)

def mpc_neg(z, prec=None, rnd=round_fast):
    a, b = z
    return mpf_neg(a, prec, rnd), mpf_neg(b, prec, rnd)

def mpc_shift(z, n):
    a, b = z
    return mpf_shift(a, n), mpf_shift(b, n)

def mpc_abs(z, prec, rnd=round_fast):
    """Absolute value of a complex number, |a+bi|.
    Returns an mpf value."""
    a, b = z
    return mpf_hypot(a, b, prec, rnd)

def mpc_arg(z, prec, rnd=round_fast):
    """Argument of a complex number. Returns an mpf value."""
    a, b = z
    return mpf_atan2(b, a, prec, rnd)

def mpc_floor(z, prec, rnd=round_fast):
    a, b = z
    return mpf_floor(a, prec, rnd), mpf_floor(b, prec, rnd)

def mpc_ceil(z, prec, rnd=round_fast):
    a, b = z
    return mpf_ceil(a, prec, rnd), mpf_ceil(b, prec, rnd)

def mpc_nint(z, prec, rnd=round_fast):
    a, b = z
    return mpf_nint(a, prec, rnd), mpf_nint(b, prec, rnd)

def mpc_frac(z, prec, rnd=round_fast):
    a, b = z
    return mpf_frac(a, prec, rnd), mpf_frac(b, prec, rnd)


def mpc_mul(z, w, prec, rnd=round_fast):
    """
    Complex multiplication.

    Returns the real and imaginary part of (a+bi)*(c+di), rounded to
    the specified precision. The rounding mode applies to the real and
    imaginary parts separately.
    """
    a, b = z
    c, d = w
    p = mpf_mul(a, c)
    q = mpf_mul(b, d)
    r = mpf_mul(a, d)
    s = mpf_mul(b, c)
    re = mpf_sub(p, q, prec, rnd)
    im = mpf_add(r, s, prec, rnd)
    return re, im

def mpc_square(z, prec, rnd=round_fast):
    # (a+b*I)**2 == a**2 - b**2 + 2*I*a*b
    a, b = z
    p = mpf_mul(a,a)
    q = mpf_mul(b,b)
    r = mpf_mul(a,b, prec, rnd)
    re = mpf_sub(p, q, prec, rnd)
    im = mpf_shift(r, 1)
    return re, im

def mpc_mul_mpf(z, p, prec, rnd=round_fast):
    a, b = z
    re = mpf_mul(a, p, prec, rnd)
    im = mpf_mul(b, p, prec, rnd)
    return re, im

def mpc_mul_imag_mpf(z, x, prec, rnd=round_fast):
    """
    Multiply the mpc value z by I*x where x is an mpf value.
    """
    a, b = z
    re = mpf_neg(mpf_mul(b, x, prec, rnd))
    im = mpf_mul(a, x, prec, rnd)
    return re, im

def mpc_mul_int(z, n, prec, rnd=round_fast):
    a, b = z
    re = mpf_mul_int(a, n, prec, rnd)
    im = mpf_mul_int(b, n, prec, rnd)
    return re, im

def mpc_div(z, w, prec, rnd=round_fast):
    a, b = z
    c, d = w
    wp = prec + 10
    # mag = c*c + d*d
    mag = mpf_add(mpf_mul(c, c), mpf_mul(d, d), wp)
    # (a*c+b*d)/mag, (b*c-a*d)/mag
    t = mpf_add(mpf_mul(a,c), mpf_mul(b,d), wp)
    u = mpf_sub(mpf_mul(b,c), mpf_mul(a,d), wp)
    return mpf_div(t,mag,prec,rnd), mpf_div(u,mag,prec,rnd)

def mpc_div_mpf(z, p, prec, rnd=round_fast):
    """Calculate z/p where p is real"""
    a, b = z
    re = mpf_div(a, p, prec, rnd)
    im = mpf_div(b, p, prec, rnd)
    return re, im

def mpc_reciprocal(z, prec, rnd=round_fast):
    """Calculate 1/z efficiently"""
    a, b = z
    m = mpf_add(mpf_mul(a,a),mpf_mul(b,b),prec+10)
    re = mpf_div(a, m, prec, rnd)
    im = mpf_neg(mpf_div(b, m, prec, rnd))
    return re, im

def mpc_mpf_div(p, z, prec, rnd=round_fast):
    """Calculate p/z where p is real efficiently"""
    a, b = z
    m = mpf_add(mpf_mul(a,a),mpf_mul(b,b), prec+10)
    re = mpf_div(mpf_mul(a,p), m, prec, rnd)
    im = mpf_div(mpf_neg(mpf_mul(b,p)), m, prec, rnd)
    return re, im

def complex_int_pow(a, b, n):
    """Complex integer power: computes (a+b*I)**n exactly for
    nonnegative n (a and b must be Python ints)."""
    wre = 1
    wim = 0
    while n:
        if n & 1:
            wre, wim = wre*a - wim*b, wim*a + wre*b
            n -= 1
        a, b = a*a - b*b, 2*a*b
        n //= 2
    return wre, wim

def mpc_pow(z, w, prec, rnd=round_fast):
    if w[1] == fzero:
        return mpc_pow_mpf(z, w[0], prec, rnd)
    return mpc_exp(mpc_mul(mpc_log(z, prec+10), w, prec+10), prec, rnd)

def mpc_pow_mpf(z, p, prec, rnd=round_fast):
    psign, pman, pexp, pbc = p
    if pexp >= 0:
        return mpc_pow_int(z, (-1)**psign * (pman<<pexp), prec, rnd)
    if pexp == -1:
        sqrtz = mpc_sqrt(z, prec+10)
        return mpc_pow_int(sqrtz, (-1)**psign * pman, prec, rnd)
    return mpc_exp(mpc_mul_mpf(mpc_log(z, prec+10), p, prec+10), prec, rnd)

def mpc_pow_int(z, n, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return mpf_pow_int(a, n, prec, rnd), fzero
    if a == fzero:
        v = mpf_pow_int(b, n, prec, rnd)
        n %= 4
        if n == 0:
            return v, fzero
        elif n == 1:
            return fzero, v
        elif n == 2:
            return mpf_neg(v), fzero
        elif n == 3:
            return fzero, mpf_neg(v)
    if n == 0: return mpc_one
    if n == 1: return mpc_pos(z, prec, rnd)
    if n == 2: return mpc_square(z, prec, rnd)
    if n == -1: return mpc_reciprocal(z, prec, rnd)
    if n < 0: return mpc_reciprocal(mpc_pow_int(z, -n, prec+4), prec, rnd)
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if asign: aman = -aman
    if bsign: bman = -bman
    de = aexp - bexp
    abs_de = abs(de)
    exact_size = n*(abs_de + max(abc, bbc))
    if exact_size < 10000:
        if de > 0:
            aman <<= de
            aexp = bexp
        else:
            bman <<= (-de)
            bexp = aexp
        re, im = complex_int_pow(aman, bman, n)
        re = from_man_exp(re, int(n*aexp), prec, rnd)
        im = from_man_exp(im, int(n*bexp), prec, rnd)
        return re, im
    return mpc_exp(mpc_mul_int(mpc_log(z, prec+10), n, prec+10), prec, rnd)

def mpc_sqrt(z, prec, rnd=round_fast):
    """Complex square root (principal branch).

    We have sqrt(a+bi) = sqrt((r+a)/2) + b/sqrt(2*(r+a))*i where
    r = abs(a+bi), when a+bi is not a negative real number."""
    a, b = z
    if b == fzero:
        if a == fzero:
            return (a, b)
        # When a+bi is a negative real number, we get a real sqrt times i
        if a[0]:
            im = mpf_sqrt(mpf_neg(a), prec, rnd)
            return (fzero, im)
        else:
            re = mpf_sqrt(a, prec, rnd)
            return (re, fzero)
    wp = prec+20
    if not a[0]:                               # case a positive
        t  = mpf_add(mpc_abs((a, b), wp), a, wp)  # t = abs(a+bi) + a
        u = mpf_shift(t, -1)                      # u = t/2
        re = mpf_sqrt(u, prec, rnd)               # re = sqrt(u)
        v = mpf_shift(t, 1)                       # v = 2*t
        w  = mpf_sqrt(v, wp)                      # w = sqrt(v)
        im = mpf_div(b, w, prec, rnd)             # im = b / w
    else:                                      # case a negative
        t = mpf_sub(mpc_abs((a, b), wp), a, wp)   # t = abs(a+bi) - a
        u = mpf_shift(t, -1)                      # u = t/2
        im = mpf_sqrt(u, prec, rnd)               # im = sqrt(u)
        v = mpf_shift(t, 1)                       # v = 2*t
        w  = mpf_sqrt(v, wp)                      # w = sqrt(v)
        re = mpf_div(b, w, prec, rnd)             # re = b/w
        if b[0]:
            re = mpf_neg(re)
            im = mpf_neg(im)
    return re, im

def mpc_nthroot_fixed(a, b, n, prec):
    # a, b signed integers at fixed precision prec
    start = 50
    a1 = int(rshift(a, prec - n*start))
    b1 = int(rshift(b, prec - n*start))
    try:
        r = (a1 + 1j * b1)**(1.0/n)
        re = r.real
        im = r.imag
        re = MPZ(int(re))
        im = MPZ(int(im))
    except OverflowError:
        a1 = from_int(a1, start)
        b1 = from_int(b1, start)
        fn = from_int(n)
        nth = mpf_rdiv_int(1, fn, start)
        re, im = mpc_pow((a1, b1), (nth, fzero), start)
        re = to_int(re)
        im = to_int(im)
    extra = 10
    prevp = start
    extra1 = n
    for p in giant_steps(start, prec+extra):
        # this is slow for large n, unlike int_pow_fixed
        re2, im2 = complex_int_pow(re, im, n-1)
        re2 = rshift(re2, (n-1)*prevp - p - extra1)
        im2 = rshift(im2, (n-1)*prevp - p - extra1)
        r4 = (re2*re2 + im2*im2) >> (p + extra1)
        ap = rshift(a, prec - p)
        bp = rshift(b, prec - p)
        rec = (ap * re2 + bp * im2) >> p
        imc = (-ap * im2 + bp * re2) >> p
        reb = (rec << p) // r4
        imb = (imc << p) // r4
        re = (reb + (n-1)*lshift(re, p-prevp))//n
        im = (imb + (n-1)*lshift(im, p-prevp))//n
        prevp = p
    return re, im

def mpc_nthroot(z, n, prec, rnd=round_fast):
    """
    Complex n-th root.

    Use Newton method as in the real case when it is faster,
    otherwise use z**(1/n)
    """
    a, b = z
    if a[0] == 0 and b == fzero:
        re = mpf_nthroot(a, n, prec, rnd)
        return (re, fzero)
    if n < 2:
        if n == 0:
            return mpc_one
        if n == 1:
            return mpc_pos((a, b), prec, rnd)
        if n == -1:
            return mpc_div(mpc_one, (a, b), prec, rnd)
        inverse = mpc_nthroot((a, b), -n, prec+5, reciprocal_rnd[rnd])
        return mpc_div(mpc_one, inverse, prec, rnd)
    if n <= 20:
        prec2 = int(1.2 * (prec + 10))
        asign, aman, aexp, abc = a
        bsign, bman, bexp, bbc = b
        pf = mpc_abs((a,b), prec)
        if pf[-2] + pf[-1] > -10  and pf[-2] + pf[-1] < prec:
            af = to_fixed(a, prec2)
            bf = to_fixed(b, prec2)
            re, im = mpc_nthroot_fixed(af, bf, n, prec2)
            extra = 10
            re = from_man_exp(re, -prec2-extra, prec2, rnd)
            im = from_man_exp(im, -prec2-extra, prec2, rnd)
            return re, im
    fn = from_int(n)
    prec2 = prec+10 + 10
    nth = mpf_rdiv_int(1, fn, prec2)
    re, im = mpc_pow((a, b), (nth, fzero), prec2, rnd)
    re = normalize(re[0], re[1], re[2], re[3], prec, rnd)
    im = normalize(im[0], im[1], im[2], im[3], prec, rnd)
    return re, im

def mpc_cbrt(z, prec, rnd=round_fast):
    """
    Complex cubic root.
    """
    return mpc_nthroot(z, 3, prec, rnd)

def mpc_exp(z, prec, rnd=round_fast):
    """
    Complex exponential function.

    We use the direct formula exp(a+bi) = exp(a) * (cos(b) + sin(b)*i)
    for the computation. This formula is very nice because it is
    pefectly stable; since we just do real multiplications, the only
    numerical errors that can creep in are single-ulp rounding errors.

    The formula is efficient since mpmath's real exp is quite fast and
    since we can compute cos and sin simultaneously.

    It is no problem if a and b are large; if the implementations of
    exp/cos/sin are accurate and efficient for all real numbers, then
    so is this function for all complex numbers.
    """
    a, b = z
    if a == fzero:
        return mpf_cos_sin(b, prec, rnd)
    if b == fzero:
        return mpf_exp(a, prec, rnd), fzero
    mag = mpf_exp(a, prec+4, rnd)
    c, s = mpf_cos_sin(b, prec+4, rnd)
    re = mpf_mul(mag, c, prec, rnd)
    im = mpf_mul(mag, s, prec, rnd)
    return re, im

def mpc_log(z, prec, rnd=round_fast):
    re = mpf_log_hypot(z[0], z[1], prec, rnd)
    im = mpc_arg(z, prec, rnd)
    return re, im

def mpc_cos(z, prec, rnd=round_fast):
    """Complex cosine. The formula used is cos(a+bi) = cos(a)*cosh(b) -
    sin(a)*sinh(b)*i.

    The same comments apply as for the complex exp: only real
    multiplications are pewrormed, so no cancellation errors are
    possible. The formula is also efficient since we can compute both
    pairs (cos, sin) and (cosh, sinh) in single stwps."""
    a, b = z
    if b == fzero:
        return mpf_cos(a, prec, rnd), fzero
    if a == fzero:
        return mpf_cosh(b, prec, rnd), fzero
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(c, ch, prec, rnd)
    im = mpf_mul(s, sh, prec, rnd)
    return re, mpf_neg(im)

def mpc_sin(z, prec, rnd=round_fast):
    """Complex sine. We have sin(a+bi) = sin(a)*cosh(b) +
    cos(a)*sinh(b)*i. See the docstring for mpc_cos for additional
    comments."""
    a, b = z
    if b == fzero:
        return mpf_sin(a, prec, rnd), fzero
    if a == fzero:
        return fzero, mpf_sinh(b, prec, rnd)
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(s, ch, prec, rnd)
    im = mpf_mul(c, sh, prec, rnd)
    return re, im

def mpc_tan(z, prec, rnd=round_fast):
    """Complex tangent. Computed as tan(a+bi) = sin(2a)/M + sinh(2b)/M*i
    where M = cos(2a) + cosh(2b)."""
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero: return mpf_tan(a, prec, rnd), fzero
    if a == fzero: return fzero, mpf_tanh(b, prec, rnd)
    wp = prec + 15
    a = mpf_shift(a, 1)
    b = mpf_shift(b, 1)
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    # TODO: handle cancellation when c ~=  -1 and ch ~= 1
    mag = mpf_add(c, ch, wp)
    re = mpf_div(s, mag, prec, rnd)
    im = mpf_div(sh, mag, prec, rnd)
    return re, im

def mpc_cos_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return mpf_cos_pi(a, prec, rnd), fzero
    b = mpf_mul(b, mpf_pi(prec+5), prec+5)
    if a == fzero:
        return mpf_cosh(b, prec, rnd), fzero
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(c, ch, prec, rnd)
    im = mpf_mul(s, sh, prec, rnd)
    return re, mpf_neg(im)

def mpc_sin_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return mpf_sin_pi(a, prec, rnd), fzero
    b = mpf_mul(b, mpf_pi(prec+5), prec+5)
    if a == fzero:
        return fzero, mpf_sinh(b, prec, rnd)
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(s, ch, prec, rnd)
    im = mpf_mul(c, sh, prec, rnd)
    return re, im

def mpc_cos_sin(z, prec, rnd=round_fast):
    a, b = z
    if a == fzero:
        ch, sh = mpf_cosh_sinh(b, prec, rnd)
        return (ch, fzero), (fzero, sh)
    if b == fzero:
        c, s = mpf_cos_sin(a, prec, rnd)
        return (c, fzero), (s, fzero)
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    cre = mpf_mul(c, ch, prec, rnd)
    cim = mpf_mul(s, sh, prec, rnd)
    sre = mpf_mul(s, ch, prec, rnd)
    sim = mpf_mul(c, sh, prec, rnd)
    return (cre, mpf_neg(cim)), (sre, sim)

def mpc_cos_sin_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        c, s = mpf_cos_sin_pi(a, prec, rnd)
        return (c, fzero), (s, fzero)
    b = mpf_mul(b, mpf_pi(prec+5), prec+5)
    if a == fzero:
        ch, sh = mpf_cosh_sinh(b, prec, rnd)
        return (ch, fzero), (fzero, sh)
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    cre = mpf_mul(c, ch, prec, rnd)
    cim = mpf_mul(s, sh, prec, rnd)
    sre = mpf_mul(s, ch, prec, rnd)
    sim = mpf_mul(c, sh, prec, rnd)
    return (cre, mpf_neg(cim)), (sre, sim)

def mpc_cosh(z, prec, rnd=round_fast):
    """Complex hyperbolic cosine. Computed as cosh(z) = cos(z*i)."""
    a, b = z
    return mpc_cos((b, mpf_neg(a)), prec, rnd)

def mpc_sinh(z, prec, rnd=round_fast):
    """Complex hyperbolic sine. Computed as sinh(z) = -i*sin(z*i)."""
    a, b = z
    b, a = mpc_sin((b, a), prec, rnd)
    return a, b

def mpc_tanh(z, prec, rnd=round_fast):
    """Complex hyperbolic tangent. Computed as tanh(z) = -i*tan(z*i)."""
    a, b = z
    b, a = mpc_tan((b, a), prec, rnd)
    return a, b

# TODO: avoid loss of accuracy
def mpc_atan(z, prec, rnd=round_fast):
    a, b = z
    # atan(z) = (I/2)*(log(1-I*z) - log(1+I*z))
    # x = 1-I*z = 1 + b - I*a
    # y = 1+I*z = 1 - b + I*a
    wp = prec + 15
    x = mpf_add(fone, b, wp), mpf_neg(a)
    y = mpf_sub(fone, b, wp), a
    l1 = mpc_log(x, wp)
    l2 = mpc_log(y, wp)
    a, b = mpc_sub(l1, l2, prec, rnd)
    # (I/2) * (a+b*I) = (-b/2 + a/2*I)
    v = mpf_neg(mpf_shift(b,-1)), mpf_shift(a,-1)
    # Subtraction at infinity gives correct real part but
    # wrong imaginary part (should be zero)
    if v[1] == fnan and mpc_is_inf(z):
        v = (v[0], fzero)
    return v

beta_crossover = from_float(0.6417)
alpha_crossover = from_float(1.5)

def acos_asin(z, prec, rnd, n):
    """ complex acos for n = 0, asin for n = 1
    The algorithm is described in
    T.E. Hull, T.F. Fairgrieve and P.T.P. Tang
    'Implementing the Complex Arcsine and Arcosine Functions
    using Exception Handling',
    ACM Trans. on Math. Software Vol. 23 (1997), p299
    The complex acos and asin can be defined as
    acos(z) = acos(beta) - I*sign(a)* log(alpha + sqrt(alpha**2 -1))
    asin(z) = asin(beta) + I*sign(a)* log(alpha + sqrt(alpha**2 -1))
    where z = a + I*b
    alpha = (1/2)*(r + s); beta = (1/2)*(r - s) = a/alpha
    r = sqrt((a+1)**2 + y**2); s = sqrt((a-1)**2 + y**2)
    These expressions are rewritten in different ways in different
    regions, delimited by two crossovers alpha_crossover and beta_crossover,
    and by abs(a) <= 1, in order to improve the numerical accuracy.
    """
    a, b = z
    wp = prec + 10
    # special cases with real argument
    if b == fzero:
        am = mpf_sub(fone, mpf_abs(a), wp)
        # case abs(a) <= 1
        if not am[0]:
            if n == 0:
                return mpf_acos(a, prec, rnd), fzero
            else:
                return mpf_asin(a, prec, rnd), fzero
        # cases abs(a) > 1
        else:
            # case a < -1
            if a[0]:
                pi = mpf_pi(prec, rnd)
                c = mpf_acosh(mpf_neg(a), prec, rnd)
                if n == 0:
                    return pi, mpf_neg(c)
                else:
                    return mpf_neg(mpf_shift(pi, -1)), c
            # case a > 1
            else:
                c = mpf_acosh(a, prec, rnd)
                if n == 0:
                    return fzero, c
                else:
                    pi = mpf_pi(prec, rnd)
                    return mpf_shift(pi, -1), mpf_neg(c)
    asign = bsign = 0
    if a[0]:
        a = mpf_neg(a)
        asign = 1
    if b[0]:
        b = mpf_neg(b)
        bsign = 1
    am = mpf_sub(fone, a, wp)
    ap = mpf_add(fone, a, wp)
    r = mpf_hypot(ap, b, wp)
    s = mpf_hypot(am, b, wp)
    alpha = mpf_shift(mpf_add(r, s, wp), -1)
    beta = mpf_div(a, alpha, wp)
    b2 = mpf_mul(b,b, wp)
    # case beta <= beta_crossover
    if not mpf_sub(beta_crossover, beta, wp)[0]:
        if n == 0:
            re = mpf_acos(beta, wp)
        else:
            re = mpf_asin(beta, wp)
    else:
        # to compute the real part in this region use the identity
        # asin(beta) = atan(beta/sqrt(1-beta**2))
        # beta/sqrt(1-beta**2) = (alpha + a) * (alpha - a)
        # alpha + a is numerically accurate; alpha - a can have
        # cancellations leading to numerical inaccuracies, so rewrite
        # it in differente ways according to the region
        Ax = mpf_add(alpha, a, wp)
        # case a <= 1
        if not am[0]:
            # c = b*b/(r + (a+1)); d = (s + (1-a))
            # alpha - a = (1/2)*(c + d)
            # case n=0: re = atan(sqrt((1/2) * Ax * (c + d))/a)
            # case n=1: re = atan(a/sqrt((1/2) * Ax * (c + d)))
            c = mpf_div(b2, mpf_add(r, ap, wp), wp)
            d = mpf_add(s, am, wp)
            re = mpf_shift(mpf_mul(Ax, mpf_add(c, d, wp), wp), -1)
            if n == 0:
                re = mpf_atan(mpf_div(mpf_sqrt(re, wp), a, wp), wp)
            else:
                re = mpf_atan(mpf_div(a, mpf_sqrt(re, wp), wp), wp)
        else:
            # c = Ax/(r + (a+1)); d = Ax/(s - (1-a))
            # alpha - a = (1/2)*(c + d)
            # case n = 0: re = atan(b*sqrt(c + d)/2/a)
            # case n = 1: re = atan(a/(b*sqrt(c + d)/2)
            c = mpf_div(Ax, mpf_add(r, ap, wp), wp)
            d = mpf_div(Ax, mpf_sub(s, am, wp), wp)
            re = mpf_shift(mpf_add(c, d, wp), -1)
            re = mpf_mul(b, mpf_sqrt(re, wp), wp)
            if n == 0:
                re = mpf_atan(mpf_div(re, a, wp), wp)
            else:
                re = mpf_atan(mpf_div(a, re, wp), wp)
    # to compute alpha + sqrt(alpha**2 - 1), if alpha <= alpha_crossover
    # replace it with 1 + Am1 + sqrt(Am1*(alpha+1)))
    # where Am1 = alpha -1
    # if alpha <= alpha_crossover:
    if not mpf_sub(alpha_crossover, alpha, wp)[0]:
        c1 = mpf_div(b2, mpf_add(r, ap, wp), wp)
        # case a < 1
        if mpf_neg(am)[0]:
            # Am1 = (1/2) * (b*b/(r + (a+1)) + b*b/(s + (1-a))
            c2 = mpf_add(s, am, wp)
            c2 = mpf_div(b2, c2, wp)
            Am1 = mpf_shift(mpf_add(c1, c2, wp), -1)
        else:
            # Am1 = (1/2) * (b*b/(r + (a+1)) + (s - (1-a)))
            c2 = mpf_sub(s, am, wp)
            Am1 = mpf_shift(mpf_add(c1, c2, wp), -1)
        # im = log(1 + Am1 + sqrt(Am1*(alpha+1)))
        im = mpf_mul(Am1, mpf_add(alpha, fone, wp), wp)
        im = mpf_log(mpf_add(fone, mpf_add(Am1, mpf_sqrt(im, wp), wp), wp), wp)
    else:
        # im = log(alpha + sqrt(alpha*alpha - 1))
        im = mpf_sqrt(mpf_sub(mpf_mul(alpha, alpha, wp), fone, wp), wp)
        im = mpf_log(mpf_add(alpha, im, wp), wp)
    if asign:
        if n == 0:
            re = mpf_sub(mpf_pi(wp), re, wp)
        else:
            re = mpf_neg(re)
    if not bsign and n == 0:
        im = mpf_neg(im)
    if bsign and n == 1:
        im = mpf_neg(im)
    re = normalize(re[0], re[1], re[2], re[3], prec, rnd)
    im = normalize(im[0], im[1], im[2], im[3], prec, rnd)
    return re, im

def mpc_acos(z, prec, rnd=round_fast):
    return acos_asin(z, prec, rnd, 0)

def mpc_asin(z, prec, rnd=round_fast):
    return acos_asin(z, prec, rnd, 1)

def mpc_asinh(z, prec, rnd=round_fast):
    # asinh(z) = I * asin(-I z)
    a, b = z
    a, b =  mpc_asin((b, mpf_neg(a)), prec, rnd)
    return mpf_neg(b), a

def mpc_acosh(z, prec, rnd=round_fast):
    # acosh(z) = -I * acos(z)   for Im(acos(z)) <= 0
    #            +I * acos(z)   otherwise
    a, b = mpc_acos(z, prec, rnd)
    if b[0] or b == fzero:
        return mpf_neg(b), a
    else:
        return b, mpf_neg(a)

def mpc_atanh(z, prec, rnd=round_fast):
    # atanh(z) = (log(1+z)-log(1-z))/2
    wp = prec + 15
    a = mpc_add(z, mpc_one, wp)
    b = mpc_sub(mpc_one, z, wp)
    a = mpc_log(a, wp)
    b = mpc_log(b, wp)
    v = mpc_shift(mpc_sub(a, b, wp), -1)
    # Subtraction at infinity gives correct imaginary part but
    # wrong real part (should be zero)
    if v[0] == fnan and mpc_is_inf(z):
        v = (fzero, v[1])
    return v

def mpc_fibonacci(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        return (mpf_fibonacci(re, prec, rnd), fzero)
    size = max(abs(re[2]+re[3]), abs(re[2]+re[3]))
    wp = prec + size + 20
    a = mpf_phi(wp)
    b = mpf_add(mpf_shift(a, 1), fnone, wp)
    u = mpc_pow((a, fzero), z, wp)
    v = mpc_cos_pi(z, wp)
    v = mpc_div(v, u, wp)
    u = mpc_sub(u, v, wp)
    u = mpc_div_mpf(u, b, prec, rnd)
    return u

def mpf_expj(x, prec, rnd='f'):
    raise ComplexResult

def mpc_expj(z, prec, rnd='f'):
    re, im = z
    if im == fzero:
        return mpf_cos_sin(re, prec, rnd)
    if re == fzero:
        return mpf_exp(mpf_neg(im), prec, rnd), fzero
    ey = mpf_exp(mpf_neg(im), prec+10)
    c, s = mpf_cos_sin(re, prec+10)
    re = mpf_mul(ey, c, prec, rnd)
    im = mpf_mul(ey, s, prec, rnd)
    return re, im

def mpf_expjpi(x, prec, rnd='f'):
    raise ComplexResult

def mpc_expjpi(z, prec, rnd='f'):
    re, im = z
    if im == fzero:
        return mpf_cos_sin_pi(re, prec, rnd)
    sign, man, exp, bc = im
    wp = prec+10
    if man:
        wp += max(0, exp+bc)
    im = mpf_neg(mpf_mul(mpf_pi(wp), im, wp))
    if re == fzero:
        return mpf_exp(im, prec, rnd), fzero
    ey = mpf_exp(im, prec+10)
    c, s = mpf_cos_sin_pi(re, prec+10)
    re = mpf_mul(ey, c, prec, rnd)
    im = mpf_mul(ey, s, prec, rnd)
    return re, im


if BACKEND == 'sage':
    try:
        import sage.libs.mpmath.ext_libmp as _lbmp
        mpc_exp = _lbmp.mpc_exp
        mpc_sqrt = _lbmp.mpc_sqrt
    except (ImportError, AttributeError):
        print("Warning: Sage imports in libmpc failed")
