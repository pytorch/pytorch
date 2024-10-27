"""
This module implements computation of hypergeometric and related
functions. In particular, it provides code for generic summation
of hypergeometric series. Optimized versions for various special
cases are also provided.
"""

import operator
import math

from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_

from .libintmath import gcd

from .libmpf import (\
    ComplexResult, round_fast, round_nearest,
    negative_rnd, bitcount, to_fixed, from_man_exp, from_int, to_int,
    from_rational,
    fzero, fone, fnone, ftwo, finf, fninf, fnan,
    mpf_sign, mpf_add, mpf_abs, mpf_pos,
    mpf_cmp, mpf_lt, mpf_le, mpf_gt, mpf_min_max,
    mpf_perturb, mpf_neg, mpf_shift, mpf_sub, mpf_mul, mpf_div,
    sqrt_fixed, mpf_sqrt, mpf_rdiv_int, mpf_pow_int,
    to_rational,
)

from .libelefun import (\
    mpf_pi, mpf_exp, mpf_log, pi_fixed, mpf_cos_sin, mpf_cos, mpf_sin,
    mpf_sqrt, agm_fixed,
)

from .libmpc import (\
    mpc_one, mpc_sub, mpc_mul_mpf, mpc_mul, mpc_neg, complex_int_pow,
    mpc_div, mpc_add_mpf, mpc_sub_mpf,
    mpc_log, mpc_add, mpc_pos, mpc_shift,
    mpc_is_infnan, mpc_zero, mpc_sqrt, mpc_abs,
    mpc_mpf_div, mpc_square, mpc_exp
)

from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed

class NoConvergence(Exception):
    pass


#-----------------------------------------------------------------------#
#                                                                       #
#                     Generic hypergeometric series                     #
#                                                                       #
#-----------------------------------------------------------------------#

"""
TODO:

1. proper mpq parsing
2. imaginary z special-cased (also: rational, integer?)
3. more clever handling of series that don't converge because of stupid
   upwards rounding
4. checking for cancellation

"""

def make_hyp_summator(key):
    """
    Returns a function that sums a generalized hypergeometric series,
    for given parameter types (integer, rational, real, complex).

    """
    p, q, param_types, ztype = key

    pstring = "".join(param_types)
    fname = "hypsum_%i_%i_%s_%s_%s" % (p, q, pstring[:p], pstring[p:], ztype)
    #print "generating hypsum", fname

    have_complex_param = 'C' in param_types
    have_complex_arg = ztype == 'C'
    have_complex = have_complex_param or have_complex_arg

    source = []
    add = source.append

    aint = []
    arat = []
    bint = []
    brat = []
    areal = []
    breal = []
    acomplex = []
    bcomplex = []

    #add("wp = prec + 40")
    add("MAX = kwargs.get('maxterms', wp*100)")
    add("HIGH = MPZ_ONE<<epsshift")
    add("LOW = -HIGH")

    # Setup code
    add("SRE = PRE = one = (MPZ_ONE << wp)")
    if have_complex:
        add("SIM = PIM = MPZ_ZERO")

    if have_complex_arg:
        add("xsign, xm, xe, xbc = z[0]")
        add("if xsign: xm = -xm")
        add("ysign, ym, ye, ybc = z[1]")
        add("if ysign: ym = -ym")
    else:
        add("xsign, xm, xe, xbc = z")
        add("if xsign: xm = -xm")

    add("offset = xe + wp")
    add("if offset >= 0:")
    add("    ZRE = xm << offset")
    add("else:")
    add("    ZRE = xm >> (-offset)")
    if have_complex_arg:
        add("offset = ye + wp")
        add("if offset >= 0:")
        add("    ZIM = ym << offset")
        add("else:")
        add("    ZIM = ym >> (-offset)")

    for i, flag in enumerate(param_types):
        W = ["A", "B"][i >= p]
        if flag == 'Z':
            ([aint,bint][i >= p]).append(i)
            add("%sINT_%i = coeffs[%i]" % (W, i, i))
        elif flag == 'Q':
            ([arat,brat][i >= p]).append(i)
            add("%sP_%i, %sQ_%i = coeffs[%i]._mpq_" % (W, i, W, i, i))
        elif flag == 'R':
            ([areal,breal][i >= p]).append(i)
            add("xsign, xm, xe, xbc = coeffs[%i]._mpf_" % i)
            add("if xsign: xm = -xm")
            add("offset = xe + wp")
            add("if offset >= 0:")
            add("    %sREAL_%i = xm << offset" % (W, i))
            add("else:")
            add("    %sREAL_%i = xm >> (-offset)" % (W, i))
        elif flag == 'C':
            ([acomplex,bcomplex][i >= p]).append(i)
            add("__re, __im = coeffs[%i]._mpc_" % i)
            add("xsign, xm, xe, xbc = __re")
            add("if xsign: xm = -xm")
            add("ysign, ym, ye, ybc = __im")
            add("if ysign: ym = -ym")

            add("offset = xe + wp")
            add("if offset >= 0:")
            add("    %sCRE_%i = xm << offset" % (W, i))
            add("else:")
            add("    %sCRE_%i = xm >> (-offset)" % (W, i))
            add("offset = ye + wp")
            add("if offset >= 0:")
            add("    %sCIM_%i = ym << offset" % (W, i))
            add("else:")
            add("    %sCIM_%i = ym >> (-offset)" % (W, i))
        else:
            raise ValueError

    l_areal = len(areal)
    l_breal = len(breal)
    cancellable_real = min(l_areal, l_breal)
    noncancellable_real_num = areal[cancellable_real:]
    noncancellable_real_den = breal[cancellable_real:]

    # LOOP
    add("for n in xrange(1,10**8):")

    add("    if n in magnitude_check:")
    add("        p_mag = bitcount(abs(PRE))")
    if have_complex:
        add("        p_mag = max(p_mag, bitcount(abs(PIM)))")
    add("        magnitude_check[n] = wp-p_mag")

    # Real factors
    multiplier = " * ".join(["AINT_#".replace("#", str(i)) for i in aint] + \
                            ["AP_#".replace("#", str(i)) for i in arat] + \
                            ["BQ_#".replace("#", str(i)) for i in brat])

    divisor    = " * ".join(["BINT_#".replace("#", str(i)) for i in bint] + \
                            ["BP_#".replace("#", str(i)) for i in brat] + \
                            ["AQ_#".replace("#", str(i)) for i in arat] + ["n"])

    if multiplier:
        add("    mul = " + multiplier)
    add("    div = " + divisor)

    # Check for singular terms
    add("    if not div:")
    if multiplier:
        add("        if not mul:")
        add("            break")
    add("        raise ZeroDivisionError")

    # Update product
    if have_complex:

        # TODO: when there are several real parameters and just a few complex
        # (maybe just the complex argument), we only need to do about
        # half as many ops if we accumulate the real factor in a single real variable
        for k in range(cancellable_real): add("    PRE = PRE * AREAL_%i // BREAL_%i" % (areal[k], breal[k]))
        for i in noncancellable_real_num: add("    PRE = (PRE * AREAL_#) >> wp".replace("#", str(i)))
        for i in noncancellable_real_den: add("    PRE = (PRE << wp) // BREAL_#".replace("#", str(i)))
        for k in range(cancellable_real): add("    PIM = PIM * AREAL_%i // BREAL_%i" % (areal[k], breal[k]))
        for i in noncancellable_real_num: add("    PIM = (PIM * AREAL_#) >> wp".replace("#", str(i)))
        for i in noncancellable_real_den: add("    PIM = (PIM << wp) // BREAL_#".replace("#", str(i)))

        if multiplier:
            if have_complex_arg:
                add("    PRE, PIM = (mul*(PRE*ZRE-PIM*ZIM))//div, (mul*(PIM*ZRE+PRE*ZIM))//div")
                add("    PRE >>= wp")
                add("    PIM >>= wp")
            else:
                add("    PRE = ((mul * PRE * ZRE) >> wp) // div")
                add("    PIM = ((mul * PIM * ZRE) >> wp) // div")
        else:
            if have_complex_arg:
                add("    PRE, PIM = (PRE*ZRE-PIM*ZIM)//div, (PIM*ZRE+PRE*ZIM)//div")
                add("    PRE >>= wp")
                add("    PIM >>= wp")
            else:
                add("    PRE = ((PRE * ZRE) >> wp) // div")
                add("    PIM = ((PIM * ZRE) >> wp) // div")

        for i in acomplex:
            add("    PRE, PIM = PRE*ACRE_#-PIM*ACIM_#, PIM*ACRE_#+PRE*ACIM_#".replace("#", str(i)))
            add("    PRE >>= wp")
            add("    PIM >>= wp")

        for i in bcomplex:
            add("    mag = BCRE_#*BCRE_#+BCIM_#*BCIM_#".replace("#", str(i)))
            add("    re = PRE*BCRE_# + PIM*BCIM_#".replace("#", str(i)))
            add("    im = PIM*BCRE_# - PRE*BCIM_#".replace("#", str(i)))
            add("    PRE = (re << wp) // mag".replace("#", str(i)))
            add("    PIM = (im << wp) // mag".replace("#", str(i)))

    else:
        for k in range(cancellable_real): add("    PRE = PRE * AREAL_%i // BREAL_%i" % (areal[k], breal[k]))
        for i in noncancellable_real_num: add("    PRE = (PRE * AREAL_#) >> wp".replace("#", str(i)))
        for i in noncancellable_real_den: add("    PRE = (PRE << wp) // BREAL_#".replace("#", str(i)))
        if multiplier:
            add("    PRE = ((PRE * mul * ZRE) >> wp) // div")
        else:
            add("    PRE = ((PRE * ZRE) >> wp) // div")

    # Add product to sum
    if have_complex:
        add("    SRE += PRE")
        add("    SIM += PIM")
        add("    if (HIGH > PRE > LOW) and (HIGH > PIM > LOW):")
        add("        break")
    else:
        add("    SRE += PRE")
        add("    if HIGH > PRE > LOW:")
        add("        break")

    #add("    from mpmath import nprint, log, ldexp")
    #add("    nprint([n, log(abs(PRE),2), ldexp(PRE,-wp)])")

    add("    if n > MAX:")
    add("        raise NoConvergence('Hypergeometric series converges too slowly. Try increasing maxterms.')")

    # +1 all parameters for next loop
    for i in aint:     add("    AINT_# += 1".replace("#", str(i)))
    for i in bint:     add("    BINT_# += 1".replace("#", str(i)))
    for i in arat:     add("    AP_# += AQ_#".replace("#", str(i)))
    for i in brat:     add("    BP_# += BQ_#".replace("#", str(i)))
    for i in areal:    add("    AREAL_# += one".replace("#", str(i)))
    for i in breal:    add("    BREAL_# += one".replace("#", str(i)))
    for i in acomplex: add("    ACRE_# += one".replace("#", str(i)))
    for i in bcomplex: add("    BCRE_# += one".replace("#", str(i)))

    if have_complex:
        add("a = from_man_exp(SRE, -wp, prec, 'n')")
        add("b = from_man_exp(SIM, -wp, prec, 'n')")

        add("if SRE:")
        add("    if SIM:")
        add("        magn = max(a[2]+a[3], b[2]+b[3])")
        add("    else:")
        add("        magn = a[2]+a[3]")
        add("elif SIM:")
        add("    magn = b[2]+b[3]")
        add("else:")
        add("    magn = -wp+1")

        add("return (a, b), True, magn")
    else:
        add("a = from_man_exp(SRE, -wp, prec, 'n')")

        add("if SRE:")
        add("    magn = a[2]+a[3]")
        add("else:")
        add("    magn = -wp+1")

        add("return a, False, magn")

    source = "\n".join(("    " + line) for line in source)
    source = ("def %s(coeffs, z, prec, wp, epsshift, magnitude_check, **kwargs):\n" % fname) + source

    namespace = {}

    exec_(source, globals(), namespace)

    #print source
    return source, namespace[fname]


if BACKEND == 'sage':

    def make_hyp_summator(key):
        """
        Returns a function that sums a generalized hypergeometric series,
        for given parameter types (integer, rational, real, complex).
        """
        from sage.libs.mpmath.ext_main import hypsum_internal
        p, q, param_types, ztype = key
        def _hypsum(coeffs, z, prec, wp, epsshift, magnitude_check, **kwargs):
            return hypsum_internal(p, q, param_types, ztype, coeffs, z,
                prec, wp, epsshift, magnitude_check, kwargs)

        return "(none)", _hypsum


#-----------------------------------------------------------------------#
#                                                                       #
#                              Error functions                          #
#                                                                       #
#-----------------------------------------------------------------------#

# TODO: mpf_erf should call mpf_erfc when appropriate (currently
#    only the converse delegation is implemented)

def mpf_erf(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero: return fzero
        if x == finf: return fone
        if x== fninf: return fnone
        return fnan
    size = exp + bc
    lg = math.log
    # The approximation erf(x) = 1 is accurate to > x^2 * log(e,2) bits
    if size > 3 and 2*(size-1) + 0.528766 > lg(prec,2):
        if sign:
            return mpf_perturb(fnone, 0, prec, rnd)
        else:
            return mpf_perturb(fone, 1, prec, rnd)
    # erf(x) ~ 2*x/sqrt(pi) close to 0
    if size < -prec:
        # 2*x
        x = mpf_shift(x,1)
        c = mpf_sqrt(mpf_pi(prec+20), prec+20)
        # TODO: interval rounding
        return mpf_div(x, c, prec, rnd)
    wp = prec + abs(size) + 25
    # Taylor series for erf, fixed-point summation
    t = abs(to_fixed(x, wp))
    t2 = (t*t) >> wp
    s, term, k = t, 12345, 1
    while term:
        t = ((t * t2) >> wp) // k
        term = t // (2*k+1)
        if k & 1:
            s -= term
        else:
            s += term
        k += 1
    s = (s << (wp+1)) // sqrt_fixed(pi_fixed(wp), wp)
    if sign:
        s = -s
    return from_man_exp(s, -wp, prec, rnd)

# If possible, we use the asymptotic series for erfc.
# This is an alternating divergent asymptotic series, so
# the error is at most equal to the first omitted term.
# Here we check if the smallest term is small enough
# for a given x and precision
def erfc_check_series(x, prec):
    n = to_int(x)
    if n**2 * 1.44 > prec:
        return True
    return False

def mpf_erfc(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero: return fone
        if x == finf: return fzero
        if x == fninf: return ftwo
        return fnan
    wp = prec + 20
    mag = bc+exp
    # Preserve full accuracy when exponent grows huge
    wp += max(0, 2*mag)
    regular_erf = sign or mag < 2
    if regular_erf or not erfc_check_series(x, wp):
        if regular_erf:
            return mpf_sub(fone, mpf_erf(x, prec+10, negative_rnd[rnd]), prec, rnd)
        # 1-erf(x) ~ exp(-x^2), increase prec to deal with cancellation
        n = to_int(x)+1
        return mpf_sub(fone, mpf_erf(x, prec + int(n**2*1.44) + 10), prec, rnd)
    s = term = MPZ_ONE << wp
    term_prev = 0
    t = (2 * to_fixed(x, wp) ** 2) >> wp
    k = 1
    while 1:
        term = ((term * (2*k - 1)) << wp) // t
        if k > 4 and term > term_prev or not term:
            break
        if k & 1:
            s -= term
        else:
            s += term
        term_prev = term
        #print k, to_str(from_man_exp(term, -wp, 50), 10)
        k += 1
    s = (s << wp) // sqrt_fixed(pi_fixed(wp), wp)
    s = from_man_exp(s, -wp, wp)
    z = mpf_exp(mpf_neg(mpf_mul(x,x,wp),wp),wp)
    y = mpf_div(mpf_mul(z, s, wp), x, prec, rnd)
    return y


#-----------------------------------------------------------------------#
#                                                                       #
#                         Exponential integrals                         #
#                                                                       #
#-----------------------------------------------------------------------#

def ei_taylor(x, prec):
    s = t = x
    k = 2
    while t:
        t = ((t*x) >> prec) // k
        s += t // k
        k += 1
    return s

def complex_ei_taylor(zre, zim, prec):
    _abs = abs
    sre = tre = zre
    sim = tim = zim
    k = 2
    while _abs(tre) + _abs(tim) > 5:
        tre, tim = ((tre*zre-tim*zim)//k)>>prec, ((tre*zim+tim*zre)//k)>>prec
        sre += tre // k
        sim += tim // k
        k += 1
    return sre, sim

def ei_asymptotic(x, prec):
    one = MPZ_ONE << prec
    x = t = ((one << prec) // x)
    s = one + x
    k = 2
    while t:
        t = (k*t*x) >> prec
        s += t
        k += 1
    return s

def complex_ei_asymptotic(zre, zim, prec):
    _abs = abs
    one = MPZ_ONE << prec
    M = (zim*zim + zre*zre) >> prec
    # 1 / z
    xre = tre = (zre << prec) // M
    xim = tim = ((-zim) << prec) // M
    sre = one + xre
    sim = xim
    k = 2
    while _abs(tre) + _abs(tim) > 1000:
        #print tre, tim
        tre, tim = ((tre*xre-tim*xim)*k)>>prec, ((tre*xim+tim*xre)*k)>>prec
        sre += tre
        sim += tim
        k += 1
        if k > prec:
            raise NoConvergence
    return sre, sim

def mpf_ei(x, prec, rnd=round_fast, e1=False):
    if e1:
        x = mpf_neg(x)
    sign, man, exp, bc = x
    if e1 and not sign:
        if x == fzero:
            return finf
        raise ComplexResult("E1(x) for x < 0")
    if man:
        xabs = 0, man, exp, bc
        xmag = exp+bc
        wp = prec + 20
        can_use_asymp = xmag > wp
        if not can_use_asymp:
            if exp >= 0:
                xabsint = man << exp
            else:
                xabsint = man >> (-exp)
            can_use_asymp = xabsint > int(wp*0.693) + 10
        if can_use_asymp:
            if xmag > wp:
                v = fone
            else:
                v = from_man_exp(ei_asymptotic(to_fixed(x, wp), wp), -wp)
            v = mpf_mul(v, mpf_exp(x, wp), wp)
            v = mpf_div(v, x, prec, rnd)
        else:
            wp += 2*int(to_int(xabs))
            u = to_fixed(x, wp)
            v = ei_taylor(u, wp) + euler_fixed(wp)
            t1 = from_man_exp(v,-wp)
            t2 = mpf_log(xabs,wp)
            v = mpf_add(t1, t2, prec, rnd)
    else:
        if x == fzero: v = fninf
        elif x == finf: v = finf
        elif x == fninf: v = fzero
        else: v = fnan
    if e1:
        v = mpf_neg(v)
    return v

def mpc_ei(z, prec, rnd=round_fast, e1=False):
    if e1:
        z = mpc_neg(z)
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero:
        if e1:
            x = mpf_neg(mpf_ei(a, prec, rnd))
            if not asign:
                y = mpf_neg(mpf_pi(prec, rnd))
            else:
                y = fzero
            return x, y
        else:
            return mpf_ei(a, prec, rnd), fzero
    if a != fzero:
        if not aman or not bman:
            return (fnan, fnan)
    wp = prec + 40
    amag = aexp+abc
    bmag = bexp+bbc
    zmag = max(amag, bmag)
    can_use_asymp = zmag > wp
    if not can_use_asymp:
        zabsint = abs(to_int(a)) + abs(to_int(b))
        can_use_asymp = zabsint > int(wp*0.693) + 20
    try:
        if can_use_asymp:
            if zmag > wp:
                v = fone, fzero
            else:
                zre = to_fixed(a, wp)
                zim = to_fixed(b, wp)
                vre, vim = complex_ei_asymptotic(zre, zim, wp)
                v = from_man_exp(vre, -wp), from_man_exp(vim, -wp)
            v = mpc_mul(v, mpc_exp(z, wp), wp)
            v = mpc_div(v, z, wp)
            if e1:
                v = mpc_neg(v, prec, rnd)
            else:
                x, y = v
                if bsign:
                    v = mpf_pos(x, prec, rnd), mpf_sub(y, mpf_pi(wp), prec, rnd)
                else:
                    v = mpf_pos(x, prec, rnd), mpf_add(y, mpf_pi(wp), prec, rnd)
            return v
    except NoConvergence:
        pass
    #wp += 2*max(0,zmag)
    wp += 2*int(to_int(mpc_abs(z, 5)))
    zre = to_fixed(a, wp)
    zim = to_fixed(b, wp)
    vre, vim = complex_ei_taylor(zre, zim, wp)
    vre += euler_fixed(wp)
    v = from_man_exp(vre,-wp), from_man_exp(vim,-wp)
    if e1:
        u = mpc_log(mpc_neg(z),wp)
    else:
        u = mpc_log(z,wp)
    v = mpc_add(v, u, prec, rnd)
    if e1:
        v = mpc_neg(v)
    return v

def mpf_e1(x, prec, rnd=round_fast):
    return mpf_ei(x, prec, rnd, True)

def mpc_e1(x, prec, rnd=round_fast):
    return mpc_ei(x, prec, rnd, True)

def mpf_expint(n, x, prec, rnd=round_fast, gamma=False):
    """
    E_n(x), n an integer, x real

    With gamma=True, computes Gamma(n,x)   (upper incomplete gamma function)

    Returns (real, None) if real, otherwise (real, imag)
    The imaginary part is an optional branch cut term

    """
    sign, man, exp, bc = x
    if not man:
        if gamma:
            if x == fzero:
                # Actually gamma function pole
                if n <= 0:
                    return finf, None
                return mpf_gamma_int(n, prec, rnd), None
            if x == finf:
                return fzero, None
            # TODO: could return finite imaginary value at -inf
            return fnan, fnan
        else:
            if x == fzero:
                if n > 1:
                    return from_rational(1, n-1, prec, rnd), None
                else:
                    return finf, None
            if x == finf:
                return fzero, None
            return fnan, fnan
    n_orig = n
    if gamma:
        n = 1-n
    wp = prec + 20
    xmag = exp + bc
    # Beware of near-poles
    if xmag < -10:
        raise NotImplementedError
    nmag = bitcount(abs(n))
    have_imag = n > 0 and sign
    negx = mpf_neg(x)
    # Skip series if direct convergence
    if n == 0 or 2*nmag - xmag < -wp:
        if gamma:
            v = mpf_exp(negx, wp)
            re = mpf_mul(v, mpf_pow_int(x, n_orig-1, wp), prec, rnd)
        else:
            v = mpf_exp(negx, wp)
            re = mpf_div(v, x, prec, rnd)
    else:
        # Finite number of terms, or...
        can_use_asymptotic_series = -3*wp < n <= 0
        # ...large enough?
        if not can_use_asymptotic_series:
            xi = abs(to_int(x))
            m = min(max(1, xi-n), 2*wp)
            siz = -n*nmag + (m+n)*bitcount(abs(m+n)) - m*xmag - (144*m//100)
            tol = -wp-10
            can_use_asymptotic_series = siz < tol
        if can_use_asymptotic_series:
            r = ((-MPZ_ONE) << (wp+wp)) // to_fixed(x, wp)
            m = n
            t = r*m
            s = MPZ_ONE << wp
            while m and t:
                s += t
                m += 1
                t = (m*r*t) >> wp
            v = mpf_exp(negx, wp)
            if gamma:
                # ~ exp(-x) * x^(n-1) * (1 + ...)
                v = mpf_mul(v, mpf_pow_int(x, n_orig-1, wp), wp)
            else:
                # ~ exp(-x)/x * (1 + ...)
                v = mpf_div(v, x, wp)
            re = mpf_mul(v, from_man_exp(s, -wp), prec, rnd)
        elif n == 1:
            re = mpf_neg(mpf_ei(negx, prec, rnd))
        elif n > 0 and n < 3*wp:
            T1 = mpf_neg(mpf_ei(negx, wp))
            if gamma:
                if n_orig & 1:
                    T1 = mpf_neg(T1)
            else:
                T1 = mpf_mul(T1, mpf_pow_int(negx, n-1, wp), wp)
            r = t = to_fixed(x, wp)
            facs = [1] * (n-1)
            for k in range(1,n-1):
                facs[k] = facs[k-1] * k
            facs = facs[::-1]
            s = facs[0] << wp
            for k in range(1, n-1):
                if k & 1:
                    s -= facs[k] * t
                else:
                    s += facs[k] * t
                t = (t*r) >> wp
            T2 = from_man_exp(s, -wp, wp)
            T2 = mpf_mul(T2, mpf_exp(negx, wp))
            if gamma:
                T2 = mpf_mul(T2, mpf_pow_int(x, n_orig, wp), wp)
            R = mpf_add(T1, T2)
            re = mpf_div(R, from_int(ifac(n-1)), prec, rnd)
        else:
            raise NotImplementedError
    if have_imag:
        M = from_int(-ifac(n-1))
        if gamma:
            im = mpf_div(mpf_pi(wp), M, prec, rnd)
            if n_orig & 1:
                im = mpf_neg(im)
        else:
            im = mpf_div(mpf_mul(mpf_pi(wp), mpf_pow_int(negx, n_orig-1, wp), wp), M, prec, rnd)
        return re, im
    else:
        return re, None

def mpf_ci_si_taylor(x, wp, which=0):
    """
    0 - Ci(x) - (euler+log(x))
    1 - Si(x)
    """
    x = to_fixed(x, wp)
    x2 = -(x*x) >> wp
    if which == 0:
        s, t, k = 0, (MPZ_ONE<<wp), 2
    else:
        s, t, k = x, x, 3
    while t:
        t = (t*x2//(k*(k-1)))>>wp
        s += t//k
        k += 2
    return from_man_exp(s, -wp)

def mpc_ci_si_taylor(re, im, wp, which=0):
    # The following code is only designed for small arguments,
    # and not too small arguments (for relative accuracy)
    if re[1]:
        mag = re[2]+re[3]
    elif im[1]:
        mag = im[2]+im[3]
    if im[1]:
        mag = max(mag, im[2]+im[3])
    if mag > 2 or mag < -wp:
        raise NotImplementedError
    wp += (2-mag)
    zre = to_fixed(re, wp)
    zim = to_fixed(im, wp)
    z2re = (zim*zim-zre*zre)>>wp
    z2im = (-2*zre*zim)>>wp
    tre = zre
    tim = zim
    one = MPZ_ONE<<wp
    if which == 0:
        sre, sim, tre, tim, k = 0, 0, (MPZ_ONE<<wp), 0, 2
    else:
        sre, sim, tre, tim, k = zre, zim, zre, zim, 3
    while max(abs(tre), abs(tim)) > 2:
        f = k*(k-1)
        tre, tim = ((tre*z2re-tim*z2im)//f)>>wp, ((tre*z2im+tim*z2re)//f)>>wp
        sre += tre//k
        sim += tim//k
        k += 2
    return from_man_exp(sre, -wp), from_man_exp(sim, -wp)

def mpf_ci_si(x, prec, rnd=round_fast, which=2):
    """
    Calculation of Ci(x), Si(x) for real x.

    which = 0 -- returns (Ci(x), -)
    which = 1 -- returns (Si(x), -)
    which = 2 -- returns (Ci(x), Si(x))

    Note: if x < 0, Ci(x) needs an additional imaginary term, pi*i.
    """
    wp = prec + 20
    sign, man, exp, bc = x
    ci, si = None, None
    if not man:
        if x == fzero:
            return (fninf, fzero)
        if x == fnan:
            return (x, x)
        ci = fzero
        if which != 0:
            if x == finf:
                si = mpf_shift(mpf_pi(prec, rnd), -1)
            if x == fninf:
                si = mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))
        return (ci, si)
    # For small x: Ci(x) ~ euler + log(x), Si(x) ~ x
    mag = exp+bc
    if mag < -wp:
        if which != 0:
            si = mpf_perturb(x, 1-sign, prec, rnd)
        if which != 1:
            y = mpf_euler(wp)
            xabs = mpf_abs(x)
            ci = mpf_add(y, mpf_log(xabs, wp), prec, rnd)
        return ci, si
    # For huge x: Ci(x) ~ sin(x)/x, Si(x) ~ pi/2
    elif mag > wp:
        if which != 0:
            if sign:
                si = mpf_neg(mpf_pi(prec, negative_rnd[rnd]))
            else:
                si = mpf_pi(prec, rnd)
            si = mpf_shift(si, -1)
        if which != 1:
            ci = mpf_div(mpf_sin(x, wp), x, prec, rnd)
        return ci, si
    else:
        wp += abs(mag)
    # Use an asymptotic series? The smallest value of n!/x^n
    # occurs for n ~ x, where the magnitude is ~ exp(-x).
    asymptotic = mag-1 > math.log(wp, 2)
    # Case 1: convergent series near 0
    if not asymptotic:
        if which != 0:
            si = mpf_pos(mpf_ci_si_taylor(x, wp, 1), prec, rnd)
        if which != 1:
            ci = mpf_ci_si_taylor(x, wp, 0)
            ci = mpf_add(ci, mpf_euler(wp), wp)
            ci = mpf_add(ci, mpf_log(mpf_abs(x), wp), prec, rnd)
        return ci, si
    x = mpf_abs(x)
    # Case 2: asymptotic series for x >> 1
    xf = to_fixed(x, wp)
    xr = (MPZ_ONE<<(2*wp)) // xf   # 1/x
    s1 = (MPZ_ONE << wp)
    s2 = xr
    t = xr
    k = 2
    while t:
        t = -t
        t = (t*xr*k)>>wp
        k += 1
        s1 += t
        t = (t*xr*k)>>wp
        k += 1
        s2 += t
    s1 = from_man_exp(s1, -wp)
    s2 = from_man_exp(s2, -wp)
    s1 = mpf_div(s1, x, wp)
    s2 = mpf_div(s2, x, wp)
    cos, sin = mpf_cos_sin(x, wp)
    # Ci(x) = sin(x)*s1-cos(x)*s2
    # Si(x) = pi/2-cos(x)*s1-sin(x)*s2
    if which != 0:
        si = mpf_add(mpf_mul(cos, s1), mpf_mul(sin, s2), wp)
        si = mpf_sub(mpf_shift(mpf_pi(wp), -1), si, wp)
        if sign:
            si = mpf_neg(si)
        si = mpf_pos(si, prec, rnd)
    if which != 1:
        ci = mpf_sub(mpf_mul(sin, s1), mpf_mul(cos, s2), prec, rnd)
    return ci, si

def mpf_ci(x, prec, rnd=round_fast):
    if mpf_sign(x) < 0:
        raise ComplexResult
    return mpf_ci_si(x, prec, rnd, 0)[0]

def mpf_si(x, prec, rnd=round_fast):
    return mpf_ci_si(x, prec, rnd, 1)[1]

def mpc_ci(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        ci = mpf_ci_si(re, prec, rnd, 0)[0]
        if mpf_sign(re) < 0:
            return (ci, mpf_pi(prec, rnd))
        return (ci, fzero)
    wp = prec + 20
    cre, cim = mpc_ci_si_taylor(re, im, wp, 0)
    cre = mpf_add(cre, mpf_euler(wp), wp)
    ci = mpc_add((cre, cim), mpc_log(z, wp), prec, rnd)
    return ci

def mpc_si(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        return (mpf_ci_si(re, prec, rnd, 1)[1], fzero)
    wp = prec + 20
    z = mpc_ci_si_taylor(re, im, wp, 1)
    return mpc_pos(z, prec, rnd)


#-----------------------------------------------------------------------#
#                                                                       #
#                             Bessel functions                          #
#                                                                       #
#-----------------------------------------------------------------------#

# A Bessel function of the first kind of integer order, J_n(x), is
# given by the power series

#             oo
#             ___         k         2 k + n
#            \        (-1)     / x \
#    J_n(x) = )    ----------- | - |
#            /___  k! (k + n)! \ 2 /
#            k = 0

# Simplifying the quotient between two successive terms gives the
# ratio x^2 / (-4*k*(k+n)). Hence, we only need one full-precision
# multiplication and one division by a small integer per term.
# The complex version is very similar, the only difference being
# that the multiplication is actually 4 multiplies.

# In the general case, we have
# J_v(x) = (x/2)**v / v! * 0F1(v+1, (-1/4)*z**2)

# TODO: for extremely large x, we could use an asymptotic
# trigonometric approximation.

# TODO: recompute at higher precision if the fixed-point mantissa
# is very small

def mpf_besseljn(n, x, prec, rounding=round_fast):
    prec += 50
    negate = n < 0 and n & 1
    mag = x[2]+x[3]
    n = abs(n)
    wp = prec + 20 + n*bitcount(n)
    if mag < 0:
        wp -= n * mag
    x = to_fixed(x, wp)
    x2 = (x**2) >> wp
    if not n:
        s = t = MPZ_ONE << wp
    else:
        s = t = (x**n // ifac(n)) >> ((n-1)*wp + n)
    k = 1
    while t:
        t = ((t * x2) // (-4*k*(k+n))) >> wp
        s += t
        k += 1
    if negate:
        s = -s
    return from_man_exp(s, -wp, prec, rounding)

def mpc_besseljn(n, z, prec, rounding=round_fast):
    negate = n < 0 and n & 1
    n = abs(n)
    origprec = prec
    zre, zim = z
    mag = max(zre[2]+zre[3], zim[2]+zim[3])
    prec += 20 + n*bitcount(n) + abs(mag)
    if mag < 0:
        prec -= n * mag
    zre = to_fixed(zre, prec)
    zim = to_fixed(zim, prec)
    z2re = (zre**2 - zim**2) >> prec
    z2im = (zre*zim) >> (prec-1)
    if not n:
        sre = tre = MPZ_ONE << prec
        sim = tim = MPZ_ZERO
    else:
        re, im = complex_int_pow(zre, zim, n)
        sre = tre = (re // ifac(n)) >> ((n-1)*prec + n)
        sim = tim = (im // ifac(n)) >> ((n-1)*prec + n)
    k = 1
    while abs(tre) + abs(tim) > 3:
        p = -4*k*(k+n)
        tre, tim = tre*z2re - tim*z2im, tim*z2re + tre*z2im
        tre = (tre // p) >> prec
        tim = (tim // p) >> prec
        sre += tre
        sim += tim
        k += 1
    if negate:
        sre = -sre
        sim = -sim
    re = from_man_exp(sre, -prec, origprec, rounding)
    im = from_man_exp(sim, -prec, origprec, rounding)
    return (re, im)

def mpf_agm(a, b, prec, rnd=round_fast):
    """
    Computes the arithmetic-geometric mean agm(a,b) for
    nonnegative mpf values a, b.
    """
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if asign or bsign:
        raise ComplexResult("agm of a negative number")
    # Handle inf, nan or zero in either operand
    if not (aman and bman):
        if a == fnan or b == fnan:
            return fnan
        if a == finf:
            if b == fzero:
                return fnan
            return finf
        if b == finf:
            if a == fzero:
                return fnan
            return finf
        # agm(0,x) = agm(x,0) = 0
        return fzero
    wp = prec + 20
    amag = aexp+abc
    bmag = bexp+bbc
    mag_delta = amag - bmag
    # Reduce to roughly the same magnitude using floating-point AGM
    abs_mag_delta = abs(mag_delta)
    if abs_mag_delta > 10:
        while abs_mag_delta > 10:
            a, b = mpf_shift(mpf_add(a,b,wp),-1), \
                mpf_sqrt(mpf_mul(a,b,wp),wp)
            abs_mag_delta //= 2
        asign, aman, aexp, abc = a
        bsign, bman, bexp, bbc = b
        amag = aexp+abc
        bmag = bexp+bbc
        mag_delta = amag - bmag
    #print to_float(a), to_float(b)
    # Use agm(a,b) = agm(x*a,x*b)/x to obtain a, b ~= 1
    min_mag = min(amag,bmag)
    max_mag = max(amag,bmag)
    n = 0
    # If too small, we lose precision when going to fixed-point
    if min_mag < -8:
        n = -min_mag
    # If too large, we waste time using fixed-point with large numbers
    elif max_mag > 20:
        n = -max_mag
    if n:
        a = mpf_shift(a, n)
        b = mpf_shift(b, n)
    #print to_float(a), to_float(b)
    af = to_fixed(a, wp)
    bf = to_fixed(b, wp)
    g = agm_fixed(af, bf, wp)
    return from_man_exp(g, -wp-n, prec, rnd)

def mpf_agm1(a, prec, rnd=round_fast):
    """
    Computes the arithmetic-geometric mean agm(1,a) for a nonnegative
    mpf value a.
    """
    return mpf_agm(fone, a, prec, rnd)

def mpc_agm(a, b, prec, rnd=round_fast):
    """
    Complex AGM.

    TODO:
    * check that convergence works as intended
    * optimize
    * select a nonarbitrary branch
    """
    if mpc_is_infnan(a) or mpc_is_infnan(b):
        return fnan, fnan
    if mpc_zero in (a, b):
        return fzero, fzero
    if mpc_neg(a) == b:
        return fzero, fzero
    wp = prec+20
    eps = mpf_shift(fone, -wp+10)
    while 1:
        a1 = mpc_shift(mpc_add(a, b, wp), -1)
        b1 = mpc_sqrt(mpc_mul(a, b, wp), wp)
        a, b = a1, b1
        size = mpf_min_max([mpc_abs(a,10), mpc_abs(b,10)])[1]
        err = mpc_abs(mpc_sub(a, b, 10), 10)
        if size == fzero or mpf_lt(err, mpf_mul(eps, size)):
            return a

def mpc_agm1(a, prec, rnd=round_fast):
    return mpc_agm(mpc_one, a, prec, rnd)

def mpf_ellipk(x, prec, rnd=round_fast):
    if not x[1]:
        if x == fzero:
            return mpf_shift(mpf_pi(prec, rnd), -1)
        if x == fninf:
            return fzero
        if x == fnan:
            return x
    if x == fone:
        return finf
    # TODO: for |x| << 1/2, one could use fall back to
    # pi/2 * hyp2f1_rat((1,2),(1,2),(1,1), x)
    wp = prec + 15
    # Use K(x) = pi/2/agm(1,a) where a = sqrt(1-x)
    # The sqrt raises ComplexResult if x > 0
    a = mpf_sqrt(mpf_sub(fone, x, wp), wp)
    v = mpf_agm1(a, wp)
    r = mpf_div(mpf_pi(wp), v, prec, rnd)
    return mpf_shift(r, -1)

def mpc_ellipk(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        if re == finf:
            return mpc_zero
        if mpf_le(re, fone):
            return mpf_ellipk(re, prec, rnd), fzero
    wp = prec + 15
    a = mpc_sqrt(mpc_sub(mpc_one, z, wp), wp)
    v = mpc_agm1(a, wp)
    r = mpc_mpf_div(mpf_pi(wp), v, prec, rnd)
    return mpc_shift(r, -1)

def mpf_ellipe(x, prec, rnd=round_fast):
    # http://functions.wolfram.com/EllipticIntegrals/
    # EllipticK/20/01/0001/
    # E = (1-m)*(K'(m)*2*m + K(m))
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return mpf_shift(mpf_pi(prec, rnd), -1)
        if x == fninf:
            return finf
        if x == fnan:
            return x
        if x == finf:
            raise ComplexResult
    if x == fone:
        return fone
    wp = prec+20
    mag = exp+bc
    if mag < -wp:
        return mpf_shift(mpf_pi(prec, rnd), -1)
    # Compute a finite difference for K'
    p = max(mag, 0) - wp
    h = mpf_shift(fone, p)
    K = mpf_ellipk(x, 2*wp)
    Kh = mpf_ellipk(mpf_sub(x, h), 2*wp)
    Kdiff = mpf_shift(mpf_sub(K, Kh), -p)
    t = mpf_sub(fone, x)
    b = mpf_mul(Kdiff, mpf_shift(x,1), wp)
    return mpf_mul(t, mpf_add(K, b), prec, rnd)

def mpc_ellipe(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        if re == finf:
            return (fzero, finf)
        if mpf_le(re, fone):
            return mpf_ellipe(re, prec, rnd), fzero
    wp = prec + 15
    mag = mpc_abs(z, 1)
    p = max(mag[2]+mag[3], 0) - wp
    h = mpf_shift(fone, p)
    K = mpc_ellipk(z, 2*wp)
    Kh = mpc_ellipk(mpc_add_mpf(z, h, 2*wp), 2*wp)
    Kdiff = mpc_shift(mpc_sub(Kh, K, wp), -p)
    t = mpc_sub(mpc_one, z, wp)
    b = mpc_mul(Kdiff, mpc_shift(z,1), wp)
    return mpc_mul(t, mpc_add(K, b, wp), prec, rnd)
