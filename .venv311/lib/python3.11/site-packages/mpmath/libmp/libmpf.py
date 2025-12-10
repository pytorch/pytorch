"""
Low-level functions for arbitrary-precision floating-point arithmetic.
"""

__docformat__ = 'plaintext'

import math

from bisect import bisect

import sys

# Importing random is slow
#from random import getrandbits
getrandbits = None

from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
    BACKEND, STRICT, HASH_MODULUS, HASH_BITS, gmpy, sage, sage_utils)

from .libintmath import (giant_steps,
    trailtable, bctable, lshift, rshift, bitcount, trailing,
    sqrt_fixed, numeral, isqrt, isqrt_fast, sqrtrem,
    bin_to_radix)

# We don't pickle tuples directly for the following reasons:
#   1: pickle uses str() for ints, which is inefficient when they are large
#   2: pickle doesn't work for gmpy mpzs
# Both problems are solved by using hex()

if BACKEND == 'sage':
    def to_pickable(x):
        sign, man, exp, bc = x
        return sign, hex(man), exp, bc
else:
    def to_pickable(x):
        sign, man, exp, bc = x
        return sign, hex(man)[2:], exp, bc

def from_pickable(x):
    sign, man, exp, bc = x
    return (sign, MPZ(man, 16), exp, bc)

class ComplexResult(ValueError):
    pass

try:
    intern
except NameError:
    intern = lambda x: x

# All supported rounding modes
round_nearest = intern('n')
round_floor = intern('f')
round_ceiling = intern('c')
round_up = intern('u')
round_down = intern('d')
round_fast = round_down

def prec_to_dps(n):
    """Return number of accurate decimals that can be represented
    with a precision of n bits."""
    return max(1, int(round(int(n)/3.3219280948873626)-1))

def dps_to_prec(n):
    """Return the number of bits required to represent n decimals
    accurately."""
    return max(1, int(round((int(n)+1)*3.3219280948873626)))

def repr_dps(n):
    """Return the number of decimal digits required to represent
    a number with n-bit precision so that it can be uniquely
    reconstructed from the representation."""
    dps = prec_to_dps(n)
    if dps == 15:
        return 17
    return dps + 3

#----------------------------------------------------------------------------#
#                    Some commonly needed float values                       #
#----------------------------------------------------------------------------#

# Regular number format:
# (-1)**sign * mantissa * 2**exponent, plus bitcount of mantissa
fzero = (0, MPZ_ZERO, 0, 0)
fnzero = (1, MPZ_ZERO, 0, 0)
fone = (0, MPZ_ONE, 0, 1)
fnone = (1, MPZ_ONE, 0, 1)
ftwo = (0, MPZ_ONE, 1, 1)
ften = (0, MPZ_FIVE, 1, 3)
fhalf = (0, MPZ_ONE, -1, 1)

# Arbitrary encoding for special numbers: zero mantissa, nonzero exponent
fnan = (0, MPZ_ZERO, -123, -1)
finf = (0, MPZ_ZERO, -456, -2)
fninf = (1, MPZ_ZERO, -789, -3)

# Was 1e1000; this is broken in Python 2.4
math_float_inf = 1e300 * 1e300


#----------------------------------------------------------------------------#
#                                  Rounding                                  #
#----------------------------------------------------------------------------#

# This function can be used to round a mantissa generally. However,
# we will try to do most rounding inline for efficiency.
def round_int(x, n, rnd):
    if rnd == round_nearest:
        if x >= 0:
            t = x >> (n-1)
            if t & 1 and ((t & 2) or (x & h_mask[n<300][n])):
                return (t>>1)+1
            else:
                return t>>1
        else:
            return -round_int(-x, n, rnd)
    if rnd == round_floor:
        return x >> n
    if rnd == round_ceiling:
        return -((-x) >> n)
    if rnd == round_down:
        if x >= 0:
            return x >> n
        return -((-x) >> n)
    if rnd == round_up:
        if x >= 0:
            return -((-x) >> n)
        return x >> n

# These masks are used to pick out segments of numbers to determine
# which direction to round when rounding to nearest.
class h_mask_big:
    def __getitem__(self, n):
        return (MPZ_ONE<<(n-1))-1

h_mask_small = [0]+[((MPZ_ONE<<(_-1))-1) for _ in range(1, 300)]
h_mask = [h_mask_big(), h_mask_small]

# The >> operator rounds to floor. shifts_down[rnd][sign]
# tells whether this is the right direction to use, or if the
# number should be negated before shifting
shifts_down = {round_floor:(1,0), round_ceiling:(0,1),
    round_down:(1,1), round_up:(0,0)}


#----------------------------------------------------------------------------#
#                          Normalization of raw mpfs                         #
#----------------------------------------------------------------------------#

# This function is called almost every time an mpf is created.
# It has been optimized accordingly.

def _normalize(sign, man, exp, bc, prec, rnd):
    """
    Create a raw mpf tuple with value (-1)**sign * man * 2**exp and
    normalized mantissa. The mantissa is rounded in the specified
    direction if its size exceeds the precision. Trailing zero bits
    are also stripped from the mantissa to ensure that the
    representation is canonical.

    Conditions on the input:
    * The input must represent a regular (finite) number
    * The sign bit must be 0 or 1
    * The mantissa must be positive
    * The exponent must be an integer
    * The bitcount must be exact

    If these conditions are not met, use from_man_exp, mpf_pos, or any
    of the conversion functions to create normalized raw mpf tuples.
    """
    if not man:
        return fzero
    # Cut mantissa down to size if larger than target precision
    n = bc - prec
    if n > 0:
        if rnd == round_nearest:
            t = man >> (n-1)
            if t & 1 and ((t & 2) or (man & h_mask[n<300][n])):
                man = (t>>1)+1
            else:
                man = t>>1
        elif shifts_down[rnd][sign]:
            man >>= n
        else:
            man = -((-man)>>n)
        exp += n
        bc = prec
    # Strip trailing bits
    if not man & 1:
        t = trailtable[int(man & 255)]
        if not t:
            while not man & 255:
                man >>= 8
                exp += 8
                bc -= 8
            t = trailtable[int(man & 255)]
        man >>= t
        exp += t
        bc -= t
    # Bit count can be wrong if the input mantissa was 1 less than
    # a power of 2 and got rounded up, thereby adding an extra bit.
    # With trailing bits removed, all powers of two have mantissa 1,
    # so this is easy to check for.
    if man == 1:
        bc = 1
    return sign, man, exp, bc

def _normalize1(sign, man, exp, bc, prec, rnd):
    """same as normalize, but with the added condition that
       man is odd or zero
    """
    if not man:
        return fzero
    if bc <= prec:
        return sign, man, exp, bc
    n = bc - prec
    if rnd == round_nearest:
        t = man >> (n-1)
        if t & 1 and ((t & 2) or (man & h_mask[n<300][n])):
            man = (t>>1)+1
        else:
            man = t>>1
    elif shifts_down[rnd][sign]:
        man >>= n
    else:
        man = -((-man)>>n)
    exp += n
    bc = prec
    # Strip trailing bits
    if not man & 1:
        t = trailtable[int(man & 255)]
        if not t:
            while not man & 255:
                man >>= 8
                exp += 8
                bc -= 8
            t = trailtable[int(man & 255)]
        man >>= t
        exp += t
        bc -= t
    # Bit count can be wrong if the input mantissa was 1 less than
    # a power of 2 and got rounded up, thereby adding an extra bit.
    # With trailing bits removed, all powers of two have mantissa 1,
    # so this is easy to check for.
    if man == 1:
        bc = 1
    return sign, man, exp, bc

try:
    _exp_types = (int, long)
except NameError:
    _exp_types = (int,)

def strict_normalize(sign, man, exp, bc, prec, rnd):
    """Additional checks on the components of an mpf. Enable tests by setting
       the environment variable MPMATH_STRICT to Y."""
    assert type(man) == MPZ_TYPE
    assert type(bc) in _exp_types
    assert type(exp) in _exp_types
    assert bc == bitcount(man)
    return _normalize(sign, man, exp, bc, prec, rnd)

def strict_normalize1(sign, man, exp, bc, prec, rnd):
    """Additional checks on the components of an mpf. Enable tests by setting
       the environment variable MPMATH_STRICT to Y."""
    assert type(man) == MPZ_TYPE
    assert type(bc) in _exp_types
    assert type(exp) in _exp_types
    assert bc == bitcount(man)
    assert (not man) or (man & 1)
    return _normalize1(sign, man, exp, bc, prec, rnd)

if BACKEND == 'gmpy' and '_mpmath_normalize' in dir(gmpy):
    _normalize = gmpy._mpmath_normalize
    _normalize1 = gmpy._mpmath_normalize

if BACKEND == 'sage':
    _normalize = _normalize1 = sage_utils.normalize

if STRICT:
    normalize = strict_normalize
    normalize1 = strict_normalize1
else:
    normalize = _normalize
    normalize1 = _normalize1

#----------------------------------------------------------------------------#
#                            Conversion functions                            #
#----------------------------------------------------------------------------#

def from_man_exp(man, exp, prec=None, rnd=round_fast):
    """Create raw mpf from (man, exp) pair. The mantissa may be signed.
    If no precision is specified, the mantissa is stored exactly."""
    man = MPZ(man)
    sign = 0
    if man < 0:
        sign = 1
        man = -man
    if man < 1024:
        bc = bctable[int(man)]
    else:
        bc = bitcount(man)
    if not prec:
        if not man:
            return fzero
        if not man & 1:
            if man & 2:
                return (sign, man >> 1, exp + 1, bc - 1)
            t = trailtable[int(man & 255)]
            if not t:
                while not man & 255:
                    man >>= 8
                    exp += 8
                    bc -= 8
                t = trailtable[int(man & 255)]
            man >>= t
            exp += t
            bc -= t
        return (sign, man, exp, bc)
    return normalize(sign, man, exp, bc, prec, rnd)

int_cache = dict((n, from_man_exp(n, 0)) for n in range(-10, 257))

if BACKEND == 'gmpy' and '_mpmath_create' in dir(gmpy):
    from_man_exp = gmpy._mpmath_create

if BACKEND == 'sage':
    from_man_exp = sage_utils.from_man_exp

def from_int(n, prec=0, rnd=round_fast):
    """Create a raw mpf from an integer. If no precision is specified,
    the mantissa is stored exactly."""
    if not prec:
        if n in int_cache:
            return int_cache[n]
    return from_man_exp(n, 0, prec, rnd)

def to_man_exp(s):
    """Return (man, exp) of a raw mpf. Raise an error if inf/nan."""
    sign, man, exp, bc = s
    if (not man) and exp:
        raise ValueError("mantissa and exponent are undefined for %s" % man)
    return man, exp

def to_int(s, rnd=None):
    """Convert a raw mpf to the nearest int. Rounding is done down by
    default (same as int(float) in Python), but can be changed. If the
    input is inf/nan, an exception is raised."""
    sign, man, exp, bc = s
    if (not man) and exp:
        raise ValueError("cannot convert inf or nan to int")
    if exp >= 0:
        if sign:
            return (-man) << exp
        return man << exp
    # Make default rounding fast
    if not rnd:
        if sign:
            return -(man >> (-exp))
        else:
            return man >> (-exp)
    if sign:
        return round_int(-man, -exp, rnd)
    else:
        return round_int(man, -exp, rnd)

def mpf_round_int(s, rnd):
    sign, man, exp, bc = s
    if (not man) and exp:
        return s
    if exp >= 0:
        return s
    mag = exp+bc
    if mag < 1:
        if rnd == round_ceiling:
            if sign: return fzero
            else:    return fone
        elif rnd == round_floor:
            if sign: return fnone
            else:    return fzero
        elif rnd == round_nearest:
            if mag < 0 or man == MPZ_ONE: return fzero
            elif sign: return fnone
            else:      return fone
        else:
            raise NotImplementedError
    return mpf_pos(s, min(bc, mag), rnd)

def mpf_floor(s, prec=0, rnd=round_fast):
    v = mpf_round_int(s, round_floor)
    if prec:
        v = mpf_pos(v, prec, rnd)
    return v

def mpf_ceil(s, prec=0, rnd=round_fast):
    v = mpf_round_int(s, round_ceiling)
    if prec:
        v = mpf_pos(v, prec, rnd)
    return v

def mpf_nint(s, prec=0, rnd=round_fast):
    v = mpf_round_int(s, round_nearest)
    if prec:
        v = mpf_pos(v, prec, rnd)
    return v

def mpf_frac(s, prec=0, rnd=round_fast):
    return mpf_sub(s, mpf_floor(s), prec, rnd)

def from_float(x, prec=53, rnd=round_fast):
    """Create a raw mpf from a Python float, rounding if necessary.
    If prec >= 53, the result is guaranteed to represent exactly the
    same number as the input. If prec is not specified, use prec=53."""
    # frexp only raises an exception for nan on some platforms
    if x != x:
        return fnan
    # in Python2.5 math.frexp gives an exception for float infinity
    # in Python2.6 it returns (float infinity, 0)
    try:
        m, e = math.frexp(x)
    except:
        if x == math_float_inf: return finf
        if x == -math_float_inf: return fninf
        return fnan
    if x == math_float_inf: return finf
    if x == -math_float_inf: return fninf
    return from_man_exp(int(m*(1<<53)), e-53, prec, rnd)

def from_npfloat(x, prec=113, rnd=round_fast):
    """Create a raw mpf from a numpy float, rounding if necessary.
    If prec >= 113, the result is guaranteed to represent exactly the
    same number as the input. If prec is not specified, use prec=113."""
    y = float(x)
    if x == y: # ldexp overflows for float16
        return from_float(y, prec, rnd)
    import numpy as np
    if np.isfinite(x):
        m, e = np.frexp(x)
        return from_man_exp(int(np.ldexp(m, 113)), int(e-113), prec, rnd)
    if np.isposinf(x): return finf
    if np.isneginf(x): return fninf
    return fnan

def from_Decimal(x, prec=None, rnd=round_fast):
    """Create a raw mpf from a Decimal, rounding if necessary.
    If prec is not specified, use the equivalent bit precision
    of the number of significant digits in x."""
    if x.is_nan(): return fnan
    if x.is_infinite(): return fninf if x.is_signed() else finf
    if prec is None:
        prec = int(len(x.as_tuple()[1])*3.3219280948873626)
    return from_str(str(x), prec, rnd)

def to_float(s, strict=False, rnd=round_fast):
    """
    Convert a raw mpf to a Python float. The result is exact if the
    bitcount of s is <= 53 and no underflow/overflow occurs.

    If the number is too large or too small to represent as a regular
    float, it will be converted to inf or 0.0. Setting strict=True
    forces an OverflowError to be raised instead.

    Warning: with a directed rounding mode, the correct nearest representable
    floating-point number in the specified direction might not be computed
    in case of overflow or (gradual) underflow.
    """
    sign, man, exp, bc = s
    if not man:
        if s == fzero: return 0.0
        if s == finf: return math_float_inf
        if s == fninf: return -math_float_inf
        return math_float_inf/math_float_inf
    if bc > 53:
        sign, man, exp, bc = normalize1(sign, man, exp, bc, 53, rnd)
    if sign:
        man = -man
    try:
        return math.ldexp(man, exp)
    except OverflowError:
        if strict:
            raise
        # Overflow to infinity
        if exp + bc > 0:
            if sign:
                return -math_float_inf
            else:
                return math_float_inf
        # Underflow to zero
        return 0.0

def from_rational(p, q, prec, rnd=round_fast):
    """Create a raw mpf from a rational number p/q, round if
    necessary."""
    return mpf_div(from_int(p), from_int(q), prec, rnd)

def to_rational(s):
    """Convert a raw mpf to a rational number. Return integers (p, q)
    such that s = p/q exactly."""
    sign, man, exp, bc = s
    if sign:
        man = -man
    if bc == -1:
        raise ValueError("cannot convert %s to a rational number" % man)
    if exp >= 0:
        return man * (1<<exp), 1
    else:
        return man, 1<<(-exp)

def to_fixed(s, prec):
    """Convert a raw mpf to a fixed-point big integer"""
    sign, man, exp, bc = s
    offset = exp + prec
    if sign:
        if offset >= 0: return (-man) << offset
        else:           return (-man) >> (-offset)
    else:
        if offset >= 0: return man << offset
        else:           return man >> (-offset)


##############################################################################
##############################################################################

#----------------------------------------------------------------------------#
#                       Arithmetic operations, etc.                          #
#----------------------------------------------------------------------------#

def mpf_rand(prec):
    """Return a raw mpf chosen randomly from [0, 1), with prec bits
    in the mantissa."""
    global getrandbits
    if not getrandbits:
        import random
        getrandbits = random.getrandbits
    return from_man_exp(getrandbits(prec), -prec, prec, round_floor)

def mpf_eq(s, t):
    """Test equality of two raw mpfs. This is simply tuple comparison
    unless either number is nan, in which case the result is False."""
    if not s[1] or not t[1]:
        if s == fnan or t == fnan:
            return False
    return s == t

def mpf_hash(s):
    # Duplicate the new hash algorithm introduces in Python 3.2.
    if sys.version_info >= (3, 2):
        ssign, sman, sexp, sbc = s

        # Handle special numbers
        if not sman:
            if s == fnan: return sys.hash_info.nan
            if s == finf: return sys.hash_info.inf
            if s == fninf: return -sys.hash_info.inf
        h = sman % HASH_MODULUS
        if sexp >= 0:
            sexp = sexp % HASH_BITS
        else:
            sexp = HASH_BITS - 1 - ((-1 - sexp) % HASH_BITS)
        h = (h << sexp) % HASH_MODULUS
        if ssign: h = -h
        if h == -1: h = -2
        return int(h)
    else:
        try:
            # Try to be compatible with hash values for floats and ints
            return hash(to_float(s, strict=1))
        except OverflowError:
            # We must unfortunately sacrifice compatibility with ints here.
            # We could do hash(man << exp) when the exponent is positive, but
            # this would cause unreasonable inefficiency for large numbers.
            return hash(s)

def mpf_cmp(s, t):
    """Compare the raw mpfs s and t. Return -1 if s < t, 0 if s == t,
    and 1 if s > t. (Same convention as Python's cmp() function.)"""

    # In principle, a comparison amounts to determining the sign of s-t.
    # A full subtraction is relatively slow, however, so we first try to
    # look at the components.
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t

    # Handle zeros and special numbers
    if not sman or not tman:
        if s == fzero: return -mpf_sign(t)
        if t == fzero: return mpf_sign(s)
        if s == t: return 0
        # Follow same convention as Python's cmp for float nan
        if t == fnan: return 1
        if s == finf: return 1
        if t == fninf: return 1
        return -1
    # Different sides of zero
    if ssign != tsign:
        if not ssign: return 1
        return -1
    # This reduces to direct integer comparison
    if sexp == texp:
        if sman == tman:
            return 0
        if sman > tman:
            if ssign: return -1
            else:     return 1
        else:
            if ssign: return 1
            else:     return -1
    # Check position of the highest set bit in each number. If
    # different, there is certainly an inequality.
    a = sbc + sexp
    b = tbc + texp
    if ssign:
        if a < b: return 1
        if a > b: return -1
    else:
        if a < b: return -1
        if a > b: return 1

    # Both numbers have the same highest bit. Subtract to find
    # how the lower bits compare.
    delta = mpf_sub(s, t, 5, round_floor)
    if delta[0]:
        return -1
    return 1

def mpf_lt(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) < 0

def mpf_le(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) <= 0

def mpf_gt(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) > 0

def mpf_ge(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) >= 0

def mpf_min_max(seq):
    min = max = seq[0]
    for x in seq[1:]:
        if mpf_lt(x, min): min = x
        if mpf_gt(x, max): max = x
    return min, max

def mpf_pos(s, prec=0, rnd=round_fast):
    """Calculate 0+s for a raw mpf (i.e., just round s to the specified
    precision)."""
    if prec:
        sign, man, exp, bc = s
        if (not man) and exp:
            return s
        return normalize1(sign, man, exp, bc, prec, rnd)
    return s

def mpf_neg(s, prec=None, rnd=round_fast):
    """Negate a raw mpf (return -s), rounding the result to the
    specified precision. The prec argument can be omitted to do the
    operation exactly."""
    sign, man, exp, bc = s
    if not man:
        if exp:
            if s == finf: return fninf
            if s == fninf: return finf
        return s
    if not prec:
        return (1-sign, man, exp, bc)
    return normalize1(1-sign, man, exp, bc, prec, rnd)

def mpf_abs(s, prec=None, rnd=round_fast):
    """Return abs(s) of the raw mpf s, rounded to the specified
    precision. The prec argument can be omitted to generate an
    exact result."""
    sign, man, exp, bc = s
    if (not man) and exp:
        if s == fninf:
            return finf
        return s
    if not prec:
        if sign:
            return (0, man, exp, bc)
        return s
    return normalize1(0, man, exp, bc, prec, rnd)

def mpf_sign(s):
    """Return -1, 0, or 1 (as a Python int, not a raw mpf) depending on
    whether s is negative, zero, or positive. (Nan is taken to give 0.)"""
    sign, man, exp, bc = s
    if not man:
        if s == finf: return 1
        if s == fninf: return -1
        return 0
    return (-1) ** sign

def mpf_add(s, t, prec=0, rnd=round_fast, _sub=0):
    """
    Add the two raw mpf values s and t.

    With prec=0, no rounding is performed. Note that this can
    produce a very large mantissa (potentially too large to fit
    in memory) if exponents are far apart.
    """
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    tsign ^= _sub
    # Standard case: two nonzero, regular numbers
    if sman and tman:
        offset = sexp - texp
        if offset:
            if offset > 0:
                # Outside precision range; only need to perturb
                if offset > 100 and prec:
                    delta = sbc + sexp - tbc - texp
                    if delta > prec + 4:
                        offset = prec + 4
                        sman <<= offset
                        if tsign == ssign: sman += 1
                        else:              sman -= 1
                        return normalize1(ssign, sman, sexp-offset,
                            bitcount(sman), prec, rnd)
                # Add
                if ssign == tsign:
                    man = tman + (sman << offset)
                # Subtract
                else:
                    if ssign: man = tman - (sman << offset)
                    else:     man = (sman << offset) - tman
                    if man >= 0:
                        ssign = 0
                    else:
                        man = -man
                        ssign = 1
                bc = bitcount(man)
                return normalize1(ssign, man, texp, bc, prec or bc, rnd)
            elif offset < 0:
                # Outside precision range; only need to perturb
                if offset < -100 and prec:
                    delta = tbc + texp - sbc - sexp
                    if delta > prec + 4:
                        offset = prec + 4
                        tman <<= offset
                        if ssign == tsign: tman += 1
                        else:              tman -= 1
                        return normalize1(tsign, tman, texp-offset,
                            bitcount(tman), prec, rnd)
                # Add
                if ssign == tsign:
                    man = sman + (tman << -offset)
                # Subtract
                else:
                    if tsign: man = sman - (tman << -offset)
                    else:     man = (tman << -offset) - sman
                    if man >= 0:
                        ssign = 0
                    else:
                        man = -man
                        ssign = 1
                bc = bitcount(man)
                return normalize1(ssign, man, sexp, bc, prec or bc, rnd)
        # Equal exponents; no shifting necessary
        if ssign == tsign:
            man = tman + sman
        else:
            if ssign: man = tman - sman
            else:     man = sman - tman
            if man >= 0:
                ssign = 0
            else:
                man = -man
                ssign = 1
        bc = bitcount(man)
        return normalize(ssign, man, texp, bc, prec or bc, rnd)
    # Handle zeros and special numbers
    if _sub:
        t = mpf_neg(t)
    if not sman:
        if sexp:
            if s == t or tman or not texp:
                return s
            return fnan
        if tman:
            return normalize1(tsign, tman, texp, tbc, prec or tbc, rnd)
        return t
    if texp:
        return t
    if sman:
        return normalize1(ssign, sman, sexp, sbc, prec or sbc, rnd)
    return s

def mpf_sub(s, t, prec=0, rnd=round_fast):
    """Return the difference of two raw mpfs, s-t. This function is
    simply a wrapper of mpf_add that changes the sign of t."""
    return mpf_add(s, t, prec, rnd, 1)

def mpf_sum(xs, prec=0, rnd=round_fast, absolute=False):
    """
    Sum a list of mpf values efficiently and accurately
    (typically no temporary roundoff occurs). If prec=0,
    the final result will not be rounded either.

    There may be roundoff error or cancellation if extremely
    large exponent differences occur.

    With absolute=True, sums the absolute values.
    """
    man = 0
    exp = 0
    max_extra_prec = prec*2 or 1000000  # XXX
    special = None
    for x in xs:
        xsign, xman, xexp, xbc = x
        if xman:
            if xsign and not absolute:
                xman = -xman
            delta = xexp - exp
            if xexp >= exp:
                # x much larger than existing sum?
                # first: quick test
                if (delta > max_extra_prec) and \
                    ((not man) or delta-bitcount(abs(man)) > max_extra_prec):
                    man = xman
                    exp = xexp
                else:
                    man += (xman << delta)
            else:
                delta = -delta
                # x much smaller than existing sum?
                if delta-xbc > max_extra_prec:
                    if not man:
                        man, exp = xman, xexp
                else:
                    man = (man << delta) + xman
                    exp = xexp
        elif xexp:
            if absolute:
                x = mpf_abs(x)
            special = mpf_add(special or fzero, x, 1)
    # Will be inf or nan
    if special:
        return special
    return from_man_exp(man, exp, prec, rnd)

def gmpy_mpf_mul(s, t, prec=0, rnd=round_fast):
    """Multiply two raw mpfs"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    sign = ssign ^ tsign
    man = sman*tman
    if man:
        bc = bitcount(man)
        if prec:
            return normalize1(sign, man, sexp+texp, bc, prec, rnd)
        else:
            return (sign, man, sexp+texp, bc)
    s_special = (not sman) and sexp
    t_special = (not tman) and texp
    if not s_special and not t_special:
        return fzero
    if fnan in (s, t): return fnan
    if (not tman) and texp: s, t = t, s
    if t == fzero: return fnan
    return {1:finf, -1:fninf}[mpf_sign(s) * mpf_sign(t)]

def gmpy_mpf_mul_int(s, n, prec, rnd=round_fast):
    """Multiply by a Python integer."""
    sign, man, exp, bc = s
    if not man:
        return mpf_mul(s, from_int(n), prec, rnd)
    if not n:
        return fzero
    if n < 0:
        sign ^= 1
        n = -n
    man *= n
    return normalize(sign, man, exp, bitcount(man), prec, rnd)

def python_mpf_mul(s, t, prec=0, rnd=round_fast):
    """Multiply two raw mpfs"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    sign = ssign ^ tsign
    man = sman*tman
    if man:
        bc = sbc + tbc - 1
        bc += int(man>>bc)
        if prec:
            return normalize1(sign, man, sexp+texp, bc, prec, rnd)
        else:
            return (sign, man, sexp+texp, bc)
    s_special = (not sman) and sexp
    t_special = (not tman) and texp
    if not s_special and not t_special:
        return fzero
    if fnan in (s, t): return fnan
    if (not tman) and texp: s, t = t, s
    if t == fzero: return fnan
    return {1:finf, -1:fninf}[mpf_sign(s) * mpf_sign(t)]

def python_mpf_mul_int(s, n, prec, rnd=round_fast):
    """Multiply by a Python integer."""
    sign, man, exp, bc = s
    if not man:
        return mpf_mul(s, from_int(n), prec, rnd)
    if not n:
        return fzero
    if n < 0:
        sign ^= 1
        n = -n
    man *= n
    # Generally n will be small
    if n < 1024:
        bc += bctable[int(n)] - 1
    else:
        bc += bitcount(n) - 1
    bc += int(man>>bc)
    return normalize(sign, man, exp, bc, prec, rnd)


if BACKEND == 'gmpy':
    mpf_mul = gmpy_mpf_mul
    mpf_mul_int = gmpy_mpf_mul_int
else:
    mpf_mul = python_mpf_mul
    mpf_mul_int = python_mpf_mul_int

def mpf_shift(s, n):
    """Quickly multiply the raw mpf s by 2**n without rounding."""
    sign, man, exp, bc = s
    if not man:
        return s
    return sign, man, exp+n, bc

def mpf_frexp(x):
    """Convert x = y*2**n to (y, n) with abs(y) in [0.5, 1) if nonzero"""
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return (fzero, 0)
        else:
            raise ValueError
    return mpf_shift(x, -bc-exp), bc+exp

def mpf_div(s, t, prec, rnd=round_fast):
    """Floating-point division"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if not sman or not tman:
        if s == fzero:
            if t == fzero: raise ZeroDivisionError
            if t == fnan: return fnan
            return fzero
        if t == fzero:
            raise ZeroDivisionError
        s_special = (not sman) and sexp
        t_special = (not tman) and texp
        if s_special and t_special:
            return fnan
        if s == fnan or t == fnan:
            return fnan
        if not t_special:
            if t == fzero:
                return fnan
            return {1:finf, -1:fninf}[mpf_sign(s) * mpf_sign(t)]
        return fzero
    sign = ssign ^ tsign
    if tman == 1:
        return normalize1(sign, sman, sexp-texp, sbc, prec, rnd)
    # Same strategy as for addition: if there is a remainder, perturb
    # the result a few bits outside the precision range before rounding
    extra = prec - sbc + tbc + 5
    if extra < 5:
        extra = 5
    quot, rem = divmod(sman<<extra, tman)
    if rem:
        quot = (quot<<1) + 1
        extra += 1
        return normalize1(sign, quot, sexp-texp-extra, bitcount(quot), prec, rnd)
    return normalize(sign, quot, sexp-texp-extra, bitcount(quot), prec, rnd)

def mpf_rdiv_int(n, t, prec, rnd=round_fast):
    """Floating-point division n/t with a Python integer as numerator"""
    sign, man, exp, bc = t
    if not n or not man:
        return mpf_div(from_int(n), t, prec, rnd)
    if n < 0:
        sign ^= 1
        n = -n
    extra = prec + bc + 5
    quot, rem = divmod(n<<extra, man)
    if rem:
        quot = (quot<<1) + 1
        extra += 1
        return normalize1(sign, quot, -exp-extra, bitcount(quot), prec, rnd)
    return normalize(sign, quot, -exp-extra, bitcount(quot), prec, rnd)

def mpf_mod(s, t, prec, rnd=round_fast):
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if ((not sman) and sexp) or ((not tman) and texp):
        return fnan
    # Important special case: do nothing if t is larger
    if ssign == tsign and texp > sexp+sbc:
        return s
    # Another important special case: this allows us to do e.g. x % 1.0
    # to find the fractional part of x, and it will work when x is huge.
    if tman == 1 and sexp > texp+tbc:
        return fzero
    base = min(sexp, texp)
    sman = (-1)**ssign * sman
    tman = (-1)**tsign * tman
    man = (sman << (sexp-base)) % (tman << (texp-base))
    if man >= 0:
        sign = 0
    else:
        man = -man
        sign = 1
    return normalize(sign, man, base, bitcount(man), prec, rnd)

reciprocal_rnd = {
  round_down : round_up,
  round_up : round_down,
  round_floor : round_ceiling,
  round_ceiling : round_floor,
  round_nearest : round_nearest
}

negative_rnd = {
  round_down : round_down,
  round_up : round_up,
  round_floor : round_ceiling,
  round_ceiling : round_floor,
  round_nearest : round_nearest
}

def mpf_pow_int(s, n, prec, rnd=round_fast):
    """Compute s**n, where s is a raw mpf and n is a Python integer."""
    sign, man, exp, bc = s

    if (not man) and exp:
        if s == finf:
            if n > 0: return s
            if n == 0: return fnan
            return fzero
        if s == fninf:
            if n > 0: return [finf, fninf][n & 1]
            if n == 0: return fnan
            return fzero
        return fnan

    n = int(n)
    if n == 0: return fone
    if n == 1: return mpf_pos(s, prec, rnd)
    if n == 2:
        _, man, exp, bc = s
        if not man:
            return fzero
        man = man*man
        if man == 1:
            return (0, MPZ_ONE, exp+exp, 1)
        bc = bc + bc - 2
        bc += bctable[int(man>>bc)]
        return normalize1(0, man, exp+exp, bc, prec, rnd)
    if n == -1: return mpf_div(fone, s, prec, rnd)
    if n < 0:
        inverse = mpf_pow_int(s, -n, prec+5, reciprocal_rnd[rnd])
        return mpf_div(fone, inverse, prec, rnd)

    result_sign = sign & n

    # Use exact integer power when the exact mantissa is small
    if man == 1:
        return (result_sign, MPZ_ONE, exp*n, 1)
    if bc*n < 1000:
        man **= n
        return normalize1(result_sign, man, exp*n, bitcount(man), prec, rnd)

    # Use directed rounding all the way through to maintain rigorous
    # bounds for interval arithmetic
    rounds_down = (rnd == round_nearest) or \
        shifts_down[rnd][result_sign]

    # Now we perform binary exponentiation. Need to estimate precision
    # to avoid rounding errors from temporary operations. Roughly log_2(n)
    # operations are performed.
    workprec = prec + 4*bitcount(n) + 4
    _, pm, pe, pbc = fone
    while 1:
        if n & 1:
            pm = pm*man
            pe = pe+exp
            pbc += bc - 2
            pbc = pbc + bctable[int(pm >> pbc)]
            if pbc > workprec:
                if rounds_down:
                    pm = pm >> (pbc-workprec)
                else:
                    pm = -((-pm) >> (pbc-workprec))
                pe += pbc - workprec
                pbc = workprec
            n -= 1
            if not n:
                break
        man = man*man
        exp = exp+exp
        bc = bc + bc - 2
        bc = bc + bctable[int(man >> bc)]
        if bc > workprec:
            if rounds_down:
                man = man >> (bc-workprec)
            else:
                man = -((-man) >> (bc-workprec))
            exp += bc - workprec
            bc = workprec
        n = n // 2

    return normalize(result_sign, pm, pe, pbc, prec, rnd)


def mpf_perturb(x, eps_sign, prec, rnd):
    """
    For nonzero x, calculate x + eps with directed rounding, where
    eps < prec relatively and eps has the given sign (0 for
    positive, 1 for negative).

    With rounding to nearest, this is taken to simply normalize
    x to the given precision.
    """
    if rnd == round_nearest:
        return mpf_pos(x, prec, rnd)
    sign, man, exp, bc = x
    eps = (eps_sign, MPZ_ONE, exp+bc-prec-1, 1)
    if sign:
        away = (rnd in (round_down, round_ceiling)) ^ eps_sign
    else:
        away = (rnd in (round_up, round_ceiling)) ^ eps_sign
    if away:
        return mpf_add(x, eps, prec, rnd)
    else:
        return mpf_pos(x, prec, rnd)


#----------------------------------------------------------------------------#
#                              Radix conversion                              #
#----------------------------------------------------------------------------#

def to_digits_exp(s, dps):
    """Helper function for representing the floating-point number s as
    a decimal with dps digits. Returns (sign, string, exponent) where
    sign is '' or '-', string is the digit string, and exponent is
    the decimal exponent as an int.

    If inexact, the decimal representation is rounded toward zero."""

    # Extract sign first so it doesn't mess up the string digit count
    if s[0]:
        sign = '-'
        s = mpf_neg(s)
    else:
        sign = ''
    _sign, man, exp, bc = s

    if not man:
        return '', '0', 0

    bitprec = int(dps * math.log(10,2)) + 10

    # Cut down to size
    # TODO: account for precision when doing this
    exp_from_1 = exp + bc
    if abs(exp_from_1) > 3500:
        from .libelefun import mpf_ln2, mpf_ln10
        # Set b = int(exp * log(2)/log(10))
        # If exp is huge, we must use high-precision arithmetic to
        # find the nearest power of ten
        expprec = bitcount(abs(exp)) + 5
        tmp = from_int(exp)
        tmp = mpf_mul(tmp, mpf_ln2(expprec))
        tmp = mpf_div(tmp, mpf_ln10(expprec), expprec)
        b = to_int(tmp)
        s = mpf_div(s, mpf_pow_int(ften, b, bitprec), bitprec)
        _sign, man, exp, bc = s
        exponent = b
    else:
        exponent = 0

    # First, calculate mantissa digits by converting to a binary
    # fixed-point number and then converting that number to
    # a decimal fixed-point number.
    fixprec = max(bitprec - exp - bc, 0)
    fixdps = int(fixprec / math.log(10,2) + 0.5)
    sf = to_fixed(s, fixprec)
    sd = bin_to_radix(sf, fixprec, 10, fixdps)
    digits = numeral(sd, base=10, size=dps)

    exponent += len(digits) - fixdps - 1
    return sign, digits, exponent

def to_str(s, dps, strip_zeros=True, min_fixed=None, max_fixed=None,
    show_zero_exponent=False):
    """
    Convert a raw mpf to a decimal floating-point literal with at
    most `dps` decimal digits in the mantissa (not counting extra zeros
    that may be inserted for visual purposes).

    The number will be printed in fixed-point format if the position
    of the leading digit is strictly between min_fixed
    (default = min(-dps/3,-5)) and max_fixed (default = dps).

    To force fixed-point format always, set min_fixed = -inf,
    max_fixed = +inf. To force floating-point format, set
    min_fixed >= max_fixed.

    The literal is formatted so that it can be parsed back to a number
    by to_str, float() or Decimal().
    """

    # Special numbers
    if not s[1]:
        if s == fzero:
            if dps: t = '0.0'
            else:   t = '.0'
            if show_zero_exponent:
                t += 'e+0'
            return t
        if s == finf: return '+inf'
        if s == fninf: return '-inf'
        if s == fnan: return 'nan'
        raise ValueError

    if min_fixed is None: min_fixed = min(-(dps//3), -5)
    if max_fixed is None: max_fixed = dps

    # to_digits_exp rounds to floor.
    # This sometimes kills some instances of "...00001"
    sign, digits, exponent = to_digits_exp(s, dps+3)

    # No digits: show only .0; round exponent to nearest
    if not dps:
        if digits[0] in '56789':
            exponent += 1
        digits = ".0"

    else:
        # Rounding up kills some instances of "...99999"
        if len(digits) > dps and digits[dps] in '56789':
            digits = digits[:dps]
            i = dps - 1
            while i >= 0 and digits[i] == '9':
                i -= 1
            if i >= 0:
                digits = digits[:i] + str(int(digits[i]) + 1) + '0' * (dps - i - 1)
            else:
                digits = '1' + '0' * (dps - 1)
                exponent += 1
        else:
            digits = digits[:dps]

        # Prettify numbers close to unit magnitude
        if min_fixed < exponent < max_fixed:
            if exponent < 0:
                digits = ("0"*int(-exponent)) + digits
                split = 1
            else:
                split = exponent + 1
                if split > dps:
                    digits += "0"*(split-dps)
            exponent = 0
        else:
            split = 1

        digits = (digits[:split] + "." + digits[split:])

        if strip_zeros:
            # Clean up trailing zeros
            digits = digits.rstrip('0')
            if digits[-1] == ".":
                digits += "0"

    if exponent == 0 and dps and not show_zero_exponent: return sign + digits
    if exponent >= 0: return sign + digits + "e+" + str(exponent)
    if exponent < 0: return sign + digits + "e" + str(exponent)

def str_to_man_exp(x, base=10):
    """Helper function for from_str."""
    x = x.lower().rstrip('l')
    # Verify that the input is a valid float literal
    float(x)
    # Split into mantissa, exponent
    parts = x.split('e')
    if len(parts) == 1:
        exp = 0
    else: # == 2
        x = parts[0]
        exp = int(parts[1])
    # Look for radix point in mantissa
    parts = x.split('.')
    if len(parts) == 2:
        a, b = parts[0], parts[1].rstrip('0')
        exp -= len(b)
        x = a + b
    x = MPZ(int(x, base))
    return x, exp

special_str = {'inf':finf, '+inf':finf, '-inf':fninf, 'nan':fnan}

def from_str(x, prec, rnd=round_fast):
    """Create a raw mpf from a decimal literal, rounding in the
    specified direction if the input number cannot be represented
    exactly as a binary floating-point number with the given number of
    bits. The literal syntax accepted is the same as for Python
    floats.

    TODO: the rounding does not work properly for large exponents.
    """
    x = x.lower().strip()
    if x in special_str:
        return special_str[x]

    if '/' in x:
        p, q = x.split('/')
        p, q = p.rstrip('l'), q.rstrip('l')
        return from_rational(int(p), int(q), prec, rnd)

    man, exp = str_to_man_exp(x, base=10)

    # XXX: appropriate cutoffs & track direction
    # note no factors of 5
    if abs(exp) > 400:
        s = from_int(man, prec+10)
        s = mpf_mul(s, mpf_pow_int(ften, exp, prec+10), prec, rnd)
    else:
        if exp >= 0:
            s = from_int(man * 10**exp, prec, rnd)
        else:
            s = from_rational(man, 10**-exp, prec, rnd)
    return s

# Binary string conversion. These are currently mainly used for debugging
# and could use some improvement in the future

def from_bstr(x):
    man, exp = str_to_man_exp(x, base=2)
    man = MPZ(man)
    sign = 0
    if man < 0:
        man = -man
        sign = 1
    bc = bitcount(man)
    return normalize(sign, man, exp, bc, bc, round_floor)

def to_bstr(x):
    sign, man, exp, bc = x
    return ['','-'][sign] + numeral(man, size=bitcount(man), base=2) + ("e%i" % exp)


#----------------------------------------------------------------------------#
#                                Square roots                                #
#----------------------------------------------------------------------------#


def mpf_sqrt(s, prec, rnd=round_fast):
    """
    Compute the square root of a nonnegative mpf value. The
    result is correctly rounded.
    """
    sign, man, exp, bc = s
    if sign:
        raise ComplexResult("square root of a negative number")
    if not man:
        return s
    if exp & 1:
        exp -= 1
        man <<= 1
        bc += 1
    elif man == 1:
        return normalize1(sign, man, exp//2, bc, prec, rnd)
    shift = max(4, 2*prec-bc+4)
    shift += shift & 1
    if rnd in 'fd':
        man = isqrt(man<<shift)
    else:
        man, rem = sqrtrem(man<<shift)
        # Perturb up
        if rem:
            man = (man<<1)+1
            shift += 2
    return from_man_exp(man, (exp-shift)//2, prec, rnd)

def mpf_hypot(x, y, prec, rnd=round_fast):
    """Compute the Euclidean norm sqrt(x**2 + y**2) of two raw mpfs
    x and y."""
    if y == fzero: return mpf_abs(x, prec, rnd)
    if x == fzero: return mpf_abs(y, prec, rnd)
    hypot2 = mpf_add(mpf_mul(x,x), mpf_mul(y,y), prec+4)
    return mpf_sqrt(hypot2, prec, rnd)


if BACKEND == 'sage':
    try:
        import sage.libs.mpmath.ext_libmp as ext_lib
        mpf_add = ext_lib.mpf_add
        mpf_sub = ext_lib.mpf_sub
        mpf_mul = ext_lib.mpf_mul
        mpf_div = ext_lib.mpf_div
        mpf_sqrt = ext_lib.mpf_sqrt
    except ImportError:
        pass
