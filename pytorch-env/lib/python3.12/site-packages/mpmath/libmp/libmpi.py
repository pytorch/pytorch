"""
Computational functions for interval arithmetic.

"""

from .backend import xrange

from .libmpf import (
    ComplexResult,
    round_down, round_up, round_floor, round_ceiling, round_nearest,
    prec_to_dps, repr_dps, dps_to_prec,
    bitcount,
    from_float,
    fnan, finf, fninf, fzero, fhalf, fone, fnone,
    mpf_sign, mpf_lt, mpf_le, mpf_gt, mpf_ge, mpf_eq, mpf_cmp,
    mpf_min_max,
    mpf_floor, from_int, to_int, to_str, from_str,
    mpf_abs, mpf_neg, mpf_pos, mpf_add, mpf_sub, mpf_mul, mpf_mul_int,
    mpf_div, mpf_shift, mpf_pow_int,
    from_man_exp, MPZ_ONE)

from .libelefun import (
    mpf_log, mpf_exp, mpf_sqrt, mpf_atan, mpf_atan2,
    mpf_pi, mod_pi2, mpf_cos_sin
)

from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma

def mpi_str(s, prec):
    sa, sb = s
    dps = prec_to_dps(prec) + 5
    return "[%s, %s]" % (to_str(sa, dps), to_str(sb, dps))
    #dps = prec_to_dps(prec)
    #m = mpi_mid(s, prec)
    #d = mpf_shift(mpi_delta(s, 20), -1)
    #return "%s +/- %s" % (to_str(m, dps), to_str(d, 3))

mpi_zero = (fzero, fzero)
mpi_one = (fone, fone)

def mpi_eq(s, t):
    return s == t

def mpi_ne(s, t):
    return s != t

def mpi_lt(s, t):
    sa, sb = s
    ta, tb = t
    if mpf_lt(sb, ta): return True
    if mpf_ge(sa, tb): return False
    return None

def mpi_le(s, t):
    sa, sb = s
    ta, tb = t
    if mpf_le(sb, ta): return True
    if mpf_gt(sa, tb): return False
    return None

def mpi_gt(s, t): return mpi_lt(t, s)
def mpi_ge(s, t): return mpi_le(t, s)

def mpi_add(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    a = mpf_add(sa, ta, prec, round_floor)
    b = mpf_add(sb, tb, prec, round_ceiling)
    if a == fnan: a = fninf
    if b == fnan: b = finf
    return a, b

def mpi_sub(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    a = mpf_sub(sa, tb, prec, round_floor)
    b = mpf_sub(sb, ta, prec, round_ceiling)
    if a == fnan: a = fninf
    if b == fnan: b = finf
    return a, b

def mpi_delta(s, prec):
    sa, sb = s
    return mpf_sub(sb, sa, prec, round_up)

def mpi_mid(s, prec):
    sa, sb = s
    return mpf_shift(mpf_add(sa, sb, prec, round_nearest), -1)

def mpi_pos(s, prec):
    sa, sb = s
    a = mpf_pos(sa, prec, round_floor)
    b = mpf_pos(sb, prec, round_ceiling)
    return a, b

def mpi_neg(s, prec=0):
    sa, sb = s
    a = mpf_neg(sb, prec, round_floor)
    b = mpf_neg(sa, prec, round_ceiling)
    return a, b

def mpi_abs(s, prec=0):
    sa, sb = s
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    # Both points nonnegative?
    if sas >= 0:
        a = mpf_pos(sa, prec, round_floor)
        b = mpf_pos(sb, prec, round_ceiling)
    # Upper point nonnegative?
    elif sbs >= 0:
        a = fzero
        negsa = mpf_neg(sa)
        if mpf_lt(negsa, sb):
            b = mpf_pos(sb, prec, round_ceiling)
        else:
            b = mpf_pos(negsa, prec, round_ceiling)
    # Both negative?
    else:
        a = mpf_neg(sb, prec, round_floor)
        b = mpf_neg(sa, prec, round_ceiling)
    return a, b

# TODO: optimize
def mpi_mul_mpf(s, t, prec):
    return mpi_mul(s, (t, t), prec)

def mpi_div_mpf(s, t, prec):
    return mpi_div(s, (t, t), prec)

def mpi_mul(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    tas = mpf_sign(ta)
    tbs = mpf_sign(tb)
    if sas == sbs == 0:
        # Should maybe be undefined
        if ta == fninf or tb == finf:
            return fninf, finf
        return fzero, fzero
    if tas == tbs == 0:
        # Should maybe be undefined
        if sa == fninf or sb == finf:
            return fninf, finf
        return fzero, fzero
    if sas >= 0:
        # positive * positive
        if tas >= 0:
            a = mpf_mul(sa, ta, prec, round_floor)
            b = mpf_mul(sb, tb, prec, round_ceiling)
            if a == fnan: a = fzero
            if b == fnan: b = finf
        # positive * negative
        elif tbs <= 0:
            a = mpf_mul(sb, ta, prec, round_floor)
            b = mpf_mul(sa, tb, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = fzero
        # positive * both signs
        else:
            a = mpf_mul(sb, ta, prec, round_floor)
            b = mpf_mul(sb, tb, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = finf
    elif sbs <= 0:
        # negative * positive
        if tas >= 0:
            a = mpf_mul(sa, tb, prec, round_floor)
            b = mpf_mul(sb, ta, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = fzero
        # negative * negative
        elif tbs <= 0:
            a = mpf_mul(sb, tb, prec, round_floor)
            b = mpf_mul(sa, ta, prec, round_ceiling)
            if a == fnan: a = fzero
            if b == fnan: b = finf
        # negative * both signs
        else:
            a = mpf_mul(sa, tb, prec, round_floor)
            b = mpf_mul(sa, ta, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = finf
    else:
        # General case: perform all cross-multiplications and compare
        # Since the multiplications can be done exactly, we need only
        # do 4 (instead of 8: two for each rounding mode)
        cases = [mpf_mul(sa, ta), mpf_mul(sa, tb), mpf_mul(sb, ta), mpf_mul(sb, tb)]
        if fnan in cases:
            a, b = (fninf, finf)
        else:
            a, b = mpf_min_max(cases)
            a = mpf_pos(a, prec, round_floor)
            b = mpf_pos(b, prec, round_ceiling)
    return a, b

def mpi_square(s, prec=0):
    sa, sb = s
    if mpf_ge(sa, fzero):
        a = mpf_mul(sa, sa, prec, round_floor)
        b = mpf_mul(sb, sb, prec, round_ceiling)
    elif mpf_le(sb, fzero):
        a = mpf_mul(sb, sb, prec, round_floor)
        b = mpf_mul(sa, sa, prec, round_ceiling)
    else:
        sa = mpf_neg(sa)
        sa, sb = mpf_min_max([sa, sb])
        a = fzero
        b = mpf_mul(sb, sb, prec, round_ceiling)
    return a, b

def mpi_div(s, t, prec):
    sa, sb = s
    ta, tb = t
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    tas = mpf_sign(ta)
    tbs = mpf_sign(tb)
    # 0 / X
    if sas == sbs == 0:
        # 0 / <interval containing 0>
        if (tas < 0 and tbs > 0) or (tas == 0 or tbs == 0):
            return fninf, finf
        return fzero, fzero
    # Denominator contains both negative and positive numbers;
    # this should properly be a multi-interval, but the closest
    # match is the entire (extended) real line
    if tas < 0 and tbs > 0:
        return fninf, finf
    # Assume denominator to be nonnegative
    if tas < 0:
        return mpi_div(mpi_neg(s), mpi_neg(t), prec)
    # Division by zero
    # XXX: make sure all results make sense
    if tas == 0:
        # Numerator contains both signs?
        if sas < 0 and sbs > 0:
            return fninf, finf
        if tas == tbs:
            return fninf, finf
        # Numerator positive?
        if sas >= 0:
            a = mpf_div(sa, tb, prec, round_floor)
            b = finf
        if sbs <= 0:
            a = fninf
            b = mpf_div(sb, tb, prec, round_ceiling)
    # Division with positive denominator
    # We still have to handle nans resulting from inf/0 or inf/inf
    else:
        # Nonnegative numerator
        if sas >= 0:
            a = mpf_div(sa, tb, prec, round_floor)
            b = mpf_div(sb, ta, prec, round_ceiling)
            if a == fnan: a = fzero
            if b == fnan: b = finf
        # Nonpositive numerator
        elif sbs <= 0:
            a = mpf_div(sa, ta, prec, round_floor)
            b = mpf_div(sb, tb, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = fzero
        # Numerator contains both signs?
        else:
            a = mpf_div(sa, ta, prec, round_floor)
            b = mpf_div(sb, ta, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = finf
    return a, b

def mpi_pi(prec):
    a = mpf_pi(prec, round_floor)
    b = mpf_pi(prec, round_ceiling)
    return a, b

def mpi_exp(s, prec):
    sa, sb = s
    # exp is monotonic
    a = mpf_exp(sa, prec, round_floor)
    b = mpf_exp(sb, prec, round_ceiling)
    return a, b

def mpi_log(s, prec):
    sa, sb = s
    # log is monotonic
    a = mpf_log(sa, prec, round_floor)
    b = mpf_log(sb, prec, round_ceiling)
    return a, b

def mpi_sqrt(s, prec):
    sa, sb = s
    # sqrt is monotonic
    a = mpf_sqrt(sa, prec, round_floor)
    b = mpf_sqrt(sb, prec, round_ceiling)
    return a, b

def mpi_atan(s, prec):
    sa, sb = s
    a = mpf_atan(sa, prec, round_floor)
    b = mpf_atan(sb, prec, round_ceiling)
    return a, b

def mpi_pow_int(s, n, prec):
    sa, sb = s
    if n < 0:
        return mpi_div((fone, fone), mpi_pow_int(s, -n, prec+20), prec)
    if n == 0:
        return (fone, fone)
    if n == 1:
        return s
    if n == 2:
        return mpi_square(s, prec)
    # Odd -- signs are preserved
    if n & 1:
        a = mpf_pow_int(sa, n, prec, round_floor)
        b = mpf_pow_int(sb, n, prec, round_ceiling)
    # Even -- important to ensure positivity
    else:
        sas = mpf_sign(sa)
        sbs = mpf_sign(sb)
        # Nonnegative?
        if sas >= 0:
            a = mpf_pow_int(sa, n, prec, round_floor)
            b = mpf_pow_int(sb, n, prec, round_ceiling)
        # Nonpositive?
        elif sbs <= 0:
            a = mpf_pow_int(sb, n, prec, round_floor)
            b = mpf_pow_int(sa, n, prec, round_ceiling)
        # Mixed signs?
        else:
            a = fzero
            # max(-a,b)**n
            sa = mpf_neg(sa)
            if mpf_ge(sa, sb):
                b = mpf_pow_int(sa, n, prec, round_ceiling)
            else:
                b = mpf_pow_int(sb, n, prec, round_ceiling)
    return a, b

def mpi_pow(s, t, prec):
    ta, tb = t
    if ta == tb and ta not in (finf, fninf):
        if ta == from_int(to_int(ta)):
            return mpi_pow_int(s, to_int(ta), prec)
        if ta == fhalf:
            return mpi_sqrt(s, prec)
    u = mpi_log(s, prec + 20)
    v = mpi_mul(u, t, prec + 20)
    return mpi_exp(v, prec)

def MIN(x, y):
    if mpf_le(x, y):
        return x
    return y

def MAX(x, y):
    if mpf_ge(x, y):
        return x
    return y

def cos_sin_quadrant(x, wp):
    sign, man, exp, bc = x
    if x == fzero:
        return fone, fzero, 0
    # TODO: combine evaluation code to avoid duplicate modulo
    c, s = mpf_cos_sin(x, wp)
    t, n, wp_ = mod_pi2(man, exp, exp+bc, 15)
    if sign:
        n = -1-n
    return c, s, n

def mpi_cos_sin(x, prec):
    a, b = x
    if a == b == fzero:
        return (fone, fone), (fzero, fzero)
    # Guaranteed to contain both -1 and 1
    if (finf in x) or (fninf in x):
        return (fnone, fone), (fnone, fone)
    wp = prec + 20
    ca, sa, na = cos_sin_quadrant(a, wp)
    cb, sb, nb = cos_sin_quadrant(b, wp)
    ca, cb = mpf_min_max([ca, cb])
    sa, sb = mpf_min_max([sa, sb])
    # Both functions are monotonic within one quadrant
    if na == nb:
        pass
    # Guaranteed to contain both -1 and 1
    elif nb - na >= 4:
        return (fnone, fone), (fnone, fone)
    else:
        # cos has maximum between a and b
        if na//4 != nb//4:
            cb = fone
        # cos has minimum
        if (na-2)//4 != (nb-2)//4:
            ca = fnone
        # sin has maximum
        if (na-1)//4 != (nb-1)//4:
            sb = fone
        # sin has minimum
        if (na-3)//4 != (nb-3)//4:
            sa = fnone
    # Perturb to force interval rounding
    more = from_man_exp((MPZ_ONE<<wp) + (MPZ_ONE<<10), -wp)
    less = from_man_exp((MPZ_ONE<<wp) - (MPZ_ONE<<10), -wp)
    def finalize(v, rounding):
        if bool(v[0]) == (rounding == round_floor):
            p = more
        else:
            p = less
        v = mpf_mul(v, p, prec, rounding)
        sign, man, exp, bc = v
        if exp+bc >= 1:
            if sign:
                return fnone
            return fone
        return v
    ca = finalize(ca, round_floor)
    cb = finalize(cb, round_ceiling)
    sa = finalize(sa, round_floor)
    sb = finalize(sb, round_ceiling)
    return (ca,cb), (sa,sb)

def mpi_cos(x, prec):
    return mpi_cos_sin(x, prec)[0]

def mpi_sin(x, prec):
    return mpi_cos_sin(x, prec)[1]

def mpi_tan(x, prec):
    cos, sin = mpi_cos_sin(x, prec+20)
    return mpi_div(sin, cos, prec)

def mpi_cot(x, prec):
    cos, sin = mpi_cos_sin(x, prec+20)
    return mpi_div(cos, sin, prec)

def mpi_from_str_a_b(x, y, percent, prec):
    wp = prec + 20
    xa = from_str(x, wp, round_floor)
    xb = from_str(x, wp, round_ceiling)
    #ya = from_str(y, wp, round_floor)
    y = from_str(y, wp, round_ceiling)
    assert mpf_ge(y, fzero)
    if percent:
        y = mpf_mul(MAX(mpf_abs(xa), mpf_abs(xb)), y, wp, round_ceiling)
        y = mpf_div(y, from_int(100), wp, round_ceiling)
    a = mpf_sub(xa, y, prec, round_floor)
    b = mpf_add(xb, y, prec, round_ceiling)
    return a, b

def mpi_from_str(s, prec):
    """
    Parse an interval number given as a string.

    Allowed forms are

    "-1.23e-27"
        Any single decimal floating-point literal.
    "a +- b"  or  "a (b)"
        a is the midpoint of the interval and b is the half-width
    "a +- b%"  or  "a (b%)"
        a is the midpoint of the interval and the half-width
        is b percent of a (`a \times b / 100`).
    "[a, b]"
        The interval indicated directly.
    "x[y,z]e"
        x are shared digits, y and z are unequal digits, e is the exponent.

    """
    e = ValueError("Improperly formed interval number '%s'" % s)
    s = s.replace(" ", "")
    wp = prec + 20
    if "+-" in s:
        x, y = s.split("+-")
        return mpi_from_str_a_b(x, y, False, prec)
    # case 2
    elif "(" in s:
        # Don't confuse with a complex number (x,y)
        if s[0] == "(" or ")" not in s:
            raise e
        s = s.replace(")", "")
        percent = False
        if "%" in s:
            if s[-1] != "%":
                raise e
            percent = True
            s = s.replace("%", "")
        x, y = s.split("(")
        return mpi_from_str_a_b(x, y, percent, prec)
    elif "," in s:
        if ('[' not in s) or (']' not in s):
            raise e
        if s[0] == '[':
            # case 3
            s = s.replace("[", "")
            s = s.replace("]", "")
            a, b = s.split(",")
            a = from_str(a, prec, round_floor)
            b = from_str(b, prec, round_ceiling)
            return a, b
        else:
            # case 4
            x, y = s.split('[')
            y, z = y.split(',')
            if 'e' in s:
                z, e = z.split(']')
            else:
                z, e = z.rstrip(']'), ''
            a = from_str(x+y+e, prec, round_floor)
            b = from_str(x+z+e, prec, round_ceiling)
            return a, b
    else:
        a = from_str(s, prec, round_floor)
        b = from_str(s, prec, round_ceiling)
        return a, b

def mpi_to_str(x, dps, use_spaces=True, brackets='[]', mode='brackets', error_dps=4, **kwargs):
    """
    Convert a mpi interval to a string.

    **Arguments**

    *dps*
        decimal places to use for printing
    *use_spaces*
        use spaces for more readable output, defaults to true
    *brackets*
        pair of strings (or two-character string) giving left and right brackets
    *mode*
        mode of display: 'plusminus', 'percent', 'brackets' (default) or 'diff'
    *error_dps*
        limit the error to *error_dps* digits (mode 'plusminus and 'percent')

    Additional keyword arguments are forwarded to the mpf-to-string conversion
    for the components of the output.

    **Examples**

        >>> from mpmath import mpi, mp
        >>> mp.dps = 30
        >>> x = mpi(1, 2)._mpi_
        >>> mpi_to_str(x, 2, mode='plusminus')
        '1.5 +- 0.5'
        >>> mpi_to_str(x, 2, mode='percent')
        '1.5 (33.33%)'
        >>> mpi_to_str(x, 2, mode='brackets')
        '[1.0, 2.0]'
        >>> mpi_to_str(x, 2, mode='brackets' , brackets=('<', '>'))
        '<1.0, 2.0>'
        >>> x = mpi('5.2582327113062393041', '5.2582327113062749951')._mpi_
        >>> mpi_to_str(x, 15, mode='diff')
        '5.2582327113062[4, 7]'
        >>> mpi_to_str(mpi(0)._mpi_, 2, mode='percent')
        '0.0 (0.0%)'

    """
    prec = dps_to_prec(dps)
    wp = prec + 20
    a, b = x
    mid = mpi_mid(x, prec)
    delta = mpi_delta(x, prec)
    a_str = to_str(a, dps, **kwargs)
    b_str = to_str(b, dps, **kwargs)
    mid_str = to_str(mid, dps, **kwargs)
    sp = ""
    if use_spaces:
        sp = " "
    br1, br2 = brackets
    if mode == 'plusminus':
        delta_str = to_str(mpf_shift(delta,-1), dps, **kwargs)
        s = mid_str + sp + "+-" + sp + delta_str
    elif mode == 'percent':
        if mid == fzero:
            p = fzero
        else:
            # p = 100 * delta(x) / (2*mid(x))
            p = mpf_mul(delta, from_int(100))
            p = mpf_div(p, mpf_mul(mid, from_int(2)), wp)
        s = mid_str + sp + "(" + to_str(p, error_dps) + "%)"
    elif mode == 'brackets':
        s = br1 + a_str + "," + sp + b_str + br2
    elif mode == 'diff':
        # use more digits if str(x.a) and str(x.b) are equal
        if a_str == b_str:
            a_str = to_str(a, dps+3, **kwargs)
            b_str = to_str(b, dps+3, **kwargs)
        # separate mantissa and exponent
        a = a_str.split('e')
        if len(a) == 1:
            a.append('')
        b = b_str.split('e')
        if len(b) == 1:
            b.append('')
        if a[1] == b[1]:
            if a[0] != b[0]:
                for i in xrange(len(a[0]) + 1):
                    if a[0][i] != b[0][i]:
                        break
                s = (a[0][:i] + br1 + a[0][i:] + ',' + sp + b[0][i:] + br2
                     + 'e'*min(len(a[1]), 1) + a[1])
            else: # no difference
                s = a[0] + br1 + br2 + 'e'*min(len(a[1]), 1) + a[1]
        else:
            s = br1 + 'e'.join(a) + ',' + sp + 'e'.join(b) + br2
    else:
        raise ValueError("'%s' is unknown mode for printing mpi" % mode)
    return s

def mpci_add(x, y, prec):
    a, b = x
    c, d = y
    return mpi_add(a, c, prec), mpi_add(b, d, prec)

def mpci_sub(x, y, prec):
    a, b = x
    c, d = y
    return mpi_sub(a, c, prec), mpi_sub(b, d, prec)

def mpci_neg(x, prec=0):
    a, b = x
    return mpi_neg(a, prec), mpi_neg(b, prec)

def mpci_pos(x, prec):
    a, b = x
    return mpi_pos(a, prec), mpi_pos(b, prec)

def mpci_mul(x, y, prec):
    # TODO: optimize for real/imag cases
    a, b = x
    c, d = y
    r1 = mpi_mul(a,c)
    r2 = mpi_mul(b,d)
    re = mpi_sub(r1,r2,prec)
    i1 = mpi_mul(a,d)
    i2 = mpi_mul(b,c)
    im = mpi_add(i1,i2,prec)
    return re, im

def mpci_div(x, y, prec):
    # TODO: optimize for real/imag cases
    a, b = x
    c, d = y
    wp = prec+20
    m1 = mpi_square(c)
    m2 = mpi_square(d)
    m = mpi_add(m1,m2,wp)
    re = mpi_add(mpi_mul(a,c), mpi_mul(b,d), wp)
    im = mpi_sub(mpi_mul(b,c), mpi_mul(a,d), wp)
    re = mpi_div(re, m, prec)
    im = mpi_div(im, m, prec)
    return re, im

def mpci_exp(x, prec):
    a, b = x
    wp = prec+20
    r = mpi_exp(a, wp)
    c, s = mpi_cos_sin(b, wp)
    a = mpi_mul(r, c, prec)
    b = mpi_mul(r, s, prec)
    return a, b

def mpi_shift(x, n):
    a, b = x
    return mpf_shift(a,n), mpf_shift(b,n)

def mpi_cosh_sinh(x, prec):
    # TODO: accuracy for small x
    wp = prec+20
    e1 = mpi_exp(x, wp)
    e2 = mpi_div(mpi_one, e1, wp)
    c = mpi_add(e1, e2, prec)
    s = mpi_sub(e1, e2, prec)
    c = mpi_shift(c, -1)
    s = mpi_shift(s, -1)
    return c, s

def mpci_cos(x, prec):
    a, b = x
    wp = prec+10
    c, s = mpi_cos_sin(a, wp)
    ch, sh = mpi_cosh_sinh(b, wp)
    re = mpi_mul(c, ch, prec)
    im = mpi_mul(s, sh, prec)
    return re, mpi_neg(im)

def mpci_sin(x, prec):
    a, b = x
    wp = prec+10
    c, s = mpi_cos_sin(a, wp)
    ch, sh = mpi_cosh_sinh(b, wp)
    re = mpi_mul(s, ch, prec)
    im = mpi_mul(c, sh, prec)
    return re, im

def mpci_abs(x, prec):
    a, b = x
    if a == mpi_zero:
        return mpi_abs(b)
    if b == mpi_zero:
        return mpi_abs(a)
    # Important: nonnegative
    a = mpi_square(a)
    b = mpi_square(b)
    t = mpi_add(a, b, prec+20)
    return mpi_sqrt(t, prec)

def mpi_atan2(y, x, prec):
    ya, yb = y
    xa, xb = x
    # Constrained to the real line
    if ya == yb == fzero:
        if mpf_ge(xa, fzero):
            return mpi_zero
        return mpi_pi(prec)
    # Right half-plane
    if mpf_ge(xa, fzero):
        if mpf_ge(ya, fzero):
            a = mpf_atan2(ya, xb, prec, round_floor)
        else:
            a = mpf_atan2(ya, xa, prec, round_floor)
        if mpf_ge(yb, fzero):
            b = mpf_atan2(yb, xa, prec, round_ceiling)
        else:
            b = mpf_atan2(yb, xb, prec, round_ceiling)
    # Upper half-plane
    elif mpf_ge(ya, fzero):
        b = mpf_atan2(ya, xa, prec, round_ceiling)
        if mpf_le(xb, fzero):
            a = mpf_atan2(yb, xb, prec, round_floor)
        else:
            a = mpf_atan2(ya, xb, prec, round_floor)
    # Lower half-plane
    elif mpf_le(yb, fzero):
        a = mpf_atan2(yb, xa, prec, round_floor)
        if mpf_le(xb, fzero):
            b = mpf_atan2(ya, xb, prec, round_ceiling)
        else:
            b = mpf_atan2(yb, xb, prec, round_ceiling)
    # Covering the origin
    else:
        b = mpf_pi(prec, round_ceiling)
        a = mpf_neg(b)
    return a, b

def mpci_arg(z, prec):
    x, y = z
    return mpi_atan2(y, x, prec)

def mpci_log(z, prec):
    x, y = z
    re = mpi_log(mpci_abs(z, prec+20), prec)
    im = mpci_arg(z, prec)
    return re, im

def mpci_pow(x, y, prec):
    # TODO: recognize/speed up real cases, integer y
    yre, yim = y
    if yim == mpi_zero:
        ya, yb = yre
        if ya == yb:
            sign, man, exp, bc = yb
            if man and exp >= 0:
                return mpci_pow_int(x, (-1)**sign * int(man<<exp), prec)
            # x^0
            if yb == fzero:
                return mpci_pow_int(x, 0, prec)
    wp = prec+20
    return mpci_exp(mpci_mul(y, mpci_log(x, wp), wp), prec)

def mpci_square(x, prec):
    a, b = x
    # (a+bi)^2 = (a^2-b^2) + 2abi
    re = mpi_sub(mpi_square(a), mpi_square(b), prec)
    im = mpi_mul(a, b, prec)
    im = mpi_shift(im, 1)
    return re, im

def mpci_pow_int(x, n, prec):
    if n < 0:
        return mpci_div((mpi_one,mpi_zero), mpci_pow_int(x, -n, prec+20), prec)
    if n == 0:
        return mpi_one, mpi_zero
    if n == 1:
        return mpci_pos(x, prec)
    if n == 2:
        return mpci_square(x, prec)
    wp = prec + 20
    result = (mpi_one, mpi_zero)
    while n:
        if n & 1:
            result = mpci_mul(result, x, wp)
            n -= 1
        x = mpci_square(x, wp)
        n >>= 1
    return mpci_pos(result, prec)

gamma_min_a = from_float(1.46163214496)
gamma_min_b = from_float(1.46163214497)
gamma_min = (gamma_min_a, gamma_min_b)
gamma_mono_imag_a = from_float(-1.1)
gamma_mono_imag_b = from_float(1.1)

def mpi_overlap(x, y):
    a, b = x
    c, d = y
    if mpf_lt(d, a): return False
    if mpf_gt(c, b): return False
    return True

# type = 0 -- gamma
# type = 1 -- factorial
# type = 2 -- 1/gamma
# type = 3 -- log-gamma

def mpi_gamma(z, prec, type=0):
    a, b = z
    wp = prec+20

    if type == 1:
        return mpi_gamma(mpi_add(z, mpi_one, wp), prec, 0)

    # increasing
    if mpf_gt(a, gamma_min_b):
        if type == 0:
            c = mpf_gamma(a, prec, round_floor)
            d = mpf_gamma(b, prec, round_ceiling)
        elif type == 2:
            c = mpf_rgamma(b, prec, round_floor)
            d = mpf_rgamma(a, prec, round_ceiling)
        elif type == 3:
            c = mpf_loggamma(a, prec, round_floor)
            d = mpf_loggamma(b, prec, round_ceiling)
    # decreasing
    elif mpf_gt(a, fzero) and mpf_lt(b, gamma_min_a):
        if type == 0:
            c = mpf_gamma(b, prec, round_floor)
            d = mpf_gamma(a, prec, round_ceiling)
        elif type == 2:
            c = mpf_rgamma(a, prec, round_floor)
            d = mpf_rgamma(b, prec, round_ceiling)
        elif type == 3:
            c = mpf_loggamma(b, prec, round_floor)
            d = mpf_loggamma(a, prec, round_ceiling)
    else:
        # TODO: reflection formula
        znew = mpi_add(z, mpi_one, wp)
        if type == 0: return mpi_div(mpi_gamma(znew, prec+2, 0), z, prec)
        if type == 2: return mpi_mul(mpi_gamma(znew, prec+2, 2), z, prec)
        if type == 3: return mpi_sub(mpi_gamma(znew, prec+2, 3), mpi_log(z, prec+2), prec)
    return c, d

def mpci_gamma(z, prec, type=0):
    (a1,a2), (b1,b2) = z

    # Real case
    if b1 == b2 == fzero and (type != 3 or mpf_gt(a1,fzero)):
        return mpi_gamma(z, prec, type), mpi_zero

    # Estimate precision
    wp = prec+20
    if type != 3:
        amag = a2[2]+a2[3]
        bmag = b2[2]+b2[3]
        if a2 != fzero:
            mag = max(amag, bmag)
        else:
            mag = bmag
        an = abs(to_int(a2))
        bn = abs(to_int(b2))
        absn = max(an, bn)
        gamma_size = max(0,absn*mag)
        wp += bitcount(gamma_size)

    # Assume type != 1
    if type == 1:
        (a1,a2) = mpi_add((a1,a2), mpi_one, wp); z = (a1,a2), (b1,b2)
        type = 0

    # Avoid non-monotonic region near the negative real axis
    if mpf_lt(a1, gamma_min_b):
        if mpi_overlap((b1,b2), (gamma_mono_imag_a, gamma_mono_imag_b)):
            # TODO: reflection formula
            #if mpf_lt(a2, mpf_shift(fone,-1)):
            #    znew = mpci_sub((mpi_one,mpi_zero),z,wp)
            #    ...
            # Recurrence:
            # gamma(z) = gamma(z+1)/z
            znew = mpi_add((a1,a2), mpi_one, wp), (b1,b2)
            if type == 0: return mpci_div(mpci_gamma(znew, prec+2, 0), z, prec)
            if type == 2: return mpci_mul(mpci_gamma(znew, prec+2, 2), z, prec)
            if type == 3: return mpci_sub(mpci_gamma(znew, prec+2, 3), mpci_log(z,prec+2), prec)

    # Use monotonicity (except for a small region close to the
    # origin and near poles)
    # upper half-plane
    if mpf_ge(b1, fzero):
        minre = mpc_loggamma((a1,b2), wp, round_floor)
        maxre = mpc_loggamma((a2,b1), wp, round_ceiling)
        minim = mpc_loggamma((a1,b1), wp, round_floor)
        maxim = mpc_loggamma((a2,b2), wp, round_ceiling)
    # lower half-plane
    elif mpf_le(b2, fzero):
        minre = mpc_loggamma((a1,b1), wp, round_floor)
        maxre = mpc_loggamma((a2,b2), wp, round_ceiling)
        minim = mpc_loggamma((a2,b1), wp, round_floor)
        maxim = mpc_loggamma((a1,b2), wp, round_ceiling)
    # crosses real axis
    else:
        maxre = mpc_loggamma((a2,fzero), wp, round_ceiling)
        # stretches more into the lower half-plane
        if mpf_gt(mpf_neg(b1), b2):
            minre = mpc_loggamma((a1,b1), wp, round_ceiling)
        else:
            minre = mpc_loggamma((a1,b2), wp, round_ceiling)
        minim = mpc_loggamma((a2,b1), wp, round_floor)
        maxim = mpc_loggamma((a2,b2), wp, round_floor)

    w = (minre[0], maxre[0]), (minim[1], maxim[1])
    if type == 3:
        return mpi_pos(w[0], prec), mpi_pos(w[1], prec)
    if type == 2:
        w = mpci_neg(w)
    return mpci_exp(w, prec)

def mpi_loggamma(z, prec): return mpi_gamma(z, prec, type=3)
def mpci_loggamma(z, prec): return mpci_gamma(z, prec, type=3)

def mpi_rgamma(z, prec): return mpi_gamma(z, prec, type=2)
def mpci_rgamma(z, prec): return mpci_gamma(z, prec, type=2)

def mpi_factorial(z, prec): return mpi_gamma(z, prec, type=1)
def mpci_factorial(z, prec): return mpci_gamma(z, prec, type=1)
