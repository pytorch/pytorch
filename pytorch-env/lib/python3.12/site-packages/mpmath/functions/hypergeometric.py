from ..libmp.backend import xrange
from .functions import defun, defun_wrapped

def _check_need_perturb(ctx, terms, prec, discard_known_zeros):
    perturb = recompute = False
    extraprec = 0
    discard = []
    for term_index, term in enumerate(terms):
        w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term
        have_singular_nongamma_weight = False
        # Avoid division by zero in leading factors (TODO:
        # also check for near division by zero?)
        for k, w in enumerate(w_s):
            if not w:
                if ctx.re(c_s[k]) <= 0 and c_s[k]:
                    perturb = recompute = True
                    have_singular_nongamma_weight = True
        pole_count = [0, 0, 0]
        # Check for gamma and series poles and near-poles
        for data_index, data in enumerate([alpha_s, beta_s, b_s]):
            for i, x in enumerate(data):
                n, d = ctx.nint_distance(x)
                # Poles
                if n > 0:
                    continue
                if d == ctx.ninf:
                    # OK if we have a polynomial
                    # ------------------------------
                    ok = False
                    if data_index == 2:
                        for u in a_s:
                            if ctx.isnpint(u) and u >= int(n):
                                ok = True
                                break
                    if ok:
                        continue
                    pole_count[data_index] += 1
                    # ------------------------------
                    #perturb = recompute = True
                    #return perturb, recompute, extraprec
                elif d < -4:
                    extraprec += -d
                    recompute = True
        if discard_known_zeros and pole_count[1] > pole_count[0] + pole_count[2] \
            and not have_singular_nongamma_weight:
            discard.append(term_index)
        elif sum(pole_count):
            perturb = recompute = True
    return perturb, recompute, extraprec, discard

_hypercomb_msg = """
hypercomb() failed to converge to the requested %i bits of accuracy
using a working precision of %i bits. The function value may be zero or
infinite; try passing zeroprec=N or infprec=M to bound finite values between
2^(-N) and 2^M. Otherwise try a higher maxprec or maxterms.
"""

@defun
def hypercomb(ctx, function, params=[], discard_known_zeros=True, **kwargs):
    orig = ctx.prec
    sumvalue = ctx.zero
    dist = ctx.nint_distance
    ninf = ctx.ninf
    orig_params = params[:]
    verbose = kwargs.get('verbose', False)
    maxprec = kwargs.get('maxprec', ctx._default_hyper_maxprec(orig))
    kwargs['maxprec'] = maxprec   # For calls to hypsum
    zeroprec = kwargs.get('zeroprec')
    infprec = kwargs.get('infprec')
    perturbed_reference_value = None
    hextra = 0
    try:
        while 1:
            ctx.prec += 10
            if ctx.prec > maxprec:
                raise ValueError(_hypercomb_msg % (orig, ctx.prec))
            orig2 = ctx.prec
            params = orig_params[:]
            terms = function(*params)
            if verbose:
                print()
                print("ENTERING hypercomb main loop")
                print("prec =", ctx.prec)
                print("hextra", hextra)
            perturb, recompute, extraprec, discard = \
                _check_need_perturb(ctx, terms, orig, discard_known_zeros)
            ctx.prec += extraprec
            if perturb:
                if "hmag" in kwargs:
                    hmag = kwargs["hmag"]
                elif ctx._fixed_precision:
                    hmag = int(ctx.prec*0.3)
                else:
                    hmag = orig + 10 + hextra
                h = ctx.ldexp(ctx.one, -hmag)
                ctx.prec = orig2 + 10 + hmag + 10
                for k in range(len(params)):
                    params[k] += h
                    # Heuristically ensure that the perturbations
                    # are "independent" so that two perturbations
                    # don't accidentally cancel each other out
                    # in a subtraction.
                    h += h/(k+1)
            if recompute:
                terms = function(*params)
            if discard_known_zeros:
                terms = [term for (i, term) in enumerate(terms) if i not in discard]
            if not terms:
                return ctx.zero
            evaluated_terms = []
            for term_index, term_data in enumerate(terms):
                w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term_data
                if verbose:
                    print()
                    print("  Evaluating term %i/%i : %iF%i" % \
                        (term_index+1, len(terms), len(a_s), len(b_s)))
                    print("    powers", ctx.nstr(w_s), ctx.nstr(c_s))
                    print("    gamma", ctx.nstr(alpha_s), ctx.nstr(beta_s))
                    print("    hyper", ctx.nstr(a_s), ctx.nstr(b_s))
                    print("    z", ctx.nstr(z))
                #v = ctx.hyper(a_s, b_s, z, **kwargs)
                #for a in alpha_s: v *= ctx.gamma(a)
                #for b in beta_s: v *= ctx.rgamma(b)
                #for w, c in zip(w_s, c_s): v *= ctx.power(w, c)
                v = ctx.fprod([ctx.hyper(a_s, b_s, z, **kwargs)] + \
                    [ctx.gamma(a) for a in alpha_s] + \
                    [ctx.rgamma(b) for b in beta_s] + \
                    [ctx.power(w,c) for (w,c) in zip(w_s,c_s)])
                if verbose:
                    print("    Value:", v)
                evaluated_terms.append(v)

            if len(terms) == 1 and (not perturb):
                sumvalue = evaluated_terms[0]
                break

            if ctx._fixed_precision:
                sumvalue = ctx.fsum(evaluated_terms)
                break

            sumvalue = ctx.fsum(evaluated_terms)
            term_magnitudes = [ctx.mag(x) for x in evaluated_terms]
            max_magnitude = max(term_magnitudes)
            sum_magnitude = ctx.mag(sumvalue)
            cancellation = max_magnitude - sum_magnitude
            if verbose:
                print()
                print("  Cancellation:", cancellation, "bits")
                print("  Increased precision:", ctx.prec - orig, "bits")

            precision_ok = cancellation < ctx.prec - orig

            if zeroprec is None:
                zero_ok = False
            else:
                zero_ok = max_magnitude - ctx.prec < -zeroprec
            if infprec is None:
                inf_ok = False
            else:
                inf_ok = max_magnitude > infprec

            if precision_ok and (not perturb) or ctx.isnan(cancellation):
                break
            elif precision_ok:
                if perturbed_reference_value is None:
                    hextra += 20
                    perturbed_reference_value = sumvalue
                    continue
                elif ctx.mag(sumvalue - perturbed_reference_value) <= \
                        ctx.mag(sumvalue) - orig:
                    break
                elif zero_ok:
                    sumvalue = ctx.zero
                    break
                elif inf_ok:
                    sumvalue = ctx.inf
                    break
                elif 'hmag' in kwargs:
                    break
                else:
                    hextra *= 2
                    perturbed_reference_value = sumvalue
            # Increase precision
            else:
                increment = min(max(cancellation, orig//2), max(extraprec,orig))
                ctx.prec += increment
                if verbose:
                    print("  Must start over with increased precision")
                continue
    finally:
        ctx.prec = orig
    return +sumvalue

@defun
def hyper(ctx, a_s, b_s, z, **kwargs):
    """
    Hypergeometric function, general case.
    """
    z = ctx.convert(z)
    p = len(a_s)
    q = len(b_s)
    a_s = [ctx._convert_param(a) for a in a_s]
    b_s = [ctx._convert_param(b) for b in b_s]
    # Reduce degree by eliminating common parameters
    if kwargs.get('eliminate', True):
        elim_nonpositive = kwargs.get('eliminate_all', False)
        i = 0
        while i < q and a_s:
            b = b_s[i]
            if b in a_s and (elim_nonpositive or not ctx.isnpint(b[0])):
                a_s.remove(b)
                b_s.remove(b)
                p -= 1
                q -= 1
            else:
                i += 1
    # Handle special cases
    if p == 0:
        if   q == 1: return ctx._hyp0f1(b_s, z, **kwargs)
        elif q == 0: return ctx.exp(z)
    elif p == 1:
        if   q == 1: return ctx._hyp1f1(a_s, b_s, z, **kwargs)
        elif q == 2: return ctx._hyp1f2(a_s, b_s, z, **kwargs)
        elif q == 0: return ctx._hyp1f0(a_s[0][0], z)
    elif p == 2:
        if   q == 1: return ctx._hyp2f1(a_s, b_s, z, **kwargs)
        elif q == 2: return ctx._hyp2f2(a_s, b_s, z, **kwargs)
        elif q == 3: return ctx._hyp2f3(a_s, b_s, z, **kwargs)
        elif q == 0: return ctx._hyp2f0(a_s, b_s, z, **kwargs)
    elif p == q+1:
        return ctx._hypq1fq(p, q, a_s, b_s, z, **kwargs)
    elif p > q+1 and not kwargs.get('force_series'):
        return ctx._hyp_borel(p, q, a_s, b_s, z, **kwargs)
    coeffs, types = zip(*(a_s+b_s))
    return ctx.hypsum(p, q, types, coeffs, z, **kwargs)

@defun
def hyp0f1(ctx,b,z,**kwargs):
    return ctx.hyper([],[b],z,**kwargs)

@defun
def hyp1f1(ctx,a,b,z,**kwargs):
    return ctx.hyper([a],[b],z,**kwargs)

@defun
def hyp1f2(ctx,a1,b1,b2,z,**kwargs):
    return ctx.hyper([a1],[b1,b2],z,**kwargs)

@defun
def hyp2f1(ctx,a,b,c,z,**kwargs):
    return ctx.hyper([a,b],[c],z,**kwargs)

@defun
def hyp2f2(ctx,a1,a2,b1,b2,z,**kwargs):
    return ctx.hyper([a1,a2],[b1,b2],z,**kwargs)

@defun
def hyp2f3(ctx,a1,a2,b1,b2,b3,z,**kwargs):
    return ctx.hyper([a1,a2],[b1,b2,b3],z,**kwargs)

@defun
def hyp2f0(ctx,a,b,z,**kwargs):
    return ctx.hyper([a,b],[],z,**kwargs)

@defun
def hyp3f2(ctx,a1,a2,a3,b1,b2,z,**kwargs):
    return ctx.hyper([a1,a2,a3],[b1,b2],z,**kwargs)

@defun_wrapped
def _hyp1f0(ctx, a, z):
    return (1-z) ** (-a)

@defun
def _hyp0f1(ctx, b_s, z, **kwargs):
    (b, btype), = b_s
    if z:
        magz = ctx.mag(z)
    else:
        magz = 0
    if magz >= 8 and not kwargs.get('force_series'):
        try:
            # http://functions.wolfram.com/HypergeometricFunctions/
            # Hypergeometric0F1/06/02/03/0004/
            # TODO: handle the all-real case more efficiently!
            # TODO: figure out how much precision is needed (exponential growth)
            orig = ctx.prec
            try:
                ctx.prec += 12 + magz//2
                def h():
                    w = ctx.sqrt(-z)
                    jw = ctx.j*w
                    u = 1/(4*jw)
                    c = ctx.mpq_1_2 - b
                    E = ctx.exp(2*jw)
                    T1 = ([-jw,E], [c,-1], [], [], [b-ctx.mpq_1_2, ctx.mpq_3_2-b], [], -u)
                    T2 = ([jw,E], [c,1], [], [], [b-ctx.mpq_1_2, ctx.mpq_3_2-b], [], u)
                    return T1, T2
                v = ctx.hypercomb(h, [], force_series=True)
                v = ctx.gamma(b)/(2*ctx.sqrt(ctx.pi))*v
            finally:
                ctx.prec = orig
            if ctx._is_real_type(b) and ctx._is_real_type(z):
                v = ctx._re(v)
            return +v
        except ctx.NoConvergence:
            pass
    return ctx.hypsum(0, 1, (btype,), [b], z, **kwargs)

@defun
def _hyp1f1(ctx, a_s, b_s, z, **kwargs):
    (a, atype), = a_s
    (b, btype), = b_s
    if not z:
        return ctx.one+z
    magz = ctx.mag(z)
    if magz >= 7 and not (ctx.isint(a) and ctx.re(a) <= 0):
        if ctx.isinf(z):
            if ctx.sign(a) == ctx.sign(b) == ctx.sign(z) == 1:
                return ctx.inf
            return ctx.nan * z
        try:
            try:
                ctx.prec += magz
                sector = ctx._im(z) < 0
                def h(a,b):
                    if sector:
                        E = ctx.expjpi(ctx.fneg(a, exact=True))
                    else:
                        E = ctx.expjpi(a)
                    rz = 1/z
                    T1 = ([E,z], [1,-a], [b], [b-a], [a, 1+a-b], [], -rz)
                    T2 = ([ctx.exp(z),z], [1,a-b], [b], [a], [b-a, 1-a], [], rz)
                    return T1, T2
                v = ctx.hypercomb(h, [a,b], force_series=True)
                if ctx._is_real_type(a) and ctx._is_real_type(b) and ctx._is_real_type(z):
                    v = ctx._re(v)
                return +v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec -= magz
    v = ctx.hypsum(1, 1, (atype, btype), [a, b], z, **kwargs)
    return v

def _hyp2f1_gosper(ctx,a,b,c,z,**kwargs):
    # Use Gosper's recurrence
    # See http://www.math.utexas.edu/pipermail/maxima/2006/000126.html
    _a,_b,_c,_z = a, b, c, z
    orig = ctx.prec
    maxprec = kwargs.get('maxprec', 100*orig)
    extra = 10
    while 1:
        ctx.prec = orig + extra
        #a = ctx.convert(_a)
        #b = ctx.convert(_b)
        #c = ctx.convert(_c)
        z = ctx.convert(_z)
        d = ctx.mpf(0)
        e = ctx.mpf(1)
        f = ctx.mpf(0)
        k = 0
        # Common subexpression elimination, unfortunately making
        # things a bit unreadable. The formula is quite messy to begin
        # with, though...
        abz = a*b*z
        ch = c * ctx.mpq_1_2
        c1h = (c+1) * ctx.mpq_1_2
        nz = 1-z
        g = z/nz
        abg = a*b*g
        cba = c-b-a
        z2 = z-2
        tol = -ctx.prec - 10
        nstr = ctx.nstr
        nprint = ctx.nprint
        mag = ctx.mag
        maxmag = ctx.ninf
        while 1:
            kch = k+ch
            kakbz = (k+a)*(k+b)*z / (4*(k+1)*kch*(k+c1h))
            d1 = kakbz*(e-(k+cba)*d*g)
            e1 = kakbz*(d*abg+(k+c)*e)
            ft = d*(k*(cba*z+k*z2-c)-abz)/(2*kch*nz)
            f1 = f + e - ft
            maxmag = max(maxmag, mag(f1))
            if mag(f1-f) < tol:
                break
            d, e, f = d1, e1, f1
            k += 1
        cancellation = maxmag - mag(f1)
        if cancellation < extra:
            break
        else:
            extra += cancellation
            if extra > maxprec:
                raise ctx.NoConvergence
    return f1

@defun
def _hyp2f1(ctx, a_s, b_s, z, **kwargs):
    (a, atype), (b, btype) = a_s
    (c, ctype), = b_s
    if z == 1:
        # TODO: the following logic can be simplified
        convergent = ctx.re(c-a-b) > 0
        finite = (ctx.isint(a) and a <= 0) or (ctx.isint(b) and b <= 0)
        zerodiv = ctx.isint(c) and c <= 0 and not \
            ((ctx.isint(a) and c <= a <= 0) or (ctx.isint(b) and c <= b <= 0))
        #print "bz", a, b, c, z, convergent, finite, zerodiv
        # Gauss's theorem gives the value if convergent
        if (convergent or finite) and not zerodiv:
            return ctx.gammaprod([c, c-a-b], [c-a, c-b], _infsign=True)
        # Otherwise, there is a pole and we take the
        # sign to be that when approaching from below
        # XXX: this evaluation is not necessarily correct in all cases
        return ctx.hyp2f1(a,b,c,1-ctx.eps*2) * ctx.inf

    # Equal to 1 (first term), unless there is a subsequent
    # division by zero
    if not z:
        # Division by zero but power of z is higher than
        # first order so cancels
        if c or a == 0 or b == 0:
            return 1+z
        # Indeterminate
        return ctx.nan

    # Hit zero denominator unless numerator goes to 0 first
    if ctx.isint(c) and c <= 0:
        if (ctx.isint(a) and c <= a <= 0) or \
           (ctx.isint(b) and c <= b <= 0):
            pass
        else:
            # Pole in series
            return ctx.inf

    absz = abs(z)

    # Fast case: standard series converges rapidly,
    # possibly in finitely many terms
    if absz <= 0.8 or (ctx.isint(a) and a <= 0 and a >= -1000) or \
                      (ctx.isint(b) and b <= 0 and b >= -1000):
        return ctx.hypsum(2, 1, (atype, btype, ctype), [a, b, c], z, **kwargs)

    orig = ctx.prec
    try:
        ctx.prec += 10

        # Use 1/z transformation
        if absz >= 1.3:
            def h(a,b):
                t = ctx.mpq_1-c; ab = a-b; rz = 1/z
                T1 = ([-z],[-a], [c,-ab],[b,c-a], [a,t+a],[ctx.mpq_1+ab],  rz)
                T2 = ([-z],[-b], [c,ab],[a,c-b], [b,t+b],[ctx.mpq_1-ab],  rz)
                return T1, T2
            v = ctx.hypercomb(h, [a,b], **kwargs)

        # Use 1-z transformation
        elif abs(1-z) <= 0.75:
            def h(a,b):
                t = c-a-b; ca = c-a; cb = c-b; rz = 1-z
                T1 = [], [], [c,t], [ca,cb], [a,b], [1-t], rz
                T2 = [rz], [t], [c,a+b-c], [a,b], [ca,cb], [1+t], rz
                return T1, T2
            v = ctx.hypercomb(h, [a,b], **kwargs)

        # Use z/(z-1) transformation
        elif abs(z/(z-1)) <= 0.75:
            v = ctx.hyp2f1(a, c-b, c, z/(z-1)) / (1-z)**a

        # Remaining part of unit circle
        else:
            v = _hyp2f1_gosper(ctx,a,b,c,z,**kwargs)

    finally:
        ctx.prec = orig
    return +v

@defun
def _hypq1fq(ctx, p, q, a_s, b_s, z, **kwargs):
    r"""
    Evaluates 3F2, 4F3, 5F4, ...
    """
    a_s, a_types = zip(*a_s)
    b_s, b_types = zip(*b_s)
    a_s = list(a_s)
    b_s = list(b_s)
    absz = abs(z)
    ispoly = False
    for a in a_s:
        if ctx.isint(a) and a <= 0:
            ispoly = True
            break
    # Direct summation
    if absz < 1 or ispoly:
        try:
            return ctx.hypsum(p, q, a_types+b_types, a_s+b_s, z, **kwargs)
        except ctx.NoConvergence:
            if absz > 1.1 or ispoly:
                raise
    # Use expansion at |z-1| -> 0.
    # Reference: Wolfgang Buhring, "Generalized Hypergeometric Functions at
    #   Unit Argument", Proc. Amer. Math. Soc., Vol. 114, No. 1 (Jan. 1992),
    #   pp.145-153
    # The current implementation has several problems:
    # 1. We only implement it for 3F2. The expansion coefficients are
    #    given by extremely messy nested sums in the higher degree cases
    #    (see reference). Is efficient sequential generation of the coefficients
    #    possible in the > 3F2 case?
    # 2. Although the series converges, it may do so slowly, so we need
    #    convergence acceleration. The acceleration implemented by
    #    nsum does not always help, so results returned are sometimes
    #    inaccurate! Can we do better?
    # 3. We should check conditions for convergence, and possibly
    #    do a better job of cancelling out gamma poles if possible.
    if z == 1:
        # XXX: should also check for division by zero in the
        # denominator of the series (cf. hyp2f1)
        S = ctx.re(sum(b_s)-sum(a_s))
        if S <= 0:
            #return ctx.hyper(a_s, b_s, 1-ctx.eps*2, **kwargs) * ctx.inf
            return ctx.hyper(a_s, b_s, 0.9, **kwargs) * ctx.inf
    if (p,q) == (3,2) and abs(z-1) < 0.05:   # and kwargs.get('sum1')
        #print "Using alternate summation (experimental)"
        a1,a2,a3 = a_s
        b1,b2 = b_s
        u = b1+b2-a3
        initial = ctx.gammaprod([b2-a3,b1-a3,a1,a2],[b2-a3,b1-a3,1,u])
        def term(k, _cache={0:initial}):
            u = b1+b2-a3+k
            if k in _cache:
                t = _cache[k]
            else:
                t = _cache[k-1]
                t *= (b1+k-a3-1)*(b2+k-a3-1)
                t /= k*(u-1)
                _cache[k] = t
            return t * ctx.hyp2f1(a1,a2,u,z)
        try:
            S = ctx.nsum(term, [0,ctx.inf], verbose=kwargs.get('verbose'),
                strict=kwargs.get('strict', True))
            return S * ctx.gammaprod([b1,b2],[a1,a2,a3])
        except ctx.NoConvergence:
            pass
    # Try to use convergence acceleration on and close to the unit circle.
    # Problem: the convergence acceleration degenerates as |z-1| -> 0,
    # except for special cases. Everywhere else, the Shanks transformation
    # is very efficient.
    if absz < 1.1 and ctx._re(z) <= 1:

        def term(kk, _cache={0:ctx.one}):
            k = int(kk)
            if k != kk:
                t = z ** ctx.mpf(kk) / ctx.fac(kk)
                for a in a_s: t *= ctx.rf(a,kk)
                for b in b_s: t /= ctx.rf(b,kk)
                return t
            if k in _cache:
                return _cache[k]
            t = term(k-1)
            m = k-1
            for j in xrange(p): t *= (a_s[j]+m)
            for j in xrange(q): t /= (b_s[j]+m)
            t *= z
            t /= k
            _cache[k] = t
            return t

        sum_method = kwargs.get('sum_method', 'r+s+e')

        try:
            return ctx.nsum(term, [0,ctx.inf], verbose=kwargs.get('verbose'),
                strict=kwargs.get('strict', True),
                method=sum_method.replace('e',''))
        except ctx.NoConvergence:
            if 'e' not in sum_method:
                raise
            pass

        if kwargs.get('verbose'):
            print("Attempting Euler-Maclaurin summation")


        """
        Somewhat slower version (one diffs_exp for each factor).
        However, this would be faster with fast direct derivatives
        of the gamma function.

        def power_diffs(k0):
            r = 0
            l = ctx.log(z)
            while 1:
                yield z**ctx.mpf(k0) * l**r
                r += 1

        def loggamma_diffs(x, reciprocal=False):
            sign = (-1) ** reciprocal
            yield sign * ctx.loggamma(x)
            i = 0
            while 1:
                yield sign * ctx.psi(i,x)
                i += 1

        def hyper_diffs(k0):
            b2 = b_s + [1]
            A = [ctx.diffs_exp(loggamma_diffs(a+k0)) for a in a_s]
            B = [ctx.diffs_exp(loggamma_diffs(b+k0,True)) for b in b2]
            Z = [power_diffs(k0)]
            C = ctx.gammaprod([b for b in b2], [a for a in a_s])
            for d in ctx.diffs_prod(A + B + Z):
                v = C * d
                yield v
        """

        def log_diffs(k0):
            b2 = b_s + [1]
            yield sum(ctx.loggamma(a+k0) for a in a_s) - \
                sum(ctx.loggamma(b+k0) for b in b2) + k0*ctx.log(z)
            i = 0
            while 1:
                v = sum(ctx.psi(i,a+k0) for a in a_s) - \
                    sum(ctx.psi(i,b+k0) for b in b2)
                if i == 0:
                    v += ctx.log(z)
                yield v
                i += 1

        def hyper_diffs(k0):
            C = ctx.gammaprod([b for b in b_s], [a for a in a_s])
            for d in ctx.diffs_exp(log_diffs(k0)):
                v = C * d
                yield v

        tol = ctx.eps / 1024
        prec = ctx.prec
        try:
            trunc = 50 * ctx.dps
            ctx.prec += 20
            for i in xrange(5):
                head = ctx.fsum(term(k) for k in xrange(trunc))
                tail, err = ctx.sumem(term, [trunc, ctx.inf], tol=tol,
                    adiffs=hyper_diffs(trunc),
                    verbose=kwargs.get('verbose'),
                    error=True,
                    _fast_abort=True)
                if err < tol:
                    v = head + tail
                    break
                trunc *= 2
                # Need to increase precision because calculation of
                # derivatives may be inaccurate
                ctx.prec += ctx.prec//2
                if i == 4:
                    raise ctx.NoConvergence(\
                        "Euler-Maclaurin summation did not converge")
        finally:
            ctx.prec = prec
        return +v

    # Use 1/z transformation
    # http://functions.wolfram.com/HypergeometricFunctions/
    #   HypergeometricPFQ/06/01/05/02/0004/
    def h(*args):
        a_s = list(args[:p])
        b_s = list(args[p:])
        Ts = []
        recz = ctx.one/z
        negz = ctx.fneg(z, exact=True)
        for k in range(q+1):
            ak = a_s[k]
            C = [negz]
            Cp = [-ak]
            Gn = b_s + [ak] + [a_s[j]-ak for j in range(q+1) if j != k]
            Gd = a_s + [b_s[j]-ak for j in range(q)]
            Fn = [ak] + [ak-b_s[j]+1 for j in range(q)]
            Fd = [1-a_s[j]+ak for j in range(q+1) if j != k]
            Ts.append((C, Cp, Gn, Gd, Fn, Fd, recz))
        return Ts
    return ctx.hypercomb(h, a_s+b_s, **kwargs)

@defun
def _hyp_borel(ctx, p, q, a_s, b_s, z, **kwargs):
    if a_s:
        a_s, a_types = zip(*a_s)
        a_s = list(a_s)
    else:
        a_s, a_types = [], ()
    if b_s:
        b_s, b_types = zip(*b_s)
        b_s = list(b_s)
    else:
        b_s, b_types = [], ()
    kwargs['maxterms'] = kwargs.get('maxterms', ctx.prec)
    try:
        return ctx.hypsum(p, q, a_types+b_types, a_s+b_s, z, **kwargs)
    except ctx.NoConvergence:
        pass
    prec = ctx.prec
    try:
        tol = kwargs.get('asymp_tol', ctx.eps/4)
        ctx.prec += 10
        # hypsum is has a conservative tolerance. So we try again:
        def term(k, cache={0:ctx.one}):
            if k in cache:
                return cache[k]
            t = term(k-1)
            for a in a_s: t *= (a+(k-1))
            for b in b_s: t /= (b+(k-1))
            t *= z
            t /= k
            cache[k] = t
            return t
        s = ctx.one
        for k in xrange(1, ctx.prec):
            t = term(k)
            s += t
            if abs(t) <= tol:
                return s
    finally:
        ctx.prec = prec
    if p <= q+3:
        contour = kwargs.get('contour')
        if not contour:
            if ctx.arg(z) < 0.25:
                u = z / max(1, abs(z))
                if ctx.arg(z) >= 0:
                    contour = [0, 2j, (2j+2)/u, 2/u, ctx.inf]
                else:
                    contour = [0, -2j, (-2j+2)/u, 2/u, ctx.inf]
                #contour = [0, 2j/z, 2/z, ctx.inf]
                #contour = [0, 2j, 2/z, ctx.inf]
                #contour = [0, 2j, ctx.inf]
            else:
                contour = [0, ctx.inf]
        quad_kwargs = kwargs.get('quad_kwargs', {})
        def g(t):
            return ctx.exp(-t)*ctx.hyper(a_s, b_s+[1], t*z)
        I, err = ctx.quad(g, contour, error=True, **quad_kwargs)
        if err <= abs(I)*ctx.eps*8:
            return I
    raise ctx.NoConvergence


@defun
def _hyp2f2(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), (a2, a2type) = a_s
    (b1, b1type), (b2, b2type) = b_s

    absz = abs(z)
    magz = ctx.mag(z)
    orig = ctx.prec

    # Asymptotic expansion is ~ exp(z)
    asymp_extraprec = magz

    # Asymptotic series is in terms of 3F1
    can_use_asymptotic = (not kwargs.get('force_series')) and \
        (ctx.mag(absz) > 3)

    # TODO: much of the following could be shared with 2F3 instead of
    # copypasted
    if can_use_asymptotic:
        #print "using asymp"
        try:
            try:
                ctx.prec += asymp_extraprec
                # http://functions.wolfram.com/HypergeometricFunctions/
                # Hypergeometric2F2/06/02/02/0002/
                def h(a1,a2,b1,b2):
                    X = a1+a2-b1-b2
                    A2 = a1+a2
                    B2 = b1+b2
                    c = {}
                    c[0] = ctx.one
                    c[1] = (A2-1)*X+b1*b2-a1*a2
                    s1 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = 1-B2+2*a1+a1**2+2*a2+a2**2-A2*B2+a1*a2+b1*b2+(2*B2-3*(A2+1))*k+2*k**2
                            uu2 = (k-A2+b1-1)*(k-A2+b2-1)*(k-X-2)
                            c[k] = ctx.one/k * (uu1*c[k-1]-uu2*c[k-2])
                        t1 = c[k] * z**(-k)
                        if abs(t1) < 0.1*ctx.eps:
                            #print "Convergence :)"
                            break
                        # Quit if the series doesn't converge quickly enough
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            #print "No convergence :("
                            raise ctx.NoConvergence
                        s1 += t1
                        tprev = t1
                        k += 1
                    S = ctx.exp(z)*s1
                    T1 = [z,S], [X,1], [b1,b2],[a1,a2],[],[],0
                    T2 = [-z],[-a1],[b1,b2,a2-a1],[a2,b1-a1,b2-a1],[a1,a1-b1+1,a1-b2+1],[a1-a2+1],-1/z
                    T3 = [-z],[-a2],[b1,b2,a1-a2],[a1,b1-a2,b2-a2],[a2,a2-b1+1,a2-b2+1],[-a1+a2+1],-1/z
                    return T1, T2, T3
                v = ctx.hypercomb(h, [a1,a2,b1,b2], force_series=True, maxterms=4*ctx.prec)
                if sum(ctx._is_real_type(u) for u in [a1,a2,b1,b2,z]) == 5:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig

    return ctx.hypsum(2, 2, (a1type, a2type, b1type, b2type), [a1, a2, b1, b2], z, **kwargs)



@defun
def _hyp1f2(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), = a_s
    (b1, b1type), (b2, b2type) = b_s

    absz = abs(z)
    magz = ctx.mag(z)
    orig = ctx.prec

    # Asymptotic expansion is ~ exp(sqrt(z))
    asymp_extraprec = z and magz//2

    # Asymptotic series is in terms of 3F0
    can_use_asymptotic = (not kwargs.get('force_series')) and \
        (ctx.mag(absz) > 19) and \
        (ctx.sqrt(absz) > 1.5*orig)  # and \
    #   ctx._hyp_check_convergence([a1, a1-b1+1, a1-b2+1], [],
    #                              1/absz, orig+40+asymp_extraprec)

    # TODO: much of the following could be shared with 2F3 instead of
    # copypasted
    if can_use_asymptotic:
        #print "using asymp"
        try:
            try:
                ctx.prec += asymp_extraprec
                # http://functions.wolfram.com/HypergeometricFunctions/
                # Hypergeometric1F2/06/02/03/
                def h(a1,b1,b2):
                    X = ctx.mpq_1_2*(a1-b1-b2+ctx.mpq_1_2)
                    c = {}
                    c[0] = ctx.one
                    c[1] = 2*(ctx.mpq_1_4*(3*a1+b1+b2-2)*(a1-b1-b2)+b1*b2-ctx.mpq_3_16)
                    c[2] = 2*(b1*b2+ctx.mpq_1_4*(a1-b1-b2)*(3*a1+b1+b2-2)-ctx.mpq_3_16)**2+\
                        ctx.mpq_1_16*(-16*(2*a1-3)*b1*b2 + \
                        4*(a1-b1-b2)*(-8*a1**2+11*a1+b1+b2-2)-3)
                    s1 = 0
                    s2 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = (3*k**2+(-6*a1+2*b1+2*b2-4)*k + 3*a1**2 - \
                                (b1-b2)**2 - 2*a1*(b1+b2-2) + ctx.mpq_1_4)
                            uu2 = (k-a1+b1-b2-ctx.mpq_1_2)*(k-a1-b1+b2-ctx.mpq_1_2)*\
                                (k-a1+b1+b2-ctx.mpq_5_2)
                            c[k] = ctx.one/(2*k)*(uu1*c[k-1]-uu2*c[k-2])
                        w = c[k] * (-z)**(-0.5*k)
                        t1 = (-ctx.j)**k * ctx.mpf(2)**(-k) * w
                        t2 = ctx.j**k * ctx.mpf(2)**(-k) * w
                        if abs(t1) < 0.1*ctx.eps:
                            #print "Convergence :)"
                            break
                        # Quit if the series doesn't converge quickly enough
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            #print "No convergence :("
                            raise ctx.NoConvergence
                        s1 += t1
                        s2 += t2
                        tprev = t1
                        k += 1
                    S = ctx.expj(ctx.pi*X+2*ctx.sqrt(-z))*s1 + \
                        ctx.expj(-(ctx.pi*X+2*ctx.sqrt(-z)))*s2
                    T1 = [0.5*S, ctx.pi, -z], [1, -0.5, X], [b1, b2], [a1],\
                        [], [], 0
                    T2 = [-z], [-a1], [b1,b2],[b1-a1,b2-a1], \
                        [a1,a1-b1+1,a1-b2+1], [], 1/z
                    return T1, T2
                v = ctx.hypercomb(h, [a1,b1,b2], force_series=True, maxterms=4*ctx.prec)
                if sum(ctx._is_real_type(u) for u in [a1,b1,b2,z]) == 4:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig

    #print "not using asymp"
    return ctx.hypsum(1, 2, (a1type, b1type, b2type), [a1, b1, b2], z, **kwargs)



@defun
def _hyp2f3(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), (a2, a2type) = a_s
    (b1, b1type), (b2, b2type), (b3, b3type) = b_s

    absz = abs(z)
    magz = ctx.mag(z)

    # Asymptotic expansion is ~ exp(sqrt(z))
    asymp_extraprec = z and magz//2
    orig = ctx.prec

    # Asymptotic series is in terms of 4F1
    # The square root below empirically provides a plausible criterion
    # for the leading series to converge
    can_use_asymptotic = (not kwargs.get('force_series')) and \
        (ctx.mag(absz) > 19) and (ctx.sqrt(absz) > 1.5*orig)

    if can_use_asymptotic:
        #print "using asymp"
        try:
            try:
                ctx.prec += asymp_extraprec
                # http://functions.wolfram.com/HypergeometricFunctions/
                # Hypergeometric2F3/06/02/03/01/0002/
                def h(a1,a2,b1,b2,b3):
                    X = ctx.mpq_1_2*(a1+a2-b1-b2-b3+ctx.mpq_1_2)
                    A2 = a1+a2
                    B3 = b1+b2+b3
                    A = a1*a2
                    B = b1*b2+b3*b2+b1*b3
                    R = b1*b2*b3
                    c = {}
                    c[0] = ctx.one
                    c[1] = 2*(B - A + ctx.mpq_1_4*(3*A2+B3-2)*(A2-B3) - ctx.mpq_3_16)
                    c[2] = ctx.mpq_1_2*c[1]**2 + ctx.mpq_1_16*(-16*(2*A2-3)*(B-A) + 32*R +\
                        4*(-8*A2**2 + 11*A2 + 8*A + B3 - 2)*(A2-B3)-3)
                    s1 = 0
                    s2 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = (k-2*X-3)*(k-2*X-2*b1-1)*(k-2*X-2*b2-1)*\
                                (k-2*X-2*b3-1)
                            uu2 = (4*(k-1)**3 - 6*(4*X+B3)*(k-1)**2 + \
                                2*(24*X**2+12*B3*X+4*B+B3-1)*(k-1) - 32*X**3 - \
                                24*B3*X**2 - 4*B - 8*R - 4*(4*B+B3-1)*X + 2*B3-1)
                            uu3 = (5*(k-1)**2+2*(-10*X+A2-3*B3+3)*(k-1)+2*c[1])
                            c[k] = ctx.one/(2*k)*(uu1*c[k-3]-uu2*c[k-2]+uu3*c[k-1])
                        w = c[k] * ctx.power(-z, -0.5*k)
                        t1 = (-ctx.j)**k * ctx.mpf(2)**(-k) * w
                        t2 = ctx.j**k * ctx.mpf(2)**(-k) * w
                        if abs(t1) < 0.1*ctx.eps:
                            break
                        # Quit if the series doesn't converge quickly enough
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            raise ctx.NoConvergence
                        s1 += t1
                        s2 += t2
                        tprev = t1
                        k += 1
                    S = ctx.expj(ctx.pi*X+2*ctx.sqrt(-z))*s1 + \
                        ctx.expj(-(ctx.pi*X+2*ctx.sqrt(-z)))*s2
                    T1 = [0.5*S, ctx.pi, -z], [1, -0.5, X], [b1, b2, b3], [a1, a2],\
                        [], [], 0
                    T2 = [-z], [-a1], [b1,b2,b3,a2-a1],[a2,b1-a1,b2-a1,b3-a1], \
                        [a1,a1-b1+1,a1-b2+1,a1-b3+1], [a1-a2+1], 1/z
                    T3 = [-z], [-a2], [b1,b2,b3,a1-a2],[a1,b1-a2,b2-a2,b3-a2], \
                        [a2,a2-b1+1,a2-b2+1,a2-b3+1],[-a1+a2+1], 1/z
                    return T1, T2, T3
                v = ctx.hypercomb(h, [a1,a2,b1,b2,b3], force_series=True, maxterms=4*ctx.prec)
                if sum(ctx._is_real_type(u) for u in [a1,a2,b1,b2,b3,z]) == 6:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig

    return ctx.hypsum(2, 3, (a1type, a2type, b1type, b2type, b3type), [a1, a2, b1, b2, b3], z, **kwargs)

@defun
def _hyp2f0(ctx, a_s, b_s, z, **kwargs):
    (a, atype), (b, btype) = a_s
    # We want to try aggressively to use the asymptotic expansion,
    # and fall back only when absolutely necessary
    try:
        kwargsb = kwargs.copy()
        kwargsb['maxterms'] = kwargsb.get('maxterms', ctx.prec)
        return ctx.hypsum(2, 0, (atype,btype), [a,b], z, **kwargsb)
    except ctx.NoConvergence:
        if kwargs.get('force_series'):
            raise
        pass
    def h(a, b):
        w = ctx.sinpi(b)
        rz = -1/z
        T1 = ([ctx.pi,w,rz],[1,-1,a],[],[a-b+1,b],[a],[b],rz)
        T2 = ([-ctx.pi,w,rz],[1,-1,1+a-b],[],[a,2-b],[a-b+1],[2-b],rz)
        return T1, T2
    return ctx.hypercomb(h, [a, 1+a-b], **kwargs)

@defun
def meijerg(ctx, a_s, b_s, z, r=1, series=None, **kwargs):
    an, ap = a_s
    bm, bq = b_s
    n = len(an)
    p = n + len(ap)
    m = len(bm)
    q = m + len(bq)
    a = an+ap
    b = bm+bq
    a = [ctx.convert(_) for _ in a]
    b = [ctx.convert(_) for _ in b]
    z = ctx.convert(z)
    if series is None:
        if p < q: series = 1
        if p > q: series = 2
        if p == q:
            if m+n == p and abs(z) > 1:
                series = 2
            else:
                series = 1
    if kwargs.get('verbose'):
        print("Meijer G m,n,p,q,series =", m,n,p,q,series)
    if series == 1:
        def h(*args):
            a = args[:p]
            b = args[p:]
            terms = []
            for k in range(m):
                bases = [z]
                expts = [b[k]/r]
                gn = [b[j]-b[k] for j in range(m) if j != k]
                gn += [1-a[j]+b[k] for j in range(n)]
                gd = [a[j]-b[k] for j in range(n,p)]
                gd += [1-b[j]+b[k] for j in range(m,q)]
                hn = [1-a[j]+b[k] for j in range(p)]
                hd = [1-b[j]+b[k] for j in range(q) if j != k]
                hz = (-ctx.one)**(p-m-n) * z**(ctx.one/r)
                terms.append((bases, expts, gn, gd, hn, hd, hz))
            return terms
    else:
        def h(*args):
            a = args[:p]
            b = args[p:]
            terms = []
            for k in range(n):
                bases = [z]
                if r == 1:
                    expts = [a[k]-1]
                else:
                    expts = [(a[k]-1)/ctx.convert(r)]
                gn = [a[k]-a[j] for j in range(n) if j != k]
                gn += [1-a[k]+b[j] for j in range(m)]
                gd = [a[k]-b[j] for j in range(m,q)]
                gd += [1-a[k]+a[j] for j in range(n,p)]
                hn = [1-a[k]+b[j] for j in range(q)]
                hd = [1+a[j]-a[k] for j in range(p) if j != k]
                hz = (-ctx.one)**(q-m-n) / z**(ctx.one/r)
                terms.append((bases, expts, gn, gd, hn, hd, hz))
            return terms
    return ctx.hypercomb(h, a+b, **kwargs)

@defun_wrapped
def appellf1(ctx,a,b1,b2,c,x,y,**kwargs):
    # Assume x smaller
    # We will use x for the outer loop
    if abs(x) > abs(y):
        x, y = y, x
        b1, b2 = b2, b1
    def ok(x):
        return abs(x) < 0.99
    # Finite cases
    if ctx.isnpint(a):
        pass
    elif ctx.isnpint(b1):
        pass
    elif ctx.isnpint(b2):
        x, y, b1, b2 = y, x, b2, b1
    else:
        #print x, y
        # Note: ok if |y| > 1, because
        # 2F1 implements analytic continuation
        if not ok(x):
            u1 = (x-y)/(x-1)
            if not ok(u1):
                raise ValueError("Analytic continuation not implemented")
            #print "Using analytic continuation"
            return (1-x)**(-b1)*(1-y)**(c-a-b2)*\
                ctx.appellf1(c-a,b1,c-b1-b2,c,u1,y,**kwargs)
    return ctx.hyper2d({'m+n':[a],'m':[b1],'n':[b2]}, {'m+n':[c]}, x,y, **kwargs)

@defun
def appellf2(ctx,a,b1,b2,c1,c2,x,y,**kwargs):
    # TODO: continuation
    return ctx.hyper2d({'m+n':[a],'m':[b1],'n':[b2]},
        {'m':[c1],'n':[c2]}, x,y, **kwargs)

@defun
def appellf3(ctx,a1,a2,b1,b2,c,x,y,**kwargs):
    outer_polynomial = ctx.isnpint(a1) or ctx.isnpint(b1)
    inner_polynomial = ctx.isnpint(a2) or ctx.isnpint(b2)
    if not outer_polynomial:
        if inner_polynomial or abs(x) > abs(y):
            x, y = y, x
            a1,a2,b1,b2 = a2,a1,b2,b1
    return ctx.hyper2d({'m':[a1,b1],'n':[a2,b2]}, {'m+n':[c]},x,y,**kwargs)

@defun
def appellf4(ctx,a,b,c1,c2,x,y,**kwargs):
    # TODO: continuation
    return ctx.hyper2d({'m+n':[a,b]}, {'m':[c1],'n':[c2]},x,y,**kwargs)

@defun
def hyper2d(ctx, a, b, x, y, **kwargs):
    r"""
    Sums the generalized 2D hypergeometric series

    .. math ::

        \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
            \frac{P((a),m,n)}{Q((b),m,n)}
            \frac{x^m y^n} {m! n!}

    where `(a) = (a_1,\ldots,a_r)`, `(b) = (b_1,\ldots,b_s)` and where
    `P` and `Q` are products of rising factorials such as `(a_j)_n` or
    `(a_j)_{m+n}`. `P` and `Q` are specified in the form of dicts, with
    the `m` and `n` dependence as keys and parameter lists as values.
    The supported rising factorials are given in the following table
    (note that only a few are supported in `Q`):

    +------------+-------------------+--------+
    | Key        |  Rising factorial | `Q`    |
    +============+===================+========+
    | ``'m'``    |   `(a_j)_m`       | Yes    |
    +------------+-------------------+--------+
    | ``'n'``    |   `(a_j)_n`       | Yes    |
    +------------+-------------------+--------+
    | ``'m+n'``  |   `(a_j)_{m+n}`   | Yes    |
    +------------+-------------------+--------+
    | ``'m-n'``  |   `(a_j)_{m-n}`   | No     |
    +------------+-------------------+--------+
    | ``'n-m'``  |   `(a_j)_{n-m}`   | No     |
    +------------+-------------------+--------+
    | ``'2m+n'`` |   `(a_j)_{2m+n}`  | No     |
    +------------+-------------------+--------+
    | ``'2m-n'`` |   `(a_j)_{2m-n}`  | No     |
    +------------+-------------------+--------+
    | ``'2n-m'`` |   `(a_j)_{2n-m}`  | No     |
    +------------+-------------------+--------+

    For example, the Appell F1 and F4 functions

    .. math ::

        F_1 = \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
              \frac{(a)_{m+n} (b)_m (c)_n}{(d)_{m+n}}
              \frac{x^m y^n}{m! n!}

        F_4 = \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
              \frac{(a)_{m+n} (b)_{m+n}}{(c)_m (d)_{n}}
              \frac{x^m y^n}{m! n!}

    can be represented respectively as

        ``hyper2d({'m+n':[a], 'm':[b], 'n':[c]}, {'m+n':[d]}, x, y)``

        ``hyper2d({'m+n':[a,b]}, {'m':[c], 'n':[d]}, x, y)``

    More generally, :func:`~mpmath.hyper2d` can evaluate any of the 34 distinct
    convergent second-order (generalized Gaussian) hypergeometric
    series enumerated by Horn, as well as the Kampe de Feriet
    function.

    The series is computed by rewriting it so that the inner
    series (i.e. the series containing `n` and `y`) has the form of an
    ordinary generalized hypergeometric series and thereby can be
    evaluated efficiently using :func:`~mpmath.hyper`. If possible,
    manually swapping `x` and `y` and the corresponding parameters
    can sometimes give better results.

    **Examples**

    Two separable cases: a product of two geometric series, and a
    product of two Gaussian hypergeometric functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> x, y = mpf(0.25), mpf(0.5)
        >>> hyper2d({'m':1,'n':1}, {}, x,y)
        2.666666666666666666666667
        >>> 1/(1-x)/(1-y)
        2.666666666666666666666667
        >>> hyper2d({'m':[1,2],'n':[3,4]}, {'m':[5],'n':[6]}, x,y)
        4.164358531238938319669856
        >>> hyp2f1(1,2,5,x)*hyp2f1(3,4,6,y)
        4.164358531238938319669856

    Some more series that can be done in closed form::

        >>> hyper2d({'m':1,'n':1},{'m+n':1},x,y)
        2.013417124712514809623881
        >>> (exp(x)*x-exp(y)*y)/(x-y)
        2.013417124712514809623881

    Six of the 34 Horn functions, G1-G3 and H1-H3::

        >>> from mpmath import *
        >>> mp.dps = 10; mp.pretty = True
        >>> x, y = 0.0625, 0.125
        >>> a1,a2,b1,b2,c1,c2,d = 1.1,-1.2,-1.3,-1.4,1.5,-1.6,1.7
        >>> hyper2d({'m+n':a1,'n-m':b1,'m-n':b2},{},x,y)  # G1
        1.139090746
        >>> nsum(lambda m,n: rf(a1,m+n)*rf(b1,n-m)*rf(b2,m-n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        1.139090746
        >>> hyper2d({'m':a1,'n':a2,'n-m':b1,'m-n':b2},{},x,y)  # G2
        0.9503682696
        >>> nsum(lambda m,n: rf(a1,m)*rf(a2,n)*rf(b1,n-m)*rf(b2,m-n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        0.9503682696
        >>> hyper2d({'2n-m':a1,'2m-n':a2},{},x,y)  # G3
        1.029372029
        >>> nsum(lambda m,n: rf(a1,2*n-m)*rf(a2,2*m-n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        1.029372029
        >>> hyper2d({'m-n':a1,'m+n':b1,'n':c1},{'m':d},x,y)  # H1
        -1.605331256
        >>> nsum(lambda m,n: rf(a1,m-n)*rf(b1,m+n)*rf(c1,n)/rf(d,m)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        -1.605331256
        >>> hyper2d({'m-n':a1,'m':b1,'n':[c1,c2]},{'m':d},x,y)  # H2
        -2.35405404
        >>> nsum(lambda m,n: rf(a1,m-n)*rf(b1,m)*rf(c1,n)*rf(c2,n)/rf(d,m)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        -2.35405404
        >>> hyper2d({'2m+n':a1,'n':b1},{'m+n':c1},x,y)  # H3
        0.974479074
        >>> nsum(lambda m,n: rf(a1,2*m+n)*rf(b1,n)/rf(c1,m+n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        0.974479074

    **References**

    1. [SrivastavaKarlsson]_
    2. [Weisstein]_ http://mathworld.wolfram.com/HornFunction.html
    3. [Weisstein]_ http://mathworld.wolfram.com/AppellHypergeometricFunction.html

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    def parse(dct, key):
        args = dct.pop(key, [])
        try:
            args = list(args)
        except TypeError:
            args = [args]
        return [ctx.convert(arg) for arg in args]
    a_s = dict(a)
    b_s = dict(b)
    a_m = parse(a, 'm')
    a_n = parse(a, 'n')
    a_m_add_n = parse(a, 'm+n')
    a_m_sub_n = parse(a, 'm-n')
    a_n_sub_m = parse(a, 'n-m')
    a_2m_add_n = parse(a, '2m+n')
    a_2m_sub_n = parse(a, '2m-n')
    a_2n_sub_m = parse(a, '2n-m')
    b_m = parse(b, 'm')
    b_n = parse(b, 'n')
    b_m_add_n = parse(b, 'm+n')
    if a: raise ValueError("unsupported key: %r" % a.keys()[0])
    if b: raise ValueError("unsupported key: %r" % b.keys()[0])
    s = 0
    outer = ctx.one
    m = ctx.mpf(0)
    ok_count = 0
    prec = ctx.prec
    maxterms = kwargs.get('maxterms', 20*prec)
    try:
        ctx.prec += 10
        tol = +ctx.eps
        while 1:
            inner_sign = 1
            outer_sign = 1
            inner_a = list(a_n)
            inner_b = list(b_n)
            outer_a = [a+m for a in a_m]
            outer_b = [b+m for b in b_m]
            # (a)_{m+n} = (a)_m (a+m)_n
            for a in a_m_add_n:
                a = a+m
                inner_a.append(a)
                outer_a.append(a)
            # (b)_{m+n} = (b)_m (b+m)_n
            for b in b_m_add_n:
                b = b+m
                inner_b.append(b)
                outer_b.append(b)
            # (a)_{n-m} = (a-m)_n / (a-m)_m
            for a in a_n_sub_m:
                inner_a.append(a-m)
                outer_b.append(a-m-1)
            # (a)_{m-n} = (-1)^(m+n) (1-a-m)_m / (1-a-m)_n
            for a in a_m_sub_n:
                inner_sign *= (-1)
                outer_sign *= (-1)**(m)
                inner_b.append(1-a-m)
                outer_a.append(-a-m)
            # (a)_{2m+n} = (a)_{2m} (a+2m)_n
            for a in a_2m_add_n:
                inner_a.append(a+2*m)
                outer_a.append((a+2*m)*(1+a+2*m))
            # (a)_{2m-n} = (-1)^(2m+n) (1-a-2m)_{2m} / (1-a-2m)_n
            for a in a_2m_sub_n:
                inner_sign *= (-1)
                inner_b.append(1-a-2*m)
                outer_a.append((a+2*m)*(1+a+2*m))
            # (a)_{2n-m} = 4^n ((a-m)/2)_n ((a-m+1)/2)_n / (a-m)_m
            for a in a_2n_sub_m:
                inner_sign *= 4
                inner_a.append(0.5*(a-m))
                inner_a.append(0.5*(a-m+1))
                outer_b.append(a-m-1)
            inner = ctx.hyper(inner_a, inner_b, inner_sign*y,
                zeroprec=ctx.prec, **kwargs)
            term = outer * inner * outer_sign
            if abs(term) < tol:
                ok_count += 1
            else:
                ok_count = 0
            if ok_count >= 3 or not outer:
                break
            s += term
            for a in outer_a: outer *= a
            for b in outer_b: outer /= b
            m += 1
            outer = outer * x / m
            if m > maxterms:
                raise ctx.NoConvergence("maxterms exceeded in hyper2d")
    finally:
        ctx.prec = prec
    return +s

"""
@defun
def kampe_de_feriet(ctx,a,b,c,d,e,f,x,y,**kwargs):
    return ctx.hyper2d({'m+n':a,'m':b,'n':c},
        {'m+n':d,'m':e,'n':f}, x,y, **kwargs)
"""

@defun
def bihyper(ctx, a_s, b_s, z, **kwargs):
    r"""
    Evaluates the bilateral hypergeometric series

    .. math ::

        \,_AH_B(a_1, \ldots, a_k; b_1, \ldots, b_B; z) =
            \sum_{n=-\infty}^{\infty}
            \frac{(a_1)_n \ldots (a_A)_n}
                 {(b_1)_n \ldots (b_B)_n} \, z^n

    where, for direct convergence, `A = B` and `|z| = 1`, although a
    regularized sum exists more generally by considering the
    bilateral series as a sum of two ordinary hypergeometric
    functions. In order for the series to make sense, none of the
    parameters may be integers.

    **Examples**

    The value of `\,_2H_2` at `z = 1` is given by Dougall's formula::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a,b,c,d = 0.5, 1.5, 2.25, 3.25
        >>> bihyper([a,b],[c,d],1)
        -14.49118026212345786148847
        >>> gammaprod([c,d,1-a,1-b,c+d-a-b-1],[c-a,d-a,c-b,d-b])
        -14.49118026212345786148847

    The regularized function `\,_1H_0` can be expressed as the
    sum of one `\,_2F_0` function and one `\,_1F_1` function::

        >>> a = mpf(0.25)
        >>> z = mpf(0.75)
        >>> bihyper([a], [], z)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)
        >>> hyper([a,1],[],z) + (hyper([1],[1-a],-1/z)-1)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)
        >>> hyper([a,1],[],z) + hyper([1],[2-a],-1/z)/z/(a-1)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)

    **References**

    1. [Slater]_ (chapter 6: "Bilateral Series", pp. 180-189)
    2. [Wikipedia]_ http://en.wikipedia.org/wiki/Bilateral_hypergeometric_series

    """
    z = ctx.convert(z)
    c_s = a_s + b_s
    p = len(a_s)
    q = len(b_s)
    if (p, q) == (0,0) or (p, q) == (1,1):
        return ctx.zero * z
    neg = (p-q) % 2
    def h(*c_s):
        a_s = list(c_s[:p])
        b_s = list(c_s[p:])
        aa_s = [2-b for b in b_s]
        bb_s = [2-a for a in a_s]
        rp = [(-1)**neg * z] + [1-b for b in b_s] + [1-a for a in a_s]
        rc = [-1] + [1]*len(b_s) + [-1]*len(a_s)
        T1 = [], [], [], [], a_s + [1], b_s, z
        T2 = rp, rc, [], [], aa_s + [1], bb_s, (-1)**neg / z
        return T1, T2
    return ctx.hypercomb(h, c_s, **kwargs)
