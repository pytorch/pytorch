from __future__ import print_function

from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static

@defun
def stieltjes(ctx, n, a=1):
    n = ctx.convert(n)
    a = ctx.convert(a)
    if n < 0:
        return ctx.bad_domain("Stieltjes constants defined for n >= 0")
    if hasattr(ctx, "stieltjes_cache"):
        stieltjes_cache = ctx.stieltjes_cache
    else:
        stieltjes_cache = ctx.stieltjes_cache = {}
    if a == 1:
        if n == 0:
            return +ctx.euler
        if n in stieltjes_cache:
            prec, s = stieltjes_cache[n]
            if prec >= ctx.prec:
                return +s
    mag = 1
    def f(x):
        xa = x/a
        v = (xa-ctx.j)*ctx.ln(a-ctx.j*x)**n/(1+xa**2)/(ctx.exp(2*ctx.pi*x)-1)
        return ctx._re(v) / mag
    orig = ctx.prec
    try:
        # Normalize integrand by approx. magnitude to
        # speed up quadrature (which uses absolute error)
        if n > 50:
            ctx.prec = 20
            mag = ctx.quad(f, [0,ctx.inf], maxdegree=3)
        ctx.prec = orig + 10 + int(n**0.5)
        s = ctx.quad(f, [0,ctx.inf], maxdegree=20)
        v = ctx.ln(a)**n/(2*a) - ctx.ln(a)**(n+1)/(n+1) + 2*s/a*mag
    finally:
        ctx.prec = orig
    if a == 1 and ctx.isint(n):
        stieltjes_cache[n] = (ctx.prec, v)
    return +v

@defun_wrapped
def siegeltheta(ctx, t, derivative=0):
    d = int(derivative)
    if  (t == ctx.inf or t == ctx.ninf):
        if d < 2:
            if t == ctx.ninf and d == 0:
                return ctx.ninf
            return ctx.inf
        else:
            return ctx.zero
    if d == 0:
        if ctx._im(t):
            # XXX: cancellation occurs
            a = ctx.loggamma(0.25+0.5j*t)
            b = ctx.loggamma(0.25-0.5j*t)
            return -ctx.ln(ctx.pi)/2*t - 0.5j*(a-b)
        else:
            if ctx.isinf(t):
                return t
            return ctx._im(ctx.loggamma(0.25+0.5j*t)) - ctx.ln(ctx.pi)/2*t
    if d > 0:
        a = (-0.5j)**(d-1)*ctx.polygamma(d-1, 0.25-0.5j*t)
        b = (0.5j)**(d-1)*ctx.polygamma(d-1, 0.25+0.5j*t)
        if ctx._im(t):
            if d == 1:
                return -0.5*ctx.log(ctx.pi)+0.25*(a+b)
            else:
                return 0.25*(a+b)
        else:
            if d == 1:
                return ctx._re(-0.5*ctx.log(ctx.pi)+0.25*(a+b))
            else:
                return ctx._re(0.25*(a+b))

@defun_wrapped
def grampoint(ctx, n):
    # asymptotic expansion, from
    # http://mathworld.wolfram.com/GramPoint.html
    g = 2*ctx.pi*ctx.exp(1+ctx.lambertw((8*n+1)/(8*ctx.e)))
    return ctx.findroot(lambda t: ctx.siegeltheta(t)-ctx.pi*n, g)


@defun_wrapped
def siegelz(ctx, t, **kwargs):
    d = int(kwargs.get("derivative", 0))
    t = ctx.convert(t)
    t1 = ctx._re(t)
    t2 = ctx._im(t)
    prec = ctx.prec
    try:
        if abs(t1) > 500*prec and t2**2 < t1:
            v = ctx.rs_z(t, d)
            if ctx._is_real_type(t):
                return ctx._re(v)
            return v
    except NotImplementedError:
        pass
    ctx.prec += 21
    e1 = ctx.expj(ctx.siegeltheta(t))
    z = ctx.zeta(0.5+ctx.j*t)
    if d == 0:
        v = e1*z
        ctx.prec=prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z1 = ctx.zeta(0.5+ctx.j*t, derivative=1)
    theta1 = ctx.siegeltheta(t, derivative=1)
    if d == 1:
        v =  ctx.j*e1*(z1+z*theta1)
        ctx.prec=prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z2 = ctx.zeta(0.5+ctx.j*t, derivative=2)
    theta2 = ctx.siegeltheta(t, derivative=2)
    comb1 = theta1**2-ctx.j*theta2
    if d == 2:
        def terms():
            return [2*z1*theta1, z2, z*comb1]
        v = ctx.sum_accurately(terms, 1)
        v =  -e1*v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    ctx.prec += 10
    z3 = ctx.zeta(0.5+ctx.j*t, derivative=3)
    theta3 = ctx.siegeltheta(t, derivative=3)
    comb2 = theta1**3-3*ctx.j*theta1*theta2-theta3
    if d == 3:
        def terms():
            return  [3*theta1*z2, 3*z1*comb1, z3+z*comb2]
        v = ctx.sum_accurately(terms, 1)
        v =  -ctx.j*e1*v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z4 = ctx.zeta(0.5+ctx.j*t, derivative=4)
    theta4 = ctx.siegeltheta(t, derivative=4)
    def terms():
        return [theta1**4, -6*ctx.j*theta1**2*theta2, -3*theta2**2,
            -4*theta1*theta3, ctx.j*theta4]
    comb3 = ctx.sum_accurately(terms, 1)
    if d == 4:
        def terms():
            return  [6*theta1**2*z2, -6*ctx.j*z2*theta2, 4*theta1*z3,
                 4*z1*comb2, z4, z*comb3]
        v = ctx.sum_accurately(terms, 1)
        v =  e1*v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    if d > 4:
        h = lambda x: ctx.siegelz(x, derivative=4)
        return ctx.diff(h, t, n=d-4)


_zeta_zeros = [
14.134725142,21.022039639,25.010857580,30.424876126,32.935061588,
37.586178159,40.918719012,43.327073281,48.005150881,49.773832478,
52.970321478,56.446247697,59.347044003,60.831778525,65.112544048,
67.079810529,69.546401711,72.067157674,75.704690699,77.144840069,
79.337375020,82.910380854,84.735492981,87.425274613,88.809111208,
92.491899271,94.651344041,95.870634228,98.831194218,101.317851006,
103.725538040,105.446623052,107.168611184,111.029535543,111.874659177,
114.320220915,116.226680321,118.790782866,121.370125002,122.946829294,
124.256818554,127.516683880,129.578704200,131.087688531,133.497737203,
134.756509753,138.116042055,139.736208952,141.123707404,143.111845808,
146.000982487,147.422765343,150.053520421,150.925257612,153.024693811,
156.112909294,157.597591818,158.849988171,161.188964138,163.030709687,
165.537069188,167.184439978,169.094515416,169.911976479,173.411536520,
174.754191523,176.441434298,178.377407776,179.916484020,182.207078484,
184.874467848,185.598783678,187.228922584,189.416158656,192.026656361,
193.079726604,195.265396680,196.876481841,198.015309676,201.264751944,
202.493594514,204.189671803,205.394697202,207.906258888,209.576509717,
211.690862595,213.347919360,214.547044783,216.169538508,219.067596349,
220.714918839,221.430705555,224.007000255,224.983324670,227.421444280,
229.337413306,231.250188700,231.987235253,233.693404179,236.524229666,
]

def _load_zeta_zeros(url):
    import urllib
    d = urllib.urlopen(url)
    L = [float(x) for x in d.readlines()]
    # Sanity check
    assert round(L[0]) == 14
    _zeta_zeros[:] = L

@defun
def oldzetazero(ctx, n, url='http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros1'):
    n = int(n)
    if n < 0:
        return ctx.zetazero(-n).conjugate()
    if n == 0:
        raise ValueError("n must be nonzero")
    if n > len(_zeta_zeros) and n <= 100000:
        _load_zeta_zeros(url)
    if n > len(_zeta_zeros):
        raise NotImplementedError("n too large for zetazeros")
    return ctx.mpc(0.5, ctx.findroot(ctx.siegelz, _zeta_zeros[n-1]))

@defun_wrapped
def riemannr(ctx, x):
    if x == 0:
        return ctx.zero
    # Check if a simple asymptotic estimate is accurate enough
    if abs(x) > 1000:
        a = ctx.li(x)
        b = 0.5*ctx.li(ctx.sqrt(x))
        if abs(b) < abs(a)*ctx.eps:
            return a
    if abs(x) < 0.01:
        # XXX
        ctx.prec += int(-ctx.log(abs(x),2))
    # Sum Gram's series
    s = t = ctx.one
    u = ctx.ln(x)
    k = 1
    while abs(t) > abs(s)*ctx.eps:
        t = t * u / k
        s += t / (k * ctx._zeta_int(k+1))
        k += 1
    return s

@defun_static
def primepi(ctx, x):
    x = int(x)
    if x < 2:
        return 0
    return len(ctx.list_primes(x))

# TODO: fix the interface wrt contexts
@defun_wrapped
def primepi2(ctx, x):
    x = int(x)
    if x < 2:
        return ctx._iv.zero
    if x < 2657:
        return ctx._iv.mpf(ctx.primepi(x))
    mid = ctx.li(x)
    # Schoenfeld's estimate for x >= 2657, assuming RH
    err = ctx.sqrt(x,rounding='u')*ctx.ln(x,rounding='u')/8/ctx.pi(rounding='d')
    a = ctx.floor((ctx._iv.mpf(mid)-err).a, rounding='d')
    b = ctx.ceil((ctx._iv.mpf(mid)+err).b, rounding='u')
    return ctx._iv.mpf([a,b])

@defun_wrapped
def primezeta(ctx, s):
    if ctx.isnan(s):
        return s
    if ctx.re(s) <= 0:
        raise ValueError("prime zeta function defined only for re(s) > 0")
    if s == 1:
        return ctx.inf
    if s == 0.5:
        return ctx.mpc(ctx.ninf, ctx.pi)
    r = ctx.re(s)
    if r > ctx.prec:
        return 0.5**s
    else:
        wp = ctx.prec + int(r)
        def terms():
            orig = ctx.prec
            # zeta ~ 1+eps; need to set precision
            # to get logarithm accurately
            k = 0
            while 1:
                k += 1
                u = ctx.moebius(k)
                if not u:
                    continue
                ctx.prec = wp
                t = u*ctx.ln(ctx.zeta(k*s))/k
                if not t:
                    return
                #print ctx.prec, ctx.nstr(t)
                ctx.prec = orig
                yield t
    return ctx.sum_accurately(terms)

# TODO: for bernpoly and eulerpoly, ensure that all exact zeros are covered

@defun_wrapped
def bernpoly(ctx, n, z):
    # Slow implementation:
    #return sum(ctx.binomial(n,k)*ctx.bernoulli(k)*z**(n-k) for k in xrange(0,n+1))
    n = int(n)
    if n < 0:
        raise ValueError("Bernoulli polynomials only defined for n >= 0")
    if z == 0 or (z == 1 and n > 1):
        return ctx.bernoulli(n)
    if z == 0.5:
        return (ctx.ldexp(1,1-n)-1)*ctx.bernoulli(n)
    if n <= 3:
        if n == 0: return z ** 0
        if n == 1: return z - 0.5
        if n == 2: return (6*z*(z-1)+1)/6
        if n == 3: return z*(z*(z-1.5)+0.5)
    if ctx.isinf(z):
        return z ** n
    if ctx.isnan(z):
        return z
    if abs(z) > 2:
        def terms():
            t = ctx.one
            yield t
            r = ctx.one/z
            k = 1
            while k <= n:
                t = t*(n+1-k)/k*r
                if not (k > 2 and k & 1):
                    yield t*ctx.bernoulli(k)
                k += 1
        return ctx.sum_accurately(terms) * z**n
    else:
        def terms():
            yield ctx.bernoulli(n)
            t = ctx.one
            k = 1
            while k <= n:
                t = t*(n+1-k)/k * z
                m = n-k
                if not (m > 2 and m & 1):
                    yield t*ctx.bernoulli(m)
                k += 1
        return ctx.sum_accurately(terms)

@defun_wrapped
def eulerpoly(ctx, n, z):
    n = int(n)
    if n < 0:
        raise ValueError("Euler polynomials only defined for n >= 0")
    if n <= 2:
        if n == 0: return z ** 0
        if n == 1: return z - 0.5
        if n == 2: return z*(z-1)
    if ctx.isinf(z):
        return z**n
    if ctx.isnan(z):
        return z
    m = n+1
    if z == 0:
        return -2*(ctx.ldexp(1,m)-1)*ctx.bernoulli(m)/m * z**0
    if z == 1:
        return 2*(ctx.ldexp(1,m)-1)*ctx.bernoulli(m)/m * z**0
    if z == 0.5:
        if n % 2:
            return ctx.zero
        # Use exact code for Euler numbers
        if n < 100 or n*ctx.mag(0.46839865*n) < ctx.prec*0.25:
            return ctx.ldexp(ctx._eulernum(n), -n)
    # http://functions.wolfram.com/Polynomials/EulerE2/06/01/02/01/0002/
    def terms():
        t = ctx.one
        k = 0
        w = ctx.ldexp(1,n+2)
        while 1:
            v = n-k+1
            if not (v > 2 and v & 1):
                yield (2-w)*ctx.bernoulli(v)*t
            k += 1
            if k > n:
                break
            t = t*z*(n-k+2)/k
            w *= 0.5
    return ctx.sum_accurately(terms) / m

@defun
def eulernum(ctx, n, exact=False):
    n = int(n)
    if exact:
        return int(ctx._eulernum(n))
    if n < 100:
        return ctx.mpf(ctx._eulernum(n))
    if n % 2:
        return ctx.zero
    return ctx.ldexp(ctx.eulerpoly(n,0.5), n)

# TODO: this should be implemented low-level
def polylog_series(ctx, s, z):
    tol = +ctx.eps
    l = ctx.zero
    k = 1
    zk = z
    while 1:
        term = zk / k**s
        l += term
        if abs(term) < tol:
            break
        zk *= z
        k += 1
    return l

def polylog_continuation(ctx, n, z):
    if n < 0:
        return z*0
    twopij = 2j * ctx.pi
    a = -twopij**n/ctx.fac(n) * ctx.bernpoly(n, ctx.ln(z)/twopij)
    if ctx._is_real_type(z) and z < 0:
        a = ctx._re(a)
    if ctx._im(z) < 0 or (ctx._im(z) == 0 and ctx._re(z) >= 1):
        a -= twopij*ctx.ln(z)**(n-1)/ctx.fac(n-1)
    return a

def polylog_unitcircle(ctx, n, z):
    tol = +ctx.eps
    if n > 1:
        l = ctx.zero
        logz = ctx.ln(z)
        logmz = ctx.one
        m = 0
        while 1:
            if (n-m) != 1:
                term = ctx.zeta(n-m) * logmz / ctx.fac(m)
                if term and abs(term) < tol:
                    break
                l += term
            logmz *= logz
            m += 1
        l += ctx.ln(z)**(n-1)/ctx.fac(n-1)*(ctx.harmonic(n-1)-ctx.ln(-ctx.ln(z)))
    elif n < 1:  # else
        l = ctx.fac(-n)*(-ctx.ln(z))**(n-1)
        logz = ctx.ln(z)
        logkz = ctx.one
        k = 0
        while 1:
            b = ctx.bernoulli(k-n+1)
            if b:
                term = b*logkz/(ctx.fac(k)*(k-n+1))
                if abs(term) < tol:
                    break
                l -= term
            logkz *= logz
            k += 1
    else:
        raise ValueError
    if ctx._is_real_type(z) and z < 0:
        l = ctx._re(l)
    return l

def polylog_general(ctx, s, z):
    v = ctx.zero
    u = ctx.ln(z)
    if not abs(u) < 5: # theoretically |u| < 2*pi
        j = ctx.j
        v = 1-s
        y = ctx.ln(-z)/(2*ctx.pi*j)
        return ctx.gamma(v)*(j**v*ctx.zeta(v,0.5+y) + j**-v*ctx.zeta(v,0.5-y))/(2*ctx.pi)**v
    t = 1
    k = 0
    while 1:
        term = ctx.zeta(s-k) * t
        if abs(term) < ctx.eps:
            break
        v += term
        k += 1
        t *= u
        t /= k
    return ctx.gamma(1-s)*(-u)**(s-1) + v

@defun_wrapped
def polylog(ctx, s, z):
    s = ctx.convert(s)
    z = ctx.convert(z)
    if z == 1:
        return ctx.zeta(s)
    if z == -1:
        return -ctx.altzeta(s)
    if s == 0:
        return z/(1-z)
    if s == 1:
        return -ctx.ln(1-z)
    if s == -1:
        return z/(1-z)**2
    if abs(z) <= 0.75 or (not ctx.isint(s) and abs(z) < 0.9):
        return polylog_series(ctx, s, z)
    if abs(z) >= 1.4 and ctx.isint(s):
        return (-1)**(s+1)*polylog_series(ctx, s, 1/z) + polylog_continuation(ctx, int(ctx.re(s)), z)
    if ctx.isint(s):
        return polylog_unitcircle(ctx, int(ctx.re(s)), z)
    return polylog_general(ctx, s, z)

@defun_wrapped
def clsin(ctx, s, z, pi=False):
    if ctx.isint(s) and s < 0 and int(s) % 2 == 1:
        return z*0
    if pi:
        a = ctx.expjpi(z)
    else:
        a = ctx.expj(z)
    if ctx._is_real_type(z) and ctx._is_real_type(s):
        return ctx.im(ctx.polylog(s,a))
    b = 1/a
    return (-0.5j)*(ctx.polylog(s,a) - ctx.polylog(s,b))

@defun_wrapped
def clcos(ctx, s, z, pi=False):
    if ctx.isint(s) and s < 0 and int(s) % 2 == 0:
        return z*0
    if pi:
        a = ctx.expjpi(z)
    else:
        a = ctx.expj(z)
    if ctx._is_real_type(z) and ctx._is_real_type(s):
        return ctx.re(ctx.polylog(s,a))
    b = 1/a
    return 0.5*(ctx.polylog(s,a) + ctx.polylog(s,b))

@defun
def altzeta(ctx, s, **kwargs):
    try:
        return ctx._altzeta(s, **kwargs)
    except NotImplementedError:
        return ctx._altzeta_generic(s)

@defun_wrapped
def _altzeta_generic(ctx, s):
    if s == 1:
        return ctx.ln2 + 0*s
    return -ctx.powm1(2, 1-s) * ctx.zeta(s)

@defun
def zeta(ctx, s, a=1, derivative=0, method=None, **kwargs):
    d = int(derivative)
    if a == 1 and not (d or method):
        try:
            return ctx._zeta(s, **kwargs)
        except NotImplementedError:
            pass
    s = ctx.convert(s)
    prec = ctx.prec
    method = kwargs.get('method')
    verbose = kwargs.get('verbose')
    if (not s) and (not derivative):
        return ctx.mpf(0.5) - ctx._convert_param(a)[0]
    if a == 1 and method != 'euler-maclaurin':
        im = abs(ctx._im(s))
        re = abs(ctx._re(s))
        #if (im < prec or method == 'borwein') and not derivative:
        #    try:
        #        if verbose:
        #            print "zeta: Attempting to use the Borwein algorithm"
        #        return ctx._zeta(s, **kwargs)
        #    except NotImplementedError:
        #        if verbose:
        #            print "zeta: Could not use the Borwein algorithm"
        #        pass
        if abs(im) > 500*prec and 10*re < prec and derivative <= 4 or \
            method == 'riemann-siegel':
            try:   #  py2.4 compatible try block
                try:
                    if verbose:
                        print("zeta: Attempting to use the Riemann-Siegel algorithm")
                    return ctx.rs_zeta(s, derivative, **kwargs)
                except NotImplementedError:
                    if verbose:
                        print("zeta: Could not use the Riemann-Siegel algorithm")
                    pass
            finally:
                ctx.prec = prec
    if s == 1:
        return ctx.inf
    abss = abs(s)
    if abss == ctx.inf:
        if ctx.re(s) == ctx.inf:
            if d == 0:
                return ctx.one
            return ctx.zero
        return s*0
    elif ctx.isnan(abss):
        return 1/s
    if ctx.re(s) > 2*ctx.prec and a == 1 and not derivative:
        return ctx.one + ctx.power(2, -s)
    return +ctx._hurwitz(s, a, d, **kwargs)

@defun
def _hurwitz(ctx, s, a=1, d=0, **kwargs):
    prec = ctx.prec
    verbose = kwargs.get('verbose')
    try:
        extraprec = 10
        ctx.prec += extraprec
        # We strongly want to special-case rational a
        a, atype = ctx._convert_param(a)
        if ctx.re(s) < 0:
            if verbose:
                print("zeta: Attempting reflection formula")
            try:
                return _hurwitz_reflection(ctx, s, a, d, atype)
            except NotImplementedError:
                pass
            if verbose:
                print("zeta: Reflection formula failed")
        if verbose:
            print("zeta: Using the Euler-Maclaurin algorithm")
        while 1:
            ctx.prec = prec + extraprec
            T1, T2 = _hurwitz_em(ctx, s, a, d, prec+10, verbose)
            cancellation = ctx.mag(T1) - ctx.mag(T1+T2)
            if verbose:
                print("Term 1:", T1)
                print("Term 2:", T2)
                print("Cancellation:", cancellation, "bits")
            if cancellation < extraprec:
                return T1 + T2
            else:
                extraprec = max(2*extraprec, min(cancellation + 5, 100*prec))
                if extraprec > kwargs.get('maxprec', 100*prec):
                    raise ctx.NoConvergence("zeta: too much cancellation")
    finally:
        ctx.prec = prec

def _hurwitz_reflection(ctx, s, a, d, atype):
    # TODO: implement for derivatives
    if d != 0:
        raise NotImplementedError
    res = ctx.re(s)
    negs = -s
    # Integer reflection formula
    if ctx.isnpint(s):
        n = int(res)
        if n <= 0:
            return ctx.bernpoly(1-n, a) / (n-1)
    if not (atype == 'Q' or atype == 'Z'):
        raise NotImplementedError
    t = 1-s
    # We now require a to be standardized
    v = 0
    shift = 0
    b = a
    while ctx.re(b) > 1:
        b -= 1
        v -= b**negs
        shift -= 1
    while ctx.re(b) <= 0:
        v += b**negs
        b += 1
        shift += 1
    # Rational reflection formula
    try:
        p, q = a._mpq_
    except:
        assert a == int(a)
        p = int(a)
        q = 1
    p += shift*q
    assert 1 <= p <= q
    g = ctx.fsum(ctx.cospi(t/2-2*k*b)*ctx._hurwitz(t,(k,q)) \
        for k in range(1,q+1))
    g *= 2*ctx.gamma(t)/(2*ctx.pi*q)**t
    v += g
    return v

def _hurwitz_em(ctx, s, a, d, prec, verbose):
    # May not be converted at this point
    a = ctx.convert(a)
    tol = -prec
    # Estimate number of terms for Euler-Maclaurin summation; could be improved
    M1 = 0
    M2 = prec // 3
    N = M2
    lsum = 0
    # This speeds up the recurrence for derivatives
    if ctx.isint(s):
        s = int(ctx._re(s))
    s1 = s-1
    while 1:
        # Truncated L-series
        l = ctx._zetasum(s, M1+a, M2-M1-1, [d])[0][0]
        #if d:
        #    l = ctx.fsum((-ctx.ln(n+a))**d * (n+a)**negs for n in range(M1,M2))
        #else:
        #    l = ctx.fsum((n+a)**negs for n in range(M1,M2))
        lsum += l
        M2a = M2+a
        logM2a = ctx.ln(M2a)
        logM2ad = logM2a**d
        logs = [logM2ad]
        logr = 1/logM2a
        rM2a = 1/M2a
        M2as = M2a**(-s)
        if d:
            tailsum = ctx.gammainc(d+1, s1*logM2a) / s1**(d+1)
        else:
            tailsum = 1/((s1)*(M2a)**s1)
        tailsum += 0.5 * logM2ad * M2as
        U = [1]
        r = M2as
        fact = 2
        for j in range(1, N+1):
            # TODO: the following could perhaps be tidied a bit
            j2 = 2*j
            if j == 1:
                upds = [1]
            else:
                upds = [j2-2, j2-1]
            for m in upds:
                D = min(m,d+1)
                if m <= d:
                    logs.append(logs[-1] * logr)
                Un = [0]*(D+1)
                for i in xrange(D): Un[i] = (1-m-s)*U[i]
                for i in xrange(1,D+1): Un[i] += (d-(i-1))*U[i-1]
                U = Un
                r *= rM2a
            t = ctx.fdot(U, logs) * r * ctx.bernoulli(j2)/(-fact)
            tailsum += t
            if ctx.mag(t) < tol:
                return lsum, (-1)**d * tailsum
            fact *= (j2+1)*(j2+2)
        if verbose:
            print("Sum range:", M1, M2, "term magnitude", ctx.mag(t), "tolerance", tol)
        M1, M2 = M2, M2*2
        if ctx.re(s) < 0:
            N += N//2



@defun
def _zetasum(ctx, s, a, n, derivatives=[0], reflect=False):
    """
    Returns [xd0,xd1,...,xdr], [yd0,yd1,...ydr] where

    xdk = D^k     ( 1/a^s     +  1/(a+1)^s      +  ...  +  1/(a+n)^s     )
    ydk = D^k conj( 1/a^(1-s) +  1/(a+1)^(1-s)  +  ...  +  1/(a+n)^(1-s) )

    D^k = kth derivative with respect to s, k ranges over the given list of
    derivatives (which should consist of either a single element
    or a range 0,1,...r). If reflect=False, the ydks are not computed.
    """
    #print "zetasum", s, a, n
    # don't use the fixed-point code if there are large exponentials
    if abs(ctx.re(s)) < 0.5 * ctx.prec:
        try:
            return ctx._zetasum_fast(s, a, n, derivatives, reflect)
        except NotImplementedError:
            pass
    negs = ctx.fneg(s, exact=True)
    have_derivatives = derivatives != [0]
    have_one_derivative = len(derivatives) == 1
    if not reflect:
        if not have_derivatives:
            return [ctx.fsum((a+k)**negs for k in xrange(n+1))], []
        if have_one_derivative:
            d = derivatives[0]
            x = ctx.fsum(ctx.ln(a+k)**d * (a+k)**negs for k in xrange(n+1))
            return [(-1)**d * x], []
    maxd = max(derivatives)
    if not have_one_derivative:
        derivatives = range(maxd+1)
    xs = [ctx.zero for d in derivatives]
    if reflect:
        ys = [ctx.zero for d in derivatives]
    else:
        ys = []
    for k in xrange(n+1):
        w = a + k
        xterm = w ** negs
        if reflect:
            yterm = ctx.conj(ctx.one / (w * xterm))
        if have_derivatives:
            logw = -ctx.ln(w)
            if have_one_derivative:
                logw = logw ** maxd
                xs[0] += xterm * logw
                if reflect:
                    ys[0] += yterm * logw
            else:
                t = ctx.one
                for d in derivatives:
                    xs[d] += xterm * t
                    if reflect:
                        ys[d] += yterm * t
                    t *= logw
        else:
            xs[0] += xterm
            if reflect:
                ys[0] += yterm
    return xs, ys

@defun
def dirichlet(ctx, s, chi=[1], derivative=0):
    s = ctx.convert(s)
    q = len(chi)
    d = int(derivative)
    if d > 2:
        raise NotImplementedError("arbitrary order derivatives")
    prec = ctx.prec
    try:
        ctx.prec += 10
        if s == 1:
            have_pole = True
            for x in chi:
                if x and x != 1:
                    have_pole = False
                    h = +ctx.eps
                    ctx.prec *= 2*(d+1)
                    s += h
            if have_pole:
                return +ctx.inf
        z = ctx.zero
        for p in range(1,q+1):
            if chi[p%q]:
                if d == 1:
                    z += chi[p%q] * (ctx.zeta(s, (p,q), 1) - \
                        ctx.zeta(s, (p,q))*ctx.log(q))
                else:
                    z += chi[p%q] * ctx.zeta(s, (p,q))
        z /= q**s
    finally:
        ctx.prec = prec
    return +z


def secondzeta_main_term(ctx, s, a, **kwargs):
    tol = ctx.eps
    f = lambda n: ctx.gammainc(0.5*s, a*gamm**2, regularized=True)*gamm**(-s)
    totsum = term = ctx.zero
    mg = ctx.inf
    n = 0
    while mg > tol:
        totsum += term
        n += 1
        gamm = ctx.im(ctx.zetazero_memoized(n))
        term = f(n)
        mg = abs(term)
    err = 0
    if kwargs.get("error"):
        sg = ctx.re(s)
        err = 0.5*ctx.pi**(-1)*max(1,sg)*a**(sg-0.5)*ctx.log(gamm/(2*ctx.pi))*\
             ctx.gammainc(-0.5, a*gamm**2)/abs(ctx.gamma(s/2))
        err = abs(err)
    return +totsum, err, n

def secondzeta_prime_term(ctx, s, a, **kwargs):
    tol = ctx.eps
    f = lambda n: ctx.gammainc(0.5*(1-s),0.25*ctx.log(n)**2 * a**(-1))*\
        ((0.5*ctx.log(n))**(s-1))*ctx.mangoldt(n)/ctx.sqrt(n)/\
        (2*ctx.gamma(0.5*s)*ctx.sqrt(ctx.pi))
    totsum = term = ctx.zero
    mg = ctx.inf
    n = 1
    while mg > tol or n < 9:
        totsum += term
        n += 1
        term = f(n)
        if term == 0:
            mg = ctx.inf
        else:
            mg = abs(term)
    if kwargs.get("error"):
        err = mg
    return +totsum, err, n

def secondzeta_exp_term(ctx, s, a):
    if ctx.isint(s) and ctx.re(s) <= 0:
        m = int(round(ctx.re(s)))
        if not m & 1:
            return ctx.mpf('-0.25')**(-m//2)
    tol = ctx.eps
    f = lambda n: (0.25*a)**n/((n+0.5*s)*ctx.fac(n))
    totsum = ctx.zero
    term = f(0)
    mg = ctx.inf
    n = 0
    while mg > tol:
        totsum += term
        n += 1
        term = f(n)
        mg = abs(term)
    v = a**(0.5*s)*totsum/ctx.gamma(0.5*s)
    return v

def secondzeta_singular_term(ctx, s, a, **kwargs):
    factor = a**(0.5*(s-1))/(4*ctx.sqrt(ctx.pi)*ctx.gamma(0.5*s))
    extraprec = ctx.mag(factor)
    ctx.prec += extraprec
    factor = a**(0.5*(s-1))/(4*ctx.sqrt(ctx.pi)*ctx.gamma(0.5*s))
    tol = ctx.eps
    f = lambda n: ctx.bernpoly(n,0.75)*(4*ctx.sqrt(a))**n*\
       ctx.gamma(0.5*n)/((s+n-1)*ctx.fac(n))
    totsum = ctx.zero
    mg1 = ctx.inf
    n = 1
    term = f(n)
    mg2 = abs(term)
    while mg2 > tol and mg2 <= mg1:
        totsum += term
        n += 1
        term = f(n)
        totsum += term
        n +=1
        term = f(n)
        mg1 = mg2
        mg2 = abs(term)
    totsum += term
    pole = -2*(s-1)**(-2)+(ctx.euler+ctx.log(16*ctx.pi**2*a))*(s-1)**(-1)
    st = factor*(pole+totsum)
    err = 0
    if kwargs.get("error"):
        if not ((mg2 > tol) and (mg2 <= mg1)):
            if mg2 <= tol:
                err = ctx.mpf(10)**int(ctx.log(abs(factor*tol),10))
            if mg2 > mg1:
                err = ctx.mpf(10)**int(ctx.log(abs(factor*mg1),10))
        err = max(err, ctx.eps*1.)
    ctx.prec -= extraprec
    return +st, err

@defun
def secondzeta(ctx, s, a = 0.015, **kwargs):
    r"""
    Evaluates the secondary zeta function `Z(s)`, defined for
    `\mathrm{Re}(s)>1` by

    .. math ::

        Z(s) = \sum_{n=1}^{\infty} \frac{1}{\tau_n^s}

    where `\frac12+i\tau_n` runs through the zeros of `\zeta(s)` with
    imaginary part positive.

    `Z(s)` extends to a meromorphic function on `\mathbb{C}`  with a
    double pole at `s=1` and  simple poles at the points `-2n` for
    `n=0`,  1, 2, ...

    **Examples**

        >>> from mpmath import *
        >>> mp.pretty = True; mp.dps = 15
        >>> secondzeta(2)
        0.023104993115419
        >>> xi = lambda s: 0.5*s*(s-1)*pi**(-0.5*s)*gamma(0.5*s)*zeta(s)
        >>> Xi = lambda t: xi(0.5+t*j)
        >>> chop(-0.5*diff(Xi,0,n=2)/Xi(0))
        0.023104993115419

    We may ask for an approximate error value::

        >>> secondzeta(0.5+100j, error=True)
        ((-0.216272011276718 - 0.844952708937228j), 2.22044604925031e-16)

    The function has poles at the negative odd integers,
    and dyadic rational values at the negative even integers::

        >>> mp.dps = 30
        >>> secondzeta(-8)
        -0.67236328125
        >>> secondzeta(-7)
        +inf

    **Implementation notes**

    The function is computed as sum of four terms `Z(s)=A(s)-P(s)+E(s)-S(s)`
    respectively main, prime, exponential and singular terms.
    The main term `A(s)` is computed from the zeros of zeta.
    The prime term depends on the von Mangoldt function.
    The singular term is responsible for the poles of the function.

    The four terms depends on a small parameter `a`. We may change the
    value of `a`. Theoretically this has no effect on the sum of the four
    terms, but in practice may be important.

    A smaller value of the parameter `a` makes `A(s)` depend on
    a smaller number of zeros of zeta, but `P(s)`  uses more values of
    von Mangoldt function.

    We may also add a verbose option to obtain data about the
    values of the four terms.

        >>> mp.dps = 10
        >>> secondzeta(0.5 + 40j, error=True, verbose=True)
        main term = (-30190318549.138656312556 - 13964804384.624622876523j)
            computed using 19 zeros of zeta
        prime term = (132717176.89212754625045 + 188980555.17563978290601j)
            computed using 9 values of the von Mangoldt function
        exponential term = (542447428666.07179812536 + 362434922978.80192435203j)
        singular term = (512124392939.98154322355 + 348281138038.65531023921j)
        ((0.059471043 + 0.3463514534j), 1.455191523e-11)

        >>> secondzeta(0.5 + 40j, a=0.04, error=True, verbose=True)
        main term = (-151962888.19606243907725 - 217930683.90210294051982j)
            computed using 9 zeros of zeta
        prime term = (2476659342.3038722372461 + 28711581821.921627163136j)
            computed using 37 values of the von Mangoldt function
        exponential term = (178506047114.7838188264 + 819674143244.45677330576j)
        singular term = (175877424884.22441310708 + 790744630738.28669174871j)
        ((0.059471043 + 0.3463514534j), 1.455191523e-11)

    Notice the great cancellation between the four terms. Changing `a`, the
    four terms are very different numbers but the cancellation gives
    the good value of Z(s).

    **References**

    A. Voros, Zeta functions for the Riemann zeros, Ann. Institute Fourier,
    53, (2003) 665--699.

    A. Voros, Zeta functions over Zeros of Zeta Functions, Lecture Notes
    of the Unione Matematica Italiana, Springer, 2009.
    """
    s = ctx.convert(s)
    a = ctx.convert(a)
    tol = ctx.eps
    if ctx.isint(s) and ctx.re(s) <= 1:
        if abs(s-1) < tol*1000:
            return ctx.inf
        m = int(round(ctx.re(s)))
        if m & 1:
            return ctx.inf
        else:
            return ((-1)**(-m//2)*\
                   ctx.fraction(8-ctx.eulernum(-m,exact=True),2**(-m+3)))
    prec = ctx.prec
    try:
        t3 = secondzeta_exp_term(ctx, s, a)
        extraprec = max(ctx.mag(t3),0)
        ctx.prec += extraprec + 3
        t1, r1, gt = secondzeta_main_term(ctx,s,a,error='True', verbose='True')
        t2, r2, pt = secondzeta_prime_term(ctx,s,a,error='True', verbose='True')
        t4, r4 = secondzeta_singular_term(ctx,s,a,error='True')
        t3 = secondzeta_exp_term(ctx, s, a)
        err = r1+r2+r4
        t = t1-t2+t3-t4
        if kwargs.get("verbose"):
            print('main term =', t1)
            print('    computed using', gt, 'zeros of zeta')
            print('prime term =', t2)
            print('    computed using', pt, 'values of the von Mangoldt function')
            print('exponential term =', t3)
            print('singular term =', t4)
    finally:
        ctx.prec = prec
    if kwargs.get("error"):
        w = max(ctx.mag(abs(t)),0)
        err = max(err*2**w, ctx.eps*1.*2**w)
        return +t, err
    return +t


@defun_wrapped
def lerchphi(ctx, z, s, a):
    r"""
    Gives the Lerch transcendent, defined for `|z| < 1` and
    `\Re{a} > 0` by

    .. math ::

        \Phi(z,s,a) = \sum_{k=0}^{\infty} \frac{z^k}{(a+k)^s}

    and generally by the recurrence `\Phi(z,s,a) = z \Phi(z,s,a+1) + a^{-s}`
    along with the integral representation valid for `\Re{a} > 0`

    .. math ::

        \Phi(z,s,a) = \frac{1}{2 a^s} +
                \int_0^{\infty} \frac{z^t}{(a+t)^s} dt -
                2 \int_0^{\infty} \frac{\sin(t \log z - s
                    \operatorname{arctan}(t/a)}{(a^2 + t^2)^{s/2}
                    (e^{2 \pi t}-1)} dt.

    The Lerch transcendent generalizes the Hurwitz zeta function :func:`zeta`
    (`z = 1`) and the polylogarithm :func:`polylog` (`a = 1`).

    **Examples**

    Several evaluations in terms of simpler functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> lerchphi(-1,2,0.5); 4*catalan
        3.663862376708876060218414
        3.663862376708876060218414
        >>> diff(lerchphi, (-1,-2,1), (0,1,0)); 7*zeta(3)/(4*pi**2)
        0.2131391994087528954617607
        0.2131391994087528954617607
        >>> lerchphi(-4,1,1); log(5)/4
        0.4023594781085250936501898
        0.4023594781085250936501898
        >>> lerchphi(-3+2j,1,0.5); 2*atanh(sqrt(-3+2j))/sqrt(-3+2j)
        (1.142423447120257137774002 + 0.2118232380980201350495795j)
        (1.142423447120257137774002 + 0.2118232380980201350495795j)

    Evaluation works for complex arguments and `|z| \ge 1`::

        >>> lerchphi(1+2j, 3-j, 4+2j)
        (0.002025009957009908600539469 + 0.003327897536813558807438089j)
        >>> lerchphi(-2,2,-2.5)
        -12.28676272353094275265944
        >>> lerchphi(10,10,10)
        (-4.462130727102185701817349e-11 - 1.575172198981096218823481e-12j)
        >>> lerchphi(10,10,-10.5)
        (112658784011940.5605789002 - 498113185.5756221777743631j)

    Some degenerate cases::

        >>> lerchphi(0,1,2)
        0.5
        >>> lerchphi(0,1,-2)
        -0.5

    Reduction to simpler functions::

        >>> lerchphi(1, 4.25+1j, 1)
        (1.044674457556746668033975 - 0.04674508654012658932271226j)
        >>> zeta(4.25+1j)
        (1.044674457556746668033975 - 0.04674508654012658932271226j)
        >>> lerchphi(1 - 0.5**10, 4.25+1j, 1)
        (1.044629338021507546737197 - 0.04667768813963388181708101j)
        >>> lerchphi(3, 4, 1)
        (1.249503297023366545192592 - 0.2314252413375664776474462j)
        >>> polylog(4, 3) / 3
        (1.249503297023366545192592 - 0.2314252413375664776474462j)
        >>> lerchphi(3, 4, 1 - 0.5**10)
        (1.253978063946663945672674 - 0.2316736622836535468765376j)

    **References**

    1. [DLMF]_ section 25.14

    """
    if z == 0:
        return a ** (-s)
    # Faster, but these cases are useful for testing right now
    if z == 1:
        return ctx.zeta(s, a)
    if a == 1:
        return ctx.polylog(s, z) / z
    if ctx.re(a) < 1:
        if ctx.isnpint(a):
            raise ValueError("Lerch transcendent complex infinity")
        m = int(ctx.ceil(1-ctx.re(a)))
        v = ctx.zero
        zpow = ctx.one
        for n in xrange(m):
            v += zpow / (a+n)**s
            zpow *= z
        return zpow * ctx.lerchphi(z,s, a+m) + v
    g = ctx.ln(z)
    v = 1/(2*a**s) + ctx.gammainc(1-s, -a*g) * (-g)**(s-1) / z**a
    h = s / 2
    r = 2*ctx.pi
    f = lambda t: ctx.sin(s*ctx.atan(t/a)-t*g) / \
        ((a**2+t**2)**h * ctx.expm1(r*t))
    v += 2*ctx.quad(f, [0, ctx.inf])
    if not ctx.im(z) and not ctx.im(s) and not ctx.im(a) and ctx.re(z) < 1:
        v = ctx.chop(v)
    return v
