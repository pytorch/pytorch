from .functions import defun, defun_wrapped

@defun
def j0(ctx, x):
    """Computes the Bessel function `J_0(x)`. See :func:`~mpmath.besselj`."""
    return ctx.besselj(0, x)

@defun
def j1(ctx, x):
    """Computes the Bessel function `J_1(x)`.  See :func:`~mpmath.besselj`."""
    return ctx.besselj(1, x)

@defun
def besselj(ctx, n, z, derivative=0, **kwargs):
    if type(n) is int:
        n_isint = True
    else:
        n = ctx.convert(n)
        n_isint = ctx.isint(n)
        if n_isint:
            n = int(ctx._re(n))
    if n_isint and n < 0:
        return (-1)**n * ctx.besselj(-n, z, derivative, **kwargs)
    z = ctx.convert(z)
    M = ctx.mag(z)
    if derivative:
        d = ctx.convert(derivative)
        # TODO: the integer special-casing shouldn't be necessary.
        # However, the hypergeometric series gets inaccurate for large d
        # because of inaccurate pole cancellation at a pole far from
        # zero (needs to be fixed in hypercomb or hypsum)
        if ctx.isint(d) and d >= 0:
            d = int(d)
            orig = ctx.prec
            try:
                ctx.prec += 15
                v = ctx.fsum((-1)**k * ctx.binomial(d,k) * ctx.besselj(2*k+n-d,z)
                    for k in range(d+1))
            finally:
                ctx.prec = orig
            v *= ctx.mpf(2)**(-d)
        else:
            def h(n,d):
                r = ctx.fmul(ctx.fmul(z, z, prec=ctx.prec+M), -0.25, exact=True)
                B = [0.5*(n-d+1), 0.5*(n-d+2)]
                T = [([2,ctx.pi,z],[d-2*n,0.5,n-d],[],B,[(n+1)*0.5,(n+2)*0.5],B+[n+1],r)]
                return T
            v = ctx.hypercomb(h, [n,d], **kwargs)
    else:
        # Fast case: J_n(x), n int, appropriate magnitude for fixed-point calculation
        if (not derivative) and n_isint and abs(M) < 10 and abs(n) < 20:
            try:
                return ctx._besselj(n, z)
            except NotImplementedError:
                pass
        if not z:
            if not n:
                v = ctx.one + n+z
            elif ctx.re(n) > 0:
                v = n*z
            else:
                v = ctx.inf + z + n
        else:
            #v = 0
            orig = ctx.prec
            try:
                # XXX: workaround for accuracy in low level hypergeometric series
                # when alternating, large arguments
                ctx.prec += min(3*abs(M), ctx.prec)
                w = ctx.fmul(z, 0.5, exact=True)
                def h(n):
                    r = ctx.fneg(ctx.fmul(w, w, prec=max(0,ctx.prec+M)), exact=True)
                    return [([w], [n], [], [n+1], [], [n+1], r)]
                v = ctx.hypercomb(h, [n], **kwargs)
            finally:
                ctx.prec = orig
        v = +v
    return v

@defun
def besseli(ctx, n, z, derivative=0, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    if not z:
        if derivative:
            raise ValueError
        if not n:
            # I(0,0) = 1
            return 1+n+z
        if ctx.isint(n):
            return 0*(n+z)
        r = ctx.re(n)
        if r == 0:
            return ctx.nan*(n+z)
        elif r > 0:
            return 0*(n+z)
        else:
            return ctx.inf+(n+z)
    M = ctx.mag(z)
    if derivative:
        d = ctx.convert(derivative)
        def h(n,d):
            r = ctx.fmul(ctx.fmul(z, z, prec=ctx.prec+M), 0.25, exact=True)
            B = [0.5*(n-d+1), 0.5*(n-d+2), n+1]
            T = [([2,ctx.pi,z],[d-2*n,0.5,n-d],[n+1],B,[(n+1)*0.5,(n+2)*0.5],B,r)]
            return T
        v = ctx.hypercomb(h, [n,d], **kwargs)
    else:
        def h(n):
            w = ctx.fmul(z, 0.5, exact=True)
            r = ctx.fmul(w, w, prec=max(0,ctx.prec+M))
            return [([w], [n], [], [n+1], [], [n+1], r)]
        v = ctx.hypercomb(h, [n], **kwargs)
    return v

@defun_wrapped
def bessely(ctx, n, z, derivative=0, **kwargs):
    if not z:
        if derivative:
            # Not implemented
            raise ValueError
        if not n:
            # ~ log(z/2)
            return -ctx.inf + (n+z)
        if ctx.im(n):
            return ctx.nan * (n+z)
        r = ctx.re(n)
        q = n+0.5
        if ctx.isint(q):
            if n > 0:
                return -ctx.inf + (n+z)
            else:
                return 0 * (n+z)
        if r < 0 and int(ctx.floor(q)) % 2:
            return ctx.inf + (n+z)
        else:
            return ctx.ninf + (n+z)
    # XXX: use hypercomb
    ctx.prec += 10
    m, d = ctx.nint_distance(n)
    if d < -ctx.prec:
        h = +ctx.eps
        ctx.prec *= 2
        n += h
    elif d < 0:
        ctx.prec -= d
    # TODO: avoid cancellation for imaginary arguments
    cos, sin = ctx.cospi_sinpi(n)
    return (ctx.besselj(n,z,derivative,**kwargs)*cos - \
        ctx.besselj(-n,z,derivative,**kwargs))/sin

@defun_wrapped
def besselk(ctx, n, z, **kwargs):
    if not z:
        return ctx.inf
    M = ctx.mag(z)
    if M < 1:
        # Represent as limit definition
        def h(n):
            r = (z/2)**2
            T1 = [z, 2], [-n, n-1], [n], [], [], [1-n], r
            T2 = [z, 2], [n, -n-1], [-n], [], [], [1+n], r
            return T1, T2
    # We could use the limit definition always, but it leads
    # to very bad cancellation (of exponentially large terms)
    # for large real z
    # Instead represent in terms of 2F0
    else:
        ctx.prec += M
        def h(n):
            return [([ctx.pi/2, z, ctx.exp(-z)], [0.5,-0.5,1], [], [], \
                [n+0.5, 0.5-n], [], -1/(2*z))]
    return ctx.hypercomb(h, [n], **kwargs)

@defun_wrapped
def hankel1(ctx,n,x,**kwargs):
    return ctx.besselj(n,x,**kwargs) + ctx.j*ctx.bessely(n,x,**kwargs)

@defun_wrapped
def hankel2(ctx,n,x,**kwargs):
    return ctx.besselj(n,x,**kwargs) - ctx.j*ctx.bessely(n,x,**kwargs)

@defun_wrapped
def whitm(ctx,k,m,z,**kwargs):
    if z == 0:
        # M(k,m,z) = 0^(1/2+m)
        if ctx.re(m) > -0.5:
            return z
        elif ctx.re(m) < -0.5:
            return ctx.inf + z
        else:
            return ctx.nan * z
    x = ctx.fmul(-0.5, z, exact=True)
    y = 0.5+m
    return ctx.exp(x) * z**y * ctx.hyp1f1(y-k, 1+2*m, z, **kwargs)

@defun_wrapped
def whitw(ctx,k,m,z,**kwargs):
    if z == 0:
        g = abs(ctx.re(m))
        if g < 0.5:
            return z
        elif g > 0.5:
            return ctx.inf + z
        else:
            return ctx.nan * z
    x = ctx.fmul(-0.5, z, exact=True)
    y = 0.5+m
    return ctx.exp(x) * z**y * ctx.hyperu(y-k, 1+2*m, z, **kwargs)

@defun
def hyperu(ctx, a, b, z, **kwargs):
    a, atype = ctx._convert_param(a)
    b, btype = ctx._convert_param(b)
    z = ctx.convert(z)
    if not z:
        if ctx.re(b) <= 1:
            return ctx.gammaprod([1-b],[a-b+1])
        else:
            return ctx.inf + z
    bb = 1+a-b
    bb, bbtype = ctx._convert_param(bb)
    try:
        orig = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.hypsum(2, 0, (atype, bbtype), [a, bb], -1/z, maxterms=ctx.prec)
            return v / z**a
        finally:
            ctx.prec = orig
    except ctx.NoConvergence:
        pass
    def h(a,b):
        w = ctx.sinpi(b)
        T1 = ([ctx.pi,w],[1,-1],[],[a-b+1,b],[a],[b],z)
        T2 = ([-ctx.pi,w,z],[1,-1,1-b],[],[a,2-b],[a-b+1],[2-b],z)
        return T1, T2
    return ctx.hypercomb(h, [a,b], **kwargs)

@defun
def struveh(ctx,n,z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/StruveH/26/01/02/
    def h(n):
        return [([z/2, 0.5*ctx.sqrt(ctx.pi)], [n+1, -1], [], [n+1.5], [1], [1.5, n+1.5], -(z/2)**2)]
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def struvel(ctx,n,z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/StruveL/26/01/02/
    def h(n):
        return [([z/2, 0.5*ctx.sqrt(ctx.pi)], [n+1, -1], [], [n+1.5], [1], [1.5, n+1.5], (z/2)**2)]
    return ctx.hypercomb(h, [n], **kwargs)

def _anger(ctx,which,v,z,**kwargs):
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)
    def h(v):
        b = ctx.mpq_1_2
        u = v*b
        m = b*3
        a1,a2,b1,b2 = m-u, m+u, 1-u, 1+u
        c, s = ctx.cospi_sinpi(u)
        if which == 0:
            A, B = [b*z, s], [c]
        if which == 1:
            A, B = [b*z, -c], [s]
        w = ctx.square_exp_arg(z, mult=-0.25)
        T1 = A, [1, 1], [], [a1,a2], [1], [a1,a2], w
        T2 = B, [1], [], [b1,b2], [1], [b1,b2], w
        return T1, T2
    return ctx.hypercomb(h, [v], **kwargs)

@defun
def angerj(ctx, v, z, **kwargs):
    return _anger(ctx, 0, v, z, **kwargs)

@defun
def webere(ctx, v, z, **kwargs):
    return _anger(ctx, 1, v, z, **kwargs)

@defun
def lommels1(ctx, u, v, z, **kwargs):
    u = ctx._convert_param(u)[0]
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)
    def h(u,v):
        b = ctx.mpq_1_2
        w = ctx.square_exp_arg(z, mult=-0.25)
        return ([u-v+1, u+v+1, z], [-1, -1, u+1], [], [], [1], \
            [b*(u-v+3),b*(u+v+3)], w),
    return ctx.hypercomb(h, [u,v], **kwargs)

@defun
def lommels2(ctx, u, v, z, **kwargs):
    u = ctx._convert_param(u)[0]
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)
    # Asymptotic expansion (GR p. 947) -- need to be careful
    # not to use for small arguments
    # def h(u,v):
    #    b = ctx.mpq_1_2
    #    w = -(z/2)**(-2)
    #    return ([z], [u-1], [], [], [b*(1-u+v)], [b*(1-u-v)], w),
    def h(u,v):
        b = ctx.mpq_1_2
        w = ctx.square_exp_arg(z, mult=-0.25)
        T1 = [u-v+1, u+v+1, z], [-1, -1, u+1], [], [], [1], [b*(u-v+3),b*(u+v+3)], w
        T2 = [2, z], [u+v-1, -v], [v, b*(u+v+1)], [b*(v-u+1)], [], [1-v], w
        T3 = [2, z], [u-v-1, v], [-v, b*(u-v+1)], [b*(1-u-v)], [], [1+v], w
        #c1 = ctx.cospi((u-v)*b)
        #c2 = ctx.cospi((u+v)*b)
        #s = ctx.sinpi(v)
        #r1 = (u-v+1)*b
        #r2 = (u+v+1)*b
        #T2 = [c1, s, z, 2], [1, -1, -v, v], [], [-v+1], [], [-v+1], w
        #T3 = [-c2, s, z, 2], [1, -1, v, -v], [], [v+1], [], [v+1], w
        #T2 = [c1, s, z, 2], [1, -1, -v, v+u-1], [r1, r2], [-v+1], [], [-v+1], w
        #T3 = [-c2, s, z, 2], [1, -1, v, -v+u-1], [r1, r2], [v+1], [], [v+1], w
        return T1, T2, T3
    return ctx.hypercomb(h, [u,v], **kwargs)

@defun
def ber(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinBer2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos, sin = ctx.cospi_sinpi(-0.75*n)
        T1 = [cos, z/2], [1, n], [], [n+1], [], [0.5, 0.5*(n+1), 0.5*n+1], r
        T2 = [sin, z/2], [1, n+2], [], [n+2], [], [1.5, 0.5*(n+3), 0.5*n+1], r
        return T1, T2
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def bei(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinBei2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos, sin = ctx.cospi_sinpi(0.75*n)
        T1 = [cos, z/2], [1, n+2], [], [n+2], [], [1.5, 0.5*(n+3), 0.5*n+1], r
        T2 = [sin, z/2], [1, n], [], [n+1], [], [0.5, 0.5*(n+1), 0.5*n+1], r
        return T1, T2
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def ker(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinKer2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos1, sin1 = ctx.cospi_sinpi(0.25*n)
        cos2, sin2 = ctx.cospi_sinpi(0.75*n)
        T1 = [2, z, 4*cos1], [-n-3, n, 1], [-n], [], [], [0.5, 0.5*(1+n), 0.5*(n+2)], r
        T2 = [2, z, -sin1], [-n-3, 2+n, 1], [-n-1], [], [], [1.5, 0.5*(3+n), 0.5*(n+2)], r
        T3 = [2, z, 4*cos2], [n-3, -n, 1], [n], [], [], [0.5, 0.5*(1-n), 1-0.5*n], r
        T4 = [2, z, -sin2], [n-3, 2-n, 1], [n-1], [], [], [1.5, 0.5*(3-n), 1-0.5*n], r
        return T1, T2, T3, T4
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def kei(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinKei2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos1, sin1 = ctx.cospi_sinpi(0.75*n)
        cos2, sin2 = ctx.cospi_sinpi(0.25*n)
        T1 = [-cos1, 2, z], [1, n-3, 2-n], [n-1], [], [], [1.5, 0.5*(3-n), 1-0.5*n], r
        T2 = [-sin1, 2, z], [1, n-1, -n], [n], [], [], [0.5, 0.5*(1-n), 1-0.5*n], r
        T3 = [-sin2, 2, z], [1, -n-1, n], [-n], [], [], [0.5, 0.5*(n+1), 0.5*(n+2)], r
        T4 = [-cos2, 2, z], [1, -n-3, n+2], [-n-1], [], [], [1.5, 0.5*(n+3), 0.5*(n+2)], r
        return T1, T2, T3, T4
    return ctx.hypercomb(h, [n], **kwargs)

# TODO: do this more generically?
def c_memo(f):
    name = f.__name__
    def f_wrapped(ctx):
        cache = ctx._misc_const_cache
        prec = ctx.prec
        p,v = cache.get(name, (-1,0))
        if p >= prec:
            return +v
        else:
            cache[name] = (prec, f(ctx))
            return cache[name][1]
    return f_wrapped

@c_memo
def _airyai_C1(ctx):
    return 1 / (ctx.cbrt(9) * ctx.gamma(ctx.mpf(2)/3))

@c_memo
def _airyai_C2(ctx):
    return -1 / (ctx.cbrt(3) * ctx.gamma(ctx.mpf(1)/3))

@c_memo
def _airybi_C1(ctx):
    return 1 / (ctx.nthroot(3,6) * ctx.gamma(ctx.mpf(2)/3))

@c_memo
def _airybi_C2(ctx):
    return ctx.nthroot(3,6) / ctx.gamma(ctx.mpf(1)/3)

def _airybi_n2_inf(ctx):
    prec = ctx.prec
    try:
        v = ctx.power(3,'2/3')*ctx.gamma('2/3')/(2*ctx.pi)
    finally:
        ctx.prec = prec
    return +v

# Derivatives at z = 0
# TODO: could be expressed more elegantly using triple factorials
def _airyderiv_0(ctx, z, n, ntype, which):
    if ntype == 'Z':
        if n < 0:
            return z
        r = ctx.mpq_1_3
        prec = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.gamma((n+1)*r) * ctx.power(3,n*r) / ctx.pi
            if which == 0:
                v *= ctx.sinpi(2*(n+1)*r)
                v /= ctx.power(3,'2/3')
            else:
                v *= abs(ctx.sinpi(2*(n+1)*r))
                v /= ctx.power(3,'1/6')
        finally:
            ctx.prec = prec
        return +v + z
    else:
        # singular (does the limit exist?)
        raise NotImplementedError

@defun
def airyai(ctx, z, derivative=0, **kwargs):
    z = ctx.convert(z)
    if derivative:
        n, ntype = ctx._convert_param(derivative)
    else:
        n = 0
    # Values at infinities
    if not ctx.isnormal(z) and z:
        if n and ntype == 'Z':
            if n == -1:
                if z == ctx.inf:
                    return ctx.mpf(1)/3 + 1/z
                if z == ctx.ninf:
                    return ctx.mpf(-2)/3 + 1/z
            if n < -1:
                if z == ctx.inf:
                    return z
                if z == ctx.ninf:
                    return (-1)**n * (-z)
        if (not n) and z == ctx.inf or z == ctx.ninf:
            return 1/z
        # TODO: limits
        raise ValueError("essential singularity of Ai(z)")
    # Account for exponential scaling
    if z:
        extraprec = max(0, int(1.5*ctx.mag(z)))
    else:
        extraprec = 0
    if n:
        if n == 1:
            def h():
                # http://functions.wolfram.com/03.07.06.0005.01
                if ctx._re(z) > 4:
                    ctx.prec += extraprec
                    w = z**1.5; r = -0.75/w; u = -2*w/3
                    ctx.prec -= extraprec
                    C = -ctx.exp(u)/(2*ctx.sqrt(ctx.pi))*ctx.nthroot(z,4)
                    return ([C],[1],[],[],[(-1,6),(7,6)],[],r),
                # http://functions.wolfram.com/03.07.26.0001.01
                else:
                    ctx.prec += extraprec
                    w = z**3 / 9
                    ctx.prec -= extraprec
                    C1 = _airyai_C1(ctx) * 0.5
                    C2 = _airyai_C2(ctx)
                    T1 = [C1,z],[1,2],[],[],[],[ctx.mpq_5_3],w
                    T2 = [C2],[1],[],[],[],[ctx.mpq_1_3],w
                    return T1, T2
            return ctx.hypercomb(h, [], **kwargs)
        else:
            if z == 0:
                return _airyderiv_0(ctx, z, n, ntype, 0)
            # http://functions.wolfram.com/03.05.20.0004.01
            def h(n):
                ctx.prec += extraprec
                w = z**3/9
                ctx.prec -= extraprec
                q13,q23,q43 = ctx.mpq_1_3, ctx.mpq_2_3, ctx.mpq_4_3
                a1=q13; a2=1; b1=(1-n)*q13; b2=(2-n)*q13; b3=1-n*q13
                T1 = [3, z], [n-q23, -n], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                a1=q23; b1=(2-n)*q13; b2=1-n*q13; b3=(4-n)*q13
                T2 = [3, z, -z], [n-q43, -n, 1], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                return T1, T2
            v = ctx.hypercomb(h, [n], **kwargs)
            if ctx._is_real_type(z) and ctx.isint(n):
                v = ctx._re(v)
            return v
    else:
        def h():
            if ctx._re(z) > 4:
                # We could use 1F1, but it results in huge cancellation;
                # the following expansion is better.
                # TODO: asymptotic series for derivatives
                ctx.prec += extraprec
                w = z**1.5; r = -0.75/w; u = -2*w/3
                ctx.prec -= extraprec
                C = ctx.exp(u)/(2*ctx.sqrt(ctx.pi)*ctx.nthroot(z,4))
                return ([C],[1],[],[],[(1,6),(5,6)],[],r),
            else:
                ctx.prec += extraprec
                w = z**3 / 9
                ctx.prec -= extraprec
                C1 = _airyai_C1(ctx)
                C2 = _airyai_C2(ctx)
                T1 = [C1],[1],[],[],[],[ctx.mpq_2_3],w
                T2 = [z*C2],[1],[],[],[],[ctx.mpq_4_3],w
                return T1, T2
        return ctx.hypercomb(h, [], **kwargs)

@defun
def airybi(ctx, z, derivative=0, **kwargs):
    z = ctx.convert(z)
    if derivative:
        n, ntype = ctx._convert_param(derivative)
    else:
        n = 0
    # Values at infinities
    if not ctx.isnormal(z) and z:
        if n and ntype == 'Z':
            if z == ctx.inf:
                return z
            if z == ctx.ninf:
                if n == -1:
                    return 1/z
                if n == -2:
                    return _airybi_n2_inf(ctx)
                if n < -2:
                    return (-1)**n * (-z)
        if not n:
            if z == ctx.inf:
                return z
            if z == ctx.ninf:
                return 1/z
        # TODO: limits
        raise ValueError("essential singularity of Bi(z)")
    if z:
        extraprec = max(0, int(1.5*ctx.mag(z)))
    else:
        extraprec = 0
    if n:
        if n == 1:
            # http://functions.wolfram.com/03.08.26.0001.01
            def h():
                ctx.prec += extraprec
                w = z**3 / 9
                ctx.prec -= extraprec
                C1 = _airybi_C1(ctx)*0.5
                C2 = _airybi_C2(ctx)
                T1 = [C1,z],[1,2],[],[],[],[ctx.mpq_5_3],w
                T2 = [C2],[1],[],[],[],[ctx.mpq_1_3],w
                return T1, T2
            return ctx.hypercomb(h, [], **kwargs)
        else:
            if z == 0:
                return _airyderiv_0(ctx, z, n, ntype, 1)
            def h(n):
                ctx.prec += extraprec
                w = z**3/9
                ctx.prec -= extraprec
                q13,q23,q43 = ctx.mpq_1_3, ctx.mpq_2_3, ctx.mpq_4_3
                q16 = ctx.mpq_1_6
                q56 = ctx.mpq_5_6
                a1=q13; a2=1; b1=(1-n)*q13; b2=(2-n)*q13; b3=1-n*q13
                T1 = [3, z], [n-q16, -n], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                a1=q23; b1=(2-n)*q13; b2=1-n*q13; b3=(4-n)*q13
                T2 = [3, z], [n-q56, 1-n], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                return T1, T2
            v = ctx.hypercomb(h, [n], **kwargs)
            if ctx._is_real_type(z) and ctx.isint(n):
                v = ctx._re(v)
            return v
    else:
        def h():
            ctx.prec += extraprec
            w = z**3 / 9
            ctx.prec -= extraprec
            C1 = _airybi_C1(ctx)
            C2 = _airybi_C2(ctx)
            T1 = [C1],[1],[],[],[],[ctx.mpq_2_3],w
            T2 = [z*C2],[1],[],[],[],[ctx.mpq_4_3],w
            return T1, T2
        return ctx.hypercomb(h, [], **kwargs)

def _airy_zero(ctx, which, k, derivative, complex=False):
    # Asymptotic formulas are given in DLMF section 9.9
    def U(t): return t**(2/3.)*(1-7/(t**2*48))
    def T(t): return t**(2/3.)*(1+5/(t**2*48))
    k = int(k)
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if not derivative in (0,1):
        raise ValueError("Derivative should lie between 0 and 1")
    if which == 0:
        if derivative:
            return ctx.findroot(lambda z: ctx.airyai(z,1),
                -U(3*ctx.pi*(4*k-3)/8))
        return ctx.findroot(ctx.airyai, -T(3*ctx.pi*(4*k-1)/8))
    if which == 1 and complex == False:
        if derivative:
            return ctx.findroot(lambda z: ctx.airybi(z,1),
                -U(3*ctx.pi*(4*k-1)/8))
        return ctx.findroot(ctx.airybi, -T(3*ctx.pi*(4*k-3)/8))
    if which == 1 and complex == True:
        if derivative:
            t = 3*ctx.pi*(4*k-3)/8 + 0.75j*ctx.ln2
            s = ctx.expjpi(ctx.mpf(1)/3) * T(t)
            return ctx.findroot(lambda z: ctx.airybi(z,1), s)
        t = 3*ctx.pi*(4*k-1)/8 + 0.75j*ctx.ln2
        s = ctx.expjpi(ctx.mpf(1)/3) * U(t)
        return ctx.findroot(ctx.airybi, s)

@defun
def airyaizero(ctx, k, derivative=0):
    return _airy_zero(ctx, 0, k, derivative, False)

@defun
def airybizero(ctx, k, derivative=0, complex=False):
    return _airy_zero(ctx, 1, k, derivative, complex)

def _scorer(ctx, z, which, kwargs):
    z = ctx.convert(z)
    if ctx.isinf(z):
        if z == ctx.inf:
            if which == 0: return 1/z
            if which == 1: return z
        if z == ctx.ninf:
            return 1/z
        raise ValueError("essential singularity")
    if z:
        extraprec = max(0, int(1.5*ctx.mag(z)))
    else:
        extraprec = 0
    if kwargs.get('derivative'):
        raise NotImplementedError
    # Direct asymptotic expansions, to avoid
    # exponentially large cancellation
    try:
        if ctx.mag(z) > 3:
            if which == 0 and abs(ctx.arg(z)) < ctx.pi/3 * 0.999:
                def h():
                    return (([ctx.pi,z],[-1,-1],[],[],[(1,3),(2,3),1],[],9/z**3),)
                return ctx.hypercomb(h, [], maxterms=ctx.prec, force_series=True)
            if which == 1 and abs(ctx.arg(-z)) < 2*ctx.pi/3 * 0.999:
                def h():
                    return (([-ctx.pi,z],[-1,-1],[],[],[(1,3),(2,3),1],[],9/z**3),)
                return ctx.hypercomb(h, [], maxterms=ctx.prec, force_series=True)
    except ctx.NoConvergence:
        pass
    def h():
        A = ctx.airybi(z, **kwargs)/3
        B = -2*ctx.pi
        if which == 1:
            A *= 2
            B *= -1
        ctx.prec += extraprec
        w = z**3/9
        ctx.prec -= extraprec
        T1 = [A], [1], [], [], [], [], 0
        T2 = [B,z], [-1,2], [], [], [1], [ctx.mpq_4_3,ctx.mpq_5_3], w
        return T1, T2
    return ctx.hypercomb(h, [], **kwargs)

@defun
def scorergi(ctx, z, **kwargs):
    return _scorer(ctx, z, 0, kwargs)

@defun
def scorerhi(ctx, z, **kwargs):
    return _scorer(ctx, z, 1, kwargs)

@defun_wrapped
def coulombc(ctx, l, eta, _cache={}):
    if (l, eta) in _cache and _cache[l,eta][0] >= ctx.prec:
        return +_cache[l,eta][1]
    G3 = ctx.loggamma(2*l+2)
    G1 = ctx.loggamma(1+l+ctx.j*eta)
    G2 = ctx.loggamma(1+l-ctx.j*eta)
    v = 2**l * ctx.exp((-ctx.pi*eta+G1+G2)/2 - G3)
    if not (ctx.im(l) or ctx.im(eta)):
        v = ctx.re(v)
    _cache[l,eta] = (ctx.prec, v)
    return v

@defun_wrapped
def coulombf(ctx, l, eta, z, w=1, chop=True, **kwargs):
    # Regular Coulomb wave function
    # Note: w can be either 1 or -1; the other may be better in some cases
    # TODO: check that chop=True chops when and only when it should
    #ctx.prec += 10
    def h(l, eta):
        try:
            jw = ctx.j*w
            jwz = ctx.fmul(jw, z, exact=True)
            jwz2 = ctx.fmul(jwz, -2, exact=True)
            C = ctx.coulombc(l, eta)
            T1 = [C, z, ctx.exp(jwz)], [1, l+1, 1], [], [], [1+l+jw*eta], \
                [2*l+2], jwz2
        except ValueError:
            T1 = [0], [-1], [], [], [], [], 0
        return (T1,)
    v = ctx.hypercomb(h, [l,eta], **kwargs)
    if chop and (not ctx.im(l)) and (not ctx.im(eta)) and (not ctx.im(z)) and \
        (ctx.re(z) >= 0):
        v = ctx.re(v)
    return v

@defun_wrapped
def _coulomb_chi(ctx, l, eta, _cache={}):
    if (l, eta) in _cache and _cache[l,eta][0] >= ctx.prec:
        return _cache[l,eta][1]
    def terms():
        l2 = -l-1
        jeta = ctx.j*eta
        return [ctx.loggamma(1+l+jeta) * (-0.5j),
            ctx.loggamma(1+l-jeta) * (0.5j),
            ctx.loggamma(1+l2+jeta) * (0.5j),
            ctx.loggamma(1+l2-jeta) * (-0.5j),
            -(l+0.5)*ctx.pi]
    v = ctx.sum_accurately(terms, 1)
    _cache[l,eta] = (ctx.prec, v)
    return v

@defun_wrapped
def coulombg(ctx, l, eta, z, w=1, chop=True, **kwargs):
    # Irregular Coulomb wave function
    # Note: w can be either 1 or -1; the other may be better in some cases
    # TODO: check that chop=True chops when and only when it should
    if not ctx._im(l):
        l = ctx._re(l)  # XXX: for isint
    def h(l, eta):
        # Force perturbation for integers and half-integers
        if ctx.isint(l*2):
            T1 = [0], [-1], [], [], [], [], 0
            return (T1,)
        l2 = -l-1
        try:
            chi = ctx._coulomb_chi(l, eta)
            jw = ctx.j*w
            s = ctx.sin(chi); c = ctx.cos(chi)
            C1 = ctx.coulombc(l,eta)
            C2 = ctx.coulombc(l2,eta)
            u = ctx.exp(jw*z)
            x = -2*jw*z
            T1 = [s, C1, z, u, c], [-1, 1, l+1, 1, 1], [], [], \
                [1+l+jw*eta], [2*l+2], x
            T2 = [-s, C2, z, u],   [-1, 1, l2+1, 1],    [], [], \
                [1+l2+jw*eta], [2*l2+2], x
            return T1, T2
        except ValueError:
            T1 = [0], [-1], [], [], [], [], 0
            return (T1,)
    v = ctx.hypercomb(h, [l,eta], **kwargs)
    if chop and (not ctx._im(l)) and (not ctx._im(eta)) and (not ctx._im(z)) and \
        (ctx._re(z) >= 0):
        v = ctx._re(v)
    return v

def mcmahon(ctx,kind,prime,v,m):
    """
    Computes an estimate for the location of the Bessel function zero
    j_{v,m}, y_{v,m}, j'_{v,m} or y'_{v,m} using McMahon's asymptotic
    expansion (Abramowitz & Stegun 9.5.12-13, DLMF 20.21(vi)).

    Returns (r,err) where r is the estimated location of the root
    and err is a positive number estimating the error of the
    asymptotic expansion.
    """
    u = 4*v**2
    if kind == 1 and not prime: b = (4*m+2*v-1)*ctx.pi/4
    if kind == 2 and not prime: b = (4*m+2*v-3)*ctx.pi/4
    if kind == 1 and prime: b = (4*m+2*v-3)*ctx.pi/4
    if kind == 2 and prime: b = (4*m+2*v-1)*ctx.pi/4
    if not prime:
        s1 = b
        s2 = -(u-1)/(8*b)
        s3 = -4*(u-1)*(7*u-31)/(3*(8*b)**3)
        s4 = -32*(u-1)*(83*u**2-982*u+3779)/(15*(8*b)**5)
        s5 = -64*(u-1)*(6949*u**3-153855*u**2+1585743*u-6277237)/(105*(8*b)**7)
    if prime:
        s1 = b
        s2 = -(u+3)/(8*b)
        s3 = -4*(7*u**2+82*u-9)/(3*(8*b)**3)
        s4 = -32*(83*u**3+2075*u**2-3039*u+3537)/(15*(8*b)**5)
        s5 = -64*(6949*u**4+296492*u**3-1248002*u**2+7414380*u-5853627)/(105*(8*b)**7)
    terms = [s1,s2,s3,s4,s5]
    s = s1
    err = 0.0
    for i in range(1,len(terms)):
        if abs(terms[i]) < abs(terms[i-1]):
            s += terms[i]
        else:
            err = abs(terms[i])
    if i == len(terms)-1:
        err = abs(terms[-1])
    return s, err

def generalized_bisection(ctx,f,a,b,n):
    """
    Given f known to have exactly n simple roots within [a,b],
    return a list of n intervals isolating the roots
    and having opposite signs at the endpoints.

    TODO: this can be optimized, e.g. by reusing evaluation points.
    """
    if n < 1:
        raise ValueError("n cannot be less than 1")
    N = n+1
    points = []
    signs = []
    while 1:
        points = ctx.linspace(a,b,N)
        signs = [ctx.sign(f(x)) for x in points]
        ok_intervals = [(points[i],points[i+1]) for i in range(N-1) \
            if signs[i]*signs[i+1] == -1]
        if len(ok_intervals) == n:
            return ok_intervals
        N = N*2

def find_in_interval(ctx, f, ab):
    return ctx.findroot(f, ab, solver='illinois', verify=False)

def bessel_zero(ctx, kind, prime, v, m, isoltol=0.01, _interval_cache={}):
    prec = ctx.prec
    workprec = max(prec, ctx.mag(v), ctx.mag(m))+10
    try:
        ctx.prec = workprec
        v = ctx.mpf(v)
        m = int(m)
        prime = int(prime)
        if v < 0:
            raise ValueError("v cannot be negative")
        if m < 1:
            raise ValueError("m cannot be less than 1")
        if not prime in (0,1):
            raise ValueError("prime should lie between 0 and 1")
        if kind == 1:
            if prime: f = lambda x: ctx.besselj(v,x,derivative=1)
            else:     f = lambda x: ctx.besselj(v,x)
        if kind == 2:
            if prime: f = lambda x: ctx.bessely(v,x,derivative=1)
            else:     f = lambda x: ctx.bessely(v,x)
        # The first root of J' is very close to 0 for small
        # orders, and this needs to be special-cased
        if kind == 1 and prime and m == 1:
            if v == 0:
                return ctx.zero
            if v <= 1:
                # TODO: use v <= j'_{v,1} < y_{v,1}?
                r = 2*ctx.sqrt(v*(1+v)/(v+2))
                return find_in_interval(ctx, f, (r/10, 2*r))
        if (kind,prime,v,m) in _interval_cache:
            return find_in_interval(ctx, f, _interval_cache[kind,prime,v,m])
        r, err = mcmahon(ctx, kind, prime, v, m)
        if err < isoltol:
            return find_in_interval(ctx, f, (r-isoltol, r+isoltol))
        # An x such that 0 < x < r_{v,1}
        if kind == 1 and not prime: low = 2.4
        if kind == 1 and prime: low = 1.8
        if kind == 2 and not prime: low = 0.8
        if kind == 2 and prime: low = 2.0
        n = m+1
        while 1:
            r1, err = mcmahon(ctx, kind, prime, v, n)
            if err < isoltol:
                r2, err2 = mcmahon(ctx, kind, prime, v, n+1)
                intervals = generalized_bisection(ctx, f, low, 0.5*(r1+r2), n)
                for k, ab in enumerate(intervals):
                    _interval_cache[kind,prime,v,k+1] = ab
                return find_in_interval(ctx, f, intervals[m-1])
            else:
                n = n*2
    finally:
        ctx.prec = prec

@defun
def besseljzero(ctx, v, m, derivative=0):
    r"""
    For a real order `\nu \ge 0` and a positive integer `m`, returns
    `j_{\nu,m}`, the `m`-th positive zero of the Bessel function of the
    first kind `J_{\nu}(z)` (see :func:`~mpmath.besselj`). Alternatively,
    with *derivative=1*, gives the first nonnegative simple zero
    `j'_{\nu,m}` of `J'_{\nu}(z)`.

    The indexing convention is that used by Abramowitz & Stegun
    and the DLMF. Note the special case `j'_{0,1} = 0`, while all other
    zeros are positive. In effect, only simple zeros are counted
    (all zeros of Bessel functions are simple except possibly `z = 0`)
    and `j_{\nu,m}` becomes a monotonic function of both `\nu`
    and `m`.

    The zeros are interlaced according to the inequalities

    .. math ::

        j'_{\nu,k} < j_{\nu,k} < j'_{\nu,k+1}

        j_{\nu,1} < j_{\nu+1,2} < j_{\nu,2} < j_{\nu+1,2} < j_{\nu,3} < \cdots

    **Examples**

    Initial zeros of the Bessel functions `J_0(z), J_1(z), J_2(z)`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> besseljzero(0,1); besseljzero(0,2); besseljzero(0,3)
        2.404825557695772768621632
        5.520078110286310649596604
        8.653727912911012216954199
        >>> besseljzero(1,1); besseljzero(1,2); besseljzero(1,3)
        3.831705970207512315614436
        7.01558666981561875353705
        10.17346813506272207718571
        >>> besseljzero(2,1); besseljzero(2,2); besseljzero(2,3)
        5.135622301840682556301402
        8.417244140399864857783614
        11.61984117214905942709415

    Initial zeros of `J'_0(z), J'_1(z), J'_2(z)`::

        0.0
        3.831705970207512315614436
        7.01558666981561875353705
        >>> besseljzero(1,1,1); besseljzero(1,2,1); besseljzero(1,3,1)
        1.84118378134065930264363
        5.331442773525032636884016
        8.536316366346285834358961
        >>> besseljzero(2,1,1); besseljzero(2,2,1); besseljzero(2,3,1)
        3.054236928227140322755932
        6.706133194158459146634394
        9.969467823087595793179143

    Zeros with large index::

        >>> besseljzero(0,100); besseljzero(0,1000); besseljzero(0,10000)
        313.3742660775278447196902
        3140.807295225078628895545
        31415.14114171350798533666
        >>> besseljzero(5,100); besseljzero(5,1000); besseljzero(5,10000)
        321.1893195676003157339222
        3148.657306813047523500494
        31422.9947255486291798943
        >>> besseljzero(0,100,1); besseljzero(0,1000,1); besseljzero(0,10000,1)
        311.8018681873704508125112
        3139.236339643802482833973
        31413.57032947022399485808

    Zeros of functions with large order::

        >>> besseljzero(50,1)
        57.11689916011917411936228
        >>> besseljzero(50,2)
        62.80769876483536093435393
        >>> besseljzero(50,100)
        388.6936600656058834640981
        >>> besseljzero(50,1,1)
        52.99764038731665010944037
        >>> besseljzero(50,2,1)
        60.02631933279942589882363
        >>> besseljzero(50,100,1)
        387.1083151608726181086283

    Zeros of functions with fractional order::

        >>> besseljzero(0.5,1); besseljzero(1.5,1); besseljzero(2.25,4)
        3.141592653589793238462643
        4.493409457909064175307881
        15.15657692957458622921634

    Both `J_{\nu}(z)` and `J'_{\nu}(z)` can be expressed as infinite
    products over their zeros::

        >>> v,z = 2, mpf(1)
        >>> (z/2)**v/gamma(v+1) * \
        ...     nprod(lambda k: 1-(z/besseljzero(v,k))**2, [1,inf])
        ...
        0.1149034849319004804696469
        >>> besselj(v,z)
        0.1149034849319004804696469
        >>> (z/2)**(v-1)/2/gamma(v) * \
        ...     nprod(lambda k: 1-(z/besseljzero(v,k,1))**2, [1,inf])
        ...
        0.2102436158811325550203884
        >>> besselj(v,z,1)
        0.2102436158811325550203884

    """
    return +bessel_zero(ctx, 1, derivative, v, m)

@defun
def besselyzero(ctx, v, m, derivative=0):
    r"""
    For a real order `\nu \ge 0` and a positive integer `m`, returns
    `y_{\nu,m}`, the `m`-th positive zero of the Bessel function of the
    second kind `Y_{\nu}(z)` (see :func:`~mpmath.bessely`). Alternatively,
    with *derivative=1*, gives the first positive zero `y'_{\nu,m}` of
    `Y'_{\nu}(z)`.

    The zeros are interlaced according to the inequalities

    .. math ::

        y_{\nu,k} < y'_{\nu,k} < y_{\nu,k+1}

        y_{\nu,1} < y_{\nu+1,2} < y_{\nu,2} < y_{\nu+1,2} < y_{\nu,3} < \cdots

    **Examples**

    Initial zeros of the Bessel functions `Y_0(z), Y_1(z), Y_2(z)`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> besselyzero(0,1); besselyzero(0,2); besselyzero(0,3)
        0.8935769662791675215848871
        3.957678419314857868375677
        7.086051060301772697623625
        >>> besselyzero(1,1); besselyzero(1,2); besselyzero(1,3)
        2.197141326031017035149034
        5.429681040794135132772005
        8.596005868331168926429606
        >>> besselyzero(2,1); besselyzero(2,2); besselyzero(2,3)
        3.384241767149593472701426
        6.793807513268267538291167
        10.02347797936003797850539

    Initial zeros of `Y'_0(z), Y'_1(z), Y'_2(z)`::

        >>> besselyzero(0,1,1); besselyzero(0,2,1); besselyzero(0,3,1)
        2.197141326031017035149034
        5.429681040794135132772005
        8.596005868331168926429606
        >>> besselyzero(1,1,1); besselyzero(1,2,1); besselyzero(1,3,1)
        3.683022856585177699898967
        6.941499953654175655751944
        10.12340465543661307978775
        >>> besselyzero(2,1,1); besselyzero(2,2,1); besselyzero(2,3,1)
        5.002582931446063945200176
        8.350724701413079526349714
        11.57419546521764654624265

    Zeros with large index::

        >>> besselyzero(0,100); besselyzero(0,1000); besselyzero(0,10000)
        311.8034717601871549333419
        3139.236498918198006794026
        31413.57034538691205229188
        >>> besselyzero(5,100); besselyzero(5,1000); besselyzero(5,10000)
        319.6183338562782156235062
        3147.086508524556404473186
        31421.42392920214673402828
        >>> besselyzero(0,100,1); besselyzero(0,1000,1); besselyzero(0,10000,1)
        313.3726705426359345050449
        3140.807136030340213610065
        31415.14112579761578220175

    Zeros of functions with large order::

        >>> besselyzero(50,1)
        53.50285882040036394680237
        >>> besselyzero(50,2)
        60.11244442774058114686022
        >>> besselyzero(50,100)
        387.1096509824943957706835
        >>> besselyzero(50,1,1)
        56.96290427516751320063605
        >>> besselyzero(50,2,1)
        62.74888166945933944036623
        >>> besselyzero(50,100,1)
        388.6923300548309258355475

    Zeros of functions with fractional order::

        >>> besselyzero(0.5,1); besselyzero(1.5,1); besselyzero(2.25,4)
        1.570796326794896619231322
        2.798386045783887136720249
        13.56721208770735123376018

    """
    return +bessel_zero(ctx, 2, derivative, v, m)
