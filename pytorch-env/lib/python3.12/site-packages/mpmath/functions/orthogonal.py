from .functions import defun, defun_wrapped

def _hermite_param(ctx, n, z, parabolic_cylinder):
    """
    Combined calculation of the Hermite polynomial H_n(z) (and its
    generalization to complex n) and the parabolic cylinder
    function D.
    """
    n, ntyp = ctx._convert_param(n)
    z = ctx.convert(z)
    q = -ctx.mpq_1_2
    # For re(z) > 0, 2F0 -- http://functions.wolfram.com/
    #     HypergeometricFunctions/HermiteHGeneral/06/02/0009/
    # Otherwise, there is a reflection formula
    # 2F0 + http://functions.wolfram.com/HypergeometricFunctions/
    #           HermiteHGeneral/16/01/01/0006/
    #
    # TODO:
    # An alternative would be to use
    # http://functions.wolfram.com/HypergeometricFunctions/
    #     HermiteHGeneral/06/02/0006/
    #
    # Also, the 1F1 expansion
    # http://functions.wolfram.com/HypergeometricFunctions/
    #     HermiteHGeneral/26/01/02/0001/
    # should probably be used for tiny z
    if not z:
        T1 = [2, ctx.pi], [n, 0.5], [], [q*(n-1)], [], [], 0
        if parabolic_cylinder:
            T1[1][0] += q*n
        return T1,
    can_use_2f0 = ctx.isnpint(-n) or ctx.re(z) > 0 or \
        (ctx.re(z) == 0 and ctx.im(z) > 0)
    expprec = ctx.prec*4 + 20
    if parabolic_cylinder:
        u = ctx.fmul(ctx.fmul(z,z,prec=expprec), -0.25, exact=True)
        w = ctx.fmul(z, ctx.sqrt(0.5,prec=expprec), prec=expprec)
    else:
        w = z
    w2 = ctx.fmul(w, w, prec=expprec)
    rw2 = ctx.fdiv(1, w2, prec=expprec)
    nrw2 = ctx.fneg(rw2, exact=True)
    nw = ctx.fneg(w, exact=True)
    if can_use_2f0:
        T1 = [2, w], [n, n], [], [], [q*n, q*(n-1)], [], nrw2
        terms = [T1]
    else:
        T1 = [2, nw], [n, n], [], [], [q*n, q*(n-1)], [], nrw2
        T2 = [2, ctx.pi, nw], [n+2, 0.5, 1], [], [q*n], [q*(n-1)], [1-q], w2
        terms = [T1,T2]
    # Multiply by prefactor for D_n
    if parabolic_cylinder:
        expu = ctx.exp(u)
        for i in range(len(terms)):
            terms[i][1][0] += q*n
            terms[i][0].append(expu)
            terms[i][1].append(1)
    return tuple(terms)

@defun
def hermite(ctx, n, z, **kwargs):
    return ctx.hypercomb(lambda: _hermite_param(ctx, n, z, 0), [], **kwargs)

@defun
def pcfd(ctx, n, z, **kwargs):
    r"""
    Gives the parabolic cylinder function in Whittaker's notation
    `D_n(z) = U(-n-1/2, z)` (see :func:`~mpmath.pcfu`).
    It solves the differential equation

    .. math ::

        y'' + \left(n + \frac{1}{2} - \frac{1}{4} z^2\right) y = 0.

    and can be represented in terms of Hermite polynomials
    (see :func:`~mpmath.hermite`) as

    .. math ::

        D_n(z) = 2^{-n/2} e^{-z^2/4} H_n\left(\frac{z}{\sqrt{2}}\right).

    **Plots**

    .. literalinclude :: /plots/pcfd.py
    .. image :: /plots/pcfd.png

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> pcfd(0,0); pcfd(1,0); pcfd(2,0); pcfd(3,0)
        1.0
        0.0
        -1.0
        0.0
        >>> pcfd(4,0); pcfd(-3,0)
        3.0
        0.6266570686577501256039413
        >>> pcfd('1/2', 2+3j)
        (-5.363331161232920734849056 - 3.858877821790010714163487j)
        >>> pcfd(2, -10)
        1.374906442631438038871515e-9

    Verifying the differential equation::

        >>> n = mpf(2.5)
        >>> y = lambda z: pcfd(n,z)
        >>> z = 1.75
        >>> chop(diff(y,z,2) + (n+0.5-0.25*z**2)*y(z))
        0.0

    Rational Taylor series expansion when `n` is an integer::

        >>> taylor(lambda z: pcfd(5,z), 0, 7)
        [0.0, 15.0, 0.0, -13.75, 0.0, 3.96875, 0.0, -0.6015625]

    """
    return ctx.hypercomb(lambda: _hermite_param(ctx, n, z, 1), [], **kwargs)

@defun
def pcfu(ctx, a, z, **kwargs):
    r"""
    Gives the parabolic cylinder function `U(a,z)`, which may be
    defined for `\Re(z) > 0` in terms of the confluent
    U-function (see :func:`~mpmath.hyperu`) by

    .. math ::

        U(a,z) = 2^{-\frac{1}{4}-\frac{a}{2}} e^{-\frac{1}{4} z^2}
            U\left(\frac{a}{2}+\frac{1}{4},
            \frac{1}{2}, \frac{1}{2}z^2\right)

    or, for arbitrary `z`,

    .. math ::

        e^{-\frac{1}{4}z^2} U(a,z) =
            U(a,0) \,_1F_1\left(-\tfrac{a}{2}+\tfrac{1}{4};
            \tfrac{1}{2}; -\tfrac{1}{2}z^2\right) +
            U'(a,0) z \,_1F_1\left(-\tfrac{a}{2}+\tfrac{3}{4};
            \tfrac{3}{2}; -\tfrac{1}{2}z^2\right).

    **Examples**

    Connection to other functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> z = mpf(3)
        >>> pcfu(0.5,z)
        0.03210358129311151450551963
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(z/sqrt(2))
        0.03210358129311151450551963
        >>> pcfu(0.5,-z)
        23.75012332835297233711255
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(-z/sqrt(2))
        23.75012332835297233711255
        >>> pcfu(0.5,-z)
        23.75012332835297233711255
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(-z/sqrt(2))
        23.75012332835297233711255

    """
    n, _ = ctx._convert_param(a)
    return ctx.pcfd(-n-ctx.mpq_1_2, z)

@defun
def pcfv(ctx, a, z, **kwargs):
    r"""
    Gives the parabolic cylinder function `V(a,z)`, which can be
    represented in terms of :func:`~mpmath.pcfu` as

    .. math ::

        V(a,z) = \frac{\Gamma(a+\tfrac{1}{2}) (U(a,-z)-\sin(\pi a) U(a,z)}{\pi}.

    **Examples**

    Wronskian relation between `U` and `V`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a, z = 2, 3
        >>> pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z)
        0.7978845608028653558798921
        >>> sqrt(2/pi)
        0.7978845608028653558798921
        >>> a, z = 2.5, 3
        >>> pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z)
        0.7978845608028653558798921
        >>> a, z = 0.25, -1
        >>> pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z)
        0.7978845608028653558798921
        >>> a, z = 2+1j, 2+3j
        >>> chop(pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z))
        0.7978845608028653558798921

    """
    n, ntype = ctx._convert_param(a)
    z = ctx.convert(z)
    q = ctx.mpq_1_2
    r = ctx.mpq_1_4
    if ntype == 'Q' and ctx.isint(n*2):
        # Faster for half-integers
        def h():
            jz = ctx.fmul(z, -1j, exact=True)
            T1terms = _hermite_param(ctx, -n-q, z, 1)
            T2terms = _hermite_param(ctx, n-q, jz, 1)
            for T in T1terms:
                T[0].append(1j)
                T[1].append(1)
                T[3].append(q-n)
            u = ctx.expjpi((q*n-r)) * ctx.sqrt(2/ctx.pi)
            for T in T2terms:
                T[0].append(u)
                T[1].append(1)
            return T1terms + T2terms
        v = ctx.hypercomb(h, [], **kwargs)
        if ctx._is_real_type(n) and ctx._is_real_type(z):
            v = ctx._re(v)
        return v
    else:
        def h(n):
            w = ctx.square_exp_arg(z, -0.25)
            u = ctx.square_exp_arg(z, 0.5)
            e = ctx.exp(w)
            l = [ctx.pi, q, ctx.exp(w)]
            Y1 = l, [-q, n*q+r, 1], [r-q*n], [], [q*n+r], [q], u
            Y2 = l + [z], [-q, n*q-r, 1, 1], [1-r-q*n], [], [q*n+1-r], [1+q], u
            c, s = ctx.cospi_sinpi(r+q*n)
            Y1[0].append(s)
            Y2[0].append(c)
            for Y in (Y1, Y2):
                Y[1].append(1)
                Y[3].append(q-n)
            return Y1, Y2
        return ctx.hypercomb(h, [n], **kwargs)


@defun
def pcfw(ctx, a, z, **kwargs):
    r"""
    Gives the parabolic cylinder function `W(a,z)` defined in (DLMF 12.14).

    **Examples**

    Value at the origin::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a = mpf(0.25)
        >>> pcfw(a,0)
        0.9722833245718180765617104
        >>> power(2,-0.75)*sqrt(abs(gamma(0.25+0.5j*a)/gamma(0.75+0.5j*a)))
        0.9722833245718180765617104
        >>> diff(pcfw,(a,0),(0,1))
        -0.5142533944210078966003624
        >>> -power(2,-0.25)*sqrt(abs(gamma(0.75+0.5j*a)/gamma(0.25+0.5j*a)))
        -0.5142533944210078966003624

    """
    n, _ = ctx._convert_param(a)
    z = ctx.convert(z)
    def terms():
        phi2 = ctx.arg(ctx.gamma(0.5 + ctx.j*n))
        phi2 = (ctx.loggamma(0.5+ctx.j*n) - ctx.loggamma(0.5-ctx.j*n))/2j
        rho = ctx.pi/8 + 0.5*phi2
        # XXX: cancellation computing k
        k = ctx.sqrt(1 + ctx.exp(2*ctx.pi*n)) - ctx.exp(ctx.pi*n)
        C = ctx.sqrt(k/2) * ctx.exp(0.25*ctx.pi*n)
        yield C * ctx.expj(rho) * ctx.pcfu(ctx.j*n, z*ctx.expjpi(-0.25))
        yield C * ctx.expj(-rho) * ctx.pcfu(-ctx.j*n, z*ctx.expjpi(0.25))
    v = ctx.sum_accurately(terms)
    if ctx._is_real_type(n) and ctx._is_real_type(z):
        v = ctx._re(v)
    return v

"""
Even/odd PCFs. Useful?

@defun
def pcfy1(ctx, a, z, **kwargs):
    a, _ = ctx._convert_param(n)
    z = ctx.convert(z)
    def h():
        w = ctx.square_exp_arg(z)
        w1 = ctx.fmul(w, -0.25, exact=True)
        w2 = ctx.fmul(w, 0.5, exact=True)
        e = ctx.exp(w1)
        return [e], [1], [], [], [ctx.mpq_1_2*a+ctx.mpq_1_4], [ctx.mpq_1_2], w2
    return ctx.hypercomb(h, [], **kwargs)

@defun
def pcfy2(ctx, a, z, **kwargs):
    a, _ = ctx._convert_param(n)
    z = ctx.convert(z)
    def h():
        w = ctx.square_exp_arg(z)
        w1 = ctx.fmul(w, -0.25, exact=True)
        w2 = ctx.fmul(w, 0.5, exact=True)
        e = ctx.exp(w1)
        return [e, z], [1, 1], [], [], [ctx.mpq_1_2*a+ctx.mpq_3_4], \
            [ctx.mpq_3_2], w2
    return ctx.hypercomb(h, [], **kwargs)
"""

@defun_wrapped
def gegenbauer(ctx, n, a, z, **kwargs):
    # Special cases: a+0.5, a*2 poles
    if ctx.isnpint(a):
        return 0*(z+n)
    if ctx.isnpint(a+0.5):
        # TODO: something else is required here
        # E.g.: gegenbauer(-2, -0.5, 3) == -12
        if ctx.isnpint(n+1):
            raise NotImplementedError("Gegenbauer function with two limits")
        def h(a):
            a2 = 2*a
            T = [], [], [n+a2], [n+1, a2], [-n, n+a2], [a+0.5], 0.5*(1-z)
            return [T]
        return ctx.hypercomb(h, [a], **kwargs)
    def h(n):
        a2 = 2*a
        T = [], [], [n+a2], [n+1, a2], [-n, n+a2], [a+0.5], 0.5*(1-z)
        return [T]
    return ctx.hypercomb(h, [n], **kwargs)

@defun_wrapped
def jacobi(ctx, n, a, b, x, **kwargs):
    if not ctx.isnpint(a):
        def h(n):
            return (([], [], [a+n+1], [n+1, a+1], [-n, a+b+n+1], [a+1], (1-x)*0.5),)
        return ctx.hypercomb(h, [n], **kwargs)
    if not ctx.isint(b):
        def h(n, a):
            return (([], [], [-b], [n+1, -b-n], [-n, a+b+n+1], [b+1], (x+1)*0.5),)
        return ctx.hypercomb(h, [n, a], **kwargs)
    # XXX: determine appropriate limit
    return ctx.binomial(n+a,n) * ctx.hyp2f1(-n,1+n+a+b,a+1,(1-x)/2, **kwargs)

@defun_wrapped
def laguerre(ctx, n, a, z, **kwargs):
    # XXX: limits, poles
    #if ctx.isnpint(n):
    #    return 0*(a+z)
    def h(a):
        return (([], [], [a+n+1], [a+1, n+1], [-n], [a+1], z),)
    return ctx.hypercomb(h, [a], **kwargs)

@defun_wrapped
def legendre(ctx, n, x, **kwargs):
    if ctx.isint(n):
        n = int(n)
        # Accuracy near zeros
        if (n + (n < 0)) & 1:
            if not x:
                return x
            mag = ctx.mag(x)
            if mag < -2*ctx.prec-10:
                return x
            if mag < -5:
                ctx.prec += -mag
    return ctx.hyp2f1(-n,n+1,1,(1-x)/2, **kwargs)

@defun
def legenp(ctx, n, m, z, type=2, **kwargs):
    # Legendre function, 1st kind
    n = ctx.convert(n)
    m = ctx.convert(m)
    # Faster
    if not m:
        return ctx.legendre(n, z, **kwargs)
    # TODO: correct evaluation at singularities
    if type == 2:
        def h(n,m):
            g = m*0.5
            T = [1+z, 1-z], [g, -g], [], [1-m], [-n, n+1], [1-m], 0.5*(1-z)
            return (T,)
        return ctx.hypercomb(h, [n,m], **kwargs)
    if type == 3:
        def h(n,m):
            g = m*0.5
            T = [z+1, z-1], [g, -g], [], [1-m], [-n, n+1], [1-m], 0.5*(1-z)
            return (T,)
        return ctx.hypercomb(h, [n,m], **kwargs)
    raise ValueError("requires type=2 or type=3")

@defun
def legenq(ctx, n, m, z, type=2, **kwargs):
    # Legendre function, 2nd kind
    n = ctx.convert(n)
    m = ctx.convert(m)
    z = ctx.convert(z)
    if z in (1, -1):
        #if ctx.isint(m):
        #    return ctx.nan
        #return ctx.inf  # unsigned
        return ctx.nan
    if type == 2:
        def h(n, m):
            cos, sin = ctx.cospi_sinpi(m)
            s = 2 * sin / ctx.pi
            c = cos
            a = 1+z
            b = 1-z
            u = m/2
            w = (1-z)/2
            T1 = [s, c, a, b], [-1, 1, u, -u], [], [1-m], \
                [-n, n+1], [1-m], w
            T2 = [-s, a, b], [-1, -u, u], [n+m+1], [n-m+1, m+1], \
                [-n, n+1], [m+1], w
            return T1, T2
        return ctx.hypercomb(h, [n, m], **kwargs)
    if type == 3:
        # The following is faster when there only is a single series
        # Note: not valid for -1 < z < 0 (?)
        if abs(z) > 1:
            def h(n, m):
                T1 = [ctx.expjpi(m), 2, ctx.pi, z, z-1, z+1], \
                     [1, -n-1, 0.5, -n-m-1, 0.5*m, 0.5*m], \
                     [n+m+1], [n+1.5], \
                     [0.5*(2+n+m), 0.5*(1+n+m)], [n+1.5], z**(-2)
                return [T1]
            return ctx.hypercomb(h, [n, m], **kwargs)
        else:
            # not valid for 1 < z < inf ?
            def h(n, m):
                s = 2 * ctx.sinpi(m) / ctx.pi
                c = ctx.expjpi(m)
                a = 1+z
                b = z-1
                u = m/2
                w = (1-z)/2
                T1 = [s, c, a, b], [-1, 1, u, -u], [], [1-m], \
                    [-n, n+1], [1-m], w
                T2 = [-s, c, a, b], [-1, 1, -u, u], [n+m+1], [n-m+1, m+1], \
                    [-n, n+1], [m+1], w
                return T1, T2
            return ctx.hypercomb(h, [n, m], **kwargs)
    raise ValueError("requires type=2 or type=3")

@defun_wrapped
def chebyt(ctx, n, x, **kwargs):
    if (not x) and ctx.isint(n) and int(ctx._re(n)) % 2 == 1:
        return x * 0
    return ctx.hyp2f1(-n,n,(1,2),(1-x)/2, **kwargs)

@defun_wrapped
def chebyu(ctx, n, x, **kwargs):
    if (not x) and ctx.isint(n) and int(ctx._re(n)) % 2 == 1:
        return x * 0
    return (n+1) * ctx.hyp2f1(-n, n+2, (3,2), (1-x)/2, **kwargs)

@defun
def spherharm(ctx, l, m, theta, phi, **kwargs):
    l = ctx.convert(l)
    m = ctx.convert(m)
    theta = ctx.convert(theta)
    phi = ctx.convert(phi)
    l_isint = ctx.isint(l)
    l_natural = l_isint and l >= 0
    m_isint = ctx.isint(m)
    if l_isint and l < 0 and m_isint:
        return ctx.spherharm(-(l+1), m, theta, phi, **kwargs)
    if theta == 0 and m_isint and m < 0:
        return ctx.zero * 1j
    if l_natural and m_isint:
        if abs(m) > l:
            return ctx.zero * 1j
        # http://functions.wolfram.com/Polynomials/
        #     SphericalHarmonicY/26/01/02/0004/
        def h(l,m):
            absm = abs(m)
            C = [-1, ctx.expj(m*phi),
                 (2*l+1)*ctx.fac(l+absm)/ctx.pi/ctx.fac(l-absm),
                 ctx.sin(theta)**2,
                 ctx.fac(absm), 2]
            P = [0.5*m*(ctx.sign(m)+1), 1, 0.5, 0.5*absm, -1, -absm-1]
            return ((C, P, [], [], [absm-l, l+absm+1], [absm+1],
                ctx.sin(0.5*theta)**2),)
    else:
        # http://functions.wolfram.com/HypergeometricFunctions/
        #     SphericalHarmonicYGeneral/26/01/02/0001/
        def h(l,m):
            if ctx.isnpint(l-m+1) or ctx.isnpint(l+m+1) or ctx.isnpint(1-m):
                return (([0], [-1], [], [], [], [], 0),)
            cos, sin = ctx.cos_sin(0.5*theta)
            C = [0.5*ctx.expj(m*phi), (2*l+1)/ctx.pi,
                 ctx.gamma(l-m+1), ctx.gamma(l+m+1),
                 cos**2, sin**2]
            P = [1, 0.5, 0.5, -0.5, 0.5*m, -0.5*m]
            return ((C, P, [], [1-m], [-l,l+1], [1-m], sin**2),)
    return ctx.hypercomb(h, [l,m], **kwargs)
