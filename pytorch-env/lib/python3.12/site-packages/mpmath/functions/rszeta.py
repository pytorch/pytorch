"""
---------------------------------------------------------------------
.. sectionauthor:: Juan Arias de Reyna <arias@us.es>

This module implements zeta-related functions using the Riemann-Siegel
expansion: zeta_offline(s,k=0)

* coef(J, eps): Need in the computation of Rzeta(s,k)

* Rzeta_simul(s, der=0) computes Rzeta^(k)(s) and Rzeta^(k)(1-s) simultaneously
  for  0 <= k <= der. Used by zeta_offline and z_offline

* Rzeta_set(s, derivatives) computes Rzeta^(k)(s) for given derivatives, used by
  z_half(t,k) and zeta_half

* z_offline(w,k): Z(w) and its derivatives of order k <= 4
* z_half(t,k): Z(t) (Riemann Siegel function) and its derivatives of order k <= 4
* zeta_offline(s): zeta(s) and its derivatives of order k<= 4
* zeta_half(1/2+it,k):  zeta(s)  and its derivatives of order k<= 4

* rs_zeta(s,k=0) Computes zeta^(k)(s)   Unifies zeta_half and zeta_offline
* rs_z(w,k=0)    Computes Z^(k)(w)      Unifies z_offline and z_half
----------------------------------------------------------------------

This program uses Riemann-Siegel expansion even to compute
zeta(s) on points s = sigma + i t  with sigma arbitrary not
necessarily equal to 1/2.

It is founded on a new deduction of the formula, with rigorous
and sharp bounds for the  terms and rest of this expansion.

More information on the papers:

 J. Arias de Reyna, High Precision Computation of Riemann's
 Zeta Function by the Riemann-Siegel Formula I, II

 We refer to them as I, II.

 In them we shall find detailed explanation of all the
 procedure.

The program uses Riemann-Siegel expansion.
This  is useful when t is big, ( say  t > 10000 ).
The precision is limited, roughly it can compute zeta(sigma+it)
with an error less than exp(-c t) for some constant c depending
on sigma.  The program gives an error when the Riemann-Siegel
formula can not compute to the wanted precision.

"""

import math

class RSCache(object):
    def __init__(ctx):
        ctx._rs_cache = [0, 10, {}, {}]

from .functions import defun

#-------------------------------------------------------------------------------#
#                                                                               #
#                       coef(ctx, J, eps, _cache=[0, 10, {} ] )                 #
#                                                                               #
#-------------------------------------------------------------------------------#

#  This function computes the coefficients c[n] defined on (I, equation (47))
#  but see also  (II, section 3.14).
#
#  Since these coefficients are very difficult to compute we save the values
#  in a cache. So if we compute several values of the functions Rzeta(s) for
#  near values of s, we do not recompute these coefficients.
#
#  c[n] are the Taylor coefficients of the function:
#
#  F(z):= (exp(pi*j*(z*z/2+3/8))-j* sqrt(2) cos(pi*z/2))/(2*cos(pi *z))
#
#

def _coef(ctx, J, eps):
    r"""
    Computes the coefficients  `c_n`  for `0\le n\le 2J` with error less than eps

    **Definition**

    The coefficients c_n are defined by

    .. math ::

        \begin{equation}
        F(z)=\frac{e^{\pi i
        \bigl(\frac{z^2}{2}+\frac38\bigr)}-i\sqrt{2}\cos\frac{\pi}{2}z}{2\cos\pi
        z}=\sum_{n=0}^\infty c_{2n} z^{2n}
        \end{equation}

    they are computed applying the relation

    .. math ::

        \begin{multline}
        c_{2n}=-\frac{i}{\sqrt{2}}\Bigl(\frac{\pi}{2}\Bigr)^{2n}
        \sum_{k=0}^n\frac{(-1)^k}{(2k)!}
        2^{2n-2k}\frac{(-1)^{n-k}E_{2n-2k}}{(2n-2k)!}+\\
        +e^{3\pi i/8}\sum_{j=0}^n(-1)^j\frac{
        E_{2j}}{(2j)!}\frac{i^{n-j}\pi^{n+j}}{(n-j)!2^{n-j+1}}.
        \end{multline}
    """

    newJ = J+2        # compute more coefficients that are needed
    neweps6 = eps/2.  # compute with a slight more precision that are needed

    #  PREPARATION FOR THE COMPUTATION OF V(N) AND W(N)
    #    See II Section 3.16
    #
    #  Computing the exponent wpvw of the error II equation (81)
    wpvw = max(ctx.mag(10*(newJ+3)), 4*newJ+5-ctx.mag(neweps6))

    #  Preparation of Euler numbers (we need until the 2*RS_NEWJ)
    E = ctx._eulernum(2*newJ)

    #  Now we have in the cache all the needed Euler numbers.
    #
    #  Computing the powers of pi
    #
    # We need to compute the powers pi**n for 1<= n <= 2*J
    # with relative error less than 2**(-wpvw)
    # it is easy to show that this is obtained
    # taking wppi as the least d with
    # 2**d>40*J and 2**d> 4.24 *newJ + 2**wpvw
    # In II Section 3.9 we need also that
    #  wppi > wptcoef[0], and that the powers
    # here computed  0<= k <= 2*newJ are more
    # than those needed there that are 2*L-2.
    # so we need  J >= L this will be checked
    # before computing tcoef[]
    wppi = max(ctx.mag(40*newJ), ctx.mag(newJ)+3 +wpvw)
    ctx.prec = wppi
    pipower = {}
    pipower[0] = ctx.one
    pipower[1] = ctx.pi
    for n in range(2,2*newJ+1):
        pipower[n] = pipower[n-1]*ctx.pi

    # COMPUTING THE COEFFICIENTS v(n) AND w(n)
    #  see II equation (61) and equations (81) and (82)
    ctx.prec = wpvw+2
    v={}
    w={}
    for n in range(0,newJ+1):
        va = (-1)**n * ctx._eulernum(2*n)
        va = ctx.mpf(va)/ctx.fac(2*n)
        v[n]=va*pipower[2*n]
    for n in range(0,2*newJ+1):
        wa = ctx.one/ctx.fac(n)
        wa=wa/(2**n)
        w[n]=wa*pipower[n]

    # COMPUTATION OF THE CONVOLUTIONS RS_P1 AND RS_P2
    #  See II Section 3.16
    ctx.prec = 15
    wpp1a = 9 - ctx.mag(neweps6)
    P1 = {}
    for n in range(0,newJ+1):
        ctx.prec = 15
        wpp1 = max(ctx.mag(10*(n+4)),4*n+wpp1a)
        ctx.prec = wpp1
        sump = 0
        for k in range(0,n+1):
            sump += ((-1)**k) * v[k]*w[2*n-2*k]
        P1[n]=((-1)**(n+1))*ctx.j*sump
    P2={}
    for n in range(0,newJ+1):
        ctx.prec = 15
        wpp2 = max(ctx.mag(10*(n+4)),4*n+wpp1a)
        ctx.prec = wpp2
        sump = 0
        for k in range(0,n+1):
            sump += (ctx.j**(n-k)) * v[k]*w[n-k]
        P2[n]=sump
    # COMPUTING THE COEFFICIENTS c[2n]
    # See II Section 3.14
    ctx.prec = 15
    wpc0 = 5 - ctx.mag(neweps6)
    wpc = max(6,4*newJ+wpc0)
    ctx.prec = wpc
    mu = ctx.sqrt(ctx.mpf('2'))/2
    nu = ctx.expjpi(3./8)/2
    c={}
    for n in range(0,newJ):
        ctx.prec = 15
        wpc = max(6,4*n+wpc0)
        ctx.prec = wpc
        c[2*n] = mu*P1[n]+nu*P2[n]
    for n in range(1,2*newJ,2):
        c[n] = 0
    return [newJ, neweps6, c, pipower]

def coef(ctx, J, eps):
    _cache = ctx._rs_cache
    if J <= _cache[0] and eps >= _cache[1]:
        return _cache[2], _cache[3]
    orig = ctx._mp.prec
    try:
        data = _coef(ctx._mp, J, eps)
    finally:
        ctx._mp.prec = orig
    if ctx is not ctx._mp:
        data[2] = dict((k,ctx.convert(v)) for (k,v) in data[2].items())
        data[3] = dict((k,ctx.convert(v)) for (k,v) in data[3].items())
    ctx._rs_cache[:] = data
    return ctx._rs_cache[2], ctx._rs_cache[3]

#-------------------------------------------------------------------------------#
#                                                                               #
#                          Rzeta_simul(s,k=0)                                   #
#                                                                               #
#-------------------------------------------------------------------------------#
#  This function return a list with the values:
#  Rzeta(sigma+it), conj(Rzeta(1-sigma+it)),Rzeta'(sigma+it), conj(Rzeta'(1-sigma+it)),
#  .... , Rzeta^{(k)}(sigma+it), conj(Rzeta^{(k)}(1-sigma+it))
#
#  Useful to compute  the function zeta(s) and Z(w)  or its derivatives.
#

def aux_M_Fp(ctx, xA, xeps4, a, xB1, xL):
    # COMPUTING M  NUMBER OF DERIVATIVES Fp[m] TO COMPUTE
    #  See II Section 3.11  equations (47) and (48)
    aux1 = 126.0657606*xA/xeps4   # 126.06.. = 316/sqrt(2*pi)
    aux1 = ctx.ln(aux1)
    aux2 = (2*ctx.ln(ctx.pi)+ctx.ln(xB1)+ctx.ln(a))/3 -ctx.ln(2*ctx.pi)/2
    m = 3*xL-3
    aux3= (ctx.loggamma(m+1)-ctx.loggamma(m/3.0+2))/2 -ctx.loggamma((m+1)/2.)
    while((aux1 < m*aux2+ aux3)and (m>1)):
        m = m - 1
        aux3 = (ctx.loggamma(m+1)-ctx.loggamma(m/3.0+2))/2 -ctx.loggamma((m+1)/2.)
    xM = m
    return xM

def aux_J_needed(ctx, xA, xeps4, a, xB1, xM):
    #  DETERMINATION OF  J  THE NUMBER OF TERMS NEEDED
    #            IN THE TAYLOR SERIES OF F.
    #  See II Section 3.11 equation (49))
    #  Only determine one
    h1 = xeps4/(632*xA)
    h2 = xB1*a * 126.31337419529260248  # = pi^2*e^2*sqrt(3)
    h2 = h1 * ctx.power((h2/xM**2),(xM-1)/3) / xM
    h3 = min(h1,h2)
    return h3

def Rzeta_simul(ctx, s, der=0):
    # First we take the value of ctx.prec
    wpinitial = ctx.prec

    # INITIALIZATION
    # Take the real and imaginary part of s
    t = ctx._im(s)
    xsigma = ctx._re(s)
    ysigma = 1 - xsigma

    # Now compute several parameter that appear on the program
    ctx.prec = 15
    a = ctx.sqrt(t/(2*ctx.pi))
    xasigma = a ** xsigma
    yasigma = a ** ysigma

    # We need a simple bound A1 < asigma  (see II Section 3.1 and 3.3)
    xA1=ctx.power(2, ctx.mag(xasigma)-1)
    yA1=ctx.power(2, ctx.mag(yasigma)-1)

    # We compute various epsilon's  (see II end of Section 3.1)
    eps = ctx.power(2, -wpinitial)
    eps1 = eps/6.
    xeps2 = eps * xA1/3.
    yeps2 = eps * yA1/3.

    #  COMPUTING SOME COEFFICIENTS THAT DEPENDS
    #                ON  sigma
    #  constant b and c  (see I  Theorem 2 formula (26) )
    #  coefficients A and B1  (see I Section 6.1 equation (50))
    #
    # here we not need high precision
    ctx.prec = 15
    if xsigma > 0:
        xb = 2.
        xc = math.pow(9,xsigma)/4.44288
        # 4.44288 =(math.sqrt(2)*math.pi)
        xA = math.pow(9,xsigma)
        xB1 = 1
    else:
        xb = 2.25158  #  math.sqrt( (3-2* math.log(2))*math.pi )
        xc = math.pow(2,-xsigma)/4.44288
        xA = math.pow(2,-xsigma)
        xB1 = 1.10789   #  = 2*sqrt(1-log(2))

    if(ysigma > 0):
        yb = 2.
        yc = math.pow(9,ysigma)/4.44288
        # 4.44288 =(math.sqrt(2)*math.pi)
        yA = math.pow(9,ysigma)
        yB1 = 1
    else:
        yb = 2.25158  #  math.sqrt( (3-2* math.log(2))*math.pi )
        yc = math.pow(2,-ysigma)/4.44288
        yA = math.pow(2,-ysigma)
        yB1 = 1.10789   #  = 2*sqrt(1-log(2))

    #  COMPUTING L THE NUMBER OF TERMS NEEDED IN THE RIEMANN-SIEGEL
    #                         CORRECTION
    #  See II Section 3.2
    ctx.prec = 15
    xL = 1
    while 3*xc*ctx.gamma(xL*0.5) * ctx.power(xb*a,-xL) >= xeps2:
        xL = xL+1
    xL = max(2,xL)
    yL = 1
    while 3*yc*ctx.gamma(yL*0.5) * ctx.power(yb*a,-yL) >= yeps2:
        yL = yL+1
    yL = max(2,yL)

    #  The number L has to satify some conditions.
    #  If not RS can not compute Rzeta(s) with the prescribed precision
    #  (see II, Section 3.2 condition (20)  ) and
    #  (II, Section 3.3 condition (22) ). Also we have added
    #  an additional technical  condition in Section 3.17 Proposition 17
    if ((3*xL >= 2*a*a/25.) or (3*xL+2+xsigma<0) or (abs(xsigma) > a/2.) or \
        (3*yL >= 2*a*a/25.) or (3*yL+2+ysigma<0) or (abs(ysigma) > a/2.)):
        ctx.prec = wpinitial
        raise NotImplementedError("Riemann-Siegel can not compute with such precision")

    #  We take the maximum of the two values
    L = max(xL, yL)

    #  INITIALIZATION (CONTINUATION)
    #
    # eps3 is the constant defined on (II, Section 3.5 equation (27) )
    # each term of the RS correction must be computed with error <= eps3
    xeps3 =  xeps2/(4*xL)
    yeps3 =  yeps2/(4*yL)

    # eps4 is defined on (II Section 3.6  equation (30) )
    # each component of the formula (II Section 3.6 equation (29) )
    # must be computed with error <= eps4
    xeps4 = xeps3/(3*xL)
    yeps4 = yeps3/(3*yL)

    # COMPUTING M NUMBER OF DERIVATIVES Fp[m] TO COMPUTE
    xM = aux_M_Fp(ctx, xA, xeps4, a, xB1, xL)
    yM = aux_M_Fp(ctx, yA, yeps4, a, yB1, yL)
    M = max(xM, yM)

    # COMPUTING NUMBER OF TERMS J NEEDED
    h3 = aux_J_needed(ctx, xA, xeps4, a, xB1, xM)
    h4 = aux_J_needed(ctx, yA, yeps4, a, yB1, yM)
    h3 = min(h3,h4)
    J = 12
    jvalue = (2*ctx.pi)**J / ctx.gamma(J+1)
    while jvalue > h3:
        J = J+1
        jvalue = (2*ctx.pi)*jvalue/J

    # COMPUTING eps5[m] for 1 <= m <= 21
    #  See II Section 10 equation (43)
    #  We choose the minimum of the two possibilities
    eps5={}
    xforeps5 = math.pi*math.pi*xB1*a
    yforeps5 = math.pi*math.pi*yB1*a
    for m in range(0,22):
        xaux1 = math.pow(xforeps5, m/3)/(316.*xA)
        yaux1 = math.pow(yforeps5, m/3)/(316.*yA)
        aux1 = min(xaux1, yaux1)
        aux2 = ctx.gamma(m+1)/ctx.gamma(m/3.0+0.5)
        aux2 = math.sqrt(aux2)
        eps5[m] = (aux1*aux2*min(xeps4,yeps4))

    # COMPUTING wpfp
    #  See II Section 3.13 equation (59)
    twenty = min(3*L-3, 21)+1
    aux = 6812*J
    wpfp = ctx.mag(44*J)
    for m in range(0,twenty):
        wpfp = max(wpfp, ctx.mag(aux*ctx.gamma(m+1)/eps5[m]))

    # COMPUTING N AND p
    #  See II Section
    ctx.prec = wpfp + ctx.mag(t)+20
    a = ctx.sqrt(t/(2*ctx.pi))
    N = ctx.floor(a)
    p = 1-2*(a-N)

    # now we get a rounded version of p
    # to the precision wpfp
    # this possibly is not necessary
    num=ctx.floor(p*(ctx.mpf('2')**wpfp))
    difference = p * (ctx.mpf('2')**wpfp)-num
    if (difference < 0.5):
        num = num
    else:
        num = num+1
    p = ctx.convert(num * (ctx.mpf('2')**(-wpfp)))

    # COMPUTING THE COEFFICIENTS c[n] = cc[n]
    # We shall use the notation cc[n], since there is
    # a constant that is called c
    # See II Section 3.14
    # We compute the coefficients and also save then in a
    # cache.  The bulk of the computation is passed to
    # the function  coef()
    #
    #  eps6 is defined in II Section 3.13  equation (58)
    eps6 = ctx.power(ctx.convert(2*ctx.pi), J)/(ctx.gamma(J+1)*3*J)

    #  Now we compute the coefficients
    cc = {}
    cont = {}
    cont, pipowers = coef(ctx, J, eps6)
    cc=cont.copy()   # we need a copy since we have to change his values.
    Fp={}            # this is the adequate locus of this
    for n in range(M, 3*L-2):
        Fp[n] = 0
    Fp={}
    ctx.prec = wpfp
    for m in range(0,M+1):
        sumP = 0
        for k in range(2*J-m-1,-1,-1):
            sumP = (sumP * p)+ cc[k]
        Fp[m] = sumP
        # preparation of the new coefficients
        for k in range(0,2*J-m-1):
            cc[k] = (k+1)* cc[k+1]

    # COMPUTING THE NUMBERS  xd[u,n,k], yd[u,n,k]
    #  See II Section 3.17
    #
    #  First we compute the working precisions xwpd[k]
    #   Se II equation (92)
    xwpd={}
    d1 = max(6,ctx.mag(40*L*L))
    xd2 = 13+ctx.mag((1+abs(xsigma))*xA)-ctx.mag(xeps4)-1
    xconst = ctx.ln(8/(ctx.pi*ctx.pi*a*a*xB1*xB1)) /2
    for n in range(0,L):
        xd3 = ctx.mag(ctx.sqrt(ctx.gamma(n-0.5)))-ctx.floor(n*xconst)+xd2
        xwpd[n]=max(xd3,d1)

    # procedure of II Section 3.17
    ctx.prec = xwpd[1]+10
    xpsigma = 1-(2*xsigma)
    xd = {}
    xd[0,0,-2]=0; xd[0,0,-1]=0; xd[0,0,0]=1; xd[0,0,1]=0
    xd[0,-1,-2]=0; xd[0,-1,-1]=0; xd[0,-1,0]=1; xd[0,-1,1]=0
    for n in range(1,L):
        ctx.prec = xwpd[n]+10
        for k in range(0,3*n//2+1):
            m = 3*n-2*k
            if(m!=0):
                m1 = ctx.one/m
                c1= m1/4
                c2=(xpsigma*m1)/2
                c3=-(m+1)
                xd[0,n,k]=c3*xd[0,n-1,k-2]+c1*xd[0,n-1,k]+c2*xd[0,n-1,k-1]
            else:
                xd[0,n,k]=0
                for r in range(0,k):
                    add=xd[0,n,r]*(ctx.mpf('1.0')*ctx.fac(2*k-2*r)/ctx.fac(k-r))
                    xd[0,n,k] -= ((-1)**(k-r))*add
        xd[0,n,-2]=0; xd[0,n,-1]=0; xd[0,n,3*n//2+1]=0
    for mu in range(-2,der+1):
        for n in range(-2,L):
            for k in range(-3,max(1,3*n//2+2)):
                if( (mu<0)or (n<0) or(k<0)or (k>3*n//2)):
                    xd[mu,n,k] = 0
    for mu in range(1,der+1):
        for n in range(0,L):
            ctx.prec = xwpd[n]+10
            for k in range(0,3*n//2+1):
                aux=(2*mu-2)*xd[mu-2,n-2,k-3]+2*(xsigma+n-2)*xd[mu-1,n-2,k-3]
                xd[mu,n,k] = aux - xd[mu-1,n-1,k-1]

    #  Now we compute the working precisions ywpd[k]
    #   Se II equation (92)
    ywpd={}
    d1 = max(6,ctx.mag(40*L*L))
    yd2 = 13+ctx.mag((1+abs(ysigma))*yA)-ctx.mag(yeps4)-1
    yconst = ctx.ln(8/(ctx.pi*ctx.pi*a*a*yB1*yB1)) /2
    for n in range(0,L):
        yd3 = ctx.mag(ctx.sqrt(ctx.gamma(n-0.5)))-ctx.floor(n*yconst)+yd2
        ywpd[n]=max(yd3,d1)

    # procedure of II Section 3.17
    ctx.prec = ywpd[1]+10
    ypsigma = 1-(2*ysigma)
    yd = {}
    yd[0,0,-2]=0; yd[0,0,-1]=0; yd[0,0,0]=1; yd[0,0,1]=0
    yd[0,-1,-2]=0; yd[0,-1,-1]=0; yd[0,-1,0]=1; yd[0,-1,1]=0
    for n in range(1,L):
        ctx.prec = ywpd[n]+10
        for k in range(0,3*n//2+1):
            m = 3*n-2*k
            if(m!=0):
                m1 = ctx.one/m
                c1= m1/4
                c2=(ypsigma*m1)/2
                c3=-(m+1)
                yd[0,n,k]=c3*yd[0,n-1,k-2]+c1*yd[0,n-1,k]+c2*yd[0,n-1,k-1]
            else:
                yd[0,n,k]=0
                for r in range(0,k):
                    add=yd[0,n,r]*(ctx.mpf('1.0')*ctx.fac(2*k-2*r)/ctx.fac(k-r))
                    yd[0,n,k] -= ((-1)**(k-r))*add
        yd[0,n,-2]=0; yd[0,n,-1]=0; yd[0,n,3*n//2+1]=0

    for mu in range(-2,der+1):
        for n in range(-2,L):
            for k in range(-3,max(1,3*n//2+2)):
                if( (mu<0)or (n<0) or(k<0)or (k>3*n//2)):
                    yd[mu,n,k] = 0
    for mu in range(1,der+1):
        for n in range(0,L):
            ctx.prec = ywpd[n]+10
            for k in range(0,3*n//2+1):
                aux=(2*mu-2)*yd[mu-2,n-2,k-3]+2*(ysigma+n-2)*yd[mu-1,n-2,k-3]
                yd[mu,n,k] = aux - yd[mu-1,n-1,k-1]

    # COMPUTING THE COEFFICIENTS xtcoef[k,l]
    #  See II Section 3.9
    #
    # computing the needed wp
    xwptcoef={}
    xwpterm={}
    ctx.prec = 15
    c1 = ctx.mag(40*(L+2))
    xc2 = ctx.mag(68*(L+2)*xA)
    xc4 = ctx.mag(xB1*a*math.sqrt(ctx.pi))-1
    for k in range(0,L):
        xc3 = xc2 - k*xc4+ctx.mag(ctx.fac(k+0.5))/2.
        xwptcoef[k] = (max(c1,xc3-ctx.mag(xeps4)+1)+1 +20)*1.5
        xwpterm[k] = (max(c1,ctx.mag(L+2)+xc3-ctx.mag(xeps3)+1)+1 +20)
    ywptcoef={}
    ywpterm={}
    ctx.prec = 15
    c1 = ctx.mag(40*(L+2))
    yc2 = ctx.mag(68*(L+2)*yA)
    yc4 = ctx.mag(yB1*a*math.sqrt(ctx.pi))-1
    for k in range(0,L):
        yc3 = yc2 - k*yc4+ctx.mag(ctx.fac(k+0.5))/2.
        ywptcoef[k] = ((max(c1,yc3-ctx.mag(yeps4)+1))+10)*1.5
        ywpterm[k] = (max(c1,ctx.mag(L+2)+yc3-ctx.mag(yeps3)+1)+1)+10

    # check of power of pi
    # computing the fortcoef[mu,k,ell]
    xfortcoef={}
    for mu in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                xfortcoef[mu,k,ell]=0
    for mu in range(0,der+1):
        for k in range(0,L):
            ctx.prec = xwptcoef[k]
            for ell in range(0,3*k//2+1):
                xfortcoef[mu,k,ell]=xd[mu,k,ell]*Fp[3*k-2*ell]/pipowers[2*k-ell]
                xfortcoef[mu,k,ell]=xfortcoef[mu,k,ell]/((2*ctx.j)**ell)

    def trunc_a(t):
        wp = ctx.prec
        ctx.prec = wp + 2
        aa = ctx.sqrt(t/(2*ctx.pi))
        ctx.prec = wp
        return aa

    # computing the tcoef[k,ell]
    xtcoef={}
    for mu in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                xtcoef[mu,k,ell]=0
    ctx.prec = max(xwptcoef[0],ywptcoef[0])+3
    aa= trunc_a(t)
    la = -ctx.ln(aa)

    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = xwptcoef[k]
            for ell in range(0,3*k//2+1):
                xtcoef[chi,k,ell] =0
                for mu in range(0, chi+1):
                    tcoefter=ctx.binomial(chi,mu)*ctx.power(la,mu)*xfortcoef[chi-mu,k,ell]
                    xtcoef[chi,k,ell] += tcoefter

    # COMPUTING THE COEFFICIENTS ytcoef[k,l]
    #  See II Section 3.9
    #
    # computing the needed wp
    # check of power of pi
    # computing the fortcoef[mu,k,ell]
    yfortcoef={}
    for mu in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                yfortcoef[mu,k,ell]=0
    for mu in range(0,der+1):
        for k in range(0,L):
            ctx.prec = ywptcoef[k]
            for ell in range(0,3*k//2+1):
                yfortcoef[mu,k,ell]=yd[mu,k,ell]*Fp[3*k-2*ell]/pipowers[2*k-ell]
                yfortcoef[mu,k,ell]=yfortcoef[mu,k,ell]/((2*ctx.j)**ell)
    # computing the tcoef[k,ell]
    ytcoef={}
    for chi in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                ytcoef[chi,k,ell]=0
    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = ywptcoef[k]
            for ell in range(0,3*k//2+1):
                ytcoef[chi,k,ell] =0
                for mu in range(0, chi+1):
                    tcoefter=ctx.binomial(chi,mu)*ctx.power(la,mu)*yfortcoef[chi-mu,k,ell]
                    ytcoef[chi,k,ell] += tcoefter

    # COMPUTING tv[k,ell]
    # See II Section 3.8
    #
    #  a has a good value
    ctx.prec = max(xwptcoef[0], ywptcoef[0])+2
    av = {}
    av[0] = 1
    av[1] = av[0]/a

    ctx.prec = max(xwptcoef[0],ywptcoef[0])
    for k in range(2,L):
        av[k] = av[k-1] * av[1]

    # Computing the quotients
    xtv = {}
    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = xwptcoef[k]
            for ell in range(0,3*k//2+1):
                xtv[chi,k,ell] = xtcoef[chi,k,ell]* av[k]
    # Computing the quotients
    ytv = {}
    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = ywptcoef[k]
            for ell in range(0,3*k//2+1):
                ytv[chi,k,ell] = ytcoef[chi,k,ell]* av[k]

    # COMPUTING THE TERMS xterm[k]
    # See II Section 3.6
    xterm = {}
    for chi in range(0,der+1):
        for n in range(0,L):
            ctx.prec = xwpterm[n]
            te = 0
            for k in range(0, 3*n//2+1):
                te += xtv[chi,n,k]
            xterm[chi,n] = te

    # COMPUTING THE TERMS yterm[k]
    # See II Section 3.6
    yterm = {}
    for chi in range(0,der+1):
        for n in range(0,L):
            ctx.prec = ywpterm[n]
            te = 0
            for k in range(0, 3*n//2+1):
                te += ytv[chi,n,k]
            yterm[chi,n] = te

    # COMPUTING  rssum
    # See II Section 3.5
    xrssum={}
    ctx.prec=15
    xrsbound = math.sqrt(ctx.pi) * xc /(xb*a)
    ctx.prec=15
    xwprssum = ctx.mag(4.4*((L+3)**2)*xrsbound / xeps2)
    xwprssum = max(xwprssum, ctx.mag(10*(L+1)))
    ctx.prec = xwprssum
    for chi in range(0,der+1):
        xrssum[chi] = 0
        for k in range(1,L+1):
            xrssum[chi] += xterm[chi,L-k]
    yrssum={}
    ctx.prec=15
    yrsbound = math.sqrt(ctx.pi) * yc /(yb*a)
    ctx.prec=15
    ywprssum = ctx.mag(4.4*((L+3)**2)*yrsbound / yeps2)
    ywprssum = max(ywprssum, ctx.mag(10*(L+1)))
    ctx.prec = ywprssum
    for chi in range(0,der+1):
        yrssum[chi] = 0
        for k in range(1,L+1):
            yrssum[chi] += yterm[chi,L-k]

    # COMPUTING S3
    # See II Section 3.19
    ctx.prec = 15
    A2 = 2**(max(ctx.mag(abs(xrssum[0])), ctx.mag(abs(yrssum[0]))))
    eps8 = eps/(3*A2)
    T = t *ctx.ln(t/(2*ctx.pi))
    xwps3 = 5 +  ctx.mag((1+(2/eps8)*ctx.power(a,-xsigma))*T)
    ywps3 = 5 +  ctx.mag((1+(2/eps8)*ctx.power(a,-ysigma))*T)

    ctx.prec = max(xwps3, ywps3)

    tpi = t/(2*ctx.pi)
    arg = (t/2)*ctx.ln(tpi)-(t/2)-ctx.pi/8
    U = ctx.expj(-arg)
    a = trunc_a(t)
    xasigma = ctx.power(a, -xsigma)
    yasigma = ctx.power(a, -ysigma)
    xS3 = ((-1)**(N-1)) * xasigma * U
    yS3 = ((-1)**(N-1)) * yasigma * U

    # COMPUTING S1 the zetasum
    # See II Section 3.18
    ctx.prec = 15
    xwpsum =  4+ ctx.mag((N+ctx.power(N,1-xsigma))*ctx.ln(N) /eps1)
    ywpsum =  4+ ctx.mag((N+ctx.power(N,1-ysigma))*ctx.ln(N) /eps1)
    wpsum = max(xwpsum, ywpsum)

    ctx.prec = wpsum +10
    '''
    # This can be improved
    xS1={}
    yS1={}
    for chi in range(0,der+1):
        xS1[chi] = 0
        yS1[chi] = 0
    for n in range(1,int(N)+1):
        ln = ctx.ln(n)
        xexpn = ctx.exp(-ln*(xsigma+ctx.j*t))
        yexpn = ctx.conj(1/(n*xexpn))
        for chi in range(0,der+1):
            pown = ctx.power(-ln, chi)
            xterm = pown*xexpn
            yterm = pown*yexpn
            xS1[chi] += xterm
            yS1[chi] += yterm
    '''
    xS1, yS1 = ctx._zetasum(s, 1, int(N)-1, range(0,der+1), True)

    # END OF COMPUTATION of xrz, yrz
    #  See II Section 3.1
    ctx.prec = 15
    xabsS1 = abs(xS1[der])
    xabsS2 = abs(xrssum[der] * xS3)
    xwpend = max(6, wpinitial+ctx.mag(6*(3*xabsS1+7*xabsS2) ) )

    ctx.prec = xwpend
    xrz={}
    for chi in range(0,der+1):
        xrz[chi] = xS1[chi]+xrssum[chi]*xS3

    ctx.prec = 15
    yabsS1 = abs(yS1[der])
    yabsS2 = abs(yrssum[der] * yS3)
    ywpend = max(6, wpinitial+ctx.mag(6*(3*yabsS1+7*yabsS2) ) )

    ctx.prec = ywpend
    yrz={}
    for chi in range(0,der+1):
        yrz[chi] = yS1[chi]+yrssum[chi]*yS3
        yrz[chi] = ctx.conj(yrz[chi])
    ctx.prec = wpinitial
    return xrz, yrz

def Rzeta_set(ctx, s, derivatives=[0]):
    r"""
    Computes several derivatives of the auxiliary function of Riemann `R(s)`.

    **Definition**

    The function is defined by

    .. math ::

        \begin{equation}
        {\mathop{\mathcal R }\nolimits}(s)=
        \int_{0\swarrow1}\frac{x^{-s} e^{\pi i x^2}}{e^{\pi i x}-
        e^{-\pi i x}}\,dx
        \end{equation}

    To this function we apply the Riemann-Siegel expansion.
    """
    der = max(derivatives)
    # First we take the value of ctx.prec
    # During the computation we will change ctx.prec, and finally we will
    # restaurate the initial value
    wpinitial = ctx.prec
    # Take the real and imaginary part of s
    t = ctx._im(s)
    sigma = ctx._re(s)
    # Now compute several parameter that appear on the program
    ctx.prec = 15
    a = ctx.sqrt(t/(2*ctx.pi))     #  Careful
    asigma = ctx.power(a, sigma)  #  Careful
    # We need a simple bound A1 < asigma  (see II Section 3.1 and 3.3)
    A1 = ctx.power(2, ctx.mag(asigma)-1)
    # We compute various epsilon's  (see II end of Section 3.1)
    eps = ctx.power(2, -wpinitial)
    eps1 = eps/6.
    eps2 = eps * A1/3.
    # COMPUTING SOME COEFFICIENTS THAT DEPENDS
    #               ON  sigma
    # constant b and c  (see I  Theorem 2 formula (26) )
    # coefficients A and B1  (see I Section 6.1 equation (50))
    # here we not need high precision
    ctx.prec = 15
    if sigma > 0:
        b = 2.
        c = math.pow(9,sigma)/4.44288
        # 4.44288 =(math.sqrt(2)*math.pi)
        A = math.pow(9,sigma)
        B1 = 1
    else:
        b = 2.25158  #  math.sqrt( (3-2* math.log(2))*math.pi )
        c = math.pow(2,-sigma)/4.44288
        A = math.pow(2,-sigma)
        B1 = 1.10789   #  = 2*sqrt(1-log(2))
    #  COMPUTING L THE NUMBER OF TERMS NEEDED IN THE RIEMANN-SIEGEL
    #                         CORRECTION
    #  See II Section 3.2
    ctx.prec = 15
    L = 1
    while 3*c*ctx.gamma(L*0.5) * ctx.power(b*a,-L) >= eps2:
        L = L+1
    L = max(2,L)
    #  The number L has to satify some conditions.
    #  If not RS can not compute Rzeta(s) with the prescribed precision
    #  (see II, Section 3.2 condition (20)  ) and
    #  (II, Section 3.3 condition (22) ). Also we have added
    #  an additional technical  condition in Section 3.17 Proposition 17
    if ((3*L >= 2*a*a/25.) or (3*L+2+sigma<0) or (abs(sigma)> a/2.)):
        #print 'Error Riemann-Siegel can not compute with such precision'
        ctx.prec = wpinitial
        raise NotImplementedError("Riemann-Siegel can not compute with such precision")

    #  INITIALIZATION (CONTINUATION)
    #
    # eps3 is the constant defined on (II, Section 3.5 equation (27) )
    # each term of the RS correction must be computed with error <= eps3
    eps3 =  eps2/(4*L)

    # eps4 is defined on (II Section 3.6  equation (30) )
    # each component of the formula (II Section 3.6 equation (29) )
    # must be computed with error <= eps4
    eps4 = eps3/(3*L)

    # COMPUTING M.  NUMBER OF DERIVATIVES Fp[m] TO COMPUTE
    M = aux_M_Fp(ctx, A, eps4, a, B1, L)
    Fp = {}
    for n in range(M, 3*L-2):
        Fp[n] = 0

    #  But I have not seen an instance of  M != 3*L-3
    #
    #  DETERMINATION OF  J  THE NUMBER OF TERMS NEEDED
    #            IN THE TAYLOR SERIES OF F.
    #  See II Section 3.11 equation (49))
    h1 = eps4/(632*A)
    h2 = ctx.pi*ctx.pi*B1*a *ctx.sqrt(3)*math.e*math.e
    h2 = h1 * ctx.power((h2/M**2),(M-1)/3) / M
    h3 = min(h1,h2)
    J=12
    jvalue = (2*ctx.pi)**J / ctx.gamma(J+1)
    while jvalue > h3:
        J = J+1
        jvalue = (2*ctx.pi)*jvalue/J

    # COMPUTING eps5[m] for 1 <= m <= 21
    #  See II Section 10 equation (43)
    eps5={}
    foreps5 = math.pi*math.pi*B1*a
    for m in range(0,22):
        aux1 = math.pow(foreps5, m/3)/(316.*A)
        aux2 = ctx.gamma(m+1)/ctx.gamma(m/3.0+0.5)
        aux2 = math.sqrt(aux2)
        eps5[m] = aux1*aux2*eps4

    # COMPUTING wpfp
    #  See II Section 3.13 equation (59)
    twenty = min(3*L-3, 21)+1
    aux = 6812*J
    wpfp = ctx.mag(44*J)
    for m in range(0, twenty):
        wpfp = max(wpfp, ctx.mag(aux*ctx.gamma(m+1)/eps5[m]))
    # COMPUTING N AND p
    #  See II Section
    ctx.prec = wpfp + ctx.mag(t) + 20
    a = ctx.sqrt(t/(2*ctx.pi))
    N = ctx.floor(a)
    p = 1-2*(a-N)

    # now we get a rounded version of p to the precision wpfp
    # this possibly is not necessary
    num = ctx.floor(p*(ctx.mpf(2)**wpfp))
    difference = p * (ctx.mpf(2)**wpfp)-num
    if difference < 0.5:
        num = num
    else:
        num = num+1
    p = ctx.convert(num * (ctx.mpf(2)**(-wpfp)))

    # COMPUTING THE COEFFICIENTS c[n] = cc[n]
    # We shall use the notation cc[n], since there is
    # a constant that is called c
    # See II Section 3.14
    # We compute the coefficients and also save then in a
    # cache.  The bulk of the computation is passed to
    # the function  coef()
    #
    #  eps6 is defined in II Section 3.13  equation (58)
    eps6 = ctx.power(2*ctx.pi, J)/(ctx.gamma(J+1)*3*J)

    #  Now we compute the coefficients
    cc={}
    cont={}
    cont, pipowers = coef(ctx, J, eps6)
    cc = cont.copy()   # we need a copy since we have
    Fp={}
    for n in range(M, 3*L-2):
        Fp[n] = 0
    ctx.prec = wpfp
    for m in range(0,M+1):
        sumP = 0
        for k in range(2*J-m-1,-1,-1):
            sumP = (sumP * p) + cc[k]
        Fp[m] = sumP
        # preparation of the new coefficients
        for k in range(0, 2*J-m-1):
            cc[k] = (k+1) * cc[k+1]

    # COMPUTING THE NUMBERS  d[n,k]
    #  See II Section 3.17

    #  First we compute the working precisions wpd[k]
    #   Se II equation (92)
    wpd = {}
    d1 = max(6, ctx.mag(40*L*L))
    d2 = 13+ctx.mag((1+abs(sigma))*A)-ctx.mag(eps4)-1
    const = ctx.ln(8/(ctx.pi*ctx.pi*a*a*B1*B1)) /2
    for n in range(0,L):
        d3 = ctx.mag(ctx.sqrt(ctx.gamma(n-0.5)))-ctx.floor(n*const)+d2
        wpd[n] = max(d3,d1)

    # procedure of II Section 3.17
    ctx.prec = wpd[1]+10
    psigma = 1-(2*sigma)
    d = {}
    d[0,0,-2]=0; d[0,0,-1]=0; d[0,0,0]=1; d[0,0,1]=0
    d[0,-1,-2]=0; d[0,-1,-1]=0; d[0,-1,0]=1; d[0,-1,1]=0
    for n in range(1,L):
        ctx.prec = wpd[n]+10
        for k in range(0,3*n//2+1):
            m = 3*n-2*k
            if (m!=0):
                m1 = ctx.one/m
                c1 = m1/4
                c2 = (psigma*m1)/2
                c3 = -(m+1)
                d[0,n,k] = c3*d[0,n-1,k-2]+c1*d[0,n-1,k]+c2*d[0,n-1,k-1]
            else:
                d[0,n,k]=0
                for r in range(0,k):
                    add = d[0,n,r]*(ctx.one*ctx.fac(2*k-2*r)/ctx.fac(k-r))
                    d[0,n,k] -= ((-1)**(k-r))*add
        d[0,n,-2]=0; d[0,n,-1]=0; d[0,n,3*n//2+1]=0

    for mu in range(-2,der+1):
        for n in range(-2,L):
            for k in range(-3,max(1,3*n//2+2)):
                if ((mu<0)or (n<0) or(k<0)or (k>3*n//2)):
                    d[mu,n,k] = 0

    for mu in range(1,der+1):
        for n in range(0,L):
            ctx.prec = wpd[n]+10
            for k in range(0,3*n//2+1):
                aux=(2*mu-2)*d[mu-2,n-2,k-3]+2*(sigma+n-2)*d[mu-1,n-2,k-3]
                d[mu,n,k] = aux - d[mu-1,n-1,k-1]

    # COMPUTING THE COEFFICIENTS t[k,l]
    #  See II Section 3.9
    #
    # computing the needed wp
    wptcoef = {}
    wpterm = {}
    ctx.prec = 15
    c1 = ctx.mag(40*(L+2))
    c2 = ctx.mag(68*(L+2)*A)
    c4 = ctx.mag(B1*a*math.sqrt(ctx.pi))-1
    for k in range(0,L):
        c3 = c2 - k*c4+ctx.mag(ctx.fac(k+0.5))/2.
        wptcoef[k] = max(c1,c3-ctx.mag(eps4)+1)+1 +10
        wpterm[k] = max(c1,ctx.mag(L+2)+c3-ctx.mag(eps3)+1)+1 +10

    # check of power of pi

    # computing the fortcoef[mu,k,ell]
    fortcoef={}
    for mu in derivatives:
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                fortcoef[mu,k,ell]=0

    for mu in derivatives:
        for k in range(0,L):
            ctx.prec = wptcoef[k]
            for ell in range(0,3*k//2+1):
                fortcoef[mu,k,ell]=d[mu,k,ell]*Fp[3*k-2*ell]/pipowers[2*k-ell]
                fortcoef[mu,k,ell]=fortcoef[mu,k,ell]/((2*ctx.j)**ell)

    def trunc_a(t):
        wp = ctx.prec
        ctx.prec = wp + 2
        aa = ctx.sqrt(t/(2*ctx.pi))
        ctx.prec = wp
        return aa

    # computing the tcoef[chi,k,ell]
    tcoef={}
    for chi in derivatives:
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                tcoef[chi,k,ell]=0
    ctx.prec = wptcoef[0]+3
    aa = trunc_a(t)
    la = -ctx.ln(aa)

    for chi in derivatives:
        for k in range(0,L):
            ctx.prec = wptcoef[k]
            for ell in range(0,3*k//2+1):
                tcoef[chi,k,ell] = 0
                for mu in range(0, chi+1):
                    tcoefter = ctx.binomial(chi,mu) * la**mu * \
                        fortcoef[chi-mu,k,ell]
                    tcoef[chi,k,ell] += tcoefter

    # COMPUTING tv[k,ell]
    # See II Section 3.8

    # Computing the powers av[k] = a**(-k)
    ctx.prec = wptcoef[0] + 2

    # a has a good value of a.
    # See II Section 3.6
    av = {}
    av[0] = 1
    av[1] = av[0]/a

    ctx.prec = wptcoef[0]
    for k in range(2,L):
        av[k] = av[k-1] * av[1]

    # Computing the quotients
    tv = {}
    for chi in derivatives:
        for k in range(0,L):
            ctx.prec = wptcoef[k]
            for ell in range(0,3*k//2+1):
                tv[chi,k,ell] = tcoef[chi,k,ell]* av[k]

    # COMPUTING THE TERMS term[k]
    # See II Section 3.6
    term = {}
    for chi in derivatives:
        for n in range(0,L):
            ctx.prec = wpterm[n]
            te = 0
            for k in range(0, 3*n//2+1):
                te += tv[chi,n,k]
            term[chi,n] = te

    # COMPUTING  rssum
    # See II Section 3.5
    rssum={}
    ctx.prec=15
    rsbound = math.sqrt(ctx.pi) * c /(b*a)
    ctx.prec=15
    wprssum = ctx.mag(4.4*((L+3)**2)*rsbound / eps2)
    wprssum = max(wprssum, ctx.mag(10*(L+1)))
    ctx.prec = wprssum
    for chi in derivatives:
        rssum[chi] = 0
        for k in range(1,L+1):
            rssum[chi] += term[chi,L-k]

    # COMPUTING S3
    # See II Section 3.19
    ctx.prec = 15
    A2 = 2**(ctx.mag(rssum[0]))
    eps8 = eps/(3* A2)
    T = t * ctx.ln(t/(2*ctx.pi))
    wps3 = 5 + ctx.mag((1+(2/eps8)*ctx.power(a,-sigma))*T)

    ctx.prec = wps3
    tpi = t/(2*ctx.pi)
    arg = (t/2)*ctx.ln(tpi)-(t/2)-ctx.pi/8
    U = ctx.expj(-arg)
    a = trunc_a(t)
    asigma = ctx.power(a, -sigma)
    S3 = ((-1)**(N-1)) * asigma * U

    # COMPUTING S1 the zetasum
    # See II Section 3.18
    ctx.prec = 15
    wpsum = 4 + ctx.mag((N+ctx.power(N,1-sigma))*ctx.ln(N)/eps1)

    ctx.prec = wpsum + 10
    '''
    # This can be improved
    S1 = {}
    for chi in derivatives:
        S1[chi] = 0
    for n in range(1,int(N)+1):
        ln = ctx.ln(n)
        expn = ctx.exp(-ln*(sigma+ctx.j*t))
        for chi in derivatives:
            term = ctx.power(-ln, chi)*expn
            S1[chi] += term
    '''
    S1 = ctx._zetasum(s, 1, int(N)-1, derivatives)[0]

    # END OF COMPUTATION
    #  See II Section 3.1
    ctx.prec = 15
    absS1 = abs(S1[der])
    absS2 = abs(rssum[der] * S3)
    wpend = max(6, wpinitial + ctx.mag(6*(3*absS1+7*absS2)))
    ctx.prec = wpend
    rz = {}
    for chi in derivatives:
        rz[chi] = S1[chi]+rssum[chi]*S3
    ctx.prec = wpinitial
    return rz


def z_half(ctx,t,der=0):
    r"""
    z_half(t,der=0) Computes Z^(der)(t)
    """
    s=ctx.mpf('0.5')+ctx.j*t
    wpinitial = ctx.prec
    ctx.prec = 15
    tt = t/(2*ctx.pi)
    wptheta = wpinitial +1 + ctx.mag(3*(tt**1.5)*ctx.ln(tt))
    wpz = wpinitial + 1 + ctx.mag(12*tt*ctx.ln(tt))
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t)
    ctx.prec = wpz
    rz = Rzeta_set(ctx,s, range(der+1))
    if der > 0: ps1 = ctx._re(ctx.psi(0,s/2)/2 - ctx.ln(ctx.pi)/2)
    if der > 1: ps2 = ctx._re(ctx.j*ctx.psi(1,s/2)/4)
    if der > 2: ps3 = ctx._re(-ctx.psi(2,s/2)/8)
    if der > 3: ps4 = ctx._re(-ctx.j*ctx.psi(3,s/2)/16)
    exptheta = ctx.expj(theta)
    if der == 0:
        z = 2*exptheta*rz[0]
    if der == 1:
        zf = 2j*exptheta
        z = zf*(ps1*rz[0]+rz[1])
    if der == 2:
        zf = 2 * exptheta
        z = -zf*(2*rz[1]*ps1+rz[0]*ps1**2+rz[2]-ctx.j*rz[0]*ps2)
    if der == 3:
        zf = -2j*exptheta
        z = 3*rz[1]*ps1**2+rz[0]*ps1**3+3*ps1*rz[2]
        z = zf*(z-3j*rz[1]*ps2-3j*rz[0]*ps1*ps2+rz[3]-rz[0]*ps3)
    if der == 4:
        zf = 2*exptheta
        z = 4*rz[1]*ps1**3+rz[0]*ps1**4+6*ps1**2*rz[2]
        z = z-12j*rz[1]*ps1*ps2-6j*rz[0]*ps1**2*ps2-6j*rz[2]*ps2-3*rz[0]*ps2*ps2
        z = z + 4*ps1*rz[3]-4*rz[1]*ps3-4*rz[0]*ps1*ps3+rz[4]+ctx.j*rz[0]*ps4
        z = zf*z
    ctx.prec = wpinitial
    return ctx._re(z)

def zeta_half(ctx, s, k=0):
    """
    zeta_half(s,k=0) Computes zeta^(k)(s) when Re s = 0.5
    """
    wpinitial = ctx.prec
    sigma = ctx._re(s)
    t = ctx._im(s)
    #--- compute wptheta, wpR, wpbasic ---
    ctx.prec = 53
    #  X see II Section 3.21 (109) and (110)
    if sigma > 0:
        X = ctx.sqrt(abs(s))
    else:
        X = (2*ctx.pi)**(sigma-1) * abs(1-s)**(0.5-sigma)
    # M1  see II Section 3.21 (111) and (112)
    if sigma > 0:
        M1 = 2*ctx.sqrt(t/(2*ctx.pi))
    else:
        M1 = 4 * t * X
    # T  see II Section 3.21 (113)
    abst = abs(0.5-s)
    T = 2* abst*math.log(abst)
    # computing wpbasic, wptheta, wpR  see II Section 3.21
    wpbasic = max(6,3+ctx.mag(t))
    wpbasic2 = 2+ctx.mag(2.12*M1+21.2*M1*X+1.3*M1*X*T)+wpinitial+1
    wpbasic = max(wpbasic, wpbasic2)
    wptheta = max(4, 3+ctx.mag(2.7*M1*X)+wpinitial+1)
    wpR = 3+ctx.mag(1.1+2*X)+wpinitial+1
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t-ctx.j*(sigma-ctx.mpf('0.5')))
    if k > 0: ps1 = (ctx._re(ctx.psi(0,s/2)))/2 - ctx.ln(ctx.pi)/2
    if k > 1: ps2 = -(ctx._im(ctx.psi(1,s/2)))/4
    if k > 2: ps3 = -(ctx._re(ctx.psi(2,s/2)))/8
    if k > 3: ps4 = (ctx._im(ctx.psi(3,s/2)))/16
    ctx.prec = wpR
    xrz = Rzeta_set(ctx,s,range(k+1))
    yrz={}
    for chi in range(0,k+1):
        yrz[chi] = ctx.conj(xrz[chi])
    ctx.prec = wpbasic
    exptheta = ctx.expj(-2*theta)
    if k==0:
        zv = xrz[0]+exptheta*yrz[0]
    if k==1:
        zv1 = -yrz[1] - 2*yrz[0]*ps1
        zv = xrz[1] + exptheta*zv1
    if k==2:
        zv1 = 4*yrz[1]*ps1+4*yrz[0]*(ps1**2)+yrz[2]+2j*yrz[0]*ps2
        zv = xrz[2]+exptheta*zv1
    if k==3:
        zv1 = -12*yrz[1]*ps1**2-8*yrz[0]*ps1**3-6*yrz[2]*ps1-6j*yrz[1]*ps2
        zv1 = zv1 - 12j*yrz[0]*ps1*ps2-yrz[3]+2*yrz[0]*ps3
        zv = xrz[3]+exptheta*zv1
    if k == 4:
        zv1 = 32*yrz[1]*ps1**3 +16*yrz[0]*ps1**4+24*yrz[2]*ps1**2
        zv1 = zv1 +48j*yrz[1]*ps1*ps2+48j*yrz[0]*(ps1**2)*ps2
        zv1 = zv1+12j*yrz[2]*ps2-12*yrz[0]*ps2**2+8*yrz[3]*ps1-8*yrz[1]*ps3
        zv1 = zv1-16*yrz[0]*ps1*ps3+yrz[4]-2j*yrz[0]*ps4
        zv = xrz[4]+exptheta*zv1
    ctx.prec = wpinitial
    return zv

def zeta_offline(ctx, s, k=0):
    """
    Computes zeta^(k)(s) off the line
    """
    wpinitial = ctx.prec
    sigma = ctx._re(s)
    t = ctx._im(s)
    #--- compute wptheta, wpR, wpbasic ---
    ctx.prec = 53
    #  X see II Section 3.21 (109) and (110)
    if sigma > 0:
        X = ctx.power(abs(s), 0.5)
    else:
        X = ctx.power(2*ctx.pi, sigma-1)*ctx.power(abs(1-s),0.5-sigma)
    # M1  see II Section 3.21 (111) and (112)
    if (sigma > 0):
        M1 = 2*ctx.sqrt(t/(2*ctx.pi))
    else:
        M1 = 4 * t * X
    # M2  see II Section 3.21 (111) and (112)
    if (1-sigma > 0):
        M2 = 2*ctx.sqrt(t/(2*ctx.pi))
    else:
        M2 = 4*t*ctx.power(2*ctx.pi, -sigma)*ctx.power(abs(s),sigma-0.5)
    # T  see II Section 3.21 (113)
    abst = abs(0.5-s)
    T = 2* abst*math.log(abst)
    # computing wpbasic, wptheta, wpR  see II Section 3.21
    wpbasic = max(6,3+ctx.mag(t))
    wpbasic2 = 2+ctx.mag(2.12*M1+21.2*M2*X+1.3*M2*X*T)+wpinitial+1
    wpbasic = max(wpbasic, wpbasic2)
    wptheta = max(4, 3+ctx.mag(2.7*M2*X)+wpinitial+1)
    wpR = 3+ctx.mag(1.1+2*X)+wpinitial+1
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t-ctx.j*(sigma-ctx.mpf('0.5')))
    s1 = s
    s2 = ctx.conj(1-s1)
    ctx.prec = wpR
    xrz, yrz = Rzeta_simul(ctx, s, k)
    if k > 0: ps1 = (ctx.psi(0,s1/2)+ctx.psi(0,(1-s1)/2))/4 - ctx.ln(ctx.pi)/2
    if k > 1: ps2 = ctx.j*(ctx.psi(1,s1/2)-ctx.psi(1,(1-s1)/2))/8
    if k > 2: ps3 = -(ctx.psi(2,s1/2)+ctx.psi(2,(1-s1)/2))/16
    if k > 3: ps4 = -ctx.j*(ctx.psi(3,s1/2)-ctx.psi(3,(1-s1)/2))/32
    ctx.prec = wpbasic
    exptheta = ctx.expj(-2*theta)
    if k == 0:
        zv = xrz[0]+exptheta*yrz[0]
    if k == 1:
        zv1 = -yrz[1]-2*yrz[0]*ps1
        zv = xrz[1]+exptheta*zv1
    if k == 2:
        zv1 = 4*yrz[1]*ps1+4*yrz[0]*(ps1**2) +yrz[2]+2j*yrz[0]*ps2
        zv = xrz[2]+exptheta*zv1
    if k == 3:
        zv1 = -12*yrz[1]*ps1**2 -8*yrz[0]*ps1**3-6*yrz[2]*ps1-6j*yrz[1]*ps2
        zv1 = zv1 - 12j*yrz[0]*ps1*ps2-yrz[3]+2*yrz[0]*ps3
        zv = xrz[3]+exptheta*zv1
    if k == 4:
        zv1 = 32*yrz[1]*ps1**3 +16*yrz[0]*ps1**4+24*yrz[2]*ps1**2
        zv1 = zv1 +48j*yrz[1]*ps1*ps2+48j*yrz[0]*(ps1**2)*ps2
        zv1 = zv1+12j*yrz[2]*ps2-12*yrz[0]*ps2**2+8*yrz[3]*ps1-8*yrz[1]*ps3
        zv1 = zv1-16*yrz[0]*ps1*ps3+yrz[4]-2j*yrz[0]*ps4
        zv = xrz[4]+exptheta*zv1
    ctx.prec = wpinitial
    return zv

def z_offline(ctx, w, k=0):
    r"""
    Computes Z(w) and its derivatives off the line
    """
    s = ctx.mpf('0.5')+ctx.j*w
    s1 = s
    s2 = ctx.conj(1-s1)
    wpinitial = ctx.prec
    ctx.prec = 35
    #  X see II Section 3.21 (109) and (110)
    # M1  see II Section 3.21 (111) and (112)
    if (ctx._re(s1) >= 0):
        M1 = 2*ctx.sqrt(ctx._im(s1)/(2 * ctx.pi))
        X = ctx.sqrt(abs(s1))
    else:
        X = (2*ctx.pi)**(ctx._re(s1)-1) * abs(1-s1)**(0.5-ctx._re(s1))
        M1 = 4 * ctx._im(s1)*X
    # M2  see II Section 3.21 (111) and (112)
    if (ctx._re(s2) >= 0):
        M2 = 2*ctx.sqrt(ctx._im(s2)/(2 * ctx.pi))
    else:
        M2 = 4 * ctx._im(s2)*(2*ctx.pi)**(ctx._re(s2)-1)*abs(1-s2)**(0.5-ctx._re(s2))
    # T  see II Section 3.21  Prop. 27
    T = 2*abs(ctx.siegeltheta(w))
    # defining some precisions
    # see II Section 3.22 (115), (116), (117)
    aux1 = ctx.sqrt(X)
    aux2 = aux1*(M1+M2)
    aux3 = 3 +wpinitial
    wpbasic = max(6, 3+ctx.mag(T), ctx.mag(aux2*(26+2*T))+aux3)
    wptheta = max(4,ctx.mag(2.04*aux2)+aux3)
    wpR = ctx.mag(4*aux1)+aux3
    # now the computations
    ctx.prec = wptheta
    theta = ctx.siegeltheta(w)
    ctx.prec = wpR
    xrz, yrz = Rzeta_simul(ctx,s,k)
    pta = 0.25 + 0.5j*w
    ptb = 0.25 - 0.5j*w
    if k > 0: ps1 = 0.25*(ctx.psi(0,pta)+ctx.psi(0,ptb)) - ctx.ln(ctx.pi)/2
    if k > 1: ps2 = (1j/8)*(ctx.psi(1,pta)-ctx.psi(1,ptb))
    if k > 2: ps3 = (-1./16)*(ctx.psi(2,pta)+ctx.psi(2,ptb))
    if k > 3: ps4 = (-1j/32)*(ctx.psi(3,pta)-ctx.psi(3,ptb))
    ctx.prec = wpbasic
    exptheta = ctx.expj(theta)
    if k == 0:
        zv = exptheta*xrz[0]+yrz[0]/exptheta
    j = ctx.j
    if k == 1:
        zv = j*exptheta*(xrz[1]+xrz[0]*ps1)-j*(yrz[1]+yrz[0]*ps1)/exptheta
    if k == 2:
        zv = exptheta*(-2*xrz[1]*ps1-xrz[0]*ps1**2-xrz[2]+j*xrz[0]*ps2)
        zv =zv + (-2*yrz[1]*ps1-yrz[0]*ps1**2-yrz[2]-j*yrz[0]*ps2)/exptheta
    if k == 3:
        zv1 = -3*xrz[1]*ps1**2-xrz[0]*ps1**3-3*xrz[2]*ps1+j*3*xrz[1]*ps2
        zv1 = (zv1+ 3j*xrz[0]*ps1*ps2-xrz[3]+xrz[0]*ps3)*j*exptheta
        zv2 = 3*yrz[1]*ps1**2+yrz[0]*ps1**3+3*yrz[2]*ps1+j*3*yrz[1]*ps2
        zv2 = j*(zv2 + 3j*yrz[0]*ps1*ps2+ yrz[3]-yrz[0]*ps3)/exptheta
        zv = zv1+zv2
    if k == 4:
        zv1 = 4*xrz[1]*ps1**3+xrz[0]*ps1**4 + 6*xrz[2]*ps1**2
        zv1 = zv1-12j*xrz[1]*ps1*ps2-6j*xrz[0]*ps1**2*ps2-6j*xrz[2]*ps2
        zv1 = zv1-3*xrz[0]*ps2*ps2+4*xrz[3]*ps1-4*xrz[1]*ps3-4*xrz[0]*ps1*ps3
        zv1 = zv1+xrz[4]+j*xrz[0]*ps4
        zv2 = 4*yrz[1]*ps1**3+yrz[0]*ps1**4 + 6*yrz[2]*ps1**2
        zv2 = zv2+12j*yrz[1]*ps1*ps2+6j*yrz[0]*ps1**2*ps2+6j*yrz[2]*ps2
        zv2 = zv2-3*yrz[0]*ps2*ps2+4*yrz[3]*ps1-4*yrz[1]*ps3-4*yrz[0]*ps1*ps3
        zv2 = zv2+yrz[4]-j*yrz[0]*ps4
        zv = exptheta*zv1+zv2/exptheta
    ctx.prec = wpinitial
    return zv

@defun
def rs_zeta(ctx, s, derivative=0, **kwargs):
    if derivative > 4:
        raise NotImplementedError
    s = ctx.convert(s)
    re = ctx._re(s); im = ctx._im(s)
    if im < 0:
        z = ctx.conj(ctx.rs_zeta(ctx.conj(s), derivative))
        return z
    critical_line = (re == 0.5)
    if critical_line:
        return zeta_half(ctx, s, derivative)
    else:
        return zeta_offline(ctx, s, derivative)

@defun
def rs_z(ctx, w, derivative=0):
    w = ctx.convert(w)
    re = ctx._re(w); im = ctx._im(w)
    if re < 0:
        return rs_z(ctx, -w, derivative)
    critical_line = (im == 0)
    if critical_line :
        return z_half(ctx, w, derivative)
    else:
        return z_offline(ctx, w, derivative)
