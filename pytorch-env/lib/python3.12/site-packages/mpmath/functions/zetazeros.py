"""
The function zetazero(n) computes the n-th nontrivial zero of zeta(s).

The general strategy is to locate a block of Gram intervals B where we
know exactly the number of zeros contained and which of those zeros
is that which we search.

If n <= 400 000 000  we know exactly the Rosser exceptions, contained
in a list in this file. Hence for n<=400 000 000 we simply
look at these list of exceptions. If our zero is implicated in one of
these exceptions we have our block B.  In other case we simply locate
the good Rosser block containing our zero.

For n > 400 000 000 we apply the method of Turing, as complemented by
Lehman, Brent and Trudgian  to find a suitable B.
"""

from .functions import defun, defun_wrapped

def find_rosser_block_zero(ctx, n):
    """for n<400 000 000 determines a block were one find our zero"""
    for k in range(len(_ROSSER_EXCEPTIONS)//2):
        a=_ROSSER_EXCEPTIONS[2*k][0]
        b=_ROSSER_EXCEPTIONS[2*k][1]
        if ((a<= n-2) and (n-1 <= b)):
            t0 = ctx.grampoint(a)
            t1 = ctx.grampoint(b)
            v0 = ctx._fp.siegelz(t0)
            v1 = ctx._fp.siegelz(t1)
            my_zero_number = n-a-1
            zero_number_block = b-a
            pattern = _ROSSER_EXCEPTIONS[2*k+1]
            return (my_zero_number, [a,b], [t0,t1], [v0,v1])
    k = n-2
    t,v,b = compute_triple_tvb(ctx, k)
    T = [t]
    V = [v]
    while b < 0:
        k -= 1
        t,v,b = compute_triple_tvb(ctx, k)
        T.insert(0,t)
        V.insert(0,v)
    my_zero_number = n-k-1
    m = n-1
    t,v,b = compute_triple_tvb(ctx, m)
    T.append(t)
    V.append(v)
    while b < 0:
        m += 1
        t,v,b = compute_triple_tvb(ctx, m)
        T.append(t)
        V.append(v)
    return (my_zero_number, [k,m], T, V)

def wpzeros(t):
    """Precision needed to compute higher zeros"""
    wp = 53
    if t > 3*10**8:
        wp = 63
    if t > 10**11:
        wp = 70
    if t > 10**14:
        wp = 83
    return wp

def separate_zeros_in_block(ctx, zero_number_block, T, V, limitloop=None,
    fp_tolerance=None):
    """Separate the zeros contained in the block T, limitloop
    determines how long one must search"""
    if limitloop is None:
        limitloop = ctx.inf
    loopnumber = 0
    variations = count_variations(V)
    while ((variations < zero_number_block) and (loopnumber <limitloop)):
        a = T[0]
        v = V[0]
        newT = [a]
        newV = [v]
        variations = 0
        for n in range(1,len(T)):
            b2 = T[n]
            u = V[n]
            if (u*v>0):
                alpha = ctx.sqrt(u/v)
                b= (alpha*a+b2)/(alpha+1)
            else:
                b = (a+b2)/2
            if fp_tolerance < 10:
                w = ctx._fp.siegelz(b)
                if abs(w)<fp_tolerance:
                    w = ctx.siegelz(b)
            else:
                w=ctx.siegelz(b)
            if v*w<0:
                variations += 1
            newT.append(b)
            newV.append(w)
            u = V[n]
            if u*w <0:
                variations += 1
            newT.append(b2)
            newV.append(u)
            a = b2
            v = u
        T = newT
        V = newV
        loopnumber +=1
        if (limitloop>ITERATION_LIMIT)and(loopnumber>2)and(variations+2==zero_number_block):
            dtMax=0
            dtSec=0
            kMax = 0
            for k1 in range(1,len(T)):
                dt = T[k1]-T[k1-1]
                if dt > dtMax:
                    kMax=k1
                    dtSec = dtMax
                    dtMax = dt
                elif  (dt<dtMax) and(dt >dtSec):
                    dtSec = dt
            if dtMax>3*dtSec:
                f = lambda x: ctx.rs_z(x,derivative=1)
                t0=T[kMax-1]
                t1 = T[kMax]
                t=ctx.findroot(f,  (t0,t1), solver ='illinois',verify=False, verbose=False)
                v = ctx.siegelz(t)
                if (t0<t) and (t<t1) and (v*V[kMax]<0):
                    T.insert(kMax,t)
                    V.insert(kMax,v)
        variations = count_variations(V)
    if variations == zero_number_block:
        separated = True
    else:
        separated = False
    return (T,V, separated)

def separate_my_zero(ctx, my_zero_number, zero_number_block, T, V, prec):
    """If we know which zero of this block is mine,
    the function separates the zero"""
    variations = 0
    v0 = V[0]
    for k in range(1,len(V)):
        v1 = V[k]
        if v0*v1 < 0:
            variations +=1
            if variations == my_zero_number:
                k0 = k
                leftv = v0
                rightv = v1
        v0 = v1
    t1 = T[k0]
    t0 = T[k0-1]
    ctx.prec = prec
    wpz = wpzeros(my_zero_number*ctx.log(my_zero_number))

    guard = 4*ctx.mag(my_zero_number)
    precs = [ctx.prec+4]
    index=0
    while precs[0] > 2*wpz:
        index +=1
        precs = [precs[0] // 2 +3+2*index] + precs
    ctx.prec = precs[0] + guard
    r = ctx.findroot(lambda x:ctx.siegelz(x), (t0,t1), solver ='illinois', verbose=False)
    #print "first step at", ctx.dps, "digits"
    z=ctx.mpc(0.5,r)
    for prec in precs[1:]:
        ctx.prec = prec + guard
        #print "refining to", ctx.dps, "digits"
        znew = z - ctx.zeta(z) / ctx.zeta(z, derivative=1)
        #print "difference", ctx.nstr(abs(z-znew))
        z=ctx.mpc(0.5,ctx.im(znew))
    return ctx.im(z)

def sure_number_block(ctx, n):
    """The number of good Rosser blocks needed to apply
    Turing method
    References:
    R. P. Brent, On the Zeros of the Riemann Zeta Function
    in the Critical Strip, Math. Comp. 33 (1979) 1361--1372
    T. Trudgian, Improvements to Turing Method, Math. Comp."""
    if n < 9*10**5:
        return(2)
    g = ctx.grampoint(n-100)
    lg = ctx._fp.ln(g)
    brent = 0.0061 * lg**2 +0.08*lg
    trudgian = 0.0031 * lg**2 +0.11*lg
    N = ctx.ceil(min(brent,trudgian))
    N = int(N)
    return N

def compute_triple_tvb(ctx, n):
    t = ctx.grampoint(n)
    v = ctx._fp.siegelz(t)
    if ctx.mag(abs(v))<ctx.mag(t)-45:
        v = ctx.siegelz(t)
    b = v*(-1)**n
    return t,v,b



ITERATION_LIMIT = 4

def search_supergood_block(ctx, n, fp_tolerance):
    """To use for n>400 000 000"""
    sb = sure_number_block(ctx, n)
    number_goodblocks = 0
    m2 = n-1
    t, v, b = compute_triple_tvb(ctx, m2)
    Tf = [t]
    Vf = [v]
    while b < 0:
        m2 += 1
        t,v,b = compute_triple_tvb(ctx, m2)
        Tf.append(t)
        Vf.append(v)
    goodpoints = [m2]
    T = [t]
    V = [v]
    while number_goodblocks < 2*sb:
        m2 += 1
        t, v, b = compute_triple_tvb(ctx, m2)
        T.append(t)
        V.append(v)
        while b < 0:
            m2 += 1
            t,v,b = compute_triple_tvb(ctx, m2)
            T.append(t)
            V.append(v)
        goodpoints.append(m2)
        zn = len(T)-1
        A, B, separated =\
           separate_zeros_in_block(ctx, zn, T, V, limitloop=ITERATION_LIMIT,
                fp_tolerance=fp_tolerance)
        Tf.pop()
        Tf.extend(A)
        Vf.pop()
        Vf.extend(B)
        if separated:
            number_goodblocks += 1
        else:
            number_goodblocks = 0
        T = [t]
        V = [v]
    # Now the same procedure to the left
    number_goodblocks = 0
    m2 = n-2
    t, v, b = compute_triple_tvb(ctx, m2)
    Tf.insert(0,t)
    Vf.insert(0,v)
    while b < 0:
        m2 -= 1
        t,v,b = compute_triple_tvb(ctx, m2)
        Tf.insert(0,t)
        Vf.insert(0,v)
    goodpoints.insert(0,m2)
    T = [t]
    V = [v]
    while number_goodblocks < 2*sb:
        m2 -= 1
        t, v, b = compute_triple_tvb(ctx, m2)
        T.insert(0,t)
        V.insert(0,v)
        while b < 0:
            m2 -= 1
            t,v,b = compute_triple_tvb(ctx, m2)
            T.insert(0,t)
            V.insert(0,v)
        goodpoints.insert(0,m2)
        zn = len(T)-1
        A, B, separated =\
           separate_zeros_in_block(ctx, zn, T, V, limitloop=ITERATION_LIMIT, fp_tolerance=fp_tolerance)
        A.pop()
        Tf = A+Tf
        B.pop()
        Vf = B+Vf
        if separated:
            number_goodblocks += 1
        else:
            number_goodblocks = 0
        T = [t]
        V = [v]
    r = goodpoints[2*sb]
    lg = len(goodpoints)
    s = goodpoints[lg-2*sb-1]
    tr, vr, br = compute_triple_tvb(ctx, r)
    ar = Tf.index(tr)
    ts, vs, bs = compute_triple_tvb(ctx, s)
    as1 = Tf.index(ts)
    T = Tf[ar:as1+1]
    V = Vf[ar:as1+1]
    zn = s-r
    A, B, separated =\
       separate_zeros_in_block(ctx, zn,T,V,limitloop=ITERATION_LIMIT, fp_tolerance=fp_tolerance)
    if separated:
        return (n-r-1,[r,s],A,B)
    q = goodpoints[sb]
    lg = len(goodpoints)
    t = goodpoints[lg-sb-1]
    tq, vq, bq = compute_triple_tvb(ctx, q)
    aq = Tf.index(tq)
    tt, vt, bt = compute_triple_tvb(ctx, t)
    at = Tf.index(tt)
    T = Tf[aq:at+1]
    V = Vf[aq:at+1]
    return (n-q-1,[q,t],T,V)

def count_variations(V):
    count = 0
    vold = V[0]
    for n in range(1, len(V)):
        vnew = V[n]
        if vold*vnew < 0:
            count +=1
        vold = vnew
    return count

def pattern_construct(ctx, block, T, V):
    pattern = '('
    a = block[0]
    b = block[1]
    t0,v0,b0 = compute_triple_tvb(ctx, a)
    k = 0
    k0 = 0
    for n in range(a+1,b+1):
        t1,v1,b1 = compute_triple_tvb(ctx, n)
        lgT =len(T)
        while (k < lgT) and (T[k] <= t1):
            k += 1
        L = V[k0:k]
        L.append(v1)
        L.insert(0,v0)
        count = count_variations(L)
        pattern = pattern + ("%s" % count)
        if b1 > 0:
            pattern = pattern + ')('
        k0 = k
        t0,v0,b0 = t1,v1,b1
    pattern = pattern[:-1]
    return pattern

@defun
def zetazero(ctx, n, info=False, round=True):
    r"""
    Computes the `n`-th nontrivial zero of `\zeta(s)` on the critical line,
    i.e. returns an approximation of the `n`-th largest complex number
    `s = \frac{1}{2} + ti` for which `\zeta(s) = 0`. Equivalently, the
    imaginary part `t` is a zero of the Z-function (:func:`~mpmath.siegelz`).

    **Examples**

    The first few zeros::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> zetazero(1)
        (0.5 + 14.13472514173469379045725j)
        >>> zetazero(2)
        (0.5 + 21.02203963877155499262848j)
        >>> zetazero(20)
        (0.5 + 77.14484006887480537268266j)

    Verifying that the values are zeros::

        >>> for n in range(1,5):
        ...     s = zetazero(n)
        ...     chop(zeta(s)), chop(siegelz(s.imag))
        ...
        (0.0, 0.0)
        (0.0, 0.0)
        (0.0, 0.0)
        (0.0, 0.0)

    Negative indices give the conjugate zeros (`n = 0` is undefined)::

        >>> zetazero(-1)
        (0.5 - 14.13472514173469379045725j)

    :func:`~mpmath.zetazero` supports arbitrarily large `n` and arbitrary precision::

        >>> mp.dps = 15
        >>> zetazero(1234567)
        (0.5 + 727690.906948208j)
        >>> mp.dps = 50
        >>> zetazero(1234567)
        (0.5 + 727690.9069482075392389420041147142092708393819935j)
        >>> chop(zeta(_)/_)
        0.0

    with *info=True*, :func:`~mpmath.zetazero` gives additional information::

        >>> mp.dps = 15
        >>> zetazero(542964976,info=True)
        ((0.5 + 209039046.578535j), [542964969, 542964978], 6, '(013111110)')

    This means that the zero is between Gram points 542964969 and 542964978;
    it is the 6-th zero between them. Finally (01311110) is the pattern
    of zeros in this interval. The numbers indicate the number of zeros
    in each Gram interval (Rosser blocks between parenthesis). In this case
    there is only one Rosser block of length nine.
    """
    n = int(n)
    if n < 0:
        return ctx.zetazero(-n).conjugate()
    if n == 0:
        raise ValueError("n must be nonzero")
    wpinitial = ctx.prec
    try:
        wpz, fp_tolerance = comp_fp_tolerance(ctx, n)
        ctx.prec = wpz
        if n < 400000000:
            my_zero_number, block, T, V =\
             find_rosser_block_zero(ctx, n)
        else:
            my_zero_number, block, T, V =\
             search_supergood_block(ctx, n, fp_tolerance)
        zero_number_block = block[1]-block[0]
        T, V, separated = separate_zeros_in_block(ctx, zero_number_block, T, V,
            limitloop=ctx.inf, fp_tolerance=fp_tolerance)
        if info:
            pattern = pattern_construct(ctx,block,T,V)
        prec = max(wpinitial, wpz)
        t = separate_my_zero(ctx, my_zero_number, zero_number_block,T,V,prec)
        v = ctx.mpc(0.5,t)
    finally:
        ctx.prec = wpinitial
    if round:
        v =+v
    if info:
        return (v,block,my_zero_number,pattern)
    else:
        return v

def gram_index(ctx, t):
    if t > 10**13:
        wp = 3*ctx.log(t, 10)
    else:
        wp = 0
    prec = ctx.prec
    try:
        ctx.prec += wp
        h = int(ctx.siegeltheta(t)/ctx.pi)
    finally:
        ctx.prec = prec
    return(h)

def count_to(ctx, t, T, V):
    count = 0
    vold = V[0]
    told = T[0]
    tnew = T[1]
    k = 1
    while tnew < t:
        vnew = V[k]
        if vold*vnew < 0:
            count += 1
        vold = vnew
        k += 1
        tnew = T[k]
    a = ctx.siegelz(t)
    if a*vold < 0:
        count += 1
    return count

def comp_fp_tolerance(ctx, n):
    wpz = wpzeros(n*ctx.log(n))
    if n < 15*10**8:
        fp_tolerance = 0.0005
    elif n <= 10**14:
        fp_tolerance = 0.1
    else:
        fp_tolerance = 100
    return wpz, fp_tolerance

@defun
def nzeros(ctx, t):
    r"""
    Computes the number of zeros of the Riemann zeta function in
    `(0,1) \times (0,t]`, usually denoted by `N(t)`.

    **Examples**

    The first zero has imaginary part between 14 and 15::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nzeros(14)
        0
        >>> nzeros(15)
        1
        >>> zetazero(1)
        (0.5 + 14.1347251417347j)

    Some closely spaced zeros::

        >>> nzeros(10**7)
        21136125
        >>> zetazero(21136125)
        (0.5 + 9999999.32718175j)
        >>> zetazero(21136126)
        (0.5 + 10000000.2400236j)
        >>> nzeros(545439823.215)
        1500000001
        >>> zetazero(1500000001)
        (0.5 + 545439823.201985j)
        >>> zetazero(1500000002)
        (0.5 + 545439823.325697j)

    This confirms the data given by J. van de Lune,
    H. J. J. te Riele and D. T. Winter in 1986.
    """
    if t < 14.1347251417347:
        return 0
    x = gram_index(ctx, t)
    k = int(ctx.floor(x))
    wpinitial = ctx.prec
    wpz, fp_tolerance = comp_fp_tolerance(ctx, k)
    ctx.prec = wpz
    a = ctx.siegelz(t)
    if k == -1 and a < 0:
        return 0
    elif k == -1 and a > 0:
        return 1
    if k+2 < 400000000:
        Rblock = find_rosser_block_zero(ctx, k+2)
    else:
        Rblock = search_supergood_block(ctx, k+2, fp_tolerance)
    n1, n2 = Rblock[1]
    if n2-n1 == 1:
        b = Rblock[3][0]
        if a*b > 0:
            ctx.prec = wpinitial
            return k+1
        else:
            ctx.prec = wpinitial
            return k+2
    my_zero_number,block, T, V = Rblock
    zero_number_block = n2-n1
    T, V, separated = separate_zeros_in_block(ctx,\
                                              zero_number_block, T, V,\
                                              limitloop=ctx.inf,\
                                            fp_tolerance=fp_tolerance)
    n = count_to(ctx, t, T, V)
    ctx.prec = wpinitial
    return n+n1+1

@defun_wrapped
def backlunds(ctx, t):
    r"""
    Computes the function
    `S(t) = \operatorname{arg} \zeta(\frac{1}{2} + it) / \pi`.

    See Titchmarsh Section 9.3 for details of the definition.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> backlunds(217.3)
        0.16302205431184

    Generally, the value is a small number. At Gram points it is an integer,
    frequently equal to 0::

        >>> chop(backlunds(grampoint(200)))
        0.0
        >>> backlunds(extraprec(10)(grampoint)(211))
        1.0
        >>> backlunds(extraprec(10)(grampoint)(232))
        -1.0

    The number of zeros of the Riemann zeta function up to height `t`
    satisfies `N(t) = \theta(t)/\pi + 1 + S(t)` (see :func:nzeros` and
    :func:`siegeltheta`)::

        >>> t = 1234.55
        >>> nzeros(t)
        842
        >>> siegeltheta(t)/pi+1+backlunds(t)
        842.0

    """
    return ctx.nzeros(t)-1-ctx.siegeltheta(t)/ctx.pi


"""
_ROSSER_EXCEPTIONS is a list of all  exceptions to
Rosser's rule for n <= 400 000 000.

Alternately the  entry is of type   [n,m], or a string.
The string is the zero pattern of the Block and the relevant
adjacent.  For example (010)3 corresponds to a block
composed of three Gram intervals, the first ant third without
a zero and the intermediate with a zero. The next Gram interval
contain three zeros. So that in total we have 4 zeros in 4 Gram
blocks. n and m are the indices of the Gram points  of this
interval of four Gram intervals. The Rosser exception is therefore
formed by the three Gram intervals that are signaled between
parenthesis.

We have included also some Rosser's exceptions beyond n=400 000 000
that are noted in the literature by some reason.

The list is composed from the data published in the references:

R. P. Brent, J. van de Lune, H. J. J. te Riele, D. T. Winter,
'On the Zeros of the Riemann Zeta Function in the Critical Strip. II',
Math. Comp. 39 (1982) 681--688.
See also Corrigenda in Math. Comp. 46 (1986) 771.

J. van de Lune, H. J. J. te Riele,
'On the Zeros of the Riemann Zeta Function in the Critical Strip. III',
Math. Comp. 41 (1983) 759--767.
See also  Corrigenda in Math. Comp. 46 (1986) 771.

J. van de Lune,
'Sums of Equal Powers of Positive Integers',
Dissertation,
Vrije Universiteit te Amsterdam, Centrum voor Wiskunde en Informatica,
Amsterdam, 1984.

Thanks to the authors all this papers and those others that have
contributed to make this possible.
"""







_ROSSER_EXCEPTIONS = \
[[13999525, 13999528], '(00)3',
[30783329, 30783332], '(00)3',
[30930926, 30930929], '3(00)',
[37592215, 37592218], '(00)3',
[40870156, 40870159], '(00)3',
[43628107, 43628110], '(00)3',
[46082042, 46082045], '(00)3',
[46875667, 46875670], '(00)3',
[49624540, 49624543], '3(00)',
[50799238, 50799241], '(00)3',
[55221453, 55221456], '3(00)',
[56948779, 56948782], '3(00)',
[60515663, 60515666], '(00)3',
[61331766, 61331770], '(00)40',
[69784843, 69784846], '3(00)',
[75052114, 75052117], '(00)3',
[79545240, 79545243], '3(00)',
[79652247, 79652250], '3(00)',
[83088043, 83088046], '(00)3',
[83689522, 83689525], '3(00)',
[85348958, 85348961], '(00)3',
[86513820, 86513823], '(00)3',
[87947596, 87947599], '3(00)',
[88600095, 88600098], '(00)3',
[93681183, 93681186], '(00)3',
[100316551, 100316554], '3(00)',
[100788444, 100788447], '(00)3',
[106236172, 106236175], '(00)3',
[106941327, 106941330], '3(00)',
[107287955, 107287958], '(00)3',
[107532016, 107532019], '3(00)',
[110571044, 110571047], '(00)3',
[111885253, 111885256], '3(00)',
[113239783, 113239786], '(00)3',
[120159903, 120159906], '(00)3',
[121424391, 121424394], '3(00)',
[121692931, 121692934], '3(00)',
[121934170, 121934173], '3(00)',
[122612848, 122612851], '3(00)',
[126116567, 126116570], '(00)3',
[127936513, 127936516], '(00)3',
[128710277, 128710280], '3(00)',
[129398902, 129398905], '3(00)',
[130461096, 130461099], '3(00)',
[131331947, 131331950], '3(00)',
[137334071, 137334074], '3(00)',
[137832603, 137832606], '(00)3',
[138799471, 138799474], '3(00)',
[139027791, 139027794], '(00)3',
[141617806, 141617809], '(00)3',
[144454931, 144454934], '(00)3',
[145402379, 145402382], '3(00)',
[146130245, 146130248], '3(00)',
[147059770, 147059773], '(00)3',
[147896099, 147896102], '3(00)',
[151097113, 151097116], '(00)3',
[152539438, 152539441], '(00)3',
[152863168, 152863171], '3(00)',
[153522726, 153522729], '3(00)',
[155171524, 155171527], '3(00)',
[155366607, 155366610], '(00)3',
[157260686, 157260689], '3(00)',
[157269224, 157269227], '(00)3',
[157755123, 157755126], '(00)3',
[158298484, 158298487], '3(00)',
[160369050, 160369053], '3(00)',
[162962787, 162962790], '(00)3',
[163724709, 163724712], '(00)3',
[164198113, 164198116], '3(00)',
[164689301, 164689305], '(00)40',
[164880228, 164880231], '3(00)',
[166201932, 166201935], '(00)3',
[168573836, 168573839], '(00)3',
[169750763, 169750766], '(00)3',
[170375507, 170375510], '(00)3',
[170704879, 170704882], '3(00)',
[172000992, 172000995], '3(00)',
[173289941, 173289944], '(00)3',
[173737613, 173737616], '3(00)',
[174102513, 174102516], '(00)3',
[174284990, 174284993], '(00)3',
[174500513, 174500516], '(00)3',
[175710609, 175710612], '(00)3',
[176870843, 176870846], '3(00)',
[177332732, 177332735], '3(00)',
[177902861, 177902864], '3(00)',
[179979095, 179979098], '(00)3',
[181233726, 181233729], '3(00)',
[181625435, 181625438], '(00)3',
[182105255, 182105259], '22(00)',
[182223559, 182223562], '3(00)',
[191116404, 191116407], '3(00)',
[191165599, 191165602], '3(00)',
[191297535, 191297539], '(00)22',
[192485616, 192485619], '(00)3',
[193264634, 193264638], '22(00)',
[194696968, 194696971], '(00)3',
[195876805, 195876808], '(00)3',
[195916548, 195916551], '3(00)',
[196395160, 196395163], '3(00)',
[196676303, 196676306], '(00)3',
[197889882, 197889885], '3(00)',
[198014122, 198014125], '(00)3',
[199235289, 199235292], '(00)3',
[201007375, 201007378], '(00)3',
[201030605, 201030608], '3(00)',
[201184290, 201184293], '3(00)',
[201685414, 201685418], '(00)22',
[202762875, 202762878], '3(00)',
[202860957, 202860960], '3(00)',
[203832577, 203832580], '3(00)',
[205880544, 205880547], '(00)3',
[206357111, 206357114], '(00)3',
[207159767, 207159770], '3(00)',
[207167343, 207167346], '3(00)',
[207482539, 207482543], '3(010)',
[207669540, 207669543], '3(00)',
[208053426, 208053429], '(00)3',
[208110027, 208110030], '3(00)',
[209513826, 209513829], '3(00)',
[212623522, 212623525], '(00)3',
[213841715, 213841718], '(00)3',
[214012333, 214012336], '(00)3',
[214073567, 214073570], '(00)3',
[215170600, 215170603], '3(00)',
[215881039, 215881042], '3(00)',
[216274604, 216274607], '3(00)',
[216957120, 216957123], '3(00)',
[217323208, 217323211], '(00)3',
[218799264, 218799267], '(00)3',
[218803557, 218803560], '3(00)',
[219735146, 219735149], '(00)3',
[219830062, 219830065], '3(00)',
[219897904, 219897907], '(00)3',
[221205545, 221205548], '(00)3',
[223601929, 223601932], '(00)3',
[223907076, 223907079], '3(00)',
[223970397, 223970400], '(00)3',
[224874044, 224874048], '22(00)',
[225291157, 225291160], '(00)3',
[227481734, 227481737], '(00)3',
[228006442, 228006445], '3(00)',
[228357900, 228357903], '(00)3',
[228386399, 228386402], '(00)3',
[228907446, 228907449], '(00)3',
[228984552, 228984555], '3(00)',
[229140285, 229140288], '3(00)',
[231810024, 231810027], '(00)3',
[232838062, 232838065], '3(00)',
[234389088, 234389091], '3(00)',
[235588194, 235588197], '(00)3',
[236645695, 236645698], '(00)3',
[236962876, 236962879], '3(00)',
[237516723, 237516727], '04(00)',
[240004911, 240004914], '(00)3',
[240221306, 240221309], '3(00)',
[241389213, 241389217], '(010)3',
[241549003, 241549006], '(00)3',
[241729717, 241729720], '(00)3',
[241743684, 241743687], '3(00)',
[243780200, 243780203], '3(00)',
[243801317, 243801320], '(00)3',
[244122072, 244122075], '(00)3',
[244691224, 244691227], '3(00)',
[244841577, 244841580], '(00)3',
[245813461, 245813464], '(00)3',
[246299475, 246299478], '(00)3',
[246450176, 246450179], '3(00)',
[249069349, 249069352], '(00)3',
[250076378, 250076381], '(00)3',
[252442157, 252442160], '3(00)',
[252904231, 252904234], '3(00)',
[255145220, 255145223], '(00)3',
[255285971, 255285974], '3(00)',
[256713230, 256713233], '(00)3',
[257992082, 257992085], '(00)3',
[258447955, 258447959], '22(00)',
[259298045, 259298048], '3(00)',
[262141503, 262141506], '(00)3',
[263681743, 263681746], '3(00)',
[266527881, 266527885], '(010)3',
[266617122, 266617125], '(00)3',
[266628044, 266628047], '3(00)',
[267305763, 267305766], '(00)3',
[267388404, 267388407], '3(00)',
[267441672, 267441675], '3(00)',
[267464886, 267464889], '(00)3',
[267554907, 267554910], '3(00)',
[269787480, 269787483], '(00)3',
[270881434, 270881437], '(00)3',
[270997583, 270997586], '3(00)',
[272096378, 272096381], '3(00)',
[272583009, 272583012], '(00)3',
[274190881, 274190884], '3(00)',
[274268747, 274268750], '(00)3',
[275297429, 275297432], '3(00)',
[275545476, 275545479], '3(00)',
[275898479, 275898482], '3(00)',
[275953000, 275953003], '(00)3',
[277117197, 277117201], '(00)22',
[277447310, 277447313], '3(00)',
[279059657, 279059660], '3(00)',
[279259144, 279259147], '3(00)',
[279513636, 279513639], '3(00)',
[279849069, 279849072], '3(00)',
[280291419, 280291422], '(00)3',
[281449425, 281449428], '3(00)',
[281507953, 281507956], '3(00)',
[281825600, 281825603], '(00)3',
[282547093, 282547096], '3(00)',
[283120963, 283120966], '3(00)',
[283323493, 283323496], '(00)3',
[284764535, 284764538], '3(00)',
[286172639, 286172642], '3(00)',
[286688824, 286688827], '(00)3',
[287222172, 287222175], '3(00)',
[287235534, 287235537], '3(00)',
[287304861, 287304864], '3(00)',
[287433571, 287433574], '(00)3',
[287823551, 287823554], '(00)3',
[287872422, 287872425], '3(00)',
[288766615, 288766618], '3(00)',
[290122963, 290122966], '3(00)',
[290450849, 290450853], '(00)22',
[291426141, 291426144], '3(00)',
[292810353, 292810356], '3(00)',
[293109861, 293109864], '3(00)',
[293398054, 293398057], '3(00)',
[294134426, 294134429], '3(00)',
[294216438, 294216441], '(00)3',
[295367141, 295367144], '3(00)',
[297834111, 297834114], '3(00)',
[299099969, 299099972], '3(00)',
[300746958, 300746961], '3(00)',
[301097423, 301097426], '(00)3',
[301834209, 301834212], '(00)3',
[302554791, 302554794], '(00)3',
[303497445, 303497448], '3(00)',
[304165344, 304165347], '3(00)',
[304790218, 304790222], '3(010)',
[305302352, 305302355], '(00)3',
[306785996, 306785999], '3(00)',
[307051443, 307051446], '3(00)',
[307481539, 307481542], '3(00)',
[308605569, 308605572], '3(00)',
[309237610, 309237613], '3(00)',
[310509287, 310509290], '(00)3',
[310554057, 310554060], '3(00)',
[310646345, 310646348], '3(00)',
[311274896, 311274899], '(00)3',
[311894272, 311894275], '3(00)',
[312269470, 312269473], '(00)3',
[312306601, 312306605], '(00)40',
[312683193, 312683196], '3(00)',
[314499804, 314499807], '3(00)',
[314636802, 314636805], '(00)3',
[314689897, 314689900], '3(00)',
[314721319, 314721322], '3(00)',
[316132890, 316132893], '3(00)',
[316217470, 316217474], '(010)3',
[316465705, 316465708], '3(00)',
[316542790, 316542793], '(00)3',
[320822347, 320822350], '3(00)',
[321733242, 321733245], '3(00)',
[324413970, 324413973], '(00)3',
[325950140, 325950143], '(00)3',
[326675884, 326675887], '(00)3',
[326704208, 326704211], '3(00)',
[327596247, 327596250], '3(00)',
[328123172, 328123175], '3(00)',
[328182212, 328182215], '(00)3',
[328257498, 328257501], '3(00)',
[328315836, 328315839], '(00)3',
[328800974, 328800977], '(00)3',
[328998509, 328998512], '3(00)',
[329725370, 329725373], '(00)3',
[332080601, 332080604], '(00)3',
[332221246, 332221249], '(00)3',
[332299899, 332299902], '(00)3',
[332532822, 332532825], '(00)3',
[333334544, 333334548], '(00)22',
[333881266, 333881269], '3(00)',
[334703267, 334703270], '3(00)',
[334875138, 334875141], '3(00)',
[336531451, 336531454], '3(00)',
[336825907, 336825910], '(00)3',
[336993167, 336993170], '(00)3',
[337493998, 337494001], '3(00)',
[337861034, 337861037], '3(00)',
[337899191, 337899194], '(00)3',
[337958123, 337958126], '(00)3',
[342331982, 342331985], '3(00)',
[342676068, 342676071], '3(00)',
[347063781, 347063784], '3(00)',
[347697348, 347697351], '3(00)',
[347954319, 347954322], '3(00)',
[348162775, 348162778], '3(00)',
[349210702, 349210705], '(00)3',
[349212913, 349212916], '3(00)',
[349248650, 349248653], '(00)3',
[349913500, 349913503], '3(00)',
[350891529, 350891532], '3(00)',
[351089323, 351089326], '3(00)',
[351826158, 351826161], '3(00)',
[352228580, 352228583], '(00)3',
[352376244, 352376247], '3(00)',
[352853758, 352853761], '(00)3',
[355110439, 355110442], '(00)3',
[355808090, 355808094], '(00)40',
[355941556, 355941559], '3(00)',
[356360231, 356360234], '(00)3',
[356586657, 356586660], '3(00)',
[356892926, 356892929], '(00)3',
[356908232, 356908235], '3(00)',
[357912730, 357912733], '3(00)',
[358120344, 358120347], '3(00)',
[359044096, 359044099], '(00)3',
[360819357, 360819360], '3(00)',
[361399662, 361399666], '(010)3',
[362361315, 362361318], '(00)3',
[363610112, 363610115], '(00)3',
[363964804, 363964807], '3(00)',
[364527375, 364527378], '(00)3',
[365090327, 365090330], '(00)3',
[365414539, 365414542], '3(00)',
[366738474, 366738477], '3(00)',
[368714778, 368714783], '04(010)',
[368831545, 368831548], '(00)3',
[368902387, 368902390], '(00)3',
[370109769, 370109772], '3(00)',
[370963333, 370963336], '3(00)',
[372541136, 372541140], '3(010)',
[372681562, 372681565], '(00)3',
[373009410, 373009413], '(00)3',
[373458970, 373458973], '3(00)',
[375648658, 375648661], '3(00)',
[376834728, 376834731], '3(00)',
[377119945, 377119948], '(00)3',
[377335703, 377335706], '(00)3',
[378091745, 378091748], '3(00)',
[379139522, 379139525], '3(00)',
[380279160, 380279163], '(00)3',
[380619442, 380619445], '3(00)',
[381244231, 381244234], '3(00)',
[382327446, 382327450], '(010)3',
[382357073, 382357076], '3(00)',
[383545479, 383545482], '3(00)',
[384363766, 384363769], '(00)3',
[384401786, 384401790], '22(00)',
[385198212, 385198215], '3(00)',
[385824476, 385824479], '(00)3',
[385908194, 385908197], '3(00)',
[386946806, 386946809], '3(00)',
[387592175, 387592179], '22(00)',
[388329293, 388329296], '(00)3',
[388679566, 388679569], '3(00)',
[388832142, 388832145], '3(00)',
[390087103, 390087106], '(00)3',
[390190926, 390190930], '(00)22',
[390331207, 390331210], '3(00)',
[391674495, 391674498], '3(00)',
[391937831, 391937834], '3(00)',
[391951632, 391951636], '(00)22',
[392963986, 392963989], '(00)3',
[393007921, 393007924], '3(00)',
[393373210, 393373213], '3(00)',
[393759572, 393759575], '(00)3',
[394036662, 394036665], '(00)3',
[395813866, 395813869], '(00)3',
[395956690, 395956693], '3(00)',
[396031670, 396031673], '3(00)',
[397076433, 397076436], '3(00)',
[397470601, 397470604], '3(00)',
[398289458, 398289461], '3(00)',
#
[368714778, 368714783], '04(010)',
[437953499, 437953504], '04(010)',
[526196233, 526196238], '032(00)',
[744719566, 744719571], '(010)40',
[750375857, 750375862], '032(00)',
[958241932, 958241937], '04(010)',
[983377342, 983377347], '(00)410',
[1003780080, 1003780085], '04(010)',
[1070232754, 1070232759], '(00)230',
[1209834865, 1209834870], '032(00)',
[1257209100, 1257209105], '(00)410',
[1368002233, 1368002238], '(00)230'
]
