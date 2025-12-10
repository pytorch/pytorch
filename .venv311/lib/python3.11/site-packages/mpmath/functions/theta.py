from .functions import defun, defun_wrapped

@defun
def _jacobi_theta2(ctx, z, q):
    extra1 = 10
    extra2 = 20
    # the loops below break when the fixed precision quantities
    # a and b go to zero;
    # right shifting small negative numbers by wp one obtains -1, not zero,
    # so the condition a**2 + b**2 > MIN is used to break the loops.
    MIN = 2
    if z == ctx.zero:
        if (not ctx._im(q)):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = (x*x) >> wp
            a = b = x2
            s = x2
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                s += a
            s = (1 << (wp+1)) + (s << 1)
            s = ctx.ldexp(s, -wp)
        else:
            wp = ctx.prec + extra1
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp-1)
            are = bre = x2re
            aim = bim = x2im
            sre = (1<<wp) + are
            sim = aim
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                sre += are
                sim += aim
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
    else:
        if (not ctx._im(q)) and (not ctx._im(z)):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = (x*x) >> wp
            a = b = x2
            c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
            cn = c1 = ctx.to_fixed(c1, wp)
            sn = s1 = ctx.to_fixed(s1, wp)
            c2 = (c1*c1 - s1*s1) >> wp
            s2 = (c1 * s1) >> (wp - 1)
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            s = c1 + ((a * cn) >> wp)
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
                s += (a * cn) >> wp
            s = (s << 1)
            s = ctx.ldexp(s, -wp)
            s *= ctx.nthroot(q, 4)
            return s
        # case z real, q complex
        elif not ctx._im(z):
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = x2re
            aim = bim = x2im
            c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
            cn = c1 = ctx.to_fixed(c1, wp)
            sn = s1 = ctx.to_fixed(s1, wp)
            c2 = (c1*c1 - s1*s1) >> wp
            s2 = (c1 * s1) >> (wp - 1)
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            sre = c1 + ((are * cn) >> wp)
            sim = ((aim * cn) >> wp)
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
                sre += ((are * cn) >> wp)
                sim += ((aim * cn) >> wp)
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
        #case z complex, q real
        elif not ctx._im(q):
            wp = ctx.prec + extra2
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = (x*x) >> wp
            a = b = x2
            prec0 = ctx.prec
            ctx.prec = wp
            c1, s1 = ctx.cos_sin(z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            #c2 = (c1*c1 - s1*s1) >> wp
            c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
            c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
            #s2 = (c1 * s1) >> (wp - 1)
            s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
            s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
            #cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            sre = c1re + ((a * cnre) >> wp)
            sim = c1im + ((a * cnim) >> wp)
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
                t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
                t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
                t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                sre += ((a * cnre) >> wp)
                sim += ((a * cnim) >> wp)
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
        # case z and q complex
        else:
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = x2re
            aim = bim = x2im
            prec0 = ctx.prec
            ctx.prec = wp
            # cos(z), sin(z) with z complex
            c1, s1 = ctx.cos_sin(z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
            c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
            s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
            s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            n = 1
            termre = c1re
            termim = c1im
            sre = c1re + ((are * cnre - aim * cnim) >> wp)
            sim = c1im + ((are * cnim + aim * cnre) >> wp)
            n = 3
            termre = ((are * cnre - aim * cnim) >> wp)
            termim = ((are * cnim + aim * cnre) >> wp)
            sre = c1re + ((are * cnre - aim * cnim) >> wp)
            sim = c1im + ((are * cnim + aim * cnre) >> wp)
            n = 5
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                #cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
                t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
                t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
                t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
                t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                termre = ((are * cnre - aim * cnim) >> wp)
                termim = ((aim * cnre + are * cnim) >> wp)
                sre += ((are * cnre - aim * cnim) >> wp)
                sim += ((aim * cnre + are * cnim) >> wp)
                n += 2
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
    s *= ctx.nthroot(q, 4)
    return s

@defun
def _djacobi_theta2(ctx, z, q, nd):
    MIN = 2
    extra1 = 10
    extra2 = 20
    if (not ctx._im(q)) and (not ctx._im(z)):
        wp = ctx.prec + extra1
        x = ctx.to_fixed(ctx._re(q), wp)
        x2 = (x*x) >> wp
        a = b = x2
        c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
        cn = c1 = ctx.to_fixed(c1, wp)
        sn = s1 = ctx.to_fixed(s1, wp)
        c2 = (c1*c1 - s1*s1) >> wp
        s2 = (c1 * s1) >> (wp - 1)
        cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
        if (nd&1):
            s = s1 + ((a * sn * 3**nd) >> wp)
        else:
            s = c1 + ((a * cn * 3**nd) >> wp)
        n = 2
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            if nd&1:
                s += (a * sn * (2*n+1)**nd) >> wp
            else:
                s += (a * cn * (2*n+1)**nd) >> wp
            n += 1
        s = -(s << 1)
        s = ctx.ldexp(s, -wp)
        # case z real, q complex
    elif not ctx._im(z):
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = x2re
        aim = bim = x2im
        c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
        cn = c1 = ctx.to_fixed(c1, wp)
        sn = s1 = ctx.to_fixed(s1, wp)
        c2 = (c1*c1 - s1*s1) >> wp
        s2 = (c1 * s1) >> (wp - 1)
        cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
        if (nd&1):
            sre = s1 + ((are * sn * 3**nd) >> wp)
            sim = ((aim * sn * 3**nd) >> wp)
        else:
            sre = c1 + ((are * cn * 3**nd) >> wp)
            sim = ((aim * cn * 3**nd) >> wp)
        n = 5
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp

            if (nd&1):
                sre += ((are * sn * n**nd) >> wp)
                sim += ((aim * sn * n**nd) >> wp)
            else:
                sre += ((are * cn * n**nd) >> wp)
                sim += ((aim * cn * n**nd) >> wp)
            n += 2
        sre = -(sre << 1)
        sim = -(sim << 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    #case z complex, q real
    elif not ctx._im(q):
        wp = ctx.prec + extra2
        x = ctx.to_fixed(ctx._re(q), wp)
        x2 = (x*x) >> wp
        a = b = x2
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        #c2 = (c1*c1 - s1*s1) >> wp
        c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
        c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
        #s2 = (c1 * s1) >> (wp - 1)
        s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
        s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
        #cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
        t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
        t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
        t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
        t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
        cnre = t1
        cnim = t2
        snre = t3
        snim = t4
        if (nd&1):
            sre = s1re + ((a * snre * 3**nd) >> wp)
            sim = s1im + ((a * snim * 3**nd) >> wp)
        else:
            sre = c1re + ((a * cnre * 3**nd) >> wp)
            sim = c1im + ((a * cnim * 3**nd) >> wp)
        n = 5
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if (nd&1):
                sre += ((a * snre * n**nd) >> wp)
                sim += ((a * snim * n**nd) >> wp)
            else:
                sre += ((a * cnre * n**nd) >> wp)
                sim += ((a * cnim * n**nd) >> wp)
            n += 2
        sre = -(sre << 1)
        sim = -(sim << 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    # case z and q complex
    else:
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = x2re
        aim = bim = x2im
        prec0 = ctx.prec
        ctx.prec = wp
        # cos(2*z), sin(2*z) with z complex
        c1, s1 = ctx.cos_sin(z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
        c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
        s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
        s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
        t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
        t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
        t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
        t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
        cnre = t1
        cnim = t2
        snre = t3
        snim = t4
        if (nd&1):
            sre = s1re + (((are * snre - aim * snim) * 3**nd) >> wp)
            sim = s1im + (((are * snim + aim * snre)* 3**nd) >> wp)
        else:
            sre = c1re + (((are * cnre - aim * cnim) * 3**nd) >> wp)
            sim = c1im + (((are * cnim + aim * cnre)* 3**nd) >> wp)
        n = 5
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            #cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if (nd&1):
                sre += (((are * snre - aim * snim) * n**nd) >> wp)
                sim += (((aim * snre + are * snim) * n**nd) >> wp)
            else:
                sre += (((are * cnre - aim * cnim) * n**nd) >> wp)
                sim += (((aim * cnre + are * cnim) * n**nd) >> wp)
            n += 2
        sre = -(sre << 1)
        sim = -(sim << 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    s *= ctx.nthroot(q, 4)
    if (nd&1):
        return (-1)**(nd//2) * s
    else:
        return (-1)**(1 + nd//2) * s

@defun
def _jacobi_theta3(ctx, z, q):
    extra1 = 10
    extra2 = 20
    MIN = 2
    if z == ctx.zero:
        if not ctx._im(q):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            s = x
            a = b = x
            x2 = (x*x) >> wp
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                s += a
            s = (1 << wp) + (s << 1)
            s = ctx.ldexp(s, -wp)
            return s
        else:
            wp = ctx.prec + extra1
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            sre = are = bre = xre
            sim = aim = bim = xim
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                sre += are
                sim += aim
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s
    else:
        if (not ctx._im(q)) and (not ctx._im(z)):
            s = 0
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            a = b = x
            x2 = (x*x) >> wp
            c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
            c1 = ctx.to_fixed(c1, wp)
            s1 = ctx.to_fixed(s1, wp)
            cn = c1
            sn = s1
            s += (a * cn) >> wp
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
                s += (a * cn) >> wp
            s = (1 << wp) + (s << 1)
            s = ctx.ldexp(s, -wp)
            return s
        # case z real, q complex
        elif not ctx._im(z):
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = xre
            aim = bim = xim
            c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
            c1 = ctx.to_fixed(c1, wp)
            s1 = ctx.to_fixed(s1, wp)
            cn = c1
            sn = s1
            sre = (are * cn) >> wp
            sim = (aim * cn) >> wp
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
                sre += (are * cn) >> wp
                sim += (aim * cn) >> wp
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s
        #case z complex, q real
        elif not ctx._im(q):
            wp = ctx.prec + extra2
            x = ctx.to_fixed(ctx._re(q), wp)
            a = b = x
            x2 = (x*x) >> wp
            prec0 = ctx.prec
            ctx.prec = wp
            c1, s1 = ctx.cos_sin(2*z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            sre = (a * cnre) >> wp
            sim = (a * cnim) >> wp
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
                t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
                t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
                t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                sre += (a * cnre) >> wp
                sim += (a * cnim) >> wp
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s
        # case z and q complex
        else:
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = xre
            aim = bim = xim
            prec0 = ctx.prec
            ctx.prec = wp
            # cos(2*z), sin(2*z) with z complex
            c1, s1 = ctx.cos_sin(2*z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            sre = (are * cnre - aim * cnim) >> wp
            sim = (aim * cnre + are * cnim) >> wp
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
                t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
                t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
                t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                sre += (are * cnre - aim * cnim) >> wp
                sim += (aim * cnre + are * cnim) >> wp
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s

@defun
def _djacobi_theta3(ctx, z, q, nd):
    """nd=1,2,3 order of the derivative with respect to z"""
    MIN = 2
    extra1 = 10
    extra2 = 20
    if (not ctx._im(q)) and (not ctx._im(z)):
        s = 0
        wp = ctx.prec + extra1
        x = ctx.to_fixed(ctx._re(q), wp)
        a = b = x
        x2 = (x*x) >> wp
        c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
        c1 = ctx.to_fixed(c1, wp)
        s1 = ctx.to_fixed(s1, wp)
        cn = c1
        sn = s1
        if (nd&1):
            s += (a * sn) >> wp
        else:
            s += (a * cn) >> wp
        n = 2
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
            if nd&1:
                s += (a * sn * n**nd) >> wp
            else:
                s += (a * cn * n**nd) >> wp
            n += 1
        s = -(s << (nd+1))
        s = ctx.ldexp(s, -wp)
    # case z real, q complex
    elif not ctx._im(z):
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = xre
        aim = bim = xim
        c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
        c1 = ctx.to_fixed(c1, wp)
        s1 = ctx.to_fixed(s1, wp)
        cn = c1
        sn = s1
        if (nd&1):
            sre = (are * sn) >> wp
            sim = (aim * sn) >> wp
        else:
            sre = (are * cn) >> wp
            sim = (aim * cn) >> wp
        n = 2
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
            if nd&1:
                sre += (are * sn * n**nd) >> wp
                sim += (aim * sn * n**nd) >> wp
            else:
                sre += (are * cn * n**nd) >> wp
                sim += (aim * cn * n**nd) >> wp
            n += 1
        sre = -(sre << (nd+1))
        sim = -(sim << (nd+1))
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    #case z complex, q real
    elif not ctx._im(q):
        wp = ctx.prec + extra2
        x = ctx.to_fixed(ctx._re(q), wp)
        a = b = x
        x2 = (x*x) >> wp
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(2*z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        if (nd&1):
            sre = (a * snre) >> wp
            sim = (a * snim) >> wp
        else:
            sre = (a * cnre) >> wp
            sim = (a * cnim) >> wp
        n = 2
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
            t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
            t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
            t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if (nd&1):
                sre += (a * snre * n**nd) >> wp
                sim += (a * snim * n**nd) >> wp
            else:
                sre += (a * cnre * n**nd) >> wp
                sim += (a * cnim * n**nd) >> wp
            n += 1
        sre = -(sre << (nd+1))
        sim = -(sim << (nd+1))
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    # case z and q complex
    else:
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = xre
        aim = bim = xim
        prec0 = ctx.prec
        ctx.prec = wp
        # cos(2*z), sin(2*z) with z complex
        c1, s1 = ctx.cos_sin(2*z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        if (nd&1):
            sre = (are * snre - aim * snim) >> wp
            sim = (aim * snre + are * snim) >> wp
        else:
            sre = (are * cnre - aim * cnim) >> wp
            sim = (aim * cnre + are * cnim) >> wp
        n = 2
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
            t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
            t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
            t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if(nd&1):
                sre += ((are * snre - aim * snim) * n**nd) >> wp
                sim += ((aim * snre + are * snim) * n**nd) >> wp
            else:
                sre += ((are * cnre - aim * cnim) * n**nd) >> wp
                sim += ((aim * cnre + are * cnim) * n**nd) >> wp
            n += 1
        sre = -(sre << (nd+1))
        sim = -(sim << (nd+1))
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    if (nd&1):
        return (-1)**(nd//2) * s
    else:
        return (-1)**(1 + nd//2) * s

@defun
def _jacobi_theta2a(ctx, z, q):
    """
    case ctx._im(z) != 0
    theta(2, z, q) =
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n=-inf, inf)
    max term for minimum (2*n+1)*log(q).real - 2* ctx._im(z)
    n0 = int(ctx._im(z)/log(q).real - 1/2)
    theta(2, z, q) =
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n=n0, inf) +
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n, n0-1, -inf)
    """
    n = n0 = int(ctx._im(z)/ctx._re(ctx.log(q)) - 1/2)
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj((2*n+1)*z)
    a = q**(n*n + n)
    # leading term
    term = a * e
    s = term
    eps1 = ctx.eps*abs(term)
    while 1:
        n += 1
        e = e * e2
        term = q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    s = s * ctx.nthroot(q, 4)
    return s

@defun
def _jacobi_theta3a(ctx, z, q):
    """
    case ctx._im(z) != 0
    theta3(z, q) = Sum(q**(n*n) * exp(j*2*n*z), n, -inf, inf)
    max term for n*abs(log(q).real) + ctx._im(z) ~= 0
    n0 = int(- ctx._im(z)/abs(log(q).real))
    """
    n = n0 = int(-ctx._im(z)/abs(ctx._re(ctx.log(q))))
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj(2*n*z)
    s = term = q**(n*n) * e
    eps1 = ctx.eps*abs(term)
    while 1:
        n += 1
        e = e * e2
        term = q**(n*n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = q**(n*n) * e
        if abs(term) < eps1:
            break
        s += term
    return s

@defun
def _djacobi_theta2a(ctx, z, q, nd):
    """
    case ctx._im(z) != 0
    dtheta(2, z, q, nd) =
    j* q**1/4 * Sum(q**(n*n + n) * (2*n+1)*exp(j*(2*n + 1)*z), n=-inf, inf)
    max term for (2*n0+1)*log(q).real - 2* ctx._im(z) ~= 0
    n0 = int(ctx._im(z)/log(q).real - 1/2)
    """
    n = n0 = int(ctx._im(z)/ctx._re(ctx.log(q)) - 1/2)
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj((2*n + 1)*z)
    a = q**(n*n + n)
    # leading term
    term = (2*n+1)**nd * a * e
    s = term
    eps1 = ctx.eps*abs(term)
    while 1:
        n += 1
        e = e * e2
        term = (2*n+1)**nd * q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = (2*n+1)**nd * q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    return ctx.j**nd * s * ctx.nthroot(q, 4)

@defun
def _djacobi_theta3a(ctx, z, q, nd):
    """
    case ctx._im(z) != 0
    djtheta3(z, q, nd) = (2*j)**nd *
      Sum(q**(n*n) * n**nd * exp(j*2*n*z), n, -inf, inf)
    max term for minimum n*abs(log(q).real) + ctx._im(z)
    """
    n = n0 = int(-ctx._im(z)/abs(ctx._re(ctx.log(q))))
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj(2*n*z)
    a = q**(n*n) * e
    s = term = n**nd * a
    if n != 0:
        eps1 = ctx.eps*abs(term)
    else:
        eps1 = ctx.eps*abs(a)
    while 1:
        n += 1
        e = e * e2
        a = q**(n*n) * e
        term = n**nd * a
        if n != 0:
            aterm = abs(term)
        else:
            aterm = abs(a)
        if aterm < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        a = q**(n*n) * e
        term = n**nd * a
        if n != 0:
            aterm = abs(term)
        else:
            aterm = abs(a)
        if aterm < eps1:
            break
        s += term
    return (2*ctx.j)**nd * s

@defun
def jtheta(ctx, n, z, q, derivative=0):
    if derivative:
        return ctx._djtheta(n, z, q, derivative)

    z = ctx.convert(z)
    q = ctx.convert(q)

    # Implementation note
    # If ctx._im(z) is close to zero, _jacobi_theta2 and _jacobi_theta3
    # are used,
    # which compute the series starting from n=0 using fixed precision
    # numbers;
    # otherwise  _jacobi_theta2a and _jacobi_theta3a are used, which compute
    # the series starting from n=n0, which is the largest term.

    # TODO: write _jacobi_theta2a and _jacobi_theta3a using fixed-point

    if abs(q) > ctx.THETA_Q_LIM:
        raise ValueError('abs(q) > THETA_Q_LIM = %f' % ctx.THETA_Q_LIM)

    extra = 10
    if z:
        M = ctx.mag(z)
        if M > 5 or (n == 1 and M < -5):
            extra += 2*abs(M)
    cz = 0.5
    extra2 = 50
    prec0 = ctx.prec
    try:
        ctx.prec += extra
        if n == 1:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta2(z - ctx.pi/2, q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta2a(z - ctx.pi/2, q)
            else:
                res = ctx._jacobi_theta2(z - ctx.pi/2, q)
        elif n == 2:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta2(z, q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta2a(z, q)
            else:
                res = ctx._jacobi_theta2(z, q)
        elif n == 3:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta3(z, q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta3a(z, q)
            else:
                res = ctx._jacobi_theta3(z, q)
        elif n == 4:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta3(z, -q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta3a(z, -q)
            else:
                res = ctx._jacobi_theta3(z, -q)
        else:
            raise ValueError
    finally:
        ctx.prec = prec0
    return res

@defun
def _djtheta(ctx, n, z, q, derivative=1):
    z = ctx.convert(z)
    q = ctx.convert(q)
    nd = int(derivative)

    if abs(q) > ctx.THETA_Q_LIM:
        raise ValueError('abs(q) > THETA_Q_LIM = %f' % ctx.THETA_Q_LIM)
    extra = 10 + ctx.prec * nd // 10
    if z:
        M = ctx.mag(z)
        if M > 5 or (n != 1 and M < -5):
            extra += 2*abs(M)
    cz = 0.5
    extra2 = 50
    prec0 = ctx.prec
    try:
        ctx.prec += extra
        if n == 1:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta2(z - ctx.pi/2, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta2a(z - ctx.pi/2, q, nd)
            else:
                res = ctx._djacobi_theta2(z - ctx.pi/2, q, nd)
        elif n == 2:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta2(z, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta2a(z, q, nd)
            else:
                res = ctx._djacobi_theta2(z, q, nd)
        elif n == 3:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta3(z, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta3a(z, q, nd)
            else:
                res = ctx._djacobi_theta3(z, q, nd)
        elif n == 4:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta3(z, -q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta3a(z, -q, nd)
            else:
                res = ctx._djacobi_theta3(z, -q, nd)
        else:
            raise ValueError
    finally:
        ctx.prec = prec0
    return +res
