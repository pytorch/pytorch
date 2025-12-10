"""Heuristic polynomial GCD algorithm (HEUGCD). """

from .polyerrors import HeuristicGCDFailed

HEU_GCD_MAX = 6

def heugcd(f, g):
    """
    Heuristic polynomial GCD in ``Z[X]``.

    Given univariate polynomials ``f`` and ``g`` in ``Z[X]``, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    ``f`` and ``g`` at certain points and computing (fast) integer GCD
    of those evaluations. The polynomial GCD is recovered from the integer
    image by interpolation. The evaluation process reduces f and g variable
    by variable into a large integer. The final step is to verify if the
    interpolated polynomial is the correct GCD. This gives cofactors of
    the input polynomials as a side effect.

    Examples
    ========

    >>> from sympy.polys.heuristicgcd import heugcd
    >>> from sympy.polys import ring, ZZ

    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> h, cff, cfg = heugcd(f, g)
    >>> h, cff, cfg
    (x + y, x + y, x)

    >>> cff*h == f
    True
    >>> cfg*h == g
    True

    References
    ==========

    .. [1] [Liao95]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    ring = f.ring
    x0 = ring.gens[0]
    domain = ring.domain

    gcd, f, g = f.extract_ground(g)

    f_norm = f.max_norm()
    g_norm = g.max_norm()

    B = domain(2*min(f_norm, g_norm) + 29)

    x = max(min(B, 99*domain.sqrt(B)),
            2*min(f_norm // abs(f.LC),
                  g_norm // abs(g.LC)) + 4)

    for i in range(0, HEU_GCD_MAX):
        ff = f.evaluate(x0, x)
        gg = g.evaluate(x0, x)

        if ff and gg:
            if ring.ngens == 1:
                h, cff, cfg = domain.cofactors(ff, gg)
            else:
                h, cff, cfg = heugcd(ff, gg)

            h = _gcd_interpolate(h, x, ring)
            h = h.primitive()[1]

            cff_, r = f.div(h)

            if not r:
                cfg_, r = g.div(h)

                if not r:
                    h = h.mul_ground(gcd)
                    return h, cff_, cfg_

            cff = _gcd_interpolate(cff, x, ring)

            h, r = f.div(cff)

            if not r:
                cfg_, r = g.div(h)

                if not r:
                    h = h.mul_ground(gcd)
                    return h, cff, cfg_

            cfg = _gcd_interpolate(cfg, x, ring)

            h, r = g.div(cfg)

            if not r:
                cff_, r = f.div(h)

                if not r:
                    h = h.mul_ground(gcd)
                    return h, cff_, cfg

        x = 73794*x * domain.sqrt(domain.sqrt(x)) // 27011

    raise HeuristicGCDFailed('no luck')

def _gcd_interpolate(h, x, ring):
    """Interpolate polynomial GCD from integer GCD. """
    f, i = ring.zero, 0

    # TODO: don't expose poly repr implementation details
    if ring.ngens == 1:
        while h:
            g = h % x
            if g > x // 2: g -= x
            h = (h - g) // x

            # f += X**i*g
            if g:
                f[(i,)] = g
            i += 1
    else:
        while h:
            g = h.trunc_ground(x)
            h = (h - g).quo_ground(x)

            # f += X**i*g
            if g:
                for monom, coeff in g.iterterms():
                    f[(i,) + monom] = coeff
            i += 1

    if f.LC < 0:
        return -f
    else:
        return  f
