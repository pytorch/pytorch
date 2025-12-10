#
# This module is deprecated and should not be used any more. The actual
# implementation of RR and CC now uses mpmath's mpf and mpc types directly.
#
"""Real and complex elements. """


from sympy.external.gmpy import MPQ
from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public

from mpmath.ctx_mp_python import PythonMPContext, _mpf, _mpc, _constant
from mpmath.libmp import (MPZ_ONE, fzero, fone, finf, fninf, fnan,
    round_nearest, mpf_mul, repr_dps, int_types,
    from_int, from_float, from_str, to_rational)


@public
class RealElement(_mpf, DomainElement):
    """An element of a real domain. """

    __slots__ = ('__mpf__',)

    def _set_mpf(self, val):
        self.__mpf__ = val

    _mpf_ = property(lambda self: self.__mpf__, _set_mpf)

    def parent(self):
        return self.context._parent

@public
class ComplexElement(_mpc, DomainElement):
    """An element of a complex domain. """

    __slots__ = ('__mpc__',)

    def _set_mpc(self, val):
        self.__mpc__ = val

    _mpc_ = property(lambda self: self.__mpc__, _set_mpc)

    def parent(self):
        return self.context._parent

new = object.__new__

@public
class MPContext(PythonMPContext):

    def __init__(ctx, prec=53, dps=None, tol=None, real=False):
        ctx._prec_rounding = [prec, round_nearest]

        if dps is None:
            ctx._set_prec(prec)
        else:
            ctx._set_dps(dps)

        ctx.mpf = RealElement
        ctx.mpc = ComplexElement
        ctx.mpf._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.mpc._ctxdata = [ctx.mpc, new, ctx._prec_rounding]

        if real:
            ctx.mpf.context = ctx
        else:
            ctx.mpc.context = ctx

        ctx.constant = _constant
        ctx.constant._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.constant.context = ctx

        ctx.types = [ctx.mpf, ctx.mpc, ctx.constant]
        ctx.trap_complex = True
        ctx.pretty = True

        if tol is None:
            ctx.tol = ctx._make_tol()
        elif tol is False:
            ctx.tol = fzero
        else:
            ctx.tol = ctx._convert_tol(tol)

        ctx.tolerance = ctx.make_mpf(ctx.tol)

        if not ctx.tolerance:
            ctx.max_denom = 1000000
        else:
            ctx.max_denom = int(1/ctx.tolerance)

        ctx.zero = ctx.make_mpf(fzero)
        ctx.one = ctx.make_mpf(fone)
        ctx.j = ctx.make_mpc((fzero, fone))
        ctx.inf = ctx.make_mpf(finf)
        ctx.ninf = ctx.make_mpf(fninf)
        ctx.nan = ctx.make_mpf(fnan)

    def _make_tol(ctx):
        hundred = (0, 25, 2, 5)
        eps = (0, MPZ_ONE, 1-ctx.prec, 1)
        return mpf_mul(hundred, eps)

    def make_tol(ctx):
        return ctx.make_mpf(ctx._make_tol())

    def _convert_tol(ctx, tol):
        if isinstance(tol, int_types):
            return from_int(tol)
        if isinstance(tol, float):
            return from_float(tol)
        if hasattr(tol, "_mpf_"):
            return tol._mpf_
        prec, rounding = ctx._prec_rounding
        if isinstance(tol, str):
            return from_str(tol, prec, rounding)
        raise ValueError("expected a real number, got %s" % tol)

    def _convert_fallback(ctx, x, strings):
        raise TypeError("cannot create mpf from " + repr(x))

    @property
    def _repr_digits(ctx):
        return repr_dps(ctx._prec)

    @property
    def _str_digits(ctx):
        return ctx._dps

    def to_rational(ctx, s, limit=True):
        p, q = to_rational(s._mpf_)

        # Needed for GROUND_TYPES=flint if gmpy2 is installed because mpmath's
        # to_rational() function returns a gmpy2.mpz instance and if MPQ is
        # flint.fmpq then MPQ(p, q) will fail.
        p = int(p)

        if not limit or q <= ctx.max_denom:
            return p, q

        p0, q0, p1, q1 = 0, 1, 1, 0
        n, d = p, q

        while True:
            a = n//d
            q2 = q0 + a*q1
            if q2 > ctx.max_denom:
                break
            p0, q0, p1, q1 = p1, q1, p0 + a*p1, q2
            n, d = d, n - a*d

        k = (ctx.max_denom - q0)//q1

        number = MPQ(p, q)
        bound1 = MPQ(p0 + k*p1, q0 + k*q1)
        bound2 = MPQ(p1, q1)

        if not bound2 or not bound1:
            return p, q
        elif abs(bound2 - number) <= abs(bound1 - number):
            return bound2.numerator, bound2.denominator
        else:
            return bound1.numerator, bound1.denominator

    def almosteq(ctx, s, t, rel_eps=None, abs_eps=None):
        t = ctx.convert(t)
        if abs_eps is None and rel_eps is None:
            rel_eps = abs_eps = ctx.tolerance or ctx.make_tol()
        if abs_eps is None:
            abs_eps = ctx.convert(rel_eps)
        elif rel_eps is None:
            rel_eps = ctx.convert(abs_eps)
        diff = abs(s-t)
        if diff <= abs_eps:
            return True
        abss = abs(s)
        abst = abs(t)
        if abss < abst:
            err = diff/abst
        else:
            err = diff/abss
        return err <= rel_eps
