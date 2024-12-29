import operator

from . import libmp

from .libmp.backend import basestring

from .libmp import (
    int_types, MPZ_ONE,
    prec_to_dps, dps_to_prec, repr_dps,
    round_floor, round_ceiling,
    fzero, finf, fninf, fnan,
    mpf_le, mpf_neg,
    from_int, from_float, from_str, from_rational,
    mpi_mid, mpi_delta, mpi_str,
    mpi_abs, mpi_pos, mpi_neg, mpi_add, mpi_sub,
    mpi_mul, mpi_div, mpi_pow_int, mpi_pow,
    mpi_from_str,
    mpci_pos, mpci_neg, mpci_add, mpci_sub, mpci_mul, mpci_div, mpci_pow,
    mpci_abs, mpci_pow, mpci_exp, mpci_log,
    ComplexResult,
    mpf_hash, mpc_hash)
from .matrices.matrices import _matrix

mpi_zero = (fzero, fzero)

from .ctx_base import StandardBaseContext

new = object.__new__

def convert_mpf_(x, prec, rounding):
    if hasattr(x, "_mpf_"): return x._mpf_
    if isinstance(x, int_types): return from_int(x, prec, rounding)
    if isinstance(x, float): return from_float(x, prec, rounding)
    if isinstance(x, basestring): return from_str(x, prec, rounding)
    raise NotImplementedError


class ivmpf(object):
    """
    Interval arithmetic class. Precision is controlled by iv.prec.
    """

    def __new__(cls, x=0):
        return cls.ctx.convert(x)

    def cast(self, cls, f_convert):
        a, b = self._mpi_
        if a == b:
            return cls(f_convert(a))
        raise ValueError

    def __int__(self):
        return self.cast(int, libmp.to_int)

    def __float__(self):
        return self.cast(float, libmp.to_float)

    def __complex__(self):
        return self.cast(complex, libmp.to_float)

    def __hash__(self):
        a, b = self._mpi_
        if a == b:
            return mpf_hash(a)
        else:
            return hash(self._mpi_)

    @property
    def real(self): return self

    @property
    def imag(self): return self.ctx.zero

    def conjugate(self): return self

    @property
    def a(self):
        a, b = self._mpi_
        return self.ctx.make_mpf((a, a))

    @property
    def b(self):
        a, b = self._mpi_
        return self.ctx.make_mpf((b, b))

    @property
    def mid(self):
        ctx = self.ctx
        v = mpi_mid(self._mpi_, ctx.prec)
        return ctx.make_mpf((v, v))

    @property
    def delta(self):
        ctx = self.ctx
        v = mpi_delta(self._mpi_, ctx.prec)
        return ctx.make_mpf((v,v))

    @property
    def _mpci_(self):
        return self._mpi_, mpi_zero

    def _compare(*args):
        raise TypeError("no ordering relation is defined for intervals")

    __gt__ = _compare
    __le__ = _compare
    __gt__ = _compare
    __ge__ = _compare

    def __contains__(self, t):
        t = self.ctx.mpf(t)
        return (self.a <= t.a) and (t.b <= self.b)

    def __str__(self):
        return mpi_str(self._mpi_, self.ctx.prec)

    def __repr__(self):
        if self.ctx.pretty:
            return str(self)
        a, b = self._mpi_
        n = repr_dps(self.ctx.prec)
        a = libmp.to_str(a, n)
        b = libmp.to_str(b, n)
        return "mpi(%r, %r)" % (a, b)

    def _compare(s, t, cmpfun):
        if not hasattr(t, "_mpi_"):
            try:
                t = s.ctx.convert(t)
            except:
                return NotImplemented
        return cmpfun(s._mpi_, t._mpi_)

    def __eq__(s, t): return s._compare(t, libmp.mpi_eq)
    def __ne__(s, t): return s._compare(t, libmp.mpi_ne)
    def __lt__(s, t): return s._compare(t, libmp.mpi_lt)
    def __le__(s, t): return s._compare(t, libmp.mpi_le)
    def __gt__(s, t): return s._compare(t, libmp.mpi_gt)
    def __ge__(s, t): return s._compare(t, libmp.mpi_ge)

    def __abs__(self):
        return self.ctx.make_mpf(mpi_abs(self._mpi_, self.ctx.prec))
    def __pos__(self):
        return self.ctx.make_mpf(mpi_pos(self._mpi_, self.ctx.prec))
    def __neg__(self):
        return self.ctx.make_mpf(mpi_neg(self._mpi_, self.ctx.prec))

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.ctx.almosteq(s, t, rel_eps, abs_eps)

class ivmpc(object):

    def __new__(cls, re=0, im=0):
        re = cls.ctx.convert(re)
        im = cls.ctx.convert(im)
        y = new(cls)
        y._mpci_ = re._mpi_, im._mpi_
        return y

    def __hash__(self):
        (a, b), (c,d) = self._mpci_
        if a == b and c == d:
            return mpc_hash((a, c))
        else:
            return hash(self._mpci_)

    def __repr__(s):
        if s.ctx.pretty:
            return str(s)
        return "iv.mpc(%s, %s)" % (repr(s.real), repr(s.imag))

    def __str__(s):
        return "(%s + %s*j)" % (str(s.real), str(s.imag))

    @property
    def a(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((a, a))

    @property
    def b(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((b, b))

    @property
    def c(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((c, c))

    @property
    def d(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((d, d))

    @property
    def real(s):
        return s.ctx.make_mpf(s._mpci_[0])

    @property
    def imag(s):
        return s.ctx.make_mpf(s._mpci_[1])

    def conjugate(s):
        a, b = s._mpci_
        return s.ctx.make_mpc((a, mpf_neg(b)))

    def overlap(s, t):
        t = s.ctx.convert(t)
        real_overlap = (s.a <= t.a <= s.b) or (s.a <= t.b <= s.b) or (t.a <= s.a <= t.b) or (t.a <= s.b <= t.b)
        imag_overlap = (s.c <= t.c <= s.d) or (s.c <= t.d <= s.d) or (t.c <= s.c <= t.d) or (t.c <= s.d <= t.d)
        return real_overlap and imag_overlap

    def __contains__(s, t):
        t = s.ctx.convert(t)
        return t.real in s.real and t.imag in s.imag

    def _compare(s, t, ne=False):
        if not isinstance(t, s.ctx._types):
            try:
                t = s.ctx.convert(t)
            except:
                return NotImplemented
        if hasattr(t, '_mpi_'):
            tval = t._mpi_, mpi_zero
        elif hasattr(t, '_mpci_'):
            tval = t._mpci_
        if ne:
            return s._mpci_ != tval
        return s._mpci_ == tval

    def __eq__(s, t): return s._compare(t)
    def __ne__(s, t): return s._compare(t, True)

    def __lt__(s, t): raise TypeError("complex intervals cannot be ordered")
    __le__ = __gt__ = __ge__ = __lt__

    def __neg__(s): return s.ctx.make_mpc(mpci_neg(s._mpci_, s.ctx.prec))
    def __pos__(s): return s.ctx.make_mpc(mpci_pos(s._mpci_, s.ctx.prec))
    def __abs__(s): return s.ctx.make_mpf(mpci_abs(s._mpci_, s.ctx.prec))

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.ctx.almosteq(s, t, rel_eps, abs_eps)

def _binary_op(f_real, f_complex):
    def g_complex(ctx, sval, tval):
        return ctx.make_mpc(f_complex(sval, tval, ctx.prec))
    def g_real(ctx, sval, tval):
        try:
            return ctx.make_mpf(f_real(sval, tval, ctx.prec))
        except ComplexResult:
            sval = (sval, mpi_zero)
            tval = (tval, mpi_zero)
            return g_complex(ctx, sval, tval)
    def lop_real(s, t):
        if isinstance(t, _matrix): return NotImplemented
        ctx = s.ctx
        if not isinstance(t, ctx._types): t = ctx.convert(t)
        if hasattr(t, "_mpi_"): return g_real(ctx, s._mpi_, t._mpi_)
        if hasattr(t, "_mpci_"): return g_complex(ctx, (s._mpi_, mpi_zero), t._mpci_)
        return NotImplemented
    def rop_real(s, t):
        ctx = s.ctx
        if not isinstance(t, ctx._types): t = ctx.convert(t)
        if hasattr(t, "_mpi_"): return g_real(ctx, t._mpi_, s._mpi_)
        if hasattr(t, "_mpci_"): return g_complex(ctx, t._mpci_, (s._mpi_, mpi_zero))
        return NotImplemented
    def lop_complex(s, t):
        if isinstance(t, _matrix): return NotImplemented
        ctx = s.ctx
        if not isinstance(t, s.ctx._types):
            try:
                t = s.ctx.convert(t)
            except (ValueError, TypeError):
                return NotImplemented
        return g_complex(ctx, s._mpci_, t._mpci_)
    def rop_complex(s, t):
        ctx = s.ctx
        if not isinstance(t, s.ctx._types):
            t = s.ctx.convert(t)
        return g_complex(ctx, t._mpci_, s._mpci_)
    return lop_real, rop_real, lop_complex, rop_complex

ivmpf.__add__, ivmpf.__radd__, ivmpc.__add__, ivmpc.__radd__ = _binary_op(mpi_add, mpci_add)
ivmpf.__sub__, ivmpf.__rsub__, ivmpc.__sub__, ivmpc.__rsub__ = _binary_op(mpi_sub, mpci_sub)
ivmpf.__mul__, ivmpf.__rmul__, ivmpc.__mul__, ivmpc.__rmul__ = _binary_op(mpi_mul, mpci_mul)
ivmpf.__div__, ivmpf.__rdiv__, ivmpc.__div__, ivmpc.__rdiv__ = _binary_op(mpi_div, mpci_div)
ivmpf.__pow__, ivmpf.__rpow__, ivmpc.__pow__, ivmpc.__rpow__ = _binary_op(mpi_pow, mpci_pow)

ivmpf.__truediv__ = ivmpf.__div__; ivmpf.__rtruediv__ = ivmpf.__rdiv__
ivmpc.__truediv__ = ivmpc.__div__; ivmpc.__rtruediv__ = ivmpc.__rdiv__

class ivmpf_constant(ivmpf):
    def __new__(cls, f):
        self = new(cls)
        self._f = f
        return self
    def _get_mpi_(self):
        prec = self.ctx._prec[0]
        a = self._f(prec, round_floor)
        b = self._f(prec, round_ceiling)
        return a, b
    _mpi_ = property(_get_mpi_)

class MPIntervalContext(StandardBaseContext):

    def __init__(ctx):
        ctx.mpf = type('ivmpf', (ivmpf,), {})
        ctx.mpc = type('ivmpc', (ivmpc,), {})
        ctx._types = (ctx.mpf, ctx.mpc)
        ctx._constant = type('ivmpf_constant', (ivmpf_constant,), {})
        ctx._prec = [53]
        ctx._set_prec(53)
        ctx._constant._ctxdata = ctx.mpf._ctxdata = ctx.mpc._ctxdata = [ctx.mpf, new, ctx._prec]
        ctx._constant.ctx = ctx.mpf.ctx = ctx.mpc.ctx = ctx
        ctx.pretty = False
        StandardBaseContext.__init__(ctx)
        ctx._init_builtins()

    def _mpi(ctx, a, b=None):
        if b is None:
            return ctx.mpf(a)
        return ctx.mpf((a,b))

    def _init_builtins(ctx):
        ctx.one = ctx.mpf(1)
        ctx.zero = ctx.mpf(0)
        ctx.inf = ctx.mpf('inf')
        ctx.ninf = -ctx.inf
        ctx.nan = ctx.mpf('nan')
        ctx.j = ctx.mpc(0,1)
        ctx.exp = ctx._wrap_mpi_function(libmp.mpi_exp, libmp.mpci_exp)
        ctx.sqrt = ctx._wrap_mpi_function(libmp.mpi_sqrt)
        ctx.ln = ctx._wrap_mpi_function(libmp.mpi_log, libmp.mpci_log)
        ctx.cos = ctx._wrap_mpi_function(libmp.mpi_cos, libmp.mpci_cos)
        ctx.sin = ctx._wrap_mpi_function(libmp.mpi_sin, libmp.mpci_sin)
        ctx.tan = ctx._wrap_mpi_function(libmp.mpi_tan)
        ctx.gamma = ctx._wrap_mpi_function(libmp.mpi_gamma, libmp.mpci_gamma)
        ctx.loggamma = ctx._wrap_mpi_function(libmp.mpi_loggamma, libmp.mpci_loggamma)
        ctx.rgamma = ctx._wrap_mpi_function(libmp.mpi_rgamma, libmp.mpci_rgamma)
        ctx.factorial = ctx._wrap_mpi_function(libmp.mpi_factorial, libmp.mpci_factorial)
        ctx.fac = ctx.factorial

        ctx.eps = ctx._constant(lambda prec, rnd: (0, MPZ_ONE, 1-prec, 1))
        ctx.pi = ctx._constant(libmp.mpf_pi)
        ctx.e = ctx._constant(libmp.mpf_e)
        ctx.ln2 = ctx._constant(libmp.mpf_ln2)
        ctx.ln10 = ctx._constant(libmp.mpf_ln10)
        ctx.phi = ctx._constant(libmp.mpf_phi)
        ctx.euler = ctx._constant(libmp.mpf_euler)
        ctx.catalan = ctx._constant(libmp.mpf_catalan)
        ctx.glaisher = ctx._constant(libmp.mpf_glaisher)
        ctx.khinchin = ctx._constant(libmp.mpf_khinchin)
        ctx.twinprime = ctx._constant(libmp.mpf_twinprime)

    def _wrap_mpi_function(ctx, f_real, f_complex=None):
        def g(x, **kwargs):
            if kwargs:
                prec = kwargs.get('prec', ctx._prec[0])
            else:
                prec = ctx._prec[0]
            x = ctx.convert(x)
            if hasattr(x, "_mpi_"):
                return ctx.make_mpf(f_real(x._mpi_, prec))
            if hasattr(x, "_mpci_"):
                return ctx.make_mpc(f_complex(x._mpci_, prec))
            raise ValueError
        return g

    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        if wrap:
            def f_wrapped(ctx, *args, **kwargs):
                convert = ctx.convert
                args = [convert(a) for a in args]
                prec = ctx.prec
                try:
                    ctx.prec += 10
                    retval = f(ctx, *args, **kwargs)
                finally:
                    ctx.prec = prec
                return +retval
        else:
            f_wrapped = f
        setattr(cls, name, f_wrapped)

    def _set_prec(ctx, n):
        ctx._prec[0] = max(1, int(n))
        ctx._dps = prec_to_dps(n)

    def _set_dps(ctx, n):
        ctx._prec[0] = dps_to_prec(n)
        ctx._dps = max(1, int(n))

    prec = property(lambda ctx: ctx._prec[0], _set_prec)
    dps = property(lambda ctx: ctx._dps, _set_dps)

    def make_mpf(ctx, v):
        a = new(ctx.mpf)
        a._mpi_ = v
        return a

    def make_mpc(ctx, v):
        a = new(ctx.mpc)
        a._mpci_ = v
        return a

    def _mpq(ctx, pq):
        p, q = pq
        a = libmp.from_rational(p, q, ctx.prec, round_floor)
        b = libmp.from_rational(p, q, ctx.prec, round_ceiling)
        return ctx.make_mpf((a, b))

    def convert(ctx, x):
        if isinstance(x, (ctx.mpf, ctx.mpc)):
            return x
        if isinstance(x, ctx._constant):
            return +x
        if isinstance(x, complex) or hasattr(x, "_mpc_"):
            re = ctx.convert(x.real)
            im = ctx.convert(x.imag)
            return ctx.mpc(re,im)
        if isinstance(x, basestring):
            v = mpi_from_str(x, ctx.prec)
            return ctx.make_mpf(v)
        if hasattr(x, "_mpi_"):
            a, b = x._mpi_
        else:
            try:
                a, b = x
            except (TypeError, ValueError):
                a = b = x
            if hasattr(a, "_mpi_"):
                a = a._mpi_[0]
            else:
                a = convert_mpf_(a, ctx.prec, round_floor)
            if hasattr(b, "_mpi_"):
                b = b._mpi_[1]
            else:
                b = convert_mpf_(b, ctx.prec, round_ceiling)
        if a == fnan or b == fnan:
            a = fninf
            b = finf
        assert mpf_le(a, b), "endpoints must be properly ordered"
        return ctx.make_mpf((a, b))

    def nstr(ctx, x, n=5, **kwargs):
        x = ctx.convert(x)
        if hasattr(x, "_mpi_"):
            return libmp.mpi_to_str(x._mpi_, n, **kwargs)
        if hasattr(x, "_mpci_"):
            re = libmp.mpi_to_str(x._mpci_[0], n, **kwargs)
            im = libmp.mpi_to_str(x._mpci_[1], n, **kwargs)
            return "(%s + %s*j)" % (re, im)

    def mag(ctx, x):
        x = ctx.convert(x)
        if isinstance(x, ctx.mpc):
            return max(ctx.mag(x.real), ctx.mag(x.imag)) + 1
        a, b = libmp.mpi_abs(x._mpi_)
        sign, man, exp, bc = b
        if man:
            return exp+bc
        if b == fzero:
            return ctx.ninf
        if b == fnan:
            return ctx.nan
        return ctx.inf

    def isnan(ctx, x):
        return False

    def isinf(ctx, x):
        return x == ctx.inf

    def isint(ctx, x):
        x = ctx.convert(x)
        a, b = x._mpi_
        if a == b:
            sign, man, exp, bc = a
            if man:
                return exp >= 0
            return a == fzero
        return None

    def ldexp(ctx, x, n):
        a, b = ctx.convert(x)._mpi_
        a = libmp.mpf_shift(a, n)
        b = libmp.mpf_shift(b, n)
        return ctx.make_mpf((a,b))

    def absmin(ctx, x):
        return abs(ctx.convert(x)).a

    def absmax(ctx, x):
        return abs(ctx.convert(x)).b

    def atan2(ctx, y, x):
        y = ctx.convert(y)._mpi_
        x = ctx.convert(x)._mpi_
        return ctx.make_mpf(libmp.mpi_atan2(y,x,ctx.prec))

    def _convert_param(ctx, x):
        if isinstance(x, libmp.int_types):
            return x, 'Z'
        if isinstance(x, tuple):
            p, q = x
            return (ctx.mpf(p) / ctx.mpf(q), 'R')
        x = ctx.convert(x)
        if isinstance(x, ctx.mpf):
            return x, 'R'
        if isinstance(x, ctx.mpc):
            return x, 'C'
        raise ValueError

    def _is_real_type(ctx, z):
        return isinstance(z, ctx.mpf) or isinstance(z, int_types)

    def _is_complex_type(ctx, z):
        return isinstance(z, ctx.mpc)

    def hypsum(ctx, p, q, types, coeffs, z, maxterms=6000, **kwargs):
        coeffs = list(coeffs)
        num = range(p)
        den = range(p,p+q)
        #tol = ctx.eps
        s = t = ctx.one
        k = 0
        while 1:
            for i in num: t *= (coeffs[i]+k)
            for i in den: t /= (coeffs[i]+k)
            k += 1; t /= k; t *= z; s += t
            if t == 0:
                return s
            #if abs(t) < tol:
            #    return s
            if k > maxterms:
                raise ctx.NoConvergence


# Register with "numbers" ABC
#     We do not subclass, hence we do not use the @abstractmethod checks. While
#     this is less invasive it may turn out that we do not actually support
#     parts of the expected interfaces.  See
#     http://docs.python.org/2/library/numbers.html for list of abstract
#     methods.
try:
    import numbers
    numbers.Complex.register(ivmpc)
    numbers.Real.register(ivmpf)
except ImportError:
    pass
