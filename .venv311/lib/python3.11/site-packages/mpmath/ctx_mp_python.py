#from ctx_base import StandardBaseContext

from .libmp.backend import basestring, exec_

from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
    round_floor, round_ceiling, dps_to_prec, round_nearest, prec_to_dps,
    ComplexResult, to_pickable, from_pickable, normalize,
    from_int, from_float, from_npfloat, from_Decimal, from_str, to_int, to_float, to_str,
    from_rational, from_man_exp,
    fone, fzero, finf, fninf, fnan,
    mpf_abs, mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul, mpf_mul_int,
    mpf_div, mpf_rdiv_int, mpf_pow_int, mpf_mod,
    mpf_eq, mpf_cmp, mpf_lt, mpf_gt, mpf_le, mpf_ge,
    mpf_hash, mpf_rand,
    mpf_sum,
    bitcount, to_fixed,
    mpc_to_str,
    mpc_to_complex, mpc_hash, mpc_pos, mpc_is_nonzero, mpc_neg, mpc_conjugate,
    mpc_abs, mpc_add, mpc_add_mpf, mpc_sub, mpc_sub_mpf, mpc_mul, mpc_mul_mpf,
    mpc_mul_int, mpc_div, mpc_div_mpf, mpc_pow, mpc_pow_mpf, mpc_pow_int,
    mpc_mpf_div,
    mpf_pow,
    mpf_pi, mpf_degree, mpf_e, mpf_phi, mpf_ln2, mpf_ln10,
    mpf_euler, mpf_catalan, mpf_apery, mpf_khinchin,
    mpf_glaisher, mpf_twinprime, mpf_mertens,
    int_types)

from . import rational
from . import function_docs

new = object.__new__

class mpnumeric(object):
    """Base class for mpf and mpc."""
    __slots__ = []
    def __new__(cls, val):
        raise NotImplementedError

class _mpf(mpnumeric):
    """
    An mpf instance holds a real-valued floating-point number. mpf:s
    work analogously to Python floats, but support arbitrary-precision
    arithmetic.
    """
    __slots__ = ['_mpf_']

    def __new__(cls, val=fzero, **kwargs):
        """A new mpf can be created from a Python float, an int, a
        or a decimal string representing a number in floating-point
        format."""
        prec, rounding = cls.context._prec_rounding
        if kwargs:
            prec = kwargs.get('prec', prec)
            if 'dps' in kwargs:
                prec = dps_to_prec(kwargs['dps'])
            rounding = kwargs.get('rounding', rounding)
        if type(val) is cls:
            sign, man, exp, bc = val._mpf_
            if (not man) and exp:
                return val
            v = new(cls)
            v._mpf_ = normalize(sign, man, exp, bc, prec, rounding)
            return v
        elif type(val) is tuple:
            if len(val) == 2:
                v = new(cls)
                v._mpf_ = from_man_exp(val[0], val[1], prec, rounding)
                return v
            if len(val) == 4:
                if val not in (finf, fninf, fnan):
                    sign, man, exp, bc = val
                    val = normalize(sign, MPZ(man), exp, bc, prec, rounding)
                v = new(cls)
                v._mpf_ = val
                return v
            raise ValueError
        else:
            v = new(cls)
            v._mpf_ = mpf_pos(cls.mpf_convert_arg(val, prec, rounding), prec, rounding)
            return v

    @classmethod
    def mpf_convert_arg(cls, x, prec, rounding):
        if isinstance(x, int_types): return from_int(x)
        if isinstance(x, float): return from_float(x)
        if isinstance(x, basestring): return from_str(x, prec, rounding)
        if isinstance(x, cls.context.constant): return x.func(prec, rounding)
        if hasattr(x, '_mpf_'): return x._mpf_
        if hasattr(x, '_mpmath_'):
            t = cls.context.convert(x._mpmath_(prec, rounding))
            if hasattr(t, '_mpf_'):
                return t._mpf_
        if hasattr(x, '_mpi_'):
            a, b = x._mpi_
            if a == b:
                return a
            raise ValueError("can only create mpf from zero-width interval")
        raise TypeError("cannot create mpf from " + repr(x))

    @classmethod
    def mpf_convert_rhs(cls, x):
        if isinstance(x, int_types): return from_int(x)
        if isinstance(x, float): return from_float(x)
        if isinstance(x, complex_types): return cls.context.mpc(x)
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return from_rational(p, q, cls.context.prec)
        if hasattr(x, '_mpf_'): return x._mpf_
        if hasattr(x, '_mpmath_'):
            t = cls.context.convert(x._mpmath_(*cls.context._prec_rounding))
            if hasattr(t, '_mpf_'):
                return t._mpf_
            return t
        return NotImplemented

    @classmethod
    def mpf_convert_lhs(cls, x):
        x = cls.mpf_convert_rhs(x)
        if type(x) is tuple:
            return cls.context.make_mpf(x)
        return x

    man_exp = property(lambda self: self._mpf_[1:3])
    man = property(lambda self: self._mpf_[1])
    exp = property(lambda self: self._mpf_[2])
    bc = property(lambda self: self._mpf_[3])

    real = property(lambda self: self)
    imag = property(lambda self: self.context.zero)

    conjugate = lambda self: self

    def __getstate__(self): return to_pickable(self._mpf_)
    def __setstate__(self, val): self._mpf_ = from_pickable(val)

    def __repr__(s):
        if s.context.pretty:
            return str(s)
        return "mpf('%s')" % to_str(s._mpf_, s.context._repr_digits)

    def __str__(s): return to_str(s._mpf_, s.context._str_digits)
    def __hash__(s): return mpf_hash(s._mpf_)
    def __int__(s): return int(to_int(s._mpf_))
    def __long__(s): return long(to_int(s._mpf_))
    def __float__(s): return to_float(s._mpf_, rnd=s.context._prec_rounding[1])
    def __complex__(s): return complex(float(s))
    def __nonzero__(s): return s._mpf_ != fzero

    __bool__ = __nonzero__

    def __abs__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_abs(s._mpf_, prec, rounding)
        return v

    def __pos__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_pos(s._mpf_, prec, rounding)
        return v

    def __neg__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_neg(s._mpf_, prec, rounding)
        return v

    def _cmp(s, t, func):
        if hasattr(t, '_mpf_'):
            t = t._mpf_
        else:
            t = s.mpf_convert_rhs(t)
            if t is NotImplemented:
                return t
        return func(s._mpf_, t)

    def __cmp__(s, t): return s._cmp(t, mpf_cmp)
    def __lt__(s, t): return s._cmp(t, mpf_lt)
    def __gt__(s, t): return s._cmp(t, mpf_gt)
    def __le__(s, t): return s._cmp(t, mpf_le)
    def __ge__(s, t): return s._cmp(t, mpf_ge)

    def __ne__(s, t):
        v = s.__eq__(t)
        if v is NotImplemented:
            return v
        return not v

    def __rsub__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if type(t) in int_types:
            v = new(cls)
            v._mpf_ = mpf_sub(from_int(t), s._mpf_, prec, rounding)
            return v
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t - s

    def __rdiv__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpf_ = mpf_rdiv_int(t, s._mpf_, prec, rounding)
            return v
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t / s

    def __rpow__(s, t):
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t ** s

    def __rmod__(s, t):
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t % s

    def sqrt(s):
        return s.context.sqrt(s)

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.context.almosteq(s, t, rel_eps, abs_eps)

    def to_fixed(self, prec):
        return to_fixed(self._mpf_, prec)

    def __round__(self, *args):
        return round(float(self), *args)

mpf_binary_op = """
def %NAME%(self, other):
    mpf, new, (prec, rounding) = self._ctxdata
    sval = self._mpf_
    if hasattr(other, '_mpf_'):
        tval = other._mpf_
        %WITH_MPF%
    ttype = type(other)
    if ttype in int_types:
        %WITH_INT%
    elif ttype is float:
        tval = from_float(other)
        %WITH_MPF%
    elif hasattr(other, '_mpc_'):
        tval = other._mpc_
        mpc = type(other)
        %WITH_MPC%
    elif ttype is complex:
        tval = from_float(other.real), from_float(other.imag)
        mpc = self.context.mpc
        %WITH_MPC%
    if isinstance(other, mpnumeric):
        return NotImplemented
    try:
        other = mpf.context.convert(other, strings=False)
    except TypeError:
        return NotImplemented
    return self.%NAME%(other)
"""

return_mpf = "; obj = new(mpf); obj._mpf_ = val; return obj"
return_mpc = "; obj = new(mpc); obj._mpc_ = val; return obj"

mpf_pow_same = """
        try:
            val = mpf_pow(sval, tval, prec, rounding) %s
        except ComplexResult:
            if mpf.context.trap_complex:
                raise
            mpc = mpf.context.mpc
            val = mpc_pow((sval, fzero), (tval, fzero), prec, rounding) %s
""" % (return_mpf, return_mpc)

def binary_op(name, with_mpf='', with_int='', with_mpc=''):
    code = mpf_binary_op
    code = code.replace("%WITH_INT%", with_int)
    code = code.replace("%WITH_MPC%", with_mpc)
    code = code.replace("%WITH_MPF%", with_mpf)
    code = code.replace("%NAME%", name)
    np = {}
    exec_(code, globals(), np)
    return np[name]

_mpf.__eq__ = binary_op('__eq__',
    'return mpf_eq(sval, tval)',
    'return mpf_eq(sval, from_int(other))',
    'return (tval[1] == fzero) and mpf_eq(tval[0], sval)')

_mpf.__add__ = binary_op('__add__',
    'val = mpf_add(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_add(sval, from_int(other), prec, rounding)' + return_mpf,
    'val = mpc_add_mpf(tval, sval, prec, rounding)' + return_mpc)

_mpf.__sub__ = binary_op('__sub__',
    'val = mpf_sub(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_sub(sval, from_int(other), prec, rounding)' + return_mpf,
    'val = mpc_sub((sval, fzero), tval, prec, rounding)' + return_mpc)

_mpf.__mul__ = binary_op('__mul__',
    'val = mpf_mul(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_mul_int(sval, other, prec, rounding)' + return_mpf,
    'val = mpc_mul_mpf(tval, sval, prec, rounding)' + return_mpc)

_mpf.__div__ = binary_op('__div__',
    'val = mpf_div(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_div(sval, from_int(other), prec, rounding)' + return_mpf,
    'val = mpc_mpf_div(sval, tval, prec, rounding)' + return_mpc)

_mpf.__mod__ = binary_op('__mod__',
    'val = mpf_mod(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_mod(sval, from_int(other), prec, rounding)' + return_mpf,
    'raise NotImplementedError("complex modulo")')

_mpf.__pow__ = binary_op('__pow__',
    mpf_pow_same,
    'val = mpf_pow_int(sval, other, prec, rounding)' + return_mpf,
    'val = mpc_pow((sval, fzero), tval, prec, rounding)' + return_mpc)

_mpf.__radd__ = _mpf.__add__
_mpf.__rmul__ = _mpf.__mul__
_mpf.__truediv__ = _mpf.__div__
_mpf.__rtruediv__ = _mpf.__rdiv__


class _constant(_mpf):
    """Represents a mathematical constant with dynamic precision.
    When printed or used in an arithmetic operation, a constant
    is converted to a regular mpf at the working precision. A
    regular mpf can also be obtained using the operation +x."""

    def __new__(cls, func, name, docname=''):
        a = object.__new__(cls)
        a.name = name
        a.func = func
        a.__doc__ = getattr(function_docs, docname, '')
        return a

    def __call__(self, prec=None, dps=None, rounding=None):
        prec2, rounding2 = self.context._prec_rounding
        if not prec: prec = prec2
        if not rounding: rounding = rounding2
        if dps: prec = dps_to_prec(dps)
        return self.context.make_mpf(self.func(prec, rounding))

    @property
    def _mpf_(self):
        prec, rounding = self.context._prec_rounding
        return self.func(prec, rounding)

    def __repr__(self):
        return "<%s: %s~>" % (self.name, self.context.nstr(self(dps=15)))


class _mpc(mpnumeric):
    """
    An mpc represents a complex number using a pair of mpf:s (one
    for the real part and another for the imaginary part.) The mpc
    class behaves fairly similarly to Python's complex type.
    """

    __slots__ = ['_mpc_']

    def __new__(cls, real=0, imag=0):
        s = object.__new__(cls)
        if isinstance(real, complex_types):
            real, imag = real.real, real.imag
        elif hasattr(real, '_mpc_'):
            s._mpc_ = real._mpc_
            return s
        real = cls.context.mpf(real)
        imag = cls.context.mpf(imag)
        s._mpc_ = (real._mpf_, imag._mpf_)
        return s

    real = property(lambda self: self.context.make_mpf(self._mpc_[0]))
    imag = property(lambda self: self.context.make_mpf(self._mpc_[1]))

    def __getstate__(self):
        return to_pickable(self._mpc_[0]), to_pickable(self._mpc_[1])

    def __setstate__(self, val):
        self._mpc_ = from_pickable(val[0]), from_pickable(val[1])

    def __repr__(s):
        if s.context.pretty:
            return str(s)
        r = repr(s.real)[4:-1]
        i = repr(s.imag)[4:-1]
        return "%s(real=%s, imag=%s)" % (type(s).__name__, r, i)

    def __str__(s):
        return "(%s)" % mpc_to_str(s._mpc_, s.context._str_digits)

    def __complex__(s):
        return mpc_to_complex(s._mpc_, rnd=s.context._prec_rounding[1])

    def __pos__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpc_ = mpc_pos(s._mpc_, prec, rounding)
        return v

    def __abs__(s):
        prec, rounding = s.context._prec_rounding
        v = new(s.context.mpf)
        v._mpf_ = mpc_abs(s._mpc_, prec, rounding)
        return v

    def __neg__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpc_ = mpc_neg(s._mpc_, prec, rounding)
        return v

    def conjugate(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpc_ = mpc_conjugate(s._mpc_, prec, rounding)
        return v

    def __nonzero__(s):
        return mpc_is_nonzero(s._mpc_)

    __bool__ = __nonzero__

    def __hash__(s):
        return mpc_hash(s._mpc_)

    @classmethod
    def mpc_convert_lhs(cls, x):
        try:
            y = cls.context.convert(x)
            return y
        except TypeError:
            return NotImplemented

    def __eq__(s, t):
        if not hasattr(t, '_mpc_'):
            if isinstance(t, str):
                return False
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
        return s.real == t.real and s.imag == t.imag

    def __ne__(s, t):
        b = s.__eq__(t)
        if b is NotImplemented:
            return b
        return not b

    def _compare(*args):
        raise TypeError("no ordering relation is defined for complex numbers")

    __gt__ = _compare
    __le__ = _compare
    __gt__ = _compare
    __ge__ = _compare

    def __add__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_add_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
        v = new(cls)
        v._mpc_ = mpc_add(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __sub__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_sub_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
        v = new(cls)
        v._mpc_ = mpc_sub(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __mul__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            if isinstance(t, int_types):
                v = new(cls)
                v._mpc_ = mpc_mul_int(s._mpc_, t, prec, rounding)
                return v
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_mul_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
            t = s.mpc_convert_lhs(t)
        v = new(cls)
        v._mpc_ = mpc_mul(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __div__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_div_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
        v = new(cls)
        v._mpc_ = mpc_div(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __pow__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpc_ = mpc_pow_int(s._mpc_, t, prec, rounding)
            return v
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        v = new(cls)
        if hasattr(t, '_mpf_'):
            v._mpc_ = mpc_pow_mpf(s._mpc_, t._mpf_, prec, rounding)
        else:
            v._mpc_ = mpc_pow(s._mpc_, t._mpc_, prec, rounding)
        return v

    __radd__ = __add__

    def __rsub__(s, t):
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t - s

    def __rmul__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpc_ = mpc_mul_int(s._mpc_, t, prec, rounding)
            return v
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t * s

    def __rdiv__(s, t):
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t / s

    def __rpow__(s, t):
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t ** s

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.context.almosteq(s, t, rel_eps, abs_eps)


complex_types = (complex, _mpc)


class PythonMPContext(object):

    def __init__(ctx):
        ctx._prec_rounding = [53, round_nearest]
        ctx.mpf = type('mpf', (_mpf,), {})
        ctx.mpc = type('mpc', (_mpc,), {})
        ctx.mpf._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.mpc._ctxdata = [ctx.mpc, new, ctx._prec_rounding]
        ctx.mpf.context = ctx
        ctx.mpc.context = ctx
        ctx.constant = type('constant', (_constant,), {})
        ctx.constant._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.constant.context = ctx

    def make_mpf(ctx, v):
        a = new(ctx.mpf)
        a._mpf_ = v
        return a

    def make_mpc(ctx, v):
        a = new(ctx.mpc)
        a._mpc_ = v
        return a

    def default(ctx):
        ctx._prec = ctx._prec_rounding[0] = 53
        ctx._dps = 15
        ctx.trap_complex = False

    def _set_prec(ctx, n):
        ctx._prec = ctx._prec_rounding[0] = max(1, int(n))
        ctx._dps = prec_to_dps(n)

    def _set_dps(ctx, n):
        ctx._prec = ctx._prec_rounding[0] = dps_to_prec(n)
        ctx._dps = max(1, int(n))

    prec = property(lambda ctx: ctx._prec, _set_prec)
    dps = property(lambda ctx: ctx._dps, _set_dps)

    def convert(ctx, x, strings=True):
        """
        Converts *x* to an ``mpf`` or ``mpc``. If *x* is of type ``mpf``,
        ``mpc``, ``int``, ``float``, ``complex``, the conversion
        will be performed losslessly.

        If *x* is a string, the result will be rounded to the present
        working precision. Strings representing fractions or complex
        numbers are permitted.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> mpmathify(3.5)
            mpf('3.5')
            >>> mpmathify('2.1')
            mpf('2.1000000000000001')
            >>> mpmathify('3/4')
            mpf('0.75')
            >>> mpmathify('2+3j')
            mpc(real='2.0', imag='3.0')

        """
        if type(x) in ctx.types: return x
        if isinstance(x, int_types): return ctx.make_mpf(from_int(x))
        if isinstance(x, float): return ctx.make_mpf(from_float(x))
        if isinstance(x, complex):
            return ctx.make_mpc((from_float(x.real), from_float(x.imag)))
        if type(x).__module__ == 'numpy': return ctx.npconvert(x)
        if isinstance(x, numbers.Rational): # e.g. Fraction
            try: x = rational.mpq(int(x.numerator), int(x.denominator))
            except: pass
        prec, rounding = ctx._prec_rounding
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return ctx.make_mpf(from_rational(p, q, prec))
        if strings and isinstance(x, basestring):
            try:
                _mpf_ = from_str(x, prec, rounding)
                return ctx.make_mpf(_mpf_)
            except ValueError:
                pass
        if hasattr(x, '_mpf_'): return ctx.make_mpf(x._mpf_)
        if hasattr(x, '_mpc_'): return ctx.make_mpc(x._mpc_)
        if hasattr(x, '_mpmath_'):
            return ctx.convert(x._mpmath_(prec, rounding))
        if type(x).__module__ == 'decimal':
            try: return ctx.make_mpf(from_Decimal(x, prec, rounding))
            except: pass
        return ctx._convert_fallback(x, strings)

    def npconvert(ctx, x):
        """
        Converts *x* to an ``mpf`` or ``mpc``. *x* should be a numpy
        scalar.
        """
        import numpy as np
        if isinstance(x, np.integer): return ctx.make_mpf(from_int(int(x)))
        if isinstance(x, np.floating): return ctx.make_mpf(from_npfloat(x))
        if isinstance(x, np.complexfloating):
            return ctx.make_mpc((from_npfloat(x.real), from_npfloat(x.imag)))
        raise TypeError("cannot create mpf from " + repr(x))

    def isnan(ctx, x):
        """
        Return *True* if *x* is a NaN (not-a-number), or for a complex
        number, whether either the real or complex part is NaN;
        otherwise return *False*::

            >>> from mpmath import *
            >>> isnan(3.14)
            False
            >>> isnan(nan)
            True
            >>> isnan(mpc(3.14,2.72))
            False
            >>> isnan(mpc(3.14,nan))
            True

        """
        if hasattr(x, "_mpf_"):
            return x._mpf_ == fnan
        if hasattr(x, "_mpc_"):
            return fnan in x._mpc_
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return False
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isnan(x)
        raise TypeError("isnan() needs a number as input")

    def isinf(ctx, x):
        """
        Return *True* if the absolute value of *x* is infinite;
        otherwise return *False*::

            >>> from mpmath import *
            >>> isinf(inf)
            True
            >>> isinf(-inf)
            True
            >>> isinf(3)
            False
            >>> isinf(3+4j)
            False
            >>> isinf(mpc(3,inf))
            True
            >>> isinf(mpc(inf,3))
            True

        """
        if hasattr(x, "_mpf_"):
            return x._mpf_ in (finf, fninf)
        if hasattr(x, "_mpc_"):
            re, im = x._mpc_
            return re in (finf, fninf) or im in (finf, fninf)
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return False
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isinf(x)
        raise TypeError("isinf() needs a number as input")

    def isnormal(ctx, x):
        """
        Determine whether *x* is "normal" in the sense of floating-point
        representation; that is, return *False* if *x* is zero, an
        infinity or NaN; otherwise return *True*. By extension, a
        complex number *x* is considered "normal" if its magnitude is
        normal::

            >>> from mpmath import *
            >>> isnormal(3)
            True
            >>> isnormal(0)
            False
            >>> isnormal(inf); isnormal(-inf); isnormal(nan)
            False
            False
            False
            >>> isnormal(0+0j)
            False
            >>> isnormal(0+3j)
            True
            >>> isnormal(mpc(2,nan))
            False
        """
        if hasattr(x, "_mpf_"):
            return bool(x._mpf_[1])
        if hasattr(x, "_mpc_"):
            re, im = x._mpc_
            re_normal = bool(re[1])
            im_normal = bool(im[1])
            if re == fzero: return im_normal
            if im == fzero: return re_normal
            return re_normal and im_normal
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return bool(x)
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isnormal(x)
        raise TypeError("isnormal() needs a number as input")

    def isint(ctx, x, gaussian=False):
        """
        Return *True* if *x* is integer-valued; otherwise return
        *False*::

            >>> from mpmath import *
            >>> isint(3)
            True
            >>> isint(mpf(3))
            True
            >>> isint(3.2)
            False
            >>> isint(inf)
            False

        Optionally, Gaussian integers can be checked for::

            >>> isint(3+0j)
            True
            >>> isint(3+2j)
            False
            >>> isint(3+2j, gaussian=True)
            True

        """
        if isinstance(x, int_types):
            return True
        if hasattr(x, "_mpf_"):
            sign, man, exp, bc = xval = x._mpf_
            return bool((man and exp >= 0) or xval == fzero)
        if hasattr(x, "_mpc_"):
            re, im = x._mpc_
            rsign, rman, rexp, rbc = re
            isign, iman, iexp, ibc = im
            re_isint = (rman and rexp >= 0) or re == fzero
            if gaussian:
                im_isint = (iman and iexp >= 0) or im == fzero
                return re_isint and im_isint
            return re_isint and im == fzero
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return p % q == 0
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isint(x, gaussian)
        raise TypeError("isint() needs a number as input")

    def fsum(ctx, terms, absolute=False, squared=False):
        """
        Calculates a sum containing a finite number of terms (for infinite
        series, see :func:`~mpmath.nsum`). The terms will be converted to
        mpmath numbers. For len(terms) > 2, this function is generally
        faster and produces more accurate results than the builtin
        Python function :func:`sum`.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fsum([1, 2, 0.5, 7])
            mpf('10.5')

        With squared=True each term is squared, and with absolute=True
        the absolute value of each term is used.
        """
        prec, rnd = ctx._prec_rounding
        real = []
        imag = []
        for term in terms:
            reval = imval = 0
            if hasattr(term, "_mpf_"):
                reval = term._mpf_
            elif hasattr(term, "_mpc_"):
                reval, imval = term._mpc_
            else:
                term = ctx.convert(term)
                if hasattr(term, "_mpf_"):
                    reval = term._mpf_
                elif hasattr(term, "_mpc_"):
                    reval, imval = term._mpc_
                else:
                    raise NotImplementedError
            if imval:
                if squared:
                    if absolute:
                        real.append(mpf_mul(reval,reval))
                        real.append(mpf_mul(imval,imval))
                    else:
                        reval, imval = mpc_pow_int((reval,imval),2,prec+10)
                        real.append(reval)
                        imag.append(imval)
                elif absolute:
                    real.append(mpc_abs((reval,imval), prec))
                else:
                    real.append(reval)
                    imag.append(imval)
            else:
                if squared:
                    reval = mpf_mul(reval, reval)
                elif absolute:
                    reval = mpf_abs(reval)
                real.append(reval)
        s = mpf_sum(real, prec, rnd, absolute)
        if imag:
            s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
        else:
            s = ctx.make_mpf(s)
        return s

    def fdot(ctx, A, B=None, conjugate=False):
        r"""
        Computes the dot product of the iterables `A` and `B`,

        .. math ::

            \sum_{k=0} A_k B_k.

        Alternatively, :func:`~mpmath.fdot` accepts a single iterable of pairs.
        In other words, ``fdot(A,B)`` and ``fdot(zip(A,B))`` are equivalent.
        The elements are automatically converted to mpmath numbers.

        With ``conjugate=True``, the elements in the second vector
        will be conjugated:

        .. math ::

            \sum_{k=0} A_k \overline{B_k}

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> A = [2, 1.5, 3]
            >>> B = [1, -1, 2]
            >>> fdot(A, B)
            mpf('6.5')
            >>> list(zip(A, B))
            [(2, 1), (1.5, -1), (3, 2)]
            >>> fdot(_)
            mpf('6.5')
            >>> A = [2, 1.5, 3j]
            >>> B = [1+j, 3, -1-j]
            >>> fdot(A, B)
            mpc(real='9.5', imag='-1.0')
            >>> fdot(A, B, conjugate=True)
            mpc(real='3.5', imag='-5.0')

        """
        if B is not None:
            A = zip(A, B)
        prec, rnd = ctx._prec_rounding
        real = []
        imag = []
        hasattr_ = hasattr
        types = (ctx.mpf, ctx.mpc)
        for a, b in A:
            if type(a) not in types: a = ctx.convert(a)
            if type(b) not in types: b = ctx.convert(b)
            a_real = hasattr_(a, "_mpf_")
            b_real = hasattr_(b, "_mpf_")
            if a_real and b_real:
                real.append(mpf_mul(a._mpf_, b._mpf_))
                continue
            a_complex = hasattr_(a, "_mpc_")
            b_complex = hasattr_(b, "_mpc_")
            if a_real and b_complex:
                aval = a._mpf_
                bre, bim = b._mpc_
                if conjugate:
                    bim = mpf_neg(bim)
                real.append(mpf_mul(aval, bre))
                imag.append(mpf_mul(aval, bim))
            elif b_real and a_complex:
                are, aim = a._mpc_
                bval = b._mpf_
                real.append(mpf_mul(are, bval))
                imag.append(mpf_mul(aim, bval))
            elif a_complex and b_complex:
                #re, im = mpc_mul(a._mpc_, b._mpc_, prec+20)
                are, aim = a._mpc_
                bre, bim = b._mpc_
                if conjugate:
                    bim = mpf_neg(bim)
                real.append(mpf_mul(are, bre))
                real.append(mpf_neg(mpf_mul(aim, bim)))
                imag.append(mpf_mul(are, bim))
                imag.append(mpf_mul(aim, bre))
            else:
                raise NotImplementedError
        s = mpf_sum(real, prec, rnd)
        if imag:
            s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
        else:
            s = ctx.make_mpf(s)
        return s

    def _wrap_libmp_function(ctx, mpf_f, mpc_f=None, mpi_f=None, doc="<no doc>"):
        """
        Given a low-level mpf_ function, and optionally similar functions
        for mpc_ and mpi_, defines the function as a context method.

        It is assumed that the return type is the same as that of
        the input; the exception is that propagation from mpf to mpc is possible
        by raising ComplexResult.

        """
        def f(x, **kwargs):
            if type(x) not in ctx.types:
                x = ctx.convert(x)
            prec, rounding = ctx._prec_rounding
            if kwargs:
                prec = kwargs.get('prec', prec)
                if 'dps' in kwargs:
                    prec = dps_to_prec(kwargs['dps'])
                rounding = kwargs.get('rounding', rounding)
            if hasattr(x, '_mpf_'):
                try:
                    return ctx.make_mpf(mpf_f(x._mpf_, prec, rounding))
                except ComplexResult:
                    # Handle propagation to complex
                    if ctx.trap_complex:
                        raise
                    return ctx.make_mpc(mpc_f((x._mpf_, fzero), prec, rounding))
            elif hasattr(x, '_mpc_'):
                return ctx.make_mpc(mpc_f(x._mpc_, prec, rounding))
            raise NotImplementedError("%s of a %s" % (name, type(x)))
        name = mpf_f.__name__[4:]
        f.__doc__ = function_docs.__dict__.get(name, "Computes the %s of x" % doc)
        return f

    # Called by SpecialFunctions.__init__()
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
        f_wrapped.__doc__ = function_docs.__dict__.get(name, f.__doc__)
        setattr(cls, name, f_wrapped)

    def _convert_param(ctx, x):
        if hasattr(x, "_mpc_"):
            v, im = x._mpc_
            if im != fzero:
                return x, 'C'
        elif hasattr(x, "_mpf_"):
            v = x._mpf_
        else:
            if type(x) in int_types:
                return int(x), 'Z'
            p = None
            if isinstance(x, tuple):
                p, q = x
            elif hasattr(x, '_mpq_'):
                p, q = x._mpq_
            elif isinstance(x, basestring) and '/' in x:
                p, q = x.split('/')
                p = int(p)
                q = int(q)
            if p is not None:
                if not p % q:
                    return p // q, 'Z'
                return ctx.mpq(p,q), 'Q'
            x = ctx.convert(x)
            if hasattr(x, "_mpc_"):
                v, im = x._mpc_
                if im != fzero:
                    return x, 'C'
            elif hasattr(x, "_mpf_"):
                v = x._mpf_
            else:
                return x, 'U'
        sign, man, exp, bc = v
        if man:
            if exp >= -4:
                if sign:
                    man = -man
                if exp >= 0:
                    return int(man) << exp, 'Z'
                if exp >= -4:
                    p, q = int(man), (1<<(-exp))
                    return ctx.mpq(p,q), 'Q'
            x = ctx.make_mpf(v)
            return x, 'R'
        elif not exp:
            return 0, 'Z'
        else:
            return x, 'U'

    def _mpf_mag(ctx, x):
        sign, man, exp, bc = x
        if man:
            return exp+bc
        if x == fzero:
            return ctx.ninf
        if x == finf or x == fninf:
            return ctx.inf
        return ctx.nan

    def mag(ctx, x):
        """
        Quick logarithmic magnitude estimate of a number. Returns an
        integer or infinity `m` such that `|x| <= 2^m`. It is not
        guaranteed that `m` is an optimal bound, but it will never
        be too large by more than 2 (and probably not more than 1).

        **Examples**

            >>> from mpmath import *
            >>> mp.pretty = True
            >>> mag(10), mag(10.0), mag(mpf(10)), int(ceil(log(10,2)))
            (4, 4, 4, 4)
            >>> mag(10j), mag(10+10j)
            (4, 5)
            >>> mag(0.01), int(ceil(log(0.01,2)))
            (-6, -6)
            >>> mag(0), mag(inf), mag(-inf), mag(nan)
            (-inf, +inf, +inf, nan)

        """
        if hasattr(x, "_mpf_"):
            return ctx._mpf_mag(x._mpf_)
        elif hasattr(x, "_mpc_"):
            r, i = x._mpc_
            if r == fzero:
                return ctx._mpf_mag(i)
            if i == fzero:
                return ctx._mpf_mag(r)
            return 1+max(ctx._mpf_mag(r), ctx._mpf_mag(i))
        elif isinstance(x, int_types):
            if x:
                return bitcount(abs(x))
            return ctx.ninf
        elif isinstance(x, rational.mpq):
            p, q = x._mpq_
            if p:
                return 1 + bitcount(abs(p)) - bitcount(q)
            return ctx.ninf
        else:
            x = ctx.convert(x)
            if hasattr(x, "_mpf_") or hasattr(x, "_mpc_"):
                return ctx.mag(x)
            else:
                raise TypeError("requires an mpf/mpc")


# Register with "numbers" ABC
#     We do not subclass, hence we do not use the @abstractmethod checks. While
#     this is less invasive it may turn out that we do not actually support
#     parts of the expected interfaces.  See
#     http://docs.python.org/2/library/numbers.html for list of abstract
#     methods.
try:
    import numbers
    numbers.Complex.register(_mpc)
    numbers.Real.register(_mpf)
except ImportError:
    pass
