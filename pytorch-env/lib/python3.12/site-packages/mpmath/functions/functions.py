from ..libmp.backend import xrange

class SpecialFunctions(object):
    """
    This class implements special functions using high-level code.

    Elementary and some other functions (e.g. gamma function, basecase
    hypergeometric series) are assumed to be predefined by the context as
    "builtins" or "low-level" functions.
    """
    defined_functions = {}

    # The series for the Jacobi theta functions converge for |q| < 1;
    # in the current implementation they throw a ValueError for
    # abs(q) > THETA_Q_LIM
    THETA_Q_LIM = 1 - 10**-7

    def __init__(self):
        cls = self.__class__
        for name in cls.defined_functions:
            f, wrap = cls.defined_functions[name]
            cls._wrap_specfun(name, f, wrap)

        self.mpq_1 = self._mpq((1,1))
        self.mpq_0 = self._mpq((0,1))
        self.mpq_1_2 = self._mpq((1,2))
        self.mpq_3_2 = self._mpq((3,2))
        self.mpq_1_4 = self._mpq((1,4))
        self.mpq_1_16 = self._mpq((1,16))
        self.mpq_3_16 = self._mpq((3,16))
        self.mpq_5_2 = self._mpq((5,2))
        self.mpq_3_4 = self._mpq((3,4))
        self.mpq_7_4 = self._mpq((7,4))
        self.mpq_5_4 = self._mpq((5,4))
        self.mpq_1_3 = self._mpq((1,3))
        self.mpq_2_3 = self._mpq((2,3))
        self.mpq_4_3 = self._mpq((4,3))
        self.mpq_1_6 = self._mpq((1,6))
        self.mpq_5_6 = self._mpq((5,6))
        self.mpq_5_3 = self._mpq((5,3))

        self._misc_const_cache = {}

        self._aliases.update({
            'phase' : 'arg',
            'conjugate' : 'conj',
            'nthroot' : 'root',
            'polygamma' : 'psi',
            'hurwitz' : 'zeta',
            #'digamma' : 'psi0',
            #'trigamma' : 'psi1',
            #'tetragamma' : 'psi2',
            #'pentagamma' : 'psi3',
            'fibonacci' : 'fib',
            'factorial' : 'fac',
        })

        self.zetazero_memoized = self.memoize(self.zetazero)

    # Default -- do nothing
    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        setattr(cls, name, f)

    # Optional fast versions of common functions in common cases.
    # If not overridden, default (generic hypergeometric series)
    # implementations will be used
    def _besselj(ctx, n, z): raise NotImplementedError
    def _erf(ctx, z): raise NotImplementedError
    def _erfc(ctx, z): raise NotImplementedError
    def _gamma_upper_int(ctx, z, a): raise NotImplementedError
    def _expint_int(ctx, n, z): raise NotImplementedError
    def _zeta(ctx, s): raise NotImplementedError
    def _zetasum_fast(ctx, s, a, n, derivatives, reflect): raise NotImplementedError
    def _ei(ctx, z): raise NotImplementedError
    def _e1(ctx, z): raise NotImplementedError
    def _ci(ctx, z): raise NotImplementedError
    def _si(ctx, z): raise NotImplementedError
    def _altzeta(ctx, s): raise NotImplementedError

def defun_wrapped(f):
    SpecialFunctions.defined_functions[f.__name__] = f, True
    return f

def defun(f):
    SpecialFunctions.defined_functions[f.__name__] = f, False
    return f

def defun_static(f):
    setattr(SpecialFunctions, f.__name__, f)
    return f

@defun_wrapped
def cot(ctx, z): return ctx.one / ctx.tan(z)

@defun_wrapped
def sec(ctx, z): return ctx.one / ctx.cos(z)

@defun_wrapped
def csc(ctx, z): return ctx.one / ctx.sin(z)

@defun_wrapped
def coth(ctx, z): return ctx.one / ctx.tanh(z)

@defun_wrapped
def sech(ctx, z): return ctx.one / ctx.cosh(z)

@defun_wrapped
def csch(ctx, z): return ctx.one / ctx.sinh(z)

@defun_wrapped
def acot(ctx, z):
    if not z:
        return ctx.pi * 0.5
    else:
        return ctx.atan(ctx.one / z)

@defun_wrapped
def asec(ctx, z): return ctx.acos(ctx.one / z)

@defun_wrapped
def acsc(ctx, z): return ctx.asin(ctx.one / z)

@defun_wrapped
def acoth(ctx, z):
    if not z:
        return ctx.pi * 0.5j
    else:
        return ctx.atanh(ctx.one / z)


@defun_wrapped
def asech(ctx, z): return ctx.acosh(ctx.one / z)

@defun_wrapped
def acsch(ctx, z): return ctx.asinh(ctx.one / z)

@defun
def sign(ctx, x):
    x = ctx.convert(x)
    if not x or ctx.isnan(x):
        return x
    if ctx._is_real_type(x):
        if x > 0:
            return ctx.one
        else:
            return -ctx.one
    return x / abs(x)

@defun
def agm(ctx, a, b=1):
    if b == 1:
        return ctx.agm1(a)
    a = ctx.convert(a)
    b = ctx.convert(b)
    return ctx._agm(a, b)

@defun_wrapped
def sinc(ctx, x):
    if ctx.isinf(x):
        return 1/x
    if not x:
        return x+1
    return ctx.sin(x)/x

@defun_wrapped
def sincpi(ctx, x):
    if ctx.isinf(x):
        return 1/x
    if not x:
        return x+1
    return ctx.sinpi(x)/(ctx.pi*x)

# TODO: tests; improve implementation
@defun_wrapped
def expm1(ctx, x):
    if not x:
        return ctx.zero
    # exp(x) - 1 ~ x
    if ctx.mag(x) < -ctx.prec:
        return x + 0.5*x**2
    # TODO: accurately eval the smaller of the real/imag parts
    return ctx.sum_accurately(lambda: iter([ctx.exp(x),-1]),1)

@defun_wrapped
def log1p(ctx, x):
    if not x:
        return ctx.zero
    if ctx.mag(x) < -ctx.prec:
        return x - 0.5*x**2
    return ctx.log(ctx.fadd(1, x, prec=2*ctx.prec))

@defun_wrapped
def powm1(ctx, x, y):
    mag = ctx.mag
    one = ctx.one
    w = x**y - one
    M = mag(w)
    # Only moderate cancellation
    if M > -8:
        return w
    # Check for the only possible exact cases
    if not w:
        if (not y) or (x in (1, -1, 1j, -1j) and ctx.isint(y)):
            return w
    x1 = x - one
    magy = mag(y)
    lnx = ctx.ln(x)
    # Small y: x^y - 1 ~ log(x)*y + O(log(x)^2 * y^2)
    if magy + mag(lnx) < -ctx.prec:
        return lnx*y + (lnx*y)**2/2
    # TODO: accurately eval the smaller of the real/imag part
    return ctx.sum_accurately(lambda: iter([x**y, -1]), 1)

@defun
def _rootof1(ctx, k, n):
    k = int(k)
    n = int(n)
    k %= n
    if not k:
        return ctx.one
    elif 2*k == n:
        return -ctx.one
    elif 4*k == n:
        return ctx.j
    elif 4*k == 3*n:
        return -ctx.j
    return ctx.expjpi(2*ctx.mpf(k)/n)

@defun
def root(ctx, x, n, k=0):
    n = int(n)
    x = ctx.convert(x)
    if k:
        # Special case: there is an exact real root
        if (n & 1 and 2*k == n-1) and (not ctx.im(x)) and (ctx.re(x) < 0):
            return -ctx.root(-x, n)
        # Multiply by root of unity
        prec = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.root(x, n, 0) * ctx._rootof1(k, n)
        finally:
            ctx.prec = prec
        return +v
    return ctx._nthroot(x, n)

@defun
def unitroots(ctx, n, primitive=False):
    gcd = ctx._gcd
    prec = ctx.prec
    try:
        ctx.prec += 10
        if primitive:
            v = [ctx._rootof1(k,n) for k in range(n) if gcd(k,n) == 1]
        else:
            # TODO: this can be done *much* faster
            v = [ctx._rootof1(k,n) for k in range(n)]
    finally:
        ctx.prec = prec
    return [+x for x in v]

@defun
def arg(ctx, x):
    x = ctx.convert(x)
    re = ctx._re(x)
    im = ctx._im(x)
    return ctx.atan2(im, re)

@defun
def fabs(ctx, x):
    return abs(ctx.convert(x))

@defun
def re(ctx, x):
    x = ctx.convert(x)
    if hasattr(x, "real"):    # py2.5 doesn't have .real/.imag for all numbers
        return x.real
    return x

@defun
def im(ctx, x):
    x = ctx.convert(x)
    if hasattr(x, "imag"):    # py2.5 doesn't have .real/.imag for all numbers
        return x.imag
    return ctx.zero

@defun
def conj(ctx, x):
    x = ctx.convert(x)
    try:
        return x.conjugate()
    except AttributeError:
        return x

@defun
def polar(ctx, z):
    return (ctx.fabs(z), ctx.arg(z))

@defun_wrapped
def rect(ctx, r, phi):
    return r * ctx.mpc(*ctx.cos_sin(phi))

@defun
def log(ctx, x, b=None):
    if b is None:
        return ctx.ln(x)
    wp = ctx.prec + 20
    return ctx.ln(x, prec=wp) / ctx.ln(b, prec=wp)

@defun
def log10(ctx, x):
    return ctx.log(x, 10)

@defun
def fmod(ctx, x, y):
    return ctx.convert(x) % ctx.convert(y)

@defun
def degrees(ctx, x):
    return x / ctx.degree

@defun
def radians(ctx, x):
    return x * ctx.degree

def _lambertw_special(ctx, z, k):
    # W(0,0) = 0; all other branches are singular
    if not z:
        if not k:
            return z
        return ctx.ninf + z
    if z == ctx.inf:
        if k == 0:
            return z
        else:
            return z + 2*k*ctx.pi*ctx.j
    if z == ctx.ninf:
        return (-z) + (2*k+1)*ctx.pi*ctx.j
    # Some kind of nan or complex inf/nan?
    return ctx.ln(z)

import math
import cmath

def _lambertw_approx_hybrid(z, k):
    imag_sign = 0
    if hasattr(z, "imag"):
        x = float(z.real)
        y = z.imag
        if y:
            imag_sign = (-1) ** (y < 0)
        y = float(y)
    else:
        x = float(z)
        y = 0.0
        imag_sign = 0
    # hack to work regardless of whether Python supports -0.0
    if not y:
        y = 0.0
    z = complex(x,y)
    if k == 0:
        if -4.0 < y < 4.0 and -1.0 < x < 2.5:
            if imag_sign:
                # Taylor series in upper/lower half-plane
                if y > 1.00: return (0.876+0.645j) + (0.118-0.174j)*(z-(0.75+2.5j))
                if y > 0.25: return (0.505+0.204j) + (0.375-0.132j)*(z-(0.75+0.5j))
                if y < -1.00: return (0.876-0.645j) + (0.118+0.174j)*(z-(0.75-2.5j))
                if y < -0.25: return (0.505-0.204j) + (0.375+0.132j)*(z-(0.75-0.5j))
            # Taylor series near -1
            if x < -0.5:
                if imag_sign >= 0:
                    return (-0.318+1.34j) + (-0.697-0.593j)*(z+1)
                else:
                    return (-0.318-1.34j) + (-0.697+0.593j)*(z+1)
            # return real type
            r = -0.367879441171442
            if (not imag_sign) and x > r:
                z = x
            # Singularity near -1/e
            if x < -0.2:
                return -1 + 2.33164398159712*(z-r)**0.5 - 1.81218788563936*(z-r)
            # Taylor series near 0
            if x < 0.5: return z
            # Simple linear approximation
            return 0.2 + 0.3*z
        if (not imag_sign) and x > 0.0:
            L1 = math.log(x); L2 = math.log(L1)
        else:
            L1 = cmath.log(z); L2 = cmath.log(L1)
    elif k == -1:
        # return real type
        r = -0.367879441171442
        if (not imag_sign) and r < x < 0.0:
            z = x
        if (imag_sign >= 0) and y < 0.1 and -0.6 < x < -0.2:
            return -1 - 2.33164398159712*(z-r)**0.5 - 1.81218788563936*(z-r)
        if (not imag_sign) and -0.2 <= x < 0.0:
            L1 = math.log(-x)
            return L1 - math.log(-L1)
        else:
            if imag_sign == -1 and (not y) and x < 0.0:
                L1 = cmath.log(z) - 3.1415926535897932j
            else:
                L1 = cmath.log(z) - 6.2831853071795865j
            L2 = cmath.log(L1)
    return L1 - L2 + L2/L1 + L2*(L2-2)/(2*L1**2)

def _lambertw_series(ctx, z, k, tol):
    """
    Return rough approximation for W_k(z) from an asymptotic series,
    sufficiently accurate for the Halley iteration to converge to
    the correct value.
    """
    magz = ctx.mag(z)
    if (-10 < magz < 900) and (-1000 < k < 1000):
        # Near the branch point at -1/e
        if magz < 1 and abs(z+0.36787944117144) < 0.05:
            if k == 0 or (k == -1 and ctx._im(z) >= 0) or \
                         (k == 1  and ctx._im(z) < 0):
                delta = ctx.sum_accurately(lambda: [z, ctx.exp(-1)])
                cancellation = -ctx.mag(delta)
                ctx.prec += cancellation
                # Use series given in Corless et al.
                p = ctx.sqrt(2*(ctx.e*z+1))
                ctx.prec -= cancellation
                u = {0:ctx.mpf(-1), 1:ctx.mpf(1)}
                a = {0:ctx.mpf(2), 1:ctx.mpf(-1)}
                if k != 0:
                    p = -p
                s = ctx.zero
                # The series converges, so we could use it directly, but unless
                # *extremely* close, it is better to just use the first few
                # terms to get a good approximation for the iteration
                for l in xrange(max(2,cancellation)):
                    if l not in u:
                        a[l] = ctx.fsum(u[j]*u[l+1-j] for j in xrange(2,l))
                        u[l] = (l-1)*(u[l-2]/2+a[l-2]/4)/(l+1)-a[l]/2-u[l-1]/(l+1)
                    term = u[l] * p**l
                    s += term
                    if ctx.mag(term) < -tol:
                        return s, True
                    l += 1
                ctx.prec += cancellation//2
                return s, False
        if k == 0 or k == -1:
            return _lambertw_approx_hybrid(z, k), False
    if k == 0:
        if magz < -1:
            return z*(1-z), False
        L1 = ctx.ln(z)
        L2 = ctx.ln(L1)
    elif k == -1 and (not ctx._im(z)) and (-0.36787944117144 < ctx._re(z) < 0):
        L1 = ctx.ln(-z)
        return L1 - ctx.ln(-L1), False
    else:
        # This holds both as z -> 0 and z -> inf.
        # Relative error is O(1/log(z)).
        L1 = ctx.ln(z) + 2j*ctx.pi*k
        L2 = ctx.ln(L1)
    return L1 - L2 + L2/L1 + L2*(L2-2)/(2*L1**2), False

@defun
def lambertw(ctx, z, k=0):
    z = ctx.convert(z)
    k = int(k)
    if not ctx.isnormal(z):
        return _lambertw_special(ctx, z, k)
    prec = ctx.prec
    ctx.prec += 20 + ctx.mag(k or 1)
    wp = ctx.prec
    tol = wp - 5
    w, done = _lambertw_series(ctx, z, k, tol)
    if not done:
        # Use Halley iteration to solve w*exp(w) = z
        two = ctx.mpf(2)
        for i in xrange(100):
            ew = ctx.exp(w)
            wew = w*ew
            wewz = wew-z
            wn = w - wewz/(wew+ew-(w+two)*wewz/(two*w+two))
            if ctx.mag(wn-w) <= ctx.mag(wn) - tol:
                w = wn
                break
            else:
                w = wn
        if i == 100:
            ctx.warn("Lambert W iteration failed to converge for z = %s" % z)
    ctx.prec = prec
    return +w

@defun_wrapped
def bell(ctx, n, x=1):
    x = ctx.convert(x)
    if not n:
        if ctx.isnan(x):
            return x
        return type(x)(1)
    if ctx.isinf(x) or ctx.isinf(n) or ctx.isnan(x) or ctx.isnan(n):
        return x**n
    if n == 1: return x
    if n == 2: return x*(x+1)
    if x == 0: return ctx.sincpi(n)
    return _polyexp(ctx, n, x, True) / ctx.exp(x)

def _polyexp(ctx, n, x, extra=False):
    def _terms():
        if extra:
            yield ctx.sincpi(n)
        t = x
        k = 1
        while 1:
            yield k**n * t
            k += 1
            t = t*x/k
    return ctx.sum_accurately(_terms, check_step=4)

@defun_wrapped
def polyexp(ctx, s, z):
    if ctx.isinf(z) or ctx.isinf(s) or ctx.isnan(z) or ctx.isnan(s):
        return z**s
    if z == 0: return z*s
    if s == 0: return ctx.expm1(z)
    if s == 1: return ctx.exp(z)*z
    if s == 2: return ctx.exp(z)*z*(z+1)
    return _polyexp(ctx, s, z)

@defun_wrapped
def cyclotomic(ctx, n, z):
    n = int(n)
    if n < 0:
        raise ValueError("n cannot be negative")
    p = ctx.one
    if n == 0:
        return p
    if n == 1:
        return z - p
    if n == 2:
        return z + p
    # Use divisor product representation. Unfortunately, this sometimes
    # includes singularities for roots of unity, which we have to cancel out.
    # Matching zeros/poles pairwise, we have (1-z^a)/(1-z^b) ~ a/b + O(z-1).
    a_prod = 1
    b_prod = 1
    num_zeros = 0
    num_poles = 0
    for d in range(1,n+1):
        if not n % d:
            w = ctx.moebius(n//d)
            # Use powm1 because it is important that we get 0 only
            # if it really is exactly 0
            b = -ctx.powm1(z, d)
            if b:
                p *= b**w
            else:
                if w == 1:
                    a_prod *= d
                    num_zeros += 1
                elif w == -1:
                    b_prod *= d
                    num_poles += 1
    #print n, num_zeros, num_poles
    if num_zeros:
        if num_zeros > num_poles:
            p *= 0
        else:
            p *= a_prod
            p /= b_prod
    return p

@defun
def mangoldt(ctx, n):
    r"""
    Evaluates the von Mangoldt function `\Lambda(n) = \log p`
    if `n = p^k` a power of a prime, and `\Lambda(n) = 0` otherwise.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> [mangoldt(n) for n in range(-2,3)]
        [0.0, 0.0, 0.0, 0.0, 0.6931471805599453094172321]
        >>> mangoldt(6)
        0.0
        >>> mangoldt(7)
        1.945910149055313305105353
        >>> mangoldt(8)
        0.6931471805599453094172321
        >>> fsum(mangoldt(n) for n in range(101))
        94.04531122935739224600493
        >>> fsum(mangoldt(n) for n in range(10001))
        10013.39669326311478372032

    """
    n = int(n)
    if n < 2:
        return ctx.zero
    if n % 2 == 0:
        # Must be a power of two
        if n & (n-1) == 0:
            return +ctx.ln2
        else:
            return ctx.zero
    # TODO: the following could be generalized into a perfect
    # power testing function
    # ---
    # Look for a small factor
    for p in (3,5,7,11,13,17,19,23,29,31):
        if not n % p:
            q, r = n // p, 0
            while q > 1:
                q, r = divmod(q, p)
                if r:
                    return ctx.zero
            return ctx.ln(p)
    if ctx.isprime(n):
        return ctx.ln(n)
    # Obviously, we could use arbitrary-precision arithmetic for this...
    if n > 10**30:
        raise NotImplementedError
    k = 2
    while 1:
        p = int(n**(1./k) + 0.5)
        if p < 2:
            return ctx.zero
        if p ** k == n:
            if ctx.isprime(p):
                return ctx.ln(p)
        k += 1

@defun
def stirling1(ctx, n, k, exact=False):
    v = ctx._stirling1(int(n), int(k))
    if exact:
        return int(v)
    else:
        return ctx.mpf(v)

@defun
def stirling2(ctx, n, k, exact=False):
    v = ctx._stirling2(int(n), int(k))
    if exact:
        return int(v)
    else:
        return ctx.mpf(v)
