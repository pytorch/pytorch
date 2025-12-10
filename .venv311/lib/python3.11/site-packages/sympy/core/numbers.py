from __future__ import annotations

from typing import overload

import numbers
import decimal
import fractions
import math

from .containers import Tuple
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
              _sympify, _is_numpy_instance)
from .singleton import S, Singleton
from .basic import Basic
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .cache import cacheit, clear_cache
from .decorators import _sympifyit
from .intfunc import num_digits, igcd, ilcm, mod_inverse, integer_nthroot
from .logic import fuzzy_not
from .kind import NumberKind
from .sorting import ordered
from sympy.external.gmpy import SYMPY_INTS, gmpy, flint
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount, round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp_python import mpnumeric
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
    prec_to_dps, dps_to_prec)
from sympy.utilities.misc import debug
from sympy.utilities.exceptions import sympy_deprecation_warning
from .parameters import global_parameters

_LOG2 = math.log(2)


def comp(z1, z2, tol=None):
    r"""Return a bool indicating whether the error between z1 and z2
    is $\le$ ``tol``.

    Examples
    ========

    If ``tol`` is ``None`` then ``True`` will be returned if
    :math:`|z1 - z2|\times 10^p \le 5` where $p$ is minimum value of the
    decimal precision of each value.

    >>> from sympy import comp, pi
    >>> pi4 = pi.n(4); pi4
    3.142
    >>> comp(_, 3.142)
    True
    >>> comp(pi4, 3.141)
    False
    >>> comp(pi4, 3.143)
    False

    A comparison of strings will be made
    if ``z1`` is a Number and ``z2`` is a string or ``tol`` is ''.

    >>> comp(pi4, 3.1415)
    True
    >>> comp(pi4, 3.1415, '')
    False

    When ``tol`` is provided and $z2$ is non-zero and
    :math:`|z1| > 1` the error is normalized by :math:`|z1|`:

    >>> abs(pi4 - 3.14)/pi4
    0.000509791731426756
    >>> comp(pi4, 3.14, .001)  # difference less than 0.1%
    True
    >>> comp(pi4, 3.14, .0005)  # difference less than 0.1%
    False

    When :math:`|z1| \le 1` the absolute error is used:

    >>> 1/pi4
    0.3183
    >>> abs(1/pi4 - 0.3183)/(1/pi4)
    3.07371499106316e-5
    >>> abs(1/pi4 - 0.3183)
    9.78393554684764e-6
    >>> comp(1/pi4, 0.3183, 1e-5)
    True

    To see if the absolute error between ``z1`` and ``z2`` is less
    than or equal to ``tol``, call this as ``comp(z1 - z2, 0, tol)``
    or ``comp(z1 - z2, tol=tol)``:

    >>> abs(pi4 - 3.14)
    0.00160156249999988
    >>> comp(pi4 - 3.14, 0, .002)
    True
    >>> comp(pi4 - 3.14, 0, .001)
    False
    """
    if isinstance(z2, str):
        if not pure_complex(z1, or_real=True):
            raise ValueError('when z2 is a str z1 must be a Number')
        return str(z1) == z2
    if not z1:
        z1, z2 = z2, z1
    if not z1:
        return True
    if not tol:
        a, b = z1, z2
        if tol == '':
            return str(a) == str(b)
        if tol is None:
            a, b = sympify(a), sympify(b)
            if not all(i.is_number for i in (a, b)):
                raise ValueError('expecting 2 numbers')
            fa = a.atoms(Float)
            fb = b.atoms(Float)
            if not fa and not fb:
                # no floats -- compare exactly
                return a == b
            # get a to be pure_complex
            for _ in range(2):
                ca = pure_complex(a, or_real=True)
                if not ca:
                    if fa:
                        a = a.n(prec_to_dps(min(i._prec for i in fa)))
                        ca = pure_complex(a, or_real=True)
                        break
                    else:
                        fa, fb = fb, fa
                        a, b = b, a
            cb = pure_complex(b)
            if not cb and fb:
                b = b.n(prec_to_dps(min(i._prec for i in fb)))
                cb = pure_complex(b, or_real=True)
            if ca and cb and (ca[1] or cb[1]):
                return all(comp(i, j) for i, j in zip(ca, cb))
            tol = 10**prec_to_dps(min(a._prec, getattr(b, '_prec', a._prec)))
            return int(abs(a - b)*tol) <= 5
    diff = abs(z1 - z2)
    az1 = abs(z1)
    if z2 and az1 > 1:
        return diff/az1 <= tol
    else:
        return diff <= tol


def mpf_norm(mpf, prec):
    """Return the mpf tuple normalized appropriately for the indicated
    precision after doing a check to see if zero should be returned or
    not when the mantissa is 0. ``mpf_normlize`` always assumes that this
    is zero, but it may not be since the mantissa for mpf's values "+inf",
    "-inf" and "nan" have a mantissa of zero, too.

    Note: this is not intended to validate a given mpf tuple, so sending
    mpf tuples that were not created by mpmath may produce bad results. This
    is only a wrapper to ``mpf_normalize`` which provides the check for non-
    zero mpfs that have a 0 for the mantissa.
    """
    sign, man, expt, bc = mpf
    if not man:
        # hack for mpf_normalize which does not do this;
        # it assumes that if man is zero the result is 0
        # (see issue 6639)
        if not bc:
            return fzero
        else:
            # don't change anything; this should already
            # be a well formed mpf tuple
            return mpf

    # Necessary if mpmath is using the gmpy backend
    from mpmath.libmp.backend import MPZ
    rv = mpf_normalize(sign, MPZ(man), expt, bc, prec, rnd)
    return rv

# TODO: we should use the warnings module
_errdict = {"divide": False}


def seterr(divide=False):
    """
    Should SymPy raise an exception on 0/0 or return a nan?

    divide == True .... raise an exception
    divide == False ... return nan
    """
    if _errdict["divide"] != divide:
        clear_cache()
        _errdict["divide"] = divide


def _as_integer_ratio(p):
    neg_pow, man, expt, _ = getattr(p, '_mpf_', mpmath.mpf(p)._mpf_)
    p = [1, -1][neg_pow % 2]*man
    if expt < 0:
        q = 2**-expt
    else:
        q = 1
        p *= 2**expt
    return int(p), int(q)


def _decimal_to_Rational_prec(dec):
    """Convert an ordinary decimal instance to a Rational."""
    if not dec.is_finite():
        raise TypeError("dec must be finite, got %s." % dec)
    s, d, e = dec.as_tuple()
    prec = len(d)
    if e >= 0:  # it's an integer
        rv = Integer(int(dec))
    else:
        s = (-1)**s
        d = sum(di*10**i for i, di in enumerate(reversed(d)))
        rv = Rational(s*d, 10**-e)
    return rv, prec

_dig = str.maketrans(dict.fromkeys('1234567890'))

def _literal_float(s):
    """return True if s is space-trimmed number literal else False

    Python allows underscore as digit separators: there must be a
    digit on each side. So neither a leading underscore nor a
    double underscore are valid as part of a number. A number does
    not have to precede the decimal point, but there must be a
    digit before the optional "e" or "E" that begins the signs
    exponent of the number which must be an integer, perhaps with
    underscore separators.

    SymPy allows space as a separator; if the calling routine replaces
    them with underscores then the same semantics will be enforced
    for them as for underscores: there can only be 1 *between* digits.

    We don't check for error from float(s) because we don't know
    whether s is malicious or not. A regex for this could maybe
    be written but will it be understood by most who read it?
    """
    # mantissa and exponent
    parts = s.split('e')
    if len(parts) > 2:
        return False
    if len(parts) == 2:
        m, e = parts
        if e.startswith(tuple('+-')):
            e = e[1:]
        if not e:
            return False
    else:
        m, e = s, '1'
    # integer and fraction of mantissa
    parts = m.split('.')
    if len(parts) > 2:
        return False
    elif len(parts) == 2:
        i, f = parts
    else:
        i, f = m, '1'
    if not i and not f:
        return False
    if i and i[0] in '+-':
        i = i[1:]
    if not i:  # -.3e4 -> -0.3e4
        i = '1'
    f = f or '1'
    # check that all groups contain only digits and are not null
    for n in (i, f, e):
        for g in n.split('_'):
            if not g or g.translate(_dig):
                return False
    return True

# (a,b) -> gcd(a,b)

# TODO caching with decorator, but not to degrade performance


class Number(AtomicExpr):
    """Represents atomic numbers in SymPy.

    Explanation
    ===========

    Floating point numbers are represented by the Float class.
    Rational numbers (of any size) are represented by the Rational class.
    Integer numbers (of any size) are represented by the Integer class.
    Float and Rational are subclasses of Number; Integer is a subclass
    of Rational.

    For example, ``2/3`` is represented as ``Rational(2, 3)`` which is
    a different object from the floating point number obtained with
    Python division ``2/3``. Even for numbers that are exactly
    represented in binary, there is a difference between how two forms,
    such as ``Rational(1, 2)`` and ``Float(0.5)``, are used in SymPy.
    The rational form is to be preferred in symbolic computations.

    Other kinds of numbers, such as algebraic numbers ``sqrt(2)`` or
    complex numbers ``3 + 4*I``, are not instances of Number class as
    they are not atomic.

    See Also
    ========

    Float, Integer, Rational
    """
    is_commutative = True
    is_number = True
    is_Number = True

    __slots__ = ()

    # Used to make max(x._prec, y._prec) return x._prec when only x is a float
    _prec = -1

    kind = NumberKind

    def __new__(cls, *obj):
        if len(obj) == 1:
            obj = obj[0]

        if isinstance(obj, Number):
            return obj
        if isinstance(obj, SYMPY_INTS):
            return Integer(obj)
        if isinstance(obj, tuple) and len(obj) == 2:
            return Rational(*obj)
        if isinstance(obj, (float, mpmath.mpf, decimal.Decimal)):
            return Float(obj)
        if isinstance(obj, str):
            _obj = obj.lower()  # float('INF') == float('inf')
            if _obj == 'nan':
                return S.NaN
            elif _obj == 'inf':
                return S.Infinity
            elif _obj == '+inf':
                return S.Infinity
            elif _obj == '-inf':
                return S.NegativeInfinity
            val = sympify(obj)
            if isinstance(val, Number):
                return val
            else:
                raise ValueError('String "%s" does not denote a Number' % obj)
        msg = "expected str|int|long|float|Decimal|Number object but got %r"
        raise TypeError(msg % type(obj).__name__)

    def could_extract_minus_sign(self):
        return bool(self.is_extended_negative)

    def invert(self, other, *gens, **args):
        from sympy.polys.polytools import invert
        if getattr(other, 'is_number', True):
            return mod_inverse(self, other)
        return invert(self, other, *gens, **args)

    def __divmod__(self, other):
        from sympy.functions.elementary.complexes import sign

        try:
            other = Number(other)
            if self.is_infinite or S.NaN in (self, other):
                return (S.NaN, S.NaN)
        except TypeError:
            return NotImplemented
        if not other:
            raise ZeroDivisionError('modulo by zero')
        if self.is_Integer and other.is_Integer:
            return Tuple(*divmod(self.p, other.p))
        elif isinstance(other, Float):
            rat = self/Rational(other)
        else:
            rat = self/other
        if other.is_finite:
            w = int(rat) if rat >= 0 else int(rat) - 1
            r = self - other*w
            if r == Float(other):
                w += 1
                r = 0
            if isinstance(self, Float) or isinstance(other, Float):
                r = Float(r)  # in case w or r is 0
        else:
            w = 0 if not self or (sign(self) == sign(other)) else -1
            r = other if w else self
        return Tuple(w, r)

    def __rdivmod__(self, other):
        try:
            other = Number(other)
        except TypeError:
            return NotImplemented
        return divmod(other, self)

    def _as_mpf_val(self, prec):
        """Evaluation of mpf tuple accurate to at least prec bits."""
        raise NotImplementedError('%s needs ._as_mpf_val() method' %
            (self.__class__.__name__))

    def _eval_evalf(self, prec):
        return Float._new(self._as_mpf_val(prec), prec)

    def _as_mpf_op(self, prec):
        prec = max(prec, self._prec)
        return self._as_mpf_val(prec), prec

    def __float__(self):
        return mlib.to_float(self._as_mpf_val(53))

    def floor(self):
        raise NotImplementedError('%s needs .floor() method' %
            (self.__class__.__name__))

    def ceiling(self):
        raise NotImplementedError('%s needs .ceiling() method' %
            (self.__class__.__name__))

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    def _eval_conjugate(self):
        return self

    def _eval_order(self, *symbols):
        from sympy.series.order import Order
        # Order(5, x, y) -> Order(1,x,y)
        return Order(S.One, *symbols)

    def _eval_subs(self, old, new):
        if old == -self:
            return -new
        return self  # there is no other possibility

    @classmethod
    def class_key(cls):
        return 1, 0, 'Number'

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (0, ()), (), self

    def __neg__(self) -> Number:
        raise NotImplementedError

    @overload
    def __add__(self, other: Number | int | float) -> Number: ...
    @overload
    def __add__(self, other: Expr) -> Expr: ...

    @_sympifyit('other', NotImplemented)
    def __add__(self, other) -> Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.Infinity
            elif other is S.NegativeInfinity:
                return S.NegativeInfinity
        return AtomicExpr.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.NegativeInfinity
            elif other is S.NegativeInfinity:
                return S.Infinity
        return AtomicExpr.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                if self.is_zero:
                    return S.NaN
                elif self.is_positive:
                    return S.Infinity
                else:
                    return S.NegativeInfinity
            elif other is S.NegativeInfinity:
                if self.is_zero:
                    return S.NaN
                elif self.is_positive:
                    return S.NegativeInfinity
                else:
                    return S.Infinity
        elif isinstance(other, Tuple):
            return NotImplemented
        return AtomicExpr.__mul__(self, other)

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other in (S.Infinity, S.NegativeInfinity):
                return S.Zero
        return AtomicExpr.__truediv__(self, other)

    def __eq__(self, other):
        raise NotImplementedError('%s needs .__eq__() method' %
            (self.__class__.__name__))

    def __ne__(self, other):
        raise NotImplementedError('%s needs .__ne__() method' %
            (self.__class__.__name__))

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        raise NotImplementedError('%s needs .__lt__() method' %
            (self.__class__.__name__))

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        raise NotImplementedError('%s needs .__le__() method' %
            (self.__class__.__name__))

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        return _sympify(other).__lt__(self)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        return _sympify(other).__le__(self)

    def __hash__(self):
        return super().__hash__()

    def is_constant(self, *wrt, **flags):
        return True

    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        # a -> c*t
        if self.is_Rational or not rational:
            return self, ()
        elif self.is_negative:
            return S.NegativeOne, (-self,)
        return S.One, (self,)

    def as_coeff_add(self, *deps):
        # a -> c + t
        if self.is_Rational:
            return self, ()
        return S.Zero, (self,)

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        if not rational:
            return self, S.One
        return S.One, self

    def as_coeff_Add(self, rational=False):
        """Efficiently extract the coefficient of a summation."""
        if not rational:
            return self, S.Zero
        return S.Zero, self

    def gcd(self, other):
        """Compute GCD of `self` and `other`. """
        from sympy.polys.polytools import gcd
        return gcd(self, other)

    def lcm(self, other):
        """Compute LCM of `self` and `other`. """
        from sympy.polys.polytools import lcm
        return lcm(self, other)

    def cofactors(self, other):
        """Compute GCD and cofactors of `self` and `other`. """
        from sympy.polys.polytools import cofactors
        return cofactors(self, other)


class Float(Number):
    """Represent a floating-point number of arbitrary precision.

    Examples
    ========

    >>> from sympy import Float
    >>> Float(3.5)
    3.50000000000000
    >>> Float(3)
    3.00000000000000

    Creating Floats from strings (and Python ``int`` and ``long``
    types) will give a minimum precision of 15 digits, but the
    precision will automatically increase to capture all digits
    entered.

    >>> Float(1)
    1.00000000000000
    >>> Float(10**20)
    100000000000000000000.
    >>> Float('1e20')
    100000000000000000000.

    However, *floating-point* numbers (Python ``float`` types) retain
    only 15 digits of precision:

    >>> Float(1e20)
    1.00000000000000e+20
    >>> Float(1.23456789123456789)
    1.23456789123457

    It may be preferable to enter high-precision decimal numbers
    as strings:

    >>> Float('1.23456789123456789')
    1.23456789123456789

    The desired number of digits can also be specified:

    >>> Float('1e-3', 3)
    0.00100
    >>> Float(100, 4)
    100.0

    Float can automatically count significant figures if a null string
    is sent for the precision; spaces or underscores are also allowed. (Auto-
    counting is only allowed for strings, ints and longs).

    >>> Float('123 456 789.123_456', '')
    123456789.123456
    >>> Float('12e-3', '')
    0.012
    >>> Float(3, '')
    3.

    If a number is written in scientific notation, only the digits before the
    exponent are considered significant if a decimal appears, otherwise the
    "e" signifies only how to move the decimal:

    >>> Float('60.e2', '')  # 2 digits significant
    6.0e+3
    >>> Float('60e2', '')  # 4 digits significant
    6000.
    >>> Float('600e-2', '')  # 3 digits significant
    6.00

    Notes
    =====

    Floats are inexact by their nature unless their value is a binary-exact
    value.

    >>> approx, exact = Float(.1, 1), Float(.125, 1)

    For calculation purposes, evalf needs to be able to change the precision
    but this will not increase the accuracy of the inexact value. The
    following is the most accurate 5-digit approximation of a value of 0.1
    that had only 1 digit of precision:

    >>> approx.evalf(5)
    0.099609

    By contrast, 0.125 is exact in binary (as it is in base 10) and so it
    can be passed to Float or evalf to obtain an arbitrary precision with
    matching accuracy:

    >>> Float(exact, 5)
    0.12500
    >>> exact.evalf(20)
    0.12500000000000000000

    Trying to make a high-precision Float from a float is not disallowed,
    but one must keep in mind that the *underlying float* (not the apparent
    decimal value) is being obtained with high precision. For example, 0.3
    does not have a finite binary representation. The closest rational is
    the fraction 5404319552844595/2**54. So if you try to obtain a Float of
    0.3 to 20 digits of precision you will not see the same thing as 0.3
    followed by 19 zeros:

    >>> Float(0.3, 20)
    0.29999999999999998890

    If you want a 20-digit value of the decimal 0.3 (not the floating point
    approximation of 0.3) you should send the 0.3 as a string. The underlying
    representation is still binary but a higher precision than Python's float
    is used:

    >>> Float('0.3', 20)
    0.30000000000000000000

    Although you can increase the precision of an existing Float using Float
    it will not increase the accuracy -- the underlying value is not changed:

    >>> def show(f): # binary rep of Float
    ...     from sympy import Mul, Pow
    ...     s, m, e, b = f._mpf_
    ...     v = Mul(int(m), Pow(2, int(e), evaluate=False), evaluate=False)
    ...     print('%s at prec=%s' % (v, f._prec))
    ...
    >>> t = Float('0.3', 3)
    >>> show(t)
    4915/2**14 at prec=13
    >>> show(Float(t, 20)) # higher prec, not higher accuracy
    4915/2**14 at prec=70
    >>> show(Float(t, 2)) # lower prec
    307/2**10 at prec=10

    The same thing happens when evalf is used on a Float:

    >>> show(t.evalf(20))
    4915/2**14 at prec=70
    >>> show(t.evalf(2))
    307/2**10 at prec=10

    Finally, Floats can be instantiated with an mpf tuple (n, c, p) to
    produce the number (-1)**n*c*2**p:

    >>> n, c, p = 1, 5, 0
    >>> (-1)**n*c*2**p
    -5
    >>> Float((1, 5, 0))
    -5.00000000000000

    An actual mpf tuple also contains the number of bits in c as the last
    element of the tuple:

    >>> _._mpf_
    (1, 5, 0, 3)

    This is not needed for instantiation and is not the same thing as the
    precision. The mpf tuple and the precision are two separate quantities
    that Float tracks.

    In SymPy, a Float is a number that can be computed with arbitrary
    precision. Although floating point 'inf' and 'nan' are not such
    numbers, Float can create these numbers:

    >>> Float('-inf')
    -oo
    >>> _.is_Float
    False

    Zero in Float only has a single value. Values are not separate for
    positive and negative zeroes.
    """
    __slots__ = ('_mpf_', '_prec')

    _mpf_: tuple[int, int, int, int]

    # A Float, though rational in form, does not behave like
    # a rational in all Python expressions so we deal with
    # exceptions (where we want to deal with the rational
    # form of the Float as a rational) at the source rather
    # than assigning a mathematically loaded category of 'rational'
    is_rational = None
    is_irrational = None
    is_number = True

    is_real = True
    is_extended_real = True

    is_Float = True

    _remove_non_digits = str.maketrans(dict.fromkeys("-+_."))

    def __new__(cls, num, dps=None, precision=None):
        if dps is not None and precision is not None:
            raise ValueError('Both decimal and binary precision supplied. '
                             'Supply only one. ')

        if isinstance(num, str):
            _num = num = num.strip()  # Python ignores leading and trailing space
            num = num.replace(' ', '_').lower()  # Float treats spaces as digit sep; E -> e
            if num.startswith('.') and len(num) > 1:
                num = '0' + num
            elif num.startswith('-.') and len(num) > 2:
                num = '-0.' + num[2:]
            elif num in ('inf', '+inf'):
                return S.Infinity
            elif num == '-inf':
                return S.NegativeInfinity
            elif num == 'nan':
                return S.NaN
            elif not _literal_float(num):
                raise ValueError('string-float not recognized: %s' % _num)
        elif isinstance(num, float) and num == 0:
            num = '0'
        elif isinstance(num, float) and num == float('inf'):
            return S.Infinity
        elif isinstance(num, float) and num == float('-inf'):
            return S.NegativeInfinity
        elif isinstance(num, float) and math.isnan(num):
            return S.NaN
        elif isinstance(num, (SYMPY_INTS, Integer)):
            num = str(num)
        elif num is S.Infinity:
            return num
        elif num is S.NegativeInfinity:
            return num
        elif num is S.NaN:
            return num
        elif _is_numpy_instance(num):  # support for numpy datatypes
            num = _convert_numpy_types(num)
        elif isinstance(num, mpmath.mpf):
            if precision is None:
                if dps is None:
                    precision = num.context.prec
            num = num._mpf_

        if dps is None and precision is None:
            dps = 15
            if isinstance(num, Float):
                return num
            if isinstance(num, str):
                try:
                    Num = decimal.Decimal(num)
                except decimal.InvalidOperation:
                    pass
                else:
                    isint = '.' not in num
                    num, dps = _decimal_to_Rational_prec(Num)
                    if num.is_Integer and isint:
                        # 12e3 is shorthand for int, not float;
                        # 12.e3 would be the float version
                        dps = max(dps, num_digits(num))
                    dps = max(15, dps)
                    precision = dps_to_prec(dps)
        elif precision == '' and dps is None or precision is None and dps == '':
            if not isinstance(num, str):
                raise ValueError('The null string can only be used when '
                'the number to Float is passed as a string or an integer.')
            try:
                Num = decimal.Decimal(num)
            except decimal.InvalidOperation:
                raise ValueError('string-float not recognized by Decimal: %s' % num)
            else:
                isint = '.' not in num
                num, dps = _decimal_to_Rational_prec(Num)
                if num.is_Integer and isint:
                    # without dec, e-notation is short for int
                    dps = max(dps, num_digits(num))
                    precision = dps_to_prec(dps)

        # decimal precision(dps) is set and maybe binary precision(precision)
        # as well.From here on binary precision is used to compute the Float.
        # Hence, if supplied use binary precision else translate from decimal
        # precision.

        if precision is None or precision == '':
            precision = dps_to_prec(dps)

        precision = int(precision)

        if isinstance(num, float):
            _mpf_ = mlib.from_float(num, precision, rnd)
        elif isinstance(num, str):
            _mpf_ = mlib.from_str(num, precision, rnd)
        elif isinstance(num, decimal.Decimal):
            if num.is_finite():
                _mpf_ = mlib.from_str(str(num), precision, rnd)
            elif num.is_nan():
                return S.NaN
            elif num.is_infinite():
                if num > 0:
                    return S.Infinity
                return S.NegativeInfinity
            else:
                raise ValueError("unexpected decimal value %s" % str(num))
        elif isinstance(num, tuple) and len(num) in (3, 4):
            if isinstance(num[1], str):
                # it's a hexadecimal (coming from a pickled object)
                num = list(num)
                # If we're loading an object pickled in Python 2 into
                # Python 3, we may need to strip a tailing 'L' because
                # of a shim for int on Python 3, see issue #13470.
                # Strip leading '0x' - gmpy2 only documents such inputs
                # with base prefix as valid when the 2nd argument (base) is 0.
                # When mpmath uses Sage as the backend, however, it
                # ends up including '0x' when preparing the picklable tuple.
                # See issue #19690.
                num[1] = num[1].removeprefix('0x').removesuffix('L')
                # Now we can assume that it is in standard form
                num[1] = MPZ(num[1], 16)
                _mpf_ = tuple(num)
            else:
                if len(num) == 4:
                    # handle normalization hack
                    return Float._new(num, precision)
                else:
                    if not all((
                            num[0] in (0, 1),
                            num[1] >= 0,
                            all(type(i) in (int, int) for i in num)
                            )):
                        raise ValueError('malformed mpf: %s' % (num,))
                    # don't compute number or else it may
                    # over/underflow
                    return Float._new(
                        (num[0], num[1], num[2], bitcount(num[1])),
                        precision)
        elif isinstance(num, (Number, NumberSymbol)):
            _mpf_ = num._as_mpf_val(precision)
        else:
            _mpf_ = mpmath.mpf(num, prec=precision)._mpf_

        return cls._new(_mpf_, precision, zero=False)

    @classmethod
    def _new(cls, _mpf_, _prec, zero=True):
        # special cases
        if zero and _mpf_ == fzero:
            return S.Zero  # Float(0) -> 0.0; Float._new((0,0,0,0)) -> 0
        elif _mpf_ == _mpf_nan:
            return S.NaN
        elif _mpf_ == _mpf_inf:
            return S.Infinity
        elif _mpf_ == _mpf_ninf:
            return S.NegativeInfinity

        obj = Expr.__new__(cls)
        obj._mpf_ = mpf_norm(_mpf_, _prec)
        obj._prec = _prec
        return obj

    def __getnewargs_ex__(self):
        sign, man, exp, bc = self._mpf_
        arg = (sign, hex(man)[2:], exp, bc)
        kwargs = {'precision': self._prec}
        return ((arg,), kwargs)

    def _hashable_content(self):
        return (self._mpf_, self._prec)

    def floor(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_floor(self._mpf_, self._prec))))

    def ceiling(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_ceil(self._mpf_, self._prec))))

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    @property
    def num(self):
        return mpmath.mpf(self._mpf_)

    def _as_mpf_val(self, prec):
        rv = mpf_norm(self._mpf_, prec)
        if rv != self._mpf_ and self._prec == prec:
            debug(self._mpf_, rv)
        return rv

    def _as_mpf_op(self, prec):
        return self._mpf_, max(prec, self._prec)

    def _eval_is_finite(self):
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return False
        return True

    def _eval_is_infinite(self):
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return True
        return False

    def _eval_is_integer(self):
        if self._mpf_ == fzero:
            return True
        if not int_valued(self):
            return False

    def _eval_is_negative(self):
        if self._mpf_ in (_mpf_ninf, _mpf_inf):
            return False
        return self.num < 0

    def _eval_is_positive(self):
        if self._mpf_ in (_mpf_ninf, _mpf_inf):
            return False
        return self.num > 0

    def _eval_is_extended_negative(self):
        if self._mpf_ == _mpf_ninf:
            return True
        if self._mpf_ == _mpf_inf:
            return False
        return self.num < 0

    def _eval_is_extended_positive(self):
        if self._mpf_ == _mpf_inf:
            return True
        if self._mpf_ == _mpf_ninf:
            return False
        return self.num > 0

    def _eval_is_zero(self):
        return self._mpf_ == fzero

    def __bool__(self):
        return self._mpf_ != fzero

    def __neg__(self):
        if not self:
            return self
        return Float._new(mlib.mpf_neg(self._mpf_), self._prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_add(self._mpf_, rhs, prec, rnd), prec)
        return Number.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_sub(self._mpf_, rhs, prec, rnd), prec)
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mul(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mul__(self, other)

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and other != 0 and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_div(self._mpf_, rhs, prec, rnd), prec)
        return Number.__truediv__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if isinstance(other, Rational) and other.q != 1 and global_parameters.evaluate:
            # calculate mod with Rationals, *then* round the result
            return Float(Rational.__mod__(Rational(self), other),
                         precision=self._prec)
        if isinstance(other, Float) and global_parameters.evaluate:
            r = self/other
            if int_valued(r):
                return Float(0, precision=max(self._prec, other._prec))
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mod(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mod__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rmod__(self, other):
        if isinstance(other, Float) and global_parameters.evaluate:
            return other.__mod__(self)
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mod(rhs, self._mpf_, prec, rnd), prec)
        return Number.__rmod__(self, other)

    def _eval_power(self, expt):
        """
        expt is symbolic object but not equal to 0, 1

        (-p)**r -> exp(r*log(-p)) -> exp(r*(log(p) + I*Pi)) ->
                  -> p**r*(sin(Pi*r) + cos(Pi*r)*I)
        """
        if equal_valued(self, 0):
            if expt.is_extended_positive:
                return self
            if expt.is_extended_negative:
                return S.ComplexInfinity
        if isinstance(expt, Number):
            if isinstance(expt, Integer):
                prec = self._prec
                return Float._new(
                    mlib.mpf_pow_int(self._mpf_, expt.p, prec, rnd), prec)
            elif isinstance(expt, Rational) and \
                    expt.p == 1 and expt.q % 2 and self.is_negative:
                return Pow(S.NegativeOne, expt, evaluate=False)*(
                    -self)._eval_power(expt)
            expt, prec = expt._as_mpf_op(self._prec)
            mpfself = self._mpf_
            try:
                y = mpf_pow(mpfself, expt, prec, rnd)
                return Float._new(y, prec)
            except mlib.ComplexResult:
                re, im = mlib.mpc_pow(
                    (mpfself, fzero), (expt, fzero), prec, rnd)
                return Float._new(re, prec) + \
                    Float._new(im, prec)*S.ImaginaryUnit

    def __abs__(self):
        return Float._new(mlib.mpf_abs(self._mpf_), self._prec)

    def __int__(self):
        if self._mpf_ == fzero:
            return 0
        return int(mlib.to_int(self._mpf_))  # uses round_fast = round_down

    def __eq__(self, other):
        if isinstance(other, float):
            other = Float(other)
        return Basic.__eq__(self, other)

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        else:
            return not eq

    def __hash__(self):
        float_val = float(self)
        if not math.isinf(float_val):
            return hash(float_val)
        return Basic.__hash__(self)

    def _Frel(self, other, op):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Rational:
            # test self*other.q <?> other.p without losing precision
            '''
            >>> f = Float(.1,2)
            >>> i = 1234567890
            >>> (f*i)._mpf_
            (0, 471, 18, 9)
            >>> mlib.mpf_mul(f._mpf_, mlib.from_int(i))
            (0, 505555550955, -12, 39)
            '''
            smpf = mlib.mpf_mul(self._mpf_, mlib.from_int(other.q))
            ompf = mlib.from_int(other.p)
            return _sympify(bool(op(smpf, ompf)))
        elif other.is_Float:
            return _sympify(bool(
                        op(self._mpf_, other._mpf_)))
        elif other.is_comparable and other not in (
                S.Infinity, S.NegativeInfinity):
            other = other.evalf(prec_to_dps(self._prec))
            if other._prec > 1:
                if other.is_Number:
                    return _sympify(bool(
                        op(self._mpf_, other._as_mpf_val(self._prec))))

    def __gt__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__lt__(self)
        rv = self._Frel(other, mlib.mpf_gt)
        if rv is None:
            return Expr.__gt__(self, other)
        return rv

    def __ge__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__le__(self)
        rv = self._Frel(other, mlib.mpf_ge)
        if rv is None:
            return Expr.__ge__(self, other)
        return rv

    def __lt__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__gt__(self)
        rv = self._Frel(other, mlib.mpf_lt)
        if rv is None:
            return Expr.__lt__(self, other)
        return rv

    def __le__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__ge__(self)
        rv = self._Frel(other, mlib.mpf_le)
        if rv is None:
            return Expr.__le__(self, other)
        return rv

    def epsilon_eq(self, other, epsilon="1e-15"):
        return abs(self - other) < Float(epsilon)

    def __format__(self, format_spec):
        return format(decimal.Decimal(str(self)), format_spec)


# Add sympify converters
_sympy_converter[float] = _sympy_converter[decimal.Decimal] = Float

# this is here to work nicely in Sage
RealNumber = Float


class Rational(Number):
    """Represents rational numbers (p/q) of any size.

    Examples
    ========

    >>> from sympy import Rational, nsimplify, S, pi
    >>> Rational(1, 2)
    1/2

    Rational is unprejudiced in accepting input. If a float is passed, the
    underlying value of the binary representation will be returned:

    >>> Rational(.5)
    1/2
    >>> Rational(.2)
    3602879701896397/18014398509481984

    If the simpler representation of the float is desired then consider
    limiting the denominator to the desired value or convert the float to
    a string (which is roughly equivalent to limiting the denominator to
    10**12):

    >>> Rational(str(.2))
    1/5
    >>> Rational(.2).limit_denominator(10**12)
    1/5

    An arbitrarily precise Rational is obtained when a string literal is
    passed:

    >>> Rational("1.23")
    123/100
    >>> Rational('1e-2')
    1/100
    >>> Rational(".1")
    1/10
    >>> Rational('1e-2/3.2')
    1/320

    The conversion of other types of strings can be handled by
    the sympify() function, and conversion of floats to expressions
    or simple fractions can be handled with nsimplify:

    >>> S('.[3]')  # repeating digits in brackets
    1/3
    >>> S('3**2/10')  # general expressions
    9/10
    >>> nsimplify(.3)  # numbers that have a simple form
    3/10

    But if the input does not reduce to a literal Rational, an error will
    be raised:

    >>> Rational(pi)
    Traceback (most recent call last):
    ...
    TypeError: invalid input: pi


    Low-level
    ---------

    Access numerator and denominator as .p and .q:

    >>> r = Rational(3, 4)
    >>> r
    3/4
    >>> r.p
    3
    >>> r.q
    4

    Note that p and q return integers (not SymPy Integers) so some care
    is needed when using them in expressions:

    >>> r.p/r.q
    0.75

    See Also
    ========
    sympy.core.sympify.sympify, sympy.simplify.simplify.nsimplify
    """
    is_real = True
    is_integer = False
    is_rational = True
    is_number = True

    __slots__ = ('p', 'q')

    p: int
    q: int

    is_Rational = True

    @cacheit
    def __new__(cls, p, q=None, gcd=None):
        if q is None:
            if isinstance(p, Rational):
                return p

            if isinstance(p, SYMPY_INTS):
                pass
            else:
                if isinstance(p, (float, Float)):
                    return Rational(*_as_integer_ratio(p))

                if not isinstance(p, str):
                    try:
                        p = sympify(p)
                    except (SympifyError, SyntaxError):
                        pass  # error will raise below
                else:
                    if p.count('/') > 1:
                        raise TypeError('invalid input: %s' % p)
                    p = p.replace(' ', '')
                    pq = p.rsplit('/', 1)
                    if len(pq) == 2:
                        p, q = pq
                        fp = fractions.Fraction(p)
                        fq = fractions.Fraction(q)
                        p = fp/fq
                    try:
                        p = fractions.Fraction(p)
                    except ValueError:
                        pass  # error will raise below
                    else:
                        return cls._new(p.numerator, p.denominator, 1)

                if not isinstance(p, Rational):
                    raise TypeError('invalid input: %s' % p)

            q = 1

        Q = 1

        if not isinstance(p, SYMPY_INTS):
            p = Rational(p)
            Q *= p.q
            p = p.p
        else:
            p = int(p)

        if not isinstance(q, SYMPY_INTS):
            q = Rational(q)
            p *= q.q
            Q *= q.p
        else:
            Q *= int(q)
        q = Q

        if gcd is not None:
            sympy_deprecation_warning(
                "gcd is deprecated in Rational, use nsimplify instead",
                deprecated_since_version="1.11",
                active_deprecations_target="deprecated-rational-gcd",
                stacklevel=4,
            )
            return cls._new(p, q, gcd)

        # p and q are now ints
        return cls._new(p, q)

    @classmethod
    def _new(cls, p, q, gcd=None):
        if q == 0:
            if p == 0:
                if _errdict["divide"]:
                    raise ValueError("Indeterminate 0/0")
                else:
                    return S.NaN
            return S.ComplexInfinity

        if q < 0:
            q = -q
            p = -p

        if gcd is None:
            gcd = igcd(abs(p), q)

        if gcd > 1:
            p //= gcd
            q //= gcd

        return cls.from_coprime_ints(p, q)

    @classmethod
    def from_coprime_ints(cls, p: int, q: int) -> Rational:
        """Create a Rational from a pair of coprime integers.

        Both ``p`` and ``q`` should be strictly of type ``int``.

        The caller should ensure that ``gcd(p,q) == 1`` and ``q > 0``.

        This may be more efficient than ``Rational(p, q)``. The validity of the
        arguments may or may not be checked so it should not be relied upon to
        pass unvalidated or invalid arguments to this function.
        """
        if q == 1:
            return Integer(p)
        if p == 1 and q == 2:
            return S.Half

        obj = Expr.__new__(cls)
        obj.p = p
        obj.q = q
        return obj

    def limit_denominator(self, max_denominator=1000000):
        """Closest Rational to self with denominator at most max_denominator.

        Examples
        ========

        >>> from sympy import Rational
        >>> Rational('3.141592653589793').limit_denominator(10)
        22/7
        >>> Rational('3.141592653589793').limit_denominator(100)
        311/99

        """
        f = fractions.Fraction(self.p, self.q)
        return Rational(f.limit_denominator(fractions.Fraction(int(max_denominator))))

    def __getnewargs__(self):
        return (self.p, self.q)

    def _hashable_content(self):
        return (self.p, self.q)

    def _eval_is_positive(self):
        return self.p > 0

    def _eval_is_zero(self):
        return self.p == 0

    def __neg__(self):
        return Rational(-self.p, self.q)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(self.p + self.q*other.p, self.q, 1)
            elif isinstance(other, Rational):
                #TODO: this can probably be optimized more
                return Rational(self.p*other.q + self.q*other.p, self.q*other.q)
            elif isinstance(other, Float):
                return other + self
            else:
                return Number.__add__(self, other)
        return Number.__add__(self, other)
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(self.p - self.q*other.p, self.q, 1)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q - self.q*other.p, self.q*other.q)
            elif isinstance(other, Float):
                return -other + self
            else:
                return Number.__sub__(self, other)
        return Number.__sub__(self, other)
    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(self.q*other.p - self.p, self.q, 1)
            elif isinstance(other, Rational):
                return Rational(self.q*other.p - self.p*other.q, self.q*other.q)
            elif isinstance(other, Float):
                return -self + other
            else:
                return Number.__rsub__(self, other)
        return Number.__rsub__(self, other)
    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(self.p*other.p, self.q, igcd(other.p, self.q))
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.p, self.q*other.q, igcd(self.p, other.q)*igcd(self.q, other.p))
            elif isinstance(other, Float):
                return other*self
            else:
                return Number.__mul__(self, other)
        return Number.__mul__(self, other)
    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                if self.p and other.p == S.Zero:
                    return S.ComplexInfinity
                else:
                    return Rational._new(self.p, self.q*other.p, igcd(self.p, other.p))
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.q, self.q*other.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            elif isinstance(other, Float):
                return self*(1/other)
            else:
                return Number.__truediv__(self, other)
        return Number.__truediv__(self, other)
    @_sympifyit('other', NotImplemented)
    def __rtruediv__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(other.p*self.q, self.p, igcd(self.p, other.p))
            elif isinstance(other, Rational):
                return Rational._new(other.p*self.q, other.q*self.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            elif isinstance(other, Float):
                return other*(1/self)
            else:
                return Number.__rtruediv__(self, other)
        return Number.__rtruediv__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Rational):
                n = (self.p*other.q) // (other.p*self.q)
                return Rational(self.p*other.q - n*other.p*self.q, self.q*other.q)
            if isinstance(other, Float):
                # calculate mod with Rationals, *then* round the answer
                return Float(self.__mod__(Rational(other)),
                             precision=other._prec)
            return Number.__mod__(self, other)
        return Number.__mod__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rmod__(self, other):
        if isinstance(other, Rational):
            return Rational.__mod__(other, self)
        return Number.__rmod__(self, other)

    def _eval_power(self, expt):
        if isinstance(expt, Number):
            if isinstance(expt, Float):
                return self._eval_evalf(expt._prec)**expt
            if expt.is_extended_negative:
                # (3/4)**-2 -> (4/3)**2
                ne = -expt
                if (ne is S.One):
                    return Rational(self.q, self.p)
                if self.is_negative:
                    return S.NegativeOne**expt*Rational(self.q, -self.p)**ne
                else:
                    return Rational(self.q, self.p)**ne
            if expt is S.Infinity:  # -oo already caught by test for negative
                if self.p > self.q:
                    # (3/2)**oo -> oo
                    return S.Infinity
                if self.p < -self.q:
                    # (-3/2)**oo -> oo + I*oo
                    return S.Infinity + S.Infinity*S.ImaginaryUnit
                return S.Zero
            if isinstance(expt, Integer):
                # (4/3)**2 -> 4**2 / 3**2
                return Rational._new(self.p**expt.p, self.q**expt.p, 1)
            if isinstance(expt, Rational):
                intpart = expt.p // expt.q
                if intpart:
                    intpart += 1
                    remfracpart = intpart*expt.q - expt.p
                    ratfracpart = Rational(remfracpart, expt.q)
                    if self.p != 1:
                        return Integer(self.p)**expt*Integer(self.q)**ratfracpart*Rational._new(1, self.q**intpart, 1)
                    return Integer(self.q)**ratfracpart*Rational._new(1, self.q**intpart, 1)
                else:
                    remfracpart = expt.q - expt.p
                    ratfracpart = Rational(remfracpart, expt.q)
                    if self.p != 1:
                        return Integer(self.p)**expt*Integer(self.q)**ratfracpart*Rational._new(1, self.q, 1)
                    return Integer(self.q)**ratfracpart*Rational._new(1, self.q, 1)

        if self.is_extended_negative and expt.is_even:
            return (-self)**expt

        return

    def _as_mpf_val(self, prec):
        return mlib.from_rational(self.p, self.q, prec, rnd)

    def _mpmath_(self, prec, rnd):
        return mpmath.make_mpf(mlib.from_rational(self.p, self.q, prec, rnd))

    def __abs__(self):
        return Rational(abs(self.p), self.q)

    def __int__(self):
        p, q = self.p, self.q
        if p < 0:
            return -int(-p//q)
        return int(p//q)

    def floor(self):
        return Integer(self.p // self.q)

    def ceiling(self):
        return -Integer(-self.p // self.q)

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    def __eq__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if not isinstance(other, Number):
            # S(0) == S.false is False
            # S(0) == False is True
            return False
        if other.is_NumberSymbol:
            if other.is_irrational:
                return False
            return other.__eq__(self)
        if other.is_Rational:
            # a Rational is always in reduced form so will never be 2/4
            # so we can just check equivalence of args
            return self.p == other.p and self.q == other.q
        return False

    def __ne__(self, other):
        return not self == other

    def _Rrel(self, other, attr):
        # if you want self < other, pass self, other, __gt__
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Number:
            op = None
            s, o = self, other
            if other.is_NumberSymbol:
                op = getattr(o, attr)
            elif other.is_Float:
                op = getattr(o, attr)
            elif other.is_Rational:
                s, o = Integer(s.p*o.q), Integer(s.q*o.p)
                op = getattr(o, attr)
            if op:
                return op(s)
            if o.is_number and o.is_extended_real:
                return Integer(s.p), s.q*o

    def __gt__(self, other):
        rv = self._Rrel(other, '__lt__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__gt__(*rv)

    def __ge__(self, other):
        rv = self._Rrel(other, '__le__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__ge__(*rv)

    def __lt__(self, other):
        rv = self._Rrel(other, '__gt__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__lt__(*rv)

    def __le__(self, other):
        rv = self._Rrel(other, '__ge__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__le__(*rv)

    def __hash__(self):
        return super().__hash__()

    def factors(self, limit=None, use_trial=True, use_rho=False,
                use_pm1=False, verbose=False, visual=False):
        """A wrapper to factorint which return factors of self that are
        smaller than limit (or cheap to compute). Special methods of
        factoring are disabled by default so that only trial division is used.
        """
        from sympy.ntheory.factor_ import factorrat

        return factorrat(self, limit=limit, use_trial=use_trial,
                      use_rho=use_rho, use_pm1=use_pm1,
                      verbose=verbose).copy()

    @property
    def numerator(self):
        return self.p

    @property
    def denominator(self):
        return self.q

    @_sympifyit('other', NotImplemented)
    def gcd(self, other):
        if isinstance(other, Rational):
            if other == S.Zero:
                return other
            return Rational(
                igcd(self.p, other.p),
                ilcm(self.q, other.q))
        return Number.gcd(self, other)

    @_sympifyit('other', NotImplemented)
    def lcm(self, other):
        if isinstance(other, Rational):
            return Rational(
                self.p // igcd(self.p, other.p) * other.p,
                igcd(self.q, other.q))
        return Number.lcm(self, other)

    def as_numer_denom(self):
        return Integer(self.p), Integer(self.q)

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import S
        >>> (S(-3)/2).as_content_primitive()
        (3/2, -1)

        See docstring of Expr.as_content_primitive for more examples.
        """

        if self:
            if self.is_positive:
                return self, S.One
            return -self, S.NegativeOne
        return S.One, self

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return self, S.One

    def as_coeff_Add(self, rational=False):
        """Efficiently extract the coefficient of a summation."""
        return self, S.Zero


class Integer(Rational):
    """Represents integer numbers of any size.

    Examples
    ========

    >>> from sympy import Integer
    >>> Integer(3)
    3

    If a float or a rational is passed to Integer, the fractional part
    will be discarded; the effect is of rounding toward zero.

    >>> Integer(3.8)
    3
    >>> Integer(-3.8)
    -3

    A string is acceptable input if it can be parsed as an integer:

    >>> Integer("9" * 20)
    99999999999999999999

    It is rarely needed to explicitly instantiate an Integer, because
    Python integers are automatically converted to Integer when they
    are used in SymPy expressions.
    """
    q = 1
    is_integer = True
    is_number = True

    is_Integer = True

    __slots__ = ()

    def _as_mpf_val(self, prec):
        return mlib.from_int(self.p, prec, rnd)

    def _mpmath_(self, prec, rnd):
        return mpmath.make_mpf(self._as_mpf_val(prec))

    @cacheit
    def __new__(cls, i):
        if isinstance(i, str):
            i = i.replace(' ', '')
        # whereas we cannot, in general, make a Rational from an
        # arbitrary expression, we can make an Integer unambiguously
        # (except when a non-integer expression happens to round to
        # an integer). So we proceed by taking int() of the input and
        # let the int routines determine whether the expression can
        # be made into an int or whether an error should be raised.
        try:
            ival = int(i)
        except TypeError:
            raise TypeError(
                "Argument of Integer should be of numeric type, got %s." % i)
        # We only work with well-behaved integer types. This converts, for
        # example, numpy.int32 instances.
        if ival == 1:
            return S.One
        if ival == -1:
            return S.NegativeOne
        if ival == 0:
            return S.Zero
        obj = Expr.__new__(cls)
        obj.p = ival
        return obj

    def __getnewargs__(self):
        return (self.p,)

    # Arithmetic operations are here for efficiency
    def __int__(self):
        return self.p

    def floor(self):
        return Integer(self.p)

    def ceiling(self):
        return Integer(self.p)

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    def __neg__(self):
        return Integer(-self.p)

    def __abs__(self):
        if self.p >= 0:
            return self
        else:
            return Integer(-self.p)

    def __divmod__(self, other):
        if isinstance(other, Integer) and global_parameters.evaluate:
            return Tuple(*(divmod(self.p, other.p)))
        else:
            return Number.__divmod__(self, other)

    def __rdivmod__(self, other):
        if isinstance(other, int) and global_parameters.evaluate:
            return Tuple(*(divmod(other, self.p)))
        else:
            try:
                other = Number(other)
            except TypeError:
                msg = "unsupported operand type(s) for divmod(): '%s' and '%s'"
                oname = type(other).__name__
                sname = type(self).__name__
                raise TypeError(msg % (oname, sname))
            return Number.__divmod__(other, self)

    # TODO make it decorator + bytecodehacks?
    def __add__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p + other)
            elif isinstance(other, Integer):
                return Integer(self.p + other.p)
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.q + other.p, other.q, 1)
            return Rational.__add__(self, other)
        else:
            return Add(self, other)

    def __radd__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other + self.p)
            elif isinstance(other, Rational):
                return Rational._new(other.p + self.p*other.q, other.q, 1)
            return Rational.__radd__(self, other)
        return Rational.__radd__(self, other)

    def __sub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p - other)
            elif isinstance(other, Integer):
                return Integer(self.p - other.p)
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.q - other.p, other.q, 1)
            return Rational.__sub__(self, other)
        return Rational.__sub__(self, other)

    def __rsub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other - self.p)
            elif isinstance(other, Rational):
                return Rational._new(other.p - self.p*other.q, other.q, 1)
            return Rational.__rsub__(self, other)
        return Rational.__rsub__(self, other)

    def __mul__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p*other)
            elif isinstance(other, Integer):
                return Integer(self.p*other.p)
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.p, other.q, igcd(self.p, other.q))
            return Rational.__mul__(self, other)
        return Rational.__mul__(self, other)

    def __rmul__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other*self.p)
            elif isinstance(other, Rational):
                return Rational._new(other.p*self.p, other.q, igcd(self.p, other.q))
            return Rational.__rmul__(self, other)
        return Rational.__rmul__(self, other)

    def __mod__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p % other)
            elif isinstance(other, Integer):
                return Integer(self.p % other.p)
            return Rational.__mod__(self, other)
        return Rational.__mod__(self, other)

    def __rmod__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other % self.p)
            elif isinstance(other, Integer):
                return Integer(other.p % self.p)
            return Rational.__rmod__(self, other)
        return Rational.__rmod__(self, other)

    def __eq__(self, other):
        if isinstance(other, int):
            return (self.p == other)
        elif isinstance(other, Integer):
            return (self.p == other.p)
        return Rational.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p > other.p)
        return Rational.__gt__(self, other)

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p < other.p)
        return Rational.__lt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p >= other.p)
        return Rational.__ge__(self, other)

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p <= other.p)
        return Rational.__le__(self, other)

    def __hash__(self):
        return hash(self.p)

    def __index__(self):
        return self.p

    ########################################

    def _eval_is_odd(self):
        return bool(self.p % 2)

    def _eval_power(self, expt):
        """
        Tries to do some simplifications on self**expt

        Returns None if no further simplifications can be done.

        Explanation
        ===========

        When exponent is a fraction (so we have for example a square root),
        we try to find a simpler representation by factoring the argument
        up to factors of 2**15, e.g.

          - sqrt(4) becomes 2
          - sqrt(-4) becomes 2*I
          - (2**(3+7)*3**(6+7))**Rational(1,7) becomes 6*18**(3/7)

        Further simplification would require a special call to factorint on
        the argument which is not done here for sake of speed.

        """
        from sympy.ntheory.factor_ import perfect_power

        if expt is S.Infinity:
            if self.p > S.One:
                return S.Infinity
            # cases -1, 0, 1 are done in their respective classes
            return S.Infinity + S.ImaginaryUnit*S.Infinity
        if expt is S.NegativeInfinity:
            return Rational._new(1, self, 1)**S.Infinity
        if not isinstance(expt, Number):
            # simplify when expt is even
            # (-2)**k --> 2**k
            if self.is_negative and expt.is_even:
                return (-self)**expt
        if isinstance(expt, Float):
            # Rational knows how to exponentiate by a Float
            return super()._eval_power(expt)
        if not isinstance(expt, Rational):
            return
        if expt is S.Half and self.is_negative:
            # we extract I for this special case since everyone is doing so
            return S.ImaginaryUnit*Pow(-self, expt)
        if expt.is_negative:
            # invert base and change sign on exponent
            ne = -expt
            if self.is_negative:
                return S.NegativeOne**expt*Rational._new(1, -self.p, 1)**ne
            else:
                return Rational._new(1, self.p, 1)**ne
        # see if base is a perfect root, sqrt(4) --> 2
        x, xexact = integer_nthroot(abs(self.p), expt.q)
        if xexact:
            # if it's a perfect root we've finished
            result = Integer(x**abs(expt.p))
            if self.is_negative:
                result *= S.NegativeOne**expt
            return result

        # The following is an algorithm where we collect perfect roots
        # from the factors of base.

        # if it's not an nth root, it still might be a perfect power
        b_pos = int(abs(self.p))
        p = perfect_power(b_pos)
        if p is not False:
            # XXX: Convert to int because perfect_power may return fmpz
            # Ideally that should be fixed in perfect_power though...
            dict = {int(p[0]): int(p[1])}
        else:
            dict = Integer(b_pos).factors(limit=2**15)

        # now process the dict of factors
        out_int = 1  # integer part
        out_rad = 1  # extracted radicals
        sqr_int = 1
        sqr_gcd = 0
        sqr_dict = {}
        for prime, exponent in dict.items():
            exponent *= expt.p
            # remove multiples of expt.q: (2**12)**(1/10) -> 2*(2**2)**(1/10)
            div_e, div_m = divmod(exponent, expt.q)
            if div_e > 0:
                out_int *= prime**div_e
            if div_m > 0:
                # see if the reduced exponent shares a gcd with e.q
                # (2**2)**(1/10) -> 2**(1/5)
                g = igcd(div_m, expt.q)
                if g != 1:
                    out_rad *= Pow(prime, Rational._new(div_m//g, expt.q//g, 1))
                else:
                    sqr_dict[prime] = div_m
        # identify gcd of remaining powers
        for p, ex in sqr_dict.items():
            if sqr_gcd == 0:
                sqr_gcd = ex
            else:
                sqr_gcd = igcd(sqr_gcd, ex)
                if sqr_gcd == 1:
                    break
        for k, v in sqr_dict.items():
            sqr_int *= k**(v//sqr_gcd)
        if sqr_int == b_pos and out_int == 1 and out_rad == 1:
            result = None
        else:
            result = out_int*out_rad*Pow(sqr_int, Rational(sqr_gcd, expt.q))
            if self.is_negative:
                result *= Pow(S.NegativeOne, expt)
        return result

    def _eval_is_prime(self):
        from sympy.ntheory.primetest import isprime

        return isprime(self)

    def _eval_is_composite(self):
        if self > 1:
            return fuzzy_not(self.is_prime)
        else:
            return False

    def as_numer_denom(self):
        return self, S.One

    @_sympifyit('other', NotImplemented)
    def __floordiv__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        if isinstance(other, Integer):
            return Integer(self.p // other)
        return divmod(self, other)[0]

    def __rfloordiv__(self, other):
        return Integer(Integer(other).p // self.p)

    # These bitwise operations (__lshift__, __rlshift__, ..., __invert__) are defined
    # for Integer only and not for general SymPy expressions. This is to achieve
    # compatibility with the numbers.Integral ABC which only defines these operations
    # among instances of numbers.Integral. Therefore, these methods check explicitly for
    # integer types rather than using sympify because they should not accept arbitrary
    # symbolic expressions and there is no symbolic analogue of numbers.Integral's
    # bitwise operations.
    def __lshift__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p << int(other))
        else:
            return NotImplemented

    def __rlshift__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) << self.p)
        else:
            return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p >> int(other))
        else:
            return NotImplemented

    def __rrshift__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) >> self.p)
        else:
            return NotImplemented

    def __and__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p & int(other))
        else:
            return NotImplemented

    def __rand__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) & self.p)
        else:
            return NotImplemented

    def __xor__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p ^ int(other))
        else:
            return NotImplemented

    def __rxor__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) ^ self.p)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p | int(other))
        else:
            return NotImplemented

    def __ror__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) | self.p)
        else:
            return NotImplemented

    def __invert__(self):
        return Integer(~self.p)

# Add sympify converters
_sympy_converter[int] = Integer


class AlgebraicNumber(Expr):
    r"""
    Class for representing algebraic numbers in SymPy.

    Symbolically, an instance of this class represents an element
    $\alpha \in \mathbb{Q}(\theta) \hookrightarrow \mathbb{C}$. That is, the
    algebraic number $\alpha$ is represented as an element of a particular
    number field $\mathbb{Q}(\theta)$, with a particular embedding of this
    field into the complex numbers.

    Formally, the primitive element $\theta$ is given by two data points: (1)
    its minimal polynomial (which defines $\mathbb{Q}(\theta)$), and (2) a
    particular complex number that is a root of this polynomial (which defines
    the embedding $\mathbb{Q}(\theta) \hookrightarrow \mathbb{C}$). Finally,
    the algebraic number $\alpha$ which we represent is then given by the
    coefficients of a polynomial in $\theta$.
    """

    __slots__ = ('rep', 'root', 'alias', 'minpoly', '_own_minpoly')

    is_AlgebraicNumber = True
    is_algebraic = True
    is_number = True


    kind = NumberKind

    # Optional alias symbol is not free.
    # Actually, alias should be a Str, but some methods
    # expect that it be an instance of Expr.
    free_symbols: set[Basic] = set()

    def __new__(cls, expr, coeffs=None, alias=None, **args):
        r"""
        Construct a new algebraic number $\alpha$ belonging to a number field
        $k = \mathbb{Q}(\theta)$.

        There are four instance attributes to be determined:

        ===========  ============================================================================
        Attribute    Type/Meaning
        ===========  ============================================================================
        ``root``     :py:class:`~.Expr` for $\theta$ as a complex number
        ``minpoly``  :py:class:`~.Poly`, the minimal polynomial of $\theta$
        ``rep``      :py:class:`~sympy.polys.polyclasses.DMP` giving $\alpha$ as poly in $\theta$
        ``alias``    :py:class:`~.Symbol` for $\theta$, or ``None``
        ===========  ============================================================================

        See Parameters section for how they are determined.

        Parameters
        ==========

        expr : :py:class:`~.Expr`, or pair $(m, r)$
            There are three distinct modes of construction, depending on what
            is passed as *expr*.

            **(1)** *expr* is an :py:class:`~.AlgebraicNumber`:
            In this case we begin by copying all four instance attributes from
            *expr*. If *coeffs* were also given, we compose the two coeff
            polynomials (see below). If an *alias* was given, it overrides.

            **(2)** *expr* is any other type of :py:class:`~.Expr`:
            Then ``root`` will equal *expr*. Therefore it
            must express an algebraic quantity, and we will compute its
            ``minpoly``.

            **(3)** *expr* is an ordered pair $(m, r)$ giving the
            ``minpoly`` $m$, and a ``root`` $r$ thereof, which together
            define $\theta$. In this case $m$ may be either a univariate
            :py:class:`~.Poly` or any :py:class:`~.Expr` which represents the
            same, while $r$ must be some :py:class:`~.Expr` representing a
            complex number that is a root of $m$, including both explicit
            expressions in radicals, and instances of
            :py:class:`~.ComplexRootOf` or :py:class:`~.AlgebraicNumber`.

        coeffs : list, :py:class:`~.ANP`, None, optional (default=None)
            This defines ``rep``, giving the algebraic number $\alpha$ as a
            polynomial in $\theta$.

            If a list, the elements should be integers or rational numbers.
            If an :py:class:`~.ANP`, we take its coefficients (using its
            :py:meth:`~.ANP.to_list()` method). If ``None``, then the list of
            coefficients defaults to ``[1, 0]``, meaning that $\alpha = \theta$
            is the primitive element of the field.

            If *expr* was an :py:class:`~.AlgebraicNumber`, let $g(x)$ be its
            ``rep`` polynomial, and let $f(x)$ be the polynomial defined by
            *coeffs*. Then ``self.rep`` will represent the composition
            $(f \circ g)(x)$.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            This is a way to provide a name for the primitive element. We
            described several ways in which the *expr* argument can define the
            value of the primitive element, but none of these methods gave it
            a name. Here, for example, *alias* could be set as
            ``Symbol('theta')``, in order to make this symbol appear when
            $\alpha$ is printed, or rendered as a polynomial, using the
            :py:meth:`~.as_poly()` method.

        Examples
        ========

        Recall that we are constructing an algebraic number as a field element
        $\alpha \in \mathbb{Q}(\theta)$.

        >>> from sympy import AlgebraicNumber, sqrt, CRootOf, S
        >>> from sympy.abc import x

        Example (1): $\alpha = \theta = \sqrt{2}$

        >>> a1 = AlgebraicNumber(sqrt(2))
        >>> a1.minpoly_of_element().as_expr(x)
        x**2 - 2
        >>> a1.evalf(10)
        1.414213562

        Example (2): $\alpha = 3 \sqrt{2} - 5$, $\theta = \sqrt{2}$. We can
        either build on the last example:

        >>> a2 = AlgebraicNumber(a1, [3, -5])
        >>> a2.as_expr()
        -5 + 3*sqrt(2)

        or start from scratch:

        >>> a2 = AlgebraicNumber(sqrt(2), [3, -5])
        >>> a2.as_expr()
        -5 + 3*sqrt(2)

        Example (3): $\alpha = 6 \sqrt{2} - 11$, $\theta = \sqrt{2}$. Again we
        can build on the previous example, and we see that the coeff polys are
        composed:

        >>> a3 = AlgebraicNumber(a2, [2, -1])
        >>> a3.as_expr()
        -11 + 6*sqrt(2)

        reflecting the fact that $(2x - 1) \circ (3x - 5) = 6x - 11$.

        Example (4): $\alpha = \sqrt{2}$, $\theta = \sqrt{2} + \sqrt{3}$. The
        easiest way is to use the :py:func:`~.to_number_field()` function:

        >>> from sympy import to_number_field
        >>> a4 = to_number_field(sqrt(2), sqrt(2) + sqrt(3))
        >>> a4.minpoly_of_element().as_expr(x)
        x**2 - 2
        >>> a4.to_root()
        sqrt(2)
        >>> a4.primitive_element()
        sqrt(2) + sqrt(3)
        >>> a4.coeffs()
        [1/2, 0, -9/2, 0]

        but if you already knew the right coefficients, you could construct it
        directly:

        >>> a4 = AlgebraicNumber(sqrt(2) + sqrt(3), [S(1)/2, 0, S(-9)/2, 0])
        >>> a4.to_root()
        sqrt(2)
        >>> a4.primitive_element()
        sqrt(2) + sqrt(3)

        Example (5): Construct the Golden Ratio as an element of the 5th
        cyclotomic field, supposing we already know its coefficients. This time
        we introduce the alias $\zeta$ for the primitive element of the field:

        >>> from sympy import cyclotomic_poly
        >>> from sympy.abc import zeta
        >>> a5 = AlgebraicNumber(CRootOf(cyclotomic_poly(5), -1),
        ...                  [-1, -1, 0, 0], alias=zeta)
        >>> a5.as_poly().as_expr()
        -zeta**3 - zeta**2
        >>> a5.evalf()
        1.61803398874989

        (The index ``-1`` to ``CRootOf`` selects the complex root with the
        largest real and imaginary parts, which in this case is
        $\mathrm{e}^{2i\pi/5}$. See :py:class:`~.ComplexRootOf`.)

        Example (6): Building on the last example, construct the number
        $2 \phi \in \mathbb{Q}(\phi)$, where $\phi$ is the Golden Ratio:

        >>> from sympy.abc import phi
        >>> a6 = AlgebraicNumber(a5.to_root(), coeffs=[2, 0], alias=phi)
        >>> a6.as_poly().as_expr()
        2*phi
        >>> a6.primitive_element().evalf()
        1.61803398874989

        Note that we needed to use ``a5.to_root()``, since passing ``a5`` as
        the first argument would have constructed the number $2 \phi$ as an
        element of the field $\mathbb{Q}(\zeta)$:

        >>> a6_wrong = AlgebraicNumber(a5, coeffs=[2, 0])
        >>> a6_wrong.as_poly().as_expr()
        -2*zeta**3 - 2*zeta**2
        >>> a6_wrong.primitive_element().evalf()
        0.309016994374947 + 0.951056516295154*I

        """
        from sympy.polys.polyclasses import ANP, DMP
        from sympy.polys.numberfields import minimal_polynomial

        expr = sympify(expr)
        rep0 = None
        alias0 = None

        if isinstance(expr, (tuple, Tuple)):
            minpoly, root = expr

            if not minpoly.is_Poly:
                from sympy.polys.polytools import Poly
                minpoly = Poly(minpoly)
        elif expr.is_AlgebraicNumber:
            minpoly, root, rep0, alias0 = (expr.minpoly, expr.root,
                                           expr.rep, expr.alias)
        else:
            minpoly, root = minimal_polynomial(
                expr, args.get('gen'), polys=True), expr

        dom = minpoly.get_domain()

        if coeffs is not None:
            if not isinstance(coeffs, ANP):
                rep = DMP.from_sympy_list(sympify(coeffs), 0, dom)
                scoeffs = Tuple(*coeffs)
            else:
                rep = DMP.from_list(coeffs.to_list(), 0, dom)
                scoeffs = Tuple(*coeffs.to_list())

        else:
            rep = DMP.from_list([1, 0], 0, dom)
            scoeffs = Tuple(1, 0)

        if rep0 is not None:
            from sympy.polys.densetools import dup_compose
            c = dup_compose(rep.to_list(), rep0.to_list(), dom)
            rep = DMP.from_list(c, 0, dom)
            scoeffs = Tuple(*c)

        if rep.degree() >= minpoly.degree():
            rep = rep.rem(minpoly.rep)

        sargs = (root, scoeffs)

        alias = alias or alias0
        if alias is not None:
            from .symbol import Symbol
            if not isinstance(alias, Symbol):
                alias = Symbol(alias)
            sargs = sargs + (alias,)

        obj = Expr.__new__(cls, *sargs)

        obj.rep = rep
        obj.root = root
        obj.alias = alias
        obj.minpoly = minpoly

        obj._own_minpoly = None

        return obj

    def __hash__(self):
        return super().__hash__()

    def _eval_evalf(self, prec):
        return self.as_expr()._evalf(prec)

    @property
    def is_aliased(self):
        """Returns ``True`` if ``alias`` was set. """
        return self.alias is not None

    def as_poly(self, x=None):
        """Create a Poly instance from ``self``. """
        from sympy.polys.polytools import Poly, PurePoly
        if x is not None:
            return Poly.new(self.rep, x)
        else:
            if self.alias is not None:
                return Poly.new(self.rep, self.alias)
            else:
                from .symbol import Dummy
                return PurePoly.new(self.rep, Dummy('x'))

    def as_expr(self, x=None):
        """Create a Basic expression from ``self``. """
        return self.as_poly(x or self.root).as_expr().expand()

    def coeffs(self):
        """Returns all SymPy coefficients of an algebraic number. """
        return [ self.rep.dom.to_sympy(c) for c in self.rep.all_coeffs() ]

    def native_coeffs(self):
        """Returns all native coefficients of an algebraic number. """
        return self.rep.all_coeffs()

    def to_algebraic_integer(self):
        """Convert ``self`` to an algebraic integer. """
        from sympy.polys.polytools import Poly

        f = self.minpoly

        if f.LC() == 1:
            return self

        coeff = f.LC()**(f.degree() - 1)
        poly = f.compose(Poly(f.gen/f.LC()))

        minpoly = poly*coeff
        root = f.LC()*self.root

        return AlgebraicNumber((minpoly, root), self.coeffs())

    def _eval_simplify(self, **kwargs):
        from sympy.polys.rootoftools import CRootOf
        from sympy.polys import minpoly
        measure, ratio = kwargs['measure'], kwargs['ratio']
        for r in [r for r in self.minpoly.all_roots() if r.func != CRootOf]:
            if minpoly(self.root - r).is_Symbol:
                # use the matching root if it's simpler
                if measure(r) < ratio*measure(self.root):
                    return AlgebraicNumber(r)
        return self

    def field_element(self, coeffs):
        r"""
        Form another element of the same number field.

        Explanation
        ===========

        If we represent $\alpha \in \mathbb{Q}(\theta)$, form another element
        $\beta \in \mathbb{Q}(\theta)$ of the same number field.

        Parameters
        ==========

        coeffs : list, :py:class:`~.ANP`
            Like the *coeffs* arg to the class
            :py:meth:`constructor<.AlgebraicNumber.__new__>`, defines the
            new element as a polynomial in the primitive element.

            If a list, the elements should be integers or rational numbers.
            If an :py:class:`~.ANP`, we take its coefficients (using its
            :py:meth:`~.ANP.to_list()` method).

        Examples
        ========

        >>> from sympy import AlgebraicNumber, sqrt
        >>> a = AlgebraicNumber(sqrt(5), [-1, 1])
        >>> b = a.field_element([3, 2])
        >>> print(a)
        1 - sqrt(5)
        >>> print(b)
        2 + 3*sqrt(5)
        >>> print(b.primitive_element() == a.primitive_element())
        True

        See Also
        ========

        AlgebraicNumber
        """
        return AlgebraicNumber(
            (self.minpoly, self.root), coeffs=coeffs, alias=self.alias)

    @property
    def is_primitive_element(self):
        r"""
        Say whether this algebraic number $\alpha \in \mathbb{Q}(\theta)$ is
        equal to the primitive element $\theta$ for its field.
        """
        c = self.coeffs()
        # Second case occurs if self.minpoly is linear:
        return c == [1, 0] or c == [self.root]

    def primitive_element(self):
        r"""
        Get the primitive element $\theta$ for the number field
        $\mathbb{Q}(\theta)$ to which this algebraic number $\alpha$ belongs.

        Returns
        =======

        AlgebraicNumber

        """
        if self.is_primitive_element:
            return self
        return self.field_element([1, 0])

    def to_primitive_element(self, radicals=True):
        r"""
        Convert ``self`` to an :py:class:`~.AlgebraicNumber` instance that is
        equal to its own primitive element.

        Explanation
        ===========

        If we represent $\alpha \in \mathbb{Q}(\theta)$, $\alpha \neq \theta$,
        construct a new :py:class:`~.AlgebraicNumber` that represents
        $\alpha \in \mathbb{Q}(\alpha)$.

        Examples
        ========

        >>> from sympy import sqrt, to_number_field
        >>> from sympy.abc import x
        >>> a = to_number_field(sqrt(2), sqrt(2) + sqrt(3))

        The :py:class:`~.AlgebraicNumber` ``a`` represents the number
        $\sqrt{2}$ in the field $\mathbb{Q}(\sqrt{2} + \sqrt{3})$. Rendering
        ``a`` as a polynomial,

        >>> a.as_poly().as_expr(x)
        x**3/2 - 9*x/2

        reflects the fact that $\sqrt{2} = \theta^3/2 - 9 \theta/2$, where
        $\theta = \sqrt{2} + \sqrt{3}$.

        ``a`` is not equal to its own primitive element. Its minpoly

        >>> a.minpoly.as_poly().as_expr(x)
        x**4 - 10*x**2 + 1

        is that of $\theta$.

        Converting to a primitive element,

        >>> a_prim = a.to_primitive_element()
        >>> a_prim.minpoly.as_poly().as_expr(x)
        x**2 - 2

        we obtain an :py:class:`~.AlgebraicNumber` whose ``minpoly`` is that of
        the number itself.

        Parameters
        ==========

        radicals : boolean, optional (default=True)
            If ``True``, then we will try to return an
            :py:class:`~.AlgebraicNumber` whose ``root`` is an expression
            in radicals. If that is not possible (or if *radicals* is
            ``False``), ``root`` will be a :py:class:`~.ComplexRootOf`.

        Returns
        =======

        AlgebraicNumber

        See Also
        ========

        is_primitive_element

        """
        if self.is_primitive_element:
            return self
        m = self.minpoly_of_element()
        r = self.to_root(radicals=radicals)
        return AlgebraicNumber((m, r))

    def minpoly_of_element(self):
        r"""
        Compute the minimal polynomial for this algebraic number.

        Explanation
        ===========

        Recall that we represent an element $\alpha \in \mathbb{Q}(\theta)$.
        Our instance attribute ``self.minpoly`` is the minimal polynomial for
        our primitive element $\theta$. This method computes the minimal
        polynomial for $\alpha$.

        """
        if self._own_minpoly is None:
            if self.is_primitive_element:
                self._own_minpoly = self.minpoly
            else:
                from sympy.polys.numberfields.minpoly import minpoly
                theta = self.primitive_element()
                self._own_minpoly = minpoly(self.as_expr(theta), polys=True)
        return self._own_minpoly

    def to_root(self, radicals=True, minpoly=None):
        """
        Convert to an :py:class:`~.Expr` that is not an
        :py:class:`~.AlgebraicNumber`, specifically, either a
        :py:class:`~.ComplexRootOf`, or, optionally and where possible, an
        expression in radicals.

        Parameters
        ==========

        radicals : boolean, optional (default=True)
            If ``True``, then we will try to return the root as an expression
            in radicals. If that is not possible, we will return a
            :py:class:`~.ComplexRootOf`.

        minpoly : :py:class:`~.Poly`
            If the minimal polynomial for `self` has been pre-computed, it can
            be passed in order to save time.

        """
        if self.is_primitive_element and not isinstance(self.root, AlgebraicNumber):
            return self.root
        m = minpoly or self.minpoly_of_element()
        roots = m.all_roots(radicals=radicals)
        if len(roots) == 1:
            return roots[0]
        ex = self.as_expr()
        for b in roots:
            if m.same_root(b, ex):
                return b


class RationalConstant(Rational):
    """
    Abstract base class for rationals with specific behaviors

    Derived classes must define class attributes p and q and should probably all
    be singletons.
    """
    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)


class IntegerConstant(Integer):
    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)


class Zero(IntegerConstant, metaclass=Singleton):
    """The number zero.

    Zero is a singleton, and can be accessed by ``S.Zero``

    Examples
    ========

    >>> from sympy import S, Integer
    >>> Integer(0) is S.Zero
    True
    >>> 1/S.Zero
    zoo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Zero
    """

    p = 0
    q = 1
    is_positive = False
    is_negative = False
    is_zero = True
    is_number = True
    is_comparable = True

    __slots__ = ()

    def __getnewargs__(self):
        return ()

    @staticmethod
    def __abs__():
        return S.Zero

    @staticmethod
    def __neg__():
        return S.Zero

    def _eval_power(self, expt):
        if expt.is_extended_positive:
            return self
        if expt.is_extended_negative:
            return S.ComplexInfinity
        if expt.is_extended_real is False:
            return S.NaN
        if expt.is_zero:
            return S.One

        # infinities are already handled with pos and neg
        # tests above; now throw away leading numbers on Mul
        # exponent since 0**-x = zoo**x even when x == 0
        coeff, terms = expt.as_coeff_Mul()
        if coeff.is_negative:
            return S.ComplexInfinity**terms
        if coeff is not S.One:  # there is a Number to discard
            return self**terms

    def _eval_order(self, *symbols):
        # Order(0,x) -> 0
        return self

    def __bool__(self):
        return False


class One(IntegerConstant, metaclass=Singleton):
    """The number one.

    One is a singleton, and can be accessed by ``S.One``.

    Examples
    ========

    >>> from sympy import S, Integer
    >>> Integer(1) is S.One
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/1_%28number%29
    """
    is_number = True
    is_positive = True

    p = 1
    q = 1

    __slots__ = ()

    def __getnewargs__(self):
        return ()

    @staticmethod
    def __abs__():
        return S.One

    @staticmethod
    def __neg__():
        return S.NegativeOne

    def _eval_power(self, expt):
        return self

    def _eval_order(self, *symbols):
        return

    @staticmethod
    def factors(limit=None, use_trial=True, use_rho=False, use_pm1=False,
                verbose=False, visual=False):
        if visual:
            return S.One
        else:
            return {}


class NegativeOne(IntegerConstant, metaclass=Singleton):
    """The number negative one.

    NegativeOne is a singleton, and can be accessed by ``S.NegativeOne``.

    Examples
    ========

    >>> from sympy import S, Integer
    >>> Integer(-1) is S.NegativeOne
    True

    See Also
    ========

    One

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/%E2%88%921_%28number%29

    """
    is_number = True

    p = -1
    q = 1

    __slots__ = ()

    def __getnewargs__(self):
        return ()

    @staticmethod
    def __abs__():
        return S.One

    @staticmethod
    def __neg__():
        return S.One

    def _eval_power(self, expt):
        if expt.is_odd:
            return S.NegativeOne
        if expt.is_even:
            return S.One
        if isinstance(expt, Number):
            if isinstance(expt, Float):
                return Float(-1.0)**expt
            if expt is S.NaN:
                return S.NaN
            if expt in (S.Infinity, S.NegativeInfinity):
                return S.NaN
            if expt is S.Half:
                return S.ImaginaryUnit
            if isinstance(expt, Rational):
                if expt.q == 2:
                    return S.ImaginaryUnit**Integer(expt.p)
                i, r = divmod(expt.p, expt.q)
                if i:
                    return self**i*self**Rational(r, expt.q)
        return


class Half(RationalConstant, metaclass=Singleton):
    """The rational number 1/2.

    Half is a singleton, and can be accessed by ``S.Half``.

    Examples
    ========

    >>> from sympy import S, Rational
    >>> Rational(1, 2) is S.Half
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/One_half
    """
    is_number = True

    p = 1
    q = 2

    __slots__ = ()

    def __getnewargs__(self):
        return ()

    @staticmethod
    def __abs__():
        return S.Half


class Infinity(Number, metaclass=Singleton):
    r"""Positive infinite quantity.

    Explanation
    ===========

    In real analysis the symbol `\infty` denotes an unbounded
    limit: `x\to\infty` means that `x` grows without bound.

    Infinity is often used not only to define a limit but as a value
    in the affinely extended real number system.  Points labeled `+\infty`
    and `-\infty` can be added to the topological space of the real numbers,
    producing the two-point compactification of the real numbers.  Adding
    algebraic properties to this gives us the extended real numbers.

    Infinity is a singleton, and can be accessed by ``S.Infinity``,
    or can be imported as ``oo``.

    Examples
    ========

    >>> from sympy import oo, exp, limit, Symbol
    >>> 1 + oo
    oo
    >>> 42/oo
    0
    >>> x = Symbol('x')
    >>> limit(exp(x), x, oo)
    oo

    See Also
    ========

    NegativeInfinity, NaN

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Infinity
    """

    is_commutative = True
    is_number = True
    is_complex = False
    is_extended_real = True
    is_infinite = True
    is_comparable = True
    is_extended_positive = True
    is_prime = False

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return r"\infty"

    def _eval_subs(self, old, new):
        if self == old:
            return new

    def _eval_evalf(self, prec=None):
        return Float('inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.NegativeInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.Infinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        return (-self).__add__(other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.NegativeInfinity
        return Number.__mul__(self, other)
    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity or \
                other is S.NegativeInfinity or \
                    other is S.NaN:
                return S.NaN
            if other.is_extended_nonnegative:
                return self
            return S.NegativeInfinity
        return Number.__truediv__(self, other)

    def __abs__(self):
        return S.Infinity

    def __neg__(self):
        return S.NegativeInfinity

    def _eval_power(self, expt):
        """
        ``expt`` is symbolic object but not equal to 0 or 1.

        ================ ======= ==============================
        Expression       Result  Notes
        ================ ======= ==============================
        ``oo ** nan``    ``nan``
        ``oo ** -p``     ``0``   ``p`` is number, ``oo``
        ================ ======= ==============================

        See Also
        ========
        Pow
        NaN
        NegativeInfinity

        """
        if expt.is_extended_positive:
            return S.Infinity
        if expt.is_extended_negative:
            return S.Zero
        if expt is S.NaN:
            return S.NaN
        if expt is S.ComplexInfinity:
            return S.NaN
        if expt.is_extended_real is False and expt.is_number:
            from sympy.functions.elementary.complexes import re
            expt_real = re(expt)
            if expt_real.is_positive:
                return S.ComplexInfinity
            if expt_real.is_negative:
                return S.Zero
            if expt_real.is_zero:
                return S.NaN

            return self**expt.evalf()

    def _as_mpf_val(self, prec):
        return mlib.finf

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return other is S.Infinity or other == float('inf')

    def __ne__(self, other):
        return other is not S.Infinity and other != float('inf')

    __gt__ = Expr.__gt__
    __ge__ = Expr.__ge__
    __lt__ = Expr.__lt__
    __le__ = Expr.__le__

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self

oo = S.Infinity


class NegativeInfinity(Number, metaclass=Singleton):
    """Negative infinite quantity.

    NegativeInfinity is a singleton, and can be accessed
    by ``S.NegativeInfinity``.

    See Also
    ========

    Infinity
    """

    is_extended_real = True
    is_complex = False
    is_commutative = True
    is_infinite = True
    is_comparable = True
    is_extended_negative = True
    is_number = True
    is_prime = False

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return r"-\infty"

    def _eval_subs(self, old, new):
        if self == old:
            return new

    def _eval_evalf(self, prec=None):
        return Float('-inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.Infinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.NegativeInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        return (-self).__add__(other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.Infinity
        return Number.__mul__(self, other)
    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity or \
                other is S.NegativeInfinity or \
                    other is S.NaN:
                return S.NaN
            if other.is_extended_nonnegative:
                return self
            return S.Infinity
        return Number.__truediv__(self, other)

    def __abs__(self):
        return S.Infinity

    def __neg__(self):
        return S.Infinity

    def _eval_power(self, expt):
        """
        ``expt`` is symbolic object but not equal to 0 or 1.

        ================ ======= ==============================
        Expression       Result  Notes
        ================ ======= ==============================
        ``(-oo) ** nan`` ``nan``
        ``(-oo) ** oo``  ``nan``
        ``(-oo) ** -oo`` ``nan``
        ``(-oo) ** e``   ``oo``  ``e`` is positive even integer
        ``(-oo) ** o``   ``-oo`` ``o`` is positive odd integer
        ================ ======= ==============================

        See Also
        ========

        Infinity
        Pow
        NaN

        """
        if expt.is_number:
            if expt is S.NaN or \
                expt is S.Infinity or \
                    expt is S.NegativeInfinity:
                return S.NaN

            if isinstance(expt, Integer) and expt.is_extended_positive:
                if expt.is_odd:
                    return S.NegativeInfinity
                else:
                    return S.Infinity

            inf_part = S.Infinity**expt
            s_part = S.NegativeOne**expt
            if inf_part == 0 and s_part.is_finite:
                return inf_part
            if (inf_part is S.ComplexInfinity and
                    s_part.is_finite and not s_part.is_zero):
                return S.ComplexInfinity
            return s_part*inf_part

    def _as_mpf_val(self, prec):
        return mlib.fninf

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return other is S.NegativeInfinity or other == float('-inf')

    def __ne__(self, other):
        return other is not S.NegativeInfinity and other != float('-inf')

    __gt__ = Expr.__gt__
    __ge__ = Expr.__ge__
    __lt__ = Expr.__lt__
    __le__ = Expr.__le__

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self

    def as_powers_dict(self):
        return {S.NegativeOne: 1, S.Infinity: 1}


class NaN(Number, metaclass=Singleton):
    """
    Not a Number.

    Explanation
    ===========

    This serves as a place holder for numeric values that are indeterminate.
    Most operations on NaN, produce another NaN.  Most indeterminate forms,
    such as ``0/0`` or ``oo - oo` produce NaN.  Two exceptions are ``0**0``
    and ``oo**0``, which all produce ``1`` (this is consistent with Python's
    float).

    NaN is loosely related to floating point nan, which is defined in the
    IEEE 754 floating point standard, and corresponds to the Python
    ``float('nan')``.  Differences are noted below.

    NaN is mathematically not equal to anything else, even NaN itself.  This
    explains the initially counter-intuitive results with ``Eq`` and ``==`` in
    the examples below.

    NaN is not comparable so inequalities raise a TypeError.  This is in
    contrast with floating point nan where all inequalities are false.

    NaN is a singleton, and can be accessed by ``S.NaN``, or can be imported
    as ``nan``.

    Examples
    ========

    >>> from sympy import nan, S, oo, Eq
    >>> nan is S.NaN
    True
    >>> oo - oo
    nan
    >>> nan + 1
    nan
    >>> Eq(nan, nan)   # mathematical equality
    False
    >>> nan == nan     # structural equality
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/NaN

    """
    is_commutative = True
    is_extended_real = None
    is_real = None
    is_rational = None
    is_algebraic = None
    is_transcendental = None
    is_integer = None
    is_comparable = False
    is_finite = None
    is_zero = None
    is_prime = None
    is_positive = None
    is_negative = None
    is_number = True

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return r"\text{NaN}"

    def __neg__(self):
        return self

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        return self

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        return self

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        return self

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        return self

    def floor(self):
        return self

    def ceiling(self):
        return self

    def _as_mpf_val(self, prec):
        return _mpf_nan

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        # NaN is structurally equal to another NaN
        return other is S.NaN

    def __ne__(self, other):
        return other is not S.NaN

    # Expr will _sympify and raise TypeError
    __gt__ = Expr.__gt__
    __ge__ = Expr.__ge__
    __lt__ = Expr.__lt__
    __le__ = Expr.__le__

nan = S.NaN

@dispatch(NaN, Expr) # type:ignore
def _eval_is_eq(a, b): # noqa:F811
    return False


class ComplexInfinity(AtomicExpr, metaclass=Singleton):
    r"""Complex infinity.

    Explanation
    ===========

    In complex analysis the symbol `\tilde\infty`, called "complex
    infinity", represents a quantity with infinite magnitude, but
    undetermined complex phase.

    ComplexInfinity is a singleton, and can be accessed by
    ``S.ComplexInfinity``, or can be imported as ``zoo``.

    Examples
    ========

    >>> from sympy import zoo
    >>> zoo + 42
    zoo
    >>> 42/zoo
    0
    >>> zoo + zoo
    nan
    >>> zoo*zoo
    zoo

    See Also
    ========

    Infinity
    """

    is_commutative = True
    is_infinite = True
    is_number = True
    is_prime = False
    is_complex = False
    is_extended_real = False

    kind = NumberKind

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return r"\tilde{\infty}"

    @staticmethod
    def __abs__():
        return S.Infinity

    def floor(self):
        return self

    def ceiling(self):
        return self

    @staticmethod
    def __neg__():
        return S.ComplexInfinity

    def _eval_power(self, expt):
        if expt is S.ComplexInfinity:
            return S.NaN

        if isinstance(expt, Number):
            if expt.is_zero:
                return S.NaN
            else:
                if expt.is_positive:
                    return S.ComplexInfinity
                else:
                    return S.Zero


zoo = S.ComplexInfinity


class NumberSymbol(AtomicExpr):

    is_commutative = True
    is_finite = True
    is_number = True

    __slots__ = ()

    is_NumberSymbol = True

    kind = NumberKind

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def approximation(self, number_cls):
        """ Return an interval with number_cls endpoints
        that contains the value of NumberSymbol.
        If not implemented, then return None.
        """

    def _eval_evalf(self, prec):
        return Float._new(self._as_mpf_val(prec), prec)

    def __eq__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if self is other:
            return True
        if other.is_Number and self.is_irrational:
            return False

        return False    # NumberSymbol != non-(Number|self)

    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        if self is other:
            return S.true
        return Expr.__le__(self, other)

    def __ge__(self, other):
        if self is other:
            return S.true
        return Expr.__ge__(self, other)

    def __int__(self):
        # subclass with appropriate return value
        raise NotImplementedError

    def __hash__(self):
        return super().__hash__()


class Exp1(NumberSymbol, metaclass=Singleton):
    r"""The `e` constant.

    Explanation
    ===========

    The transcendental number `e = 2.718281828\ldots` is the base of the
    natural logarithm and of the exponential function, `e = \exp(1)`.
    Sometimes called Euler's number or Napier's constant.

    Exp1 is a singleton, and can be accessed by ``S.Exp1``,
    or can be imported as ``E``.

    Examples
    ========

    >>> from sympy import exp, log, E
    >>> E is exp(1)
    True
    >>> log(E)
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/E_%28mathematical_constant%29
    """

    is_real = True
    is_positive = True
    is_negative = False  # XXX Forces is_negative/is_nonnegative
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True

    __slots__ = ()

    def _latex(self, printer):
        return r"e"

    @staticmethod
    def __abs__():
        return S.Exp1

    def __int__(self):
        return 2

    def _as_mpf_val(self, prec):
        return mpf_e(prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(2), Integer(3))
        elif issubclass(number_cls, Rational):
            pass

    def _eval_power(self, expt):
        if global_parameters.exp_is_pow:
            return self._eval_power_exp_is_pow(expt)
        else:
            from sympy.functions.elementary.exponential import exp
            return exp(expt)

    def _eval_power_exp_is_pow(self, arg):
        if arg.is_Number:
            if arg is oo:
                return oo
            elif arg == -oo:
                return S.Zero
        from sympy.functions.elementary.exponential import log
        if isinstance(arg, log):
            return arg.args[0]

        # don't autoexpand Pow or Mul (see the issue 3351):
        elif not arg.is_Add:
            Ioo = I*oo
            if arg in [Ioo, -Ioo]:
                return nan

            coeff = arg.coeff(pi*I)
            if coeff:
                if (2*coeff).is_integer:
                    if coeff.is_even:
                        return S.One
                    elif coeff.is_odd:
                        return S.NegativeOne
                    elif (coeff + S.Half).is_even:
                        return -I
                    elif (coeff + S.Half).is_odd:
                        return I
                elif coeff.is_Rational:
                    ncoeff = coeff % 2 # restrict to [0, 2pi)
                    if ncoeff > 1: # restrict to (-pi, pi]
                        ncoeff -= 2
                    if ncoeff != coeff:
                        return S.Exp1**(ncoeff*S.Pi*S.ImaginaryUnit)

            # Warning: code in risch.py will be very sensitive to changes
            # in this (see DifferentialExtension).

            # look for a single log factor

            coeff, terms = arg.as_coeff_Mul()

            # but it can't be multiplied by oo
            if coeff in (oo, -oo):
                return

            coeffs, log_term = [coeff], None
            for term in Mul.make_args(terms):
                if isinstance(term, log):
                    if log_term is None:
                        log_term = term.args[0]
                    else:
                        return
                elif term.is_comparable:
                    coeffs.append(term)
                else:
                    return

            return log_term**Mul(*coeffs) if log_term else None
        elif arg.is_Add:
            out = []
            add = []
            argchanged = False
            for a in arg.args:
                if a is S.One:
                    add.append(a)
                    continue
                newa = self**a
                if isinstance(newa, Pow) and newa.base is self:
                    if newa.exp != a:
                        add.append(newa.exp)
                        argchanged = True
                    else:
                        add.append(a)
                else:
                    out.append(newa)
            if out or argchanged:
                return Mul(*out)*Pow(self, Add(*add), evaluate=False)
        elif arg.is_Matrix:
            return arg.exp()

    def _eval_rewrite_as_sin(self, **kwargs):
        from sympy.functions.elementary.trigonometric import sin
        return sin(I + S.Pi/2) - I*sin(I)

    def _eval_rewrite_as_cos(self, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        return cos(I) + I*cos(I + S.Pi/2)

E = S.Exp1


class Pi(NumberSymbol, metaclass=Singleton):
    r"""The `\pi` constant.

    Explanation
    ===========

    The transcendental number `\pi = 3.141592654\ldots` represents the ratio
    of a circle's circumference to its diameter, the area of the unit circle,
    the half-period of trigonometric functions, and many other things
    in mathematics.

    Pi is a singleton, and can be accessed by ``S.Pi``, or can
    be imported as ``pi``.

    Examples
    ========

    >>> from sympy import S, pi, oo, sin, exp, integrate, Symbol
    >>> S.Pi
    pi
    >>> pi > 3
    True
    >>> pi.is_irrational
    True
    >>> x = Symbol('x')
    >>> sin(x + 2*pi)
    sin(x)
    >>> integrate(exp(-x**2), (x, -oo, oo))
    sqrt(pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pi
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True

    __slots__ = ()

    def _latex(self, printer):
        return r"\pi"

    @staticmethod
    def __abs__():
        return S.Pi

    def __int__(self):
        return 3

    def _as_mpf_val(self, prec):
        return mpf_pi(prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(3), Integer(4))
        elif issubclass(number_cls, Rational):
            return (Rational(223, 71, 1), Rational(22, 7, 1))

pi = S.Pi


class GoldenRatio(NumberSymbol, metaclass=Singleton):
    r"""The golden ratio, `\phi`.

    Explanation
    ===========

    `\phi = \frac{1 + \sqrt{5}}{2}` is an algebraic number.  Two quantities
    are in the golden ratio if their ratio is the same as the ratio of
    their sum to the larger of the two quantities, i.e. their maximum.

    GoldenRatio is a singleton, and can be accessed by ``S.GoldenRatio``.

    Examples
    ========

    >>> from sympy import S
    >>> S.GoldenRatio > 1
    True
    >>> S.GoldenRatio.expand(func=True)
    1/2 + sqrt(5)/2
    >>> S.GoldenRatio.is_irrational
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Golden_ratio
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = True
    is_transcendental = False

    __slots__ = ()

    def _latex(self, printer):
        return r"\phi"

    def __int__(self):
        return 1

    def _as_mpf_val(self, prec):
         # XXX track down why this has to be increased
        rv = mlib.from_man_exp(phi_fixed(prec + 10), -prec - 10)
        return mpf_norm(rv, prec)

    def _eval_expand_func(self, **hints):
        from sympy.functions.elementary.miscellaneous import sqrt
        return S.Half + S.Half*sqrt(5)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.One, Rational(2))
        elif issubclass(number_cls, Rational):
            pass

    _eval_rewrite_as_sqrt = _eval_expand_func


class TribonacciConstant(NumberSymbol, metaclass=Singleton):
    r"""The tribonacci constant.

    Explanation
    ===========

    The tribonacci numbers are like the Fibonacci numbers, but instead
    of starting with two predetermined terms, the sequence starts with
    three predetermined terms and each term afterwards is the sum of the
    preceding three terms.

    The tribonacci constant is the ratio toward which adjacent tribonacci
    numbers tend. It is a root of the polynomial `x^3 - x^2 - x - 1 = 0`,
    and also satisfies the equation `x + x^{-3} = 2`.

    TribonacciConstant is a singleton, and can be accessed
    by ``S.TribonacciConstant``.

    Examples
    ========

    >>> from sympy import S
    >>> S.TribonacciConstant > 1
    True
    >>> S.TribonacciConstant.expand(func=True)
    1/3 + (19 - 3*sqrt(33))**(1/3)/3 + (3*sqrt(33) + 19)**(1/3)/3
    >>> S.TribonacciConstant.is_irrational
    True
    >>> S.TribonacciConstant.n(20)
    1.8392867552141611326

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalizations_of_Fibonacci_numbers#Tribonacci_numbers
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = True
    is_transcendental = False

    __slots__ = ()

    def _latex(self, printer):
        return r"\text{TribonacciConstant}"

    def __int__(self):
        return 1

    def _as_mpf_val(self, prec):
        return self._eval_evalf(prec)._mpf_

    def _eval_evalf(self, prec):
        rv = self._eval_expand_func(function=True)._eval_evalf(prec + 4)
        return Float(rv, precision=prec)

    def _eval_expand_func(self, **hints):
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        return (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.One, Rational(2))
        elif issubclass(number_cls, Rational):
            pass

    _eval_rewrite_as_sqrt = _eval_expand_func


class EulerGamma(NumberSymbol, metaclass=Singleton):
    r"""The Euler-Mascheroni constant.

    Explanation
    ===========

    `\gamma = 0.5772157\ldots` (also called Euler's constant) is a mathematical
    constant recurring in analysis and number theory.  It is defined as the
    limiting difference between the harmonic series and the
    natural logarithm:

    .. math:: \gamma = \lim\limits_{n\to\infty}
              \left(\sum\limits_{k=1}^n\frac{1}{k} - \ln n\right)

    EulerGamma is a singleton, and can be accessed by ``S.EulerGamma``.

    Examples
    ========

    >>> from sympy import S
    >>> S.EulerGamma.is_irrational
    >>> S.EulerGamma > 0
    True
    >>> S.EulerGamma > 1
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = None
    is_number = True

    __slots__ = ()

    def _latex(self, printer):
        return r"\gamma"

    def __int__(self):
        return 0

    def _as_mpf_val(self, prec):
         # XXX track down why this has to be increased
        v = mlib.libhyper.euler_fixed(prec + 10)
        rv = mlib.from_man_exp(v, -prec - 10)
        return mpf_norm(rv, prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.Zero, S.One)
        elif issubclass(number_cls, Rational):
            return (S.Half, Rational(3, 5, 1))


class Catalan(NumberSymbol, metaclass=Singleton):
    r"""Catalan's constant.

    Explanation
    ===========

    $G = 0.91596559\ldots$ is given by the infinite series

    .. math:: G = \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)^2}

    Catalan is a singleton, and can be accessed by ``S.Catalan``.

    Examples
    ========

    >>> from sympy import S
    >>> S.Catalan.is_irrational
    >>> S.Catalan > 0
    True
    >>> S.Catalan > 1
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Catalan%27s_constant
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = None
    is_number = True

    __slots__ = ()

    def __int__(self):
        return 0

    def _as_mpf_val(self, prec):
        # XXX track down why this has to be increased
        v = mlib.catalan_fixed(prec + 10)
        rv = mlib.from_man_exp(v, -prec - 10)
        return mpf_norm(rv, prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.Zero, S.One)
        elif issubclass(number_cls, Rational):
            return (Rational(9, 10, 1), S.One)

    def _eval_rewrite_as_Sum(self, k_sym=None, symbols=None, **hints):
        if (k_sym is not None) or (symbols is not None):
            return self
        from .symbol import Dummy
        from sympy.concrete.summations import Sum
        k = Dummy('k', integer=True, nonnegative=True)
        return Sum(S.NegativeOne**k / (2*k+1)**2, (k, 0, S.Infinity))

    def _latex(self, printer):
        return "G"


class ImaginaryUnit(AtomicExpr, metaclass=Singleton):
    r"""The imaginary unit, `i = \sqrt{-1}`.

    I is a singleton, and can be accessed by ``S.I``, or can be
    imported as ``I``.

    Examples
    ========

    >>> from sympy import I, sqrt
    >>> sqrt(-1)
    I
    >>> I*I
    -1
    >>> 1/I
    -I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Imaginary_unit
    """

    is_commutative = True
    is_imaginary = True
    is_finite = True
    is_number = True
    is_algebraic = True
    is_transcendental = False

    kind = NumberKind

    __slots__ = ()

    def _latex(self, printer):
        return printer._settings['imaginary_unit_latex']

    @staticmethod
    def __abs__():
        return S.One

    def _eval_evalf(self, prec):
        return self

    def _eval_conjugate(self):
        return -S.ImaginaryUnit

    def _eval_power(self, expt):
        """
        b is I = sqrt(-1)
        e is symbolic object but not equal to 0, 1

        I**r -> (-1)**(r/2) -> exp(r/2*Pi*I) -> sin(Pi*r/2) + cos(Pi*r/2)*I, r is decimal
        I**0 mod 4 -> 1
        I**1 mod 4 -> I
        I**2 mod 4 -> -1
        I**3 mod 4 -> -I
        """

        if isinstance(expt, Integer):
            expt = expt % 4
            if expt == 0:
                return S.One
            elif expt == 1:
                return S.ImaginaryUnit
            elif expt == 2:
                return S.NegativeOne
            elif expt == 3:
                return -S.ImaginaryUnit
        if isinstance(expt, Rational):
            i, r = divmod(expt, 2)
            rv = Pow(S.ImaginaryUnit, r, evaluate=False)
            if i % 2:
                return Mul(S.NegativeOne, rv, evaluate=False)
            return rv

    def as_base_exp(self):
        return S.NegativeOne, S.Half

    @property
    def _mpc_(self):
        return (Float(0)._mpf_, Float(1)._mpf_)


I = S.ImaginaryUnit


def int_valued(x):
    """return True only for a literal Number whose internal
    representation as a fraction has a denominator of 1,
    else False, i.e. integer, with no fractional part.
    """
    if isinstance(x, (SYMPY_INTS, int)):
        return True
    if type(x) is float:
        return x.is_integer()
    if isinstance(x, Integer):
        return True
    if isinstance(x, Float):
        # x = s*m*2**p; _mpf_ = s,m,e,p
        return x._mpf_[2] >= 0
    return False  # or add new types to recognize


def equal_valued(x, y):
    """Compare expressions treating plain floats as rationals.

    Examples
    ========

    >>> from sympy import S, symbols, Rational, Float
    >>> from sympy.core.numbers import equal_valued
    >>> equal_valued(1, 2)
    False
    >>> equal_valued(1, 1)
    True

    In SymPy expressions with Floats compare unequal to corresponding
    expressions with rationals:

    >>> x = symbols('x')
    >>> x**2 == x**2.0
    False

    However an individual Float compares equal to a Rational:

    >>> Rational(1, 2) == Float(0.5)
    False

    In a future version of SymPy this might change so that Rational and Float
    compare unequal. This function provides the behavior currently expected of
    ``==`` so that it could still be used if the behavior of ``==`` were to
    change in future.

    >>> equal_valued(1, 1.0) # Float vs Rational
    True
    >>> equal_valued(S(1).n(3), S(1).n(5)) # Floats of different precision
    True

    Explanation
    ===========

    In future SymPy versions Float and Rational might compare unequal and floats
    with different precisions might compare unequal. In that context a function
    is needed that can check if a number is equal to 1 or 0 etc. The idea is
    that instead of testing ``if x == 1:`` if we want to accept floats like
    ``1.0`` as well then the test can be written as ``if equal_valued(x, 1):``
    or ``if equal_valued(x, 2):``. Since this function is intended to be used
    in situations where one or both operands are expected to be concrete
    numbers like 1 or 0 the function does not recurse through the args of any
    compound expression to compare any nested floats.

    References
    ==========

    .. [1] https://github.com/sympy/sympy/pull/20033
    """
    x = _sympify(x)
    y = _sympify(y)

    # Handle everything except Float/Rational first
    if not x.is_Float and not y.is_Float:
        return x == y
    elif x.is_Float and y.is_Float:
        # Compare values without regard for precision
        return x._mpf_ == y._mpf_
    elif x.is_Float:
        x, y = y, x
    if not x.is_Rational:
        return False

    # Now y is Float and x is Rational. A simple approach at this point would
    # just be x == Rational(y) but if y has a large exponent creating a
    # Rational could be prohibitively expensive.

    sign, man, exp, _ = y._mpf_
    p, q = x.p, x.q

    if sign:
        man = -man

    if exp == 0:
        # y odd integer
        return q == 1 and man == p
    elif exp > 0:
        # y even integer
        if q != 1:
            return False
        if p.bit_length() != man.bit_length() + exp:
            return False
        return man << exp == p
    else:
        # y non-integer. Need p == man and q == 2**-exp
        if p != man:
            return False
        neg_exp = -exp
        if q.bit_length() - 1 != neg_exp:
            return False
        return (1 << neg_exp) == q


def all_close(expr1, expr2, rtol=1e-5, atol=1e-8):
    """Return True if expr1 and expr2 are numerically close.

    The expressions must have the same structure, but any Rational, Integer, or
    Float numbers they contain are compared approximately using rtol and atol.
    Any other parts of expressions are compared exactly. However, allowance is
    made to allow for the additive and multiplicative identities.

    Relative tolerance is measured with respect to expr2 so when used in
    testing expr2 should be the expected correct answer.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.abc import x, y
    >>> from sympy.core.numbers import all_close
    >>> expr1 = 0.1*exp(x - y)
    >>> expr2 = exp(x - y)/10
    >>> expr1
    0.1*exp(x - y)
    >>> expr2
    exp(x - y)/10
    >>> expr1 == expr2
    False
    >>> all_close(expr1, expr2)
    True

    Identities are automatically supplied:

    >>> all_close(x, x + 1e-10)
    True
    >>> all_close(x, 1.0*x)
    True
    >>> all_close(x, 1.0*x + 1e-10)
    True

    """
    NUM_TYPES = (Rational, Float)

    def _all_close(obj1, obj2):
        if type(obj1) == type(obj2) and isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            return all(_all_close(e1, e2) for e1, e2 in zip(obj1, obj2))
        else:
            return _all_close_expr(_sympify(obj1), _sympify(obj2))

    def _all_close_expr(expr1, expr2):
        num1 = isinstance(expr1, NUM_TYPES)
        num2 = isinstance(expr2, NUM_TYPES)
        if num1 != num2:
            return False
        elif num1:
            return _close_num(expr1, expr2)
        if expr1.is_Add or expr1.is_Mul or expr2.is_Add or expr2.is_Mul:
            return _all_close_ac(expr1, expr2)
        if expr1.func != expr2.func or len(expr1.args) != len(expr2.args):
            return False
        args = zip(expr1.args, expr2.args)
        return all(_all_close_expr(a1, a2) for a1, a2 in args)

    def _close_num(num1, num2):
        return bool(abs(num1 - num2) <= atol + rtol*abs(num2))

    def _all_close_ac(expr1, expr2):
        # compare expressions with associative commutative operators for
        # approximate equality by seeing that all terms have equivalent
        # coefficients (which are always Rational or Float)
        if expr1.is_Mul or expr2.is_Mul:
            # as_coeff_mul automatically will supply coeff of 1
            c1, e1 = expr1.as_coeff_mul(rational=False)
            c2, e2 = expr2.as_coeff_mul(rational=False)
            if not _close_num(c1, c2):
                return False
            s1 = set(e1)
            s2 = set(e2)
            common = s1 & s2
            s1 -= common
            s2 -= common
            if not s1:
                return True
            if not any(i.has(Float) for j in (s1, s2) for i in j):
                return False
            # factors might not be matching, e.g.
            # x != x**1.0, exp(x) != exp(1.0*x), etc...
            s1 = [i.as_base_exp() for i in ordered(s1)]
            s2 = [i.as_base_exp() for i in ordered(s2)]
            unmatched = list(range(len(s1)))
            for be1 in s1:
                for i in unmatched:
                    be2 = s2[i]
                    if _all_close(be1, be2):
                        unmatched.remove(i)
                        break
                else:
                    return False
            return not(unmatched)
        assert expr1.is_Add or expr2.is_Add
        cd1 = expr1.as_coefficients_dict()
        cd2 = expr2.as_coefficients_dict()
        # this test will assure that the key of 1 is in
        # each dict and that they have equal values
        if not _close_num(cd1[1], cd2[1]):
            return False
        if len(cd1) != len(cd2):
            return False
        for k in list(cd1):
            if k in cd2:
                if not _close_num(cd1.pop(k), cd2.pop(k)):
                    return False
            # k (or a close version in cd2) might have
            # Floats in a factor of the term which will
            # be handled below
        else:
            if not cd1:
                return True
        for k1 in cd1:
            for k2 in cd2:
                if _all_close_expr(k1, k2):
                    # found a matching key
                    # XXX there could be a corner case where
                    # more than 1 might match and the numbers are
                    # such that one is better than the other
                    # that is not being considered here
                    if not _close_num(cd1[k1], cd2[k2]):
                        return False
                    break
            else:
                # no key matched
                return False
        return True

    return _all_close(expr1, expr2)


@dispatch(Tuple, Number) # type:ignore
def _eval_is_eq(self, other): # noqa: F811
    return False


def sympify_fractions(f):
    return Rational._new(f.numerator, f.denominator, 1)

_sympy_converter[fractions.Fraction] = sympify_fractions


if gmpy is not None:

    def sympify_mpz(x):
        return Integer(int(x))

    def sympify_mpq(x):
        return Rational(int(x.numerator), int(x.denominator))

    _sympy_converter[type(gmpy.mpz(1))] = sympify_mpz
    _sympy_converter[type(gmpy.mpq(1, 2))] = sympify_mpq


if flint is not None:

    def sympify_fmpz(x):
        return Integer(int(x))

    def sympify_fmpq(x):
        return Rational(int(x.numerator), int(x.denominator))

    _sympy_converter[type(flint.fmpz(1))] = sympify_fmpz
    _sympy_converter[type(flint.fmpq(1, 2))] = sympify_fmpq


def sympify_mpmath(x):
    return Expr._from_mpmath(x, x.context.prec)

_sympy_converter[mpnumeric] = sympify_mpmath


def sympify_complex(a):
    real, imag = list(map(sympify, (a.real, a.imag)))
    return real + S.ImaginaryUnit*imag

_sympy_converter[complex] = sympify_complex

from .power import Pow
from .mul import Mul
Mul.identity = One()
from .add import Add
Add.identity = Zero()


def _register_classes():
    numbers.Number.register(Number)
    numbers.Real.register(Float)
    numbers.Rational.register(Rational)
    numbers.Integral.register(Integer)

_register_classes()

_illegal = (S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity)
