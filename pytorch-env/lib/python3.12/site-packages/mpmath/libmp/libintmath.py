"""
Utility functions for integer math.

TODO: rename, cleanup, perhaps move the gmpy wrapper code
here from settings.py

"""

import math
from bisect import bisect

from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO

small_trailing = [0] * 256
for j in range(1,8):
    small_trailing[1<<j::1<<(j+1)] = [j] * (1<<(7-j))

def giant_steps(start, target, n=2):
    """
    Return a list of integers ~=

    [start, n*start, ..., target/n^2, target/n, target]

    but conservatively rounded so that the quotient between two
    successive elements is actually slightly less than n.

    With n = 2, this describes suitable precision steps for a
    quadratically convergent algorithm such as Newton's method;
    with n = 3 steps for cubic convergence (Halley's method), etc.

        >>> giant_steps(50,1000)
        [66, 128, 253, 502, 1000]
        >>> giant_steps(50,1000,4)
        [65, 252, 1000]

    """
    L = [target]
    while L[-1] > start*n:
        L = L + [L[-1]//n + 2]
    return L[::-1]

def rshift(x, n):
    """For an integer x, calculate x >> n with the fastest (floor)
    rounding. Unlike the plain Python expression (x >> n), n is
    allowed to be negative, in which case a left shift is performed."""
    if n >= 0: return x >> n
    else:      return x << (-n)

def lshift(x, n):
    """For an integer x, calculate x << n. Unlike the plain Python
    expression (x << n), n is allowed to be negative, in which case a
    right shift with default (floor) rounding is performed."""
    if n >= 0: return x << n
    else:      return x >> (-n)

if BACKEND == 'sage':
    import operator
    rshift = operator.rshift
    lshift = operator.lshift

def python_trailing(n):
    """Count the number of trailing zero bits in abs(n)."""
    if not n:
        return 0
    low_byte = n & 0xff
    if low_byte:
        return small_trailing[low_byte]
    t = 8
    n >>= 8
    while not n & 0xff:
        n >>= 8
        t += 8
    return t + small_trailing[n & 0xff]

if BACKEND == 'gmpy':
    if gmpy.version() >= '2':
        def gmpy_trailing(n):
            """Count the number of trailing zero bits in abs(n) using gmpy."""
            if n: return MPZ(n).bit_scan1()
            else: return 0
    else:
        def gmpy_trailing(n):
            """Count the number of trailing zero bits in abs(n) using gmpy."""
            if n: return MPZ(n).scan1()
            else: return 0

# Small powers of 2
powers = [1<<_ for _ in range(300)]

def python_bitcount(n):
    """Calculate bit size of the nonnegative integer n."""
    bc = bisect(powers, n)
    if bc != 300:
        return bc
    bc = int(math.log(n, 2)) - 4
    return bc + bctable[n>>bc]

def gmpy_bitcount(n):
    """Calculate bit size of the nonnegative integer n."""
    if n: return MPZ(n).numdigits(2)
    else: return 0

#def sage_bitcount(n):
#    if n: return MPZ(n).nbits()
#    else: return 0

def sage_trailing(n):
    return MPZ(n).trailing_zero_bits()

if BACKEND == 'gmpy':
    bitcount = gmpy_bitcount
    trailing = gmpy_trailing
elif BACKEND == 'sage':
    sage_bitcount = sage_utils.bitcount
    bitcount = sage_bitcount
    trailing = sage_trailing
else:
    bitcount = python_bitcount
    trailing = python_trailing

if BACKEND == 'gmpy' and 'bit_length' in dir(gmpy):
    bitcount = gmpy.bit_length

# Used to avoid slow function calls as far as possible
trailtable = [trailing(n) for n in range(256)]
bctable = [bitcount(n) for n in range(1024)]

# TODO: speed up for bases 2, 4, 8, 16, ...

def bin_to_radix(x, xbits, base, bdigits):
    """Changes radix of a fixed-point number; i.e., converts
    x * 2**xbits to floor(x * 10**bdigits)."""
    return x * (MPZ(base)**bdigits) >> xbits

stddigits = '0123456789abcdefghijklmnopqrstuvwxyz'

def small_numeral(n, base=10, digits=stddigits):
    """Return the string numeral of a positive integer in an arbitrary
    base. Most efficient for small input."""
    if base == 10:
        return str(n)
    digs = []
    while n:
        n, digit = divmod(n, base)
        digs.append(digits[digit])
    return "".join(digs[::-1])

def numeral_python(n, base=10, size=0, digits=stddigits):
    """Represent the integer n as a string of digits in the given base.
    Recursive division is used to make this function about 3x faster
    than Python's str() for converting integers to decimal strings.

    The 'size' parameters specifies the number of digits in n; this
    number is only used to determine splitting points and need not be
    exact."""
    if n <= 0:
        if not n:
            return "0"
        return "-" + numeral(-n, base, size, digits)
    # Fast enough to do directly
    if size < 250:
        return small_numeral(n, base, digits)
    # Divide in half
    half = (size // 2) + (size & 1)
    A, B = divmod(n, base**half)
    ad = numeral(A, base, half, digits)
    bd = numeral(B, base, half, digits).rjust(half, "0")
    return ad + bd

def numeral_gmpy(n, base=10, size=0, digits=stddigits):
    """Represent the integer n as a string of digits in the given base.
    Recursive division is used to make this function about 3x faster
    than Python's str() for converting integers to decimal strings.

    The 'size' parameters specifies the number of digits in n; this
    number is only used to determine splitting points and need not be
    exact."""
    if n < 0:
        return "-" + numeral(-n, base, size, digits)
    # gmpy.digits() may cause a segmentation fault when trying to convert
    # extremely large values to a string. The size limit may need to be
    # adjusted on some platforms, but 1500000 works on Windows and Linux.
    if size < 1500000:
        return gmpy.digits(n, base)
    # Divide in half
    half = (size // 2) + (size & 1)
    A, B = divmod(n, MPZ(base)**half)
    ad = numeral(A, base, half, digits)
    bd = numeral(B, base, half, digits).rjust(half, "0")
    return ad + bd

if BACKEND == "gmpy":
    numeral = numeral_gmpy
else:
    numeral = numeral_python

_1_800 = 1<<800
_1_600 = 1<<600
_1_400 = 1<<400
_1_200 = 1<<200
_1_100 = 1<<100
_1_50 = 1<<50

def isqrt_small_python(x):
    """
    Correctly (floor) rounded integer square root, using
    division. Fast up to ~200 digits.
    """
    if not x:
        return x
    if x < _1_800:
        # Exact with IEEE double precision arithmetic
        if x < _1_50:
            return int(x**0.5)
        # Initial estimate can be any integer >= the true root; round up
        r = int(x**0.5 * 1.00000000000001) + 1
    else:
        bc = bitcount(x)
        n = bc//2
        r = int((x>>(2*n-100))**0.5+2)<<(n-50)  # +2 is to round up
    # The following iteration now precisely computes floor(sqrt(x))
    # See e.g. Crandall & Pomerance, "Prime Numbers: A Computational
    # Perspective"
    while 1:
        y = (r+x//r)>>1
        if y >= r:
            return r
        r = y

def isqrt_fast_python(x):
    """
    Fast approximate integer square root, computed using division-free
    Newton iteration for large x. For random integers the result is almost
    always correct (floor(sqrt(x))), but is 1 ulp too small with a roughly
    0.1% probability. If x is very close to an exact square, the answer is
    1 ulp wrong with high probability.

    With 0 guard bits, the largest error over a set of 10^5 random
    inputs of size 1-10^5 bits was 3 ulp. The use of 10 guard bits
    almost certainly guarantees a max 1 ulp error.
    """
    # Use direct division-based iteration if sqrt(x) < 2^400
    # Assume floating-point square root accurate to within 1 ulp, then:
    # 0 Newton iterations good to 52 bits
    # 1 Newton iterations good to 104 bits
    # 2 Newton iterations good to 208 bits
    # 3 Newton iterations good to 416 bits
    if x < _1_800:
        y = int(x**0.5)
        if x >= _1_100:
            y = (y + x//y) >> 1
            if x >= _1_200:
                y = (y + x//y) >> 1
                if x >= _1_400:
                    y = (y + x//y) >> 1
        return y
    bc = bitcount(x)
    guard_bits = 10
    x <<= 2*guard_bits
    bc += 2*guard_bits
    bc += (bc&1)
    hbc = bc//2
    startprec = min(50, hbc)
    # Newton iteration for 1/sqrt(x), with floating-point starting value
    r = int(2.0**(2*startprec) * (x >> (bc-2*startprec)) ** -0.5)
    pp = startprec
    for p in giant_steps(startprec, hbc):
        # r**2, scaled from real size 2**(-bc) to 2**p
        r2 = (r*r) >> (2*pp - p)
        # x*r**2, scaled from real size ~1.0 to 2**p
        xr2 = ((x >> (bc-p)) * r2) >> p
        # New value of r, scaled from real size 2**(-bc/2) to 2**p
        r = (r * ((3<<p) - xr2)) >> (pp+1)
        pp = p
    # (1/sqrt(x))*x = sqrt(x)
    return (r*(x>>hbc)) >> (p+guard_bits)

def sqrtrem_python(x):
    """Correctly rounded integer (floor) square root with remainder."""
    # to check cutoff:
    # plot(lambda x: timing(isqrt, 2**int(x)), [0,2000])
    if x < _1_600:
        y = isqrt_small_python(x)
        return y, x - y*y
    y = isqrt_fast_python(x) + 1
    rem = x - y*y
    # Correct remainder
    while rem < 0:
        y -= 1
        rem += (1+2*y)
    else:
        if rem:
            while rem > 2*(1+y):
                y += 1
                rem -= (1+2*y)
    return y, rem

def isqrt_python(x):
    """Integer square root with correct (floor) rounding."""
    return sqrtrem_python(x)[0]

def sqrt_fixed(x, prec):
    return isqrt_fast(x<<prec)

sqrt_fixed2 = sqrt_fixed

if BACKEND == 'gmpy':
    if gmpy.version() >= '2':
        isqrt_small = isqrt_fast = isqrt = gmpy.isqrt
        sqrtrem = gmpy.isqrt_rem
    else:
        isqrt_small = isqrt_fast = isqrt = gmpy.sqrt
        sqrtrem = gmpy.sqrtrem
elif BACKEND == 'sage':
    isqrt_small = isqrt_fast = isqrt = \
        getattr(sage_utils, "isqrt", lambda n: MPZ(n).isqrt())
    sqrtrem = lambda n: MPZ(n).sqrtrem()
else:
    isqrt_small = isqrt_small_python
    isqrt_fast = isqrt_fast_python
    isqrt = isqrt_python
    sqrtrem = sqrtrem_python


def ifib(n, _cache={}):
    """Computes the nth Fibonacci number as an integer, for
    integer n."""
    if n < 0:
        return (-1)**(-n+1) * ifib(-n)
    if n in _cache:
        return _cache[n]
    m = n
    # Use Dijkstra's logarithmic algorithm
    # The following implementation is basically equivalent to
    # http://en.literateprograms.org/Fibonacci_numbers_(Scheme)
    a, b, p, q = MPZ_ONE, MPZ_ZERO, MPZ_ZERO, MPZ_ONE
    while n:
        if n & 1:
            aq = a*q
            a, b = b*q+aq+a*p, b*p+aq
            n -= 1
        else:
            qq = q*q
            p, q = p*p+qq, qq+2*p*q
            n >>= 1
    if m < 250:
        _cache[m] = b
    return b

MAX_FACTORIAL_CACHE = 1000

def ifac(n, memo={0:1, 1:1}):
    """Return n factorial (for integers n >= 0 only)."""
    f = memo.get(n)
    if f:
        return f
    k = len(memo)
    p = memo[k-1]
    MAX = MAX_FACTORIAL_CACHE
    while k <= n:
        p *= k
        if k <= MAX:
            memo[k] = p
        k += 1
    return p

def ifac2(n, memo_pair=[{0:1}, {1:1}]):
    """Return n!! (double factorial), integers n >= 0 only."""
    memo = memo_pair[n&1]
    f = memo.get(n)
    if f:
        return f
    k = max(memo)
    p = memo[k]
    MAX = MAX_FACTORIAL_CACHE
    while k < n:
        k += 2
        p *= k
        if k <= MAX:
            memo[k] = p
    return p

if BACKEND == 'gmpy':
    ifac = gmpy.fac
elif BACKEND == 'sage':
    ifac = lambda n: int(sage.factorial(n))
    ifib = sage.fibonacci

def list_primes(n):
    n = n + 1
    sieve = list(xrange(n))
    sieve[:2] = [0, 0]
    for i in xrange(2, int(n**0.5)+1):
        if sieve[i]:
            for j in xrange(i**2, n, i):
                sieve[j] = 0
    return [p for p in sieve if p]

if BACKEND == 'sage':
    # Note: it is *VERY* important for performance that we convert
    # the list to Python ints.
    def list_primes(n):
        return [int(_) for _ in sage.primes(n+1)]

small_odd_primes = (3,5,7,11,13,17,19,23,29,31,37,41,43,47)
small_odd_primes_set = set(small_odd_primes)

def isprime(n):
    """
    Determines whether n is a prime number. A probabilistic test is
    performed if n is very large. No special trick is used for detecting
    perfect powers.

        >>> sum(list_primes(100000))
        454396537
        >>> sum(n*isprime(n) for n in range(100000))
        454396537

    """
    n = int(n)
    if not n & 1:
        return n == 2
    if n < 50:
        return n in small_odd_primes_set
    for p in small_odd_primes:
        if not n % p:
            return False
    m = n-1
    s = trailing(m)
    d = m >> s
    def test(a):
        x = pow(a,d,n)
        if x == 1 or x == m:
            return True
        for r in xrange(1,s):
            x = x**2 % n
            if x == m:
                return True
        return False
    # See http://primes.utm.edu/prove/prove2_3.html
    if n < 1373653:
        witnesses = [2,3]
    elif n < 341550071728321:
        witnesses = [2,3,5,7,11,13,17]
    else:
        witnesses = small_odd_primes
    for a in witnesses:
        if not test(a):
            return False
    return True

def moebius(n):
    """
    Evaluates the Moebius function which is `mu(n) = (-1)^k` if `n`
    is a product of `k` distinct primes and `mu(n) = 0` otherwise.

    TODO: speed up using factorization
    """
    n = abs(int(n))
    if n < 2:
        return n
    factors = []
    for p in xrange(2, n+1):
        if not (n % p):
            if not (n % p**2):
                return 0
            if not sum(p % f for f in factors):
                factors.append(p)
    return (-1)**len(factors)

def gcd(*args):
    a = 0
    for b in args:
        if a:
            while b:
                a, b = b, a % b
        else:
            a = b
    return a


#  Comment by Juan Arias de Reyna:
#
#  I learn this method to compute EulerE[2n] from van de Lune.
#
#  We apply the formula   EulerE[2n] = (-1)^n 2**(-2n) sum_{j=0}^n a(2n,2j+1)
#
#  where the numbers a(n,j) vanish for  j > n+1 or j <= -1  and satisfies
#
#  a(0,-1) = a(0,0) = 0;  a(0,1)= 1; a(0,2) = a(0,3) = 0
#
#  a(n,j) = a(n-1,j)                              when n+j is even
#  a(n,j) = (j-1) a(n-1,j-1) + (j+1) a(n-1,j+1)   when n+j is odd
#
#
#  But we can use only one array unidimensional a(j) since to compute
#  a(n,j) we only need to know a(n-1,k) where k and j are of different parity
#  and we have not to conserve the used values.
#
#  We cached up the values of Euler numbers to sufficiently high order.
#
#  Important Observation: If we pretend to use the numbers
#     EulerE[1], EulerE[2], ... , EulerE[n]
#     it is convenient to compute first EulerE[n], since the algorithm
#     computes first all
#     the previous ones, and keeps them in the CACHE

MAX_EULER_CACHE = 500

def eulernum(m, _cache={0:MPZ_ONE}):
    r"""
    Computes the Euler numbers `E(n)`, which can be defined as
    coefficients of the Taylor expansion of `1/cosh x`:

    .. math ::

        \frac{1}{\cosh x} = \sum_{n=0}^\infty \frac{E_n}{n!} x^n

    Example::

        >>> [int(eulernum(n)) for n in range(11)]
        [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]
        >>> [int(eulernum(n)) for n in range(11)]   # test cache
        [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]

    """
    # for odd m > 1, the Euler numbers are zero
    if m & 1:
        return MPZ_ZERO
    f = _cache.get(m)
    if f:
        return f
    MAX = MAX_EULER_CACHE
    n = m
    a = [MPZ(_) for _ in [0,0,1,0,0,0]]
    for  n in range(1, m+1):
        for j in range(n+1, -1, -2):
            a[j+1] = (j-1)*a[j] + (j+1)*a[j+2]
        a.append(0)
        suma = 0
        for k in range(n+1, -1, -2):
            suma += a[k+1]
            if n <= MAX:
                _cache[n] = ((-1)**(n//2))*(suma // 2**n)
        if n == m:
            return ((-1)**(n//2))*suma // 2**n

def stirling1(n, k):
    """
    Stirling number of the first kind.
    """
    if n < 0 or k < 0:
        raise ValueError
    if k >= n:
        return MPZ(n == k)
    if k < 1:
        return MPZ_ZERO
    L = [MPZ_ZERO] * (k+1)
    L[1] = MPZ_ONE
    for m in xrange(2, n+1):
        for j in xrange(min(k, m), 0, -1):
            L[j] = (m-1) * L[j] + L[j-1]
    return (-1)**(n+k) * L[k]

def stirling2(n, k):
    """
    Stirling number of the second kind.
    """
    if n < 0 or k < 0:
        raise ValueError
    if k >= n:
        return MPZ(n == k)
    if k <= 1:
        return MPZ(k == 1)
    s = MPZ_ZERO
    t = MPZ_ONE
    for j in xrange(k+1):
        if (k + j) & 1:
            s -= t * MPZ(j)**n
        else:
            s += t * MPZ(j)**n
        t = t * (k - j) // (j + 1)
    return s // ifac(k)
