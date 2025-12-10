# sympy.external.ntheory
#
# This module provides pure Python implementations of some number theory
# functions that are alternately used from gmpy2 if it is installed.

import math

import mpmath.libmp as mlib


_small_trailing = [0] * 256
for j in range(1, 8):
    _small_trailing[1 << j :: 1 << (j + 1)] = [j] * (1 << (7 - j))


def bit_scan1(x, n=0):
    if not x:
        return
    x = abs(x >> n)
    low_byte = x & 0xFF
    if low_byte:
        return _small_trailing[low_byte] + n

    t = 8 + n
    x >>= 8
    # 2**m is quick for z up through 2**30
    z = x.bit_length() - 1
    if x == 1 << z:
        return z + t

    if z < 300:
        # fixed 8-byte reduction
        while not x & 0xFF:
            x >>= 8
            t += 8
    else:
        # binary reduction important when there might be a large
        # number of trailing 0s
        p = z >> 1
        while not x & 0xFF:
            while x & ((1 << p) - 1):
                p >>= 1
            x >>= p
            t += p
    return t + _small_trailing[x & 0xFF]


def bit_scan0(x, n=0):
    return bit_scan1(x + (1 << n), n)


def remove(x, f):
    if f < 2:
        raise ValueError("factor must be > 1")
    if x == 0:
        return 0, 0
    if f == 2:
        b = bit_scan1(x)
        return x >> b, b
    m = 0
    y, rem = divmod(x, f)
    while not rem:
        x = y
        m += 1
        if m > 5:
            pow_list = [f**2]
            while pow_list:
                _f = pow_list[-1]
                y, rem = divmod(x, _f)
                if not rem:
                    m += 1 << len(pow_list)
                    x = y
                    pow_list.append(_f**2)
                else:
                    pow_list.pop()
        y, rem = divmod(x, f)
    return x, m


def factorial(x):
    """Return x!."""
    return int(mlib.ifac(int(x)))


def sqrt(x):
    """Integer square root of x."""
    return int(mlib.isqrt(int(x)))


def sqrtrem(x):
    """Integer square root of x and remainder."""
    s, r = mlib.sqrtrem(int(x))
    return (int(s), int(r))


gcd = math.gcd
lcm = math.lcm


def _sign(n):
    if n < 0:
        return -1, -n
    return 1, n


def gcdext(a, b):
    if not a or not b:
        g = abs(a) or abs(b)
        if not g:
            return (0, 0, 0)
        return (g, a // g, b // g)

    x_sign, a = _sign(a)
    y_sign, b = _sign(b)
    x, r = 1, 0
    y, s = 0, 1

    while b:
        q, c = divmod(a, b)
        a, b = b, c
        x, r = r, x - q*r
        y, s = s, y - q*s

    return (a, x * x_sign, y * y_sign)


def is_square(x):
    """Return True if x is a square number."""
    if x < 0:
        return False

    # Note that the possible values of y**2 % n for a given n are limited.
    # For example, when n=4, y**2 % n can only take 0 or 1.
    # In other words, if x % 4 is 2 or 3, then x is not a square number.
    # Mathematically, it determines if it belongs to the set {y**2 % n},
    # but implementationally, it can be realized as a logical conjunction
    # with an n-bit integer.
    # see https://mersenneforum.org/showpost.php?p=110896
    # def magic(n):
    #     s = {y**2 % n for y in range(n)}
    #     s = set(range(n)) - s
    #     return sum(1 << bit for bit in s)
    # >>> print(hex(magic(128)))
    # 0xfdfdfdedfdfdfdecfdfdfdedfdfcfdec
    # >>> print(hex(magic(99)))
    # 0x5f6f9ffb6fb7ddfcb75befdec
    # >>> print(hex(magic(91)))
    # 0x6fd1bfcfed5f3679d3ebdec
    # >>> print(hex(magic(85)))
    # 0xdef9ae771ffe3b9d67dec
    if 0xfdfdfdedfdfdfdecfdfdfdedfdfcfdec & (1 << (x & 127)):
        return False  # e.g. 2, 3
    m = x % 765765 # 765765 = 99 * 91 * 85
    if 0x5f6f9ffb6fb7ddfcb75befdec & (1 << (m % 99)):
        return False  # e.g. 17, 68
    if 0x6fd1bfcfed5f3679d3ebdec & (1 << (m % 91)):
        return False  # e.g. 97, 388
    if 0xdef9ae771ffe3b9d67dec & (1 << (m % 85)):
        return False  # e.g. 793, 1408
    return mlib.sqrtrem(int(x))[1] == 0


def invert(x, m):
    """Modular inverse of x modulo m.

    Returns y such that x*y == 1 mod m.

    Uses ``math.pow`` but reproduces the behaviour of ``gmpy2.invert``
    which raises ZeroDivisionError if no inverse exists.
    """
    try:
        return pow(x, -1, m)
    except ValueError:
        raise ZeroDivisionError("invert() no inverse exists")


def legendre(x, y):
    """Legendre symbol (x / y).

    Following the implementation of gmpy2,
    the error is raised only when y is an even number.
    """
    if y <= 0 or not y % 2:
        raise ValueError("y should be an odd prime")
    x %= y
    if not x:
        return 0
    if pow(x, (y - 1) // 2, y) == 1:
        return 1
    return -1


def jacobi(x, y):
    """Jacobi symbol (x / y)."""
    if y <= 0 or not y % 2:
        raise ValueError("y should be an odd positive integer")
    x %= y
    if not x:
        return int(y == 1)
    if y == 1 or x == 1:
        return 1
    if gcd(x, y) != 1:
        return 0
    j = 1
    while x != 0:
        while x % 2 == 0 and x > 0:
            x >>= 1
            if y % 8 in [3, 5]:
                j = -j
        x, y = y, x
        if x % 4 == y % 4 == 3:
            j = -j
        x %= y
    return j


def kronecker(x, y):
    """Kronecker symbol (x / y)."""
    if gcd(x, y) != 1:
        return 0
    if y == 0:
        return 1
    sign = -1 if y < 0 and x < 0 else 1
    y = abs(y)
    s = bit_scan1(y)
    y >>= s
    if s % 2 and x % 8 in [3, 5]:
        sign = -sign
    return sign * jacobi(x, y)


def iroot(y, n):
    if y < 0:
        raise ValueError("y must be nonnegative")
    if n < 1:
        raise ValueError("n must be positive")
    if y in (0, 1):
        return y, True
    if n == 1:
        return y, True
    if n == 2:
        x, rem = mlib.sqrtrem(y)
        return int(x), not rem
    if n >= y.bit_length():
        return 1, False
    # Get initial estimate for Newton's method. Care must be taken to
    # avoid overflow
    try:
        guess = int(y**(1./n) + 0.5)
    except OverflowError:
        exp = math.log2(y)/n
        if exp > 53:
            shift = int(exp - 53)
            guess = int(2.0**(exp - shift) + 1) << shift
        else:
            guess = int(2.0**exp)
    if guess > 2**50:
        # Newton iteration
        xprev, x = -1, guess
        while 1:
            t = x**(n - 1)
            xprev, x = x, ((n - 1)*x + y//t)//n
            if abs(x - xprev) < 2:
                break
    else:
        x = guess
    # Compensate
    t = x**n
    while t < y:
        x += 1
        t = x**n
    while t > y:
        x -= 1
        t = x**n
    return x, t == y


def is_fermat_prp(n, a):
    if a < 2:
        raise ValueError("is_fermat_prp() requires 'a' greater than or equal to 2")
    if n < 1:
        raise ValueError("is_fermat_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    a %= n
    if gcd(n, a) != 1:
        raise ValueError("is_fermat_prp() requires gcd(n,a) == 1")
    return pow(a, n - 1, n) == 1


def is_euler_prp(n, a):
    if a < 2:
        raise ValueError("is_euler_prp() requires 'a' greater than or equal to 2")
    if n < 1:
        raise ValueError("is_euler_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    a %= n
    if gcd(n, a) != 1:
        raise ValueError("is_euler_prp() requires gcd(n,a) == 1")
    return pow(a, n >> 1, n) == jacobi(a, n) % n


def _is_strong_prp(n, a):
    s = bit_scan1(n - 1)
    a = pow(a, n >> s, n)
    if a == 1 or a == n - 1:
        return True
    for _ in range(s - 1):
        a = pow(a, 2, n)
        if a == n - 1:
            return True
        if a == 1:
            return False
    return False


def is_strong_prp(n, a):
    if a < 2:
        raise ValueError("is_strong_prp() requires 'a' greater than or equal to 2")
    if n < 1:
        raise ValueError("is_strong_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    a %= n
    if gcd(n, a) != 1:
        raise ValueError("is_strong_prp() requires gcd(n,a) == 1")
    return _is_strong_prp(n, a)


def _lucas_sequence(n, P, Q, k):
    r"""Return the modular Lucas sequence (U_k, V_k, Q_k).

    Explanation
    ===========

    Given a Lucas sequence defined by P, Q, returns the kth values for
    U and V, along with Q^k, all modulo n. This is intended for use with
    possibly very large values of n and k, where the combinatorial functions
    would be completely unusable.

    .. math ::
        U_k = \begin{cases}
             0 & \text{if } k = 0\\
             1 & \text{if } k = 1\\
             PU_{k-1} - QU_{k-2} & \text{if } k > 1
        \end{cases}\\
        V_k = \begin{cases}
             2 & \text{if } k = 0\\
             P & \text{if } k = 1\\
             PV_{k-1} - QV_{k-2} & \text{if } k > 1
        \end{cases}

    The modular Lucas sequences are used in numerous places in number theory,
    especially in the Lucas compositeness tests and the various n + 1 proofs.

    Parameters
    ==========

    n : int
        n is an odd number greater than or equal to 3
    P : int
    Q : int
        D determined by D = P**2 - 4*Q is non-zero
    k : int
        k is a nonnegative integer

    Returns
    =======

    U, V, Qk : (int, int, int)
        `(U_k \bmod{n}, V_k \bmod{n}, Q^k \bmod{n})`

    Examples
    ========

    >>> from sympy.external.ntheory import _lucas_sequence
    >>> N = 10**2000 + 4561
    >>> sol = U, V, Qk = _lucas_sequence(N, 3, 1, N//2); sol
    (0, 2, 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas_sequence

    """
    if k == 0:
        return (0, 2, 1)
    D = P**2 - 4*Q
    U = 1
    V = P
    Qk = Q % n
    if Q == 1:
        # Optimization for extra strong tests.
        for b in bin(k)[3:]:
            U = (U*V) % n
            V = (V*V - 2) % n
            if b == "1":
                U, V = U*P + V, V*P + U*D
                if U & 1:
                    U += n
                if V & 1:
                    V += n
                U, V = U >> 1, V >> 1
    elif P == 1 and Q == -1:
        # Small optimization for 50% of Selfridge parameters.
        for b in bin(k)[3:]:
            U = (U*V) % n
            if Qk == 1:
                V = (V*V - 2) % n
            else:
                V = (V*V + 2) % n
                Qk = 1
            if b == "1":
                # new_U = (U + V) // 2
                # new_V = (5*U + V) // 2 = 2*U + new_U
                U, V  = U + V, U << 1
                if U & 1:
                    U += n
                U >>= 1
                V += U
                Qk = -1
        Qk %= n
    elif P == 1:
        for b in bin(k)[3:]:
            U = (U*V) % n
            V = (V*V - 2*Qk) % n
            Qk *= Qk
            if b == "1":
                # new_U = (U + V) // 2
                # new_V = new_U - 2*Q*U
                U, V  = U + V, (Q*U) << 1
                if U & 1:
                    U += n
                U >>= 1
                V = U - V
                Qk *= Q
            Qk %= n
    else:
        # The general case with any P and Q.
        for b in bin(k)[3:]:
            U = (U*V) % n
            V = (V*V - 2*Qk) % n
            Qk *= Qk
            if b == "1":
                U, V = U*P + V, V*P + U*D
                if U & 1:
                    U += n
                if V & 1:
                    V += n
                U, V = U >> 1, V >> 1
                Qk *= Q
            Qk %= n
    return (U % n, V % n, Qk)


def is_fibonacci_prp(n, p, q):
    d = p**2 - 4*q
    if d == 0 or p <= 0 or q not in [1, -1]:
        raise ValueError("invalid values for p,q in is_fibonacci_prp()")
    if n < 1:
        raise ValueError("is_fibonacci_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _lucas_sequence(n, p, q, n)[1] == p % n


def is_lucas_prp(n, p, q):
    d = p**2 - 4*q
    if d == 0:
        raise ValueError("invalid values for p,q in is_lucas_prp()")
    if n < 1:
        raise ValueError("is_lucas_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    if gcd(n, q*d) not in [1, n]:
        raise ValueError("is_lucas_prp() requires gcd(n,2*q*D) == 1")
    return _lucas_sequence(n, p, q, n - jacobi(d, n))[0] == 0


def _is_selfridge_prp(n):
    """Lucas compositeness test with the Selfridge parameters for n.

    Explanation
    ===========

    The Lucas compositeness test checks whether n is a prime number.
    The test can be run with arbitrary parameters ``P`` and ``Q``, which also change the performance of the test.
    So, which parameters are most effective for running the Lucas compositeness test?
    As an algorithm for determining ``P`` and ``Q``, Selfridge proposed method A [1]_ page 1401
    (Since two methods were proposed, referred to simply as A and B in the paper,
    we will refer to one of them as "method A").

    method A fixes ``P = 1``. Then, ``D`` defined by ``D = P**2 - 4Q`` is varied from 5, -7, 9, -11, 13, and so on,
    with the first ``D`` being ``jacobi(D, n) == -1``. Once ``D`` is determined,
    ``Q`` is determined to be ``(P**2 - D)//4``.

    References
    ==========

    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf

    """
    for D in range(5, 1_000_000, 2):
        if D & 2: # if D % 4 == 3
            D = -D
        j = jacobi(D, n)
        if j == -1:
            return _lucas_sequence(n, 1, (1-D) // 4, n + 1)[0] == 0
        if j == 0 and D % n:
            return False
        # When j == -1 is hard to find, suspect a square number
        if D == 13 and is_square(n):
            return False
    raise ValueError("appropriate value for D cannot be found in is_selfridge_prp()")


def is_selfridge_prp(n):
    if n < 1:
        raise ValueError("is_selfridge_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_selfridge_prp(n)


def is_strong_lucas_prp(n, p, q):
    D = p**2 - 4*q
    if D == 0:
        raise ValueError("invalid values for p,q in is_strong_lucas_prp()")
    if n < 1:
        raise ValueError("is_selfridge_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    if gcd(n, q*D) not in [1, n]:
        raise ValueError("is_strong_lucas_prp() requires gcd(n,2*q*D) == 1")
    j = jacobi(D, n)
    s = bit_scan1(n - j)
    U, V, Qk = _lucas_sequence(n, p, q, (n - j) >> s)
    if U == 0 or V == 0:
        return True
    for _ in range(s - 1):
        V = (V*V - 2*Qk) % n
        if V == 0:
            return True
        Qk = pow(Qk, 2, n)
    return False


def _is_strong_selfridge_prp(n):
    for D in range(5, 1_000_000, 2):
        if D & 2: # if D % 4 == 3
            D = -D
        j = jacobi(D, n)
        if j == -1:
            s = bit_scan1(n + 1)
            U, V, Qk = _lucas_sequence(n, 1, (1-D) // 4, (n + 1) >> s)
            if U == 0 or V == 0:
                return True
            for _ in range(s - 1):
                V = (V*V - 2*Qk) % n
                if V == 0:
                    return True
                Qk = pow(Qk, 2, n)
            return False
        if j == 0 and D % n:
            return False
        # When j == -1 is hard to find, suspect a square number
        if D == 13 and is_square(n):
            return False
    raise ValueError("appropriate value for D cannot be found in is_strong_selfridge_prp()")


def is_strong_selfridge_prp(n):
    if n < 1:
        raise ValueError("is_strong_selfridge_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_strong_selfridge_prp(n)


def is_bpsw_prp(n):
    if n < 1:
        raise ValueError("is_bpsw_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_strong_prp(n, 2) and _is_selfridge_prp(n)


def is_strong_bpsw_prp(n):
    if n < 1:
        raise ValueError("is_strong_bpsw_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_strong_prp(n, 2) and _is_strong_selfridge_prp(n)
