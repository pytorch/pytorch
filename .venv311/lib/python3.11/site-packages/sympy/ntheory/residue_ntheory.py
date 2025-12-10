from __future__ import annotations

from sympy.external.gmpy import (gcd, lcm, invert, sqrt, jacobi,
                                 bit_scan1, remove)
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence, gf_csolve
from .primetest import isprime
from .generate import primerange
from .factor_ import factorint, _perfect_power
from .modular import crt
from sympy.utilities.decorator import deprecated
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int
from sympy.utilities.iterables import iproduct
from sympy.core.random import _randint, randint

from itertools import product


def n_order(a, n):
    r""" Returns the order of ``a`` modulo ``n``.

    Explanation
    ===========

    The order of ``a`` modulo ``n`` is the smallest integer
    ``k`` such that `a^k` leaves a remainder of 1 with ``n``.

    Parameters
    ==========

    a : integer
    n : integer, n > 1. a and n should be relatively prime

    Returns
    =======

    int : the order of ``a`` modulo ``n``

    Raises
    ======

    ValueError
        If `n \le 1` or `\gcd(a, n) \neq 1`.
        If ``a`` or ``n`` is not an integer.

    Examples
    ========

    >>> from sympy.ntheory import n_order
    >>> n_order(3, 7)
    6
    >>> n_order(4, 7)
    3

    See Also
    ========

    is_primitive_root
        We say that ``a`` is a primitive root of ``n``
        when the order of ``a`` modulo ``n`` equals ``totient(n)``

    """
    a, n = as_int(a), as_int(n)
    if n <= 1:
        raise ValueError("n should be an integer greater than 1")
    a = a % n
    # Trivial
    if a == 1:
        return 1
    if gcd(a, n) != 1:
        raise ValueError("The two numbers should be relatively prime")
    a_order = 1
    for p, e in factorint(n).items():
        pe = p**e
        pe_order = (p - 1) * p**(e - 1)
        factors = factorint(p - 1)
        if e > 1:
            factors[p] = e - 1
        order = 1
        for px, ex in factors.items():
            x = pow(a, pe_order // px**ex, pe)
            while x != 1:
                x = pow(x, px, pe)
                order *= px
        a_order = lcm(a_order, order)
    return int(a_order)


def _primitive_root_prime_iter(p):
    r""" Generates the primitive roots for a prime ``p``.

    Explanation
    ===========

    The primitive roots generated are not necessarily sorted.
    However, the first one is the smallest primitive root.

    Find the element whose order is ``p-1`` from the smaller one.
    If we can find the first primitive root ``g``, we can use the following theorem.

    .. math ::
        \operatorname{ord}(g^k) = \frac{\operatorname{ord}(g)}{\gcd(\operatorname{ord}(g), k)}

    From the assumption that `\operatorname{ord}(g)=p-1`,
    it is a necessary and sufficient condition for
    `\operatorname{ord}(g^k)=p-1` that `\gcd(p-1, k)=1`.

    Parameters
    ==========

    p : odd prime

    Yields
    ======

    int
        the primitive roots of ``p``

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_iter
    >>> sorted(_primitive_root_prime_iter(19))
    [2, 3, 10, 13, 14, 15]

    References
    ==========

    .. [1] W. Stein "Elementary Number Theory" (2011), page 44

    """
    if p == 3:
        yield 2
        return
    # Let p = +-1 (mod 4a). Legendre symbol (a/p) = 1, so `a` is not the primitive root.
    # Corollary : If p = +-1 (mod 8), then 2 is not the primitive root of p.
    g_min = 3 if p % 8 in [1, 7] else 2
    if p < 41:
        # small case
        g = 5 if p == 23 else g_min
    else:
        v = [(p - 1) // i for i in factorint(p - 1).keys()]
        for g in range(g_min, p):
            if all(pow(g, pw, p) != 1 for pw in v):
                break
    yield g
    # g**k is the primitive root of p iff gcd(p - 1, k) = 1
    for k in range(3, p, 2):
        if gcd(p - 1, k) == 1:
            yield pow(g, k, p)


def _primitive_root_prime_power_iter(p, e):
    r""" Generates the primitive roots of `p^e`.

    Explanation
    ===========

    Let ``g`` be the primitive root of ``p``.
    If `g^{p-1} \not\equiv 1 \pmod{p^2}`, then ``g`` is primitive root of `p^e`.
    Thus, if we find a primitive root ``g`` of ``p``,
    then `g, g+p, g+2p, \ldots, g+(p-1)p` are primitive roots of `p^2` except one.
    That one satisfies `\hat{g}^{p-1} \equiv 1 \pmod{p^2}`.
    If ``h`` is the primitive root of `p^2`,
    then `h, h+p^2, h+2p^2, \ldots, h+(p^{e-2}-1)p^e` are primitive roots of `p^e`.

    Parameters
    ==========

    p : odd prime
    e : positive integer

    Yields
    ======

    int
        the primitive roots of `p^e`

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power_iter
    >>> sorted(_primitive_root_prime_power_iter(5, 2))
    [2, 3, 8, 12, 13, 17, 22, 23]

    """
    if e == 1:
        yield from _primitive_root_prime_iter(p)
    else:
        p2 = p**2
        for g in _primitive_root_prime_iter(p):
            t = (g - pow(g, 2 - p, p2)) % p2
            for k in range(0, p2, p):
                if k != t:
                    yield from (g + k + m for m in range(0, p**e, p2))


def _primitive_root_prime_power2_iter(p, e):
    r""" Generates the primitive roots of `2p^e`.

    Explanation
    ===========

    If ``g`` is the primitive root of ``p**e``,
    then the odd one of ``g`` and ``g+p**e`` is the primitive root of ``2*p**e``.

    Parameters
    ==========

    p : odd prime
    e : positive integer

    Yields
    ======

    int
        the primitive roots of `2p^e`

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power2_iter
    >>> sorted(_primitive_root_prime_power2_iter(5, 2))
    [3, 13, 17, 23, 27, 33, 37, 47]

    """
    for g in _primitive_root_prime_power_iter(p, e):
        if g % 2 == 1:
            yield g
        else:
            yield g + p**e


def primitive_root(p, smallest=True):
    r""" Returns a primitive root of ``p`` or None.

    Explanation
    ===========

    For the definition of primitive root,
    see the explanation of ``is_primitive_root``.

    The primitive root of ``p`` exist only for
    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).
    Now, if we know the primitive root of ``q``,
    we can calculate the primitive root of `q^e`,
    and if we know the primitive root of `q^e`,
    we can calculate the primitive root of `2q^e`.
    When there is no need to find the smallest primitive root,
    this property can be used to obtain a fast primitive root.
    On the other hand, when we want the smallest primitive root,
    we naively determine whether it is a primitive root or not.

    Parameters
    ==========

    p : integer, p > 1
    smallest : if True the smallest primitive root is returned or None

    Returns
    =======

    int | None :
        If the primitive root exists, return the primitive root of ``p``.
        If not, return None.

    Raises
    ======

    ValueError
        If `p \le 1` or ``p`` is not an integer.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import primitive_root
    >>> primitive_root(19)
    2
    >>> primitive_root(21) is None
    True
    >>> primitive_root(50, smallest=False)
    27

    See Also
    ========

    is_primitive_root

    References
    ==========

    .. [1] W. Stein "Elementary Number Theory" (2011), page 44
    .. [2] P. Hackman "Elementary Number Theory" (2009), Chapter C

    """
    p = as_int(p)
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")
    if p <= 4:
        return p - 1
    p_even = p % 2 == 0
    if not p_even:
        q = p  # p is odd
    elif p % 4:
        q = p//2  # p had 1 factor of 2
    else:
        return None  # p had more than one factor of 2
    if isprime(q):
        e = 1
    else:
        m = _perfect_power(q, 3)
        if not m:
            return None
        q, e = m
        if not isprime(q):
            return None
    if not smallest:
        if p_even:
            return next(_primitive_root_prime_power2_iter(q, e))
        return next(_primitive_root_prime_power_iter(q, e))
    if p_even:
        for i in range(3, p, 2):
            if i % q and is_primitive_root(i, p):
                return i
    g = next(_primitive_root_prime_iter(q))
    if e == 1 or pow(g, q - 1, q**2) != 1:
        return g
    for i in range(g + 1, p):
        if i % q and is_primitive_root(i, p):
            return i


def is_primitive_root(a, p):
    r""" Returns True if ``a`` is a primitive root of ``p``.

    Explanation
    ===========

    ``a`` is said to be the primitive root of ``p`` if `\gcd(a, p) = 1` and
    `\phi(p)` is the smallest positive number s.t.

        `a^{\phi(p)} \equiv 1 \pmod{p}`.

    where `\phi(p)` is Euler's totient function.

    The primitive root of ``p`` exist only for
    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).
    Hence, if it is not such a ``p``, it returns False.
    To determine the primitive root, we need to know
    the prime factorization of ``q-1``.
    The hardness of the determination depends on this complexity.

    Parameters
    ==========

    a : integer
    p : integer, ``p`` > 1. ``a`` and ``p`` should be relatively prime

    Returns
    =======

    bool : If True, ``a`` is the primitive root of ``p``.

    Raises
    ======

    ValueError
        If `p \le 1` or `\gcd(a, p) \neq 1`.
        If ``a`` or ``p`` is not an integer.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> from sympy.ntheory import is_primitive_root, n_order
    >>> is_primitive_root(3, 10)
    True
    >>> is_primitive_root(9, 10)
    False
    >>> n_order(3, 10) == totient(10)
    True
    >>> n_order(9, 10) == totient(10)
    False

    See Also
    ========

    primitive_root

    """
    a, p = as_int(a), as_int(p)
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")
    a = a % p
    if gcd(a, p) != 1:
        raise ValueError("The two numbers should be relatively prime")
    # Primitive root of p exist only for
    # p = 2, 4, q**e, 2*q**e (q is odd prime)
    if p <= 4:
        # The primitive root is only p-1.
        return a == p - 1
    if p % 2:
        q = p  # p is odd
    elif p % 4:
        q = p//2  # p had 1 factor of 2
    else:
        return False  # p had more than one factor of 2
    if isprime(q):
        group_order = q - 1
        factors = factorint(q - 1).keys()
    else:
        m = _perfect_power(q, 3)
        if not m:
            return False
        q, e = m
        if not isprime(q):
            return False
        group_order = q**(e - 1)*(q - 1)
        factors = set(factorint(q - 1).keys())
        factors.add(q)
    return all(pow(a, group_order // prime, p) != 1 for prime in factors)


def _sqrt_mod_tonelli_shanks(a, p):
    """
    Returns the square root in the case of ``p`` prime with ``p == 1 (mod 8)``

    Assume that the root exists.

    Parameters
    ==========

    a : int
    p : int
        prime number. should be ``p % 8 == 1``

    Returns
    =======

    int : Generally, there are two roots, but only one is returned.
          Which one is returned is random.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_tonelli_shanks
    >>> _sqrt_mod_tonelli_shanks(2, 17) in [6, 11]
    True

    References
    ==========

    .. [1] Carl Pomerance, Richard Crandall, Prime Numbers: A Computational Perspective,
           2nd Edition (2005), page 101, ISBN:978-0387252827

    """
    s = bit_scan1(p - 1)
    t = p >> s
    # find a non-quadratic residue
    if p % 12 == 5:
        # Legendre symbol (3/p) == -1 if p % 12 in [5, 7]
        d = 3
    elif p % 5 in [2, 3]:
        # Legendre symbol (5/p) == -1 if p % 5 in [2, 3]
        d = 5
    else:
        while 1:
            d = randint(6, p - 1)
            if jacobi(d, p) == -1:
                break
    #assert legendre_symbol(d, p) == -1
    A = pow(a, t, p)
    D = pow(d, t, p)
    m = 0
    for i in range(s):
        adm = A*pow(D, m, p) % p
        adm = pow(adm, 2**(s - 1 - i), p)
        if adm % p == p - 1:
            m += 2**i
    #assert A*pow(D, m, p) % p == 1
    x = pow(a, (t + 1)//2, p)*pow(D, m//2, p) % p
    return x


def sqrt_mod(a, p, all_roots=False):
    """
    Find a root of ``x**2 = a mod p``.

    Parameters
    ==========

    a : integer
    p : positive integer
    all_roots : if True the list of roots is returned or None

    Notes
    =====

    If there is no root it is returned None; else the returned root
    is less or equal to ``p // 2``; in general is not the smallest one.
    It is returned ``p // 2`` only if it is the only root.

    Use ``all_roots`` only when it is expected that all the roots fit
    in memory; otherwise use ``sqrt_mod_iter``.

    Examples
    ========

    >>> from sympy.ntheory import sqrt_mod
    >>> sqrt_mod(11, 43)
    21
    >>> sqrt_mod(17, 32, True)
    [7, 9, 23, 25]
    """
    if all_roots:
        return sorted(sqrt_mod_iter(a, p))
    p = abs(as_int(p))
    halfp = p // 2
    x = None
    for r in sqrt_mod_iter(a, p):
        if r < halfp:
            return r
        elif r > halfp:
            return p - r
        else:
            x = r
    return x


def sqrt_mod_iter(a, p, domain=int):
    """
    Iterate over solutions to ``x**2 = a mod p``.

    Parameters
    ==========

    a : integer
    p : positive integer
    domain : integer domain, ``int``, ``ZZ`` or ``Integer``

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import sqrt_mod_iter
    >>> list(sqrt_mod_iter(11, 43))
    [21, 22]

    See Also
    ========

    sqrt_mod : Same functionality, but you want a sorted list or only one solution.

    """
    a, p = as_int(a), abs(as_int(p))
    v = []
    pv = []
    _product = product
    for px, ex in factorint(p).items():
        if a % px:
            # `len(rx)` is at most 4
            rx = _sqrt_mod_prime_power(a, px, ex)
        else:
            # `len(list(rx))` can be assumed to be large.
            # The `itertools.product` is disadvantageous in terms of memory usage.
            # It is also inferior to iproduct in speed if not all Cartesian products are needed.
            rx = _sqrt_mod1(a, px, ex)
            _product = iproduct
        if not rx:
            return
        v.append(rx)
        pv.append(px**ex)
    if len(v) == 1:
        yield from map(domain, v[0])
    else:
        mm, e, s = gf_crt1(pv, ZZ)
        for vx in _product(*v):
            yield domain(gf_crt2(vx, pv, mm, e, s, ZZ))


def _sqrt_mod_prime_power(a, p, k):
    """
    Find the solutions to ``x**2 = a mod p**k`` when ``a % p != 0``.
    If no solution exists, return ``None``.
    Solutions are returned in an ascending list.

    Parameters
    ==========

    a : integer
    p : prime number
    k : positive integer

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
    >>> _sqrt_mod_prime_power(11, 43, 1)
    [21, 22]

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 160
    .. [2] http://www.numbertheory.org/php/squareroot.html
    .. [3] [Gathen99]_
    """
    pk = p**k
    a = a % pk

    if p == 2:
        # see Ref.[2]
        if a % 8 != 1:
            return None
        # Trivial
        if k <= 3:
            return list(range(1, pk, 2))
        r = 1
        # r is one of the solutions to x**2 - a = 0 (mod 2**3).
        # Hensel lift them to solutions of x**2 - a = 0 (mod 2**k)
        # if r**2 - a = 0 mod 2**nx but not mod 2**(nx+1)
        # then r + 2**(nx - 1) is a root mod 2**(nx+1)
        for nx in range(3, k):
            if ((r**2 - a) >> nx) % 2:
                r += 1 << (nx - 1)
        # r is a solution of x**2 - a = 0 (mod 2**k), and
        # there exist other solutions -r, r+h, -(r+h), and these are all solutions.
        h = 1 << (k - 1)
        return sorted([r, pk - r, (r + h) % pk, -(r + h) % pk])

    # If the Legendre symbol (a/p) is not 1, no solution exists.
    if jacobi(a, p) != 1:
        return None
    if p % 4 == 3:
        res = pow(a, (p + 1) // 4, p)
    elif p % 8 == 5:
        res = pow(a, (p + 3) // 8, p)
        if pow(res, 2, p) != a % p:
            res = res * pow(2, (p - 1) // 4, p) % p
    else:
        res = _sqrt_mod_tonelli_shanks(a, p)
    if k > 1:
        # Hensel lifting with Newton iteration, see Ref.[3] chapter 9
        # with f(x) = x**2 - a; one has f'(a) != 0 (mod p) for p != 2
        px = p
        for _ in range(k.bit_length() - 1):
            px = px**2
            frinv = invert(2*res, px)
            res = (res - (res**2 - a)*frinv) % px
        if k & (k - 1): # If k is not a power of 2
            frinv = invert(2*res, pk)
            res = (res - (res**2 - a)*frinv) % pk
    return sorted([res, pk - res])


def _sqrt_mod1(a, p, n):
    """
    Find solution to ``x**2 == a mod p**n`` when ``a % p == 0``.
    If no solution exists, return ``None``.

    Parameters
    ==========

    a : integer
    p : prime number, p must divide a
    n : positive integer

    References
    ==========

    .. [1] http://www.numbertheory.org/php/squareroot.html
    """
    pn = p**n
    a = a % pn
    if a == 0:
        # case gcd(a, p**k) = p**n
        return range(0, pn, p**((n + 1) // 2))
    # case gcd(a, p**k) = p**r, r < n
    a, r = remove(a, p)
    if r % 2 == 1:
        return None
    res = _sqrt_mod_prime_power(a, p, n - r)
    if res is None:
        return None
    m = r // 2
    return (x for rx in res for x in range(rx*p**m, pn, p**(n - m)))


def is_quad_residue(a, p):
    """
    Returns True if ``a`` (mod ``p``) is in the set of squares mod ``p``,
    i.e a % p in set([i**2 % p for i in range(p)]).

    Parameters
    ==========

    a : integer
    p : positive integer

    Returns
    =======

    bool : If True, ``x**2 == a (mod p)`` has solution.

    Raises
    ======

    ValueError
        If ``a``, ``p`` is not integer.
        If ``p`` is not positive.

    Examples
    ========

    >>> from sympy.ntheory import is_quad_residue
    >>> is_quad_residue(21, 100)
    True

    Indeed, ``pow(39, 2, 100)`` would be 21.

    >>> is_quad_residue(21, 120)
    False

    That is, for any integer ``x``, ``pow(x, 2, 120)`` is not 21.

    If ``p`` is an odd
    prime, an iterative method is used to make the determination:

    >>> from sympy.ntheory import is_quad_residue
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]
    >>> [j for j in range(7) if is_quad_residue(j, 7)]
    [0, 1, 2, 4]

    See Also
    ========

    legendre_symbol, jacobi_symbol, sqrt_mod
    """
    a, p = as_int(a), as_int(p)
    if p < 1:
        raise ValueError('p must be > 0')
    a %= p
    if a < 2 or p < 3:
        return True
    # Since we want to compute the Jacobi symbol,
    # we separate p into the odd part and the rest.
    t = bit_scan1(p)
    if t:
        # The existence of a solution to a power of 2 is determined
        # using the logic of `p==2` in `_sqrt_mod_prime_power` and `_sqrt_mod1`.
        a_ = a % (1 << t)
        if a_:
            r = bit_scan1(a_)
            if r % 2 or (a_ >> r) & 6:
                return False
        p >>= t
        a %= p
        if a < 2 or p < 3:
            return True
    # If Jacobi symbol is -1 or p is prime, can be determined by Jacobi symbol only
    j = jacobi(a, p)
    if j == -1 or isprime(p):
        return j == 1
    # Checks if `x**2 = a (mod p)` has a solution
    for px, ex in factorint(p).items():
        if a % px:
            if jacobi(a, px) != 1:
                return False
        else:
            a_ = a % px**ex
            if a_ == 0:
                continue
            a_, r = remove(a_, px)
            if r % 2 or jacobi(a_, px) != 1:
                return False
    return True


def is_nthpow_residue(a, n, m):
    """
    Returns True if ``x**n == a (mod m)`` has solutions.

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76

    """
    a = a % m
    a, n, m = as_int(a), as_int(n), as_int(m)
    if m <= 0:
        raise ValueError('m must be > 0')
    if n < 0:
        raise ValueError('n must be >= 0')
    if n == 0:
        if m == 1:
            return False
        return a == 1
    if a == 0:
        return True
    if n == 1:
        return True
    if n == 2:
        return is_quad_residue(a, m)
    return all(_is_nthpow_residue_bign_prime_power(a, n, p, e)
               for p, e in factorint(m).items())


def _is_nthpow_residue_bign_prime_power(a, n, p, k):
    r"""
    Returns True if `x^n = a \pmod{p^k}` has solutions for `n > 2`.

    Parameters
    ==========

    a : positive integer
    n : integer, n > 2
    p : prime number
    k : positive integer

    """
    while a % p == 0:
        a %= pow(p, k)
        if not a:
            return True
        a, mu = remove(a, p)
        if mu % n:
            return False
        k -= mu
    if p != 2:
        f = p**(k - 1)*(p - 1) # f = totient(p**k)
        return pow(a, f // gcd(f, n), pow(p, k)) == 1
    if n & 1:
        return True
    c = min(bit_scan1(n) + 2, k)
    return a % pow(2, c) == 1


def _nthroot_mod1(s, q, p, all_roots):
    """
    Root of ``x**q = s mod p``, ``p`` prime and ``q`` divides ``p - 1``.
    Assume that the root exists.

    Parameters
    ==========

    s : integer
    q : integer, n > 2. ``q`` divides ``p - 1``.
    p : prime number
    all_roots : if False returns the smallest root, else the list of roots

    Returns
    =======

    list[int] | int :
        Root of ``x**q = s mod p``. If ``all_roots == True``,
        returned ascending list. otherwise, returned an int.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _nthroot_mod1
    >>> _nthroot_mod1(5, 3, 13, False)
    7
    >>> _nthroot_mod1(13, 4, 17, True)
    [3, 5, 12, 14]

    References
    ==========

    .. [1] A. M. Johnston, A Generalized qth Root Algorithm,
           ACM-SIAM Symposium on Discrete Algorithms (1999), pp. 929-930

    """
    g = next(_primitive_root_prime_iter(p))
    r = s
    for qx, ex in factorint(q).items():
        f = (p - 1) // qx**ex
        while f % qx == 0:
            f //= qx
        z = f*invert(-f, qx)
        x = (1 + z) // qx
        t = discrete_log(p, pow(r, f, p), pow(g, f*qx, p))
        for _ in range(ex):
            # assert t == discrete_log(p, pow(r, f, p), pow(g, f*qx, p))
            r = pow(r, x, p)*pow(g, -z*t % (p - 1), p) % p
            t //= qx
    res = [r]
    h = pow(g, (p - 1) // q, p)
    #assert pow(h, q, p) == 1
    hx = r
    for _ in range(q - 1):
        hx = (hx*h) % p
        res.append(hx)
    if all_roots:
        res.sort()
        return res
    return min(res)


def _nthroot_mod_prime_power(a, n, p, k):
    """ Root of ``x**n = a mod p**k``.

    Parameters
    ==========

    a : integer
    n : integer, n > 2
    p : prime number
    k : positive integer

    Returns
    =======

    list[int] :
        Ascending list of roots of ``x**n = a mod p**k``.
        If no solution exists, return ``[]``.

    """
    if not _is_nthpow_residue_bign_prime_power(a, n, p, k):
        return []
    a_mod_p = a % p
    if a_mod_p == 0:
        base_roots = [0]
    elif (p - 1) % n == 0:
        base_roots = _nthroot_mod1(a_mod_p, n, p, all_roots=True)
    else:
        # The roots of ``x**n - a = 0 (mod p)`` are roots of
        # ``gcd(x**n - a, x**(p - 1) - 1) = 0 (mod p)``
        pa = n
        pb = p - 1
        b = 1
        if pa < pb:
            a_mod_p, pa, b, pb = b, pb, a_mod_p, pa
        # gcd(x**pa - a, x**pb - b) = gcd(x**pb - b, x**pc - c)
        # where pc = pa % pb; c = b**-q * a mod p
        while pb:
            q, pc = divmod(pa, pb)
            c = pow(b, -q, p) * a_mod_p % p
            pa, pb = pb, pc
            a_mod_p, b = b, c
        if pa == 1:
            base_roots = [a_mod_p]
        elif pa == 2:
            base_roots = sqrt_mod(a_mod_p, p, all_roots=True)
        else:
            base_roots = _nthroot_mod1(a_mod_p, pa, p, all_roots=True)
    if k == 1:
        return base_roots
    a %= p**k
    tot_roots = set()
    for root in base_roots:
        diff = pow(root, n - 1, p)*n % p
        new_base = p
        if diff != 0:
            m_inv = invert(diff, p)
            for _ in range(k - 1):
                new_base *= p
                tmp = pow(root, n, new_base) - a
                tmp *= m_inv
                root = (root - tmp) % new_base
            tot_roots.add(root)
        else:
            roots_in_base = {root}
            for _ in range(k - 1):
                new_base *= p
                new_roots = set()
                for k_ in roots_in_base:
                    if pow(k_, n, new_base) != a % new_base:
                        continue
                    while k_ not in new_roots:
                        new_roots.add(k_)
                        k_ = (k_ + (new_base // p)) % new_base
                roots_in_base = new_roots
            tot_roots = tot_roots | roots_in_base
    return sorted(tot_roots)


def nthroot_mod(a, n, p, all_roots=False):
    """
    Find the solutions to ``x**n = a mod p``.

    Parameters
    ==========

    a : integer
    n : positive integer
    p : positive integer
    all_roots : if False returns the smallest root, else the list of roots

    Returns
    =======

        list[int] | int | None :
            solutions to ``x**n = a mod p``.
            The table of the output type is:

            ========== ========== ==========
            all_roots  has roots  Returns
            ========== ========== ==========
            True       Yes        list[int]
            True       No         []
            False      Yes        int
            False      No         None
            ========== ========== ==========

    Raises
    ======

        ValueError
            If ``a``, ``n`` or ``p`` is not integer.
            If ``n`` or ``p`` is not positive.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import nthroot_mod
    >>> nthroot_mod(11, 4, 19)
    8
    >>> nthroot_mod(11, 4, 19, True)
    [8, 11]
    >>> nthroot_mod(68, 3, 109)
    23

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76

    """
    a = a % p
    a, n, p = as_int(a), as_int(n), as_int(p)

    if n < 1:
        raise ValueError("n should be positive")
    if p < 1:
        raise ValueError("p should be positive")
    if n == 1:
        return [a] if all_roots else a
    if n == 2:
        return sqrt_mod(a, p, all_roots)
    base = []
    prime_power = []
    for q, e in factorint(p).items():
        tot_roots = _nthroot_mod_prime_power(a, n, q, e)
        if not tot_roots:
            return [] if all_roots else None
        prime_power.append(q**e)
        base.append(sorted(tot_roots))
    P, E, S = gf_crt1(prime_power, ZZ)
    ret = sorted(map(int, {gf_crt2(c, prime_power, P, E, S, ZZ)
                           for c in product(*base)}))
    if all_roots:
        return ret
    if ret:
        return ret[0]


def quadratic_residues(p) -> list[int]:
    """
    Returns the list of quadratic residues.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import quadratic_residues
    >>> quadratic_residues(7)
    [0, 1, 2, 4]
    """
    p = as_int(p)
    r = {pow(i, 2, p) for i in range(p // 2 + 1)}
    return sorted(r)


@deprecated("""\
The `sympy.ntheory.residue_ntheory.legendre_symbol` has been moved to `sympy.functions.combinatorial.numbers.legendre_symbol`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def legendre_symbol(a, p):
    r"""
    Returns the Legendre symbol `(a / p)`.

    .. deprecated:: 1.13

        The ``legendre_symbol`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.legendre_symbol`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    For an integer ``a`` and an odd prime ``p``, the Legendre symbol is
    defined as

    .. math ::
        \genfrac(){}{}{a}{p} = \begin{cases}
             0 & \text{if } p \text{ divides } a\\
             1 & \text{if } a \text{ is a quadratic residue modulo } p\\
            -1 & \text{if } a \text{ is a quadratic nonresidue modulo } p
        \end{cases}

    Parameters
    ==========

    a : integer
    p : odd prime

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import legendre_symbol
    >>> [legendre_symbol(i, 7) for i in range(7)]
    [0, 1, 1, -1, 1, -1, -1]
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]

    See Also
    ========

    is_quad_residue, jacobi_symbol

    """
    from sympy.functions.combinatorial.numbers import legendre_symbol as _legendre_symbol
    return _legendre_symbol(a, p)


@deprecated("""\
The `sympy.ntheory.residue_ntheory.jacobi_symbol` has been moved to `sympy.functions.combinatorial.numbers.jacobi_symbol`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def jacobi_symbol(m, n):
    r"""
    Returns the Jacobi symbol `(m / n)`.

    .. deprecated:: 1.13

        The ``jacobi_symbol`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.jacobi_symbol`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    For any integer ``m`` and any positive odd integer ``n`` the Jacobi symbol
    is defined as the product of the Legendre symbols corresponding to the
    prime factors of ``n``:

    .. math ::
        \genfrac(){}{}{m}{n} =
            \genfrac(){}{}{m}{p^{1}}^{\alpha_1}
            \genfrac(){}{}{m}{p^{2}}^{\alpha_2}
            ...
            \genfrac(){}{}{m}{p^{k}}^{\alpha_k}
            \text{ where } n =
                p_1^{\alpha_1}
                p_2^{\alpha_2}
                ...
                p_k^{\alpha_k}

    Like the Legendre symbol, if the Jacobi symbol `\genfrac(){}{}{m}{n} = -1`
    then ``m`` is a quadratic nonresidue modulo ``n``.

    But, unlike the Legendre symbol, if the Jacobi symbol
    `\genfrac(){}{}{m}{n} = 1` then ``m`` may or may not be a quadratic residue
    modulo ``n``.

    Parameters
    ==========

    m : integer
    n : odd positive integer

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import jacobi_symbol, legendre_symbol
    >>> from sympy import S
    >>> jacobi_symbol(45, 77)
    -1
    >>> jacobi_symbol(60, 121)
    1

    The relationship between the ``jacobi_symbol`` and ``legendre_symbol`` can
    be demonstrated as follows:

    >>> L = legendre_symbol
    >>> S(45).factors()
    {3: 2, 5: 1}
    >>> jacobi_symbol(7, 45) == L(7, 3)**2 * L(7, 5)**1
    True

    See Also
    ========

    is_quad_residue, legendre_symbol
    """
    from sympy.functions.combinatorial.numbers import jacobi_symbol as _jacobi_symbol
    return _jacobi_symbol(m, n)


@deprecated("""\
The `sympy.ntheory.residue_ntheory.mobius` has been moved to `sympy.functions.combinatorial.numbers.mobius`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def mobius(n):
    """
    Mobius function maps natural number to {-1, 0, 1}

    .. deprecated:: 1.13

        The ``mobius`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.mobius`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    It is defined as follows:
        1) `1` if `n = 1`.
        2) `0` if `n` has a squared prime factor.
        3) `(-1)^k` if `n` is a square-free positive integer with `k`
           number of prime factors.

    It is an important multiplicative function in number theory
    and combinatorics.  It has applications in mathematical series,
    algebraic number theory and also physics (Fermion operator has very
    concrete realization with Mobius Function model).

    Parameters
    ==========

    n : positive integer

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import mobius
    >>> mobius(13*7)
    1
    >>> mobius(1)
    1
    >>> mobius(13*7*5)
    -1
    >>> mobius(13**2)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_function
    .. [2] Thomas Koshy "Elementary Number Theory with Applications"

    """
    from sympy.functions.combinatorial.numbers import mobius as _mobius
    return _mobius(n)


def _discrete_log_trial_mul(n, a, b, order=None):
    """
    Trial multiplication algorithm for computing the discrete logarithm of
    ``a`` to the base ``b`` modulo ``n``.

    The algorithm finds the discrete logarithm using exhaustive search. This
    naive method is used as fallback algorithm of ``discrete_log`` when the
    group order is very small. The value ``n`` must be greater than 1.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_trial_mul
    >>> _discrete_log_trial_mul(41, 15, 7)
    3

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    a %= n
    b %= n
    if order is None:
        order = n
    x = 1
    for i in range(order):
        if x == a:
            return i
        x = x * b % n
    raise ValueError("Log does not exist")


def _discrete_log_shanks_steps(n, a, b, order=None):
    """
    Baby-step giant-step algorithm for computing the discrete logarithm of
    ``a`` to the base ``b`` modulo ``n``.

    The algorithm is a time-memory trade-off of the method of exhaustive
    search. It uses `O(sqrt(m))` memory, where `m` is the group order.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_shanks_steps
    >>> _discrete_log_shanks_steps(41, 15, 7)
    3

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    a %= n
    b %= n
    if order is None:
        order = n_order(b, n)
    m = sqrt(order) + 1
    T = {}
    x = 1
    for i in range(m):
        T[x] = i
        x = x * b % n
    z = pow(b, -m, n)
    x = a
    for i in range(m):
        if x in T:
            return i * m + T[x]
        x = x * z % n
    raise ValueError("Log does not exist")


def _discrete_log_pollard_rho(n, a, b, order=None, retries=10, rseed=None):
    """
    Pollard's Rho algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    It is a randomized algorithm with the same expected running time as
    ``_discrete_log_shanks_steps``, but requires a negligible amount of memory.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_pollard_rho
    >>> _discrete_log_pollard_rho(227, 3**7, 3)
    7

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    a %= n
    b %= n

    if order is None:
        order = n_order(b, n)
    randint = _randint(rseed)

    for i in range(retries):
        aa = randint(1, order - 1)
        ba = randint(1, order - 1)
        xa = pow(b, aa, n) * pow(a, ba, n) % n

        c = xa % 3
        if c == 0:
            xb = a * xa % n
            ab = aa
            bb = (ba + 1) % order
        elif c == 1:
            xb = xa * xa % n
            ab = (aa + aa) % order
            bb = (ba + ba) % order
        else:
            xb = b * xa % n
            ab = (aa + 1) % order
            bb = ba

        for j in range(order):
            c = xa % 3
            if c == 0:
                xa = a * xa % n
                ba = (ba + 1) % order
            elif c == 1:
                xa = xa * xa % n
                aa = (aa + aa) % order
                ba = (ba + ba) % order
            else:
                xa = b * xa % n
                aa = (aa + 1) % order

            c = xb % 3
            if c == 0:
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                xb = b * xb % n
                ab = (ab + 1) % order

            c = xb % 3
            if c == 0:
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                xb = b * xb % n
                ab = (ab + 1) % order

            if xa == xb:
                r = (ba - bb) % order
                try:
                    e = invert(r, order) * (ab - aa) % order
                    if (pow(b, e, n) - a) % n == 0:
                        return e
                except ZeroDivisionError:
                    pass
                break
    raise ValueError("Pollard's Rho failed to find logarithm")


def _discrete_log_is_smooth(n: int, factorbase: list):
    """Try to factor n with respect to a given factorbase.
    Upon success a list of exponents with respect to the factorbase is returned.
    Otherwise None."""
    factors = [0]*len(factorbase)
    for i, p in enumerate(factorbase):
        while n % p == 0: # divide by p as many times as possible
            factors[i] += 1
            n = n // p
    if n != 1:
        return None # the number factors if at the end nothing is left
    return factors


def _discrete_log_index_calculus(n, a, b, order, rseed=None):
    """
    Index Calculus algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    The group order must be given and prime. It is not suitable for small orders
    and the algorithm might fail to find a solution in such situations.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_index_calculus
    >>> _discrete_log_index_calculus(24570203447, 23859756228, 2, 12285101723)
    4519867240

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    randint = _randint(rseed)
    from math import sqrt, exp, log
    a %= n
    b %= n
    # assert isprime(order), "The order of the base must be prime."
    # First choose a heuristic the bound B for the factorbase.
    # We have added an extra term to the asymptotic value which
    # is closer to the theoretical optimum for n up to 2^70.
    B = int(exp(0.5 * sqrt( log(n) * log(log(n)) )*( 1 + 1/log(log(n)) )))
    max = 5 * B * B  # expected number of tries to find a relation
    factorbase = list(primerange(B)) # compute the factorbase
    lf = len(factorbase) # length of the factorbase
    ordermo = order-1
    abx = a
    for x in range(order):
        if abx == 1:
            return (order - x) % order
        relationa = _discrete_log_is_smooth(abx, factorbase)
        if relationa:
            relationa = [r % order for r in relationa] + [x]
            break
        abx = abx * b % n # abx = a*pow(b, x, n) % n

    else:
        raise ValueError("Index Calculus failed")

    relations = [None] * lf
    k = 1  # number of relations found
    kk = 0
    while k < 3 * lf and kk < max:  # find relations for all primes in our factor base
        x = randint(1,ordermo)
        relation = _discrete_log_is_smooth(pow(b,x,n), factorbase)
        if relation is None:
            kk += 1
            continue
        k += 1
        kk = 0
        relation += [ x ]
        index = lf  # determine the index of the first nonzero entry
        for i in range(lf):
            ri = relation[i] % order
            if ri> 0 and relations[i] is not None:  # make this entry zero if we can
                for j in range(lf+1):
                    relation[j] = (relation[j] - ri*relations[i][j]) % order
            else:
                relation[i] = ri
            if relation[i] > 0 and index == lf:  # is this the index of the first nonzero entry?
                index = i
        if index == lf or relations[index] is not None:  # the relation contains no new information
            continue
        # the relation contains new information
        rinv = pow(relation[index],-1,order)  # normalize the first nonzero entry
        for j in range(index,lf+1):
            relation[j] = rinv * relation[j] % order
        relations[index] = relation
        for i in range(lf):  # subtract the new relation from the one for a
            if relationa[i] > 0 and relations[i] is not None:
                rbi = relationa[i]
                for j in range(lf+1):
                    relationa[j] = (relationa[j] - rbi*relations[i][j]) % order
            if relationa[i] > 0:  # the index of the first nonzero entry
                break  # we do not need to reduce further at this point
        else:  # all unknowns are gone
            #print(f"Success after {k} relations out of {lf}")
            x = (order -relationa[lf]) % order
            if pow(b,x,n) == a:
                return x
            raise ValueError("Index Calculus failed")
    raise ValueError("Index Calculus failed")


def _discrete_log_pohlig_hellman(n, a, b, order=None, order_factors=None):
    """
    Pohlig-Hellman algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    In order to compute the discrete logarithm, the algorithm takes advantage
    of the factorization of the group order. It is more efficient when the
    group order factors into many small primes.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_pohlig_hellman
    >>> _discrete_log_pohlig_hellman(251, 210, 71)
    197

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    from .modular import crt
    a %= n
    b %= n

    if order is None:
        order = n_order(b, n)
    if order_factors is None:
        order_factors = factorint(order)
    l = [0] * len(order_factors)

    for i, (pi, ri) in enumerate(order_factors.items()):
        for j in range(ri):
            aj = pow(a * pow(b, -l[i], n), order // pi**(j + 1), n)
            bj = pow(b, order // pi, n)
            cj = discrete_log(n, aj, bj, pi, True)
            l[i] += cj * pi**j

    d, _ = crt([pi**ri for pi, ri in order_factors.items()], l)
    return d


def discrete_log(n, a, b, order=None, prime_order=None):
    """
    Compute the discrete logarithm of ``a`` to the base ``b`` modulo ``n``.

    This is a recursive function to reduce the discrete logarithm problem in
    cyclic groups of composite order to the problem in cyclic groups of prime
    order.

    It employs different algorithms depending on the problem (subgroup order
    size, prime order or not):

        * Trial multiplication
        * Baby-step giant-step
        * Pollard's Rho
        * Index Calculus
        * Pohlig-Hellman

    Examples
    ========

    >>> from sympy.ntheory import discrete_log
    >>> discrete_log(41, 15, 7)
    3

    References
    ==========

    .. [1] https://mathworld.wolfram.com/DiscreteLogarithm.html
    .. [2] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).

    """
    from math import sqrt, log
    n, a, b = as_int(n), as_int(a), as_int(b)

    if n < 1:
        raise ValueError("n should be positive")
    if n == 1:
        return 0

    if order is None:
        # Compute the order and its factoring in one pass
        # order = totient(n), factors = factorint(order)
        factors = {}
        for px, kx in factorint(n).items():
            if kx > 1:
                if px in factors:
                    factors[px] += kx - 1
                else:
                    factors[px] = kx - 1
            for py, ky in factorint(px - 1).items():
                if py in factors:
                    factors[py] += ky
                else:
                    factors[py] = ky
        order = 1
        for px, kx in factors.items():
            order *= px**kx
        # Now the `order` is the order of the group and factors = factorint(order)
        # The order of `b` divides the order of the group.
        order_factors = {}
        for p, e in factors.items():
            i = 0
            for _ in range(e):
                if pow(b, order // p, n) == 1:
                    order //= p
                    i += 1
                else:
                    break
            if i < e:
                order_factors[p] = e - i

    if prime_order is None:
        prime_order = isprime(order)

    if order < 1000:
        return _discrete_log_trial_mul(n, a, b, order)
    elif prime_order:
        # Shanks and Pollard rho are O(sqrt(order)) while index calculus is O(exp(2*sqrt(log(n)log(log(n)))))
        # we compare the expected running times to determine the algorithm which is expected to be faster
        if 4*sqrt(log(n)*log(log(n))) < log(order) - 10:  # the number 10 was determined experimental
            return _discrete_log_index_calculus(n, a, b, order)
        elif order < 1000000000000:
            # Shanks seems typically faster, but uses O(sqrt(order)) memory
            return _discrete_log_shanks_steps(n, a, b, order)
        return _discrete_log_pollard_rho(n, a, b, order)

    return _discrete_log_pohlig_hellman(n, a, b, order, order_factors)



def quadratic_congruence(a, b, c, n):
    r"""
    Find the solutions to `a x^2 + b x + c \equiv 0 \pmod{n}`.

    Parameters
    ==========

    a : int
    b : int
    c : int
    n : int
        A positive integer.

    Returns
    =======

    list[int] :
        A sorted list of solutions. If no solution exists, ``[]``.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import quadratic_congruence
    >>> quadratic_congruence(2, 5, 3, 7) # 2x^2 + 5x + 3 = 0 (mod 7)
    [2, 6]
    >>> quadratic_congruence(8, 6, 4, 15) # No solution
    []

    See Also
    ========

    polynomial_congruence : Solve the polynomial congruence

    """
    a = as_int(a)
    b = as_int(b)
    c = as_int(c)
    n = as_int(n)
    if n <= 1:
        raise ValueError("n should be an integer greater than 1")
    a %= n
    b %= n
    c %= n

    if a == 0:
        return linear_congruence(b, -c, n)
    if n == 2:
        # assert a == 1
        roots = []
        if c == 0:
            roots.append(0)
        if (b + c) % 2:
            roots.append(1)
        return roots
    if gcd(2*a, n) == 1:
        inv_a = invert(a, n)
        b *= inv_a
        c *= inv_a
        if b % 2:
            b += n
        b >>= 1
        return sorted((i - b) % n for i in sqrt_mod_iter(b**2 - c, n))
    res = set()
    for i in sqrt_mod_iter(b**2 - 4*a*c, 4*a*n):
        q, rem = divmod(i - b, 2*a)
        if rem == 0:
            res.add(q % n)

    return sorted(res)


def _valid_expr(expr):
    """
    return coefficients of expr if it is a univariate polynomial
    with integer coefficients else raise a ValueError.
    """

    if not expr.is_polynomial():
        raise ValueError("The expression should be a polynomial")
    polynomial = Poly(expr)
    if not polynomial.is_univariate:
        raise ValueError("The expression should be univariate")
    if not polynomial.domain == ZZ:
        raise ValueError("The expression should should have integer coefficients")
    return polynomial.all_coeffs()


def polynomial_congruence(expr, m):
    """
    Find the solutions to a polynomial congruence equation modulo m.

    Parameters
    ==========

    expr : integer coefficient polynomial
    m : positive integer

    Examples
    ========

    >>> from sympy.ntheory import polynomial_congruence
    >>> from sympy.abc import x
    >>> expr = x**6 - 2*x**5 -35
    >>> polynomial_congruence(expr, 6125)
    [3257]

    See Also
    ========

    sympy.polys.galoistools.gf_csolve : low level solving routine used by this routine

    """
    coefficients = _valid_expr(expr)
    coefficients = [num % m for num in coefficients]
    rank = len(coefficients)
    if rank == 3:
        return quadratic_congruence(*coefficients, m)
    if rank == 2:
        return quadratic_congruence(0, *coefficients, m)
    if coefficients[0] == 1 and 1 + coefficients[-1] == sum(coefficients):
        return nthroot_mod(-coefficients[-1], rank - 1, m, True)
    return gf_csolve(coefficients, m)


def binomial_mod(n, m, k):
    """Compute ``binomial(n, m) % k``.

    Explanation
    ===========

    Returns ``binomial(n, m) % k`` using a generalization of Lucas'
    Theorem for prime powers given by Granville [1]_, in conjunction with
    the Chinese Remainder Theorem.  The residue for each prime power
    is calculated in time O(log^2(n) + q^4*log(n)log(p) + q^4*p*log^3(p)).

    Parameters
    ==========

    n : an integer
    m : an integer
    k : a positive integer

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import binomial_mod
    >>> binomial_mod(10, 2, 6)  # binomial(10, 2) = 45
    3
    >>> binomial_mod(17, 9, 10)  # binomial(17, 9) = 24310
    0

    References
    ==========

    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """
    if k < 1: raise ValueError('k is required to be positive')
    # We decompose q into a product of prime powers and apply
    # the generalization of Lucas' Theorem given by Granville
    # to obtain binomial(n, k) mod p^e, and then use the Chinese
    # Remainder Theorem to obtain the result mod q
    if n < 0 or m < 0 or m > n: return 0
    factorisation = factorint(k)
    residues = [_binomial_mod_prime_power(n, m, p, e) for p, e in factorisation.items()]
    return crt([p**pw for p, pw in factorisation.items()], residues, check=False)[0]


def _binomial_mod_prime_power(n, m, p, q):
    """Compute ``binomial(n, m) % p**q`` for a prime ``p``.

    Parameters
    ==========

    n : positive integer
    m : a nonnegative integer
    p : a prime
    q : a positive integer (the prime exponent)

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _binomial_mod_prime_power
    >>> _binomial_mod_prime_power(10, 2, 3, 2)  # binomial(10, 2) = 45
    0
    >>> _binomial_mod_prime_power(17, 9, 2, 4)  # binomial(17, 9) = 24310
    6

    References
    ==========

    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """
    # Function/variable naming within this function follows Ref.[1]
    # n!_p will be used to denote the product of integers <= n not divisible by
    # p, with binomial(n, m)_p the same as binomial(n, m), but defined using
    # n!_p in place of n!
    modulo = pow(p, q)

    def up_factorial(u):
        """Compute (u*p)!_p modulo p^q."""
        r = q // 2
        fac = prod = 1
        if r == 1 and p == 2 or 2*r + 1 in (p, p*p):
            if q % 2 == 1: r += 1
            modulo, div = pow(p, 2*r), pow(p, 2*r - q)
        else:
            modulo, div = pow(p, 2*r + 1), pow(p, (2*r + 1) - q)
        for j in range(1, r + 1):
            for mul in range((j - 1)*p + 1, j*p):  # ignore jp itself
                fac *= mul
                fac %= modulo
            bj_ = bj(u, j, r)
            prod *= pow(fac, bj_, modulo)
            prod %= modulo
        if p == 2:
            sm = u // 2
            for j in range(1, r + 1): sm += j//2 * bj(u, j, r)
            if sm % 2 == 1: prod *= -1
        prod %= modulo//div
        return prod % modulo

    def bj(u, j, r):
        """Compute the exponent of (j*p)!_p in the calculation of (u*p)!_p."""
        prod = u
        for i in range(1, r + 1):
            if i != j: prod *= u*u - i*i
        for i in range(1, r + 1):
            if i != j: prod //= j*j - i*i
        return prod // j

    def up_plus_v_binom(u, v):
        """Compute binomial(u*p + v, v)_p modulo p^q."""
        prod = 1
        div = invert(factorial(v), modulo)
        for j in range(1, q):
            b = div
            for v_ in range(j*p + 1, j*p + v + 1):
                b *= v_
                b %= modulo
            aj = u
            for i in range(1, q):
                if i != j: aj *= u - i
            for i in range(1, q):
                if i != j: aj //= j - i
            aj //= j
            prod *= pow(b, aj, modulo)
            prod %= modulo
        return prod

    @recurrence_memo([1])
    def factorial(v, prev):
        """Compute v! modulo p^q."""
        return v*prev[-1] % modulo

    def factorial_p(n):
        """Compute n!_p modulo p^q."""
        u, v = divmod(n, p)
        return (factorial(v) * up_factorial(u) * up_plus_v_binom(u, v)) % modulo

    prod = 1
    Nj, Mj, Rj = n, m, n - m
    # e0 will be the p-adic valuation of binomial(n, m) at p
    e0 = carry = eq_1 = j = 0
    while Nj:
        numerator = factorial_p(Nj % modulo)
        denominator = factorial_p(Mj % modulo) * factorial_p(Rj % modulo) % modulo
        Nj, (Mj, mj), (Rj, rj) = Nj//p, divmod(Mj, p), divmod(Rj, p)
        carry = (mj + rj + carry) // p
        e0 += carry
        if j >= q - 1: eq_1 += carry
        prod *= numerator * invert(denominator, modulo)
        prod %= modulo
        j += 1

    mul = pow(1 if p == 2 and q >= 3 else -1, eq_1, modulo)
    return (pow(p, e0, modulo) * mul * prod) % modulo
