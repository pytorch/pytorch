"""
Integer factorization
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict, OrderedDict
from collections.abc import MutableMapping
import math

from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, Integer
from sympy.core.intfunc import num_digits
from sympy.core.power import Pow
from sympy.core.random import _randint
from sympy.core.singleton import S
from sympy.external.gmpy import (SYMPY_INTS, gcd, sqrt as isqrt,
                                 sqrtrem, iroot, bit_scan1, remove)
from .primetest import isprime, MERSENNE_PRIME_EXPONENTS, is_mersenne_prime
from .generate import sieve, primerange, nextprime
from .digits import digits
from sympy.utilities.decorator import deprecated
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor


def smoothness(n):
    """
    Return the B-smooth and B-power smooth values of n.

    The smoothness of n is the largest prime factor of n; the power-
    smoothness is the largest divisor raised to its multiplicity.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import smoothness
    >>> smoothness(2**7*3**2)
    (3, 128)
    >>> smoothness(2**4*13)
    (13, 16)
    >>> smoothness(2)
    (2, 2)

    See Also
    ========

    factorint, smoothness_p
    """

    if n == 1:
        return (1, 1)  # not prime, but otherwise this causes headaches
    facs = factorint(n)
    return max(facs), max(m**facs[m] for m in facs)


def smoothness_p(n, m=-1, power=0, visual=None):
    """
    Return a list of [m, (p, (M, sm(p + m), psm(p + m)))...]
    where:

    1. p**M is the base-p divisor of n
    2. sm(p + m) is the smoothness of p + m (m = -1 by default)
    3. psm(p + m) is the power smoothness of p + m

    The list is sorted according to smoothness (default) or by power smoothness
    if power=1.

    The smoothness of the numbers to the left (m = -1) or right (m = 1) of a
    factor govern the results that are obtained from the p +/- 1 type factoring
    methods.

        >>> from sympy.ntheory.factor_ import smoothness_p, factorint
        >>> smoothness_p(10431, m=1)
        (1, [(3, (2, 2, 4)), (19, (1, 5, 5)), (61, (1, 31, 31))])
        >>> smoothness_p(10431)
        (-1, [(3, (2, 2, 2)), (19, (1, 3, 9)), (61, (1, 5, 5))])
        >>> smoothness_p(10431, power=1)
        (-1, [(3, (2, 2, 2)), (61, (1, 5, 5)), (19, (1, 3, 9))])

    If visual=True then an annotated string will be returned:

        >>> print(smoothness_p(21477639576571, visual=1))
        p**i=4410317**1 has p-1 B=1787, B-pow=1787
        p**i=4869863**1 has p-1 B=2434931, B-pow=2434931

    This string can also be generated directly from a factorization dictionary
    and vice versa:

        >>> factorint(17*9)
        {3: 2, 17: 1}
        >>> smoothness_p(_)
        'p**i=3**2 has p-1 B=2, B-pow=2\\np**i=17**1 has p-1 B=2, B-pow=16'
        >>> smoothness_p(_)
        {3: 2, 17: 1}

    The table of the output logic is:

        ====== ====== ======= =======
        |              Visual
        ------ ----------------------
        Input  True   False   other
        ====== ====== ======= =======
        dict    str    tuple   str
        str     str    tuple   dict
        tuple   str    tuple   str
        n       str    tuple   tuple
        mul     str    tuple   tuple
        ====== ====== ======= =======

    See Also
    ========

    factorint, smoothness
    """

    # visual must be True, False or other (stored as None)
    if visual in (1, 0):
        visual = bool(visual)
    elif visual not in (True, False):
        visual = None

    if isinstance(n, str):
        if visual:
            return n
        d = {}
        for li in n.splitlines():
            k, v = [int(i) for i in
                    li.split('has')[0].split('=')[1].split('**')]
            d[k] = v
        if visual is not True and visual is not False:
            return d
        return smoothness_p(d, visual=False)
    elif not isinstance(n, tuple):
        facs = factorint(n, visual=False)

    if power:
        k = -1
    else:
        k = 1
    if isinstance(n, tuple):
        rv = n
    else:
        rv = (m, sorted([(f,
                          tuple([M] + list(smoothness(f + m))))
                         for f, M in list(facs.items())],
                        key=lambda x: (x[1][k], x[0])))

    if visual is False or (visual is not True) and (type(n) in [int, Mul]):
        return rv
    lines = []
    for dat in rv[1]:
        dat = flatten(dat)
        dat.insert(2, m)
        lines.append('p**i=%i**%i has p%+i B=%i, B-pow=%i' % tuple(dat))
    return '\n'.join(lines)


def multiplicity(p, n):
    """
    Find the greatest integer m such that p**m divides n.

    Examples
    ========

    >>> from sympy import multiplicity, Rational
    >>> [multiplicity(5, n) for n in [8, 5, 25, 125, 250]]
    [0, 1, 2, 3, 3]
    >>> multiplicity(3, Rational(1, 9))
    -2

    Note: when checking for the multiplicity of a number in a
    large factorial it is most efficient to send it as an unevaluated
    factorial or to call ``multiplicity_in_factorial`` directly:

    >>> from sympy.ntheory import multiplicity_in_factorial
    >>> from sympy import factorial
    >>> p = factorial(25)
    >>> n = 2**100
    >>> nfac = factorial(n, evaluate=False)
    >>> multiplicity(p, nfac)
    52818775009509558395695966887
    >>> _ == multiplicity_in_factorial(p, n)
    True

    See Also
    ========

    trailing

    """
    try:
        p, n = as_int(p), as_int(n)
    except ValueError:
        from sympy.functions.combinatorial.factorials import factorial
        if all(isinstance(i, (SYMPY_INTS, Rational)) for i in (p, n)):
            p = Rational(p)
            n = Rational(n)
            if p.q == 1:
                if n.p == 1:
                    return -multiplicity(p.p, n.q)
                return multiplicity(p.p, n.p) - multiplicity(p.p, n.q)
            elif p.p == 1:
                return multiplicity(p.q, n.q)
            else:
                like = min(
                    multiplicity(p.p, n.p),
                    multiplicity(p.q, n.q))
                cross = min(
                    multiplicity(p.q, n.p),
                    multiplicity(p.p, n.q))
                return like - cross
        elif (isinstance(p, (SYMPY_INTS, Integer)) and
                isinstance(n, factorial) and
                isinstance(n.args[0], Integer) and
                n.args[0] >= 0):
            return multiplicity_in_factorial(p, n.args[0])
        raise ValueError('expecting ints or fractions, got %s and %s' % (p, n))

    if n == 0:
        raise ValueError('no such integer exists: multiplicity of %s is not-defined' %(n))
    return remove(n, p)[1]


def multiplicity_in_factorial(p, n):
    """return the largest integer ``m`` such that ``p**m`` divides ``n!``
    without calculating the factorial of ``n``.

    Parameters
    ==========

    p : Integer
        positive integer
    n : Integer
        non-negative integer

    Examples
    ========

    >>> from sympy.ntheory import multiplicity_in_factorial
    >>> from sympy import factorial

    >>> multiplicity_in_factorial(2, 3)
    1

    An instructive use of this is to tell how many trailing zeros
    a given factorial has. For example, there are 6 in 25!:

    >>> factorial(25)
    15511210043330985984000000
    >>> multiplicity_in_factorial(10, 25)
    6

    For large factorials, it is much faster/feasible to use
    this function rather than computing the actual factorial:

    >>> multiplicity_in_factorial(factorial(25), 2**100)
    52818775009509558395695966887

    See Also
    ========

    multiplicity

    """

    p, n = as_int(p), as_int(n)

    if p <= 0:
        raise ValueError('expecting positive integer got %s' % p )

    if n < 0:
        raise ValueError('expecting non-negative integer got %s' % n )

    # keep only the largest of a given multiplicity since those
    # of a given multiplicity will be goverened by the behavior
    # of the largest factor
    f = defaultdict(int)
    for k, v in factorint(p).items():
        f[v] = max(k, f[v])
    # multiplicity of p in n! depends on multiplicity
    # of prime `k` in p, so we floor divide by `v`
    # and keep it if smaller than the multiplicity of p
    # seen so far
    return min((n + k - sum(digits(n, k)))//(k - 1)//v for v, k in f.items())


def _perfect_power(n, next_p=2):
    """ Return integers ``(b, e)`` such that ``n == b**e`` if ``n`` is a unique
    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a perfect power).

    Explanation
    ===========

    This is a low-level helper for ``perfect_power``, for internal use.

    Parameters
    ==========

    n : int
        assume that n is a nonnegative integer
    next_p : int
        Assume that n has no factor less than next_p.
        i.e., all(n % p for p in range(2, next_p)) is True

    Examples
    ========
    >>> from sympy.ntheory.factor_ import _perfect_power
    >>> _perfect_power(16)
    (2, 4)
    >>> _perfect_power(17)
    False

    """
    if n <= 3:
        return False

    factors = {}
    g = 0
    multi = 1

    def done(n, factors, g, multi):
        g = gcd(g, multi)
        if g == 1:
            return False
        factors[n] = multi
        return math.prod(p**(e//g) for p, e in factors.items()), g

    # If n is small, only trial factoring is faster
    if n <= 1_000_000:
        n = _factorint_small(factors, n, 1_000, 1_000, next_p)[0]
        if n > 1:
            return False
        g = gcd(*factors.values())
        if g == 1:
            return False
        return math.prod(p**(e//g) for p, e in factors.items()), g

    # divide by 2
    if next_p < 3:
        g = bit_scan1(n)
        if g:
            if g == 1:
                return False
            n >>= g
            factors[2] = g
            if n == 1:
                return 2, g
            else:
                # If `m**g`, then we have found perfect power.
                # Otherwise, there is no possibility of perfect power, especially if `g` is prime.
                m, _exact = iroot(n, g)
                if _exact:
                    return 2*m, g
                elif isprime(g):
                    return False
        next_p = 3

    # square number?
    while n & 7 == 1: # n % 8 == 1:
        m, _exact = iroot(n, 2)
        if _exact:
            n = m
            multi <<= 1
        else:
            break
    if n < next_p**3:
        return done(n, factors, g, multi)

    # trial factoring
    # Since the maximum value an exponent can take is `log_{next_p}(n)`,
    # the number of exponents to be checked can be reduced by performing a trial factoring.
    # The value of `tf_max` needs more consideration.
    tf_max = n.bit_length()//27 + 24
    if next_p < tf_max:
        for p in primerange(next_p, tf_max):
            m, t = remove(n, p)
            if t:
                n = m
                t *= multi
                _g = gcd(g, t)
                if _g == 1:
                    return False
                factors[p] = t
                if n == 1:
                    return math.prod(p**(e//_g)
                                        for p, e in factors.items()), _g
                elif g == 0 or _g < g: # If g is updated
                    g = _g
                    m, _exact = iroot(n**multi, g)
                    if _exact:
                        return m * math.prod(p**(e//g)
                                            for p, e in factors.items()), g
                    elif isprime(g):
                        return False
        next_p = tf_max
    if n < next_p**3:
        return done(n, factors, g, multi)

    # check iroot
    if g:
        # If g is non-zero, the exponent is a divisor of g.
        # 2 can be omitted since it has already been checked.
        prime_iter = sorted(factorint(g >> bit_scan1(g)).keys())
    else:
        # The maximum possible value of the exponent is `log_{next_p}(n)`.
        # To compensate for the presence of computational error, 2 is added.
        prime_iter = primerange(3, int(math.log(n, next_p)) + 2)
    logn = math.log2(n)
    threshold = logn / 40 # Threshold for direct calculation
    for p in prime_iter:
        if threshold < p:
            # If p is large, find the power root p directly without `iroot`.
            while True:
                b = pow(2, logn / p)
                rb = int(b + 0.5)
                if abs(rb - b) < 0.01 and rb**p == n:
                    n = rb
                    multi *= p
                    logn = math.log2(n)
                else:
                    break
        else:
            while True:
                m, _exact = iroot(n, p)
                if _exact:
                    n = m
                    multi *= p
                    logn = math.log2(n)
                else:
                    break
        if n < next_p**(p + 2):
            break
    return done(n, factors, g, multi)


def perfect_power(n, candidates=None, big=True, factor=True):
    """
    Return ``(b, e)`` such that ``n`` == ``b**e`` if ``n`` is a unique
    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a
    perfect power). A ValueError is raised if ``n`` is not Rational.

    By default, the base is recursively decomposed and the exponents
    collected so the largest possible ``e`` is sought. If ``big=False``
    then the smallest possible ``e`` (thus prime) will be chosen.

    If ``factor=True`` then simultaneous factorization of ``n`` is
    attempted since finding a factor indicates the only possible root
    for ``n``. This is True by default since only a few small factors will
    be tested in the course of searching for the perfect power.

    The use of ``candidates`` is primarily for internal use; if provided,
    False will be returned if ``n`` cannot be written as a power with one
    of the candidates as an exponent and factoring (beyond testing for
    a factor of 2) will not be attempted.

    Examples
    ========

    >>> from sympy import perfect_power, Rational
    >>> perfect_power(16)
    (2, 4)
    >>> perfect_power(16, big=False)
    (4, 2)

    Negative numbers can only have odd perfect powers:

    >>> perfect_power(-4)
    False
    >>> perfect_power(-8)
    (-2, 3)

    Rationals are also recognized:

    >>> perfect_power(Rational(1, 2)**3)
    (1/2, 3)
    >>> perfect_power(Rational(-3, 2)**3)
    (-3/2, 3)

    Notes
    =====

    To know whether an integer is a perfect power of 2 use

        >>> is2pow = lambda n: bool(n and not n & (n - 1))
        >>> [(i, is2pow(i)) for i in range(5)]
        [(0, False), (1, True), (2, True), (3, False), (4, True)]

    It is not necessary to provide ``candidates``. When provided
    it will be assumed that they are ints. The first one that is
    larger than the computed maximum possible exponent will signal
    failure for the routine.

        >>> perfect_power(3**8, [9])
        False
        >>> perfect_power(3**8, [2, 4, 8])
        (3, 8)
        >>> perfect_power(3**8, [4, 8], big=False)
        (9, 4)

    See Also
    ========
    sympy.core.intfunc.integer_nthroot
    sympy.ntheory.primetest.is_square
    """
    # negative handling
    if n < 0:
        if candidates is None:
            pp = perfect_power(-n, big=True, factor=factor)
            if not pp:
                return False

            b, e = pp
            e2 = e & (-e)
            b, e = b ** e2, e // e2

            if e <= 1:
                return False

            if big or isprime(e):
                return -b, e

            for p in primerange(3, e + 1):
                if e % p == 0:
                    return - b ** (e // p), p

        odd_candidates = {i for i in candidates if i % 2}
        if not odd_candidates:
            return False

        pp = perfect_power(-n, odd_candidates, big, factor)
        if pp:
            return -pp[0], pp[1]

        return False

    # non-integer handling
    if isinstance(n, Rational) and not isinstance(n, Integer):
        p, q = n.p, n.q

        if p == 1:
            qq = perfect_power(q, candidates, big, factor)
            return (S.One / qq[0], qq[1]) if qq is not False else False

        if not (pp:=perfect_power(p, factor=factor)):
            return False
        if not (qq:=perfect_power(q, factor=factor)):
            return False
        (num_base, num_exp), (den_base, den_exp) = pp, qq

        def compute_tuple(exponent):
            """Helper to compute final result given an exponent"""
            new_num = num_base ** (num_exp // exponent)
            new_den = den_base ** (den_exp // exponent)
            return n.func(new_num, new_den), exponent

        if candidates:
            valid_candidates = [i for i in candidates
                                if num_exp % i == 0 and den_exp % i == 0]
            if not valid_candidates:
                return False

            e = max(valid_candidates) if big else min(valid_candidates)
            return compute_tuple(e)

        g = math.gcd(num_exp, den_exp)
        if g == 1:
            return False

        if big:
            return compute_tuple(g)

        e = next(p for p in primerange(2, g + 1) if g % p == 0)
        return compute_tuple(e)

    if candidates is not None:
        candidates = set(candidates)

    # positive integer handling
    n = as_int(n)

    if candidates is None and big:
        return _perfect_power(n)

    if n <= 3:
        # no unique exponent for 0, 1
        # 2 and 3 have exponents of 1
        return False
    logn = math.log2(n)
    max_possible = int(logn) + 2  # only check values less than this
    not_square = n % 10 in [2, 3, 7, 8]  # squares cannot end in 2, 3, 7, 8
    min_possible = 2 + not_square
    if not candidates:
        candidates = primerange(min_possible, max_possible)
    else:
        candidates = sorted([i for i in candidates
            if min_possible <= i < max_possible])
        if n%2 == 0:
            e = bit_scan1(n)
            candidates = [i for i in candidates if e%i == 0]
        if big:
            candidates = reversed(candidates)
        for e in candidates:
            r, ok = iroot(n, e)
            if ok:
                return int(r), e
        return False

    def _factors():
        rv = 2 + n % 2
        while True:
            yield rv
            rv = nextprime(rv)

    for fac, e in zip(_factors(), candidates):
        # see if there is a factor present
        if factor and n % fac == 0:
            # find what the potential power is
            e = remove(n, fac)[1]
            # if it's a trivial power we are done
            if e == 1:
                return False

            # maybe the e-th root of n is exact
            r, exact = iroot(n, e)
            if not exact:
                # Having a factor, we know that e is the maximal
                # possible value for a root of n.
                # If n = fac**e*m can be written as a perfect
                # power then see if m can be written as r**E where
                # gcd(e, E) != 1 so n = (fac**(e//E)*r)**E
                m = n//fac**e
                rE = perfect_power(m, candidates=divisors(e, generator=True))
                if not rE:
                    return False
                else:
                    r, E = rE
                    r, e = fac**(e//E)*r, E
            if not big:
                e0 = primefactors(e)
                if e0[0] != e:
                    r, e = r**(e//e0[0]), e0[0]
            return int(r), e

        # Weed out downright impossible candidates
        if logn/e < 40:
            b = 2.0**(logn/e)
            if abs(int(b + 0.5) - b) > 0.01:
                continue

        # now see if the plausible e makes a perfect power
        r, exact = iroot(n, e)
        if exact:
            if big:
                m = perfect_power(r, big=big, factor=factor)
                if m:
                    r, e = m[0], e*m[1]
            return int(r), e

    return False


class FactorCache(MutableMapping):
    """ Provides a cache for prime factors.
    ``factor_cache`` is pre-prepared as an instance of ``FactorCache``,
    and ``factorint`` internally references it to speed up
    the factorization of prime factors.

    While cache is automatically added during the execution of ``factorint``,
    users can also manually add prime factors independently.

    >>> from sympy import factor_cache
    >>> factor_cache[15] = 5

    Furthermore, by customizing ``get_external``,
    it is also possible to use external databases.
    The following is an example using http://factordb.com .

    .. code-block:: python

        import requests
        from sympy import factor_cache

        def get_external(self, n: int) -> list[int] | None:
            res = requests.get("http://factordb.com/api", params={"query": str(n)})
            if res.status_code != requests.codes.ok:
                return None
            j = res.json()
            if j.get("status") in ["FF", "P"]:
                return list(int(p) for p, _ in j.get("factors"))

        factor_cache.get_external = get_external

    Be aware that writing this code will trigger internet access
    to factordb.com when calling ``factorint``.

    """
    def __init__(self, maxsize: int | None = None):
        self._cache: OrderedDict[int, int] = OrderedDict()
        self.maxsize = maxsize

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, n) -> bool:
        return n in self._cache

    def __getitem__(self, n: int) -> int:
        factor = self.get(n)
        if factor is None:
            raise KeyError(f"{n} does not exist.")
        return factor

    def __setitem__(self, n: int, factor: int):
        if not (1 < factor <= n and n % factor == 0 and isprime(factor)):
            raise ValueError(f"{factor} is not a prime factor of {n}")
        self._cache[n] = max(self._cache.get(n, 0), factor)
        if self.maxsize is not None and len(self._cache) > self.maxsize:
            self._cache.popitem(False)

    def __delitem__(self, n: int):
        if n not in self._cache:
            raise KeyError(f"{n} does not exist.")
        del self._cache[n]

    def __iter__(self):
        return self._cache.__iter__()

    def cache_clear(self) -> None:
        """ Clear the cache """
        self._cache = OrderedDict()

    @property
    def maxsize(self) -> int | None:
        """ Returns the maximum cache size; if ``None``, it is unlimited. """
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value: int | None) -> None:
        if value is not None and value <= 0:
            raise ValueError("maxsize must be None or a non-negative integer.")
        self._maxsize = value
        if value is not None:
            while len(self._cache) > value:
                self._cache.popitem(False)

    def get(self, n: int, default=None):
        """ Return the prime factor of ``n``.
        If it does not exist in the cache, return the value of ``default``.
        """
        if n <= sieve._list[-1]:
            if sieve._list[bisect_left(sieve._list, n)] == n:
                return n
        if n in self._cache:
            self._cache.move_to_end(n)
            return self._cache[n]
        if factors := self.get_external(n):
            self.add(n, factors)
            return self._cache[n]
        return default

    def add(self, n: int, factors: list[int]) -> None:
        for p in sorted(factors, reverse=True):
            self[n] = p
            n, _ = remove(n, p)

    def get_external(self, n: int) -> list[int] | None:
        return None


factor_cache = FactorCache(maxsize=1000)


def pollard_rho(n, s=2, a=1, retries=5, seed=1234, max_steps=None, F=None):
    r"""
    Use Pollard's rho method to try to extract a nontrivial factor
    of ``n``. The returned factor may be a composite number. If no
    factor is found, ``None`` is returned.

    The algorithm generates pseudo-random values of x with a generator
    function, replacing x with F(x). If F is not supplied then the
    function x**2 + ``a`` is used. The first value supplied to F(x) is ``s``.
    Upon failure (if ``retries`` is > 0) a new ``a`` and ``s`` will be
    supplied; the ``a`` will be ignored if F was supplied.

    The sequence of numbers generated by such functions generally have a
    a lead-up to some number and then loop around back to that number and
    begin to repeat the sequence, e.g. 1, 2, 3, 4, 5, 3, 4, 5 -- this leader
    and loop look a bit like the Greek letter rho, and thus the name, 'rho'.

    For a given function, very different leader-loop values can be obtained
    so it is a good idea to allow for retries:

    >>> from sympy.ntheory.generate import cycle_length
    >>> n = 16843009
    >>> F = lambda x:(2048*pow(x, 2, n) + 32767) % n
    >>> for s in range(5):
    ...     print('loop length = %4i; leader length = %3i' % next(cycle_length(F, s)))
    ...
    loop length = 2489; leader length =  43
    loop length =   78; leader length = 121
    loop length = 1482; leader length = 100
    loop length = 1482; leader length = 286
    loop length = 1482; leader length = 101

    Here is an explicit example where there is a three element leadup to
    a sequence of 3 numbers (11, 14, 4) that then repeat:

    >>> x=2
    >>> for i in range(9):
    ...     print(x)
    ...     x=(x**2+12)%17
    ...
    2
    16
    13
    11
    14
    4
    11
    14
    4
    >>> next(cycle_length(lambda x: (x**2+12)%17, 2))
    (3, 3)
    >>> list(cycle_length(lambda x: (x**2+12)%17, 2, values=True))
    [2, 16, 13, 11, 14, 4]

    Instead of checking the differences of all generated values for a gcd
    with n, only the kth and 2*kth numbers are checked, e.g. 1st and 2nd,
    2nd and 4th, 3rd and 6th until it has been detected that the loop has been
    traversed. Loops may be many thousands of steps long before rho finds a
    factor or reports failure. If ``max_steps`` is specified, the iteration
    is cancelled with a failure after the specified number of steps.

    Examples
    ========

    >>> from sympy import pollard_rho
    >>> n=16843009
    >>> F=lambda x:(2048*pow(x,2,n) + 32767) % n
    >>> pollard_rho(n, F=F)
    257

    Use the default setting with a bad value of ``a`` and no retries:

    >>> pollard_rho(n, a=n-2, retries=0)

    If retries is > 0 then perhaps the problem will correct itself when
    new values are generated for a:

    >>> pollard_rho(n, a=n-2, retries=1)
    257

    References
    ==========

    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 229-231

    """
    n = int(n)
    if n < 5:
        raise ValueError('pollard_rho should receive n > 4')
    randint = _randint(seed + retries)
    V = s
    for i in range(retries + 1):
        U = V
        if not F:
            F = lambda x: (pow(x, 2, n) + a) % n
        j = 0
        while 1:
            if max_steps and (j > max_steps):
                break
            j += 1
            U = F(U)
            V = F(F(V))  # V is 2x further along than U
            g = gcd(U - V, n)
            if g == 1:
                continue
            if g == n:
                break
            return int(g)
        V = randint(0, n - 1)
        a = randint(1, n - 3)  # for x**2 + a, a%n should not be 0 or -2
        F = None
    return None


def pollard_pm1(n, B=10, a=2, retries=0, seed=1234):
    """
    Use Pollard's p-1 method to try to extract a nontrivial factor
    of ``n``. Either a divisor (perhaps composite) or ``None`` is returned.

    The value of ``a`` is the base that is used in the test gcd(a**M - 1, n).
    The default is 2.  If ``retries`` > 0 then if no factor is found after the
    first attempt, a new ``a`` will be generated randomly (using the ``seed``)
    and the process repeated.

    Note: the value of M is lcm(1..B) = reduce(ilcm, range(2, B + 1)).

    A search is made for factors next to even numbers having a power smoothness
    less than ``B``. Choosing a larger B increases the likelihood of finding a
    larger factor but takes longer. Whether a factor of n is found or not
    depends on ``a`` and the power smoothness of the even number just less than
    the factor p (hence the name p - 1).

    Although some discussion of what constitutes a good ``a`` some
    descriptions are hard to interpret. At the modular.math site referenced
    below it is stated that if gcd(a**M - 1, n) = N then a**M % q**r is 1
    for every prime power divisor of N. But consider the following:

        >>> from sympy.ntheory.factor_ import smoothness_p, pollard_pm1
        >>> n=257*1009
        >>> smoothness_p(n)
        (-1, [(257, (1, 2, 256)), (1009, (1, 7, 16))])

    So we should (and can) find a root with B=16:

        >>> pollard_pm1(n, B=16, a=3)
        1009

    If we attempt to increase B to 256 we find that it does not work:

        >>> pollard_pm1(n, B=256)
        >>>

    But if the value of ``a`` is changed we find that only multiples of
    257 work, e.g.:

        >>> pollard_pm1(n, B=256, a=257)
        1009

    Checking different ``a`` values shows that all the ones that did not
    work had a gcd value not equal to ``n`` but equal to one of the
    factors:

        >>> from sympy import ilcm, igcd, factorint, Pow
        >>> M = 1
        >>> for i in range(2, 256):
        ...     M = ilcm(M, i)
        ...
        >>> set([igcd(pow(a, M, n) - 1, n) for a in range(2, 256) if
        ...      igcd(pow(a, M, n) - 1, n) != n])
        {1009}

    But does aM % d for every divisor of n give 1?

        >>> aM = pow(255, M, n)
        >>> [(d, aM%Pow(*d.args)) for d in factorint(n, visual=True).args]
        [(257**1, 1), (1009**1, 1)]

    No, only one of them. So perhaps the principle is that a root will
    be found for a given value of B provided that:

    1) the power smoothness of the p - 1 value next to the root
       does not exceed B
    2) a**M % p != 1 for any of the divisors of n.

    By trying more than one ``a`` it is possible that one of them
    will yield a factor.

    Examples
    ========

    With the default smoothness bound, this number cannot be cracked:

        >>> from sympy.ntheory import pollard_pm1
        >>> pollard_pm1(21477639576571)

    Increasing the smoothness bound helps:

        >>> pollard_pm1(21477639576571, B=2000)
        4410317

    Looking at the smoothness of the factors of this number we find:

        >>> from sympy.ntheory.factor_ import smoothness_p, factorint
        >>> print(smoothness_p(21477639576571, visual=1))
        p**i=4410317**1 has p-1 B=1787, B-pow=1787
        p**i=4869863**1 has p-1 B=2434931, B-pow=2434931

    The B and B-pow are the same for the p - 1 factorizations of the divisors
    because those factorizations had a very large prime factor:

        >>> factorint(4410317 - 1)
        {2: 2, 617: 1, 1787: 1}
        >>> factorint(4869863-1)
        {2: 1, 2434931: 1}

    Note that until B reaches the B-pow value of 1787, the number is not cracked;

        >>> pollard_pm1(21477639576571, B=1786)
        >>> pollard_pm1(21477639576571, B=1787)
        4410317

    The B value has to do with the factors of the number next to the divisor,
    not the divisors themselves. A worst case scenario is that the number next
    to the factor p has a large prime divisisor or is a perfect power. If these
    conditions apply then the power-smoothness will be about p/2 or p. The more
    realistic is that there will be a large prime factor next to p requiring
    a B value on the order of p/2. Although primes may have been searched for
    up to this level, the p/2 is a factor of p - 1, something that we do not
    know. The modular.math reference below states that 15% of numbers in the
    range of 10**15 to 15**15 + 10**4 are 10**6 power smooth so a B of 10**6
    will fail 85% of the time in that range. From 10**8 to 10**8 + 10**3 the
    percentages are nearly reversed...but in that range the simple trial
    division is quite fast.

    References
    ==========

    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 236-238
    .. [2] https://web.archive.org/web/20150716201437/http://modular.math.washington.edu/edu/2007/spring/ent/ent-html/node81.html
    .. [3] https://www.cs.toronto.edu/~yuvalf/Factorization.pdf
    """

    n = int(n)
    if n < 4 or B < 3:
        raise ValueError('pollard_pm1 should receive n > 3 and B > 2')
    randint = _randint(seed + B)

    # computing a**lcm(1,2,3,..B) % n for B > 2
    # it looks weird, but it's right: primes run [2, B]
    # and the answer's not right until the loop is done.
    for i in range(retries + 1):
        aM = a
        for p in sieve.primerange(2, B + 1):
            e = int(math.log(B, p))
            aM = pow(aM, pow(p, e), n)
        g = gcd(aM - 1, n)
        if 1 < g < n:
            return int(g)

        # get a new a:
        # since the exponent, lcm(1..B), is even, if we allow 'a' to be 'n-1'
        # then (n - 1)**even % n will be 1 which will give a g of 0 and 1 will
        # give a zero, too, so we set the range as [2, n-2]. Some references
        # say 'a' should be coprime to n, but either will detect factors.
        a = randint(2, n - 2)


def _trial(factors, n, candidates, verbose=False):
    """
    Helper function for integer factorization. Trial factors ``n`
    against all integers given in the sequence ``candidates``
    and updates the dict ``factors`` in-place. Returns the reduced
    value of ``n`` and a flag indicating whether any factors were found.
    """
    if verbose:
        factors0 = list(factors.keys())
    nfactors = len(factors)
    for d in candidates:
        if n % d == 0:
            if n != d:
                factor_cache[n] = d
            n, m = remove(n // d, d)
            factors[d] = m + 1
    if verbose:
        for k in sorted(set(factors).difference(set(factors0))):
            print(factor_msg % (k, factors[k]))
    return int(n), len(factors) != nfactors


def _check_termination(factors, n, limit, use_trial, use_rho, use_pm1,
                       verbose, next_p):
    """
    Helper function for integer factorization. Checks if ``n``
    is a prime or a perfect power, and in those cases updates the factorization.
    """
    if verbose:
        print('Check for termination')
    if n == 1:
        if verbose:
            print(complete_msg)
        return True
    if n < next_p**2 or isprime(n):
        factor_cache[n] = n
        factors[int(n)] = 1
        if verbose:
            print(complete_msg)
        return True

    # since we've already been factoring there is no need to do
    # simultaneous factoring with the power check
    p = _perfect_power(n, next_p)
    if not p:
        return False
    base, exp = p
    if base < next_p**2 or isprime(base):
        factor_cache[n] = base
        factors[base] = exp
    else:
        facs = factorint(base, limit, use_trial, use_rho, use_pm1,
                         verbose=False)
        for b, e in facs.items():
            if verbose:
                print(factor_msg % (b, e))
            factors[b] = exp*e
    if verbose:
        print(complete_msg)
    return True


trial_int_msg = "Trial division with ints [%i ... %i] and fail_max=%i"
trial_msg = "Trial division with primes [%i ... %i]"
rho_msg = "Pollard's rho with retries %i, max_steps %i and seed %i"
pm1_msg = "Pollard's p-1 with smoothness bound %i and seed %i"
ecm_msg = "Elliptic Curve with B1 bound %i, B2 bound %i, num_curves %i"
factor_msg = '\t%i ** %i'
fermat_msg = 'Close factors satisfying Fermat condition found.'
complete_msg = 'Factorization is complete.'


def _factorint_small(factors, n, limit, fail_max, next_p=2):
    """
    Return the value of n and either a 0 (indicating that factorization up
    to the limit was complete) or else the next near-prime that would have
    been tested.

    Factoring stops if there are fail_max unsuccessful tests in a row.

    If factors of n were found they will be in the factors dictionary as
    {factor: multiplicity} and the returned value of n will have had those
    factors removed. The factors dictionary is modified in-place.

    """

    def done(n, d):
        """return n, d if the sqrt(n) was not reached yet, else
           n, 0 indicating that factoring is done.
        """
        if d*d <= n:
            return n, d
        return n, 0

    limit2 = limit**2
    threshold2 = min(n, limit2)

    if next_p < 3:
        if not n & 1:
            m = bit_scan1(n)
            factors[2] = m
            n >>= m
            threshold2 = min(n, limit2)
        next_p = 3
        if threshold2 < 9: # next_p**2 = 9
            return done(n, next_p)

    if next_p < 5:
        if not n % 3:
            n //= 3
            m = 1
            while not n % 3:
                n //= 3
                m += 1
                if m == 20:
                    n, mm = remove(n, 3)
                    m += mm
                    break
            factors[3] = m
            threshold2 = min(n, limit2)
        next_p = 5
        if threshold2 < 25: # next_p**2 = 25
            return done(n, next_p)

    # Because of the order of checks, starting from `min_p = 6k+5`,
    # useless checks are caused.
    # We want to calculate
    # next_p += [-1, -2, 3, 2, 1, 0][next_p % 6]
    p6 = next_p % 6
    next_p += (-1 if p6 < 2 else 5) - p6

    fails = 0
    while fails < fail_max:
        # next_p % 6 == 5
        if n % next_p:
            fails += 1
        else:
            n //= next_p
            m = 1
            while not n % next_p:
                n //= next_p
                m += 1
                if m == 20:
                    n, mm = remove(n, next_p)
                    m += mm
                    break
            factors[next_p] = m
            fails = 0
            threshold2 = min(n, limit2)
        next_p += 2
        if threshold2 < next_p**2:
            return done(n, next_p)

        # next_p % 6 == 1
        if n % next_p:
            fails += 1
        else:
            n //= next_p
            m = 1
            while not n % next_p:
                n //= next_p
                m += 1
                if m == 20:
                    n, mm = remove(n, next_p)
                    m += mm
                    break
            factors[next_p] = m
            fails = 0
            threshold2 = min(n, limit2)
        next_p += 4
        if threshold2 < next_p**2:
            return done(n, next_p)
    return done(n, next_p)


def factorint(n, limit=None, use_trial=True, use_rho=True, use_pm1=True,
              use_ecm=True, verbose=False, visual=None, multiple=False):
    r"""
    Given a positive integer ``n``, ``factorint(n)`` returns a dict containing
    the prime factors of ``n`` as keys and their respective multiplicities
    as values. For example:

    >>> from sympy.ntheory import factorint
    >>> factorint(2000)    # 2000 = (2**4) * (5**3)
    {2: 4, 5: 3}
    >>> factorint(65537)   # This number is prime
    {65537: 1}

    For input less than 2, factorint behaves as follows:

        - ``factorint(1)`` returns the empty factorization, ``{}``
        - ``factorint(0)`` returns ``{0:1}``
        - ``factorint(-n)`` adds ``-1:1`` to the factors and then factors ``n``

    Partial Factorization:

    If ``limit`` (> 3) is specified, the search is stopped after performing
    trial division up to (and including) the limit (or taking a
    corresponding number of rho/p-1 steps). This is useful if one has
    a large number and only is interested in finding small factors (if
    any). Note that setting a limit does not prevent larger factors
    from being found early; it simply means that the largest factor may
    be composite. Since checking for perfect power is relatively cheap, it is
    done regardless of the limit setting.

    This number, for example, has two small factors and a huge
    semi-prime factor that cannot be reduced easily:

    >>> from sympy.ntheory import isprime
    >>> a = 1407633717262338957430697921446883
    >>> f = factorint(a, limit=10000)
    >>> f == {991: 1, int(202916782076162456022877024859): 1, 7: 1}
    True
    >>> isprime(max(f))
    False

    This number has a small factor and a residual perfect power whose
    base is greater than the limit:

    >>> factorint(3*101**7, limit=5)
    {3: 1, 101: 7}

    List of Factors:

    If ``multiple`` is set to ``True`` then a list containing the
    prime factors including multiplicities is returned.

    >>> factorint(24, multiple=True)
    [2, 2, 2, 3]

    Visual Factorization:

    If ``visual`` is set to ``True``, then it will return a visual
    factorization of the integer.  For example:

    >>> from sympy import pprint
    >>> pprint(factorint(4200, visual=True))
     3  1  2  1
    2 *3 *5 *7

    Note that this is achieved by using the evaluate=False flag in Mul
    and Pow. If you do other manipulations with an expression where
    evaluate=False, it may evaluate.  Therefore, you should use the
    visual option only for visualization, and use the normal dictionary
    returned by visual=False if you want to perform operations on the
    factors.

    You can easily switch between the two forms by sending them back to
    factorint:

    >>> from sympy import Mul
    >>> regular = factorint(1764); regular
    {2: 2, 3: 2, 7: 2}
    >>> pprint(factorint(regular))
     2  2  2
    2 *3 *7

    >>> visual = factorint(1764, visual=True); pprint(visual)
     2  2  2
    2 *3 *7
    >>> print(factorint(visual))
    {2: 2, 3: 2, 7: 2}

    If you want to send a number to be factored in a partially factored form
    you can do so with a dictionary or unevaluated expression:

    >>> factorint(factorint({4: 2, 12: 3})) # twice to toggle to dict form
    {2: 10, 3: 3}
    >>> factorint(Mul(4, 12, evaluate=False))
    {2: 4, 3: 1}

    The table of the output logic is:

        ====== ====== ======= =======
                       Visual
        ------ ----------------------
        Input  True   False   other
        ====== ====== ======= =======
        dict    mul    dict    mul
        n       mul    dict    dict
        mul     mul    dict    dict
        ====== ====== ======= =======

    Notes
    =====

    Algorithm:

    The function switches between multiple algorithms. Trial division
    quickly finds small factors (of the order 1-5 digits), and finds
    all large factors if given enough time. The Pollard rho and p-1
    algorithms are used to find large factors ahead of time; they
    will often find factors of the order of 10 digits within a few
    seconds:

    >>> factors = factorint(12345678910111213141516)
    >>> for base, exp in sorted(factors.items()):
    ...     print('%s %s' % (base, exp))
    ...
    2 2
    2507191691 1
    1231026625769 1

    Any of these methods can optionally be disabled with the following
    boolean parameters:

        - ``use_trial``: Toggle use of trial division
        - ``use_rho``: Toggle use of Pollard's rho method
        - ``use_pm1``: Toggle use of Pollard's p-1 method

    ``factorint`` also periodically checks if the remaining part is
    a prime number or a perfect power, and in those cases stops.

    For unevaluated factorial, it uses Legendre's formula(theorem).


    If ``verbose`` is set to ``True``, detailed progress is printed.

    See Also
    ========

    smoothness, smoothness_p, divisors

    """
    if isinstance(n, Dict):
        n = dict(n)
    if multiple:
        fac = factorint(n, limit=limit, use_trial=use_trial,
                           use_rho=use_rho, use_pm1=use_pm1,
                           verbose=verbose, visual=False, multiple=False)
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One/p]*(-fac[p])
                               for p in sorted(fac)), [])
        return factorlist

    factordict = {}
    if visual and not isinstance(n, (Mul, dict)):
        factordict = factorint(n, limit=limit, use_trial=use_trial,
                               use_rho=use_rho, use_pm1=use_pm1,
                               verbose=verbose, visual=False)
    elif isinstance(n, Mul):
        factordict = {int(k): int(v) for k, v in
            n.as_powers_dict().items()}
    elif isinstance(n, dict):
        factordict = n
    if factordict and isinstance(n, (Mul, dict)):
        # check it
        for key in list(factordict.keys()):
            if isprime(key):
                continue
            e = factordict.pop(key)
            d = factorint(key, limit=limit, use_trial=use_trial, use_rho=use_rho,
                          use_pm1=use_pm1, verbose=verbose, visual=False)
            for k, v in d.items():
                if k in factordict:
                    factordict[k] += v*e
                else:
                    factordict[k] = v*e
    if visual or (type(n) is dict and
                  visual is not True and
                  visual is not False):
        if factordict == {}:
            return S.One
        if -1 in factordict:
            factordict.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        args.extend([Pow(*i, evaluate=False)
                     for i in sorted(factordict.items())])
        return Mul(*args, evaluate=False)
    elif isinstance(n, (dict, Mul)):
        return factordict

    assert use_trial or use_rho or use_pm1 or use_ecm

    from sympy.functions.combinatorial.factorials import factorial
    if isinstance(n, factorial):
        x = as_int(n.args[0])
        if x >= 20:
            factors = {}
            m = 2 # to initialize the if condition below
            for p in sieve.primerange(2, x + 1):
                if m > 1:
                    m, q = 0, x // p
                    while q != 0:
                        m += q
                        q //= p
                factors[p] = m
            if factors and verbose:
                for k in sorted(factors):
                    print(factor_msg % (k, factors[k]))
            if verbose:
                print(complete_msg)
            return factors
        else:
            # if n < 20!, direct computation is faster
            # since it uses a lookup table
            n = n.func(x)

    n = as_int(n)
    if limit:
        limit = int(limit)
        use_ecm = False

    # special cases
    if n < 0:
        factors = factorint(
            -n, limit=limit, use_trial=use_trial, use_rho=use_rho,
            use_pm1=use_pm1, verbose=verbose, visual=False)
        factors[-1] = 1
        return factors

    if limit and limit < 2:
        if n == 1:
            return {}
        return {n: 1}
    elif n < 10:
        # doing this we are assured of getting a limit > 2
        # when we have to compute it later
        return [{0: 1}, {}, {2: 1}, {3: 1}, {2: 2}, {5: 1},
                {2: 1, 3: 1}, {7: 1}, {2: 3}, {3: 2}][n]

    factors = {}

    # do simplistic factorization
    if verbose:
        sn = str(n)
        if len(sn) > 50:
            print('Factoring %s' % sn[:5] + \
                  '..(%i other digits)..' % (len(sn) - 10) + sn[-5:])
        else:
            print('Factoring', n)

    # this is the preliminary factorization for small factors
    # We want to guarantee that there are no small prime factors,
    # so we run even if `use_trial` is False.
    small = 2**15
    fail_max = 600
    small = min(small, limit or small)
    if verbose:
        print(trial_int_msg % (2, small, fail_max))
    n, next_p = _factorint_small(factors, n, small, fail_max)
    if factors and verbose:
        for k in sorted(factors):
            print(factor_msg % (k, factors[k]))
    if next_p == 0:
        if n > 1:
            factors[int(n)] = 1
        if verbose:
            print(complete_msg)
        return factors
    # Check if it exists in the cache
    while p := factor_cache.get(n):
        n, e = remove(n, p)
        factors[int(p)] = int(e)
    # first check if the simplistic run didn't finish
    # because of the limit and check for a perfect
    # power before exiting
    if limit and next_p > limit:
        if verbose:
            print('Exceeded limit:', limit)
        if _check_termination(factors, n, limit, use_trial,
                              use_rho, use_pm1, verbose, next_p):
            return factors
        if n > 1:
            factors[int(n)] = 1
        return factors
    if _check_termination(factors, n, limit, use_trial,
                          use_rho, use_pm1, verbose, next_p):
        return factors

    # continue with more advanced factorization methods
    # ...do a Fermat test since it's so easy and we need the
    # square root anyway. Finding 2 factors is easy if they are
    # "close enough." This is the big root equivalent of dividing by
    # 2, 3, 5.
    sqrt_n = isqrt(n)
    a = sqrt_n + 1
    # If `n % 4 == 1`, `a` must be odd for `a**2 - n` to be a square number.
    if (n % 4 == 1) ^ (a & 1):
        a += 1
    a2 = a**2
    b2 = a2 - n
    for _ in range(3):
        b, fermat = sqrtrem(b2)
        if not fermat:
            if verbose:
                print(fermat_msg)
            for r in [a - b, a + b]:
                facs = factorint(r, limit=limit, use_trial=use_trial,
                                 use_rho=use_rho, use_pm1=use_pm1,
                                 verbose=verbose)
                for k, v in facs.items():
                    factors[k] = factors.get(k, 0) + v
                factor_cache.add(n, facs)
            if verbose:
                print(complete_msg)
            return factors
        b2 += (a + 1) << 2  # equiv to (a + 2)**2 - n
        a += 2

    # these are the limits for trial division which will
    # be attempted in parallel with pollard methods
    low, high = next_p, 2*next_p

    # add 1 to make sure limit is reached in primerange calls
    _limit = (limit or sqrt_n) + 1
    iteration = 0
    while 1:
        high_ = min(high, _limit)

        # Trial division
        if use_trial:
            if verbose:
                print(trial_msg % (low, high_))
            ps = sieve.primerange(low, high_)
            n, found_trial = _trial(factors, n, ps, verbose)
            next_p = high_
            if found_trial and _check_termination(factors, n, limit, use_trial,
                                                  use_rho, use_pm1, verbose, next_p):
                return factors
        else:
            found_trial = False

        if high > _limit:
            if verbose:
                print('Exceeded limit:', _limit)
            if n > 1:
                factors[int(n)] = 1
            if verbose:
                print(complete_msg)
            return factors

        # Only used advanced methods when no small factors were found
        if not found_trial:
            # Pollard p-1
            if use_pm1:
                if verbose:
                    print(pm1_msg % (low, high_))
                c = pollard_pm1(n, B=low, seed=high_)
                if c:
                    if c < next_p**2 or isprime(c):
                        ps = [c]
                    else:
                        ps = factorint(c, limit=limit,
                                       use_trial=use_trial,
                                       use_rho=use_rho,
                                       use_pm1=use_pm1,
                                       use_ecm=use_ecm,
                                       verbose=verbose)
                    n, _ = _trial(factors, n, ps, verbose=False)
                    if _check_termination(factors, n, limit, use_trial,
                                          use_rho, use_pm1, verbose, next_p):
                        return factors

            # Pollard rho
            if use_rho:
                if verbose:
                    print(rho_msg % (1, low, high_))
                c = pollard_rho(n, retries=1, max_steps=low, seed=high_)
                if c:
                    if c < next_p**2 or isprime(c):
                        ps = [c]
                    else:
                        ps = factorint(c, limit=limit,
                                       use_trial=use_trial,
                                       use_rho=use_rho,
                                       use_pm1=use_pm1,
                                       use_ecm=use_ecm,
                                       verbose=verbose)
                    n, _ = _trial(factors, n, ps, verbose=False)
                    if _check_termination(factors, n, limit, use_trial,
                                          use_rho, use_pm1, verbose, next_p):
                        return factors
        # Use subexponential algorithms if use_ecm
        # Use pollard algorithms for finding small factors for 3 iterations
        # if after small factors the number of digits of n >= 25 then use ecm
        iteration += 1
        if use_ecm and iteration >= 3 and num_digits(n) >= 24:
            break
        low, high = high, high*2

    B1 = 10000
    B2 = 100*B1
    num_curves = 50
    while(1):
        if verbose:
            print(ecm_msg % (B1, B2, num_curves))
        factor = _ecm_one_factor(n, B1, B2, num_curves, seed=B1)
        if factor:
            if factor < next_p**2 or isprime(factor):
                ps = [factor]
            else:
                ps = factorint(factor, limit=limit,
                           use_trial=use_trial,
                           use_rho=use_rho,
                           use_pm1=use_pm1,
                           use_ecm=use_ecm,
                           verbose=verbose)
            n, _ = _trial(factors, n, ps, verbose=False)
            if _check_termination(factors, n, limit, use_trial,
                                  use_rho, use_pm1, verbose, next_p):
                return factors
        B1 *= 5
        B2 = 100*B1
        num_curves *= 4


def factorrat(rat, limit=None, use_trial=True, use_rho=True, use_pm1=True,
              verbose=False, visual=None, multiple=False):
    r"""
    Given a Rational ``r``, ``factorrat(r)`` returns a dict containing
    the prime factors of ``r`` as keys and their respective multiplicities
    as values. For example:

    >>> from sympy import factorrat, S
    >>> factorrat(S(8)/9)    # 8/9 = (2**3) * (3**-2)
    {2: 3, 3: -2}
    >>> factorrat(S(-1)/987)    # -1/789 = -1 * (3**-1) * (7**-1) * (47**-1)
    {-1: 1, 3: -1, 7: -1, 47: -1}

    Please see the docstring for ``factorint`` for detailed explanations
    and examples of the following keywords:

        - ``limit``: Integer limit up to which trial division is done
        - ``use_trial``: Toggle use of trial division
        - ``use_rho``: Toggle use of Pollard's rho method
        - ``use_pm1``: Toggle use of Pollard's p-1 method
        - ``verbose``: Toggle detailed printing of progress
        - ``multiple``: Toggle returning a list of factors or dict
        - ``visual``: Toggle product form of output
    """
    if multiple:
        fac = factorrat(rat, limit=limit, use_trial=use_trial,
                  use_rho=use_rho, use_pm1=use_pm1,
                  verbose=verbose, visual=False, multiple=False)
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One/p]*(-fac[p])
                               for p, _ in sorted(fac.items(),
                                                        key=lambda elem: elem[0]
                                                        if elem[1] > 0
                                                        else 1/elem[0])), [])
        return factorlist

    f = factorint(rat.p, limit=limit, use_trial=use_trial,
                  use_rho=use_rho, use_pm1=use_pm1,
                  verbose=verbose).copy()
    f = defaultdict(int, f)
    for p, e in factorint(rat.q, limit=limit,
                          use_trial=use_trial,
                          use_rho=use_rho,
                          use_pm1=use_pm1,
                          verbose=verbose).items():
        f[p] += -e

    if len(f) > 1 and 1 in f:
        del f[1]
    if not visual:
        return dict(f)
    else:
        if -1 in f:
            f.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        args.extend([Pow(*i, evaluate=False)
                     for i in sorted(f.items())])
        return Mul(*args, evaluate=False)


def primefactors(n, limit=None, verbose=False, **kwargs):
    """Return a sorted list of n's prime factors, ignoring multiplicity
    and any composite factor that remains if the limit was set too low
    for complete factorization. Unlike factorint(), primefactors() does
    not return -1 or 0.

    Parameters
    ==========

    n : integer
    limit, verbose, **kwargs :
        Additional keyword arguments to be passed to ``factorint``.
        Since ``kwargs`` is new in version 1.13,
        ``limit`` and ``verbose`` are retained for compatibility purposes.

    Returns
    =======

    list(int) : List of prime numbers dividing ``n``

    Examples
    ========

    >>> from sympy.ntheory import primefactors, factorint, isprime
    >>> primefactors(6)
    [2, 3]
    >>> primefactors(-5)
    [5]

    >>> sorted(factorint(123456).items())
    [(2, 6), (3, 1), (643, 1)]
    >>> primefactors(123456)
    [2, 3, 643]

    >>> sorted(factorint(10000000001, limit=200).items())
    [(101, 1), (99009901, 1)]
    >>> isprime(99009901)
    False
    >>> primefactors(10000000001, limit=300)
    [101]

    See Also
    ========

    factorint, divisors

    """
    n = int(n)
    kwargs.update({"visual": None, "multiple": False,
                   "limit": limit, "verbose": verbose})
    factors = sorted(factorint(n=n, **kwargs).keys())
    # We want to calculate
    # s = [f for f in factors if isprime(f)]
    s = [f for f in factors[:-1:] if f not in [-1, 0, 1]]
    if factors and isprime(factors[-1]):
        s += [factors[-1]]
    return s


def _divisors(n, proper=False):
    """Helper function for divisors which generates the divisors.

    Parameters
    ==========

    n : int
        a nonnegative integer
    proper: bool
        If `True`, returns the generator that outputs only the proper divisor (i.e., excluding n).

    """
    if n <= 1:
        if not proper and n:
            yield 1
        return

    factordict = factorint(n)
    ps = sorted(factordict.keys())

    def rec_gen(n=0):
        if n == len(ps):
            yield 1
        else:
            pows = [1]
            for _ in range(factordict[ps[n]]):
                pows.append(pows[-1] * ps[n])
            yield from (p * q for q in rec_gen(n + 1) for p in pows)

    if proper:
        yield from (p for p in rec_gen() if p != n)
    else:
        yield from rec_gen()


def divisors(n, generator=False, proper=False):
    r"""
    Return all divisors of n sorted from 1..n by default.
    If generator is ``True`` an unordered generator is returned.

    The number of divisors of n can be quite large if there are many
    prime factors (counting repeated factors). If only the number of
    factors is desired use divisor_count(n).

    Examples
    ========

    >>> from sympy import divisors, divisor_count
    >>> divisors(24)
    [1, 2, 3, 4, 6, 8, 12, 24]
    >>> divisor_count(24)
    8

    >>> list(divisors(120, generator=True))
    [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 15, 30, 60, 120]

    Notes
    =====

    This is a slightly modified version of Tim Peters referenced at:
    https://stackoverflow.com/questions/1010381/python-factorization

    See Also
    ========

    primefactors, factorint, divisor_count
    """
    rv = _divisors(as_int(abs(n)), proper)
    return rv if generator else sorted(rv)


def divisor_count(n, modulus=1, proper=False):
    """
    Return the number of divisors of ``n``. If ``modulus`` is not 1 then only
    those that are divisible by ``modulus`` are counted. If ``proper`` is True
    then the divisor of ``n`` will not be counted.

    Examples
    ========

    >>> from sympy import divisor_count
    >>> divisor_count(6)
    4
    >>> divisor_count(6, 2)
    2
    >>> divisor_count(6, proper=True)
    3

    See Also
    ========

    factorint, divisors, totient, proper_divisor_count

    """

    if not modulus:
        return 0
    elif modulus != 1:
        n, r = divmod(n, modulus)
        if r:
            return 0
    if n == 0:
        return 0
    n = Mul(*[v + 1 for k, v in factorint(n).items() if k > 1])
    if n and proper:
        n -= 1
    return n


def proper_divisors(n, generator=False):
    """
    Return all divisors of n except n, sorted by default.
    If generator is ``True`` an unordered generator is returned.

    Examples
    ========

    >>> from sympy import proper_divisors, proper_divisor_count
    >>> proper_divisors(24)
    [1, 2, 3, 4, 6, 8, 12]
    >>> proper_divisor_count(24)
    7
    >>> list(proper_divisors(120, generator=True))
    [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 15, 30, 60]

    See Also
    ========

    factorint, divisors, proper_divisor_count

    """
    return divisors(n, generator=generator, proper=True)


def proper_divisor_count(n, modulus=1):
    """
    Return the number of proper divisors of ``n``.

    Examples
    ========

    >>> from sympy import proper_divisor_count
    >>> proper_divisor_count(6)
    3
    >>> proper_divisor_count(6, modulus=2)
    1

    See Also
    ========

    divisors, proper_divisors, divisor_count

    """
    return divisor_count(n, modulus=modulus, proper=True)


def _udivisors(n):
    """Helper function for udivisors which generates the unitary divisors.

    Parameters
    ==========

    n : int
        a nonnegative integer

    """
    if n <= 1:
        if n == 1:
            yield 1
        return

    factorpows = [p**e for p, e in factorint(n).items()]
    # We want to calculate
    # yield from (math.prod(s) for s in powersets(factorpows))
    for i in range(2**len(factorpows)):
        d = 1
        for k in range(i.bit_length()):
            if i & 1:
                d *= factorpows[k]
            i >>= 1
        yield d


def udivisors(n, generator=False):
    r"""
    Return all unitary divisors of n sorted from 1..n by default.
    If generator is ``True`` an unordered generator is returned.

    The number of unitary divisors of n can be quite large if there are many
    prime factors. If only the number of unitary divisors is desired use
    udivisor_count(n).

    Examples
    ========

    >>> from sympy.ntheory.factor_ import udivisors, udivisor_count
    >>> udivisors(15)
    [1, 3, 5, 15]
    >>> udivisor_count(15)
    4

    >>> sorted(udivisors(120, generator=True))
    [1, 3, 5, 8, 15, 24, 40, 120]

    See Also
    ========

    primefactors, factorint, divisors, divisor_count, udivisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Unitary_divisor
    .. [2] https://mathworld.wolfram.com/UnitaryDivisor.html

    """
    rv = _udivisors(as_int(abs(n)))
    return rv if generator else sorted(rv)


def udivisor_count(n):
    """
    Return the number of unitary divisors of ``n``.

    Parameters
    ==========

    n : integer

    Examples
    ========

    >>> from sympy.ntheory.factor_ import udivisor_count
    >>> udivisor_count(120)
    8

    See Also
    ========

    factorint, divisors, udivisors, divisor_count, totient

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """

    if n == 0:
        return 0
    return 2**len([p for p in factorint(n) if p > 1])


def _antidivisors(n):
    """Helper function for antidivisors which generates the antidivisors.

    Parameters
    ==========

    n : int
        a nonnegative integer

    """
    if n <= 2:
        return
    for d in _divisors(n):
        y = 2*d
        if n > y and n % y:
            yield y
    for d in _divisors(2*n-1):
        if n > d >= 2 and n % d:
            yield d
    for d in _divisors(2*n+1):
        if n > d >= 2 and n % d:
            yield d


def antidivisors(n, generator=False):
    r"""
    Return all antidivisors of n sorted from 1..n by default.

    Antidivisors [1]_ of n are numbers that do not divide n by the largest
    possible margin.  If generator is True an unordered generator is returned.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import antidivisors
    >>> antidivisors(24)
    [7, 16]

    >>> sorted(antidivisors(128, generator=True))
    [3, 5, 15, 17, 51, 85]

    See Also
    ========

    primefactors, factorint, divisors, divisor_count, antidivisor_count

    References
    ==========

    .. [1] definition is described in https://oeis.org/A066272/a066272a.html

    """
    rv = _antidivisors(as_int(abs(n)))
    return rv if generator else sorted(rv)


def antidivisor_count(n):
    """
    Return the number of antidivisors [1]_ of ``n``.

    Parameters
    ==========

    n : integer

    Examples
    ========

    >>> from sympy.ntheory.factor_ import antidivisor_count
    >>> antidivisor_count(13)
    4
    >>> antidivisor_count(27)
    5

    See Also
    ========

    factorint, divisors, antidivisors, divisor_count, totient

    References
    ==========

    .. [1] formula from https://oeis.org/A066272

    """

    n = as_int(abs(n))
    if n <= 2:
        return 0
    return divisor_count(2*n - 1) + divisor_count(2*n + 1) + \
        divisor_count(n) - divisor_count(n, 2) - 5

@deprecated("""\
The `sympy.ntheory.factor_.totient` has been moved to `sympy.functions.combinatorial.numbers.totient`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def totient(n):
    r"""
    Calculate the Euler totient function phi(n)

    .. deprecated:: 1.13

        The ``totient`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.totient`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``totient(n)`` or `\phi(n)` is the number of positive integers `\leq` n
    that are relatively prime to n.

    Parameters
    ==========

    n : integer

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> totient(1)
    1
    >>> totient(25)
    20
    >>> totient(45) == totient(5)*totient(9)
    True

    See Also
    ========

    divisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%27s_totient_function
    .. [2] https://mathworld.wolfram.com/TotientFunction.html

    """
    from sympy.functions.combinatorial.numbers import totient as _totient
    return _totient(n)


@deprecated("""\
The `sympy.ntheory.factor_.reduced_totient` has been moved to `sympy.functions.combinatorial.numbers.reduced_totient`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def reduced_totient(n):
    r"""
    Calculate the Carmichael reduced totient function lambda(n)

    .. deprecated:: 1.13

        The ``reduced_totient`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.reduced_totient`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``reduced_totient(n)`` or `\lambda(n)` is the smallest m > 0 such that
    `k^m \equiv 1 \mod n` for all k relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import reduced_totient
    >>> reduced_totient(1)
    1
    >>> reduced_totient(8)
    2
    >>> reduced_totient(30)
    4

    See Also
    ========

    totient

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_function
    .. [2] https://mathworld.wolfram.com/CarmichaelFunction.html

    """
    from sympy.functions.combinatorial.numbers import reduced_totient as _reduced_totient
    return _reduced_totient(n)


@deprecated("""\
The `sympy.ntheory.factor_.divisor_sigma` has been moved to `sympy.functions.combinatorial.numbers.divisor_sigma`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def divisor_sigma(n, k=1):
    r"""
    Calculate the divisor function `\sigma_k(n)` for positive integer n

    .. deprecated:: 1.13

        The ``divisor_sigma`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.divisor_sigma`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``divisor_sigma(n, k)`` is equal to ``sum([x**k for x in divisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k(n) = \prod_{i=1}^\omega (1+p_i^k+p_i^{2k}+\cdots
        + p_i^{m_ik}).

    Parameters
    ==========

    n : integer

    k : integer, optional
        power of divisors in the sum

        for k = 0, 1:
        ``divisor_sigma(n, 0)`` is equal to ``divisor_count(n)``
        ``divisor_sigma(n, 1)`` is equal to ``sum(divisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> divisor_sigma(18, 0)
    6
    >>> divisor_sigma(39, 1)
    56
    >>> divisor_sigma(12, 2)
    210
    >>> divisor_sigma(37)
    38

    See Also
    ========

    divisor_count, totient, divisors, factorint

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Divisor_function

    """
    from sympy.functions.combinatorial.numbers import divisor_sigma as func_divisor_sigma
    return func_divisor_sigma(n, k)


def _divisor_sigma(n:int, k:int=1) -> int:
    r""" Calculate the divisor function `\sigma_k(n)` for positive integer n

    Parameters
    ==========

    n : int
        positive integer
    k : int
        nonnegative integer

    See Also
    ========

    sympy.functions.combinatorial.numbers.divisor_sigma

    """
    if k == 0:
        return math.prod(e + 1 for e in factorint(n).values())
    return math.prod((p**(k*(e + 1)) - 1)//(p**k - 1) for p, e in factorint(n).items())


def core(n, t=2):
    r"""
    Calculate core(n, t) = `core_t(n)` of a positive integer n

    ``core_2(n)`` is equal to the squarefree part of n

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        core_t(n) = \prod_{i=1}^\omega p_i^{m_i \mod t}.

    Parameters
    ==========

    n : integer

    t : integer
        core(n, t) calculates the t-th power free part of n

        ``core(n, 2)`` is the squarefree part of ``n``
        ``core(n, 3)`` is the cubefree part of ``n``

        Default for t is 2.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import core
    >>> core(24, 2)
    6
    >>> core(9424, 3)
    1178
    >>> core(379238)
    379238
    >>> core(15**11, 10)
    15

    See Also
    ========

    factorint, sympy.solvers.diophantine.diophantine.square_factor

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square-free_integer#Squarefree_core

    """

    n = as_int(n)
    t = as_int(t)
    if n <= 0:
        raise ValueError("n must be a positive integer")
    elif t <= 1:
        raise ValueError("t must be >= 2")
    else:
        y = 1
        for p, e in factorint(n).items():
            y *= p**(e % t)
        return y


@deprecated("""\
The `sympy.ntheory.factor_.udivisor_sigma` has been moved to `sympy.functions.combinatorial.numbers.udivisor_sigma`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def udivisor_sigma(n, k=1):
    r"""
    Calculate the unitary divisor function `\sigma_k^*(n)` for positive integer n

    .. deprecated:: 1.13

        The ``udivisor_sigma`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.udivisor_sigma`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``udivisor_sigma(n, k)`` is equal to ``sum([x**k for x in udivisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k^*(n) = \prod_{i=1}^\omega (1+ p_i^{m_ik}).

    Parameters
    ==========

    k : power of divisors in the sum

        for k = 0, 1:
        ``udivisor_sigma(n, 0)`` is equal to ``udivisor_count(n)``
        ``udivisor_sigma(n, 1)`` is equal to ``sum(udivisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import udivisor_sigma
    >>> udivisor_sigma(18, 0)
    4
    >>> udivisor_sigma(74, 1)
    114
    >>> udivisor_sigma(36, 3)
    47450
    >>> udivisor_sigma(111)
    152

    See Also
    ========

    divisor_count, totient, divisors, udivisors, udivisor_count, divisor_sigma,
    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """
    from sympy.functions.combinatorial.numbers import udivisor_sigma as _udivisor_sigma
    return _udivisor_sigma(n, k)


@deprecated("""\
The `sympy.ntheory.factor_.primenu` has been moved to `sympy.functions.combinatorial.numbers.primenu`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def primenu(n):
    r"""
    Calculate the number of distinct prime factors for a positive integer n.

    .. deprecated:: 1.13

        The ``primenu`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.primenu`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primenu(n)`` or `\nu(n)` is:

    .. math ::
        \nu(n) = k.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primenu
    >>> primenu(1)
    0
    >>> primenu(30)
    3

    See Also
    ========

    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html

    """
    from sympy.functions.combinatorial.numbers import primenu as _primenu
    return _primenu(n)


@deprecated("""\
The `sympy.ntheory.factor_.primeomega` has been moved to `sympy.functions.combinatorial.numbers.primeomega`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def primeomega(n):
    r"""
    Calculate the number of prime factors counting multiplicities for a
    positive integer n.

    .. deprecated:: 1.13

        The ``primeomega`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.primeomega`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primeomega(n)``  or `\Omega(n)` is:

    .. math ::
        \Omega(n) = \sum_{i=1}^k m_i.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primeomega
    >>> primeomega(1)
    0
    >>> primeomega(20)
    3

    See Also
    ========

    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html

    """
    from sympy.functions.combinatorial.numbers import primeomega as _primeomega
    return _primeomega(n)


def mersenne_prime_exponent(nth):
    """Returns the exponent ``i`` for the nth Mersenne prime (which
    has the form `2^i - 1`).

    Examples
    ========

    >>> from sympy.ntheory.factor_ import mersenne_prime_exponent
    >>> mersenne_prime_exponent(1)
    2
    >>> mersenne_prime_exponent(20)
    4423
    """
    n = as_int(nth)
    if n < 1:
        raise ValueError("nth must be a positive integer; mersenne_prime_exponent(1) == 2")
    if n > 51:
        raise ValueError("There are only 51 perfect numbers; nth must be less than or equal to 51")
    return MERSENNE_PRIME_EXPONENTS[n - 1]


def is_perfect(n):
    """Returns True if ``n`` is a perfect number, else False.

    A perfect number is equal to the sum of its positive, proper divisors.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> from sympy.ntheory.factor_ import is_perfect, divisors
    >>> is_perfect(20)
    False
    >>> is_perfect(6)
    True
    >>> 6 == divisor_sigma(6) - 6 == sum(divisors(6)[:-1])
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PerfectNumber.html
    .. [2] https://en.wikipedia.org/wiki/Perfect_number

    """
    n = as_int(n)
    if n < 1:
        return False
    if n % 2 == 0:
        m = (n.bit_length() + 1) >> 1
        if (1 << (m - 1)) * ((1 << m) - 1) != n:
            # Even perfect numbers must be of the form `2^{m-1}(2^m-1)`
            return False
        return m in MERSENNE_PRIME_EXPONENTS or is_mersenne_prime(2**m - 1)

    # n is an odd integer
    if n < 10**2000:  # https://www.lirmm.fr/~ochem/opn/
        return False
    if n % 105 == 0:  # not divis by 105
        return False
    if all(n % m != r for m, r in [(12, 1), (468, 117), (324, 81)]):
        return False
    # there are many criteria that the factor structure of n
    # must meet; since we will have to factor it to test the
    # structure we will have the factors and can then check
    # to see whether it is a perfect number or not. So we
    # skip the structure checks and go straight to the final
    # test below.
    result = abundance(n) == 0
    if result:
        raise ValueError(filldedent('''In 1888, Sylvester stated: "
            ...a prolonged meditation on the subject has satisfied
            me that the existence of any one such [odd perfect number]
            -- its escape, so to say, from the complex web of conditions
            which hem it in on all sides -- would be little short of a
            miracle." I guess SymPy just found that miracle and it
            factors like this: %s''' % factorint(n)))
    return result


def abundance(n):
    """Returns the difference between the sum of the positive
    proper divisors of a number and the number.

    Examples
    ========

    >>> from sympy.ntheory import abundance, is_perfect, is_abundant
    >>> abundance(6)
    0
    >>> is_perfect(6)
    True
    >>> abundance(10)
    -2
    >>> is_abundant(10)
    False
    """
    return _divisor_sigma(n) - 2 * n


def is_abundant(n):
    """Returns True if ``n`` is an abundant number, else False.

    A abundant number is smaller than the sum of its positive proper divisors.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import is_abundant
    >>> is_abundant(20)
    True
    >>> is_abundant(15)
    False

    References
    ==========

    .. [1] https://mathworld.wolfram.com/AbundantNumber.html

    """
    n = as_int(n)
    if is_perfect(n):
        return False
    return n % 6 == 0 or bool(abundance(n) > 0)


def is_deficient(n):
    """Returns True if ``n`` is a deficient number, else False.

    A deficient number is greater than the sum of its positive proper divisors.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import is_deficient
    >>> is_deficient(20)
    False
    >>> is_deficient(15)
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/DeficientNumber.html

    """
    n = as_int(n)
    if is_perfect(n):
        return False
    return bool(abundance(n) < 0)


def is_amicable(m, n):
    """Returns True if the numbers `m` and `n` are "amicable", else False.

    Amicable numbers are two different numbers so related that the sum
    of the proper divisors of each is equal to that of the other.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> from sympy.ntheory.factor_ import is_amicable
    >>> is_amicable(220, 284)
    True
    >>> divisor_sigma(220) == divisor_sigma(284)
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Amicable_numbers

    """
    return m != n and m + n == _divisor_sigma(m) == _divisor_sigma(n)


def is_carmichael(n):
    """ Returns True if the numbers `n` is Carmichael number, else False.

    Parameters
    ==========

    n : Integer

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_number
    .. [2] https://oeis.org/A002997

    """
    if n < 561:
        return False
    return n % 2 and not isprime(n) and \
           all(e == 1 and (n - 1) % (p - 1) == 0 for p, e in factorint(n).items())


def find_carmichael_numbers_in_range(x, y):
    """ Returns a list of the number of Carmichael in the range

    See Also
    ========

    is_carmichael

    """
    if 0 <= x <= y:
        if x % 2 == 0:
            return [i for i in range(x + 1, y, 2) if is_carmichael(i)]
        else:
            return [i for i in range(x, y, 2) if is_carmichael(i)]
    else:
        raise ValueError('The provided range is not valid. x and y must be non-negative integers and x <= y')


def find_first_n_carmichaels(n):
    """ Returns the first n Carmichael numbers.

    Parameters
    ==========

    n : Integer

    See Also
    ========

    is_carmichael

    """
    i = 561
    carmichaels = []

    while len(carmichaels) < n:
        if is_carmichael(i):
            carmichaels.append(i)
        i += 2

    return carmichaels


def dra(n, b):
    """
    Returns the additive digital root of a natural number ``n`` in base ``b``
    which is a single digit value obtained by an iterative process of summing
    digits, on each iteration using the result from the previous iteration to
    compute a digit sum.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import dra
    >>> dra(3110, 12)
    8

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Digital_root

    """

    num = abs(as_int(n))
    b = as_int(b)
    if b <= 1:
        raise ValueError("Base should be an integer greater than 1")

    if num == 0:
        return 0

    return (1 + (num - 1) % (b - 1))


def drm(n, b):
    """
    Returns the multiplicative digital root of a natural number ``n`` in a given
    base ``b`` which is a single digit value obtained by an iterative process of
    multiplying digits, on each iteration using the result from the previous
    iteration to compute the digit multiplication.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import drm
    >>> drm(9876, 10)
    0

    >>> drm(49, 10)
    8

    References
    ==========

    .. [1] https://mathworld.wolfram.com/MultiplicativeDigitalRoot.html

    """

    n = abs(as_int(n))
    b = as_int(b)
    if b <= 1:
        raise ValueError("Base should be an integer greater than 1")
    while n > b:
        mul = 1
        while n > 1:
            n, r = divmod(n, b)
            if r == 0:
                return 0
            mul *= r
        n = mul
    return n
