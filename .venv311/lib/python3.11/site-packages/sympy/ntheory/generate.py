"""
Generating and counting primes.

"""

from bisect import bisect, bisect_left
from itertools import count
# Using arrays for sieving instead of lists greatly reduces
# memory consumption
from array import array as _array

from sympy.core.random import randint
from sympy.external.gmpy import sqrt
from .primetest import isprime
from sympy.utilities.decorator import deprecated
from sympy.utilities.misc import as_int


def _as_int_ceiling(a):
    """ Wrapping ceiling in as_int will raise an error if there was a problem
        determining whether the expression was exactly an integer or not."""
    from sympy.functions.elementary.integers import ceiling
    return as_int(ceiling(a))


class Sieve:
    """A list of prime numbers, implemented as a dynamically
    growing sieve of Eratosthenes. When a lookup is requested involving
    an odd number that has not been sieved, the sieve is automatically
    extended up to that number. Implementation details limit the number of
    primes to ``2^32-1``.

    Examples
    ========

    >>> from sympy import sieve
    >>> sieve._reset() # this line for doctest only
    >>> 25 in sieve
    False
    >>> sieve._list
    array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])
    """

    # data shared (and updated) by all Sieve instances
    def __init__(self, sieve_interval=1_000_000):
        """ Initial parameters for the Sieve class.

        Parameters
        ==========

        sieve_interval (int): Amount of memory to be used

        Raises
        ======

        ValueError
            If ``sieve_interval`` is not positive.

        """
        self._n = 6
        self._list = _array('L', [2, 3, 5, 7, 11, 13]) # primes
        self._tlist = _array('L', [0, 1, 1, 2, 2, 4]) # totient
        self._mlist = _array('i', [0, 1, -1, -1, 0, -1]) # mobius
        if sieve_interval <= 0:
            raise ValueError("sieve_interval should be a positive integer")
        self.sieve_interval = sieve_interval
        assert all(len(i) == self._n for i in (self._list, self._tlist, self._mlist))

    def __repr__(self):
        return ("<%s sieve (%i): %i, %i, %i, ... %i, %i\n"
             "%s sieve (%i): %i, %i, %i, ... %i, %i\n"
             "%s sieve (%i): %i, %i, %i, ... %i, %i>") % (
             'prime', len(self._list),
                 self._list[0], self._list[1], self._list[2],
                 self._list[-2], self._list[-1],
             'totient', len(self._tlist),
                 self._tlist[0], self._tlist[1],
                 self._tlist[2], self._tlist[-2], self._tlist[-1],
             'mobius', len(self._mlist),
                 self._mlist[0], self._mlist[1],
                 self._mlist[2], self._mlist[-2], self._mlist[-1])

    def _reset(self, prime=None, totient=None, mobius=None):
        """Reset all caches (default). To reset one or more set the
            desired keyword to True."""
        if all(i is None for i in (prime, totient, mobius)):
            prime = totient = mobius = True
        if prime:
            self._list = self._list[:self._n]
        if totient:
            self._tlist = self._tlist[:self._n]
        if mobius:
            self._mlist = self._mlist[:self._n]

    def extend(self, n):
        """Grow the sieve to cover all primes <= n.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend(30)
        >>> sieve[10] == 29
        True
        """
        n = int(n)
        # `num` is even at any point in the function.
        # This satisfies the condition required by `self._primerange`.
        num = self._list[-1] + 1
        if n < num:
            return
        num2 = num**2
        while num2 <= n:
            self._list += _array('L', self._primerange(num, num2))
            num, num2 = num2, num2**2
        # Merge the sieves
        self._list += _array('L', self._primerange(num, n + 1))

    def _primerange(self, a, b):
        """ Generate all prime numbers in the range (a, b).

        Parameters
        ==========

        a, b : positive integers assuming the following conditions
                * a is an even number
                * 2 < self._list[-1] < a < b < nextprime(self._list[-1])**2

        Yields
        ======

        p (int): prime numbers such that ``a < p < b``

        Examples
        ========

        >>> from sympy.ntheory.generate import Sieve
        >>> s = Sieve()
        >>> s._list[-1]
        13
        >>> list(s._primerange(18, 31))
        [19, 23, 29]

        """
        if b % 2:
            b -= 1
        while a < b:
            block_size = min(self.sieve_interval, (b - a) // 2)
            # Create the list such that block[x] iff (a + 2x + 1) is prime.
            # Note that even numbers are not considered here.
            block = [True] * block_size
            for p in self._list[1:bisect(self._list, sqrt(a + 2 * block_size + 1))]:
                for t in range((-(a + 1 + p) // 2) % p, block_size, p):
                    block[t] = False
            for idx, p in enumerate(block):
                if p:
                    yield a + 2 * idx + 1
            a += 2 * block_size

    def extend_to_no(self, i):
        """Extend to include the ith prime number.

        Parameters
        ==========

        i : integer

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend_to_no(9)
        >>> sieve._list
        array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])

        Notes
        =====

        The list is extended by 50% if it is too short, so it is
        likely that it will be longer than requested.
        """
        i = as_int(i)
        while len(self._list) < i:
            self.extend(int(self._list[-1] * 1.5))

    def primerange(self, a, b=None):
        """Generate all prime numbers in the range [2, a) or [a, b).

        Examples
        ========

        >>> from sympy import sieve, prime

        All primes less than 19:

        >>> print([i for i in sieve.primerange(19)])
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> print([i for i in sieve.primerange(7, 19)])
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(sieve.primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        """
        if b is None:
            b = _as_int_ceiling(a)
            a = 2
        else:
            a = max(2, _as_int_ceiling(a))
            b = _as_int_ceiling(b)
        if a >= b:
            return
        self.extend(b)
        yield from self._list[bisect_left(self._list, a):
                              bisect_left(self._list, b)]

    def totientrange(self, a, b):
        """Generate all totient numbers for the range [a, b).

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.totientrange(7, 18)])
        [6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16]
        """
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._tlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._tlist[i]
        else:
            self._tlist += _array('L', range(n, b))
            for i in range(1, n):
                ti = self._tlist[i]
                if ti == i - 1:
                    startindex = (n + i - 1) // i * i
                    for j in range(startindex, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield ti

            for i in range(n, b):
                ti = self._tlist[i]
                if ti == i:
                    for j in range(i, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield self._tlist[i]

    def mobiusrange(self, a, b):
        """Generate all mobius numbers for the range [a, b).

        Parameters
        ==========

        a : integer
            First number in range

        b : integer
            First number outside of range

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.mobiusrange(7, 18)])
        [-1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1]
        """
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._mlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._mlist[i]
        else:
            self._mlist += _array('i', [0]*(b - n))
            for i in range(1, n):
                mi = self._mlist[i]
                startindex = (n + i - 1) // i * i
                for j in range(startindex, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

            for i in range(n, b):
                mi = self._mlist[i]
                for j in range(2 * i, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

    def search(self, n):
        """Return the indices i, j of the primes that bound n.

        If n is prime then i == j.

        Although n can be an expression, if ceiling cannot convert
        it to an integer then an n error will be raised.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve.search(25)
        (9, 10)
        >>> sieve.search(23)
        (9, 9)
        """
        test = _as_int_ceiling(n)
        n = as_int(n)
        if n < 2:
            raise ValueError("n should be >= 2 but got: %s" % n)
        if n > self._list[-1]:
            self.extend(n)
        b = bisect(self._list, n)
        if self._list[b - 1] == test:
            return b, b
        else:
            return b, b + 1

    def __contains__(self, n):
        try:
            n = as_int(n)
            assert n >= 2
        except (ValueError, AssertionError):
            return False
        if n % 2 == 0:
            return n == 2
        a, b = self.search(n)
        return a == b

    def __iter__(self):
        for n in count(1):
            yield self[n]

    def __getitem__(self, n):
        """Return the nth prime number"""
        if isinstance(n, slice):
            self.extend_to_no(n.stop)
            start = n.start if n.start is not None else 0
            if start < 1:
                # sieve[:5] would be empty (starting at -1), let's
                # just be explicit and raise.
                raise IndexError("Sieve indices start at 1.")
            return self._list[start - 1:n.stop - 1:n.step]
        else:
            if n < 1:
                # offset is one, so forbid explicit access to sieve[0]
                # (would surprisingly return the last one).
                raise IndexError("Sieve indices start at 1.")
            n = as_int(n)
            self.extend_to_no(n)
            return self._list[n - 1]

# Generate a global object for repeated use in trial division etc
sieve = Sieve()

def prime(nth):
    r"""
    Return the nth prime number, where primes are indexed starting from 1:
    prime(1) = 2, prime(2) = 3, etc.

    Parameters
    ==========

    nth : int
        The position of the prime number to return (must be a positive integer).

    Returns
    =======

    int
        The nth prime number.

    Examples
    ========

    >>> from sympy import prime
    >>> prime(10)
    29
    >>> prime(1)
    2
    >>> prime(100000)
    1299709

    See Also
    ========

    sympy.ntheory.primetest.isprime : Test if a number is prime.
    primerange : Generate all primes in a given range.
    primepi : Return the number of primes less than or equal to a given number.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Prime_number_theorem
    .. [2] https://en.wikipedia.org/wiki/Logarithmic_integral_function
    .. [3] https://en.wikipedia.org/wiki/Skewes%27_number
    """
    n = as_int(nth)
    if n < 1:
        raise ValueError("nth must be a positive integer; prime(1) == 2")

    # Check if n is within the sieve range
    if n <= len(sieve._list):
        return sieve[n]

    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li

    if n < 1000:
        # Extend sieve up to 8*n as this is empirically sufficient
        sieve.extend(8 * n)
        return sieve[n]

    a = 2
    # Estimate an upper bound for the nth prime using the prime number theorem
    b = int(n * (log(n).evalf() + log(log(n)).evalf()))

    # Binary search for the least m such that li(m) > n
    while a < b:
        mid = (a + b) >> 1
        if li(mid).evalf() > n:
            b = mid
        else:
            a = mid + 1

    return nextprime(a - 1, n - _primepi(a - 1))


@deprecated("""\
The `sympy.ntheory.generate.primepi` has been moved to `sympy.functions.combinatorial.numbers.primepi`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def primepi(n):
    r""" Represents the prime counting function pi(n) = the number
        of prime numbers less than or equal to n.

        .. deprecated:: 1.13

            The ``primepi`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.primepi`
            instead. See its documentation for more information. See
            :ref:`deprecated-ntheory-symbolic-functions` for details.

        Algorithm Description:

        In sieve method, we remove all multiples of prime p
        except p itself.

        Let phi(i,j) be the number of integers 2 <= k <= i
        which remain after sieving from primes less than
        or equal to j.
        Clearly, pi(n) = phi(n, sqrt(n))

        If j is not a prime,
        phi(i,j) = phi(i, j - 1)

        if j is a prime,
        We remove all numbers(except j) whose
        smallest prime factor is j.

        Let $x= j \times a$ be such a number, where $2 \le a \le i / j$
        Now, after sieving from primes $\le j - 1$,
        a must remain
        (because x, and hence a has no prime factor $\le j - 1$)
        Clearly, there are phi(i / j, j - 1) such a
        which remain on sieving from primes $\le j - 1$

        Now, if a is a prime less than equal to j - 1,
        $x= j \times a$ has smallest prime factor = a, and
        has already been removed(by sieving from a).
        So, we do not need to remove it again.
        (Note: there will be pi(j - 1) such x)

        Thus, number of x, that will be removed are:
        phi(i / j, j - 1) - phi(j - 1, j - 1)
        (Note that pi(j - 1) = phi(j - 1, j - 1))

        $\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

        So,following recursion is used and implemented as dp:

        phi(a, b) = phi(a, b - 1), if b is not a prime
        phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

        Clearly a is always of the form floor(n / k),
        which can take at most $2\sqrt{n}$ values.
        Two arrays arr1,arr2 are maintained
        arr1[i] = phi(i, j),
        arr2[i] = phi(n // i, j)

        Finally the answer is arr2[1]

        Examples
        ========

        >>> from sympy import primepi, prime, prevprime, isprime
        >>> primepi(25)
        9

        So there are 9 primes less than or equal to 25. Is 25 prime?

        >>> isprime(25)
        False

        It is not. So the first prime less than 25 must be the
        9th prime:

        >>> prevprime(25) == prime(9)
        True

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        prime : Return the nth prime
    """
    from sympy.functions.combinatorial.numbers import primepi as func_primepi
    return func_primepi(n)


def _primepi(n:int) -> int:
    r""" Represents the prime counting function pi(n) = the number
    of prime numbers less than or equal to n.

    Explanation
    ===========

    In sieve method, we remove all multiples of prime p
    except p itself.

    Let phi(i,j) be the number of integers 2 <= k <= i
    which remain after sieving from primes less than
    or equal to j.
    Clearly, pi(n) = phi(n, sqrt(n))

    If j is not a prime,
    phi(i,j) = phi(i, j - 1)

    if j is a prime,
    We remove all numbers(except j) whose
    smallest prime factor is j.

    Let $x= j \times a$ be such a number, where $2 \le a \le i / j$
    Now, after sieving from primes $\le j - 1$,
    a must remain
    (because x, and hence a has no prime factor $\le j - 1$)
    Clearly, there are phi(i / j, j - 1) such a
    which remain on sieving from primes $\le j - 1$

    Now, if a is a prime less than equal to j - 1,
    $x= j \times a$ has smallest prime factor = a, and
    has already been removed(by sieving from a).
    So, we do not need to remove it again.
    (Note: there will be pi(j - 1) such x)

    Thus, number of x, that will be removed are:
    phi(i / j, j - 1) - phi(j - 1, j - 1)
    (Note that pi(j - 1) = phi(j - 1, j - 1))

    $\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

    So,following recursion is used and implemented as dp:

    phi(a, b) = phi(a, b - 1), if b is not a prime
    phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

    Clearly a is always of the form floor(n / k),
    which can take at most $2\sqrt{n}$ values.
    Two arrays arr1,arr2 are maintained
    arr1[i] = phi(i, j),
    arr2[i] = phi(n // i, j)

    Finally the answer is arr2[1]

    Parameters
    ==========

    n : int

    """
    if n < 2:
        return 0
    if n <= sieve._list[-1]:
        return sieve.search(n)[0]
    lim = sqrt(n)
    arr1 = [(i + 1) >> 1 for i in range(lim + 1)]
    arr2 = [0] + [(n//i + 1) >> 1 for i in range(1, lim + 1)]
    skip = [False] * (lim + 1)
    for i in range(3, lim + 1, 2):
        # Presently, arr1[k]=phi(k,i - 1),
        # arr2[k] = phi(n // k,i - 1) # not all k's do this
        if skip[i]:
            # skip if i is a composite number
            continue
        p = arr1[i - 1]
        for j in range(i, lim + 1, i):
            skip[j] = True
        # update arr2
        # phi(n/j, i) = phi(n/j, i-1) - phi(n/(i*j), i-1) + phi(i-1, i-1)
        for j in range(1, min(n // (i * i), lim) + 1, 2):
            # No need for arr2[j] in j such that skip[j] is True to
            # compute the final required arr2[1].
            if skip[j]:
                continue
            st = i * j
            if st <= lim:
                arr2[j] -= arr2[st] - p
            else:
                arr2[j] -= arr1[n // st] - p
        # update arr1
        # phi(j, i) = phi(j, i-1) - phi(j/i, i-1) + phi(i-1, i-1)
        # where the range below i**2 is fixed and
        # does not need to be calculated.
        for j in range(lim, min(lim, i*i - 1), -1):
            arr1[j] -= arr1[j // i] - p
    return arr2[1]


def nextprime(n, ith=1):
    """ Return the ith prime greater than n.

        Parameters
        ==========

        n : integer
        ith : positive integer

        Returns
        =======

        int : Return the ith prime greater than n

        Raises
        ======

        ValueError
            If ``ith <= 0``.
            If ``n`` or ``ith`` is not an integer.

        Notes
        =====

        Potential primes are located at 6*j +/- 1. This
        property is used during searching.

        >>> from sympy import nextprime
        >>> [(i, nextprime(i)) for i in range(10, 15)]
        [(10, 11), (11, 13), (12, 13), (13, 17), (14, 17)]
        >>> nextprime(2, ith=2) # the 2nd prime after 2
        5

        See Also
        ========

        prevprime : Return the largest prime smaller than n
        primerange : Generate all primes in a given range

    """
    n = int(n)
    i = as_int(ith)
    if i <= 0:
        raise ValueError("ith should be positive")
    if n < 2:
        n = 2
        i -= 1
    if n <= sieve._list[-2]:
        l, _ = sieve.search(n)
        if l + i - 1 < len(sieve._list):
            return sieve._list[l + i - 1]
        n = sieve._list[-1]
        i += l - len(sieve._list)
    nn = 6*(n//6)
    if nn == n:
        n += 1
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4
    elif n - nn == 5:
        n += 2
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4
    else:
        n = nn + 5
    while 1:
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 2
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4


def prevprime(n):
    """ Return the largest prime smaller than n.

        Notes
        =====

        Potential primes are located at 6*j +/- 1. This
        property is used during searching.

        >>> from sympy import prevprime
        >>> [(i, prevprime(i)) for i in range(10, 15)]
        [(10, 7), (11, 7), (12, 11), (13, 11), (14, 13)]

        See Also
        ========

        nextprime : Return the ith prime greater than n
        primerange : Generates all primes in a given range
    """
    n = _as_int_ceiling(n)
    if n < 3:
        raise ValueError("no preceding primes")
    if n < 8:
        return {3: 2, 4: 3, 5: 3, 6: 5, 7: 5}[n]
    if n <= sieve._list[-1]:
        l, u = sieve.search(n)
        if l == u:
            return sieve[l-1]
        else:
            return sieve[l]
    nn = 6*(n//6)
    if n - nn <= 1:
        n = nn - 1
        if isprime(n):
            return n
        n -= 4
    else:
        n = nn + 1
    while 1:
        if isprime(n):
            return n
        n -= 2
        if isprime(n):
            return n
        n -= 4


def primerange(a, b=None):
    """ Generate a list of all prime numbers in the range [2, a),
        or [a, b).

        If the range exists in the default sieve, the values will
        be returned from there; otherwise values will be returned
        but will not modify the sieve.

        Examples
        ========

        >>> from sympy import primerange, prime

        All primes less than 19:

        >>> list(primerange(19))
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> list(primerange(7, 19))
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        The Sieve method, primerange, is generally faster but it will
        occupy more memory as the sieve stores values. The default
        instance of Sieve, named sieve, can be used:

        >>> from sympy import sieve
        >>> list(sieve.primerange(1, 30))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        Notes
        =====

        Some famous conjectures about the occurrence of primes in a given
        range are [1]:

        - Twin primes: though often not, the following will give 2 primes
                    an infinite number of times:
                        primerange(6*n - 1, 6*n + 2)
        - Legendre's: the following always yields at least one prime
                        primerange(n**2, (n+1)**2+1)
        - Bertrand's (proven): there is always a prime in the range
                        primerange(n, 2*n)
        - Brocard's: there are at least four primes in the range
                        primerange(prime(n)**2, prime(n+1)**2)

        The average gap between primes is log(n) [2]; the gap between
        primes can be arbitrarily large since sequences of composite
        numbers are arbitrarily large, e.g. the numbers in the sequence
        n! + 2, n! + 3 ... n! + n are all composite.

        See Also
        ========

        prime : Return the nth prime
        nextprime : Return the ith prime greater than n
        prevprime : Return the largest prime smaller than n
        randprime : Returns a random prime in a given range
        primorial : Returns the product of primes based on condition
        Sieve.primerange : return range from already computed primes
                           or extend the sieve to contain the requested
                           range.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Prime_number
        .. [2] https://primes.utm.edu/notes/gaps.html
    """
    if b is None:
        a, b = 2, a
    if a >= b:
        return
    # If we already have the range, return it.
    largest_known_prime = sieve._list[-1]
    if b <= largest_known_prime:
        yield from sieve.primerange(a, b)
        return
    # If we know some of it, return it.
    if a <= largest_known_prime:
        yield from sieve._list[bisect_left(sieve._list, a):]
        a = largest_known_prime + 1
    elif a % 2:
        a -= 1
    tail = min(b, (largest_known_prime)**2)
    if a < tail:
        yield from sieve._primerange(a, tail)
        a = tail
    if b <= a:
        return
    # otherwise compute, without storing, the desired range.
    while 1:
        a = nextprime(a)
        if a < b:
            yield a
        else:
            return


def randprime(a, b):
    """ Return a random prime number in the range [a, b).

        Bertrand's postulate assures that
        randprime(a, 2*a) will always succeed for a > 1.

        Note that due to implementation difficulties,
        the prime numbers chosen are not uniformly random.
        For example, there are two primes in the range [112, 128),
        ``113`` and ``127``, but ``randprime(112, 128)`` returns ``127``
        with a probability of 15/17.

        Examples
        ========

        >>> from sympy import randprime, isprime
        >>> randprime(1, 30) #doctest: +SKIP
        13
        >>> isprime(randprime(1, 30))
        True

        See Also
        ========

        primerange : Generate all primes in a given range

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Bertrand's_postulate

    """
    if a >= b:
        return
    a, b = map(int, (a, b))
    n = randint(a - 1, b)
    p = nextprime(n)
    if p >= b:
        p = prevprime(b)
    if p < a:
        raise ValueError("no primes exist in the specified range")
    return p


def primorial(n, nth=True):
    """
    Returns the product of the first n primes (default) or
    the primes less than or equal to n (when ``nth=False``).

    Examples
    ========

    >>> from sympy.ntheory.generate import primorial, primerange
    >>> from sympy import factorint, Mul, primefactors, sqrt
    >>> primorial(4) # the first 4 primes are 2, 3, 5, 7
    210
    >>> primorial(4, nth=False) # primes <= 4 are 2 and 3
    6
    >>> primorial(1)
    2
    >>> primorial(1, nth=False)
    1
    >>> primorial(sqrt(101), nth=False)
    210

    One can argue that the primes are infinite since if you take
    a set of primes and multiply them together (e.g. the primorial) and
    then add or subtract 1, the result cannot be divided by any of the
    original factors, hence either 1 or more new primes must divide this
    product of primes.

    In this case, the number itself is a new prime:

    >>> factorint(primorial(4) + 1)
    {211: 1}

    In this case two new primes are the factors:

    >>> factorint(primorial(4) - 1)
    {11: 1, 19: 1}

    Here, some primes smaller and larger than the primes multiplied together
    are obtained:

    >>> p = list(primerange(10, 20))
    >>> sorted(set(primefactors(Mul(*p) + 1)).difference(set(p)))
    [2, 5, 31, 149]

    See Also
    ========

    primerange : Generate all primes in a given range

    """
    if nth:
        n = as_int(n)
    else:
        n = int(n)
    if n < 1:
        raise ValueError("primorial argument must be >= 1")
    p = 1
    if nth:
        for i in range(1, n + 1):
            p *= prime(i)
    else:
        for i in primerange(2, n + 1):
            p *= i
    return p


def cycle_length(f, x0, nmax=None, values=False):
    """For a given iterated sequence, return a generator that gives
    the length of the iterated cycle (lambda) and the length of terms
    before the cycle begins (mu); if ``values`` is True then the
    terms of the sequence will be returned instead. The sequence is
    started with value ``x0``.

    Note: more than the first lambda + mu terms may be returned and this
    is the cost of cycle detection with Brent's method; there are, however,
    generally less terms calculated than would have been calculated if the
    proper ending point were determined, e.g. by using Floyd's method.

    >>> from sympy.ntheory.generate import cycle_length

    This will yield successive values of i <-- func(i):

        >>> def gen(func, i):
        ...     while 1:
        ...         yield i
        ...         i = func(i)
        ...

    A function is defined:

        >>> func = lambda i: (i**2 + 1) % 51

    and given a seed of 4 and the mu and lambda terms calculated:

        >>> next(cycle_length(func, 4))
        (6, 3)

    We can see what is meant by looking at the output:

        >>> iter = cycle_length(func, 4, values=True)
        >>> list(iter)
        [4, 17, 35, 2, 5, 26, 14, 44, 50, 2, 5, 26, 14]

    There are 6 repeating values after the first 3.

    If a sequence is suspected of being longer than you might wish, ``nmax``
    can be used to exit early (and mu will be returned as None):

        >>> next(cycle_length(func, 4, nmax = 4))
        (4, None)
        >>> list(cycle_length(func, 4, nmax = 4, values=True))
        [4, 17, 35, 2]

    Code modified from:
        https://en.wikipedia.org/wiki/Cycle_detection.
    """

    nmax = int(nmax or 0)

    # main phase: search successive powers of two
    power = lam = 1
    tortoise, hare = x0, f(x0)  # f(x0) is the element/node next to x0.
    i = 1
    if values:
        yield tortoise
    while tortoise != hare and (not nmax or i < nmax):
        i += 1
        if power == lam:   # time to start a new power of two?
            tortoise = hare
            power *= 2
            lam = 0
        if values:
            yield hare
        hare = f(hare)
        lam += 1
    if nmax and i == nmax:
        if values:
            return
        else:
            yield nmax, None
            return
    if not values:
        # Find the position of the first repetition of length lambda
        mu = 0
        tortoise = hare = x0
        for i in range(lam):
            hare = f(hare)
        while tortoise != hare:
            tortoise = f(tortoise)
            hare = f(hare)
            mu += 1
        yield lam, mu


def composite(nth):
    """ Return the nth composite number, with the composite numbers indexed as
        composite(1) = 4, composite(2) = 6, etc....

        Examples
        ========

        >>> from sympy import composite
        >>> composite(36)
        52
        >>> composite(1)
        4
        >>> composite(17737)
        20000

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        primepi : Return the number of primes less than or equal to n
        prime : Return the nth prime
        compositepi : Return the number of positive composite numbers less than or equal to n
    """
    n = as_int(nth)
    if n < 1:
        raise ValueError("nth must be a positive integer; composite(1) == 4")
    composite_arr = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]
    if n <= 10:
        return composite_arr[n - 1]

    a, b = 4, sieve._list[-1]
    if n <= b - _primepi(b) - 1:
        while a < b - 1:
            mid = (a + b) >> 1
            if mid - _primepi(mid) - 1 > n:
                b = mid
            else:
                a = mid
        if isprime(a):
            a -= 1
        return a

    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li
    a = 4 # Lower bound for binary search
    b = int(n*(log(n) + log(log(n)))) # Upper bound for the search.

    while a < b:
        mid = (a + b) >> 1
        if mid - li(mid) - 1 > n:
            b = mid
        else:
            a = mid + 1

    n_composites = a - _primepi(a) - 1
    while n_composites > n:
        if not isprime(a):
            n_composites -= 1
        a -= 1
    if isprime(a):
        a -= 1
    return a


def compositepi(n):
    """ Return the number of positive composite numbers less than or equal to n.
        The first positive composite is 4, i.e. compositepi(4) = 1.

        Examples
        ========

        >>> from sympy import compositepi
        >>> compositepi(25)
        15
        >>> compositepi(1000)
        831

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        prime : Return the nth prime
        primepi : Return the number of primes less than or equal to n
        composite : Return the nth composite number
    """
    n = int(n)
    if n < 4:
        return 0
    return n - _primepi(n) - 1
