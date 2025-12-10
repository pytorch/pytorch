from math import prod

from sympy.external.gmpy import gcd, gcdext
from sympy.ntheory.primetest import isprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
from sympy.utilities.misc import as_int


def symmetric_residue(a, m):
    """Return the residual mod m such that it is within half of the modulus.

    >>> from sympy.ntheory.modular import symmetric_residue
    >>> symmetric_residue(1, 6)
    1
    >>> symmetric_residue(4, 6)
    -2
    """
    if a <= m // 2:
        return a
    return a - m


def crt(m, v, symmetric=False, check=True):
    r"""Chinese Remainder Theorem.

    The moduli in m are assumed to be pairwise coprime.  The output
    is then an integer f, such that f = v_i mod m_i for each pair out
    of v and m. If ``symmetric`` is False a positive integer will be
    returned, else \|f\| will be less than or equal to the LCM of the
    moduli, and thus f may be negative.

    If the moduli are not co-prime the correct result will be returned
    if/when the test of the result is found to be incorrect. This result
    will be None if there is no solution.

    The keyword ``check`` can be set to False if it is known that the moduli
    are coprime.

    Examples
    ========

    As an example consider a set of residues ``U = [49, 76, 65]``
    and a set of moduli ``M = [99, 97, 95]``. Then we have::

       >>> from sympy.ntheory.modular import crt

       >>> crt([99, 97, 95], [49, 76, 65])
       (639985, 912285)

    This is the correct result because::

       >>> [639985 % m for m in [99, 97, 95]]
       [49, 76, 65]

    If the moduli are not co-prime, you may receive an incorrect result
    if you use ``check=False``:

       >>> crt([12, 6, 17], [3, 4, 2], check=False)
       (954, 1224)
       >>> [954 % m for m in [12, 6, 17]]
       [6, 0, 2]
       >>> crt([12, 6, 17], [3, 4, 2]) is None
       True
       >>> crt([3, 6], [2, 5])
       (5, 6)

    Note: the order of gf_crt's arguments is reversed relative to crt,
    and that solve_congruence takes residue, modulus pairs.

    Programmer's note: rather than checking that all pairs of moduli share
    no GCD (an O(n**2) test) and rather than factoring all moduli and seeing
    that there is no factor in common, a check that the result gives the
    indicated residuals is performed -- an O(n) operation.

    See Also
    ========

    solve_congruence
    sympy.polys.galoistools.gf_crt : low level crt routine used by this routine
    """
    if check:
        m = list(map(as_int, m))
        v = list(map(as_int, v))

    result = gf_crt(v, m, ZZ)
    mm = prod(m)

    if check:
        if not all(v % m == result % m for v, m in zip(v, m)):
            result = solve_congruence(*list(zip(v, m)),
                    check=False, symmetric=symmetric)
            if result is None:
                return result
            result, mm = result

    if symmetric:
        return int(symmetric_residue(result, mm)), int(mm)
    return int(result), int(mm)


def crt1(m):
    """First part of Chinese Remainder Theorem, for multiple application.

    Examples
    ========

    >>> from sympy.ntheory.modular import crt, crt1, crt2
    >>> m = [99, 97, 95]
    >>> v = [49, 76, 65]

    The following two codes have the same result.

    >>> crt(m, v)
    (639985, 912285)

    >>> mm, e, s = crt1(m)
    >>> crt2(m, v, mm, e, s)
    (639985, 912285)

    However, it is faster when we want to fix ``m`` and
    compute for multiple ``v``, i.e. the following cases:

    >>> mm, e, s = crt1(m)
    >>> vs = [[52, 21, 37], [19, 46, 76]]
    >>> for v in vs:
    ...     print(crt2(m, v, mm, e, s))
    (397042, 912285)
    (803206, 912285)

    See Also
    ========

    sympy.polys.galoistools.gf_crt1 : low level crt routine used by this routine
    sympy.ntheory.modular.crt
    sympy.ntheory.modular.crt2

    """

    return gf_crt1(m, ZZ)


def crt2(m, v, mm, e, s, symmetric=False):
    """Second part of Chinese Remainder Theorem, for multiple application.

    See ``crt1`` for usage.

    Examples
    ========

    >>> from sympy.ntheory.modular import crt1, crt2
    >>> mm, e, s = crt1([18, 42, 6])
    >>> crt2([18, 42, 6], [0, 0, 0], mm, e, s)
    (0, 4536)

    See Also
    ========

    sympy.polys.galoistools.gf_crt2 : low level crt routine used by this routine
    sympy.ntheory.modular.crt
    sympy.ntheory.modular.crt1

    """

    result = gf_crt2(v, m, mm, e, s, ZZ)

    if symmetric:
        return int(symmetric_residue(result, mm)), int(mm)
    return int(result), int(mm)


def solve_congruence(*remainder_modulus_pairs, **hint):
    """Compute the integer ``n`` that has the residual ``ai`` when it is
    divided by ``mi`` where the ``ai`` and ``mi`` are given as pairs to
    this function: ((a1, m1), (a2, m2), ...). If there is no solution,
    return None. Otherwise return ``n`` and its modulus.

    The ``mi`` values need not be co-prime. If it is known that the moduli are
    not co-prime then the hint ``check`` can be set to False (default=True) and
    the check for a quicker solution via crt() (valid when the moduli are
    co-prime) will be skipped.

    If the hint ``symmetric`` is True (default is False), the value of ``n``
    will be within 1/2 of the modulus, possibly negative.

    Examples
    ========

    >>> from sympy.ntheory.modular import solve_congruence

    What number is 2 mod 3, 3 mod 5 and 2 mod 7?

    >>> solve_congruence((2, 3), (3, 5), (2, 7))
    (23, 105)
    >>> [23 % m for m in [3, 5, 7]]
    [2, 3, 2]

    If you prefer to work with all remainder in one list and
    all moduli in another, send the arguments like this:

    >>> solve_congruence(*zip((2, 3, 2), (3, 5, 7)))
    (23, 105)

    The moduli need not be co-prime; in this case there may or
    may not be a solution:

    >>> solve_congruence((2, 3), (4, 6)) is None
    True

    >>> solve_congruence((2, 3), (5, 6))
    (5, 6)

    The symmetric flag will make the result be within 1/2 of the modulus:

    >>> solve_congruence((2, 3), (5, 6), symmetric=True)
    (-1, 6)

    See Also
    ========

    crt : high level routine implementing the Chinese Remainder Theorem

    """
    def combine(c1, c2):
        """Return the tuple (a, m) which satisfies the requirement
        that n = a + i*m satisfy n = a1 + j*m1 and n = a2 = k*m2.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Method_of_successive_substitution
        """
        a1, m1 = c1
        a2, m2 = c2
        a, b, c = m1, a2 - a1, m2
        g = gcd(a, b, c)
        a, b, c = [i//g for i in [a, b, c]]
        if a != 1:
            g, inv_a, _ = gcdext(a, c)
            if g != 1:
                return None
            b *= inv_a
        a, m = a1 + m1*b, m1*c
        return a, m

    rm = remainder_modulus_pairs
    symmetric = hint.get('symmetric', False)

    if hint.get('check', True):
        rm = [(as_int(r), as_int(m)) for r, m in rm]

        # ignore redundant pairs but raise an error otherwise; also
        # make sure that a unique set of bases is sent to gf_crt if
        # they are all prime.
        #
        # The routine will work out less-trivial violations and
        # return None, e.g. for the pairs (1,3) and (14,42) there
        # is no answer because 14 mod 42 (having a gcd of 14) implies
        # (14/2) mod (42/2), (14/7) mod (42/7) and (14/14) mod (42/14)
        # which, being 0 mod 3, is inconsistent with 1 mod 3. But to
        # preprocess the input beyond checking of another pair with 42
        # or 3 as the modulus (for this example) is not necessary.
        uniq = {}
        for r, m in rm:
            r %= m
            if m in uniq:
                if r != uniq[m]:
                    return None
                continue
            uniq[m] = r
        rm = [(r, m) for m, r in uniq.items()]
        del uniq

        # if the moduli are co-prime, the crt will be significantly faster;
        # checking all pairs for being co-prime gets to be slow but a prime
        # test is a good trade-off
        if all(isprime(m) for r, m in rm):
            r, m = list(zip(*rm))
            return crt(m, r, symmetric=symmetric, check=False)

    rv = (0, 1)
    for rmi in rm:
        rv = combine(rv, rmi)
        if rv is None:
            break
        n, m = rv
        n = n % m
    else:
        if symmetric:
            return symmetric_residue(n, m), m
        return n, m
