from math import exp, log
from sympy.core.random import _randint
from sympy.external.gmpy import bit_scan1, gcd, invert, sqrt as isqrt
from sympy.ntheory.factor_ import _perfect_power
from sympy.ntheory.primetest import isprime
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power


class SievePolynomial:
    def __init__(self, a, b, N):
        """This class denotes the sieve polynomial.
        Provide methods to compute `(a*x + b)**2 - N` and
        `a*x + b` when given `x`.

        Parameters
        ==========

        a : parameter of the sieve polynomial
        b : parameter of the sieve polynomial
        N : number to be factored

        """
        self.a = a
        self.b = b
        self.a2 = a**2
        self.ab = 2*a*b
        self.b2 = b**2 - N

    def eval_u(self, x):
        return self.a*x + self.b

    def eval_v(self, x):
        return (self.a2*x + self.ab)*x + self.b2


class FactorBaseElem:
    """This class stores an element of the `factor_base`.
    """
    def __init__(self, prime, tmem_p, log_p):
        """
        Initialization of factor_base_elem.

        Parameters
        ==========

        prime : prime number of the factor_base
        tmem_p : Integer square root of x**2 = n mod prime
        log_p : Compute Natural Logarithm of the prime
        """
        self.prime = prime
        self.tmem_p = tmem_p
        self.log_p = log_p
        # `soln1` and `soln2` are solutions to
        # the equation `(a*x + b)**2 - N = 0 (mod p)`.
        self.soln1 = None
        self.soln2 = None
        self.b_ainv = None


def _generate_factor_base(prime_bound, n):
    """Generate `factor_base` for Quadratic Sieve. The `factor_base`
    consists of all the points whose ``legendre_symbol(n, p) == 1``
    and ``p < num_primes``. Along with the prime `factor_base` also stores
    natural logarithm of prime and the residue n modulo p.
    It also returns the of primes numbers in the `factor_base` which are
    close to 1000 and 5000.

    Parameters
    ==========

    prime_bound : upper prime bound of the factor_base
    n : integer to be factored
    """
    from sympy.ntheory.generate import sieve
    factor_base = []
    idx_1000, idx_5000 = None, None
    for prime in sieve.primerange(1, prime_bound):
        if pow(n, (prime - 1) // 2, prime) == 1:
            if prime > 1000 and idx_1000 is None:
                idx_1000 = len(factor_base) - 1
            if prime > 5000 and idx_5000 is None:
                idx_5000 = len(factor_base) - 1
            residue = _sqrt_mod_prime_power(n, prime, 1)[0]
            log_p = round(log(prime)*2**10)
            factor_base.append(FactorBaseElem(prime, residue, log_p))
    return idx_1000, idx_5000, factor_base


def _generate_polynomial(N, M, factor_base, idx_1000, idx_5000, randint):
    """ Generate sieve polynomials indefinitely.
    Information such as `soln1` in the `factor_base` associated with
    the polynomial is modified in place.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    idx_1000 : index of prime number in the factor_base near 1000
    idx_5000 : index of prime number in the factor_base near to 5000
    randint : A callable that takes two integers (a, b) and returns a random integer
              n such that a <= n <= b, similar to `random.randint`.
    """
    approx_val = log(2*N)/2 - log(M)
    start = idx_1000 or 0
    end = idx_5000 or (len(factor_base) - 1)
    while True:
        # Choose `a` that is close to `sqrt(2*N) / M`
        best_a, best_q, best_ratio = None, None, None
        for _ in range(50):
            a = 1
            q = []
            while log(a) < approx_val:
                rand_p = 0
                while(rand_p == 0 or rand_p in q):
                    rand_p = randint(start, end)
                p = factor_base[rand_p].prime
                a *= p
                q.append(rand_p)
            ratio = exp(log(a) - approx_val)
            if best_ratio is None or abs(ratio - 1) < abs(best_ratio - 1):
                best_q = q
                best_a = a
                best_ratio = ratio

        # Set `b` using the Chinese remainder theorem
        a = best_a
        q = best_q
        B = []
        for val in q:
            q_l = factor_base[val].prime
            gamma = factor_base[val].tmem_p * invert(a // q_l, q_l) % q_l
            if 2*gamma > q_l:
                gamma = q_l - gamma
            B.append(a//q_l*gamma)
        b = sum(B)
        g = SievePolynomial(a, b, N)
        for fb in factor_base:
            if a % fb.prime == 0:
                fb.soln1 = None
                continue
            a_inv = invert(a, fb.prime)
            fb.b_ainv = [2*b_elem*a_inv % fb.prime for b_elem in B]
            fb.soln1 = (a_inv*(fb.tmem_p - b)) % fb.prime
            fb.soln2 = (a_inv*(-fb.tmem_p - b)) % fb.prime
        yield g

        # Update `b` with Gray code
        for i in range(1, 2**(len(B)-1)):
            v = bit_scan1(i)
            neg_pow = 2*((i >> (v + 1)) % 2) - 1
            b = g.b + 2*neg_pow*B[v]
            a = g.a
            g = SievePolynomial(a, b, N)
            for fb in factor_base:
                if fb.soln1 is None:
                    continue
                fb.soln1 = (fb.soln1 - neg_pow*fb.b_ainv[v]) % fb.prime
                fb.soln2 = (fb.soln2 - neg_pow*fb.b_ainv[v]) % fb.prime
            yield g


def _gen_sieve_array(M, factor_base):
    """Sieve Stage of the Quadratic Sieve. For every prime in the factor_base
    that does not divide the coefficient `a` we add log_p over the sieve_array
    such that ``-M <= soln1 + i*p <=  M`` and ``-M <= soln2 + i*p <=  M`` where `i`
    is an integer. When p = 2 then log_p is only added using
    ``-M <= soln1 + i*p <=  M``.

    Parameters
    ==========

    M : sieve interval
    factor_base : factor_base primes
    """
    sieve_array = [0]*(2*M + 1)
    for factor in factor_base:
        if factor.soln1 is None: #The prime does not divides a
            continue
        for idx in range((M + factor.soln1) % factor.prime, 2*M, factor.prime):
            sieve_array[idx] += factor.log_p
        if factor.prime == 2:
            continue
        #if prime is 2 then sieve only with soln_1_p
        for idx in range((M + factor.soln2) % factor.prime, 2*M, factor.prime):
            sieve_array[idx] += factor.log_p
    return sieve_array


def _check_smoothness(num, factor_base):
    r""" Check if `num` is smooth with respect to the given `factor_base`
    and compute its factorization vector.

    Parameters
    ==========

    num : integer whose smootheness is to be checked
    factor_base : factor_base primes
    """
    if num < 0:
        num *= -1
        vec = 1
    else:
        vec = 0
    for i, fb in enumerate(factor_base, 1):
        if num % fb.prime:
            continue
        e = 1
        num //= fb.prime
        while num % fb.prime == 0:
            e += 1
            num //= fb.prime
        if e % 2:
            vec += 1 << i
    return vec, num


def _trial_division_stage(N, M, factor_base, sieve_array, sieve_poly, partial_relations, ERROR_TERM):
    """Trial division stage. Here we trial divide the values generetated
    by sieve_poly in the sieve interval and if it is a smooth number then
    it is stored in `smooth_relations`. Moreover, if we find two partial relations
    with same large prime then they are combined to form a smooth relation.
    First we iterate over sieve array and look for values which are greater
    than accumulated_val, as these values have a high chance of being smooth
    number. Then using these values we find smooth relations.
    In general, let ``t**2 = u*p modN`` and ``r**2 = v*p modN`` be two partial relations
    with the same large prime p. Then they can be combined ``(t*r/p)**2 = u*v modN``
    to form a smooth relation.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    sieve_array : stores log_p values
    sieve_poly : polynomial from which we find smooth relations
    partial_relations : stores partial relations with one large prime
    ERROR_TERM : error term for accumulated_val
    """
    accumulated_val = (log(M) + log(N)/2 - ERROR_TERM) * 2**10
    smooth_relations = []
    proper_factor = set()
    partial_relation_upper_bound = 128*factor_base[-1].prime
    for x, val in enumerate(sieve_array, -M):
        if val < accumulated_val:
            continue
        v = sieve_poly.eval_v(x)
        vec, num = _check_smoothness(v, factor_base)
        if num == 1:
            smooth_relations.append((sieve_poly.eval_u(x), v, vec))
        elif num < partial_relation_upper_bound and isprime(num):
            if N % num == 0:
                proper_factor.add(num)
                continue
            u = sieve_poly.eval_u(x)
            if num in partial_relations:
                u_prev, v_prev, vec_prev = partial_relations.pop(num)
                u = u*u_prev*invert(num, N) % N
                v = v*v_prev // num**2
                vec ^= vec_prev
                smooth_relations.append((u, v, vec))
            else:
                partial_relations[num] = (u, v, vec)
    return smooth_relations, proper_factor


def _find_factor(N, smooth_relations, col):
    """ Finds proper factor of N using fast gaussian reduction for modulo 2 matrix.

    Parameters
    ==========

    N : Number to be factored
    smooth_relations : Smooth relations vectors matrix
    col : Number of columns in the matrix

    Reference
    ==========

    .. [1] A fast algorithm for gaussian elimination over GF(2) and
    its implementation on the GAPP. Cetin K.Koc, Sarath N.Arachchige
    """
    matrix = [s_relation[2] for s_relation in smooth_relations]
    row = len(matrix)
    mark = [False] * row
    for pos in range(col):
        m = 1 << pos
        for i in range(row):
            if p := matrix[i] & m:
                add_col = p ^ matrix[i]
                matrix[i] = m
                mark[i] = True
                for j in range(i + 1, row):
                    if matrix[j] & m:
                        matrix[j] ^= add_col
                break

    for m, mat, rel in zip(mark, matrix, smooth_relations):
        if m:
            continue
        u, v = rel[0], rel[1]
        for m1, mat1, rel1 in zip(mark, matrix, smooth_relations):
            if m1 and mat & mat1:
                u *= rel1[0]
                v *= rel1[1]
        # assert is_square(v)
        v = isqrt(v)
        if 1 < (g := gcd(u - v, N)) < N:
            yield g


def qs(N, prime_bound, M, ERROR_TERM=25, seed=1234):
    """Performs factorization using Self-Initializing Quadratic Sieve.
    In SIQS, let N be a number to be factored, and this N should not be a
    perfect power. If we find two integers such that ``X**2 = Y**2 modN`` and
    ``X != +-Y modN``, then `gcd(X + Y, N)` will reveal a proper factor of N.
    In order to find these integers X and Y we try to find relations of form
    t**2 = u modN where u is a product of small primes. If we have enough of
    these relations then we can form ``(t1*t2...ti)**2 = u1*u2...ui modN`` such that
    the right hand side is a square, thus we found a relation of ``X**2 = Y**2 modN``.

    Here, several optimizations are done like using multiple polynomials for
    sieving, fast changing between polynomials and using partial relations.
    The use of partial relations can speeds up the factoring by 2 times.

    Parameters
    ==========

    N : Number to be Factored
    prime_bound : upper bound for primes in the factor base
    M : Sieve Interval
    ERROR_TERM : Error term for checking smoothness
    seed : seed of random number generator

    Returns
    =======

    set(int) : A set of factors of N without considering multiplicity.
               Returns ``{N}`` if factorization fails.

    Examples
    ========

    >>> from sympy.ntheory import qs
    >>> qs(25645121643901801, 2000, 10000)
    {5394769, 4753701529}
    >>> qs(9804659461513846513, 2000, 10000)
    {4641991, 2112166839943}

    See Also
    ========

    qs_factor

    References
    ==========

    .. [1] https://pdfs.semanticscholar.org/5c52/8a975c1405bd35c65993abf5a4edb667c1db.pdf
    .. [2] https://www.rieselprime.de/ziki/Self-initializing_quadratic_sieve
    """
    return set(qs_factor(N, prime_bound, M, ERROR_TERM, seed))


def qs_factor(N, prime_bound, M, ERROR_TERM=25, seed=1234):
    """ Performs factorization using Self-Initializing Quadratic Sieve.

    Parameters
    ==========

    N : Number to be Factored
    prime_bound : upper bound for primes in the factor base
    M : Sieve Interval
    ERROR_TERM : Error term for checking smoothness
    seed : seed of random number generator

    Returns
    =======

    dict[int, int] : Factors of N.
                     Returns ``{N: 1}`` if factorization fails.
                     Note that the key is not always a prime number.

    Examples
    ========

    >>> from sympy.ntheory import qs_factor
    >>> qs_factor(1009 * 100003, 2000, 10000)
    {1009: 1, 100003: 1}

    See Also
    ========

    qs

    """
    if N < 2:
        raise ValueError("N should be greater than 1")
    factors = {}
    smooth_relations = []
    partial_relations = {}
    # Eliminate the possibility of even numbers,
    # prime numbers, and perfect powers.
    if N % 2 == 0:
        e = 1
        N //= 2
        while N % 2 == 0:
            N //= 2
            e += 1
        factors[2] = e
    if isprime(N):
        factors[N] = 1
        return factors
    if result := _perfect_power(N, 3):
        n, e = result
        factors[n] = e
        return factors
    N_copy = N
    randint = _randint(seed)
    idx_1000, idx_5000, factor_base = _generate_factor_base(prime_bound, N)
    threshold = len(factor_base) * 105//100
    for g in _generate_polynomial(N, M, factor_base, idx_1000, idx_5000, randint):
        sieve_array = _gen_sieve_array(M, factor_base)
        s_rel, p_f = _trial_division_stage(N, M, factor_base, sieve_array, g, partial_relations, ERROR_TERM)
        smooth_relations += s_rel
        for p in p_f:
            if N_copy % p:
                continue
            e = 1
            N_copy //= p
            while N_copy % p == 0:
                N_copy //= p
                e += 1
            factors[p] = e
        if threshold <= len(smooth_relations):
            break

    for factor in _find_factor(N, smooth_relations, len(factor_base) + 1):
        if N_copy % factor == 0:
            e = 1
            N_copy //= factor
            while N_copy % factor == 0:
                N_copy //= factor
                e += 1
            factors[factor] = e
            if N_copy == 1 or isprime(N_copy):
                break
    if N_copy != 1:
        factors[N_copy] = 1
    return factors
