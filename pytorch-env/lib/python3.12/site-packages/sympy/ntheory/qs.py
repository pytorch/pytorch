from sympy.core.random import _randint
from sympy.external.gmpy import gcd, invert, sqrt as isqrt
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt


class SievePolynomial:
    def __init__(self, modified_coeff=(), a=None, b=None):
        """This class denotes the seive polynomial.
        If ``g(x) = (a*x + b)**2 - N``. `g(x)` can be expanded
        to ``a*x**2 + 2*a*b*x + b**2 - N``, so the coefficient
        is stored in the form `[a**2, 2*a*b, b**2 - N]`. This
        ensures faster `eval` method because we dont have to
        perform `a**2, 2*a*b, b**2` every time we call the
        `eval` method. As multiplication is more expensive
        than addition, by using modified_coefficient we get
        a faster seiving process.

        Parameters
        ==========

        modified_coeff : modified_coefficient of sieve polynomial
        a : parameter of the sieve polynomial
        b : parameter of the sieve polynomial
        """
        self.modified_coeff = modified_coeff
        self.a = a
        self.b = b

    def eval(self, x):
        """
        Compute the value of the sieve polynomial at point x.

        Parameters
        ==========

        x : Integer parameter for sieve polynomial
        """
        ans = 0
        for coeff in self.modified_coeff:
            ans *= x
            ans += coeff
        return ans


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
        self.soln1 = None
        self.soln2 = None
        self.a_inv = None
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


def _initialize_first_polynomial(N, M, factor_base, idx_1000, idx_5000, seed=None):
    """This step is the initialization of the 1st sieve polynomial.
    Here `a` is selected as a product of several primes of the factor_base
    such that `a` is about to ``sqrt(2*N) / M``. Other initial values of
    factor_base elem are also initialized which includes a_inv, b_ainv, soln1,
    soln2 which are used when the sieve polynomial is changed. The b_ainv
    is required for fast polynomial change as we do not have to calculate
    `2*b*invert(a, prime)` every time.
    We also ensure that the `factor_base` primes which make `a` are between
    1000 and 5000.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    idx_1000 : index of prime number in the factor_base near 1000
    idx_5000 : index of prime number in the factor_base near to 5000
    seed : Generate pseudoprime numbers
    """
    randint = _randint(seed)
    approx_val = sqrt(2*N) / M
    # `a` is a parameter of the sieve polynomial and `q` is the prime factors of `a`
    # randomly search for a combination of primes whose multiplication is close to approx_val
    # This multiplication of primes will be `a` and the primes will be `q`
    # `best_a` denotes that `a` is close to approx_val in the random search of combination
    best_a, best_q, best_ratio = None, None, None
    start = 0 if idx_1000 is None else idx_1000
    end = len(factor_base) - 1 if idx_5000 is None else idx_5000
    for _ in range(50):
        a = 1
        q = []
        while(a < approx_val):
            rand_p = 0
            while(rand_p == 0 or rand_p in q):
                rand_p = randint(start, end)
            p = factor_base[rand_p].prime
            a *= p
            q.append(rand_p)
        ratio = a / approx_val
        if best_ratio is None or abs(ratio - 1) < abs(best_ratio - 1):
            best_q = q
            best_a = a
            best_ratio = ratio

    a = best_a
    q = best_q

    B = []
    for val in q:
        q_l = factor_base[val].prime
        gamma = factor_base[val].tmem_p * invert(a // q_l, q_l) % q_l
        if gamma > q_l / 2:
            gamma = q_l - gamma
        B.append(a//q_l*gamma)

    b = sum(B)
    g = SievePolynomial([a*a, 2*a*b, b*b - N], a, b)

    for fb in factor_base:
        if a % fb.prime == 0:
            continue
        fb.a_inv = invert(a, fb.prime)
        fb.b_ainv = [2*b_elem*fb.a_inv % fb.prime for b_elem in B]
        fb.soln1 = (fb.a_inv*(fb.tmem_p - b)) % fb.prime
        fb.soln2 = (fb.a_inv*(-fb.tmem_p - b)) % fb.prime
    return g, B


def _initialize_ith_poly(N, factor_base, i, g, B):
    """Initialization stage of ith poly. After we finish sieving 1`st polynomial
    here we quickly change to the next polynomial from which we will again
    start sieving. Suppose we generated ith sieve polynomial and now we
    want to generate (i + 1)th polynomial, where ``1 <= i <= 2**(j - 1) - 1``
    where `j` is the number of prime factors of the coefficient `a`
    then this function can be used to go to the next polynomial. If
    ``i = 2**(j - 1) - 1`` then go to _initialize_first_polynomial stage.

    Parameters
    ==========

    N : number to be factored
    factor_base : factor_base primes
    i : integer denoting ith polynomial
    g : (i - 1)th polynomial
    B : array that stores a//q_l*gamma
    """
    from sympy.functions.elementary.integers import ceiling
    v = 1
    j = i
    while(j % 2 == 0):
        v += 1
        j //= 2
    if ceiling(i / (2**v)) % 2 == 1:
        neg_pow = -1
    else:
        neg_pow = 1
    b = g.b + 2*neg_pow*B[v - 1]
    a = g.a
    g = SievePolynomial([a*a, 2*a*b, b*b - N], a, b)
    for fb in factor_base:
        if a % fb.prime == 0:
            continue
        fb.soln1 = (fb.soln1 - neg_pow*fb.b_ainv[v - 1]) % fb.prime
        fb.soln2 = (fb.soln2 - neg_pow*fb.b_ainv[v - 1]) % fb.prime

    return g


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
    """Here we check that if `num` is a smooth number or not. If `a` is a smooth
    number then it returns a vector of prime exponents modulo 2. For example
    if a = 2 * 5**2 * 7**3 and the factor base contains {2, 3, 5, 7} then
    `a` is a smooth number and this function returns ([1, 0, 0, 1], True). If
    `a` is a partial relation which means that `a` a has one prime factor
    greater than the `factor_base` then it returns `(a, False)` which denotes `a`
    is a partial relation.

    Parameters
    ==========

    a : integer whose smootheness is to be checked
    factor_base : factor_base primes
    """
    vec = []
    if num < 0:
        vec.append(1)
        num *= -1
    else:
        vec.append(0)
     #-1 is not included in factor_base add -1 in vector
    for factor in factor_base:
        if num % factor.prime != 0:
            vec.append(0)
            continue
        factor_exp = 0
        while num % factor.prime == 0:
            factor_exp += 1
            num //= factor.prime
        vec.append(factor_exp % 2)
    if num == 1:
        return vec, True
    if isprime(num):
        return num, False
    return None, None


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
    sqrt_n = isqrt(N)
    accumulated_val = log(M * sqrt_n)*2**10 - ERROR_TERM
    smooth_relations = []
    proper_factor = set()
    partial_relation_upper_bound = 128*factor_base[-1].prime
    for idx, val in enumerate(sieve_array):
        if val < accumulated_val:
            continue
        x = idx - M
        v = sieve_poly.eval(x)
        vec, is_smooth = _check_smoothness(v, factor_base)
        if is_smooth is None:#Neither smooth nor partial
            continue
        u = sieve_poly.a*x + sieve_poly.b
        # Update the partial relation
        # If 2 partial relation with same large prime is found then generate smooth relation
        if is_smooth is False:#partial relation found
            large_prime = vec
            #Consider the large_primes under 128*F
            if large_prime > partial_relation_upper_bound:
                continue
            if large_prime not in partial_relations:
                partial_relations[large_prime] = (u, v)
                continue
            else:
                u_prev, v_prev = partial_relations[large_prime]
                partial_relations.pop(large_prime)
                try:
                    large_prime_inv = invert(large_prime, N)
                except ZeroDivisionError:#if large_prime divides N
                    proper_factor.add(large_prime)
                    continue
                u = u*u_prev*large_prime_inv
                v = v*v_prev // (large_prime*large_prime)
                vec, is_smooth = _check_smoothness(v, factor_base)
        #assert u*u % N == v % N
        smooth_relations.append((u, v, vec))
    return smooth_relations, proper_factor


#LINEAR ALGEBRA STAGE
def _build_matrix(smooth_relations):
    """Build a 2D matrix from smooth relations.

    Parameters
    ==========

    smooth_relations : Stores smooth relations
    """
    matrix = []
    for s_relation in smooth_relations:
        matrix.append(s_relation[2])
    return matrix


def _gauss_mod_2(A):
    """Fast gaussian reduction for modulo 2 matrix.

    Parameters
    ==========

    A : Matrix

    Examples
    ========

    >>> from sympy.ntheory.qs import _gauss_mod_2
    >>> _gauss_mod_2([[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1]])
    ([[[1, 0, 1], 3]],
     [True, True, True, False],
     [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])

    Reference
    ==========

    .. [1] A fast algorithm for gaussian elimination over GF(2) and
    its implementation on the GAPP. Cetin K.Koc, Sarath N.Arachchige"""
    import copy
    matrix = copy.deepcopy(A)
    row = len(matrix)
    col = len(matrix[0])
    mark = [False]*row
    for c in range(col):
        for r in range(row):
            if matrix[r][c] == 1:
                break
        mark[r] = True
        for c1 in range(col):
            if c1 == c:
                continue
            if matrix[r][c1] == 1:
                for r2 in range(row):
                    matrix[r2][c1] = (matrix[r2][c1] + matrix[r2][c]) % 2
    dependent_row = []
    for idx, val in enumerate(mark):
        if val == False:
            dependent_row.append([matrix[idx], idx])
    return dependent_row, mark, matrix


def _find_factor(dependent_rows, mark, gauss_matrix, index, smooth_relations, N):
    """Finds proper factor of N. Here, transform the dependent rows as a
    combination of independent rows of the gauss_matrix to form the desired
    relation of the form ``X**2 = Y**2 modN``. After obtaining the desired relation
    we obtain a proper factor of N by `gcd(X - Y, N)`.

    Parameters
    ==========

    dependent_rows : denoted dependent rows in the reduced matrix form
    mark : boolean array to denoted dependent and independent rows
    gauss_matrix : Reduced form of the smooth relations matrix
    index : denoted the index of the dependent_rows
    smooth_relations : Smooth relations vectors matrix
    N : Number to be factored
    """
    idx_in_smooth = dependent_rows[index][1]
    independent_u = [smooth_relations[idx_in_smooth][0]]
    independent_v = [smooth_relations[idx_in_smooth][1]]
    dept_row = dependent_rows[index][0]

    for idx, val in enumerate(dept_row):
        if val == 1:
            for row in range(len(gauss_matrix)):
                if gauss_matrix[row][idx] == 1 and mark[row] == True:
                    independent_u.append(smooth_relations[row][0])
                    independent_v.append(smooth_relations[row][1])
                    break

    u = 1
    v = 1
    for i in independent_u:
        u *= i
    for i in independent_v:
        v *= i
    #assert u**2 % N == v % N
    v = isqrt(v)
    return gcd(u - v, N)


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
    threshold : Extra smooth relations for factorization
    seed : generate pseudo prime numbers

    Examples
    ========

    >>> from sympy.ntheory import qs
    >>> qs(25645121643901801, 2000, 10000)
    {5394769, 4753701529}
    >>> qs(9804659461513846513, 2000, 10000)
    {4641991, 2112166839943}

    References
    ==========

    .. [1] https://pdfs.semanticscholar.org/5c52/8a975c1405bd35c65993abf5a4edb667c1db.pdf
    .. [2] https://www.rieselprime.de/ziki/Self-initializing_quadratic_sieve
    """
    ERROR_TERM*=2**10
    idx_1000, idx_5000, factor_base = _generate_factor_base(prime_bound, N)
    smooth_relations = []
    ith_poly = 0
    partial_relations = {}
    proper_factor = set()
    threshold = 5*len(factor_base) // 100
    while True:
        if ith_poly == 0:
            ith_sieve_poly, B_array = _initialize_first_polynomial(N, M, factor_base, idx_1000, idx_5000)
        else:
            ith_sieve_poly = _initialize_ith_poly(N, factor_base, ith_poly, ith_sieve_poly, B_array)
        ith_poly += 1
        if ith_poly >= 2**(len(B_array) - 1): # time to start with a new sieve polynomial
            ith_poly = 0
        sieve_array = _gen_sieve_array(M, factor_base)
        s_rel, p_f = _trial_division_stage(N, M, factor_base, sieve_array, ith_sieve_poly, partial_relations, ERROR_TERM)
        smooth_relations += s_rel
        proper_factor |= p_f
        if len(smooth_relations) >= len(factor_base) + threshold:
            break
    matrix = _build_matrix(smooth_relations)
    dependent_row, mark, gauss_matrix = _gauss_mod_2(matrix)
    N_copy = N
    for index in range(len(dependent_row)):
        factor = _find_factor(dependent_row, mark, gauss_matrix, index, smooth_relations, N)
        if factor > 1 and factor < N:
            proper_factor.add(factor)
            while(N_copy % factor == 0):
                N_copy //= factor
            if isprime(N_copy):
                proper_factor.add(N_copy)
                break
            if(N_copy == 1):
                break
    return proper_factor
