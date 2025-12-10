from math import log

from sympy.core.random import _randint
from sympy.external.gmpy import gcd, invert, sqrt
from sympy.utilities.misc import as_int
from .generate import sieve, primerange
from .primetest import isprime


#----------------------------------------------------------------------------#
#                                                                            #
#                   Lenstra's Elliptic Curve Factorization                   #
#                                                                            #
#----------------------------------------------------------------------------#


class Point:
    """Montgomery form of Points in an elliptic curve.
    In this form, the addition and doubling of points
    does not need any y-coordinate information thus
    decreasing the number of operations.
    Using Montgomery form we try to perform point addition
    and doubling in least amount of multiplications.

    The elliptic curve used here is of the form
    (E : b*y**2*z = x**3 + a*x**2*z + x*z**2).
    The a_24 parameter is equal to (a + 2)/4.

    References
    ==========

    .. [1] Kris Gaj, Soonhak Kwon, Patrick Baier, Paul Kohlbrenner, Hoang Le, Mohammed Khaleeluddin, Ramakrishna Bachimanchi,
           Implementing the Elliptic Curve Method of Factoring in Reconfigurable Hardware,
           Cryptographic Hardware and Embedded Systems - CHES 2006 (2006), pp. 119-133,
           https://doi.org/10.1007/11894063_10
           https://www.hyperelliptic.org/tanja/SHARCS/talks06/Gaj.pdf

    """

    def __init__(self, x_cord, z_cord, a_24, mod):
        """
        Initial parameters for the Point class.

        Parameters
        ==========

        x_cord : X coordinate of the Point
        z_cord : Z coordinate of the Point
        a_24 : Parameter of the elliptic curve in Montgomery form
        mod : modulus
        """
        self.x_cord = x_cord
        self.z_cord = z_cord
        self.a_24 = a_24
        self.mod = mod

    def __eq__(self, other):
        """Two points are equal if X/Z of both points are equal
        """
        if self.a_24 != other.a_24 or self.mod != other.mod:
            return False
        return self.x_cord * other.z_cord % self.mod ==\
            other.x_cord * self.z_cord % self.mod

    def add(self, Q, diff):
        """
        Add two points self and Q where diff = self - Q. Moreover the assumption
        is self.x_cord*Q.x_cord*(self.x_cord - Q.x_cord) != 0. This algorithm
        requires 6 multiplications. Here the difference between the points
        is already known and using this algorithm speeds up the addition
        by reducing the number of multiplication required. Also in the
        mont_ladder algorithm is constructed in a way so that the difference
        between intermediate points is always equal to the initial point.
        So, we always know what the difference between the point is.


        Parameters
        ==========

        Q : point on the curve in Montgomery form
        diff : self - Q

        Examples
        ========

        >>> from sympy.ntheory.ecm import Point
        >>> p1 = Point(11, 16, 7, 29)
        >>> p2 = Point(13, 10, 7, 29)
        >>> p3 = p2.add(p1, p1)
        >>> p3.x_cord
        23
        >>> p3.z_cord
        17
        """
        u = (self.x_cord - self.z_cord)*(Q.x_cord + Q.z_cord)
        v = (self.x_cord + self.z_cord)*(Q.x_cord - Q.z_cord)
        add, subt = u + v, u - v
        x_cord = diff.z_cord * add * add % self.mod
        z_cord = diff.x_cord * subt * subt % self.mod
        return Point(x_cord, z_cord, self.a_24, self.mod)

    def double(self):
        """
        Doubles a point in an elliptic curve in Montgomery form.
        This algorithm requires 5 multiplications.

        Examples
        ========

        >>> from sympy.ntheory.ecm import Point
        >>> p1 = Point(11, 16, 7, 29)
        >>> p2 = p1.double()
        >>> p2.x_cord
        13
        >>> p2.z_cord
        10
        """
        u = pow(self.x_cord + self.z_cord, 2, self.mod)
        v = pow(self.x_cord - self.z_cord, 2, self.mod)
        diff = u - v
        x_cord = u*v % self.mod
        z_cord = diff*(v + self.a_24*diff) % self.mod
        return Point(x_cord, z_cord, self.a_24, self.mod)

    def mont_ladder(self, k):
        """
        Scalar multiplication of a point in Montgomery form
        using Montgomery Ladder Algorithm.
        A total of 11 multiplications are required in each step of this
        algorithm.

        Parameters
        ==========

        k : The positive integer multiplier

        Examples
        ========

        >>> from sympy.ntheory.ecm import Point
        >>> p1 = Point(11, 16, 7, 29)
        >>> p3 = p1.mont_ladder(3)
        >>> p3.x_cord
        23
        >>> p3.z_cord
        17
        """
        Q = self
        R = self.double()
        for i in bin(k)[3:]:
            if  i  == '1':
                Q = R.add(Q, self)
                R = R.double()
            else:
                R = Q.add(R, self)
                Q = Q.double()
        return Q


def _ecm_one_factor(n, B1=10000, B2=100000, max_curve=200, seed=None):
    """Returns one factor of n using
    Lenstra's 2 Stage Elliptic curve Factorization
    with Suyama's Parameterization. Here Montgomery
    arithmetic is used for fast computation of addition
    and doubling of points in elliptic curve.

    Explanation
    ===========

    This ECM method considers elliptic curves in Montgomery
    form (E : b*y**2*z = x**3 + a*x**2*z + x*z**2) and involves
    elliptic curve operations (mod N), where the elements in
    Z are reduced (mod N). Since N is not a prime, E over FF(N)
    is not really an elliptic curve but we can still do point additions
    and doubling as if FF(N) was a field.

    Stage 1 : The basic algorithm involves taking a random point (P) on an
    elliptic curve in FF(N). The compute k*P using Montgomery ladder algorithm.
    Let q be an unknown factor of N. Then the order of the curve E, |E(FF(q))|,
    might be a smooth number that divides k. Then we have k = l * |E(FF(q))|
    for some l. For any point belonging to the curve E, |E(FF(q))|*P = O,
    hence k*P = l*|E(FF(q))|*P. Thus kP.z_cord = 0 (mod q), and the unknownn
    factor of N (q) can be recovered by taking gcd(kP.z_cord, N).

    Stage 2 : This is a continuation of Stage 1 if k*P != O. The idea utilize
    the fact that even if kP != 0, the value of k might miss just one large
    prime divisor of |E(FF(q))|. In this case we only need to compute the
    scalar multiplication by p to get p*k*P = O. Here a second bound B2
    restrict the size of possible values of p.

    Parameters
    ==========

    n : Number to be Factored. Assume that it is a composite number.
    B1 : Stage 1 Bound. Must be an even number.
    B2 : Stage 2 Bound. Must be an even number.
    max_curve : Maximum number of curves generated

    Returns
    =======

    integer | None : a non-trivial divisor of ``n``. ``None`` if not found

    References
    ==========

    .. [1] Carl Pomerance, Richard Crandall, Prime Numbers: A Computational Perspective,
           2nd Edition (2005), page 344, ISBN:978-0387252827
    """
    randint = _randint(seed)

    # When calculating T, if (B1 - 2*D) is negative, it cannot be calculated.
    D = min(sqrt(B2), B1 // 2 - 1)
    sieve.extend(D)
    beta = [0] * D
    S = [0] * D
    k = 1
    for p in primerange(2, B1 + 1):
        k *= pow(p, int(log(B1, p)))

    # Pre-calculate the prime numbers to be used in stage 2.
    # Using the fact that the x-coordinates of point P and its
    # inverse -P coincide, the number of primes to be checked
    # in stage 2 can be reduced.
    deltas_list = []
    for r in range(B1 + 2*D, B2 + 2*D, 4*D):
        # d in deltas iff r+(2d+1) and/or r-(2d+1) is prime
        deltas = {abs(q - r) >> 1 for q in primerange(r - 2*D, r + 2*D)}
        deltas_list.append(list(deltas))

    for _ in range(max_curve):
        #Suyama's Parametrization
        sigma = randint(6, n - 1)
        u = (sigma**2 - 5) % n
        v = (4*sigma) % n
        u_3 = pow(u, 3, n)

        try:
            # We use the elliptic curve y**2 = x**3 + a*x**2 + x
            # where a = pow(v - u, 3, n)*(3*u + v)*invert(4*u_3*v, n) - 2
            # However, we do not declare a because it is more convenient
            # to use a24 = (a + 2)*invert(4, n) in the calculation.
            a24 = pow(v - u, 3, n)*(3*u + v)*invert(16*u_3*v, n) % n
        except ZeroDivisionError:
            #If the invert(16*u_3*v, n) doesn't exist (i.e., g != 1)
            g = gcd(2*u_3*v, n)
            #If g = n, try another curve
            if g == n:
                continue
            return g

        Q = Point(u_3, pow(v, 3, n), a24, n)
        Q = Q.mont_ladder(k)
        g = gcd(Q.z_cord, n)

        #Stage 1 factor
        if g != 1 and g != n:
            return g
        #Stage 1 failure. Q.z = 0, Try another curve
        elif g == n:
            continue

        #Stage 2 - Improved Standard Continuation
        S[0] = Q
        Q2 = Q.double()
        S[1] = Q2.add(Q, Q)
        beta[0] = (S[0].x_cord*S[0].z_cord) % n
        beta[1] = (S[1].x_cord*S[1].z_cord) % n
        for d in range(2, D):
            S[d] = S[d - 1].add(Q2, S[d - 2])
            beta[d] = (S[d].x_cord*S[d].z_cord) % n
        # i.e., S[i] = Q.mont_ladder(2*i + 1)

        g = 1
        W = Q.mont_ladder(4*D)
        T = Q.mont_ladder(B1 - 2*D)
        R = Q.mont_ladder(B1 + 2*D)
        for deltas in deltas_list:
            # R = Q.mont_ladder(r) where r in range(B1 + 2*D, B2 + 2*D, 4*D)
            alpha = (R.x_cord*R.z_cord) % n
            for delta in deltas:
                # We want to calculate
                # f = R.x_cord * S[delta].z_cord - S[delta].x_cord * R.z_cord
                f = (R.x_cord - S[delta].x_cord)*\
                    (R.z_cord + S[delta].z_cord) - alpha + beta[delta]
                g = (g*f) % n
            T, R = R, R.add(W, T)
        g = gcd(n, g)

        #Stage 2 Factor found
        if g != 1 and g != n:
            return g


def ecm(n, B1=10000, B2=100000, max_curve=200, seed=1234):
    """Performs factorization using Lenstra's Elliptic curve method.

    This function repeatedly calls ``_ecm_one_factor`` to compute the factors
    of n. First all the small factors are taken out using trial division.
    Then ``_ecm_one_factor`` is used to compute one factor at a time.

    Parameters
    ==========

    n : Number to be Factored
    B1 : Stage 1 Bound. Must be an even number.
    B2 : Stage 2 Bound. Must be an even number.
    max_curve : Maximum number of curves generated
    seed : Initialize pseudorandom generator

    Examples
    ========

    >>> from sympy.ntheory import ecm
    >>> ecm(25645121643901801)
    {5394769, 4753701529}
    >>> ecm(9804659461513846513)
    {4641991, 2112166839943}
    """
    from .factor_ import _perfect_power
    n = as_int(n)
    if B1 % 2 != 0 or B2 % 2 != 0:
        raise ValueError("both bounds must be even")
    TF_LIMIT = 100000
    factors = set()
    for prime in sieve.primerange(2, TF_LIMIT):
        if n % prime == 0:
            factors.add(prime)
            while(n % prime == 0):
                n //= prime

    queue = []
    def check(m):
        if isprime(m):
            factors.add(m)
            return
        if result := _perfect_power(m, TF_LIMIT):
            return check(result[0])
        queue.append(m)
    check(n)
    while queue:
        n = queue.pop()
        factor = _ecm_one_factor(n, B1, B2, max_curve, seed)
        if factor is None:
            raise ValueError("Increase the bounds")
        check(factor)
        check(n // factor)
    return factors
