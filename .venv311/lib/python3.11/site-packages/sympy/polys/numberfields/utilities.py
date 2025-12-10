"""Utilities for algebraic number theory. """

from sympy.core.sympify import sympify
from sympy.ntheory.factor_ import factorint
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.minpoly import minpoly
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.utilities.decorator import public
from sympy.utilities.lambdify import lambdify

from mpmath import mp


def is_rat(c):
    r"""
    Test whether an argument is of an acceptable type to be used as a rational
    number.

    Explanation
    ===========

    Returns ``True`` on any argument of type ``int``, :ref:`ZZ`, or :ref:`QQ`.

    See Also
    ========

    is_int

    """
    # ``c in QQ`` is too accepting (e.g. ``3.14 in QQ`` is ``True``),
    # ``QQ.of_type(c)`` is too demanding (e.g. ``QQ.of_type(3)`` is ``False``).
    #
    # Meanwhile, if gmpy2 is installed then ``ZZ.of_type()`` accepts only
    # ``mpz``, not ``int``, so we need another clause to ensure ``int`` is
    # accepted.
    return isinstance(c, int) or ZZ.of_type(c) or QQ.of_type(c)


def is_int(c):
    r"""
    Test whether an argument is of an acceptable type to be used as an integer.

    Explanation
    ===========

    Returns ``True`` on any argument of type ``int`` or :ref:`ZZ`.

    See Also
    ========

    is_rat

    """
    # If gmpy2 is installed then ``ZZ.of_type()`` accepts only
    # ``mpz``, not ``int``, so we need another clause to ensure ``int`` is
    # accepted.
    return isinstance(c, int) or ZZ.of_type(c)


def get_num_denom(c):
    r"""
    Given any argument on which :py:func:`~.is_rat` is ``True``, return the
    numerator and denominator of this number.

    See Also
    ========

    is_rat

    """
    r = QQ(c)
    return r.numerator, r.denominator


@public
def extract_fundamental_discriminant(a):
    r"""
    Extract a fundamental discriminant from an integer *a*.

    Explanation
    ===========

    Given any rational integer *a* that is 0 or 1 mod 4, write $a = d f^2$,
    where $d$ is either 1 or a fundamental discriminant, and return a pair
    of dictionaries ``(D, F)`` giving the prime factorizations of $d$ and $f$
    respectively, in the same format returned by :py:func:`~.factorint`.

    A fundamental discriminant $d$ is different from unity, and is either
    1 mod 4 and squarefree, or is 0 mod 4 and such that $d/4$ is squarefree
    and 2 or 3 mod 4. This is the same as being the discriminant of some
    quadratic field.

    Examples
    ========

    >>> from sympy.polys.numberfields.utilities import extract_fundamental_discriminant
    >>> print(extract_fundamental_discriminant(-432))
    ({3: 1, -1: 1}, {2: 2, 3: 1})

    For comparison:

    >>> from sympy import factorint
    >>> print(factorint(-432))
    {2: 4, 3: 3, -1: 1}

    Parameters
    ==========

    a: int, must be 0 or 1 mod 4

    Returns
    =======

    Pair ``(D, F)``  of dictionaries.

    Raises
    ======

    ValueError
        If *a* is not 0 or 1 mod 4.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Prop. 5.1.3)

    """
    if a % 4 not in [0, 1]:
        raise ValueError('To extract fundamental discriminant, number must be 0 or 1 mod 4.')
    if a == 0:
        return {}, {0: 1}
    if a == 1:
        return {}, {}
    a_factors = factorint(a)
    D = {}
    F = {}
    # First pass: just make d squarefree, and a/d a perfect square.
    # We'll count primes (and units! i.e. -1) that are 3 mod 4 and present in d.
    num_3_mod_4 = 0
    for p, e in a_factors.items():
        if e % 2 == 1:
            D[p] = 1
            if p % 4 == 3:
                num_3_mod_4 += 1
            if e >= 3:
                F[p] = (e - 1) // 2
        else:
            F[p] = e // 2
    # Second pass: if d is cong. to 2 or 3 mod 4, then we must steal away
    # another factor of 4 from f**2 and give it to d.
    even = 2 in D
    if even or num_3_mod_4 % 2 == 1:
        e2 = F[2]
        assert e2 > 0
        if e2 == 1:
            del F[2]
        else:
            F[2] = e2 - 1
        D[2] = 3 if even else 2
    return D, F


@public
class AlgIntPowers:
    r"""
    Compute the powers of an algebraic integer.

    Explanation
    ===========

    Given an algebraic integer $\theta$ by its monic irreducible polynomial
    ``T`` over :ref:`ZZ`, this class computes representations of arbitrarily
    high powers of $\theta$, as :ref:`ZZ`-linear combinations over
    $\{1, \theta, \ldots, \theta^{n-1}\}$, where $n = \deg(T)$.

    The representations are computed using the linear recurrence relations for
    powers of $\theta$, derived from the polynomial ``T``. See [1], Sec. 4.2.2.

    Optionally, the representations may be reduced with respect to a modulus.

    Examples
    ========

    >>> from sympy import Poly, cyclotomic_poly
    >>> from sympy.polys.numberfields.utilities import AlgIntPowers
    >>> T = Poly(cyclotomic_poly(5))
    >>> zeta_pow = AlgIntPowers(T)
    >>> print(zeta_pow[0])
    [1, 0, 0, 0]
    >>> print(zeta_pow[1])
    [0, 1, 0, 0]
    >>> print(zeta_pow[4])  # doctest: +SKIP
    [-1, -1, -1, -1]
    >>> print(zeta_pow[24])  # doctest: +SKIP
    [-1, -1, -1, -1]

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*

    """

    def __init__(self, T, modulus=None):
        """
        Parameters
        ==========

        T : :py:class:`~.Poly`
            The monic irreducible polynomial over :ref:`ZZ` defining the
            algebraic integer.

        modulus : int, None, optional
            If not ``None``, all representations will be reduced w.r.t. this.

        """
        self.T = T
        self.modulus = modulus
        self.n = T.degree()
        self.powers_n_and_up = [[-c % self for c in reversed(T.rep.to_list())][:-1]]
        self.max_so_far = self.n

    def red(self, exp):
        return exp if self.modulus is None else exp % self.modulus

    def __rmod__(self, other):
        return self.red(other)

    def compute_up_through(self, e):
        m = self.max_so_far
        if e <= m: return
        n = self.n
        r = self.powers_n_and_up
        c = r[0]
        for k in range(m+1, e+1):
            b = r[k-1-n][n-1]
            r.append(
                [c[0]*b % self] + [
                    (r[k-1-n][i-1] + c[i]*b) % self for i in range(1, n)
                ]
            )
        self.max_so_far = e

    def get(self, e):
        n = self.n
        if e < 0:
            raise ValueError('Exponent must be non-negative.')
        elif e < n:
            return [1 if i == e else 0 for i in range(n)]
        else:
            self.compute_up_through(e)
            return self.powers_n_and_up[e - n]

    def __getitem__(self, item):
        return self.get(item)


@public
def coeff_search(m, R):
    r"""
    Generate coefficients for searching through polynomials.

    Explanation
    ===========

    Lead coeff is always non-negative. Explore all combinations with coeffs
    bounded in absolute value before increasing the bound. Skip the all-zero
    list, and skip any repeats. See examples.

    Examples
    ========

    >>> from sympy.polys.numberfields.utilities import coeff_search
    >>> cs = coeff_search(2, 1)
    >>> C = [next(cs) for i in range(13)]
    >>> print(C)
    [[1, 1], [1, 0], [1, -1], [0, 1], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2],
     [1, 2], [1, -2], [0, 2], [3, 3]]

    Parameters
    ==========

    m : int
        Length of coeff list.
    R : int
        Initial max abs val for coeffs (will increase as search proceeds).

    Returns
    =======

    generator
        Infinite generator of lists of coefficients.

    """
    R0 = R
    c = [R] * m
    while True:
        if R == R0 or R in c or -R in c:
            yield c[:]
        j = m - 1
        while c[j] == -R:
            j -= 1
        c[j] -= 1
        for i in range(j + 1, m):
            c[i] = R
        for j in range(m):
            if c[j] != 0:
                break
        else:
            R += 1
            c = [R] * m


def supplement_a_subspace(M):
    r"""
    Extend a basis for a subspace to a basis for the whole space.

    Explanation
    ===========

    Given an $n \times r$ matrix *M* of rank $r$ (so $r \leq n$), this function
    computes an invertible $n \times n$ matrix $B$ such that the first $r$
    columns of $B$ equal *M*.

    This operation can be interpreted as a way of extending a basis for a
    subspace, to give a basis for the whole space.

    To be precise, suppose you have an $n$-dimensional vector space $V$, with
    basis $\{v_1, v_2, \ldots, v_n\}$, and an $r$-dimensional subspace $W$ of
    $V$, spanned by a basis $\{w_1, w_2, \ldots, w_r\}$, where the $w_j$ are
    given as linear combinations of the $v_i$. If the columns of *M* represent
    the $w_j$ as such linear combinations, then the columns of the matrix $B$
    computed by this function give a new basis $\{u_1, u_2, \ldots, u_n\}$ for
    $V$, again relative to the $\{v_i\}$ basis, and such that $u_j = w_j$
    for $1 \leq j \leq r$.

    Examples
    ========

    Note: The function works in terms of columns, so in these examples we
    print matrix transposes in order to make the columns easier to inspect.

    >>> from sympy.polys.matrices import DM
    >>> from sympy import QQ, FF
    >>> from sympy.polys.numberfields.utilities import supplement_a_subspace
    >>> M = DM([[1, 7, 0], [2, 3, 4]], QQ).transpose()
    >>> print(supplement_a_subspace(M).to_Matrix().transpose())
    Matrix([[1, 7, 0], [2, 3, 4], [1, 0, 0]])

    >>> M2 = M.convert_to(FF(7))
    >>> print(M2.to_Matrix().transpose())
    Matrix([[1, 0, 0], [2, 3, -3]])
    >>> print(supplement_a_subspace(M2).to_Matrix().transpose())
    Matrix([[1, 0, 0], [2, 3, -3], [0, 1, 0]])

    Parameters
    ==========

    M : :py:class:`~.DomainMatrix`
        The columns give the basis for the subspace.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        This matrix is invertible and its first $r$ columns equal *M*.

    Raises
    ======

    DMRankError
        If *M* was not of maximal rank.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*
       (See Sec. 2.3.2.)

    """
    n, r = M.shape
    # Let In be the n x n identity matrix.
    # Form the augmented matrix [M | In] and compute RREF.
    Maug = M.hstack(M.eye(n, M.domain))
    R, pivots = Maug.rref()
    if pivots[:r] != tuple(range(r)):
        raise DMRankError('M was not of maximal rank')
    # Let J be the n x r matrix equal to the first r columns of In.
    # Since M is of rank r, RREF reduces [M | In] to [J | A], where A is the product of
    # elementary matrices Ei corresp. to the row ops performed by RREF. Since the Ei are
    # invertible, so is A. Let B = A^(-1).
    A = R[:, r:]
    B = A.inv()
    # Then B is the desired matrix. It is invertible, since B^(-1) == A.
    # And A * [M | In] == [J | A]
    #  => A * M == J
    #  => M == B * J == the first r columns of B.
    return B


@public
def isolate(alg, eps=None, fast=False):
    """
    Find a rational isolating interval for a real algebraic number.

    Examples
    ========

    >>> from sympy import isolate, sqrt, Rational
    >>> print(isolate(sqrt(2)))  # doctest: +SKIP
    (1, 2)
    >>> print(isolate(sqrt(2), eps=Rational(1, 100)))
    (24/17, 17/12)

    Parameters
    ==========

    alg : str, int, :py:class:`~.Expr`
        The algebraic number to be isolated. Must be a real number, to use this
        particular function. However, see also :py:meth:`.Poly.intervals`,
        which isolates complex roots when you pass ``all=True``.
    eps : positive element of :ref:`QQ`, None, optional (default=None)
        Precision to be passed to :py:meth:`.Poly.refine_root`
    fast : boolean, optional (default=False)
        Say whether fast refinement procedure should be used.
        (Will be passed to :py:meth:`.Poly.refine_root`.)

    Returns
    =======

    Pair of rational numbers defining an isolating interval for the given
    algebraic number.

    See Also
    ========

    .Poly.intervals

    """
    alg = sympify(alg)

    if alg.is_Rational:
        return (alg, alg)
    elif not alg.is_real:
        raise NotImplementedError(
            "complex algebraic numbers are not supported")

    func = lambdify((), alg, modules="mpmath", printer=IntervalPrinter())

    poly = minpoly(alg, polys=True)
    intervals = poly.intervals(sqf=True)

    dps, done = mp.dps, False

    try:
        while not done:
            alg = func()

            for a, b in intervals:
                if a <= alg.a and alg.b <= b:
                    done = True
                    break
            else:
                mp.dps *= 2
    finally:
        mp.dps = dps

    if eps is not None:
        a, b = poly.refine_root(a, b, eps=eps, fast=fast)

    return (a, b)
