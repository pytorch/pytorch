r"""
Galois resolvents

Each of the functions in ``sympy.polys.numberfields.galoisgroups`` that
computes Galois groups for a particular degree $n$ uses resolvents. Given the
polynomial $T$ whose Galois group is to be computed, a resolvent is a
polynomial $R$ whose roots are defined as functions of the roots of $T$.

One way to compute the coefficients of $R$ is by approximating the roots of $T$
to sufficient precision. This module defines a :py:class:`~.Resolvent` class
that handles this job, determining the necessary precision, and computing $R$.

In some cases, the coefficients of $R$ are symmetric in the roots of $T$,
meaning they are equal to fixed functions of the coefficients of $T$. Therefore
another approach is to compute these functions once and for all, and record
them in a lookup table. This module defines code that can compute such tables.
The tables for polynomials $T$ of degrees 4 through 6, produced by this code,
are recorded in the resolvent_lookup.py module.

"""

from sympy.core.evalf import (
    evalf, fastlog, _evalf_with_bounded_error, quad_to_mpmath,
)
from sympy.core.symbol import symbols, Dummy
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import ZZ
from sympy.polys.orderings import lex
from sympy.polys.polyroots import preprocess_roots
from sympy.polys.polytools import Poly
from sympy.polys.rings import xring
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.lambdify import lambdify

from mpmath import MPContext
from mpmath.libmp.libmpf import prec_to_dps


class GaloisGroupException(Exception):
    ...


class ResolventException(GaloisGroupException):
    ...


class Resolvent:
    r"""
    If $G$ is a subgroup of the symmetric group $S_n$,
    $F$ a multivariate polynomial in $\mathbb{Z}[X_1, \ldots, X_n]$,
    $H$ the stabilizer of $F$ in $G$ (i.e. the permutations $\sigma$ such that
    $F(X_{\sigma(1)}, \ldots, X_{\sigma(n)}) = F(X_1, \ldots, X_n)$), and $s$
    a set of left coset representatives of $H$ in $G$, then the resolvent
    polynomial $R(Y)$ is the product over $\sigma \in s$ of
    $Y - F(X_{\sigma(1)}, \ldots, X_{\sigma(n)})$.

    For example, consider the resolvent for the form
    $$F = X_0 X_2 + X_1 X_3$$
    and the group $G = S_4$. In this case, the stabilizer $H$ is the dihedral
    group $D4 = < (0123), (02) >$, and a set of representatives of $G/H$ is
    $\{I, (01), (03)\}$. The resolvent can be constructed as follows:

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.core.symbol import symbols
    >>> from sympy.polys.numberfields.galoisgroups import Resolvent
    >>> X = symbols('X0 X1 X2 X3')
    >>> F = X[0]*X[2] + X[1]*X[3]
    >>> s = [Permutation([0, 1, 2, 3]), Permutation([1, 0, 2, 3]),
    ... Permutation([3, 1, 2, 0])]
    >>> R = Resolvent(F, X, s)

    This resolvent has three roots, which are the conjugates of ``F`` under the
    three permutations in ``s``:

    >>> R.root_lambdas[0](*X)
    X0*X2 + X1*X3
    >>> R.root_lambdas[1](*X)
    X0*X3 + X1*X2
    >>> R.root_lambdas[2](*X)
    X0*X1 + X2*X3

    Resolvents are useful for computing Galois groups. Given a polynomial $T$
    of degree $n$, we will use a resolvent $R$ where $Gal(T) \leq G \leq S_n$.
    We will then want to substitute the roots of $T$ for the variables $X_i$
    in $R$, and study things like the discriminant of $R$, and the way $R$
    factors over $\mathbb{Q}$.

    From the symmetry in $R$'s construction, and since $Gal(T) \leq G$, we know
    from Galois theory that the coefficients of $R$ must lie in $\mathbb{Z}$.
    This allows us to compute the coefficients of $R$ by approximating the
    roots of $T$ to sufficient precision, plugging these values in for the
    variables $X_i$ in the coefficient expressions of $R$, and then simply
    rounding to the nearest integer.

    In order to determine a sufficient precision for the roots of $T$, this
    ``Resolvent`` class imposes certain requirements on the form ``F``. It
    could be possible to design a different ``Resolvent`` class, that made
    different precision estimates, and different assumptions about ``F``.

    ``F`` must be homogeneous, and all terms must have unit coefficient.
    Furthermore, if $r$ is the number of terms in ``F``, and $t$ the total
    degree, and if $m$ is the number of conjugates of ``F``, i.e. the number
    of permutations in ``s``, then we require that $m < r 2^t$. Again, it is
    not impossible to work with forms ``F`` that violate these assumptions, but
    this ``Resolvent`` class requires them.

    Since determining the integer coefficients of the resolvent for a given
    polynomial $T$ is one of the main problems this class solves, we take some
    time to explain the precision bounds it uses.

    The general problem is:
    Given a multivariate polynomial $P \in \mathbb{Z}[X_1, \ldots, X_n]$, and a
    bound $M \in \mathbb{R}_+$, compute an $\varepsilon > 0$ such that for any
    complex numbers $a_1, \ldots, a_n$ with $|a_i| < M$, if the $a_i$ are
    approximated to within an accuracy of $\varepsilon$ by $b_i$, that is,
    $|a_i - b_i| < \varepsilon$ for $i = 1, \ldots, n$, then
    $|P(a_1, \ldots, a_n) - P(b_1, \ldots, b_n)| < 1/2$. In other words, if it
    is known that $P(a_1, \ldots, a_n) = c$ for some $c \in \mathbb{Z}$, then
    $P(b_1, \ldots, b_n)$ can be rounded to the nearest integer in order to
    determine $c$.

    To derive our error bound, consider the monomial $xyz$. Defining
    $d_i = b_i - a_i$, our error is
    $|(a_1 + d_1)(a_2 + d_2)(a_3 + d_3) - a_1 a_2 a_3|$, which is bounded
    above by $|(M + \varepsilon)^3 - M^3|$. Passing to a general monomial of
    total degree $t$, this expression is bounded by
    $M^{t-1}\varepsilon(t + 2^t\varepsilon/M)$ provided $\varepsilon < M$,
    and by $(t+1)M^{t-1}\varepsilon$ provided $\varepsilon < M/2^t$.
    But since our goal is to make the error less than $1/2$, we will choose
    $\varepsilon < 1/(2(t+1)M^{t-1})$, which implies the condition that
    $\varepsilon < M/2^t$, as long as $M \geq 2$.

    Passing from the general monomial to the general polynomial is easy, by
    scaling and summing error bounds.

    In our specific case, we are given a homogeneous polynomial $F$ of
    $r$ terms and total degree $t$, all of whose coefficients are $\pm 1$. We
    are given the $m$ permutations that make the conjugates of $F$, and
    we want to bound the error in the coefficients of the monic polynomial
    $R(Y)$ having $F$ and its conjugates as roots (i.e. the resolvent).

    For $j$ from $1$ to $m$, the coefficient of $Y^{m-j}$ in $R(Y)$ is the
    $j$th elementary symmetric polynomial in the conjugates of $F$. This sums
    the products of these conjugates, taken $j$ at a time, in all possible
    combinations. There are $\binom{m}{j}$ such combinations, and each product
    of $j$ conjugates of $F$ expands to a sum of $r^j$ terms, each of unit
    coefficient, and total degree $jt$. An error bound for the $j$th coeff of
    $R$ is therefore
    $$\binom{m}{j} r^j (jt + 1) M^{jt - 1} \varepsilon$$
    When our goal is to evaluate all the coefficients of $R$, we will want to
    use the maximum of these error bounds. It is clear that this bound is
    strictly increasing for $j$ up to the ceiling of $m/2$. After that point,
    the first factor $\binom{m}{j}$ begins to decrease, while the others
    continue to increase. However, the binomial coefficient never falls by more
    than a factor of $1/m$ at a time, so our assumptions that $M \geq 2$ and
    $m < r 2^t$ are enough to tell us that the constant coefficient of $R$,
    i.e. that where $j = m$, has the largest error bound. Therefore we can use
    $$r^m (mt + 1) M^{mt - 1} \varepsilon$$
    as our error bound for all the coefficients.

    Note that this bound is also (more than) adequate to determine whether any
    of the roots of $R$ is an integer. Each of these roots is a single
    conjugate of $F$, which contains less error than the trace, i.e. the
    coefficient of $Y^{m - 1}$. By rounding the roots of $R$ to the nearest
    integers, we therefore get all the candidates for integer roots of $R$. By
    plugging these candidates into $R$, we can check whether any of them
    actually is a root.

    Note: We take the definition of resolvent from Cohen, but the error bound
    is ours.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.
       (Def 6.3.2)

    """

    def __init__(self, F, X, s):
        r"""
        Parameters
        ==========

        F : :py:class:`~.Expr`
            polynomial in the symbols in *X*
        X : list of :py:class:`~.Symbol`
        s : list of :py:class:`~.Permutation`
            representing the cosets of the stabilizer of *F* in
            some subgroup $G$ of $S_n$, where $n$ is the length of *X*.
        """
        self.F = F
        self.X = X
        self.s = s

        # Number of conjugates:
        self.m = len(s)
        # Total degree of F (computed below):
        self.t = None
        # Number of terms in F (computed below):
        self.r = 0

        for monom, coeff in Poly(F).terms():
            if abs(coeff) != 1:
                raise ResolventException('Resolvent class expects forms with unit coeffs')
            t = sum(monom)
            if t != self.t and self.t is not None:
                raise ResolventException('Resolvent class expects homogeneous forms')
            self.t = t
            self.r += 1

        m, t, r = self.m, self.t, self.r
        if not m < r * 2**t:
            raise ResolventException('Resolvent class expects m < r*2^t')
        M = symbols('M')
        # Precision sufficient for computing the coeffs of the resolvent:
        self.coeff_prec_func = Poly(r**m*(m*t + 1)*M**(m*t - 1))
        # Precision sufficient for checking whether any of the roots of the
        # resolvent are integers:
        self.root_prec_func = Poly(r*(t + 1)*M**(t - 1))

        # The conjugates of F are the roots of the resolvent.
        # For evaluating these to required numerical precisions, we need
        # lambdified versions.
        # Note: for a given permutation sigma, the conjugate (sigma F) is
        # equivalent to lambda [sigma^(-1) X]: F.
        self.root_lambdas = [
            lambdify((~s[j])(X), F)
            for j in range(self.m)
        ]

        # For evaluating the coeffs, we'll also need lambdified versions of
        # the elementary symmetric functions for degree m.
        Y = symbols('Y')
        R = symbols(' '.join(f'R{i}' for i in range(m)))
        f = 1
        for r in R:
            f *= (Y - r)
        C = Poly(f, Y).coeffs()
        self.esf_lambdas = [lambdify(R, c) for c in C]

    def get_prec(self, M, target='coeffs'):
        r"""
        For a given upper bound *M* on the magnitude of the complex numbers to
        be plugged in for this resolvent's symbols, compute a sufficient
        precision for evaluating those complex numbers, such that the
        coefficients, or the integer roots, of the resolvent can be determined.

        Parameters
        ==========

        M : real number
            Upper bound on magnitude of the complex numbers to be plugged in.

        target : str, 'coeffs' or 'roots', default='coeffs'
            Name the task for which a sufficient precision is desired.
            This is either determining the coefficients of the resolvent
            ('coeffs') or determining its possible integer roots ('roots').
            The latter may require significantly lower precision.

        Returns
        =======

        int $m$
            such that $2^{-m}$ is a sufficient upper bound on the
            error in approximating the complex numbers to be plugged in.

        """
        # As explained in the docstring for this class, our precision estimates
        # require that M be at least 2.
        M = max(M, 2)
        f = self.coeff_prec_func if target == 'coeffs' else self.root_prec_func
        r, _, _, _ = evalf(2*f(M), 1, {})
        return fastlog(r) + 1

    def approximate_roots_of_poly(self, T, target='coeffs'):
        """
        Approximate the roots of a given polynomial *T* to sufficient precision
        in order to evaluate this resolvent's coefficients, or determine
        whether the resolvent has an integer root.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        target : str, 'coeffs' or 'roots', default='coeffs'
            Set the approximation precision to be sufficient for the desired
            task, which is either determining the coefficients of the resolvent
            ('coeffs') or determining its possible integer roots ('roots').
            The latter may require significantly lower precision.

        Returns
        =======

        list of elements of :ref:`CC`

        """
        ctx = MPContext()
        # Because sympy.polys.polyroots._integer_basis() is called when a CRootOf
        # is formed, we proactively extract the integer basis now. This means that
        # when we call T.all_roots(), every root will be a CRootOf, not a Mul
        # of Integer*CRootOf.
        coeff, T = preprocess_roots(T)
        coeff = ctx.mpf(str(coeff))

        scaled_roots = T.all_roots(radicals=False)

        # Since we're going to be approximating the roots of T anyway, we can
        # get a good upper bound on the magnitude of the roots by starting with
        # a very low precision approx.
        approx0 = [coeff * quad_to_mpmath(_evalf_with_bounded_error(r, m=0)) for r in scaled_roots]
        # Here we add 1 to account for the possible error in our initial approximation.
        M = max(abs(b) for b in approx0) + 1
        m = self.get_prec(M, target=target)
        n = fastlog(M._mpf_) + 1
        p = m + n + 1
        ctx.prec = p
        d = prec_to_dps(p)

        approx1 = [r.eval_approx(d, return_mpmath=True) for r in scaled_roots]
        approx1 = [coeff*ctx.mpc(r) for r in approx1]

        return approx1

    @staticmethod
    def round_mpf(a):
        if isinstance(a, int):
            return a
        # If we use python's built-in `round()`, we lose precision.
        # If we use `ZZ` directly, we may add or subtract 1.
        #
        # XXX: We have to convert to int before converting to ZZ because
        # flint.fmpz cannot convert a mpmath mpf.
        return ZZ(int(a.context.nint(a)))

    def round_roots_to_integers_for_poly(self, T):
        """
        For a given polynomial *T*, round the roots of this resolvent to the
        nearest integers.

        Explanation
        ===========

        None of the integers returned by this method is guaranteed to be a
        root of the resolvent; however, if the resolvent has any integer roots
        (for the given polynomial *T*), then they must be among these.

        If the coefficients of the resolvent are also desired, then this method
        should not be used. Instead, use the ``eval_for_poly`` method. This
        method may be significantly faster than ``eval_for_poly``.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        Returns
        =======

        dict
            Keys are the indices of those permutations in ``self.s`` such that
            the corresponding root did round to a rational integer.

            Values are :ref:`ZZ`.


        """
        approx_roots_of_T = self.approximate_roots_of_poly(T, target='roots')
        approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
        return {
            i: self.round_mpf(r.real)
            for i, r in enumerate(approx_roots_of_self)
            if self.round_mpf(r.imag) == 0
        }

    def eval_for_poly(self, T, find_integer_root=False):
        r"""
        Compute the integer values of the coefficients of this resolvent, when
        plugging in the roots of a given polynomial.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        find_integer_root : ``bool``, default ``False``
            If ``True``, then also determine whether the resolvent has an
            integer root, and return the first one found, along with its
            index, i.e. the index of the permutation ``self.s[i]`` it
            corresponds to.

        Returns
        =======

        Tuple ``(R, a, i)``

            ``R`` is this resolvent as a dense univariate polynomial over
            :ref:`ZZ`, i.e. a list of :ref:`ZZ`.

            If *find_integer_root* was ``True``, then ``a`` and ``i`` are the
            first integer root found, and its index, if one exists.
            Otherwise ``a`` and ``i`` are both ``None``.

        """
        approx_roots_of_T = self.approximate_roots_of_poly(T, target='coeffs')
        approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
        approx_coeffs_of_self = [c(*approx_roots_of_self) for c in self.esf_lambdas]

        R = []
        for c in approx_coeffs_of_self:
            if self.round_mpf(c.imag) != 0:
                # If precision was enough, this should never happen.
                raise ResolventException(f"Got non-integer coeff for resolvent: {c}")
            R.append(self.round_mpf(c.real))

        a0, i0 = None, None

        if find_integer_root:
            for i, r in enumerate(approx_roots_of_self):
                if self.round_mpf(r.imag) != 0:
                    continue
                if not dup_eval(R, (a := self.round_mpf(r.real)), ZZ):
                    a0, i0 = a, i
                    break

        return R, a0, i0


def wrap(text, width=80):
    """Line wrap a polynomial expression. """
    out = ''
    col = 0
    for c in text:
        if c == ' ' and col > width:
            c, col = '\n', 0
        else:
            col += 1
        out += c
    return out


def s_vars(n):
    """Form the symbols s1, s2, ..., sn to stand for elem. symm. polys. """
    return symbols([f's{i + 1}' for i in range(n)])


def sparse_symmetrize_resolvent_coeffs(F, X, s, verbose=False):
    """
    Compute the coefficients of a resolvent as functions of the coefficients of
    the associated polynomial.

    F must be a sparse polynomial.
    """
    import time, sys
    # Roots of resolvent as multivariate forms over vars X:
    root_forms = [
        F.compose(list(zip(X, sigma(X))))
        for sigma in s
    ]

    # Coeffs of resolvent (besides lead coeff of 1) as symmetric forms over vars X:
    Y = [Dummy(f'Y{i}') for i in range(len(s))]
    coeff_forms = []
    for i in range(1, len(s) + 1):
        if verbose:
            print('----')
            print(f'Computing symmetric poly of degree {i}...')
            sys.stdout.flush()
        t0 = time.time()
        G = symmetric_poly(i, *Y)
        t1 = time.time()
        if verbose:
            print(f'took {t1 - t0} seconds')
            print('lambdifying...')
            sys.stdout.flush()
        t0 = time.time()
        C = lambdify(Y, (-1)**i*G)
        t1 = time.time()
        if verbose:
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()
        coeff_forms.append(C)

    coeffs = []
    for i, f in enumerate(coeff_forms):
        if verbose:
            print('----')
            print(f'Plugging root forms into elem symm poly {i+1}...')
            sys.stdout.flush()
        t0 = time.time()
        g = f(*root_forms)
        t1 = time.time()
        coeffs.append(g)
        if verbose:
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()

    # Now symmetrize these coeffs. This means recasting them as polynomials in
    # the elementary symmetric polys over X.
    symmetrized = []
    symmetrization_times = []
    ss = s_vars(len(X))
    for i, A in list(enumerate(coeffs)):
        if verbose:
            print('-----')
            print(f'Coeff {i+1}...')
            sys.stdout.flush()
        t0 = time.time()
        B, rem, _ = A.symmetrize()
        t1 = time.time()
        if rem != 0:
            msg = f"Got nonzero remainder {rem} for resolvent (F, X, s) = ({F}, {X}, {s})"
            raise ResolventException(msg)
        B_str = str(B.as_expr(*ss))
        symmetrized.append(B_str)
        symmetrization_times.append(t1 - t0)
        if verbose:
            print(wrap(B_str))
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()

    return symmetrized, symmetrization_times


def define_resolvents():
    """Define all the resolvents for polys T of degree 4 through 6. """
    from sympy.combinatorics.galois import PGL2F5
    from sympy.combinatorics.permutations import Permutation

    R4, X4 = xring("X0,X1,X2,X3", ZZ, lex)
    X = X4

    # The one resolvent used in `_galois_group_degree_4_lookup()`:
    F40 = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[0]**2
    s40 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 2),
        Permutation(3)(0, 3),
        Permutation(3)(1, 2),
        Permutation(3)(2, 3),
    ]

    # First resolvent used in `_galois_group_degree_4_root_approx()`:
    F41 = X[0]*X[2] + X[1]*X[3]
    s41 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 3)
    ]

    R5, X5 = xring("X0,X1,X2,X3,X4", ZZ, lex)
    X = X5

    # First resolvent used in `_galois_group_degree_5_hybrid()`,
    # and only one used in `_galois_group_degree_5_lookup_ext_factor()`:
    F51 = (  X[0]**2*(X[1]*X[4] + X[2]*X[3])
           + X[1]**2*(X[2]*X[0] + X[3]*X[4])
           + X[2]**2*(X[3]*X[1] + X[4]*X[0])
           + X[3]**2*(X[4]*X[2] + X[0]*X[1])
           + X[4]**2*(X[0]*X[3] + X[1]*X[2]))
    s51 = [
        Permutation(4),
        Permutation(4)(0, 1),
        Permutation(4)(0, 2),
        Permutation(4)(0, 3),
        Permutation(4)(0, 4),
        Permutation(4)(1, 4)
    ]

    R6, X6 = xring("X0,X1,X2,X3,X4,X5", ZZ, lex)
    X = X6

    # First resolvent used in `_galois_group_degree_6_lookup()`:
    H = PGL2F5()
    term0 = X[0]**2*X[5]**2*(X[1]*X[4] + X[2]*X[3])
    terms = {term0.compose(list(zip(X, s(X)))) for s in H.elements}
    F61 = sum(terms)
    s61 = [Permutation(5)] + [Permutation(5)(0, n) for n in range(1, 6)]

    # Second resolvent used in `_galois_group_degree_6_lookup()`:
    F62 = X[0]*X[1]*X[2] + X[3]*X[4]*X[5]
    s62 = [Permutation(5)] + [
        Permutation(5)(i, j + 3) for i in range(3) for j in range(3)
    ]

    return {
        (4, 0): (F40, X4, s40),
        (4, 1): (F41, X4, s41),
        (5, 1): (F51, X5, s51),
        (6, 1): (F61, X6, s61),
        (6, 2): (F62, X6, s62),
    }


def generate_lambda_lookup(verbose=False, trial_run=False):
    """
    Generate the whole lookup table of coeff lambdas, for all resolvents.
    """
    jobs = define_resolvents()
    lambda_lists = {}
    total_time = 0
    time_for_61 = 0
    time_for_61_last = 0
    for k, (F, X, s) in jobs.items():
        symmetrized, times = sparse_symmetrize_resolvent_coeffs(F, X, s, verbose=verbose)

        total_time += sum(times)
        if k == (6, 1):
            time_for_61 = sum(times)
            time_for_61_last = times[-1]

        sv = s_vars(len(X))
        head = f'lambda {", ".join(str(v) for v in sv)}:'
        lambda_lists[k] = ',\n        '.join([
            f'{head} ({wrap(f)})'
            for f in symmetrized
        ])

        if trial_run:
            break

    table = (
         "# This table was generated by a call to\n"
         "# `sympy.polys.numberfields.galois_resolvents.generate_lambda_lookup()`.\n"
        f"# The entire job took {total_time:.2f}s.\n"
        f"# Of this, Case (6, 1) took {time_for_61:.2f}s.\n"
        f"# The final polynomial of Case (6, 1) alone took {time_for_61_last:.2f}s.\n"
         "resolvent_coeff_lambdas = {\n")

    for k, L in lambda_lists.items():
        table += f"    {k}: [\n"
        table +=  "        " + L + '\n'
        table +=  "    ],\n"
    table += "}\n"
    return table


def get_resolvent_by_lookup(T, number):
    """
    Use the lookup table, to return a resolvent (as dup) for a given
    polynomial *T*.

    Parameters
    ==========

    T : Poly
        The polynomial whose resolvent is needed

    number : int
        For some degrees, there are multiple resolvents.
        Use this to indicate which one you want.

    Returns
    =======

    dup

    """
    from sympy.polys.numberfields.resolvent_lookup import resolvent_coeff_lambdas
    degree = T.degree()
    L = resolvent_coeff_lambdas[(degree, number)]
    T_coeffs = T.rep.to_list()[1:]
    return [ZZ(1)] + [c(*T_coeffs) for c in L]


# Use
#   (.venv) $ python -m sympy.polys.numberfields.galois_resolvents
# to reproduce the table found in resolvent_lookup.py
if __name__ == "__main__":
    import sys
    verbose = '-v' in sys.argv[1:]
    trial_run = '-t' in sys.argv[1:]
    table = generate_lambda_lookup(verbose=verbose, trial_run=trial_run)
    print(table)
