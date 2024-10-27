"""
Compute Galois groups of polynomials.

We use algorithms from [1], with some modifications to use lookup tables for
resolvents.

References
==========

.. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.

"""

from collections import defaultdict
import random

from sympy.core.symbol import Dummy, symbols
from sympy.ntheory.primetest import is_square
from sympy.polys.domains import ZZ
from sympy.polys.densebasic import dup_random
from sympy.polys.densetools import dup_eval
from sympy.polys.euclidtools import dup_discriminant
from sympy.polys.factortools import dup_factor_list, dup_irreducible_p
from sympy.polys.numberfields.galois_resolvents import (
    GaloisGroupException, get_resolvent_by_lookup, define_resolvents,
    Resolvent,
)
from sympy.polys.numberfields.utilities import coeff_search
from sympy.polys.polytools import (Poly, poly_from_expr,
                                   PolificationFailed, ComputationFailed)
from sympy.polys.sqfreetools import dup_sqf_p
from sympy.utilities import public


class MaxTriesException(GaloisGroupException):
    ...


def tschirnhausen_transformation(T, max_coeff=10, max_tries=30, history=None,
                                 fixed_order=True):
    r"""
    Given a univariate, monic, irreducible polynomial over the integers, find
    another such polynomial defining the same number field.

    Explanation
    ===========

    See Alg 6.3.4 of [1].

    Parameters
    ==========

    T : Poly
        The given polynomial
    max_coeff : int
        When choosing a transformation as part of the process,
        keep the coeffs between plus and minus this.
    max_tries : int
        Consider at most this many transformations.
    history : set, None, optional (default=None)
        Pass a set of ``Poly.rep``'s in order to prevent any of these
        polynomials from being returned as the polynomial ``U`` i.e. the
        transformation of the given polynomial *T*. The given poly *T* will
        automatically be added to this set, before we try to find a new one.
    fixed_order : bool, default True
        If ``True``, work through candidate transformations A(x) in a fixed
        order, from small coeffs to large, resulting in deterministic behavior.
        If ``False``, the A(x) are chosen randomly, while still working our way
        up from small coefficients to larger ones.

    Returns
    =======

    Pair ``(A, U)``

        ``A`` and ``U`` are ``Poly``, ``A`` is the
        transformation, and ``U`` is the transformed polynomial that defines
        the same number field as *T*. The polynomial ``A`` maps the roots of
        *T* to the roots of ``U``.

    Raises
    ======

    MaxTriesException
        if could not find a polynomial before exceeding *max_tries*.

    """
    X = Dummy('X')
    n = T.degree()
    if history is None:
        history = set()
    history.add(T.rep)

    if fixed_order:
        coeff_generators = {}
        deg_coeff_sum = 3
        current_degree = 2

    def get_coeff_generator(degree):
        gen = coeff_generators.get(degree, coeff_search(degree, 1))
        coeff_generators[degree] = gen
        return gen

    for i in range(max_tries):

        # We never use linear A(x), since applying a fixed linear transformation
        # to all roots will only multiply the discriminant of T by a square
        # integer. This will change nothing important. In particular, if disc(T)
        # was zero before, it will still be zero now, and typically we apply
        # the transformation in hopes of replacing T by a squarefree poly.

        if fixed_order:
            # If d is degree and c max coeff, we move through the dc-space
            # along lines of constant sum. First d + c = 3 with (d, c) = (2, 1).
            # Then d + c = 4 with (d, c) = (3, 1), (2, 2). Then d + c = 5 with
            # (d, c) = (4, 1), (3, 2), (2, 3), and so forth. For a given (d, c)
            # we go though all sets of coeffs where max = c, before moving on.
            gen = get_coeff_generator(current_degree)
            coeffs = next(gen)
            m = max(abs(c) for c in coeffs)
            if current_degree + m > deg_coeff_sum:
                if current_degree == 2:
                    deg_coeff_sum += 1
                    current_degree = deg_coeff_sum - 1
                else:
                    current_degree -= 1
                gen = get_coeff_generator(current_degree)
                coeffs = next(gen)
            a = [ZZ(1)] + [ZZ(c) for c in coeffs]

        else:
            # We use a progressive coeff bound, up to the max specified, since it
            # is preferable to succeed with smaller coeffs.
            # Give each coeff bound five tries, before incrementing.
            C = min(i//5 + 1, max_coeff)
            d = random.randint(2, n - 1)
            a = dup_random(d, -C, C, ZZ)

        A = Poly(a, T.gen)
        U = Poly(T.resultant(X - A), X)
        if U.rep not in history and dup_sqf_p(U.rep.to_list(), ZZ):
            return A, U
    raise MaxTriesException


def has_square_disc(T):
    """Convenience to check if a Poly or dup has square discriminant. """
    d = T.discriminant() if isinstance(T, Poly) else dup_discriminant(T, ZZ)
    return is_square(d)


def _galois_group_degree_3(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 3.

    Explanation
    ===========

    Uses Prop 6.3.5 of [1].

    """
    from sympy.combinatorics.galois import S3TransitiveSubgroups
    return ((S3TransitiveSubgroups.A3, True) if has_square_disc(T)
            else (S3TransitiveSubgroups.S3, False))


def _galois_group_degree_4_root_approx(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 4.

    Explanation
    ===========

    Follows Alg 6.3.7 of [1], using a pure root approximation approach.

    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.galois import S4TransitiveSubgroups

    X = symbols('X0 X1 X2 X3')
    # We start by considering the resolvent for the form
    #   F = X0*X2 + X1*X3
    # and the group G = S4. In this case, the stabilizer H is D4 = < (0123), (02) >,
    # and a set of representatives of G/H is {I, (01), (03)}
    F1 = X[0]*X[2] + X[1]*X[3]
    s1 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 3)
    ]
    R1 = Resolvent(F1, X, s1)

    # In the second half of the algorithm (if we reach it), we use another
    # form and set of coset representatives. However, we may need to permute
    # them first, so cannot form their resolvent now.
    F2_pre = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[0]**2
    s2_pre = [
        Permutation(3),
        Permutation(3)(0, 2)
    ]

    history = set()
    for i in range(max_tries):
        if i > 0:
            # If we're retrying, need a new polynomial T.
            _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                                history=history,
                                                fixed_order=not randomize)

        R_dup, _, i0 = R1.eval_for_poly(T, find_integer_root=True)
        # If R is not squarefree, must retry.
        if not dup_sqf_p(R_dup, ZZ):
            continue

        # By Prop 6.3.1 of [1], Gal(T) is contained in A4 iff disc(T) is square.
        sq_disc = has_square_disc(T)

        if i0 is None:
            # By Thm 6.3.3 of [1], Gal(T) is not conjugate to any subgroup of the
            # stabilizer H = D4 that we chose. This means Gal(T) is either A4 or S4.
            return ((S4TransitiveSubgroups.A4, True) if sq_disc
                    else (S4TransitiveSubgroups.S4, False))

        # Gal(T) is conjugate to a subgroup of H = D4, so it is either V, C4
        # or D4 itself.

        if sq_disc:
            # Neither C4 nor D4 is contained in A4, so Gal(T) must be V.
            return (S4TransitiveSubgroups.V, True)

        # Gal(T) can only be D4 or C4.
        # We will now use our second resolvent, with G being that conjugate of D4 that
        # Gal(T) is contained in. To determine the right conjugate, we will need
        # the permutation corresponding to the integer root we found.
        sigma = s1[i0]
        # Applying sigma means permuting the args of F, and
        # conjugating the set of coset representatives.
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma*tau*sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        R_dup, _, _ = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)
        # If d is zero (R has a repeated root), must retry.
        if d == 0:
            continue
        if is_square(d):
            return (S4TransitiveSubgroups.C4, False)
        else:
            return (S4TransitiveSubgroups.D4, False)

    raise MaxTriesException


def _galois_group_degree_4_lookup(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 4.

    Explanation
    ===========

    Based on Alg 6.3.6 of [1], but uses resolvent coeff lookup.

    """
    from sympy.combinatorics.galois import S4TransitiveSubgroups

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 0)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    # Compute list L of degrees of irreducible factors of R, in increasing order:
    fl = dup_factor_list(R_dup, ZZ)
    L = sorted(sum([
        [len(r) - 1] * e for r, e in fl[1]
    ], []))

    if L == [6]:
        return ((S4TransitiveSubgroups.A4, True) if has_square_disc(T)
            else (S4TransitiveSubgroups.S4, False))

    if L == [1, 1, 4]:
        return (S4TransitiveSubgroups.C4, False)

    if L == [2, 2, 2]:
        return (S4TransitiveSubgroups.V, True)

    assert L == [2, 4]
    return (S4TransitiveSubgroups.D4, False)


def _galois_group_degree_5_hybrid(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 5.

    Explanation
    ===========

    Based on Alg 6.3.9 of [1], but uses a hybrid approach, combining resolvent
    coeff lookup, with root approximation.

    """
    from sympy.combinatorics.galois import S5TransitiveSubgroups
    from sympy.combinatorics.permutations import Permutation

    X5 = symbols("X0,X1,X2,X3,X4")
    res = define_resolvents()
    F51, _, s51 = res[(5, 1)]
    F51 = F51.as_expr(*X5)
    R51 = Resolvent(F51, X5, s51)

    history = set()
    reached_second_stage = False
    for i in range(max_tries):
        if i > 0:
            _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                                history=history,
                                                fixed_order=not randomize)
        R51_dup = get_resolvent_by_lookup(T, 1)
        if not dup_sqf_p(R51_dup, ZZ):
            continue

        # First stage
        # If we have not yet reached the second stage, then the group still
        # might be S5, A5, or M20, so must test for that.
        if not reached_second_stage:
            sq_disc = has_square_disc(T)

            if dup_irreducible_p(R51_dup, ZZ):
                return ((S5TransitiveSubgroups.A5, True) if sq_disc
                        else (S5TransitiveSubgroups.S5, False))

            if not sq_disc:
                return (S5TransitiveSubgroups.M20, False)

        # Second stage
        reached_second_stage = True
        # R51 must have an integer root for T.
        # To choose our second resolvent, we need to know which conjugate of
        # F51 is a root.
        rounded_roots = R51.round_roots_to_integers_for_poly(T)
        # These are integers, and candidates to be roots of R51.
        # We find the first one that actually is a root.
        for permutation_index, candidate_root in rounded_roots.items():
            if not dup_eval(R51_dup, candidate_root, ZZ):
                break

        X = X5
        F2_pre = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[4]**2 + X[4]*X[0]**2
        s2_pre = [
            Permutation(4),
            Permutation(4)(0, 1)(2, 4)
        ]

        i0 = permutation_index
        sigma = s51[i0]
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma*tau*sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        R_dup, _, _ = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)

        if d == 0:
            continue
        if is_square(d):
            return (S5TransitiveSubgroups.C5, True)
        else:
            return (S5TransitiveSubgroups.D5, True)

    raise MaxTriesException


def _galois_group_degree_5_lookup_ext_factor(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 5.

    Explanation
    ===========

    Based on Alg 6.3.9 of [1], but uses resolvent coeff lookup, plus
    factorization over an algebraic extension.

    """
    from sympy.combinatorics.galois import S5TransitiveSubgroups

    _T = T

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 1)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    sq_disc = has_square_disc(T)

    if dup_irreducible_p(R_dup, ZZ):
        return ((S5TransitiveSubgroups.A5, True) if sq_disc
                else (S5TransitiveSubgroups.S5, False))

    if not sq_disc:
        return (S5TransitiveSubgroups.M20, False)

    # If we get this far, Gal(T) can only be D5 or C5.
    # But for Gal(T) to have order 5, T must already split completely in
    # the extension field obtained by adjoining a single one of its roots.
    fl = Poly(_T, domain=ZZ.alg_field_from_poly(_T)).factor_list()[1]
    if len(fl) == 5:
        return (S5TransitiveSubgroups.C5, True)
    else:
        return (S5TransitiveSubgroups.D5, True)


def _galois_group_degree_6_lookup(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 6.

    Explanation
    ===========

    Based on Alg 6.3.10 of [1], but uses resolvent coeff lookup.

    """
    from sympy.combinatorics.galois import S6TransitiveSubgroups

    # First resolvent:

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 1)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    fl = dup_factor_list(R_dup, ZZ)

    # Group the factors by degree.
    factors_by_deg = defaultdict(list)
    for r, _ in fl[1]:
        factors_by_deg[len(r) - 1].append(r)

    L = sorted(sum([
        [d] * len(ff) for d, ff in factors_by_deg.items()
    ], []))

    T_has_sq_disc = has_square_disc(T)

    if L == [1, 2, 3]:
        f1 = factors_by_deg[3][0]
        return ((S6TransitiveSubgroups.C6, False) if has_square_disc(f1)
                else (S6TransitiveSubgroups.D6, False))

    elif L == [3, 3]:
        f1, f2 = factors_by_deg[3]
        any_square = has_square_disc(f1) or has_square_disc(f2)
        return ((S6TransitiveSubgroups.G18, False) if any_square
                else (S6TransitiveSubgroups.G36m, False))

    elif L == [2, 4]:
        if T_has_sq_disc:
            return (S6TransitiveSubgroups.S4p, True)
        else:
            f1 = factors_by_deg[4][0]
            return ((S6TransitiveSubgroups.A4xC2, False) if has_square_disc(f1)
                    else (S6TransitiveSubgroups.S4xC2, False))

    elif L == [1, 1, 4]:
        return ((S6TransitiveSubgroups.A4, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.S4m, False))

    elif L == [1, 5]:
        return ((S6TransitiveSubgroups.PSL2F5, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.PGL2F5, False))

    elif L == [1, 1, 1, 3]:
        return (S6TransitiveSubgroups.S3, False)

    assert L == [6]

    # Second resolvent:

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 2)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    T_has_sq_disc = has_square_disc(T)

    if dup_irreducible_p(R_dup, ZZ):
        return ((S6TransitiveSubgroups.A6, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.S6, False))
    else:
        return ((S6TransitiveSubgroups.G36p, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.G72, False))


@public
def galois_group(f, *gens, by_name=False, max_tries=30, randomize=False, **args):
    r"""
    Compute the Galois group for polynomials *f* up to degree 6.

    Examples
    ========

    >>> from sympy import galois_group
    >>> from sympy.abc import x
    >>> f = x**4 + 1
    >>> G, alt = galois_group(f)
    >>> print(G)
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    The group is returned along with a boolean, indicating whether it is
    contained in the alternating group $A_n$, where $n$ is the degree of *T*.
    Along with other group properties, this can help determine which group it
    is:

    >>> alt
    True
    >>> G.order()
    4

    Alternatively, the group can be returned by name:

    >>> G_name, _ = galois_group(f, by_name=True)
    >>> print(G_name)
    S4TransitiveSubgroups.V

    The group itself can then be obtained by calling the name's
    ``get_perm_group()`` method:

    >>> G_name.get_perm_group()
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    Group names are values of the enum classes
    :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`,
    :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`,
    etc.

    Parameters
    ==========

    f : Expr
        Irreducible polynomial over :ref:`ZZ` or :ref:`QQ`, whose Galois group
        is to be determined.
    gens : optional list of symbols
        For converting *f* to Poly, and will be passed on to the
        :py:func:`~.poly_from_expr` function.
    by_name : bool, default False
        If ``True``, the Galois group will be returned by name.
        Otherwise it will be returned as a :py:class:`~.PermutationGroup`.
    max_tries : int, default 30
        Make at most this many attempts in those steps that involve
        generating Tschirnhausen transformations.
    randomize : bool, default False
        If ``True``, then use random coefficients when generating Tschirnhausen
        transformations. Otherwise try transformations in a fixed order. Both
        approaches start with small coefficients and degrees and work upward.
    args : optional
        For converting *f* to Poly, and will be passed on to the
        :py:func:`~.poly_from_expr` function.

    Returns
    =======

    Pair ``(G, alt)``
        The first element ``G`` indicates the Galois group. It is an instance
        of one of the :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`
        :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`, etc. enum
        classes if *by_name* was ``True``, and a :py:class:`~.PermutationGroup`
        if ``False``.

        The second element is a boolean, saying whether the group is contained
        in the alternating group $A_n$ ($n$ the degree of *T*).

    Raises
    ======

    ValueError
        if *f* is of an unsupported degree.

    MaxTriesException
        if could not complete before exceeding *max_tries* in those steps
        that involve generating Tschirnhausen transformations.

    See Also
    ========

    .Poly.galois_group

    """
    gens = gens or []
    args = args or {}

    try:
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('galois_group', 1, exc)

    return F.galois_group(by_name=by_name, max_tries=max_tries,
                          randomize=randomize)
