r"""
Functions in ``polys.numberfields.subfield`` solve the "Subfield Problem" and
allied problems, for algebraic number fields.

Following Cohen (see [Cohen93]_ Section 4.5), we can define the main problem as
follows:

* **Subfield Problem:**

  Given two number fields $\mathbb{Q}(\alpha)$, $\mathbb{Q}(\beta)$
  via the minimal polynomials for their generators $\alpha$ and $\beta$, decide
  whether one field is isomorphic to a subfield of the other.

From a solution to this problem flow solutions to the following problems as
well:

* **Primitive Element Problem:**

  Given several algebraic numbers
  $\alpha_1, \ldots, \alpha_m$, compute a single algebraic number $\theta$
  such that $\mathbb{Q}(\alpha_1, \ldots, \alpha_m) = \mathbb{Q}(\theta)$.

* **Field Isomorphism Problem:**

  Decide whether two number fields
  $\mathbb{Q}(\alpha)$, $\mathbb{Q}(\beta)$ are isomorphic.

* **Field Membership Problem:**

  Given two algebraic numbers $\alpha$,
  $\beta$, decide whether $\alpha \in \mathbb{Q}(\beta)$, and if so write
  $\alpha = f(\beta)$ for some $f(x) \in \mathbb{Q}[x]$.
"""

from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.ntheory import sieve
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import QQ
from sympy.polys.numberfields.minpoly import _choose_factor, minimal_polynomial
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly, PurePoly, factor_list
from sympy.utilities import public

from mpmath import MPContext


def is_isomorphism_possible(a, b):
    """Necessary but not sufficient test for isomorphism. """
    n = a.minpoly.degree()
    m = b.minpoly.degree()

    if m % n != 0:
        return False

    if n == m:
        return True

    da = a.minpoly.discriminant()
    db = b.minpoly.discriminant()

    i, k, half = 1, m//n, db//2

    while True:
        p = sieve[i]
        P = p**k

        if P > half:
            break

        if ((da % p) % 2) and not (db % P):
            return False

        i += 1

    return True


def field_isomorphism_pslq(a, b):
    """Construct field isomorphism using PSLQ algorithm. """
    if not a.root.is_real or not b.root.is_real:
        raise NotImplementedError("PSLQ doesn't support complex coefficients")

    f = a.minpoly
    g = b.minpoly.replace(f.gen)

    n, m, prev = 100, b.minpoly.degree(), None
    ctx = MPContext()

    for i in range(1, 5):
        A = a.root.evalf(n)
        B = b.root.evalf(n)

        basis = [1, B] + [ B**i for i in range(2, m) ] + [-A]

        ctx.dps = n
        coeffs = ctx.pslq(basis, maxcoeff=10**10, maxsteps=1000)

        if coeffs is None:
            # PSLQ can't find an integer linear combination. Give up.
            break

        if coeffs != prev:
            prev = coeffs
        else:
            # Increasing precision didn't produce anything new. Give up.
            break

        # We have
        #   c0 + c1*B + c2*B^2 + ... + cm-1*B^(m-1) - cm*A ~ 0.
        # So bring cm*A to the other side, and divide through by cm,
        # for an approximate representation of A as a polynomial in B.
        # (We know cm != 0 since `b.minpoly` is irreducible.)
        coeffs = [S(c)/coeffs[-1] for c in coeffs[:-1]]

        # Throw away leading zeros.
        while not coeffs[-1]:
            coeffs.pop()

        coeffs = list(reversed(coeffs))
        h = Poly(coeffs, f.gen, domain='QQ')

        # We only have A ~ h(B). We must check whether the relation is exact.
        if f.compose(h).rem(g).is_zero:
            # Now we know that h(b) is in fact equal to _some conjugate of_ a.
            # But from the very precise approximation A ~ h(B) we can assume
            # the conjugate is a itself.
            return coeffs
        else:
            n *= 2

    return None


def field_isomorphism_factor(a, b):
    """Construct field isomorphism via factorization. """
    _, factors = factor_list(a.minpoly, extension=b)
    for f, _ in factors:
        if f.degree() == 1:
            # Any linear factor f(x) represents some conjugate of a in QQ(b).
            # We want to know whether this linear factor represents a itself.
            # Let f = x - c
            c = -f.rep.TC()
            # Write c as polynomial in b
            coeffs = c.to_sympy_list()
            d, terms = len(coeffs) - 1, []
            for i, coeff in enumerate(coeffs):
                terms.append(coeff*b.root**(d - i))
            r = Add(*terms)
            # Check whether we got the number a
            if a.minpoly.same_root(r, a):
                return coeffs

    # If none of the linear factors represented a in QQ(b), then in fact a is
    # not an element of QQ(b).
    return None


@public
def field_isomorphism(a, b, *, fast=True):
    r"""
    Find an embedding of one number field into another.

    Explanation
    ===========

    This function looks for an isomorphism from $\mathbb{Q}(a)$ onto some
    subfield of $\mathbb{Q}(b)$. Thus, it solves the Subfield Problem.

    Examples
    ========

    >>> from sympy import sqrt, field_isomorphism, I
    >>> print(field_isomorphism(3, sqrt(2)))  # doctest: +SKIP
    [3]
    >>> print(field_isomorphism( I*sqrt(3), I*sqrt(3)/2))  # doctest: +SKIP
    [2, 0]

    Parameters
    ==========

    a : :py:class:`~.Expr`
        Any expression representing an algebraic number.
    b : :py:class:`~.Expr`
        Any expression representing an algebraic number.
    fast : boolean, optional (default=True)
        If ``True``, we first attempt a potentially faster way of computing the
        isomorphism, falling back on a slower method if this fails. If
        ``False``, we go directly to the slower method, which is guaranteed to
        return a result.

    Returns
    =======

    List of rational numbers, or None
        If $\mathbb{Q}(a)$ is not isomorphic to some subfield of
        $\mathbb{Q}(b)$, then return ``None``. Otherwise, return a list of
        rational numbers representing an element of $\mathbb{Q}(b)$ to which
        $a$ may be mapped, in order to define a monomorphism, i.e. an
        isomorphism from $\mathbb{Q}(a)$ to some subfield of $\mathbb{Q}(b)$.
        The elements of the list are the coefficients of falling powers of $b$.

    """
    a, b = sympify(a), sympify(b)

    if not a.is_AlgebraicNumber:
        a = AlgebraicNumber(a)

    if not b.is_AlgebraicNumber:
        b = AlgebraicNumber(b)

    a = a.to_primitive_element()
    b = b.to_primitive_element()

    if a == b:
        return a.coeffs()

    n = a.minpoly.degree()
    m = b.minpoly.degree()

    if n == 1:
        return [a.root]

    if m % n != 0:
        return None

    if fast:
        try:
            result = field_isomorphism_pslq(a, b)

            if result is not None:
                return result
        except NotImplementedError:
            pass

    return field_isomorphism_factor(a, b)


def _switch_domain(g, K):
    # An algebraic relation f(a, b) = 0 over Q can also be written
    # g(b) = 0 where g is in Q(a)[x] and h(a) = 0 where h is in Q(b)[x].
    # This function transforms g into h where Q(b) = K.
    frep = g.rep.inject()
    hrep = frep.eject(K, front=True)

    return g.new(hrep, g.gens[0])


def _linsolve(p):
    # Compute root of linear polynomial.
    c, d = p.rep.to_list()
    return -d/c


@public
def primitive_element(extension, x=None, *, ex=False, polys=False):
    r"""
    Find a single generator for a number field given by several generators.

    Explanation
    ===========

    The basic problem is this: Given several algebraic numbers
    $\alpha_1, \alpha_2, \ldots, \alpha_n$, find a single algebraic number
    $\theta$ such that
    $\mathbb{Q}(\alpha_1, \alpha_2, \ldots, \alpha_n) = \mathbb{Q}(\theta)$.

    This function actually guarantees that $\theta$ will be a linear
    combination of the $\alpha_i$, with non-negative integer coefficients.

    Furthermore, if desired, this function will tell you how to express each
    $\alpha_i$ as a $\mathbb{Q}$-linear combination of the powers of $\theta$.

    Examples
    ========

    >>> from sympy import primitive_element, sqrt, S, minpoly, simplify
    >>> from sympy.abc import x
    >>> f, lincomb, reps = primitive_element([sqrt(2), sqrt(3)], x, ex=True)

    Then ``lincomb`` tells us the primitive element as a linear combination of
    the given generators ``sqrt(2)`` and ``sqrt(3)``.

    >>> print(lincomb)
    [1, 1]

    This means the primtiive element is $\sqrt{2} + \sqrt{3}$.
    Meanwhile ``f`` is the minimal polynomial for this primitive element.

    >>> print(f)
    x**4 - 10*x**2 + 1
    >>> print(minpoly(sqrt(2) + sqrt(3), x))
    x**4 - 10*x**2 + 1

    Finally, ``reps`` (which was returned only because we set keyword arg
    ``ex=True``) tells us how to recover each of the generators $\sqrt{2}$ and
    $\sqrt{3}$ as $\mathbb{Q}$-linear combinations of the powers of the
    primitive element $\sqrt{2} + \sqrt{3}$.

    >>> print([S(r) for r in reps[0]])
    [1/2, 0, -9/2, 0]
    >>> theta = sqrt(2) + sqrt(3)
    >>> print(simplify(theta**3/2 - 9*theta/2))
    sqrt(2)
    >>> print([S(r) for r in reps[1]])
    [-1/2, 0, 11/2, 0]
    >>> print(simplify(-theta**3/2 + 11*theta/2))
    sqrt(3)

    Parameters
    ==========

    extension : list of :py:class:`~.Expr`
        Each expression must represent an algebraic number $\alpha_i$.
    x : :py:class:`~.Symbol`, optional (default=None)
        The desired symbol to appear in the computed minimal polynomial for the
        primitive element $\theta$. If ``None``, we use a dummy symbol.
    ex : boolean, optional (default=False)
        If and only if ``True``, compute the representation of each $\alpha_i$
        as a $\mathbb{Q}$-linear combination over the powers of $\theta$.
    polys : boolean, optional (default=False)
        If ``True``, return the minimal polynomial as a :py:class:`~.Poly`.
        Otherwise return it as an :py:class:`~.Expr`.

    Returns
    =======

    Pair (f, coeffs) or triple (f, coeffs, reps), where:
        ``f`` is the minimal polynomial for the primitive element.
        ``coeffs`` gives the primitive element as a linear combination of the
        given generators.
        ``reps`` is present if and only if argument ``ex=True`` was passed,
        and is a list of lists of rational numbers. Each list gives the
        coefficients of falling powers of the primitive element, to recover
        one of the original, given generators.

    """
    if not extension:
        raise ValueError("Cannot compute primitive element for empty extension")
    extension = [_sympify(ext) for ext in extension]

    if x is not None:
        x, cls = sympify(x), Poly
    else:
        x, cls = Dummy('x'), PurePoly

    if not ex:
        gen, coeffs = extension[0], [1]
        g = minimal_polynomial(gen, x, polys=True)
        for ext in extension[1:]:
            if ext.is_Rational:
                coeffs.append(0)
                continue
            _, factors = factor_list(g, extension=ext)
            g = _choose_factor(factors, x, gen)
            [s], _, g = g.sqf_norm()
            gen += s*ext
            coeffs.append(s)

        if not polys:
            return g.as_expr(), coeffs
        else:
            return cls(g), coeffs

    gen, coeffs = extension[0], [1]
    f = minimal_polynomial(gen, x, polys=True)
    K = QQ.algebraic_field((f, gen))  # incrementally constructed field
    reps = [K.unit]  # representations of extension elements in K
    for ext in extension[1:]:
        if ext.is_Rational:
            coeffs.append(0)    # rational ext is not included in the expression of a primitive element
            reps.append(K.convert(ext))    # but it is included in reps
            continue
        p = minimal_polynomial(ext, x, polys=True)
        L = QQ.algebraic_field((p, ext))
        _, factors = factor_list(f, domain=L)
        f = _choose_factor(factors, x, gen)
        [s], g, f = f.sqf_norm()
        gen += s*ext
        coeffs.append(s)
        K = QQ.algebraic_field((f, gen))
        h = _switch_domain(g, K)
        erep = _linsolve(h.gcd(p))  # ext as element of K
        ogen = K.unit - s*erep  # old gen as element of K
        reps = [dup_eval(_.to_list(), ogen, K) for _ in reps] + [erep]

    if K.ext.root.is_Rational:  # all extensions are rational
        H = [K.convert(_).rep for _ in extension]
        coeffs = [0]*len(extension)
        f = cls(x, domain=QQ)
    else:
        H = [_.to_list() for _ in reps]
    if not polys:
        return f.as_expr(), coeffs, H
    else:
        return f, coeffs, H


@public
def to_number_field(extension, theta=None, *, gen=None, alias=None):
    r"""
    Express one algebraic number in the field generated by another.

    Explanation
    ===========

    Given two algebraic numbers $\eta, \theta$, this function either expresses
    $\eta$ as an element of $\mathbb{Q}(\theta)$, or else raises an exception
    if $\eta \not\in \mathbb{Q}(\theta)$.

    This function is essentially just a convenience, utilizing
    :py:func:`~.field_isomorphism` (our solution of the Subfield Problem) to
    solve this, the Field Membership Problem.

    As an additional convenience, this function allows you to pass a list of
    algebraic numbers $\alpha_1, \alpha_2, \ldots, \alpha_n$ instead of $\eta$.
    It then computes $\eta$ for you, as a solution of the Primitive Element
    Problem, using :py:func:`~.primitive_element` on the list of $\alpha_i$.

    Examples
    ========

    >>> from sympy import sqrt, to_number_field
    >>> eta = sqrt(2)
    >>> theta = sqrt(2) + sqrt(3)
    >>> a = to_number_field(eta, theta)
    >>> print(type(a))
    <class 'sympy.core.numbers.AlgebraicNumber'>
    >>> a.root
    sqrt(2) + sqrt(3)
    >>> print(a)
    sqrt(2)
    >>> a.coeffs()
    [1/2, 0, -9/2, 0]

    We get an :py:class:`~.AlgebraicNumber`, whose ``.root`` is $\theta$, whose
    value is $\eta$, and whose ``.coeffs()`` show how to write $\eta$ as a
    $\mathbb{Q}$-linear combination in falling powers of $\theta$.

    Parameters
    ==========

    extension : :py:class:`~.Expr` or list of :py:class:`~.Expr`
        Either the algebraic number that is to be expressed in the other field,
        or else a list of algebraic numbers, a primitive element for which is
        to be expressed in the other field.
    theta : :py:class:`~.Expr`, None, optional (default=None)
        If an :py:class:`~.Expr` representing an algebraic number, behavior is
        as described under **Explanation**. If ``None``, then this function
        reduces to a shorthand for calling :py:func:`~.primitive_element` on
        ``extension`` and turning the computed primitive element into an
        :py:class:`~.AlgebraicNumber`.
    gen : :py:class:`~.Symbol`, None, optional (default=None)
        If provided, this will be used as the generator symbol for the minimal
        polynomial in the returned :py:class:`~.AlgebraicNumber`.
    alias : str, :py:class:`~.Symbol`, None, optional (default=None)
        If provided, this will be used as the alias symbol for the returned
        :py:class:`~.AlgebraicNumber`.

    Returns
    =======

    AlgebraicNumber
        Belonging to $\mathbb{Q}(\theta)$ and equaling $\eta$.

    Raises
    ======

    IsomorphismFailed
        If $\eta \not\in \mathbb{Q}(\theta)$.

    See Also
    ========

    field_isomorphism
    primitive_element

    """
    if hasattr(extension, '__iter__'):
        extension = list(extension)
    else:
        extension = [extension]

    if len(extension) == 1 and isinstance(extension[0], tuple):
        return AlgebraicNumber(extension[0], alias=alias)

    minpoly, coeffs = primitive_element(extension, gen, polys=True)
    root = sum(coeff*ext for coeff, ext in zip(coeffs, extension))

    if theta is None:
        return AlgebraicNumber((minpoly, root), alias=alias)
    else:
        theta = sympify(theta)

        if not theta.is_AlgebraicNumber:
            theta = AlgebraicNumber(theta, gen=gen, alias=alias)

        coeffs = field_isomorphism(root, theta)

        if coeffs is not None:
            return AlgebraicNumber(theta, coeffs, alias=alias)
        else:
            raise IsomorphismFailed(
                "%s is not in a subfield of %s" % (root, theta.root))
