"""Computing integral bases for number fields. """

from sympy.polys.polytools import Poly
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.utilities.decorator import public
from .modules import ModuleEndomorphism, ModuleHomomorphism, PowerBasis
from .utilities import extract_fundamental_discriminant


def _apply_Dedekind_criterion(T, p):
    r"""
    Apply the "Dedekind criterion" to test whether the order needs to be
    enlarged relative to a given prime *p*.
    """
    x = T.gen
    T_bar = Poly(T, modulus=p)
    lc, fl = T_bar.factor_list()
    assert lc == 1
    g_bar = Poly(1, x, modulus=p)
    for ti_bar, _ in fl:
        g_bar *= ti_bar
    h_bar = T_bar // g_bar
    g = Poly(g_bar, domain=ZZ)
    h = Poly(h_bar, domain=ZZ)
    f = (g * h - T) // p
    f_bar = Poly(f, modulus=p)
    Z_bar = f_bar
    for b in [g_bar, h_bar]:
        Z_bar = Z_bar.gcd(b)
    U_bar = T_bar // Z_bar
    m = Z_bar.degree()
    return U_bar, m


def nilradical_mod_p(H, p, q=None):
    r"""
    Compute the nilradical mod *p* for a given order *H*, and prime *p*.

    Explanation
    ===========

    This is the ideal $I$ in $H/pH$ consisting of all elements some positive
    power of which is zero in this quotient ring, i.e. is a multiple of *p*.

    Parameters
    ==========

    H : :py:class:`~.Submodule`
        The given order.
    p : int
        The rational prime.
    q : int, optional
        If known, the smallest power of *p* that is $>=$ the dimension of *H*.
        If not provided, we compute it here.

    Returns
    =======

    :py:class:`~.Module` representing the nilradical mod *p* in *H*.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.
    (See Lemma 6.1.6.)

    """
    n = H.n
    if q is None:
        q = p
        while q < n:
            q *= p
    phi = ModuleEndomorphism(H, lambda x: x**q)
    return phi.kernel(modulus=p)


def _second_enlargement(H, p, q):
    r"""
    Perform the second enlargement in the Round Two algorithm.
    """
    Ip = nilradical_mod_p(H, p, q=q)
    B = H.parent.submodule_from_matrix(H.matrix * Ip.matrix, denom=H.denom)
    C = B + p*H
    E = C.endomorphism_ring()
    phi = ModuleHomomorphism(H, E, lambda x: E.inner_endomorphism(x))
    gamma = phi.kernel(modulus=p)
    G = H.parent.submodule_from_matrix(H.matrix * gamma.matrix, denom=H.denom * p)
    H1 = G + H
    return H1, Ip


@public
def round_two(T, radicals=None):
    r"""
    Zassenhaus's "Round 2" algorithm.

    Explanation
    ===========

    Carry out Zassenhaus's "Round 2" algorithm on an irreducible polynomial
    *T* over :ref:`ZZ` or :ref:`QQ`. This computes an integral basis and the
    discriminant for the field $K = \mathbb{Q}[x]/(T(x))$.

    Alternatively, you may pass an :py:class:`~.AlgebraicField` instance, in
    place of the polynomial *T*, in which case the algorithm is applied to the
    minimal polynomial for the field's primitive element.

    Ordinarily this function need not be called directly, as one can instead
    access the :py:meth:`~.AlgebraicField.maximal_order`,
    :py:meth:`~.AlgebraicField.integral_basis`, and
    :py:meth:`~.AlgebraicField.discriminant` methods of an
    :py:class:`~.AlgebraicField`.

    Examples
    ========

    Working through an AlgebraicField:

    >>> from sympy import Poly, QQ
    >>> from sympy.abc import x
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> K = QQ.alg_field_from_poly(T, "theta")
    >>> print(K.maximal_order())
    Submodule[[2, 0, 0], [0, 2, 0], [0, 1, 1]]/2
    >>> print(K.discriminant())
    -503
    >>> print(K.integral_basis(fmt='sympy'))
    [1, theta, theta/2 + theta**2/2]

    Calling directly:

    >>> from sympy import Poly
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.basis import round_two
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> print(round_two(T))
    (Submodule[[2, 0, 0], [0, 2, 0], [0, 1, 1]]/2, -503)

    The nilradicals mod $p$ that are sometimes computed during the Round Two
    algorithm may be useful in further calculations. Pass a dictionary under
    `radicals` to receive these:

    >>> T = Poly(x**3 + 3*x**2 + 5)
    >>> rad = {}
    >>> ZK, dK = round_two(T, radicals=rad)
    >>> print(rad)
    {3: Submodule[[-1, 1, 0], [-1, 0, 1]]}

    Parameters
    ==========

    T : :py:class:`~.Poly`, :py:class:`~.AlgebraicField`
        Either (1) the irreducible polynomial over :ref:`ZZ` or :ref:`QQ`
        defining the number field, or (2) an :py:class:`~.AlgebraicField`
        representing the number field itself.

    radicals : dict, optional
        This is a way for any $p$-radicals (if computed) to be returned by
        reference. If desired, pass an empty dictionary. If the algorithm
        reaches the point where it computes the nilradical mod $p$ of the ring
        of integers $Z_K$, then an $\mathbb{F}_p$-basis for this ideal will be
        stored in this dictionary under the key ``p``. This can be useful for
        other algorithms, such as prime decomposition.

    Returns
    =======

    Pair ``(ZK, dK)``, where:

        ``ZK`` is a :py:class:`~sympy.polys.numberfields.modules.Submodule`
        representing the maximal order.

        ``dK`` is the discriminant of the field $K = \mathbb{Q}[x]/(T(x))$.

    See Also
    ========

    .AlgebraicField.maximal_order
    .AlgebraicField.integral_basis
    .AlgebraicField.discriminant

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*

    """
    K = None
    if isinstance(T, AlgebraicField):
        K, T = T, T.ext.minpoly_of_element()
    if (   not T.is_univariate
        or not T.is_irreducible
        or T.domain not in [ZZ, QQ]):
        raise ValueError('Round 2 requires an irreducible univariate polynomial over ZZ or QQ.')
    T, _ = T.make_monic_over_integers_by_scaling_roots()
    n = T.degree()
    D = T.discriminant()
    D_modulus = ZZ.from_sympy(abs(D))
    # D must be 0 or 1 mod 4 (see Cohen Sec 4.4), which ensures we can write
    # it in the form D = D_0 * F**2, where D_0 is 1 or a fundamental discriminant.
    _, F = extract_fundamental_discriminant(D)
    Ztheta = PowerBasis(K or T)
    H = Ztheta.whole_submodule()
    nilrad = None
    while F:
        # Next prime:
        p, e = F.popitem()
        U_bar, m = _apply_Dedekind_criterion(T, p)
        if m == 0:
            continue
        # For a given prime p, the first enlargement of the order spanned by
        # the current basis can be done in a simple way:
        U = Ztheta.element_from_poly(Poly(U_bar, domain=ZZ))
        # TODO:
        #  Theory says only first m columns of the U//p*H term below are needed.
        #  Could be slightly more efficient to use only those. Maybe `Submodule`
        #  class should support a slice operator?
        H = H.add(U // p * H, hnf_modulus=D_modulus)
        if e <= m:
            continue
        # A second, and possibly more, enlargements for p will be needed.
        # These enlargements require a more involved procedure.
        q = p
        while q < n:
            q *= p
        H1, nilrad = _second_enlargement(H, p, q)
        while H1 != H:
            H = H1
            H1, nilrad = _second_enlargement(H, p, q)
    # Note: We do not store all nilradicals mod p, only the very last. This is
    # because, unless computed against the entire integral basis, it might not
    # be accurate. (In other words, if H was not already equal to ZK when we
    # passed it to `_second_enlargement`, then we can't trust the nilradical
    # so computed.) Example: if T(x) = x ** 3 + 15 * x ** 2 - 9 * x + 13, then
    # F is divisible by 2, 3, and 7, and the nilradical mod 2 as computed above
    # will not be accurate for the full, maximal order ZK.
    if nilrad is not None and isinstance(radicals, dict):
        radicals[p] = nilrad
    ZK = H
    # Pre-set expensive boolean properties which we already know to be true:
    ZK._starts_with_unity = True
    ZK._is_sq_maxrank_HNF = True
    dK = (D * ZK.matrix.det() ** 2) // ZK.denom ** (2 * n)
    return ZK, dK
