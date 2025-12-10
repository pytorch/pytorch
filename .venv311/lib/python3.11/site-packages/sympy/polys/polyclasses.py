"""OO layer for several polynomial representations. """

from __future__ import annotations

from sympy.external.gmpy import GROUND_TYPES

from sympy.utilities.exceptions import sympy_deprecation_warning

from sympy.core.numbers import oo
from sympy.core.sympify import CantSympify
from sympy.polys.polyutils import PicklableWithSlots, _sort_factors
from sympy.polys.domains import Domain, ZZ, QQ

from sympy.polys.polyerrors import (
    CoercionFailed,
    ExactQuotientFailed,
    DomainError,
    NotInvertible,
)

from sympy.polys.densebasic import (
    ninf,
    dmp_validate,
    dup_normal, dmp_normal,
    dup_convert, dmp_convert,
    dmp_from_sympy,
    dup_strip,
    dmp_degree_in,
    dmp_degree_list,
    dmp_negative_p,
    dmp_ground_LC,
    dmp_ground_TC,
    dmp_ground_nth,
    dmp_one, dmp_ground,
    dmp_zero, dmp_zero_p, dmp_one_p, dmp_ground_p,
    dup_from_dict, dmp_from_dict,
    dmp_to_dict,
    dmp_deflate,
    dmp_inject, dmp_eject,
    dmp_terms_gcd,
    dmp_list_terms, dmp_exclude,
    dup_slice, dmp_slice_in, dmp_permute,
    dmp_to_tuple,)

from sympy.polys.densearith import (
    dmp_add_ground,
    dmp_sub_ground,
    dmp_mul_ground,
    dmp_quo_ground,
    dmp_exquo_ground,
    dmp_abs,
    dmp_neg,
    dmp_add,
    dmp_sub,
    dmp_mul,
    dmp_sqr,
    dmp_pow,
    dmp_pdiv,
    dmp_prem,
    dmp_pquo,
    dmp_pexquo,
    dmp_div,
    dmp_rem,
    dmp_quo,
    dmp_exquo,
    dmp_add_mul, dmp_sub_mul,
    dmp_max_norm,
    dmp_l1_norm,
    dmp_l2_norm_squared)

from sympy.polys.densetools import (
    dmp_clear_denoms,
    dmp_integrate_in,
    dmp_diff_in,
    dmp_eval_in,
    dup_revert,
    dmp_ground_trunc,
    dmp_ground_content,
    dmp_ground_primitive,
    dmp_ground_monic,
    dmp_compose,
    dup_decompose,
    dup_shift,
    dmp_shift,
    dup_transform,
    dmp_lift)

from sympy.polys.euclidtools import (
    dup_half_gcdex, dup_gcdex, dup_invert,
    dmp_subresultants,
    dmp_resultant,
    dmp_discriminant,
    dmp_inner_gcd,
    dmp_gcd,
    dmp_lcm,
    dmp_cancel)

from sympy.polys.sqfreetools import (
    dup_gff_list,
    dmp_norm,
    dmp_sqf_p,
    dmp_sqf_norm,
    dmp_sqf_part,
    dmp_sqf_list, dmp_sqf_list_include)

from sympy.polys.factortools import (
    dup_cyclotomic_p, dmp_irreducible_p,
    dmp_factor_list, dmp_factor_list_include)

from sympy.polys.rootisolation import (
    dup_isolate_real_roots_sqf,
    dup_isolate_real_roots,
    dup_isolate_all_roots_sqf,
    dup_isolate_all_roots,
    dup_refine_real_root,
    dup_count_real_roots,
    dup_count_complex_roots,
    dup_sturm,
    dup_cauchy_upper_bound,
    dup_cauchy_lower_bound,
    dup_mignotte_sep_bound_squared)

from sympy.polys.polyerrors import (
    UnificationFailed,
    PolynomialError)


if GROUND_TYPES == 'flint':
    import flint
    def _supported_flint_domain(D):
        return D.is_ZZ or D.is_QQ or D.is_FF and D._is_flint
else:
    flint = None
    def _supported_flint_domain(D):
        return False


class DMP(CantSympify):
    """Dense Multivariate Polynomials over `K`. """

    __slots__ = ()

    lev: int
    dom: Domain

    def __new__(cls, rep, dom, lev=None):

        if lev is None:
            rep, lev = dmp_validate(rep)
        elif not isinstance(rep, list):
            raise CoercionFailed("expected list, got %s" % type(rep))

        return cls.new(rep, dom, lev)

    @classmethod
    def new(cls, rep, dom, lev):
        # It would be too slow to call _validate_args always at runtime.
        # Ideally this checking would be handled by a static type checker.
        #
        #cls._validate_args(rep, dom, lev)
        if flint is not None:
            if lev == 0 and _supported_flint_domain(dom):
                return DUP_Flint._new(rep, dom, lev)

        return DMP_Python._new(rep, dom, lev)

    @property
    def rep(f):
        """Get the representation of ``f``. """

        sympy_deprecation_warning("""
        Accessing the ``DMP.rep`` attribute is deprecated. The internal
        representation of ``DMP`` instances can now be ``DUP_Flint`` when the
        ground types are ``flint``. In this case the ``DMP`` instance does not
        have a ``rep`` attribute. Use ``DMP.to_list()`` instead. Using
        ``DMP.to_list()`` also works in previous versions of SymPy.
        """,
            deprecated_since_version="1.13",
            active_deprecations_target="dmp-rep",
        )

        return f.to_list()

    def to_best(f):
        """Convert to DUP_Flint if possible.

        This method should be used when the domain or level is changed and it
        potentially becomes possible to convert from DMP_Python to DUP_Flint.
        """
        if flint is not None:
            if isinstance(f, DMP_Python) and f.lev == 0 and _supported_flint_domain(f.dom):
                return DUP_Flint.new(f._rep, f.dom, f.lev)

        return f

    @classmethod
    def _validate_args(cls, rep, dom, lev):
        assert isinstance(dom, Domain)
        assert isinstance(lev, int) and lev >= 0

        def validate_rep(rep, lev):
            assert isinstance(rep, list)
            if lev == 0:
                assert all(dom.of_type(c) for c in rep)
            else:
                for r in rep:
                    validate_rep(r, lev - 1)

        validate_rep(rep, lev)

    @classmethod
    def from_dict(cls, rep, lev, dom):
        rep = dmp_from_dict(rep, lev, dom)
        return cls.new(rep, dom, lev)

    @classmethod
    def from_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of native coefficients. """
        return cls.new(dmp_convert(rep, lev, None, dom), dom, lev)

    @classmethod
    def from_sympy_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of SymPy coefficients. """
        return cls.new(dmp_from_sympy(rep, lev, dom), dom, lev)

    @classmethod
    def from_monoms_coeffs(cls, monoms, coeffs, lev, dom):
        return cls(dict(list(zip(monoms, coeffs))), dom, lev)

    def convert(f, dom):
        """Convert ``f`` to a ``DMP`` over the new domain. """
        if f.dom == dom:
            return f
        elif f.lev or flint is None:
            return f._convert(dom)
        elif isinstance(f, DUP_Flint):
            if _supported_flint_domain(dom):
                return f._convert(dom)
            else:
                return f.to_DMP_Python()._convert(dom)
        elif isinstance(f, DMP_Python):
            if _supported_flint_domain(dom):
                return f._convert(dom).to_DUP_Flint()
            else:
                return f._convert(dom)
        else:
            raise RuntimeError("unreachable code")

    def _convert(f, dom):
        raise NotImplementedError

    @classmethod
    def zero(cls, lev, dom):
        return DMP(dmp_zero(lev), dom, lev)

    @classmethod
    def one(cls, lev, dom):
        return DMP(dmp_one(lev, dom), dom, lev)

    def _one(f):
        raise NotImplementedError

    def __repr__(f):
        return "%s(%s, %s)" % (f.__class__.__name__, f.to_list(), f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, f.to_tuple(), f.lev, f.dom))

    def __getnewargs__(self):
        return self.to_list(), self.dom, self.lev

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        raise NotImplementedError

    def unify_DMP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom != g.dom:
            dom = f.dom.unify(g.dom)
            f = f.convert(dom)
            g = g.convert(dom)

        return f, g

    def to_dict(f, zero=False):
        """Convert ``f`` to a dict representation with native coefficients. """
        return dmp_to_dict(f.to_list(), f.lev, f.dom, zero=zero)

    def to_sympy_dict(f, zero=False):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
        rep = f.to_dict(zero=zero)

        for k, v in rep.items():
            rep[k] = f.dom.to_sympy(v)

        return rep

    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
        def sympify_nested_list(rep):
            out = []
            for val in rep:
                if isinstance(val, list):
                    out.append(sympify_nested_list(val))
                else:
                    out.append(f.dom.to_sympy(val))
            return out

        return sympify_nested_list(f.to_list())

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        raise NotImplementedError

    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
        raise NotImplementedError

    def to_ring(f):
        """Make the ground domain a ring. """
        return f.convert(f.dom.get_ring())

    def to_field(f):
        """Make the ground domain a field. """
        return f.convert(f.dom.get_field())

    def to_exact(f):
        """Make the ground domain exact. """
        return f.convert(f.dom.get_exact())

    def slice(f, m, n, j=0):
        """Take a continuous subsequence of terms of ``f``. """
        if not f.lev and not j:
            return f._slice(m, n)
        else:
            return f._slice_lev(m, n, j)

    def _slice(f, m, n):
        raise NotImplementedError

    def _slice_lev(f, m, n, j):
        raise NotImplementedError

    def coeffs(f, order=None):
        """Returns all non-zero coefficients from ``f`` in lex order. """
        return [ c for _, c in f.terms(order=order) ]

    def monoms(f, order=None):
        """Returns all non-zero monomials from ``f`` in lex order. """
        return [ m for m, _ in f.terms(order=order) ]

    def terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        if f.is_zero:
            zero_monom = (0,)*(f.lev + 1)
            return [(zero_monom, f.dom.zero)]
        else:
            return f._terms(order=order)

    def _terms(f, order=None):
        raise NotImplementedError

    def all_coeffs(f):
        """Returns all coefficients from ``f``. """
        if f.lev:
            raise PolynomialError('multivariate polynomials not supported')

        if not f:
            return [f.dom.zero]
        else:
            return list(f.to_list())

    def all_monoms(f):
        """Returns all monomials from ``f``. """
        if f.lev:
            raise PolynomialError('multivariate polynomials not supported')

        n = f.degree()

        if n < 0:
            return [(0,)]
        else:
            return [ (n - i,) for i, c in enumerate(f.to_list()) ]

    def all_terms(f):
        """Returns all terms from a ``f``. """
        if f.lev:
            raise PolynomialError('multivariate polynomials not supported')

        n = f.degree()

        if n < 0:
            return [((0,), f.dom.zero)]
        else:
            return [ ((n - i,), c) for i, c in enumerate(f.to_list()) ]

    def lift(f):
        """Convert algebraic coefficients to rationals. """
        return f._lift().to_best()

    def _lift(f):
        raise NotImplementedError

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        raise NotImplementedError

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        raise NotImplementedError

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        raise NotImplementedError

    def exclude(f):
        r"""
        Remove useless generators from ``f``.

        Returns the removed generators and the new excluded ``f``.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(1)]], [[ZZ(1)], [ZZ(2)]]], ZZ).exclude()
        ([2], DMP_Python([[1], [1, 2]], ZZ))

        """
        J, F = f._exclude()
        return J, F.to_best()

    def _exclude(f):
        raise NotImplementedError

    def permute(f, P):
        r"""
        Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 0, 2])
        DMP_Python([[[2], []], [[1, 0], []]], ZZ)

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 2, 0])
        DMP_Python([[[1], []], [[2, 0], []]], ZZ)

        """
        return f._permute(P)

    def _permute(f, P):
        raise NotImplementedError

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        raise NotImplementedError

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        raise NotImplementedError

    def neg(f):
        """Negate all coefficients in ``f``. """
        raise NotImplementedError

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f._add_ground(f.dom.convert(c))

    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f._sub_ground(f.dom.convert(c))

    def mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f._mul_ground(f.dom.convert(c))

    def quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f._quo_ground(f.dom.convert(c))

    def exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
        return f._exquo_ground(f.dom.convert(c))

    def add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._add(G)

    def sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._sub(G)

    def mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._mul(G)

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        return f._sqr()

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if not isinstance(n, int):
            raise TypeError("``int`` expected, got %s" % type(n))
        return f._pow(n)

    def pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._pdiv(G)

    def prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._prem(G)

    def pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._pquo(G)

    def pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._pexquo(G)

    def div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._div(G)

    def rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._rem(G)

    def quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._quo(G)

    def exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._exquo(G)

    def _add_ground(f, c):
        raise NotImplementedError

    def _sub_ground(f, c):
        raise NotImplementedError

    def _mul_ground(f, c):
        raise NotImplementedError

    def _quo_ground(f, c):
        raise NotImplementedError

    def _exquo_ground(f, c):
        raise NotImplementedError

    def _add(f, g):
        raise NotImplementedError

    def _sub(f, g):
        raise NotImplementedError

    def _mul(f, g):
        raise NotImplementedError

    def _sqr(f):
        raise NotImplementedError

    def _pow(f, n):
        raise NotImplementedError

    def _pdiv(f, g):
        raise NotImplementedError

    def _prem(f, g):
        raise NotImplementedError

    def _pquo(f, g):
        raise NotImplementedError

    def _pexquo(f, g):
        raise NotImplementedError

    def _div(f, g):
        raise NotImplementedError

    def _rem(f, g):
        raise NotImplementedError

    def _quo(f, g):
        raise NotImplementedError

    def _exquo(f, g):
        raise NotImplementedError

    def degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        return f._degree(j)

    def _degree(f, j):
        raise NotImplementedError

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        raise NotImplementedError

    def total_degree(f):
        """Returns the total degree of ``f``. """
        raise NotImplementedError

    def homogenize(f, s):
        """Return homogeneous polynomial of ``f``"""
        td = f.total_degree()
        result = {}
        new_symbol = (s == len(f.terms()[0][0]))
        for term in f.terms():
            d = sum(term[0])
            if d < td:
                i = td - d
            else:
                i = 0
            if new_symbol:
                result[term[0] + (i,)] = term[1]
            else:
                l = list(term[0])
                l[s] += i
                result[tuple(l)] = term[1]
        return DMP.from_dict(result, f.lev + int(new_symbol), f.dom)

    def homogeneous_order(f):
        """Returns the homogeneous order of ``f``. """
        if f.is_zero:
            return -oo

        monoms = f.monoms()
        tdeg = sum(monoms[0])

        for monom in monoms:
            _tdeg = sum(monom)

            if _tdeg != tdeg:
                return None

        return tdeg

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        raise NotImplementedError

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        raise NotImplementedError

    def nth(f, *N):
        """Returns the ``n``-th coefficient of ``f``. """
        if all(isinstance(n, int) for n in N):
            return f._nth(N)
        else:
            raise TypeError("a sequence of integers expected")

    def _nth(f, N):
        raise NotImplementedError

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        raise NotImplementedError

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        raise NotImplementedError

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        raise NotImplementedError

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        raise NotImplementedError

    def integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        if not isinstance(m, int):
            raise TypeError("``int`` expected, got %s" % type(m))

        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        return f._integrate(m, j)

    def _integrate(f, m, j):
        raise NotImplementedError

    def diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f`` in ``x_j``. """
        if not isinstance(m, int):
            raise TypeError("``int`` expected, got %s" % type(m))

        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        return f._diff(m, j)

    def _diff(f, m, j):
        raise NotImplementedError

    def eval(f, a, j=0):
        """Evaluates ``f`` at the given point ``a`` in ``x_j``. """
        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))
        elif not (0 <= j <= f.lev):
            raise ValueError("invalid variable index %s" % j)

        if f.lev:
            return f._eval_lev(a, j)
        else:
            return f._eval(a)

    def _eval(f, a):
        raise NotImplementedError

    def _eval_lev(f, a, j):
        raise NotImplementedError

    def half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
        F, G = f.unify_DMP(g)

        if F.lev:
            raise ValueError('univariate polynomial expected')

        return F._half_gcdex(G)

    def _half_gcdex(f, g):
        raise NotImplementedError

    def gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
        F, G = f.unify_DMP(g)

        if F.lev:
            raise ValueError('univariate polynomial expected')

        if not F.dom.is_Field:
            raise DomainError('ground domain must be a field')

        return F._gcdex(G)

    def _gcdex(f, g):
        raise NotImplementedError

    def invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        F, G = f.unify_DMP(g)

        if F.lev:
            raise ValueError('univariate polynomial expected')

        return F._invert(G)

    def _invert(f, g):
        raise NotImplementedError

    def revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._revert(n)

    def _revert(f, n):
        raise NotImplementedError

    def subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._subresultants(G)

    def _subresultants(f, g):
        raise NotImplementedError

    def resultant(f, g, includePRS=False):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        F, G = f.unify_DMP(g)
        if includePRS:
            return F._resultant_includePRS(G)
        else:
            return F._resultant(G)

    def _resultant(f, g, includePRS=False):
        raise NotImplementedError

    def discriminant(f):
        """Computes discriminant of ``f``. """
        raise NotImplementedError

    def cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        F, G = f.unify_DMP(g)
        return F._cofactors(G)

    def _cofactors(f, g):
        raise NotImplementedError

    def gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._gcd(G)

    def _gcd(f, g):
        raise NotImplementedError

    def lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._lcm(G)

    def _lcm(f, g):
        raise NotImplementedError

    def cancel(f, g, include=True):
        """Cancel common factors in a rational function ``f/g``. """
        F, G = f.unify_DMP(g)

        if include:
            return F._cancel_include(G)
        else:
            return F._cancel(G)

    def _cancel(f, g):
        raise NotImplementedError

    def _cancel_include(f, g):
        raise NotImplementedError

    def trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f._trunc(f.dom.convert(p))

    def _trunc(f, p):
        raise NotImplementedError

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        raise NotImplementedError

    def content(f):
        """Returns GCD of polynomial coefficients. """
        raise NotImplementedError

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        raise NotImplementedError

    def compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._compose(G)

    def _compose(f, g):
        raise NotImplementedError

    def decompose(f):
        """Computes functional decomposition of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._decompose()

    def _decompose(f):
        raise NotImplementedError

    def shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._shift(f.dom.convert(a))

    def shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
        a = [f.dom.convert(ai) for ai in a]
        return f._shift_list(a)

    def _shift(f, a):
        raise NotImplementedError

    def transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        if f.lev:
            raise ValueError('univariate polynomial expected')

        P, Q = p.unify_DMP(q)
        F, P = f.unify_DMP(P)
        F, Q = F.unify_DMP(Q)

        return F._transform(P, Q)

    def _transform(f, p, q):
        raise NotImplementedError

    def sturm(f):
        """Computes the Sturm sequence of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._sturm()

    def _sturm(f):
        raise NotImplementedError

    def cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._cauchy_upper_bound()

    def _cauchy_upper_bound(f):
        raise NotImplementedError

    def cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._cauchy_lower_bound()

    def _cauchy_lower_bound(f):
        raise NotImplementedError

    def mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._mignotte_sep_bound_squared()

    def _mignotte_sep_bound_squared(f):
        raise NotImplementedError

    def gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._gff_list()

    def _gff_list(f):
        raise NotImplementedError

    def norm(f):
        """Computes ``Norm(f)``."""
        raise NotImplementedError

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        raise NotImplementedError

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        raise NotImplementedError

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        raise NotImplementedError

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        raise NotImplementedError

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """
        raise NotImplementedError

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        raise NotImplementedError

    def intervals(f, all=False, eps=None, inf=None, sup=None, fast=False, sqf=False):
        """Compute isolating intervals for roots of ``f``. """
        if f.lev:
            raise PolynomialError("Cannot isolate roots of a multivariate polynomial")

        if all and sqf:
            return f._isolate_all_roots_sqf(eps=eps, inf=inf, sup=sup, fast=fast)
        elif all and not sqf:
            return f._isolate_all_roots(eps=eps, inf=inf, sup=sup, fast=fast)
        elif not all and sqf:
            return f._isolate_real_roots_sqf(eps=eps, inf=inf, sup=sup, fast=fast)
        else:
            return f._isolate_real_roots(eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        raise NotImplementedError

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        raise NotImplementedError

    def _isolate_real_roots(f, eps, inf, sup, fast):
        raise NotImplementedError

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        raise NotImplementedError

    def refine_root(f, s, t, eps=None, steps=None, fast=False):
        """
        Refine an isolating interval to the given precision.

        ``eps`` should be a rational number.

        """
        if f.lev:
            raise PolynomialError(
                "Cannot refine a root of a multivariate polynomial")

        return f._refine_real_root(s, t, eps=eps, steps=steps, fast=fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        raise NotImplementedError

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        raise NotImplementedError

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        raise NotImplementedError

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        raise NotImplementedError

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        raise NotImplementedError

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        raise NotImplementedError

    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        raise NotImplementedError

    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        raise NotImplementedError

    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        raise NotImplementedError

    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        raise NotImplementedError

    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        raise NotImplementedError

    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        raise NotImplementedError

    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        raise NotImplementedError

    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        raise NotImplementedError

    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        raise NotImplementedError

    def __abs__(f):
        return f.abs()

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, DMP):
            return f.add(g)
        else:
            try:
                return f.add_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, DMP):
            return f.sub(g)
        else:
            try:
                return f.sub_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, DMP):
            return f.mul(g)
        else:
            try:
                return f.mul_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __rmul__(f, g):
        return f.__mul__(g)

    def __truediv__(f, g):
        if isinstance(g, DMP):
            return f.exquo(g)
        else:
            try:
                return f.mul_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __rtruediv__(f, g):
        if isinstance(g, DMP):
            return g.exquo(f)
        else:
            try:
                return f._one().mul_ground(g).exquo(f)
            except CoercionFailed:
                return NotImplemented

    def __pow__(f, n):
        return f.pow(n)

    def __divmod__(f, g):
        return f.div(g)

    def __mod__(f, g):
        return f.rem(g)

    def __floordiv__(f, g):
        if isinstance(g, DMP):
            return f.quo(g)
        else:
            try:
                return f.quo_ground(g)
            except TypeError:
                return NotImplemented

    def __eq__(f, g):
        if f is g:
            return True
        if not isinstance(g, DMP):
            return NotImplemented
        try:
            F, G = f.unify_DMP(g)
        except UnificationFailed:
            return False
        else:
            return F._strict_eq(G)

    def _strict_eq(f, g):
        raise NotImplementedError

    def eq(f, g, strict=False):
        if not strict:
            return f == g
        else:
            return f._strict_eq(g)

    def ne(f, g, strict=False):
        return not f.eq(g, strict=strict)

    def __lt__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() < G.to_list()

    def __le__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() <= G.to_list()

    def __gt__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() > G.to_list()

    def __ge__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() >= G.to_list()

    def __bool__(f):
        return not f.is_zero


class DMP_Python(DMP):
    """Dense Multivariate Polynomials over `K`. """

    __slots__ = ('_rep', 'dom', 'lev')

    @classmethod
    def _new(cls, rep, dom, lev):
        obj = object.__new__(cls)
        obj._rep = rep
        obj.lev = lev
        obj.dom = dom
        return obj

    def _strict_eq(f, g):
        if type(f) != type(g):
            return False
        return f.lev == g.lev and f.dom == g.dom and f._rep == g._rep

    def per(f, rep):
        """Create a DMP out of the given representation. """
        return f._new(rep, f.dom, f.lev)

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        return f._new(dmp_ground(coeff, f.lev), f.dom, f.lev)

    def _one(f):
        return f.one(f.lev, f.dom)

    def unify(f, g):
        """Unify representations of two multivariate polynomials. """
        # XXX: This function is not really used any more since there is
        # unify_DMP now.
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return f.lev, f.dom, f.per, f._rep, g._rep
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)

            F = dmp_convert(f._rep, lev, f.dom, dom)
            G = dmp_convert(g._rep, lev, g.dom, dom)

            def per(rep):
                return f._new(rep, dom, lev)

            return lev, dom, per, F, G

    def to_DUP_Flint(f):
        """Convert ``f`` to a Flint representation. """
        return DUP_Flint._new(f._rep, f.dom, f.lev)

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return list(f._rep)

    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
        return dmp_to_tuple(f._rep, f.lev)

    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
        return f._new(dmp_convert(f._rep, f.lev, f.dom, dom), dom, f.lev)

    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
        rep = dup_slice(f._rep, m, n, f.dom)
        return f._new(rep, f.dom, f.lev)

    def _slice_lev(f, m, n, j):
        """Take a continuous subsequence of terms of ``f``. """
        rep = dmp_slice_in(f._rep, m, n, j, f.lev, f.dom)
        return f._new(rep, f.dom, f.lev)

    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        return dmp_list_terms(f._rep, f.lev, f.dom, order=order)

    def _lift(f):
        """Convert algebraic coefficients to rationals. """
        r = dmp_lift(f._rep, f.lev, f.dom)
        return f._new(r, f.dom.dom, f.lev)

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        J, F = dmp_deflate(f._rep, f.lev, f.dom)
        return J, f.per(F)

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        F, lev = dmp_inject(f._rep, f.lev, f.dom, front=front)
        # XXX: domain and level changed here
        return f._new(F, f.dom.dom, lev)

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        F = dmp_eject(f._rep, f.lev, dom, front=front)
        # XXX: domain and level changed here
        return f._new(F, dom, f.lev - len(dom.symbols))

    def _exclude(f):
        """Remove useless generators from ``f``. """
        J, F, u = dmp_exclude(f._rep, f.lev, f.dom)
        # XXX: level changed here
        return J, f._new(F, f.dom, u)

    def _permute(f, P):
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
        return f.per(dmp_permute(f._rep, P, f.lev, f.dom))

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        J, F = dmp_terms_gcd(f._rep, f.lev, f.dom)
        return J, f.per(F)

    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.per(dmp_add_ground(f._rep, c, f.lev, f.dom))

    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.per(dmp_sub_ground(f._rep, c, f.lev, f.dom))

    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f.per(dmp_mul_ground(f._rep, c, f.lev, f.dom))

    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f.per(dmp_quo_ground(f._rep, c, f.lev, f.dom))

    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
        return f.per(dmp_exquo_ground(f._rep, c, f.lev, f.dom))

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        return f.per(dmp_abs(f._rep, f.lev, f.dom))

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.per(dmp_neg(f._rep, f.lev, f.dom))

    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        return f.per(dmp_add(f._rep, g._rep, f.lev, f.dom))

    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        return f.per(dmp_sub(f._rep, g._rep, f.lev, f.dom))

    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        return f.per(dmp_mul(f._rep, g._rep, f.lev, f.dom))

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        return f.per(dmp_sqr(f._rep, f.lev, f.dom))

    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        return f.per(dmp_pow(f._rep, n, f.lev, f.dom))

    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        q, r = dmp_pdiv(f._rep, g._rep, f.lev, f.dom)
        return f.per(q), f.per(r)

    def _prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        return f.per(dmp_prem(f._rep, g._rep, f.lev, f.dom))

    def _pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        return f.per(dmp_pquo(f._rep, g._rep, f.lev, f.dom))

    def _pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        return f.per(dmp_pexquo(f._rep, g._rep, f.lev, f.dom))

    def _div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        q, r = dmp_div(f._rep, g._rep, f.lev, f.dom)
        return f.per(q), f.per(r)

    def _rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        return f.per(dmp_rem(f._rep, g._rep, f.lev, f.dom))

    def _quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        return f.per(dmp_quo(f._rep, g._rep, f.lev, f.dom))

    def _exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        return f.per(dmp_exquo(f._rep, g._rep, f.lev, f.dom))

    def _degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        return dmp_degree_in(f._rep, j, f.lev)

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        return dmp_degree_list(f._rep, f.lev)

    def total_degree(f):
        """Returns the total degree of ``f``. """
        return max(sum(m) for m in f.monoms())

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return dmp_ground_LC(f._rep, f.lev, f.dom)

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return dmp_ground_TC(f._rep, f.lev, f.dom)

    def _nth(f, N):
        """Returns the ``n``-th coefficient of ``f``. """
        return dmp_ground_nth(f._rep, N, f.lev, f.dom)

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        return dmp_max_norm(f._rep, f.lev, f.dom)

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        return dmp_l1_norm(f._rep, f.lev, f.dom)

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        return dmp_l2_norm_squared(f._rep, f.lev, f.dom)

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        coeff, F = dmp_clear_denoms(f._rep, f.lev, f.dom)
        return coeff, f.per(F)

    def _integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        return f.per(dmp_integrate_in(f._rep, m, j, f.lev, f.dom))

    def _diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f`` in ``x_j``. """
        return f.per(dmp_diff_in(f._rep, m, j, f.lev, f.dom))

    def _eval(f, a):
        return dmp_eval_in(f._rep, f.dom.convert(a), 0, f.lev, f.dom)

    def _eval_lev(f, a, j):
        rep = dmp_eval_in(f._rep, f.dom.convert(a), j, f.lev, f.dom)
        return f.new(rep, f.dom, f.lev - 1)

    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
        s, h = dup_half_gcdex(f._rep, g._rep, f.dom)
        return f.per(s), f.per(h)

    def _gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
        s, t, h = dup_gcdex(f._rep, g._rep, f.dom)
        return f.per(s), f.per(t), f.per(h)

    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        s = dup_invert(f._rep, g._rep, f.dom)
        return f.per(s)

    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        return f.per(dup_revert(f._rep, n, f.dom))

    def _subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        R = dmp_subresultants(f._rep, g._rep, f.lev, f.dom)
        return list(map(f.per, R))

    def _resultant_includePRS(f, g):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        res, R = dmp_resultant(f._rep, g._rep, f.lev, f.dom, includePRS=True)
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        return res, list(map(f.per, R))

    def _resultant(f, g):
        res = dmp_resultant(f._rep, g._rep, f.lev, f.dom)
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        return res

    def discriminant(f):
        """Computes discriminant of ``f``. """
        res = dmp_discriminant(f._rep, f.lev, f.dom)
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        return res

    def _cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        h, cff, cfg = dmp_inner_gcd(f._rep, g._rep, f.lev, f.dom)
        return f.per(h), f.per(cff), f.per(cfg)

    def _gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        return f.per(dmp_gcd(f._rep, g._rep, f.lev, f.dom))

    def _lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        return f.per(dmp_lcm(f._rep, g._rep, f.lev, f.dom))

    def _cancel(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        cF, cG, F, G = dmp_cancel(f._rep, g._rep, f.lev, f.dom, include=False)
        return cF, cG, f.per(F), f.per(G)

    def _cancel_include(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        F, G = dmp_cancel(f._rep, g._rep, f.lev, f.dom, include=True)
        return f.per(F), f.per(G)

    def _trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f.per(dmp_ground_trunc(f._rep, p, f.lev, f.dom))

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        return f.per(dmp_ground_monic(f._rep, f.lev, f.dom))

    def content(f):
        """Returns GCD of polynomial coefficients. """
        return dmp_ground_content(f._rep, f.lev, f.dom)

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        cont, F = dmp_ground_primitive(f._rep, f.lev, f.dom)
        return cont, f.per(F)

    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        return f.per(dmp_compose(f._rep, g._rep, f.lev, f.dom))

    def _decompose(f):
        """Computes functional decomposition of ``f``. """
        return list(map(f.per, dup_decompose(f._rep, f.dom)))

    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        return f.per(dup_shift(f._rep, a, f.dom))

    def _shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
        return f.per(dmp_shift(f._rep, a, f.lev, f.dom))

    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        return f.per(dup_transform(f._rep, p._rep, q._rep, f.dom))

    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
        return list(map(f.per, dup_sturm(f._rep, f.dom)))

    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        return dup_cauchy_upper_bound(f._rep, f.dom)

    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        return dup_cauchy_lower_bound(f._rep, f.dom)

    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        return dup_mignotte_sep_bound_squared(f._rep, f.dom)

    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        return [ (f.per(g), k) for g, k in dup_gff_list(f._rep, f.dom) ]

    def norm(f):
        """Computes ``Norm(f)``."""
        r = dmp_norm(f._rep, f.lev, f.dom)
        return f.new(r, f.dom.dom, f.lev)

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        s, g, r = dmp_sqf_norm(f._rep, f.lev, f.dom)
        return s, f.per(g), f.new(r, f.dom.dom, f.lev)

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        return f.per(dmp_sqf_part(f._rep, f.lev, f.dom))

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        coeff, factors = dmp_sqf_list(f._rep, f.lev, f.dom, all)
        return coeff, [ (f.per(g), k) for g, k in factors ]

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        factors = dmp_sqf_list_include(f._rep, f.lev, f.dom, all)
        return [ (f.per(g), k) for g, k in factors ]

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """
        coeff, factors = dmp_factor_list(f._rep, f.lev, f.dom)
        return coeff, [ (f.per(g), k) for g, k in factors ]

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        factors = dmp_factor_list_include(f._rep, f.lev, f.dom)
        return [ (f.per(g), k) for g, k in factors ]

    def _isolate_real_roots(f, eps, inf, sup, fast):
        return dup_isolate_real_roots(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        return dup_isolate_real_roots_sqf(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        return dup_isolate_all_roots(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        return dup_isolate_all_roots_sqf(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        return dup_refine_real_root(f._rep, s, t, f.dom, eps=eps, steps=steps, fast=fast)

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        return dup_count_real_roots(f._rep, f.dom, inf=inf, sup=sup)

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        return dup_count_complex_roots(f._rep, f.dom, inf=inf, sup=sup)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        return dmp_zero_p(f._rep, f.lev)

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        return dmp_one_p(f._rep, f.lev, f.dom)

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return dmp_ground_p(f._rep, None, f.lev)

    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        return dmp_sqf_p(f._rep, f.lev, f.dom)

    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        return f.dom.is_one(dmp_ground_LC(f._rep, f.lev, f.dom))

    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        return f.dom.is_one(dmp_ground_content(f._rep, f.lev, f.dom))

    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        return all(sum(monom) <= 1 for monom in dmp_to_dict(f._rep, f.lev, f.dom).keys())

    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        return all(sum(monom) <= 2 for monom in dmp_to_dict(f._rep, f.lev, f.dom).keys())

    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        return len(f.to_dict()) <= 1

    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        return f.homogeneous_order() is not None

    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        return dmp_irreducible_p(f._rep, f.lev, f.dom)

    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        if not f.lev:
            return dup_cyclotomic_p(f._rep, f.dom)
        else:
            return False


class DUP_Flint(DMP):
    """Dense Multivariate Polynomials over `K`. """

    lev = 0

    __slots__ = ('_rep', 'dom', '_cls')

    def __reduce__(self):
        return self.__class__, (self.to_list(), self.dom, self.lev)

    @classmethod
    def _new(cls, rep, dom, lev):
        rep = cls._flint_poly(rep[::-1], dom, lev)
        return cls.from_rep(rep, dom)

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return f._rep.coeffs()[::-1]

    @classmethod
    def _flint_poly(cls, rep, dom, lev):
        assert _supported_flint_domain(dom)
        assert lev == 0
        flint_cls = cls._get_flint_poly_cls(dom)
        return flint_cls(rep)

    @classmethod
    def _get_flint_poly_cls(cls, dom):
        if dom.is_ZZ:
            return flint.fmpz_poly
        elif dom.is_QQ:
            return flint.fmpq_poly
        elif dom.is_FF:
            return dom._poly_ctx
        else:
            raise RuntimeError("Domain %s is not supported with flint" % dom)

    @classmethod
    def from_rep(cls, rep, dom):
        """Create a DMP from the given representation. """

        if dom.is_ZZ:
            assert isinstance(rep, flint.fmpz_poly)
            _cls = flint.fmpz_poly
        elif dom.is_QQ:
            assert isinstance(rep, flint.fmpq_poly)
            _cls = flint.fmpq_poly
        elif dom.is_FF:
            assert isinstance(rep, (flint.nmod_poly, flint.fmpz_mod_poly))
            c = dom.characteristic()
            __cls = type(rep)
            _cls = lambda e: __cls(e, c)
        else:
            raise RuntimeError("Domain %s is not supported with flint" % dom)

        obj = object.__new__(cls)
        obj.dom = dom
        obj._rep = rep
        obj._cls = _cls

        return obj

    def _strict_eq(f, g):
        if type(f) != type(g):
            return False
        return f.dom == g.dom and f._rep == g._rep

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        return f.from_rep(f._cls([coeff]), f.dom)

    def _one(f):
        return f.ground_new(f.dom.one)

    def unify(f, g):
        """Unify representations of two polynomials. """
        raise RuntimeError

    def to_DMP_Python(f):
        """Convert ``f`` to a Python native representation. """
        return DMP_Python._new(f.to_list(), f.dom, f.lev)

    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
        return tuple(f.to_list())

    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
        if dom == QQ and f.dom == ZZ:
            return f.from_rep(flint.fmpq_poly(f._rep), dom)
        elif _supported_flint_domain(dom) and _supported_flint_domain(f.dom):
            # XXX: python-flint should provide a faster way to do this.
            return f.to_DMP_Python()._convert(dom).to_DUP_Flint()
        else:
            raise RuntimeError(f"DUP_Flint: Cannot convert {f.dom} to {dom}")

    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
        coeffs = f._rep.coeffs()[m:n]
        return f.from_rep(f._cls(coeffs), f.dom)

    def _slice_lev(f, m, n, j):
        """Take a continuous subsequence of terms of ``f``. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        if order is None or order.alias == 'lex':
            terms = [ ((n,), c) for n, c in enumerate(f._rep.coeffs()) if c ]
            return terms[::-1]
        else:
            # XXX: InverseOrder (ilex) comes here. We could handle that case
            # efficiently by reversing the coefficients but it is not clear
            # how to test if the order is InverseOrder.
            #
            # Otherwise why would the order ever be different for univariate
            # polynomials?
            return f.to_DMP_Python()._terms(order=order)

    def _lift(f):
        """Convert algebraic coefficients to rationals. """
        # This is for algebraic number fields which DUP_Flint does not support
        raise NotImplementedError

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        # XXX: Check because otherwise this segfaults with python-flint:
        #
        #  >>> flint.fmpz_poly([]).deflation()
        #  Exception (fmpz_poly_deflate). Division by zero.
        #  Aborted (core dumped
        #
        if f.is_zero:
            return (1,), f
        g, n = f._rep.deflation()
        return (n,), f.from_rep(g, f.dom)

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        # Ground domain would need to be a poly ring
        raise NotImplementedError

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _exclude(f):
        """Remove useless generators from ``f``. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _permute(f, P):
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        # XXX: python-flint should have primitive, content, etc methods.
        J, F = f.to_DMP_Python().terms_gcd()
        return J, F.to_DUP_Flint()

    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.from_rep(f._rep + c, f.dom)

    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.from_rep(f._rep - c, f.dom)

    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f.from_rep(f._rep * c, f.dom)

    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f.from_rep(f._rep // c, f.dom)

    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by an element of the ground domain. """
        q, r = divmod(f._rep, c)
        if r:
            raise ExactQuotientFailed(f, c)
        return f.from_rep(q, f.dom)

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        return f.to_DMP_Python().abs().to_DUP_Flint()

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.from_rep(-f._rep, f.dom)

    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        return f.from_rep(f._rep + g._rep, f.dom)

    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        return f.from_rep(f._rep - g._rep, f.dom)

    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        return f.from_rep(f._rep * g._rep, f.dom)

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        return f.from_rep(f._rep ** 2, f.dom)

    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        return f.from_rep(f._rep ** n, f.dom)

    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        q, r = divmod(g.LC()**d * f._rep, g._rep)
        return f.from_rep(q, f.dom), f.from_rep(r, f.dom)

    def _prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        q = (g.LC()**d * f._rep) % g._rep
        return f.from_rep(q, f.dom)

    def _pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        r = (g.LC()**d * f._rep) // g._rep
        return f.from_rep(r, f.dom)

    def _pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        q, r = divmod(g.LC()**d * f._rep, g._rep)
        if r:
            raise ExactQuotientFailed(f, g)
        return f.from_rep(q, f.dom)

    def _div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        if f.dom.is_Field:
            q, r = divmod(f._rep, g._rep)
            return f.from_rep(q, f.dom), f.from_rep(r, f.dom)
        else:
            # XXX: python-flint defines division in ZZ[x] differently
            q, r = f.to_DMP_Python()._div(g.to_DMP_Python())
            return q.to_DUP_Flint(), r.to_DUP_Flint()

    def _rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        return f.from_rep(f._rep % g._rep, f.dom)

    def _quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        return f.from_rep(f._rep // g._rep, f.dom)

    def _exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        q, r = f._div(g)
        if r:
            raise ExactQuotientFailed(f, g)
        return q

    def _degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        d = f._rep.degree()
        if d == -1:
            d = ninf
        return d

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        return ( f._degree() ,)

    def total_degree(f):
        """Returns the total degree of ``f``. """
        return f._degree()

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return f._rep[f._rep.degree()]

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return f._rep[0]

    def _nth(f, N):
        """Returns the ``n``-th coefficient of ``f``. """
        [n] = N
        return f._rep[n]

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        return f.to_DMP_Python().max_norm()

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        return f.to_DMP_Python().l1_norm()

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        return f.to_DMP_Python().l2_norm_squared()

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        R = f.dom
        if R.is_QQ:
            denom = f._rep.denom()
            numer = f.from_rep(f._cls(f._rep.numer()), f.dom)
            return denom, numer
        elif R.is_ZZ or R.is_FiniteField:
            return R.one, f
        else:
            raise NotImplementedError

    def _integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        assert j == 0
        if f.dom.is_Field:
            rep = f._rep
            for i in range(m):
                rep = rep.integral()
            return f.from_rep(rep, f.dom)
        else:
            return f.to_DMP_Python()._integrate(m=m, j=j).to_DUP_Flint()

    def _diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f``. """
        assert j == 0
        rep = f._rep
        for i in range(m):
            rep = rep.derivative()
        return f.from_rep(rep, f.dom)

    def _eval(f, a):
        # XXX: This method is called with many different input types. Ideally
        # we could use e.g. fmpz_poly.__call__ here but more thought needs to
        # go into which types this is supposed to be called with and what types
        # it should return.
        return f.to_DMP_Python()._eval(a)

    def _eval_lev(f, a, j):
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm. """
        s, h = f.to_DMP_Python()._half_gcdex(g.to_DMP_Python())
        return s.to_DUP_Flint(), h.to_DUP_Flint()

    def _gcdex(f, g):
        """Extended Euclidean algorithm. """
        h, s, t = f._rep.xgcd(g._rep)
        return f.from_rep(s, f.dom), f.from_rep(t, f.dom), f.from_rep(h, f.dom)

    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        R = f.dom
        if R.is_Field:
            gcd, F_inv, _ = f._rep.xgcd(g._rep)
            # XXX: Should be gcd != 1 but nmod_poly does not compare equal to
            # other types.
            if gcd != 0*gcd + 1:
                raise NotInvertible("zero divisor")
            return f.from_rep(F_inv, R)
        else:
            # fmpz_poly does not have xgcd or invert and this is not well
            # defined in general.
            return f.to_DMP_Python()._invert(g.to_DMP_Python()).to_DUP_Flint()

    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        # XXX: Use fmpz_series etc for reversion?
        # Maybe python-flint should provide revert for fmpz_poly...
        return f.to_DMP_Python()._revert(n).to_DUP_Flint()

    def _subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        # XXX: Maybe _fmpz_poly_pseudo_rem_cohen could be used...
        R = f.to_DMP_Python()._subresultants(g.to_DMP_Python())
        return [ g.to_DUP_Flint() for g in R ]

    def _resultant_includePRS(f, g):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        # XXX: Maybe _fmpz_poly_pseudo_rem_cohen could be used...
        res, R = f.to_DMP_Python()._resultant_includePRS(g.to_DMP_Python())
        return res, [ g.to_DUP_Flint() for g in R ]

    def _resultant(f, g):
        """Computes resultant of ``f`` and ``g``. """
        # XXX: Use fmpz_mpoly etc when possible...
        return f.to_DMP_Python()._resultant(g.to_DMP_Python())

    def discriminant(f):
        """Computes discriminant of ``f``. """
        # XXX: Use fmpz_mpoly etc when possible...
        return f.to_DMP_Python().discriminant()

    def _cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        h = f.gcd(g)
        return h, f.exquo(h), g.exquo(h)

    def _gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        return f.from_rep(f._rep.gcd(g._rep), f.dom)

    def _lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        # XXX: python-flint should have a lcm method
        if not (f and g):
            return f.ground_new(f.dom.zero)

        l = f._mul(g)._exquo(f._gcd(g))

        if l.dom.is_Field:
            l = l.monic()
        elif l.LC() < 0:
            l = l.neg()

        return l

    def _cancel(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        assert f.dom == g.dom
        R = f.dom

        # Think carefully about how to handle denominators and coefficient
        # canonicalisation if more domains are permitted...
        assert R.is_ZZ or R.is_QQ or R.is_FiniteField

        if R.is_FiniteField:
            h = f._gcd(g)
            F, G = f.exquo(h), g.exquo(h)
            return R.one, R.one, F, G

        if R.is_QQ:
            cG, F = f.clear_denoms()
            cF, G = g.clear_denoms()
        else:
            cG, F = R.one, f
            cF, G = R.one, g

        cH = cF.gcd(cG)
        cF, cG = cF // cH, cG // cH

        H = F._gcd(G)
        F, G = F.exquo(H), G.exquo(H)

        f_neg = F.LC() < 0
        g_neg = G.LC() < 0

        if f_neg and g_neg:
            F, G = F.neg(), G.neg()
        elif f_neg:
            cF, F = -cF, F.neg()
        elif g_neg:
            cF, G = -cF, G.neg()

        return cF, cG, F, G

    def _cancel_include(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        cF, cG, F, G = f._cancel(g)
        return F._mul_ground(cF), G._mul_ground(cG)

    def _trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f.to_DMP_Python()._trunc(p).to_DUP_Flint()

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        # XXX: python-flint should add monic
        return f._exquo_ground(f.LC())

    def content(f):
        """Returns GCD of polynomial coefficients. """
        # XXX: python-flint should have a content method
        return f.to_DMP_Python().content()

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        cont = f.content()
        if f.is_zero:
            return f.dom.zero, f
        prim = f._exquo_ground(cont)
        return cont, prim

    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        return f.from_rep(f._rep(g._rep), f.dom)

    def _decompose(f):
        """Computes functional decomposition of ``f``. """
        return [ g.to_DUP_Flint() for g in f.to_DMP_Python()._decompose() ]

    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        x_plus_a = f._cls([a, f.dom.one])
        return f.from_rep(f._rep(x_plus_a), f.dom)

    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        F, P, Q = f.to_DMP_Python(), p.to_DMP_Python(), q.to_DMP_Python()
        return F.transform(P, Q).to_DUP_Flint()

    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
        return [ g.to_DUP_Flint() for g in f.to_DMP_Python()._sturm() ]

    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        return f.to_DMP_Python()._cauchy_upper_bound()

    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        return f.to_DMP_Python()._cauchy_lower_bound()

    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        return f.to_DMP_Python()._mignotte_sep_bound_squared()

    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        F = f.to_DMP_Python()
        return [ (g.to_DUP_Flint(), k) for g, k in F.gff_list() ]

    def norm(f):
        """Computes ``Norm(f)``."""
        # This is for algebraic number fields which DUP_Flint does not support
        raise NotImplementedError

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        # This is for algebraic number fields which DUP_Flint does not support
        raise NotImplementedError

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        return f._exquo(f._gcd(f._diff()))

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        # XXX: python-flint should provide square free factorisation.
        coeff, factors = f.to_DMP_Python().sqf_list(all=all)
        return coeff, [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        factors = f.to_DMP_Python().sqf_list_include(all=all)
        return [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """

        if f.dom.is_ZZ or f.dom.is_FF:
            # python-flint matches polys here
            coeff, factors = f._rep.factor()
            factors = [ (f.from_rep(g, f.dom), k) for g, k in factors ]

        elif f.dom.is_QQ:
            # python-flint returns monic factors over QQ whereas polys returns
            # denominator free factors.
            coeff, factors = f._rep.factor()
            factors_monic = [ (f.from_rep(g, f.dom), k) for g, k in factors ]

            # Absorb the denominators into coeff
            factors = []
            for g, k in factors_monic:
                d, g = g.clear_denoms()
                coeff /= d**k
                factors.append((g, k))

        else:
            # Check carefully when adding more domains here...
            raise RuntimeError("Domain %s is not supported with flint" % f.dom)

        # We need to match the way that polys orders the factors
        factors = f._sort_factors(factors)

        return coeff, factors

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        # XXX: factor_list_include seems to be broken in general:
        #
        #   >>> Poly(2*(x - 1)**3, x).factor_list_include()
        #   [(Poly(2*x - 2, x, domain='ZZ'), 3)]
        #
        # Let's not try to implement it here.
        factors = f.to_DMP_Python().factor_list_include()
        return [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def _sort_factors(f, factors):
        """Sort a list of factors to canonical order. """
        # Convert the factors to lists and use _sort_factors from polys
        factors = [ (g.to_list(), k) for g, k in factors ]
        factors = _sort_factors(factors, multiple=True)
        to_dup_flint = lambda g: f.from_rep(f._cls(g[::-1]), f.dom)
        return [ (to_dup_flint(g), k) for g, k in factors ]

    def _isolate_real_roots(f, eps, inf, sup, fast):
        return f.to_DMP_Python()._isolate_real_roots(eps, inf, sup, fast)

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        return f.to_DMP_Python()._isolate_real_roots_sqf(eps, inf, sup, fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        # fmpz_poly and fmpq_poly have a complex_roots method that could be
        # used here. It probably makes more sense to add analogous methods in
        # python-flint though.
        return f.to_DMP_Python()._isolate_all_roots(eps, inf, sup, fast)

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        return f.to_DMP_Python()._isolate_all_roots_sqf(eps, inf, sup, fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        return f.to_DMP_Python()._refine_real_root(s, t, eps, steps, fast)

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        return f.to_DMP_Python().count_real_roots(inf=inf, sup=sup)

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        return f.to_DMP_Python().count_complex_roots(inf=inf, sup=sup)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        return not f._rep

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        return f._rep == f.dom.one

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return f._rep.degree() <= 0

    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        return f._rep.degree() <= 1

    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        return f._rep.degree() <= 2

    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        fr = f._rep
        return fr.degree() < 0 or not any(fr[n] for n in range(fr.degree()))

    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        return f.LC() == f.dom.one

    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        return f.to_DMP_Python().is_primitive

    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        return f.to_DMP_Python().is_homogeneous

    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        g = f._rep.gcd(f._rep.derivative())
        return g.degree() <= 0

    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        _, factors = f._rep.factor()
        if len(factors) == 0:
            return True
        elif len(factors) == 1:
            return factors[0][1] == 1
        else:
            return False

    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        if f.dom.is_QQ:
            try:
                f = f.convert(ZZ)
            except CoercionFailed:
                return False
        if f.dom.is_ZZ:
            return bool(f._rep.is_cyclotomic())
        else:
            # This is what dup_cyclotomic_p does...
            return False


def init_normal_DMF(num, den, lev, dom):
    return DMF(dmp_normal(num, lev, dom),
               dmp_normal(den, lev, dom), dom, lev)


class DMF(PicklableWithSlots, CantSympify):
    """Dense Multivariate Fractions over `K`. """

    __slots__ = ('num', 'den', 'lev', 'dom')

    def __init__(self, rep, dom, lev=None):
        num, den, lev = self._parse(rep, dom, lev)
        num, den = dmp_cancel(num, den, lev, dom)

        self.num = num
        self.den = den
        self.lev = lev
        self.dom = dom

    @classmethod
    def new(cls, rep, dom, lev=None):
        num, den, lev = cls._parse(rep, dom, lev)

        obj = object.__new__(cls)

        obj.num = num
        obj.den = den
        obj.lev = lev
        obj.dom = dom

        return obj

    def ground_new(self, rep):
        return self.new(rep, self.dom, self.lev)

    @classmethod
    def _parse(cls, rep, dom, lev=None):
        if isinstance(rep, tuple):
            num, den = rep

            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)

                if isinstance(den, dict):
                    den = dmp_from_dict(den, lev, dom)
            else:
                num, num_lev = dmp_validate(num)
                den, den_lev = dmp_validate(den)

                if num_lev == den_lev:
                    lev = num_lev
                else:
                    raise ValueError('inconsistent number of levels')

            if dmp_zero_p(den, lev):
                raise ZeroDivisionError('fraction denominator')

            if dmp_zero_p(num, lev):
                den = dmp_one(lev, dom)
            else:
                if dmp_negative_p(den, lev, dom):
                    num = dmp_neg(num, lev, dom)
                    den = dmp_neg(den, lev, dom)
        else:
            num = rep

            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)
                elif not isinstance(num, list):
                    num = dmp_ground(dom.convert(num), lev)
            else:
                num, lev = dmp_validate(num)

            den = dmp_one(lev, dom)

        return num, den, lev

    def __repr__(f):
        return "%s((%s, %s), %s)" % (f.__class__.__name__, f.num, f.den, f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, dmp_to_tuple(f.num, f.lev),
            dmp_to_tuple(f.den, f.lev), f.lev, f.dom))

    def poly_unify(f, g):
        """Unify a multivariate fraction and a polynomial. """
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return (f.lev, f.dom, f.per, (f.num, f.den), g._rep)
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)

            F = (dmp_convert(f.num, lev, f.dom, dom),
                 dmp_convert(f.den, lev, f.dom, dom))

            G = dmp_convert(g._rep, lev, g.dom, dom)

            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num/den
                    else:
                        lev = lev - 1

                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)

                return f.__class__.new((num, den), dom, lev)

            return lev, dom, per, F, G

    def frac_unify(f, g):
        """Unify representations of two multivariate fractions. """
        if not isinstance(g, DMF) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return (f.lev, f.dom, f.per, (f.num, f.den),
                                         (g.num, g.den))
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)

            F = (dmp_convert(f.num, lev, f.dom, dom),
                 dmp_convert(f.den, lev, f.dom, dom))

            G = (dmp_convert(g.num, lev, g.dom, dom),
                 dmp_convert(g.den, lev, g.dom, dom))

            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num/den
                    else:
                        lev = lev - 1

                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)

                return f.__class__.new((num, den), dom, lev)

            return lev, dom, per, F, G

    def per(f, num, den, cancel=True, kill=False):
        """Create a DMF out of the given representation. """
        lev, dom = f.lev, f.dom

        if kill:
            if not lev:
                return num/den
            else:
                lev -= 1

        if cancel:
            num, den = dmp_cancel(num, den, lev, dom)

        return f.__class__.new((num, den), dom, lev)

    def half_per(f, rep, kill=False):
        """Create a DMP out of the given representation. """
        lev = f.lev

        if kill:
            if not lev:
                return rep
            else:
                lev -= 1

        return DMP(rep, f.dom, lev)

    @classmethod
    def zero(cls, lev, dom):
        return cls.new(0, dom, lev)

    @classmethod
    def one(cls, lev, dom):
        return cls.new(1, dom, lev)

    def numer(f):
        """Returns the numerator of ``f``. """
        return f.half_per(f.num)

    def denom(f):
        """Returns the denominator of ``f``. """
        return f.half_per(f.den)

    def cancel(f):
        """Remove common factors from ``f.num`` and ``f.den``. """
        return f.per(f.num, f.den)

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.per(dmp_neg(f.num, f.lev, f.dom), f.den, cancel=False)

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f + f.ground_new(c)

    def add(f, g):
        """Add two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_add_mul(F_num, F_den, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_add(dmp_mul(F_num, G_den, lev, dom),
                          dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def sub(f, g):
        """Subtract two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_sub_mul(F_num, F_den, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_sub(dmp_mul(F_num, G_den, lev, dom),
                          dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def mul(f, g):
        """Multiply two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_mul(F_num, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_mul(F_num, G_num, lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if isinstance(n, int):
            num, den = f.num, f.den
            if n < 0:
                num, den, n = den, num, -n
            return f.per(dmp_pow(num, n, f.lev, f.dom),
                         dmp_pow(den, n, f.lev, f.dom), cancel=False)
        else:
            raise TypeError("``int`` expected, got %s" % type(n))

    def quo(f, g):
        """Computes quotient of fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = F_num, dmp_mul(F_den, G, lev, dom)
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_mul(F_num, G_den, lev, dom)
            den = dmp_mul(F_den, G_num, lev, dom)

        return per(num, den)

    exquo = quo

    def invert(f, check=True):
        """Computes inverse of a fraction ``f``. """
        return f.per(f.den, f.num, cancel=False)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero fraction. """
        return dmp_zero_p(f.num, f.lev)

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit fraction. """
        return dmp_one_p(f.num, f.lev, f.dom) and \
            dmp_one_p(f.den, f.lev, f.dom)

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.add(g)
        elif g in f.dom:
            return f.add_ground(f.dom.convert(g))

        try:
            return f.add(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.sub(g)

        try:
            return f.sub(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.mul(g)

        try:
            return f.mul(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __rmul__(f, g):
        return f.__mul__(g)

    def __pow__(f, n):
        return f.pow(n)

    def __truediv__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.quo(g)

        try:
            return f.quo(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __rtruediv__(self, g):
        return self.invert(check=False)*g

    def __eq__(f, g):
        try:
            if isinstance(g, DMP):
                _, _, _, (F_num, F_den), G = f.poly_unify(g)

                if f.lev == g.lev:
                    return dmp_one_p(F_den, f.lev, f.dom) and F_num == G
            else:
                _, _, _, F, G = f.frac_unify(g)

                if f.lev == g.lev:
                    return F == G
        except UnificationFailed:
            pass

        return False

    def __ne__(f, g):
        try:
            if isinstance(g, DMP):
                _, _, _, (F_num, F_den), G = f.poly_unify(g)

                if f.lev == g.lev:
                    return not (dmp_one_p(F_den, f.lev, f.dom) and F_num == G)
            else:
                _, _, _, F, G = f.frac_unify(g)

                if f.lev == g.lev:
                    return F != G
        except UnificationFailed:
            pass

        return True

    def __lt__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F < G

    def __le__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F <= G

    def __gt__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F > G

    def __ge__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F >= G

    def __bool__(f):
        return not dmp_zero_p(f.num, f.lev)


def init_normal_ANP(rep, mod, dom):
    return ANP(dup_normal(rep, dom),
               dup_normal(mod, dom), dom)


class ANP(CantSympify):
    """Dense Algebraic Number Polynomials over a field. """

    __slots__ = ('_rep', '_mod', 'dom')

    def __new__(cls, rep, mod, dom):
        if isinstance(rep, DMP):
            pass
        elif type(rep) is dict: # don't use isinstance
            rep = DMP(dup_from_dict(rep, dom), dom, 0)
        else:
            if isinstance(rep, list):
                rep = [dom.convert(a) for a in rep]
            else:
                rep = [dom.convert(rep)]
            rep = DMP(dup_strip(rep), dom, 0)

        if isinstance(mod, DMP):
            pass
        elif isinstance(mod, dict):
            mod = DMP(dup_from_dict(mod, dom), dom, 0)
        else:
            mod = DMP(dup_strip(mod), dom, 0)

        return cls.new(rep, mod, dom)

    @classmethod
    def new(cls, rep, mod, dom):
        if not (rep.dom == mod.dom == dom):
            raise RuntimeError("Inconsistent domain")
        obj = super().__new__(cls)
        obj._rep = rep
        obj._mod = mod
        obj.dom = dom
        return obj

    # XXX: It should be possible to use __getnewargs__ rather than __reduce__
    # but it doesn't work for some reason. Probably this would be easier if
    # python-flint supported pickling for polynomial types.
    def __reduce__(self):
        return ANP, (self.rep, self.mod, self.dom)

    @property
    def rep(self):
        return self._rep.to_list()

    @property
    def mod(self):
        return self.mod_to_list()

    def to_DMP(self):
        return self._rep

    def mod_to_DMP(self):
        return self._mod

    def per(f, rep):
        return f.new(rep, f._mod, f.dom)

    def __repr__(f):
        return "%s(%s, %s, %s)" % (f.__class__.__name__, f._rep.to_list(), f._mod.to_list(), f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, f.to_tuple(), f._mod.to_tuple(), f.dom))

    def convert(f, dom):
        """Convert ``f`` to a ``ANP`` over a new domain. """
        if f.dom == dom:
            return f
        else:
            return f.new(f._rep.convert(dom), f._mod.convert(dom), dom)

    def unify(f, g):
        """Unify representations of two algebraic numbers. """

        # XXX: This unify method is not used any more because unify_ANP is used
        # instead.

        if not isinstance(g, ANP) or f.mod != g.mod:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return f.dom, f.per, f.rep, g.rep, f.mod
        else:
            dom = f.dom.unify(g.dom)

            F = dup_convert(f.rep, f.dom, dom)
            G = dup_convert(g.rep, g.dom, dom)

            if dom != f.dom and dom != g.dom:
                mod = dup_convert(f.mod, f.dom, dom)
            else:
                if dom == f.dom:
                    mod = f.mod
                else:
                    mod = g.mod

            per = lambda rep: ANP(rep, mod, dom)

        return dom, per, F, G, mod

    def unify_ANP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
        if not isinstance(g, ANP) or f._mod != g._mod:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        # The domain is almost always QQ but there are some tests involving ZZ
        if f.dom != g.dom:
            dom = f.dom.unify(g.dom)
            f = f.convert(dom)
            g = g.convert(dom)

        return f._rep, g._rep, f._mod, f.dom

    @classmethod
    def zero(cls, mod, dom):
        return ANP(0, mod, dom)

    @classmethod
    def one(cls, mod, dom):
        return ANP(1, mod, dom)

    def to_dict(f):
        """Convert ``f`` to a dict representation with native coefficients. """
        return f._rep.to_dict()

    def to_sympy_dict(f):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
        rep = dmp_to_dict(f.rep, 0, f.dom)

        for k, v in rep.items():
            rep[k] = f.dom.to_sympy(v)

        return rep

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return f._rep.to_list()

    def mod_to_list(f):
        """Return ``f.mod`` as a list with native coefficients. """
        return f._mod.to_list()

    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
        return [ f.dom.to_sympy(c) for c in f.to_list() ]

    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
        return f._rep.to_tuple()

    @classmethod
    def from_list(cls, rep, mod, dom):
        return ANP(dup_strip(list(map(dom.convert, rep))), mod, dom)

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.per(f._rep.add_ground(c))

    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.per(f._rep.sub_ground(c))

    def mul_ground(f, c):
        """Multiply ``f`` by an element of the ground domain. """
        return f.per(f._rep.mul_ground(c))

    def quo_ground(f, c):
        """Quotient of ``f`` by an element of the ground domain. """
        return f.per(f._rep.quo_ground(c))

    def neg(f):
        return f.per(f._rep.neg())

    def add(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.add(G), mod, dom)

    def sub(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.sub(G), mod, dom)

    def mul(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.mul(G).rem(mod), mod, dom)

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if not isinstance(n, int):
            raise TypeError("``int`` expected, got %s" % type(n))

        mod = f._mod
        F = f._rep

        if n < 0:
            F, n = F.invert(mod), -n

        # XXX: Need a pow_mod method for DMP
        return f.new(F.pow(n).rem(f._mod), mod, f.dom)

    def exquo(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.mul(G.invert(mod)).rem(mod), mod, dom)

    def div(f, g):
        return f.exquo(g), f.zero(f._mod, f.dom)

    def quo(f, g):
        return f.exquo(g)

    def rem(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        s, h = F.half_gcdex(G)

        if h.is_one:
            return f.zero(mod, dom)
        else:
            raise NotInvertible("zero divisor")

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return f._rep.LC()

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return f._rep.TC()

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero algebraic number. """
        return f._rep.is_zero

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit algebraic number. """
        return f._rep.is_one

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return f._rep.is_ground

    def __pos__(f):
        return f

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, ANP):
            return f.add(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.add_ground(g)

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, ANP):
            return f.sub(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.sub_ground(g)

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, ANP):
            return f.mul(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.mul_ground(g)

    def __rmul__(f, g):
        return f.__mul__(g)

    def __pow__(f, n):
        return f.pow(n)

    def __divmod__(f, g):
        return f.div(g)

    def __mod__(f, g):
        return f.rem(g)

    def __truediv__(f, g):
        if isinstance(g, ANP):
            return f.quo(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.quo_ground(g)

    def __eq__(f, g):
        try:
            F, G, _, _ = f.unify_ANP(g)
        except UnificationFailed:
            return NotImplemented
        return F == G

    def __ne__(f, g):
        try:
            F, G, _, _ = f.unify_ANP(g)
        except UnificationFailed:
            return NotImplemented
        return F != G

    def __lt__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F < G

    def __le__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F <= G

    def __gt__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F > G

    def __ge__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F >= G

    def __bool__(f):
        return bool(f._rep)
