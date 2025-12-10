"""Implementation of RootOf class and related tools. """



from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
    symbols, sympify, Rational, Dummy)
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    GeneratorsNeeded,
    PolynomialError,
    DomainError)
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
    roots_linear, roots_quadratic, roots_binomial,
    preprocess_roots, roots)
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
    dup_isolate_complex_roots_sqf,
    dup_isolate_real_roots_sqf)
from sympy.utilities import lambdify, public, sift, numbered_symbols

from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain


__all__ = ['CRootOf']



class _pure_key_dict:
    """A minimal dictionary that makes sure that the key is a
    univariate PurePoly instance.

    Examples
    ========

    Only the following actions are guaranteed:

    >>> from sympy.polys.rootoftools import _pure_key_dict
    >>> from sympy import PurePoly
    >>> from sympy.abc import x, y

    1) creation

    >>> P = _pure_key_dict()

    2) assignment for a PurePoly or univariate polynomial

    >>> P[x] = 1
    >>> P[PurePoly(x - y, x)] = 2

    3) retrieval based on PurePoly key comparison (use this
       instead of the get method)

    >>> P[y]
    1

    4) KeyError when trying to retrieve a nonexisting key

    >>> P[y + 1]
    Traceback (most recent call last):
    ...
    KeyError: PurePoly(y + 1, y, domain='ZZ')

    5) ability to query with ``in``

    >>> x + 1 in P
    False

    NOTE: this is a *not* a dictionary. It is a very basic object
    for internal use that makes sure to always address its cache
    via PurePoly instances. It does not, for example, implement
    ``get`` or ``setdefault``.
    """
    def __init__(self):
        self._dict = {}

    def __getitem__(self, k):
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise KeyError
            k = PurePoly(k, expand=False)
        return self._dict[k]

    def __setitem__(self, k, v):
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise ValueError('expecting univariate expression')
            k = PurePoly(k, expand=False)
        self._dict[k] = v

    def __contains__(self, k):
        try:
            self[k]
            return True
        except KeyError:
            return False

_reals_cache = _pure_key_dict()
_complexes_cache = _pure_key_dict()


def _pure_factors(poly):
    _, factors = poly.factor_list()
    return [(PurePoly(f, expand=False), m) for f, m in factors]


def _imag_count_of_factor(f):
    """Return the number of imaginary roots for irreducible
    univariate polynomial ``f``.
    """
    terms = [(i, j) for (i,), j in f.terms()]
    if any(i % 2 for i, j in terms):
        return 0
    # update signs
    even = [(i, I**i*j) for i, j in terms]
    even = Poly.from_dict(dict(even), Dummy('x'))
    return int(even.count_roots(-oo, oo))


@public
def rootof(f, x, index=None, radicals=True, expand=True):
    """An indexed root of a univariate polynomial.

    Returns either a :obj:`ComplexRootOf` object or an explicit
    expression involving radicals.

    Parameters
    ==========

    f : Expr
        Univariate polynomial.
    x : Symbol, optional
        Generator for ``f``.
    index : int or Integer
    radicals : bool
               Return a radical expression if possible.
    expand : bool
             Expand ``f``.
    """
    return CRootOf(f, x, index=index, radicals=radicals, expand=expand)


@public
class RootOf(Expr):
    """Represents a root of a univariate polynomial.

    Base class for roots of different kinds of polynomials.
    Only complex roots are currently supported.
    """

    __slots__ = ('poly',)

    def __new__(cls, f, x, index=None, radicals=True, expand=True):
        """Construct a new ``CRootOf`` object for ``k``-th root of ``f``."""
        return rootof(f, x, index=index, radicals=radicals, expand=expand)

@public
class ComplexRootOf(RootOf):
    """Represents an indexed complex root of a polynomial.

    Roots of a univariate polynomial separated into disjoint
    real or complex intervals and indexed in a fixed order:

    * real roots come first and are sorted in increasing order;
    * complex roots come next and are sorted primarily by increasing
      real part, secondarily by increasing imaginary part.

    Currently only rational coefficients are allowed.
    Can be imported as ``CRootOf``. To avoid confusion, the
    generator must be a Symbol.


    Examples
    ========

    >>> from sympy import CRootOf, rootof
    >>> from sympy.abc import x

    CRootOf is a way to reference a particular root of a
    polynomial. If there is a rational root, it will be returned:

    >>> CRootOf.clear_cache()  # for doctest reproducibility
    >>> CRootOf(x**2 - 4, 0)
    -2

    Whether roots involving radicals are returned or not
    depends on whether the ``radicals`` flag is true (which is
    set to True with rootof):

    >>> CRootOf(x**2 - 3, 0)
    CRootOf(x**2 - 3, 0)
    >>> CRootOf(x**2 - 3, 0, radicals=True)
    -sqrt(3)
    >>> rootof(x**2 - 3, 0)
    -sqrt(3)

    The following cannot be expressed in terms of radicals:

    >>> r = rootof(4*x**5 + 16*x**3 + 12*x**2 + 7, 0); r
    CRootOf(4*x**5 + 16*x**3 + 12*x**2 + 7, 0)

    The root bounds can be seen, however, and they are used by the
    evaluation methods to get numerical approximations for the root.

    >>> interval = r._get_interval(); interval
    (-1, 0)
    >>> r.evalf(2)
    -0.98

    The evalf method refines the width of the root bounds until it
    guarantees that any decimal approximation within those bounds
    will satisfy the desired precision. It then stores the refined
    interval so subsequent requests at or below the requested
    precision will not have to recompute the root bounds and will
    return very quickly.

    Before evaluation above, the interval was

    >>> interval
    (-1, 0)

    After evaluation it is now

    >>> r._get_interval() # doctest: +SKIP
    (-165/169, -206/211)

    To reset all intervals for a given polynomial, the :meth:`_reset` method
    can be called from any CRootOf instance of the polynomial:

    >>> r._reset()
    >>> r._get_interval()
    (-1, 0)

    The :meth:`eval_approx` method will also find the root to a given
    precision but the interval is not modified unless the search
    for the root fails to converge within the root bounds. And
    the secant method is used to find the root. (The ``evalf``
    method uses bisection and will always update the interval.)

    >>> r.eval_approx(2)
    -0.98

    The interval needed to be slightly updated to find that root:

    >>> r._get_interval()
    (-1, -1/2)

    The ``evalf_rational`` will compute a rational approximation
    of the root to the desired accuracy or precision.

    >>> r.eval_rational(n=2)
    -69629/71318

    >>> t = CRootOf(x**3 + 10*x + 1, 1)
    >>> t.eval_rational(1e-1)
    15/256 - 805*I/256
    >>> t.eval_rational(1e-1, 1e-4)
    3275/65536 - 414645*I/131072
    >>> t.eval_rational(1e-4, 1e-4)
    6545/131072 - 414645*I/131072
    >>> t.eval_rational(n=2)
    104755/2097152 - 6634255*I/2097152

    Notes
    =====

    Although a PurePoly can be constructed from a non-symbol generator
    RootOf instances of non-symbols are disallowed to avoid confusion
    over what root is being represented.

    >>> from sympy import exp, PurePoly
    >>> PurePoly(x) == PurePoly(exp(x))
    True
    >>> CRootOf(x - 1, 0)
    1
    >>> CRootOf(exp(x) - 1, 0)  # would correspond to x == 0
    Traceback (most recent call last):
    ...
    sympy.polys.polyerrors.PolynomialError: generator must be a Symbol

    See Also
    ========

    eval_approx
    eval_rational

    """

    __slots__ = ('index',)
    is_complex = True
    is_number = True
    is_finite = True
    is_algebraic = True

    def __new__(cls, f, x, index=None, radicals=False, expand=True):
        """ Construct an indexed complex root of a polynomial.

        See ``rootof`` for the parameters.

        The default value of ``radicals`` is ``False`` to satisfy
        ``eval(srepr(expr) == expr``.
        """
        x = sympify(x)

        if index is None and x.is_Integer:
            x, index = None, x
        else:
            index = sympify(index)

        if index is not None and index.is_Integer:
            index = int(index)
        else:
            raise ValueError("expected an integer root index, got %s" % index)

        poly = PurePoly(f, x, greedy=False, expand=expand)

        if not poly.is_univariate:
            raise PolynomialError("only univariate polynomials are allowed")

        if not poly.gen.is_Symbol:
            # PurePoly(sin(x) + 1) == PurePoly(x + 1) but the roots of
            # x for each are not the same: issue 8617
            raise PolynomialError("generator must be a Symbol")

        degree = poly.degree()

        if degree <= 0:
            raise PolynomialError("Cannot construct CRootOf object for %s" % f)

        if index < -degree or index >= degree:
            raise IndexError("root index out of [%d, %d] range, got %d" %
                             (-degree, degree - 1, index))
        elif index < 0:
            index += degree

        dom = poly.get_domain()

        if not dom.is_Exact:
            poly = poly.to_exact()

        roots = cls._roots_trivial(poly, radicals)

        if roots is not None:
            return roots[index]

        coeff, poly = preprocess_roots(poly)
        dom = poly.get_domain()

        if not dom.is_ZZ:
            raise NotImplementedError("CRootOf is not supported over %s" % dom)

        root = cls._indexed_root(poly, index, lazy=True)
        return coeff * cls._postprocess_root(root, radicals)

    @classmethod
    def _new(cls, poly, index):
        """Construct new ``CRootOf`` object from raw data. """
        obj = Expr.__new__(cls)

        obj.poly = PurePoly(poly)
        obj.index = index

        try:
            _reals_cache[obj.poly] = _reals_cache[poly]
            _complexes_cache[obj.poly] = _complexes_cache[poly]
        except KeyError:
            pass

        return obj

    def _hashable_content(self):
        return (self.poly, self.index)

    @property
    def expr(self):
        return self.poly.as_expr()

    @property
    def args(self):
        return (self.expr, Integer(self.index))

    @property
    def free_symbols(self):
        # CRootOf currently only works with univariate expressions
        # whose poly attribute should be a PurePoly with no free
        # symbols
        return set()

    def _eval_is_real(self):
        """Return ``True`` if the root is real. """
        self._ensure_reals_init()
        return self.index < len(_reals_cache[self.poly])

    def _eval_is_imaginary(self):
        """Return ``True`` if the root is imaginary. """
        self._ensure_reals_init()
        if self.index >= len(_reals_cache[self.poly]):
            ivl = self._get_interval()
            return ivl.ax*ivl.bx <= 0  # all others are on one side or the other
        return False  # XXX is this necessary?

    @classmethod
    def real_roots(cls, poly, radicals=True):
        """Get real roots of a polynomial. """
        return cls._get_roots("_real_roots", poly, radicals)

    @classmethod
    def all_roots(cls, poly, radicals=True):
        """Get real and complex roots of a polynomial. """
        return cls._get_roots("_all_roots", poly, radicals)

    @classmethod
    def _get_reals_sqf(cls, currentfactor, use_cache=True):
        """Get real root isolating intervals for a square-free factor."""
        if use_cache and currentfactor in _reals_cache:
            real_part = _reals_cache[currentfactor]
        else:
            _reals_cache[currentfactor] = real_part = \
                dup_isolate_real_roots_sqf(
                    currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)

        return real_part

    @classmethod
    def _get_complexes_sqf(cls, currentfactor, use_cache=True):
        """Get complex root isolating intervals for a square-free factor."""
        if use_cache and currentfactor in _complexes_cache:
            complex_part = _complexes_cache[currentfactor]
        else:
            _complexes_cache[currentfactor] = complex_part = \
                dup_isolate_complex_roots_sqf(
                currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)
        return complex_part

    @classmethod
    def _get_reals(cls, factors, use_cache=True):
        """Compute real root isolating intervals for a list of factors. """
        reals = []

        for currentfactor, k in factors:
            try:
                if not use_cache:
                    raise KeyError
                r = _reals_cache[currentfactor]
                reals.extend([(i, currentfactor, k) for i in r])
            except KeyError:
                real_part = cls._get_reals_sqf(currentfactor, use_cache)
                new = [(root, currentfactor, k) for root in real_part]
                reals.extend(new)

        reals = cls._reals_sorted(reals)
        return reals

    @classmethod
    def _get_complexes(cls, factors, use_cache=True):
        """Compute complex root isolating intervals for a list of factors. """
        complexes = []

        for currentfactor, k in ordered(factors):
            try:
                if not use_cache:
                    raise KeyError
                c = _complexes_cache[currentfactor]
                complexes.extend([(i, currentfactor, k) for i in c])
            except KeyError:
                complex_part = cls._get_complexes_sqf(currentfactor, use_cache)
                new = [(root, currentfactor, k) for root in complex_part]
                complexes.extend(new)

        complexes = cls._complexes_sorted(complexes)
        return complexes

    @classmethod
    def _reals_sorted(cls, reals):
        """Make real isolating intervals disjoint and sort roots. """
        cache = {}

        for i, (u, f, k) in enumerate(reals):
            for j, (v, g, m) in enumerate(reals[i + 1:]):
                u, v = u.refine_disjoint(v)
                reals[i + j + 1] = (v, g, m)

            reals[i] = (u, f, k)

        reals = sorted(reals, key=lambda r: r[0].a)

        for root, currentfactor, _ in reals:
            if currentfactor in cache:
                cache[currentfactor].append(root)
            else:
                cache[currentfactor] = [root]

        for currentfactor, root in cache.items():
            _reals_cache[currentfactor] = root

        return reals

    @classmethod
    def _refine_imaginary(cls, complexes):
        sifted = sift(complexes, lambda c: c[1])
        complexes = []
        for f in ordered(sifted):
            nimag = _imag_count_of_factor(f)
            if nimag == 0:
                # refine until xbounds are neg or pos
                for u, f, k in sifted[f]:
                    while u.ax*u.bx <= 0:
                        u = u._inner_refine()
                    complexes.append((u, f, k))
            else:
                # refine until all but nimag xbounds are neg or pos
                potential_imag = list(range(len(sifted[f])))
                while True:
                    assert len(potential_imag) > 1
                    for i in list(potential_imag):
                        u, f, k = sifted[f][i]
                        if u.ax*u.bx > 0:
                            potential_imag.remove(i)
                        elif u.ax != u.bx:
                            u = u._inner_refine()
                            sifted[f][i] = u, f, k
                    if len(potential_imag) == nimag:
                        break
                complexes.extend(sifted[f])
        return complexes

    @classmethod
    def _refine_complexes(cls, complexes):
        """return complexes such that no bounding rectangles of non-conjugate
        roots would intersect. In addition, assure that neither ay nor by is
        0 to guarantee that non-real roots are distinct from real roots in
        terms of the y-bounds.
        """
        # get the intervals pairwise-disjoint.
        # If rectangles were drawn around the coordinates of the bounding
        # rectangles, no rectangles would intersect after this procedure.
        for i, (u, f, k) in enumerate(complexes):
            for j, (v, g, m) in enumerate(complexes[i + 1:]):
                u, v = u.refine_disjoint(v)
                complexes[i + j + 1] = (v, g, m)

            complexes[i] = (u, f, k)

        # refine until the x-bounds are unambiguously positive or negative
        # for non-imaginary roots
        complexes = cls._refine_imaginary(complexes)

        # make sure that all y bounds are off the real axis
        # and on the same side of the axis
        for i, (u, f, k) in enumerate(complexes):
            while u.ay*u.by <= 0:
                u = u.refine()
            complexes[i] = u, f, k
        return complexes

    @classmethod
    def _complexes_sorted(cls, complexes):
        """Make complex isolating intervals disjoint and sort roots. """
        complexes = cls._refine_complexes(complexes)
        # XXX don't sort until you are sure that it is compatible
        # with the indexing method but assert that the desired state
        # is not broken
        C, F = 0, 1  # location of ComplexInterval and factor
        fs = {i[F] for i in complexes}
        for i in range(1, len(complexes)):
            if complexes[i][F] != complexes[i - 1][F]:
                # if this fails the factors of a root were not
                # contiguous because a discontinuity should only
                # happen once
                fs.remove(complexes[i - 1][F])
        for i, cmplx in enumerate(complexes):
            # negative im part (conj=True) comes before
            # positive im part (conj=False)
            assert cmplx[C].conj is (i % 2 == 0)

        # update cache
        cache = {}
        # -- collate
        for root, currentfactor, _ in complexes:
            cache.setdefault(currentfactor, []).append(root)
        # -- store
        for currentfactor, root in cache.items():
            _complexes_cache[currentfactor] = root

        return complexes

    @classmethod
    def _reals_index(cls, reals, index):
        """
        Map initial real root index to an index in a factor where
        the root belongs.
        """
        i = 0

        for j, (_, currentfactor, k) in enumerate(reals):
            if index < i + k:
                poly, index = currentfactor, 0

                for _, currentfactor, _ in reals[:j]:
                    if currentfactor == poly:
                        index += 1

                return poly, index
            else:
                i += k

    @classmethod
    def _complexes_index(cls, complexes, index):
        """
        Map initial complex root index to an index in a factor where
        the root belongs.
        """
        i = 0
        for j, (_, currentfactor, k) in enumerate(complexes):
            if index < i + k:
                poly, index = currentfactor, 0

                for _, currentfactor, _ in complexes[:j]:
                    if currentfactor == poly:
                        index += 1

                index += len(_reals_cache[poly])

                return poly, index
            else:
                i += k

    @classmethod
    def _count_roots(cls, roots):
        """Count the number of real or complex roots with multiplicities."""
        return sum(k for _, _, k in roots)

    @classmethod
    def _indexed_root(cls, poly, index, lazy=False):
        """Get a root of a composite polynomial by index. """
        factors = _pure_factors(poly)

        # If the given poly is already irreducible, then the index does not
        # need to be adjusted, and we can postpone the heavy lifting of
        # computing and refining isolating intervals until that is needed.
        # Note, however, that `_pure_factors()` extracts a negative leading
        # coeff if present, so `factors[0][0]` may differ from `poly`, and
        # is the "normalized" version of `poly` that we must return.
        if lazy and len(factors) == 1 and factors[0][1] == 1:
            return factors[0][0], index

        reals = cls._get_reals(factors)
        reals_count = cls._count_roots(reals)

        if index < reals_count:
            return cls._reals_index(reals, index)
        else:
            complexes = cls._get_complexes(factors)
            return cls._complexes_index(complexes, index - reals_count)

    def _ensure_reals_init(self):
        """Ensure that our poly has entries in the reals cache. """
        if self.poly not in _reals_cache:
            self._indexed_root(self.poly, self.index)

    def _ensure_complexes_init(self):
        """Ensure that our poly has entries in the complexes cache. """
        if self.poly not in _complexes_cache:
            self._indexed_root(self.poly, self.index)

    @classmethod
    def _real_roots(cls, poly):
        """Get real roots of a composite polynomial. """
        factors = _pure_factors(poly)

        reals = cls._get_reals(factors)
        reals_count = cls._count_roots(reals)

        roots = []

        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))

        return roots

    def _reset(self):
        """
        Reset all intervals
        """
        self._all_roots(self.poly, use_cache=False)

    @classmethod
    def _all_roots(cls, poly, use_cache=True):
        """Get real and complex roots of a composite polynomial. """
        factors = _pure_factors(poly)

        reals = cls._get_reals(factors, use_cache=use_cache)
        reals_count = cls._count_roots(reals)

        roots = []

        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))

        complexes = cls._get_complexes(factors, use_cache=use_cache)
        complexes_count = cls._count_roots(complexes)

        for index in range(0, complexes_count):
            roots.append(cls._complexes_index(complexes, index))

        return roots

    @classmethod
    @cacheit
    def _roots_trivial(cls, poly, radicals):
        """Compute roots in linear, quadratic and binomial cases. """
        if poly.degree() == 1:
            return roots_linear(poly)

        if not radicals:
            return None

        if poly.degree() == 2:
            return roots_quadratic(poly)
        elif poly.length() == 2 and poly.TC():
            return roots_binomial(poly)
        else:
            return None

    @classmethod
    def _preprocess_roots(cls, poly):
        """Take heroic measures to make ``poly`` compatible with ``CRootOf``."""
        dom = poly.get_domain()

        if not dom.is_Exact:
            poly = poly.to_exact()

        coeff, poly = preprocess_roots(poly)
        dom = poly.get_domain()

        if not dom.is_ZZ:
            raise NotImplementedError(
                "sorted roots not supported over %s" % dom)

        return coeff, poly

    @classmethod
    def _postprocess_root(cls, root, radicals):
        """Return the root if it is trivial or a ``CRootOf`` object. """
        poly, index = root
        roots = cls._roots_trivial(poly, radicals)

        if roots is not None:
            return roots[index]
        else:
            return cls._new(poly, index)

    @classmethod
    def _get_roots(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind. """
        if not poly.is_univariate:
            raise PolynomialError("only univariate polynomials are allowed")

        dom = poly.get_domain()

        # get rid of gen and it's free symbol
        d = Dummy()
        poly = poly.subs(poly.gen, d)
        x = symbols('x')
        # see what others are left and select x or a numbered x
        # that doesn't clash
        free_names = {str(i) for i in poly.free_symbols}
        for x in chain((symbols('x'),), numbered_symbols('x')):
            if x.name not in free_names:
                poly = poly.replace(d, x)
                break

        if dom.is_QQ or dom.is_ZZ:
            return cls._get_roots_qq(method, poly, radicals)
        elif dom.is_AlgebraicField or dom.is_ZZ_I or dom.is_QQ_I:
            return cls._get_roots_alg(method, poly, radicals)
        else:
            # XXX: not sure how to handle ZZ[x] which appears in some tests?
            # this makes the tests pass alright but has to be a better way?
            return cls._get_roots_qq(method, poly, radicals)


    @classmethod
    def _get_roots_qq(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind
         for polynomials with rational coefficients. """
        coeff, poly = cls._preprocess_roots(poly)
        roots = []

        for root in getattr(cls, method)(poly):
            roots.append(coeff*cls._postprocess_root(root, radicals))

        return roots

    @classmethod
    def _get_roots_alg(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind
         for polynomials with algebraic coefficients. It assumes
         the domain is already an algebraic field. First it
         finds the roots using _get_roots_qq, then uses the
         square-free factors to filter roots and get the correct
         multiplicity.
         """

        # Existing QQ code can find and sort the roots
        roots = cls._get_roots_qq(method, poly.lift(), radicals)

        subroots = {}
        for f, m in poly.sqf_list()[1]:
            if method == "_real_roots":
                roots_filt = f.which_real_roots(roots)
            elif method == "_all_roots":
                roots_filt = f.which_all_roots(roots)
            for r in roots_filt:
                subroots[r] = m

        roots_seen = set()
        roots_flat = []
        for r in roots:
            if r in subroots and r not in roots_seen:
                m = subroots[r]
                roots_flat.extend([r] * m)
                roots_seen.add(r)

        return roots_flat

    @classmethod
    def clear_cache(cls):
        """Reset cache for reals and complexes.

        The intervals used to approximate a root instance are updated
        as needed. When a request is made to see the intervals, the
        most current values are shown. `clear_cache` will reset all
        CRootOf instances back to their original state.

        See Also
        ========

        _reset
        """
        global _reals_cache, _complexes_cache
        _reals_cache = _pure_key_dict()
        _complexes_cache = _pure_key_dict()

    def _get_interval(self):
        """Internal function for retrieving isolation interval from cache. """
        self._ensure_reals_init()
        if self.is_real:
            return _reals_cache[self.poly][self.index]
        else:
            reals_count = len(_reals_cache[self.poly])
            self._ensure_complexes_init()
            return _complexes_cache[self.poly][self.index - reals_count]

    def _set_interval(self, interval):
        """Internal function for updating isolation interval in cache. """
        self._ensure_reals_init()
        if self.is_real:
            _reals_cache[self.poly][self.index] = interval
        else:
            reals_count = len(_reals_cache[self.poly])
            self._ensure_complexes_init()
            _complexes_cache[self.poly][self.index - reals_count] = interval

    def _eval_subs(self, old, new):
        # don't allow subs to change anything
        return self

    def _eval_conjugate(self):
        if self.is_real:
            return self
        expr, i = self.args
        return self.func(expr, i + (1 if self._get_interval().conj else -1))

    def eval_approx(self, n, return_mpmath=False):
        """Evaluate this complex root to the given precision.

        This uses secant method and root bounds are used to both
        generate an initial guess and to check that the root
        returned is valid. If ever the method converges outside the
        root bounds, the bounds will be made smaller and updated.
        """
        prec = dps_to_prec(n)
        with workprec(prec):
            g = self.poly.gen
            if not g.is_Symbol:
                d = Dummy('x')
                if self.is_imaginary:
                    d *= I
                func = lambdify(d, self.expr.subs(g, d))
            else:
                expr = self.expr
                if self.is_imaginary:
                    expr = self.expr.subs(g, I*g)
                func = lambdify(g, expr)

            interval = self._get_interval()
            while True:
                if self.is_real:
                    a = mpf(str(interval.a))
                    b = mpf(str(interval.b))
                    if a == b:
                        root = a
                        break
                    x0 = mpf(str(interval.center))
                    x1 = x0 + mpf(str(interval.dx))/4
                elif self.is_imaginary:
                    a = mpf(str(interval.ay))
                    b = mpf(str(interval.by))
                    if a == b:
                        root = mpc(mpf('0'), a)
                        break
                    x0 = mpf(str(interval.center[1]))
                    x1 = x0 + mpf(str(interval.dy))/4
                else:
                    ax = mpf(str(interval.ax))
                    bx = mpf(str(interval.bx))
                    ay = mpf(str(interval.ay))
                    by = mpf(str(interval.by))
                    if ax == bx and ay == by:
                        root = mpc(ax, ay)
                        break
                    x0 = mpc(*map(str, interval.center))
                    x1 = x0 + mpc(*map(str, (interval.dx, interval.dy)))/4
                try:
                    # without a tolerance, this will return when (to within
                    # the given precision) x_i == x_{i-1}
                    root = findroot(func, (x0, x1))
                    # If the (real or complex) root is not in the 'interval',
                    # then keep refining the interval. This happens if findroot
                    # accidentally finds a different root outside of this
                    # interval because our initial estimate 'x0' was not close
                    # enough. It is also possible that the secant method will
                    # get trapped by a max/min in the interval; the root
                    # verification by findroot will raise a ValueError in this
                    # case and the interval will then be tightened -- and
                    # eventually the root will be found.
                    #
                    # It is also possible that findroot will not have any
                    # successful iterations to process (in which case it
                    # will fail to initialize a variable that is tested
                    # after the iterations and raise an UnboundLocalError).
                    if self.is_real or self.is_imaginary:
                        if not bool(root.imag) == self.is_real and (
                                a <= root <= b):
                            if self.is_imaginary:
                                root = mpc(mpf('0'), root.real)
                            break
                    elif (ax <= root.real <= bx and ay <= root.imag <= by):
                        break
                except (UnboundLocalError, ValueError):
                    pass
                interval = interval.refine()

        # update the interval so we at least (for this precision or
        # less) don't have much work to do to recompute the root
        self._set_interval(interval)
        if return_mpmath:
            return root
        return (Float._new(root.real._mpf_, prec) +
            I*Float._new(root.imag._mpf_, prec))

    def _eval_evalf(self, prec, **kwargs):
        """Evaluate this complex root to the given precision."""
        # all kwargs are ignored
        return self.eval_rational(n=prec_to_dps(prec))._evalf(prec)

    def eval_rational(self, dx=None, dy=None, n=15):
        """
        Return a Rational approximation of ``self`` that has real
        and imaginary component approximations that are within ``dx``
        and ``dy`` of the true values, respectively. Alternatively,
        ``n`` digits of precision can be specified.

        The interval is refined with bisection and is sure to
        converge. The root bounds are updated when the refinement
        is complete so recalculation at the same or lesser precision
        will not have to repeat the refinement and should be much
        faster.

        The following example first obtains Rational approximation to
        1e-8 accuracy for all roots of the 4-th order Legendre
        polynomial. Since the roots are all less than 1, this will
        ensure the decimal representation of the approximation will be
        correct (including rounding) to 6 digits:

        >>> from sympy import legendre_poly, Symbol
        >>> x = Symbol("x")
        >>> p = legendre_poly(4, x, polys=True)
        >>> r = p.real_roots()[-1]
        >>> r.eval_rational(10**-8).n(6)
        0.861136

        It is not necessary to a two-step calculation, however: the
        decimal representation can be computed directly:

        >>> r.evalf(17)
        0.86113631159405258

        """
        dy = dy or dx
        if dx:
            rtol = None
            dx = dx if isinstance(dx, Rational) else Rational(str(dx))
            dy = dy if isinstance(dy, Rational) else Rational(str(dy))
        else:
            # 5 binary (or 2 decimal) digits are needed to ensure that
            # a given digit is correctly rounded
            # prec_to_dps(dps_to_prec(n) + 5) - n <= 2 (tested for
            # n in range(1000000)
            rtol = S(10)**-(n + 2)  # +2 for guard digits
        interval = self._get_interval()
        while True:
            if self.is_real:
                if rtol:
                    dx = abs(interval.center*rtol)
                interval = interval.refine_size(dx=dx)
                c = interval.center
                real = Rational(c)
                imag = S.Zero
                if not rtol or interval.dx < abs(c*rtol):
                    break
            elif self.is_imaginary:
                if rtol:
                    dy = abs(interval.center[1]*rtol)
                    dx = 1
                interval = interval.refine_size(dx=dx, dy=dy)
                c = interval.center[1]
                imag = Rational(c)
                real = S.Zero
                if not rtol or interval.dy < abs(c*rtol):
                    break
            else:
                if rtol:
                    dx = abs(interval.center[0]*rtol)
                    dy = abs(interval.center[1]*rtol)
                interval = interval.refine_size(dx, dy)
                c = interval.center
                real, imag = map(Rational, c)
                if not rtol or (
                        interval.dx < abs(c[0]*rtol) and
                        interval.dy < abs(c[1]*rtol)):
                    break

        # update the interval so we at least (for this precision or
        # less) don't have much work to do to recompute the root
        self._set_interval(interval)
        return real + I*imag


CRootOf = ComplexRootOf


@dispatch(ComplexRootOf, ComplexRootOf)
def _eval_is_eq(lhs, rhs): # noqa:F811
    # if we use is_eq to check here, we get infinite recursion
    return lhs == rhs


@dispatch(ComplexRootOf, Basic)  # type:ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    # CRootOf represents a Root, so if rhs is that root, it should set
    # the expression to zero *and* it should be in the interval of the
    # CRootOf instance. It must also be a number that agrees with the
    # is_real value of the CRootOf instance.
    if not rhs.is_number:
        return None
    if not rhs.is_finite:
        return False
    z = lhs.expr.subs(lhs.expr.free_symbols.pop(), rhs).is_zero
    if z is False:  # all roots will make z True but we don't know
        # whether this is the right root if z is True
        return False
    o = rhs.is_real, rhs.is_imaginary
    s = lhs.is_real, lhs.is_imaginary
    assert None not in s  # this is part of initial refinement
    if o != s and None not in o:
        return False
    re, im = rhs.as_real_imag()
    if lhs.is_real:
        if im:
            return False
        i = lhs._get_interval()
        a, b = [Rational(str(_)) for _ in (i.a, i.b)]
        return sympify(a <= rhs and rhs <= b)
    i = lhs._get_interval()
    r1, r2, i1, i2 = [Rational(str(j)) for j in (
        i.ax, i.bx, i.ay, i.by)]
    return is_le(r1, re) and is_le(re,r2) and is_le(i1,im) and is_le(im,i2)


@public
class RootSum(Expr):
    """Represents a sum of all roots of a univariate polynomial. """

    __slots__ = ('poly', 'fun', 'auto')

    def __new__(cls, expr, func=None, x=None, auto=True, quadratic=False):
        """Construct a new ``RootSum`` instance of roots of a polynomial."""
        coeff, poly = cls._transform(expr, x)

        if not poly.is_univariate:
            raise MultivariatePolynomialError(
                "only univariate polynomials are allowed")

        if func is None:
            func = Lambda(poly.gen, poly.gen)
        else:
            is_func = getattr(func, 'is_Function', False)

            if is_func and 1 in func.nargs:
                if not isinstance(func, Lambda):
                    func = Lambda(poly.gen, func(poly.gen))
            else:
                raise ValueError(
                    "expected a univariate function, got %s" % func)

        var, expr = func.variables[0], func.expr

        if coeff is not S.One:
            expr = expr.subs(var, coeff*var)

        deg = poly.degree()

        if not expr.has(var):
            return deg*expr

        if expr.is_Add:
            add_const, expr = expr.as_independent(var)
        else:
            add_const = S.Zero

        if expr.is_Mul:
            mul_const, expr = expr.as_independent(var)
        else:
            mul_const = S.One

        func = Lambda(var, expr)

        rational = cls._is_func_rational(poly, func)
        factors, terms = _pure_factors(poly), []

        for poly, k in factors:
            if poly.is_linear:
                term = func(roots_linear(poly)[0])
            elif quadratic and poly.is_quadratic:
                term = sum(map(func, roots_quadratic(poly)))
            else:
                if not rational or not auto:
                    term = cls._new(poly, func, auto)
                else:
                    term = cls._rational_case(poly, func)

            terms.append(k*term)

        return mul_const*Add(*terms) + deg*add_const

    @classmethod
    def _new(cls, poly, func, auto=True):
        """Construct new raw ``RootSum`` instance. """
        obj = Expr.__new__(cls)

        obj.poly = poly
        obj.fun = func
        obj.auto = auto

        return obj

    @classmethod
    def new(cls, poly, func, auto=True):
        """Construct new ``RootSum`` instance. """
        if not func.expr.has(*func.variables):
            return func.expr

        rational = cls._is_func_rational(poly, func)

        if not rational or not auto:
            return cls._new(poly, func, auto)
        else:
            return cls._rational_case(poly, func)

    @classmethod
    def _transform(cls, expr, x):
        """Transform an expression to a polynomial. """
        poly = PurePoly(expr, x, greedy=False)
        return preprocess_roots(poly)

    @classmethod
    def _is_func_rational(cls, poly, func):
        """Check if a lambda is a rational function. """
        var, expr = func.variables[0], func.expr
        return expr.is_rational_function(var)

    @classmethod
    def _rational_case(cls, poly, func):
        """Handle the rational function case. """
        roots = symbols('r:%d' % poly.degree())
        var, expr = func.variables[0], func.expr

        f = sum(expr.subs(var, r) for r in roots)
        p, q = together(f).as_numer_denom()

        domain = QQ[roots]

        p = p.expand()
        q = q.expand()

        try:
            p = Poly(p, domain=domain, expand=False)
        except GeneratorsNeeded:
            p, p_coeff = None, (p,)
        else:
            p_monom, p_coeff = zip(*p.terms())

        try:
            q = Poly(q, domain=domain, expand=False)
        except GeneratorsNeeded:
            q, q_coeff = None, (q,)
        else:
            q_monom, q_coeff = zip(*q.terms())

        coeffs, mapping = symmetrize(p_coeff + q_coeff, formal=True)
        formulas, values = viete(poly, roots), []

        for (sym, _), (_, val) in zip(mapping, formulas):
            values.append((sym, val))

        for i, (coeff, _) in enumerate(coeffs):
            coeffs[i] = coeff.subs(values)

        n = len(p_coeff)

        p_coeff = coeffs[:n]
        q_coeff = coeffs[n:]

        if p is not None:
            p = Poly(dict(zip(p_monom, p_coeff)), *p.gens).as_expr()
        else:
            (p,) = p_coeff

        if q is not None:
            q = Poly(dict(zip(q_monom, q_coeff)), *q.gens).as_expr()
        else:
            (q,) = q_coeff

        return factor(p/q)

    def _hashable_content(self):
        return (self.poly, self.fun)

    @property
    def expr(self):
        return self.poly.as_expr()

    @property
    def args(self):
        return (self.expr, self.fun, self.poly.gen)

    @property
    def free_symbols(self):
        return self.poly.free_symbols | self.fun.free_symbols

    @property
    def is_commutative(self):
        return True

    def doit(self, **hints):
        if not hints.get('roots', True):
            return self

        _roots = roots(self.poly, multiple=True)

        if len(_roots) < self.poly.degree():
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_evalf(self, prec):
        try:
            _roots = self.poly.nroots(n=prec_to_dps(prec))
        except (DomainError, PolynomialError):
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_derivative(self, x):
        var, expr = self.fun.args
        func = Lambda(var, expr.diff(x))
        return self.new(self.poly, func, self.auto)
