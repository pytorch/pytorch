"""Implementation of :class:`AlgebraicField` class. """


from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import CoercionFailed, DomainError, NotAlgebraic, IsomorphismFailed
from sympy.utilities import public

@public
class AlgebraicField(Field, CharacteristicZero, SimpleDomain):
    r"""Algebraic number field :ref:`QQ(a)`

    A :ref:`QQ(a)` domain represents an `algebraic number field`_
    `\mathbb{Q}(a)` as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    A :py:class:`~.Poly` created from an expression involving `algebraic
    numbers`_ will treat the algebraic numbers as generators if the generators
    argument is not specified.

    >>> from sympy import Poly, Symbol, sqrt
    >>> x = Symbol('x')
    >>> Poly(x**2 + sqrt(2))
    Poly(x**2 + (sqrt(2)), x, sqrt(2), domain='ZZ')

    That is a multivariate polynomial with ``sqrt(2)`` treated as one of the
    generators (variables). If the generators are explicitly specified then
    ``sqrt(2)`` will be considered to be a coefficient but by default the
    :ref:`EX` domain is used. To make a :py:class:`~.Poly` with a :ref:`QQ(a)`
    domain the argument ``extension=True`` can be given.

    >>> Poly(x**2 + sqrt(2), x)
    Poly(x**2 + sqrt(2), x, domain='EX')
    >>> Poly(x**2 + sqrt(2), x, extension=True)
    Poly(x**2 + sqrt(2), x, domain='QQ<sqrt(2)>')

    A generator of the algebraic field extension can also be specified
    explicitly which is particularly useful if the coefficients are all
    rational but an extension field is needed (e.g. to factor the
    polynomial).

    >>> Poly(x**2 + 1)
    Poly(x**2 + 1, x, domain='ZZ')
    >>> Poly(x**2 + 1, extension=sqrt(2))
    Poly(x**2 + 1, x, domain='QQ<sqrt(2)>')

    It is possible to factorise a polynomial over a :ref:`QQ(a)` domain using
    the ``extension`` argument to :py:func:`~.factor` or by specifying the domain
    explicitly.

    >>> from sympy import factor, QQ
    >>> factor(x**2 - 2)
    x**2 - 2
    >>> factor(x**2 - 2, extension=sqrt(2))
    (x - sqrt(2))*(x + sqrt(2))
    >>> factor(x**2 - 2, domain='QQ<sqrt(2)>')
    (x - sqrt(2))*(x + sqrt(2))
    >>> factor(x**2 - 2, domain=QQ.algebraic_field(sqrt(2)))
    (x - sqrt(2))*(x + sqrt(2))

    The ``extension=True`` argument can be used but will only create an
    extension that contains the coefficients which is usually not enough to
    factorise the polynomial.

    >>> p = x**3 + sqrt(2)*x**2 - 2*x - 2*sqrt(2)
    >>> factor(p)                         # treats sqrt(2) as a symbol
    (x + sqrt(2))*(x**2 - 2)
    >>> factor(p, extension=True)
    (x - sqrt(2))*(x + sqrt(2))**2
    >>> factor(x**2 - 2, extension=True)  # all rational coefficients
    x**2 - 2

    It is also possible to use :ref:`QQ(a)` with the :py:func:`~.cancel`
    and :py:func:`~.gcd` functions.

    >>> from sympy import cancel, gcd
    >>> cancel((x**2 - 2)/(x - sqrt(2)))
    (x**2 - 2)/(x - sqrt(2))
    >>> cancel((x**2 - 2)/(x - sqrt(2)), extension=sqrt(2))
    x + sqrt(2)
    >>> gcd(x**2 - 2, x - sqrt(2))
    1
    >>> gcd(x**2 - 2, x - sqrt(2), extension=sqrt(2))
    x - sqrt(2)

    When using the domain directly :ref:`QQ(a)` can be used as a constructor
    to create instances which then support the operations ``+,-,*,**,/``. The
    :py:meth:`~.Domain.algebraic_field` method is used to construct a
    particular :ref:`QQ(a)` domain. The :py:meth:`~.Domain.from_sympy` method
    can be used to create domain elements from normal SymPy expressions.

    >>> K = QQ.algebraic_field(sqrt(2))
    >>> K
    QQ<sqrt(2)>
    >>> xk = K.from_sympy(3 + 4*sqrt(2))
    >>> xk  # doctest: +SKIP
    ANP([4, 3], [1, 0, -2], QQ)

    Elements of :ref:`QQ(a)` are instances of :py:class:`~.ANP` which have
    limited printing support. The raw display shows the internal
    representation of the element as the list ``[4, 3]`` representing the
    coefficients of ``1`` and ``sqrt(2)`` for this element in the form
    ``a * sqrt(2) + b * 1`` where ``a`` and ``b`` are elements of :ref:`QQ`.
    The minimal polynomial for the generator ``(x**2 - 2)`` is also shown in
    the :ref:`dup-representation` as the list ``[1, 0, -2]``. We can use
    :py:meth:`~.Domain.to_sympy` to get a better printed form for the
    elements and to see the results of operations.

    >>> xk = K.from_sympy(3 + 4*sqrt(2))
    >>> yk = K.from_sympy(2 + 3*sqrt(2))
    >>> xk * yk  # doctest: +SKIP
    ANP([17, 30], [1, 0, -2], QQ)
    >>> K.to_sympy(xk * yk)
    17*sqrt(2) + 30
    >>> K.to_sympy(xk + yk)
    5 + 7*sqrt(2)
    >>> K.to_sympy(xk ** 2)
    24*sqrt(2) + 41
    >>> K.to_sympy(xk / yk)
    sqrt(2)/14 + 9/7

    Any expression representing an algebraic number can be used to generate
    a :ref:`QQ(a)` domain provided its `minimal polynomial`_ can be computed.
    The function :py:func:`~.minpoly` function is used for this.

    >>> from sympy import exp, I, pi, minpoly
    >>> g = exp(2*I*pi/3)
    >>> g
    exp(2*I*pi/3)
    >>> g.is_algebraic
    True
    >>> minpoly(g, x)
    x**2 + x + 1
    >>> factor(x**3 - 1, extension=g)
    (x - 1)*(x - exp(2*I*pi/3))*(x + 1 + exp(2*I*pi/3))

    It is also possible to make an algebraic field from multiple extension
    elements.

    >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
    >>> K
    QQ<sqrt(2) + sqrt(3)>
    >>> p = x**4 - 5*x**2 + 6
    >>> factor(p)
    (x**2 - 3)*(x**2 - 2)
    >>> factor(p, domain=K)
    (x - sqrt(2))*(x + sqrt(2))*(x - sqrt(3))*(x + sqrt(3))
    >>> factor(p, extension=[sqrt(2), sqrt(3)])
    (x - sqrt(2))*(x + sqrt(2))*(x - sqrt(3))*(x + sqrt(3))

    Multiple extension elements are always combined together to make a single
    `primitive element`_. In the case of ``[sqrt(2), sqrt(3)]`` the primitive
    element chosen is ``sqrt(2) + sqrt(3)`` which is why the domain displays
    as ``QQ<sqrt(2) + sqrt(3)>``. The minimal polynomial for the primitive
    element is computed using the :py:func:`~.primitive_element` function.

    >>> from sympy import primitive_element
    >>> primitive_element([sqrt(2), sqrt(3)], x)
    (x**4 - 10*x**2 + 1, [1, 1])
    >>> minpoly(sqrt(2) + sqrt(3), x)
    x**4 - 10*x**2 + 1

    The extension elements that generate the domain can be accessed from the
    domain using the :py:attr:`~.ext` and :py:attr:`~.orig_ext` attributes as
    instances of :py:class:`~.AlgebraicNumber`. The minimal polynomial for
    the primitive element as a :py:class:`~.DMP` instance is available as
    :py:attr:`~.mod`.

    >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
    >>> K
    QQ<sqrt(2) + sqrt(3)>
    >>> K.ext
    sqrt(2) + sqrt(3)
    >>> K.orig_ext
    (sqrt(2), sqrt(3))
    >>> K.mod  # doctest: +SKIP
    DMP_Python([1, 0, -10, 0, 1], QQ)

    The `discriminant`_ of the field can be obtained from the
    :py:meth:`~.discriminant` method, and an `integral basis`_ from the
    :py:meth:`~.integral_basis` method. The latter returns a list of
    :py:class:`~.ANP` instances by default, but can be made to return instances
    of :py:class:`~.Expr` or :py:class:`~.AlgebraicNumber` by passing a ``fmt``
    argument. The maximal order, or ring of integers, of the field can also be
    obtained from the :py:meth:`~.maximal_order` method, as a
    :py:class:`~sympy.polys.numberfields.modules.Submodule`.

    >>> zeta5 = exp(2*I*pi/5)
    >>> K = QQ.algebraic_field(zeta5)
    >>> K
    QQ<exp(2*I*pi/5)>
    >>> K.discriminant()
    125
    >>> K = QQ.algebraic_field(sqrt(5))
    >>> K
    QQ<sqrt(5)>
    >>> K.integral_basis(fmt='sympy')
    [1, 1/2 + sqrt(5)/2]
    >>> K.maximal_order()
    Submodule[[2, 0], [1, 1]]/2

    The factorization of a rational prime into prime ideals of the field is
    computed by the :py:meth:`~.primes_above` method, which returns a list
    of :py:class:`~sympy.polys.numberfields.primes.PrimeIdeal` instances.

    >>> zeta7 = exp(2*I*pi/7)
    >>> K = QQ.algebraic_field(zeta7)
    >>> K
    QQ<exp(2*I*pi/7)>
    >>> K.primes_above(11)
    [(11, _x**3 + 5*_x**2 + 4*_x - 1), (11, _x**3 - 4*_x**2 - 5*_x - 1)]

    The Galois group of the Galois closure of the field can be computed (when
    the minimal polynomial of the field is of sufficiently small degree).

    >>> K.galois_group(by_name=True)[0]
    S6TransitiveSubgroups.C6

    Notes
    =====

    It is not currently possible to generate an algebraic extension over any
    domain other than :ref:`QQ`. Ideally it would be possible to generate
    extensions like ``QQ(x)(sqrt(x**2 - 2))``. This is equivalent to the
    quotient ring ``QQ(x)[y]/(y**2 - x**2 + 2)`` and there are two
    implementations of this kind of quotient ring/extension in the
    :py:class:`~.QuotientRing` and :py:class:`~.MonogenicFiniteExtension`
    classes.  Each of those implementations needs some work to make them fully
    usable though.

    .. _algebraic number field: https://en.wikipedia.org/wiki/Algebraic_number_field
    .. _algebraic numbers: https://en.wikipedia.org/wiki/Algebraic_number
    .. _discriminant: https://en.wikipedia.org/wiki/Discriminant_of_an_algebraic_number_field
    .. _integral basis: https://en.wikipedia.org/wiki/Algebraic_number_field#Integral_basis
    .. _minimal polynomial: https://en.wikipedia.org/wiki/Minimal_polynomial_(field_theory)
    .. _primitive element: https://en.wikipedia.org/wiki/Primitive_element_theorem
    """

    dtype = ANP

    is_AlgebraicField = is_Algebraic = True
    is_Numerical = True

    has_assoc_Ring = False
    has_assoc_Field = True

    def __init__(self, dom, *ext, alias=None):
        r"""
        Parameters
        ==========

        dom : :py:class:`~.Domain`
            The base field over which this is an extension field.
            Currently only :ref:`QQ` is accepted.

        *ext : One or more :py:class:`~.Expr`
            Generators of the extension. These should be expressions that are
            algebraic over `\mathbb{Q}`.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            If provided, this will be used as the alias symbol for the
            primitive element of the :py:class:`~.AlgebraicField`.
            If ``None``, while ``ext`` consists of exactly one
            :py:class:`~.AlgebraicNumber`, its alias (if any) will be used.
        """
        if not dom.is_QQ:
            raise DomainError("ground domain must be a rational field")

        from sympy.polys.numberfields import to_number_field
        if len(ext) == 1 and isinstance(ext[0], tuple):
            orig_ext = ext[0][1:]
        else:
            orig_ext = ext

        if alias is None and len(ext) == 1:
            alias = getattr(ext[0], 'alias', None)

        self.orig_ext = orig_ext
        """
        Original elements given to generate the extension.

        >>> from sympy import QQ, sqrt
        >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
        >>> K.orig_ext
        (sqrt(2), sqrt(3))
        """

        self.ext = to_number_field(ext, alias=alias)
        """
        Primitive element used for the extension.

        >>> from sympy import QQ, sqrt
        >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
        >>> K.ext
        sqrt(2) + sqrt(3)
        """

        self.mod = self.ext.minpoly.rep
        """
        Minimal polynomial for the primitive element of the extension.

        >>> from sympy import QQ, sqrt
        >>> K = QQ.algebraic_field(sqrt(2))
        >>> K.mod
        DMP([1, 0, -2], QQ)
        """

        self.domain = self.dom = dom

        self.ngens = 1
        self.symbols = self.gens = (self.ext,)
        self.unit = self([dom(1), dom(0)])

        self.zero = self.dtype.zero(self.mod.to_list(), dom)
        self.one = self.dtype.one(self.mod.to_list(), dom)

        self._maximal_order = None
        self._discriminant = None
        self._nilradicals_mod_p = {}

    def new(self, element):
        return self.dtype(element, self.mod.to_list(), self.dom)

    def __str__(self):
        return str(self.dom) + '<' + str(self.ext) + '>'

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.dom, self.ext))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, AlgebraicField):
            return self.dtype == other.dtype and self.ext == other.ext
        else:
            return NotImplemented

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`. """
        return AlgebraicField(self.dom, *((self.ext,) + extension), alias=alias)

    def to_alg_num(self, a):
        """Convert ``a`` of ``dtype`` to an :py:class:`~.AlgebraicNumber`. """
        return self.ext.field_element(a)

    def to_sympy(self, a):
        """Convert ``a`` of ``dtype`` to a SymPy object. """
        # Precompute a converter to be reused:
        if not hasattr(self, '_converter'):
            self._converter = _make_converter(self)

        return self._converter(a)

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        try:
            return self([self.dom.from_sympy(a)])
        except CoercionFailed:
            pass

        from sympy.polys.numberfields import to_number_field

        try:
            return self(to_number_field(a, self.ext).native_coeffs())
        except (NotAlgebraic, IsomorphismFailed):
            raise CoercionFailed(
                "%s is not a valid algebraic number in %s" % (a, self))

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        raise DomainError('there is no ring associated with %s' % self)

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return self.dom.is_positive(a.LC())

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return self.dom.is_negative(a.LC())

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return self.dom.is_nonpositive(a.LC())

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return self.dom.is_nonnegative(a.LC())

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a

    def denom(self, a):
        """Returns denominator of ``a``. """
        return self.one

    def from_AlgebraicField(K1, a, K0):
        """Convert AlgebraicField element 'a' to another AlgebraicField """
        return K1.from_sympy(K0.to_sympy(a))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a GaussianInteger element 'a' to ``dtype``. """
        return K1.from_sympy(K0.to_sympy(a))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a GaussianRational element 'a' to ``dtype``. """
        return K1.from_sympy(K0.to_sympy(a))

    def _do_round_two(self):
        from sympy.polys.numberfields.basis import round_two
        ZK, dK = round_two(self, radicals=self._nilradicals_mod_p)
        self._maximal_order = ZK
        self._discriminant = dK

    def maximal_order(self):
        """
        Compute the maximal order, or ring of integers, of the field.

        Returns
        =======

        :py:class:`~sympy.polys.numberfields.modules.Submodule`.

        See Also
        ========

        integral_basis

        """
        if self._maximal_order is None:
            self._do_round_two()
        return self._maximal_order

    def integral_basis(self, fmt=None):
        r"""
        Get an integral basis for the field.

        Parameters
        ==========

        fmt : str, None, optional (default=None)
            If ``None``, return a list of :py:class:`~.ANP` instances.
            If ``"sympy"``, convert each element of the list to an
            :py:class:`~.Expr`, using ``self.to_sympy()``.
            If ``"alg"``, convert each element of the list to an
            :py:class:`~.AlgebraicNumber`, using ``self.to_alg_num()``.

        Examples
        ========

        >>> from sympy import QQ, AlgebraicNumber, sqrt
        >>> alpha = AlgebraicNumber(sqrt(5), alias='alpha')
        >>> k = QQ.algebraic_field(alpha)
        >>> B0 = k.integral_basis()
        >>> B1 = k.integral_basis(fmt='sympy')
        >>> B2 = k.integral_basis(fmt='alg')
        >>> print(B0[1])  # doctest: +SKIP
        ANP([mpq(1,2), mpq(1,2)], [mpq(1,1), mpq(0,1), mpq(-5,1)], QQ)
        >>> print(B1[1])
        1/2 + alpha/2
        >>> print(B2[1])
        alpha/2 + 1/2

        In the last two cases we get legible expressions, which print somewhat
        differently because of the different types involved:

        >>> print(type(B1[1]))
        <class 'sympy.core.add.Add'>
        >>> print(type(B2[1]))
        <class 'sympy.core.numbers.AlgebraicNumber'>

        See Also
        ========

        to_sympy
        to_alg_num
        maximal_order
        """
        ZK = self.maximal_order()
        M = ZK.QQ_matrix
        n = M.shape[1]
        B = [self.new(list(reversed(M[:, j].flat()))) for j in range(n)]
        if fmt == 'sympy':
            return [self.to_sympy(b) for b in B]
        elif fmt == 'alg':
            return [self.to_alg_num(b) for b in B]
        return B

    def discriminant(self):
        """Get the discriminant of the field."""
        if self._discriminant is None:
            self._do_round_two()
        return self._discriminant

    def primes_above(self, p):
        """Compute the prime ideals lying above a given rational prime *p*."""
        from sympy.polys.numberfields.primes import prime_decomp
        ZK = self.maximal_order()
        dK = self.discriminant()
        rad = self._nilradicals_mod_p.get(p)
        return prime_decomp(p, ZK=ZK, dK=dK, radical=rad)

    def galois_group(self, by_name=False, max_tries=30, randomize=False):
        """
        Compute the Galois group of the Galois closure of this field.

        Examples
        ========

        If the field is Galois, the order of the group will equal the degree
        of the field:

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> k = QQ.alg_field_from_poly(x**4 + 1)
        >>> G, _ = k.galois_group()
        >>> G.order()
        4

        If the field is not Galois, then its Galois closure is a proper
        extension, and the order of the Galois group will be greater than the
        degree of the field:

        >>> k = QQ.alg_field_from_poly(x**4 - 2)
        >>> G, _ = k.galois_group()
        >>> G.order()
        8

        See Also
        ========

        sympy.polys.numberfields.galoisgroups.galois_group

        """
        return self.ext.minpoly_of_element().galois_group(
            by_name=by_name, max_tries=max_tries, randomize=randomize)


def _make_converter(K):
    """Construct the converter to convert back to Expr"""
    # Precompute the effect of converting to SymPy and expanding expressions
    # like (sqrt(2) + sqrt(3))**2. Asking Expr to do the expansion on every
    # conversion from K to Expr is slow. Here we compute the expansions for
    # each power of the generator and collect together the resulting algebraic
    # terms and the rational coefficients into a matrix.

    gen = K.ext.as_expr()
    todom = K.dom.from_sympy

    # We'll let Expr compute the expansions. We won't make any presumptions
    # about what this results in except that it is QQ-linear in some terms
    # that we will call algebraics. The final result will be expressed in
    # terms of those.
    powers = [S.One, gen]
    for n in range(2, K.mod.degree()):
        powers.append((gen * powers[-1]).expand())

    # Collect the rational coefficients and algebraic Expr that can
    # map the ANP coefficients into an expanded SymPy expression
    terms = [dict(t.as_coeff_Mul()[::-1] for t in Add.make_args(p)) for p in powers]
    algebraics = set().union(*terms)
    matrix = [[todom(t.get(a, S.Zero)) for t in terms] for a in algebraics]

    # Create a function to do the conversion efficiently:

    def converter(a):
        """Convert a to Expr using converter"""
        ai = a.to_list()[::-1]
        tosympy = K.dom.to_sympy
        coeffs_dom = [sum(mij*aj for mij, aj in zip(mi, ai)) for mi in matrix]
        coeffs_sympy = [tosympy(c) for c in coeffs_dom]
        res = Add(*(Mul(c, a) for c, a in zip(coeffs_sympy, algebraics)))
        return res

    return converter
