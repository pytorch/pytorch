r"""Modules in number fields.

The classes defined here allow us to work with finitely generated, free
modules, whose generators are algebraic numbers.

There is an abstract base class called :py:class:`~.Module`, which has two
concrete subclasses, :py:class:`~.PowerBasis` and :py:class:`~.Submodule`.

Every module is defined by its basis, or set of generators:

* For a :py:class:`~.PowerBasis`, the generators are the first $n$ powers
  (starting with the zeroth) of an algebraic integer $\theta$ of degree $n$.
  The :py:class:`~.PowerBasis` is constructed by passing either the minimal
  polynomial of $\theta$, or an :py:class:`~.AlgebraicField` having $\theta$
  as its primitive element.

* For a :py:class:`~.Submodule`, the generators are a set of
  $\mathbb{Q}$-linear combinations of the generators of another module. That
  other module is then the "parent" of the :py:class:`~.Submodule`. The
  coefficients of the $\mathbb{Q}$-linear combinations may be given by an
  integer matrix, and a positive integer denominator. Each column of the matrix
  defines a generator.

>>> from sympy.polys import Poly, cyclotomic_poly, ZZ
>>> from sympy.abc import x
>>> from sympy.polys.matrices import DomainMatrix, DM
>>> from sympy.polys.numberfields.modules import PowerBasis
>>> T = Poly(cyclotomic_poly(5, x))
>>> A = PowerBasis(T)
>>> print(A)
PowerBasis(x**4 + x**3 + x**2 + x + 1)
>>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ), denom=3)
>>> print(B)
Submodule[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]/3
>>> print(B.parent)
PowerBasis(x**4 + x**3 + x**2 + x + 1)

Thus, every module is either a :py:class:`~.PowerBasis`,
or a :py:class:`~.Submodule`, some ancestor of which is a
:py:class:`~.PowerBasis`. (If ``S`` is a :py:class:`~.Submodule`, then its
ancestors are ``S.parent``, ``S.parent.parent``, and so on).

The :py:class:`~.ModuleElement` class represents a linear combination of the
generators of any module. Critically, the coefficients of this linear
combination are not restricted to be integers, but may be any rational
numbers. This is necessary so that any and all algebraic integers be
representable, starting from the power basis in a primitive element $\theta$
for the number field in question. For example, in a quadratic field
$\mathbb{Q}(\sqrt{d})$ where $d \equiv 1 \mod{4}$, a denominator of $2$ is
needed.

A :py:class:`~.ModuleElement` can be constructed from an integer column vector
and a denominator:

>>> U = Poly(x**2 - 5)
>>> M = PowerBasis(U)
>>> e = M(DM([[1], [1]], ZZ), denom=2)
>>> print(e)
[1, 1]/2
>>> print(e.module)
PowerBasis(x**2 - 5)

The :py:class:`~.PowerBasisElement` class is a subclass of
:py:class:`~.ModuleElement` that represents elements of a
:py:class:`~.PowerBasis`, and adds functionality pertinent to elements
represented directly over powers of the primitive element $\theta$.


Arithmetic with module elements
===============================

While a :py:class:`~.ModuleElement` represents a linear combination over the
generators of a particular module, recall that every module is either a
:py:class:`~.PowerBasis` or a descendant (along a chain of
:py:class:`~.Submodule` objects) thereof, so that in fact every
:py:class:`~.ModuleElement` represents an algebraic number in some field
$\mathbb{Q}(\theta)$, where $\theta$ is the defining element of some
:py:class:`~.PowerBasis`. It thus makes sense to talk about the number field
to which a given :py:class:`~.ModuleElement` belongs.

This means that any two :py:class:`~.ModuleElement` instances can be added,
subtracted, multiplied, or divided, provided they belong to the same number
field. Similarly, since $\mathbb{Q}$ is a subfield of every number field,
any :py:class:`~.ModuleElement` may be added, multiplied, etc. by any
rational number.

>>> from sympy import QQ
>>> from sympy.polys.numberfields.modules import to_col
>>> T = Poly(cyclotomic_poly(5))
>>> A = PowerBasis(T)
>>> C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
>>> e = A(to_col([0, 2, 0, 0]), denom=3)
>>> f = A(to_col([0, 0, 0, 7]), denom=5)
>>> g = C(to_col([1, 1, 1, 1]))
>>> e + f
[0, 10, 0, 21]/15
>>> e - f
[0, 10, 0, -21]/15
>>> e - g
[-9, -7, -9, -9]/3
>>> e + QQ(7, 10)
[21, 20, 0, 0]/30
>>> e * f
[-14, -14, -14, -14]/15
>>> e ** 2
[0, 0, 4, 0]/9
>>> f // g
[7, 7, 7, 7]/15
>>> f * QQ(2, 3)
[0, 0, 0, 14]/15

However, care must be taken with arithmetic operations on
:py:class:`~.ModuleElement`, because the module $C$ to which the result will
belong will be the nearest common ancestor (NCA) of the modules $A$, $B$ to
which the two operands belong, and $C$ may be different from either or both
of $A$ and $B$.

>>> A = PowerBasis(T)
>>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
>>> C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
>>> print((B(0) * C(0)).module == A)
True

Before the arithmetic operation is performed, copies of the two operands are
automatically converted into elements of the NCA (the operands themselves are
not modified). This upward conversion along an ancestor chain is easy: it just
requires the successive multiplication by the defining matrix of each
:py:class:`~.Submodule`.

Conversely, downward conversion, i.e. representing a given
:py:class:`~.ModuleElement` in a submodule, is also supported -- namely by
the :py:meth:`~sympy.polys.numberfields.modules.Submodule.represent` method
-- but is not guaranteed to succeed in general, since the given element may
not belong to the submodule. The main circumstance in which this issue tends
to arise is with multiplication, since modules, while closed under addition,
need not be closed under multiplication.


Multiplication
--------------

Generally speaking, a module need not be closed under multiplication, i.e. need
not form a ring. However, many of the modules we work with in the context of
number fields are in fact rings, and our classes do support multiplication.

Specifically, any :py:class:`~.Module` can attempt to compute its own
multiplication table, but this does not happen unless an attempt is made to
multiply two :py:class:`~.ModuleElement` instances belonging to it.

>>> A = PowerBasis(T)
>>> print(A._mult_tab is None)
True
>>> a = A(0)*A(1)
>>> print(A._mult_tab is None)
False

Every :py:class:`~.PowerBasis` is, by its nature, closed under multiplication,
so instances of :py:class:`~.PowerBasis` can always successfully compute their
multiplication table.

When a :py:class:`~.Submodule` attempts to compute its multiplication table,
it converts each of its own generators into elements of its parent module,
multiplies them there, in every possible pairing, and then tries to
represent the results in itself, i.e. as $\mathbb{Z}$-linear combinations
over its own generators. This will succeed if and only if the submodule is
in fact closed under multiplication.


Module Homomorphisms
====================

Many important number theoretic algorithms require the calculation of the
kernel of one or more module homomorphisms. Accordingly we have several
lightweight classes, :py:class:`~.ModuleHomomorphism`,
:py:class:`~.ModuleEndomorphism`, :py:class:`~.InnerEndomorphism`, and
:py:class:`~.EndomorphismRing`, which provide the minimal necessary machinery
to support this.

"""

from sympy.core.intfunc import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom


def to_col(coeffs):
    r"""Transform a list of integer coefficients into a column vector."""
    return DomainMatrix([[ZZ(c) for c in coeffs]], (1, len(coeffs)), ZZ).transpose()


class Module:
    """
    Generic finitely-generated module.

    This is an abstract base class, and should not be instantiated directly.
    The two concrete subclasses are :py:class:`~.PowerBasis` and
    :py:class:`~.Submodule`.

    Every :py:class:`~.Submodule` is derived from another module, referenced
    by its ``parent`` attribute. If ``S`` is a submodule, then we refer to
    ``S.parent``, ``S.parent.parent``, and so on, as the "ancestors" of
    ``S``. Thus, every :py:class:`~.Module` is either a
    :py:class:`~.PowerBasis` or a :py:class:`~.Submodule`, some ancestor of
    which is a :py:class:`~.PowerBasis`.
    """

    @property
    def n(self):
        """The number of generators of this module."""
        raise NotImplementedError

    def mult_tab(self):
        """
        Get the multiplication table for this module (if closed under mult).

        Explanation
        ===========

        Computes a dictionary ``M`` of dictionaries of lists, representing the
        upper triangular half of the multiplication table.

        In other words, if ``0 <= i <= j < self.n``, then ``M[i][j]`` is the
        list ``c`` of coefficients such that
        ``g[i] * g[j] == sum(c[k]*g[k], k in range(self.n))``,
        where ``g`` is the list of generators of this module.

        If ``j < i`` then ``M[i][j]`` is undefined.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> print(A.mult_tab())  # doctest: +SKIP
        {0: {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0],     3: [0, 0, 0, 1]},
                          1: {1: [0, 0, 1, 0], 2: [0, 0, 0, 1],     3: [-1, -1, -1, -1]},
                                           2: {2: [-1, -1, -1, -1], 3: [1, 0, 0, 0]},
                                                                3: {3: [0, 1, 0, 0]}}

        Returns
        =======

        dict of dict of lists

        Raises
        ======

        ClosureFailure
            If the module is not closed under multiplication.

        """
        raise NotImplementedError

    @property
    def parent(self):
        """
        The parent module, if any, for this module.

        Explanation
        ===========

        For a :py:class:`~.Submodule` this is its ``parent`` attribute; for a
        :py:class:`~.PowerBasis` this is ``None``.

        Returns
        =======

        :py:class:`~.Module`, ``None``

        See Also
        ========

        Module

        """
        return None

    def represent(self, elt):
        r"""
        Represent a module element as an integer-linear combination over the
        generators of this module.

        Explanation
        ===========

        In our system, to "represent" always means to write a
        :py:class:`~.ModuleElement` as a :ref:`ZZ`-linear combination over the
        generators of the present :py:class:`~.Module`. Furthermore, the
        incoming :py:class:`~.ModuleElement` must belong to an ancestor of
        the present :py:class:`~.Module` (or to the present
        :py:class:`~.Module` itself).

        The most common application is to represent a
        :py:class:`~.ModuleElement` in a :py:class:`~.Submodule`. For example,
        this is involved in computing multiplication tables.

        On the other hand, representing in a :py:class:`~.PowerBasis` is an
        odd case, and one which tends not to arise in practice, except for
        example when using a :py:class:`~.ModuleEndomorphism` on a
        :py:class:`~.PowerBasis`.

        In such a case, (1) the incoming :py:class:`~.ModuleElement` must
        belong to the :py:class:`~.PowerBasis` itself (since the latter has no
        proper ancestors) and (2) it is "representable" iff it belongs to
        $\mathbb{Z}[\theta]$ (although generally a
        :py:class:`~.PowerBasisElement` may represent any element of
        $\mathbb{Q}(\theta)$, i.e. any algebraic number).

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis, to_col
        >>> from sympy.abc import zeta
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> a = A(to_col([2, 4, 6, 8]))

        The :py:class:`~.ModuleElement` ``a`` has all even coefficients.
        If we represent ``a`` in the submodule ``B = 2*A``, the coefficients in
        the column vector will be halved:

        >>> B = A.submodule_from_gens([2*A(i) for i in range(4)])
        >>> b = B.represent(a)
        >>> print(b.transpose())  # doctest: +SKIP
        DomainMatrix([[1, 2, 3, 4]], (1, 4), ZZ)

        However, the element of ``B`` so defined still represents the same
        algebraic number:

        >>> print(a.poly(zeta).as_expr())
        8*zeta**3 + 6*zeta**2 + 4*zeta + 2
        >>> print(B(b).over_power_basis().poly(zeta).as_expr())
        8*zeta**3 + 6*zeta**2 + 4*zeta + 2

        Parameters
        ==========

        elt : :py:class:`~.ModuleElement`
            The module element to be represented. Must belong to some ancestor
            module of this module (including this module itself).

        Returns
        =======

        :py:class:`~.DomainMatrix` over :ref:`ZZ`
            This will be a column vector, representing the coefficients of a
            linear combination of this module's generators, which equals the
            given element.

        Raises
        ======

        ClosureFailure
            If the given element cannot be represented as a :ref:`ZZ`-linear
            combination over this module.

        See Also
        ========

        .Submodule.represent
        .PowerBasis.represent

        """
        raise NotImplementedError

    def ancestors(self, include_self=False):
        """
        Return the list of ancestor modules of this module, from the
        foundational :py:class:`~.PowerBasis` downward, optionally including
        ``self``.

        See Also
        ========

        Module

        """
        c = self.parent
        a = [] if c is None else c.ancestors(include_self=True)
        if include_self:
            a.append(self)
        return a

    def power_basis_ancestor(self):
        """
        Return the :py:class:`~.PowerBasis` that is an ancestor of this module.

        See Also
        ========

        Module

        """
        if isinstance(self, PowerBasis):
            return self
        c = self.parent
        if c is not None:
            return c.power_basis_ancestor()
        return None

    def nearest_common_ancestor(self, other):
        """
        Locate the nearest common ancestor of this module and another.

        Returns
        =======

        :py:class:`~.Module`, ``None``

        See Also
        ========

        Module

        """
        sA = self.ancestors(include_self=True)
        oA = other.ancestors(include_self=True)
        nca = None
        for sa, oa in zip(sA, oA):
            if sa == oa:
                nca = sa
            else:
                break
        return nca

    @property
    def number_field(self):
        r"""
        Return the associated :py:class:`~.AlgebraicField`, if any.

        Explanation
        ===========

        A :py:class:`~.PowerBasis` can be constructed on a :py:class:`~.Poly`
        $f$ or on an :py:class:`~.AlgebraicField` $K$. In the latter case, the
        :py:class:`~.PowerBasis` and all its descendant modules will return $K$
        as their ``.number_field`` property, while in the former case they will
        all return ``None``.

        Returns
        =======

        :py:class:`~.AlgebraicField`, ``None``

        """
        return self.power_basis_ancestor().number_field

    def is_compat_col(self, col):
        """Say whether *col* is a suitable column vector for this module."""
        return isinstance(col, DomainMatrix) and col.shape == (self.n, 1) and col.domain.is_ZZ

    def __call__(self, spec, denom=1):
        r"""
        Generate a :py:class:`~.ModuleElement` belonging to this module.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis, to_col
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> e = A(to_col([1, 2, 3, 4]), denom=3)
        >>> print(e)  # doctest: +SKIP
        [1, 2, 3, 4]/3
        >>> f = A(2)
        >>> print(f)  # doctest: +SKIP
        [0, 0, 1, 0]

        Parameters
        ==========

        spec : :py:class:`~.DomainMatrix`, int
            Specifies the numerators of the coefficients of the
            :py:class:`~.ModuleElement`. Can be either a column vector over
            :ref:`ZZ`, whose length must equal the number $n$ of generators of
            this module, or else an integer ``j``, $0 \leq j < n$, which is a
            shorthand for column $j$ of $I_n$, the $n \times n$ identity
            matrix.
        denom : int, optional (default=1)
            Denominator for the coefficients of the
            :py:class:`~.ModuleElement`.

        Returns
        =======

        :py:class:`~.ModuleElement`
            The coefficients are the entries of the *spec* vector, divided by
            *denom*.

        """
        if isinstance(spec, int) and 0 <= spec < self.n:
            spec = DomainMatrix.eye(self.n, ZZ)[:, spec].to_dense()
        if not self.is_compat_col(spec):
            raise ValueError('Compatible column vector required.')
        return make_mod_elt(self, spec, denom=denom)

    def starts_with_unity(self):
        """Say whether the module's first generator equals unity."""
        raise NotImplementedError

    def basis_elements(self):
        """
        Get list of :py:class:`~.ModuleElement` being the generators of this
        module.
        """
        return [self(j) for j in range(self.n)]

    def zero(self):
        """Return a :py:class:`~.ModuleElement` representing zero."""
        return self(0) * 0

    def one(self):
        """
        Return a :py:class:`~.ModuleElement` representing unity,
        and belonging to the first ancestor of this module (including
        itself) that starts with unity.
        """
        return self.element_from_rational(1)

    def element_from_rational(self, a):
        """
        Return a :py:class:`~.ModuleElement` representing a rational number.

        Explanation
        ===========

        The returned :py:class:`~.ModuleElement` will belong to the first
        module on this module's ancestor chain (including this module
        itself) that starts with unity.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, QQ
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> a = A.element_from_rational(QQ(2, 3))
        >>> print(a)  # doctest: +SKIP
        [2, 0, 0, 0]/3

        Parameters
        ==========

        a : int, :ref:`ZZ`, :ref:`QQ`

        Returns
        =======

        :py:class:`~.ModuleElement`

        """
        raise NotImplementedError

    def submodule_from_gens(self, gens, hnf=True, hnf_modulus=None):
        """
        Form the submodule generated by a list of :py:class:`~.ModuleElement`
        belonging to this module.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> gens = [A(0), 2*A(1), 3*A(2), 4*A(3)//5]
        >>> B = A.submodule_from_gens(gens)
        >>> print(B)  # doctest: +SKIP
        Submodule[[5, 0, 0, 0], [0, 10, 0, 0], [0, 0, 15, 0], [0, 0, 0, 4]]/5

        Parameters
        ==========

        gens : list of :py:class:`~.ModuleElement` belonging to this module.
        hnf : boolean, optional (default=True)
            If True, we will reduce the matrix into Hermite Normal Form before
            forming the :py:class:`~.Submodule`.
        hnf_modulus : int, None, optional (default=None)
            Modulus for use in the HNF reduction algorithm. See
            :py:func:`~sympy.polys.matrices.normalforms.hermite_normal_form`.

        Returns
        =======

        :py:class:`~.Submodule`

        See Also
        ========

        submodule_from_matrix

        """
        if not all(g.module == self for g in gens):
            raise ValueError('Generators must belong to this module.')
        n = len(gens)
        if n == 0:
            raise ValueError('Need at least one generator.')
        m = gens[0].n
        d = gens[0].denom if n == 1 else ilcm(*[g.denom for g in gens])
        B = DomainMatrix.zeros((m, 0), ZZ).hstack(*[(d // g.denom) * g.col for g in gens])
        if hnf:
            B = hermite_normal_form(B, D=hnf_modulus)
        return self.submodule_from_matrix(B, denom=d)

    def submodule_from_matrix(self, B, denom=1):
        """
        Form the submodule generated by the elements of this module indicated
        by the columns of a matrix, with an optional denominator.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
        >>> from sympy.polys.matrices import DM
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_matrix(DM([
        ...     [0, 10, 0, 0],
        ...     [0,  0, 7, 0],
        ... ], ZZ).transpose(), denom=15)
        >>> print(B)  # doctest: +SKIP
        Submodule[[0, 10, 0, 0], [0, 0, 7, 0]]/15

        Parameters
        ==========

        B : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            Each column gives the numerators of the coefficients of one
            generator of the submodule. Thus, the number of rows of *B* must
            equal the number of generators of the present module.
        denom : int, optional (default=1)
            Common denominator for all generators of the submodule.

        Returns
        =======

        :py:class:`~.Submodule`

        Raises
        ======

        ValueError
            If the given matrix *B* is not over :ref:`ZZ` or its number of rows
            does not equal the number of generators of the present module.

        See Also
        ========

        submodule_from_gens

        """
        m, n = B.shape
        if not B.domain.is_ZZ:
            raise ValueError('Matrix must be over ZZ.')
        if not m == self.n:
            raise ValueError('Matrix row count must match base module.')
        return Submodule(self, B, denom=denom)

    def whole_submodule(self):
        """
        Return a submodule equal to this entire module.

        Explanation
        ===========

        This is useful when you have a :py:class:`~.PowerBasis` and want to
        turn it into a :py:class:`~.Submodule` (in order to use methods
        belonging to the latter).

        """
        B = DomainMatrix.eye(self.n, ZZ)
        return self.submodule_from_matrix(B)

    def endomorphism_ring(self):
        """Form the :py:class:`~.EndomorphismRing` for this module."""
        return EndomorphismRing(self)


class PowerBasis(Module):
    """The module generated by the powers of an algebraic integer."""

    def __init__(self, T):
        """
        Parameters
        ==========

        T : :py:class:`~.Poly`, :py:class:`~.AlgebraicField`
            Either (1) the monic, irreducible, univariate polynomial over
            :ref:`ZZ`, a root of which is the generator of the power basis,
            or (2) an :py:class:`~.AlgebraicField` whose primitive element
            is the generator of the power basis.

        """
        K = None
        if isinstance(T, AlgebraicField):
            K, T = T, T.ext.minpoly_of_element()
        # Sometimes incoming Polys are formally over QQ, although all their
        # coeffs are integral. We want them to be formally over ZZ.
        T = T.set_domain(ZZ)
        self.K = K
        self.T = T
        self._n = T.degree()
        self._mult_tab = None

    @property
    def number_field(self):
        return self.K

    def __repr__(self):
        return f'PowerBasis({self.T.as_expr()})'

    def __eq__(self, other):
        if isinstance(other, PowerBasis):
            return self.T == other.T
        return NotImplemented

    @property
    def n(self):
        return self._n

    def mult_tab(self):
        if self._mult_tab is None:
            self.compute_mult_tab()
        return self._mult_tab

    def compute_mult_tab(self):
        theta_pow = AlgIntPowers(self.T)
        M = {}
        n = self.n
        for u in range(n):
            M[u] = {}
            for v in range(u, n):
                M[u][v] = theta_pow[u + v]
        self._mult_tab = M

    def represent(self, elt):
        r"""
        Represent a module element as an integer-linear combination over the
        generators of this module.

        See Also
        ========

        .Module.represent
        .Submodule.represent

        """
        if elt.module == self and elt.denom == 1:
            return elt.column()
        else:
            raise ClosureFailure('Element not representable in ZZ[theta].')

    def starts_with_unity(self):
        return True

    def element_from_rational(self, a):
        return self(0) * a

    def element_from_poly(self, f):
        """
        Produce an element of this module, representing *f* after reduction mod
        our defining minimal polynomial.

        Parameters
        ==========

        f : :py:class:`~.Poly` over :ref:`ZZ` in same var as our defining poly.

        Returns
        =======

        :py:class:`~.PowerBasisElement`

        """
        n, k = self.n, f.degree()
        if k >= n:
            f = f % self.T
        if f == 0:
            return self.zero()
        d, c = dup_clear_denoms(f.rep.to_list(), QQ, convert=True)
        c = list(reversed(c))
        ell = len(c)
        z = [ZZ(0)] * (n - ell)
        col = to_col(c + z)
        return self(col, denom=d)

    def _element_from_rep_and_mod(self, rep, mod):
        """
        Produce a PowerBasisElement representing a given algebraic number.

        Parameters
        ==========

        rep : list of coeffs
            Represents the number as polynomial in the primitive element of the
            field.

        mod : list of coeffs
            Represents the minimal polynomial of the primitive element of the
            field.

        Returns
        =======

        :py:class:`~.PowerBasisElement`

        """
        if mod != self.T.rep.to_list():
            raise UnificationFailed('Element does not appear to be in the same field.')
        return self.element_from_poly(Poly(rep, self.T.gen))

    def element_from_ANP(self, a):
        """Convert an ANP into a PowerBasisElement. """
        return self._element_from_rep_and_mod(a.to_list(), a.mod_to_list())

    def element_from_alg_num(self, a):
        """Convert an AlgebraicNumber into a PowerBasisElement. """
        return self._element_from_rep_and_mod(a.rep.to_list(), a.minpoly.rep.to_list())


class Submodule(Module, IntegerPowerable):
    """A submodule of another module."""

    def __init__(self, parent, matrix, denom=1, mult_tab=None):
        """
        Parameters
        ==========

        parent : :py:class:`~.Module`
            The module from which this one is derived.
        matrix : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            The matrix whose columns define this submodule's generators as
            linear combinations over the parent's generators.
        denom : int, optional (default=1)
            Denominator for the coefficients given by the matrix.
        mult_tab : dict, ``None``, optional
            If already known, the multiplication table for this module may be
            supplied.

        """
        self._parent = parent
        self._matrix = matrix
        self._denom = denom
        self._mult_tab = mult_tab
        self._n = matrix.shape[1]
        self._QQ_matrix = None
        self._starts_with_unity = None
        self._is_sq_maxrank_HNF = None

    def __repr__(self):
        r = 'Submodule' + repr(self.matrix.transpose().to_Matrix().tolist())
        if self.denom > 1:
            r += f'/{self.denom}'
        return r

    def reduced(self):
        """
        Produce a reduced version of this submodule.

        Explanation
        ===========

        In the reduced version, it is guaranteed that 1 is the only positive
        integer dividing both the submodule's denominator, and every entry in
        the submodule's matrix.

        Returns
        =======

        :py:class:`~.Submodule`

        """
        if self.denom == 1:
            return self
        g = igcd(self.denom, *self.coeffs)
        if g == 1:
            return self
        return type(self)(self.parent, (self.matrix / g).convert_to(ZZ), denom=self.denom // g, mult_tab=self._mult_tab)

    def discard_before(self, r):
        """
        Produce a new module by discarding all generators before a given
        index *r*.
        """
        W = self.matrix[:, r:]
        s = self.n - r
        M = None
        mt = self._mult_tab
        if mt is not None:
            M = {}
            for u in range(s):
                M[u] = {}
                for v in range(u, s):
                    M[u][v] = mt[r + u][r + v][r:]
        return Submodule(self.parent, W, denom=self.denom, mult_tab=M)

    @property
    def n(self):
        return self._n

    def mult_tab(self):
        if self._mult_tab is None:
            self.compute_mult_tab()
        return self._mult_tab

    def compute_mult_tab(self):
        gens = self.basis_element_pullbacks()
        M = {}
        n = self.n
        for u in range(n):
            M[u] = {}
            for v in range(u, n):
                M[u][v] = self.represent(gens[u] * gens[v]).flat()
        self._mult_tab = M

    @property
    def parent(self):
        return self._parent

    @property
    def matrix(self):
        return self._matrix

    @property
    def coeffs(self):
        return self.matrix.flat()

    @property
    def denom(self):
        return self._denom

    @property
    def QQ_matrix(self):
        """
        :py:class:`~.DomainMatrix` over :ref:`QQ`, equal to
        ``self.matrix / self.denom``, and guaranteed to be dense.

        Explanation
        ===========

        Depending on how it is formed, a :py:class:`~.DomainMatrix` may have
        an internal representation that is sparse or dense. We guarantee a
        dense representation here, so that tests for equivalence of submodules
        always come out as expected.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
        >>> from sympy.abc import x
        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5, x))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_matrix(3*DomainMatrix.eye(4, ZZ), denom=6)
        >>> C = A.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=2)
        >>> print(B.QQ_matrix == C.QQ_matrix)
        True

        Returns
        =======

        :py:class:`~.DomainMatrix` over :ref:`QQ`

        """
        if self._QQ_matrix is None:
            self._QQ_matrix = (self.matrix / self.denom).to_dense()
        return self._QQ_matrix

    def starts_with_unity(self):
        if self._starts_with_unity is None:
            self._starts_with_unity = self(0).equiv(1)
        return self._starts_with_unity

    def is_sq_maxrank_HNF(self):
        if self._is_sq_maxrank_HNF is None:
            self._is_sq_maxrank_HNF = is_sq_maxrank_HNF(self._matrix)
        return self._is_sq_maxrank_HNF

    def is_power_basis_submodule(self):
        return isinstance(self.parent, PowerBasis)

    def element_from_rational(self, a):
        if self.starts_with_unity():
            return self(0) * a
        else:
            return self.parent.element_from_rational(a)

    def basis_element_pullbacks(self):
        """
        Return list of this submodule's basis elements as elements of the
        submodule's parent module.
        """
        return [e.to_parent() for e in self.basis_elements()]

    def represent(self, elt):
        """
        Represent a module element as an integer-linear combination over the
        generators of this module.

        See Also
        ========

        .Module.represent
        .PowerBasis.represent

        """
        if elt.module == self:
            return elt.column()
        elif elt.module == self.parent:
            try:
                # The given element should be a ZZ-linear combination over our
                # basis vectors; however, due to the presence of denominators,
                # we need to solve over QQ.
                A = self.QQ_matrix
                b = elt.QQ_col
                x = A._solve(b)[0].transpose()
                x = x.convert_to(ZZ)
            except DMBadInputError:
                raise ClosureFailure('Element outside QQ-span of this basis.')
            except CoercionFailed:
                raise ClosureFailure('Element in QQ-span but not ZZ-span of this basis.')
            return x
        elif isinstance(self.parent, Submodule):
            coeffs_in_parent = self.parent.represent(elt)
            parent_element = self.parent(coeffs_in_parent)
            return self.represent(parent_element)
        else:
            raise ClosureFailure('Element outside ancestor chain of this module.')

    def is_compat_submodule(self, other):
        return isinstance(other, Submodule) and other.parent == self.parent

    def __eq__(self, other):
        if self.is_compat_submodule(other):
            return other.QQ_matrix == self.QQ_matrix
        return NotImplemented

    def add(self, other, hnf=True, hnf_modulus=None):
        """
        Add this :py:class:`~.Submodule` to another.

        Explanation
        ===========

        This represents the module generated by the union of the two modules'
        sets of generators.

        Parameters
        ==========

        other : :py:class:`~.Submodule`
        hnf : boolean, optional (default=True)
            If ``True``, reduce the matrix of the combined module to its
            Hermite Normal Form.
        hnf_modulus : :ref:`ZZ`, None, optional
            If a positive integer is provided, use this as modulus in the
            HNF reduction. See
            :py:func:`~sympy.polys.matrices.normalforms.hermite_normal_form`.

        Returns
        =======

        :py:class:`~.Submodule`

        """
        d, e = self.denom, other.denom
        m = ilcm(d, e)
        a, b = m // d, m // e
        B = (a * self.matrix).hstack(b * other.matrix)
        if hnf:
            B = hermite_normal_form(B, D=hnf_modulus)
        return self.parent.submodule_from_matrix(B, denom=m)

    def __add__(self, other):
        if self.is_compat_submodule(other):
            return self.add(other)
        return NotImplemented

    __radd__ = __add__

    def mul(self, other, hnf=True, hnf_modulus=None):
        """
        Multiply this :py:class:`~.Submodule` by a rational number, a
        :py:class:`~.ModuleElement`, or another :py:class:`~.Submodule`.

        Explanation
        ===========

        To multiply by a rational number or :py:class:`~.ModuleElement` means
        to form the submodule whose generators are the products of this
        quantity with all the generators of the present submodule.

        To multiply by another :py:class:`~.Submodule` means to form the
        submodule whose generators are all the products of one generator from
        the one submodule, and one generator from the other.

        Parameters
        ==========

        other : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.ModuleElement`, :py:class:`~.Submodule`
        hnf : boolean, optional (default=True)
            If ``True``, reduce the matrix of the product module to its
            Hermite Normal Form.
        hnf_modulus : :ref:`ZZ`, None, optional
            If a positive integer is provided, use this as modulus in the
            HNF reduction. See
            :py:func:`~sympy.polys.matrices.normalforms.hermite_normal_form`.

        Returns
        =======

        :py:class:`~.Submodule`

        """
        if is_rat(other):
            a, b = get_num_denom(other)
            if a == b == 1:
                return self
            else:
                return Submodule(self.parent,
                             self.matrix * a, denom=self.denom * b,
                             mult_tab=None).reduced()
        elif isinstance(other, ModuleElement) and other.module == self.parent:
            # The submodule is multiplied by an element of the parent module.
            # We presume this means we want a new submodule of the parent module.
            gens = [other * e for e in self.basis_element_pullbacks()]
            return self.parent.submodule_from_gens(gens, hnf=hnf, hnf_modulus=hnf_modulus)
        elif self.is_compat_submodule(other):
            # This case usually means you're multiplying ideals, and want another
            # ideal, i.e. another submodule of the same parent module.
            alphas, betas = self.basis_element_pullbacks(), other.basis_element_pullbacks()
            gens = [a * b for a in alphas for b in betas]
            return self.parent.submodule_from_gens(gens, hnf=hnf, hnf_modulus=hnf_modulus)
        return NotImplemented

    def __mul__(self, other):
        return self.mul(other)

    __rmul__ = __mul__

    def _first_power(self):
        return self

    def reduce_element(self, elt):
        r"""
        If this submodule $B$ has defining matrix $W$ in square, maximal-rank
        Hermite normal form, then, given an element $x$ of the parent module
        $A$, we produce an element $y \in A$ such that $x - y \in B$, and the
        $i$th coordinate of $y$ satisfies $0 \leq y_i < w_{i,i}$. This
        representative $y$ is unique, in the sense that every element of
        the coset $x + B$ reduces to it under this procedure.

        Explanation
        ===========

        In the special case where $A$ is a power basis for a number field $K$,
        and $B$ is a submodule representing an ideal $I$, this operation
        represents one of a few important ways of reducing an element of $K$
        modulo $I$ to obtain a "small" representative. See [Cohen00]_ Section
        1.4.3.

        Examples
        ========

        >>> from sympy import QQ, Poly, symbols
        >>> t = symbols('t')
        >>> k = QQ.alg_field_from_poly(Poly(t**3 + t**2 - 2*t + 8))
        >>> Zk = k.maximal_order()
        >>> A = Zk.parent
        >>> B = (A(2) - 3*A(0))*Zk
        >>> B.reduce_element(A(2))
        [3, 0, 0]

        Parameters
        ==========

        elt : :py:class:`~.ModuleElement`
            An element of this submodule's parent module.

        Returns
        =======

        elt : :py:class:`~.ModuleElement`
            An element of this submodule's parent module.

        Raises
        ======

        NotImplementedError
            If the given :py:class:`~.ModuleElement` does not belong to this
            submodule's parent module.
        StructureError
            If this submodule's defining matrix is not in square, maximal-rank
            Hermite normal form.

        References
        ==========

        .. [Cohen00] Cohen, H. *Advanced Topics in Computational Number
           Theory.*

        """
        if not elt.module == self.parent:
            raise NotImplementedError
        if not self.is_sq_maxrank_HNF():
            msg = "Reduction not implemented unless matrix square max-rank HNF"
            raise StructureError(msg)
        B = self.basis_element_pullbacks()
        a = elt
        for i in range(self.n - 1, -1, -1):
            b = B[i]
            q = a.coeffs[i]*b.denom // (b.coeffs[i]*a.denom)
            a -= q*b
        return a


def is_sq_maxrank_HNF(dm):
    r"""
    Say whether a :py:class:`~.DomainMatrix` is in that special case of Hermite
    Normal Form, in which the matrix is also square and of maximal rank.

    Explanation
    ===========

    We commonly work with :py:class:`~.Submodule` instances whose matrix is in
    this form, and it can be useful to be able to check that this condition is
    satisfied.

    For example this is the case with the :py:class:`~.Submodule` ``ZK``
    returned by :py:func:`~sympy.polys.numberfields.basis.round_two`, which
    represents the maximal order in a number field, and with ideals formed
    therefrom, such as ``2 * ZK``.

    """
    if dm.domain.is_ZZ and dm.is_square and dm.is_upper:
        n = dm.shape[0]
        for i in range(n):
            d = dm[i, i].element
            if d <= 0:
                return False
            for j in range(i + 1, n):
                if not (0 <= dm[i, j].element < d):
                    return False
        return True
    return False


def make_mod_elt(module, col, denom=1):
    r"""
    Factory function which builds a :py:class:`~.ModuleElement`, but ensures
    that it is a :py:class:`~.PowerBasisElement` if the module is a
    :py:class:`~.PowerBasis`.
    """
    if isinstance(module, PowerBasis):
        return PowerBasisElement(module, col, denom=denom)
    else:
        return ModuleElement(module, col, denom=denom)


class ModuleElement(IntegerPowerable):
    r"""
    Represents an element of a :py:class:`~.Module`.

    NOTE: Should not be constructed directly. Use the
    :py:meth:`~.Module.__call__` method or the :py:func:`make_mod_elt()`
    factory function instead.
    """

    def __init__(self, module, col, denom=1):
        """
        Parameters
        ==========

        module : :py:class:`~.Module`
            The module to which this element belongs.
        col : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            Column vector giving the numerators of the coefficients of this
            element.
        denom : int, optional (default=1)
            Denominator for the coefficients of this element.

        """
        self.module = module
        self.col = col
        self.denom = denom
        self._QQ_col = None

    def __repr__(self):
        r = str([int(c) for c in self.col.flat()])
        if self.denom > 1:
            r += f'/{self.denom}'
        return r

    def reduced(self):
        """
        Produce a reduced version of this ModuleElement, i.e. one in which the
        gcd of the denominator together with all numerator coefficients is 1.
        """
        if self.denom == 1:
            return self
        g = igcd(self.denom, *self.coeffs)
        if g == 1:
            return self
        return type(self)(self.module,
                            (self.col / g).convert_to(ZZ),
                            denom=self.denom // g)

    def reduced_mod_p(self, p):
        """
        Produce a version of this :py:class:`~.ModuleElement` in which all
        numerator coefficients have been reduced mod *p*.
        """
        return make_mod_elt(self.module,
                            self.col.convert_to(FF(p)).convert_to(ZZ),
                            denom=self.denom)

    @classmethod
    def from_int_list(cls, module, coeffs, denom=1):
        """
        Make a :py:class:`~.ModuleElement` from a list of ints (instead of a
        column vector).
        """
        col = to_col(coeffs)
        return cls(module, col, denom=denom)

    @property
    def n(self):
        """The length of this element's column."""
        return self.module.n

    def __len__(self):
        return self.n

    def column(self, domain=None):
        """
        Get a copy of this element's column, optionally converting to a domain.
        """
        if domain is None:
            return self.col.copy()
        else:
            return self.col.convert_to(domain)

    @property
    def coeffs(self):
        return self.col.flat()

    @property
    def QQ_col(self):
        """
        :py:class:`~.DomainMatrix` over :ref:`QQ`, equal to
        ``self.col / self.denom``, and guaranteed to be dense.

        See Also
        ========

        .Submodule.QQ_matrix

        """
        if self._QQ_col is None:
            self._QQ_col = (self.col / self.denom).to_dense()
        return self._QQ_col

    def to_parent(self):
        """
        Transform into a :py:class:`~.ModuleElement` belonging to the parent of
        this element's module.
        """
        if not isinstance(self.module, Submodule):
            raise ValueError('Not an element of a Submodule.')
        return make_mod_elt(
            self.module.parent, self.module.matrix * self.col,
            denom=self.module.denom * self.denom)

    def to_ancestor(self, anc):
        """
        Transform into a :py:class:`~.ModuleElement` belonging to a given
        ancestor of this element's module.

        Parameters
        ==========

        anc : :py:class:`~.Module`

        """
        if anc == self.module:
            return self
        else:
            return self.to_parent().to_ancestor(anc)

    def over_power_basis(self):
        """
        Transform into a :py:class:`~.PowerBasisElement` over our
        :py:class:`~.PowerBasis` ancestor.
        """
        e = self
        while not isinstance(e.module, PowerBasis):
            e = e.to_parent()
        return e

    def is_compat(self, other):
        """
        Test whether other is another :py:class:`~.ModuleElement` with same
        module.
        """
        return isinstance(other, ModuleElement) and other.module == self.module

    def unify(self, other):
        """
        Try to make a compatible pair of :py:class:`~.ModuleElement`, one
        equivalent to this one, and one equivalent to the other.

        Explanation
        ===========

        We search for the nearest common ancestor module for the pair of
        elements, and represent each one there.

        Returns
        =======

        Pair ``(e1, e2)``
            Each ``ei`` is a :py:class:`~.ModuleElement`, they belong to the
            same :py:class:`~.Module`, ``e1`` is equivalent to ``self``, and
            ``e2`` is equivalent to ``other``.

        Raises
        ======

        UnificationFailed
            If ``self`` and ``other`` have no common ancestor module.

        """
        if self.module == other.module:
            return self, other
        nca = self.module.nearest_common_ancestor(other.module)
        if nca is not None:
            return self.to_ancestor(nca), other.to_ancestor(nca)
        raise UnificationFailed(f"Cannot unify {self} with {other}")

    def __eq__(self, other):
        if self.is_compat(other):
            return self.QQ_col == other.QQ_col
        return NotImplemented

    def equiv(self, other):
        """
        A :py:class:`~.ModuleElement` may test as equivalent to a rational
        number or another :py:class:`~.ModuleElement`, if they represent the
        same algebraic number.

        Explanation
        ===========

        This method is intended to check equivalence only in those cases in
        which it is easy to test; namely, when *other* is either a
        :py:class:`~.ModuleElement` that can be unified with this one (i.e. one
        which shares a common :py:class:`~.PowerBasis` ancestor), or else a
        rational number (which is easy because every :py:class:`~.PowerBasis`
        represents every rational number).

        Parameters
        ==========

        other : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.ModuleElement`

        Returns
        =======

        bool

        Raises
        ======

        UnificationFailed
            If ``self`` and ``other`` do not share a common
            :py:class:`~.PowerBasis` ancestor.

        """
        if self == other:
            return True
        elif isinstance(other, ModuleElement):
            a, b = self.unify(other)
            return a == b
        elif is_rat(other):
            if isinstance(self, PowerBasisElement):
                return self == self.module(0) * other
            else:
                return self.over_power_basis().equiv(other)
        return False

    def __add__(self, other):
        """
        A :py:class:`~.ModuleElement` can be added to a rational number, or to
        another :py:class:`~.ModuleElement`.

        Explanation
        ===========

        When the other summand is a rational number, it will be converted into
        a :py:class:`~.ModuleElement` (belonging to the first ancestor of this
        module that starts with unity).

        In all cases, the sum belongs to the nearest common ancestor (NCA) of
        the modules of the two summands. If the NCA does not exist, we return
        ``NotImplemented``.
        """
        if self.is_compat(other):
            d, e = self.denom, other.denom
            m = ilcm(d, e)
            u, v = m // d, m // e
            col = to_col([u * a + v * b for a, b in zip(self.coeffs, other.coeffs)])
            return type(self)(self.module, col, denom=m).reduced()
        elif isinstance(other, ModuleElement):
            try:
                a, b = self.unify(other)
            except UnificationFailed:
                return NotImplemented
            return a + b
        elif is_rat(other):
            return self + self.module.element_from_rational(other)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        """
        A :py:class:`~.ModuleElement` can be multiplied by a rational number,
        or by another :py:class:`~.ModuleElement`.

        Explanation
        ===========

        When the multiplier is a rational number, the product is computed by
        operating directly on the coefficients of this
        :py:class:`~.ModuleElement`.

        When the multiplier is another :py:class:`~.ModuleElement`, the product
        will belong to the nearest common ancestor (NCA) of the modules of the
        two operands, and that NCA must have a multiplication table. If the NCA
        does not exist, we return ``NotImplemented``. If the NCA does not have
        a mult. table, ``ClosureFailure`` will be raised.
        """
        if self.is_compat(other):
            M = self.module.mult_tab()
            A, B = self.col.flat(), other.col.flat()
            n = self.n
            C = [0] * n
            for u in range(n):
                for v in range(u, n):
                    c = A[u] * B[v]
                    if v > u:
                        c += A[v] * B[u]
                    if c != 0:
                        R = M[u][v]
                        for k in range(n):
                            C[k] += c * R[k]
            d = self.denom * other.denom
            return self.from_int_list(self.module, C, denom=d)
        elif isinstance(other, ModuleElement):
            try:
                a, b = self.unify(other)
            except UnificationFailed:
                return NotImplemented
            return a * b
        elif is_rat(other):
            a, b = get_num_denom(other)
            if a == b == 1:
                return self
            else:
                return make_mod_elt(self.module,
                                 self.col * a, denom=self.denom * b).reduced()
        return NotImplemented

    __rmul__ = __mul__

    def _zeroth_power(self):
        return self.module.one()

    def _first_power(self):
        return self

    def __floordiv__(self, a):
        if is_rat(a):
            a = QQ(a)
            return self * (1/a)
        elif isinstance(a, ModuleElement):
            return self * (1//a)
        return NotImplemented

    def __rfloordiv__(self, a):
        return a // self.over_power_basis()

    def __mod__(self, m):
        r"""
        Reduce this :py:class:`~.ModuleElement` mod a :py:class:`~.Submodule`.

        Parameters
        ==========

        m : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.Submodule`
            If a :py:class:`~.Submodule`, reduce ``self`` relative to this.
            If an integer or rational, reduce relative to the
            :py:class:`~.Submodule` that is our own module times this constant.

        See Also
        ========

        .Submodule.reduce_element

        """
        if is_rat(m):
            m = m * self.module.whole_submodule()
        if isinstance(m, Submodule) and m.parent == self.module:
            return m.reduce_element(self)
        return NotImplemented


class PowerBasisElement(ModuleElement):
    r"""
    Subclass for :py:class:`~.ModuleElement` instances whose module is a
    :py:class:`~.PowerBasis`.
    """

    @property
    def T(self):
        """Access the defining polynomial of the :py:class:`~.PowerBasis`."""
        return self.module.T

    def numerator(self, x=None):
        """Obtain the numerator as a polynomial over :ref:`ZZ`."""
        x = x or self.T.gen
        return Poly(reversed(self.coeffs), x, domain=ZZ)

    def poly(self, x=None):
        """Obtain the number as a polynomial over :ref:`QQ`."""
        return self.numerator(x=x) // self.denom

    @property
    def is_rational(self):
        """Say whether this element represents a rational number."""
        return self.col[1:, :].is_zero_matrix

    @property
    def generator(self):
        """
        Return a :py:class:`~.Symbol` to be used when expressing this element
        as a polynomial.

        If we have an associated :py:class:`~.AlgebraicField` whose primitive
        element has an alias symbol, we use that. Otherwise we use the variable
        of the minimal polynomial defining the power basis to which we belong.
        """
        K = self.module.number_field
        return K.ext.alias if K and K.ext.is_aliased else self.T.gen

    def as_expr(self, x=None):
        """Create a Basic expression from ``self``. """
        return self.poly(x or self.generator).as_expr()

    def norm(self, T=None):
        """Compute the norm of this number."""
        T = T or self.T
        x = T.gen
        A = self.numerator(x=x)
        return T.resultant(A) // self.denom ** self.n

    def inverse(self):
        f = self.poly()
        f_inv = f.invert(self.T)
        return self.module.element_from_poly(f_inv)

    def __rfloordiv__(self, a):
        return self.inverse() * a

    def _negative_power(self, e, modulo=None):
        return self.inverse() ** abs(e)

    def to_ANP(self):
        """Convert to an equivalent :py:class:`~.ANP`. """
        return ANP(list(reversed(self.QQ_col.flat())), QQ.map(self.T.rep.to_list()), QQ)

    def to_alg_num(self):
        """
        Try to convert to an equivalent :py:class:`~.AlgebraicNumber`.

        Explanation
        ===========

        In general, the conversion from an :py:class:`~.AlgebraicNumber` to a
        :py:class:`~.PowerBasisElement` throws away information, because an
        :py:class:`~.AlgebraicNumber` specifies a complex embedding, while a
        :py:class:`~.PowerBasisElement` does not. However, in some cases it is
        possible to convert a :py:class:`~.PowerBasisElement` back into an
        :py:class:`~.AlgebraicNumber`, namely when the associated
        :py:class:`~.PowerBasis` has a reference to an
        :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicNumber`

        Raises
        ======

        StructureError
            If the :py:class:`~.PowerBasis` to which this element belongs does
            not have an associated :py:class:`~.AlgebraicField`.

        """
        K = self.module.number_field
        if K:
            return K.to_alg_num(self.to_ANP())
        raise StructureError("No associated AlgebraicField")


class ModuleHomomorphism:
    r"""A homomorphism from one module to another."""

    def __init__(self, domain, codomain, mapping):
        r"""
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain of the mapping.

        codomain : :py:class:`~.Module`
            The codomain of the mapping.

        mapping : callable
            An arbitrary callable is accepted, but should be chosen so as
            to represent an actual module homomorphism. In particular, should
            accept elements of *domain* and return elements of *codomain*.

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis, ModuleHomomorphism
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_gens([2*A(j) for j in range(4)])
        >>> phi = ModuleHomomorphism(A, B, lambda x: 6*x)
        >>> print(phi.matrix())  # doctest: +SKIP
        DomainMatrix([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]], (4, 4), ZZ)

        """
        self.domain = domain
        self.codomain = codomain
        self.mapping = mapping

    def matrix(self, modulus=None):
        r"""
        Compute the matrix of this homomorphism.

        Parameters
        ==========

        modulus : int, optional
            A positive prime number $p$ if the matrix should be reduced mod
            $p$.

        Returns
        =======

        :py:class:`~.DomainMatrix`
            The matrix is over :ref:`ZZ`, or else over :ref:`GF(p)` if a
            modulus was given.

        """
        basis = self.domain.basis_elements()
        cols = [self.codomain.represent(self.mapping(elt)) for elt in basis]
        if not cols:
            return DomainMatrix.zeros((self.codomain.n, 0), ZZ).to_dense()
        M = cols[0].hstack(*cols[1:])
        if modulus:
            M = M.convert_to(FF(modulus))
        return M

    def kernel(self, modulus=None):
        r"""
        Compute a Submodule representing the kernel of this homomorphism.

        Parameters
        ==========

        modulus : int, optional
            A positive prime number $p$ if the kernel should be computed mod
            $p$.

        Returns
        =======

        :py:class:`~.Submodule`
            This submodule's generators span the kernel of this
            homomorphism over :ref:`ZZ`, or else over :ref:`GF(p)` if a
            modulus was given.

        """
        M = self.matrix(modulus=modulus)
        if modulus is None:
            M = M.convert_to(QQ)
        # Note: Even when working over a finite field, what we want here is
        # the pullback into the integers, so in this case the conversion to ZZ
        # below is appropriate. When working over ZZ, the kernel should be a
        # ZZ-submodule, so, while the conversion to QQ above was required in
        # order for the nullspace calculation to work, conversion back to ZZ
        # afterward should always work.
        # TODO:
        #  Watch <https://github.com/sympy/sympy/issues/21834>, which calls
        #  for fraction-free algorithms. If this is implemented, we can skip
        #  the conversion to `QQ` above.
        K = M.nullspace().convert_to(ZZ).transpose()
        return self.domain.submodule_from_matrix(K)


class ModuleEndomorphism(ModuleHomomorphism):
    r"""A homomorphism from one module to itself."""

    def __init__(self, domain, mapping):
        r"""
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The common domain and codomain of the mapping.

        mapping : callable
            An arbitrary callable is accepted, but should be chosen so as
            to represent an actual module endomorphism. In particular, should
            accept and return elements of *domain*.

        """
        super().__init__(domain, domain, mapping)


class InnerEndomorphism(ModuleEndomorphism):
    r"""
    An inner endomorphism on a module, i.e. the endomorphism corresponding to
    multiplication by a fixed element.
    """

    def __init__(self, domain, multiplier):
        r"""
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain and codomain of the endomorphism.

        multiplier : :py:class:`~.ModuleElement`
            The element $a$ defining the mapping as $x \mapsto a x$.

        """
        super().__init__(domain, lambda x: multiplier * x)
        self.multiplier = multiplier


class EndomorphismRing:
    r"""The ring of endomorphisms on a module."""

    def __init__(self, domain):
        """
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain and codomain of the endomorphisms.

        """
        self.domain = domain

    def inner_endomorphism(self, multiplier):
        r"""
        Form an inner endomorphism belonging to this endomorphism ring.

        Parameters
        ==========

        multiplier : :py:class:`~.ModuleElement`
            Element $a$ defining the inner endomorphism $x \mapsto a x$.

        Returns
        =======

        :py:class:`~.InnerEndomorphism`

        """
        return InnerEndomorphism(self.domain, multiplier)

    def represent(self, element):
        r"""
        Represent an element of this endomorphism ring, as a single column
        vector.

        Explanation
        ===========

        Let $M$ be a module, and $E$ its ring of endomorphisms. Let $N$ be
        another module, and consider a homomorphism $\varphi: N \rightarrow E$.
        In the event that $\varphi$ is to be represented by a matrix $A$, each
        column of $A$ must represent an element of $E$. This is possible when
        the elements of $E$ are themselves representable as matrices, by
        stacking the columns of such a matrix into a single column.

        This method supports calculating such matrices $A$, by representing
        an element of this endomorphism ring first as a matrix, and then
        stacking that matrix's columns into a single column.

        Examples
        ========

        Note that in these examples we print matrix transposes, to make their
        columns easier to inspect.

        >>> from sympy import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> from sympy.polys.numberfields.modules import ModuleHomomorphism
        >>> T = Poly(cyclotomic_poly(5))
        >>> M = PowerBasis(T)
        >>> E = M.endomorphism_ring()

        Let $\zeta$ be a primitive 5th root of unity, a generator of our field,
        and consider the inner endomorphism $\tau$ on the ring of integers,
        induced by $\zeta$:

        >>> zeta = M(1)
        >>> tau = E.inner_endomorphism(zeta)
        >>> tau.matrix().transpose()  # doctest: +SKIP
        DomainMatrix(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]],
            (4, 4), ZZ)

        The matrix representation of $\tau$ is as expected. The first column
        shows that multiplying by $\zeta$ carries $1$ to $\zeta$, the second
        column that it carries $\zeta$ to $\zeta^2$, and so forth.

        The ``represent`` method of the endomorphism ring ``E`` stacks these
        into a single column:

        >>> E.represent(tau).transpose()  # doctest: +SKIP
        DomainMatrix(
            [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1]],
            (1, 16), ZZ)

        This is useful when we want to consider a homomorphism $\varphi$ having
        ``E`` as codomain:

        >>> phi = ModuleHomomorphism(M, E, lambda x: E.inner_endomorphism(x))

        and we want to compute the matrix of such a homomorphism:

        >>> phi.matrix().transpose()  # doctest: +SKIP
        DomainMatrix(
            [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1],
            [0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1, 1, 0, 0, 0],
            [0, 0, 0, 1, -1, -1, -1, -1, 1, 0, 0, 0, 0, 1, 0, 0]],
            (4, 16), ZZ)

        Note that the stacked matrix of $\tau$ occurs as the second column in
        this example. This is because $\zeta$ is the second basis element of
        ``M``, and $\varphi(\zeta) = \tau$.

        Parameters
        ==========

        element : :py:class:`~.ModuleEndomorphism` belonging to this ring.

        Returns
        =======

        :py:class:`~.DomainMatrix`
            Column vector equalling the vertical stacking of all the columns
            of the matrix that represents the given *element* as a mapping.

        """
        if isinstance(element, ModuleEndomorphism) and element.domain == self.domain:
            M = element.matrix()
            # Transform the matrix into a single column, which should reproduce
            # the original columns, one after another.
            m, n = M.shape
            if n == 0:
                return M
            return M[:, 0].vstack(*[M[:, j] for j in range(1, n)])
        raise NotImplementedError


def find_min_poly(alpha, domain, x=None, powers=None):
    r"""
    Find a polynomial of least degree (not necessarily irreducible) satisfied
    by an element of a finitely-generated ring with unity.

    Examples
    ========

    For the $n$th cyclotomic field, $n$ an odd prime, consider the quadratic
    equation whose roots are the two periods of length $(n-1)/2$. Article 356
    of Gauss tells us that we should get $x^2 + x - (n-1)/4$ or
    $x^2 + x + (n+1)/4$ according to whether $n$ is 1 or 3 mod 4, respectively.

    >>> from sympy import Poly, cyclotomic_poly, primitive_root, QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.modules import PowerBasis, find_min_poly
    >>> n = 13
    >>> g = primitive_root(n)
    >>> C = PowerBasis(Poly(cyclotomic_poly(n, x)))
    >>> ee = [g**(2*k+1) % n for k in range((n-1)//2)]
    >>> eta = sum(C(e) for e in ee)
    >>> print(find_min_poly(eta, QQ, x=x).as_expr())
    x**2 + x - 3
    >>> n = 19
    >>> g = primitive_root(n)
    >>> C = PowerBasis(Poly(cyclotomic_poly(n, x)))
    >>> ee = [g**(2*k+2) % n for k in range((n-1)//2)]
    >>> eta = sum(C(e) for e in ee)
    >>> print(find_min_poly(eta, QQ, x=x).as_expr())
    x**2 + x + 5

    Parameters
    ==========

    alpha : :py:class:`~.ModuleElement`
        The element whose min poly is to be found, and whose module has
        multiplication and starts with unity.

    domain : :py:class:`~.Domain`
        The desired domain of the polynomial.

    x : :py:class:`~.Symbol`, optional
        The desired variable for the polynomial.

    powers : list, optional
        If desired, pass an empty list. The powers of *alpha* (as
        :py:class:`~.ModuleElement` instances) from the zeroth up to the degree
        of the min poly will be recorded here, as we compute them.

    Returns
    =======

    :py:class:`~.Poly`, ``None``
        The minimal polynomial for alpha, or ``None`` if no polynomial could be
        found over the desired domain.

    Raises
    ======

    MissingUnityError
        If the module to which alpha belongs does not start with unity.
    ClosureFailure
        If the module to which alpha belongs is not closed under
        multiplication.

    """
    R = alpha.module
    if not R.starts_with_unity():
        raise MissingUnityError("alpha must belong to finitely generated ring with unity.")
    if powers is None:
        powers = []
    one = R(0)
    powers.append(one)
    powers_matrix = one.column(domain=domain)
    ak = alpha
    m = None
    for k in range(1, R.n + 1):
        powers.append(ak)
        ak_col = ak.column(domain=domain)
        try:
            X = powers_matrix._solve(ak_col)[0]
        except DMBadInputError:
            # This means alpha^k still isn't in the domain-span of the lower powers.
            powers_matrix = powers_matrix.hstack(ak_col)
            ak *= alpha
        else:
            # alpha^k is in the domain-span of the lower powers, so we have found a
            # minimal-degree poly for alpha.
            coeffs = [1] + [-c for c in reversed(X.to_list_flat())]
            x = x or Dummy('x')
            if domain.is_FF:
                m = Poly(coeffs, x, modulus=domain.mod)
            else:
                m = Poly(coeffs, x, domain=domain)
            break
    return m
