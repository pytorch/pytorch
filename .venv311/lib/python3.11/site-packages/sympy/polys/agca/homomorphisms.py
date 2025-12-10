"""
Computations with homomorphisms of modules and rings.

This module implements classes for representing homomorphisms of rings and
their modules. Instead of instantiating the classes directly, you should use
the function ``homomorphism(from, to, matrix)`` to create homomorphism objects.
"""


from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
    SubModule, SubQuotientModule)
from sympy.polys.polyerrors import CoercionFailed

# The main computational task for module homomorphisms is kernels.
# For this reason, the concrete classes are organised by domain module type.


class ModuleHomomorphism:
    """
    Abstract base class for module homomoprhisms. Do not instantiate.

    Instead, use the ``homomorphism`` function:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> homomorphism(F, F, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : QQ[x]**2 -> QQ[x]**2
    [0, 1]])

    Attributes:

    - ring - the ring over which we are considering modules
    - domain - the domain module
    - codomain - the codomain module
    - _ker - cached kernel
    - _img - cached image

    Non-implemented methods:

    - _kernel
    - _image
    - _restrict_domain
    - _restrict_codomain
    - _quotient_domain
    - _quotient_codomain
    - _apply
    - _mul_scalar
    - _compose
    - _add
    """

    def __init__(self, domain, codomain):
        if not isinstance(domain, Module):
            raise TypeError('Source must be a module, got %s' % domain)
        if not isinstance(codomain, Module):
            raise TypeError('Target must be a module, got %s' % codomain)
        if domain.ring != codomain.ring:
            raise ValueError('Source and codomain must be over same ring, '
                             'got %s != %s' % (domain, codomain))
        self.domain = domain
        self.codomain = codomain
        self.ring = domain.ring
        self._ker = None
        self._img = None

    def kernel(self):
        r"""
        Compute the kernel of ``self``.

        That is, if ``self`` is the homomorphism `\phi: M \to N`, then compute
        `ker(\phi) = \{x \in M | \phi(x) = 0\}`.  This is a submodule of `M`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> homomorphism(F, F, [[1, 0], [x, 0]]).kernel()
        <[x, -1]>
        """
        if self._ker is None:
            self._ker = self._kernel()
        return self._ker

    def image(self):
        r"""
        Compute the image of ``self``.

        That is, if ``self`` is the homomorphism `\phi: M \to N`, then compute
        `im(\phi) = \{\phi(x) | x \in M \}`.  This is a submodule of `N`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> homomorphism(F, F, [[1, 0], [x, 0]]).image() == F.submodule([1, 0])
        True
        """
        if self._img is None:
            self._img = self._image()
        return self._img

    def _kernel(self):
        """Compute the kernel of ``self``."""
        raise NotImplementedError

    def _image(self):
        """Compute the image of ``self``."""
        raise NotImplementedError

    def _restrict_domain(self, sm):
        """Implementation of domain restriction."""
        raise NotImplementedError

    def _restrict_codomain(self, sm):
        """Implementation of codomain restriction."""
        raise NotImplementedError

    def _quotient_domain(self, sm):
        """Implementation of domain quotient."""
        raise NotImplementedError

    def _quotient_codomain(self, sm):
        """Implementation of codomain quotient."""
        raise NotImplementedError

    def restrict_domain(self, sm):
        """
        Return ``self``, with the domain restricted to ``sm``.

        Here ``sm`` has to be a submodule of ``self.domain``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.restrict_domain(F.submodule([1, 0]))
        Matrix([
        [1, x], : <[1, 0]> -> QQ[x]**2
        [0, 0]])

        This is the same as just composing on the right with the submodule
        inclusion:

        >>> h * F.submodule([1, 0]).inclusion_hom()
        Matrix([
        [1, x], : <[1, 0]> -> QQ[x]**2
        [0, 0]])
        """
        if not self.domain.is_submodule(sm):
            raise ValueError('sm must be a submodule of %s, got %s'
                             % (self.domain, sm))
        if sm == self.domain:
            return self
        return self._restrict_domain(sm)

    def restrict_codomain(self, sm):
        """
        Return ``self``, with codomain restricted to to ``sm``.

        Here ``sm`` has to be a submodule of ``self.codomain`` containing the
        image.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.restrict_codomain(F.submodule([1, 0]))
        Matrix([
        [1, x], : QQ[x]**2 -> <[1, 0]>
        [0, 0]])
        """
        if not sm.is_submodule(self.image()):
            raise ValueError('the image %s must contain sm, got %s'
                             % (self.image(), sm))
        if sm == self.codomain:
            return self
        return self._restrict_codomain(sm)

    def quotient_domain(self, sm):
        """
        Return ``self`` with domain replaced by ``domain/sm``.

        Here ``sm`` must be a submodule of ``self.kernel()``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.quotient_domain(F.submodule([-x, 1]))
        Matrix([
        [1, x], : QQ[x]**2/<[-x, 1]> -> QQ[x]**2
        [0, 0]])
        """
        if not self.kernel().is_submodule(sm):
            raise ValueError('kernel %s must contain sm, got %s' %
                             (self.kernel(), sm))
        if sm.is_zero():
            return self
        return self._quotient_domain(sm)

    def quotient_codomain(self, sm):
        """
        Return ``self`` with codomain replaced by ``codomain/sm``.

        Here ``sm`` must be a submodule of ``self.codomain``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.quotient_codomain(F.submodule([1, 1]))
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2/<[1, 1]>
        [0, 0]])

        This is the same as composing with the quotient map on the left:

        >>> (F/[(1, 1)]).quotient_hom() * h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2/<[1, 1]>
        [0, 0]])
        """
        if not self.codomain.is_submodule(sm):
            raise ValueError('sm must be a submodule of codomain %s, got %s'
                             % (self.codomain, sm))
        if sm.is_zero():
            return self
        return self._quotient_codomain(sm)

    def _apply(self, elem):
        """Apply ``self`` to ``elem``."""
        raise NotImplementedError

    def __call__(self, elem):
        return self.codomain.convert(self._apply(self.domain.convert(elem)))

    def _compose(self, oth):
        """
        Compose ``self`` with ``oth``, that is, return the homomorphism
        obtained by first applying then ``self``, then ``oth``.

        (This method is private since in this syntax, it is non-obvious which
        homomorphism is executed first.)
        """
        raise NotImplementedError

    def _mul_scalar(self, c):
        """Scalar multiplication. ``c`` is guaranteed in self.ring."""
        raise NotImplementedError

    def _add(self, oth):
        """
        Homomorphism addition.
        ``oth`` is guaranteed to be a homomorphism with same domain/codomain.
        """
        raise NotImplementedError

    def _check_hom(self, oth):
        """Helper to check that oth is a homomorphism with same domain/codomain."""
        if not isinstance(oth, ModuleHomomorphism):
            return False
        return oth.domain == self.domain and oth.codomain == self.codomain

    def __mul__(self, oth):
        if isinstance(oth, ModuleHomomorphism) and self.domain == oth.codomain:
            return oth._compose(self)
        try:
            return self._mul_scalar(self.ring.convert(oth))
        except CoercionFailed:
            return NotImplemented

    # NOTE: _compose will never be called from rmul
    __rmul__ = __mul__

    def __truediv__(self, oth):
        try:
            return self._mul_scalar(1/self.ring.convert(oth))
        except CoercionFailed:
            return NotImplemented

    def __add__(self, oth):
        if self._check_hom(oth):
            return self._add(oth)
        return NotImplemented

    def __sub__(self, oth):
        if self._check_hom(oth):
            return self._add(oth._mul_scalar(self.ring.convert(-1)))
        return NotImplemented

    def is_injective(self):
        """
        Return True if ``self`` is injective.

        That is, check if the elements of the domain are mapped to the same
        codomain element.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_injective()
        False
        >>> h.quotient_domain(h.kernel()).is_injective()
        True
        """
        return self.kernel().is_zero()

    def is_surjective(self):
        """
        Return True if ``self`` is surjective.

        That is, check if every element of the codomain has at least one
        preimage.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_surjective()
        False
        >>> h.restrict_codomain(h.image()).is_surjective()
        True
        """
        return self.image() == self.codomain

    def is_isomorphism(self):
        """
        Return True if ``self`` is an isomorphism.

        That is, check if every element of the codomain has precisely one
        preimage. Equivalently, ``self`` is both injective and surjective.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h = h.restrict_codomain(h.image())
        >>> h.is_isomorphism()
        False
        >>> h.quotient_domain(h.kernel()).is_isomorphism()
        True
        """
        return self.is_injective() and self.is_surjective()

    def is_zero(self):
        """
        Return True if ``self`` is a zero morphism.

        That is, check if every element of the domain is mapped to zero
        under self.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_zero()
        False
        >>> h.restrict_domain(F.submodule()).is_zero()
        True
        >>> h.quotient_codomain(h.image()).is_zero()
        True
        """
        return self.image().is_zero()

    def __eq__(self, oth):
        try:
            return (self - oth).is_zero()
        except TypeError:
            return False

    def __ne__(self, oth):
        return not (self == oth)


class MatrixHomomorphism(ModuleHomomorphism):
    r"""
    Helper class for all homomoprhisms which are expressed via a matrix.

    That is, for such homomorphisms ``domain`` is contained in a module
    generated by finitely many elements `e_1, \ldots, e_n`, so that the
    homomorphism is determined uniquely by its action on the `e_i`. It
    can thus be represented as a vector of elements of the codomain module,
    or potentially a supermodule of the codomain module
    (and hence conventionally as a matrix, if there is a similar interpretation
    for elements of the codomain module).

    Note that this class does *not* assume that the `e_i` freely generate a
    submodule, nor that ``domain`` is even all of this submodule. It exists
    only to unify the interface.

    Do not instantiate.

    Attributes:

    - matrix - the list of images determining the homomorphism.
    NOTE: the elements of matrix belong to either self.codomain or
          self.codomain.container

    Still non-implemented methods:

    - kernel
    - _apply
    """

    def __init__(self, domain, codomain, matrix):
        ModuleHomomorphism.__init__(self, domain, codomain)
        if len(matrix) != domain.rank:
            raise ValueError('Need to provide %s elements, got %s'
                             % (domain.rank, len(matrix)))

        converter = self.codomain.convert
        if isinstance(self.codomain, (SubModule, SubQuotientModule)):
            converter = self.codomain.container.convert
        self.matrix = tuple(converter(x) for x in matrix)

    def _sympy_matrix(self):
        """Helper function which returns a SymPy matrix ``self.matrix``."""
        from sympy.matrices import Matrix
        c = lambda x: x
        if isinstance(self.codomain, (QuotientModule, SubQuotientModule)):
            c = lambda x: x.data
        return Matrix([[self.ring.to_sympy(y) for y in c(x)] for x in self.matrix]).T

    def __repr__(self):
        lines = repr(self._sympy_matrix()).split('\n')
        t = " : %s -> %s" % (self.domain, self.codomain)
        s = ' '*len(t)
        n = len(lines)
        for i in range(n // 2):
            lines[i] += s
        lines[n // 2] += t
        for i in range(n//2 + 1, n):
            lines[i] += s
        return '\n'.join(lines)

    def _restrict_domain(self, sm):
        """Implementation of domain restriction."""
        return SubModuleHomomorphism(sm, self.codomain, self.matrix)

    def _restrict_codomain(self, sm):
        """Implementation of codomain restriction."""
        return self.__class__(self.domain, sm, self.matrix)

    def _quotient_domain(self, sm):
        """Implementation of domain quotient."""
        return self.__class__(self.domain/sm, self.codomain, self.matrix)

    def _quotient_codomain(self, sm):
        """Implementation of codomain quotient."""
        Q = self.codomain/sm
        converter = Q.convert
        if isinstance(self.codomain, SubModule):
            converter = Q.container.convert
        return self.__class__(self.domain, self.codomain/sm,
            [converter(x) for x in self.matrix])

    def _add(self, oth):
        return self.__class__(self.domain, self.codomain,
                              [x + y for x, y in zip(self.matrix, oth.matrix)])

    def _mul_scalar(self, c):
        return self.__class__(self.domain, self.codomain, [c*x for x in self.matrix])

    def _compose(self, oth):
        return self.__class__(self.domain, oth.codomain, [oth(x) for x in self.matrix])


class FreeModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphisms with domain a free module or a quotient
    thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> homomorphism(F, F, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : QQ[x]**2 -> QQ[x]**2
    [0, 1]])
    """

    def _apply(self, elem):
        if isinstance(self.domain, QuotientModule):
            elem = elem.data
        return sum(x * e for x, e in zip(elem, self.matrix))

    def _image(self):
        return self.codomain.submodule(*self.matrix)

    def _kernel(self):
        # The domain is either a free module or a quotient thereof.
        # It does not matter if it is a quotient, because that won't increase
        # the kernel.
        # Our generators {e_i} are sent to the matrix entries {b_i}.
        # The kernel is essentially the syzygy module of these {b_i}.
        syz = self.image().syzygy_module()
        return self.domain.submodule(*syz.gens)


class SubModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphism with domain a submodule of a free module
    or a quotient thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> M = QQ.old_poly_ring(x).free_module(2)*x
    >>> homomorphism(M, M, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : <[x, 0], [0, x]> -> <[x, 0], [0, x]>
    [0, 1]])
    """

    def _apply(self, elem):
        if isinstance(self.domain, SubQuotientModule):
            elem = elem.data
        return sum(x * e for x, e in zip(elem, self.matrix))

    def _image(self):
        return self.codomain.submodule(*[self(x) for x in self.domain.gens])

    def _kernel(self):
        syz = self.image().syzygy_module()
        return self.domain.submodule(
            *[sum(xi*gi for xi, gi in zip(s, self.domain.gens))
              for s in syz.gens])


def homomorphism(domain, codomain, matrix):
    r"""
    Create a homomorphism object.

    This function tries to build a homomorphism from ``domain`` to ``codomain``
    via the matrix ``matrix``.

    Examples
    ========

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> R = QQ.old_poly_ring(x)
    >>> T = R.free_module(2)

    If ``domain`` is a free module generated by `e_1, \ldots, e_n`, then
    ``matrix`` should be an n-element iterable `(b_1, \ldots, b_n)` where
    the `b_i` are elements of ``codomain``. The constructed homomorphism is the
    unique homomorphism sending `e_i` to `b_i`.

    >>> F = R.free_module(2)
    >>> h = homomorphism(F, T, [[1, x], [x**2, 0]])
    >>> h
    Matrix([
    [1, x**2], : QQ[x]**2 -> QQ[x]**2
    [x,    0]])
    >>> h([1, 0])
    [1, x]
    >>> h([0, 1])
    [x**2, 0]
    >>> h([1, 1])
    [x**2 + 1, x]

    If ``domain`` is a submodule of a free module, them ``matrix`` determines
    a homomoprhism from the containing free module to ``codomain``, and the
    homomorphism returned is obtained by restriction to ``domain``.

    >>> S = F.submodule([1, 0], [0, x])
    >>> homomorphism(S, T, [[1, x], [x**2, 0]])
    Matrix([
    [1, x**2], : <[1, 0], [0, x]> -> QQ[x]**2
    [x,    0]])

    If ``domain`` is a (sub)quotient `N/K`, then ``matrix`` determines a
    homomorphism from `N` to ``codomain``. If the kernel contains `K`, this
    homomorphism descends to ``domain`` and is returned; otherwise an exception
    is raised.

    >>> homomorphism(S/[(1, 0)], T, [0, [x**2, 0]])
    Matrix([
    [0, x**2], : <[1, 0] + <[1, 0]>, [0, x] + <[1, 0]>, [1, 0] + <[1, 0]>> -> QQ[x]**2
    [0,    0]])
    >>> homomorphism(S/[(0, x)], T, [0, [x**2, 0]])
    Traceback (most recent call last):
    ...
    ValueError: kernel <[1, 0], [0, 0]> must contain sm, got <[0,x]>

    """
    def freepres(module):
        """
        Return a tuple ``(F, S, Q, c)`` where ``F`` is a free module, ``S`` is a
        submodule of ``F``, and ``Q`` a submodule of ``S``, such that
        ``module = S/Q``, and ``c`` is a conversion function.
        """
        if isinstance(module, FreeModule):
            return module, module, module.submodule(), lambda x: module.convert(x)
        if isinstance(module, QuotientModule):
            return (module.base, module.base, module.killed_module,
                    lambda x: module.convert(x).data)
        if isinstance(module, SubQuotientModule):
            return (module.base.container, module.base, module.killed_module,
                    lambda x: module.container.convert(x).data)
        # an ordinary submodule
        return (module.container, module, module.submodule(),
                lambda x: module.container.convert(x))

    SF, SS, SQ, _ = freepres(domain)
    TF, TS, TQ, c = freepres(codomain)
    # NOTE this is probably a bit inefficient (redundant checks)
    return FreeModuleHomomorphism(SF, TF, [c(x) for x in matrix]
         ).restrict_domain(SS).restrict_codomain(TS
         ).quotient_codomain(TQ).quotient_domain(SQ)
