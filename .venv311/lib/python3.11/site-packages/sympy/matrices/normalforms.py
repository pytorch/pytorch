'''Functions returning normal forms of matrices'''

from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import (
        smith_normal_form as _snf,
        is_smith_normal_form as _is_snf,
        smith_normal_decomp as _snd,
        invariant_factors as _invf,
        hermite_normal_form as _hnf,
    )


def _to_domain(m, domain=None):
    """Convert Matrix to DomainMatrix"""
    # XXX: deprecated support for RawMatrix:
    ring = getattr(m, "ring", None)
    m = m.applyfunc(lambda e: e.as_expr() if isinstance(e, Poly) else e)

    dM = DomainMatrix.from_Matrix(m)

    domain = domain or ring
    if domain is not None:
        dM = dM.convert_to(domain)
    return dM


def smith_normal_form(m, domain=None):
    '''
    Return the Smith Normal Form of a matrix `m` over the ring `domain`.
    This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import Matrix, ZZ
    >>> from sympy.matrices.normalforms import smith_normal_form
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> print(smith_normal_form(m, domain=ZZ))
    Matrix([[1, 0, 0], [0, 10, 0], [0, 0, 30]])

    '''
    dM = _to_domain(m, domain)
    return _snf(dM).to_Matrix()


def is_smith_normal_form(m, domain=None):
    '''
    Checks that the matrix is in Smith Normal Form
    '''
    dM = _to_domain(m, domain)
    return _is_snf(dM)


def smith_normal_decomp(m, domain=None):
    '''
    Return the Smith Normal Decomposition of a matrix `m` over the ring
    `domain`. This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import Matrix, ZZ
    >>> from sympy.matrices.normalforms import smith_normal_decomp
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> a, s, t = smith_normal_decomp(m, domain=ZZ)
    >>> assert a == s * m * t
    '''
    dM = _to_domain(m, domain)
    a, s, t = _snd(dM)
    return a.to_Matrix(), s.to_Matrix(), t.to_Matrix()


def invariant_factors(m, domain=None):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    .. [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    '''
    dM = _to_domain(m, domain)
    factors = _invf(dM)
    factors = tuple(dM.domain.to_sympy(f) for f in factors)
    # XXX: deprecated.
    if hasattr(m, "ring"):
        if m.ring.is_PolynomialRing:
            K = m.ring
            to_poly = lambda f: Poly(f, K.symbols, domain=K.domain)
            factors = tuple(to_poly(f) for f in factors)
    return factors


def hermite_normal_form(A, *, D=None, check_rank=False):
    r"""
    Compute the Hermite Normal Form of a Matrix *A* of integers.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.matrices.normalforms import hermite_normal_form
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> print(hermite_normal_form(m))
    Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])

    Parameters
    ==========

    A : $m \times n$ ``Matrix`` of integers.

    D : int, optional
        Let $W$ be the HNF of *A*. If known in advance, a positive integer *D*
        being any multiple of $\det(W)$ may be provided. In this case, if *A*
        also has rank $m$, then we may use an alternative algorithm that works
        mod *D* in order to prevent coefficient explosion.

    check_rank : boolean, optional (default=False)
        The basic assumption is that, if you pass a value for *D*, then
        you already believe that *A* has rank $m$, so we do not waste time
        checking it for you. If you do want this to be checked (and the
        ordinary, non-modulo *D* algorithm to be used if the check fails), then
        set *check_rank* to ``True``.

    Returns
    =======

    ``Matrix``
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`.

    DMShapeError
        If the mod *D* algorithm is used but the matrix has more rows than
        columns.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithms 2.4.5 and 2.4.8.)

    """
    # Accept any of Python int, SymPy Integer, and ZZ itself:
    if D is not None and not ZZ.of_type(D):
        D = ZZ(int(D))
    return _hnf(A._rep, D=D, check_rank=check_rank).to_Matrix()
