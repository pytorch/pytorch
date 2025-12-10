'''Functions returning normal forms of matrices'''

from collections import defaultdict

from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ


# TODO (future work):
#  There are faster algorithms for Smith and Hermite normal forms, which
#  we should implement. See e.g. the Kannan-Bachem algorithm:
#  <https://www.researchgate.net/publication/220617516_Polynomial_Algorithms_for_Computing_the_Smith_and_Hermite_Normal_Forms_of_an_Integer_Matrix>


def smith_normal_form(m):
    '''
    Return the Smith Normal Form of a matrix `m` over the ring `domain`.
    This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import smith_normal_form
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> print(smith_normal_form(m).to_Matrix())
    Matrix([[1, 0, 0], [0, 10, 0], [0, 0, 30]])

    '''
    invs = invariant_factors(m)
    smf = DomainMatrix.diag(invs, m.domain, m.shape)
    return smf


def is_smith_normal_form(m):
    '''
    Checks that the matrix is in Smith Normal Form
    '''
    domain = m.domain
    shape = m.shape
    zero = domain.zero
    m = m.to_list()

    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == j:
                continue
            if not m[i][j] == zero:
                return False

    upper = min(shape[0], shape[1])
    for i in range(1, upper):
        if m[i-1][i-1] == zero:
            if m[i][i] != zero:
                return False
        else:
            r = domain.div(m[i][i], m[i-1][i-1])[1]
            if r != zero:
                return False

    return True


def add_columns(m, i, j, a, b, c, d):
    # replace m[:, i] by a*m[:, i] + b*m[:, j]
    # and m[:, j] by c*m[:, i] + d*m[:, j]
    for k in range(len(m)):
        e = m[k][i]
        m[k][i] = a*e + b*m[k][j]
        m[k][j] = c*e + d*m[k][j]


def invariant_factors(m):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form)

    References
    ==========

    [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    '''
    domain = m.domain
    shape = m.shape
    m = m.to_list()
    return _smith_normal_decomp(m, domain, shape=shape, full=False)


def smith_normal_decomp(m):
    '''
    Return the Smith-Normal form decomposition of matrix `m`.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import smith_normal_decomp
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> a, s, t = smith_normal_decomp(m)
    >>> assert a == s * m * t
    '''
    domain = m.domain
    rows, cols = shape = m.shape
    m = m.to_list()

    invs, s, t = _smith_normal_decomp(m, domain, shape=shape, full=True)
    smf = DomainMatrix.diag(invs, domain, shape).to_dense()

    s = DomainMatrix(s, domain=domain, shape=(rows, rows))
    t = DomainMatrix(t, domain=domain, shape=(cols, cols))
    return smf, s, t


def _smith_normal_decomp(m, domain, shape, full):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form). If `full=True` then invertible matrices
    ``s, t`` such that the product ``s, m, t`` is the Smith Normal Form
    are also returned.
    '''
    if not domain.is_PID:
        msg = f"The matrix entries must be over a principal ideal domain, but got {domain}"
        raise ValueError(msg)

    rows, cols = shape
    zero = domain.zero
    one = domain.one

    def eye(n):
        return [[one if i == j else zero for i in range(n)] for j in range(n)]

    if 0 in shape:
        if full:
            return (), eye(rows), eye(cols)
        else:
            return ()

    if full:
        s = eye(rows)
        t = eye(cols)

    def add_rows(m, i, j, a, b, c, d):
        # replace m[i, :] by a*m[i, :] + b*m[j, :]
        # and m[j, :] by c*m[i, :] + d*m[j, :]
        for k in range(len(m[0])):
            e = m[i][k]
            m[i][k] = a*e + b*m[j][k]
            m[j][k] = c*e + d*m[j][k]

    def clear_column():
        # make m[1:, 0] zero by row and column operations
        pivot = m[0][0]
        for j in range(1, rows):
            if m[j][0] == zero:
                continue
            d, r = domain.div(m[j][0], pivot)
            if r == zero:
                add_rows(m, 0, j, 1, 0, -d, 1)
                if full:
                    add_rows(s, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[j][0])
                d_0 = domain.exquo(m[j][0], g)
                d_j = domain.exquo(pivot, g)
                add_rows(m, 0, j, a, b, d_0, -d_j)
                if full:
                    add_rows(s, 0, j, a, b, d_0, -d_j)
                pivot = g

    def clear_row():
        # make m[0, 1:] zero by row and column operations
        pivot = m[0][0]
        for j in range(1, cols):
            if m[0][j] == zero:
                continue
            d, r = domain.div(m[0][j], pivot)
            if r == zero:
                add_columns(m, 0, j, 1, 0, -d, 1)
                if full:
                    add_columns(t, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[0][j])
                d_0 = domain.exquo(m[0][j], g)
                d_j = domain.exquo(pivot, g)
                add_columns(m, 0, j, a, b, d_0, -d_j)
                if full:
                    add_columns(t, 0, j, a, b, d_0, -d_j)
                pivot = g

    # permute the rows and columns until m[0,0] is non-zero if possible
    ind = [i for i in range(rows) if m[i][0] != zero]
    if ind and ind[0] != zero:
        m[0], m[ind[0]] = m[ind[0]], m[0]
        if full:
            s[0], s[ind[0]] = s[ind[0]], s[0]
    else:
        ind = [j for j in range(cols) if m[0][j] != zero]
        if ind and ind[0] != zero:
            for row in m:
                row[0], row[ind[0]] = row[ind[0]], row[0]
            if full:
                for row in t:
                    row[0], row[ind[0]] = row[ind[0]], row[0]

    # make the first row and column except m[0,0] zero
    while (any(m[0][i] != zero for i in range(1,cols)) or
           any(m[i][0] != zero for i in range(1,rows))):
        clear_column()
        clear_row()

    def to_domain_matrix(m):
        return DomainMatrix(m, shape=(len(m), len(m[0])), domain=domain)

    if m[0][0] != 0:
        c = domain.canonical_unit(m[0][0])
        if domain.is_Field:
            c = 1 / m[0][0]
        if c != domain.one:
            m[0][0] *= c
            if full:
                s[0] = [elem * c for elem in s[0]]

    if 1 in shape:
        invs = ()
    else:
        lower_right = [r[1:] for r in m[1:]]
        ret = _smith_normal_decomp(lower_right, domain,
                shape=(rows - 1, cols - 1), full=full)
        if full:
            invs, s_small, t_small = ret
            s2 = [[1] + [0]*(rows-1)] + [[0] + row for row in s_small]
            t2 = [[1] + [0]*(cols-1)] + [[0] + row for row in t_small]
            s, s2, t, t2 = list(map(to_domain_matrix, [s, s2, t, t2]))
            s = s2 * s
            t = t * t2
            s = s.to_list()
            t = t.to_list()
        else:
            invs = ret

    if m[0][0]:
        result = [m[0][0]]
        result.extend(invs)
        # in case m[0] doesn't divide the invariants of the rest of the matrix
        for i in range(len(result)-1):
            a, b = result[i], result[i+1]
            if b and domain.div(b, a)[1] != zero:
                if full:
                    x, y, d = domain.gcdex(a, b)
                else:
                    d = domain.gcd(a, b)

                alpha = domain.div(a, d)[0]
                if full:
                    beta = domain.div(b, d)[0]
                    add_rows(s, i, i + 1, 1, 0, x, 1)
                    add_columns(t, i, i + 1, 1, y, 0, 1)
                    add_rows(s, i, i + 1, 1, -alpha, 0, 1)
                    add_columns(t, i, i + 1, 1, 0, -beta, 1)
                    add_rows(s, i, i + 1, 0, 1, -1, 0)

                result[i+1] = b * alpha
                result[i] = d
            else:
                break
    else:
        if full:
            if rows > 1:
                s = s[1:] + [s[0]]
            if cols > 1:
                t = [row[1:] + [row[0]] for row in t]
        result = invs + (m[0][0],)

    if full:
        return tuple(result), s, t
    else:
        return tuple(result)


def _gcdex(a, b):
    r"""
    This supports the functions that compute Hermite Normal Form.

    Explanation
    ===========

    Let x, y be the coefficients returned by the extended Euclidean
    Algorithm, so that x*a + y*b = g. In the algorithms for computing HNF,
    it is critical that x, y not only satisfy the condition of being small
    in magnitude -- namely that |x| <= |b|/g, |y| <- |a|/g -- but also that
    y == 0 when a | b.

    """
    x, y, g = ZZ.gcdex(a, b)
    if a != 0 and b % a == 0:
        y = 0
        x = -1 if a < 0 else 1
    return x, y, g


def _hermite_normal_form(A):
    r"""
    Compute the Hermite Normal Form of DomainMatrix *A* over :ref:`ZZ`.

    Parameters
    ==========

    A : :py:class:`~.DomainMatrix` over domain :ref:`ZZ`.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 2.4.5.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    # We work one row at a time, starting from the bottom row, and working our
    # way up.
    m, n = A.shape
    A = A.to_ddm().copy()
    # Our goal is to put pivot entries in the rightmost columns.
    # Invariant: Before processing each row, k should be the index of the
    # leftmost column in which we have so far put a pivot.
    k = n
    for i in range(m - 1, -1, -1):
        if k == 0:
            # This case can arise when n < m and we've already found n pivots.
            # We don't need to consider any more rows, because this is already
            # the maximum possible number of pivots.
            break
        k -= 1
        # k now points to the column in which we want to put a pivot.
        # We want zeros in all entries to the left of the pivot column.
        for j in range(k - 1, -1, -1):
            if A[i][j] != 0:
                # Replace cols j, k by lin combs of these cols such that, in row i,
                # col j has 0, while col k has the gcd of their row i entries. Note
                # that this ensures a nonzero entry in col k.
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = A[i][k] // d, A[i][j] // d
                add_columns(A, k, j, u, v, -s, r)
        b = A[i][k]
        # Do not want the pivot entry to be negative.
        if b < 0:
            add_columns(A, k, k, -1, 0, -1, 0)
            b = -b
        # The pivot entry will be 0 iff the row was 0 from the pivot col all the
        # way to the left. In this case, we are still working on the same pivot
        # col for the next row. Therefore:
        if b == 0:
            k += 1
        # If the pivot entry is nonzero, then we want to reduce all entries to its
        # right in the sense of the division algorithm, i.e. make them all remainders
        # w.r.t. the pivot as divisor.
        else:
            for j in range(k + 1, n):
                q = A[i][j] // b
                add_columns(A, j, k, 1, -q, 0, 1)
    # Finally, the HNF consists of those columns of A in which we succeeded in making
    # a nonzero pivot.
    return DomainMatrix.from_rep(A.to_dfm_or_ddm())[:, k:]


def _hermite_normal_form_modulo_D(A, D):
    r"""
    Perform the mod *D* Hermite Normal Form reduction algorithm on
    :py:class:`~.DomainMatrix` *A*.

    Explanation
    ===========

    If *A* is an $m \times n$ matrix of rank $m$, having Hermite Normal Form
    $W$, and if *D* is any positive integer known in advance to be a multiple
    of $\det(W)$, then the HNF of *A* can be computed by an algorithm that
    works mod *D* in order to prevent coefficient explosion.

    Parameters
    ==========

    A : :py:class:`~.DomainMatrix` over :ref:`ZZ`
        $m \times n$ matrix, having rank $m$.
    D : :ref:`ZZ`
        Positive integer, known to be a multiple of the determinant of the
        HNF of *A*.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`, or
        if *D* is given but is not in :ref:`ZZ`.

    DMShapeError
        If the matrix has more rows than columns.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 2.4.8.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    if not ZZ.of_type(D) or D < 1:
        raise DMDomainError('Modulus D must be positive element of domain ZZ.')

    def add_columns_mod_R(m, R, i, j, a, b, c, d):
        # replace m[:, i] by (a*m[:, i] + b*m[:, j]) % R
        # and m[:, j] by (c*m[:, i] + d*m[:, j]) % R
        for k in range(len(m)):
            e = m[k][i]
            m[k][i] = symmetric_residue((a * e + b * m[k][j]) % R, R)
            m[k][j] = symmetric_residue((c * e + d * m[k][j]) % R, R)

    W = defaultdict(dict)

    m, n = A.shape
    if n < m:
        raise DMShapeError('Matrix must have at least as many columns as rows.')
    A = A.to_list()
    k = n
    R = D
    for i in range(m - 1, -1, -1):
        k -= 1
        for j in range(k - 1, -1, -1):
            if A[i][j] != 0:
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = A[i][k] // d, A[i][j] // d
                add_columns_mod_R(A, R, k, j, u, v, -s, r)
        b = A[i][k]
        if b == 0:
            A[i][k] = b = R
        u, v, d = _gcdex(b, R)
        for ii in range(m):
            W[ii][i] = u*A[ii][k] % R
        if W[i][i] == 0:
            W[i][i] = R
        for j in range(i + 1, m):
            q = W[i][j] // W[i][i]
            add_columns(W, j, i, 1, -q, 0, 1)
        R //= d
    return DomainMatrix(W, (m, m), ZZ).to_dense()


def hermite_normal_form(A, *, D=None, check_rank=False):
    r"""
    Compute the Hermite Normal Form of :py:class:`~.DomainMatrix` *A* over
    :ref:`ZZ`.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import hermite_normal_form
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> print(hermite_normal_form(m).to_Matrix())
    Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])

    Parameters
    ==========

    A : $m \times n$ ``DomainMatrix`` over :ref:`ZZ`.

    D : :ref:`ZZ`, optional
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

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`, or
        if *D* is given but is not in :ref:`ZZ`.

    DMShapeError
        If the mod *D* algorithm is used but the matrix has more rows than
        columns.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithms 2.4.5 and 2.4.8.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    if D is not None and (not check_rank or A.convert_to(QQ).rank() == A.shape[0]):
        return _hermite_normal_form_modulo_D(A, D)
    else:
        return _hermite_normal_form(A)
