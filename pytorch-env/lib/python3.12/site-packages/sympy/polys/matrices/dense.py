"""

Module for the ddm_* routines for operating on a matrix in list of lists
matrix representation.

These routines are used internally by the DDM class which also provides a
friendlier interface for them. The idea here is to implement core matrix
routines in a way that can be applied to any simple list representation
without the need to use any particular matrix class. For example we can
compute the RREF of a matrix like:

    >>> from sympy.polys.matrices.dense import ddm_irref
    >>> M = [[1, 2, 3], [4, 5, 6]]
    >>> pivots = ddm_irref(M)
    >>> M
    [[1.0, 0.0, -1.0], [0, 1.0, 2.0]]

These are lower-level routines that work mostly in place.The routines at this
level should not need to know what the domain of the elements is but should
ideally document what operations they will use and what functions they need to
be provided with.

The next-level up is the DDM class which uses these routines but wraps them up
with an interface that handles copying etc and keeps track of the Domain of
the elements of the matrix:

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.matrices.ddm import DDM
    >>> M = DDM([[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]], (2, 3), QQ)
    >>> M
    [[1, 2, 3], [4, 5, 6]]
    >>> Mrref, pivots = M.rref()
    >>> Mrref
    [[1, 0, -1], [0, 1, 2]]

"""
from __future__ import annotations
from operator import mul
from .exceptions import (
    DMShapeError,
    DMDomainError,
    DMNonInvertibleMatrixError,
    DMNonSquareMatrixError,
)
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement


#: Type variable for the elements of the matrix
T = TypeVar('T')

#: Type variable for the elements of the matrix that are in a ring
R = TypeVar('R', bound=RingElement)


def ddm_transpose(matrix: Sequence[Sequence[T]]) -> list[list[T]]:
    """matrix transpose"""
    return list(map(list, zip(*matrix)))


def ddm_iadd(a: list[list[R]], b: Sequence[Sequence[R]]) -> None:
    """a += b"""
    for ai, bi in zip(a, b):
        for j, bij in enumerate(bi):
            ai[j] += bij


def ddm_isub(a: list[list[R]], b: Sequence[Sequence[R]]) -> None:
    """a -= b"""
    for ai, bi in zip(a, b):
        for j, bij in enumerate(bi):
            ai[j] -= bij


def ddm_ineg(a: list[list[R]]) -> None:
    """a <-- -a"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = -aij


def ddm_imul(a: list[list[R]], b: R) -> None:
    """a <-- a*b"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = aij * b


def ddm_irmul(a: list[list[R]], b: R) -> None:
    """a <-- b*a"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = b * aij


def ddm_imatmul(
    a: list[list[R]], b: Sequence[Sequence[R]], c: Sequence[Sequence[R]]
) -> None:
    """a += b @ c"""
    cT = list(zip(*c))

    for bi, ai in zip(b, a):
        for j, cTj in enumerate(cT):
            ai[j] = sum(map(mul, bi, cTj), ai[j])


def ddm_irref(a, _partial_pivot=False):
    """In-place reduced row echelon form of a matrix.

    Compute the reduced row echelon form of $a$. Modifies $a$ in place and
    returns a list of the pivot columns.

    Uses naive Gauss-Jordan elimination in the ground domain which must be a
    field.

    This routine is only really suitable for use with simple field domains like
    :ref:`GF(p)`, :ref:`QQ` and :ref:`QQ(a)` although even for :ref:`QQ` with
    larger matrices it is possibly more efficient to use fraction free
    approaches.

    This method is not suitable for use with rational function fields
    (:ref:`K(x)`) because the elements will blowup leading to costly gcd
    operations. In this case clearing denominators and using fraction free
    approaches is likely to be more efficient.

    For inexact numeric domains like :ref:`RR` and :ref:`CC` pass
    ``_partial_pivot=True`` to use partial pivoting to control rounding errors.

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_irref
    >>> from sympy import QQ
    >>> M = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]]
    >>> pivots = ddm_irref(M)
    >>> M
    [[1, 0, -1], [0, 1, 2]]
    >>> pivots
    [0, 1]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        Higher level interface to this routine.
    ddm_irref_den
        The fraction free version of this routine.
    sdm_irref
        A sparse version of this routine.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form
    """
    # We compute aij**-1 below and then use multiplication instead of division
    # in the innermost loop. The domain here is a field so either operation is
    # defined. There are significant performance differences for some domains
    # though. In the case of e.g. QQ or QQ(x) inversion is free but
    # multiplication and division have the same cost so it makes no difference.
    # In cases like GF(p), QQ<sqrt(2)>, RR or CC though multiplication is
    # faster than division so reusing a precomputed inverse for many
    # multiplications can be a lot faster. The biggest win is QQ<a> when
    # deg(minpoly(a)) is large.
    #
    # With domains like QQ(x) this can perform badly for other reasons.
    # Typically the initial matrix has simple denominators and the
    # fraction-free approach with exquo (ddm_irref_den) will preserve that
    # property throughout. The method here causes denominator blowup leading to
    # expensive gcd reductions in the intermediate expressions. With many
    # generators like QQ(x,y,z,...) this is extremely bad.
    #
    # TODO: Use a nontrivial pivoting strategy to control intermediate
    # expression growth. Rearranging rows and/or columns could defer the most
    # complicated elements until the end. If the first pivot is a
    # complicated/large element then the first round of reduction will
    # immediately introduce expression blowup across the whole matrix.

    # a is (m x n)
    m = len(a)
    if not m:
        return []
    n = len(a[0])

    i = 0
    pivots = []

    for j in range(n):
        # Proper pivoting should be used for all domains for performance
        # reasons but it is only strictly needed for RR and CC (and possibly
        # other domains like RR(x)). This path is used by DDM.rref() if the
        # domain is RR or CC. It uses partial (row) pivoting based on the
        # absolute value of the pivot candidates.
        if _partial_pivot:
            ip = max(range(i, m), key=lambda ip: abs(a[ip][j]))
            a[i], a[ip] = a[ip], a[i]

        # pivot
        aij = a[i][j]

        # zero-pivot
        if not aij:
            for ip in range(i+1, m):
                aij = a[ip][j]
                # row-swap
                if aij:
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # next column
                continue

        # normalise row
        ai = a[i]
        aijinv = aij**-1
        for l in range(j, n):
            ai[l] *= aijinv # ai[j] = one

        # eliminate above and below to the right
        for k, ak in enumerate(a):
            if k == i or not ak[j]:
                continue
            akj = ak[j]
            ak[j] -= akj # ak[j] = zero
            for l in range(j+1, n):
                ak[l] -= akj * ai[l]

        # next row
        pivots.append(j)
        i += 1

        # no more rows?
        if i >= m:
            break

    return pivots


def ddm_irref_den(a, K):
    """a  <--  rref(a); return (den, pivots)

    Compute the fraction-free reduced row echelon form (RREF) of $a$. Modifies
    $a$ in place and returns a tuple containing the denominator of the RREF and
    a list of the pivot columns.

    Explanation
    ===========

    The algorithm used is the fraction-free version of Gauss-Jordan elimination
    described as FFGJ in [1]_. Here it is modified to handle zero or missing
    pivots and to avoid redundant arithmetic.

    The domain $K$ must support exact division (``K.exquo``) but does not need
    to be a field. This method is suitable for most exact rings and fields like
    :ref:`ZZ`, :ref:`QQ` and :ref:`QQ(a)`. In the case of :ref:`QQ` or
    :ref:`K(x)` it might be more efficient to clear denominators and use
    :ref:`ZZ` or :ref:`K[x]` instead.

    For inexact domains like :ref:`RR` and :ref:`CC` use ``ddm_irref`` instead.

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_irref_den
    >>> from sympy import ZZ, Matrix
    >>> M = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)]]
    >>> den, pivots = ddm_irref_den(M, ZZ)
    >>> M
    [[-3, 0, 3], [0, -3, -6]]
    >>> den
    -3
    >>> pivots
    [0, 1]
    >>> Matrix(M).rref()[0]
    Matrix([
    [1, 0, -1],
    [0, 1,  2]])

    See Also
    ========

    ddm_irref
        A version of this routine that uses field division.
    sdm_irref
        A sparse version of :func:`ddm_irref`.
    sdm_rref_den
        A sparse version of :func:`ddm_irref_den`.
    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        Higher level interface.

    References
    ==========

    .. [1] Fraction-free algorithms for linear and polynomial equations.
        George C. Nakos , Peter R. Turner , Robert M. Williams.
        https://dl.acm.org/doi/10.1145/271130.271133
    """
    #
    # A simpler presentation of this algorithm is given in [1]:
    #
    # Given an n x n matrix A and n x 1 matrix b:
    #
    #   for i in range(n):
    #       if i != 0:
    #           d = a[i-1][i-1]
    #       for j in range(n):
    #           if j == i:
    #               continue
    #           b[j] = a[i][i]*b[j] - a[j][i]*b[i]
    #           for k in range(n):
    #               a[j][k] = a[i][i]*a[j][k] - a[j][i]*a[i][k]
    #               if i != 0:
    #                   a[j][k] /= d
    #
    # Our version here is a bit more complicated because:
    #
    #  1. We use row-swaps to avoid zero pivots.
    #  2. We allow for some columns to be missing pivots.
    #  3. We avoid a lot of redundant arithmetic.
    #
    # TODO: Use a non-trivial pivoting strategy. Even just row swapping makes a
    # big difference to performance if e.g. the upper-left entry of the matrix
    # is a huge polynomial.

    # a is (m x n)
    m = len(a)
    if not m:
        return K.one, []
    n = len(a[0])

    d = None
    pivots = []
    no_pivots = []

    # i, j will be the row and column indices of the current pivot
    i = 0
    for j in range(n):
        # next pivot?
        aij = a[i][j]

        # swap rows if zero
        if not aij:
            for ip in range(i+1, m):
                aij = a[ip][j]
                # row-swap
                if aij:
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # go to next column
                no_pivots.append(j)
                continue

        # Now aij is the pivot and i,j are the row and column. We need to clear
        # the column above and below but we also need to keep track of the
        # denominator of the RREF which means also multiplying everything above
        # and to the left by the current pivot aij and dividing by d (which we
        # multiplied everything by in the previous iteration so this is an
        # exact division).
        #
        # First handle the upper left corner which is usually already diagonal
        # with all diagonal entries equal to the current denominator but there
        # can be other non-zero entries in any column that has no pivot.

        # Update previous pivots in the matrix
        if pivots:
            pivot_val = aij * a[0][pivots[0]]
            # Divide out the common factor
            if d is not None:
                pivot_val = K.exquo(pivot_val, d)

            # Could defer this until the end but it is pretty cheap and
            # helps when debugging.
            for ip, jp in enumerate(pivots):
                a[ip][jp] = pivot_val

        # Update columns without pivots
        for jnp in no_pivots:
            for ip in range(i):
                aijp = a[ip][jnp]
                if aijp:
                    aijp *= aij
                    if d is not None:
                        aijp = K.exquo(aijp, d)
                    a[ip][jnp] = aijp

        # Eliminate above, below and to the right as in ordinary division free
        # Gauss-Jordan elmination except also dividing out d from every entry.

        for jp, aj in enumerate(a):

            # Skip the current row
            if jp == i:
                continue

            # Eliminate to the right in all rows
            for kp in range(j+1, n):
                ajk = aij * aj[kp] - aj[j] * a[i][kp]
                if d is not None:
                    ajk = K.exquo(ajk, d)
                aj[kp] = ajk

            # Set to zero above and below the pivot
            aj[j] = K.zero

        # next row
        pivots.append(j)
        i += 1

        # no more rows left?
        if i >= m:
            break

        if not K.is_one(aij):
            d = aij
        else:
            d = None

    if not pivots:
        denom = K.one
    else:
        denom = a[0][pivots[0]]

    return denom, pivots


def ddm_idet(a, K):
    """a  <--  echelon(a); return det

    Explanation
    ===========

    Compute the determinant of $a$ using the Bareiss fraction-free algorithm.
    The matrix $a$ is modified in place. Its diagonal elements are the
    determinants of the leading principal minors. The determinant of $a$ is
    returned.

    The domain $K$ must support exact division (``K.exquo``). This method is
    suitable for most exact rings and fields like :ref:`ZZ`, :ref:`QQ` and
    :ref:`QQ(a)` but not for inexact domains like :ref:`RR` and :ref:`CC`.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices.ddm import ddm_idet
    >>> a = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]]
    >>> a
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> ddm_idet(a, ZZ)
    0
    >>> a
    [[1, 2, 3], [4, -3, -6], [7, -6, 0]]
    >>> [a[i][i] for i in range(len(a))]
    [1, -3, 0]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.det

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bareiss_algorithm
    .. [2] https://www.math.usm.edu/perry/Research/Thesis_DRL.pdf
    """
    # Bareiss algorithm
    # https://www.math.usm.edu/perry/Research/Thesis_DRL.pdf

    # a is (m x n)
    m = len(a)
    if not m:
        return K.one
    n = len(a[0])

    exquo = K.exquo
    # uf keeps track of the sign change from row swaps
    uf = K.one

    for k in range(n-1):
        if not a[k][k]:
            for i in range(k+1, n):
                if a[i][k]:
                    a[k], a[i] = a[i], a[k]
                    uf = -uf
                    break
            else:
                return K.zero

        akkm1 = a[k-1][k-1] if k else K.one

        for i in range(k+1, n):
            for j in range(k+1, n):
                a[i][j] = exquo(a[i][j]*a[k][k] - a[i][k]*a[k][j], akkm1)

    return uf * a[-1][-1]


def ddm_iinv(ainv, a, K):
    """ainv  <--  inv(a)

    Compute the inverse of a matrix $a$ over a field $K$ using Gauss-Jordan
    elimination. The result is stored in $ainv$.

    Uses division in the ground domain which should be an exact field.

    Examples
    ========

    >>> from sympy.polys.matrices.ddm import ddm_iinv, ddm_imatmul
    >>> from sympy import QQ
    >>> a = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> ainv = [[None, None], [None, None]]
    >>> ddm_iinv(ainv, a, QQ)
    >>> ainv
    [[-2, 1], [3/2, -1/2]]
    >>> result = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    >>> ddm_imatmul(result, a, ainv)
    >>> result
    [[1, 0], [0, 1]]

    See Also
    ========

    ddm_irref: the underlying routine.
    """
    if not K.is_Field:
        raise DMDomainError('Not a field')

    # a is (m x n)
    m = len(a)
    if not m:
        return
    n = len(a[0])
    if m != n:
        raise DMNonSquareMatrixError

    eye = [[K.one if i==j else K.zero for j in range(n)] for i in range(n)]
    Aaug = [row + eyerow for row, eyerow in zip(a, eye)]
    pivots = ddm_irref(Aaug)
    if pivots != list(range(n)):
        raise DMNonInvertibleMatrixError('Matrix det == 0; not invertible.')
    ainv[:] = [row[n:] for row in Aaug]


def ddm_ilu_split(L, U, K):
    """L, U  <--  LU(U)

    Compute the LU decomposition of a matrix $L$ in place and store the lower
    and upper triangular matrices in $L$ and $U$, respectively. Returns a list
    of row swaps that were performed.

    Uses division in the ground domain which should be an exact field.

    Examples
    ========

    >>> from sympy.polys.matrices.ddm import ddm_ilu_split
    >>> from sympy import QQ
    >>> L = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    >>> U = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> swaps = ddm_ilu_split(L, U, QQ)
    >>> swaps
    []
    >>> L
    [[0, 0], [3, 0]]
    >>> U
    [[1, 2], [0, -2]]

    See Also
    ========

    ddm_ilu
    ddm_ilu_solve
    """
    m = len(U)
    if not m:
        return []
    n = len(U[0])

    swaps = ddm_ilu(U)

    zeros = [K.zero] * min(m, n)
    for i in range(1, m):
        j = min(i, n)
        L[i][:j] = U[i][:j]
        U[i][:j] = zeros[:j]

    return swaps


def ddm_ilu(a):
    """a  <--  LU(a)

    Computes the LU decomposition of a matrix in place. Returns a list of
    row swaps that were performed.

    Uses division in the ground domain which should be an exact field.

    This is only suitable for domains like :ref:`GF(p)`, :ref:`QQ`, :ref:`QQ_I`
    and :ref:`QQ(a)`. With a rational function field like :ref:`K(x)` it is
    better to clear denominators and use division-free algorithms. Pivoting is
    used to avoid exact zeros but not for floating point accuracy so :ref:`RR`
    and :ref:`CC` are not suitable (use :func:`ddm_irref` instead).

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_ilu
    >>> from sympy import QQ
    >>> a = [[QQ(1, 2), QQ(1, 3)], [QQ(1, 4), QQ(1, 5)]]
    >>> swaps = ddm_ilu(a)
    >>> swaps
    []
    >>> a
    [[1/2, 1/3], [1/2, 1/30]]

    The same example using ``Matrix``:

    >>> from sympy import Matrix, S
    >>> M = Matrix([[S(1)/2, S(1)/3], [S(1)/4, S(1)/5]])
    >>> L, U, swaps = M.LUdecomposition()
    >>> L
    Matrix([
    [  1, 0],
    [1/2, 1]])
    >>> U
    Matrix([
    [1/2,  1/3],
    [  0, 1/30]])
    >>> swaps
    []

    See Also
    ========

    ddm_irref
    ddm_ilu_solve
    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    """
    m = len(a)
    if not m:
        return []
    n = len(a[0])

    swaps = []

    for i in range(min(m, n)):
        if not a[i][i]:
            for ip in range(i+1, m):
                if a[ip][i]:
                    swaps.append((i, ip))
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # M = Matrix([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 2]])
                continue
        for j in range(i+1, m):
            l_ji = a[j][i] / a[i][i]
            a[j][i] = l_ji
            for k in range(i+1, n):
                a[j][k] -= l_ji * a[i][k]

    return swaps


def ddm_ilu_solve(x, L, U, swaps, b):
    """x  <--  solve(L*U*x = swaps(b))

    Solve a linear system, $A*x = b$, given an LU factorization of $A$.

    Uses division in the ground domain which must be a field.

    Modifies $x$ in place.

    Examples
    ========

    Compute the LU decomposition of $A$ (in place):

    >>> from sympy import QQ
    >>> from sympy.polys.matrices.dense import ddm_ilu, ddm_ilu_solve
    >>> A = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> swaps = ddm_ilu(A)
    >>> A
    [[1, 2], [3, -2]]
    >>> L = U = A

    Solve the linear system:

    >>> b = [[QQ(5)], [QQ(6)]]
    >>> x = [[None], [None]]
    >>> ddm_ilu_solve(x, L, U, swaps, b)
    >>> x
    [[-4], [9/2]]

    See Also
    ========

    ddm_ilu
        Compute the LU decomposition of a matrix in place.
    ddm_ilu_split
        Compute the LU decomposition of a matrix and separate $L$ and $U$.
    sympy.polys.matrices.domainmatrix.DomainMatrix.lu_solve
        Higher level interface to this function.
    """
    m = len(U)
    if not m:
        return
    n = len(U[0])

    m2 = len(b)
    if not m2:
        raise DMShapeError("Shape mismtch")
    o = len(b[0])

    if m != m2:
        raise DMShapeError("Shape mismtch")
    if m < n:
        raise NotImplementedError("Underdetermined")

    if swaps:
        b = [row[:] for row in b]
        for i1, i2 in swaps:
            b[i1], b[i2] = b[i2], b[i1]

    # solve Ly = b
    y = [[None] * o for _ in range(m)]
    for k in range(o):
        for i in range(m):
            rhs = b[i][k]
            for j in range(i):
                rhs -= L[i][j] * y[j][k]
            y[i][k] = rhs

    if m > n:
        for i in range(n, m):
            for j in range(o):
                if y[i][j]:
                    raise DMNonInvertibleMatrixError

    # Solve Ux = y
    for k in range(o):
        for i in reversed(range(n)):
            if not U[i][i]:
                raise DMNonInvertibleMatrixError
            rhs = y[i][k]
            for j in range(i+1, n):
                rhs -= U[i][j] * x[j][k]
            x[i][k] = rhs / U[i][i]


def ddm_berk(M, K):
    """
    Berkowitz algorithm for computing the characteristic polynomial.

    Explanation
    ===========

    The Berkowitz algorithm is a division-free algorithm for computing the
    characteristic polynomial of a matrix over any commutative ring using only
    arithmetic in the coefficient ring.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices.dense import ddm_berk
    >>> from sympy.polys.domains import ZZ
    >>> M = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    >>> ddm_berk(M, ZZ)
    [[1], [-5], [-2]]
    >>> Matrix(M).charpoly()
    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
        The high-level interface to this function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm
    """
    m = len(M)
    if not m:
        return [[K.one]]
    n = len(M[0])

    if m != n:
        raise DMShapeError("Not square")

    if n == 1:
        return [[K.one], [-M[0][0]]]

    a = M[0][0]
    R = [M[0][1:]]
    C = [[row[0]] for row in M[1:]]
    A = [row[1:] for row in M[1:]]

    q = ddm_berk(A, K)

    T = [[K.zero] * n for _ in range(n+1)]
    for i in range(n):
        T[i][i] = K.one
        T[i+1][i] = -a
    for i in range(2, n+1):
        if i == 2:
            AnC = C
        else:
            C = AnC
            AnC = [[K.zero] for row in C]
            ddm_imatmul(AnC, A, C)
        RAnC = [[K.zero]]
        ddm_imatmul(RAnC, R, AnC)
        for j in range(0, n+1-i):
            T[i+j][j] = -RAnC[0][0]

    qout = [[K.zero] for _ in range(n+1)]
    ddm_imatmul(qout, T, q)
    return qout
