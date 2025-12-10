from types import FunctionType

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains import ZZ, QQ

from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot


def _row_reduce_list(mat, rows, cols, one, iszerofunc, simpfunc,
                normalize_last=True, normalize=True, zero_above=True):
    """Row reduce a flat list representation of a matrix and return a tuple
    (rref_matrix, pivot_cols, swaps) where ``rref_matrix`` is a flat list,
    ``pivot_cols`` are the pivot columns and ``swaps`` are any row swaps that
    were used in the process of row reduction.

    Parameters
    ==========

    mat : list
        list of matrix elements, must be ``rows`` * ``cols`` in length

    rows, cols : integer
        number of rows and columns in flat list representation

    one : SymPy object
        represents the value one, from ``Matrix.one``

    iszerofunc : determines if an entry can be used as a pivot

    simpfunc : used to simplify elements and test if they are
        zero if ``iszerofunc`` returns `None`

    normalize_last : indicates where all row reduction should
        happen in a fraction-free manner and then the rows are
        normalized (so that the pivots are 1), or whether
        rows should be normalized along the way (like the naive
        row reduction algorithm)

    normalize : whether pivot rows should be normalized so that
        the pivot value is 1

    zero_above : whether entries above the pivot should be zeroed.
        If ``zero_above=False``, an echelon matrix will be returned.
    """

    def get_col(i):
        return mat[i::cols]

    def row_swap(i, j):
        mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
            mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    def cross_cancel(a, i, b, j):
        """Does the row op row[i] = a*row[i] - b*row[j]"""
        q = (j - i)*cols
        for p in range(i*cols, (i + 1)*cols):
            mat[p] = isimp(a*mat[p] - b*mat[p + q])

    isimp = _get_intermediate_simp(_dotprodsimp)
    piv_row, piv_col = 0, 0
    pivot_cols = []
    swaps = []

    # use a fraction free method to zero above and below each pivot
    while piv_col < cols and piv_row < rows:
        pivot_offset, pivot_val, \
        assumed_nonzero, newly_determined = _find_reasonable_pivot(
                get_col(piv_col)[piv_row:], iszerofunc, simpfunc)

        # _find_reasonable_pivot may have simplified some things
        # in the process.  Let's not let them go to waste
        for (offset, val) in newly_determined:
            offset += piv_row
            mat[offset*cols + piv_col] = val

        if pivot_offset is None:
            piv_col += 1
            continue

        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            row_swap(piv_row, pivot_offset + piv_row)
            swaps.append((piv_row, pivot_offset + piv_row))

        # if we aren't normalizing last, we normalize
        # before we zero the other rows
        if normalize_last is False:
            i, j = piv_row, piv_col
            mat[i*cols + j] = one
            for p in range(i*cols + j + 1, (i + 1)*cols):
                mat[p] = isimp(mat[p] / pivot_val)
            # after normalizing, the pivot value is 1
            pivot_val = one

        # zero above and below the pivot
        for row in range(rows):
            # don't zero our current row
            if row == piv_row:
                continue
            # don't zero above the pivot unless we're told.
            if zero_above is False and row < piv_row:
                continue
            # if we're already a zero, don't do anything
            val = mat[row*cols + piv_col]
            if iszerofunc(val):
                continue

            cross_cancel(pivot_val, row, val, piv_row)
        piv_row += 1

    # normalize each row
    if normalize_last is True and normalize is True:
        for piv_i, piv_j in enumerate(pivot_cols):
            pivot_val = mat[piv_i*cols + piv_j]
            mat[piv_i*cols + piv_j] = one
            for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols):
                mat[p] = isimp(mat[p] / pivot_val)

    return mat, tuple(pivot_cols), tuple(swaps)


# This functions is a candidate for caching if it gets implemented for matrices.
def _row_reduce(M, iszerofunc, simpfunc, normalize_last=True,
                normalize=True, zero_above=True):

    mat, pivot_cols, swaps = _row_reduce_list(list(M), M.rows, M.cols, M.one,
            iszerofunc, simpfunc, normalize_last=normalize_last,
            normalize=normalize, zero_above=zero_above)

    return M._new(M.rows, M.cols, mat), pivot_cols, swaps


def _is_echelon(M, iszerofunc=_iszero):
    """Returns `True` if the matrix is in echelon form. That is, all rows of
    zeros are at the bottom, and below each leading non-zero in a row are
    exclusively zeros."""

    if M.rows <= 0 or M.cols <= 0:
        return True

    zeros_below = all(iszerofunc(t) for t in M[1:, 0])

    if iszerofunc(M[0, 0]):
        return zeros_below and _is_echelon(M[:, 1:], iszerofunc)

    return zeros_below and _is_echelon(M[1:, 1:], iszerofunc)


def _echelon_form(M, iszerofunc=_iszero, simplify=False, with_pivots=False):
    """Returns a matrix row-equivalent to ``M`` that is in echelon form. Note
    that echelon form of a matrix is *not* unique, however, properties like the
    row space and the null space are preserved.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.echelon_form()
    Matrix([
    [1,  2],
    [0, -2]])
    """

    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify

    mat, pivots, _ = _row_reduce(M, iszerofunc, simpfunc,
            normalize_last=True, normalize=False, zero_above=False)

    if with_pivots:
        return mat, pivots

    return mat


# This functions is a candidate for caching if it gets implemented for matrices.
def _rank(M, iszerofunc=_iszero, simplify=False):
    """Returns the rank of a matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rank()
    2
    >>> n = Matrix(3, 3, range(1, 10))
    >>> n.rank()
    2
    """

    def _permute_complexity_right(M, iszerofunc):
        """Permute columns with complicated elements as
        far right as they can go.  Since the ``sympy`` row reduction
        algorithms start on the left, having complexity right-shifted
        speeds things up.

        Returns a tuple (mat, perm) where perm is a permutation
        of the columns to perform to shift the complex columns right, and mat
        is the permuted matrix."""

        def complexity(i):
            # the complexity of a column will be judged by how many
            # element's zero-ness cannot be determined
            return sum(1 if iszerofunc(e) is None else 0 for e in M[:, i])

        complex = [(complexity(i), i) for i in range(M.cols)]
        perm    = [j for (i, j) in sorted(complex)]

        return (M.permute(perm, orientation='cols'), perm)

    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify

    # for small matrices, we compute the rank explicitly
    # if is_zero on elements doesn't answer the question
    # for small matrices, we fall back to the full routine.
    if M.rows <= 0 or M.cols <= 0:
        return 0

    if M.rows <= 1 or M.cols <= 1:
        zeros = [iszerofunc(x) for x in M]

        if False in zeros:
            return 1

    if M.rows == 2 and M.cols == 2:
        zeros = [iszerofunc(x) for x in M]

        if False not in zeros and None not in zeros:
            return 0

        d = M.det()

        if iszerofunc(d) and False in zeros:
            return 1
        if iszerofunc(d) is False:
            return 2

    mat, _       = _permute_complexity_right(M, iszerofunc=iszerofunc)
    _, pivots, _ = _row_reduce(mat, iszerofunc, simpfunc, normalize_last=True,
            normalize=False, zero_above=False)

    return len(pivots)


def _to_DM_ZZ_QQ(M):
    # We have to test for _rep here because there are tests that otherwise fail
    # with e.g. "AttributeError: 'SubspaceOnlyMatrix' object has no attribute
    # '_rep'." There is almost certainly no value in such tests. The
    # presumption seems to be that someone could create a new class by
    # inheriting some of the Matrix classes and not the full set that is used
    # by the standard Matrix class but if anyone tried that it would fail in
    # many ways.
    if not hasattr(M, '_rep'):
        return None

    rep = M._rep
    K = rep.domain

    if K.is_ZZ:
        return rep
    elif K.is_QQ:
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep
    else:
        if not all(e.is_Rational for e in M):
            return None
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep.convert_to(QQ)


def _rref_dm(dM):
    """Compute the reduced row echelon form of a DomainMatrix."""
    K = dM.domain

    if K.is_ZZ:
        dM_rref, den, pivots = dM.rref_den(keep_domain=False)
        dM_rref = dM_rref.to_field() / den
    elif K.is_QQ:
        dM_rref, pivots = dM.rref()
    else:
        assert False  # pragma: no cover

    M_rref = dM_rref.to_Matrix()

    return M_rref, pivots


def _rref(M, iszerofunc=_iszero, simplify=False, pivots=True,
        normalize_last=True):
    """Return reduced row-echelon form of matrix and indices
    of pivot vars.

    Parameters
    ==========

    iszerofunc : Function
        A function used for detecting whether an element can
        act as a pivot.  ``lambda x: x.is_zero`` is used by default.

    simplify : Function
        A function used to simplify elements when looking for a pivot.
        By default SymPy's ``simplify`` is used.

    pivots : True or False
        If ``True``, a tuple containing the row-reduced matrix and a tuple
        of pivot columns is returned.  If ``False`` just the row-reduced
        matrix is returned.

    normalize_last : True or False
        If ``True``, no pivots are normalized to `1` until after all
        entries above and below each pivot are zeroed.  This means the row
        reduction algorithm is fraction free until the very last step.
        If ``False``, the naive row reduction procedure is used where
        each pivot is normalized to be `1` before row operations are
        used to zero above and below the pivot.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rref()
    (Matrix([
    [1, 0],
    [0, 1]]), (0, 1))
    >>> rref_matrix, rref_pivots = m.rref()
    >>> rref_matrix
    Matrix([
    [1, 0],
    [0, 1]])
    >>> rref_pivots
    (0, 1)

    ``iszerofunc`` can correct rounding errors in matrices with float
    values. In the following example, calling ``rref()`` leads to
    floating point errors, incorrectly row reducing the matrix.
    ``iszerofunc= lambda x: abs(x) < 1e-9`` sets sufficiently small numbers
    to zero, avoiding this error.

    >>> m = Matrix([[0.9, -0.1, -0.2, 0], [-0.8, 0.9, -0.4, 0], [-0.1, -0.8, 0.6, 0]])
    >>> m.rref()
    (Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]]), (0, 1, 2))
    >>> m.rref(iszerofunc=lambda x:abs(x)<1e-9)
    (Matrix([
    [1, 0, -0.301369863013699, 0],
    [0, 1, -0.712328767123288, 0],
    [0, 0,         0,          0]]), (0, 1))

    Notes
    =====

    The default value of ``normalize_last=True`` can provide significant
    speedup to row reduction, especially on matrices with symbols.  However,
    if you depend on the form row reduction algorithm leaves entries
    of the matrix, set ``normalize_last=False``
    """
    # Try to use DomainMatrix for ZZ or QQ
    dM = _to_DM_ZZ_QQ(M)

    if dM is not None:
        # Use DomainMatrix for ZZ or QQ
        mat, pivot_cols = _rref_dm(dM)
    else:
        # Use the generic Matrix routine.
        if isinstance(simplify, FunctionType):
            simpfunc = simplify
        else:
            simpfunc = _simplify

        mat, pivot_cols, _ = _row_reduce(M, iszerofunc, simpfunc,
                normalize_last, normalize=True, zero_above=True)

    if pivots:
        return mat, pivot_cols
    else:
        return mat
