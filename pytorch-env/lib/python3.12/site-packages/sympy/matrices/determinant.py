from types import FunctionType

from sympy.core.cache import cacheit
from sympy.core.numbers import Float, Integer
from sympy.core.singleton import S
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.mul import Mul
from sympy.polys import PurePoly, cancel
from sympy.functions.combinatorial.numbers import nC
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.ddm import DDM

from .exceptions import NonSquareMatrixError
from .utilities import (
    _get_intermediate_simp, _get_intermediate_simp_bool,
    _iszero, _is_zero_after_expand_mul, _dotprodsimp, _simplify)


def _find_reasonable_pivot(col, iszerofunc=_iszero, simpfunc=_simplify):
    """ Find the lowest index of an item in ``col`` that is
    suitable for a pivot.  If ``col`` consists only of
    Floats, the pivot with the largest norm is returned.
    Otherwise, the first element where ``iszerofunc`` returns
    False is used.  If ``iszerofunc`` does not return false,
    items are simplified and retested until a suitable
    pivot is found.

    Returns a 4-tuple
        (pivot_offset, pivot_val, assumed_nonzero, newly_determined)
    where pivot_offset is the index of the pivot, pivot_val is
    the (possibly simplified) value of the pivot, assumed_nonzero
    is True if an assumption that the pivot was non-zero
    was made without being proved, and newly_determined are
    elements that were simplified during the process of pivot
    finding."""

    newly_determined = []
    col = list(col)
    # a column that contains a mix of floats and integers
    # but at least one float is considered a numerical
    # column, and so we do partial pivoting
    if all(isinstance(x, (Float, Integer)) for x in col) and any(
            isinstance(x, Float) for x in col):
        col_abs = [abs(x) for x in col]
        max_value = max(col_abs)
        if iszerofunc(max_value):
            # just because iszerofunc returned True, doesn't
            # mean the value is numerically zero.  Make sure
            # to replace all entries with numerical zeros
            if max_value != 0:
                newly_determined = [(i, 0) for i, x in enumerate(col) if x != 0]
            return (None, None, False, newly_determined)
        index = col_abs.index(max_value)
        return (index, col[index], False, newly_determined)

    # PASS 1 (iszerofunc directly)
    possible_zeros = []
    for i, x in enumerate(col):
        is_zero = iszerofunc(x)
        # is someone wrote a custom iszerofunc, it may return
        # BooleanFalse or BooleanTrue instead of True or False,
        # so use == for comparison instead of `is`
        if is_zero == False:
            # we found something that is definitely not zero
            return (i, x, False, newly_determined)
        possible_zeros.append(is_zero)

    # by this point, we've found no certain non-zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 2 (iszerofunc after simplify)
    # we haven't found any for-sure non-zeros, so
    # go through the elements iszerofunc couldn't
    # make a determination about and opportunistically
    # simplify to see if we find something
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        simped = simpfunc(x)
        is_zero = iszerofunc(simped)
        if is_zero in (True, False):
            newly_determined.append((i, simped))
        if is_zero == False:
            return (i, simped, False, newly_determined)
        possible_zeros[i] = is_zero

    # after simplifying, some things that were recognized
    # as zeros might be zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 3 (.equals(0))
    # some expressions fail to simplify to zero, but
    # ``.equals(0)`` evaluates to True.  As a last-ditch
    # attempt, apply ``.equals`` to these expressions
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        if x.equals(S.Zero):
            # ``.iszero`` may return False with
            # an implicit assumption (e.g., ``x.equals(0)``
            # when ``x`` is a symbol), so only treat it
            # as proved when ``.equals(0)`` returns True
            possible_zeros[i] = True
            newly_determined.append((i, S.Zero))

    if all(possible_zeros):
        return (None, None, False, newly_determined)

    # at this point there is nothing that could definitely
    # be a pivot.  To maintain compatibility with existing
    # behavior, we'll assume that an illdetermined thing is
    # non-zero.  We should probably raise a warning in this case
    i = possible_zeros.index(None)
    return (i, col[i], True, newly_determined)


def _find_reasonable_pivot_naive(col, iszerofunc=_iszero, simpfunc=None):
    """
    Helper that computes the pivot value and location from a
    sequence of contiguous matrix column elements. As a side effect
    of the pivot search, this function may simplify some of the elements
    of the input column. A list of these simplified entries and their
    indices are also returned.
    This function mimics the behavior of _find_reasonable_pivot(),
    but does less work trying to determine if an indeterminate candidate
    pivot simplifies to zero. This more naive approach can be much faster,
    with the trade-off that it may erroneously return a pivot that is zero.

    ``col`` is a sequence of contiguous column entries to be searched for
    a suitable pivot.
    ``iszerofunc`` is a callable that returns a Boolean that indicates
    if its input is zero, or None if no such determination can be made.
    ``simpfunc`` is a callable that simplifies its input. It must return
    its input if it does not simplify its input. Passing in
    ``simpfunc=None`` indicates that the pivot search should not attempt
    to simplify any candidate pivots.

    Returns a 4-tuple:
    (pivot_offset, pivot_val, assumed_nonzero, newly_determined)
    ``pivot_offset`` is the sequence index of the pivot.
    ``pivot_val`` is the value of the pivot.
    pivot_val and col[pivot_index] are equivalent, but will be different
    when col[pivot_index] was simplified during the pivot search.
    ``assumed_nonzero`` is a boolean indicating if the pivot cannot be
    guaranteed to be zero. If assumed_nonzero is true, then the pivot
    may or may not be non-zero. If assumed_nonzero is false, then
    the pivot is non-zero.
    ``newly_determined`` is a list of index-value pairs of pivot candidates
    that were simplified during the pivot search.
    """

    # indeterminates holds the index-value pairs of each pivot candidate
    # that is neither zero or non-zero, as determined by iszerofunc().
    # If iszerofunc() indicates that a candidate pivot is guaranteed
    # non-zero, or that every candidate pivot is zero then the contents
    # of indeterminates are unused.
    # Otherwise, the only viable candidate pivots are symbolic.
    # In this case, indeterminates will have at least one entry,
    # and all but the first entry are ignored when simpfunc is None.
    indeterminates = []
    for i, col_val in enumerate(col):
        col_val_is_zero = iszerofunc(col_val)
        if col_val_is_zero == False:
            # This pivot candidate is non-zero.
            return i, col_val, False, []
        elif col_val_is_zero is None:
            # The candidate pivot's comparison with zero
            # is indeterminate.
            indeterminates.append((i, col_val))

    if len(indeterminates) == 0:
        # All candidate pivots are guaranteed to be zero, i.e. there is
        # no pivot.
        return None, None, False, []

    if simpfunc is None:
        # Caller did not pass in a simplification function that might
        # determine if an indeterminate pivot candidate is guaranteed
        # to be nonzero, so assume the first indeterminate candidate
        # is non-zero.
        return indeterminates[0][0], indeterminates[0][1], True, []

    # newly_determined holds index-value pairs of candidate pivots
    # that were simplified during the search for a non-zero pivot.
    newly_determined = []
    for i, col_val in indeterminates:
        tmp_col_val = simpfunc(col_val)
        if id(col_val) != id(tmp_col_val):
            # simpfunc() simplified this candidate pivot.
            newly_determined.append((i, tmp_col_val))
            if iszerofunc(tmp_col_val) == False:
                # Candidate pivot simplified to a guaranteed non-zero value.
                return i, tmp_col_val, False, newly_determined

    return indeterminates[0][0], indeterminates[0][1], True, newly_determined


# This functions is a candidate for caching if it gets implemented for matrices.
def _berkowitz_toeplitz_matrix(M):
    """Return (A,T) where T the Toeplitz matrix used in the Berkowitz algorithm
    corresponding to ``M`` and A is the first principal submatrix.
    """

    # the 0 x 0 case is trivial
    if M.rows == 0 and M.cols == 0:
        return M._new(1,1, [M.one])

    #
    # Partition M = [ a_11  R ]
    #                  [ C     A ]
    #

    a, R = M[0,0],   M[0, 1:]
    C, A = M[1:, 0], M[1:,1:]

    #
    # The Toeplitz matrix looks like
    #
    #  [ 1                                     ]
    #  [ -a         1                          ]
    #  [ -RC       -a        1                 ]
    #  [ -RAC     -RC       -a       1         ]
    #  [ -RA**2C -RAC      -RC      -a       1 ]
    #  etc.

    # Compute the diagonal entries.
    # Because multiplying matrix times vector is so much
    # more efficient than matrix times matrix, recursively
    # compute -R * A**n * C.
    diags = [C]
    for i in range(M.rows - 2):
        diags.append(A.multiply(diags[i], dotprodsimp=None))
    diags = [(-R).multiply(d, dotprodsimp=None)[0, 0] for d in diags]
    diags = [M.one, -a] + diags

    def entry(i,j):
        if j > i:
            return M.zero
        return diags[i - j]

    toeplitz = M._new(M.cols + 1, M.rows, entry)
    return (A, toeplitz)


# This functions is a candidate for caching if it gets implemented for matrices.
def _berkowitz_vector(M):
    """ Run the Berkowitz algorithm and return a vector whose entries
        are the coefficients of the characteristic polynomial of ``M``.

        Given N x N matrix, efficiently compute
        coefficients of characteristic polynomials of ``M``
        without division in the ground domain.

        This method is particularly useful for computing determinant,
        principal minors and characteristic polynomial when ``M``
        has complicated coefficients e.g. polynomials. Semi-direct
        usage of this algorithm is also important in computing
        efficiently sub-resultant PRS.

        Assuming that M is a square matrix of dimension N x N and
        I is N x N identity matrix, then the Berkowitz vector is
        an N x 1 vector whose entries are coefficients of the
        polynomial

                        charpoly(M) = det(t*I - M)

        As a consequence, all polynomials generated by Berkowitz
        algorithm are monic.

        For more information on the implemented algorithm refer to:

        [1] S.J. Berkowitz, On computing the determinant in small
            parallel time using a small number of processors, ACM,
            Information Processing Letters 18, 1984, pp. 147-150

        [2] M. Keber, Division-Free computation of sub-resultants
            using Bezout matrices, Tech. Report MPI-I-2006-1-006,
            Saarbrucken, 2006
    """

    # handle the trivial cases
    if M.rows == 0 and M.cols == 0:
        return M._new(1, 1, [M.one])
    elif M.rows == 1 and M.cols == 1:
        return M._new(2, 1, [M.one, -M[0,0]])

    submat, toeplitz = _berkowitz_toeplitz_matrix(M)

    return toeplitz.multiply(_berkowitz_vector(submat), dotprodsimp=None)


def _adjugate(M, method="berkowitz"):
    """Returns the adjugate, or classical adjoint, of
    a matrix.  That is, the transpose of the matrix of cofactors.

    https://en.wikipedia.org/wiki/Adjugate

    Parameters
    ==========

    method : string, optional
        Method to use to find the cofactors, can be "bareiss", "berkowitz",
        "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.adjugate()
    Matrix([
    [ 4, -2],
    [-3,  1]])

    See Also
    ========

    cofactor_matrix
    sympy.matrices.matrixbase.MatrixBase.transpose
    """

    return M.cofactor_matrix(method=method).transpose()


# This functions is a candidate for caching if it gets implemented for matrices.
def _charpoly(M, x='lambda', simplify=_simplify):
    """Computes characteristic polynomial det(x*I - M) where I is
    the identity matrix.

    A PurePoly is returned, so using different variables for ``x`` does
    not affect the comparison or the polynomials:

    Parameters
    ==========

    x : string, optional
        Name for the "lambda" variable, defaults to "lambda".

    simplify : function, optional
        Simplification function to use on the characteristic polynomial
        calculated. Defaults to ``simplify``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x, y
    >>> M = Matrix([[1, 3], [2, 0]])
    >>> M.charpoly()
    PurePoly(lambda**2 - lambda - 6, lambda, domain='ZZ')
    >>> M.charpoly(x) == M.charpoly(y)
    True
    >>> M.charpoly(x) == M.charpoly(y)
    True

    Specifying ``x`` is optional; a symbol named ``lambda`` is used by
    default (which looks good when pretty-printed in unicode):

    >>> M.charpoly().as_expr()
    lambda**2 - lambda - 6

    And if ``x`` clashes with an existing symbol, underscores will
    be prepended to the name to make it unique:

    >>> M = Matrix([[1, 2], [x, 0]])
    >>> M.charpoly(x).as_expr()
    _x**2 - _x - 2*x

    Whether you pass a symbol or not, the generator can be obtained
    with the gen attribute since it may not be the same as the symbol
    that was passed:

    >>> M.charpoly(x).gen
    _x
    >>> M.charpoly(x).gen == x
    False

    Notes
    =====

    The Samuelson-Berkowitz algorithm is used to compute
    the characteristic polynomial efficiently and without any
    division operations.  Thus the characteristic polynomial over any
    commutative ring without zero divisors can be computed.

    If the determinant det(x*I - M) can be found out easily as
    in the case of an upper or a lower triangular matrix, then
    instead of Samuelson-Berkowitz algorithm, eigenvalues are computed
    and the characteristic polynomial with their help.

    See Also
    ========

    det
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    # Use DomainMatrix. We are already going to convert this to a Poly so there
    # is no need to worry about expanding powers etc. Also since this algorithm
    # does not require division or zero detection it is fine to use EX.
    #
    # M.to_DM() will fall back on EXRAW rather than EX. EXRAW is a lot faster
    # for elementary arithmetic because it does not call cancel for each
    # operation but it generates large unsimplified results that are slow in
    # the subsequent call to simplify. Using EX instead is faster overall
    # but at least in some cases EXRAW+simplify gives a simpler result so we
    # preserve that existing behaviour of charpoly for now...
    dM = M.to_DM()

    K = dM.domain

    cp = dM.charpoly()

    x = uniquely_named_symbol(x, [M], modify=lambda s: '_' + s)

    if K.is_EXRAW or simplify is not _simplify:
        # XXX: Converting back to Expr is expensive. We only do it if the
        # caller supplied a custom simplify function for backwards
        # compatibility or otherwise if the domain was EX. For any other domain
        # there should be no benefit in simplifying at this stage because Poly
        # will put everything into canonical form anyway.
        berk_vector = [K.to_sympy(c) for c in cp]
        berk_vector = [simplify(a) for a in berk_vector]
        p = PurePoly(berk_vector, x)

    else:
        # Convert from the list of domain elements directly to Poly.
        p = PurePoly(cp, x, domain=K)

    return p


def _cofactor(M, i, j, method="berkowitz"):
    """Calculate the cofactor of an element.

    Parameters
    ==========

    method : string, optional
        Method to use to find the cofactors, can be "bareiss", "berkowitz",
        "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.cofactor(0, 1)
    -3

    See Also
    ========

    cofactor_matrix
    minor
    minor_submatrix
    """

    if not M.is_square or M.rows < 1:
        raise NonSquareMatrixError()

    return S.NegativeOne**((i + j) % 2) * M.minor(i, j, method)


def _cofactor_matrix(M, method="berkowitz"):
    """Return a matrix containing the cofactor of each element.

    Parameters
    ==========

    method : string, optional
        Method to use to find the cofactors, can be "bareiss", "berkowitz",
        "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.cofactor_matrix()
    Matrix([
    [ 4, -3],
    [-2,  1]])

    See Also
    ========

    cofactor
    minor
    minor_submatrix
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    return M._new(M.rows, M.cols,
            lambda i, j: M.cofactor(i, j, method))

def _per(M):
    """Returns the permanent of a matrix. Unlike determinant,
    permanent is defined for both square and non-square matrices.

    For an m x n matrix, with m less than or equal to n,
    it is given as the sum over the permutations s of size
    less than or equal to m on [1, 2, . . . n] of the product
    from i = 1 to m of M[i, s[i]]. Taking the transpose will
    not affect the value of the permanent.

    In the case of a square matrix, this is the same as the permutation
    definition of the determinant, but it does not take the sign of the
    permutation into account. Computing the permanent with this definition
    is quite inefficient, so here the Ryser formula is used.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.per()
    450
    >>> M = Matrix([1, 5, 7])
    >>> M.per()
    13

    References
    ==========

    .. [1] Prof. Frank Ben's notes: https://math.berkeley.edu/~bernd/ban275.pdf
    .. [2] Wikipedia article on Permanent: https://en.wikipedia.org/wiki/Permanent_%28mathematics%29
    .. [3] https://reference.wolfram.com/language/ref/Permanent.html
    .. [4] Permanent of a rectangular matrix : https://arxiv.org/pdf/0904.3251.pdf
    """
    import itertools

    m, n = M.shape
    if m > n:
        M = M.T
        m, n = n, m
    s = list(range(n))

    subsets = []
    for i in range(1, m + 1):
        subsets += list(map(list, itertools.combinations(s, i)))

    perm = 0
    for subset in subsets:
        prod = 1
        sub_len = len(subset)
        for i in range(m):
             prod *= sum(M[i, j] for j in subset)
        perm += prod * S.NegativeOne**sub_len * nC(n - sub_len, m - sub_len)
    perm *= S.NegativeOne**m
    return perm.simplify()

def _det_DOM(M):
    DOM = DomainMatrix.from_Matrix(M, field=True, extension=True)
    K = DOM.domain
    return K.to_sympy(DOM.det())

# This functions is a candidate for caching if it gets implemented for matrices.
def _det(M, method="bareiss", iszerofunc=None):
    """Computes the determinant of a matrix if ``M`` is a concrete matrix object
    otherwise return an expressions ``Determinant(M)`` if ``M`` is a
    ``MatrixSymbol`` or other expression.

    Parameters
    ==========

    method : string, optional
        Specifies the algorithm used for computing the matrix determinant.

        If the matrix is at most 3x3, a hard-coded formula is used and the
        specified method is ignored. Otherwise, it defaults to
        ``'bareiss'``.

        Also, if the matrix is an upper or a lower triangular matrix, determinant
        is computed by simple multiplication of diagonal elements, and the
        specified method is ignored.

        If it is set to ``'domain-ge'``, then Gaussian elimination method will
        be used via using DomainMatrix.

        If it is set to ``'bareiss'``, Bareiss' fraction-free algorithm will
        be used.

        If it is set to ``'berkowitz'``, Berkowitz' algorithm will be used.

        If it is set to ``'bird'``, Bird's algorithm will be used [1]_.

        If it is set to ``'laplace'``, Laplace's algorithm will be used.

        Otherwise, if it is set to ``'lu'``, LU decomposition will be used.

        .. note::
            For backward compatibility, legacy keys like "bareis" and
            "det_lu" can still be used to indicate the corresponding
            methods.
            And the keys are also case-insensitive for now. However, it is
            suggested to use the precise keys for specifying the method.

    iszerofunc : FunctionType or None, optional
        If it is set to ``None``, it will be defaulted to ``_iszero`` if the
        method is set to ``'bareiss'``, and ``_is_zero_after_expand_mul`` if
        the method is set to ``'lu'``.

        It can also accept any user-specified zero testing function, if it
        is formatted as a function which accepts a single symbolic argument
        and returns ``True`` if it is tested as zero and ``False`` if it
        tested as non-zero, and also ``None`` if it is undecidable.

    Returns
    =======

    det : Basic
        Result of determinant.

    Raises
    ======

    ValueError
        If unrecognized keys are given for ``method`` or ``iszerofunc``.

    NonSquareMatrixError
        If attempted to calculate determinant from a non-square matrix.

    Examples
    ========

    >>> from sympy import Matrix, eye, det
    >>> I3 = eye(3)
    >>> det(I3)
    1
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> det(M)
    -2
    >>> det(M) == M.det()
    True
    >>> M.det(method="domain-ge")
    -2

    References
    ==========

    .. [1] Bird, R. S. (2011). A simple division-free algorithm for computing
           determinants. Inf. Process. Lett., 111(21), 1072-1074. doi:
           10.1016/j.ipl.2011.08.006
    """

    # sanitize `method`
    method = method.lower()

    if method == "bareis":
        method = "bareiss"
    elif method == "det_lu":
        method = "lu"

    if method not in ("bareiss", "berkowitz", "lu", "domain-ge", "bird",
                      "laplace"):
        raise ValueError("Determinant method '%s' unrecognized" % method)

    if iszerofunc is None:
        if method == "bareiss":
            iszerofunc = _is_zero_after_expand_mul
        elif method == "lu":
            iszerofunc = _iszero

    elif not isinstance(iszerofunc, FunctionType):
        raise ValueError("Zero testing method '%s' unrecognized" % iszerofunc)

    n = M.rows

    if n == M.cols: # square check is done in individual method functions
        if n == 0:
            return M.one
        elif n == 1:
            return M[0, 0]
        elif n == 2:
            m = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            return _get_intermediate_simp(_dotprodsimp)(m)
        elif n == 3:
            m =  (M[0, 0] * M[1, 1] * M[2, 2]
                + M[0, 1] * M[1, 2] * M[2, 0]
                + M[0, 2] * M[1, 0] * M[2, 1]
                - M[0, 2] * M[1, 1] * M[2, 0]
                - M[0, 0] * M[1, 2] * M[2, 1]
                - M[0, 1] * M[1, 0] * M[2, 2])
            return _get_intermediate_simp(_dotprodsimp)(m)

    dets = []
    for b in M.strongly_connected_components():
        if method == "domain-ge": # uses DomainMatrix to evaluate determinant
            det = _det_DOM(M[b, b])
        elif method == "bareiss":
            det = M[b, b]._eval_det_bareiss(iszerofunc=iszerofunc)
        elif method == "berkowitz":
            det = M[b, b]._eval_det_berkowitz()
        elif method == "lu":
            det = M[b, b]._eval_det_lu(iszerofunc=iszerofunc)
        elif method == "bird":
            det = M[b, b]._eval_det_bird()
        elif method == "laplace":
            det = M[b, b]._eval_det_laplace()
        dets.append(det)
    return Mul(*dets)


# This functions is a candidate for caching if it gets implemented for matrices.
def _det_bareiss(M, iszerofunc=_is_zero_after_expand_mul):
    """Compute matrix determinant using Bareiss' fraction-free
    algorithm which is an extension of the well known Gaussian
    elimination method. This approach is best suited for dense
    symbolic matrices and will result in a determinant with
    minimal number of fractions. It means that less term
    rewriting is needed on resulting formulae.

    Parameters
    ==========

    iszerofunc : function, optional
        The function to use to determine zeros when doing an LU decomposition.
        Defaults to ``lambda x: x.is_zero``.

    TODO: Implement algorithm for sparse matrices (SFF),
    http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.
    """

    # Recursively implemented Bareiss' algorithm as per Deanna Richelle Leggett's
    # thesis http://www.math.usm.edu/perry/Research/Thesis_DRL.pdf
    def bareiss(mat, cumm=1):
        if mat.rows == 0:
            return mat.one
        elif mat.rows == 1:
            return mat[0, 0]

        # find a pivot and extract the remaining matrix
        # With the default iszerofunc, _find_reasonable_pivot slows down
        # the computation by the factor of 2.5 in one test.
        # Relevant issues: #10279 and #13877.
        pivot_pos, pivot_val, _, _ = _find_reasonable_pivot(mat[:, 0], iszerofunc=iszerofunc)
        if pivot_pos is None:
            return mat.zero

        # if we have a valid pivot, we'll do a "row swap", so keep the
        # sign of the det
        sign = (-1) ** (pivot_pos % 2)

        # we want every row but the pivot row and every column
        rows = [i for i in range(mat.rows) if i != pivot_pos]
        cols = list(range(mat.cols))
        tmp_mat = mat.extract(rows, cols)

        def entry(i, j):
            ret = (pivot_val*tmp_mat[i, j + 1] - mat[pivot_pos, j + 1]*tmp_mat[i, 0]) / cumm
            if _get_intermediate_simp_bool(True):
                return _dotprodsimp(ret)
            elif not ret.is_Atom:
                return cancel(ret)
            return ret

        return sign*bareiss(M._new(mat.rows - 1, mat.cols - 1, entry), pivot_val)

    if not M.is_square:
        raise NonSquareMatrixError()

    if M.rows == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.

    return bareiss(M)


def _det_berkowitz(M):
    """ Use the Berkowitz algorithm to compute the determinant."""

    if not M.is_square:
        raise NonSquareMatrixError()

    if M.rows == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.

    berk_vector = _berkowitz_vector(M)
    return (-1)**(len(berk_vector) - 1) * berk_vector[-1]


# This functions is a candidate for caching if it gets implemented for matrices.
def _det_LU(M, iszerofunc=_iszero, simpfunc=None):
    """ Computes the determinant of a matrix from its LU decomposition.
    This function uses the LU decomposition computed by
    LUDecomposition_Simple().

    The keyword arguments iszerofunc and simpfunc are passed to
    LUDecomposition_Simple().
    iszerofunc is a callable that returns a boolean indicating if its
    input is zero, or None if it cannot make the determination.
    simpfunc is a callable that simplifies its input.
    The default is simpfunc=None, which indicate that the pivot search
    algorithm should not attempt to simplify any candidate pivots.
    If simpfunc fails to simplify its input, then it must return its input
    instead of a copy.

    Parameters
    ==========

    iszerofunc : function, optional
        The function to use to determine zeros when doing an LU decomposition.
        Defaults to ``lambda x: x.is_zero``.

    simpfunc : function, optional
        The simplification function to use when looking for zeros for pivots.
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    if M.rows == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.

    lu, row_swaps = M.LUdecomposition_Simple(iszerofunc=iszerofunc,
            simpfunc=simpfunc)
    # P*A = L*U => det(A) = det(L)*det(U)/det(P) = det(P)*det(U).
    # Lower triangular factor L encoded in lu has unit diagonal => det(L) = 1.
    # P is a permutation matrix => det(P) in {-1, 1} => 1/det(P) = det(P).
    # LUdecomposition_Simple() returns a list of row exchange index pairs, rather
    # than a permutation matrix, but det(P) = (-1)**len(row_swaps).

    # Avoid forming the potentially time consuming  product of U's diagonal entries
    # if the product is zero.
    # Bottom right entry of U is 0 => det(A) = 0.
    # It may be impossible to determine if this entry of U is zero when it is symbolic.
    if iszerofunc(lu[lu.rows-1, lu.rows-1]):
        return M.zero

    # Compute det(P)
    det = -M.one if len(row_swaps)%2 else M.one

    # Compute det(U) by calculating the product of U's diagonal entries.
    # The upper triangular portion of lu is the upper triangular portion of the
    # U factor in the LU decomposition.
    for k in range(lu.rows):
        det *= lu[k, k]

    # return det(P)*det(U)
    return det


@cacheit
def __det_laplace(M):
    """Compute the determinant of a matrix using Laplace expansion.

    This is a recursive function, and it should not be called directly.
    Use _det_laplace() instead. The reason for splitting this function
    into two is to allow caching of determinants of submatrices. While
    one could also define this function inside _det_laplace(), that
    would remove the advantage of using caching in Cramer Solve.
    """
    n = M.shape[0]
    if n == 1:
        return M[0]
    elif n == 2:
        return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    else:
        return sum((-1) ** i * M[0, i] *
                   __det_laplace(M.minor_submatrix(0, i)) for i in range(n))


def _det_laplace(M):
    """Compute the determinant of a matrix using Laplace expansion.

    While Laplace expansion is not the most efficient method of computing
    a determinant, it is a simple one, and it has the advantage of
    being division free. To improve efficiency, this function uses
    caching to avoid recomputing determinants of submatrices.
    """
    if not M.is_square:
        raise NonSquareMatrixError()
    if M.shape[0] == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.
    return __det_laplace(M.as_immutable())


def _det_bird(M):
    r"""Compute the determinant of a matrix using Bird's algorithm.

    Bird's algorithm is a simple division-free algorithm for computing, which
    is of lower order than the Laplace's algorithm. It is described in [1]_.

    References
    ==========

    .. [1] Bird, R. S. (2011). A simple division-free algorithm for computing
           determinants. Inf. Process. Lett., 111(21), 1072-1074. doi:
           10.1016/j.ipl.2011.08.006
    """
    def mu(X):
        n = X.shape[0]
        zero = X.domain.zero

        total = zero
        diag_sums = [zero]
        for i in reversed(range(1, n)):
            total -= X[i][i]
            diag_sums.append(total)
        diag_sums = diag_sums[::-1]

        elems = [[zero] * i + [diag_sums[i]] + X_i[i + 1:] for i, X_i in
                 enumerate(X)]
        return DDM(elems, X.shape, X.domain)

    Mddm = M._rep.to_ddm()
    n = M.shape[0]
    if n == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.
    Fn1 = Mddm
    for _ in range(n - 1):
        Fn1 = mu(Fn1).matmul(Mddm)
    detA = Fn1[0][0]
    if n % 2 == 0:
        detA = -detA

    return Mddm.domain.to_sympy(detA)


def _minor(M, i, j, method="berkowitz"):
    """Return the (i,j) minor of ``M``.  That is,
    return the determinant of the matrix obtained by deleting
    the `i`th row and `j`th column from ``M``.

    Parameters
    ==========

    i, j : int
        The row and column to exclude to obtain the submatrix.

    method : string, optional
        Method to use to find the determinant of the submatrix, can be
        "bareiss", "berkowitz", "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.minor(1, 1)
    -12

    See Also
    ========

    minor_submatrix
    cofactor
    det
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    return M.minor_submatrix(i, j).det(method=method)


def _minor_submatrix(M, i, j):
    """Return the submatrix obtained by removing the `i`th row
    and `j`th column from ``M`` (works with Pythonic negative indices).

    Parameters
    ==========

    i, j : int
        The row and column to exclude to obtain the submatrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.minor_submatrix(1, 1)
    Matrix([
    [1, 3],
    [7, 9]])

    See Also
    ========

    minor
    cofactor
    """

    if i < 0:
        i += M.rows
    if j < 0:
        j += M.cols

    if not 0 <= i < M.rows or not 0 <= j < M.cols:
        raise ValueError("`i` and `j` must satisfy 0 <= i < ``M.rows`` "
                            "(%d)" % M.rows + "and 0 <= j < ``M.cols`` (%d)." % M.cols)

    rows = [a for a in range(M.rows) if a != i]
    cols = [a for a in range(M.cols) if a != j]

    return M.extract(rows, cols)
