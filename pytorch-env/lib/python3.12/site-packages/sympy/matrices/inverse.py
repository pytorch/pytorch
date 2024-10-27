from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.domains import EX

from .exceptions import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero


def _pinv_full_rank(M):
    """Subroutine for full row or column rank matrices.

    For full row rank matrices, inverse of ``A * A.H`` Exists.
    For full column rank matrices, inverse of ``A.H * A`` Exists.

    This routine can apply for both cases by checking the shape
    and have small decision.
    """

    if M.is_zero_matrix:
        return M.H

    if M.rows >= M.cols:
        return M.H.multiply(M).inv().multiply(M.H)
    else:
        return M.H.multiply(M.multiply(M.H).inv())

def _pinv_rank_decomposition(M):
    """Subroutine for rank decomposition

    With rank decompositions, `A` can be decomposed into two full-
    rank matrices, and each matrix can take pseudoinverse
    individually.
    """

    if M.is_zero_matrix:
        return M.H

    B, C = M.rank_decomposition()

    Bp = _pinv_full_rank(B)
    Cp = _pinv_full_rank(C)

    return Cp.multiply(Bp)

def _pinv_diagonalization(M):
    """Subroutine using diagonalization

    This routine can sometimes fail if SymPy's eigenvalue
    computation is not reliable.
    """

    if M.is_zero_matrix:
        return M.H

    A  = M
    AH = M.H

    try:
        if M.rows >= M.cols:
            P, D   = AH.multiply(A).diagonalize(normalize=True)
            D_pinv = D.applyfunc(lambda x: 0 if _iszero(x) else 1 / x)

            return P.multiply(D_pinv).multiply(P.H).multiply(AH)

        else:
            P, D   = A.multiply(AH).diagonalize(
                        normalize=True)
            D_pinv = D.applyfunc(lambda x: 0 if _iszero(x) else 1 / x)

            return AH.multiply(P).multiply(D_pinv).multiply(P.H)

    except MatrixError:
        raise NotImplementedError(
            'pinv for rank-deficient matrices where '
            'diagonalization of A.H*A fails is not supported yet.')

def _pinv(M, method='RD'):
    """Calculate the Moore-Penrose pseudoinverse of the matrix.

    The Moore-Penrose pseudoinverse exists and is unique for any matrix.
    If the matrix is invertible, the pseudoinverse is the same as the
    inverse.

    Parameters
    ==========

    method : String, optional
        Specifies the method for computing the pseudoinverse.

        If ``'RD'``, Rank-Decomposition will be used.

        If ``'ED'``, Diagonalization will be used.

    Examples
    ========

    Computing pseudoinverse by rank decomposition :

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> A.pinv()
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    Computing pseudoinverse by diagonalization :

    >>> B = A.pinv(method='ED')
    >>> B.simplify()
    >>> B
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    See Also
    ========

    inv
    pinv_solve

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse

    """

    # Trivial case: pseudoinverse of all-zero matrix is its transpose.
    if M.is_zero_matrix:
        return M.H

    if method == 'RD':
        return _pinv_rank_decomposition(M)
    elif method == 'ED':
        return _pinv_diagonalization(M)
    else:
        raise ValueError('invalid pinv method %s' % repr(method))


def _verify_invertible(M, iszerofunc=_iszero):
    """Initial check to see if a matrix is invertible. Raises or returns
    determinant for use in _inv_ADJ."""

    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    d    = M.det(method='berkowitz')
    zero = d.equals(0)

    if zero is None: # if equals() can't decide, will rref be able to?
        ok   = M.rref(simplify=True)[0]
        zero = any(iszerofunc(ok[j, j]) for j in range(ok.rows))

    if zero:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    return d

def _inv_ADJ(M, iszerofunc=_iszero):
    """Calculates the inverse using the adjugate matrix and a determinant.

    See Also
    ========

    inv
    inverse_GE
    inverse_LU
    inverse_CH
    inverse_LDL
    """

    d = _verify_invertible(M, iszerofunc=iszerofunc)

    return M.adjugate() / d

def _inv_GE(M, iszerofunc=_iszero):
    """Calculates the inverse using Gaussian elimination.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_LU
    inverse_CH
    inverse_LDL
    """

    from .dense import Matrix

    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    big = Matrix.hstack(M.as_mutable(), Matrix.eye(M.rows))
    red = big.rref(iszerofunc=iszerofunc, simplify=True)[0]

    if any(iszerofunc(red[j, j]) for j in range(red.rows)):
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    return M._new(red[:, big.rows:])

def _inv_LU(M, iszerofunc=_iszero):
    """Calculates the inverse using LU decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """

    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")
    if M.free_symbols:
        _verify_invertible(M, iszerofunc=iszerofunc)

    return M.LUsolve(M.eye(M.rows), iszerofunc=_iszero)

def _inv_CH(M, iszerofunc=_iszero):
    """Calculates the inverse using cholesky decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_LDL
    """

    _verify_invertible(M, iszerofunc=iszerofunc)

    return M.cholesky_solve(M.eye(M.rows))

def _inv_LDL(M, iszerofunc=_iszero):
    """Calculates the inverse using LDL decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_CH
    """

    _verify_invertible(M, iszerofunc=iszerofunc)

    return M.LDLsolve(M.eye(M.rows))

def _inv_QR(M, iszerofunc=_iszero):
    """Calculates the inverse using QR decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """

    _verify_invertible(M, iszerofunc=iszerofunc)

    return M.QRsolve(M.eye(M.rows))

def _try_DM(M, use_EX=False):
    """Try to convert a matrix to a ``DomainMatrix``."""
    dM = M.to_DM()
    K = dM.domain

    # Return DomainMatrix if a domain is found. Only use EX if use_EX=True.
    if not use_EX and K.is_EXRAW:
        return None
    elif K.is_EXRAW:
        return dM.convert_to(EX)
    else:
        return dM


def _use_exact_domain(dom):
    """Check whether to convert to an exact domain."""
    # DomainMatrix can handle RR and CC with partial pivoting. Other inexact
    # domains like RR[a,b,...] can only be handled by converting to an exact
    # domain like QQ[a,b,...]
    if dom.is_RR or dom.is_CC:
        return False
    else:
        return not dom.is_Exact


def _inv_DM(dM, cancel=True):
    """Calculates the inverse using ``DomainMatrix``.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    sympy.polys.matrices.domainmatrix.DomainMatrix.inv
    """
    m, n = dM.shape
    dom = dM.domain

    if m != n:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    # Convert RR[a,b,...] to QQ[a,b,...]
    use_exact = _use_exact_domain(dom)

    if use_exact:
        dom_exact = dom.get_exact()
        dM = dM.convert_to(dom_exact)

    try:
        dMi, den = dM.inv_den()
    except DMNonInvertibleMatrixError:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    if use_exact:
        dMi = dMi.convert_to(dom)
        den = dom.convert_from(den, dom_exact)

    if cancel:
        # Convert to field and cancel with the denominator.
        if not dMi.domain.is_Field:
            dMi = dMi.to_field()
        Mi = (dMi / den).to_Matrix()
    else:
        # Convert to Matrix and divide without cancelling
        Mi = dMi.to_Matrix() / dMi.domain.to_sympy(den)

    return Mi

def _inv_block(M, iszerofunc=_iszero):
    """Calculates the inverse using BLOCKWISE inversion.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """
    from sympy.matrices.expressions.blockmatrix import BlockMatrix
    i = M.shape[0]
    if i <= 20 :
        return M.inv(method="LU", iszerofunc=_iszero)
    A = M[:i // 2, :i //2]
    B = M[:i // 2, i // 2:]
    C = M[i // 2:, :i // 2]
    D = M[i // 2:, i // 2:]
    try:
        D_inv = _inv_block(D)
    except NonInvertibleMatrixError:
        return M.inv(method="LU", iszerofunc=_iszero)
    B_D_i = B*D_inv
    BDC = B_D_i*C
    A_n = A - BDC
    try:
        A_n = _inv_block(A_n)
    except NonInvertibleMatrixError:
        return M.inv(method="LU", iszerofunc=_iszero)
    B_n = -A_n*B_D_i
    dc = D_inv*C
    C_n = -dc*A_n
    D_n = D_inv + dc*-B_n
    nn = BlockMatrix([[A_n, B_n], [C_n, D_n]]).as_explicit()
    return nn

def _inv(M, method=None, iszerofunc=_iszero, try_block_diag=False):
    """
    Return the inverse of a matrix using the method indicated. The default
    is DM if a suitable domain is found or otherwise GE for dense matrices
    LDL for sparse matrices.

    Parameters
    ==========

    method : ('DM', 'DMNC', 'GE', 'LU', 'ADJ', 'CH', 'LDL', 'QR')

    iszerofunc : function, optional
        Zero-testing function to use.

    try_block_diag : bool, optional
        If True then will try to form block diagonal matrices using the
        method get_diag_blocks(), invert these individually, and then
        reconstruct the full inverse matrix.

    Examples
    ========

    >>> from sympy import SparseMatrix, Matrix
    >>> A = SparseMatrix([
    ... [ 2, -1,  0],
    ... [-1,  2, -1],
    ... [ 0,  0,  2]])
    >>> A.inv('CH')
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A.inv(method='LDL') # use of 'method=' is optional
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A * _
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> A = Matrix(A)
    >>> A.inv('CH')
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A.inv('ADJ') == A.inv('GE') == A.inv('LU') == A.inv('CH') == A.inv('LDL') == A.inv('QR')
    True

    Notes
    =====

    According to the ``method`` keyword, it calls the appropriate method:

        DM .... Use DomainMatrix ``inv_den`` method
        DMNC .... Use DomainMatrix ``inv_den`` method without cancellation
        GE .... inverse_GE(); default for dense matrices
        LU .... inverse_LU()
        ADJ ... inverse_ADJ()
        CH ... inverse_CH()
        LDL ... inverse_LDL(); default for sparse matrices
        QR ... inverse_QR()

    Note, the GE and LU methods may require the matrix to be simplified
    before it is inverted in order to properly detect zeros during
    pivoting. In difficult cases a custom zero detection function can
    be provided by setting the ``iszerofunc`` argument to a function that
    should return True if its argument is zero. The ADJ routine computes
    the determinant and uses that to detect singular matrices in addition
    to testing for zeros on the diagonal.

    See Also
    ========

    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_CH
    inverse_LDL

    Raises
    ======

    ValueError
        If the determinant of the matrix is zero.
    """

    from sympy.matrices import diag, SparseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    if try_block_diag:
        blocks = M.get_diag_blocks()
        r      = []

        for block in blocks:
            r.append(block.inv(method=method, iszerofunc=iszerofunc))

        return diag(*r)

    # Default: Use DomainMatrix if the domain is not EX.
    # If DM is requested explicitly then use it even if the domain is EX.
    if method is None and iszerofunc is _iszero:
        dM = _try_DM(M, use_EX=False)
        if dM is not None:
            method = 'DM'
    elif method in ("DM", "DMNC"):
        dM = _try_DM(M, use_EX=True)

    # A suitable domain was not found, fall back to GE for dense matrices
    # and LDL for sparse matrices.
    if method is None:
        if isinstance(M, SparseMatrix):
            method = 'LDL'
        else:
            method = 'GE'

    if method == "DM":
        rv = _inv_DM(dM)
    elif method == "DMNC":
        rv = _inv_DM(dM, cancel=False)
    elif method == "GE":
        rv = M.inverse_GE(iszerofunc=iszerofunc)
    elif method == "LU":
        rv = M.inverse_LU(iszerofunc=iszerofunc)
    elif method == "ADJ":
        rv = M.inverse_ADJ(iszerofunc=iszerofunc)
    elif method == "CH":
        rv = M.inverse_CH(iszerofunc=iszerofunc)
    elif method == "LDL":
        rv = M.inverse_LDL(iszerofunc=iszerofunc)
    elif method == "QR":
        rv = M.inverse_QR(iszerofunc=iszerofunc)
    elif method == "BLOCK":
        rv = M.inverse_BLOCK(iszerofunc=iszerofunc)
    else:
        raise ValueError("Inversion method unrecognized")

    return M._new(rv)
