from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols

from .exceptions import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero


def _diagonal_solve(M, rhs):
    """Solves ``Ax = B`` efficiently, where A is a diagonal Matrix,
    with non-zero diagonal entries.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = eye(2)*2
    >>> B = Matrix([[1, 2], [3, 4]])
    >>> A.diagonal_solve(B) == B/2
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_diagonal():
        raise TypeError("Matrix should be diagonal")
    if rhs.rows != M.rows:
        raise TypeError("Size mismatch")

    return M._new(
        rhs.rows, rhs.cols, lambda i, j: rhs[i, j] / M[i, i])


def _lower_triangular_solve(M, rhs):
    """Solves ``Ax = B``, where A is a lower triangular matrix.

    See Also
    ========

    upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrices size mismatch.")
    if not M.is_lower:
        raise ValueError("Matrix must be lower triangular.")

    dps = _get_intermediate_simp()
    X   = MutableDenseMatrix.zeros(M.rows, rhs.cols)

    for j in range(rhs.cols):
        for i in range(M.rows):
            if M[i, i] == 0:
                raise TypeError("Matrix must be non-singular.")

            X[i, j] = dps((rhs[i, j] - sum(M[i, k]*X[k, j]
                                        for k in range(i))) / M[i, i])

    return M._new(X)

def _lower_triangular_solve_sparse(M, rhs):
    """Solves ``Ax = B``, where A is a lower triangular matrix.

    See Also
    ========

    upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrices size mismatch.")
    if not M.is_lower:
        raise ValueError("Matrix must be lower triangular.")

    dps  = _get_intermediate_simp()
    rows = [[] for i in range(M.rows)]

    for i, j, v in M.row_list():
        if i > j:
            rows[i].append((j, v))

    X = rhs.as_mutable()

    for j in range(rhs.cols):
        for i in range(rhs.rows):
            for u, v in rows[i]:
                X[i, j] -= v*X[u, j]

            X[i, j] = dps(X[i, j] / M[i, i])

    return M._new(X)


def _upper_triangular_solve(M, rhs):
    """Solves ``Ax = B``, where A is an upper triangular matrix.

    See Also
    ========

    lower_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrix size mismatch.")
    if not M.is_upper:
        raise TypeError("Matrix is not upper triangular.")

    dps = _get_intermediate_simp()
    X   = MutableDenseMatrix.zeros(M.rows, rhs.cols)

    for j in range(rhs.cols):
        for i in reversed(range(M.rows)):
            if M[i, i] == 0:
                raise ValueError("Matrix must be non-singular.")

            X[i, j] = dps((rhs[i, j] - sum(M[i, k]*X[k, j]
                                        for k in range(i + 1, M.rows))) / M[i, i])

    return M._new(X)

def _upper_triangular_solve_sparse(M, rhs):
    """Solves ``Ax = B``, where A is an upper triangular matrix.

    See Also
    ========

    lower_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrix size mismatch.")
    if not M.is_upper:
        raise TypeError("Matrix is not upper triangular.")

    dps  = _get_intermediate_simp()
    rows = [[] for i in range(M.rows)]

    for i, j, v in M.row_list():
        if i < j:
            rows[i].append((j, v))

    X = rhs.as_mutable()

    for j in range(rhs.cols):
        for i in reversed(range(rhs.rows)):
            for u, v in reversed(rows[i]):
                X[i, j] -= v*X[u, j]

            X[i, j] = dps(X[i, j] / M[i, i])

    return M._new(X)


def _cholesky_solve(M, rhs):
    """Solves ``Ax = B`` using Cholesky decomposition,
    for a general square non-singular matrix.
    For a non-square matrix with rows > cols,
    the least squares solution is returned.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if M.rows < M.cols:
        raise NotImplementedError(
            'Under-determined System. Try M.gauss_jordan_solve(rhs)')

    hermitian = True
    reform    = False

    if M.is_symmetric():
        hermitian = False
    elif not M.is_hermitian:
        reform = True

    if reform or _fuzzy_positive_definite(M) is False:
        H         = M.H
        M         = H.multiply(M)
        rhs       = H.multiply(rhs)
        hermitian = not M.is_symmetric()

    L = M.cholesky(hermitian=hermitian)
    Y = L.lower_triangular_solve(rhs)

    if hermitian:
        return (L.H).upper_triangular_solve(Y)
    else:
        return (L.T).upper_triangular_solve(Y)


def _LDLsolve(M, rhs):
    """Solves ``Ax = B`` using LDL decomposition,
    for a general square and non-singular matrix.

    For a non-square matrix with rows > cols,
    the least squares solution is returned.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = eye(2)*2
    >>> B = Matrix([[1, 2], [3, 4]])
    >>> A.LDLsolve(B) == B/2
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if M.rows < M.cols:
        raise NotImplementedError(
            'Under-determined System. Try M.gauss_jordan_solve(rhs)')

    hermitian = True
    reform    = False

    if M.is_symmetric():
        hermitian = False
    elif not M.is_hermitian:
        reform = True

    if reform or _fuzzy_positive_definite(M) is False:
        H         = M.H
        M         = H.multiply(M)
        rhs       = H.multiply(rhs)
        hermitian = not M.is_symmetric()

    L, D = M.LDLdecomposition(hermitian=hermitian)
    Y    = L.lower_triangular_solve(rhs)
    Z    = D.diagonal_solve(Y)

    if hermitian:
        return (L.H).upper_triangular_solve(Z)
    else:
        return (L.T).upper_triangular_solve(Z)


def _LUsolve(M, rhs, iszerofunc=_iszero):
    """Solve the linear system ``Ax = rhs`` for ``x`` where ``A = M``.

    This is for symbolic matrices, for real or complex ones use
    mpmath.lu_solve or mpmath.qr_solve.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    QRsolve
    pinv_solve
    LUdecomposition
    cramer_solve
    """

    if rhs.rows != M.rows:
        raise ShapeError(
            "``M`` and ``rhs`` must have the same number of rows.")

    m = M.rows
    n = M.cols

    if m < n:
        raise NotImplementedError("Underdetermined systems not supported.")

    try:
        A, perm = M.LUdecomposition_Simple(
            iszerofunc=iszerofunc, rankcheck=True)
    except ValueError:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    dps = _get_intermediate_simp()
    b   = rhs.permute_rows(perm).as_mutable()

    # forward substitution, all diag entries are scaled to 1
    for i in range(m):
        for j in range(min(i, n)):
            scale = A[i, j]
            b.zip_row_op(i, j, lambda x, y: dps(x - scale * y))

    # consistency check for overdetermined systems
    if m > n:
        for i in range(n, m):
            for j in range(b.cols):
                if not iszerofunc(b[i, j]):
                    raise ValueError("The system is inconsistent.")

        b = b[0:n, :]   # truncate zero rows if consistent

    # backward substitution
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            scale = A[i, j]
            b.zip_row_op(i, j, lambda x, y: dps(x - scale * y))

        scale = A[i, i]
        b.row_op(i, lambda x, _: dps(scale**-1 * x))

    return rhs.__class__(b)


def _QRsolve(M, b):
    """Solve the linear system ``Ax = b``.

    ``M`` is the matrix ``A``, the method argument is the vector
    ``b``.  The method returns the solution vector ``x``.  If ``b`` is a
    matrix, the system is solved for each column of ``b`` and the
    return value is a matrix of the same shape as ``b``.

    This method is slower (approximately by a factor of 2) but
    more stable for floating-point arithmetic than the LUsolve method.
    However, LUsolve usually uses an exact arithmetic, so you do not need
    to use QRsolve.

    This is mainly for educational purposes and symbolic matrices, for real
    (or complex) matrices use mpmath.qr_solve.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    pinv_solve
    QRdecomposition
    cramer_solve
    """

    dps  = _get_intermediate_simp(expand_mul, expand_mul)
    Q, R = M.QRdecomposition()
    y    = Q.T * b

    # back substitution to solve R*x = y:
    # We build up the result "backwards" in the vector 'x' and reverse it
    # only in the end.
    x = []
    n = R.rows

    for j in range(n - 1, -1, -1):
        tmp = y[j, :]

        for k in range(j + 1, n):
            tmp -= R[j, k] * x[n - 1 - k]

        tmp = dps(tmp)

        x.append(tmp / R[j, j])

    return M.vstack(*x[::-1])


def _gauss_jordan_solve(M, B, freevar=False):
    """
    Solves ``Ax = B`` using Gauss Jordan elimination.

    There may be zero, one, or infinite solutions.  If one solution
    exists, it will be returned. If infinite solutions exist, it will
    be returned parametrically. If no solutions exist, It will throw
    ValueError.

    Parameters
    ==========

    B : Matrix
        The right hand side of the equation to be solved for.  Must have
        the same number of rows as matrix A.

    freevar : boolean, optional
        Flag, when set to `True` will return the indices of the free
        variables in the solutions (column Matrix), for a system that is
        undetermined (e.g. A has more columns than rows), for which
        infinite solutions are possible, in terms of arbitrary
        values of free variables. Default `False`.

    Returns
    =======

    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    params : Matrix
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of arbitrary
        parameters. These arbitrary parameters are returned as params
        Matrix.

    free_var_index : List, optional
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of arbitrary
        values of free variables. Then the indices of the free variables
        in the solutions (column Matrix) are returned by free_var_index,
        if the flag `freevar` is set to `True`.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> taus_zeroes = { tau:0 for tau in params }
    >>> sol_unique = sol.xreplace(taus_zeroes)
    >>> sol_unique
        Matrix([
    [2],
    [0],
    [5],
    [0]])


    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> B = Matrix([3, 6, 9])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-1],
    [ 2],
    [ 0]])
    >>> params
    Matrix(0, 1, [])

    >>> A = Matrix([[2, -7], [-1, 4]])
    >>> B = Matrix([[-21, 3], [12, -2]])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [0, -2],
    [3, -1]])
    >>> params
    Matrix(0, 2, [])


    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params, freevars = A.gauss_jordan_solve(B, freevar=True)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> freevars
    [1, 3]


    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_elimination

    """

    from sympy.matrices import Matrix, zeros

    cls      = M.__class__
    aug      = M.hstack(M.copy(), B.copy())
    B_cols   = B.cols
    row, col = aug[:, :-B_cols].shape

    # solve by reduced row echelon form
    A, pivots = aug.rref(simplify=True)
    A, v      = A[:, :-B_cols], A[:, -B_cols:]
    pivots    = list(filter(lambda p: p < col, pivots))
    rank      = len(pivots)

    # Get index of free symbols (free parameters)
    # non-pivots columns are free variables
    free_var_index = [c for c in range(A.cols) if c not in pivots]

    # Bring to block form
    permutation = Matrix(pivots + free_var_index).T

    # check for existence of solutions
    # rank of aug Matrix should be equal to rank of coefficient matrix
    if not v[rank:, :].is_zero_matrix:
        raise ValueError("Linear system has no solution")

    # Free parameters
    # what are current unnumbered free symbol names?
    name = uniquely_named_symbol('tau', [aug],
            compare=lambda i: str(i).rstrip('1234567890'),
            modify=lambda s: '_' + s).name
    gen  = numbered_symbols(name)
    tau  = Matrix([next(gen) for k in range((col - rank)*B_cols)]).reshape(
            col - rank, B_cols)

    # Full parametric solution
    V        = A[:rank, free_var_index]
    vt       = v[:rank, :]
    free_sol = tau.vstack(vt - V * tau, tau)

    # Undo permutation
    sol = zeros(col, B_cols)

    for k in range(col):
        sol[permutation[k], :] = free_sol[k,:]

    sol, tau = cls(sol), cls(tau)

    if freevar:
        return sol, tau, free_var_index
    else:
        return sol, tau


def _pinv_solve(M, B, arbitrary_matrix=None):
    """Solve ``Ax = B`` using the Moore-Penrose pseudoinverse.

    There may be zero, one, or infinite solutions.  If one solution
    exists, it will be returned.  If infinite solutions exist, one will
    be returned based on the value of arbitrary_matrix.  If no solutions
    exist, the least-squares solution is returned.

    Parameters
    ==========

    B : Matrix
        The right hand side of the equation to be solved for.  Must have
        the same number of rows as matrix A.
    arbitrary_matrix : Matrix
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of an arbitrary
        matrix.  This parameter may be set to a specific matrix to use
        for that purpose; if so, it must be the same shape as x, with as
        many rows as matrix A has columns, and as many columns as matrix
        B.  If left as None, an appropriate matrix containing dummy
        symbols in the form of ``wn_m`` will be used, with n and m being
        row and column position of each symbol.

    Returns
    =======

    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> B = Matrix([7, 8])
    >>> A.pinv_solve(B)
    Matrix([
    [ _w0_0/6 - _w1_0/3 + _w2_0/6 - 55/18],
    [-_w0_0/3 + 2*_w1_0/3 - _w2_0/3 + 1/9],
    [ _w0_0/6 - _w1_0/3 + _w2_0/6 + 59/18]])
    >>> A.pinv_solve(B, arbitrary_matrix=Matrix([0, 0, 0]))
    Matrix([
    [-55/18],
    [   1/9],
    [ 59/18]])

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv

    Notes
    =====

    This may return either exact solutions or least squares solutions.
    To determine which, check ``A * A.pinv() * B == B``.  It will be
    True if exact solutions exist, and False if only a least-squares
    solution exists.  Be aware that the left hand side of that equation
    may need to be simplified to correctly compare to the right hand
    side.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse#Obtaining_all_solutions_of_a_linear_system

    """

    from sympy.matrices import eye

    A      = M
    A_pinv = M.pinv()

    if arbitrary_matrix is None:
        rows, cols       = A.cols, B.cols
        w                = symbols('w:{}_:{}'.format(rows, cols), cls=Dummy)
        arbitrary_matrix = M.__class__(cols, rows, w).T

    return A_pinv.multiply(B) + (eye(A.cols) -
            A_pinv.multiply(A)).multiply(arbitrary_matrix)


def _cramer_solve(M, rhs, det_method="laplace"):
    """Solves system of linear equations using Cramer's rule.

    This method is relatively inefficient compared to other methods.
    However it only uses a single division, assuming a division-free determinant
    method is provided. This is helpful to minimize the chance of divide-by-zero
    cases in symbolic solutions to linear systems.

    Parameters
    ==========
    M : Matrix
        The matrix representing the left hand side of the equation.
    rhs : Matrix
        The matrix representing the right hand side of the equation.
    det_method : str or callable
        The method to use to calculate the determinant of the matrix.
        The default is ``'laplace'``.  If a callable is passed, it should take a
        single argument, the matrix, and return the determinant of the matrix.

    Returns
    =======
    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[0, -6, 1], [0, -6, -1], [-5, -2, 3]])
    >>> B = Matrix([[-30, -9], [-18, -27], [-26, 46]])
    >>> x = A.cramer_solve(B)
    >>> x
    Matrix([
    [ 0, -5],
    [ 4,  3],
    [-6,  9]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cramer%27s_rule#Explicit_formulas_for_small_systems

    """
    from .dense import zeros

    def entry(i, j):
        return rhs[i, sol] if j == col else M[i, j]

    if det_method == "bird":
        from .determinant import _det_bird
        det = _det_bird
    elif det_method == "laplace":
        from .determinant import _det_laplace
        det = _det_laplace
    elif isinstance(det_method, str):
        det = lambda matrix: matrix.det(method=det_method)
    else:
        det = det_method
    det_M = det(M)
    x = zeros(*rhs.shape)
    for sol in range(rhs.shape[1]):
        for col in range(rhs.shape[0]):
            x[col, sol] = det(M.__class__(*M.shape, entry)) / det_M
    return M.__class__(x)


def _solve(M, rhs, method='GJ'):
    """Solves linear equation where the unique solution exists.

    Parameters
    ==========

    rhs : Matrix
        Vector representing the right hand side of the linear equation.

    method : string, optional
        If set to ``'GJ'`` or ``'GE'``, the Gauss-Jordan elimination will be
        used, which is implemented in the routine ``gauss_jordan_solve``.

        If set to ``'LU'``, ``LUsolve`` routine will be used.

        If set to ``'QR'``, ``QRsolve`` routine will be used.

        If set to ``'PINV'``, ``pinv_solve`` routine will be used.

        If set to ``'CRAMER'``, ``cramer_solve`` routine will be used.

        It also supports the methods available for special linear systems

        For positive definite systems:

        If set to ``'CH'``, ``cholesky_solve`` routine will be used.

        If set to ``'LDL'``, ``LDLsolve`` routine will be used.

        To use a different method and to compute the solution via the
        inverse, use a method defined in the .inv() docstring.

    Returns
    =======

    solutions : Matrix
        Vector representing the solution.

    Raises
    ======

    ValueError
        If there is not a unique solution then a ``ValueError`` will be
        raised.

        If ``M`` is not square, a ``ValueError`` and a different routine
        for solving the system will be suggested.
    """

    if method in ('GJ', 'GE'):
        try:
            soln, param = M.gauss_jordan_solve(rhs)

            if param:
                raise NonInvertibleMatrixError("Matrix det == 0; not invertible. "
                "Try ``M.gauss_jordan_solve(rhs)`` to obtain a parametric solution.")

        except ValueError:
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

        return soln

    elif method == 'LU':
        return M.LUsolve(rhs)
    elif method == 'CH':
        return M.cholesky_solve(rhs)
    elif method == 'QR':
        return M.QRsolve(rhs)
    elif method == 'LDL':
        return M.LDLsolve(rhs)
    elif method == 'PINV':
        return M.pinv_solve(rhs)
    elif method == 'CRAMER':
        return M.cramer_solve(rhs)
    else:
        return M.inv(method=method).multiply(rhs)


def _solve_least_squares(M, rhs, method='CH'):
    """Return the least-square fit to the data.

    Parameters
    ==========

    rhs : Matrix
        Vector representing the right hand side of the linear equation.

    method : string or boolean, optional
        If set to ``'CH'``, ``cholesky_solve`` routine will be used.

        If set to ``'LDL'``, ``LDLsolve`` routine will be used.

        If set to ``'QR'``, ``QRsolve`` routine will be used.

        If set to ``'PINV'``, ``pinv_solve`` routine will be used.

        Otherwise, the conjugate of ``M`` will be used to create a system
        of equations that is passed to ``solve`` along with the hint
        defined by ``method``.

    Returns
    =======

    solutions : Matrix
        Vector representing the solution.

    Examples
    ========

    >>> from sympy import Matrix, ones
    >>> A = Matrix([1, 2, 3])
    >>> B = Matrix([2, 3, 4])
    >>> S = Matrix(A.row_join(B))
    >>> S
    Matrix([
    [1, 2],
    [2, 3],
    [3, 4]])

    If each line of S represent coefficients of Ax + By
    and x and y are [2, 3] then S*xy is:

    >>> r = S*Matrix([2, 3]); r
    Matrix([
    [ 8],
    [13],
    [18]])

    But let's add 1 to the middle value and then solve for the
    least-squares value of xy:

    >>> xy = S.solve_least_squares(Matrix([8, 14, 18])); xy
    Matrix([
    [ 5/3],
    [10/3]])

    The error is given by S*xy - r:

    >>> S*xy - r
    Matrix([
    [1/3],
    [1/3],
    [1/3]])
    >>> _.norm().n(2)
    0.58

    If a different xy is used, the norm will be higher:

    >>> xy += ones(2, 1)/10
    >>> (S*xy - r).norm().n(2)
    1.5

    """

    if method == 'CH':
        return M.cholesky_solve(rhs)
    elif method == 'QR':
        return M.QRsolve(rhs)
    elif method == 'LDL':
        return M.LDLsolve(rhs)
    elif method == 'PINV':
        return M.pinv_solve(rhs)
    else:
        t = M.H
        return (t * M).solve(t * rhs, method=method)
