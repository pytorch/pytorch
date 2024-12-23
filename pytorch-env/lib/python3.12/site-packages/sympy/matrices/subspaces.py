from .utilities import _iszero


def _columnspace(M, simplify=False):
    """Returns a list of vectors (Matrix objects) that span columnspace of ``M``

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.columnspace()
    [Matrix([
    [ 1],
    [-2],
    [ 3]]), Matrix([
    [0],
    [0],
    [6]])]

    See Also
    ========

    nullspace
    rowspace
    """

    reduced, pivots = M.echelon_form(simplify=simplify, with_pivots=True)

    return [M.col(i) for i in pivots]


def _nullspace(M, simplify=False, iszerofunc=_iszero):
    """Returns list of vectors (Matrix objects) that span nullspace of ``M``

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.nullspace()
    [Matrix([
    [-3],
    [ 1],
    [ 0]])]

    See Also
    ========

    columnspace
    rowspace
    """

    reduced, pivots = M.rref(iszerofunc=iszerofunc, simplify=simplify)

    free_vars = [i for i in range(M.cols) if i not in pivots]
    basis     = []

    for free_var in free_vars:
        # for each free variable, we will set it to 1 and all others
        # to 0.  Then, we will use back substitution to solve the system
        vec           = [M.zero] * M.cols
        vec[free_var] = M.one

        for piv_row, piv_col in enumerate(pivots):
            vec[piv_col] -= reduced[piv_row, free_var]

        basis.append(vec)

    return [M._new(M.cols, 1, b) for b in basis]


def _rowspace(M, simplify=False):
    """Returns a list of vectors that span the row space of ``M``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.rowspace()
    [Matrix([[1, 3, 0]]), Matrix([[0, 0, 6]])]
    """

    reduced, pivots = M.echelon_form(simplify=simplify, with_pivots=True)

    return [reduced.row(i) for i in range(len(pivots))]


def _orthogonalize(cls, *vecs, normalize=False, rankcheck=False):
    """Apply the Gram-Schmidt orthogonalization procedure
    to vectors supplied in ``vecs``.

    Parameters
    ==========

    vecs
        vectors to be made orthogonal

    normalize : bool
        If ``True``, return an orthonormal basis.

    rankcheck : bool
        If ``True``, the computation does not stop when encountering
        linearly dependent vectors.

        If ``False``, it will raise ``ValueError`` when any zero
        or linearly dependent vectors are found.

    Returns
    =======

    list
        List of orthogonal (or orthonormal) basis vectors.

    Examples
    ========

    >>> from sympy import I, Matrix
    >>> v = [Matrix([1, I]), Matrix([1, -I])]
    >>> Matrix.orthogonalize(*v)
    [Matrix([
    [1],
    [I]]), Matrix([
    [ 1],
    [-I]])]

    See Also
    ========

    MatrixBase.QRdecomposition

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    """
    from .decompositions import _QRdecomposition_optional

    if not vecs:
        return []

    all_row_vecs = (vecs[0].rows == 1)

    vecs = [x.vec() for x in vecs]
    M = cls.hstack(*vecs)
    Q, R = _QRdecomposition_optional(M, normalize=normalize)

    if rankcheck and Q.cols < len(vecs):
        raise ValueError("GramSchmidt: vector set not linearly independent")

    ret = []
    for i in range(Q.cols):
        if all_row_vecs:
            col = cls(Q[:, i].T)
        else:
            col = cls(Q[:, i])
        ret.append(col)
    return ret
