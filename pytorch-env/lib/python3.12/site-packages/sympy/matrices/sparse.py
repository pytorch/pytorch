from collections.abc import Callable

from sympy.core.containers import Dict
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int

from .matrixbase import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix

from .utilities import _iszero

from .decompositions import (
    _liupc, _row_structure_symbolic_cholesky, _cholesky_sparse,
    _LDLdecomposition_sparse)

from .solvers import (
    _lower_triangular_solve_sparse, _upper_triangular_solve_sparse)


class SparseRepMatrix(RepMatrix):
    """
    A sparse matrix (a matrix with a large number of zero elements).

    Examples
    ========

    >>> from sympy import SparseMatrix, ones
    >>> SparseMatrix(2, 2, range(4))
    Matrix([
    [0, 1],
    [2, 3]])
    >>> SparseMatrix(2, 2, {(1, 1): 2})
    Matrix([
    [0, 0],
    [0, 2]])

    A SparseMatrix can be instantiated from a ragged list of lists:

    >>> SparseMatrix([[1, 2, 3], [1, 2], [1]])
    Matrix([
    [1, 2, 3],
    [1, 2, 0],
    [1, 0, 0]])

    For safety, one may include the expected size and then an error
    will be raised if the indices of any element are out of range or
    (for a flat list) if the total number of elements does not match
    the expected shape:

    >>> SparseMatrix(2, 2, [1, 2])
    Traceback (most recent call last):
    ...
    ValueError: List length (2) != rows*columns (4)

    Here, an error is not raised because the list is not flat and no
    element is out of range:

    >>> SparseMatrix(2, 2, [[1, 2]])
    Matrix([
    [1, 2],
    [0, 0]])

    But adding another element to the first (and only) row will cause
    an error to be raised:

    >>> SparseMatrix(2, 2, [[1, 2, 3]])
    Traceback (most recent call last):
    ...
    ValueError: The location (0, 2) is out of designated range: (1, 1)

    To autosize the matrix, pass None for rows:

    >>> SparseMatrix(None, [[1, 2, 3]])
    Matrix([[1, 2, 3]])
    >>> SparseMatrix(None, {(1, 1): 1, (3, 3): 3})
    Matrix([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 3]])

    Values that are themselves a Matrix are automatically expanded:

    >>> SparseMatrix(4, 4, {(1, 1): ones(2)})
    Matrix([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]])

    A ValueError is raised if the expanding matrix tries to overwrite
    a different element already present:

    >>> SparseMatrix(3, 3, {(0, 0): ones(2), (1, 1): 2})
    Traceback (most recent call last):
    ...
    ValueError: collision at (1, 1)

    See Also
    ========
    DenseMatrix
    MutableSparseMatrix
    ImmutableSparseMatrix
    """

    @classmethod
    def _handle_creation_inputs(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], MatrixBase):
            rows = args[0].rows
            cols = args[0].cols
            smat = args[0].todok()
            return rows, cols, smat

        smat = {}
        # autosizing
        if len(args) == 2 and args[0] is None:
            args = [None, None, args[1]]

        if len(args) == 3:
            r, c = args[:2]
            if r is c is None:
                rows = cols = None
            elif None in (r, c):
                raise ValueError(
                    'Pass rows=None and no cols for autosizing.')
            else:
                rows, cols = as_int(args[0]), as_int(args[1])

            if isinstance(args[2], Callable):
                op = args[2]

                if None in (rows, cols):
                    raise ValueError(
                        "{} and {} must be integers for this "
                        "specification.".format(rows, cols))

                row_indices = [cls._sympify(i) for i in range(rows)]
                col_indices = [cls._sympify(j) for j in range(cols)]

                for i in row_indices:
                    for j in col_indices:
                        value = cls._sympify(op(i, j))
                        if value != cls.zero:
                            smat[i, j] = value

                return rows, cols, smat

            elif isinstance(args[2], (dict, Dict)):
                def update(i, j, v):
                    # update smat and make sure there are no collisions
                    if v:
                        if (i, j) in smat and v != smat[i, j]:
                            raise ValueError(
                                "There is a collision at {} for {} and {}."
                                .format((i, j), v, smat[i, j])
                            )
                        smat[i, j] = v

                # manual copy, copy.deepcopy() doesn't work
                for (r, c), v in args[2].items():
                    if isinstance(v, MatrixBase):
                        for (i, j), vv in v.todok().items():
                            update(r + i, c + j, vv)
                    elif isinstance(v, (list, tuple)):
                        _, _, smat = cls._handle_creation_inputs(v, **kwargs)
                        for i, j in smat:
                            update(r + i, c + j, smat[i, j])
                    else:
                        v = cls._sympify(v)
                        update(r, c, cls._sympify(v))

            elif is_sequence(args[2]):
                flat = not any(is_sequence(i) for i in args[2])
                if not flat:
                    _, _, smat = \
                        cls._handle_creation_inputs(args[2], **kwargs)
                else:
                    flat_list = args[2]
                    if len(flat_list) != rows * cols:
                        raise ValueError(
                            "The length of the flat list ({}) does not "
                            "match the specified size ({} * {})."
                            .format(len(flat_list), rows, cols)
                        )

                    for i in range(rows):
                        for j in range(cols):
                            value = flat_list[i*cols + j]
                            value = cls._sympify(value)
                            if value != cls.zero:
                                smat[i, j] = value

            if rows is None:  # autosizing
                keys = smat.keys()
                rows = max(r for r, _ in keys) + 1 if keys else 0
                cols = max(c for _, c in keys) + 1 if keys else 0

            else:
                for i, j in smat.keys():
                    if i and i >= rows or j and j >= cols:
                        raise ValueError(
                            "The location {} is out of the designated range"
                            "[{}, {}]x[{}, {}]"
                            .format((i, j), 0, rows - 1, 0, cols - 1)
                        )

            return rows, cols, smat

        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            # list of values or lists
            v = args[0]
            c = 0
            for i, row in enumerate(v):
                if not isinstance(row, (list, tuple)):
                    row = [row]
                for j, vv in enumerate(row):
                    if vv != cls.zero:
                        smat[i, j] = cls._sympify(vv)
                c = max(c, len(row))
            rows = len(v) if c else 0
            cols = c
            return rows, cols, smat

        else:
            # handle full matrix forms with _handle_creation_inputs
            rows, cols, mat = super()._handle_creation_inputs(*args)
            for i in range(rows):
                for j in range(cols):
                    value = mat[cols*i + j]
                    if value != cls.zero:
                        smat[i, j] = value

            return rows, cols, smat

    @property
    def _smat(self):

        sympy_deprecation_warning(
            """
            The private _smat attribute of SparseMatrix is deprecated. Use the
            .todok() method instead.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-private-matrix-attributes"
        )

        return self.todok()

    def _eval_inverse(self, **kwargs):
        return self.inv(method=kwargs.get('method', 'LDL'),
                        iszerofunc=kwargs.get('iszerofunc', _iszero),
                        try_block_diag=kwargs.get('try_block_diag', False))

    def applyfunc(self, f):
        """Apply a function to each element of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> m = SparseMatrix(2, 2, lambda i, j: i*2+j)
        >>> m
        Matrix([
        [0, 1],
        [2, 3]])
        >>> m.applyfunc(lambda i: 2*i)
        Matrix([
        [0, 2],
        [4, 6]])

        """
        if not callable(f):
            raise TypeError("`f` must be callable.")

        # XXX: This only applies the function to the nonzero elements of the
        # matrix so is inconsistent with DenseMatrix.applyfunc e.g.
        #   zeros(2, 2).applyfunc(lambda x: x + 1)
        dok = {}
        for k, v in self.todok().items():
            fv = f(v)
            if fv != 0:
                dok[k] = fv

        return self._new(self.rows, self.cols, dok)

    def as_immutable(self):
        """Returns an Immutable version of this Matrix."""
        from .immutable import ImmutableSparseMatrix
        return ImmutableSparseMatrix(self)

    def as_mutable(self):
        """Returns a mutable version of this matrix.

        Examples
        ========

        >>> from sympy import ImmutableMatrix
        >>> X = ImmutableMatrix([[1, 2], [3, 4]])
        >>> Y = X.as_mutable()
        >>> Y[1, 1] = 5 # Can set values in Y
        >>> Y
        Matrix([
        [1, 2],
        [3, 5]])
        """
        return MutableSparseMatrix(self)

    def col_list(self):
        """Returns a column-sorted list of non-zero elements of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a=SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.CL
        [(0, 0, 1), (1, 0, 3), (0, 1, 2), (1, 1, 4)]

        See Also
        ========

        sympy.matrices.sparse.SparseMatrix.row_list
        """
        return [tuple(k + (self[k],)) for k in sorted(self.todok().keys(), key=lambda k: list(reversed(k)))]

    def nnz(self):
        """Returns the number of non-zero elements in Matrix."""
        return len(self.todok())

    def row_list(self):
        """Returns a row-sorted list of non-zero elements of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a = SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.RL
        [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)]

        See Also
        ========

        sympy.matrices.sparse.SparseMatrix.col_list
        """
        return [tuple(k + (self[k],)) for k in
            sorted(self.todok().keys(), key=list)]

    def scalar_multiply(self, scalar):
        "Scalar element-wise multiplication"
        return scalar * self

    def solve_least_squares(self, rhs, method='LDL'):
        """Return the least-square fit to the data.

        By default the cholesky_solve routine is used (method='CH'); other
        methods of matrix inversion can be used. To find out which are
        available, see the docstring of the .inv() method.

        Examples
        ========

        >>> from sympy import SparseMatrix, Matrix, ones
        >>> A = Matrix([1, 2, 3])
        >>> B = Matrix([2, 3, 4])
        >>> S = SparseMatrix(A.row_join(B))
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
        t = self.T
        return (t*self).inv(method=method)*t*rhs

    def solve(self, rhs, method='LDL'):
        """Return solution to self*soln = rhs using given inversion method.

        For a list of possible inversion methods, see the .inv() docstring.
        """
        if not self.is_square:
            if self.rows < self.cols:
                raise ValueError('Under-determined system.')
            elif self.rows > self.cols:
                raise ValueError('For over-determined system, M, having '
                    'more rows than columns, try M.solve_least_squares(rhs).')
        else:
            return self.inv(method=method).multiply(rhs)

    RL = property(row_list, None, None, "Alternate faster representation")
    CL = property(col_list, None, None, "Alternate faster representation")

    def liupc(self):
        return _liupc(self)

    def row_structure_symbolic_cholesky(self):
        return _row_structure_symbolic_cholesky(self)

    def cholesky(self, hermitian=True):
        return _cholesky_sparse(self, hermitian=hermitian)

    def LDLdecomposition(self, hermitian=True):
        return _LDLdecomposition_sparse(self, hermitian=hermitian)

    def lower_triangular_solve(self, rhs):
        return _lower_triangular_solve_sparse(self, rhs)

    def upper_triangular_solve(self, rhs):
        return _upper_triangular_solve_sparse(self, rhs)

    liupc.__doc__                           = _liupc.__doc__
    row_structure_symbolic_cholesky.__doc__ = _row_structure_symbolic_cholesky.__doc__
    cholesky.__doc__                        = _cholesky_sparse.__doc__
    LDLdecomposition.__doc__                = _LDLdecomposition_sparse.__doc__
    lower_triangular_solve.__doc__          = lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__          = upper_triangular_solve.__doc__


class MutableSparseMatrix(SparseRepMatrix, MutableRepMatrix):

    @classmethod
    def _new(cls, *args, **kwargs):
        rows, cols, smat = cls._handle_creation_inputs(*args, **kwargs)

        rep = cls._smat_to_DomainMatrix(rows, cols, smat)

        return cls._fromrep(rep)


SparseMatrix = MutableSparseMatrix
