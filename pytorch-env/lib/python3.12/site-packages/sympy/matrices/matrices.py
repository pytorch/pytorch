#
# A module consisting of deprecated matrix classes. New code should not be
# added here.
#
from sympy.core.basic import Basic
from sympy.core.symbol import Dummy

from .common import MatrixCommon

from .exceptions import NonSquareMatrixError

from .utilities import _iszero, _is_zero_after_expand_mul, _simplify

from .determinant import (
    _find_reasonable_pivot, _find_reasonable_pivot_naive,
    _adjugate, _charpoly, _cofactor, _cofactor_matrix, _per,
    _det, _det_bareiss, _det_berkowitz, _det_bird, _det_laplace, _det_LU,
    _minor, _minor_submatrix)

from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize

from .eigen import (
    _eigenvals, _eigenvects,
    _bidiagonalize, _bidiagonal_decomposition,
    _is_diagonalizable, _diagonalize,
    _is_positive_definite, _is_positive_semidefinite,
    _is_negative_definite, _is_negative_semidefinite, _is_indefinite,
    _jordan_form, _left_eigenvects, _singular_values)


# This class was previously defined in this module, but was moved to
# sympy.matrices.matrixbase. We import it here for backwards compatibility in
# case someone was importing it from here.
from .matrixbase import MatrixBase


__doctest_requires__ = {
    ('MatrixEigen.is_indefinite',
     'MatrixEigen.is_negative_definite',
     'MatrixEigen.is_negative_semidefinite',
     'MatrixEigen.is_positive_definite',
     'MatrixEigen.is_positive_semidefinite'): ['matplotlib'],
}


class MatrixDeterminant(MatrixCommon):
    """Provides basic matrix determinant operations. Should not be instantiated
    directly. See ``determinant.py`` for their implementations."""

    def _eval_det_bareiss(self, iszerofunc=_is_zero_after_expand_mul):
        return _det_bareiss(self, iszerofunc=iszerofunc)

    def _eval_det_berkowitz(self):
        return _det_berkowitz(self)

    def _eval_det_lu(self, iszerofunc=_iszero, simpfunc=None):
        return _det_LU(self, iszerofunc=iszerofunc, simpfunc=simpfunc)

    def _eval_det_bird(self):
        return _det_bird(self)

    def _eval_det_laplace(self):
        return _det_laplace(self)

    def _eval_determinant(self): # for expressions.determinant.Determinant
        return _det(self)

    def adjugate(self, method="berkowitz"):
        return _adjugate(self, method=method)

    def charpoly(self, x='lambda', simplify=_simplify):
        return _charpoly(self, x=x, simplify=simplify)

    def cofactor(self, i, j, method="berkowitz"):
        return _cofactor(self, i, j, method=method)

    def cofactor_matrix(self, method="berkowitz"):
        return _cofactor_matrix(self, method=method)

    def det(self, method="bareiss", iszerofunc=None):
        return _det(self, method=method, iszerofunc=iszerofunc)

    def per(self):
        return _per(self)

    def minor(self, i, j, method="berkowitz"):
        return _minor(self, i, j, method=method)

    def minor_submatrix(self, i, j):
        return _minor_submatrix(self, i, j)

    _find_reasonable_pivot.__doc__       = _find_reasonable_pivot.__doc__
    _find_reasonable_pivot_naive.__doc__ = _find_reasonable_pivot_naive.__doc__
    _eval_det_bareiss.__doc__            = _det_bareiss.__doc__
    _eval_det_berkowitz.__doc__          = _det_berkowitz.__doc__
    _eval_det_bird.__doc__            = _det_bird.__doc__
    _eval_det_laplace.__doc__            = _det_laplace.__doc__
    _eval_det_lu.__doc__                 = _det_LU.__doc__
    _eval_determinant.__doc__            = _det.__doc__
    adjugate.__doc__                     = _adjugate.__doc__
    charpoly.__doc__                     = _charpoly.__doc__
    cofactor.__doc__                     = _cofactor.__doc__
    cofactor_matrix.__doc__              = _cofactor_matrix.__doc__
    det.__doc__                          = _det.__doc__
    per.__doc__                          = _per.__doc__
    minor.__doc__                        = _minor.__doc__
    minor_submatrix.__doc__              = _minor_submatrix.__doc__


class MatrixReductions(MatrixDeterminant):
    """Provides basic matrix row/column operations. Should not be instantiated
    directly. See ``reductions.py`` for some of their implementations."""

    def echelon_form(self, iszerofunc=_iszero, simplify=False, with_pivots=False):
        return _echelon_form(self, iszerofunc=iszerofunc, simplify=simplify,
                with_pivots=with_pivots)

    @property
    def is_echelon(self):
        return _is_echelon(self)

    def rank(self, iszerofunc=_iszero, simplify=False):
        return _rank(self, iszerofunc=iszerofunc, simplify=simplify)

    def rref_rhs(self, rhs):
        """Return reduced row-echelon form of matrix, matrix showing
        rhs after reduction steps. ``rhs`` must have the same number
        of rows as ``self``.

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> r1, r2 = symbols('r1 r2')
        >>> Matrix([[1, 1], [2, 1]]).rref_rhs(Matrix([r1, r2]))
        (Matrix([
        [1, 0],
        [0, 1]]), Matrix([
        [ -r1 + r2],
        [2*r1 - r2]]))
        """
        r, _ = _rref(self.hstack(self, self.eye(self.rows), rhs))
        return r[:, :self.cols], r[:, -rhs.cols:]

    def rref(self, iszerofunc=_iszero, simplify=False, pivots=True,
            normalize_last=True):
        return _rref(self, iszerofunc=iszerofunc, simplify=simplify,
            pivots=pivots, normalize_last=normalize_last)

    echelon_form.__doc__ = _echelon_form.__doc__
    is_echelon.__doc__   = _is_echelon.__doc__
    rank.__doc__         = _rank.__doc__
    rref.__doc__         = _rref.__doc__

    def _normalize_op_args(self, op, col, k, col1, col2, error_str="col"):
        """Validate the arguments for a row/column operation.  ``error_str``
        can be one of "row" or "col" depending on the arguments being parsed."""
        if op not in ["n->kn", "n<->m", "n->n+km"]:
            raise ValueError("Unknown {} operation '{}'. Valid col operations "
                             "are 'n->kn', 'n<->m', 'n->n+km'".format(error_str, op))

        # define self_col according to error_str
        self_cols = self.cols if error_str == 'col' else self.rows

        # normalize and validate the arguments
        if op == "n->kn":
            col = col if col is not None else col1
            if col is None or k is None:
                raise ValueError("For a {0} operation 'n->kn' you must provide the "
                                 "kwargs `{0}` and `k`".format(error_str))
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))

        elif op == "n<->m":
            # we need two cols to swap. It does not matter
            # how they were specified, so gather them together and
            # remove `None`
            cols = {col, k, col1, col2}.difference([None])
            if len(cols) > 2:
                # maybe the user left `k` by mistake?
                cols = {col, col1, col2}.difference([None])
            if len(cols) != 2:
                raise ValueError("For a {0} operation 'n<->m' you must provide the "
                                 "kwargs `{0}1` and `{0}2`".format(error_str))
            col1, col2 = cols
            if not 0 <= col1 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col1))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        elif op == "n->n+km":
            col = col1 if col is None else col
            col2 = col1 if col2 is None else col2
            if col is None or col2 is None or k is None:
                raise ValueError("For a {0} operation 'n->n+km' you must provide the "
                                 "kwargs `{0}`, `k`, and `{0}2`".format(error_str))
            if col == col2:
                raise ValueError("For a {0} operation 'n->n+km' `{0}` and `{0}2` must "
                                 "be different.".format(error_str))
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        else:
            raise ValueError('invalid operation %s' % repr(op))

        return op, col, k, col1, col2

    def _eval_col_op_multiply_col_by_const(self, col, k):
        def entry(i, j):
            if j == col:
                return k * self[i, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_swap(self, col1, col2):
        def entry(i, j):
            if j == col1:
                return self[i, col2]
            elif j == col2:
                return self[i, col1]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_add_multiple_to_other_col(self, col, k, col2):
        def entry(i, j):
            if j == col:
                return self[i, j] + k * self[i, col2]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_swap(self, row1, row2):
        def entry(i, j):
            if i == row1:
                return self[row2, j]
            elif i == row2:
                return self[row1, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_multiply_row_by_const(self, row, k):
        def entry(i, j):
            if i == row:
                return k * self[i, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_add_multiple_to_other_row(self, row, k, row2):
        def entry(i, j):
            if i == row:
                return self[i, j] + k * self[row2, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def elementary_col_op(self, op="n->kn", col=None, k=None, col1=None, col2=None):
        """Performs the elementary column operation `op`.

        `op` may be one of

            * ``"n->kn"`` (column n goes to k*n)
            * ``"n<->m"`` (swap column n and column m)
            * ``"n->n+km"`` (column n goes to column n + k*column m)

        Parameters
        ==========

        op : string; the elementary row operation
        col : the column to apply the column operation
        k : the multiple to apply in the column operation
        col1 : one column of a column swap
        col2 : second column of a column swap or column "m" in the column operation
               "n->n+km"
        """

        op, col, k, col1, col2 = self._normalize_op_args(op, col, k, col1, col2, "col")

        # now that we've validated, we're all good to dispatch
        if op == "n->kn":
            return self._eval_col_op_multiply_col_by_const(col, k)
        if op == "n<->m":
            return self._eval_col_op_swap(col1, col2)
        if op == "n->n+km":
            return self._eval_col_op_add_multiple_to_other_col(col, k, col2)

    def elementary_row_op(self, op="n->kn", row=None, k=None, row1=None, row2=None):
        """Performs the elementary row operation `op`.

        `op` may be one of

            * ``"n->kn"`` (row n goes to k*n)
            * ``"n<->m"`` (swap row n and row m)
            * ``"n->n+km"`` (row n goes to row n + k*row m)

        Parameters
        ==========

        op : string; the elementary row operation
        row : the row to apply the row operation
        k : the multiple to apply in the row operation
        row1 : one row of a row swap
        row2 : second row of a row swap or row "m" in the row operation
               "n->n+km"
        """

        op, row, k, row1, row2 = self._normalize_op_args(op, row, k, row1, row2, "row")

        # now that we've validated, we're all good to dispatch
        if op == "n->kn":
            return self._eval_row_op_multiply_row_by_const(row, k)
        if op == "n<->m":
            return self._eval_row_op_swap(row1, row2)
        if op == "n->n+km":
            return self._eval_row_op_add_multiple_to_other_row(row, k, row2)


class MatrixSubspaces(MatrixReductions):
    """Provides methods relating to the fundamental subspaces of a matrix.
    Should not be instantiated directly. See ``subspaces.py`` for their
    implementations."""

    def columnspace(self, simplify=False):
        return _columnspace(self, simplify=simplify)

    def nullspace(self, simplify=False, iszerofunc=_iszero):
        return _nullspace(self, simplify=simplify, iszerofunc=iszerofunc)

    def rowspace(self, simplify=False):
        return _rowspace(self, simplify=simplify)

    # This is a classmethod but is converted to such later in order to allow
    # assignment of __doc__ since that does not work for already wrapped
    # classmethods in Python 3.6.
    def orthogonalize(cls, *vecs, **kwargs):
        return _orthogonalize(cls, *vecs, **kwargs)

    columnspace.__doc__   = _columnspace.__doc__
    nullspace.__doc__     = _nullspace.__doc__
    rowspace.__doc__      = _rowspace.__doc__
    orthogonalize.__doc__ = _orthogonalize.__doc__

    orthogonalize         = classmethod(orthogonalize)  # type:ignore


class MatrixEigen(MatrixSubspaces):
    """Provides basic matrix eigenvalue/vector operations.
    Should not be instantiated directly. See ``eigen.py`` for their
    implementations."""

    def eigenvals(self, error_when_incomplete=True, **flags):
        return _eigenvals(self, error_when_incomplete=error_when_incomplete, **flags)

    def eigenvects(self, error_when_incomplete=True, iszerofunc=_iszero, **flags):
        return _eigenvects(self, error_when_incomplete=error_when_incomplete,
                iszerofunc=iszerofunc, **flags)

    def is_diagonalizable(self, reals_only=False, **kwargs):
        return _is_diagonalizable(self, reals_only=reals_only, **kwargs)

    def diagonalize(self, reals_only=False, sort=False, normalize=False):
        return _diagonalize(self, reals_only=reals_only, sort=sort,
                normalize=normalize)

    def bidiagonalize(self, upper=True):
        return _bidiagonalize(self, upper=upper)

    def bidiagonal_decomposition(self, upper=True):
        return _bidiagonal_decomposition(self, upper=upper)

    @property
    def is_positive_definite(self):
        return _is_positive_definite(self)

    @property
    def is_positive_semidefinite(self):
        return _is_positive_semidefinite(self)

    @property
    def is_negative_definite(self):
        return _is_negative_definite(self)

    @property
    def is_negative_semidefinite(self):
        return _is_negative_semidefinite(self)

    @property
    def is_indefinite(self):
        return _is_indefinite(self)

    def jordan_form(self, calc_transform=True, **kwargs):
        return _jordan_form(self, calc_transform=calc_transform, **kwargs)

    def left_eigenvects(self, **flags):
        return _left_eigenvects(self, **flags)

    def singular_values(self):
        return _singular_values(self)

    eigenvals.__doc__                  = _eigenvals.__doc__
    eigenvects.__doc__                 = _eigenvects.__doc__
    is_diagonalizable.__doc__          = _is_diagonalizable.__doc__
    diagonalize.__doc__                = _diagonalize.__doc__
    is_positive_definite.__doc__       = _is_positive_definite.__doc__
    is_positive_semidefinite.__doc__   = _is_positive_semidefinite.__doc__
    is_negative_definite.__doc__       = _is_negative_definite.__doc__
    is_negative_semidefinite.__doc__   = _is_negative_semidefinite.__doc__
    is_indefinite.__doc__              = _is_indefinite.__doc__
    jordan_form.__doc__                = _jordan_form.__doc__
    left_eigenvects.__doc__            = _left_eigenvects.__doc__
    singular_values.__doc__            = _singular_values.__doc__
    bidiagonalize.__doc__              = _bidiagonalize.__doc__
    bidiagonal_decomposition.__doc__   = _bidiagonal_decomposition.__doc__


class MatrixCalculus(MatrixCommon):
    """Provides calculus-related matrix operations."""

    def diff(self, *args, evaluate=True, **kwargs):
        """Calculate the derivative of each element in the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.diff(x)
        Matrix([
        [1, 0],
        [0, 0]])

        See Also
        ========

        integrate
        limit
        """
        # XXX this should be handled here rather than in Derivative
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        deriv = ArrayDerivative(self, *args, evaluate=evaluate)
        # XXX This can rather changed to always return immutable matrix
        if not isinstance(self, Basic) and evaluate:
            return deriv.as_mutable()
        return deriv

    def _eval_derivative(self, arg):
        return self.applyfunc(lambda x: x.diff(arg))

    def integrate(self, *args, **kwargs):
        """Integrate each element of the matrix.  ``args`` will
        be passed to the ``integrate`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.integrate((x, ))
        Matrix([
        [x**2/2, x*y],
        [     x,   0]])
        >>> M.integrate((x, 0, 2))
        Matrix([
        [2, 2*y],
        [2,   0]])

        See Also
        ========

        limit
        diff
        """
        return self.applyfunc(lambda x: x.integrate(*args, **kwargs))

    def jacobian(self, X):
        """Calculates the Jacobian matrix (derivative of a vector-valued function).

        Parameters
        ==========

        ``self`` : vector of expressions representing functions f_i(x_1, ..., x_n).
        X : set of x_i's in order, it can be a list or a Matrix

        Both ``self`` and X can be a row or a column matrix in any order
        (i.e., jacobian() should always work).

        Examples
        ========

        >>> from sympy import sin, cos, Matrix
        >>> from sympy.abc import rho, phi
        >>> X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
        >>> Y = Matrix([rho, phi])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)],
        [   2*rho,             0]])
        >>> X = Matrix([rho*cos(phi), rho*sin(phi)])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)]])

        See Also
        ========

        hessian
        wronskian
        """
        if not isinstance(X, MatrixBase):
            X = self._new(X)
        # Both X and ``self`` can be a row or a column matrix, so we need to make
        # sure all valid combinations work, but everything else fails:
        if self.shape[0] == 1:
            m = self.shape[1]
        elif self.shape[1] == 1:
            m = self.shape[0]
        else:
            raise TypeError("``self`` must be a row or a column matrix")
        if X.shape[0] == 1:
            n = X.shape[1]
        elif X.shape[1] == 1:
            n = X.shape[0]
        else:
            raise TypeError("X must be a row or a column matrix")

        # m is the number of functions and n is the number of variables
        # computing the Jacobian is now easy:
        return self._new(m, n, lambda j, i: self[j].diff(X[i]))

    def limit(self, *args):
        """Calculate the limit of each element in the matrix.
        ``args`` will be passed to the ``limit`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.limit(x, 2)
        Matrix([
        [2, y],
        [1, 0]])

        See Also
        ========

        integrate
        diff
        """
        return self.applyfunc(lambda x: x.limit(*args))


# https://github.com/sympy/sympy/pull/12854
class MatrixDeprecated(MatrixCommon):
    """A class to house deprecated matrix methods."""
    def berkowitz_charpoly(self, x=Dummy('lambda'), simplify=_simplify):
        return self.charpoly(x=x)

    def berkowitz_det(self):
        """Computes determinant using Berkowitz method.

        See Also
        ========

        det
        berkowitz
        """
        return self.det(method='berkowitz')

    def berkowitz_eigenvals(self, **flags):
        """Computes eigenvalues of a Matrix using Berkowitz method.

        See Also
        ========

        berkowitz
        """
        return self.eigenvals(**flags)

    def berkowitz_minors(self):
        """Computes principal minors using Berkowitz method.

        See Also
        ========

        berkowitz
        """
        sign, minors = self.one, []

        for poly in self.berkowitz():
            minors.append(sign * poly[-1])
            sign = -sign

        return tuple(minors)

    def berkowitz(self):
        from sympy.matrices import zeros
        berk = ((1,),)
        if not self:
            return berk

        if not self.is_square:
            raise NonSquareMatrixError()

        A, N = self, self.rows
        transforms = [0] * (N - 1)

        for n in range(N, 1, -1):
            T, k = zeros(n + 1, n), n - 1

            R, C = -A[k, :k], A[:k, k]
            A, a = A[:k, :k], -A[k, k]

            items = [C]

            for i in range(0, n - 2):
                items.append(A * items[i])

            for i, B in enumerate(items):
                items[i] = (R * B)[0, 0]

            items = [self.one, a] + items

            for i in range(n):
                T[i:, i] = items[:n - i + 1]

            transforms[k - 1] = T

        polys = [self._new([self.one, -A[0, 0]])]

        for i, T in enumerate(transforms):
            polys.append(T * polys[i])

        return berk + tuple(map(tuple, polys))

    def cofactorMatrix(self, method="berkowitz"):
        return self.cofactor_matrix(method=method)

    def det_bareis(self):
        return _det_bareiss(self)

    def det_LU_decomposition(self):
        """Compute matrix determinant using LU decomposition.


        Note that this method fails if the LU decomposition itself
        fails. In particular, if the matrix has no inverse this method
        will fail.

        TODO: Implement algorithm for sparse matrices (SFF),
        https://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps

        See Also
        ========


        det
        det_bareiss
        berkowitz_det
        """
        return self.det(method='lu')

    def jordan_cell(self, eigenval, n):
        return self.jordan_block(size=n, eigenvalue=eigenval)

    def jordan_cells(self, calc_transformation=True):
        P, J = self.jordan_form()
        return P, J.get_diag_blocks()

    def minorEntry(self, i, j, method="berkowitz"):
        return self.minor(i, j, method=method)

    def minorMatrix(self, i, j):
        return self.minor_submatrix(i, j)

    def permuteBkwd(self, perm):
        """Permute the rows of the matrix with the given permutation in reverse."""
        return self.permute_rows(perm, direction='backward')

    def permuteFwd(self, perm):
        """Permute the rows of the matrix with the given permutation."""
        return self.permute_rows(perm, direction='forward')
