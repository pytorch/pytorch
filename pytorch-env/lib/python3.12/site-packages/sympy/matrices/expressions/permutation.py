from sympy.core import S
from sympy.core.sympify import _sympify
from sympy.functions import KroneckerDelta

from .matexpr import MatrixExpr
from .special import ZeroMatrix, Identity, OneMatrix


class PermutationMatrix(MatrixExpr):
    """A Permutation Matrix

    Parameters
    ==========

    perm : Permutation
        The permutation the matrix uses.

        The size of the permutation determines the matrix size.

        See the documentation of
        :class:`sympy.combinatorics.permutations.Permutation` for
        the further information of how to create a permutation object.

    Examples
    ========

    >>> from sympy import Matrix, PermutationMatrix
    >>> from sympy.combinatorics import Permutation

    Creating a permutation matrix:

    >>> p = Permutation(1, 2, 0)
    >>> P = PermutationMatrix(p)
    >>> P = P.as_explicit()
    >>> P
    Matrix([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]])

    Permuting a matrix row and column:

    >>> M = Matrix([0, 1, 2])
    >>> Matrix(P*M)
    Matrix([
    [1],
    [2],
    [0]])

    >>> Matrix(M.T*P)
    Matrix([[2, 0, 1]])

    See Also
    ========

    sympy.combinatorics.permutations.Permutation
    """

    def __new__(cls, perm):
        from sympy.combinatorics.permutations import Permutation

        perm = _sympify(perm)
        if not isinstance(perm, Permutation):
            raise ValueError(
                "{} must be a SymPy Permutation instance.".format(perm))

        return super().__new__(cls, perm)

    @property
    def shape(self):
        size = self.args[0].size
        return (size, size)

    @property
    def is_Identity(self):
        return self.args[0].is_Identity

    def doit(self, **hints):
        if self.is_Identity:
            return Identity(self.rows)
        return self

    def _entry(self, i, j, **kwargs):
        perm = self.args[0]
        return KroneckerDelta(perm.apply(i), j)

    def _eval_power(self, exp):
        return PermutationMatrix(self.args[0] ** exp).doit()

    def _eval_inverse(self):
        return PermutationMatrix(self.args[0] ** -1)

    _eval_transpose = _eval_adjoint = _eval_inverse

    def _eval_determinant(self):
        sign = self.args[0].signature()
        if sign == 1:
            return S.One
        elif sign == -1:
            return S.NegativeOne
        raise NotImplementedError

    def _eval_rewrite_as_BlockDiagMatrix(self, *args, **kwargs):
        from sympy.combinatorics.permutations import Permutation
        from .blockmatrix import BlockDiagMatrix

        perm = self.args[0]
        full_cyclic_form = perm.full_cyclic_form

        cycles_picks = []

        # Stage 1. Decompose the cycles into the blockable form.
        a, b, c = 0, 0, 0
        flag = False
        for cycle in full_cyclic_form:
            l = len(cycle)
            m = max(cycle)

            if not flag:
                if m + 1 > a + l:
                    flag = True
                    temp = [cycle]
                    b = m
                    c = l
                else:
                    cycles_picks.append([cycle])
                    a += l

            else:
                if m > b:
                    if m + 1 == a + c + l:
                        temp.append(cycle)
                        cycles_picks.append(temp)
                        flag = False
                        a = m+1
                    else:
                        b = m
                        temp.append(cycle)
                        c += l
                else:
                    if b + 1 == a + c + l:
                        temp.append(cycle)
                        cycles_picks.append(temp)
                        flag = False
                        a = b+1
                    else:
                        temp.append(cycle)
                        c += l

        # Stage 2. Normalize each decomposed cycles and build matrix.
        p = 0
        args = []
        for pick in cycles_picks:
            new_cycles = []
            l = 0
            for cycle in pick:
                new_cycle = [i - p for i in cycle]
                new_cycles.append(new_cycle)
                l += len(cycle)
            p += l
            perm = Permutation(new_cycles)
            mat = PermutationMatrix(perm)
            args.append(mat)

        return BlockDiagMatrix(*args)


class MatrixPermute(MatrixExpr):
    r"""Symbolic representation for permuting matrix rows or columns.

    Parameters
    ==========

    perm : Permutation, PermutationMatrix
        The permutation to use for permuting the matrix.
        The permutation can be resized to the suitable one,

    axis : 0 or 1
        The axis to permute alongside.
        If `0`, it will permute the matrix rows.
        If `1`, it will permute the matrix columns.

    Notes
    =====

    This follows the same notation used in
    :meth:`sympy.matrices.matrixbase.MatrixBase.permute`.

    Examples
    ========

    >>> from sympy import Matrix, MatrixPermute
    >>> from sympy.combinatorics import Permutation

    Permuting the matrix rows:

    >>> p = Permutation(1, 2, 0)
    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> B = MatrixPermute(A, p, axis=0)
    >>> B.as_explicit()
    Matrix([
    [4, 5, 6],
    [7, 8, 9],
    [1, 2, 3]])

    Permuting the matrix columns:

    >>> B = MatrixPermute(A, p, axis=1)
    >>> B.as_explicit()
    Matrix([
    [2, 3, 1],
    [5, 6, 4],
    [8, 9, 7]])

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.permute
    """
    def __new__(cls, mat, perm, axis=S.Zero):
        from sympy.combinatorics.permutations import Permutation

        mat = _sympify(mat)
        if not mat.is_Matrix:
            raise ValueError(
                "{} must be a SymPy matrix instance.".format(perm))

        perm = _sympify(perm)
        if isinstance(perm, PermutationMatrix):
            perm = perm.args[0]

        if not isinstance(perm, Permutation):
            raise ValueError(
                "{} must be a SymPy Permutation or a PermutationMatrix " \
                "instance".format(perm))

        axis = _sympify(axis)
        if axis not in (0, 1):
            raise ValueError("The axis must be 0 or 1.")

        mat_size = mat.shape[axis]
        if mat_size != perm.size:
            try:
                perm = perm.resize(mat_size)
            except ValueError:
                raise ValueError(
                    "Size does not match between the permutation {} "
                    "and the matrix {} threaded over the axis {} "
                    "and cannot be converted."
                    .format(perm, mat, axis))

        return super().__new__(cls, mat, perm, axis)

    def doit(self, deep=True, **hints):
        mat, perm, axis = self.args

        if deep:
            mat = mat.doit(deep=deep, **hints)
            perm = perm.doit(deep=deep, **hints)

        if perm.is_Identity:
            return mat

        if mat.is_Identity:
            if axis is S.Zero:
                return PermutationMatrix(perm)
            elif axis is S.One:
                return PermutationMatrix(perm**-1)

        if isinstance(mat, (ZeroMatrix, OneMatrix)):
            return mat

        if isinstance(mat, MatrixPermute) and mat.args[2] == axis:
            return MatrixPermute(mat.args[0], perm * mat.args[1], axis)

        return self

    @property
    def shape(self):
        return self.args[0].shape

    def _entry(self, i, j, **kwargs):
        mat, perm, axis = self.args

        if axis == 0:
            return mat[perm.apply(i), j]
        elif axis == 1:
            return mat[i, perm.apply(j)]

    def _eval_rewrite_as_MatMul(self, *args, **kwargs):
        from .matmul import MatMul

        mat, perm, axis = self.args

        deep = kwargs.get("deep", True)

        if deep:
            mat = mat.rewrite(MatMul)

        if axis == 0:
            return MatMul(PermutationMatrix(perm), mat)
        elif axis == 1:
            return MatMul(mat, PermutationMatrix(perm**-1))
