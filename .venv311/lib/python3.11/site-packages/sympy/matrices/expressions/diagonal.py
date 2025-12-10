from sympy.core.sympify import _sympify

from sympy.matrices.expressions import MatrixExpr
from sympy.core import S, Eq, Ge
from sympy.core.mul import Mul
from sympy.functions.special.tensor_functions import KroneckerDelta


class DiagonalMatrix(MatrixExpr):
    """DiagonalMatrix(M) will create a matrix expression that
    behaves as though all off-diagonal elements,
    `M[i, j]` where `i != j`, are zero.

    Examples
    ========

    >>> from sympy import MatrixSymbol, DiagonalMatrix, Symbol
    >>> n = Symbol('n', integer=True)
    >>> m = Symbol('m', integer=True)
    >>> D = DiagonalMatrix(MatrixSymbol('x', 2, 3))
    >>> D[1, 2]
    0
    >>> D[1, 1]
    x[1, 1]

    The length of the diagonal -- the lesser of the two dimensions of `M` --
    is accessed through the `diagonal_length` property:

    >>> D.diagonal_length
    2
    >>> DiagonalMatrix(MatrixSymbol('x', n + 1, n)).diagonal_length
    n

    When one of the dimensions is symbolic the other will be treated as
    though it is smaller:

    >>> tall = DiagonalMatrix(MatrixSymbol('x', n, 3))
    >>> tall.diagonal_length
    3
    >>> tall[10, 1]
    0

    When the size of the diagonal is not known, a value of None will
    be returned:

    >>> DiagonalMatrix(MatrixSymbol('x', n, m)).diagonal_length is None
    True

    """
    arg = property(lambda self: self.args[0])

    shape = property(lambda self: self.arg.shape)  # type:ignore

    @property
    def diagonal_length(self):
        r, c = self.shape
        if r.is_Integer and c.is_Integer:
            m = min(r, c)
        elif r.is_Integer and not c.is_Integer:
            m = r
        elif c.is_Integer and not r.is_Integer:
            m = c
        elif r == c:
            m = r
        else:
            try:
                m = min(r, c)
            except TypeError:
                m = None
        return m

    def _entry(self, i, j, **kwargs):
        if self.diagonal_length is not None:
            if Ge(i, self.diagonal_length) is S.true:
                return S.Zero
            elif Ge(j, self.diagonal_length) is S.true:
                return S.Zero
        eq = Eq(i, j)
        if eq is S.true:
            return self.arg[i, i]
        elif eq is S.false:
            return S.Zero
        return self.arg[i, j]*KroneckerDelta(i, j)


class DiagonalOf(MatrixExpr):
    """DiagonalOf(M) will create a matrix expression that
    is equivalent to the diagonal of `M`, represented as
    a single column matrix.

    Examples
    ========

    >>> from sympy import MatrixSymbol, DiagonalOf, Symbol
    >>> n = Symbol('n', integer=True)
    >>> m = Symbol('m', integer=True)
    >>> x = MatrixSymbol('x', 2, 3)
    >>> diag = DiagonalOf(x)
    >>> diag.shape
    (2, 1)

    The diagonal can be addressed like a matrix or vector and will
    return the corresponding element of the original matrix:

    >>> diag[1, 0] == diag[1] == x[1, 1]
    True

    The length of the diagonal -- the lesser of the two dimensions of `M` --
    is accessed through the `diagonal_length` property:

    >>> diag.diagonal_length
    2
    >>> DiagonalOf(MatrixSymbol('x', n + 1, n)).diagonal_length
    n

    When only one of the dimensions is symbolic the other will be
    treated as though it is smaller:

    >>> dtall = DiagonalOf(MatrixSymbol('x', n, 3))
    >>> dtall.diagonal_length
    3

    When the size of the diagonal is not known, a value of None will
    be returned:

    >>> DiagonalOf(MatrixSymbol('x', n, m)).diagonal_length is None
    True

    """
    arg = property(lambda self: self.args[0])
    @property
    def shape(self):
        r, c = self.arg.shape
        if r.is_Integer and c.is_Integer:
            m = min(r, c)
        elif r.is_Integer and not c.is_Integer:
            m = r
        elif c.is_Integer and not r.is_Integer:
            m = c
        elif r == c:
            m = r
        else:
            try:
                m = min(r, c)
            except TypeError:
                m = None
        return m, S.One

    @property
    def diagonal_length(self):
        return self.shape[0]

    def _entry(self, i, j, **kwargs):
        return self.arg._entry(i, i, **kwargs)


class DiagMatrix(MatrixExpr):
    """
    Turn a vector into a diagonal matrix.
    """
    def __new__(cls, vector):
        vector = _sympify(vector)
        obj = MatrixExpr.__new__(cls, vector)
        shape = vector.shape
        dim = shape[1] if shape[0] == 1 else shape[0]
        if vector.shape[0] != 1:
            obj._iscolumn = True
        else:
            obj._iscolumn = False
        obj._shape = (dim, dim)
        obj._vector = vector
        return obj

    @property
    def shape(self):
        return self._shape

    def _entry(self, i, j, **kwargs):
        if self._iscolumn:
            result = self._vector._entry(i, 0, **kwargs)
        else:
            result = self._vector._entry(0, j, **kwargs)
        if i != j:
            result *= KroneckerDelta(i, j)
        return result

    def _eval_transpose(self):
        return self

    def as_explicit(self):
        from sympy.matrices.dense import diag
        return diag(*list(self._vector.as_explicit()))

    def doit(self, **hints):
        from sympy.assumptions import ask, Q
        from sympy.matrices.expressions.matmul import MatMul
        from sympy.matrices.expressions.transpose import Transpose
        from sympy.matrices.dense import eye
        from sympy.matrices.matrixbase import MatrixBase
        vector = self._vector
        # This accounts for shape (1, 1) and identity matrices, among others:
        if ask(Q.diagonal(vector)):
            return vector
        if isinstance(vector, MatrixBase):
            ret = eye(max(vector.shape))
            for i in range(ret.shape[0]):
                ret[i, i] = vector[i]
            return type(vector)(ret)
        if vector.is_MatMul:
            matrices = [arg for arg in vector.args if arg.is_Matrix]
            scalars = [arg for arg in vector.args if arg not in matrices]
            if scalars:
                return Mul.fromiter(scalars)*DiagMatrix(MatMul.fromiter(matrices).doit()).doit()
        if isinstance(vector, Transpose):
            vector = vector.arg
        return DiagMatrix(vector)


def diagonalize_vector(vector):
    return DiagMatrix(vector).doit()
