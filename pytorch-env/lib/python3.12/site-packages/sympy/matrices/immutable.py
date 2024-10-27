from mpmath.matrices.matrices import _matrix

from sympy.core import Basic, Dict, Tuple
from sympy.core.numbers import Integer
from sympy.core.cache import cacheit
from sympy.core.sympify import _sympy_converter as sympify_converter, _sympify
from sympy.matrices.dense import DenseMatrix
from sympy.matrices.expressions import MatrixExpr
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.repmatrix import RepMatrix
from sympy.matrices.sparse import SparseRepMatrix
from sympy.multipledispatch import dispatch


def sympify_matrix(arg):
    return arg.as_immutable()


sympify_converter[MatrixBase] = sympify_matrix


def sympify_mpmath_matrix(arg):
    mat = [_sympify(x) for x in arg]
    return ImmutableDenseMatrix(arg.rows, arg.cols, mat)


sympify_converter[_matrix] = sympify_mpmath_matrix


class ImmutableRepMatrix(RepMatrix, MatrixExpr): # type: ignore
    """Immutable matrix based on RepMatrix

    Uses DomainMAtrix as the internal representation.
    """

    #
    # This is a subclass of RepMatrix that adds/overrides some methods to make
    # the instances Basic and immutable. ImmutableRepMatrix is a superclass for
    # both ImmutableDenseMatrix and ImmutableSparseMatrix.
    #

    def __new__(cls, *args, **kwargs):
        return cls._new(*args, **kwargs)

    __hash__ = MatrixExpr.__hash__

    def copy(self):
        return self

    @property
    def cols(self):
        return self._cols

    @property
    def rows(self):
        return self._rows

    @property
    def shape(self):
        return self._rows, self._cols

    def as_immutable(self):
        return self

    def _entry(self, i, j, **kwargs):
        return self[i, j]

    def __setitem__(self, *args):
        raise TypeError("Cannot set values of {}".format(self.__class__))

    def is_diagonalizable(self, reals_only=False, **kwargs):
        return super().is_diagonalizable(
            reals_only=reals_only, **kwargs)

    is_diagonalizable.__doc__ = SparseRepMatrix.is_diagonalizable.__doc__
    is_diagonalizable = cacheit(is_diagonalizable)

    def analytic_func(self, f, x):
        return self.as_mutable().analytic_func(f, x).as_immutable()


class ImmutableDenseMatrix(DenseMatrix, ImmutableRepMatrix):  # type: ignore
    """Create an immutable version of a matrix.

    Examples
    ========

    >>> from sympy import eye, ImmutableMatrix
    >>> ImmutableMatrix(eye(3))
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> _[0, 0] = 42
    Traceback (most recent call last):
    ...
    TypeError: Cannot set values of ImmutableDenseMatrix
    """

    # MatrixExpr is set as NotIterable, but we want explicit matrices to be
    # iterable
    _iterable = True
    _class_priority = 8
    _op_priority = 10.001

    @classmethod
    def _new(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], ImmutableDenseMatrix):
            return args[0]
        if kwargs.get('copy', True) is False:
            if len(args) != 3:
                raise TypeError("'copy=False' requires a matrix be initialized as rows,cols,[list]")
            rows, cols, flat_list = args
        else:
            rows, cols, flat_list = cls._handle_creation_inputs(*args, **kwargs)
            flat_list = list(flat_list) # create a shallow copy

        rep = cls._flat_list_to_DomainMatrix(rows, cols, flat_list)

        return cls._fromrep(rep)

    @classmethod
    def _fromrep(cls, rep):
        rows, cols = rep.shape
        flat_list = rep.to_sympy().to_list_flat()
        obj = Basic.__new__(cls,
            Integer(rows),
            Integer(cols),
            Tuple(*flat_list, sympify=False))
        obj._rows = rows
        obj._cols = cols
        obj._rep = rep
        return obj


# make sure ImmutableDenseMatrix is aliased as ImmutableMatrix
ImmutableMatrix = ImmutableDenseMatrix


class ImmutableSparseMatrix(SparseRepMatrix, ImmutableRepMatrix):  # type:ignore
    """Create an immutable version of a sparse matrix.

    Examples
    ========

    >>> from sympy import eye, ImmutableSparseMatrix
    >>> ImmutableSparseMatrix(1, 1, {})
    Matrix([[0]])
    >>> ImmutableSparseMatrix(eye(3))
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> _[0, 0] = 42
    Traceback (most recent call last):
    ...
    TypeError: Cannot set values of ImmutableSparseMatrix
    >>> _.shape
    (3, 3)
    """
    is_Matrix = True
    _class_priority = 9

    @classmethod
    def _new(cls, *args, **kwargs):
        rows, cols, smat = cls._handle_creation_inputs(*args, **kwargs)

        rep = cls._smat_to_DomainMatrix(rows, cols, smat)

        return cls._fromrep(rep)

    @classmethod
    def _fromrep(cls, rep):
        rows, cols = rep.shape
        smat = rep.to_sympy().to_dok()
        obj = Basic.__new__(cls, Integer(rows), Integer(cols), Dict(smat))
        obj._rows = rows
        obj._cols = cols
        obj._rep = rep
        return obj


@dispatch(ImmutableDenseMatrix, ImmutableDenseMatrix)
def _eval_is_eq(lhs, rhs): # noqa:F811
    """Helper method for Equality with matrices.sympy.

    Relational automatically converts matrices to ImmutableDenseMatrix
    instances, so this method only applies here.  Returns True if the
    matrices are definitively the same, False if they are definitively
    different, and None if undetermined (e.g. if they contain Symbols).
    Returning None triggers default handling of Equalities.

    """
    if lhs.shape != rhs.shape:
        return False
    return (lhs - rhs).is_zero_matrix
