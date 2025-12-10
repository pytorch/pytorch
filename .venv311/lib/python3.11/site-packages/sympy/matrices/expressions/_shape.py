from sympy.core.relational import Eq
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.logic.boolalg import Boolean, And
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.exceptions import ShapeError
from typing import Union


def is_matadd_valid(*args: MatrixExpr) -> Boolean:
    """Return the symbolic condition how ``MatAdd``, ``HadamardProduct``
    makes sense.

    Parameters
    ==========

    args
        The list of arguments of matrices to be tested for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_matadd_valid

    >>> m, n, p, q = symbols('m n p q')
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', p, q)
    >>> is_matadd_valid(A, B)
    Eq(m, p) & Eq(n, q)
    """
    rows, cols = zip(*(arg.shape for arg in args))
    return And(
        *(Eq(i, j) for i, j in zip(rows[:-1], rows[1:])),
        *(Eq(i, j) for i, j in zip(cols[:-1], cols[1:])),
    )


def is_matmul_valid(*args: Union[MatrixExpr, Expr]) -> Boolean:
    """Return the symbolic condition how ``MatMul`` makes sense

    Parameters
    ==========

    args
        The list of arguments of matrices and scalar expressions to be tested
        for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_matmul_valid

    >>> m, n, p, q = symbols('m n p q')
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', p, q)
    >>> is_matmul_valid(A, B)
    Eq(n, p)
    """
    rows, cols = zip(*(arg.shape for arg in args if isinstance(arg, MatrixExpr)))
    return And(*(Eq(i, j) for i, j in zip(cols[:-1], rows[1:])))


def is_square(arg: MatrixExpr, /) -> Boolean:
    """Return the symbolic condition how the matrix is assumed to be square

    Parameters
    ==========

    arg
        The matrix to be tested for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_square

    >>> m, n = symbols('m n')
    >>> A = MatrixSymbol('A', m, n)
    >>> is_square(A)
    Eq(m, n)
    """
    return Eq(arg.rows, arg.cols)


def validate_matadd_integer(*args: MatrixExpr) -> None:
    """Validate matrix shape for addition only for integer values"""
    rows, cols = zip(*(x.shape for x in args))
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), rows))) > 1:
        raise ShapeError(f"Matrices have mismatching shape: {rows}")
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), cols))) > 1:
        raise ShapeError(f"Matrices have mismatching shape: {cols}")


def validate_matmul_integer(*args: MatrixExpr) -> None:
    """Validate matrix shape for multiplication only for integer values"""
    for A, B in zip(args[:-1], args[1:]):
        i, j = A.cols, B.rows
        if isinstance(i, (int, Integer)) and isinstance(j, (int, Integer)) and i != j:
            raise ShapeError("Matrices are not aligned", i, j)
