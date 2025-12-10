"""
Additional AST nodes for operations on matrices. The nodes in this module
are meant to represent optimization of matrix expressions within codegen's
target languages that cannot be represented by SymPy expressions.

As an example, we can use :meth:`sympy.codegen.rewriting.optimize` and the
``matin_opt`` optimization provided in :mod:`sympy.codegen.rewriting` to
transform matrix multiplication under certain assumptions:

    >>> from sympy import symbols, MatrixSymbol
    >>> n = symbols('n', integer=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> x = MatrixSymbol('x', n, 1)
    >>> expr = A**(-1) * x
    >>> from sympy import assuming, Q
    >>> from sympy.codegen.rewriting import matinv_opt, optimize
    >>> with assuming(Q.fullrank(A)):
    ...     optimize(expr, [matinv_opt])
    MatrixSolve(A, vector=x)
"""

from .ast import Token
from sympy.matrices import MatrixExpr
from sympy.core.sympify import sympify


class MatrixSolve(Token, MatrixExpr):
    """Represents an operation to solve a linear matrix equation.

    Parameters
    ==========

    matrix : MatrixSymbol

      Matrix representing the coefficients of variables in the linear
      equation. This matrix must be square and full-rank (i.e. all columns must
      be linearly independent) for the solving operation to be valid.

    vector : MatrixSymbol

      One-column matrix representing the solutions to the equations
      represented in ``matrix``.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol
    >>> from sympy.codegen.matrix_nodes import MatrixSolve
    >>> n = symbols('n', integer=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> x = MatrixSymbol('x', n, 1)
    >>> from sympy.printing.numpy import NumPyPrinter
    >>> NumPyPrinter().doprint(MatrixSolve(A, x))
    'numpy.linalg.solve(A, x)'
    >>> from sympy import octave_code
    >>> octave_code(MatrixSolve(A, x))
    'A \\\\ x'

    """
    __slots__ = _fields = ('matrix', 'vector')

    _construct_matrix = staticmethod(sympify)
    _construct_vector = staticmethod(sympify)

    @property
    def shape(self):
        return self.vector.shape

    def _eval_derivative(self, x):
        A, b = self.matrix, self.vector
        return MatrixSolve(A, b.diff(x) - A.diff(x) * MatrixSolve(A, b))
