from sympy.core.basic import Basic
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.exceptions import NonSquareMatrixError


class Trace(Expr):
    """Matrix Trace

    Represents the trace of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Trace, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Trace(A)
    Trace(A)
    >>> Trace(eye(3))
    Trace(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> Trace(eye(3)).simplify()
    3
    """
    is_Trace = True
    is_commutative = True

    def __new__(cls, mat):
        mat = sympify(mat)

        if not mat.is_Matrix:
            raise TypeError("input to Trace, %s, is not a matrix" % str(mat))

        if mat.is_square is False:
            raise NonSquareMatrixError("Trace of a non-square matrix")

        return Basic.__new__(cls, mat)

    def _eval_transpose(self):
        return self

    def _eval_derivative(self, v):
        from sympy.concrete.summations import Sum
        from .matexpr import MatrixElement
        if isinstance(v, MatrixElement):
            return self.rewrite(Sum).diff(v)
        expr = self.doit()
        if isinstance(expr, Trace):
            # Avoid looping infinitely:
            raise NotImplementedError
        return expr._eval_derivative(v)

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayContraction
        r = self.args[0]._eval_derivative_matrix_lines(x)
        for lr in r:
            if lr.higher == 1:
                lr.higher = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                lr._lines[0],
                                lr._lines[1],
                            ]
                        ),
                        (1, 3),
                    ],
                    validator=ArrayContraction._validate
                )
            else:
                # This is not a matrix line:
                lr.higher = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                lr._lines[0],
                                lr._lines[1],
                                lr.higher,
                            ]
                        ),
                        (1, 3), (0, 2)
                    ]
                )
            lr._lines = [S.One, S.One]
            lr._first_pointer_parent = lr._lines
            lr._second_pointer_parent = lr._lines
            lr._first_pointer_index = 0
            lr._second_pointer_index = 1
        return r

    @property
    def arg(self):
        return self.args[0]

    def doit(self, **hints):
        if hints.get('deep', True):
            arg = self.arg.doit(**hints)
            result = arg._eval_trace()
            if result is not None:
                return result
            else:
                return Trace(arg)
        else:
            # _eval_trace would go too deep here
            if isinstance(self.arg, MatrixBase):
                return trace(self.arg)
            else:
                return Trace(self.arg)

    def as_explicit(self):
        return Trace(self.arg.as_explicit()).doit()

    def _normalize(self):
        # Normalization of trace of matrix products. Use transposition and
        # cyclic properties of traces to make sure the arguments of the matrix
        # product are sorted and the first argument is not a transposition.
        from sympy.matrices.expressions.matmul import MatMul
        from sympy.matrices.expressions.transpose import Transpose
        trace_arg = self.arg
        if isinstance(trace_arg, MatMul):

            def get_arg_key(x):
                a = trace_arg.args[x]
                if isinstance(a, Transpose):
                    a = a.arg
                return default_sort_key(a)

            indmin = min(range(len(trace_arg.args)), key=get_arg_key)
            if isinstance(trace_arg.args[indmin], Transpose):
                trace_arg = Transpose(trace_arg).doit()
                indmin = min(range(len(trace_arg.args)), key=lambda x: default_sort_key(trace_arg.args[x]))
            trace_arg = MatMul.fromiter(trace_arg.args[indmin:] + trace_arg.args[:indmin])
            return Trace(trace_arg)
        return self

    def _eval_rewrite_as_Sum(self, expr, **kwargs):
        from sympy.concrete.summations import Sum
        i = uniquely_named_symbol('i', [expr])
        s = Sum(self.arg[i, i], (i, 0, self.arg.rows - 1))
        return s.doit()


def trace(expr):
    """Trace of a Matrix.  Sum of the diagonal elements.

    Examples
    ========

    >>> from sympy import trace, Symbol, MatrixSymbol, eye
    >>> n = Symbol('n')
    >>> X = MatrixSymbol('X', n, n)  # A square matrix
    >>> trace(2*X)
    2*Trace(X)
    >>> trace(eye(3))
    3
    """
    return Trace(expr).doit()
