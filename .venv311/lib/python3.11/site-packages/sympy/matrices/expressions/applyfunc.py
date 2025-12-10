from sympy.core.expr import ExprBuilder
from sympy.core.function import (Function, FunctionClass, Lambda)
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.matrices.matrixbase import MatrixBase


class ElementwiseApplyFunction(MatrixExpr):
    r"""
    Apply function to a matrix elementwise without evaluating.

    Examples
    ========

    It can be created by calling ``.applyfunc(<function>)`` on a matrix
    expression:

    >>> from sympy import MatrixSymbol
    >>> from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
    >>> from sympy import exp
    >>> X = MatrixSymbol("X", 3, 3)
    >>> X.applyfunc(exp)
    Lambda(_d, exp(_d)).(X)

    Otherwise using the class constructor:

    >>> from sympy import eye
    >>> expr = ElementwiseApplyFunction(exp, eye(3))
    >>> expr
    Lambda(_d, exp(_d)).(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> expr.doit()
    Matrix([
    [E, 1, 1],
    [1, E, 1],
    [1, 1, E]])

    Notice the difference with the real mathematical functions:

    >>> exp(eye(3))
    Matrix([
    [E, 0, 0],
    [0, E, 0],
    [0, 0, E]])
    """

    def __new__(cls, function, expr):
        expr = _sympify(expr)
        if not expr.is_Matrix:
            raise ValueError("{} must be a matrix instance.".format(expr))

        if expr.shape == (1, 1):
            # Check if the function returns a matrix, in that case, just apply
            # the function instead of creating an ElementwiseApplyFunc object:
            ret = function(expr)
            if isinstance(ret, MatrixExpr):
                return ret

        if not isinstance(function, (FunctionClass, Lambda)):
            d = Dummy('d')
            function = Lambda(d, function(d))

        function = sympify(function)
        if not isinstance(function, (FunctionClass, Lambda)):
            raise ValueError(
                "{} should be compatible with SymPy function classes."
                .format(function))

        if 1 not in function.nargs:
            raise ValueError(
                '{} should be able to accept 1 arguments.'.format(function))

        if not isinstance(function, Lambda):
            d = Dummy('d')
            function = Lambda(d, function(d))

        obj = MatrixExpr.__new__(cls, function, expr)
        return obj

    @property
    def function(self):
        return self.args[0]

    @property
    def expr(self):
        return self.args[1]

    @property
    def shape(self):
        return self.expr.shape

    def doit(self, **hints):
        deep = hints.get("deep", True)
        expr = self.expr
        if deep:
            expr = expr.doit(**hints)
        function = self.function
        if isinstance(function, Lambda) and function.is_identity:
            # This is a Lambda containing the identity function.
            return expr
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(self.function)
        elif isinstance(expr, ElementwiseApplyFunction):
            return ElementwiseApplyFunction(
                lambda x: self.function(expr.function(x)),
                expr.expr
            ).doit(**hints)
        else:
            return self

    def _entry(self, i, j, **kwargs):
        return self.function(self.expr._entry(i, j, **kwargs))

    def _get_function_fdiff(self):
        d = Dummy("d")
        function = self.function(d)
        fdiff = function.diff(d)
        if isinstance(fdiff, Function):
            fdiff = type(fdiff)
        else:
            fdiff = Lambda(d, fdiff)
        return fdiff

    def _eval_derivative(self, x):
        from sympy.matrices.expressions.hadamard import hadamard_product
        dexpr = self.expr.diff(x)
        fdiff = self._get_function_fdiff()
        return hadamard_product(
            dexpr,
            ElementwiseApplyFunction(fdiff, self.expr)
        )

    def _eval_derivative_matrix_lines(self, x):
        from sympy.matrices.expressions.special import Identity
        from sympy.tensor.array.expressions.array_expressions import ArrayContraction
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct

        fdiff = self._get_function_fdiff()
        lr = self.expr._eval_derivative_matrix_lines(x)
        ewdiff = ElementwiseApplyFunction(fdiff, self.expr)
        if 1 in x.shape:
            # Vector:
            iscolumn = self.shape[1] == 1
            for i in lr:
                if iscolumn:
                    ptr1 = i.first_pointer
                    ptr2 = Identity(self.shape[1])
                else:
                    ptr1 = Identity(self.shape[0])
                    ptr2 = i.second_pointer

                subexpr = ExprBuilder(
                    ArrayDiagonal,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                ewdiff,
                                ptr1,
                                ptr2,
                            ]
                        ),
                        (0, 2) if iscolumn else (1, 4)
                    ],
                    validator=ArrayDiagonal._validate
                )
                i._lines = [subexpr]
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 2
        else:
            # Matrix case:
            for i in lr:
                ptr1 = i.first_pointer
                ptr2 = i.second_pointer
                newptr1 = Identity(ptr1.shape[1])
                newptr2 = Identity(ptr2.shape[1])
                subexpr = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [ptr1, newptr1, ewdiff, ptr2, newptr2]
                        ),
                        (1, 2, 4),
                        (5, 7, 8),
                    ],
                    validator=ArrayContraction._validate
                )
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 4
                i._lines = [subexpr]
        return lr

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import Transpose
        return self.func(self.function, Transpose(self.expr).doit())
