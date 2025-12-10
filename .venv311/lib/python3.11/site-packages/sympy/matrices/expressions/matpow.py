from .matexpr import MatrixExpr
from .special import Identity
from sympy.core import S
from sympy.core.expr import ExprBuilder
from sympy.core.cache import cacheit
from sympy.core.power import Pow
from sympy.core.sympify import _sympify
from sympy.matrices import MatrixBase
from sympy.matrices.exceptions import NonSquareMatrixError


class MatPow(MatrixExpr):
    def __new__(cls, base, exp, evaluate=False, **options):
        base = _sympify(base)
        if not base.is_Matrix:
            raise TypeError("MatPow base should be a matrix")

        if base.is_square is False:
            raise NonSquareMatrixError("Power of non-square matrix %s" % base)

        exp = _sympify(exp)
        obj = super().__new__(cls, base, exp)

        if evaluate:
            obj = obj.doit(deep=False)

        return obj

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

    @property
    def shape(self):
        return self.base.shape

    @cacheit
    def _get_explicit_matrix(self):
        return self.base.as_explicit()**self.exp

    def _entry(self, i, j, **kwargs):
        from sympy.matrices.expressions import MatMul
        A = self.doit()
        if isinstance(A, MatPow):
            # We still have a MatPow, make an explicit MatMul out of it.
            if A.exp.is_Integer and A.exp.is_positive:
                A = MatMul(*[A.base for k in range(A.exp)])
            elif not self._is_shape_symbolic():
                return A._get_explicit_matrix()[i, j]
            else:
                # Leave the expression unevaluated:
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)
        return A[i, j]

    def doit(self, **hints):
        if hints.get('deep', True):
            base, exp = (arg.doit(**hints) for arg in self.args)
        else:
            base, exp = self.args

        # combine all powers, e.g. (A ** 2) ** 3 -> A ** 6
        while isinstance(base, MatPow):
            exp *= base.args[1]
            base = base.args[0]

        if isinstance(base, MatrixBase):
            # Delegate
            return base ** exp

        # Handle simple cases so that _eval_power() in MatrixExpr sub-classes can ignore them
        if exp == S.One:
            return base
        if exp == S.Zero:
            return Identity(base.rows)
        if exp == S.NegativeOne:
            from sympy.matrices.expressions import Inverse
            return Inverse(base).doit(**hints)

        eval_power = getattr(base, '_eval_power', None)
        if eval_power is not None:
            return eval_power(exp)

        return MatPow(base, exp)

    def _eval_transpose(self):
        base, exp = self.args
        return MatPow(base.transpose(), exp)

    def _eval_adjoint(self):
        base, exp = self.args
        return MatPow(base.adjoint(), exp)

    def _eval_conjugate(self):
        base, exp = self.args
        return MatPow(base.conjugate(), exp)

    def _eval_derivative(self, x):
        return Pow._eval_derivative(self, x)

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayContraction
        from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
        from .matmul import MatMul
        from .inverse import Inverse
        exp = self.exp
        if self.base.shape == (1, 1) and not exp.has(x):
            lr = self.base._eval_derivative_matrix_lines(x)
            for i in lr:
                subexpr = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                Identity(1),
                                i._lines[0],
                                exp*self.base**(exp-1),
                                i._lines[1],
                                Identity(1),
                            ]
                        ),
                        (0, 3, 4), (5, 7, 8)
                    ],
                    validator=ArrayContraction._validate
                )
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 0
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 4
                i._lines = [subexpr]
            return lr
        if (exp > 0) == True:
            newexpr = MatMul.fromiter([self.base for i in range(exp)])
        elif (exp == -1) == True:
            return Inverse(self.base)._eval_derivative_matrix_lines(x)
        elif (exp < 0) == True:
            newexpr = MatMul.fromiter([Inverse(self.base) for i in range(-exp)])
        elif (exp == 0) == True:
            return self.doit()._eval_derivative_matrix_lines(x)
        else:
            raise NotImplementedError("cannot evaluate %s derived by %s" % (self, x))
        return newexpr._eval_derivative_matrix_lines(x)

    def _eval_inverse(self):
        return MatPow(self.base, -self.exp)
