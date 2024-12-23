from collections import Counter

from sympy.core import Mul, sympify
from sympy.core.add import Add
from sympy.core.expr import ExprBuilder
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import log
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix
from sympy.strategies import (
    unpack, flatten, condition, exhaust, rm_id, sort
)
from sympy.utilities.exceptions import sympy_deprecation_warning


def hadamard_product(*matrices):
    """
    Return the elementwise (aka Hadamard) product of matrices.

    Examples
    ========

    >>> from sympy import hadamard_product, MatrixSymbol
    >>> A = MatrixSymbol('A', 2, 3)
    >>> B = MatrixSymbol('B', 2, 3)
    >>> hadamard_product(A)
    A
    >>> hadamard_product(A, B)
    HadamardProduct(A, B)
    >>> hadamard_product(A, B)[0, 1]
    A[0, 1]*B[0, 1]
    """
    if not matrices:
        raise TypeError("Empty Hadamard product is undefined")
    if len(matrices) == 1:
        return matrices[0]
    return HadamardProduct(*matrices).doit()


class HadamardProduct(MatrixExpr):
    """
    Elementwise product of matrix expressions

    Examples
    ========

    Hadamard product for matrix symbols:

    >>> from sympy import hadamard_product, HadamardProduct, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> isinstance(hadamard_product(A, B), HadamardProduct)
    True

    Notes
    =====

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the product, use the function
    ``hadamard_product()`` or ``HadamardProduct.doit``
    """
    is_HadamardProduct = True

    def __new__(cls, *args, evaluate=False, check=None):
        args = list(map(sympify, args))
        if len(args) == 0:
            # We currently don't have a way to support one-matrices of generic dimensions:
            raise ValueError("HadamardProduct needs at least one argument")

        if not all(isinstance(arg, MatrixExpr) for arg in args):
            raise TypeError("Mix of Matrix and Scalar symbols")

        if check is not None:
            sympy_deprecation_warning(
                "Passing check to HadamardProduct is deprecated and the check argument will be removed in a future version.",
                deprecated_since_version="1.11",
                active_deprecations_target='remove-check-argument-from-matrix-operations')

        if check is not False:
            validate(*args)

        obj = super().__new__(cls, *args)
        if evaluate:
            obj = obj.doit(deep=False)
        return obj

    @property
    def shape(self):
        return self.args[0].shape

    def _entry(self, i, j, **kwargs):
        return Mul(*[arg._entry(i, j, **kwargs) for arg in self.args])

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return HadamardProduct(*list(map(transpose, self.args)))

    def doit(self, **hints):
        expr = self.func(*(i.doit(**hints) for i in self.args))
        # Check for explicit matrices:
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.matrices.immutable import ImmutableMatrix

        explicit = [i for i in expr.args if isinstance(i, MatrixBase)]
        if explicit:
            remainder = [i for i in expr.args if i not in explicit]
            expl_mat = ImmutableMatrix([
                Mul.fromiter(i) for i in zip(*explicit)
            ]).reshape(*self.shape)
            expr = HadamardProduct(*([expl_mat] + remainder))

        return canonicalize(expr)

    def _eval_derivative(self, x):
        terms = []
        args = list(self.args)
        for i in range(len(args)):
            factors = args[:i] + [args[i].diff(x)] + args[i+1:]
            terms.append(hadamard_product(*factors))
        return Add.fromiter(terms)

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        from sympy.matrices.expressions.matexpr import _make_matrix

        with_x_ind = [i for i, arg in enumerate(self.args) if arg.has(x)]
        lines = []
        for ind in with_x_ind:
            left_args = self.args[:ind]
            right_args = self.args[ind+1:]

            d = self.args[ind]._eval_derivative_matrix_lines(x)
            hadam = hadamard_product(*(right_args + left_args))
            diagonal = [(0, 2), (3, 4)]
            diagonal = [e for j, e in enumerate(diagonal) if self.shape[j] != 1]
            for i in d:
                l1 = i._lines[i._first_line_index]
                l2 = i._lines[i._second_line_index]
                subexpr = ExprBuilder(
                    ArrayDiagonal,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                ExprBuilder(_make_matrix, [l1]),
                                hadam,
                                ExprBuilder(_make_matrix, [l2]),
                            ]
                        ),
                    *diagonal],

                )
                i._first_pointer_parent = subexpr.args[0].args[0].args
                i._first_pointer_index = 0
                i._second_pointer_parent = subexpr.args[0].args[2].args
                i._second_pointer_index = 0
                i._lines = [subexpr]
                lines.append(i)

        return lines


# TODO Implement algorithm for rewriting Hadamard product as diagonal matrix
# if matmul identy matrix is multiplied.
def canonicalize(x):
    """Canonicalize the Hadamard product ``x`` with mathematical properties.

    Examples
    ========

    >>> from sympy import MatrixSymbol, HadamardProduct
    >>> from sympy import OneMatrix, ZeroMatrix
    >>> from sympy.matrices.expressions.hadamard import canonicalize
    >>> from sympy import init_printing
    >>> init_printing(use_unicode=False)

    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = MatrixSymbol('B', 2, 2)
    >>> C = MatrixSymbol('C', 2, 2)

    Hadamard product associativity:

    >>> X = HadamardProduct(A, HadamardProduct(B, C))
    >>> X
    A.*(B.*C)
    >>> canonicalize(X)
    A.*B.*C

    Hadamard product commutativity:

    >>> X = HadamardProduct(A, B)
    >>> Y = HadamardProduct(B, A)
    >>> X
    A.*B
    >>> Y
    B.*A
    >>> canonicalize(X)
    A.*B
    >>> canonicalize(Y)
    A.*B

    Hadamard product identity:

    >>> X = HadamardProduct(A, OneMatrix(2, 2))
    >>> X
    A.*1
    >>> canonicalize(X)
    A

    Absorbing element of Hadamard product:

    >>> X = HadamardProduct(A, ZeroMatrix(2, 2))
    >>> X
    A.*0
    >>> canonicalize(X)
    0

    Rewriting to Hadamard Power

    >>> X = HadamardProduct(A, A, A)
    >>> X
    A.*A.*A
    >>> canonicalize(X)
     .3
    A

    Notes
    =====

    As the Hadamard product is associative, nested products can be flattened.

    The Hadamard product is commutative so that factors can be sorted for
    canonical form.

    A matrix of only ones is an identity for Hadamard product,
    so every matrices of only ones can be removed.

    Any zero matrix will make the whole product a zero matrix.

    Duplicate elements can be collected and rewritten as HadamardPower

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
    """
    # Associativity
    rule = condition(
            lambda x: isinstance(x, HadamardProduct),
            flatten
        )
    fun = exhaust(rule)
    x = fun(x)

    # Identity
    fun = condition(
            lambda x: isinstance(x, HadamardProduct),
            rm_id(lambda x: isinstance(x, OneMatrix))
        )
    x = fun(x)

    # Absorbing by Zero Matrix
    def absorb(x):
        if any(isinstance(c, ZeroMatrix) for c in x.args):
            return ZeroMatrix(*x.shape)
        else:
            return x
    fun = condition(
            lambda x: isinstance(x, HadamardProduct),
            absorb
        )
    x = fun(x)

    # Rewriting with HadamardPower
    if isinstance(x, HadamardProduct):
        tally = Counter(x.args)

        new_arg = []
        for base, exp in tally.items():
            if exp == 1:
                new_arg.append(base)
            else:
                new_arg.append(HadamardPower(base, exp))

        x = HadamardProduct(*new_arg)

    # Commutativity
    fun = condition(
            lambda x: isinstance(x, HadamardProduct),
            sort(default_sort_key)
        )
    x = fun(x)

    # Unpacking
    x = unpack(x)
    return x


def hadamard_power(base, exp):
    base = sympify(base)
    exp = sympify(exp)
    if exp == 1:
        return base
    if not base.is_Matrix:
        return base**exp
    if exp.is_Matrix:
        raise ValueError("cannot raise expression to a matrix")
    return HadamardPower(base, exp)


class HadamardPower(MatrixExpr):
    r"""
    Elementwise power of matrix expressions

    Parameters
    ==========

    base : scalar or matrix

    exp : scalar or matrix

    Notes
    =====

    There are four definitions for the hadamard power which can be used.
    Let's consider `A, B` as `(m, n)` matrices, and `a, b` as scalars.

    Matrix raised to a scalar exponent:

    .. math::
        A^{\circ b} = \begin{bmatrix}
        A_{0, 0}^b   & A_{0, 1}^b   & \cdots & A_{0, n-1}^b   \\
        A_{1, 0}^b   & A_{1, 1}^b   & \cdots & A_{1, n-1}^b   \\
        \vdots       & \vdots       & \ddots & \vdots         \\
        A_{m-1, 0}^b & A_{m-1, 1}^b & \cdots & A_{m-1, n-1}^b
        \end{bmatrix}

    Scalar raised to a matrix exponent:

    .. math::
        a^{\circ B} = \begin{bmatrix}
        a^{B_{0, 0}}   & a^{B_{0, 1}}   & \cdots & a^{B_{0, n-1}}   \\
        a^{B_{1, 0}}   & a^{B_{1, 1}}   & \cdots & a^{B_{1, n-1}}   \\
        \vdots         & \vdots         & \ddots & \vdots           \\
        a^{B_{m-1, 0}} & a^{B_{m-1, 1}} & \cdots & a^{B_{m-1, n-1}}
        \end{bmatrix}

    Matrix raised to a matrix exponent:

    .. math::
        A^{\circ B} = \begin{bmatrix}
        A_{0, 0}^{B_{0, 0}}     & A_{0, 1}^{B_{0, 1}}     &
        \cdots & A_{0, n-1}^{B_{0, n-1}}     \\
        A_{1, 0}^{B_{1, 0}}     & A_{1, 1}^{B_{1, 1}}     &
        \cdots & A_{1, n-1}^{B_{1, n-1}}     \\
        \vdots                  & \vdots                  &
        \ddots & \vdots                      \\
        A_{m-1, 0}^{B_{m-1, 0}} & A_{m-1, 1}^{B_{m-1, 1}} &
        \cdots & A_{m-1, n-1}^{B_{m-1, n-1}}
        \end{bmatrix}

    Scalar raised to a scalar exponent:

    .. math::
        a^{\circ b} = a^b
    """

    def __new__(cls, base, exp):
        base = sympify(base)
        exp = sympify(exp)

        if base.is_scalar and exp.is_scalar:
            return base ** exp

        if isinstance(base, MatrixExpr) and isinstance(exp, MatrixExpr):
            validate(base, exp)

        obj = super().__new__(cls, base, exp)
        return obj

    @property
    def base(self):
        return self._args[0]

    @property
    def exp(self):
        return self._args[1]

    @property
    def shape(self):
        if self.base.is_Matrix:
            return self.base.shape
        return self.exp.shape

    def _entry(self, i, j, **kwargs):
        base = self.base
        exp = self.exp

        if base.is_Matrix:
            a = base._entry(i, j, **kwargs)
        elif base.is_scalar:
            a = base
        else:
            raise ValueError(
                'The base {} must be a scalar or a matrix.'.format(base))

        if exp.is_Matrix:
            b = exp._entry(i, j, **kwargs)
        elif exp.is_scalar:
            b = exp
        else:
            raise ValueError(
                'The exponent {} must be a scalar or a matrix.'.format(exp))

        return a ** b

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return HadamardPower(transpose(self.base), self.exp)

    def _eval_derivative(self, x):
        dexp = self.exp.diff(x)
        logbase = self.base.applyfunc(log)
        dlbase = logbase.diff(x)
        return hadamard_product(
            dexp*logbase + self.exp*dlbase,
            self
        )

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.matrices.expressions.matexpr import _make_matrix

        lr = self.base._eval_derivative_matrix_lines(x)
        for i in lr:
            diagonal = [(1, 2), (3, 4)]
            diagonal = [e for j, e in enumerate(diagonal) if self.base.shape[j] != 1]
            l1 = i._lines[i._first_line_index]
            l2 = i._lines[i._second_line_index]
            subexpr = ExprBuilder(
                ArrayDiagonal,
                [
                    ExprBuilder(
                        ArrayTensorProduct,
                        [
                            ExprBuilder(_make_matrix, [l1]),
                            self.exp*hadamard_power(self.base, self.exp-1),
                            ExprBuilder(_make_matrix, [l2]),
                        ]
                    ),
                *diagonal],
                validator=ArrayDiagonal._validate
            )
            i._first_pointer_parent = subexpr.args[0].args[0].args
            i._first_pointer_index = 0
            i._first_line_index = 0
            i._second_pointer_parent = subexpr.args[0].args[2].args
            i._second_pointer_index = 0
            i._second_line_index = 0
            i._lines = [subexpr]
        return lr
