import operator
from functools import reduce, singledispatch

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.matrices.expressions.hadamard import HadamardProduct
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import (MatrixExpr, MatrixSymbol)
from sympy.matrices.expressions.special import Identity, OneMatrix
from sympy.matrices.expressions.transpose import Transpose
from sympy.combinatorics.permutations import _af_invert
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.tensor.array.expressions.array_expressions import (
    _ArrayExpr, ZeroArray, ArraySymbol, ArrayTensorProduct, ArrayAdd,
    PermuteDims, ArrayDiagonal, ArrayElementwiseApplyFunc, get_rank,
    get_shape, ArrayContraction, _array_tensor_product, _array_contraction,
    _array_diagonal, _array_add, _permute_dims, Reshape)
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array


@singledispatch
def array_derive(expr, x):
    """
    Derivatives (gradients) for array expressions.
    """
    raise NotImplementedError(f"not implemented for type {type(expr)}")


@array_derive.register(Expr)
def _(expr: Expr, x: _ArrayExpr):
    return ZeroArray(*x.shape)


@array_derive.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct, x: Expr):
    args = expr.args
    addend_list = []
    for i, arg in enumerate(expr.args):
        darg = array_derive(arg, x)
        if darg == 0:
            continue
        args_prev = args[:i]
        args_succ = args[i+1:]
        shape_prev = reduce(operator.add, map(get_shape, args_prev), ())
        shape_succ = reduce(operator.add, map(get_shape, args_succ), ())
        addend = _array_tensor_product(*args_prev, darg, *args_succ)
        tot1 = len(get_shape(x))
        tot2 = tot1 + len(shape_prev)
        tot3 = tot2 + len(get_shape(arg))
        tot4 = tot3 + len(shape_succ)
        perm = list(range(tot1, tot2)) + \
               list(range(tot1)) + list(range(tot2, tot3)) + \
               list(range(tot3, tot4))
        addend = _permute_dims(addend, _af_invert(perm))
        addend_list.append(addend)
    if len(addend_list) == 1:
        return addend_list[0]
    elif len(addend_list) == 0:
        return S.Zero
    else:
        return _array_add(*addend_list)


@array_derive.register(ArraySymbol)
def _(expr: ArraySymbol, x: _ArrayExpr):
    if expr == x:
        return _permute_dims(
            ArrayTensorProduct.fromiter(Identity(i) for i in expr.shape),
            [2*i for i in range(len(expr.shape))] + [2*i+1 for i in range(len(expr.shape))]
        )
    return ZeroArray(*(x.shape + expr.shape))


@array_derive.register(MatrixSymbol)
def _(expr: MatrixSymbol, x: _ArrayExpr):
    m, n = expr.shape
    if expr == x:
        return _permute_dims(
            _array_tensor_product(Identity(m), Identity(n)),
            [0, 2, 1, 3]
        )
    return ZeroArray(*(x.shape + expr.shape))


@array_derive.register(Identity)
def _(expr: Identity, x: _ArrayExpr):
    return ZeroArray(*(x.shape + expr.shape))


@array_derive.register(OneMatrix)
def _(expr: OneMatrix, x: _ArrayExpr):
    return ZeroArray(*(x.shape + expr.shape))


@array_derive.register(Transpose)
def _(expr: Transpose, x: Expr):
    # D(A.T, A) ==> (m,n,i,j) ==> D(A_ji, A_mn) = d_mj d_ni
    # D(B.T, A) ==> (m,n,i,j) ==> D(B_ji, A_mn)
    fd = array_derive(expr.arg, x)
    return _permute_dims(fd, [0, 1, 3, 2])


@array_derive.register(Inverse)
def _(expr: Inverse, x: Expr):
    mat = expr.I
    dexpr = array_derive(mat, x)
    tp = _array_tensor_product(-expr, dexpr, expr)
    mp = _array_contraction(tp, (1, 4), (5, 6))
    pp = _permute_dims(mp, [1, 2, 0, 3])
    return pp


@array_derive.register(ElementwiseApplyFunction)
def _(expr: ElementwiseApplyFunction, x: Expr):
    assert get_rank(expr) == 2
    assert get_rank(x) == 2
    fdiff = expr._get_function_fdiff()
    dexpr = array_derive(expr.expr, x)
    tp = _array_tensor_product(
        ElementwiseApplyFunction(fdiff, expr.expr),
        dexpr
    )
    td = _array_diagonal(
        tp, (0, 4), (1, 5)
    )
    return td


@array_derive.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc, x: Expr):
    fdiff = expr._get_function_fdiff()
    subexpr = expr.expr
    dsubexpr = array_derive(subexpr, x)
    tp = _array_tensor_product(
        dsubexpr,
        ArrayElementwiseApplyFunc(fdiff, subexpr)
    )
    b = get_rank(x)
    c = get_rank(expr)
    diag_indices = [(b + i, b + c + i) for i in range(c)]
    return _array_diagonal(tp, *diag_indices)


@array_derive.register(MatrixExpr)
def _(expr: MatrixExpr, x: Expr):
    cg = convert_matrix_to_array(expr)
    return array_derive(cg, x)


@array_derive.register(HadamardProduct)
def _(expr: HadamardProduct, x: Expr):
    raise NotImplementedError()


@array_derive.register(ArrayContraction)
def _(expr: ArrayContraction, x: Expr):
    fd = array_derive(expr.expr, x)
    rank_x = len(get_shape(x))
    contraction_indices = expr.contraction_indices
    new_contraction_indices = [tuple(j + rank_x for j in i) for i in contraction_indices]
    return _array_contraction(fd, *new_contraction_indices)


@array_derive.register(ArrayDiagonal)
def _(expr: ArrayDiagonal, x: Expr):
    dsubexpr = array_derive(expr.expr, x)
    rank_x = len(get_shape(x))
    diag_indices = [[j + rank_x for j in i] for i in expr.diagonal_indices]
    return _array_diagonal(dsubexpr, *diag_indices)


@array_derive.register(ArrayAdd)
def _(expr: ArrayAdd, x: Expr):
    return _array_add(*[array_derive(arg, x) for arg in expr.args])


@array_derive.register(PermuteDims)
def _(expr: PermuteDims, x: Expr):
    de = array_derive(expr.expr, x)
    perm = [0, 1] + [i + 2 for i in expr.permutation.array_form]
    return _permute_dims(de, perm)


@array_derive.register(Reshape)
def _(expr: Reshape, x: Expr):
    de = array_derive(expr.expr, x)
    return Reshape(de, get_shape(x) + expr.shape)


def matrix_derive(expr, x):
    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    ce = convert_matrix_to_array(expr)
    dce = array_derive(ce, x)
    return convert_array_to_matrix(dce).doit()
