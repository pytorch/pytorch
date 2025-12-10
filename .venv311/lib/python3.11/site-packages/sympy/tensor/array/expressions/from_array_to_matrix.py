from __future__ import annotations
import itertools
from collections import defaultdict
from typing import FrozenSet
from functools import singledispatch
from itertools import accumulate

from sympy import MatMul, Basic, Wild, KroneckerProduct
from sympy.assumptions.ask import (Q, ask)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import hadamard_product, HadamardPower
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import (Identity, ZeroMatrix, OneMatrix)
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.combinatorics.permutations import _af_invert, Permutation
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import PermuteDims, ArrayDiagonal, \
    ArrayTensorProduct, OneArray, get_rank, _get_subrank, ZeroArray, ArrayContraction, \
    ArrayAdd, _CodegenArrayAbstract, get_shape, ArrayElementwiseApplyFunc, _ArrayExpr, _EditArrayContraction, _ArgE, \
    ArrayElement, _array_tensor_product, _array_contraction, _array_diagonal, _array_add, _permute_dims
from sympy.tensor.array.expressions.utils import _get_mapping_from_subranks


def _get_candidate_for_matmul_from_contraction(scan_indices: list[int | None], remaining_args: list[_ArgE]) -> tuple[_ArgE | None, bool, int]:

    scan_indices_int: list[int] = [i for i in scan_indices if i is not None]
    if len(scan_indices_int) == 0:
        return None, False, -1

    transpose: bool = False
    candidate: _ArgE | None = None
    candidate_index: int = -1
    for arg_with_ind2 in remaining_args:
        if not isinstance(arg_with_ind2.element, MatrixExpr):
            continue
        for index in scan_indices_int:
            if candidate_index != -1 and candidate_index != index:
                # A candidate index has already been selected, check
                # repetitions only for that index:
                continue
            if index in arg_with_ind2.indices:
                if set(arg_with_ind2.indices) == {index}:
                    # Index repeated twice in arg_with_ind2
                    candidate = None
                    break
                if candidate is None:
                    candidate = arg_with_ind2
                    candidate_index = index
                    transpose = (index == arg_with_ind2.indices[1])
                else:
                    # Index repeated more than twice, break
                    candidate = None
                    break
    return candidate, transpose, candidate_index


def _insert_candidate_into_editor(editor: _EditArrayContraction, arg_with_ind: _ArgE, candidate: _ArgE, transpose1: bool, transpose2: bool):
    other = candidate.element
    other_index: int | None
    if transpose2:
        other = Transpose(other)
        other_index = candidate.indices[0]
    else:
        other_index = candidate.indices[1]
    new_element = (Transpose(arg_with_ind.element) if transpose1 else arg_with_ind.element) * other
    editor.args_with_ind.remove(candidate)
    new_arge = _ArgE(new_element)
    return new_arge, other_index


def _support_function_tp1_recognize(contraction_indices, args):
    if len(contraction_indices) == 0:
        return _a2m_tensor_product(*args)

    ac = _array_contraction(_array_tensor_product(*args), *contraction_indices)
    editor = _EditArrayContraction(ac)
    editor.track_permutation_start()

    while True:
        flag_stop = True
        for i, arg_with_ind in enumerate(editor.args_with_ind):
            if not isinstance(arg_with_ind.element, MatrixExpr):
                continue

            first_index = arg_with_ind.indices[0]
            second_index = arg_with_ind.indices[1]

            first_frequency = editor.count_args_with_index(first_index)
            second_frequency = editor.count_args_with_index(second_index)

            if first_index is not None and first_frequency == 1 and first_index == second_index:
                flag_stop = False
                arg_with_ind.element = Trace(arg_with_ind.element)._normalize()
                arg_with_ind.indices = []
                break

            scan_indices = []
            if first_frequency == 2:
                scan_indices.append(first_index)
            if second_frequency == 2:
                scan_indices.append(second_index)

            candidate, transpose, found_index = _get_candidate_for_matmul_from_contraction(scan_indices, editor.args_with_ind[i+1:])
            if candidate is not None:
                flag_stop = False
                editor.track_permutation_merge(arg_with_ind, candidate)
                transpose1 = found_index == first_index
                new_arge, other_index = _insert_candidate_into_editor(editor, arg_with_ind, candidate, transpose1, transpose)
                if found_index == first_index:
                    new_arge.indices = [second_index, other_index]
                else:
                    new_arge.indices = [first_index, other_index]
                set_indices = set(new_arge.indices)
                if len(set_indices) == 1 and set_indices != {None}:
                    # This is a trace:
                    new_arge.element = Trace(new_arge.element)._normalize()
                    new_arge.indices = []
                editor.args_with_ind[i] = new_arge
                # TODO: is this break necessary?
                break

        if flag_stop:
            break

    editor.refresh_indices()
    return editor.to_array_contraction()


def _find_trivial_matrices_rewrite(expr: ArrayTensorProduct):
    # If there are matrices of trivial shape in the tensor product (i.e. shape
    # (1, 1)), try to check if there is a suitable non-trivial MatMul where the
    # expression can be inserted.

    # For example, if "a" has shape (1, 1) and "b" has shape (k, 1), the
    # expressions "_array_tensor_product(a, b*b.T)" can be rewritten as
    # "b*a*b.T"

    trivial_matrices = []
    pos: int | None = None  # must be initialized else causes UnboundLocalError
    first: MatrixExpr | None = None  # may cause UnboundLocalError if not initialized
    second: MatrixExpr | None = None  # may cause UnboundLocalError if not initialized
    removed: list[int] = []
    counter: int = 0
    args: list[Basic | None] = list(expr.args)
    for i, arg in enumerate(expr.args):
        if isinstance(arg, MatrixExpr):
            if arg.shape == (1, 1):
                trivial_matrices.append(arg)
                args[i] = None
                removed.extend([counter, counter+1])
            elif pos is None and isinstance(arg, MatMul):
                margs = arg.args
                for j, e in enumerate(margs):
                    if isinstance(e, MatrixExpr) and e.shape[1] == 1:
                        pos = i
                        first = MatMul.fromiter(margs[:j+1])
                        second = MatMul.fromiter(margs[j+1:])
                        break
        counter += get_rank(arg)
    if pos is None:
        return expr, []
    args[pos] = (first*MatMul.fromiter(i for i in trivial_matrices)*second).doit()
    return _array_tensor_product(*[i for i in args if i is not None]), removed


def _find_trivial_kronecker_products_broadcast(expr: ArrayTensorProduct):
    newargs: list[Basic] = []
    removed = []
    count_dims = 0
    for arg in expr.args:
        count_dims += get_rank(arg)
        shape = get_shape(arg)
        current_range = [count_dims-i for i in range(len(shape), 0, -1)]
        if (shape == (1, 1) and len(newargs) > 0 and 1 not in get_shape(newargs[-1]) and
            isinstance(newargs[-1], MatrixExpr) and isinstance(arg, MatrixExpr)):
            # KroneckerProduct object allows the trick of broadcasting:
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            removed.extend(current_range)
        elif 1 not in shape and len(newargs) > 0 and get_shape(newargs[-1]) == (1, 1):
            # Broadcast:
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            prev_range = [i for i in range(min(current_range)) if i not in removed]
            removed.extend(prev_range[-2:])
        else:
            newargs.append(arg)
    return _array_tensor_product(*newargs), removed


@singledispatch
def _array2matrix(expr):
    return expr


@_array2matrix.register(ZeroArray)
def _(expr: ZeroArray):
    if get_rank(expr) == 2:
        return ZeroMatrix(*expr.shape)
    else:
        return expr


@_array2matrix.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct):
    return _a2m_tensor_product(*[_array2matrix(arg) for arg in expr.args])


@_array2matrix.register(ArrayContraction)
def _(expr: ArrayContraction):
    expr = expr.flatten_contraction_of_diagonal()
    expr = identify_removable_identity_matrices(expr)
    expr = expr.split_multiple_contractions()
    expr = identify_hadamard_products(expr)
    if not isinstance(expr, ArrayContraction):
        return _array2matrix(expr)
    subexpr = expr.expr
    contraction_indices: tuple[tuple[int]] = expr.contraction_indices
    if contraction_indices == ((0,), (1,)) or (
        contraction_indices == ((0,),) and subexpr.shape[1] == 1
    ) or (
        contraction_indices == ((1,),) and subexpr.shape[0] == 1
    ):
        shape = subexpr.shape
        subexpr = _array2matrix(subexpr)
        if isinstance(subexpr, MatrixExpr):
            return OneMatrix(1, shape[0])*subexpr*OneMatrix(shape[1], 1)
    if isinstance(subexpr, ArrayTensorProduct):
        newexpr = _array_contraction(_array2matrix(subexpr), *contraction_indices)
        contraction_indices = newexpr.contraction_indices
        if any(i > 2 for i in newexpr.subranks):
            addends = _array_add(*[_a2m_tensor_product(*j) for j in itertools.product(*[i.args if isinstance(i,
                                                                                                                             ArrayAdd) else [i] for i in expr.expr.args])])
            newexpr = _array_contraction(addends, *contraction_indices)
        if isinstance(newexpr, ArrayAdd):
            ret = _array2matrix(newexpr)
            return ret
        assert isinstance(newexpr, ArrayContraction)
        ret = _support_function_tp1_recognize(contraction_indices, list(newexpr.expr.args))
        return ret
    elif not isinstance(subexpr, _CodegenArrayAbstract):
        ret = _array2matrix(subexpr)
        if isinstance(ret, MatrixExpr):
            assert expr.contraction_indices == ((0, 1),)
            return _a2m_trace(ret)
        else:
            return _array_contraction(ret, *expr.contraction_indices)


@_array2matrix.register(ArrayDiagonal)
def _(expr: ArrayDiagonal):
    pexpr = _array_diagonal(_array2matrix(expr.expr), *expr.diagonal_indices)
    pexpr = identify_hadamard_products(pexpr)
    if isinstance(pexpr, ArrayDiagonal):
        pexpr = _array_diag2contr_diagmatrix(pexpr)
    if expr == pexpr:
        return expr
    return _array2matrix(pexpr)


@_array2matrix.register(PermuteDims)
def _(expr: PermuteDims):
    if expr.permutation.array_form == [1, 0]:
        return _a2m_transpose(_array2matrix(expr.expr))
    elif isinstance(expr.expr, ArrayTensorProduct):
        ranks = expr.expr.subranks
        inv_permutation = expr.permutation**(-1)
        newrange = [inv_permutation(i) for i in range(sum(ranks))]
        newpos = []
        counter = 0
        for rank in ranks:
            newpos.append(newrange[counter:counter+rank])
            counter += rank
        newargs = []
        newperm = []
        scalars = []
        for pos, arg in zip(newpos, expr.expr.args):
            if len(pos) == 0:
                scalars.append(_array2matrix(arg))
            elif pos == sorted(pos):
                newargs.append((_array2matrix(arg), pos[0]))
                newperm.extend(pos)
            elif len(pos) == 2:
                newargs.append((_a2m_transpose(_array2matrix(arg)), pos[0]))
                newperm.extend(reversed(pos))
            else:
                raise NotImplementedError()
        newargs = [i[0] for i in newargs]
        return _permute_dims(_a2m_tensor_product(*scalars, *newargs), _af_invert(newperm))
    elif isinstance(expr.expr, ArrayContraction):
        mat_mul_lines = _array2matrix(expr.expr)
        if not isinstance(mat_mul_lines, ArrayTensorProduct):
            return _permute_dims(mat_mul_lines, expr.permutation)
        # TODO: this assumes that all arguments are matrices, it may not be the case:
        permutation = Permutation(2*len(mat_mul_lines.args)-1)*expr.permutation
        permuted = [permutation(i) for i in range(2*len(mat_mul_lines.args))]
        args_array = [None for i in mat_mul_lines.args]
        for i in range(len(mat_mul_lines.args)):
            p1 = permuted[2*i]
            p2 = permuted[2*i+1]
            if p1 // 2 != p2 // 2:
                return _permute_dims(mat_mul_lines, permutation)
            if p1 > p2:
                args_array[i] = _a2m_transpose(mat_mul_lines.args[p1 // 2])
            else:
                args_array[i] = mat_mul_lines.args[p1 // 2]
        return _a2m_tensor_product(*args_array)
    else:
        return expr


@_array2matrix.register(ArrayAdd)
def _(expr: ArrayAdd):
    addends = [_array2matrix(arg) for arg in expr.args]
    return _a2m_add(*addends)


@_array2matrix.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc):
    subexpr = _array2matrix(expr.expr)
    if isinstance(subexpr, MatrixExpr):
        if subexpr.shape != (1, 1):
            d = expr.function.bound_symbols[0]
            w = Wild("w", exclude=[d])
            p = Wild("p", exclude=[d])
            m = expr.function.expr.match(w*d**p)
            if m is not None:
                return m[w]*HadamardPower(subexpr, m[p])
        return ElementwiseApplyFunction(expr.function, subexpr)
    else:
        return ArrayElementwiseApplyFunc(expr.function, subexpr)


@_array2matrix.register(ArrayElement)
def _(expr: ArrayElement):
    ret = _array2matrix(expr.name)
    if isinstance(ret, MatrixExpr):
        return MatrixElement(ret, *expr.indices)
    return ArrayElement(ret, expr.indices)


@singledispatch
def _remove_trivial_dims(expr):
    return expr, []


@_remove_trivial_dims.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct):
    # Recognize expressions like [x, y] with shape (k, 1, k, 1) as `x*y.T`.
    # The matrix expression has to be equivalent to the tensor product of the
    # matrices, with trivial dimensions (i.e. dim=1) dropped.
    # That is, add contractions over trivial dimensions:

    removed = []
    newargs = []
    cumul = list(accumulate([0] + [get_rank(arg) for arg in expr.args]))
    pending = None
    prev_i = None
    for i, arg in enumerate(expr.args):
        current_range = list(range(cumul[i], cumul[i+1]))
        if isinstance(arg, OneArray):
            removed.extend(current_range)
            continue
        if not isinstance(arg, (MatrixExpr, MatrixBase)):
            rarg, rem = _remove_trivial_dims(arg)
            removed.extend(rem)
            newargs.append(rarg)
            continue
        elif getattr(arg, "is_Identity", False) and arg.shape == (1, 1):
            if arg.shape == (1, 1):
                # Ignore identity matrices of shape (1, 1) - they are equivalent to scalar 1.
                removed.extend(current_range)
            continue
        elif arg.shape == (1, 1):
            arg, _ = _remove_trivial_dims(arg)
            # Matrix is equivalent to scalar:
            if len(newargs) == 0:
                newargs.append(arg)
            elif 1 in get_shape(newargs[-1]):
                if newargs[-1].shape[1] == 1:
                    newargs[-1] = newargs[-1]*arg
                else:
                    newargs[-1] = arg*newargs[-1]
                removed.extend(current_range)
            else:
                newargs.append(arg)
        elif 1 in arg.shape:
            k = [i for i in arg.shape if i != 1][0]
            if pending is None:
                pending = k
                prev_i = i
                newargs.append(arg)
            elif pending == k:
                prev = newargs[-1]
                if prev.shape[0] == 1:
                    d1 = cumul[prev_i]  # type: ignore
                    prev = _a2m_transpose(prev)
                else:
                    d1 = cumul[prev_i] + 1  # type: ignore
                if arg.shape[1] == 1:
                    d2 = cumul[i] + 1
                    arg = _a2m_transpose(arg)
                else:
                    d2 = cumul[i]
                newargs[-1] = prev*arg
                pending = None
                removed.extend([d1, d2])
            else:
                newargs.append(arg)
                pending = k
                prev_i = i
        else:
            newargs.append(arg)
            pending = None
    newexpr, newremoved = _a2m_tensor_product(*newargs), sorted(removed)
    if isinstance(newexpr, ArrayTensorProduct):
        newexpr, newremoved2 = _find_trivial_matrices_rewrite(newexpr)
        newremoved = _combine_removed(-1, newremoved, newremoved2)
    if isinstance(newexpr, ArrayTensorProduct):
        newexpr, newremoved2 = _find_trivial_kronecker_products_broadcast(newexpr)
        newremoved = _combine_removed(-1, newremoved, newremoved2)
    return newexpr, newremoved


@_remove_trivial_dims.register(ArrayAdd)
def _(expr: ArrayAdd):
    rec = [_remove_trivial_dims(arg) for arg in expr.args]
    newargs, removed = zip(*rec)
    if len({get_shape(i) for i in newargs}) > 1:
        return expr, []
    if len(removed) == 0:
        return expr, removed
    removed1 = removed[0]
    return _a2m_add(*newargs), removed1


@_remove_trivial_dims.register(PermuteDims)
def _(expr: PermuteDims):
    subexpr, subremoved = _remove_trivial_dims(expr.expr)
    p = expr.permutation.array_form
    pinv = _af_invert(expr.permutation.array_form)
    shift = list(accumulate([1 if i in subremoved else 0 for i in range(len(p))]))
    premoved = [pinv[i] for i in subremoved]
    p2 = [e - shift[e] for e in p if e not in subremoved]
    # TODO: check if subremoved should be permuted as well...
    newexpr = _permute_dims(subexpr, p2)
    premoved = sorted(premoved)
    if newexpr != expr:
        newexpr, removed2 = _remove_trivial_dims(_array2matrix(newexpr))
        premoved = _combine_removed(-1, premoved, removed2)
    return newexpr, premoved


@_remove_trivial_dims.register(ArrayContraction)
def _(expr: ArrayContraction):
    new_expr, removed0 = _array_contraction_to_diagonal_multiple_identity(expr)
    if new_expr != expr:
        new_expr2, removed1 = _remove_trivial_dims(_array2matrix(new_expr))
        removed = _combine_removed(-1, removed0, removed1)
        return new_expr2, removed
    rank1 = get_rank(expr)
    expr, removed1 = remove_identity_matrices(expr)
    if not isinstance(expr, ArrayContraction):
        expr2, removed2 = _remove_trivial_dims(expr)
        return expr2, _combine_removed(rank1, removed1, removed2)
    newexpr, removed2 = _remove_trivial_dims(expr.expr)
    shifts = list(accumulate([1 if i in removed2 else 0 for i in range(get_rank(expr.expr))]))
    new_contraction_indices = [tuple(j for j in i if j not in removed2) for i in expr.contraction_indices]
    # Remove possible empty tuples "()":
    new_contraction_indices = [i for i in new_contraction_indices if len(i) > 0]
    contraction_indices_flat = [j for i in expr.contraction_indices for j in i]
    removed2 = [i for i in removed2 if i not in contraction_indices_flat]
    new_contraction_indices = [tuple(j - shifts[j] for j in i) for i in new_contraction_indices]
    # Shift removed2:
    removed2 = ArrayContraction._push_indices_up(expr.contraction_indices, removed2)
    removed = _combine_removed(rank1, removed1, removed2)
    return _array_contraction(newexpr, *new_contraction_indices), list(removed)


def _remove_diagonalized_identity_matrices(expr: ArrayDiagonal):
    assert isinstance(expr, ArrayDiagonal)
    editor = _EditArrayContraction(expr)
    mapping = {i: {j for j in editor.args_with_ind if i in j.indices} for i in range(-1, -1-editor.number_of_diagonal_indices, -1)}
    removed = []
    counter: int = 0
    for i, arg_with_ind in enumerate(editor.args_with_ind):
        counter += len(arg_with_ind.indices)
        if isinstance(arg_with_ind.element, Identity):
            if None in arg_with_ind.indices and any(i is not None and (i < 0) == True for i in arg_with_ind.indices):
                diag_ind = [j for j in arg_with_ind.indices if j is not None][0]
                other = [j for j in mapping[diag_ind] if j != arg_with_ind][0]
                if not isinstance(other.element, MatrixExpr):
                    continue
                if 1 not in other.element.shape:
                    continue
                if None not in other.indices:
                    continue
                editor.args_with_ind[i].element = None
                none_index = other.indices.index(None)
                other.element = DiagMatrix(other.element)
                other_range = editor.get_absolute_range(other)
                removed.extend([other_range[0] + none_index])
    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    removed = ArrayDiagonal._push_indices_up(expr.diagonal_indices, removed, get_rank(expr.expr))
    return editor.to_array_contraction(), removed


@_remove_trivial_dims.register(ArrayDiagonal)
def _(expr: ArrayDiagonal):
    newexpr, removed = _remove_trivial_dims(expr.expr)
    shifts = list(accumulate([0] + [1 if i in removed else 0 for i in range(get_rank(expr.expr))]))
    new_diag_indices_map = {i: tuple(j for j in i if j not in removed) for i in expr.diagonal_indices}
    for old_diag_tuple, new_diag_tuple in new_diag_indices_map.items():
        if len(new_diag_tuple) == 1:
            removed = [i for i in removed if i not in old_diag_tuple]
    new_diag_indices = [tuple(j - shifts[j] for j in i) for i in new_diag_indices_map.values()]
    rank = get_rank(expr.expr)
    removed = ArrayDiagonal._push_indices_up(expr.diagonal_indices, removed, rank)
    removed = sorted(set(removed))
    # If there are single axes to diagonalize remaining, it means that their
    # corresponding dimension has been removed, they no longer need diagonalization:
    new_diag_indices = [i for i in new_diag_indices if len(i) > 0]
    if len(new_diag_indices) > 0:
        newexpr2 = _array_diagonal(newexpr, *new_diag_indices, allow_trivial_diags=True)
    else:
        newexpr2 = newexpr
    if isinstance(newexpr2, ArrayDiagonal):
        newexpr3, removed2 = _remove_diagonalized_identity_matrices(newexpr2)
        removed = _combine_removed(-1, removed, removed2)
        return newexpr3, removed
    else:
        return newexpr2, removed


@_remove_trivial_dims.register(ElementwiseApplyFunction)
def _(expr: ElementwiseApplyFunction):
    subexpr, removed = _remove_trivial_dims(expr.expr)
    if subexpr.shape == (1, 1):
        # TODO: move this to ElementwiseApplyFunction
        return expr.function(subexpr), removed + [0, 1]
    return ElementwiseApplyFunction(expr.function, subexpr), []


@_remove_trivial_dims.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc):
    subexpr, removed = _remove_trivial_dims(expr.expr)
    return ArrayElementwiseApplyFunc(expr.function, subexpr), removed


def convert_array_to_matrix(expr):
    r"""
    Recognize matrix expressions in codegen objects.

    If more than one matrix multiplication line have been detected, return a
    list with the matrix expressions.

    Examples
    ========

    >>> from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    >>> from sympy.tensor.array import tensorcontraction, tensorproduct
    >>> from sympy import MatrixSymbol, Sum
    >>> from sympy.abc import i, j, k, l, N
    >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> A = MatrixSymbol("A", N, N)
    >>> B = MatrixSymbol("B", N, N)
    >>> C = MatrixSymbol("C", N, N)
    >>> D = MatrixSymbol("D", N, N)

    >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A.T

    Transposition is detected:

    >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A.T*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A

    Detect the trace:

    >>> expr = Sum(A[i, i], (i, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A)

    Recognize some more complex traces:

    >>> expr = Sum(A[i, j]*B[j, i], (i, 0, N-1), (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A*B)

    More complicated expressions:

    >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B.T*A.T

    Expressions constructed from matrix expressions do not contain literal
    indices, the positions of free indices are returned instead:

    >>> expr = A*B
    >>> cg = convert_matrix_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B

    If more than one line of matrix multiplications is detected, return
    separate matrix multiplication factors embedded in a tensor product object:

    >>> cg = tensorcontraction(tensorproduct(A, B, C, D), (1, 2), (5, 6))
    >>> convert_array_to_matrix(cg)
    ArrayTensorProduct(A*B, C*D)

    The two lines have free indices at axes 0, 3 and 4, 7, respectively.
    """
    rec = _array2matrix(expr)
    rec, removed = _remove_trivial_dims(rec)
    return rec


def _array_diag2contr_diagmatrix(expr: ArrayDiagonal):
    if isinstance(expr.expr, ArrayTensorProduct):
        args = list(expr.expr.args)
        diag_indices = list(expr.diagonal_indices)
        mapping = _get_mapping_from_subranks([_get_subrank(arg) for arg in args])
        tuple_links = [[mapping[j] for j in i] for i in diag_indices]
        contr_indices = []
        total_rank = get_rank(expr)
        replaced = [False for arg in args]
        for i, (abs_pos, rel_pos) in enumerate(zip(diag_indices, tuple_links)):
            if len(abs_pos) != 2:
                continue
            (pos1_outer, pos1_inner), (pos2_outer, pos2_inner) = rel_pos
            arg1 = args[pos1_outer]
            arg2 = args[pos2_outer]
            if get_rank(arg1) != 2 or get_rank(arg2) != 2:
                if replaced[pos1_outer]:
                    diag_indices[i] = None
                if replaced[pos2_outer]:
                    diag_indices[i] = None
                continue
            pos1_in2 = 1 - pos1_inner
            pos2_in2 = 1 - pos2_inner
            if arg1.shape[pos1_in2] == 1:
                if arg1.shape[pos1_inner] != 1:
                    darg1 = DiagMatrix(arg1)
                else:
                    darg1 = arg1
                args.append(darg1)
                contr_indices.append(((pos2_outer, pos2_inner), (len(args)-1, pos1_inner)))
                total_rank += 1
                diag_indices[i] = None
                args[pos1_outer] = OneArray(arg1.shape[pos1_in2])
                replaced[pos1_outer] = True
            elif arg2.shape[pos2_in2] == 1:
                if arg2.shape[pos2_inner] != 1:
                    darg2 = DiagMatrix(arg2)
                else:
                    darg2 = arg2
                args.append(darg2)
                contr_indices.append(((pos1_outer, pos1_inner), (len(args)-1, pos2_inner)))
                total_rank += 1
                diag_indices[i] = None
                args[pos2_outer] = OneArray(arg2.shape[pos2_in2])
                replaced[pos2_outer] = True
        diag_indices_new = [i for i in diag_indices if i is not None]
        cumul = list(accumulate([0] + [get_rank(arg) for arg in args]))
        contr_indices2 = [tuple(cumul[a] + b for a, b in i) for i in contr_indices]
        tc = _array_contraction(
            _array_tensor_product(*args), *contr_indices2
        )
        td = _array_diagonal(tc, *diag_indices_new)
        return td
    return expr


def _a2m_mul(*args):
    if not any(isinstance(i, _CodegenArrayAbstract) for i in args):
        from sympy.matrices.expressions.matmul import MatMul
        return MatMul(*args).doit()
    else:
        return _array_contraction(
            _array_tensor_product(*args),
            *[(2*i-1, 2*i) for i in range(1, len(args))]
        )


def _a2m_tensor_product(*args):
    scalars = []
    arrays = []
    for arg in args:
        if isinstance(arg, (MatrixExpr, _ArrayExpr, _CodegenArrayAbstract)):
            arrays.append(arg)
        else:
            scalars.append(arg)
    scalar = Mul.fromiter(scalars)
    if len(arrays) == 0:
        return scalar
    if scalar != 1:
        if isinstance(arrays[0], _CodegenArrayAbstract):
            arrays = [scalar] + arrays
        else:
            arrays[0] *= scalar
    return _array_tensor_product(*arrays)


def _a2m_add(*args):
    if not any(isinstance(i, _CodegenArrayAbstract) for i in args):
        from sympy.matrices.expressions.matadd import MatAdd
        return MatAdd(*args).doit()
    else:
        return _array_add(*args)


def _a2m_trace(arg):
    if isinstance(arg, _CodegenArrayAbstract):
        return _array_contraction(arg, (0, 1))
    else:
        from sympy.matrices.expressions.trace import Trace
        return Trace(arg)


def _a2m_transpose(arg):
    if isinstance(arg, _CodegenArrayAbstract):
        return _permute_dims(arg, [1, 0])
    else:
        from sympy.matrices.expressions.transpose import Transpose
        return Transpose(arg).doit()


def identify_hadamard_products(expr: ArrayContraction | ArrayDiagonal):

    editor: _EditArrayContraction = _EditArrayContraction(expr)

    map_contr_to_args: dict[FrozenSet, list[_ArgE]] = defaultdict(list)
    map_ind_to_inds: dict[int | None, int] = defaultdict(int)
    for arg_with_ind in editor.args_with_ind:
        for ind in arg_with_ind.indices:
            map_ind_to_inds[ind] += 1
        if None in arg_with_ind.indices:
            continue
        map_contr_to_args[frozenset(arg_with_ind.indices)].append(arg_with_ind)

    k: FrozenSet[int]
    v: list[_ArgE]
    for k, v in map_contr_to_args.items():
        make_trace: bool = False
        if len(k) == 1 and next(iter(k)) >= 0 and sum(next(iter(k)) in i for i in map_contr_to_args) == 1:
            # This is a trace: the arguments are fully contracted with only one
            # index, and the index isn't used anywhere else:
            make_trace = True
            first_element = S.One
        elif len(k) != 2:
            # Hadamard product only defined for matrices:
            continue
        if len(v) == 1:
            # Hadamard product with a single argument makes no sense:
            continue
        for ind in k:
            if map_ind_to_inds[ind] <= 2:
                # There is no other contraction, skip:
                continue

        def check_transpose(x):
            x = [i if i >= 0 else -1-i for i in x]
            return x == sorted(x)

        # Check if expression is a trace:
        if all(map_ind_to_inds[j] == len(v) and j >= 0 for j in k) and all(j >= 0 for j in k):
            # This is a trace
            make_trace = True
            first_element = v[0].element
            if not check_transpose(v[0].indices):
                first_element = first_element.T # type: ignore
            hadamard_factors = v[1:]
        else:
            hadamard_factors = v

        # This is a Hadamard product:

        hp = hadamard_product(*[i.element if check_transpose(i.indices) else Transpose(i.element) for i in hadamard_factors])
        hp_indices = v[0].indices
        if not check_transpose(hadamard_factors[0].indices):
            hp_indices = list(reversed(hp_indices))
        if make_trace:
            hp = Trace(first_element*hp.T)._normalize()
            hp_indices = []
        editor.insert_after(v[0], _ArgE(hp, hp_indices))
        for i in v:
            editor.args_with_ind.remove(i)

    return editor.to_array_contraction()


def identify_removable_identity_matrices(expr):
    editor = _EditArrayContraction(expr)

    flag = True
    while flag:
        flag = False
        for arg_with_ind in editor.args_with_ind:
            if isinstance(arg_with_ind.element, Identity):
                k = arg_with_ind.element.shape[0]
                # Candidate for removal:
                if arg_with_ind.indices == [None, None]:
                    # Free identity matrix, will be cleared by _remove_trivial_dims:
                    continue
                elif None in arg_with_ind.indices:
                    ind = [j for j in arg_with_ind.indices if j is not None][0]
                    counted = editor.count_args_with_index(ind)
                    if counted == 1:
                        # Identity matrix contracted only on one index with itself,
                        # transform to a OneArray(k) element:
                        editor.insert_after(arg_with_ind, OneArray(k))
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    elif counted > 2:
                        # Case counted = 2 is a matrix multiplication by identity matrix, skip it.
                        # Case counted > 2 is a multiple contraction,
                        # this is a case where the contraction becomes a diagonalization if the
                        # identity matrix is dropped.
                        continue
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    ind = arg_with_ind.indices[0]
                    counted = editor.count_args_with_index(ind)
                    if counted > 1:
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    else:
                        # This is a trace, skip it as it will be recognized somewhere else:
                        pass
            elif ask(Q.diagonal(arg_with_ind.element)):
                if arg_with_ind.indices == [None, None]:
                    continue
                elif None in arg_with_ind.indices:
                    pass
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    ind = arg_with_ind.indices[0]
                    counted = editor.count_args_with_index(ind)
                    if counted == 3:
                        # A_ai B_bi D_ii ==> A_ai D_ij B_bj
                        ind_new = editor.get_new_contraction_index()
                        other_args = [j for j in editor.args_with_ind if j != arg_with_ind]
                        other_args[1].indices = [ind_new if j == ind else j for j in other_args[1].indices]
                        arg_with_ind.indices = [ind, ind_new]
                        flag = True
                        break

    return editor.to_array_contraction()


def remove_identity_matrices(expr: ArrayContraction):
    editor = _EditArrayContraction(expr)
    removed: list[int] = []

    permutation_map = {}

    free_indices = list(accumulate([0] + [sum(i is None for i in arg.indices) for arg in editor.args_with_ind]))
    free_map = dict(zip(editor.args_with_ind, free_indices[:-1]))

    update_pairs = {}

    for ind in range(editor.number_of_contraction_indices):
        args = editor.get_args_with_index(ind)
        identity_matrices = [i for i in args if isinstance(i.element, Identity)]
        number_identity_matrices = len(identity_matrices)
        # If the contraction involves a non-identity matrix and multiple identity matrices:
        if number_identity_matrices != len(args) - 1 or number_identity_matrices == 0:
            continue
        # Get the non-identity element:
        non_identity = [i for i in args if not isinstance(i.element, Identity)][0]
        # Check that all identity matrices have at least one free index
        # (otherwise they would be contractions to some other elements)
        if any(None not in i.indices for i in identity_matrices):
            continue
        # Mark the identity matrices for removal:
        for i in identity_matrices:
            i.element = None
            removed.extend(range(free_map[i], free_map[i] + len([j for j in i.indices if j is None])))
        last_removed = removed.pop(-1)
        update_pairs[last_removed, ind] = non_identity.indices[:]
        # Remove the indices from the non-identity matrix, as the contraction
        # no longer exists:
        non_identity.indices = [None if i == ind else i for i in non_identity.indices]

    removed.sort()

    shifts = list(accumulate([1 if i in removed else 0 for i in range(get_rank(expr))]))
    for (last_removed, ind), non_identity_indices in update_pairs.items():
        pos = [free_map[non_identity] + i for i, e in enumerate(non_identity_indices) if e == ind]
        assert len(pos) == 1
        for j in pos:
            permutation_map[j] = last_removed

    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    ret_expr = editor.to_array_contraction()
    permutation = []
    counter = 0
    counter2 = 0
    for j in range(get_rank(expr)):
        if j in removed:
            continue
        if counter2 in permutation_map:
            target = permutation_map[counter2]
            permutation.append(target - shifts[target])
            counter2 += 1
        else:
            while counter in permutation_map.values():
                counter += 1
            permutation.append(counter)
            counter += 1
            counter2 += 1
    ret_expr2 = _permute_dims(ret_expr, _af_invert(permutation))
    return ret_expr2, removed


def _combine_removed(dim: int, removed1: list[int], removed2: list[int]) -> list[int]:
    # Concatenate two axis removal operations as performed by
    # _remove_trivial_dims,
    removed1 = sorted(removed1)
    removed2 = sorted(removed2)
    i = 0
    j = 0
    removed = []
    while True:
        if j >= len(removed2):
            while i < len(removed1):
                removed.append(removed1[i])
                i += 1
            break
        elif i < len(removed1) and removed1[i] <= i + removed2[j]:
            removed.append(removed1[i])
            i += 1
        else:
            removed.append(i + removed2[j])
            j += 1
    return removed


def _array_contraction_to_diagonal_multiple_identity(expr: ArrayContraction):
    editor = _EditArrayContraction(expr)
    editor.track_permutation_start()
    removed: list[int] = []
    diag_index_counter: int = 0
    for i in range(editor.number_of_contraction_indices):
        identities = []
        args = []
        for j, arg in enumerate(editor.args_with_ind):
            if i not in arg.indices:
                continue
            if isinstance(arg.element, Identity):
                identities.append(arg)
            else:
                args.append(arg)
        if len(identities) == 0:
            continue
        if len(args) + len(identities) < 3:
            continue
        new_diag_ind = -1 - diag_index_counter
        diag_index_counter += 1
        # Variable "flag" to control whether to skip this contraction set:
        flag: bool = True
        for i1, id1 in enumerate(identities):
            if None not in id1.indices:
                flag = True
                break
            free_pos = list(range(*editor.get_absolute_free_range(id1)))[0]
            editor._track_permutation[-1].append(free_pos) # type: ignore
            id1.element = None
            flag = False
            break
        if flag:
            continue
        for arg in identities[:i1] + identities[i1+1:]:
            arg.element = None
            removed.extend(range(*editor.get_absolute_free_range(arg)))
        for arg in args:
            arg.indices = [new_diag_ind if j == i else j for j in arg.indices]
    for j, e in enumerate(editor.args_with_ind):
        if e.element is None:
            editor._track_permutation[j] = None # type: ignore
    editor._track_permutation = [i for i in editor._track_permutation if i is not None] # type: ignore
    # Renumber permutation array form in order to deal with deleted positions:
    remap = {e: i for i, e in enumerate(sorted({k for j in editor._track_permutation for k in j}))}
    editor._track_permutation = [[remap[j] for j in i] for i in editor._track_permutation]
    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    new_expr = editor.to_array_contraction()
    return new_expr, removed
