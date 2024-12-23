import collections.abc
import operator
from itertools import accumulate

from sympy import Mul, Sum, Dummy, Add
from sympy.tensor.array.expressions import PermuteDims, ArrayAdd, ArrayElementwiseApplyFunc, Reshape
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, get_rank, ArrayContraction, \
    ArrayDiagonal, get_shape, _get_array_element_or_slice, _ArrayExpr
from sympy.tensor.array.expressions.utils import _apply_permutation_to_list


def convert_array_to_indexed(expr, indices):
    return _ConvertArrayToIndexed().do_convert(expr, indices)


class _ConvertArrayToIndexed:

    def __init__(self):
        self.count_dummies = 0

    def do_convert(self, expr, indices):
        if isinstance(expr, ArrayTensorProduct):
            cumul = list(accumulate([0] + [get_rank(arg) for arg in expr.args]))
            indices_grp = [indices[cumul[i]:cumul[i+1]] for i in range(len(expr.args))]
            return Mul.fromiter(self.do_convert(arg, ind) for arg, ind in zip(expr.args, indices_grp))
        if isinstance(expr, ArrayContraction):
            new_indices = [None for i in range(get_rank(expr.expr))]
            limits = []
            bottom_shape = get_shape(expr.expr)
            for contraction_index_grp in expr.contraction_indices:
                d = Dummy(f"d{self.count_dummies}")
                self.count_dummies += 1
                dim = bottom_shape[contraction_index_grp[0]]
                limits.append((d, 0, dim-1))
                for i in contraction_index_grp:
                    new_indices[i] = d
            j = 0
            for i in range(len(new_indices)):
                if new_indices[i] is None:
                    new_indices[i] = indices[j]
                    j += 1
            newexpr = self.do_convert(expr.expr, new_indices)
            return Sum(newexpr, *limits)
        if isinstance(expr, ArrayDiagonal):
            new_indices = [None for i in range(get_rank(expr.expr))]
            ind_pos = expr._push_indices_down(expr.diagonal_indices, list(range(len(indices))), get_rank(expr))
            for i, index in zip(ind_pos, indices):
                if isinstance(i, collections.abc.Iterable):
                    for j in i:
                        new_indices[j] = index
                else:
                    new_indices[i] = index
            newexpr = self.do_convert(expr.expr, new_indices)
            return newexpr
        if isinstance(expr, PermuteDims):
            permuted_indices = _apply_permutation_to_list(expr.permutation, indices)
            return self.do_convert(expr.expr, permuted_indices)
        if isinstance(expr, ArrayAdd):
            return Add.fromiter(self.do_convert(arg, indices) for arg in expr.args)
        if isinstance(expr, _ArrayExpr):
            return expr.__getitem__(tuple(indices))
        if isinstance(expr, ArrayElementwiseApplyFunc):
            return expr.function(self.do_convert(expr.expr, indices))
        if isinstance(expr, Reshape):
            shape_up = expr.shape
            shape_down = get_shape(expr.expr)
            cumul = list(accumulate([1] + list(reversed(shape_up)), operator.mul))
            one_index = Add.fromiter(i*s for i, s in zip(reversed(indices), cumul))
            dest_indices = [None for _ in shape_down]
            c = 1
            for i, e in enumerate(reversed(shape_down)):
                if c == 1:
                    if i == len(shape_down) - 1:
                        dest_indices[i] = one_index
                    else:
                        dest_indices[i] = one_index % e
                elif i == len(shape_down) - 1:
                    dest_indices[i] = one_index // c
                else:
                    dest_indices[i] = one_index // c % e
                c *= e
            dest_indices.reverse()
            return self.do_convert(expr.expr, dest_indices)
        return _get_array_element_or_slice(expr, indices)
