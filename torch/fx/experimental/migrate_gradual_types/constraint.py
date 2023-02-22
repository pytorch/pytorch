# -*- coding: utf-8 -*-
from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
    op_mod, op_gt, op_lt, op_neq, op_eq
from torch.fx.tensor_type import TensorType, Dyn


class Constraint:
    pass


class Conj(Constraint):
    def __init__(self, conjuncts):
        """
        :param conjuncts: Conjuction of constraints
        """
        self.conjucts = conjuncts

    def __eq__(self, other):
        if isinstance(other, Conj):
            return self.conjucts == other.conjucts and self.conjucts == other.conjucts
        else:
            return False

    def __repr__(self):
        return f'And({self.conjucts})'


class Disj(Constraint):
    def __init__(self, disjuncts):
        """
        :param disjuncts: Disjunction of constraints
        """
        self.disjuncts = disjuncts

    def __eq__(self, other):
        if isinstance(other, Disj):
            return self.disjuncts == other.disjuncts and self.disjuncts == other.disjuncts
        else:
            return False

    def __repr__(self):
        return f'Or({self.disjuncts})'


class Prod(Constraint):
    def __init__(self, products):
        """
        :param products: lists of dimensions to multiply
        """
        self.products = products

    def __eq__(self, other):
        if isinstance(other, Prod):
            return self.products == other.products and self.products == other.products
        else:
            return False

    def __repr__(self):
        return f'Product({self.products})'


class T(Constraint):
    """
    True
    """
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, T)

    def __repr__(self):
        return 'True'

class F(Constraint):
    """
    False
    """
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, F)

    def __repr__(self):
        return 'False'


class BinaryConstraint(Constraint):
    """
    Represents all binary operations
    """
    def __init__(self, lhs, rhs, op):
        """
        :param lhs: lhs of the constraint
        :param rhs: rhs of the constraint
        :param op: string reprsenting the operation
        """
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __eq__(self, other):
        if isinstance(other, BinaryConstraint):
            return self.lhs == other.lhs and self.rhs == other.rhs and self.op == other.op
        else:
            return False

    def __repr__(self):
        return f'({self.lhs} {self.op} {self.rhs})'


class BinConstraintT(BinaryConstraint):
    """
    Binary constraints about tensors
    """
    def __init__(self, lhs, rhs, op):
        assert (isinstance(lhs, (TVar, TensorType, int)) or lhs == Dyn) and \
               (isinstance(rhs, (TVar, TensorType, int)) or rhs == Dyn)
        super().__init__(lhs, rhs, op)

    def __eq__(self, other):
        return super().__eq__(other)


class BinConstraintD(BinaryConstraint):
    """
    Binary constraints about dimensions
    """
    def __init__(self, lhs, rhs, op):
        assert is_algebraic_expression(lhs) or is_dim(lhs) or is_bool_expr(lhs)
        assert is_algebraic_expression(rhs) or is_dim(rhs) or is_bool_expr(rhs)

        super().__init__(lhs, rhs, op)

    def __eq__(self, other):
        return super().__eq__(other)



class TGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for tensors with dynamic type
    """
    def __init__(self, res, rhs1, rhs2):
        """
        :param res: tensor variable that stores the result of the outout
        :param rhs1: tensor or tensor variable
        :param rhs2: tensor or tensor variabke
        """
        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self):
        return f'{self.res} = {self.rhs1}⊔*{self.rhs2}'

    def __eq__(self, other):
        if isinstance(other, TGreatestUpperBound):
            return self.res == other.res and self.rhs1 == other.rhs1 and self.rhs2 == other.rhs2
        else:
            return False


class DGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for dimensions
    """
    def __init__(self, res, rhs1, rhs2):
        """
        :param res: Dimension variable to store the result
        :param rhs1: dimension variable 1
        :param rhs2: dimension variable 2
        """
        assert is_dim(res)
        assert is_dim(rhs1)
        assert is_dim(rhs2)

        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self):
        return f'{self.res} = {self.rhs1}⊔{self.rhs2}'

    def __eq__(self, other):
        if isinstance(other, DGreatestUpperBound):
            return self.res == other.res and self.rhs1 == other.rhs1 and self.rhs2 == other.rhs2
        else:
            return False


class CanReshape(Constraint):
    """
    can_reshape constraint
    """
    def __init__(self, src, target):
        """
        :param src: tensor variable
        :param target: tensor
        """
        self.src = src
        self.target = target

    def __repr__(self):
        return f'can-reshape({self.src}, {self.target})'

    def __eq__(self, other):
        if isinstance(other, CanReshape):
            return self.src == other.src and self.target == other.target
        else:
            return False


class IndexSelect(Constraint):

    def __init__(self, tensor_size, input_var, dim_replace, index, output):
        """
        Args:
            input_var: input to index_select
            tensor_size: tensor size we are considering
            dim_replace: the dimension of the output at "index"
            index: location of the dimensions to replace in the input
            outut: variable to store the result
        """
        assert isinstance(input_var, TVar)
        assert isinstance(output, TVar)
        assert isinstance(dim_replace, DVar) or dim_replace == Dyn
        assert isinstance(index, int)

        self.input_var = input_var
        self.tensor_size = tensor_size
        self.dim_replace = dim_replace
        self.index = index
        self.output = output

    def __repr__(self):

        return f' {self.output} = ' \
               f'IndexSelect({self.input_var}, ' \
               f'tensor_size: {self.tensor_size}, ' \
               f'{self.dim_replace}, ' \
               f'{self.index})'

    def __eq__(self, other):
        if isinstance(other, IndexSelect):
            return self.tensor_size == other.tensor_size and \
                self.dim_replace == other.dim_replace and \
                self.index == other.index and \
                self.output == other.output and \
                self.input_var == other.input_var
        else:
            return False


class Transpose(Constraint):

    def __init__(self, tensor_size, input_var, index1, index2, output):
        """
        Args:
            tensor_size: current tensor size
            input_var: variable to hold input
            index1: dimension 1
            index2: dimension 2
            output: output that stores result
        """
        assert isinstance(input_var, TVar)
        assert isinstance(output, TVar)
        assert isinstance(index1, int)
        assert isinstance(index2, int)

        self.input_var = input_var
        self.tensor_size = tensor_size
        self.index1 = index1
        self.index2 = index2
        self.output = output

    def __repr__(self):

        return f' {self.output} = ' \
               f'Transpose({self.input_var}, ' \
               f'tensor_size: {self.tensor_size}, ' \
               f'{self.index1}, ' \
               f'{self.index2})'

    def __eq__(self, other):
        if isinstance(other, Transpose):
            return self.tensor_size == other.tensor_size and \
                self.index1 == other.index1 and \
                self.index2 == other.index2 and \
                self.output == other.output and \
                self.input_var == other.input_var
        else:
            return False


class GetItem(Constraint):

    def __init__(self, tensor_size, index, res, input_var):
        """
        Constraint for getting item given a tensor size
        :param tensor_size: actual number
        :param index: actual number representing the index
        :param res: dimension variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
        assert isinstance(res, DVar)

        self.res = res
        self.tensor_size = tensor_size
        self.index = index
        self.input_var = input_var

    def __repr__(self):
        return f' {self.res} = GetItem({self.input_var}, tensor_size: {self.tensor_size}, {self.index})'

    def __eq__(self, other):
        if isinstance(other, GetItem):
            return self.res == other.res and \
                self.tensor_size == other.tensor_size and \
                self.index == other.index and \
                self.input_var == other.input_var
        else:
            return False

class GetItemTensor(Constraint):

    def __init__(self, tensor_size, index_tuple, res, input_var):
        """
        Constraint for getting item given a tensor size
        However, when the argument is a tuple, we will
        expect a tensor
        :param tensor_size: actual number representing the rank
        :param index_tuple: tuple for indexing
        :param res: tensor variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
        assert isinstance(res, TVar)

        self.res = res
        self.tensor_size = tensor_size
        self.index_tuple = index_tuple
        self.input_var = input_var

    def __repr__(self):
        return f' {self.res} = GetItemT({self.input_var}, tensor_size: {self.tensor_size}, {self.index_tuple})'

    def __eq__(self, other):
        if isinstance(other, GetItemTensor):
            return self.res == other.res and \
                self.tensor_size == other.tensor_size and \
                self.index_tuple == other.index_tuple and \
                self.input_var == other.input_var
        else:
            return False

class CalcConv(Constraint):

    def __init__(self, conv_result, input_var, c_out, kernel, padding, stride, dilation, matching_constraint_vars):
        """
        :param conv_result: the convolution result
        :param input_var: input to convolution
        :param c_out: output chanel type
        :param kernel: kernel tuple
        """
        self.conv_result = conv_result
        self.input_var = input_var
        self.c_out = c_out
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self):
        return f'{self.conv_result} =' \
               f' calc-conv({self.input_var},' \
               f' {self.c_out}, {self.kernel}, ' \
               f'{self.padding}, {self.stride},' \
               f' {self.dilation})'

    def __eq__(self, other):
        if isinstance(other, CalcConv):
            return self.conv_result == other.conv_result and self.input_var == other.input_var and \
                self.c_out == other.c_out and self.kernel == other.kernel and self.padding == other.padding \
                and self.stride == other.stride and self.dilation == other.dilation \
                and self.matching_constraint == other.matching_constraint
        else:
            return False


class CalcMaxPool(Constraint):

    def __init__(self, maxpool_result, input_var, kernel, padding, stride, dilation, matching_constraint_vars):
        """
        :param maxpool_result: the result of maxpool
        :param input_var: input to convolution
        :param kernel: kernel tuple
        """
        self.maxpool_result = maxpool_result
        self.input_var = input_var
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self):
        return f'{self.maxpool_result} =' \
               f' calc-maxpool({self.input_var},' \
               f'  {self.kernel}, ' \
               f'{self.padding}, {self.stride},' \
               f' {self.dilation})'

    def __eq__(self, other):
        if isinstance(other, CalcMaxPool):
            return self.maxpool_result == other.maxpool_result and self.input_var == other.input_var \
                and self.kernel == other.kernel and self.padding == other.padding \
                and self.stride == other.stride and self.dilation == other.dilation \
                and self.matching_constraint == other.matching_constraint
        else:
            return False


class ApplyBroadcasting(Constraint):
    def __init__(self, res1, res2, input1, input2):
        """
        :param res1: resulting tensor 1
        :param res2: resulting tensor 2
        :param input1: tensor variable 1
        :param input2: tensor variable 2
        """
        self.res1 = res1
        self.res2 = res2
        self.input1 = input1
        self.input2 = input2

    def __eq__(self, other):
        if isinstance(other, ApplyBroadcasting):
            return self.res1 == other.res1 \
                and self.res2 == other.res2 \
                and self.input1 == other.input1 \
                and self.input2 == other.input2
        else:
            return False

    def __repr__(self):
        return f'{self.res1}, {self.res2} ='f' apply-broadcasting({self.input1},' f' {self.input2})'


class CalcProduct(Constraint):
    """
    Given correct dimensions, calculate the product for flatten accounting for Dyn
    """
    def __init__(self, start, end, flattened, dims_to_flatten):
        """
        :param start: start index
        :param end: end index
        :param flattened: variable to store the product
        :param dims_to_flatten: the type which we will flatten
        """
        assert isinstance(dims_to_flatten, list)
        assert isinstance(flattened, TVar)
        assert isinstance(start, int)
        assert isinstance(end, int)

        self.start = start
        self.end = end
        self.dims_to_flatten = dims_to_flatten
        self.flattened = flattened

    def __eq__(self, other):
        if isinstance(other, CalcProduct):
            return self.start == other.start and self.end == other.end and \
                self.dims_to_flatten == other.dims_to_flatten and self.flattened == other.flattened

        else:
            return False

    def __repr__(self):
        return f'{self.flattened} = CalcProduct({self.start}, {self.end}, {self.dims_to_flatten})'


class TVar:
    """
    Tensor variable with no tensor constructor
    """
    def __init__(self, tvar):
        """
        :param tvar: tensor variable
        """
        self.tvar = tvar

    def __repr__(self):
        return f'TV({self.tvar})'

    def __eq__(self, other):
        if isinstance(other, TVar):
            return self.tvar == other.tvar
        else:
            return False


class DVar:
    """
    Dimension variable
    """
    def __init__(self, c):
        """
        :param c: character or number
        """
        self.c = c

    def __repr__(self):
        return f'DV({self.c})'

    def __eq__(self, other):
        if isinstance(other, DVar):
            return self.c == other.c
        else:
            return False


class BVar:
    """
    Boolean variable
    """
    def __init__(self, c):
        """
        :param c: character or number
        """
        self.c = c

    def __repr__(self):
        return f'BV({self.c})'

    def __eq__(self, other):
        if isinstance(other, BVar):
            return self.c == other.c
        else:
            return False


def is_algebraic_expression(constraint):
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_add, op_sub, op_div, op_mul, op_mod]
    else:
        return isinstance(constraint, Prod)


def is_bool_expr(constraint):
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_gt, op_lt, op_neq, op_eq]
    else:
        return isinstance(constraint, (BVar, Conj, Disj))

def is_dim(d):
    return isinstance(d, (DVar, int)) or d == Dyn
