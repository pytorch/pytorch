# mypy: ignore-errors
import copy
import itertools
from torch.fx.experimental.migrate_gradual_types.constraint_generator import BinConstraintT, MAX_TENSOR_RANK
from torch.fx.experimental.migrate_gradual_types.constraint import T, BinConstraintD, Conj, Constraint, DVar, TVar, \
    Transpose
from torch.fx.experimental.migrate_gradual_types.constraint import Disj, TGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import DGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import CalcConv, CalcMaxPool
from torch.fx.experimental.migrate_gradual_types.constraint import CalcProduct, CanReshape
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, Prod, F, GetItem, GetItemTensor, IndexSelect
from torch.fx.experimental.migrate_gradual_types.operation import op_eq, op_precision, op_leq, op_matching
from torch.fx.experimental.migrate_gradual_types.operation import op_consistency, op_neq
from torch.fx.experimental.migrate_gradual_types.operation import op_mul, op_add, op_sub, op_div, op_mod
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar
from torch.fx.tensor_type import TensorType, Dyn
from typing import Callable, Dict, List

_TRANSFORMATION_RULES: Dict[Constraint, Callable] = {}


def register_transformation_rule(call_target):
    def register(fn):
        if call_target in _TRANSFORMATION_RULES:
            raise RuntimeError(f'Transformation rule already registered for {call_target}!')
        _TRANSFORMATION_RULES[call_target] = fn
        return fn
    return register


def valid_index(index, dims):
    """
    Given a list of dimensions, checks if an index is valid in the list
    """
    try:
        dims[index]
        return T()
    except IndexError:
        return F()


@register_transformation_rule(Transpose)
def transform_transpose(constraint, counter):
    """
    Similar to a sequence of two index-selects
    """
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    is_valid_index1 = valid_index(constraint.index1, dims)
    is_valid_index2 = valid_index(constraint.index2, dims)
    new_dims = copy.deepcopy(dims)
    nat_constraints = gen_nat_constraints(dims)

    if is_valid_index1 == T() and is_valid_index2 == T():
        new_dims[constraint.index1] = dims[constraint.index2]
        new_dims[constraint.index2] = dims[constraint.index1]

    transformed_constraint = Conj([BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
                                   *nat_constraints,
                                   is_valid_index1, is_valid_index2,
                                   BinConstraintT(constraint.output, TensorType(new_dims), op_eq)])
    return transformed_constraint, counter


@register_transformation_rule(IndexSelect)
def transform_index_select(constraint, counter):
    """
    The constraints consider the given tensor size, checks if the index is valid
    and if so, generates a constraint for replacing the input dimension
    with the required dimension
    """
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    is_valid_index = valid_index(constraint.index, dims)
    nat_constraints = gen_nat_constraints(dims)

    # if the index is valid then replace the input dimension with the new dimension
    # otherwise the dimension will not be replaced and the clause will contain False
    if is_valid_index == T():
        new_dims = copy.deepcopy((dims))
        new_dims[constraint.index] = constraint.dim_replace

    transformed_constraint = Conj([BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
                                   *nat_constraints,
                                   is_valid_index,
                                   BinConstraintT(constraint.output, TensorType(new_dims), op_eq)])

    # print(constraints)
    return transformed_constraint, counter


@register_transformation_rule(GetItem)
def transform_get_item(constraint, counter):
    """
    generate an equality of the form:
    t = [a1, ..., an]
    then generate constraints that check if the given index is valid
    given this particular tensor size.
    If the index is valid, generate a constraint to get the item
    Note that we already handled the Dyn input case in the previous
    step.
    Args:
        constraint: GetItem which assumes we are getting an item from a tensor (not Dyn)
        counter: variable tracking
    Returns: simplified constraints for GetItem

    """
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    nat_constraints = gen_nat_constraints(dims)


    is_valid_index = valid_index(constraint.index, dims)

    all_constraints = [BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
                       *nat_constraints,
                       is_valid_index]

    # if the index is valid, we generate a constraint for getting an item
    # otherwise this clause will have been UNSAT due to the wrong index
    if is_valid_index == T():
        all_constraints.append(BinConstraintD(constraint.res, dims[constraint.index], op_eq))

    return Conj(all_constraints), counter

def valid_index_tensor(index, dims):
    """
    if the slice instances exceed the length of the dimensions
    then this is a type error so we return False
    """
    slice_count = 0
    for s in index:
        if isinstance(s, slice):
            slice_count += 1
    if slice_count > len(dims):
        return F()
    else:
        return T()

@register_transformation_rule(GetItemTensor)
def transform_get_item_tensor(constraint, counter):
    """
    When the index is a tuple, then the output will be a tensor
    TODO: we have to check if this is the case for all HF models

    The cases we are covrering here are a tuple with one of:
     - slice with default argument
     - None

     None appends 1 to the input tensor dimensions
     so each occurrence of 'None' increases the rank by 1

     slice with default arguments does not change the rank
    """
    assert isinstance(constraint.index_tuple, tuple)


    # generate a result tensor of the expected size
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    nat_constraints = gen_nat_constraints(dims)

    # generate a place-holder list of the right rank
    # where "slice" does not contribute to the rank and "None" does
    none_c = constraint.index_tuple.count(None)
    resulting_tensor_dims = (none_c + len(dims)) * [None]

    dim_index = 0
    for i in range(len(constraint.index_tuple)):

        # append 1 to the right location of the resulting tensor
        if constraint.index_tuple[i] is None:
            resulting_tensor_dims[i] = 1

        elif constraint.index_tuple[i] == slice(None, None, None):
            pass

        else:
            raise NotImplementedError('Method not yet implemented')

    # append the remaining dimensions to the right location
    dim_index = 0
    for i in range(len(resulting_tensor_dims)):
        if resulting_tensor_dims[i] is None:
            resulting_tensor_dims[i] = dims[dim_index]
            dim_index += 1

    # check if the index is valid
    is_valid_index = valid_index_tensor(constraint.index_tuple, dims)

    # check if the resulting tensor is within bounds
    if len(resulting_tensor_dims) > 4:
        return F(), counter

    else:
        constraints = [BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
                       BinConstraintT(constraint.res, TensorType(resulting_tensor_dims), op_eq),
                       *nat_constraints,
                       is_valid_index]
        return Conj(constraints), counter


@register_transformation_rule(BinConstraintT)
def generate_binconstraint_t(constraint, counter):
    """
    Transform binary constraints for tensors
    """

    # precision constraints
    if constraint.op == op_precision:
        if constraint.lhs == Dyn:
            return T(), counter
        elif isinstance(constraint.lhs, TensorType):
            is_fully_static = all([d != Dyn for d in constraint.lhs.__args__])
            if is_fully_static:
                return BinConstraintT(constraint.lhs, constraint.rhs, op_eq), counter
            else:
                new_dims = []

                for _ in range(len(constraint.lhs.__args__)):
                    dim, counter = gen_dvar(counter)
                    new_dims.append(dim)

                new_dim_constraints = [BinConstraintD(old_dim, new_dim, op_precision) for
                                       new_dim, old_dim in zip(new_dims, constraint.lhs.__args__)] + \
                                      [BinConstraintT(constraint.rhs, TensorType(new_dims), op_eq)] + \
                                      [BinConstraintD(1, new_dim, op_leq) for
                                       new_dim in new_dims]
                return Conj(new_dim_constraints), counter

    # matching
    elif constraint.op == op_matching:
        assert isinstance(constraint.rhs, TensorType)
        d1 = constraint.rhs.__args__[0]
        d2 = constraint.rhs.__args__[1]
        d3 = constraint.rhs.__args__[2]
        d4 = constraint.rhs.__args__[3]

        conj = [BinConstraintT(constraint.lhs, Dyn, op_eq),
                BinConstraintD(d1, Dyn, op_eq),
                BinConstraintD(d2, Dyn, op_eq),
                BinConstraintD(d3, Dyn, op_eq),
                BinConstraintD(d4, Dyn, op_eq)]
        return Disj([Conj(conj),
                     BinConstraintT(constraint.lhs, TensorType([d1, d2, d3, d4]), op_eq)]), counter

    elif constraint.op == op_consistency:
        c_dyn = Disj([BinConstraintT(constraint.lhs, Dyn, op_eq), BinConstraintT(constraint.rhs, Dyn, op_eq)])
        [c_tensor_1, c_tensor_2, c_tensor_3, c_tensor_4], counter = gen_consistency_constraints(constraint, counter)

        return Disj([c_dyn, c_tensor_1, c_tensor_2, c_tensor_3, c_tensor_4]), counter

    elif constraint.op == op_leq:
        assert isinstance(constraint.rhs, int)
        disj = [BinConstraintT(constraint.lhs, Dyn, op_eq)]
        for i in range(1, constraint.rhs + 1):
            dims = []
            for j in range(1, i + 1):
                dim_var, counter = gen_dvar(counter)
                dims.append(dim_var)
            disj.append(BinConstraintT(constraint.lhs, TensorType(dims), op_eq))
        return Disj(disj), counter
    else:
        return constraint, counter


@register_transformation_rule(BinConstraintD)
def generate_binconstraint_d(constraint, counter):
    """
    Transform binary constraints for dimensions
    """
    if constraint.op == op_precision:
        if isinstance(constraint.lhs, int):
            return BinConstraintD(constraint.lhs, constraint.rhs, op_eq), counter
        elif constraint.lhs == Dyn:
            return T(), counter

    elif constraint.op == op_consistency:
        return Disj([BinConstraintD(constraint.lhs, constraint.rhs, op_eq),
                     BinConstraintD(constraint.rhs, Dyn, op_eq), BinConstraintD(constraint.lhs, Dyn, op_eq)]), counter

    else:
        return constraint, counter


@register_transformation_rule(Conj)
def generate_conj(constraint, counter):
    """
    Transform conjunctions
    """
    new = []
    for c in constraint.conjucts:
        new_c, counter = transform_constraint(c, counter)
        new.append(new_c)
    return Conj(new), counter


@register_transformation_rule(Disj)
def generate_disj(constraint, counter):
    """
    Transform disjunctions
    """
    new = []
    for c in constraint.disjuncts:
        new_c, counter = transform_constraint(c, counter)
        new.append(new_c)
    return Disj(new), counter


@register_transformation_rule(TGreatestUpperBound)
def generate_gub(constraint, counter):
    """
    Transform greatest upper bound for tensors. Results in equality and Greatest Upper Bound
    on dimensions
    """
    c1 = Conj([Disj([BinConstraintT(constraint.rhs1, Dyn, op_eq),
                     BinConstraintT(constraint.rhs2, Dyn, op_eq)]), BinConstraintT(constraint.res, Dyn, op_eq)])

    [c2, c3, c4, c5], counter = gen_greatest_upper_bound(constraint, counter)

    return Disj([c1, c2, c3, c4, c5]), counter


@register_transformation_rule(DGreatestUpperBound)
def generate_d_gub(constraint, counter):
    """
    Transform greatest upper bound for dimensions into equality constraints
    """
    c1 = Conj([BinConstraintD(constraint.rhs1, Dyn, op_eq), BinConstraintD(constraint.res, constraint.rhs2, op_eq)])
    c2 = Conj([BinConstraintD(constraint.rhs2, Dyn, op_eq), BinConstraintD(constraint.res, constraint.rhs1, op_eq)])
    c3 = Conj([BinConstraintD(constraint.rhs2, constraint.rhs1, op_eq), BinConstraintD(constraint.res, constraint.rhs1, op_eq)])
    return Disj([c1, c2, c3]), counter


@register_transformation_rule(CalcConv)
def generate_calc_conv(constraint, counter):
    d, counter = gen_tensor_dims(4, counter)
    conv_result = TensorType([d[0], d[1], d[2], d[3]])

    # the convolution result is a tensor of size 4
    c1 = BinConstraintT(constraint.conv_result, conv_result, op_eq)

    # the second dimension of the output is equal to the output channels
    c2 = Conj([BinConstraintD(d[1], constraint.c_out, op_eq), BinConstraintD(d[1], Dyn, op_neq)])

    # the input corresponds to the output in the first dimension of the convolution
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)

    c4, c5 = calc_last_two_dims(constraint, d)

    leq_constraints = Conj([BinConstraintD(0, d[0], op_leq),
                            BinConstraintD(0, d[1], op_leq),
                            BinConstraintD(0, d[2], op_leq),
                            BinConstraintD(0, d[3], op_leq)])

    return Conj([c1, c2, c3, c4, c5, leq_constraints]), counter


@register_transformation_rule(CalcMaxPool)
def generate_calc_maxpool(constraint, counter):
    """
    Transform maxpool constraints
    """
    d, counter = gen_tensor_dims(4, counter)
    maxpool_result = TensorType([d[0], d[1], d[2], d[3]])

    # the maxpool result is a tensor of size 4
    c1 = BinConstraintT(constraint.maxpool_result, maxpool_result, op_eq)

    # the input corresponds to the output in the first and second dimension of maxpool
    c2 = BinConstraintD(constraint.matching_constraint[1], d[1], op_eq)
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)
    c4, c5 = calc_last_two_dims(constraint, d)

    leq_constraints = Conj([BinConstraintD(0, d[0], op_leq),
                            BinConstraintD(0, d[1], op_leq),
                            BinConstraintD(0, d[2], op_leq),
                            BinConstraintD(0, d[3], op_leq)])

    return Conj([c1, c2, c3, c4, c5, leq_constraints]), counter


@register_transformation_rule(CalcProduct)
def generate_calc_product(constraint, counter):
    """
    Transform flatten constraints
    """
    start = constraint.start
    end = constraint.end
    dims = constraint.dims_to_flatten
    flattened = constraint.flattened
    n = len(constraint.dims_to_flatten)

    # this will be evaluated right here
    boundary_check = (0 <= start and start < end and end <= n)

    c_boundary = T() if boundary_check else F()

    lhs = dims[0:start]
    rhs = dims[end:]
    mid = dims[start:end]

    all_possibilities = generate_all_int_dyn_dim_possibilities(mid)

    all_constraints = []

    for p in all_possibilities:
        p = list(p)
        # this tells us there is a dynamic variable
        contains_dyn = not(all([constraint.op == op_neq for constraint in p]))
        if contains_dyn:
            mid_var = [Dyn]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq)] + p))
        else:
            new_var, counter = gen_dvar(counter)
            mid_eq_prod = Conj([BinConstraintD(new_var, Prod(mid), op_eq), BinConstraintD(new_var, Dyn, op_neq)])
            mid_var = [new_var]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq), mid_eq_prod] + p))

    return Conj([Disj(all_constraints), c_boundary]), counter


@register_transformation_rule(CanReshape)
def generate_reshape(constraint, counter):
    """
    Transform reshape constraints
    """
    d, counter = gen_tensor_dims(4, counter)

    d1 = d[0]
    d2 = d[1]
    d3 = d[2]
    d4 = d[3]

    target = constraint.target.__args__

    is_fully_static = all([d != Dyn for d in target])

    # dynamic tensor
    c1_dyn = BinConstraintT(constraint.src, Dyn, op_eq)
    c2_tensor1 = BinConstraintT(constraint.src, TensorType([d1]), op_eq)
    c2_tensor2 = BinConstraintT(constraint.src, TensorType([d1, d2]), op_eq)
    c2_tensor3 = BinConstraintT(constraint.src, TensorType([d1, d2, d3]), op_eq)
    c2_tensor4 = BinConstraintT(constraint.src, TensorType([d1, d2, d3, d4]), op_eq)

    d1_eq_dyn = BinConstraintD(d1, Dyn, op_eq)
    d1_neq_dyn = BinConstraintD(d1, Dyn, op_neq)

    d2_eq_dyn = BinConstraintD(d2, Dyn, op_eq)
    d2_neq_dyn = BinConstraintD(d2, Dyn, op_neq)

    d3_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d3_neq_dyn = BinConstraintD(d3, Dyn, op_neq)

    d4_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d4_neq_dyn = BinConstraintD(d3, Dyn, op_neq)

    nat_d1 = BinConstraintD(0, d1, op_leq)
    nat_d2 = BinConstraintD(0, d2, op_leq)
    nat_d3 = BinConstraintD(0, d3, op_leq)
    nat_d4 = BinConstraintD(0, d4, op_leq)

    if is_fully_static:
        # size 1 tensor
        c3_tensor1 = Disj([d1_eq_dyn,
                           (Conj([d1_neq_dyn,
                                  BinConstraintD(d1, Prod(target), op_eq)]))])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])

        # size 2 tensor
        all_tensor_2 = Conj([c2_tensor2, gen_all_reshape_possibilities([d1, d2], target)])

        # size 3 tensor
        all_tensor_3 = Conj([c2_tensor3, gen_all_reshape_possibilities([d1, d2, d3], target)])

        # size 4 tensor
        all_tensor_4 = Conj([c2_tensor4, gen_all_reshape_possibilities([d1, d2, d3, d4], target)])

        return Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]),
                     nat_d1, nat_d2, nat_d3, nat_d4]), counter

    # then there must be exactly one occurrence of dyn
    else:
        new_target = []

        for n in target:
            if n != Dyn:
                new_target.append(n)

        # tensor 1
        c3_tensor1 = Disj([d1_eq_dyn,
                           (Conj([d1_neq_dyn,
                                  is_dim_div_by_target(new_target, d1)]))])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])

        # tensor 2
        c21 = Disj([d1_eq_dyn, d2_eq_dyn])
        c22 = Conj([d1_neq_dyn, d2_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2]))])
        all_tensor_2 = Conj([c2_tensor2, Disj([c21, c22])])

        # tensor 3
        c31 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn])
        c32 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3]))])
        all_tensor_3 = Conj([c2_tensor3, Disj([c31, c32])])

        # tensor 4
        c41 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn, d4_eq_dyn])
        c42 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, d4_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3, d4]))])
        all_tensor_4 = Conj([c2_tensor4, Disj([c41, c42])])

        return Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]),
                     nat_d1, nat_d2, nat_d3, nat_d4]), counter


@register_transformation_rule(ApplyBroadcasting)
def generate_broadcasting(constraint, counter):
    """
    Transform broadcasting constraints
    """
    e11, e12 = constraint.res1, constraint.res2
    e1, e2 = constraint.input1, constraint.input2

    e1_dyn = BinConstraintT(e1, Dyn, op_eq)
    e2_dyn = BinConstraintT(e2, Dyn, op_eq)

    # Introduce dimensions
    e1_equal_e11 = BinConstraintT(e1, e11, op_eq)
    e2_equal_e12 = BinConstraintT(e2, e12, op_eq)

    # dyn possibility
    e1_dyn_constraint = Conj([e1_dyn, e1_equal_e11, e2_equal_e12])
    e2_dyn_constraint = Conj([e2_dyn, e1_equal_e11, e2_equal_e12])

    # tensor possibility
    # generate dimensions to create tensors of size 1
    final_tensor_1_constraint, _, _, nat_dims_1, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 1, counter)

    # generate dimensions to create tensors of size 2
    final_tensor_2_constraint_no_padding, final_tensor_2_constraint_padding_arg1, \
        final_tensor_2_constraint_padding_arg2, nat_dims_2, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 2, counter)

    # generate dimensions to create tensors of size 3
    final_tensor_3_constraint_no_padding, final_tensor_3_constraint_padding_arg1, \
        final_tensor_3_constraint_padding_arg2, nat_dims_3, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 3, counter)

    # generate dimensions to create tensors of size 4
    final_tensor_4_constraint_no_padding, final_tensor_4_constraint_padding_arg1, \
        final_tensor_4_constraint_padding_arg2, nat_dims_4, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 4, counter)

    final_result = Disj([
        e1_dyn_constraint,
        e2_dyn_constraint,
        final_tensor_1_constraint,
        final_tensor_2_constraint_no_padding,
        final_tensor_2_constraint_padding_arg1,
        final_tensor_2_constraint_padding_arg2,
        final_tensor_3_constraint_no_padding,
        final_tensor_3_constraint_padding_arg1,
        final_tensor_3_constraint_padding_arg2,
        final_tensor_4_constraint_no_padding,
        final_tensor_4_constraint_padding_arg1,
        final_tensor_4_constraint_padding_arg2
    ])

    return Conj([final_result, *nat_dims_1, *nat_dims_2, *nat_dims_3, *nat_dims_4]), counter


def transform_constraint(constraint: Constraint, counter: int):
    """
    Transforms a constraint into a simpler constraint.
    Ex: precision and consistency are transformed to equality
    Args:
        constraint: constraint to be transformed
        counter: for variable tracking

    Returns: Constraint

    """
    if type(constraint) in _TRANSFORMATION_RULES:
        return _TRANSFORMATION_RULES[type(constraint)](constraint, counter)

    else:
        return constraint, counter




def calc_last_two_dims(constraint, d: List[DVar]):
    """
    Generates constraints for the last two dimensions of a convolution or a maxpool output
    Args:
        constraint: CalcConv or CalcMaxPool
        d: The list of output dimensions

    Returns: Constraints for calculating the last two dimensions of the output

    """

    assert isinstance(constraint, (CalcConv, CalcMaxPool))

    b3 = constraint.matching_constraint[2]
    b4 = constraint.matching_constraint[3]

    b3_dyn = Conj([BinConstraintD(d[2], Dyn, op_eq), BinConstraintD(b3, Dyn, op_eq)])
    b4_dyn = Conj([BinConstraintD(d[3], Dyn, op_eq), BinConstraintD(b4, Dyn, op_eq)])

    d3_not_dyn = Conj([BinConstraintD(d[2], Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq)])
    d4_not_dyn = Conj([BinConstraintD(d[3], Dyn, op_neq), BinConstraintD(b4, Dyn, op_neq)])

    # transform parameters into tuples incase they are not already
    padding = (constraint.padding, constraint.padding) \
        if isinstance(constraint.padding, int) else constraint.padding
    kernel = (constraint.kernel, constraint.kernel) \
        if isinstance(constraint.kernel, int) else constraint.kernel
    stride = (constraint.stride, constraint.stride) \
        if isinstance(constraint.stride, int) else constraint.stride
    dilation = (constraint.dilation, constraint.dilation) \
        if isinstance(constraint.dilation, int) else constraint.dilation

    f1 = BinConstraintD(b3, BinConstraintD(2, padding[0], op_mul), op_add)
    f2 = BinConstraintD(dilation[0], BinConstraintD(kernel[0], 1, op_sub), op_mul)
    f3 = BinConstraintD(BinConstraintD(BinConstraintD(f1, f2, op_sub), 1, op_sub), stride[0], op_div)
    f4 = BinConstraintD(f3, 1, op_add)

    c4 = Disj([b3_dyn, Conj([d3_not_dyn, BinConstraintD(d[2], f4, op_eq)])])

    f11 = BinConstraintD(b4, BinConstraintD(2, padding[1], op_mul), op_add)
    f22 = BinConstraintD(dilation[1], BinConstraintD(kernel[1], 1, op_sub), op_mul)
    f33 = BinConstraintD(BinConstraintD(BinConstraintD(f11, f22, op_sub), 1, op_sub), stride[1], op_div)
    f44 = BinConstraintD(f33, 1, op_add)

    c5 = Disj([b4_dyn, Conj([d4_not_dyn, BinConstraintD(d[3], f44, op_eq)])])

    return c4, c5


def generate_all_int_dyn_dim_possibilities(my_list: List[DVar]):
    """
    Generate all possibilities of being equal or not equal to dyn for my_list
    Args:
        my_list: List of tensor dimensions

    Returns: A list of a list of constraints. Each list of constraints corresponds to
    one possibility about the values of the dimension variables
    """
    # generate all possibilities of being equal or not equal to dyn for my_list
    eq_possibilities = [BinConstraintD(my_list[i], Dyn, op_eq) for i in range(len(my_list))]
    neq_possibilities = [BinConstraintD(my_list[i], Dyn, op_neq) for i in range(len(my_list))]
    d_possibilities = []

    for i in zip(eq_possibilities, neq_possibilities):
        d_possibilities.append(list(i))
    all_possibilities = list(itertools.product(*d_possibilities))
    return all_possibilities


def is_target_div_by_dim(target: List[int], dim: List[DVar]):
    """
    Generate constraints to check if the target dimensions are divisible by the input dimensions
    Args:
        target: Target dimensions
        dim: Input dimensions

    Returns: Constraints to check divisibility

    """
    return BinConstraintD(BinConstraintD(Prod(target), dim, op_mod), 0, op_eq)


def is_dim_div_by_target(target: List[int], dim: List[DVar]):
    """
    Generate constraints to check if the input dimensions is divisible by the target dimensions
    Args:
        target: Target dimensions
        dim:  Input dimensions

    Returns: Constraints to check divisibility

    """
    return BinConstraintD(BinConstraintD(dim, Prod(target), op_mod), 0, op_eq)


def gen_all_reshape_possibilities(list_of_dims, target):
    """
    Consider all possibilities what the input dimensions could be (number or dynamic)
    Then generate the appropriate constraints using multiplication or mod depending on the possibility
    The possibilities we consider here are the cross product of being equal to dyn or not equal to dyn
    for the input. Target is fixed because at most one dimension could be dyn.
    We have different cases for this.

    Args:
        list_of_dims: The input list of dimensions
        target: The tensor we want to reshape to

    Returns: A disjuncition of transformed reshape constraints

    """
    all_possibilities = generate_all_int_dyn_dim_possibilities(list_of_dims)

    all_constraints = []

    for p in all_possibilities:
        to_multiply = []

        p = list(p)

        for constraint in p:
            assert isinstance(constraint, BinConstraintD)
            if constraint.op == op_neq:
                to_multiply.append(constraint.lhs)

        if not to_multiply:
            all_constraints.append(Conj(p))

        elif len(to_multiply) < len(list_of_dims):
            all_constraints.append(Conj(p + [is_target_div_by_dim(target, Prod(to_multiply))]))
        else:
            all_constraints.append(Conj(p + [BinConstraintD(Prod(list_of_dims),
                                                            Prod(target), op_eq)]))

    return Disj(all_constraints)


def broadcast_dim(tensor_input1, tensor_input2, res1, res2, index, padding=False):
    """
    Apply broadcasting to the 'index' dimension of tensor_input1.
    Args:
        tensor_input1: should represent [d1, ..., d_index, ...] where d_index = 1
        tensor_input2: represents the second input
        res1: broadcasted result 1
        res2: broadcasted result 2
        index: the index to broadcast
        padding: If padding was used, then tensor_input1[index] does not exist

    Returns:

    """
    if tensor_input1[index] is None:
        assert padding


    if not padding:
        # then the inputs are the same length so they all have dimensions at "index"
        return Conj([BinConstraintD(tensor_input1[index], 1, op_eq),
                     BinConstraintD(res1[index], res2[index], op_eq),
                     BinConstraintD(res2[index], tensor_input2[index], op_eq)])

    else:
        # we don't set the input dimension to 1, since it doesn't exist.
        return Conj([BinConstraintD(res1[index], res2[index], op_eq),
                     BinConstraintD(res2[index], tensor_input2[index], op_eq)])


def apply_padding(e1_var: TVar,
                  e11: BinConstraintT,
                  e2: BinConstraintT,
                  e12: BinConstraintT,
                  d2: List[DVar],
                  d11: List[DVar],
                  d12: List[DVar],
                  counter: int):
    """
    We are considering the possibility where one input has less dimensions than
    another input, so we apply padding to the broadcasted results

    Args:
        e1_var: Variable representing the first input where padding will be
        e11: constraint of the form e11 = Tensortype[d1, ..., dn]
        e2:  constraint of the form e2 = Tensortype[d1, ..., dn]
        e12: constraint of the form e11 = Tensortype[d1, ..., dn]
        d2: Tensor variables for the second input
        d11: Tensor variables for the broadcasted first input
        d12: Tensor variables for the broadcasted second input
        counter: variable tracking

    Returns: A new constraint whose goal is to apply padding to the broadcasted result

    """

    res = []

    # pad the shorter input with None so we can pass it to the broadcasting helper function
    for i in range(1, len(d2)):

        d1, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(d1 + d2 + d11 + d12)

        e1 = BinConstraintT(e1_var, TensorType(d1), op_eq)

        simulate_padding = [None] * (len(d2) - i)

        assert len(simulate_padding + d1) == len(d2)

        broadcast_padding = []

        # for every padding size, we also consider broadcasting
        for j in range((len(d2) - i)):
            broadcast_padding.append(broadcast_dim(simulate_padding, d2, d11, d12, j, True))

        # we consider the possibilities for broadcasting for every dimension. Since we already
        # padded d1, we do not consider it while broadcasting
        all_broadcasting_possibilities = generate_all_broadcasting_possibilities_no_padding(d1,
                                                                                            d2[(len(d2) - i):],
                                                                                            d11[(len(d2) - i):],
                                                                                            d12[(len(d2) - i):])
        # combine all constraints into a conjunction
        c = Conj([e1, e11, e2, e12,
                  *broadcast_padding,
                  all_broadcasting_possibilities,
                  *nat_constraints
                  ])
        res.append(c)

    return Disj(res), counter


def no_broadcast_dim_with_index(d1: List[DVar],
                                d2: List[DVar],
                                d3: List[DVar],
                                d4: List[DVar],
                                i: int):
    """
    Args:
        d1: inpput 1
        d2: inpput 2
        d3: simulated broadcasting for input 1
        d4: simulated broadcasting for input 2
        i: the rank of the resulting tensor addition

    Returns: Constraints for when no broadcasting occurs
    """
    return Conj([
        Disj([
            Conj([BinConstraintD(d1[i], 1, op_eq),
                  BinConstraintD(d2[i], 1, op_eq)]),

            Conj([BinConstraintD(d1[i], 1, op_neq),
                  BinConstraintD(d2[i], 1, op_neq)])]),

        BinConstraintD(d1[i], d3[i], op_eq),
        BinConstraintD(d2[i], d4[i], op_eq)])



def gen_lists_of_dims(num_tensors: int, dim_size: int, counter: int):
    """
    Generate lists of DVar to represent tensor dimensions
    Args:
        num_tensors: the required number of tensors
        dim_size: the number of dimensions for each tensor
        counter: variable tracking

    Returns: A list of a list of tensor dimensions

    """
    res = []

    for _ in range(num_tensors):
        dims, counter = gen_tensor_dims(dim_size, counter)
        res.append(dims)

    return res, counter


def create_equality_constraints_for_broadcasting(e1: TVar,
                                                 e2: TVar,
                                                 e11: TVar,
                                                 e12: TVar,
                                                 d1: List[DVar],
                                                 d2: List[DVar],
                                                 d11: List[DVar],
                                                 d12: List[DVar]):
    """
    Create equality constraints for when no broadcasting occurs
    Args:
        e1: Input 1
        e2: Input 2
        e11: Broadcasted input 1
        e12: Broadcasted input 2
        d1: Variables that store dimensions for e1
        d2: Variables that store dimensions for e2
        d11: Variables that store dimensions for e11
        d12: Variables that store dimensions for e22

    Returns: Four equality constraints

    """

    e1_tensor = BinConstraintT(e1, TensorType(d1), op_eq)
    e11_tensor = BinConstraintT(e11, TensorType(d11), op_eq)
    e2_tensor = BinConstraintT(e2, TensorType(d2), op_eq)
    e12_tensor = BinConstraintT(e12, TensorType(d12), op_eq)
    return [e1_tensor, e11_tensor, e2_tensor, e12_tensor]


def gen_consistency_constraints(constraint: Constraint, counter: int):
    """
    Args:
        constraint: Consistency constraint on tensors
        counter: for variable tracking

    Returns: Equality and consistency constraints on dimensions

    """

    all_constraints = []

    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)

        c_tensor_i = Conj([BinConstraintT(constraint.lhs, TensorType(new_dims_rhs_1), op_eq),
                           BinConstraintT(constraint.rhs, TensorType(new_dims_rhs_2), op_eq)] +
                          [BinConstraintD(d1, d2, op_consistency) for
                           d1, d2 in zip(new_dims_rhs_1, new_dims_rhs_2)] + nat_constraints)

        all_constraints.append(c_tensor_i)

    return all_constraints, counter


def gen_greatest_upper_bound(constraint: TGreatestUpperBound, counter: int):
    """
    Args:
        constraint: Greatest upper bound on tensors
        counter: variable tracking

    Returns: A set of equality constraints and DGreatestUpperBound constraints

    """

    all_constraints = []

    for i in range(1, MAX_TENSOR_RANK + 1):
        c = []
        dims1, counter = gen_tensor_dims(i, counter)
        c1tensor = TensorType(dims1)

        dims2, counter = gen_tensor_dims(i, counter)
        c2tensor = TensorType(dims2)

        dims3, counter = gen_tensor_dims(i, counter)
        c3tensor = TensorType(dims3)

        c += [BinConstraintT(constraint.rhs1, c1tensor, op_eq),
              BinConstraintT(constraint.rhs2, c2tensor, op_eq),
              BinConstraintT(constraint.res, c3tensor, op_eq)] + \
            gen_nat_constraints(dims1 + dims2 + dims3)

        assert len(c3tensor.__args__) == len(c1tensor.__args__) == len(c2tensor.__args__)
        for i in range(len(c3tensor.__args__)):
            c.append(DGreatestUpperBound(c3tensor.__args__[i],
                                         c1tensor.__args__[i],
                                         c2tensor.__args__[i]))

        all_constraints.append(Conj(c))
    return all_constraints, counter


def generate_all_broadcasting_possibilities_no_padding(d1: List[DVar], d2: List[DVar], d11: List[DVar], d12: List[DVar]):
    """
    Generate broadcasting constraints assuming no padding. Broadcasting can happen at any dimension.
    We look at all combinations for all dimendions in d1 and d2
    Args:
        d1: input1 dimensions
        d2: input2 dimensions
        d11: broadcasted input1 dimensions
        d12: broadcasted input2 dimensions

    Returns: broadcasting constraints relating the input dimensions to the broadcasted dimensions

    """

    size = len(d1)

    res2 = []

    for i in range(size):
        t1 = broadcast_dim(d1, d2, d11, d12, i)
        t2 = broadcast_dim(d2, d1, d12, d11, i)
        t3 = no_broadcast_dim_with_index(d1, d2, d11, d12, i)

        res2.append(Disj([t1, t2, t3]))

    return Conj(res2)


def gen_broadcasting_constraints(e1: TVar, e2: TVar, e11: TVar, e12: TVar, i: int, counter: int):
    """
    Simulates broadcasting on e1 and e2 and returns the results
    respectively in e11 and e12. Because of gradual types,
    e1 and e2 may not be equal. Similarly, e11 and e12 may not
    be equal. e11 and e12 should be guaranteed to be consistent
    as they represent the shapes of the tensors to be added after
    broadcasting.
    Args:
        e1: TVar representing the type of input 1
        e2: TVar representing the type of input 2
        e11: TVar representing the representing broadcasted input 1
        e12: TVar representing the representing broadcasted input 2
        i: The rank of the resulting type of addition
        counter: for variable tracking

    Returns: Simplified broadcasting constraints

    """
    dims, counter = gen_lists_of_dims(4, i, counter)
    [d1, d2, d3, d4] = dims
    nat_dims_i = gen_nat_constraints(list(itertools.chain(*dims)))

    initialize_tensors_constraints = create_equality_constraints_for_broadcasting(e1, e2, e11, e12,
                                                                                  d1, d2, d3, d4)

    [e1_tensor, e11_tensor, e2_tensor, e12_tensor] = initialize_tensors_constraints

    # without padding, broadcast all possibilities for tensors of size i
    final_tensor_constraint_no_padding = Conj([*initialize_tensors_constraints,
                                               generate_all_broadcasting_possibilities_no_padding(d1, d2, d3, d4)])

    # with padding, broadcast all possibilities for tensors of size i
    final_tensor_constraint_padding_arg1, counter = \
        apply_padding(e1, e11_tensor, e2_tensor, e12_tensor, d2, d3, d4, counter)

    final_tensor_constraint_padding_arg2, counter = \
        apply_padding(e2, e12_tensor, e1_tensor, e11_tensor, d1, d4, d3, counter)

    return final_tensor_constraint_no_padding, \
        final_tensor_constraint_padding_arg1, \
        final_tensor_constraint_padding_arg2, nat_dims_i, counter
