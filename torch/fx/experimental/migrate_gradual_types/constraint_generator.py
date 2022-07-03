import torch
import operator
from typing import Callable, Dict, Iterable

from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
    Disj, TGreatestUpperBound, CalcMaxPool, CalcConv, Conj, BinConstraintT, CanReshape, BinConstraintD, GetItem
from torch.fx.experimental.migrate_gradual_types.operation import \
    op_eq, op_matching, op_consistency, op_leq, op_precision, op_gt, op_div, op_sub, op_neq
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar

from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d

_INFERENCE_RULES: Dict[Target, Callable] = {}

MAX_TENSOR_RANK = 4

def register_inference_rule(call_target):
    def register(fn):
        if call_target in _INFERENCE_RULES:
            raise RuntimeError(f'Inference rule already registered for {call_target}!')
        _INFERENCE_RULES[call_target] = fn
        return fn
    return register


def generate_flatten_constraints(start_dim, end_dim, input, flattened, n, counter):
    d, counter = gen_tensor_dims(n, counter)
    c1 = BinConstraintT(input, TensorType(d), op_eq)
    start_dim = n if start_dim == -1 else abs(start_dim)
    end_dim = n + end_dim + 1 if end_dim < 0 else end_dim + 1
    c2 = CalcProduct(start_dim, end_dim, flattened, d)
    nat_constraints = gen_nat_constraints(d)
    return Conj([c1, c2, *nat_constraints]), counter


# TODO
@register_inference_rule("long")
def long_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    raise NotImplementedError('Not yet implemented')

# TODO
@register_inference_rule("type_as")
def type_as_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    raise NotImplementedError('Not yet implemented')

# TODO
@register_inference_rule("int")
def int_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    raise NotImplementedError('Not yet implemented')

# TODO
@register_inference_rule("ne")
def ne_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    raise NotImplementedError('Not yet implemented')

# TODO
@register_inference_rule(getattr)
def get_attr_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    raise NotImplementedError('Not yet implemented')

# TODO:
@register_inference_rule("expand")
def expand_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    raise NotImplementedError('Not yet implemented')


# TODO:
@register_inference_rule("to")
def to_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    raise NotImplementedError('Not yet implemented')


# TODO:
@register_inference_rule("masked_fill_")
def masked_fill_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    raise NotImplementedError('Not yet implemented')


@register_inference_rule(torch.nn.modules.sparse.Embedding)
def embedding_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    """
    The output shape differs from the input shape in the last dimension
    """
    assert isinstance(n.args[0], Node)

    embedding_dim = module_instance.embedding_dim  # number

    embedding_output, counter = gen_tvar(counter)
    symbols[n] = embedding_output
    embedding_input = symbols[n.args[0]]

    input_dyn = BinConstraintT(embedding_input, Dyn, op_eq)
    output_dyn = BinConstraintT(embedding_output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])
    c2 = []

    for i in range(1, MAX_TENSOR_RANK):
        new_dims, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims)

        # we consider all tensor sizes and append embedding_dim to the end of the output dimension in all cases
        c_tensor_i = Conj([BinConstraintT(embedding_input, TensorType(new_dims), op_eq),
                           BinConstraintT(embedding_output, TensorType(new_dims + [embedding_dim]), op_eq)] +
                          nat_constraints)
        c2.append(c_tensor_i)

    return [Disj([c1, Disj(c2)])], counter



@register_inference_rule("view")
def view_inference_rule(n: Node, symbols, constraints, counter):
    """
    Similar to reshape but with an extra condition on the strides
    """
    assert isinstance(n.args[0], Node)

    # generate the new variable
    my_view, counter = gen_tvar(counter)
    symbols[n] = my_view

    src_var = symbols[n.args[0]]
    t2 = n.args[1:]  # target shape
    t2_type = TensorType([Dyn if elem == -1 else symbols[elem] for elem in t2])  # type: ignore[union-attr]
    c1 = BinConstraintT(my_view, t2_type, op_eq)
    c2 = CanReshape(src_var, t2_type)

    # TODO: add the extra check mentioned here:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view

    return [c1, c2], counter


@register_inference_rule("size")
def size_inference_rule(n: Node, symbols, constraints, counter):
    """
    The constraint is just lhs = rhs.
    Ex: size = input_ids.size()
    """


    if len(n.args) == 1:
        # generate the new variable
        size, counter = gen_tvar(counter)
        symbols[n] = size
        input = symbols[n.args[0]]
        c = BinConstraintT(input, size, op_eq)
        return [c], counter

    elif len(n.args) == 2:
        # TODO: review this rule; should input = dyn; output = dyn be included here?
        if isinstance(n.args[1], int):
            # generate the new variable
            size_index, counter = gen_dvar(counter)
            symbols[n] = size_index
            input = symbols[n.args[0]]
            c2 = [GetItem(i + 1, n.args[1], size_index, input) for i in range(MAX_TENSOR_RANK)]
            c3 = BinConstraintD(0, size_index, op_leq)

            input_dyn = BinConstraintT(input, Dyn, op_eq)
            output_dyn = BinConstraintD(size_index, Dyn, op_eq)
            c1 = Conj([input_dyn, output_dyn])

            # print(c1)

            return [Disj([c1, Conj([Disj(c2), c3])])], counter

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


# TODO
@register_inference_rule(torch.cumsum)
def cumsum_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    raise NotImplementedError('Not yet implemented')


@register_inference_rule(_assert_is_none)
def assert_inference_rule(n: Node, symbols, constraints, counter):
    assert len(n.users) == 0
    return [], counter


@register_inference_rule(operator.getitem)
def getitem_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], int)

    # create and store the new variable
    get_item_output, counter = gen_dvar(counter)
    symbols[n] = get_item_output

    # retreive arg variables
    get_item_arg = symbols[n.args[0]]

    # if the input is dynamic, we accept any index and return
    # a dynamic dimension as output
    input_dyn = BinConstraintT(get_item_arg, Dyn, op_eq)
    output_dyn = BinConstraintD(get_item_output, Dyn, op_eq)
    c1 = Conj([input_dyn, output_dyn])

    # if the input is a tensor,
    # generate a getItem constraint which will be expanded based on the
    # tensor dimension.

    c2 = [GetItem(i + 1, n.args[1], get_item_output, get_item_arg) for i in range(MAX_TENSOR_RANK)]


    # since the output is a dimension, we make sure it's a natural number
    # added as a conjunction to the disjuction of c2
    c3 = BinConstraintD(0, get_item_output, op_leq)

    # print(Disj([c1, Conj([Disj(c2), c3])]))
    return [Disj([c1, Conj([Disj(c2), c3])])], counter

@register_inference_rule(operator.mul)
def mul_inference_rule(n: Node, symbols, constraints, counter):

    my_mul, counter = gen_tvar(counter)
    symbols[n] = my_mul

    # since in this case, we have scalar multiplication
    # the input shape should be the same as the output shape
    if isinstance(n.args[0], Node) and isinstance(n.args[1], float):
        # retrieve arg variables
        e1 = symbols[n.args[0]]
        return [BinConstraintT(my_mul, e1, op_eq)], counter
    else:
        raise NotImplementedError('Case not yet implemented')


@register_inference_rule(operator.gt)
def gt_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node) or isinstance(n.args[0], int)
    assert isinstance(n.args[1], Node) or isinstance(n.args[1], int)

    # We make sure this node will not be used again. We do not
    # generate a constraint about that node. Only about the operands.
    # assert len(n.users) == 0
    # print(len(n.users))

    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]
    return [BinConstraintD(e1, e2, op_gt)], counter

# TODO
@register_inference_rule(operator.lt)
def lt_inference_rule(n: Node, symbols, constraints, counter):
    raise NotImplementedError('Not yet implemented')


@register_inference_rule(torch.full)
def full_inference_rule(n: Node, symbols, constraints, counter):
    full, counter = gen_tvar(counter)
    symbols[n] = full
    res = []
    assert isinstance(n.args[0], Iterable)
    for arg in n.args[0]:
        res.append(symbols[arg])
    c = BinConstraintT(full, TensorType(list(res)), op_eq)  # type: ignore[arg-type]
    return [c], counter


# TODO normalize index
@register_inference_rule(torch.arange)
def arange_inference_rule(n: Node, symbols, constraints, counter):
    start = 0
    step = 1

    if len(n.args) == 1:
        end = symbols[n.args[0]]
    else:
        raise NotImplementedError('Not yet implemented')

    # int((end - start) / step)
    d1, counter = gen_dvar(counter)
    size_constraint = BinConstraintD(d1, BinConstraintD(BinConstraintD(end, start, op_sub), step, op_div), op_eq)
    arange, counter = gen_tvar(counter)
    symbols[n] = arange

    # either the a parameter is a number or it is Dyn
    c1 = Disj([BinConstraintD(end, Dyn, op_eq),
               BinConstraintD(start, Dyn, op_eq),
               BinConstraintD(step, Dyn, op_eq)])
    c2 = BinConstraintD(d1, Dyn, op_eq)
    both_dyn = Conj([c1, c2])

    c11 = Conj([BinConstraintD(end, Dyn, op_neq),
                BinConstraintD(start, Dyn, op_neq),
                BinConstraintD(step, Dyn, op_neq)])
    c22 = BinConstraintD(d1, Dyn, op_neq)
    both_numbers = Conj([c11, c22, size_constraint])

    return [BinConstraintT(arange, TensorType([d1]), op_eq), Disj([both_dyn, both_numbers])], counter

@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def add_inference_rule(n: Node, symbols, constraints, counter):

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        # create and store the new variable
        my_add, counter = gen_tvar(counter)
        symbols[n] = my_add

        # retrive arg variables
        e1 = symbols[n.args[0]]
        e2 = symbols[n.args[1]]

        # additional vars that don't correspond to expressions
        e11, counter = gen_tvar(counter)
        e22, counter = gen_tvar(counter)

        # generate constraints
        c1 = TGreatestUpperBound(my_add, e11, e22)
        c2 = ApplyBroadcasting(e11, e22, e1, e2)
        c3 = BinConstraintT(e11, e22, op_consistency)
        return [c1, c2, c3], counter
    else:
        # TODO generate add constraints for scalar addition
        raise NotImplementedError('Addition not yet implemented')


@register_inference_rule(torch.flatten)
def flatten_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    # generate the new variable
    flattened, counter = gen_tvar(counter)
    symbols[n] = flattened

    input = symbols[n.args[0]]

    # set the default start and end dims
    start_dim = 1
    end_dim = -1

    if len(n.args) > 1:
        assert isinstance(n.args[1], int)
        start_dim = n.args[1]

    if len(n.args) > 2:
        assert isinstance(n.args[2], int)
        end_dim = n.args[2]

    c1 = BinConstraintT(input, Dyn, op_eq)
    c2 = BinConstraintT(flattened, Dyn, op_eq)
    both_dyn = Conj([c1, c2])

    const = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        c, counter = generate_flatten_constraints(start_dim, end_dim, input, flattened, i, counter)
        const.append(c)

    return [Disj([both_dyn, *const])], counter

@register_inference_rule(torch.nn.Dropout)
@register_inference_rule(torch.nn.ReLU)
def relu_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    """
    Input and output shapes should be equal.
    """
    assert isinstance(n.args[0], Node)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]
    return [BinConstraintT(input, output, op_eq)], counter

@register_inference_rule(torch.nn.Linear)
def linear_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    """
    Input and output sizes should be the same except for the last dimension
    If the input is Dyn, then so should the output
    """
    assert isinstance(n.args[0], Node)
    linear_output, counter = gen_tvar(counter)
    symbols[n] = linear_output
    linear_input = symbols[n.args[0]]

    input_dyn = BinConstraintT(linear_input, Dyn, op_eq)
    output_dyn = BinConstraintT(linear_output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])

    c2 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)

        c_tensor_i = Conj([BinConstraintT(linear_input, TensorType(new_dims_rhs_1), op_eq),
                           BinConstraintT(linear_output, TensorType(new_dims_rhs_2), op_eq)] +
                          add_linear_constraints(new_dims_rhs_1, new_dims_rhs_2, module_instance) +
                          nat_constraints)
        c2.append(c_tensor_i)


    return [Disj([c1, Disj(c2)])], counter


def add_linear_constraints(dims1, dims2, module_instance):
    assert len(dims1) == len(dims2)
    constraints = []
    for i in range(len(dims1)):
        if i == len(dims1) - 1:
            constraints.append(BinConstraintD(dims1[i], module_instance.in_features, op_consistency))
            constraints.append(BinConstraintD(dims2[i], module_instance.out_features, op_eq))
        else:
            constraints.append(BinConstraintD(dims1[i], dims2[i], op_eq))

    return constraints

# module_instance.out_features
@register_inference_rule(torch.reshape)
def reshape_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    # generate the new variable
    my_reshape, counter = gen_tvar(counter)
    symbols[n] = my_reshape

    src_var = symbols[n.args[0]]
    t2 = n.args[1]
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])  # type: ignore[union-attr]
    c1 = BinConstraintT(my_reshape, t2_type, op_eq)  # type: ignore[union-attr]
    c2 = CanReshape(src_var, t2_type)

    return [c1, c2], counter


@register_inference_rule(BatchNorm2d)
def batchnorm_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    # generate the new variable
    batchnorm_output, counter = gen_tvar(counter)
    symbols[n] = batchnorm_output
    batchnorm_input = symbols[n.args[0]]

    # dim vars
    d1, counter = gen_dvar(counter)
    d2, counter = gen_dvar(counter)
    d3, counter = gen_dvar(counter)
    d4, counter = gen_dvar(counter)

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    c1 = BinConstraintT(batchnorm_input, TensorType([d1, d2, d3, d4]), op_matching)
    c2 = BinConstraintT(batchnorm_input, batchnorm_output, op_eq)
    return [c1, c2, *nat_constraints], counter


@register_inference_rule(torch.nn.AdaptiveAvgPool2d)
def adaptive_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    avg_pool, counter = gen_tvar(counter)

    symbols[n] = avg_pool
    input_var = symbols[n.args[0]]

    # dim vars
    d1, counter = gen_dvar(counter)
    d2, counter = gen_dvar(counter)
    d3, counter = gen_dvar(counter)
    d4, counter = gen_dvar(counter)
    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)
    c2 = BinConstraintT(avg_pool, TensorType([d1, d2, module_instance.output_size[0], module_instance.output_size[1]]), op_eq)

    return [c1, c2, *nat_constraints], counter


@register_inference_rule(Conv2d)
def conv2d_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    my_conv, counter = gen_tvar(counter)
    symbols[n] = my_conv
    input_var = symbols[n.args[0]]

    # dim vars
    [d1, d2, d3, d4], counter = gen_tensor_dims(MAX_TENSOR_RANK, counter)

    # c1 = Matching(input_var, TensorType([d1, d2, d3, d4]))
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)

    # c2 = DConsistency(module_instance.in_channels, d2)
    c2 = BinConstraintD(module_instance.in_channels, d2, op_consistency)

    c3 = CalcConv(my_conv, input_var,
                  module_instance.out_channels,
                  module_instance.kernel_size,
                  module_instance.padding,
                  module_instance.stride,
                  module_instance.dilation, [d1, d2, d3, d4])

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    return [c1, c2, c3, *nat_constraints], counter


@register_inference_rule(torch.nn.MaxPool2d)
def maxpool_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    maxpool, counter = gen_tvar(counter)
    symbols[n] = maxpool
    input_var = symbols[n.args[0]]

    # dim vars
    [d1, d2, d3, d4], counter = gen_tensor_dims(MAX_TENSOR_RANK, counter)

    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)

    c2 = CalcMaxPool(maxpool, input_var, module_instance.kernel_size, module_instance.padding,
                     module_instance.stride, module_instance.dilation, [d1, d2, d3, d4])

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    return [c1, c2, *nat_constraints], counter


class ConstraintGenerator:
    def __init__(self, traced, graph=None):
        self.traced = traced  # traced or tracer.root
        self.constraints = []
        self.symbol_dict = {}
        self.graph = traced.graph if hasattr(traced, 'graph') else graph


    def generate_constraints(self, counter=0):
        """
        Iterate through every node and generate constraints
        Effect: self.constraints will be populated with the final constraints
        """
        graph = self.graph

        all_constraints = []

        # Annotate with Dyn if no type exists
        for n in graph.nodes:
            if n.type is None:
                n.type = Dyn

        for n in graph.nodes:
            (constraints, counter) = self.generate_constraints_node(n, counter)
            all_constraints += constraints
        return Conj(all_constraints), counter

    def generate_constraints_node(self, n: Node, counter):
        """
        Generate constraints the given node:
        Currently supported operations:
        - Reshape
        - Add
        - conv2d
        """

        if n.op == 'placeholder':
            x, counter = gen_tvar(counter)
            self.symbol_dict[n] = x
            c1 = BinConstraintT(n.type, x, op_precision)
            c2 = BinConstraintT(x, MAX_TENSOR_RANK, op_leq)
            return [c1, c2], counter

        elif n.op == 'call_function':
            if n.target == getattr:
                assert getattr in _INFERENCE_RULES
                return _INFERENCE_RULES[n.target](n, self.traced, self.symbol_dict, self.constraints)

            elif n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n, self.symbol_dict, self.constraints, counter)
            else:
                # print(n)
                raise RuntimeError(f'No inference rule registered for target {n.target}!')

        elif n.op == 'call_module':

            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](n,
                                                               module_instance,
                                                               self.symbol_dict,
                                                               self.constraints, counter)
            else:
                raise RuntimeError(f'No inference rule registered for class {type(module_instance)}!')

        elif n.op == 'call_method':
            if n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n, self.symbol_dict, self.constraints, counter)
            else:
                raise RuntimeError(f'No inference rule registered for target {n.target}!')

        # TODO: verify that no constraint should be generated here
        elif n.op == 'get_attr':
            return [], counter

        elif n.op == 'output':
            return [], counter

        else:
            raise NotImplementedError(f"Method {n.op} not yet implemented")
