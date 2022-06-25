import torch
import operator
from typing import Callable, Dict
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
    Disj, TGreatestUpperBound, CalcMaxPool, CalcConv, Conj, BinConstraintT, CanReshape, BinConstraintD
from torch.fx.experimental.migrate_gradual_types.operation import op_eq, op_matching, op_consistency, op_leq, op_precision
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar

from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
_INFERENCE_RULES: Dict[Target, Callable] = {}


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


@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def add_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)

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
    # c4 = BinConstraintT(my_add, 2, op_leq)
    # c5 = BinConstraintT(e1, 2, op_leq)
    # c6 = BinConstraintT(e2, 2, op_leq)
    # c7 = BinConstraintT(e11, 2, op_leq)
    # c8 = BinConstraintT(e22, 2, op_leq)

    # print([c1, c2, c3])

    # store constraints
    return [c1, c2, c3], counter


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

    const1, counter = generate_flatten_constraints(start_dim, end_dim, input, flattened, 1, counter)
    const2, counter = generate_flatten_constraints(start_dim, end_dim, input, flattened, 2, counter)
    const3, counter = generate_flatten_constraints(start_dim, end_dim, input, flattened, 3, counter)
    const4, counter = generate_flatten_constraints(start_dim, end_dim, input, flattened, 4, counter)
    return [Disj([both_dyn, const1, const2, const3, const4])], counter

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
    for i in range(1, 5):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)

        # Todo: add back natural number constraints
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
    # c3 = BinConstraintT(src_var, 4, op_leq)
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
    [d1, d2, d3, d4], counter = gen_tensor_dims(4, counter)

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
    [d1, d2, d3, d4], counter = gen_tensor_dims(4, counter)

    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)

    c2 = CalcMaxPool(maxpool, input_var, module_instance.kernel_size, module_instance.padding,
                     module_instance.stride, module_instance.dilation, [d1, d2, d3, d4])

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    return [c1, c2, *nat_constraints], counter


class ConstraintGenerator:
    def __init__(self, traced):
        self.traced = traced
        self.constraints = []
        self.symbol_dict = {}

    def generate_constraints(self, counter=0):
        """
        Iterate through every node and generate constraints
        Effect: self.constraints will be populated with the final constraints
        """
        graph = self.traced.graph

        all_constraints = []

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
            c2 = BinConstraintT(x, 4, op_leq)
            return [c1, c2], counter

        elif n.op == 'call_function':
            if n.target == getattr:
                assert getattr in _INFERENCE_RULES
                return _INFERENCE_RULES[n.target](n, self.traced, self.symbol_dict, self.constraints)
            elif n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n, self.symbol_dict, self.constraints, counter)
            else:
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

        elif n.op == 'output':
            return [], counter

        else:
            raise NotImplementedError(f"Method {n.op} not yet implemented")
