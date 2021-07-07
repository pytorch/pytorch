from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d


_INFERENCE_RULES: Dict[Target, Callable] = {}


def expand_to_tensor_dim(t, n):
    """
    Expand a type to the desired tensor dimension if possible
    Raise an error otherwise.
    - t is the given type
    - n is a number to expand to
    """
    if t == Dyn:
        dims = [Dyn] * n
        return TensorType(tuple(dims))
    elif isinstance(t, TensorType):
        if len(t.__args__) != n:
            raise TypeError(f'Cannot extend tensor dimension. Tensor {t} has rank {len(t.__args__)}. It should have rank {n}')
        return t
    else:
        raise TypeError(f'Cannot match the type {t}')


def broadcast_types(t1, t2):
    if t1 == Dyn or t2 == Dyn:
        return t1, t2

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        s1 = len(t1.__args__)
        s2 = len(t2.__args__)

        new_t1 = list(t1.__args__)
        new_t2 = list(t2.__args__)

        if abs(s1 - s2) > 1 or s1 == 0 or s2 == 0:
            raise TypeError(f'Cannot broadcast the tensors {t1} and {t2}')

        if s1 > s2:
            new_t2.insert(0, t1.__args__[0])

        elif s2 > s1:
            new_t1.insert(0, t2.__args__[0])

        for i, (x, y) in enumerate(zip(new_t1, new_t2)):
            if x == 1:
                new_t1[i] = y
            elif y == 1:
                new_t2[i] = x
            else:
                continue

        if tuple(new_t1) != t1.__args__ and tuple(new_t2) != t2.__args__:
            raise TypeError('In-place operations cannot not change shape')

        return TensorType(tuple(new_t1)), TensorType(tuple(new_t2))
    else:
        raise TypeError(f'Cannot broadcast types {t1} and {t2}')

def register_inference_rule(call_target):
    def register(fn):
        if call_target in _INFERENCE_RULES:
            raise RuntimeError('Inference rule already registered for {call_target}!')
        _INFERENCE_RULES[call_target] = fn
        return fn
    return register


@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def add_inference_rule(n: Node):
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)
    t1 = n.args[0].type
    t2 = n.args[1].type

    # handle scalar addition
    if t1 == int and isinstance(t2, TensorType):
        n.type = t2
        return n.type

    elif t2 == int and isinstance(t1, TensorType):
        n.type = t1
        return n.type

    (new_t1, new_t2) = broadcast_types(t1, t2)
    n.args[0].type = new_t1
    n.args[1].type = new_t2

    if is_consistent(new_t1, new_t2):
        # we return the more precise type
        if is_more_precise(new_t1, new_t2):
            n.type = new_t2
        else:
            n.type = new_t1
        return n.type
    else:
        raise TypeError(f'Cannot add arguments {n.args[0]} ({ n.args[0].type}) and {n.args[1]} ({ n.args[1].type}) in node {n}.'
                        f' Types should match ')


@register_inference_rule(torch.transpose)
def transpose_inference_rule(n: Node):
    if n.target == torch.transpose:
        assert isinstance(n.args[0], Node)
        t = n.args[0].type

        assert isinstance(n.args[1], int)
        assert isinstance(n.args[2], int)
        dim1, dim2 = n.args[1], n.args[2]

        if t == Dyn:
            n.type = Dyn
            return n.type

        elif isinstance(t, TensorType):

            if 0 <= dim1 < len(t.__args__) and 0 <= dim2 < len(t.__args__):
                new_type = list(t.__args__)
                new_type[dim1], new_type[dim2] = new_type[dim2], new_type[dim1]
                final = TensorType(new_type)
                n.type = final
                return n.type
            else:
                raise TypeError(f'Cannot transpose {dim1} and {dim2} in type {t} for node {n}')
        else:
            raise TypeError(f'Cannot transpose {dim1} and {dim2} in type {t} for node {n}')


@register_inference_rule(torch.reshape)
def reshape_inference_rule(n: Node):
    assert isinstance(n.args[0], Node)
    t1 = n.args[0].type

    assert isinstance(n.args[1], list)
    t2 = n.args[1]
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])

    # if we do not know the original tensor dimension,
    # we return the required dimension
    if t1 == Dyn:
        n.type = t2_type
        return t2_type

    # if any of the dimensions are unknown,
    # we check for divisibility
    elif isinstance(t1, TensorType) and Dyn in t1.__args__ or -1 in t2:
        assert isinstance(t1, TensorType)
        a = [e if e != Dyn else 1 for e in t1.__args__]
        p1 = reduce(lambda x, y: x * y, a)
        p2 = reduce(lambda x, y: x * y, t2)
        if p1 % p2 == 0 or p2 % p1 == 0:
            n.type = t2_type
            return t2_type
        else:
            raise TypeError(f'Cannot reshape in node {n} from {t1} to {t2_type}')

    # if all dimensions are known we check the products
    elif isinstance(t1, TensorType):
        p1 = reduce(lambda x, y: x * y, t1.__args__)
        p2 = reduce(lambda x, y: x * y, t2)
        if p1 == p2:
            n.type = t2_type
            return t2_type
        else:
            raise TypeError(f'Cannot reshape in node {n} from {t1} to {t2_type}')

    else:
        raise TypeError(f'Cannot reshape in node {n} from {t1} to {t2_type}')

@register_inference_rule(BatchNorm2d)
def bn2d_inference_rule(n: Node, module_instance):
    """
    Given a BatchNorm2D instance and a node check the following conditions:
    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, x_3, x_4)
    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')
    - t is consistent with t'
    - x_2 is consistent with the module's num_features
    - x_2' is consistent with the module's num_features
    output type: the more precise type of t and t'
    """
    assert isinstance(n.args[0], Node)
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    arg_type = n.args[0].type
    n.type = expand_to_tensor_dim(n.type, 4)

    # we check the conditions on the incoming argument
    # and any existing annotation
    # we also check for consistency between both annotations
    if is_consistent(arg_type.__args__[1], module_instance.num_features) and \
            is_consistent(n.type.__args__[1], module_instance.num_features) and \
            is_consistent(arg_type, n.type):

        # we choose the more precise type
        # to be the node type
        # so if an incoming argument has more type information
        # we set this node's type to be the argument type
        if is_more_precise(arg_type, n.type):
            n.type = arg_type
        return n.type
    else:
        raise TypeError(f'Cannot apply {module_instance} with input type {arg_type} and existing type {n.type} on {n}')

def calculate(d_in, module_instance, index):
    """
    For calculating h_in and w_out.
    """
    if d_in == Dyn:
        return Dyn

    elif isinstance(d_in, int):
        n = d_in + 2 * module_instance.padding[index] - \
            module_instance.dilation[index] * \
            (module_instance.kernel_size[index] - 1) - 1

        return (n // module_instance.stride[0]) + 1
    else:
        raise TypeError(f'{d_in} in {module_instance} must be a number or Dyn')


def get_greatest_upper_bound(type1, type2):
    """
    Get the most precise type that's consistent with the given types
    """
    if type1 == Dyn:
        return type2
    elif type2 == Dyn:
        return type1
    elif isinstance(type1, TensorType) and isinstance(type2, TensorType):
        assert len(type1.__args__) == len(type2.__args__)
        gub = [t1 if is_more_precise(t1, t2) else t2 for (t1, t2) in zip(type1.__args__, type2.__args__)]
        return TensorType(tuple(gub))
    else:
        raise NotImplementedError(f'Greatest upper bound not yet implemented for these types {type1}, {type2}')

@register_inference_rule(Conv2d)
def conv2d_inference_rule(n: Node, module_instance):
    """
    Given a Conv2D instance and a node check the following conditions:
    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, H, W)
    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')
    - x_2 is consistent with the module's in_channels
    - let o = (x_1, out_channels, H_out, W_out)
    then the outout is the greatest upper bound of o and the existing node type t'.
    """
    assert isinstance(n.args[0], Node)
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    arg_type = n.args[0].type
    curr_node_type = expand_to_tensor_dim(n.type, 4)

    if is_consistent(arg_type.__args__[1], module_instance.in_channels):
        w_in = arg_type.__args__[3]
        h_in = arg_type.__args__[2]
        h_out = calculate(h_in, module_instance, 0)
        w_out = calculate(w_in, module_instance, 1)
        new_type = TensorType((arg_type.__args__[0], module_instance.out_channels, h_out, w_out))

        if not is_consistent(new_type, curr_node_type):
            raise TypeError(f'Inconsistent types {new_type} and {curr_node_type}')
        else:
            gub = get_greatest_upper_bound(new_type, curr_node_type)
            n.type = gub
    else:
        raise TypeError(f'Cannot apply {module_instance} with input type { arg_type} and existing type {n.type} on {n}')

class GraphTypeChecker:
    def __init__(self, env, traced):
        self.env = env
        self.traced = traced

    def type_check(self):
        """
        A gradual type checker for graphs
        Effect: every node's field type will be
        populated with a type after type-checking is done
        """
        graph = self.traced.graph

        # type check every node with gradual type rules
        # if any node does not type check return false
        for n in graph.nodes:
            self.type_check_node(n)
        return True

    def type_check_node(self, n: Node):
        """
        Type check a given fx node.
        Current operations:
        - Reshape
        - Transpose
        - Add
        """
        if n.type is None:
            n.type = Dyn

        if n.op == 'placeholder':
            return n.type

        if n.op == 'call_function':
            if n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n)
            else:
                raise RuntimeError(f'No inference rule registered for target {n.target}!')

        if n.op == 'call_module':
            module_instance = getattr(self.traced, str(n.target))
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](n, module_instance)
            else:
                raise RuntimeError(f'No inference rule registered for class {type(module_instance)}!')

        if n.op == 'output':
            assert isinstance(n.args[0], Node)
            n.type = n.args[0].type
            return n.type

        else:
            raise NotImplementedError("Method not yet implemented")
