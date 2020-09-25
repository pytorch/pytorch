import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Node, Proxy, symbolic_trace, Graph, GraphModule
from typing import List, Dict

# Gradient formulas. 
# FIXME: There are a bunch of interesting problems with defining grad formulas.
# 1. What do we do with PyTorch ops that are composite w.r.t. autograd?
#    In FX we may need to write a symbolic gradient for *every* operator,
#    but that could quickly become intractable.
# 2. Some ops save intermediate values that are created inside the op for
#    backwards. How do we do that in FX?
#    One example is dropout. The dropout mask created inside dropout MUST
#    be saved for backwards so it can be re-applied.
# 
# One potential workaround for both of the above is implement some decomposing
# logic for the autodiff pass to decompose operations like dropout into
# something more workable.
def sum_backward(grad, tensor):
    return grad.expand_as(tensor)

def mul_backward(grad, tensor, other):
    return grad * other, grad * tensor

def add_backward(grad, tensor, other):
    return grad, grad

def sub_backward(grad, tensor, other):
    return grad, -grad

def relu_backward(grad, tensor):
    return grad * (tensor >= 0)

# TODO: default arg handling
def flatten_backward(grad, tensor, start_dim):
    return grad.unflatten(-1, tensor.shape[start_dim:])

# Register gradient formulas. Feels really hacky to me.
vjp_map = {
    torch.sum: sum_backward,
    torch.mul: mul_backward,
    torch.sub: sub_backward,
    torch.add: add_backward,
    torch.flatten: flatten_backward,
    F.relu: relu_backward,
}

# Since modules are first-class citizens, they need their own gradient formulas
# This is not very ideal because:
# 1. many of our modules are composite (conv1d, conv2d, conv3d)
# 2. We may rely on intermediate values (see Dropout) to compute gradients.
# We can definitely work around (2), but (1) seems important.
def linear_backward(grad, tensor, weight, bias):
    # TODO: we're assuming that linear operates on 2D input here.
    return torch.matmul(grad, weight), torch.matmul(grad.t(), tensor), grad.sum(0)

def linear_backward_no_bias(grad, tensor, weight):
    return torch.matmul(grad, weight), torch.matmul(grad.t(), tensor)

def conv2d_backward(grad, tensor, weight, bias):
    # grad_weight computation is tricky.
    tensor_unfolded = (
        F.unfold(tensor, weight.shape[2:])
        .unflatten(1, [tensor.shape[1], weight.shape[2] * weight.shape[3]])
    )
    grad_weight = (
        torch.einsum('nckp,ndp->dck', tensor_unfolded, grad.flatten(-2, -1))
        .unflatten(-1, weight.shape[2:])
    )

    return (
        F.conv_transpose2d(grad, weight),
        grad_weight,
        grad.sum([0, 2, 3]),
    )

# Register gradient formulae for modules here.
def module_backward_rule(module):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            return linear_backward
        return linear_backward_no_bias
    if isinstance(module, nn.Conv2d):
        return conv2d_backward
    assert False

# Gradient transformation logic
def get_params(module_instance: str, module: nn.Module):
    return [f'{module_instance}.{name}' for name, _ in module.named_parameters()]

GradDict = Dict[str, Proxy]

def update_grad(grad_dict, key: str, value: Proxy):
    if key not in grad_dict:
        grad_dict[key] = value
    else:
        grad_dict[key] = torch.add(grad_dict[key], value)

def function_backward(node: Node, grad_dict: GradDict):
    grad_fn = vjp_map[node.target]
    args = [Proxy(arg) if isinstance(arg, Node) else arg
            for arg in node.args]

    # TODO: does fx support nodes with multiple outputs?
    grad_output = grad_dict[node.name]
    assert grad_output is not None

    grad_inputs = grad_fn(grad_output, *args)
    if not isinstance(grad_inputs, tuple):
        grad_inputs = (grad_inputs,)

    for grad_inp, argname in zip(grad_inputs, map(lambda arg: arg.name, node.args)):
        update_grad(grad_dict, argname, grad_inp)

def module_backward(
        owning_module: nn.Module,
        node: Node,
        grad_dict: GradDict,
        leaf_attrs):
    # For each output, pull the grads. NB: assumes every module has a single output.
    grad: Proxy = grad_dict[node.name]

    # Pull the module parameters as proxies
    actual_module = getattr(owning_module, node.target)
    param_names = get_params(node.target, actual_module)
    leaf_attrs.update(param_names)
    params: List[Proxy] = [Proxy(node.graph.get_attr(param_name))
                           for param_name in param_names]

    # Perform the backward computation
    module_args = list(map(Proxy, node.args))
    result = module_backward_rule(actual_module)(grad, *module_args, *params)

    names = [arg.node.name for arg in module_args] + list(param_names)
    assert len(names) == len(result)
    for grad, name in zip(result, names):
        update_grad(grad_dict, name, grad)

def grad(module, only_compute_param_grads=True):
    gm = symbolic_trace(module)
    grad_graph = Graph()
    grad_graph.graph_copy(gm.graph)
    grad_dict: GradDict = {}
    leaf_attrs = set({})

    # TODO: Is there a way to create a Tensor in FX?
    ones = grad_graph.create_node('get_attr', 'ones')
    grad_dict[gm.graph.result.name] = Proxy(ones)

    # In reverse topological order, update `grad_dict` and `leaf_attrs`.
    nodes = grad_graph.nodes
    for node in reversed(nodes):
        if node.op == 'call_function':
            function_backward(node, grad_dict)
        elif node.op == 'call_module':
            module_backward(module, node, grad_dict, leaf_attrs)
        elif node.op == 'call_method':
            raise RuntimeError("NYI")
        else:
            continue

    # Set the outputs to be either:
    # - [all parameters]
    # - [all inputs] + [all parameters]
    # TODO: This doesn't actually catch all the parameers.
    output = {leaf: grad_dict[leaf].node for leaf in leaf_attrs}
    if not only_compute_param_grads:
        for inp in [node.name for node in gm.graph.nodes if node.op == 'placeholder']:
            output[inp] = grad_dict[inp]
    grad_graph.output(output)
    return GraphModule(module, grad_graph)
