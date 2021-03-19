import torch
from torch import vmap
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch.make_functional import make_functional
import gc

# x = torch.ones(2, 3)
# y = torch.ones(2, 3)
# # result = vmap(torch.add)(x, y)
# result = vmap(vmap(torch.add))(x, y)

# assert torch.allclose(result, x + y)

def _create_differentiable(tensor_or_tuple_of_tensors):
    if isinstance(tensor_or_tuple_of_tensors, torch.Tensor):
        tensor = tensor_or_tuple_of_tensors
        # if tensor.requires_grad:
        #     return tensor
        assert not tensor.requires_grad
        # NB: view is needed because autograd is silly.
        # autograd saved the variable before executing the op, which matters...
        return tensor.view_as(tensor).requires_grad_()
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return tuple(map(_create_differentiable, tensor_or_tuple_of_tensors))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return tuple(map(_create_differentiable, tensor_or_tuple_of_tensors))
    assert False

def _undo_create_differentiable(tensor_or_tuple_of_tensors):
    if isinstance(tensor_or_tuple_of_tensors, torch.Tensor):
        tensor = tensor_or_tuple_of_tensors
        return tensor.requires_grad_(False)
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return tuple(map(_undo_create_differentiable, tensor_or_tuple_of_tensors))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return tuple(map(_undo_create_differentiable, tensor_or_tuple_of_tensors))
    assert False

def _any_differentiable(tensor_or_tuple_of_tensors):
    if isinstance(tensor_or_tuple_of_tensors, torch.Tensor):
        tensor = tensor_or_tuple_of_tensors
        return tensor.requires_grad
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return any(tuple(map(_any_differentiable, tensor_or_tuple_of_tensors)))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return any(tuple(map(_any_differentiable, tensor_or_tuple_of_tensors)))
    return False


def grad_with_value(f, diff_argnums=(0,), has_aux=False):
    def wrapper(*args):
        torch._C._grad_increment_nesting()
        output, aux = None, None
        try:
            args = [_create_differentiable(arg) if i in diff_argnums else arg
                    for i, arg in enumerate(args)]
            output = f(*args)
            if has_aux:
                output, aux = output
            assert output.dim() == 0
            diff_args = [args[i] for i in diff_argnums]
            single_diff_arg = isinstance(diff_args[0], torch.Tensor) and len(diff_args) == 1
            # TODO: quick hack...
            if len(diff_args) == 1 and isinstance(diff_args[0], tuple):
                diff_args = diff_args[0]
            # NB: need create_graph so that backward pass isn't run in no_grad mode
            # import torchviz; import graphviz
            # graph = torchviz.make_dot(output)
            # graph.save("inner.dot")
            grad_input = torch.autograd.grad(
                output, diff_args, create_graph=True)
            if single_diff_arg:
                grad_input = grad_input[0]
        finally:
            _undo_create_differentiable(args)
            torch._C._grad_decrement_nesting()
        if has_aux:
            return grad_input, output, aux
        return grad_input, output
    return wrapper

def grad(f, diff_argnums=(0,), has_aux=False):
    def wrapper(*args):
        results = grad_with_value(f, diff_argnums, has_aux=has_aux)(*args)
        if has_aux:
            return results[0], results[2]
        return results[0]
    return wrapper

