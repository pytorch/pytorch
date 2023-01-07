from torch._C import _TensorBase
import torch
import functools

from typing import Callable, Dict, cast

HANDLED_FUNCTIONS: Dict[Callable, torch.autograd.Function] = {}

def implements_per_sample_grads(torch_function):
    @functools.wraps(torch_function)
    def decorator(autograd_func):
        HANDLED_FUNCTIONS[torch_function] = autograd_func
        return autograd_func
    return decorator

# ExpandedWeight represents a weight (parameter) Tensor that has an expanded
# batch dimension. Operations on the ExpandedWeight Tensor act exactly like
# those without an expanded batch dimension but a call to .backward() populates
# the original (unexpanded) tensor with per-sample-gradients for in the grad_sample field
#
# ExpandedWeight has a fallback that always fails since we cannot know what the batch
# dimension of the input tensor is and therefore cannot know if this is a valid call
#
# This is a __torch_function__ object but it could have also been a Tensor Extension
# with a dispatch key.
#
# Needs to be a tensor subclass to allow reparamaterization
class ExpandedWeight(torch.Tensor):
    def __init__(self, orig_weight, batch_size, loss_reduction):
        self.batch_size = batch_size
        self.orig_weight = orig_weight
        self.loss_reduction = loss_reduction

    handled_functions = HANDLED_FUNCTIONS

    def __new__(cls, orig_weight, batch_size, loss_reduction):
        if not isinstance(orig_weight, torch.Tensor):
            raise RuntimeError(f"Can only make Expanded Weights of Tensors, got {type(orig_weight).__name__}")
        if not orig_weight.requires_grad:
            raise RuntimeError("Can only build ExpandedWeights objects of tensors that require_grad")
        ret = torch.Tensor._make_subclass(cast(_TensorBase, cls), orig_weight, True)
        return ret

    @classmethod
    def __torch_function__(cls, func, _, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in cls.handled_functions:
            return cls.handled_functions[func].apply(tuple(kwargs.keys()), func, *(args + tuple(kwargs.values())))
        # We cannot use a fallback here because we do not know the batch dimension for any regular tensor inputs,
        # i.e. torch.add(torch.Tensor, ExpandedWeight)
        raise RuntimeError(f"Expanded Weights encountered but cannot handle function {func.__name__}")

    @property
    def dtype(self):
        return self.orig_weight.dtype

    @property
    def shape(self):
        return self.orig_weight.shape
