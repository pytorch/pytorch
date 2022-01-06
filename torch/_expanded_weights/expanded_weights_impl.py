from torch._C import _TensorBase
import torch
import functools

from torch import Tensor
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
class ExpandedWeight(torch.Tensor):
    def __init__(self, orig_weight, batch_size):
        if not isinstance(orig_weight, torch.Tensor):
            raise RuntimeError(f"Can only make Expanded Weights of Tensors, got {type(orig_weight).__name__}")
        if not orig_weight.requires_grad:
            raise RuntimeError("Can only build ExpandedWeights objects of tensors that require_grad")
        self.batch_size = batch_size
        self.orig_weight = orig_weight
        self._expanded_weight_grad = None  # for mypy typing of a read-write property

    handled_functions = HANDLED_FUNCTIONS

    # needed for conv2d default kwargs
    conv_kwarg_options = ['stride', 'padding', 'dilation', 'groups']
    conv_kwarg_defaults = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1}

    def __new__(cls, orig_weight, _):
        ret = torch.Tensor._make_subclass(cast(_TensorBase, cls), orig_weight, orig_weight.requires_grad)
        return ret

    supported_methods = (Tensor.size, Tensor.numel, Tensor.is_contiguous, Tensor.stride, Tensor.requires_grad_,
                         Tensor.__format__, Tensor.__eq__, Tensor.__getitem__, Tensor.detach, Tensor.dim,
                         Tensor.ndimension, Tensor.register_hook, Tensor.__len__, Tensor.__index__, Tensor.__bool__,
                         Tensor.__int__, Tensor.__float__, Tensor.__long__, Tensor.__nonzero__, Tensor.__setstate__,
                         Tensor.__contains__, Tensor.__array__, Tensor.__array_wrap__, Tensor.eq, Tensor.__iter__,
                         Tensor.detach_)

    @classmethod
    def __torch_function__(cls, func, _, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func == torch.nn.functional.conv2d:
            remaining_kwargs = 7 - len(args)
            remaining_kwargs_options = cls.conv_kwarg_options[4 - remaining_kwargs:]
            ordered_kwargs = tuple(kwargs.get(key, cls.conv_kwarg_defaults[key]) for key in remaining_kwargs_options)
            return cls.handled_functions[torch.nn.functional.conv2d].apply(*(args + ordered_kwargs))
        if func in cls.handled_functions:
            return cls.handled_functions[func].apply(*(args + tuple(kwargs.values())))
        if func in cls.supported_methods:
            return func(args[0].orig_weight, *args[1:])
        # We cannot use a fallback here because we do not know the batch dimension for any regular tensor inputs,
        # i.e. torch.add(torch.Tensor, ExpandedWeight)
        raise RuntimeError(f"Expanded Weights encountered but cannot handle function {func.__name__}")


    @property
    def shape(self):
        return self.orig_weight.shape

    @property
    def dtype(self):
        return self.orig_weight.dtype

    @property
    def requires_grad(self):
        return self.orig_weight.requires_grad

    @property
    def grad_fn(self):
        return None

    @property
    def is_sparse(self):
        return self.orig_weight.is_sparse

    @property
    def is_quantized(self):
        return self.orig_weight.is_quantized

    @property
    def device(self):
        return self.orig_weight.device

    @property
    def is_cuda(self):
        return self.orig_weight.is_cuda

    @property
    def layout(self):
        return self.orig_weight.layout

    @property
    def ndim(self):
        return self.orig_weight.ndim

    @property
    def grad(self):
        return self._expanded_weight_grad

    @grad.setter
    def grad(self, value):
        if value is None:
            return
        else:
            raise RuntimeError("ExpandedWeights should never have a grad value set on it.")

    def to(self, device):
        if device == self.orig_weight.device:
            return self
        return ExpandedWeight(self.orig_weight.to(device), self.batch_size)

    def __deepcopy__(self, memo):
        return ExpandedWeight(self.orig_weight.__deepcopy__(memo), self.batch_size)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "ExpandedWeight for:\n" + self.orig_weight.__repr__() + f" with batch size {self.batch_size}"

@implements_per_sample_grads(torch.allclose)
class AllCloseHelper:
    # This is needed for equality checking, but there's no per sample grad computation
    @staticmethod
    def apply(a, b, rtol, atol, equal_nan):
        if isinstance(a, ExpandedWeight):
            a = a.orig_weight
        if isinstance(b, ExpandedWeight):
            b = b.orig_weight
        return torch.allclose(a, b, rtol, atol, equal_nan)
