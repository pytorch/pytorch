import torch
import functools

from typing import Callable, Dict

HANDLED_FUNCTIONS: Dict[Callable, torch.autograd.Function] = {}

def implements_per_sample_grads(torch_function):
    @functools.wraps(torch_function)
    def decorator(autograd_func):
        HANDLED_FUNCTIONS[torch_function] = autograd_func
        return autograd_func
    return decorator

# ExpandedWeight represents a weight (parameter) Tensor that has an expanded
# batch dimension. Operations on the ExpandedWeight Tensor take advantage of
# how the batch dimension is expanded by de-expanding the weight before
# computation. A subsequent call to .backward() computes gradients for
# ExpandedWeight. Those gradients are equivalent to per-sample-grads for the
# unexpanded weight Tensors.
#
# ExpandedWeight has a fallback that does the forward + backward computation.
# The backward computation is not optimized: it runs torch.autograd.grad in
# a loop. To optimize the backward computation further, we must register
# overrides for specific operators.
#
# This is a __torch_function__ object but it could have also been a Tensor Extension
# with a dispatch key.
class ExpandedWeight(torch.Tensor):
    handled_functions = HANDLED_FUNCTIONS

    # needed for conv2d default kwargs
    conv_kwarg_options = ['stride', 'padding', 'dilation', 'groups']
    conv_kwarg_defaults = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1}

    def __new__(cls, orig_weight, batch_size):
        ret = torch.Tensor._make_subclass(cls, orig_weight.detach(), orig_weight.requires_grad)
        if not isinstance(orig_weight, torch.Tensor):
            raise RuntimeError(f"Can only make Expanded Weights of Tensors, got {type(orig_weight).__name__}")
        ret.batch_size = batch_size
        ret.orig_weight = orig_weight
        return ret

    @classmethod
    def __torch_function__(cls, func, _, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in cls.handled_functions:
            # We cannot use a fallback here because we do not know the batch dimension for any regular tensor inputs,
            # i.e. torch.add(torch.Tensor, ExpandedWeight)
            raise RuntimeError(f"Expanded Weights encountered but cannot handle function {func.__name__}")
        if func == torch.nn.functional.conv2d:
            remaining_kwargs = 7 - len(args)
            remaining_kwargs_options = cls.conv_kwarg_options[4 - remaining_kwargs:]
            kwargs = {key: cls.conv_kwarg_defaults[key] for key in remaining_kwargs_options} | kwargs
        return cls.handled_functions[func].apply(*(args + tuple(kwargs.values())))

    @property
    def shape(self):
        return self.orig_weight.shape

    def size(self):
        return self.orig_weight.size()

    @property
    def dtype(self):
        return self.orig_weight.dtype

    @property
    def grad(self):
        return None

    @grad.setter
    def grad(self, value):
        if value is None:
            return
        else:
            raise RuntimeError("ExpandedWeights should never have a grad value set on it.")

    @property
    def requires_grad(self):
        return self.orig_weight.requires_grad

    @property
    def grad_fn(self):
        return None

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "ExpandedWeight for:\n" + self.orig_weight.__repr__() + f" with batch size {self.batch_size}"
