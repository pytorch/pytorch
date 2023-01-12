from contextlib import contextmanager

from torch._C import _TensorBase
import torch
import functools
from torch._decomp import decomposition_table

from typing import Callable, Dict, cast

from torch.utils._pytree import tree_map_only

HANDLED_FUNCTIONS: Dict[Callable, torch.autograd.Function] = {}

# __torch_function__ runs before the pydispatcher so we need to use the same
# decompositions indexed by their torch equivalent
expanded_weights_rnn_decomps = {
    # func: (input_decomp, data_decomp)
    torch.rnn_relu: (decomposition_table[torch._ops.ops.aten.rnn_relu.input], None),
    torch.rnn_tanh: (decomposition_table[torch._ops.ops.aten.rnn_tanh.input], None),
    torch.lstm: (decomposition_table[torch._ops.ops.aten.lstm.input], None),
}

@contextmanager
def batch_second(args, kwargs):
    tree_map_only(ExpandedWeight, functools.partial(ExpandedWeight.set_batch_first, is_batch_first=False), args)
    tree_map_only(ExpandedWeight, functools.partial(ExpandedWeight.set_batch_first, is_batch_first=False), kwargs)
    try:
        yield
    finally:
        tree_map_only(ExpandedWeight, functools.partial(ExpandedWeight.set_batch_first, is_batch_first=True), args)
        tree_map_only(ExpandedWeight, functools.partial(ExpandedWeight.set_batch_first, is_batch_first=True), kwargs)


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
        self.batch_first = True
        self.orig_weight = orig_weight
        self.loss_reduction = loss_reduction

    handled_functions = HANDLED_FUNCTIONS
    supported_unary = (torch.t, torch.Tensor.t)

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
        if func in expanded_weights_rnn_decomps:
            # in aten, choosing the input or data variants is done by parsing logic. This mimics some of that
            decomp_opts = expanded_weights_rnn_decomps[func]
            use_input_variant = not isinstance(args[1], list)  # data variant uses a list here
            decomp = decomp_opts[0] if use_input_variant else decomp_opts[1]

            if decomp is not None:
                with batch_second(args, kwargs):
                    return decomp(*args, **kwargs)
        if func == torch._cudnn_rnn_flatten_weight:
            # this updates in place, so just pass the underyling tensors through
            args = tree_map_only(ExpandedWeight, lambda ew: ew.orig_weight, args)
            kwargs = tree_map_only(ExpandedWeight, lambda ew: ew.orig_weight, kwargs)
            return torch._cudnn_rnn_flatten_weight(*args, **kwargs)
        if func in cls.handled_functions:
            return cls.handled_functions[func].apply(tuple(kwargs.keys()), func, *(args + tuple(kwargs.values())))
        # We cannot use a fallback here because we do not know the batch dimension for any regular tensor inputs,
        # i.e. torch.add(torch.Tensor, ExpandedWeight)
        raise RuntimeError(f"Expanded Weights encountered but cannot handle function {func.__name__}")

    @property
    def dtype(self):
        return self.orig_weight.dtype

    @property
    def data(self):
        return self.orig_weight.data

    @property
    def shape(self):
        return self.orig_weight.shape

    @property
    def device(self):
        return self.orig_weight.device

    @property
    def is_cuda(self):
        return self.orig_weight.is_cuda

    def data_ptr(self):
        return self.orig_weight.data_ptr()

    def get_device(self):
        return self.orig_weight.get_device()

    def set_batch_first(self, is_batch_first=True):
        self.batch_first = is_batch_first
