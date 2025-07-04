# mypy: allow-untyped-defs
import functools
from contextlib import contextmanager
from typing import Callable

import torch
from torch._decomp import decomposition_table
from torch.utils._pytree import tree_map_only


HANDLED_FUNCTIONS: dict[Callable, torch.autograd.Function] = {}

aten = torch._ops.ops.aten
# __torch_function__ runs before the pydispatcher so we need to manually use the same
# decompositions indexed by their torch equivalent
expanded_weights_rnn_decomps = {
    # func: (input_decomp, data_decomp)
    torch.rnn_relu: (
        decomposition_table[aten.rnn_relu.input],
        decomposition_table[aten.rnn_relu.data],
    ),
    torch.rnn_tanh: (
        decomposition_table[aten.rnn_tanh.input],
        decomposition_table[aten.rnn_tanh.data],
    ),
    torch.lstm: (
        decomposition_table[aten.lstm.input],
        decomposition_table[aten.lstm.data],
    ),
    torch.gru: (
        decomposition_table[aten.gru.input],
        decomposition_table[aten.gru.data],
    ),
}


# all of the RNN decomps run linear with the batch dimension second, even if batch_first was set
@contextmanager
def batch_second(args, kwargs):
    def set_batch_second(ew):
        ew.set_batch_first(False)

    def reset_batch_first(ew):
        ew.set_batch_first(True)

    tree_map_only(ExpandedWeight, set_batch_second, args)
    tree_map_only(ExpandedWeight, set_batch_second, kwargs)
    try:
        yield
    finally:
        tree_map_only(ExpandedWeight, reset_batch_first, args)
        tree_map_only(ExpandedWeight, reset_batch_first, kwargs)


# to support packed sequences, we need to allow for smaller batches. Expanded weights represents the largest batch
@contextmanager
def allow_smaller_batches(args, kwargs):
    def allow(ew):
        ew.set_allow_smaller_batches(True)

    def reset(ew):
        ew.set_allow_smaller_batches(False)

    tree_map_only(ExpandedWeight, allow, args)
    tree_map_only(ExpandedWeight, allow, kwargs)
    try:
        yield
    finally:
        tree_map_only(ExpandedWeight, reset, args)
        tree_map_only(ExpandedWeight, reset, kwargs)


@contextmanager
def setup_rnn(use_input_variant, args, kwargs):
    with (
        batch_second(args, kwargs)
        if use_input_variant
        else allow_smaller_batches(args, kwargs)
    ):
        yield


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
        self.allow_smaller_batches = False
        self.orig_weight = orig_weight
        self.loss_reduction = loss_reduction

    handled_functions = HANDLED_FUNCTIONS

    def __new__(cls, orig_weight, batch_size, loss_reduction):
        if not isinstance(orig_weight, torch.Tensor):
            raise RuntimeError(
                f"Can only make Expanded Weights of Tensors, got {type(orig_weight).__name__}"
            )
        if not orig_weight.requires_grad:
            raise RuntimeError(
                "Can only build ExpandedWeights objects of tensors that require_grad"
            )
        ret = torch.Tensor._make_subclass(cls, orig_weight, True)
        return ret

    @classmethod
    def __torch_function__(cls, func, _, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in expanded_weights_rnn_decomps:
            # in aten, choosing the input or data variants is done by parsing logic. This mimics some of that
            decomp_opts = expanded_weights_rnn_decomps[func]
            use_input_variant = isinstance(
                args[2], list
            )  # data variant uses a list here
            decomp = decomp_opts[0] if use_input_variant else decomp_opts[1]

            if decomp is not None:
                with setup_rnn(use_input_variant, args, kwargs):
                    return decomp(*args, **kwargs)
        if func == torch._cudnn_rnn_flatten_weight:
            # since we aren't using the fused cuda kernels for RNNs, don't do this
            return
        if func in cls.handled_functions:
            return cls.handled_functions[func].apply(
                tuple(kwargs.keys()), func, *(args + tuple(kwargs.values()))
            )
        # We cannot use a fallback here because we do not know the batch dimension for any regular tensor inputs,
        # i.e. torch.add(torch.Tensor, ExpandedWeight)
        raise RuntimeError(
            f"Expanded Weights encountered but cannot handle function {func.__name__}"
        )

    @property
    def dtype(self):  # type: ignore[override]
        return self.orig_weight.dtype

    @property
    def data(self):  # type: ignore[override]
        return self.orig_weight.data

    @property
    def shape(self):  # type: ignore[override]
        return self.orig_weight.shape

    @property
    def device(self):  # type: ignore[override]
        return self.orig_weight.device

    @property
    def is_cuda(self):  # type: ignore[override]
        return self.orig_weight.is_cuda

    def data_ptr(self):
        return self.orig_weight.data_ptr()

    def get_device(self):
        return self.orig_weight.get_device()

    def set_allow_smaller_batches(self, is_allow_smaller_batches):
        self.allow_smaller_batches = is_allow_smaller_batches

    def set_batch_first(self, is_batch_first=True):
        self.batch_first = is_batch_first
