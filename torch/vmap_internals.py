import torch
import functools
from torch import Tensor
import warnings

REQUIRE_SAME_MAP_SIZE = (
    'vmap: Expected all tensors to have the same size in the mapped dimension, '
    'got sizes {sizes} for the mapped dimension'
)

ELEMENT_MUST_BE_TENSOR = (
    'vmap({fn}, ...): `{fn}` must return a Tensor or a flat sequence of tensors, got '
    'type {out} for return {idx}.'
)

MUST_BE_TENSOR_LIST = (
    'vmap({fn}, ...): `{fn}` must return a Tensor or a flat sequence of tensors, got '
    'type {out} as the return.'
)

NO_INPUTS = (
    'vmap({fn})(<inputs>): got no inputs. Maybe you forgot '
    'to add inputs, or you are trying to vmap over a '
    'function with no inputs. The latter is unsupported.'
)

# Checks that all args have the same batch dim size.
def _validate_and_get_batch_size(args):
    batch_sizes = [arg.size(0) for arg in args]
    if batch_sizes and any([size != batch_sizes[0] for size in batch_sizes]):
        raise ValueError(REQUIRE_SAME_MAP_SIZE.format(sizes=batch_sizes))
    return batch_sizes[0]


def _validate_inputs_and_get_batch_size(args, fn_name):
    if len(args) == 0:
        raise ValueError(NO_INPUTS.format(fn=fn_name))
    return _validate_and_get_batch_size(args)

# Undos the batching (and any batch dimensions) associated with the `vmap_level`.
def _unwrap_batched(batched_outputs, vmap_level, batch_size):
    if isinstance(batched_outputs, Tensor):
        return torch._remove_batch_dim(batched_outputs, vmap_level, batch_size, 0)  # type: ignore
    return tuple(torch._remove_batch_dim(out, vmap_level, batch_size, 0)  # type: ignore
                 for out in batched_outputs)

# Checks that `outputs` is either a Tensor or a sequence of Tensors.
def _validate_outputs(outputs, fn_name):
    if isinstance(outputs, Tensor):
        return
    if not hasattr(outputs, '__iter__'):
        raise ValueError(MUST_BE_TENSOR_LIST.format(fn=fn_name, out=type(outputs)))
    for idx, output in enumerate(outputs):
        if isinstance(output, Tensor):
            continue
        raise ValueError(ELEMENT_MUST_BE_TENSOR.format(fn=fn_name, out=type(output), idx=idx))

# This is the global tracker for how many nested vmaps we are currently inside.
VMAP_LEVEL = 0

# vmap(func)(inputs) wraps all Tensor inputs to be batched in BatchedTensors,
# sends those into func, and then unwraps the output BatchedTensors. Operations
# on BatchedTensors perform the batched operations that the user is asking for.
def vmap(func, in_dims=0, out_dims=0):
    warnings.warn(
        'torch.vmap is an experimental prototype that is subject to '
        'change and/or deletion. Please use at your own risk.')

    if in_dims != 0:
        raise NotImplementedError('NYI: vmap with `in_dims` other than 0')
    if out_dims != 0:
        raise NotImplementedError('NYI: vmap with `out_dims` other than 0')

    @functools.wraps(func)
    def wrapped(*args):
        if any(not isinstance(arg, Tensor) for arg in args):
            raise NotImplementedError('NYI: vmap with non-tensor inputs')

        batch_size = _validate_inputs_and_get_batch_size(args, func.__name__)
        global VMAP_LEVEL
        VMAP_LEVEL += 1
        try:
            batched_inputs = [torch._add_batch_dim(arg, 0, VMAP_LEVEL) for arg in args]  # type: ignore
            batched_outputs = func(*batched_inputs)
            _validate_outputs(batched_outputs, func.__name__)
            return _unwrap_batched(batched_outputs, VMAP_LEVEL, batch_size)
        finally:
            VMAP_LEVEL -= 1
    return wrapped
