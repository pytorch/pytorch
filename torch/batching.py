import torch
import functools
from torch import Tensor

VMAP_LEVEL = 0

def _make_batched(args, dims, level):
    batch_size = None
    batch_sizes = [arg.size(dim)
                   for arg, dim in zip(args, dims)
                   if isinstance(arg, Tensor) and dim is not None]
    if batch_sizes:
        batch_size = batch_sizes[0]
        assert all([size == batch_size for size in batch_sizes])
    return [torch._make_batched(arg, dim, level)
            if isinstance(arg, Tensor) else arg
            for arg, dim in zip(args, dims)], batch_size


def _unwrap_batched_single(output, batch_size):
    if isinstance(output, torch.Tensor):
        if torch._is_batched(output):
            return torch._unwrap_batched(output, 0)
        output = output.expand(batch_size, *output.shape)
        return output
    else:
        assert False  # NYI


def _unwrapped_batched(batched_outputs, batch_size):
    return [_unwrap_batched_single(out, batch_size)
            for out in batched_outputs]


def vmap(fn, in_axes):
    @functools.wraps(fn)
    def wrapped(*args):
        global VMAP_LEVEL
        VMAP_LEVEL += 1
        try:
            batched_inputs, batch_size = _make_batched(args, in_axes, VMAP_LEVEL)
            batched_outputs = fn(*batched_inputs)
            # TODO: we assume only one output for now
            return _unwrap_batched_single(batched_outputs, batch_size)
        finally:
            VMAP_LEVEL -= 1
    return wrapped
