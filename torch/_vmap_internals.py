import torch
import functools
from torch import Tensor
import warnings

REQUIRE_SAME_MAP_SIZE = (
    'vmap: Expected all tensors to have the same size in the mapped dimension, '
    'got sizes {sizes} for the mapped dimension'
)

ELEMENT_MUST_BE_TENSOR = (
    'vmap({fn}, ...): `{fn}` must only return Tensors, got '
    'type {out} for return {idx}.'
)

MUST_RETURN_TENSORS = (
    'vmap({fn}, ...): `{fn}` must only return Tensors, got '
    'type {out} as the return.'
)

NO_INPUTS = (
    'vmap({fn})(<inputs>): got no inputs. Maybe you forgot '
    'to add inputs, or you are trying to vmap over a '
    'function with no inputs. The latter is unsupported.'
)

OUT_DIMS_MUST_BE_INT_OR_TUPLE_OF_INT = (
    'vmap({fn}, ..., out_dims={out_dims}): `out_dims` must be an int or a tuple '
    'of int representing where in the outputs the vmapped dimension should appear.'
)

OUT_DIMS_AND_NUM_OUTPUTS_MISMATCH = (
    'vmap({fn}, ..., out_dims={out_dims}): `out_dims` must have one dim per '
    'output (got {num_outputs} outputs) of {fn}.'
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

def _num_outputs(batched_outputs):
    if isinstance(batched_outputs, tuple):
        return len(batched_outputs)
    return 1

# If value is a tuple, check it has length `num_elements`.
# If value is not a tuple, make a tuple with `value` repeated `num_elements` times
def _as_tuple(value, num_elements, error_message_lambda):
    if not isinstance(value, tuple):
        return (value,) * num_elements
    if len(value) != num_elements:
        raise ValueError(error_message_lambda())
    return value

# Undos the batching (and any batch dimensions) associated with the `vmap_level`.
def _unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, fn_name):
    num_outputs = _num_outputs(batched_outputs)
    out_dims_as_tuple = _as_tuple(
        out_dims, num_outputs,
        lambda: OUT_DIMS_AND_NUM_OUTPUTS_MISMATCH.format(
            fn=fn_name, out_dims=out_dims, num_outputs=num_outputs))

    # NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    # There is something wrong with our type bindings for functions that begin
    # with '_', see #40397.
    if isinstance(batched_outputs, Tensor):
        out_dim = out_dims_as_tuple[0]
        return torch._remove_batch_dim(batched_outputs, vmap_level, batch_size, out_dim)  # type: ignore
    return tuple(torch._remove_batch_dim(out, vmap_level, batch_size, out_dim)  # type: ignore
                 for out, out_dim in zip(batched_outputs, out_dims_as_tuple))

# Checks that `fn` returned one or more Tensors and nothing else.
# NB: A python function that return multiple arguments returns a single tuple,
# so we are effectively checking that `outputs` is a single Tensor or a tuple of
# Tensors.
def _validate_outputs(outputs, fn_name):
    if isinstance(outputs, Tensor):
        return
    if not isinstance(outputs, tuple):
        raise ValueError(MUST_RETURN_TENSORS.format(fn=fn_name, out=type(outputs)))
    for idx, output in enumerate(outputs):
        if isinstance(output, Tensor):
            continue
        raise ValueError(ELEMENT_MUST_BE_TENSOR.format(fn=fn_name, out=type(output), idx=idx))

def _check_out_dims_is_int_or_int_tuple(out_dims, fn_name):
    if isinstance(out_dims, int):
        return
    if not isinstance(out_dims, tuple) or \
            not all([isinstance(out_dim, int) for out_dim in out_dims]):
        raise ValueError(OUT_DIMS_MUST_BE_INT_OR_TUPLE_OF_INT.format(out_dims=out_dims, fn=fn_name))

# This is the global tracker for how many nested vmaps we are currently inside.
VMAP_LEVEL = 0

# vmap(func)(inputs) wraps all Tensor inputs to be batched in BatchedTensors,
# sends those into func, and then unwraps the output BatchedTensors. Operations
# on BatchedTensors perform the batched operations that the user is asking for.
def vmap(func, in_dims=0, out_dims=0):
    """
    vmap is the vectorizing map. Returns a new function that maps `func` over some
    dimension of the inputs. Semantically, vmap pushes the map into PyTorch
    operations called by `func`, effectively vectorizing those operations.

    vmap is useful for handling batch dimensions: one can write a function `func`
    that runs on examples and the lift it to a function that can take batches of
    examples with `vmap(func)`. Furthermore, it is possible to use vmap to obtain
    batched gradients when composed with autograd.

    Args:
        func (function): A Python function that takes one or more arguments.
            Must return one or more Tensors.
        in_dims (int or Tuple[Optional[int]]): Specifies which dimension of the
            inputs should be mapped over. If `in_dims` is a Tuple, then it should have
            one element per input. If the `in_dim` for a particular input is
            None, then that indicates there is no map dimension. Default: 0.
        out_dims (int or Tuple[int]): Specifies where the mapped dimension
            should appear in the outputs. If `out_dims` is a Tuple, then it should
            have one element per output. Default: 0.

    Returns:
        Returns a new "batched" function. It takes the same inputs as `func`,
        except each input has an extra dimension at the index specified by `in_dims`.
        It takes returns the same outputs as `func`, except each output has
        an extra dimension at the index specified by `out_dims`.

    .. warning:
        vmap works best with functional-style code. Please do not perform any
        side-effects in `func`, with the exception of in-place PyTorch operations.
        Examples of side-effects include mutating Python data structures and
        assigning values to variables not captured in `func`.

    .. warning::
        torch.vmap is an experimental prototype that is subject to
        change and/or deletion. Please use at your own risk.
    """
    warnings.warn(
        'torch.vmap is an experimental prototype that is subject to '
        'change and/or deletion. Please use at your own risk.')

    if in_dims != 0:
        raise NotImplementedError('NYI: vmap with `in_dims` other than 0')

    @functools.wraps(func)
    def wrapped(*args):
        if any(not isinstance(arg, Tensor) for arg in args):
            raise NotImplementedError('NYI: vmap with non-tensor inputs')

        fn_name = func.__name__
        _check_out_dims_is_int_or_int_tuple(out_dims, fn_name)
        batch_size = _validate_inputs_and_get_batch_size(args, fn_name)
        global VMAP_LEVEL
        VMAP_LEVEL += 1
        try:
            # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
            batched_inputs = [torch._add_batch_dim(arg, 0, VMAP_LEVEL) for arg in args]  # type: ignore
            batched_outputs = func(*batched_inputs)
            _validate_outputs(batched_outputs, fn_name)
            return _unwrap_batched(batched_outputs, out_dims, VMAP_LEVEL, batch_size, fn_name)
        finally:
            VMAP_LEVEL -= 1
    return wrapped
