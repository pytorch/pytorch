# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import functools
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import tree_flatten, tree_unflatten, _broadcast_to_and_flatten, TreeSpec
from .pytree_hacks import tree_map_
from functools import partial

from torch._C._functorch import (
    _add_batch_dim,
    _remove_batch_dim,
    _vmap_decrement_nesting,
    _vmap_increment_nesting,
)

in_dims_t = Union[int, Tuple]
out_dims_t = Union[int, Tuple[int, ...]]


def doesnt_support_saved_tensors_hooks(f):
    message = (
        "functorch transforms don't yet support saved tensor hooks. "
        "Please open an issue with your use case."
    )

    @functools.wraps(f)
    def fn(*args, **kwargs):
        with torch.autograd.graph.disable_saved_tensors_hooks(message):
            return f(*args, **kwargs)
    return fn


# Checks that all args-to-be-batched have the same batch dim size
def _validate_and_get_batch_size(
        flat_in_dims: List[Optional[int]],
        flat_args: List) -> int:
    batch_sizes = [arg.size(in_dim) for in_dim, arg in zip(flat_in_dims, flat_args)
                   if in_dim is not None]
    if len(batch_sizes) == 0:
        raise ValueError('vmap: Expected at least one Tensor to vmap over')
    if batch_sizes and any(size != batch_sizes[0] for size in batch_sizes):
        raise ValueError(
            f'vmap: Expected all tensors to have the same size in the mapped '
            f'dimension, got sizes {batch_sizes} for the mapped dimension')
    return batch_sizes[0]


def _num_outputs(batched_outputs: Union[Tensor, Tuple[Tensor, ...]]) -> int:
    if isinstance(batched_outputs, tuple):
        return len(batched_outputs)
    return 1

# If value is a tuple, check it has length `num_elements`.
# If value is not a tuple, make a tuple with `value` repeated `num_elements` times


def _as_tuple(value: Any, num_elements: int, error_message_lambda: Callable[[], str]) -> Tuple:
    if not isinstance(value, tuple):
        return (value,) * num_elements
    if len(value) != num_elements:
        raise ValueError(error_message_lambda())
    return value


def _process_batched_inputs(
    in_dims: in_dims_t, args: Tuple, func: Callable
) -> Tuple[int, List[Any], List[Any], TreeSpec]:
    if not isinstance(in_dims, int) and not isinstance(in_dims, tuple):
        raise ValueError(
            f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): '
            f'expected `in_dims` to be int or a (potentially nested) tuple '
            f'matching the structure of inputs, got: {type(in_dims)}.')
    if len(args) == 0:
        raise ValueError(
            f'vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add '
            f'inputs, or you are trying to vmap over a function with no inputs. '
            f'The latter is unsupported.')

    flat_args, args_spec = tree_flatten(args)
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    if flat_in_dims is None:
        raise ValueError(
            f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): '
            f'in_dims is not compatible with the structure of `inputs`. '
            f'in_dims has structure {tree_flatten(in_dims)[1]} but inputs '
            f'has structure {args_spec}.')

    for i, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(
                f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): '
                f'Got in_dim={in_dim} for an input but in_dim must be either '
                f'an integer dimension or None.')
        if isinstance(in_dim, int) and not isinstance(arg, Tensor):
            raise ValueError(
                f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): '
                f'Got in_dim={in_dim} for an input but the input is of type '
                f'{type(arg)}. We cannot vmap over non-Tensor arguments, '
                f'please use None as the respective in_dim')
        if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
            raise ValueError(
                f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): '
                f'Got in_dim={in_dim} for some input, but that input is a Tensor '
                f'of dimensionality {arg.dim()} so expected in_dim to satisfy '
                f'-{arg.dim()} <= in_dim < {arg.dim()}.')
        if in_dim is not None and in_dim < 0:
            flat_in_dims[i] = in_dim % arg.dim()

    return _validate_and_get_batch_size(flat_in_dims, flat_args), flat_in_dims, flat_args, args_spec

# Creates BatchedTensors for every Tensor in arg that should be batched.
# Returns the (potentially) batched arguments and the batch_size.


def _create_batched_inputs(
        flat_in_dims: List[Any], flat_args: List[Any], vmap_level: int, args_spec) -> Tuple:
    # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    batched_inputs = [arg if in_dim is None else
                      _add_batch_dim(arg, in_dim, vmap_level)
                      for in_dim, arg in zip(flat_in_dims, flat_args)]
    return tree_unflatten(batched_inputs, args_spec)

# Undos the batching (and any batch dimensions) associated with the `vmap_level`.


def _unwrap_batched(
        batched_outputs: Union[Tensor, Tuple[Tensor, ...]],
        out_dims: out_dims_t,
        vmap_level: int, batch_size: int, func: Callable) -> Tuple:
    flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

    for out in flat_batched_outputs:
        if isinstance(out, torch.Tensor):
            continue
        raise ValueError(f'vmap({_get_name(func)}, ...): `{_get_name(func)}` must only return '
                         f'Tensors, got type {type(out)} as a return.')

    def incompatible_error():
        raise ValueError(
            f'vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): '
            f'out_dims is not compatible with the structure of `outputs`. '
            f'out_dims has structure {tree_flatten(out_dims)[1]} but outputs '
            f'has structure {output_spec}.')

    if isinstance(batched_outputs, torch.Tensor):
        # Some weird edge case requires us to spell out the following
        # see test_out_dims_edge_case
        if isinstance(out_dims, int):
            flat_out_dims = [out_dims]
        elif isinstance(out_dims, tuple) and len(out_dims) == 1:
            flat_out_dims = out_dims
            out_dims = out_dims[0]
        else:
            incompatible_error()
    else:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, output_spec)
        if flat_out_dims is None:
            incompatible_error()

    flat_outputs = [
        _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims)
    ]
    return tree_unflatten(flat_outputs, output_spec)


def _check_int(x, func, out_dims):
    if isinstance(x, int):
        return
    raise ValueError(
        f'vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be '
        f'an int or a python collection of ints representing where in the outputs the '
        f'vmapped dimension should appear.')


def _check_out_dims_is_int_or_int_pytree(out_dims: out_dims_t, func: Callable) -> None:
    if isinstance(out_dims, int):
        return
    tree_map_(partial(_check_int, func=func, out_dims=out_dims), out_dims)


def _get_name(func: Callable):
    if hasattr(func, '__name__'):
        return func.__name__

    # Not all callables have __name__, in fact, only static functions/methods do.
    # A callable created via functools.partial or an nn.Module, to name some
    # examples, don't have a __name__.
    return repr(func)

# vmap(func)(inputs) wraps all Tensor inputs to be batched in BatchedTensors,
# sends those into func, and then unwraps the output BatchedTensors. Operations
# on BatchedTensors perform the batched operations that the user is asking for.
#
# vmap's randomness behavior differs from JAX's, which would require a PRNG key
# to be passed everywhere.


def vmap(
        func: Callable,
        in_dims: in_dims_t = 0,
        out_dims: out_dims_t = 0,
        randomness: str = 'error') -> Callable:
    """
    vmap is the vectorizing map; ``vmap(func)`` returns a new function that
    maps :attr:`func` over some dimension of the inputs. Semantically, vmap
    pushes the map into PyTorch operations called by :attr:`func`, effectively
    vectorizing those operations.

    vmap is useful for handling batch dimensions: one can write a function
    :attr:`func` that runs on examples and then lift it to a function that can
    take batches of examples with ``vmap(func)``. vmap can also be used to
    compute batched gradients when composed with autograd.

    Args:
        func (function): A Python function that takes one or more arguments.
            Must return one or more Tensors.
        in_dims (int or nested structure): Specifies which dimension of the
            inputs should be mapped over. :attr:`in_dims` should have a
            structure like the inputs. If the :attr:`in_dim` for a particular
            input is None, then that indicates there is no map dimension.
            Default: 0.
        out_dims (int or Tuple[int]): Specifies where the mapped dimension
            should appear in the outputs. If :attr:`out_dims` is a Tuple, then
            it should have one element per output. Default: 0.
        randomness (str): Specifies whether the randomness in this
            vmap should be the same or different across batches. If 'different',
            the randomness for each batch will be different. If 'same', the
            randomness will be the same across batches. If 'error', any calls to
            random functions will error. Default: 'error'. WARNING: this flag
            only applies to random PyTorch operations and does not apply to
            Python's random module or numpy randomness.

    Returns:
        Returns a new "batched" function. It takes the same inputs as
        :attr:`func`, except each input has an extra dimension at the index
        specified by :attr:`in_dims`. It takes returns the same outputs as
        :attr:`func`, except each output has an extra dimension at the index
        specified by :attr:`out_dims`.

    .. warning:
        :func:`vmap` works best with functional-style code. Please do not
        perform any side-effects in :attr:`func`, with the exception of
        in-place PyTorch operations. Examples of side-effects include mutating
        Python data structures and assigning values to variables not captured
        in :attr:`func`.

    One example of using :func:`vmap` is to compute batched dot products. PyTorch
    doesn't provide a batched ``torch.dot`` API; instead of unsuccessfully
    rummaging through docs, use :func:`vmap` to construct a new function.

        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = functorch.vmap(torch.dot)  # [N, D], [N, D] -> [N]
        >>> x, y = torch.randn(2, 5), torch.randn(2, 5)
        >>> batched_dot(x, y)

    :func:`vmap` can be helpful in hiding batch dimensions, leading to a simpler
    model authoring experience.

        >>> batch_size, feature_size = 3, 5
        >>> weights = torch.randn(feature_size, requires_grad=True)
        >>>
        >>> def model(feature_vec):
        >>>     # Very simple linear model with activation
        >>>     return feature_vec.dot(weights).relu()
        >>>
        >>> examples = torch.randn(batch_size, feature_size)
        >>> result = functorch.vmap(model)(examples)

    :func:`vmap` can also help vectorize computations that were previously difficult
    or impossible to batch. One example is higher-order gradient computation.
    The PyTorch autograd engine computes vjps (vector-Jacobian products).
    Computing a full Jacobian matrix for some function f: R^N -> R^N usually
    requires N calls to ``autograd.grad``, one per Jacobian row. Using :func:`vmap`,
    we can vectorize the whole computation, computing the Jacobian in a single
    call to ``autograd.grad``.

        >>> # Setup
        >>> N = 5
        >>> f = lambda x: x ** 2
        >>> x = torch.randn(N, requires_grad=True)
        >>> y = f(x)
        >>> I_N = torch.eye(N)
        >>>
        >>> # Sequential approach
        >>> jacobian_rows = [torch.autograd.grad(y, x, v, retain_graph=True)[0]
        >>>                  for v in I_N.unbind()]
        >>> jacobian = torch.stack(jacobian_rows)
        >>>
        >>> # vectorized gradient computation
        >>> def get_vjp(v):
        >>>     return torch.autograd.grad(y, x, v)
        >>> jacobian = functorch.vmap(get_vjp)(I_N)

    :func:`vmap` can also be nested, producing an output with multiple batched dimensions

        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = functorch.vmap(functorch.vmap(torch.dot))  # [N1, N0, D], [N1, N0, D] -> [N1, N0]
        >>> x, y = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
        >>> batched_dot(x, y) # tensor of size [2, 3]

    If the inputs are not batched along the first dimension, :attr:`in_dims` specifies
    the dimension that each inputs are batched along as

        >>> torch.dot                            # [N], [N] -> []
        >>> batched_dot = functorch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
        >>> x, y = torch.randn(2, 5), torch.randn(2, 5)
        >>> batched_dot(x, y)   # output is [5] instead of [2] if batched along the 0th dimension

    If there are multiple inputs each of which is batched along different dimensions,
    :attr:`in_dims` must be a tuple with the batch dimension for each input as

        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = functorch.vmap(torch.dot, in_dims=(0, None))  # [N, D], [D] -> [N]
        >>> x, y = torch.randn(2, 5), torch.randn(5)
        >>> batched_dot(x, y) # second arg doesn't have a batch dim because in_dim[1] was None

    If the input is a Python struct, :attr:`in_dims` must be a tuple containing a struct
    matching the shape of the input:

        >>> f = lambda dict: torch.dot(dict['x'], dict['y'])
        >>> x, y = torch.randn(2, 5), torch.randn(5)
        >>> input = {'x': x, 'y': y}
        >>> batched_dot = functorch.vmap(f, in_dims=({'x': 0, 'y': None},))
        >>> batched_dot(input)

    By default, the output is batched along the first dimension. However, it can be batched
    along any dimension by using :attr:`out_dims`

        >>> f = lambda x: x ** 2
        >>> x = torch.randn(2, 5)
        >>> batched_pow = functorch.vmap(f, out_dims=1)
        >>> batched_pow(x) # [5, 2]

    For any function that uses kwargs, the returned function will not batch the kwargs but will
    accept kwargs

        >>> x = torch.randn([2, 5])
        >>> def f(x, scale=4.):
        >>>   return x * scale
        >>>
        >>> batched_pow = functorch.vmap(f)
        >>> assert torch.allclose(batched_pow(x), x * 4)
        >>> batched_pow(x, scale=x) # scale is not batched, output has shape [2, 2, 5]

    .. note::
        vmap does not provide general autobatching or handle variable-length
        sequences out of the box.
    """
    _check_randomness_arg(randomness)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        _check_out_dims_is_int_or_int_pytree(out_dims, func)
        batch_size, flat_in_dims, flat_args, args_spec = _process_batched_inputs(in_dims, args, func)
        return _flat_vmap(
            func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs
        )

    return wrapped


def chunk_vmap(
        func: Callable,
        in_dims: in_dims_t = 0,
        out_dims: out_dims_t = 0,
        randomness: str = 'error',
        chunks=2) -> Callable:
    """
    chunk_vmap is the vectorizing map (vmap) using chunks of input data. It is a mix of vmap (which vectorizes
    everything) and map (which executes things sequentially). ``chunk_vmap`` vectorizes the input with number of
    chunks at a time. For more details about vectorizing map, see :func:`vmap`.

    Args:
        func (function): A Python function that takes one or more arguments.
            Must return one or more Tensors.
        in_dims (int or nested structure): Specifies which dimension of the
            inputs should be mapped over. :attr:`in_dims` should have a
            structure like the inputs. If the :attr:`in_dim` for a particular
            input is None, then that indicates there is no map dimension.
            Default: 0.
        out_dims (int or Tuple[int]): Specifies where the mapped dimension
            should appear in the outputs. If :attr:`out_dims` is a Tuple, then
            it should have one element per output. Default: 0.
        randomness (str): Specifies whether the randomness in this
            vmap should be the same or different across batches. If 'different',
            the randomness for each batch will be different. If 'same', the
            randomness will be the same across batches. If 'error', any calls to
            random functions will error. Default: 'error'. WARNING: this flag
            only applies to random PyTorch operations and does not apply to
            Python's random module or numpy randomness.
        chunks (int): Number of chunks to use to split the input data. Default is 2.
            If equals to 1 then :func:`vmap` is called.

    Returns:
        Returns a new "batched" function. It takes the same inputs as
        :attr:`func`, except each input has an extra dimension at the index
        specified by :attr:`in_dims`. It takes returns the same outputs as
        :attr:`func`, except each output has an extra dimension at the index
        specified by :attr:`out_dims`.
    """
    _check_randomness_arg(randomness)

    if chunks == 1:
        return vmap(func, in_dims=in_dims, out_dims=out_dims, randomness=randomness)

    def _get_chunk_flat_args(flat_args_, flat_in_dims_, chunks_):
        flat_args_chunks = tuple(
            t.chunk(chunks_, dim=in_dim) if in_dim is not None else [t, ] * chunks_
            for t, in_dim in zip(flat_args_, flat_in_dims_)
        )
        # transpose chunk dim and flatten structure
        # chunks_flat_args is a list of flatten args
        chunks_flat_args = zip(*flat_args_chunks)
        return chunks_flat_args

    def _flatten_chunks_output(chunks_output_):
        # chunks_output is a list of chunked outputs
        # flatten chunked outputs:
        flat_chunks_output = []
        arg_spec_list = []
        for output in chunks_output_:
            flat_output, arg_specs = tree_flatten(output)
            flat_chunks_output.append(flat_output)
            arg_spec_list.append(arg_specs)

        arg_spec = arg_spec_list[0]  # all specs should be the same
        # transpose chunk dim and flatten structure
        # flat_output_chunks is flat list of chunks
        flat_output_chunks = list(zip(*flat_chunks_output))
        return flat_output_chunks, arg_spec

    @functools.wraps(func)
    def wrapped_with_chunks(*args, **kwargs):
        _check_out_dims_is_int_or_int_pytree(out_dims, func)
        _, flat_in_dims, flat_args, args_spec = _process_batched_inputs(in_dims, args, func)
        # Chunk flat arguments
        chunks_flat_args = _get_chunk_flat_args(flat_args, flat_in_dims, chunks)

        # Apply vmap on chunks
        chunks_output = []
        rs = torch.get_rng_state() if randomness == "same" else None
        for flat_args in chunks_flat_args:
            batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)
            if rs is not None:
                torch.set_rng_state(rs)
            chunks_output.append(
                _flat_vmap(
                    func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs
                )
            )
        flat_output_chunks, arg_spec = _flatten_chunks_output(chunks_output)
        # Removing temporary variables helps to reduce memory usage on device like CUDA
        del chunks_output

        # concat chunks on out_dim
        flat_out_dims = _broadcast_to_and_flatten(out_dims, arg_spec)
        assert len(flat_out_dims) == len(flat_output_chunks)
        flat_output = []
        for out_dim in flat_out_dims:
            flat_output.append(torch.cat(flat_output_chunks[0], dim=out_dim))
            # release source data
            del flat_output_chunks[0]
        del flat_output_chunks

        # finally unflatten the output
        return tree_unflatten(flat_output, arg_spec)

    return wrapped_with_chunks


# Vmap refactored helper funcions:
def _check_randomness_arg(randomness):
    if randomness not in ['error', 'different', 'same']:
        raise RuntimeError(f"Only allowed values for randomness are 'error', 'different', or 'same'. Got {randomness}")


@doesnt_support_saved_tensors_hooks
def _flat_vmap(func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs):
    vmap_level = _vmap_increment_nesting(batch_size, randomness)
    try:
        batched_inputs = _create_batched_inputs(flat_in_dims, flat_args, vmap_level, args_spec)
        batched_outputs = func(*batched_inputs, **kwargs)
        return _unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, func)
    finally:
        _vmap_decrement_nesting()
