

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import itertools
import os
import threading
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch._C._functorch import (
    _add_batch_dim,
    _remove_batch_dim,
    _vmap_decrement_nesting,
    _vmap_increment_nesting,
    is_batchedtensor,
)
from torch.utils._pytree import (
    _broadcast_to_and_flatten,
    tree_flatten,
    tree_map_,
    tree_unflatten,
    TreeSpec,
)


in_dims_t = Union[int, Tuple]
out_dims_t = Union[int, Tuple[int, ...]]


def doesnt_support_saved_tensors_hooks(f):
    message = (
        "torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. "
        "Please open an issue with your use case."
    )

    @functools.wraps(f)
    def fn(*args, **kwargs):
        with torch.autograd.graph.disable_saved_tensors_hooks(message):
            return f(*args, **kwargs)

    return fn


# Checks that all args-to-be-batched have the same batch dim size
def _validate_and_get_batch_size(
    flat_in_dims: List[Optional[int]], flat_args: List
) -> int:
    batch_sizes = [
        arg.size(in_dim)
        for in_dim, arg in zip(flat_in_dims, flat_args)
        if in_dim is not None
    ]
    if len(batch_sizes) == 0:
        raise ValueError("vmap: Expected at least one Tensor to vmap over")
    if batch_sizes and any(size != batch_sizes[0] for size in batch_sizes):
        raise ValueError(
            f"vmap: Expected all tensors to have the same size in the mapped "
            f"dimension, got sizes {batch_sizes} for the mapped dimension"
        )
    return batch_sizes[0]


def _num_outputs(batched_outputs: Union[Tensor, Tuple[Tensor, ...]]) -> int:
    if isinstance(batched_outputs, tuple):
        return len(batched_outputs)
    return 1


# If value is a tuple, check it has length `num_elements`.
# If value is not a tuple, make a tuple with `value` repeated `num_elements` times


def _as_tuple(
    value: Any, num_elements: int, error_message_lambda: Callable[[], str]
) -> Tuple:
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
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"expected `in_dims` to be int or a (potentially nested) tuple "
            f"matching the structure of inputs, got: {type(in_dims)}."
        )
    if len(args) == 0:
        raise ValueError(
            f"vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add "
            f"inputs, or you are trying to vmap over a function with no inputs. "
            f"The latter is unsupported."
        )

    flat_args, args_spec = tree_flatten(args)
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    if flat_in_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"in_dims is not compatible with the structure of `inputs`. "
            f"in_dims has structure {tree_flatten(in_dims)[1]} but inputs "
            f"has structure {args_spec}."
        )

    for i, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but in_dim must be either "
                f"an integer dimension or None."
            )
        if isinstance(in_dim, int) and not isinstance(arg, Tensor):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but the input is of type "
                f"{type(arg)}. We cannot vmap over non-Tensor arguments, "
                f"please use None as the respective in_dim"
            )
        if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for some input, but that input is a Tensor "
                f"of dimensionality {arg.dim()} so expected in_dim to satisfy "
                f"-{arg.dim()} <= in_dim < {arg.dim()}."
            )
        if in_dim is not None and in_dim < 0:
            flat_in_dims[i] = in_dim % arg.dim()

    return (
        _validate_and_get_batch_size(flat_in_dims, flat_args),
        flat_in_dims,
        flat_args,
        args_spec,
    )


# Creates BatchedTensors for every Tensor in arg that should be batched.
# Returns the (potentially) batched arguments and the batch_size.


def _create_batched_inputs(
    flat_in_dims: List[Any], flat_args: List[Any], vmap_level: int, args_spec
) -> Tuple:
    # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    batched_inputs = [
        arg if in_dim is None else _add_batch_dim(arg, in_dim, vmap_level)
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    return tree_unflatten(batched_inputs, args_spec)


def _maybe_remove_batch_dim(name, batched_output, vmap_level, batch_size, out_dim):
    if out_dim is None:
        if isinstance(batched_output, torch.Tensor) and is_batchedtensor(
            batched_output
        ):
            raise ValueError(
                f"vmap({name}, ...): `{name}` can not return a "
                f"BatchedTensor when out_dim is None"
            )
        return batched_output

    # out_dim is non None
    if not isinstance(batched_output, torch.Tensor):
        raise ValueError(
            f"vmap({name}, ...): `{name}` must only return "
            f"Tensors, got type {type(batched_output)}. "
            "Did you mean to set out_dims= to None for output?"
        )

    return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)


# Undos the batching (and any batch dimensions) associated with the `vmap_level`.
def _unwrap_batched(
    batched_outputs: Union[Tensor, Tuple[Tensor, ...]],
    out_dims: out_dims_t,
    vmap_level: int,
    batch_size: int,
    func: Callable,
) -> Tuple:
    flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

    def incompatible_error():
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
            f"out_dims is not compatible with the structure of `outputs`. "
            f"out_dims has structure {tree_flatten(out_dims)[1]} but outputs "
            f"has structure {output_spec}."
        )

    if isinstance(batched_outputs, torch.Tensor):
        # Some weird edge case requires us to spell out the following
        # see test_out_dims_edge_case
        if isinstance(out_dims, int):
            flat_out_dims = [out_dims]
        elif isinstance(out_dims, tuple) and len(out_dims) == 1:
            flat_out_dims = out_dims
        elif out_dims is None:
            flat_out_dims = [out_dims]
        else:
            incompatible_error()
    else:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, output_spec)
        if flat_out_dims is None:
            incompatible_error()

    flat_outputs = [
        _maybe_remove_batch_dim(
            _get_name(func), batched_output, vmap_level, batch_size, out_dim
        )
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims)
    ]
    return tree_unflatten(flat_outputs, output_spec)


def _check_int_or_none(x, func, out_dims):
    if isinstance(x, int):
        return
    if x is None:
        return
    raise ValueError(
        f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be "
        f"an int, None or a python collection of ints representing where in the outputs the "
        f"vmapped dimension should appear."
    )


def _check_out_dims_is_int_or_int_pytree(out_dims: out_dims_t, func: Callable) -> None:
    if isinstance(out_dims, int):
        return
    tree_map_(partial(_check_int_or_none, func=func, out_dims=out_dims), out_dims)


def _get_name(func: Callable):
    if hasattr(func, "__name__"):
        return func.__name__

    # Not all callables have __name__, in fact, only static functions/methods do.
    # A callable created via functools.partial or an nn.Module, to name some
    # examples, don't have a __name__.
    return repr(func)


DECOMPOSITIONS_LOADED = False
DECOMPOSITIONS_LOCK = threading.Lock()
VMAP_DECOMPOSITIONS_LIB = None


# torch.package, Python 3.11, and torch.jit-less environments are unhappy with
# decompositions. Only load them when needed if possible.
def lazy_load_decompositions():
    global DECOMPOSITIONS_LOADED
    if DECOMPOSITIONS_LOADED:
        return

    with DECOMPOSITIONS_LOCK:
        if DECOMPOSITIONS_LOADED:
            return

        if not (os.environ.get("PYTORCH_JIT", "1") == "1" and __debug__):
            DECOMPOSITIONS_LOADED = True
            return

        # use an alternate way to register an operator into the decomposition table
        # _register_jit_decomposition doesn't work for some operators, e.g. addr,
        #  because the Tensor types generated cannot be unioned by torchscript
        # decomp should be type OpOverload
        global VMAP_DECOMPOSITIONS_LIB
        VMAP_DECOMPOSITIONS_LIB = torch.library.Library(
            "aten", "IMPL", "FuncTorchBatched"
        )

        from torch._decomp import decomposition_table

        def _register_python_decomposition_vmap(decomp):
            if decomp in decomposition_table:
                VMAP_DECOMPOSITIONS_LIB.impl(decomp, decomposition_table[decomp])
            else:
                raise RuntimeError(f"could not find decomposition for {decomp}")

        _register_python_decomposition_vmap(torch.ops.aten.mse_loss_backward.default)
        _register_python_decomposition_vmap(
            torch.ops.aten.smooth_l1_loss_backward.default
        )
        _register_python_decomposition_vmap(torch.ops.aten.huber_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.addr.default)

        DECOMPOSITIONS_LOADED = True


def vmap_impl(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs):
    lazy_load_decompositions()
    _check_out_dims_is_int_or_int_pytree(out_dims, func)
    batch_size, flat_in_dims, flat_args, args_spec = _process_batched_inputs(
        in_dims, args, func
    )

    if chunk_size is not None:
        chunks_flat_args = _get_chunked_inputs(
            flat_args, flat_in_dims, batch_size, chunk_size
        )
        return _chunked_vmap(
            func,
            flat_in_dims,
            chunks_flat_args,
            args_spec,
            out_dims,
            randomness,
            **kwargs,
        )

    # If chunk_size is not specified.
    return _flat_vmap(
        func,
        batch_size,
        flat_in_dims,
        flat_args,
        args_spec,
        out_dims,
        randomness,
        **kwargs,
    )


def get_chunk_sizes(total_elems, chunk_size):
    n_chunks = n_chunks = total_elems // chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    # remainder chunk
    remainder = total_elems % chunk_size
    if remainder != 0:
        chunk_sizes.append(remainder)
    return chunk_sizes


def _get_chunked_inputs(flat_args, flat_in_dims, batch_size, chunk_size):
    split_idxs = (batch_size,)
    if chunk_size is not None:
        chunk_sizes = get_chunk_sizes(batch_size, chunk_size)
        split_idxs = tuple(itertools.accumulate(chunk_sizes))

    flat_args_chunks = tuple(
        t.tensor_split(split_idxs, dim=in_dim)
        if in_dim is not None
        else [
            t,
        ]
        * len(split_idxs)
        for t, in_dim in zip(flat_args, flat_in_dims)
    )

    # transpose chunk dim and flatten structure
    # chunks_flat_args is a list of flatten args
    chunks_flat_args = zip(*flat_args_chunks)
    return chunks_flat_args


def _flatten_chunks_output(chunks_output_):
    # chunks_output is a list of chunked outputs
    # flatten chunked outputs:
    flat_chunks_output = []
    arg_spec = None
    for output in chunks_output_:
        flat_output, arg_specs = tree_flatten(output)
        flat_chunks_output.append(flat_output)
        if arg_spec is None:
            arg_spec = arg_specs

    # transpose chunk dim and flatten structure
    # flat_output_chunks is flat list of chunks
    flat_output_chunks = list(zip(*flat_chunks_output))
    return flat_output_chunks, arg_spec


def _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks):
    # concat chunks on out_dim
    flat_out_dims = _broadcast_to_and_flatten(out_dims, arg_spec)
    assert len(flat_out_dims) == len(flat_output_chunks)
    flat_output = []
    for idx, out_dim in enumerate(flat_out_dims):
        flat_output.append(torch.cat(flat_output_chunks[idx], dim=out_dim))
        # release tensors
        flat_output_chunks[idx] = None

    return flat_output


# Applies vmap on chunked_input and returns concatenated output over the chunks.
def _chunked_vmap(
    func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs
):
    chunks_output = []
    rs = torch.get_rng_state() if randomness == "same" else None
    for flat_args in chunks_flat_args:
        batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)

        # The way we compute split the input in `_get_chunked_inputs`,
        # we may get a tensor with `0` batch-size. We skip any computation
        # in that case.
        # Eg.
        # >>> chunk_size = 1
        # >>> batch_size = 6
        # >>> t = torch.zeros(batch_size, 1)
        # >>> t.tensor_split([1, 2, 3, 4, 5, 6])
        # (tensor([[0.]]), tensor([[0.]]), tensor([[0.]]), tensor([[0.]]),
        #  tensor([[0.]]), tensor([[0.]]), tensor([], size=(0, 1)))
        if batch_size == 0:
            continue

        if rs is not None:
            torch.set_rng_state(rs)
        chunks_output.append(
            _flat_vmap(
                func,
                batch_size,
                flat_in_dims,
                flat_args,
                args_spec,
                out_dims,
                randomness,
                **kwargs,
            )
        )

    flat_output_chunks, arg_spec = _flatten_chunks_output(chunks_output)

    # chunked output tensors are held by both `flat_output_chunks` and `chunks_output`.
    # eagerly remove the reference from `chunks_output`.
    del chunks_output

    # concat chunks on out_dim
    flat_output = _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks)

    # finally unflatten the output
    return tree_unflatten(flat_output, arg_spec)


# Vmap refactored helper functions:
def _check_randomness_arg(randomness):
    if randomness not in ["error", "different", "same"]:
        raise RuntimeError(
            f"Only allowed values for randomness are 'error', 'different', or 'same'. Got {randomness}"
        )


@contextlib.contextmanager
def vmap_increment_nesting(batch_size, randomness):
    try:
        vmap_level = _vmap_increment_nesting(batch_size, randomness)
        yield vmap_level
    finally:
        _vmap_decrement_nesting()


def _flat_vmap(
    func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs
):
    with vmap_increment_nesting(batch_size, randomness) as vmap_level:
        batched_inputs = _create_batched_inputs(
            flat_in_dims, flat_args, vmap_level, args_spec
        )
        batched_outputs = func(*batched_inputs, **kwargs)
        return _unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, func)


# `restore_vmap` is a private helper function. It is vmap but has the following
# differences:
# - instead of returning outputs, it returns an (outputs, out_dims) tuple.
#   out_dims is a pytree of same shape as outputs and contains Optional[int]
#   specifying where the vmapped dimension, if it exists, is in the corresponding output.
# - does no validation on in_dims or inputs (vmap expects at least one Tensor to be vmapped).
#   restore_vmap allows for no inputs to have the vmap dimension
# - does no validation on outputs (vmap expects only Tensor outputs)
#   restore_vmap allows for return of arbitrary outputs (not just Tensors)
#
# The TL;DR is that restore_vmap is more general than vmap and has a slightly
# different API. The relaxations are so that we can "pause" vmap in the middle
# of its execution and then "restore" it later (this is what we do in
# the generate_vmap_rule=True implementation of autograd.Function).
#
# restore_vmap can be technically used in the implementation of vmap, but doing
# that refactor is a bit technically challenging because:
# - vmap couples the tensor-wrapping code with error checking
# - vmap's tensor unwrapping code is in C++; we would need to rewrite part of it
#   in python because it overlaps with unwrap_batched
def restore_vmap(func, in_dims, batch_size, randomness):
    def inner(*args, **kwargs):
        with vmap_increment_nesting(batch_size, randomness) as vmap_level:
            batched_inputs = wrap_batched(args, in_dims, vmap_level)
            batched_outputs = func(*batched_inputs, **kwargs)
            return unwrap_batched(batched_outputs, vmap_level)

    return inner


def wrap_batched(args, bdims, level):
    flat_args, spec = tree_flatten(args)
    flat_bdims = _broadcast_to_and_flatten(bdims, spec)
    assert flat_bdims is not None
    result = _create_batched_inputs(flat_bdims, flat_args, level, spec)
    return result


def unwrap_batched(args, level):
    flat_args, spec = tree_flatten(args)
    if len(flat_args) == 0:
        return args, ()
    result = [
        torch._C._functorch._unwrap_batched(arg, level)
        if isinstance(arg, torch.Tensor)
        else (arg, None)
        for arg in flat_args
    ]
    output, bdims = zip(*result)
    return tree_unflatten(output, spec), tree_unflatten(bdims, spec)
