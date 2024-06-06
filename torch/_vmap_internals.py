import functools
from typing import Any, Callable, List, Optional, Tuple, Union
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, tree_unflatten

in_dims_t = Union[int, Tuple]
out_dims_t = Union[int, Tuple[int, ...]]


# Checks that all args-to-be-batched have the same batch dim size
def _validate_and_get_batch_size(
    flat_in_dims: List[Optional[int]], flat_args: List
) -> int:
    batch_sizes = [
        arg.size(in_dim)
        for in_dim, arg in zip(flat_in_dims, flat_args)
        if in_dim is not None
    ]
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


# Creates BatchedTensors for every Tensor in arg that should be batched.
# Returns the (potentially) batched arguments and the batch_size.
def _create_batched_inputs(
    in_dims: in_dims_t, args: Tuple, vmap_level: int, func: Callable
) -> Tuple[Tuple, int]:
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

    for arg, in_dim in zip(flat_args, flat_in_dims):
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
        if in_dim is not None and (in_dim < 0 or in_dim >= arg.dim()):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for some input, but that input is a Tensor "
                f"of dimensionality {arg.dim()} so expected in_dim to satisfy "
                f"0 <= in_dim < {arg.dim()}."
            )

    batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)
    # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    batched_inputs = [
        arg if in_dim is None else torch._add_batch_dim(arg, in_dim, vmap_level)
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    return tree_unflatten(batched_inputs, args_spec), batch_size


# Undos the batching (and any batch dimensions) associated with the `vmap_level`.
def _unwrap_batched(
    batched_outputs: Union[Tensor, Tuple[Tensor, ...]],
    out_dims: out_dims_t,
    vmap_level: int,
    batch_size: int,
    func: Callable,
    allow_none_pass_through: bool = False,
) -> Tuple:
    num_outputs = _num_outputs(batched_outputs)
    out_dims_as_tuple = _as_tuple(
        out_dims,
        num_outputs,
        lambda: f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must "
        f"have one dim per output (got {num_outputs} outputs) of {_get_name(func)}.",
    )

    # NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    # There is something wrong with our type bindings for functions that begin
    # with '_', see #40397.
    if isinstance(batched_outputs, Tensor):
        out_dim = out_dims_as_tuple[0]
        return torch._remove_batch_dim(batched_outputs, vmap_level, batch_size, out_dim)  # type: ignore[return-value]
    if allow_none_pass_through:
        return tuple(
            (
                torch._remove_batch_dim(out, vmap_level, batch_size, out_dim)
                if out is not None
                else None
            )
            for out, out_dim in zip(batched_outputs, out_dims_as_tuple)
        )
    else:
        return tuple(
            torch._remove_batch_dim(out, vmap_level, batch_size, out_dim)
            for out, out_dim in zip(batched_outputs, out_dims_as_tuple)
        )


# Checks that `fn` returned one or more Tensors and nothing else.
# NB: A python function that return multiple arguments returns a single tuple,
# so we are effectively checking that `outputs` is a single Tensor or a tuple of
# Tensors.
def _validate_outputs(outputs: Any, func: Callable) -> None:
    if isinstance(outputs, Tensor):
        return
    if not isinstance(outputs, tuple):
        raise ValueError(
            f"vmap({_get_name(func)}, ...): `{_get_name(func)}` must only return "
            f"Tensors, got type {type(outputs)} as the return."
        )
    for idx, output in enumerate(outputs):
        if isinstance(output, Tensor):
            continue
        raise ValueError(
            f"vmap({_get_name(func)}, ...): `{_get_name(func)}` must only return "
            f"Tensors, got type {type(output)} for return {idx}."
        )


def _check_out_dims_is_int_or_int_tuple(out_dims: out_dims_t, func: Callable) -> None:
    if isinstance(out_dims, int):
        return
    if not isinstance(out_dims, tuple) or not all(
        isinstance(out_dim, int) for out_dim in out_dims
    ):
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be "
            f"an int or a tuple of int representing where in the outputs the "
            f"vmapped dimension should appear."
        )


def _get_name(func: Callable):
    if hasattr(func, "__name__"):
        return func.__name__

    # Not all callables have __name__, in fact, only static functions/methods do.
    # A callable created via functools.partial or an nn.Module, to name some
    # examples, don't have a __name__.
    return repr(func)


# vmap(func)(inputs) wraps all Tensor inputs to be batched in BatchedTensors,
# sends those into func, and then unwraps the output BatchedTensors. Operations
# on BatchedTensors perform the batched operations that the user is asking for.
@deprecated(
    "Please use `torch.vmap` instead of `torch._vmap_internals.vmap`.",
    category=FutureWarning,
)
def vmap(func: Callable, in_dims: in_dims_t = 0, out_dims: out_dims_t = 0) -> Callable:
    """
    Please use torch.vmap instead of this API.
    """
    return _vmap(func, in_dims, out_dims)


# A version of vmap but without the initial "experimental prototype" warning
def _vmap(
    func: Callable,
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    allow_none_pass_through: bool = False,
) -> Callable:
    # The `allow_none_pass_through` argument is a temporary workaround may be removed.
    # Currently it enables us to wrap the call in `autograd.grad` to the autograd engine,
    # which may return None if any of the inputs are unused. See the issue discussing this:
    # https://github.com/facebookresearch/functorch/issues/159.
    @functools.wraps(func)
    def wrapped(*args):
        _check_out_dims_is_int_or_int_tuple(out_dims, func)
        vmap_level = torch._C._vmapmode_increment_nesting()
        try:
            batched_inputs, batch_size = _create_batched_inputs(
                in_dims, args, vmap_level, func
            )
            batched_outputs = func(*batched_inputs)
            if not allow_none_pass_through:
                _validate_outputs(batched_outputs, func)
            return _unwrap_batched(
                batched_outputs,
                out_dims,
                vmap_level,
                batch_size,
                func,
                allow_none_pass_through=allow_none_pass_through,
            )
        finally:
            torch._C._vmapmode_decrement_nesting()

    return wrapped
