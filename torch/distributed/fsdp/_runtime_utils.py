from typing import Any, List, no_type_check, Optional, Tuple

import torch
from torch.distributed.fsdp._common_utils import _State
from torch.distributed.fsdp._utils import _apply_to_tensors, p_assert
from torch.distributed.fsdp.flat_param import FlatParamHandle
from torch.distributed.utils import _to_kwargs


@no_type_check
def _unshard(
    state: _State,
    handles: List[FlatParamHandle],
    unshard_stream: torch.cuda.Stream,
    pre_unshard_stream: torch.cuda.Stream,
) -> None:
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: Each handle's ``FlatParameter`` 's data is the padded
    unsharded flattened parameter on the compute device.
    """
    if not handles:
        return
    if state.limit_all_gathers:
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            event.synchronize()
    any_ran_pre_unshard = False
    with torch.cuda.stream(pre_unshard_stream):
        for handle in handles:
            ran_pre_unshard = handle.pre_unshard()
            any_ran_pre_unshard = any_ran_pre_unshard or ran_pre_unshard
    if any_ran_pre_unshard:
        unshard_stream.wait_stream(pre_unshard_stream)
    with torch.cuda.stream(unshard_stream):
        for handle in handles:
            handle.unshard()
            handle.post_unshard()


@no_type_check
def _reshard(
    state: _State,
    handles: List[FlatParamHandle],
    free_unsharded_flat_params: List[bool],
):
    """
    Reshards the handles in ``handles``. ``free_unsharded_flat_params`` should
    have the same length as ``handles``, and each element should give whether
    the corresponding handle should free its padded unsharded flattened
    parameter.
    """
    if not handles:
        return
    p_assert(
        len(handles) == len(free_unsharded_flat_params),
        "Expects both lists to have equal length but got "
        f"{len(handles)} and {len(free_unsharded_flat_params)}",
    )
    for handle, free_unsharded_flat_param in zip(
        handles,
        free_unsharded_flat_params,
    ):
        handle.reshard(free_unsharded_flat_param)
        if state.limit_all_gathers and free_unsharded_flat_param:
            free_event = torch.cuda.Event()
            free_event.record()
            state._free_event_queue.enqueue(free_event)
        handle.post_reshard()
    # Since we prefetch entire handles keys at a time, conservatively mark
    # the entire key as no longer prefetched once we free at least one
    handles_key = tuple(handles)
    if any(free_unsharded_flat_params):
        state._handles_prefetched.pop(handles_key, None)


def _unshard_grads(
    handles: List[FlatParamHandle],
) -> None:
    for handle in handles:
        handle.unshard_grad()


def _reshard_grads(
    handles: List[FlatParamHandle],
) -> None:
    for handle in handles:
        handle.reshard_grad()


def _wait_for_computation_stream(
    computation_stream: torch.cuda.Stream,
    unshard_stream: torch.cuda.Stream,
    pre_unshard_stream: torch.cuda.Stream,
):
    """
    Has the unshard and pre-unshard streams wait for the computation stream.
    For example, this should be called in the FSDP root's pre-forward to
    respect optimizer step computation.
    """
    unshard_stream.wait_stream(computation_stream)
    # Having the pre-all-gather stream wait for the current stream even if we
    # do not leverage the pre-all-gather stream is tolerable since this only
    # runs once per iteration
    pre_unshard_stream.wait_stream(computation_stream)


def _clear_grads_if_needed(
    handles: List[FlatParamHandle],
):
    """
    Clears the original parameters' gradients if needed. This method's CPU
    overhead is minimal, so we may call it throughout FSDP methods, which serve
    as callsites to free the gradient memory earlier.
    """
    for handle in handles:
        if handle._use_orig_params:
            handle._clear_grads_if_needed()


def _prepare_forward_inputs(
    device: torch.device,
    input_dtype: Optional[torch.dtype],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Prepares the forward inputs by moving them to ``device`` and casting them
    to ``input_dtype`` if it is not ``None``.
    """
    # TODO: Do not use the side stream for tensor copies for now; investigate
    # the perf with/without it.
    # TODO: For mixed precision, move the inputs to the compute device and cast
    # to reduced-precision in a single `to()` call.
    args_tuple, kwargs_tuple = _to_kwargs(args, kwargs, device.index, False)
    args = args_tuple[0]
    kwargs = kwargs_tuple[0]
    if input_dtype is not None:
        args, kwargs = _cast_fp_inputs_to_dtype(input_dtype, *args, **kwargs)
    return args, kwargs


def _cast_fp_inputs_to_dtype(
    dtype: torch.dtype,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Casts floating point tensors in ``args`` and ``kwargs`` to ``input_dtype``.
    This respects the existing ``requires_grad`` on the tensors.
    """

    def cast_fn(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            return x
        y = x.to(dtype)
        # Explicitly copy over `requires_grad` since this runs inside
        # `torch.no_grad()`
        if x.is_leaf:
            y.requires_grad = x.requires_grad
        return y

    with torch.no_grad():
        return (_apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs))
