import functools
from typing import Any, Callable, List, no_type_check, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
    _assert_in_training_states,
    _is_composable,
    _State,
    TrainingState,
)
from torch.distributed.fsdp._utils import p_assert
from torch.distributed.utils import apply_to_tensors
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.fsdp.flat_param import (
    _HandlesKey,
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
    HandleTrainingState,
)
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


@no_type_check
def _pre_forward(
    state: _State,
    handles: List[FlatParamHandle],
    unshard_fn: Callable,
    module: nn.Module,
    input: Any,
):
    """
    Runs the pre-forward logic. This includes an opportunity to unshard
    currently sharded parameters such as those for the current forward and
    registering post-backward hooks for these current parameters.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        unshard_fn (Optional[Callable]): A callable to unshard any currently
            sharded parameters or ``None`` to not do any unsharding.
        module (nn.Module): Module whose forward this method runs right before.
        input (Any): Unused; expected by the hook signature.
    """
    state.training_state = TrainingState.FORWARD_BACKWARD
    state._exec_order_data.record_pre_forward(handles, module.training)
    for handle in handles:
        handle._training_state = HandleTrainingState.FORWARD
    if unshard_fn is not None:
        unshard_fn()
    # Register post-backward hooks to reshard the parameters and reduce-scatter
    # their gradients. They must be re-registered every forward pass in case
    # the `grad_fn` is mutated.
    _register_post_backward_hooks(state, handles)


@no_type_check
def _post_forward(
    state: _State,
    handles: List[FlatParamHandle],
    reshard_fn: Callable,
    module: nn.Module,
    input: Any,
    output: Any,
) -> Any:
    """
    Runs the post-forward logic. This includes an opportunity to reshard
    currently unsharded parameters such as those used in the current forward
    and registering pre-backward hooks on the forward outputs.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        reshard_fn (Optional[Callable]): A callable to reshard any currently
            unsharded parameters (e.g. from the current forward) or ``None`` to
            not do any resharding.
        module (nn.Module): Unused; expected by the hook signature.
        input (Any): Unused; exepcted by the hook signature.
        output (Any): Forward pass output; pre-backward hooks are registered on
            the tensors that require gradients in this output.

    Postcondition: Each ``FlatParameter`` 's data points to the sharded
    flattened parameter.
    """
    state._exec_order_data.record_post_forward(handles)
    if reshard_fn is not None:
        reshard_fn()
    # Register pre-backward hooks to unshard the flattened parameters
    # for the gradient computation (if needed)
    output = _register_pre_backward_hooks(state, output, handles)
    state.training_state = TrainingState.IDLE
    for handle in handles:
        handle._training_state = HandleTrainingState.IDLE
    return output


@no_type_check
def _pre_backward_hook(
    state: _State,
    _handles: List[FlatParamHandle],
    *unused: Any,
) -> Any:
    """Prepares ``_handles`` 's ``FlatParameter`` s for gradient computation."""
    _handles_key = tuple(_handles)  # avoid shadowing `handles_key`
    # Only run the pre-backward hook once per group of handles involved in the
    # same module forward computation
    if _handles_key and state._ran_pre_backward_hook.get(_handles_key, False):
        return

    with torch.autograd.profiler.record_function(
        "FullyShardedDataParallel._pre_backward_hook"
    ):
        # Queue the post-backward callback once for the root FSDP instance to
        # attach it to the outermost backward graph task so that it is called
        # after all backward calls complete
        if state._is_root and not state._post_backward_callback_queued:
            state._queue_wait_for_post_backward()
            all_handles = (
                state._fsdp_handles(state) if _is_composable(state) else state._handles
            )
            _clear_grads_if_needed(all_handles)
        elif _handles_key:
            _assert_in_training_states(state, [TrainingState.IDLE])
        state.training_state = TrainingState.FORWARD_BACKWARD
        # Queueing the post-backward callback is the only logic that is not
        # per-handle in the pre-backward hook, so we can return early here if
        # there are no handles.
        if not _handles_key:
            return
        for handle in _handles:
            handle._training_state = HandleTrainingState.BACKWARD_PRE

        # If the handles have been prefetched, this `_unshard()` simply
        # switches to using the unsharded parameter
        _unshard(
            state, _handles, state._streams["unshard"], state._streams["pre_unshard"]
        )
        torch.cuda.current_stream().wait_stream(state._streams["unshard"])

        # Set this to `False` to ensure that a mistargeted prefetch does not
        # actually unshard these handles
        state._needs_pre_backward_unshard[_handles_key] = False
        _prefetch_handles(state, _handles_key)
        for handle in _handles:
            handle.prepare_gradient_for_backward()
        state._ran_pre_backward_hook[_handles_key] = True


@no_type_check
@torch.no_grad()
def _post_backward_hook(
    state: _State,
    handle: FlatParamHandle,
    *unused: Any,
):
    """
    Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

    Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
    unsharded gradient for the local batch.

    Postcondition:
    - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced
    unsharded gradient.
    - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded
    gradient (accumulating with any existing gradient).
    """
    param = handle.flat_param
    param._post_backward_called = True
    with torch.autograd.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook"
    ):
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        state.training_state = TrainingState.FORWARD_BACKWARD
        p_assert(
            handle._training_state == HandleTrainingState.BACKWARD_PRE,
            f"Expects `BACKWARD_PRE` state but got {handle._training_state}",
        )
        handle._training_state = HandleTrainingState.BACKWARD_POST

        if param.grad is None:
            return
        if param.grad.requires_grad:
            raise RuntimeError("FSDP does not support gradients of gradients")

        free_unsharded_flat_param = _should_free_in_backward(state, handle)
        _reshard(state, [handle], [free_unsharded_flat_param])

        # TODO: Post-backward prefetching does not support the multiple handles
        # per module case since the post-backward hook runs per handle, not per
        # group of handles.
        handles_key = (handle,)
        _prefetch_handles(state, handles_key)

        if not state._sync_gradients:
            return

        # Wait for all ops in the current stream (e.g. gradient
        # computation) to finish before reduce-scattering the gradient
        state._streams["post_backward"].wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(state._streams["post_backward"]):
            unsharded_grad_data = param.grad.data
            if state._exec_order_data.is_first_iter:  # only check once
                _check_comm_hook(
                    state._communication_hook, state._communication_hook_state
                )
            if handle._uses_reduce_mixed_precision and not _low_precision_hook_enabled(
                state
            ):
                # TODO: Use the low precision communication hook directly
                param.grad.data = param.grad.to(state.mixed_precision.reduce_dtype)

            if handle.uses_sharded_strategy:
                # We clear `.grad` to permit multiple backwards. This avoids a
                # race where the second backward pass computation precedes
                # ahead of the first backward pass reduction, which is possible
                # since the reduction is issued in a separate stream and is
                # async and would result in reducing the wrong gradient.
                unsharded_grad = param.grad.data
                param.grad = None
                p_assert(
                    len(unsharded_grad.size()) == 1,
                    f"Expects gradient to be flattened but got {unsharded_grad.size()}",
                )
                chunks = list(unsharded_grad.chunk(state.world_size))
                numel_to_pad = (
                    state.world_size * chunks[0].numel() - unsharded_grad.numel()
                )
                padded_unsharded_grad = F.pad(unsharded_grad, [0, numel_to_pad])
                new_sharded_grad = torch.zeros_like(chunks[0])  # padded
                state._communication_hook(
                    state._communication_hook_state,
                    padded_unsharded_grad,
                    new_sharded_grad,
                )
                _cast_grad_to_param_dtype(state, handle, new_sharded_grad, param)

                # Save the sharded gradient in `_saved_grad_shard` to support
                # gradient accumulation -- for multiple backwards, the gradient
                # reductions may happen in arbitrary order
                accumulate_grad = hasattr(param, "_saved_grad_shard")
                if accumulate_grad:
                    _check_grad_to_accumulate(new_sharded_grad, param._saved_grad_shard)
                    param._saved_grad_shard += new_sharded_grad
                else:
                    param._saved_grad_shard = new_sharded_grad
                sharded_grad = param._saved_grad_shard
            else:
                state._communication_hook(state._communication_hook_state, param.grad)
                # For `NO_SHARD`, we can keep the low precision gradients by
                # simply omitting the cast altogether
                if not handle._keep_low_precision_grads:
                    _cast_grad_to_param_dtype(state, handle, param.grad, param)
                sharded_grad = param.grad.data

            if handle._config.offload_params:
                # Offload the gradient to CPU to ensure parameters and
                # gradients are on the same device as required by the optimizer
                param._cpu_grad.copy_(  # type: ignore[attr-defined]
                    sharded_grad.detach(), non_blocking=True
                )  # synchronized in the post-backward callback
                # Since the sharded gradient is produced in the post-backward
                # stream and consumed later in the computation stream, inform
                # the caching allocator
                sharded_grad.data.record_stream(torch.cuda.current_stream())

            # Since the unsharded gradient is produced in the computation
            # stream and consumed in the post-backward stream, inform the
            # caching allocator (before it goes out of scope)
            unsharded_grad_data.record_stream(state._streams["post_backward"])

            if handle._use_orig_params:
                # Since the handle's `FlatParameter` completed its gradient
                # computation, we should reset the gradient noneness mask
                handle._reset_is_grad_none()
                # Delay using sharded gradient views until after the
                # reduce-scatter instead of immediately after resharding
                handle._use_sharded_grad_views()


@no_type_check
def _should_free_in_backward(
    state: _State,
    handle: FlatParamHandle,
) -> bool:
    """
    Returns whether FSDP should free the unsharded flattened parameter in the
    post-backward or not.
    """
    return (
        state._sync_gradients and handle.uses_sharded_strategy
    ) or handle._config.sharding_strategy == HandleShardingStrategy.FULL_SHARD


@no_type_check
def _cast_grad_to_param_dtype(
    state: _State,
    handle: FlatParamHandle,
    sharded_grad: torch.Tensor,
    param: FlatParameter,
):
    """
    Casts ``sharded_grad`` back to the full parameter dtype so that the
    optimizer step runs with that dtype. This performs an actual cast if
    1. parameters were in reduced precision during the forward since then
    gradients would be in that reduced precision, or
    2. parameters were not in reduced precision but gradients were in
    reduced precision for communication.
    However, if a low precision communication hook is registered, then this
    dtype cast happens in the hook instead.
    """
    _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
    if not _low_precision_hook_enabled(state) and (
        handle._uses_param_mixed_precision or handle._uses_reduce_mixed_precision
    ):
        low_prec_grad_data = sharded_grad.data
        sharded_grad.data = sharded_grad.data.to(dtype=param.dtype)
        # Since for `NO_SHARD`, the gradient is produced in the computation
        # stream and consumed here in the post-backward stream, inform the
        # caching allocator; for the sharded strategies, the gradient is
        # produced in the post-backward stream, so this `record_stream()`
        # should be a no-op
        low_prec_grad_data.record_stream(torch.cuda.current_stream())


def _check_comm_hook(
    comm_hook: Any,
    comm_hook_state: Any,
) -> None:
    p_assert(comm_hook is not None, "Communication hook should not be `None`")
    p_assert(
        comm_hook_state is not None, "Communication hook state should not be `None`"
    )


def _check_grad_to_accumulate(
    new_sharded_grad: torch.Tensor,
    accumulated_grad: torch.Tensor,
) -> None:
    p_assert(
        accumulated_grad.shape == new_sharded_grad.shape,
        "Shape mismatch when accumulating gradients: "
        f"existing gradient shape={accumulated_grad.shape} "
        f"new gradient shape={new_sharded_grad.shape}",
    )
    p_assert(
        accumulated_grad.device == new_sharded_grad.device,
        "Device mismatch when accumulating gradients: "
        f"existing gradient device={accumulated_grad.device} "
        f"new gradient device={new_sharded_grad.device}",
    )


@no_type_check
def _low_precision_hook_enabled(state: _State) -> bool:
    return state._communication_hook in LOW_PRECISION_HOOKS


@no_type_check
def _prefetch_handles(
    state: _State,
    current_handles_key: _HandlesKey,
) -> None:
    """
    Prefetches the next handles if needed (without synchronization). An empty
    handles key cannot prefetch.
    """
    if not current_handles_key:
        return
    handles_to_prefetch = _get_handles_to_prefetch(state, current_handles_key)
    for handles_key in handles_to_prefetch:
        # Prefetch the next set of handles without synchronizing to allow
        # the sync to happen as late as possible to maximize overlap
        _unshard(
            state, handles_key, state._streams["unshard"], state._streams["pre_unshard"]
        )
        state._handles_prefetched[handles_key] = True


@no_type_check
def _get_handles_to_prefetch(
    state: _State,
    current_handles_key: _HandlesKey,
) -> List[_HandlesKey]:
    """
    Returns a :class:`list` of the handles keys to prefetch for the next
    module(s), where ``current_handles_key`` represents the current module.

    "Prefetching" refers to running the unshard logic early (without
    synchronization), and the "next" modules depend on the recorded execution
    order and the current training state.
    """
    training_state = _get_training_state(current_handles_key)
    valid_training_states = (
        HandleTrainingState.BACKWARD_PRE,
        HandleTrainingState.BACKWARD_POST,
        HandleTrainingState.FORWARD,
    )
    p_assert(
        training_state in valid_training_states,
        f"Prefetching is only supported in {valid_training_states} but "
        f"currently in {training_state}",
    )
    eod = state._exec_order_data
    target_handles_keys: List[_HandlesKey] = []
    if (
        training_state == HandleTrainingState.BACKWARD_PRE
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
    ) or (
        training_state == HandleTrainingState.BACKWARD_POST
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST
    ):
        target_handles_keys = [
            target_handles_key
            for target_handles_key in eod.get_handles_to_backward_prefetch(
                current_handles_key
            )
            if state._needs_pre_backward_unshard.get(target_handles_key, False)
            and not state._handles_prefetched.get(target_handles_key, False)
        ]
    elif training_state == HandleTrainingState.FORWARD and state.forward_prefetch:
        target_handles_keys = [
            target_handles_key
            for target_handles_key in eod.get_handles_to_forward_prefetch(
                current_handles_key
            )
            if state._needs_pre_forward_unshard.get(target_handles_key, False)
            and not state._handles_prefetched.get(target_handles_key, False)
        ]
    return target_handles_keys


def _get_training_state(
    handles_key: _HandlesKey,
) -> HandleTrainingState:
    """Returns the training state of the handles in ``handles_key``."""
    p_assert(len(handles_key) > 0, "Expects a non-empty handles key")
    training_states = set(handle._training_state for handle in handles_key)
    p_assert(
        len(training_states) == 1,
        f"Expects uniform training state but got {training_states}",
    )
    return next(iter(training_states))


@no_type_check
def _register_pre_backward_hooks(
    state: _State,
    outputs: Any,
    handles: List[FlatParamHandle],
) -> None:
    """
    Registers pre-backward hooks on the tensors that require gradients in the
    forward pass outputs ``outputs``, which were computed using the
    ``FlatParameter`` s of ``handles``.

    Returns:
        Forward pass outputs with pre-backward hooks registered to tensors that
        require gradients.
    """
    # If there is no gradient computation, then there is no need for
    # pre-backward logic
    if not torch.is_grad_enabled():
        return outputs
    if state._is_root:
        state._post_backward_callback_queued = False  # only defined on the root

    handles_key = tuple(handles)
    if handles_key:
        # Since these handles' `FlatParameter`s participated in a forward, we
        # conservatively assume that they will be used in the backward
        state._needs_pre_backward_unshard[handles_key] = False
        state._ran_pre_backward_hook[handles_key] = False

    def _register_hook(t: torch.Tensor) -> torch.Tensor:
        if t.requires_grad:
            t.register_hook(functools.partial(_pre_backward_hook, state, handles))
            state._needs_pre_backward_unshard[handles_key] = True
        return t

    return apply_to_tensors(_register_hook, outputs)


def _register_post_backward_hooks(
    state: _State,
    handles: List[FlatParamHandle],
) -> None:
    """
    Registers post-backward hooks on the ``FlatParameter`` s'
    ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.

    The ``AccumulateGrad`` object represents the last function that finalizes
    the ``FlatParameter`` 's gradient, so it only runs after its entire
    gradient computation has finished.

    We register the post-backward hook only once in the *first* forward that a
    ``FlatParameter`` participates in. This relies on the ``AccumulateGrad``
    object being preserved through multiple forwards.
    """
    # If there is no gradient computation, then there is no need for
    # post-backward logic
    if not torch.is_grad_enabled():
        return
    for handle in handles:
        flat_param = handle.flat_param
        already_registered = hasattr(flat_param, "_post_backward_hook_state")
        if already_registered or not flat_param.requires_grad:
            continue
        # Get the `AccumulateGrad` object
        temp_flat_param = flat_param.expand_as(flat_param)
        p_assert(
            temp_flat_param.grad_fn is not None,
            "The `grad_fn` is needed to access the `AccumulateGrad` and "
            "register the post-backward hook",
        )
        acc_grad = temp_flat_param.grad_fn.next_functions[0][0]
        hook_handle = acc_grad.register_hook(
            functools.partial(_post_backward_hook, state, handle)
        )
        flat_param._post_backward_hook_state = (acc_grad, hook_handle)  # type: ignore[attr-defined]


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
        return (apply_to_tensors(cast_fn, args), apply_to_tensors(cast_fn, kwargs))
