import functools
import logging
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
)

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributed.algorithms._comm_hooks import default_hooks, LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
    _assert_in_training_states,
    _FSDPState,
    _get_module_fsdp_state,
    _get_sharding_strategy,
    _is_composable,
    TrainingState,
)
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp._utils import _no_dispatch_record_stream
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.fsdp.flat_param import (
    _HandlesKey,
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
    HandleTrainingState,
    RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
)
from torch.distributed.utils import _apply_to_tensors, _p_assert, _to_kwargs

log = logging.getLogger(__name__)

# Do not include "process_group" to enable hybrid shard and MoE cases
HOMOGENEOUS_ATTR_NAMES = (
    "_use_orig_params",
    "limit_all_gathers",
)


class _PrefetchMode(Enum):
    BACKWARD = auto()
    FORWARD = auto()


def _get_fsdp_root_states_with_modules(
    module: nn.Module,
) -> Tuple[List[_FSDPState], List[nn.Module]]:
    """
    Returns a tuple containing:
    1. A list of the root ``_FSDPState`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the root modules owning the states in the first
    list.

    This is similar to :func:`_get_fsdp_states_with_modules` except that we
    must call :func:`_is_fsdp_root` to force a lazy initialization to determine
    the FSDP root in case lazy initialization has not yet happened.
    """
    fsdp_root_states: List[_FSDPState] = []
    fsdp_root_modules: List[nn.Module] = []
    visited_fsdp_states: Set[_FSDPState] = set()
    # NOTE: This function assumes that `module.modules()` proceeds top-down.
    for submodule in module.modules():
        optional_state = _get_module_fsdp_state(submodule)
        if (
            optional_state is not None
            and optional_state not in visited_fsdp_states
            and _is_fsdp_root(optional_state, submodule)
        ):
            visited_fsdp_states.add(optional_state)
            fsdp_root_states.append(optional_state)
            fsdp_root_modules.append(submodule)
    return fsdp_root_states, fsdp_root_modules


def _get_fsdp_root_states(module: nn.Module) -> List[_FSDPState]:
    """See :func:`_get_fsdp_root_states_with_modules`."""
    fsdp_root_states, _ = _get_fsdp_root_states_with_modules(module)
    return fsdp_root_states


def _is_fsdp_root(state: _FSDPState, module: nn.Module) -> bool:
    """
    Returns if ``state`` corresponds to that of an FSDP root.

    For the wrapper code path, ``state`` and ``module`` should be the same. For
    the non-wrapper code path, ``state`` should be ``module`` 's state.
    """
    # Force a lazy initialization to determine the FSDP root
    _lazy_init(state, module)
    assert state._is_root is not None  # mypy
    return state._is_root


@no_type_check
def _validate_and_get_hybrid_shard_state(
    root_module: nn.Module,
) -> default_hooks.DefaultState:
    """
    Precondition: ``root_module`` is a ``FullyShardedDataParallel`` instance.

    This checks that all instances using a hybrid sharding strategy have the
    same intra- and inter-node process groups.

    Returns:
        DefaultState: One of the instances' inter-node state (does not
        matter which since they will share the same one).
    """
    intra_node_pgs = set()
    inter_node_pgs = set()
    inter_node_states = set()
    for fsdp_module in traversal_utils._get_fsdp_states(root_module):
        # TODO: Change this to handle's sharding strategy if we deprecate
        # `ShardingStrategy` internally.
        # https://github.com/pytorch/pytorch/issues/90857
        if fsdp_module.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
            intra_node_pgs.add(fsdp_module.process_group)
            inter_node_pgs.add(fsdp_module._inter_node_pg)
            inter_node_states.add(fsdp_module._inter_node_state)
    if len(intra_node_pgs) == 0 and len(inter_node_pgs) == 0:
        # No instances use a hybrid sharding strategy
        return None
    error_prefix = "At least one instance uses a hybrid sharding strategy but has no "
    if len(intra_node_pgs) > 0 and len(inter_node_pgs) == 0:
        raise AssertionError(error_prefix + "inter-node process group set")
    if len(intra_node_pgs) == 0 and len(inter_node_pgs) > 0:
        raise AssertionError(error_prefix + "intra-node process group set")
    error_prefix = "Some instances use a hybrid sharding strategy, but "
    if len(intra_node_pgs) != 1:
        raise ValueError(error_prefix + "intra-node process groups do not match")
    if len(inter_node_pgs) != 1:
        raise ValueError(error_prefix + "inter-node process groups do not match")
    return next(iter(inter_node_states))


@no_type_check
def _lazy_init(
    state: _FSDPState,
    root_module: nn.Module,
) -> _FSDPState:
    """
    Performs initialization lazily, typically right before the first forward
    pass. The laziness is needed to ensure that the parameter device/dtype and
    the FSDP hierarchy have finalized. This method's actual logic only runs on
    the root FSDP instance, which performs initialization for all non-root FSDP
    instances to avoid partial initialization.

    For the non-composable code path, ``state`` and ``root_module`` should be
    the same, namely the FSDP instance itself.
    """
    if state._is_root is not None:
        return  # no-op: already lazily initialized
    if not state._device_handle.is_available():
        # Allow the FSDP constructor to run even without CUDA but check this
        # once we start real execution
        raise RuntimeError("FSDP does not support CPU only execution")
    # The following logic is only run on the root FSDP instance since it will
    # set `_is_root=False` for the non-root instances
    state._is_root = True
    _assert_in_training_states(state, [TrainingState.IDLE])
    _check_flat_params_on_expected_device(state, root_module)
    _init_streams(state)
    buffers, buffer_dtypes = _get_buffers_and_dtypes_for_computation(state, root_module)
    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes, state.compute_device)
    state._exec_order_data.init(state, root_module, state.process_group)
    _share_state_and_init_handle_attrs(state, root_module)
    return state


def _check_flat_params_on_expected_device(state: _FSDPState, module: nn.Module):
    """
    Checks that all ``FlatParameter``s in ``module`` 's tree managed by
    ``state`` are on the expected device for *lazy initialization*.
    """
    cpu_device = torch.device("cpu")
    for handle in traversal_utils._get_fsdp_handles(module):
        if (
            not handle._offload_params
            and handle.flat_param.device != state.compute_device
        ):
            raise RuntimeError(
                "An FSDP-managed module unexpectedly has parameters on "
                f"{handle.flat_param.device}. Make sure to move the module to "
                f"{state.compute_device} before training."
            )
        elif handle._offload_params and handle.flat_param.device != cpu_device:
            raise RuntimeError(
                "An FSDP-managed module with parameter CPU offloading enabled "
                f"has parameters on {handle.flat_param.device}. Make sure to "
                f"not move the module from CPU when offloading parameters."
            )


@no_type_check
def _share_state_and_init_handle_attrs(
    root_state: _FSDPState,
    root_module: nn.Module,
) -> None:
    """
    Shares data structure state from the ``root_state`` to all FSDP states in
    ``root_module`` 's module tree, and initializes handle attributes. These
    are done together to require a single loop over the states.
    """
    for handle in root_state._handles:
        handle.init_flat_param_attributes()
    inter_node_state = _validate_and_get_hybrid_shard_state(root_module)
    attr_name_to_values: Dict[str, Set[Any]] = {}
    for attr_name in HOMOGENEOUS_ATTR_NAMES:
        attr_name_to_values[attr_name] = set()
    root_state._all_fsdp_states = traversal_utils._get_fsdp_states(root_module)
    root_state._all_handles = root_state._exec_order_data.all_handles  # share reference
    for fsdp_state in root_state._all_fsdp_states:
        for attr_name in HOMOGENEOUS_ATTR_NAMES:
            _p_assert(
                hasattr(fsdp_state, attr_name),
                f"FSDP state missing attribute {attr_name}",
            )
            attr_name_to_values[attr_name].add(getattr(fsdp_state, attr_name))
        if fsdp_state is root_state:
            continue
        handle_sharding_strategy = _get_sharding_strategy(fsdp_state._handles)
        if handle_sharding_strategy in (
            HandleShardingStrategy.HYBRID_SHARD,
            HandleShardingStrategy._HYBRID_SHARD_ZERO2,
        ):
            # Share the all-reduce state across FSDP units. This is not strictly necessary
            # as each one already uses the same process group, but can slightly save memory
            # since other FSDP units allreduce state can be garbage collected.
            assert inter_node_state is not None, (
                "`_validate_and_get_hybrid_shard_state()` should have returned "
                "a valid inter-node state if there exists an FSDP instance "
                "using a hybrid sharding strategy"
            )
            fsdp_state._inter_node_state = inter_node_state

        # Relax the assert for non-root FSDP instances in case the nested
        # initialized module is wrapped again in FSDP later (e.g. after
        # training to run inference)
        _p_assert(
            fsdp_state._is_root is None or not fsdp_state._is_root,
            "Non-root FSDP instance's `_is_root` should not have been "
            "set yet or should have been set to `False`",
        )
        fsdp_state._is_root = False
        fsdp_state._streams = root_state._streams
        fsdp_state._stream_to_name = root_state._stream_to_name
        fsdp_state._exec_order_data = root_state._exec_order_data
        fsdp_state._free_event_queue = root_state._free_event_queue
        fsdp_state._handles_prefetched = root_state._handles_prefetched
        fsdp_state._needs_pre_backward_unshard = root_state._needs_pre_backward_unshard
        for handle in fsdp_state._handles:
            handle.init_flat_param_attributes()
    for attr_name, attr_values in attr_name_to_values.items():
        if len(attr_values) != 1:
            raise ValueError(
                f"Expects one homogeneous value for {attr_name} but got {attr_values}"
            )


@no_type_check
def _init_streams(
    state: _FSDPState,
) -> _FSDPState:
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    assert state._is_root
    assert state._device_handle.is_available()
    # Stream for unshard logic, including allocating the all-gather destination
    # tensors and the all-gathers themselves.
    state._streams["unshard"] = state._device_handle.Stream()
    # Stream for overlapping gradient reduction with the backward pass gradient
    # computation.
    state._streams["post_backward"] = state._device_handle.Stream()
    # Stream for pre-unshard logic, namely allocations and writes for CPU
    # offloading (H2D copy) and mixed precision (low precision cast).
    state._streams["pre_unshard"] = state._device_handle.Stream()
    # Default stream for computation
    state._streams["default"] = state._device_handle.current_stream()
    state._stream_to_name = {
        state._device_handle.current_stream(): "default",
        state._streams["unshard"]: "unshard",
        state._streams["pre_unshard"]: "pre_unshard",
        state._streams["post_backward"]: "post_backward",
    }


@no_type_check
def _unshard(
    state: _FSDPState,
    handles: List[FlatParamHandle],
    unshard_stream: torch.cuda.Stream,
    pre_unshard_stream: torch.cuda.Stream,
) -> None:
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: Each handle's ``FlatParameter`` 's data is the padded
    unsharded flat parameter on the compute device.
    """
    if not handles:
        return
    any_ran_pre_unshard = False
    with state._device_handle.stream(pre_unshard_stream):
        for handle in handles:
            ran_pre_unshard = handle.pre_unshard()
            any_ran_pre_unshard = any_ran_pre_unshard or ran_pre_unshard
    if any_ran_pre_unshard:
        unshard_stream.wait_stream(pre_unshard_stream)
    if state.limit_all_gathers:
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            event.synchronize()
    with state._device_handle.stream(unshard_stream):
        for handle in handles:
            handle.unshard()
            handle.post_unshard()


@no_type_check
def _reshard(
    state: _FSDPState,
    handles: List[FlatParamHandle],
    free_unsharded_flat_params: List[bool],
):
    """
    Reshards the handles in ``handles``. ``free_unsharded_flat_params`` should
    have the same length as ``handles``, and each element should give whether
    the corresponding handle should free its padded unsharded flat parameter.
    """
    if not handles:
        return
    _p_assert(
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
            free_event = state._device_handle.Event()
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
    state: _FSDPState,
    handles: List[FlatParamHandle],
    unshard_fn: Callable,
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Runs the pre-forward logic. This includes an opportunity to unshard
    currently sharded parameters such as those for the current forward and
    registering post-backward hooks for these current parameters. This function
    also converts forward ``args`` and ``kwargs`` to the given precision.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        unshard_fn (Optional[Callable]): A callable to unshard any currently
            sharded parameters or ``None`` to not do any unsharding.
        module (nn.Module): Module whose forward this method runs right before;
            expected by the hook signature.
        args (Tuple[Any, ...]): Module forward ``args``.
        kwargs (Dict[str, Any]): Module forward ``kwargs``.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._pre_forward"):
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

        should_cast_forward_inputs = len(state._handles) > 0 and all(
            not handle._force_full_precision for handle in state._handles
        )

        if should_cast_forward_inputs and state.mixed_precision.cast_forward_inputs:
            # Recursively convert args and kwargs to specified precision.
            input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
            args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
        return args, kwargs


@no_type_check
def _pre_forward_unshard(
    state: _FSDPState,
    handles: List[FlatParamHandle],
) -> None:
    """Unshards parameters in the pre-forward."""
    if not handles:
        return
    handles_key = tuple(handles)
    # If the handles have been prefetched, then there is no need to call
    # `_unshard()` again
    if not state._handles_prefetched.get(handles_key, False):
        _unshard(
            state, handles, state._streams["unshard"], state._streams["pre_unshard"]
        )
    state._needs_pre_forward_unshard[handles_key] = False
    state._device_handle.current_stream().wait_stream(state._streams["unshard"])
    _prefetch_handles(state, handles_key, _PrefetchMode.FORWARD)


@no_type_check
def _post_forward(
    state: _FSDPState,
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
        module (nn.Module): Module whose forward just ran, which should be a
            fully sharded module (see [Note: Fully Sharded Module]); expected
            by the hook signature.
        input (Any): Unused; expected by the hook signature.
        output (Any): Forward pass output; pre-backward hooks are registered on
            the tensors that require gradients in this output.

    Postcondition: Each ``FlatParameter`` 's data points to the sharded flat
    parameter.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._post_forward"):
        state._exec_order_data.record_post_forward(handles)
        if reshard_fn is not None:
            reshard_fn()
        # Register pre-backward hooks to unshard the flat parameters for the
        # gradient computation (if needed)
        output = _register_pre_backward_hooks(state, module, output, handles)
        state.training_state = TrainingState.IDLE
        for handle in handles:
            handle._training_state = HandleTrainingState.IDLE
        return output


@no_type_check
def _post_forward_reshard(
    state: _FSDPState,
    handles: List[FlatParamHandle],
) -> None:
    """Reshards parameters in the post-forward."""
    if not handles:
        return
    # Do not free the root's parameters in the post-forward for `FULL_SHARD`
    # with the intention that they are immediately used for backward
    # computation (though this may not be true)
    free_unsharded_flat_params = [
        not state._is_root
        and handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
        for handle in handles
    ]
    _reshard(state, handles, free_unsharded_flat_params)


@no_type_check
def _root_pre_forward(
    state: _FSDPState,
    module: nn.Module,
    args,
    kwargs,
) -> None:
    """
    Runs pre-forward logic specific to the root FSDP instance, which should run
    before any individual module's pre-forward. This starts with an attempt at
    lazy initialization (which only runs non-vacuously once). Otherwise, if
    this is called on a non-root FSDP instance, then it returns directly.

    Args:
        module (nn.Module): Module for which this logic tries to run. It may or
            may not be the root. If not, then this method does not do anything.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._root_pre_forward"):
        _lazy_init(state, module)
        _p_assert(state._is_root is not None, "Expects a root FSDP to have been set")
        if not state._is_root:
            # Always cast forward inputs in the root of this local FSDP unit for mixed
            # precision, as this is where mixed precision could be configed.
            # This is more useful for auto wrapping that is recommended in composable path.
            # For manual wrapping, cast forward inputs on each local FSDP unit root will
            # increase some overhead, so not turned on for model wrapper path right now where
            # manual wrapping is more broadly used.
            if _is_composable(state):
                return _root_cast_forward_input(state, args, kwargs)
            return args, kwargs

        # We cast buffers back to full precision if we're forcing full precision. Disjointly, we check if buffers
        # are in full precision and if we should cast them back to lower precision, which happens when
        # exiting eval() mode.
        should_cast_buffers_to_full_prec = any(
            handle._force_full_precision for handle in state._handles
        )

        if should_cast_buffers_to_full_prec:
            _cast_buffers_to_dtype_and_device(
                buffers=dict(module.named_buffers()).values(),
                buffer_dtypes=list(state._buffer_name_to_orig_dtype.values()),
                device=state.compute_device,
            )
            # This flag is only set when we cast buffers to full precision, to avoid the
            # CPU overhead that can stem from retrieving all buffers and their types in the
            # following else branch.
            state._needs_buffer_dtype_restore_check = True
        elif getattr(state, "_needs_buffer_dtype_restore_check", False):
            # Check if buffers are in full precision and we need to cast them
            # back down.
            (
                buffers,
                buffer_dtypes_for_computation,
            ) = _get_buffers_and_dtypes_for_computation(state, module)
            if len(buffers) > 0 and len(buffer_dtypes_for_computation) > 0:
                if any(
                    buffer.dtype != buffer_dtype_for_computation
                    for buffer, buffer_dtype_for_computation in zip(
                        buffers, buffer_dtypes_for_computation
                    )
                ):
                    # Assume we have to cast everything if there is one mismatch
                    _cast_buffers_to_dtype_and_device(
                        buffers, buffer_dtypes_for_computation, state.compute_device
                    )
            # We don't have to check this again until we cast buffers to full precision again.
            state._needs_buffer_dtype_restore_check = False

        if state.forward_prefetch:
            handles_keys = []
            for fsdp_state in state._all_fsdp_states:
                # TODO: Forward prefetch assumes singleton handles key. For the
                # composable path, `_handles` may have more than one handle,
                # whereas for the wrapper path, it has at most one handle.
                handles_keys.extend((handle,) for handle in fsdp_state._handles)
            for handles_key in handles_keys:
                state._needs_pre_forward_unshard[handles_key] = True
        _wait_for_computation_stream(
            state._device_handle.current_stream(),
            state._streams["unshard"],
            state._streams["pre_unshard"],
        )
        _clear_grads_if_needed(state._all_handles)

        # Prepares the forward inputs by moving them to ``compute_device``
        # TODO: Do not use the side stream for tensor copies for now; investigate
        # the perf with/without it.
        with torch.profiler.record_function("FullyShardedDataParallel._to_kwargs"):
            args_tuple, kwargs_tuple = _to_kwargs(
                args, kwargs, state.compute_device, False
            )
        args = args_tuple[0]
        kwargs = kwargs_tuple[0]

        return _root_cast_forward_input(state, args, kwargs)


@no_type_check
def _root_cast_forward_input(state: _FSDPState, args, kwargs) -> Tuple[Any, Any]:
    should_cast_forward_inputs = (
        all(not handle._force_full_precision for handle in state._handles)
        and state.mixed_precision.cast_root_forward_inputs
    )

    if should_cast_forward_inputs:
        input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
        args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)

    return args, kwargs


def _cast_forward_inputs(
    input_dtype: Optional[torch.dtype],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Prepares the forward inputs by casting them to ``input_dtype`` if it is not ``None``.
    """
    # TODO: For mixed precision, cast to reduced-precision in a single `to()` call.
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
        if not torch.is_floating_point(x) or x.dtype == dtype:
            return x
        return x.to(dtype)

    return (_apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs))


@no_type_check
def _pre_backward_hook(
    state: _FSDPState,
    module: nn.Module,
    _handles: List[FlatParamHandle],
    *unused: Any,
) -> Any:
    """
    Prepares ``_handles`` 's ``FlatParameter`` s for gradient computation.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).
    """
    _handles_key = tuple(_handles)  # avoid shadowing `handles_key`
    # Only run the pre-backward hook once per group of handles involved in the
    # same module forward computation
    if _handles_key and state._ran_pre_backward_hook.get(_handles_key, False):
        return

    with torch.profiler.record_function("FullyShardedDataParallel._pre_backward_hook"):
        # Queue the post-backward callback once for the root FSDP instance to
        # attach it to the outermost backward graph task so that it is called
        # after all backward calls complete
        if state._is_root and not state._post_backward_callback_queued:
            _register_post_backward_final_callback(state, module)
            _clear_grads_if_needed(state._all_handles)
        elif _handles_key:
            allowed_states = [TrainingState.IDLE]
            if _is_composable(state):
                allowed_states.append(TrainingState.FORWARD_BACKWARD)
            _assert_in_training_states(state, allowed_states)
        state.training_state = TrainingState.FORWARD_BACKWARD
        # Queueing the post-backward callback is the only logic that is not
        # per-handle in the pre-backward hook, so we can return early here if
        # there are no handles.
        if not _handles_key:
            return
        for handle in _handles:
            handle._training_state = HandleTrainingState.BACKWARD_PRE

        if state._needs_pre_backward_unshard[_handles_key]:
            # If the handles have been prefetched, then there is no need to
            # call `_unshard()` again
            if not state._handles_prefetched.get(_handles_key, False):
                _unshard(
                    state,
                    _handles,
                    state._streams["unshard"],
                    state._streams["pre_unshard"],
                )
            state._device_handle.current_stream().wait_stream(state._streams["unshard"])

        # Set this to `False` to ensure that a mistargeted prefetch does not
        # actually unshard these handles
        state._needs_pre_backward_unshard[_handles_key] = False
        _prefetch_handles(state, _handles_key, _PrefetchMode.BACKWARD)
        for handle in _handles:
            handle.prepare_gradient_for_backward()
        state._ran_pre_backward_hook[_handles_key] = True


@no_type_check
@torch.no_grad()
def _post_backward_hook(
    state: _FSDPState,
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
    # Under TORCH_DISTRIBUTED_DEBUG=INFO, log the module names this hook fires for.
    # Below logging of module names this post-bwd hook fires for can help debug certain
    # cases where hooks don't fire, such as under certain activation checkpoint configs.
    if state._use_orig_params and handle._debug_level == dist.DebugLevel.INFO:
        param_to_fqn = state._exec_order_data.param_to_fqn
        handle_params = handle.flat_param._params  # only populated for use_orig_params
        param_fqns = [
            param
            for param_list in [param_to_fqn[p] for p in handle_params]
            for param in param_list
        ]
        log.warning("FSDP firing post-backward hooks for parameters %s", param_fqns)

    flat_param = handle.flat_param
    flat_param._post_backward_called = True
    with torch.autograd.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook"
    ):
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        # For multiple applications of reentrant AC across submodules sharing
        # the same `FlatParameter`, the post-backward hook may run multiple
        # times in one backward, in which case we permit the state to already
        # be in `BACKWARD_POST`.
        _p_assert(
            handle._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST),
            f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}",
        )
        handle._training_state = HandleTrainingState.BACKWARD_POST

        if flat_param.grad is None:
            return
        if flat_param.grad.requires_grad:
            raise RuntimeError("FSDP does not support gradients of gradients")

        free_unsharded_flat_param = _should_free_in_backward(state, handle)
        _reshard(state, [handle], [free_unsharded_flat_param])

        # TODO: Post-backward prefetching does not support the multiple handles
        # per module case since the post-backward hook runs per handle, not per
        # group of handles.
        handles_key = (handle,)
        _prefetch_handles(state, handles_key, _PrefetchMode.BACKWARD)

        if not state._sync_gradients:
            if handle._use_orig_params:
                handle._use_unsharded_grad_views()
            return

        # Wait for all ops in the current stream (e.g. gradient
        # computation) to finish before reduce-scattering the gradient
        state._streams["post_backward"].wait_stream(
            state._device_handle.current_stream()
        )

        with state._device_handle.stream(state._streams["post_backward"]):
            autograd_computed_grad = flat_param.grad.data
            if state._exec_order_data.is_first_iter:  # only check once
                _check_comm_hook(
                    state._communication_hook, state._communication_hook_state
                )
            if (
                not _low_precision_hook_enabled(state)
                and flat_param.grad.dtype != handle._reduce_dtype
                # If we are forcing full precision but communicating grads
                # (i.e. model.eval()), don't downcast gradient.
                and not handle._force_full_precision
            ):
                flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)

            if handle.uses_sharded_strategy:
                # We clear `.grad` to permit multiple backwards. This avoids a
                # race where the second backward pass computation precedes
                # ahead of the first backward pass reduction, which is possible
                # since the reduction is issued in a separate stream and is
                # async and would result in reducing the wrong gradient.
                unsharded_grad = flat_param.grad.data
                flat_param.grad = None
                chunks = list(unsharded_grad.chunk(state.world_size))
                numel_to_pad = (
                    state.world_size * chunks[0].numel() - unsharded_grad.numel()
                )
                padded_unsharded_grad = (
                    F.pad(unsharded_grad, [0, numel_to_pad])
                    if numel_to_pad > 0
                    else unsharded_grad
                )
                new_sharded_grad = torch.empty_like(chunks[0])  # padded
                state._communication_hook(
                    state._communication_hook_state,
                    padded_unsharded_grad,
                    new_sharded_grad,
                )
                if handle._sharding_strategy in (
                    HandleShardingStrategy.HYBRID_SHARD,
                    HandleShardingStrategy._HYBRID_SHARD_ZERO2,
                ):
                    default_hooks.allreduce_hook(
                        state=state._inter_node_state,
                        grad=new_sharded_grad,
                    )
                _cast_grad_to_param_dtype(state, new_sharded_grad, flat_param)
                # Save the sharded gradient in `_saved_grad_shard` to support
                # gradient accumulation -- for multiple backwards, the gradient
                # reductions may happen in arbitrary order
                accumulate_grad = hasattr(flat_param, "_saved_grad_shard")
                if accumulate_grad:
                    _check_grad_to_accumulate(
                        new_sharded_grad, flat_param._saved_grad_shard
                    )
                    flat_param._saved_grad_shard += new_sharded_grad
                else:
                    flat_param._saved_grad_shard = new_sharded_grad
                grad_to_offload = flat_param._saved_grad_shard
            else:
                state._communication_hook(
                    state._communication_hook_state, flat_param.grad
                )
                # For `NO_SHARD`, we can keep the low precision gradients by
                # simply omitting the cast altogether
                if not handle._keep_low_precision_grads:
                    _cast_grad_to_param_dtype(state, flat_param.grad, flat_param)
                grad_to_offload = flat_param.grad.data

            if handle._offload_params:
                # Offload the gradient to CPU to ensure parameters and
                # gradients are on the same device as required by the optimizer
                # TODO: Investigate why `NO_SHARD` breaks correctness when
                # using `non_blocking=True` here.
                non_blocking = handle.uses_sharded_strategy
                flat_param._cpu_grad.copy_(  # type: ignore[attr-defined]
                    grad_to_offload.detach(), non_blocking=non_blocking
                )  # synchronized in the post-backward callback
                # Since the gradient being offloaded may have been produced in
                # the computation stream and is being consumed here in the
                # post-backward stream, inform the caching allocator
                _no_dispatch_record_stream(
                    grad_to_offload.data,
                    state._streams["post_backward"],
                )

            # Since the unsharded gradient is produced in the computation
            # stream and consumed in the post-backward stream, inform the
            # caching allocator (before it goes out of scope)
            _no_dispatch_record_stream(
                autograd_computed_grad, state._streams["post_backward"]
            )

            if handle._use_orig_params:
                # Since the handle's `FlatParameter` completed its gradient
                # computation, we should reset the gradient noneness mask
                handle._reset_is_grad_none()
                # Delay using sharded gradient views until after the
                # reduce-scatter instead of immediately after resharding
                handle._use_sharded_grad_views()


@no_type_check
def _should_free_in_backward(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> bool:
    """
    Returns whether FSDP should free the unsharded flat parameter in the
    post-backward or not.
    """
    if not handle.uses_sharded_strategy:
        return False
    # If not syncing gradients, then we do not free for strategies that do not
    # reshard after forward as a *heuristic* to tradeoff higher memory for
    # higher throughput.
    return (
        state._sync_gradients
        or handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    )


@no_type_check
def _cast_grad_to_param_dtype(
    state: _FSDPState,
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
    if not _low_precision_hook_enabled(state) and sharded_grad.dtype != param.dtype:
        low_prec_grad_data = sharded_grad.data
        sharded_grad.data = sharded_grad.data.to(dtype=param.dtype)
        # Since for `NO_SHARD`, the gradient is produced in the computation
        # stream and consumed here in the post-backward stream, inform the
        # caching allocator; for the sharded strategies, the gradient is
        # produced in the post-backward stream, so this `record_stream()`
        # should be a no-op
        _no_dispatch_record_stream(
            low_prec_grad_data, state._device_handle.current_stream()
        )


def _check_comm_hook(
    comm_hook: Any,
    comm_hook_state: Any,
) -> None:
    _p_assert(comm_hook is not None, "Communication hook should not be `None`")
    _p_assert(
        comm_hook_state is not None, "Communication hook state should not be `None`"
    )


def _check_grad_to_accumulate(
    new_sharded_grad: torch.Tensor,
    accumulated_grad: torch.Tensor,
) -> None:
    _p_assert(
        accumulated_grad.shape == new_sharded_grad.shape,
        "Shape mismatch when accumulating gradients: "
        f"existing gradient shape={accumulated_grad.shape} "
        f"new gradient shape={new_sharded_grad.shape}",
    )
    _p_assert(
        accumulated_grad.device == new_sharded_grad.device,
        "Device mismatch when accumulating gradients: "
        f"existing gradient device={accumulated_grad.device} "
        f"new gradient device={new_sharded_grad.device}",
    )


@no_type_check
def _low_precision_hook_enabled(state: _FSDPState) -> bool:
    return state._communication_hook in LOW_PRECISION_HOOKS


@no_type_check
@torch.no_grad()
def _post_backward_final_callback(
    state: _FSDPState,
    module: nn.Module,
):
    """
    This waits for the post-backward to finish and performs some final cleanup.
    This runs at the end of the entire backward pass and should only be called
    on the root FSDP instance.
    """
    _p_assert(
        state._is_root,
        "The post-backward callback should only be called on the root FSDP instance",
    )
    root_state = state

    if root_state._sync_gradients:
        state._device_handle.current_stream().wait_stream(
            root_state._streams["post_backward"]
        )
        if root_state.cpu_offload.offload_params:
            # Wait for non-blocking GPU -> CPU sharded gradient copies from the
            # post-backward hooks to finish explicitly since CPU gradients do
            # not automatically synchronize with the GPU
            state._device_handle.current_stream().synchronize()
    root_state._exec_order_data.next_iter()

    for fsdp_state in state._all_fsdp_states:
        _catch_all_reshard(fsdp_state)
        _finalize_params(fsdp_state)
        fsdp_state._needs_pre_backward_unshard.clear()
        fsdp_state._ran_pre_backward_hook.clear()
        fsdp_state.training_state = TrainingState.IDLE
        for handle in fsdp_state._handles:
            handle._training_state = HandleTrainingState.IDLE
        fsdp_state._handles_prefetched.clear()
    # Reset for cases like one forward and multiple backwards
    root_state._post_backward_callback_queued = False


@no_type_check
def _catch_all_reshard(
    state: _FSDPState,
) -> None:
    """
    Reshards the parameters that may not have been resharded in the
    post-backward hook. This can happen when a module's output is used in the
    forward pass, meaning that its pre-backward hook runs (unsharding the
    parameter), but the post-backward hook does not run because the output was
    not jused in the loss computation corresponding to this backward pass.
    """
    # Wrap with a try-except to provide a more informative traceback if an
    # error is raised
    try:
        free_unsharded_flat_params: List[bool] = []
        handles_to_reshard: List[FlatParamHandle] = []
        for handle in state._handles:
            # TODO: This already-resharded check is brittle:
            # https://github.com/pytorch/pytorch/issues/83956
            already_resharded = (
                handle.flat_param.data_ptr()
                == handle.flat_param._local_shard.data_ptr()
            )
            if already_resharded:
                continue
            free_unsharded_flat_params.append(_should_free_in_backward(state, handle))
            handles_to_reshard.append(handle)
        if handles_to_reshard:
            _reshard(state, handles_to_reshard, free_unsharded_flat_params)
    except Exception as e:
        _p_assert(
            False,
            f"Got exception in the catch-all reshard for {state}: {str(e)}",
            raise_assertion_error=False,
        )
        raise e


@no_type_check
def _finalize_params(
    state: _FSDPState,
) -> None:
    """Finalizes the parameters before the next iteration."""
    for handle in state._handles:
        flat_param = handle.flat_param
        if flat_param.requires_grad:
            if hasattr(flat_param, "_post_backward_hook_state"):
                _p_assert(
                    len(flat_param._post_backward_hook_state) == 2,
                    f"Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}",
                )
                flat_param._post_backward_hook_state[1].remove()
                delattr(flat_param, "_post_backward_hook_state")
            if not state._sync_gradients:
                # Preserve the gradient accumulation state if not synchronizing
                # gradients: `.grad` remains the unsharded gradient  from prior
                # `no_sync()` iterations, and `_saved_grad_shard` remains the
                # sharded gradient from the last synchronized iteration
                continue
            handle.prepare_gradient_for_optim()
            _p_assert(
                hasattr(flat_param, "_post_backward_called"),
                "Expects `_post_backward_called` to be set on the `FlatParameter`",
            )
            flat_param._post_backward_called = False


@no_type_check
def _prefetch_handles(
    state: _FSDPState,
    current_handles_key: _HandlesKey,
    prefetch_mode: _PrefetchMode,
) -> None:
    """
    Prefetches the next handles if needed (without synchronization). An empty
    handles key cannot prefetch.
    """
    if not current_handles_key:
        return
    handles_to_prefetch = _get_handles_to_prefetch(state, current_handles_key)
    for handles_key in handles_to_prefetch:
        # Temporarily emulate the training state while calling `_unshard` to
        # ensure the correct `as_params` for `_use_unsharded_views()`
        prev_training_states: List[HandleTrainingState] = []
        for handle in handles_key:
            prev_training_states.append(handle._training_state)
            if prefetch_mode == _PrefetchMode.BACKWARD:
                handle._training_state = HandleTrainingState.BACKWARD_PRE
            elif prefetch_mode == _PrefetchMode.FORWARD:
                handle._training_state = HandleTrainingState.FORWARD
            else:
                raise ValueError(
                    f"Invalid prefetch mode on rank {state.rank}: {prefetch_mode}"
                )
        # Prefetch the next set of handles without synchronizing to allow
        # the sync to happen as late as possible to maximize overlap
        _unshard(
            state, handles_key, state._streams["unshard"], state._streams["pre_unshard"]
        )
        for handle, prev_training_state in zip(handles_key, prev_training_states):
            handle._training_state = prev_training_state
        state._handles_prefetched[handles_key] = True


@no_type_check
def _get_handles_to_prefetch(
    state: _FSDPState,
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
    _p_assert(
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
    _p_assert(len(handles_key) > 0, "Expects a non-empty handles key")
    training_states = {handle._training_state for handle in handles_key}
    _p_assert(
        len(training_states) == 1,
        f"Expects uniform training state but got {training_states}",
    )
    return next(iter(training_states))


@no_type_check
def _register_pre_forward_hooks(
    state: _FSDPState,
    modules: Iterable[nn.Module],
) -> None:
    """
    Registers pre-forward hooks on all modules in ``modules``. The pre-forward
    hooks are partially applied based on the current ``FlatParamHandle``
    construction, meaning that they must be re-registered if the construction
    changes.
    """
    for forward_handle in state._pre_forward_handles:
        forward_handle.remove()
    state._pre_forward_handles.clear()
    for module in modules:
        module_param_handles = state._fully_sharded_module_to_handles.get(module, [])
        if module_param_handles:
            unshard_fn = functools.partial(
                _pre_forward_unshard,
                state,
                module_param_handles,
            )
            hook = functools.partial(
                _pre_forward, state, module_param_handles, unshard_fn
            )
            state._pre_forward_handles.append(
                module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
            )


@no_type_check
def _register_post_forward_hooks(
    state: _FSDPState,
    modules: Iterable[nn.Module],
) -> None:
    """
    Registers post-forward hooks on all modules in ``modules``. The
    post-forward hooks are partially applied based on the current
    ``FlatParamHandle`` construction, meaning that they must be re-registered
    if the construction changes.
    """
    for forward_handle in state._post_forward_handles:
        forward_handle.remove()
    state._post_forward_handles.clear()
    for module in modules:
        module_param_handles = state._fully_sharded_module_to_handles.get(module, [])
        if module_param_handles:
            reshard_fn = functools.partial(
                _post_forward_reshard,
                state,
                module_param_handles,
            )
            hook = functools.partial(
                _post_forward,
                state,
                module_param_handles,
                reshard_fn,
            )
            state._post_forward_handles.append(module.register_forward_hook(hook))


@no_type_check
def _register_root_pre_forward_hook(
    state: _FSDPState,
    module: nn.Module,
):
    """
    Registers root pre-forward hook on ``module``, which should be the local
    FSDP root.

    NOTE: For the current composable FSDP design, we have each application of
    ``fully_shard()`` to a module to indicate that that module is the local
    FSDP root. We may remove this assumption in the future, in which case we
    will need to register this root pre-forward hook on any candidate module
    that may be the local FSDP root.
    """
    for forward_handle in state._root_pre_forward_handles:
        forward_handle.remove()
    state._root_pre_forward_handles.clear()
    hook = functools.partial(_root_pre_forward, state)
    state._root_pre_forward_handles.append(
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    )


@no_type_check
def _register_pre_backward_hooks(
    state: _FSDPState,
    module: nn.Module,
    outputs: Any,
    handles: List[FlatParamHandle],
) -> None:
    """
    Registers pre-backward hooks on the tensors that require gradients in the
    forward pass outputs ``outputs``, which were computed using the
    ``FlatParameter`` s of ``handles``.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).

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
        state._needs_pre_backward_unshard[handles_key] = False
        # Since these handles' `FlatParameter`s participated in a forward, we
        # conservatively assume that they will be used in the backward
        state._ran_pre_backward_hook[handles_key] = False

    def _register_hook(t: torch.Tensor) -> torch.Tensor:
        if t.requires_grad:
            t.register_hook(
                functools.partial(_pre_backward_hook, state, module, handles)
            )
            state._needs_pre_backward_unshard[handles_key] = True
        return t

    return _apply_to_tensors(_register_hook, outputs)


def _register_post_backward_hooks(
    state: _FSDPState,
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

    NOTE: We follow this heuristic to prefer the *first* forward to target the
    parameter mixed precision case, where there are *separate*
    ``AccumulateGrad`` objects across the different forwards. (Without
    parameter mixed precision, the ``AccumulateGrad`` objects are the same.) If
    we instead prefer the *last* forward, then the hook runs early.
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
        _p_assert(
            temp_flat_param.grad_fn is not None,
            "The `grad_fn` is needed to access the `AccumulateGrad` and "
            "register the post-backward hook",
        )
        acc_grad = temp_flat_param.grad_fn.next_functions[0][0]  # type: ignore[union-attr]
        assert acc_grad is not None
        hook_handle = acc_grad.register_hook(
            functools.partial(_post_backward_hook, state, handle)
        )
        flat_param._post_backward_hook_state = (acc_grad, hook_handle)  # type: ignore[attr-defined]


@no_type_check
def _register_post_backward_final_callback(
    state: _FSDPState, module: nn.Module
) -> None:
    """
    Registers the post-backward final callback that runs at the end of the
    backward pass. This should be called from the root FSDP instance at the
    beginning of the pre-backward.
    """
    _p_assert(
        state._is_root,
        "Only the root FSDP instance should register the post-backward callback",
    )
    if state._post_backward_callback_queued:
        return
    _assert_in_training_states(state, [TrainingState.IDLE])
    state._post_backward_callback_queued = True
    Variable._execution_engine.queue_callback(
        functools.partial(_post_backward_final_callback, state, module)
    )


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


@no_type_check
def _get_buffers_and_dtypes_for_computation(
    state: _FSDPState,
    root_module: nn.Module,
) -> Tuple[List[torch.Tensor], List[Optional[torch.dtype]]]:
    """
    Returns all buffers in the module tree rooted at ``root_module`` and a
    corresponding list of the buffer dtypes for computation. Each buffer dtype
    is either ``None`` if buffer mixed precision is not enabled or the buffer
    low precision dtype otherwise.
    """
    _p_assert(state._is_root, "Expects the root to cast buffers")
    buffers: List[torch.Tensor] = []
    buffer_dtypes: List[Optional[torch.dtype]] = []
    if _is_composable(state):
        buffers = [
            buffer for module in root_module.modules() for buffer in module.buffers()
        ]
        buffer_dtypes = [
            state.mixed_precision.buffer_dtype for _ in range(len(buffers))
        ]
    else:
        visited_buffers = set()
        # Traverse the FSDP instances bottom-up so that we prefer the owning
        # FSDP instance's mixed precision setting for each buffer
        for fsdp_module in reversed(traversal_utils._get_fsdp_states(root_module)):
            for buffer in fsdp_module.buffers():
                if buffer in visited_buffers:
                    continue
                visited_buffers.add(buffer)
                buffers.append(buffer)
                buffer_dtypes.append(fsdp_module.mixed_precision.buffer_dtype)
    assert len(buffers) == len(buffer_dtypes), f"{len(buffers)} {len(buffer_dtypes)}"
    return buffers, buffer_dtypes


@no_type_check
def _get_orig_buffer_dtypes(
    state: _FSDPState,
    buffer_names: List[str],
) -> List[torch.dtype]:
    """
    Returns the original buffer types of the given buffer names.
    """
    buffer_dtypes: List[torch.dtype] = []
    for buffer_name in buffer_names:
        _p_assert(
            buffer_name in state._buffer_name_to_orig_dtype,
            f"{buffer_name} is missing from pre-computed dict on rank "
            f"{state.rank}, which only has keys "
            f"{state._buffer_name_to_orig_dtype.keys()}",
        )
        buffer_dtypes.append(state._buffer_name_to_orig_dtype[buffer_name])
    return buffer_dtypes


def _cast_buffers_to_dtype_and_device(
    buffers: List[torch.Tensor],
    buffer_dtypes: List[Optional[torch.dtype]],
    device: torch.device,
) -> None:
    """
    Casts ``buffers`` to the dtypes given by ``buffer_dtypes`` and moves them
    to ``device``. If an element in ``buffer_dtypes`` is ``None``, then the
    corresponding buffer is only moved to ``device``.
    """
    _p_assert(
        buffer_dtypes is None or len(buffers) == len(buffer_dtypes),
        f"Expects `buffers` and `buffer_dtypes` to have the same length if "
        f"`buffer_dtypes` is specified but got {len(buffers)} and "
        f"{len(buffer_dtypes)}",
    )
    for buffer, buffer_dtype in zip(buffers, buffer_dtypes):
        if not torch.is_floating_point(buffer) or buffer_dtype is None:
            buffer.data = buffer.to(device=device)
        else:
            buffer.data = buffer.to(device=device, dtype=buffer_dtype)
