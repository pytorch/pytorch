import collections
import warnings
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.distributed as dist
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_param_to_fqns,
    _is_fsdp_flattened,
    clean_tensor_name,
    TrainingState,
)
from torch.distributed.fsdp._exec_order_utils import _ExecOrderData
from torch.distributed.fsdp._limiter_utils import _FreeEventQueue
from torch.distributed.fsdp._wrap_utils import _get_submodule_to_states
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.flat_param import (
    _HandlesKey,
    FlatParameter,
    FlatParamHandle,
    HandleConfig,
    HandleShardingStrategy,
)
from torch.distributed.utils import _sync_params_and_buffers
from torch.utils.hooks import RemovableHandle

_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init, fake  # type: ignore[import]
except ImportError:
    _TORCHDISTX_AVAIL = False

PARAM_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)
FSDP_SYNCED = "_fsdp_synced"

# TODO (awgu): Refactor this later
SHARDING_STRATEGY_MAP = {
    ShardingStrategy.NO_SHARD: HandleShardingStrategy.NO_SHARD,
    ShardingStrategy.FULL_SHARD: HandleShardingStrategy.FULL_SHARD,
    ShardingStrategy.SHARD_GRAD_OP: HandleShardingStrategy.SHARD_GRAD_OP,
}


# NOTE: Since non-self attributes cannot be type annotated, several attributes
# on `state` are defined first as local variables before being assigned.


@no_type_check
def _init_process_group_state(
    state: _FSDPState,
    process_group: Optional[dist.ProcessGroup],
) -> _FSDPState:
    state.process_group = process_group or _get_default_group()
    state.rank = state.process_group.rank()
    state.world_size = state.process_group.size()
    return state


@no_type_check
def _init_ignored_module_states(
    state: _FSDPState,
    module: nn.Module,
    ignored_modules: Optional[Iterable[torch.nn.Module]],
) -> _FSDPState:
    state._ignored_modules = _get_ignored_modules(module, ignored_modules)
    state._ignored_params, state._ignored_param_names = _get_ignored_params(
        module,
        state._ignored_modules,
    )
    # TODO: FSDP's contract for buffers is not well-defined. They are
    # implicitly ignored for most functionality since they are not sharded;
    # however, FSDP still imposes some semantics on buffers (e.g. buffer mixed
    # precision). We should formalize this contract and decide if we need to
    # compute and store `_ignored_buffers`.
    return state


@no_type_check
def _init_buffer_state(
    state: _FSDPState,
    module: nn.Module,
) -> _FSDPState:
    state._buffer_names = _get_buffer_names(module)
    # Save a mapping from clean fully-qualified buffer name (starting from
    # `module`) to its original dtype for restoring that dtype during model
    # checkpointing when buffer mixed precision is enabled. The names should
    # be clean since the casting happens in a `summon_full_params()` context.
    _buffer_name_to_orig_dtype: Dict[str, torch.dtype] = {}
    for buffer_name, buffer in module.named_buffers():
        buffer_name = clean_tensor_name(buffer_name)
        _buffer_name_to_orig_dtype[buffer_name] = buffer.dtype
    state._buffer_name_to_orig_dtype = _buffer_name_to_orig_dtype
    return state


@no_type_check
def _init_core_state(
    state: _FSDPState,
    sharding_strategy: Optional[ShardingStrategy],
    mixed_precision: Optional[MixedPrecision],
    cpu_offload: Optional[CPUOffload],
    limit_all_gathers: bool,
    use_orig_params: bool,
    backward_prefetch_limit: int,
    forward_prefetch_limit: int,
) -> _FSDPState:
    # We clamp the strategy to `NO_SHARD` for world size of 1 since they are
    # currently functionally equivalent. This may change if/when we integrate
    # FSDP with MoE.
    if state.world_size == 1:
        sharding_strategy = ShardingStrategy.NO_SHARD
    state.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
    state.mixed_precision = mixed_precision or MixedPrecision()
    state.cpu_offload = cpu_offload or CPUOffload()
    state.limit_all_gathers = limit_all_gathers
    state._use_orig_params = use_orig_params
    state.training_state = TrainingState.IDLE
    state._is_root = None
    _streams: Dict[str, torch.cuda.Stream] = {}
    state._streams = _streams
    state._free_event_queue = _FreeEventQueue()
    state._debug_level = dist.get_debug_level()
    state._exec_order_data = _ExecOrderData(
        state._debug_level,
        backward_prefetch_limit,
        forward_prefetch_limit,
    )
    _module_to_handles: Dict[
        nn.Module, List[FlatParamHandle]
    ] = collections.defaultdict(list)
    state._module_to_handles = _module_to_handles
    # Invariant: `state.params` contains exactly the `FlatParameter`s of the
    # handles in `state._handles`
    _handles: List[FlatParamHandle] = []
    state._handles = _handles
    params: List[FlatParameter] = []
    state.params = params
    return state


@no_type_check
def _init_runtime_state(
    state: _FSDPState,
) -> _FSDPState:
    _root_pre_forward_handles: List[RemovableHandle] = []
    state._root_pre_forward_handles = _root_pre_forward_handles
    _pre_forward_handles: List[RemovableHandle] = []
    state._pre_forward_handles = _pre_forward_handles
    _post_forward_handles: List[RemovableHandle] = []
    state._post_forward_handles = _post_forward_handles
    _module_to_handles: Dict[
        nn.Module, List[FlatParamHandle]
    ] = collections.defaultdict(list)
    state._module_to_handles = _module_to_handles
    state._sync_gradients = True
    state._communication_hook = _get_default_comm_hook(state.sharding_strategy)
    state._communication_hook_state = _get_default_comm_hook_state(state.process_group)
    state._hook_registered = False
    # Used to prevent running the pre-backward hook multiple times
    _ran_pre_backward_hook: Dict[_HandlesKey, bool] = {}
    state._ran_pre_backward_hook = _ran_pre_backward_hook
    return state


@no_type_check
def _init_prefetching_state(
    state: _FSDPState,
    backward_prefetch: BackwardPrefetch,
    forward_prefetch: bool,
) -> _FSDPState:
    state.backward_prefetch = backward_prefetch
    state.forward_prefetch = forward_prefetch
    _handles_prefetched: Dict[_HandlesKey, bool] = {}
    state._handles_prefetched = _handles_prefetched
    # Used for guarding against mistargeted backward prefetches
    _needs_pre_backward_unshard: Dict[_HandlesKey, bool] = {}
    state._needs_pre_backward_unshard = _needs_pre_backward_unshard
    # Used for guarding against mistargeted forward prefetches
    _needs_pre_forward_unshard: Dict[_HandlesKey, bool] = {}
    state._needs_pre_forward_unshard = _needs_pre_forward_unshard
    # The data structures use tuples of handles to generalize over the case
    # where a module's forward involves multiple handles.
    return state


def _init_state_dict_state(state: _FSDPState) -> _FSDPState:
    # TODO: after rebase
    return state


@no_type_check
def _init_param_handle_from_module(
    state: _FSDPState,
    root_module: nn.Module,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    sync_module_states: bool,
    module_wrapper_cls: Type,
) -> _FSDPState:
    """
    Initializes a ``FlatParamHandle`` from a module ``root_module``. This is
    the module wrapper code path.
    """
    _check_single_device_module(root_module, state._ignored_params)
    device_from_device_id = _get_device_from_device_id(device_id, state.rank)
    _materialize_module(
        root_module,
        param_init_fn,
        state._ignored_params,
        device_from_device_id,
        lambda k: not isinstance(k, module_wrapper_cls),
    )
    # TODO: Investigate refactoring `_move_module_to_device()` to
    # `_move_states_to_device()` to avoid the `device_id` + CPU offload hack
    _move_module_to_device(root_module, state._ignored_params, device_from_device_id)
    state.compute_device = _get_compute_device(
        root_module,
        state._ignored_params,
        device_from_device_id,
        state.rank,
    )
    managed_params = list(_get_orig_params(root_module, state._ignored_params))
    if sync_module_states:
        _sync_module_params_and_buffers(
            root_module, managed_params, state.process_group
        )
    _init_param_handle_from_params(state, managed_params, root_module)
    return state


@no_type_check
def _init_param_handles_from_module(
    state: _FSDPState,
    root_module: nn.Module,
    auto_wrap_policy: Callable,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    sync_module_states: bool,
) -> _FSDPState:
    """
    Initializes all ``FlatParamHandle`` s from a module ``root_module``. This
    is the non-module-wrapper code path.
    """
    submodule_to_states = _get_submodule_to_states(
        root_module,
        auto_wrap_policy,
        state._ignored_modules,
        state._ignored_params,
    )
    _check_single_device_module(root_module, state._ignored_params)
    device_from_device_id = _get_device_from_device_id(device_id, state.rank)
    # Initialize and shard `FlatParamHandle`s one by one following bottom-up
    # order (hence the `reversed`) to avoid increasing peak GPU memory usage
    materialized_module = False
    for submodule, (params, buffers, param_names, buffer_names) in reversed(
        submodule_to_states.items()
    ):
        materialized_module |= _materialize_module(
            submodule,
            param_init_fn,
            state._ignored_params,
            device_from_device_id,
            lambda _: True,
        )
        if materialized_module:
            # Materializing from meta device can change the parameter/buffer
            # variables, so reacquire references
            params = [submodule.get_parameter(param_name) for param_name in param_names]
            buffers = [
                submodule.get_buffer(buffer_name) for buffer_name in buffer_names
            ]
        _move_states_to_device(params, buffers, device_from_device_id)
        if not hasattr(state, "compute_device"):  # only need to set once
            state.compute_device = _get_compute_device(
                submodule,
                state._ignored_params,
                device_from_device_id,
                state.rank,
            )
        if sync_module_states:
            _sync_module_states(params, buffers, state.process_group)
        # Pass `root_module` to have internal FQN metadata prefix starting from
        # it instead of `submodule`
        _init_param_handle_from_params(state, params, root_module)
    # Reverse to preserve top-down order like `_fsdp_handles()`
    state._handles.reverse()
    return state


@no_type_check
def _init_param_handle_from_params(
    state: _FSDPState,
    params: List[nn.Parameter],
    root_module: nn.Module,
):
    if len(params) == 0:
        return
    handle_config = HandleConfig(
        SHARDING_STRATEGY_MAP[state.sharding_strategy],
        state.cpu_offload.offload_params,
        state.mixed_precision.param_dtype,
        state.mixed_precision.reduce_dtype,
        state.mixed_precision.keep_low_precision_grads,
    )
    handle = FlatParamHandle(
        params,
        root_module,
        state.compute_device,
        handle_config,
        state.process_group,
        state._use_orig_params,
    )
    # TODO: Can simplify call `shard()` in the `FlatParamHandle` ctor
    handle.shard()
    assert handle not in state._handles
    state.params.append(handle.flat_param)
    state._handles.append(handle)
    for module in handle.flat_param._modules:
        state._module_to_handles[module].append(handle)
    cpu_device = torch.device("cpu")
    if state.cpu_offload.offload_params and handle.flat_param.device != cpu_device:
        handle.flat_param_to(cpu_device)


def _get_ignored_modules(
    root_module: nn.Module,
    _ignored_modules: Optional[Iterable[torch.nn.Module]],
) -> Set[nn.Module]:
    """
    Checks that ``_ignored_modules`` is an iterable of ``nn.Module`` s without
    any FSDP instances, and returns the modules contained in their module
    subtrees as a :class:`set`. Nested FSDP instances are excluded, but their
    already-computed ignored modules are included.
    """
    if _ignored_modules is None:
        return set()
    msg_prefix = "`ignored_modules` should be an iterable of `torch.nn.Module`s "
    try:
        ignored_root_modules = set(_ignored_modules)
    except TypeError:
        raise TypeError(msg_prefix + f"but got {type(_ignored_modules)}")
    for module in ignored_root_modules:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(msg_prefix + f"but got an iterable with {type(module)}")
        if isinstance(module, fsdp_file.FullyShardedDataParallel):
            raise ValueError("`ignored_modules` should not include FSDP modules")
    # Include child modules and exclude nested FSDP modules themselves
    ignored_modules = set(
        child
        for module in ignored_root_modules
        for child in module.modules()
        if not isinstance(child, fsdp_file.FullyShardedDataParallel)
    )
    if root_module in ignored_modules:
        warnings.warn(
            "Trying to ignore the top-level module passed into the FSDP "
            "constructor itself will result in all parameters being "
            f"ignored and is not well-supported: {module}"
        )
    # Include nested FSDP modules' ignored modules
    for submodule in root_module.modules():
        if isinstance(submodule, fsdp_file.FullyShardedDataParallel):
            assert hasattr(submodule, "_ignored_modules")
            ignored_modules.update(submodule._ignored_modules)
    return ignored_modules


def _get_ignored_params(
    root_module: torch.nn.Module,
    ignored_modules: Set[torch.nn.Module],
) -> Tuple[Set[torch.nn.Parameter], Set[str]]:
    """
    Returns the parameters of the modules in ``ignored_modules``,
    excluding any :class:`FlatParameter` s, and their fully prefixed names,
    both as :class:`set` s.
    """
    ignored_params = set(
        p for m in ignored_modules for p in m.parameters() if not _is_fsdp_flattened(p)
    )
    # Conservatively include all shared parameters' names
    param_to_unflat_param_names = _get_param_to_fqns(
        root_module,
        dedup_shared_params=False,
    )
    ignored_param_names = set()
    for param in ignored_params:
        unflat_param_names = param_to_unflat_param_names[param]
        clean_names = []
        for k in unflat_param_names:
            # Clean any module wrapper prefixes in case of nested wrapping
            clean_names.append(clean_tensor_name(k))
        ignored_param_names.update(clean_names)
    return ignored_params, ignored_param_names


def _get_buffer_names(root_module: nn.Module) -> Set[str]:
    """
    Returns the fully prefixed names of all buffers in the module hierarchy
    rooted at ``root_module`` as a class:`set`.
    """
    return set(
        clean_tensor_name(buffer_name) for buffer_name, _ in root_module.named_buffers()
    )


def _check_single_device_module(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> None:
    """
    Raises an error if ``module`` has original parameters on multiple devices,
    ignoring the parameters in ``ignored_params``. Thus, after this method, the
    module must be either fully on the CPU or fully on a non-CPU device.
    """
    devices = set(param.device for param in _get_orig_params(module, ignored_params))
    if len(devices) > 1:
        raise RuntimeError(
            f"FSDP only supports single device modules but got params on {devices}"
        )


def _get_device_from_device_id(
    device_id: Optional[Union[int, torch.device]],
    rank: int,
) -> Optional[torch.device]:
    """
    Processes ``device_id`` and returns either the corresponding device or
    ``None`` if ``device_id`` is ``None``.
    """
    if device_id is None:
        return None
    device = (
        device_id if isinstance(device_id, torch.device) else torch.device(device_id)
    )
    if device == torch.device("cuda"):
        warnings.warn(
            f"FSDP got the argument `device_id` {device_id} on rank "
            f"{rank}, which does not have an explicit index. "
            f"FSDP will use the current device {torch.cuda.current_device()}. "
            "If this is incorrect, please explicitly call `torch.cuda.set_device()` "
            "before FSDP initialization or pass in the explicit device "
            "index as the `device_id` argument."
        )
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _materialize_module(
    module: nn.Module,
    param_init_fn: Optional[Callable[[nn.Module], None]],
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
    deferred_init_check_fn: Callable,
) -> bool:
    """
    Materializes the wrapped module ``module`` in place if needed: either
    if the module has parameters that use meta device or are torchdistX
    fake tensors.

    This method uses ``param_init_fn`` to materialize the module if the
    function is not ``None`` and falls back to default behavior otherwise.
    For meta device, this moves the module to ``device_from_device_id`` if
    it is not ``None`` or the current device otherwise and calls
    ``reset_parameters()``, and for torchdistX fake tensors, this calls
    ``deferred_init.materialize_module()``.

    Returns:
        bool: ``True`` if ``module`` was materialized and ``False`` if this was
        a no-op.
    """
    managed_params = _get_orig_params(module, ignored_params)
    is_meta_module = any(param.is_meta for param in managed_params)
    is_torchdistX_deferred_init = (
        not is_meta_module
        and _TORCHDISTX_AVAIL
        and any(fake.is_fake(param) for param in managed_params)
    )
    if (is_meta_module or is_torchdistX_deferred_init) and param_init_fn is not None:
        if not callable(param_init_fn):
            raise ValueError(
                f"Expected {param_init_fn} to be callable but got {type(param_init_fn)}"
            )
        param_init_fn(module)
        return True
    elif is_meta_module:
        # Run default meta device initialization
        materialization_device = device_from_device_id or torch.device(
            torch.cuda.current_device()
        )
        module.to_empty(device=materialization_device)
        try:
            with torch.no_grad():
                module.reset_parameters()  # type: ignore[operator]
        except BaseException as e:
            warnings.warn(
                "Unable to call `reset_parameters()` for module on meta "
                f"device with error {str(e)}. Please ensure your "
                "module implements a `reset_parameters()` method."
            )
            raise e
        return True
    elif is_torchdistX_deferred_init:
        # Run default torchdistX initialization
        deferred_init.materialize_module(module, check_fn=deferred_init_check_fn)
        return True
    return False


def _move_module_to_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
) -> None:
    """
    Moves ``module`` depending on ``device_from_device_id`` and its current
    device. This includes moving ignored modules' parameters.

    - If ``device_from_device_id`` is not ``None``, then this moves
    ``module`` to the device.
    - If ``device_from_device_id`` is ``None``, then this does not move
    ``module`` but warns the user if it is on CPU.

    Precondition: ``_check_single_device_module()``.
    """
    param = next(_get_orig_params(module, ignored_params), None)
    if param is None:
        return  # no original parameters to manage
    cpu_device = torch.device("cpu")
    if device_from_device_id is not None:
        if param.device == cpu_device:
            # NOTE: This includes moving ignored modules' parameters.
            module = module.to(device_from_device_id)
            # TODO: This is a temporary fix to move already-constructed
            # `FlatParameter`s back to CPU if needed. This is needed to
            # make CPU offload work with `device_id`.
            for submodule in module.modules():
                if (
                    isinstance(submodule, fsdp_file.FullyShardedDataParallel)
                    and submodule.cpu_offload.offload_params
                ):
                    for handle in submodule._handles:
                        handle.flat_param_to(torch.device("cpu"))
    elif param.device == cpu_device:
        _warn_cpu_init()


def _move_states_to_device(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    device_from_device_id: Optional[torch.device],
) -> None:
    """
    Precondition: ``_check_single_device_module()``.
    """
    if len(params) == 0 and len(buffers) == 0:
        return
    if len(params) > 0:
        current_device = params[0].device
    elif len(buffers) > 0:
        current_device = buffers[0].device
    cpu_device = torch.device("cpu")
    if device_from_device_id is not None:
        # Move the parameters and buffers like the `.data` code path in
        # `nn.Module._apply()`, which underlies `nn.Module.to()`
        for param in params:
            with torch.no_grad():
                param.data = param.to(device_from_device_id)
                if param.grad is not None:
                    param.grad.data = param.grad.to(device_from_device_id)
        for buffer in buffers:
            buffer.data = buffer.to(device_from_device_id)
    elif current_device == cpu_device:
        _warn_cpu_init()


def _warn_cpu_init():
    warnings.warn(
        "The passed-in `module` is on CPU and will thus have FSDP's sharding "
        "initialization run on CPU, which may be slower than on GPU. We "
        "recommend passing in the `device_id` argument for FSDP to move "
        "`module` to GPU for the sharding initialization. `module` must also "
        "be on GPU device to work with the `sync_module_states=True` flag "
        "since that requires GPU communication."
    )


def _get_compute_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
    rank: int,
) -> torch.device:
    """
    Determines and returns this FSDP instance's compute device. If the module
    is already on a non-CPU device, then the compute device is that non-CPU
    device. If the module is on CPU, then the compute device is the current
    device.

    Since this method should be called after materializing the module, any
    non-CPU device should not be meta device. For now, the compute device is
    always a CUDA GPU device with its explicit index.

    Precondition: ``_check_single_device_module()`` and
    ``_move_module_to_device()``.
    """
    # If the module is on GPU already, then that GPU device has priority
    # over the current device
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device.type == "cuda":
        compute_device = param.device
    else:
        compute_device = torch.device("cuda", torch.cuda.current_device())
    if device_from_device_id is not None and compute_device != device_from_device_id:
        raise ValueError(
            f"Inconsistent compute device and `device_id` on rank {rank}: "
            f"{compute_device} vs {device_from_device_id}"
        )
    return compute_device


# TODO: See how to deprecate!
def _sync_module_params_and_buffers(
    module: nn.Module,
    params: List[nn.Parameter],
    process_group: dist.ProcessGroup,
) -> None:
    """
    Synchronizes module states (i.e. parameters ``params`` and all
    not-yet-synced buffers) by broadcasting from rank 0 to all ranks.

    Precondition: ``sync_module_states == True`` and ``self.process_group`` has
    been set.
    """
    _check_params_for_sync_module_states(params)
    module_states: List[torch.Tensor] = []
    # TODO (awgu): When exposing the original parameters, we need to also
    # use this attribute to prevent re-synchronizing parameters.
    for buffer in module.buffers():
        # Avoid re-synchronizing buffers in case of nested wrapping
        if not getattr(buffer, FSDP_SYNCED, False):
            setattr(buffer, FSDP_SYNCED, True)
            module_states.append(buffer.detach())
    module_states.extend(param.detach() for param in params)
    _sync_params_and_buffers(
        process_group,
        module_states,
        PARAM_BROADCAST_BUCKET_SIZE,
        src=0,
    )


def _sync_module_states(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    process_group: dist.ProcessGroup,
) -> None:
    _check_params_for_sync_module_states(params)
    # Assumes that each call to this method passes in disjoint `params` and
    # and `buffers` across calls, so there is no chance of re-synchronizing
    params_and_buffers = [param.detach() for param in params] + [
        buffer.detach() for buffer in buffers
    ]
    _sync_params_and_buffers(
        process_group,
        params_and_buffers,
        PARAM_BROADCAST_BUCKET_SIZE,
        src=0,
    )


def _check_params_for_sync_module_states(
    params: List[nn.Parameter],
) -> None:
    if params and any(param.device == torch.device("cpu") for param in params):
        raise ValueError(
            "The module has CPU parameters when `sync_module_states=True`, "
            "which only works when all parameters are on GPU. Please specify "
            "the `device_id` argument or move the module to GPU before passing "
            "into FSDP."
        )


def _get_orig_params(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> Iterator[nn.Parameter]:
    """
    Returns an iterator over the original parameters in ``module``, ignoring
    the parameters in ``ignored_params``, any ``FlatParameter`` s (which may be
    present due to nested FSDP wrapping), and any original parameters already
    flattened (only relevant when ``use_orig_params=True``).
    """
    param_gen = module.parameters()
    try:
        while True:
            param = next(param_gen)
            if param not in ignored_params and not _is_fsdp_flattened(param):
                yield param
    except StopIteration:
        pass


def _check_orig_params_flattened(
    fsdp_module,
    ignored_params: Set[nn.Parameter],
) -> None:
    """
    Checks that all original parameters have been flattened and hence made
    invisible to ``named_parameters()`` for the module hierarchy rooted at
    ``fsdp_module``. This should be called as a sanity check after flattening
    the wrapped module's parameters.
    """
    for param_name, param in fsdp_module.named_parameters():
        if param not in ignored_params and not _is_fsdp_flattened(param):
            raise RuntimeError(
                f"Found an unflattened parameter: {param_name}; "
                f"{param.size()} {param.__class__}"
            )


def _get_default_comm_hook(sharding_strategy: ShardingStrategy):
    return (
        default_hooks.allreduce_hook
        if sharding_strategy == ShardingStrategy.NO_SHARD
        else default_hooks.reduce_scatter_hook
    )


def _get_default_comm_hook_state(
    process_group: dist.ProcessGroup,
) -> default_hooks.DefaultState:
    return default_hooks.DefaultState(process_group=process_group)
