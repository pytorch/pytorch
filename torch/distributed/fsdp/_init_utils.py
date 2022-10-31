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
    Union,
)

import torch
import torch.distributed as dist
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._common_utils import (
    _apply_to_modules,
    _get_param_to_unflat_param_names,
    _is_fsdp_flattened,
    _State,
    clean_tensor_name,
    TrainingState,
)
from torch.distributed.fsdp._exec_order_utils import _ExecOrderData
from torch.distributed.fsdp._limiter_utils import _FreeEventQueue
from torch.distributed.fsdp._wrap_utils import _get_params_per_wrapped_module
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
    state: _State,
    process_group: Optional[dist.ProcessGroup],
) -> _State:
    state.process_group = process_group or _get_default_group()
    state.rank = state.process_group.rank()
    state.world_size = state.process_group.size()
    return state


@no_type_check
def _init_ignored_module_states(
    state: _State,
    module: nn.Module,
    ignored_modules: Optional[Iterable[torch.nn.Module]],
) -> _State:
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
def _init_module_and_device_state(
    state: _State,
    module: nn.Module,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    sync_module_states: bool,
) -> _State:
    state._buffer_names = _get_buffer_names(module)
    # Save a mapping from fully prefixed buffer name to its original dtype
    # since when buffer mixed precision is enabled, buffers are restored to
    # their original dtype for model checkpointing
    _buffer_name_to_orig_dtype: Dict[str, torch.dtype] = {}
    state._buffer_name_to_orig_dtype = _buffer_name_to_orig_dtype
    _check_single_device_module(module, state._ignored_params)
    device_from_device_id = _get_device_from_device_id(device_id, state.rank)
    _materialize_module(
        module,
        param_init_fn,
        state._ignored_params,
        device_from_device_id,
        lambda _: True,
    )
    # TODO: We need to skip this for functional-like to avoid moving the entire
    # unsharded module onto GPU before any sharding.
    _move_module_to_device(module, state._ignored_params, device_from_device_id)
    state.compute_device = _get_compute_device(
        module,
        state._ignored_params,
        device_from_device_id,
        state.rank,
    )
    state._managed_params = list(_get_orig_params(module, state._ignored_params))
    if sync_module_states:
        _sync_module_states(module, state._managed_params, state.process_group)
    return state


@no_type_check
def _init_core_state(
    state: _State,
    sharding_strategy: Optional[ShardingStrategy],
    mixed_precision: Optional[MixedPrecision],
    cpu_offload: Optional[CPUOffload],
    limit_all_gathers: bool,
    use_orig_params: bool,
    backward_prefetch_limit: int,
    forward_prefetch_limit: int,
) -> _State:
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
    # Invariant: `self.params` contains exactly the `FlatParameter`s of the
    # handles in `self._handles`
    _handles: List[FlatParamHandle] = []
    state._handles = _handles
    params: List[FlatParameter] = []
    state.params = params
    return state


@no_type_check
def _init_runtime_state(
    state: _State,
) -> _State:
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
def _init_prefetching_state(state: _State) -> _State:
    state.backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    state.forward_prefetch = False
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


def _init_state_dict_state(state: _State) -> _State:
    # TODO: after rebase
    return state


def _init_param_handles_from_module(
    state: _State,
    root_module: nn.Module,
    auto_wrap_policy: Callable,
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
) -> _State:
    params_per_wrapped_module = _get_params_per_wrapped_module(
        root_module,
        auto_wrap_policy,
        ignored_modules,
        ignored_params,
    )
    for params in params_per_wrapped_module:
        _init_param_handle_from_params(state, params, root_module)
    return state


@no_type_check
def _init_param_handle_from_params(
    state: _State,
    params: List[nn.Parameter],
    root_module: nn.Module,
):
    if len(params) == 0:
        return
    # TODO: Move module to GPU if needed (for non-wrapper code path) -- we need
    # to fuse this method with `_init_module_and_device_state()`
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
    assert handle.flat_param not in state.params
    assert handle not in state._handles
    state.params.append(handle.flat_param)
    state._handles.append(handle)
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
    param_to_unflat_param_names = _get_param_to_unflat_param_names(
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

    def module_fn(module: nn.Module, prefix: str, buffer_names: Set[str]):
        for buffer_name, _ in module.named_buffers(recurse=False):
            # Clean module wrapper prefixes in case of nested wrapping
            prefixed_buffer_name = clean_tensor_name(prefix + buffer_name)
            buffer_names.add(prefixed_buffer_name)

    def return_fn(buffer_names: Set[str], *args):
        return buffer_names

    buffer_names: Set[str] = set()
    return _apply_to_modules(
        root_module,
        module_fn,
        return_fn,
        buffer_names,
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
) -> None:
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
    """
    is_meta_module = any(p.is_meta for p in _get_orig_params(module, ignored_params))
    is_torchdistX_deferred_init = (
        not is_meta_module
        and _TORCHDISTX_AVAIL
        and any(fake.is_fake(p) for p in _get_orig_params(module, ignored_params))
    )
    if (is_meta_module or is_torchdistX_deferred_init) and param_init_fn is not None:
        if not callable(param_init_fn):
            raise ValueError(
                f"Expected {param_init_fn} to be callable but got {type(param_init_fn)}"
            )
        param_init_fn(module)
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
    elif is_torchdistX_deferred_init:
        # Run default torchdistX initialization
        deferred_init.materialize_module(module, check_fn=deferred_init_check_fn)


def _move_module_to_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
):
    """
    Moves ``module`` depending on ``device_from_device_id`` and its current
    device. This includes moving ignored modules' parameters.

    - If ``device_from_device_id`` is not ``None``, then this moves
    ``module`` to the device.
    - If ``device_from_device_id`` is ``None``, then this does not move
    ``module`` but warns the user if it is on CPU.

    Precondition: ``_check_single_device_module()``.
    """
    cpu_device = torch.device("cpu")
    param = next(_get_orig_params(module, ignored_params), None)
    if param is None:
        return  # no original parameters to manage
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
                    with torch.no_grad():
                        for handle in submodule._handles:
                            handle.flat_param_to(torch.device("cpu"))
    elif param.device == cpu_device:
        warnings.warn(
            "Module is put on CPU and will thus have flattening and sharding"
            " run on CPU, which is less efficient than on GPU. We recommend passing in "
            "`device_id` argument which will enable FSDP to put module on GPU device,"
            " module must also be on GPU device to work with `sync_module_states=True` flag"
            " which requires GPU communication."
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


def _sync_module_states(
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
    if params and any(param.device == torch.device("cpu") for param in params):
        raise ValueError(
            "Module has CPU parameters, but sync_module_states=True is specified."
            "This only works for GPU module, please specify `device_id` argument or move"
            " module to GPU before init."
        )
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
