"""
This file includes private common utilities for FSDP.
"""

import traceback
import warnings
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
)

import torch
import torch.distributed as dist
import torch.distributed.fsdp.flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
)

from .api import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    OptimStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictType,
)

FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"
FSDP_PREFIX = FSDP_WRAPPED_MODULE + "."
FSDP_FLATTENED = "_fsdp_flattened"


class _FSDPDeviceHandle:
    """
    This is a simple abstraction for FSDP computing devices,
    which enables custom backends that implement CUDA-like
    semantics to be integrated with FSDP.
    """

    def __init__(self, device: torch.device, backend: Any = None):
        if backend is None:
            try:
                self.__backend = getattr(torch, device.type)
                self.__device = device
            except AttributeError:
                raise AttributeError(
                    f"Device '{device}' does not have a corresponding backend registered as 'torch.{device.type}'."
                )
        else:
            self.__backend = backend

    @classmethod
    def from_device(cls, device: torch.device) -> "_FSDPDeviceHandle":
        """
        Return an device handle corresponding to the device, and through this handle,
        operations with the same semantics as CUDA can be performed on the device.
        Just return torch.cuda if the device is cuda to make attribute-access faster.
        Custom backend must first register a module with the same name with {device.type} on torch.
        """
        if device.type == "cuda":
            return cast(_FSDPDeviceHandle, torch.cuda)
        return cls(device)

    def __getattr__(self, __name: str) -> Any:
        try:
            return getattr(self.__backend, __name)
        except AttributeError:
            raise AttributeError(
                f"Custom backend '{self.__device.type}' not implement 'torch.{self.__device.type}.{__name}'"
            )


class _UninitializedDeviceHandle(_FSDPDeviceHandle):
    def __init__(self):
        pass

    def __getattribute__(self, __name: str) -> Any:
        raise RuntimeError("Trying to use an uninitialized device handle.")


class _FSDPState(_State):
    def __init__(self) -> None:
        # TODO: Move all the attributes to this class to enable typing for
        # FSDP/fully_shard.
        self._ignored_modules: Set[nn.Module] = set()
        self._ignored_params: Set[nn.Parameter] = set()
        self.process_group: Optional[dist.ProcessGroup] = None
        self.rank: int = -1
        self.world_size: int = -1
        self.sharding_strategy = ShardingStrategy.FULL_SHARD
        self._use_orig_params: bool = False
        self.training_state = TrainingState.IDLE
        self._unshard_params_ctx: Dict[nn.Module, Generator] = {}
        self._state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT
        self._state_dict_config: StateDictConfig = FullStateDictConfig()
        self._optim_state_dict_config: OptimStateDictConfig = FullOptimStateDictConfig()
        self._is_root: Optional[bool] = None
        self._handle: Optional[flat_param_file.FlatParamHandle] = None
        self._fully_sharded_module_to_handle: Dict[
            nn.Module, Optional[flat_param_file.FlatParamHandle]
        ] = {}
        self.compute_device: Optional[torch.device] = None
        # Abstract device handle for fsdp compute device. For now,
        # the compute device must implement cuda semantics used by fsdp
        self._device_handle: _FSDPDeviceHandle = _UninitializedDeviceHandle()
        # All following attributes should only be used for root states:
        # Save these static lists to avoid the repeated tree traversals
        self._all_fsdp_states: List[_FSDPState] = []
        self._all_handles: List[flat_param_file.FlatParamHandle] = []
        self._device_mesh: Optional[DeviceMesh] = None


def _get_module_fsdp_state(module: nn.Module) -> Optional[_FSDPState]:
    state = _get_module_state(module)
    if state is None or not isinstance(state, _FSDPState):
        return None
    return state


def _get_module_fsdp_state_if_fully_sharded_module(
    module: nn.Module,
) -> Optional[_FSDPState]:
    state = _get_module_fsdp_state(module)
    if state is None:
        return None
    if state == module:  # FullyShardedDataParallel module case.
        return state
    if module in state._fully_sharded_module_to_handle:  # fully_shard case.
        return state
    return None


class TrainingState(Enum):
    """
    An enum that indicates the state of a ``FullyShardedDataParallel` instance.
    """

    IDLE = auto()
    FORWARD_BACKWARD = auto()
    SUMMON_FULL_PARAMS = auto()


class HandleTrainingState(Enum):
    """
    An enum that indicates the state of a ``FlatParamHandle`.
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


def _is_composable(state: _FSDPState):
    # TODO: This is a temporary hack for differentiate between code paths.
    return not isinstance(state, nn.Module)


@no_type_check
def _module_handle(state: _FSDPState, module: nn.Module) -> Optional["FlatParamHandle"]:
    """
    Returns the ``FlatParamHandle`` s corresponding to ``module``. This is
    the handle that contains some parameter in ``module``.
    """
    if _is_composable(state):
        # A valid FSDP state may have no managed parameters and hence no
        # handles, meaning no entry in `_fully_sharded_module_to_handles`
        if state._handle is None:
            return None
        assert (
            module in state._fully_sharded_module_to_handle
        ), f"Expects a fully sharded module but got {module} on rank {state.rank}"
        return state._fully_sharded_module_to_handle[module]
    else:
        # NOTE: This assumes `module` is a `FullyShardedDataParallel` instance.
        return module._handle


@no_type_check
def _has_fsdp_params(state: _FSDPState, module: nn.Module) -> bool:
    """Returns if ``module`` has parameters managed by FSDP."""
    return _module_handle(state, module) is not None


def _get_sharding_strategy(handle):
    """
    Returns the sharding strategy of the handle.
    """
    return handle._sharding_strategy if handle else None


def clean_tensor_name(tensor_name: str) -> str:
    """
    Cleans the parameter or buffer name by removing any module wrapper
    prefixes.
    """
    tensor_name = tensor_name.replace(FSDP_PREFIX, "")
    # TODO: Explicitly replacing the checkpoint wrapper prefix is not ideal as
    # it couples `CheckpointWrapper` and FSDP and also does not scale for more
    # module wrappers.
    tensor_name = tensor_name.replace(_CHECKPOINT_PREFIX, "")
    return tensor_name


def _set_fsdp_flattened(tensor: torch.Tensor) -> None:
    """
    Sets an attribute on ``tensor`` to mark it as flattened by FSDP. This is to
    avoid re-flattening it during nested construction.
    """
    setattr(tensor, FSDP_FLATTENED, True)


def _is_fsdp_flattened(tensor: torch.Tensor) -> bool:
    """Returns if ``tensor`` has been marked as flattened by FSDP."""
    return getattr(tensor, FSDP_FLATTENED, False)


def _named_parameters_with_duplicates(
    module: nn.Module, **kwargs: Any
) -> List[Tuple[str, nn.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    assert (
        "remove_duplicate" not in kwargs
    ), "_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument."
    kwargs["remove_duplicate"] = False
    try:
        ret = list(module.named_parameters(**kwargs))
    except AssertionError as e:
        kwargs.pop("remove_duplicate")
        ret = list(module.named_parameters(**kwargs))
    return ret


def _get_param_to_fqns(
    model: torch.nn.Module,
    dedup_shared_params: bool = True,
) -> Dict[nn.Parameter, List[str]]:
    """
    Constructs a mapping from parameter to a list of its FQNs. Each normal
    parameter maps to a singleton list containing its FQN, while each
    ``FlatParameter`` maps to a list of its original parameter FQNs, which may
    have length greater than one. All FQNs are prefixed starting from
    ``model``.

    Args:
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance).
        dedup_shared_params (bool): For shared parameters, if ``True``, only
            includes the FQNs corresponding to the first encounter of the
            shared parameter in the module traversal; if ``False``, then
            includes the FQNs across all encounters. (Default: ``True``)
    """

    def module_fn(module, prefix, tree_level, param_to_fqns):
        for param_name, param in _named_parameters_with_duplicates(
            module, recurse=False
        ):
            local_fqns = (
                param._fqns
                if isinstance(param, flat_param_file.FlatParameter)
                else [param_name]
            )  # prefixed from `module`
            global_fqns = [
                clean_tensor_name(prefix + name) for name in local_fqns
            ]  # prefixed from the top level `model` (i.e. including `prefix`)
            is_shared_param = param in param_to_fqns
            if not is_shared_param:
                param_to_fqns[param] = global_fqns
            else:
                if isinstance(param, flat_param_file.FlatParameter):
                    # DMP overwrites `named_parameters` and skip (advance to
                    # the next child module) the wrapped_module (e.g.,
                    # _dmp_wrapped_module and _fsdp_wrapped_module). When a user
                    # calls `named_child` to traverse the module recursively and
                    # calls `named_parameters` with `recurse=False`, parameters
                    # will be traversed more than once.
                    # This hack is specified designed for DMP + FSDP. We
                    # overwrite the flat_parameters traversal result to only obtain
                    # the last one, which happens to be the correct one.
                    #
                    # TODO: Remove this hack once DMP + FSDP is not supported.
                    warnings.warn(
                        "FlatParameter is being traversed more than once. "
                        "This case should only happen when using "
                        "DistributedModelParallel with FullyShardedDataParallel."
                    )
                    param_to_fqns[param] = global_fqns
                elif not dedup_shared_params:
                    param_to_fqns[param].extend(global_fqns)

    def return_fn(param_to_fqns):
        return param_to_fqns

    param_to_unflat_param_names: Dict[torch.nn.Parameter, List[str]] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [key for key, _ in _named_parameters_with_duplicates(model)],
        param_to_unflat_param_names,
    )


def _apply_to_modules(
    root_module: torch.nn.Module,
    module_fn: Callable,
    return_fn: Callable,
    filter_fqns: Optional[List[str]] = None,
    *args,
    **kwargs,
):
    """
    Performs a pre-order traversal of the modules in the hierarchy rooted at
    ``root_module``, applying ``module_fn`` at each module and finally
    returning a value using ``return_fn``. The traversal constructs the full
    module prefix name (e.g. "module.submodule." just like in model state dict)
    and makes that available to ``module_fn``.

    ``filter_fqns`` is used because some module may have its own prefix similar
    to ``FullyShardedDataParallel`` and the ``named_parameters()`` is overwritten
    to remove the prefix.
    """

    def f(module: torch.nn.Module, prefix: str, tree_level: int, *args, **kwargs):
        # Call the module function before recursing over children (pre-order)
        module_fn(module, prefix, tree_level, *args, **kwargs)
        for submodule_name, submodule in module.named_children():
            if submodule is None:
                continue
            new_prefix = prefix + submodule_name + "."
            new_tree_level = tree_level + 1
            if filter_fqns is not None:
                for fqn in filter_fqns:
                    if fqn.startswith(new_prefix):
                        break
                else:
                    # DMP's named_parameter() will mess up the traversal with
                    # ``named_children`` + `named_parameter(recurse=False)``.
                    # This hack is a must to make the traversal work.
                    # TODO: Remove this hack once DMP + FSDP is not supported.
                    if (
                        submodule_name == "_fsdp_wrapped_module"
                        or submodule_name == "_dmp_wrapped_module"
                    ):
                        warnings.warn(
                            "An unexpected prefix is detected. This case "
                            " should only happen when using DMP with FSDP. "
                            f"prefix = {prefix}, "
                            f"submodule_name = {submodule_name}"
                        )
                        new_prefix = prefix
                    elif submodule_name == "module":
                        warnings.warn(
                            "An unexpected prefix is detected. This case "
                            " should only happen when DDP wraps the outer "
                            " modules while FSDP wraps the inner ones."
                            f"prefix = {prefix}, "
                            f"submodule_name = {submodule_name}"
                        )
                        new_prefix = prefix
            f(submodule, new_prefix, new_tree_level, *args, **kwargs)

    f(root_module, "", 0, *args, **kwargs)
    return return_fn(*args, **kwargs)


@no_type_check
def _assert_in_training_states(
    state: _FSDPState,
    training_states: List[TrainingState],
) -> None:
    """Asserts that FSDP is in the states ``_training_states``."""
    # Raise a `ValueError` instead of using `assert` to ensure that these
    # logical assertions run even if `assert`s are disabled
    if state.training_state not in training_states:
        msg = (
            f"expected to be in states {training_states} but current state is "
            f"{state.training_state}"
        )
        # Print the error on rank 0 in case this is called in the backward pass
        if state.rank == 0:
            if isinstance(state, nn.Module):
                print(f"Asserting FSDP instance is: {state}")
            print(f"ERROR: {msg}")
            traceback.print_stack()
        raise ValueError(msg)


def _get_root_modules(modules: Set[nn.Module]) -> Set[nn.Module]:
    """
    Returns:
        Set[nn.Module]: The subset of ``modules`` that are root modules (i.e.
        parent-less) with respect to the modules in the set itself. In other
        words, these are the modules in ``modules`` that are not the child of
        any other module in ``modules``.
    """
    root_modules: Set[nn.Module] = set()
    module_to_submodules = {module: set(module.modules()) for module in modules}
    for candidate_module in modules:
        is_root_module = True
        for module, submodules in module_to_submodules.items():
            is_child_module = (
                candidate_module is not module and candidate_module in submodules
            )
            if is_child_module:
                is_root_module = False
                break
        if is_root_module:
            root_modules.add(candidate_module)
    return root_modules
