# mypy: allow-untyped-defs
import contextlib
import functools
import gc
import warnings
from collections.abc import Callable, Generator, Iterable
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import Any, cast, no_type_check, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._state_dict_utils import (
    _broadcast_state_dict,
    _distribute_state_dict,
    _flatten_state_dict,
    _gather_state_dict,
    _offload_state_dict_to_cpu,
    _unflatten_state_dict,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
)
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state_if_fully_sharded_module,
    FSDP_WRAPPED_MODULE,
)
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils._pytree import tree_map_only


__all__ = [
    "FQNS_T",
    "PrimitiveType",
    "ValueType",
    "DictValueType",
    "ListDictValueType",
    "OptimizerStateType",
    "StateDictOptions",
    "get_model_state_dict",
    "get_optimizer_state_dict",
    "get_state_dict",
    "set_model_state_dict",
    "set_optimizer_state_dict",
    "set_state_dict",
]


_FLAT_PARAM = "_flat_param"
_PG = "param_groups"
_PARAMS = "params"
_STATE = "state"

FQNS_T = set[str]
PrimitiveType = Union[DTensor, ShardedTensor, torch.Tensor, int, float, str]
ValueType = Union[
    PrimitiveType, list[PrimitiveType], tuple[PrimitiveType], dict[str, "ValueType"]
]
DictValueType = dict[str, ValueType]
ListDictValueType = list[DictValueType]
OptimizerStateType = dict[str, DictValueType | ListDictValueType]


_patched_state_dict: set[Callable] = set()


@contextlib.contextmanager
def _gc_context():
    is_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if is_enabled:
            gc.enable()


@dataclass
class StateDictOptions:
    """
    This dataclass specifies how get_state_dict/set_state_dict will work.

    - ``full_state_dict``: if this is set to True, all the tensors in the
      returned state_dict will be gathered. No ShardedTensor and DTensor
      will be in the returned state_dict.

    - ``cpu_offload``: offload all the tensors to cpu. To prevent CPU OOM, if
      ``full_state_dict`` is also true, then only the rank0 will get the
      state_dict and all other ranks will get empty state_dict.

    - ``ignore_frozen_params``: if the value is True, the returned state_dict
      won't contain any frozen parameters -- the ``requires_grad`` is False.
      The default value is False.

    - ``keep_submodule_prefixes`` (deprecated): when ``submodules`` is not None, this option
      indicates whether to keep the submodule prefixes from the state_dict keys.
      or example, if the submodule is ``module.pretrain`` and the full FQN of
      the parameter is ``pretrain.layer1.weight`` of the param. When this option
      is True, the parameter's key in the returned state_dict will be
      ``pretrain.layer1.weight``. If the options is False, the key will be
      ``layer1.weight``.
      Note that if ``keep_submodule_prefixes`` is False, there may be conflicted
      FQNs, hence there should be only one submodule in ``submodules``.

    - ``strict``: the ``strict`` option when ``set_state_dict`` calls
      model.load_state_dict().

    - ``broadcast_from_rank0``: when the option is True, rank0 should receive a
       full state_dict and will broadcast the tensors in the state_dict/
       optim_state_dict one by one to other ranks. Other ranks will receive
       the tensors and shard according to the local shards in the model and
       optimizer. ``full_state_dict`` must be set to True when using this option.
       This option currently only supports DTensor, not the legacy ShardedTensor.
    """

    full_state_dict: bool = False
    cpu_offload: bool = False
    ignore_frozen_params: bool = False
    keep_submodule_prefixes: bool = True
    strict: bool = True
    broadcast_from_rank0: bool = False
    flatten_optimizer_state_dict: bool = False
    dsd_fqn_modifiers: str = "_fqn_modifiers"


@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: dict[
        str | torch.Tensor,
        FQNS_T | torch.Tensor,
    ] = field(default_factory=dict)
    shared_params_mapping: dict[
        str | torch.Tensor,
        FQNS_T | torch.Tensor,
    ] = field(default_factory=dict)
    submodule_prefixes: set[str] = field(default_factory=set)
    handle_model: bool = True
    handle_optim: bool = True
    fsdp_context: Callable = contextlib.nullcontext
    fsdp_modules: list[nn.Module] = field(default_factory=list)


def _get_fqns(
    model: nn.Module,
    name: str,
    dsd_fqn_modifiers: str = "_fqn_modifiers",
    skip_ddp_prefix: bool = True,
    skip_compiler_prefix: bool = True,
) -> FQNS_T:
    """
    This API is used to convert the name of a parameter to the FQNs. For FSDP
    without `use_orig_params`, the name of FlatParameter can be mapped to
    multiple original parameters. As a result, the return type of this function
    is `set[str]`.

    Args:
        module (nn.Module): the root model.
        name (str): the name
        skip_ddp_prefix (bool): whether to skip DDP's `module` prefix

    Returns:
        The canonical FQNs based on the model traversal.
    """

    # Remove the checkpoint prefix, if it exists.
    name = name.replace(_CHECKPOINT_PREFIX, "")
    if "." not in name:
        return {name}

    obj_names = name.split(".")
    fqn_obj_names = []
    curr_obj = model
    for i, curr_obj_name in enumerate(obj_names):
        if isinstance(curr_obj, DDP):
            if curr_obj_name != "module":
                raise AssertionError(f"Expected 'module', got '{curr_obj_name}'")
            curr_obj = curr_obj.module
            if not skip_ddp_prefix:
                fqn_obj_names.append(curr_obj_name)
        elif isinstance(curr_obj, FSDP):
            if i < len(obj_names) - 1 and obj_names[i + 1] == _FLAT_PARAM:
                prefix = ".".join(fqn_obj_names)
                flat_param = getattr(curr_obj, _FLAT_PARAM)
                if prefix:
                    prefix = f"{prefix}."
                return {f"{prefix}{fqn}" for fqn in flat_param._fqns}
            curr_obj = getattr(curr_obj, FSDP_WRAPPED_MODULE)
            if curr_obj_name != FSDP_WRAPPED_MODULE:
                # pyrefly: ignore [bad-argument-type]
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)
        elif isinstance(curr_obj, torch._dynamo.eval_frame.OptimizedModule):
            if curr_obj_name != "_orig_mod":
                raise AssertionError(f"Expected '_orig_mod', got '{curr_obj_name}'")
            curr_obj = curr_obj._orig_mod
            if not skip_compiler_prefix:
                fqn_obj_names.append(curr_obj_name)
        else:
            # In some modules, _fqn_modifiers would not shown in the state_dict keys,
            # skip them in the fqn to ensure load stat dict successfully for them.
            if hasattr(curr_obj, dsd_fqn_modifiers):
                if removed_fqn := getattr(curr_obj, dsd_fqn_modifiers)().get(
                    curr_obj_name
                ):
                    if hasattr(curr_obj, removed_fqn):
                        curr_obj = getattr(curr_obj, removed_fqn)
            # pyrefly: ignore [bad-argument-type]
            fqn_obj_names.append(curr_obj_name)
            if curr_obj_name == nn.modules.module._EXTRA_STATE_KEY_SUFFIX:
                if i != len(obj_names) - 1:
                    raise RuntimeError("Expect `_extra_state` to be the last obj name")
            else:
                curr_obj = getattr(curr_obj, curr_obj_name)

    return {".".join(fqn_obj_names).replace(_CHECKPOINT_PREFIX, "")}


class _EXTRA_STATE:
    pass


def _iterate_valid_model_state(model, dsd_fqn_modifiers="_fqn_modifiers"):
    visited_modules: set[nn.Module] = set()

    def recurse(module: nn.Module, curr_fqn: str) -> Generator:
        visited_modules.add(module)

        curr_fqn = f"{curr_fqn}." if curr_fqn else ""
        for name, submodule in module.named_children():
            if submodule in visited_modules:
                continue
            # if user have state_dict_hooks in their model, they can add the state_dict key changes
            # at dsd_fqn_modifiers in input to align with the function of state_dict_hook
            if (
                hasattr(module, dsd_fqn_modifiers)
                and name in getattr(module, dsd_fqn_modifiers)().values()
            ):
                # skip _fqn_modifiers here thus remove the last `.` added
                new_fqn = curr_fqn[:-1]
            else:
                new_fqn = f"{curr_fqn}{name}"
            yield from recurse(submodule, new_fqn)

        for name, obj in chain(
            module.named_buffers(recurse=False), module.named_parameters(recurse=False)
        ):
            if name in module._non_persistent_buffers_set:
                continue
            new_fqn = f"{curr_fqn}{name}"
            yield new_fqn, obj

        if (
            getattr(module.__class__, "get_extra_state", nn.Module.get_extra_state)
            != nn.Module.get_extra_state
        ):
            new_fqn = f"{curr_fqn}{nn.modules.module._EXTRA_STATE_KEY_SUFFIX}"
            yield new_fqn, _EXTRA_STATE()

    yield from recurse(model, "")


def _verify_options(
    model: nn.Module,
    optims: tuple[torch.optim.Optimizer, ...],
    optim_only: bool,
    *,
    submodules: set[nn.Module] | None = None,
    options: StateDictOptions | None = None,
) -> _StateDictInfo:
    """
    Verify the model and options passed by the user and generates _StateDictInfo.
    """
    if submodules:
        warnings.warn(
            "Getting submodules only model/optim state_dict is deprecated and "
            "will be removed in 2.5. This feature can be achieved by manually "
            "filtering out the state_dict returned from get_state_dict.",
            FutureWarning,
            stacklevel=2,
        )
    if optim_only and not optims:
        raise RuntimeError(
            "Optimizers are not passed in but optim_only is set to True."
        )

    options = options or StateDictOptions()

    fqn_param_mapping: dict[str | torch.Tensor, set[str] | torch.Tensor] = {}
    shared_params_mapping: dict[str | torch.Tensor, set[str] | torch.Tensor] = {}
    for name, param in _iterate_valid_model_state(model):
        if isinstance(param, _EXTRA_STATE):
            continue

        fqns = _get_fqns(model, name)
        fqn = fqn_param_mapping.get(param)
        if fqn is not None:
            cast(set[str], fqn_param_mapping[param]).update(fqns)
            shared_params_mapping[param] = fqn_param_mapping[param]
        else:
            # We need to do copy as _get_fqns is lru_cached
            fqn_param_mapping[param] = fqns.copy()
        for fqn in fqns:
            if not isinstance(param, _EXTRA_STATE):
                fqn_param_mapping[fqn] = param

    for param_, fqns_ in list(shared_params_mapping.items()):
        for fqn in fqns_:
            shared_params_mapping[fqn] = cast(torch.Tensor, param_)

    submodule_prefixes: set[str] = set()
    if submodules:
        submodules = set(submodules)
        for name, module in model.named_modules():
            if module not in submodules:
                continue
            fqns = _get_fqns(model, name)
            if len(fqns) != 1:
                raise AssertionError("Submodule FQN should only have 1 instance")
            submodule_prefixes.update(f"{fqn}." for fqn in fqns)

    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError(
            "full_state_dict must be True when broadcast_from_rank0 is True."
        )
    fsdp_modules = FSDP.fsdp_modules(model)
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
    fsdp_context: Callable
    if fsdp_modules:
        # FSDP API only work if at least one FSDP instance exists.
        if options.full_state_dict:
            state_dict_config = FullStateDictConfig(
                offload_to_cpu=options.cpu_offload, rank0_only=options.cpu_offload
            )
            optim_state_dict_config = FullOptimStateDictConfig(
                offload_to_cpu=options.cpu_offload,
                rank0_only=(options.cpu_offload or options.broadcast_from_rank0),
            )
            state_dict_type = StateDictType.FULL_STATE_DICT
        else:
            state_dict_config = ShardedStateDictConfig(
                offload_to_cpu=options.cpu_offload,
            )
            optim_state_dict_config = ShardedOptimStateDictConfig(
                offload_to_cpu=options.cpu_offload,
            )
            state_dict_type = StateDictType.SHARDED_STATE_DICT

        @contextlib.contextmanager
        def fsdp_state_dict_type_without_warning(
            module,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="FSDP.state_dict_type", category=FutureWarning
                )
                with FSDP.state_dict_type(
                    module=module,
                    state_dict_type=state_dict_type,
                    state_dict_config=state_dict_config,
                    optim_state_dict_config=optim_state_dict_config,
                ):
                    yield

        fsdp_context = functools.partial(
            fsdp_state_dict_type_without_warning,
            module=model,
            state_dict_type=state_dict_type,
            state_dict_config=state_dict_config,
            optim_state_dict_config=optim_state_dict_config,
        )
    else:
        fsdp_context = contextlib.nullcontext

    return _StateDictInfo(
        **asdict(options),
        fqn_param_mapping=fqn_param_mapping,
        shared_params_mapping=shared_params_mapping,
        submodule_prefixes=submodule_prefixes,
        fsdp_context=fsdp_context,
        fsdp_modules=cast(list[nn.Module], fsdp_modules),
        handle_model=not optim_only,
        handle_optim=(len(optims) > 0),
    )


def _verify_state_dict(
    model_state_dict: dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> None:
    for module in info.fsdp_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        if fsdp_state is None:
            raise AssertionError("Expected a fsdp_state with a fsdp module.")

    # Verify if the model_state_dict and optim_state_dict are valid. This API
    # should give the users an explicit error message to debug or report.
    if (
        info.handle_model
        and not model_state_dict
        and not info.submodule_prefixes
        and not info.ignore_frozen_params
        and not (info.cpu_offload and info.full_state_dict)
        and info.strict
        and not info.broadcast_from_rank0
    ):
        raise RuntimeError(
            "The option indicates that model state_dict is required to save "
            "or load, but model state_dict is empty."
            f"rank = {dist.get_rank()=}."
        )

    if info.handle_optim:
        if (
            not optim_state_dict
            and not (info.cpu_offload and info.full_state_dict)
            and (not info.broadcast_from_rank0)
        ):
            raise RuntimeError(
                "The option indicates that model state_dict is required to save, "
                f"or load but optim state_dict is empty. {optim_state_dict}"
            )

    for key in model_state_dict:
        if _FLAT_PARAM in key:
            raise RuntimeError(
                f"{key} contains {_FLAT_PARAM}. This can happen if the model "
                "is not the root module."
            )


def _state_dict_fn(obj: nn.Module | torch.optim.Optimizer, api: str) -> Callable:
    call = getattr(obj, api)
    if call in _patched_state_dict:
        call = functools.partial(getattr(obj.__class__, api), self=obj)
    return call


def _maybe_full_or_cpu_state_dict(
    state_dict: dict[str, Any], info: _StateDictInfo
) -> dict[str, Any]:
    if info.full_state_dict:
        ranks_only = (
            ()
            if (not info.cpu_offload or not torch.distributed.is_initialized())
            else (0,)
        )
        return _gather_state_dict(
            state_dict, cpu_offload=info.cpu_offload, ranks_only=ranks_only
        )
    elif info.cpu_offload:
        return _offload_state_dict_to_cpu(state_dict)
    else:
        return state_dict


@torch.no_grad()
def _get_model_state_dict(
    model: nn.Module, info: _StateDictInfo
) -> dict[str, ValueType]:
    if not info.handle_model:
        return {}

    with info.fsdp_context():
        state_dict = _state_dict_fn(model, "state_dict")()

    for key in list(state_dict.keys()):
        fqns = _get_fqns(model, key)
        if len(fqns) != 1:
            raise AssertionError(
                f"Expected 1 FQN for key '{key}', got {len(fqns)}: {fqns}"
            )
        fqn = next(iter(fqns))
        if fqn != key:
            # As we only support FSDP, DDP, and TP, the only cases are
            # wrapper-based DDP and compiler. Verify if the assumption
            # is correct.
            def verify(key, fqn) -> bool:
                if len(fqn) >= len(key):
                    return False
                fqn_split = fqn.split(".")
                key_split = key.split(".")
                fqn_idx = 0
                for key_idx, key_name in enumerate(key_split):
                    if key_name == fqn_split[fqn_idx]:
                        fqn_idx += 1
                        if fqn_idx == len(fqn_split):
                            return key_idx == len(key_split) - 1
                    elif key_name in ("module", "_orig_mod"):
                        continue
                    else:
                        return False
                return True

            if not verify(key, fqn):
                raise RuntimeError(f"An unexpected key, {key}, exists. FQN is {fqn}")
            state_dict[fqn] = state_dict.pop(key)

    if info.submodule_prefixes:
        new_state_dict: dict[str, ValueType] = {}
        # TODO: make this faster.
        for fqn in state_dict:
            for prefix in info.submodule_prefixes:
                if not fqn.startswith(prefix):
                    continue
                if info.keep_submodule_prefixes:
                    new_state_dict[fqn] = state_dict[fqn]
                else:
                    new_fqn = fqn[len(prefix) :]
                    new_state_dict[new_fqn] = state_dict[fqn]
        state_dict = new_state_dict

    if info.ignore_frozen_params:
        for key, param in model.named_parameters():
            if param.requires_grad:
                continue
            fqns = _get_fqns(model, key)
            for fqn in fqns:
                state_dict.pop(fqn)

    return _maybe_full_or_cpu_state_dict(state_dict, info)


@torch.no_grad()
def _load_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, ValueType],
    info: _StateDictInfo,
) -> _IncompatibleKeys:
    if not info.handle_model or (not state_dict and not info.broadcast_from_rank0):
        return _IncompatibleKeys({}, {})

    local_state_dict = {}
    for key, value in _iterate_valid_model_state(model, info.dsd_fqn_modifiers):
        fqns = _get_fqns(model, key, info.dsd_fqn_modifiers)
        fqns_with_prefix = _get_fqns(
            model,
            key,
            info.dsd_fqn_modifiers,
            skip_ddp_prefix=False,
            skip_compiler_prefix=False,
        )

        for fqn, fqn_with_prefix in zip(fqns, fqns_with_prefix):
            if (
                not info.broadcast_from_rank0 or dist.get_rank() == 0
            ) and fqn != fqn_with_prefix:
                load_value = state_dict.pop(fqn, None)
                if load_value is None:
                    if info.strict:
                        raise RuntimeError(f"Missing key: {fqn}.")
                else:
                    state_dict[fqn_with_prefix] = load_value
            local_state_dict[fqn_with_prefix] = value

    assign = False
    if info.broadcast_from_rank0 or info.full_state_dict:
        devices = set()
        for value in local_state_dict.values():
            if torch.is_tensor(value) and value.dim() > 0:
                devices.add(value.device)
        # In lora state_dict, there could be multiple devices, with meta device inside.
        # Take the other device in the broadcast/distribtue, and set assign to True
        if torch.device("meta") in devices:
            devices.remove(torch.device("meta"))
            assign = True
        if len(devices) == 0:
            devices.add(dist.distributed_c10d._get_pg_default_device())
        elif len(devices) > 1:
            raise ValueError("Multiple devices found")

        if info.broadcast_from_rank0:
            _broadcast_state_dict(
                state_dict,
                local_state_dict,
                device=devices.pop(),
                strict=info.strict,
                cpu_offload=info.cpu_offload,
            )
        elif info.full_state_dict:
            _distribute_state_dict(state_dict, local_state_dict, device=devices.pop())
        state_dict.update(local_state_dict)

    with info.fsdp_context():
        return cast(
            _IncompatibleKeys,
            _state_dict_fn(model, "load_state_dict")(
                state_dict=state_dict, strict=info.strict, assign=assign
            ),
        )


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    """
    Initialize optim states by calling the step() with zero grads.
    """
    if optim.state:
        # The optimizer state is initialized.
        return

    # There are some stateless optimizers like SGD. These optimizer will
    # not return in the above condition. So if gradients exist, we should also
    # return. If gradients do not exist, the following initialization should
    # not disturb SGD because the gradients and lr are both zero.
    for param_group in optim.param_groups:
        for param in param_group[_PARAMS]:
            if param.grad is not None:
                return

    for param_group in optim.param_groups:
        for param in param_group[_PARAMS]:
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

    # Some optimizers will update parameters regardless of grads due to lr, so
    # make lr to zero when calling `step()`.
    lrs = []
    for param_group in optim.param_groups:
        if "lr" in param_group:
            lrs.append(param_group["lr"])
            param_group["lr"] = (
                torch.tensor(0.0)
                if isinstance(param_group["lr"], torch.Tensor)
                else 0.0
            )
    optim.step(closure=None)
    # Whether to recover the "lr" should not matter too much as we will
    # restore checkpointing later.
    for param_group in optim.param_groups:
        if "lr" in param_group:
            param_group["lr"] = lrs.pop(0)
    optim.zero_grad(set_to_none=True)


def _flatten_optim_state_dict(state_dict: OptimizerStateType) -> dict[str, ValueType]:
    """
    This API flattens the optimizer state_dict to support optimizer resharding for
    MPMD, e.g., pipeline parallelism.

    Without the API, the original optimizer state_dict looks like:
    {
        "state": {
            "layer1.weight": {
                "step": 10, "exp_avg": SomeTensor, "exp_avg_sq": SomeTensor
            },
            "layer2.weight": {
                "step": 10, "exp_avg": SomeTensor, "exp_avg_sq": SomeTensor
            },
        },
        "param_groups": [
            {
                "lr": 0.0,
                "betas": (0.9, 0.95), ...,
                "params": ["layer1.weight", "layer2.weight"]
            }
        ]
    }

    With this API, the optimizer state_dict looks like:
    {
        "state.layer1.weight.step": 10,
        "state.layer2.weight.step": 10,
        "state.layer1.weight.exp_avg": SomeTensor,
        "state.layer2.weight.exp_avg": SomeTensor,
        "state.layer1.weight.exp_avg_sq": SomeTensor,
        "state.layer2.weight.exp_avg_sq": SomeTensor,
        "param_groups.layer1.weight.lr": 0.1,
        "param_groups.layer2.weight.lr": 0.1,
        "param_groups.layer1.weight.betas": (0.9, 0.95),
        "param_groups.layer2.weight.betas": (0.9, 0.95),
    }

    The "state" section supports arbitrary levels of nesting for optimizers like Shampoo.
    """

    def _flatten_state_nested_dict(
        nested_dict: dict[str, Any], prefix: str
    ) -> dict[str, ValueType]:
        """
        Recursively flatten a nested dictionary with dot-separated keys.

        Args:
            nested_dict: The dictionary to flatten
            prefix: The prefix to prepend to all keys

        Returns:
            Flattened dictionary with dot-separated keys
        """
        flattened: dict[str, ValueType] = {}

        for key, value in nested_dict.items():
            # Convert all keys to strings for flattening
            str_key = str(key)
            full_key = f"{prefix}.{str_key}" if prefix else str_key

            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened.update(_flatten_state_nested_dict(value, full_key))
            else:
                # Base case: store the value with the flattened key
                _raise_if_type_not_supported(value)
                flattened[full_key] = value

        return flattened

    def _raise_if_type_not_supported(v):
        if not isinstance(v, (torch.Tensor, int, float, dict)):
            raise NotImplementedError(
                "Flattening optimizer state_dict only supports "
                "tensor, int, float, dict states now. "
                f"Type is {type(v)}."
            )

    ret: dict[str, ValueType] = {}

    # Handle the "state" section with recursive flattening
    for fqn, state in cast(DictValueType, state_dict[_STATE]).items():
        state_prefix = f"{_STATE}.{fqn}"
        ret.update(
            _flatten_state_nested_dict(cast(dict[str, Any], state), state_prefix)
        )

    # Handle the "param_groups" section with two-level flattening
    for param_group in cast(ListDictValueType, state_dict[_PG]):
        fqns = param_group.pop(_PARAMS)
        for fqn in cast(list[str], fqns):
            for k, v in param_group.items():
                ret[f"{_PG}.{fqn}.{k}"] = v

    return ret


def _unflatten_optim_state_dict(
    optim: torch.optim.Optimizer,
    state_dict: dict[str, ValueType],
    info: _StateDictInfo,
) -> OptimizerStateType:
    """
    This API unflattens the state_dict generated by _flatten_optim_state_dict().
    Supports arbitrary levels of nesting in the state section through recursive reconstruction.

    See the docstring of _flatten_optim_state_dict() for more detail.
    """

    def _reconstruct_nested_dict(
        flattened_key: str, flattened_dict: dict[str, ValueType]
    ) -> dict[str, ValueType]:
        """
        Reconstructs a potentially nested value from flattened keys.
        For non-nested values, returns the value directly.
        For nested values, reconstructs the nested structure with string keys.
        """

        # Create the prefix to search for nested keys
        # e.g., if flattened_key is "state.layer1.weight", prefix becomes "state.layer1.weight."
        prefix = f"{flattened_key}."
        # Initialize an empty dictionary to build our nested structure
        nested_dict: dict[str, Any] = {}

        # Iterate through all keys in the flattened dictionary
        for key, value in flattened_dict.items():
            # Check if this key is nested under our target key
            # e.g., "state.layer1.weight.exp_avg" starts with "state.layer1.weight."
            if not key.startswith(prefix):
                # Skip keys that don't belong to this nested structure
                continue

            # Remove the prefix to get just the nested part
            # e.g., "state.layer1.weight.exp_avg" -> "exp_avg"
            remaining_key = key[len(prefix) :]
            # Split the remaining key into parts to build the nested structure
            # e.g., "step" -> ["step"] or "momentum_buffer" -> ["momentum_buffer"]
            parts = remaining_key.split(".")
            # Start at the root of our new nested dictionary
            current = nested_dict

            # Navigate through or create the nested dictionary structure
            # For each part except the last one (which will hold the value)
            for part in parts[:-1]:
                # Create the nested dictionary if it doesn't exist yet
                if part not in current:
                    current[part] = {}
                # Move deeper into the nested structure
                assert isinstance(current[part], dict)
                current = current[part]

            # Set the value at the final level using the last part as the key
            # e.g., current["exp_avg"] = tensor(...)
            current[parts[-1]] = value

        # Return the reconstructed nested dictionary (empty dict if no keys matched at all)
        return nested_dict

    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {_STATE: state, _PG: pg_state}

    for param_group in optim.param_groups:
        pg_state.append({_PARAMS: []})
        for param in param_group[_PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                # If a parameter is shared, only one of the FQN will be used.
                # So we need to verify which if this fqn is actually used in
                # the state_dict.
                if fqn in info.shared_params_mapping:
                    in_params = False
                    for k in param_group:
                        if k == _PARAMS:
                            continue
                        flatten_key = f"{_PG}.{fqn}.{k}"
                        if flatten_key in state_dict:
                            in_params = True
                        break
                else:
                    in_params = True

                if not in_params:
                    continue

                params = pg_state[-1][_PARAMS]
                if not isinstance(params, list):
                    raise AssertionError(f"Expected list, got {type(params)}")
                params.append(fqn)

                # Only add state if param requires grad
                if not param.requires_grad:
                    continue

                # Reconstruct state for this parameter
                state[fqn] = {}
                for state_name in optim.state[param]:
                    flattened_state_key = f"{_STATE}.{fqn}.{state_name}"

                    if flattened_state_key not in state_dict:
                        # Try to reconstruct the value
                        reconstructed_value = _reconstruct_nested_dict(
                            flattened_state_key, state_dict
                        )
                        cast(DictValueType, state[fqn])[state_name] = (
                            reconstructed_value
                        )
                    else:
                        # Existing keys mean no nesting, directly use the value.
                        cast(DictValueType, state[fqn])[state_name] = state_dict[
                            flattened_state_key
                        ]

        first_param_fqn = cast(list[str], pg_state[-1][_PARAMS])[0]
        for k in param_group:
            if k == _PARAMS:
                continue
            value = state_dict[f"{_PG}.{first_param_fqn}.{k}"]
            if k not in pg_state[-1]:
                pg_state[-1][k] = value
            elif pg_state[-1][k] != value:
                raise RuntimeError(
                    "All the parameters in the same parameter group should have "
                    f"the same saved param_group value. But {first_param_fqn}.{k} "
                    f"is {value} while other(s) is {pg_state[-1][k]}."
                )

    return return_osd


@torch.no_grad()
def _get_optim_state_dict(
    model: nn.Module,
    optimizers: tuple[torch.optim.Optimizer, ...],
    info: _StateDictInfo,
) -> OptimizerStateType:
    if not info.handle_optim:
        return {}

    optim_state_dict: OptimizerStateType = {_STATE: {}, _PG: []}
    for optim in optimizers:
        _init_optim_state(optim)
        osd = _state_dict_fn(optim, "state_dict")()
        if info.fsdp_modules:
            with info.fsdp_context():
                osd = FSDP.optim_state_dict(model, optim, osd)

            # We need to specially handle FlatParameter FSDP as
            # FlatParameter FSDP converts the FQNs.
            # There are no easy ways to do this conversion systematically.
            # We can only use a string replacement without correctness check.
            if not osd:
                continue
            for k in list(osd[_STATE].keys()):
                if "_orig_mod" in k:
                    osd[_STATE][k.replace("_orig_mod.", "")] = osd[_STATE].pop(k)
            for g in osd[_PG]:
                params = [k.replace("_orig_mod.", "") for k in g[_PARAMS]]
                g[_PARAMS] = params
        else:
            params = list(chain.from_iterable(g[_PARAMS] for g in optim.param_groups))
            param_pid_mapping = dict(zip(params, range(len(params))))
            fqn_pid_mapping = {}
            for key, param in model.named_parameters():
                fqns = _get_fqns(model, key)
                if len(fqns) != 1:
                    raise AssertionError(
                        f"Expected 1 FQN for key '{key}', got {len(fqns)}"
                    )
                fqn = next(iter(fqns))
                if param not in param_pid_mapping:
                    continue
                pid = param_pid_mapping[param]
                fqn_pid_mapping[fqn] = pid
                fqn_pid_mapping[pid] = fqn

            # Only convert top-level parameter IDs to FQNs, preserve nested key types
            for key in list(osd[_STATE].keys()):
                fqn = fqn_pid_mapping[key]
                # Move the entire state dict value (which may contain nested integer keys)
                # without modifying its internal structure
                osd[_STATE][fqn] = osd[_STATE].pop(key)

            for group in osd[_PG]:
                group[_PARAMS] = [fqn_pid_mapping[pid] for pid in group[_PARAMS]]

        if not osd:
            continue

        cast(DictValueType, optim_state_dict[_STATE]).update(osd[_STATE])
        cast(ListDictValueType, optim_state_dict[_PG]).extend(osd[_PG])

    if info.flatten_optimizer_state_dict:
        optim_state_dict = cast(
            OptimizerStateType, _flatten_optim_state_dict(optim_state_dict)
        )

    return _maybe_full_or_cpu_state_dict(optim_state_dict, info)


def _split_optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> OptimizerStateType:
    """
    Extract the corresponding optim state_dict from ``optim_state_dict`` for
    ``optim`` and return the result optim state_dict.

    Args:
        model (nn.Module): the root model.
        optim (torch.optim.Optimizer): the optimizer.
        optim_state_dict (Dict[str, ValueType]): the superset optim state_dict that
            contains the optim state_dict of ``optim``.
        info (_StateDictInfo): state dict information.

    Returns:
        The optim state_dict of ``optim``.
    """

    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {_STATE: state, _PG: pg_state}
    pg_mapping: dict[int, int] = {}

    if all(isinstance(k, int) for k in cast(DictValueType, optim_state_dict[_STATE])):
        return optim_state_dict

    for param_group in optim.param_groups:
        pg_state.append({_PARAMS: []})
        for param in param_group[_PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                if fqn in info.shared_params_mapping:
                    in_params = False
                    for loaded_param_group in cast(
                        ListDictValueType, optim_state_dict[_PG]
                    ):
                        if fqn in cast(list[str], loaded_param_group[_PARAMS]):
                            in_params = True
                            break
                else:
                    in_params = True
                if not in_params:
                    continue

                params = pg_state[-1][_PARAMS]
                if not isinstance(params, list):
                    raise AssertionError(f"Expected list, got {type(params)}")
                params.append(fqn)
                if param.requires_grad:
                    if fqn in cast(DictValueType, optim_state_dict[_STATE]):
                        state[fqn] = cast(DictValueType, optim_state_dict[_STATE])[fqn]
                    elif info.strict:
                        raise RuntimeError(
                            f"Missing optimizer state for parameter '{fqn}' in checkpoint. "
                            "The parameter requires gradients but has no saved optimizer state. "
                            "To load anyway, use StateDictOptions(strict=False)."
                        )
                for loaded_param_group in cast(
                    ListDictValueType, optim_state_dict[_PG]
                ):
                    if fqn in cast(list[str], loaded_param_group[_PARAMS]):
                        pg_mapping[id(loaded_param_group)] = len(return_osd[_PG]) - 1

        if len(param_group[_PARAMS]) == 0:
            # Param_group with empty params.
            ret = []
            for loaded_param_group in cast(ListDictValueType, optim_state_dict[_PG]):
                if len(cast(list[str], loaded_param_group[_PARAMS])) == 0:
                    ret.append(loaded_param_group)
            if len(ret) != 1:
                raise ValueError(
                    "There are param groups that have zero parameters. "
                    "In such a case, DSD only support exactly one param group "
                    "with zero parameters."
                    "But the loaded state_dict has zero or more than one param groups "
                    "that have zero parameters."
                )
            if len(optim_state_dict[_PG]) != len(optim.param_groups):
                raise ValueError(
                    "When there is a parameter group that has zero parameters, "
                    "multiple optimizers are not supported."
                )
            pg_mapping[id(loaded_param_group)] = len(return_osd[_PG]) - 1

    for param_group in cast(ListDictValueType, optim_state_dict[_PG]):
        pg_idx = pg_mapping.get(id(param_group), -1)
        if pg_idx == -1:
            continue

        for key, value in param_group.items():
            if key == _PARAMS:
                continue
            # TODO: check if value is the same if exists.
            pg_state[pg_idx][key] = value

    return return_osd


@torch.no_grad()
def _load_optim_state_dict(
    model: nn.Module,
    optimizers: tuple[torch.optim.Optimizer, ...],
    state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> None:
    if not info.handle_optim:
        return

    for optim in optimizers:
        _init_optim_state(optim)
        if state_dict:
            if _STATE in state_dict:
                optim_state_dict = _split_optim_state_dict(
                    model, optim, state_dict, info
                )
            else:
                optim_state_dict = _unflatten_optim_state_dict(
                    optim, cast(dict[str, ValueType], state_dict), info
                )
        else:
            optim_state_dict = {}
        if info.fsdp_modules:
            # We need to specially handle FlatParameter FSDP as
            # FlatParameter FSDP converts the FQNs.
            for original_fqn, _ in model.named_parameters():
                fqns = _get_fqns(model, original_fqn)
                fqns_with_compiler = _get_fqns(
                    model, original_fqn, skip_compiler_prefix=False
                )
                if fqns == fqns_with_compiler:
                    continue

                if len(fqns) != 1:
                    raise AssertionError(
                        f"Expected 1 FQN for '{original_fqn}', got {len(fqns)}"
                    )
                fqn = fqns.pop()
                fqn_with_compiler = fqns_with_compiler.pop()
                for g in optim_state_dict[_PG]:
                    val = cast(dict[str, Any], g)
                    params = [
                        key.replace(fqn, fqn_with_compiler) for key in val[_PARAMS]
                    ]
                    val[_PARAMS] = params
                osd_state = cast(DictValueType, optim_state_dict[_STATE])
                for k in list(osd_state.keys()):
                    if fqn in k:
                        osd_state[k.replace(fqn, fqn_with_compiler)] = osd_state.pop(k)

            with info.fsdp_context():
                optim_state_dict = FSDP.optim_state_dict_to_load(
                    model, optim, optim_state_dict
                )
        elif info.full_state_dict:
            info.full_state_dict = False
            local_state_dict = _get_optim_state_dict(model, (optim,), info)
            info.full_state_dict = True
            device = None

            def _device(t):
                if t.dim() > 0:
                    nonlocal device
                    if device is None:
                        device = t.device
                    elif device != t.device:
                        raise ValueError("Device mismatch")
                return t

            _ = tree_map_only(torch.Tensor, _device, local_state_dict)
            if device is None:
                raise AssertionError("Expected device to be set")
            flatten_osd, osd_mapping = _flatten_state_dict(optim_state_dict)
            flatten_local_osd, local_osd_mapping = _flatten_state_dict(local_state_dict)
            if info.broadcast_from_rank0:
                _broadcast_state_dict(flatten_osd, flatten_local_osd, device=device)
            else:
                _distribute_state_dict(flatten_osd, flatten_local_osd, device=device)
            # The modifications listed seek to address the problem where optim might possess
            # dissimilar parameters in comparison to optim_state_dict. This is achieved by
            # incorporating differential parameters within local, which may result in optim
            # having additional parameters ultimately.
            for optim_key in flatten_osd:
                if optim_key not in flatten_local_osd:
                    if optim_key not in osd_mapping:
                        raise AssertionError(
                            f"Expected key '{optim_key}' in osd_mapping"
                        )
                    flatten_local_osd[optim_key] = flatten_osd[optim_key]
                    local_osd_mapping[optim_key] = osd_mapping[optim_key]
            optim_state_dict = _unflatten_state_dict(
                flatten_local_osd, local_osd_mapping
            )
            for pg in optim_state_dict[_PG]:
                if _PARAMS not in pg:
                    cast(dict[str, ValueType], pg)[_PARAMS] = []

        # Note that we do not have to convert the FQN back to param id here if
        # order in optim.param_groups[idx][_PARAMS] is the same as the one in
        # optim_state_dict[_PG][idx][_PARAMS].
        _state_dict_fn(optim, "load_state_dict")(state_dict=optim_state_dict)


def get_model_state_dict(
    model: nn.Module,
    *,
    submodules: set[nn.Module] | None = None,
    options: StateDictOptions | None = None,
) -> dict[str, ValueType]:
    """
    Return the model state_dict of ``model``.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        submodules (deprecated): Optional[set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        The state_dict for ``model``.

    :rtype: typing.Dict[str, ValueType]
    """
    with _gc_context():
        info = _verify_options(
            model,
            (),
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        model_state_dict = _get_model_state_dict(model, info)
        _verify_state_dict(model_state_dict, {}, info)
        return model_state_dict


def get_optimizer_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    *,
    submodules: set[nn.Module] | None = None,
    options: StateDictOptions | None = None,
) -> OptimizerStateType:
    """
    Return the combined state_dict for optimizers.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        The state_dict for ``optimizers``.

    :rtype: OptimizerStateType
    """
    with _gc_context():
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        info = _verify_options(
            model,
            optimizers,
            optim_only=True,
            submodules=submodules,
            options=options,
        )
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        _verify_state_dict({}, optim_state_dict, info)
        return optim_state_dict


def get_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    *,
    submodules: set[nn.Module] | None = None,
    options: StateDictOptions | None = None,
) -> tuple[dict[str, ValueType], OptimizerStateType]:
    """
    Return the model state_dict and optimizers state_dict.

    ``get_state_dict`` can process any module that is parallelized by PyTorch
    FSDP/fully_shard, DDP/replicate, tensor_parallel/parallelize_module, and any
    combination of these parallelisms. The main functions of ``get_state_dict``
    are: 1.) returning a model and optimizer state_dict that can be resharded
    with a different number of trainers and/or different parallelisms.
    2.) hiding the parallelism-specific state_dict APIs. Users don't have to call
    these APIs.
    3.) sanity checking the result state_dict.

    The keys of the result state dictionary are the canonical FQNs (Fully
    Qualified Names).  A canonical FQN refers to the FQN based on a parameter's
    position in an nn.Module hierarchy. More specifically, a canonical FQN to a
    parameter is the FQN returned by ``module.named_parameters()`` or
    ``module.named_buffers()`` when the module is not distributed by any
    parallelisms. Since the optimizer internally uses parameter IDs to represent
    a parameter, there will be a conversion from the parameter IDs to the
    canonical FQNs when calling this API.

    ``get_state_dict`` can also process a module that is not parallelized. In
    such a case, ``get_state_dict`` only performs one function -- converting the
    optimizer parameter IDs to the canonical FQNs.

    Example:
        >>> # xdoctest: +SKIP
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch.distributed.checkpoint.state_dict import get_state_dict

        >>> fsdp_model = FSDP(copy.deepcopy(model))
        >>> fsdp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> ddp_model = DDP(copy.deepcopy(model))
        >>> ddp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)


        >>> ddp_state_dict, ddp_optim_state_dict = get_state_dict(ddp_model, ddp_optim)
        >>> fsdp_state_dict, fsdp_optim_state_dict = get_state_dict(
        ...     fsdp_model, fsdp_optim
        ... )

        >>> # if we simply call ddp_model.state_dict() and fsdp_model.state_dict(),
        >>> # the asserts will fail.
        >>> assert ddp_state_dict == fsdp_state_dict
        >>> assert ddp_optim_state == fsdp_optim_state_dict


    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        ``Tuple`` that contain model state_dict and optimizer state_dict.

    :rtype: typing.Tuple[typing.Dict[str, ValueType], OptimizerStateType]
    """

    with _gc_context():
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        info = _verify_options(
            model,
            optimizers,
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        model_state_dict = _get_model_state_dict(model, info)
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        _verify_state_dict(model_state_dict, optim_state_dict, info)
        return model_state_dict, optim_state_dict


def _unflatten_model_state_dict(
    model: nn.Module,
    state_dict: dict[nn.Module, dict[str, ValueType]] | dict[str, ValueType],
) -> dict[str, ValueType]:
    if not state_dict:
        return {}

    if isinstance(next(iter(state_dict.keys())), nn.Module):
        warnings.warn(
            "Passing model_state_dict as a ``Dict[nn.Module, Dict[str, Any]]``"
            "is deprecated and will be removed in 2.5. If you need this "
            "feature, please preprocessing the model_state_dict to achieve the "
            "same functionality.",
            FutureWarning,
            stacklevel=2,
        )
        cast_state_dict = cast(dict[nn.Module, dict[str, ValueType]], state_dict)
        new_state_dict: dict[str, ValueType] = {}
        for submodule, sub_state_dict in cast_state_dict.items():
            for name, m in model.named_modules():
                if m != submodule:
                    continue

                fqns = _get_fqns(model, name)
                if len(fqns) != 1:
                    raise AssertionError(
                        "FQNs for a submodule should only have 1 element"
                    )
                prefix = f"{next(iter(fqns))}."
                new_state_dict.update(
                    {prefix + subfqn: value for subfqn, value in sub_state_dict.items()}
                )
        return new_state_dict
    else:
        return cast(dict[str, ValueType], state_dict)


def set_model_state_dict(
    model: nn.Module,
    model_state_dict: dict[str, ValueType],
    *,
    options: StateDictOptions | None = None,
) -> _IncompatibleKeys:
    """Load the model state_dict.

    The counterpart of ``get_model_state_dict`` to set the state_dict to the
    model. See ``set_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        model_state_dict: (Dict[str, ValueType]):
           the model state_dict to load. If the key of the ``model_state_dict``
           is nn.Module, the key is a submodule of ``model`` and the value should
           be the state_dict of the submodule. When loading the state_dict,
           the prefix of the submodule will be append to the state_dict.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    :type model_state_dict: typing.Dict[str, ValueType]
    """
    model_state_dict: dict[str, ValueType] = _unflatten_model_state_dict(
        model, model_state_dict
    )
    with _gc_context():
        info = _verify_options(model, (), optim_only=False, options=options)

        _verify_state_dict(model_state_dict, {}, info)
        return _load_model_state_dict(model, model_state_dict, info)


def set_optimizer_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    optim_state_dict: OptimizerStateType,
    *,
    options: StateDictOptions | None = None,
) -> None:
    """Load the optimizers state_dict.

    The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the
    optimizers. See ``set_state_dict`` for the detail usage.

    WARN: ``set_optimizer_state_dict`` can only be called before ``backward()`` or after
        ``step()`` is called on the optimizers. Otherwise, the optimizer states won't be
        initialized correctly.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        optim_state_dict: OptimizerStateType:
            the optimizer state_dict to load.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        None

    :type optim_state_dict: typing.OptimizerStateType
    """
    with _gc_context():
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        info = _verify_options(model, optimizers, optim_only=True, options=options)

        _verify_state_dict({}, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)


def set_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    *,
    model_state_dict: dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
    options: StateDictOptions | None = None,
) -> _IncompatibleKeys:
    """Load the model state_dict and optimizers state_dict.

    The counterpart of ``get_state_dict`` to set the state_dict to the model and
    optimizers.  The given ``model_state_dict`` and ``optim_state_dict`` do not
    have to be returned by ``get_state_dict`` but must meet the following
    requirements: 1) all FQNs are canonical FQNs as defined in ``get_state_dict``,
    2) if a tensor is sharded, it must be either a ShardedTensor or DTensor,
    3) optimizer state_dict cannot contain the parameter IDs; the keys should be
    the canonical FQNs.

    WARN: ``set_state_dict`` can only be called before ``backward()`` or after ``step()``
        is called on the optimizers. Otherwise, the optimizer states won't be initialized
        correctly.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        model_state_dict: (Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]]):
           the model state_dict to load. If the key of the ``model_state_dict``
           is nn.Module, the key is a submodule of ``model`` and the value should
           be the state_dict of the submodule. When loading the state_dict,
           the prefix of the submodule will be append to the state_dict.
        optim_state_dict: OptimizerStateType:
            the optimizer state_dict to load.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys of the model state_dict.
            * **unexpected_keys** is a list of str containing the unexpected keys of the model state_dict.

    :type model_state_dict: typing.Dict[str, ValueType]
    :type optim_state_dict: typing.OptimizerStateType
    """

    model_state_dict: dict[str, ValueType] = _unflatten_model_state_dict(
        model, model_state_dict
    )
    with _gc_context():
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )
        info = _verify_options(
            model, optimizers, optim_only=not model_state_dict, options=options
        )

        _verify_state_dict(model_state_dict, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)
        return _load_model_state_dict(model, model_state_dict, info)


# TODO: correct the state_dict function signature.
# TODO: this API is not yet fully tested. Make it private
@no_type_check
def _patch_model_state_dict(
    model: nn.Module,
    *,
    options: StateDictOptions | None = None,
) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model`` to
    be a partial function to call ``get_state_dict`` and ``set_state_dict``.

    Example:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import patch_model_state_dict

        model = fsdp(model)
        patch_model_state_dict(model)

    Args:
        model (nn.Module): the nn.Module to the model.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.
    Returns:
        None
    """

    _state_dict_call = functools.partial(
        get_model_state_dict,
        model=model,
        options=options,
    )

    def state_dict_call():
        return _state_dict_call()

    model.state_dict = state_dict_call

    _load_state_dict_call = functools.partial(
        set_model_state_dict,
        model=model,
        options=options,
    )

    def load_state_dict_call(state_dict: dict[str, Any]):
        _load_state_dict_call(model_state_dict=state_dict)

    model.load_state_dict = load_state_dict_call

    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)


# TODO: correct the load_state_dict function signature.
# TODO: this API is not yet fully tested. Make it private
@no_type_check
def _patch_optimizer_state_dict(
    model: nn.Module,
    *,
    optimizers: tuple[torch.optim.Optimizer, ...],
    options: StateDictOptions | None = None,
) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers`` to
    be a partial function to call ``get_state_dict`` and ``set_state_dict``.

    Note that if there are multiple optimizers, all of the optimizers will be patched.
    So users only need to call one of the state_dict() to get the full result.

    Example:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import patch_model_state_dict

        model = fsdp(model)
        patch_model_state_dict(model)

    Args:
        model (nn.Module): the nn.Module to the model.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.
    Returns:
        None
    """

    _state_dict_call = functools.partial(
        get_optimizer_state_dict,
        model=model,
        optimizers=optimizers,
        options=options,
    )

    def state_dict_call():
        return _state_dict_call()

    _load_state_dict_call = functools.partial(
        set_optimizer_state_dict,
        model=model,
        optimizers=optimizers,
        options=options,
    )

    def load_state_dict_call(state_dict: dict[str, Any]):
        _load_state_dict_call(optim_state_dict=state_dict)

    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)
    optimizers = (
        (optimizers,)
        if isinstance(optimizers, torch.optim.Optimizer)
        else tuple(optimizers)
    )
    for optim in optimizers:
        optim.state_dict = state_dict_call
        optim.load_state_dict = load_state_dict_call
