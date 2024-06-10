# mypy: allow-untyped-defs
import contextlib
import functools
import gc
import warnings
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Iterable,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._state_dict_utils import (
    _broadcast_state_dict,
    _flatten_state_dict,
    _gather_state_dict,
    _offload_state_dict_to_cpu,
    _unflatten_state_dict,
)
from torch.distributed._tensor import DTensor
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

FQNS_T = Set[str]
PrimitiveType = Union[DTensor, ShardedTensor, torch.Tensor, int, float, str]
ValueType = Union[
    PrimitiveType, List[PrimitiveType], Tuple[PrimitiveType], Dict[str, "ValueType"]
]
DictValueType = Dict[str, ValueType]
ListDictValueType = List[DictValueType]
OptimizerStateType = Dict[str, Union[DictValueType, ListDictValueType]]


_patched_state_dict: Set[Callable] = set()


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


@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: Dict[
        Union[str, torch.Tensor], Union[FQNS_T, torch.Tensor]
    ] = field(default_factory=dict)
    submodule_prefixes: Set[str] = field(default_factory=set)
    handle_model: bool = True
    handle_optim: bool = True
    fsdp_context: Callable = contextlib.nullcontext
    fsdp_modules: List[nn.Module] = field(default_factory=list)


@functools.lru_cache(maxsize=None)
def _get_fqns(
    model: nn.Module,
    name: str,
    skip_ddp_prefix: bool = True,
    skip_compiler_prefix: bool = True,
) -> FQNS_T:
    """
    This API is used to convert the name of a parameter to the FQNs. For FSDP
    without `use_orig_params`, the name of FlatParameter can be mapped to
    multiple original parameters. As a result, the return type of this function
    is `Set[str]`.

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
            assert curr_obj_name == "module"
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
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)
        elif isinstance(curr_obj, torch._dynamo.eval_frame.OptimizedModule):
            assert curr_obj_name == "_orig_mod"
            curr_obj = curr_obj._orig_mod
            if not skip_compiler_prefix:
                fqn_obj_names.append(curr_obj_name)
        else:
            fqn_obj_names.append(curr_obj_name)
            if curr_obj_name == nn.modules.module._EXTRA_STATE_KEY_SUFFIX:
                if i != len(obj_names) - 1:
                    raise RuntimeError("Expect `_extra_state` to be the last obj name")
            else:
                curr_obj = getattr(curr_obj, curr_obj_name)

    return {".".join(fqn_obj_names).replace(_CHECKPOINT_PREFIX, "")}


class _EXTRA_STATE:
    pass


def _iterate_valid_model_state(model):
    visited_modules: Set[nn.Module] = set()

    def recurse(module: nn.Module, curr_fqn: str) -> Generator:
        visited_modules.add(module)

        curr_fqn = f"{curr_fqn}." if curr_fqn else ""
        for name, submodule in module.named_children():
            if submodule in visited_modules:
                continue
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
    optims: Tuple[torch.optim.Optimizer, ...],
    optim_only: bool,
    *,
    submodules: Optional[Set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
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
        )
    if optim_only and not optims:
        raise RuntimeError(
            "Optimizers are not passed in but optim_only is set to True."
        )

    options = options or StateDictOptions()

    fqn_param_mapping: Dict[
        Union[str, torch.Tensor], Union[Set[str], torch.Tensor]
    ] = {}
    for name, param in _iterate_valid_model_state(model):
        fqns = _get_fqns(model, name)
        if not isinstance(param, _EXTRA_STATE):
            fqn_param_mapping[param] = fqns
        for fqn in fqns:
            if not isinstance(param, _EXTRA_STATE):
                fqn_param_mapping[fqn] = param

    submodule_prefixes: Set[str] = set()
    if submodules:
        submodules = set(submodules)
        for name, module in model.named_modules():
            if module not in submodules:
                continue
            fqns = _get_fqns(model, name)
            assert len(fqns) == 1, "Submodule FQN should only have 1 instance"
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
        submodule_prefixes=submodule_prefixes,
        fsdp_context=fsdp_context,
        fsdp_modules=cast(List[nn.Module], fsdp_modules),
        handle_model=not optim_only,
        handle_optim=(len(optims) > 0),
    )


def _verify_state_dict(
    model_state_dict: Dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> None:
    for module in info.fsdp_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        assert fsdp_state is not None, "Expected a fsdp_state with a fsdp module."

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

    for key in model_state_dict.keys():
        if _FLAT_PARAM in key:
            raise RuntimeError(
                f"{key} contains {_FLAT_PARAM}. This can happen if the model "
                "is not the root module."
            )


def _state_dict_fn(obj: Union[nn.Module, torch.optim.Optimizer], api: str) -> Callable:
    call = getattr(obj, api)
    if call in _patched_state_dict:
        call = functools.partial(getattr(obj.__class__, api), self=obj)
    return call


def _maybe_full_or_cpu_state_dict(
    state_dict: Dict[str, Any], info: _StateDictInfo
) -> Dict[str, Any]:
    if info.full_state_dict:
        ranks_only = (
            tuple()
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


def _get_model_state_dict(
    model: nn.Module, info: _StateDictInfo
) -> Dict[str, ValueType]:
    if not info.handle_model:
        return {}

    with info.fsdp_context():
        state_dict = _state_dict_fn(model, "state_dict")()

    for key in list(state_dict.keys()):
        fqns = _get_fqns(model, key)
        assert len(fqns) == 1
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
        new_state_dict: Dict[str, ValueType] = {}
        # TODO: make this faster.
        for fqn in state_dict.keys():
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

    for key, p in list(state_dict.items()):
        if torch.is_tensor(p) and p.is_meta:
            state_dict.pop(key)

    return _maybe_full_or_cpu_state_dict(state_dict, info)


def _load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, ValueType],
    info: _StateDictInfo,
) -> _IncompatibleKeys:
    if not info.handle_model or (not state_dict and not info.broadcast_from_rank0):
        return _IncompatibleKeys({}, {})

    local_state_dict = {}
    for key, value in _iterate_valid_model_state(model):
        fqns = _get_fqns(model, key)
        fqns_with_prefix = _get_fqns(
            model, key, skip_ddp_prefix=False, skip_compiler_prefix=False
        )

        for fqn, fqn_with_prefix in zip(fqns, fqns_with_prefix):
            if (
                not info.broadcast_from_rank0 or dist.get_rank() == 0
            ) and fqn != fqn_with_prefix:
                state_dict[fqn_with_prefix] = state_dict.pop(fqn)
            local_state_dict[fqn_with_prefix] = value

    if info.broadcast_from_rank0:
        device = None
        for key, value in local_state_dict.items():
            if torch.is_tensor(value) and value.dim() > 0:
                if device is None:
                    device = value.device
                else:
                    assert device == value.device
        assert device is not None
        _broadcast_state_dict(
            state_dict, local_state_dict, device=device, strict=info.strict
        )
        for fqn, local_state in local_state_dict.items():
            state_dict[fqn] = local_state

    with info.fsdp_context():
        return cast(
            _IncompatibleKeys,
            _state_dict_fn(model, "load_state_dict")(
                state_dict=state_dict, strict=info.strict
            ),
        )


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    """
    Initialize optim states by calling the step() with zero grads.
    """
    if optim.state:
        # The optimizer state is initialized.
        return

    for param_group in optim.param_groups:
        for param in param_group[_PARAMS]:
            if param.grad is not None:
                raise RuntimeError(
                    "state_dict can only be used if the optimizer "
                    "states are initialized (usually after one step() with "
                    "gradients) or gradients are None. For the later case, "
                    "state_dict will fake the gradients as zero "
                    "to initialize the optimizer states. However, the "
                    "gradients are not None."
                )
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

    # Some optimizers will update parameters regardless of grads due to lr, so
    # make lr to zero when calling `step()`.
    lrs = []
    for param_group in optim.param_groups:
        if "lr" in param_group:
            lrs.append(param_group["lr"])
            param_group["lr"] = 0.0
    optim.step(closure=None)
    # Whether to recover the "lr" should not matter too much as we will
    # restore checkpointing later.
    for param_group in optim.param_groups:
        if "lr" in param_group:
            param_group["lr"] = lrs.pop(0)
    optim.zero_grad(set_to_none=True)


def _flatten_optim_state_dict(state_dict: OptimizerStateType) -> Dict[str, ValueType]:
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
        "param_group": [
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
        "param_group.layer1.weight.lr" : 0.1,
        "param_group.layer2.weight.lr" : 0.1,
        "param_group.layer1.weight.betas" : (0.9, 0.95),
        "param_group.layer2.weight.betas" : (0.9, 0.95),
    }

    Note that if any of the value is a container, like the betas in the example,
    this API won't flattent it.
    """

    def _raise_if_type_not_supported(v):
        if not isinstance(v, (torch.Tensor, int, float)):
            raise NotImplementedError(
                "Flattening optimizer state_dict only supports "
                "tensor, int, float states now. "
                f"Type is {type(v)}."
            )

    ret: Dict[str, ValueType] = {}
    for fqn, state in cast(DictValueType, state_dict[_STATE]).items():
        for k, v in cast(DictValueType, state).items():
            _raise_if_type_not_supported(v)
            ret[f"{_STATE}.{fqn}.{k}"] = v

    for param_group in cast(ListDictValueType, state_dict[_PG]):
        fqns = param_group.pop(_PARAMS)
        for fqn in cast(List[str], fqns):
            for k, v in param_group.items():
                ret[f"{_PG}.{fqn}.{k}"] = v
    return ret


def _unflatten_optim_state_dict(
    optim: torch.optim.Optimizer,
    state_dict: Dict[str, ValueType],
    info: _StateDictInfo,
) -> OptimizerStateType:
    """
    This API unflattens the state_dict generated by _flatten_optim_state_dict().
    See the docstring of _flatten_optim_state_dict() for more detail.
    """
    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {_STATE: state, _PG: pg_state}

    for param_group in optim.param_groups:
        pg_state.append({_PARAMS: []})
        for param in param_group[_PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                params = pg_state[-1][_PARAMS]
                assert isinstance(params, list)  # typing
                params.append(fqn)
                if not param.requires_grad:
                    continue
                state[fqn] = {}
                for state_name in optim.state[param].keys():
                    cast(DictValueType, state[fqn])[state_name] = state_dict[
                        f"{_STATE}.{fqn}.{state_name}"
                    ]

        first_param_fqn = cast(List[str], pg_state[-1][_PARAMS])[0]
        for k in param_group.keys():
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


def _get_optim_state_dict(
    model: nn.Module,
    optimizers: Tuple[torch.optim.Optimizer, ...],
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
            # We can only use a string replacment without correctness check.
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
                assert len(fqns) == 1
                fqn = next(iter(fqns))
                if param not in param_pid_mapping:
                    continue
                pid = param_pid_mapping[param]
                fqn_pid_mapping[fqn] = pid
                fqn_pid_mapping[pid] = fqn

            for key in list(osd[_STATE].keys()):
                fqn = fqn_pid_mapping[key]
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
    pg_mapping: Dict[int, int] = {}

    if all(
        isinstance(k, int) for k in cast(DictValueType, optim_state_dict[_STATE]).keys()
    ):
        return optim_state_dict

    for param_group in optim.param_groups:
        pg_state.append({_PARAMS: []})
        for param in param_group[_PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                params = pg_state[-1][_PARAMS]
                assert isinstance(params, list)
                params.append(fqn)
                if param.requires_grad:
                    state[fqn] = cast(DictValueType, optim_state_dict[_STATE])[fqn]
                for loaded_param_group in cast(
                    ListDictValueType, optim_state_dict[_PG]
                ):
                    params = loaded_param_group[_PARAMS]
                    assert isinstance(params, list)
                    if fqn in params:
                        pg_mapping[id(loaded_param_group)] = len(return_osd[_PG]) - 1

    for param_group in cast(ListDictValueType, optim_state_dict[_PG]):
        idx = pg_mapping.get(id(param_group), -1)
        if idx == -1:
            continue
        for key, value in param_group.items():
            if key == _PARAMS:
                continue
            # TODO: check if value is the same if exists.
            pg_state[idx][key] = value

    return return_osd


def _load_optim_state_dict(
    model: nn.Module,
    optimizers: Tuple[torch.optim.Optimizer, ...],
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
                    optim, cast(Dict[str, ValueType], state_dict), info
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

                assert len(fqns) == 1
                fqn = fqns.pop()
                fqn_with_compiler = fqns_with_compiler.pop()
                for g in optim_state_dict[_PG]:
                    val = cast(Dict[str, Any], g)
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
        elif info.broadcast_from_rank0:
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
            assert device is not None
            flatten_osd, osd_mapping = _flatten_state_dict(optim_state_dict)
            flatten_local_osd, local_osd_mapping = _flatten_state_dict(local_state_dict)
            _broadcast_state_dict(flatten_osd, flatten_local_osd, device=device)
            # The modifications listed seek to address the problem where optim might possess
            # dissimilar parameters in comparison to optim_state_dict. This is achieved by
            # incorporating differential parameters within local, which may result in optim
            # having additional parameters ultimately.
            for optim_key in flatten_osd.keys():
                if optim_key not in flatten_local_osd:
                    assert optim_key in osd_mapping
                    flatten_local_osd[optim_key] = flatten_osd[optim_key]
                    local_osd_mapping[optim_key] = osd_mapping[optim_key]
            optim_state_dict = _unflatten_state_dict(
                flatten_local_osd, local_osd_mapping
            )

        # Note that we do not have to convert the FQN back to param id here if
        # order in optim.param_groups[idx][_PARAMS] is the same as the one in
        # optim_state_dict[_PG][idx][_PARAMS].
        _state_dict_fn(optim, "load_state_dict")(state_dict=optim_state_dict)


def get_model_state_dict(
    model: nn.Module,
    *,
    submodules: Optional[Set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> Dict[str, ValueType]:
    """
    Return the model state_dict of ``model``.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        submodules (deprecated): Optional[Set[nn.Module]]: only return the model parameters
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
            tuple(),
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        model_state_dict = _get_model_state_dict(model, info)
        _verify_state_dict(model_state_dict, {}, info)
        return model_state_dict


def get_optimizer_state_dict(
    model: nn.Module,
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    *,
    submodules: Optional[Set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> OptimizerStateType:
    """
    Return the combined state_dict for optimizers.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[Set[nn.Module]]: only return the model parameters
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
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    *,
    submodules: Optional[Set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> Tuple[Dict[str, ValueType], OptimizerStateType]:
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
        >>> fsdp_state_dict, fsdp_optim_state_dict = get_state_dict(fsdp_model, fsdp_optim)

        >>> # if we simply call ddp_model.state_dict() and fsdp_model.state_dict(),
        >>> # the asserts will fail.
        >>> assert ddp_state_dict == fsdp_state_dict
        >>> assert ddp_optim_state == fsdp_optim_state_dict


    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[Set[nn.Module]]: only return the model parameters
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
    state_dict: Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]],
) -> Dict[str, ValueType]:
    if not state_dict:
        return {}

    if isinstance(next(iter(state_dict.keys())), nn.Module):
        warnings.warn(
            "Passing model_state_dict as a ``Dict[nn.Module, Dict[str, Any]]``"
            "is deprecated and will be removed in 2.5. If you need this "
            "feature, please preprocessing the model_state_dict to achieve the "
            "same functionality.",
            FutureWarning,
        )
        cast_state_dict = cast(Dict[nn.Module, Dict[str, ValueType]], state_dict)
        new_state_dict: Dict[str, ValueType] = {}
        for submodule, sub_state_dict in cast_state_dict.items():
            for name, m in model.named_modules():
                if m != submodule:
                    continue

                fqns = _get_fqns(model, name)
                assert len(fqns) == 1, "FQNs for a submodule should only have 1 element"
                prefix = f"{next(iter(fqns))}."
                new_state_dict.update(
                    {prefix + subfqn: value for subfqn, value in sub_state_dict.items()}
                )
        return new_state_dict
    else:
        return cast(Dict[str, ValueType], state_dict)


def set_model_state_dict(
    model: nn.Module,
    model_state_dict: Dict[str, ValueType],
    *,
    options: Optional[StateDictOptions] = None,
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
    model_state_dict: Dict[str, ValueType] = _unflatten_model_state_dict(
        model, model_state_dict
    )
    with _gc_context():
        info = _verify_options(model, tuple(), optim_only=False, options=options)

        _verify_state_dict(model_state_dict, {}, info)
        return _load_model_state_dict(model, model_state_dict, info)


def set_optimizer_state_dict(
    model: nn.Module,
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    optim_state_dict: OptimizerStateType,
    *,
    options: Optional[StateDictOptions] = None,
) -> None:
    """Load the optimizers state_dict.

    The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the
    optimizers. See ``set_state_dict`` for the detail usage.

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
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    *,
    model_state_dict: Dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
    options: Optional[StateDictOptions] = None,
) -> _IncompatibleKeys:
    """Load the model state_dict and optimizers state_dict.

    The counterpart of ``get_state_dict`` to set the state_dict to the model and
    optimizers.  The given ``model_state_dict`` and ``optim_state_dict`` do not
    have to be returned by ``get_state_dict`` but must meet the following
    requirements: 1) all FQNs are canonical FQNs as defined in ``get_state_dict``,
    2) if a tensor is sharded, it must be either a ShardedTensor or DTensor,
    3) optimizer state_dict cannot contain the parameter IDs; the keys should be
    the canonical FQNs.

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

    model_state_dict: Dict[str, ValueType] = _unflatten_model_state_dict(
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
    options: Optional[StateDictOptions] = None,
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

    def load_state_dict_call(state_dict: Dict[str, Any]):
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
    optimizers: Tuple[torch.optim.Optimizer, ...],
    options: Optional[StateDictOptions] = None,
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

    def load_state_dict_call(state_dict: Dict[str, Any]):
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
