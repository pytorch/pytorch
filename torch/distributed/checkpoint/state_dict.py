import contextlib
import functools
import gc
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
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
from torch.nn.parallel import DistributedDataParallel as DDP


FLAT_PARAM = "_flat_param"
PG = "param_groups"
PG_PREFIX = f"{PG}."
STATE = "state"
STATE_PREFIX = f"{STATE}."
PARAMS = "params"
FQNS_T = Set[str]

_patched_state_dict: Set[Callable] = set()


PrimitiveType = Union[DTensor, ShardedTensor, torch.Tensor, int, float, str]
ValueType = Union[
    PrimitiveType, List[PrimitiveType], Tuple[PrimitiveType], Dict[str, "ValueType"]
]
DictValueType = Dict[str, ValueType]
ListDictValueType = List[DictValueType]
OptimizerStateType = Dict[str, Union[DictValueType, ListDictValueType]]


@contextlib.contextmanager
def gc_context():
    is_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        # TODO: add logging for the gc details/time
        gc.collect()
        if is_enabled:
            gc.enable()


@dataclass
class DistributedStateDictOptions:
    # The default should be sharded_state_dict
    fsdp_state_dict_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    save_to_cpu: bool = True
    # Whether to save the frozen parameters. The default is True.
    save_frozen_params: bool = True


@dataclass
class _StateDictInfo(DistributedStateDictOptions):
    fqn_param_mapping: Dict[
        Union[str, torch.Tensor], Union[FQNS_T, torch.Tensor]
    ] = field(default_factory=dict)
    all_fqns: Set[str] = field(default_factory=set)
    handle_model: bool = True
    handle_optim: bool = True
    fsdp_context: Callable = contextlib.nullcontext
    fsdp_modules: List[nn.Module] = field(default_factory=list)


def _get_fqns(model: nn.Module, name: str, skip_ddp_prefix: bool = True) -> FQNS_T:
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
            if obj_names[i + 1] == FLAT_PARAM:
                prefix = ".".join(fqn_obj_names)
                flat_param = getattr(curr_obj, FLAT_PARAM)
                if prefix:
                    prefix = f"{prefix}."
                return {f"{prefix}{fqn}" for fqn in flat_param._fqns}
            curr_obj = getattr(curr_obj, FSDP_WRAPPED_MODULE)
            if curr_obj_name != FSDP_WRAPPED_MODULE:
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)
        else:
            fqn_obj_names.append(curr_obj_name)
            curr_obj = getattr(curr_obj, curr_obj_name)

    return {".".join(fqn_obj_names)}


def _verify_options(
    model: nn.Module,
    optims: Tuple[torch.optim.Optimizer, ...],
    model_only: bool,
    optim_only: bool,
    options: Optional[DistributedStateDictOptions] = None,
) -> _StateDictInfo:
    """
    Verify the model and options passed by the user and generates _StateDictInfo.
    """
    if model_only and optim_only:
        raise RuntimeError(
            "Both model_only and optim_only are set, which one do you need?"
        )
    if model_only and optims:
        raise RuntimeError(
            "If model_only is True optims must be an empty iterable object."
        )
    if optim_only and not optims:
        raise RuntimeError(
            "Optimizers are not passed in but optim_only is set to True."
        )

    options = options or DistributedStateDictOptions()

    fqn_param_mapping: Dict[
        Union[str, torch.Tensor], Union[Set[str], torch.Tensor]
    ] = {}
    all_fqns = set()
    for name, param in model.named_parameters():
        fqns = _get_fqns(model, name)
        fqn_param_mapping[param] = fqns
        for fqn in fqns:
            fqn_param_mapping[fqn] = param
            all_fqns.add(fqn)

    fsdp_modules = FSDP.fsdp_modules(model)
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
    fsdp_context: Callable
    if fsdp_modules:
        # FSDP API only work if at least one FSDP instance exists.
        if options.fsdp_state_dict_type == StateDictType.FULL_STATE_DICT:
            state_dict_config = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )
            optim_state_dict_config = FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )
        elif options.fsdp_state_dict_type == StateDictType.SHARDED_STATE_DICT:
            state_dict_config = ShardedStateDictConfig()
            optim_state_dict_config = ShardedOptimStateDictConfig()
        else:
            raise RuntimeError(
                "state_dict currently support only FSDP "
                "FULL_STATE_DICT and SHARDED_STATE_DICT"
            )
        fsdp_context = functools.partial(
            FSDP.state_dict_type,
            module=model,
            state_dict_type=options.fsdp_state_dict_type,
            state_dict_config=state_dict_config,
            optim_state_dict_config=optim_state_dict_config,
        )
    else:
        fsdp_context = contextlib.nullcontext

    return _StateDictInfo(
        **asdict(options),
        fqn_param_mapping=fqn_param_mapping,
        all_fqns=all_fqns,
        fsdp_context=fsdp_context,
        fsdp_modules=cast(List[nn.Module], fsdp_modules),
        handle_model=model_only or not optim_only,
        handle_optim=optim_only or (not model_only and len(optims) > 0),
    )


def _verify_state_dict(
    model_state_dict: Dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> None:
    # FSDP root must exist otherwise FSDP state_dict will be incorrect.
    has_fsdp_root = False
    for module in info.fsdp_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        assert fsdp_state is not None, "Expected a fsdp_state with a fsdp module."
        if fsdp_state._is_root:
            has_fsdp_root = True
            break
    if info.fsdp_modules and not has_fsdp_root:
        raise RuntimeError("The model has FSDP modules but no FSDP root module exists.")

    # Verify if the model_state_dict and optim_state_dict are valid. This API
    # should give the users an explicit error message to debug or report.
    if info.handle_model and not model_state_dict:
        raise RuntimeError(
            "The option indicates that model state_dict is required to save "
            "or load, but model state_dict is empty."
        )

    if info.handle_optim:
        if not (optim_state_dict and optim_state_dict[STATE]):
            raise RuntimeError(
                "The option indicates that model state_dict is required to save, "
                f"or load but optim state_dict is empty. {optim_state_dict}"
            )

    for key in model_state_dict.keys():
        if FLAT_PARAM in key:
            raise RuntimeError(
                f"{key} contains {FLAT_PARAM}. This can happen if the model "
                "is not the root module."
            )


def _state_dict_fn(obj: Union[nn.Module, torch.optim.Optimizer], api: str) -> Callable:
    call = getattr(obj, api)
    if call in _patched_state_dict:
        call = functools.partial(getattr(obj.__class__, api), self=obj)
    return call


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
            # As we only support FSDP, DDP, and TP, the only case is
            # wrapper-based DDP. Verify the assumption is correct.
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
                    elif key_name == "module":
                        continue
                    else:
                        return False
                return True

            if not verify(key, fqn):
                raise RuntimeError(f"An unexpected key, {key}, exists. FQN is {fqn}")
            state_dict[fqn] = state_dict.pop(key)

    if not info.save_frozen_params:
        for key, param in model.named_parameters():
            if param.requires_grad:
                continue
            fqns = _get_fqns(model, key)
            for fqn in fqns:
                state_dict.pop(fqn)
    return state_dict


def _load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, ValueType],
    info: _StateDictInfo,
) -> None:
    if not info.handle_model:
        return

    for key, _ in model.named_parameters():
        fqns = _get_fqns(model, key)
        fqns_with_ddp_prefix = _get_fqns(model, key, skip_ddp_prefix=False)
        for fqn, fqn_with_ddp_prefix in zip(fqns, fqns_with_ddp_prefix):
            if fqn != fqn_with_ddp_prefix:
                state_dict[fqn_with_ddp_prefix] = state_dict.pop(fqn)

    with info.fsdp_context():
        return _state_dict_fn(model, "load_state_dict")(state_dict)


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    """
    Initialize optim states by calling the step() with zero grads.
    """
    if optim.state:
        # The optimizer state is initialized.
        return

    for param_group in optim.param_groups:
        for param in param_group[PARAMS]:
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
    optim.step(closure=None)
    optim.zero_grad(set_to_none=True)


def _get_optim_state_dict(
    model: nn.Module,
    optimizers: Tuple[torch.optim.Optimizer, ...],
    info: _StateDictInfo,
) -> OptimizerStateType:
    if not info.handle_optim:
        return {}

    optim_state_dict: OptimizerStateType = {STATE: {}, PG: []}
    for optim in optimizers:
        _init_optim_state(optim)
        osd = _state_dict_fn(optim, "state_dict")()
        if info.fsdp_modules:
            with info.fsdp_context():
                osd = FSDP.optim_state_dict(model, optim, osd)
        else:
            params = list(chain.from_iterable(g[PARAMS] for g in optim.param_groups))
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

            for key in list(osd[STATE].keys()):
                fqn = fqn_pid_mapping[key]
                osd[STATE][fqn] = osd[STATE].pop(key)

            for group in osd[PG]:
                group[PARAMS] = [fqn_pid_mapping[pid] for pid in group[PARAMS]]

        cast(DictValueType, optim_state_dict[STATE]).update(osd[STATE])
        cast(ListDictValueType, optim_state_dict[PG]).extend(osd[PG])

    return optim_state_dict


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
    return_osd: OptimizerStateType = {STATE: state, PG: pg_state}
    pg_mapping: Dict[int, int] = {}

    for param_group in optim.param_groups:
        pg_state.append({PARAMS: []})
        for param in param_group[PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                params = pg_state[-1][PARAMS]
                assert isinstance(params, list)
                params.append(fqn)
                if param.requires_grad:
                    state[fqn] = cast(DictValueType, optim_state_dict[STATE])[fqn]
                for loaded_param_group in cast(ListDictValueType, optim_state_dict[PG]):
                    params = loaded_param_group[PARAMS]
                    assert isinstance(params, list)
                    if fqn in params:
                        pg_mapping[id(loaded_param_group)] = len(return_osd[PG]) - 1

    for param_group in cast(ListDictValueType, optim_state_dict[PG]):
        idx = pg_mapping.get(id(param_group), -1)
        if idx == -1:
            continue
        for key, value in param_group.items():
            if key == PARAMS:
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
        optim_state_dict = _split_optim_state_dict(model, optim, state_dict, info)
        if info.fsdp_modules:
            with info.fsdp_context():
                optim_state_dict = FSDP.optim_state_dict_to_load(
                    model, optim, optim_state_dict
                )

        # Note that we do not have to convert the FQN back to param id here if
        # order in optim.param_groups[idx][PARAMS] is the same as the one in
        # optim_state_dict[PG][idx][PARAMS].
        _init_optim_state(optim)
        _state_dict_fn(optim, "load_state_dict")(optim_state_dict)


def state_dict(
    model: nn.Module,
    optimizers: Iterable[torch.optim.Optimizer] = tuple(),
    *,
    model_only: bool = False,
    optim_only: bool = False,
    options: Optional[DistributedStateDictOptions] = None,
) -> Tuple[Dict[str, ValueType], OptimizerStateType]:
    """
    Return the model state_dict and optimizers state_dict.

    ``state_dict`` is a function that can process any module
    that is parallelized by FSDP/fully_shard, DDP/replicate,
    tensor_parallel/parallelize_module,
    and any combination of these parallelisms. The main functions of
    ``state_dict`` are:
            1. Creating a model and optimizer state_dict that can be resharded with
               different workers and/or different parallelisms.
            2. Eliminating the need for users to call parallelism-specific
                    state_dict APIs.
            3. Sanity checking the result state_dict.
    The keys of the result state_dict are the canonical FQNs (Fully Qualified Names).
    A canonical FQN refers to the FQN based on a parameter's position in an
    nn.Module hierarchy. More specifically, a canonical FQN to a parameter is the
    FQN returned by ``module.named_parameters()`` or ``module.named_buffers()``
    when module is not distributed by any parallelisms. Since the optimizer
    internally uses parameter IDs to represent a parameter, a conversion will
    happen to convert the parameter IDs to the canonical FQNs. The value of the
    result state_dict will be either ShardedTensor, DTensor if the value is
    sharded across ranks or Tensor, scalar values if the value is duplicated
    because of DDP/replicate.

    ``state_dict`` can also process a module that is not parallelized.
    In such a case, ``state_dict`` only performs one function --
    converting the optimizer parameter IDs to the canonical FQNs.

    Example:

        import torch
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.distributed.checkpoint.state_dict import state_dict

        fsdp_model = FSDP(copy.deepcopy(model))
        fsdp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        ddp_model = DDP(copy.deepcopy(model))
        ddp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)


        ddp_state_dict, ddp_optim_state_dict = state_dict(ddp_model, ddp_optim)
        fsdp_state_dict, fsdp_optim_state_dict = state_dict(fsdp_model, fsdp_optim)

        # if we simply call ddp_model.state_dict() and fsdp_model.state_dict(),
        # the asserts will fail.
        assert ddp_state_dict == fsdp_state_dict
        assert ddp_optim_state == fsdp_optim_state_dict


    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Iterable[Optimizer]): The optimizers that are used to optimize
            ``model``. Note that optimizers accept multiple optimizers so the type
            is Iterable. If optimizers is empty, the returned optimizer state_dict
            will also be empty.
        model_only (bool): if model_only is True, the returned optimizer
            state_dict will be empty (default: False)

        optim_only (bool): if optim_only is True, the returned model state_dict
            will be empty (default: False)
        options (DistributedStateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `DistributedStateDictOptions` for the details.
    Returns:
        A tuple of state_dict's. The first one is the module  state_dict and the second
        one is the optimizer state_dict. The model state_dict will be empty if
        `optim_only` is True. The optimizer state_dict will be empty if
        `model_only` is True or `optimizers` is empty.
    """
    with gc_context():
        optimizers = tuple(optimizers)
        info = _verify_options(model, optimizers, model_only, optim_only, options)
        model_state_dict = _get_model_state_dict(model, info)
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        _verify_state_dict(model_state_dict, optim_state_dict, info)
        return model_state_dict, optim_state_dict


def load_state_dict(
    model: nn.Module,
    optimizers: Iterable[torch.optim.Optimizer] = tuple(),
    *,
    model_state_dict: Optional[Dict[str, ValueType]] = None,
    optim_state_dict: Optional[OptimizerStateType] = None,
    model_only: bool = False,
    optim_only: bool = False,
    options: Optional[DistributedStateDictOptions] = None,
) -> None:
    """Load the model state_dict and optimizers state_dict.

    The counterpart of ``state_dict`` to load the state_dict
    generated by ``state_dict`` back to the model and optimizers.
    The given ``model_state_dict`` and ``optim_state_dict`` do not have to be
    returned by ``state_dict`` but must meet the following
    conditions:
        1. All FQNs are canonical FQNs as defined in ``state_dict``.
        2. If a tensor is sharded, it must be a ShardedTensor or DTensor.
        3. Optimizer state_dict must contain the canonical FQNs instead of
           parameter IDs.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Iterable[Optimizer]): The optimizers that are used to optimize
            ``model``. Note that optimizers accept multiple optimizers so the typing
            is Iterable. ``optimizers`` can be an empty Iterable.
        model_only (bool): if model_only is True, only the model state_dict will
            be loaded (default: False)
        optim_only (bool): if optim_only is True, only the optimizer state_dict
            will be loaded (default: False)
        options (DistributedStateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `DistributedStateDictOptions` for the details.
    Returns:
        None
    """
    model_state_dict = model_state_dict if model_state_dict else {}
    optim_state_dict = optim_state_dict if optim_state_dict else {}
    with gc_context():
        optimizers = tuple(optimizers)
        info = _verify_options(model, optimizers, model_only, optim_only, options)
        _verify_state_dict(model_state_dict, optim_state_dict, info)
        _load_model_state_dict(model, model_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)


# TODO: correct the state_dict function signature.
# TODO: this API is not yet fully tested. Make it private
@no_type_check
def _patch_model_state_dict(
    model: nn.Module,
    *,
    options: Optional[DistributedStateDictOptions] = None,
) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model`` to
    be a partial function to call ``state_dict``.

    Example:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import patch_model_state_dict

        model = fsdp(model)
        patch_model_state_dict(model)

    Args:
        model (nn.Module): the nn.Module to the model.
        options (DistributedStateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `DistributedStateDictOptions` for the details.
    Returns:
        None
    """

    _state_dict_call = functools.partial(
        state_dict,
        model=model,
        optimizers=tuple(),
        model_only=True,
        options=options,
    )

    def state_dict_call():
        return _state_dict_call()[0]

    model.state_dict = state_dict_call

    _load_state_dict_call = functools.partial(
        load_state_dict,
        model=model,
        optimizers=tuple(),
        model_only=True,
        options=options,
    )

    def load_state_dict_call():
        _load_state_dict_call(state_dict=state_dict)[1]

    model.load_state_dict = load_state_dict_call

    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)


# TODO: correct the load_state_dict function signature.
# TODO: this API is not yet fully tested. Make it private
@no_type_check
def _patch_optimizer_state_dict(
    model: nn.Module,
    optimizers: Tuple[torch.optim.Optimizer, ...],
    *,
    options: Optional[DistributedStateDictOptions] = None,
) -> None:
    """Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers``.

    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers`` to
    be a partial function to call ``state_dict``.

    Note that if there are multiple optimizers, all of the optimizers will be patched.
    So users only need to call one of the state_dict() to get the full result.

    Example:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import patch_model_state_dict

        model = fsdp(model)
        patch_model_state_dict(model)

    Args:
        model (nn.Module): the nn.Module to the model.
        options (DistributedStateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `DistributedStateDictOptions` for the details.
    Returns:
        None
    """

    _state_dict_call = functools.partial(
        state_dict,
        model=model,
        optimizers=optimizers,
        optim_only=True,
        options=options,
    )

    def state_dict_call():
        return _state_dict_call()[1]

    _load_state_dict_call = functools.partial(
        load_state_dict,
        model=model,
        optimizers=optimizers,
        optim_only=True,
        options=options,
    )

    def load_state_dict_call(state_dict: Dict[str, Any]):
        _load_state_dict_call(optim_state_dict=state_dict)

    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)
    for optim in optimizers:
        optim.state_dict = state_dict_call
        optim.load_state_dict = load_state_dict_call
