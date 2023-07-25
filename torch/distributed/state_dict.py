import functools
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp._common_utils import FSDP_WRAPPED_MODULE
from torch.nn.parallel import DistributedDataParallel as DDP


FLAT_PARAM = "_flat_param"
PG = "param_groups"


@dataclass
class DistributedStateDictOptions:
    # The default should be sharded_state_dict
    fsdp_state_dict_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    save_to_cpu: bool = True
    # Do not save parameters that requires_grad is False.
    no_return_frozen_parameters: bool = False


FQNS_T = Set[str]


@dataclass
class StateDictInfo(DistributedStateDictOptions):
    fqn_param_mapping: Dict[
        Union[FQNS_T, torch.Tensor], Union[FQNS_T, torch.Tensor]
    ] = field(default_factory=dict)
    handle_model: bool = True
    handle_optim: bool = True


def _get_fqns(model: nn.Module, name: str, skip_ddp_prefix: bool = True) -> FQNS_T:
    """
    This API is used to convert a name of a parameter to the FQNs. The type of
    the returned FQNs is a set of string. For FSDP case, a FlatParameter may
    contain multiple original parameters, hence multiple FQNs.

    Args:
        module (nn.Module): the root model.
        name (str): the name
        skip_ddp_prefix (bool): whether to skip DDP's `module` prefix

    Returns:
        The canonical FQNs based on the model traversal.
    """
    if "." not in name:
        return set([name])

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
                return set(f"{prefix}{fqn}" for fqn in flat_param._fqns)
            curr_obj = getattr(curr_obj, FSDP_WRAPPED_MODULE)
            if curr_obj_name != FSDP_WRAPPED_MODULE:
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)
        else:
            fqn_obj_names.append(curr_obj_name)
            curr_obj = getattr(curr_obj, curr_obj_name)

    return set([".".join(fqn_obj_names)])


def _verify_options(
    model: nn.Module,
    optims: Tuple[torch.optim.Optimizer],
    model_only: bool,
    optim_only: bool,
    options: Optional[DistributedStateDictOptions] = None,
) -> StateDictInfo:
    """
    Verify the model and options passed by the user and generates StateDictInfo.
    """
    # Initialize StateDictInfo
    fqn_param_mapping: Dict[
        Union[str, torch.Tensor], Union[Set[str], torch.Tensor]
    ] = {}
    for name, param in model.named_parameters():
        fqns = _get_fqns(model, name)
        fqn_param_mapping[param] = fqns
        for fqn in fqns:
            fqn_param_mapping[fqn] = param

    if options is not None:
        info = StateDictInfo(**asdict(options), fqn_param_mapping=fqn_param_mapping)
    else:
        info = StateDictInfo(fqn_param_mapping=fqn_param_mapping)

    if model_only and optim_only:
        raise RuntimeError(
            "Both model_only and optim_only are set, which one do you need?"
        )
    if optim_only and not optims:
        raise RuntimeError(
            "Optimizers are not passed in but optim_only is set to True."
        )

    info.handle_model = model_only or not optim_only
    info.handle_optim = optim_only or (not model_only and not optims)

    # TODO: verify the model setting
    # Traverse the model and if FSDP/fully_shard exist, then the root FSDP module
    # must be in the model too.

    # TODO: verify options
    return info


def _verify_state_dict(
    model_state_dict: Dict[str, Any],
    optim_state_dict: Dict[str, Any],
    info: StateDictInfo,
) -> None:
    """
    Verify if the model_state_dict and optim_state_dict are valid. This API
    should give the users an explicit error message to debug or report.
    """
    if info.handle_model and not model_state_dict:
        raise RuntimeError(
            "The option indicates that model state_dict is required to save "
            "or load, but model state_dict is empty."
        )

    if info.handle_optim and (not optim_state_dict or not optim_state_dict["state"]):
        raise RuntimeError(
            "The option indicates that model state_dict is required to save, "
            f"or load but optim state_dict is empty. {optim_state_dict}"
        )

    for key, param in model_state_dict.items():
        if FLAT_PARAM in key:
            raise RuntimeError(
                f"{key} contains {FLAT_PARAM}. This can happen if the model "
                "is not the root module."
            )


def _get_model_state_dict(model: nn.Module, info: StateDictInfo) -> Dict[str, Any]:
    fsdp_modules = FSDP.fsdp_modules(model)
    if fsdp_modules:
        # FSDP API only work if at least one FSDP instance exists.
        with FSDP.state_dict_type(model, info.fsdp_state_dict_type):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

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

    if info.no_return_frozen_parameters:
        for key, param in model.named_parameters():
            if param.requires_grad:
                continue
            fqns = _get_fqns(model, key)
            for fqn in fqns:
                state_dict.pop(fqn)
    return state_dict


def _load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, Any],
    info: StateDictInfo,
) -> Dict[str, Any]:
    for key, _ in model.named_parameters():
        fqns = _get_fqns(model, key)
        fqns_with_ddp_prefix = _get_fqns(model, key, skip_ddp_prefix=False)
        for fqn, fqn_with_ddp_prefix in zip(fqns, fqns_with_ddp_prefix):
            if fqn != fqn_with_ddp_prefix:
                state_dict[fqn_with_ddp_prefix] = state_dict.pop(fqn)

    fsdp_modules = FSDP.fsdp_modules(model)
    if fsdp_modules:
        with FSDP.state_dict_type(model, info.fsdp_state_dict_type):
            return model.load_state_dict(state_dict)
    else:
        return model.load_state_dict(state_dict)


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    """
    Initialize optim states by using a step with zero grads.
    """
    if optim.state:
        # The optimizer state is initialized.
        return

    for param_group in optim.param_groups:
        for param in param_group["params"]:
            if param.requires_grad:
                grad = torch.zeros_like(param)
                param.grad = torch.autograd.Variable(grad)
    optim.step(closure=None)
    optim.zero_grad(set_to_none=True)


def _get_optim_state_dict(
    model: nn.Module,
    optims: Tuple[torch.optim.Optimizer],
    info: StateDictInfo,
) -> Dict[str, Any]:
    optim_state_dict = {"state": {}, PG: []}
    fsdp_modules = FSDP.fsdp_modules(model)
    for optim in optims:
        _init_optim_state(optim)
        osd = optim.state_dict()
        if fsdp_modules:
            with FSDP.state_dict_type(model, info.fsdp_state_dict_type):
                osd = FSDP.optim_state_dict(model, optim, osd)
        else:
            params = list(chain.from_iterable(g["params"] for g in optim.param_groups))
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

            for key in list(osd["state"].keys()):
                fqn = fqn_pid_mapping[key]
                osd["state"][fqn] = osd["state"].pop(key)

            for group in osd[PG]:
                group["params"] = [fqn_pid_mapping[pid] for pid in group["params"]]

        optim_state_dict["state"].update(osd["state"])
        optim_state_dict[PG].extend(osd[PG])

    return optim_state_dict


def _split_optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: Dict[str, Any],
    info: StateDictInfo,
) -> Dict[str, Any]:
    """
    Extract the corresponding optim state_dict from ``optim_state_dict`` for
    ``optim`` and return the result optim state_dict.

    Args:
        model (nn.Module): the root model.
        optim (torch.optim.Optimizer): the optimizer.
        optim_state_dict (Dict[str, Any]): the superset optim state_dict that
            contains the optim state_dict of ``optim``.
        info (StateDictInfo): state dict information.

    Returns:
        The optim state_dict of ``optim``.
    """

    return_osd = {"state": {}, PG: []}
    param_group_ids = set()

    for param_group in optim.param_groups:
        for param in param_group["params"]:
            if not param.requires_grad:
                continue
            for fqn in info.fqn_param_mapping[param]:
                return_osd["state"][fqn] = optim_state_dict["state"][fqn]
                for loaded_param_group in optim_state_dict[PG]:
                    if fqn in loaded_param_group["params"]:
                        param_group_ids.add(id(loaded_param_group))

    for param_group in optim_state_dict[PG]:
        if id(param_group) in param_group_ids:
            return_osd[PG].append(param_group)
    return return_osd


def _load_optim_state_dict(
    model: nn.Module,
    optims: Tuple[torch.optim.Optimizer],
    state_dict: Dict[str, Any],
    info: StateDictInfo,
) -> None:
    fsdp_modules = FSDP.fsdp_modules(model)
    for optim in optims:
        optim_state_dict = _split_optim_state_dict(model, optim, state_dict, info)
        if fsdp_modules:
            with FSDP.state_dict_type(model, info.fsdp_state_dict_type):
                optim_state_dict = FSDP.optim_state_dict_to_load(
                    model, optim, optim_state_dict
                )

        # Note that we do not have to convert the FQN back to param id here if
        # the optim is initizlied by the `_init_optim_state()`. The way
        # torch.optim.Optimizer.load_state_dict() is able to directly map
        # the FQN to param id by using the order saved in the param group.
        _init_optim_state(optim)
        optim.load_state_dict(optim_state_dict)


def distributed_state_dict(
    model: nn.Module,
    optims: Iterable[torch.optim.Optimizer] = tuple(),
    *,
    model_only: bool = False,
    optim_only: bool = False,
    options: Optional[DistributedStateDictOptions] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    TODO: document
    """
    optims = tuple(optims)
    info = _verify_options(model, optims, model_only, optim_only, options)
    model_state_dict = _get_model_state_dict(model, info)
    optim_state_dict = _get_optim_state_dict(model, optims, info)
    _verify_state_dict(model_state_dict, optim_state_dict, info)
    return model_state_dict, optim_state_dict


def distributed_load_state_dict(
    model: nn.Module,
    optims: Iterable[torch.optim.Optimizer] = tuple(),
    *,
    model_state_dict: Dict[str, Any] = {},
    optim_state_dict: Dict[str, Any] = {},
    model_only: bool = False,
    optim_only: bool = False,
    options: Optional[DistributedStateDictOptions] = None,
) -> None:
    """
    TODO: document
    """
    optims = tuple(optims)
    info = _verify_options(model, optims, model_only, optim_only, options)
    _verify_state_dict(model_state_dict, optim_state_dict, info)
    _load_model_state_dict(model, model_state_dict, info)
    _load_optim_state_dict(model, optims, optim_state_dict, info)


def patch_model_state_dict(
    model: nn.Module,
    *,
    options: Optional[DistributedStateDictOptions] = None,
) -> None:
    model.state_dict = functools.partial(
        distributed_state_dict,
        model=model,
        optims=tuple(),
        model_only=True,
        options=options,
    )

    model.load_state_dict = functools.partial(
        distributed_load_state_dict,
        model=model,
        optims=tuple(),
        model_only=True,
        options=options,
    )


def patch_optimizer_state_dict(
    model: nn.Module,
    optims: Tuple[torch.optim.Optimizer],
    *,
    options: Optional[DistributedStateDictOptions] = None,
) -> None:
    for optim in optims:
        optim.state_dict = functools.partial(
            distributed_state_dict,
            model=model,
            optims=optims,
            optim_only=True,
            options=options,
        )

        optim_load_state_dict = functools.partial(
            distributed_load_state_dict,
            model=model,
            optims=optims,
            optim_only=True,
            options=options,
        )
        model.load_state_dict = lambda state_dict: optim_load_state_dict(
            optim_state_dict=state_dict
        )
