"""
This file includes private common utilities for FSDP.
"""

import traceback
from enum import auto, Enum
from typing import Callable, Dict, List, no_type_check, Union

import torch
import torch.distributed.fsdp.flat_param as flat_param_file
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
)

FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"
FSDP_PREFIX = FSDP_WRAPPED_MODULE + "."
FSDP_FLATTENED = "_fsdp_flattened"


class ComposableFSDPState:
    """
    This encompasses all FSDP state for composable FSDP.
    """


# We leverage Python's dynamic attribute definition to unify the state
# management for the wrapper and non-wrapper approaches.
_FSDPState = Union[nn.Module, ComposableFSDPState]


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
def _all_handles(state: _FSDPState) -> List:
    return (
        state._handles
        if _is_composable(state)
        else state._fsdp_handles(state)  # `FullyShardedDataParallel`
    )


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

    def module_fn(module, prefix, param_to_fqns):
        for param_name, param in module.named_parameters(recurse=False):
            local_fqns = (
                param._fqns
                if type(param) is flat_param_file.FlatParameter
                else [param_name]
            )  # prefixed from `module`
            global_fqns = [
                clean_tensor_name(prefix + name) for name in local_fqns
            ]  # prefixed from the top level `model` (i.e. including `prefix`)
            is_shared_param = param in param_to_fqns
            if not is_shared_param:
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
        param_to_unflat_param_names,
    )


def _apply_to_modules(
    root_module: torch.nn.Module,
    module_fn: Callable,
    return_fn: Callable,
    *args,
    **kwargs,
):
    """
    Performs a pre-order traversal of the modules in the hierarchy rooted at
    ``root_module``, applying ``module_fn`` at each module and finally
    returning a value using ``return_fn``. The traversal constructs the full
    module prefix name (e.g. "module.submodule." just like in model state dict)
    and makes that available to ``module_fn``.
    """

    def f(module: torch.nn.Module, prefix: str, *args, **kwargs):
        # Call the module function before recursing over children (pre-order)
        module_fn(module, prefix, *args, **kwargs)
        for submodule_name, submodule in module.named_children():
            if submodule is not None:
                new_prefix = prefix + submodule_name + "."
                f(submodule, new_prefix, *args, **kwargs)

    f(root_module, "", *args, **kwargs)
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
