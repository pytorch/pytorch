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
from torch.distributed.utils import _apply_to_modules

FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"
FSDP_PREFIX = FSDP_WRAPPED_MODULE + "."
FSDP_FLATTENED = "_fsdp_flattened"


class FSDPState:
    """
    This encompasses all FSDP state.
    """


# We leverage Python's dynamic attribute definition to unify the state
# management for the wrapper and non-wrapper approaches.
_State = Union[nn.Module, FSDPState]


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


def _is_composable(state: _State):
    # TODO: This is a temporary hack for differentiate between code paths.
    return not isinstance(state, nn.Module)


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


def _get_param_to_unflat_param_names(
    model: torch.nn.Module,
    dedup_shared_params: bool = True,
) -> Dict[torch.nn.Parameter, List[str]]:
    """
    Constructs a mapping from flattened parameter (including non-FSDP-module
    parameters) to its unflattened parameter names. For non-FSDP-module
    parameters, these mapped-to lists always contain a single element. The
    unflattened parameter names should match the keys of the model state dict.

    For shared parameters, only the first parameter name is included (following
    the ``torch.nn.Module.parameters()`` order).

    Args:
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance).
        dedup_shared_params (bool): If ``True``, only includes the first
            list of unflattened parameter names corresponding to a parameter
            in the module walk order; if ``False``, then includes all of the
            unflattened parameter names.
    """

    def module_fn(module, prefix, param_to_unflat_param_names):
        for param_name, param in module.named_parameters(recurse=False):
            module_prefixed_param_names = (
                param._fqns
                if type(param) is flat_param_file.FlatParameter
                else [param_name]
            )  # prefixed from `module`
            fully_prefixed_param_names = [
                clean_tensor_name(prefix + name) for name in module_prefixed_param_names
            ]  # fully prefixed from the top level including `prefix`
            # If this parameter has already been visited, then it is a
            # shared parameter; then, only take the first parameter name
            is_shared_param = param in param_to_unflat_param_names
            if not is_shared_param:
                param_to_unflat_param_names[param] = fully_prefixed_param_names
            elif not dedup_shared_params:
                param_to_unflat_param_names[param].extend(fully_prefixed_param_names)

    def return_fn(param_to_unflat_param_names):
        return param_to_unflat_param_names

    param_to_unflat_param_names: Dict[torch.nn.Parameter, List[str]] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        param_to_unflat_param_names,
    )

@no_type_check
def _assert_in_training_states(
    state: _State,
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
