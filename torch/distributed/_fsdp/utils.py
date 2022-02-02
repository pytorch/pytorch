from typing import Dict, List, Tuple, Union, Any, Callable, Set
from enum import Enum, auto

import torch


"""Useful functions to deal with tensor types with other python container types."""

class TrainingState_(Enum):
    """
    Simple enum to indicate what state FSDP is in. Used for asserting
    to make sure APIs are called in the correct state.
    ..note::
        ``BACKWARD_PRE`` and ``BACKWARD_POST`` states are used to ensure we
        receives backward hooks in the correct order. It is used to catch
        unexpected order of hooks being called (likely due to our
        hook registration logic or autograd engine logic changes).
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()

def _apply_to_tensors(
    fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set]
) -> Any:
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x: Union[torch.Tensor, Dict, List, Tuple, Set]) -> Any:
        if torch.is_tensor(x):
            return fn(x)
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)

def _replace_by_prefix(
    state_dict: Dict[str, Any], old_prefix: str, new_prefix: str,
):
    """
        Replace all keys that match a given old_prefix with a new_prefix (in-place).
        Usage::
            state_dict = {"layer.xyz": torch.tensor(1)}
            replace_by_prefix_(state_dict, "layer.", "module.layer.")
            assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

def post_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix,
    *args
) -> Dict[str, Any]:
    """
    Hook that runs after model.state_dict() is called before returning result to
    user. For FSDP, we may have to clone the tensors in state_dict as params go
    back to sharded version after summon_full_params ends, and also remove
    "_fsdp_wrapped_module" prefix.
    """
    for key in state_dict.keys():
        # Due to recursive call of summon_full_params, avoid unnecessary reclone of
        if (
            module.training_state == TrainingState_.SUMMON_FULL_PARAMS and
            not getattr(state_dict[key], "_has_been_cloned", False)
        ):
            state_dict[key] = state_dict[key].clone()
            state_dict[key]._has_been_cloned = True
    # TODO: remove prefix?
    _replace_by_prefix(state_dict, prefix + "_fsdp_wrapped_module.", prefix)
    return state_dict

def pre_load_state_dict_hook(state_dict, prefix, *args) -> None:
    """
    Hook that runs before load_state_dict() call to model. For FSDP, we
    resubstitute the "_fsdp_wrapped_module" prefix so that FSDP instances can
    correctly load state_dicts. Removing and addied back the prefixes also helps
    extend FSDP state_dict to be loaded by non-FSDP instances in the future.
    """
    _replace_by_prefix(state_dict, prefix, prefix + "_fsdp_wrapped_module.")
