from enum import Enum, auto
from contextlib import suppress

import torch
from torch.autograd.graph import save_on_cpu
from torch.utils.checkpoint import checkpoint
from torch.distributed.utils import _replace_by_prefix
import torch.nn as nn
from typing import Dict, Any

_CHECKPOINT_PREFIX = "mod"


class CheckpointImpl(Enum):
    REENTRANT = auto()
    NO_REENTRANT = auto()


class CheckpointWrapper(torch.nn.Module):
    """
    An nn.Module that wraps another nn.Module with checkpointing.
    """
    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT,
        offload_to_cpu: bool = False,
    ):
        super().__init__()
        self.mod = mod
        self.checkpoint_impl = checkpoint_impl
        self.offload_to_cpu = offload_to_cpu
        # state_dict post hook to remove prefix to allow loading into a
        # non-checkpoint wrapped module.
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to allow loading back into
        # checkpoint-wrapped module.
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )

    def forward(self, *args, **kwargs):
        offload_mgr = save_on_cpu(pin_memory=True) if self.offload_to_cpu else suppress()
        with offload_mgr:  # type: ignore[attr-defined]
            return checkpoint(
                self.mod,
                use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
                *args,
                **kwargs,
            )

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> Dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this
        FSDP module is executed. For ``checkpoint_wrapper``, it will strip
        checkpoint-wrapped module prefix so that this module can be loaded into
        non-checkpointed modules. It would still be able to be loaded into
        checkpoint-wrapped modules as this class adds the prefix back before
        loading the state_dict.
        """
        _replace_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}.", prefix)
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()``
        is called. For ``checkpoint_wrapper``, it will add back the module
        prefix so that non-checkpointed modules can be loaded into
        checkpoint_wrapper modules properly.
        """
        _replace_by_prefix(state_dict, prefix, prefix + f"{_CHECKPOINT_PREFIX}.")


def checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT,
    offload_to_cpu: bool = False,
) -> torch.nn.Module:
    """
    A convenience wrapper for activation checkpointing. If the module is wrapped
    with this function, all subsequent calls to the module will automatically
    perform checkpointing without the user having to explicitly call ``checkpoint``
    function.
    Usage::
        checkpointed_module = checkpoint_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
        checkpoint_impl (Optional[CheckpointImpl]):
            The checkpointing implementation to use. Currently only
            CheckpointImpl.REENTRANT is supported.
        offload_to_cpu (Optional[bool]):
            Whether to offload outer activations to CPU. Note that this
            currently only works with CheckpointImpl.REENTRANT.

    Returns:
        (nn.Module):
            Wrapped module
    """
    # saved tensor hooks based-checkpoint wrapper is not yet supported.
    if checkpoint_impl == CheckpointImpl.NO_REENTRANT:
        raise ValueError(
            "No support for non-reentrant based checkpoint implementation."
        )

    if offload_to_cpu and checkpoint_impl != CheckpointImpl.REENTRANT:
        raise ValueError(
            "No support for CPU offload activations and non-reentrant based "
            "checkpoint implementation."
        )

    return CheckpointWrapper(module, checkpoint_impl, offload_to_cpu)
