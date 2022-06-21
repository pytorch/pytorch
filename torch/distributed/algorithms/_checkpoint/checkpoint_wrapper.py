from enum import Enum, auto
from contextlib import suppress

import torch
from torch.autograd.graph import save_on_cpu
from torch.utils.checkpoint import checkpoint
from torch.distributed.utils import _replace_by_prefix
from torch.distributed.fsdp.wrap import _recursive_wrap, lambda_auto_wrap_policy
import torch.nn as nn
from typing import Dict, Any
from functools import partial

_CHECKPOINT_PREFIX = "_checkpoint_wrapped_module"

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
        self._checkpoint_wrapped_module = mod
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

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._checkpoint_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self._checkpoint_wrapped_module.__getitem__(key)  # type: ignore[operator]

    def forward(self, *args, **kwargs):
        offload_mgr = save_on_cpu(pin_memory=True) if self.offload_to_cpu else suppress()
        with offload_mgr:  # type: ignore[attr-defined]
            return checkpoint(
                self._checkpoint_wrapped_module,
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

    return CheckpointWrapper(module, checkpoint_impl, offload_to_cpu)


def apply_activation_checkpointing_wrapper(
    model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda _: True
):
    """
    Applies :func:`checkpoint_wrapper` to modules within `model` based on a user-defined
    configuration. For each module within `model`, the `check_fn` is used to decide
    whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

    Note::
        This function modifies `model` in place and replaces appropriate layers with
        their checkpoint-wrapped modules.
    Note::
        This function will not wrap the overall root module. If this is needed, please directly use
        :class:`CheckpointWrapper`.
    Usage::
        model = nn.Sequential(
            nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
        )
        check_fn = lambda l: isinstance(l, nn.Linear)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
    Args:
        module (nn.Module):
            The model who's submodules (or self) should be wrapped with activation checkpointing.
        checkpoint_wrapper_fn (Optional[Callable[nn.Module]])
            A `Callable` which will wrap modules
        check_fn (Optional[Callable[nn.Module, nn.Module]])
            A lambda function which will be passed current layer and returns
            ``True`` or ``False`` depending on whether input layer should be wrapped.
    Returns: None (`model` is modified inplace)
    """
    return _recursive_wrap(
        module=model,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
        wrapper_cls=checkpoint_wrapper_fn,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True
    )
