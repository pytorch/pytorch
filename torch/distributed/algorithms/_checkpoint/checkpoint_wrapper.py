import warnings
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint

_CHECKPOINT_WRAPPED_MODULE = "_checkpoint_wrapped_module"
_CHECKPOINT_PREFIX = _CHECKPOINT_WRAPPED_MODULE + "."


class CheckpointImpl(Enum):
    REENTRANT = auto()
    NO_REENTRANT = auto()


class ActivationWrapper(torch.nn.Module):
    """
    Base class for Activation Checkpoint and Activation Offload.

    Not meant to be instantiated directly.
    """

    def __init__(self, mod):
        super().__init__()
        self._checkpoint_wrapped_module = mod
        # state_dict post hook to remove prefix to allow loading into a
        # non-checkpoint wrapped module.
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to allow loading back into
        # checkpoint-wrapped module.
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )

    def forward(self, *args, **kwargs):
        raise ValueError("Subclasses should implement forward().")

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._checkpoint_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self._checkpoint_wrapped_module.__getitem__(key)  # type: ignore[operator]

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Override :meth:`named_parameters()` to intercept parameter names.

        remove all occurrences of ``_CHECKPOINT_PREFIX``.
        """
        for param_name, param in super().named_parameters(*args, **kwargs):
            yield param_name.replace(_CHECKPOINT_PREFIX, ""), param

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> Dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this FSDP module is executed.

        For ``checkpoint_wrapper``, it will strip checkpoint-wrapped module prefix,
        so that this module can be loaded into non-checkpointed modules.
        It would still be able to be loaded into checkpoint-wrapped modules as this class,
        adds the prefix back before loading the state_dict.
        """
        _replace_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}", prefix)
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()`` is called.

        For ``checkpoint_wrapper``, it will add back the module
        prefix so that non-checkpointed modules can be loaded into
        checkpoint_wrapper modules properly.
        """
        _replace_by_prefix(state_dict, prefix, prefix + f"{_CHECKPOINT_PREFIX}")


class OffloadWrapper(ActivationWrapper):
    def __init__(self, mod):
        super().__init__(mod)

    def forward(self, *args, **kwargs):
        with save_on_cpu(pin_memory=True):
            return self._checkpoint_wrapped_module(*args, **kwargs)


class CheckpointWrapper(ActivationWrapper):
    """
    An ``nn.Module`` that wraps another ``nn.Module`` with checkpointing.

    Note that this module is not meant to be used directly but instead,
    it is to be used through the ``checkpoint_wrapper`` function.
    """

    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
        checkpoint_fn=None,
        **checkpoint_fn_kwargs,
    ):
        super().__init__(mod)
        self.checkpoint_impl = checkpoint_impl
        if checkpoint_fn is None:
            # use torch.utils.checkpoint
            self.checkpoint_fn = partial(
                torch_utils_checkpoint,
                use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
                **checkpoint_fn_kwargs,
            )
        else:
            # Construct user-specified checkpoint function.
            self.checkpoint_fn = partial(
                checkpoint_fn,
                **checkpoint_fn_kwargs,
            )

    def forward(self, *args, **kwargs):
        # Support keyword arguments for reentrant checkpoint. Note that this
        # only works if user has specified self.checkpoint_impl and is not
        # using their own custom checkpoint_fn.
        if self.checkpoint_impl == CheckpointImpl.REENTRANT and kwargs != {}:
            # Pack the args and kwargs
            flat_args, kwarg_keys = _pack_kwargs(*args, **kwargs)

            # Function that only takes (packed) args, but can unpack them
            # into the original args and kwargs for the checkpointed
            # function, and runs that function.
            def my_function(*inputs):
                # unpack back into args and kwargs
                unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
                # run original module
                return self._checkpoint_wrapped_module(
                    *unpacked_args, **unpacked_kwargs
                )

            # Pass the function that only takes packed args into reentrant
            # checkpoint API.
            return self.checkpoint_fn(  # type: ignore[misc]
                my_function,
                *flat_args,
            )
        else:
            return self.checkpoint_fn(  # type: ignore[misc]
                self._checkpoint_wrapped_module, *args, **kwargs
            )


def offload_wrapper(module: torch.nn.Module) -> torch.nn.Module:
    """
    Wrap a module for activation offloading to CPU.

    Offloads intermediate activations to the CPU for modules wrapped with this function.
    Wrappers with activation offload can be composed with ones that do recomputation-based
    checkpoint to trade off increased compute versus increased CPU
    memory usage and additional H2D transfers.

    Usage::
        offloaded_module = offload_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
    Returns:
        (nn.Module):
            Wrapped module
    """
    return OffloadWrapper(module)


def checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
    checkpoint_fn=None,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    """
    Wrap a module for activation checkpointing.

    If the module is wrapped with this function, all subsequent calls to the module will,
    automatically perform checkpointing without the user having to explicitly call ``checkpoint`` function.

    Usage::
        checkpointed_module = checkpoint_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
        checkpoint_impl (Optional[CheckpointImpl]):
            The checkpointing implementation to use. Note that this will only
            be passed into the ``torch.utils.checkpoint.checkpoint``
            implementation, and is ignored if a custom ``checkpoint_fn`` is
            specified. Note that for implementations using reentrant checkpoint
            from ``torch.utils.checkpoint``, keyword arguments will only be
            supported if ``checkpoint_impl`` is passed as ``CheckpointImpl.REENTRANT`.
        checkpoint_fn (Optional[Callable]):
            Functional checkpoint implementation to use. If this is specified,
            it will be used over the default ``torch.utils.checkpoint.checkpoint``
            implementation and the `checkpoint_impl` argument will be ignored.
        **checkpoint_fn_kwargs: (Dict[str, Any]): Keyword arguments to pass into `checkpoint_fn`.

    Returns:
        (nn.Module):
            Wrapped module
    """

    if checkpoint_impl == CheckpointImpl.REENTRANT:
        warnings.warn(
            f"Please specify {CheckpointImpl.NO_REENTRANT} as "
            f"{CheckpointImpl.REENTRANT} will soon be removed as "
            "the default and eventually deprecated.",
            stacklevel=1,
        )
    return CheckpointWrapper(
        module,
        checkpoint_impl,
        checkpoint_fn,
        **checkpoint_fn_kwargs,
    )


def apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda _: True,
    auto_wrap_policy: Optional[Callable[[nn.Module, bool, int], bool]] = None,
):
    """
    Apply :func:`checkpoint_wrapper` to modules within `model` based on a user-defined configuration.

    For each module within `model`, the `check_fn` is used to decide
    whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

    Note::
        This function modifies `model` in place and replaces appropriate layers with
        their checkpoint-wrapped modules.
    Note::
        This function will not wrap the overall root module. If this is needed, please directly use
        :func:`checkpoint_wrapper` or :func:`offload_wrapper`.
    Usage::
        model = nn.Sequential(
            nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
        )
        check_fn = lambda l: isinstance(l, nn.Linear)
        # checkpoint activations
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
        # Or offload activations to CPU
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=offload_wrapper, check_fn=check_fn)
    Args:
        model (nn.Module):
            The model whose submodules should be wrapped with activation checkpointing.
        checkpoint_wrapper_fn (Optional[Callable[nn.Module]])
            A ``Callable`` which will wrap modules
        check_fn (Optional[Callable[nn.Module, nn.Module]])
            A lambda function which will be passed each child submodule of ``model`` and returns
            ``True`` or ``False`` depending on whether the submodule should be wrapped.
        auto_wrap_policy (Optional[Callable[[nn.Module, bool, int], bool]]): A policy to wrap model's
            submodules with AC. Note that if this is specified, it takes precedence over ``check_fn``.
    Returns: None (`model` is modified inplace)
    """
    # TODO: Importing inside function to avoid circular import issue between FSDP and
    # checkpoint_wrapper. This can be resolved once wrap() APIs are decoupled from FSDP code.
    from torch.distributed.fsdp.wrap import _recursive_wrap, lambda_auto_wrap_policy, _Policy
    from torch.distributed.fsdp._wrap_utils import _construct_wrap_fn, _post_order_apply

    policy = (
        auto_wrap_policy
        if auto_wrap_policy is not None
        else partial(lambda_auto_wrap_policy, lambda_fn=check_fn)
    )
    if not callable(policy):
        if not isinstance(policy, _Policy):
            raise ValueError(
                f"Expected {policy} to be callable or be a pre-defined wrap policy"
            )
        target_module_to_kwargs = policy._run_policy(
            model, ignored_modules=set(), root_kwargs={}
        )
        wrap_fn = _construct_wrap_fn(model, target_module_to_kwargs, checkpoint_wrapper_fn)
        _post_order_apply(model, wrap_fn)
        return

    _recursive_wrap(
        module=model,
        auto_wrap_policy=policy,  # type: ignore[arg-type]
        wrapper_cls=checkpoint_wrapper_fn,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True,
    )
