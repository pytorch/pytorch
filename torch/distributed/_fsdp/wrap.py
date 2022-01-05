# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, cast

import torch.nn as nn


def default_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    # These are customizable for this default policy function.
    min_num_params: int = int(1e8),
    force_leaf_modules: Optional[Set[Type[nn.Module]]] = None,
    exclude_wrap_modules: Optional[Set[Type[nn.Module]]] = None,
) -> bool:
    """Default policy function for :func:`auto_wrap`.

       Return if a module should be wrapped during :func:`auto_wrap`.

       The first three parameters are used by :func:`auto_wrap`. If
       you write a custom version of this policy function, your version
       needs to at least accept the first three parameters and free
       to do whatever you want in the function.

    Args:
       module (nn.Module):
           The module to be considered in this decision.
       recurse (bool):
           Indicate if this is called to make a decision on whether we
           should recurse down a subgraph of the module structure.
           If False, it means this function is called to make a decision
           on whether we should wrap the said module.
       unwrapped_params (int):
           The number of parameters yet to be wrapped in this module.

       min_num_params (int):
           Customizable policy input. It controls the size threshold
           on how big should a module be to be considered wrapped.
       force_leaf_modules (Set[Type[nn.Module]]): set of module types to
           keep as leaves, i.e., their children will never be wrapped.
       exclude_wrap_modules (Set[Type[nn.Module]]):
           Customizable set of module types to be excluded in wrapping.
    """
    force_leaf_modules = (
        default_auto_wrap_policy.FORCE_LEAF_MODULES  # type: ignore[attr-defined]
        if force_leaf_modules is None
        else force_leaf_modules
    )
    exclude_wrap_modules = (
        default_auto_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore[attr-defined]
        if exclude_wrap_modules is None
        else exclude_wrap_modules
    )

    is_large = unwrapped_params >= min_num_params
    if recurse:
        # We should recurse if the module is big enough but not in force_leaf_modules list.
        return is_large and not isinstance(module, tuple(force_leaf_modules))
    else:
        # If we are not recursing, determine if we should wrap.
        return is_large and not isinstance(module, tuple(exclude_wrap_modules))


# Set those defaults to the default_auto_wrap_policy function. Make them easy to be imported.
default_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {nn.ModuleList, nn.ModuleDict}  # type: ignore[attr-defined]
default_auto_wrap_policy.FORCE_LEAF_MODULES = {nn.MultiheadAttention}  # type: ignore[attr-defined]


@contextlib.contextmanager
def enable_wrap(
    *, wrapper_cls: Any, **wrapper_kwargs: Any
) -> Generator[None, None, None]:
    """
    Context manager to wrap modules using a wrapper.

    Useful for when you'd like to apply the same parameters to all child modules
    that you wrap. A particularly important use case is wrapping large layers so
    that they get sharded (in-place) during initialization, to avoid running out of
    system memory. Large layers can indicate that they should be sharded via
    the ``wrap`` annotation and this context manager can provide the
    exact configuration for these nested instances.

    Usage::

        with enable_wrap(wrapper_cls, **params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        wrapper_cls:
            Class that `wrap` annotation will `wrap` modules with, such as
            `FullyShardedDataParallel`.
        **wrapper_kwargs:
            Configuration settings that will be passed to all ``wrap``
            instances inside the context
    """
    kwargs = {
        **{"wrapper_cls": wrapper_cls},
        **wrapper_kwargs,
    }
    with ConfigAutoWrap(**kwargs):
        yield


def wrap(module: nn.Module, **wrap_overrides: Any) -> nn.Module:
    """
    Annotate that a module should be wrapped. Annotated modules will only be
    wrapped if inside of an :func:`enable_wrap` context manager. This allows
    a module to be initialized both with and without a wrapper without code
    change.

    The class that this function wraps the passed in ``nn.Module`` with is the
    passed in ``wrapper_cls`` argument into ``enable_wrap``. Both
    ``enable_wrap`` and ``wrap`` can take in kwargs specifying how to construct
    the ``wrapper_cls`` instance. In the case of duplicate kwargs in
    ``enable_wrap`` and ``wrap``, the argument passed into ``wrap`` will be
    respected.

    Usage::

        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        module (nn.Module): module to wrap (if in :func:`enable_wrap` context)
        **wrap_overrides: configuration overrides that will take priority over
            the values provided by the :func:`enable_wrap` context
    """
    if ConfigAutoWrap.in_autowrap_context:
        assert ConfigAutoWrap.wrapper_cls is not None

        wrap_overrides = {**ConfigAutoWrap.kwargs, **wrap_overrides}
        return _wrap(
            module,
            ConfigAutoWrap.wrapper_cls,
            **wrap_overrides,
        )
    return module


def _wrap(module: nn.Module, wrapper_cls: Callable, **kwargs) -> nn.Module:
    assert wrapper_cls is not None
    return wrapper_cls(module, **kwargs)


def _recursive_wrap(
    module: nn.Module,
    auto_wrap_policy: Callable,
    wrapper_cls: Callable,
    only_wrap_children: bool = False,
    **kwargs: Any
) -> Tuple[nn.Module, int]:
    """
    Automatically wrap child modules of *module* that meet the given
    criteria with :func:`auto_wrap`. Does not rely on ConfigAutoWrap.
    Args:
        module (nn.Module):
            module to recursively wrap
        auto_wrap_policy (Callable):
            A callable specifying a policy to recursively wrap layers with FSDP.
    Returns:
        (nn.Module, int):
            Wrapped module and the number parameters wrapped recursively.
    """
    assert auto_wrap_policy is not None, "Must specify auto_wrap_policy."
    assert wrapper_cls is not None, "Must specify wrapper_cls"
    # Make sure no child is already wrapped.
    for _, child in module.named_modules():
        assert not isinstance(child, cast(type, wrapper_cls))

    # We count all params, assuming none of them is already wrapped.
    num_params = sum([p.numel() for p in module.parameters()])

    assert auto_wrap_policy is not None
    if auto_wrap_policy(module=module, recurse=True, unwrapped_params=num_params):
        total_wrapped_params = 0
        # Iterate through the children, recursively wrap if necessary
        for name, child in module.named_children():
            wrapped_child, num_wrapped_params = _recursive_wrap(
                module=child,
                auto_wrap_policy=auto_wrap_policy,
                wrapper_cls=wrapper_cls,
                **kwargs,
            )
            setattr(module, name, wrapped_child)
            # Keep track of how many parameters have been wrapped
            total_wrapped_params += num_wrapped_params
        # decide if we need to wrap the current module,
        # since the left over parameters exceed the number of params to wrap
        remainder = num_params - total_wrapped_params
        if not only_wrap_children and auto_wrap_policy(
            module=module, recurse=False, unwrapped_params=remainder
        ):
            # Leaf node or final wrapping of the remainder both happen here.
            return _wrap(module, wrapper_cls, **kwargs), num_params
        else:
            return module, total_wrapped_params
    return module, 0


class ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """

    in_autowrap_context: bool = False  # Context flag
    wrapper_cls: Optional[Callable] = None  # The wrapper class
    kwargs: Dict[str, Any] = {}  # Wrapper's args

    def __init__(self, **kwargs: Dict[str, Any]):
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(kwargs: Any) -> None:
        if ConfigAutoWrap.in_autowrap_context:
            raise NotImplementedError(
                "You are already within an autowrap context and we currently do not supported nested autowrap."
            )
        ConfigAutoWrap.in_autowrap_context = True
        # Get and save the wrapper cls for the context.
        assert (
            "wrapper_cls" in kwargs.keys()
        ), "Expected to pass in wrapper_cls arg into ConfigAutoWrap."
        ConfigAutoWrap.wrapper_cls = cast(Callable, kwargs["wrapper_cls"])
        del kwargs["wrapper_cls"]
        # Save the rest.
        ConfigAutoWrap.kwargs = kwargs

    @staticmethod
    def disable_autowrap_context() -> None:
        ConfigAutoWrap.in_autowrap_context = False
        ConfigAutoWrap.wrapper_cls = None
        ConfigAutoWrap.kwargs = {}

    def __enter__(self) -> None:
        self.enable_autowrap_context(self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable_autowrap_context()
