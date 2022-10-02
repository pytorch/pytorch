# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Optional,
    Set,
    Tuple,
    Type,
    cast,
)

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


__all__ = [
    "always_wrap_policy",
    "lambda_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "size_based_auto_wrap_policy",
    "enable_wrap",
    "wrap",
    "ParamExecOrderWrapPolicy",
]


def always_wrap_policy(*args, **kwargs) -> bool:
    """
    A simple wrapper policy that always returns ``True``,
    i.e. when passed as the `auto_wrap_policy` into FSDP,
    this will result in all submodules being wrapped as
    distinct FSDP instances.
    """
    return True

def lambda_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    lambda_fn: Callable
) -> bool:
    """
    A convenient auto wrap policy to wrap submodules based on an arbitrary user
    function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
    a `wrapper_cls` unit.

    Return if a module should be wrapped during auto wrapping.

    The first three parameters are required by :func:`_recursive_wrap`.

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

       lambda_fn (Callable[nn.Module] -> bool):
           If this returns ``True``, this module will be wrapped by
           wrapper_cls individually.
    """
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return lambda_fn(module)

def transformer_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    transformer_layer_cls: Set[Type[nn.Module]],
) -> bool:
    """
    A convenient auto wrap policy for transformer models. If the submodule
    is an instance of transformer_layer_cls, the submodule will be wrapped
    as a FSDP unit. Otherwise, all the other remainder submodules are wrapped
    by the outermost FSDP unit. Right now, FSDP requires submodules that share
    weights to be wrapped in the same FSDP unit, this auto wrap policy can
    conviniently wrap the shared embeddings into the same FSDP unit for transformer
    models. In the near future, FSDP will support submodules that share weights
    to be wrapped in the separated FSDP units.

    Return if a module should be wrapped during FSDP auto wrapping.

    The first three parameters are required by :func:`_recursive_wrap`.


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

       transformer_layer_cls (int):
           Submodules with one of the `transformer_layer_cls` names
           will be wrapped as separated FSDP units
    """
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return isinstance(module, tuple(transformer_layer_cls))

def _wrap_batchnorm_individually(
    module: nn.Module,
    recurse: bool,
    *args,
    **kwargs,
) -> bool:
    """
    A policy that wraps ``BatchNorm`` instances in their own FSDP unit.
    """
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap based on whether it is a
        # BN layer or not.
        return isinstance(module, _BatchNorm)

def _or_policy(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    policies,
) -> bool:
    """
    A policy that wraps ``module`` if any policy in the passed in iterable of
    ``policies`` returns ``True``.
    """
    return any(
        policy(module, recurse, unwrapped_params) for policy in policies
    )


def size_based_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    # These are customizable for this policy function.
    min_num_params: int = int(1e8),
    force_leaf_modules: Optional[Set[Type[nn.Module]]] = None,
    exclude_wrap_modules: Optional[Set[Type[nn.Module]]] = None,
) -> bool:
    """A size based auto_wrap_policy function for FSDP API.

       Return if a module should be wrapped during FSDP auto wrapping.

       The first three parameters are used by :func:`_recursive_wrap`. If
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
        size_based_auto_wrap_policy.FORCE_LEAF_MODULES  # type: ignore[attr-defined]
        if force_leaf_modules is None
        else force_leaf_modules
    )
    exclude_wrap_modules = (
        size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore[attr-defined]
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


# Set those defaults to the size_based_auto_wrap_policy function. Make them easy to be imported.
size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {nn.ModuleList, nn.ModuleDict}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {nn.MultiheadAttention}  # type: ignore[attr-defined]


@contextlib.contextmanager
def enable_wrap(
    *, wrapper_cls: Any, **wrapper_kwargs: Any
) -> Generator[None, None, None]:
    """
    Context manager to wrap modules using a wrapper.

    Useful for when you'd like to apply the same configuration arguments to all
    child modules that you wrap. A particularly important use case is wrapping
    large layers so that they get sharded (in-place) during initialization, to
    avoid running out of system memory. Large layers can indicate that they
    should be sharded via the ``wrap`` annotation and this context manager can
    provide the exact configuration for these nested instances.

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
    with _ConfigAutoWrap(**kwargs):
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
    if _ConfigAutoWrap.in_autowrap_context:
        assert _ConfigAutoWrap.wrapper_cls is not None

        wrap_overrides = {**_ConfigAutoWrap.kwargs, **wrap_overrides}
        return _wrap(
            module,
            _ConfigAutoWrap.wrapper_cls,
            **wrap_overrides,
        )
    return module


@dataclass
class ParamExecOrderWrapPolicy:
    """
    This is the class used for the wrapping policy that wraps parameters and performs
    the communication scheduling based on the parameter execution order in the forward pass
    (also called non-recursive wrapping policy).

    The policy contains multiple wraps. Each wrap contains original parameters that will be executed together,
    and the wrap transfers these parameters into one ``FlattenParameter``. In both forward and the backward passes,
    the sharded parameters in each wrap will be gathered just before these parameters are used in the passes.
    These parameters will then be reshaded once they have been used.

    TODO (linjianma): For now, the parameters contained in each wrap of ``ParamExecOrderWrapPolicy``
    are the parameters in each wrap of the ``init_policy`` (a recursive wrapping policy).
    Later we will wrap parameters based on bucket size.

    Args:
        init_policy (Callable):
            The initial recursive wrapping policy used to guide the wrapping of
            this policy. If tracing_config is none, in the first forward and
            backward iteration, ``init_policy`` is used to record parameter
            execution order. Otherwise, init_policy is only used in FSDP
            constructor for module level wrapping.

            The default ``always_wrap_policy`` might not be the best choice for every model. For example, for
            transformer based models, setting ``transformer_auto_wrap_policy`` as the ``init_policy`` will guarantee
            wrapping each transformer layer into one FSDP unit, and can be easily combined with checkpointing
            within each transformer layer.

        tracing_config (Optional[TracingConfig]):
            The configuration used to perform symbolic tracing at FSDP
            constructor to get the module and parameter execution order. The
            type of ``tracing_config`` needs to be either ``None`` or
            ``TracingConfig``. If set as ``None``, then symbolic tracing is not
            enabled, and one forward as well as backward iteration are needed to
            get the parameter execution order.

    ..warning :: Note that not all modules can be successfully traced when
    ``tracing_config`` is not None and symbolic tracing is enabled. The two
    cases below may be unable to trace: 1. when there is a data-dependent
    branch, 2. when the forward pass contains operators that don't support
    ``torch.fx.Proxy`` as the input type (e.g. ``arange``, ``zeros``, ``ones``,
    ``full``, ``full_like``, ``eye``, ``empty``, ``tensor``). For those cases,
    users can set ``tracing_config = None`` to disable symbolic tracing.
    """
    init_policy: Callable = always_wrap_policy
    tracing_config: Any = None


def _wrap(module: nn.Module, wrapper_cls: Callable, **kwargs) -> nn.Module:
    assert wrapper_cls is not None
    if hasattr(module, '_wrap_overrides'):
        # If module has a _wrap_overrides attribute, we force overriding the
        # FSDP config with these attributes for this module. Currently this
        # is only used to disable mixed precision for BatchNorm when
        # auto_wrapping.
        overrides = {**kwargs, **module._wrap_overrides}  # type: ignore[arg-type]
        return wrapper_cls(module, **overrides)

    return wrapper_cls(module, **kwargs)


def _recursive_wrap(
    module: nn.Module,
    auto_wrap_policy: Callable,
    wrapper_cls: Callable,
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
    only_wrap_children: bool = False,
    **kwargs: Any
) -> Tuple[nn.Module, int]:
    """
    Automatically wrap child modules of *module* that meet the given
    criteria with :func:`auto_wrap`. Does not rely on _ConfigAutoWrap.
    Args:
        module (nn.Module):
            module to recursively wrap
        auto_wrap_policy (Callable):
            A callable specifying a policy to recursively wrap layers with FSDP.
        ignored_modules (Set[torch.nn.Module]): Modules to ignore when
            wrapping.
        ignored_params (Set[torch.nn.Parameter]): Parameters to ignore when
            wrapping; these should be the parameters contained in the modules
            in ``ignored_modules``.
    Returns:
        (nn.Module, int):
            Wrapped module and the number parameters wrapped recursively.
    """
    assert auto_wrap_policy is not None, "Must specify auto_wrap_policy."
    assert wrapper_cls is not None, "Must specify wrapper_cls"
    # Make sure no child is already wrapped.
    for _, child in module.named_modules():
        if child in ignored_modules:
            continue
        try:
            assert not isinstance(child, cast(type, wrapper_cls))
        except TypeError:
            # wrapper_cls is a function as opposed to a class type, just bypass above check.
            pass

    # We count all params, assuming none of them are already wrapped.
    num_params = sum(
        p.numel() for p in module.parameters() if p not in ignored_params
    )

    assert auto_wrap_policy is not None
    if auto_wrap_policy(module=module, recurse=True, unwrapped_params=num_params):
        total_wrapped_params = 0
        # Iterate through the children, recursively wrap if necessary
        for name, child in module.named_children():
            if child in ignored_modules:
                continue
            wrapped_child, num_wrapped_params = _recursive_wrap(
                module=child,
                auto_wrap_policy=auto_wrap_policy,
                wrapper_cls=wrapper_cls,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
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


class _ConfigAutoWrap:
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
        if _ConfigAutoWrap.in_autowrap_context:
            raise NotImplementedError(
                "You are already within an autowrap context and we currently do not supported nested autowrap."
            )
        _ConfigAutoWrap.in_autowrap_context = True
        # Get and save the wrapper cls for the context.
        assert (
            "wrapper_cls" in kwargs.keys()
        ), "Expected to pass in wrapper_cls arg into _ConfigAutoWrap."
        _ConfigAutoWrap.wrapper_cls = cast(Callable, kwargs["wrapper_cls"])
        del kwargs["wrapper_cls"]
        # Save the rest.
        _ConfigAutoWrap.kwargs = kwargs

    @staticmethod
    def disable_autowrap_context() -> None:
        _ConfigAutoWrap.in_autowrap_context = False
        _ConfigAutoWrap.wrapper_cls = None
        _ConfigAutoWrap.kwargs = {}

    def __enter__(self) -> None:
        self.enable_autowrap_context(self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable_autowrap_context()
