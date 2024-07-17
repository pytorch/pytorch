# mypy: allow-untyped-defs
import uuid
from collections import OrderedDict
from functools import wraps
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.distributed.utils import _get_root_modules


def generate_state_key(string="__composable_api_state_key"):
    return f"{string}_{str(uuid.uuid4())}"


STATE_KEY = generate_state_key()
REGISTRY_KEY = generate_state_key()


# TODO: we can add additional info to RegistryItem to share across APIs. E.g.,
# we can add args and kwargs here, and then we can detect whether fully_shard
# is combined with reentrant activation checkpointing and error out with a clear
# message.
class RegistryItem:
    pass


def contract(state_cls: Type[_State] = _State):
    r"""
    Decorate a function as a composable distributed API, where the first
    argument of the function must be an :class:`nn.Module` instance. The
    decorator verifies that the wrapped function does not modify parameter,
    buffer or sub-module fully-qualified names (FQN).

    When a function ``func`` is decorated by ``@contract()``, a
    ``.state(module: nn.Module)`` method will be installed to the decorated
    function. Then you can retrieve and modify the state on a module by calling
    ``func.state(module)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> @contract()
        >>> def my_feature(module: nn.Module) -> nn.Module:
        >>>     my_feature.state(module).some_state = "any value"
        >>>     return module
        >>>
        >>> model = MyModel()
        >>> my_feature(model.l1)
        >>> assert my_feature.state(model.l1).some_state == "any value"
        >>> my_feature(model.l2)
        >>> model(torch.randn(2, 10)).sum().backward()
    """

    # wraps will make functions decorated with contract() pickleable - needed for integration with torch.package
    @wraps(state_cls)
    def inner(func):
        @wraps(func)
        def wrapper(
            module: Union[nn.Module, Sequence[nn.Module]], *args, **kwargs
        ) -> Optional[nn.Module]:
            inp_module = module
            if isinstance(module, nn.Module):
                modules = [module]
            else:
                # Special case: only FSDP permits a sequence of modules, in
                # which case we only want to insert the state object on the
                # root modules (i.e. those without a parent) with respect to
                # the passed-in modules.
                modules = _get_root_modules(list(module))
            state = state_cls()  # shared across all modules
            registry_item = RegistryItem()  # shared across all modules
            module_to_orig_named_params: Dict[nn.Module, Dict[str, nn.Parameter]] = {}
            module_to_orig_named_buffers: Dict[nn.Module, Dict[str, torch.Tensor]] = {}
            module_to_orig_named_modules: Dict[nn.Module, Dict[str, nn.Module]] = {}
            for module in modules:
                default_all_state: Dict[Callable, _State] = OrderedDict()
                default_registry: Dict[str, RegistryItem] = OrderedDict()
                all_state: Dict[Callable, _State] = module.__dict__.setdefault(  # type: ignore[call-overload]
                    STATE_KEY, default_all_state
                )
                assert isinstance(
                    all_state, dict
                ), "Distributed composable API states corrupted"
                registry: Dict[str, RegistryItem] = module.__dict__.setdefault(  # type: ignore[call-overload]
                    REGISTRY_KEY, default_registry
                )
                assert isinstance(
                    registry, dict
                ), "Distributed composable API registry corrupted"
                # Make sure that func has not been applied to the module yet
                assert func not in all_state and func.__name__ not in registry, (
                    "Each distinct composable distributed API can only be applied to a "
                    f"module once. {func.__name__} has already been applied to the "
                    f"following module:\n{module}"
                )
                all_state.setdefault(func, state)
                registry.setdefault(func.__name__, registry_item)

                module_to_orig_named_params[module] = OrderedDict(
                    module.named_parameters()
                )
                module_to_orig_named_buffers[module] = OrderedDict(
                    module.named_buffers(remove_duplicate=False)
                )
                module_to_orig_named_modules[module] = OrderedDict(
                    module.named_modules(remove_duplicate=False)
                )

            # `func` should return the same type as the input module/modules
            updated = func(inp_module, *args, **kwargs)
            if updated is None:
                updated = inp_module
            if isinstance(updated, nn.Module):
                updated_modules = [updated]
            else:
                updated_modules = _get_root_modules(list(inp_module))

            module_to_new_named_params: Dict[nn.Module, Dict[str, nn.Parameter]] = {}
            module_to_new_named_buffers: Dict[nn.Module, Dict[str, torch.Tensor]] = {}
            module_to_new_named_modules: Dict[nn.Module, Dict[str, nn.Module]] = {}
            for module in updated_modules:
                module_to_new_named_params[module] = OrderedDict(
                    module.named_parameters()
                )
                module_to_new_named_buffers[module] = OrderedDict(
                    module.named_buffers(remove_duplicate=False)
                )
                module_to_new_named_modules[module] = OrderedDict(
                    module.named_modules(remove_duplicate=False)
                )

            def check_fqn(orig_fqns: List[str], new_fqns: List[str], check_key: str):
                if orig_fqns == new_fqns:
                    return

                orig_fqn_set, new_fqn_set = set(orig_fqns), set(new_fqns)
                orig_only = orig_fqn_set - new_fqn_set
                new_only = new_fqn_set - orig_fqn_set
                if len(orig_only) or len(new_only):
                    raise RuntimeError(
                        f"{check_key}"
                        "Composable distributed API implementations cannot modify "
                        "FQNs.\n"
                        f"Only in original FQNs: {orig_only},\n"
                        f"Only in new FQNs: {new_only}"
                    )
                else:
                    raise RuntimeError(
                        f"{check_key}"
                        "Composable distributed API implementations cannot modify "
                        "the order of FQNs.\n"
                        f"Original FQNs: {orig_only}\n"
                        f"New FQNs: {new_only}"
                    )

            if set(module_to_new_named_modules.keys()) != set(
                module_to_orig_named_modules.keys()
            ):
                raise RuntimeError(
                    f"{func.__name__} should not change the module structure.\n"
                    f"Before: {[str(type(m)) for m in module_to_orig_named_modules]}\n"
                    f"After: {[str(type(m)) for m in module_to_new_named_modules]}"
                )
            for module in module_to_new_named_modules:
                check_fqn(
                    list(module_to_orig_named_params[module].keys()),
                    list(module_to_new_named_params[module].keys()),
                    "Check parameters, ",
                )
                check_fqn(
                    list(module_to_orig_named_buffers[module].keys()),
                    list(module_to_new_named_buffers[module].keys()),
                    "Check buffers, ",
                )
                check_fqn(
                    list(module_to_orig_named_modules[module].keys()),
                    list(module_to_new_named_modules[module].keys()),
                    "Check modules, ",
                )

            # TODO: a stricter verification should also reject changing module
            # types and monkey-patching forward() method implementations.

            # TODO: verify that installed distributed paradigms are compatible with
            # each other.

            return updated

        def get_state(module: nn.Module) -> Optional[_State]:
            return module.__dict__.setdefault(  # type: ignore[call-overload]
                STATE_KEY,
                {},  # TODO(@yhcharles): this is a temporary fix, need a better way
            ).get(
                func
            )  # type: ignore[call-overload]

        wrapper.state = get_state  # type: ignore[attr-defined]

        return wrapper

    return inner


def _get_registry(module: nn.Module) -> Optional[Dict[str, RegistryItem]]:
    r"""
    Get an ``OrderedDict`` of composable APIs that have been applied to the
    ``module``, indexed by the API name. If no API has been applied, then this
    returns ``None``.
    """
    return getattr(module, REGISTRY_KEY, None)
