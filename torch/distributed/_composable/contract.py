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
    argument of the function must be an :class:`nn.Module` instance or sequence
    of :class:`nn.Module` instances.

    The decorator verifies that the decorated function does not modify
    fully-qualified names (FQNs) for parameters, buffers, or modules. The
    decorated function can return different module instances than the input
    modules; the FQN invariant will be enforced following the input order.

    When a function ``func`` is decorated by ``@contract()``, a
    ``.state(module: nn.Module)`` method will be installed to the decorated
    function. Then you can retrieve and modify the state on a module by calling
    ``func.state(module)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self) -> None:
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
                # If the user passes a sequence of modules, then we assume that
                # we only need to insert the state object on the root modules
                # (i.e. those without a parent) among the passed-in modules.
                modules = _get_root_modules(list(module))
            state = state_cls()  # shared across all modules
            registry_item = RegistryItem()  # shared across all modules

            # `func` is allowed to return different module instances than the
            # input modules as long as FQNs are preserved following the input
            # module order
            all_orig_named_params: List[Dict[str, nn.Parameter]] = []
            all_orig_named_buffers: List[Dict[str, torch.Tensor]] = []
            all_orig_named_modules: List[Dict[str, nn.Module]] = []

            for module in modules:
                default_all_state: Dict[Callable, _State] = OrderedDict()
                default_registry: Dict[str, RegistryItem] = OrderedDict()
                all_state: Dict[Callable, _State] = module.__dict__.setdefault(
                    STATE_KEY, default_all_state
                )
                if not isinstance(all_state, dict):
                    raise AssertionError(
                        f"Distributed composable API states corrupted: {all_state}"
                    )
                registry: Dict[str, RegistryItem] = module.__dict__.setdefault(
                    REGISTRY_KEY, default_registry
                )
                if not isinstance(registry, dict):
                    raise AssertionError(
                        f"Distributed composable API registry corrupted: {registry}"
                    )
                if func in all_state or func.__name__ in registry:
                    raise AssertionError(
                        "Each distinct composable distributed API can only be applied to a "
                        f"module once. {func.__name__} has already been applied to the "
                        f"following module:\n{module}"
                    )
                all_state.setdefault(func, state)
                registry.setdefault(func.__name__, registry_item)

                all_orig_named_params.append(OrderedDict(module.named_parameters()))
                all_orig_named_buffers.append(OrderedDict(module.named_buffers()))
                all_orig_named_modules.append(OrderedDict(module.named_modules()))

            updated = func(inp_module, *args, **kwargs)
            if updated is None:
                updated = inp_module
            if isinstance(updated, nn.Module):
                updated_modules = [updated]
            else:
                updated_modules = _get_root_modules(list(inp_module))  # type: ignore[arg-type]

            all_new_named_params: List[Dict[str, nn.Parameter]] = []
            all_new_named_buffers: List[Dict[str, torch.Tensor]] = []
            all_new_named_modules: List[Dict[str, nn.Module]] = []
            for module in updated_modules:
                all_new_named_params.append(OrderedDict(module.named_parameters()))
                all_new_named_buffers.append(OrderedDict(module.named_buffers()))
                all_new_named_modules.append(OrderedDict(module.named_modules()))

            num_orig_modules = len(all_orig_named_modules)
            num_new_modules = len(all_new_named_modules)
            if num_orig_modules != num_new_modules:
                raise AssertionError(
                    f"{func.__name__} should return the same number of modules as input modules"
                    f"Inputs: {num_orig_modules} modules\n"
                    f"Outputs: {num_new_modules} modules"
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
                        "Composable distributed API implementations cannot modify FQNs.\n"
                        f"FQNs only in original: {orig_only}\n"
                        f"FQNs only in new: {new_only}"
                    )
                else:
                    raise RuntimeError(
                        f"{check_key}"
                        "Composable distributed API implementations cannot modify "
                        "the order of FQNs.\n"
                        f"Original FQNs: {orig_only}\n"
                        f"New FQNs: {new_only}"
                    )

            for orig_named_params, new_named_params in zip(
                all_orig_named_params, all_new_named_params
            ):
                check_fqn(
                    list(orig_named_params.keys()),
                    list(new_named_params.keys()),
                    "Checking parameters: ",
                )
            for orig_named_buffers, new_named_buffers in zip(
                all_orig_named_buffers, all_new_named_buffers
            ):
                check_fqn(
                    list(orig_named_buffers.keys()),
                    list(new_named_buffers.keys()),
                    "Checking buffers: ",
                )
            for orig_named_modules, new_named_modules in zip(
                all_orig_named_modules, all_new_named_modules
            ):
                check_fqn(
                    list(orig_named_modules.keys()),
                    list(new_named_modules.keys()),
                    "Checking modules: ",
                )

            # TODO: verify that installed distributed paradigms are compatible with
            # each other.

            return updated

        def get_state(module: nn.Module) -> Optional[_State]:
            return module.__dict__.setdefault(
                STATE_KEY,
                {},  # TODO(@yhcharles): this is a temporary fix, need a better way
            ).get(func)

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
