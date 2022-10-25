import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Any, Callable, Dict, Optional


# use state_slot as key for module.__dict__ to avoid coliding with other
# properties.
# TODO: since all composable distributed features can share the same slot.
class _StateKey:

    # implement operator < to avoid breaking dir()
    def __lt__(self, other: Any) -> bool:
        return True if isinstance(other, str) else id(self) < id(other)


state_key = _StateKey()


def contract(func):
    def get_all_state(module: nn.Module) -> Dict[Callable, Dict]:
        d = module.__dict__.setdefault(state_key, {})
        assert isinstance(
            d, dict
        ), "Distributed composable API states corrupted"
        return d

    r"""
    Decorate a function as a composable distributed API, where the first
    argument of the function must be an :class:`nn.Module` instance. The
    decorator verifies that the wrapped function does not modify parameter,
    buffer, and sub-module fully-qualified names (FQN).
    """

    def wrapper(module: nn.Module, *args, **kwargs) -> Optional[nn.Module]:
        # install states specific to the wrapped ``func``
        all_state: Dict[Callable, dict] = get_all_state(module)
        assert func not in all_state, (
            "Each distinct composable distributed API can only be applied to a "
            f"module once. {func} has already been applied to the following "
            f"module.\n{module}"
        )
        all_state.setdefault(func, {})

        orig_named_params = OrderedDict(module.named_parameters())
        orig_named_buffers = OrderedDict(
            module.named_buffers(remove_duplicate=False)
        )
        orig_named_modules = OrderedDict(
            module.named_modules(remove_duplicate=False)
        )

        updated = func(module, *args, **kwargs)

        if updated is None:
            updated = module

        new_named_params = OrderedDict(updated.named_parameters())
        new_named_buffers = OrderedDict(
            updated.named_buffers(remove_duplicate=False)
        )
        new_named_modules = OrderedDict(
            updated.named_modules(remove_duplicate=False)
        )

        assert isinstance(updated, nn.Module), (
            "Output of composable distributed APIs must be either None or "
            f"nn.Module, but got {type(updated)}"
        )
        assert list(orig_named_params.keys()) == list(
            new_named_params.keys()
        ), (
            "Composable distributed API implementations cannot modify "
            "parameter FQNs. \n"
            f"Original FQNs: {list(orig_named_params.keys())}, \n"
            f"Updated FQNs: {list(OrderedDict(module.named_parameters()).keys())}"
        )
        assert list(orig_named_buffers.keys()) == list(
            new_named_buffers.keys()
        ), (
            "Composable distributed API implementations cannot modify "
            "buffer FQNs. \n"
            f"Original FQNs: {list(orig_named_buffers.keys())}, \n"
            f"Updated FQNs: {list(OrderedDict(module.named_buffers()).keys())}"
        )
        assert new_named_modules.keys() == new_named_modules.keys(), (
            "Composable distributed API implementations cannot modify "
            "submodule FQNs. \n"
            f"Original FQNs: {orig_named_modules.keys()}, \n"
            f"Updated FQNs: {OrderedDict(module.named_modules()).keys()}"
        )

        # TODO: a stricter verification should also reject changing module
        # types and monkey-patching forward() method implementations.

        # TODO: verify that installed distributed paradigms are compatible with
        # each other.

        return updated

    def get_state(module: nn.Module, key: Any) -> Any:
        return get_all_state(module).get(func).get(key, None)

    def set_state(module: nn.Module, key: Any, value: Any) -> None:
        get_all_state(module).setdefault(func, {})[key] = value

    wrapper.get_state = get_state
    wrapper.set_state = set_state

    return wrapper
