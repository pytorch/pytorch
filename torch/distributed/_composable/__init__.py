import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Optional


def contract(func):
    r"""
    docstring TBA
    """

    def wrapper(module: nn.Module, *args, **kwargs) -> Optional[nn.Module]:
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

        return updated

    return wrapper
