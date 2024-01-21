from typing import Any

import typing_extensions

import torch.nn as nn
from torch._prims_common import DeviceLikeType

from torch.distributed._composable import contract
from torch.distributed._composable_state import _insert_module_state

from ._fsdp_init import _normalize_device
from ._fsdp_state import FSDPState


@contract(state_cls=FSDPState)
def fully_shard(
    module: nn.Module,
    *,
    device: DeviceLikeType = "cuda",
):
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    device = _normalize_device(device)
    state = fully_shard.state(module)
    _insert_module_state(module, state)
    state._module = module
    state._device = device
    # Place FSDP leftmost for highest priority in the method resolution order
    cls = module.__class__
    dct = {"__deepcopy__": unimplemented_deepcopy}
    new_cls = type(f"FSDP{cls.__name__}", (FSDP, cls), dct)
    module.__class__ = new_cls
    return module


def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> typing_extensions.Never:
    raise AssertionError(
        "FSDP does not support deepcopy. Please use state dict for serialization."
    )


class FSDP:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the FSDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `FSDP<...>` class
        # and index 1 is the `FSDP` class itself
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self
