from typing import Any, Optional, Union

import torch
import torch.nn as nn

from torch.distributed._composable import contract
from torch.distributed._composable_state import _insert_module_state
from torch.distributed._tensor import DeviceMesh

from ._fsdp_common import FSDPMeshInfo, HSDPMeshInfo
from ._fsdp_init import _init_default_fully_shard_mesh, _normalize_device
from ._fsdp_state import FSDPState


@contract(state_cls=FSDPState)
def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    device: Union[torch.device, int, str] = "cuda",
):
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    device = _normalize_device(device)
    mesh = mesh or _init_default_fully_shard_mesh(device.type)
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    elif mesh.ndim == 2:
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    if device.type != mesh.device_type:
        raise ValueError(
            f"device and mesh must be of the same type but got {device.type} "
            f"for device and {mesh.device_type} for mesh"
        )

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


def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> None:
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
