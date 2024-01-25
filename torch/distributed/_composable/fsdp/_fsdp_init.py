from typing import List, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from torch._prims_common import DeviceLikeType

from torch.distributed._tensor import DeviceMesh, init_device_mesh

from ._fsdp_common import _is_composable_with_fsdp
from ._fsdp_state import _get_module_fsdp_state


def _normalize_device(device: DeviceLikeType) -> torch.device:
    if isinstance(device, torch.device):
        if device == torch.device("cuda"):
            return torch.device("cuda", torch.cuda.current_device())
        return device
    elif isinstance(device, int):
        return torch.device("cuda", device)
    elif isinstance(device, str):
        if device == "cuda":
            return torch.device(device, torch.cuda.current_device())
        return torch.device(device)
    else:
        raise TypeError(f"Invalid type for device {device}: {type(device)}")


def _init_default_fully_shard_mesh(device_type: str) -> DeviceMesh:
    """The default fully-shard mesh shards over the global mesh."""
    if not dist.distributed_c10d.is_initialized():
        dist.distributed_c10d.init_process_group()
    default_pg = dist.distributed_c10d._get_default_group()
    mesh = init_device_mesh(device_type=device_type, mesh_shape=(default_pg.size(),))
    return mesh


def _get_managed_modules(root_module: nn.Module) -> List[nn.Module]:
    modules: List[nn.Module] = []
    # Track visisted modules to avoid visiting shared modules multiple times
    visited_modules: Set[nn.Module] = set()

    def dfs(module: nn.Module) -> None:
        """
        Runs a DFS to collect managed modules, not recursing into modules with
        a non-composable API or ``fully_shard`` already applied.
        """
        if not _is_composable_with_fsdp(module):
            return
        elif module is not root_module and _get_module_fsdp_state(module) is not None:
            return  # nested `fully_shard` module
        visited_modules.add(module)
        for submodule in module.children():
            if submodule is not None and submodule not in visited_modules:
                dfs(submodule)
        modules.append(module)

    dfs(root_module)
    return modules


def _get_managed_states(
    modules: List[nn.Module],
) -> Tuple[List[nn.Parameter], List[torch.Tensor]]:
    params: List[nn.Parameter] = []
    buffers: List[torch.Tensor] = []
    # Track visited parameters/buffers to avoid visiting shared parameters and
    # buffers multiple times
    visited_params: Set[nn.Parameter] = set()
    visited_buffers: Set[torch.Tensor] = set()
    for module in modules:
        for param_name, param in module.named_parameters(recurse=False):
            if param in visited_params:
                continue
            params.append(param)
            visited_params.add(param)
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer in visited_buffers:
                continue
            buffers.append(buffer)
            visited_buffers.add(buffer)
    return params, buffers
