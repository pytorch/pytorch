import itertools
from typing import List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from torch._prims_common import DeviceLikeType

from torch.distributed._tensor import DeviceMesh, DTensor, init_device_mesh

from ._fsdp_common import _is_composable_with_fsdp, FSDPMeshInfo
from ._fsdp_state import _get_module_fsdp_state


def _get_post_forward_mesh_info(
    reshard_after_forward: Union[bool, int], mesh_info: FSDPMeshInfo
) -> Optional[FSDPMeshInfo]:
    shard_mesh_size = mesh_info.shard_mesh_size
    if not isinstance(reshard_after_forward, (bool, int)):
        raise ValueError(
            "reshard_after_forward should be a bool or an int representing the "
            f"group size to reshard to, not {reshard_after_forward}"
        )
    # NOTE: `isinstance(False, int)` returns `True`.
    if not isinstance(reshard_after_forward, bool) and isinstance(
        reshard_after_forward, int
    ):
        if (
            reshard_after_forward < 1
            or reshard_after_forward > shard_mesh_size
            or shard_mesh_size % reshard_after_forward != 0
        ):
            raise ValueError(
                "If passing reshard_after_forward as an int, it should be a "
                f"factor of {shard_mesh_size}, not {reshard_after_forward}"
            )
        elif reshard_after_forward == 1:
            reshard_after_forward = False
        elif reshard_after_forward == shard_mesh_size:
            reshard_after_forward = True
    if reshard_after_forward is True:
        post_forward_mesh_info = mesh_info
    elif reshard_after_forward is False:
        post_forward_mesh_info = None
    else:
        post_forward_shard_mesh_size = reshard_after_forward
        num_post_forward_meshes = (
            mesh_info.shard_mesh_size // post_forward_shard_mesh_size
        )
        shard_pg = mesh_info.mesh.get_group(mesh_info.shard_mesh_dim)
        assert isinstance(shard_pg, dist.ProcessGroup)  # mypy
        mesh_shard_ranks = sorted(
            dist.distributed_c10d.get_process_group_ranks(shard_pg)
        )
        for i in range(num_post_forward_meshes):
            # E.g., ranks (i, i+1, ..., i+7) for 1D FSDP or ranks
            # (i, i+8, ..., i+56) for 2D intra-node TP + inter-node FSDP
            post_forward_shard_mesh_ranks = mesh_shard_ranks[
                i
                * post_forward_shard_mesh_size : (i + 1)
                * post_forward_shard_mesh_size
            ]
            post_forward_mesh = DeviceMesh("cuda", post_forward_shard_mesh_ranks)
            if i == mesh_info.shard_mesh_rank // post_forward_shard_mesh_size:
                post_forward_mesh_info = FSDPMeshInfo(
                    post_forward_mesh, shard_mesh_dim=0
                )
    return post_forward_mesh_info


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


def _move_states_to_device(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    device: torch.device,
    mesh_info: FSDPMeshInfo,
) -> None:
    """
    We have FSDP move states to device for simpler and faster initialization
    since FSDP almost always uses CUDA for training. We move parameters/buffers
    rather than modules since modules to support ignoring parameters/buffers in
    the future.
    """
    # TODO: De-duplicate with `_apply` after `swap_tensors` path lands:
    # https://github.com/pytorch/pytorch/issues/115792
    for tensor in itertools.chain(params, buffers):
        if tensor.device == device:
            continue
        if isinstance(tensor, DTensor):
            if (dtensor_mesh_type := tensor._spec.mesh.device_type) != device.type:
                raise ValueError(
                    "Requires DTensor to have mesh of the same type as the FSDP mesh "
                    f"but got {dtensor_mesh_type} for DTensor and {device.type} for FSDP"
                )
            raise AssertionError(
                f"Expects DTensor to be moved to {dtensor_mesh_type} but got {tensor.device}"
            )
        tensor.data = tensor.to(device)
