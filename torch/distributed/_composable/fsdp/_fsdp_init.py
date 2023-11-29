import itertools
import warnings

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, init_device_mesh

from torch.distributed.utils import _sync_params_and_buffers

from ._fsdp_common import (
    _is_composable_with_fsdp,
    FSDP_IGNORED,
    FSDP_SHARDED,
    FSDPInternalError,
    FSDPMeshInfo,
    HSDPMeshInfo,
)
from ._fsdp_state import _get_module_fsdp_state

PARAM_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)


def _init_default_fully_shard_mesh(device_type: str) -> DeviceMesh:
    """
    The default fully shard mesh shards over the global mesh.
    """
    default_pg = dist.distributed_c10d._get_default_group()
    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(default_pg.size(),),
        mesh_dim_names=("dp_shard",),
    )
    return mesh


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
        shard_pg = mesh_info.mesh.get_dim_groups(mesh_info.shard_mesh_dim)
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


def _get_managed_modules(root_module: nn.Module) -> List[nn.Module]:
    modules: List[nn.Module] = []
    # Track visisted modules to avoid visiting shared modules multiple times
    visited_modules: Set[nn.Module] = set()

    def dfs(module: nn.Module) -> None:
        """
        Runs a depth-first search to collect managed modules, not recursing
        into any module that is not composable with FSDP.
        """
        if not _is_composable_with_fsdp(module):
            return
        elif module is not root_module and _get_module_fsdp_state(module) is not None:
            # Nested `fully_shard` module
            return
        visited_modules.add(module)
        for submodule in module.children():
            if submodule is not None and submodule not in visited_modules:
                dfs(submodule)
        modules.append(module)

    dfs(root_module)
    return modules


@torch.no_grad()
def _materialize_meta_modules(
    modules: List[nn.Module],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    device: torch.device,
) -> None:
    if param_init_fn is not None:
        if not callable(param_init_fn):
            raise ValueError(
                f"Expects param_init_fn to be a callable but got {type(param_init_fn)}"
            )
    for module in modules:
        module_states = list(_get_module_states_nonrecurse(module))
        # TODO: Add a `is_fp8_mask` to preserve the `_is_fp8_weight` attr or
        # figure out more principled approach.
        ignore_mask = [getattr(t, FSDP_IGNORED, False) for t in module_states]
        if all(ignore_mask) or not any(t.is_meta for t in module_states):
            continue
        if param_init_fn is None:
            module.to_empty(device=device, recurse=False)
            # Assume that each module's `reset_parameters()` only initializes
            # its own parameters and not those of its children, and only call
            # it if a module has at least one managed parameter or buffer
            try:
                module.reset_parameters()  # type: ignore[operator]
            except BaseException as e:
                warnings.warn(
                    "Unable to call `reset_parameters()` for module on meta "
                    f"device with error {str(e)}. Please ensure that your module of"
                    f"type {type(module)} implements a `reset_parameters()` method."
                )
                raise e
        else:
            param_init_fn(module)
        # Get references again since materializing from meta device can
        # construct new tensor objects
        module_states = list(_get_module_states_nonrecurse(module))
        if len(module_states) != len(ignore_mask):
            if param_init_fn is not None:
                fn_str = "param_init_fn"
                suffix_str = f"{param_init_fn} on {type(module)}"
            else:
                fn_str = "reset_parameters"
                suffix_str = f"{type(module)}"
            raise AssertionError(
                f"Calling {fn_str} changed the module's registered parameters "
                f"or buffers (before {len(ignore_mask)} vs. after {len(module_states)}), "
                f"which is unsupported: {suffix_str}"
            )
        for tensor, to_ignore in zip(module_states, ignore_mask):
            if to_ignore:
                setattr(tensor, FSDP_IGNORED, True)


def _get_managed_states(
    modules: List[nn.Module],
    param_to_param_name: Dict[nn.Parameter, str],  # for error messaging
) -> Tuple[List[nn.Parameter], List[torch.Tensor]]:
    params: List[nn.Parameter] = []
    buffers: List[torch.Tensor] = []
    # Track visited parameters/buffers to avoid visiting shared parameters and
    # buffers multiple times
    visited_params: Set[nn.Parameter] = set()
    visited_buffers: Set[torch.Tensor] = set()
    for module in modules:
        for param_name, param in module.named_parameters(recurse=False):
            if getattr(param, FSDP_SHARDED, False):
                raise FSDPInternalError(
                    "Error in FSDP initialization: trying to manage a parameter "
                    f"that is already sharded by FSDP: {param_to_param_name.get(param, None)}"
                )
            if param in visited_params or getattr(param, FSDP_IGNORED, False):
                continue
            params.append(param)
            visited_params.add(param)
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer in visited_buffers or getattr(buffer, FSDP_IGNORED, False):
                continue
            buffers.append(buffer)
            visited_buffers.add(buffer)
    return params, buffers


def _move_params_and_buffers_to_device(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    device: torch.device,
    mesh_info: FSDPMeshInfo,
    sync_module_states: bool,
) -> None:
    # TODO: Since this does not call `nn.Module.to`, it does not consider any
    # overrides. Ideally, we support `_apply`, and users can all `.to()` (or
    # similar) themselves after FSDP initialization if needed.
    for tensor in itertools.chain(params, buffers):
        # Unlike `nn.Module.to()`, we use `.data` for buffers as well for
        # simplicity, avoiding the `setattr` on the module
        if tensor.device != device:
            tensor.data = tensor.to(device)
    if not sync_module_states:
        return
    detached_module_states = [param.detach() for param in params] + [
        buf.detach() for buf in buffers
    ]
    _sync_params_and_buffers(
        mesh_info.shard_process_group,
        detached_module_states,
        PARAM_BROADCAST_BUCKET_SIZE,
        src=0,
    )
    if isinstance(mesh_info, HSDPMeshInfo):
        _sync_params_and_buffers(
            mesh_info.replicate_process_group,
            detached_module_states,
            PARAM_BROADCAST_BUCKET_SIZE,
            src=0,
        )


def _get_module_states_nonrecurse(module: nn.Module) -> Iterable[torch.Tensor]:
    return itertools.chain(
        module.parameters(recurse=False), module.buffers(recurse=False)
    )
