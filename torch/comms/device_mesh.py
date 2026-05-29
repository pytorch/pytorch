# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
TorchComm DeviceMesh integration module.

This module provides integration between TorchComm and PyTorch's DeviceMesh
abstraction. It allows creating DeviceMesh instances backed by TorchComm
communicators, enabling seamless use of TorchComm with PyTorch's distributed
tensor parallelism APIs.

Status: This module is under active development. The core functionality
(init_device_mesh and _flatten_with_comm) is stable, but the API may evolve
as PyTorch's DeviceMesh API changes.

Key functions:
- init_device_mesh: Initialize a DeviceMesh from TorchComm instances
- _flatten_with_comm: Flatten a DeviceMesh dimension with a custom TorchComm
"""

import math
from typing import Any, cast

import torch
import torch.distributed as dist
from torch._C._comms import _BackendWrapper, new_comm, TorchComm
from torch.distributed.device_mesh import _mesh_resources


__all__ = ["init_device_mesh"]


try:
    from torch.distributed.distributed_c10d import GroupName
except ImportError:
    print("GroupName is not available.")
    # Fallback: GroupName is effectively just str when not available from torch
    # We use cast to satisfy type checkers while keeping runtime behavior simple
    GroupName = str  # type: ignore[misc]


def _create_torchcomm_process_group(
    comm: TorchComm,
    group_name: str,
    backend_str: str = "torchcomm",
    global_ranks_mapping: dict[int, int] | None = None,
) -> dist.ProcessGroup:
    """
    Helper function to create a ProcessGroup backed by TorchComm and register it
    with the distributed runtime.

    Args:
        comm: TorchComm instance to wrap
        group_name: Name for the process group
        backend_str: Backend string identifier
        global_ranks_mapping: Mapping from global rank to group rank

    Returns:
        The created and registered ProcessGroup instance
    """
    # Make the linter happy. GroupName is just an alias for str. The cost of
    # this conversion is negligible.
    group_name = GroupName(group_name)

    wrapper = _BackendWrapper(comm)
    backend_type = dist.ProcessGroup.BackendType.CUSTOM
    backend_config = dist.BackendConfig(dist.Backend(backend_str))

    # Create process group. TorchComm backends handle all communication
    # directly, so no real store is needed. We use a dummy HashStore to
    # satisfy the ProcessGroup interface (e.g., splitGroup calls
    # store_->clone()).
    dummy_store = dist.HashStore()
    pg = dist.ProcessGroup(dummy_store, comm.get_rank(), comm.get_size())

    # Register backend
    # pyre-fixme[6]: BackendWrapper implements dist.Backend but types isn't aware
    pg._register_backend(comm.get_device(), backend_type, wrapper)
    pg._set_group_name(group_name)  # pyrefly: ignore[bad-argument-type]

    # Update global state
    dist.distributed_c10d._world.pg_map[pg] = (backend_str, dummy_store)
    dist.distributed_c10d._world.pg_names[pg] = (
        group_name  # pyrefly: ignore[unsupported-operation]
    )
    dist.distributed_c10d._world.pg_backend_config[pg] = str(backend_config)
    # pyrefly: ignore[bad-argument-type]
    dist.distributed_c10d._register_process_group(group_name, pg)

    # Set up rank mapping
    if global_ranks_mapping is not None:
        dist.distributed_c10d._world.pg_group_ranks[pg] = global_ranks_mapping
    else:
        # Default mapping for global process groups
        dist.distributed_c10d._world.pg_group_ranks[pg] = {
            i: i for i in range(comm.get_size())
        }

    # Set up process group tag
    pg_tag = f"ptd:{group_name}"
    dist.distributed_c10d._world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    dist.distributed_c10d._world.pg_to_tag[pg] = pg_tag

    return pg


def init_device_mesh(
    mesh_dim_comms: tuple[TorchComm, ...],
    mesh_dim_names: tuple[str, ...],
    _global_comm: TorchComm | None = None,
) -> dist.DeviceMesh:
    """
    Initializes a `DeviceMesh` from the list of provided `TorchComm` instances.

    See `DeviceMesh` for more details.
    """

    device = mesh_dim_comms[0].get_device()
    mesh_shape = tuple(comm.get_size() for comm in mesh_dim_comms)
    world_size = math.prod(mesh_shape)

    mesh = torch.arange(world_size, dtype=torch.int, device="cpu").view(mesh_shape)

    local_ranks = [comm.get_rank() for comm in mesh_dim_comms]
    global_rank = cast(int, mesh[tuple(local_ranks)].item())
    backend_str = "torchcomm"
    # Register the backend
    dist.Backend.register_backend(backend_str, new_comm)

    global_pg = None
    if _global_comm is not None:
        global_pg = _create_torchcomm_process_group(
            comm=_global_comm,
            group_name=_global_comm.get_name(),
            global_ranks_mapping=None,  # Will use default mapping
        )
    elif len(mesh_dim_comms) != 1:
        raise RuntimeError(
            "More than one torch comm objects are passed but no global comm(_global_comm) is provided. "
            "Please provide a global comm object via _global_comm."
        )

    group_names = []
    for idx, (comm, name) in enumerate(zip(mesh_dim_comms, mesh_dim_names)):
        group_name = name

        # Calculate global ranks mapping for this mesh dimension
        global_ranks = mesh.transpose(idx, -1).reshape(-1, mesh.size(idx))
        # Find the row containing the global rank
        row_idx = int(torch.where(global_ranks == global_rank)[0].item())
        list_rank = global_ranks[row_idx].tolist()
        global_ranks_mapping = {x: j for j, x in enumerate(list_rank)}

        # Use helper function to create the process group
        pg = _create_torchcomm_process_group(
            comm=comm,
            group_name=group_name,
            backend_str=backend_str,
            global_ranks_mapping=global_ranks_mapping,
        )
        if _global_comm is None and idx == 0:
            global_pg = pg

        group_names.append(group_name)

    # Set as the default world process group
    dist.distributed_c10d.GroupMember.WORLD = global_pg

    device_mesh = dist.DeviceMesh(
        device_type=device.type,
        mesh=mesh,
        mesh_dim_names=mesh_dim_names,
        _init_backend=False,
        _rank=global_rank,
    )
    device_mesh._dim_group_names = group_names  # pyrefly: ignore[bad-assignment]

    return device_mesh


def _flatten_with_comm(
    mesh: dist.DeviceMesh,
    mesh_dim_name: str,
    comm: TorchComm,
    global_ranks: list[int],
    layout: Any,
) -> dist.DeviceMesh:
    backend_str = "torchcomm"
    global_ranks_mapping = {global_ranks[i]: i for i in range(comm.get_size())}
    # We still need to register the process group for the flattened mesh
    _create_torchcomm_process_group(
        comm=comm,
        group_name=mesh_dim_name,
        backend_str=backend_str,
        global_ranks_mapping=global_ranks_mapping,
    )

    # Coalesce the layout for the flattened mesh dimension.
    # New PyTorch API (2.13+): _MeshLayout has collapse() instead of coalesce(),
    # and auto-coalesces _FlatLayouts in the constructor.
    if hasattr(layout, "coalesce"):
        coalesced_layout = layout.coalesce()
    else:
        from torch.distributed._mesh_layout import _MeshLayout

        coalesced_layout = _MeshLayout([layout.collapse()])  # pyre-ignore[19]

    # Compatibility layer for DeviceMesh API changes. The new API uses _rank_map
    # while the older API requires passing mesh tensor directly. This conditional
    # can be removed once all supported PyTorch versions include _rank_map support.
    if hasattr(mesh, "_rank_map"):
        flattened_device_mesh = dist.DeviceMesh(
            device_type=comm.get_device(),
            mesh_dim_names=(mesh_dim_name,),
            _init_backend=False,
            _rank=comm.get_rank(),
            _layout=coalesced_layout,
            _rank_map=mesh._rank_map,
            _root_mesh=mesh,
        )
    else:
        flattened_device_mesh = dist.DeviceMesh(
            device_type=comm.get_device(),
            mesh=torch.tensor(global_ranks, device="cpu"),
            mesh_dim_names=(mesh_dim_name,),
            _init_backend=False,
            _rank=comm.get_rank(),
            _layout=coalesced_layout,
        )
    # pyrefly: ignore[bad-assignment]
    flattened_device_mesh._dim_group_names = [mesh_dim_name]

    try:
        flattened_device_mesh._root_mesh = mesh._get_root_mesh()
        flattened_device_mesh._root_mesh._flatten_mapping[mesh_dim_name] = (
            flattened_device_mesh
        )
    except Exception:
        if hasattr(_mesh_resources, "flatten_name_to_root_dims"):
            raise NotImplementedError(
                "Flattening with torchcomm is not supported for device mesh without mesh layout."
            ) from None
        root_mesh = _mesh_resources.get_root_mesh(mesh)
        _mesh_resources.child_to_root_mapping[  # pyre-ignore[16]
            flattened_device_mesh
        ] = root_mesh
        _mesh_resources.root_to_flatten_mapping.setdefault(  # pyre-ignore[16]
            root_mesh, {}
        )[mesh_dim_name] = flattened_device_mesh

    return flattened_device_mesh
