# mypy: allow-untyped-defs
import copy
from typing import Any, cast, List, Optional, Tuple

import torch
import torch.distributed as dist

import torch.distributed._shard.sharding_spec as shard_spec
import torch.distributed.distributed_c10d as c10d
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)

from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard as DShard
from torch.distributed.device_mesh import _mesh_resources

from torch.distributed.fsdp._common_utils import _set_fsdp_flattened
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor.parallel._data_parallel_utils import (
    _flatten_tensor,
    _unflatten_tensor,
)

__all__ = ["DTensorExtensions"]


def _get_box(tensor: DTensor) -> Tuple[torch.Size, torch.Size]:
    device_mesh = tensor.device_mesh
    assert device_mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

    placement = tensor.placements[0]
    offsets = [0] * len(tensor.size())
    num_chunks = device_mesh.size(mesh_dim=0)

    if tensor.placements[0].is_shard():
        shard_dim = cast(DShard, placement).dim
        chunk_size = tensor.size(shard_dim) // num_chunks
        offsets[shard_dim] = chunk_size

    return (torch.Size(offsets), tensor._local_tensor.size())


def _get_box_for(tensor: DTensor, idx: int) -> Tuple[torch.Size, torch.Size]:
    offsets, size = _get_box(tensor)
    return (torch.Size([val * idx for val in offsets]), size)


def _get_local_box(tensor: DTensor) -> Tuple[torch.Size, torch.Size]:
    device_mesh = tensor.device_mesh
    coord = device_mesh.get_coordinate()
    assert coord is not None
    return _get_box_for(tensor, coord[0])


def _create_shard_md_from_dt(dt: DTensor, current_rank: int) -> ShardMetadata:
    mesh = dt.device_mesh
    assert mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

    offsets, sizes = _get_local_box(dt)
    return ShardMetadata(
        shard_offsets=list(offsets),
        shard_sizes=list(sizes),
        placement=f"rank:{current_rank}/{dt._local_tensor.device}",
    )


def _create_sharded_tensor_md_from_dt(
    dt: DTensor, dt_pg: c10d.ProcessGroup
) -> ShardedTensorMetadata:
    # This is where it gets tricky, we have to produce a ShardedTensor that has full coverage
    # and yet has only one valid shard for the current rank.

    shards_md = []
    my_rank = dist.get_rank(dt_pg)
    scapegoat_rank = 0 if my_rank > 0 else 1

    if dt.placements[0].is_shard():
        shard_count = dt_pg.size()
    else:
        shard_count = 1

    for i in range(shard_count):
        offsets, sizes = _get_box_for(dt, i)
        shards_md.append(
            ShardMetadata(
                shard_offsets=list(offsets),
                shard_sizes=list(sizes),
                placement=(
                    f"rank:{scapegoat_rank if i > 0 else my_rank}/{dt._local_tensor.device}"
                ),
            )
        )

    return ShardedTensorMetadata(
        shards_metadata=shards_md,
        size=dt.size(),
        tensor_properties=TensorProperties(
            dtype=dt.dtype,
            layout=dt.layout,
            requires_grad=dt.requires_grad,
            # ignore memory_format and pin_memory as those are not supported by DT
        ),
    )


def _get_dt_pg(dt: DTensor) -> c10d.ProcessGroup:
    mesh = dt.device_mesh
    assert mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"
    return mesh.get_group()


def _rewrite_spec_if_needed(
    spec: shard_spec.ShardingSpec, tensor: torch.Tensor, rank: int
) -> shard_spec.ShardingSpec:
    """
    Rewrite ``spec`` to match the device of ``tensor``.

    FSDP.sharded_optim_state_dict sneakly ships optimizer state to CPU so if the original ShardingSpec
    produces CUDA metadata, ST construction bombs.
    """
    if not isinstance(spec, ChunkShardingSpec):
        return spec

    # let's see if we need
    rewrite = False
    for p in spec.placements:
        p = cast(_remote_device, p)
        if p.rank() == rank and p.device() != tensor.device:
            rewrite = True
            break
    if rewrite:
        spec = copy.deepcopy(spec)
        for i, placement in enumerate(spec.placements):
            placement = cast(_remote_device, placement)
            if placement.rank() == rank and placement.device() != tensor.device:
                spec.placements[i] = _remote_device(f"rank:{rank}/{tensor.device}")

    return spec


def _chunk_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
) -> torch.Tensor:
    if type(tensor) is ShardedTensor:
        assert len(tensor.local_shards()) == 1

        inner_param = tensor.local_tensor()
        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            num_devices_per_node,
            pg,
        )

        outer_local_shard = tensor.local_shards()[0]
        shards: List[Shard] = [
            Shard(inner_st, copy.deepcopy(outer_local_shard.metadata))
        ]
        st_meta = copy.deepcopy(tensor.metadata())
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=tensor._process_group,
            init_rrefs=False,
        )
        return st_outer
    elif type(tensor) is DTensor:
        device_mesh = tensor.device_mesh
        assert device_mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

        inner_param = tensor._local_tensor

        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            torch.cuda.device_count(),
            pg,
        )

        dt_pg = _get_dt_pg(tensor)
        # We do this differently here, we create a ST with no local shards then patch it
        shards = [
            Shard(inner_st, _create_shard_md_from_dt(tensor, dist.get_rank(dt_pg)))
        ]

        st_meta = _create_sharded_tensor_md_from_dt(tensor, dt_pg)
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=dt_pg,
            init_rrefs=False,
        )

        return st_outer
    else:
        return _create_chunk_sharded_tensor(
            tensor,
            rank,
            world_size,
            num_devices_per_node,
            pg,
        )


def _chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
) -> DTensor:
    """
    Shard a tensor to chunks along the first dimension.

    The local rank will gets its corresponding chunk as the local tensor to create a DTensor.
    """
    parent_mesh = _mesh_resources.get_parent_mesh(device_mesh)
    if parent_mesh is None:
        raise RuntimeError("No parent device_mesh is found for FSDP device_mesh.")
    if parent_mesh.ndim < 2:
        raise RuntimeError(
            f"Found parent device_mesh of ndim={parent_mesh.ndim},",
            "but meshes must be at least 2D.",
        )

    # We need to explicitly call .detach() to return a new tensor detached from the current graph.
    tensor = tensor.clone().detach()

    # When a layer is not involved in TP, then the tensor will not be a DTensor.
    # e.g. When a layer is not sppecified in the parallelize_plan, TP will have no effect on the layer.
    # e.g. When you do PairwiseParallel on a 3 layer model, TP will have no effect on the third layer.
    if isinstance(tensor, torch.Tensor) and not isinstance(tensor, DTensor):

        # For tensors, it is replicated across tp dimension and sharded across FSDP dimension.
        # TP is the inner dimension and FSDP is the outer dimension.
        # Therefore, shard placements for tensor is (Shard(0), Replicate()).
        replicate_placements = [Replicate() for _ in range(parent_mesh.ndim)]
        shard_placements = [Replicate() for _ in range(parent_mesh.ndim)]
        shard_placements[0] = DShard(0)  # type: ignore[call-overload]

        return DTensor.from_local(
            tensor, parent_mesh, replicate_placements, run_check=False
        ).redistribute(
            device_mesh=parent_mesh,
            placements=shard_placements,
        )

    else:
        tp_placements = tensor.placements
        tp_placement = tp_placements[0]

        tensor = tensor.to_local()

        # For DTensors, it is sharded across tp dimension first and then sharded across FSDP dimension.
        # TP is the inner dimension and FSDP is the outer dimension.
        # Therefore, shard placements for tensor is (Shard(0), tp_placement).
        # For higher dimensional meshes, it is replicated across other dimensions. For example, with
        # HSDP the shard placements for tensor is (Replicate, Shard(0), tp_placement).
        replicate_placements = [Replicate() for _ in range(parent_mesh.ndim)]
        replicate_placements[-1] = tp_placement  # type: ignore[call-overload]
        shard_placements = [Replicate() for i in range(parent_mesh.ndim)]  # type: ignore[misc]
        shard_placements[-2] = DShard(0)  # type: ignore[call-overload]
        shard_placements[-1] = tp_placement  # type: ignore[call-overload]

        return DTensor.from_local(
            tensor, parent_mesh, replicate_placements, run_check=False
        ).redistribute(
            device_mesh=parent_mesh,
            placements=shard_placements,
        )


def _pre_load_state_dict(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, List[Shard]]:
    shards = cast(ShardedTensor, tensor).local_shards()
    if len(shards) == 1 and type(shards[0].tensor) is ShardedTensor:
        inner_tensor = shards[0].tensor
        shards = inner_tensor.local_shards()  # pyre-ignore[16]
        tensor = inner_tensor

    return (tensor, shards if len(shards) > 0 else [])


def _all_gather_dtensor(
    tensor: DTensor,
    parent_mesh: Optional[DeviceMesh],
) -> torch.Tensor:
    """All gather a DTensor in its FSDP dimension and return the local tensor."""
    assert parent_mesh == tensor.device_mesh

    placements = list(copy.deepcopy(tensor.placements))
    # FSDP + TP: [Shard(0), tp_placement] -> [Replicate(), tp_placement]
    # HSDP + TP: [Replicate(), Shard(0), tp_placement] -> [Replicate(), Replicate(), tp_placement]
    for i in range(0, len(placements) - 1):
        placements[i] = Replicate()
    tensor = tensor.redistribute(
        device_mesh=tensor.device_mesh,
        placements=placements,
    )

    return tensor.to_local()


class DTensorExtensions(FSDPExtensions):
    """
    DTensorExtension is the TensorFlattener extension needed for 2D FSDP + TP.

    This is the implementation for FSDPExtensions defined in
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fsdp_extensions.py
    """
    def __init__(self, device_handle) -> None:
        super().__init__()
        self.compute_stream = None
        self.device_handle = device_handle
        # we have to use the dynamo disable this way to disable dynamo as the decorater way would
        # trigger build failure with torch deploy...
        self.post_unflatten_transform = torch._dynamo.disable(self.post_unflatten_transform)  # type: ignore[method-assign]

    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        return _flatten_tensor(tensor)

    def post_unflatten_transform(
        self, tensor: torch.Tensor, param_extension: Any
    ) -> torch.Tensor:
        stream = self.compute_stream or self.device_handle.current_stream()
        with self.device_handle.stream(stream):
            # runtime we put the unflattened tensor call on the compute stream since
            # the unflattened tensor might contain computations in fwd/bwd where we
            # need to sync properly.
            # TODO: this is a short term fix and we should make the get_unflat_views
            # directly happen in the compute stream.
            result = _unflatten_tensor(
                tensor,
                param_extension,
                device_handle=self.device_handle,
                compute_stream=self.compute_stream
            )
            _set_fsdp_flattened(result)
            return result

    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return _chunk_tensor(tensor, rank, world_size, num_devices_per_node, pg)

    def chunk_dtensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        return _chunk_dtensor(tensor, rank, device_mesh)

    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Shard]]:
        return _pre_load_state_dict(tensor)

    def all_gather_dtensor(
        self,
        tensor: DTensor,
        parent_mesh: Optional[DeviceMesh],
    ) -> torch.Tensor:
        return _all_gather_dtensor(tensor, parent_mesh)
