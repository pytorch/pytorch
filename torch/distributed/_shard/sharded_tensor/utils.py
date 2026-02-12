# mypy: allow-untyped-defs
import collections.abc
import copy
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch.distributed import distributed_c10d as c10d, rpc
from torch.distributed._shard.sharding_spec._internals import (
    check_tensor,
    validate_non_overlapping_shards_metadata,
)

from .metadata import ShardedTensorMetadata, TensorProperties
from .shard import Shard


if TYPE_CHECKING:
    from torch.distributed._shard.metadata import ShardMetadata


def _parse_and_validate_remote_device(pg, remote_device):
    if remote_device is None:
        raise ValueError("remote device is None")

    worker_name = remote_device.worker_name()
    rank = remote_device.rank()
    device = remote_device.device()

    # Validate rank, skip validation if rank is not part of process group.
    if rank is not None and not c10d._rank_not_in_group(pg):
        pg_global_ranks = c10d.get_process_group_ranks(pg)
        if rank not in pg_global_ranks:
            raise ValueError(
                f"Global rank {rank} does not exist in input process group: {pg_global_ranks}"
            )

    if worker_name is not None:
        if not rpc._is_current_rpc_agent_set():
            raise RuntimeError(
                f"RPC framework needs to be initialized for using worker names: {worker_name}"
            )

        workers = rpc._get_current_rpc_agent().get_worker_infos()
        for worker in workers:
            if worker.name == worker_name:
                return worker.id, device

        raise ValueError(f"Invalid worker name: {worker_name}")

    return rank, device


def _validate_output_tensor_for_gather(
    my_rank: int,
    dst_rank: int,
    size: torch.Size,
    dst_tensor: torch.Tensor | None,
) -> None:
    if dst_rank == my_rank:
        if dst_tensor is None:
            raise ValueError(
                f"Argument ``dst_tensor`` must be specified on destination rank {dst_rank}"
            )
        if tuple(size) != (dst_tensor.size()):
            raise ValueError(
                f"Argument ``dst_tensor`` have size {tuple(dst_tensor.size())},"
                f"but should be {tuple(size)}"
            )
    elif dst_tensor:
        raise ValueError(
            "Argument ``dst_tensor`` must NOT be specified on non-destination ranks."
        )


def _flatten_tensor_size(size) -> torch.Size:
    """
    Checks if tensor size is valid, then flatten/return a torch.Size object.
    """
    if len(size) == 1 and isinstance(size[0], collections.abc.Sequence):
        dims = list(*size)
    else:
        dims = list(size)

    for dim in dims:
        if not isinstance(dim, int):
            raise TypeError(f"size has to be a sequence of ints, found: {dims}")

    return torch.Size(dims)


def _raise_if_mismatch(expected, actual, prop_name, ranks, is_local=True):
    if is_local:
        assert isinstance(ranks, int)
        if expected != actual:
            raise ValueError(
                f"Local shards' tensor {prop_name} property need to be the same on rank:{ranks}! "
                f"Found one local shard tensor {prop_name}={expected}, "
                f"the other local shard tensor {prop_name}={actual}."
            )
    else:
        # compare failure check across ranks, ranks list should have two rank
        assert len(ranks) == 2
        if expected != actual:
            raise ValueError(
                f"ShardedTensor {prop_name} property does not match from different ranks! "
                f"Found {prop_name}={expected} on rank:{ranks[0]}, "
                f"and {prop_name}={actual} on rank:{ranks[1]}."
            )


def build_metadata_from_local_shards(
    local_shards: list[Shard],
    global_size: torch.Size,
    current_rank: int,
    pg: c10d.ProcessGroup,
) -> ShardedTensorMetadata:
    assert len(local_shards) > 0, "must have local shards!"
    local_shard_metadatas: list[ShardMetadata] = []

    first_shard_dtype = local_shards[0].tensor.dtype
    first_shard_layout = local_shards[0].tensor.layout
    first_shard_requires_grad = local_shards[0].tensor.requires_grad
    first_shard_is_pinned = local_shards[0].tensor.is_pinned()

    # 1). Validate local tensors and associated metadatas
    for local_shard in local_shards:
        local_shard_tensor = local_shard.tensor
        local_shard_meta = local_shard.metadata
        local_shard_metadatas.append(local_shard_meta)
        rank, local_device = _parse_and_validate_remote_device(
            pg, local_shard_meta.placement
        )

        if (
            local_shard_tensor.layout != torch.strided
            or local_shard_tensor.layout != first_shard_layout
        ):
            raise ValueError(
                f"Only torch.strided layout is currently supported, but found "
                f"{local_shard_tensor.layout} on rank:{current_rank}!"
            )

        if not local_shard_tensor.is_contiguous():
            raise ValueError(
                "Only torch.contiguous_format memory_format is currently supported!"
            )

        if rank != current_rank:
            raise ValueError(
                f"Local shard metadata's rank does not match with the rank in its process group! "
                f"Found current rank in the process group: {current_rank}, "
                f"local ShardMetadata placement's rank: {rank}"
            )
        if local_shard_tensor.device != local_device:
            raise ValueError(
                f"Local shard tensor device does not match with local Shard's placement! "
                f"Found local shard tensor device: {local_shard_tensor.device}, "
                f"local shard metadata placement device: {local_device}"
            )

        _raise_if_mismatch(
            local_shard_meta.shard_sizes,
            list(local_shard_tensor.size()),
            "size",
            current_rank,
        )
        _raise_if_mismatch(
            local_shard_tensor.is_pinned(),
            first_shard_is_pinned,
            "pin_memory",
            current_rank,
        )
        _raise_if_mismatch(
            local_shard_tensor.dtype, first_shard_dtype, "dtype", current_rank
        )
        _raise_if_mismatch(
            local_shard_tensor.requires_grad,
            first_shard_requires_grad,
            "requires_grad",
            current_rank,
        )

    # 2). Build a "local" ShardedTensorMetadata with all local shards on this rank, then
    #    do all_gather to collect local_sharded_tensor_metadata from all ranks
    local_tensor_properties = TensorProperties(
        dtype=first_shard_dtype,
        layout=first_shard_layout,
        requires_grad=first_shard_requires_grad,
        memory_format=torch.contiguous_format,
        pin_memory=first_shard_is_pinned,
    )

    local_sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=local_shard_metadatas,
        size=global_size,
        tensor_properties=local_tensor_properties,
    )

    return local_sharded_tensor_metadata


def build_global_metadata(
    gathered_metadatas: Sequence[ShardedTensorMetadata | None],
    recalc_metadata: bool = False,
):
    global_sharded_tensor_metadata = None
    global_metadata_rank = 0

    # pyrefly: ignore [bad-assignment]
    for rank, rank_metadata in enumerate(gathered_metadatas):
        if rank_metadata is None:
            continue

        if global_sharded_tensor_metadata is None:
            global_sharded_tensor_metadata = copy.deepcopy(rank_metadata)
            global_metadata_rank = rank
        else:
            _raise_if_mismatch(
                global_sharded_tensor_metadata.size,
                rank_metadata.size,
                "global_size",
                [global_metadata_rank, rank],
                is_local=False,
            )

            # don't need to check layout and memory format as we already checked in local shards validation stage
            _raise_if_mismatch(
                global_sharded_tensor_metadata.tensor_properties.dtype,
                rank_metadata.tensor_properties.dtype,
                "dtype",
                [global_metadata_rank, rank],
                is_local=False,
            )

            _raise_if_mismatch(
                global_sharded_tensor_metadata.tensor_properties.requires_grad,
                rank_metadata.tensor_properties.requires_grad,
                "requires_grad",
                [global_metadata_rank, rank],
                is_local=False,
            )

            _raise_if_mismatch(
                global_sharded_tensor_metadata.tensor_properties.pin_memory,
                rank_metadata.tensor_properties.pin_memory,
                "pin_memory",
                [global_metadata_rank, rank],
                is_local=False,
            )
            # pass all validations, extend shards metadata
            global_sharded_tensor_metadata.shards_metadata.extend(
                rank_metadata.shards_metadata
            )

    if global_sharded_tensor_metadata is not None:
        if recalc_metadata:
            recalc_global_sharded_tensor_metadata(
                global_sharded_tensor_metadata,
                0,  # sharded on 0th dim
            )

        # check if shards_metadata have overlap shards
        validate_non_overlapping_shards_metadata(
            global_sharded_tensor_metadata.shards_metadata
        )

        # check if the shards_metadata is compatible with global size of the sharded tensor.
        check_tensor(
            global_sharded_tensor_metadata.shards_metadata,
            global_sharded_tensor_metadata.size,
        )
    else:
        raise ValueError("ShardedTensor have no local shards on all ranks!")

    return global_sharded_tensor_metadata


def recalc_global_sharded_tensor_metadata(
    global_sharded_tensor_metadata: ShardedTensorMetadata, sharded_dim: int
) -> None:
    # recalculate global ShardedTensorMetadata

    # reorder here in case shard metadata is not sorted on sharded_dim
    placement_idx_pairs = []
    for i, shard_metadata in enumerate(global_sharded_tensor_metadata.shards_metadata):
        if shard_metadata.placement:
            placement_idx_pairs.append((shard_metadata.placement.rank(), i))
        else:
            raise AssertionError(
                "currently only support rw, it should always have valid rank info"
            )
    sorted_idx = sorted(placement_idx_pairs)
    shard_sizes = [
        global_sharded_tensor_metadata.shards_metadata[idx].shard_sizes[sharded_dim]
        for _, idx in sorted_idx
    ]
    cum_sum = [0] + list(itertools.accumulate(shard_sizes))

    for shard_id, shard_metadata in enumerate(
        global_sharded_tensor_metadata.shards_metadata
    ):
        # update shard offset for each shard on the sharded dimension
        shard_metadata.shard_offsets[sharded_dim] = cum_sum[shard_id]
        for other_dim in range(
            len(global_sharded_tensor_metadata.shards_metadata[0].shard_sizes)
        ):
            if other_dim != sharded_dim:
                # shard offset for each shard on the unsharded dimension
                shard_metadata.shard_offsets[other_dim] = 0

    # update global size for ShardedTensorMetadata
    global_size_list = []
    for other_dim in range(
        len(global_sharded_tensor_metadata.shards_metadata[0].shard_sizes)
    ):
        if other_dim != sharded_dim:
            global_size_list.append(
                global_sharded_tensor_metadata.shards_metadata[0].shard_sizes[other_dim]
            )
        else:
            global_size_list.append(cum_sum[-1])
    global_sharded_tensor_metadata.size = torch.Size(global_size_list)
