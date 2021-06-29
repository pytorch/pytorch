from dataclasses import dataclass
from typing import (
    Dict,
    List
)

import threading
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.distributed import distributed_c10d
from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
    ShardingSpec,
)
from torch.distributed._sharding_spec._internals import is_valid_device
from torch.distributed.utils import _parse_remote_device

# Tracking for sharded tensor objects.
_sharded_tensor_lock = threading.Lock()
_sharded_tensor_current_id = 0
_sharded_tensor_map: Dict[int, 'ShardedTensor'] = {}


@dataclass
class Shard(object):
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.
    """
    __slots__ = ['tensor', 'metadata']

    tensor: torch.Tensor
    metadata: ShardMetadata


def _register_remote_shards(sharded_tensor_id: int, rrefs: List[rpc.RRef[Shard]], rpc_rank: int):
    with _sharded_tensor_lock:
        if sharded_tensor_id not in _sharded_tensor_map:
            raise RuntimeError(
                f'Could not find sharded_tensor_id: {sharded_tensor_id} in map: {_sharded_tensor_map.keys()}')

        _sharded_tensor_map[sharded_tensor_id]._register_remote_shards(rrefs, rpc_rank)


class ShardedTensor(object):
    """
    ShardedTensor is an abstraction to represent Tensors that are sharded
    across multiple devices and multiple processes.

    ShardedTensor is initialized in an SPMD like fashion where each rank
    initializes the ShardedTensor. The ShardedTensor object on each rank
    then only stores the local shard for the Tensor and provides global
    metadata for all the shards.

    ShardedTensor doesn't provide any Tensor like operations but is a wrapper
    providing the Tensor representing the local shard and the global metadata.
    Using these, users can build their custom distributed sharded computations
    on top of this primitive. The local shards are all initialized using
    :meth:`torch.empty`.

    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.contiguous_format``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. If specified the ShardedTensor is only
            built on ranks that are part of this process group and the provided ``sharding_spec``
            is applied in the context of this process group.
    """

    def __init__(
        self,
        sharding_spec: ShardingSpec,
        *size,
        dtype=None,
        layout=torch.strided,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
        process_group=None,
    ):
        self._rpc_initialized = False
        self._sharded_tensor_id = None
        if rpc._is_current_rpc_agent_set():
            # Validate PG and RPC ranks match.
            pg_rank = dist.get_rank()
            rpc_rank = rpc.get_worker_info().id
            if pg_rank != rpc_rank:
                raise ValueError(
                    f'Default ProcessGroup and RPC ranks must be '
                    f'the same for ShardedTensor, found process group rank: '
                    f'{pg_rank} and RPC rank: {rpc_rank}'
                )

        if layout != torch.strided:
            raise ValueError('Only torch.strided layout is currently supported')

        if memory_format != torch.contiguous_format:
            raise ValueError('Only torch.contiguous_format memory_format is currently supported')

        self._sharding_spec = sharding_spec
        self._dims = list(size)
        self._process_group = (
            process_group
            if process_group is not None
            else distributed_c10d._get_default_group()
        )

        if distributed_c10d._rank_not_in_group(self._process_group):
            raise ValueError(f'Global rank: {dist.get_rank()} not part of process group')

        self._local_shards: List[Shard] = []
        self._remote_shards: Dict[int, List[rpc.RRef[Shard]]] = {}
        self._sharding_metadata: List[ShardMetadata] = []
        if isinstance(self._sharding_spec, ChunkShardingSpec):
            self._init_chunked(
                dtype,
                layout,
                requires_grad,
                pin_memory,
                memory_format,
            )
        elif isinstance(self._sharding_spec, EnumerableShardingSpec):
            self._init_enumerable(
                dtype,
                layout,
                requires_grad,
                pin_memory,
                memory_format,
            )
        else:
            raise ValueError(f'Unsupported sharding_spec: {self._sharding_spec}')

        with _sharded_tensor_lock:
            global _sharded_tensor_current_id, _sharded_tensor_map
            self._sharded_tensor_id = _sharded_tensor_current_id
            _sharded_tensor_map[self._sharded_tensor_id] = self
            _sharded_tensor_current_id += 1

        # Initialize RPC if available.
        if rpc._is_current_rpc_agent_set():
            self._init_rpc()

    def __del__(self):
        # Clean up the global map.
        with _sharded_tensor_lock:
            global _sharded_tensor_current_id, _sharded_tensor_map
            if self._sharded_tensor_id in _sharded_tensor_map:
                _sharded_tensor_map.pop(self._sharded_tensor_id)

    def _init_rpc(self):
        self._rpc_initialized = True
        self._remote_shards = {}

        # Gather all the sharded tensor ids.
        world_size = dist.get_world_size(self._process_group)
        worker_infos = rpc._get_current_rpc_agent().get_worker_infos()
        rank_to_name = {}
        name_to_rank = {}

        for worker_info in worker_infos:
            rank_to_name[worker_info.id] = worker_info.name
            name_to_rank[worker_info.name] = worker_info.id

        rpc_workers = set()
        for rank in range(world_size):
            if self._process_group == distributed_c10d._get_default_group():
                global_rank = rank
            else:
                global_rank = distributed_c10d._get_global_rank(self._process_group, rank)
            rpc_workers.add(rank_to_name[global_rank])

        all_tensor_ids = rpc.api._all_gather(self._sharded_tensor_id, rpc_workers)

        # Share the local shards to the entire world.
        futs = []
        rpc_rank = rpc.get_worker_info().id
        for rank in range(world_size):
            # Skip self.
            if rank == dist.get_rank(self._process_group):
                continue

            if self._process_group == distributed_c10d._get_default_group():
                global_rank = rank
            else:
                global_rank = distributed_c10d._get_global_rank(self._process_group, rank)

            if len(self.local_shards()) != 0:
                rrefs: List[rpc.RRef[Shard]] = [rpc.RRef(shard) for shard in self.local_shards()]
                fut = rpc.rpc_async(
                    global_rank,
                    _register_remote_shards,
                    args=(all_tensor_ids[rank_to_name[global_rank]], rrefs, rpc_rank))
                futs.append(fut)

        torch.futures.wait_all(futs)

        # Barrier for all RPCs to finish on all ranks.
        rpc.api._barrier(rpc_workers)

    def _init_chunked(
        self,
        dtype,
        layout,
        requires_grad,
        pin_memory,
        memory_format,
    ):
        current_rank = dist.get_rank(self._process_group)
        sharding_dim = self._sharding_spec.dim  # type: ignore[attr-defined]

        # Validate the sharding spec.
        if not isinstance(sharding_dim, int):
            raise ValueError(
                f"Sharding dim needs to be an integer, found: {sharding_dim}"
            )
        if sharding_dim >= len(self._dims) or sharding_dim < -len(self._dims):
            raise ValueError(f"Invalid sharding dim: {sharding_dim}")

        dim_size = self._dims[sharding_dim]
        devices = self._sharding_spec.placements  # type: ignore[attr-defined]
        chunks = len(devices)
        # split_size computed similar to 'torch.chunk'
        split_size = (dim_size + chunks - 1) // chunks

        for idx, device in enumerate(devices):
            if not is_valid_device(device):
                raise ValueError(f"{device} is not a valid device")

            rank, local_device = self._parse_and_validate_remote_device(device)

            # Adjust the sharding dim for this rank.
            sharded_dim_size = min(dim_size, split_size * (idx + 1)) - split_size * idx

            if sharded_dim_size > 0:
                # Build sharding_metadata.

                # deepcopy for modification.
                rank_dims = self._dims.copy()

                rank_offsets = [0] * len(self._dims)
                rank_offsets[sharding_dim] = split_size * idx
                rank_dims[sharding_dim] = sharded_dim_size

                shard_metadata = ShardMetadata(rank_offsets, rank_dims, device)
                self._sharding_metadata.append(shard_metadata)

                # Build the local shard for the current rank if it is involved in the sharding spec.
                if current_rank == rank:
                    # Initialize the local shard.
                    local_shard = torch.empty(
                        *rank_dims,
                        dtype=dtype,
                        layout=layout,
                        device=local_device,
                        requires_grad=requires_grad,
                        memory_format=memory_format,
                        pin_memory=pin_memory,
                    )

                    self._local_shards.append(Shard(local_shard, shard_metadata))

    def _init_enumerable(
        self,
        dtype,
        layout,
        requires_grad,
        pin_memory,
        memory_format,
    ):
        # Validate the sharding spec is compatible with the tensor.
        self._sharding_spec.check_tensor(self._dims)  # type: ignore[attr-defined]

        current_rank = dist.get_rank(self._process_group)

        for shard_metadata in self._sharding_spec.shards:  # type: ignore[attr-defined]
            rank, local_device = self._parse_and_validate_remote_device(shard_metadata.placement)
            self._sharding_metadata.append(shard_metadata)

            if current_rank == rank:
                # Initialize the local shard.
                local_shard = torch.empty(
                    *shard_metadata.shard_lengths,
                    dtype=dtype,
                    layout=layout,
                    device=local_device,
                    requires_grad=requires_grad,
                    memory_format=memory_format,
                    pin_memory=pin_memory,
                )

                self._local_shards.append(Shard(local_shard, shard_metadata))

    def _parse_and_validate_remote_device(self, device):

        on, local_device = _parse_remote_device(device)

        # Validate rank.
        if isinstance(on, int) and (on < 0 or on >= dist.get_world_size(self._process_group)):
            raise ValueError(f'Invalid rank: {on}')

        if isinstance(on, str):
            if not rpc._is_current_rpc_agent_set():
                raise RuntimeError(f'RPC framework needs to be initialized for using worker names: {on}')

            workers = rpc._get_current_rpc_agent().get_worker_infos()
            for worker in workers:
                if worker.name == on:
                    return worker.id, local_device

            raise ValueError(f'Invalid worker name: {on}')

        return on, local_device

    def sharding_spec(self) -> ShardingSpec:
        """
        Returns the ShardingSpec for the tensor.
        """
        return self._sharding_spec

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise RuntimeError(f"torch function '{func.__name__}' not supported for ShardedTensor!")

    def sharding_metadata(self) -> List[ShardMetadata]:
        """
        Returns a list of :class:`ShardeMetadata` objects corresponding to the
        metadata for each shard.
        """
        return self._sharding_metadata

    def local_shards(self) -> List[Shard]:
        """
        Returns a list of :class:`Shard' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    def size(self) -> torch.Size:
        """
        Returns the size of the self tensor. The returned value is a subclass of tuple.
        """
        return torch.Size(self._dims)

    def _register_remote_shards(self, remote_shards: List[rpc.RRef[Shard]], rpc_rank: int):
        self._remote_shards[rpc_rank] = remote_shards

    @property
    def remote_shards(self) -> Dict[int, List[rpc.RRef[Shard]]]:
        """
        Returns a Dict[int, RRef] with keys being the RPC rank and values
        being RRefs to shards on that rank. Need to initialize the
        RPC framework for this functionality.
        """
        if not self._rpc_initialized:
            raise RuntimeError(
                "RPC was not initialized before creating the ShardedTensor. Please initialize it using "
                "torch.distributed.rpc.init_rpc before creating the ShardedTensor for remote_shards support"
            )
        return self._remote_shards
