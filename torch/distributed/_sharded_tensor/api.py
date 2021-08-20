import collections
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
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
from torch.distributed._sharding_spec._internals import (
    check_tensor,
    validate_non_overlapping_shards_metadata
)


# Tracking for sharded tensor objects.
_sharded_tensor_lock = threading.Lock()
_sharded_tensor_current_id = 0
_sharded_tensor_map: Dict[int, 'ShardedTensor'] = {}

# Tracks the current process group in the load context manager.
_CURRENT_PROCESS_GROUP = None

@contextmanager
def load_with_process_group(process_group):
    """
    Context manager to set the process group with which to load a ShardedTensor.
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is not None:
        raise RuntimeError(
            'ProcessGroup already set by previous "load_with_process_group" '
            'context manager')
    _CURRENT_PROCESS_GROUP = process_group
    try:
        yield process_group
    finally:
        _CURRENT_PROCESS_GROUP = None

@dataclass
class Shard(object):
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.
    """
    __slots__ = ['tensor', 'metadata']

    tensor: torch.Tensor
    metadata: ShardMetadata

@dataclass
class ShardedTensorMetadata(object):
    """
    Represents metadata for :class:`ShardedTensor`
    """

    # Metadata about each shard of the Tensor
    shards_metadata: List[ShardMetadata] = field(default_factory=list)

    # Size of each dim of the overall Tensor.
    size: torch.Size = field(default=torch.Size([]))

    # Regular tensor fields
    dtype: torch.dtype = field(default=torch.get_default_dtype())
    layout: torch.layout = field(default=torch.strided)
    requires_grad: bool = False
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    pin_memory: bool = False

    def __getstate__(self):
        # Since torch.memory_format cannot be pickled!
        if self.memory_format == torch.contiguous_format:
            mem_format_encoding = 0
        elif self.memory_format == torch.channels_last:
            mem_format_encoding = 1
        elif self.memory_format == torch.preserve_format:
            mem_format_encoding = 1
        else:
            raise RuntimeError(f'Invalid torch.memory_format: {self.memory_format}')

        return (
            self.shards_metadata,
            self.size,
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (self.shards_metadata, self.size, self.dtype, self.layout,
            self.requires_grad, mem_format_encoding, self.pin_memory) = state

        if mem_format_encoding == 0:
            self.memory_format = torch.contiguous_format
        elif mem_format_encoding == 1:
            self.memory_format = torch.channels_last
        elif mem_format_encoding == 2:
            self.memory_format = torch.preserve_format
        else:
            raise RuntimeError(f'Invalid torch.memory_format encoding: {mem_format_encoding}')


def _register_remote_shards(sharded_tensor_id: int, rrefs: List[rpc.RRef[Shard]], rpc_rank: int):
    with _sharded_tensor_lock:
        if sharded_tensor_id not in _sharded_tensor_map:
            raise RuntimeError(
                f'Could not find sharded_tensor_id: {sharded_tensor_id} in map: {_sharded_tensor_map.keys()}')

        _sharded_tensor_map[sharded_tensor_id]._register_remote_shards(rrefs, rpc_rank)


class CreateOp(Enum):
    EMPTY = 0
    ONES = 1


@dataclass
class TensorInitParams(object):
    """ Container for list of common params to create new local tensor. """

    __slots__ = ['create_op', 'dtype', 'layout', 'requires_grad', 'pin_memory',
                 'memory_format']

    create_op: CreateOp
    dtype: torch.dtype
    layout: torch.layout
    requires_grad: bool
    pin_memory: bool
    memory_format: torch.memory_format


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
    on top of this primitive. The local shards are all initialized using the
    create_op specified by tensor_init_params.create_op, e.g., torch.ones, or
    torch.empty

    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        tensor_init_params (:class: `TensorInitParams`): common params to create tensor.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.
    """

    def __init__(
        self,
        sharding_spec: ShardingSpec,
        *size,
        tensor_init_params: TensorInitParams,
        process_group=None,
        init_rrefs=False,
    ):
        # prepare initialization, initialize fields like
        # _process_group, _local_shards, etc.
        self._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        if tensor_init_params.dtype is None:
            tensor_init_params.dtype = torch.get_default_dtype()

        if tensor_init_params.layout != torch.strided:
            raise ValueError('Only torch.strided layout is currently supported')

        if tensor_init_params.memory_format != torch.contiguous_format:
            raise ValueError('Only torch.contiguous_format memory_format is currently supported')

        if len(size) == 1 and isinstance(size[0], collections.Sequence):
            dims = list(*size)
        else:
            dims = list(size)

        for dim in dims:
            if not isinstance(dim, int):
                raise TypeError(f'size has to be a sequence of ints, found: {type(dim)}')

        self._sharding_spec = sharding_spec

        if isinstance(self._sharding_spec, ChunkShardingSpec):
            self._init_chunked(dims, tensor_init_params)
        elif isinstance(self._sharding_spec, EnumerableShardingSpec):
            self._init_enumerable(dims, tensor_init_params)
        else:
            raise ValueError(f'Unsupported sharding_spec: {self._sharding_spec}')

        # do post initialization (i.e. register sharded_tensor_id, initialize_rpc)
        self._post_init()

    def _prepare_init(self, process_group=None, init_rrefs=False):
        self._init_rrefs = init_rrefs
        self._sharded_tensor_id = None

        self._process_group = (
            process_group
            if process_group is not None
            else distributed_c10d._get_default_group()
        )

        self._local_shards: List[Shard] = []
        self._remote_shards: Dict[int, List[rpc.RRef[Shard]]] = {}
        self._sharding_metadata: List[ShardMetadata] = []

    def _post_init(self):
        with _sharded_tensor_lock:
            global _sharded_tensor_current_id, _sharded_tensor_map
            self._sharded_tensor_id = _sharded_tensor_current_id
            _sharded_tensor_map[self._sharded_tensor_id] = self
            _sharded_tensor_current_id += 1

        # Initialize RPC if available.
        if self._init_rrefs:
            if not rpc._is_current_rpc_agent_set():
                raise RuntimeError(
                    'RPC Framework needs to be initialized using'
                    ' torch.distributed.rpc.init_rpc if init_rrefs is set to True')
            self._init_rpc()

    def __del__(self):
        # Clean up the global map.
        with _sharded_tensor_lock:
            global _sharded_tensor_current_id, _sharded_tensor_map
            if self._sharded_tensor_id in _sharded_tensor_map:
                _sharded_tensor_map.pop(self._sharded_tensor_id)  # type: ignore[call-overload]

    def _init_rpc(self):
        # Validate PG and RPC ranks match.
        pg_rank = dist.get_rank()
        rpc_rank = rpc.get_worker_info().id
        if pg_rank != rpc_rank:
            raise ValueError(
                f'Default ProcessGroup and RPC ranks must be '
                f'the same for ShardedTensor, found process group rank: '
                f'{pg_rank} and RPC rank: {rpc_rank}'
            )

        self._remote_shards = {}

        # Gather all the sharded tensor ids.
        world_size = dist.get_world_size(self._process_group)
        worker_infos = rpc._get_current_rpc_agent().get_worker_infos()
        rank_to_name = {}
        name_to_rank = {}

        for worker_info in worker_infos:
            rank_to_name[worker_info.id] = worker_info.name
            name_to_rank[worker_info.name] = worker_info.id

        all_tensor_ids = rpc.api._all_gather(self._sharded_tensor_id)

        # Share the local shards to the entire world.
        futs = []
        rpc_rank = rpc.get_worker_info().id
        for rank in range(dist.get_world_size()):
            # Skip self.
            if rank == dist.get_rank():
                continue

            if len(self.local_shards()) != 0:
                rrefs: List[rpc.RRef[Shard]] = [rpc.RRef(shard) for shard in self.local_shards()]
                fut = rpc.rpc_async(
                    rank,
                    _register_remote_shards,
                    args=(all_tensor_ids[rank_to_name[rank]], rrefs, rpc_rank))
                futs.append(fut)

        torch.futures.wait_all(futs)

        # Barrier for all RPCs to finish on all ranks.
        rpc.api._all_gather(None)

    @classmethod
    def _init_from_local_shards(
        cls,
        local_shards: List[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
    ):
        shards_metadata = sharded_tensor_metadata.shards_metadata

        if len(shards_metadata) == 0:
            raise ValueError("shards_metadata must not be empty!")

        if sharded_tensor_metadata.layout != torch.strided:
            raise ValueError('Only torch.strided layout is currently supported')

        sharded_tensor = cls.__new__(cls)

        # prepare initialization
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        sharded_tensor._metadata = sharded_tensor_metadata

        # no sharding spec for sharded tensors that initialized
        # from this API.
        sharded_tensor._sharding_spec = None

        current_rank = dist.get_rank(sharded_tensor._process_group)

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = sharded_tensor._parse_and_validate_remote_device(shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        if len(local_shards) != len(local_shard_metadatas):
            raise RuntimeError(
                f'Number of local shards ({len(local_shards)}) does not match number of local '
                f'shards metadata in sharded_tensor_metadata ({len(local_shard_metadatas)}) '
                f'on rank ({current_rank}) '
            )

        for shard in local_shards:
            shard_meta = shard.metadata
            local_shard_tensor = shard.tensor
            rank, local_device = sharded_tensor._parse_and_validate_remote_device(shard_meta.placement)

            # validate if shard_meta in the metadatas collected from sharded_tensor_metadata
            assert shard_meta in local_shard_metadatas, \
                "local shard metadata not in sharded_tensor_metadata!"

            if local_shard_tensor.layout != sharded_tensor_metadata.layout:
                raise ValueError(
                    f'Local shard tensor layout does not match with sharded_tensor_metadata! '
                    f'local shard tensor layout: {local_shard_tensor.dtype}, '
                    f'sharded_tensor_metadata layout: {sharded_tensor_metadata.layout}'
                )

            if not local_shard_tensor.is_contiguous():
                raise ValueError('Only torch.contiguous_format memory_format is currently supported')

            if shard_meta.shard_lengths != list(local_shard_tensor.size()):
                raise ValueError(
                    f'Local shard tensor is incompatible with local ShardMetadata! '
                    f'local shard tensor size: {local_shard_tensor.size()}, '
                    f'local ShardMetadata shard lengths: {shard_meta.shard_lengths}'
                )

            if local_shard_tensor.is_pinned() != sharded_tensor_metadata.pin_memory:
                raise ValueError(
                    f'Local shard tensor pin_memory does not match with sharded_tensor_metadata! '
                    f'local shard tensor pin_memory: {local_shard_tensor.is_pinned()}, '
                    f'sharded_tensor_metadata pin_memory: {sharded_tensor_metadata.pin_memory}'
                )

            if local_shard_tensor.device != local_device:
                raise ValueError(
                    f'Local shard tensor device does not match with local Shard placement! '
                    f'local shard tensor device: {local_shard_tensor.device}, '
                    f'local shard metadata placement device: {local_device}'
                )

            if local_shard_tensor.dtype != sharded_tensor_metadata.dtype:
                raise ValueError(
                    f'Local shard tensor dtype does not match with sharded_tensor_metadata! '
                    f'local shard tensor dtype: {local_shard_tensor.dtype}, '
                    f'sharded_tensor_metadata dtype: {sharded_tensor_metadata.dtype}'
                )

            if local_shard_tensor.requires_grad != sharded_tensor_metadata.requires_grad:
                raise ValueError(
                    f'Local shard tensor requires_grad does not match with sharded_tensor_metadata! '
                    f'local shard tensor requires_grad: {local_shard_tensor.requires_grad}, '
                    f'sharded_tensor_metadata requires_grad: {sharded_tensor_metadata.requires_grad}'
                )

        # check if shards_metadata have overlap shards
        validate_non_overlapping_shards_metadata(shards_metadata)

        # check if the shards_metadata is compatible with overall size of the sharded tensor.
        check_tensor(shards_metadata, list(sharded_tensor_metadata.size))

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    def _init_chunked(self, dims, tensor_init_params: TensorInitParams, ):
        current_rank = dist.get_rank(self._process_group)
        sharding_dim = self._sharding_spec.dim  # type: ignore[attr-defined]

        # Validate the sharding spec.
        if not isinstance(sharding_dim, int):
            raise ValueError(
                f"Sharding dim needs to be an integer, found: {sharding_dim}"
            )
        if sharding_dim >= len(dims) or sharding_dim < -len(dims):
            raise ValueError(f"Invalid sharding dim: {sharding_dim}")

        dim_size = dims[sharding_dim]
        remote_devices = self._sharding_spec.placements  # type: ignore[attr-defined]
        chunks = len(remote_devices)
        # split_size computed similar to 'torch.chunk'
        split_size = (dim_size + chunks - 1) // chunks

        shards_metadata = []
        for idx, remote_device in enumerate(remote_devices):
            rank, local_device = self._parse_and_validate_remote_device(remote_device)

            # Adjust the sharding dim for this rank.
            sharded_dim_size = min(dim_size, split_size * (idx + 1)) - split_size * idx

            if sharded_dim_size > 0:
                # Build sharding_metadata.

                # deepcopy for modification.
                rank_dims = dims.copy()

                rank_offsets = [0] * len(dims)
                rank_offsets[sharding_dim] = split_size * idx
                rank_dims[sharding_dim] = sharded_dim_size

                shard_metadata = ShardMetadata(rank_offsets, rank_dims, remote_device)
                shards_metadata.append(shard_metadata)

                # Build the local shard for the current rank if it is involved in the sharding spec.
                if current_rank == rank:
                    # Initialize the local shard.
                    local_shard = _create_tensor_from_params(
                        *rank_dims, local_device=local_device, tensor_init_params=tensor_init_params)
                    self._local_shards.append(Shard(local_shard, shard_metadata))

        # Build overall metadata
        self._metadata = ShardedTensorMetadata(
            shards_metadata,
            dims,
            tensor_init_params.dtype,
            tensor_init_params.layout,
            tensor_init_params.requires_grad,
            tensor_init_params.memory_format,
            tensor_init_params.pin_memory,
        )

    def _init_enumerable(self, dims, tensor_init_params: TensorInitParams):
        # Validate the sharding spec is compatible with the tensor.
        check_tensor(self._sharding_spec.shards, dims)  # type: ignore[attr-defined]

        current_rank = dist.get_rank(self._process_group)

        shards_metadata = []
        for shard_metadata in self._sharding_spec.shards:  # type: ignore[attr-defined]
            rank, local_device = self._parse_and_validate_remote_device(shard_metadata.placement)
            shards_metadata.append(shard_metadata)

            if current_rank == rank:
                # Initialize the local shard.
                local_shard = _create_tensor_from_params(
                    *shard_metadata.shard_lengths, local_device=local_device,
                    tensor_init_params=tensor_init_params)
                self._local_shards.append(Shard(local_shard, shard_metadata))

        # Build overall metadata
        self._metadata = ShardedTensorMetadata(
            shards_metadata,
            dims,
            tensor_init_params.dtype,
            tensor_init_params.layout,
            tensor_init_params.requires_grad,
            tensor_init_params.memory_format,
            tensor_init_params.pin_memory,
        )

    def _parse_and_validate_remote_device(self, remote_device: torch.distributed._remote_device):

        worker_name = remote_device.worker_name()
        rank = remote_device.rank()
        device = remote_device.device()

        # Validate rank, skip validation if rank is not part of process group.
        if not distributed_c10d._rank_not_in_group(self._process_group):
            if rank is not None and (rank < 0 or rank >= dist.get_world_size(self._process_group)):
                raise ValueError(f'Invalid rank: {rank}')

        if worker_name is not None:
            if not rpc._is_current_rpc_agent_set():
                raise RuntimeError(f'RPC framework needs to be initialized for using worker names: {worker_name}')

            workers = rpc._get_current_rpc_agent().get_worker_infos()
            for worker in workers:
                if worker.name == worker_name:
                    return worker.id, device

            raise ValueError(f'Invalid worker name: {worker_name}')

        return rank, device

    def sharding_spec(self) -> ShardingSpec:
        """
        Returns the ShardingSpec for the tensor.
        """
        return self._sharding_spec

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise RuntimeError(f"torch function '{func.__name__}' not supported for ShardedTensor!")

    def metadata(self) -> ShardedTensorMetadata:
        """
        Returns a :class:`ShardedTensorMetadata` object corresponding to the
        metadata for the entire tensor.
        """
        return self._metadata

    def local_shards(self) -> List[Shard]:
        """
        Returns a list of :class:`Shard' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    def size(self) -> torch.Size:
        """
        Returns the size of the tensor. The returned value is a subclass of tuple.
        """
        return self._metadata.size

    def _register_remote_shards(self, remote_shards: List[rpc.RRef[Shard]], rpc_rank: int):
        self._remote_shards[rpc_rank] = remote_shards

    def remote_shards(self) -> Dict[int, List[rpc.RRef[Shard]]]:
        """
        Returns a Dict[int, RRef] with keys being the RPC rank and values
        being RRefs to shards on that rank. Need to initialize the
        RPC framework for this functionality.

        Raises an exception if ShardedTensor was created with ``init_rrefs=False``
        """
        if not self._init_rrefs:
            raise RuntimeError(
                'ShardedTensor created with init_rrefs=False, no RRefs to remote shards available'
            )
        return self._remote_shards

    def __repr__(self):
        return str(self._metadata)

    @dataclass
    class ProcessGroupState:
        """
        State for ser-de of process group
        """
        local_rank: int
        global_rank: int
        local_world_size: int
        global_world_size: int

    def __getstate__(self):
        pg_state = ShardedTensor.ProcessGroupState(
            distributed_c10d.get_rank(self._process_group),
            distributed_c10d.get_rank(),
            distributed_c10d.get_world_size(self._process_group),
            distributed_c10d.get_world_size(),
        )

        return self._local_shards, self._metadata, pg_state, self._sharding_spec, self._init_rrefs

    def __setstate__(self, state):
        self._sharded_tensor_id = None
        if not distributed_c10d.is_initialized():
            raise RuntimeError(
                'Need to initialize default process group using '
                '"init_process_group" before loading ShardedTensor')

        self._local_shards, self._metadata, pg_state, self._sharding_spec, self._init_rrefs = state

        # Setup process group
        global _CURRENT_PROCESS_GROUP
        if _CURRENT_PROCESS_GROUP is None:
            self._process_group = distributed_c10d._get_default_group()
        else:
            self._process_group = _CURRENT_PROCESS_GROUP

        # Validate process group.
        local_rank = distributed_c10d.get_rank(self._process_group)
        if pg_state.local_rank != local_rank:
            raise RuntimeError(
                f'Local rank at save time was {pg_state.local_rank}, but at '
                f'load time was {local_rank}')

        global_rank = distributed_c10d.get_rank()
        if pg_state.global_rank != global_rank:
            raise RuntimeError(
                f'Global rank at save time was {pg_state.global_rank}, but at '
                f'load time was {global_rank}')

        local_world_size = distributed_c10d.get_world_size(self._process_group)
        if pg_state.local_world_size != local_world_size:
            raise RuntimeError(
                f'Local world size at save time was {pg_state.local_world_size}, '
                f'but at load time was {local_world_size}')

        global_world_size = distributed_c10d.get_world_size()
        if pg_state.global_world_size != global_world_size:
            raise RuntimeError(
                f'Global world size at save time was {pg_state.global_world_size}, '
                f'but at load time was {global_world_size}')

        self._post_init()


def _create_tensor_from_params(*size, local_device, tensor_init_params: TensorInitParams):
    """ Helper to construct tensor from size, device and common params. """

    if tensor_init_params.create_op == CreateOp.ONES:
        return torch.ones(*size,
                          dtype=tensor_init_params.dtype,
                          layout=tensor_init_params.layout,
                          device=local_device,
                          pin_memory=tensor_init_params.pin_memory,
                          requires_grad=tensor_init_params.requires_grad,)
    elif tensor_init_params.create_op == CreateOp.EMPTY:
        return torch.empty(*size,
                           dtype=tensor_init_params.dtype,
                           layout=tensor_init_params.layout,
                           device=local_device,
                           requires_grad=tensor_init_params.requires_grad,
                           # Note memory_format param is not accepted by torch.ones
                           memory_format=tensor_init_params.memory_format,
                           pin_memory=tensor_init_params.pin_memory,)
    else:
        raise ValueError(f'Unsupported create_op: {tensor_init_params.create_op}')
