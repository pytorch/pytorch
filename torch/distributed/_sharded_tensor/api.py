from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict,
    List,
    Optional
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
    check_tensor_size,
    validate_non_overlapping_shards_metadata
)
from torch.types import Number

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
class TensorProperties(object):
    """ Properties used to create :class:`Tensor` """

    # Regular tensor fields
    dtype: torch.dtype = field(default=torch.get_default_dtype())
    layout: torch.layout = field(default=torch.strided)
    requires_grad: bool = False
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    pin_memory: bool = False


class MEM_FORMAT_ENCODING(Enum):
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


@dataclass
class ShardedTensorMetadata(object):
    """
    Represents metadata for :class:`ShardedTensor`
    """

    # Metadata about each shard of the Tensor
    shards_metadata: List[ShardMetadata] = field(default_factory=list)

    # Size of each dim of the overall Tensor.
    size: torch.Size = field(default=torch.Size([]))

    tensor_properties: TensorProperties = field(
        default=TensorProperties(dtype=torch.get_default_dtype(),
                                 layout=torch.strided,
                                 requires_grad=False,
                                 memory_format=torch.contiguous_format,
                                 pin_memory=False))

    def __getstate__(self):
        # Since torch.memory_format cannot be pickled!
        memory_format = self.tensor_properties.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f'Invalid torch.memory_format: {memory_format}')

        # Keep old seriazation to ensure backward compatibility
        return (
            self.shards_metadata,
            self.size,
            self.tensor_properties.dtype,
            self.tensor_properties.layout,
            self.tensor_properties.requires_grad,
            mem_format_encoding,
            self.tensor_properties.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (self.shards_metadata, self.size, dtype, layout, requires_grad, mem_format_encoding, pin_memory) = state

        if mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format
        else:
            raise RuntimeError(f'Invalid torch.memory_format encoding: {mem_format_encoding}')

        self.tensor_properties = TensorProperties(
            dtype=dtype, layout=layout, requires_grad=requires_grad,
            memory_format=memory_format, pin_memory=pin_memory, )


def _register_remote_shards(sharded_tensor_id: int, rrefs: List[rpc.RRef[Shard]], rpc_rank: int):
    with _sharded_tensor_lock:
        if sharded_tensor_id not in _sharded_tensor_map:
            raise RuntimeError(
                f'Could not find sharded_tensor_id: {sharded_tensor_id} in map: {_sharded_tensor_map.keys()}')

        _sharded_tensor_map[sharded_tensor_id]._register_remote_shards(rrefs, rpc_rank)


class CreateOp(Enum):
    EMPTY = 0
    FULL = 1
    ONES = 2
    RAND = 3
    ZEROS = 4


@dataclass
class TensorInitParams(object):
    """ Container for list of common params to create new local tensor. """

    create_op: CreateOp

    # needed when create_op is FULL
    # default set to False (not None) since None is incompatible with Number.
    fill_value: Number = field(default=False)

    tensor_properties: TensorProperties = field(
        default=TensorProperties(dtype=torch.get_default_dtype(),
                                 layout=torch.strided,
                                 requires_grad=False,
                                 memory_format=torch.contiguous_format,
                                 pin_memory=False))


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

        if tensor_init_params.tensor_properties is None:
            raise ValueError('tensor_properties must not be None.')

        if tensor_init_params.tensor_properties.dtype is None:
            tensor_init_params.tensor_properties.dtype = torch.get_default_dtype()

        if tensor_init_params.tensor_properties.layout != torch.strided:
            raise ValueError('Only torch.strided layout is currently supported')

        if tensor_init_params.tensor_properties.memory_format != torch.contiguous_format:
            raise ValueError('Only torch.contiguous_format memory_format is currently supported')

        dims = check_tensor_size(size)

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
        *overall_size,
        process_group=None,
        init_rrefs=False,
    ):
        # prepare initialization
        sharded_tensor = cls.__new__(cls)
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)
        current_rank = dist.get_rank(sharded_tensor._process_group)
        world_size = dist.get_world_size(sharded_tensor._process_group)

        if len(local_shards) == 0:
            raise ValueError("local_shards must not be empty!")

        local_shard_metadatas: List[ShardMetadata] = []
        local_shards_dtype = local_shards[0].tensor.dtype
        local_shards_layout = local_shards[0].tensor.layout
        local_shards_requires_grad = local_shards[0].tensor.requires_grad
        local_shards_is_pinned = local_shards[0].tensor.is_pinned()

        # First, do local validations on local shards
        for local_shard in local_shards:
            local_shard_tensor = local_shard.tensor
            local_shard_meta = local_shard.metadata
            local_shard_metadatas.append(local_shard_meta)
            rank, local_device = sharded_tensor._parse_and_validate_remote_device(local_shard_meta.placement)

            if not local_shard_tensor.is_contiguous():
                raise ValueError('Only torch.contiguous_format memory_format is currently supported!')

            if local_shard_tensor.layout != torch.strided or local_shard_tensor.layout != local_shards_layout:
                raise ValueError(
                    f'Only torch.strided layout is currently supported, but found '
                    f'{local_shard_tensor.layout} on rank:{current_rank}!'
                )

            if local_shard_meta.shard_lengths != list(local_shard_tensor.size()):
                raise ValueError(
                    f'Local shard tensor size does not match with local ShardMetadata on rank:{current_rank}! '
                    f'Found local shard tensor size: {local_shard_tensor.size()}, '
                    f'local ShardMetadata shard lengths: {local_shard_meta.shard_lengths}'
                )

            if local_shard_tensor.device != local_device:
                raise ValueError(
                    f"Local shard tensor device does not match with local Shard's placement! "
                    f"Found local shard tensor device: {local_shard_tensor.device}, "
                    f"local shard metadata placement device: {local_device}"
                )

            if local_shard_tensor.dtype != local_shards_dtype:
                raise ValueError(
                    f"Local shards' tensor dtype property need to be the same! "
                    f"Found one local shard tensor dtype: {local_shards_dtype}, "
                    f"the other local shard tensor dtype: {local_shard_tensor.dtype}"
                )

            if local_shard_tensor.requires_grad != local_shards_requires_grad:
                raise ValueError(
                    f"Local shards' tensor requires_grad property need to be the same! "
                    f"Found one local shard tensor requires_grad: {local_shards_requires_grad}, "
                    f"the other local shard tensor requires_grad: {local_shard_tensor.requires_grad}"
                )

            if local_shard_tensor.is_pinned() != local_shards_is_pinned:
                raise ValueError(
                    f"Local shards' tensor pin_memory property need to be the same! "
                    f"Found one local shard tensor is_pinned(): {local_shards_is_pinned}, "
                    f"the other local shard tensor is_pinned(): {local_shard_tensor.is_pinned()}"
                )

        # build a "local" ShardedTensorMetadata with all local shards on this rank
        global_size = torch.Size(check_tensor_size(overall_size))

        local_shard_tensor_metadata = ShardedTensorMetadata(
            shards_metadata=local_shard_metadatas,
            size=global_size,
            tensor_properties=TensorProperties(
                dtype=local_shards_dtype,
                layout=local_shards_layout,
                requires_grad=local_shards_requires_grad,
                memory_format=torch.contiguous_format,
                pin_memory=local_shards_is_pinned
            )
        )

        # do all_gather to gather all sharded tensor metadatas in each rank
        all_metadatas: List[Optional[ShardedTensorMetadata]] = [None for _ in range(world_size)]

        dist.all_gather_object(
            all_metadatas,
            local_shard_tensor_metadata,
            group=sharded_tensor._process_group
        )

        # do validation with gathered metadatas and build up a global sharded tensor metadata
        global_sharded_tensor_metadata = all_metadatas[0]

        for i in range(1, len(all_metadatas)):
            global_sharded_tensor_metadata.shards_metadata.extend(all_metadatas[i].shards_metadata)
            if global_sharded_tensor_metadata.size != all_metadatas[i].size:
                raise ValueError(
                    f'ShardedTensor overall_size does not match from different ranks! '
                    f'Found overall_size={global_sharded_tensor_metadata.size} on rank 0, '
                    f'and overall_size={all_metadatas[i].size} on rank {i}'
                )

            # don't need to check layout and memory format as we already checked in local shards validation stage
            if global_sharded_tensor_metadata.tensor_properties.dtype != all_metadatas[i].tensor_properties.dtype:
                raise ValueError(
                    f'ShardedTensor dtype property does not match from different ranks! '
                    f'Found dtype={global_sharded_tensor_metadata.tensor_properties.dtype} on rank 0, '
                    f'and dtype={all_metadatas[i].tensor_properties.dtype} on rank {i}'
                )

            if global_sharded_tensor_metadata.tensor_properties.requires_grad != all_metadatas[i].tensor_properties.requires_grad:
                raise ValueError(
                    f'ShardedTensor requires_grad property does not match from different ranks! '
                    f'Found requires_grad={global_sharded_tensor_metadata.tensor_properties.requires_grad} on rank 0, '
                    f'and requires_grad={all_metadatas[i].tensor_properties.requires_grad} on rank {i}'
                )

            if global_sharded_tensor_metadata.tensor_properties.pin_memory != all_metadatas[i].tensor_properties.pin_memory:
                raise ValueError(
                    f'ShardedTensor pin_memory property does not match from different ranks! '
                    f'Found is_pinned(): {global_sharded_tensor_metadata.tensor_properties.pin_memory} on rank 0, '
                    f'and is_pinned(): {all_metadatas[i].tensor_properties.pin_memory} on rank {i}'
                )

        # check if shards_metadata have overlap shards
        validate_non_overlapping_shards_metadata(global_sharded_tensor_metadata.shards_metadata)

        # check if the shards_metadata is compatible with overall size of the sharded tensor.
        check_tensor(global_sharded_tensor_metadata.shards_metadata, global_sharded_tensor_metadata.size)

        # done validation, add to metadata and local_shards
        sharded_tensor._metadata = global_sharded_tensor_metadata
        sharded_tensor._local_shards = local_shards
        # no sharding spec for sharded tensors that initialized from this API.
        sharded_tensor._sharding_spec = None

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
            shards_metadata, dims, tensor_init_params.tensor_properties, )

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
            shards_metadata, dims, tensor_init_params.tensor_properties, )

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

    def is_pinned(self) -> bool:
        """
        Returns True if the sharded tensor (each local shard) resides in pinned memory.
        """
        return self._metadata.tensor_properties.pin_memory

    def is_contiguous(self) -> bool:
        """
        Returns True if the sharded tensor (each local shard) is contiguous in memory
        in the order specified by memory format.
        """
        return self._metadata.tensor_properties.memory_format == torch.contiguous_format

    @property
    def shape(self):
        return self._metadata.size

    @property
    def requires_grad(self):
        return self._metadata.tensor_properties.requires_grad

    @property
    def dtype(self):
        return self._metadata.tensor_properties.dtype

    @property
    def layout(self):
        return self._metadata.tensor_properties.layout

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

    create_op = tensor_init_params.create_op
    dtype = tensor_init_params.tensor_properties.dtype
    layout = tensor_init_params.tensor_properties.layout
    requires_grad = tensor_init_params.tensor_properties.requires_grad
    memory_format = tensor_init_params.tensor_properties.memory_format
    pin_memory = tensor_init_params.tensor_properties.pin_memory

    if create_op == CreateOp.ONES:
        return torch.ones(*size, dtype=dtype, layout=layout,
                          device=local_device, pin_memory=pin_memory,
                          requires_grad=requires_grad,)
    elif create_op == CreateOp.EMPTY:
        return torch.empty(*size, dtype=dtype, layout=layout,
                           device=local_device, requires_grad=requires_grad,
                           # NB: memory_format param is not accepted by torch.ones
                           memory_format=memory_format, pin_memory=pin_memory,)
    elif tensor_init_params.create_op == CreateOp.ZEROS:
        return torch.zeros(*size,
                           dtype=dtype,
                           layout=layout,
                           device=local_device,
                           pin_memory=pin_memory,
                           requires_grad=requires_grad,)
    elif tensor_init_params.create_op == CreateOp.RAND:
        return torch.rand(*size,
                          dtype=dtype,
                          layout=layout,
                          device=local_device,
                          pin_memory=pin_memory,
                          requires_grad=requires_grad,)
    elif tensor_init_params.create_op == CreateOp.FULL:
        return torch.full(size=size,
                          fill_value=tensor_init_params.fill_value,
                          layout=layout,
                          dtype=dtype,
                          requires_grad=requires_grad,
                          device=local_device, )
    else:
        raise ValueError(f'Unsupported create_op: {tensor_init_params.create_op}')
