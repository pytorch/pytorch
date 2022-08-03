import io
from typing import Any, Dict, List, Tuple, Optional, cast
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.shard import Shard

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)

from .metadata import (
    BytesReadRequest,
    BytesStorageMetadata,
    TensorReadRequest,
    TensorStorageMetadata,
    Metadata,
    MetadataIndex,
)
from .resharding import (
    _prepare_generic_tensor_read,
)
from .storage import (
    StorageReader,
)

from .utils import _DistWrapper

def _create_shard_metadata(size: torch.Size) -> ShardMetadata:
    return ShardMetadata(
        shard_offsets=[0] * len(size),
        shard_sizes=list(size),
    )

def _create_shard_for(tensor: Tensor) -> Shard:
    return Shard(
        tensor=tensor,
        metadata=_create_shard_metadata(tensor.size()),
    )

def _reshard_and_prepare_read_request(
    state_dict: Dict[str, Any], metadata_from_storage: Metadata
) -> Tuple[List[BytesReadRequest], List[TensorReadRequest]]:
    """
    Use the loaded metadata and the current state dict to map the saved tensors to current tensor
    """
    tensor_read_requests = []
    bytes_read_requests = []
    storage_md = cast(Dict[MetadataIndex, str], metadata_from_storage.storage_data)
    for fqn, obj in state_dict.items():
        md = metadata_from_storage.state_dict_metadata[fqn]
        if isinstance(obj, ShardedTensor):
            local_shards = obj.local_shards()
        elif isinstance(obj, torch.Tensor):
            local_shards = [_create_shard_for(obj)]
        else:
            if isinstance(md, BytesStorageMetadata):
                bytes_io = io.BytesIO()
                brr = BytesReadRequest(
                    bytes=bytes_io,
                    storage_key=storage_md[MetadataIndex(fqn)],
                    fqn=fqn
                )
                bytes_read_requests.append(brr)
            else:
                raise ValueError(
                    f"Invalid checkpoint metadata for {fqn}, " +
                    f"expected BytesStorageMetadata but found {type(md)}"
                )
            continue

        if isinstance(md, TensorStorageMetadata):
            checkpoint_shards = md.chunks
        else:
            raise ValueError(
                f"Invalid checkpoint metadata for {fqn}, " +
                f"expected TensorStorageMetadata but found {type(md)}"
            )

        tensor_read_requests += _prepare_generic_tensor_read(fqn, checkpoint_shards, local_shards, storage_md)



    return (bytes_read_requests, tensor_read_requests)


def load_state_dict(
    state_dict: Dict[str, Any],
    storage_reader: StorageReader,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False
) -> None:
    """
    Load a distributed state_dict in SPMD style.

    Each rank will try to read the least amount of data necessary
    to fullfill the requested `state_dict`.

    When loading ShardedTensor instances, each rank only
    reads data for their local shards.

    All tensors in ``state_dict`` must be allocated on their
    destination device prior to calling this function.

    All non-tensor data is loaded using `torch.load()` and modified in place
    on state_dict.

    Users must call `load_state_dict` on the root module to ensure load
    pos-processing and non-tensor data properly propagates.

    This function can be used for local inference and load a checkpoint
    produced by ``save_state_dict`` without having a process group initialized
    by passing ``no_dist=True`` and by using Tensors instead of ShardedTensors.

    Args:
        state_dict (Dict[str, Any]) : The state_dict to load. Note that this
            state dict will updated in places.
        storage_reader (StorageReader): StorageReader used to load data from.
        process_group (ProcessGroup): ProcessGroup to be used for cross-rank synchronization
        coordinator_rank (int): Rank to use to coordinate the checkpoint, rank0 is used by default
        no_dist (bool): Don't attempt to load in SPMD style. Default to False

    Returns:
        None.

    Examples
        >>> my_model = MyModule()
        >>> optimizer = Adagrad(my_model.parameters())
        >>> model_state_dict = my_model.state_dict()
        >>> fs_storage_loader = torch.distributed._shard.checkpoint.FileSystemLoader("/checkpoint/1")

        >>> torch.distributed._shard.checkpoint.load_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_reader=fs_storage_loader,
        >>> )

        >>> # module.load_state_dict() function might have customized steps
        >>> # to flush the state_dict, must call it to
        >>> # ensure correct behavior.
        >>> my_model.load_state_dict(model_state_dict)

    .. note:: load_state_dict uses collectives to coordinate reads across ranks.
        For NCCL-based process groups, internal tensor representations of objects
        must be moved to the GPU device before communication takes place. In this
        case, the device used is given by ``torch.cuda.current_device()`` and it
        is the user's responsibility to ensure that this is set so that each rank
        has an individual GPU, via ``torch.cuda.set_device()``
    """
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)

    def load_model():
        metadata = storage_reader.read_metadata()
        bytes_read_requests, tensor_read_requests = _reshard_and_prepare_read_request(
            state_dict=state_dict, metadata_from_storage=metadata
        )
        bytes_futures = storage_reader.read_bytes(bytes_read_requests)
        tensor_futures = storage_reader.read_tensors(tensor_read_requests)

        bytes_futures.wait()

        # Addtional steps are required to convert the bytes to its original type
        # Note that this is NOT inplace,
        # it creating a new object and replace what's in the state dict
        for req in bytes_read_requests:
            # Ensure the BytesIO is rewound
            req.bytes.seek(0)
            state_dict[req.fqn] = torch.load(req.bytes)

        tensor_futures.wait()

    distW.all_gather("checkpoint read", load_model)
