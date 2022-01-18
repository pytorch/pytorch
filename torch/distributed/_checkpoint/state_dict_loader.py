from typing import Any, Dict, List
import torch
from torch.futures import Future
from .metadata import ReadWriteRequest, Metadata
from torch.distributed._sharded_tensor import (
    ShardedTensor,
)
from .storage_reader import StorageReader


def load_state_dict(state_dict: Dict[str, Any], storage_reader: StorageReader) -> None:
    """
    The same as load_async, but blocking until everything is finished
    """
    futures = load_state_dict_async(state_dict=state_dict, storage_reader=storage_reader)
    torch.futures.wait_all(futures)


def load_state_dict_async(state_dict: Dict[str, Any], storage_reader: StorageReader) -> List[Future]:
    """
    Run the load request in parallel.
    """
    metadata = storage_reader.read_metadata()
    read_requests = _reshard_and_prepare_read_request(state_dict=state_dict, metadata_from_stroage=metadata)
    return [storage_reader.read(rr) for rr in read_requests]

# -------------- private functions --------------
def _reshard_and_prepare_read_request(state_dict: Dict[str, Any], metadata_from_stroage: Metadata) -> List[ReadWriteRequest]:
    """
    Use the loaded metadata and the current state dict to map the saved tensors to current tensors

    NOTE:
    The metadata loaded for storage is not reference in this code.
    I did not implement ANY resharding at all. This code only work if the two model are exactly the same.
    """
    read_requests = []
    for fqn, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            # Reshape to 1D, the might make loading eaiser
            target_tensor = tensor.detach().reshape(-1)
            storage_size = target_tensor.nelement() * target_tensor.element_size()

            rr = ReadWriteRequest(
                target_tensor=target_tensor,
                storage_key=fqn,
                offset=0,
                length=storage_size,
            )

            read_requests.append(rr)
        elif isinstance(tensor, ShardedTensor):
            for shard in tensor.local_shards():
                suffix = "_".join([str(i) for i in shard.metadata.shard_offsets + shard.metadata.shard_sizes])
                storage_key = f"{fqn}_{suffix}"

                # Reshape to 1D, the might make loading eaiser
                target_tensor = shard.tensor.detach().reshape(-1)
                storage_size = target_tensor.nelement() * target_tensor.element_size()

                wr = ReadWriteRequest(
                    target_tensor=target_tensor,
                    storage_key=storage_key,
                    offset=0,
                    length=storage_size,
                )
                read_requests.append(rr)

    return read_requests
