import torch
from torch.distributed._shard.metadata import ShardMetadata

def narrow_tensor(tensor: torch.Tensor, metadata: ShardMetadata):
    """
    narrow the tensor according to the metadata
    """
    narrowed_tensor = tensor
    shard_offsets = metadata.shard_offsets
    shard_sizes = metadata.shard_sizes
    for idx, (offset, size) in enumerate(zip(shard_offsets, shard_sizes)):
        if size < tensor.size(idx):
            # Reshape to get shard for this rank and we don't want autograd
            # recording here for the narrow op and 'local_shard' should be a
            # leaf variable in the autograd graph.
            narrowed_tensor = narrowed_tensor.narrow(
                idx,
                shard_offsets[idx],
                shard_sizes[idx]
            )
    return narrowed_tensor
