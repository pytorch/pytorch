from typing import Sequence

import torch
from torch.distributed._shard.metadata import ShardMetadata


DEPRECATE_MSG = "Please use DTensor instead and we are deprecating ShardedTensor."


def narrow_tensor_by_index(
    tensor: torch.Tensor,
    offsets: Sequence[int],
    sizes: Sequence[int],
) -> torch.Tensor:
    """
    Narrow the tensor according to ``offsets`` and ``sizes``.
    """
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            # Reshape to get shard for this rank and we don't want autograd
            # recording here for the narrow op and 'local_shard' should be a
            # leaf variable in the autograd graph.
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor


def narrow_tensor(tensor: torch.Tensor, metadata: ShardMetadata) -> torch.Tensor:
    """
    Narrow the tensor according to the metadata
    """
    return narrow_tensor_by_index(tensor, metadata.shard_offsets, metadata.shard_sizes)
