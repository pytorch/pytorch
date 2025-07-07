import torch
from torch._C._distributed_c10d import (
    ProcessGroup,
    Store,
)

__all__ = [
    "split_group",
    "merge_group",
]

def split_group(self, new_ranks: list[int], ...) -> ProcessGroup:
    """
    This creates a new subgroup using the specified ranks. The current rank must be included in the list of new_ranks.

    shrink in NCCL is just a special case of split and we can automatically detect it.
    """
    pass

def merge_group(self, store: Store, options: MergeOptions) -> ProcessGroup:
    """
    Merge multiple groups together. For the case where we want N independent groups to be merged, you should split to a single rank and then call merge_group across the size 1 groups.

    Ranks are assigned by the backend.
    """
    pass
