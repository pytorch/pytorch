"""
Functional collectives.

Functional collectives return new tensors instead of mutating their inputs and
are traceable by ``torch.compile``. In eager mode the result is an
:class:`AsyncCollectiveTensor` which waits on the collective on first use.
"""

from torch.distributed._functional_collectives import (
    all_gather_single,
    all_gather_single_coalesced,
    all_reduce,
    all_reduce_coalesced,
    all_to_all_single,
    AsyncCollectiveTensor,
    broadcast,
    permute_tensor,
    reduce_scatter_single,
    reduce_scatter_single_coalesced,
    wait_tensor,
    wait_tensors,
)


__all__ = [
    "AsyncCollectiveTensor",
    "all_gather_single",
    "all_gather_single_coalesced",
    "all_reduce",
    "all_reduce_coalesced",
    "all_to_all_single",
    "broadcast",
    "permute_tensor",
    "reduce_scatter_single",
    "reduce_scatter_single_coalesced",
    "wait_tensor",
    "wait_tensors",
]

for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
