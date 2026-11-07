r"""Experimental FSDP2 customization APIs.

FSDP copies nonzero-dimension shards through intermediate buffers by default,
except in parameter groups where some rank pads a ``Shard(i)``, ``i > 0``,
shard. The ``with_native_copy`` functions, which those groups use by default,
copy directly between parameter layouts and collective buffers instead.
Register the reduce-scatter copy-in on a fully sharded model::

    model.set_reduce_scatter_input_fn(reduce_scatter_input_fn_with_native_copy)

Each native CUDA copy is one launch for all tensors of a parameter group, with
metadata of a few integers per tensor, and can improve performance for tensors
sharded on a nonzero dimension, such as ``Shard(1)`` over ``[2, F, D]``. Strided
collective buffers take a slower path. Benchmark the callbacks on the target
workload before enabling them.

All-gather extensions can return ``AllGatherInput`` records from
``fsdp_pre_all_gather`` to declare each payload's concatenation dimension. FSDP
batches these copies before calling ``fsdp_post_all_gather`` to reconstruct the
parameter. Existing hooks returning tensors remain supported.

.. warning::
    These APIs are experimental. Callback signatures and supported FSDP
    internals may change without backward compatibility.
"""

from collections.abc import Callable

import torch
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGatherInput
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _native_all_gather_output_fn,
    _native_reduce_scatter_input_fn,
)


__all__ = [
    "AllGatherInput",
    "all_gather_output_fn_with_native_copy",
    "reduce_scatter_input_fn_with_native_copy",
]

AllGatherInput.__module__ = "torch.distributed.fsdp.experimental"


def all_gather_output_fn_with_native_copy(
    all_gather_output: torch.Tensor,
    outputs: list[torch.Tensor],
    *,
    split_sizes: list[int],
    outer_sizes: list[int],
    world_size: int,
) -> None:
    r"""Copy gathered payloads directly into their final layout.

    See the module documentation for the performance tradeoffs.
    """
    _native_all_gather_output_fn(
        all_gather_output,
        outputs,
        split_sizes=split_sizes,
        outer_sizes=outer_sizes,
        world_size=world_size,
    )


def reduce_scatter_input_fn_with_native_copy(
    unsharded_grads: list[torch.Tensor],
    *,
    shard_dims: list[int],
    world_size: int,
) -> Callable[[torch.Tensor], None]:
    r"""Prepare gradients for a native copy into the reduce-scatter buffer.

    Register with
    :meth:`torch.distributed.fsdp.FSDPModule.set_reduce_scatter_input_fn`.
    See the module documentation for the performance tradeoffs.

    Contiguous nonzero-dimension shards copy directly into the collective buffer,
    which zero-pads uneven ones. Noncontiguous gradients sharded on a nonzero dim
    take the default reorder.
    """
    return _native_reduce_scatter_input_fn(
        unsharded_grads, shard_dims=shard_dims, world_size=world_size
    )
