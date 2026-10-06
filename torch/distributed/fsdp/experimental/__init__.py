r"""Experimental FSDP2 customization APIs.

FSDP copies nonzero-dimension shards through intermediate buffers by default.
The ``with_native_copy`` functions copy directly between parameter layouts and
collective buffers instead. Register the reduce-scatter copy-in on a fully
sharded model::

    model.set_reduce_scatter_input_fn(reduce_scatter_input_fn_with_native_copy)

The native CUDA copies can improve performance for wide contiguous tensors with
small ``outer_size`` (the product of dimensions before the shard dimension),
such as ``Shard(1)`` over ``[2, F, D]``. Large outer sizes, skinny tensors, and
strided collective buffers can increase CPU time, GPU time, and metadata memory.
Benchmark the callbacks on the target workload before enabling them.

All-gather extensions can return ``AllGatherInput`` records from
``fsdp_pre_all_gather`` to declare each payload's concatenation dimension and
gathered shape. FSDP batches these copies before calling ``fsdp_post_all_gather``
to reconstruct the parameter. Existing hooks returning tensors remain supported.

.. warning::
    These APIs are experimental. Callback signatures and supported FSDP
    internals may change without backward compatibility.
"""

from collections.abc import Callable

import torch
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGatherInput
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _default_all_gather_output_fn,
    _default_reduce_scatter_input_fn,
    AllGatherOutputFn,
    PrepareReduceScatterInputsFn,
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
    split_sizes: list[int],
    outer_sizes: list[int],
    world_size: int,
) -> None:
    r"""Copy gathered payloads directly into their final layout.

    See the module documentation for the performance tradeoffs. Groups with a
    tensor payload smaller than its cached output use the default copy-out.
    """
    if any(
        output.numel() != split_size * world_size
        for output, split_size in zip(outputs, split_sizes)
    ):
        _default_all_gather_output_fn(
            all_gather_output, outputs, split_sizes, outer_sizes, world_size
        )
        return
    torch.ops.fsdp._all_gather_copy_out_(
        outputs, all_gather_output, split_sizes, outer_sizes, world_size
    )


def reduce_scatter_input_fn_with_native_copy(
    unsharded_grads: list[torch.Tensor],
    shard_dims: list[int],
    world_size: int,
) -> Callable[[torch.Tensor], None]:
    r"""Prepare gradients for a native copy into the reduce-scatter buffer.

    Register with
    :meth:`torch.distributed.fsdp.FSDPModule.set_reduce_scatter_input_fn`.
    See the module documentation for the performance tradeoffs.

    Contiguous nonzero-dimension shards copy directly into the collective buffer.
    Noncontiguous gradients use the existing chunk-and-concatenate reorder.
    Groups with mixed gradient dtypes use the default copy-in.
    """
    if len({grad.dtype for grad in unsharded_grads}) > 1:
        return _default_reduce_scatter_input_fn(unsharded_grads, shard_dims, world_size)
    num_leading_dims = [0] * len(unsharded_grads)
    if world_size > 1:
        for i, shard_dim in enumerate(shard_dims):
            if shard_dim == 0:
                continue
            if unsharded_grads[i].is_contiguous():
                num_leading_dims[i] = shard_dim
            else:
                chunks = torch.chunk(unsharded_grads[i], world_size, dim=shard_dim)
                unsharded_grads[i] = torch.cat(chunks, dim=0)

    def copy_in(output: torch.Tensor) -> None:
        torch.ops.fsdp._reduce_scatter_copy_in_(
            output.view(world_size, -1),
            unsharded_grads,
            num_leading_dims,
            world_size,
        )

    return copy_in
