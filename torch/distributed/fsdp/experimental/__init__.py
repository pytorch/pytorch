r"""Experimental FSDP2 customization APIs.

FSDP copies nonzero-dimension shards through intermediate buffers by default.
Register the ``with_native_copy`` callbacks to copy directly between parameter
layouts and collective buffers. Each direction can be selected independently
on a fully sharded model::

    model.set_all_gather_output_fn(all_gather_output_fn_with_native_copy)
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

import torch
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    AllGatherInput,
    ReduceScatterInput,
)
from torch.distributed.fsdp._fully_shard._fsdp_collectives import AllGatherResult
from torch.distributed.fsdp._fully_shard._fsdp_common import _get_dim0_padded_size
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam


__all__ = [
    "AllGatherInput",
    "ReduceScatterInput",
    "all_gather_output_fn_with_native_copy",
    "reduce_scatter_input_fn_with_native_copy",
]


def all_gather_output_fn_with_native_copy(
    fsdp_params: list[FSDPParam],
    all_gather_result: AllGatherResult,
    world_size: int,
) -> None:
    r"""Copy gathered payloads directly into their final layout.

    Register with
    :meth:`torch.distributed.fsdp.FSDPModule.set_all_gather_output_fn`.
    See the module documentation for the performance tradeoffs.
    """
    all_gather_output = all_gather_result.all_gather_output
    device = all_gather_output.device
    copy_outputs: list[torch.Tensor] = []
    outer_sizes: list[int] = []
    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
        all_gather_result.param_all_gather_input_numels,
        all_gather_result.param_all_gather_input_dtypes,
        fsdp_params,
    ):
        fsdp_param.init_all_gather_outputs(
            all_gather_input_numels, all_gather_input_dtypes, world_size, device
        )
        fsdp_param.alloc_all_gather_outputs()
        copy_outputs.extend(fsdp_param.all_gather_outputs)
        for layout in fsdp_param.all_gather_copy_layouts:
            outer_sizes.append(layout.outer_size)
    non_inference_outputs = tuple(t for t in copy_outputs if not t.is_inference())
    with torch.autograd._unsafe_preserve_version_counter(non_inference_outputs):
        torch.ops.fsdp._all_gather_copy_out_(
            copy_outputs,
            all_gather_output,
            all_gather_result.all_gather_input_split_sizes,
            outer_sizes,
            world_size,
        )


def reduce_scatter_input_fn_with_native_copy(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    world_size: int,
) -> ReduceScatterInput:
    r"""Prepare gradients for a native copy into the reduce-scatter buffer.

    Register with
    :meth:`torch.distributed.fsdp.FSDPModule.set_reduce_scatter_input_fn`.
    See the module documentation for the performance tradeoffs.

    Contiguous nonzero-dimension shards copy directly into the collective buffer.
    Noncontiguous gradients use the existing chunk-and-concatenate reorder.
    Groups with only Shard(0) gradients, or of size one, use the original operator.
    """
    padded_unsharded_sizes: list[torch.Size] = []
    num_leading_dims: list[int] = []
    for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
        shard_dim = fsdp_param.fsdp_placement.dim
        if world_size > 1 and shard_dim != 0:
            if unsharded_grad.size(shard_dim) % world_size != 0:
                raise AssertionError(
                    f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} "
                    f"{world_size=}"
                )
            if unsharded_grad.is_contiguous():
                num_leading_dims.append(shard_dim)
                # Even nonzero-dim shards need no padding.
                padded_unsharded_sizes.append(unsharded_grad.size())
                continue
            unsharded_grad = torch.cat(
                torch.chunk(unsharded_grad, world_size, dim=shard_dim),
                dim=0,
            )
            unsharded_grads[i] = unsharded_grad
        num_leading_dims.append(0)
        padded_unsharded_sizes.append(
            _get_dim0_padded_size(unsharded_grad.size(), world_size)
        )

    def copy_in(
        unsharded_grads: list[torch.Tensor],
        output: torch.Tensor,
        world_size: int,
    ) -> None:
        torch.ops.fsdp._reduce_scatter_copy_in_(
            output.view(world_size, -1),
            unsharded_grads,
            num_leading_dims,
            world_size,
        )

    return ReduceScatterInput(padded_unsharded_sizes, copy_in)
