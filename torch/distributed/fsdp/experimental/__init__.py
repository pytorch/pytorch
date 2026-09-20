"""Experimental FSDP2 APIs.

.. warning::
    These APIs are experimental. Callback signatures and supported FSDP
    internals may change without backward compatibility.
"""

import math
from functools import partial

import torch
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatterInput
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _reassemble_all_gather_outputs,
    AllGatherResult,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import _get_dim0_padded_size
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState


__all__ = [
    "ReduceScatterInput",
    "all_gather_output_fn_for_nonzero_dim_shards",
    "reduce_scatter_input_fn_for_nonzero_dim_shards",
]


def all_gather_output_fn_for_nonzero_dim_shards(
    fsdp_params: list[FSDPParam],
    all_gather_result: AllGatherResult,
    world_size: int,
) -> None:
    """Optimize all-gather output copies for ``Shard(1)`` and higher shard dimensions.

    For supported layouts, copy gathered rank shards directly into contiguous
    regions of the final outputs, avoiding temporary buffers and a separate
    reassembly. All-gather extensions, post-forward shards, and other unsupported
    layouts use temporary outputs followed by reassembly. ``Shard(0)`` parameters
    use the usual copy path, so groups may mix shard dimensions.

    Register with :meth:`torch.distributed.fsdp.FSDPModule.set_all_gather_output_fn`,
    which documents the callback contract.
    """
    all_gather_output = all_gather_result.all_gather_output
    device = all_gather_output.device
    copy_outputs: list[torch.Tensor] = []
    num_prefixes: list[int] = []
    reorder_infos: list[tuple[FSDPParam, list[torch.Tensor]]] = []
    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
        all_gather_result.param_all_gather_input_numels,
        all_gather_result.param_all_gather_input_dtypes,
        fsdp_params,
    ):
        fsdp_param.init_all_gather_outputs(
            all_gather_input_numels, all_gather_input_dtypes, world_size, device
        )
        fsdp_param.alloc_all_gather_outputs()
        outputs = fsdp_param.all_gather_outputs
        shard_dim = fsdp_param.fsdp_placement.dim
        if shard_dim == 0:
            copy_outputs.extend(outputs)
            num_prefixes.extend([1] * len(outputs))
            continue

        num_leading_elements = math.prod(
            fsdp_param.padded_sharded_param_size[:shard_dim]
        )
        if (
            fsdp_param.sharded_state == ShardedState.SHARDED
            and num_leading_elements > 0
            and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        ):
            copy_outputs.extend(outputs)
            num_prefixes.extend([num_leading_elements] * len(outputs))
            continue

        # Extension and post-forward shard layouts may require a separate reorder.
        outputs = [torch.empty_like(t) for t in outputs]
        copy_outputs.extend(outputs)
        num_prefixes.extend([1] * len(outputs))
        reorder_infos.append((fsdp_param, outputs))
    non_inference_outputs = tuple(t for t in copy_outputs if not t.is_inference())
    with torch.autograd._unsafe_preserve_version_counter(non_inference_outputs):
        torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
            copy_outputs,
            all_gather_output,
            all_gather_result.all_gather_input_split_sizes,
            num_prefixes,
            world_size,
        )
    _reassemble_all_gather_outputs(reorder_infos, world_size)


def reduce_scatter_input_fn_for_nonzero_dim_shards(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    world_size: int,
) -> ReduceScatterInput:
    """Optimize reduce-scatter inputs for ``Shard(1)`` and higher shard dimensions.

    Copy contiguous unsharded gradients directly into the reduce-scatter buffer,
    avoiding an intermediate chunk-and-concatenate reorder.
    Noncontiguous gradients use the usual chunk-and-concatenate path.
    Nonzero-dimension sharding must be even. ``Shard(0)`` gradients and groups of
    size one use the usual preparation, so groups may mix shard dimensions.

    Register with :meth:`torch.distributed.fsdp.FSDPModule.set_reduce_scatter_input_fn`,
    which documents the callback contract.
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
    copy_in = foreach_reduce_scatter_copy_in
    if any(num_leading_dims):
        copy_in = partial(_copy_reduce_scatter_input, num_leading_dims=num_leading_dims)
    return ReduceScatterInput(padded_unsharded_sizes, copy_in)


def _copy_reduce_scatter_input(
    unsharded_grads: list[torch.Tensor],
    output: torch.Tensor,
    world_size: int,
    *,
    num_leading_dims: list[int],
) -> None:
    torch.ops.fsdp._chunk_cat_with_prefixes_(
        output.view(world_size, -1), unsharded_grads, num_leading_dims, world_size
    )
