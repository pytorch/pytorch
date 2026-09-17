"""Experimental FSDP2 APIs.

.. warning::
    These APIs are experimental. Callback signatures and supported FSDP
    internals may change without backward compatibility.
"""

import math

import torch
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _copy_all_gather_outputs,
    _reassemble_all_gather_outputs,
    AllGatherResult,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import _get_dim0_padded_size
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState


__all__ = [
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
    views of the final outputs, avoiding temporary buffers and a separate
    reassembly. All-gather extensions, post-forward shards, and other unsupported
    layouts use temporary outputs followed by reassembly. ``Shard(0)`` parameters
    use the usual copy path, so groups may mix shard dimensions.

    Register with :meth:`torch.distributed.fsdp.FSDPModule.set_all_gather_output_fn`,
    which documents the callback contract.
    """
    all_gather_output = all_gather_result.all_gather_output
    device = all_gather_output.device
    copy_outputs: list[torch.Tensor] = []
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
            continue

        num_leading_elements = math.prod(
            fsdp_param.padded_sharded_param_size[:shard_dim]
        )
        if (
            fsdp_param.sharded_state == ShardedState.SHARDED
            and num_leading_elements > 0
            and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        ):
            # Each prefix indexes a contiguous suffix sharded on its dim 0.
            # Copy rank chunks directly into these views of the final buffer.
            copy_outputs.extend(
                view
                for tensor in outputs
                for view in tensor.view(num_leading_elements, -1).unbind(0)
            )
            continue

        # Extension and post-forward shard layouts may require a separate reorder.
        outputs = [torch.empty_like(t) for t in outputs]
        copy_outputs.extend(outputs)
        reorder_infos.append((fsdp_param, outputs))
    # Split sizes use collective-buffer elements, or bytes for mixed dtypes.
    copy_split_sizes = [
        t.numel() * t.element_size() // (world_size * all_gather_output.element_size())
        for t in copy_outputs
    ]
    _copy_all_gather_outputs(
        all_gather_output,
        copy_split_sizes,
        copy_outputs,
        world_size,
    )
    _reassemble_all_gather_outputs(reorder_infos, world_size)


def reduce_scatter_input_fn_for_nonzero_dim_shards(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    world_size: int,
) -> list[torch.Size]:
    """Optimize reduce-scatter inputs for ``Shard(1)`` and higher shard dimensions.

    Pass contiguous unsharded gradients to the dimension-0 copy operation through
    contiguous views, avoiding an intermediate chunk-and-concatenate reorder.
    Noncontiguous gradients use the usual chunk-and-concatenate path.
    Nonzero-dimension sharding must be even. ``Shard(0)`` gradients and groups of
    size one use the usual preparation, so groups may mix shard dimensions.

    Register with :meth:`torch.distributed.fsdp.FSDPModule.set_reduce_scatter_input_fn`,
    which documents the callback contract.
    """
    copy_in_grads: list[torch.Tensor] = []
    padded_unsharded_sizes: list[torch.Size] = []
    for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
        shard_dim = fsdp_param.fsdp_placement.dim
        if world_size > 1 and shard_dim != 0:
            if unsharded_grad.size(shard_dim) % world_size != 0:
                raise AssertionError(
                    f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} "
                    f"{world_size=}"
                )
            if unsharded_grad.is_contiguous():
                copy_in_grads.extend(unsharded_grad.flatten(0, shard_dim - 1).unbind(0))
                # Even nonzero-dim shards need no padding.
                padded_unsharded_sizes.append(unsharded_grad.size())
                continue
            unsharded_grad = torch.cat(
                torch.chunk(unsharded_grad, world_size, dim=shard_dim),
                dim=0,
            )
            unsharded_grads[i] = unsharded_grad
        copy_in_grads.append(unsharded_grad)
        padded_unsharded_sizes.append(
            _get_dim0_padded_size(unsharded_grad.size(), world_size)
        )
    unsharded_grads[:] = copy_in_grads
    return padded_unsharded_sizes
