r"""Experimental FSDP2 customization APIs.

FSDP uses optimized copies for supported nonzero-dimension shards by default.
Register the ``with_reorder`` callbacks to use the original copy-and-reorder
behavior. Each direction can be selected independently on a fully sharded model::

    model.set_all_gather_output_fn(all_gather_output_fn_with_reorder)
    model.set_reduce_scatter_input_fn(reduce_scatter_input_fn_with_reorder)

The ``for_nonzero_dim_shards`` names remain aliases for the optimized defaults.

.. warning::
    These APIs are experimental. Callback signatures and supported FSDP
    internals may change without backward compatibility.
"""

import torch
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatterInput
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _copy_all_gather_outputs,
    _default_all_gather_output_fn as all_gather_output_fn_for_nonzero_dim_shards,
    _default_reduce_scatter_input_fn as reduce_scatter_input_fn_for_nonzero_dim_shards,
    _reassemble_all_gather_outputs,
    AllGatherResult,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import _get_dim0_padded_size
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam


__all__ = [
    "ReduceScatterInput",
    "all_gather_output_fn_for_nonzero_dim_shards",
    "all_gather_output_fn_with_reorder",
    "reduce_scatter_input_fn_for_nonzero_dim_shards",
    "reduce_scatter_input_fn_with_reorder",
]


def all_gather_output_fn_with_reorder(
    fsdp_params: list[FSDPParam],
    all_gather_result: AllGatherResult,
    world_size: int,
) -> None:
    r"""Use the original all-gather copy followed by nonzero-dimension reassembly.

    Nonzero-dimension shards copy through temporary outputs. Shard(0) copies
    directly to the final outputs. Register with
    :meth:`torch.distributed.fsdp.FSDPModule.set_all_gather_output_fn`, which
    documents the callback contract.
    """
    (
        all_gather_output,
        _,
        _,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_input_split_sizes,
    ) = all_gather_result
    device = all_gather_output.device
    split_with_sizes_out: list[torch.Tensor] = []
    shard_i_copy_infos: list[tuple[FSDPParam, list[torch.Tensor]]] = []
    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
        param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params
    ):
        fsdp_param.init_all_gather_outputs(
            all_gather_input_numels,
            all_gather_input_dtypes,
            world_size,
            device,
        )
        fsdp_param.alloc_all_gather_outputs()
        param_all_gather_outputs = fsdp_param.all_gather_outputs
        if fsdp_param.fsdp_placement.dim != 0:
            param_all_gather_outputs = [
                torch.empty_like(t) for t in param_all_gather_outputs
            ]
            shard_i_copy_infos.append((fsdp_param, param_all_gather_outputs))
        split_with_sizes_out.extend(param_all_gather_outputs)

    _copy_all_gather_outputs(
        all_gather_output,
        all_gather_input_split_sizes,
        split_with_sizes_out,
        world_size,
    )
    _reassemble_all_gather_outputs(shard_i_copy_infos, world_size)


def reduce_scatter_input_fn_with_reorder(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    world_size: int,
) -> ReduceScatterInput:
    r"""Reorder nonzero-dimension gradients before the original reduce-scatter copy.

    Nonzero-dimension shards use a separate chunk-and-concatenate reorder when
    the group has more than one rank. Register with
    :meth:`torch.distributed.fsdp.FSDPModule.set_reduce_scatter_input_fn`, which
    documents the callback contract.
    """
    if world_size > 1:
        for i, (fsdp_param, unsharded_grad) in enumerate(
            zip(fsdp_params, unsharded_grads)
        ):
            if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
                continue
            if unsharded_grad.size(shard_dim) % world_size != 0:
                raise AssertionError(
                    f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
                )
            chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
            unsharded_grads[i] = torch.cat(chunks, dim=0)

    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    return ReduceScatterInput(padded_unsharded_sizes, foreach_reduce_scatter_copy_in)
