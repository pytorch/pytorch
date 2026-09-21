r"""Experimental FSDP2 customization APIs.

FSDP uses optimized copies for supported nonzero-dimension shards by default.
Register the ``with_intermediate_copy`` callbacks to copy nonzero-dimension
shards through intermediate buffers. Each direction can be selected independently
on a fully sharded model::

    model.set_all_gather_output_fn(all_gather_output_fn_with_intermediate_copy)
    model.set_reduce_scatter_input_fn(reduce_scatter_input_fn_with_intermediate_copy)

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
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _copy_all_gather_outputs,
    AllGatherResult,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import _get_dim0_padded_size
from torch.distributed.fsdp._fully_shard._fsdp_param import (
    _AllGatherOutputLayout,
    FSDPParam,
)


__all__ = [
    "AllGatherInput",
    "ReduceScatterInput",
    "all_gather_output_fn_with_intermediate_copy",
    "reduce_scatter_input_fn_with_intermediate_copy",
]


def all_gather_output_fn_with_intermediate_copy(
    fsdp_params: list[FSDPParam],
    all_gather_result: AllGatherResult,
    world_size: int,
) -> None:
    r"""Copy gathered payloads through intermediate buffers when needed.

    Nonempty payloads concatenated along a nonzero dimension copy through
    intermediate buffers, then concatenate into their final layout. Dimension-0
    and empty payloads copy directly. Register with
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
    reorder_infos: list[tuple[torch.Tensor, torch.Tensor, _AllGatherOutputLayout]] = []
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
        for output, layout in zip(
            fsdp_param.all_gather_outputs, fsdp_param.all_gather_copy_layouts
        ):
            if layout.dim != 0 and output.numel():
                copy_output = torch.empty_like(output)
                reorder_infos.append((copy_output, output, layout))
            else:
                copy_output = output
            split_with_sizes_out.append(copy_output)

    _copy_all_gather_outputs(
        all_gather_output,
        all_gather_input_split_sizes,
        split_with_sizes_out,
        world_size,
    )
    _reassemble_all_gather_outputs(reorder_infos, world_size)


def _reassemble_all_gather_outputs(
    reorder_infos: list[tuple[torch.Tensor, torch.Tensor, _AllGatherOutputLayout]],
    world_size: int,
) -> None:
    outputs = tuple(out for _, out, _ in reorder_infos if not out.is_inference())
    with torch.autograd._unsafe_preserve_version_counter(outputs):
        for copy_output, output, layout in reorder_infos:
            copy_size = list(layout.input_size)
            copy_size[0] *= world_size
            chunks = torch.chunk(copy_output.view(copy_size), world_size, dim=0)
            gathered_size = list(layout.input_size)
            gathered_size[layout.dim] *= world_size
            torch.cat(chunks, dim=layout.dim, out=output.view(gathered_size))


def reduce_scatter_input_fn_with_intermediate_copy(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    world_size: int,
) -> ReduceScatterInput:
    r"""Pack nonzero-dimension gradients into intermediate buffers before copying.

    When the group has more than one rank, nonzero-dimension gradients are
    chunked and concatenated into intermediate buffers grouped by destination
    rank, then copied into the reduce-scatter buffer. Register with
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
