from typing import List, NamedTuple, Optional

import torch
import torch.distributed as dist
from torch.autograd.grad_mode import _unsafe_preserve_version_counter

from ._fsdp_common import chunk_with_empty, print_and_raise_internal, to_dtype_if_needed
from ._fsdp_param import FSDPParam, ShardedState


"""
[Note: ``.data`` Usage in Per-Parameter FSDP]
In the copy-out part of the foreach all-gather, we cat the all-gathered
parameter shards, writing the result out to the newly allocated unsharded
parameter data. Autograd detects this as an extra in-place op, e.g.:
    RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.cuda.FloatTensor [15, 3]], which is
output 0 of AsStridedBackward0, is at version 3; expected version 2 instead.

Using ``.data`` is one way to avoid this version counter increase from the cat
(see NOTE [ Version Counter Sharing ] in the pytorch repo). However, we can
also do so with the new ``_unsafe_preserve_version_counter`` context, avoiding
the ``.data`` usage and being more explicit with our intention. Thus, this is
the approach that we take.
"""


class AllGatherResult(NamedTuple):
    flat_tensor_shards: List[torch.Tensor]
    all_gather_event: Optional[torch.cuda.Event]
    all_gather_work: Optional[dist.distributed_c10d.Work]


@torch.no_grad()
def foreach_all_gather(
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.cuda.Stream,
    all_gather_stream: torch.cuda.Stream,
    use_uint8: bool,
    device: torch.device,
) -> Optional[AllGatherResult]:
    """
    Args:
        use_int8 (bool): If ``False``, then this requires that all parameters
            have the same dtype, which is used for the all-gather. If ``True``,
            then this permits different dtypes and converts everything to uint8
            for the all-gather.

    Returns:
        Optional[AllGatherResult]: The all-gathered interleaved parameter
        shards (``flat_tensor_shards``) to be copied out and a CUDA event
        recording the end of the all-gather collective. If there are no
        parameters that need to be all-gathered, returns ``None``.
    """
    fsdp_params = [
        fsdp_param
        for fsdp_param in fsdp_params
        if fsdp_param.state != ShardedState.UNSHARDED
    ]
    if len(fsdp_params) == 0:
        return None
    # Already checked at construction time for a uniform unsharded parameter
    # dtype if not using uint8
    dtype = torch.uint8 if use_uint8 else fsdp_params[0].unsharded_param_data_dtype
    world_size = group.size()
    group_rank = group.rank()
    if use_uint8:
        total_padded_unsharded_numel = sum(
            fsdp_param._padded_unsharded_size.numel()
            # Multiply by the dtype size to convert to uint8 numel
            * fsdp_param.unsharded_param_data_dtype.itemsize
            for fsdp_param in fsdp_params
        )
    else:
        total_padded_unsharded_numel = sum(
            fsdp_param._padded_unsharded_size.numel() for fsdp_param in fsdp_params
        )
    total_sharded_numel = total_padded_unsharded_numel // world_size
    with torch.cuda.stream(all_gather_copy_in_stream):
        # - Copy in
        flat_tensor = torch.empty(
            (total_padded_unsharded_numel,), dtype=dtype, device=device
        )
        flat_tensor_shards = [
            flat_tensor.narrow(0, total_sharded_numel * rank, total_sharded_numel)
            for rank in range(world_size)
        ]
        input_flat_tensor_shard = flat_tensor_shards[group_rank]
        shards_to_cat = [fsdp_param.all_gather_input for fsdp_param in fsdp_params]
        if use_uint8:
            shards_to_cat = [t.view(torch.uint8) for t in shards_to_cat]
        # This cat implicitly casts to `flat_tensor_shard.dtype` (if needed)
        torch.cat(shards_to_cat, out=input_flat_tensor_shard)
        all_gather_copy_in_event = torch.cuda.Event()
        all_gather_copy_in_event.record()
    all_gather_stream.wait_event(all_gather_copy_in_event)
    with torch.cuda.stream(all_gather_stream):
        # - All-gather
        all_gather_work = dist.all_gather_into_tensor(
            output_tensor=flat_tensor,
            input_tensor=input_flat_tensor_shard,
            group=group,
            async_op=async_op,
        )
        all_gather_event = torch.cuda.Event()
        all_gather_event.record()
        return AllGatherResult(flat_tensor_shards, all_gather_event, all_gather_work)


def foreach_all_gather_copy_out(
    flat_tensor_shards: List[torch.Tensor],
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    use_uint8: bool,
) -> None:
    world_size = group.size()
    flat_tensor_offset = 0  # [0, total_sharded_numel - 1]
    for fsdp_param in fsdp_params:
        param_shards: List[torch.Tensor] = []
        padded_sharded_numel = fsdp_param.all_gather_input_numel
        dtype_size = fsdp_param.unsharded_param_data_dtype.itemsize
        for rank in range(world_size):
            param_shard_start = rank * padded_sharded_numel
            numel_in_shard = min(
                fsdp_param._unsharded_size.numel() - param_shard_start,
                padded_sharded_numel,
            )
            if use_uint8:
                # Multiply by the dtype size to convert to uint8 numel
                numel_in_shard *= dtype_size
            if numel_in_shard > 0:
                param_shard = flat_tensor_shards[rank].narrow(
                    0, flat_tensor_offset, numel_in_shard
                )
                if use_uint8:
                    # View back to the expected dtype before the cat
                    param_shard = param_shard.view(
                        fsdp_param.unsharded_param_data_dtype
                    )
                param_shards.append(param_shard)
        offset_increment = padded_sharded_numel
        if use_uint8:
            offset_increment *= dtype_size
        flat_tensor_offset += offset_increment
        fsdp_param.alloc_unsharded_param()
        # See [Note: ``.data`` Usage in Per-Parameter Sharding]
        out = fsdp_param._unsharded_param_data[: fsdp_param._unsharded_size.numel()]
        if fsdp_param.unsharded_param_data_dtype == torch.float8_e4m3fn:
            # HACK: While `aten::cat` is not supported, view as uint8
            param_shards = [t.view(torch.uint8) for t in param_shards]
            torch.cat(param_shards, out=out.data.view(torch.uint8))
            continue
        with _unsafe_preserve_version_counter(out):
            torch.cat(
                param_shards,
                out=out,
            )


@torch.no_grad()
def foreach_reduce_scatter(
    fsdp_params: List[FSDPParam],
    unsharded_grads: List[torch.Tensor],
    group: dist.ProcessGroup,
    reduce_scatter_stream: torch.cuda.Stream,
    orig_dtype: torch.dtype,
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    predivide_factor: float,
    postdivide_factor: float,
) -> torch.cuda.Event:
    """
    Args:
        unsharded_grads (List[torch.Tensor]): This list owns the strong
            references to the unsharded gradients computed by autograd, so
            clearing this list frees the gradients.
    """
    if len(fsdp_params) != len(unsharded_grads) or len(fsdp_params) == 0:
        print_and_raise_internal(
            f"Invalid args: {len(fsdp_params)} parameters and {len(unsharded_grads)} gradients"
        )
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        print_and_raise_internal(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    world_size = group.size()
    total_padded_unsharded_numel = sum(
        fsdp_param._padded_unsharded_size.numel() for fsdp_param in fsdp_params
    )
    total_sharded_numel = total_padded_unsharded_numel // world_size
    current_stream = torch.cuda.current_stream()
    reduce_scatter_stream.wait_stream(current_stream)
    with torch.cuda.stream(reduce_scatter_stream):
        # - Copy in
        flat_grad = torch.empty(
            (total_padded_unsharded_numel,), dtype=reduce_dtype, device=device
        )
        foreach_copy_dests: List[torch.Tensor] = []
        foreach_copy_srcs: List[torch.Tensor] = []
        flat_grad_offset = 0  # [0, total_sharded_numel - 1]
        for fsdp_param, grad in zip(fsdp_params, unsharded_grads):
            padded_sharded_numel = fsdp_param._padded_sharded_size.numel()
            chunks = chunk_with_empty(grad, world_size, dim=0)
            for rank in range(world_size):
                grad_shard_start = rank * padded_sharded_numel
                numel_in_shard = min(
                    fsdp_param._unsharded_size.numel() - grad_shard_start,
                    padded_sharded_numel,
                )
                if numel_in_shard > 0:
                    flat_grad_start = flat_grad_offset + rank * total_sharded_numel
                    foreach_copy_dests.append(
                        flat_grad[flat_grad_start : flat_grad_start + numel_in_shard]
                    )
                    foreach_copy_srcs.append(chunks[rank].view(-1))
            flat_grad_offset += padded_sharded_numel
            del chunks
        torch._foreach_copy_(foreach_copy_dests, foreach_copy_srcs)
        del foreach_copy_dests
        del foreach_copy_srcs
        _div_if_needed(flat_grad, predivide_factor)
        # Record a CUDA event in the reduce-scatter stream to mark the end of
        # the copy-in for the reduce-scatter input
        copy_in_event = torch.cuda.Event()
        copy_in_event.record()
        flat_grad_shard = torch.empty(
            (total_sharded_numel,), dtype=reduce_dtype, device=device
        )
        # - Reduce-scatter
        dist.reduce_scatter_tensor(output=flat_grad_shard, input=flat_grad, group=group)
        # - Post-reduce-scatter
        _div_if_needed(flat_grad_shard, postdivide_factor)
        flat_grad_shard = to_dtype_if_needed(flat_grad_shard, orig_dtype)
        # - View out and accumulate
        flat_grad_offset = 0  # [0, total_sharded_numel - 1]
        for fsdp_param in fsdp_params:
            padded_sharded_numel = fsdp_param._padded_sharded_size.numel()
            sharded_numel = fsdp_param._sharded_size.numel()
            new_sharded_grad = flat_grad_shard[
                flat_grad_offset : flat_grad_offset + sharded_numel
            ].view(fsdp_param._sharded_size)
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # TODO: If we want to overlap the D2H copy while accumulating
                # gradient, we need to refactor to run the CPU add kernel at
                # the end of backward (since otherwise we do not have a way to
                # run a CPU callback). We use a blocking copy (to non-pinned)
                # memory for now.
                gpu_sharded_grad = new_sharded_grad
                cpu_sharded_grad = (
                    fsdp_param._cpu_sharded_grad
                    if not to_accumulate_grad
                    else torch.empty_like(new_sharded_grad, device="cpu")
                )
                cpu_sharded_grad.copy_(
                    gpu_sharded_grad, non_blocking=not to_accumulate_grad
                )
                new_sharded_grad = cpu_sharded_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here (by not keeping the reference) without
                # waiting for the D2H copy since future ops in the RS stream
                # (i.e. copy-in/RS) are serialized to run after the copy
            new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
            if to_accumulate_grad:
                fsdp_param.sharded_param.grad += new_sharded_dtensor_grad
            elif new_sharded_dtensor_grad._local_tensor.numel() > 0:
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            # Else the parameter is padding-only
            if fsdp_param.offload_to_cpu:
                # Record an event on which to block the CPU thread to ensure
                # that the D2H gradient offload completes before the optimizer
                grad_offload_event = torch.cuda.Event()
                grad_offload_event.record()
                fsdp_param._grad_offload_event = grad_offload_event
            flat_grad_offset += padded_sharded_numel
        reduce_scatter_view_out_event = torch.cuda.Event()
        reduce_scatter_view_out_event.record()
    # Only after the copy-in finishes can we free the gradients, which were
    # computed in the default stream
    current_stream.wait_event(copy_in_event)
    unsharded_grads.clear()
    # The flat gradient shard is allocated in the RS stream and used in the
    # default stream (for optimizer). We need to make sure that its memory is
    # not reused in the RS stream for subsequent RSs. We do so without extra
    # synchronization since the sharded parameters hold references to the data
    # through the end of backward, and the RS stream transitively waits for the
    # default stream before the next backward.
    return reduce_scatter_view_out_event


def _div_if_needed(tensor: torch.Tensor, div_factor: float) -> None:
    if div_factor > 1:
        tensor.div_(div_factor)
