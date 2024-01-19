from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils._contextlib import _DecoratorContextManager

from ._fsdp_common import (
    get_dim0_padded_size,
    print_and_raise_internal,
    to_dtype_if_needed,
)
from ._fsdp_param import FSDPParam


class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: Optional[torch.cuda.Event]
    all_gather_work: Optional[dist.distributed_c10d.Work]
    all_gather_input_numels: List[int]


class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.cuda.Event  # copy-out


class AllGatherStateHolder:
    def __init__(self):
        self._state: Optional[AllGatherState] = None

    def put(self, state: AllGatherState) -> None:
        assert self._state is None, "Expects to hold only one all-gather state"
        self._state = state

    def pop(self) -> Optional[AllGatherState]:
        state = self._state
        self._state = None
        return state


@torch.no_grad()
def foreach_all_gather(
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.cuda.Stream,
    all_gather_stream: torch.cuda.Stream,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[AllGatherResult]:
    # Assume that all parameters are in some sharded state
    world_size, rank = (group.size(), group.rank())
    # - Copy in
    with torch.cuda.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = [
            fsdp_param.all_gather_input for fsdp_param in fsdp_params
        ]
        inp_split_sizes = [inp.numel() for inp in param_all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        all_gather_output_numel = all_gather_input_numel * world_size
        all_gather_output = torch.empty(
            (all_gather_output_numel,), dtype=dtype, device=device
        )
        all_gather_input = all_gather_output.narrow(
            0, all_gather_input_numel * rank, all_gather_input_numel
        )
        foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
        torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
        del param_all_gather_inputs
        all_gather_copy_in_event = torch.cuda.Event()
        all_gather_copy_in_event.record()
    all_gather_stream.wait_event(all_gather_copy_in_event)
    with torch.cuda.stream(all_gather_stream):
        # - All-gather
        all_gather_work = dist.all_gather_into_tensor(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        all_gather_event = torch.cuda.Event()
        all_gather_event.record()
        return AllGatherResult(
            all_gather_output, all_gather_event, all_gather_work, inp_split_sizes
        )


def foreach_all_gather_copy_out(
    all_gather_output: torch.Tensor,  # 1D
    all_gather_input_numels: List[int],
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
) -> None:
    world_size = group.size()
    dtype, device = all_gather_output.dtype, all_gather_output.device
    for all_gather_input_numel, fsdp_param in zip(all_gather_input_numels, fsdp_params):
        fsdp_param.init_all_gather_output(
            all_gather_input_numel, world_size, dtype, device
        )  # no-op after 1st call
        fsdp_param.alloc_all_gather_output()
    # TODO: Replace with foreach copy to prepare for custom kernel.
    split_sizes = all_gather_input_numels * world_size
    splits = torch.split(all_gather_output, split_sizes, dim=0)
    all_param_shards: List[List[torch.Tensor]] = []
    outs: List[torch.Tensor] = []
    for fsdp_param_idx, fsdp_param in enumerate(fsdp_params):
        param_shards: List[torch.Tensor] = [
            splits[fsdp_param_idx + rank * len(fsdp_params)]
            for rank in range(world_size)
        ]
        all_param_shards.append(param_shards)
        outs.append(fsdp_param.all_gather_output)
    with _unsafe_preserve_version_counters(outs):
        for param_shards, out in zip(all_param_shards, outs):
            torch.cat(param_shards, out=out)


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
    padded_unsharded_sizes = tuple(
        get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    current_stream = torch.cuda.current_stream()
    reduce_scatter_stream.wait_stream(current_stream)
    with torch.cuda.stream(reduce_scatter_stream):
        reduce_scatter_input = torch.empty(
            (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
        )
        foreach_reduce_scatter_copy_in(
            fsdp_params, unsharded_grads, reduce_scatter_input, world_size
        )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        # Record to mark the end of the reduce-scatter copy-in in the RS stream
        copy_in_event = torch.cuda.Event()
        copy_in_event.record()
        reduce_scatter_output = torch.empty(
            (reduce_scatter_output_numel,), dtype=reduce_dtype, device=device
        )
        dist.reduce_scatter_tensor(
            output=reduce_scatter_output, input=reduce_scatter_input, group=group
        )
        _div_if_needed(reduce_scatter_output, postdivide_factor)
        reduce_scatter_output = to_dtype_if_needed(reduce_scatter_output, orig_dtype)
        # - View out and accumulate
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            sharded_numel = fsdp_param.sharded_size.numel()
            new_sharded_grad = reduce_scatter_output[
                flat_grad_offset : flat_grad_offset + sharded_numel
            ].view(fsdp_param.sharded_size)
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # TODO: If we want to overlap the D2H copy while accumulating
                # gradient, we need to refactor to run the CPU add kernel at
                # the end of backward (since otherwise we do not have a way to
                # run a CPU callback). We use a blocking copy (to non-pinned)
                # memory for now.
                gpu_sharded_grad = new_sharded_grad
                cpu_sharded_grad = (
                    fsdp_param.cpu_sharded_grad
                    if not to_accumulate_grad
                    else torch.empty_like(new_sharded_grad, device="cpu")
                )
                cpu_sharded_grad.copy_(
                    gpu_sharded_grad, non_blocking=not to_accumulate_grad
                )
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here (by not keeping the reference) without
                # waiting for the D2H copy since future ops in the RS stream
                # (i.e. copy-in/RS) are serialized to run after the D2H copy
                new_sharded_grad = cpu_sharded_grad
                # Record an event on which to block the CPU thread to ensure
                # that the D2H gradient offload completes before the optimizer
                grad_offload_event = torch.cuda.Event()
                grad_offload_event.record()
                fsdp_param._grad_offload_event = grad_offload_event
            new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
            if to_accumulate_grad:
                fsdp_param.sharded_param.grad += new_sharded_dtensor_grad
            elif new_sharded_dtensor_grad._local_tensor.numel() > 0:
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            flat_grad_offset += padded_sharded_numel
        reduce_scatter_view_out_event = torch.cuda.Event()
        reduce_scatter_view_out_event.record()
    # Only after the copy-in finishes can we free the gradients, which were
    # computed in the default stream
    current_stream.wait_event(copy_in_event)
    unsharded_grads.clear()
    # The reduce-scatter output is allocated in the RS stream and used in the
    # default stream (for optimizer). We need to make sure that its memory is
    # not reused in the RS stream for subsequent RSs. We do so without extra
    # synchronization since the sharded parameters hold references to the data
    # through the end of backward, and the RS stream transitively waits for the
    # default stream before the next backward.
    return reduce_scatter_view_out_event


def foreach_reduce_scatter_copy_in(
    fsdp_params: List[FSDPParam],
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    # Use `torch.split` to reduce CPU overhead since it pushes for loops of
    # slices into C++ only
    foreach_copy_dests: List[torch.Tensor] = []  # 1D tensors
    foreach_copy_srcs: List[torch.Tensor] = []  # 1D tensors
    split_sizes: List[int] = []
    is_padding_mask: List[bool] = []
    for rank in range(world_size):
        for fsdp_param in fsdp_params:
            split_sizes.extend(fsdp_param.padded_unsharded_chunk_numels[rank])
            is_padding_mask.extend(fsdp_param.is_padding_mask[rank])
    splits = torch.split(reduce_scatter_input, split_sizes, dim=0)
    all_flat_grad_splits: List[Tuple[torch.Tensor, ...]] = []
    for fsdp_param, grad in zip(fsdp_params, unsharded_grads):
        # Flatten once per gradient to reduce number of `view` calls
        flat_grad_splits = torch.split(grad.view(-1), fsdp_param.unsharded_chunk_numels)
        all_flat_grad_splits.append(flat_grad_splits)
    for rank in range(world_size):
        for fsdp_param_idx in range(len(fsdp_params)):
            if (split := all_flat_grad_splits[fsdp_param_idx][rank]).numel() > 0:
                foreach_copy_srcs.append(split)
            # Else pure padding
    for is_padding, split in zip(is_padding_mask, splits):
        if is_padding:
            continue
        foreach_copy_dests.append(split)
    if len(foreach_copy_dests) != len(foreach_copy_srcs):
        print_and_raise_internal(
            f"dests={len(foreach_copy_dests)} srcs={len(foreach_copy_srcs)}"
        )
    torch._foreach_copy_(foreach_copy_dests, foreach_copy_srcs)


def _div_if_needed(tensor: torch.Tensor, div_factor: float) -> None:
    if div_factor > 1:
        tensor.div_(div_factor)


class _unsafe_preserve_version_counters(_DecoratorContextManager):
    # Same as `_unsafe_preserve_version_counter` but only entering/exiting the
    # context manager once for a list of tensors to reduce CPU overhead
    def __init__(self, tensors: List[torch.Tensor]) -> None:
        self.tensors = tensors
        self.prev_versions = [t._version for t in tensors]

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args) -> None:
        for tensor, prev_version in zip(self.tensors, self.prev_versions):
            torch._C._autograd._unsafe_set_version_counter(tensor, prev_version)
