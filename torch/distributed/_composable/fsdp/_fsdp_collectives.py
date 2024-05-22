import contextlib
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from ._fsdp_common import (
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
)
from ._fsdp_param import FSDPParam
from torch._inductor import config as inductor_config

lib = torch.library.Library("fsdp", "DEF")


class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: Optional[torch.cuda.Event]
    all_gather_work: Optional[dist.distributed_c10d.Work]
    # For each parameter, the all-gather input dtype for each input
    param_all_gather_input_dtypes: List[List[torch.dtype]]
    # For each parameter, the all-gather input numel for each input
    param_all_gather_input_numels: List[List[int]]
    # 1D flattened version of `param_all_gather_input_numels` saved to avoid
    # CPU overhead from recomputing
    all_gather_input_split_sizes: List[int]


lib.define("all_gather_copy_in(SymInt all_gather_input_numel, SymInt world_size, SymInt rank, ScalarType dtype, Device device, SymInt[] inp_split_sizes, Tensor[] param_all_gather_inputs) -> (Tensor, Tensor)")

@torch.library.impl(lib, "all_gather_copy_in", "Meta")
def all_gather_copy_in(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs):
    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device="meta"
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
    return all_gather_input, all_gather_output

def all_gather_copy_in_impl(
    all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs
):
    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device=device
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
    return all_gather_input, all_gather_output

@torch.library.impl(lib, "all_gather_copy_in", "CUDA")
def all_gather_copy_in(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs):
    return all_gather_copy_in_impl(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs)


lib.define("split_contiguous_view_as_strided(Tensor all_gather_output, SymInt[] all_gather_input_numels, SymInt[][] orig_sizes, SymInt[][] contiguous_orig_strides) -> Tensor[]")

@torch.library.impl(lib, "split_contiguous_view_as_strided", "Meta")
def split_contiguous_view_as_strided(all_gather_output, all_gather_input_numels, orig_sizes, contiguous_orig_strides):
    splits = torch.split(all_gather_output, all_gather_input_numels, dim=1)
    out = []
    for i in range(len(orig_sizes)):
        split = splits[i]
        orig_size = orig_sizes[i]
        contiguous_orig_stride = contiguous_orig_strides[i]
        split_flattened = split.contiguous().view(split.numel())
        split_unpadded = torch.as_strided(
            split_flattened,
            orig_size,
            contiguous_orig_stride,
            storage_offset=0,
        )
        out.append(split_unpadded)
    return out

@torch.library.impl(lib, "split_contiguous_view_as_strided", "CUDA")
def split_contiguous_view_as_strided(all_gather_output, all_gather_input_numels, orig_sizes, contiguous_orig_strides):
    splits = torch.split(all_gather_output, all_gather_input_numels, dim=1)
    out = []
    for i in range(len(orig_sizes)):
        split = splits[i]
        orig_size = orig_sizes[i]
        contiguous_orig_stride = contiguous_orig_strides[i]
        split_flattened = split.contiguous().view(split.numel())
        split_unpadded = torch.as_strided(
            split_flattened,
            orig_size,
            contiguous_orig_stride,
            storage_offset=0,
        )
        out.append(split_unpadded)
    return out


lib.define("chunk_cat(Tensor[] tensors, int dim, int num_chunks) -> Tensor")

@torch.library.impl(lib, "chunk_cat", "Meta")
def chunk_cat(tensors, dim, num_chunks):
    return torch._chunk_cat(tensors, dim, num_chunks)

@torch.library.impl(lib, "chunk_cat", "CUDA")
def chunk_cat(tensors, dim, num_chunks):
    return torch._chunk_cat(tensors, dim, num_chunks)



@torch.no_grad()
def foreach_all_gather(
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.cuda.Stream,
    all_gather_stream: torch.cuda.Stream,
    device: torch.device,
) -> Optional[AllGatherResult]:
    world_size, rank = group.size(), group.rank()
    with torch.cuda.stream(all_gather_copy_in_stream):
        param_all_gather_inputs: List[List[torch.Tensor]] = [
            fsdp_param.all_gather_inputs for fsdp_param in fsdp_params
        ]
        (
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
        if dtype == torch.uint8:
            all_gather_inputs = [
                t.view(torch.uint8) for ts in param_all_gather_inputs for t in ts
            ]
        else:
            all_gather_inputs = [t for ts in param_all_gather_inputs for t in ts]
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        if not torch._dynamo.compiled_autograd.compiled_autograd_enabled:
            all_gather_output = torch.empty(
                (all_gather_input_numel * world_size,), dtype=dtype, device=device
            )
            all_gather_input = all_gather_output.narrow(
                0, all_gather_input_numel * rank, all_gather_input_numel
            )
            foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
            torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
        else:
            # TODO(yf225): support the len(self.all_gather_outputs) > 1 case (i.e. support custom fsdp_pre_all_gather)
            assert all(len(all_gather_inputs) == 1 for all_gather_inputs in param_all_gather_inputs)
            all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
                all_gather_input_numel,
                world_size,
                rank,
                dtype,
                device,
                inp_split_sizes,
                [all_gather_inputs[0] for all_gather_inputs in param_all_gather_inputs],
            )
        del param_all_gather_inputs
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with torch.cuda.stream(all_gather_stream):
        all_gather_work = dist.all_gather_into_tensor(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        all_gather_event = all_gather_stream.record_event()
        return AllGatherResult(
            all_gather_output,
            all_gather_event,
            all_gather_work,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            inp_split_sizes,
        )


@torch.no_grad()
def foreach_all_gather_copy_out(
    all_gather_result: AllGatherResult,
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
) -> None:
    (
        all_gather_output,
        all_gather_event,
        all_gather_work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_input_split_sizes,
    ) = all_gather_result
    if all_gather_event is not None:  # sync op
        torch.cuda.current_stream().wait_event(all_gather_event)
    if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
        all_gather_work.wait()
    world_size, device = group.size(), all_gather_output.device
    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
        param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params
    ):
        fsdp_param.init_all_gather_outputs(
            all_gather_input_numels, all_gather_input_dtypes, world_size, device
        )
        fsdp_param.init_unsharded_param()  # needed for compile
        fsdp_param.alloc_all_gather_outputs()
    all_gather_output = all_gather_output.view(world_size, -1)
    # NOTE: This is the biggest difference between eager and compile code path.
    # In eager, we directly copy from `all_gather_output` into `fsdp_param.all_gather_output` (`fsdp_param._unsharded_param` will be updated because of shared storage),
    # but in compile path we copy from `as_strided(all_gather_output)` into `fsdp_param._unsharded_param` to avoid having `fsdp_param.all_gather_output` as graph input.
    # They are equivalent and must produce the same result.
    if not torch._dynamo.compiled_autograd.compiled_autograd_enabled:
        gen = (t for fsdp_param in fsdp_params for t in fsdp_param.all_gather_outputs)
        if all_gather_output.dtype == torch.uint8:
            out = [t.view(world_size, -1).view(torch.uint8) for t in gen]
        else:
            out = [t.view(world_size, -1) for t in gen]
        torch.split_with_sizes_copy(
            all_gather_output, all_gather_input_split_sizes, dim=1, out=out
        )
    else:
        # TODO(yf225): support uint8 similar to the eager case (i.e. support fsdp_pre_all_gather)
        assert all(len(fsdp_param.all_gather_outputs) == 1 for fsdp_param in fsdp_params)
        unsharded_params = []
        orig_sizes = []
        contiguous_orig_strides = []
        for i, fsdp_param in enumerate(fsdp_params):
            unsharded_param = fsdp_param._unsharded_param
            if fsdp_param.is_dtensor:
                unsharded_param = unsharded_param.to_local()
            unsharded_params.append(unsharded_param)
            orig_sizes.append(fsdp_param._orig_size)
            contiguous_orig_strides.append(fsdp_param._contiguous_orig_stride)
        out = torch.ops.fsdp.split_contiguous_view_as_strided(all_gather_output, all_gather_input_split_sizes, orig_sizes, contiguous_orig_strides)
        for i, unsharded_param in enumerate(unsharded_params):
            ctx = contextlib.nullcontext()
            if not torch._dynamo.compiled_autograd.compiled_autograd_enabled:
                ctx = torch.autograd._unsafe_preserve_version_counter(unsharded_param)
            with torch.no_grad(), ctx:
                unsharded_param.copy_(out[i])


@torch.no_grad()
def foreach_reduce(
    fsdp_params: List[FSDPParam],
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.cuda.Stream,
    orig_dtype: torch.dtype,
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
    all_reduce_stream: torch.cuda.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
) -> Tuple[torch.cuda.Event, Optional[torch.Tensor]]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    predivide_factor, postdivide_factor = _get_gradient_divide_factors(
        reduce_scatter_group, all_reduce_group, reduce_dtype
    )
    world_size = reduce_scatter_group.size()
    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
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
            unsharded_grads, reduce_scatter_input, world_size
        )
        # Only after the copy-in finishes can we free the gradients, which were
        # computed in the default stream
        current_stream.wait_stream(reduce_scatter_stream)
        unsharded_grads.clear()
        reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
        _div_if_needed(reduce_scatter_input, predivide_factor)
        dist.reduce_scatter_tensor(
            output=reduce_output,
            input=reduce_scatter_input,
            group=reduce_scatter_group,
            op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
        )
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return post_reduce_stream.record_event(), partial_reduce_output
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            all_reduce_stream.wait_stream(reduce_scatter_stream)
            with torch.cuda.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
                )
    with torch.cuda.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) if not
                # accumulating gradients since the CPU add kernel depends on
                # the copy result and we cannot run the add as a callback
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = reduce_scatter_stream.record_event()
            new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
            if to_accumulate_grad:
                fsdp_param.sharded_param.grad += new_sharded_dtensor_grad
            else:
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return post_reduce_event, None


def foreach_reduce_scatter_copy_in(
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    if not torch._dynamo.compiled_autograd.compiled_autograd_enabled:
        torch._chunk_cat(
            unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
        )
    else:
        out = torch.ops.fsdp.chunk_cat(unsharded_grads, dim=0, num_chunks=world_size)
        with torch.no_grad():
            reduce_scatter_input.copy_(out)


def _get_all_gather_input_metadatas(
    param_all_gather_inputs: List[List[torch.Tensor]],
) -> Tuple[List[List[torch.dtype]], List[List[int]], torch.dtype]:
    param_all_gather_input_dtypes: List[List[torch.dtype]] = []
    param_all_gather_input_numels: List[List[int]] = []
    all_gather_dtype = param_all_gather_inputs[0][0].dtype
    for all_gather_inputs in param_all_gather_inputs:
        input_dtypes: List[torch.dtype] = []
        input_numels: List[int] = []
        for all_gather_input in all_gather_inputs:
            if all_gather_input.dtype != all_gather_dtype:
                all_gather_dtype = torch.uint8
            input_dtypes.append(all_gather_input.dtype)
            input_numels.append(all_gather_input.numel())
        param_all_gather_input_dtypes.append(input_dtypes)
        param_all_gather_input_numels.append(input_numels)
    return (
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_dtype,
    )


def _get_gradient_divide_factors(
    reduce_scatter_group: dist.ProcessGroup,
    all_reduce_group: Optional[dist.ProcessGroup],
    reduce_dtype: torch.dtype,
) -> Union[Tuple[None, None], Tuple[float, float]]:
    # For fp32/bf16, we do not need to worry about overflow/underflow, so we
    # use NCCL's built-in division to avoid separate div kernels
    if reduce_dtype in (torch.float32, torch.bfloat16):
        return None, None
    data_parallel_size = reduce_scatter_group.size()
    if all_reduce_group is not None:
        data_parallel_size *= all_reduce_group.size()
    # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
    # overflow/underflow. For N data parallel workers, each worker computes
    # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
    # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
    factor: int = 1
    while data_parallel_size % factor == 0 and data_parallel_size / factor > factor:
        factor *= 2
    factor = float(factor)
    return (factor, data_parallel_size / factor)


def _div_if_needed(tensor: torch.Tensor, div_factor: Optional[float]) -> None:
    if div_factor is not None and div_factor > 1:
        tensor.div_(div_factor)
