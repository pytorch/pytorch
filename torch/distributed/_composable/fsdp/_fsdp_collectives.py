import contextlib
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed._functional_collectives import AsyncCollectiveTensor
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
    all_gather_input_numels: List[int]


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
    # - Copy in
    with torch.cuda.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = [
            fsdp_param.all_gather_input for fsdp_param in fsdp_params
        ]
        dtype = param_all_gather_inputs[0].dtype
        if not all(t.dtype == dtype for t in param_all_gather_inputs):
            raise NotImplementedError(
                f"Mixed dtype not supported yet: {[t.dtype for t in param_all_gather_inputs]}"
            )
        inp_split_sizes = [inp.numel() for inp in param_all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        if inductor_config.use_fsdp_custom_op:
            all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs)
        else:
            all_gather_input, all_gather_output = all_gather_copy_in_impl(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs)
        del param_all_gather_inputs
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with torch.cuda.stream(all_gather_stream):
        # - All-gather
        all_gather_work = dist.all_gather_into_tensor(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        all_gather_event = all_gather_stream.record_event()
        return AllGatherResult(
            all_gather_output, all_gather_event, all_gather_work, inp_split_sizes
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
        all_gather_input_numels,
    ) = all_gather_result
    if all_gather_event is not None:  # sync op
        torch.cuda.current_stream().wait_event(all_gather_event)
    if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
        all_gather_work.wait()
    world_size = group.size()
    dtype, device = all_gather_output.dtype, all_gather_output.device
    for all_gather_input_numel, fsdp_param in zip(all_gather_input_numels, fsdp_params):
        fsdp_param.init_all_gather_output(
            all_gather_input_numel, world_size, dtype, device
        )  # no-op after 1st call
        fsdp_param.alloc_all_gather_output()
        fsdp_param.init_unsharded_param()  # no-op after 1st call. Need to call here so that ._unsharded_param access below doesn't fail.
    all_gather_output = all_gather_output.view(world_size, -1)
    # NOTE: This is the biggest difference between eager and compile code path.
    # In eager, we directly copy from `all_gather_output` into `fsdp_param.all_gather_output` (`fsdp_param._unsharded_param` will be updated because of shared storage),
    # but in compile path we copy from `as_strided(all_gather_output)` into `fsdp_param._unsharded_param` to avoid having `fsdp_param.all_gather_output` as graph input.
    # They are equivalent and must produce the same result.
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        out = [
            fsdp_param.all_gather_output.view(world_size, -1) for fsdp_param in fsdp_params
        ]
        torch.split_with_sizes_copy(
            all_gather_output, all_gather_input_numels, dim=1, out=out
        )
    else:
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
        out = torch.ops.fsdp.split_contiguous_view_as_strided(all_gather_output, all_gather_input_numels, orig_sizes, contiguous_orig_strides)
        for i, unsharded_param in enumerate(unsharded_params):
            ctx = contextlib.nullcontext()
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
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
    divide_factors: Union[Tuple[None, None], Tuple[float, float]],
    all_reduce_group: Optional[dist.ProcessGroup],
    all_reduce_stream: torch.cuda.Stream,
) -> torch.cuda.Event:
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
    predivide_factor, postdivide_factor = divide_factors
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
        post_reduce_output = reduce_scatter_input.new_empty(
            (reduce_scatter_output_numel,)
        )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        _reduce_scatter(
            post_reduce_output,
            reduce_scatter_input,
            reduce_scatter_group,
            divide_factors,
        )
    view_out_stream = reduce_scatter_stream
    if all_reduce_group is not None:
        view_out_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with torch.cuda.stream(all_reduce_stream):
            _all_reduce(post_reduce_output, all_reduce_group, divide_factors)
    with torch.cuda.stream(view_out_stream):
        _div_if_needed(post_reduce_output, postdivide_factor)
        post_reduce_output = _to_dtype_if_needed(post_reduce_output, orig_dtype)
        # - View out and accumulate
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            new_sharded_grad = torch.as_strided(
                post_reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
            if to_accumulate_grad:
                fsdp_param.sharded_param.grad += new_sharded_dtensor_grad
            else:
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_view_out_event = view_out_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return post_reduce_view_out_event


def foreach_reduce_scatter_copy_in(
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    torch._chunk_cat(
        unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
    )


def _reduce_scatter(
    output: torch.Tensor,
    input: torch.Tensor,
    group: dist.ProcessGroup,
    divide_factors: Union[Tuple[None, None], Tuple[float, float]],
) -> None:
    if divide_factors[0]:
        dist.reduce_scatter_tensor(output, input, group=group)
    else:
        # Using NCCL's reduce-scatter to do the division by world size saves
        # extra memory read/write from a separate division kernel
        dist.reduce_scatter_tensor(output, input, op=ReduceOp.AVG, group=group)


def _all_reduce(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    divide_factors: Union[Tuple[None, None], Tuple[float, float]],
) -> None:
    if divide_factors[0]:
        dist.all_reduce(tensor, group=group)
    else:
        # saves extra memory read/write from a separate division kernel
        dist.all_reduce(tensor, op=ReduceOp.AVG, group=group)


def _div_if_needed(tensor: torch.Tensor, div_factor: Optional[float]) -> None:
    if div_factor is not None and div_factor > 1:
        tensor.div_(div_factor)
