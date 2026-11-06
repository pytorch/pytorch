import math
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, Literal, NamedTuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter

from ._all_gather_layout import (
    _adopt_layout_outputs,
    _check_layout_refill,
    _DefaultAllGatherCopyPlan,
    _PersistentBuffers,
    AllGatherFinalizeMetadata,
    AllGatherLayout,
    AllGatherParamMetadata,
    DEFAULT_ALL_GATHER_LAYOUT,
    DefaultAllGatherLayout,
)
from ._fsdp_api import _ReduceOp
from ._fsdp_param import FSDPParam


class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: torch.Event | None
    all_gather_work: dist.distributed_c10d.Work | None
    # For each parameter, the all-gather input dtype for each input
    param_all_gather_input_dtypes: list[list[torch.dtype]]
    # For each parameter, the all-gather input numel for each input
    param_all_gather_input_numels: list[list[int]]
    # 1D flattened version of `param_all_gather_input_numels` saved to avoid
    # CPU overhead from recomputing
    all_gather_input_split_sizes: list[int]
    # Per input, the product of the dims before the dim that ranks are
    # concatenated along (see _AllGatherOutputLayout)
    all_gather_input_outer_sizes: list[int]
    layout: AllGatherLayout = DEFAULT_ALL_GATHER_LAYOUT
    output_metadata: object | None = None
    # Kept until the collective completes, since a custom layout may pack it
    # into storage independent of the output
    all_gather_input: torch.Tensor | None = None


lib = torch.library.Library("fsdp", "FRAGMENT")

lib.define(
    """
    all_gather_copy_in(
        Tensor[] all_gather_inputs,
        Tensor all_gather_output,
        SymInt[] inp_split_sizes,
        SymInt all_gather_input_numel,
        SymInt rank
    ) -> (Tensor, Tensor)
    """
)


class DefaultAllocMixin:
    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)


class ProcessGroupAllocMixin:
    def __init__(self, group: dist.ProcessGroup, *args: Any, **kwargs: Any):
        self._group = group
        super().__init__(*args, **kwargs)

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        backend = self._group._get_backend(device)
        if backend.supports_tensor_alloc(device):
            size_1d = math.prod(int(s) for s in size)
            return backend.allocate_tensor(size_1d, dtype=dtype, device=device)
        return torch.empty(*size, dtype=dtype, device=device)


class SymmMemAllocMixin:
    def __init__(
        self,
        group: dist.ProcessGroup,
        backend: Literal["NCCL"] = "NCCL",
        *args: Any,
        **kwargs: Any,
    ):
        self._group = group
        symm_mem.set_backend(backend)
        # Force initialization of communicator; otherwise, the rendezvous may
        # see empty communicator.
        # TODO: Remove this, maybe by warning user to perform eager dist init.
        # For now, it is okay since it is just a one-time cost at init.
        dist.barrier(group=group)

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # Leverage MemPool to reuse the symmetric buffer, avoiding allocation
        # and rendezvous overhead
        mempool = symm_mem.get_mem_pool(device)
        with torch.get_device_module(device).use_mem_pool(mempool):
            return torch.empty(size, dtype=dtype, device=device)


class DefaultAllGather(DefaultAllocMixin, AllGather):
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class ProcessGroupAllocAllGather(ProcessGroupAllocMixin, AllGather):
    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__(group)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class SymmMemAllGather(SymmMemAllocMixin, AllGather):
    def __init__(
        self,
        group: dist.ProcessGroup,
        backend: Literal["NCCL"] = "NCCL",
    ) -> None:
        super().__init__(group, backend)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        # We are doing inplace all-gather, so we need to rendezvous the output tensor only
        symm_mem.rendezvous(output_tensor, group=group.group_name)
        # Calling regular all-gather would already cause libraries like NCCL to
        # use its optimized all-gather implementation for symmetric memory:
        # - Copy Engine All-Gather (when zero-CTA policy is enabled)
        # - Symmetric Kernel All-Gather (when zero-CTA policy is not enabled)
        return dist.all_gather_single(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class DefaultReduceScatter(DefaultAllocMixin, ReduceScatter):
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None:
        return dist.reduce_scatter_single(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


class ProcessGroupAllocReduceScatter(ProcessGroupAllocMixin, ReduceScatter):
    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__(group)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None:
        return dist.reduce_scatter_single(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


class SymmMemReduceScatter(SymmMemAllocMixin, ReduceScatter):
    def __init__(
        self,
        group: dist.ProcessGroup,
        backend: Literal["NCCL"] = "NCCL",
    ) -> None:
        super().__init__(group, backend)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None:
        symm_mem.rendezvous(input_tensor, group=group.group_name)
        symm_mem.rendezvous(output_tensor, group=group.group_name)
        # Calling regular reduce-scatter would already cause libraries like NCCL to
        # use its optimized reduce-scatter implementation for symmetric memory
        return dist.reduce_scatter_single(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


# These ops only call other ATen ops, so their impls register once to a
# composite-explicit alias key instead of once per backend: that covers every
# backend (Meta included) and, unlike CompositeImplicitAutograd, keeps the ops
# from being decomposed when traced, so compile still sees them.
# all_gather_copy_in fills its output by mutating views of it, so it uses the
# NonFunctional variant, which leaves out functional backends (XLA, Lazy). Its
# Meta impl, which skips the copy, takes precedence on meta tensors.
@torch.library.impl(lib, "all_gather_copy_in", "Meta")
def all_gather_copy_in_meta(
    all_gather_inputs: list[torch.Tensor],
    all_gather_output: torch.Tensor,
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    return all_gather_input, all_gather_output


@torch.library.impl(lib, "all_gather_copy_in", "CompositeExplicitAutogradNonFunctional")
def all_gather_copy_in_default(
    all_gather_inputs: list[torch.Tensor],
    all_gather_output: torch.Tensor,
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output


lib.define(
    "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
)


@torch.library.impl(lib, "split_with_sizes_copy", "CompositeExplicitAutograd")
def split_with_sizes_copy(
    all_gather_output: torch.Tensor,
    all_gather_input_split_sizes: list[int],
    dim: int = 0,
    *,
    out: list[torch.Tensor],
) -> None:
    torch.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
    )


# Unused within FSDP since the reduce-scatter copy-in moved to
# fsdp::chunk_cat_mixed_dtype; kept to avoid breaking potential external callers.
lib.define(
    "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
)


@torch.library.impl(lib, "chunk_cat", "CompositeExplicitAutograd")
def chunk_cat(
    tensors: list[torch.Tensor],
    dim: int,
    num_chunks: int,
    out: torch.Tensor,
) -> None:
    torch._chunk_cat(tensors, dim, num_chunks, out=out)


lib.define(
    "record_grad_output_stream(Tensor buffer, int stream_id, int device_index, int device_type) -> ()"
)


@torch.library.impl(lib, "record_grad_output_stream", "CompositeExplicitAutograd")
def _record_grad_output_stream_default(
    buffer: torch.Tensor,
    stream_id: int,
    device_index: int,
    device_type: int,
) -> None:
    # No-op default. Where free is host-side bookkeeping (e.g. CUDA) the block
    # stays mapped, so no read-before-free edge is needed. A backend whose free
    # releases device memory registers its own impl. See BucketRuntime.reduce_grads.
    return


def record_grad_output_stream(buffer: torch.Tensor, stream: torch.Stream) -> None:
    # A CPU buffer has no device free to order against and no stream to record,
    # so skip the op to avoid crashing on a CPU stream.
    if buffer.device.type == "cpu":
        return
    torch.ops.fsdp.record_grad_output_stream(
        buffer, stream.stream_id, stream.device_index, stream.device_type
    )


def _prepare_default_all_gather_outputs(
    fsdp_params: list[FSDPParam],
    all_gather_result: AllGatherResult,
    world_size: int,
) -> _DefaultAllGatherCopyPlan:
    device = all_gather_result.all_gather_output.device
    plan = _DefaultAllGatherCopyPlan(
        [],
        all_gather_result.all_gather_input_split_sizes,
        all_gather_result.all_gather_input_outer_sizes,
    )
    for fsdp_param, numels, dtypes in zip(
        fsdp_params,
        all_gather_result.param_all_gather_input_numels,
        all_gather_result.param_all_gather_input_dtypes,
    ):
        fsdp_param.init_all_gather_outputs(numels, dtypes, world_size, device)
        # Skips outputs that view the group's buffers, which it re-allocated
        fsdp_param.alloc_all_gather_outputs()
        plan.outputs.append(fsdp_param.all_gather_outputs)
    return plan


# Note [All-gather output fn]
# fn(all_gather_output, outputs, split_sizes=..., outer_sizes=..., world_size=...)
# runs under no_grad on the current stream and copies the flat rank-major
# all_gather_output into outputs, flat buffers that FSDP allocates. Each rank's
# chunk has split_sizes[i] elements for outputs[i], and the ranks' chunks, viewed
# as (outer_sizes[i], -1), are concatenated along dim 1 of outputs[i] viewed the
# same way. Outputs with outer_sizes[i] == 1 may be larger and then fill only a
# prefix. A uint8 all_gather_output has split_sizes in bytes, while outputs keep
# their dtypes. Of its arguments, fn may only write outputs, whose version
# counters FSDP preserves, and must keep none. FSDP skips it for a single rank or
# an empty all-gather.
def _default_all_gather_output_fn(
    all_gather_output: torch.Tensor,
    outputs: list[torch.Tensor],
    *,
    split_sizes: list[int],
    outer_sizes: list[int],
    world_size: int,
) -> None:
    """Copy outputs with outer_size > 1 via intermediate buffers, others directly."""
    byte_views = all_gather_output.dtype == torch.uint8
    split_outputs: list[torch.Tensor] = []
    reassembled: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for output, split_size, outer_size in zip(outputs, split_sizes, outer_sizes):
        copy_output = output
        if outer_size > 1:
            copy_output = torch.empty_like(output)
            reassembled.append((copy_output, output, outer_size))
        if byte_views and copy_output.nbytes == world_size * split_size:
            # View as bytes last: a view of a byte view keeps the byte view alive
            split_outputs.append(copy_output.view(world_size, -1).view(torch.uint8))
            continue
        split_output = copy_output.view(torch.uint8) if byte_views else copy_output
        if split_output.numel() != world_size * split_size:  # a smaller payload
            split_output = split_output[: world_size * split_size]
        split_outputs.append(split_output.view(world_size, split_size))
    torch.ops.fsdp.split_with_sizes_copy(
        all_gather_output.view(world_size, -1), split_sizes, dim=1, out=split_outputs
    )
    for copy_output, output, outer_size in reassembled:
        chunks = copy_output.view(world_size, outer_size, -1).unbind(0)
        torch.cat(chunks, dim=1, out=output.view(outer_size, -1))


def _wait_all_gather(all_gather_result: AllGatherResult) -> None:
    all_gather_work = all_gather_result.all_gather_work
    device = all_gather_result.all_gather_output.device
    device_handle = _get_device_handle(device.type)
    if (all_gather_event := all_gather_result.all_gather_event) is not None:
        device_handle.current_stream().wait_event(all_gather_event)
    if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
        all_gather_work.wait()


@torch.no_grad()
def foreach_all_gather_copy_out(
    all_gather_result: AllGatherResult,
    fsdp_params: list[FSDPParam],
    group: dist.ProcessGroup,
    persistent_buffers: _PersistentBuffers | None = None,
) -> _PersistentBuffers | None:
    """Finalizes the all-gather with the layout that prepared it and returns the
    group's FSDP-owned persistent buffers: those a custom layout returns on the
    group's first unshard, else ``persistent_buffers`` (None if the default
    layout allocated each output)."""
    _wait_all_gather(all_gather_result)
    world_size = group.size()
    layout = all_gather_result.layout
    all_gather_output = all_gather_result.all_gather_output
    if persistent_buffers is not None:
        persistent_buffers.alloc()
    if isinstance(layout, DefaultAllGatherLayout):
        # Rank-major output, from the default layout or a custom layout's fallback
        plan = _prepare_default_all_gather_outputs(
            fsdp_params, all_gather_result, world_size
        )
        default_fn = _default_all_gather_output_fn
        layout._copy_out(all_gather_output, plan, world_size, default_fn)
        return persistent_buffers
    first_unshard = not any(p.all_gather_outputs for p in fsdp_params)
    buffers: list[torch.Tensor] = []
    if not first_unshard:
        for fsdp_param in fsdp_params:
            fsdp_param.alloc_all_gather_outputs()
        if persistent_buffers is not None:
            buffers = list(persistent_buffers.tensors)
        else:
            # Outputs that the default layout created are their own buffers
            buffers = [t for p in fsdp_params for t in p.all_gather_outputs]
    outer_sizes = iter(all_gather_result.all_gather_input_outer_sizes)
    # Copies, so that a layout cannot change FSDP's lists behind the checks below
    param_metadata = [
        AllGatherParamMetadata(
            input_numels=list(numels),
            input_dtypes=list(dtypes),
            outer_sizes=[next(outer_sizes) for _ in numels],
            outputs=list(fsdp_param.all_gather_outputs),
        )
        for fsdp_param, numels, dtypes in zip(
            fsdp_params,
            all_gather_result.param_all_gather_input_numels,
            all_gather_result.param_all_gather_input_dtypes,
        )
    ]
    # Refills write into outputs that saved parameters may alias
    preserved_outputs = tuple(
        output
        for param in param_metadata
        for output in param.outputs
        if not output.is_inference()
    )
    with torch.autograd._unsafe_preserve_version_counter(preserved_outputs):
        outputs = layout.finalize_outputs(
            AllGatherFinalizeMetadata(
                all_gather_output=all_gather_output,
                param_metadata=param_metadata,
                world_size=world_size,
                output_metadata=all_gather_result.output_metadata,
                buffers=buffers,
            )
        )
    if first_unshard:
        return _adopt_layout_outputs(
            fsdp_params, outputs, all_gather_result, world_size
        )
    _check_layout_refill(fsdp_params, outputs)
    return persistent_buffers


# Note [Reduce-scatter input fn]
# copy_in = fn(unsharded_grads, shard_dims=..., world_size=...) runs before FSDP
# allocates the reduce-scatter input and may replace entries of unsharded_grads.
# copy_in(reduce_scatter_input) then fills that flat buffer so that, viewed as
# (world_size, -1), row r holds each gradient's padded shard r in order, cast to the
# buffer's dtype. Both run under no_grad on the current stream, also for a single
# rank; FSDP frees copy_in and the gradients afterward, so keep neither.
def _default_reduce_scatter_input_fn(
    unsharded_grads: list[torch.Tensor],
    *,
    shard_dims: list[int],
    world_size: int,
) -> Callable[[torch.Tensor], None]:
    """Reorder gradients sharded on a nonzero dim by rank and return their copy-in."""
    if world_size > 1:
        for i, shard_dim in enumerate(shard_dims):
            if shard_dim != 0:
                chunks = torch.chunk(unsharded_grads[i], world_size, dim=shard_dim)
                unsharded_grads[i] = torch.cat(chunks, dim=0)
    return lambda out: foreach_reduce_scatter_copy_in(unsharded_grads, out, world_size)


def foreach_reduce_scatter_copy_in(
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    # With reduce_dtype set, gradients share it. Otherwise each follows its
    # parameter's grad_dtype policy (default: the parameter's dtype), so a group
    # can mix dtypes, e.g. bf16 weights with fp32 norms. Uniform groups run the
    # same _chunk_cat kernels as fsdp.chunk_cat, at the same cost. On CUDA, mixed
    # bf16/fp32 and fp16/fp32 groups run that kernel once per input dtype, casting
    # during the copy-in; other mixes cast each gradient first.
    torch.ops.fsdp.chunk_cat_mixed_dtype(
        unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
    )


def _get_all_gather_input_metadatas(
    param_all_gather_inputs: list[list[torch.Tensor]],
) -> tuple[list[list[torch.dtype]], list[list[int]], torch.dtype]:
    param_all_gather_input_dtypes: list[list[torch.dtype]] = []
    param_all_gather_input_numels: list[list[int]] = []
    first_input = next(chain.from_iterable(param_all_gather_inputs), None)
    if first_input is None:
        raise ValueError("cannot all-gather a parameter group without any payloads")
    all_gather_dtype = first_input.dtype
    for all_gather_inputs in param_all_gather_inputs:
        input_dtypes: list[torch.dtype] = []
        input_numels: list[int] = []
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
    reduce_scatter_group: dist.ProcessGroup | None,
    all_reduce_group: dist.ProcessGroup | None,
    reduce_dtype: torch.dtype,
    device_type: str = "",
    factor: float | None = None,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    float | None,
    float | None,
    dist.ReduceOp | dist.ReduceOp.RedOpType,
    dist.ReduceOp | dist.ReduceOp.RedOpType,
]:
    # MTIA appears to only support SUM reduction, hence we force it implicitly
    if device_type == "mtia":
        force_sum_reduction_for_comms = True

    # For fp32/bf16, we do not need to worry about overflow/underflow, so we
    # use NCCL's built-in division to avoid separate div kernels
    overflow_risk = reduce_dtype not in (torch.float32, torch.bfloat16)
    if reduce_scatter_group is not None:
        data_parallel_size = reduce_scatter_group.size()
    else:
        data_parallel_size = 1

    if all_reduce_group is not None:
        data_parallel_size *= all_reduce_group.size()

    if not overflow_risk and not force_sum_reduction_for_comms:
        if factor is None:
            # Warning: NCCL ReduceOp.AVG may produce incorrect results with
            # world size 1.
            if data_parallel_size == 1:
                return None, None, ReduceOp.SUM, ReduceOp.SUM
            return None, None, ReduceOp.AVG, ReduceOp.AVG
        if reduce_scatter_group is not None and factor == reduce_scatter_group.size():
            reduce_scatter_op = ReduceOp.AVG
        else:
            reduce_scatter_op = ReduceOp.PREMUL_SUM(1 / factor)
        return None, None, reduce_scatter_op, ReduceOp.SUM

    if factor is None:
        factor = float(data_parallel_size)
    pre_factor: float | None
    if overflow_risk:
        # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
        # overflow/underflow. For N data parallel workers, each worker computes
        # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
        # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
        pre_factor = 1
        while factor % pre_factor == 0 and factor / pre_factor > pre_factor:
            pre_factor *= 2
        post_factor = factor / pre_factor
    else:
        # Prefer post-multiplying as it operates on less data and is thus faster
        pre_factor, post_factor = None, factor

    return pre_factor, post_factor, ReduceOp.SUM, ReduceOp.SUM


def _div_if_needed(tensor: torch.Tensor, div_factor: float | None) -> None:
    if div_factor is not None and div_factor != 1:
        tensor.div_(div_factor)
