import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter

from ._fsdp_api import _ReduceOp


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
