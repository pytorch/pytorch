import math
from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter

from ._fsdp_api import _ReduceOp


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
