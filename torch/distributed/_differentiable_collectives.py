# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Differentiable Collectives with Explicit Gradient Semantics.

This module provides distributed collectives with well-defined gradient behavior,
enabling automatic differentiation through distributed operations. Unlike
_functional_collectives which focus on async execution and compiler optimization,
these collectives explicitly define forward and backward semantics.

Key Features:
- Explicit gradient semantics for each collective
- Python-only implementation (no C++ dependency initially)
- Explicit wait_tensor calls in both forward and backward (synchronous behavior)
- Flexible group specification (str, DeviceMesh, tuple)

Collective Semantics:
- all_reduce_sum: Forward: all_reduce(sum), Backward: all_reduce(sum)
- all_reduce_sum_invariant: Forward: all_reduce(sum), Backward: identity (no-op)
- mark_varying: Forward: identity, Backward: all_reduce(sum)
- all_gather: Forward: all_gather, Backward: reduce_scatter
- reduce_scatter: Forward: reduce_scatter, Backward: all_gather
- all_to_all: Forward: all_to_all, Backward: all_to_all (reversed splits)

Example:
    >>> import torch.distributed as dist
    >>> from torch.distributed._c10d_differentiable import all_gather
    >>>
    >>> # Initialize distributed
    >>> dist.init_process_group(...)
    >>>
    >>> # All-gather with automatic gradient handling
    >>> input = torch.randn(4, 4, requires_grad=True)
    >>> output = all_gather(input, gather_dim=0, group=dist.group.WORLD)
    >>> loss = output.sum()
    >>> loss.backward()  # Gradients automatically reduce-scattered
"""

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import DeviceMesh


def _resolve_group_name(group: str, mesh: Optional[DeviceMesh] = None) -> str:
    """
    Resolve group name from group identifier and optional mesh.

    Args:
        group: Group name (if mesh is None) or dimension name (if mesh is provided)
        mesh: Optional DeviceMesh for resolving group from dimension name

    Returns:
        Group name as string

    Raises:
        ValueError: If group specification is invalid

    Examples:
        >>> # Direct group name (no mesh)
        >>> group_name = _resolve_group_name("tp_group")
        >>>
        >>> # With mesh and dimension name
        >>> mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=["tp", "dp"])
        >>> group_name = _resolve_group_name("tp", mesh=mesh)
    """
    if mesh is None:
        # group is the actual group name
        return group

    # mesh is provided, group is a dimension name
    try:
        pg = mesh.get_group(group)
        return pg.group_name
    except (KeyError, IndexError, RuntimeError) as e:
        raise ValueError(
            f"Failed to resolve group from dimension '{group}' in mesh: {e}"
        ) from e


class _AllGather(torch.autograd.Function):
    """
    All-gather collective with reduce-scatter gradient.

    Forward: Gather tensors from all ranks along gather_dim
    Backward: Reduce-scatter gradients with sum
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        input: torch.Tensor,
        gather_dim: int,
        group_name: str,
    ) -> torch.Tensor:
        ctx.group_name = group_name
        ctx.gather_dim = gather_dim

        group_size = c10d._get_group_size_by_name(group_name)
        tensor = torch.ops._c10d_functional.all_gather_into_tensor(
            input, group_size, group_name
        )
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        # Handle non-zero gather_dim
        if gather_dim != 0:
            res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)

        return res

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        # Backward is reduce_scatter
        group_size = c10d._get_group_size_by_name(ctx.group_name)

        # Handle non-zero gather_dim
        if ctx.gather_dim != 0:
            tensor_list = torch.chunk(grad_output, group_size, dim=ctx.gather_dim)
            grad_output = torch.cat(tensor_list, dim=0)

        tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
            grad_output.contiguous(),
            "sum",
            group_size,
            ctx.group_name,
        )
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        return res, None, None


class _ReduceScatter(torch.autograd.Function):
    """
    Reduce-scatter collective with all-gather gradient.

    Forward: Reduce (sum) and scatter chunks to ranks
    Backward: All-gather gradients
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        input: torch.Tensor,
        scatter_dim: int,
        group_name: str,
        op: str,
    ) -> torch.Tensor:
        ctx.group_name = group_name
        ctx.scatter_dim = scatter_dim

        group_size = c10d._get_group_size_by_name(group_name)

        # Verify input size
        if input.size(scatter_dim) % group_size != 0:
            raise ValueError(
                f"Input dimension {scatter_dim} ({input.size(scatter_dim)}) "
                f"must be divisible by group_size {group_size}"
            )

        # Handle non-zero scatter_dim
        if scatter_dim != 0:
            tensor_list = torch.chunk(input, group_size, dim=scatter_dim)
            input = torch.cat(tensor_list)

        tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
            input,
            op.lower(),
            group_size,
            group_name,
        )
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        return res

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        # Backward is all_gather
        group_size = c10d._get_group_size_by_name(ctx.group_name)
        tensor = torch.ops._c10d_functional.all_gather_into_tensor(
            grad_output.contiguous(), group_size, ctx.group_name
        )
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        # Handle non-zero scatter_dim
        if ctx.scatter_dim != 0:
            res = torch.cat(torch.chunk(res, group_size, dim=0), dim=ctx.scatter_dim)

        return res, None, None, None


class _AllReduceSum(torch.autograd.Function):
    """
    All-reduce (sum) collective with all-reduce gradient.

    Forward: All-reduce with sum
    Backward: All-reduce gradients with sum
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input: torch.Tensor, group_name: str
    ) -> torch.Tensor:
        ctx.group_name = group_name

        tensor = torch.ops._c10d_functional.all_reduce(input, "sum", group_name)
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        return res

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        # Backward is also all_reduce (sum)
        tensor = torch.ops._c10d_functional.all_reduce(
            grad_output.contiguous(), "sum", ctx.group_name
        )
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        return res, None


class _AllReduceSumInvariant(torch.autograd.Function):
    """
    All-reduce (sum) for invariant tensors (no gradient aggregation).

    Forward: All-reduce with sum
    Backward: Identity (no-op) - assumes tensor is already identical across ranks

    Use this for tensors that are known to be identical across ranks (e.g., loss values),
    where you want reduction for stability but don't need gradient aggregation.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input: torch.Tensor, group_name: str
    ) -> torch.Tensor:
        # Forward: all_reduce
        tensor = torch.ops._c10d_functional.all_reduce(input, "sum", group_name)
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        return res

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        # Backward: identity (no gradient aggregation)
        return grad_output, None


class _MarkVarying(torch.autograd.Function):
    """
    Mark tensor as varying across ranks (identity forward, all-reduce backward).

    Forward: Identity (no-op)
    Backward: All-reduce gradients with sum

    Use this to mark tensors that differ across ranks and need gradient aggregation
    in the backward pass.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input: torch.Tensor, group_name: str
    ) -> torch.Tensor:
        ctx.group_name = group_name
        # Forward: identity
        return input

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        # Backward: all_reduce
        tensor = torch.ops._c10d_functional.all_reduce(
            grad_output.contiguous(), "sum", ctx.group_name
        )
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        return res, None


class _AllToAll(torch.autograd.Function):
    """
    All-to-all collective with reversed split sizes in gradient.

    Forward: All-to-all with specified split sizes
    Backward: All-to-all with reversed split sizes
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        input: torch.Tensor,
        output_split_sizes: Optional[list[int]],
        input_split_sizes: Optional[list[int]],
        group_name: str,
    ) -> torch.Tensor:
        group_size = c10d._get_group_size_by_name(group_name)

        if (output_split_sizes is None) != (input_split_sizes is None):
            raise ValueError(
                "output_split_sizes and input_split_sizes must both be None or both be non-None"
            )

        if (output_split_sizes is None) or (input_split_sizes is None):
            if input.size(0) % group_size != 0:
                raise ValueError(
                    f"Input dimension 0 ({input.size(0)}) "
                    f"must be divisible by group_size {group_size}"
                )
            output_split_sizes = [input.size(0) // group_size] * group_size
            input_split_sizes = output_split_sizes

        ctx.group_name = group_name
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        tensor = torch.ops._c10d_functional.all_to_all_single(
            input, output_split_sizes, input_split_sizes, group_name
        )
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        return res

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        # Backward: all_to_all with reversed split sizes
        tensor = torch.ops._c10d_functional.all_to_all_single(
            grad_output.contiguous(),
            ctx.input_split_sizes,  # Reversed
            ctx.output_split_sizes,  # Reversed
            ctx.group_name,
        )
        res = torch.ops._c10d_functional.wait_tensor(tensor)

        return res, None, None, None


# Public API functions


def all_reduce_sum(
    input: torch.Tensor, *, group: str, mesh: Optional[DeviceMesh] = None
) -> torch.Tensor:
    """
    All-reduce (sum) tensor across all ranks.

    Forward: all_reduce with sum
    Backward: all_reduce gradients with sum

    Args:
        input: Input tensor
        group: Process group name or dimension identifier (if mesh is provided)
        mesh: Optional DeviceMesh for resolving group from dimension

    Returns:
        Reduced tensor

    Example:
        >>> # Sum values across all ranks
        >>> output = all_reduce_sum(input, group="dp_group")
        >>> # Or with mesh
        >>> output = all_reduce_sum(input, group="dp", mesh=mesh_2d)
    """
    group_name = _resolve_group_name(group, mesh)
    return _AllReduceSum.apply(input, group_name)


def all_reduce_sum_invariant(
    input: torch.Tensor, *, group: str, mesh: Optional[DeviceMesh] = None
) -> torch.Tensor:
    """
    All-reduce (sum) tensor that is invariant across ranks.

    Forward: all_reduce with sum
    Backward: identity (no gradient aggregation)

    Use this when the tensor is already identical across ranks (e.g., loss values)
    but you want to ensure numerical stability via reduction.

    Args:
        input: Input tensor (assumed identical across ranks)
        group: Process group name or dimension identifier (if mesh is provided)
        mesh: Optional DeviceMesh for resolving group from dimension

    Returns:
        Reduced tensor

    Example:
        >>> # Loss is already same on all ranks, but reduce for stability
        >>> loss = all_reduce_sum_invariant(local_loss, group="dp_group")
        >>> # Or with mesh
        >>> loss = all_reduce_sum_invariant(local_loss, group="dp", mesh=mesh_2d)
    """
    group_name = _resolve_group_name(group, mesh)
    return _AllReduceSumInvariant.apply(input, group_name)


def mark_varying(
    input: torch.Tensor, *, group: str, mesh: Optional[DeviceMesh] = None
) -> torch.Tensor:
    """
    Mark a tensor as varying across ranks (identity in forward).

    Forward: identity (no-op)
    Backward: all_reduce gradients with sum

    Use this to mark tensors that differ across ranks and need gradient aggregation.

    Args:
        input: Input tensor (varying across ranks)
        group: Process group name or dimension identifier (if mesh is provided)
        mesh: Optional DeviceMesh for resolving group from dimension

    Returns:
        Input tensor (unchanged)

    Example:
        >>> # Per-rank data needs gradient aggregation
        >>> output = mark_varying(per_rank_data, group="dp_group")
        >>> # Or with mesh
        >>> output = mark_varying(per_rank_data, group="dp", mesh=mesh_2d)
    """
    group_name = _resolve_group_name(group, mesh)
    return _MarkVarying.apply(input, group_name)


def all_gather(
    input: torch.Tensor,
    *,
    gather_dim: int,
    group: str,
    mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    Gather tensors from all ranks and concatenate along gather_dim.

    Forward: all_gather into tensor
    Backward: reduce_scatter gradients with sum

    Args:
        input: Input tensor
        gather_dim: Dimension along which to concatenate gathered tensors
        group: Process group name or dimension identifier (if mesh is provided)
        mesh: Optional DeviceMesh for resolving group from dimension

    Returns:
        Gathered tensor with size[gather_dim] = input.size[gather_dim] * world_size

    Example:
        >>> # Gather along batch dimension (dim 0)
        >>> # Input shape: [B, H], Output shape: [B*world_size, H]
        >>> output = all_gather(input, gather_dim=0, group="tp_group")
        >>> # Or with mesh
        >>> output = all_gather(input, gather_dim=0, group="tp", mesh=mesh_2d)
    """
    group_name = _resolve_group_name(group, mesh)
    return _AllGather.apply(input, gather_dim, group_name)


def reduce_scatter(
    input: torch.Tensor,
    *,
    scatter_dim: int,
    group: str,
    mesh: Optional[DeviceMesh] = None,
    op: str = "sum",
) -> torch.Tensor:
    """
    Reduce-scatter tensor across all ranks.

    Forward: reduce (with op) and scatter chunks to ranks
    Backward: all_gather gradients

    Args:
        input: Input tensor (size[scatter_dim] must be divisible by world_size)
        scatter_dim: Dimension along which to scatter
        group: Process group name or dimension identifier (if mesh is provided)
        mesh: Optional DeviceMesh for resolving group from dimension
        op: Reduction operation (currently only "sum" supported)

    Returns:
        Scattered tensor with size[scatter_dim] = input.size[scatter_dim] / world_size

    Example:
        >>> # Reduce-scatter along batch dimension (dim 0)
        >>> # Input shape: [B*world_size, H], Output shape: [B, H]
        >>> output = reduce_scatter(input, scatter_dim=0, group="tp_group")
        >>> # Or with mesh
        >>> output = reduce_scatter(input, scatter_dim=0, group="tp", mesh=mesh_2d)
    """
    group_name = _resolve_group_name(group, mesh)
    return _ReduceScatter.apply(input, scatter_dim, group_name, op)


def all_to_all(
    input: torch.Tensor,
    *,
    output_split_sizes: Optional[list[int]],
    input_split_sizes: Optional[list[int]],
    group: str,
    mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    All-to-all exchange of tensor chunks between ranks.

    Forward: all_to_all with specified split sizes
    Backward: all_to_all with reversed split sizes

    Args:
        input: Input tensor
        output_split_sizes: Sizes of output chunks (None for uniform split)
        input_split_sizes: Sizes of input chunks (None for uniform split)
        group: Process group name or dimension identifier (if mesh is provided)
        mesh: Optional DeviceMesh for resolving group from dimension

    Returns:
        Tensor after all-to-all exchange

    Example:
        >>> # Uniform split
        >>> output = all_to_all(
        ...     input, output_split_sizes=None, input_split_sizes=None, group="ep_group"
        ... )
        >>> # Or with mesh
        >>> output = all_to_all(
        ...     input,
        ...     output_split_sizes=None,
        ...     input_split_sizes=None,
        ...     group="ep",
        ...     mesh=mesh_2d,
        ... )
    """
    group_name = _resolve_group_name(group, mesh)
    return _AllToAll.apply(input, output_split_sizes, input_split_sizes, group_name)
