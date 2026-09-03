"""Output and group metadata helpers for DeepSeek grouped mm."""

import functools
from typing import NamedTuple

import torch

from ._common import BLOCKWISE_1X128
from .hopper_config import HopperDeepSeekConfig, select_kernel_config


class GroupMetadata(NamedTuple):
    problem_sizes: torch.Tensor
    ptrs_abc: torch.Tensor
    ptrs_scale: torch.Tensor
    tile_offsets: torch.Tensor
    total_tiles: torch.Tensor


def expected_out_size_stride(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out_dtype: torch.dtype,
    group_count: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = mat_b.size(-1)
    elem_size = torch.empty((), dtype=out_dtype).element_size()
    alignment = max(16 // elem_size, 1)
    padded_n = -(-n // alignment) * alignment
    if mat_a.dim() == 3:
        if mat_b.dim() == 2:
            # offs splits N: one 2-D output, groups own column ranges.
            return (mat_a.size(1), n), (padded_n, 1)
        batch, m = mat_a.size(0), mat_a.size(1)
        return (batch, m, n), (m * padded_n, padded_n, 1)
    m = mat_a.size(0)
    if mat_b.dim() == 2:
        if group_count is None:
            raise AssertionError("group_count is required for a 2d mat_b")
        return (group_count, m, n), (m * padded_n, padded_n, 1)
    return (m, n), (padded_n, 1)


def allocate_output(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out_dtype: torch.dtype,
    group_count: int | None = None,
) -> torch.Tensor:
    size, stride = expected_out_size_stride(mat_a, mat_b, out_dtype, group_count)
    return torch.empty_strided(size, stride, dtype=out_dtype, device=mat_a.device)


@functools.lru_cache(maxsize=32)
def _alloc_group_metadata(
    device_index: int, cap: int, stream_ptr: int
) -> GroupMetadata:
    device = torch.device("cuda", device_index)
    problem_sizes = torch.empty((cap, 4), device=device, dtype=torch.int32)
    ptrs_abc = torch.empty((cap, 3), device=device, dtype=torch.int64)
    ptrs_scale = torch.empty((cap, 2), device=device, dtype=torch.int64)
    tile_offsets = torch.empty((cap + 1,), device=device, dtype=torch.int32)
    total_tiles = torch.empty((1,), device=device, dtype=torch.int32)
    return GroupMetadata(problem_sizes, ptrs_abc, ptrs_scale, tile_offsets, total_tiles)


@functools.lru_cache(maxsize=32)
def _get_group_metadata_tensors(
    group_count: int, device_index: int, stream_ptr: int
) -> GroupMetadata:
    cap = max(64, 1 << (group_count - 1).bit_length())
    m = _alloc_group_metadata(device_index, cap, stream_ptr)
    return GroupMetadata(
        m.problem_sizes[:group_count],
        m.ptrs_abc[:group_count],
        m.ptrs_scale[:group_count],
        m.tile_offsets[: group_count + 1],
        m.total_tiles,
    )


def build_group_metadata(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    recipe_a: int,
    offs: torch.Tensor,
    out: torch.Tensor,
    config: HopperDeepSeekConfig | None = None,
    offs_align: int = 1,
) -> GroupMetadata:
    # mat_b.size(0) is K when offs splits K; offs is absent when batched.
    b_is_2d = mat_b.dim() == 2
    jagged_n = mat_a.dim() == 3 and b_is_2d
    batched = mat_a.dim() == 3 and not b_is_2d
    group_count = mat_a.size(0) if batched else offs.numel()
    # M is fixed when offs splits N; mat_a.size(0) is the group count there.
    total_m = mat_a.size(-2) if jagged_n else mat_a.size(0)
    device_index = mat_a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_ptr = torch.cuda.current_stream(device_index).cuda_stream
    meta = _get_group_metadata_tensors(group_count, device_index, stream_ptr)
    if batched:
        # Uniform per-batch problem: the kernel derives tile counts itself.
        return meta
    if group_count == 0:
        meta.tile_offsets.zero_()
        meta.total_tiles.zero_()
        return meta
    if config is None:
        config = select_kernel_config(
            total_m=total_m,
            n=mat_b.size(-1),
            k=mat_a.size(-1),
            group_count=group_count,
            groups_split_k=b_is_2d and not jagged_n,
            jagged_n=jagged_n,
        )
    if config.cluster_n > 1:
        tiles_n = -(-mat_b.size(-1) // config.tile_n)
        if tiles_n % config.cluster_n != 0:
            raise ValueError(
                f"cluster_n={config.cluster_n} requires N tiles ({tiles_n}, "
                f"from N={mat_b.size(-1)}/tile_n={config.tile_n}) to be "
                f"divisible by cluster_n"
            )

    groups_split_k = b_is_2d and not jagged_n
    if jagged_n:
        scale_a_rows_per_block = 1
        scale_a_group_stride = scale_a.stride(0)
        scale_b_group_stride = scale_b.stride(0)
    elif groups_split_k:
        scale_a_rows_per_block = 128
        scale_a_group_stride = scale_a.stride(1)
        scale_b_group_stride = scale_b.stride(0)
    else:
        scale_a_rows_per_block = 1 if recipe_a == BLOCKWISE_1X128 else 128
        scale_a_group_stride = (
            scale_a.stride(0) if recipe_a == BLOCKWISE_1X128 else scale_a.stride(1)
        )
        scale_b_group_stride = scale_b.stride(0)

    # Local import: keeps cutlass off the `import torch` path.
    from .group_metadata_kernel import launch_build_group_metadata

    launch_build_group_metadata(
        offs,
        mat_a.data_ptr(),
        mat_b.data_ptr(),
        out.data_ptr(),
        scale_a.data_ptr(),
        scale_b.data_ptr(),
        mat_a.stride(0),
        mat_b.stride(1) if jagged_n else mat_b.stride(0),
        out.stride(1) if jagged_n else out.stride(0),
        scale_a_group_stride,
        scale_b_group_stride,
        scale_a_rows_per_block,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        total_m,
        mat_b.size(-1),
        mat_a.size(-1),
        meta.problem_sizes,
        meta.ptrs_abc,
        meta.ptrs_scale,
        meta.tile_offsets,
        meta.total_tiles,
        mat_a.element_size(),
        scale_a.element_size(),
        out.element_size(),
        groups_split_k,
        jagged_n,
        offs_align,
    )
    return meta
