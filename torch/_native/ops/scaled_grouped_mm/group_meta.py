"""Output and group metadata helpers for DeepSeek grouped mm."""

import functools

import torch

from ._common import BLOCKWISE_1X128
from .hopper_config import HopperDeepSeekConfig, select_kernel_config


_launch_build_group_metadata = None


def _get_launch_build_group_metadata():
    global _launch_build_group_metadata
    if _launch_build_group_metadata is None:
        from .group_metadata_kernel import launch_build_group_metadata

        _launch_build_group_metadata = launch_build_group_metadata
    return _launch_build_group_metadata


def expected_out_size_stride(
    mat_a: torch.Tensor, mat_b: torch.Tensor, out_dtype: torch.dtype
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    total_m = mat_a.size(0)
    n = mat_b.size(-1)
    elem_size = torch.empty((), dtype=out_dtype).element_size()
    alignment = max(16 // elem_size, 1)
    padded_n = -(-n // alignment) * alignment
    return (total_m, n), (padded_n, 1)


def allocate_output(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    size, stride = expected_out_size_stride(mat_a, mat_b, out_dtype)
    return torch.empty_strided(size, stride, dtype=out_dtype, device=mat_a.device)


@functools.lru_cache(maxsize=32)
def _alloc_group_metadata(
    device_index: int, cap: int, stream_ptr: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda", device_index)
    problem_sizes = torch.empty((cap, 4), device=device, dtype=torch.int32)
    ptrs_abc = torch.empty((cap, 3), device=device, dtype=torch.int64)
    ptrs_scale = torch.empty((cap, 2), device=device, dtype=torch.int64)
    tile_offsets = torch.empty((cap + 1,), device=device, dtype=torch.int32)
    total_tiles = torch.empty((1,), device=device, dtype=torch.int32)
    return problem_sizes, ptrs_abc, ptrs_scale, tile_offsets, total_tiles


@functools.lru_cache(maxsize=32)
def _get_group_metadata_tensors(
    group_count: int, device_index: int, stream_ptr: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cap = max(64, 1 << (group_count - 1).bit_length())
    tensors = _alloc_group_metadata(device_index, cap, stream_ptr)
    problem_sizes, ptrs_abc, ptrs_scale, tile_offsets, total_tiles = tensors
    return (
        problem_sizes[:group_count],
        ptrs_abc[:group_count],
        ptrs_scale[:group_count],
        tile_offsets[: group_count + 1],
        total_tiles,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group_count = mat_b.size(0)
    device_index = mat_a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_ptr = torch.cuda.current_stream(device_index).cuda_stream
    metadata = _get_group_metadata_tensors(group_count, device_index, stream_ptr)
    problem_sizes, ptrs_abc, ptrs_scale, tile_offsets, total_tiles = metadata
    if group_count == 0:
        tile_offsets.zero_()
        total_tiles.zero_()
        return problem_sizes, ptrs_abc, ptrs_scale, tile_offsets, total_tiles
    if config is None:
        config = select_kernel_config(
            total_m=mat_a.size(0),
            n=mat_b.size(-1),
            k=mat_a.size(-1),
            group_count=group_count,
        )
    if config.cluster_n > 1:
        tiles_n = -(-mat_b.size(-1) // config.tile_n)
        if tiles_n % config.cluster_n != 0:
            raise ValueError(
                f"cluster_n={config.cluster_n} requires N tiles ({tiles_n}, "
                f"from N={mat_b.size(-1)}/tile_n={config.tile_n}) to be "
                f"divisible by cluster_n"
            )

    scale_a_rows_per_block = 1 if recipe_a == BLOCKWISE_1X128 else 128
    scale_a_group_stride = (
        scale_a.stride(0) if recipe_a == BLOCKWISE_1X128 else scale_a.stride(1)
    )

    _get_launch_build_group_metadata()(
        offs,
        mat_a.data_ptr(),
        mat_b.data_ptr(),
        out.data_ptr(),
        scale_a.data_ptr(),
        scale_b.data_ptr(),
        mat_a.stride(0),
        mat_b.stride(0),
        out.stride(0),
        scale_a_group_stride,
        scale_b.stride(0),
        scale_a_rows_per_block,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        mat_a.size(0),
        mat_b.size(-1),
        mat_a.size(-1),
        problem_sizes,
        ptrs_abc,
        ptrs_scale,
        tile_offsets,
        total_tiles,
        mat_a.element_size(),
        scale_a.element_size(),
        out.element_size(),
    )
    return problem_sizes, ptrs_abc, ptrs_scale, tile_offsets, total_tiles
