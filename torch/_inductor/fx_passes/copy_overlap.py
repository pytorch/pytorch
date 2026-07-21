# mypy: allow-untyped-defs
import itertools
import math

import torch
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode


_COPY_OVERLAP_EXACT_MAX_ELEMENTS = 65536


def static_int(x: object) -> int | None:
    try:
        return int(x)  # type: ignore[arg-type]
    except (GuardOnDataDependentSymNode, TypeError, ValueError):
        return None


def same_storage_byte_offset(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    lhs_offset = static_int(lhs.storage_offset())
    rhs_offset = static_int(rhs.storage_offset())
    return (
        lhs_offset is not None
        and rhs_offset is not None
        and lhs_offset * lhs.element_size() == rhs_offset * rhs.element_size()
    )


def copy_same_mapping(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return (
        lhs.dtype == rhs.dtype
        and lhs.size() == rhs.size()
        and lhs.stride() == rhs.stride()
        and same_storage_byte_offset(lhs, rhs)
    )


def copy_allows_same_start_expanded_src(
    dst: torch.Tensor, src: torch.Tensor
) -> bool:
    if src.dim() > dst.dim() or dst.dtype != src.dtype:
        return False
    if not same_storage_byte_offset(dst, src):
        return False

    has_expanded_src_dim = False
    dst_sizes = list(dst.size())
    dst_strides = list(dst.stride())
    src_sizes = list(src.size())
    src_strides = list(src.stride())
    for offset in range(dst.dim()):
        dst_dim = dst.dim() - 1 - offset
        dst_size = static_int(dst_sizes[dst_dim])
        if dst_size is None:
            return False
        if offset >= src.dim():
            if dst_size > 1:
                has_expanded_src_dim = True
            continue

        src_dim = src.dim() - 1 - offset
        src_size = static_int(src_sizes[src_dim])
        src_stride = static_int(src_strides[src_dim])
        dst_stride = static_int(dst_strides[dst_dim])
        if src_size is None or src_stride is None or dst_stride is None:
            return False

        if dst_size == src_size:
            if dst_size <= 1:
                continue
            if src_stride == 0:
                has_expanded_src_dim = True
                continue
            if dst_stride != src_stride:
                return False
            continue

        if src_size == 1:
            if dst_size > 1:
                has_expanded_src_dim = True
            continue

        return False

    return has_expanded_src_dim or dst.dim() != src.dim()


def tensor_byte_range(val: torch.Tensor) -> tuple[int, int] | None:
    storage_offset = static_int(val.storage_offset())
    sizes = [static_int(size) for size in val.size()]
    strides = [static_int(stride) for stride in val.stride()]
    if (
        storage_offset is None
        or any(size is None for size in sizes)
        or any(stride is None for stride in strides)
    ):
        return None
    if val.numel() == 0:
        return 0, 0

    itemsize = val.element_size()
    min_offset = storage_offset
    max_offset = storage_offset
    for size, stride in zip(sizes, strides):
        if size is None or stride is None or size <= 1:
            continue
        delta = (size - 1) * stride
        min_offset += min(delta, 0)
        max_offset += max(delta, 0)
    return min_offset * itemsize, max_offset * itemsize + itemsize


def tensor_residual_byte_range(
    val: torch.Tensor, excluded_dim: int
) -> tuple[int, int] | None:
    storage_offset = static_int(val.storage_offset())
    sizes = [static_int(size) for size in val.size()]
    strides = [static_int(stride) for stride in val.stride()]
    if (
        storage_offset is None
        or any(size is None for size in sizes)
        or any(stride is None for stride in strides)
    ):
        return None

    itemsize = val.element_size()
    start = storage_offset * itemsize
    min_offset = start
    max_offset = start
    for dim, (size, stride) in enumerate(zip(sizes, strides)):
        if dim == excluded_dim or size is None or stride is None or size <= 1:
            continue
        delta = (size - 1) * stride * itemsize
        min_offset += min(delta, 0)
        max_offset += max(delta, 0)
    return min_offset, max_offset + itemsize


def tensors_have_disjoint_byte_bands(
    lhs: torch.Tensor, rhs: torch.Tensor
) -> bool:
    lhs_sizes = [static_int(size) for size in lhs.size()]
    rhs_sizes = [static_int(size) for size in rhs.size()]
    lhs_strides = [static_int(stride) for stride in lhs.stride()]
    rhs_strides = [static_int(stride) for stride in rhs.stride()]
    if (
        any(size is None for size in lhs_sizes)
        or any(size is None for size in rhs_sizes)
        or any(stride is None for stride in lhs_strides)
        or any(stride is None for stride in rhs_strides)
    ):
        return False

    lhs_itemsize = lhs.element_size()
    rhs_itemsize = rhs.element_size()
    for lhs_dim, (lhs_size, lhs_stride) in enumerate(zip(lhs_sizes, lhs_strides)):
        if lhs_size is None or lhs_stride is None or lhs_size <= 1:
            continue
        lhs_byte_stride = lhs_stride * lhs_itemsize
        if lhs_byte_stride <= 0:
            continue
        for rhs_dim, (rhs_size, rhs_stride) in enumerate(zip(rhs_sizes, rhs_strides)):
            if rhs_size is None or rhs_stride is None or rhs_size <= 1:
                continue
            rhs_byte_stride = rhs_stride * rhs_itemsize
            if rhs_byte_stride != lhs_byte_stride:
                continue
            lhs_range = tensor_residual_byte_range(lhs, lhs_dim)
            rhs_range = tensor_residual_byte_range(rhs, rhs_dim)
            if lhs_range is None or rhs_range is None:
                continue
            if lhs_range[1] <= rhs_range[0] or rhs_range[1] <= lhs_range[0]:
                union_start = min(lhs_range[0], rhs_range[0])
                union_end = max(lhs_range[1], rhs_range[1])
                if union_end - union_start <= lhs_byte_stride:
                    return True
    return False


def byte_stride_gcd(val: torch.Tensor) -> int | None:
    sizes = [static_int(size) for size in val.size()]
    strides = [static_int(stride) for stride in val.stride()]
    if any(size is None for size in sizes) or any(stride is None for stride in strides):
        return None

    result = 0
    for size, stride in zip(sizes, strides):
        if size is None or stride is None or size <= 1 or stride == 0:
            continue
        result = math.gcd(result, abs(stride * val.element_size()))
    return result


def tensors_have_disjoint_byte_residue(
    lhs: torch.Tensor, rhs: torch.Tensor
) -> bool:
    lhs_offset = static_int(lhs.storage_offset())
    rhs_offset = static_int(rhs.storage_offset())
    lhs_gcd = byte_stride_gcd(lhs)
    rhs_gcd = byte_stride_gcd(rhs)
    if lhs_offset is None or rhs_offset is None or lhs_gcd is None or rhs_gcd is None:
        return False
    combined_gcd = math.gcd(lhs_gcd, rhs_gcd)
    if combined_gcd == 0:
        return False

    lhs_start = lhs_offset * lhs.element_size()
    rhs_start = rhs_offset * rhs.element_size()
    residue = (lhs_start - rhs_start) % combined_gcd
    distance = min(residue, combined_gcd - residue)
    return distance >= max(lhs.element_size(), rhs.element_size())


def byte_intervals_for_tensor(
    val: torch.Tensor,
) -> list[tuple[int, int]] | None:
    sizes = [static_int(size) for size in val.size()]
    strides = [static_int(stride) for stride in val.stride()]
    storage_offset = static_int(val.storage_offset())
    if (
        storage_offset is None
        or any(size is None for size in sizes)
        or any(stride is None for stride in strides)
    ):
        return None

    concrete_sizes = [size for size in sizes if size is not None]
    concrete_strides = [stride for stride in strides if stride is not None]
    numel = 1
    for size in concrete_sizes:
        numel *= size
        if numel > _COPY_OVERLAP_EXACT_MAX_ELEMENTS:
            return None

    itemsize = val.element_size()
    intervals = []
    for point in itertools.product(*(range(size) for size in concrete_sizes)):
        storage_index = storage_offset + sum(
            index * stride for index, stride in zip(point, concrete_strides)
        )
        start = storage_index * itemsize
        intervals.append((start, start + itemsize))
    return intervals


def byte_interval_sets_are_disjoint(
    lhs: list[tuple[int, int]], rhs: list[tuple[int, int]]
) -> bool:
    lhs = sorted(lhs)
    rhs = sorted(rhs)
    lhs_idx = rhs_idx = 0
    while lhs_idx < len(lhs) and rhs_idx < len(rhs):
        lhs_start, lhs_end = lhs[lhs_idx]
        rhs_start, rhs_end = rhs[rhs_idx]
        if lhs_end <= rhs_start:
            lhs_idx += 1
        elif rhs_end <= lhs_start:
            rhs_idx += 1
        else:
            return False
    return True


def tensors_have_exact_disjoint_byte_intervals(
    lhs: torch.Tensor, rhs: torch.Tensor
) -> bool:
    lhs_intervals = byte_intervals_for_tensor(lhs)
    rhs_intervals = byte_intervals_for_tensor(rhs)
    if lhs_intervals is None or rhs_intervals is None:
        return False
    return byte_interval_sets_are_disjoint(lhs_intervals, rhs_intervals)


def aliasing_mutation_copy_may_overlap(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if copy_same_mapping(dst, src) or copy_allows_same_start_expanded_src(dst, src):
        return False

    dst_range = tensor_byte_range(dst)
    src_range = tensor_byte_range(src)
    if (
        dst_range is not None
        and src_range is not None
        and (dst_range[1] <= src_range[0] or src_range[1] <= dst_range[0])
    ):
        return False
    if tensors_have_disjoint_byte_bands(dst, src):
        return False
    if tensors_have_disjoint_byte_residue(dst, src):
        return False
    if tensors_have_exact_disjoint_byte_intervals(dst, src):
        return False
    return True
