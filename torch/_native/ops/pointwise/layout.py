r"""Host layout classification for pointwise kernels."""

from __future__ import annotations

import math
from functools import cache, lru_cache
from typing import NamedTuple, TYPE_CHECKING

import torch
from torch._tensor_iterator import TensorIterator

from ...cutedsl import config


if TYPE_CHECKING:
    from ._types import RowGap, Shape

TMA_TILE = config.TMA_TILE
TMA_STAGES = config.TMA_STAGES
TMA_THREADS = config.TMA_THREADS
TMA_SMEM_ALIGNMENT = config.TMA_SMEM_ALIGNMENT


@lru_cache(maxsize=256)
def output_strides(
    operands: tuple[tuple[torch.dtype, Shape, Shape], ...], dtype: torch.dtype
) -> Shape:
    inputs = [
        torch.empty_strided(shape, strides, dtype=d, device="meta")
        for d, shape, strides in operands
    ]
    iterator = TensorIterator(
        outputs=[None],
        const_inputs=inputs,
        check_all_same_dtype=False,
        static_dtype=dtype,
        static_device=torch.device("meta"),
    )
    return iterator.output().stride()


class TmaLayout(NamedTuple):
    r"""Descriptor/runtime shape-stride pairs; ``None`` marks scalar inputs."""

    transposed: tuple[bool | None, ...]
    inputs: tuple[tuple[Shape, Shape], ...]
    output: tuple[Shape, Shape]
    rows_first: bool


class IterationLayout(NamedTuple):
    shape: Shape
    strides: tuple[Shape, ...]


def iteration_layout(
    inputs: list[torch.Tensor], outputs: list[torch.Tensor], *, with_index: bool = False
) -> IterationLayout:
    iterator = TensorIterator(
        outputs=[*outputs],
        const_inputs=inputs,
        resize_outputs=False,
        check_all_same_dtype=False,
        check_all_same_device=False,
        check_mem_overlap=False,
        allow_cpu_scalars=True,
        enforce_linear_iteration=with_index,
    )
    return IterationLayout(
        tuple(iterator.shape),
        tuple(iterator.element_strides(i) for i in range(iterator.ntensors)),
    )


@cache
def tma_layout(
    operands: tuple[tuple[torch.dtype, Shape, Shape], ...],
    shape: Shape,
    out_dtype: torch.dtype,
    out_strides: Shape,
) -> TmaLayout | None:
    r"""Classify shapes/strides; ``tma_eligible`` checks runtime device,
    alignment, aliases and shared memory limits."""
    if not operands or not 0 < math.prod(shape) < 2**31 - 1:
        return None
    inputs = [
        torch.empty_strided(s, st, dtype=d, device="meta") for d, s, st in operands
    ]
    output = torch.empty_strided(shape, out_strides, dtype=out_dtype, device="meta")
    try:
        iterator = TensorIterator(
            outputs=[output],
            const_inputs=inputs,
            resize_outputs=False,
            check_all_same_dtype=False,
        )
    except RuntimeError:
        return None
    dims = list(iterator.shape)
    strides = [list(iterator.strides(i)) for i in range(iterator.ntensors)]
    dim = 0
    while dim + 1 < len(dims):
        if all(s[dim] * dims[dim] == s[dim + 1] for s in strides):
            dims[dim] *= dims.pop(dim + 1)
            for stride in strides:
                stride.pop(dim + 1)
        else:
            dim += 1
    if not 1 <= len(dims) <= 3:
        return None
    dense = [out_dtype.itemsize]
    for extent in dims[:-1]:
        dense.append(dense[-1] * extent)
    if strides[0] != dense:
        return None
    if len(dims) == 1:
        if dims[0] % (TMA_TILE * TMA_TILE):
            return None
        cols = (
            shape[-1]
            if shape[-1] % TMA_TILE == 0 and math.prod(shape[:-1]) % TMA_TILE == 0
            else TMA_TILE
        )
        if cols == TMA_TILE and dims[0] % (TMA_TILE**3) == 0:
            cols = TMA_TILE * TMA_TILE
        rows, batch = dims[0] // cols, 1
    else:
        cols, rows = dims[:2]
        batch = dims[2] if len(dims) == 3 else 1
    if cols % TMA_TILE or rows % TMA_TILE or batch > 65535:
        return None
    views, orientations = [], []
    for i, inp in enumerate(inputs, 1):
        if inp.numel() == 1:
            views.append(((1,), (1,)))
            orientations.append(None)
            continue
        if tuple(inp.shape) != shape:
            return None
        stride = strides[i]
        width = inp.element_size()
        if len(dims) == 1:
            if stride[0] != width:
                return None
            fast, slow = width, cols * width
        else:
            fast, slow = stride[:2]
        transposed = slow == width and fast >= rows * width
        if not transposed and not (fast == width and slow >= cols * width):
            return None
        pitch = fast if transposed else slow
        if pitch % 16:
            return None
        if len(dims) == 3 and stride[2] != (cols * fast if transposed else rows * slow):
            return None
        view = (
            ((rows, batch * cols), (1, pitch // width))
            if transposed
            else ((batch * rows, cols), (pitch // width, 1))
        )
        views.append(view)
        orientations.append(transposed)
    row_io = sum(
        t.element_size()
        for t, orientation in zip(inputs, orientations)
        if orientation is True
    )
    col_io = (
        sum(
            t.element_size()
            for t, orientation in zip(inputs, orientations)
            if orientation is False
        )
        + out_dtype.itemsize
    )
    rows_first = row_io > col_io
    groups = (cols + TMA_TILE * TMA_STAGES - 1) // (TMA_TILE * TMA_STAGES)
    if (groups if rows_first else rows // TMA_TILE) > 65535:
        return None
    return TmaLayout(
        tuple(orientations),
        tuple(views),
        ((batch, rows, cols), (rows * cols, cols, 1)),
        rows_first,
    )


def row_gap_view(t: torch.Tensor) -> RowGap | None:
    shape, stride = tuple(t.shape), t.stride()
    if not shape or stride[-1] != 1:
        return None
    run, expect, k = 1, 1, 0
    for i in range(len(shape) - 1, -1, -1):
        if stride[i] == expect:
            run *= shape[i]
            expect *= shape[i]
            k += 1
        else:
            break
    if k == len(shape):
        return None  # Dense inputs use the flat vector path.
    outer_shape, outer_stride = shape[: len(shape) - k], stride[: len(shape) - k]
    for i in range(len(outer_shape) - 1):
        if outer_shape[i + 1] * outer_stride[i + 1] != outer_stride[i]:
            return None
    return t.numel() // run, run, outer_stride[-1]
