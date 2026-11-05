# Pure metadata layout selection shared by JIT and AOT; no torch/CuTeDSL imports.
from __future__ import annotations

import math
from typing import NamedTuple

from ._limits import (
    _BARRIER_BYTES,
    _MAX_GRID_YZ,
    _MAX_INDEX,
    _MIN_ALIGNMENT,
    _SMEM_ALIGNMENT,
    _THREADS_PER_BLOCK,
    _TILES_PER_BLOCK,
    _TMA_ALIGNMENT,
    _TMA_BOX_SIDE,
)


class _TmaConfig(NamedTuple):
    # Tensor flags describe orientation relative to output.
    transposed: tuple[bool, ...]
    box: int
    block: int
    input_views: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    output_view: tuple[tuple[int, ...], tuple[int, ...]]


class _OperandLayout(NamedTuple):
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    element_bytes: int
    address_mod: int
    contiguous: bool


def _tma_geometry(element_bytes: int) -> tuple[int, int] | None:
    # Return (square box side in elements, threads per block).
    # Each thread handles eight elements of the square box.
    if element_bytes in (2, 4):
        return _TMA_BOX_SIDE, _THREADS_PER_BLOCK
    return None


def _physical_dim_order(strides: tuple[int, ...]) -> tuple[int, ...] | None:
    # Order dimensions from physically contiguous to outermost; reject broadcasts.
    if not strides or any(stride <= 0 for stride in strides):
        return None
    order = tuple(sorted(range(len(strides)), key=strides.__getitem__))
    if strides[order[0]] != 1:
        return None
    if any(strides[order[i]] <= strides[order[i - 1]] for i in range(1, len(order))):
        return None
    return order


def _dense_run(
    shape: tuple[int, ...], strides: tuple[int, ...], dims: tuple[int, ...]
) -> bool:
    # These dimensions can be flattened together without crossing storage gaps.
    return all(
        shape[dims[i - 1]] * strides[dims[i - 1]] == strides[dims[i]]
        for i in range(1, len(dims))
    )


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    # Output order is logical row-major, independent of the input permutations.
    strides = [0] * len(shape)
    stride = 1
    for dim in range(len(shape) - 1, -1, -1):
        strides[dim] = stride
        stride *= shape[dim]
    return tuple(strides)


def _transpose_split(
    orders: list[tuple[int, ...]],
    shape: tuple[int, ...],
    strides: list[tuple[int, ...]],
    output_order: tuple[int, ...],
) -> tuple[int, int | None] | None:
    # Return group boundaries in output's fastest-first order [A, B, batch].
    # Transposed inputs have order [B, A, batch]. Without a transpose, return
    # (end_of_A, None) so the caller can choose the row/batch boundary.
    identity = tuple(range(len(shape)))
    swapped = next((order for order in orders if order != identity), None)
    if swapped is not None:
        a_end = swapped[0]
        b_end = a_end + swapped.index(0)
        expected = (
            tuple(range(a_end, b_end))
            + tuple(range(a_end))
            + tuple(range(b_end, len(shape)))
        )
        if (
            a_end == 0
            or swapped != expected
            or any(order not in (identity, swapped) for order in orders)
        ):
            return None
        return a_end, b_end
    # Row-padded inputs have one common dense-run break separating columns
    # from rows. It must occur at the same boundary in every input.
    breaks = {
        i
        for stride in strides
        for i in range(1, len(shape))
        if shape[output_order[i - 1]] * stride[output_order[i - 1]]
        != stride[output_order[i]]
    }
    return (breaks.pop(), None) if len(breaks) == 1 else None


def _tma2d_config(
    inputs: tuple[_OperandLayout, ...],
    shape: tuple[int, ...],
    smem_limit: int,
) -> _TmaConfig | None:
    # Accept higher-order permutations that collapse to one 2-D transpose, plus
    # a batch group. Physical orders identify the swapped groups directly.
    # Common admission has checked shapes, index spans and output alignment.
    # Worked example (two compatible 4-D permutations, contiguous output):
    # x = empty(2, 64, 3, 32).permute(0, 2, 3, 1)
    # y = empty(2, 64, 128)[..., :96].view(2, 64, 3, 32).permute(0, 2, 3, 1)
    # Both have shape (2, 3, 32, 64); their strides are (6144, 32, 1, 96)
    # and (8192, 32, 1, 128). The gap in y is between its dense row groups.
    # The steps below derive the 2-D input views and a (batch, row, col) output.
    # 1. Select 32x32-element boxes, 128 threads, and check pipeline storage.
    geometry = _tma_geometry(inputs[0].element_bytes)
    if geometry is None:
        return None
    box, block = geometry
    tile_bytes = box * box * inputs[0].element_bytes
    smem_bytes = len(inputs) * (_TILES_PER_BLOCK * tile_bytes + _SMEM_ALIGNMENT)
    if smem_bytes + _TILES_PER_BLOCK * _BARRIER_BYTES > smem_limit:
        return None
    # 2. Output strides are (6144, 2048, 64, 1), giving physical order
    # (3, 2, 1, 0), fastest first.
    output_strides = _contiguous_strides(shape)
    output_order = _physical_dim_order(output_strides)
    if output_order is None:
        return None
    # 3. Both inputs have physical order (2, 1, 3, 0). In output-order
    # positions this is (1, 2, 0, 3): swap groups A=(3,) and B=(2, 1).
    positions = {dim: position for position, dim in enumerate(output_order)}
    orders = []
    strides = [t.strides for t in inputs]
    for stride in strides:
        order = _physical_dim_order(stride)
        if order is None:
            return None
        orders.append(tuple(positions[dim] for dim in order))
    split = _transpose_split(orders, shape, strides, output_order)
    if split is None:
        return None
    # 4. Split boundaries (1, 3) give A=64 columns, B=32*3=96 rows,
    # with dimension 0 as a batch of 2; both matrix extents fit full boxes.
    a_end, b_end = split
    a_dims = output_order[:a_end]
    shape_a = math.prod(shape[dim] for dim in a_dims)
    if shape_a % box != 0:
        return None
    identity = tuple(range(len(shape)))
    transposed = [order != identity for order in orders]
    # Only untransposed padded layouts need to choose a row/batch factoring.
    # Scan boundaries linearly to satisfy tile divisibility and grid Y/Z limits.
    ends = range(a_end + 1, len(shape) + 1) if b_end is None else (b_end,)
    for end in ends:
        b_dims, batch_dims = output_order[a_end:end], output_order[end:]
        shape_b = math.prod(shape[dim] for dim in b_dims)
        batch = math.prod(shape[dim] for dim in batch_dims)
        if shape_b % box != 0 or shape_b // box > _MAX_GRID_YZ or batch > _MAX_GRID_YZ:
            continue
        # 5. Verify each group is dense; padding is allowed only in the
        # pitch between groups. x has pitch 96 elements, y has pitch 128.
        input_views = []
        for t, is_transposed in zip(inputs, transposed):
            first = b_dims if is_transposed else a_dims
            second = a_dims if is_transposed else b_dims
            if not _dense_run(shape, t.strides, first) or not _dense_run(
                shape, t.strides, second + batch_dims
            ):
                return None
            extent = shape_b if is_transposed else shape_a
            pitch = t.strides[second[0]]
            alignment = _TMA_ALIGNMENT if is_transposed else _MIN_ALIGNMENT
            if (
                t.address_mod % alignment != 0
                or pitch < extent
                or pitch * t.element_bytes % _MIN_ALIGNMENT != 0
            ):
                return None
            view = (
                ((shape_b, batch * shape_a), (t.strides[first[0]], pitch))
                if is_transposed
                else ((batch * shape_b, shape_a), (pitch, t.strides[first[0]]))
            )
            input_views.append(view)
        # 6. Fold batch into each input's slow axis: shape (96, 128),
        # strides (1, 96) / (1, 128). Output is (2, 96, 64) with strides
        # (6144, 64, 1), so every input/output addresses the same logical element.
        output_stride_a, output_stride_b = (
            output_strides[a_dims[0]],
            output_strides[b_dims[0]],
        )
        output_batch_stride = (
            output_strides[batch_dims[0]] if batch_dims else shape_b * output_stride_b
        )
        return _TmaConfig(
            tuple(transposed),
            box,
            block,
            tuple(input_views),
            (
                (batch, shape_b, shape_a),
                (output_batch_stride, output_stride_b, output_stride_a),
            ),
        )
    return None


def _layout_config(
    inputs: tuple[_OperandLayout, ...],
    shape: tuple[int, ...],
    output_address: int,
    smem_limit: int,
) -> _TmaConfig | None:
    # Batch extents affect views and launch limits, so this decision is not cached.
    if (
        not inputs
        or not 0 < math.prod(shape) <= _MAX_INDEX
        or any(t.shape != shape for t in inputs)
        or all(t.contiguous for t in inputs)
    ):
        return None
    if len(shape) < 2 or output_address != 0:
        return None
    if any(
        sum((n - 1) * s for n, s in zip(t.shape, t.strides)) > _MAX_INDEX
        for t in inputs
    ):
        return None
    return _tma2d_config(inputs, shape, smem_limit)
