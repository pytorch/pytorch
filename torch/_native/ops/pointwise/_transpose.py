from typing import Any

import cutlass
import cutlass.cute as cute

from ...cutedsl import launch
from ...cutedsl.types import Index, NumericType


@cute.jit
def source_row(row: Index, a: int, middle: int, b: int) -> Index:
    a_idx = row % a
    row = row // a
    middle_idx = row % middle
    row = row // middle
    return ((row // b * a + a_idx) * middle + middle_idx) * b + row % b


@cute.jit
def source_offset(
    row: Index, q: Index, b: int, middle: int, a: int, inner: int
) -> Index:
    return source_row(row, a, middle, b) * inner + q


@cute.jit
def source_row_fast(
    row: Index, divisors: tuple[Any, Any, Any], middle_one: cutlass.Constexpr = False
) -> Index:
    a, middle, b = divisors
    row, a_idx = divmod(cutlass.Int32(row), a)
    if cutlass.const_expr(middle_one):
        middle_idx = cutlass.Int32(0)
        middle_size = 1
    else:
        row, middle_idx = divmod(row, middle)
        middle_size = middle.divisor
    outer, b_idx = divmod(row, b)
    return ((outer * a.divisor + a_idx) * middle_size + middle_idx) * b.divisor + b_idx


@cute.jit
def source_row_pow2(row: Index, a_shift: Index, b_shift: Index) -> Index:
    row = cutlass.Int32(row)
    a_idx = row & ((1 << a_shift) - 1)
    row = row >> a_shift
    b_idx = row & ((1 << b_shift) - 1)
    outer = row >> b_shift
    return (((outer << a_shift) + a_idx) << b_shift) + b_idx


@cute.jit
def matrix_view(t: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(
        t.iterator,
        cute.make_layout(
            (t.shape[3], t.shape[1], t.shape[4], t.shape[0], t.shape[2]),
            stride=(t.stride[3], t.stride[1], t.stride[4], t.stride[0], t.stride[2]),
        ),
    )


@cute.jit
def predicate(coords: cute.Tensor, rows: Index, cols: Index) -> cute.Tensor:
    pred = cute.make_rmem_tensor((1, *coords.shape[1:]), cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(coords.shape[1:])):
        rest = cute.idx2crd(i, coords.shape[1:])
        point = coords[(0, *rest)]
        pred[(0, *rest)] = (point[0] < rows) & (point[1] < cols)
    return pred


def fake_tiled(
    dtype: NumericType,
    lanes: int,
    *,
    packed: bool = False,
    align: int = 1,
    shape: tuple[int | cute.SymInt, ...] | None = None,
) -> tuple[cute.Tensor, cute.Tensor]:
    if shape is None:
        divisor = 4 if packed else 1
        shape = (
            launch.sym(),
            launch.sym(divisor),
            launch.sym(),
            launch.sym(divisor),
            lanes,
        )
    src_order = (4, 0, 1, 3, 2) if packed else (4, 1, 2, 3, 0)
    dst_order = (4, 3, 1, 0, 2) if packed else (4, 3, 2, 1, 0)
    return (
        launch.fake_compact(dtype, shape, stride_order=src_order, align=align),
        launch.fake_compact(dtype, shape, stride_order=dst_order, align=align),
    )
