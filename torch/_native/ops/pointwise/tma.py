from typing import Any

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync

from ...cutedsl.types import MathFn, NumericType
from ._types import Shape
from .kernel import _PointwiseOp
from .layout import (
    TMA_SMEM_ALIGNMENT,
    TMA_STAGES as _STAGES,
    TMA_THREADS as _THREADS,
    TMA_TILE as _BOX,
    TmaLayout,
)


try:
    from cutlass.block import block_copy
    from cutlass.memory import SmemAllocator
except ImportError:
    from cutlass.utils import block_copy, SmemAllocator


def fake_tensor(
    dtype: NumericType,
    view: tuple[Shape, Shape],
    transposed: bool | None = False,
    *,
    dynamic: bool = False,
) -> cute.Tensor:
    align = (
        max(dtype.width // 8, 1) if transposed is None else 128 if transposed else 16
    )
    shape, strides = view
    if dynamic and transposed is not None:
        shape = tuple(
            cute.sym_int32(divisibility=1 if len(shape) == 3 and i == 0 else _BOX)
            for i in range(len(shape))
        )
        strides = tuple(
            1 if s == 1 else cute.sym_int64(divisibility=16 // (dtype.width // 8))
            for s in strides
        )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=strides, assumed_align=align
    )


def shared_layout(dtype: NumericType, transposed: bool) -> cute.ComposedLayout:
    if not transposed:
        outer = cute.make_ordered_layout((_BOX, _BOX, _STAGES), order=(1, 0, 2))
        return cute.make_composed_layout(cute.make_swizzle(0, 4, 3), 0, outer)
    bits = 2 if dtype.width == 16 else 3
    elements = (128 << bits) // dtype.width
    outer = cute.make_layout((elements, 8), stride=(1, elements))
    atom = cute.make_composed_layout(cute.make_swizzle(bits, 4, 3), 0, outer)
    return cute.tile_to_shape(atom, (_BOX, _BOX, _STAGES), order=(1, 0, 2))


class ElementwiseTma(_PointwiseOp):
    def __init__(
        self,
        fn: MathFn,
        nconsts: int,
        compute: NumericType,
        output: NumericType,
        storage: NumericType,
        layout: TmaLayout,
        input_types: tuple[NumericType, ...],
        result_types: tuple[NumericType, ...],
    ) -> None:
        super().__init__(
            fn,
            len(layout.inputs),
            1,
            nconsts,
            compute,
            [output],
            input_types,
            result_types=result_types,
        )
        self.storage: NumericType = storage
        self.transposed: tuple[bool | None, ...] = layout.transposed
        self.ntma: int = sum(t is not None for t in self.transposed)
        self.rows_first: bool = layout.rows_first

    def fake_args(
        self,
        layout: TmaLayout,
        storage_types: tuple[NumericType, ...],
        *,
        dynamic: bool = False,
    ) -> tuple[list[cute.Tensor], list[cute.Tensor]]:
        r"""Build aligned operands for JIT compilation and AOT export."""
        inputs = [
            fake_tensor(dtype, view, transposed, dynamic=dynamic)
            for dtype, view, transposed in zip(
                storage_types, layout.inputs, layout.transposed, strict=True
            )
        ]
        return inputs, [fake_tensor(self.out_types[0], layout.output, dynamic=dynamic)]

    @cute.jit
    def __call__(
        self,
        inputs: Any,
        outputs: list[cute.Tensor],
        consts: Any,
        stream: cuda.CUstream,
    ) -> None:
        operands, layouts = [], []
        for k in cutlass.range_constexpr(self.nin):
            if const_expr(self.transposed[k] is None):
                operands.append(inputs[k])
                layouts.append(None)
            else:
                layout = shared_layout(self.storage, self.transposed[k])
                atom, tma_tensor = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileG2SOp(), inputs[k], layout, (_BOX, _BOX)
                )
                operands.append((atom, cute.zipped_divide(tma_tensor, (_BOX, _BOX))))
                layouts.append(layout)
        batch, rows, cols = outputs[0].shape
        row_tiles, groups = rows // _BOX, cute.ceil_div(cols, _BOX * _STAGES)
        grid = (
            [row_tiles, groups, batch]
            if self.rows_first
            else [groups, row_tiles, batch]
        )
        self.kernel(operands, outputs[0], consts, layouts).launch(
            grid=grid,
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, inputs: Any, output: cute.Tensor, consts: Any, layouts: Any
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        first, second, batch = cute.arch.block_idx()
        group, row = (second, first) if const_expr(self.rows_first) else (first, second)
        col = group * _STAGES
        flat_output = cute.make_tensor(
            output.iterator, cute.make_layout(cute.size(output))
        )
        ncols, nrows = output.shape[2] // _BOX, output.shape[1] // _BOX
        stages = cutlass.min(ncols - col, _STAGES)
        smem = SmemAllocator()
        buffers: Any = [
            None
            if self.transposed[k] is None
            else smem.allocate_tensor(
                self.storage,
                layouts[k].outer,
                byte_alignment=TMA_SMEM_ALIGNMENT,
                swizzle=layouts[k].inner,
            )
            for k in range(const_expr(self.nin))
        ]
        barriers = smem.allocate_array(cutlass.Int64, num_elems=_STAGES)
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == 0:
            for k in cutlass.range_constexpr(self.nin):
                if const_expr(self.transposed[k] is not None):
                    cpasync.prefetch_descriptor(inputs[k][0])
            for stage in cutlass.range_constexpr(_STAGES):
                if stage < stages:
                    barrier = barriers + stage
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(barrier, 1)
                        cute.arch.fence_proxy(kind="async.shared", space="cta")
                    for k in cutlass.range_constexpr(self.nin):
                        if const_expr(self.transposed[k] is not None):
                            atom, tiles = inputs[k]
                            coord = (
                                (row, batch * ncols + col + stage)
                                if self.transposed[k]
                                else (batch * nrows + row, col + stage)
                            )
                            block_copy(
                                atom,
                                cute.group_modes(tiles[((None, None), coord)], 0, 2),
                                cute.group_modes(buffers[k][None, None, stage], 0, 2),
                                tma_bar_ptr=barrier,
                            )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            barrier, self.ntma * _BOX * _BOX * self.storage.width // 8
                        )
        cute.arch.barrier()
        scalars = []
        for k in cutlass.range_constexpr(self.nin):
            if const_expr(self.transposed[k] is None):
                scalars.append(self._convert(k, inputs[k][0]))
            else:
                scalars.append(None)
        lane, iteration = tid % 32, (tid // 32) * 8
        # Lane XORs match the shared swizzle and keep output stores contiguous.
        for stage in cutlass.range_constexpr(_STAGES):
            if stage < stages:
                cute.arch.mbarrier_wait(barriers + stage, 0)
                for i in cutlass.range_constexpr(8):
                    xor = lane // 8 if self.storage.width == 32 else (lane // 16) * 2
                    slow = (iteration + i) ^ xor
                    values = tuple(
                        scalars[k]
                        if self.transposed[k] is None
                        else self._convert(k, buffers[k][slow, lane, stage])
                        for k in range(const_expr(self.nin))
                    )
                    value = self._evaluate(values, consts)
                    index = (
                        (batch * output.shape[1] + row * _BOX + slow) * output.shape[2]
                        + (col + stage) * _BOX
                        + lane
                    )
                    flat_output[index] = self._cast_result(0, value)
