from typing import Any, TYPE_CHECKING

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.tensor_utils import LayoutEnum

from ...cutedsl import launch as _L
from ...cutedsl.types import NumericType
from ._transpose import matrix_view


if TYPE_CHECKING:
    from tvm_ffi import Function  # pyrefly: ignore[missing-import]


try:
    from cutlass.block import block_copy
    from cutlass.memory import SmemAllocator
except ImportError:
    from cutlass.utils import block_copy, SmemAllocator


class _TmaTranspose:
    storage_type: Any

    def __init__(self, dtype: NumericType) -> None:
        self.dtype: NumericType = dtype
        self.tile: int = 64
        self.threads: int = 128

    @cute.jit
    def __call__(
        self, src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream
    ) -> None:
        src = matrix_view(src)[None, None, 0, None, None]
        dst = matrix_view(dst)[None, None, 0, None, None]
        src = cute.make_tensor(
            src.iterator,
            cute.make_layout(
                (src.shape[1], src.shape[0], src.shape[2], src.shape[3]),
                stride=(src.stride[1], src.stride[0], src.stride[2], src.stride[3]),
            ),
        )
        tile = (self.tile, self.tile)
        layout = sm100_utils.make_smem_layout(
            LayoutEnum.from_tensor(src).mma_major_mode(),
            tile,
            self.dtype,
            1,
        )
        layout = cute.slice_(layout, (None, None, 0))
        load, tma_src = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            src,
            layout,
            tile,
        )
        store, tma_dst = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dst,
            layout,
            tile,
        )

        @cute.struct
        class Storage:
            barrier: cute.struct.MemRange[cutlass.Int64, 1]
            src: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)],
                128,
            ]
            dst: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)],
                128,
            ]

        self.storage_type = Storage
        # Whole strips keep registers contiguous and spread SIMT reads across banks.
        vector = self.tile * self.tile // self.threads
        fast = self.tile // vector
        copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype),
            cute.make_layout((self.threads // fast, fast), stride=(fast, 1)),
            cute.make_layout((1, vector)),
        )
        middle_div = cute.FastDivmodDivisorV2(cutlass.Int32(src.shape[3]))
        self.kernel(load, tma_src, store, tma_dst, layout, copy, middle_div).launch(
            grid=(
                cute.ceil_div(src.shape[0], self.tile),
                cute.ceil_div(src.shape[1], self.tile),
                src.shape[2] * src.shape[3],
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        load: cute.CopyAtom,
        src: cute.Tensor,
        store: cute.CopyAtom,
        dst: cute.Tensor,
        layout: cute.ComposedLayout,
        copy: cute.TiledCopy,
        middle_div: Any,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        n, m, batch = cute.arch.block_idx()
        outer, middle = divmod(batch, middle_div)
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        storage = SmemAllocator().allocate(self.storage_type)
        s_src = storage.src.get_tensor(layout.outer, swizzle=layout.inner)
        s_dst = storage.dst.get_tensor(layout.outer, swizzle=layout.inner)
        barrier = storage.barrier.data_ptr()
        if tid == 0:
            cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        tile = (self.tile, self.tile, 1, 1)
        g_src = cute.local_tile(src, tile, (n, m, outer, middle))
        g_dst = cute.local_tile(dst, tile, (m, n, outer, middle))
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    barrier,
                    self.tile * self.tile * self.dtype.width // 8,
                )
            block_copy(
                load,
                cute.group_modes(g_src, 0, 4),
                cute.group_modes(s_src, 0, 2),
                tma_bar_ptr=barrier,
            )
        cute.arch.mbarrier_wait(barrier, 0)
        transposed = cute.composition(
            s_dst.layout,
            cute.make_layout((self.tile, self.tile), stride=(self.tile, 1)),
        )
        out = cute.make_tensor(s_dst.iterator, transposed)
        thread = copy.get_slice(tid)
        source, target = thread.partition_S(s_src), thread.partition_D(out)
        regs = cute.make_rmem_tensor_like(target)
        cute.autovec_copy(source, regs)
        cute.autovec_copy(regs, target)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()
        if warp == 0:
            block_copy(
                store, cute.group_modes(s_dst, 0, 2), cute.group_modes(g_dst, 0, 4)
            )
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)


def compile_tma(word_size: int) -> "Function":
    from .transpose_copy import tiled_kernel

    vector = 16 // word_size
    shape = (
        cute.sym_int64(),
        cute.sym_int64(divisibility=vector),
        cute.sym_int64(),
        cute.sym_int64(divisibility=vector),
        1,
    )
    op, (src, dst) = tiled_kernel(word_size, 1, False, True, tma=True, shape=shape)
    return _L.compile_kernel(op, src, dst, _L.stream())
