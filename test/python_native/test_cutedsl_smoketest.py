# Owner(s): ["module: dsl-native-ops"]
#
# Smoke tests for cuteDSL API surface. Run before bumping version pins in
# torch/_native/cutedsl_utils.py.  Each test compiles, runs, and verifies a
# small kernel against a PyTorch reference.
#
# Kernel code is adapted from:
#   cutlass/examples/python/CuTeDSL/ampere/elementwise_add.py  (BSD-3)
#   cutlass/examples/python/CuTeDSL/ampere/sgemm.py            (BSD-3)
#   cutlass/examples/python/CuTeDSL/hopper/cta_norm.py         (BSD-3)

import math
import unittest

import torch
from torch.testing._internal.common_cuda import SM80OrLater, TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


try:
    import cuda.bindings.driver as cuda
except ImportError:
    cuda = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Elementwise add kernel
# APIs: @cute.kernel, @cute.jit, cute.compile, from_dlpack, make_tiled_copy_tv,
#       zipped_divide, make_fragment_like, copy with predication,
#       make_identity_tensor, elem_less, make_copy_atom
# ---------------------------------------------------------------------------
def _build_elementwise_add():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    @cute.kernel
    def _kernel(gA, gB, gC, cC, shape, thr_layout, val_layout):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        blk_coord = ((None, None), bidx)
        blkA = gA[blk_coord]
        blkB = gB[blk_coord]
        blkC = gC[blk_coord]
        blkCrd = cC[blk_coord]

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gA.element_type
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gC.element_type
        )

        tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
        tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
        tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)

        thrA = thr_copy_A.partition_S(blkA)
        thrB = thr_copy_B.partition_S(blkB)
        thrC = thr_copy_C.partition_S(blkC)

        frgA = cute.make_fragment_like(thrA)
        frgB = cute.make_fragment_like(thrB)
        frgC = cute.make_fragment_like(thrC)

        thrCrd = thr_copy_C.partition_S(blkCrd)
        frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
        for i in range(0, cute.size(frgPred), 1):
            frgPred[i] = cute.elem_less(thrCrd[i], shape)

        cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
        cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)

        result = frgA.load() + frgB.load()
        frgC.store(result)

        cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)

    @cute.jit
    def _host(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
        dtype = mA.element_type
        vector_size = copy_bits // dtype.width
        thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
        val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gA = cute.zipped_divide(mA, tiler_mn)
        gB = cute.zipped_divide(mB, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        idC = cute.make_identity_tensor(mC.shape)
        cC = cute.zipped_divide(idC, tiler=tiler_mn)

        _kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
            grid=[cute.size(gC, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )

    return _host, from_dlpack


# ---------------------------------------------------------------------------
# SIMT SGEMM kernel (FP32, Ampere cp.async pipeline)
# APIs: SmemAllocator, cpasync.CopyG2SOp, cp_async_commit/wait_group,
#       make_tiled_mma (MmaUniversalOp), cute.gemm, NamedBarrier,
#       local_tile, domain_offset, make_composed_layout, autovec_copy
# ---------------------------------------------------------------------------
def _build_sgemm():
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack

    class SGemm:
        def __init__(self, cta_tiler=(128, 128, 8), num_stages=3, num_threads=256):
            self._cta_tiler = cta_tiler
            self._num_stages = num_stages
            self._num_threads = num_threads
            self._bM, self._bN, self._bK = cta_tiler
            self.cta_sync_barrier = pipeline.NamedBarrier(
                barrier_id=1, num_threads=num_threads
            )

        @cute.jit
        def __call__(
            self,
            mA,
            mB,
            mC,
            epilogue_op: cutlass.Constexpr = lambda x: x,
            stream: cuda.CUstream = cuda.CUstream(
                cuda.CUstream_flags.CU_STREAM_DEFAULT
            ),
        ):
            self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
            self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
            self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

            padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
            padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
            sA_layout = cute.make_layout(
                (self._bM, self._bK, self._num_stages),
                stride=(1, self._bM + padding_a, self._bK * (self._bM + padding_a)),
            )
            sB_layout = cute.make_layout(
                (self._bN, self._bK, self._num_stages),
                stride=(1, self._bN + padding_b, self._bK * (self._bN + padding_b)),
            )

            tA = cute.make_layout(
                (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
            )
            tB = cute.make_layout(
                (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
            )
            vA = cute.make_layout((1, 1))
            vB = cute.make_layout((1, 1))
            atom_async_copy_A = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mA.element_type.width,
            )
            atom_async_copy_B = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mB.element_type.width,
            )
            if cutlass.const_expr(self.a_major_mode == utils.LayoutEnum.COL_MAJOR):
                num_vectorized = 4 if (mA.layout[0].max_alignment % 16 == 0) else 1
                atom_async_copy_A = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    mA.element_type,
                    num_bits_per_copy=mA.element_type.width * num_vectorized,
                )
                major_mode_size = self._bM // num_vectorized
                tA = cute.make_layout(
                    (major_mode_size, self._num_threads // major_mode_size),
                    stride=(1, major_mode_size),
                )
                vA = cute.make_layout((num_vectorized, 1))
            if cutlass.const_expr(self.b_major_mode == utils.LayoutEnum.COL_MAJOR):
                num_vectorized = 4 if (mB.layout[0].max_alignment % 16 == 0) else 1
                atom_async_copy_B = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    mA.element_type,
                    num_bits_per_copy=mB.element_type.width * num_vectorized,
                )
                major_mode_size = self._bN // num_vectorized
                tB = cute.make_layout(
                    (major_mode_size, self._num_threads // major_mode_size),
                    stride=(1, major_mode_size),
                )
                vB = cute.make_layout((num_vectorized, 1))

            tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA, vA)
            tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_B, tB, vB)

            atoms_layout = cute.make_layout(
                (self._num_threads // 16, 16, 1), stride=(16, 1, 0)
            )
            if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):
                atoms_layout = cute.make_layout(
                    (16, self._num_threads // 16, 1), stride=(1, 16, 0)
                )
            op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
            permutation_tiler_M = cute.make_layout(
                (atoms_layout.shape[0], 4), stride=(4, 1)
            )
            permutation_tiler_N = cute.make_layout(
                (atoms_layout.shape[1], 4), stride=(4, 1)
            )
            tiled_mma = cute.make_tiled_mma(
                op,
                atoms_layout,
                permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
            )

            grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
            self.kernel(
                mA,
                mB,
                mC,
                sA_layout,
                sB_layout,
                tiled_copy_A,
                tiled_copy_B,
                tiled_mma,
                epilogue_op,
            ).launch(
                grid=grid_dim,
                block=[cute.size(atoms_layout), 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            epilogue_op: cutlass.Constexpr = lambda x: x,
        ):
            tidx, tidy, tidz = cute.arch.thread_idx()
            bidx, bidy, bidz = cute.arch.block_idx()
            tiler_coord = (bidx, bidy, None)
            thr_mma = tiled_mma.get_slice(tidx)

            gA = cute.local_tile(
                mA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
            )
            gB = cute.local_tile(
                mB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
            )
            gC = cute.local_tile(
                mC, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, 1, None)
            )

            residue_k = mA.shape[1] - self._bK * gA.shape[2]
            gA = cute.domain_offset((0, residue_k, 0), gA)
            gB = cute.domain_offset((0, residue_k, 0), gB)

            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
            sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)

            mcA = cute.make_identity_tensor(mA.shape)
            mcB = cute.make_identity_tensor(mB.shape)
            cA = cute.local_tile(
                mcA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
            )
            cB = cute.local_tile(
                mcB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
            )
            cA = cute.domain_offset((0, residue_k, 0), cA)
            cB = cute.domain_offset((0, residue_k, 0), cB)
            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tAsA.shape[0][1],
                        cute.size(tAsA, mode=[1]),
                        cute.size(tAsA, mode=[2]),
                    ),
                    stride=(cute.size(tAsA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tBsB.shape[0][1],
                        cute.size(tBsB, mode=[1]),
                        cute.size(tBsB, mode=[2]),
                    ),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            tApA_residue_k = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tAsA.shape[0][1],
                        cute.size(tAsA, mode=[1]),
                        cute.size(tAsA, mode=[2]),
                    ),
                    stride=(
                        cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                        cute.size(tAsA, mode=[2]),
                        1,
                    ),
                ),
                cutlass.Boolean,
            )
            tBpB_residue_k = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tBsB.shape[0][1],
                        cute.size(tBsB, mode=[1]),
                        cute.size(tBsB, mode=[2]),
                    ),
                    stride=(
                        cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                        cute.size(tBsB, mode=[2]),
                        1,
                    ),
                ),
                cutlass.Boolean,
            )

            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(
                        tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                    )
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(
                        tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                    )
            for rest_v in range(tApA_residue_k.shape[0]):
                for m in range(tApA_residue_k.shape[1]):
                    for k in range(tApA_residue_k.shape[2]):
                        coord_A = tAcA[(0, rest_v), m, k, 0]
                        tApA_residue_k[rest_v, m, k] = cute.elem_less(
                            (coord_A[0], cutlass.Int32(-1)), (mA.shape[0], coord_A[1])
                        )
            for rest_v in range(tBpB_residue_k.shape[0]):
                for n in range(tBpB_residue_k.shape[1]):
                    for k in range(tBpB_residue_k.shape[2]):
                        coord_B = tBcB[(0, rest_v), n, k, 0]
                        tBpB_residue_k[rest_v, n, k] = cute.elem_less(
                            (coord_B[0], cutlass.Int32(-1)), (mB.shape[0], coord_B[1])
                        )

            # Prefetch prologue
            k_pipe_max = cute.size(tAsA, mode=[3])
            k_tile_count = cute.size(tAgA, mode=[3])
            gmem_pipe_read = cutlass.Int32(0)
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, 0],
                pred=tApA_residue_k,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, 0],
                pred=tBpB_residue_k,
            )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < k_tile_count
                else cutlass.Int32(0)
            )
            for k_tile in range(1, k_pipe_max - 1):
                if k_tile < k_tile_count:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, k_tile],
                        pred=tApA,
                    )
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, gmem_pipe_read],
                        tBsB[None, None, None, k_tile],
                        pred=tBpB,
                    )
                gmem_pipe_read = (
                    gmem_pipe_read + 1
                    if gmem_pipe_read + 1 < k_tile_count
                    else cutlass.Int32(0)
                )
                cute.arch.cp_async_commit_group()

            if k_tile_count < k_pipe_max:
                for rest_v in range(tApA.shape[0]):
                    for m in range(tApA.shape[1]):
                        tApA[rest_v, m, 0] = cutlass.Boolean(0)
                for rest_v in range(tBpB.shape[0]):
                    for n in range(tBpB.shape[1]):
                        tBpB[rest_v, n, 0] = cutlass.Boolean(0)

            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCgC = thr_mma.partition_C(gC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            smem_pipe_read = cutlass.Int32(0)
            smem_pipe_write = cutlass.Int32(k_pipe_max - 1)
            tCsA_p = tCsA[None, None, None, smem_pipe_read]
            tCsB_p = tCsB[None, None, None, smem_pipe_read]

            k_block_max = cute.size(tCrA, mode=[2])
            if k_block_max > 1:
                cute.arch.cp_async_wait_group(k_pipe_max - 2)
                self.cta_sync_barrier.arrive_and_wait()
                cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
                cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

            # Mainloop
            for _ in range(k_tile_count):
                for k_block in range(k_block_max, unroll_full=True):
                    if k_block == k_block_max - 1:
                        tCsA_p = tCsA[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(k_pipe_max - 2)
                        self.cta_sync_barrier.arrive_and_wait()

                    k_block_next = (k_block + 1) % k_block_max
                    cute.autovec_copy(
                        tCsA_p[None, None, k_block_next], tCrA[None, None, k_block_next]
                    )
                    cute.autovec_copy(
                        tCsB_p[None, None, k_block_next], tCrB[None, None, k_block_next]
                    )

                    if k_block == 0:
                        cute.copy(
                            tiled_copy_A,
                            tAgA[None, None, None, gmem_pipe_read],
                            tAsA[None, None, None, smem_pipe_write],
                            pred=tApA,
                        )

                    cute.gemm(
                        tiled_mma,
                        tCrC,
                        tCrA[None, None, k_block],
                        tCrB[None, None, k_block],
                        tCrC,
                    )

                    if k_block == 0:
                        cute.copy(
                            tiled_copy_B,
                            tBgB[None, None, None, gmem_pipe_read],
                            tBsB[None, None, None, smem_pipe_write],
                            pred=tBpB,
                        )
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == k_pipe_max:
                            smem_pipe_read = cutlass.Int32(0)
                        gmem_pipe_read = (
                            gmem_pipe_read + 1
                            if gmem_pipe_read + 1 < k_tile_count
                            else cutlass.Int32(1)
                        )

            # Epilogue
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            tCrC.store(epilogue_op(tCrC.load()))

            cC = cute.make_identity_tensor(gC.shape)
            tCpC = thr_mma.partition_C(cC)
            predC = cute.make_rmem_tensor(tCrC.layout, cutlass.Boolean)
            residue_m = mC.shape[0] - cutlass.Int32(self._bM) * bidx
            residue_n = mC.shape[1] - cutlass.Int32(self._bN) * bidy
            for i in range(cute.size(tCrC.shape)):
                predC[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
            cute.copy(atom, tCrC, tCgC, pred=predC)

    return SGemm, from_dlpack


# ---------------------------------------------------------------------------
# CTA-level LayerNorm / RMSNorm kernel
# APIs: warp reductions (shuffle_sync_bfly), SmemAllocator, vectorized
#       128-bit copies (CopyUniversalOp + num_bits_per_copy), TensorSSA
#       load/store, cute.rsqrt, predication, autovec_copy
# ---------------------------------------------------------------------------
def _build_cta_norm():
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    class CtaNorm:
        def __init__(self, N, norm_type, threads_per_cta=None):
            self.N = N
            self.norm_type = norm_type
            self.elems_per_thread = 8
            self.warp_size = 32
            self.threads_per_cta = threads_per_cta or self._heuristic_threads()
            self.warps_per_cta = (self.threads_per_cta + 31) // self.warp_size

        def _heuristic_threads(self):
            elems_per_warp = self.elems_per_thread * self.warp_size
            heu_warps = (self.N + elems_per_warp - 1) // elems_per_warp // 4
            heu_warps = max(heu_warps, 1)
            heu_warps = (heu_warps + 1) // 2 * 2
            return heu_warps * 32

        @cute.jit
        def __call__(self, mY, mX, mWeight, mBias, eps: cutlass.Float32 = 1e-6):
            M, _ = mX.shape
            atom_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mX.element_type,
                num_bits_per_copy=128,
            )
            t_layout = cute.make_layout(self.threads_per_cta)
            v_layout = cute.make_layout(self.elems_per_thread)
            tiled_copy = cute.make_tiled_copy_tv(atom_copy, t_layout, v_layout)
            self.kernel(mY, mX, mWeight, mBias, tiled_copy, eps).launch(
                grid=[M, 1, 1],
                block=[self.warps_per_cta * self.warp_size, 1, 1],
            )

        @cute.kernel
        def kernel(self, mY, mX, mWeight, mBias, tiled_copy, eps):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            thr_copy = tiled_copy.get_slice(tidx)

            gY = cute.local_tile(mY, tiler=(1, self.N), coord=(bidx, 0))
            gX = cute.local_tile(mX, tiler=(1, self.N), coord=(bidx, 0))
            gY, gX = gY[0, None], gX[0, None]

            tYgY = thr_copy.partition_S(gY)
            pred = cute.make_rmem_tensor(cute.size(tYgY, mode=[1]), cutlass.Boolean)
            for i in range(cute.size(pred)):
                offset = (i * self.threads_per_cta + tidx) * self.elems_per_thread
                pred[i] = offset < self.N

            tXgX = thr_copy.partition_S(gX)
            tWgW = thr_copy.partition_S(mWeight)
            if cutlass.const_expr(self.norm_type == "layer"):
                tBgB = thr_copy.partition_S(mBias)
            tXrX = cute.make_fragment_like(tXgX)
            tXrX.fill(0)
            tWrW = cute.make_fragment_like(tWgW)
            if cutlass.const_expr(self.norm_type == "layer"):
                tBrB = cute.make_fragment_like(tBgB)

            for i in range(cute.size(tXrX, mode=[1])):
                if pred[i]:
                    cute.autovec_copy(tXgX[None, i], tXrX[None, i])
                    cute.autovec_copy(tWgW[None, i], tWrW[None, i])
                    if cutlass.const_expr(self.norm_type == "layer"):
                        cute.autovec_copy(tBgB[None, i], tBrB[None, i])

            if cutlass.const_expr(self.norm_type == "layer"):
                tYrY = self.apply_layernorm(tXrX, tWrW, tBrB, eps, tidx, pred)
            elif cutlass.const_expr(self.norm_type == "rms"):
                tYrY = self.apply_rmsnorm(tXrX, tWrW, eps, tidx, pred)

            for i in range(cute.size(tXrX, mode=[1])):
                if pred[i]:
                    cute.autovec_copy(tYrY[None, i], tYgY[None, i])

        @cute.jit
        def warp_reduce(self, val, reduce_size=32):
            iters = int(math.log2(reduce_size))
            for i in range(iters):
                val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
            return val

        @cute.jit
        def cta_reduce(self, val, acc, tidx):
            warp_id = tidx >> 5
            lane_id = tidx & 31
            if lane_id == 0:
                acc[warp_id] = val
            cute.arch.sync_threads()
            if warp_id == 0:
                val = (
                    acc[lane_id] if lane_id < self.warps_per_cta else cutlass.Float32(0)
                )
                val = self.warp_reduce(val)
                acc[self.warps_per_cta] = val
            cute.arch.sync_threads()
            val = acc[self.warps_per_cta]
            return val

        @cute.jit
        def apply_layernorm(self, x, weight, bias, eps, tidx, pred):
            smem = cutlass.utils.SmemAllocator()
            acc = smem.allocate_tensor(cutlass.Float32, self.warps_per_cta + 1)
            val = cute.Float32(0.0)
            for idx in range(cute.size(x)):
                val += x[idx].to(cutlass.Float32)
            val = self.warp_reduce(val)
            val = self.cta_reduce(val, acc, tidx)
            mean = val / self.N
            val = cute.Float32(0.0)
            for i in range(cute.size(x, mode=[1])):
                if pred[i]:
                    for idx in range(cute.size(x[None, i])):
                        x_fp32 = x[None, i][idx].to(cutlass.Float32)
                        val += (x_fp32 - mean) * (x_fp32 - mean)
            val = self.warp_reduce(val)
            val = self.cta_reduce(val, acc, tidx)
            factor = cute.rsqrt(val / self.N + eps)
            normed = cute.make_fragment_like(x)
            value = (x.load() - mean) * factor * weight.load() + bias.load()
            normed.store(value.to(normed.element_type))
            return normed

        @cute.jit
        def apply_rmsnorm(self, x, weight, eps, tidx, pred):
            smem = cutlass.utils.SmemAllocator()
            acc = smem.allocate_tensor(cutlass.Float32, self.warps_per_cta + 1)
            val = cute.Float32(0.0)
            for i in range(cute.size(x, mode=[1])):
                if pred[i]:
                    for idx in range(cute.size(x[None, i])):
                        x_fp32 = x[None, i][idx].to(cutlass.Float32)
                        val += x_fp32 * x_fp32
            val = self.warp_reduce(val)
            acc_sq = self.cta_reduce(val, acc, tidx)
            factor = cute.rsqrt(acc_sq / self.N + eps)
            tNrN = cute.make_fragment_like(x)
            tNrN.store((x.load() * factor * weight.load()).to(tNrN.element_type))
            return tNrN

    return CtaNorm, from_dlpack, cutlass_torch


# ===========================================================================
# Tests
# ===========================================================================


@skipIfNoCuteDSL
@unittest.skipIf(not TEST_CUDA, "CUDA required")
class TestCuteDSLSmoketest(TestCase):
    """Smoke tests that compile, run, and verify cuteDSL kernels.

    Run before bumping cuteDSL version pins in torch/_native/cutedsl_utils.py.
    """

    @unittest.skipIf(not SM80OrLater, "SM80+ required")
    def test_elementwise_add(self):
        _host, from_dlpack = _build_elementwise_add()
        import cutlass.cute as cute

        M, N = 128, 128
        a = torch.randn(M, N, device="cuda", dtype=torch.float32)
        b = torch.randn(M, N, device="cuda", dtype=torch.float32)
        c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

        a_cute = from_dlpack(a).mark_layout_dynamic()
        b_cute = from_dlpack(b).mark_layout_dynamic()
        c_cute = from_dlpack(c).mark_layout_dynamic()

        compiled = cute.compile(_host, a_cute, b_cute, c_cute)
        compiled(a_cute, b_cute, c_cute)
        torch.cuda.synchronize()

        torch.testing.assert_close(c, a + b)

    @unittest.skipIf(not SM80OrLater, "SM80+ required")
    def test_sgemm(self):
        SGemm, from_dlpack = _build_sgemm()
        import cutlass.cute as cute

        M, N, K = 256, 256, 256
        a = (
            torch.empty(K, M, dtype=torch.int32)
            .random_(-5, 5)
            .to(dtype=torch.float32)
            .permute(1, 0)
            .cuda()
        )
        b = (
            torch.empty(K, N, dtype=torch.int32)
            .random_(-5, 5)
            .to(dtype=torch.float32)
            .permute(1, 0)
            .cuda()
        )
        c = (
            torch.empty(N, M, dtype=torch.int32)
            .random_(-5, 5)
            .to(dtype=torch.float32)
            .permute(1, 0)
            .cuda()
        )

        a_cute = from_dlpack(a, assumed_align=16)
        b_cute = from_dlpack(b, assumed_align=16)
        c_cute = from_dlpack(c, assumed_align=16)

        sgemm = SGemm()
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(sgemm, a_cute, b_cute, c_cute, stream=stream)
        compiled(a_cute, b_cute, c_cute)
        torch.cuda.synchronize()

        ref = torch.einsum("mk,nk->mn", a, b)
        torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-3, rtol=1e-5)

    @unittest.skipIf(not SM80OrLater, "SM80+ required")
    def test_cta_rmsnorm(self):
        CtaNorm, from_dlpack, cutlass_torch = _build_cta_norm()
        import cutlass
        import cutlass.cute as cute

        M, N = 32, 256
        dtype = cutlass.Float16
        eps = 1e-6
        torch_dtype = cutlass_torch.dtype(dtype)

        x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
        weight = torch.randn(N, device="cuda", dtype=torch_dtype)
        y = torch.empty_like(x)

        _x = from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)
        _w = from_dlpack(weight, assumed_align=16, enable_tvm_ffi=True)
        _y = from_dlpack(y, assumed_align=16, enable_tvm_ffi=True)

        norm = CtaNorm(N, "rms")
        compiled = cute.compile(norm, _y, _x, _w, None, options="--enable-tvm-ffi")
        compiled(y, x, weight, None, eps)
        torch.cuda.synchronize()

        ref = torch.rms_norm(x, (N,), weight, eps)
        torch.testing.assert_close(y, ref, atol=1e-3, rtol=1e-3)

    @unittest.skipIf(not SM80OrLater, "SM80+ required")
    def test_cta_layernorm(self):
        CtaNorm, from_dlpack, cutlass_torch = _build_cta_norm()
        import cutlass
        import cutlass.cute as cute

        M, N = 32, 256
        dtype = cutlass.Float16
        eps = 1e-6
        torch_dtype = cutlass_torch.dtype(dtype)

        x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
        weight = torch.randn(N, device="cuda", dtype=torch_dtype)
        bias = torch.randn(N, device="cuda", dtype=torch_dtype)
        y = torch.empty_like(x)

        _x = from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)
        _w = from_dlpack(weight, assumed_align=16, enable_tvm_ffi=True)
        _b = from_dlpack(bias, assumed_align=16, enable_tvm_ffi=True)
        _y = from_dlpack(y, assumed_align=16, enable_tvm_ffi=True)

        norm = CtaNorm(N, "layer")
        compiled = cute.compile(norm, _y, _x, _w, _b, options="--enable-tvm-ffi")
        compiled(y, x, weight, bias, eps)
        torch.cuda.synchronize()

        ref = torch.layer_norm(x, (N,), weight, bias, eps)
        torch.testing.assert_close(y, ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
