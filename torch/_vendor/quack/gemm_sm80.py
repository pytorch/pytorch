# Copyright (c) 2026, Tri Dao.
#
# Ampere GEMM using warp-level MMA and cp.async global-to-shared loads.
# All CTA threads participate in cp.async, MMA, and epilogue.

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import warp

from .gemm_base import GemmBase, NamedBarrierGemm
from .tile_scheduler import TileSchedulerOptions
from .varlen_utils import VarlenArguments


class GemmSm80(GemmBase):
    """SM80 GEMM with cp.async loads and warp-level tensor-core MMA.

    SM80 has no TMA, so both the A/B mainloop and the epilogue global memory
    movement use per-thread copies. The epilogue still reuses the standard
    composable epilogue hooks used by the SM90/SM120 classes.
    """

    arch = 80
    _supported_archs = (80, 86, 87, 89)

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Union[Tuple[int, int], Tuple[int, int, int]],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = False,
        gather_A: bool = False,
        concat_layout: tuple | None = None,
        use_pdl: bool = False,
        num_warps: Optional[int] = None,
        occupancy: Optional[int] = None,
        arch: int = 80,
    ):
        if arch not in self._supported_archs:
            raise ValueError(
                f"SM80-family GEMM supports arch in {self._supported_archs}, got {arch}"
            )
        self.arch = arch
        self.acc_dtype = acc_dtype
        self.pingpong = False
        self.is_persistent = is_persistent
        self.use_clc_persistence = False
        self.use_pdl = use_pdl
        self.fp8_slow_accum = False
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()

        assert not pingpong, "SM8x GEMM does not use pingpong scheduling"
        assert cluster_shape_mnk == (1, 1, 1), "SM8x GEMM does not support CTA clusters"
        # The shared tile scheduler API still takes a cluster shape. SM8x only launches
        # independent CTAs, so keep this fixed at a single-CTA cluster.
        self.cluster_shape_mnk = (1, 1, 1)
        self.mma_inst_mnk = (16, 8, 16)
        if len(tile_shape_mnk) == 3:
            tile_m, tile_n, tile_k = tile_shape_mnk
        elif len(tile_shape_mnk) == 2:
            tile_m, tile_n = tile_shape_mnk
            tile_k = 4 * self.mma_inst_mnk[2]
        else:
            raise ValueError("SM80 tile shape must be (M, N) or (M, N, K)")
        if tile_m % 16 != 0 or tile_n % 16 != 0:
            raise ValueError("SM80 tile shape M/N must be divisible by 16")
        if tile_k <= 0 or tile_k % self.mma_inst_mnk[2] != 0:
            raise ValueError("SM80 tile shape K must be a positive multiple of MMA instruction K")
        self.cta_tile_shape_mnk = (tile_m, tile_n, tile_k)

        self.num_warps = (
            num_warps
            if num_warps is not None
            else (8 if (tile_m, tile_n) in ((128, 256), (128, 192)) else 4)
        )
        if self.num_warps not in (4, 8):
            raise ValueError("SM80 GEMM supports num_warps=4 or 8")
        self.atom_layout_mnk = (2, self.num_warps // 2, 1)
        self.mma_inst_tile_k = tile_k // self.mma_inst_mnk[2]
        self.threads_per_cta = self.num_warps * cute.arch.WARP_SIZE

        self.smem_capacity = self._smem_capacity_for_arch(self.arch)
        self.buffer_align_bytes = 1024
        default_occupancy = 1 if self.num_warps == 8 else 2
        ab_bytes_per_stage = (tile_m + tile_n) * tile_k * a_dtype.width // 8
        if (
            3 * ab_bytes_per_stage
            > self.smem_capacity // default_occupancy - self.buffer_align_bytes
        ):
            default_occupancy = 1
        self.occupancy = occupancy if occupancy is not None else default_occupancy
        self.num_epi_warps = self.num_warps
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )

        self.ab_stage = None
        self.epi_stage = None
        self.epi_c_stage = None
        self.epi_m_major = True
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_c_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None

    @staticmethod
    def _smem_capacity_for_arch(arch: int) -> int:
        # CUDA documents CC 8.7 with the same shared-memory capacity as CC 8.0,
        # but this CUTLASS helper build does not accept "sm_87".
        if arch == 87:
            arch = 80
        return cutlass.utils.get_smem_capacity_in_bytes(f"sm_{arch}")

    def _setup_tiled_mma(self):
        op = warp.MmaF16BF16Op(self.a_dtype, self.acc_dtype, self.mma_inst_mnk)
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        tC = cute.make_layout(self.atom_layout_mnk)
        permutation_mnk = (
            atom_m * self.mma_inst_mnk[0],
            atom_n * self.mma_inst_mnk[1] * 2,
            atom_k * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
        self.tiled_mma_gated_postact = self.tiled_mma

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args: TileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        raise NotImplementedError("Gemm Sm80 is not implemented yet")
