from typing import Dict, Tuple, Optional, Callable

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Boolean, const_expr
from cutlass.cute.runtime import make_ptr

from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from .activation import act_fn_map
from .gemm_act import GemmActMixin
from .gemm_sm80 import GemmSm80
from .gemm_sm90 import GemmSm90
from .gemm_sm100 import GemmSm100
from .gemm_sm120 import GemmSm120
from .gemm_tvm_ffi_utils import (
    div_for_dtype,
    perm3d,
    get_majors,
    get_dtypes,
    make_scheduler_args,
    make_fake_scheduler_args,
    compile_gemm_kernel,
)
from .cache_utils import jit_cache
from .tile_scheduler import TriangularTileScheduler
from .varlen_utils import VarlenManager
from . import copy_utils
from .rounding import RoundingMode, epilogue_sr_seed


class GemmSymmetricMixin(GemmActMixin):
    def get_scheduler_class(self, varlen_m: bool = False):
        return TriangularTileScheduler

    @cute.jit
    def epilogue(
        self,
        params: GemmActMixin.EpilogueParams,
        epi_smem_tensors: Dict[str, cute.Tensor],
        epi_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        epi_store_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        epi_read_state: Optional[cutlass.pipeline.PipelineState],
        epi_producer_state: Optional[cutlass.pipeline.PipelineState],
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_t2r: Optional[cute.TiledCopy],  # Only for Sm100
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: cute.Tensor,
        tiled_copy_s2r: Optional[cute.ThrCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        copy_C: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        has_C = const_expr(tRS_rC is not None)
        has_D = const_expr(copy_D is not None)
        use_tma_epi = const_expr(epi_store_pipeline is not None)
        use_tma_c = const_expr(epi_pipeline is not None)
        use_stochastic_rounding = const_expr(
            self.rounding_mode == RoundingMode.RS
            and self.acc_dtype == cutlass.Float32
            and self.d_dtype == cutlass.BFloat16
        )

        # Setup aux output (returns None for default epilogue, context tuple for Act)
        aux_out_ctx = self.epi_setup_aux_out(
            params,
            epi_smem_tensors,
            tiled_copy_r2s,
            tiled_copy_t2r,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )

        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
        )
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        # Symmetric guard: skip the mirrored aux write on the diagonal,
        # otherwise we'd write the same gmem location twice.
        square_tile_m = tile_coord_mnkl[0] // self.cluster_shape_mnk[0]
        square_tile_n = tile_coord_mnkl[1] // self.cluster_shape_mnk[1]

        epi_tensors = self.epi_begin(
            params,
            epi_smem_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            epilogue_barrier,
            tidx,
        )

        if const_expr(copy_C is not None):
            for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
                epi_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
                if const_expr(use_tma_c):
                    if is_tma_warp:
                        epi_pipeline.producer_acquire(epi_producer_state)
                        copy_C(src_idx=epi_coord_C, producer_state=epi_producer_state)
                        epi_pipeline.producer_commit(epi_producer_state)
                    epi_producer_state.advance()
                else:
                    # TODO: turn this to cp.async instead of direct G2R copy
                    copy_C(src_idx=epi_coord_C, dst_idx=epi_idx % self.epi_c_stage)
            if const_expr(use_tma_c):
                epilogue_barrier.arrive_and_wait()

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)  # (epi_m, epi_n)
            # Copy from acc to D registers
            load_acc_subtile(tRS_rD, epi_coord)
            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, epi_coord)
            if const_expr(has_C):
                if const_expr(use_tma_c):
                    epi_pipeline.consumer_wait(epi_read_state)
                    cute.copy(
                        tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC
                    )
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        epi_pipeline.consumer_release(epi_read_state)
                    epi_read_state.advance()
                else:
                    c_buffer = epi_idx % self.epi_c_stage
                    cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, c_buffer], tSR_rC)
                    # TODO: cp.async wait once we switch to cp.async
                    epilogue_barrier.arrive_and_wait()
            if const_expr(copy_C is not None and epi_idx + self.epi_c_stage < epi_tile_num):
                epi_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
                if const_expr(use_tma_c):
                    if is_tma_warp:
                        epi_pipeline.producer_acquire(epi_producer_state)
                        copy_C(src_idx=epi_coord_C, producer_state=epi_producer_state)
                        epi_pipeline.producer_commit(epi_producer_state)
                    epi_producer_state.advance()
                else:
                    epilogue_barrier.arrive_and_wait()
                    copy_C(
                        src_idx=epi_coord_C,
                        dst_idx=(epi_idx + self.epi_c_stage) % self.epi_c_stage,
                    )
            tRS_rAuxOut = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)
            self.epi_end_loop(
                params,
                epi_tensors,
                epi_coord,
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )
            if const_expr(aux_out_ctx is not None):
                tRS_rAuxOut_out = self.epi_convert_aux_out(
                    tRS_rAuxOut,
                    epi_loop_tensors["sr_seed"],
                    tidx,
                    tile_coord_mnkl,
                    num_prev_subtiles,
                    epi_idx,
                )
            if const_expr(use_tma_epi):
                if is_tma_warp:
                    epi_store_pipeline.producer_acquire()
            else:
                epilogue_barrier.arrive_and_wait()
            if const_expr(use_tma_epi):
                epilogue_barrier.arrive_and_wait()
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if const_expr(has_D):
                tRS_sD_cur = tRS_sD[None, None, None, epi_buffer]
                if const_expr(use_stochastic_rounding):
                    seed = epilogue_sr_seed(
                        epi_loop_tensors["sr_seed"], tile_coord_mnkl, num_prev_subtiles + epi_idx
                    )
                    copy_utils.sr_cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD_cur, seed, tidx)
                else:
                    copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD_cur)
            if const_expr(aux_out_ctx is not None):
                tiled_copy_aux_out_r2s, tRS_sAuxOut, copy_aux_out = aux_out_ctx
                cute.copy(
                    tiled_copy_aux_out_r2s,
                    # Need contiguous for Sm80 and Sm120 where acc layout is ((2, 2), MMA_M, MMA_N)
                    copy_utils.contiguous(tiled_copy_aux_out_r2s.retile(tRS_rAuxOut_out)),
                    tRS_sAuxOut[None, None, None, epi_buffer],
                )
            if const_expr(use_tma_epi):
                cute.arch.fence_view_async_shared()
                epilogue_barrier.arrive_and_wait()
                if is_tma_warp:
                    if const_expr(has_D):
                        copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
                    if const_expr(aux_out_ctx is not None):
                        if square_tile_m != square_tile_n:  # don't write twice on the diagonal
                            copy_aux_out(src_idx=epi_buffer, dst_idx=epi_coord)
                    epi_store_pipeline.producer_commit()
            else:
                epilogue_barrier.arrive_and_wait()
                if const_expr(has_D):
                    copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
                if const_expr(aux_out_ctx is not None):
                    if square_tile_m != square_tile_n:  # don't write twice on the diagonal
                        copy_aux_out(src_idx=epi_buffer, dst_idx=epi_coord)
                epilogue_barrier.arrive_and_wait()

        self.epi_end(
            params,
            epi_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )

        return epi_read_state, epi_producer_state


class GemmSymmetricSm80(GemmSymmetricMixin, GemmSm80):
    pass


class GemmSymmetricSm90(GemmSymmetricMixin, GemmSm90):
    pass


class GemmSymmetricSm100(GemmSymmetricMixin, GemmSm100):
    pass


class GemmSymmetricSm120(GemmSymmetricMixin, GemmSm120):
    pass


@jit_cache
def _compile_gemm_symmetric(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    c_major,
    postact_dtype,
    a_major,
    b_major,
    d_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    alpha_mode,
    beta_mode,
    device_capacity,
):
    sm_to_cls = {
        8: GemmSymmetricSm80,
        9: GemmSymmetricSm90,
        10: GemmSymmetricSm100,
        11: GemmSymmetricSm100,
        12: GemmSymmetricSm120,
    }
    GemmCls = sm_to_cls[device_capacity[0]]
    # Symmetric GEMM: m == n, so reuse the same sym_int for shape checking
    m, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int()
    a_leading = 1 if a_major == "k" else 0
    b_leading = 1 if b_major == "k" else 0
    d_leading = 1 if d_major == "n" else 0
    c_leading = 1 if c_major == "n" else 0
    div_a, div_b = div_for_dtype(a_dtype), div_for_dtype(b_dtype)
    div_d, div_c = div_for_dtype(d_dtype), div_for_dtype(c_dtype) if c_dtype else 1
    mA = fake_tensor(a_dtype, (m, k, l), leading_dim=a_leading, divisibility=div_a)
    mB = fake_tensor(b_dtype, (m, k, l), leading_dim=b_leading, divisibility=div_b)
    mD = fake_tensor(d_dtype, (m, m, l), leading_dim=d_leading, divisibility=div_d)
    mC = fake_tensor(c_dtype, (m, m, l), leading_dim=c_leading, divisibility=div_c)
    # PostAct = D.mT, so it has the opposite major from D (m↔n swapped)
    div_pa = div_for_dtype(postact_dtype)
    postact_leading = 1 if postact_major == "n" else 0
    mAuxOut = fake_tensor(
        postact_dtype, (m, m, l), leading_dim=postact_leading, divisibility=div_pa
    )

    def fake_scalar(mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Float32(1.0)
        else:
            return make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=4)

    activation = None  # identity
    act_fn = act_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(
        mAuxOut,
        act_fn,
        alpha=fake_scalar(alpha_mode),
        beta=fake_scalar(beta_mode),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = None
    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        False,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
    )


def gemm_symmetric(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, m, k)
    D: Optional[Tensor],  # (l, m, m)
    C: Optional[Tensor],  # (l, m, m)
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
) -> None:
    # Transpose D so the "activation" is a write to the mirrored tile
    PostAct = D.mT

    A_p, B_p, D_p, C_p = perm3d(A, B, D, C)
    PostAct_p = PostAct.permute(1, 2, 0) if PostAct.ndim == 3 else PostAct
    a_major, b_major, d_major, c_major = get_majors(A_p, B_p, D_p, C_p)
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D, C)
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    # PostAct = D.mT has swapped major: if D is n-major, PostAct is m-major
    postact_major = "n" if PostAct_p.stride(1) == 1 else "m"

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )

    if is_dynamic_persistent and device_capacity[0] <= 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    tile_shape_mn = (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    alpha_mode = 2 if isinstance(alpha, Tensor) else (1 if alpha != 1.0 else 0)
    beta_mode = 2 if isinstance(beta, Tensor) else (1 if beta != 1.0 else 0)

    compiled_fn = _compile_gemm_symmetric(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        c_major,
        postact_dtype,
        a_major,
        b_major,
        d_major,
        postact_major,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        is_dynamic_persistent,
        alpha_mode,
        beta_mode,
        device_capacity,
    )

    from .cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    cluster_size = cluster_M * cluster_N
    max_active_clusters = (
        get_max_active_clusters(cluster_size, device_capacity=device_capacity) if persistent else 0
    )

    def scalar_arg(scalar, mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Float32(scalar)
        else:
            return scalar.data_ptr()

    epi_args = GemmActMixin.EpilogueArguments(
        PostAct_p,
        None,  # act_fn is Constexpr, baked in at compile time
        alpha=scalar_arg(alpha, alpha_mode),
        beta=scalar_arg(beta, beta_mode),
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
    )
    varlen_args = None

    if device_capacity[0] in [10, 11]:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None)
