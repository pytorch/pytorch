# Copyright (c) 2026, QuACK team.
# Staged split-K reduction with the full (linear) epilogue:
#   D[m, n, l] = convert(alpha * sum_s Ws[m, n, l*split_k + s] + beta*C + rowvec + colvec (+ D))
# This is the second kernel of the staged split-K GEMM pipeline: the GEMM kernel writes
# RAW per-split f32 accumulator partials (no epilogue math at all) to the workspace, and
# this kernel sums the splits in fixed ascending order and applies the epilogue exactly
# once on the completed sum — mirroring the serial/parallel modes' finalizing split. The
# reduction order is deterministic, so results are bitwise reproducible run to run. The
# epilogue math is shared with the GEMM mixin via gemm_default_epi.apply_linear_epilogue.

import math
from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cute.runtime import make_ptr

from torch import Tensor

import torch._vendor.quack.copy_utils as copy_utils
import torch._vendor.quack.utils as utils
from torch._vendor.quack.compile_utils import make_fake_tensor as fake_tensor
from torch._vendor.quack.cache import jit_cache
from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map
from torch._vendor.quack.dsl import cute_op

# The shared linear-epilogue math (also used by the GEMM mixins' epi_visit_subtile), so
# the reduction kernel and the in-GEMM finalizer apply bitwise-identical epilogues.
from torch._vendor.quack.gemm_default_epi import apply_linear_epilogue


class SplitKReduce:
    """Sum split_k f32 partial slices, apply the linear epilogue, store in out_dtype.

    mWs has shape (M, N, L * split_k) with the combined (l * split_k + s) batch mode the
    staged split-K GEMM stored to; mOut has shape (M, N, L); mC (M, N, L), mRowVec (N, L)
    and mColVec (M, L) are the optional epilogue operands. All matrix views must be
    M-row/N-contiguous (the host wrapper transposes M/N views — and swaps the vector
    roles — for m-major outputs). With add_to_output, the prior value of D is added once
    (after the epilogue terms, unscaled).

    vec_elems must divide N: predicate_k gates whole vectors by their first element's
    column, so a vector must never straddle the N boundary (same contract as
    rms_final_reduce, where the host derives vec_elems = gcd(N, max_vec)).
    """

    num_threads = 128
    threads_per_row = 16

    def __init__(
        self,
        out_dtype: Type[cutlass.Numeric],
        split_k: int,
        add_to_output: bool,
        vec_elems: int,
        c_dtype: Optional[Type[cutlass.Numeric]] = None,
        rowvec_dtype: Optional[Type[cutlass.Numeric]] = None,
        colvec_dtype: Optional[Type[cutlass.Numeric]] = None,
    ):
        assert split_k >= 1
        assert vec_elems in (1, 2, 4)
        self.out_dtype = out_dtype
        self.split_k = split_k
        self.add_to_output = add_to_output
        self.vec_elems = vec_elems
        self.c_dtype = c_dtype
        self.rowvec_dtype = rowvec_dtype
        self.colvec_dtype = colvec_dtype
        self.tiler_mn = (
            self.num_threads // self.threads_per_row,
            self.threads_per_row * self.vec_elems,
        )

    @cute.jit
    def __call__(
        self,
        mWs: cute.Tensor,
        mOut: cute.Tensor,
        mC: Optional[cute.Tensor],
        mRowVec: Optional[cute.Tensor],
        mColVec: Optional[cute.Tensor],
        alpha: Optional[Float32 | cute.Pointer],
        beta: Optional[Float32 | cute.Pointer],
        stream: cuda.CUstream,
    ):
        assert mWs.element_type == Float32
        assert mOut.element_type == self.out_dtype
        tiled_copy_ws = copy_utils.tiled_copy_2d(
            Float32, self.threads_per_row, self.num_threads, num_copy_elems=self.vec_elems
        )
        tiled_copy_out = copy_utils.tiled_copy_2d(
            self.out_dtype, self.threads_per_row, self.num_threads, num_copy_elems=self.vec_elems
        )
        tiled_copy_c = None
        if const_expr(mC is not None):
            tiled_copy_c = copy_utils.tiled_copy_2d(
                self.c_dtype, self.threads_per_row, self.num_threads, num_copy_elems=self.vec_elems
            )
        tiled_copy_rv = None
        if const_expr(mRowVec is not None):
            tiled_copy_rv = copy_utils.tiled_copy_2d(
                self.rowvec_dtype,
                self.threads_per_row,
                self.num_threads,
                num_copy_elems=self.vec_elems,
            )
        self.kernel(
            mWs,
            mOut,
            mC,
            mRowVec,
            mColVec,
            alpha,
            beta,
            tiled_copy_ws,
            tiled_copy_out,
            tiled_copy_c,
            tiled_copy_rv,
        ).launch(
            grid=[
                cute.ceil_div(mOut.shape[0], self.tiler_mn[0]),
                cute.ceil_div(mOut.shape[1], self.tiler_mn[1]),
                mOut.shape[2],
            ],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mWs: cute.Tensor,
        mOut: cute.Tensor,
        mC: Optional[cute.Tensor],
        mRowVec: Optional[cute.Tensor],
        mColVec: Optional[cute.Tensor],
        alpha: Optional[Float32 | cute.Pointer],
        beta: Optional[Float32 | cute.Pointer],
        tiled_copy_ws: cute.TiledCopy,
        tiled_copy_out: cute.TiledCopy,
        tiled_copy_c: Optional[cute.TiledCopy],
        tiled_copy_rv: Optional[cute.TiledCopy],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        shape_mn = (mOut.shape[0], mOut.shape[1])

        thr_copy_ws = tiled_copy_ws.get_slice(tidx)
        thr_copy_out = tiled_copy_out.get_slice(tidx)

        idMN = cute.make_identity_tensor(shape_mn)
        cMN = cute.local_tile(idMN, self.tiler_mn, (bidx, bidy))
        # Each thread owns one row of the tile and vec_elems consecutive columns;
        # predicate the column dim, guard the row dim with an if (same scheme as
        # rms_final_reduce).
        tXcX = thr_copy_ws.partition_S(cMN)[(0, None), None, None]
        row = tXcX[0][0]
        tXpX = copy_utils.predicate_k(thr_copy_ws.partition_S(cMN), limit=shape_mn[1])
        row_ok = row < shape_mn[0]

        gOut = cute.local_tile(mOut[None, None, bidz], self.tiler_mn, (bidx, bidy))
        tXgOut = thr_copy_out.partition_S(gOut)

        # Load all split partials (and the epilogue operands), then sum in fixed
        # ascending split order for a deterministic reduction.
        tXgWs = [
            thr_copy_ws.partition_S(
                cute.local_tile(
                    mWs[None, None, bidz * self.split_k + s], self.tiler_mn, (bidx, bidy)
                )
            )
            for s in range(self.split_k)
        ]
        frags = [cute.make_rmem_tensor_like(tXgW) for tXgW in tXgWs]
        tXrC = None
        if const_expr(mC is not None):
            gC = cute.local_tile(mC[None, None, bidz], self.tiler_mn, (bidx, bidy))
            tXgC = tiled_copy_c.get_slice(tidx).partition_S(gC)
            tXrC = cute.make_rmem_tensor_like(tXgC)
        tXrRowVec = None
        if const_expr(mRowVec is not None):
            # Broadcast (M, N) view of the (N, L) row vector: row stride 0, col stride 1
            # (each thread loads its own duplicate — vectorized along the contiguous N).
            mRV_mn = cute.make_tensor(
                utils.elem_pointer(mRowVec, (0, bidz)),
                cute.make_layout(shape_mn, stride=(0, 1)),
            )
            gRV = cute.local_tile(mRV_mn, self.tiler_mn, (bidx, bidy))
            tXgRV = tiled_copy_rv.get_slice(tidx).partition_S(gRV)
            tXrRV_in = cute.make_rmem_tensor_like(tXgRV)
        tXrOut_in = None
        if const_expr(self.add_to_output):
            tXrOut_in = cute.make_rmem_tensor_like(tXgOut)
        if row_ok:
            for s in cutlass.range_constexpr(self.split_k):
                copy_utils.copy(tXgWs[s], frags[s], pred=tXpX)
            if const_expr(mC is not None):
                copy_utils.copy(tXgC, tXrC, pred=tXpX)
            if const_expr(mRowVec is not None):
                copy_utils.copy(tXgRV, tXrRV_in, pred=tXpX)
            if const_expr(self.add_to_output):
                copy_utils.copy(tXgOut, tXrOut_in, pred=tXpX)
        acc = frags[0].load().to(Float32)
        for s in cutlass.range_constexpr(1, self.split_k):
            acc = acc + frags[s].load().to(Float32)

        # The epilogue, exactly once, on the completed f32 sum — the same shared math as
        # the GEMM mixin (alpha * acc + beta * C + rowvec + colvec).
        tXrD = cute.make_rmem_tensor_like(frags[0])
        tXrD.store(acc)
        if const_expr(mRowVec is not None):
            tXrRowVec = cute.make_rmem_tensor_like(frags[0])
            tXrRowVec.store(tXrRV_in.load().to(Float32))
        tXrColVec = None
        if const_expr(mColVec is not None):
            cv = Float32(0.0)
            if row_ok:
                cv = Float32(mColVec[row, bidz])
            tXrColVec = cute.make_rmem_tensor_like(frags[0])
            for i in cutlass.range_constexpr(cute.size(tXrColVec)):
                tXrColVec[i] = cv
        apply_linear_epilogue(tXrD, tXrC, alpha, beta, tXrRowVec, tXrColVec)
        acc = tXrD.load()
        if const_expr(self.add_to_output):
            acc = acc + tXrOut_in.load().to(Float32)

        tXrOut = cute.make_rmem_tensor_like(tXgOut)
        tXrOut.store(acc.to(self.out_dtype))
        if row_ok:
            copy_utils.copy(tXrOut, tXgOut, pred=tXpX)


@jit_cache
def _compile_split_k_reduce(
    out_dtype,
    split_k,
    add_to_output,
    vec_elems,
    alpha_mode,
    beta_mode,
    c_dtype,
    rowvec_dtype,
    colvec_dtype,
):
    m_sym, n_sym, l_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ls_sym = cute.sym_int()
    # Workspace rows are padded to a multiple of 4 f32 elements (16 bytes) by the host.
    ws_cute = fake_tensor(Float32, (m_sym, n_sym, ls_sym), divisibility=4, leading_dim=1)
    out_cute = fake_tensor(
        out_dtype, (m_sym, n_sym, l_sym), divisibility=128 // out_dtype.width, leading_dim=1
    )
    # The epilogue operands only need vec_elems-granular alignment (we vectorize by at
    # most vec_elems); C/rowvec rows can be unpadded user tensors.
    c_cute = (
        fake_tensor(c_dtype, (m_sym, n_sym, l_sym), divisibility=vec_elems, leading_dim=1)
        if c_dtype is not None
        else None
    )
    rowvec_cute = (
        fake_tensor(rowvec_dtype, (n_sym, l_sym), divisibility=vec_elems, leading_dim=0)
        if rowvec_dtype is not None
        else None
    )
    colvec_cute = (
        fake_tensor(colvec_dtype, (m_sym, l_sym), divisibility=1, leading_dim=0)
        if colvec_dtype is not None
        else None
    )

    def fake_scalar(mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Float32(1.0)
        else:
            return make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=4)

    return cute.compile(
        SplitKReduce(
            out_dtype, split_k, add_to_output, vec_elems, c_dtype, rowvec_dtype, colvec_dtype
        ),
        ws_cute,
        out_cute,
        c_cute,
        rowvec_cute,
        colvec_cute,
        fake_scalar(alpha_mode),
        fake_scalar(beta_mode),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cute_op(
    "torch_vendor_quack::split_k_reduce_out",
    mutates_args=("out",),
    device_types="cuda",
)
def _split_k_reduce_out(
    ws: Tensor,
    out: Tensor,
    split_k: int,
    add_to_output: bool,
    alpha: float,
    alpha_t: Optional[Tensor],
    beta: float,
    beta_t: Optional[Tensor],
    C: Optional[Tensor],
    rowvec_bias: Optional[Tensor],
    colvec_bias: Optional[Tensor],
) -> None:
    """out[l] = convert(alpha * sum_s ws[l*split_k+s] + beta*C + rowvec + colvec (+ out))."""
    out_dtype = torch2cute_dtype_map[out.dtype]
    # Vectors must not straddle the N boundary (predicate granularity is per vector); a
    # non-N-contiguous C cannot be vector-loaded with the shared column predicate.
    vec_elems = math.gcd(out.shape[-1], 4)
    if C is not None and C.stride(-1) != 1:
        vec_elems = 1
    alpha_mode = 2 if alpha_t is not None else (1 if alpha != 1.0 else 0)
    beta_mode = 2 if beta_t is not None else (1 if beta != 1.0 else 0)
    compiled_fn = _compile_split_k_reduce(
        out_dtype,
        split_k,
        add_to_output,
        vec_elems,
        alpha_mode,
        beta_mode,
        torch2cute_dtype_map[C.dtype] if C is not None else None,
        torch2cute_dtype_map[rowvec_bias.dtype] if rowvec_bias is not None else None,
        torch2cute_dtype_map[colvec_bias.dtype] if colvec_bias is not None else None,
    )

    def scalar_arg(mode, imm, tens):
        if mode == 0:
            return None
        elif mode == 1:
            return Float32(imm)
        else:
            return tens.data_ptr()

    # Kernel-facing layouts are (M, N, L) / (N, L) / (M, L): permute the torch views.
    compiled_fn(
        ws.permute(1, 2, 0),
        out.permute(1, 2, 0),
        C.permute(1, 2, 0) if C is not None else None,
        rowvec_bias.permute(1, 0) if rowvec_bias is not None else None,
        colvec_bias.permute(1, 0) if colvec_bias is not None else None,
        scalar_arg(alpha_mode, alpha, alpha_t),
        scalar_arg(beta_mode, beta, beta_t),
    )


def split_k_reduce(
    ws: Tensor,  # (L * split_k, M, N) f32 raw partials, N-contiguous
    out: Tensor,  # (L, M, N), N-contiguous
    split_k: int,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    C: Optional[Tensor] = None,  # (L, M, N)
    rowvec_bias: Optional[Tensor] = None,  # (L, N)
    colvec_bias: Optional[Tensor] = None,  # (L, M)
    add_to_output: bool = False,
) -> None:
    """Deterministically reduce staged split-K partials and apply the linear epilogue."""
    assert ws.ndim == 3 and out.ndim == 3
    assert ws.shape[0] == out.shape[0] * split_k
    assert ws.shape[1:] == out.shape[1:]
    alpha_t = alpha if isinstance(alpha, Tensor) else None
    beta_t = beta if isinstance(beta, Tensor) else None
    # Dispatch through the cute_op so torch.compile sees a registered custom op (the
    # fake is a pure no-op; compilation is owned by jit_cache + the async compile pool).
    _split_k_reduce_out(
        ws,
        out,
        split_k,
        add_to_output,
        alpha if alpha_t is None else 1.0,
        alpha_t,
        beta if beta_t is None else 1.0,
        beta_t,
        C,
        rowvec_bias,
        colvec_bias,
    )
