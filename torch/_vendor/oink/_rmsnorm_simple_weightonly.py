"""Measured same-dtype bf16 RMSNorm forward specializations.

This module implements a narrow fast path for row-major bf16 tensors with a
bf16 1D weight and no residual/bias/rstd outputs.  The math is
``y = x * rsqrt(mean(x * x) + eps) * weight`` with fp32 reduction/multiply and
bf16 output.  The shape table below is deliberately measured and narrow; shapes
not listed fall back to the generic RMSNorm pointer path.
"""
# ruff: noqa: E402  # CuTeDSL cache setup must run before importing cutlass.

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import torch
from torch import Tensor

from ._cutedsl_cache import ensure_versioned_cutedsl_cache_dir

ensure_versioned_cutedsl_cache_dir()

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_ptr

from .fast_launch import (
    StableF32Arg,
    StableI32Arg,
    build_fast_launcher,
)
from .lite_quack import row_reduce_add

_COMPILED_CACHE: dict[tuple[object, int, int, int], object] = {}

_SIMPLE_WEIGHTONLY_SHAPES: dict[tuple[int, int], tuple[int, int]] = {
    # DeepSeek-V4-Flash q_lora same-dtype RMSNorm shape.  Larger M and the
    # kv/per-head N=512 cases are faster through the generic pointer path on SM103.
    (4096, 1536): (96, 96),
    # DeepSeek-V3 hidden-state same-dtype RMSNorm shapes.
    (4096, 6144): (192, 192),
    (4096, 7168): (224, 224),
    (4096, 8192): (256, 256),
    (16384, 7168): (128, 128),
    (16384, 8192): (128, 128),
}


@dataclass(frozen=True, slots=True)
class _SimpleWeightOnlyConfig:
    COPY_BITS = 128

    dtype: cutlass.Numeric
    N: int
    threads_per_row: int
    num_threads: int

    @property
    def vec_size(self) -> int:
        return self.COPY_BITS // self.dtype.width

    @property
    def rows_per_block(self) -> int:
        return self.num_threads // self.threads_per_row

    @property
    def num_vec_blocks(self) -> int:
        return self.N // (self.vec_size * self.threads_per_row)

    @property
    def cols_per_tile(self) -> int:
        return self.vec_size * self.num_vec_blocks * self.threads_per_row

    @property
    def warps_per_row(self) -> int:
        return max(self.threads_per_row // 32, 1)

    @property
    def cache_key(self) -> tuple[object, int, int, int]:
        return (
            self.dtype,
            int(self.N),
            int(self.threads_per_row),
            int(self.num_threads),
        )

    @staticmethod
    def make_tv_layout(
        threads_per_row: int,
        rows_per_block: int,
        vec_size: int,
        num_vec_blocks: int,
    ):
        shape = ((threads_per_row, rows_per_block), (vec_size, num_vec_blocks))
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def smem_bytes(self) -> int:
        return (
            self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            + self.rows_per_block * self.warps_per_row * 4
        )


class _SimpleWeightOnlyRMSNorm:
    def __init__(self, cfg: _SimpleWeightOnlyConfig):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self, x_ptr, w_ptr, o_ptr, M: Int32, eps: Float32, stream: cuda.CUstream
    ):
        cfg = self.cfg
        mX = cute.make_tensor(x_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        mW = cute.make_tensor(w_ptr, cute.make_layout((cfg.N,), stride=(1,)))
        mO = cute.make_tensor(o_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        tv_shape, tv_stride = _SimpleWeightOnlyConfig.make_tv_layout(
            cfg.threads_per_row,
            cfg.rows_per_block,
            cfg.vec_size,
            cfg.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (cfg.rows_per_block, cfg.cols_per_tile)
        self.kernel(mX, mW, mO, eps, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, cfg.rows_per_block), 1, 1],
            block=[cfg.num_threads, 1, 1],
            smem=cfg.smem_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        M = mX.shape[0]
        threads_per_row = tv_layout.shape[0][0]
        warps_per_row = max(threads_per_row // 32, 1)
        rows_per_block = tiler_mn[0]

        smem = utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout(
                (rows_per_block, (warps_per_row, 1)), order=(1, 0)
            ),
            byte_alignment=4,
        )

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gO = cute.local_tile(mO, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        mW2 = cute.make_tensor(
            mW.iterator,
            cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
        )
        gW = cute.local_tile(mW2, tiler_mn, (0, 0))

        copy_atom_load_x = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=_SimpleWeightOnlyConfig.COPY_BITS,
        )
        copy_atom_load_w = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=_SimpleWeightOnlyConfig.COPY_BITS,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO.element_type,
            num_bits_per_copy=_SimpleWeightOnlyConfig.COPY_BITS,
        )

        tiled_copy_x = cute.make_tiled_copy(copy_atom_load_x, tv_layout, tiler_mn)
        tiled_copy_w = cute.make_tiled_copy(copy_atom_load_w, tv_layout, tiler_mn)
        tiled_copy_o = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_x = tiled_copy_x.get_slice(tidx)
        thr_copy_w = tiled_copy_w.get_slice(tidx)
        thr_copy_o = tiled_copy_o.get_slice(tidx)

        tXgX = thr_copy_x.partition_S(gX)
        tXsX = thr_copy_x.partition_D(sX)
        tXgO = thr_copy_o.partition_D(gO)
        tXcX = thr_copy_x.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        tXrO = cute.make_fragment_like(tXgO)
        tWgW = thr_copy_w.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)

        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        if row_in_bounds:
            cute.copy(copy_atom_load_x, tXgX, tXsX)
        cute.arch.cp_async_commit_group()
        cute.copy(copy_atom_load_w, tWgW, tWrW)
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)
        sum_sq = row_reduce_add(
            x * x, threads_per_row, reduction_buffer, None, None, Float32(0.0)
        )
        rstd = cute.math.rsqrt(sum_sq / cfg.N + eps, fastmath=True)
        cute.arch.barrier()

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)
        w = tWrW.load().to(Float32)
        y = x * rstd * w
        tXrO.store(y.to(cfg.dtype))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO)


def _get_simple_weightonly_config(
    x: Tensor,
    weight: Tensor,
    out: Tensor,
) -> _SimpleWeightOnlyConfig | None:
    if (
        x.dtype is not torch.bfloat16
        or weight.dtype is not x.dtype
        or out.dtype is not x.dtype
    ):
        return None
    if not x.is_cuda or not weight.is_cuda or not out.is_cuda:
        return None
    if x.dim() != 2 or out.dim() != 2 or weight.dim() != 1:
        return None
    if x.shape != out.shape or int(weight.shape[0]) != int(x.shape[1]):
        return None
    if (
        x.stride() != (int(x.shape[1]), 1)
        or out.stride() != x.stride()
        or weight.stride() != (1,)
    ):
        return None
    M, N = int(x.shape[0]), int(x.shape[1])
    threads = _SIMPLE_WEIGHTONLY_SHAPES.get((M, N))
    if threads is None:
        return None
    threads_per_row, num_threads = threads
    return _SimpleWeightOnlyConfig(
        dtype=cutlass.BFloat16,
        N=N,
        threads_per_row=threads_per_row,
        num_threads=num_threads,
    )


def _get_compiled(cfg: _SimpleWeightOnlyConfig, stream: cuda.CUstream):
    key = cfg.cache_key
    compiled = _COMPILED_CACHE.get(key)
    if compiled is None:
        kernel = _SimpleWeightOnlyRMSNorm(cfg)
        compiled = cute.compile(
            kernel,
            make_ptr(cfg.dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cfg.dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cfg.dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
            Int32(1),
            Float32(1e-6),
            stream,
        )
        _COMPILED_CACHE[key] = compiled
    return compiled


def _get_fast_launcher(
    *,
    compiled: object,
    cfg: _SimpleWeightOnlyConfig,
    device_index: int,
    stream_handle: int,
    eps: float,
):
    ptr_x = make_ptr(cfg.dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    ptr_w = make_ptr(cfg.dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    ptr_o = make_ptr(cfg.dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    arg_m = StableI32Arg(0)
    arg_eps = StableF32Arg(eps)
    key = (
        "simple_weightonly_fast",
        id(compiled),
        cfg.dtype,
        int(cfg.N),
        int(cfg.threads_per_row),
        int(cfg.num_threads),
        int(device_index),
        int(stream_handle),
    )
    return build_fast_launcher(
        key=key,
        compiled=compiled,
        device_index=device_index,
        stream_handle=stream_handle,
        execution_args_builder=lambda stream: (
            ptr_x,
            ptr_w,
            ptr_o,
            arg_m,
            arg_eps,
            stream,
        ),
        keepalive_items=(ptr_x, ptr_w, ptr_o, arg_m, arg_eps),
        ptr_slots=((ptr_x, "x"), (ptr_w, "weight"), (ptr_o, "out")),
        scalar_slots=((arg_m, "M", -1), (arg_eps, "eps", float("nan"))),
        fallback_launch_builder=lambda stream: (
            lambda **kwargs: compiled(
                make_ptr(
                    cfg.dtype,
                    kwargs["x"].data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    cfg.dtype,
                    kwargs["weight"].data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    cfg.dtype,
                    kwargs["out"].data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                Int32(kwargs["M"]),
                Float32(kwargs["eps"]),
                stream,
            )
        ),
    )


def try_simple_weightonly_rmsnorm_forward(
    x: Tensor,
    weight: Tensor,
    out: Tensor,
    eps: float,
) -> bool:
    cfg = _get_simple_weightonly_config(x, weight, out)
    if cfg is None:
        return False
    device_index = int(x.device.index)
    stream_handle = int(torch.cuda.current_stream(x.device).cuda_stream)
    stream = cuda.CUstream(stream_handle)
    compiled = _get_compiled(cfg, stream)
    launcher = _get_fast_launcher(
        compiled=compiled,
        cfg=cfg,
        device_index=device_index,
        stream_handle=stream_handle,
        eps=float(eps),
    )
    if launcher is not None and int(x.shape[0]) <= 4096:
        launcher.launch(x=x, weight=weight, out=out, M=int(x.shape[0]), eps=float(eps))
        return True
    compiled(
        make_ptr(cfg.dtype, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            cfg.dtype, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(cfg.dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int32(int(x.shape[0])),
        Float32(float(eps)),
        stream,
    )
    return True
