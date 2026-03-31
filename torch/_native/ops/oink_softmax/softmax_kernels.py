# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Softmax forward + backward kernels for SM100 (Blackwell) in CuteDSL.

This module implements numerically stable softmax over the last dimension of
2D tensors (M, N) and its backward pass, targeting SM100 with Quack-style
tiling, cp.async pipelines, and cluster reductions, but without depending on
the `quack` package at runtime.

The kernels are self-contained and use only local helpers in
`kernelagent_oink.blackwell.lite_quack` plus CuTeDSL/CUTLASS.
"""

from __future__ import annotations

import importlib.metadata
import os
import re

import cuda.bindings.driver as cuda  # provided by NVIDIA cuda-python

import torch
from torch import Tensor


# CuTeDSL caches generated MLIR into a tempdir under a global default
# (`/tmp/$USER/cutlass_python_cache`). The cache bytecode format can differ across
# `nvidia-cutlass-dsl` versions, and cross-version cache sharing causes noisy
# warnings (and disables cache reuse).
if "CUTE_DSL_CACHE_DIR" not in os.environ:
    try:
        _dsl_ver = importlib.metadata.version("nvidia-cutlass-dsl")
    except Exception:
        _dsl_ver = "unknown"
    _dsl_ver = re.sub(r"[^0-9A-Za-z]+", "_", _dsl_ver)
    _user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    _tmp = os.environ.get("TMPDIR") or "/tmp"
    os.environ["CUTE_DSL_CACHE_DIR"] = os.path.join(
        _tmp, _user, f"cutlass_python_cache_{_dsl_ver}"
    )

try:
    import cutlass  # type: ignore[import-untyped]  # noqa: F401
except Exception as e:
    raise ImportError(
        "kernelagent_oink.blackwell.softmax requires CuTeDSL's Python package "
        "(`cutlass`, typically provided by `nvidia-cutlass-dsl`)."
    ) from e

import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32
from cutlass.cute import runtime as rt
from cutlass.cute.runtime import from_dlpack

from .._oink_utils.fast_launch import (
    disable_fast_launch,
    fast_launch_enabled,
    set_runtime_ptr,
    StableI32Arg,
    tls_cache as _tls_fast_launch_cache,
)
from .._oink_utils.lite_quack import (
    _KERNEL_ACCEPTS_LAYOUT_ARGS,
    fill_oob,
    online_softmax_reduce,
    predicate_k,
    ReductionBase,
    row_reduce,
    TORCH2CUTE_DTYPE,
)


_FWD_COMPILE_CACHE: dict[tuple[type[cutlass.Numeric], int], object] = {}
_BWD_COMPILE_CACHE: dict[tuple[type[cutlass.Numeric], int], object] = {}
_PTR_FWD_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_PTR_BWD_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_PTR_FWDBWD_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


class _PtrSoftmaxFastLaunch:
    def __init__(
        self,
        *,
        compiled: object,
        executor: object,
        capi_func: object,
        ptr_a: object,
        ptr_b: object,
        ptr_c: object | None,
        arg_m: StableI32Arg,
        arg_ld: StableI32Arg,
        stream: cuda.CUstream,
        assumed_align: int,
        packed_args: object,
        keepalive: tuple[object, ...],
    ):
        self._compiled = compiled
        self._executor = executor
        self._capi_func = capi_func
        self._ptr_a = ptr_a
        self._ptr_b = ptr_b
        self._ptr_c = ptr_c
        self._arg_m = arg_m
        self._arg_ld = arg_ld
        self._stream = stream
        self._assumed_align = int(assumed_align)
        self._packed_args = packed_args
        self._keepalive = keepalive

        self._use_fast_launch = True
        self._cuda_result = getattr(executor, "cuda_result", None)

        self._last_a_ptr = -1
        self._last_b_ptr = -1
        self._last_c_ptr = -1
        self._last_m = -1
        self._last_ld = -1

    def launch(
        self,
        *,
        a_ptr: int,
        b_ptr: int,
        c_ptr: int | None,
        M: int,
        ld: int,
        stream_handle: int,
        dtype: type[cutlass.Numeric],
    ) -> None:
        if not fast_launch_enabled() or not self._use_fast_launch:
            self._fallback_launch(
                a_ptr=a_ptr,
                b_ptr=b_ptr,
                c_ptr=c_ptr,
                M=M,
                ld=ld,
                stream_handle=stream_handle,
                dtype=dtype,
            )
            return

        if a_ptr != self._last_a_ptr:
            try:
                set_runtime_ptr(self._ptr_a, a_ptr)
                self._last_a_ptr = a_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    a_ptr=a_ptr,
                    b_ptr=b_ptr,
                    c_ptr=c_ptr,
                    M=M,
                    ld=ld,
                    stream_handle=stream_handle,
                    dtype=dtype,
                )
                return

        if b_ptr != self._last_b_ptr:
            try:
                set_runtime_ptr(self._ptr_b, b_ptr)
                self._last_b_ptr = b_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    a_ptr=a_ptr,
                    b_ptr=b_ptr,
                    c_ptr=c_ptr,
                    M=M,
                    ld=ld,
                    stream_handle=stream_handle,
                    dtype=dtype,
                )
                return

        if self._ptr_c is not None and c_ptr is not None:
            if c_ptr != self._last_c_ptr:
                try:
                    set_runtime_ptr(self._ptr_c, c_ptr)
                    self._last_c_ptr = c_ptr
                except AttributeError:
                    self._disable_fast_launch()
                    self._fallback_launch(
                        a_ptr=a_ptr,
                        b_ptr=b_ptr,
                        c_ptr=c_ptr,
                        M=M,
                        ld=ld,
                        stream_handle=stream_handle,
                        dtype=dtype,
                    )
                    return

        if M != self._last_m:
            self._arg_m.set(M)
            self._last_m = M
        if ld != self._last_ld:
            self._arg_ld.set(ld)
            self._last_ld = ld

        if self._cuda_result is not None:
            self._cuda_result.value = 0
        ret = self._capi_func(self._packed_args)  # type: ignore[misc]
        if ret != 0:
            raise RuntimeError(f"CuTeDSL capi_func returned non-zero: {ret}")
        if self._cuda_result is not None:
            err = int(self._cuda_result.value)
            if err != 0:
                raise RuntimeError(f"CuTeDSL kernel launch failed (cuda_result={err})")

    def _disable_fast_launch(self) -> None:
        self._use_fast_launch = False
        disable_fast_launch()

    def _fallback_launch(
        self,
        *,
        a_ptr: int,
        b_ptr: int,
        c_ptr: int | None,
        M: int,
        ld: int,
        stream_handle: int,
        dtype: type[cutlass.Numeric],
    ) -> None:
        stream = cuda.CUstream(int(stream_handle))
        ptr_a = rt.make_ptr(
            dtype,
            a_ptr,
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align,
        )
        ptr_b = rt.make_ptr(
            dtype,
            b_ptr,
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align,
        )
        if self._ptr_c is not None and c_ptr is not None:
            ptr_c = rt.make_ptr(
                dtype,
                c_ptr,
                mem_space=rt.AddressSpace.gmem,
                assumed_align=self._assumed_align,
            )
            self._compiled(ptr_a, ptr_b, ptr_c, Int32(int(M)), Int32(int(ld)), stream)
        else:
            self._compiled(ptr_a, ptr_b, Int32(int(M)), Int32(int(ld)), stream)


def _get_fast_ptr_softmax_launcher(
    *,
    compiled: object,
    dtype: type[cutlass.Numeric],
    N: int,
    device_index: int,
    stream_handle: int,
    assumed_align: int,
    is_bwd: bool,
) -> _PtrSoftmaxFastLaunch | None:
    if not fast_launch_enabled():
        return None
    key = (
        "ptr_fast_bwd" if is_bwd else "ptr_fast_fwd",
        id(compiled),
        int(N),
        dtype,
        int(device_index),
        int(stream_handle),
        int(assumed_align),
    )
    cache = _tls_fast_launch_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    assumed_align = int(assumed_align)
    ptr_a = rt.make_ptr(
        dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align
    )
    ptr_b = rt.make_ptr(
        dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align
    )
    ptr_c = (
        rt.make_ptr(
            dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align
        )
        if is_bwd
        else None
    )

    arg_m = StableI32Arg(0)
    arg_ld = StableI32Arg(N)
    stream = cuda.CUstream(int(stream_handle))
    executor = compiled.to(device_index)  # type: ignore[attr-defined]
    try:
        if ptr_c is not None:
            exe_args, adapted_args = executor.generate_execution_args(
                ptr_a,
                ptr_b,
                ptr_c,
                arg_m,
                arg_ld,
                stream,
            )
        else:
            exe_args, adapted_args = executor.generate_execution_args(
                ptr_a,
                ptr_b,
                arg_m,
                arg_ld,
                stream,
            )
        packed_args = executor._get_invoke_packed_args(list(exe_args))  # type: ignore[attr-defined]
        capi_func = compiled.capi_func  # type: ignore[attr-defined]
    except AttributeError:
        disable_fast_launch()
        return None

    keepalive: tuple[object, ...] = (
        executor,
        ptr_a,
        ptr_b,
        ptr_c,
        arg_m,
        arg_ld,
        stream,
        *adapted_args,
    )
    launcher = _PtrSoftmaxFastLaunch(
        compiled=compiled,
        executor=executor,
        capi_func=capi_func,
        ptr_a=ptr_a,
        ptr_b=ptr_b,
        ptr_c=ptr_c,
        arg_m=arg_m,
        arg_ld=arg_ld,
        stream=stream,
        assumed_align=assumed_align,
        packed_args=packed_args,
        keepalive=keepalive,
    )
    cache[key] = launcher
    return launcher


class SoftmaxFwdSM100(ReductionBase):
    def __init__(self, dtype: type[cutlass.Numeric], N: int):
        # One-stage online reduction: pack (max, sum_exp) into Int64 reduction buffer.
        super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Int64)

    def _get_num_threads(self) -> int:
        # SM100 tuning note:
        # For N=4096, we use 32 threads per row (1 warp) and run 1 row per CTA
        # (32 threads total). This keeps the reduction fully warp-local and
        # improves throughput on this GB200 versus Quack's default 2-rows-per-CTA
        # schedule with 64 threads per row (4 warps total).
        if self.N == 4096:
            return 32
        return super()._get_num_threads()

    def _calculate_threads_per_row(self) -> int:
        # Match Quack's bucketed policy for Softmax.
        N = self.N
        if N == 4096:
            return 32
        if N == 6144:
            return 128
        if N <= 64:
            return 8
        if N <= 128:
            return 16
        if N <= 3072:
            return 32
        if N <= 6144:
            return 64
        if N <= 16384:
            return 128
        return 256

    def _set_cluster_n(self) -> None:
        # Quack-style growth of cluster_n with N and dtype.
        N = self.N
        if const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        else:
            cluster_n = (
                1
                if N <= 32 * 1024
                else (
                    2
                    if N <= 64 * 1024
                    else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream) -> None:
        assert mX.element_type == self.dtype  # noqa: S101
        assert mO.element_type == self.dtype  # noqa: S101
        # Use the generic ReductionBase tiling with 128-bit vectorization.
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        kernel = (
            self.kernel(mX, mO, tv_layout, tiler_mn)
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(mX, mO)
        )
        kernel.launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_x: cute.Pointer,
        ptr_out: cute.Pointer,
        M: Int32,
        ld: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Pointer-based entrypoint that bypasses DLPack conversions.

        Reconstructs cute.Tensor views from raw pointers + explicit layouts
        inside the JIT graph, matching the existing SM100 schedule.
        """
        # Mirror Quack/LayerNorm contracts: assume 16B alignment and an LD that
        # preserves 128-bit vectorized copies for every row start.
        ld_assumed = cute.assume(ld, divby=128 // self.dtype.width)
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        mX = cute.make_tensor(ptr_x, layout_mn)
        mO = cute.make_tensor(ptr_out, layout_mn)
        self.__call__(mX, mO, stream)

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        # Quack-style CTA tiling.
        gX, gO, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, mO, idX)
        ]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        # Copy atoms for gmem <-> smem and smem <-> gmem.
        # Use 128-bit cp.async for global->shared and 128-bit vectorized stores.
        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gO.element_type,
            num_bits_per_copy=128,
        )

        num_copy_elems = (
            tv_layout.shape[1]
            if const_expr(cute.rank(tv_layout.shape[1]) == 1)
            else tv_layout.shape[1][0]
        )
        threads_per_row = (
            tv_layout.shape[0]
            if const_expr(cute.rank(tv_layout.shape[0]) == 1)
            else tv_layout.shape[0][0]
        )
        thr_layout = cute.make_ordered_layout(
            (tiler_mn[0], threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, num_copy_elems))
        thr_copy_load = cute.make_tiled_copy_tv(
            copy_atom_load, thr_layout, val_layout
        ).get_slice(tidx)
        thr_copy_store = cute.make_tiled_copy_tv(
            copy_atom_store, thr_layout, val_layout
        ).get_slice(tidx)

        tXgX = thr_copy_load.partition_S(gX)
        tXsX = thr_copy_load.partition_D(sX)
        tXgO = thr_copy_store.partition_D(gO)
        tXcX = thr_copy_load.partition_S(cX)[(0, None), None, None]

        # Register fragments.
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        # Predicate and cp.async pipeline for potential tail tiles.
        is_even_N = const_expr(self.N == tiler_mn[1] * self.cluster_n)
        tXpX = (
            predicate_k(thr_copy_load.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        if const_expr(not is_even_N):
            fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # Online softmax reduction: compute max and sum_exp in a single pass, with
        # optional cluster-wide aggregation via an Int64 reduction buffer.
        max_x, denom, exp_x = online_softmax_reduce(
            x,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            phase=None,
            return_exp_x=True,
        )

        y = exp_x * cute.arch.rcp_approx(denom)
        tXrO.store(y.to(tXrO.element_type))

        tOpO = (
            predicate_k(thr_copy_store.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tOpO)

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mO: cute.Tensor,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ) -> None:
            self._kernel_impl(mX, mO, tv_layout, tiler_mn)
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mO: cute.Tensor,
        ) -> None:
            tiler_mn, tv_layout = self._get_tv_layout()
            self._kernel_impl(mX, mO, tv_layout, tiler_mn)


class SoftmaxBwdSM100(ReductionBase):
    def __init__(self, dtype: type[cutlass.Numeric], N: int):
        # One stage for dot(dy, y) per row.
        super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Float32)

    def _calculate_threads_per_row(self) -> int:
        # Match Quack backward softmax buckets.
        N = self.N
        if N in (4096, 6144):
            return 128
        if N <= 64:
            return 8
        if N <= 128:
            return 16
        if N <= 3072:
            return 32
        if N <= 6144:
            return 64
        if N <= 8192:
            return 128
        return 256

    def _set_cluster_n(self) -> None:
        N = self.N
        if const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        else:
            cluster_n = (
                1
                if N <= 32 * 1024
                else (
                    2
                    if N <= 64 * 1024
                    else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    def _get_num_threads(self) -> int:
        # Slightly more aggressive threading for large N than the base class.
        return 128 if self.N <= 8192 else 256

    def _smem_size_in_bytes(self, tiler_mn, num_warps: int) -> int:
        # Store both y and dy tiles plus reduction buffers and mbarriers.
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        assert mdY.element_type == self.dtype  # noqa: S101
        assert mY.element_type == self.dtype  # noqa: S101
        assert mdX.element_type == self.dtype  # noqa: S101
        # Use the generic ReductionBase tiling with 128-bit vectorization.
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        kernel = (
            self.kernel(mdY, mY, mdX, tv_layout, tiler_mn)
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(mdY, mY, mdX)
        )
        kernel.launch(
            grid=[cute.ceil_div(mdY.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_dy: cute.Pointer,
        ptr_y: cute.Pointer,
        ptr_dx: cute.Pointer,
        M: Int32,
        ld: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Pointer-based entrypoint that bypasses DLPack conversions."""
        ld_assumed = cute.assume(ld, divby=128 // self.dtype.width)
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        mdY = cute.make_tensor(ptr_dy, layout_mn)
        mY = cute.make_tensor(ptr_y, layout_mn)
        mdX = cute.make_tensor(ptr_dx, layout_mn)
        self.__call__(mdY, mY, mdX, stream)

    @cute.jit
    def _kernel_impl(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        shape = mdY.shape
        idX = cute.make_identity_tensor(shape)

        gdY, gY, gdX, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y))
            for mT in (mdY, mY, mdX, idX)
        ]

        smem = cutlass.utils.SmemAllocator()
        sdY = smem.allocate_tensor(
            mdY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        sY = smem.allocate_tensor(
            mY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mdY.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gdX.element_type,
            num_bits_per_copy=128,
        )

        num_copy_elems = (
            tv_layout.shape[1]
            if const_expr(cute.rank(tv_layout.shape[1]) == 1)
            else tv_layout.shape[1][0]
        )
        threads_per_row = (
            tv_layout.shape[0]
            if const_expr(cute.rank(tv_layout.shape[0]) == 1)
            else tv_layout.shape[0][0]
        )
        thr_layout = cute.make_ordered_layout(
            (tiler_mn[0], threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, num_copy_elems))
        thr_copy_load = cute.make_tiled_copy_tv(
            copy_atom_load, thr_layout, val_layout
        ).get_slice(tidx)
        thr_copy_store = cute.make_tiled_copy_tv(
            copy_atom_store, thr_layout, val_layout
        ).get_slice(tidx)

        tdYgdY = thr_copy_load.partition_S(gdY)
        tdYsdY = thr_copy_load.partition_D(sdY)
        tYgY = thr_copy_load.partition_S(gY)
        tYsY = thr_copy_load.partition_D(sY)
        tdXgdX = thr_copy_store.partition_D(gdX)
        tXcX = thr_copy_load.partition_S(cX)[(0, None), None, None]

        tdYrdY, tYrY, tdXrdX = [
            cute.make_fragment_like(thr) for thr in (tdYgdY, tYgY, tdXgdX)
        ]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(self.N == tiler_mn[1] * self.cluster_n)
        tdYpdY = (
            predicate_k(thr_copy_load.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tdYgdY, tdYsdY, pred=tdYpdY)
            cute.copy(copy_atom_load, tYgY, tYsY, pred=tdYpdY)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tdYsdY, tdYrdY)
        cute.autovec_copy(tYsY, tYrY)
        dy = tdYrdY.load().to(Float32)
        y = tYrY.load().to(Float32)
        dot = row_reduce(
            dy * y,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        dx = y * (dy - dot)
        tdXrdX.store(dx.to(tdXrdX.element_type))

        tdXpdX = (
            predicate_k(thr_copy_store.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tdXrdX, tdXgdX, pred=tdXpdX)

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mdY: cute.Tensor,
            mY: cute.Tensor,
            mdX: cute.Tensor,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ) -> None:
            self._kernel_impl(mdY, mY, mdX, tv_layout, tiler_mn)
    else:

        @cute.kernel
        def kernel(
            self,
            mdY: cute.Tensor,
            mY: cute.Tensor,
            mdX: cute.Tensor,
        ) -> None:
            tiler_mn, tv_layout = self._get_tv_layout()
            self._kernel_impl(mdY, mY, mdX, tv_layout, tiler_mn)


class SoftmaxFwdBwdSM100(ReductionBase):
    """Fused softmax forward+backward producing dx from (x, dy).

    Computes:
      y = softmax(x)
      dot = sum(dy * y)
      dx = y * (dy - dot)

    This avoids materializing the intermediate `y` in global memory, which is
    the dominant overhead in a naive `softmax_backward(dy, softmax_forward(x))`
    composition.
    """

    def __init__(self, dtype: type[cutlass.Numeric], N: int):
        # Online softmax reduction uses an Int64 reduction buffer packing
        # (max, sum_exp) pairs. We allocate a separate Float32 reduction buffer
        # for dot(dy, y).
        super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Int64)

    def _calculate_threads_per_row(self) -> int:
        # Favor the backward bucket policy (better for the dot reduction).
        N = self.N
        if N in (4096, 6144):
            return 128
        if N <= 64:
            return 8
        if N <= 128:
            return 16
        if N <= 3072:
            return 32
        if N <= 6144:
            return 64
        if N <= 8192:
            return 128
        return 256

    def _set_cluster_n(self) -> None:
        # Quack-style growth of cluster_n with N and dtype.
        N = self.N
        if const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        else:
            cluster_n = (
                1
                if N <= 32 * 1024
                else (
                    2
                    if N <= 64 * 1024
                    else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    def _get_num_threads(self) -> int:
        # Keep in sync with _calculate_threads_per_row.
        return 128 if self.N <= 8192 else 256

    def _smem_size_in_bytes(self, tiler_mn, num_warps: int) -> int:
        # Allocation order:
        #   1) sX (16B aligned)
        #   2) sdY (16B aligned)
        #   3) reduction_buffer_stats (8B aligned)
        #   4) reduction_buffer_dot (8B aligned)
        #   5) optional mbarrier array (8B aligned)
        def _align_up(x: int, align: int) -> int:
            return ((x + align - 1) // align) * align

        tile_bytes = int(cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)))
        reduction_stats_bytes = int(
            num_warps * self.cluster_n * (cutlass.Int64.width // 8)
        )
        reduction_dot_bytes = int(
            num_warps * self.cluster_n * (cutlass.Float32.width // 8)
        )
        mbar_bytes = (
            int(2 * (cutlass.Int64.width // 8)) if const_expr(self.cluster_n > 1) else 0
        )

        offset = _align_up(tile_bytes, 16)
        offset = _align_up(offset, 16) + tile_bytes
        offset = _align_up(offset, 8) + reduction_stats_bytes
        offset = _align_up(offset, 8) + reduction_dot_bytes
        offset = _align_up(offset, 8) + mbar_bytes
        return int(offset)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mdY: cute.Tensor,
        mdX: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        assert mX.element_type == self.dtype  # noqa: S101
        assert mdY.element_type == self.dtype  # noqa: S101
        assert mdX.element_type == self.dtype  # noqa: S101
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        kernel = (
            self.kernel(mX, mdY, mdX, tv_layout, tiler_mn)
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(mX, mdY, mdX)
        )
        kernel.launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_x: cute.Pointer,
        ptr_dy: cute.Pointer,
        ptr_dx: cute.Pointer,
        M: Int32,
        ld: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Pointer-based entrypoint that bypasses DLPack conversions."""
        ld_assumed = cute.assume(ld, divby=128 // self.dtype.width)
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        mX = cute.make_tensor(ptr_x, layout_mn)
        mdY = cute.make_tensor(ptr_dy, layout_mn)
        mdX = cute.make_tensor(ptr_dx, layout_mn)
        self.__call__(mX, mdY, mdX, stream)

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,
        mdY: cute.Tensor,
        mdX: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        gX, gdY, gdX, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y))
            for mT in (mX, mdY, mdX, idX)
        ]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        sdY = smem.allocate_tensor(
            mdY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        reduction_layout = self._get_reduction_buffer_layout(tv_layout, self.cluster_n)
        reduction_buffer_stats = smem.allocate_tensor(
            cutlass.Int64, reduction_layout, byte_alignment=8
        )
        reduction_buffer_dot = smem.allocate_tensor(
            cutlass.Float32, reduction_layout, byte_alignment=8
        )

        if const_expr(self.cluster_n > 1):
            mbar_ptr_base = smem.allocate_array(cutlass.Int64, num_elems=2)
            mbar_ptr_stats = mbar_ptr_base
            mbar_ptr_dot = mbar_ptr_base + Int32(1)
        else:
            mbar_ptr_stats = None
            mbar_ptr_dot = None

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gdX.element_type,
            num_bits_per_copy=128,
        )

        num_copy_elems = (
            tv_layout.shape[1]
            if const_expr(cute.rank(tv_layout.shape[1]) == 1)
            else tv_layout.shape[1][0]
        )
        threads_per_row = (
            tv_layout.shape[0]
            if const_expr(cute.rank(tv_layout.shape[0]) == 1)
            else tv_layout.shape[0][0]
        )
        thr_layout = cute.make_ordered_layout(
            (tiler_mn[0], threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, num_copy_elems))
        thr_copy_load = cute.make_tiled_copy_tv(
            copy_atom_load, thr_layout, val_layout
        ).get_slice(tidx)
        thr_copy_store = cute.make_tiled_copy_tv(
            copy_atom_store, thr_layout, val_layout
        ).get_slice(tidx)

        tXgX = thr_copy_load.partition_S(gX)
        tXsX = thr_copy_load.partition_D(sX)
        tdYgdY = thr_copy_load.partition_S(gdY)
        tdYsdY = thr_copy_load.partition_D(sdY)
        tdXgdX = thr_copy_store.partition_D(gdX)
        tXcX = thr_copy_load.partition_S(cX)[(0, None), None, None]

        tXrX, tdYrdY, tdXrdX = [
            cute.make_fragment_like(thr) for thr in (tXgX, tdYgdY, tdXgdX)
        ]

        if const_expr(
            self.cluster_n > 1
            and mbar_ptr_stats is not None
            and mbar_ptr_dot is not None
        ):
            if tidx < 2:
                cute.arch.mbarrier_init(mbar_ptr_stats + tidx, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()

        is_even_N = const_expr(self.N == tiler_mn[1] * self.cluster_n)
        tXpX = (
            predicate_k(thr_copy_load.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tXgX, tXsX, pred=tXpX)
            cute.copy(copy_atom_load, tdYgdY, tdYsdY, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        if const_expr(not is_even_N):
            fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
            fill_oob(tdYsdY, tXpX, 0.0)

        cute.autovec_copy(tXsX, tXrX)
        cute.autovec_copy(tdYsdY, tdYrdY)
        x = tXrX.load().to(Float32)
        dy = tdYrdY.load().to(Float32)

        _, denom, exp_x = online_softmax_reduce(
            x,
            threads_per_row,
            reduction_buffer_stats[None, None, 0],
            mbar_ptr_stats,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            phase=None,
            return_exp_x=True,
        )
        assert exp_x is not None  # noqa: S101
        y = exp_x * cute.arch.rcp_approx(denom)

        dot = row_reduce(
            dy * y,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer_dot[None, None, 0],
            mbar_ptr_dot,
            phase=None,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        dx = y * (dy - dot)
        tdXrdX.store(dx.to(tdXrdX.element_type))

        tOpO = (
            predicate_k(thr_copy_store.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tdXrdX, tdXgdX, pred=tOpO)

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mdY: cute.Tensor,
            mdX: cute.Tensor,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ) -> None:
            self._kernel_impl(mX, mdY, mdX, tv_layout, tiler_mn)
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mdY: cute.Tensor,
            mdX: cute.Tensor,
        ) -> None:
            tiler_mn, tv_layout = self._get_tv_layout()
            self._kernel_impl(mX, mdY, mdX, tv_layout, tiler_mn)


def _convert_2d_tensor(x: Tensor) -> cute.Tensor:
    # Match Quack's Softmax conversion exactly: assume 16B alignment and mark
    # the shape compact with row-major stride order (0, 1), with mode=0 (batch).
    # We intentionally do not call mark_layout_dynamic here to avoid the
    # leading_dim stride==1 constraint used in RMSNorm.
    return from_dlpack(x.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1)
    )


def _can_use_ptr_path_2d(x: Tensor) -> bool:
    """Conservative guard for the pointer-based fast path."""
    if not x.is_cuda or x.dim() != 2:
        return False
    if x.dtype not in TORCH2CUTE_DTYPE:
        return False
    # Require row-major last-dim contiguous.
    if x.stride(1) != 1:
        return False
    # Require 16B alignment (matches from_dlpack(..., assumed_align=16)).
    if (x.data_ptr() % 16) != 0:
        return False
    dtype_x = TORCH2CUTE_DTYPE[x.dtype]
    divby = 128 // dtype_x.width
    # Softmax uses ReductionBase default num_copy_bits=128, so N must be divisible.
    if (x.shape[1] % divby) != 0:
        return False
    # Ensure each row start remains aligned for 128-bit vectorized copies.
    if (x.stride(0) % divby) != 0:
        return False
    return True


def _softmax_forward_ptr_into(*, x: Tensor, out: Tensor) -> None:
    """Launch the pointer-based Softmax forward kernel into preallocated `out`."""
    assert x.is_cuda and x.dim() == 2  # noqa: S101
    assert out.is_cuda and out.shape == x.shape and out.dtype == x.dtype  # noqa: S101
    assert out.stride() == x.stride(), "Pointer path expects out to match x strides"  # noqa: S101

    M, N = x.shape
    device_index = x.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream_handle = int(torch.cuda.current_stream().cuda_stream)
    stream = cuda.CUstream(stream_handle)

    dtype_x = TORCH2CUTE_DTYPE[x.dtype]
    key = ("ptr_fwd", int(N), dtype_x, int(device_index))
    compiled = _PTR_FWD_COMPILE_CACHE.get(key)
    if compiled is None:
        op = SoftmaxFwdSM100(dtype_x, int(N))
        ptr_x = rt.make_ptr(
            dtype_x, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_out = rt.make_ptr(
            dtype_x, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ld = Int32(int(x.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_x,
            ptr_out,
            Int32(int(M)),
            ld,
            stream,
        )
        _PTR_FWD_COMPILE_CACHE[key] = compiled

    launcher = _get_fast_ptr_softmax_launcher(
        compiled=compiled,
        dtype=dtype_x,
        N=int(N),
        device_index=int(device_index),
        stream_handle=stream_handle,
        assumed_align=16,
        is_bwd=False,
    )
    if launcher is not None:
        launcher.launch(
            a_ptr=int(x.data_ptr()),
            b_ptr=int(out.data_ptr()),
            c_ptr=None,
            M=int(M),
            ld=int(x.stride(0)),
            stream_handle=stream_handle,
            dtype=dtype_x,
        )
        return

    ptr_x = rt.make_ptr(
        dtype_x, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_out = rt.make_ptr(
        dtype_x, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    compiled(ptr_x, ptr_out, Int32(int(M)), Int32(int(x.stride(0))), stream)


def _softmax_backward_ptr_into(*, dy: Tensor, y: Tensor, dx: Tensor) -> None:
    """Launch the pointer-based Softmax backward kernel into preallocated `dx`."""
    assert dy.is_cuda and dy.dim() == 2  # noqa: S101
    assert y.is_cuda and y.shape == dy.shape and y.dtype == dy.dtype  # noqa: S101
    assert dx.is_cuda and dx.shape == dy.shape and dx.dtype == dy.dtype  # noqa: S101
    assert dy.stride() == y.stride() == dx.stride(), (  # noqa: S101
        "Pointer path expects matching strides"
    )

    M, N = dy.shape
    device_index = dy.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream_handle = int(torch.cuda.current_stream().cuda_stream)
    stream = cuda.CUstream(stream_handle)

    dtype_x = TORCH2CUTE_DTYPE[dy.dtype]
    key = ("ptr_bwd", int(N), dtype_x, int(device_index))
    compiled = _PTR_BWD_COMPILE_CACHE.get(key)
    if compiled is None:
        op = SoftmaxBwdSM100(dtype_x, int(N))
        ptr_dy = rt.make_ptr(
            dtype_x, dy.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_y = rt.make_ptr(
            dtype_x, y.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_dx = rt.make_ptr(
            dtype_x, dx.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ld = Int32(int(dy.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_dy,
            ptr_y,
            ptr_dx,
            Int32(int(M)),
            ld,
            stream,
        )
        _PTR_BWD_COMPILE_CACHE[key] = compiled

    launcher = _get_fast_ptr_softmax_launcher(
        compiled=compiled,
        dtype=dtype_x,
        N=int(N),
        device_index=int(device_index),
        stream_handle=stream_handle,
        assumed_align=16,
        is_bwd=True,
    )
    if launcher is not None:
        launcher.launch(
            a_ptr=int(dy.data_ptr()),
            b_ptr=int(y.data_ptr()),
            c_ptr=int(dx.data_ptr()),
            M=int(M),
            ld=int(dy.stride(0)),
            stream_handle=stream_handle,
            dtype=dtype_x,
        )
        return

    ptr_dy = rt.make_ptr(
        dtype_x, dy.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_y = rt.make_ptr(
        dtype_x, y.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_dx = rt.make_ptr(
        dtype_x, dx.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    compiled(ptr_dy, ptr_y, ptr_dx, Int32(int(M)), Int32(int(dy.stride(0))), stream)


def _softmax_fwd_bwd_ptr_into(*, x: Tensor, dy: Tensor, dx: Tensor) -> None:
    """Launch the fused pointer-based Softmax fwd+bwd kernel into preallocated `dx`."""
    assert x.is_cuda and x.dim() == 2  # noqa: S101
    assert dy.is_cuda and dy.shape == x.shape and dy.dtype == x.dtype  # noqa: S101
    assert dx.is_cuda and dx.shape == x.shape and dx.dtype == x.dtype  # noqa: S101
    assert x.stride() == dy.stride() == dx.stride(), (  # noqa: S101
        "Pointer path expects matching strides"
    )

    M, N = x.shape
    device_index = x.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream_handle = int(torch.cuda.current_stream().cuda_stream)
    stream = cuda.CUstream(stream_handle)

    dtype_x = TORCH2CUTE_DTYPE[x.dtype]
    key = ("ptr_fwd_bwd", int(N), dtype_x, int(device_index))
    compiled = _PTR_FWDBWD_COMPILE_CACHE.get(key)
    if compiled is None:
        op = SoftmaxFwdBwdSM100(dtype_x, int(N))
        ptr_x = rt.make_ptr(
            dtype_x, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_dy = rt.make_ptr(
            dtype_x, dy.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_dx = rt.make_ptr(
            dtype_x, dx.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ld = Int32(int(x.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_x,
            ptr_dy,
            ptr_dx,
            Int32(int(M)),
            ld,
            stream,
        )
        _PTR_FWDBWD_COMPILE_CACHE[key] = compiled

    launcher = _get_fast_ptr_softmax_launcher(
        compiled=compiled,
        dtype=dtype_x,
        N=int(N),
        device_index=int(device_index),
        stream_handle=stream_handle,
        assumed_align=16,
        is_bwd=True,
    )
    if launcher is not None:
        launcher.launch(
            a_ptr=int(x.data_ptr()),
            b_ptr=int(dy.data_ptr()),
            c_ptr=int(dx.data_ptr()),
            M=int(M),
            ld=int(x.stride(0)),
            stream_handle=stream_handle,
            dtype=dtype_x,
        )
        return

    ptr_x = rt.make_ptr(
        dtype_x, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_dy = rt.make_ptr(
        dtype_x, dy.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_dx = rt.make_ptr(
        dtype_x, dx.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    compiled(ptr_x, ptr_dy, ptr_dx, Int32(int(M)), Int32(int(x.stride(0))), stream)


def softmax_forward(x: Tensor) -> Tensor:
    """SM100 CuteDSL softmax forward pass: y = softmax(x, dim=-1)."""
    assert x.dim() == 2, "Input must be 2D (M, N)"  # noqa: S101
    assert x.is_cuda, "Input must be on CUDA device"  # noqa: S101
    assert x.dtype in TORCH2CUTE_DTYPE, "Unsupported dtype"  # noqa: S101

    N = x.size(1)
    dtype = TORCH2CUTE_DTYPE[x.dtype]
    if _can_use_ptr_path_2d(x):
        out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
        _softmax_forward_ptr_into(x=x, out=out)
        return out

    out = torch.empty_like(x)

    x_tensor = _convert_2d_tensor(x)
    out_tensor = _convert_2d_tensor(out)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N)
    kernel = _FWD_COMPILE_CACHE.get(compile_key)
    if kernel is None:
        op = SoftmaxFwdSM100(dtype, N)
        kernel = cute.compile(op, x_tensor, out_tensor, current_stream)
        _FWD_COMPILE_CACHE[compile_key] = kernel
    kernel(x_tensor, out_tensor, current_stream)
    return out


def softmax_backward(dy: Tensor, y: Tensor) -> Tensor:
    """SM100 CuteDSL softmax backward pass."""
    assert dy.dim() == 2 and y.dim() == 2, "dy and y must be 2D (M, N)"  # noqa: S101
    assert dy.shape == y.shape, "dy and y must have the same shape"  # noqa: S101
    assert dy.is_cuda and y.is_cuda, "dy and y must be on CUDA device"  # noqa: S101
    assert dy.dtype in TORCH2CUTE_DTYPE, "Unsupported dtype"  # noqa: S101
    assert y.dtype == dy.dtype, "dy and y must have the same dtype"  # noqa: S101

    N = dy.size(1)
    dtype = TORCH2CUTE_DTYPE[dy.dtype]
    if (
        _can_use_ptr_path_2d(dy)
        and _can_use_ptr_path_2d(y)
        and dy.stride() == y.stride()
    ):
        dx = torch.empty_strided(
            dy.shape, dy.stride(), device=dy.device, dtype=dy.dtype
        )
        _softmax_backward_ptr_into(dy=dy, y=y, dx=dx)
        return dx

    dx = torch.empty_like(dy)

    dy_tensor = _convert_2d_tensor(dy)
    y_tensor = _convert_2d_tensor(y)
    dx_tensor = _convert_2d_tensor(dx)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N)
    kernel = _BWD_COMPILE_CACHE.get(compile_key)
    if kernel is None:
        op = SoftmaxBwdSM100(dtype, N)
        kernel = cute.compile(op, dy_tensor, y_tensor, dx_tensor, current_stream)
        _BWD_COMPILE_CACHE[compile_key] = kernel
    kernel(dy_tensor, y_tensor, dx_tensor, current_stream)
    return dx


def softmax_fwd_bwd(dy: Tensor, x: Tensor) -> Tensor:
    """Fused softmax forward+backward producing ``dx`` from ``(x, dy)``.

    This is intended for benchmarks and training-like use-cases where the
    intermediate ``y = softmax(x)`` is not needed outside the backward pass.
    """
    assert x.dim() == 2 and dy.dim() == 2, "x and dy must be 2D (M, N)"  # noqa: S101
    assert x.shape == dy.shape, "x and dy must have the same shape"  # noqa: S101
    assert x.is_cuda and dy.is_cuda, "x and dy must be on CUDA device"  # noqa: S101
    assert x.dtype in TORCH2CUTE_DTYPE, "Unsupported dtype"  # noqa: S101
    assert dy.dtype == x.dtype, "x and dy must have the same dtype"  # noqa: S101

    if (
        _can_use_ptr_path_2d(x)
        and _can_use_ptr_path_2d(dy)
        and x.stride() == dy.stride()
    ):
        dx = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
        _softmax_fwd_bwd_ptr_into(x=x, dy=dy, dx=dx)
        return dx

    with torch.no_grad():
        return softmax_backward(dy, softmax_forward(x))


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = softmax_forward(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy: Tensor) -> tuple[Tensor]:
        (y,) = ctx.saved_tensors
        dx = softmax_backward(dy, y)
        return dx


def softmax(x: Tensor) -> Tensor:
    """Autograd-friendly softmax using the SM100 CuteDSL kernel."""
    return SoftmaxFunction.apply(x)


def _torch_softmax_reference(x: Tensor) -> Tensor:
    return torch.nn.functional.softmax(x, dim=-1)


def verify_softmax_parity(
    M: int,
    N: int,
    dtype: torch.dtype = torch.bfloat16,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> None:
    """Compare SM100 CuteDSL softmax against PyTorch for a single shape."""
    device = torch.device("cuda")
    x = torch.randn(M, N, device=device, dtype=dtype)
    x.requires_grad_(True)

    # Forward parity
    y_ref = _torch_softmax_reference(x)
    y = softmax(x)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)

    # Backward parity
    dy = torch.randn_like(y)
    (dx_ref,) = torch.autograd.grad(y_ref, x, dy, retain_graph=False)
    dx = softmax_backward(dy, y)
    torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)
