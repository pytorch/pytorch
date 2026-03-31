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
RMSNorm kernel for SM100 (Blackwell) in CuteDSL, with the experimental
stage-2 cp.async ping-pong path preserved for N≈6k/8k.

This file is a fork of rmsnorm.py that keeps the K-loop cp.async path
behind `self.stage > 1` while the main implementation has been simplified
to a single-stage schedule.
"""

from __future__ import annotations

import importlib.metadata
import re

import cuda.bindings.driver as cuda  # provided by NVIDIA cuda-python

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32
from cutlass.cute.runtime import from_dlpack

import torch
from torch import Tensor

from .._oink_utils import lite_quack as qutils
from .._oink_utils.lite_quack import row_reduce, TORCH2CUTE_DTYPE


_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def _parse_version_tuple(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    nums: list[int] = []
    for part in parts[:3]:
        match = re.match(r"^(\d+)", part)
        nums.append(int(match.group(1)) if match is not None else 0)
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _cutlass_dsl_version() -> tuple[int, int, int] | None:
    try:
        return _parse_version_tuple(importlib.metadata.version("nvidia-cutlass-dsl"))
    except Exception:
        return None


_CUTLASS_DSL_VERSION = _cutlass_dsl_version()
# CuTeDSL 4.3.4 tightened some kernel argument expectations (notably around
# passing Layout/Shape/Constexpr objects into @cute.kernel functions). Keep the
# older signature for <4.3.4, but switch to a 4.3.4+ compatible signature when
# we detect 4.3.4+ (or when version detection is unavailable).
_KERNEL_ACCEPTS_LAYOUT_ARGS = (
    _CUTLASS_DSL_VERSION is not None and _CUTLASS_DSL_VERSION < (4, 3, 4)
)


@cute.jit
def get_copy_atom_bw(
    dtype: type[cutlass.Numeric], num_copy_elems: int, is_async: bool = False
) -> cute.CopyAtom:
    max_bits = const_expr(128 if is_async else 256)
    num_copy_bits = const_expr(min(max_bits, num_copy_elems * dtype.width))
    from cutlass.cute.nvgpu import cpasync

    copy_op = (
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
        if is_async
        else cute.nvgpu.CopyUniversalOp()
    )
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@cute.jit
def copy_tiled(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: cute.Tensor | None = None,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> None:
    atom = get_copy_atom_bw(src.element_type, num_copy_elems, is_async)
    cute.copy(atom, src, dst, pred=pred)


class RMSNormSM100WithStage2:
    def __init__(self, N: int, dtype: type[cutlass.Numeric], stage: int | None = None):
        self.N = N
        self.dtype = dtype
        self.stage = 1 if stage is None else stage
        self.reduction_dtype = cutlass.Float32

    def _threads_per_row(self) -> int:
        N = self.N
        if N <= 64:
            return 8
        elif N <= 128:
            return 16
        elif N <= 1024:
            return 32
        elif N <= 4096:
            return 128
        elif N <= 8192:
            try:
                return self._tpr_override  # type: ignore[attr-defined]
            except Exception:
                return 128
        elif N <= 16384:
            return 256
        else:
            return 256

    def _cluster_n(self) -> int:
        N = self.N
        if N <= 8192:
            return 1
        if const_expr(self.dtype.width == 16):
            if N <= 16 * 1024:
                return 2
            elif N <= 32 * 1024:
                return 2
            elif N <= 64 * 1024:
                return 4
            elif N <= 128 * 1024:
                return 8
            else:
                return 16
        else:
            if N <= 32 * 1024:
                return 1
            elif N <= 64 * 1024:
                return 2
            elif N <= 128 * 1024:
                return 4
            elif N <= 256 * 1024:
                return 8
            else:
                return 16

    def _num_threads(self) -> int:
        try:
            return self._nt_override  # type: ignore[attr-defined]
        except Exception:
            return 128 if self.N <= 16384 else 256

    def _tv_layout(self, num_copy_bits: int = 256) -> tuple[cute.Shape, cute.Layout]:
        vecsize = num_copy_bits // self.dtype.width
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0  # noqa: S101
        tpr = self._threads_per_row()
        cluster_n = self._cluster_n()
        num_cols_vec = cute.ceil_div(self.N, vecsize)
        num_blocks_N = cute.ceil_div(num_cols_vec, tpr * cluster_n)
        cols_per_block = num_threads // tpr
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * tpr)
        tv_layout = cute.make_layout(
            ((tpr, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * tpr),
            ),
        )
        return tiler_mn, tv_layout

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRes: cute.Tensor | None,
        mO: cute.Tensor,
        mResO: cute.Tensor | None,
        mRstd: cute.Tensor | None,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        semistatic_shape = (*mX.shape[:-1], self.N)

        def new_stride(t):
            return (
                cute.assume(t.stride[0], divby=256 // t.element_type.width),
                t.stride[1],
            )

        mX, mRes, mO, mResO = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            if const_expr(t is not None)
            else None
            for t in (mX, mRes, mO, mResO)
        ]
        assert mX.element_type == self.dtype  # noqa: S101
        assert mO.element_type == self.dtype  # noqa: S101

        copy_bits = const_expr(128)
        tiler_mn, tv_layout = self._tv_layout(num_copy_bits=copy_bits)
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        threads_per_row = (
            tv_layout.shape[0][0]
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._threads_per_row()
        )
        warps_per_row = max(threads_per_row // cute.arch.WARP_SIZE, 1)
        cluster_n = self._cluster_n()

        if const_expr(mW is not None):
            mW = cute.make_tensor(
                mW.iterator,
                cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
            )
        if const_expr(mB is not None):
            mB = cute.make_tensor(
                mB.iterator,
                cute.prepend(mB.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
            )
        if const_expr(mRstd is not None):
            mRstd = cute.make_tensor(
                mRstd.iterator,
                cute.append(mRstd.layout, cute.make_layout((self.N,), stride=(0,))),
            )

        stage_bufs = 2 if self.stage > 1 else 1
        tile_bytes_x = (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * stage_bufs
        )
        tile_bytes_res = (
            cute.size_in_bytes(mRes.element_type, cute.make_layout(tiler_mn))
            * stage_bufs
            if const_expr(mRes is not None)
            else 0
        )
        red_bytes = (
            self.stage * num_warps * cluster_n * (self.reduction_dtype.width // 8)
        )
        mbar_bytes = self.stage * (cutlass.Int64.width // 8)
        smem_bytes = tile_bytes_x + tile_bytes_res + red_bytes + mbar_bytes

        kernel = (
            self.kernel(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
                tv_layout,
                tiler_mn,
                const_expr(num_warps),
                const_expr(warps_per_row),
                const_expr(threads_per_row),
            )
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
            )
        )

        kernel.launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=([1, cluster_n, 1] if const_expr(cluster_n > 1) else None),
            smem=smem_bytes,
            stream=stream,
        )

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRes: cute.Tensor | None,
        mO: cute.Tensor,
        mResO: cute.Tensor | None,
        mRstd: cute.Tensor | None,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        num_warps: cutlass.Constexpr[int],
        warps_per_row: cutlass.Constexpr[int],
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_n = self._cluster_n()
        cluster_y = (
            const_expr(0) if const_expr(cluster_n == 1) else cute.arch.block_idx()[1]
        )

        smem = cutlass.utils.SmemAllocator()
        sX0 = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=32,
        )
        sX1 = (
            smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(self.stage > 1)
            else None
        )
        sRes0 = (
            smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(mRes is not None)
            else None
        )
        sRes1 = (
            smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(mRes is not None and self.stage > 1)
            else None
        )

        reduction_buffer, mbar_ptr = self._alloc_reduction_and_mbar(
            smem, num_warps, warps_per_row
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        num_copy_elems_X = tv_layout.shape[1][0]
        use_async = const_expr(self.N >= 1024)
        copy_atom = get_copy_atom_bw(
            mX.element_type, num_copy_elems_X, is_async=use_async
        )
        thr_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn).get_slice(tidx)

        gW, gB = [
            cute.local_tile(t, tiler_mn, (0, cluster_y))
            if const_expr(t is not None)
            else None
            for t in (mW, mB)
        ]
        tXgW = thr_copy.partition_S(gW) if const_expr(mW is not None) else None
        tXgB = thr_copy.partition_S(gB) if const_expr(mB is not None) else None
        tXrW = cute.make_fragment_like(tXgW) if const_expr(mW is not None) else None
        tXrB = cute.make_fragment_like(tXgB) if const_expr(mB is not None) else None
        if const_expr(mW is not None):
            cute.copy(
                get_copy_atom_bw(mW.element_type, num_copy_elems_X, is_async=False),
                tXgW,
                tXrW,
            )
        if const_expr(mB is not None):
            cute.copy(
                get_copy_atom_bw(mB.element_type, num_copy_elems_X, is_async=False),
                tXgB,
                tXrB,
            )

        self._init_cluster(tidx, mbar_ptr)

        mX_i, mRes_i, mO_i, mResO_i = [
            qutils.domain_offset_i64((bidx * tiler_mn[0], 0), t)
            if t is not None
            else None
            for t in (mX, mRes, mO, mResO)
        ]
        gX_i = cute.local_tile(mX_i, tiler_mn, (0, cluster_y))
        gO_i = cute.local_tile(mO_i, tiler_mn, (0, cluster_y))
        gRes_i = (
            cute.local_tile(mRes_i, tiler_mn, (0, cluster_y))
            if const_expr(mRes is not None)
            else None
        )
        gResO_i = (
            cute.local_tile(mResO_i, tiler_mn, (0, cluster_y))
            if const_expr(mResO is not None)
            else None
        )
        gRstd_i = (
            cute.local_tile(mRstd, tiler_mn, (bidx, cluster_y))
            if const_expr(mRstd is not None)
            else None
        )
        cX_i = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        tXcX_i = thr_copy.partition_S(cX_i)[(0, None), None, None]
        row_i = tXcX_i[0][0]
        tXgRstd_i = (
            thr_copy.partition_D(gRstd_i) if const_expr(mRstd is not None) else None
        )

        # Intra-row K-loop cp.async ping-pong (two-pass) for N≈6k/8k (stage=2)
        if const_expr(self.stage > 1 and (shape[1] == 6144 or shape[1] == 8192)):
            vecsize = tv_layout.shape[1][0]
            tpr = threads_per_row
            target_tile_n = const_expr(4096 if shape[1] == 6144 else 8192)
            tile_factor = const_expr(target_tile_n // (vecsize * tpr))
            tile_n = vecsize * tpr * tile_factor
            num_tiles = cute.ceil_div(shape[1], tile_n)

            tiler_mn_tile = (tiler_mn[0], tile_n)
            sX0_tile = cute.local_tile(sX0, tiler_mn_tile, (0, 0))
            sX1_tile = (
                cute.local_tile(sX1, tiler_mn_tile, (0, 0))
                if const_expr(self.stage > 1)
                else None
            )
            sRes0_tile = (
                cute.local_tile(sRes0, tiler_mn_tile, (0, 0))
                if const_expr(mRes is not None)
                else None
            )
            sRes1_tile = (
                cute.local_tile(sRes1, tiler_mn_tile, (0, 0))
                if const_expr(mRes is not None and self.stage > 1)
                else None
            )

            tv_layout_tile = cute.make_layout(
                ((tpr, tiler_mn[0]), (vecsize, tile_factor)),
                stride=(
                    (vecsize * tiler_mn[0], 1),
                    (tiler_mn[0], tiler_mn[0] * vecsize * tpr),
                ),
            )
            thr_copy_tile = cute.make_tiled_copy(
                copy_atom, tv_layout_tile, tiler_mn_tile
            ).get_slice(tidx)

            sum_sq_acc = cute.Float32(0.0)
            k_off0 = const_expr(0) * tile_n
            gX_0 = cute.local_tile(
                qutils.domain_offset_i64((0, k_off0), mX_i),
                tiler_mn_tile,
                (0, cluster_y),
            )
            tXgX_0 = thr_copy_tile.partition_S(gX_0)
            tXsX_0 = thr_copy_tile.partition_D(sX0_tile)
            cX_0 = cute.local_tile(
                cute.domain_offset((0, k_off0), cX_i), tiler_mn_tile, (0, cluster_y)
            )
            tXc_0 = thr_copy_tile.partition_S(cX_0)
            tXp_0 = qutils.predicate_k(tXc_0, limit=shape[1])
            tXp_ping = tXp_0
            tXp_pong = tXp_0
            if row_i < shape[0]:
                copy_tiled(
                    tXgX_0,
                    tXsX_0,
                    num_copy_elems=vecsize,
                    is_async=use_async,
                    pred=tXp_0,
                )
                if const_expr(mRes is not None):
                    gRes_0 = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off0), mRes_i),
                        tiler_mn_tile,
                        (0, cluster_y),
                    )
                    tXgRes_0 = thr_copy_tile.partition_S(gRes_0)
                    tXsRes_0 = thr_copy_tile.partition_D(sRes0_tile)
                    copy_tiled(
                        tXgRes_0,
                        tXsRes_0,
                        num_copy_elems=vecsize,
                        is_async=use_async,
                        pred=tXp_0,
                    )
            if const_expr(use_async):
                cute.arch.cp_async_commit_group()

            for t in cutlass.range_constexpr(num_tiles):
                next_t = t + 1
                if next_t < num_tiles:
                    k_off_n = next_t * tile_n
                    gX_n = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off_n), mX_i),
                        tiler_mn_tile,
                        (0, cluster_y),
                    )
                    tXgX_n = thr_copy_tile.partition_S(gX_n)
                    cX_n = cute.local_tile(
                        cute.domain_offset((0, k_off_n), cX_i),
                        tiler_mn_tile,
                        (0, cluster_y),
                    )
                    tXc_n = thr_copy_tile.partition_S(cX_n)
                    tXp_n = qutils.predicate_k(tXc_n, limit=shape[1])
                    if const_expr((t % 2) == 0):
                        tXsX_n = thr_copy_tile.partition_D(sX1_tile)
                        tXsRes_n = (
                            thr_copy_tile.partition_D(sRes1_tile)
                            if const_expr(mRes is not None)
                            else None
                        )
                        tXp_pong = tXp_n
                    else:
                        tXsX_n = thr_copy_tile.partition_D(sX0_tile)
                        tXsRes_n = (
                            thr_copy_tile.partition_D(sRes0_tile)
                            if const_expr(mRes is not None)
                            else None
                        )
                        tXp_ping = tXp_n
                    if row_i < shape[0]:
                        copy_tiled(
                            tXgX_n,
                            tXsX_n,
                            num_copy_elems=vecsize,
                            is_async=use_async,
                            pred=tXp_n,
                        )
                        if const_expr(mRes is not None):
                            gRes_n = cute.local_tile(
                                qutils.domain_offset_i64((0, k_off_n), mRes_i),
                                tiler_mn_tile,
                                (0, cluster_y),
                            )
                            tXgRes_n = thr_copy_tile.partition_S(gRes_n)
                            copy_tiled(
                                tXgRes_n,
                                tXsRes_n,
                                num_copy_elems=vecsize,
                                is_async=use_async,
                                pred=tXp_n,
                            )
                    if const_expr(use_async):
                        cute.arch.cp_async_commit_group()
                if const_expr(use_async):
                    cute.arch.cp_async_wait_group(1 if next_t < num_tiles else 0)

                if const_expr((t % 2) == 0):
                    tXsX_cur = thr_copy_tile.partition_D(sX0_tile)
                    tXsRes_cur = (
                        thr_copy_tile.partition_D(sRes0_tile)
                        if const_expr(mRes is not None)
                        else None
                    )
                    pred_cur = tXp_ping
                else:
                    tXsX_cur = thr_copy_tile.partition_D(sX1_tile)
                    tXsRes_cur = (
                        thr_copy_tile.partition_D(sRes1_tile)
                        if const_expr(mRes is not None)
                        else None
                    )
                    pred_cur = tXp_pong
                qutils.fill_oob(tXsX_cur, pred_cur, mX.element_type.zero)
                if const_expr(mRes is not None):
                    qutils.fill_oob(tXsRes_cur, pred_cur, mRes.element_type.zero)

                k_off = t * tile_n
                gX_t = cute.local_tile(
                    qutils.domain_offset_i64((0, k_off), mX_i),
                    tiler_mn_tile,
                    (0, cluster_y),
                )
                tXgX_t = thr_copy_tile.partition_S(gX_t)
                tXrX = cute.make_fragment_like(tXgX_t)
                cute.autovec_copy(tXsX_cur, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    gRes_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mRes_i),
                        tiler_mn_tile,
                        (0, cluster_y),
                    )
                    tXgRes_t = thr_copy_tile.partition_S(gRes_t)
                    tXrRes = cute.make_fragment_like(tXgRes_t)
                    cute.autovec_copy(tXsRes_cur, tXrRes)
                    x += tXrRes.load().to(cute.Float32)

                if const_expr(mResO is not None):
                    gResO_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mResO_i),
                        tiler_mn_tile,
                        (0, cluster_y),
                    )
                    tXgResO_t = thr_copy_tile.partition_D(gResO_t)
                    tXrResO = cute.make_fragment_like(tXgResO_t)
                    tXrResO.store(x.to(tXrResO.element_type))
                    if row_i < shape[0]:
                        copy_tiled(
                            tXrResO,
                            tXgResO_t,
                            num_copy_elems=vecsize,
                            is_async=False,
                            pred=pred_cur,
                        )

                sum_sq_tile = row_reduce(
                    x * x,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, 0],
                    mbar_ptr,
                    init_val=0.0,
                    hook_fn=(
                        cute.arch.cluster_wait if const_expr(cluster_n > 1) else None
                    ),
                )
                sum_sq_acc = sum_sq_acc + sum_sq_tile

            rstd = cute.math.rsqrt(sum_sq_acc / shape[1] + eps, fastmath=True)
            if const_expr(mRstd is not None):
                if (
                    tXcX_i[0][1] == 0
                    and row_i < shape[0]
                    and (cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
                ):
                    tXgRstd_i[0] = rstd

            for t in cutlass.range_constexpr(num_tiles):
                k_off = t * tile_n
                cX_t = cute.local_tile(
                    cute.domain_offset((0, k_off), cX_i), tiler_mn_tile, (0, cluster_y)
                )
                tXc_t = thr_copy_tile.partition_S(cX_t)
                tXp_t = qutils.predicate_k(tXc_t, limit=shape[1])

                if const_expr((t % 2) == 0):
                    tXsX_cur = thr_copy_tile.partition_D(sX0_tile)
                    tXsRes_cur = (
                        thr_copy_tile.partition_D(sRes0_tile)
                        if const_expr(mRes is not None)
                        else None
                    )
                else:
                    tXsX_cur = thr_copy_tile.partition_D(sX1_tile)
                    tXsRes_cur = (
                        thr_copy_tile.partition_D(sRes1_tile)
                        if const_expr(mRes is not None)
                        else None
                    )

                qutils.fill_oob(tXsX_cur, tXp_t, mX.element_type.zero)
                if const_expr(mRes is not None):
                    qutils.fill_oob(tXsRes_cur, tXp_t, mRes.element_type.zero)

                gX_t = cute.local_tile(
                    qutils.domain_offset_i64((0, k_off), mX_i),
                    tiler_mn_tile,
                    (0, cluster_y),
                )
                tXgX_t = thr_copy_tile.partition_S(gX_t)
                tXrX = cute.make_fragment_like(tXgX_t)
                cute.autovec_copy(tXsX_cur, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    gRes_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mRes_i),
                        tiler_mn_tile,
                        (0, cluster_y),
                    )
                    tXgRes_t = thr_copy_tile.partition_S(gRes_t)
                    tXrRes = cute.make_fragment_like(tXgRes_t)
                    cute.autovec_copy(tXsRes_cur, tXrRes)
                    x += tXrRes.load().to(cute.Float32)

                y = x * rstd
                if const_expr(mW is not None):
                    gW_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mW),
                        tiler_mn_tile,
                        (0, cluster_y),
                    )
                    tWgW_t = thr_copy_tile.partition_S(gW_t)
                    tWrW_t = cute.make_fragment_like(tWgW_t)
                    copy_tiled(
                        tWgW_t,
                        tWrW_t,
                        num_copy_elems=vecsize,
                        is_async=False,
                        pred=tXp_t,
                    )
                    y = y * tWrW_t.load().to(cute.Float32)
                if const_expr(mB is not None):
                    gB_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mB),
                        tiler_mn_tile,
                        (0, cluster_y),
                    )
                    tWgB_t = thr_copy_tile.partition_S(gB_t)
                    tWrB_t = cute.make_fragment_like(tWgB_t)
                    copy_tiled(
                        tWgB_t,
                        tWrB_t,
                        num_copy_elems=vecsize,
                        is_async=False,
                        pred=tXp_t,
                    )
                    y = y + tWrB_t.load().to(cute.Float32)

                gO_t = cute.local_tile(
                    qutils.domain_offset_i64((0, k_off), mO_i),
                    tiler_mn_tile,
                    (0, cluster_y),
                )
                tXgO_t = thr_copy_tile.partition_D(gO_t)
                tXrO = cute.make_fragment_like(tXgO_t)
                tXrO.store(y.to(tXrO.element_type))
                if row_i < shape[0]:
                    copy_tiled(
                        tXrO, tXgO_t, num_copy_elems=vecsize, is_async=False, pred=tXp_t
                    )

            return

        # Fallback: single-stage path identical to current rmsnorm.py
        tXgX_i = thr_copy.partition_S(gX_i)
        tXgRes_i = (
            thr_copy.partition_S(gRes_i) if const_expr(mRes is not None) else None
        )
        tXgO_i = thr_copy.partition_D(gO_i)
        tXgResO_i = (
            thr_copy.partition_D(gResO_i) if const_expr(mResO is not None) else None
        )
        is_even_N_i = const_expr(shape[1] == tiler_mn[1] * cluster_n)
        tXpX_i = (
            qutils.predicate_k(thr_copy.partition_S(cX_i), limit=shape[1])
            if not is_even_N_i
            else None
        )

        if row_i < shape[0]:
            cute.copy(copy_atom, tXgX_i, thr_copy.partition_D(sX0), pred=tXpX_i)
            if const_expr(mRes is not None):
                cute.copy(copy_atom, tXgRes_i, thr_copy.partition_D(sRes0), pred=tXpX_i)
        if const_expr(use_async):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)

        tXrX = cute.make_fragment_like(tXgX_i)
        cute.autovec_copy(thr_copy.partition_D(sX0), tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(mRes is not None):
            tXrRes = cute.make_fragment_like(tXgRes_i)
            cute.autovec_copy(thr_copy.partition_D(sRes0), tXrRes)
            x += tXrRes.load().to(cute.Float32)

        if const_expr(mResO is not None):
            tXrResO = cute.make_fragment_like(tXgResO_i)
            tXrResO.store(x.to(tXrResO.element_type))
            if row_i < shape[0]:
                cute.copy(
                    get_copy_atom_bw(
                        tXrResO.element_type, num_copy_elems_X, is_async=False
                    ),
                    tXrResO,
                    tXgResO_i,
                )

        sum_sq = row_reduce(
            x * x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=(cute.arch.cluster_wait if const_expr(cluster_n > 1) else None),
        )
        rstd = cute.math.rsqrt(sum_sq / shape[1] + eps, fastmath=True)

        if const_expr(mRstd is not None):
            if (
                tXcX_i[0][1] == 0
                and row_i < shape[0]
                and (cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXgRstd_i[0] = rstd

        y = x * rstd
        if const_expr(mW is not None):
            y = y * tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y = y + tXrB.load().to(cute.Float32)

        tXrO = cute.make_fragment_like(tXgO_i)
        tXrO.store(y.to(tXrO.element_type))
        if row_i < shape[0]:
            cute.copy(
                get_copy_atom_bw(tXrO.element_type, num_copy_elems_X, is_async=False),
                tXrO,
                tXgO_i,
            )

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor | None,
            mB: cute.Tensor | None,
            mRes: cute.Tensor | None,
            mO: cute.Tensor,
            mResO: cute.Tensor | None,
            mRstd: cute.Tensor | None,
            eps: Float32,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
            num_warps: cutlass.Constexpr[int],
            warps_per_row: cutlass.Constexpr[int],
            threads_per_row: cutlass.Constexpr[int],
        ):
            self._kernel_impl(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
                tv_layout,
                tiler_mn,
                num_warps,
                warps_per_row,
                threads_per_row,
            )
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor | None,
            mB: cute.Tensor | None,
            mRes: cute.Tensor | None,
            mO: cute.Tensor,
            mResO: cute.Tensor | None,
            mRstd: cute.Tensor | None,
            eps: Float32,
        ):
            copy_bits = const_expr(128)
            tiler_mn, tv_layout = self._tv_layout(num_copy_bits=copy_bits)
            num_threads = self._num_threads()
            num_warps = num_threads // cute.arch.WARP_SIZE
            threads_per_row = self._threads_per_row()
            warps_per_row = max(threads_per_row // cute.arch.WARP_SIZE, 1)
            self._kernel_impl(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
                tv_layout,
                tiler_mn,
                const_expr(num_warps),
                const_expr(warps_per_row),
                const_expr(threads_per_row),
            )

    @cute.jit
    def _alloc_reduction_and_mbar(
        self,
        smem: cutlass.utils.SmemAllocator,
        num_warps: cutlass.Constexpr[int],
        warps_per_row: cutlass.Constexpr[int],
    ) -> tuple[cute.Tensor, cute.Pointer | None]:
        cluster_n = self._cluster_n()
        red_layout = cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype, red_layout, byte_alignment=4
        )
        if const_expr(cluster_n > 1):
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=self.stage)
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _init_cluster(self, tidx: cutlass.Int32, mbar_ptr: cute.Pointer | None):
        if const_expr(mbar_ptr is not None):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()


def rmsnorm_forward_with_stage2(
    x: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    residual: Tensor | None = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    assert x.is_cuda  # noqa: S101
    assert x.dim() == 2  # noqa: S101
    M, N = x.shape
    dtype = TORCH2CUTE_DTYPE[x.dtype]

    def _convert_x(t: Tensor) -> cute.Tensor:
        return from_dlpack(t.detach(), assumed_align=32).mark_layout_dynamic(
            leading_dim=1
        )

    mX = _convert_x(x)
    mRes = _convert_x(residual) if residual is not None else None
    out = torch.empty_like(x, dtype=x.dtype)
    mO = from_dlpack(out.detach(), assumed_align=32).mark_layout_dynamic(leading_dim=1)

    mW = (
        from_dlpack(weight.detach(), assumed_align=32).mark_layout_dynamic(
            leading_dim=0
        )
        if weight is not None
        else None
    )
    mB = (
        from_dlpack(bias.detach(), assumed_align=32).mark_layout_dynamic(leading_dim=0)
        if bias is not None
        else None
    )
    if store_rstd:
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
        mRstd = from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=0
        )
    else:
        rstd = None
        mRstd = None

    residual_out = None
    mResO = None
    if residual is not None:
        residual_out = torch.empty_like(residual)
        mResO = from_dlpack(
            residual_out.detach(), assumed_align=32
        ).mark_layout_dynamic(leading_dim=1)

    # Enable the intra-row cp.async K-loop only for DSv3-style large-N rows
    # with very large M, where there is enough work per row to amortize the
    # pipeline start-up cost. Mid-size M shapes are better served by the
    # simpler single-stage schedule.
    use_kloop = bool(M >= 65536 and N in (6144, 8192))
    stage = 2 if use_kloop else 1
    op = RMSNormSM100WithStage2(N, dtype, stage=stage)
    if use_kloop:
        op._tpr_override = 128  # type: ignore[attr-defined]
        # Prefer 1 row/CTA at N=6144; keep 2 rows/CTA at N=8192 to match
        # the original tuning there.
        op._nt_override = 128 if N == 6144 else 256  # type: ignore[attr-defined]

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (
        N,
        dtype,
        mRes is not None,
        mW is not None,
        mB is not None,
        mResO is not None,
        mRstd is not None,
        stage,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            op, mX, mW, mB, mRes, mO, mResO, mRstd, stream, Float32(eps)
        )
        _COMPILE_CACHE[key] = compiled
    compiled(mX, mW, mB, mRes, mO, mResO, mRstd, stream, Float32(eps))
    return out, rstd, residual_out
