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
Lightweight local clone of the small subset of Quack helpers that the SM100
RMSNorm CuteDSL kernels depend on.

This module intentionally avoids importing the `quack` package so that
KernelAgent Oink SM100 kernels can run without Quack installed, while keeping
numerical behaviour and performance identical to the reference kernels.

Reference implementation provenance:
- Quack repo: https://github.com/Dao-AILab/quack
- Base commit: c227eb56abc1b434f366d31b9d4a6ab4f00e8900
  ("[RmsNorm] Compile with tvm-ffi and fake tensors")
"""

import importlib.metadata
import math
import operator
import re
from collections.abc import Callable
from functools import partial

import cuda.bindings.driver as cuda  # type: ignore[import-untyped]

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op, T

import torch
from torch import Tensor


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

# Cache device properties lookups (notably `multi_processor_count`) since some
# dispatch paths call `get_sm_count` inside tight benchmark loops.
_DEVICE_NUM_SMS_CACHE: dict[int, int] = {}


def get_num_sms(device: torch.device) -> int:
    """Return the number of SMs for a CUDA device (cached)."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_index = int(device_index)
    cached = _DEVICE_NUM_SMS_CACHE.get(device_index)
    if cached is not None:
        return cached
    num_sms = int(torch.cuda.get_device_properties(device_index).multi_processor_count)
    _DEVICE_NUM_SMS_CACHE[device_index] = num_sms
    return num_sms


# -------------------------
# Dtype mapping (from quack.cute_dsl_utils)
# -------------------------

TORCH2CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


# -------------------------
# Tensor conversion helpers (from quack.utils)
# -------------------------


def convert_from_dlpack(
    x: Tensor,
    leading_dim: int,
    alignment: int = 16,
    divisibility: int = 1,
) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=x.dim_order(),
            divisibility=divisibility,
        )
    )


# -------------------------
# SM90/SM100 cluster helpers (from quack.utils)
# -------------------------


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr,
        peer_cta_rank_in_cluster,
        loc=loc,
        ip=ip,
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr,
        peer_cta_rank_in_cluster,
        loc=loc,
        ip=ip,
    ).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), (  # noqa: S101
        "val must be Float32, Int32, or Int64"
    )
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def atomic_add_f32(
    a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> Float32:
    """Atomic add into global memory (float32)."""
    from cutlass import CUDA_VERSION

    # NVVM atomicrmw API changed in CUDA 13.x: older builds (notably CUDA 12.9)
    # require an explicit result type, while newer builds infer it.
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        return nvvm.atomicrmw(
            res=T.f32(),
            op=nvvm.AtomicOpKind.FADD,
            ptr=gmem_ptr.llvm_ptr,
            a=Float32(a).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    return nvvm.atomicrmw(
        op=nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Float32(a).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@cute.jit
def atomic_add_tensor_f32(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: cute.Tensor | None = None,
) -> None:
    """Atomic-add a register fragment into a GMEM tile (float32)."""
    if const_expr(pred is None):
        for i in cutlass.range_constexpr(cute.size(src.shape)):
            coord = cute.idx2crd(i, src.shape)
            atomic_add_f32(src[i], elem_pointer(dst, coord))
    else:
        for i in cutlass.range_constexpr(cute.size(src.shape)):
            # CuTeDSL 4.3.4+ disallows introducing new tuple-typed values inside
            # a dynamic `if`. Compute `coord` unconditionally, then predicate the
            # atomic update.
            coord = cute.idx2crd(i, src.shape)
            if pred[i]:
                atomic_add_f32(src[i], elem_pointer(dst, coord))


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if".
    tApA = cute.make_fragment(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(
                tAcA[(0, rest_v), 0, rest_k][1], limit
            )
    return tApA


@dsl_user_op
def domain_offset_i64(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride), (  # noqa: S101
        "Coordinate and stride must have the same length"
    )
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)  # noqa: S101
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def coord_offset_i64(
    idx: cute.typing.Int,
    tensor: cute.Tensor,
    dim: int,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    offset = cutlass.Int64(idx) * cute.size(tensor.stride[dim])
    assert isinstance(tensor.iterator, cute.Pointer)  # noqa: S101
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@cute.jit
def fill_oob(
    tXsX: cute.Tensor, tXpX: cute.Tensor | None, fill_value: cutlass.Numeric
) -> None:
    """Fill out-of-bounds values in shared memory tensor."""
    tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(const_expr(tXsX.shape[0][1])):
        for rest_k in cutlass.range_constexpr(const_expr(tXsX.shape[2])):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    """Pack two f32 values into a single i64.

    This mirrors quack.utils.f32x2_to_i64 and is used by online_softmax_reduce
    to store (max, sum_exp) pairs in an Int64 reduction buffer.
    """
    vec_f32x2 = vector.from_elements(
        T.vector(2, T.f32()),
        (a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)),
        loc=loc,
        ip=ip,
    )
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2, loc=loc, ip=ip)
    res = cutlass.Int64(
        vector.extract(
            vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip
        )
    )
    return res


@dsl_user_op
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    """Unpack a single i64 into two f32 values, inverse of f32x2_to_i64."""
    vec_i64x1 = vector.from_elements(
        T.vector(1, T.i64()),
        (c.ir_value(loc=loc, ip=ip),),
        loc=loc,
        ip=ip,
    )
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1, loc=loc, ip=ip)
    res0 = Float32(
        vector.extract(
            vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip
        )
    )
    res1 = Float32(
        vector.extract(
            vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip
        )
    )
    return res0, res1


# -------------------------
# Reduction helpers (from quack.reduce)
# -------------------------


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    return cute.arch.warp_reduction(val, op, threads_in_group=width)


@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warp_per_row, warps_per_row)."""
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: cutlass.Int32 | None = None,
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warps_per_row, (warps_per_row, cluster_n))."""
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps * cluster_n * reduction_buffer.element_type.width // 8,
            )
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    return warp_reduce(block_reduce_val, op)


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer | None,
    phase: cutlass.Int32 | None = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Perform either block or cluster reduction based on whether mbar_ptr is provided."""
    if cutlass.const_expr(mbar_ptr is None):
        return block_reduce(val, op, reduction_buffer, init_val=init_val)
    return cluster_reduce(
        val, op, reduction_buffer, mbar_ptr, init_val=init_val, phase=phase
    )


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor | None = None,
    mbar_ptr: cute.Pointer | None = None,
    phase: cutlass.Int32 | None = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Callable | None = None,
) -> cute.Numeric:
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n))."""
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax
        if cutlass.const_expr(x.dtype == Float32)
        else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = warp_reduce(
        val,
        warp_op,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if cutlass.const_expr(hook_fn is not None):
        hook_fn()
    if cutlass.const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert cluster_n == 1 or mbar_ptr is not None, (  # noqa: S101
            "mbar_ptr must be provided for cluster reduction"
        )
        if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
            val = block_or_cluster_reduce(
                val,
                warp_op,
                reduction_buffer,
                mbar_ptr,
                phase=phase,
                init_val=init_val,
            )
    return val


@cute.jit
def row_reduce_add(
    x: cute.TensorSSA | cute.Numeric,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor | None = None,
    mbar_ptr: cute.Pointer | None = None,
    phase: cutlass.Int32 | None = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Callable | None = None,
) -> cute.Numeric:
    """Specialized row_reduce for ADD reductions.

    This mirrors row_reduce but hardcodes the ADD operation so we avoid
    dynamic dispatch on the reduction op. It is used by bandwidth-bound
    kernels like RMSNorm backward where the reduction is always ADD in
    Float32.
    """
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(cute.ReductionOp.ADD, init_val=init_val, reduction_profile=0)
    else:
        val = x
    val = warp_reduce(
        val,
        operator.add,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if cutlass.const_expr(hook_fn is not None):
        hook_fn()
    if cutlass.const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert cluster_n == 1 or mbar_ptr is not None, (  # noqa: S101
            "mbar_ptr must be provided for cluster reduction"
        )
        if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
            val = block_or_cluster_reduce(
                val,
                operator.add,
                reduction_buffer,
                mbar_ptr,
                phase=phase,
                init_val=init_val,
            )
    return val


@cute.jit
def online_softmax_reduce(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor | None = None,
    mbar_ptr: cute.Pointer | None = None,
    hook_fn: Callable | None = None,
    phase: cutlass.Int32 | None = None,
    return_exp_x: bool = False,
) -> tuple[Float32, Float32, cute.TensorSSA | None]:
    """Online softmax reduction over a row.

    This mirrors quack.reduce.online_softmax_reduce and computes:
      - max_x: row-wise maximum of x
      - sum_exp_x: row-wise sum of exp(x - max_x)
      - exp_x (optional): per-element exp(x - max_x_final) if return_exp_x is True
    """
    assert x.dtype == Float32, "x must be of type Float32"  # noqa: S101
    # reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n), 2)
    max_x = warp_reduce(
        x.reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
        cute.arch.fmax,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    log2_e = math.log2(math.e)
    exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)
    sum_exp_x = warp_reduce(
        exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if cutlass.const_expr(hook_fn is not None):
        hook_fn()
    if cutlass.const_expr(reduction_buffer is not None):
        rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
        assert cluster_n == 1 or mbar_ptr is not None, (  # noqa: S101
            "mbar_ptr must be provided for cluster reduction"
        )
        if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
            assert reduction_buffer.element_type == cutlass.Int64, (  # noqa: S101
                "reduction_buffer must be of type Int64"
            )
            lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
            row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
            if cutlass.const_expr(mbar_ptr is None):
                if lane_idx == 0:
                    reduction_buffer[row_idx, col_idx] = f32x2_to_i64(max_x, sum_exp_x)
                cute.arch.barrier()
                max_x_single_warp = -Float32.inf
                sum_exp_x = 0.0
                if lane_idx < warps_per_row:
                    max_x_single_warp, sum_exp_x = i64_to_f32x2(
                        reduction_buffer[row_idx, lane_idx]
                    )
                max_x_final = warp_reduce(max_x_single_warp, cute.arch.fmax)
                sum_exp_x *= cute.math.exp(
                    max_x_single_warp - max_x_final, fastmath=True
                )
                sum_exp_x = warp_reduce(sum_exp_x, operator.add)
                if cutlass.const_expr(return_exp_x):
                    exp_x *= cute.math.exp(max_x - max_x_final, fastmath=True)
                max_x = max_x_final
            else:
                cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        num_warps = rows_per_block * warps_per_row
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr,
                            num_warps
                            * cluster_n
                            * reduction_buffer.element_type.width
                            // 8,
                        )
                if lane_idx < cluster_n:
                    store_shared_remote(
                        f32x2_to_i64(max_x, sum_exp_x),
                        elem_pointer(
                            reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))
                        ),
                        mbar_ptr,
                        peer_cta_rank_in_cluster=lane_idx,
                    )
                cute.arch.mbarrier_wait(
                    mbar_ptr, phase=phase if phase is not None else 0
                )
                num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
                max_x_single_warp = cute.make_fragment(num_iter, Float32)
                max_x_single_warp.fill(-Float32.inf)
                sum_exp_x_single_warp = cute.make_fragment(num_iter, Float32)
                sum_exp_x_single_warp.fill(0.0)
                for i in cutlass.range_constexpr(num_iter):
                    idx = lane_idx + i * cute.arch.WARP_SIZE
                    if idx < cute.size(reduction_buffer, mode=[1]):
                        max_x_single_warp[i], sum_exp_x_single_warp[i] = i64_to_f32x2(
                            reduction_buffer[row_idx, idx]
                        )
                max_x_final = max_x_single_warp.load().reduce(
                    cute.ReductionOp.MAX,
                    init_val=-Float32.inf,
                    reduction_profile=0,
                )
                max_x_final = warp_reduce(max_x_final, cute.arch.fmax)
                sum_exp_x = 0.0
                for i in cutlass.range_constexpr(num_iter):
                    sum_exp_x += sum_exp_x_single_warp[i] * cute.math.exp(
                        max_x_single_warp[i] - max_x_final,
                        fastmath=True,
                    )
                sum_exp_x = warp_reduce(sum_exp_x, operator.add)
                if cutlass.const_expr(return_exp_x):
                    exp_x *= cute.math.exp(max_x - max_x_final, fastmath=True)
                max_x = max_x_final
    return max_x, sum_exp_x, (exp_x if cutlass.const_expr(return_exp_x) else None)


# -------------------------
# Copy helpers (minimal subset of quack.copy_utils)
# -------------------------


@dsl_user_op
def get_copy_atom(
    dtype: type[cutlass.Numeric],
    num_copy_elems: int,
    is_async: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    from cutlass.cute.nvgpu import cpasync

    # cp.async is limited to 128b per op; synchronous vectorized copies can go wider.
    max_bits = const_expr(128 if is_async else 256)
    num_copy_bits = const_expr(min(max_bits, num_copy_elems * dtype.width))
    # Match Quack's default cp.async cache policy (leave cache_mode unspecified).
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(
        copy_op, dtype, num_bits_per_copy=num_copy_bits, loc=loc, ip=ip
    )


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: cute.Tensor | None = None,
    num_copy_elems: int = 1,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    copy_atom = get_copy_atom(
        src.element_type, num_copy_elems, is_async, loc=loc, ip=ip
    )
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


# -------------------------
# Reduction base (from quack.reduction_base)
# -------------------------


class ReductionBase:
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        N: int,
        stage: int,
        reduction_dtype: type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.dtype = dtype
        self.N = N
        self.stage = stage
        self.reduction_dtype = reduction_dtype

    def _calculate_threads_per_row(self) -> int:
        raise NotImplementedError

    def _set_cluster_n(self) -> None:
        self.cluster_n = 1

    def _get_num_threads(self) -> int:
        return 128 if self.N <= 16384 else 256

    def _get_tv_layout(
        self, num_copy_bits: int = 128
    ) -> tuple[cute.Shape, cute.Layout]:
        """Return (tiler_mn, tv_layout) for SM100 reduction kernels.

        This intentionally mirrors Quack's `ReductionBase._get_tiled_copy(...)`:
        - `tiler_mn` spans the full N range for the CTA, including any "K-loop"
          repeats (`num_blocks_N`).
        - `tv_layout` is the *tiled* thread/value layout used by CuTe's copy
          partitioning (does **not** bake in `num_blocks_N`), matching
          `quack.copy_utils.tiled_copy_2d(...).layout_tv_tiled`.
        """
        if num_copy_bits > 128:
            raise ValueError(
                f"num_copy_bits={num_copy_bits} exceeds 128b; Quack-style SM100 reduction "
                "tiling assumes <=128b vectorization (cp.async and common CopyAtoms)."
            )
        vecsize = num_copy_bits // self.dtype.width
        assert self.N % vecsize == 0, (  # noqa: S101
            f"Input N {self.N} is not divisible by vector size {vecsize}"
        )
        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0  # noqa: S101

        threads_per_row = self._calculate_threads_per_row()
        self._set_cluster_n()
        num_blocks_N = cute.ceil_div(
            self.N // vecsize, threads_per_row * self.cluster_n
        )
        cols_per_block = num_threads // threads_per_row
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)

        # Construct the same tv layout that Quack gets from `tiled_copy_2d(...).layout_tv_tiled`.
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=num_copy_bits,
        )
        thr_layout = cute.make_ordered_layout(
            (cols_per_block, threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, vecsize))
        tv_layout = cute.make_tiled_copy_tv(
            copy_atom, thr_layout, val_layout
        ).layout_tv_tiled
        return tiler_mn, tv_layout

    def _smem_size_in_bytes(self, tiler_mn, num_warps: int) -> int:
        # Mirror the allocation order used by the SM100 reduction kernels:
        #   1) sX (byte_alignment=16)
        #   2) reduction_buffer (byte_alignment=8)
        #   3) mbar_ptr (Int64, 8B)
        #
        # CuTeDSL's SmemAllocator may insert padding between allocations to satisfy
        # alignment. Be conservative and round up offsets accordingly so we never
        # under-allocate dynamic shared memory.

        def _align_up(x: int, align: int) -> int:
            return ((x + align - 1) // align) * align

        sx_bytes = int(cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)))
        reduction_bytes = int(
            self.stage * num_warps * self.cluster_n * (self.reduction_dtype.width // 8)
        )
        mbar_bytes = int(self.stage * (cutlass.Int64.width // 8))

        offset = _align_up(sx_bytes, 16)
        offset = _align_up(offset, 8) + reduction_bytes
        offset = _align_up(offset, 8) + mbar_bytes
        return int(offset)

    def _get_reduction_buffer_layout(
        self, tv_layout: cute.Layout, cluster_n: int
    ) -> cute.Layout:
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps
            if cutlass.const_expr(cute.rank(tv_layout.shape[0]) == 1)
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(
        self,
        smem: cutlass.utils.SmemAllocator,
        tv_layout: cute.Layout,
        is_persistent: bool = False,
    ) -> tuple[cute.Tensor, cute.Pointer | None]:
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=8,
        )
        if cutlass.const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(
                cutlass.Int64,
                num_elems=self.stage if not is_persistent else self.stage * 2,
            )
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _initialize_cluster(
        self,
        tidx: cutlass.Int32,
        mbar_ptr: cute.Pointer | None,
        num_warps: int,
        is_persistent: bool = False,
    ) -> None:
        if cutlass.const_expr(self.cluster_n > 1 and mbar_ptr is not None):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                if cutlass.const_expr(is_persistent):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.stage + tidx,
                        num_warps * self.cluster_n,
                    )
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()


# -------------------------
# RMSNorm backward base (from quack.rmsnorm.RMSNormBackward)
# -------------------------


class RMSNormBackward(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int):
        # 2 stages for double buffering when computing mean of x_hat * wdy
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        self.reload_wdy = None if N <= 16 * 1024 else "smem"
        # Optional optimization: atomically accumulate mdW into a single (N,)
        # buffer instead of writing an (sm_count, N) partial buffer + torch.sum.
        self.atomic_dw = False
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            raise ValueError(
                "RMSNormBackward does not support N > 128k with dtype >= 32 bits"
            )

    def _get_num_threads(self) -> int:
        return 128 if self.N <= 4096 else 256

    def _calculate_threads_per_row(self) -> int:
        N = self.N
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (
                    32
                    if N <= 256
                    else (64 if N <= 512 else (128 if N <= 4096 else 256))
                )
            )
        )

    def _set_cluster_n(self) -> None:
        N = self.N
        cluster_n = (
            1
            if N <= 8 * 1024
            else (
                2
                if N <= 16 * 1024
                else (4 if N <= 32 * 1024 else (8 if N <= 64 * 1024 else 16))
            )
        )
        self.cluster_n = cluster_n

    def _smem_size_in_bytes(self, tiler_mn, num_warps: int, do_dtype=None) -> int:
        if do_dtype is None:
            do_dtype = self.dtype
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2
            + cute.size_in_bytes(do_dtype, cute.make_layout(tiler_mn)) * 2
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8) * 2
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mdResO: cute.Tensor | None,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor | None,
        mdRes: cute.Tensor | None,
        mdB: cute.Tensor | None,
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        semistatic_shape = (*mX.shape[:-1], self.N)

        def new_stride(t):
            return (
                cute.assume(t.stride[0], divby=128 // t.element_type.width),
                t.stride[1],
            )

        mX, mdO, mdResO, mdX, mdRes = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            if const_expr(t is not None)
            else None
            for t in (mX, mdO, mdResO, mdX, mdRes)
        ]
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                mX.element_type.width,
                mW.element_type.width if mW is not None else 0,
                mdO.element_type.width,
                mdX.element_type.width,
                mdResO.element_type.width if mdResO is not None else 0,
                mdRes.element_type.width if mdRes is not None else 0,
            )
        )
        # Quack-style policy: cap the *largest* dtype to 128b, then scale the
        # activation copy width down proportionally (e.g. fp16 + fp32-weight
        # => 64b activation vectors so the fp32 path stays at 128b).
        num_copy_bits = const_expr(128 // largest_dtype_width * mX.element_type.width)
        tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=int(num_copy_bits))
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        if const_expr(mW is not None):
            mW_expanded_layout = cute.prepend(
                mW.layout,
                cute.make_layout((tiler_mn[0],), stride=(0,)),
            )
            mW = cute.make_tensor(mW.iterator, mW_expanded_layout)

        # NOTE: `sm_count` is the launch grid x-dimension. When `cluster_n > 1`,
        # this is effectively a "cluster count" heuristic (total CTAs =
        # sm_count * cluster_n), matching Quack's naming/launch shape.
        num_blocks = sm_count
        kernel = (
            self.kernel(
                mX, mW, mdO, mdResO, mRstd, mdX, mdW, mdB, mdRes, tv_layout, tiler_mn
            )
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(mX, mW, mdO, mdResO, mRstd, mdX, mdW, mdB, mdRes)
        )
        kernel.launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            smem=self._smem_size_in_bytes(
                tiler_mn, num_warps, do_dtype=mdO.element_type
            ),
            stream=stream,
        )

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mdResO: cute.Tensor | None,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor | None,
        mdB: cute.Tensor | None,
        mdRes: cute.Tensor | None,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        shape = mX.shape
        M = shape[0]
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout(
            (tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2)
        )
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem,
            tv_layout,
            is_persistent=True,
        )
        if const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        num_copy_elems_X = (
            tv_layout.shape[1]
            if cutlass.const_expr(cute.rank(tv_layout.shape[1]) == 1)
            else tv_layout.shape[1][0]
        )
        threads_per_row = (
            tv_layout.shape[0]
            if cutlass.const_expr(cute.rank(tv_layout.shape[0]) == 1)
            else tv_layout.shape[0][0]
        )
        copy_atom_load_X = get_copy_atom(
            mX.element_type, num_copy_elems_X, is_async=False
        )
        thr_layout = cute.make_ordered_layout(
            (tiler_mn[0], threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, num_copy_elems_X))
        thr_copy_X = cute.make_tiled_copy_tv(
            copy_atom_load_X, thr_layout, val_layout
        ).get_slice(tidx)
        copy_fn = partial(copy, num_copy_elems=num_copy_elems_X)

        gX, gdO, gdResO, gdX, gdRes, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y)) if mT is not None else None
            for mT in (mX, mdO, mdResO, mdX, mdRes, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None
        gdW, gdB = [
            cute.local_tile(mT, (1, tiler_mn[1]), (bidx_start, cluster_y))
            if const_expr(mT is not None)
            else None
            for mT in (mdW, mdB)
        ]

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdO = thr_copy_X.partition_S(gdO)
        tXsdO = thr_copy_X.partition_D(sdO)
        tXgdX = thr_copy_X.partition_D(gdX)
        if const_expr(mdResO is not None):
            tXgdResO = thr_copy_X.partition_S(gdResO)
        if const_expr(mdRes is not None):
            tXgdRes = thr_copy_X.partition_D(gdRes)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdO, tXrdX = [
            cute.make_fragment_like(thr[None, None, None, 0])
            for thr in (tXgX, tXgdO, tXgdX)
        ]
        tXrdResO = None
        if const_expr(mdResO is not None):
            tXrdResO = cute.make_fragment_like(tXgdResO[None, None, None, 0])
        tXrdRes = None
        if const_expr(mdRes is not None):
            tXrdRes = cute.make_fragment_like(tXgdRes[None, None, None, 0])

        tXpX = (
            predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
            if not is_even_N
            else None
        )

        tXgdW, tXrdW = None, None
        tXgdB, tXrdB = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            tXrdW = cute.make_fragment_like(tXgdW, Float32)
        if const_expr(mdB is not None):
            tXgdB = thr_copy_X.partition_S(gdB)
            tXrdB = cute.make_fragment_like(tXgdB, Float32)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE

        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            if not is_even_N:
                tXrW.fill(0.0)
            copy_fn(tXgW, tXrW, pred=tXpX)

        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            tXgX_cur = coord_offset_i64(bidx_start, tXgX, dim=3)[None, None, None, 0]
            tXgdO_cur = coord_offset_i64(bidx_start, tXgdO, dim=3)[None, None, None, 0]
            copy_fn(tXgX_cur, tXsX[None, None, None, 0], pred=tXpX, is_async=True)
            copy_fn(tXgdO_cur, tXsdO[None, None, None, 0], pred=tXpX, is_async=True)
        elif tiler_mn[0] > 1:
            fill_oob(tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero)
            fill_oob(tXsdO[None, None, None, 0], None, fill_value=mdO.element_type.zero)
        cute.arch.cp_async_commit_group()

        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        if const_expr(mdW is not None):
            tXrdW.fill(0.0)
        if const_expr(mdB is not None):
            tXrdB.fill(0.0)
        stage = Int32(0)
        producer_phase = Int32(1)
        consumer_phase = Int32(0)
        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            if row + gdim * tiler_mn[0] < M:
                tXgX_cur = coord_offset_i64(bidx + gdim, tXgX, dim=3)[
                    None, None, None, 0
                ]
                tXgdO_cur = coord_offset_i64(bidx + gdim, tXgdO, dim=3)[
                    None, None, None, 0
                ]
                copy_fn(
                    tXgX_cur,
                    tXsX[None, None, None, stage ^ 1],
                    pred=tXpX,
                    is_async=True,
                )
                copy_fn(
                    tXgdO_cur,
                    tXsdO[None, None, None, stage ^ 1],
                    pred=tXpX,
                    is_async=True,
                )
            elif tiler_mn[0] > 1:
                fill_oob(
                    tXsX[None, None, None, stage ^ 1],
                    None,
                    fill_value=mX.element_type.zero,
                )
                fill_oob(
                    tXsdO[None, None, None, stage ^ 1],
                    None,
                    fill_value=mdO.element_type.zero,
                )
            cute.arch.cp_async_commit_group()
            rstd_val = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                rstd_val = mRstd[row]
            if const_expr(mdResO is not None):
                tXgdResO_cur = coord_offset_i64(bidx, tXgdResO, dim=3)[
                    None, None, None, 0
                ]
                if row < M or tiler_mn[0] == 1:
                    copy_fn(tXgdResO_cur, tXrdResO, pred=tXpX)
                elif tiler_mn[0] > 1:
                    tXrdResO.fill(0.0)
            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            x_hat = x * rstd_val
            wdy = dout
            if const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)
            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
            mean_xhat_wdy = (
                row_reduce_add(
                    x_hat * wdy,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                cute.arch.sync_warp()
                lane_idx = cute.arch.lane_idx()
                if lane_idx < self.cluster_n:
                    cute.arch.mbarrier_arrive(
                        mbar_empty_ptr + stage,
                        peer_cta_rank_in_cluster=lane_idx,
                    )

            if const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
                dout = tXrdO.load().to(cute.Float32)
                wdy = dout
                if const_expr(mW is not None):
                    wdy *= tXrW.load().to(Float32)

            dx = (wdy - x_hat * mean_xhat_wdy) * rstd_val
            if const_expr(mdResO is not None):
                dx += tXrdResO.load().to(cute.Float32)
            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                tXgdX_cur = coord_offset_i64(bidx, tXgdX, dim=3)[None, None, None, 0]
                copy_fn(tXrdX, tXgdX_cur, pred=tXpX)
            if const_expr(mdRes is not None):
                tXrdRes.store(dx.to(tXrdRes.element_type))
                tXgdRes_cur = coord_offset_i64(bidx, tXgdRes, dim=3)[
                    None, None, None, 0
                ]
                if row < M or tiler_mn[0] == 1:
                    copy_fn(tXrdRes, tXgdRes_cur, pred=tXpX)
            if const_expr(mdW is not None):
                tXrdW.store(tXrdW.load() + dout * x_hat)
            if const_expr(mdB is not None):
                tXrdB.store(tXrdB.load() + dout)

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        if const_expr(tiler_mn[0] > 1):
            if const_expr(mdW is not None):
                sdW = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdW = thr_copy_X.partition_D(sdW)
                cute.arch.barrier()
                row0 = tXcX[None, None, None, 0][0][0]
                if row0 > 0:
                    cute.autovec_copy(tXrdW, tXsdW)
                cute.arch.barrier()
                if row0 == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdW_other = cute.make_fragment_like(tXrdW)
                        tXsdW_other = cute.make_tensor(
                            tXsdW.iterator + i * sdW.stride[0],
                            tXsdW.layout,
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    if const_expr(self.atomic_dw):
                        atomic_add_tensor_f32(tXrdW, tXgdW, pred=tXpX)
                    else:
                        copy_fn(tXrdW, tXgdW, pred=tXpX)
                cute.arch.barrier()
            if const_expr(mdB is not None):
                sdB = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdB = thr_copy_X.partition_D(sdB)
                cute.arch.barrier()
                row0 = tXcX[None, None, None, 0][0][0]
                if row0 > 0:
                    cute.autovec_copy(tXrdB, tXsdB)
                cute.arch.barrier()
                if row0 == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdB_other = cute.make_fragment_like(tXrdB)
                        tXsdB_other = cute.make_tensor(
                            tXsdB.iterator + i * sdB.stride[0],
                            tXsdB.layout,
                        )
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy_fn(tXrdB, tXgdB, pred=tXpX)
        else:
            if const_expr(mdW is not None):
                if const_expr(self.atomic_dw):
                    atomic_add_tensor_f32(tXrdW, tXgdW, pred=tXpX)
                else:
                    copy_fn(tXrdW, tXgdW, pred=tXpX)
            if const_expr(mdB is not None):
                copy_fn(tXrdB, tXgdB, pred=tXpX)

        if const_expr(self.cluster_n > 1):
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor | None,
            mdO: cute.Tensor,
            mdResO: cute.Tensor | None,
            mRstd: cute.Tensor,
            mdX: cute.Tensor,
            mdW: cute.Tensor | None,
            mdB: cute.Tensor | None,
            mdRes: cute.Tensor | None,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ):
            self._kernel_impl(
                mX,
                mW,
                mdO,
                mdResO,
                mRstd,
                mdX,
                mdW,
                mdB,
                mdRes,
                tv_layout,
                tiler_mn,
            )
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor | None,
            mdO: cute.Tensor,
            mdResO: cute.Tensor | None,
            mRstd: cute.Tensor,
            mdX: cute.Tensor,
            mdW: cute.Tensor | None,
            mdB: cute.Tensor | None,
            mdRes: cute.Tensor | None,
        ):
            largest_dtype_width = const_expr(
                max(
                    mX.element_type.width,
                    mdO.element_type.width,
                    mdX.element_type.width,
                    mdResO.element_type.width if mdResO is not None else 0,
                    mdRes.element_type.width if mdRes is not None else 0,
                )
            )
            tiler_mn, tv_layout = self._get_tv_layout(
                num_copy_bits=128 // largest_dtype_width * mX.element_type.width
            )
            self._kernel_impl(
                mX,
                mW,
                mdO,
                mdResO,
                mRstd,
                mdX,
                mdW,
                mdB,
                mdRes,
                tv_layout,
                tiler_mn,
            )


# -------------------------
# SM count helper (from quack.rmsnorm._get_sm_count)
# -------------------------


def get_sm_count(
    N: int,
    device: torch.device,
    M: int | None = None,
    dtype: torch.dtype | None = None,
) -> int:
    """
    SM count heuristic for reduction-style kernels.

    This starts from Quack's _get_sm_count policy and layers on SM100 /
    DSv3-specific tuning so that:
      - For DSv3-style shapes (large-M, N in {6144, 8192}, fp16/bf16),
        sm_count is reduced for very large M to cut down the number of
        dw_partial/db_partial rows that ever hit HBM.
      - For Quack-suite hidden=4096, small-M shapes, sm_count is modestly
        increased to improve SM occupancy, matching the existing SM100
        tuning used by both RMSNorm and LayerNorm.
    """
    num_sms = get_num_sms(device)

    sm_count_multiple = (
        16
        if N <= 256
        else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    sm_count = num_sms
    if N <= 8192:
        sm_count = sm_count * sm_count_multiple
    elif N <= 16384:
        sm_count = sm_count // 2
    else:
        sm_count = sm_count * 2

    # Quack-suite tuning: for small-M, hidden=4096 shapes (M<=8192) and
    # 16-bit dtypes, increase sm_count to improve occupancy. This mirrors
    # the existing SM100 RMSNorm/LayerNorm heuristics.
    if (
        dtype in (torch.float16, torch.bfloat16)
        and M is not None
        and M <= 8192
        and N == 4096
    ):
        sm_count = min(sm_count * 2, num_sms * 4)

    return sm_count
