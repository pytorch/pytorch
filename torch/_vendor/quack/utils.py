# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import math
from typing import Optional, Tuple, Union

import cutlass
import cutlass.cute as cute

from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def load_scalar_or_pointer(x, dtype=Float32):
    if const_expr(isinstance(x, cute.Pointer)):
        return dtype(cute.make_tensor(x, cute.make_layout(1))[0])
    else:
        return x


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
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
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), "val must be Float32, Int32, or Int64"
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def store_shared_remote_x4(
    val0: Float32 | Int32,
    val1: Float32 | Int32,
    val2: Float32 | Int32,
    val3: Float32 | Int32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    assert isinstance(val0, (Float32, Int32)), "val must be Float32, or Int32"
    dtype = Float32 if isinstance(val0, Float32) else Int32
    suffix = {Float32: "f32", Int32: "s32"}[dtype]
    constraint = {Float32: "f", Int32: "r"}[dtype]
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            dtype(val0).ir_value(loc=loc, ip=ip),
            dtype(val1).ir_value(loc=loc, ip=ip),
            dtype(val2).ir_value(loc=loc, ip=ip),
            dtype(val3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        f".reg .v4 .{suffix} abcd;\n\t"
        f"mov.{suffix} abcd.x, $2;\n\t"
        f"mov.{suffix} abcd.y, $3;\n\t"
        f"mov.{suffix} abcd.z, $4;\n\t"
        f"mov.{suffix} abcd.w, $5;\n\t"
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.{suffix} [$0], abcd, [$1];\n\t"
        "}\n",
        f"r,r,{constraint},{constraint},{constraint},{constraint}",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def fmin(a: Union[float, Float32], b: Union[float, Float32], *, loc=None, ip=None) -> Float32:
    if cutlass.const_expr(cutlass.CUDA_VERSION.major) == 12:
        return Float32(
            nvvm.fmin(
                T.f32(),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
        )
    return Float32(
        nvvm.fmin(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sqrt(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "sqrt.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def ceil(a: float | Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rpi.ftz.s32.f32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    """Fill out-of-bounds values in shared memory tensor.

    Args:
        tXsX: Shared memory tensor to fill
        tXpX: Predicate tensor indicating valid elements
        fill_value: Value to fill OOB locations with
    """
    tXrX_fill = cute.make_rmem_tensor_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


# ---------------------------------------------------------------------------
# General-purpose DSL store / vector helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def make_vector(elem_type, *values, loc=None, ip=None):
    """Build an MLIR vector <N x elem_type> from N scalar DSL values.

    Example: make_vector(cutlass.Uint32, v0, v1) -> <2 x i32> MLIR vector
    """
    from cutlass._mlir import ir

    n = len(values)
    mlir_ty = elem_type.mlir_type
    vec_ty = ir.VectorType.get([n], mlir_ty)
    vec = llvm.mlir_undef(vec_ty, loc=loc, ip=ip)
    for i, v in enumerate(values):
        vec = vector.insertelement(
            elem_type(v).ir_value(loc=loc, ip=ip),
            vec,
            position=_arith.constant(T.i32(), i, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    return vec


@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    vec_f32x2 = vector.from_elements(
        T.vector(2, T.f32()), (a.ir_value(), b.ir_value()), loc=loc, ip=ip
    )
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2)
    res = cutlass.Int64(
        vector.extract(vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    return res


@dsl_user_op
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    vec_i64x1 = vector.from_elements(T.vector(1, T.i64()), (c.ir_value(),), loc=loc, ip=ip)
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1)
    res0 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    res1 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    )
    return res0, res1


@cute.jit
def warp_prefix_sum(val: Int32, lane: Optional[Int32] = None) -> Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


@dsl_user_op
def atomic_inc_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    from cutlass import CUDA_VERSION

    # * NVVM call based on nvvm version
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        # Old API: requires explicit result type as first positional argument
        return nvvm.atomicrmw(
            res=T.i32(), op=nvvm.AtomicOpKind.INC, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )
    else:
        # New API: infers result type automatically
        return nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.INC, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )


@dsl_user_op
def atomic_add_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    from cutlass import CUDA_VERSION

    # * NVVM call based on nvvm version
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        # Old API: requires explicit result type as first positional argument
        return nvvm.atomicrmw(
            res=T.i32(), op=nvvm.AtomicOpKind.ADD, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )
    else:
        # New API: infers result type automatically
        return nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.ADD, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )


@dsl_user_op
def issue_clc_query_nomulticast(
    mbar_ptr: cute.Pointer,
    clc_response_ptr: cute.Pointer,
    loc=None,
    ip=None,
) -> None:
    """
    The clusterlaunchcontrol.try_cancel instruction requests atomically cancelling the launch
    of a cluster that has not started running yet. It asynchronously writes an opaque response
    to shared memory indicating whether the operation succeeded or failed. On success, the
    opaque response contains the ctaid of the first CTA of the canceled cluster.

    :param mbar_ptr: A pointer to the mbarrier address in SMEM
    :type mbar_ptr:  Pointer
    :param clc_response_ptr: A pointer to the cluster launch control response address in SMEM
    :type clc_response_ptr:  Pointer
    """
    mbar_llvm_ptr = mbar_ptr.llvm_ptr
    clc_response_llvm_ptr = clc_response_ptr.llvm_ptr
    nvvm.clusterlaunchcontrol_try_cancel(
        clc_response_llvm_ptr,
        mbar_llvm_ptr,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def domain_offset_aligned(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    assert isinstance(tensor.iterator, cute.Pointer)
    # We assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        elem_pointer(tensor, coord).toint(),
        tensor.memspace,
        assumed_align=tensor.iterator.alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)
