# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_hirsheybar/n6/cn6nknfxnlbi7ppj5yaegpbxlzaej6wchlkou5ftk4wtscxpoqou.py
# Topologically Sorted Source Nodes: [rms_norm, output], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.clamp, aten.reciprocal, aten.log2, aten.floor, aten.exp2]
# Source node to ATen node mapping:
#   output => abs_1, amax, clamp_max, clamp_min, clamp_min_1, convert_element_type_2, convert_element_type_3, convert_element_type_4, convert_element_type_5, exp2, floor, log2, mul_2, mul_3, reciprocal, reciprocal_2
#   rms_norm => add, convert_element_type, convert_element_type_1, mean, mul, mul_1, pow_1, rsqrt
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 9.999999747378752e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_3), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_1,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_1, [-1], True), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax, torch.float64), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 1e-12), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 448.0), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float32), kwargs = {})
#   %log2 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_3,), kwargs = {})
#   %floor : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2,), kwargs = {})
#   %exp2 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor,), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.float32), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %exp2), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_3, -448.0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 448.0), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_2 : [num_users=4] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_1,), kwargs = {})
triton_per_fused__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_0 = async_compile.triton('triton_per_fused__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr2': '*fp8e4nv', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
# danvm: in_ptr0/primals_1 = input tensor (https://github.com/pytorch/pytorch/blob/05236b5045340f8e0ca68bb996d81d1dd7d72d10/aten/src/ATen/native/layer_norm.cpp#L265)
# danvm: in_ptr1/primals_3 = weight_opt (https://github.com/pytorch/pytorch/blob/05236b5045340f8e0ca68bb996d81d1dd7d72d10/aten/src/ATen/native/layer_norm.cpp#L267)
# danvmn: out_ptr2/buf2 = "A tensor" in scaled_mm projection of K
# danvm: out_ptr3/buf4 = "B tensor"  in scaled_mm projection of K
@triton.jit
def triton_per_fused__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_0(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex

    # danvm: load in input block
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None).to(tl.float32)

    # danvm: load in weight_opt as fp32
    tmp12 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)

    # cast input block to fp32 again?
    tmp1 = tmp0.to(tl.float32)

    # square the input block
    tmp2 = tmp1 * tmp1

    # sum squares along dim 0
    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))

    # compute mean-square by dividing by num elems 
    tmp6 = 256.0
    tmp7 = (tmp5 / tmp6)

    # add epsilon (still fp32)
    tmp8 = 9.999999747378752e-06
    tmp9 = tmp7 + tmp8

    # reciprocal sqrt of mean-square 
    tmp10 = libdevice.rsqrt(tmp9)

    # normalize input block by multiplying by reciprocal sqrt of mean square
    # this is fp32 * fp32
    tmp11 = tmp1 * tmp10

    # convert to fp32 again?
    tmp13 = tmp12.to(tl.float32)
    
    # apply weight_opt to normalized values (fp32 * fp32)
    tmp14 = tmp11 * tmp13

    # convert normalized values to fp32 yet again
    tmp15 = tmp14.to(tl.float32)

    # now convert normalized values to bf16
    tmp16 = tmp15.to(tl.bfloat16)

    # now convert normalized values back to fp32??
    tmp17 = tmp16.to(tl.float32)

    # now we begin float8 conversion by taking abs of normalized values
    tmp18 = tl_math.abs(tmp17)

    # convert abs normalized values to bf16
    tmp19 = tmp18.to(tl.bfloat16)

    # convert abs normalized values back to fp32 :D
    tmp20 = tmp19.to(tl.float32)

    # compute amax along dim0 with max2
    tmp21 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp23 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp21, 0))

    # convert original normalized values from fp32 -> fp32
    tmp24 = tmp15.to(tl.float32)

    # convert amaxes from fp32 -> fp64
    tmp25 = tmp23.to(tl.float64)

    # create eps
    tmp26 = tl.full([1], 1e-12, tl.float64)

    # apply max(amax, eps) to ensure amaxes are atleast eps
    tmp27 = triton_helpers.maximum(tmp25, tmp26) # float64

    # reciprocal of amax
    tmp28 = tl.full([1], 1, tl.int32)
    tmp29 = (tmp28 / tmp27) 

    # scale =  multiply reciprocal of amax by 448 (max float8 value)
    # (aka, divide max float8 value by amax, like here: https://github.com/pytorch/ao/blob/main/torchao/float8/float8_utils.py#L47)
    tmp30 = tl.full([1], 448.0, tl.float64)
    tmp31 = tmp29 * tmp30

    # convert scale from fp64 -> fp32
    tmp32 = tmp31.to(tl.float32)

    # round scales to power of 2: https://github.com/pytorch/ao/blob/a81322e4f58b828c5ddf42623fce809d170e4caa/torchao/float8/float8_utils.py#L241
    # exp2(floor(log2(scale)))
    tmp33 = libdevice.log2(tmp32)
    tmp34 = libdevice.floor(tmp33)
    tmp35 = libdevice.exp2(tmp34)
    tmp36 = tmp24 * tmp35 # danvm: apply scale
    tmp37 = -448.0
    tmp38 = triton_helpers.maximum(tmp36, tmp37) # danvm: clamp floor to -448
    tmp39 = 448.0
    tmp40 = triton_helpers.minimum(tmp38, tmp39) # danvm: clamp ceiling to 448
    tmp41 = tmp40.to(tl.float8e4nv) # danvm: conversion to float8
    tmp42 = (tmp28 / tmp35)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp41, None) # danvm: "A tensor" of K projection
    tl.store(out_ptr3 + (x0), tmp42, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/35/c35ot3zvtvg2jgmzk7tdlc7scnt37v4k6jv2mqt4zcszd3pyf2pk.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
#   output => _scaled_mm, abs_2, amax_1, clamp_max_1, clamp_min_2, clamp_min_3, convert_element_type_6, convert_element_type_7, convert_element_type_8, convert_element_type_9, exp2_1, floor_1, log2_1, mul_4, mul_5, reciprocal_1, reciprocal_3
# Graph fragment:
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%permute,), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_2, [0], True), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_1, torch.float64), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_6, 1e-12), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_2,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 448.0), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.float32), kwargs = {})
#   %log2_1 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_7,), kwargs = {})
#   %floor_1 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_1,), kwargs = {})
#   %exp2_1 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_1,), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute, torch.float32), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %exp2_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_5, -448.0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 448.0), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_1, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_3 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_1,), kwargs = {})
#   %_scaled_mm : [num_users=2] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view, %convert_element_type_9, %reciprocal_2, %reciprocal_3, None, None, torch.bfloat16, True), kwargs = {})
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_1 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_1(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl_math.abs(tmp2)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp9 = tmp0.to(tl.float32)
    tmp10 = tmp8.to(tl.float64)
    tmp11 = tl.full([1], 1e-12, tl.float64)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = (tmp13 / tmp12)
    tmp15 = tl.full([1], 448.0, tl.float64)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = libdevice.log2(tmp17)
    tmp19 = libdevice.floor(tmp18)
    tmp20 = libdevice.exp2(tmp19)
    tmp21 = tmp9 * tmp20
    tmp22 = -448.0
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = 448.0
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp25.to(tl.float8e4nv)
    tmp27 = (tmp13 / tmp20)
    tl.store(out_ptr1 + (r0_1 + 256*x0), tmp26, None)
    tl.store(out_ptr2 + (x0), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/hr/chr4hs4zac2bdunxu673zsm6k4fnubaejzfckt2f4xc7aknbcopg.py
# Topologically Sorted Source Nodes: [xq_], Original ATen: [aten.view_as_complex]
# Source node to ATen node mapping:
#   xq_ => view_as_complex
# Graph fragment:
#   %view_as_complex : [num_users=1] = call_function[target=torch.ops.aten.view_as_complex.default](args = (%view_12,), kwargs = {})
triton_poi_fused_view_as_complex_2 = async_compile.triton('triton_poi_fused_view_as_complex_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_as_complex_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_as_complex_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/wf/cwfekv75thvyiskzkt5nxupszi7c3iajufabgm5ehxvqmwa3sjg5.py
# Topologically Sorted Source Nodes: [output_3], Original ATen: [aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   output_3 => _scaled_dot_product_flash_attention
# Graph fragment:
#   %_scaled_dot_product_flash_attention : [num_users=4] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%permute_3, %permute_4, %permute_5, 0.0, True), kwargs = {scale: 0.25})
triton_poi_fused__scaled_dot_product_flash_attention_3 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/7j/c7jqsejzgrrg5uw4cmgxwew5v2xkvpu2tvfo7b5koxh6d57ql7eq.py
# Topologically Sorted Source Nodes: [output_6], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
#   output_6 => _scaled_mm_3, abs_7, amax_6, clamp_max_7, clamp_min_14, clamp_min_15, convert_element_type_34, convert_element_type_35, convert_element_type_36, convert_element_type_37, exp2_7, floor_7, log2_7, mul_18, mul_19, reciprocal_13, reciprocal_14, reciprocal_15
# Graph fragment:
#   %abs_7 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_17,), kwargs = {})
#   %amax_6 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_7, [-1], True), kwargs = {})
#   %convert_element_type_34 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_7, torch.float64), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_34, 1e-12), kwargs = {})
#   %reciprocal_13 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_14,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_13, 448.0), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18, torch.float32), kwargs = {})
#   %log2_7 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_35,), kwargs = {})
#   %floor_7 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_7,), kwargs = {})
#   %exp2_7 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_7,), kwargs = {})
#   %convert_element_type_36 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_7, torch.float32), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_36, %exp2_7), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_19, -448.0), kwargs = {})
#   %clamp_max_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 448.0), kwargs = {})
#   %convert_element_type_37 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_7, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_14 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_19,), kwargs = {})
#   %reciprocal_15 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_7,), kwargs = {})
#   %_scaled_mm_3 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_18, %convert_element_type_37, %reciprocal_14, %reciprocal_15, None, None, torch.bfloat16, True), kwargs = {})
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_4 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_4(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl_math.abs(tmp2)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp9 = tmp0.to(tl.float32)
    tmp10 = tmp8.to(tl.float64)
    tmp11 = tl.full([1], 1e-12, tl.float64)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = (tmp13 / tmp12)
    tmp15 = tl.full([1], 448.0, tl.float64)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = libdevice.log2(tmp17)
    tmp19 = libdevice.floor(tmp18)
    tmp20 = libdevice.exp2(tmp19)
    tmp21 = tmp9 * tmp20
    tmp22 = -448.0
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = 448.0
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp25.to(tl.float8e4nv)
    tmp27 = (tmp13 / tmp20)
    tl.store(out_ptr1 + (r0_1 + 256*x0), tmp26, None)
    tl.store(out_ptr2 + (x0), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/ew/cew5u24aguebcgfuwvzsy4ieq4ktelacbds7o36hkiidi3boawqp.py
# Topologically Sorted Source Nodes: [h, rms_norm_1, output_7, output_8], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.clamp, aten.reciprocal, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
#   h => add_1
#   output_7 => _scaled_mm_4, abs_9, amax_8, clamp_max_9, clamp_min_16, clamp_min_18, clamp_min_19, convert_element_type_40, convert_element_type_41, convert_element_type_42, convert_element_type_44, convert_element_type_45, convert_element_type_46, convert_element_type_47, exp2_8, exp2_9, floor_8, floor_9, log2_8, log2_9, mul_22, mul_23, mul_24, mul_25, reciprocal_16, reciprocal_17, reciprocal_18, reciprocal_19
#   output_8 => _scaled_mm_5, clamp_max_11, clamp_min_22, clamp_min_23, convert_element_type_54, convert_element_type_55, convert_element_type_56, convert_element_type_57, exp2_11, floor_11, log2_11, mul_29, mul_30, reciprocal_21, reciprocal_23
#   rms_norm_1 => add_2, convert_element_type_38, convert_element_type_39, mean_1, mul_20, mul_21, pow_2, rsqrt_1
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_20), kwargs = {})
#   %convert_element_type_38 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_38, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 9.999999747378752e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_38, %rsqrt_1), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %primals_8), kwargs = {})
#   %convert_element_type_39 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
#   %abs_9 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_39,), kwargs = {})
#   %amax_8 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_9, [-1], True), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_8, torch.float64), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_40, 1e-12), kwargs = {})
#   %reciprocal_16 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_16,), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 448.0), kwargs = {})
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.float32), kwargs = {})
#   %log2_8 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_41,), kwargs = {})
#   %floor_8 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_8,), kwargs = {})
#   %exp2_8 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_8,), kwargs = {})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_39, torch.float32), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_42, %exp2_8), kwargs = {})
#   %convert_element_type_44 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_9, torch.float64), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_44, 1e-12), kwargs = {})
#   %reciprocal_17 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_18,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_17, 448.0), kwargs = {})
#   %convert_element_type_45 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.float32), kwargs = {})
#   %log2_9 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_45,), kwargs = {})
#   %floor_9 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_9,), kwargs = {})
#   %exp2_9 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_9,), kwargs = {})
#   %convert_element_type_46 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_8, torch.float32), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_46, %exp2_9), kwargs = {})
#   %clamp_min_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_25, -448.0), kwargs = {})
#   %clamp_max_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 448.0), kwargs = {})
#   %convert_element_type_47 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_9, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_18 : [num_users=3] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_22,), kwargs = {})
#   %reciprocal_19 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_9,), kwargs = {})
#   %_scaled_mm_4 : [num_users=2] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_21, %convert_element_type_47, %reciprocal_18, %reciprocal_19, None, None, torch.bfloat16, True), kwargs = {})
#   %convert_element_type_54 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_11, torch.float64), kwargs = {})
#   %clamp_min_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_54, 1e-12), kwargs = {})
#   %reciprocal_21 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_22,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_21, 448.0), kwargs = {})
#   %convert_element_type_55 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_29, torch.float32), kwargs = {})
#   %log2_11 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_55,), kwargs = {})
#   %floor_11 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_11,), kwargs = {})
#   %exp2_11 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_11,), kwargs = {})
#   %convert_element_type_56 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_9, torch.float32), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_56, %exp2_11), kwargs = {})
#   %clamp_min_23 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_30, -448.0), kwargs = {})
#   %clamp_max_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_23, 448.0), kwargs = {})
#   %convert_element_type_57 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_11, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_23 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_11,), kwargs = {})
#   %_scaled_mm_5 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_21, %convert_element_type_57, %reciprocal_18, %reciprocal_23, None, None, torch.bfloat16, True), kwargs = {})
triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_5 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr3': '*fp8e4nv', 'out_ptr4': '*fp8e4nv', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_5(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 256*x0), None).to(tl.float32)
    tmp20 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp2 + tmp5
    tmp7 = tmp6.to(tl.bfloat16)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 256.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 9.999999747378752e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp9 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23.to(tl.bfloat16)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tl_math.abs(tmp25)
    tmp27 = tmp26.to(tl.bfloat16)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [R0_BLOCK])
    tmp31 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp29, 0))
    tmp32 = tmp23.to(tl.float32)
    tmp33 = tmp31.to(tl.float64)
    tmp34 = tl.full([1], 1e-12, tl.float64)
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = (tmp36 / tmp35)
    tmp38 = tl.full([1], 448.0, tl.float64)
    tmp39 = tmp37 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = libdevice.log2(tmp40)
    tmp42 = libdevice.floor(tmp41)
    tmp43 = libdevice.exp2(tmp42)
    tmp44 = tmp32 * tmp43
    tmp45 = -448.0
    tmp46 = triton_helpers.maximum(tmp44, tmp45)
    tmp47 = 448.0
    tmp48 = triton_helpers.minimum(tmp46, tmp47)
    tmp49 = tmp48.to(tl.float8e4nv)
    tmp50 = (tmp36 / tmp43)
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp49, None)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp49, None)
    tl.store(out_ptr5 + (x0), tmp50, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/4w/c4wnbdivxzbnsirugsiyivnvklz3htxmiahgfy46aka4fmbfp5xo.py
# Topologically Sorted Source Nodes: [output_7], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
#   output_7 => _scaled_mm_4, abs_10, amax_9, clamp_max_9, clamp_min_18, clamp_min_19, convert_element_type_44, convert_element_type_45, convert_element_type_46, convert_element_type_47, exp2_9, floor_9, log2_9, mul_24, mul_25, reciprocal_17, reciprocal_19
# Graph fragment:
#   %abs_10 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%permute_8,), kwargs = {})
#   %amax_9 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_10, [0], True), kwargs = {})
#   %convert_element_type_44 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_9, torch.float64), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_44, 1e-12), kwargs = {})
#   %reciprocal_17 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_18,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_17, 448.0), kwargs = {})
#   %convert_element_type_45 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.float32), kwargs = {})
#   %log2_9 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_45,), kwargs = {})
#   %floor_9 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_9,), kwargs = {})
#   %exp2_9 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_9,), kwargs = {})
#   %convert_element_type_46 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_8, torch.float32), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_46, %exp2_9), kwargs = {})
#   %clamp_min_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_25, -448.0), kwargs = {})
#   %clamp_max_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 448.0), kwargs = {})
#   %convert_element_type_47 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_9, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_19 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_9,), kwargs = {})
#   %_scaled_mm_4 : [num_users=2] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_21, %convert_element_type_47, %reciprocal_18, %reciprocal_19, None, None, torch.bfloat16, True), kwargs = {})
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl_math.abs(tmp2)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp9 = tmp0.to(tl.float32)
    tmp10 = tmp8.to(tl.float64)
    tmp11 = tl.full([1], 1e-12, tl.float64)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = (tmp13 / tmp12)
    tmp15 = tl.full([1], 448.0, tl.float64)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = libdevice.log2(tmp17)
    tmp19 = libdevice.floor(tmp18)
    tmp20 = libdevice.exp2(tmp19)
    tmp21 = tmp9 * tmp20
    tmp22 = -448.0
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = 448.0
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp25.to(tl.float8e4nv)
    tmp27 = (tmp13 / tmp20)
    tl.store(out_ptr1 + (r0_1 + 256*x0), tmp26, None)
    tl.store(out_ptr2 + (x0), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/tw/ctwqoaujpgozuaxebqeuuqyhlzmqfjzt2onb2hsidj3l6nqrr7dj.py
# Topologically Sorted Source Nodes: [silu, mul_2, output_9], Original ATen: [aten.silu, aten.mul, aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
#   mul_2 => mul_31
#   output_9 => _scaled_mm_6, abs_13, amax_12, clamp_max_12, clamp_max_13, clamp_min_24, clamp_min_25, clamp_min_26, clamp_min_27, convert_element_type_58, convert_element_type_59, convert_element_type_60, convert_element_type_62, convert_element_type_63, convert_element_type_64, convert_element_type_65, exp2_12, exp2_13, floor_12, floor_13, log2_12, log2_13, mul_32, mul_33, mul_34, mul_35, reciprocal_24, reciprocal_25, reciprocal_26, reciprocal_27
#   silu => convert_element_type_48, convert_element_type_49, mul_26, sigmoid
# Graph fragment:
#   %convert_element_type_48 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_23, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_48,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_48, %sigmoid), kwargs = {})
#   %convert_element_type_49 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_26, torch.bfloat16), kwargs = {})
#   %mul_31 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_49, %view_26), kwargs = {})
#   %abs_13 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_31,), kwargs = {})
#   %amax_12 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_13, [-1], True), kwargs = {})
#   %convert_element_type_58 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_12, torch.float64), kwargs = {})
#   %clamp_min_24 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_58, 1e-12), kwargs = {})
#   %reciprocal_24 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_24,), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_24, 448.0), kwargs = {})
#   %convert_element_type_59 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_32, torch.float32), kwargs = {})
#   %log2_12 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_59,), kwargs = {})
#   %floor_12 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_12,), kwargs = {})
#   %exp2_12 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_12,), kwargs = {})
#   %convert_element_type_60 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_31, torch.float32), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_60, %exp2_12), kwargs = {})
#   %clamp_min_25 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_33, -448.0), kwargs = {})
#   %clamp_max_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_25, 448.0), kwargs = {})
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_13, torch.float64), kwargs = {})
#   %clamp_min_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_62, 1e-12), kwargs = {})
#   %reciprocal_25 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_26,), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_25, 448.0), kwargs = {})
#   %convert_element_type_63 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_34, torch.float32), kwargs = {})
#   %log2_13 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_63,), kwargs = {})
#   %floor_13 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_13,), kwargs = {})
#   %exp2_13 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_13,), kwargs = {})
#   %convert_element_type_64 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_10, torch.float32), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, %exp2_13), kwargs = {})
#   %clamp_min_27 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_35, -448.0), kwargs = {})
#   %clamp_max_13 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_27, 448.0), kwargs = {})
#   %convert_element_type_65 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_13, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_26 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_28,), kwargs = {})
#   %reciprocal_27 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_13,), kwargs = {})
#   %_scaled_mm_6 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_27, %convert_element_type_65, %reciprocal_26, %reciprocal_27, None, None, torch.bfloat16, True), kwargs = {})
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_silu_7 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_silu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr2': '*fp8e4nv', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_silu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_silu_7(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4.to(tl.bfloat16)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6.to(tl.bfloat16)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp9.to(tl.bfloat16)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12.to(tl.bfloat16)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.bfloat16)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl_math.abs(tmp16)
    tmp18 = tmp17.to(tl.bfloat16)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, float("-inf"))
    tmp23 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp22, 0))
    tmp24 = tmp14.to(tl.float32)
    tmp25 = tmp23.to(tl.float64)
    tmp26 = tl.full([1], 1e-12, tl.float64)
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp28 = tl.full([1], 1, tl.int32)
    tmp29 = (tmp28 / tmp27)
    tmp30 = tl.full([1], 448.0, tl.float64)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = libdevice.log2(tmp32)
    tmp34 = libdevice.floor(tmp33)
    tmp35 = libdevice.exp2(tmp34)
    tmp36 = tmp24 * tmp35
    tmp37 = -448.0
    tmp38 = triton_helpers.maximum(tmp36, tmp37)
    tmp39 = 448.0
    tmp40 = triton_helpers.minimum(tmp38, tmp39)
    tmp41 = tmp40.to(tl.float8e4nv)
    tmp42 = (tmp28 / tmp35)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp41, r0_mask)
    tl.store(out_ptr3 + (x0), tmp42, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/gq/cgq6tobxquckdz55hmetwcth4vrnqlowhvu4rm26sbuxf5scqsrj.py
# Topologically Sorted Source Nodes: [output_9], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
#   output_9 => _scaled_mm_6, abs_14, amax_13, clamp_max_13, clamp_min_26, clamp_min_27, convert_element_type_62, convert_element_type_63, convert_element_type_64, convert_element_type_65, exp2_13, floor_13, log2_13, mul_34, mul_35, reciprocal_25, reciprocal_26, reciprocal_27
# Graph fragment:
#   %abs_14 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%permute_10,), kwargs = {})
#   %amax_13 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_14, [0], True), kwargs = {})
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_13, torch.float64), kwargs = {})
#   %clamp_min_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_62, 1e-12), kwargs = {})
#   %reciprocal_25 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_26,), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_25, 448.0), kwargs = {})
#   %convert_element_type_63 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_34, torch.float32), kwargs = {})
#   %log2_13 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_63,), kwargs = {})
#   %floor_13 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_13,), kwargs = {})
#   %exp2_13 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_13,), kwargs = {})
#   %convert_element_type_64 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_10, torch.float32), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, %exp2_13), kwargs = {})
#   %clamp_min_27 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_35, -448.0), kwargs = {})
#   %clamp_max_13 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_27, 448.0), kwargs = {})
#   %convert_element_type_65 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_13, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_26 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_28,), kwargs = {})
#   %reciprocal_27 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_13,), kwargs = {})
#   %_scaled_mm_6 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_27, %convert_element_type_65, %reciprocal_26, %reciprocal_27, None, None, torch.bfloat16, True), kwargs = {})
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl_math.abs(tmp2)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, float("-inf"))
    tmp9 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp8, 0))
    tmp10 = tmp0.to(tl.float32)
    tmp11 = tmp9.to(tl.float64)
    tmp12 = tl.full([1], 1e-12, tl.float64)
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = (tmp14 / tmp13)
    tmp16 = tl.full([1], 448.0, tl.float64)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = libdevice.log2(tmp18)
    tmp20 = libdevice.floor(tmp19)
    tmp21 = libdevice.exp2(tmp20)
    tmp22 = tmp10 * tmp21
    tmp23 = -448.0
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp25 = 448.0
    tmp26 = triton_helpers.minimum(tmp24, tmp25)
    tmp27 = tmp26.to(tl.float8e4nv)
    tmp28 = (tmp14 / tmp21)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp27, r0_mask)
    tl.store(out_ptr2 + (x0), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/zi/czigypbwjnczaw32la27bahkiuyfgqqbct3kknbw7rnmem32n6ln.py
# Topologically Sorted Source Nodes: [h, out], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   h => add_1
#   out => add_3
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_20), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_29), kwargs = {})
triton_poi_fused_add_9 = async_compile.triton('triton_poi_fused_add_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_9(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp2 + tmp5
    tmp7 = tmp6.to(tl.bfloat16)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.bfloat16)
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp11.to(tl.bfloat16)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 + tmp13
    tmp15 = tmp14.to(tl.bfloat16)
    tmp16 = tmp15.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp16, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (8, 2048, 256), (524288, 256, 1))
    assert_size_stride(primals_2, (2048, 8), (8, 1))
    assert_size_stride(primals_3, (256, ), (1, ))
    assert_size_stride(primals_4, (256, 256), (256, 1))
    assert_size_stride(primals_5, (256, 256), (256, 1))
    assert_size_stride(primals_6, (256, 256), (256, 1))
    assert_size_stride(primals_7, (256, 256), (256, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (768, 256), (256, 1))
    assert_size_stride(primals_10, (768, 256), (256, 1))
    assert_size_stride(primals_11, (256, 768), (768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # danvm: buf2 becomes "A" tensor in K projection
        buf2 = empty_strided_cuda((8, 2048, 256), (524288, 256, 1), torch.float8_e4m3fn)
        # danvm: buf4 becomes "B" tensor in K projection 
        buf4 = empty_strided_cuda((16384, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rms_norm, output], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.clamp, aten.reciprocal, aten.log2, aten.floor, aten.exp2]
        stream0 = get_raw_stream(0)
        # danvm: buf2 
        triton_per_fused__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_0.run(primals_1, primals_3, buf2, buf4, 16384, 256, stream=stream0)
        buf5 = empty_strided_cuda((256, 256), (1, 256), torch.float8_e4m3fn)
        buf6 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_1.run(primals_4, buf5, buf6, 256, 256, stream=stream0)
        buf7 = empty_strided_cuda((16384, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(reinterpret_tensor(buf2, (16384, 256), (256, 1), 0), buf5, buf4, buf6, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf7)
        buf9 = buf5; del buf5  # reuse
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_1.run(primals_5, buf9, buf10, 256, 256, stream=stream0)

        # danvm: buf11 part of K ancestry (becomes buf19), it is output of scaled_mm coming up
        buf11 = empty_strided_cuda((16384, 256), (256, 1), torch.bfloat16)

        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        # danvm: buf11 is output of scaled_mm
        extern_kernels._scaled_mm(reinterpret_tensor(buf2, (16384, 256), (256, 1), 0), buf9, buf4, buf10, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf11)
        buf13 = buf9; del buf9  # reuse
        buf14 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_1.run(primals_6, buf13, buf14, 256, 256, stream=stream0)
        buf15 = empty_strided_cuda((16384, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(reinterpret_tensor(buf2, (16384, 256), (256, 1), 0), buf13, buf4, buf14, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf15)
        buf16 = empty_strided_cuda((8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xq_], Original ATen: [aten.view_as_complex]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_as_complex_2.run(buf7, buf16, 4194304, stream=stream0)
        # Topologically Sorted Source Nodes: [xq_], Original ATen: [aten.view_as_complex]
        buf17 = torch.ops.aten.view_as_complex.default(buf16)
        buf18 = buf17
        assert_size_stride(buf18, (8, 2048, 16, 8), (262144, 128, 8, 1))

        # danvm: buf19 will be used in rope for K
        buf19 = empty_strided_cuda((8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1), torch.float32)

        # Topologically Sorted Source Nodes: [xk_], Original ATen: [aten.view_as_complex]
        stream0 = get_raw_stream(0)

        # danvm: triton_poi_fused_view_as_complex_2 converts buf11 to fp32 and writes it to buf19
        triton_poi_fused_view_as_complex_2.run(buf11, buf19, 4194304, stream=stream0)
        # Topologically Sorted Source Nodes: [xk_], Original ATen: [aten.view_as_complex]
        # danvm: buf20 is view as complex buf 19 https://github.com/pytorch/torchtitan/blob/4f532e092afbabb958cd92fc38b39c92ec9d1044/torchtitan/models/llama3/model.py#L149
        buf20 = torch.ops.aten.view_as_complex.default(buf19)

        # danvm: buf21 is operand for applying rope, comes from buf20
        buf21 = buf20
        assert_size_stride(buf21, (8, 2048, 16, 8), (262144, 128, 8, 1))
        # Topologically Sorted Source Nodes: [freqs_cis], Original ATen: [aten.slice]
        buf22 = torch.ops.aten.slice.Tensor(primals_2, 0, 0, 2048)
        buf23 = buf22
        assert_size_stride(buf23, (2048, 8), (8, 1))

        # Topologically Sorted Source Nodes: [freqs_cis_1], Original ATen: [aten.view]
        buf24 = torch.ops.aten.reshape.default(buf23, [1, 2048, 1, 8])

        # danvm: buf25 is operand for applying rope, comes from buf24
        buf25 = buf24
        assert_size_stride(buf25, (1, 2048, 1, 8), (16384, 8, 8, 1))
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        buf26 = torch.ops.aten.mul.Tensor(buf18, buf25)
        del buf16
        del buf17
        del buf18
        buf27 = buf26
        assert_size_stride(buf27, (8, 2048, 16, 8), (262144, 128, 8, 1))
        del buf26
        # Topologically Sorted Source Nodes: [view_as_real], Original ATen: [aten.view_as_real]
        buf28 = torch.ops.aten.view_as_real.default(buf27)
        buf29 = buf28
        assert_size_stride(buf29, (8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1))

        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        # (danvm): buf30 parents are buf21 * buf25 for rope: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model.py#L151-L152
        buf30 = torch.ops.aten.mul.Tensor(buf21, buf25)
        del buf19
        del buf20
        del buf21
        # (danvm) buf31 from buf30, part of K ancestry
        buf31 = buf30
        assert_size_stride(buf31, (8, 2048, 16, 8), (262144, 128, 8, 1))
        del buf30
        # Topologically Sorted Source Nodes: [view_as_real_1], Original ATen: [aten.view_as_real]
        # (danvm) buf32 = viewing buf31 as real as it was just complex from RoPE
        buf32 = torch.ops.aten.view_as_real.default(buf31)

        # (danvm) buff33 from buf32, part of K ancestry
        buf33 = buf32
        assert_size_stride(buf33, (8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1))
        buf34 = reinterpret_tensor(buf11, (8, 16, 2048, 16), (524288, 16, 256, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [output_3], Original ATen: [aten._scaled_dot_product_flash_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(buf29, buf34, 4194304, stream=stream0)
        del buf27
        del buf28
        del buf29

        # (danvm) buf35 will be fp32 K tensor in FA
        buf35 = empty_strided_cuda((8, 16, 2048, 16), (524288, 16, 256, 1), torch.bfloat16)

        # Topologically Sorted Source Nodes: [output_3], Original ATen: [aten._scaled_dot_product_flash_attention]
        stream0 = get_raw_stream(0)

        # (danvm) converts K (buf33) to fp32 and writes to buf35
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(buf33, buf35, 4194304, stream=stream0)
        del buf31
        del buf32
        del buf33

        # Topologically Sorted Source Nodes: [output_3], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf36 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf34, buf35, reinterpret_tensor(buf15, (8, 16, 2048, 16), (524288, 16, 256, 1), 0), 0.0, True, scale=0.25)
        buf37 = buf36[0]
        assert_size_stride(buf37, (8, 16, 2048, 16), (524288, 16, 256, 1))
        buf38 = buf36[1]
        assert_size_stride(buf38, (8, 16, 2048), (32768, 2048, 1))
        buf39 = buf36[6]
        assert_size_stride(buf39, (2, ), (1, ))
        buf40 = buf36[7]
        assert_size_stride(buf40, (), ())
        del buf36
        buf44 = reinterpret_tensor(buf2, (16384, 256), (256, 1), 0); del buf2  # reuse
        buf46 = empty_strided_cuda((16384, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_6], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_4.run(buf37, buf44, buf46, 16384, 256, stream=stream0)
        buf45 = buf13; del buf13  # reuse
        buf47 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [output_6], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_1.run(primals_7, buf45, buf47, 256, 256, stream=stream0)
        buf48 = reinterpret_tensor(buf35, (16384, 256), (256, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [output_6], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf44, buf45, buf46, buf47, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf48)
        del buf45
        buf54 = buf44; del buf44  # reuse
        buf59 = empty_strided_cuda((16384, 256), (256, 1), torch.float8_e4m3fn)
        buf53 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [h, rms_norm_1, output_7, output_8], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.clamp, aten.reciprocal, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_5.run(primals_1, buf48, primals_8, buf54, buf59, buf53, 16384, 256, stream=stream0)
        buf55 = empty_strided_cuda((256, 768), (1, 256), torch.float8_e4m3fn)
        buf56 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_7], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6.run(primals_9, buf55, buf56, 768, 256, stream=stream0)
        buf57 = empty_strided_cuda((16384, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [output_7], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf54, buf55, buf53, buf56, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf57)
        del buf54
        buf60 = buf55; del buf55  # reuse
        buf61 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [output_8], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6.run(primals_10, buf60, buf61, 768, 256, stream=stream0)
        buf62 = empty_strided_cuda((16384, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [output_8], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf59, buf60, buf53, buf61, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf62)
        del buf59
        del buf61
        buf66 = empty_strided_cuda((16384, 768), (768, 1), torch.float8_e4m3fn)
        buf68 = empty_strided_cuda((16384, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [silu, mul_2, output_9], Original ATen: [aten.silu, aten.mul, aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_silu_7.run(buf57, buf62, buf66, buf68, 16384, 768, stream=stream0)
        del buf62
        buf67 = reinterpret_tensor(buf60, (768, 256), (1, 768), 0); del buf60  # reuse
        buf69 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [output_9], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8.run(primals_11, buf67, buf69, 256, 768, stream=stream0)
        buf70 = reinterpret_tensor(buf34, (16384, 256), (256, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [output_9], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf66, buf67, buf68, buf69, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf70)
        del buf66
        del buf67
        del buf68
        del buf69
        buf71 = reinterpret_tensor(buf48, (8, 2048, 256), (524288, 256, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [h, out], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_9.run(buf71, primals_1, buf70, 4194304, stream=stream0)
        del buf70
    return (buf71, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, buf4, buf7, buf15, buf37, buf38, buf39, buf40, buf53, buf57, )


def benchmark_compiled_module(times=10, repeat=10):#
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, 2048, 256), (524288, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((2048, 8), (8, 1), device='cuda:0', dtype=torch.complex64)
    primals_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
