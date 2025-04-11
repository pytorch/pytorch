# AOT ID: ['0_backward']
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


# kernel path: /tmp/torchinductor_hirsheybar/rc/crcfjugzyb7oamzjjhoylpuoqmh35g7a4efs52ztaxmiirj6m5tx.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.new_ones, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_15 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%view_30,), kwargs = {})
#   %amax_14 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_15, [-1], True), kwargs = {})
#   %convert_element_type_66 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_14, torch.float64), kwargs = {})
#   %clamp_min_28 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_66, 1e-12), kwargs = {})
#   %reciprocal_28 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_28,), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_28, 448.0), kwargs = {})
#   %convert_element_type_67 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_36, torch.float32), kwargs = {})
#   %log2_14 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_67,), kwargs = {})
#   %floor_14 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_14,), kwargs = {})
#   %exp2_14 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_14,), kwargs = {})
#   %convert_element_type_68 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_30, torch.float32), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_68, %exp2_14), kwargs = {})
#   %clamp_min_29 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_37, -448.0), kwargs = {})
#   %clamp_max_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_29, 448.0), kwargs = {})
#   %convert_element_type_69 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_14, torch.float8_e4m3fn), kwargs = {})
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %_scaled_mm_7 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_69, %permute_14, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_new_ones_reciprocal_0 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_new_ones_reciprocal_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_new_ones_reciprocal_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_new_ones_reciprocal_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel):
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
    tl.store(out_ptr1 + (r0_1 + 256*x0), tmp26, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/f3/cf3oo56xa5u3fx57bksiqf352yyb2w5epve34mrx6yf3uexp4vez.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_2, 0), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_10, %mul_39), kwargs = {})
#   %abs_16 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_4,), kwargs = {})
#   %amax_15 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_16, [-1], True), kwargs = {})
triton_red_fused_abs_add_amax_mul_1 = async_compile.triton('triton_red_fused_abs_add_amax_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_add_amax_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_add_amax_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_2 + 98304*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = 0.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp7.to(tl.bfloat16)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9.to(tl.bfloat16)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp6
        tmp13 = tmp12.to(tl.bfloat16)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14.to(tl.bfloat16)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp2 + tmp16
        tmp18 = tmp17.to(tl.bfloat16)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19.to(tl.bfloat16)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl_math.abs(tmp21)
        tmp23 = tmp22.to(tl.bfloat16)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = triton_helpers.maximum(_tmp26, tmp25)
        _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
    tmp26 = triton_helpers.max2(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/f4/cf43c47wh2nzeu2kmjoiobgugc3az4dvockfjqb7lrq3djrkpski.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_2, 0), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_10, %mul_39), kwargs = {})
#   %abs_16 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_4,), kwargs = {})
#   %amax_15 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_16, [-1], True), kwargs = {})
triton_per_fused_abs_add_amax_mul_2 = async_compile.triton('triton_per_fused_abs_add_amax_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 2},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_amax_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_amax_mul_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/xb/cxbppgd56kptuio5oj7rv2lm3oslgvmnnee53odidm6ujflizhgk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten.new_ones, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_66 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_14, torch.float64), kwargs = {})
#   %clamp_min_28 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_66, 1e-12), kwargs = {})
#   %reciprocal_28 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_28,), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_28, 448.0), kwargs = {})
#   %convert_element_type_67 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_36, torch.float32), kwargs = {})
#   %log2_14 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_67,), kwargs = {})
#   %floor_14 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_14,), kwargs = {})
#   %exp2_14 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_14,), kwargs = {})
#   %convert_element_type_68 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_30, torch.float32), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_68, %exp2_14), kwargs = {})
#   %clamp_min_29 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_37, -448.0), kwargs = {})
#   %clamp_max_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_29, 448.0), kwargs = {})
#   %convert_element_type_69 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_14, torch.float8_e4m3fn), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_2, 0), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_10, %mul_39), kwargs = {})
#   %convert_element_type_70 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_15, torch.float64), kwargs = {})
#   %clamp_min_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_70, 1e-12), kwargs = {})
#   %reciprocal_29 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_30,), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_29, 448.0), kwargs = {})
#   %convert_element_type_71 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_40, torch.float32), kwargs = {})
#   %log2_15 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_71,), kwargs = {})
#   %floor_15 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_15,), kwargs = {})
#   %exp2_15 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_15,), kwargs = {})
#   %convert_element_type_72 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_72, %exp2_15), kwargs = {})
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %_scaled_mm_7 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_69, %permute_14, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_new_ones_reciprocal_3 = async_compile.triton('triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_new_ones_reciprocal_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_new_ones_reciprocal_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_new_ones_reciprocal_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 768*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 0.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.bfloat16)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9.to(tl.bfloat16)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp6
    tmp13 = tmp12.to(tl.bfloat16)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.bfloat16)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp2 + tmp16
    tmp18 = tmp17.to(tl.bfloat16)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp21.to(tl.float64)
    tmp23 = tl.full([1, 1], 1e-12, tl.float64)
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp25 = tl.full([1, 1], 1, tl.int32)
    tmp26 = (tmp25 / tmp24)
    tmp27 = tl.full([1, 1], 448.0, tl.float64)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = libdevice.log2(tmp29)
    tmp31 = libdevice.floor(tmp30)
    tmp32 = libdevice.exp2(tmp31)
    tmp33 = tmp20 * tmp32
    tmp34 = -448.0
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp36 = 448.0
    tmp37 = triton_helpers.minimum(tmp35, tmp36)
    tmp38 = tmp37.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y0 + 256*x1), tmp38, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/6x/c6xy4j7icbhaogsqra6tmz5j22t2fqxu5g2krzto3xogbbygtuz5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.new_ones]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_new_ones_4 = async_compile.triton('triton_poi_fused_new_ones_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_ones_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_ones_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = 1.0
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/ea/ceapd2chee5pe7h326mzlg5klvukpulal5ebohhmm72dax5vzt36.py
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
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_5 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_5(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
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


# kernel path: /tmp/torchinductor_hirsheybar/uk/cukxaans3lvynj2vzmcwzkej2vrypiatz4dc2pemndcqhd6r6oy2.py
# Topologically Sorted Source Nodes: [output_6], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
#   output_6 => _scaled_mm_3, abs_8, amax_7, clamp_max_7, clamp_min_14, clamp_min_15, convert_element_type_34, convert_element_type_35, convert_element_type_36, convert_element_type_37, exp2_7, floor_7, log2_7, mul_18, mul_19, reciprocal_13, reciprocal_14, reciprocal_15
# Graph fragment:
#   %abs_8 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%permute_7,), kwargs = {})
#   %amax_7 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_8, [0], True), kwargs = {})
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
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
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


# kernel path: /tmp/torchinductor_hirsheybar/oo/cooed3hk534aiexeaoedpenpomi5w6q7rwflflnzpehaegr7tl4q.py
# Topologically Sorted Source Nodes: [h, rms_norm_1, output_7, output_8], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.clamp, aten.reciprocal, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
#   h => add_1
#   output_7 => abs_9, amax_8, clamp_min_16, convert_element_type_40, convert_element_type_41, convert_element_type_42, exp2_8, floor_8, log2_8, mul_22, mul_23, reciprocal_16
#   output_8 => _scaled_mm_5, clamp_max_11, clamp_min_22, clamp_min_23, convert_element_type_54, convert_element_type_55, convert_element_type_56, convert_element_type_57, exp2_11, floor_11, log2_11, mul_29, mul_30, reciprocal_21, reciprocal_23
#   rms_norm_1 => add_2, convert_element_type_38, convert_element_type_39, mean_1, mul_20, mul_21, pow_2, rsqrt_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_20), kwargs = {})
#   %convert_element_type_38 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_38, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 9.999999747378752e-06), kwargs = {})
#   %rsqrt_1 : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_20 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_38, %rsqrt_1), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %primals_8), kwargs = {})
#   %convert_element_type_39 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
#   %abs_9 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_39,), kwargs = {})
#   %amax_8 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_9, [-1], True), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_8, torch.float64), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_40, 1e-12), kwargs = {})
#   %reciprocal_16 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_16,), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 448.0), kwargs = {})
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.float32), kwargs = {})
#   %log2_8 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_41,), kwargs = {})
#   %floor_8 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_8,), kwargs = {})
#   %exp2_8 : [num_users=1] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_8,), kwargs = {})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_39, torch.float32), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_42, %exp2_8), kwargs = {})
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
triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_7 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr3': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr3, xnumel, r0_numel):
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
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp49, None)
    tl.store(out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/cf/ccftiqww33bgxv2yclrnftemzhdzpa6xtsrust7iiqzrnglksqg7.py
# Topologically Sorted Source Nodes: [output_8], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
#   output_8 => _scaled_mm_5, abs_12, amax_11, clamp_max_11, clamp_min_22, clamp_min_23, convert_element_type_54, convert_element_type_55, convert_element_type_56, convert_element_type_57, exp2_11, floor_11, log2_11, mul_29, mul_30, reciprocal_21, reciprocal_23
# Graph fragment:
#   %abs_12 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%permute_9,), kwargs = {})
#   %amax_11 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_12, [0], True), kwargs = {})
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
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
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


# kernel path: /tmp/torchinductor_hirsheybar/b3/cb3km3j3jp2dc6jhginapslivaar5lhg7xbjmggat2ejrqykql3x.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_18 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_33,), kwargs = {})
#   %amax_17 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_18, [0], True), kwargs = {})
triton_red_fused_abs_amax_9 = async_compile.triton('triton_red_fused_abs_amax_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_amax_9(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 98304
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_2 + 98304*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (x0 + 768*r0_2 + 98304*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = triton_helpers.maximum(_tmp21, tmp20)
        _tmp21 = tl.where(r0_mask, tmp22, _tmp21)
    tmp21 = triton_helpers.max2(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/64/c64mwqyqbviixfuewvgdzb26bpt6pjtj2idbw6znn64lfxb2bgy4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_18 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_33,), kwargs = {})
#   %amax_17 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_18, [0], True), kwargs = {})
triton_red_fused_abs_amax_10 = async_compile.triton('triton_red_fused_abs_amax_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_amax_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/vq/cvqsdpv2jljywjh5377cghyjo23zwtbk2zcuhqez7pkrc754ssip.py
# Topologically Sorted Source Nodes: [silu], Original ATen: [aten.silu, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.clone, aten._scaled_mm, aten.abs, aten.amax, aten.sigmoid, aten.fill, aten.sub, aten.add]
# Source node to ATen node mapping:
#   silu => convert_element_type_48, convert_element_type_49, mul_26, sigmoid
# Graph fragment:
#   %convert_element_type_48 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_23, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_48,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_48, %sigmoid), kwargs = {})
#   %convert_element_type_49 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_26, torch.bfloat16), kwargs = {})
#   %convert_element_type_79 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_17, torch.float64), kwargs = {})
#   %clamp_min_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_79, 1e-12), kwargs = {})
#   %reciprocal_33 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_34,), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_33, 448.0), kwargs = {})
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_46, torch.float32), kwargs = {})
#   %log2_17 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_80,), kwargs = {})
#   %floor_17 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_17,), kwargs = {})
#   %exp2_17 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_17,), kwargs = {})
#   %convert_element_type_81 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_33, torch.float32), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_81, %exp2_17), kwargs = {})
#   %clamp_min_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_47, -448.0), kwargs = {})
#   %clamp_max_17 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_35, 448.0), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_8 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_1, %permute_18, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_34, %convert_element_type_49), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_34, %view_26), kwargs = {})
#   %abs_19 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%view_35,), kwargs = {})
#   %amax_18 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_19, [-1], True), kwargs = {})
#   %convert_element_type_84 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_18, torch.float64), kwargs = {})
#   %clamp_min_36 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_84, 1e-12), kwargs = {})
#   %reciprocal_36 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_36,), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_36, 448.0), kwargs = {})
#   %convert_element_type_85 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_52, torch.float32), kwargs = {})
#   %log2_18 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_85,), kwargs = {})
#   %floor_18 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_18,), kwargs = {})
#   %exp2_18 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_18,), kwargs = {})
#   %convert_element_type_86 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.float32), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_86, %exp2_18), kwargs = {})
#   %clamp_min_37 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_53, -448.0), kwargs = {})
#   %clamp_max_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_37, 448.0), kwargs = {})
#   %convert_element_type_87 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_18, torch.float8_e4m3fn), kwargs = {})
#   %_scaled_mm_9 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_87, %permute_24, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
#   %sigmoid_1 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_23,), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 2048, 768], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_8, %sigmoid_1), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_23, %sub), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_66, 1), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %add_6), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %mul_67), kwargs = {})
#   %abs_23 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%view_39,), kwargs = {})
#   %amax_22 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_23, [-1], True), kwargs = {})
#   %convert_element_type_102 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_22, torch.float64), kwargs = {})
#   %clamp_min_44 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_102, 1e-12), kwargs = {})
#   %reciprocal_44 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_44,), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_44, 448.0), kwargs = {})
#   %convert_element_type_103 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_69, torch.float32), kwargs = {})
#   %log2_22 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_103,), kwargs = {})
#   %floor_22 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_22,), kwargs = {})
#   %exp2_22 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_22,), kwargs = {})
#   %convert_element_type_104 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_39, torch.float32), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_104, %exp2_22), kwargs = {})
#   %clamp_min_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_70, -448.0), kwargs = {})
#   %clamp_max_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_45, 448.0), kwargs = {})
#   %convert_element_type_105 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_22, torch.float8_e4m3fn), kwargs = {})
#   %_scaled_mm_11 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_105, %permute_35, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_exp2_fill_floor_log2_mul_reciprocal_sigmoid_silu_sub_11 = async_compile.triton('triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_exp2_fill_floor_log2_mul_reciprocal_sigmoid_silu_sub_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*fp8e4nv', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*fp8e4nv', 'out_ptr5': '*bf16', 'out_ptr6': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_exp2_fill_floor_log2_mul_reciprocal_sigmoid_silu_sub_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_exp2_fill_floor_log2_mul_reciprocal_sigmoid_silu_sub_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp36 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    _tmp106 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp46 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float64)
        tmp18 = tl.full([1, 1], 1e-12, tl.float64)
        tmp19 = triton_helpers.maximum(tmp17, tmp18)
        tmp20 = tl.full([1, 1], 1, tl.int32)
        tmp21 = (tmp20 / tmp19)
        tmp22 = tl.full([1, 1], 448.0, tl.float64)
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp25 = libdevice.log2(tmp24)
        tmp26 = libdevice.floor(tmp25)
        tmp27 = libdevice.exp2(tmp26)
        tmp28 = tmp15 * tmp27
        tmp29 = -448.0
        tmp30 = triton_helpers.maximum(tmp28, tmp29)
        tmp31 = 448.0
        tmp32 = triton_helpers.minimum(tmp30, tmp31)
        tmp33 = tmp32.to(tl.float8e4nv)
        tmp35 = tmp34.to(tl.float32)
        tmp37 = tmp36.to(tl.float64)
        tmp38 = triton_helpers.maximum(tmp37, tmp18)
        tmp39 = (tmp20 / tmp38)
        tmp40 = tmp39 * tmp22
        tmp41 = tmp40.to(tl.float32)
        tmp42 = libdevice.log2(tmp41)
        tmp43 = libdevice.floor(tmp42)
        tmp44 = libdevice.exp2(tmp43)
        tmp45 = (tmp20 / tmp44)
        tmp47 = tmp46.to(tl.float64)
        tmp48 = triton_helpers.maximum(tmp47, tmp18)
        tmp49 = (tmp20 / tmp48)
        tmp50 = tmp49 * tmp22
        tmp51 = tmp50.to(tl.float32)
        tmp52 = libdevice.log2(tmp51)
        tmp53 = libdevice.floor(tmp52)
        tmp54 = libdevice.exp2(tmp53)
        tmp55 = (tmp20 / tmp54)
        tmp56 = tmp45 * tmp55
        tmp57 = tmp35 * tmp56
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp58.to(tl.bfloat16)
        tmp60 = tmp59.to(tl.float32)
        tmp61 = tmp60 * tmp8
        tmp62 = tmp61.to(tl.bfloat16)
        tmp63 = tmp62.to(tl.float32)
        tmp64 = tmp60 * tmp11
        tmp65 = tmp64.to(tl.bfloat16)
        tmp66 = tmp65.to(tl.float32)
        tmp67 = tmp66.to(tl.bfloat16)
        tmp68 = tmp67.to(tl.float32)
        tmp69 = tmp0.to(tl.bfloat16)
        tmp70 = tmp69.to(tl.float32)
        tmp71 = tl.sigmoid(tmp70)
        tmp72 = tmp71.to(tl.bfloat16)
        tmp73 = tmp72.to(tl.float32)
        tmp74 = tmp73.to(tl.bfloat16)
        tmp75 = tmp74.to(tl.float32)
        tmp76 = 1.0
        tmp77 = tmp76 - tmp75
        tmp78 = tmp77.to(tl.bfloat16)
        tmp79 = tmp78.to(tl.float32)
        tmp80 = tmp79.to(tl.bfloat16)
        tmp81 = tmp80.to(tl.float32)
        tmp82 = tmp70 * tmp81
        tmp83 = tmp82.to(tl.bfloat16)
        tmp84 = tmp83.to(tl.float32)
        tmp85 = tmp84.to(tl.bfloat16)
        tmp86 = tmp85.to(tl.float32)
        tmp87 = tmp86 + tmp76
        tmp88 = tmp87.to(tl.bfloat16)
        tmp89 = tmp88.to(tl.float32)
        tmp90 = tmp89.to(tl.bfloat16)
        tmp91 = tmp90.to(tl.float32)
        tmp92 = tmp75 * tmp91
        tmp93 = tmp92.to(tl.bfloat16)
        tmp94 = tmp93.to(tl.float32)
        tmp95 = tmp94.to(tl.bfloat16)
        tmp96 = tmp95.to(tl.float32)
        tmp97 = tmp68 * tmp96
        tmp98 = tmp97.to(tl.bfloat16)
        tmp99 = tmp98.to(tl.float32)
        tmp100 = tmp99.to(tl.bfloat16)
        tmp101 = tmp100.to(tl.float32)
        tmp102 = tl_math.abs(tmp101)
        tmp103 = tmp102.to(tl.bfloat16)
        tmp104 = tmp103.to(tl.float32)
        tmp105 = tl.broadcast_to(tmp104, [XBLOCK, R0_BLOCK])
        tmp107 = triton_helpers.maximum(_tmp106, tmp105)
        _tmp106 = tl.where(r0_mask, tmp107, _tmp106)
        tl.store(out_ptr1 + (x0 + 16384*r0_1), tmp33, r0_mask)
        tl.store(out_ptr2 + (r0_1 + 768*x0), tmp63, r0_mask)
        tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp99, r0_mask)
    tmp106 = triton_helpers.max2(_tmp106, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp106, None)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp108 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp109 = tmp108.to(tl.float32)
        tmp110 = tmp106.to(tl.float64)
        tmp111 = tl.full([1, 1], 1e-12, tl.float64)
        tmp112 = triton_helpers.maximum(tmp110, tmp111)
        tmp113 = tl.full([1, 1], 1, tl.int32)
        tmp114 = (tmp113 / tmp112)
        tmp115 = tl.full([1, 1], 448.0, tl.float64)
        tmp116 = tmp114 * tmp115
        tmp117 = tmp116.to(tl.float32)
        tmp118 = libdevice.log2(tmp117)
        tmp119 = libdevice.floor(tmp118)
        tmp120 = libdevice.exp2(tmp119)
        tmp121 = tmp109 * tmp120
        tmp122 = -448.0
        tmp123 = triton_helpers.maximum(tmp121, tmp122)
        tmp124 = 448.0
        tmp125 = triton_helpers.minimum(tmp123, tmp124)
        tmp126 = tmp125.to(tl.float8e4nv)
        tl.store(out_ptr4 + (r0_1 + 768*x0), tmp126, r0_mask)
    _tmp134 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp127 = tl.load(out_ptr2 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp128 = tmp127.to(tl.bfloat16)
        tmp129 = tmp128.to(tl.float32)
        tmp130 = tl_math.abs(tmp129)
        tmp131 = tmp130.to(tl.bfloat16)
        tmp132 = tmp131.to(tl.float32)
        tmp133 = tl.broadcast_to(tmp132, [XBLOCK, R0_BLOCK])
        tmp135 = triton_helpers.maximum(_tmp134, tmp133)
        _tmp134 = tl.where(r0_mask, tmp135, _tmp134)
    tmp134 = triton_helpers.max2(_tmp134, 1)[:, None]
    tl.store(out_ptr5 + (x0), tmp134, None)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp136 = tl.load(out_ptr2 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp137 = tmp136.to(tl.float32)
        tmp138 = tmp134.to(tl.float64)
        tmp139 = tl.full([1, 1], 1e-12, tl.float64)
        tmp140 = triton_helpers.maximum(tmp138, tmp139)
        tmp141 = tl.full([1, 1], 1, tl.int32)
        tmp142 = (tmp141 / tmp140)
        tmp143 = tl.full([1, 1], 448.0, tl.float64)
        tmp144 = tmp142 * tmp143
        tmp145 = tmp144.to(tl.float32)
        tmp146 = libdevice.log2(tmp145)
        tmp147 = libdevice.floor(tmp146)
        tmp148 = libdevice.exp2(tmp147)
        tmp149 = tmp137 * tmp148
        tmp150 = -448.0
        tmp151 = triton_helpers.maximum(tmp149, tmp150)
        tmp152 = 448.0
        tmp153 = triton_helpers.minimum(tmp151, tmp152)
        tmp154 = tmp153.to(tl.float8e4nv)
        tl.store(out_ptr6 + (r0_1 + 768*x0), tmp154, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/un/cuneqwxusv7yvaa26y4vzqrpfnka64crlswgwmn52uvpajrnwxeb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_3, 0), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_9, %mul_55), kwargs = {})
#   %abs_20 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_5,), kwargs = {})
#   %amax_19 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_20, [-1], True), kwargs = {})
triton_red_fused_abs_add_amax_mul_12 = async_compile.triton('triton_red_fused_abs_add_amax_mul_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_add_amax_mul_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_add_amax_mul_12(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = 0.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp7.to(tl.bfloat16)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9.to(tl.bfloat16)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp6
        tmp13 = tmp12.to(tl.bfloat16)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14.to(tl.bfloat16)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp2 + tmp16
        tmp18 = tmp17.to(tl.bfloat16)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19.to(tl.bfloat16)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl_math.abs(tmp21)
        tmp23 = tmp22.to(tl.bfloat16)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = triton_helpers.maximum(_tmp26, tmp25)
        _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
    tmp26 = triton_helpers.max2(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/sb/csb6didlf3oq7cfvioab37z3favgenx6mmdnoljryqxc55sf2rh6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_3, 0), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_9, %mul_55), kwargs = {})
#   %abs_20 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_5,), kwargs = {})
#   %amax_19 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_20, [-1], True), kwargs = {})
triton_per_fused_abs_add_amax_mul_13 = async_compile.triton('triton_per_fused_abs_add_amax_mul_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_amax_mul_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_amax_mul_13(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 6
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask & r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/pa/cpafdr4tjnkvpccm4w3emjlszubsyfkjg3f3ledzc2fv36c4iv4l.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_84 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_18, torch.float64), kwargs = {})
#   %clamp_min_36 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_84, 1e-12), kwargs = {})
#   %reciprocal_36 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_36,), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_36, 448.0), kwargs = {})
#   %convert_element_type_85 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_52, torch.float32), kwargs = {})
#   %log2_18 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_85,), kwargs = {})
#   %floor_18 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_18,), kwargs = {})
#   %exp2_18 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_18,), kwargs = {})
#   %convert_element_type_86 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.float32), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_86, %exp2_18), kwargs = {})
#   %clamp_min_37 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_53, -448.0), kwargs = {})
#   %clamp_max_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_37, 448.0), kwargs = {})
#   %convert_element_type_87 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_18, torch.float8_e4m3fn), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_3, 0), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_9, %mul_55), kwargs = {})
#   %convert_element_type_88 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_19, torch.float64), kwargs = {})
#   %clamp_min_38 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_88, 1e-12), kwargs = {})
#   %reciprocal_37 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_38,), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_37, 448.0), kwargs = {})
#   %convert_element_type_89 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_56, torch.float32), kwargs = {})
#   %log2_19 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_89,), kwargs = {})
#   %floor_19 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_19,), kwargs = {})
#   %exp2_19 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_19,), kwargs = {})
#   %convert_element_type_90 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_90, %exp2_19), kwargs = {})
#   %_scaled_mm_9 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_87, %permute_24, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_14 = async_compile.triton('triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_14(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 0.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.bfloat16)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9.to(tl.bfloat16)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp6
    tmp13 = tmp12.to(tl.bfloat16)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.bfloat16)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp2 + tmp16
    tmp18 = tmp17.to(tl.bfloat16)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp21.to(tl.float64)
    tmp23 = tl.full([1, 1], 1e-12, tl.float64)
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp25 = tl.full([1, 1], 1, tl.int32)
    tmp26 = (tmp25 / tmp24)
    tmp27 = tl.full([1, 1], 448.0, tl.float64)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = libdevice.log2(tmp29)
    tmp31 = libdevice.floor(tmp30)
    tmp32 = libdevice.exp2(tmp31)
    tmp33 = tmp20 * tmp32
    tmp34 = -448.0
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp36 = 448.0
    tmp37 = triton_helpers.minimum(tmp35, tmp36)
    tmp38 = tmp37.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y0 + 768*x1), tmp38, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/rc/crc5kwoit4tatyu5avowadopx6ucvvvjuclxvyqasaxquj5z5j2u.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_43, %view_44), kwargs = {})
triton_poi_fused_add_15 = async_compile.triton('triton_poi_fused_add_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 256
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp30 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp32 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp42 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = triton_helpers.maximum(tmp16, tmp4)
    tmp18 = (tmp6 / tmp17)
    tmp19 = tmp18 * tmp8
    tmp20 = tmp19.to(tl.float32)
    tmp21 = libdevice.log2(tmp20)
    tmp22 = libdevice.floor(tmp21)
    tmp23 = libdevice.exp2(tmp22)
    tmp24 = (tmp6 / tmp23)
    tmp25 = tmp14 * tmp24
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.bfloat16)
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp33 = tmp32.to(tl.float64)
    tmp34 = triton_helpers.maximum(tmp33, tmp4)
    tmp35 = (tmp6 / tmp34)
    tmp36 = tmp35 * tmp8
    tmp37 = tmp36.to(tl.float32)
    tmp38 = libdevice.log2(tmp37)
    tmp39 = libdevice.floor(tmp38)
    tmp40 = libdevice.exp2(tmp39)
    tmp41 = (tmp6 / tmp40)
    tmp43 = tmp42.to(tl.float64)
    tmp44 = triton_helpers.maximum(tmp43, tmp4)
    tmp45 = (tmp6 / tmp44)
    tmp46 = tmp45 * tmp8
    tmp47 = tmp46.to(tl.float32)
    tmp48 = libdevice.log2(tmp47)
    tmp49 = libdevice.floor(tmp48)
    tmp50 = libdevice.exp2(tmp49)
    tmp51 = (tmp6 / tmp50)
    tmp52 = tmp41 * tmp51
    tmp53 = tmp31 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp54.to(tl.bfloat16)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp29 + tmp56
    tmp58 = tmp57.to(tl.bfloat16)
    tmp59 = tmp58.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp59, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/yq/cyqzlzbf65zswweoq74vgvkfybmxmmoytp4ltvfmppvcuqpvivl2.py
# Topologically Sorted Source Nodes: [h, rms_norm_1], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.sum]
# Source node to ATen node mapping:
#   h => add_1
#   rms_norm_1 => add_2, convert_element_type_38, mean_1, mul_20, pow_2, rsqrt_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_20), kwargs = {})
#   %convert_element_type_38 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_38, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 9.999999747378752e-06), kwargs = {})
#   %rsqrt_1 : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_20 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_38, %rsqrt_1), kwargs = {})
#   %abs_22 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_38,), kwargs = {})
#   %amax_21 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_22, [0], True), kwargs = {})
#   %convert_element_type_120 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float32), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_120, %mul_20), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_83, [0, 1], True), kwargs = {})
triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_16 = async_compile.triton('triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp17 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr2 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr4 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.bfloat16)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = 256.0
        tmp12 = (tmp10 / tmp11)
        tmp13 = 9.999999747378752e-06
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp9 * tmp15
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20.to(tl.bfloat16)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl_math.abs(tmp22)
        tmp24 = tmp23.to(tl.bfloat16)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = triton_helpers.maximum(_tmp27, tmp26)
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 * tmp16
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask, tmp34, _tmp33)
    tmp27 = triton_helpers.max2(_tmp27, 1)[:, None]
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp27, None)
    tl.store(out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/52/c52km2qy2c2xjux3f2p7y647ncmhpaplvooedt7gkcekxepc7s7y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_22 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_38,), kwargs = {})
#   %amax_21 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_22, [0], True), kwargs = {})
triton_red_fused_abs_amax_17 = async_compile.triton('triton_red_fused_abs_amax_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_amax_17(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/lu/clu4ybpw7gck2f72vmu6wrfqhpbi37xijigc3hz6v65bwos5vlqf.py
# Topologically Sorted Source Nodes: [h, rms_norm_1, rms_norm, output], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.clone, aten._scaled_mm, aten.sum, aten.div, aten.abs, aten.amax]
# Source node to ATen node mapping:
#   h => add_1
#   output => abs_1, amax, clamp_max, clamp_min, clamp_min_1, convert_element_type_2, convert_element_type_3, convert_element_type_4, convert_element_type_5, exp2, floor, log2, mul_2, mul_3, reciprocal
#   rms_norm => add, convert_element_type, convert_element_type_1, mean, mul, mul_1, pow_1, rsqrt
#   rms_norm_1 => add_2, convert_element_type_38, mean_1, pow_2, rsqrt_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_20), kwargs = {})
#   %convert_element_type_38 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_38, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 9.999999747378752e-06), kwargs = {})
#   %rsqrt_1 : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %convert_element_type_97 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_21, torch.float64), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_97, 1e-12), kwargs = {})
#   %reciprocal_41 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_42,), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_41, 448.0), kwargs = {})
#   %convert_element_type_98 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_62, torch.float32), kwargs = {})
#   %log2_21 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_98,), kwargs = {})
#   %floor_21 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_21,), kwargs = {})
#   %exp2_21 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_21,), kwargs = {})
#   %convert_element_type_99 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_38, torch.float32), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_99, %exp2_21), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_25,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_10 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_4, %permute_28, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_36,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_12 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_7, %permute_28, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
#   %convert_element_type_120 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float32), kwargs = {})
#   %mul_84 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_120, %primals_8), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %convert_element_type_38), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %rsqrt_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_85, [2], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 256), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_38, 1.0), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_4, 2.0), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_89), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %mul_90), kwargs = {})
#   %convert_element_type : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 9.999999747378752e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_3), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_1,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_1, [-1], True), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax, torch.float64), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 1e-12), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 448.0), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float32), kwargs = {})
#   %log2 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_3,), kwargs = {})
#   %floor : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2,), kwargs = {})
#   %exp2 : [num_users=1] = call_function[target=torch.ops.aten.exp2.default](args = (%floor,), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.float32), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %exp2), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_3, -448.0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 448.0), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.float8_e4m3fn), kwargs = {})
triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_18 = async_compile.triton('triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr1': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp8e4nv', 'out_ptr6': '*fp8e4nv', 'out_ptr7': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 3, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp6 = tmp5.to(tl.bfloat16)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.bfloat16)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp7 + tmp10
        tmp12 = tmp11.to(tl.bfloat16)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp4 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
        tmp19 = tmp5.to(tl.float32)
        tmp20 = tmp19 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask, tmp23, _tmp22)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp22, None)
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    tmp53 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp24 = tl.load(in_ptr2 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp46 = tl.load(in_ptr3 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp58 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp63 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp76 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 256.0
        tmp27 = (tmp22 / tmp26)
        tmp28 = 9.999999747378752e-06
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp31 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35.to(tl.bfloat16)
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tl_math.abs(tmp37)
        tmp39 = tmp38.to(tl.bfloat16)
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = triton_helpers.maximum(_tmp42, tmp41)
        _tmp42 = tl.where(r0_mask, tmp43, _tmp42)
        tmp44 = tmp24.to(tl.bfloat16)
        tmp45 = tmp44.to(tl.float32)
        tmp47 = tmp46.to(tl.bfloat16)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tmp45 + tmp48
        tmp50 = tmp49.to(tl.bfloat16)
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp51.to(tl.float32)
        tmp54 = (tmp53 / tmp26)
        tmp55 = tmp54 + tmp28
        tmp56 = libdevice.rsqrt(tmp55)
        tmp57 = tmp52 * tmp56
        tmp59 = tmp58.to(tl.float32)
        tmp60 = tmp57 * tmp59
        tmp61 = tmp60.to(tl.float32)
        tmp62 = tmp61.to(tl.float32)
        tmp64 = tmp63.to(tl.float64)
        tmp65 = tl.full([1, 1], 1e-12, tl.float64)
        tmp66 = triton_helpers.maximum(tmp64, tmp65)
        tmp67 = tl.full([1, 1], 1, tl.int32)
        tmp68 = (tmp67 / tmp66)
        tmp69 = tl.full([1, 1], 448.0, tl.float64)
        tmp70 = tmp68 * tmp69
        tmp71 = tmp70.to(tl.float32)
        tmp72 = libdevice.log2(tmp71)
        tmp73 = libdevice.floor(tmp72)
        tmp74 = libdevice.exp2(tmp73)
        tmp75 = tmp62 * tmp74
        tmp77 = tmp76.to(tl.float32)
        tmp78 = tmp77 * tmp59
        tmp79 = tmp78 * tmp56
        tmp80 = -0.5
        tmp81 = tmp17 * tmp80
        tmp82 = tmp56 * tmp56
        tmp83 = tmp82 * tmp56
        tmp84 = tmp81 * tmp83
        tmp85 = 0.00390625
        tmp86 = tmp84 * tmp85
        tmp87 = 2.0
        tmp88 = tmp52 * tmp87
        tmp89 = tmp86 * tmp88
        tmp90 = tmp79 + tmp89
        tmp91 = -448.0
        tmp92 = triton_helpers.maximum(tmp75, tmp91)
        tmp93 = 448.0
        tmp94 = triton_helpers.minimum(tmp92, tmp93)
        tmp95 = tmp94.to(tl.float8e4nv)
        tl.store(out_ptr4 + (r0_1 + 256*x0), tmp90, r0_mask)
        tl.store(out_ptr5 + (x0 + 16384*r0_1), tmp95, r0_mask)
        tl.store(out_ptr6 + (x0 + 16384*r0_1), tmp95, r0_mask)
    tmp42 = triton_helpers.max2(_tmp42, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp96 = tl.load(in_ptr2 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp104 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp97 = tmp96.to(tl.float32)
        tmp98 = 256.0
        tmp99 = (tmp22 / tmp98)
        tmp100 = 9.999999747378752e-06
        tmp101 = tmp99 + tmp100
        tmp102 = libdevice.rsqrt(tmp101)
        tmp103 = tmp97 * tmp102
        tmp105 = tmp104.to(tl.float32)
        tmp106 = tmp103 * tmp105
        tmp107 = tmp106.to(tl.float32)
        tmp108 = tmp107.to(tl.float32)
        tmp109 = tmp42.to(tl.float64)
        tmp110 = tl.full([1, 1], 1e-12, tl.float64)
        tmp111 = triton_helpers.maximum(tmp109, tmp110)
        tmp112 = tl.full([1, 1], 1, tl.int32)
        tmp113 = (tmp112 / tmp111)
        tmp114 = tl.full([1, 1], 448.0, tl.float64)
        tmp115 = tmp113 * tmp114
        tmp116 = tmp115.to(tl.float32)
        tmp117 = libdevice.log2(tmp116)
        tmp118 = libdevice.floor(tmp117)
        tmp119 = libdevice.exp2(tmp118)
        tmp120 = tmp108 * tmp119
        tmp121 = -448.0
        tmp122 = triton_helpers.maximum(tmp120, tmp121)
        tmp123 = 448.0
        tmp124 = triton_helpers.minimum(tmp122, tmp123)
        tmp125 = tmp124.to(tl.float8e4nv)
        tl.store(out_ptr7 + (r0_1 + 256*x0), tmp125, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/ib/cib6cgub7xtaoisgfrm4w6gbc7n7xbxqjsdjcyqpksll2d55juea.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_15 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%view_30,), kwargs = {})
#   %amax_16 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_15, [0], True), kwargs = {})
#   %abs_27 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%view_46,), kwargs = {})
#   %amax_28 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_27, [0], True), kwargs = {})
triton_red_fused_abs_amax_19 = async_compile.triton('triton_red_fused_abs_amax_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_amax_19(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl_math.abs(tmp2)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10.to(tl.bfloat16)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp2 + tmp12
        tmp14 = tmp13.to(tl.bfloat16)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15.to(tl.bfloat16)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl_math.abs(tmp17)
        tmp19 = tmp18.to(tl.bfloat16)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = triton_helpers.maximum(_tmp22, tmp21)
        _tmp22 = tl.where(r0_mask, tmp23, _tmp22)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tmp22 = triton_helpers.max2(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/wa/cwaptw2tvrpczzl4qe4pwtezhhf2r24jn2eqng5guympw3xbn4vs.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_8 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_1, %permute_18, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_poi_fused__scaled_mm_clone_20 = async_compile.triton('triton_poi_fused__scaled_mm_clone_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm_clone_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm_clone_20(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1, 1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1, 1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1, 1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = tmp1 * tmp13
    tmp15 = -448.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 448.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr0 + (x1 + 16384*y0), tmp19, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/s6/cs6bjav7msng2gzp5pwdppt5avyxlrw3qnbt3w74jixiq3sbtn6v.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_79 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_17, torch.float64), kwargs = {})
#   %clamp_min_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_79, 1e-12), kwargs = {})
#   %reciprocal_33 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_34,), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_33, 448.0), kwargs = {})
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_46, torch.float32), kwargs = {})
#   %log2_17 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_80,), kwargs = {})
#   %floor_17 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_17,), kwargs = {})
#   %exp2_17 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_17,), kwargs = {})
#   %reciprocal_34 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%permute_16,), kwargs = {})
#   %reciprocal_35 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_17,), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_34, %reciprocal_35), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_scaled_mm_8, %mul_48), kwargs = {})
#   %convert_element_type_83 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_49, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_21 = async_compile.triton('triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_21(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 768
    x0 = (xindex % 768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = triton_helpers.maximum(tmp16, tmp4)
    tmp18 = (tmp6 / tmp17)
    tmp19 = tmp18 * tmp8
    tmp20 = tmp19.to(tl.float32)
    tmp21 = libdevice.log2(tmp20)
    tmp22 = libdevice.floor(tmp21)
    tmp23 = libdevice.exp2(tmp22)
    tmp24 = (tmp6 / tmp23)
    tmp25 = tmp14 * tmp24
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/xn/cxn32gwnqcidcbzr76lakyaesibq24gerskj44pdmijhqk5wtixh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_19 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%view_35,), kwargs = {})
#   %amax_20 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_19, [0], True), kwargs = {})
triton_red_fused_abs_amax_22 = async_compile.triton('triton_red_fused_abs_amax_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_amax_22(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 98304
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_2 + 98304*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl_math.abs(tmp2)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/gk/cgkylp244hctxjq7p4lxkevhj46ckh5mn2ptzzharehmtj4mybyq.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_25,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_10 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_4, %permute_28, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_poi_fused__scaled_mm_clone_23 = async_compile.triton('triton_poi_fused__scaled_mm_clone_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm_clone_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm_clone_23(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 768*x1), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1, 1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1, 1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1, 1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = tmp1 * tmp13
    tmp15 = -448.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 448.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr0 + (x1 + 16384*y0), tmp19, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/nt/cntvk7bnkzrktfomdnqak2kfgnhyi7q6srtmivissn73binsnvlo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_97 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_21, torch.float64), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_97, 1e-12), kwargs = {})
#   %reciprocal_41 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_42,), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_41, 448.0), kwargs = {})
#   %convert_element_type_98 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_62, torch.float32), kwargs = {})
#   %log2_21 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_98,), kwargs = {})
#   %floor_21 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_21,), kwargs = {})
#   %exp2_21 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_21,), kwargs = {})
#   %reciprocal_42 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%permute_26,), kwargs = {})
#   %reciprocal_43 : [num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_21,), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_42, %reciprocal_43), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_scaled_mm_10, %mul_64), kwargs = {})
#   %convert_element_type_101 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_65, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_24 = async_compile.triton('triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_24(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 256
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = triton_helpers.maximum(tmp16, tmp4)
    tmp18 = (tmp6 / tmp17)
    tmp19 = tmp18 * tmp8
    tmp20 = tmp19.to(tl.float32)
    tmp21 = libdevice.log2(tmp20)
    tmp22 = libdevice.floor(tmp21)
    tmp23 = libdevice.exp2(tmp22)
    tmp24 = (tmp6 / tmp23)
    tmp25 = tmp14 * tmp24
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/iz/cizyfcdhyl7ywiql57nowi56sddt5htvohhdv5wzxarllfuwodrb.py
# Topologically Sorted Source Nodes: [h, rms_norm_1], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   h => add_1
#   rms_norm_1 => add_2, convert_element_type_38, mean_1, mul_20, pow_2, rsqrt_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_20), kwargs = {})
#   %convert_element_type_38 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_38, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 9.999999747378752e-06), kwargs = {})
#   %rsqrt_1 : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_20 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_38, %rsqrt_1), kwargs = {})
#   %convert_element_type_120 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float32), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_120, %mul_20), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_83, [0, 1], True), kwargs = {})
#   %convert_element_type_121 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_45, torch.bfloat16), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_25 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_25(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/i2/ci264ypjwbk5giyqda6qcnk2jyqikrj4boj7aldnuvzewx4pwdv4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_27 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%view_46,), kwargs = {})
#   %amax_26 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_27, [-1], True), kwargs = {})
#   %convert_element_type_123 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_26, torch.float64), kwargs = {})
#   %clamp_min_52 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_123, 1e-12), kwargs = {})
#   %reciprocal_52 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_52,), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_52, 448.0), kwargs = {})
#   %convert_element_type_124 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_91, torch.float32), kwargs = {})
#   %log2_26 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_124,), kwargs = {})
#   %floor_26 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_26,), kwargs = {})
#   %exp2_26 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_26,), kwargs = {})
#   %convert_element_type_125 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_46, torch.float32), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_125, %exp2_26), kwargs = {})
#   %clamp_min_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_92, -448.0), kwargs = {})
#   %clamp_max_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_53, 448.0), kwargs = {})
#   %convert_element_type_126 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_26, torch.float8_e4m3fn), kwargs = {})
#   %_scaled_mm_13 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_126, %permute_45, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_46,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_14 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_10, %permute_49, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_clone_exp2_floor_log2_mul_reciprocal_26 = async_compile.triton('triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_clone_exp2_floor_log2_mul_reciprocal_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'out_ptr2': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_clone_exp2_floor_log2_mul_reciprocal_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_clone_exp2_floor_log2_mul_reciprocal_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4.to(tl.bfloat16)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp2 + tmp6
        tmp8 = tmp7.to(tl.bfloat16)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9.to(tl.bfloat16)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl_math.abs(tmp11)
        tmp13 = tmp12.to(tl.bfloat16)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = triton_helpers.maximum(_tmp16, tmp15)
        _tmp16 = tl.where(r0_mask, tmp17, _tmp16)
    tmp16 = triton_helpers.max2(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp16, None)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp18 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr1 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp46 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tmp18.to(tl.bfloat16)
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.bfloat16)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp20 + tmp24
        tmp26 = tmp25.to(tl.bfloat16)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp16.to(tl.float64)
        tmp30 = tl.full([1, 1], 1e-12, tl.float64)
        tmp31 = triton_helpers.maximum(tmp29, tmp30)
        tmp32 = tl.full([1, 1], 1, tl.int32)
        tmp33 = (tmp32 / tmp31)
        tmp34 = tl.full([1, 1], 448.0, tl.float64)
        tmp35 = tmp33 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tmp37 = libdevice.log2(tmp36)
        tmp38 = libdevice.floor(tmp37)
        tmp39 = libdevice.exp2(tmp38)
        tmp40 = tmp28 * tmp39
        tmp41 = -448.0
        tmp42 = triton_helpers.maximum(tmp40, tmp41)
        tmp43 = 448.0
        tmp44 = triton_helpers.minimum(tmp42, tmp43)
        tmp45 = tmp44.to(tl.float8e4nv)
        tmp47 = tmp46.to(tl.float64)
        tmp48 = triton_helpers.maximum(tmp47, tmp30)
        tmp49 = (tmp32 / tmp48)
        tmp50 = tmp49 * tmp34
        tmp51 = tmp50.to(tl.float32)
        tmp52 = libdevice.log2(tmp51)
        tmp53 = libdevice.floor(tmp52)
        tmp54 = libdevice.exp2(tmp53)
        tmp55 = tmp28 * tmp54
        tmp56 = triton_helpers.maximum(tmp55, tmp41)
        tmp57 = triton_helpers.minimum(tmp56, tmp43)
        tmp58 = tmp57.to(tl.float8e4nv)
        tl.store(out_ptr1 + (r0_1 + 256*x0), tmp45, r0_mask)
        tl.store(out_ptr2 + (x0 + 16384*r0_1), tmp58, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/bx/cbx7aouisq5h75azvl6hpo6jeok4d3vhjsyi3npjst55mm5p42ja.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_5, 0), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_7, %mul_94), kwargs = {})
triton_poi_fused_add_mul_27 = async_compile.triton('triton_poi_fused_add_mul_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7.to(tl.bfloat16)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp5 + tmp9
    tmp11 = tmp10.to(tl.bfloat16)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12.to(tl.bfloat16)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 0.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.bfloat16)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.bfloat16)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp15
    tmp22 = tmp21.to(tl.bfloat16)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23.to(tl.bfloat16)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp2 + tmp25
    tmp27 = tmp26.to(tl.bfloat16)
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/3e/c3ekcbmctdknjrljqj5w63rayb577oxum4wq26gaatpn7bkzr5df.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_28 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_11,), kwargs = {})
#   %amax_27 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_28, [-1], True), kwargs = {})
triton_red_fused_abs_amax_28 = async_compile.triton('triton_red_fused_abs_amax_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_amax_28(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl_math.abs(tmp2)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/lx/clxyxm36poz5qwspn26pval2qflu4ycbibhv2vvjbdvsltjztksc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_28 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_11,), kwargs = {})
#   %amax_27 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_28, [-1], True), kwargs = {})
triton_per_fused_abs_amax_29 = async_compile.triton('triton_per_fused_abs_amax_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_amax_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_amax_29(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/5t/c5tupclnks6czovxvtb7y6k4bb343e6ve6tfa62jvh4xrwaodund.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_123 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_26, torch.float64), kwargs = {})
#   %clamp_min_52 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_123, 1e-12), kwargs = {})
#   %reciprocal_52 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_52,), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_52, 448.0), kwargs = {})
#   %convert_element_type_124 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_91, torch.float32), kwargs = {})
#   %log2_26 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_124,), kwargs = {})
#   %floor_26 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_26,), kwargs = {})
#   %exp2_26 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_26,), kwargs = {})
#   %convert_element_type_125 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_46, torch.float32), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_125, %exp2_26), kwargs = {})
#   %clamp_min_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_92, -448.0), kwargs = {})
#   %clamp_max_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_53, 448.0), kwargs = {})
#   %convert_element_type_126 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_26, torch.float8_e4m3fn), kwargs = {})
#   %_scaled_mm_13 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_126, %permute_45, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_poi_fused__scaled_mm__to_copy_clamp_exp2_floor_log2_mul_reciprocal_30 = async_compile.triton('triton_poi_fused__scaled_mm__to_copy_clamp_exp2_floor_log2_mul_reciprocal_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm__to_copy_clamp_exp2_floor_log2_mul_reciprocal_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm__to_copy_clamp_exp2_floor_log2_mul_reciprocal_30(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask & ymask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1, 1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1, 1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1, 1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = tmp1 * tmp13
    tmp15 = -448.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 448.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr0 + (y0 + 256*x1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/ak/cakevgoekfxfl2dclnpu5i4om5qphrv7vppenk3xqxrlfxmtgrb5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_30 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_49,), kwargs = {})
#   %amax_29 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_30, [0], True), kwargs = {})
triton_red_fused_abs_amax_31 = async_compile.triton('triton_red_fused_abs_amax_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_amax_31(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl_math.abs(tmp2)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/dn/cdnppefhqgz2awwid3hmt7wfxbg2d24aj3wenbjwkqnelqmjiedr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_46,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_14 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_10, %permute_49, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_poi_fused__scaled_mm_clone_32 = async_compile.triton('triton_poi_fused__scaled_mm_clone_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm_clone_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm_clone_32(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1, 1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1, 1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1, 1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = tmp1 * tmp13
    tmp15 = -448.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 448.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr0 + (y0 + 16384*x1), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/n4/cn47wmg7rx6xtrbnkzwyx5lolbo5hdfhhtntnw6huiietowwcyfl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_136 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_29, torch.float64), kwargs = {})
#   %clamp_min_58 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_136, 1e-12), kwargs = {})
#   %reciprocal_57 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_58,), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_57, 448.0), kwargs = {})
#   %convert_element_type_137 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_101, torch.float32), kwargs = {})
#   %log2_29 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_137,), kwargs = {})
#   %floor_29 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_29,), kwargs = {})
#   %exp2_29 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_29,), kwargs = {})
#   %reciprocal_58 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%permute_47,), kwargs = {})
#   %reciprocal_59 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%exp2_29,), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_58, %reciprocal_59), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_scaled_mm_14, %mul_103), kwargs = {})
#   %convert_element_type_140 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_104, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_33 = async_compile.triton('triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_33(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 256
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = triton_helpers.maximum(tmp16, tmp4)
    tmp18 = (tmp6 / tmp17)
    tmp19 = tmp18 * tmp8
    tmp20 = tmp19.to(tl.float32)
    tmp21 = libdevice.log2(tmp20)
    tmp22 = libdevice.floor(tmp21)
    tmp23 = libdevice.exp2(tmp22)
    tmp24 = (tmp6 / tmp23)
    tmp25 = tmp14 * tmp24
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/ys/cysup6ki2rnb4eekvr2wcylsh3gkfmkixscirqd7kwtvl3ux2d37.py
# Topologically Sorted Source Nodes: [xq_], Original ATen: [aten.view_as_complex]
# Source node to ATen node mapping:
#   xq_ => view_as_complex
# Graph fragment:
#   %view_as_complex : [num_users=1] = call_function[target=torch.ops.aten.view_as_complex.default](args = (%view_12,), kwargs = {})
triton_poi_fused_view_as_complex_34 = async_compile.triton('triton_poi_fused_view_as_complex_34', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_as_complex_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_as_complex_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/te/cteq65sa6lslscv4o2g3md73da6u5hn2zegqvhlqhn6px75f76fe.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_scaled_dot_product_flash_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_53, %permute_3, %permute_4, %permute_5, %getitem, %getitem_1, None, None, 2048, 2048, 0.0, True, %getitem_6, %getitem_7), kwargs = {scale: 0.25})
triton_poi_fused__scaled_dot_product_flash_attention_backward_35 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention_backward_35', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_backward_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention_backward_35(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 256
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = triton_helpers.maximum(tmp16, tmp4)
    tmp18 = (tmp6 / tmp17)
    tmp19 = tmp18 * tmp8
    tmp20 = tmp19.to(tl.float32)
    tmp21 = libdevice.log2(tmp20)
    tmp22 = libdevice.floor(tmp21)
    tmp23 = libdevice.exp2(tmp22)
    tmp24 = (tmp6 / tmp23)
    tmp25 = tmp14 * tmp24
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/dd/cddc3jdtzo4zobx67o3slbxdylzccfsxjvdhegezf6jwjzga4rp5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_scaled_dot_product_flash_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_53, %permute_3, %permute_4, %permute_5, %getitem, %getitem_1, None, None, 2048, 2048, 0.0, True, %getitem_6, %getitem_7), kwargs = {scale: 0.25})
triton_poi_fused__scaled_dot_product_flash_attention_backward_36 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention_backward_36', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_backward_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention_backward_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/ig/cigndpfoqhqqi3qdma44emo74ueeoeng3dxuau3qg5v35xdjicue.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_6, 0), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_2, %mul_110), kwargs = {})
#   %abs_32 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_12,), kwargs = {})
#   %amax_31 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_32, [-1], True), kwargs = {})
triton_red_fused_abs_add_amax_mul_37 = async_compile.triton('triton_red_fused_abs_add_amax_mul_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_add_amax_mul_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_add_amax_mul_37(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = 0.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp7.to(tl.bfloat16)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9.to(tl.bfloat16)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp6
        tmp13 = tmp12.to(tl.bfloat16)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14.to(tl.bfloat16)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp2 + tmp16
        tmp18 = tmp17.to(tl.bfloat16)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19.to(tl.bfloat16)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl_math.abs(tmp21)
        tmp23 = tmp22.to(tl.bfloat16)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = triton_helpers.maximum(_tmp26, tmp25)
        _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
    tmp26 = triton_helpers.max2(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/uq/cuqfcanvt7eax2546u3afp4x3fb4gwgh5aao2nuonw4sukjgdqnb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_145 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_30, torch.float64), kwargs = {})
#   %clamp_min_60 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_145, 1e-12), kwargs = {})
#   %reciprocal_60 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_60,), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_60, 448.0), kwargs = {})
#   %convert_element_type_146 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_107, torch.float32), kwargs = {})
#   %log2_30 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_146,), kwargs = {})
#   %floor_30 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_30,), kwargs = {})
#   %exp2_30 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_30,), kwargs = {})
#   %convert_element_type_147 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_60, torch.float32), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_147, %exp2_30), kwargs = {})
#   %clamp_min_61 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_108, -448.0), kwargs = {})
#   %clamp_max_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_61, 448.0), kwargs = {})
#   %convert_element_type_148 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_30, torch.float8_e4m3fn), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_6, 0), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_2, %mul_110), kwargs = {})
#   %convert_element_type_149 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_31, torch.float64), kwargs = {})
#   %clamp_min_62 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_149, 1e-12), kwargs = {})
#   %reciprocal_61 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_62,), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_61, 448.0), kwargs = {})
#   %convert_element_type_150 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_111, torch.float32), kwargs = {})
#   %log2_31 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_150,), kwargs = {})
#   %floor_31 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_31,), kwargs = {})
#   %exp2_31 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_31,), kwargs = {})
#   %convert_element_type_151 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_151, %exp2_31), kwargs = {})
#   %_scaled_mm_15 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_148, %permute_60, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_38 = async_compile.triton('triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_38(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask & ymask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.bfloat16)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 0.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.bfloat16)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9.to(tl.bfloat16)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp6
    tmp13 = tmp12.to(tl.bfloat16)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.bfloat16)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp2 + tmp16
    tmp18 = tmp17.to(tl.bfloat16)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp21.to(tl.float64)
    tmp23 = tl.full([1, 1], 1e-12, tl.float64)
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp25 = tl.full([1, 1], 1, tl.int32)
    tmp26 = (tmp25 / tmp24)
    tmp27 = tl.full([1, 1], 448.0, tl.float64)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = libdevice.log2(tmp29)
    tmp31 = libdevice.floor(tmp30)
    tmp32 = libdevice.exp2(tmp31)
    tmp33 = tmp20 * tmp32
    tmp34 = -448.0
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp36 = 448.0
    tmp37 = triton_helpers.minimum(tmp35, tmp36)
    tmp38 = tmp37.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y0 + 256*x1), tmp38, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/3t/c3tyzmkbmqmylhtur4bdotuhuehbbhhcquyokk4bbkv5x2xuou5x.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_35 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%view_64,), kwargs = {})
#   %amax_34 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_35, [-1], True), kwargs = {})
#   %convert_element_type_163 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_34, torch.float64), kwargs = {})
#   %clamp_min_68 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_163, 1e-12), kwargs = {})
#   %reciprocal_68 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_68,), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_68, 448.0), kwargs = {})
#   %convert_element_type_164 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_121, torch.float32), kwargs = {})
#   %log2_34 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_164,), kwargs = {})
#   %floor_34 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_34,), kwargs = {})
#   %exp2_34 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_34,), kwargs = {})
#   %convert_element_type_165 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_64, torch.float32), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_165, %exp2_34), kwargs = {})
#   %clamp_min_69 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_122, -448.0), kwargs = {})
#   %clamp_max_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_69, 448.0), kwargs = {})
#   %convert_element_type_166 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_34, torch.float8_e4m3fn), kwargs = {})
#   %_scaled_mm_17 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_166, %permute_70, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_39 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_39(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.bfloat16)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl_math.abs(tmp3)
    tmp5 = tmp4.to(tl.bfloat16)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp9 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp7, 0))
    tmp10 = tmp1.to(tl.float32)
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
    tl.store(out_ptr1 + (r0_1 + 256*x0), tmp27, None)
    tl.store(out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/4c/c4cq4sd4z4sj3v7a2hdmyxfd2pphdfe7pycudnjutz32wvfa637p.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_7, 0), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_1, %mul_124), kwargs = {})
#   %abs_36 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_13,), kwargs = {})
#   %amax_35 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_36, [-1], True), kwargs = {})
triton_red_fused_abs_add_amax_mul_40 = async_compile.triton('triton_red_fused_abs_add_amax_mul_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_add_amax_mul_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_add_amax_mul_40(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4.to(tl.bfloat16)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = 0.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.bfloat16)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10.to(tl.bfloat16)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 * tmp7
        tmp14 = tmp13.to(tl.bfloat16)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15.to(tl.bfloat16)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp2 + tmp17
        tmp19 = tmp18.to(tl.bfloat16)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20.to(tl.bfloat16)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl_math.abs(tmp22)
        tmp24 = tmp23.to(tl.bfloat16)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = triton_helpers.maximum(_tmp27, tmp26)
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp27 = triton_helpers.max2(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/7l/c7lzfm7tdl2omwzs23ccwkhnf62cftsy5cp5tgc2ewj46c52jhzv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_163 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_34, torch.float64), kwargs = {})
#   %clamp_min_68 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_163, 1e-12), kwargs = {})
#   %reciprocal_68 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_68,), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_68, 448.0), kwargs = {})
#   %convert_element_type_164 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_121, torch.float32), kwargs = {})
#   %log2_34 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_164,), kwargs = {})
#   %floor_34 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_34,), kwargs = {})
#   %exp2_34 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_34,), kwargs = {})
#   %convert_element_type_165 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_64, torch.float32), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_165, %exp2_34), kwargs = {})
#   %clamp_min_69 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_122, -448.0), kwargs = {})
#   %clamp_max_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_69, 448.0), kwargs = {})
#   %convert_element_type_166 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_34, torch.float8_e4m3fn), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_7, 0), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_1, %mul_124), kwargs = {})
#   %convert_element_type_167 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_35, torch.float64), kwargs = {})
#   %clamp_min_70 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_167, 1e-12), kwargs = {})
#   %reciprocal_69 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_70,), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_69, 448.0), kwargs = {})
#   %convert_element_type_168 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_125, torch.float32), kwargs = {})
#   %log2_35 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_168,), kwargs = {})
#   %floor_35 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_35,), kwargs = {})
#   %exp2_35 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_35,), kwargs = {})
#   %convert_element_type_169 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.float32), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_169, %exp2_35), kwargs = {})
#   %_scaled_mm_17 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%convert_element_type_166, %permute_70, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_41 = async_compile.triton('triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_41(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask & ymask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4.to(tl.bfloat16)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.bfloat16)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10.to(tl.bfloat16)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp7
    tmp14 = tmp13.to(tl.bfloat16)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.bfloat16)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp2 + tmp17
    tmp19 = tmp18.to(tl.bfloat16)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp22.to(tl.float64)
    tmp24 = tl.full([1, 1], 1e-12, tl.float64)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp26 = tl.full([1, 1], 1, tl.int32)
    tmp27 = (tmp26 / tmp25)
    tmp28 = tl.full([1, 1], 448.0, tl.float64)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = libdevice.log2(tmp30)
    tmp32 = libdevice.floor(tmp31)
    tmp33 = libdevice.exp2(tmp32)
    tmp34 = tmp21 * tmp33
    tmp35 = -448.0
    tmp36 = triton_helpers.maximum(tmp34, tmp35)
    tmp37 = 448.0
    tmp38 = triton_helpers.minimum(tmp36, tmp37)
    tmp39 = tmp38.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y0 + 256*x1), tmp39, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/ov/covnbiuram6zudfys3jgtmdb3iwhzkokxrgqy67xvfxrzswmwsmf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_68, %view_69), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %view_74), kwargs = {})
triton_poi_fused_add_42 = async_compile.triton('triton_poi_fused_add_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 256
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp30 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp32 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp42 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp62 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp64 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp74 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1], 1e-12, tl.float64)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tl.full([1], 448.0, tl.float64)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = libdevice.floor(tmp11)
    tmp13 = libdevice.exp2(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = triton_helpers.maximum(tmp16, tmp4)
    tmp18 = (tmp6 / tmp17)
    tmp19 = tmp18 * tmp8
    tmp20 = tmp19.to(tl.float32)
    tmp21 = libdevice.log2(tmp20)
    tmp22 = libdevice.floor(tmp21)
    tmp23 = libdevice.exp2(tmp22)
    tmp24 = (tmp6 / tmp23)
    tmp25 = tmp14 * tmp24
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.bfloat16)
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp33 = tmp32.to(tl.float64)
    tmp34 = triton_helpers.maximum(tmp33, tmp4)
    tmp35 = (tmp6 / tmp34)
    tmp36 = tmp35 * tmp8
    tmp37 = tmp36.to(tl.float32)
    tmp38 = libdevice.log2(tmp37)
    tmp39 = libdevice.floor(tmp38)
    tmp40 = libdevice.exp2(tmp39)
    tmp41 = (tmp6 / tmp40)
    tmp43 = tmp42.to(tl.float64)
    tmp44 = triton_helpers.maximum(tmp43, tmp4)
    tmp45 = (tmp6 / tmp44)
    tmp46 = tmp45 * tmp8
    tmp47 = tmp46.to(tl.float32)
    tmp48 = libdevice.log2(tmp47)
    tmp49 = libdevice.floor(tmp48)
    tmp50 = libdevice.exp2(tmp49)
    tmp51 = (tmp6 / tmp50)
    tmp52 = tmp41 * tmp51
    tmp53 = tmp31 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp54.to(tl.bfloat16)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp29 + tmp56
    tmp58 = tmp57.to(tl.bfloat16)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp59.to(tl.bfloat16)
    tmp61 = tmp60.to(tl.float32)
    tmp63 = tmp62.to(tl.float32)
    tmp65 = tmp64.to(tl.float64)
    tmp66 = triton_helpers.maximum(tmp65, tmp4)
    tmp67 = (tmp6 / tmp66)
    tmp68 = tmp67 * tmp8
    tmp69 = tmp68.to(tl.float32)
    tmp70 = libdevice.log2(tmp69)
    tmp71 = libdevice.floor(tmp70)
    tmp72 = libdevice.exp2(tmp71)
    tmp73 = (tmp6 / tmp72)
    tmp75 = tmp74.to(tl.float64)
    tmp76 = triton_helpers.maximum(tmp75, tmp4)
    tmp77 = (tmp6 / tmp76)
    tmp78 = tmp77 * tmp8
    tmp79 = tmp78.to(tl.float32)
    tmp80 = libdevice.log2(tmp79)
    tmp81 = libdevice.floor(tmp80)
    tmp82 = libdevice.exp2(tmp81)
    tmp83 = (tmp6 / tmp82)
    tmp84 = tmp73 * tmp83
    tmp85 = tmp63 * tmp84
    tmp86 = tmp85.to(tl.float32)
    tmp87 = tmp86.to(tl.bfloat16)
    tmp88 = tmp87.to(tl.float32)
    tmp89 = tmp61 + tmp88
    tmp90 = tmp89.to(tl.bfloat16)
    tmp91 = tmp90.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp91, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/s4/cs4bwckogmhta7noondxznmygbwilma4a4cvzfyi62ryabthi3nc.py
# Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.sum]
# Source node to ATen node mapping:
#   rms_norm => add, convert_element_type, mean, mul, pow_1, rsqrt
# Graph fragment:
#   %convert_element_type : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 9.999999747378752e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %abs_34 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_63,), kwargs = {})
#   %amax_33 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_34, [0], True), kwargs = {})
#   %convert_element_type_199 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_16, torch.float32), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_199, %mul), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_149, [0, 1], True), kwargs = {})
triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_43 = async_compile.triton('triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 256.0
        tmp4 = (tmp2 / tmp3)
        tmp5 = 9.999999747378752e-06
        tmp6 = tmp4 + tmp5
        tmp7 = libdevice.rsqrt(tmp6)
        tmp8 = tmp1 * tmp7
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12.to(tl.bfloat16)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl_math.abs(tmp14)
        tmp16 = tmp15.to(tl.bfloat16)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = triton_helpers.maximum(_tmp19, tmp18)
        _tmp19 = tl.where(r0_mask, tmp20, _tmp19)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 * tmp8
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask, tmp26, _tmp25)
    tmp19 = triton_helpers.max2(_tmp19, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, None)
    tl.store(out_ptr1 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/kq/ckqz2oyud5enqautoxpojr6ujaqdgynuqkclpowl3xahmd6idcja.py
# Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.clone, aten._scaled_mm, aten.sum, aten.div]
# Source node to ATen node mapping:
#   rms_norm => add, convert_element_type, mean, pow_1, rsqrt
# Graph fragment:
#   %convert_element_type_122 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.bfloat16), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %convert_element_type_122), kwargs = {})
#   %convert_element_type : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 9.999999747378752e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %convert_element_type_158 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_33, torch.float64), kwargs = {})
#   %clamp_min_66 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_158, 1e-12), kwargs = {})
#   %reciprocal_65 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_66,), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_65, 448.0), kwargs = {})
#   %convert_element_type_159 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_117, torch.float32), kwargs = {})
#   %log2_33 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%convert_element_type_159,), kwargs = {})
#   %floor_33 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%log2_33,), kwargs = {})
#   %exp2_33 : [num_users=2] = call_function[target=torch.ops.aten.exp2.default](args = (%floor_33,), kwargs = {})
#   %convert_element_type_160 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_63, torch.float32), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_160, %exp2_33), kwargs = {})
#   %clamp_min_67 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_118, -448.0), kwargs = {})
#   %clamp_max_33 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_67, 448.0), kwargs = {})
#   %convert_element_type_161 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_33, torch.float8_e4m3fn), kwargs = {})
#   %clone_15 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_61,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_16 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_15, %permute_64, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
#   %clone_18 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_71,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_18 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_18, %permute_64, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
#   %clone_21 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_81,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_20 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_21, %permute_64, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
#   %convert_element_type_199 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_16, torch.float32), kwargs = {})
#   %mul_150 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_199, %primals_3), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_150, %convert_element_type), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_150, %rsqrt), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_151, [2], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_1, 256), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 1.0), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_6, 2.0), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %mul_155), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_152, %mul_156), kwargs = {})
#   %convert_element_type_201 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.bfloat16), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %convert_element_type_201), kwargs = {})
triton_red_fused__scaled_mm__to_copy_add_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_44 = async_compile.triton('triton_red_fused__scaled_mm__to_copy_add_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'out_ptr1': '*fp8e4nv', 'out_ptr2': '*fp8e4nv', 'out_ptr3': '*fp8e4nv', 'out_ptr4': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__scaled_mm__to_copy_add_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__scaled_mm__to_copy_add_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
        tmp12 = 256.0
        tmp13 = (tmp11 / tmp12)
        tmp14 = 9.999999747378752e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp6 * tmp16
        tmp18 = tmp17 * tmp3
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float64)
        tmp23 = tl.full([1, 1], 1e-12, tl.float64)
        tmp24 = triton_helpers.maximum(tmp22, tmp23)
        tmp25 = tl.full([1, 1], 1, tl.int32)
        tmp26 = (tmp25 / tmp24)
        tmp27 = tl.full([1, 1], 448.0, tl.float64)
        tmp28 = tmp26 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = libdevice.log2(tmp29)
        tmp31 = libdevice.floor(tmp30)
        tmp32 = libdevice.exp2(tmp31)
        tmp33 = tmp20 * tmp32
        tmp34 = -448.0
        tmp35 = triton_helpers.maximum(tmp33, tmp34)
        tmp36 = 448.0
        tmp37 = triton_helpers.minimum(tmp35, tmp36)
        tmp38 = tmp37.to(tl.float8e4nv)
        tl.store(out_ptr1 + (r0_1 + 256*x0), tmp38, r0_mask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp39 = tl.load(in_ptr4 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr5 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp51 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp69 = tl.load(in_ptr1 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp81 = tl.load(out_ptr1 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp40 = tmp39.to(tl.bfloat16)
        tmp41 = tmp40.to(tl.float32)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp43.to(tl.bfloat16)
        tmp45 = tmp44.to(tl.float32)
        tmp46 = tmp41 + tmp45
        tmp47 = tmp46.to(tl.bfloat16)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tmp48.to(tl.bfloat16)
        tmp50 = tmp49.to(tl.float32)
        tmp52 = tmp51.to(tl.float32)
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tmp52 * tmp54
        tmp56 = 256.0
        tmp57 = (tmp11 / tmp56)
        tmp58 = 9.999999747378752e-06
        tmp59 = tmp57 + tmp58
        tmp60 = libdevice.rsqrt(tmp59)
        tmp61 = tmp55 * tmp60
        tmp62 = -0.5
        tmp63 = tmp9 * tmp62
        tmp64 = tmp60 * tmp60
        tmp65 = tmp64 * tmp60
        tmp66 = tmp63 * tmp65
        tmp67 = 0.00390625
        tmp68 = tmp66 * tmp67
        tmp70 = tmp69.to(tl.float32)
        tmp71 = 2.0
        tmp72 = tmp70 * tmp71
        tmp73 = tmp68 * tmp72
        tmp74 = tmp61 + tmp73
        tmp75 = tmp74.to(tl.float32)
        tmp76 = tmp75.to(tl.bfloat16)
        tmp77 = tmp76.to(tl.float32)
        tmp78 = tmp50 + tmp77
        tmp79 = tmp78.to(tl.bfloat16)
        tmp80 = tmp79.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp80, r0_mask)
        tl.store(out_ptr2 + (x0 + 16384*r0_1), tmp81, r0_mask)
        tl.store(out_ptr3 + (x0 + 16384*r0_1), tmp81, r0_mask)
        tl.store(out_ptr4 + (x0 + 16384*r0_1), tmp81, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/sl/cslyyxmtwr5fshy6hnpanarqq5boluannkttmpyrqq4zpuo4u3f7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
# Source node to ATen node mapping:
# Graph fragment:
#   %abs_35 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%view_64,), kwargs = {})
#   %amax_36 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_35, [0], True), kwargs = {})
triton_red_fused_abs_amax_45 = async_compile.triton('triton_red_fused_abs_amax_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_amax_45(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1.to(tl.bfloat16)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl_math.abs(tmp3)
        tmp5 = tmp4.to(tl.bfloat16)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_hirsheybar/eg/ceg372gqnvcajk23apx5nfucztoozuw56g6b47fv2plak7w3ramt.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_18 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_71,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_mm_18 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%clone_18, %permute_64, %full_default, %full_default, None, None, torch.bfloat16), kwargs = {})
triton_poi_fused__scaled_mm_clone_46 = async_compile.triton('triton_poi_fused__scaled_mm_clone_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm_clone_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C51E3648951F99808FEE4FFFB5092F3EEB2A1728A25E14F7852B0BB8AA4F42AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm_clone_46(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tl.full([1, 1], 1e-12, tl.float64)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1, 1], 448.0, tl.float64)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = libdevice.log2(tmp11)
    tmp13 = libdevice.floor(tmp12)
    tmp14 = libdevice.exp2(tmp13)
    tmp15 = tmp2 * tmp14
    tmp16 = -448.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 448.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp19.to(tl.float8e4nv)
    tl.store(out_ptr0 + (x1 + 16384*y0), tmp20, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, reciprocal_2, _scaled_mm, _scaled_mm_2, getitem, getitem_1, getitem_6, getitem_7, reciprocal_18, _scaled_mm_4, tangents_1 = args
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
    assert_size_stride(reciprocal_2, (16384, 1), (1, 1))
    assert_size_stride(_scaled_mm, (16384, 256), (256, 1))
    assert_size_stride(_scaled_mm_2, (16384, 256), (256, 1))
    assert_size_stride(getitem, (8, 16, 2048, 16), (524288, 16, 256, 1))
    assert_size_stride(getitem_1, (8, 16, 2048), (32768, 2048, 1))
    assert_size_stride(getitem_6, (2, ), (1, ))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(reciprocal_18, (16384, 1), (1, 1))
    assert_size_stride(_scaled_mm_4, (16384, 768), (768, 1))
    assert_size_stride(tangents_1, (8, 2048, 256), (524288, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 1), (1, 16384), torch.bfloat16)
        buf4 = empty_strided_cuda((16384, 256), (256, 1), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.new_ones, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        # danvm: buf4 is float8 output, 
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_new_ones_reciprocal_0.run(tangents_1, buf0, buf4, 16384, 256, stream=stream0)
        buf1 = empty_strided_cuda((768, 1, 2), (1, 1536, 768), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_amax_mul_1.run(primals_11, tangents_1, buf1, 1536, 128, stream=stream0)
        buf2 = empty_strided_cuda((768, 1), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_add_amax_mul_2.run(buf1, buf2, 768, 2, stream=stream0)
        buf5 = empty_strided_cuda((256, 768), (1, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten.new_ones, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_new_ones_reciprocal_3.run(primals_11, tangents_1, buf2, buf5, 256, 768, stream=stream0)
        del primals_11
        buf6 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.new_ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_ones_4.run(buf6, 1, stream=stream0)
        buf7 = empty_strided_cuda((16384, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.new_ones, aten._scaled_mm]
        # danvm: this scaled mm output buf7 is part of K ancestry
        extern_kernels._scaled_mm(buf4, buf5, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf7)
        buf10 = buf4; del buf4  # reuse
        buf12 = empty_strided_cuda((16384, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_6], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_5.run(getitem, buf10, buf12, 16384, 256, stream=stream0)
        buf11 = empty_strided_cuda((256, 256), (1, 256), torch.float8_e4m3fn)
        buf13 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_6], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6.run(primals_7, buf11, buf13, 256, 256, stream=stream0)
        buf14 = empty_strided_cuda((16384, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [output_6], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf10, buf11, buf12, buf13, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf14)
        buf15 = reinterpret_tensor(buf12, (8, 2048, 1), (2048, 1, 16384), 0); del buf12  # reuse
        buf19 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [h, rms_norm_1, output_7, output_8], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.clamp, aten.reciprocal, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_add_amax_clamp_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_7.run(primals_1, buf14, primals_8, buf15, buf19, 16384, 256, stream=stream0)
        buf20 = buf5; del buf5  # reuse
        buf21 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_8], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_8.run(primals_10, buf20, buf21, 768, 256, stream=stream0)
        buf22 = empty_strided_cuda((16384, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [output_8], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf19, buf20, reciprocal_18, buf21, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf22)
        del buf21
        del reciprocal_18
        buf25 = empty_strided_cuda((1, 768, 128), (98304, 1, 768), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_9.run(_scaled_mm_4, buf22, buf25, 98304, 128, stream=stream0)
        buf26 = empty_strided_cuda((1, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_10.run(buf25, buf26, 768, 128, stream=stream0)
        buf29 = empty_strided_cuda((16384, 768), (1, 16384), torch.float8_e4m3fn)
        buf32 = empty_strided_cuda((8, 2048, 768), (1572864, 768, 1), torch.bfloat16)
        buf33 = empty_strided_cuda((8, 2048, 768), (1572864, 768, 1), torch.bfloat16)
        buf50 = buf33; del buf33  # reuse
        buf51 = empty_strided_cuda((16384, 1), (1, 16384), torch.bfloat16)
        buf55 = empty_strided_cuda((16384, 768), (768, 1), torch.float8_e4m3fn)
        buf34 = empty_strided_cuda((16384, 1), (1, 16384), torch.bfloat16)
        buf38 = empty_strided_cuda((16384, 768), (768, 1), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [silu], Original ATen: [aten.silu, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.clone, aten._scaled_mm, aten.abs, aten.amax, aten.sigmoid, aten.fill, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_exp2_fill_floor_log2_mul_reciprocal_sigmoid_silu_sub_11.run(buf50, _scaled_mm_4, buf22, buf26, buf7, buf0, buf2, buf29, buf32, buf51, buf55, buf34, buf38, 16384, 768, stream=stream0)
        del _scaled_mm_4
        del buf2
        del buf22
        del buf7
        buf35 = reinterpret_tensor(buf1, (256, 1, 6), (1, 1536, 256), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_amax_mul_12.run(primals_10, buf32, buf35, 1536, 128, stream=stream0)
        buf36 = empty_strided_cuda((256, 1), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_add_amax_mul_13.run(buf35, buf36, 256, 6, stream=stream0)
        buf39 = reinterpret_tensor(buf20, (768, 256), (1, 768), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_14.run(primals_10, buf32, buf36, buf39, 768, 256, stream=stream0)
        del primals_10
        buf40 = empty_strided_cuda((16384, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf38, buf39, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf40)
        del buf38
        buf52 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_amax_mul_12.run(primals_9, buf50, buf52, 1536, 128, stream=stream0)
        buf53 = empty_strided_cuda((256, 1), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_add_amax_mul_13.run(buf52, buf53, 256, 6, stream=stream0)
        del buf52
        buf56 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_14.run(primals_9, buf50, buf53, buf56, 768, 256, stream=stream0)
        del primals_9
        buf57 = empty_strided_cuda((16384, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf55, buf56, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf57)
        del buf55
        del buf56
        buf64 = reinterpret_tensor(buf40, (8, 2048, 256), (524288, 256, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_15.run(buf64, buf34, buf36, buf57, buf51, buf53, 4194304, stream=stream0)
        buf43 = empty_strided_cuda((1, 256, 128), (32768, 1, 256), torch.float32)
        buf65 = empty_strided_cuda((1, 1, 256, 128), (32768, 32768, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h, rms_norm_1], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_16.run(primals_1, buf14, buf15, primals_8, buf64, buf43, buf65, 32768, 128, stream=stream0)
        buf44 = reinterpret_tensor(buf53, (1, 256), (256, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_17.run(buf43, buf44, 256, 128, stream=stream0)
        buf85 = empty_strided_cuda((8, 2048, 1), (2048, 1, 16384), torch.float32)
        buf69 = empty_strided_cuda((8, 2048, 256), (524288, 256, 1), torch.float32)
        buf47 = reinterpret_tensor(buf19, (16384, 256), (1, 16384), 0); del buf19  # reuse
        buf61 = empty_strided_cuda((16384, 256), (1, 16384), torch.float8_e4m3fn)
        buf87 = empty_strided_cuda((8, 2048, 256), (524288, 256, 1), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [h, rms_norm_1, rms_norm, output], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.clone, aten._scaled_mm, aten.sum, aten.div, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_18.run(buf64, primals_8, primals_1, buf14, primals_3, buf15, buf44, buf85, buf69, buf47, buf61, buf87, 16384, 256, stream=stream0)
        del buf15
        del primals_8
        buf23 = buf43; del buf43  # reuse
        buf77 = empty_strided_cuda((1, 256, 128), (32768, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_19.run(tangents_1, buf69, buf23, buf77, 32768, 128, stream=stream0)
        buf24 = reinterpret_tensor(buf36, (1, 256), (256, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_17.run(buf23, buf24, 256, 128, stream=stream0)
        del buf23
        buf28 = empty_strided_cuda((256, 16384), (16384, 1), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm_clone_20.run(tangents_1, buf24, buf28, 256, 16384, stream=stream0)
        buf30 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        extern_kernels._scaled_mm(buf28, buf29, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf30)
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_21.run(buf31, buf24, buf26, 196608, stream=stream0)
        buf41 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_22.run(buf32, buf41, 98304, 128, stream=stream0)
        buf42 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_10.run(buf41, buf42, 768, 128, stream=stream0)
        buf46 = reinterpret_tensor(buf29, (768, 16384), (16384, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm_clone_23.run(buf32, buf42, buf46, 768, 16384, stream=stream0)
        del buf32
        buf48 = empty_strided_cuda((768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        extern_kernels._scaled_mm(buf46, buf47, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf48)
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_24.run(buf49, buf42, buf44, 196608, stream=stream0)
        buf58 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_22.run(buf50, buf58, 98304, 128, stream=stream0)
        buf59 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_10.run(buf58, buf59, 768, 128, stream=stream0)
        del buf58
        buf60 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm_clone_23.run(buf50, buf59, buf60, 768, 16384, stream=stream0)
        del buf50
        buf62 = empty_strided_cuda((768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        extern_kernels._scaled_mm(buf60, buf61, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf62)
        del buf60
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_24.run(buf63, buf59, buf44, 196608, stream=stream0)
        del buf59
        buf67 = reinterpret_tensor(buf44, (256, ), (1, ), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [h, rms_norm_1], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_25.run(buf65, buf67, 256, 128, stream=stream0)
        buf78 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_17.run(buf77, buf78, 256, 128, stream=stream0)
        buf70 = buf51; del buf51  # reuse
        buf74 = reinterpret_tensor(buf61, (16384, 256), (256, 1), 0); del buf61  # reuse
        buf81 = reinterpret_tensor(buf47, (256, 16384), (16384, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm, aten.clone]
        stream0 = get_raw_stream(0)
        triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_clone_exp2_floor_log2_mul_reciprocal_26.run(tangents_1, buf69, buf78, buf70, buf74, buf81, 16384, 256, stream=stream0)
        buf71 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_27.run(primals_7, tangents_1, buf69, buf71, 65536, stream=stream0)
        del primals_7
        buf72 = empty_strided_cuda((256, 1, 2), (1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_28.run(buf71, buf72, 512, 128, stream=stream0)
        buf73 = empty_strided_cuda((256, 1), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_amax_29.run(buf72, buf73, 256, 2, stream=stream0)
        buf75 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_clamp_exp2_floor_log2_mul_reciprocal_30.run(buf71, buf73, buf75, 256, 256, stream=stream0)
        buf76 = reinterpret_tensor(buf64, (16384, 256), (256, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf74, buf75, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf76)
        buf79 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_31.run(getitem, buf79, 32768, 128, stream=stream0)
        buf80 = empty_strided_cuda((1, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_17.run(buf79, buf80, 256, 128, stream=stream0)
        buf82 = reinterpret_tensor(buf74, (16384, 256), (1, 16384), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm_clone_32.run(getitem, buf80, buf82, 16384, 256, stream=stream0)
        buf83 = reinterpret_tensor(buf71, (256, 256), (256, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        extern_kernels._scaled_mm(buf81, buf82, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf83)
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_33.run(buf84, buf78, buf80, 65536, stream=stream0)
        buf89 = buf75; del buf75  # reuse
        buf90 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_6.run(primals_5, buf89, buf90, 256, 256, stream=stream0)
        buf91 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(reinterpret_tensor(buf87, (16384, 256), (256, 1), 0), buf89, reciprocal_2, buf90, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf91)
        del buf90
        del reciprocal_2
        buf92 = empty_strided_cuda((8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xq_], Original ATen: [aten.view_as_complex]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_as_complex_34.run(_scaled_mm, buf92, 4194304, stream=stream0)
        del _scaled_mm
        # Topologically Sorted Source Nodes: [xq_], Original ATen: [aten.view_as_complex]
        buf93 = torch.ops.aten.view_as_complex.default(buf92)
        buf94 = buf93
        assert_size_stride(buf94, (8, 2048, 16, 8), (262144, 128, 8, 1))
        buf95 = empty_strided_cuda((8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xk_], Original ATen: [aten.view_as_complex]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_as_complex_34.run(buf91, buf95, 4194304, stream=stream0)
        # Topologically Sorted Source Nodes: [xk_], Original ATen: [aten.view_as_complex]
        buf96 = torch.ops.aten.view_as_complex.default(buf95)
        buf97 = buf96
        assert_size_stride(buf97, (8, 2048, 16, 8), (262144, 128, 8, 1))
        # Topologically Sorted Source Nodes: [freqs_cis], Original ATen: [aten.slice]
        buf98 = torch.ops.aten.slice.Tensor(primals_2, 0, 0, 2048)
        buf99 = buf98
        assert_size_stride(buf99, (2048, 8), (8, 1))
        # Topologically Sorted Source Nodes: [freqs_cis_1], Original ATen: [aten.view]
        buf100 = torch.ops.aten.reshape.default(buf99, [1, 2048, 1, 8])
        buf101 = buf100
        assert_size_stride(buf101, (1, 2048, 1, 8), (16384, 8, 8, 1))
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        buf102 = torch.ops.aten.mul.Tensor(buf94, buf101)
        del buf92
        del buf93
        del buf94
        buf103 = buf102
        assert_size_stride(buf103, (8, 2048, 16, 8), (262144, 128, 8, 1))
        del buf102
        # Topologically Sorted Source Nodes: [view_as_real], Original ATen: [aten.view_as_real]
        buf104 = torch.ops.aten.view_as_real.default(buf103)
        buf105 = buf104
        assert_size_stride(buf105, (8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1))
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        buf106 = torch.ops.aten.mul.Tensor(buf97, buf101)
        del buf96
        del buf97
        buf107 = buf106
        assert_size_stride(buf107, (8, 2048, 16, 8), (262144, 128, 8, 1))
        del buf106
        # Topologically Sorted Source Nodes: [view_as_real_1], Original ATen: [aten.view_as_real]
        buf108 = torch.ops.aten.view_as_real.default(buf107)
        buf109 = buf108
        assert_size_stride(buf109, (8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1))
        buf110 = reinterpret_tensor(buf76, (8, 16, 2048, 16), (524288, 16, 256, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward_35.run(buf110, buf70, buf73, 4194304, stream=stream0)
        buf111 = reinterpret_tensor(buf91, (8, 16, 2048, 16), (524288, 16, 256, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward_36.run(buf105, buf111, 4194304, stream=stream0)
        del buf103
        del buf104
        del buf105
        # danvm: buff57 is K
        buf112 = reinterpret_tensor(buf57, (8, 16, 2048, 16), (524288, 16, 256, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward_36.run(buf109, buf112, 4194304, stream=stream0)
        del buf107
        del buf108
        del buf109
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        # danvm: buf112 is K
        buf113 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(buf110, buf111, buf112, reinterpret_tensor(_scaled_mm_2, (8, 16, 2048, 16), (524288, 16, 256, 1), 0), getitem, getitem_1, None, None, 2048, 2048, 0.0, True, getitem_6, getitem_7, scale=0.25)
        del _scaled_mm_2
        del buf110
        del buf111
        del getitem
        del getitem_1
        del getitem_6
        del getitem_7
        buf114 = buf113[0]
        assert_size_stride(buf114, (8, 16, 2048, 16), (524288, 16, 256, 1))
        buf115 = buf113[1]
        assert_size_stride(buf115, (8, 16, 2048, 16), (524288, 16, 256, 1))
        buf116 = buf113[2]
        assert_size_stride(buf116, (8, 16, 2048, 16), (524288, 16, 256, 1))
        del buf113
        buf117 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_as_complex_34.run(buf115, buf117, 4194304, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
        buf118 = torch.ops.aten.view_as_complex.default(buf117)
        buf119 = buf118
        assert_size_stride(buf119, (8, 2048, 16, 8), (262144, 128, 8, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._conj]
        buf120 = torch.ops.aten._conj.default(buf101)
        buf121 = buf120
        assert_size_stride(buf121, (1, 2048, 1, 8), (16384, 8, 8, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        buf122 = torch.ops.aten.mul.Tensor(buf119, buf121)
        del buf118
        del buf119
        buf123 = buf122
        assert_size_stride(buf123, (8, 2048, 16, 8), (262144, 128, 8, 1))
        del buf122
        buf124 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_as_complex_34.run(buf114, buf124, 4194304, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
        buf125 = torch.ops.aten.view_as_complex.default(buf124)
        buf126 = buf125
        assert_size_stride(buf126, (8, 2048, 16, 8), (262144, 128, 8, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        buf127 = torch.ops.aten.mul.Tensor(buf126, buf121)
        del buf100
        del buf101
        del buf120
        del buf121
        del buf124
        del buf125
        del buf126
        del buf98
        del buf99
        del primals_2
        buf128 = buf127
        assert_size_stride(buf128, (8, 2048, 16, 8), (262144, 128, 8, 1))
        del buf127
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_real]
        buf129 = torch.ops.aten.view_as_real.default(buf123)
        buf130 = buf129
        assert_size_stride(buf130, (8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_real]
        buf131 = torch.ops.aten.view_as_real.default(buf128)
        buf132 = buf131
        assert_size_stride(buf132, (8, 2048, 16, 8, 2), (524288, 256, 16, 2, 1))
        buf133 = buf70; del buf70  # reuse
        buf137 = reinterpret_tensor(buf87, (16384, 256), (256, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_new_ones_reciprocal_0.run(buf116, buf133, buf137, 16384, 256, stream=stream0)
        buf134 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_amax_mul_37.run(primals_6, buf116, buf134, 512, 128, stream=stream0)
        buf135 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_amax_29.run(buf134, buf135, 256, 2, stream=stream0)
        buf138 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_38.run(primals_6, buf116, buf135, buf138, 256, 256, stream=stream0)
        del primals_6
        buf139 = reinterpret_tensor(buf114, (16384, 256), (256, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf137, buf138, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf139)
        buf140 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_31.run(buf116, buf140, 32768, 128, stream=stream0)
        buf141 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_17.run(buf140, buf141, 256, 128, stream=stream0)
        buf149 = buf34; del buf34  # reuse
        buf153 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_39.run(buf130, buf149, buf153, 16384, 256, stream=stream0)
        buf150 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_amax_mul_40.run(primals_5, buf130, buf150, 512, 128, stream=stream0)
        buf151 = reinterpret_tensor(buf78, (256, 1), (1, 256), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_amax_29.run(buf150, buf151, 256, 2, stream=stream0)
        buf154 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_41.run(primals_5, buf130, buf151, buf154, 256, 256, stream=stream0)
        del primals_5
        buf155 = reinterpret_tensor(buf115, (16384, 256), (256, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf153, buf154, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf155)
        buf163 = buf0; del buf0  # reuse
        buf167 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_exp2_floor_log2_mul_reciprocal_39.run(buf132, buf163, buf167, 16384, 256, stream=stream0)
        buf164 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_amax_mul_40.run(primals_4, buf132, buf164, 512, 128, stream=stream0)
        buf165 = empty_strided_cuda((256, 1), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_amax_29.run(buf164, buf165, 256, 2, stream=stream0)
        del buf164
        buf168 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.add, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_add_clamp_exp2_floor_log2_mul_reciprocal_41.run(primals_4, buf132, buf165, buf168, 256, 256, stream=stream0)
        del primals_4
        buf169 = reinterpret_tensor(buf112, (16384, 256), (256, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten._scaled_mm]
        extern_kernels._scaled_mm(buf167, buf168, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf169)
        del buf168
        buf162 = reinterpret_tensor(buf139, (8, 2048, 256), (524288, 256, 1), 0); del buf139  # reuse
        buf176 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_42.run(buf176, buf133, buf135, buf155, buf149, buf151, buf169, buf163, buf165, 4194304, stream=stream0)
        del buf133
        del buf135
        del buf149
        del buf151
        del buf155
        del buf163
        del buf169
        buf142 = buf140; del buf140  # reuse
        buf177 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_abs_add_amax_mean_mul_pow_rsqrt_sum_43.run(primals_1, buf85, primals_3, buf176, buf142, buf177, 32768, 128, stream=stream0)
        buf143 = reinterpret_tensor(buf165, (1, 256), (256, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_17.run(buf142, buf143, 256, 128, stream=stream0)
        buf144 = buf167; del buf167  # reuse
        buf181 = buf176; del buf176  # reuse
        buf146 = buf82; del buf82  # reuse
        buf159 = reinterpret_tensor(buf81, (16384, 256), (1, 16384), 0); del buf81  # reuse
        buf173 = reinterpret_tensor(buf28, (16384, 256), (1, 16384), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2, aten.clone, aten._scaled_mm, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused__scaled_mm__to_copy_add_clamp_clone_div_exp2_floor_log2_mean_mul_pow_reciprocal_rsqrt_sum_44.run(buf181, primals_3, primals_1, buf85, buf143, tangents_1, buf69, buf144, buf146, buf159, buf173, 16384, 256, stream=stream0)
        del buf69
        del buf85
        del primals_1
        del primals_3
        del tangents_1
        buf145 = reinterpret_tensor(buf144, (256, 16384), (16384, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm_clone_20.run(buf116, buf141, buf145, 256, 16384, stream=stream0)
        del buf116
        buf147 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        extern_kernels._scaled_mm(buf145, buf146, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf147)
        del buf145
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_33.run(buf148, buf141, buf143, 65536, stream=stream0)
        buf156 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_45.run(buf130, buf156, 32768, 128, stream=stream0)
        buf157 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_17.run(buf156, buf157, 256, 128, stream=stream0)
        buf158 = reinterpret_tensor(buf146, (256, 16384), (16384, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm_clone_46.run(buf130, buf157, buf158, 256, 16384, stream=stream0)
        del buf123
        del buf129
        del buf130
        buf160 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        extern_kernels._scaled_mm(buf158, buf159, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf160)
        del buf158
        buf161 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_33.run(buf161, buf157, buf143, 65536, stream=stream0)
        buf170 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_45.run(buf132, buf170, 32768, 128, stream=stream0)
        buf171 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_17.run(buf170, buf171, 256, 128, stream=stream0)
        del buf170
        buf172 = reinterpret_tensor(buf159, (256, 16384), (16384, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm_clone_46.run(buf132, buf171, buf172, 256, 16384, stream=stream0)
        del buf128
        del buf131
        del buf132
        buf174 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._scaled_mm]
        extern_kernels._scaled_mm(buf172, buf173, buf6, buf6, out_dtype=torch.bfloat16, use_fast_accum=False, out=buf174)
        del buf172
        del buf173
        del buf6
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten.log2, aten.floor, aten.exp2]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_exp2_floor_log2_mul_reciprocal_33.run(buf175, buf171, buf143, 65536, stream=stream0)
        del buf143
        buf179 = reinterpret_tensor(buf171, (256, ), (1, ), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_25.run(buf177, buf179, 256, 128, stream=stream0)
        del buf177
    return (buf181, None, buf179, buf175, buf161, buf148, buf84, buf67, buf63, buf49, buf31, )


def benchmark_compiled_module(times=10, repeat=10):
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
    reciprocal_2 = rand_strided((16384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    _scaled_mm = rand_strided((16384, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    _scaled_mm_2 = rand_strided((16384, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem = rand_strided((8, 16, 2048, 16), (524288, 16, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((8, 16, 2048), (32768, 2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    reciprocal_18 = rand_strided((16384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    _scaled_mm_4 = rand_strided((16384, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((8, 2048, 256), (524288, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, reciprocal_2, _scaled_mm, _scaled_mm_2, getitem, getitem_1, getitem_6, getitem_7, reciprocal_18, _scaled_mm_4, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
