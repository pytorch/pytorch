
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_chilli/54/c54qoo5uewkqsadvbtdpr72n3w6c4xmz5hbggymhfbfohouz2skj.py
# Source Nodes: [encoder_layer], Original ATen: [aten._scaled_dot_product_efficient_attention]
# encoder_layer => _scaled_dot_product_efficient_attention
triton_poi_fused__scaled_dot_product_efficient_attention_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '2af0d8d97d2323882e9b91da64c8a9ac39ea8a97ea88d5425e2a8cf77d3f1889', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 32
    x2 = (xindex // 16384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*x1) + (49152*x2) + (49152*((x0 + (512*x1)) // 16384))), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_chilli/sh/csh66kzg42n5nmdgnaoqmzduljnlybcgm6vsb2e7imn5jg7lnpac.py
# Source Nodes: [encoder_layer], Original ATen: [aten._scaled_dot_product_efficient_attention]
# encoder_layer => _scaled_dot_product_efficient_attention
triton_poi_fused__scaled_dot_product_efficient_attention_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '2af0d8d97d2323882e9b91da64c8a9ac39ea8a97ea88d5425e2a8cf77d3f1889', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 32
    x2 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (1536*x1) + (49152*x2) + (49152*((x0 + (512*x1)) // 16384))), None)
    tmp1 = tl.load(in_ptr1 + (512 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_chilli/7p/c7pjydxl4q53kemk37h274f46bxf5zo324brwmt4swz4l7hctj7o.py
# Source Nodes: [encoder_layer], Original ATen: [aten._scaled_dot_product_efficient_attention]
# encoder_layer => _scaled_dot_product_efficient_attention
triton_poi_fused__scaled_dot_product_efficient_attention_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '2af0d8d97d2323882e9b91da64c8a9ac39ea8a97ea88d5425e2a8cf77d3f1889', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 32
    x2 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (1536*x1) + (49152*x2) + (49152*((x0 + (512*x1)) // 16384))), None)
    tmp1 = tl.load(in_ptr1 + (1024 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_chilli/5g/c5gt6nondfz72covw5vqd5dsicyyon7aqnydm6lhvxxyeald6osx.py
# Source Nodes: [encoder_layer], Original ATen: [aten.clone]
# encoder_layer => clone_1
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '2af0d8d97d2323882e9b91da64c8a9ac39ea8a97ea88d5425e2a8cf77d3f1889', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 32
    x2 = (xindex // 16384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x2) + (5120*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_chilli/zn/cznq4ku3ignj37qk6cibhjyqv6fvhgzfcuq2fsw6kixrvcp3ky6m.py
# Source Nodes: [encoder_layer], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm]
# encoder_layer => add, add_1, add_2, gt, mul, mul_1, mul_2, mul_3, rsqrt, sub, var_mean
triton_per_fused_add_native_dropout_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_4', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': '2af0d8d97d2323882e9b91da64c8a9ac39ea8a97ea88d5425e2a8cf77d3f1889', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, load_seed_offset, xnumel, rnumel):
    xnumel = 320
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (512*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp4 = 0.1
    tmp5 = tmp2 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tl.full([1], 512, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tmp13 - tmp23
    tmp31 = 512.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp2, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp40, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_chilli/i3/ci32ht5wcxqlqwtieqy4ac5oqtxgnkiyspu6kyvmcgtw4qtfbkck.py
# Source Nodes: [encoder_layer], Original ATen: [aten.native_dropout, aten.relu]
# encoder_layer => gt_1, mul_4, mul_5, relu
triton_poi_fused_native_dropout_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {3: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '2af0d8d97d2323882e9b91da64c8a9ac39ea8a97ea88d5425e2a8cf77d3f1889', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 2048
    tmp6 = tl.load(in_out_ptr0 + (x0), None)
    tmp7 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(0, tmp8)
    tmp10 = tmp5 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1536, 512), (512, 1))
    assert_size_stride(arg1_1, (1536, ), (1, ))
    assert_size_stride(arg2_1, (512, 512), (512, 1))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (2048, 512), (512, 1))
    assert_size_stride(arg5_1, (2048, ), (1, ))
    assert_size_stride(arg6_1, (512, 2048), (2048, 1))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (10, 32, 512), (16384, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((320, 1536), (1536, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(arg12_1, (320, 512), (512, 1), 0), reinterpret_tensor(arg0_1, (512, 1536), (1, 512), 0), out=buf0)
        del arg0_1
        buf1 = empty_strided_cuda((32, 8, 10, 64), (512, 64, 16384, 1), torch.float32)
        # Source Nodes: [encoder_layer], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_0.run(buf0, arg1_1, buf1, 163840, grid=grid(163840), stream=stream0)
        buf2 = empty_strided_cuda((32, 8, 10, 64), (512, 64, 16384, 1), torch.float32)
        # Source Nodes: [encoder_layer], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf0, arg1_1, buf2, 163840, grid=grid(163840), stream=stream0)
        buf3 = empty_strided_cuda((32, 8, 10, 64), (512, 64, 16384, 1), torch.float32)
        # Source Nodes: [encoder_layer], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf0, arg1_1, buf3, 163840, grid=grid(163840), stream=stream0)
        del arg1_1
        del buf0
        # Source Nodes: [encoder_layer], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf4 = aten._scaled_dot_product_efficient_attention.default(buf1, buf2, buf3, None, False, 0.1)
        buf5 = buf4[0]
        del buf4
        buf9 = reinterpret_tensor(buf3, (10, 32, 8, 64), (16384, 512, 64, 1), 0); del buf3  # reuse
        # Source Nodes: [encoder_layer], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf5, buf9, 163840, grid=grid(163840), stream=stream0)
        buf10 = reinterpret_tensor(buf5, (320, 512), (512, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (320, 512), (512, 1), 0), reinterpret_tensor(arg2_1, (512, 512), (1, 512), 0), out=buf10)
        del arg2_1
        buf11 = empty_strided_cuda((3, ), (1, ), torch.int64)
        # Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf11)
        buf12 = reinterpret_tensor(buf9, (10, 32, 512), (16384, 512, 1), 0); del buf9  # reuse
        buf16 = reinterpret_tensor(buf2, (10, 32, 512), (16384, 512, 1), 0); del buf2  # reuse
        # Source Nodes: [encoder_layer], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm]
        triton_per_fused_add_native_dropout_native_layer_norm_4.run(buf11, arg12_1, buf10, arg3_1, arg8_1, arg9_1, buf12, buf16, 0, 320, 512, grid=grid(320), stream=stream0)
        del arg12_1
        del arg3_1
        del arg8_1
        del arg9_1
        buf17 = empty_strided_cuda((320, 2048), (2048, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf16, (320, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 2048), (1, 512), 0), out=buf17)
        del arg4_1
        buf19 = reinterpret_tensor(buf17, (10, 32, 2048), (65536, 2048, 1), 0); del buf17  # reuse
        # Source Nodes: [encoder_layer], Original ATen: [aten.native_dropout, aten.relu]
        triton_poi_fused_native_dropout_relu_5.run(buf19, buf11, arg5_1, 1, 655360, grid=grid(655360), stream=stream0)
        del arg5_1
        buf20 = reinterpret_tensor(buf12, (320, 512), (512, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (320, 2048), (2048, 1), 0), reinterpret_tensor(arg6_1, (2048, 512), (1, 2048), 0), out=buf20)
        del arg6_1
        del buf19
        buf21 = reinterpret_tensor(buf10, (10, 32, 512), (16384, 512, 1), 0); del buf10  # reuse
        buf25 = reinterpret_tensor(buf1, (10, 32, 512), (16384, 512, 1), 0); del buf1  # reuse
        # Source Nodes: [encoder_layer], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm]
        triton_per_fused_add_native_dropout_native_layer_norm_4.run(buf11, buf16, buf20, arg7_1, arg10_1, arg11_1, buf21, buf25, 2, 320, 512, grid=grid(320), stream=stream0)
        del arg10_1
        del arg11_1
        del arg7_1
        del buf11
        del buf16
        del buf20
        del buf21
    return (buf25, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((10, 32, 512), (16384, 512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
