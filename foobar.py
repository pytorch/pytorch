
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


# kernel path: /tmp/torchinductor_chilli/mz/cmzatgdbalkyxo3nmn76koh3oi75aybspijsd7b3nhl4abrbix5u.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._softmax, aten.cos]
# x_1 => cos
# x_2 => amax, div, exp, sub, sum_1
# triton_per_fused__softmax_cos_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cos_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '2af0d8d97d2323882e9b91da64c8a9ac39ea8a97ea88d5425e2a8cf77d3f1889', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_cos_0(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    rnumel = 200
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (200*x0)), rmask & xmask, other=0.0)
    tmp1 = tl_math.cos(tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, float("-inf"))
    tmp5 = triton_helpers.max2(tmp4, 1)[:, None]
    tmp6 = tmp1 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tmp7 / tmp11
    tl.store(out_ptr2 + (r1 + (200*x0)), tmp12, rmask & xmask)
# ''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (200, 200), (200, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((200, 200), (200, 1), torch.float32)
        # Source Nodes: [x], Original ATen: [aten.mm]
        extern_kernels.mm(arg0_1, arg0_1, out=buf0)
        del arg0_1
        buf3 = empty_strided_cuda((200, 200), (200, 1), torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten._softmax, aten.cos]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_cos_0.run(buf0, buf3, 200, 200, grid=grid(200), stream=stream0)
        buf4 = buf0; del buf0  # reuse
        # Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._softmax, aten.cos, aten.mm]
        extern_kernels.mm(buf3, buf3, out=buf4)
        buf7 = buf3; del buf3  # reuse
        # Source Nodes: [x_4, x_5], Original ATen: [aten._softmax, aten.cos]
        triton_per_fused__softmax_cos_0.run(buf4, buf7, 200, 200, grid=grid(200), stream=stream0)
        buf8 = buf4; del buf4  # reuse
        # Source Nodes: [x_4, x_5, x_6], Original ATen: [aten._softmax, aten.cos, aten.mm]
        extern_kernels.mm(buf7, buf7, out=buf8)
        buf11 = buf7; del buf7  # reuse
        # Source Nodes: [x_7, x_8], Original ATen: [aten._softmax, aten.cos]
        triton_per_fused__softmax_cos_0.run(buf8, buf11, 200, 200, grid=grid(200), stream=stream0)
        buf12 = buf8; del buf8  # reuse
        # Source Nodes: [x_7, x_8, x_9], Original ATen: [aten._softmax, aten.cos, aten.mm]
        extern_kernels.mm(buf11, buf11, out=buf12)
        buf15 = buf11; del buf11  # reuse
        # Source Nodes: [x_10, x_11], Original ATen: [aten._softmax, aten.cos]
        triton_per_fused__softmax_cos_0.run(buf12, buf15, 200, 200, grid=grid(200), stream=stream0)
        buf16 = buf12; del buf12  # reuse
        # Source Nodes: [x_10, x_11, x_12], Original ATen: [aten._softmax, aten.cos, aten.mm]
        extern_kernels.mm(buf15, buf15, out=buf16)
        buf19 = buf15; del buf15  # reuse
        # Source Nodes: [x_13, x_14], Original ATen: [aten._softmax, aten.cos]
        triton_per_fused__softmax_cos_0.run(buf16, buf19, 200, 200, grid=grid(200), stream=stream0)
        buf20 = buf16; del buf16  # reuse
        # Source Nodes: [x_13, x_14, x_15], Original ATen: [aten._softmax, aten.cos, aten.mm]
        extern_kernels.mm(buf19, buf19, out=buf20)
        buf23 = buf19; del buf19  # reuse
        # Source Nodes: [x_16, x_17], Original ATen: [aten._softmax, aten.cos]
        triton_per_fused__softmax_cos_0.run(buf20, buf23, 200, 200, grid=grid(200), stream=stream0)
        buf24 = buf20; del buf20  # reuse
        # Source Nodes: [x_16, x_17, x_18], Original ATen: [aten._softmax, aten.cos, aten.mm]
        extern_kernels.mm(buf23, buf23, out=buf24)
        buf27 = buf23; del buf23  # reuse
        # Source Nodes: [x_19, x_20], Original ATen: [aten._softmax, aten.cos]
        triton_per_fused__softmax_cos_0.run(buf24, buf27, 200, 200, grid=grid(200), stream=stream0)
        buf28 = buf24; del buf24  # reuse
        # Source Nodes: [x_19, x_20, x_21], Original ATen: [aten._softmax, aten.cos, aten.mm]
        extern_kernels.mm(buf27, buf27, out=buf28)
        buf31 = buf27; del buf27  # reuse
        # Source Nodes: [x_22, x_23], Original ATen: [aten._softmax, aten.cos]
        triton_per_fused__softmax_cos_0.run(buf28, buf31, 200, 200, grid=grid(200), stream=stream0)
        buf32 = buf28; del buf28  # reuse
        # Source Nodes: [x_22, x_23, x_24], Original ATen: [aten._softmax, aten.cos, aten.mm]
        extern_kernels.mm(buf31, buf31, out=buf32)
        buf35 = buf31; del buf31  # reuse
        # Source Nodes: [x_25, x_26], Original ATen: [aten._softmax, aten.cos]
        triton_per_fused__softmax_cos_0.run(buf32, buf35, 200, 200, grid=grid(200), stream=stream0)
        buf36 = buf32; del buf32  # reuse
        # Source Nodes: [x_25, x_26, x_27], Original ATen: [aten._softmax, aten.cos, aten.mm]
        extern_kernels.mm(buf35, buf35, out=buf36)
        buf39 = buf35; del buf35  # reuse
        # Source Nodes: [x_28, x_29], Original ATen: [aten._softmax, aten.cos]
        triton_per_fused__softmax_cos_0.run(buf36, buf39, 200, 200, grid=grid(200), stream=stream0)
        del buf36
    return (buf39, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((200, 200), (200, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


# if __name__ == "__main__":
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)
