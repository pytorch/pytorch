# AOT ID: ['0_inference']
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

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
        args.clear()
        assert_size_stride(arg0_1, (4096, 4096), (4096, 1))
        assert_size_stride(arg1_1, (128, 4096), (4096, 1))
        assert_size_stride(arg2_1, (4096, 4096), (4096, 1))
        assert_size_stride(arg3_1, (4096, 4096), (4096, 1))
        assert_size_stride(arg4_1, (4096, 4096), (4096, 1))
        assert_size_stride(arg5_1, (4096, 4096), (4096, 1))
        assert_size_stride(arg6_1, (4096, 4096), (4096, 1))
        assert_size_stride(arg7_1, (4096, 4096), (4096, 1))
        assert_size_stride(arg8_1, (4096, 4096), (4096, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_p2p((128, 4096), (4096, 1), torch.bfloat16, torch.device("cuda:0"), group_name="0", alloc_id=8872082608279973266)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.mm]
            extern_kernels.mm(arg1_1, arg0_1, out=buf0)
            del arg0_1
            del arg1_1
            buf1 = empty_strided_cuda((128, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [symm_mem.one_shot_all_reduce]
            torch.ops.symm_mem.one_shot_all_reduce_out.default(buf0, 'sum', '0', out=buf1)
            buf2 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.mm]
            extern_kernels.mm(buf1, arg2_1, out=buf2)
            del arg2_1
            buf3 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [symm_mem.one_shot_all_reduce]
            torch.ops.symm_mem.one_shot_all_reduce_out.default(buf2, 'sum', '0', out=buf3)
            buf4 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.mm]
            extern_kernels.mm(buf3, arg3_1, out=buf4)
            del arg3_1
            buf5 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [x_5], Original ATen: [symm_mem.one_shot_all_reduce]
            torch.ops.symm_mem.one_shot_all_reduce_out.default(buf4, 'sum', '0', out=buf5)
            buf6 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.mm]
            extern_kernels.mm(buf5, arg4_1, out=buf6)
            del arg4_1
            buf7 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [x_7], Original ATen: [symm_mem.one_shot_all_reduce]
            torch.ops.symm_mem.one_shot_all_reduce_out.default(buf6, 'sum', '0', out=buf7)
            buf8 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.mm]
            extern_kernels.mm(buf7, arg5_1, out=buf8)
            del arg5_1
            buf9 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [x_9], Original ATen: [symm_mem.one_shot_all_reduce]
            torch.ops.symm_mem.one_shot_all_reduce_out.default(buf8, 'sum', '0', out=buf9)
            buf10 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.mm]
            extern_kernels.mm(buf9, arg6_1, out=buf10)
            del arg6_1
            buf11 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [x_11], Original ATen: [symm_mem.one_shot_all_reduce]
            torch.ops.symm_mem.one_shot_all_reduce_out.default(buf10, 'sum', '0', out=buf11)
            buf12 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.mm]
            extern_kernels.mm(buf11, arg7_1, out=buf12)
            del arg7_1
            buf13 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [x_13], Original ATen: [symm_mem.one_shot_all_reduce]
            torch.ops.symm_mem.one_shot_all_reduce_out.default(buf12, 'sum', '0', out=buf13)
            buf14 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.mm]
            extern_kernels.mm(buf13, arg8_1, out=buf14)
            del arg8_1
            buf15 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [x_15], Original ATen: [symm_mem.one_shot_all_reduce]
            torch.ops.symm_mem.one_shot_all_reduce_out.default(buf14, 'sum', '0', out=buf15)
            del buf14 # symm_mem buffer free
        return (buf15, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
