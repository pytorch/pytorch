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
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            buf0 = empty_strided_p2p((128, 4096), (4096, 1), torch.bfloat16, torch.device("cuda:1"), group_name="0", alloc_id=13588390060682276167)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.mm]
            extern_kernels.mm(arg1_1, arg0_1, out=buf0)
            del arg0_1
            del arg1_1
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [symm_mem.one_shot_all_reduce]
            buf1 = torch.ops.symm_mem.one_shot_all_reduce.default(buf0, 'sum', '0')
            buf2 = buf1
            assert_size_stride(buf2, (128, 4096), (4096, 1), 'torch.ops.symm_mem.one_shot_all_reduce.default')
            assert_alignment(buf2, 16, 'torch.ops.symm_mem.one_shot_all_reduce.default')
            del buf1
            buf3 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.mm]
            extern_kernels.mm(buf2, arg2_1, out=buf3)
            del arg2_1
            del buf2
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [symm_mem.one_shot_all_reduce]
            buf4 = torch.ops.symm_mem.one_shot_all_reduce.default(buf3, 'sum', '0')
            buf5 = buf4
            assert_size_stride(buf5, (128, 4096), (4096, 1), 'torch.ops.symm_mem.one_shot_all_reduce.default')
            assert_alignment(buf5, 16, 'torch.ops.symm_mem.one_shot_all_reduce.default')
            del buf4
            buf6 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.mm]
            extern_kernels.mm(buf5, arg3_1, out=buf6)
            del arg3_1
            del buf5
            # Topologically Sorted Source Nodes: [x_5], Original ATen: [symm_mem.one_shot_all_reduce]
            buf7 = torch.ops.symm_mem.one_shot_all_reduce.default(buf6, 'sum', '0')
            buf8 = buf7
            assert_size_stride(buf8, (128, 4096), (4096, 1), 'torch.ops.symm_mem.one_shot_all_reduce.default')
            assert_alignment(buf8, 16, 'torch.ops.symm_mem.one_shot_all_reduce.default')
            del buf7
            buf9 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.mm]
            extern_kernels.mm(buf8, arg4_1, out=buf9)
            del arg4_1
            del buf8
            # Topologically Sorted Source Nodes: [x_7], Original ATen: [symm_mem.one_shot_all_reduce]
            buf10 = torch.ops.symm_mem.one_shot_all_reduce.default(buf9, 'sum', '0')
            buf11 = buf10
            assert_size_stride(buf11, (128, 4096), (4096, 1), 'torch.ops.symm_mem.one_shot_all_reduce.default')
            assert_alignment(buf11, 16, 'torch.ops.symm_mem.one_shot_all_reduce.default')
            del buf10
            buf12 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.mm]
            extern_kernels.mm(buf11, arg5_1, out=buf12)
            del arg5_1
            del buf11
            # Topologically Sorted Source Nodes: [x_9], Original ATen: [symm_mem.one_shot_all_reduce]
            buf13 = torch.ops.symm_mem.one_shot_all_reduce.default(buf12, 'sum', '0')
            buf14 = buf13
            assert_size_stride(buf14, (128, 4096), (4096, 1), 'torch.ops.symm_mem.one_shot_all_reduce.default')
            assert_alignment(buf14, 16, 'torch.ops.symm_mem.one_shot_all_reduce.default')
            del buf13
            buf15 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.mm]
            extern_kernels.mm(buf14, arg6_1, out=buf15)
            del arg6_1
            del buf14
            # Topologically Sorted Source Nodes: [x_11], Original ATen: [symm_mem.one_shot_all_reduce]
            buf16 = torch.ops.symm_mem.one_shot_all_reduce.default(buf15, 'sum', '0')
            buf17 = buf16
            assert_size_stride(buf17, (128, 4096), (4096, 1), 'torch.ops.symm_mem.one_shot_all_reduce.default')
            assert_alignment(buf17, 16, 'torch.ops.symm_mem.one_shot_all_reduce.default')
            del buf16
            buf18 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.mm]
            extern_kernels.mm(buf17, arg7_1, out=buf18)
            del arg7_1
            del buf17
            # Topologically Sorted Source Nodes: [x_13], Original ATen: [symm_mem.one_shot_all_reduce]
            buf19 = torch.ops.symm_mem.one_shot_all_reduce.default(buf18, 'sum', '0')
            buf20 = buf19
            assert_size_stride(buf20, (128, 4096), (4096, 1), 'torch.ops.symm_mem.one_shot_all_reduce.default')
            assert_alignment(buf20, 16, 'torch.ops.symm_mem.one_shot_all_reduce.default')
            del buf19
            buf21 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.mm]
            extern_kernels.mm(buf20, arg8_1, out=buf21)
            del arg8_1
            del buf20
            # Topologically Sorted Source Nodes: [x_15], Original ATen: [symm_mem.one_shot_all_reduce]
            buf22 = torch.ops.symm_mem.one_shot_all_reduce.default(buf21, 'sum', '0')
            del buf21 # symm_mem buffer free
            buf23 = buf22
            assert_size_stride(buf23, (128, 4096), (4096, 1), 'torch.ops.symm_mem.one_shot_all_reduce.default')
            assert_alignment(buf23, 16, 'torch.ops.symm_mem.one_shot_all_reduce.default')
            del buf22
        return (buf23, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:1', dtype=torch.bfloat16)
    arg1_1 = rand_strided((128, 4096), (4096, 1), device='cuda:1', dtype=torch.bfloat16)
    arg2_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:1', dtype=torch.bfloat16)
    arg3_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:1', dtype=torch.bfloat16)
    arg4_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:1', dtype=torch.bfloat16)
    arg5_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:1', dtype=torch.bfloat16)
    arg6_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:1', dtype=torch.bfloat16)
    arg7_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:1', dtype=torch.bfloat16)
    arg8_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:1', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
