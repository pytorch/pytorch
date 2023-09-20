# Owner(s): ["module: inductor"]
import functools
import unittest
from unittest.mock import patch
import torch
from torch._C import FileCheck
# for some reason importing functional collectives after dynamo breaks collectives handling!
import torch.distributed._functional_collectives as _functional_collectives
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.utils import same
from torch._dynamo.testing import CompileCounter
from torch.distributed.distributed_c10d import GroupMember
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import (
    DynamoDistributedSingleProcTestCase,
    DynamoDistributedMultiProcTestCase,
    _dynamo_dist_per_rank_init,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch._inductor.utils import has_triton, run_and_get_triton_code
import torch._dynamo.logging
from torch._inductor import ir



@requires_nccl()
class TestComputeCommReorderingMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under
    """
    def get_world_trs(self):
        return {
            "tag": "",
            "ranks": list(range(self.world_size)),
            "group_size": self.world_size,
        }

    @property
    def world_size(self) -> int:
        # hack: no matter whether we have 2 or 3 or 4 gpus, just run on 2
        # works around issue with skipif<2 and workers with unpredictable #s gpu
        return 2

    """
    TODO: unit tests to add:
    1. sink_waits()
        TODO: design good # and variaty of compute ops
        TODO: assert correct ordering
    2. raise_comms()
        TODO: design good # and variaty of compute ops
        TODO: assert correct ordering
    3. reorder_compute_for_overlap()
        1) only 1 comm
        2) 2+ comms
        TODO: design good # and variaty of compute ops
        TODO: assert correct ordering
    4. reorder_compute_and_comm_for_overlap() integration test
        TODO: assert correct ordering
    """

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap_passes", [
        "sink_waits",
    ])
    def test_sink_waits(self):
        def func(a, *, tag, ranks, group_size):
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            c = torch.relu(a)
            d = torch.matmul(c, c)
            e = d + ar
            return (e,)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: notice that `_wait_tensor` is delayed until right before first use
            FileCheck() \
                .check("buf0 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("buf0.copy_(arg0_1) #no reuse") \
                .check("buf1_pg = c10d._find_or_create_pg_by_ranks_and_tag('', [0, 1], 2)") \
                .check("buf1 = buf0") \
                .check("buf1_work = dist.all_reduce(buf1, async_op=True, group=buf1_pg, op=fun_col_impl._str_to_reduce_op('sum'))") \
                .check("fun_col_impl._register_tensor_work(buf1, buf1_work)") \
                .check("buf3 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("triton_poi_fused_relu_0.run(arg0_1, buf3, 16, grid=grid(16),") \
                .check("buf0 = _wait_tensor(buf0)") \
                .check("buf2 = buf0") \
                .run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap_passes", [
        "raise_comms",
    ])
    def test_raise_comms(self):
        def func(a, *, tag, ranks, group_size):
            c = torch.relu(a)
            d = torch.matmul(c, c)
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            e = d + ar
            return (e,)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: notice that `dist.all_reduce` is raised above relu and matmul
            FileCheck() \
                .check("buf0 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("buf0.copy_(arg0_1) #no reuse") \
                .check("buf1_pg = c10d._find_or_create_pg_by_ranks_and_tag('', [0, 1], 2)") \
                .check("buf1 = buf0") \
                .check("buf1_work = dist.all_reduce(buf1, async_op=True, group=buf1_pg, op=fun_col_impl._str_to_reduce_op('sum'))") \
                .check("fun_col_impl._register_tensor_work(buf1, buf1_work)") \
                .check("buf0 = _wait_tensor(buf0)") \
                .check("buf2 = buf0") \
                .check("buf3 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("triton_poi_fused_relu_0.run(arg0_1, buf3, 16, grid=grid(16),") \
                .check("buf4 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("extern_kernels.addmm(buf2, buf3, buf3, alpha=1, beta=1, out=buf4)") \
                .run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap_passes", [
        "sink_waits",
        "raise_comms",
    ])
    def test_sink_waits_raise_comms(self):
        def func(a, *, tag, ranks, group_size):
            c = torch.relu(a)
            d = torch.matmul(c, c)
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            e = d + ar
            return (e,)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: notice that `dist.all_reduce` is raised above relu and matmul,
            # and `_wait_tensor` is delayed until right before first use
            FileCheck() \
                .check("buf0 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("buf0.copy_(arg0_1) #no reuse") \
                .check("buf1_pg = c10d._find_or_create_pg_by_ranks_and_tag('', [0, 1], 2)") \
                .check("buf1 = buf0") \
                .check("buf1_work = dist.all_reduce(buf1, async_op=True, group=buf1_pg, op=fun_col_impl._str_to_reduce_op('sum'))") \
                .check("fun_col_impl._register_tensor_work(buf1, buf1_work)") \
                .check("buf3 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("triton_poi_fused_relu_0.run(arg0_1, buf3, 16, grid=grid(16),") \
                .check("buf0 = _wait_tensor(buf0)") \
                .check("buf2 = buf0") \
                .check("buf4 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("extern_kernels.addmm(buf2, buf3, buf3, alpha=1, beta=1, out=buf4)") \
                .run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    def get_snode_runtime_for_reorder_compute_test(snode):
        # NOTE: custom cost model to show that the compute reordering algorithm is working
        # Collective kernels
        if isinstance(snode.node, ir.CollectiveKernel):
            if isinstance(snode.node, ir.AllReduce):
                return 100
            else:
                return 100
        elif isinstance(snode.node, ir.Wait):
            return 0
        # High-arithmetic-intensity compute kernels
        elif isinstance(snode.node, ir.ExternKernel):
            return 5
        # All other kernels
        return 1

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap_passes", [
        "reorder_compute_for_overlap",
    ])
    @patch.object(torch._inductor.config, "estimate_op_runtime", get_snode_runtime_for_reorder_compute_test)
    def test_reorder_compute_for_overlap(self):
        def func(a, *, tag, ranks, group_size):
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            g = torch.matmul(a, a)
            c = torch.relu(a)
            d = torch.matmul(c, c)
            f = d * c * ar
            fr = _functional_collectives.all_reduce(f, "sum", ranks, tag)
            e = torch.matmul(d + ar + fr, g)
            return (e,)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: after scheduling the first all_reduce:
            # 1. we first schedule the ops (c and d) that ARE required for second all_reduce but DO NOT depend on first all_reduce.
            # 2. then, we schedule the ops (g) that ARE NOT required for second all_reduce and DO NOT depend on first all_reduce.
            # 3. then, we schedule the ops (f) that ARE required for second all_reduce and DO depend on first all_reduce.
            # and then, we schedule the second all_reduce. And then schedule all ops that depend on second all_reduce.
            FileCheck() \
                .check("buf2 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("buf2.copy_(arg0_1) #no reuse") \
                .check("buf3_pg = c10d._find_or_create_pg_by_ranks_and_tag('', [0, 1], 2)") \
                .check("buf3 = buf2") \
                .check("buf3_work = dist.all_reduce(buf3, async_op=True, group=buf3_pg, op=fun_col_impl._str_to_reduce_op('sum'))") \
                .check("fun_col_impl._register_tensor_work(buf3, buf3_work)") \
                .check("buf0 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("# Source Nodes: [c], Original ATen: [aten.relu]") \
                .check("triton_poi_fused_relu_0.run(arg0_1, buf0, 16, grid=grid(16),") \
                .check("buf1 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("# Source Nodes: [d], Original ATen: [aten.mm]") \
                .check("extern_kernels.mm(buf0, buf0, out=buf1)") \
                .check("buf9 = empty_strided((4, 4), (4, 1), device='cuda', dtype=torch.float32)") \
                .check("# Source Nodes: [g], Original ATen: [aten.mm]") \
                .check("extern_kernels.mm(arg0_1, arg0_1, out=buf9)") \
                .check("buf2 = _wait_tensor(buf2)") \
                .check("buf4 = buf2") \
                .check("buf5 = buf0; del buf0  # reuse") \
                .check("# Source Nodes: [f, mul], Original ATen: [aten.mul]") \
                .check("triton_poi_fused_mul_1.run(buf5, buf1, buf4, 16, grid=grid(16),") \
                .check("buf6 = buf5; del buf5  # reuse") \
                .check("buf7_pg = c10d._find_or_create_pg_by_ranks_and_tag('', [0, 1], 2)") \
                .check("buf7 = buf6") \
                .check("buf7_work = dist.all_reduce(buf7, async_op=True, group=buf7_pg, op=fun_col_impl._str_to_reduce_op('sum'))") \
                .check("fun_col_impl._register_tensor_work(buf7, buf7_work)") \
                .check("buf6 = _wait_tensor(buf6)") \
                .check("buf8 = buf6") \
                .check("buf10 = buf1; del buf1  # reuse") \
                .check("# Source Nodes: [add, add_1], Original ATen: [aten.add]") \
                .check("triton_poi_fused_add_2.run(buf10, buf4, buf8, 16, grid=grid(16),") \
                .check("buf11 = buf6; del buf6  # reuse") \
                .check("# Source Nodes: [add, add_1, e], Original ATen: [aten.add, aten.mm]") \
                .check("extern_kernels.mm(buf10, buf9, out=buf11)") \
                .run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))
