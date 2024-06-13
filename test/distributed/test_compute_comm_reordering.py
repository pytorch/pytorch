# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case

# for some reason importing functional collectives after dynamo breaks collectives handling!
import torch.distributed._functional_collectives as _functional_collectives
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import ir
from torch._inductor.comm_analysis import (
    baseLat,
    hwLat,
    llMaxBws,
    NCCL_ALGO,
    NCCL_HW,
    NCCL_PROTO,
    NVIDIA_GPU_TYPE,
)
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    DynamoDistributedMultiProcTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.utils._triton import has_triton


def get_snode_runtime_for_reorder_compute_test(snode):
    # NOTE: custom cost model to show that the compute reordering algorithm is working
    # Collective kernels
    if isinstance(snode.node, ir._CollectiveKernel):
        return 100
    elif isinstance(snode.node, ir._WaitKernel):
        return 0
    # High-arithmetic-intensity compute kernels
    elif isinstance(snode.node, ir.ExternKernel):
        return 5
    # All other kernels
    return 1


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

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "sink_waits",
        ],
    )
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
            FileCheck().check("dist.all_reduce(").check("triton_poi_fused_relu").check(
                "_wait_tensor("
            ).run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "raise_comms",
        ],
    )
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
            FileCheck().check("dist.all_reduce(").check("_wait_tensor(").check(
                "triton_poi_fused_relu"
            ).check("extern_kernels.addmm(").run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "sink_waits",
            "raise_comms",
        ],
    )
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
            FileCheck().check("dist.all_reduce(").check("triton_poi_fused_relu").check(
                "_wait_tensor("
            ).check("extern_kernels.addmm(").run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "reorder_compute_for_overlap",
        ],
    )
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
            FileCheck().check("dist.all_reduce(").check("triton_poi_fused_relu").check(
                "extern_kernels.mm("
            ).check("extern_kernels.mm(").check("_wait_tensor(").check(
                "triton_poi_fused_mul"
            ).check(
                "dist.all_reduce("
            ).check(
                "_wait_tensor("
            ).check(
                "triton_poi_fused_add"
            ).check(
                "extern_kernels.mm("
            ).run(
                code
            )
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "reorder_compute_for_overlap",
        ],
    )
    @patch.object(
        torch._inductor.config,
        "estimate_op_runtime",
        get_snode_runtime_for_reorder_compute_test,
    )
    def test_reorder_compute_for_overlap_custom_runtime_estimation(self):
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
            FileCheck().check("dist.all_reduce(").check("triton_poi_fused_relu").check(
                "extern_kernels.mm("
            ).check("extern_kernels.mm(").check("_wait_tensor(").check(
                "triton_poi_fused_mul"
            ).check(
                "dist.all_reduce("
            ).check(
                "_wait_tensor("
            ).check(
                "triton_poi_fused_add"
            ).check(
                "extern_kernels.mm("
            ).run(
                code
            )
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    def test_nccl_heuristics(self):
        assert len(baseLat) == len(NCCL_ALGO)
        assert all(len(x) == len(NCCL_PROTO) for x in baseLat)

        assert len(hwLat) == len(NCCL_HW)
        assert all(len(x) == len(NCCL_ALGO) for x in hwLat)
        assert all(len(y) == len(NCCL_PROTO) for x in hwLat for y in x)

        assert len(llMaxBws) == len(NVIDIA_GPU_TYPE)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
