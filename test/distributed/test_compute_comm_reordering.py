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
from torch._inductor import ir, scheduler
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
    at_least_x_gpu,
    DynamoDistributedMultiProcTestCase,
    requires_nccl,
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


def create_grouped_node_for_allreduce_and_its_deps(snodes):
    name_to_snode = {snode.node.name: snode for snode in snodes}
    all_reduce_snodes = [
        snode
        for snode in snodes
        if isinstance(snode.node, ir._CollectiveKernel)
        and snode.node.op_overload == torch.ops._c10d_functional.all_reduce_.default
    ]
    assert len(all_reduce_snodes) == 1
    all_reduce_snode = all_reduce_snodes[0]
    all_reduce_dep_snodes = [
        name_to_snode[node.name] for node in all_reduce_snode.node.inputs
    ]
    assert len(all_reduce_dep_snodes) == 1
    all_reduce_dep_snode = all_reduce_dep_snodes[0]

    grouped_snode = scheduler.GroupedSchedulerNode.create(
        [all_reduce_dep_snode, all_reduce_snode]
    )
    new_snode_order = []
    new_snode_order.append(grouped_snode)
    for snode in snodes:
        if snode in grouped_snode.snodes:
            continue
        new_snode_order.append(snode)
    return new_snode_order


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
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_locality", False)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "sink_waits",
        ],
    )
    def test_sink_waits(self):
        def func(a):
            ar = _functional_collectives.all_reduce(a, "sum", "0")
            b = torch.matmul(a, a)
            return torch.matmul(ar, b)

        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size, fake_pg=not at_least_x_gpu(2)
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs)
            # Verify that the wait_tensor is sinked below the 1st matmul but
            # above the 2nd matmul.
            (
                FileCheck()
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("extern_kernels.mm")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs)
            correct = func(inputs)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_locality", False)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "raise_comms",
        ],
    )
    def test_raise_comms(self):
        def func(a):
            b = torch.matmul(a, a)
            c = torch.relu(b)
            d = torch.matmul(c, c)
            e = _functional_collectives.all_reduce(b, "sum", "0")
            return torch.matmul(d, e)

        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size, fake_pg=not at_least_x_gpu(2)
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs)
            print(code)
            # Verify that the all_reduce_ has been raised above the 2nd matmul
            # but below the 1st matmul. Note that the all_reduce_ directly
            # writes to the output buffer of the 1st matmul, which is an input
            # to the first relu. Therefore, the all_reduce_ should be scheduled
            # after the first relu.
            (
                FileCheck()
                .check("extern_kernels.mm")
                .check("triton_poi_fused_relu")
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("extern_kernels.mm")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs)
            correct = func(inputs)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
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
            b = torch.matmul(a, a)
            c = torch.relu(b)
            d = torch.matmul(c, c)
            e = _functional_collectives.all_reduce(b, "sum", "0")
            f = torch.relu(d)
            g = torch.matmul(f, f)
            return torch.mm(e, g)

        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size, fake_pg=not at_least_x_gpu(2)
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # Things to verify:
            # - The clone prologue of the all_reduce_ should not be fused with
            # any relus.
            # - The all_reduce_ and its prologue should be raised above the 2nd
            # matmul but below the 1st matmul.
            # - The wait_tensor should be sinked below the 3rd matmul but above
            # the 4th matmul.
            (
                FileCheck()
                .check("extern_kernels.mm")
                .check("triton_poi_fused_all_reduce_0")
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("triton_poi_fused_relu")
                .check("extern_kernels.mm")
                .check("triton_poi_fused_relu")
                .check("extern_kernels.mm")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
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

        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size, fake_pg=not at_least_x_gpu(2)
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: after scheduling the first all_reduce:
            # 1. we first schedule the ops (c and d) that ARE required for second all_reduce but DO NOT depend on first all_reduce.
            # 2. then, we schedule the ops (g) that ARE NOT required for second all_reduce and DO NOT depend on first all_reduce.
            # 3. then, we schedule the ops (f) that ARE required for second all_reduce and DO depend on first all_reduce.
            # and then, we schedule the second all_reduce. And then schedule all ops that depend on second all_reduce.
            (
                FileCheck()
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("triton_poi_fused_relu")
                .check("extern_kernels.mm")
                .check("extern_kernels.mm")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("triton_poi_fused_mul")
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("triton_poi_fused_add")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
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

        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size, fake_pg=not at_least_x_gpu(2)
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: after scheduling the first all_reduce:
            # 1. we first schedule the ops (c and d) that ARE required for second all_reduce but DO NOT depend on first all_reduce.
            # 2. then, we schedule the ops (g) that ARE NOT required for second all_reduce and DO NOT depend on first all_reduce.
            # 3. then, we schedule the ops (f) that ARE required for second all_reduce and DO depend on first all_reduce.
            # and then, we schedule the second all_reduce. And then schedule all ops that depend on second all_reduce.
            (
                FileCheck()
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("triton_poi_fused_relu")
                .check("extern_kernels.mm")
                .check("extern_kernels.mm")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("triton_poi_fused_mul")
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("triton_poi_fused_add")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(
        torch._inductor.config,
        "_pre_fusion_custom_pass",
        create_grouped_node_for_allreduce_and_its_deps,
    )
    def test_grouped_scheduler_node(self):
        def func(a, *, tag, ranks, group_size):
            add = a + a
            div = add / a
            ar = _functional_collectives.all_reduce(div, "sum", ranks, tag)
            # Normally, we would fuse `add = a + a`, `div = add / a` and `mul = a * a` together into a single fused op,
            # but here in this unit test, we intentionally put `add`, `div` and `ar` computation
            # into a GroupedSchedulerNode, which prevents them from being fused with any other ops.
            mul = a * a
            mm = torch.matmul(mul, ar)
            return (mm,)

        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size, fake_pg=not at_least_x_gpu(2)
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device="cuda") + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # Expectations:
            # 1. `add = a + a` and `div = add / a` are still fused, which means fusion
            #    still happens among nodes within a GroupedSchedulerNode.
            # 2. `mul = a * a` is not fused with `add` or `div`, because the latter two are within
            #    GroupedSchedulerNode and thus are prevented from being fused with any outside ops.
            FileCheck().check("triton_poi_fused_add_div_0.").check(
                "_c10d_functional.all_reduce_."
            ).check("triton_poi_fused_mul_1.").run(code)
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
