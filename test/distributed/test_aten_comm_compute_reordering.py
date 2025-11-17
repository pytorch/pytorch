# flake8: noqa: B950
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
from torch._dynamo.utils import counters, same
from torch._inductor.utils import run_and_get_code, run_and_get_triton_code
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    at_least_x_gpu,
    DynamoDistributedMultiProcTestCase,
    requires_accelerator_dist_backend,
)


aten = torch.ops.aten
import functools

from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import skipIfRocm
from torch.testing._internal.inductor_utils import HAS_GPU


def estimate_aten_runtime(fx_node, compute_multiplier=1.0):
    # for tests, assume a matmul can hide a single collective
    if "c10" in str(fx_node.target):
        return 1.0
    elif fx_node.target == aten.mm.default:
        return compute_multiplier
    else:
        return None


device_type = str(get_devtype())


def apply_reordering_and_get_graph(graph, out_li) -> None:
    gm = graph.owning_module
    from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing

    schedule_overlap_bucketing(gm)
    gm.graph.lint()
    out_li.append(str(gm.graph))


def run_and_get_aten_graph(fn, *inputs):
    li = []
    apply = functools.partial(apply_reordering_and_get_graph, out_li=li)
    with torch._inductor.config.patch(post_grad_custom_post_pass=apply):
        out = fn(*inputs)

    return out, li[0]


def get_patches():
    return {
        "test_configs.estimate_aten_runtime": estimate_aten_runtime,
        "reorder_for_locality": False,
        "reorder_for_compute_comm_overlap_passes": [],
        "compile_threads": 1,
        "force_disable_caches": True,
        # Messes up existing test strings
        "test_configs.aten_fx_overlap_insert_overlap_deps": False,
    }


@requires_accelerator_dist_backend()
# TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestComputeCommReorderingMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under

    Note: these tests are a fork of test/distributed/test_compute_comm_reordering.py

    """

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()

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

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_patches())
    def test_sink_waits(self):
        def func(a):
            ar = _functional_collectives.all_reduce(a, "sum", "0")
            b = torch.matmul(a, a)
            return torch.matmul(ar, b)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank

            out, aten_graph_str = run_and_get_aten_graph(torch.compile(func), inputs)

            # Verify that the wait_tensor is sinked below the 1st matmul but
            # above the 2nd matmul.
            (
                FileCheck()
                .check("all_reduce.default")
                .check("aten.mm.default")
                .check("wait_tensor.default")
                .check("aten.mm.default")
                .run(aten_graph_str)
            )
            correct = func(inputs)
            self.assertTrue(same(out, correct))
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)

    @torch._inductor.config.patch(get_patches())
    def test_raise_comms(self):
        def func(a):
            b = torch.matmul(a, a)
            c = torch.relu(b)
            d = torch.matmul(c, c)
            e = _functional_collectives.all_reduce((b + 1), "sum", "0")
            return torch.matmul(d, e)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            compiled = torch.compile(func)
            out, aten_graph_str = run_and_get_aten_graph(torch.compile(func), inputs)
            # Verify that the all_reduce_ has been raised above the 2nd matmul
            # but below the 1st matmul. Note that the all_reduce_ directly
            # writes to the output buffer of the 1st matmul, which is an input
            # to the first relu. Therefore, the all_reduce_ should be scheduled
            # after the first relu.
            (
                FileCheck()
                .check("aten.mm")
                .check("all_reduce.default")
                .check("aten.mm")
                .check("wait_tensor.default")
                .check("aten.mm")
                .run(aten_graph_str)
            )
            out = compiled(inputs)
            correct = func(inputs)
            self.assertTrue(same(out, correct))
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)

    @torch._inductor.config.patch(get_patches())
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
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(
                4, 4, dtype=torch.float, device=device_type
            )  # + self.rank
            kwargs = self.get_world_trs()
            func = functools.partial(func, **kwargs)
            compiled = torch.compile(func)
            out, aten_graph_str = run_and_get_aten_graph(compiled, inputs)
            # Things to verify:
            # - The all_reduce_ and its prologue should be raised above the 2nd
            # matmul but below the 1st matmul.
            # - The wait_tensor should be sinked below the 3rd matmul but above
            # the 4th matmul.

            self.assertExpectedInline(
                aten_graph_str,
                """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %mm : [num_users=2] = call_function[target=torch.ops.aten.mm.default](args = (%arg0_1, %arg0_1), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mm,), kwargs = {})
    %all_reduce : [num_users=1] = call_function[target=torch.ops._c10d_functional.all_reduce.default](args = (%mm, sum, 0), kwargs = {})
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%relu, %relu), kwargs = {})
    %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mm_1,), kwargs = {})
    %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%relu_1, %relu_1), kwargs = {})
    %wait_tensor : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor.default](args = (%all_reduce,), kwargs = {})
    %mm_3 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%wait_tensor, %mm_2), kwargs = {})
    return (mm_3,)""",
            )

            # Note: this triggered an all_reduce_ bug
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)

    @torch._inductor.config.patch(get_patches())
    def test_reorder_compute_for_overlap_mul(self):
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
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            func_c = functools.partial(func, **self.get_world_trs())
            compiled = torch.compile(func_c)
            out_c, aten_graph_str = run_and_get_aten_graph(compiled, inputs)
            # Note: because we have given collectives and mms equal estimation,
            # we overlap each collective with a single mm.
            # Same schedule as in test_reorder_compute_for_overlap_custom_runtime_estimation
            # although there is an exposed collective
            (
                FileCheck()
                .check("all_reduce.default")
                .check("aten.mm")
                .check("aten.mm")
                .check("wait_tensor.default")
                .check("aten.mul")
                .check("all_reduce.default")
                .check("wait_tensor.default")
                .check("aten.mm")
                .run(aten_graph_str)
            )
            correct = func(inputs, **self.get_world_trs())
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 1)
            self.assertTrue(same(out_c, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skipIfRocm
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @unittest.skipIf(True, "Logic not yet implemented")
    @torch._inductor.config.patch(get_patches())
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
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # Expectations:
            # 1. `add = a + a` and `div = add / a` are still fused, which means fusion
            #    still happens among nodes within a GroupedSchedulerNode.
            # 2. `mul = a * a` is not fused with `add` or `div`, because the latter two are within
            #    GroupedSchedulerNode and thus are prevented from being fused with any outside ops.
            FileCheck().check("triton_poi_fused_add_all_reduce_div_0.").check(
                "_c10d_functional.all_reduce_."
            ).check("triton_poi_fused_mul_1.").run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_patches())
    def test_inductor_default_comms_ordering(self):
        pg_info = self.get_world_trs()
        tag = pg_info["tag"]
        ranks = pg_info["ranks"]
        group_size = pg_info["group_size"]

        g1 = torch.ones(10, 10, device=device_type)
        g2 = torch.ones(11, 11, device=device_type)
        g3 = torch.ones(12, 12, device=device_type)

        @torch.compile
        def fn(g1, g2, g3):
            handle1 = torch.ops.c10d_functional.all_reduce(
                g1, "avg", tag, ranks, group_size
            )
            handle2 = torch.ops.c10d_functional.all_reduce(
                g2, "avg", tag, ranks, group_size
            )
            handle3 = torch.ops.c10d_functional.all_reduce(
                g3, "avg", tag, ranks, group_size
            )

            # wait on them in a different order
            grad3 = torch.ops._c10d_functional.wait_tensor.default(handle3)
            grad2 = torch.ops._c10d_functional.wait_tensor.default(handle2)
            grad1 = torch.ops._c10d_functional.wait_tensor.default(handle1)
            return grad3, grad2, grad1

        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size, self.backend(device_type), fake_pg=True
        ):
            # all_reduces remain in order!
            # note: this isnt actually invariant of pass currently..
            # but we should keep collectives stable without reordering opportunities

            _, code = run_and_get_aten_graph(fn, g1, g2, g3)

            FileCheck().check("all_reduce").check_same("arg0_1").check(
                "all_reduce"
            ).check_same("arg1_1").check("all_reduce").check_same("arg2_1").run(code)
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 3)
            # these have no overlap opportunities
            self.assertEqual(counters["inductor"]["overlap_scheduling_bad_exposed"], 0)


def get_bucket_patches(compute_multiplier=1.0):
    estimate_aten_runtime_part = functools.partial(
        estimate_aten_runtime, compute_multiplier=compute_multiplier
    )
    return {
        "test_configs.estimate_aten_runtime": estimate_aten_runtime_part,
        "test_configs.aten_fx_overlap_preserving_bucketing": True,
        "reorder_for_locality": False,
        "reorder_for_compute_comm_overlap_passes": [],
        "compile_threads": 1,
        "force_disable_caches": True,
        # messes up test strings
        "test_configs.aten_fx_overlap_insert_overlap_deps": False,
    }


class TestComputeCommReorderingBucketing(TestComputeCommReorderingMultiProc):
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_basic_all_gather_bucketing(self):
        """Test that independent all_gather operations get bucketed together."""

        def func(a, b, c, *, ranks):
            # Three independent all_gathers that should be bucketed
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks) + 3
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks) + 4
            ag3 = _functional_collectives.all_gather_tensor(c, 0, ranks) + 5
            return ag1 + ag2 + ag3

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs_a = (
                torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            )
            inputs_b = torch.ones(4, 4, dtype=torch.float, device=device_type) * 2
            inputs_c = torch.ones(4, 4, dtype=torch.float, device=device_type) * 3
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(
                compiled, inputs_a, inputs_b, inputs_c
            )

            # Should see a single bucketed all_gather
            FileCheck().check_count(
                "torch.ops._c10d_functional.all_gather_into_tensor", 1, exactly=True
            ).run(aten_graph_str)

            correct = func(inputs_a, inputs_b, inputs_c, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_reduce_scatter_bucketing(self):
        """Test bucketing of reduce_scatter operations."""

        def func(a, b, c):
            rs1 = _functional_collectives.reduce_scatter_tensor(a, "sum", 0, "0")
            rs2 = _functional_collectives.reduce_scatter_tensor(b, "sum", 0, "0")
            rs3 = _functional_collectives.reduce_scatter_tensor(c, "sum", 0, "0")
            return torch.cat([rs1, rs2, rs3])

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs_a = torch.ones(8, 4, dtype=torch.float, device=device_type)
            inputs_b = torch.ones(8, 4, dtype=torch.float, device=device_type) * 2
            inputs_c = torch.ones(8, 4, dtype=torch.float, device=device_type) * 3

            out, aten_graph_str = run_and_get_aten_graph(
                torch.compile(func), inputs_a, inputs_b, inputs_c
            )

            # Should bucket reduce_scatter ops
            FileCheck().check_count(
                "torch.ops._c10d_functional.reduce_scatter_tensor", 1, exactly=True
            ).run(aten_graph_str)

            # TODO: debug - on ci this fails.
            # correct = func(inputs_a, inputs_b, inputs_c)
            # self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_no_bucketing_with_dependent_hiding_nodes(self):
        """Test that collectives with dependent hiding nodes don't get bucketed."""

        def func(a, b, *, ranks):
            # ag1 could be hidden by mm1
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            mm1 = torch.matmul(a, a)

            # ag2 can be hidden by mm2, but mm2 depends on ag1's result
            # ag2 start
            mm2 = torch.matmul(ag1[:4], b)
            # ag2 end
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)

            return ag1.sum() * ag2.sum() * mm1 * mm2

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs_a = torch.ones(4, 4, dtype=torch.float, device=device_type)
            inputs_b = torch.ones(4, 4, dtype=torch.float, device=device_type)
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, inputs_a, inputs_b)

            # mm2 depends on ag1, so if mm2 is to hide ag2, we can't bucket ag1 and ag2
            # because that would create a dependency issue, even though we could bucket them
            FileCheck().check_count(
                "torch.ops._c10d_functional.all_gather_into_tensor", 2, exactly=True
            ).run(aten_graph_str)

            correct = func(inputs_a, inputs_b, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_no_bucketing_when_collective_depends_on_hiding_node(self):
        """Test that collectives don't get bucketed when one depends on another's hiding node."""

        def func(a, *, ranks):
            # ag1 hidden by mm1
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            mm1 = torch.matmul(a, a)

            # ag2 depends on mm1 (which hides ag1)
            b = mm1 * 2
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)

            return ag1.sum() * ag2.sum() * mm1

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type)
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, inputs)

            # ag2 depends on mm1 (ag1's hiding node), so they can't be bucketed
            FileCheck().check_count(
                "_c10d_functional.all_gather_into_tensor", 2, exactly=True
            ).run(aten_graph_str)

            correct = func(inputs, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches(2.0))
    def test_bucketing_wait_sink(self):
        """Test that 4 independent all-gathers split bucketed."""

        def func(a, b, c, d, *, ranks):
            # All 4 all-gathers are independent - COULD be bucketed together
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_tensor(c[:4], 0, ranks)
            ag4 = _functional_collectives.all_gather_tensor(d[:4], 0, ranks)

            # First compute - can hide ag1 and ag2
            e = a * 5
            mm1 = torch.matmul(e, e.T)

            # Second compute - can hide ag3 and ag4
            f = b * 6
            mm2 = torch.matmul(f, f.T)

            # Use all collective results
            result = (
                ag1.sum() * 1.1
                + ag2.sum() * 1.2
                + ag3.sum() * 1.3
                + ag4.sum() * 1.4
                + mm1.sum()
                + mm2.sum()
            )

            return result

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c, d)

            # The 4 all gathers can be bucketed, and their waits should be sunk below the mms
            FileCheck().check_count(
                "_c10d_functional.all_gather_into_tensor", 1, exactly=True
            ).check_count("ops.aten.mm", 2, exactly=True).check(
                "_c10d_functional.wait_tensor"
            ).run(aten_graph_str)

            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches(2.0))
    def test_bucketing_split_for_overlap_blocking(self):
        """Test that 4 independent all-gathers split into 2+2 buckets for better overlap with compute."""

        def func(a, b, c, d, *, ranks):
            # All 4 all-gathers are independent - COULD be bucketed together
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_tensor(c[:4], 0, ranks)
            ag4 = _functional_collectives.all_gather_tensor(d[:4], 0, ranks)

            # First compute - can hide ag1 and ag2
            e = a * 5  # Use a to avoid fusion
            mm1 = torch.matmul(e, e.T)

            # Force ag1/ag2 to complete before mm2 (but ag3/ag4 can still be deferred)
            # Use first 8x8 elements to match mm1's shape
            intermediate = ag1[:8, :8] + ag2[:8, :8]

            # Second compute - depends on ag1/ag2 through intermediate, can hide ag3/ag4
            mm2 = torch.matmul(mm1 + intermediate, c[:8])

            # Use all results
            result = (
                ag1.sum() * 1.1
                + ag2.sum() * 1.2
                + ag3.sum() * 1.3
                + ag4.sum() * 1.4
                + mm1.sum()
                + mm2.sum()
            )
            return result

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c, d)

            # The 4 all gathers can be bucketed, and the wait should be sunk below the mms
            FileCheck().check_count(
                "_c10d_functional.all_gather_into_tensor", 1, exactly=True
            ).check_count("ops.aten.mm", 2, exactly=True).check_count(
                "_c10d_functional.wait_tensor", 1, exactly=True
            ).run(aten_graph_str)

            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches(2.0))
    def test_bucketing_split_for_overlap(self):
        """Test that 4 independent all-gathers split into 2+2 buckets for better overlap with compute."""

        def func(a, b, c, d, *, ranks):
            # All 4 all-gathers are independent - COULD be bucketed together
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_tensor(c[:4], 0, ranks)
            ag4 = _functional_collectives.all_gather_tensor(d[:4], 0, ranks)

            # First compute - can hide ag1 and ag2
            e = a * 5  # Use a to avoid fusion
            mm1 = torch.matmul(e, e.T)

            # Force ag1/ag2 to complete before mm2 (but ag3/ag4 can still be deferred)
            intermediate = ag1[:2, :2] + ag2[:2, :2]  # Small slice to minimize compute

            # Second compute - depends on ag1/ag2 through intermediate, can hide ag3/ag4
            f = b * 6
            # Expand intermediate to match mm1's shape for broadcasting
            intermediate_expanded = torch.nn.functional.pad(intermediate, (0, 6, 0, 6))
            mm2 = torch.matmul(mm1 + intermediate_expanded, f.T)

            # Use all results
            result = (
                ag1.sum() * 1.1
                + ag2.sum() * 1.2
                + ag3.sum() * 1.3
                + ag4.sum() * 1.4
                + mm1.sum()
                + mm2.sum()
            )

            return result

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c, d)

            # Should have 2 bucketed all-gathers (one for ag1+ag2, one for ag3+ag4)
            FileCheck().check_count(
                "_c10d_functional.all_gather_into_tensor_out", 2, exactly=True
            ).run(aten_graph_str)

            # Verify the ordering - first bucket, then mm1, then second bucket, then mm2
            FileCheck().check("_c10d_functional.all_gather_into_tensor_out").check(
                "ops.aten.mm"
            ).check("_c10d_functional.all_gather_into_tensor_out").check(
                "ops.aten.mm"
            ).run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_bucket_exposed_with_hidden_single_overlap(self):
        """Test that exposed and hidden collectives bucket together when overlap is preserved."""

        def func(a, b, c, *, ranks):
            # ag1 will be hidden by mm1
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)

            # ag2 and ag3 are exposed (no compute to hide them)
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_tensor(c, 0, ranks)

            # can only hide one collective
            mm1 = torch.matmul(a[:2], a[:2].T)  # 2x2 matmul, hides only ag1

            # All three can bucket together because:
            # bucketing ag1, ag2, ag3 together does not prevent ag1 being hidden by mm1.

            return ag1.sum() + ag2.sum() + ag3.sum() + mm1.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c)

            # Should have 1 bucketed operation containing all 3 all-gathers
            FileCheck().check_count("wait_tensor.default", 1, exactly=True).run(
                aten_graph_str
            )

            # Verify bucketed collective overlaps with mm1
            FileCheck().check("functional.all_gather_into_tensor").check(
                "aten.mm"
            ).check("wait_tensor").run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, c, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches(2.0))
    def test_bucketing_split_for_overlap_blocking_deps_inductor(self):
        """Test that 4 independent all-gathers split into 2+2 buckets for better overlap with compute."""

        # check that ordering is preserved in inductor

        def func(a, b, c, d, *, ranks):
            # All 4 all-gathers are independent - COULD be bucketed together
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_tensor(c[:4], 0, ranks)
            ag4 = _functional_collectives.all_gather_tensor(d[:4], 0, ranks)

            # First compute - can hide ag1 and ag2
            e = a * 5  # Use a to avoid fusion
            mm1 = torch.matmul(e, e.T)

            # Force ag1/ag2 to complete before mm2 (but ag3/ag4 can still be deferred)
            # Use first 8x8 elements to match mm1's shape
            intermediate = ag1[:8, :8] + ag2[:8, :8]

            # Second compute - depends on ag1/ag2 through intermediate, can hide ag3/ag4
            mm2 = torch.matmul(mm1 + intermediate, c[:8])

            # Use all results
            result = (
                ag1.sum() * 1.1
                + ag2.sum() * 1.2
                + ag3.sum() * 1.3
                + ag4.sum() * 1.4
                + mm1.sum()
                + mm2.sum()
            )
            return result

        li = []
        apply = functools.partial(apply_reordering_and_get_graph, out_li=li)
        with (
            _dynamo_dist_per_rank_init(
                self.rank,
                self.world_size,
                self.backend(device_type),
                fake_pg=not at_least_x_gpu(2),
            ),
            torch._inductor.config.patch(
                "test_configs.aten_fx_overlap_insert_overlap_deps", True
            ),
            torch._inductor.config.patch(post_grad_custom_post_pass=apply),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            test_out, (code,) = run_and_get_code(compiled, a, b, c, d)

            # Check that right deps are added
            f = FileCheck()
            for _ in range(2):
                f.check("control_deps").check_same("all_gather").check_same(
                    "subgraph_mm"
                )
                f.check("control_deps").check_same("mm").check_same("subgraph_wait")
            f.run(li[0])

            f = FileCheck()
            for _ in range(2):
                f.check_count("all_gather_into_tensor_out.default(", 1, exactly=True)
                f.check_count("extern_kernels.mm(", 1, exactly=True)
                f.check_count("wait_tensor.default(", 1, exactly=True)
            f.run(code)

            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(test_out, correct))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
