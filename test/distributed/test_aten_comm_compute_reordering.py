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
from torch._inductor.utils import run_and_get_triton_code
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


def estimate_aten_runtime(fx_node):
    # for tests, assume a matmul can hide a single collective
    if "c10" in str(fx_node.target):
        return 1.0
    elif fx_node.target == aten.mm.default:
        return 1.0
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
