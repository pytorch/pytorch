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
import torch.fx as fx
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


def estimate_aten_runtime(fx_node, override_size=None, compute_multiplier=1.0):
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
    from torch._inductor.config import aten_distributed_optimizations as dist_opts
    from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing

    # Read config values, only pass non-None values to use function defaults
    kwargs: dict[str, object] = {}
    config_keys = (
        "collective_bucketing",
        "max_compute_pre_fetch",
        "custom_runtime_estimation",
        "insert_overlap_deps",
        "collective_estimator",
        "bucket_exposed_first",
        "bucket_only_internode_comms",
    )
    for key in config_keys:
        if (val := getattr(dist_opts, key)) is not None:
            kwargs[key] = val

    schedule_overlap_bucketing(gm, **kwargs)
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
        "aten_distributed_optimizations.custom_runtime_estimation": estimate_aten_runtime,
        "reorder_for_locality": False,
        "triton.native_matmul": False,
        "reorder_for_compute_comm_overlap_passes": [],
        "compile_threads": 1,
        "force_disable_caches": True,
        # Messes up existing test strings
        "aten_distributed_optimizations.insert_overlap_deps": False,
        # interferes with testing, / custom estimation
        "test_configs.assume_bucketing_reduces_latency": False,
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

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_patches())
    def test_schedulable_wait(self):
        """Test that if a wait node is scheduable or not."""
        from torch._inductor.fx_passes.bucketing import _schedulable_wait_node

        def test_graph():
            graph = fx.Graph()

            inp = graph.placeholder("inp")
            group_size = graph.placeholder("group_size")
            group_name = graph.placeholder("group_name")

            ag_0_out = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(inp, group_size, group_name),
            )
            ag_0_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(ag_0_out,),
            )
            ag_1_out = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(ag_0_wait, group_size, group_name),
            )
            ag_1_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(ag_1_out,),
            )
            ag_2_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(ag_1_wait,),
            )

            graph.output(ag_2_wait)
            return graph

        graph = test_graph()
        schedulable = {"wait_tensor_default", "wait_tensor_default_1"}
        for node in list(graph.nodes):
            expected = node.name in schedulable
            if _schedulable_wait_node(node) is not expected:
                raise AssertionError(
                    f"Expected _schedulable_wait_node({node.name}) is {expected}"
                )

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
            # note: this isn't actually invariant of pass currently..
            # but we should keep collectives stable without reordering opportunities

            _, code = run_and_get_aten_graph(fn, g1, g2, g3)

            FileCheck().check("all_reduce").check_same("arg0_1").check(
                "all_reduce"
            ).check_same("arg1_1").check("all_reduce").check_same("arg2_1").run(code)
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 3)
            # these have no overlap opportunities
            self.assertEqual(counters["inductor"]["overlap_scheduling_bad_exposed"], 0)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_overlap_scheduling_via_config(self):
        """Test overlap scheduling enabled via config in post_grad pass."""

        def func(a):
            ar = _functional_collectives.all_reduce(a, "sum", "0")
            b = torch.matmul(a, a)
            return torch.matmul(ar, b)

        patches = {
            **get_patches(),
            "aten_distributed_optimizations.enable_overlap_scheduling": True,
        }

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank

            with torch._inductor.config.patch(patches):
                compiled_func = torch.compile(func)
                out, code = run_and_get_code(compiled_func, inputs)

                # Verify that wait_tensor is sinked below matmul
                FileCheck().check("all_reduce").check("mm").check("wait_tensor").check(
                    "mm"
                ).run(code[0])

                correct = func(inputs)
                self.assertTrue(same(out, correct))
                self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)

    @torch._inductor.config.patch(get_patches())
    def test_custom_estimator_for_non_compute_nodes(self):
        """Test that non-compute nodes with custom runtime estimates can trigger collective prefetching."""

        def custom_estimator_with_relu(fx_node, override_size=None):
            """Custom estimator that provides runtime for relu."""
            # Collective ops
            if "c10" in str(fx_node.target):
                return 1.0
            # Non-compute ops that we want to overlap
            elif fx_node.target == aten.relu.default:
                return 1.0  # relu has same time as collective
            else:
                return None

        def func(a, b):
            c = torch.relu(a)
            d = torch.mm(c, c)

            # Collective that is independent and should be prefetched during relu
            ar = _functional_collectives.all_reduce(b, "sum", "0")

            # Use both results
            return d * ar

        patches = {
            **get_patches(),
            "aten_distributed_optimizations.custom_runtime_estimation": custom_estimator_with_relu,
        }

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

            with torch._inductor.config.patch(patches):
                out, aten_graph_str = run_and_get_aten_graph(
                    torch.compile(func), inputs_a, inputs_b
                )

                # Verify that all_reduce is prefetched to run concurrently with relu
                # The collective should start before relu completes to enable perfect overlap
                FileCheck().check("all_reduce").check("relu").check("wait_tensor").run(
                    aten_graph_str
                )

                correct = func(inputs_a, inputs_b)
                self.assertTrue(same(out, correct))
                self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)


def get_bucket_patches(compute_multiplier=1.0):
    estimate_aten_runtime_part = functools.partial(
        estimate_aten_runtime, compute_multiplier=compute_multiplier
    )
    return {
        "aten_distributed_optimizations.custom_runtime_estimation": estimate_aten_runtime_part,
        "aten_distributed_optimizations.collective_bucketing": True,
        "aten_distributed_optimizations.bucket_exposed_first": False,
        "aten_distributed_optimizations.bucket_only_internode_comms": False,
        "reorder_for_locality": False,
        "triton.native_matmul": False,
        "reorder_for_compute_comm_overlap_passes": [],
        "compile_threads": 1,
        "force_disable_caches": True,
        # messes up test strings
        "aten_distributed_optimizations.insert_overlap_deps": False,
        # interferes with testing, / custom estimation
        "test_configs.assume_bucketing_reduces_latency": False,
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
                "aten_distributed_optimizations.insert_overlap_deps", True
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

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_custom_estimation_with_fake_tensor_mode(self):
        """Test that custom estimation can use FakeTensorMode for analysis."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        estimation_calls = 0

        def estimate_with_fake_mode(fx_node, compute_multiplier=1.0):
            with FakeTensorMode():
                nonlocal estimation_calls
                estimation_calls += 1
                if not isinstance(torch.rand([20]), torch._subclasses.FakeTensor):
                    raise AssertionError("Expected FakeTensor")

            return 1.0

        patches = get_bucket_patches()
        patches["aten_distributed_optimizations.custom_runtime_estimation"] = (
            estimate_with_fake_mode
        )

        def func(a, b, *, ranks):
            # Two independent all_gathers that should be bucketed
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)

            # Matmul that can hide the collectives
            mm1 = torch.matmul(a, a)

            return ag1.sum() + ag2.sum() + mm1.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs_a = torch.ones(4, 4, dtype=torch.float, device=device_type)
            inputs_b = torch.ones(4, 4, dtype=torch.float, device=device_type) * 2
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            with torch._inductor.config.patch(patches):
                compiled = torch.compile(func_c)
                out, aten_graph_str = run_and_get_aten_graph(
                    compiled, inputs_a, inputs_b
                )

            # Verify the custom estimation was called
            self.assertTrue(
                estimation_calls > 0, "Custom estimation should have been called"
            )

            correct = func(inputs_a, inputs_b, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_collective_benchmarking_with_real_pg(self):
        """Test collective benchmarking with real process group (falls back on fake)."""

        def func(a):
            # Test all three collective types with 8x8 (power of 2 size = 256 elements = 1024 bytes for fp32)
            ar = _functional_collectives.all_reduce(a, "sum", "0")
            ag = _functional_collectives.all_gather_tensor(
                a, 0, list(range(self.world_size))
            )
            rs = _functional_collectives.reduce_scatter_tensor(a, "sum", 0, "0")

            b = torch.matmul(a, a)
            c = torch.matmul(ar, b)
            return c.sum() + ag.sum() + rs.sum()

        patches = {
            **get_patches(),
            "aten_distributed_optimizations.collective_estimator": "benchmark",
            "aten_distributed_optimizations.custom_runtime_estimation": None,  # Remove custom estimation so benchmarking happens
        }

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(8, 8, dtype=torch.float, device=device_type) + self.rank

            with torch._inductor.config.patch(patches):
                compiled = torch.compile(func)
                out, aten_graph_str = run_and_get_aten_graph(compiled, inputs)

                # Verify all three collective types are present
                FileCheck().check_dag("all_reduce").check_dag("all_gather").check_dag(
                    "reduce_scatter"
                ).run(aten_graph_str)

                # Test passes if compilation succeeded with benchmarking enabled
                # Cache verification is tricky due to multiprocess test setup
                correct = func(inputs)
                self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_multidtype_bucketing(self):
        """Test that all_gathers with different dtypes get bucketed together."""

        def func(a, b, c, *, ranks):
            # Three all_gathers with different dtypes
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)  # float32
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)  # float16
            ag3 = _functional_collectives.all_gather_tensor(c, 0, ranks)  # float16

            # Use all results
            return ag1.sum() + ag2.sum() + ag3.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(4, 4, dtype=torch.float32, device=device_type)
            b = torch.ones(4, 4, dtype=torch.float16, device=device_type) * 2
            c = torch.ones(4, 4, dtype=torch.float16, device=device_type) * 3
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c)

            # Should have 1 bucketed all_gather despite different dtypes
            FileCheck().check_count(
                "torch.ops._c10d_functional.wait_tensor.default", 1, exactly=True
            ).run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, c, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_basic_all_reduce_bucketing(self):
        """Test that independent all_reduce operations get bucketed together."""

        def func(a, b, c):
            # Three independent all_reduces that should be bucketed
            ar1 = _functional_collectives.all_reduce(a, "sum", "0")
            ar2 = _functional_collectives.all_reduce(b, "sum", "0")
            ar3 = _functional_collectives.all_reduce(c, "sum", "0")

            return ar1.sum() + ar2.sum() + ar3.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            b = torch.ones(4, 4, dtype=torch.float, device=device_type) * 2
            c = torch.ones(4, 4, dtype=torch.float, device=device_type) * 3

            compiled = torch.compile(func)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c)

            # Should see a single bucketed all_reduce
            FileCheck().check_count(
                "torch.ops._c10d_functional.wait_tensor.default", 1, exactly=True
            ).run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, c)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_multiple_hiding_nodes_bucketing(self):
        """Test that collectives hidden by multiple compute ops can bucket together."""

        # Use 0.5 compute multiplier so each collective needs 2 matmuls to be fully hidden
        def estimate_with_half_compute(fx_node, override_size=None):
            return estimate_aten_runtime(fx_node, override_size, compute_multiplier=0.5)

        def func(a, b, *, ranks):
            # Two all_gathers that will be hidden by multiple compute operations
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)

            # Multiple compute operations that can hide the collectives
            # With 0.5 multiplier: mm1 and mm2 together hide ag1, mm2 and mm3 together hide ag2
            mm1 = torch.matmul(a, a.T)
            mm2 = torch.matmul(b, b.T)
            mm3 = torch.matmul(a + b, (a + b).T)

            return ag1.sum() + ag2.sum() + mm1.sum() + mm2.sum() + mm3.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)

            # Patch with custom estimation that uses 0.5 multiplier
            with torch._inductor.config.patch(
                {
                    "aten_distributed_optimizations.custom_runtime_estimation": estimate_with_half_compute
                }
            ):
                compiled = torch.compile(func_c)
                out, aten_graph_str = run_and_get_aten_graph(compiled, a, b)

            # Should have 1 bucketed all_gather (both ag1 and ag2 bucketed together)
            FileCheck().check_count(
                "torch.ops._c10d_functional.wait_tensor.default", 1, exactly=True
            ).run(aten_graph_str)

            # Verify bucketed collective is scheduled before all matmuls
            FileCheck().check("functional.all_gather_into_tensor").check(
                "aten.mm"
            ).check("aten.mm").check("aten.mm").check("wait_tensor").run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_bucketing_with_convert_dtype(self):
        """Test that all_gathers with dtype conversion get bucketed and produce correct results."""

        def func(a, b, c, d, *, ranks):
            # Convert inputs to float16 before all_gather
            a_fp16 = a.to(torch.float16)
            b_fp16 = b.to(torch.float16)

            # Two all_gathers with converted dtypes
            ag1 = _functional_collectives.all_gather_tensor(a_fp16, 0, ranks)
            ag2 = _functional_collectives.all_gather_tensor(b_fp16, 0, ranks)

            # same dtype
            ag3 = _functional_collectives.all_gather_tensor(c, 0, ranks)
            ag4 = _functional_collectives.all_gather_tensor(d, 0, ranks)

            return ag1, ag2, ag3, ag4

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(4, 4, dtype=torch.float32, device=device_type)
            b = torch.ones(4, 4, dtype=torch.float64, device=device_type) * 2
            c = torch.ones(4, 4, dtype=torch.float16, device=device_type) * 3
            d = torch.ones(4, 4, dtype=torch.float64, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c, d)

            # Should have 1 bucketed all_gather (both ag1 and ag2 bucketed together)
            FileCheck().check_count(
                "torch.ops._c10d_functional.wait_tensor.default", 1, exactly=True
            ).run(aten_graph_str)

            # Verify convert_element_type ops are removed (dtype conversion handled by _pre_bucket_all_gather)
            FileCheck().check_not("torch.ops.prims.convert_element_type").run(
                aten_graph_str
            )

            # Verify correctness - this tests that dtype conversion is handled correctly
            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))


def get_toy_model(device_type: str):
    """
    Helper to construct a small multi-layer ToyModel
    """

    class ToyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = torch.nn.Linear(4, 4)
            self.wk = torch.nn.Linear(4, 4)
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x):
            attn = self.wq(x) + self.wk(x)
            return self.proj(torch.nn.functional.relu(attn))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([ToyBlock() for _ in range(2)])
            self.norm = torch.nn.LayerNorm(4)

        def forward(self, x):
            for blk in self.layers:
                x = blk(x)
            return self.norm(x)

    model = ToyModel().to(device_type)
    return model


def apply_manual_reordering_and_get_graph(
    graph, module_bucket_plans, out_li, custom_module_stack_fn=None
) -> None:
    gm = graph.owning_module
    from torch._inductor.fx_passes.overlap_manual_scheduling import (
        ManualOverlapScheduler,
    )

    for node in list(gm.graph.nodes):
        # Handle both all-gather and reduce-scatter nodes for module_1
        if node.name in (
            "all_gather_into_tensor",
            "all_gather_into_tensor_1",
            "reduce_scatter_tensor",
            "reduce_scatter_tensor_1",
            "wait_tensor",
            "wait_tensor_1",
        ):
            node.meta["nn_module_stack"] = {"test": ["module_1", ""]}
        # Handle both all-gather and reduce-scatter nodes for module_2
        if node.name in (
            "all_gather_into_tensor_2",
            "all_gather_into_tensor_3",
            "reduce_scatter_tensor_2",
            "reduce_scatter_tensor_3",
            "wait_tensor_2",
            "wait_tensor_3",
        ):
            node.meta["nn_module_stack"] = {"test": ["module_2", ""]}

    overlapped_gm = ManualOverlapScheduler(
        gm,
        module_bucket_plans,
        insert_overlap_deps=False,
        module_stack_fn=custom_module_stack_fn,
    ).run()
    overlapped_gm.graph.lint()
    out_li.append(overlapped_gm.graph)


def run_and_get_manual_aten_graph(
    fn, module_bucket_plans, *inputs, custom_module_stack_fn=None
):
    li = []
    apply = functools.partial(
        apply_manual_reordering_and_get_graph,
        module_bucket_plans=module_bucket_plans,
        out_li=li,
        custom_module_stack_fn=custom_module_stack_fn,
    )
    with torch._inductor.config.patch(post_grad_custom_post_pass=apply):
        out = fn(*inputs)

    return out, li[0]


class TestManualOverlapBucketing(TestComputeCommReorderingMultiProc):
    """
    Tests for manual overlap scheduling and subgraph utilities.
    """

    def _get_all_gather_test_func(self):
        """Return the all-gather test function used by bucketing tests."""

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

        return func

    def _get_reduce_scatter_test_func(self):
        """Return the reduce-scatter test function used by bucketing tests.

        For reduce-scatter to be bucketed by ManualOverlapScheduler, the wait nodes
        must be directly returned as output (FSDP gradient pattern).
        """

        def func(a, b, c, d):
            # All 4 reduce-scatters are independent - COULD be bucketed together
            rs1 = _functional_collectives.reduce_scatter_tensor(a, "sum", 0, "0")
            rs2 = _functional_collectives.reduce_scatter_tensor(b, "sum", 0, "0")
            rs3 = _functional_collectives.reduce_scatter_tensor(c, "sum", 0, "0")
            rs4 = _functional_collectives.reduce_scatter_tensor(d, "sum", 0, "0")

            # Return reduce-scatter results directly as outputs (FSDP gradient pattern)
            return rs1, rs2, rs3, rs4

        return func

    def _run_manual_bucketing_test(
        self,
        collective_type,
        module_bucket_plans,
        expected_checks,
    ):
        """Common test logic for manual bucketing tests.

        Args:
            collective_type: Either "all_gather" or "reduce_scatter"
            module_bucket_plans: The bucket plans to use
            expected_checks: List of strings to check in the aten graph (in order)
        """
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

            if collective_type == "all_gather":
                func = self._get_all_gather_test_func()
                ranks = list(range(self.world_size))
                func_c = functools.partial(func, ranks=ranks)
                compiled = torch.compile(func_c)
                out, aten_graph = run_and_get_manual_aten_graph(
                    compiled, module_bucket_plans, a, b, c, d
                )
                correct = func(a, b, c, d, ranks=ranks)
            else:  # reduce_scatter
                func = self._get_reduce_scatter_test_func()
                compiled = torch.compile(func)
                out, aten_graph = run_and_get_manual_aten_graph(
                    compiled, module_bucket_plans, a, b, c, d
                )
                correct = func(a, b, c, d)

            # Run expected checks in order
            fc = FileCheck()
            for check in expected_checks:
                fc.check(check)
            fc.run(str(aten_graph))

            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_make_graph_view_and_get_subgraph_by_path(self):
        from torch._inductor.fx_passes.graph_view import (
            get_subgraph_by_path,
            make_graph_view,
        )

        model = get_toy_model(device_type)
        gm = torch.fx.symbolic_trace(model)
        graph_view = make_graph_view(gm.graph)
        # Fetch subgraph for first transformer layer
        sub_nodes = get_subgraph_by_path(graph_view, "layers.0.wq")
        self.assertEqual([n.name for n in sub_nodes], ["layers_0_wq"])

        # Fetch multiple paths at once
        multi_nodes = get_subgraph_by_path(graph_view, ["layers.0.wq", "layers.0.proj"])
        self.assertEqual(
            [n.name for n in multi_nodes], ["layers_0_wq", "layers_0_proj"]
        )

        # Fetch non existing paths
        non_exist_nodes = get_subgraph_by_path(graph_view, "nonexistent.module.path")
        self.assertEqual(non_exist_nodes, [])

        # Fetch mixed of existing and non existing paths
        mixed_nodes = get_subgraph_by_path(
            graph_view, ["layers.0.wq", "nonexistent.module.path"]
        )
        self.assertEqual([n.name for n in mixed_nodes], ["layers_0_wq"])

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_make_graph_view_and_get_subgraph_by_path_custom_module_stack_fn(self):
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export
        from torch._inductor.fx_passes.graph_view import (
            get_subgraph_by_path,
            make_graph_view,
        )

        model = get_toy_model(device_type)

        module_path_key = "module_path"
        # Add annotation to node.meta["custom"]
        for name, m in model.named_modules():
            m.forward = torch.fx.traceback.annotate_fn({module_path_key: name})(
                m.forward
            )

        def module_stack_fn(node):
            module_stack = node.meta.get("custom", {}).get(module_path_key, "")
            return [(module_stack, torch.nn.Module)]

        gm = dynamo_graph_capture_for_export(model)(torch.randn(2, 4).to(device_type))

        # delete "nn_module_stack" to make sure the graph view is only constructed from annotation
        for n in gm.graph.nodes:
            if "nn_module_stack" in n.meta:
                del n.meta["nn_module_stack"]

        graph_view = make_graph_view(gm.graph, module_stack_fn=module_stack_fn)
        # Fetch subgraph for first transformer layer
        sub_nodes = get_subgraph_by_path(graph_view, "layers.0.wq")
        self.assertEqual(
            [n.name for n in sub_nodes],
            [
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_weight_",
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_bias_",
                "linear",
            ],
        )

        # Fetch multiple paths at once
        multi_nodes = get_subgraph_by_path(graph_view, ["layers.0.wq", "layers.0.proj"])
        self.assertEqual(
            [n.name for n in multi_nodes],
            [
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_weight_",
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_bias_",
                "linear",
                "l_func_self_modules_layers_modules_0_modules_proj_parameters_weight_",
                "l_func_self_modules_layers_modules_0_modules_proj_parameters_bias_",
                "x",
            ],
        )

        # Fetch non existing paths
        non_exist_nodes = get_subgraph_by_path(graph_view, "nonexistent.module.path")
        self.assertEqual(non_exist_nodes, [])

        # Fetch mixed of existing and non existing paths
        mixed_nodes = get_subgraph_by_path(
            graph_view, ["layers.0.wq", "nonexistent.module.path"]
        )
        self.assertEqual(
            [n.name for n in mixed_nodes],
            [
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_weight_",
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_bias_",
                "linear",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_reordering_bucketing_pass_all_gather_separate_buckets(self):
        self._run_manual_bucketing_test(
            collective_type="all_gather",
            module_bucket_plans=["module_1", "module_2"],
            expected_checks=[
                "_pre_bucket_all_gather",
                "all_gather_into_tensor_out",
                "_pre_bucket_all_gather_1",
                "all_gather_into_tensor_out_1",
                "wait_tensor_4",
                "wait_tensor_5",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_all_gather_no_bucket(self):
        self._run_manual_bucketing_test(
            collective_type="all_gather",
            module_bucket_plans=[],
            expected_checks=[
                "all_gather_into_tensor",
                "all_gather_into_tensor_1",
                "all_gather_into_tensor_2",
                "all_gather_into_tensor_3",
                "wait_tensor",
                "wait_tensor_1",
                "wait_tensor_2",
                "wait_tensor_3",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_all_gather_single_bucket(self):
        self._run_manual_bucketing_test(
            collective_type="all_gather",
            module_bucket_plans=[["module_1", "module_2"]],
            expected_checks=[
                "_pre_bucket_all_gather",
                "all_gather_into_tensor_out",
                "wait_tensor_4",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_bucketing_reordering_pass_all_gather_single_bucket_custom_module_stack_fn(
        self,
    ):
        module_path_key = "module_path"

        def module_stack_fn(node):
            module_stack = node.meta.get("custom", {}).get(module_path_key, "")
            return [(module_stack, torch.nn.Module)]

        def func(a, b, c, d, *, ranks):
            # All 4 all-gathers are independent - COULD be bucketed together
            with torch.fx.traceback.annotate({module_path_key: "my_module_1"}):
                ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
                ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)
            with torch.fx.traceback.annotate({module_path_key: "my_module_2"}):
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
            out, aten_graph = run_and_get_manual_aten_graph(
                compiled,
                [["my_module_1", "my_module_2"]],
                a,
                b,
                c,
                d,
                custom_module_stack_fn=module_stack_fn,
            )

            (
                FileCheck()
                .check("_pre_bucket_all_gather")
                .check("all_gather_into_tensor_out")
                .check("wait_tensor_4")
                .run(str(aten_graph))
            )

            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))

            # Add metadata to the collective nodes to test preservation
            test_metadata = {
                "nn_module_stack": {
                    "test": ("module_1", ""),
                },
                "custom": {
                    "module_path": "my_module_1",
                },
            }

            # Verify metadata preservation: new bucketed nodes should have the metadata
            new_ag_nodes = aten_graph.find_nodes(
                op="call_function",
                target=torch.ops.bucketing._pre_bucket_all_gather.default,
            )
            new_wait_nodes = aten_graph.find_nodes(
                op="call_function",
                target=torch.ops._c10d_functional.wait_tensor.default,
            )

            all_new_nodes = list(new_ag_nodes) + list(new_wait_nodes)
            self.assertGreater(len(all_new_nodes), 0, "Should have created new nodes")

            for node in all_new_nodes:
                self.assertEqual(
                    node.meta.get("nn_module_stack"), test_metadata["nn_module_stack"]
                )
                self.assertEqual(node.meta.get("custom"), test_metadata["custom"])
                self.assertTrue(node.meta.get("stack_trace", None) is not None)
                self.assertTrue(
                    node.meta.get("bucketing_stack_trace_sources", None) is not None
                )
                self.assertTrue(
                    node.meta.get("bucketing_custom_sources", None) is not None
                )
                self.assertTrue(
                    node.meta.get("bucketing_nn_module_stack_sources", None) is not None
                )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_reduce_scatter_separate_buckets(self):
        self._run_manual_bucketing_test(
            collective_type="reduce_scatter",
            module_bucket_plans=["module_1", "module_2"],
            expected_checks=[
                "_pre_bucket_reduce_scatter",
                "reduce_scatter_tensor_4",
                "_pre_bucket_reduce_scatter_1",
                "reduce_scatter_tensor_5",
                "wait_tensor_4",
                "wait_tensor_5",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_reduce_scatter_single_bucket(self):
        self._run_manual_bucketing_test(
            collective_type="reduce_scatter",
            module_bucket_plans=[["module_1", "module_2"]],
            expected_checks=[
                "_pre_bucket_reduce_scatter",
                "reduce_scatter_tensor_4",
                "wait_tensor_4",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_reduce_scatter_no_bucket(self):
        self._run_manual_bucketing_test(
            collective_type="reduce_scatter",
            module_bucket_plans=[],
            expected_checks=[
                "reduce_scatter_tensor",
                "reduce_scatter_tensor_1",
                "reduce_scatter_tensor_2",
                "reduce_scatter_tensor_3",
                "wait_tensor",
                "wait_tensor_1",
                "wait_tensor_2",
                "wait_tensor_3",
            ],
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
