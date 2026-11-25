# Owner(s): ["module: inductor"]

"""Tests for fusion region detection."""

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.fx as fx
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.fx_passes import fusion_regions
from torch._inductor.fx_passes.control_dependencies import get_mutation_dependencies
from torch._inductor.fx_passes.fusion_regions import (
    build_fusion_regions,
    estimate_fused_node_costs,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    gather_node_runtime_estimations,
    OverlapScheduler,
)
from torch._inductor.fx_utils import get_storage
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.ops import aten
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    xfailIfNoAcceleratorTriton,
)
from torch.testing._internal.inductor_utils import GPU_TYPE


HAS_GPU = torch.cuda.is_available() or torch.xpu.is_available()


@instantiate_parametrized_tests
class TestFusionCandidates(InductorTestCase):
    def _make_graph(self, fn, *shapes):
        with FakeTensorMode():
            return make_fx(fn)(*(torch.ones(shape) for shape in shapes))

    def test_mutation_deps_without_distributed(self):
        def fn(x, y):
            old = x.sin()
            x.copy_(y)
            return old, x

        gm = self._make_graph(fn, (4,), (4,))
        reader = next(n for n in gm.graph.nodes if n.target is aten.sin.default)
        writer = next(n for n in gm.graph.nodes if n.target is aten.copy_.default)
        with patch.object(torch.ops, "_c10d_functional", object()):
            deps = get_mutation_dependencies(gm)
        self.assertIn(reader, (deps or {}).get(writer, ()))

    @parametrize(
        "kind", ("mm", "bmm", "addmm", "baddbmm", "row_reduction", "horizontal")
    )
    @config.patch(prologue_fusion=True, epilogue_fusion=True)
    @config.patch({"triton.native_matmul": True})
    def test_matmul_candidates(self, kind):
        def fn(x, w, bias):
            op = kind if kind in ("mm", "bmm", "addmm", "baddbmm") else "mm"
            args = (x.sin(), w)
            if kind in ("addmm", "baddbmm"):
                args = (bias, *args)
            result = getattr(torch, op)(*args)
            if kind == "row_reduction":
                return result.sum(-1).abs()
            if kind == "horizontal":
                result = result + x @ w
            return result.relu()

        prefix = (2,) if kind in ("bmm", "baddbmm") else ()
        gm = self._make_graph(fn, (*prefix, 4, 8), (*prefix, 8, 4), (*prefix, 4, 4))
        regions = build_fusion_regions(gm, prioritize=True)
        calls = [n for n in gm.graph.nodes if n.op == "call_function"]
        self.assertTrue(all(regions[n] is regions[calls[0]] for n in calls))

    @parametrize("prioritize", (False, True))
    def test_gemm_costs_and_mutable_destination_writes(self, prioritize):
        def fn(x, w, buffer):
            result = (x.sin() @ w).relu()
            buffer.copy_(torch.cat((result, result)))
            return result, buffer

        gm = self._make_graph(fn, (4, 8), (8, 4), (8, 4))
        buffer = list(gm.graph.find_nodes(op="placeholder"))[-1]
        writer = next(n for n in gm.graph.nodes if n.target is aten.copy_.default)
        output = next(n for n in gm.graph.nodes if n.op == "output")
        output.replace_input_with(writer, buffer)
        cat = next(n for n in gm.graph.nodes if n.target is aten.cat.default)
        compute = next(n for n in gm.graph.nodes if n.target is aten.mm.default)
        regions = build_fusion_regions(gm, prioritize=prioritize)
        self.assertIs(regions[cat], regions[writer])
        with (
            patch.object(
                fusion_regions, "estimate_roofline_runtime_ms", return_value=17.0
            ),
            patch.object(
                fusion_regions, "get_transfer_time", return_value=1.0
            ) as transfer,
        ):
            costs = estimate_fused_node_costs(regions, prioritize=prioritize)
            custom = gather_node_runtime_estimations(
                gm,
                custom_runtime_estimation=lambda n, _: 42.0,
                enable_fusion_regions=True,
                prioritize_fusion_regions=prioritize,
            )
        self.assertEqual(costs.get(compute), 17.0 if prioritize else None)
        self.assertEqual(costs[cat], 0.0)
        self.assertGreater(costs[writer], 0.0)
        outputs = [t for call in transfer.call_args_list for t in call.args[1]]
        storage = get_storage(buffer.meta["val"])
        self.assertEqual(any(get_storage(t) == storage for t in outputs), prioritize)
        self.assertEqual(custom[cat], 42.0 if prioritize else 0.0)

    @unittest.skipUnless(dist.is_available(), "requires distributed")
    @parametrize("destination", ("input", "out"))
    def test_collective_accesses_last_until_wait(self, destination):
        c10d = torch.ops._c10d_functional
        graph = fx.Graph()
        buffer, out, source = [
            graph.placeholder(n) for n in ("buffer", "out", "source")
        ]
        with FakeTensorMode():
            sizes = (4, 8, 4 if destination == "input" else 8)
            for node, size in zip((buffer, out, source), sizes):
                node.meta["val"] = torch.ones(size)
        collective = graph.call_function(
            c10d.all_gather_into_tensor_out.default,
            (buffer, 2, "group"),
            {"out": out},
        )
        collective.meta["val"] = out.meta["val"]
        wait = graph.call_function(c10d.wait_tensor.default, (collective,))
        wait.meta["val"] = out.meta["val"]
        target = buffer if destination == "input" else out
        writer = graph.call_function(aten.copy_.default, (target, source))
        writer.meta["val"] = target.meta["val"]
        graph.output((wait, writer))
        gm = fx.GraphModule({}, graph)
        self.assertIn(wait, (get_mutation_dependencies(gm) or {}).get(writer, ()))
        source.meta.pop("val")
        self.assertIsNone(get_mutation_dependencies(gm))


@unittest.skipUnless(dist.is_available(), "requires distributed operators")
class TestMutationDependencyLowering(InductorTestCase):
    @parametrize("kind", ("copy", "out", "foreach", "materializing_view", "getitem"))
    def test_reordered_mutations_compile(self, device, kind):
        def fn(buffer, source):
            alias = buffer.view(4, 4)
            if kind == "materializing_view":
                old = buffer.T.reshape(-1)
            elif kind == "getitem":
                old = buffer[[0, 1, 2, 3]]
            else:
                old = buffer.sin()
            if kind == "out":
                torch.add(source, source, out=alias)
            elif kind == "foreach":
                torch._foreach_copy_([alias], [source])
            else:
                alias.copy_(source)
            middle = buffer.sum()
            buffer.copy_(source + 1)
            return old, middle, alias.sum(), buffer

        with FakeTensorMode():
            gm = make_fx(fn)(
                torch.ones(4, 4, device=device), torch.ones(4, 4, device=device)
            )
        original_nodes = set(gm.graph.nodes)
        with (
            patch.object(
                OverlapScheduler,
                "_compute_off_path_score",
                lambda self, n: -self.node_idx[n],
            ),
            patch.object(
                OverlapScheduler,
                "_align_compute_nodes_runtime_estimations_across_all_distributed_ranks",
            ),
        ):
            OverlapScheduler(
                gm,
                max_in_flight_gb=1.0,
                max_compute_pre_fetch=1,
                collective_bucketing=False,
                insert_overlap_deps=False,
                compute_overlap_multipler=1.0,
                max_coll_distance=1,
                custom_runtime_estimation=lambda node, size: 1.0,
                prioritize_fusion_regions=True,
            ).run()
        self.assertEqual(set(gm.graph.nodes), original_nodes)
        gm.recompile()

        buffer = torch.ones(4, 4, device=device)
        source = torch.full((4, 4), 2.0, device=device)
        expected = fn(buffer.clone(), source)
        self.assertEqual(gm(buffer.clone(), source), expected)
        if kind == "out":
            actual = torch.compile(gm, fullgraph=True)(buffer.clone(), source)
        else:
            actual = compile_fx_inner(gm, [buffer, source], is_inference=True)(
                [buffer.clone(), source]
            )
        self.assertEqual(actual, expected)


instantiate_device_type_tests(
    TestMutationDependencyLowering, globals(), only_for=("cpu", GPU_TYPE)
)


@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestFusionRegionDetection(InductorTestCase):
    """Tests for fusion region detection and grouping."""

    def setUp(self):
        super().setUp()
        self.device = GPU_TYPE

    def test_fusion_region_grouping(self):
        """Test that connected fusible ops are grouped into regions."""
        from torch._inductor.fx_passes.fusion_regions import build_fusion_regions

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            traced = make_fx(lambda a: (a + 1) * 2 - 3)(a)

        region_of = build_fusion_regions(traced)
        (add_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.add.Tensor
        )
        (mul_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.mul.Tensor
        )
        (sub_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.sub.Tensor
        )

        # All pointwise ops should be in the same region
        self.assertIs(region_of[add_node], region_of[mul_node])
        self.assertIs(region_of[mul_node], region_of[sub_node])

    def test_mm_not_in_fusion_region(self):
        """Test that mm ops are not included in fusion regions."""
        from torch._inductor.fx_passes.fusion_regions import build_fusion_regions

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            traced = make_fx(lambda a: torch.mm(a + 1, a) * 2)(a)

        region_of = build_fusion_regions(traced)
        (mm_node,) = traced.graph.find_nodes(op="call_function", target=aten.mm.default)
        self.assertNotIn(mm_node, region_of)

    def test_strict_local_fusion_no_cross_mm(self):
        """Test that fusion regions don't cross non-fusible (mm) boundaries.

        Pattern:
            b = a + 1
            b2 = b - 1
            c = y @ y   # mm boundary
            d = b * 2
            d2 = d / 2

        Even though b/b2 and d/d2 are connected (d uses b), they should NOT be
        in the same fusion region because mm appears between them in topological order.
        """
        from torch._inductor.fx_passes.fusion_regions import build_fusion_regions

        def model(a, y):
            b = a + 1
            b2 = b - 1
            c = y @ y  # mm boundary
            d = b * 2
            d2 = d / 2
            return b2, c, d2

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            y = torch.ones(64, 64, device=self.device)
            traced = make_fx(model)(a, y)

        region_of = build_fusion_regions(traced)

        (add_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.add.Tensor
        )
        (div_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.div.Tensor
        )

        # add and div are connected but mm is between them - must be separate regions
        self.assertIn(add_node, region_of)
        self.assertIn(div_node, region_of)
        self.assertIsNot(
            region_of[add_node],
            region_of[div_node],
            "Ops before and after mm should not be in the same fusion region",
        )

    @xfailIfNoAcceleratorTriton
    def test_estimate_fused_node_costs(self):
        """Test that fused costs correctly exclude internal I/O."""
        from torch._inductor.fx_passes.fusion_regions import (
            build_fusion_regions,
            estimate_fused_node_costs,
        )

        def model(a, b):
            # region 1: 3-node chain, middle node has only internal I/O
            x = a + 1  # reads a (ext), writes to neg (int)
            x = -x  # reads add (int), writes to mul (int) -> cost=0
            x = x * 2  # reads neg (int), writes to mm (ext)
            x = torch.mm(x, a)  # mm boundary
            x = (x - 1) / 2  # region 2
            y = (b + 3) * 4  # region 3 (independent)
            return x, y

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            b = torch.ones(64, 64, device=self.device)
            traced = make_fx(model)(a, b)

        region_of = build_fusion_regions(traced)
        fused_costs = estimate_fused_node_costs(region_of)

        # All region nodes should have a fused cost entry
        for node in region_of:
            self.assertIn(node, fused_costs)

        # Graph should NOT be mutated (no call_module nodes)
        self.assertEqual(len(list(traced.graph.find_nodes(op="call_module"))), 0)

        # Nodes with only internal I/O should have cost 0
        zero_cost_nodes = [n for n, c in fused_costs.items() if c == 0.0]
        nonzero_cost_nodes = [n for n, c in fused_costs.items() if c > 0.0]
        self.assertGreater(len(zero_cost_nodes), 0, "Should have internal-only nodes")
        self.assertGreater(len(nonzero_cost_nodes), 0, "Should have external I/O nodes")

        # Total fused cost should be less than sum of individual roofline estimates
        from torch._inductor.fx_passes.overlap_scheduling import (
            estimate_roofline_runtime_ms,
        )

        individual_total = sum(estimate_roofline_runtime_ms(n) for n in region_of)
        fused_total = sum(fused_costs.values())
        self.assertLessEqual(fused_total, individual_total)

    def test_is_fusible_node(self):
        """Test is_fusible_node correctly classifies ops."""
        from torch._inductor.fx_passes.fusion_regions import is_fusible_node

        with FakeTensorMode():
            a = torch.randn(64, 64, device=self.device)
            traced = make_fx(lambda a: torch.linalg.qr(torch.mm(a + 1, a)))(a)

        def find_node(target):
            for n in traced.graph.nodes:
                if n.op == "call_function" and n.target == target:
                    return n
            return None

        # aten.add - pointwise, should be fusible
        add_node = find_node(aten.add.Tensor)
        self.assertIsNotNone(add_node)
        self.assertTrue(is_fusible_node(add_node))

        # aten.mm - requires enabled GEMM fusion support
        mm_node = find_node(aten.mm.default)
        self.assertIsNotNone(mm_node)
        with config.patch(
            {
                "prologue_fusion": False,
                "epilogue_fusion": False,
                "triton.native_matmul": False,
            }
        ):
            self.assertFalse(is_fusible_node(mm_node))
        with config.patch(prologue_fusion=True):
            self.assertTrue(is_fusible_node(mm_node))

        # aten.linalg_qr - fallback op, should NOT be fusible
        qr_node = find_node(aten.linalg_qr.default)
        self.assertIsNotNone(qr_node)
        self.assertFalse(is_fusible_node(qr_node))

    @xfailIfNoAcceleratorTriton
    def test_fused_costs_does_not_mutate_graph(self):
        """estimate_fused_node_costs must not mutate the graph."""
        from torch._inductor.fx_passes.fusion_regions import (
            build_fusion_regions,
            estimate_fused_node_costs,
        )

        def model(a, b):
            x = a + 1
            x = x * 2
            x = torch.mm(x, a)
            x = x - 1
            x = x / 2
            y = b + 3
            z = y * 4
            return x, y, z

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            b = torch.ones(64, 64, device=self.device)
            traced = make_fx(model)(a, b)

        graph_str_before = traced.print_readable(print_output=False)

        region_of = build_fusion_regions(traced)
        estimate_fused_node_costs(region_of)

        graph_str_after = traced.print_readable(print_output=False)
        self.assertEqual(
            graph_str_before,
            graph_str_after,
            "Graph must not be mutated by estimate_fused_node_costs",
        )

    @xfailIfNoAcceleratorTriton
    def test_fused_costs_handles_forced_bad_region(self):
        """estimate_fused_node_costs works for any region_of mapping."""
        from torch._inductor.fx_passes.fusion_regions import estimate_fused_node_costs

        with FakeTensorMode():
            t = torch.randn(64, 64, device=self.device)

            def fn(x):
                a = x.neg()
                b = a.relu()
                mm_out = torch.mm(b, x)
                v = mm_out.view(64, 64)
                c = v.neg()
                d = c.abs()
                return b + d

            traced = make_fx(fn)(t)

        fusible_nodes = [
            n
            for n in traced.graph.nodes
            if n.op == "call_function"
            and n.target
            in (
                aten.neg.default,
                aten.relu.default,
                aten.abs.default,
            )
        ]

        self.assertGreaterEqual(len(fusible_nodes), 4)
        from torch.utils._ordered_set import OrderedSet

        bad_group = OrderedSet(fusible_nodes)
        region_of = dict.fromkeys(fusible_nodes, bad_group)

        costs = estimate_fused_node_costs(region_of)
        self.assertEqual(len(costs), len(fusible_nodes))
        self.assertTrue(all(c >= 0.0 for c in costs.values()))


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
