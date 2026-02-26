# Owner(s): ["module: inductor"]

"""Tests for fusion region detection."""

import unittest

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.ops import aten


HAS_GPU = torch.cuda.is_available()


@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestFusionRegionDetection(InductorTestCase):
    """Tests for fusion region detection and grouping."""

    def setUp(self):
        super().setUp()
        self.device = "cuda"

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

    def test_collapse_and_expand_fusion_regions(self):
        """Test collapse creates call_module nodes and expand restores graph."""
        from torch._inductor.fx_passes.fusion_regions import (
            build_fusion_regions,
            collapse_fusion_regions,
            expand_fusion_regions,
        )

        def model(a, b):
            x = (a + 1) * 2  # region 1
            x = torch.mm(x, a)  # mm boundary
            x = (x - 1) / 2  # region 2
            y = (b + 3) * 4  # region 3 (independent)
            return x, y

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            b = torch.ones(64, 64, device=self.device)
            traced = make_fx(model)(a, b)

        region_of = build_fusion_regions(traced)
        new_region_of = collapse_fusion_regions(traced, region_of)

        # Should have 3 fusion regions
        self.assertEqual(len(new_region_of), 3)
        call_modules = list(traced.graph.find_nodes(op="call_module"))
        self.assertEqual(len(call_modules), 3)

        # Verify metas and bytes on each region
        tensor_bytes = 64 * 64 * 4  # 64x64 float32
        for module_node in call_modules:
            self.assertIn("val", module_node.meta)
            region = new_region_of[module_node]
            # Each region has 1 input + 1 output = 2 tensors
            self.assertEqual(region.total_bytes, 2 * tensor_bytes)
            self.assertGreater(region.cost_ms, 0)

        # Expand back
        expand_fusion_regions(traced, new_region_of)
        self.assertEqual(len(list(traced.graph.find_nodes(op="call_module"))), 0)

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

        # aten.mm - has flop counter, should NOT be fusible
        mm_node = find_node(aten.mm.default)
        self.assertIsNotNone(mm_node)
        self.assertFalse(is_fusible_node(mm_node))

        # aten.linalg_qr - fallback op, should NOT be fusible
        qr_node = find_node(aten.linalg_qr.default)
        self.assertIsNotNone(qr_node)
        self.assertFalse(is_fusible_node(qr_node))

    def test_collapse_expand_preserves_correctness(self):
        """Test that collapse and expand preserve numerical correctness.

        Run the module before collapse, after collapse, and after expand,
        verifying all produce the same results. Also verify the graph string
        is identical before and after the round-trip.
        """
        from torch._inductor.fx_passes.fusion_regions import (
            build_fusion_regions,
            collapse_fusion_regions,
            expand_fusion_regions,
        )

        def model(a, b):
            # Group 1: before mm
            x = a + 1
            x = x * 2

            # mm boundary
            x = torch.mm(x, a)

            # Group 2: after mm
            x = x - 1
            x = x / 2

            # Group 3: separate chain with multi-output
            y = b + 3
            z = y * 4

            return x, y, z

        # Use real tensors for numerical correctness check
        torch.manual_seed(42)
        a = torch.randn(64, 64, device=self.device)
        b = torch.randn(64, 64, device=self.device)

        # Get expected results from eager execution
        expected = model(a, b)

        # Trace with FakeTensorMode, then run with real tensors
        with FakeTensorMode():
            a_fake = torch.ones(64, 64, device=self.device)
            b_fake = torch.ones(64, 64, device=self.device)
            traced = make_fx(model)(a_fake, b_fake)

        # Capture graph string before any transformation
        graph_str_before = traced.print_readable(print_output=False)

        # Run traced module before any transformation
        result_before = traced(a, b)
        for i, (exp, res) in enumerate(zip(expected, result_before)):
            self.assertEqual(exp, res, f"Output {i} mismatch before collapse")

        # Build and collapse fusion regions
        region_of = build_fusion_regions(traced)
        new_region_of = collapse_fusion_regions(traced, region_of)

        # Run traced module after collapse (with call_module nodes)
        traced.recompile()
        result_after_collapse = traced(a, b)
        for i, (exp, res) in enumerate(zip(expected, result_after_collapse)):
            self.assertEqual(exp, res, f"Output {i} mismatch after collapse")

        # Expand (inline) the fusion regions back
        expand_fusion_regions(traced, new_region_of)

        # Run traced module after expand
        traced.recompile()
        result_after_expand = traced(a, b)
        for i, (exp, res) in enumerate(zip(expected, result_after_expand)):
            self.assertEqual(exp, res, f"Output {i} mismatch after expand")

        # Verify graph string is identical after round-trip
        # Note: Multi-output regions may add getitem nodes, so we run DCE first
        traced.graph.eliminate_dead_code()
        traced.recompile()
        graph_str_after = traced.print_readable(print_output=False)
        self.assertEqual(
            graph_str_before,
            graph_str_after,
            "Graph string should be identical after collapse/expand round-trip",
        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
