"""Tests for fusion region detection."""

import unittest

import torch
from torch.ops import aten

from torch._inductor.test_case import TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx


HAS_GPU = torch.cuda.is_available()


@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestFusionRegionDetection(InductorTestCase):
    """Tests for fusion region detection and grouping."""

    def setUp(self):
        super().setUp()
        self.device = "cuda"

    def test_view_nodes_identified(self):
        """Test that view ops are identified correctly."""
        from torch._inductor.fx_passes.fusion_regions import is_view_node

        with FakeTensorMode():
            a = torch.ones(1024, device=self.device)
            traced = make_fx(lambda a: a.view(16, 64))(a)

        (view_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.view.default
        )
        self.assertTrue(is_view_node(view_node))

    def test_fusible_nodes_identified(self):
        """Test that fusible nodes (pointwise, reduction) are identified."""
        from torch._inductor.fx_passes.fusion_regions import is_fusible_node, is_view_node

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            traced = make_fx(lambda a: a + 1)(a)

        (add_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.add.Tensor
        )
        self.assertTrue(is_fusible_node(add_node))
        self.assertFalse(is_view_node(add_node))

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

        This is the key property of strict local fusion: ops before and after
        a non-fusible node are in separate regions, even if they're connected.
        """
        from torch._inductor.fx_passes.fusion_regions import build_fusion_regions

        # a + 1 -> mm -> result * 2
        # The add and mul should NOT be in the same region (mm is between them)
        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            traced = make_fx(lambda a: torch.mm(a + 1, a) * 2)(a)

        region_of = build_fusion_regions(traced)

        add_nodes = list(
            traced.graph.find_nodes(op="call_function", target=aten.add.Tensor)
        )
        mul_nodes = list(
            traced.graph.find_nodes(op="call_function", target=aten.mul.Tensor)
        )

        # add is before mm, mul is after mm - they should be in different regions
        # (or one/both not in a region at all if they're singletons)
        if add_nodes and mul_nodes and add_nodes[0] in region_of and mul_nodes[0] in region_of:
            self.assertIsNot(
                region_of[add_nodes[0]],
                region_of[mul_nodes[0]],
                "Ops before and after mm should not be in the same fusion region",
            )

    def test_fusion_regions_with_mm_boundary_and_bytes(self):
        """Test fusion regions with mm boundary, verifying group count and exact bytes.

        Pattern:
            pointwise1 -> pointwise2 -> mm -> pointwise3 -> pointwise4
            separate_pointwise1 -> separate_pointwise2

        Expected: 3 fusion groups
        - Group 1: pointwise1, pointwise2 (before mm)
        - Group 2: pointwise3, pointwise4 (after mm)
        - Group 3: separate_pointwise1, separate_pointwise2 (independent)
        """
        from torch._inductor.fx_passes.fusion_regions import (
            FusionRegion,
            build_fusion_regions,
            collapse_fusion_regions,
            expand_fusion_regions,
        )

        def model(a, b):
            # Group 1: before mm
            x = a + 1  # pointwise1
            x = x * 2  # pointwise2

            # mm boundary
            x = torch.mm(x, a)

            # Group 2: after mm
            x = x - 1  # pointwise3
            x = x / 2  # pointwise4

            # Group 3: separate chain with multi-output (not connected to above)
            y = b + 3  # separate_pointwise1
            z = y * 4  # separate_pointwise2
            # Both y and z are used, so this region has 2 outputs

            return x, y, z

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            b = torch.ones(64, 64, device=self.device)
            traced = make_fx(model)(a, b)

        # Count nodes before collapse
        nodes_before = [n for n in traced.graph.nodes if n.op == "call_function"]
        num_nodes_before = len(nodes_before)

        # Build fusion regions
        region_of = build_fusion_regions(traced)

        # Should have 6 pointwise nodes in regions (2+2+2)
        self.assertEqual(len(region_of), 6, f"Expected 6 nodes in regions, got {len(region_of)}")

        # Collapse fusion regions
        new_region_of, replaced = collapse_fusion_regions(traced, region_of)

        # Should have exactly 3 fusion regions (call_module nodes)
        self.assertEqual(len(new_region_of), 3, f"Expected 3 fusion regions, got {len(new_region_of)}")

        # After collapse: should have call_module nodes instead of pointwise
        call_modules = list(traced.graph.find_nodes(op="call_module"))
        self.assertEqual(len(call_modules), 3, "Should have 3 call_module nodes after collapse")

        # Verify each region has correct exact bytes and cost
        # 64x64 float32 tensor = 64*64*4 = 16384 bytes
        from torch._inductor.utils import get_gpu_dram_gbps

        tensor_bytes = 64 * 64 * 4
        fusion_bw_gbps = get_gpu_dram_gbps()
        fusion_bw_bytes_per_s = fusion_bw_gbps * 1e9

        for module_node, region in new_region_of.items():
            self.assertIsInstance(region, FusionRegion)
            self.assertEqual(len(region.nodes), 2, f"Region {module_node.name} should have 2 nodes")

        # Verify metas are preserved on call_module nodes (including multi-output)
        single_output_count = 0
        multi_output_count = 0
        for module_node in call_modules:
            self.assertIn("val", module_node.meta, f"call_module {module_node.name} should have 'val' meta")
            val = module_node.meta["val"]

            if isinstance(val, (list, tuple)):
                # Multi-output region (Group 3 with y, z)
                multi_output_count += 1
                self.assertEqual(len(val), 2, f"Multi-output region should have 2 outputs")
                for i, v in enumerate(val):
                    self.assertEqual(v.shape, (64, 64), f"Output {i} should have shape (64, 64)")
                # 1 input + 2 outputs = 3 tensors
                expected_bytes = 3 * tensor_bytes
            else:
                # Single output region (Groups 1 and 2)
                single_output_count += 1
                self.assertEqual(val.shape, (64, 64), f"call_module {module_node.name} should have shape (64, 64)")
                # 1 input + 1 output = 2 tensors
                expected_bytes = 2 * tensor_bytes

            region = new_region_of[module_node]
            expected_cost_ms = (expected_bytes / fusion_bw_bytes_per_s) * 1000
            self.assertEqual(
                region.total_bytes, expected_bytes,
                f"Region {module_node.name} should have {expected_bytes} bytes, got {region.total_bytes}"
            )
            self.assertEqual(
                region.cost_ms, expected_cost_ms,
                f"Region {module_node.name} should have {expected_cost_ms} ms, got {region.cost_ms}"
            )

        # Verify we have both single and multi-output regions
        self.assertEqual(single_output_count, 2, "Should have 2 single-output regions")
        self.assertEqual(multi_output_count, 1, "Should have 1 multi-output region")

        # Now expand (inline) the fusion regions back
        expand_replaced = expand_fusion_regions(traced, new_region_of, replaced)

        # After expand: should have no call_module nodes
        call_modules_after = list(traced.graph.find_nodes(op="call_module"))
        self.assertEqual(len(call_modules_after), 0, "Should have no call_module nodes after expand")

        # Verify metas are preserved on expanded nodes
        for erased_node, replacement_node in expand_replaced.items():
            self.assertIn("val", replacement_node.meta, f"Expanded node {replacement_node.name} should have 'val' meta")

        # Count non-getitem nodes after expand
        # Multi-output regions add getitem nodes when expanded, so exclude those
        nodes_after = [
            n for n in traced.graph.nodes
            if n.op == "call_function" and "getitem" not in str(n.target)
        ]
        num_nodes_after = len(nodes_after)
        self.assertEqual(num_nodes_after, num_nodes_before, "Non-getitem node count should be same after expand")


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
