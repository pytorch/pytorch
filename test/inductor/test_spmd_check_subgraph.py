# Owner(s): ["module: inductor"]
"""Tests for spmd_check subgraph handling.

Verifies that spmd_check runs only at top level (via gm.meta["is_subgraph"])
and hashes subgraphs recursively, avoiding per-subgraph collective deadlocks.
"""

from unittest.mock import MagicMock, patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSpmdCheckSubgraph(TestCase):
    def test_recursive_passes_sets_subgraph_meta(self):
        """_recursive_post_grad_passes sets meta['is_subgraph'] on subgraphs."""
        from torch._inductor.compile_fx import _recursive_post_grad_passes

        main_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        sub_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())

        main_gm.subgraph_0 = sub_gm
        with main_gm.graph.inserting_before():
            main_gm.graph.get_attr("subgraph_0")
        main_gm.graph.output(None)
        main_gm.graph.lint()

        with (
            patch("torch._inductor.compile_fx.post_grad_passes"),
            patch("torch._inductor.compile_fx.config") as mock_config,
        ):
            mock_config.use_post_grad_passes = True
            _recursive_post_grad_passes(main_gm, is_inference=False)

        self.assertTrue(
            sub_gm.meta.get("is_subgraph", False),
            "Subgraph should have meta['is_subgraph'] = True",
        )
        self.assertFalse(
            main_gm.meta.get("is_subgraph", False),
            "Top-level should not have meta['is_subgraph']",
        )

    def test_nested_subgraphs_all_marked(self):
        """Deeply nested subgraphs should all have meta['is_subgraph'] = True."""
        from torch._inductor.compile_fx import _recursive_post_grad_passes

        main_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        mid_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        leaf_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())

        # main → mid → leaf
        mid_gm.leaf_sub = leaf_gm
        with mid_gm.graph.inserting_before():
            mid_gm.graph.get_attr("leaf_sub")
        mid_gm.graph.output(None)
        mid_gm.graph.lint()

        main_gm.mid_sub = mid_gm
        with main_gm.graph.inserting_before():
            main_gm.graph.get_attr("mid_sub")
        main_gm.graph.output(None)
        main_gm.graph.lint()

        with (
            patch("torch._inductor.compile_fx.post_grad_passes"),
            patch("torch._inductor.compile_fx.config") as mock_config,
        ):
            mock_config.use_post_grad_passes = True
            _recursive_post_grad_passes(main_gm, is_inference=False)

        self.assertTrue(leaf_gm.meta.get("is_subgraph", False))
        self.assertTrue(mid_gm.meta.get("is_subgraph", False))
        self.assertFalse(main_gm.meta.get("is_subgraph", False))

    def test_spmd_check_skipped_for_subgraph(self):
        """post_grad_passes skips spmd_check when meta['is_subgraph'] is True."""
        from torch._inductor.fx_passes.post_grad import post_grad_passes

        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        gm.graph.output(None)
        gm.meta["is_subgraph"] = True

        mock_spmd = MagicMock()

        with (
            patch("torch._inductor.fx_passes.spmd_check.spmd_check", mock_spmd),
            patch(
                "torch._inductor.fx_passes.post_grad._needs_spmd_graph_preservation",
                return_value=True,
            ),
            torch._inductor.config.patch(
                {
                    "aten_distributed_optimizations.spmd_check": True,
                }
            ),
        ):
            post_grad_passes(gm, is_inference=False)
            mock_spmd.assert_not_called()

    def test_spmd_check_called_for_top_level(self):
        """post_grad_passes calls spmd_check when meta['is_subgraph'] is absent."""
        from torch._inductor.fx_passes.post_grad import post_grad_passes

        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        gm.graph.output(None)

        mock_spmd = MagicMock(return_value=True)

        with (
            patch("torch._inductor.fx_passes.spmd_check.spmd_check", mock_spmd),
            patch(
                "torch._inductor.fx_passes.post_grad._needs_spmd_graph_preservation",
                return_value=True,
            ),
            torch._inductor.config.patch(
                {
                    "aten_distributed_optimizations.spmd_check": True,
                }
            ),
        ):
            post_grad_passes(gm, is_inference=False)
            mock_spmd.assert_called_once()

    def test_hash_bundle_includes_subgraphs(self):
        """_compute_hash_bundle recurses into subgraphs."""
        from torch._inductor.fx_passes.spmd_check import _compute_hash_bundle

        main_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        sub_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())

        sub_gm.graph.output(None)
        main_gm.subgraph_0 = sub_gm
        with main_gm.graph.inserting_before():
            main_gm.graph.get_attr("subgraph_0")
        main_gm.graph.output(None)

        bundle = _compute_hash_bundle(main_gm)
        self.assertIsNotNone(bundle)
        self.assertIn("", bundle, "Should have top-level hash")
        self.assertIn("subgraph_0", bundle, "Should have subgraph hash")

    def test_hash_bundle_nested_subgraphs(self):
        """_compute_hash_bundle recurses into deeply nested subgraphs."""
        from torch._inductor.fx_passes.spmd_check import _compute_hash_bundle

        leaf_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        leaf_gm.graph.output(None)

        mid_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        mid_gm.leaf_sub = leaf_gm
        with mid_gm.graph.inserting_before():
            mid_gm.graph.get_attr("leaf_sub")
        mid_gm.graph.output(None)

        main_gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        main_gm.mid_sub = mid_gm
        with main_gm.graph.inserting_before():
            main_gm.graph.get_attr("mid_sub")
        main_gm.graph.output(None)

        bundle = _compute_hash_bundle(main_gm)
        self.assertIsNotNone(bundle)
        self.assertEqual(sorted(bundle.keys()), ["", "mid_sub", "mid_sub.leaf_sub"])

    def test_different_subgraph_contents_different_hashes(self):
        """Subgraphs with different ops produce different hashes."""
        from torch._inductor.fx_passes.spmd_check import _compute_hash_bundle

        def _make_graph_with_subgraph(add_op: bool) -> torch.fx.GraphModule:
            sub_graph = torch.fx.Graph()
            if add_op:
                x = sub_graph.placeholder("x")
                sub_graph.call_function(torch.relu, (x,))
            sub_graph.output(None)
            sub_gm = torch.fx.GraphModule(torch.nn.Module(), sub_graph)

            main_graph = torch.fx.Graph()
            main_gm = torch.fx.GraphModule(torch.nn.Module(), main_graph)
            main_gm.subgraph_0 = sub_gm
            with main_graph.inserting_before():
                main_graph.get_attr("subgraph_0")
            main_graph.output(None)
            return main_gm

        bundle_with = _compute_hash_bundle(_make_graph_with_subgraph(True))
        bundle_without = _compute_hash_bundle(_make_graph_with_subgraph(False))

        self.assertIsNotNone(bundle_with)
        self.assertIsNotNone(bundle_without)
        self.assertNotEqual(
            bundle_with["subgraph_0"],
            bundle_without["subgraph_0"],
        )


if __name__ == "__main__":
    run_tests()
