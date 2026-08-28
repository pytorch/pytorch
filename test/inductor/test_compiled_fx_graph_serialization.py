# Owner(s): ["module: inductor"]

"""Tests that CompiledFxGraph serializes _original_gm via GraphPickler.

_original_gm is a deepcopy of the pre-compiled FX graph, stored when
wrap_inductor_compiled_regions is enabled. It is used for FakeTensor
shape inference (inductor_compiled_code_fake). Regular pickle cannot
serialize GraphModules containing higher-order ops with lifted buffers
(e.g. flex_attention with BlockMask mask_mod_other_buffers), because
deserialization retraces via KeepModules().trace() which calls
validate_subgraph_args_types on Proxy-typed arguments.

prepare_for_serialization converts _original_gm to bytes via
GraphPickler (which serializes the graph structure directly without
retracing), and post_compile deserializes it back on load.
"""

import torch
from torch._inductor.output_code import CompiledFxGraph
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCompiledFxGraphSerialization(TestCase):
    def _make_gm(self):
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        gm.graph.placeholder("x")
        gm.graph.output(next(iter(gm.graph.nodes)))
        gm.recompile()
        return gm

    def _make_cfg(self, gm=None):
        cfg = CompiledFxGraph.__new__(CompiledFxGraph)
        cfg._original_gm = gm
        cfg._serialized_original_gm = None
        cfg.current_callable = None
        cfg.recursively_apply_fns = None
        cfg.compiled_fn_runner = None
        return cfg

    def test_prepare_serializes_original_gm_via_graph_pickler(self):
        """prepare_for_serialization converts _original_gm to GraphPickler bytes."""
        cfg = self._make_cfg(self._make_gm())

        self.assertIsNotNone(cfg._original_gm)
        self.assertIsNone(cfg._serialized_original_gm)

        cfg.prepare_for_serialization()

        self.assertIsNone(cfg._original_gm)
        self.assertIsNotNone(cfg._serialized_original_gm)
        self.assertIsInstance(cfg._serialized_original_gm, bytes)

    def test_prepare_with_none_original_gm(self):
        """prepare_for_serialization is a no-op when _original_gm is None."""
        cfg = self._make_cfg(None)
        cfg.prepare_for_serialization()

        self.assertIsNone(cfg._original_gm)
        self.assertIsNone(cfg._serialized_original_gm)

    def test_prepare_with_hop_containing_lifted_buffers(self):
        """GraphModule with flex_attention HOP + lifted buffers serializes cleanly.

        This is the exact pattern that crashes with regular pickle:
        flex_attention with non-empty mask_mod_other_buffers. GraphPickler
        handles it without retracing.
        """
        score_mod = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        for name in ["score", "b", "h", "m", "n"]:
            score_mod.graph.placeholder(name)
        score_mod.graph.output(next(iter(score_mod.graph.nodes)))
        score_mod.recompile()

        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        gm.score_mod_0 = score_mod

        g = gm.graph
        q = g.placeholder("q")
        k = g.placeholder("k")
        v = g.placeholder("v")
        mask_buf = g.placeholder("mask_buf")
        score_attr = g.get_attr("score_mod_0")

        block_mask = (
            128,
            128,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            128,
            128,
        )
        g.call_function(
            torch.ops.higher_order.flex_attention,
            args=(q, k, v, score_attr, block_mask, 0.125, {}, (), (mask_buf,)),
        )
        g.output(next(n for n in g.nodes if n.op == "call_function"))
        gm.recompile()

        cfg = self._make_cfg(gm)
        cfg.prepare_for_serialization()

        self.assertIsNone(cfg._original_gm)
        self.assertIsNotNone(cfg._serialized_original_gm)


if __name__ == "__main__":
    run_tests()
