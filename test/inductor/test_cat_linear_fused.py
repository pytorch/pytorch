# Owner(s): ["module: inductor"]
import operator

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.cat_linear_fused import (
    cat_linear_fused_pre_grad_pass,
    fuse_cat_linear_in_graph,
    MAX_PARTS,
    MAX_TOTAL_CAT_WIDTH,
    MIN_PIECE_WIDTH,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _shape_meta(node, shape):
    """Attach a FakeTensor-like meta['val'] so the matcher can read shapes."""
    node.meta["val"] = torch.empty(shape)
    return node


def _build_linear_cat_graph(num_parts, K_per_part, M=4, N=8, cat_dim=-1):
    g = fx.Graph()
    parts = []
    for i in range(num_parts):
        p = g.placeholder(f"t{i}")
        _shape_meta(p, (M, K_per_part))
        parts.append(p)
    w = g.placeholder("w")
    _shape_meta(w, (N, num_parts * K_per_part))
    b = g.placeholder("b")
    _shape_meta(b, (N,))
    cat = g.call_function(torch.cat, args=(parts, cat_dim))
    _shape_meta(cat, (M, num_parts * K_per_part))
    lin = g.call_function(F.linear, args=(cat, w, b))
    _shape_meta(lin, (M, N))
    g.output(lin)
    return g


class TestCatLinearFusedMatcher(TestCase):
    def test_canonical_pattern_fires(self):
        g = _build_linear_cat_graph(num_parts=2, K_per_part=16)
        n = fuse_cat_linear_in_graph(g)
        self.assertEqual(n, 1)

    def test_three_parts_fires(self):
        g = _build_linear_cat_graph(num_parts=3, K_per_part=16)
        n = fuse_cat_linear_in_graph(g)
        self.assertEqual(n, 1)

    def test_rejects_too_many_parts(self):
        g = _build_linear_cat_graph(num_parts=MAX_PARTS + 1, K_per_part=16)
        n = fuse_cat_linear_in_graph(g)
        self.assertEqual(n, 0)

    def test_rejects_piece_below_min_width(self):
        g = _build_linear_cat_graph(num_parts=2, K_per_part=MIN_PIECE_WIDTH - 1)
        n = fuse_cat_linear_in_graph(g)
        self.assertEqual(n, 0)

    def test_rejects_total_above_max_width(self):
        # Two parts, each just over MAX_TOTAL_CAT_WIDTH/2, so total > cap.
        K = MAX_TOTAL_CAT_WIDTH // 2 + 8
        g = _build_linear_cat_graph(num_parts=2, K_per_part=K)
        n = fuse_cat_linear_in_graph(g)
        self.assertEqual(n, 0)

    def test_rejects_mul_parented_part(self):
        g = fx.Graph()
        a = g.placeholder("a")
        _shape_meta(a, (4, 16))
        b = g.placeholder("b")
        _shape_meta(b, (4, 16))
        # One of the parts is the output of a `mul` - should be skipped by
        # the matcher.
        m = g.call_function(operator.mul, args=(a, b))
        _shape_meta(m, (4, 16))
        c = g.placeholder("c")
        _shape_meta(c, (4, 16))
        cat = g.call_function(torch.cat, args=([m, c], -1))
        _shape_meta(cat, (4, 32))
        w = g.placeholder("w")
        _shape_meta(w, (8, 32))
        lin = g.call_function(F.linear, args=(cat, w, None))
        _shape_meta(lin, (4, 8))
        g.output(lin)
        n = fuse_cat_linear_in_graph(g)
        self.assertEqual(n, 0)

    def test_rejects_non_lastdim_cat(self):
        g = _build_linear_cat_graph(num_parts=2, K_per_part=16, cat_dim=0)
        n = fuse_cat_linear_in_graph(g)
        self.assertEqual(n, 0)


class _CatLinearMod(nn.Module):
    """F.linear(torch.cat([proj_a(a), proj_b(b)], dim=-1), W, b) head."""

    def __init__(self, dim_a=64, dim_b=64, out=32):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, dim_a)
        self.proj_b = nn.Linear(dim_b, dim_b)
        self.head = nn.Linear(dim_a + dim_b, out)

    def forward(self, a, b):
        ha = F.relu(self.proj_a(a))
        hb = F.relu(self.proj_b(b))
        return self.head(torch.cat([ha, hb], dim=-1))


class TestCatLinearFusedIntegration(TestCase):
    def test_compile_fires_and_matches_reference(self):
        torch._inductor.config.pre_grad_custom_pass = cat_linear_fused_pre_grad_pass
        try:
            counters.clear()
            mod = _CatLinearMod().eval()
            a = torch.randn(8, 64)
            b = torch.randn(8, 64)
            ref = mod(a, b)
            traced = torch.compile(mod, mode="default", dynamic=False)
            out = traced(a, b)
            self.assertTrue(counters["inductor"]["cat_linear_fused"] >= 1)
            self.assertEqual(ref.shape, out.shape)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
        finally:
            torch._inductor.config.pre_grad_custom_pass = None

    def test_disabled_by_default(self):
        counters.clear()
        mod = _CatLinearMod().eval()
        a = torch.randn(8, 64)
        b = torch.randn(8, 64)
        traced = torch.compile(mod, mode="default", dynamic=False)
        traced(a, b)
        self.assertEqual(counters["inductor"]["cat_linear_fused"], 0)


if __name__ == "__main__":
    run_tests()
