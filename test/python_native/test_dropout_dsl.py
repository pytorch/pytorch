# Owner(s): ["module: dsl-native-ops"]
#
# Correctness tests for the example _native fused dropout overrides
# (Triton and CuTeDSL). Both kernels replicate aten's
# fused_dropout_kernel_vec (VEC=4) exactly, drawing RNG state through
# Generator.philox_state, so output, mask, and generator offset
# advancement must all be bit-identical to stock aten. References are
# computed with the DSL override disabled, per the stock-aten rule.

import unittest
from contextlib import contextmanager

import torch
import torch.backends.python_native as pn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA_GRAPH,
    TestCase,
)


_SHAPES = [
    (16,),
    (1024,),
    (262144,),  # below the grid cap
    (1 << 22,),  # grid-capped: multiple iterations per thread
    (333, 4),
    (4096, 1024),
]


def _dsl_registered(dsl: str) -> bool:
    from torch._native.registry import _graphs

    key = ("native_dropout", "CUDA")
    return any(node.dsl_name == dsl for node in _graphs.get(key, []))


_OTHER = {"triton": "cutedsl", "cutedsl": "triton"}


@contextmanager
def _only(dsl: str):
    """Route native_dropout to exactly this DSL's override."""
    with getattr(pn, _OTHER[dsl]).disabled():
        yield


def _run_dropout_pair(dsl: str, shape, p: float, seed: int = 1234):
    """Return ((aten out, mask, offset), (native out, mask, offset))."""
    g = torch.cuda.default_generators[0]
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    with getattr(pn, dsl).disabled(), getattr(pn, _OTHER[dsl]).disabled():
        g.manual_seed(seed)
        a_out, a_mask = torch.ops.aten.native_dropout(x, p, True)
        a_off = g.get_offset()
    with _only(dsl):
        g.manual_seed(seed)
        n_out, n_mask = torch.ops.aten.native_dropout(x, p, True)
        n_off = g.get_offset()
    return (a_out, a_mask, a_off), (n_out, n_mask, n_off)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestDropoutDSL(TestCase):
    def _require(self, dsl):
        if not _dsl_registered(dsl):
            self.skipTest(f"native_dropout not registered for {dsl}")

    @parametrize("dsl", ["triton", "cutedsl"])
    @parametrize("shape", _SHAPES, name_fn=lambda s: "x".join(map(str, s)))
    @parametrize("p", [0.1, 0.5, 0.77])
    def test_bit_exact_vs_aten(self, dsl, shape, p):
        self._require(dsl)
        (a_out, a_mask, a_off), (n_out, n_mask, n_off) = _run_dropout_pair(
            dsl, shape, p
        )
        self.assertEqual(n_out, a_out, atol=0, rtol=0)
        self.assertEqual(n_mask, a_mask, atol=0, rtol=0)
        self.assertEqual(n_off, a_off)

    @parametrize("dsl", ["triton", "cutedsl"])
    def test_stream_composition(self, dsl):
        # A native dropout must leave the generator exactly where aten
        # would, so a following eager RNG op produces identical values.
        self._require(dsl)
        g = torch.cuda.default_generators[0]
        x = torch.randn(1 << 20, device="cuda")
        with getattr(pn, dsl).disabled(), getattr(pn, _OTHER[dsl]).disabled():
            g.manual_seed(7)
            torch.ops.aten.native_dropout(x, 0.3, True)
            ref_follow = torch.rand(1000, device="cuda")
        with _only(dsl):
            g.manual_seed(7)
            torch.ops.aten.native_dropout(x, 0.3, True)
            got_follow = torch.rand(1000, device="cuda")
        self.assertEqual(got_follow, ref_follow, atol=0, rtol=0)

    def test_ineligible_falls_back_to_aten(self):
        # fp64 is outside the override's gate; must still work via aten.
        x64 = torch.randn(1024, device="cuda", dtype=torch.float64)
        out, mask = torch.ops.aten.native_dropout(x64, 0.5, True)
        self.assertEqual(out.dtype, torch.float64)
        self.assertEqual(mask.dtype, torch.bool)

    @unittest.skipIf(not TEST_CUDA_GRAPH, "CUDA graphs required")
    @parametrize("dsl", ["triton", "cutedsl"])
    def test_graph_replay_matches_eager_aten(self, dsl):
        # One captured native dropout replayed N times must reproduce N
        # consecutive eager aten dropouts, including offset advancement.
        self._require(dsl)
        g = torch.cuda.default_generators[0]
        x = torch.randn(1 << 20, device="cuda")
        with getattr(pn, dsl).disabled(), getattr(pn, _OTHER[dsl]).disabled():
            g.manual_seed(42)
            refs = [torch.ops.aten.native_dropout(x, 0.5, True) for _ in range(3)]
            ref_off = g.get_offset()

        with _only(dsl):
            torch.ops.aten.native_dropout(x, 0.5, True)  # warm the DSL compile
            g.manual_seed(42)
            graph = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                graph.capture_begin()
                out, mask = torch.ops.aten.native_dropout(x, 0.5, True)
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(s)

        for ref_out, ref_mask in refs:
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(out, ref_out, atol=0, rtol=0)
            self.assertEqual(mask, ref_mask, atol=0, rtol=0)
        self.assertEqual(g.get_offset(), ref_off)


instantiate_parametrized_tests(TestDropoutDSL)


if __name__ == "__main__":
    run_tests()
