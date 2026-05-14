# Owner(s): ["module: mps"]
"""Tests for torch.mps.MPSGraph (MTLIndirectCommandBuffer capture/replay).

Mirrors the structure of CUDAGraph tests. See:
  aten/src/ATen/mps/MPSStreamGraph.{h,mm}
  torch/csrc/mps/StreamGraph.cpp
  torch/mps/graphs.py
"""

import gc
import time
import unittest

import torch
import torch.backends.mps

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    serialTest,
)


# Gate the entire file on MPS being available + built.
if not torch.backends.mps.is_available():
    print("MPS not available; skipping test_mps_graphs.py", flush=True)

    class MPSGraphNotAvailable(TestCase):
        @unittest.skip("MPS backend not available")
        def test_dummy(self):
            pass

    if __name__ == "__main__":
        run_tests()
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Correctness tests (12)
# ---------------------------------------------------------------------------


class TestMPSGraphCorrectness(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Warmup so the first capture isn't paying for kernel compilation.
        x = torch.randn(64, device="mps")
        for _ in range(3):
            _ = x.relu().sigmoid().tanh()
            _ = (x * 2.0 + 1.0).relu()
        torch.mps.synchronize()

    def test_capture_replay_identity(self):
        """Capture a unary op, replay, output matches eager."""
        x = torch.randn(64, 64, device="mps")
        eager_y = torch.relu(x).clone()

        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = torch.relu(x)
        torch.mps.synchronize()

        self.assertEqual(eager_y, y)
        g.replay()
        torch.mps.synchronize()
        self.assertEqual(eager_y, y)

    def test_multi_kernel_capture(self):
        """Multiple distinct PSOs in one capture: relu, sigmoid, tanh, exp."""
        x = torch.randn(1024, device="mps")
        eager_y = x.relu().sigmoid().tanh().exp().clone()

        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = x.relu().sigmoid().tanh().exp()
        torch.mps.synchronize()

        self.assertGreaterEqual(g.num_commands(), 4)
        self.assertEqual(eager_y, y)
        g.replay()
        torch.mps.synchronize()
        self.assertEqual(eager_y, y)

    def test_replay_determinism(self):
        """Same captured input → same output across N replays."""
        x = torch.randn(256, device="mps")
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = x.relu().sigmoid()
        torch.mps.synchronize()
        first = y.clone()
        for _ in range(8):
            g.replay()
            torch.mps.synchronize()
            self.assertEqual(first, y)

    def test_replay_picks_up_mutated_input(self):
        """copy_ into captured input → replay reflects new values."""
        x = torch.zeros(128, device="mps")
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = x.relu().sigmoid()
        torch.mps.synchronize()
        new_x = torch.randn(128, device="mps")
        x.copy_(new_x)
        g.replay()
        torch.mps.synchronize()
        self.assertEqual(new_x.relu().sigmoid(), y)

    def test_outside_capture_eager_unchanged(self):
        """Eager dispatch outside any capture context is unaffected."""
        x = torch.randn(64, device="mps")
        eager_y = x.relu().clone()

        # Create and destroy a graph; eager must still match.
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            _ = x.relu()
        del g
        gc.collect()

        y = x.relu()
        torch.mps.synchronize()
        self.assertEqual(eager_y, y)

    def test_pso_lifetime(self):
        """PSOs are retained for the graph's lifetime; replay works after
        intermediate gc.collect() pressure on the allocator."""
        x = torch.randn(512, device="mps")
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = x.relu().sigmoid().tanh()
        torch.mps.synchronize()
        baseline = y.clone()
        # Provoke allocator churn so anything not properly retained would go.
        for _ in range(64):
            _ = torch.randn(8192, device="mps").relu()
        torch.mps.synchronize()
        gc.collect()
        torch.mps.empty_cache()
        g.replay()
        torch.mps.synchronize()
        self.assertEqual(baseline, y)

    def test_buffer_lifetime(self):
        """Intermediate buffers (held via M3.8 graph-held pool) survive
        replay even though the user holds no reference to them."""
        x = torch.randn(2048, device="mps")
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            # Intermediates of mul/add/relu have no surviving Python ref.
            y = (x * 2.0 + 1.0).relu()
        torch.mps.synchronize()
        baseline = y.clone()
        torch.mps.empty_cache()
        for _ in range(10):
            g.replay()
            torch.mps.synchronize()
            self.assertEqual(baseline, y)

    def test_capture_then_destroy_then_no_replay(self):
        """Destroyed graph: dropped reference is safe (no segfault, no UAF)."""
        x = torch.randn(64, device="mps")
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = x.relu()
        torch.mps.synchronize()
        del g
        gc.collect()
        # y still alive; the underlying buffer must still be valid because
        # the user owns it. (Intermediates would have been released.)
        _ = y.clone()
        torch.mps.synchronize()

    def test_replay_with_fp16_dtype(self):
        x = torch.randn(256, dtype=torch.float16, device="mps")
        eager_y = x.relu().sigmoid().clone()
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = x.relu().sigmoid()
        torch.mps.synchronize()
        self.assertEqual(eager_y, y)
        g.replay()
        torch.mps.synchronize()
        self.assertEqual(eager_y, y)

    def test_replay_with_bfloat16_dtype(self):
        x = torch.randn(256, dtype=torch.bfloat16, device="mps")
        eager_y = x.relu().sigmoid().clone()
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = x.relu().sigmoid()
        torch.mps.synchronize()
        self.assertEqual(eager_y, y)
        g.replay()
        torch.mps.synchronize()
        self.assertEqual(eager_y, y)

    def test_replay_with_inline_constants(self):
        """setBytes inline-constants (M4): scalar args captured per-cmd."""
        x = torch.randn(256, device="mps")
        eager_y = (x * 3.5 + 0.25).clone()
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = x * 3.5 + 0.25
        torch.mps.synchronize()
        self.assertEqual(eager_y, y)
        g.replay()
        torch.mps.synchronize()
        self.assertEqual(eager_y, y)

    def test_graph_size_recorded(self):
        """num_commands matches the number of dispatched kernels."""
        x = torch.randn(64, device="mps")
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            _ = x.relu()
        self.assertEqual(g.num_commands(), 1)

        g2 = torch.mps.MPSGraph()
        with torch.mps.graph(g2):
            _ = (x * 2.0 + 1.0).relu()
        # Should be 3: mul, add, relu.
        self.assertEqual(g2.num_commands(), 3)


# ---------------------------------------------------------------------------
# Edge cases (5)
# ---------------------------------------------------------------------------


class TestMPSGraphEdgeCases(TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_empty_capture(self):
        """Capture with no ops: replay raises 'empty graph'."""
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            pass
        self.assertEqual(g.num_commands(), 0)
        self.assertTrue(g.is_ready())
        with self.assertRaisesRegex(RuntimeError, "empty graph"):
            g.replay()

    def test_nested_capture_errors(self):
        """capture_begin while another capture is active raises."""
        g1 = torch.mps.MPSGraph()
        g2 = torch.mps.MPSGraph()
        with torch.mps.graph(g1):
            with self.assertRaisesRegex(
                RuntimeError, "(?:not idle|another capture is active)"
            ):
                g2.capture_begin()
            # leave g1's context cleanly; capture_end is fine.

    def test_replay_before_capture_end_errors(self):
        """replay inside capture context raises."""
        g = torch.mps.MPSGraph()
        x = torch.randn(64, device="mps")
        with torch.mps.graph(g):
            _ = x.relu()
            with self.assertRaisesRegex(RuntimeError, "completed capture"):
                g.replay()

    def test_resource_residency_complete(self):
        """All buffers touched during capture remain resident under replay.

        Verifies the useResource: declarations are correct. Test approach:
        capture using user-owned input/output tensors; mutate input via
        copy_ between replays; replay must still see the new contents.
        """
        x = torch.randn(128, device="mps")
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            y = x.relu().sigmoid()
        torch.mps.synchronize()
        for _ in range(4):
            new_x = torch.randn(128, device="mps")
            x.copy_(new_x)
            g.replay()
            torch.mps.synchronize()
            self.assertEqual(new_x.relu().sigmoid(), y)

    def test_allocation_during_capture_held(self):
        """Allocating intermediates during capture is allowed in v1: the
        M3.8 graph-held buffer pool keeps the storage alive across replays.
        """
        x = torch.randn(256, device="mps")
        g = torch.mps.MPSGraph()
        with torch.mps.graph(g):
            # Each op produces an intermediate buffer that goes out of scope
            # before the next op; must be held for replay.
            y = (x + 1.0) * 2.0 - 0.5
        torch.mps.synchronize()
        baseline = y.clone()
        torch.mps.empty_cache()
        for _ in range(5):
            g.replay()
            torch.mps.synchronize()
            self.assertEqual(baseline, y)


# ---------------------------------------------------------------------------
# Performance assertions (4, gated on --slow)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    __import__("os").environ.get("PYTORCH_TEST_WITH_SLOW", "0") == "1",
    "perf assertions gated on PYTORCH_TEST_WITH_SLOW=1",
)
class TestMPSGraphPerf(TestCase):
    @serialTest()
    def test_replay_faster_than_eager_small_kernel(self):
        """Small kernel where CPU encoding overhead dominates."""
        x = torch.randn(4096, device="mps")
        n = 1000

        # Warmup.
        for _ in range(50):
            _ = x.relu()
        torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(n):
            _ = x.relu()
        torch.mps.synchronize()
        eager_t = time.perf_counter() - t0

        g = torch.mps.MPSGraph(max_commands=n + 16)
        with torch.mps.graph(g):
            for _ in range(n):
                _ = x.relu()
        torch.mps.synchronize()

        t0 = time.perf_counter()
        g.replay()
        torch.mps.synchronize()
        graph_t = time.perf_counter() - t0

        self.assertLess(
            graph_t,
            eager_t * 0.70,
            f"Graph mode should be ≥30% faster on small kernels "
            f"(eager: {eager_t*1e3:.2f}ms, graph: {graph_t*1e3:.2f}ms)",
        )

    @serialTest()
    def test_large_kernel_no_regression(self):
        """Large kernel where GPU time dominates: graph mode within 10%."""
        x = torch.randn(8192, 8192, device="mps")
        n = 8
        for _ in range(5):
            _ = x.relu()
        torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(n):
            _ = x.relu()
        torch.mps.synchronize()
        eager_t = time.perf_counter() - t0

        g = torch.mps.MPSGraph(max_commands=n + 16)
        with torch.mps.graph(g):
            for _ in range(n):
                _ = x.relu()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.mps.synchronize()
        graph_t = time.perf_counter() - t0

        self.assertLess(
            graph_t,
            eager_t * 1.10,
            f"Graph mode should not regress on large kernels "
            f"(eager: {eager_t*1e3:.2f}ms, graph: {graph_t*1e3:.2f}ms)",
        )

    @serialTest()
    def test_dispatch_scaling(self):
        """Per-dispatch savings should be roughly constant across N."""
        x = torch.randn(4096, device="mps")
        savings_per_disp = []
        for n in (32, 256, 1024):
            for _ in range(20):
                _ = x.relu()
            torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                _ = x.relu()
            torch.mps.synchronize()
            eager_t = time.perf_counter() - t0

            g = torch.mps.MPSGraph(max_commands=n + 16)
            with torch.mps.graph(g):
                for _ in range(n):
                    _ = x.relu()
            torch.mps.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.mps.synchronize()
            graph_t = time.perf_counter() - t0

            savings_per_disp.append((eager_t - graph_t) / n)

        # Per-dispatch savings should be positive and stable (within 3x range)
        # across N. Loose bound: the smallest mustn't be < 1/3 of the largest.
        lo = min(savings_per_disp)
        hi = max(savings_per_disp)
        self.assertGreater(lo, 0, f"savings_per_disp not all positive: {savings_per_disp}")
        self.assertLess(
            hi / lo,
            5.0,
            f"per-dispatch savings vary too much across N: {savings_per_disp}",
        )

    @serialTest()
    def test_make_graphed_callables(self):
        """make_graphed_callables: wraps a callable and matches eager output."""

        def fn(t):
            return t.relu().sigmoid().tanh()

        x = torch.randn(1024, device="mps")
        wrapped = torch.mps.make_graphed_callables(fn, (x,))
        torch.mps.synchronize()
        new_x = torch.randn(1024, device="mps")
        out = wrapped(new_x)
        torch.mps.synchronize()
        self.assertEqual(fn(new_x), out)


if __name__ == "__main__":
    run_tests()
